"""
DH-PPL: Dynamic Horizon & Latent Barrier PPL
=============================================
Drop-in replacement for ppl.py.  Works with any environment that follows
the PPL shared-control interface (MetaDrive, Robosuite, …).

Novelties over base PPL
-----------------------
1. EnsembleDynamicsModel
   Five-head MLP ensemble trained online on (s, a) → Δs transitions.
   Inter-head variance is used as an epistemic uncertainty estimate U(s, a).

2. Dynamic Horizon gate  (L_dyn)
   Before a preference pair is stored, the environment calls
   ``should_add_to_preference_buffer(obs, action)`` which:
     • Computes U(s, a) via the ensemble.
     • Maintains a rolling window of the last 200 uncertainty values.
     • Sets the adaptive threshold L = mean(window).
     • Admits the pair iff U < L (below-average uncertainty → reliable signal).
   Returns (accepted: bool, u: float) in one call so the caller never has
   to query the model twice per step.

3. Latent Barrier Loss  (LBO)
   An additional actor penalty  λ · E[U(s, π(s))]  that discourages the
   policy from selecting actions the dynamics model finds unpredictable.
   λ = 0.1 by default.

Bug fixes over base PPL
-----------------------
• _n_updates double-increment: incremented only inside the gradient loop.
• Preference buffer under-fill guard: DPO loss is skipped when
  preference_buffer.pos < batch_size to avoid index errors.
• Per-train-call accept/reject counters reset at the start of each train()
  so logged accept_rate reflects the current training window only.
• biased_bce_with_logits replaced by F.softplus(adv_neg - adv_pos) which
  is numerically cleaner and avoids the manual log-sum-exp bookkeeping.
"""

import copy
import io
import logging
import os
import pathlib
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
import torch
from torch.nn import functional as F

from ppl.sb3.common.buffers import ReplayBuffer
from ppl.sb3.common.save_util import load_from_pkl, save_to_pkl
from ppl.sb3.common.type_aliases import GymEnv, MaybeCallback
from ppl.sb3.common.utils import polyak_update
from ppl.sb3.haco.haco_buffer import HACOReplayBuffer, concat_samples, PrefReplayBuffer
from ppl.sb3.td3.td3 import TD3


# ---------------------------------------------------------------------------
# Novelty 1: Ensemble Dynamics Model
# ---------------------------------------------------------------------------

class EnsembleDynamicsModel(torch.nn.Module):
    """
    An ensemble of ``num_heads`` independent MLP heads, each mapping
    (obs, action) → predicted next-obs delta.

    Epistemic uncertainty is estimated as the summed variance of the
    heads' predictions across the output dimensions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_heads: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(obs_dim + action_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, obs_dim),
                )
                for _ in range(num_heads)
            ]
        )

    def predict(
        self, obs: torch.Tensor, action: torch.Tensor
    ):
        """
        Returns
        -------
        mean_pred : Tensor  (batch, obs_dim)  – ensemble mean prediction
        variance  : Tensor  (batch,)          – summed variance (uncertainty)
        """
        x = torch.cat([obs, action], dim=-1)
        preds = torch.stack([head(x) for head in self.heads])   # (H, B, obs_dim)
        mean_pred = preds.mean(dim=0)
        variance  = preds.var(dim=0).sum(dim=-1)                # (B,)
        return mean_pred, variance

    def predict_uncertainty(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Variance only (no mean), (batch,)."""
        _, variance = self.predict(obs, action)
        return variance

    def predict_uncertainty_numpy(
        self,
        obs_np: np.ndarray,
        action_np: np.ndarray,
        device: torch.device,
    ) -> float:
        """
        Convenience wrapper accepting numpy arrays.
        Returns a scalar Python float.
        Called by the environment at data-collection time.
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        act_t = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            u = self.predict_uncertainty(obs_t, act_t)
        return float(u.item())


# ---------------------------------------------------------------------------
# PVPTD3  (unchanged from base PPL except .get() guards on extra_config)
# ---------------------------------------------------------------------------

class PVPTD3(TD3):
    """
    Physical-Virtual Pairing TD3 with dual replay buffers and optional
    proxy value losses.  This is the shared base for both PPL and DH-PPL.
    """

    def __init__(self, use_balance_sample: bool = True, q_value_bound: float = 1.0,
                 *args, **kwargs):
        if "cql_coefficient" in kwargs:
            self.cql_coefficient = kwargs.pop("cql_coefficient")
        else:
            self.cql_coefficient = 1

        if "replay_buffer_class" not in kwargs:
            kwargs["replay_buffer_class"] = HACOReplayBuffer

        self.extra_config: Dict[str, Any] = {}
        for k in [
            "no_done_for_positive", "no_done_for_negative",
            "reward_0_for_positive", "reward_0_for_negative",
            "reward_n2_for_intervention", "reward_1_for_all",
            "use_weighted_reward", "remove_negative",
            "adaptive_batch_size", "add_bc_loss", "only_bc_loss",
            "with_human_proxy_value_loss", "with_agent_proxy_value_loss",
            "simple_batch",
        ]:
            if k in kwargs:
                v = kwargs.pop(k)
                assert v in ["True", "False"], f"Expected 'True'/'False' for {k}, got {v!r}"
                self.extra_config[k] = v == "True"

        for k in ["agent_data_ratio", "bc_loss_weight", "beta"]:
            if k in kwargs:
                self.extra_config[k] = kwargs.pop(k)

        self.q_value_bound    = q_value_bound
        self.use_balance_sample = use_balance_sample
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.use_balance_sample:
            self.human_data_buffer = HACOReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )
        else:
            self.human_data_buffer = self.replay_buffer

    # ------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        with_hpvl = self.extra_config.get("with_human_proxy_value_loss", False)
        with_apvl = self.extra_config.get("with_agent_proxy_value_loss", False)
        stat_recorder: Dict[str, list] = defaultdict(list)

        should_concat   = False
        replay_data     = None
        replay_data_human = None

        if self.replay_buffer.pos > 0 and self.human_data_buffer.pos > 0:
            replay_data_human = self.human_data_buffer.sample(
                int(batch_size), env=self._vec_normalize_env, return_all=True
            )
            human_data_size = len(replay_data_human.observations)
            human_data_size = max(
                1,
                int(self.extra_config.get("agent_data_ratio", 1.0) * human_data_size),
            )
            should_concat = True
        elif self.human_data_buffer.pos > 0:
            replay_data = self.human_data_buffer.sample(
                batch_size, env=self._vec_normalize_env, return_all=True
            )
        elif self.replay_buffer.pos > 0:
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        else:
            gradient_steps = 0

        for _step in range(gradient_steps):
            self._n_updates += 1

            # --- batch assembly ---
            if self.extra_config.get("simple_batch", False):
                if self.replay_buffer.pos == 0:
                    replay_data = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                elif self.human_data_buffer.pos == 0:
                    replay_data = self.replay_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                else:
                    a = self.replay_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                    b = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                    replay_data = concat_samples(a, b)

            elif self.extra_config.get("adaptive_batch_size", False):
                if should_concat:
                    replay_data_agent = self.replay_buffer.sample(human_data_size, env=self._vec_normalize_env)
                    replay_data = concat_samples(replay_data_agent, replay_data_human)
            else:
                if self.replay_buffer.pos > batch_size and self.human_data_buffer.pos > batch_size:
                    a = self.replay_buffer.sample(int(batch_size / 2), env=self._vec_normalize_env)
                    b = self.human_data_buffer.sample(int(batch_size / 2), env=self._vec_normalize_env)
                    replay_data = concat_samples(a, b)
                elif self.human_data_buffer.pos > batch_size:
                    replay_data = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)
                elif self.replay_buffer.pos > batch_size:
                    replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                else:
                    break

            # --- critic update ---
            with th.no_grad():
                noise = replay_data.actions_behavior.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                nqv = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                nqv, _ = th.min(nqv, dim=1, keepdim=True)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * nqv

            cbv = self.critic(replay_data.observations, replay_data.actions_behavior)
            cnv = self.critic(replay_data.observations, replay_data.actions_novice)
            stat_recorder["q_value_behavior"].append(cbv[0].mean().item())
            stat_recorder["q_value_novice"].append(cnv[0].mean().item())

            critic_loss_list = []
            for cb, cn in zip(cbv, cnv):
                l = F.mse_loss(cb, target_q)
                if with_hpvl:
                    l += th.mean(replay_data.interventions * self.cql_coefficient *
                                 F.mse_loss(cb, self.q_value_bound * th.ones_like(cb), reduction="none"))
                if with_apvl:
                    l += th.mean(replay_data.interventions * self.cql_coefficient *
                                 F.mse_loss(cn, -self.q_value_bound * th.ones_like(cb), reduction="none"))
                critic_loss_list.append(l)
            critic_loss = sum(critic_loss_list)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            stat_recorder["critic_loss"].append(critic_loss.item())

            # --- actor update ---
            if self._n_updates % self.policy_delay == 0:
                new_action = self.actor(replay_data.observations)
                bc_loss = F.mse_loss(replay_data.actions_behavior, new_action, reduction="none").mean(axis=-1)
                masked_bc_loss = (replay_data.interventions.flatten() * bc_loss).sum() / \
                                 (replay_data.interventions.flatten().sum() + 1e-5)

                if self.extra_config.get("only_bc_loss", False):
                    actor_loss = masked_bc_loss
                else:
                    actor_loss = -self.critic.q1_forward(replay_data.observations, new_action).mean()
                    if self.extra_config.get("add_bc_loss", False):
                        actor_loss += masked_bc_loss * self.extra_config.get("bc_loss_weight", 1.0)

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                stat_recorder["actor_loss"].append(actor_loss.item())
                stat_recorder["masked_bc_loss"].append(masked_bc_loss.item())
                stat_recorder["bc_loss"].append(bc_loss.mean().item())
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in stat_recorder.items():
            self.logger.record(f"train/{key}", np.mean(values))

    # ------------------------------------------------------------------

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        if infos[0].get("takeover") or infos[0].get("takeover_start"):
            replay_buffer = self.human_data_buffer
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def save_replay_buffer(self, path_human, path_replay) -> None:
        save_to_pkl(path_human, self.human_data_buffer, self.verbose)
        super().save_replay_buffer(path_replay)

    def load_replay_buffer(self, path_human, path_replay, truncate_last_traj: bool = True) -> None:
        self.human_data_buffer = load_from_pkl(path_human, self.verbose)
        assert isinstance(self.human_data_buffer, ReplayBuffer)
        if not hasattr(self.human_data_buffer, "handle_timeout_termination"):
            self.human_data_buffer.handle_timeout_termination = False
            self.human_data_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)
        super().load_replay_buffer(path_replay, truncate_last_traj)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        save_timesteps: int = 2000,
        buffer_save_timesteps: int = 2000,
        save_path_human: str = "",
        save_path_replay: str = "",
        save_buffer: bool = True,
        load_buffer: bool = False,
        load_path_human: str = "",
        load_path_replay: str = "",
        warmup: bool = False,
        warmup_steps: int = 5000,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes,
            eval_log_path, reset_num_timesteps, tb_log_name,
        )
        if load_buffer:
            self.load_replay_buffer(load_path_human, load_path_replay)
        callback.on_training_start(locals(), globals())
        if warmup:
            assert load_buffer, "warmup is only useful when loading a buffer"
            print(f"Start warmup: {warmup_steps} steps")
            self.train(batch_size=self.batch_size, gradient_steps=warmup_steps)

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
            if not rollout.continue_training:
                break
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                gradient_steps = (
                    self.gradient_steps if self.gradient_steps >= 0
                    else rollout.episode_timesteps
                )
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
            if save_buffer and self.num_timesteps > 0 and \
                    self.num_timesteps % buffer_save_timesteps == 0:
                self.save_replay_buffer(
                    os.path.join(save_path_human, f"human_buffer_{self.num_timesteps}.pkl"),
                    os.path.join(save_path_replay, f"replay_buffer_{self.num_timesteps}.pkl"),
                )
        callback.on_training_end()
        return self


# ---------------------------------------------------------------------------
# DH-PPL
# ---------------------------------------------------------------------------

class PPL(PVPTD3):
    """
    DH-PPL: Dynamic Horizon Proximal Policy Learning.

    Extends PVPTD3 with:
      • A ``PrefReplayBuffer`` for contrastive preference data.
      • An ``EnsembleDynamicsModel`` trained online on human-buffer transitions.
      • An adaptive uncertainty gate (``should_add_to_preference_buffer``).
      • A Latent Barrier Loss term in the actor update.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Preference replay buffer
        self.preference_buffer = PrefReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            **self.replay_buffer_kwargs,
        )

        # Ensemble dynamics model
        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        self.uncertainty_model = EnsembleDynamicsModel(obs_dim, act_dim).to(self.device)
        self.uncertainty_optimizer = torch.optim.Adam(
            self.uncertainty_model.parameters(), lr=1e-3
        )

        # Rolling uncertainty window for the adaptive threshold L
        self._uncertainty_window: deque = deque(maxlen=200)
        self._uncertainty_threshold: float = float("inf")

        # Lifetime accept / reject counters (for reference)
        self._pref_accepted_total: int = 0
        self._pref_rejected_total: int = 0
        # Per-train-call counters (reset at the start of each train())
        self._pref_accepted: int = 0
        self._pref_rejected: int = 0

    # ------------------------------------------------------------------

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["preference_buffer", "human_data_buffer"]

    # ------------------------------------------------------------------
    # Novelty 2: Dynamic Horizon gate
    # ------------------------------------------------------------------

    def should_add_to_preference_buffer(
        self,
        obs_np: np.ndarray,
        action_np: np.ndarray,
    ):
        """
        Gate function called by the environment before storing a preference pair.

        Computes U(obs, action) via the ensemble, updates the rolling window,
        and returns (accepted, u) in a single call so the environment never
        queries the model twice for the same step.

        Parameters
        ----------
        obs_np    : (obs_dim,) numpy array
        action_np : (act_dim,) numpy array

        Returns
        -------
        accepted : bool   – whether the pair should be admitted
        u        : float  – raw uncertainty value
        """
        u = self.uncertainty_model.predict_uncertainty_numpy(obs_np, action_np, self.device)
        self._uncertainty_window.append(u)

        # Need at least 10 samples before the threshold is meaningful
        if len(self._uncertainty_window) < 10:
            return True, u

        self._uncertainty_threshold = float(np.mean(self._uncertainty_window))
        accepted = u < self._uncertainty_threshold

        if accepted:
            self._pref_accepted       += 1
            self._pref_accepted_total += 1
        else:
            self._pref_rejected       += 1
            self._pref_rejected_total += 1

        return accepted, u

    # ------------------------------------------------------------------
    # DH-PPL training loop
    # ------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer])

        stat_recorder: Dict[str, list] = defaultdict(list)

        # Reset per-train-call counters so logged accept_rate reflects
        # this training window only, not a stale lifetime average.
        self._pref_accepted = 0
        self._pref_rejected = 0

        for _step in range(gradient_steps):
            # Increment _n_updates exactly once per gradient step.
            # (The original ppl.py also incremented after the loop — that was a bug.)
            self._n_updates += 1

            if self.human_data_buffer.pos == 0:
                break

            replay_data = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)

            # ----------------------------------------------------------
            # Train ensemble dynamics model: (s, a) → Δs
            # ----------------------------------------------------------
            delta_target = replay_data.next_observations - replay_data.observations
            x_input = torch.cat(
                [replay_data.observations, replay_data.actions_behavior], dim=-1
            )
            uncertainty_loss = torch.tensor(0.0, device=self.device)
            for head in self.uncertainty_model.heads:
                pred_delta = head(x_input)
                uncertainty_loss = uncertainty_loss + F.mse_loss(pred_delta, delta_target)

            self.uncertainty_optimizer.zero_grad()
            uncertainty_loss.backward()
            self.uncertainty_optimizer.step()
            stat_recorder["uncertainty_loss"].append(uncertainty_loss.item())

            # ----------------------------------------------------------
            # Behaviour-cloning loss
            # ----------------------------------------------------------
            new_action = self.actor(replay_data.observations)
            bc_loss = F.mse_loss(
                replay_data.actions_behavior, new_action, reduction="none"
            ).mean()

            stat_recorder["new_action_steering"].append(new_action[:, 0].mean().item())
            stat_recorder["new_action_abs_steering"].append(th.abs(new_action[:, 0]).mean().item())
            if new_action.shape[1] > 1:
                stat_recorder["new_action_accel"].append(new_action[:, 1].mean().item())

            # ----------------------------------------------------------
            # DPO preference loss
            # Guard: only sample when the buffer has enough entries.
            # ----------------------------------------------------------
            dpo_loss = torch.tensor(0.0, device=self.device)
            if self.preference_buffer.pos >= batch_size:
                pref_data = self.preference_buffer.sample(int(batch_size), env=self._vec_normalize_env)

                pos_obs    = pref_data.pos_observations.squeeze()
                pos_action = pref_data.pos_actions.squeeze()
                neg_obs    = pref_data.neg_observations.squeeze()
                neg_action = pref_data.neg_actions.squeeze()

                def _log_prob(obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                    mean = self.actor(obs)
                    return -((mean - target) ** 2).sum(dim=-1)

                beta = self.extra_config.get("beta", 0.1)
                adv_pos = beta * _log_prob(pos_obs, pos_action)
                adv_neg = beta * _log_prob(neg_obs, neg_action)

                # Numerically stable DPO: softplus( adv_neg - adv_pos )
                dpo_loss = F.softplus(adv_neg - adv_pos).mean()

            stat_recorder["dpo_loss"].append(dpo_loss.item())

            # ----------------------------------------------------------
            # Novelty 3: Latent Barrier Loss
            # Penalise the actor for choosing actions in high-uncertainty regions.
            # ----------------------------------------------------------
            _, pred_var = self.uncertainty_model.predict(
                replay_data.observations, new_action
            )
            latent_barrier_loss = pred_var.mean() * 0.1
            stat_recorder["latent_barrier_loss"].append(latent_barrier_loss.item())

            # ----------------------------------------------------------
            # Total loss
            # ----------------------------------------------------------
            bc_loss_weight = self.extra_config.get("bc_loss_weight", 1.0)
            if self.extra_config.get("only_bc_loss", False):
                loss = bc_loss
            else:
                loss = bc_loss_weight * bc_loss + dpo_loss + latent_barrier_loss

            self.actor.optimizer.zero_grad()
            loss.backward()
            self.actor.optimizer.step()

            stat_recorder["bc_loss"].append(bc_loss.item())
            stat_recorder["total_loss"].append(loss.item())

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        total_attempts = self._pref_accepted + self._pref_rejected
        accept_rate = self._pref_accepted / max(total_attempts, 1)

        self.logger.record("train/pref_buffer_accept_rate",  accept_rate)
        self.logger.record("train/pref_buffer_admitted",     self._pref_accepted)
        self.logger.record("train/pref_buffer_rejected",     self._pref_rejected)
        self.logger.record("train/uncertainty_threshold_L",  self._uncertainty_threshold)
        self.logger.record("train/predicted_steps",          self.preference_buffer.pos)
        self.logger.record("train/human_involved_steps",     self.human_data_buffer.pos)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in stat_recorder.items():
            self.logger.record(f"train/{key}", np.mean(values))
