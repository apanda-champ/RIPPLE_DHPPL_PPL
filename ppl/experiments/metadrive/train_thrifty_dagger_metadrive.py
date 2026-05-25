"""
Thrifty DAgger training script for MetaDrive  —  FIXED VERSION
===============================================================

Bugs fixed vs. the original:
─────────────────────────────
BUG 1 (critical – zero expert steps):
  ThriftyTakeoverEnv did NOT override _get_step_return.
  MetaDrive's base step() calls _get_step_return internally, which
  resolved to DrivingEnv._get_step_return.  That method queries the
  engine's shared-control policy:

      self.takeover = shared_control_policy.takeover   # always False
                                                        # (keyboard never pressed)

  The Thrifty DAgger decision computed three lines earlier was therefore
  silently discarded on every step, so human_data_buffer received nothing
  and the critic never got any expert signal.

  FIX: Override _get_step_return (mirroring ExpertTakeoverEnv) to keep
  self.takeover exactly as the Thrifty DAgger criterion set it, rather
  than reading it back from the engine policy.

BUG 2 (critical – corrupt raw_action / immediate crash):
  ThriftyTakeoverEnv inherited agent_policy=TakeoverPolicyWithoutBrake
  from DrivingEnv.  That policy reads from the keyboard and stores its
  own (zero) output as info["raw_action"].
  • SharedControlMonitor asserts "raw_action" in info → crash.
  • Even if the assert were absent, HACOReplayBuffer.add() would record
    the keyboard action (zeros) as the behavior action for every step.

  FIX: Set "agent_policy": EnvInputPolicy in default_config() (exactly
  like ExpertTakeoverEnv).  EnvInputPolicy makes MetaDrive record the
  action passed to step() as raw_action — the expert action during
  takeover, the novice action otherwise.

BUG 3 (accounting double-count):
  After fixing Bug 1, the cost/counter updates would execute twice:
  once inside _get_step_return and again in ThriftyTakeoverEnv.step().
  The redundant post-super().step() block in step() has been removed.

Risk / novelty parameter guidance:
────────────────────────────────────
  --risk_threshold 0.0   (default)
    Take over when the critic predicts Q(s, a_novice) < 0.  Early in
    training the critic is noisy, so novelty carries most of the load.
    As training converges, Q-values for safe states rise well above 0
    and only genuinely dangerous states stay negative — the "thrifty"
    property emerges naturally.  Tune upward (e.g. 1.0–2.0) if the
    takeover rate stays too high after 5 k steps.

  --novelty_threshold 0.5  (default)
    k-NN distance in 32-dim random-projection space.  With MetaDrive
    observations ≈ 259-dim (mostly [0,1] range), projected typical
    distances are in [0.2, 2.0].  0.5 gives aggressive early takeovers
    (good — the expert teaches the novice from the start) and gradual
    decay as the novelty buffer fills.  Raise to 0.8–1.0 if the
    takeover rate exceeds 80 % after 2 k steps; lower to 0.3 if the
    rate falls below 10 % before 500 steps.

  --novelty_k 5            (default)
    Number of nearest neighbours.  Increasing to 10–15 smooths the
    novelty signal and prevents single-point aliasing.

  --novelty_proj_dim 32    (default)
    Higher values (64) improve accuracy at the cost of more memory /
    CPU per step.  32 is a good trade-off for 259-dim obs.
"""

import argparse
import copy
import os
import pathlib
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from ppl.sb3.common.buffers import ReplayBuffer
from ppl.sb3.common.save_util import load_from_pkl, save_to_pkl
from ppl.sb3.common.type_aliases import GymEnv, MaybeCallback
from ppl.sb3.common.utils import polyak_update
from ppl.sb3.haco.haco_buffer import HACOReplayBuffer, concat_samples
from ppl.sb3.td3.td3 import TD3
from ppl.sb3.td3.policies import TD3Policy
from ppl.experiments.metadrive.driving_env import DrivingEnv

FOLDER_PATH = pathlib.Path(__file__).parent


# =========================================================================== #
#  Novelty estimator                                                            #
# =========================================================================== #

class KNNNoveltyEstimator:
    """
    Estimates observation novelty via k-nearest-neighbour distance in a
    random-projection subspace.

    A state is declared *novel* when the k-NN distance exceeds
    `novelty_threshold`.  Before enough data has been collected (< k
    observations) every state is considered novel so the expert guides
    the agent from the start.
    """

    def __init__(
        self,
        obs_dim: int,
        proj_dim: int = 32,
        k: int = 5,
        max_obs: int = 10_000,
        novelty_threshold: float = 0.5,
        seed: int = 0,
    ):
        self.k = k
        self.proj_dim = proj_dim
        self.max_obs = max_obs
        self.novelty_threshold = novelty_threshold

        rng = np.random.RandomState(seed)
        self.proj = rng.randn(obs_dim, proj_dim).astype(np.float32)
        self.proj /= np.linalg.norm(self.proj, axis=0, keepdims=True) + 1e-8

        self._buf = np.zeros((max_obs, proj_dim), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def _project(self, obs: np.ndarray) -> np.ndarray:
        return obs.flatten() @ self.proj

    def add(self, obs: np.ndarray) -> None:
        self._buf[self._ptr] = self._project(obs)
        self._ptr = (self._ptr + 1) % self.max_obs
        self._size = min(self._size + 1, self.max_obs)

    def is_novel(self, obs: np.ndarray) -> bool:
        if self._size < self.k:
            return True
        q = self._project(obs)
        stored = self._buf[: self._size]
        dists = np.linalg.norm(stored - q, axis=1)
        knn_dist = float(np.partition(dists, self.k - 1)[self.k - 1])
        return knn_dist > self.novelty_threshold

    @property
    def size(self) -> int:
        return self._size


# =========================================================================== #
#  Expert loader                                                                #
# =========================================================================== #

def _load_expert():
    from ppl.sb3.common.save_util import load_from_zip_file
    from ppl.sb3.ppo import PPO
    from ppl.sb3.ppo.policies import ActorCriticPolicy

    _env = DrivingEnv(config={"manual_control": False, "use_render": False})
    model = PPO(
        policy=ActorCriticPolicy,
        n_steps=1024,
        n_epochs=20,
        learning_rate=5e-5,
        batch_size=256,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=10.0,
        create_eval_env=False,
        verbose=0,
        device="auto",
        env=_env,
    )
    ckpt = FOLDER_PATH / "metadrive_ppo_expert_20m_steps.zip"
    print(f"[ThriftyDAgger] Loading expert from {ckpt}")
    data, params, _ = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=True, device=model.device)
    _env.close()
    return model.policy


_expert_policy = _load_expert()


# =========================================================================== #
#  ThriftyTakeoverEnv                                                           #
# =========================================================================== #

class ThriftyTakeoverEnv(DrivingEnv):
    """
    Wraps DrivingEnv and replaces the takeover decision with the Thrifty DAgger
    criterion:  takeover = RISK  OR  NOVELTY

    Key design decisions that differ from the buggy original:

    1.  agent_policy = EnvInputPolicy   (BUG 2 FIX)
        MetaDrive stores the action passed to step() as info["raw_action"].
        HACOReplayBuffer reads raw_action as the behavior action.
        TakeoverPolicyWithoutBrake (the old default) would store its own
        keyboard action (zeros) instead — corrupting the replay buffer and
        triggering SharedControlMonitor's assertion.

    2.  _get_step_return is overridden   (BUG 1 FIX)
        DrivingEnv._get_step_return queries the engine's shared-control
        policy and overwrites self.takeover with its value (always False
        with no human at the keyboard).  Our override skips that query and
        preserves whatever self.takeover was set to in step().
    """

    last_obs: Optional[np.ndarray] = None
    last_takeover: bool = False
    expert = None
    drawn_points: list = []

    def default_config(self):
        config = super().default_config()
        # ------------------------------------------------------------------ #
        # BUG 2 FIX: use EnvInputPolicy so that MetaDrive sets               #
        # info["raw_action"] to the action passed to step() (expert action   #
        # during takeover, novice action otherwise).                          #
        # TakeoverPolicyWithoutBrake was inherited from DrivingEnv and        #
        # would have written keyboard-input zeros as raw_action on every step #
        # while also setting shared_control_policy.takeover = False always,   #
        # which was the root cause of Bug 1.                                  #
        # ------------------------------------------------------------------ #
        from metadrive.policy.env_input_policy import EnvInputPolicy
        config.update(
            {
                "agent_policy": EnvInputPolicy,      # ← BUG 2 FIX
                "manual_control": False,
                "use_render": False,
                # ---- Thrifty DAgger takeover parameters ----
                "risk_threshold": 3.0,
                "novelty_threshold": 0.65,
                "novelty_k": 5,
                "novelty_proj_dim": 32,
                "novelty_max_obs": 10_000,
                # ---- misc ----
                "expert_noise": 0.0,
                "disable_expert": False,
                # Kept for env-wrapper chain compatibility; unused here
                "num_predicted_steps": 1,
                "preference_horizon": 1,
            },
            allow_add_new_key=True,
        )
        return config

    def __init__(self, config: dict):
        super().__init__(config)
        self._novelty: Optional[KNNNoveltyEstimator] = None
        self.model = None  # set externally

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_novelty_estimator(self, obs: np.ndarray) -> None:
        if self._novelty is None:
            self._novelty = KNNNoveltyEstimator(
                obs_dim=obs.flatten().shape[0],
                proj_dim=self.config["novelty_proj_dim"],
                k=self.config["novelty_k"],
                max_obs=self.config["novelty_max_obs"],
                novelty_threshold=self.config["novelty_threshold"],
            )

    def _compute_risk(self, obs: np.ndarray, novice_action: np.ndarray) -> float:
        """
        Risk = min Q-value of the novice's action under the current critic.
        Returns +inf (= not risky) before the critic is warmed up.
        """
        if self.model is None or not hasattr(self.model, "critic"):
            return float("inf")
        if self.model.num_timesteps < self.model.learning_starts:
            return float("inf")

        obs_t = th.as_tensor(obs[np.newaxis], dtype=th.float32).to(self.model.device)
        act_t = th.as_tensor(novice_action[np.newaxis], dtype=th.float32).to(self.model.device)
        with th.no_grad():
            q_values = self.model.critic(obs_t, act_t)
            q_min = min(q.item() for q in q_values)
        return q_min

    def _decide_takeover(self, obs: np.ndarray, novice_action: np.ndarray) -> bool:
        risky = self._compute_risk(obs, novice_action) < self.config["risk_threshold"]
        novel = self._novelty.is_novel(obs)
        return risky or novel

    def _get_expert_action(self, obs: np.ndarray) -> np.ndarray:
        if self.expert is None:
            global _expert_policy
            self.expert = _expert_policy
        expert_action, _ = self.expert.predict(obs, deterministic=True)
        noise_bound = self.config["expert_noise"]
        if noise_bound > 0:
            expert_action = np.clip(
                expert_action + np.random.randn(*expert_action.shape) * noise_bound,
                self.action_space.low,
                self.action_space.high,
            )
        return expert_action.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  BUG 1 FIX: override _get_step_return                               #
    # ------------------------------------------------------------------ #

    def _get_step_return(self, actions, engine_info):
        """
        BUG 1 FIX — replaces DrivingEnv._get_step_return.

        DrivingEnv._get_step_return does:
            self.takeover = shared_control_policy.takeover   # always False

        This silently overwrote the Thrifty DAgger decision and caused
        every transition to be stored with takeover=False, so the expert
        buffer stayed empty and success_rate remained 0.

        We skip the engine-policy query entirely and keep self.takeover
        exactly as _decide_takeover() set it in step().
        """
        # Call *grandparent* _get_step_return (SafeMetaDriveEnv / BasePredictionEnv),
        # skipping DrivingEnv._get_step_return which would clobber self.takeover.
        o, r, tm, tc, engine_info = super(DrivingEnv, self)._get_step_return(
            actions, engine_info
        )
        self.last_obs = o
        d = tm or tc

        # self.takeover was set by _decide_takeover() in step() — do NOT touch it.
        engine_info["takeover_start"] = (not self.last_takeover) and self.takeover
        engine_info["takeover"] = self.takeover

        condition = (
            engine_info["takeover_start"]
            if self.config["only_takeover_start_cost"]
            else self.takeover
        )
        if not condition:
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost

        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info.get("cost", 0)
        engine_info["episode_native_cost"] = self.episode_cost
        self.total_cost += engine_info.get("cost", 0)
        engine_info["total_cost"] = self.total_cost
        self.total_takeover_count += 1 if self.takeover else 0
        engine_info["total_takeover_count"] = self.total_takeover_count

        return o, r, d, engine_info

    # ------------------------------------------------------------------ #
    #  Env overrides                                                       #
    # ------------------------------------------------------------------ #

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.last_obs = obs
        self.last_takeover = False
        self._ensure_novelty_estimator(obs)
        return obs

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        self.agent_action = copy.copy(actions)

        # Capture last_takeover BEFORE deciding the new one so
        # _get_step_return can compute takeover_start correctly.
        self.last_takeover = self.takeover
        obs = self.last_obs

        self._ensure_novelty_estimator(obs)

        # ---- Thrifty DAgger takeover decision ---- #
        if self.config["disable_expert"]:
            self.takeover = False
        else:
            self.takeover = self._decide_takeover(obs, actions)

        executed_actions = self._get_expert_action(obs) if self.takeover else actions

        # Record observation into novelty buffer (every step).
        self._novelty.add(obs)

        # ------------------------------------------------------------------ #
        # Call BasePredictionEnv.step(), bypassing DrivingEnv.step().        #
        # MetaDrive will internally call _get_step_return, which now         #
        # resolves to our override above (BUG 1 FIX).                        #
        # All info-key population and cost accounting happen there.          #
        # ------------------------------------------------------------------ #
        o, r, d, i = super(DrivingEnv, self).step(executed_actions)

        # BUG 3 FIX: do NOT repeat cost / counter accounting here.
        # _get_step_return already handled takeover, takeover_start,
        # takeover_cost, total_takeover_cost, native_cost, episode_native_cost,
        # total_cost, total_takeover_count.  Repeating them caused double-counting.

        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1

        if self.config["use_render"]:
            self.render(
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover": "TAKEOVER" if self.takeover else "NO",
                    "Total Step": self.total_steps,
                    "Novelty Buf": self._novelty.size,
                    "Takeover Rate": "{:.2f}%".format(
                        np.mean(np.array(self.takeover_recorder) * 100)
                    ),
                }
            )

        # last_obs is already set inside _get_step_return via self.last_obs = o
        return o, r, d, i

    def _get_reset_return(self, reset_info):
        o, info = super(DrivingEnv, self)._get_reset_return(reset_info)
        self.last_obs = o
        self.last_takeover = False
        return o, info


# =========================================================================== #
#  ThriftyDAgger algorithm                                                     #
# =========================================================================== #

class ThriftyDAgger(TD3):
    """
    Thrifty DAgger built on TD3.

    Two replay buffers:
      human_data_buffer : expert-intervention transitions.
      replay_buffer     : novice-only transitions.

    Critic : TD3 Bellman loss on 50 % novice + 50 % expert data.
    Actor  : intervention-masked BC loss on expert data only.
    """

    def __init__(
        self,
        use_balance_sample: bool = True,
        q_value_bound: float = 1.0,
        bc_loss_weight: float = 1.0,
        *args,
        **kwargs,
    ):
        self.bc_loss_weight = bc_loss_weight
        self.q_value_bound = q_value_bound
        self.use_balance_sample = use_balance_sample

        if "replay_buffer_class" not in kwargs:
            kwargs["replay_buffer_class"] = HACOReplayBuffer

        super().__init__(*args, **kwargs)

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

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        stat_recorder = defaultdict(list)

        for _ in range(gradient_steps):
            self._n_updates += 1

            if self.human_data_buffer.pos == 0:
                continue

            # ---- Balanced batch for critic -------------------------------- #
            if self.replay_buffer.pos > 0 and self.use_balance_sample:
                half = max(1, batch_size // 2)
                replay_data = concat_samples(
                    self.replay_buffer.sample(half, env=self._vec_normalize_env),
                    self.human_data_buffer.sample(half, env=self._vec_normalize_env),
                )
            else:
                replay_data = self.human_data_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )

            # ---- Critic update (TD3 Bellman) ------------------------------ #
            with th.no_grad():
                noise = replay_data.actions_behavior.clone().data.normal_(
                    0, self.target_policy_noise
                ).clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (
                    self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)
                next_q = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q, _ = th.min(next_q, dim=1, keepdim=True)
                target_q = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q
                )

            current_qs = self.critic(
                replay_data.observations, replay_data.actions_behavior
            )
            critic_loss = sum(F.mse_loss(q, target_q) for q in current_qs)

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            stat_recorder["critic_loss"].append(critic_loss.item())

            # ---- Actor update (delayed, BC on expert data only) ----------- #
            if self._n_updates % self.policy_delay == 0:
                expert_batch = self.human_data_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )
                pred_actions = self.actor(expert_batch.observations)
                per_step_bc = F.mse_loss(
                    expert_batch.actions_behavior, pred_actions, reduction="none"
                ).mean(axis=-1)
                masked_bc = (
                    expert_batch.interventions.flatten() * per_step_bc
                ).sum() / (expert_batch.interventions.flatten().sum() + 1e-5)

                actor_loss = self.bc_loss_weight * masked_bc

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                stat_recorder["actor_loss"].append(actor_loss.item())
                stat_recorder["bc_loss"].append(per_step_bc.mean().item())

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/human_involved_steps", self.human_data_buffer.pos)
        for key, values in stat_recorder.items():
            self.logger.record(f"train/{key}", np.mean(values))

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        """Route takeover transitions to the expert buffer."""
        if infos[0].get("takeover") or infos[0].get("takeover_start"):
            replay_buffer = self.human_data_buffer
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def save_replay_buffer(self, path_human, path_replay):
        save_to_pkl(path_human, self.human_data_buffer, self.verbose)
        super().save_replay_buffer(path_replay)

    def load_replay_buffer(self, path_human, path_replay, truncate_last_traj=True):
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
        save_buffer: bool = False,
        save_path_human: Union[str, pathlib.Path] = "",
        save_path_replay: Union[str, pathlib.Path] = "",
        buffer_save_timesteps: int = 2000,
        **kwargs,
    ) -> "ThriftyDAgger":
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes,
            eval_log_path, reset_num_timesteps, tb_log_name,
        )
        callback.on_training_start(locals(), globals())

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
                    self.gradient_steps
                    if self.gradient_steps >= 0
                    else rollout.episode_timesteps
                )
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

            if (
                save_buffer
                and self.num_timesteps > 0
                and self.num_timesteps % buffer_save_timesteps == 0
            ):
                self.save_replay_buffer(
                    os.path.join(save_path_human, f"human_buffer_{self.num_timesteps}.pkl"),
                    os.path.join(save_path_replay, f"replay_buffer_{self.num_timesteps}.pkl"),
                )

        callback.on_training_end()
        return self


# =========================================================================== #
#  Training script entry point                                                  #
# =========================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="thrifty_dagger_metadrive", type=str)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--save_freq", default=150, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_team", type=str, default="")
    parser.add_argument("--bc_loss_weight", type=float, default=1.0)
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--toy_env", action="store_true")
    # ---- Thrifty DAgger takeover thresholds ---- #
    parser.add_argument(
        "--risk_threshold", type=float, default=0.0,
        help=(
            "Q-value below this triggers expert takeover (RISK criterion). "
            "0.0 = take over when critic predicts negative future return. "
            "Raise to 1.0–2.0 if takeover rate stays >80 %% after 5 k steps."
        ),
    )
    parser.add_argument(
        "--novelty_threshold", type=float, default=0.5,
        help=(
            "k-NN distance above this triggers expert takeover (NOVELTY criterion). "
            "0.5 gives aggressive early takeovers that decay as the buffer fills. "
            "Raise to 0.8 if takeover rate >80 %% at 2 k steps; "
            "lower to 0.3 if rate <10 %% before 500 steps."
        ),
    )
    parser.add_argument(
        "--novelty_k", type=int, default=5,
        help="Number of nearest neighbours for novelty estimation.",
    )
    parser.add_argument(
        "--novelty_proj_dim", type=int, default=32,
        help="Dimension of the random projection used in the novelty buffer.",
    )
    args = parser.parse_args()

    # ===== Experiment naming / directory setup ===== #
    experiment_batch_name = "ThriftyDAgger"
    seed = args.seed
    trial_name = "{}_{}".format(experiment_batch_name, uuid.uuid4().hex[:8])
    print("Trial name is set to: ", trial_name)

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    log_dir = FOLDER_PATH.parent.parent
    experiment_dir = Path(log_dir) / "runs" / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=False)
    print(f"We start logging training data into {trial_dir}")

    # ===== Config ===== #
    config = dict(
        env_config=dict(
            risk_threshold=args.risk_threshold,
            novelty_threshold=args.novelty_threshold,
            novelty_k=args.novelty_k,
            novelty_proj_dim=args.novelty_proj_dim,
            novelty_max_obs=10_000,
        ),
        algo=dict(
            policy=TD3Policy,
            policy_kwargs=dict(net_arch=[256, 256]),
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(),
            bc_loss_weight=args.bc_loss_weight,
            use_balance_sample=True,
            q_value_bound=1,
            env=None,
            learning_rate=1e-4,
            optimize_memory_usage=True,
            buffer_size=50_000,
            learning_starts=10,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir),
    )

    if args.toy_env:
        config["env_config"].update(
            num_scenarios=1,
            traffic_density=0.0,
            map="COT",
            use_render=True,
        )

    # ===== Training environment ===== #
    from ppl.sb3.common.callbacks import CallbackList, CheckpointCallback
    from ppl.sb3.common.monitor import Monitor
    from ppl.sb3.common.vec_env import SubprocVecEnv
    from ppl.sb3.common.wandb_callback import WandbCallback
    from ppl.utils.shared_control_monitor import SharedControlMonitor

    train_env = ThriftyTakeoverEnv(config=config["env_config"])
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    train_env = SharedControlMonitor(
        env=train_env, folder=trial_dir / "data", prefix=trial_name
    )
    config["algo"]["env"] = train_env

    # ===== Eval environment (identical to PPL) ===== #
    def _make_eval_env():
        from ppl.sb3.common.monitor import Monitor as M
        e = DrivingEnv(config=dict(start_seed=1000))
        return M(env=e, filename=str(trial_dir))

    eval_env, eval_freq = SubprocVecEnv([_make_eval_env]), 150

    # ===== Callbacks ===== #
    callbacks = [
        CheckpointCallback(
            name_prefix="rl_model",
            verbose=2,
            save_freq=args.save_freq,
            save_path=str(trial_dir / "models"),
        )
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config=config,
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Instantiate model and wire into env ===== #
    model = ThriftyDAgger(**config["algo"])

    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from ppl.sb3.common.save_util import load_from_zip_file
        data, params, _ = load_from_zip_file(
            ckpt, device=model.device, print_system_info=False
        )
        model.set_parameters(params, exact_match=True, device=model.device)

    # Attach model so ThriftyTakeoverEnv can query the critic for risk scoring.
    train_env.env.env.model = model

    # ===== Launch training ===== #
    model.learn(
        total_timesteps=10_000,
        callback=callbacks,
        reset_num_timesteps=True,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=50,
        eval_log_path=str(trial_dir),
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
    )
