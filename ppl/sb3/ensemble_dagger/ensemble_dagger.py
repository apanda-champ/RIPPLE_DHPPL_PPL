"""
Ensemble DAgger (Dataset Aggregation with Ensemble Uncertainty)
===============================================================

Algorithm overview
------------------
1.  Maintain K actor networks (one main actor + K-1 ensemble members).
2.  Roll out the *main* actor in the environment.
3.  At every step the environment queries `model.get_uncertainty(obs)`.
    If the ensemble disagreement (std of predicted actions across K members)
    exceeds `uncertainty_threshold`, the expert takes over for that step.
4.  Expert-labeled transitions are stored in `human_data_buffer`.
5.  Every `train_freq` steps all K actors are updated via behaviour-cloning
    (BC) loss on `human_data_buffer`.

Key differences from PPL
-------------------------
* No preference buffer, no DPO/CPL loss.
* No critic / Q-value learning — pure BC throughout.
* Takeover criterion is ensemble disagreement, not trajectory-failure
  prediction.
* The ensemble is used *only* during training to gate expert queries;
  at evaluation time only the main actor (self.actor / self.policy.actor)
  is used.

Integration with existing code
-------------------------------
* Inherits from PVPTD3 so the entire SB3 `learn` loop, callback system,
  checkpoint saving, and `_store_transition` routing (human vs. agent
  buffer) are reused without modification.
* EnsembleDAggerEnv (see ensemble_dagger_env.py) calls
  `model.get_uncertainty(obs)` to decide whether to invoke the expert.
"""

import copy
import io
import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F

from ppl.ppl import PVPTD3
from ppl.sb3.common.type_aliases import GymEnv, MaybeCallback


class EnsembleDAgger(PVPTD3):
    """
    Ensemble DAgger algorithm built on top of the PVPTD3 scaffold.

    Parameters
    ----------
    num_ensemble : int
        Total number of actor networks in the ensemble (including the main
        actor).  Must be >= 2 so that disagreement can be measured.
    uncertainty_threshold : float
        Mean per-action-dimension standard-deviation threshold above which
        the expert is queried. Tune this to control how often the expert
        intervenes.
    *args, **kwargs
        Forwarded to PVPTD3 / TD3.  Note that critic-related hyper-
        parameters are still accepted (they appear in the parent's
        __init__) but the critic is **not trained** in this class.
    """

    def __init__(
        self,
        num_ensemble: int = 5,
        uncertainty_threshold: float = 0.05,
        *args,
        **kwargs,
    ):
        if num_ensemble < 2:
            raise ValueError("num_ensemble must be >= 2 to measure disagreement.")

        # IMPORTANT: these must be set BEFORE super().__init__() because
        # TD3.__init__ calls self._setup_model() internally, and our
        # override of _setup_model() reads these attributes.
        self.num_ensemble = num_ensemble
        self.uncertainty_threshold = uncertainty_threshold
        self._ensemble_actors: Optional[nn.ModuleList] = None
        self._ensemble_optimizers: Optional[List[th.optim.Optimizer]] = None
        self.human_steps = 0  # counts steps where expert took over

        # Force pure BC — critic losses from PVPTD3 are disabled in our
        # overridden train(), but setting these flags keeps extra_config
        # consistent in case any parent method inspects them.
        kwargs.setdefault("only_bc_loss", "True")
        kwargs.setdefault("add_bc_loss", "True")
        kwargs.setdefault("with_human_proxy_value_loss", "False")
        kwargs.setdefault("with_agent_proxy_value_loss", "False")

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        # Build policy, actor, critic targets etc. via parent.
        super()._setup_model()

        # After super(), self.actor == self.policy.actor is available.
        # Create K-1 additional actors that are deep-copies of the main one
        # AND reset their parameters to ensure distinct initializations.
        extra_actors = []
        for _ in range(self.num_ensemble - 1):
            cloned_actor = copy.deepcopy(self.actor)
            
            # Re-initialize the weights for the clone to break symmetry
            for module in cloned_actor.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                    
            extra_actors.append(cloned_actor.to(self.device))
            
        self._ensemble_actors = nn.ModuleList(extra_actors).to(self.device)

        # Each ensemble member gets its own Adam optimiser with the same
        # initial learning rate as the main actor.
        init_lr = self.actor.optimizer.param_groups[0]["lr"]
        self._ensemble_optimizers = [
            th.optim.Adam(actor.parameters(), lr=init_lr)
            for actor in self._ensemble_actors
        ]

    # ------------------------------------------------------------------
    # Count expert steps
    # ------------------------------------------------------------------

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        # The parent routes takeover transitions to human_data_buffer.
        # We detect this by checking the info flag before the parent call.
        if infos[0].get("takeover", False) or infos[0].get("takeover_start", False):
            self.human_steps += 1
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    # ------------------------------------------------------------------
    # Uncertainty estimation  (called by EnsembleDAggerEnv every step)
    # ------------------------------------------------------------------

    def get_uncertainty(self, obs: np.ndarray):
        """
        Compute ensemble disagreement for a single observation.
        """
        with th.no_grad():
            obs_tensor, _ = self.policy.obs_to_tensor(obs)

            # Collect predictions from main actor + ensemble members.
            all_actions: List[np.ndarray] = []
            all_actions.append(self.actor(obs_tensor).cpu().numpy())
            for actor in self._ensemble_actors:
                all_actions.append(actor(obs_tensor).cpu().numpy())

        # Shape: (K, 1, action_dim)
        stacked = np.stack(all_actions, axis=0)
        mean_action = stacked.mean(axis=0)           # (1, action_dim)
        std = stacked.std(axis=0).mean(axis=-1)      # (1,)  — avg across dims
        return mean_action, std

    # ------------------------------------------------------------------
    # Training  (pure BC for all K ensemble members)
    # ------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Train all ensemble members with behaviour-cloning on the
        human_data_buffer (transitions where the expert took over).
        The critic is intentionally not updated.
        """
        self.policy.set_training_mode(True)
        for actor in self._ensemble_actors:
            actor.train()

        # Update main actor's learning-rate schedule.
        self._update_learning_rate([self.actor.optimizer])

        stat_recorder: Dict[str, list] = defaultdict(list)

        for _ in range(gradient_steps):
            self._n_updates += 1

            # We only train once we have expert data.
            if self.human_data_buffer.pos == 0:
                break

            # ---- Main actor BC loss ----
            # Sample a batch specifically for the main actor
            replay_data_main = self.human_data_buffer.sample(
                int(batch_size), env=self._vec_normalize_env
            )

            pred_action = self.actor(replay_data_main.observations)
            bc_loss_main = F.mse_loss(
                replay_data_main.actions_behavior, pred_action
            )
            self.actor.optimizer.zero_grad()
            bc_loss_main.backward()
            self.actor.optimizer.step()

            stat_recorder["bc_loss_main"].append(bc_loss_main.item())

            # ---- Ensemble member BC losses ----
            ensemble_losses = []
            for actor, optim in zip(
                self._ensemble_actors, self._ensemble_optimizers
            ):
                # Sample a fresh, independent batch for each ensemble member
                replay_data_e = self.human_data_buffer.sample(
                    int(batch_size), env=self._vec_normalize_env
                )
                
                pred_e = actor(replay_data_e.observations)
                bc_loss_e = F.mse_loss(
                    replay_data_e.actions_behavior, pred_e
                )
                optim.zero_grad()
                bc_loss_e.backward()
                optim.step()
                ensemble_losses.append(bc_loss_e.item())

            stat_recorder["bc_loss_ensemble"].append(np.mean(ensemble_losses))
            stat_recorder["bc_loss_all"].append(
                (bc_loss_main.item() + np.mean(ensemble_losses)) / 2
            )

        self.logger.record("train/human_buffer_size", self.human_data_buffer.pos)
        self.logger.record("train/human_steps_total", self.human_steps)
        self.logger.record(
            "train/human_steps_ratio",
            round(self.human_steps / max(self.num_timesteps, 1), 3)
        )
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in stat_recorder.items():
            self.logger.record(f"train/{key}", np.mean(values))

    # ------------------------------------------------------------------
    # Save / load helpers for the ensemble
    # ------------------------------------------------------------------

    def save_ensemble(self, path: Union[str, pathlib.Path]) -> None:
        """
        Save the K-1 extra ensemble actors to *path*.
        The main actor is saved by the standard SB3 ``save()`` method.
        """
        os.makedirs(path, exist_ok=True)
        for i, actor in enumerate(self._ensemble_actors):
            save_path = os.path.join(path, f"ensemble_actor_{i}.pt")
            th.save(actor.state_dict(), save_path)
        print(f"[EnsembleDAgger] Saved {len(self._ensemble_actors)} "
              f"ensemble actors to {path}")

    def load_ensemble(self, path: Union[str, pathlib.Path]) -> None:
        """
        Load the K-1 extra ensemble actors from *path* (saved via
        ``save_ensemble``).  Requires ``_setup_model`` to have run first
        (i.e. the model must be initialised).
        """
        for i, actor in enumerate(self._ensemble_actors):
            load_path = os.path.join(path, f"ensemble_actor_{i}.pt")
            actor.load_state_dict(
                th.load(load_path, map_location=self.device)
            )
        print(f"[EnsembleDAgger] Loaded {len(self._ensemble_actors)} "
              f"ensemble actors from {path}")

    # ------------------------------------------------------------------
    # Override _excluded_save_params so ensemble actors are not included
    # in the standard SB3 zip (they are managed separately above).
    # ------------------------------------------------------------------

    def _excluded_save_params(self) -> List[str]:
        excluded = super()._excluded_save_params()
        # _ensemble_actors and _ensemble_optimizers are large and managed
        # separately; exclude from the standard zip file.
        return excluded + ["_ensemble_actors", "_ensemble_optimizers",
                           "preference_buffer"]

    # ------------------------------------------------------------------
    # learn() — identical to PVPTD3.learn but with a cleaner log name
    # and without the buffer save/load paths (not needed for DAgger).
    # We override only to remove unused parameters and clarify intent.
    # ------------------------------------------------------------------

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
        save_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        save_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        save_buffer: bool = False,       # disabled by default for DAgger
        load_buffer: bool = False,
        load_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        load_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        warmup: bool = False,
        warmup_steps: int = 5000,
    ) -> "EnsembleDAgger":
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            save_timesteps=save_timesteps,
            buffer_save_timesteps=buffer_save_timesteps,
            save_path_human=save_path_human,
            save_path_replay=save_path_replay,
            save_buffer=save_buffer,
            load_buffer=load_buffer,
            load_path_human=load_path_human,
            load_path_replay=load_path_replay,
            warmup=warmup,
            warmup_steps=warmup_steps,
        )
