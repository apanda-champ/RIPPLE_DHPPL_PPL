"""
RobosuiteExpertTakeoverEnv  –  Robosuite analogue of MetaDrive's ExpertTakeoverEnv.

This environment:
  1.  At each step, predicts the agent's future trajectory using the
      current SB3 policy.
  2.  If the predicted trajectory is "failing", the OSC expert takes over
      and its action is sent to the simulator instead.
  3.  Preference pairs (expert preferred over agent) are stored in the
      PPL ``preference_buffer`` for contrastive-preference learning.

PPL info dict keys injected into every ``step()`` return:
    takeover        (bool)   – whether the expert acted this step
    takeover_start  (bool)   – first step of a takeover episode
    takeover_cost   (float)  – cosine-based cost (0 if no takeover)
    raw_action      (array)  – the action *actually* applied to the env
    cost            (float)  – task-level cost (= takeover_cost here)

Robosuite setup
---------------
  Robot      : Franka Emika Panda (7-DOF)
  Controller : OSC_POSE  →  action = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
  Task       : Wipe  (wipe the surface clean)
"""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ppl.experiments.robosuite.robosuite_base_env import (
    RobosuiteBaseEnv,
    _flatten_obs,
)
from ppl.experiments.robosuite.osc_expert import get_osc_expert


# ---------------------------------------------------------------------------
# RobosuiteExpertTakeoverEnv
# ---------------------------------------------------------------------------

class RobosuiteExpertTakeoverEnv(RobosuiteBaseEnv):
    """
    Robosuite environment with automatic OSC-expert takeover for PPL training.

    Parameters
    ----------
    config : dict, optional
        Merged on top of ``DEFAULT_ENV_CFG``.  PPL-specific keys:

        num_predicted_steps  (int,   default 20)
            How many steps ahead to simulate when checking for failure.
        failure_check_freq   (int,   default 10)
            Run the trajectory predictor every N environment steps.
        preference_horizon   (int,   default 3)
            How many (expert, agent) preference pairs to store per takeover.
        expert_deterministic (bool,  default True)
            Whether the expert acts deterministically.
        expert_noise         (float, default 0.0)
            Std of Gaussian noise added to the expert action.
        expert_checkpoint    (str,   default None)
            Path to the ``osc_expert.zip`` checkpoint.
            If None, looks for ``osc_expert.zip`` next to this file.
        disable_expert       (bool,  default False)
            Bypass the expert; useful for ablations.
        only_takeover_start_cost (bool, default False)
            If True, cost is only counted on the first step of a takeover.
    """

    # Class-level singleton – shared across instances (mirrors MetaDrive)
    _expert = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Pull PPL-specific keys out before passing to super()
        cfg = copy.deepcopy(config or {})
        self._ppl_cfg = {
            "num_predicted_steps":      cfg.pop("num_predicted_steps",      20),
            "failure_check_freq":       cfg.pop("failure_check_freq",       10),
            "preference_horizon":       cfg.pop("preference_horizon",        3),
            "expert_deterministic":     cfg.pop("expert_deterministic",    True),
            "expert_noise":             cfg.pop("expert_noise",             0.0),
            "expert_checkpoint":        cfg.pop("expert_checkpoint",        None),
            "disable_expert":           cfg.pop("disable_expert",          False),
            "only_takeover_start_cost": cfg.pop("only_takeover_start_cost",False),
        }
        super().__init__(config=cfg)

        # Per-episode state
        self._last_takeover: bool = False
        self.total_steps_global: int = 0   # never resets across episodes

    # ------------------------------------------------------------------
    # Expert loading
    # ------------------------------------------------------------------

    def _get_expert(self):
        if RobosuiteExpertTakeoverEnv._expert is None:
            RobosuiteExpertTakeoverEnv._expert = get_osc_expert(
                self._ppl_cfg["expert_checkpoint"]
            )
        return RobosuiteExpertTakeoverEnv._expert

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> np.ndarray:
        obs = super().reset(**kwargs)
        self._last_takeover = False
        self.takeover = False
        return obs

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        PPL step:
          1. Record the agent's proposed action.
          2. Periodically predict the future trajectory to detect failure.
          3. If failure predicted → expert takes over.
          4. Store preference pairs when the expert takes over.
          5. Apply the chosen action, collect PPL-compatible info dict.
        """
        actions = np.asarray(actions, dtype=np.float32)
        self.agent_action = copy.copy(actions)
        self._last_takeover = self.takeover  # save last-step takeover flag

        num_pred   = self._ppl_cfg["num_predicted_steps"]
        check_freq = self._ppl_cfg["failure_check_freq"]
        pref_horiz = self._ppl_cfg["preference_horizon"]
        noise_std  = self._ppl_cfg["expert_noise"]
        deterministic = self._ppl_cfg["expert_deterministic"]

        expert = self._get_expert()

        # ---- Compute expert action (always needed for log-prob and fallback) ----
        expert_action, _ = expert.predict(self._last_obs, deterministic=deterministic)
        if noise_std > 0.0:
            expert_action = np.clip(
                expert_action + np.random.randn(*expert_action.shape) * noise_std,
                self.action_space.low,
                self.action_space.high,
            )

        # ---- Periodically check whether the agent will fail ----
        if self.total_steps_global % check_freq == 0:
            self.takeover = self._decide_takeover(self._last_obs, num_pred)

        # ---- Expert takes over if failure is predicted ----
        if self.takeover and not self._ppl_cfg["disable_expert"]:
            applied_action = expert_action.copy()

            # Store preference pairs: expert vs predicted-agent trajectory
            if self.model is not None and hasattr(self.model, "preference_buffer"):
                pred_traj, _ = self.predict_agent_future_trajectory(
                    self._last_obs, num_pred, action_behavior=self.agent_action.copy()
                )
                self._store_preference_pairs(pred_traj, pref_horiz, expert_action.copy())
        else:
            applied_action = actions.copy()

        # ---- Step the underlying environment ----
        obs, reward, done, info = super().step(applied_action)

        # ---- Expert log-probability (for logging / diagnostics) ----
        if not self._ppl_cfg["disable_expert"]:
            obs_tensor, _ = expert.obs_to_tensor(self._last_obs)
            dist = expert.get_distribution(obs_tensor)
            act_tensor = torch.from_numpy(actions).to(obs_tensor.device)
            log_prob = dist.log_prob(act_tensor).detach()
            info["takeover_log_prob"] = log_prob.item()

        # ---- Populate PPL info fields ----
        info["raw_action"]     = applied_action.copy()
        info["takeover"]       = bool(self.takeover)
        info["takeover_start"] = bool(not self._last_takeover and self.takeover)

        # Takeover cost
        condition = info["takeover_start"] if self._ppl_cfg["only_takeover_start_cost"] \
                    else self.takeover
        if condition:
            cost = self.get_takeover_cost(info)
            self.total_takeover_cost += cost
            info["takeover_cost"] = cost
            self.total_takeover_count += 1
        else:
            info["takeover_cost"] = 0.0

        info["cost"] = info["takeover_cost"]
        info["total_takeover_cost"]  = self.total_takeover_cost
        info["total_takeover_count"] = self.total_takeover_count
        info["total_steps"]          = self.total_steps_global

        # ---- Takeover rate logging ----
        self.takeover_recorder.append(self.takeover)
        self.total_steps_global += 1

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Failure detection
    # ------------------------------------------------------------------

    def _decide_takeover(self, obs: np.ndarray, num_predicted_steps: int) -> bool:
        """
        Predict the agent's future trajectory and return True if it looks
        like a failure (mirrors ``ExpertTakeoverEnv.decide_takeover``).
        """
        _traj, info = self.predict_agent_future_trajectory(obs, num_predicted_steps)
        return bool(info["failure"])

    # ------------------------------------------------------------------
    # Preference pair storage
    # ------------------------------------------------------------------

    def _store_preference_pairs(
        self,
        predicted_traj: List[Dict],
        preference_horizon: int,
        expert_action: np.ndarray,
    ) -> None:
        """
        Store (expert, agent) preference pairs into the PPL preference buffer.

        Mirrors ``ExpertTakeoverEnv.store_preference_pairs``:
          positive = (current_obs, expert_action)  [preferred]
          negative = predicted agent trajectory     [dispreferred]
        """
        buf = self.model.preference_buffer
        n = min(len(predicted_traj) - 1, preference_horizon)
        for step in range(n):
            positive_step = {
                "obs":      predicted_traj[step]["obs"].copy(),
                "action":   expert_action.copy(),
                "next_obs": predicted_traj[step]["obs"].copy(),  # same obs (expert corrects)
                "done":     False,
            }
            positive_traj = [positive_step]
            negative_traj = predicted_traj[step + 1:]
            buf.add(positive_traj, negative_traj)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _failure_reward_threshold(self) -> float:
        # Wipe shaped reward: roughly 0.05–0.15 per step for meaningful progress.
        # Over num_predicted_steps, a trajectory making no progress scores near 0.
        # This threshold flags trajectories accumulating less than 0.05 per step.
        return 0.05 * self._ppl_cfg["num_predicted_steps"]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    env = RobosuiteExpertTakeoverEnv(
        config={
            "has_renderer": args.render,
            "num_predicted_steps": 10,
            "failure_check_freq": 5,
            "disable_expert": True,   # No checkpoint needed for smoke test
        }
    )
    obs = env.reset()
    print(f"obs shape: {obs.shape}, action shape: {env.action_space.shape}")

    n_takeovers = 0
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if info["takeover"]:
            n_takeovers += 1
        if done:
            obs = env.reset()

    print(f"Finished {args.steps} steps, takeovers (disabled) = {n_takeovers}")
    env.close()
