"""
RobosuiteRIPPLEEnv
==================
Robosuite analogue of RIPPLEEnv (ppl/experiments/metadrive/ripple_env.py).

Extends RobosuiteExpertTakeoverEnv with three additional signal sources on
top of the forward preferences that the base takeover env already collects:

  1. Backward propagation
     At the first step of a new takeover, the K most recent pre-intervention
     (obs, agent_action) pairs are retroactively labelled as
       positive = (obs, expert_action)   [what the expert would have done]
       negative = (obs, agent_action)    [what the agent actually did]
     and written to model.back_pref_buffer.

  2. Silent approval
     At every decision boundary where the expert declines to intervene
     (self.takeover = False), the closed-loop forecast is walked and
       positive = (obs, planned_action)
       negative = (obs, planned_action + clipped_gaussian_noise)
     are written to model.silent_pref_buffer for the first
     preference_horizon predicted states.  A minimum Euclidean margin is
     enforced on the noise so positives and negatives are meaningfully
     separated.

  3. Trajectory-level preferences
     On takeover start the rejected closed-loop forecast is saved as
     bad_traj.  The actual expert rollout is accumulated step-by-step as
     good_traj.  When the takeover ends (or good_traj hits traj_max_len)
     the pair is committed to model.traj_pref_buffer.

The step() method bypasses RobosuiteExpertTakeoverEnv.step() (by calling
super(RobosuiteExpertTakeoverEnv, self).step()) and re-implements the full
PPL info-dict bookkeeping inline, which is the same pattern used by
RIPPLEEnv in the MetaDrive codebase.

Place at:
    ppl/experiments/robosuite/robosuite_ripple_env.py

Usage:
    env = RobosuiteRIPPLEEnv(config={...})
    env.model = ripple_model          # attach after RIPPLE instantiation
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ppl.experiments.robosuite.robosuite_base_env import RobosuiteBaseEnv
from ppl.experiments.robosuite.robosuite_expert_takeover_env import (
    RobosuiteExpertTakeoverEnv,
)


class RobosuiteRIPPLEEnv(RobosuiteExpertTakeoverEnv):
    """
    Robosuite expert-takeover environment extended with RIPPLE signal sources.

    Parameters
    ----------
    config : dict, optional
        All keys accepted by RobosuiteExpertTakeoverEnv, plus:

        backward_horizon   (int,   default 3)
            Number of pre-intervention steps to retroactively label.
        silent_noise_scale (float, default 0.3)
            Std of Gaussian noise used to build silent-approval negatives.
        silent_margin      (float, default 0.15)
            Minimum Euclidean separation enforced between positive and
            negative actions in the silent-approval signal.
        traj_max_len       (int,   default 10)
            Maximum length of a (good_traj, bad_traj) trajectory pair
            before it is committed to traj_pref_buffer.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = copy.deepcopy(config or {})

        # Pop RIPPLE-specific keys before passing to super()
        self._ripple_cfg = {
            "backward_horizon":    cfg.pop("backward_horizon",    3),
            "silent_noise_scale":  cfg.pop("silent_noise_scale",  0.3),
            "silent_margin":       cfg.pop("silent_margin",       0.15),
            "traj_max_len":        cfg.pop("traj_max_len",        10),
        }

        super().__init__(config=cfg)
        self._init_ripple_state()

    # ------------------------------------------------------------------
    # RIPPLE state management
    # ------------------------------------------------------------------

    def _init_ripple_state(self) -> None:
        K = max(1, int(self._ripple_cfg["backward_horizon"]))
        # Stores {obs, action} dicts for the K most recent agent steps
        self._action_history: deque = deque(maxlen=K)
        # Rejected closed-loop forecast (bad side of traj-level pair)
        self._pending_traj_bad: Optional[List[Dict]] = None
        # Actual expert rollout during takeover (good side)
        self._pending_traj_good: List[Dict] = []

    def _has_buffer(self, name: str) -> bool:
        return (
            hasattr(self, "model")
            and self.model is not None
            and hasattr(self.model, name)
        )

    def _commit_pending_trajectory(self, min_len: int = 2) -> None:
        """
        Commit the accumulated (good_traj, bad_traj) pair to traj_pref_buffer
        if both sides are long enough and the buffer exists.
        """
        if self._pending_traj_bad is None:
            return
        if len(self._pending_traj_good) < min_len:
            return
        if not self._has_buffer("traj_pref_buffer"):
            return
        self.model.traj_pref_buffer.add(
            list(self._pending_traj_good),
            list(self._pending_traj_bad),
        )

    # ------------------------------------------------------------------
    # RIPPLE signal writers
    # ------------------------------------------------------------------

    def _store_backward_preferences(self, expert_action: np.ndarray) -> None:
        """
        Retroactively label the K most recent pre-intervention agent steps.

        positive = (past_obs, expert_action)   [what the expert would have done]
        negative = (past_obs, past_agent_action)
        """
        if not self._has_buffer("back_pref_buffer"):
            return
        exp_a = np.asarray(expert_action, dtype=np.float32).copy()
        for past in list(self._action_history):
            if past["obs"] is None:
                continue
            pos = [{"obs": past["obs"].copy(), "action": exp_a.copy()}]
            neg = [{"obs": past["obs"].copy(), "action": np.asarray(past["action"], dtype=np.float32).copy()}]
            self.model.back_pref_buffer.add(pos, neg)

    def _store_silent_preferences(
        self,
        predicted_traj: List[Dict],
        horizon: int,
    ) -> None:
        """
        Non-intervention at a decision boundary = implicit approval.

        For each of the first ``horizon`` steps in the closed-loop forecast:
          positive = (obs, planned_action)
          negative = (obs, planned_action + noise)    [min margin enforced]
        """
        if not self._has_buffer("silent_pref_buffer"):
            return

        noise_scale = float(self._ripple_cfg["silent_noise_scale"])
        margin      = float(self._ripple_cfg["silent_margin"])
        low  = np.asarray(self.action_space.low,  dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)

        n = min(len(predicted_traj), horizon)
        for i in range(n):
            state    = predicted_traj[i]
            approved = np.asarray(state["action"], dtype=np.float32).reshape(-1)

            noise = np.random.randn(approved.shape[0]).astype(np.float32) * noise_scale
            nrm   = float(np.linalg.norm(noise))
            if nrm < margin:
                # Ensure the negative is meaningfully different from the positive
                noise = noise / (nrm + 1e-6) * margin

            neg_action = np.clip(approved + noise, low, high)

            pos = [{"obs": state["obs"].copy(), "action": approved.copy()}]
            neg = [{"obs": state["obs"].copy(), "action": neg_action}]
            self.model.silent_pref_buffer.add(pos, neg)

    # ------------------------------------------------------------------
    # Core step — bypasses RobosuiteExpertTakeoverEnv.step()
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        PPL + RIPPLE step.

        Structure
        ---------
        1.  Record the agent's proposed action and snapshot (obs, action).
        2.  Periodically predict the closed-loop future trajectory to detect
            failure (same frequency as the base takeover env).
        3.  If failure predicted → expert takes over:
              a. Forward preferences  (base PPL, via _store_preference_pairs)
              b. Backward preferences (RIPPLE, on fresh takeover start)
              c. Trajectory tracking  (RIPPLE, accumulate good_traj)
        4.  If no failure predicted → silent approval (RIPPLE).
        5.  Apply the chosen action via RobosuiteBaseEnv.step() and populate
            the full PPL info dict.
        6.  Update action history with the pre-step (obs, agent_action).
        """
        actions = np.asarray(actions, dtype=np.float32)
        self.agent_action   = copy.copy(actions)
        self._last_takeover = self.takeover   # save flag from previous step

        num_pred      = self._ppl_cfg["num_predicted_steps"]
        check_freq    = self._ppl_cfg["failure_check_freq"]
        pref_horiz    = self._ppl_cfg["preference_horizon"]
        noise_std     = self._ppl_cfg["expert_noise"]
        deterministic = self._ppl_cfg["expert_deterministic"]
        traj_max_len  = int(self._ripple_cfg["traj_max_len"])

        expert = self._get_expert()

        # ---- Expert action (always computed for log-prob diagnostics) --------
        expert_action, _ = expert.predict(self._last_obs, deterministic=deterministic)
        if noise_std > 0.0:
            expert_action = np.clip(
                expert_action + np.random.randn(*expert_action.shape) * noise_std,
                self.action_space.low,
                self.action_space.high,
            )

        # ---- Snapshot BEFORE the env step (needed for backward prefs) --------
        state_at_t  = self._last_obs.copy()
        action_at_t = self.agent_action.copy()

        # ---- Periodically predict the closed-loop future trajectory ----------
        decision_made = (self.total_steps_global % check_freq == 0)
        forecast_traj: Optional[List[Dict]] = None

        if decision_made:
            forecast_traj, forecast_info = self.predict_agent_future_trajectory(
                self._last_obs, num_pred
            )
            self.takeover = bool(forecast_info["failure"])

        # ---- Takeover branch -------------------------------------------------
        if self.takeover and not self._ppl_cfg["disable_expert"]:
            applied_action = expert_action.copy()

            if self.model is not None and hasattr(self.model, "preference_buffer"):

                # ---- Forward preferences (base PPL) --------------------------
                pred_traj_open, _ = self.predict_agent_future_trajectory(
                    self._last_obs, num_pred, action_behavior=self.agent_action.copy()
                )
                self._store_preference_pairs(
                    pred_traj_open, pref_horiz, expert_action.copy()
                )

                # ---- RIPPLE: backward preferences on fresh takeover start ----
                if decision_made and not self._last_takeover:
                    self._store_backward_preferences(expert_action)

                # ---- RIPPLE: (re-)initialise trajectory pair -----------------
                # On every decision boundary during takeover we commit whatever
                # was pending and start fresh with the new forecast.
                if decision_made:
                    self._commit_pending_trajectory()
                    self._pending_traj_bad  = (
                        forecast_traj[:traj_max_len] if forecast_traj else None
                    )
                    self._pending_traj_good = []

                # ---- RIPPLE: accumulate expert rollout -----------------------
                if self._pending_traj_bad is not None:
                    self._pending_traj_good.append(
                        {"obs": state_at_t, "action": expert_action.copy()}
                    )
                    if len(self._pending_traj_good) >= traj_max_len:
                        self._commit_pending_trajectory()
                        self._pending_traj_bad  = None
                        self._pending_traj_good = []

        # ---- No-takeover branch ----------------------------------------------
        else:
            applied_action = actions.copy()

            # Just exited a takeover → commit accumulated trajectory pair
            if self._last_takeover and self._pending_traj_bad is not None:
                self._commit_pending_trajectory()
                self._pending_traj_bad  = None
                self._pending_traj_good = []

            # ---- RIPPLE: silent approval at non-intervention decision boundary
            if decision_made and forecast_traj is not None:
                self._store_silent_preferences(forecast_traj, pref_horiz)

        # ---- Step the base env (skips ExpertTakeoverEnv.step()) --------------
        obs, reward, done, info = super(RobosuiteExpertTakeoverEnv, self).step(
            applied_action
        )

        # ---- Expert log-probability (diagnostics) ----------------------------
        if not self._ppl_cfg["disable_expert"]:
            obs_tensor, _ = expert.obs_to_tensor(self._last_obs)
            dist           = expert.get_distribution(obs_tensor)
            act_tensor     = torch.from_numpy(actions).to(obs_tensor.device)
            log_prob       = dist.log_prob(act_tensor).detach()
            info["takeover_log_prob"] = log_prob.item()

        # ---- Populate PPL info fields ----------------------------------------
        info["raw_action"]     = applied_action.copy()
        info["takeover"]       = bool(self.takeover)
        info["takeover_start"] = bool(not self._last_takeover and self.takeover)

        condition = (
            info["takeover_start"]
            if self._ppl_cfg["only_takeover_start_cost"]
            else self.takeover
        )
        if condition:
            cost = self.get_takeover_cost(info)
            self.total_takeover_cost  += cost
            info["takeover_cost"]      = cost
            self.total_takeover_count += 1
        else:
            info["takeover_cost"] = 0.0

        info["cost"]                  = info["takeover_cost"]
        info["total_takeover_cost"]   = self.total_takeover_cost
        info["total_takeover_count"]  = self.total_takeover_count
        info["total_steps"]           = self.total_steps_global

        # ---- Update action history with pre-step snapshot -------------------
        # Done AFTER the env step so self._last_obs has been updated by
        # RobosuiteBaseEnv.step() before we snapshot next time.
        self._action_history.append({"obs": state_at_t, "action": action_at_t})

        self.takeover_recorder.append(self.takeover)
        self.total_steps_global += 1

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Reset — flush pending trajectory and reinit RIPPLE state
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> np.ndarray:
        # Commit anything pending from the previous episode before resetting
        self._commit_pending_trajectory()
        self._init_ripple_state()
        return super().reset(**kwargs)


# ---------------------------------------------------------------------------
# Quick smoke-test (no expert checkpoint required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps",  type=int, default=100)
    args = parser.parse_args()

    env = RobosuiteRIPPLEEnv(
        config={
            "has_renderer":        args.render,
            "num_predicted_steps": 10,
            "failure_check_freq":  5,
            "disable_expert":      True,    # no checkpoint needed for smoke test
            "backward_horizon":    3,
            "silent_noise_scale":  0.3,
            "silent_margin":       0.15,
            "traj_max_len":        10,
        }
    )
    obs = env.reset()
    print(f"obs shape: {obs.shape}  |  action shape: {env.action_space.shape}")

    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    print("Smoke-test passed.")
    env.close()
