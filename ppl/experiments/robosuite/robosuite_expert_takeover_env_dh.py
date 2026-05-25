"""
RobosuiteExpertTakeoverEnvDH
============================
Drop-in replacement for RobosuiteExpertTakeoverEnv when training with DH-PPL.

The only behavioural difference from the base takeover env is in
``_store_preference_pairs``: before each (expert, agent) preference pair is
added to the buffer, the method calls ``model.should_add_to_preference_buffer``
(provided by the DH-PPL ``PPL`` class in ``ppl_dh.py``).  That single call:
  1. Computes the epistemic uncertainty U(obs, action) via the ensemble.
  2. Updates the rolling mean window.
  3. Returns (accepted: bool, u: float).

Pairs with above-average uncertainty are rejected at storage time rather than
soft-weighted at training time, keeping the preference buffer clean.

All other PPL mechanics (takeover detection, reward shaping, info dict) are
inherited unchanged from RobosuiteExpertTakeoverEnv / RobosuiteBaseEnv.

Usage
-----
    from ppl.experiments.robosuite.robosuite_expert_takeover_env_dh import (
        RobosuiteExpertTakeoverEnvDH,
    )
    env = RobosuiteExpertTakeoverEnvDH(config={...})
    env.model = ppl_dh_model          # attach after PPL instantiation
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ppl.experiments.robosuite.robosuite_expert_takeover_env import (
    RobosuiteExpertTakeoverEnv,
)


class RobosuiteExpertTakeoverEnvDH(RobosuiteExpertTakeoverEnv):
    """
    Robosuite expert-takeover environment with DH-PPL uncertainty gating.

    Inherits everything from ``RobosuiteExpertTakeoverEnv`` and overrides
    only ``_store_preference_pairs`` to add the DH gate.

    Parameters
    ----------
    config : dict, optional
        Same keys as ``RobosuiteExpertTakeoverEnv`` — no additional keys
        are needed.  The DH gate parameters (window size, threshold) are
        managed inside the ``PPL`` model and require no env-level config.
    """

    # ------------------------------------------------------------------
    # Preference pair storage with DH gate
    # ------------------------------------------------------------------

    def _store_preference_pairs(
        self,
        predicted_traj: List[Dict],
        preference_horizon: int,
        expert_action: np.ndarray,
    ) -> None:
        """
        Store (expert, agent) preference pairs with Dynamic Horizon gating.

        For each step up to ``preference_horizon``:
          1. Call ``model.should_add_to_preference_buffer(obs, agent_action)``
             exactly once — returns (accepted: bool, u: float).
          2. If not accepted, skip this step (uncertainty too high).
          3. Otherwise build the contrastive pair and add to the buffer.

        All admit/reject decisions and uncertainty values are logged to
        stdout and (when a W&B run is active) to W&B.

        Graceful degradation
        --------------------
        If ``self.model`` does not have ``should_add_to_preference_buffer``
        (e.g. the environment is paired with vanilla PPL instead of DH-PPL),
        the method falls back to unconditional insertion — identical to the
        base ``RobosuiteExpertTakeoverEnv`` behaviour.
        """
        model = self.model
        has_gate = (
            model is not None
            and hasattr(model, "should_add_to_preference_buffer")
            and hasattr(model, "preference_buffer")
        )

        admitted  = 0
        rejected  = 0
        u_values: List[float] = []

        n = min(len(predicted_traj) - 1, preference_horizon)

        for step in range(n):
            step_obs          = predicted_traj[step]["obs"]
            step_agent_action = predicted_traj[step]["action"]

            if has_gate:
                # Single call — never query the model twice per step.
                should_add, u = model.should_add_to_preference_buffer(
                    step_obs, step_agent_action
                )
                u_values.append(u)

                print(
                    f"  [PrefBuffer] step={step} | "
                    f"U={u:.4f} | "
                    f"L(threshold)={model._uncertainty_threshold:.4f} | "
                    f"window={len(model._uncertainty_window)} | "
                    f"{'ADMIT' if should_add else 'REJECT'}"
                )

                if not should_add:
                    rejected += 1
                    continue

                admitted += 1

            # Build contrastive pair:
            #   positive = (current obs, expert action)   [preferred]
            #   negative = remaining predicted trajectory [dispreferred]
            positive_step = {
                "obs":      step_obs.copy(),
                "action":   expert_action.copy(),
                "next_obs": step_obs.copy(),   # expert corrects in-place
                "done":     False,
            }
            negative_traj = predicted_traj[step + 1:]

            if has_gate:
                model.preference_buffer.add([positive_step], negative_traj)
            else:
                # Fallback: unconditional insertion (vanilla PPL behaviour)
                self.model.preference_buffer.add([positive_step], negative_traj)

        # ------------------------------------------------------------------
        # Summary logging
        # ------------------------------------------------------------------
        if has_gate:
            threshold_val = model._uncertainty_threshold
            buffer_pos    = model.preference_buffer.pos
            total         = admitted + rejected

            print(
                f"[PrefBuffer SUMMARY] "
                f"env_step={self.total_steps_global} | "
                f"admitted={admitted} | "
                f"rejected={rejected} | "
                f"buffer_size={buffer_pos} | "
                f"threshold_L={threshold_val:.4f}"
            )

            # W&B logging (non-fatal if W&B is not installed / not running)
            try:
                import wandb
                if wandb.run is not None:
                    log_threshold = (
                        threshold_val if threshold_val != float("inf") else 0.0
                    )
                    wandb.log(
                        {
                            "pref_buffer/admitted":          admitted,
                            "pref_buffer/rejected":          rejected,
                            "pref_buffer/total_attempted":   total,
                            "pref_buffer/accept_rate":       admitted / max(total, 1),
                            "pref_buffer/threshold_L":       log_threshold,
                            "pref_buffer/uncertainty_mean":  float(np.mean(u_values))  if u_values else 0.0,
                            "pref_buffer/uncertainty_max":   float(np.max(u_values))   if u_values else 0.0,
                            "pref_buffer/uncertainty_min":   float(np.min(u_values))   if u_values else 0.0,
                            "pref_buffer/buffer_size":       buffer_pos,
                            "pref_buffer/env_step":          self.total_steps_global,
                        },
                        step=self.total_steps_global,
                    )
            except Exception as exc:
                print(f"[wandb log failed] {exc}")


# ---------------------------------------------------------------------------
# Quick smoke-test (no expert checkpoint required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    env = RobosuiteExpertTakeoverEnvDH(
        config={
            "has_renderer":        args.render,
            "num_predicted_steps": 10,
            "failure_check_freq":  5,
            "disable_expert":      True,   # no checkpoint needed for smoke test
        }
    )
    obs = env.reset()
    print(f"obs shape: {obs.shape}  |  action shape: {env.action_space.shape}")

    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    print("Smoke-test passed.")
    env.close()
