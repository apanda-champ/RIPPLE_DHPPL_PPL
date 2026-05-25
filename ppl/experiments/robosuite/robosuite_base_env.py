"""
Base Robosuite gym-compatible environment for PPL.

Wraps a Robosuite task (default: Wipe with Panda + OSC_POSE) as a gym.Env
and adds MuJoCo state-snapshot trajectory prediction – the Robosuite analogue
of BasePredictionEnv in MetaDrive.

Tested with:
  robosuite==1.4.1  |  Python 3.8  |  Ubuntu 22.04
  Robot: Franka Emika Panda (7-DOF)
  Controller: OSC_POSE  →  action dim = 7 (Δeef xyz+rpy + gripper)
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper


# ---------------------------------------------------------------------------
# Default environment / controller configuration
# ---------------------------------------------------------------------------

DEFAULT_ENV_CFG: Dict[str, Any] = {
    "env_name": "Wipe",
    "robots": "Panda",
    "controller_configs": load_controller_config(default_controller="OSC_POSE"),
    "horizon": 750,
    "ignore_done": False,
    "reward_shaping": True,
    "use_object_obs": True,
    "use_camera_obs": False,
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "render_camera": "frontview",
    "control_freq": 20,
    "task_config": {
        # reward settings — all at Robosuite defaults except num_markers
        "arm_limit_collision_penalty": -10.0,
        "wipe_contact_reward": 0.01,
        "unit_wiped_reward": 50.0,
        "ee_accel_penalty": 0,
        "excess_force_penalty_mul": 0.05,
        "distance_multiplier": 5.0,
        "distance_th_multiplier": 5.0,
        # table settings
        "table_full_size": [0.5, 0.8, 0.05],
        "table_offset": [0.15, 0, 0.9],
        "table_friction": [0.03, 0.005, 0.0001],
        "table_friction_std": 0,
        "table_height": 0.0,
        "table_height_std": 0.0,
        "line_width": 0.04,
        "two_clusters": False,
        "coverage_factor": 0.6,
        "num_markers": 50,          # reduced from 100; makes full coverage tractable
        # threshold settings
        "contact_threshold": 1.0,
        "pressure_threshold": 0.5,
        "pressure_threshold_max": 60.0,
        # misc settings
        "print_results": False,
        "get_info": False,
        "use_robot_obs": True,
        "use_contact_obs": True,
        "early_terminations": True,
        "use_condensed_obj_obs": True,
    },
}

# Keys that are included in the flat observation vector.
# Order matters – must be identical at train and eval time.
OBS_KEYS: List[str] = [
    "robot0_joint_pos_cos",       # 7
    "robot0_joint_pos_sin",       # 7
    "robot0_joint_vel",           # 7
    "robot0_eef_pos",             # 3
    "robot0_eef_quat",            # 4
    "robot0_contact",             # 1 (scalar contact force)
    "wipe_centroid",              # 3 (centre of remaining dirt)
    "wipe_radius",                # 1 (spread of remaining dirt)
    "gripper_to_wipe_centroid",   # 3 (vector from EEF to dirt centre)
    "proportion_wiped",           # 1 (task progress) — must stay last
]

# ---------------------------------------------------------------------------
# CHANGED: Success threshold lowered from 0.94 → 0.90.
# Used in:
#   (1) predict_agent_future_trajectory() — failure detection during PPL rollouts
#   (2) step() — is_success flag reported to EvalCallback for success_rate logging
# ---------------------------------------------------------------------------
PROPORTION_WIPED_SUCCESS_THRESHOLD = 1.0


def _flatten_obs(obs_dict: Dict[str, np.ndarray], keys: List[str]) -> np.ndarray:
    """Flatten selected keys from an obs dict into a single 1-D array."""
    try:
        return np.concatenate([np.asarray(obs_dict[k]).flatten() for k in keys])
    except KeyError as e:
        raise KeyError(
            f"Observation key {e} not found in Robosuite. "
            f"Available keys: {list(obs_dict.keys())}"
        )


# ---------------------------------------------------------------------------
# RobosuiteBaseEnv
# ---------------------------------------------------------------------------

class RobosuiteBaseEnv(gym.Env):
    """
    Gym-compatible wrapper around a Robosuite environment.

    Key extras over the stock GymWrapper:
      • MuJoCo state save / restore (needed for PPL rollout prediction).
      • ``predict_agent_future_trajectory`` – rolls the simulation forward
        N steps using the agent's current policy, then restores the state.
      • PPL-style bookkeeping: takeover flag, costs, step counters.

    Success criterion (used in both trajectory failure detection and eval):
      An episode is successful if ANY of the following hold at done=True:
        (a) all num_markers are physically contacted (_check_success()), OR
        (b) proportion_wiped >= PROPORTION_WIPED_SUCCESS_THRESHOLD (0.90)
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    # Class-level defaults overridden by subclasses / config dicts
    total_steps: int = 0
    total_takeover_cost: float = 0.0
    total_takeover_count: int = 0
    total_cost: float = 0.0
    takeover: bool = False
    agent_action: Optional[np.ndarray] = None

    # Will be set externally by the training script (mirrors MetaDrive pattern)
    model = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = copy.deepcopy(DEFAULT_ENV_CFG)
        if config:
            cfg.update(config)

        self._cfg = cfg
        self._obs_keys = cfg.pop("obs_keys", OBS_KEYS)

        # Instantiate underlying Robosuite env
        self._rs_env = suite.make(**cfg)

        # Cache action / observation spaces
        act_dim = self._rs_env.action_spec[0].shape[0]
        act_low  = self._rs_env.action_spec[0]
        act_high = self._rs_env.action_spec[1]
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        obs_dim = self._compute_obs_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # PPL bookkeeping
        self.takeover_recorder: deque = deque(maxlen=2000)
        self._last_obs: Optional[np.ndarray] = None
        self._last_takeover: bool = False

        # Snapshot for wiped_markers restoration after trajectory prediction
        self._saved_wiped_markers: List = []

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> np.ndarray:
        raw_obs = self._rs_env.reset()
        obs = _flatten_obs(raw_obs, self._obs_keys).astype(np.float32)
        self._last_obs = obs
        self.takeover = False
        self._last_takeover = False
        self.agent_action = None
        # Per-episode resets
        self.total_steps = 0
        self.total_takeover_cost = 0.0
        self.total_takeover_count = 0
        self.total_cost = 0.0
        self.takeover_recorder.clear()
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = np.asarray(action, dtype=np.float32)
        self.agent_action = copy.copy(action)

        raw_obs, reward, done, info = self._rs_env.step(action)
        obs = _flatten_obs(raw_obs, self._obs_keys).astype(np.float32)
        self._last_obs = obs

        # ---- Completion bonus ----
        # +25.0 awarded when Robosuite native success fires.
        # Native success = all markers physically contacted (_check_success()).
        # The proportion_wiped >= 0.90 success criterion is evaluated separately
        # below for is_success and in trajectory failure detection; it does NOT
        # trigger the bonus reward to avoid interfering with Robosuite's shaping.
        native_success = bool(self._rs_env._check_success())
        if native_success:
            reward += 25.0

        # ---- is_success flag for EvalCallback success_rate logging ----
        # CHANGED: episode is successful if EITHER:
        #   (a) Robosuite native success fires (_check_success()), OR
        #   (b) proportion_wiped >= 0.90 at episode end.
        # obs[-1] is proportion_wiped (guaranteed last by OBS_KEYS ordering).
        # EvalCallback reads this only when done=True, so it correctly
        # captures both early-termination and horizon-reached (750 steps) success.
        proportion_wiped = float(obs[-1])
        coverage_success = (proportion_wiped >= PROPORTION_WIPED_SUCCESS_THRESHOLD)
        info["is_success"] = native_success or coverage_success

        # PPL info fields (filled by ExpertTakeoverEnv; defaults here)
        info.setdefault("takeover", False)
        info.setdefault("takeover_start", False)
        info.setdefault("takeover_cost", 0.0)
        info.setdefault("raw_action", action.copy())
        info.setdefault("cost", 0.0)

        info["total_takeover_cost"] = self.total_takeover_cost
        info["total_takeover_count"] = self.total_takeover_count
        info["total_steps"] = self.total_steps

        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1
        return obs, float(reward), done, info

    def render(self, mode: str = "human"):
        if mode == "human":
            self._rs_env.render()
        elif mode == "rgb_array":
            return self._rs_env.sim.render(
                camera_name="frontview", width=256, height=256, depth=False
            )

    def close(self):
        self._rs_env.close()

    # ------------------------------------------------------------------
    # MuJoCo state snapshot helpers
    # ------------------------------------------------------------------

    def _save_sim_state(self) -> np.ndarray:
        """
        Return a flattened numpy array of the current MuJoCo sim state.
        Also saves wiped_markers so they can be restored after rollouts.
        """
        # CRITICAL: save wiped_markers alongside MuJoCo state so that
        # markers wiped during trajectory prediction do not persist
        # into the real episode and cause false successes.
        self._saved_wiped_markers = list(self._rs_env.wiped_markers)
        return self._rs_env.sim.get_state().flatten()

    def _restore_sim_state(self, state: np.ndarray) -> None:
        """
        Restore MuJoCo sim from a flattened state array.
        Also restores wiped_markers to pre-rollout state.
        """
        self._rs_env.sim.set_state_from_flattened(state)
        self._rs_env.sim.forward()

        # Reset Robosuite's internal termination flag
        if hasattr(self._rs_env, "done"):
            self._rs_env.done = False

        # CRITICAL: restore wiped_markers to pre-rollout state.
        # Without this, markers wiped during trajectory prediction
        # persist into the real episode causing false successes.
        self._rs_env.wiped_markers = list(self._saved_wiped_markers)

    # ------------------------------------------------------------------
    # Trajectory prediction (PPL core primitive)
    # ------------------------------------------------------------------

    def predict_agent_future_trajectory(
        self,
        current_obs: np.ndarray,
        n_steps: int,
        action_behavior: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict], Dict]:
        """
        Roll the simulation forward ``n_steps`` steps using either a fixed
        ``action_behavior`` or the agent's learned policy (``self.model``),
        then restore the original sim state.

        Returns
        -------
        traj : list of step dicts
            Each entry has keys: obs, action, reward, next_obs, done.
        info : dict
            ``failure`` (bool) and ``total_reward`` (float).
        """
        saved_state = self._save_sim_state()

        traj: List[Dict] = []
        obs = current_obs.copy()
        total_reward = 0.0
        failure = False

        for _ in range(n_steps):
            # Choose action
            if action_behavior is not None:
                action = action_behavior.copy()
            elif self.model is not None:
                action, _ = self.model.policy.predict(obs, deterministic=True)
            else:
                # Fallback: zero action
                action = np.zeros(self.action_space.shape, dtype=np.float32)

            action = np.clip(action, self.action_space.low, self.action_space.high)

            raw_obs, reward, done, step_info = self._rs_env.step(action)
            next_obs = _flatten_obs(raw_obs, self._obs_keys).astype(np.float32)
            total_reward += reward

            traj.append(
                {
                    "obs": obs.copy(),
                    "action": action.copy(),
                    "reward": float(reward),
                    "next_obs": next_obs.copy(),
                    "done": bool(done),
                    "info": step_info,
                }
            )
            obs = next_obs

            if done:
                # Mark failure if the episode ended without task success.
                # Uses both native success and coverage threshold (0.90).
                proportion_wiped = float(next_obs[-1])
                native_success = (
                    step_info.get("success", False)
                    or step_info.get("task_completion", False)
                    or bool(self._rs_env._check_success())
                )
                coverage_success = (
                    proportion_wiped >= PROPORTION_WIPED_SUCCESS_THRESHOLD
                )
                failure = not (native_success or coverage_success)
                break

        # Decide failure by reward threshold when episode did not terminate
        if not failure and total_reward < self._failure_reward_threshold():
            failure = True

        self._restore_sim_state(saved_state)

        return traj, {"failure": failure, "total_reward": total_reward}

    def _failure_reward_threshold(self) -> float:
        """
        Overridden by subclasses per task.
        Base value of 0.0 is a safe conservative default:
        any trajectory with negative cumulative reward is flagged as failure.
        """
        return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_obs_dim(self) -> int:
        raw_obs = self._rs_env.reset()
        flat = _flatten_obs(raw_obs, self._obs_keys)
        return flat.shape[0]

    def get_takeover_cost(self, info: Dict) -> float:
        """
        Cosine-similarity-based takeover cost, identical to MetaDrive version.
        Falls back to a flat cost of 1 when actions are near-zero.
        """
        if self.agent_action is None:
            return 1.0
        takeover_action = np.clip(np.asarray(info["raw_action"]), -1, 1)
        agent_action    = np.clip(np.asarray(self.agent_action), -1, 1)
        denom = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if denom < 1e-6:
            return 1.0
        cos_sim = float(np.dot(takeover_action, agent_action) / denom)
        return 1.0 - cos_sim
