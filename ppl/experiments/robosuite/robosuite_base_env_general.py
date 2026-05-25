"""
Base Robosuite gym-compatible environment for PPL (Generalized).
Optimized for 100% coverage with non-linear rewards and step penalties.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config

# ---------------------------------------------------------------------------
# Default environment configuration
# ---------------------------------------------------------------------------

DEFAULT_ENV_CFG: Dict[str, Any] = {
    "env_name": "Wipe",
    "robots": "Panda",
    "controller_configs": load_controller_config(default_controller="OSC_POSE"),
    "horizon": 1000,
    "ignore_done": False,
    "reward_shaping": True,
    "use_object_obs": True,
    "use_camera_obs": False,
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "render_camera": "frontview",
    "control_freq": 20,
    "task_config": {
        "arm_limit_collision_penalty": -10.0,
        "wipe_contact_reward": 0.01,
        "unit_wiped_reward": 0.0,      # Disabled native reward to use our custom one
        "ee_accel_penalty": 0,
        "excess_force_penalty_mul": 0.05,
        "distance_multiplier": 5.0,
        "distance_th_multiplier": 5.0,
        "table_full_size": [0.5, 0.8, 0.05],
        "table_offset": [0.15, 0, 0.9],
        "table_friction": [0.03, 0.005, 0.0001],
        "table_friction_std": 0,
        "table_height": 0.0,
        "table_height_std": 0.0,
        "line_width": 0.04,
        "two_clusters": False,
        "coverage_factor": 1.0,        # 100% coverage required
        "num_markers": 100,
        "contact_threshold": 1.0,
        "pressure_threshold": 0.5,
        "pressure_threshold_max": 60.0,
        "print_results": False,
        "get_info": False,
        "use_robot_obs": True,
        "use_contact_obs": True,
        "early_terminations": False,   # Force full search
        "use_condensed_obj_obs": True,
    },
}

OBS_KEYS: List[str] = [
    "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel",
    "robot0_eef_pos", "robot0_eef_quat", "robot0_contact",
    "wipe_centroid", "wipe_radius", "gripper_to_wipe_centroid",
]

def _flatten_obs(obs_dict: Dict[str, np.ndarray], keys: List[str]) -> np.ndarray:
    return np.concatenate([np.asarray(obs_dict[k]).flatten() for k in keys])

class RobosuiteBaseEnv(gym.Env):
    total_steps: int = 0
    total_takeover_cost: float = 0.0
    total_takeover_count: int = 0
    total_cost: float = 0.0
    takeover: bool = False
    agent_action: Optional[np.ndarray] = None
    model = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = copy.deepcopy(DEFAULT_ENV_CFG)
        if config: cfg.update(config)
        self._obs_keys = cfg.pop("obs_keys", OBS_KEYS)
        self._rs_env = suite.make(**cfg)
        
        self.action_space = gym.spaces.Box(low=self._rs_env.action_spec[0], high=self._rs_env.action_spec[1], dtype=np.float32)
        obs_dim = self._compute_obs_dim()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.prev_wiped = 0
        self.takeover_recorder = deque(maxlen=2000)

    def reset(self, **kwargs) -> np.ndarray:
        raw_obs = self._rs_env.reset()
        self.prev_wiped = 0
        self.total_steps = 0
        return _flatten_obs(raw_obs, self._obs_keys).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.agent_action = copy.copy(action)
        raw_obs, reward, done, info = self._rs_env.step(action)
        
        # Calculate progress manually using the environment's internal marker list
        curr_wiped = len(self._rs_env.task.wiped) if hasattr(self._rs_env.task, 'wiped') else self.prev_wiped
        wiped_this_step = max(0, curr_wiped - self.prev_wiped)
        
        # Non-linear Reward: Worth more as we approach 100%
        progress = curr_wiped / 100.0
        non_linear_reward = 100.0 * (1.0 + (progress ** 2))
        reward = (wiped_this_step * non_linear_reward)
        
        # Step Penalty: Discourage wandering
        if wiped_this_step == 0:
            reward -= 0.05
            
        task_success = self._rs_env._check_success()
        if task_success and self.prev_wiped < 100:
            reward += 100.0
            
        self.prev_wiped = curr_wiped
        self.total_steps += 1
        
        info["is_success"] = bool(task_success)
        return _flatten_obs(raw_obs, self._obs_keys).astype(np.float32), float(reward), done, info

    def _compute_obs_dim(self) -> int:
        return _flatten_obs(self._rs_env.reset(), self._obs_keys).shape[0]

    # ... [Keep your existing _save_sim_state, _restore_sim_state, and predict_agent_future_trajectory methods here] ...
