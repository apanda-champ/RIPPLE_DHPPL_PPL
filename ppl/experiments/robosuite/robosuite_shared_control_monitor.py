"""
SharedControlMonitor for Robosuite PPL.

Stores per-step shared-control data to disk in pickle format.
Adapted from the original PPL SharedControlMonitor to:
  • Work with either 4-tuple (obs, rew, done, info) or 5-tuple
    (obs, rew, terminated, truncated, info) gym returns.
  • Use 'raw_action' from the info dict (set by RobosuiteExpertTakeoverEnv)
    as the behaviour action actually applied to the environment.
  • Record 'intervention' as a boolean column so downstream analysis is
    consistent with the MetaDrive dataset format.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

logger = logging.getLogger(__name__)


class SharedControlMonitor(gym.Wrapper):
    """
    Records shared-control trajectories for PPL post-hoc analysis.

    Columns written to each pickle file
    ------------------------------------
    observation       : (T, obs_dim)  observation *before* the action
    action_agent      : (T, act_dim)  novice action (from the learning agent)
    action_behavior   : (T, act_dim)  action actually applied (expert or agent)
    reward            : (T,)
    cost              : (T,)          takeover cost
    terminated        : (T,)          episode-end flags
    truncated         : (T,)
    intervention      : (T,)          1 if expert took over, else 0
    episode_count     : (T,)
    step_count        : (T,)
    """

    def __init__(
        self,
        env: gym.Env,
        folder: str = "recorded_data",
        prefix: str = "data",
        save_freq: int = 1000,
    ):
        super().__init__(env)
        self.folder = folder
        self.prefix = prefix
        self.save_freq = save_freq

        self.step_count = 0
        self.last_save_step = 0
        self.episode_count = 0
        self.last_observation: Optional[np.ndarray] = None

        self._data: Dict[str, list] = self._empty_data()

        os.makedirs(self.folder, exist_ok=True)

    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> np.ndarray:
        # Support both 1-value (old gym) and 2-value (new gymnasium) reset
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        self.last_observation = obs
        return obs   # Always return obs-only for SB3 compatibility

    def step(self, action: np.ndarray):
        result = self.env.step(action)

        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            observation, reward, done, info = result
            terminated = done
            truncated = info.get("TimeLimit.truncated", False)

        self._record_step(
            obs=self.last_observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self._check_and_save()
        self.last_observation = observation

        if done:
            self.episode_count += 1

        # Return 4-tuple (SB3 expects this)
        return observation, reward, done, info

    # ------------------------------------------------------------------

    def _record_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict,
    ) -> None:
        self._data["observation"].append(np.asarray(obs, dtype=np.float32))
        self._data["reward"].append(float(reward))
        self._data["terminated"].append(bool(terminated))
        self._data["truncated"].append(bool(truncated))
        self._data["cost"].append(float(info.get("cost", info.get("takeover_cost", 0.0))))
        self._data["episode_count"].append(self.episode_count)
        self._data["step_count"].append(self.step_count)

        # The novice (agent) action is the *input* to step()
        self._data["action_agent"].append(np.asarray(action, dtype=np.float32))

        # The behaviour action is what was *applied* (may differ during takeover)
        raw_action = info.get("raw_action", action)
        self._data["action_behavior"].append(np.asarray(raw_action, dtype=np.float32))

        # Intervention flag
        self._data["intervention"].append(1 if info.get("takeover", False) else 0)

        self.step_count += 1

    def _check_and_save(self) -> None:
        if self.step_count - self.last_save_step >= self.save_freq:
            self._save_data()

    def _save_data(self) -> None:
        n = self.step_count - self.last_save_step
        if n == 0:
            return

        save_data: Dict[str, np.ndarray] = {}
        for key, vals in self._data.items():
            if not vals:
                continue
            arr = np.array(vals)
            save_data[key] = arr

        fname = f"{self.prefix}_step_{self.last_save_step}_{self.step_count}.pkl"
        fpath = os.path.join(self.folder, fname)
        with open(fpath, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"[SharedControlMonitor] Saved {n} steps "
            f"(steps {self.last_save_step}–{self.step_count}) → {fpath}"
        )

        self.last_save_step = self.step_count
        self._data = self._empty_data()

    @staticmethod
    def _empty_data() -> Dict[str, list]:
        return {
            "observation":     [],
            "action_agent":    [],
            "action_behavior": [],
            "reward":          [],
            "cost":            [],
            "terminated":      [],
            "truncated":       [],
            "intervention":    [],
            "episode_count":   [],
            "step_count":      [],
        }
