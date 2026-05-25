"""
robosuite_ppl_env.py
====================
Bridges a Robosuite GymWrapper environment with the PPL training pipeline.

Responsibilities
----------------
1.  Expert-takeover logic
    - At every ``failure_check_freq`` step, the wrapper predicts the agent's
      future trajectory by rolling out the current policy open-loop for
      ``num_predicted_steps`` steps.
    - If the predicted trajectory is "bad" (total predicted reward below
      ``intervention_threshold``), the expert takes over for this step.

2.  ``info`` dict fields required by HACOReplayBuffer / PVPTD3
    - ``takeover``        : bool  – expert is acting this step
    - ``takeover_start``  : bool  – first step of a new takeover episode
    - ``takeover_cost``   : float – cost incurred on takeover steps
    - ``raw_action``      : the action actually applied to the env

3.  Success signal for the SB3 pipeline
    - ``is_success``      : bool  – Robosuite's task-completion flag.
      Populated from ``info["success"]`` (the key GymWrapper always sets).
      The SB3 base class picks this up in ``_update_info_buffer`` and fills
      ``model.ep_success_buffer``, which feeds ``rollout/success_rate``.

4.  Human-step counter
    - ``_human_steps``    : int   – running total of expert-takeover steps.
      Read by PPLWandbCallback directly from the wrapper.

5.  Preference pair generation (for PPL's DPO loss)
    - On the first step of a new takeover, the wrapper rolls out the agent
      policy in a cloned environment, pairs positive (expert) and negative
      (predicted agent) segments, and pushes them to
      ``model.preference_buffer``.

Key design choices
------------------
* Robosuite v1.4.1 + GymWrapper always sets ``info["success"]`` at every
  step.  We copy it as ``info["is_success"]`` so SB3's standard success-rate
  pipeline works without modification.
* State save/restore for the lookahead uses ``copy.deepcopy`` because
  Robosuite v1.4 has no public set_state/get_state API.  If deepcopy fails
  (some MuJoCo bindings block it), the wrapper falls back to always
  intervening (conservative).
* ``_human_steps`` is incremented here rather than inside HACOReplayBuffer
  so the count is available to the callback even during the warm-up phase.
"""

import copy
from typing import Any, Dict, List, Optional

import gym
import numpy as np


class RobosuitePPLWrapper(gym.Wrapper):
    """
    Wraps a Robosuite env (already wrapped with GymWrapper + LegacyEnvAdapter
    + Monitor) and provides the full PPL intervention + preference-pair
    interface.

    Parameters
    ----------
    env : gym.Env
        The underlying env.
    expert_policy : stable_baselines3.PPO
        Loaded expert PPO model.  Used to:
          (a) decide takeover via predicted-reward check, and
          (b) supply positive actions for the preference buffer.
    config : dict
        Required keys:
            num_predicted_steps    (int)   – lookahead horizon H
            preference_horizon     (int)   – pref-pair window L per takeover
            intervention_threshold (float) – intervene when pred reward < this
        Optional keys:
            failure_check_freq     (int)   – re-evaluate every N steps (default 10)
            expert_noise           (float) – Gaussian noise std on expert action (default 0)
    """

    def __init__(self, env: gym.Env, expert_policy, config: Dict[str, Any]):
        super().__init__(env)

        self.expert_policy = expert_policy

        # ── Config ──────────────────────────────────────────────────────────
        self.num_predicted_steps    = int(config["num_predicted_steps"])
        self.preference_horizon     = int(config["preference_horizon"])
        self.intervention_threshold = float(config["intervention_threshold"])
        self.failure_check_freq     = int(config.get("failure_check_freq", 10))
        self.expert_noise           = float(config.get("expert_noise", 0.0))

        # ── Per-step state ────────────────────────────────────────────────
        self.takeover: bool               = False
        self._last_takeover: bool         = False
        self._last_obs: Optional[np.ndarray] = None
        self._total_steps: int            = 0

        # ── Cumulative human-step counters ────────────────────────────────
        # _human_steps    : total expert-takeover steps across all episodes
        # _human_steps_ep : expert-takeover steps in the current episode
        # _total_episodes : episodes completed so far
        self._human_steps: int            = 0
        self._human_steps_ep: int         = 0
        self._total_episodes: int         = 0

        # ── Intervention cost accumulators (reset each episode) ───────────
        self._total_takeover_cost: float  = 0.0
        self._takeover_count: int         = 0

        # Reference to the PPL model — wired in by train_ppl_robo.py
        # before model.learn() is called.
        self.model = None

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def human_steps(self) -> int:
        """Total expert-takeover steps collected across all episodes."""
        return self._human_steps

    @property
    def human_steps_this_episode(self) -> int:
        """Expert-takeover steps in the current episode only."""
        return self._human_steps_ep

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _expert_action(self, obs: np.ndarray) -> np.ndarray:
        """Query expert for a deterministic action, with optional noise."""
        action, _ = self.expert_policy.predict(obs, deterministic=True)
        if self.expert_noise > 0:
            noise  = np.random.randn(*action.shape) * self.expert_noise
            action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        return action.astype(np.float32)

    def _predict_future_reward(self, obs: np.ndarray) -> float:
        """
        Roll out the expert policy in a cloned env for ``num_predicted_steps``
        steps and return the cumulative reward.
        Falls back to -inf (always intervene) if deepcopy fails.
        """
        try:
            env_clone = copy.deepcopy(self.env)
        except Exception:
            return float("-inf")

        total_reward = 0.0
        current_obs  = obs.copy()
        for _ in range(self.num_predicted_steps):
            action, _ = self.expert_policy.predict(current_obs, deterministic=False)
            result = env_clone.step(action)
            current_obs, reward, done = result[0], result[1], result[2]
            total_reward += reward
            if done:
                break
        try:
            env_clone.close()
        except Exception:
            pass
        return total_reward

    def _predict_trajectory_for_pref(self, obs: np.ndarray) -> List[Dict[str, Any]]:
        """
        Roll out the expert policy in a cloned env and record (obs, action)
        pairs — these form the *negative* half of preference pairs.
        """
        try:
            env_clone = copy.deepcopy(self.env)
        except Exception:
            return []

        traj: List[Dict[str, Any]] = []
        current_obs = obs.copy()
        for _ in range(self.num_predicted_steps):
            action, _ = self.expert_policy.predict(current_obs, deterministic=False)
            result     = env_clone.step(action)
            next_obs, reward, done = result[0], result[1], result[2]
            traj.append({
                "obs":      current_obs.copy(),
                "action":   action.copy(),
                "reward":   reward,
                "next_obs": next_obs.copy(),
                "done":     done,
            })
            current_obs = next_obs.copy()
            if done:
                break
        try:
            env_clone.close()
        except Exception:
            pass
        return traj

    def _store_preference_pairs(
        self,
        predicted_traj: List[Dict[str, Any]],
        expert_action: np.ndarray,
        obs: np.ndarray,
    ) -> None:
        """
        Push (positive, negative) preference pairs into model.preference_buffer.

        Positive: (current obs, expert action)
        Negative: (predicted future obs, predicted agent action)
        """
        if self.model is None or not hasattr(self.model, "preference_buffer"):
            return
        if len(predicted_traj) < 2:
            return

        horizon = min(len(predicted_traj) - 1, self.preference_horizon)
        for step_i in range(horizon):
            self.model.preference_buffer.add(
                [{"obs": obs.copy(), "action": expert_action.copy()}],
                [{"obs": predicted_traj[step_i]["obs"].copy(),
                  "action": predicted_traj[step_i]["action"].copy()}],
            )

    # ── gym.Wrapper interface ─────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]

        self._last_obs      = obs.copy()
        self.takeover       = False
        self._last_takeover = False

        # Reset per-episode accumulators
        self._human_steps_ep      = 0
        self._total_takeover_cost = 0.0
        self._takeover_count      = 0

        return obs

    def step(self, agent_action: np.ndarray):
        """
        1. Every ``failure_check_freq`` steps, predict future reward and set
           the takeover flag.
        2. On the first step of a new takeover, build preference pairs.
        3. Replace the agent action with the expert action if taking over.
        4. Step the underlying env.
        5. Stamp all required fields onto ``info``.
        """
        agent_action        = np.asarray(agent_action, dtype=np.float32)
        self._last_takeover = self.takeover

        # ── Decide takeover ──────────────────────────────────────────────────
        if self._total_steps % self.failure_check_freq == 0 and self._last_obs is not None:
            pred_reward   = self._predict_future_reward(self._last_obs)
            self.takeover = pred_reward < self.intervention_threshold

        # ── Expert override + preference pair generation ─────────────────────
        if self.takeover:
            expert_action  = self._expert_action(self._last_obs)
            takeover_start = not self._last_takeover
            if takeover_start and self._last_obs is not None:
                traj = self._predict_trajectory_for_pref(self._last_obs)
                self._store_preference_pairs(traj, expert_action, self._last_obs)
            behavior_action = expert_action
        else:
            behavior_action = agent_action

        # ── Env step ─────────────────────────────────────────────────────────
        obs, reward, done, info = self.env.step(behavior_action)

        # ── Update human-step counters ────────────────────────────────────────
        if self.takeover:
            self._human_steps    += 1
            self._human_steps_ep += 1
            self._takeover_count += 1

        takeover_cost              = float(self.takeover)
        self._total_takeover_cost += takeover_cost

        if done:
            self._total_episodes += 1

        # ── Stamp info fields required by HACOReplayBuffer ──────────────────
        takeover_start_flag = bool(self.takeover and not self._last_takeover)

        info["takeover"]             = bool(self.takeover)
        info["takeover_start"]       = takeover_start_flag
        info["takeover_cost"]        = takeover_cost
        info["total_takeover_cost"]  = self._total_takeover_cost
        info["total_takeover_count"] = self._takeover_count
        info["raw_action"]           = behavior_action.copy()

        # ── is_success: pass through Robosuite's task-completion flag ────────
        # GymWrapper sets info["success"] at every step from the underlying env.
        # We expose it as "is_success" so SB3's _update_info_buffer() can fill
        # model.ep_success_buffer → rollout/success_rate log.
        info["is_success"] = bool(info.get("success", False))

        self._last_obs    = obs.copy()
        self._total_steps += 1

        return obs, reward, done, info

    def seed(self, seed: Optional[int] = None):
        try:
            return self.env.seed(seed)
        except Exception:
            return []