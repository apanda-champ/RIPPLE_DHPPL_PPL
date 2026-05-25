"""
EnsembleDAggerEnv — MetaDrive environment for Ensemble DAgger training
=======================================================================

Differences from ExpertTakeoverEnv (used by PPL)
-------------------------------------------------
1.  **Takeover criterion**: Instead of predicting future trajectory failure,
    the expert is queried whenever the *ensemble disagreement* (std of
    actions across K actor networks) exceeds `uncertainty_threshold`.
    This is cheaper to compute (K forward passes vs. H environment
    rollouts) and does not require the BasePredictionEnv trajectory-
    simulation machinery.

2.  **No preference pairs**: EnsembleDAgger uses pure behaviour cloning,
    so there is no preference_buffer and no store_preference_pairs call.

3.  **Expert called every step**: The expert action is always computed so
    that it is ready if a takeover is needed.  This is the same pattern
    as ExpertTakeoverEnv.

4.  **Graceful warm-up**: Before the model is attached (or before the
    ensemble is initialised), `decide_takeover` returns True so that the
    expert always acts.  This bootstraps the human_data_buffer.

Shared with ExpertTakeoverEnv
-----------------------------
* Same PPO expert loaded from metadrive_ppo_expert_20m_steps.zip.
* Same DrivingEnv base (reward shaping, takeover cost, rendering text).
* Same info-dict keys (``takeover``, ``takeover_start``, etc.) so all
  downstream logging / callbacks work unchanged.
"""

import copy
import math

import gymnasium as gym
import numpy as np
import torch
from metadrive.engine.logger import get_logger

from ppl.experiments.metadrive.driving_env import DrivingEnv
from metadrive.policy.env_input_policy import EnvInputPolicy

logger = get_logger()


# ---------------------------------------------------------------------------
# Reuse the exact same expert loader from experttakeover_env.py
# ---------------------------------------------------------------------------

_expert = None  # module-level cache — loaded once on first env construction


def _get_expert():
    """Load the PPO expert from the bundled checkpoint (cached globally)."""
    global _expert
    if _expert is not None:
        return _expert

    import pathlib
    from ppl.experiments.metadrive.driving_env import DrivingEnv
    from ppl.sb3.common.save_util import load_from_zip_file
    from ppl.sb3.ppo import PPO
    from ppl.sb3.ppo.policies import ActorCriticPolicy

    FOLDER_PATH = pathlib.Path(__file__).parent

    train_env = DrivingEnv(config={"manual_control": False, "use_render": False})
    algo_config = dict(
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
        verbose=2,
        device="auto",
        env=train_env,
    )
    model = PPO(**algo_config)
    ckpt = FOLDER_PATH / "metadrive_ppo_expert_20m_steps.zip"
    print(f"[EnsembleDAggerEnv] Loading expert from {ckpt}")
    data, params, _ = load_from_zip_file(
        ckpt, device=model.device, print_system_info=False
    )
    model.set_parameters(params, exact_match=True, device=model.device)
    print("[EnsembleDAggerEnv] Expert loaded.")
    train_env.close()

    _expert = model.policy
    return _expert


# Pre-load at import time (mirrors experttakeover_env.py behaviour).
_expert = _get_expert()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class EnsembleDAggerEnv(DrivingEnv):
    """
    MetaDrive driving environment with uncertainty-based expert takeover
    for Ensemble DAgger training.

    The model must be attached after construction:

        env.model = ensemble_dagger_model

    If the model is not yet attached, or if the ensemble is not yet
    initialised, ``decide_takeover`` returns True (expert always acts),
    which bootstraps the human_data_buffer with high-quality initial data.
    """

    last_takeover = None
    last_obs = None
    expert = None

    def default_config(self):
        config = super().default_config()
        config.update(
            {
                # Uncertainty threshold for expert intervention.
                # Lower → expert intervenes more; higher → agent acts more.
                "uncertainty_threshold": 0.05,
                # Noise added to expert action (0 = deterministic expert).
                "expert_noise": 0.0,
                # Keep the same default flags as ExpertTakeoverEnv.
                "disable_expert": False,
                "agent_policy": EnvInputPolicy,
                "manual_control": False,
                "use_render": False,
            }
        )
        return config

    # ------------------------------------------------------------------
    # Takeover decision
    # ------------------------------------------------------------------

    def decide_takeover(self, obs: np.ndarray) -> bool:
        """
        Return True if the ensemble disagreement on *obs* exceeds the
        configured threshold.

        Falls back to True (always take over) when:
          * ``self.model`` has not been set yet, or
          * the model does not expose ``get_uncertainty`` (wrong class).
        """
        if not hasattr(self, "model") or self.model is None:
            return True

        if not hasattr(self.model, "get_uncertainty"):
            logger.warning(
                "[EnsembleDAggerEnv] model has no get_uncertainty method "
                "— defaulting to expert takeover."
            )
            return True

        # Ensemble has not been set up yet (e.g. before _setup_model).
        if self.model._ensemble_actors is None:
            return True

        _mean_action, std = self.model.get_uncertainty(obs)
        # std shape: (1,) — a scalar uncertainty value.
        uncertainty_value = float(std.mean())
        return uncertainty_value > self.config["uncertainty_threshold"]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions):
        """
        Step the environment.

        * Always compute the expert action (needed if takeover occurs).
        * Decide takeover via ensemble disagreement.
        * If takeover: replace the novice action with the expert action.
        * Delegate to SafeMetaDriveEnv.step (skipping DrivingEnv.step,
          same pattern as ExpertTakeoverEnv).
        """
        actions = np.asarray(actions).astype(np.float32)
        self.agent_action = copy.copy(actions)
        self.last_takeover = self.takeover  # carry over for info dict

        # ---- Load expert (once) ----
        if self.expert is None:
            global _expert
            self.expert = _expert

        # ---- Expert action ----
        expert_action, _ = self.expert.predict(
            self.last_obs, deterministic=True
        )
        expert_noise_bound = self.config["expert_noise"]
        if expert_noise_bound > 0:
            noise = np.random.randn(*expert_action.shape) * expert_noise_bound
            expert_action = np.clip(
                expert_action + noise,
                self.action_space.low,
                self.action_space.high,
            )

        # ---- Log-prob of novice action under expert distribution ----
        # (kept for compatibility with downstream logging that reads
        #  info["takeover_log_prob"])
        last_obs_tensor, _ = self.expert.obs_to_tensor(self.last_obs)
        distribution = self.expert.get_distribution(last_obs_tensor)
        log_prob = distribution.log_prob(
            torch.from_numpy(actions).to(last_obs_tensor.device)
        )

        # ---- Decide takeover via ensemble uncertainty ----
        self.takeover = self.decide_takeover(self.last_obs)

        if self.takeover:
            actions = expert_action

        # ---- Advance the simulation ----
        # Call super(DrivingEnv, self).step so that DrivingEnv._get_step_return
        # is still invoked (it reads self.takeover / self.last_takeover to
        # populate the info dict and cost accounting).
        o, r, d, i = super(DrivingEnv, self).step(actions)

        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1

        if not self.config["disable_expert"]:
            i["takeover_log_prob"] = log_prob.item()

        if self.config["use_render"]:
            self.render(
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover": "EXPERT" if self.takeover else "AGENT",
                    "Total Step": self.total_steps,
                    "Takeover Rate": "{:.1f}%".format(
                        np.mean(np.array(self.takeover_recorder) * 100)
                    ),
                }
            )

        assert i["takeover"] == self.takeover
        return o, r, d, i

    # ------------------------------------------------------------------
    # Reset helpers (mirror ExpertTakeoverEnv)
    # ------------------------------------------------------------------

    def _get_step_return(self, actions, engine_info):
        o, r, tm, tc, engine_info = super(DrivingEnv, self)._get_step_return(
            actions, engine_info
        )
        self.last_obs = o
        d = tm or tc
        last_t = self.last_takeover
        engine_info["takeover_start"] = (
            True if not last_t and self.takeover else False
        )
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
        engine_info["native_cost"] = engine_info["cost"]
        engine_info["episode_native_cost"] = self.episode_cost
        self.total_cost += engine_info["cost"]
        self.total_takeover_count += 1 if self.takeover else 0
        engine_info["total_takeover_count"] = self.total_takeover_count
        engine_info["total_cost"] = self.total_cost
        return o, r, d, engine_info

    def _get_reset_return(self, reset_info):
        o, info = super(DrivingEnv, self)._get_reset_return(reset_info)
        self.last_obs = o
        self.last_takeover = False
        return o, info


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = EnsembleDAggerEnv(
        config={"use_render": True, "num_scenarios": 1, "traffic_density": 0}
    )
    env.reset()
    for _ in range(200):
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()
    env.close()
