"""
OSC Expert for Robosuite – PPL equivalent of the MetaDrive PPO expert.

The 'osc_expert' is a Stable-Baselines 3 PPO policy that was pre-trained on
the Robosuite Lift task with a Panda robot and an OSC_POSE controller.

Usage
-----
1. Train the expert first:
       python train_osc_expert.py --total_timesteps 500000 --save_path osc_expert.zip

2. Load it in the takeover env:
       policy = get_osc_expert("osc_expert.zip")

The returned object behaves exactly like an SB3 policy:
    action, _   = policy.predict(obs, deterministic=True)
    log_prob     = policy.get_log_prob(obs_tensor, action_tensor)
    distribution = policy.get_distribution(obs_tensor)
"""

from __future__ import annotations

import pathlib
from typing import Optional, Tuple

import numpy as np
import torch

# PPL re-uses the SB3 fork bundled in the repo
from ppl.sb3.ppo import PPO
from ppl.sb3.ppo.policies import ActorCriticPolicy
from ppl.sb3.common.save_util import load_from_zip_file

from ppl.experiments.robosuite.robosuite_base_env import (
    RobosuiteBaseEnv,
    DEFAULT_ENV_CFG,
)

# ---------------------------------------------------------------------------
# Module-level singleton – loaded once and shared across all env instances
# (mirrors the MetaDrive pattern: `_expert = get_expert()`)
# ---------------------------------------------------------------------------

_osc_expert: Optional["OscExpertPolicy"] = None


# ---------------------------------------------------------------------------
# Thin wrapper that mimics the SB3 ActorCriticPolicy interface
# ---------------------------------------------------------------------------

class OscExpertPolicy:
    """
    Wraps a loaded SB3 PPO policy so that ExpertTakeoverEnv can call:
        policy.predict(obs, deterministic=True)
        policy.get_distribution(obs_tensor)
        policy.obs_to_tensor(obs)
    """

    def __init__(self, inner_policy: ActorCriticPolicy):
        self._policy = inner_policy
        self._policy.set_training_mode(False)

    # ---- forward-delegate everything ----

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        return self._policy.predict(obs, deterministic=deterministic)

    def obs_to_tensor(self, obs: np.ndarray):
        return self._policy.obs_to_tensor(obs)

    def get_distribution(self, obs_tensor: torch.Tensor):
        return self._policy.get_distribution(obs_tensor)

    def get_log_prob(
        self,
        obs_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
    ) -> torch.Tensor:
        distribution = self.get_distribution(obs_tensor)
        return distribution.log_prob(action_tensor)

    # Expose the device of the underlying model
    @property
    def device(self):
        return next(self._policy.parameters()).device


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_osc_expert(checkpoint_path: Optional[str] = None) -> OscExpertPolicy:
    """
    Load (or return the cached) OSC expert.

    Parameters
    ----------
    checkpoint_path : str, optional
        Path to a ``.zip`` file saved by ``PPO.save()`` or the training
        script ``train_osc_expert.py``.
        When *None* the function looks for ``osc_expert.zip`` in the same
        directory as this file.

    Returns
    -------
    OscExpertPolicy
        A ready-to-use expert wrapper.
    """
    global _osc_expert
    if _osc_expert is not None:
        return _osc_expert

    if checkpoint_path is None:
        checkpoint_path = pathlib.Path(__file__).parent / "osc_expert.zip"
    checkpoint_path = pathlib.Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"OSC expert checkpoint not found at '{checkpoint_path}'.\n"
            "Train it first with:\n"
            "    python train_osc_expert.py --total_timesteps 500000 "
            "--save_path osc_expert.zip"
        )

    # Build a temporary env just to get the spaces
    tmp_env = RobosuiteBaseEnv()
    algo_cfg = dict(
        policy=ActorCriticPolicy,
        env=tmp_env,
        n_steps=1024,
        n_epochs=10,
        learning_rate=3e-4,
        batch_size=256,
        verbose=0,
        device="auto",
        policy_kwargs=dict(net_arch=[dict(pi=[512, 512], vf=[512, 512])]),
    )
    model = PPO(**algo_cfg)
    tmp_env.close()

    print(f"[OscExpert] Loading checkpoint from '{checkpoint_path}' …")
    _data, params, _pv = load_from_zip_file(
        str(checkpoint_path), device=model.device, print_system_info=False
    )
    model.set_parameters(params, exact_match=True, device=model.device)
    print("[OscExpert] Expert loaded successfully.")

    _osc_expert = OscExpertPolicy(model.policy)
    return _osc_expert


# ---------------------------------------------------------------------------
# Convenience: reset the singleton (useful for tests)
# ---------------------------------------------------------------------------

def reset_osc_expert() -> None:
    global _osc_expert
    _osc_expert = None
