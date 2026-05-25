"""
weighted_pref_buffer.py

Extends PrefReplayBuffer with a per-sample scalar weight that represents
the strength of the expert intervention that generated each preference pair.

The weight is defined as:
    w = -log_prob(a_novice | s_expert_distribution)
      = 0.5 * sum((a_novice - mu_expert(s))^2 / sigma^2) + constant

Since sigma is a fixed parameter for the PPO DiagGaussianDistribution expert,
the varying part is 0.5 * ||a_novice - mu_expert(s)||^2 / sigma^2, which is
the squared Mahalanobis distance between the novice action and the expert mean.

A large weight (large -log_prob) means:
  - The novice action is far from the expert mean
  - The intervention is strong and unambiguous
  - The preference label is reliable
  - Longer L is justified and the sample deserves more weight in the CPO loss

A small weight (small -log_prob) means:
  - The novice action is close to the expert mean
  - The intervention is marginal
  - The preference label is less reliable
  - Shorter L is safer and the sample should contribute less to the loss

Weights are stored raw (unnormalized) and normalized to [0,1] at sample time
using the running max observed in the buffer, to keep loss scales stable.

NOTE on slot safety
-------------------
PrefReplayBuffer.add() writes to self.pos and THEN increments it (post-write).
So capturing current_slot = self.pos before calling super().add() is safe:
we write the weight to the same slot that the base class just wrote the pair to.
"""

import numpy as np
import torch as th
from typing import List, Optional, Union
from gym import spaces

from ppl.sb3.haco.haco_buffer import PrefReplayBuffer, PrefReplayBufferSamples
from ppl.sb3.common.vec_env import VecNormalize
from typing import NamedTuple


class WeightedPrefReplayBufferSamples(NamedTuple):
    pos_observations: th.Tensor
    pos_actions: th.Tensor
    neg_observations: th.Tensor
    neg_actions: th.Tensor
    weights: th.Tensor          # normalized intervention strength, shape [B]


class WeightedPrefReplayBuffer(PrefReplayBuffer):
    """
    PrefReplayBuffer extended with per-sample intervention-strength weights.

    The only public API change is that add() accepts an extra keyword argument
    `intervention_strength` (float, >=0).  Everything else is unchanged.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            handle_timeout_termination=handle_timeout_termination,
        )
        # One scalar weight per stored pair, initialised to 1.0
        self.intervention_strengths = np.ones(
            (self.buffer_size,), dtype=np.float32
        )
        # Running max for normalisation (avoids division-by-zero)
        self._max_strength_seen = 1.0

    # ------------------------------------------------------------------
    def add(
        self,
        pos_traj: List,
        neg_traj: List,
        intervention_strength: float = 1.0,
    ) -> None:
        """
        Same as parent, plus stores intervention_strength at the current slot.

        PrefReplayBuffer.add() writes to self.pos then increments it, so
        capturing current_slot = self.pos BEFORE super().add() is correct.

        Parameters
        ----------
        pos_traj, neg_traj : same as PrefReplayBuffer.add
        intervention_strength : float
            -log_prob(a_novice | expert_dist) at the state where the
            intervention occurred.  Must be >= 0.
        """
        # Capture write pointer BEFORE parent increments it.
        # Safe because PrefReplayBuffer.add() is post-increment (writes then pos += 1).
        current_slot = self.pos
        super().add(pos_traj, neg_traj)

        # Store raw strength
        strength = float(max(intervention_strength, 0.0))
        self.intervention_strengths[current_slot] = strength
        if strength > self._max_strength_seen:
            self._max_strength_seen = strength

    # ------------------------------------------------------------------
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> WeightedPrefReplayBufferSamples:
        base = super()._get_samples(batch_inds, env=env)

        raw = self.intervention_strengths[batch_inds]          # shape [B]
        # Normalise to (0, 1] using running max so loss scale is stable.
        # Add small epsilon so zero-strength samples still contribute.
        normalised = (raw + 1e-6) / (self._max_strength_seen + 1e-6)
        weights = th.tensor(normalised, dtype=th.float32, device=self.device)

        return WeightedPrefReplayBufferSamples(
            pos_observations=base.pos_observations,
            pos_actions=base.pos_actions,
            neg_observations=base.neg_observations,
            neg_actions=base.neg_actions,
            weights=weights,
        )
