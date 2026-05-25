"""
TrajPrefReplayBuffer
====================
Circular replay buffer for trajectory-level preference pairs used by RIPPLE's
L_traj loss term.

Each entry is a (good_traj, bad_traj) pair where both sides are variable-length
sequences of (obs, action) steps.  Sequences are zero-padded to ``max_traj_len``
at insertion time and a binary mask is stored alongside so the loss function can
ignore padding positions.

Interface expected by RIPPLE
-----------------------------
  buf.add(good_traj, bad_traj)
      good_traj / bad_traj : List[Dict]
          Each dict must contain at least:
              "obs"    : np.ndarray  (obs_dim,)
              "action" : np.ndarray  (act_dim,)
          Extra keys (reward, next_obs, done, info) are silently ignored.
          Sequences longer than max_traj_len are truncated.

  data = buf.sample(batch_size, env=None)
      Returns a TrajSamples named-tuple with torch.Tensor fields:
          good_observations  (B, T, obs_dim)
          good_actions       (B, T, act_dim)
          good_mask          (B, T)   – 1.0 for valid steps, 0.0 for padding
          bad_observations   (B, T, obs_dim)
          bad_actions        (B, T, act_dim)
          bad_mask           (B, T)
      Returns None when the buffer is empty (RIPPLE.train() guards against this).

  buf.pos  : int   – number of pairs added so far (capped at buffer_size)

Place at:
    ppl/sb3/haco/traj_buffer.py
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

import gym
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TrajSamples(NamedTuple):
    good_observations: torch.Tensor   # (B, T, obs_dim)
    good_actions:      torch.Tensor   # (B, T, act_dim)
    good_mask:         torch.Tensor   # (B, T)
    bad_observations:  torch.Tensor   # (B, T, obs_dim)
    bad_actions:       torch.Tensor   # (B, T, act_dim)
    bad_mask:          torch.Tensor   # (B, T)


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

class TrajPrefReplayBuffer:
    """
    Fixed-size circular buffer storing padded trajectory preference pairs.

    Parameters
    ----------
    buffer_size       : int    – maximum number of (good, bad) pairs stored
    observation_space : gym.Space
    action_space      : gym.Space
    device            : torch.device | str
    n_envs            : int    – unused; kept for API consistency with SB3 buffers
    max_traj_len      : int    – all sequences are padded / truncated to this length
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        device,
        n_envs: int = 1,
        max_traj_len: int = 10,
    ):
        self.buffer_size  = int(buffer_size)
        self.max_traj_len = int(max_traj_len)
        self.device       = device

        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))

        T = self.max_traj_len
        N = self.buffer_size

        # Pre-allocate storage — zero-initialised (padding value = 0)
        self._good_obs  = np.zeros((N, T, obs_dim), dtype=np.float32)
        self._good_act  = np.zeros((N, T, act_dim), dtype=np.float32)
        self._good_mask = np.zeros((N, T),           dtype=np.float32)
        self._bad_obs   = np.zeros((N, T, obs_dim), dtype=np.float32)
        self._bad_act   = np.zeros((N, T, act_dim), dtype=np.float32)
        self._bad_mask  = np.zeros((N, T),           dtype=np.float32)

        # pos counts total insertions; index into array = pos % buffer_size
        self.pos  = 0
        self.full = False

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(
        self,
        good_traj: List[Dict],
        bad_traj:  List[Dict],
    ) -> None:
        """
        Insert one (good_traj, bad_traj) pair.

        Parameters
        ----------
        good_traj : List[Dict]  – expert (preferred) trajectory
        bad_traj  : List[Dict]  – agent (dispreferred) trajectory

        Each dict must have "obs" and "action" keys.
        Sequences are truncated to max_traj_len before storage.
        """
        idx = self.pos % self.buffer_size

        self._write_traj(good_traj, self._good_obs, self._good_act, self._good_mask, idx)
        self._write_traj(bad_traj,  self._bad_obs,  self._bad_act,  self._bad_mask,  idx)

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def _write_traj(
        self,
        traj:  List[Dict],
        obs_buf:  np.ndarray,
        act_buf:  np.ndarray,
        mask_buf: np.ndarray,
        idx:   int,
    ) -> None:
        """Zero-pad and write a single trajectory into row ``idx``."""
        T = self.max_traj_len

        # Reset this row (clears stale data from a previous overwrite)
        obs_buf[idx]  = 0.0
        act_buf[idx]  = 0.0
        mask_buf[idx] = 0.0

        steps = traj[:T]   # truncate silently if longer than max_traj_len
        for t, step in enumerate(steps):
            obs_buf[idx, t]  = np.asarray(step["obs"],    dtype=np.float32).flatten()
            act_buf[idx, t]  = np.asarray(step["action"], dtype=np.float32).flatten()
            mask_buf[idx, t] = 1.0

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        env=None,          # unused; kept for API consistency with SB3 buffers
    ) -> Optional[TrajSamples]:
        """
        Sample a batch of trajectory preference pairs uniformly at random.

        Returns None when the buffer is empty so RIPPLE.train() can skip
        the trajectory loss term gracefully.
        """
        max_idx = self.buffer_size if self.full else self.pos
        if max_idx == 0:
            return None

        size    = min(batch_size, max_idx)
        indices = np.random.randint(0, max_idx, size=size)

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr[indices], dtype=torch.float32, device=self.device)

        return TrajSamples(
            good_observations = _t(self._good_obs),
            good_actions      = _t(self._good_act),
            good_mask         = _t(self._good_mask),
            bad_observations  = _t(self._bad_obs),
            bad_actions       = _t(self._bad_act),
            bad_mask          = _t(self._bad_mask),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def __repr__(self) -> str:
        return (
            f"TrajPrefReplayBuffer("
            f"pos={self.pos}, "
            f"full={self.full}, "
            f"buffer_size={self.buffer_size}, "
            f"max_traj_len={self.max_traj_len})"
        )
