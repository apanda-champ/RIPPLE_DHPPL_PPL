"""
ppl_wandb_callback.py
=====================
W&B callback for PPL training on Robosuite.

Logged metrics (all under the ``ppl/`` namespace)
--------------------------------------------------
human_steps_total       – cumulative expert-takeover steps (from wrapper)
human_steps_this_ep     – expert-takeover steps in the most recent episode
human_data_buffer_size  – entries currently in model.human_data_buffer
pref_buffer_size        – entries currently in model.preference_buffer
expert_data_ratio       – human_data_buffer / total_steps  (0–1)
pref_pairs_per_step     – preference pairs collected per env step so far

rollout/success_rate    – fraction of recent episodes that succeeded
                          (sourced from model.ep_success_buffer; only logged
                          when at least one episode has finished)
rollout/ep_rew_mean     – mean episode return over the recent buffer
rollout/ep_len_mean     – mean episode length over the recent buffer

train/bc_loss           – behaviour-cloning loss (from model logger)
train/cpl_loss          – DPO/CPL preference loss
train/cpl_accuracy      – preference prediction accuracy

Usage
-----
Instantiate and pass to model.learn() via the ``callback`` argument:

    from ppl_wandb_callback import PPLWandbCallback
    import wandb

    wandb.init(project="my_project", name="ppl_panda_wipe",
               sync_tensorboard=True)

    cb = PPLWandbCallback(
        env_wrapper=ppl_wrapper,   # the RobosuitePPLWrapper instance
        log_interval=500,          # log every 500 env steps
        verbose=1,
    )
    model.learn(total_timesteps=50_000, callback=cb)
"""

from collections import deque
from typing import Optional

import numpy as np
import wandb

from ppl.sb3.common.callbacks import BaseCallback


class PPLWandbCallback(BaseCallback):
    """
    SB3 callback that pushes PPL-specific metrics to Weights & Biases.

    Parameters
    ----------
    env_wrapper : RobosuitePPLWrapper
        The outermost wrapper in the training env stack.  Used to read
        ``_human_steps`` and ``_human_steps_ep``.
    log_interval : int
        Log metrics every this many environment steps.  Default 500.
    verbose : int
        0 = silent, 1 = print a one-liner each time metrics are logged.
    """

    def __init__(
        self,
        env_wrapper,
        log_interval: int = 500,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        self.env_wrapper  = env_wrapper
        self.log_interval = log_interval

        # Local buffers for metrics the SB3 logger already computes but
        # that we want to push to W&B explicitly.
        self._last_human_steps = 0

    # ── BaseCallback interface ────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        """Log all hyperparameters from the PPL model config."""
        config_to_log = {}

        # Scalar attributes on the model
        for key, val in self.model.__dict__.items():
            if isinstance(val, (int, float, str, bool)):
                config_to_log[key] = val

        # PPL-specific extra_config dict (bc_loss_weight, beta, flags …)
        if hasattr(self.model, "extra_config"):
            for key, val in self.model.extra_config.items():
                config_to_log[f"ppl/{key}"] = val

        wandb.config.update(config_to_log, allow_val_change=True)

        if self.verbose:
            print(f"[PPLWandbCallback] Hyperparameters logged to W&B.")

    def _on_step(self) -> bool:
        """Called after every env step.  Log metrics every log_interval steps."""
        if self.num_timesteps % self.log_interval != 0:
            return True

        metrics = self._collect_metrics()
        wandb.log(metrics, step=self.num_timesteps)

        if self.verbose:
            sr   = metrics.get("rollout/success_rate", float("nan"))
            hs   = metrics.get("ppl/human_steps_total", 0)
            ratio = metrics.get("ppl/expert_data_ratio", float("nan"))
            print(
                f"[PPLWandbCallback] step={self.num_timesteps:>7d} | "
                f"human_steps={hs:>6d} | expert_ratio={ratio:.3f} | "
                f"success_rate={sr:.3f}"
            )

        return True

    def _on_training_end(self) -> None:
        """Final log at the end of training."""
        metrics = self._collect_metrics()
        wandb.log(metrics, step=self.num_timesteps)
        if self.verbose:
            print("[PPLWandbCallback] Training ended — final metrics logged.")

    # ── Metric collection ─────────────────────────────────────────────────────

    def _collect_metrics(self) -> dict:
        metrics = {}
        T = max(self.num_timesteps, 1)

        # ── Human / expert data usage ────────────────────────────────────────
        human_total = self.env_wrapper._human_steps
        human_ep    = self.env_wrapper._human_steps_ep

        metrics["ppl/human_steps_total"]    = human_total
        metrics["ppl/human_steps_this_ep"]  = human_ep
        metrics["ppl/expert_data_ratio"]    = human_total / T

        # Steps collected since last log
        delta = human_total - self._last_human_steps
        metrics["ppl/human_steps_delta"] = delta
        self._last_human_steps = human_total

        # ── Buffer sizes ─────────────────────────────────────────────────────
        hdb_size  = self.model.human_data_buffer.pos
        pref_size = self.model.preference_buffer.pos if hasattr(self.model, "preference_buffer") else 0

        metrics["ppl/human_data_buffer_size"] = hdb_size
        metrics["ppl/pref_buffer_size"]       = pref_size
        metrics["ppl/pref_pairs_per_step"]    = pref_size / T

        # If the buffer has wrapped around, use actual capacity
        if self.model.human_data_buffer.full:
            metrics["ppl/human_data_buffer_size"] = self.model.human_data_buffer.buffer_size

        # ── Success rate ─────────────────────────────────────────────────────
        # model.ep_success_buffer is a deque(maxlen=100) filled by
        # _update_info_buffer() whenever info["is_success"] is set on a done step.
        ep_success_buf = getattr(self.model, "ep_success_buffer", None)
        if ep_success_buf is not None and len(ep_success_buf) > 0:
            metrics["rollout/success_rate"] = float(np.mean(list(ep_success_buf)))

        # ── Episode return / length ───────────────────────────────────────────
        ep_info_buf = getattr(self.model, "ep_info_buffer", None)
        if ep_info_buf is not None and len(ep_info_buf) > 0:
            metrics["rollout/ep_rew_mean"] = float(
                np.mean([ep["r"] for ep in ep_info_buf])
            )
            metrics["rollout/ep_len_mean"] = float(
                np.mean([ep["l"] for ep in ep_info_buf])
            )

        # ── Training losses (from SB3 logger cache) ──────────────────────────
        # SB3's logger stores the latest value for each key in logger.name_to_value
        logger_vals = getattr(self.model.logger, "name_to_value", {})

        for src_key, dst_key in [
            ("train/bc_loss",       "train/bc_loss"),
            ("train/cpl_loss",      "train/cpl_loss"),
            ("train/cpl_accuracy",  "train/cpl_accuracy"),
            ("train/loss",          "train/total_loss"),
            ("train/n_updates",     "train/n_updates"),
        ]:
            if src_key in logger_vals:
                metrics[dst_key] = logger_vals[src_key]

        return metrics