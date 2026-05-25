"""
train_adaptive_ppl_metadrive.py

Training script for Adaptive PPL.

Drop-in replacement for train_ppl_metadrive.py.
All original files are untouched.

New CLI flags:
  --adaptive_L       True/False  (default True)
  --L_min            int         (default 1)
  --L_max            int         (default 8)

All original flags are preserved with identical defaults.
"""

import argparse
import os
import uuid
from collections import deque
from pathlib import Path

import numpy as np

from ppl.adaptive_ppl import AdaptivePPL
from ppl.experiments.metadrive.adaptive_experttakeover_env import (
    AdaptiveExpertTakeoverEnv,
)
from ppl.sb3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from ppl.sb3.common.monitor import Monitor
from ppl.sb3.common.vec_env import SubprocVecEnv
from ppl.sb3.haco import HACOReplayBuffer
from ppl.sb3.td3.policies import TD3Policy
from ppl.utils.shared_control_monitor import SharedControlMonitor

import pathlib
FOLDER_PATH = pathlib.Path(__file__).parent.parent.parent  # repo root

# FIX: sliding window size for WandB L histogram — shows recent distribution
# rather than the full run which makes early training dominate late plots
_WANDB_L_WINDOW = 500


# ══════════════════════════════════════════════════════════════════════════════
#  Callbacks
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveMetricsCallback(BaseCallback):
    """
    Reads intervention_strength, adaptive_L, sample_weight from info dicts at
    each env step and:
      - Forwards them to AdaptivePPL.log_env_stats() every `flush_freq` steps
        so they appear in TensorBoard / WandB.
      - Writes a CSV log at trial_dir/adaptive_L_log.csv for offline analysis
        (one row per intervention, i.e. when takeover=True).
    """

    def __init__(self, log_dir: str, flush_freq: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir    = Path(log_dir)
        self.flush_freq = flush_freq
        self._strength_buf:   list = []
        self._adaptive_L_buf: list = []
        self._weight_buf:     list = []
        self._step_count = 0

        # CSV: only write rows when a takeover happens to keep the file small
        self._csv_path = self.log_dir / "adaptive_L_log.csv"
        with open(self._csv_path, "w") as f:
            f.write(
                "global_step,intervention_strength,adaptive_L,sample_weight\n"
            )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            strength = info.get("intervention_strength", None)
            adapt_L  = info.get("adaptive_L",            None)
            weight   = info.get("sample_weight",          None)
            takeover = bool(info.get("takeover",          False))

            if strength is not None:
                self._strength_buf.append(float(strength))
            if adapt_L is not None:
                self._adaptive_L_buf.append(int(adapt_L))
            if weight is not None:
                self._weight_buf.append(float(weight))

            # Write CSV only on actual interventions
            if takeover and strength is not None:
                with open(self._csv_path, "a") as f:
                    f.write(
                        f"{self.num_timesteps},"
                        f"{strength:.6f},"
                        f"{adapt_L},"
                        f"{weight:.6f}\n"
                    )

        self._step_count += 1
        if self._step_count % self.flush_freq == 0:
            self._flush()

        return True

    def _flush(self):
        if self._strength_buf and hasattr(self.model, "log_env_stats"):
            self.model.log_env_stats(
                self._strength_buf[:],
                self._adaptive_L_buf[:],
            )
        if self._weight_buf and hasattr(self.model, "logger"):
            self.model.logger.record(
                "env/sample_weight_mean",
                float(np.mean(self._weight_buf))
            )
        self._strength_buf.clear()
        self._adaptive_L_buf.clear()
        self._weight_buf.clear()

    def _on_training_end(self) -> None:
        self._flush()


class AdaptiveWandbCallback(BaseCallback):
    """
    WandB callback that logs all SB3 metrics AND produces a WandB Histogram
    of adaptive_L values at each `log_freq` steps.

    FIX: uses a sliding window deque of size _WANDB_L_WINDOW for the histogram
    so late-training plots show recent L distribution rather than being
    dominated by the full accumulated history from early training.
    """

    def __init__(
        self,
        trial_name: str,
        exp_name: str,
        team_name: str,
        project_name: str,
        config: dict,
        log_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        import wandb
        self.run = wandb.init(
            project=project_name or "adaptive_ppl",
            entity=team_name or None,
            name=trial_name,
            group=exp_name,
            config=config,
            sync_tensorboard=True,
        )
        self.log_freq = log_freq
        # FIX: bounded deque instead of ever-growing list
        self._recent_L_values: deque = deque(maxlen=_WANDB_L_WINDOW)
        self._recent_intervention_L: deque = deque(maxlen=_WANDB_L_WINDOW)

    def _on_step(self) -> bool:
        import wandb

        infos = self.locals.get("infos", [])
        for info in infos:
            adapt_L  = info.get("adaptive_L", None)
            takeover = bool(info.get("takeover", False))
            if adapt_L is not None:
                self._recent_L_values.append(int(adapt_L))
                if takeover:
                    self._recent_intervention_L.append(int(adapt_L))

        if self.num_timesteps % self.log_freq == 0 and self._recent_L_values:
            recent = list(self._recent_L_values)
            log_dict = {
                "adaptive_L/recent_histogram": wandb.Histogram(recent),
                "adaptive_L/recent_mean": float(np.mean(recent)),
                "adaptive_L/recent_std":  float(np.std(recent)),
                "global_step": self.num_timesteps,
            }
            if self._recent_intervention_L:
                intervention = list(self._recent_intervention_L)
                log_dict["adaptive_L/intervention_histogram"] = (
                    wandb.Histogram(intervention)
                )
                log_dict["adaptive_L/intervention_mean"] = float(
                    np.mean(intervention)
                )
            wandb.log(log_dict, step=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        import wandb
        if self._recent_L_values:
            wandb.log(
                {
                    "adaptive_L/final_histogram": wandb.Histogram(
                        list(self._recent_L_values)
                    ),
                    "global_step": self.num_timesteps,
                }
            )
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ── original flags (identical defaults) ────────────────────────────
    parser.add_argument("--exp_name",          default="adaptive_ppl_metadrive", type=str)
    parser.add_argument("--batch_size",        default=1024, type=int)
    parser.add_argument("--save_freq",         default=150,  type=int)
    parser.add_argument("--seed",              default=0,    type=int)
    parser.add_argument("--wandb",             action="store_true")
    parser.add_argument("--wandb_project",     type=str,  default="")
    parser.add_argument("--wandb_team",        type=str,  default="")
    parser.add_argument("--only_bc_loss",      default="False", type=str)
    parser.add_argument("--ckpt",              default="",   type=str)
    parser.add_argument("--num_predicted_steps", default=10, type=int)
    parser.add_argument("--preference_horizon",  default=4,  type=int,
                        help="Fixed fallback L used when adaptive_L=False.")
    parser.add_argument("--toy_env",           action="store_true")
    parser.add_argument("--bc_loss_weight",    type=float, default=1.0)
    parser.add_argument("--beta",              default=0.1, type=float)

    # ── new adaptive-L flags ────────────────────────────────────────────
    parser.add_argument(
        "--adaptive_L", default="True", type=str,
        help="Enable intervention-strength-based adaptive preference horizon."
    )
    parser.add_argument(
        "--L_min", default=1, type=int,
        help="Minimum preference horizon (used when adaptive_L=True)."
    )
    parser.add_argument(
        "--L_max", default=8, type=int,
        help="Maximum preference horizon (used when adaptive_L=True)."
    )

    args = parser.parse_args()

    # ── naming ──────────────────────────────────────────────────────────
    experiment_batch_name = "AdaptivePPL"
    if args.only_bc_loss == "True":
        experiment_batch_name = "AdaptivePPL_BCLossOnly"

    seed       = args.seed
    trial_name = f"{experiment_batch_name}_{uuid.uuid4().hex[:8]}"
    print("Trial name:", trial_name)

    use_wandb    = args.wandb
    project_name = args.wandb_project
    team_name    = args.wandb_team
    if not use_wandb:
        print("[WARNING] WandB disabled. Pass --wandb to enable it.")

    log_dir        = FOLDER_PATH
    experiment_dir = Path(log_dir) / "runs" / experiment_batch_name
    trial_dir      = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir,      exist_ok=False)
    print(f"Logging to {trial_dir}")

    # ── config ──────────────────────────────────────────────────────────
    config = dict(
        env_config=dict(
            num_predicted_steps = args.num_predicted_steps,
            preference_horizon  = args.preference_horizon,
            adaptive_L          = (args.adaptive_L == "True"),
            L_min               = args.L_min,
            L_max               = args.L_max,
        ),
        algo=dict(
            only_bc_loss         = args.only_bc_loss,
            bc_loss_weight       = args.bc_loss_weight,
            beta                 = args.beta,
            add_bc_loss          = "True" if args.bc_loss_weight > 0.0 else "False",
            use_balance_sample   = True,
            agent_data_ratio     = 1.0,
            policy               = TD3Policy,
            replay_buffer_class  = HACOReplayBuffer,
            replay_buffer_kwargs = dict(),
            policy_kwargs        = dict(net_arch=[256, 256]),
            env                  = None,
            learning_rate        = 1e-4,
            q_value_bound        = 1,
            optimize_memory_usage= True,
            buffer_size          = 50_000,
            learning_starts      = 10,
            batch_size           = args.batch_size,
            tau                  = 0.005,
            gamma                = 0.99,
            train_freq           = (1, "step"),
            action_noise         = None,
            tensorboard_log      = trial_dir,
            create_eval_env      = False,
            verbose              = 2,
            seed                 = seed,
            device               = "auto",
        ),
        exp_name   = experiment_batch_name,
        seed       = seed,
        use_wandb  = use_wandb,
        trial_name = trial_name,
        log_dir    = str(trial_dir),
        adaptive_L_config=dict(
            adaptive_L = (args.adaptive_L == "True"),
            L_min      = args.L_min,
            L_max      = args.L_max,
        ),
    )

    if args.toy_env:
        config["env_config"].update(
            num_scenarios   = 1,
            traffic_density = 0.0,
            map             = "COT",
            use_render      = True,
        )

    # ── training env ────────────────────────────────────────────────────
    train_env = AdaptiveExpertTakeoverEnv(config=config["env_config"])
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    train_env = SharedControlMonitor(
        env=train_env,
        folder=trial_dir / "data",
        prefix=trial_name,
    )
    config["algo"]["env"] = train_env

    # ── eval env ────────────────────────────────────────────────────────
    def _make_eval_env():
        from ppl.experiments.metadrive.driving_env import DrivingEnv
        from ppl.sb3.common.monitor import Monitor as _Monitor
        eval_env = DrivingEnv(config={"start_seed": 1000})
        return _Monitor(env=eval_env, filename=str(trial_dir))

    eval_env, eval_freq = SubprocVecEnv([_make_eval_env]), 150

    # ── callbacks ───────────────────────────────────────────────────────
    callbacks = [
        CheckpointCallback(
            name_prefix = "rl_model",
            verbose     = 2,
            save_freq   = args.save_freq,
            save_path   = str(trial_dir / "models"),
        ),
        AdaptiveMetricsCallback(
            log_dir    = str(trial_dir),
            flush_freq = 50,
            verbose    = 1,
        ),
    ]
    if use_wandb:
        callbacks.append(
            AdaptiveWandbCallback(
                trial_name   = trial_name,
                exp_name     = experiment_batch_name,
                team_name    = team_name,
                project_name = project_name,
                config       = config,
                log_freq     = 100,
            )
        )
    callbacks = CallbackList(callbacks)

    # ── model ────────────────────────────────────────────────────────────
    model = AdaptivePPL(**config["algo"])

    if args.ckpt:
        from ppl.sb3.common.save_util import load_from_zip_file
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}")
        data, params, _ = load_from_zip_file(
            ckpt, device=model.device, print_system_info=False
        )
        model.set_parameters(params, exact_match=True, device=model.device)

    # Expose model to env so store_preference_pairs can write to its buffer
    train_env.env.env.model = model

    # ── train ────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = 10_000,
        callback            = callbacks,
        reset_num_timesteps = True,
        eval_env            = eval_env,
        eval_freq           = eval_freq,
        n_eval_episodes     = 50,
        eval_log_path       = str(trial_dir),
        tb_log_name         = experiment_batch_name,
        log_interval        = 1,
        save_buffer         = False,
    )
