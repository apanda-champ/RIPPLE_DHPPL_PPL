"""
train_ripple_robosuite.py  –  RIPPLE training on Robosuite Wipe (Panda + OSC_POSE).
=====================================================================================
Mirrors train_ripple_metadrive.py with Robosuite-specific components substituted in.

Prerequisites
-------------
1. Pre-train the OSC expert (one-time, ~5 M steps):
       python train_osc_expert.py --total_timesteps 5000000 \\
           --save_path experiments/robosuite/osc_expert.zip

2. Run RIPPLE (all four terms active by default):
       python train_ripple_robosuite.py \\
           --expert_checkpoint experiments/robosuite/osc_expert.zip \\
           --total_timesteps 20000 --seed 0

3. BC-only ablation (disables all preference terms):
       python train_ripple_robosuite.py --only_bc_loss True

4. Enable W&B:
       python train_ripple_robosuite.py --wandb \\
           --wandb_project my_project --wandb_team my_team

WandB / TensorBoard metrics
----------------------------
eval/success_rate
    Fraction of eval episodes where proportion_wiped >= 1.0 OR native
    _check_success() fires.  Evaluated every eval_freq steps over 20 episodes.

train/expert_steps_total
    Cumulative env steps where the OSC expert intervened.
    Logged every step by ExpertStepsCallback.

buffers/fwd_pref    – forward preference buffer fill level
buffers/back_pref   – backward preference buffer fill level
buffers/silent_pref – silent-approval buffer fill level
buffers/traj_pref   – trajectory-level buffer fill level
buffers/human       – human (intervention) data buffer fill level

train/bc_loss, train/fwd_loss, train/back_loss,
train/silent_loss, train/traj_loss, train/total
    Per-term and total losses averaged over the training window.

train/fwd_acc, train/back_acc, train/silent_acc, train/traj_acc
    Pairwise preference accuracy for each active loss term.

hparams/beta_fwd, hparams/beta_back,
hparams/beta_silent, hparams/beta_traj
    Per-term DPO temperature values (logged once per train call for
    reference).

Key differences from train_ppl_robosuite.py
--------------------------------------------
• Imports RIPPLE from ppl.ripple instead of PPL from ppl.ppl.
• Imports RobosuiteRIPPLEEnv instead of RobosuiteExpertTakeoverEnv.
• Adds eight RIPPLE-specific CLI flags (four loss weights, four betas).
• Passes backward_horizon / silent_noise_scale / silent_margin / traj_max_len
  to both env_config (read by RobosuiteRIPPLEEnv) and algo_cfg (popped by
  RIPPLE.__init__).
• set_all_seeds() is called three times for full reproducibility (mirrors
  train_ripple_metadrive.py).
• Experiment name is RIPPLE_Robosuite for run-level disambiguation.
"""

from __future__ import annotations

import argparse
import os
import random
import uuid
from pathlib import Path

# Must be set BEFORE numpy/torch imports for full determinism
os.environ["PYTHONHASHSEED"]          = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

# RIPPLE algorithm
from ppl.ripple import RIPPLE

# SB3 utilities
from ppl.sb3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from ppl.sb3.common.monitor import Monitor
from ppl.sb3.common.vec_env import DummyVecEnv
from ppl.sb3.common.wandb_callback import WandbCallback
from ppl.sb3.haco import HACOReplayBuffer
from ppl.sb3.td3.policies import TD3Policy

# Robosuite-specific components
from ppl.experiments.robosuite.robosuite_base_env import RobosuiteBaseEnv
from ppl.experiments.robosuite.robosuite_ripple_env import RobosuiteRIPPLEEnv
from ppl.experiments.robosuite.robosuite_shared_control_monitor import (
    SharedControlMonitor,
)


# ---------------------------------------------------------------------------
# Determinism helper
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# ExpertStepsCallback
# ---------------------------------------------------------------------------

class ExpertStepsCallback(BaseCallback):
    """
    Accumulates a global expert-intervention step counter across all episodes
    and records it as ``train/expert_steps_total`` every training step.
    env.total_takeover_count resets each episode; this counter never does.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._expert_steps_total: int = 0

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        if info.get("takeover", False):
            self._expert_steps_total += 1
        self.logger.record("train/expert_steps_total", self._expert_steps_total)
        return True


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RIPPLE training on Robosuite Wipe.")

    # Experiment
    p.add_argument("--exp_name",         default="ripple_robosuite", type=str)
    p.add_argument("--seed",             default=0,          type=int)
    p.add_argument("--total_timesteps",  default=20_000,     type=int)
    p.add_argument("--log_dir",          default="",         type=str,
                   help="Root for run logs (default: ./runs/).")

    # Training schedule
    p.add_argument("--batch_size",       default=1024,       type=int)
    p.add_argument("--save_freq",        default=150,        type=int,
                   help="Checkpoint every N steps; also used as eval_freq.")
    p.add_argument("--learning_starts",  default=10,         type=int)
    p.add_argument("--buffer_size",      default=50_000,     type=int)

    # Base PPL hyper-parameters
    p.add_argument("--only_bc_loss",     default="False",     type=str,
                   choices=["True", "False"],
                   help="True disables all preference terms (BC-only ablation).")
    p.add_argument("--bc_loss_weight",   default=1.0,        type=float)
    p.add_argument("--beta",             default=0.1,        type=float,
                   help="Shared DPO temperature β (fallback if per-term beta unset).")

    # Trajectory prediction / takeover
    p.add_argument("--num_predicted_steps", default=20,      type=int)
    p.add_argument("--failure_check_freq",  default=10,      type=int)
    p.add_argument("--preference_horizon",  default=4,       type=int,
                   help="Max forward/silent preference pairs per takeover.")

    # RIPPLE loss weights
    p.add_argument("--lambda_fwd",       type=float, default=1.0,
                   help="L_pref^fwd weight (forward preferences, base PPL term).")
    p.add_argument("--lambda_back",      type=float, default=1.0,
                   help="L_pref^back weight (backward propagation).")
    p.add_argument("--lambda_silent",    type=float, default=0.0,
                   help="L_silent weight (silent-approval preferences).")
    p.add_argument("--lambda_traj",      type=float, default=1.0,
                   help="L_traj weight (trajectory-level preferences).")

    # RIPPLE per-term betas
    p.add_argument("--beta_fwd",         type=float, default=0.1,
                   help="DPO temperature for forward preferences.")
    p.add_argument("--beta_back",        type=float, default=0.1,
                   help="DPO temperature for backward preferences.")
    p.add_argument("--beta_silent",      type=float, default=0.05,
                   help="DPO temperature for silent-approval (softer).")
    p.add_argument("--beta_traj",        type=float, default=0.1,
                   help="DPO temperature for trajectory-level preferences.")

    # RIPPLE data-collection hyperparameters
    p.add_argument("--backward_horizon",    type=int,   default=4,
                   help="Number of pre-intervention steps to retroactively label.")
    p.add_argument("--silent_noise_scale",  type=float, default=0.3,
                   help="Std of Gaussian noise for silent-approval negatives.")
    p.add_argument("--silent_margin",       type=float, default=0.15,
                   help="Min Euclidean margin between silent pos/neg actions.")
    p.add_argument("--traj_max_len",        type=int,   default=10,
                   help="Max trajectory pair length before committing to buffer.")

    # Expert
    p.add_argument("--expert_checkpoint", default="",   type=str,
                   help="Path to osc_expert.zip.  Empty = auto-discover.")
    p.add_argument("--expert_noise",      default=0.0,  type=float)
    p.add_argument("--disable_expert",    action="store_true")

    # Checkpoint resume
    p.add_argument("--ckpt",             default="",    type=str,
                   help="Resume a RIPPLE model from a .zip checkpoint.")

    # Logging
    p.add_argument("--wandb",            action="store_true")
    p.add_argument("--wandb_project",    default="",    type=str)
    p.add_argument("--wandb_team",       default="",    type=str)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    set_all_seeds(args.seed)

    # ---- Experiment naming --------------------------------------------------
    experiment_batch_name = "RIPPLE_Robosuite"
    if args.only_bc_loss == "True":
        experiment_batch_name += "_BCOnly"

    # Use explicit exp_name when provided; otherwise generate a UUID suffix
    trial_name = (
        args.exp_name
        if args.exp_name != "ripple_robosuite"
        else f"{experiment_batch_name}_{uuid.uuid4().hex[:8]}"
    )
    print(f"[RIPPLE] Trial name: {trial_name}")

    if not args.wandb:
        print("[WARNING] W&B logging is disabled.  Pass --wandb to enable.")

    root_log       = Path(args.log_dir) if args.log_dir else Path("runs")
    experiment_dir = root_log / experiment_batch_name
    trial_dir      = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir,      exist_ok=False)
    print(f"[RIPPLE] Logging to: {trial_dir}")

    # ---- Environment config -------------------------------------------------
    env_cfg: dict = dict(
        # PPL takeover settings
        num_predicted_steps = args.num_predicted_steps,
        failure_check_freq  = args.failure_check_freq,
        preference_horizon  = args.preference_horizon,
        expert_noise        = args.expert_noise,
        disable_expert      = args.disable_expert,
        # Robosuite task settings
        reward_shaping         = True,
        has_renderer           = False,
        has_offscreen_renderer = False,
        # RIPPLE data-collection settings
        backward_horizon   = args.backward_horizon,
        silent_noise_scale = args.silent_noise_scale,
        silent_margin      = args.silent_margin,
        traj_max_len       = args.traj_max_len,
    )
    if args.expert_checkpoint:
        env_cfg["expert_checkpoint"] = args.expert_checkpoint

    # ---- Training environment -----------------------------------------------
    # Stack: RobosuiteRIPPLEEnv → Monitor → SharedControlMonitor
    set_all_seeds(args.seed)
    train_env = RobosuiteRIPPLEEnv(config=env_cfg)
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    train_env = SharedControlMonitor(
        env=train_env,
        folder=str(trial_dir / "data"),
        prefix=trial_name,
        save_freq=1000,
    )

    # ---- Eval environment ---------------------------------------------------
    # Uses plain RobosuiteBaseEnv (no expert) to measure pure agent performance.
    # EvalCallback reads info["is_success"] at done=True:
    #   True iff _check_success() fires OR proportion_wiped >= 1.0.
    def _make_eval_env():
        eval_env = RobosuiteBaseEnv(
            config={"reward_shaping": True, "has_renderer": False}
        )
        return Monitor(env=eval_env)

    eval_env  = DummyVecEnv([_make_eval_env])
    eval_freq = args.save_freq   # evaluate every save_freq training steps

    # ---- RIPPLE algorithm config --------------------------------------------
    algo_cfg = dict(
        # Core SB3 / TD3 settings
        policy               = TD3Policy,
        env                  = train_env,
        replay_buffer_class  = HACOReplayBuffer,
        replay_buffer_kwargs = {},
        policy_kwargs        = dict(net_arch=[256, 256]),
        learning_rate        = 1e-4,
        buffer_size          = args.buffer_size,
        learning_starts      = args.learning_starts,
        batch_size           = args.batch_size,
        tau                  = 0.005,
        gamma                = 0.99,
        train_freq           = (1, "step"),
        action_noise         = None,
        tensorboard_log      = str(trial_dir),
        create_eval_env      = False,
        verbose              = 2,
        seed                 = args.seed,
        device               = "auto",

        # Base PPL settings
        only_bc_loss          = args.only_bc_loss,
        bc_loss_weight        = float(args.bc_loss_weight),
        beta                  = args.beta,
        add_bc_loss           = "True" if args.bc_loss_weight > 0.0 else "False",
        use_balance_sample    = True,
        agent_data_ratio      = 1.0,
        q_value_bound         = 1.0,
        optimize_memory_usage = True,

        # PVPTD3 extra-config flags (mirrors MetaDrive defaults)
        no_done_for_positive       = "False",
        no_done_for_negative       = "False",
        reward_0_for_positive      = "False",
        reward_0_for_negative      = "False",
        reward_n2_for_intervention = "False",
        reward_1_for_all           = "False",
        use_weighted_reward        = "False",
        remove_negative            = "False",
        adaptive_batch_size        = "False",
        with_human_proxy_value_loss= "False",
        with_agent_proxy_value_loss= "False",
        simple_batch               = "True",

        # RIPPLE loss weights
        lambda_fwd    = args.lambda_fwd,
        lambda_back   = args.lambda_back,
        lambda_silent = args.lambda_silent,
        lambda_traj   = args.lambda_traj,

        # RIPPLE per-term betas
        beta_fwd    = args.beta_fwd,
        beta_back   = args.beta_back,
        beta_silent = args.beta_silent,
        beta_traj   = args.beta_traj,

        # RIPPLE data-collection hyperparameters
        # (RIPPLE.__init__ pops these; the env also reads them from env_cfg)
        backward_horizon   = args.backward_horizon,
        silent_noise_scale = args.silent_noise_scale,
        silent_margin      = args.silent_margin,
        traj_max_len       = args.traj_max_len,
    )

    # ---- Callbacks ----------------------------------------------------------
    callbacks = [
        CheckpointCallback(
            name_prefix = "ripple_robosuite",
            verbose     = 2,
            save_freq   = args.save_freq,
            save_path   = str(trial_dir / "models"),
        ),
        ExpertStepsCallback(verbose=1),
    ]

    if args.wandb:
        config_for_wandb = {
            "env_config":  env_cfg,
            "algo":        {k: v for k, v in algo_cfg.items()
                            if not callable(v) and k != "env"},
            "exp_name":    experiment_batch_name,
            "seed":        args.seed,
            "trial_name":  trial_name,
        }
        callbacks.append(
            WandbCallback(
                trial_name   = trial_name,
                exp_name     = experiment_batch_name,
                team_name    = args.wandb_team,
                project_name = args.wandb_project,
                config       = config_for_wandb,
            )
        )

    callbacks = CallbackList(callbacks)

    # ---- Instantiate RIPPLE model -------------------------------------------
    set_all_seeds(args.seed)
    model = RIPPLE(**algo_cfg)

    # Resume from checkpoint if requested
    if args.ckpt:
        from ppl.sb3.common.save_util import load_from_zip_file
        ckpt = Path(args.ckpt)
        print(f"[RIPPLE] Resuming from checkpoint: {ckpt}")
        _data, params, _pv = load_from_zip_file(
            str(ckpt), device=model.device, print_system_info=False
        )
        model.set_parameters(params, exact_match=True, device=model.device)

    # Attach model to the inner env so the RIPPLE env can:
    #   • Write to model.preference_buffer  (forward)
    #   • Write to model.back_pref_buffer   (backward)
    #   • Write to model.silent_pref_buffer (silent)
    #   • Write to model.traj_pref_buffer   (trajectory)
    #   • Use model.policy for trajectory prediction
    # Walk the wrapper chain: SharedControlMonitor → Monitor → RobosuiteRIPPLEEnv
    inner_env = train_env.env.env   # type: RobosuiteRIPPLEEnv
    inner_env.model = model

    # ---- Launch training ----------------------------------------------------
    print(f"\n[RIPPLE] Starting training: {args.total_timesteps:,} timesteps")
    print(f"[RIPPLE] Eval every {eval_freq} steps over 20 episodes.")
    print(f"[RIPPLE] Success: proportion_wiped >= 1.0 OR native _check_success().")
    print(f"[RIPPLE] Loss weights: fwd={args.lambda_fwd} back={args.lambda_back} "
          f"silent={args.lambda_silent} traj={args.lambda_traj}")
    print(f"[RIPPLE] Per-term betas: fwd={args.beta_fwd} back={args.beta_back} "
          f"silent={args.beta_silent} traj={args.beta_traj}\n")

    model.learn(
        total_timesteps     = args.total_timesteps,
        callback            = callbacks,
        reset_num_timesteps = True,
        # Evaluation
        eval_env            = eval_env,
        eval_freq           = eval_freq,
        n_eval_episodes     = 20,
        eval_log_path       = str(trial_dir),
        # Logging
        tb_log_name         = experiment_batch_name,
        log_interval        = 1,
        # Buffer persistence
        save_buffer         = False,
    )

    print(f"\n[RIPPLE] Training complete.  Artefacts in: {trial_dir}")
    train_env.close()
    eval_env.close()
