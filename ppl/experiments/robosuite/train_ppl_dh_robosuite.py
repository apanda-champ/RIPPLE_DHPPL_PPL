"""
train_ppl_dh_robosuite.py  –  DH-PPL training on Robosuite Wipe (Panda + OSC_POSE).
======================================================================================
Mirrors train_ppl_robosuite.py but uses:
  • ppl_dh.PPL          — the DH-PPL algorithm with ensemble uncertainty model,
                           dynamic horizon gate, and latent barrier loss.
  • RobosuiteExpertTakeoverEnvDH
                         — takeover env that calls model.should_add_to_preference_buffer
                           before storing each preference pair.

Prerequisites
-------------
1. Pre-train the OSC expert (one-time, ~5 M steps):
       python train_osc_expert.py --total_timesteps 5000000 \
           --save_path experiments/robosuite/osc_expert.zip

2. Run DH-PPL:
       python train_ppl_dh_robosuite.py \
           --expert_checkpoint experiments/robosuite/osc_expert.zip \
           --total_timesteps 20000 \
           --seed 0

3. (Optional) Enable W&B:
       python train_ppl_dh_robosuite.py --wandb \
           --wandb_project my_project --wandb_team my_team

WandB metrics logged
--------------------
eval/success_rate
    Fraction of eval episodes where proportion_wiped >= 1.0 OR native
    _check_success() fires.  Evaluated every eval_freq steps over
    n_eval_episodes=20 episodes.  Flows via sync_tensorboard=True.

train/expert_steps_total
    Cumulative env steps where the OSC expert intervened.
    Logged every step by ExpertStepsCallback.

train/pref_buffer_accept_rate
    Fraction of candidate preference pairs admitted by the DH gate in
    the most recent training window.  Logged by DH-PPL's train().

train/uncertainty_threshold_L
    Current rolling-mean uncertainty threshold used by the DH gate.

train/latent_barrier_loss
    Mean latent barrier penalty across the training window.

train/dpo_loss
    Mean DPO preference loss across the training window.

train/bc_loss
    Mean behaviour-cloning loss across the training window.

pref_buffer/*
    Per-takeover admission stats logged directly to W&B by the env
    (admitted, rejected, accept_rate, uncertainty_{mean,max,min},
     buffer_size, threshold_L).  These are env-side logs and therefore
     appear at the takeover step rather than the training step.

Key differences from train_ppl_robosuite.py
--------------------------------------------
• Imports PPL from ppl_dh instead of ppl.
• Imports RobosuiteExpertTakeoverEnvDH instead of RobosuiteExpertTakeoverEnv.
• Removes the ``add_bc_loss`` flag from algo_cfg — DH-PPL's train() reads
  bc_loss_weight directly and does not use the PVPTD3 extra_config flag
  system for this (which was a bug in the original robosuite training script).
• Experiment name is PPL_DH_Robosuite for run-level disambiguation.
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

# DH-PPL algorithm
from ppl.DH_PPL import PPL

# SB3 utilities
from ppl.sb3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from ppl.sb3.common.monitor import Monitor
from ppl.sb3.common.vec_env import DummyVecEnv
from ppl.sb3.common.wandb_callback import WandbCallback
from ppl.sb3.haco import HACOReplayBuffer
from ppl.sb3.td3.policies import TD3Policy

# Robosuite-specific components
from ppl.experiments.robosuite.robosuite_base_env import RobosuiteBaseEnv
from ppl.experiments.robosuite.robosuite_expert_takeover_env_dh import (
    RobosuiteExpertTakeoverEnvDH,
)
from ppl.experiments.robosuite.robosuite_shared_control_monitor import (
    SharedControlMonitor,
)


# ---------------------------------------------------------------------------
# ExpertStepsCallback
# (Identical to the one in train_ppl_robosuite.py — reproduced here so
#  this script is self-contained.)
# ---------------------------------------------------------------------------

class ExpertStepsCallback(BaseCallback):
    """
    Accumulates a global expert-intervention step counter across all episodes
    and records it as ``train/expert_steps_total`` via the SB3 logger every
    training step.  WandbCallback mirrors this via sync_tensorboard=True.

    env.total_takeover_count resets every episode, so we maintain our own
    monotonically increasing counter here.
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
    p = argparse.ArgumentParser(description="DH-PPL training on Robosuite Wipe.")

    # Experiment
    p.add_argument("--exp_name",          default="ppl_dh_robosuite", type=str)
    p.add_argument("--seed",              default=0,         type=int)
    p.add_argument("--total_timesteps",   default=20_000,    type=int)
    p.add_argument("--log_dir",           default="",        type=str,
                   help="Root for run logs (default: ./runs/).")

    # Training schedule
    p.add_argument("--batch_size",        default=1024,      type=int)
    p.add_argument("--save_freq",         default=150,       type=int,
                   help="Checkpoint every N agent steps; also used as eval_freq.")
    p.add_argument("--learning_starts",   default=10,        type=int,
                   help="Steps of random exploration before training begins.")
    p.add_argument("--buffer_size",       default=50_000,    type=int)

    # DH-PPL hyper-parameters
    p.add_argument("--only_bc_loss",      default="False",   type=str,
                   choices=["True", "False"])
    p.add_argument("--bc_loss_weight",    default=1.0,       type=float,
                   help="Weight on the BC loss term.")
    p.add_argument("--beta",              default=0.1,       type=float,
                   help="DPO temperature β.")

    # Trajectory prediction / takeover
    p.add_argument("--num_predicted_steps", default=20,      type=int,
                   help="Prediction horizon H for failure detection.")
    p.add_argument("--failure_check_freq",  default=10,      type=int,
                   help="Run the trajectory predictor every N steps.")
    p.add_argument("--preference_horizon",  default=3,       type=int,
                   help="Max preference pairs L stored per takeover.")

    # Expert
    p.add_argument("--expert_checkpoint", default="",        type=str,
                   help="Path to osc_expert.zip.  Empty = auto-discover.")
    p.add_argument("--expert_noise",      default=0.0,       type=float,
                   help="Std of Gaussian noise added to expert actions.")
    p.add_argument("--disable_expert",    action="store_true",
                   help="Bypass the expert (ablation).")

    # Checkpoint resume
    p.add_argument("--ckpt",              default="",        type=str,
                   help="Resume a DH-PPL model from a .zip checkpoint.")

    # Logging
    p.add_argument("--wandb",             action="store_true")
    p.add_argument("--wandb_project",     default="",        type=str)
    p.add_argument("--wandb_team",        default="",        type=str)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # ---- Experiment naming ----
    experiment_batch_name = "PPL_DH_Robosuite"
    if args.only_bc_loss == "True":
        experiment_batch_name += "_BCOnly"
    trial_name = f"{experiment_batch_name}_{uuid.uuid4().hex[:8]}"
    print(f"[DH-PPL] Trial name: {trial_name}")

    if not args.wandb:
        print("[WARNING] W&B logging is disabled.  Pass --wandb to enable.")

    root_log       = Path(args.log_dir) if args.log_dir else Path("runs")
    experiment_dir = root_log / experiment_batch_name
    trial_dir      = experiment_dir / trial_name
    os.makedirs(trial_dir, exist_ok=True)
    print(f"[DH-PPL] Logging to: {trial_dir}")

    # ---- Environment config ----
    env_cfg: dict = dict(
        num_predicted_steps = args.num_predicted_steps,
        failure_check_freq  = args.failure_check_freq,
        preference_horizon  = args.preference_horizon,
        expert_noise        = args.expert_noise,
        disable_expert      = args.disable_expert,
        reward_shaping      = True,
        has_renderer        = False,
        has_offscreen_renderer = False,
    )
    if args.expert_checkpoint:
        env_cfg["expert_checkpoint"] = args.expert_checkpoint

    # ---- Training environment ----
    # Stack: RobosuiteExpertTakeoverEnvDH → Monitor → SharedControlMonitor
    train_env = RobosuiteExpertTakeoverEnvDH(config=env_cfg)
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    train_env = SharedControlMonitor(
        env=train_env,
        folder=str(trial_dir / "data"),
        prefix=trial_name,
        save_freq=1000,
    )

    # ---- Eval environment ----
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

    # ---- DH-PPL algorithm config ----
    # Note: add_bc_loss is intentionally absent.
    # DH-PPL's train() reads bc_loss_weight directly from extra_config and
    # includes the BC term whenever bc_loss_weight > 0 — there is no separate
    # add_bc_loss flag in the DH-PPL training loop.
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

        # PPL / DH-PPL specific
        only_bc_loss     = args.only_bc_loss,
        bc_loss_weight   = float(args.bc_loss_weight),
        beta             = args.beta,
        use_balance_sample = True,
        agent_data_ratio   = 1.0,
        q_value_bound      = 1.0,
        optimize_memory_usage = True,

        # PVPTD3 extra-config flags (mirrors MetaDrive defaults)
        no_done_for_positive      = "False",
        no_done_for_negative      = "False",
        reward_0_for_positive     = "False",
        reward_0_for_negative     = "False",
        reward_n2_for_intervention= "False",
        reward_1_for_all          = "False",
        use_weighted_reward       = "False",
        remove_negative           = "False",
        adaptive_batch_size       = "False",
        with_human_proxy_value_loss = "False",
        with_agent_proxy_value_loss = "False",
        simple_batch              = "True",
    )

    # ---- Callbacks ----
    callbacks = [
        CheckpointCallback(
            name_prefix = "ppl_dh_robosuite",
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

    # ---- Instantiate DH-PPL model ----
    model = PPL(**algo_cfg)

    # Resume from checkpoint if requested
    if args.ckpt:
        from ppl.sb3.common.save_util import load_from_zip_file
        ckpt = Path(args.ckpt)
        print(f"[DH-PPL] Resuming from checkpoint: {ckpt}")
        _data, params, _pv = load_from_zip_file(
            str(ckpt), device=model.device, print_system_info=False
        )
        model.set_parameters(params, exact_match=True, device=model.device)

    # Attach model to the inner env so the takeover env can:
    #   (a) call model.should_add_to_preference_buffer() for the DH gate
    #   (b) use model.policy for trajectory prediction
    #   (c) write to model.preference_buffer
    # Walk the wrapper chain: SharedControlMonitor → Monitor → ExpertTakeoverEnvDH
    inner_env = train_env.env.env   # type: RobosuiteExpertTakeoverEnvDH
    inner_env.model = model

    # ---- Launch training ----
    print(f"\n[DH-PPL] Starting training: {args.total_timesteps:,} timesteps")
    print(f"[DH-PPL] Eval every {eval_freq} steps over 20 episodes.")
    print(f"[DH-PPL] Success: proportion_wiped >= 1.0 OR native _check_success().")
    print(f"[DH-PPL] DH gate: rolling-mean uncertainty threshold, window=200.\n")

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
        # Buffer persistence (disable for speed; enable for reproducibility)
        save_buffer         = False,
    )

    print(f"\n[DH-PPL] Training complete.  Artefacts in: {trial_dir}")
    train_env.close()
    eval_env.close()
