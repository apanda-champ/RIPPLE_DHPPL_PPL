"""
train_ppl_robosuite.py  –  PPL training on Robosuite Wipe (Panda + OSC_POSE).

Mirrors experiments/metadrive/train_ppl_metadrive.py from the original PPL
codebase.  The key difference is that DrivingEnv / ExpertTakeoverEnv are
replaced by their Robosuite equivalents.

Typical run
-----------
    # 1. Pre-train the OSC expert (one-time)
    python train_osc_expert.py --total_timesteps 5000000 \
        --save_path experiments/robosuite/osc_expert.zip

    # 2. Run PPL
    python train_ppl_robosuite.py \
        --expert_checkpoint experiments/robosuite/osc_expert.zip \
        --total_timesteps 20000 \
        --seed 0

    # 3. (Optional) Enable W&B logging
    python train_ppl_robosuite.py --wandb \
        --wandb_project my_project --wandb_team my_team

WandB metrics logged
--------------------
    eval/success_rate        – fraction of eval episodes where proportion_wiped
                               >= 0.90 OR native _check_success() fires.
                               Evaluated every eval_freq (= save_freq) training
                               steps over n_eval_episodes (= 20) episodes.
                               Flows automatically via sync_tensorboard=True.

    train/expert_steps_total – cumulative number of environment steps where the
                               OSC expert took over (i.e. info["takeover"]=True).
                               Logged every training step by ExpertStepsCallback.
                               Flows automatically via sync_tensorboard=True.
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

# PPL algorithm and SB3 utilities
from ppl.ppl import PPL
from ppl.sb3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from ppl.sb3.common.monitor import Monitor
from ppl.sb3.common.vec_env import DummyVecEnv, SubprocVecEnv
from ppl.sb3.common.wandb_callback import WandbCallback
from ppl.sb3.haco import HACOReplayBuffer
from ppl.sb3.td3.policies import TD3Policy

# Robosuite-specific PPL components
from ppl.experiments.robosuite.robosuite_base_env import RobosuiteBaseEnv
from ppl.experiments.robosuite.robosuite_expert_takeover_env import (
    RobosuiteExpertTakeoverEnv,
)
from ppl.experiments.robosuite.robosuite_shared_control_monitor import (
    SharedControlMonitor,
)


# ---------------------------------------------------------------------------
# ExpertStepsCallback
# ---------------------------------------------------------------------------

class ExpertStepsCallback(BaseCallback):
    """
    Logs cumulative expert intervention steps to TensorBoard / WandB.

    Increments a running counter whenever info["takeover"] is True and
    records it as ``train/expert_steps_total`` via the SB3 logger.
    Because WandbCallback is initialised with ``sync_tensorboard=True``,
    this metric is automatically mirrored to WandB without any extra
    wandb.log() calls.

    Note: env.total_takeover_count resets every episode, so we accumulate
    here across all episodes for a monotonically increasing global count.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._expert_steps_total: int = 0

    def _on_step(self) -> bool:
        # self.locals["infos"] is a list with one entry per parallel env.
        # Training uses a single (non-vectorised) env so index 0 is correct.
        info = self.locals.get("infos", [{}])[0]
        if info.get("takeover", False):
            self._expert_steps_total += 1
        self.logger.record("train/expert_steps_total", self._expert_steps_total)
        return True


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PPL training on Robosuite.")
    p.add_argument("--exp_name",     default="ppl_robosuite", type=str)
    p.add_argument("--batch_size",   default=1024,  type=int)
    p.add_argument("--save_freq",    default=150,   type=int,
                   help="Checkpoint every N agent steps. Also used as eval_freq.")
    p.add_argument("--seed",         default=0,     type=int)
    p.add_argument("--total_timesteps", default=20_000, type=int)
    p.add_argument("--learning_starts", default=10, type=int,
                   help="Steps of random exploration before training begins.")
    p.add_argument("--buffer_size",  default=50_000, type=int)

    # PPL hyper-parameters
    p.add_argument("--only_bc_loss",      default="False", type=str,
                   choices=["True", "False"])
    p.add_argument("--bc_loss_weight",    default=1.0,     type=float)
    p.add_argument("--beta",              default=0.1,     type=float,
                   help="DPO temperature β.")
    p.add_argument("--num_predicted_steps", default=20,   type=int,
                   help="Prediction horizon H for failure detection.")
    p.add_argument("--failure_check_freq",  default=10,   type=int,
                   help="Run trajectory predictor every N steps.")
    p.add_argument("--preference_horizon",  default=3,    type=int,
                   help="Number of preference pairs L stored per takeover.")

    # Expert
    p.add_argument("--expert_checkpoint", default="", type=str,
                   help="Path to osc_expert.zip.  Empty = auto-discover.")
    p.add_argument("--expert_noise",      default=0.0, type=float)
    p.add_argument("--disable_expert",    action="store_true",
                   help="Run without expert interventions (ablation).")

    # Checkpoint resume
    p.add_argument("--ckpt", default="", type=str,
                   help="Resume PPL model from a .zip checkpoint.")

    # Logging
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb_project", default="", type=str)
    p.add_argument("--wandb_team",    default="", type=str)
    p.add_argument("--log_dir",       default="", type=str,
                   help="Root for run logs (default: ./runs/).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # ---- Experiment naming ----
    experiment_batch_name = "PPL_Robosuite"
    if args.only_bc_loss == "True":
        experiment_batch_name += "_BCOnly"
    trial_name = f"{experiment_batch_name}_{uuid.uuid4().hex[:8]}"
    print(f"[PPL] Trial name: {trial_name}")

    if not args.wandb:
        print("[WARNING] W&B logging is disabled.  Pass --wandb to enable.")

    root_log = Path(args.log_dir) if args.log_dir else Path("runs")
    experiment_dir = root_log / experiment_batch_name
    trial_dir      = experiment_dir / trial_name
    os.makedirs(trial_dir, exist_ok=True)
    print(f"[PPL] Logging to: {trial_dir}")

    # ---- PPL environment config ----
    env_cfg = dict(
        num_predicted_steps = args.num_predicted_steps,
        failure_check_freq  = args.failure_check_freq,
        preference_horizon  = args.preference_horizon,
        expert_noise        = args.expert_noise,
        disable_expert      = args.disable_expert,
        # Robosuite task settings
        reward_shaping = True,
        has_renderer   = False,
        has_offscreen_renderer = False,
    )
    if args.expert_checkpoint:
        env_cfg["expert_checkpoint"] = args.expert_checkpoint

    # ---- Build training environment ----
    train_env = RobosuiteExpertTakeoverEnv(config=env_cfg)
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    train_env = SharedControlMonitor(
        env=train_env,
        folder=str(trial_dir / "data"),
        prefix=trial_name,
        save_freq=1000,
    )

    # ---- Build eval environment ----
    # Eval uses RobosuiteBaseEnv (no expert) to measure pure agent performance.
    # is_success is set to True when:
    #   (a) _check_success() fires (all markers contacted), OR
    #   (b) proportion_wiped >= 0.90 at episode end (horizon 750 or early done).
    # EvalCallback reads is_success only at done=True and accumulates over
    # n_eval_episodes=20 to compute eval/success_rate.
    def _make_eval_env():
        eval_env = RobosuiteBaseEnv(
            config={"reward_shaping": True, "has_renderer": False}
        )
        return Monitor(env=eval_env)

    eval_env  = DummyVecEnv([_make_eval_env])
    eval_freq = args.save_freq  # evaluate every save_freq training steps

    # ---- PPL algorithm config ----
    algo_cfg = dict(
        # Core SB3/TD3 settings
        policy              = TD3Policy,
        env                 = train_env,
        replay_buffer_class = HACOReplayBuffer,
        replay_buffer_kwargs= {},
        policy_kwargs       = dict(net_arch=[256, 256]),
        learning_rate       = 1e-4,
        buffer_size         = args.buffer_size,
        learning_starts     = args.learning_starts,
        batch_size          = args.batch_size,
        tau                 = 0.005,
        gamma               = 0.99,
        train_freq          = (1, "step"),
        action_noise        = None,
        tensorboard_log     = str(trial_dir),
        create_eval_env     = False,
        verbose             = 2,
        seed                = args.seed,
        device              = "auto",
        # PPL-specific
        only_bc_loss            = args.only_bc_loss,
        bc_loss_weight          = float(args.bc_loss_weight),
        beta                    = args.beta,
        add_bc_loss             = "True" if args.bc_loss_weight > 0.0 else "False",
        use_balance_sample      = True,
        agent_data_ratio        = 1.0,
        q_value_bound           = 1.0,
        optimize_memory_usage   = True,
        # Extra PPL flags (mirror MetaDrive defaults)
        no_done_for_positive    = "False",
        no_done_for_negative    = "False",
        reward_0_for_positive   = "False",
        reward_0_for_negative   = "False",
        reward_n2_for_intervention = "False",
        reward_1_for_all        = "False",
        use_weighted_reward     = "False",
        remove_negative         = "False",
        adaptive_batch_size     = "False",
        with_human_proxy_value_loss = "False",
        with_agent_proxy_value_loss = "False",
        simple_batch            = "True",
    )

    # ---- Callbacks ----
    callbacks = [
        CheckpointCallback(
            name_prefix = "ppl_robosuite",
            verbose     = 2,
            save_freq   = args.save_freq,
            save_path   = str(trial_dir / "models"),
        ),
        # Logs train/expert_steps_total to TensorBoard/WandB every step.
        # Accumulates across episodes; never resets unlike env.total_takeover_count.
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

    # ---- Instantiate PPL model ----
    model = PPL(**algo_cfg)

    # Resume from checkpoint if requested
    if args.ckpt:
        from ppl.sb3.common.save_util import load_from_zip_file
        ckpt = Path(args.ckpt)
        print(f"[PPL] Resuming from checkpoint: {ckpt}")
        _data, params, _pv = load_from_zip_file(
            str(ckpt), device=model.device, print_system_info=False
        )
        model.set_parameters(params, exact_match=True, device=model.device)

    # Attach the PPL model to the training env so the takeover env can
    # access the preference buffer and the agent policy for rollouts.
    # Walk the wrapper chain: SharedControlMonitor → Monitor → ExpertTakeoverEnv
    inner_env = train_env.env.env  # type: RobosuiteExpertTakeoverEnv
    inner_env.model = model

    print(f"\n[PPL] Starting training for {args.total_timesteps:,} timesteps …\n")
    print(f"[PPL] Eval every {eval_freq} steps over {20} episodes.")
    print(f"[PPL] Success criterion: proportion_wiped >= 0.90 OR native _check_success().\n")

    # ---- Launch PPL ----
    model.learn(
        total_timesteps   = args.total_timesteps,
        callback          = callbacks,
        reset_num_timesteps = True,
        # Eval — runs every eval_freq training steps, 20 episodes each time.
        # EvalCallback logs eval/success_rate using info["is_success"] at done=True.
        # WandbCallback mirrors this via sync_tensorboard=True automatically.
        eval_env          = eval_env,
        eval_freq         = eval_freq,
        n_eval_episodes   = 20,
        eval_log_path     = str(trial_dir),
        # Logging
        tb_log_name       = experiment_batch_name,
        log_interval      = 1,
        # Buffer persistence (disable for speed; enable for reproducibility)
        save_buffer       = False,
    )

    print(f"\n[PPL] Training complete.  Artefacts in: {trial_dir}")
    train_env.close()
    eval_env.close()
