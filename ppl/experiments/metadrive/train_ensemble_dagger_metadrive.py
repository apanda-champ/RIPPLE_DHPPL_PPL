"""
Train Ensemble DAgger on MetaDrive
===================================

Usage example
-------------
    python train_ensemble_dagger_metadrive.py \
        --num_ensemble 5 \
        --uncertainty_threshold 0.05 \
        --batch_size 512 \
        --seed 0 \
        --wandb

Algorithm recap
---------------
* K actor networks are maintained (default K=5).
* The environment calls model.get_uncertainty(obs) at each step.
  If the ensemble std of predicted actions exceeds `uncertainty_threshold`,
  the PPO expert takes over and its action is stored in human_data_buffer.
* All K actors are trained with behaviour-cloning loss on human_data_buffer.
* No critic / Q-value loss — pure imitation learning.

Parallel to train_ppl_metadrive.py
-----------------------------------
This script is intentionally structured identically to
train_ppl_metadrive.py so that experiments can be compared directly.
Only the algorithm class, environment class, and a small set of hyper-
parameters differ.
"""

import argparse
import os
import uuid
from pathlib import Path

import pathlib

from ppl.experiments.metadrive.ensemble_dagger_env import EnsembleDAggerEnv
from ppl.experiments.metadrive.driving_env import DrivingEnv
from ppl.sb3.ensemble_dagger import EnsembleDAgger
from ppl.sb3.common.callbacks import CallbackList, CheckpointCallback
from ppl.sb3.common.monitor import Monitor
from ppl.sb3.common.vec_env import SubprocVecEnv
from ppl.sb3.common.wandb_callback import WandbCallback
from ppl.sb3.haco import HACOReplayBuffer
from ppl.sb3.td3.policies import TD3Policy
from ppl.utils.shared_control_monitor import SharedControlMonitor
from ppl.utils.utils import get_time_str

FOLDER_PATH = pathlib.Path(__file__).parent.parent  # points to ppl/experiments/


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Ensemble DAgger on MetaDrive."
    )
    # Experiment meta
    parser.add_argument(
        "--exp_name", default="ensemble_dagger_metadrive", type=str,
        help="Human-readable experiment name (used in log paths).",
    )
    parser.add_argument("--seed", default=0, type=int)

    # DAgger / ensemble hyper-parameters
    parser.add_argument(
        "--num_ensemble", default=5, type=int,
        help="Number of actor networks in the ensemble (>= 2).",
    )
    parser.add_argument(
        "--uncertainty_threshold", default=0.05, type=float,
        help="Ensemble std threshold above which the expert is queried. "
             "Lower values → more frequent expert intervention.",
    )

    # Training hyper-parameters
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--total_timesteps", default=10_000, type=int)
    parser.add_argument("--buffer_size", default=50_000, type=int)
    parser.add_argument("--learning_starts", default=10, type=int,
                        help="Steps before the first gradient update.")

    # Checkpoint / logging
    parser.add_argument("--save_freq", default=150, type=int,
                        help="Model checkpoint frequency (in steps).")
    parser.add_argument("--ckpt", default="", type=str,
                        help="Path to an existing model checkpoint to resume.")

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_team", type=str, default="")

    # Debug
    parser.add_argument(
        "--toy_env", action="store_true",
        help="Single-map, no traffic, renderer on — for quick debugging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # ===== Experiment naming & directory setup =====
    experiment_batch_name = "EnsembleDAgger"
    seed = args.seed
    trial_name = "{}_{}".format(experiment_batch_name, uuid.uuid4().hex[:8])
    print(f"Trial name: {trial_name}")

    use_wandb = args.wandb
    if not use_wandb:
        print("[WARNING] Wandb is disabled — stats will only be logged locally.")

    log_dir = FOLDER_PATH.parent.parent          # repo root
    experiment_dir = Path(log_dir) / "runs" / experiment_batch_name
    trial_dir = experiment_dir / trial_name

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=False)       # fail if dir already exists
    print(f"Logging training data to: {trial_dir}")

    # ===== Config dict (mirrors train_ppl_metadrive.py structure) =====
    config = dict(

        # --- Environment ---
        env_config=dict(
            uncertainty_threshold=args.uncertainty_threshold,
        ),

        # --- Algorithm ---
        algo=dict(
            # EnsembleDAgger-specific
            num_ensemble=args.num_ensemble,
            uncertainty_threshold=args.uncertainty_threshold,

            # Inherited PVPTD3 / TD3 settings
            # (BC-only flags are set automatically in EnsembleDAgger.__init__)
            only_bc_loss="True",
            add_bc_loss="True",
            with_human_proxy_value_loss="False",
            with_agent_proxy_value_loss="False",
            bc_loss_weight=1.0,        # weight on BC loss (always 1 for DAgger)
            beta=0.0,                  # unused; required by PVPTD3 extra_config
            agent_data_ratio=1.0,
            simple_batch="False",
            adaptive_batch_size="False",
            no_done_for_positive="False",
            no_done_for_negative="False",
            reward_0_for_positive="False",
            reward_0_for_negative="False",
            reward_n2_for_intervention="False",
            reward_1_for_all="False",
            use_weighted_reward="False",
            remove_negative="False",

            # SB3 / TD3 settings
            use_balance_sample=True,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(),
            policy_kwargs=dict(net_arch=[148, 148]),
            env=None,                  # filled in below
            learning_rate=args.learning_rate,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=str(trial_dir),
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # --- Logging meta ---
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir),
    )

    # Optional toy-env overrides for debugging
    if args.toy_env:
        config["env_config"].update(
            num_scenarios=1,
            traffic_density=0.0,
            map="COT",
            use_render=True,
        )

    # ===== Training environment =====
    train_env = EnsembleDAggerEnv(config=config["env_config"])
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    train_env = SharedControlMonitor(
        env=train_env, folder=trial_dir / "data", prefix=trial_name
    )
    config["algo"]["env"] = train_env

    # ===== Evaluation environment =====
    def _make_eval_env():
        from ppl.sb3.common.monitor import Monitor as _Monitor
        eval_env = DrivingEnv(config=dict(start_seed=1000))
        return _Monitor(env=eval_env, filename=str(trial_dir))

    eval_env = SubprocVecEnv([_make_eval_env])
    eval_freq = args.save_freq

    # ===== Callbacks =====
    callbacks = [
        CheckpointCallback(
            name_prefix="rl_model",
            verbose=2,
            save_freq=args.save_freq,
            save_path=str(trial_dir / "models"),
        )
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=args.wandb_team,
                project_name=args.wandb_project,
                config=config,
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Build the model =====
    # Pop EnsembleDAgger-specific keys before passing to avoid duplicate
    # keyword arguments (they are passed positionally to __init__).
    algo_cfg = config["algo"].copy()
    num_ensemble = algo_cfg.pop("num_ensemble")
    uncertainty_threshold = algo_cfg.pop("uncertainty_threshold")

    model = EnsembleDAgger(
        num_ensemble=num_ensemble,
        uncertainty_threshold=uncertainty_threshold,
        **algo_cfg,
    )

    # Optionally resume from a checkpoint.
    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}")
        from ppl.sb3.common.save_util import load_from_zip_file
        data, params, _ = load_from_zip_file(
            ckpt, device=model.device, print_system_info=False
        )
        model.set_parameters(params, exact_match=True, device=model.device)

    # Attach model to the training env so EnsembleDAggerEnv can call
    # model.get_uncertainty(obs).  The wrapper chain is:
    #   SharedControlMonitor → Monitor → EnsembleDAggerEnv
    train_env.env.env.model = model

    # ===== Training =====
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=True,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=50,
        eval_log_path=str(trial_dir),
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
    )
