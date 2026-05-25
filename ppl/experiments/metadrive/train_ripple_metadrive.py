/mnt/extra_SSD/rushikesh/PPL_RIPPLE/PPL/ppl/experiments/metadrive/train_ripple_metadrive.py

"""
Training script for RIPPLE on MetaDrive. Mirrors train_ppl_metadrive.py with
added CLI flags for the four loss weights and data-collection hyperparameters.

Place at ppl/experiments/metadrive/train_ripple_metadrive.py.
"""
import argparse
import os
import pathlib
import uuid
from pathlib import Path

from ppl.experiments.metadrive.ripple_env import RIPPLEEnv
from ppl.ripple import RIPPLE
from ppl.sb3.common.callbacks import CallbackList, CheckpointCallback
from ppl.sb3.common.monitor import Monitor
from ppl.sb3.common.vec_env import SubprocVecEnv
from ppl.sb3.common.wandb_callback import WandbCallback
from ppl.sb3.haco import HACOReplayBuffer
from ppl.sb3.td3.policies import TD3Policy
from ppl.utils.shared_control_monitor import SharedControlMonitor

import os
import sys
import random

# Must be set BEFORE numpy/torch imports
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


FOLDER_PATH = pathlib.Path(__file__).parent.parent

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="ripple_metadrive", type=str)
    p.add_argument("--batch_size", default=1024, type=int)
    p.add_argument("--save_freq", default=150, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--wandb_team", type=str, default="")
    p.add_argument("--only_bc_loss", default="True", type=str)
    p.add_argument("--ckpt", default="", type=str)
    p.add_argument("--num_predicted_steps", default=20, type=int)
    p.add_argument("--preference_horizon", default=4, type=int)
    p.add_argument("--toy_env", action="store_true")
    p.add_argument("--bc_loss_weight", type=float, default=1.0)
    p.add_argument("--beta", default=0.1, type=float)

    # # RIPPLE-specific flags
    # p.add_argument("--lambda_fwd",    type=float, default=1.0, help="L_pref^fwd weight")
    # p.add_argument("--lambda_back",   type=float, default=0.5, help="L_pref^back weight")
    # p.add_argument("--lambda_silent", type=float, default=0.3, help="L_silent weight")
    # p.add_argument("--lambda_traj",   type=float, default=0.5, help="L_traj weight")
    # p.add_argument("--backward_horizon",   type=int,   default=3)
    # p.add_argument("--silent_noise_scale", type=float, default=0.3)
    # p.add_argument("--silent_margin",      type=float, default=0.15)
    # p.add_argument("--traj_max_len",       type=int,   default=10)

    # RIPPLE-specific flags
    p.add_argument("--lambda_fwd",    type=float, default=1.0, help="L_pref^fwd weight")
    p.add_argument("--lambda_back",   type=float, default=1.0, help="L_pref^back weight")
    p.add_argument("--lambda_silent", type=float, default=0.0, help="L_silent weight")
    p.add_argument("--lambda_traj",   type=float, default=1.0, help="L_traj weight")
    p.add_argument("--backward_horizon",   type=int,   default=4)
    p.add_argument("--silent_noise_scale", type=float, default=0.3)
    p.add_argument("--silent_margin",      type=float, default=0.15)
    p.add_argument("--traj_max_len",       type=int,   default=10)

    # Per-term beta flags (logit sharpness for each preference loss)
    p.add_argument("--beta_fwd",    type=float, default=0.1,  help="Beta for forward preferences")
    p.add_argument("--beta_back",   type=float, default=0.1,  help="Beta for backward preferences")
    p.add_argument("--beta_silent", type=float, default=0.05, help="Beta for silent (gentler)")
    p.add_argument("--beta_traj",   type=float, default=0.1,  help="Beta for trajectory (sharper)")


    args = p.parse_args()
    set_all_seeds(args.seed)

    # ---- Experiment bookkeeping ---------------------------------------------
    experiment_batch_name = "RIPPLE"
    if args.only_bc_loss == "True":
        experiment_batch_name = "RIPPLE_BCLossOnly"
    # trial_name = f"{experiment_batch_name}_{uuid.uuid4().hex[:8]}"
    trial_name = args.exp_name if args.exp_name != "ripple_metadrive" else "{}_{}".format(experiment_batch_name, uuid.uuid4().hex[:8])
    print("Trial name:", trial_name)

    log_dir = FOLDER_PATH.parent.parent
    experiment_dir = Path(log_dir) / "runs" / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=False)
    print(f"Logging to {trial_dir}")

    # ---- Config --------------------------------------------------------------
    config = dict(
        env_config=dict(
            num_predicted_steps=args.num_predicted_steps,
            preference_horizon=args.preference_horizon,
            # Pass RIPPLE collection settings down to the env so it can populate
            # the backward/silent/traj buffers on the fly
            backward_horizon=args.backward_horizon,
            silent_noise_scale=args.silent_noise_scale,
            silent_margin=args.silent_margin,
            traj_max_len=args.traj_max_len,
            start_seed=100, 
        ),
        algo=dict(
            only_bc_loss=args.only_bc_loss,
            bc_loss_weight=args.bc_loss_weight,
            beta=args.beta,
            add_bc_loss="True" if args.bc_loss_weight > 0.0 else "False",
            use_balance_sample=True,
            agent_data_ratio=1.0,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,
            learning_starts=10,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=args.seed,
            device="auto",

            # RIPPLE loss weights + collection hyperparameters
            lambda_fwd=args.lambda_fwd,
            lambda_back=args.lambda_back,
            lambda_silent=args.lambda_silent,
            lambda_traj=args.lambda_traj,
            beta_fwd=args.beta_fwd,
            beta_back=args.beta_back,
            beta_silent=args.beta_silent,
            beta_traj=args.beta_traj,
            backward_horizon=args.backward_horizon,
            silent_noise_scale=args.silent_noise_scale,
            silent_margin=args.silent_margin,
            traj_max_len=args.traj_max_len,
        ),
        exp_name=experiment_batch_name,
        seed=args.seed,
        use_wandb=args.wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir),
    )
    if args.toy_env:
        config["env_config"].update(num_scenarios=1, traffic_density=0.0, map="COT", use_render=True)

    # ---- Envs ---------------------------------------------------------------
    set_all_seeds(args.seed)
    train_env = RIPPLEEnv(config=config["env_config"])
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)
    config["algo"]["env"] = train_env

    def _make_eval_env():
        from ppl.experiments.metadrive.driving_env import DrivingEnv
        from ppl.sb3.common.monitor import Monitor as _M
        eval_env = DrivingEnv(config=dict(start_seed=1000))
        return _M(env=eval_env, filename=str(trial_dir))

    eval_env, eval_freq = SubprocVecEnv([_make_eval_env]), 150

    # ---- Callbacks ----------------------------------------------------------
    callbacks = [CheckpointCallback(
        name_prefix="rl_model", verbose=2, save_freq=args.save_freq,
        save_path=str(trial_dir / "models"),
    )]
    if args.wandb:
        callbacks.append(WandbCallback(
            trial_name=trial_name, exp_name=experiment_batch_name,
            team_name=args.wandb_team, project_name=args.wandb_project, config=config,
        ))
    callbacks = CallbackList(callbacks)

    # ---- Train --------------------------------------------------------------
    set_all_seeds(args.seed)
    model = RIPPLE(**config["algo"])
    if args.ckpt:
        from ppl.sb3.common.save_util import load_from_zip_file
        data, params, _ = load_from_zip_file(Path(args.ckpt), device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)

    train_env.env.env.model = model  # env reads model.back_pref_buffer etc. off this
    model.learn(
        total_timesteps=10_000,
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
