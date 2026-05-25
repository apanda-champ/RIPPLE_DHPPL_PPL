"""
train_ppl_robosuite_general.py  –  PPL training with generalized spatial logic.
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

from ppl.ppl import PPL
from ppl.sb3.common.callbacks import CallbackList, CheckpointCallback
from ppl.sb3.common.monitor import Monitor
from ppl.sb3.common.vec_env import DummyVecEnv
from ppl.sb3.common.wandb_callback import WandbCallback
from ppl.sb3.haco import HACOReplayBuffer
from ppl.sb3.td3.policies import TD3Policy

# Import the new generalized env for evaluation
from ppl.experiments.robosuite.robosuite_base_env_general import RobosuiteBaseEnv, OBS_KEYS
from ppl.experiments.robosuite.robosuite_expert_takeover_env import RobosuiteExpertTakeoverEnv
from ppl.experiments.robosuite.robosuite_shared_control_monitor import SharedControlMonitor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name",     default="ppl_robosuite_general", type=str)
    p.add_argument("--batch_size",   default=1024,  type=int)
    p.add_argument("--save_freq",    default=150,   type=int)
    p.add_argument("--seed",         default=0,     type=int)
    p.add_argument("--total_timesteps", default=20_000, type=int)
    p.add_argument("--learning_starts", default=10, type=int)
    p.add_argument("--buffer_size",  default=50_000, type=int)

    p.add_argument("--only_bc_loss",      default="False", type=str, choices=["True", "False"])
    p.add_argument("--bc_loss_weight",    default=1.0,     type=float)
    p.add_argument("--beta",              default=0.1,     type=float)
    p.add_argument("--num_predicted_steps", default=20,   type=int)
    p.add_argument("--failure_check_freq",  default=10,   type=int)
    p.add_argument("--preference_horizon",  default=3,    type=int)

    p.add_argument("--expert_checkpoint", default="", type=str)
    p.add_argument("--expert_noise",      default=0.0, type=float)
    p.add_argument("--disable_expert",    action="store_true")

    p.add_argument("--ckpt", default="", type=str)
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb_project", default="", type=str)
    p.add_argument("--wandb_team",    default="", type=str)
    p.add_argument("--log_dir",       default="", type=str)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    experiment_batch_name = "PPL_Robosuite_General"
    if args.only_bc_loss == "True":
        experiment_batch_name += "_BCOnly"
    trial_name = f"{experiment_batch_name}_{uuid.uuid4().hex[:8]}"
    print(f"[PPL] Trial name: {trial_name}")

    root_log = Path(args.log_dir) if args.log_dir else Path("runs")
    trial_dir = root_log / experiment_batch_name / trial_name
    os.makedirs(trial_dir, exist_ok=True)
    print(f"[PPL] Logging to: {trial_dir}")

    # Inject the generalized OBS_KEYS directly into the takeover environment
    env_cfg = dict(
        obs_keys            = OBS_KEYS, 
        num_predicted_steps = args.num_predicted_steps,
        failure_check_freq  = args.failure_check_freq,
        preference_horizon  = args.preference_horizon,
        expert_noise        = args.expert_noise,
        disable_expert      = args.disable_expert,
        reward_shaping      = True,
        has_renderer        = False,
        has_offscreen_renderer = False,
        horizon             = 1000, 
    )
    if args.expert_checkpoint:
        env_cfg["expert_checkpoint"] = args.expert_checkpoint

    train_env = RobosuiteExpertTakeoverEnv(config=env_cfg)
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    train_env = SharedControlMonitor(
        env=train_env, folder=str(trial_dir / "data"), prefix=trial_name, save_freq=1000
    )

    def _make_eval_env():
        eval_env = RobosuiteBaseEnv(config={"reward_shaping": True, "has_renderer": False})
        return Monitor(env=eval_env)

    eval_env  = DummyVecEnv([_make_eval_env])
    eval_freq = args.save_freq

    algo_cfg = dict(
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
        only_bc_loss            = args.only_bc_loss,
        bc_loss_weight          = float(args.bc_loss_weight),
        beta                    = args.beta,
        add_bc_loss             = "True" if args.bc_loss_weight > 0.0 else "False",
        use_balance_sample      = True,
        agent_data_ratio        = 1.0,
        q_value_bound           = 1.0,
        optimize_memory_usage   = True,
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

    callbacks = [CheckpointCallback(name_prefix="ppl_robosuite", verbose=2, save_freq=args.save_freq, save_path=str(trial_dir / "models"))]
    callbacks = CallbackList(callbacks)

    model = PPL(**algo_cfg)

    inner_env = train_env.env.env  
    inner_env.model = model

    print(f"\n[PPL] Starting training for {args.total_timesteps:,} timesteps …\n")

    model.learn(
        total_timesteps   = args.total_timesteps,
        callback          = callbacks,
        reset_num_timesteps = True,
        eval_env          = eval_env,
        eval_freq         = eval_freq,
        n_eval_episodes   = 20,
        eval_log_path     = str(trial_dir),
        tb_log_name       = experiment_batch_name,
        log_interval      = 1,
        save_buffer       = False,
    )

    print(f"\n[PPL] Training complete.  Artefacts in: {trial_dir}")
    train_env.close()
    eval_env.close()
