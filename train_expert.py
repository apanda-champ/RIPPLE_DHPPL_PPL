"""
train_expert.py
===============
Train a PPO expert policy on a Robosuite task using the Panda robot.
The resulting model is saved to:
    ./expert_models/<task>/final_expert.zip

Usage
-----
# TableWiping (default)
python train_expert.py --task TableWiping --steps 500000 --seed 0

# NutAssembly
python train_expert.py --task NutAssembly  --steps 1000000 --seed 0
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── silence noisy robosuite controller logs ──────────────────────────────────
logging.getLogger("robosuite").setLevel(logging.WARNING)

# ── numpy / old-gym compatibility shim ───────────────────────────────────────
if not hasattr(np, "float"):
    np.float = float


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium → old-Gym adapter
# ─────────────────────────────────────────────────────────────────────────────
class LegacyEnvAdapter(gym.Wrapper):
    """
    Convert Gymnasium-style (5-tuple) outputs to old-Gym-style (4-tuple).
    Also adds a default metadata dict if missing.
    """

    def __init__(self, env):
        super().__init__(env)
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {
                "render.modes": ["human", "rgb_array"],
                "video.frames_per_second": 20,
            }

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        return ret[0] if isinstance(ret, tuple) else ret

    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            return obs, reward, terminated or truncated, info
        return ret


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────
TASK_CONFIGS = {
    "TableWiping": dict(env_id="Wipe"),
    "NutAssembly": dict(env_id="NutAssemblySquare"),
}


def make_env(task_name: str, seed: int = 0, monitor_dir: str = ""):
    """
    Build a single Robosuite env wrapped for SB3 PPO training.

    Stack:
        robosuite env
        → GymWrapper
        → LegacyEnvAdapter
        → Monitor
    """
    cfg = TASK_CONFIGS[task_name]

    raw_env = suite.make(
        cfg["env_id"],
        robots="Panda",          # ← Franka Emika Panda (7-DOF)
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        has_renderer=False,
        has_offscreen_renderer=False,
    )

    env = GymWrapper(raw_env)
    env = LegacyEnvAdapter(env)
    env = Monitor(env, filename=monitor_dir if monitor_dir else None)

    try:
        env.seed(seed)
    except Exception:
        pass

    return env


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train a PPO expert for PPL on Robosuite / Panda."
    )
    parser.add_argument(
        "--task",
        default="TableWiping",
        choices=list(TASK_CONFIGS.keys()),
        help="Robosuite task name.",
    )
    parser.add_argument("--seed",  default=0,       type=int)
    parser.add_argument("--steps", default=500_000, type=int,
                        help="Total PPO training timesteps.")
    parser.add_argument("--n_steps", default=2048,  type=int,
                        help="Steps per rollout per env (PPO n_steps).")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr",   default=3e-4,     type=float)
    parser.add_argument("--save_freq", default=50_000, type=int,
                        help="Checkpoint every N timesteps.")
    parser.add_argument("--eval_freq",  default=10_000, type=int)
    parser.add_argument("--n_eval_eps", default=10,     type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)

    log_dir   = Path("./expert_runs") / args.task
    save_dir  = Path("./expert_models") / args.task
    ckpt_dir  = save_dir / "checkpoints"
    for d in (log_dir, save_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Task      : {args.task}")
    print(f"  Robot     : Panda (7-DOF)")
    print(f"  Seed      : {args.seed}")
    print(f"  Steps     : {args.steps:,}")
    print(f"  Save dir  : {save_dir}")
    print("=" * 70)

    # ── Training env (vectorised so SB3 PPO is happy) ──────────────────────
    train_env = DummyVecEnv([
        lambda: make_env(args.task, seed=args.seed, monitor_dir=str(log_dir))
    ])

    # ── Eval env ────────────────────────────────────────────────────────────
    eval_env = DummyVecEnv([
        lambda: make_env(args.task, seed=args.seed + 1000)
    ])

    # ── Callbacks ───────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(ckpt_dir),
        name_prefix="expert_ppo",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_eps,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # ── PPO model ───────────────────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=args.seed,
    )

    # ── Train ───────────────────────────────────────────────────────────────
    print("\nStarting PPO expert training …\n")
    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="expert_ppo",
        reset_num_timesteps=True,
    )

    # ── Save final model ────────────────────────────────────────────────────
    final_path = save_dir / "final_expert"
    model.save(str(final_path))

    print("\n" + "=" * 70)
    print(f"  Expert training complete.")
    print(f"  Final model : {final_path}.zip")
    print(f"  Best model  : {save_dir / 'best' / 'best_model.zip'}")
    print("=" * 70)


if __name__ == "__main__":
    main()