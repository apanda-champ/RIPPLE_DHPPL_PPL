"""
train_osc_expert.py  –  Pre-train the OSC expert policy on Robosuite Wipe.

This script trains a PPO agent (using the PPL SB3 fork) on the Robosuite
Wipe task with the Franka Panda robot and OSC_POSE controller.
The resulting checkpoint is later loaded by OscExpert / ExpertTakeoverEnv.

Typical run
-----------
    python train_osc_expert.py \\
        --total_timesteps 5000000 \\
        --save_path experiments/robosuite/osc_expert

The expert needs to reach a consistent ~75-85% success rate before being used
for PPL takeover.
"""

import argparse
import os
from pathlib import Path

# PPL's bundled SB3
from ppl.sb3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from ppl.sb3.common.monitor import Monitor
from ppl.sb3.common.vec_env import DummyVecEnv, SubprocVecEnv
from ppl.sb3.ppo import PPO
from ppl.sb3.ppo.policies import ActorCriticPolicy

from ppl.experiments.robosuite.robosuite_base_env import RobosuiteBaseEnv


# ---------------------------------------------------------------------------
# Entropy coefficient scheduler
# ---------------------------------------------------------------------------

class EntropyCoefficientScheduler(BaseCallback):
    """
    Linearly decays ent_coef from start to end over
    the first end_fraction of total training timesteps.
    """
    def __init__(self, start: float, end: float, end_fraction: float, verbose: int = 0):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.end_fraction = end_fraction

    def _on_step(self) -> bool:
        fraction_done = self.num_timesteps / self.model._total_timesteps
        progress = min(fraction_done / self.end_fraction, 1.0)
        self.model.ent_coef = self.start + progress * (self.end - self.start)
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(seed: int = 0, use_render: bool = False):
    def _init():
        env = RobosuiteBaseEnv(
            config={
                "has_renderer": use_render,
                "reward_shaping": True,
            }
        )
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init


def make_eval_env(seed: int = 9999):
    def _init():
        env = RobosuiteBaseEnv(
            config={
                "has_renderer": False,
                "reward_shaping": True,
            }
        )
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train OSC expert for PPL.")
    parser.add_argument("--total_timesteps", type=int,   default=5_000_000)
    parser.add_argument("--save_path",       type=str,   default="osc_expert")
    parser.add_argument("--n_envs",          type=int,   default=4,
                        help="Number of parallel training envs.")
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--log_dir",         type=str,   default="runs/osc_expert")
    parser.add_argument("--eval_freq",       type=int,   default=10_000,
                        help="Evaluate every N environment steps.")
    parser.add_argument("--n_eval_episodes", type=int,   default=20)
    parser.add_argument("--render",          action="store_true")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[OSC Expert Training]")
    print(f"  total_timesteps : {args.total_timesteps:,}")
    print(f"  n_envs          : {args.n_envs}")
    print(f"  log_dir         : {args.log_dir}")
    print(f"  save_path       : {save_path}.zip")

    # ---- Training envs ----
    if args.n_envs == 1:
        train_env = DummyVecEnv([make_env(seed=args.seed, use_render=args.render)])
    else:
        train_env = SubprocVecEnv(
            [make_env(seed=args.seed + i) for i in range(args.n_envs)]
        )

    # ---- Eval env ----
    eval_env = DummyVecEnv([make_eval_env(seed=9999)])

    # ---- PPO configuration ----
    model = PPO(
        policy=ActorCriticPolicy,
        env=train_env,
        # --- Rollout ---
        n_steps=4096 // args.n_envs,
        n_epochs=5,
        batch_size=256,
        # --- Optimisation ---
        learning_rate=3e-4,
        clip_range=0.1,
        # --- Discount & GAE ---
        gamma=0.995,
        gae_lambda=0.97,
        # --- Entropy: initial value; decayed by EntropyCoefficientScheduler ---
        ent_coef=1e-2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # --- Network ---
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 512], vf=[512, 512])]
        ),
        tensorboard_log=args.log_dir,
        verbose=1,
        seed=args.seed,
        device="auto",
    )

    # ---- Callbacks ----
    ent_scheduler = EntropyCoefficientScheduler(
        start=1e-2,
        end=1e-4,
        end_fraction=0.8,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.eval_freq // args.n_envs, 1),
        save_path=str(Path(args.log_dir) / "checkpoints"),
        name_prefix="osc_expert",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(Path(args.log_dir) / "best_model"),
        log_path=args.log_dir,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks = CallbackList([ent_scheduler, checkpoint_cb, eval_cb])

    # ---- Train ----
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="osc_expert_ppo",
        reset_num_timesteps=True,
    )

    # ---- Save final model ----
    final_path = str(save_path)
    model.save(final_path)
    print(f"\n[OSC Expert Training] Saved final model → {final_path}.zip")

    # Also copy best model to the expected default location
    best_model_path = Path(args.log_dir) / "best_model" / "best_model.zip"
    if best_model_path.exists():
        import shutil
        dest = save_path.parent / "osc_expert_best.zip"
        shutil.copy(best_model_path, dest)
        print(f"[OSC Expert Training] Best model also saved → {dest}")

    train_env.close()
    eval_env.close()
