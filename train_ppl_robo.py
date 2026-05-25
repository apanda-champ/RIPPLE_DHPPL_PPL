"""
train_ppl_robo.py
=================
Online PPL training on Robosuite using the Franka Emika Panda robot,
with W&B logging of human-step usage and task success rate.

Pipeline
--------
    PPO expert  (train_expert.py)
        ↓
    RobosuitePPLWrapper  (robosuite_ppl_env.py)
        – decides expert takeover via trajectory-rollout failure check
        – populates info["takeover"], info["takeover_start"], info["raw_action"]
        – populates info["is_success"] from Robosuite's info["success"]
        – tracks _human_steps / _human_steps_ep counters
        – builds preference pairs → model.preference_buffer
        ↓
    PPL.learn()  (ppl/ppl.py)
        – collects rollouts online
        – PVPTD3._store_transition routes takeover steps → human_data_buffer
        – PPL.train() = BC loss + DPO/CPL preference loss
        ↓
    PPLWandbCallback  (ppl_wandb_callback.py)
        – logs to W&B every --log_interval steps:
            ppl/human_steps_total, ppl/human_data_buffer_size,
            ppl/expert_data_ratio, ppl/pref_buffer_size,
            rollout/success_rate, rollout/ep_rew_mean,
            train/bc_loss, train/cpl_loss, train/cpl_accuracy

Usage
-----
# Step 1 – train expert (only once)
python train_expert.py --task TableWiping --steps 500000

# Step 2 – online PPL (with W&B)
python train_ppl_robo.py --task TableWiping --seed 0 --steps 50000 \\
    --wandb --wandb_project my_project --wandb_entity my_team

# Without W&B
python train_ppl_robo.py --task TableWiping --seed 0 --steps 50000
"""

import os
import sys
import argparse
import logging
import types
from pathlib import Path

import gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO

# ── PPL library imports ───────────────────────────────────────────────────────
from ppl.ppl import PPL
from ppl.sb3.td3.policies import TD3Policy
from ppl.sb3.haco import HACOReplayBuffer
from ppl.sb3.common.monitor import Monitor
from ppl.sb3.common.callbacks import CallbackList, CheckpointCallback

# ── Project-local files ───────────────────────────────────────────────────────
from robosuite_ppl_env import RobosuitePPLWrapper
from ppl_wandb_callback import PPLWandbCallback

# ── Misc setup ────────────────────────────────────────────────────────────────
logging.getLogger("robosuite").setLevel(logging.WARNING)

if not hasattr(np, "float"):
    np.float = float


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium → old-Gym adapter
# ─────────────────────────────────────────────────────────────────────────────
class LegacyEnvAdapter(gym.Wrapper):
    """Convert Gymnasium 5-tuple step returns to old-Gym 4-tuple."""

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
# Per-task defaults
# ─────────────────────────────────────────────────────────────────────────────
TASK_CONFIGS = {
    "TableWiping": dict(
        env_id="Wipe",
        num_predicted_steps=10,   # H: lookahead horizon for failure check
        preference_horizon=4,     # L: preference pairs per takeover event
        intervention_threshold=1.0,
        failure_check_freq=10,
    ),
    "NutAssembly": dict(
        env_id="NutAssemblySquare",
        num_predicted_steps=10,
        preference_horizon=6,
        intervention_threshold=2.0,
        failure_check_freq=10,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────
def make_env(task_name: str, expert_policy, seed: int = 0):
    """
    Build the full PPL training env stack:

        robosuite env
        → GymWrapper
        → LegacyEnvAdapter    (old-Gym compatibility)
        → Monitor             (episode-stat tracking)
        → RobosuitePPLWrapper (intervention + preference pairs + counters)
    """
    cfg = TASK_CONFIGS[task_name]

    raw_env = suite.make(
        cfg["env_id"],
        robots="Panda",          # Franka Emika Panda (7-DOF)
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        has_renderer=False,
        has_offscreen_renderer=False,
    )

    env = GymWrapper(raw_env)
    env = LegacyEnvAdapter(env)
    env = Monitor(env)

    ppl_config = {k: cfg[k] for k in
                  ("num_predicted_steps", "preference_horizon",
                   "intervention_threshold", "failure_check_freq")}

    env = RobosuitePPLWrapper(env, expert_policy=expert_policy, config=ppl_config)

    try:
        env.seed(seed)
    except Exception:
        pass

    return env


def _find_ppl_wrapper(env) -> RobosuitePPLWrapper:
    """Walk the wrapper stack and return the RobosuitePPLWrapper instance."""
    node = env
    while node is not None:
        if isinstance(node, RobosuitePPLWrapper):
            return node
        node = getattr(node, "env", None)
    raise RuntimeError(
        "RobosuitePPLWrapper not found in the env stack. "
        "Check the make_env() wrapper order."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Online PPL training on Robosuite (Panda robot) with W&B logging."
    )

    # ── Task / experiment ────────────────────────────────────────────────────
    parser.add_argument(
        "--task", default="TableWiping",
        choices=list(TASK_CONFIGS.keys()),
    )
    parser.add_argument("--seed",       default=0,      type=int)
    parser.add_argument("--steps",      default=50_000, type=int,
                        help="Total PPL training timesteps.")
    parser.add_argument("--save_name",  default="",     type=str,
                        help="Run folder name (auto-generated if empty).")
    parser.add_argument("--expert_path", default="",   type=str,
                        help="Path to expert .zip. Defaults to "
                             "./expert_models/<task>/final_expert.zip")
    parser.add_argument("--ckpt",       default="",     type=str,
                        help="Resume PPL training from this checkpoint .zip.")

    # ── PPL hyperparameters ──────────────────────────────────────────────────
    parser.add_argument("--batch_size",      default=256,   type=int)
    parser.add_argument("--lr",              default=1e-4,  type=float)
    parser.add_argument("--bc_loss_weight",  default=1.0,   type=float,
                        help="Weight on BC loss term (0 = disable BC).")
    parser.add_argument("--beta",            default=0.1,   type=float,
                        help="DPO/CPL temperature β.")
    parser.add_argument("--only_bc_loss",    default="False",
                        choices=["True", "False"],
                        help="Use pure BC loss (skip DPO preference loss).")
    parser.add_argument("--min_pref_pairs",  default=64,    type=int,
                        help="Collect this many pref pairs before DPO loss starts.")
    parser.add_argument("--buffer_size",     default=50_000, type=int)
    parser.add_argument("--learning_starts", default=500,   type=int,
                        help="Warm-up steps before gradient updates begin.")

    # ── Logging ───────────────────────────────────────────────────────────────
    parser.add_argument("--log_interval",    default=500,   type=int,
                        help="Log metrics (W&B + console) every N env steps.")
    parser.add_argument("--save_freq",       default=10_000, type=int,
                        help="Save a checkpoint every N env steps.")

    # ── W&B ──────────────────────────────────────────────────────────────────
    parser.add_argument("--wandb",           action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project",   default="ppl_robosuite", type=str,
                        help="W&B project name.")
    parser.add_argument("--wandb_entity",    default="",    type=str,
                        help="W&B team / entity name (optional).")
    parser.add_argument("--wandb_run_name",  default="",    type=str,
                        help="W&B run name (defaults to run_name).")

    args = parser.parse_args()

    np.random.seed(args.seed)

    run_name = args.save_name or f"PPL_{args.task}_seed{args.seed}"
    log_dir  = Path("./runs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Task             : {args.task}")
    print(f"  Robot            : Panda (7-DOF)")
    print(f"  Seed             : {args.seed}")
    print(f"  Steps            : {args.steps:,}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  LR               : {args.lr}")
    print(f"  BC loss weight   : {args.bc_loss_weight}")
    print(f"  Beta (DPO temp)  : {args.beta}")
    print(f"  Only BC loss     : {args.only_bc_loss}")
    print(f"  Min pref pairs   : {args.min_pref_pairs}")
    print(f"  Log dir          : {log_dir}")
    print(f"  W&B enabled      : {args.wandb}")
    if args.wandb:
        print(f"  W&B project      : {args.wandb_project}")
        print(f"  W&B entity       : {args.wandb_entity or '(default)'}")
    print("=" * 70)

    # ── W&B initialisation ───────────────────────────────────────────────────
    if args.wandb:
        import wandb as _wandb
        _wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or run_name,
            config={
                "task":             args.task,
                "robot":            "Panda",
                "seed":             args.seed,
                "steps":            args.steps,
                "batch_size":       args.batch_size,
                "lr":               args.lr,
                "bc_loss_weight":   args.bc_loss_weight,
                "beta":             args.beta,
                "only_bc_loss":     args.only_bc_loss,
                "min_pref_pairs":   args.min_pref_pairs,
                "buffer_size":      args.buffer_size,
                "learning_starts":  args.learning_starts,
                **TASK_CONFIGS[args.task],
            },
            sync_tensorboard=True,   # also mirrors SB3's TB scalars to W&B
            save_code=True,
        )

    # ── Load expert ──────────────────────────────────────────────────────────
    expert_path = args.expert_path or \
        f"./expert_models/{args.task}/final_expert.zip"

    if not os.path.exists(expert_path):
        raise FileNotFoundError(
            f"Expert model not found at: {expert_path}\n"
            f"Train it first with:  python train_expert.py --task {args.task}"
        )

    print(f"\nLoading expert from: {expert_path}")
    expert = PPO.load(expert_path)

    # ── Build training env ───────────────────────────────────────────────────
    env         = make_env(args.task, expert, seed=args.seed)
    ppl_wrapper = _find_ppl_wrapper(env)   # direct reference for the callback

    # ── Build or restore PPL model ───────────────────────────────────────────
    if args.ckpt:
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
        print(f"Resuming PPL from checkpoint: {args.ckpt}")
        model = PPL.load(args.ckpt, env=env)

    else:
        print("Creating new PPL model …")
        model = PPL(
            policy=TD3Policy,
            env=env,

            # ── Optimisation ────────────────────────────────────────────────
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,

            # ── Network ──────────────────────────────────────────────────────
            policy_kwargs=dict(net_arch=[256, 256]),

            # ── Buffer ───────────────────────────────────────────────────────
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(),

            # ── PPL / PVP flags ───────────────────────────────────────────────
            use_balance_sample=True,
            q_value_bound=1.0,
            agent_data_ratio=1.0,

            bc_loss_weight=args.bc_loss_weight,
            beta=args.beta,

            only_bc_loss=args.only_bc_loss,
            add_bc_loss="True" if args.bc_loss_weight > 0.0 else "False",

            no_done_for_positive="False",
            no_done_for_negative="False",
            reward_0_for_positive="False",
            reward_0_for_negative="False",
            reward_n2_for_intervention="False",
            reward_1_for_all="False",
            use_weighted_reward="False",
            remove_negative="False",
            adaptive_batch_size="False",
            with_human_proxy_value_loss="False",
            with_agent_proxy_value_loss="False",
            simple_batch="False",

            # ── Logging ───────────────────────────────────────────────────────
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=args.seed,
        )

    # ── Wire model → wrapper (so wrapper can push to preference_buffer) ───────
    ppl_wrapper.model = model

    # ── Guard PPL.train() against an empty preference buffer ─────────────────
    # PPL.train() immediately samples preference_buffer. We monkey-patch a thin
    # guard: fall back to BC-only until min_pref_pairs are collected, then
    # restore the configured only_bc_loss flag.
    _orig_train    = model.train.__func__
    min_pref_pairs = args.min_pref_pairs

    def _guarded_train(self_ref, gradient_steps, batch_size=100):
        if self_ref.preference_buffer.pos < min_pref_pairs:
            _saved = self_ref.extra_config.get("only_bc_loss", False)
            self_ref.extra_config["only_bc_loss"] = True
            _orig_train(self_ref, gradient_steps, batch_size)
            self_ref.extra_config["only_bc_loss"] = _saved
        else:
            _orig_train(self_ref, gradient_steps, batch_size)

    model.train = types.MethodType(_guarded_train, model)

    # ── Build callbacks ───────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppl_model",
        verbose=1,
    )

    cb_list = [checkpoint_cb]

    if args.wandb:
        ppl_cb = PPLWandbCallback(
            env_wrapper=ppl_wrapper,
            log_interval=args.log_interval,
            verbose=1,
        )
        cb_list.append(ppl_cb)
    else:
        # Even without W&B, print a lightweight console summary on the same interval
        class _ConsoleLogCallback(BaseCallback := __import__(
                "ppl.sb3.common.callbacks", fromlist=["BaseCallback"]
            ).BaseCallback):
            def __init__(self, wrapper, interval):
                super().__init__(verbose=1)
                self._w = wrapper
                self._interval = interval

            def _on_step(self):
                if self.num_timesteps % self._interval == 0:
                    sr = (
                        float(np.mean(list(self.model.ep_success_buffer)))
                        if getattr(self.model, "ep_success_buffer", None)
                           and len(self.model.ep_success_buffer) > 0
                        else float("nan")
                    )
                    T  = max(self.num_timesteps, 1)
                    print(
                        f"[PPL] step={self.num_timesteps:>7d} | "
                        f"human_steps={self._w._human_steps:>6d} "
                        f"({self._w._human_steps / T * 100:.1f}%) | "
                        f"pref_pairs={self.model.preference_buffer.pos:>5d} | "
                        f"success_rate={sr:.3f}"
                    )
                return True

        cb_list.append(_ConsoleLogCallback(ppl_wrapper, args.log_interval))

    callbacks = CallbackList(cb_list)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting online PPL training …\n")
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        tb_log_name=run_name,
        reset_num_timesteps=(args.ckpt == ""),
        save_buffer=False,
    )

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = log_dir / "final_ppl_model"
    model.save(str(final_path))

    if args.wandb:
        import wandb as _wandb
        _wandb.save(str(final_path) + ".zip")
        _wandb.finish()

    print("\n" + "=" * 70)
    print("  PPL training complete.")
    print(f"  Final model : {final_path}.zip")
    print("=" * 70)


if __name__ == "__main__":
    main()