"""
Behaviour Cloning (BC) training script for MetaDrive — Online / Step-based.

Mirrors the PPL training loop exactly:
  - Collect 150 expert steps  →  train BC on the growing buffer  →  evaluate 50 episodes
  - Repeat until 10,000 total steps have been collected.

Total iterations : 10,000 / 150 = ~66
Each iteration   : 150 expert steps collected + 150 gradient updates + 50-episode eval

Usage
-----
    python train_bc_metadrive.py \
        --wandb --wandb_project ahan-ppl \
        --total_steps 10000 \
        --eval_freq 150 \
        --n_eval_episodes 50 \
        --seed 0
"""

import argparse
import os
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pathlib
FOLDER_PATH = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Imports from the PPL codebase
# ---------------------------------------------------------------------------
from ppl.experiments.metadrive.driving_env import DrivingEnv
from ppl.experiments.metadrive.experttakeover_env import get_expert
from ppl.sb3.common.monitor import Monitor
from ppl.sb3.ppo.policies import ActorCriticPolicy
from ppl.utils.utils import get_time_str


# ---------------------------------------------------------------------------
# Growing demonstration buffer  (acts like a replay buffer)
# ---------------------------------------------------------------------------

class DemonstrationBuffer:
    """
    A growing ring buffer of (obs, expert_action) pairs.
    New data is always appended; once buffer_size is reached the oldest
    entries are overwritten (FIFO), matching PPL's replay buffer behaviour.
    """

    def __init__(self, buffer_size: int = 50_000):
        self.buffer_size = buffer_size
        self.observations = []
        self.actions = []
        self._ptr = 0
        self._full = False

    def add(self, obs: np.ndarray, action: np.ndarray):
        if len(self.observations) < self.buffer_size:
            self.observations.append(obs.copy())
            self.actions.append(action.copy())
        else:
            self.observations[self._ptr] = obs.copy()
            self.actions[self._ptr]      = action.copy()
            self._ptr = (self._ptr + 1) % self.buffer_size
            self._full = True

    def __len__(self):
        return len(self.observations)

    def sample(self, batch_size: int, device: torch.device):
        """Sample a random mini-batch from the buffer."""
        indices  = np.random.randint(0, len(self), size=batch_size)
        obs_batch = torch.tensor(
            np.array([self.observations[i] for i in indices]),
            dtype=torch.float32, device=device
        )
        act_batch = torch.tensor(
            np.array([self.actions[i] for i in indices]),
            dtype=torch.float32, device=device
        )
        return obs_batch, act_batch


# ---------------------------------------------------------------------------
# BC policy trainer
# ---------------------------------------------------------------------------

class BCTrainer:
    """
    Wraps an ActorCriticPolicy and trains it with behaviour cloning loss.
    BC loss = MSE between the student action-distribution mean and the
    expert action (equivalent to MLE under a Gaussian policy).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch=None,
        learning_rate: float = 1e-4,
        device: str = "auto",
    ):
        if net_arch is None:
            net_arch = [256, 256]

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Same architecture as the PPO expert and PPL student
        self.policy = ActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: learning_rate,
            net_arch=net_arch,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def update(self, obs: torch.Tensor, expert_actions: torch.Tensor):
        """
        One gradient update on a single mini-batch.

        Returns
        -------
        bc_loss  : float
        log_prob : float  (log-likelihood of expert actions, for diagnostics)
        """
        self.policy.set_training_mode(True)
        self.optimizer.zero_grad()

        # Forward pass → action distribution
        distribution = self.policy.get_distribution(obs)

        # Student action mean  (mode of DiagGaussian == mean)
        predicted_actions = distribution.distribution.mean   # (B, action_dim)

        # BC loss: MSE between student mean and expert action
        bc_loss = nn.functional.mse_loss(predicted_actions, expert_actions)
        bc_loss.backward()

        # Gradient clipping — same as PPO expert (max_grad_norm=10)
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.policy.set_training_mode(False)

        with torch.no_grad():
            log_prob = distribution.log_prob(expert_actions).mean().item()

        return bc_loss.item(), log_prob

    def save(self, path: Path):
        torch.save(
            {
                "policy_state_dict":    self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"[BC] Checkpoint saved → {path}")

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[BC] Checkpoint loaded ← {path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(student_policy, eval_env_config, trial_dir, n_episodes: int = 50):
    """
    Run the student policy for n_episodes in the eval env (deterministic).

    NOTE: MetaDrive only allows one engine instance at a time.
    The caller must close the training env BEFORE calling this function,
    and reopen it AFTER. This function opens and closes its own eval env.

    Returns success_rate, crash_rate, out_of_road_rate, mean_reward,
    std_reward, mean_length.
    """
    # Open a fresh eval env (training env must already be closed)
    env = DrivingEnv(config=eval_env_config)
    env = Monitor(env=env, filename=str(trial_dir))

    rewards, lengths, successes, crashes, out_of_roads, route_completions = [], [], [], [], [], []

    student_policy.set_training_mode(False)

    for _ in range(n_episodes):
        obs = env.reset()
        ep_reward, ep_length, done = 0.0, 0, False
        while not done:
            action, _ = student_policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_length += 1
        rewards.append(ep_reward)
        lengths.append(ep_length)
        successes.append(float(info.get("arrive_dest", 0)))
        crashes.append(float(info.get("crash", 0)))
        out_of_roads.append(float(info.get("out_of_road", 0)))
        # route_completion is in info at every step; at episode end it holds
        # the final completion fraction (same key PPL's callback uses)
        route_completions.append(float(info.get("route_completion", 0.0)))

    env.close()

    return {
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "success_rate":     float(np.mean(successes)),
        "crash_rate":       float(np.mean(crashes)),
        "out_of_road_rate": float(np.mean(out_of_roads)),
        "mean_length":      float(np.mean(lengths)),
        "route_completion": float(np.mean(route_completions)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Online Behaviour Cloning in MetaDrive (PPL-style step loop)"
    )

    # Experiment
    parser.add_argument("--exp_name",        default="bc_metadrive", type=str)
    parser.add_argument("--seed",            default=0,              type=int)
    parser.add_argument("--ckpt",            default="",             type=str,
                        help="Path to a BC checkpoint to resume from.")

    # Loop config — mirrors PPL defaults exactly
    parser.add_argument("--total_steps",     default=10_000,         type=int,
                        help="Total expert steps to collect (default: 10,000).")
    parser.add_argument("--eval_freq",       default=150,            type=int,
                        help="Steps per interval before evaluation (default: 150).")
    parser.add_argument("--n_eval_episodes", default=50,             type=int,
                        help="Episodes per evaluation (default: 50).")
    parser.add_argument("--save_freq",       default=150,            type=int,
                        help="Save a checkpoint every N steps (default: 150).")

    # BC hyper-parameters
    parser.add_argument("--batch_size",      default=256,            type=int)
    parser.add_argument("--lr",              default=1e-4,           type=float,
                        help="Learning rate — same default as PPL (1e-4).")
    parser.add_argument("--learning_starts", default=150,            type=int,
                        help="Start gradient updates only after this many steps "
                             "are in the buffer (default: 150).")
    parser.add_argument("--buffer_size",     default=50_000,         type=int,
                        help="Max buffer capacity — same as PPL (50,000).")

    # Expert
    parser.add_argument("--expert_deterministic", action="store_true",
                        help="Use deterministic expert actions (default: stochastic).")

    # W&B
    parser.add_argument("--wandb",           action="store_true")
    parser.add_argument("--wandb_project",   default="",             type=str)
    parser.add_argument("--wandb_team",      default="",             type=str)

    # Debug
    parser.add_argument("--toy_env",         action="store_true",
                        help="Toy environment for quick debugging.")

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    experiment_batch_name = "BC"
    trial_name = "{}_{}_{}".format(experiment_batch_name, get_time_str(), uuid.uuid4().hex[:8])
    print(f"[BC] Trial name : {trial_name}")

    log_dir        = FOLDER_PATH.parent.parent
    experiment_dir = Path(log_dir) / "runs" / experiment_batch_name
    trial_dir      = experiment_dir / trial_name
    models_dir     = trial_dir / "models"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir,      exist_ok=False)
    os.makedirs(models_dir,     exist_ok=True)
    print(f"[BC] Logging to : {trial_dir}")

    # -----------------------------------------------------------------------
    # Environment configs
    # -----------------------------------------------------------------------
    train_env_config = dict(
        num_scenarios=50,
        start_seed=100,           # same training maps as PPL
        traffic_density=0.06,
        use_render=False,
        manual_control=False,
        out_of_route_done=True,
        horizon=1500,
    )
    eval_env_config = dict(
        num_scenarios=50,
        start_seed=1000,          # held-out maps, same as PPL
        traffic_density=0.06,
        use_render=False,
        manual_control=False,
        out_of_route_done=True,
        horizon=1500,
    )

    if args.toy_env:
        for cfg in (train_env_config, eval_env_config):
            cfg.update(num_scenarios=1, traffic_density=0.0, map="COT")

    # -----------------------------------------------------------------------
    # Load PPO expert  (identical to ExpertTakeoverEnv.get_expert())
    # -----------------------------------------------------------------------
    print("\n[BC] Loading PPO expert …")
    expert_policy = get_expert()
    expert_policy.set_training_mode(False)
    print("[BC] Expert loaded.\n")

    # -----------------------------------------------------------------------
    # Build student BC trainer  (same architecture as expert)
    # -----------------------------------------------------------------------
    _tmp_env          = DrivingEnv(config=train_env_config)
    observation_space = _tmp_env.observation_space
    action_space      = _tmp_env.action_space
    _tmp_env.close()

    bc_trainer = BCTrainer(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=[256, 256],      # same as PPL / PPO expert
        learning_rate=args.lr,
        device="auto",
    )

    if args.ckpt:
        bc_trainer.load(Path(args.ckpt))

    # -----------------------------------------------------------------------
    # Growing replay-style buffer
    # -----------------------------------------------------------------------
    buffer = DemonstrationBuffer(buffer_size=args.buffer_size)

    # -----------------------------------------------------------------------
    # Training environment  (stays open for the whole run)
    # -----------------------------------------------------------------------
    train_env = DrivingEnv(config=train_env_config)
    train_env = Monitor(env=train_env, filename=str(trial_dir))

    # -----------------------------------------------------------------------
    # W&B
    # -----------------------------------------------------------------------
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project or "bc_metadrive",
                entity=args.wandb_team    or None,
                name=trial_name,
                config=vars(args),
            )
            print("[BC] W&B initialised.")
        except ImportError:
            print("[BC][WARNING] wandb not installed — disabling W&B logging.")
            use_wandb = False

    # -----------------------------------------------------------------------
    # Training loop
    #
    #   10,000 total steps / 150 steps per interval = 66 intervals
    #
    #   Each interval:
    #     Phase 1 — collect 150 expert steps into the growing buffer
    #     Phase 2 — perform 150 gradient updates from the buffer
    #     Phase 3 — evaluate student for 50 episodes
    # -----------------------------------------------------------------------
    n_intervals          = args.total_steps // args.eval_freq
    total_steps_collected = 0
    best_success_rate     = -1.0
    stats_history         = defaultdict(list)

    obs = train_env.reset()

    print(f"\n[BC] Starting online training loop.")
    print(f"     total_steps    = {args.total_steps}")
    print(f"     eval_freq      = {args.eval_freq}  steps per interval")
    print(f"     n_intervals    = {n_intervals}")
    print(f"     n_eval_ep      = {args.n_eval_episodes}")
    print(f"     batch_size     = {args.batch_size}")
    print(f"     learning_rate  = {args.lr}")
    print(f"     learning_starts= {args.learning_starts}\n")

    for interval in range(1, n_intervals + 1):

        # ===================================================================
        # PHASE 1 — Collect 150 expert steps into the growing buffer
        # ===================================================================
        for _ in range(args.eval_freq):
            expert_action, _ = expert_policy.predict(
                obs, deterministic=args.expert_deterministic
            )
            buffer.add(obs, expert_action)

            obs, _, done, _ = train_env.step(expert_action)
            total_steps_collected += 1

            if done:
                obs = train_env.reset()

        # ===================================================================
        # PHASE 2 — 150 gradient updates  (train_freq=(1,"step") like PPL)
        #           Only start once the buffer has enough samples.
        # ===================================================================
        interval_losses, interval_logprobs = [], []

        if len(buffer) >= args.learning_starts:
            for _ in range(args.eval_freq):
                obs_batch, act_batch = buffer.sample(args.batch_size, bc_trainer.device)
                loss, log_prob       = bc_trainer.update(obs_batch, act_batch)
                interval_losses.append(loss)
                interval_logprobs.append(log_prob)

        mean_loss    = float(np.mean(interval_losses))    if interval_losses    else float("nan")
        mean_logprob = float(np.mean(interval_logprobs))  if interval_logprobs  else float("nan")

        stats_history["bc_loss"].append(mean_loss)
        stats_history["log_prob"].append(mean_logprob)

        print(
            f"Interval [{interval:3d}/{n_intervals}] "
            f"Steps: {total_steps_collected:6d}/{args.total_steps}  |  "
            f"Buffer: {len(buffer):6d}  |  "
            f"BC loss: {mean_loss:.5f}  |  "
            f"log_prob: {mean_logprob:.4f}"
        )

        # ===================================================================
        # PHASE 3 — Evaluate student for 50 episodes
        #
        # MetaDrive only supports one engine at a time, so we must:
        #   1. Close the training env
        #   2. Open eval env, run evaluation, close eval env
        #   3. Reopen the training env and reset obs
        # ===================================================================
        print(f"  [Eval] Running {args.n_eval_episodes} episodes …")
        train_env.close()

        eval_stats = evaluate_policy(
            student_policy=bc_trainer.policy,
            eval_env_config=eval_env_config,
            trial_dir=trial_dir,
            n_episodes=args.n_eval_episodes,
        )

        # Reopen training env and get a fresh obs
        train_env = DrivingEnv(config=train_env_config)
        train_env = Monitor(env=train_env, filename=str(trial_dir))
        obs = train_env.reset()

        stats_history["eval/mean_reward"].append(eval_stats["mean_reward"])
        stats_history["eval/success_rate"].append(eval_stats["success_rate"])
        stats_history["eval/crash_rate"].append(eval_stats["crash_rate"])
        stats_history["eval/out_of_road_rate"].append(eval_stats["out_of_road_rate"])
        stats_history["eval/mean_length"].append(eval_stats["mean_length"])
        stats_history["eval/route_completion"].append(eval_stats["route_completion"])

        print(
            f"  [Eval] reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}  |  "
            f"success: {eval_stats['success_rate']:.2%}  |  "
            f"route_completion: {eval_stats['route_completion']:.2%}  |  "
            f"crash: {eval_stats['crash_rate']:.2%}  |  "
            f"OOR: {eval_stats['out_of_road_rate']:.2%}  |  "
            f"len: {eval_stats['mean_length']:.1f}"
        )

        # Save best model
        if eval_stats["success_rate"] > best_success_rate:
            best_success_rate = eval_stats["success_rate"]
            bc_trainer.save(models_dir / "best_model.pt")
            print(f"  [Eval] ★ New best success rate: {best_success_rate:.2%}")

        # Periodic checkpoint
        if total_steps_collected % args.save_freq == 0:
            bc_trainer.save(models_dir / f"bc_model_step_{total_steps_collected:06d}.pt")

        # W&B logging
        if use_wandb:
            wandb.log({
                "step":        total_steps_collected,
                "bc_loss":     mean_loss,
                "log_prob":    mean_logprob,
                "buffer_size": len(buffer),
                **{f"eval/{k}": v for k, v in eval_stats.items()},
            })

    # -----------------------------------------------------------------------
    # Final save and cleanup
    # -----------------------------------------------------------------------
    bc_trainer.save(models_dir / "final_model.pt")
    try:
        train_env.close()
    except Exception:
        pass  # env may already be closed after the last eval cycle

    print(f"\n[BC] Training complete.")
    print(f"     Total steps collected : {total_steps_collected}")
    print(f"     Best success rate     : {best_success_rate:.2%}")
    print(f"     All artefacts saved to: {trial_dir}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
