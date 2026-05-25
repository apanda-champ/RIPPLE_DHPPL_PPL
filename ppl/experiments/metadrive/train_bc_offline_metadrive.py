"""
Behaviour Cloning (BC) training script for MetaDrive — Offline / Epoch-based.

Pipeline
--------
1. Run the PPO expert for 10,000 steps  →  build a fixed demonstration dataset.
2. Train the student for 10 epochs on the dataset  →  evaluate for 50 episodes.
3. Repeat step 2 until 100 total epochs are complete  (10 evaluation checkpoints).

Structure
---------
  Phase 1 (one-time) : Expert collects 10,000 (obs, action) pairs.
  Loop (×10)         :
      Train 10 epochs on the fixed dataset  →  150 gradient updates per epoch
      Evaluate student for 50 episodes on held-out maps
      Log: success_rate, route_completion, mean_reward to terminal + W&B

Usage
-----
    python train_bc_metadrive.py \
        --wandb --wandb_project ahan-ppl \
        --collect_steps 10000 \
        --train_epochs 100 \
        --eval_freq 10 \
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
from torch.utils.data import DataLoader, TensorDataset

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
# Expert data collection
# ---------------------------------------------------------------------------

def collect_expert_demonstrations(expert_policy, num_steps, env_config,
                                  trial_dir, deterministic=False, verbose=True):
    """
    Roll the expert in DrivingEnv for num_steps steps.
    Returns obs_array (N, obs_dim) and act_array (N, act_dim) as numpy arrays.

    NOTE: Opens and closes its own DrivingEnv — no engine conflict as long as
    no other env is open at the same time.
    """
    env = DrivingEnv(config=env_config)
    env = Monitor(env=env, filename=str(trial_dir))

    observations, actions = [], []
    obs      = env.reset()
    episode  = 0
    step     = 0

    while step < num_steps:
        action, _ = expert_policy.predict(obs, deterministic=deterministic)
        observations.append(obs.copy())
        actions.append(action.copy())

        obs, _, done, _ = env.step(action)
        step += 1

        if done:
            episode += 1
            obs = env.reset()
            if verbose and episode % 5 == 0:
                print(f"  [Collection] Episode {episode} | Steps: {step}/{num_steps}")

    env.close()

    print(f"[Collection] Done — {step} steps across {episode} episodes collected.")
    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.float32)


# ---------------------------------------------------------------------------
# BC policy trainer
# ---------------------------------------------------------------------------

class BCTrainer:
    """
    Wraps an ActorCriticPolicy and trains it with behaviour cloning loss.
    BC loss = MSE between the student action-distribution mean and the
    expert action (equivalent to MLE under a Gaussian policy).
    """

    def __init__(self, observation_space, action_space,
                 net_arch=None, learning_rate=1e-4, device="auto"):
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

    def train_epoch(self, obs_tensor, act_tensor, batch_size):
        """
        One full pass over the demonstration dataset.
        Returns mean BC loss and mean log-probability for the epoch.
        """
        dataset = TensorDataset(obs_tensor, act_tensor)
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, drop_last=False)

        epoch_losses, epoch_logprobs = [], []

        self.policy.set_training_mode(True)
        for obs_batch, act_batch in loader:
            self.optimizer.zero_grad()

            distribution      = self.policy.get_distribution(obs_batch)
            predicted_actions = distribution.distribution.mean  # (B, act_dim)

            bc_loss = nn.functional.mse_loss(predicted_actions, act_batch)
            bc_loss.backward()

            # Gradient clipping — same as PPO expert (max_grad_norm=10)
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
            self.optimizer.step()

            with torch.no_grad():
                log_prob = distribution.log_prob(act_batch).mean().item()

            epoch_losses.append(bc_loss.item())
            epoch_logprobs.append(log_prob)

        self.policy.set_training_mode(False)
        return float(np.mean(epoch_losses)), float(np.mean(epoch_logprobs))

    def save(self, path):
        torch.save({
            "policy_state_dict":    self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"[BC] Checkpoint saved → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[BC] Checkpoint loaded ← {path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(student_policy, eval_env_config, trial_dir, n_episodes=50):
    """
    Run the student policy for n_episodes on held-out maps (deterministic).

    Metrics returned
    ----------------
    success_rate     : fraction of episodes where the car reached the destination
    route_completion : mean fraction of the route completed per episode
    mean_reward      : mean episodic return
    std_reward       : std of episodic return
    crash_rate       : fraction of episodes ending in a crash
    out_of_road_rate : fraction of episodes ending out-of-road
    mean_length      : mean episode length in steps

    NOTE: MetaDrive only allows one engine at a time.
    All other envs must be closed before calling this function.
    """
    env = DrivingEnv(config=eval_env_config)
    env = Monitor(env=env, filename=str(trial_dir))

    rewards, lengths   = [], []
    successes, crashes = [], []
    out_of_roads       = []
    route_completions  = []

    student_policy.set_training_mode(False)

    for _ in range(n_episodes):
        obs                          = env.reset()
        ep_reward, ep_length, done   = 0.0, 0, False

        while not done:
            action, _ = student_policy.predict(obs, deterministic=True)
            obs, reward, done, info  = env.step(action)
            ep_reward  += reward
            ep_length  += 1

        rewards.append(ep_reward)
        lengths.append(ep_length)
        successes.append(float(info.get("arrive_dest",     0)))
        crashes.append(float(info.get("crash",             0)))
        out_of_roads.append(float(info.get("out_of_road",  0)))
        route_completions.append(float(info.get("route_completion", 0.0)))

    env.close()

    return {
        "success_rate":     float(np.mean(successes)),
        "route_completion": float(np.mean(route_completions)),
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "crash_rate":       float(np.mean(crashes)),
        "out_of_road_rate": float(np.mean(out_of_roads)),
        "mean_length":      float(np.mean(lengths)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline Behaviour Cloning in MetaDrive — epoch-based training"
    )

    # Experiment
    parser.add_argument("--exp_name",        default="bc_metadrive", type=str)
    parser.add_argument("--seed",            default=0,              type=int)
    parser.add_argument("--ckpt",            default="",             type=str,
                        help="Path to a BC checkpoint to resume from.")

    # Data collection
    parser.add_argument("--collect_steps",   default=10_000,         type=int,
                        help="Expert steps to collect (default: 10,000).")
    parser.add_argument("--expert_deterministic", action="store_true",
                        help="Use deterministic expert actions (default: stochastic).")

    # Training
    parser.add_argument("--train_epochs",    default=100,            type=int,
                        help="Total training epochs (default: 100).")
    parser.add_argument("--eval_freq",       default=10,             type=int,
                        help="Evaluate every N epochs (default: 10).")
    parser.add_argument("--n_eval_episodes", default=50,             type=int,
                        help="Episodes per evaluation (default: 50).")
    parser.add_argument("--batch_size",      default=256,            type=int)
    parser.add_argument("--lr",              default=1e-4,           type=float,
                        help="Learning rate (default: 1e-4).")
    parser.add_argument("--save_freq",       default=10,             type=int,
                        help="Save checkpoint every N epochs (default: 10).")

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
    trial_name = "{}_{}_{}".format(experiment_batch_name,
                                   get_time_str(), uuid.uuid4().hex[:8])
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
    # Load PPO expert
    # -----------------------------------------------------------------------
    print("\n[BC] Loading PPO expert ...")
    expert_policy = get_expert()
    expert_policy.set_training_mode(False)
    print("[BC] Expert loaded.\n")

    # -----------------------------------------------------------------------
    # Phase 1 — Collect 10,000 expert steps (one-time, fixed dataset)
    # -----------------------------------------------------------------------
    print(f"[BC] Collecting {args.collect_steps} expert steps ...")
    obs_np, act_np = collect_expert_demonstrations(
        expert_policy  = expert_policy,
        num_steps      = args.collect_steps,
        env_config     = train_env_config,
        trial_dir      = trial_dir,
        deterministic  = args.expert_deterministic,
        verbose        = True,
    )
    print(f"[BC] Dataset: obs {obs_np.shape}  |  actions {act_np.shape}\n")

    # -----------------------------------------------------------------------
    # Build student (same architecture as PPO expert / PPL)
    # -----------------------------------------------------------------------
    # Use a temporary env just to read observation/action spaces
    _tmp_env          = DrivingEnv(config=train_env_config)
    observation_space = _tmp_env.observation_space
    action_space      = _tmp_env.action_space
    _tmp_env.close()

    bc_trainer = BCTrainer(
        observation_space = observation_space,
        action_space      = action_space,
        net_arch          = [256, 256],   # same as PPL / PPO expert
        learning_rate     = args.lr,
        device            = "auto",
    )

    if args.ckpt:
        bc_trainer.load(Path(args.ckpt))

    # Convert dataset to tensors once — training is purely supervised
    obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=bc_trainer.device)
    act_tensor = torch.tensor(act_np, dtype=torch.float32, device=bc_trainer.device)

    # -----------------------------------------------------------------------
    # W&B
    # -----------------------------------------------------------------------
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project = args.wandb_project or "bc_metadrive",
                entity  = args.wandb_team    or None,
                name    = trial_name,
                config  = vars(args),
            )
            print("[BC] W&B initialised.")
        except ImportError:
            print("[BC][WARNING] wandb not installed — disabling.")
            use_wandb = False

    # -----------------------------------------------------------------------
    # Phase 2 — Epoch-based training with periodic evaluation
    #
    #   Total epochs : 100
    #   Eval every   : 10 epochs  →  10 evaluation checkpoints
    #
    #   Each epoch   : full pass over 10,000 samples in batches of 256
    #                  ≈ 39 gradient updates per epoch
    # -----------------------------------------------------------------------
    n_eval_checkpoints = args.train_epochs // args.eval_freq   # = 10
    best_success_rate  = -1.0
    stats_history      = defaultdict(list)

    print(f"\n[BC] Starting epoch-based training.")
    print(f"     collect_steps  = {args.collect_steps}")
    print(f"     train_epochs   = {args.train_epochs}")
    print(f"     eval_freq      = {args.eval_freq}  epochs")
    print(f"     n_checkpoints  = {n_eval_checkpoints}")
    print(f"     n_eval_ep      = {args.n_eval_episodes}")
    print(f"     batch_size     = {args.batch_size}")
    print(f"     learning_rate  = {args.lr}\n")

    for epoch in range(1, args.train_epochs + 1):

        # -------------------------------------------------------------------
        # Train one epoch over the full dataset
        # -------------------------------------------------------------------
        bc_loss, log_prob = bc_trainer.train_epoch(
            obs_tensor, act_tensor, args.batch_size
        )

        stats_history["bc_loss"].append(bc_loss)
        stats_history["log_prob"].append(log_prob)

        print(
            f"Epoch [{epoch:3d}/{args.train_epochs}]  "
            f"BC loss: {bc_loss:.5f}  |  log_prob: {log_prob:.4f}"
        )

        # -------------------------------------------------------------------
        # Evaluate every eval_freq epochs
        # -------------------------------------------------------------------
        if epoch % args.eval_freq == 0:

            print(f"  [Eval] Running {args.n_eval_episodes} episodes ...")
            eval_stats = evaluate_policy(
                student_policy  = bc_trainer.policy,
                eval_env_config = eval_env_config,
                trial_dir       = trial_dir,
                n_episodes      = args.n_eval_episodes,
            )

            stats_history["eval/success_rate"].append(eval_stats["success_rate"])
            stats_history["eval/route_completion"].append(eval_stats["route_completion"])
            stats_history["eval/mean_reward"].append(eval_stats["mean_reward"])
            stats_history["eval/crash_rate"].append(eval_stats["crash_rate"])
            stats_history["eval/out_of_road_rate"].append(eval_stats["out_of_road_rate"])
            stats_history["eval/mean_length"].append(eval_stats["mean_length"])

            print(
                f"  [Eval] "
                f"success: {eval_stats['success_rate']:.2%}  |  "
                f"route_completion: {eval_stats['route_completion']:.2%}  |  "
                f"mean_reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}  |  "
                f"crash: {eval_stats['crash_rate']:.2%}  |  "
                f"OOR: {eval_stats['out_of_road_rate']:.2%}  |  "
                f"len: {eval_stats['mean_length']:.1f}"
            )

            # Save best model based on success rate
            if eval_stats["success_rate"] > best_success_rate:
                best_success_rate = eval_stats["success_rate"]
                bc_trainer.save(models_dir / "best_model.pt")
                print(f"  [Eval] ★ New best success rate: {best_success_rate:.2%}")

            # W&B logging
            if use_wandb:
                wandb.log({
                    "epoch":    epoch,
                    "bc_loss":  bc_loss,
                    "log_prob": log_prob,
                    **{f"eval/{k}": v for k, v in eval_stats.items()},
                })

        else:
            # Log training metrics every epoch even without eval
            if use_wandb:
                wandb.log({
                    "epoch":    epoch,
                    "bc_loss":  bc_loss,
                    "log_prob": log_prob,
                })

        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            bc_trainer.save(models_dir / f"bc_model_epoch_{epoch:04d}.pt")

    # -----------------------------------------------------------------------
    # Final save
    # -----------------------------------------------------------------------
    bc_trainer.save(models_dir / "final_model.pt")

    print(f"\n[BC] Training complete.")
    print(f"     Best success rate : {best_success_rate:.2%}")
    print(f"     Artefacts saved to: {trial_dir}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
