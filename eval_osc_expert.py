"""
eval_osc_expert.py  –  Evaluate the trained OSC expert on the Robosuite Wipe task.
"""

import argparse
import numpy as np
from ppl.experiments.robosuite.osc_expert import get_osc_expert
from ppl.experiments.robosuite.robosuite_base_env import RobosuiteBaseEnv

# Success threshold for proportion_wiped
# An episode is successful if ALL markers are wiped OR proportion_wiped >= this value
PROPORTION_WIPED_SUCCESS_THRESHOLD = 0.94


def evaluate_expert(
    checkpoint_path: str,
    n_episodes: int = 50,
    max_steps: int = 750,
    render: bool = False,
):
    print(f"Loading environment and expert from: {checkpoint_path}...")

    env = RobosuiteBaseEnv(
        config={
            "has_renderer": render,
            "has_offscreen_renderer": False,
            "reward_shaping": True,
            "horizon": max_steps,
        }
    )

    expert = get_osc_expert(checkpoint_path)

    success_count = 0
    total_rewards = []
    total_wiped = []

    print(f"\nRunning {n_episodes} evaluation episodes (max {max_steps} steps each)...")
    print(f"Success criterion: all 50 markers wiped OR proportion_wiped >= {PROPORTION_WIPED_SUCCESS_THRESHOLD * 100:.0f}%")
    print("-" * 60)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_success = False
        ep_reward = 0.0
        step_count = 0

        while not done and step_count < max_steps:
            action, _ = expert.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward

            proportion_wiped = float(obs[-1])

            # Success criterion:
            # (a) Robosuite native: all 50 markers physically contacted
            # (b) Coverage: proportion_wiped >= 0.94 (47+ out of 50 markers)
            if env._rs_env._check_success() or proportion_wiped >= PROPORTION_WIPED_SUCCESS_THRESHOLD:
                ep_success = True
                done = True

            step_count += 1

        if ep_success:
            success_count += 1

        proportion_wiped = float(obs[-1])
        total_rewards.append(ep_reward)
        total_wiped.append(proportion_wiped)

        status = "SUCCESS" if ep_success else "FAILED"
        print(
            f"Episode {ep + 1:02d}: {status} | "
            f"Steps: {step_count:4d} | "
            f"Reward: {ep_reward:7.2f} | "
            f"Wiped: {proportion_wiped * 100:5.1f}%"
        )

    success_rate = (success_count / n_episodes) * 100

    print("\n" + "=" * 60)
    print(f"  FINAL SUCCESS RATE : {success_rate:.1f}%  ({success_count}/{n_episodes})")
    print(f"  MEAN EPISODE REWARD: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  MEAN WIPED         : {np.mean(total_wiped) * 100:.1f}% ± {np.std(total_wiped) * 100:.1f}%")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the OSC expert on Robosuite Wipe.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ppl/experiments/robosuite/osc_expert.zip",
        help="Path to the trained expert checkpoint.",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=750,
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the simulation (slow; for visual inspection only).",
    )
    args = parser.parse_args()

    evaluate_expert(
        checkpoint_path=args.checkpoint,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        render=args.render,
    )
