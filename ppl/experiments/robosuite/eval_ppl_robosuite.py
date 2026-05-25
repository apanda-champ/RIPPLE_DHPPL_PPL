"""
eval_ppl_robosuite.py  –  Visualise a trained PPL agent on Robosuite Wipe.

Loads a saved PPL checkpoint and runs the agent in the Wipe environment
with on-screen rendering so you can watch it perform the wiping task.

Usage
-----
    # Watch 5 episodes with the on-screen MuJoCo viewer
    python eval_ppl_robosuite.py \
        --checkpoint runs/PPL_Robosuite/<trial>/models/ppl_robosuite_<N>_steps.zip \
        --n_episodes 5

    # Save an off-screen video instead of on-screen rendering
    python eval_ppl_robosuite.py \
        --checkpoint runs/PPL_Robosuite/<trial>/models/ppl_robosuite_<N>_steps.zip \
        --n_episodes 5 \
        --offscreen \
        --video_path eval_video.mp4

    # Run without any rendering (just print stats)
    python eval_ppl_robosuite.py \
        --checkpoint runs/PPL_Robosuite/<trial>/models/ppl_robosuite_<N>_steps.zip \
        --n_episodes 20 \
        --no_render
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from ppl.experiments.robosuite.robosuite_base_env import (
    RobosuiteBaseEnv,
    PROPORTION_WIPED_SUCCESS_THRESHOLD,
)
from ppl.ppl import PPL
from ppl.sb3.common.save_util import load_from_zip_file
from ppl.sb3.td3.policies import TD3Policy


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate and visualise a trained PPL agent on Robosuite Wipe."
    )
    p.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to the PPL .zip checkpoint saved by CheckpointCallback."
    )
    p.add_argument(
        "--n_episodes", default=5, type=int,
        help="Number of episodes to run."
    )
    p.add_argument(
        "--seed", default=42, type=int,
        help="Random seed for reproducibility."
    )
    p.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic actions from the policy (default: True)."
    )
    p.add_argument(
        "--no_render", action="store_true",
        help="Disable all rendering (stats-only mode)."
    )
    p.add_argument(
        "--offscreen", action="store_true",
        help="Render off-screen and save to a video file instead of opening "
             "the MuJoCo viewer window."
    )
    p.add_argument(
        "--video_path", default="eval_ppl_robosuite.mp4", type=str,
        help="Output path for the video when --offscreen is set."
    )
    p.add_argument(
        "--camera", default="frontview", type=str,
        help="Camera name for off-screen rendering "
             "(e.g. frontview, agentview, sideview)."
    )
    p.add_argument(
        "--video_width",  default=640, type=int,
        help="Off-screen video width in pixels."
    )
    p.add_argument(
        "--video_height", default=480, type=int,
        help="Off-screen video height in pixels."
    )
    p.add_argument(
        "--render_delay", default=0.0, type=float,
        help="Seconds to sleep between steps for on-screen rendering "
             "(increase if viewer feels too fast)."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

def _make_video_writer(path: str, fps: int, width: int, height: int):
    """Return an imageio writer, or raise a clear error if imageio is missing."""
    try:
        import imageio
        writer = imageio.get_writer(path, fps=fps, macro_block_size=None)
        return writer
    except ImportError:
        raise ImportError(
            "imageio is required for --offscreen video saving.\n"
            "Install it with:  pip install imageio imageio-ffmpeg"
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, env: RobosuiteBaseEnv) -> PPL:
    """
    Load a PPL model from a CheckpointCallback .zip file.

    We reconstruct a minimal PPL instance (actor-only is sufficient for eval)
    and copy the saved parameters into it.  The extra PPL-specific kwargs
    (beta, bc_loss_weight, etc.) are set to their training defaults; they do
    not affect inference since only the actor's forward pass is used.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"[eval] Loading checkpoint: {ckpt}")

    # Build a shell model with the same architecture used during training.
    # The critic is present but unused at eval time.
    model = PPL(
        policy              = TD3Policy,
        env                 = env,
        policy_kwargs       = dict(net_arch=[256, 256]),
        learning_rate       = 1e-4,
        buffer_size         = 1000,        # Minimal — not used during eval
        learning_starts     = 0,
        batch_size          = 256,
        device              = "auto",
        verbose             = 0,
        # PPL-specific flags (must match training defaults to avoid __init__ errors)
        only_bc_loss            = "False",
        bc_loss_weight          = 1.0,
        beta                    = 0.1,
        add_bc_loss             = "True",
        use_balance_sample      = True,
        agent_data_ratio        = 1.0,
        q_value_bound           = 1.0,
        optimize_memory_usage   = False,   # Off — avoids next_obs aliasing on tiny buffer
        no_done_for_positive        = "False",
        no_done_for_negative        = "False",
        reward_0_for_positive       = "False",
        reward_0_for_negative       = "False",
        reward_n2_for_intervention  = "False",
        reward_1_for_all            = "False",
        use_weighted_reward         = "False",
        remove_negative             = "False",
        adaptive_batch_size         = "False",
        with_human_proxy_value_loss = "False",
        with_agent_proxy_value_loss = "False",
        simple_batch                = "True",
    )

    # Load saved weights into the shell model
    _data, params, _pv = load_from_zip_file(
        str(ckpt), device=model.device, print_system_info=False
    )
    model.set_parameters(params, exact_match=False, device=model.device)
    # exact_match=False: checkpoint may not contain preference_buffer /
    # human_data_buffer keys (they are in _excluded_save_params), so a
    # strict match would raise a KeyError.

    print(f"[eval] Model loaded successfully.  Device: {model.device}")
    return model


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: RobosuiteBaseEnv,
    model: PPL,
    deterministic: bool,
    render_mode: str,           # "onscreen" | "offscreen" | "none"
    video_writer=None,
    camera: str = "frontview",
    video_width: int = 640,
    video_height: int = 480,
    render_delay: float = 0.0,
) -> dict:
    """
    Run one episode and return per-episode statistics.

    Returns
    -------
    dict with keys:
        total_reward    (float)
        steps           (int)
        proportion_wiped (float)   – final value of obs[-1]
        success         (bool)     – True if proportion_wiped >= threshold OR
                                     native _check_success() fires
        native_success  (bool)     – True only if _check_success() fires
    """
    obs = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    proportion_wiped = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)

        total_reward += reward
        step_count += 1
        proportion_wiped = float(obs[-1])   # proportion_wiped is always obs[-1]

        # ---- Rendering ----
        if render_mode == "onscreen":
            env.render(mode="human")
            if render_delay > 0.0:
                time.sleep(render_delay)

        elif render_mode == "offscreen" and video_writer is not None:
            frame = env.render(mode="rgb_array")
            # Robosuite returns (H, W, C) uint8; imageio expects the same
            if frame is not None:
                video_writer.append_data(frame)

    # Final success check aligned with the updated is_success logic
    native_success  = bool(env._rs_env._check_success())
    coverage_success = proportion_wiped >= PROPORTION_WIPED_SUCCESS_THRESHOLD
    success = native_success or coverage_success

    return {
        "total_reward":     total_reward,
        "steps":            step_count,
        "proportion_wiped": proportion_wiped,
        "success":          success,
        "native_success":   native_success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)

    # ---- Determine render mode ----
    if args.no_render:
        render_mode = "none"
    elif args.offscreen:
        render_mode = "offscreen"
    else:
        render_mode = "onscreen"

    # ---- Build environment ----
    env_config = {
        "reward_shaping":          True,
        "has_renderer":            render_mode == "onscreen",
        "has_offscreen_renderer":  render_mode == "offscreen",
        "render_camera":           args.camera,
        "ignore_done":             False,
    }
    print(f"[eval] Building environment (render_mode={render_mode}) …")
    env = RobosuiteBaseEnv(config=env_config)

    # ---- Load model ----
    model = load_model(args.checkpoint, env)

    # ---- Set up video writer ----
    video_writer = None
    if render_mode == "offscreen":
        video_writer = _make_video_writer(
            args.video_path,
            fps    = env._cfg.get("control_freq", 20),
            width  = args.video_width,
            height = args.video_height,
        )
        print(f"[eval] Recording video → {args.video_path}")

    # ---- Run episodes ----
    print(
        f"\n[eval] Running {args.n_episodes} episode(s)  |  "
        f"deterministic={args.deterministic}  |  "
        f"success_threshold={PROPORTION_WIPED_SUCCESS_THRESHOLD:.0%}\n"
        f"{'─' * 70}"
    )

    results: List[dict] = []

    for ep in range(1, args.n_episodes + 1):
        ep_result = run_episode(
            env            = env,
            model          = model,
            deterministic  = args.deterministic,
            render_mode    = render_mode,
            video_writer   = video_writer,
            camera         = args.camera,
            video_width    = args.video_width,
            video_height   = args.video_height,
            render_delay   = args.render_delay,
        )
        results.append(ep_result)

        print(
            f"  Episode {ep:>3d}/{args.n_episodes}  |  "
            f"steps={ep_result['steps']:>4d}/750  |  "
            f"reward={ep_result['total_reward']:>8.2f}  |  "
            f"wiped={ep_result['proportion_wiped']:.1%}  |  "
            f"{'✓ SUCCESS' if ep_result['success'] else '✗ FAIL   '}"
            + (f"  [native]" if ep_result["native_success"] else "")
        )

    # ---- Aggregate statistics ----
    print(f"{'─' * 70}")
    success_rate    = np.mean([r["success"]          for r in results])
    mean_wiped      = np.mean([r["proportion_wiped"] for r in results])
    mean_reward     = np.mean([r["total_reward"]     for r in results])
    mean_steps      = np.mean([r["steps"]            for r in results])
    native_rate     = np.mean([r["native_success"]   for r in results])

    print(f"\n[eval] Results over {args.n_episodes} episode(s):")
    print(f"  Success rate      : {success_rate:.1%}  "
          f"(proportion_wiped >= {PROPORTION_WIPED_SUCCESS_THRESHOLD:.0%} "
          f"OR native success)")
    print(f"  Native success    : {native_rate:.1%}  "
          f"(all markers contacted)")
    print(f"  Mean wiped        : {mean_wiped:.1%}")
    print(f"  Mean total reward : {mean_reward:.2f}")
    print(f"  Mean steps/ep     : {mean_steps:.1f}  (max 750)")

    # ---- Cleanup ----
    if video_writer is not None:
        video_writer.close()
        print(f"\n[eval] Video saved → {args.video_path}")

    env.close()


if __name__ == "__main__":
    main()
