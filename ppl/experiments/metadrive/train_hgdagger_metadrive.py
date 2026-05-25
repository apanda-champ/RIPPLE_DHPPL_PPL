import argparse
import os
import uuid
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from ppl.experiments.metadrive.experttakeover_env import ExpertTakeoverEnv
from ppl.ppl import PVPTD3
from ppl.sb3.common.buffers import ReplayBuffer
from ppl.sb3.common.callbacks import CallbackList, CheckpointCallback
from ppl.sb3.common.monitor import Monitor
from ppl.sb3.common.vec_env import SubprocVecEnv
from ppl.sb3.common.wandb_callback import WandbCallback
from ppl.sb3.haco import HACOReplayBuffer
from ppl.sb3.td3.policies import TD3Policy
from ppl.utils.shared_control_monitor import SharedControlMonitor
from ppl.utils.utils import get_time_str
import pathlib


class HGDAggerPVPTD3(PVPTD3):
    """PVPTD3 with only_bc_loss=True (HG-DAgger) plus expert-step logging.

    Tracks three counters that are written to the SB3 logger every time
    ``train()`` is called, and therefore appear in W&B / TensorBoard:

      train/expert_steps_total   – cumulative steps where the expert took over
      train/agent_steps_total    – cumulative steps where the agent acted alone
      train/expert_step_rate     – rolling takeover rate (expert / total so far)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._expert_steps_total: int = 0
        self._agent_steps_total: int = 0

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict],
    ) -> None:
        # infos[0]["takeover"] is set by ExpertTakeoverEnv for every step.
        # This is the single source of truth already used by PVPTD3 to route
        # transitions into human_data_buffer vs replay_buffer.
        if infos[0].get("takeover", False) or infos[0].get("takeover_start", False):
            self._expert_steps_total += 1
        else:
            self._agent_steps_total += 1

        # Delegate storage to the parent (keeps human_data_buffer routing intact)
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)

        total = self._expert_steps_total + self._agent_steps_total
        rate = self._expert_steps_total / total if total > 0 else 0.0

        self.logger.record("train/expert_steps_total", self._expert_steps_total)
        self.logger.record("train/agent_steps_total", self._agent_steps_total)
        self.logger.record("train/expert_step_rate", round(rate, 4))

FOLDER_PATH = pathlib.Path(__file__).parent.parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="hgdagger_metadrive", type=str,
        help="The name for this batch of experiments."
    )
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--learning_starts", default=10, type=int)
    # Checkpoint saved every 150 steps (same as train/eval cadence)
    parser.add_argument("--save_freq", default=150, type=int)
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true",
                        help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="",
                        help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="",
                        help="The team name for wandb.")
    parser.add_argument("--log_dir", type=str, default=FOLDER_PATH.parent.parent,
                        help="Folder to store the logs.")
    parser.add_argument("--bc_loss_weight", type=float, default=1.0)
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--toy_env", action="store_true",
                        help="Whether to use a toy environment.")

    args = parser.parse_args()

    # ===== HG-DAgger: PVPTD3 with only_bc_loss=True =====
    # HG-DAgger learns purely via behavioural cloning on human-takeover
    # transitions (no RL value/cost critic loss). This matches the
    # `only_bc_loss` branch in PVPTD3.train().
    experiment_batch_name = "HGDAgger"
    seed = args.seed
    trial_name = "{}_{}".format(experiment_batch_name, uuid.uuid4().hex[:8])
    print("Trial name is set to: ", trial_name)

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    log_dir = args.log_dir
    experiment_dir = Path(log_dir) / Path("runs") / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=False)   # Avoid overwriting old experiments
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(),

        # Algorithm config – identical hyper-parameters to the PPL baseline
        # except only_bc_loss=True which turns PVPTD3 into HG-DAgger.
        algo=dict(
            # --- HG-DAgger key flag ---
            only_bc_loss="True",       # Pure BC on human takeovers (HG-DAgger)
            add_bc_loss="True",
            bc_loss_weight=args.bc_loss_weight,

            # Flags unused by HG-DAgger but required by PVPTD3 signature
            with_human_proxy_value_loss="False",
            with_agent_proxy_value_loss="False",
            adaptive_batch_size="False",
            simple_batch="True",

            use_balance_sample=True,
            agent_data_ratio=1.0,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(
                discard_reward=True,   # Reward-free; HG-DAgger does not use reward
            ),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Experiment log
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir),
    )

    if args.toy_env:
        config["env_config"].update(
            num_scenarios=1,
            traffic_density=0.0,
            map="COT",
            use_render=True,
        )

    # ===== Setup the training environment =====
    train_env = ExpertTakeoverEnv(config=config["env_config"])
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    train_env = SharedControlMonitor(
        env=train_env, folder=trial_dir / "data", prefix=trial_name
    )
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Build the evaluation environment =====
    def _make_eval_env():
        eval_env_config = dict(
            use_render=False,
            manual_control=False,
            start_seed=1000,
            horizon=1500,
        )
        from ppl.experiments.metadrive.driving_env import DrivingEnv
        from ppl.sb3.common.monitor import Monitor
        eval_env = DrivingEnv(config=eval_env_config)
        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env

    # Evaluate every 150 training steps, over 50 episodes
    eval_env = SubprocVecEnv([_make_eval_env])
    eval_freq = 150        # Evaluate after every 150 training steps
    n_eval_episodes = 50   # Run 50 episodes per evaluation round

    # ===== Setup the callbacks =====
    save_freq = args.save_freq   # Checkpoint every 150 steps
    callbacks = [
        CheckpointCallback(
            name_prefix="rl_model",
            verbose=2,
            save_freq=save_freq,
            save_path=str(trial_dir / "models"),
        )
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config=config,
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Instantiate the HG-DAgger model =====
    model = HGDAggerPVPTD3(**config["algo"])

    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from ppl.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(
            ckpt, device=model.device, print_system_info=False
        )
        model.set_parameters(params, exact_match=True, device=model.device)

    train_env.env.env.model = model

    # ===== Launch training =====
    # Total training: 10 000 steps.
    # Evaluation cadence: every 150 steps, 50 episodes each time.
    model.learn(
        # Training
        total_timesteps=10_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # Evaluation
        eval_env=eval_env,
        eval_freq=eval_freq,           # Evaluate every 150 steps
        n_eval_episodes=n_eval_episodes,  # 50 episodes per evaluation
        eval_log_path=str(trial_dir),

        # Logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
        load_buffer=False,
    )