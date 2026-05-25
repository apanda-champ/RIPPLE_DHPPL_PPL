import copy
import math
import pathlib
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from metadrive.engine.logger import get_logger
from metadrive.examples.ppo_expert.numpy_expert import ckpt_path
from metadrive.policy.env_input_policy import EnvInputPolicy

from ppl.experiments.metadrive.driving_env import DrivingEnv

FOLDER_PATH = pathlib.Path(__file__).parent

logger = get_logger()


def get_expert():
    from ppl.sb3.common.save_util import load_from_zip_file
    from ppl.sb3.ppo import PPO
    from ppl.sb3.ppo.policies import ActorCriticPolicy

    train_env = DrivingEnv(config={'manual_control': False, "use_render": False})

    algo_config = dict(
        policy=ActorCriticPolicy,
        n_steps=1024,
        n_epochs=20,
        learning_rate=5e-5,
        batch_size=256,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=10.0,
        create_eval_env=False,
        verbose=2,
        device="auto",
        env=train_env
    )
    model = PPO(**algo_config)

    ckpt = FOLDER_PATH / "metadrive_ppo_expert_20m_steps.zip"

    print(f"Loading checkpoint from {ckpt}!")
    data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=True, device=model.device)
    print(f"Model is loaded from {ckpt}!")

    train_env.close()

    return model.policy


def obs_correction(obs):
    obs[15] = 1 - obs[15]
    obs[10] = 1 - obs[10]
    return obs


def normpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


_expert = get_expert()


class ExpertTakeoverEnv(DrivingEnv):
    last_takeover = None
    last_obs = None
    expert = None
    drawn_points = []

    def __init__(self, config):
        super(ExpertTakeoverEnv, self).__init__(config)
        if self.config["use_discrete"]:
            self._num_bins = 13
            self._grid = np.linspace(-1, 1, self._num_bins)
            self._actions = np.array(np.meshgrid(self._grid, self._grid)).T.reshape(-1, 2)

    @property
    def action_space(self) -> gym.Space:
        if self.config["use_discrete"]:
            return gym.spaces.Discrete(self._num_bins ** 2)
        else:
            return super(ExpertTakeoverEnv, self).action_space

    def default_config(self):
        config = super(ExpertTakeoverEnv, self).default_config()
        config.update(
            {
                "use_discrete": False,
                "disable_expert": False,
                "agent_policy": EnvInputPolicy,
                "manual_control": False,
                "use_render": False,
                "expert_deterministic": False,
                "num_predicted_steps": 20,
                "failure_check_freq": 10,
                "preference_horizon": 3,
                "expert_noise": 0,
            }
        )
        return config

    def continuous_to_discrete(self, a):
        distances = np.linalg.norm(self._actions - a, axis=1)
        discrete_index = np.argmin(distances)
        return discrete_index

    def discrete_to_continuous(self, a):
        continuous_action = self._actions[a.astype(int)]
        return continuous_action

    def decide_takeover(self, obs, num_predicted_steps):
        predicted_traj_real, info_real = self.predict_agent_future_trajectory(obs, num_predicted_steps)
        assert info_real["failure"] == (info_real["total_reward"] < 0)
        self.render_traj(predicted_traj_real, (info_real["failure"], 1 - info_real["failure"], 0))
        return info_real["failure"]

    def store_preference_pairs(self, predicted_traj, preference_horizon, expert_action):
        """
        DH-PPL: Dynamic Horizon gating.

        For each step in the predicted trajectory (up to preference_horizon),
        we call model.should_add_to_preference_buffer() ONCE per step.

        FIX (Bug 1): should_add_to_preference_buffer() now returns (bool, float).
        We unpack both values from a single call — this avoids calling the model
        twice per step (which was inflating the rolling window and skewing the
        uncertainty threshold).
        """
        model = getattr(self, "model", None)
        has_gate = (
            model is not None
            and hasattr(model, "should_add_to_preference_buffer")
            and hasattr(model, "preference_buffer")
        )

        admitted = 0
        rejected = 0
        u_values = []

        for step in range(min(len(predicted_traj) - 1, preference_horizon)):
            step_obs = predicted_traj[step]["obs"]
            step_agent_action = predicted_traj[step]["action"]

            if has_gate:
                # FIX (Bug 1): single call returns (bool, float) — no double-append to window
                should_add, u = model.should_add_to_preference_buffer(
                    step_obs, step_agent_action
                )
                u_values.append(u)

                print(
                    f"  [PrefBuffer] step={step} | U={u:.4f} | "
                    f"L(threshold)={model._uncertainty_threshold:.4f} | "
                    f"window_size={len(model._uncertainty_window)} | "
                    f"{'ADMIT' if should_add else 'REJECT'}"
                )

                if not should_add:
                    rejected += 1
                    continue

                admitted += 1

            # Build contrastive preference pair: expert action is preferred
            # over the agent action at this predicted future state.
            step_info = {
                "obs": step_obs.copy(),
                "action": expert_action.copy(),
                "next_obs": step_obs.copy(),
                "done": False,
            }
            positive_traj = [step_info]
            negative_traj = predicted_traj[step + 1:]

            if has_gate:
                model.preference_buffer.add(positive_traj, negative_traj)
            else:
                self.model.preference_buffer.add(positive_traj, negative_traj)

        # --- Summary logging ---
        if has_gate:
            threshold_val = model._uncertainty_threshold
            buffer_pos = model.preference_buffer.pos
            total = admitted + rejected
            print(
                f"[PrefBuffer SUMMARY] takeover at step={self.total_steps} | "
                f"admitted={admitted} | rejected={rejected} | "
                f"buffer_size={buffer_pos} | "
                f"threshold_L={threshold_val:.4f}"
            )

            try:
                import wandb
                if wandb.run is not None:
                    log_threshold = (
                        model._uncertainty_threshold
                        if model._uncertainty_threshold != float("inf")
                        else 0.0
                    )
                    wandb.log(
                        {
                            "pref_buffer/admitted": admitted,
                            "pref_buffer/rejected": rejected,
                            "pref_buffer/total_attempted": total,
                            "pref_buffer/accept_rate": admitted / max(total, 1),
                            "pref_buffer/threshold_L": log_threshold,
                            "pref_buffer/uncertainty_mean": float(np.mean(u_values)) if u_values else 0.0,
                            "pref_buffer/uncertainty_max": float(np.max(u_values)) if u_values else 0.0,
                            "pref_buffer/uncertainty_min": float(np.min(u_values)) if u_values else 0.0,
                            "pref_buffer/buffer_size": buffer_pos,
                            "pref_buffer/env_step": self.total_steps,
                        },
                        step=self.total_steps,
                    )
            except Exception as e:
                print(f"[wandb log failed] {e}")

    def step(self, actions):
        actions = np.asarray(actions).astype(np.float32)

        if self.config["use_discrete"]:
            actions = self.discrete_to_continuous(actions)

        self.agent_action = copy.copy(actions)
        self.last_takeover = self.takeover

        num_predicted_steps = self.config["num_predicted_steps"]
        failure_check_freq = self.config["failure_check_freq"]
        preference_horizon = self.config["preference_horizon"]
        expert_noise_bound = self.config["expert_noise"]

        if self.expert is None:
            global _expert
            self.expert = _expert

        last_obs, _ = self.expert.obs_to_tensor(self.last_obs)
        distribution = self.expert.get_distribution(last_obs)
        log_prob = distribution.log_prob(torch.from_numpy(actions).to(last_obs.device))
        action_prob = log_prob.exp().detach().cpu().numpy()
        action_prob = action_prob[0]  # noqa: F841
        expert_action, _ = self.expert.predict(self.last_obs, deterministic=True)
        enoise = np.random.randn(2) * expert_noise_bound
        expert_action = np.clip(enoise + expert_action, self.action_space.low, self.action_space.high)

        if self.total_steps % failure_check_freq == 0:
            self.render_reset()
            self.takeover = self.decide_takeover(self.last_obs, num_predicted_steps)

        if self.takeover:
            if self.config["use_discrete"]:
                expert_action = self.continuous_to_discrete(expert_action)
                expert_action = self.discrete_to_continuous(expert_action)
            actions = expert_action
            if hasattr(self, "model") and hasattr(self.model, "preference_buffer"):
                predicted_traj, _ = self.predict_agent_future_trajectory(
                    self.last_obs, num_predicted_steps, action_behavior=self.agent_action.copy()
                )
                self.store_preference_pairs(predicted_traj, preference_horizon, expert_action.copy())

        o, r, d, i = super(DrivingEnv, self).step(actions)

        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1

        if not self.config["disable_expert"]:
            i["takeover_log_prob"] = log_prob.item()

        if self.config["use_render"]:
            self.render(
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover": "TAKEOVER" if self.takeover else "NO",
                    "Total Step": self.total_steps,
                    "Takeover Rate": "{:.2f}%".format(np.mean(np.array(self.takeover_recorder) * 100)),
                    "Pause": "Press E",
                }
            )

        assert i["takeover"] == self.takeover

        if self.config["use_discrete"]:
            i["raw_action"] = self.continuous_to_discrete(i["raw_action"])
        return o, r, d, i

    def _get_step_return(self, actions, engine_info):
        o, r, tm, tc, engine_info = super(DrivingEnv, self)._get_step_return(actions, engine_info)
        self.last_obs = o
        d = tm or tc
        last_t = self.last_takeover
        engine_info["takeover_start"] = True if not last_t and self.takeover else False
        engine_info["takeover"] = self.takeover
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        if not condition:
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost
        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info["cost"]
        engine_info["episode_native_cost"] = self.episode_cost
        self.total_cost += engine_info["cost"]
        self.total_takeover_count += 1 if self.takeover else 0
        engine_info["total_takeover_count"] = self.total_takeover_count
        engine_info["total_cost"] = self.total_cost
        return o, r, d, engine_info

    def _get_reset_return(self, reset_info):
        o, info = super(DrivingEnv, self)._get_reset_return(reset_info)
        self.last_obs = o
        self.last_takeover = False
        self.render_reset()
        return o, info


if __name__ == "__main__":
    env = ExpertTakeoverEnv(dict(use_render=True, num_scenarios=1, traffic_density=0))
    env.reset()
    ss = 0
    while True:
        if ss < 10:
            _, _, done, info = env.step([0, 1])
        else:
            _, _, done, info = env.step([0, 0.1])
        ss += 1
        if done:
            print(info)
            env.reset()
            ss = 0
