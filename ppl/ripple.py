
# """
# RIPPLE: Retrospective and Implicit Preferences from Predictive Learning of Experts.

# Composite loss:
#     L(pi) = L_BC
#           + lambda_fwd    * L_pref^fwd       (PPL's original forward preferences)
#           + lambda_back   * L_pref^back      (backward propagation from intervention)
#           + lambda_silent * L_silent         (non-intervention as implicit positive)
#           + lambda_traj   * L_traj           (trajectory-level good-vs-bad preference)

# Any lambda_i = 0 disables that term, which makes ablations trivial.

# Drop this file at ppl/ripple.py (sibling to ppl/ppl.py) and import RIPPLE in the
# training script. Requires the companion file ppl/sb3/haco/traj_buffer.py.
# """

# from collections import defaultdict
# from typing import List

# import numpy as np
# import torch
# import torch as th
# from torch.nn import functional as F

# from ppl.ppl import PPL, biased_bce_with_logits
# from ppl.sb3.haco.haco_buffer import PrefReplayBuffer
# from ppl.sb3.haco.traj_buffer import TrajPrefReplayBuffer


# class RIPPLE(PPL):
#     """PPL extended with backward propagation, silent-approval, and trajectory-level
#     preferences. All three signal sources share the same actor and the same DPO/CPL-style
#     pairwise objective used in PPL — the only thing that differs is *which states and
#     actions* populate the positive/negative slots.
#     """

#     # ------------------------------------------------------------------ init
#     def __init__(self, *args, **kwargs):
#         # Loss weights (set any to 0 to disable that term)
#         self.lambda_fwd    = float(kwargs.pop("lambda_fwd",    1.0))
#         self.lambda_back   = float(kwargs.pop("lambda_back",   0.5))
#         self.lambda_silent = float(kwargs.pop("lambda_silent", 0.3))
#         self.lambda_traj   = float(kwargs.pop("lambda_traj",   0.5))

#         # Data-collection hyperparameters (consumed by the env, but stored here so
#         # the env wrapper can read them off the model instance — matches how the
#         # existing code passes `model` into the env via `train_env.env.env.model = model`)
#         self.backward_horizon    = int(kwargs.pop("backward_horizon", 3))
#         self.silent_noise_scale  = float(kwargs.pop("silent_noise_scale", 0.3))
#         self.silent_margin       = float(kwargs.pop("silent_margin", 0.15))
#         self.temporal_decay      = float(kwargs.pop("temporal_decay", 0.9))
#         self.traj_max_len        = int(kwargs.pop("traj_max_len", 10))

#         super().__init__(*args, **kwargs)

#     # --------------------------------------------------------------- buffers
#     def _setup_model(self) -> None:
#         super()._setup_model()  # creates self.preference_buffer (forward) + human_data_buffer

#         # Backward and silent buffers are structurally identical to the forward one
#         self.back_pref_buffer = PrefReplayBuffer(
#             self.buffer_size, self.observation_space, self.action_space,
#             self.device, n_envs=self.n_envs,
#         )
#         self.silent_pref_buffer = PrefReplayBuffer(
#             self.buffer_size, self.observation_space, self.action_space,
#             self.device, n_envs=self.n_envs,
#         )
#         # Trajectory-level buffer stores padded (good_seq, bad_seq) pairs
#         self.traj_pref_buffer = TrajPrefReplayBuffer(
#             self.buffer_size, self.observation_space, self.action_space,
#             self.device, n_envs=self.n_envs, max_traj_len=self.traj_max_len,
#         )

#     def _excluded_save_params(self) -> List[str]:
#         return super()._excluded_save_params() + [
#             "back_pref_buffer", "silent_pref_buffer", "traj_pref_buffer",
#         ]

#     # ----------------------------------------------------------- loss pieces
#     def _pair_loss(self, pos_obs, pos_act, neg_obs, neg_act, beta):
#         """Pointwise CPL/DPO loss, identical to PPL's preference loss."""
#         def log_prob(obs, act):
#             mean = self.actor(obs)
#             return -((mean - act) ** 2).sum(dim=-1)
#         adv_pos = beta * log_prob(pos_obs, pos_act)
#         adv_neg = beta * log_prob(neg_obs, neg_act)
#         label = torch.ones_like(adv_pos)
#         return biased_bce_with_logits(adv_neg, adv_pos, label.float())

#     def _traj_loss(self, good_obs, good_act, good_mask,
#                    bad_obs,  bad_act,  bad_mask,  beta):
#         """Trajectory-level preference loss: compare mean log-prob across sequences."""
#         B, T = good_mask.shape
#         # Flatten time dim for actor forward pass, then restore
#         g_mean = self.actor(good_obs.reshape(B * T, -1)).reshape(B, T, -1)
#         b_mean = self.actor(bad_obs.reshape(B * T, -1)).reshape(B, T, -1)
#         g_logp = -((g_mean - good_act) ** 2).sum(dim=-1) * good_mask        # (B, T)
#         b_logp = -((b_mean - bad_act)  ** 2).sum(dim=-1) * bad_mask
#         # Mean over valid steps (avoid length bias)
#         g = g_logp.sum(dim=-1) / (good_mask.sum(dim=-1) + 1e-6)
#         b = b_logp.sum(dim=-1) / (bad_mask.sum(dim=-1)  + 1e-6)
#         adv_good, adv_bad = beta * g, beta * b
#         label = torch.ones_like(adv_good)
#         return biased_bce_with_logits(adv_bad, adv_good, label.float())

#     # ---------------------------------------------------------------- train
#     def train(self, gradient_steps: int, batch_size: int = 100) -> None:
#         self.policy.set_training_mode(True)
#         self._update_learning_rate([self.actor.optimizer])

#         stat = defaultdict(list)
#         beta            = self.extra_config["beta"]
#         bc_loss_weight  = self.extra_config["bc_loss_weight"]
#         only_bc         = self.extra_config["only_bc_loss"]

#         for _ in range(gradient_steps):
#             self._n_updates += 1

#             if self.human_data_buffer.pos == 0:
#                 break

#             # ---- BC on human demonstrations ------------------------------------
#             replay_data = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)
#             new_action = self.actor(replay_data.observations)
#             bc_loss = F.mse_loss(replay_data.actions_behavior, new_action)

#             total = bc_loss_weight * bc_loss

#             # When ablating to BC-only, skip all preference terms
#             if only_bc:
#                 self.actor.optimizer.zero_grad(); total.backward(); self.actor.optimizer.step()
#                 stat["bc_loss"].append(bc_loss.item())
#                 stat["total"].append(total.item())
#                 continue

#             zero = torch.tensor(0.0, device=self.device)
#             fwd_loss = back_loss = silent_loss = traj_loss = zero

#             # ---- Forward preferences (PPL's original term) ---------------------
#             if self.preference_buffer.pos > 0 and self.lambda_fwd > 0:
#                 d = self.preference_buffer.sample(batch_size, env=self._vec_normalize_env)
#                 fwd_loss, fwd_acc = self._pair_loss(
#                     d.pos_observations.squeeze(1), d.pos_actions.squeeze(1),
#                     d.neg_observations.squeeze(1), d.neg_actions.squeeze(1), beta,
#                 )
#                 total = total + self.lambda_fwd * fwd_loss
#                 stat["fwd_acc"].append(fwd_acc.item())

#             # ---- Backward preferences ------------------------------------------
#             if self.back_pref_buffer.pos > 0 and self.lambda_back > 0:
#                 d = self.back_pref_buffer.sample(batch_size, env=self._vec_normalize_env)
#                 back_loss, back_acc = self._pair_loss(
#                     d.pos_observations.squeeze(1), d.pos_actions.squeeze(1),
#                     d.neg_observations.squeeze(1), d.neg_actions.squeeze(1), beta,
#                 )
#                 total = total + self.lambda_back * back_loss
#                 stat["back_acc"].append(back_acc.item())

#             # ---- Silent-approval preferences -----------------------------------
#             if self.silent_pref_buffer.pos > 0 and self.lambda_silent > 0:
#                 d = self.silent_pref_buffer.sample(batch_size, env=self._vec_normalize_env)
#                 silent_loss, silent_acc = self._pair_loss(
#                     d.pos_observations.squeeze(1), d.pos_actions.squeeze(1),
#                     d.neg_observations.squeeze(1), d.neg_actions.squeeze(1), beta,
#                 )
#                 total = total + self.lambda_silent * silent_loss
#                 stat["silent_acc"].append(silent_acc.item())

#             # ---- Trajectory-level preferences ----------------------------------
#             if self.traj_pref_buffer.pos > 0 and self.lambda_traj > 0:
#                 d = self.traj_pref_buffer.sample(batch_size, env=self._vec_normalize_env)
#                 if d is not None:
#                     traj_loss, traj_acc = self._traj_loss(
#                         d.good_observations, d.good_actions, d.good_mask,
#                         d.bad_observations,  d.bad_actions,  d.bad_mask,  beta,
#                     )
#                     total = total + self.lambda_traj * traj_loss
#                     stat["traj_acc"].append(traj_acc.item())

#             # ---- Step ----------------------------------------------------------
#             self.actor.optimizer.zero_grad()
#             total.backward()
#             self.actor.optimizer.step()

#             stat["bc_loss"].append(bc_loss.item())
#             stat["fwd_loss"].append(fwd_loss.item())
#             stat["back_loss"].append(back_loss.item())
#             stat["silent_loss"].append(silent_loss.item())
#             stat["traj_loss"].append(traj_loss.item())
#             stat["total"].append(total.item())

#         # ---- Logging -----------------------------------------------------------
#         self.logger.record("buffers/fwd_pref",    self.preference_buffer.pos)
#         self.logger.record("buffers/back_pref",   self.back_pref_buffer.pos)
#         self.logger.record("buffers/silent_pref", self.silent_pref_buffer.pos)
#         self.logger.record("buffers/traj_pref",   self.traj_pref_buffer.pos)
#         self.logger.record("buffers/human",       self.human_data_buffer.pos)
#         self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
#         for k, v in stat.items():
#             self.logger.record(f"train/{k}", float(np.mean(v)))

"""
RIPPLE: Retrospective and Implicit Preferences from Predictive Learning of Experts.

Composite loss:
    L(pi) = L_BC
          + lambda_fwd    * L_pref^fwd       (PPL's original forward preferences)
          + lambda_back   * L_pref^back      (backward propagation from intervention)
          + lambda_silent * L_silent         (non-intervention as implicit positive)
          + lambda_traj   * L_traj           (trajectory-level good-vs-bad preference)

Any lambda_i = 0 disables that term, which makes ablations trivial.

Each loss term has its own beta (logit sharpness) for finer control. Default
values: beta_fwd=0.1, beta_back=0.1, beta_silent=0.05, beta_traj=0.2. The
shared 'beta' kwarg is used as a fallback if a per-term beta is not provided.

Drop this file at ppl/ripple.py (sibling to ppl/ppl.py) and import RIPPLE in the
training script. Requires the companion file ppl/sb3/haco/traj_buffer.py.
"""

from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch as th
from torch.nn import functional as F

from ppl.ppl import PPL, biased_bce_with_logits
from ppl.sb3.haco.haco_buffer import PrefReplayBuffer
from ppl.sb3.haco.traj_buffer import TrajPrefReplayBuffer


class RIPPLE(PPL):
    """PPL extended with backward propagation, silent-approval, and trajectory-level
    preferences. All three signal sources share the same actor and the same DPO/CPL-style
    pairwise objective used in PPL — the only thing that differs is *which states and
    actions* populate the positive/negative slots, and the per-term beta controlling
    logit sharpness.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, *args, **kwargs):
        # Loss weights (set any to 0 to disable that term)
        self.lambda_fwd    = float(kwargs.pop("lambda_fwd",    1.0))
        self.lambda_back   = float(kwargs.pop("lambda_back",   0.5))
        self.lambda_silent = float(kwargs.pop("lambda_silent", 0.3))
        self.lambda_traj   = float(kwargs.pop("lambda_traj",   0.5))

        # Per-term beta. If a term's beta is not explicitly set, it falls back
        # to the shared 'beta' kwarg that PPL reads from extra_config.
        # We can't read the fallback here because extra_config is populated in
        # super().__init__(), so store None and resolve later.
        self.beta_fwd    = kwargs.pop("beta_fwd",    None)
        self.beta_back   = kwargs.pop("beta_back",   None)
        self.beta_silent = kwargs.pop("beta_silent", None)
        self.beta_traj   = kwargs.pop("beta_traj",   None)

        # Data-collection hyperparameters (consumed by the env, but stored here so
        # the env wrapper can read them off the model instance — matches how the
        # existing code passes `model` into the env via `train_env.env.env.model = model`)
        self.backward_horizon    = int(kwargs.pop("backward_horizon", 3))
        self.silent_noise_scale  = float(kwargs.pop("silent_noise_scale", 0.3))
        self.silent_margin       = float(kwargs.pop("silent_margin", 0.15))
        self.temporal_decay      = float(kwargs.pop("temporal_decay", 0.9))
        self.traj_max_len        = int(kwargs.pop("traj_max_len", 10))

        super().__init__(*args, **kwargs)

        # Resolve per-term betas after super().__init__() populates extra_config
        shared_beta = float(self.extra_config.get("beta", 0.1))
        self.beta_fwd    = float(self.beta_fwd)    if self.beta_fwd    is not None else shared_beta
        self.beta_back   = float(self.beta_back)   if self.beta_back   is not None else shared_beta
        self.beta_silent = float(self.beta_silent) if self.beta_silent is not None else shared_beta
        self.beta_traj   = float(self.beta_traj)   if self.beta_traj   is not None else shared_beta

    # --------------------------------------------------------------- buffers
    def _setup_model(self) -> None:
        super()._setup_model()  # creates self.preference_buffer (forward) + human_data_buffer

        # Backward and silent buffers are structurally identical to the forward one
        self.back_pref_buffer = PrefReplayBuffer(
            self.buffer_size, self.observation_space, self.action_space,
            self.device, n_envs=self.n_envs,
        )
        self.silent_pref_buffer = PrefReplayBuffer(
            self.buffer_size, self.observation_space, self.action_space,
            self.device, n_envs=self.n_envs,
        )
        # Trajectory-level buffer stores padded (good_seq, bad_seq) pairs
        self.traj_pref_buffer = TrajPrefReplayBuffer(
            self.buffer_size, self.observation_space, self.action_space,
            self.device, n_envs=self.n_envs, max_traj_len=self.traj_max_len,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + [
            "back_pref_buffer", "silent_pref_buffer", "traj_pref_buffer",
        ]

    # ----------------------------------------------------------- loss pieces
    def _pair_loss(self, pos_obs, pos_act, neg_obs, neg_act, beta):
        """Pointwise CPL/DPO loss, identical to PPL's preference loss."""
        def log_prob(obs, act):
            mean = self.actor(obs)
            return -((mean - act) ** 2).sum(dim=-1)
        adv_pos = beta * log_prob(pos_obs, pos_act)
        adv_neg = beta * log_prob(neg_obs, neg_act)
        label = torch.ones_like(adv_pos)
        return biased_bce_with_logits(adv_neg, adv_pos, label.float())

    def _traj_loss(self, good_obs, good_act, good_mask,
                   bad_obs,  bad_act,  bad_mask,  beta):
        """Trajectory-level preference loss: compare mean log-prob across sequences."""
        B, T = good_mask.shape
        # Flatten time dim for actor forward pass, then restore
        g_mean = self.actor(good_obs.reshape(B * T, -1)).reshape(B, T, -1)
        b_mean = self.actor(bad_obs.reshape(B * T, -1)).reshape(B, T, -1)
        g_logp = -((g_mean - good_act) ** 2).sum(dim=-1) * good_mask        # (B, T)
        b_logp = -((b_mean - bad_act)  ** 2).sum(dim=-1) * bad_mask
        # Mean over valid steps (avoid length bias)
        g = g_logp.sum(dim=-1) / (good_mask.sum(dim=-1) + 1e-6)
        b = b_logp.sum(dim=-1) / (bad_mask.sum(dim=-1)  + 1e-6)
        adv_good, adv_bad = beta * g, beta * b
        label = torch.ones_like(adv_good)
        return biased_bce_with_logits(adv_bad, adv_good, label.float())

    # ---------------------------------------------------------------- train
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer])

        stat = defaultdict(list)
        bc_loss_weight  = self.extra_config["bc_loss_weight"]
        only_bc         = self.extra_config["only_bc_loss"]

        for _ in range(gradient_steps):
            self._n_updates += 1

            if self.human_data_buffer.pos == 0:
                break

            # ---- BC on human demonstrations ------------------------------------
            replay_data = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)
            new_action = self.actor(replay_data.observations)
            bc_loss = F.mse_loss(replay_data.actions_behavior, new_action)

            total = bc_loss_weight * bc_loss

            # When ablating to BC-only, skip all preference terms
            if only_bc:
                self.actor.optimizer.zero_grad(); total.backward(); self.actor.optimizer.step()
                stat["bc_loss"].append(bc_loss.item())
                stat["total"].append(total.item())
                continue

            zero = torch.tensor(0.0, device=self.device)
            fwd_loss = back_loss = silent_loss = traj_loss = zero

            # ---- Forward preferences (PPL's original term) ---------------------
            if self.preference_buffer.pos > 0 and self.lambda_fwd > 0:
                d = self.preference_buffer.sample(batch_size, env=self._vec_normalize_env)
                fwd_loss, fwd_acc = self._pair_loss(
                    d.pos_observations.squeeze(1), d.pos_actions.squeeze(1),
                    d.neg_observations.squeeze(1), d.neg_actions.squeeze(1),
                    self.beta_fwd,
                )
                total = total + self.lambda_fwd * fwd_loss
                stat["fwd_acc"].append(fwd_acc.item())

            # ---- Backward preferences ------------------------------------------
            if self.back_pref_buffer.pos > 0 and self.lambda_back > 0:
                d = self.back_pref_buffer.sample(batch_size, env=self._vec_normalize_env)
                back_loss, back_acc = self._pair_loss(
                    d.pos_observations.squeeze(1), d.pos_actions.squeeze(1),
                    d.neg_observations.squeeze(1), d.neg_actions.squeeze(1),
                    self.beta_back,
                )
                total = total + self.lambda_back * back_loss
                stat["back_acc"].append(back_acc.item())

            # ---- Silent-approval preferences -----------------------------------
            if self.silent_pref_buffer.pos > 0 and self.lambda_silent > 0:
                d = self.silent_pref_buffer.sample(batch_size, env=self._vec_normalize_env)
                silent_loss, silent_acc = self._pair_loss(
                    d.pos_observations.squeeze(1), d.pos_actions.squeeze(1),
                    d.neg_observations.squeeze(1), d.neg_actions.squeeze(1),
                    self.beta_silent,
                )
                total = total + self.lambda_silent * silent_loss
                stat["silent_acc"].append(silent_acc.item())

            # ---- Trajectory-level preferences ----------------------------------
            if self.traj_pref_buffer.pos > 0 and self.lambda_traj > 0:
                d = self.traj_pref_buffer.sample(batch_size, env=self._vec_normalize_env)
                if d is not None:
                    traj_loss, traj_acc = self._traj_loss(
                        d.good_observations, d.good_actions, d.good_mask,
                        d.bad_observations,  d.bad_actions,  d.bad_mask,
                        self.beta_traj,
                    )
                    total = total + self.lambda_traj * traj_loss
                    stat["traj_acc"].append(traj_acc.item())

            # ---- Step ----------------------------------------------------------
            self.actor.optimizer.zero_grad()
            total.backward()
            self.actor.optimizer.step()

            stat["bc_loss"].append(bc_loss.item())
            stat["fwd_loss"].append(fwd_loss.item())
            stat["back_loss"].append(back_loss.item())
            stat["silent_loss"].append(silent_loss.item())
            stat["traj_loss"].append(traj_loss.item())
            stat["total"].append(total.item())

        # ---- Logging -----------------------------------------------------------
        self.logger.record("buffers/fwd_pref",    self.preference_buffer.pos)
        self.logger.record("buffers/back_pref",   self.back_pref_buffer.pos)
        self.logger.record("buffers/silent_pref", self.silent_pref_buffer.pos)
        self.logger.record("buffers/traj_pref",   self.traj_pref_buffer.pos)
        self.logger.record("buffers/human",       self.human_data_buffer.pos)
        self.logger.record("hparams/beta_fwd",    self.beta_fwd)
        self.logger.record("hparams/beta_back",   self.beta_back)
        self.logger.record("hparams/beta_silent", self.beta_silent)
        self.logger.record("hparams/beta_traj",   self.beta_traj)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for k, v in stat.items():
            self.logger.record(f"train/{k}", float(np.mean(v)))
