"""
adaptive_ppl.py

Subclass of PPL that:
  1. Uses WeightedPrefReplayBuffer so each preference pair carries its
     intervention-strength weight.
  2. Computes the CPO loss as a WEIGHTED mean rather than a uniform mean:
         L_cpo = sum_i(w_i * l_i) / sum_i(w_i)
     where w_i is the normalised intervention strength of sample i.
  3. Logs the following extra metrics to TensorBoard / WandB:
       train/weighted_cpl_loss
       train/unweighted_cpl_loss
       train/mean_sample_weight
       env/intervention_strength_mean  (forwarded from env info)
       env/intervention_strength_std
       env/adaptive_L_mean
       env/adaptive_L_std
       env/adaptive_L_count_L{k}       (histogram of L values at interventions)

Nothing in ppl.py, haco_buffer.py, or any other original file is modified.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from ppl.ppl import PPL, biased_bce_with_logits
from ppl.sb3.haco.weighted_pref_buffer import WeightedPrefReplayBuffer


class AdaptivePPL(PPL):
    """
    PPL variant with intervention-strength-weighted CPO loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the standard PrefReplayBuffer with the weighted version.
        # We replicate the constructor call from PPL.__init__ exactly.
        self.preference_buffer = WeightedPrefReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            **self.replay_buffer_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer])

        stat_recorder = defaultdict(list)

        for _ in range(gradient_steps):
            # FIX: removed duplicate self._n_updates += 1 that was here before.
            # self._n_updates is incremented ONCE at the end of the loop only.

            if self.human_data_buffer.pos == 0:
                break

            replay_data     = self.human_data_buffer.sample(
                int(batch_size), env=self._vec_normalize_env
            )
            preference_data = self.preference_buffer.sample(
                int(batch_size), env=self._vec_normalize_env
            )

            # ── Behavioural Cloning loss (unchanged from PPL) ──────────
            new_action = self.actor(replay_data.observations)
            bc_loss = F.mse_loss(
                replay_data.actions_behavior, new_action, reduction="none"
            ).mean()

            stat_recorder["new_action_steering"].append(
                new_action[:, 0].mean().item()
            )
            stat_recorder["new_action_abs_steering"].append(
                torch.abs(new_action[:, 0]).mean().item()
            )
            stat_recorder["new_action_accerler"].append(
                new_action[:, 1].mean().item()
            )

            # ── Contrastive Preference Optimisation loss ───────────────
            pos_obs    = preference_data.pos_observations.squeeze()
            pos_action = preference_data.pos_actions.squeeze()
            neg_obs    = preference_data.neg_observations.squeeze()
            neg_action = preference_data.neg_actions.squeeze()
            # weights: normalised intervention strengths, shape [B]
            weights = preference_data.weights.to(self.device)

            def get_log_prob(obs, target_action):
                mean = self.actor(obs)
                return -((mean - target_action) ** 2).sum(dim=-1)

            beta         = self.extra_config["beta"]
            log_prob_pos = get_log_prob(pos_obs, pos_action)
            log_prob_neg = get_log_prob(neg_obs, neg_action)
            adv_pos      = beta * log_prob_pos
            adv_neg      = beta * log_prob_neg
            label        = torch.ones_like(adv_pos)

            # Unweighted CPO loss (for logging comparison)
            dpo_loss_unweighted, accuracy = biased_bce_with_logits(
                adv_neg, adv_pos, label.float()
            )

            # Per-sample CPO loss (recompute without .mean() reduction)
            # biased_bce_with_logits uses:
            #   y * NLP(adv2 > adv1) + (1-y) * NLP(adv1 > adv2)
            # where NLP is the numerically stable negative log probability.
            logit21 = adv_pos - 0.5 * adv_neg
            logit12 = adv_neg - 0.5 * adv_pos
            max21   = torch.clamp(-logit21, min=0)
            max12   = torch.clamp(-logit12, min=0)
            nlp21   = (torch.log(torch.exp(-max21) + torch.exp(-logit21 - max21))
                       + max21)
            nlp12   = (torch.log(torch.exp(-max12) + torch.exp(-logit12 - max12))
                       + max12)
            per_sample_cpo = label * nlp21 + (1 - label) * nlp12  # shape [B]

            # Weighted mean: sum(w * l) / sum(w)
            dpo_loss_weighted = (
                (weights * per_sample_cpo).sum()
                / (weights.sum() + 1e-8)
            )

            bc_loss_weight = self.extra_config["bc_loss_weight"]
            if self.extra_config["only_bc_loss"]:
                loss = bc_loss
            else:
                loss = bc_loss_weight * bc_loss + dpo_loss_weighted

            self.actor.optimizer.zero_grad()
            loss.backward()
            self.actor.optimizer.step()

            # ── record stats ───────────────────────────────────────────
            stat_recorder["bc_loss"].append(bc_loss.item())
            stat_recorder["cpl_loss"].append(dpo_loss_weighted.item())
            stat_recorder["unweighted_cpl_loss"].append(
                dpo_loss_unweighted.item()
            )
            stat_recorder["cpl_accuracy"].append(accuracy.item())
            stat_recorder["loss"].append(loss.item())
            stat_recorder["mean_sample_weight"].append(weights.mean().item())

        # FIX: increment _n_updates exactly ONCE after the loop (not inside it too)
        self._n_updates += gradient_steps

        self.logger.record(
            "train/predicted_steps", self.preference_buffer.pos
        )
        self.logger.record(
            "train/human_involved_steps", self.human_data_buffer.pos
        )
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        for key, values in stat_recorder.items():
            if values:
                self.logger.record(f"train/{key}", np.mean(values))

    # ------------------------------------------------------------------ #
    # Called by AdaptiveMetricsCallback after each rollout
    # ------------------------------------------------------------------ #

    def log_env_stats(
        self,
        strength_values: List[float],
        adaptive_L_values: List[int],
    ) -> None:
        """
        Forward per-step env statistics into the SB3 logger so they appear
        in TensorBoard and WandB alongside the training metrics.

        Parameters
        ----------
        strength_values : list of -log_prob values collected during the rollout
        adaptive_L_values : list of adaptive L values at intervention steps
        """
        if strength_values:
            self.logger.record(
                "env/intervention_strength_mean",
                float(np.mean(strength_values))
            )
            self.logger.record(
                "env/intervention_strength_std",
                float(np.std(strength_values))
            )

        if adaptive_L_values:
            self.logger.record(
                "env/adaptive_L_mean", float(np.mean(adaptive_L_values))
            )
            self.logger.record(
                "env/adaptive_L_std", float(np.std(adaptive_L_values))
            )
            # Per-value counts so WandB can show distribution over L choices
            for lv in range(1, 10):
                count = int(np.sum(np.array(adaptive_L_values) == lv))
                self.logger.record(
                    f"env/adaptive_L_count_L{lv}", count
                )
