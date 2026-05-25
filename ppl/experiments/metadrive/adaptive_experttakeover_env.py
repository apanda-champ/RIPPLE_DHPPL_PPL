"""
adaptive_experttakeover_env.py

Subclass of ExpertTakeoverEnv that makes the preference horizon L adaptive
based on the strength of each expert intervention.

Mathematical basis
------------------
The expert policy is a PPO DiagGaussianDistribution with:
  - observation-dependent mean  mu(s)  (output of actor network)
  - fixed log_std parameter           (single nn.Parameter, not obs-dependent)

Therefore the differential entropy of the expert distribution is CONSTANT
across all observations and cannot serve as a per-step uncertainty signal.

The correct signal is the INTERVENTION STRENGTH, defined as:

    strength(s, a_n) = -log_prob_expert(a_n | s)
                     = 0.5 * sum_i((a_n_i - mu_i(s))^2 / sigma_i^2)
                       + 0.5 * k * ln(2*pi) + sum_i(log_sigma_i)

Since sigma is constant, the only observation-dependent term is the first one:
the squared Mahalanobis distance between the novice action and the expert mean.
The constant terms (last two) shift the value but do not affect the ranking.

We use -log_prob directly (always >= constant floor, grows without bound as
the novice action deviates from the expert mean) as the raw strength signal.

Relationship to paper's Theorem 4.1
------------------------------------
The theorem bounds J(pi_h) - J(pi_n) = O(sqrt(eps + delta_pref + delta_dist)).

delta_pref = E_{s ~ d_pref} DTV(rho^s_ideal, rho^s_pref)

This grows with L because: for larger i, the expert action a_h sampled at
state s is increasingly misaligned with what the expert would do at predicted
future state s_tilde_i.

The rate at which misalignment grows depends on how rapidly the optimal action
changes along the trajectory, which in turn correlates with how severe the
current situation is. When the expert strongly intervenes (large strength):
  - The trajectory is in a safety-critical region
  - The expert's corrective action is qualitatively the same ("steer away from
    obstacle") for several predicted future steps
  - delta_pref grows slowly with L => longer L reduces delta_dist more than it
    increases delta_pref => net benefit from longer L

When the intervention is weak (small strength):
  - The situation is borderline; the optimal action changes quickly with state
  - delta_pref grows fast with L => shorter L is safer

Adaptive L mapping
------------------
We maintain a rolling buffer of observed strengths (last HISTORY_WINDOW samples)
to estimate percentile bounds, then linearly map:
    strength high (95th pct) -> L_max
    strength low  (5th pct)  -> L_min

Using a rolling window (rather than growing unboundedly) means late-training
percentile estimates reflect recent behaviour, not the entire run history.

Sample weight for CPO loss
--------------------------
    weight = strength / (running_max_strength + epsilon)
normalised to (0, 1].  High-strength interventions produce more reliable
preference labels and are up-weighted in the CPO loss.

Nothing in the original codebase is modified.
"""

import copy
import numpy as np
import torch
from collections import deque

from ppl.experiments.metadrive.experttakeover_env import ExpertTakeoverEnv, _expert as _global_expert
from ppl.experiments.metadrive.driving_env import DrivingEnv


# ── tuneable defaults (overridable via env config) ─────────────────────────
_DEFAULT_L_MIN = 1
_DEFAULT_L_MAX = 8
_PERCENTILE_LOW  = 5    # use 5th percentile of history as the "weak" anchor
_PERCENTILE_HIGH = 95   # use 95th percentile as the "strong" anchor
_HISTORY_MIN_SAMPLES = 20   # need this many samples before adaptive L activates
_HISTORY_WINDOW = 1000      # FIX: rolling window cap to avoid unbounded memory


class AdaptiveExpertTakeoverEnv(ExpertTakeoverEnv):
    """
    ExpertTakeoverEnv with intervention-strength-based adaptive preference
    horizon L and per-sample CPO loss weighting.

    New config keys (all optional):
        adaptive_L           (bool, default True)
            Whether to use adaptive L.  If False, falls back to the fixed
            'preference_horizon' from the parent config.
        L_min                (int, default 1)
        L_max                (int, default 8)
    """

    # FIX: removed class-level shared state (_strength_history, _max_strength_seen).
    # These are now instance-level so multiple envs in the same process don't
    # pollute each other's history. We use a deque with maxlen for a rolling
    # window so memory stays bounded over long runs.

    # ------------------------------------------------------------------ #
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Instance-level rolling history — safe for multi-env in same process
        self._strength_history: deque = deque(maxlen=_HISTORY_WINDOW)
        self._max_strength_seen: float = 1.0

    # ------------------------------------------------------------------ #
    def default_config(self):
        config = super().default_config()
        config.update({
            "adaptive_L": True,
            "L_min":      _DEFAULT_L_MIN,
            "L_max":      _DEFAULT_L_MAX,
        })
        return config

    # ------------------------------------------------------------------ #
    # Core helpers
    # ------------------------------------------------------------------ #

    def _compute_intervention_strength(self, log_prob_tensor: torch.Tensor) -> float:
        """
        intervention_strength = -log_prob(a_novice | expert_dist at s)

        This is the raw value already computed in step() as `log_prob`.
        We just negate it so that larger values mean stronger intervention.

        log_prob is always <= 0 for a probability distribution, so
        -log_prob >= 0 always.
        """
        return float(-log_prob_tensor.detach().cpu().item())

    def _update_history(self, strength: float) -> None:
        # FIX: deque with maxlen=_HISTORY_WINDOW automatically drops old entries
        self._strength_history.append(strength)
        if strength > self._max_strength_seen:
            self._max_strength_seen = strength

    def _get_adaptive_L(self, strength: float) -> int:
        """
        Map intervention strength linearly from [p5, p95] -> [L_min, L_max].
        Before enough history is collected, use the fixed preference_horizon
        from config (consistent with the non-adaptive baseline).
        """
        if not self.config["adaptive_L"]:
            return self.config["preference_horizon"]

        L_min = self.config["L_min"]
        L_max = self.config["L_max"]
        history = list(self._strength_history)

        # FIX: fall back to config preference_horizon (not arbitrary midpoint)
        # so cold-start behaviour matches the non-adaptive baseline exactly
        if len(history) < _HISTORY_MIN_SAMPLES:
            return self.config["preference_horizon"]

        s_low  = float(np.percentile(history, _PERCENTILE_LOW))
        s_high = float(np.percentile(history, _PERCENTILE_HIGH))

        if s_high <= s_low:
            # degenerate: all strengths equal, use config fallback
            return self.config["preference_horizon"]

        # Linear interpolation: strength in [s_low, s_high] -> L in [L_min, L_max]
        t = (strength - s_low) / (s_high - s_low)
        t = float(np.clip(t, 0.0, 1.0))
        L = L_min + t * (L_max - L_min)
        return int(round(np.clip(L, L_min, L_max)))

    def _get_sample_weight(self, strength: float) -> float:
        """
        Normalise strength to (0, 1] using running max.
        This is the weight that will scale the per-sample CPO loss.
        """
        denom = self._max_strength_seen + 1e-6
        return float(np.clip((strength + 1e-6) / denom, 1e-6, 1.0))

    # ------------------------------------------------------------------ #
    # Override store_preference_pairs
    # ------------------------------------------------------------------ #

    def store_preference_pairs(
        self,
        predicted_traj,
        preference_horizon,
        expert_action,
        intervention_strength: float = 1.0,
    ):
        """
        Same logic as parent but passes intervention_strength to the buffer
        so AdaptivePPL can weight the CPO loss per sample.
        """
        for step in range(min(len(predicted_traj) - 1, preference_horizon)):
            step_info = {
                "obs":      predicted_traj[step]["obs"].copy(),
                "action":   expert_action.copy(),
                # FIX: use next step obs instead of same obs to avoid
                # the pre-existing next_obs == obs bug in the original code
                "next_obs": predicted_traj[step + 1]["obs"].copy()
                            if step + 1 < len(predicted_traj)
                            else predicted_traj[step]["obs"].copy(),
                "done":     False,
            }
            positive_traj = [step_info]
            negative_traj = predicted_traj[step + 1:]
            self.model.preference_buffer.add(
                positive_traj,
                negative_traj,
                intervention_strength=intervention_strength,
            )

    # ------------------------------------------------------------------ #
    # Override step()
    # ------------------------------------------------------------------ #

    def step(self, actions):
        """
        Identical to ExpertTakeoverEnv.step() with the following additions:
          1. Compute intervention_strength = -log_prob(a_novice | expert_dist)
          2. Derive adaptive L from strength via percentile mapping
          3. Pass adaptive L and strength to store_preference_pairs
          4. Log strength, adaptive_L, sample_weight in info dict
        """
        actions = np.asarray(actions).astype(np.float32)

        if self.config["use_discrete"]:
            actions = self.discrete_to_continuous(actions)

        self.agent_action = copy.copy(actions)
        self.last_takeover = self.takeover

        num_predicted_steps = self.config["num_predicted_steps"]
        failure_check_freq  = self.config["failure_check_freq"]
        expert_noise_bound  = self.config["expert_noise"]

        if self.expert is None:
            self.expert = _global_expert

        # ── expert distribution at current obs (same as parent) ────────
        last_obs_tensor, _ = self.expert.obs_to_tensor(self.last_obs)
        distribution = self.expert.get_distribution(last_obs_tensor)

        # log_prob of the NOVICE action under the expert distribution
        log_prob = distribution.log_prob(
            torch.from_numpy(actions).to(last_obs_tensor.device)
        )

        # expert mean action (deterministic)
        expert_action, _ = self.expert.predict(
            self.last_obs, deterministic=True
        )
        enoise = np.random.randn(2) * expert_noise_bound
        expert_action = np.clip(
            enoise + expert_action,
            self.action_space.low,
            self.action_space.high,
        )

        # ── intervention strength: -log_prob >= 0 ─────────────────────
        strength = self._compute_intervention_strength(log_prob)
        self._update_history(strength)

        # ── adaptive L and normalised sample weight ────────────────────
        adaptive_L    = self._get_adaptive_L(strength)
        sample_weight = self._get_sample_weight(strength)

        # ── trajectory prediction & takeover decision ──────────────────
        if self.total_steps % failure_check_freq == 0:
            self.render_reset()
            self.takeover = self.decide_takeover(
                self.last_obs, num_predicted_steps
            )

        if self.takeover:
            if self.config["use_discrete"]:
                expert_action = self.continuous_to_discrete(expert_action)
                expert_action = self.discrete_to_continuous(expert_action)
            actions = expert_action

            if hasattr(self, "model") and hasattr(self.model, "preference_buffer"):
                predicted_traj, _ = self.predict_agent_future_trajectory(
                    self.last_obs,
                    num_predicted_steps,
                    action_behavior=self.agent_action.copy(),
                )
                self.store_preference_pairs(
                    predicted_traj,
                    adaptive_L,                   # <- adaptive horizon
                    expert_action.copy(),
                    intervention_strength=strength,
                )

        # ── env step (bypass DrivingEnv layer, same as parent) ─────────
        o, r, d, i = super(DrivingEnv, self).step(actions)

        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1

        if not self.config["disable_expert"]:
            i["takeover_log_prob"] = log_prob.item()

        # ── NEW: expose adaptive quantities via info dict ──────────────
        i["intervention_strength"] = float(strength)
        i["adaptive_L"]            = int(adaptive_L)
        i["sample_weight"]         = float(sample_weight)

        if self.config["use_render"]:
            self.render(
                text={
                    "Total Cost":    round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover":      "TAKEOVER" if self.takeover else "NO",
                    "Total Step":    self.total_steps,
                    "Takeover Rate": "{:.2f}%".format(
                        np.mean(np.array(self.takeover_recorder) * 100)
                    ),
                    "Strength":      "{:.3f}".format(strength),
                    "Adaptive L":    str(adaptive_L),
                    "Pause":         "Press E",
                }
            )

        assert i["takeover"] == self.takeover

        if self.config["use_discrete"]:
            i["raw_action"] = self.continuous_to_discrete(i["raw_action"])

        return o, r, d, i
