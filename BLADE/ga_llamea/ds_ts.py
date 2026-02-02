"""Discounted Thompson Sampling with Gaussian priors for operator selection.

This is a simplified version focused on operator selection (mutation, crossover, random_new).
Based on the D-TS algorithm from "Discounted Thompson Sampling for Non-Stationary Bandit Problems" (Qi et al.).
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ArmState:
    """Holds the sufficient statistics for a single arm."""

    discounted_count: float = 0.0
    discounted_sum: float = 0.0
    discounted_sum_sq: float = 0.0
    posterior_mean: float = 0.0
    posterior_var: float = 1.0
    last_theta: float = 0.0
    pulls: int = 0  # Total number of times this arm was selected


class DiscountedThompsonSampler:
    """Implements DS-TS with Gaussian priors for adaptive operator selection.

    This bandit learns which operator (mutation, crossover, random_new) produces
    the best offspring over time, adapting to non-stationary reward distributions
    through exponential discounting.
    """

    def __init__(
        self,
        arm_names: List[str],
        discount: float = 0.9,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        reward_variance: float = 1.0,
        tau_max: float = 5.0,
        epsilon: float = 1e-6,
    ):
        """Initialize the D-TS bandit.

        Args:
            arm_names: List of operator names (e.g., ["mutation", "crossover", "random_new"])
            discount: Exponential discount factor γ ∈ (0, 1]. Lower = faster adaptation.
            prior_mean: Prior mean for arm rewards (μ₀)
            prior_variance: Prior variance for arm rewards (σ₀²)
            reward_variance: Expected variance of observed rewards (σ²)
            tau_max: Maximum posterior standard deviation (caps exploration)
            epsilon: Small constant for numerical stability
        """
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1].")
        if prior_variance <= 0:
            raise ValueError("prior_variance must be positive.")
        if tau_max <= 0:
            raise ValueError("tau_max must be positive.")

        self.arm_names = arm_names
        self.discount = discount
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.reward_variance = reward_variance
        self.tau_max = tau_max
        self.epsilon = epsilon

        # Initialize arm states
        self.arms: Dict[str, ArmState] = {name: ArmState() for name in arm_names}
        self.total_pulls = 0

    def select_arm(self) -> Tuple[str, float]:
        """Sample an arm according to Thompson Sampling.

        Returns:
            Tuple of (selected_arm_name, sampled_theta_value)
        """
        best_arm = self.arm_names[0]
        best_theta = float("-inf")

        for arm_name, arm_state in self.arms.items():
            # Update posterior statistics
            self._update_posterior(arm_state)

            # Sample θ ~ N(μ̂, τ²)
            std_dev = math.sqrt(max(self.epsilon, arm_state.posterior_var))
            std_dev = min(std_dev, self.tau_max)
            theta = random.gauss(arm_state.posterior_mean, std_dev)

            arm_state.last_theta = theta

            # Select arm with maximum sampled value
            if theta > best_theta:
                best_theta = theta
                best_arm = arm_name

        self.total_pulls += 1
        return best_arm, best_theta

    def update(self, arm_name: str, reward: float) -> None:
        """Update the bandit with the observed reward.

        Args:
            arm_name: The arm that was selected
            reward: The observed reward (fitness improvement)
        """
        # Apply discount to ALL arms (exponential forgetting)
        self._apply_discount()

        # Update selected arm with new observation
        arm_state = self.arms[arm_name]
        arm_state.discounted_count += 1.0
        arm_state.discounted_sum += reward
        arm_state.discounted_sum_sq += reward**2
        arm_state.pulls += 1

        # Recompute posterior
        self._update_posterior(arm_state)

    def get_state_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return a serializable view of the current posterior statistics.

        Returns:
            Dictionary mapping arm names to their statistics
        """
        snapshot: Dict[str, Dict[str, float]] = {}
        for arm_name, arm_state in self.arms.items():
            snapshot[arm_name] = {
                "count": arm_state.discounted_count,
                "mean": arm_state.posterior_mean,
                "var": arm_state.posterior_var,
                "std": math.sqrt(max(self.epsilon, arm_state.posterior_var)),
                "theta": arm_state.last_theta,
                "pulls": arm_state.pulls,
            }
        return snapshot

    def _apply_discount(self) -> None:
        """Apply exponential discount to all arm statistics."""
        if self.discount == 1.0:
            return

        for arm_state in self.arms.values():
            arm_state.discounted_count *= self.discount
            arm_state.discounted_sum *= self.discount
            arm_state.discounted_sum_sq *= self.discount

    def _update_posterior(self, arm_state: ArmState) -> None:
        """Update posterior mean and variance using Bayesian update.

        Combines prior with observed data to compute posterior distribution.
        """
        count = max(self.epsilon, arm_state.discounted_count)
        sample_mean = arm_state.discounted_sum / count

        # Compute observed variance
        if arm_state.discounted_count <= self.epsilon:
            observed_var = self.reward_variance
        else:
            mean_sq = sample_mean**2
            moment = arm_state.discounted_sum_sq / count
            observed_var = max(
                self.epsilon,
                moment - mean_sq if moment > mean_sq else self.reward_variance,
            )

        # Bayesian update: posterior = prior + likelihood
        inv_prior = 1.0 / self.prior_variance
        inv_likelihood = count / max(self.epsilon, observed_var)
        posterior_var = 1.0 / max(self.epsilon, inv_prior + inv_likelihood)
        posterior_var = min(self.tau_max**2, posterior_var)
        posterior_mean = posterior_var * (
            self.prior_mean * inv_prior + sample_mean * inv_likelihood
        )

        arm_state.posterior_mean = posterior_mean
        arm_state.posterior_var = posterior_var
