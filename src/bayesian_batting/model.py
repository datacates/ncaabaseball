"""
Core Bayesian Model Classes for Batting Component Statistics

This module provides two conjugate prior models:
- BetaComponentModel: For rate statistics (K%, BB%) using Beta-Binomial
- NormalComponentModel: For continuous statistics (ISO, BABIP) using Normal-Normal
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy import stats


class BattingComponentModel(ABC):
    """Abstract base class for component statistic models."""

    @abstractmethod
    def get_posterior_params(self) -> dict:
        """Return the current posterior parameters."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> 'BattingComponentModel':
        """Update the posterior with new observations."""
        pass

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Draw n samples from the posterior distribution."""
        pass

    @abstractmethod
    def get_mean(self) -> float:
        """Return the posterior mean."""
        pass

    @abstractmethod
    def get_std(self) -> float:
        """Return the posterior standard deviation."""
        pass

    @abstractmethod
    def get_prediction_interval(self, alpha: float = 0.9) -> Tuple[float, float]:
        """Return the (1-alpha) prediction interval."""
        pass


@dataclass
class BetaComponentModel(BattingComponentModel):
    """
    Beta-Binomial conjugate model for rate statistics (K%, BB%).

    Prior: Beta(alpha, beta)
    Likelihood: Binomial(n, p)
    Posterior: Beta(alpha + successes, beta + failures)

    The parameters alpha and beta represent:
    - alpha: equivalent prior successes (e.g., strikeouts)
    - beta: equivalent prior failures (e.g., non-strikeouts)
    - Prior strength = alpha + beta (equivalent sample size)

    Attributes:
        alpha: Shape parameter (prior + observed successes)
        beta: Shape parameter (prior + observed failures)
        stat_name: Name of the statistic (e.g., 'K%', 'BB%')
    """
    alpha: float
    beta: float
    stat_name: str = ''

    def __post_init__(self):
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(f"Alpha and beta must be positive. Got alpha={self.alpha}, beta={self.beta}")

    @classmethod
    def from_mean_strength(cls, mean: float, strength: float, stat_name: str = '') -> 'BetaComponentModel':
        """
        Create a Beta model from mean and prior strength (equivalent sample size).

        Args:
            mean: Prior mean (between 0 and 1)
            strength: Prior strength (equivalent PA)
            stat_name: Name of the statistic

        Returns:
            BetaComponentModel with alpha = mean * strength, beta = (1-mean) * strength
        """
        mean = np.clip(mean, 0.001, 0.999)  # Avoid edge cases
        alpha = mean * strength
        beta = (1 - mean) * strength
        return cls(alpha=alpha, beta=beta, stat_name=stat_name)

    @property
    def prior_strength(self) -> float:
        """Return the total prior strength (equivalent sample size)."""
        return self.alpha + self.beta

    def get_posterior_params(self) -> dict:
        """Return the posterior parameters."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'mean': self.get_mean(),
            'std': self.get_std(),
            'strength': self.prior_strength
        }

    def update(self, successes: int, failures: int) -> 'BetaComponentModel':
        """
        Update posterior with new observations (conjugate update).

        Args:
            successes: Number of successes (e.g., strikeouts for K%)
            failures: Number of failures (e.g., non-strikeouts for K%)

        Returns:
            New BetaComponentModel with updated parameters
        """
        return BetaComponentModel(
            alpha=self.alpha + successes,
            beta=self.beta + failures,
            stat_name=self.stat_name
        )

    def sample(self, n: int) -> np.ndarray:
        """Draw n samples from the posterior Beta distribution."""
        return stats.beta.rvs(self.alpha, self.beta, size=n)

    def get_mean(self) -> float:
        """Return the posterior mean: alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    def get_std(self) -> float:
        """Return the posterior standard deviation."""
        a, b = self.alpha, self.beta
        return np.sqrt((a * b) / ((a + b)**2 * (a + b + 1)))

    def get_prediction_interval(self, alpha: float = 0.9) -> Tuple[float, float]:
        """
        Return the central (alpha) prediction interval.

        Args:
            alpha: Coverage probability (default 0.9 for 90% interval)

        Returns:
            Tuple of (lower, upper) bounds
        """
        lower = (1 - alpha) / 2
        upper = 1 - lower
        return (
            stats.beta.ppf(lower, self.alpha, self.beta),
            stats.beta.ppf(upper, self.alpha, self.beta)
        )

    def get_percentiles(self) -> dict:
        """Return common percentiles (10, 25, 50, 75, 90)."""
        percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        return {
            f'p{int(p*100)}': stats.beta.ppf(p, self.alpha, self.beta)
            for p in percentiles
        }


@dataclass
class NormalComponentModel(BattingComponentModel):
    """
    Normal-Normal conjugate model for continuous statistics (ISO, BABIP).

    Uses precision (tau = 1/variance) parameterization for conjugate updates.

    Prior: Normal(mu, 1/tau)
    Likelihood: Normal(x, 1/tau_likelihood) where tau_likelihood is known
    Posterior: Normal(mu_post, 1/tau_post)

    Attributes:
        mu: Posterior mean
        tau: Posterior precision (1/variance)
        likelihood_variance: Known variance for observations
        stat_name: Name of the statistic (e.g., 'ISO', 'BABIP')
    """
    mu: float
    tau: float  # Precision = 1/variance
    likelihood_variance: float = 0.01  # Default, should be set from data
    stat_name: str = ''

    def __post_init__(self):
        if self.tau <= 0:
            raise ValueError(f"Tau (precision) must be positive. Got tau={self.tau}")
        if self.likelihood_variance <= 0:
            raise ValueError(f"Likelihood variance must be positive. Got {self.likelihood_variance}")

    @classmethod
    def from_mean_strength(
        cls,
        mean: float,
        strength: float,
        likelihood_variance: float,
        stat_name: str = ''
    ) -> 'NormalComponentModel':
        """
        Create a Normal model from mean and prior strength.

        Args:
            mean: Prior mean
            strength: Prior strength (equivalent PA)
            likelihood_variance: Known variance for single observations
            stat_name: Name of the statistic

        Returns:
            NormalComponentModel with tau = strength / likelihood_variance
        """
        tau = strength / likelihood_variance
        return cls(mu=mean, tau=tau, likelihood_variance=likelihood_variance, stat_name=stat_name)

    @property
    def variance(self) -> float:
        """Return the posterior variance (1/tau)."""
        return 1.0 / self.tau

    @property
    def prior_strength(self) -> float:
        """Return the equivalent prior strength in PA."""
        return self.tau * self.likelihood_variance

    def get_posterior_params(self) -> dict:
        """Return the posterior parameters."""
        return {
            'mu': self.mu,
            'tau': self.tau,
            'variance': self.variance,
            'std': self.get_std(),
            'strength': self.prior_strength
        }

    def update(self, observations: np.ndarray) -> 'NormalComponentModel':
        """
        Update posterior with new observations (conjugate update).

        The update formulas are:
        tau_post = tau_prior + n / likelihood_variance
        mu_post = (tau_prior * mu_prior + sum(x) / likelihood_variance) / tau_post

        Args:
            observations: Array of observed values (e.g., ISO per PA)

        Returns:
            New NormalComponentModel with updated parameters
        """
        observations = np.atleast_1d(observations)
        n = len(observations)

        if n == 0:
            return self

        obs_mean = np.mean(observations)

        # Conjugate update
        tau_post = self.tau + n / self.likelihood_variance
        mu_post = (self.tau * self.mu + n * obs_mean / self.likelihood_variance) / tau_post

        return NormalComponentModel(
            mu=mu_post,
            tau=tau_post,
            likelihood_variance=self.likelihood_variance,
            stat_name=self.stat_name
        )

    def update_with_summary(self, mean: float, n: int) -> 'NormalComponentModel':
        """
        Update posterior with summary statistics.

        Args:
            mean: Mean of observations
            n: Number of observations

        Returns:
            New NormalComponentModel with updated parameters
        """
        if n == 0:
            return self

        tau_post = self.tau + n / self.likelihood_variance
        mu_post = (self.tau * self.mu + n * mean / self.likelihood_variance) / tau_post

        return NormalComponentModel(
            mu=mu_post,
            tau=tau_post,
            likelihood_variance=self.likelihood_variance,
            stat_name=self.stat_name
        )

    def sample(self, n: int) -> np.ndarray:
        """Draw n samples from the posterior Normal distribution."""
        return stats.norm.rvs(loc=self.mu, scale=np.sqrt(self.variance), size=n)

    def get_mean(self) -> float:
        """Return the posterior mean."""
        return self.mu

    def get_std(self) -> float:
        """Return the posterior standard deviation."""
        return np.sqrt(self.variance)

    def get_prediction_interval(self, alpha: float = 0.9) -> Tuple[float, float]:
        """
        Return the central (alpha) prediction interval.

        Args:
            alpha: Coverage probability (default 0.9 for 90% interval)

        Returns:
            Tuple of (lower, upper) bounds
        """
        z = stats.norm.ppf((1 + alpha) / 2)
        std = self.get_std()
        return (self.mu - z * std, self.mu + z * std)

    def get_percentiles(self) -> dict:
        """Return common percentiles (10, 25, 50, 75, 90)."""
        percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        std = self.get_std()
        return {
            f'p{int(p*100)}': stats.norm.ppf(p, loc=self.mu, scale=std)
            for p in percentiles
        }


@dataclass
class PlayerPosteriors:
    """
    Container for a player's posterior distributions across all component stats.

    Attributes:
        player_name: Player identifier
        team: Current team
        season: Season year
        k_pct: BetaComponentModel for strikeout rate
        bb_pct: BetaComponentModel for walk rate
        iso: NormalComponentModel for isolated power
        babip: NormalComponentModel for batting average on balls in play
    """
    player_name: str
    team: str
    season: int
    k_pct: BetaComponentModel
    bb_pct: BetaComponentModel
    iso: NormalComponentModel
    babip: NormalComponentModel

    def sample_all(self, n: int) -> dict:
        """
        Draw n samples from all posterior distributions.

        Returns:
            Dict with arrays of samples for each stat
        """
        return {
            'K%': self.k_pct.sample(n),
            'BB%': self.bb_pct.sample(n),
            'ISO': self.iso.sample(n),
            'BABIP': self.babip.sample(n)
        }

    def get_means(self) -> dict:
        """Return the posterior means for all stats."""
        return {
            'K%': self.k_pct.get_mean(),
            'BB%': self.bb_pct.get_mean(),
            'ISO': self.iso.get_mean(),
            'BABIP': self.babip.get_mean()
        }

    def get_prediction_intervals(self, alpha: float = 0.9) -> dict:
        """Return prediction intervals for all stats."""
        return {
            'K%': self.k_pct.get_prediction_interval(alpha),
            'BB%': self.bb_pct.get_prediction_interval(alpha),
            'ISO': self.iso.get_prediction_interval(alpha),
            'BABIP': self.babip.get_prediction_interval(alpha)
        }

    def to_dict(self) -> dict:
        """Serialize posteriors to dictionary for JSON export."""
        return {
            'player_name': self.player_name,
            'team': self.team,
            'season': self.season,
            'K%': {
                'alpha': self.k_pct.alpha,
                'beta': self.k_pct.beta,
                'mean': self.k_pct.get_mean(),
                'std': self.k_pct.get_std()
            },
            'BB%': {
                'alpha': self.bb_pct.alpha,
                'beta': self.bb_pct.beta,
                'mean': self.bb_pct.get_mean(),
                'std': self.bb_pct.get_std()
            },
            'ISO': {
                'mu': self.iso.mu,
                'tau': self.iso.tau,
                'mean': self.iso.get_mean(),
                'std': self.iso.get_std()
            },
            'BABIP': {
                'mu': self.babip.mu,
                'tau': self.babip.tau,
                'mean': self.babip.get_mean(),
                'std': self.babip.get_std()
            }
        }

    @classmethod
    def from_dict(cls, data: dict, likelihood_vars: dict) -> 'PlayerPosteriors':
        """Reconstruct PlayerPosteriors from serialized dictionary."""
        return cls(
            player_name=data['player_name'],
            team=data['team'],
            season=data['season'],
            k_pct=BetaComponentModel(
                alpha=data['K%']['alpha'],
                beta=data['K%']['beta'],
                stat_name='K%'
            ),
            bb_pct=BetaComponentModel(
                alpha=data['BB%']['alpha'],
                beta=data['BB%']['beta'],
                stat_name='BB%'
            ),
            iso=NormalComponentModel(
                mu=data['ISO']['mu'],
                tau=data['ISO']['tau'],
                likelihood_variance=likelihood_vars['ISO'],
                stat_name='ISO'
            ),
            babip=NormalComponentModel(
                mu=data['BABIP']['mu'],
                tau=data['BABIP']['tau'],
                likelihood_variance=likelihood_vars['BABIP'],
                stat_name='BABIP'
            )
        )
