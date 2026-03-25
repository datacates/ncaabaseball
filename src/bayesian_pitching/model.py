"""
Pitcher Posterior Model

Imports the distribution-agnostic BetaComponentModel and NormalComponentModel
from the batting model and provides PitcherPosteriors as the pitching-specific
container class (parallel to PlayerPosteriors for batting).
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from bayesian_batting.model import BetaComponentModel, NormalComponentModel


@dataclass
class PitcherPosteriors:
    """
    Container for a pitcher's posterior distributions across all component stats.

    Attributes:
        player_name: Pitcher identifier
        team: Current team
        season: Season year
        k_pct: BetaComponentModel for strikeout rate (K/BF)
        bb_pct: BetaComponentModel for walk rate (BB/BF)
        hr_fb_pct: NormalComponentModel for HR per fly ball rate
        babip: NormalComponentModel for BABIP allowed
    """
    player_name: str
    team: str
    season: int
    k_pct: BetaComponentModel
    bb_pct: BetaComponentModel
    hr_fb_pct: NormalComponentModel
    babip: NormalComponentModel

    def sample_all(self, n: int) -> dict:
        """Draw n samples from all posterior distributions."""
        return {
            'K%': self.k_pct.sample(n),
            'BB%': self.bb_pct.sample(n),
            'HR/FB%': self.hr_fb_pct.sample(n),
            'BABIP': self.babip.sample(n),
        }

    def get_means(self) -> dict:
        """Return the posterior means for all stats."""
        return {
            'K%': self.k_pct.get_mean(),
            'BB%': self.bb_pct.get_mean(),
            'HR/FB%': self.hr_fb_pct.get_mean(),
            'BABIP': self.babip.get_mean(),
        }

    def get_prediction_intervals(self, alpha: float = 0.9) -> dict:
        """Return prediction intervals for all stats."""
        return {
            'K%': self.k_pct.get_prediction_interval(alpha),
            'BB%': self.bb_pct.get_prediction_interval(alpha),
            'HR/FB%': self.hr_fb_pct.get_prediction_interval(alpha),
            'BABIP': self.babip.get_prediction_interval(alpha),
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
                'std': self.k_pct.get_std(),
            },
            'BB%': {
                'alpha': self.bb_pct.alpha,
                'beta': self.bb_pct.beta,
                'mean': self.bb_pct.get_mean(),
                'std': self.bb_pct.get_std(),
            },
            'HR/FB%': {
                'mu': self.hr_fb_pct.mu,
                'tau': self.hr_fb_pct.tau,
                'mean': self.hr_fb_pct.get_mean(),
                'std': self.hr_fb_pct.get_std(),
            },
            'BABIP': {
                'mu': self.babip.mu,
                'tau': self.babip.tau,
                'mean': self.babip.get_mean(),
                'std': self.babip.get_std(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict, likelihood_vars: dict) -> 'PitcherPosteriors':
        """Reconstruct PitcherPosteriors from serialized dictionary."""
        return cls(
            player_name=data['player_name'],
            team=data['team'],
            season=data['season'],
            k_pct=BetaComponentModel(
                alpha=data['K%']['alpha'],
                beta=data['K%']['beta'],
                stat_name='K%',
            ),
            bb_pct=BetaComponentModel(
                alpha=data['BB%']['alpha'],
                beta=data['BB%']['beta'],
                stat_name='BB%',
            ),
            hr_fb_pct=NormalComponentModel(
                mu=data['HR/FB%']['mu'],
                tau=data['HR/FB%']['tau'],
                likelihood_variance=likelihood_vars['HR/FB%'],
                stat_name='HR/FB%',
            ),
            babip=NormalComponentModel(
                mu=data['BABIP']['mu'],
                tau=data['BABIP']['tau'],
                likelihood_variance=likelihood_vars['BABIP'],
                stat_name='BABIP',
            ),
        )
