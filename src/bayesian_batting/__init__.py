"""
Bayesian Batting Performance Prediction Model

This package provides a Bayesian framework for predicting college baseball
batting statistics (K%, BB%, ISO, BABIP) with in-season updating capabilities
and team-level aggregation.

Modules:
- model: Core Bayesian model classes (Beta and Normal distributions)
- priors: Prior construction from historical data
- updates: In-season Bayesian update logic
- aggregation: Team-level Monte Carlo simulation
- validation: Model validation and calibration checks
"""

from .model import BetaComponentModel, NormalComponentModel, PlayerPosteriors
from .priors import (
    build_player_priors,
    calculate_population_means,
    PopulationMeans,
    load_population_means,
    save_population_means
)
from .updates import InSeasonUpdater, WeeklyStats, aggregate_weekly_stats
from .aggregation import (
    TeamAggregator,
    WOBAWeights,
    simulate_team_woba,
    create_freshman_prior_posteriors,
    EXPECTED_LINEUP_SIZE
)

__version__ = '0.1.0'
