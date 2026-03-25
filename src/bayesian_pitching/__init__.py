"""
Bayesian Pitching Performance Prediction Model

This package provides a Bayesian framework for predicting college baseball
pitching statistics (K%, BB%, HR/FB%, BABIP) with in-season updating
capabilities and team-level FIP aggregation.

Modules:
- model: PitcherPosteriors class (imports core classes from batting model)
- priors: Prior construction from historical data
- updates: In-season Bayesian update logic
- aggregation: Team-level Monte Carlo FIP simulation
- validation: Model validation and calibration checks
"""

from bayesian_batting.model import BetaComponentModel, NormalComponentModel
from .model import PitcherPosteriors
from .priors import (
    build_pitcher_priors,
    calculate_population_means,
    PitchingPopulationMeans,
    load_population_means,
    save_population_means,
)
from .updates import InSeasonPitchingUpdater, WeeklyPitchingStats
from .aggregation import (
    TeamPitchingAggregator,
    simulate_team_fip,
    components_to_fip,
    EXPECTED_ROTATION_SIZE,
)

__version__ = '0.1.0'
