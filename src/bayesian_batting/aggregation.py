"""
Team-Level Aggregation with Monte Carlo Simulation

This module provides:
- Monte Carlo simulation to aggregate player posteriors to team-level wOBA
- Component-to-wOBA conversion using linear weights
- Uncertainty propagation from player to team level
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from .model import PlayerPosteriors, BetaComponentModel, NormalComponentModel
from .priors import PopulationMeans, FRESHMAN_PRIOR_STRENGTH

# Expected number of regular position players in a lineup
EXPECTED_LINEUP_SIZE = 9


# Default wOBA linear weights (FanGraphs 2023 approximation)
# These should be calculated from training data for college baseball
DEFAULT_WOBA_WEIGHTS = {
    'wBB': 0.69,   # Walk
    'wHBP': 0.72,  # Hit by pitch
    'w1B': 0.87,   # Single
    'w2B': 1.22,   # Double
    'w3B': 1.56,   # Triple
    'wHR': 1.95,   # Home run
}

# wOBA scale factor (to normalize to OBP scale)
WOBA_SCALE = 1.24


@dataclass
class WOBAWeights:
    """Container for wOBA linear weights."""
    w_bb: float
    w_hbp: float
    w_1b: float
    w_2b: float
    w_3b: float
    w_hr: float
    scale: float = 1.24

    @classmethod
    def from_dict(cls, d: dict) -> 'WOBAWeights':
        return cls(
            w_bb=d.get('wBB', DEFAULT_WOBA_WEIGHTS['wBB']),
            w_hbp=d.get('wHBP', DEFAULT_WOBA_WEIGHTS['wHBP']),
            w_1b=d.get('w1B', DEFAULT_WOBA_WEIGHTS['w1B']),
            w_2b=d.get('w2B', DEFAULT_WOBA_WEIGHTS['w2B']),
            w_3b=d.get('w3B', DEFAULT_WOBA_WEIGHTS['w3B']),
            w_hr=d.get('wHR', DEFAULT_WOBA_WEIGHTS['wHR']),
            scale=d.get('scale', WOBA_SCALE)
        )

    @classmethod
    def calculate_from_data(cls, df: pd.DataFrame) -> 'WOBAWeights':
        """
        Calculate wOBA weights from training data using PA-weighted OLS.

        Regresses observed wOBA on component rates: BB%, imputed HBP (2% of PA),
        and estimated 1B/2B/3B/HR rates from estimate_hit_distribution().
        Regression is weighted by PA to downweight small samples.

        Prints fitted weights and R² for evaluation. Does NOT replace defaults
        in the pipeline — caller decides whether to use the returned weights.

        Args:
            df: Training data with columns: wOBA, K%, BB%, ISO, BABIP, PA

        Returns:
            WOBAWeights fitted from data (defaults unchanged in pipeline)
        """
        required_cols = ['wOBA', 'K%', 'BB%', 'ISO', 'BABIP', 'PA']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Warning: Missing columns for wOBA weight calculation: {missing}")
            print("Returning default MLB weights.")
            return cls(w_bb=0.69, w_hbp=0.72, w_1b=0.87, w_2b=1.22,
                       w_3b=1.56, w_hr=1.95, scale=1.24)

        df_clean = df[required_cols].dropna().copy()
        # Require minimum PA to avoid noise from tiny samples
        df_clean = df_clean[df_clean['PA'] >= 20].reset_index(drop=True)

        print(f"\nFitting wOBA weights via PA-weighted OLS "
              f"({len(df_clean)} player-seasons, PA >= 20)...")

        # Build feature matrix: [BB%, HBP%, 1B_rate, 2B_rate, 3B_rate, HR_rate]
        # HBP imputed as a constant 2% of PA for every player-season
        HBP_RATE = 0.02
        rows = []
        for _, row in df_clean.iterrows():
            hits = estimate_hit_distribution(
                float(row['ISO']), float(row['BABIP']), float(row['K%'])
            )
            rows.append([
                float(row['BB%']),
                HBP_RATE,
                hits['1B'],
                hits['2B'],
                hits['3B'],
                hits['HR'],
            ])

        X = np.array(rows)            # (N, 6)  — no intercept; wOBA has none
        y = df_clean['wOBA'].values   # (N,)
        w = df_clean['PA'].values     # (N,)  — PA weights

        # PA-weighted OLS: scale rows by sqrt(PA), then ordinary lstsq
        sqrt_w = np.sqrt(w)
        X_w = X * sqrt_w[:, np.newaxis]
        y_w = y * sqrt_w

        coeffs, _, rank, _ = np.linalg.lstsq(X_w, y_w, rcond=None)

        # Weighted R²
        y_hat = X @ coeffs
        ss_res = float(np.sum(w * (y - y_hat) ** 2))
        y_mean_w = float(np.average(y, weights=w))
        ss_tot = float(np.sum(w * (y - y_mean_w) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        print("\n--- Fitted wOBA Weights (PA-weighted OLS) ---")
        labels = ['wBB (BB%)', 'wHBP (HBP%)', 'w1B (1B)', 'w2B (2B)', 'w3B (3B)', 'wHR (HR)']
        defaults = [DEFAULT_WOBA_WEIGHTS['wBB'], DEFAULT_WOBA_WEIGHTS['wHBP'],
                    DEFAULT_WOBA_WEIGHTS['w1B'], DEFAULT_WOBA_WEIGHTS['w2B'],
                    DEFAULT_WOBA_WEIGHTS['w3B'], DEFAULT_WOBA_WEIGHTS['wHR']]
        for label, coef, default in zip(labels, coeffs, defaults):
            print(f"  {label:18s}  fitted={coef:+.4f}  default={default:.2f}  "
                  f"diff={coef - default:+.4f}")
        print(f"  PA-weighted R²: {r_squared:.4f}")
        print(f"  N: {len(df_clean)}, PA range: {int(w.min())}-{int(w.max())}")
        print("----------------------------------------------")
        print("NOTE: defaults are unchanged in the pipeline — "
              "replace DEFAULT_WOBA_WEIGHTS manually if desired.\n")

        return cls(
            w_bb=float(coeffs[0]),
            w_hbp=float(coeffs[1]),
            w_1b=float(coeffs[2]),
            w_2b=float(coeffs[3]),
            w_3b=float(coeffs[4]),
            w_hr=float(coeffs[5]),
            scale=1.24
        )


def estimate_hit_distribution(iso: float, babip: float, k_pct: float) -> Dict[str, float]:
    """
    Estimate the distribution of hit types from ISO and BABIP.

    This is an approximation based on typical relationships:
    - Higher ISO = more extra base hits (especially HR)
    - BABIP affects overall hit rate on balls in play

    Args:
        iso: Isolated power
        babip: Batting average on balls in play
        k_pct: Strikeout rate

    Returns:
        Dict with estimated rates per PA for 1B, 2B, 3B, HR
    """
    # Contact rate
    contact_rate = 1 - k_pct

    # Approximate at-bat outcomes
    # Balls in play rate (excluding HR and K)
    # Assume ~8% of PA are walks, ~2% HBP

    # HR rate can be estimated from ISO
    # Rough approximation: HR contribute ~3 bases to ISO
    # So HR_rate ≈ ISO / 3 * contact_rate (very rough)
    hr_rate = min(iso / 3.5, 0.10)  # Cap at 10%

    # Remaining ISO comes from XBH
    remaining_iso = iso - (hr_rate * 3)

    # 2B contributes 1 base, 3B contributes 2 bases to ISO
    # Assume 3B/2B ratio is about 0.1
    xbh_rate = remaining_iso / 1.1 if remaining_iso > 0 else 0
    triple_rate = xbh_rate * 0.1
    double_rate = xbh_rate * 0.9

    # Total hit rate from BABIP and contact
    # H = BABIP * BIP + HR
    # BIP ≈ contact_rate - HR_rate
    bip_rate = contact_rate - hr_rate
    hit_rate = babip * bip_rate + hr_rate

    # Singles = hits - XBH - HR
    single_rate = max(hit_rate - hr_rate - double_rate - triple_rate, 0)

    return {
        '1B': single_rate,
        '2B': double_rate,
        '3B': triple_rate,
        'HR': hr_rate
    }


def components_to_woba(
    k_pct: float,
    bb_pct: float,
    iso: float,
    babip: float,
    weights: WOBAWeights,
    hbp_rate: float = 0.02
) -> float:
    """
    Convert component statistics to wOBA.

    wOBA = (wBB*BB + wHBP*HBP + w1B*1B + w2B*2B + w3B*3B + wHR*HR) / PA

    Args:
        k_pct: Strikeout rate
        bb_pct: Walk rate
        iso: Isolated power
        babip: BABIP
        weights: wOBA linear weights
        hbp_rate: Hit by pitch rate (default 2%)

    Returns:
        Estimated wOBA
    """
    # Get hit distribution
    hits = estimate_hit_distribution(iso, babip, k_pct)

    # Calculate wOBA
    woba = (
        weights.w_bb * bb_pct +
        weights.w_hbp * hbp_rate +
        weights.w_1b * hits['1B'] +
        weights.w_2b * hits['2B'] +
        weights.w_3b * hits['3B'] +
        weights.w_hr * hits['HR']
    )

    return woba


def create_freshman_prior_posteriors(
    pop_means: PopulationMeans,
    team: str,
    season: int,
    player_index: int = 0
) -> PlayerPosteriors:
    """
    Create a PlayerPosteriors using freshman prior distributions.

    Used to represent unknown/unrostered lineup spots with weak priors
    matching the freshman population means.

    Args:
        pop_means: Population means (must have freshman means populated)
        team: Team name for the posteriors object
        season: Season year
        player_index: Index to differentiate multiple unknown players

    Returns:
        PlayerPosteriors with weak freshman priors
    """
    # Use true freshman means if available, otherwise overall means
    k_mean = pop_means.freshman_k_pct if pop_means.freshman_k_pct is not None else pop_means.overall_k_pct
    bb_mean = pop_means.freshman_bb_pct if pop_means.freshman_bb_pct is not None else pop_means.overall_bb_pct
    iso_mean = pop_means.freshman_iso if pop_means.freshman_iso is not None else pop_means.overall_iso
    babip_mean = pop_means.freshman_babip if pop_means.freshman_babip is not None else pop_means.overall_babip

    # Weak priors matching the freshman prior strength
    strength = FRESHMAN_PRIOR_STRENGTH

    k_model = BetaComponentModel.from_mean_strength(k_mean, strength, stat_name='K%')
    bb_model = BetaComponentModel.from_mean_strength(bb_mean, strength, stat_name='BB%')
    iso_model = NormalComponentModel.from_mean_strength(
        iso_mean, strength * 0.5, pop_means.iso_variance, stat_name='ISO'
    )
    babip_model = NormalComponentModel.from_mean_strength(
        babip_mean, strength * 0.5, pop_means.babip_variance, stat_name='BABIP'
    )

    return PlayerPosteriors(
        player_name=f"Unknown_{player_index + 1}",
        team=team,
        season=season,
        k_pct=k_model,
        bb_pct=bb_model,
        iso=iso_model,
        babip=babip_model
    )


def simulate_player_woba(
    posteriors: PlayerPosteriors,
    weights: WOBAWeights,
    n_samples: int = 1000
) -> np.ndarray:
    """
    Generate wOBA samples for a single player via Monte Carlo.

    Args:
        posteriors: Player's posterior distributions
        weights: wOBA linear weights
        n_samples: Number of samples to draw

    Returns:
        Array of wOBA samples
    """
    # Draw samples from each component posterior
    k_samples = posteriors.k_pct.sample(n_samples)
    bb_samples = posteriors.bb_pct.sample(n_samples)
    iso_samples = posteriors.iso.sample(n_samples)
    babip_samples = posteriors.babip.sample(n_samples)

    # Clip to reasonable bounds
    k_samples = np.clip(k_samples, 0.05, 0.50)
    bb_samples = np.clip(bb_samples, 0.02, 0.25)
    iso_samples = np.clip(iso_samples, 0.01, 0.40)
    babip_samples = np.clip(babip_samples, 0.20, 0.45)

    # Convert each sample to wOBA
    woba_samples = np.array([
        components_to_woba(k, bb, iso, babip, weights)
        for k, bb, iso, babip in zip(k_samples, bb_samples, iso_samples, babip_samples)
    ])

    return woba_samples


@dataclass
class TeamAggregation:
    """Results of team-level Monte Carlo simulation."""
    team: str
    season: int
    week: Optional[int]
    n_players: int
    woba_mean: float
    woba_std: float
    woba_p10: float
    woba_p25: float
    woba_p50: float
    woba_p75: float
    woba_p90: float
    player_contributions: Dict[str, float]  # Player name -> weighted wOBA contribution
    n_returning_players: int = 0   # Count of actual returning players
    n_unknown_players: int = 0     # Count of freshman priors added (9 - n_returning)
    pa_coverage: float = 1.0       # Fraction of PA assigned to returning players

    def to_dict(self) -> dict:
        return {
            'team': self.team,
            'season': self.season,
            'week': self.week,
            'n_players': self.n_players,
            'woba_mean': self.woba_mean,
            'woba_std': self.woba_std,
            'woba_p10': self.woba_p10,
            'woba_p25': self.woba_p25,
            'woba_p50': self.woba_p50,
            'woba_p75': self.woba_p75,
            'woba_p90': self.woba_p90,
            'n_returning_players': self.n_returning_players,
            'n_unknown_players': self.n_unknown_players,
            'pa_coverage': self.pa_coverage,
        }


def simulate_team_woba(
    team_posteriors: Dict[str, PlayerPosteriors],
    pa_shares: Dict[str, float],
    weights: WOBAWeights,
    n_simulations: int = 10000,
    team_name: Optional[str] = None,
    season: Optional[int] = None,
    week: Optional[int] = None,
    pop_means: Optional[PopulationMeans] = None
) -> TeamAggregation:
    """
    Monte Carlo simulation for team-level wOBA.

    For each simulation:
    1. Sample wOBA for each player from their posteriors
    2. Weight by PA share
    3. Aggregate to team total

    If fewer than EXPECTED_LINEUP_SIZE (9) players are present and pop_means
    is provided, the roster is padded with unknown players using freshman
    prior distributions. This prevents teams with few returning players from
    being artificially ranked high.

    Args:
        team_posteriors: Dict of player posteriors
        pa_shares: Dict of predicted PA shares per player (should sum to ~1.0)
        weights: wOBA linear weights
        n_simulations: Number of Monte Carlo iterations
        team_name: Team identifier
        season: Season year
        week: Week number (None for preseason)
        pop_means: Population means for freshman priors (enables roster padding)

    Returns:
        TeamAggregation with distribution statistics
    """
    n_returning = len(team_posteriors)
    n_unknown = 0
    resolved_name = team_name or "Unknown"
    resolved_season = season or 0

    # Pad roster with freshman priors if needed
    if pop_means is not None and n_returning < EXPECTED_LINEUP_SIZE:
        n_unknown = EXPECTED_LINEUP_SIZE - n_returning

        # Each player (known + unknown) gets an equal share of a 9-player lineup.
        # Known players keep their relative PA share proportions, but are scaled
        # down to (n_returning / 9) of total PA. Unknown players split the rest.
        known_fraction = n_returning / EXPECTED_LINEUP_SIZE
        unknown_fraction = 1.0 - known_fraction
        unknown_share_each = unknown_fraction / n_unknown

        # Scale down known players' shares to make room for unknowns
        known_share_total = sum(pa_shares.get(name, 0) for name in team_posteriors)
        if known_share_total > 0:
            pa_shares = {
                name: (share / known_share_total) * known_fraction
                for name, share in pa_shares.items()
                if name in team_posteriors
            }
        else:
            pa_shares = {name: known_fraction / n_returning for name in team_posteriors}

        # Add unknown players with freshman priors
        for i in range(n_unknown):
            unknown_posteriors = create_freshman_prior_posteriors(
                pop_means, resolved_name, resolved_season, player_index=i
            )
            unknown_name = unknown_posteriors.player_name
            team_posteriors = {**team_posteriors, unknown_name: unknown_posteriors}
            pa_shares[unknown_name] = unknown_share_each

    # Calculate PA coverage: fraction of PA going to known returning players
    if n_unknown > 0:
        pa_coverage = n_returning / EXPECTED_LINEUP_SIZE
    else:
        pa_coverage = 1.0

    team_woba_samples = np.zeros(n_simulations)
    player_contributions = {}

    # Normalize PA shares to sum to 1
    total_share = sum(pa_shares.values())
    if total_share > 0:
        normalized_shares = {k: v / total_share for k, v in pa_shares.items()}
    else:
        # Equal shares if no PA data
        n_players = len(team_posteriors)
        normalized_shares = {k: 1.0 / n_players for k in team_posteriors.keys()}

    for name, posteriors in team_posteriors.items():
        share = normalized_shares.get(name, 0)
        if share <= 0:
            continue

        # Sample wOBA for this player
        player_woba = simulate_player_woba(posteriors, weights, n_simulations)

        # Add weighted contribution to team total
        team_woba_samples += player_woba * share

        # Track mean contribution
        player_contributions[name] = float(np.mean(player_woba) * share)

    # Calculate distribution statistics
    return TeamAggregation(
        team=resolved_name,
        season=resolved_season,
        week=week,
        n_players=n_returning + n_unknown,
        woba_mean=float(np.mean(team_woba_samples)),
        woba_std=float(np.std(team_woba_samples)),
        woba_p10=float(np.percentile(team_woba_samples, 10)),
        woba_p25=float(np.percentile(team_woba_samples, 25)),
        woba_p50=float(np.percentile(team_woba_samples, 50)),
        woba_p75=float(np.percentile(team_woba_samples, 75)),
        woba_p90=float(np.percentile(team_woba_samples, 90)),
        player_contributions=player_contributions,
        n_returning_players=n_returning,
        n_unknown_players=n_unknown,
        pa_coverage=pa_coverage
    )


class TeamAggregator:
    """
    Manages team-level aggregations for multiple teams.
    """

    def __init__(
        self,
        weights: Optional[WOBAWeights] = None,
        pa_model_path: Optional[Path] = None,
        pop_means: Optional[PopulationMeans] = None
    ):
        """
        Initialize the aggregator.

        Args:
            weights: wOBA linear weights (uses defaults if None)
            pa_model_path: Path to PA share model (if None, uses equal shares)
            pop_means: Population means for freshman roster padding (if None, no padding)
        """
        self.weights = weights or WOBAWeights.from_dict(DEFAULT_WOBA_WEIGHTS)
        self.pop_means = pop_means
        self.pa_model = None

        if pa_model_path and pa_model_path.exists():
            self.pa_model = joblib.load(pa_model_path)

    def predict_pa_shares(
        self,
        roster_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Predict PA shares for a roster.

        Args:
            roster_df: DataFrame with player features for PA model

        Returns:
            Dict mapping player name to predicted PA share
        """
        if self.pa_model is None:
            # Equal shares as fallback
            n_players = len(roster_df)
            return {row['Player']: 1.0 / n_players for _, row in roster_df.iterrows()}

        # Use PA share model
        # Note: This requires the same feature engineering as in pa_share_model.py
        # For now, return equal shares as placeholder
        n_players = len(roster_df)
        return {row['Player']: 1.0 / n_players for _, row in roster_df.iterrows()}

    def aggregate_team(
        self,
        team_name: str,
        team_posteriors: Dict[str, PlayerPosteriors],
        pa_shares: Optional[Dict[str, float]] = None,
        season: Optional[int] = None,
        week: Optional[int] = None,
        n_simulations: int = 10000
    ) -> TeamAggregation:
        """
        Aggregate a single team's player posteriors to team-level wOBA.

        Args:
            team_name: Team identifier
            team_posteriors: Player posteriors for this team
            pa_shares: PA shares (if None, uses equal shares)
            season: Season year
            week: Week number
            n_simulations: Number of Monte Carlo iterations

        Returns:
            TeamAggregation result
        """
        if pa_shares is None:
            n_players = len(team_posteriors)
            pa_shares = {name: 1.0 / n_players for name in team_posteriors.keys()}

        return simulate_team_woba(
            team_posteriors=team_posteriors,
            pa_shares=pa_shares,
            weights=self.weights,
            n_simulations=n_simulations,
            team_name=team_name,
            season=season,
            week=week,
            pop_means=self.pop_means
        )

    def aggregate_all_teams(
        self,
        all_posteriors: Dict[str, PlayerPosteriors],
        season: int,
        week: Optional[int] = None,
        n_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Aggregate all teams from a pool of player posteriors.

        Args:
            all_posteriors: Dict of all player posteriors
            season: Season year
            week: Week number
            n_simulations: Number of Monte Carlo iterations

        Returns:
            DataFrame with team-level aggregations
        """
        # Group players by team
        teams = {}
        for name, post in all_posteriors.items():
            team = post.team
            if team not in teams:
                teams[team] = {}
            teams[team][name] = post

        # Aggregate each team
        results = []
        for team_name, team_posteriors in teams.items():
            agg = self.aggregate_team(
                team_name=team_name,
                team_posteriors=team_posteriors,
                season=season,
                week=week,
                n_simulations=n_simulations
            )
            results.append(agg.to_dict())

        df = pd.DataFrame(results)

        # Sort by wOBA mean descending
        df = df.sort_values('woba_mean', ascending=False).reset_index(drop=True)

        return df


def calculate_woba_weights_from_data(df: pd.DataFrame) -> WOBAWeights:
    """
    Calculate wOBA weights from training data.

    Uses the relationship between component stats and overall wOBA
    to derive weights via regression.

    Args:
        df: Training data with wOBA and component rates

    Returns:
        Calculated WOBAWeights
    """
    # This is a simplified implementation
    # A full implementation would use run expectancy and linear regression

    # For now, we'll estimate weights from the correlation structure
    # and typical MLB values as a baseline

    # Calculate average rates
    avg_bb = df['BB%'].mean()
    avg_k = df['K%'].mean()
    avg_iso = df['ISO'].mean()
    avg_babip = df['BABIP'].mean()
    avg_woba = df['wOBA'].mean()

    # Scale MLB weights to college context
    # College typically has higher variance
    scale_factor = avg_woba / 0.320  # 0.320 is typical MLB wOBA

    return WOBAWeights(
        w_bb=0.69 * scale_factor,
        w_hbp=0.72 * scale_factor,
        w_1b=0.87 * scale_factor,
        w_2b=1.22 * scale_factor,
        w_3b=1.56 * scale_factor,
        w_hr=1.95 * scale_factor,
        scale=1.24
    )


def reconstruct_posteriors_from_projections(df: pd.DataFrame) -> Dict[str, PlayerPosteriors]:
    """
    Reconstruct PlayerPosteriors objects from saved projection mean/std.

    Since the CSV only has means and stds, we reconstruct the Beta and Normal
    parameters from those moments.

    For Beta: Solve for alpha, beta given mean and variance
    For Normal: Use mean as mu, and estimate tau from std

    Args:
        df: DataFrame with projection columns (K%_mean, K%_std, etc.)

    Returns:
        Dict mapping player names to PlayerPosteriors objects
    """
    posteriors = {}

    # Likelihood variances from training
    ISO_LIKELIHOOD_VAR = 0.00683
    BABIP_LIKELIHOOD_VAR = 0.00265

    for _, row in df.iterrows():
        player_name = row['Player']

        # Reconstruct Beta distributions for K% and BB%
        # Beta distribution: mean = alpha / (alpha + beta)
        #                    var = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        # Given mean and var, solve for alpha and beta

        def beta_from_mean_std(mean, std):
            """Recover alpha, beta from mean and std."""
            mean = np.clip(mean, 0.001, 0.999)
            var = std ** 2
            var = np.clip(var, 0.0001, mean * (1 - mean) * 0.99)  # Must be less than max variance

            # From Beta distribution formulas
            # mean = alpha / (alpha + beta)
            # var = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))
            # Solving: alpha + beta = mean * (1 - mean) / var - 1
            alpha_plus_beta = mean * (1 - mean) / var - 1
            alpha_plus_beta = max(alpha_plus_beta, 2)  # Must be > 1

            alpha = mean * alpha_plus_beta
            beta = (1 - mean) * alpha_plus_beta

            return max(alpha, 0.1), max(beta, 0.1)

        k_alpha, k_beta = beta_from_mean_std(row['K%_mean'], row['K%_std'])
        bb_alpha, bb_beta = beta_from_mean_std(row['BB%_mean'], row['BB%_std'])

        k_model = BetaComponentModel(
            alpha=k_alpha,
            beta=k_beta,
            stat_name='K%'
        )

        bb_model = BetaComponentModel(
            alpha=bb_alpha,
            beta=bb_beta,
            stat_name='BB%'
        )

        # Reconstruct Normal distributions for ISO and BABIP
        # For Normal: mu = mean, and tau relates to posterior std
        # posterior_variance = 1/tau + likelihood_variance
        # So: tau = 1 / (posterior_variance - likelihood_variance)

        iso_mean = row['ISO_mean']
        iso_var = row['ISO_std'] ** 2
        iso_tau = 1.0 / max(iso_var, 0.0001)  # tau = 1/variance

        babip_mean = row['BABIP_mean']
        babip_var = row['BABIP_std'] ** 2
        babip_tau = 1.0 / max(babip_var, 0.0001)

        iso_model = NormalComponentModel(
            mu=iso_mean,
            tau=iso_tau,
            likelihood_variance=ISO_LIKELIHOOD_VAR,
            stat_name='ISO'
        )

        babip_model = NormalComponentModel(
            mu=babip_mean,
            tau=babip_tau,
            likelihood_variance=BABIP_LIKELIHOOD_VAR,
            stat_name='BABIP'
        )

        posteriors[player_name] = PlayerPosteriors(
            player_name=player_name,
            team=row['Team'],
            season=row['Season'],
            k_pct=k_model,
            bb_pct=bb_model,
            iso=iso_model,
            babip=babip_model
        )

    return posteriors


def estimate_pa_shares(df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate PA shares for players based on prior year PA.

    Uses a simple heuristic: players with more prior PA get larger shares.
    Could be replaced with a more sophisticated PA projection model.

    Args:
        df: DataFrame with Prior_PA column

    Returns:
        Dict mapping player names to PA share estimates (0-1)
    """
    # Players without prior PA get minimum share
    min_pa = 50

    # Cap maximum PA to avoid overweighting one player
    max_pa = 350

    pa_values = {}
    for _, row in df.iterrows():
        prior_pa = row.get('Prior_PA', min_pa)
        # Clip to reasonable range
        pa = np.clip(prior_pa, min_pa, max_pa)
        pa_values[row['Player']] = pa

    # Normalize to shares
    total_pa = sum(pa_values.values())
    pa_shares = {name: pa / total_pa for name, pa in pa_values.items()}

    return pa_shares


def aggregate_fg_projections_to_teams(
    projections_df: pd.DataFrame,
    pop_means: Optional[PopulationMeans] = None,
    weights: Optional[WOBAWeights] = None,
    n_simulations: int = 10000,
    min_players_for_projection: int = 3
) -> pd.DataFrame:
    """
    Aggregate FanGraphs player projections to team-level wOBA distributions.

    This function:
    1. Reconstructs posterior distributions from saved projection parameters
    2. Estimates PA shares for each player
    3. Runs Monte Carlo simulation to aggregate to team level
    4. Optionally pads rosters with freshman priors for incomplete teams

    Args:
        projections_df: Player projections with posterior parameters
        pop_means: Population means for freshman padding (optional)
        weights: wOBA weights (uses defaults if None)
        n_simulations: Number of Monte Carlo iterations
        min_players_for_projection: Minimum players required to project a team

    Returns:
        DataFrame with team-level wOBA projections and uncertainty
    """
    if weights is None:
        weights = WOBAWeights.from_dict(DEFAULT_WOBA_WEIGHTS)

    # Reconstruct posteriors from projection parameters
    print("Reconstructing player posteriors from projections...")
    all_posteriors = reconstruct_posteriors_from_projections(projections_df)

    # Estimate PA shares
    print("Estimating PA shares...")
    all_pa_shares = estimate_pa_shares(projections_df)

    # Group by team
    teams = {}
    team_pa_shares = {}

    for _, row in projections_df.iterrows():
        player_name = row['Player']
        team_name = row['Team']

        if team_name not in teams:
            teams[team_name] = {}
            team_pa_shares[team_name] = {}

        teams[team_name][player_name] = all_posteriors[player_name]
        team_pa_shares[team_name][player_name] = all_pa_shares[player_name]

    # Filter teams with too few players
    teams = {
        name: roster
        for name, roster in teams.items()
        if len(roster) >= min_players_for_projection
    }

    print(f"\nAggregating {len(teams)} teams with >= {min_players_for_projection} players...")

    # Aggregate each team
    results = []
    for i, (team_name, team_posteriors) in enumerate(teams.items()):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(teams)} teams...")

        agg = simulate_team_woba(
            team_posteriors=team_posteriors,
            pa_shares=team_pa_shares[team_name],
            weights=weights,
            n_simulations=n_simulations,
            team_name=team_name,
            season=projections_df['Season'].iloc[0],
            week=None,
            pop_means=pop_means
        )

        result = agg.to_dict()

        # Add component means for reference
        result['avg_K%'] = np.mean([p.k_pct.get_mean() for p in team_posteriors.values()])
        result['avg_BB%'] = np.mean([p.bb_pct.get_mean() for p in team_posteriors.values()])
        result['avg_ISO'] = np.mean([p.iso.get_mean() for p in team_posteriors.values()])
        result['avg_BABIP'] = np.mean([p.babip.get_mean() for p in team_posteriors.values()])

        results.append(result)

    print(f"Completed aggregation for {len(results)} teams")

    df = pd.DataFrame(results)

    # Sort by wOBA mean descending
    df = df.sort_values('woba_mean', ascending=False).reset_index(drop=True)

    # Add rank
    df['rank'] = range(1, len(df) + 1)

    return df
