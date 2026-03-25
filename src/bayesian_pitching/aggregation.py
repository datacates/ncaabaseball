"""
Team-Level FIP Aggregation with Monte Carlo Simulation

This module provides:
- Monte Carlo simulation to aggregate pitcher posteriors to team-level FIP
- Component-to-FIP conversion from rate statistics
- IP-share-weighted team aggregation
- Uncertainty propagation from pitcher to team level
"""

from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from bayesian_batting.model import BetaComponentModel, NormalComponentModel
from .model import PitcherPosteriors
from .priors import PitchingPopulationMeans

# Expected number of pitchers who get meaningful IP on a college staff
EXPECTED_ROTATION_SIZE = 12

# Prior strength for true freshmen with no data
FRESHMAN_STRENGTH = 20

# Prior strength for unknown roster padding (much weaker to allow more variance)
UNKNOWN_ROSTER_SPOT_STRENGTH = 3

# Only pad roster when coverage is below this threshold
MIN_COVERAGE_FOR_PADDING = 0.5

# Default FIP constant (NCAA D1 approximate; overridden from pop_means)
DEFAULT_FIP_CONSTANT = 3.10

# Conference quality adjustments for unknown pitchers
# Based on historical conference strength (Power 5 vs mid-major)
CONF_QUALITY_FACTORS = {
    'SEC': 1.05,        # Elite conference
    'ACC': 1.04,
    'Big 12': 1.03,
    'Pac-12': 1.02,
    'Big Ten': 1.02,
    # Mid-majors closer to 1.0 (league average)
    # Will default to 1.0 for unlisted conferences
}


def components_to_fip(
    k_pct: float,
    bb_pct: float,
    hr_fb_pct: float,
    babip: float,
    fip_constant: float = DEFAULT_FIP_CONSTANT,
    fb_pct: float = 0.35,
    ip_per_bf: float = 0.36,
    hbp_rate: float = 0.025,
) -> float:
    """
    Convert component statistics to FIP.

    FIP = ((13*HR + 3*(BB+HBP) - 2*K) / IP) + FIP_constant

    We convert per-BF rates to per-IP rates:
        BIP_rate = 1 - K% - BB% - HBP_rate
        HR_per_BF = HR/FB% * FB% * BIP_rate
        BF_per_IP = 1 / IP_per_BF
        Then: K/IP = K% * BF/IP, etc.

    Args:
        k_pct: Strikeout rate (K/BF)
        bb_pct: Walk rate (BB/BF)
        hr_fb_pct: HR per fly ball rate
        babip: BABIP allowed (not directly in FIP but used for ERA estimation)
        fip_constant: Calibration constant
        fb_pct: Fly ball fraction of BIP (league average ~0.35)
        ip_per_bf: Innings pitched per batter faced (league average ~0.36)
        hbp_rate: HBP rate per BF (league average ~0.025)
    """
    # Balls-in-play rate
    bip_rate = max(1 - k_pct - bb_pct - hbp_rate, 0.30)

    # HR per BF
    hr_per_bf = hr_fb_pct * fb_pct * bip_rate

    # Convert to per-IP
    bf_per_ip = 1.0 / ip_per_bf

    hr_per_ip = hr_per_bf * bf_per_ip
    bb_per_ip = bb_pct * bf_per_ip
    hbp_per_ip = hbp_rate * bf_per_ip
    k_per_ip = k_pct * bf_per_ip

    fip = (13 * hr_per_ip + 3 * (bb_per_ip + hbp_per_ip) - 2 * k_per_ip) + fip_constant

    return fip


def create_freshman_pitcher_posteriors(
    pop_means: PitchingPopulationMeans,
    team: str,
    season: int,
    player_index: int = 0,
    conference: Optional[str] = None,
    team_historical_fip: Optional[float] = None,
    is_true_freshman: bool = False,
) -> PitcherPosteriors:
    """
    Create a PitcherPosteriors using freshman/unknown pitcher prior distributions.

    Used to represent unknown/unrostered pitching spots with weak priors.

    Args:
        pop_means: Population statistics
        team: Team name
        season: Season year
        player_index: Index for naming unknown players
        conference: Conference name for conference-specific adjustments
        team_historical_fip: Team's historical FIP for quality adjustments
        is_true_freshman: If True, use stronger priors (20); if False (roster padding), use weak priors (3)
    """
    k_mean = pop_means.freshman_k_pct if pop_means.freshman_k_pct is not None else pop_means.overall_k_pct
    bb_mean = pop_means.freshman_bb_pct if pop_means.freshman_bb_pct is not None else pop_means.overall_bb_pct
    hr_fb_mean = pop_means.freshman_hr_fb_pct if pop_means.freshman_hr_fb_pct is not None else pop_means.overall_hr_fb_pct
    babip_mean = pop_means.freshman_babip if pop_means.freshman_babip is not None else pop_means.overall_babip

    # Apply conference-specific adjustments for unknown pitchers
    # Better conferences recruit better pitchers on average
    if conference:
        conf_factor = CONF_QUALITY_FACTORS.get(conference, 1.0)
        k_mean *= conf_factor
        bb_mean *= (2.0 - conf_factor)  # Inverse for BB% (lower is better)
        hr_fb_mean *= (2.0 - conf_factor)  # Inverse for HR/FB%

    # Apply team historical FIP adjustments
    # Teams with good historical FIP likely have better unknown pitchers
    if team_historical_fip is not None:
        # League average FIP is around 4.7
        # If team has 4.3 FIP (0.4 below average), adjust stats favorably
        fip_adjustment = (4.7 - team_historical_fip) * 0.15  # Scale factor
        k_mean *= (1.0 + fip_adjustment * 0.5)  # Better teams have higher K%
        bb_mean *= (1.0 - fip_adjustment * 0.3)  # Better teams have lower BB%
        hr_fb_mean *= (1.0 - fip_adjustment * 0.2)  # Better teams have lower HR/FB%

    # Clip to valid ranges
    k_mean = np.clip(k_mean, 0.15, 0.35)
    bb_mean = np.clip(bb_mean, 0.05, 0.15)
    hr_fb_mean = np.clip(hr_fb_mean, 0.06, 0.14)

    # Use different strength based on whether this is a true freshman or roster padding
    strength = FRESHMAN_STRENGTH if is_true_freshman else UNKNOWN_ROSTER_SPOT_STRENGTH

    k_model = BetaComponentModel.from_mean_strength(k_mean, strength, stat_name='K%')
    bb_model = BetaComponentModel.from_mean_strength(bb_mean, strength, stat_name='BB%')
    hr_fb_model = NormalComponentModel.from_mean_strength(
        hr_fb_mean, strength * 0.5, pop_means.hr_fb_pct_variance, stat_name='HR/FB%'
    )
    babip_model = NormalComponentModel.from_mean_strength(
        babip_mean, strength * 0.5, pop_means.babip_variance, stat_name='BABIP'
    )

    return PitcherPosteriors(
        player_name=f"Unknown_P{player_index + 1}",
        team=team,
        season=season,
        k_pct=k_model,
        bb_pct=bb_model,
        hr_fb_pct=hr_fb_model,
        babip=babip_model,
    )


def simulate_pitcher_fip(
    posteriors: PitcherPosteriors,
    n_samples: int = 1000,
    fip_constant: float = DEFAULT_FIP_CONSTANT,
    fb_pct: float = 0.35,
    ip_per_bf: float = 0.36,
    hbp_rate: float = 0.025,
) -> np.ndarray:
    """
    Generate FIP samples for a single pitcher via Monte Carlo.

    Returns array of FIP samples (lower = better).
    """
    k_samples = np.clip(posteriors.k_pct.sample(n_samples), 0.05, 0.50)
    bb_samples = np.clip(posteriors.bb_pct.sample(n_samples), 0.02, 0.25)
    hr_fb_samples = np.clip(posteriors.hr_fb_pct.sample(n_samples), 0.01, 0.30)
    babip_samples = np.clip(posteriors.babip.sample(n_samples), 0.20, 0.40)

    fip_samples = np.array([
        components_to_fip(k, bb, hr_fb, babip,
                          fip_constant=fip_constant,
                          fb_pct=fb_pct,
                          ip_per_bf=ip_per_bf,
                          hbp_rate=hbp_rate)
        for k, bb, hr_fb, babip in zip(k_samples, bb_samples, hr_fb_samples, babip_samples)
    ])

    return fip_samples


@dataclass
class TeamPitchingAggregation:
    """Results of team-level Monte Carlo FIP simulation."""
    team: str
    season: int
    week: Optional[int]
    n_pitchers: int
    fip_mean: float
    fip_std: float
    fip_p10: float
    fip_p25: float
    fip_p50: float
    fip_p75: float
    fip_p90: float
    pitcher_contributions: Dict[str, float]
    n_returning_pitchers: int = 0
    n_unknown_pitchers: int = 0
    ip_coverage: float = 1.0

    def to_dict(self) -> dict:
        return {
            'team': self.team,
            'season': self.season,
            'week': self.week,
            'n_pitchers': self.n_pitchers,
            'fip_mean': self.fip_mean,
            'fip_std': self.fip_std,
            'fip_p10': self.fip_p10,
            'fip_p25': self.fip_p25,
            'fip_p50': self.fip_p50,
            'fip_p75': self.fip_p75,
            'fip_p90': self.fip_p90,
            'n_returning_pitchers': self.n_returning_pitchers,
            'n_unknown_pitchers': self.n_unknown_pitchers,
            'ip_coverage': self.ip_coverage,
        }


def simulate_team_fip(
    team_posteriors: Dict[str, PitcherPosteriors],
    ip_shares: Dict[str, float],
    fip_constant: float = DEFAULT_FIP_CONSTANT,
    n_simulations: int = 10000,
    team_name: Optional[str] = None,
    season: Optional[int] = None,
    week: Optional[int] = None,
    pop_means: Optional[PitchingPopulationMeans] = None,
    fb_pct: float = 0.35,
    ip_per_bf: float = 0.36,
    hbp_rate: float = 0.025,
    conference: Optional[str] = None,
    team_historical_fip: Optional[float] = None,
) -> TeamPitchingAggregation:
    """
    Monte Carlo simulation for team-level FIP.

    For each simulation:
    1. Sample component stats for each pitcher from posteriors
    2. Convert to FIP
    3. Weight by IP share
    4. Aggregate to team FIP

    Only pads roster if coverage is below MIN_COVERAGE_FOR_PADDING threshold.

    Args:
        conference: Conference name for conference-specific unknown pitcher adjustments
        team_historical_fip: Team's 3-year average FIP for quality adjustments
    """
    n_returning = len(team_posteriors)
    n_unknown = 0
    resolved_name = team_name or "Unknown"
    resolved_season = season or 0

    # Only pad roster with unknown pitchers if coverage is below threshold
    coverage = n_returning / EXPECTED_ROTATION_SIZE
    if pop_means is not None and coverage < MIN_COVERAGE_FOR_PADDING:
        n_unknown = EXPECTED_ROTATION_SIZE - n_returning

        known_fraction = n_returning / EXPECTED_ROTATION_SIZE
        unknown_fraction = 1.0 - known_fraction
        unknown_share_each = unknown_fraction / n_unknown

        known_share_total = sum(ip_shares.get(name, 0) for name in team_posteriors)
        if known_share_total > 0:
            ip_shares = {
                name: (share / known_share_total) * known_fraction
                for name, share in ip_shares.items()
                if name in team_posteriors
            }
        else:
            ip_shares = {name: known_fraction / n_returning for name in team_posteriors}

        for i in range(n_unknown):
            unknown_posteriors = create_freshman_pitcher_posteriors(
                pop_means, resolved_name, resolved_season,
                player_index=i,
                conference=conference,
                team_historical_fip=team_historical_fip,
                is_true_freshman=False,  # Roster padding uses weak priors
            )
            unknown_name = unknown_posteriors.player_name
            team_posteriors = {**team_posteriors, unknown_name: unknown_posteriors}
            ip_shares[unknown_name] = unknown_share_each

    # PA coverage equivalent
    ip_coverage = n_returning / EXPECTED_ROTATION_SIZE if n_unknown > 0 else 1.0

    team_fip_samples = np.zeros(n_simulations)
    pitcher_contributions = {}

    # Normalize IP shares to sum to 1
    total_share = sum(ip_shares.values())
    if total_share > 0:
        normalized_shares = {k: v / total_share for k, v in ip_shares.items()}
    else:
        n_players = len(team_posteriors)
        normalized_shares = {k: 1.0 / n_players for k in team_posteriors.keys()}

    for name, posteriors in team_posteriors.items():
        share = normalized_shares.get(name, 0)
        if share <= 0:
            continue

        pitcher_fip = simulate_pitcher_fip(
            posteriors, n_simulations,
            fip_constant=fip_constant,
            fb_pct=fb_pct,
            ip_per_bf=ip_per_bf,
            hbp_rate=hbp_rate,
        )

        team_fip_samples += pitcher_fip * share
        pitcher_contributions[name] = float(np.mean(pitcher_fip) * share)

    return TeamPitchingAggregation(
        team=resolved_name,
        season=resolved_season,
        week=week,
        n_pitchers=n_returning + n_unknown,
        fip_mean=float(np.mean(team_fip_samples)),
        fip_std=float(np.std(team_fip_samples)),
        fip_p10=float(np.percentile(team_fip_samples, 10)),
        fip_p25=float(np.percentile(team_fip_samples, 25)),
        fip_p50=float(np.percentile(team_fip_samples, 50)),
        fip_p75=float(np.percentile(team_fip_samples, 75)),
        fip_p90=float(np.percentile(team_fip_samples, 90)),
        pitcher_contributions=pitcher_contributions,
        n_returning_pitchers=n_returning,
        n_unknown_pitchers=n_unknown,
        ip_coverage=ip_coverage,
    )


class TeamPitchingAggregator:
    """Manages team-level FIP aggregations for multiple teams."""

    def __init__(
        self,
        fip_constant: float = DEFAULT_FIP_CONSTANT,
        pop_means: Optional[PitchingPopulationMeans] = None,
        fb_pct: float = 0.35,
        ip_per_bf: float = 0.36,
        hbp_rate: float = 0.025,
    ):
        self.fip_constant = fip_constant
        self.pop_means = pop_means
        self.fb_pct = fb_pct
        self.ip_per_bf = ip_per_bf
        self.hbp_rate = hbp_rate

    def aggregate_team(
        self,
        team_name: str,
        team_posteriors: Dict[str, PitcherPosteriors],
        ip_shares: Optional[Dict[str, float]] = None,
        season: Optional[int] = None,
        week: Optional[int] = None,
        n_simulations: int = 10000,
        conference: Optional[str] = None,
        team_historical_fip: Optional[float] = None,
    ) -> TeamPitchingAggregation:
        if ip_shares is None:
            n_players = len(team_posteriors)
            ip_shares = {name: 1.0 / n_players for name in team_posteriors.keys()}

        return simulate_team_fip(
            team_posteriors=team_posteriors,
            ip_shares=ip_shares,
            fip_constant=self.fip_constant,
            n_simulations=n_simulations,
            team_name=team_name,
            season=season,
            week=week,
            pop_means=self.pop_means,
            fb_pct=self.fb_pct,
            ip_per_bf=self.ip_per_bf,
            hbp_rate=self.hbp_rate,
            conference=conference,
            team_historical_fip=team_historical_fip,
        )

    def aggregate_all_teams(
        self,
        all_posteriors: Dict[str, PitcherPosteriors],
        season: int,
        week: Optional[int] = None,
        n_simulations: int = 10000,
    ) -> pd.DataFrame:
        """Aggregate all teams from a pool of pitcher posteriors."""
        teams = {}
        for name, post in all_posteriors.items():
            team = post.team
            if team not in teams:
                teams[team] = {}
            teams[team][name] = post

        results = []
        for team_name, team_posteriors in teams.items():
            agg = self.aggregate_team(
                team_name=team_name,
                team_posteriors=team_posteriors,
                season=season,
                week=week,
                n_simulations=n_simulations,
            )
            results.append(agg.to_dict())

        df = pd.DataFrame(results)
        # Sort by FIP ascending (lower = better)
        df = df.sort_values('fip_mean', ascending=True).reset_index(drop=True)
        return df
