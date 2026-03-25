"""
In-Season Bayesian Update Logic

This module handles weekly Bayesian updates for player statistics using
closed-form conjugate update formulas:
- Beta-Binomial for K% and BB%
- Normal-Normal for ISO and BABIP
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .model import BetaComponentModel, NormalComponentModel, PlayerPosteriors


# Minimum thresholds before updating (to avoid noisy updates)
UPDATE_THRESHOLDS = {
    'K%': 20,      # Minimum PA before updating K%
    'BB%': 20,     # Minimum PA before updating BB%
    'ISO': 40,     # Minimum PA before updating ISO
    'BABIP': 50,   # Minimum balls in play before updating BABIP
}


@dataclass
class WeeklyStats:
    """Container for a player's weekly performance data."""
    player_name: str
    team: str
    pa: int  # Plate appearances
    ab: int  # At bats
    hits: int
    doubles: int
    triples: int
    hr: int
    bb: int  # Walks
    hbp: int  # Hit by pitch
    k: int  # Strikeouts
    sf: int  # Sacrifice flies
    bip: int  # Balls in play (AB - K - HR + SF)

    @property
    def singles(self) -> int:
        return self.hits - self.doubles - self.triples - self.hr

    @property
    def xbh(self) -> int:
        """Extra base hits."""
        return self.doubles + self.triples + self.hr

    @property
    def k_rate(self) -> float:
        """Strikeout rate."""
        return self.k / self.pa if self.pa > 0 else 0

    @property
    def bb_rate(self) -> float:
        """Walk rate."""
        return self.bb / self.pa if self.pa > 0 else 0

    @property
    def iso(self) -> float:
        """Isolated power = SLG - BA = (2B + 2*3B + 3*HR) / AB"""
        if self.ab == 0:
            return 0
        return (self.doubles + 2 * self.triples + 3 * self.hr) / self.ab

    @property
    def babip(self) -> float:
        """BABIP = (H - HR) / BIP"""
        if self.bip == 0:
            return 0
        return (self.hits - self.hr) / self.bip


def aggregate_weekly_stats(
    game_log_df: pd.DataFrame,
    week_start: str,
    week_end: str
) -> pd.DataFrame:
    """
    Aggregate game-by-game data into weekly totals per player.

    Args:
        game_log_df: DataFrame with game-level stats
            Required columns: Player, Team, Date, PA, AB, H, 2B, 3B, HR, BB, HBP, K, SF
        week_start: Start date (YYYY-MM-DD)
        week_end: End date (YYYY-MM-DD)

    Returns:
        DataFrame with weekly aggregates per player
    """
    # Filter to date range
    mask = (game_log_df['Date'] >= week_start) & (game_log_df['Date'] <= week_end)
    week_data = game_log_df[mask].copy()

    if len(week_data) == 0:
        return pd.DataFrame()

    # Aggregate by player
    agg_cols = {
        'PA': 'sum',
        'AB': 'sum',
        'H': 'sum',
        '2B': 'sum',
        '3B': 'sum',
        'HR': 'sum',
        'BB': 'sum',
        'HBP': 'sum',
        'K': 'sum',
        'SF': 'sum'
    }

    weekly = week_data.groupby(['Player', 'Team']).agg(agg_cols).reset_index()

    # Calculate balls in play
    weekly['BIP'] = weekly['AB'] - weekly['K'] - weekly['HR'] + weekly['SF']

    return weekly


def parse_weekly_row(row: pd.Series) -> WeeklyStats:
    """Convert a DataFrame row to WeeklyStats object."""
    return WeeklyStats(
        player_name=row['Player'],
        team=row['Team'],
        pa=int(row.get('PA', 0)),
        ab=int(row.get('AB', 0)),
        hits=int(row.get('H', 0)),
        doubles=int(row.get('2B', 0)),
        triples=int(row.get('3B', 0)),
        hr=int(row.get('HR', 0)),
        bb=int(row.get('BB', 0)),
        hbp=int(row.get('HBP', 0)),
        k=int(row.get('K', 0)),
        sf=int(row.get('SF', 0)),
        bip=int(row.get('BIP', row.get('AB', 0) - row.get('K', 0) - row.get('HR', 0)))
    )


def update_k_pct(
    prior: BetaComponentModel,
    weekly: WeeklyStats,
    cumulative_pa: int
) -> BetaComponentModel:
    """
    Update K% posterior with weekly observations.

    Beta-Binomial conjugate update:
    alpha_post = alpha_prior + strikeouts
    beta_post = beta_prior + non-strikeouts

    Args:
        prior: Prior BetaComponentModel for K%
        weekly: Weekly stats
        cumulative_pa: Total PA accumulated so far this season

    Returns:
        Updated BetaComponentModel (or prior if threshold not met)
    """
    if cumulative_pa < UPDATE_THRESHOLDS['K%']:
        return prior

    # K% update: successes = strikeouts, failures = non-strikeouts
    successes = weekly.k
    failures = weekly.pa - weekly.k

    return prior.update(successes, failures)


def update_bb_pct(
    prior: BetaComponentModel,
    weekly: WeeklyStats,
    cumulative_pa: int
) -> BetaComponentModel:
    """
    Update BB% posterior with weekly observations.

    Beta-Binomial conjugate update:
    alpha_post = alpha_prior + walks
    beta_post = beta_prior + non-walks

    Args:
        prior: Prior BetaComponentModel for BB%
        weekly: Weekly stats
        cumulative_pa: Total PA accumulated so far this season

    Returns:
        Updated BetaComponentModel (or prior if threshold not met)
    """
    if cumulative_pa < UPDATE_THRESHOLDS['BB%']:
        return prior

    # BB% update: successes = walks, failures = non-walks
    successes = weekly.bb
    failures = weekly.pa - weekly.bb

    return prior.update(successes, failures)


def update_iso(
    prior: NormalComponentModel,
    weekly: WeeklyStats,
    cumulative_pa: int
) -> NormalComponentModel:
    """
    Update ISO posterior with weekly observations.

    Normal-Normal conjugate update using summary statistics.

    Args:
        prior: Prior NormalComponentModel for ISO
        weekly: Weekly stats
        cumulative_pa: Total PA accumulated so far this season

    Returns:
        Updated NormalComponentModel (or prior if threshold not met)
    """
    if cumulative_pa < UPDATE_THRESHOLDS['ISO']:
        return prior

    if weekly.ab == 0:
        return prior

    # Calculate weekly ISO
    weekly_iso = weekly.iso

    # Update with summary statistics
    return prior.update_with_summary(weekly_iso, weekly.ab)


def update_babip(
    prior: NormalComponentModel,
    weekly: WeeklyStats,
    cumulative_bip: int
) -> NormalComponentModel:
    """
    Update BABIP posterior with weekly observations.

    Normal-Normal conjugate update using summary statistics.

    Args:
        prior: Prior NormalComponentModel for BABIP
        weekly: Weekly stats
        cumulative_bip: Total balls in play accumulated so far this season

    Returns:
        Updated NormalComponentModel (or prior if threshold not met)
    """
    if cumulative_bip < UPDATE_THRESHOLDS['BABIP']:
        return prior

    if weekly.bip == 0:
        return prior

    # Calculate weekly BABIP
    weekly_babip = weekly.babip

    # Update with summary statistics
    return prior.update_with_summary(weekly_babip, weekly.bip)


def update_player_posteriors(
    current: PlayerPosteriors,
    weekly: WeeklyStats,
    cumulative_pa: int,
    cumulative_bip: int
) -> PlayerPosteriors:
    """
    Apply weekly updates to all component posteriors for a player.

    Args:
        current: Current PlayerPosteriors
        weekly: Weekly stats
        cumulative_pa: Season-to-date PA
        cumulative_bip: Season-to-date balls in play

    Returns:
        Updated PlayerPosteriors
    """
    new_k = update_k_pct(current.k_pct, weekly, cumulative_pa)
    new_bb = update_bb_pct(current.bb_pct, weekly, cumulative_pa)
    new_iso = update_iso(current.iso, weekly, cumulative_pa)
    new_babip = update_babip(current.babip, weekly, cumulative_bip)

    return PlayerPosteriors(
        player_name=current.player_name,
        team=current.team,
        season=current.season,
        k_pct=new_k,
        bb_pct=new_bb,
        iso=new_iso,
        babip=new_babip
    )


@dataclass
class SeasonTracker:
    """
    Tracks cumulative season stats for threshold checking and update logic.
    """
    player_name: str
    team: str
    season: int
    cumulative_pa: int = 0
    cumulative_ab: int = 0
    cumulative_bip: int = 0
    cumulative_k: int = 0
    cumulative_bb: int = 0
    cumulative_h: int = 0
    cumulative_hr: int = 0
    week_count: int = 0

    def add_weekly(self, weekly: WeeklyStats):
        """Add weekly stats to cumulative totals."""
        self.cumulative_pa += weekly.pa
        self.cumulative_ab += weekly.ab
        self.cumulative_bip += weekly.bip
        self.cumulative_k += weekly.k
        self.cumulative_bb += weekly.bb
        self.cumulative_h += weekly.hits
        self.cumulative_hr += weekly.hr
        self.week_count += 1


class InSeasonUpdater:
    """
    Manages in-season Bayesian updates for a roster of players.

    Usage:
        updater = InSeasonUpdater(preseason_posteriors)
        updater.process_week(weekly_stats_df)
        current_predictions = updater.get_current_posteriors()
    """

    def __init__(self, preseason_posteriors: Dict[str, PlayerPosteriors]):
        """
        Initialize with preseason priors.

        Args:
            preseason_posteriors: Dict mapping player_name to PlayerPosteriors
        """
        self.posteriors = preseason_posteriors.copy()
        self.trackers: Dict[str, SeasonTracker] = {}

        # Initialize trackers for each player
        for name, post in self.posteriors.items():
            self.trackers[name] = SeasonTracker(
                player_name=post.player_name,
                team=post.team,
                season=post.season
            )

    def process_week(self, weekly_stats_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Process a week of stats and update posteriors.

        Args:
            weekly_stats_df: DataFrame with weekly aggregated stats

        Returns:
            Dict mapping player_name to whether they were updated
        """
        updates = {}

        for _, row in weekly_stats_df.iterrows():
            player_name = row['Player']
            weekly = parse_weekly_row(row)

            if player_name not in self.posteriors:
                # New player mid-season - would need to create priors
                updates[player_name] = False
                continue

            # Update cumulative tracker
            if player_name not in self.trackers:
                post = self.posteriors[player_name]
                self.trackers[player_name] = SeasonTracker(
                    player_name=post.player_name,
                    team=post.team,
                    season=post.season
                )

            tracker = self.trackers[player_name]
            tracker.add_weekly(weekly)

            # Update posteriors
            current = self.posteriors[player_name]
            updated = update_player_posteriors(
                current,
                weekly,
                tracker.cumulative_pa,
                tracker.cumulative_bip
            )

            self.posteriors[player_name] = updated
            updates[player_name] = True

        return updates

    def get_current_posteriors(self) -> Dict[str, PlayerPosteriors]:
        """Return current posterior distributions for all players."""
        return self.posteriors

    def get_tracker(self, player_name: str) -> Optional[SeasonTracker]:
        """Get cumulative season tracker for a player."""
        return self.trackers.get(player_name)

    def get_prediction_df(self) -> pd.DataFrame:
        """
        Export current predictions as a DataFrame.

        Returns:
            DataFrame with columns for each stat's mean and std
        """
        rows = []
        for name, post in self.posteriors.items():
            tracker = self.trackers.get(name)

            row = {
                'Player': post.player_name,
                'Team': post.team,
                'Season': post.season,
                'Weeks_Played': tracker.week_count if tracker else 0,
                'Cumulative_PA': tracker.cumulative_pa if tracker else 0,
                'K%_mean': post.k_pct.get_mean(),
                'K%_std': post.k_pct.get_std(),
                'BB%_mean': post.bb_pct.get_mean(),
                'BB%_std': post.bb_pct.get_std(),
                'ISO_mean': post.iso.get_mean(),
                'ISO_std': post.iso.get_std(),
                'BABIP_mean': post.babip.get_mean(),
                'BABIP_std': post.babip.get_std(),
            }

            # Add percentiles
            for stat_name, model in [
                ('K%', post.k_pct),
                ('BB%', post.bb_pct),
                ('ISO', post.iso),
                ('BABIP', post.babip)
            ]:
                percentiles = model.get_percentiles()
                for pct_name, value in percentiles.items():
                    row[f'{stat_name}_{pct_name}'] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def to_json(self) -> dict:
        """
        Serialize current state to JSON-compatible dict.

        Returns:
            Dict with posterior parameters for each player
        """
        result = {
            'week': max((t.week_count for t in self.trackers.values()), default=0),
            'players': {}
        }

        for name, post in self.posteriors.items():
            result['players'][name] = post.to_dict()

        return result


def simulate_season_updates(
    preseason_posteriors: Dict[str, PlayerPosteriors],
    full_season_df: pd.DataFrame,
    weeks: List[Tuple[str, str]]
) -> List[Dict[str, PlayerPosteriors]]:
    """
    Simulate weekly updates through a full season.

    Args:
        preseason_posteriors: Starting posteriors
        full_season_df: Full season game log data
        weeks: List of (start_date, end_date) tuples

    Returns:
        List of posterior states after each week
    """
    updater = InSeasonUpdater(preseason_posteriors)
    history = [preseason_posteriors.copy()]

    for week_start, week_end in weeks:
        weekly_df = aggregate_weekly_stats(full_season_df, week_start, week_end)
        updater.process_week(weekly_df)
        history.append(updater.get_current_posteriors().copy())

    return history
