"""
In-Season Bayesian Update Logic for Pitching

This module handles weekly Bayesian updates for pitcher statistics using
closed-form conjugate update formulas:
- Beta-Binomial for K% and BB% (per batters faced)
- Normal-Normal for HR/FB% and BABIP
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from bayesian_batting.model import BetaComponentModel, NormalComponentModel
from .model import PitcherPosteriors


# Minimum cumulative thresholds before updating (BF or BIP based)
UPDATE_THRESHOLDS = {
    'K%': 30,       # Minimum cumulative BF before updating K%
    'BB%': 30,      # Minimum cumulative BF before updating BB%
    'HR/FB%': 50,   # Minimum cumulative BIP before updating HR/FB%
    'BABIP': 60,    # Minimum cumulative BIP before updating BABIP
}


@dataclass
class WeeklyPitchingStats:
    """Container for a pitcher's weekly performance data."""
    player_name: str
    team: str
    ip: float
    bf: int          # Batters faced
    k: int           # Strikeouts
    bb: int          # Walks
    hbp: int         # Hit by pitch
    h: int           # Hits allowed
    er: int          # Earned runs
    hr: int          # Home runs allowed
    bip: int         # Balls in play (BF - K - BB - HBP)
    fb_count: int    # Fly balls allowed

    @property
    def k_rate(self) -> float:
        return self.k / self.bf if self.bf > 0 else 0

    @property
    def bb_rate(self) -> float:
        return self.bb / self.bf if self.bf > 0 else 0

    @property
    def hr_fb_rate(self) -> float:
        return self.hr / self.fb_count if self.fb_count > 0 else 0

    @property
    def babip_val(self) -> float:
        return (self.h - self.hr) / self.bip if self.bip > 0 else 0


def aggregate_weekly_pitching_stats(
    game_log_df: pd.DataFrame,
    week_start: str,
    week_end: str,
) -> pd.DataFrame:
    """
    Aggregate game-by-game pitching data into weekly totals per pitcher.

    Args:
        game_log_df: DataFrame with game-level pitching stats
            Required columns: Player, Team, Date, IP, BF, H, ER, BB, HBP, K, HR
        week_start: Start date (YYYY-MM-DD)
        week_end: End date (YYYY-MM-DD)

    Returns:
        DataFrame with weekly aggregates per pitcher
    """
    mask = (game_log_df['Date'] >= week_start) & (game_log_df['Date'] <= week_end)
    week_data = game_log_df[mask].copy()

    if len(week_data) == 0:
        return pd.DataFrame()

    agg_cols = {
        'IP': 'sum',
        'BF': 'sum',
        'H': 'sum',
        'ER': 'sum',
        'BB': 'sum',
        'HBP': 'sum',
        'K': 'sum',
        'HR': 'sum',
    }

    weekly = week_data.groupby(['Player', 'Team']).agg(agg_cols).reset_index()

    # Calculate derived columns
    weekly['BIP'] = (weekly['BF'] - weekly['K'] - weekly['BB'] - weekly['HBP']).clip(lower=0)
    # Estimate fly ball count (default ~35% of BIP)
    if 'FB' in week_data.columns:
        weekly['FB_count'] = week_data.groupby(['Player', 'Team'])['FB'].sum().values
    else:
        weekly['FB_count'] = (weekly['BIP'] * 0.35).round().astype(int)

    return weekly


def parse_weekly_pitching_row(row: pd.Series) -> WeeklyPitchingStats:
    """Convert a DataFrame row to WeeklyPitchingStats object."""
    return WeeklyPitchingStats(
        player_name=row['Player'],
        team=row['Team'],
        ip=float(row.get('IP', 0)),
        bf=int(row.get('BF', 0)),
        k=int(row.get('K', 0)),
        bb=int(row.get('BB', 0)),
        hbp=int(row.get('HBP', 0)),
        h=int(row.get('H', 0)),
        er=int(row.get('ER', 0)),
        hr=int(row.get('HR', 0)),
        bip=int(row.get('BIP', max(row.get('BF', 0) - row.get('K', 0) - row.get('BB', 0) - row.get('HBP', 0), 0))),
        fb_count=int(row.get('FB_count', 0)),
    )


# ---------------------------------------------------------------------------
# Component update functions
# ---------------------------------------------------------------------------

def update_k_pct(
    prior: BetaComponentModel,
    weekly: WeeklyPitchingStats,
    cumulative_bf: int,
) -> BetaComponentModel:
    """Beta-Binomial update for K%: successes = K, failures = BF - K."""
    if cumulative_bf < UPDATE_THRESHOLDS['K%']:
        return prior
    return prior.update(weekly.k, weekly.bf - weekly.k)


def update_bb_pct(
    prior: BetaComponentModel,
    weekly: WeeklyPitchingStats,
    cumulative_bf: int,
) -> BetaComponentModel:
    """Beta-Binomial update for BB%: successes = BB, failures = BF - BB."""
    if cumulative_bf < UPDATE_THRESHOLDS['BB%']:
        return prior
    return prior.update(weekly.bb, weekly.bf - weekly.bb)


def update_hr_fb_pct(
    prior: NormalComponentModel,
    weekly: WeeklyPitchingStats,
    cumulative_bip: int,
) -> NormalComponentModel:
    """Normal-Normal update for HR/FB%."""
    if cumulative_bip < UPDATE_THRESHOLDS['HR/FB%']:
        return prior
    if weekly.fb_count == 0:
        return prior
    return prior.update_with_summary(weekly.hr_fb_rate, weekly.fb_count)


def update_babip(
    prior: NormalComponentModel,
    weekly: WeeklyPitchingStats,
    cumulative_bip: int,
) -> NormalComponentModel:
    """Normal-Normal update for BABIP."""
    if cumulative_bip < UPDATE_THRESHOLDS['BABIP']:
        return prior
    if weekly.bip == 0:
        return prior
    return prior.update_with_summary(weekly.babip_val, weekly.bip)


def update_pitcher_posteriors(
    current: PitcherPosteriors,
    weekly: WeeklyPitchingStats,
    cumulative_bf: int,
    cumulative_bip: int,
) -> PitcherPosteriors:
    """Apply weekly updates to all component posteriors for a pitcher."""
    new_k = update_k_pct(current.k_pct, weekly, cumulative_bf)
    new_bb = update_bb_pct(current.bb_pct, weekly, cumulative_bf)
    new_hr_fb = update_hr_fb_pct(current.hr_fb_pct, weekly, cumulative_bip)
    new_babip = update_babip(current.babip, weekly, cumulative_bip)

    return PitcherPosteriors(
        player_name=current.player_name,
        team=current.team,
        season=current.season,
        k_pct=new_k,
        bb_pct=new_bb,
        hr_fb_pct=new_hr_fb,
        babip=new_babip,
    )


# ---------------------------------------------------------------------------
# Season tracker
# ---------------------------------------------------------------------------

@dataclass
class PitchingSeasonTracker:
    """Tracks cumulative season stats for threshold checking."""
    player_name: str
    team: str
    season: int
    cumulative_ip: float = 0
    cumulative_bf: int = 0
    cumulative_bip: int = 0
    cumulative_k: int = 0
    cumulative_bb: int = 0
    cumulative_h: int = 0
    cumulative_hr: int = 0
    cumulative_er: int = 0
    week_count: int = 0

    def add_weekly(self, weekly: WeeklyPitchingStats):
        self.cumulative_ip += weekly.ip
        self.cumulative_bf += weekly.bf
        self.cumulative_bip += weekly.bip
        self.cumulative_k += weekly.k
        self.cumulative_bb += weekly.bb
        self.cumulative_h += weekly.h
        self.cumulative_hr += weekly.hr
        self.cumulative_er += weekly.er
        self.week_count += 1


# ---------------------------------------------------------------------------
# In-season updater class
# ---------------------------------------------------------------------------

class InSeasonPitchingUpdater:
    """
    Manages in-season Bayesian updates for a pitching staff.

    Usage:
        updater = InSeasonPitchingUpdater(preseason_posteriors)
        updater.process_week(weekly_stats_df)
        current = updater.get_current_posteriors()
    """

    def __init__(self, preseason_posteriors: Dict[str, PitcherPosteriors]):
        self.posteriors = preseason_posteriors.copy()
        self.trackers: Dict[str, PitchingSeasonTracker] = {}

        for name, post in self.posteriors.items():
            self.trackers[name] = PitchingSeasonTracker(
                player_name=post.player_name,
                team=post.team,
                season=post.season,
            )

    def process_week(self, weekly_stats_df: pd.DataFrame) -> Dict[str, bool]:
        """Process a week of pitching stats and update posteriors."""
        updates = {}

        for _, row in weekly_stats_df.iterrows():
            player_name = row['Player']
            weekly = parse_weekly_pitching_row(row)

            if player_name not in self.posteriors:
                updates[player_name] = False
                continue

            if player_name not in self.trackers:
                post = self.posteriors[player_name]
                self.trackers[player_name] = PitchingSeasonTracker(
                    player_name=post.player_name,
                    team=post.team,
                    season=post.season,
                )

            tracker = self.trackers[player_name]
            tracker.add_weekly(weekly)

            current = self.posteriors[player_name]
            updated = update_pitcher_posteriors(
                current, weekly,
                tracker.cumulative_bf,
                tracker.cumulative_bip,
            )

            self.posteriors[player_name] = updated
            updates[player_name] = True

        return updates

    def get_current_posteriors(self) -> Dict[str, PitcherPosteriors]:
        return self.posteriors

    def get_tracker(self, player_name: str) -> Optional[PitchingSeasonTracker]:
        return self.trackers.get(player_name)

    def get_prediction_df(self) -> pd.DataFrame:
        """Export current predictions as a DataFrame."""
        rows = []
        for name, post in self.posteriors.items():
            tracker = self.trackers.get(name)

            row = {
                'Player': post.player_name,
                'Team': post.team,
                'Season': post.season,
                'Weeks_Played': tracker.week_count if tracker else 0,
                'Cumulative_IP': tracker.cumulative_ip if tracker else 0,
                'Cumulative_BF': tracker.cumulative_bf if tracker else 0,
                'K%_mean': post.k_pct.get_mean(),
                'K%_std': post.k_pct.get_std(),
                'BB%_mean': post.bb_pct.get_mean(),
                'BB%_std': post.bb_pct.get_std(),
                'HR/FB%_mean': post.hr_fb_pct.get_mean(),
                'HR/FB%_std': post.hr_fb_pct.get_std(),
                'BABIP_mean': post.babip.get_mean(),
                'BABIP_std': post.babip.get_std(),
            }

            # Add percentiles
            for stat_name, model in [
                ('K%', post.k_pct),
                ('BB%', post.bb_pct),
                ('HR/FB%', post.hr_fb_pct),
                ('BABIP', post.babip),
            ]:
                percentiles = model.get_percentiles()
                for pct_name, value in percentiles.items():
                    row[f'{stat_name}_{pct_name}'] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def to_json(self) -> dict:
        result = {
            'week': max((t.week_count for t in self.trackers.values()), default=0),
            'players': {},
        }
        for name, post in self.posteriors.items():
            result['players'][name] = post.to_dict()
        return result


def simulate_season_updates(
    preseason_posteriors: Dict[str, PitcherPosteriors],
    full_season_df: pd.DataFrame,
    weeks: List[Tuple[str, str]],
) -> List[Dict[str, PitcherPosteriors]]:
    """Simulate weekly updates through a full season."""
    updater = InSeasonPitchingUpdater(preseason_posteriors)
    history = [preseason_posteriors.copy()]

    for week_start, week_end in weeks:
        weekly_df = aggregate_weekly_pitching_stats(full_season_df, week_start, week_end)
        updater.process_week(weekly_df)
        history.append(updater.get_current_posteriors().copy())

    return history
