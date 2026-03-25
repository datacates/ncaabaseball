"""
Prior Construction for Bayesian Batting Model

This module handles construction of preseason priors for each player based on:
- Prior year performance (regressed to population mean)
- Class-specific population means
- Conference strength adjustment
- Park factor adjustment
- Batted ball profile for BABIP
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .model import BetaComponentModel, NormalComponentModel, PlayerPosteriors


# Reliability multipliers for prior strength weighting
# Higher values = more weight on prior performance, less regression
# These are tuned based on validation results to balance prediction accuracy and calibration
RELIABILITY_MULTIPLIERS = {
    'K%': 0.9,    # Highly persistent, skill-based
    'BB%': 0.7,   # Persistent, plate discipline
    'ISO': 0.35,  # Moderate variance, some skill signal
    'BABIP': 0.18 # Very noisy, needs heavy regression to mean
}

# Reference PA for population weight (typical full season)
# Lower = more regression to population mean
REFERENCE_PA = 250

# Minimum prior strength for freshmen/new players (weak prior)
FRESHMAN_PRIOR_STRENGTH = 25


@dataclass
class PopulationMeans:
    """Class-specific population means for each statistic."""
    k_pct: Dict[str, float]
    bb_pct: Dict[str, float]
    iso: Dict[str, float]
    babip: Dict[str, float]

    # Overall means across all classes
    overall_k_pct: float
    overall_bb_pct: float
    overall_iso: float
    overall_babip: float

    # Empirical variances for normal distributions
    iso_variance: float
    babip_variance: float

    # League average BABIP by batted ball type
    babip_gb: float  # Ground balls
    babip_ld: float  # Line drives
    babip_fb: float  # Fly balls

    # Mean conference strength for adjustment
    mean_conf_strength: float

    # True freshman means computed from all raw freshman data
    # (not just freshmen with prior data in the training set)
    freshman_k_pct: Optional[float] = None
    freshman_bb_pct: Optional[float] = None
    freshman_iso: Optional[float] = None
    freshman_babip: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        d = {
            'k_pct': self.k_pct,
            'bb_pct': self.bb_pct,
            'iso': self.iso,
            'babip': self.babip,
            'overall_k_pct': self.overall_k_pct,
            'overall_bb_pct': self.overall_bb_pct,
            'overall_iso': self.overall_iso,
            'overall_babip': self.overall_babip,
            'iso_variance': self.iso_variance,
            'babip_variance': self.babip_variance,
            'babip_gb': self.babip_gb,
            'babip_ld': self.babip_ld,
            'babip_fb': self.babip_fb,
            'mean_conf_strength': self.mean_conf_strength,
        }
        if self.freshman_k_pct is not None:
            d['freshman_k_pct'] = self.freshman_k_pct
            d['freshman_bb_pct'] = self.freshman_bb_pct
            d['freshman_iso'] = self.freshman_iso
            d['freshman_babip'] = self.freshman_babip
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'PopulationMeans':
        """Reconstruct from dictionary."""
        return cls(
            k_pct=data['k_pct'],
            bb_pct=data['bb_pct'],
            iso=data['iso'],
            babip=data['babip'],
            overall_k_pct=data['overall_k_pct'],
            overall_bb_pct=data['overall_bb_pct'],
            overall_iso=data['overall_iso'],
            overall_babip=data['overall_babip'],
            iso_variance=data['iso_variance'],
            babip_variance=data['babip_variance'],
            babip_gb=data['babip_gb'],
            babip_ld=data['babip_ld'],
            babip_fb=data['babip_fb'],
            mean_conf_strength=data['mean_conf_strength'],
            freshman_k_pct=data.get('freshman_k_pct'),
            freshman_bb_pct=data.get('freshman_bb_pct'),
            freshman_iso=data.get('freshman_iso'),
            freshman_babip=data.get('freshman_babip'),
        )


def calculate_population_means(df: pd.DataFrame) -> PopulationMeans:
    """
    Calculate population means from training data, stratified by class.

    Args:
        df: Training DataFrame with columns:
            - K%, BB%, ISO, BABIP (current season actuals)
            - Class_FR, Class_SO, Class_JR, Class_SR, Class_GR (class indicators)
            - GB%, LD%, FB% (batted ball data)
            - Conf_Strength (conference strength)
            - PA (plate appearances for weighting)

    Returns:
        PopulationMeans with class-stratified means and variances
    """
    # Map class indicators to class names
    class_mapping = {
        'Class_FR': 'FR',
        'Class_SO': 'SO',
        'Class_JR': 'JR',
        'Class_SR': 'SR',
        'Class_GR': 'GR'
    }

    # Initialize result dictionaries
    k_pct_means = {}
    bb_pct_means = {}
    iso_means = {}
    babip_means = {}

    # Calculate PA-weighted means by class
    for class_col, class_name in class_mapping.items():
        if class_col not in df.columns:
            continue

        class_df = df[df[class_col] == True]
        if len(class_df) == 0:
            continue

        # PA-weighted means
        total_pa = class_df['PA'].sum()

        k_pct_means[class_name] = (class_df['K%'] * class_df['PA']).sum() / total_pa
        bb_pct_means[class_name] = (class_df['BB%'] * class_df['PA']).sum() / total_pa
        iso_means[class_name] = (class_df['ISO'] * class_df['PA']).sum() / total_pa
        babip_means[class_name] = (class_df['BABIP'] * class_df['PA']).sum() / total_pa

    # Overall weighted means
    total_pa = df['PA'].sum()
    overall_k_pct = (df['K%'] * df['PA']).sum() / total_pa
    overall_bb_pct = (df['BB%'] * df['PA']).sum() / total_pa
    overall_iso = (df['ISO'] * df['PA']).sum() / total_pa
    overall_babip = (df['BABIP'] * df['PA']).sum() / total_pa

    # Calculate empirical variances for normal distributions
    # Weight by PA to get population variance
    iso_variance = np.average((df['ISO'] - overall_iso)**2, weights=df['PA'])
    babip_variance = np.average((df['BABIP'] - overall_babip)**2, weights=df['PA'])

    # Calculate BABIP by batted ball type
    # Need to estimate from current season data
    # Use simplified calculation based on aggregate data
    babip_gb = calculate_babip_by_type(df, 'GB%', 'BABIP')
    babip_ld = calculate_babip_by_type(df, 'LD%', 'BABIP')
    babip_fb = calculate_babip_by_type(df, 'FB%', 'BABIP')

    # Mean conference strength
    mean_conf_strength = df['Conf_Strength'].mean()

    return PopulationMeans(
        k_pct=k_pct_means,
        bb_pct=bb_pct_means,
        iso=iso_means,
        babip=babip_means,
        overall_k_pct=overall_k_pct,
        overall_bb_pct=overall_bb_pct,
        overall_iso=overall_iso,
        overall_babip=overall_babip,
        iso_variance=iso_variance,
        babip_variance=babip_variance,
        babip_gb=babip_gb,
        babip_ld=babip_ld,
        babip_fb=babip_fb,
        mean_conf_strength=mean_conf_strength
    )


def calculate_babip_by_type(df: pd.DataFrame, type_col: str, babip_col: str) -> float:
    """
    Estimate BABIP contribution from a specific batted ball type.

    Uses regression approach to estimate the marginal BABIP for each type.
    Simplified: use known typical values with adjustment from data.
    """
    # Typical MLB values as baseline (college tends to be similar)
    typical_values = {
        'GB%': 0.240,  # Ground balls rarely go for hits
        'LD%': 0.680,  # Line drives are usually hits
        'FB%': 0.120   # Fly balls rarely fall in (excluding HR)
    }

    # Use data to adjust if we have sufficient sample
    if type_col in df.columns and babip_col in df.columns:
        # Simple adjustment: correlate high type% with BABIP
        corr = df[type_col].corr(df[babip_col])

        # Scale adjustment by correlation
        if type_col == 'LD%':
            # Line drives should increase BABIP
            adjustment = corr * 0.1 if corr > 0 else 0
        else:
            # GB% and FB% typically decrease BABIP
            adjustment = corr * 0.05

        return typical_values[type_col] + adjustment

    return typical_values.get(type_col, 0.300)


def calculate_expected_babip(
    gb_pct: float,
    ld_pct: float,
    fb_pct: float,
    pop_means: PopulationMeans
) -> float:
    """
    Calculate expected BABIP from batted ball profile.

    Args:
        gb_pct: Ground ball percentage (0-1)
        ld_pct: Line drive percentage (0-1)
        fb_pct: Fly ball percentage (0-1)
        pop_means: PopulationMeans with BABIP rates by type

    Returns:
        Expected BABIP based on batted ball mix
    """
    # Normalize percentages to sum to 1 (excluding pop-ups)
    total = gb_pct + ld_pct + fb_pct
    if total > 0:
        gb_pct = gb_pct / total
        ld_pct = ld_pct / total
        fb_pct = fb_pct / total
    else:
        # Default to average profile
        gb_pct, ld_pct, fb_pct = 0.40, 0.20, 0.40

    return (
        gb_pct * pop_means.babip_gb +
        ld_pct * pop_means.babip_ld +
        fb_pct * pop_means.babip_fb
    )


def get_class_from_indicators(row: pd.Series) -> str:
    """Extract class name from indicator columns."""
    class_mapping = {
        'Class_FR': 'FR',
        'Class_SO': 'SO',
        'Class_JR': 'JR',
        'Class_SR': 'SR',
        'Class_GR': 'GR'
    }

    for col, class_name in class_mapping.items():
        if col in row and row[col] == True:
            return class_name

    return 'SO'  # Default to sophomore if unknown


def calculate_regressed_mean(
    prior_stat: float,
    prior_pa: float,
    class_mean: float,
    reliability_mult: float,
    reliability_at_reference: float = 0.6
) -> float:
    """
    Calculate regressed prior mean using weighted combination.

    Formula:
        prior_weight = sqrt(Prior_PA) × reliability_multiplier
        pop_weight = sqrt(300) × (1 - reliability_at_300)
        regressed_mean = (prior_weight × prior_stat + pop_weight × class_mean) / (prior_weight + pop_weight)

    Args:
        prior_stat: Prior year statistic value
        prior_pa: Prior year plate appearances
        class_mean: Population mean for player's class
        reliability_mult: Statistic-specific reliability multiplier
        reliability_at_reference: Reliability at REFERENCE_PA (default 0.6)

    Returns:
        Regressed mean value
    """
    prior_weight = np.sqrt(prior_pa) * reliability_mult
    pop_weight = np.sqrt(REFERENCE_PA) * (1 - reliability_at_reference)

    return (prior_weight * prior_stat + pop_weight * class_mean) / (prior_weight + pop_weight)


def apply_conference_adjustment(
    mean: float,
    conf_strength: float,
    mean_conf_strength: float,
    adjustment_factor: float = 0.5
) -> float:
    """
    Adjust prediction based on conference strength.

    Moving to a stronger conference should decrease counting stats (K%, BB%)
    but we apply minimal adjustment as the effect is complex.

    Args:
        mean: Base prediction
        conf_strength: Player's current conference strength
        mean_conf_strength: Average conference strength
        adjustment_factor: How much to weight the adjustment (default 0.5)

    Returns:
        Adjusted mean
    """
    # Note: Higher conf_strength = tougher competition
    # For K%, expect slight increase against better pitching
    # For BB%, expect slight decrease
    # For ISO/BABIP, effects are mixed

    # Using multiplicative adjustment centered at 1.0
    adjustment = 1 + adjustment_factor * (conf_strength - mean_conf_strength)

    # Keep adjustment modest (within 10%)
    adjustment = np.clip(adjustment, 0.90, 1.10)

    return mean * adjustment


def apply_park_adjustment(
    mean: float,
    park_factor: float,
    stat_type: str
) -> float:
    """
    Adjust prediction based on park factor.

    Park factors are stored as decimals:
    - > 1.0 = hitter-friendly park
    - < 1.0 = pitcher-friendly park
    - 1.0 = neutral

    Args:
        mean: Base prediction
        park_factor: Park factor (1.0 = neutral, stored as decimal)
        stat_type: Which stat ('K%', 'BB%', 'ISO', 'BABIP')

    Returns:
        Park-adjusted mean
    """
    if pd.isna(park_factor) or park_factor == 0:
        return mean

    # ISO is most affected by park
    # BABIP moderately affected
    # K% and BB% less affected (mostly hitter skill)

    park_sensitivity = {
        'K%': 0.0,     # Minimal park effect on K%
        'BB%': 0.0,    # Minimal park effect on BB%
        'ISO': 0.5,    # Moderate park effect on ISO
        'BABIP': 0.2   # Small park effect on BABIP
    }

    sensitivity = park_sensitivity.get(stat_type, 0.0)

    # Park factor is already a decimal (e.g., 1.05 for hitter friendly)
    # Neutral adjustment = 1.0 / 1.0 = 1.0
    # Hitter-friendly park (1.1): adjustment = 1.0 / 1.1 = 0.91 (lower predicted stats)
    # Pitcher-friendly park (0.9): adjustment = 1.0 / 0.9 = 1.11 (higher predicted stats)
    adjustment = 1.0 / park_factor

    # Apply partial adjustment based on sensitivity
    # adjusted = mean * (1 + sensitivity * (adjustment - 1))
    # For park_factor = 1.1, adjustment = 0.91: adjusted = mean * (1 + 0.5 * -0.09) = mean * 0.955
    adjusted = mean * (1 + sensitivity * (adjustment - 1))

    return adjusted


def calculate_prior_strength(
    prior_pa: float,
    reliability_mult: float,
    min_strength: float = 10.0
) -> float:
    """
    Calculate the prior strength (equivalent sample size) for a statistic.

    Args:
        prior_pa: Prior year plate appearances
        reliability_mult: Statistic-specific reliability multiplier
        min_strength: Minimum prior strength

    Returns:
        Prior strength (equivalent PA)
    """
    strength = prior_pa * reliability_mult
    return max(strength, min_strength)


def build_player_prior(
    row: pd.Series,
    pop_means: PopulationMeans,
    has_prior_data: bool
) -> PlayerPosteriors:
    """
    Build prior distributions for a single player.

    Args:
        row: DataFrame row with player data
        pop_means: Population means calculated from training data
        has_prior_data: Whether player has prior year statistics

    Returns:
        PlayerPosteriors object with prior distributions
    """
    player_class = get_class_from_indicators(row)

    # Get class means (fall back to overall if class not found)
    k_class_mean = pop_means.k_pct.get(player_class, pop_means.overall_k_pct)
    bb_class_mean = pop_means.bb_pct.get(player_class, pop_means.overall_bb_pct)
    iso_class_mean = pop_means.iso.get(player_class, pop_means.overall_iso)
    babip_class_mean = pop_means.babip.get(player_class, pop_means.overall_babip)

    if has_prior_data and pd.notna(row.get('Prior_PA')) and row['Prior_PA'] > 0:
        # Player has prior year data - use regression
        prior_pa = row['Prior_PA']

        # Calculate regressed means for each stat
        k_mean = calculate_regressed_mean(
            row['Prior_K%'], prior_pa, k_class_mean,
            RELIABILITY_MULTIPLIERS['K%']
        )
        bb_mean = calculate_regressed_mean(
            row['Prior_BB%'], prior_pa, bb_class_mean,
            RELIABILITY_MULTIPLIERS['BB%']
        )
        iso_mean = calculate_regressed_mean(
            row['Prior_ISO'], prior_pa, iso_class_mean,
            RELIABILITY_MULTIPLIERS['ISO']
        )

        # BABIP: blend prior actual with batted ball expected
        prior_babip = row['Prior_BABIP']

        # If we have batted ball data, calculate expected BABIP
        if all(pd.notna(row.get(col)) for col in ['Prior_GB%', 'Prior_LD%', 'Prior_FB%']):
            expected_babip = calculate_expected_babip(
                row['Prior_GB%'], row['Prior_LD%'], row['Prior_FB%'],
                pop_means
            )
            # Blend actual and expected (actual gets more weight with more PA)
            actual_weight = min(prior_pa / 300, 0.7)
            babip_prior = actual_weight * prior_babip + (1 - actual_weight) * expected_babip
        else:
            babip_prior = prior_babip

        babip_mean = calculate_regressed_mean(
            babip_prior, prior_pa, babip_class_mean,
            RELIABILITY_MULTIPLIERS['BABIP']
        )

        # Apply conference adjustment
        if pd.notna(row.get('Conf_Strength')):
            # K% increases against better competition
            k_mean = apply_conference_adjustment(
                k_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 0.3
            )
            # BB% relatively stable
            bb_mean = apply_conference_adjustment(
                bb_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 0.1
            )

        # Apply park adjustment
        if pd.notna(row.get('Park_Factor')):
            iso_mean = apply_park_adjustment(iso_mean, row['Park_Factor'], 'ISO')
            babip_mean = apply_park_adjustment(babip_mean, row['Park_Factor'], 'BABIP')

        # Calculate prior strengths
        k_strength = calculate_prior_strength(prior_pa, RELIABILITY_MULTIPLIERS['K%'])
        bb_strength = calculate_prior_strength(prior_pa, RELIABILITY_MULTIPLIERS['BB%'])
        iso_strength = calculate_prior_strength(prior_pa, RELIABILITY_MULTIPLIERS['ISO'])
        babip_strength = calculate_prior_strength(prior_pa, RELIABILITY_MULTIPLIERS['BABIP'])

    else:
        # Freshman or no prior data - use true freshman means if available,
        # otherwise fall back to class means
        if pop_means.freshman_k_pct is not None:
            k_mean = pop_means.freshman_k_pct
            bb_mean = pop_means.freshman_bb_pct
            iso_mean = pop_means.freshman_iso
            babip_mean = pop_means.freshman_babip
        else:
            k_mean = k_class_mean
            bb_mean = bb_class_mean
            iso_mean = iso_class_mean
            babip_mean = babip_class_mean

        # Apply conference adjustment for context
        if pd.notna(row.get('Conf_Strength')):
            k_mean = apply_conference_adjustment(
                k_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 0.3
            )

        # Apply park adjustment
        if pd.notna(row.get('Park_Factor')):
            iso_mean = apply_park_adjustment(iso_mean, row['Park_Factor'], 'ISO')
            babip_mean = apply_park_adjustment(babip_mean, row['Park_Factor'], 'BABIP')

        # Weak priors for freshmen
        k_strength = FRESHMAN_PRIOR_STRENGTH
        bb_strength = FRESHMAN_PRIOR_STRENGTH
        iso_strength = FRESHMAN_PRIOR_STRENGTH * 0.5  # Extra weak for ISO
        babip_strength = FRESHMAN_PRIOR_STRENGTH * 0.5  # Extra weak for BABIP

    # Ensure means are within valid bounds
    k_mean = np.clip(k_mean, 0.05, 0.50)
    bb_mean = np.clip(bb_mean, 0.02, 0.25)
    iso_mean = np.clip(iso_mean, 0.01, 0.40)
    babip_mean = np.clip(babip_mean, 0.20, 0.45)

    # Build model objects
    k_model = BetaComponentModel.from_mean_strength(k_mean, k_strength, stat_name='K%')
    bb_model = BetaComponentModel.from_mean_strength(bb_mean, bb_strength, stat_name='BB%')

    iso_model = NormalComponentModel.from_mean_strength(
        iso_mean, iso_strength, pop_means.iso_variance, stat_name='ISO'
    )
    babip_model = NormalComponentModel.from_mean_strength(
        babip_mean, babip_strength, pop_means.babip_variance, stat_name='BABIP'
    )

    return PlayerPosteriors(
        player_name=row['Player'],
        team=row['Team'],
        season=row['Season'],
        k_pct=k_model,
        bb_pct=bb_model,
        iso=iso_model,
        babip=babip_model
    )


def build_player_priors(
    df: pd.DataFrame,
    pop_means: PopulationMeans
) -> pd.DataFrame:
    """
    Build preseason priors for all players in the dataset.

    Args:
        df: DataFrame with player data
        pop_means: Population means from training data

    Returns:
        DataFrame with prior parameters for each player
    """
    results = []

    for idx, row in df.iterrows():
        has_prior = (
            pd.notna(row.get('Prior_PA')) and
            row.get('Prior_PA', 0) > 0
        )

        posteriors = build_player_prior(row, pop_means, has_prior)

        # Extract parameters for DataFrame
        result = {
            'Player': posteriors.player_name,
            'Team': posteriors.team,
            'Season': posteriors.season,
            'Has_Prior_Data': has_prior,
            # K% Beta parameters
            'K%_alpha': posteriors.k_pct.alpha,
            'K%_beta': posteriors.k_pct.beta,
            'K%_mean': posteriors.k_pct.get_mean(),
            'K%_std': posteriors.k_pct.get_std(),
            # BB% Beta parameters
            'BB%_alpha': posteriors.bb_pct.alpha,
            'BB%_beta': posteriors.bb_pct.beta,
            'BB%_mean': posteriors.bb_pct.get_mean(),
            'BB%_std': posteriors.bb_pct.get_std(),
            # ISO Normal parameters
            'ISO_mu': posteriors.iso.mu,
            'ISO_tau': posteriors.iso.tau,
            'ISO_mean': posteriors.iso.get_mean(),
            'ISO_std': posteriors.iso.get_std(),
            # BABIP Normal parameters
            'BABIP_mu': posteriors.babip.mu,
            'BABIP_tau': posteriors.babip.tau,
            'BABIP_mean': posteriors.babip.get_mean(),
            'BABIP_std': posteriors.babip.get_std(),
        }

        # Add prediction intervals
        for stat_name, model in [
            ('K%', posteriors.k_pct),
            ('BB%', posteriors.bb_pct),
            ('ISO', posteriors.iso),
            ('BABIP', posteriors.babip)
        ]:
            for alpha in [0.50, 0.80, 0.90]:
                lower, upper = model.get_prediction_interval(alpha)
                pct = int(alpha * 100)
                result[f'{stat_name}_p{(100-pct)//2}'] = lower
                result[f'{stat_name}_p{100 - (100-pct)//2}'] = upper

        results.append(result)

    return pd.DataFrame(results)


def save_population_means(pop_means: PopulationMeans, filepath: Path):
    """Save population means to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(pop_means.to_dict(), f, indent=2)


def load_population_means(filepath: Path) -> PopulationMeans:
    """Load population means from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return PopulationMeans.from_dict(data)


# ══════════════════════════════════════════════════════════════════════════════
# FanGraphs-Specific Functions (Age-Based, No Batted Ball Data)
# ══════════════════════════════════════════════════════════════════════════════


def get_age_bin(age: float) -> str:
    """
    Map continuous age to age bin for lookback and population stratification.

    Age bins:
        <=20: Youngest players (freshmen/young sophomores)
        21: Typical sophomores
        22: Typical juniors
        23+: Seniors and graduate students

    Args:
        age: Player's age

    Returns:
        Age bin string
    """
    if pd.isna(age):
        return '21'  # default to typical sophomore
    if age <= 20:
        return '<=20'
    elif age <= 21:
        return '21'
    elif age <= 22:
        return '22'
    else:
        return '23+'


def calculate_fg_population_means(df: pd.DataFrame) -> PopulationMeans:
    """
    Calculate population means from FanGraphs training data, stratified by age bin.

    This version uses Age instead of Class for stratification and does not require
    batted ball data (GB%, LD%, FB%). BABIP is projected using simple regression
    to population mean rather than batted ball blending.

    Args:
        df: Training DataFrame with columns:
            - K%, BB%, ISO, BABIP (current season actuals)
            - Age (player age)
            - PA (plate appearances for weighting)
            - Conf_Strength (optional, for adjustment)

    Returns:
        PopulationMeans with age-stratified means (stored in class dictionaries
        for compatibility with existing code structure)
    """
    # Define age bins
    age_bins = {
        '<=20': df['Age'] <= 20,
        '21': (df['Age'] > 20) & (df['Age'] <= 21),
        '22': (df['Age'] > 21) & (df['Age'] <= 22),
        '23+': df['Age'] > 22,
    }

    # Initialize result dictionaries
    # We store age bins using the same dictionary structure as class-based model
    # for compatibility with existing PopulationMeans structure
    k_pct_means = {}
    bb_pct_means = {}
    iso_means = {}
    babip_means = {}

    # Calculate PA-weighted means by age bin
    for bin_name, mask in age_bins.items():
        bin_df = df[mask]
        if len(bin_df) == 0:
            continue

        total_pa = bin_df['PA'].sum()
        if total_pa == 0:
            continue

        k_pct_means[bin_name] = (bin_df['K%'] * bin_df['PA']).sum() / total_pa
        bb_pct_means[bin_name] = (bin_df['BB%'] * bin_df['PA']).sum() / total_pa
        iso_means[bin_name] = (bin_df['ISO'] * bin_df['PA']).sum() / total_pa
        babip_means[bin_name] = (bin_df['BABIP'] * bin_df['PA']).sum() / total_pa

    # Overall weighted means
    total_pa = df['PA'].sum()
    overall_k_pct = (df['K%'] * df['PA']).sum() / total_pa
    overall_bb_pct = (df['BB%'] * df['PA']).sum() / total_pa
    overall_iso = (df['ISO'] * df['PA']).sum() / total_pa
    overall_babip = (df['BABIP'] * df['PA']).sum() / total_pa

    # Calculate empirical variances for normal distributions
    iso_variance = np.average((df['ISO'] - overall_iso)**2, weights=df['PA'])
    babip_variance = np.average((df['BABIP'] - overall_babip)**2, weights=df['PA'])

    # BABIP by batted ball type: use defaults since FG lacks batted ball data
    # These values are typical for college baseball
    babip_gb = 0.240  # Ground balls
    babip_ld = 0.680  # Line drives
    babip_fb = 0.120  # Fly balls (excluding HR)

    # Mean conference strength
    mean_conf_strength = df['Conf_Strength'].mean() if 'Conf_Strength' in df.columns else 0.5

    return PopulationMeans(
        k_pct=k_pct_means,
        bb_pct=bb_pct_means,
        iso=iso_means,
        babip=babip_means,
        overall_k_pct=overall_k_pct,
        overall_bb_pct=overall_bb_pct,
        overall_iso=overall_iso,
        overall_babip=overall_babip,
        iso_variance=iso_variance,
        babip_variance=babip_variance,
        babip_gb=babip_gb,
        babip_ld=babip_ld,
        babip_fb=babip_fb,
        mean_conf_strength=mean_conf_strength
    )


def build_fg_player_prior(
    row: pd.Series,
    pop_means: PopulationMeans,
    has_prior_data: bool
) -> PlayerPosteriors:
    """
    Build prior distributions for a single player using FanGraphs age-based approach.

    Key differences from D1B version:
    - Uses Age instead of Class for population mean selection
    - No batted ball blending for BABIP (simple regression only)
    - Otherwise follows same regression and adjustment logic

    Args:
        row: DataFrame row with player data including:
            - Age: player age
            - Prior_PA, Prior_K%, Prior_BB%, Prior_ISO, Prior_BABIP (if available)
            - Conf_Strength, Park_Factor (for adjustments)
        pop_means: Population means calculated from FG training data
        has_prior_data: Whether player has prior year statistics

    Returns:
        PlayerPosteriors object with prior distributions
    """
    # Get age bin and corresponding population means
    age_bin = get_age_bin(row.get('Age'))

    # Get age-specific means (fall back to overall if bin not found)
    k_age_mean = pop_means.k_pct.get(age_bin, pop_means.overall_k_pct)
    bb_age_mean = pop_means.bb_pct.get(age_bin, pop_means.overall_bb_pct)
    iso_age_mean = pop_means.iso.get(age_bin, pop_means.overall_iso)
    babip_age_mean = pop_means.babip.get(age_bin, pop_means.overall_babip)

    if has_prior_data and pd.notna(row.get('Prior_PA')) and row['Prior_PA'] > 0:
        # Player has prior year data - use regression
        prior_pa = row['Prior_PA']

        # Calculate regressed means for each stat
        k_mean = calculate_regressed_mean(
            row['Prior_K%'], prior_pa, k_age_mean,
            RELIABILITY_MULTIPLIERS['K%']
        )
        bb_mean = calculate_regressed_mean(
            row['Prior_BB%'], prior_pa, bb_age_mean,
            RELIABILITY_MULTIPLIERS['BB%']
        )
        iso_mean = calculate_regressed_mean(
            row['Prior_ISO'], prior_pa, iso_age_mean,
            RELIABILITY_MULTIPLIERS['ISO']
        )

        # BABIP: simple regression (NO batted ball blending for FG)
        babip_mean = calculate_regressed_mean(
            row['Prior_BABIP'], prior_pa, babip_age_mean,
            RELIABILITY_MULTIPLIERS['BABIP']
        )

        # Apply conference adjustment
        if pd.notna(row.get('Conf_Strength')):
            # K% increases against better competition
            k_mean = apply_conference_adjustment(
                k_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 0.3
            )
            # BB% relatively stable
            bb_mean = apply_conference_adjustment(
                bb_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 0.1
            )
            # ISO conference adjustment
            iso_mean = apply_conference_adjustment(
                iso_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 0.2
            )

        # Apply park adjustment
        if pd.notna(row.get('Park_Factor')):
            iso_mean = apply_park_adjustment(iso_mean, row['Park_Factor'], 'ISO')
            babip_mean = apply_park_adjustment(babip_mean, row['Park_Factor'], 'BABIP')

        # Calculate prior strengths
        k_strength = calculate_prior_strength(prior_pa, RELIABILITY_MULTIPLIERS['K%'])
        bb_strength = calculate_prior_strength(prior_pa, RELIABILITY_MULTIPLIERS['BB%'])
        iso_strength = calculate_prior_strength(prior_pa, RELIABILITY_MULTIPLIERS['ISO'])
        babip_strength = calculate_prior_strength(prior_pa, RELIABILITY_MULTIPLIERS['BABIP'])

    else:
        # Young player or no prior data - use freshman means if available,
        # otherwise fall back to age bin means
        if pop_means.freshman_k_pct is not None:
            k_mean = pop_means.freshman_k_pct
            bb_mean = pop_means.freshman_bb_pct
            iso_mean = pop_means.freshman_iso
            babip_mean = pop_means.freshman_babip
        else:
            k_mean = k_age_mean
            bb_mean = bb_age_mean
            iso_mean = iso_age_mean
            babip_mean = babip_age_mean

        # Apply context adjustments
        if pd.notna(row.get('Conf_Strength')):
            k_mean = apply_conference_adjustment(
                k_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 0.3
            )
            bb_mean = apply_conference_adjustment(
                bb_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 0.1
            )
            # ISO conference adjustment
            iso_mean = apply_conference_adjustment(
                iso_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 0.2
            )

        if pd.notna(row.get('Park_Factor')):
            iso_mean = apply_park_adjustment(iso_mean, row['Park_Factor'], 'ISO')
            babip_mean = apply_park_adjustment(babip_mean, row['Park_Factor'], 'BABIP')

        # Weak priors for new players
        k_strength = FRESHMAN_PRIOR_STRENGTH
        bb_strength = FRESHMAN_PRIOR_STRENGTH
        iso_strength = FRESHMAN_PRIOR_STRENGTH
        babip_strength = FRESHMAN_PRIOR_STRENGTH

    # Build models
    k_model = BetaComponentModel.from_mean_strength(k_mean, k_strength, 'K%')
    bb_model = BetaComponentModel.from_mean_strength(bb_mean, bb_strength, 'BB%')
    iso_model = NormalComponentModel.from_mean_strength(
        iso_mean, iso_strength, pop_means.iso_variance, 'ISO'
    )
    babip_model = NormalComponentModel.from_mean_strength(
        babip_mean, babip_strength, pop_means.babip_variance, 'BABIP'
    )

    return PlayerPosteriors(
        player_name=row.get('Player', 'Unknown'),
        team=row.get('Team', 'Unknown'),
        season=row.get('Season', 2025),
        k_pct=k_model,
        bb_pct=bb_model,
        iso=iso_model,
        babip=babip_model
    )


def build_fg_player_priors(
    df: pd.DataFrame,
    pop_means: PopulationMeans
) -> pd.DataFrame:
    """
    Build prior distributions for all players in FanGraphs training set.

    Wrapper function that applies build_fg_player_prior() to each player
    and returns a DataFrame with projection statistics.

    Args:
        df: Training DataFrame with player data
        pop_means: Population means from FG training data

    Returns:
        DataFrame with prior distributions and projections for each player
    """
    results = []

    for idx, row in df.iterrows():
        has_prior = pd.notna(row.get('Prior_PA')) and row['Prior_PA'] > 0

        try:
            posteriors = build_fg_player_prior(row, pop_means, has_prior)

            result = {
                'Player': posteriors.player_name,
                'Team': posteriors.team,
                'Season': posteriors.season,
                'Age': row.get('Age'),
                'Age_Bin': get_age_bin(row.get('Age')),
                'Has_Prior_Data': has_prior,
                # K% Beta parameters
                'K%_alpha': posteriors.k_pct.alpha,
                'K%_beta': posteriors.k_pct.beta,
                'K%_mean': posteriors.k_pct.get_mean(),
                'K%_std': posteriors.k_pct.get_std(),
                # BB% Beta parameters
                'BB%_alpha': posteriors.bb_pct.alpha,
                'BB%_beta': posteriors.bb_pct.beta,
                'BB%_mean': posteriors.bb_pct.get_mean(),
                'BB%_std': posteriors.bb_pct.get_std(),
                # ISO Normal parameters
                'ISO_mu': posteriors.iso.mu,
                'ISO_tau': posteriors.iso.tau,
                'ISO_mean': posteriors.iso.get_mean(),
                'ISO_std': posteriors.iso.get_std(),
                # BABIP Normal parameters
                'BABIP_mu': posteriors.babip.mu,
                'BABIP_tau': posteriors.babip.tau,
                'BABIP_mean': posteriors.babip.get_mean(),
                'BABIP_std': posteriors.babip.get_std(),
            }

            # Add prediction intervals
            for stat_name, model in [
                ('K%', posteriors.k_pct),
                ('BB%', posteriors.bb_pct),
                ('ISO', posteriors.iso),
                ('BABIP', posteriors.babip)
            ]:
                for alpha in [0.50, 0.80, 0.90]:
                    lower, upper = model.get_prediction_interval(alpha)
                    pct = int(alpha * 100)
                    result[f'{stat_name}_p{(100-pct)//2}'] = lower
                    result[f'{stat_name}_p{100 - (100-pct)//2}'] = upper

            results.append(result)

        except Exception as e:
            print(f"Warning: Failed to build prior for {row.get('Player', 'Unknown')}: {e}")
            continue

    return pd.DataFrame(results)
