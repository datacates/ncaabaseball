"""
Prior Construction for Bayesian Pitching Model

This module handles construction of preseason priors for each pitcher based on:
- Prior year performance (regressed to population mean)
- Class-specific population means (IP-weighted)
- Conference strength adjustment (inverted from batting for K%)
- Park factor adjustment (HR/FB% heavily park-dependent)
- Batted ball profile for BABIP estimation
- FIP constant calibration
"""

import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from bayesian_batting.model import BetaComponentModel, NormalComponentModel
from .model import PitcherPosteriors


# ---------------------------------------------------------------------------
# Hyperparameters (starting values; tune via cross-validation)
# ---------------------------------------------------------------------------

# Reliability multipliers control REGRESSION toward the mean (point estimate).
# Higher = more weight on prior performance, less regression.
RELIABILITY_MULTIPLIERS = {
    'K%': 0.85,       # Highly persistent for pitchers
    'BB%': 0.65,      # Moderately persistent
    'HR/FB%': 0.12,   # Mostly noise for pitchers — heavy regression
    'BABIP': 0.03,    # Extremely noisy for pitchers — near-total regression
}

# Strength multipliers control INTERVAL WIDTH (prior confidence).
# Decoupled from regression because a stat can be predictable on average
# (good R²) yet have wide per-pitcher uncertainty.
# These are calibrated so 90% prediction intervals capture ~90% of outcomes.
STRENGTH_MULTIPLIERS = {
    'K%': 0.22,       # Calibrated for ~90% coverage at 90% level
    'BB%': 0.28,      # Calibrated for ~90% coverage at 90% level
    'HR/FB%': 0.0015, # Very weak — HR/FB% is extremely noisy per season
    'BABIP': 0.002,   # Very weak — pitcher BABIP is nearly random per season
}

# Reference IP for population weight (typical full-season starter)
REFERENCE_IP = 80

# Approximate BF per IP for converting IP to BF
BF_PER_IP = 2.78

# Reference BF derived from REFERENCE_IP
REFERENCE_BF = REFERENCE_IP * BF_PER_IP

# Minimum prior strength for freshmen / no-data pitchers
FRESHMAN_PRIOR_STRENGTH = 20

# Park factor sensitivity by stat
PARK_SENSITIVITY = {
    'K%': 0.0,        # No park effect on K%
    'BB%': 0.0,       # No park effect on BB%
    'HR/FB%': 0.7,    # Highly park-dependent
    'BABIP': 0.3,     # Moderate park effect
}

# Conference adjustment direction factors for pitchers
# Positive = stat INCREASES in tougher conference
CONF_DIRECTION = {
    'K%': -0.3,       # K% DECREASES in tougher conferences (better hitters)
    'BB%': 0.1,       # BB% slightly increases (pitcher nibbles more)
    'HR/FB%': 0.2,    # HR/FB% increases (better hitters)
    'BABIP': 0.15,    # BABIP slightly increases (better contact)
}


# ---------------------------------------------------------------------------
# Population means dataclass
# ---------------------------------------------------------------------------

@dataclass
class PitchingPopulationMeans:
    """Class-specific population means for pitching statistics."""
    k_pct: Dict[str, float]
    bb_pct: Dict[str, float]
    hr_fb_pct: Dict[str, float]
    babip: Dict[str, float]

    overall_k_pct: float
    overall_bb_pct: float
    overall_hr_fb_pct: float
    overall_babip: float

    # Empirical variances for Normal distributions
    hr_fb_pct_variance: float
    babip_variance: float

    # BABIP by batted ball type (pitcher-allowed)
    babip_gb: float
    babip_ld: float
    babip_fb: float

    # Mean conference strength for centering adjustments
    mean_conf_strength: float

    # FIP constant (calibrated to make league FIP ≈ league ERA)
    fip_constant: float

    # League-average rates for FIP conversion
    avg_fb_pct: float = 0.35
    avg_ip_per_bf: float = 0.36
    avg_hbp_rate: float = 0.025

    # True freshman means (from all freshman data, not just training set)
    freshman_k_pct: Optional[float] = None
    freshman_bb_pct: Optional[float] = None
    freshman_hr_fb_pct: Optional[float] = None
    freshman_babip: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            'k_pct': self.k_pct,
            'bb_pct': self.bb_pct,
            'hr_fb_pct': self.hr_fb_pct,
            'babip': self.babip,
            'overall_k_pct': self.overall_k_pct,
            'overall_bb_pct': self.overall_bb_pct,
            'overall_hr_fb_pct': self.overall_hr_fb_pct,
            'overall_babip': self.overall_babip,
            'hr_fb_pct_variance': self.hr_fb_pct_variance,
            'babip_variance': self.babip_variance,
            'babip_gb': self.babip_gb,
            'babip_ld': self.babip_ld,
            'babip_fb': self.babip_fb,
            'mean_conf_strength': self.mean_conf_strength,
            'fip_constant': self.fip_constant,
            'avg_fb_pct': self.avg_fb_pct,
            'avg_ip_per_bf': self.avg_ip_per_bf,
            'avg_hbp_rate': self.avg_hbp_rate,
        }
        if self.freshman_k_pct is not None:
            d['freshman_k_pct'] = self.freshman_k_pct
            d['freshman_bb_pct'] = self.freshman_bb_pct
            d['freshman_hr_fb_pct'] = self.freshman_hr_fb_pct
            d['freshman_babip'] = self.freshman_babip
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'PitchingPopulationMeans':
        return cls(
            k_pct=data['k_pct'],
            bb_pct=data['bb_pct'],
            hr_fb_pct=data['hr_fb_pct'],
            babip=data['babip'],
            overall_k_pct=data['overall_k_pct'],
            overall_bb_pct=data['overall_bb_pct'],
            overall_hr_fb_pct=data['overall_hr_fb_pct'],
            overall_babip=data['overall_babip'],
            hr_fb_pct_variance=data['hr_fb_pct_variance'],
            babip_variance=data['babip_variance'],
            babip_gb=data['babip_gb'],
            babip_ld=data['babip_ld'],
            babip_fb=data['babip_fb'],
            mean_conf_strength=data['mean_conf_strength'],
            fip_constant=data['fip_constant'],
            avg_fb_pct=data.get('avg_fb_pct', 0.35),
            avg_ip_per_bf=data.get('avg_ip_per_bf', 0.36),
            avg_hbp_rate=data.get('avg_hbp_rate', 0.025),
            freshman_k_pct=data.get('freshman_k_pct'),
            freshman_bb_pct=data.get('freshman_bb_pct'),
            freshman_hr_fb_pct=data.get('freshman_hr_fb_pct'),
            freshman_babip=data.get('freshman_babip'),
        )


# ---------------------------------------------------------------------------
# Population means calculation
# ---------------------------------------------------------------------------

def calculate_fip_constant(df: pd.DataFrame) -> float:
    """
    Calculate FIP constant to make league-average FIP ≈ league-average ERA.

    FIP_const = league_ERA - ((13*HR + 3*(BB+HBP) - 2*K) / IP)
    """
    total_ip = df['IP'].sum()
    if total_ip == 0:
        return 3.10

    # Use ER column if available; otherwise estimate from ERA
    if 'ER' in df.columns:
        total_er = df['ER'].sum()
        league_era = (total_er / total_ip) * 9
    elif 'ERA' in df.columns:
        league_era = np.average(df['ERA'], weights=df['IP'])
    else:
        league_era = 4.50

    total_hr = df['HR'].sum() if 'HR' in df.columns else 0
    total_bb = df['BB'].sum()
    total_hbp = df['HBP'].sum() if 'HBP' in df.columns else 0
    total_k = df['K'].sum()

    fip_raw = (13 * total_hr + 3 * (total_bb + total_hbp) - 2 * total_k) / total_ip
    return league_era - fip_raw


def calculate_babip_by_type(df: pd.DataFrame, type_col: str) -> float:
    """Estimate BABIP for a specific batted ball type."""
    typical_values = {
        'GB%': 0.240,
        'LD%': 0.680,
        'FB%': 0.120,
    }

    if type_col in df.columns and 'BABIP' in df.columns:
        corr = df[type_col].corr(df['BABIP'])
        if type_col == 'LD%':
            adjustment = corr * 0.1 if corr > 0 else 0
        else:
            adjustment = corr * 0.05
        return typical_values[type_col] + adjustment

    return typical_values.get(type_col, 0.300)


def calculate_expected_babip(
    gb_pct: float,
    ld_pct: float,
    fb_pct: float,
    pop_means: PitchingPopulationMeans,
) -> float:
    """Calculate expected BABIP from batted ball profile."""
    total = gb_pct + ld_pct + fb_pct
    if total > 0:
        gb_pct = gb_pct / total
        ld_pct = ld_pct / total
        fb_pct = fb_pct / total
    else:
        gb_pct, ld_pct, fb_pct = 0.40, 0.20, 0.40

    return (
        gb_pct * pop_means.babip_gb +
        ld_pct * pop_means.babip_ld +
        fb_pct * pop_means.babip_fb
    )


def _compute_likelihood_variance(
    df: pd.DataFrame,
    current_col: str,
    prior_col: str,
    overall_mean: float,
) -> float:
    """
    Compute likelihood variance from year-to-year residuals.

    The Normal-Normal conjugate model needs the per-season observation noise,
    NOT the population variance.  We estimate this as the variance of the
    residual (current_stat − prior_stat) for pitchers who have both years.

    If insufficient data, fall back to an inflated population variance
    (5× the cross-sectional variance, a conservative floor).
    """
    both = df[[current_col, prior_col]].dropna()
    if len(both) >= 50:
        residuals = both[current_col] - both[prior_col]
        # Year-to-year residual variance = 2 × observation_noise_variance
        # (because both years are noisy), so divide by 2.
        yoy_var = residuals.var()
        return max(yoy_var / 2.0, 0.001)

    # Fallback: inflate the population cross-sectional variance
    pop_var = np.average((df[current_col] - overall_mean) ** 2, weights=df['IP'])
    return pop_var * 5.0


def calculate_population_means(df: pd.DataFrame) -> PitchingPopulationMeans:
    """
    Calculate population means from pitching training data, stratified by class.

    Uses IP-weighting for all rate statistics.

    Args:
        df: Training DataFrame with K%, BB%, HR/FB%, BABIP, IP,
            Class_FR..Class_GR, GB%, LD%, FB%, Conf_Strength, ER/ERA
    """
    class_mapping = {
        'Class_FR': 'FR',
        'Class_SO': 'SO',
        'Class_JR': 'JR',
        'Class_SR': 'SR',
        'Class_GR': 'GR',
    }

    k_pct_means = {}
    bb_pct_means = {}
    hr_fb_pct_means = {}
    babip_means = {}

    for class_col, class_name in class_mapping.items():
        if class_col not in df.columns:
            continue
        class_df = df[df[class_col] == True]
        if len(class_df) == 0:
            continue

        total_ip = class_df['IP'].sum()
        if total_ip == 0:
            continue

        k_pct_means[class_name] = (class_df['K%'] * class_df['IP']).sum() / total_ip
        bb_pct_means[class_name] = (class_df['BB%'] * class_df['IP']).sum() / total_ip
        hr_fb_pct_means[class_name] = (class_df['HR/FB%'] * class_df['IP']).sum() / total_ip
        babip_means[class_name] = (class_df['BABIP'] * class_df['IP']).sum() / total_ip

    # Overall IP-weighted means
    total_ip = df['IP'].sum()
    overall_k_pct = (df['K%'] * df['IP']).sum() / total_ip
    overall_bb_pct = (df['BB%'] * df['IP']).sum() / total_ip
    overall_hr_fb_pct = (df['HR/FB%'] * df['IP']).sum() / total_ip
    overall_babip = (df['BABIP'] * df['IP']).sum() / total_ip

    # Likelihood variances: use year-to-year residual variance
    # This captures the true observation noise per season, which is what the
    # Normal-Normal conjugate model needs.  Population variance drastically
    # underestimates observation noise for noisy pitcher stats.
    hr_fb_pct_variance = _compute_likelihood_variance(
        df, 'HR/FB%', 'Prior_HR/FB%', overall_hr_fb_pct
    )
    babip_variance = _compute_likelihood_variance(
        df, 'BABIP', 'Prior_BABIP', overall_babip
    )

    # BABIP by batted ball type
    babip_gb = calculate_babip_by_type(df, 'GB%')
    babip_ld = calculate_babip_by_type(df, 'LD%')
    babip_fb = calculate_babip_by_type(df, 'FB%')

    # Mean conference strength
    mean_conf_strength = df['Conf_Strength'].mean() if 'Conf_Strength' in df.columns else 0.5

    # FIP constant
    fip_constant = calculate_fip_constant(df)

    # League-average rates for FIP conversion
    avg_fb_pct = (df['FB%'] * df['IP']).sum() / total_ip if 'FB%' in df.columns else 0.35
    total_bf = df['BF'].sum() if 'BF' in df.columns else total_ip * BF_PER_IP
    avg_ip_per_bf = total_ip / total_bf if total_bf > 0 else 0.36
    avg_hbp_rate = (
        df['HBP'].sum() / total_bf if 'HBP' in df.columns and total_bf > 0 else 0.025
    )

    return PitchingPopulationMeans(
        k_pct=k_pct_means,
        bb_pct=bb_pct_means,
        hr_fb_pct=hr_fb_pct_means,
        babip=babip_means,
        overall_k_pct=overall_k_pct,
        overall_bb_pct=overall_bb_pct,
        overall_hr_fb_pct=overall_hr_fb_pct,
        overall_babip=overall_babip,
        hr_fb_pct_variance=hr_fb_pct_variance,
        babip_variance=babip_variance,
        babip_gb=babip_gb,
        babip_ld=babip_ld,
        babip_fb=babip_fb,
        mean_conf_strength=mean_conf_strength,
        fip_constant=fip_constant,
        avg_fb_pct=avg_fb_pct,
        avg_ip_per_bf=avg_ip_per_bf,
        avg_hbp_rate=avg_hbp_rate,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_class_from_indicators(row: pd.Series) -> str:
    """Extract class name from indicator columns."""
    class_mapping = {
        'Class_FR': 'FR',
        'Class_SO': 'SO',
        'Class_JR': 'JR',
        'Class_SR': 'SR',
        'Class_GR': 'GR',
    }
    for col, class_name in class_mapping.items():
        if col in row and row[col] == True:
            return class_name
    return 'SO'


def calculate_regressed_mean(
    prior_stat: float,
    prior_bf: float,
    class_mean: float,
    reliability_mult: float,
    reliability_at_reference: float = 0.6,
) -> float:
    """
    Calculate regressed prior mean using weighted combination.

    Uses BF (batters faced) as the sample-size basis, since K% and BB%
    are per-BF rates.  For HR/FB% and BABIP the caller should pass the
    appropriate count (BIP or FB_count) but the formula is the same.

    Formula:
        prior_weight  = sqrt(prior_bf) * reliability_mult
        pop_weight    = sqrt(REFERENCE_BF) * (1 - reliability_at_reference)
        regressed_mean = weighted average of prior_stat and class_mean
    """
    prior_weight = np.sqrt(prior_bf) * reliability_mult
    pop_weight = np.sqrt(REFERENCE_BF) * (1 - reliability_at_reference)

    return (prior_weight * prior_stat + pop_weight * class_mean) / (prior_weight + pop_weight)


def apply_pitching_conference_adjustment(
    mean: float,
    conf_strength: float,
    mean_conf_strength: float,
    stat_type: str,
) -> float:
    """
    Adjust prediction based on conference strength for pitchers.

    Tougher conference (higher conf_strength) effects:
    - K% DECREASES (better hitters harder to strike out)
    - BB% slightly INCREASES (pitcher nibbles more)
    - HR/FB% INCREASES (better hitters hit more HR)
    - BABIP slightly INCREASES (better contact quality)
    """
    factor = CONF_DIRECTION.get(stat_type, 0.0)
    adjustment = 1 + factor * (conf_strength - mean_conf_strength)
    adjustment = np.clip(adjustment, 0.90, 1.10)
    return mean * adjustment


def apply_pitching_park_adjustment(
    mean: float,
    park_factor: float,
    stat_type: str,
) -> float:
    """
    Adjust prediction based on park factor for pitchers.

    Same logic as batting: strip prior park effect to estimate true talent.
    - Hitter-friendly park (>1.0) → higher HR/FB%, BABIP allowed
    - Pitcher-friendly park (<1.0) → lower HR/FB%, BABIP allowed

    We invert the park factor to remove its effect from the prior observation,
    giving us a park-neutral true-talent estimate.
    """
    if pd.isna(park_factor) or park_factor == 0:
        return mean

    sensitivity = PARK_SENSITIVITY.get(stat_type, 0.0)
    adjustment = 1.0 / park_factor
    adjusted = mean * (1 + sensitivity * (adjustment - 1))
    return adjusted


def calculate_prior_strength(
    prior_bf: float,
    reliability_mult: float,
    min_strength: float = 10.0,
) -> float:
    """Calculate prior strength (equivalent sample size) from BF."""
    strength = prior_bf * reliability_mult
    return max(strength, min_strength)


# ---------------------------------------------------------------------------
# Build priors
# ---------------------------------------------------------------------------

def build_pitcher_prior(
    row: pd.Series,
    pop_means: PitchingPopulationMeans,
    has_prior_data: bool,
) -> PitcherPosteriors:
    """
    Build prior distributions for a single pitcher.

    Args:
        row: DataFrame row with pitcher data (Prior_IP, Prior_K%, etc.)
        pop_means: Population means calculated from training data
        has_prior_data: Whether pitcher has prior year statistics
    """
    player_class = get_class_from_indicators(row)

    # Class means (fall back to overall)
    k_class_mean = pop_means.k_pct.get(player_class, pop_means.overall_k_pct)
    bb_class_mean = pop_means.bb_pct.get(player_class, pop_means.overall_bb_pct)
    hr_fb_class_mean = pop_means.hr_fb_pct.get(player_class, pop_means.overall_hr_fb_pct)
    babip_class_mean = pop_means.babip.get(player_class, pop_means.overall_babip)

    if has_prior_data and pd.notna(row.get('Prior_IP')) and row['Prior_IP'] > 0:
        prior_ip = row['Prior_IP']

        # Estimate prior BF from Prior_BF column or from IP
        if pd.notna(row.get('Prior_BF')) and row['Prior_BF'] > 0:
            prior_bf = row['Prior_BF']
        else:
            prior_bf = prior_ip * BF_PER_IP

        # Calculate regressed means
        k_mean = calculate_regressed_mean(
            row['Prior_K%'], prior_bf, k_class_mean,
            RELIABILITY_MULTIPLIERS['K%'],
        )
        bb_mean = calculate_regressed_mean(
            row['Prior_BB%'], prior_bf, bb_class_mean,
            RELIABILITY_MULTIPLIERS['BB%'],
        )
        hr_fb_mean = calculate_regressed_mean(
            row['Prior_HR/FB%'], prior_bf, hr_fb_class_mean,
            RELIABILITY_MULTIPLIERS['HR/FB%'],
        )

        # BABIP: blend actual with batted-ball expected
        prior_babip = row['Prior_BABIP']
        if all(pd.notna(row.get(col)) for col in ['Prior_GB%', 'Prior_LD%', 'Prior_FB%']):
            expected_babip = calculate_expected_babip(
                row['Prior_GB%'], row['Prior_LD%'], row['Prior_FB%'],
                pop_means,
            )
            # Pitcher BABIP is noisier, so lean more on expected
            actual_weight = min(prior_ip / 150, 0.5)
            babip_prior = actual_weight * prior_babip + (1 - actual_weight) * expected_babip
        else:
            babip_prior = prior_babip

        babip_mean = calculate_regressed_mean(
            babip_prior, prior_bf, babip_class_mean,
            RELIABILITY_MULTIPLIERS['BABIP'],
        )

        # Conference adjustment
        if pd.notna(row.get('Conf_Strength')):
            for stat_type, stat_var in [('K%', 'k_mean'), ('BB%', 'bb_mean'),
                                         ('HR/FB%', 'hr_fb_mean'), ('BABIP', 'babip_mean')]:
                val = locals()[stat_var]
                val = apply_pitching_conference_adjustment(
                    val, row['Conf_Strength'], pop_means.mean_conf_strength, stat_type,
                )
                # Write back
                if stat_var == 'k_mean':
                    k_mean = val
                elif stat_var == 'bb_mean':
                    bb_mean = val
                elif stat_var == 'hr_fb_mean':
                    hr_fb_mean = val
                elif stat_var == 'babip_mean':
                    babip_mean = val

        # Park adjustment (strip prior park, apply new park)
        if pd.notna(row.get('Prior_Park_Factor')):
            hr_fb_mean = apply_pitching_park_adjustment(
                hr_fb_mean, row['Prior_Park_Factor'], 'HR/FB%'
            )
            babip_mean = apply_pitching_park_adjustment(
                babip_mean, row['Prior_Park_Factor'], 'BABIP'
            )
        if pd.notna(row.get('Park_Factor')) and row.get('Park_Factor', 1.0) != 1.0:
            # Apply new park effect (multiply, not invert)
            pf = row['Park_Factor']
            hr_fb_mean = hr_fb_mean * (1 + PARK_SENSITIVITY['HR/FB%'] * (pf - 1))
            babip_mean = babip_mean * (1 + PARK_SENSITIVITY['BABIP'] * (pf - 1))

        # Prior strengths — use STRENGTH_MULTIPLIERS (not RELIABILITY_MULTIPLIERS)
        # so that intervals are properly calibrated.
        # Normal model stats need a much lower min_strength because
        # tau = strength / variance, and with variance ~0.001 even strength=10
        # gives tau=10000 (posterior std=0.01), which is far too tight.
        k_strength = calculate_prior_strength(prior_bf, STRENGTH_MULTIPLIERS['K%'])
        bb_strength = calculate_prior_strength(prior_bf, STRENGTH_MULTIPLIERS['BB%'])
        hr_fb_strength = calculate_prior_strength(prior_bf, STRENGTH_MULTIPLIERS['HR/FB%'], min_strength=0.1)
        babip_strength = calculate_prior_strength(prior_bf, STRENGTH_MULTIPLIERS['BABIP'], min_strength=0.1)

    else:
        # Freshman / no prior data
        if pop_means.freshman_k_pct is not None:
            k_mean = pop_means.freshman_k_pct
            bb_mean = pop_means.freshman_bb_pct
            hr_fb_mean = pop_means.freshman_hr_fb_pct
            babip_mean = pop_means.freshman_babip
        else:
            k_mean = k_class_mean
            bb_mean = bb_class_mean
            hr_fb_mean = hr_fb_class_mean
            babip_mean = babip_class_mean

        # Conference adjustment for context
        if pd.notna(row.get('Conf_Strength')):
            k_mean = apply_pitching_conference_adjustment(
                k_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 'K%'
            )

        # Park adjustment
        if pd.notna(row.get('Park_Factor')) and row.get('Park_Factor', 1.0) != 1.0:
            pf = row['Park_Factor']
            hr_fb_mean = hr_fb_mean * (1 + PARK_SENSITIVITY['HR/FB%'] * (pf - 1))
            babip_mean = babip_mean * (1 + PARK_SENSITIVITY['BABIP'] * (pf - 1))

        # Weak priors — Normal model stats need much lower strength
        # because tau = strength / variance (~0.001), so strength=10
        # would give tau=10000 and absurdly tight intervals
        k_strength = FRESHMAN_PRIOR_STRENGTH
        bb_strength = FRESHMAN_PRIOR_STRENGTH
        hr_fb_strength = 0.5  # tau ≈ 500 → std ≈ 0.045
        babip_strength = 0.5  # tau ≈ 430 → std ≈ 0.048

    # Clip to valid bounds (tighter than batting)
    k_mean = np.clip(k_mean, 0.10, 0.45)
    bb_mean = np.clip(bb_mean, 0.03, 0.20)
    hr_fb_mean = np.clip(hr_fb_mean, 0.02, 0.25)
    babip_mean = np.clip(babip_mean, 0.250, 0.350)

    # Build model objects
    k_model = BetaComponentModel.from_mean_strength(k_mean, k_strength, stat_name='K%')
    bb_model = BetaComponentModel.from_mean_strength(bb_mean, bb_strength, stat_name='BB%')

    hr_fb_model = NormalComponentModel.from_mean_strength(
        hr_fb_mean, hr_fb_strength, pop_means.hr_fb_pct_variance, stat_name='HR/FB%'
    )
    babip_model = NormalComponentModel.from_mean_strength(
        babip_mean, babip_strength, pop_means.babip_variance, stat_name='BABIP'
    )

    return PitcherPosteriors(
        player_name=row['Player'],
        team=row['Team'],
        season=row['Season'],
        k_pct=k_model,
        bb_pct=bb_model,
        hr_fb_pct=hr_fb_model,
        babip=babip_model,
    )


def build_pitcher_priors(
    df: pd.DataFrame,
    pop_means: PitchingPopulationMeans,
) -> pd.DataFrame:
    """
    Build preseason priors for all pitchers in the dataset.

    Returns DataFrame with prior parameters for each pitcher.
    """
    results = []

    for idx, row in df.iterrows():
        has_prior = (
            pd.notna(row.get('Prior_IP')) and
            row.get('Prior_IP', 0) > 0
        )

        posteriors = build_pitcher_prior(row, pop_means, has_prior)

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
            # HR/FB% Normal parameters
            'HR/FB%_mu': posteriors.hr_fb_pct.mu,
            'HR/FB%_tau': posteriors.hr_fb_pct.tau,
            'HR/FB%_mean': posteriors.hr_fb_pct.get_mean(),
            'HR/FB%_std': posteriors.hr_fb_pct.get_std(),
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
            ('HR/FB%', posteriors.hr_fb_pct),
            ('BABIP', posteriors.babip),
        ]:
            for alpha in [0.50, 0.80, 0.90]:
                lower, upper = model.get_prediction_interval(alpha)
                pct = int(alpha * 100)
                result[f'{stat_name}_p{(100 - pct) // 2}'] = lower
                result[f'{stat_name}_p{100 - (100 - pct) // 2}'] = upper

        results.append(result)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# FanGraphs-specific functions (Age-based, no batted ball data)
# ---------------------------------------------------------------------------

AGE_LOOKBACK = {
    '<=20': 1,
    '21': 2,
    '22': 3,
    '23+': 4,
}


def get_age_bin(age: float) -> str:
    """Map continuous age to age bin key."""
    if age <= 20:
        return '<=20'
    elif age <= 21:
        return '21'
    elif age <= 22:
        return '22'
    else:
        return '23+'


def calculate_fg_population_means(df: pd.DataFrame) -> PitchingPopulationMeans:
    """
    Calculate population means from FanGraphs training data, stratified by age bin.

    HR/FB% is estimated from counts since FanGraphs lacks batted ball splits.
    BABIP by-type fields are set to defaults (unused without batted ball data).
    """
    age_bins = {
        '<=20': df['Age'] <= 20,
        '21': (df['Age'] > 20) & (df['Age'] <= 21),
        '22': (df['Age'] > 21) & (df['Age'] <= 22),
        '23+': df['Age'] > 22,
    }

    avg_fb_pct = 0.38

    k_pct_means = {}
    bb_pct_means = {}
    hr_fb_pct_means = {}
    babip_means = {}

    for bin_name, mask in age_bins.items():
        bin_df = df[mask]
        if len(bin_df) == 0:
            continue
        total_ip = bin_df['IP'].sum()
        if total_ip == 0:
            continue

        k_pct_means[bin_name] = (bin_df['K%'] * bin_df['IP']).sum() / total_ip
        bb_pct_means[bin_name] = (bin_df['BB%'] * bin_df['IP']).sum() / total_ip
        babip_means[bin_name] = (bin_df['BABIP'] * bin_df['IP']).sum() / total_ip

        # Estimate HR/FB% from counts
        total_bf = bin_df['BF'].sum() if 'BF' in bin_df.columns else total_ip * BF_PER_IP
        total_k = bin_df['K'].sum() if 'K' in bin_df.columns else 0
        total_bb = bin_df['BB'].sum()
        total_hbp = bin_df['HBP'].sum() if 'HBP' in bin_df.columns else 0
        total_hr = bin_df['HR'].sum() if 'HR' in bin_df.columns else 0
        total_bip = max(total_bf - total_k - total_bb - total_hbp, 1)
        total_fb = total_bip * avg_fb_pct
        hr_fb_pct_means[bin_name] = total_hr / total_fb if total_fb > 0 else 0.09

    # Overall means
    total_ip = df['IP'].sum()
    overall_k_pct = (df['K%'] * df['IP']).sum() / total_ip
    overall_bb_pct = (df['BB%'] * df['IP']).sum() / total_ip
    overall_babip = (df['BABIP'] * df['IP']).sum() / total_ip

    total_bf = df['BF'].sum() if 'BF' in df.columns else total_ip * BF_PER_IP
    total_k = df['K'].sum() if 'K' in df.columns else 0
    total_bb_all = df['BB'].sum()
    total_hbp_all = df['HBP'].sum() if 'HBP' in df.columns else 0
    total_hr_all = df['HR'].sum() if 'HR' in df.columns else 0
    total_bip_all = max(total_bf - total_k - total_bb_all - total_hbp_all, 1)
    overall_hr_fb_pct = total_hr_all / (total_bip_all * avg_fb_pct)

    # Compute per-pitcher HR/FB% for variance estimation
    df = df.copy()
    df['_BIP'] = (df['BF'] - df['K'] - df['BB'] - df.get('HBP', 0)).clip(lower=1)
    df['HR/FB%'] = (df['HR'] / (df['_BIP'] * avg_fb_pct)).clip(0, 0.5)

    if 'Prior_HR' in df.columns and 'Prior_TBF' in df.columns:
        prior_so = df.get('Prior_SO', 0)
        prior_bb = df.get('Prior_BB', 0)
        prior_hbp = df.get('Prior_HBP', 0)
        df['_Prior_BIP'] = (df['Prior_TBF'] - prior_so - prior_bb - prior_hbp).clip(lower=1)
        df['Prior_HR/FB%'] = (df['Prior_HR'] / (df['_Prior_BIP'] * avg_fb_pct)).clip(0, 0.5)

    hr_fb_variance = _compute_likelihood_variance(
        df, 'HR/FB%', 'Prior_HR/FB%', overall_hr_fb_pct
    ) if 'Prior_HR/FB%' in df.columns else 0.00109
    babip_variance = _compute_likelihood_variance(
        df, 'BABIP', 'Prior_BABIP', overall_babip
    )

    mean_conf_strength = df['Conf_Strength'].mean() if 'Conf_Strength' in df.columns else 0.5
    fip_constant = calculate_fip_constant(df)
    avg_ip_per_bf = total_ip / total_bf if total_bf > 0 else 0.36
    avg_hbp_rate = total_hbp_all / total_bf if total_bf > 0 else 0.025

    return PitchingPopulationMeans(
        k_pct=k_pct_means,
        bb_pct=bb_pct_means,
        hr_fb_pct=hr_fb_pct_means,
        babip=babip_means,
        overall_k_pct=overall_k_pct,
        overall_bb_pct=overall_bb_pct,
        overall_hr_fb_pct=overall_hr_fb_pct,
        overall_babip=overall_babip,
        hr_fb_pct_variance=hr_fb_variance,
        babip_variance=babip_variance,
        babip_gb=0.244,
        babip_ld=0.716,
        babip_fb=0.108,
        mean_conf_strength=mean_conf_strength,
        fip_constant=fip_constant,
        avg_fb_pct=avg_fb_pct,
        avg_ip_per_bf=avg_ip_per_bf,
        avg_hbp_rate=avg_hbp_rate,
    )


def build_fg_pitcher_prior(
    row: pd.Series,
    pop_means: PitchingPopulationMeans,
    has_prior_data: bool,
) -> PitcherPosteriors:
    """
    Build prior distributions for a single pitcher using FanGraphs data.

    Uses Age bins instead of Class indicators, computes HR/FB% from counts,
    and skips BABIP batted-ball blending.
    """
    age = row.get('Age', 21)
    age_bin = get_age_bin(age)

    k_class_mean = pop_means.k_pct.get(age_bin, pop_means.overall_k_pct)
    bb_class_mean = pop_means.bb_pct.get(age_bin, pop_means.overall_bb_pct)
    hr_fb_class_mean = pop_means.hr_fb_pct.get(age_bin, pop_means.overall_hr_fb_pct)
    babip_class_mean = pop_means.babip.get(age_bin, pop_means.overall_babip)

    if has_prior_data and pd.notna(row.get('Prior_IP')) and row['Prior_IP'] > 0:
        prior_ip = row['Prior_IP']
        if pd.notna(row.get('Prior_TBF')) and row['Prior_TBF'] > 0:
            prior_bf = row['Prior_TBF']
        else:
            prior_bf = prior_ip * BF_PER_IP

        # K% and BB%
        k_mean = calculate_regressed_mean(
            row['Prior_K%'], prior_bf, k_class_mean, RELIABILITY_MULTIPLIERS['K%'])
        bb_mean = calculate_regressed_mean(
            row['Prior_BB%'], prior_bf, bb_class_mean, RELIABILITY_MULTIPLIERS['BB%'])

        # HR/FB% from counts
        prior_hr = row.get('Prior_HR', 0) or 0
        prior_so = row.get('Prior_SO', 0) or 0
        prior_bb_count = row.get('Prior_BB', 0) or 0
        prior_hbp = row.get('Prior_HBP', 0) or 0
        prior_bip = max(prior_bf - prior_so - prior_bb_count - prior_hbp, 1)
        prior_fb_count = prior_bip * pop_means.avg_fb_pct
        prior_hr_fb = prior_hr / prior_fb_count if prior_fb_count > 0 else hr_fb_class_mean
        prior_hr_fb = np.clip(prior_hr_fb, 0.0, 0.5)

        hr_fb_mean = calculate_regressed_mean(
            prior_hr_fb, prior_bf, hr_fb_class_mean, RELIABILITY_MULTIPLIERS['HR/FB%'])

        # BABIP: direct regression (no batted-ball blending)
        babip_mean = calculate_regressed_mean(
            row['Prior_BABIP'], prior_bf, babip_class_mean, RELIABILITY_MULTIPLIERS['BABIP'])

        # Conference adjustment
        if pd.notna(row.get('Conf_Strength')):
            k_mean = apply_pitching_conference_adjustment(
                k_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 'K%')
            bb_mean = apply_pitching_conference_adjustment(
                bb_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 'BB%')
            hr_fb_mean = apply_pitching_conference_adjustment(
                hr_fb_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 'HR/FB%')
            babip_mean = apply_pitching_conference_adjustment(
                babip_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 'BABIP')

        # Park adjustment
        if pd.notna(row.get('Prior_Park_Factor')):
            hr_fb_mean = apply_pitching_park_adjustment(
                hr_fb_mean, row['Prior_Park_Factor'], 'HR/FB%')
            babip_mean = apply_pitching_park_adjustment(
                babip_mean, row['Prior_Park_Factor'], 'BABIP')
        if pd.notna(row.get('Park_Factor')) and row.get('Park_Factor', 1.0) != 1.0:
            pf = row['Park_Factor']
            hr_fb_mean = hr_fb_mean * (1 + PARK_SENSITIVITY['HR/FB%'] * (pf - 1))
            babip_mean = babip_mean * (1 + PARK_SENSITIVITY['BABIP'] * (pf - 1))

        k_strength = calculate_prior_strength(prior_bf, STRENGTH_MULTIPLIERS['K%'])
        bb_strength = calculate_prior_strength(prior_bf, STRENGTH_MULTIPLIERS['BB%'])
        hr_fb_strength = calculate_prior_strength(prior_bf, STRENGTH_MULTIPLIERS['HR/FB%'], min_strength=0.1)
        babip_strength = calculate_prior_strength(prior_bf, STRENGTH_MULTIPLIERS['BABIP'], min_strength=0.1)

    else:
        if pop_means.freshman_k_pct is not None:
            k_mean = pop_means.freshman_k_pct
            bb_mean = pop_means.freshman_bb_pct
            hr_fb_mean = pop_means.freshman_hr_fb_pct
            babip_mean = pop_means.freshman_babip
        else:
            k_mean = k_class_mean
            bb_mean = bb_class_mean
            hr_fb_mean = hr_fb_class_mean
            babip_mean = babip_class_mean

        if pd.notna(row.get('Conf_Strength')):
            k_mean = apply_pitching_conference_adjustment(
                k_mean, row['Conf_Strength'], pop_means.mean_conf_strength, 'K%')

        if pd.notna(row.get('Park_Factor')) and row.get('Park_Factor', 1.0) != 1.0:
            pf = row['Park_Factor']
            hr_fb_mean = hr_fb_mean * (1 + PARK_SENSITIVITY['HR/FB%'] * (pf - 1))
            babip_mean = babip_mean * (1 + PARK_SENSITIVITY['BABIP'] * (pf - 1))

        k_strength = FRESHMAN_PRIOR_STRENGTH
        bb_strength = FRESHMAN_PRIOR_STRENGTH
        hr_fb_strength = 0.5
        babip_strength = 0.5

    k_mean = np.clip(k_mean, 0.10, 0.45)
    bb_mean = np.clip(bb_mean, 0.03, 0.20)
    hr_fb_mean = np.clip(hr_fb_mean, 0.02, 0.25)
    babip_mean = np.clip(babip_mean, 0.250, 0.350)

    k_model = BetaComponentModel.from_mean_strength(k_mean, k_strength, stat_name='K%')
    bb_model = BetaComponentModel.from_mean_strength(bb_mean, bb_strength, stat_name='BB%')
    hr_fb_model = NormalComponentModel.from_mean_strength(
        hr_fb_mean, hr_fb_strength, pop_means.hr_fb_pct_variance, stat_name='HR/FB%')
    babip_model = NormalComponentModel.from_mean_strength(
        babip_mean, babip_strength, pop_means.babip_variance, stat_name='BABIP')

    return PitcherPosteriors(
        player_name=row.get('Player', row.get('PlayerName', 'Unknown')),
        team=row['Team'],
        season=row.get('Season', 2026),
        k_pct=k_model,
        bb_pct=bb_model,
        hr_fb_pct=hr_fb_model,
        babip=babip_model,
    )


def build_fg_pitcher_priors(
    df: pd.DataFrame,
    pop_means: PitchingPopulationMeans,
) -> pd.DataFrame:
    """Build preseason priors for all pitchers using FanGraphs data."""
    results = []

    for idx, row in df.iterrows():
        has_prior = (
            pd.notna(row.get('Prior_IP')) and
            row.get('Prior_IP', 0) > 0
        )
        posteriors = build_fg_pitcher_prior(row, pop_means, has_prior)

        result = {
            'Player': posteriors.player_name,
            'Team': posteriors.team,
            'Season': posteriors.season,
            'Has_Prior_Data': has_prior,
            'K%_alpha': posteriors.k_pct.alpha,
            'K%_beta': posteriors.k_pct.beta,
            'K%_mean': posteriors.k_pct.get_mean(),
            'K%_std': posteriors.k_pct.get_std(),
            'BB%_alpha': posteriors.bb_pct.alpha,
            'BB%_beta': posteriors.bb_pct.beta,
            'BB%_mean': posteriors.bb_pct.get_mean(),
            'BB%_std': posteriors.bb_pct.get_std(),
            'HR/FB%_mu': posteriors.hr_fb_pct.mu,
            'HR/FB%_tau': posteriors.hr_fb_pct.tau,
            'HR/FB%_mean': posteriors.hr_fb_pct.get_mean(),
            'HR/FB%_std': posteriors.hr_fb_pct.get_std(),
            'BABIP_mu': posteriors.babip.mu,
            'BABIP_tau': posteriors.babip.tau,
            'BABIP_mean': posteriors.babip.get_mean(),
            'BABIP_std': posteriors.babip.get_std(),
        }

        for stat_name, model in [
            ('K%', posteriors.k_pct), ('BB%', posteriors.bb_pct),
            ('HR/FB%', posteriors.hr_fb_pct), ('BABIP', posteriors.babip),
        ]:
            for alpha in [0.50, 0.80, 0.90]:
                lower, upper = model.get_prediction_interval(alpha)
                pct = int(alpha * 100)
                result[f'{stat_name}_p{(100 - pct) // 2}'] = lower
                result[f'{stat_name}_p{100 - (100 - pct) // 2}'] = upper

        results.append(result)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_population_means(pop_means: PitchingPopulationMeans, filepath: Path):
    """Save population means to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(pop_means.to_dict(), f, indent=2)


def load_population_means(filepath: Path) -> PitchingPopulationMeans:
    """Load population means from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return PitchingPopulationMeans.from_dict(data)
