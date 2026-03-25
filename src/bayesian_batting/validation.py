"""
Validation Framework for Bayesian Batting Model

This module provides:
1. Out-of-sample testing (R² for each component stat)
2. Calibration checks (prediction interval coverage)
3. Sanity checks (impossible combinations, bounds)
4. Update validation (convergence tracking)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .model import BetaComponentModel, NormalComponentModel, PlayerPosteriors
from .priors import (
    calculate_population_means, build_player_priors, PopulationMeans,
    build_player_prior
)


@dataclass
class ValidationResults:
    """Container for validation metrics."""
    r2_scores: Dict[str, float]
    mae_scores: Dict[str, float]
    rmse_scores: Dict[str, float]
    calibration: Dict[str, Dict[str, float]]
    sample_size: int
    by_player_type: Dict[str, Dict[str, float]]


def calculate_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate R-squared (coefficient of determination)."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(actual - predicted))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def evaluate_preseason_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    verbose: bool = True
) -> ValidationResults:
    """
    Evaluate preseason predictions using out-of-sample testing.

    Args:
        train_df: Training data (e.g., 2018-2022)
        test_df: Test data (e.g., 2023-2024)
        verbose: Whether to print results

    Returns:
        ValidationResults with R², MAE, RMSE for each stat
    """
    # Calculate population means from training data
    pop_means = calculate_population_means(train_df)

    if verbose:
        print("=" * 60)
        print("Population Means (from training data)")
        print("=" * 60)
        print(f"\nK% by class: {pop_means.k_pct}")
        print(f"BB% by class: {pop_means.bb_pct}")
        print(f"ISO by class: {pop_means.iso}")
        print(f"BABIP by class: {pop_means.babip}")
        print(f"\nISO variance: {pop_means.iso_variance:.6f}")
        print(f"BABIP variance: {pop_means.babip_variance:.6f}")

    # Build priors for test data
    predictions_df = build_player_priors(test_df, pop_means)

    # Merge with actual outcomes
    merged = predictions_df.merge(
        test_df[['Season', 'Player', 'Team', 'K%', 'BB%', 'ISO', 'BABIP', 'PA',
                 'Changed_Team', 'Class_FR', 'Class_SO', 'Class_JR', 'Class_SR', 'Class_GR',
                 'Prior_PA']],
        on=['Season', 'Player', 'Team'],
        suffixes=('_pred', '_actual')
    )

    # Filter to players with sufficient PA for reliable actuals
    min_pa = 100
    merged_filtered = merged[merged['PA'] >= min_pa].copy()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Out-of-Sample Validation (test set)")
        print(f"{'=' * 60}")
        print(f"Test players with PA >= {min_pa}: {len(merged_filtered)}")

    # Calculate metrics for each stat
    stats_to_evaluate = ['K%', 'BB%', 'ISO', 'BABIP']
    r2_scores = {}
    mae_scores = {}
    rmse_scores = {}

    for stat in stats_to_evaluate:
        actual = merged_filtered[stat].values
        predicted = merged_filtered[f'{stat}_mean'].values

        # Remove any NaN
        valid = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[valid]
        predicted = predicted[valid]

        r2_scores[stat] = calculate_r2(actual, predicted)
        mae_scores[stat] = calculate_mae(actual, predicted)
        rmse_scores[stat] = calculate_rmse(actual, predicted)

        if verbose:
            print(f"\n{stat}:")
            print(f"  R²:   {r2_scores[stat]:.4f}")
            print(f"  MAE:  {mae_scores[stat]:.4f}")
            print(f"  RMSE: {rmse_scores[stat]:.4f}")

    # Calibration checks
    calibration = check_calibration(merged_filtered, verbose=verbose)

    # By player type analysis
    by_type = analyze_by_player_type(merged_filtered, stats_to_evaluate, verbose=verbose)

    return ValidationResults(
        r2_scores=r2_scores,
        mae_scores=mae_scores,
        rmse_scores=rmse_scores,
        calibration=calibration,
        sample_size=len(merged_filtered),
        by_player_type=by_type
    )


def check_calibration(
    df: pd.DataFrame,
    coverage_levels: List[float] = [0.50, 0.80, 0.90],
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Check if prediction intervals have correct coverage.

    Args:
        df: DataFrame with predictions and actuals
        coverage_levels: List of coverage levels to check
        verbose: Whether to print results

    Returns:
        Dict mapping stat -> {coverage_level: actual_coverage}
    """
    stats = ['K%', 'BB%', 'ISO', 'BABIP']
    calibration = {}

    if verbose:
        print(f"\n{'=' * 60}")
        print("Calibration Check (Prediction Interval Coverage)")
        print("=" * 60)

    for stat in stats:
        calibration[stat] = {}

        for level in coverage_levels:
            pct = int(level * 100)
            lower_col = f'{stat}_p{(100-pct)//2}'
            upper_col = f'{stat}_p{100 - (100-pct)//2}'

            if lower_col not in df.columns or upper_col not in df.columns:
                continue

            actual = df[stat].values
            lower = df[lower_col].values
            upper = df[upper_col].values

            # Check how many actuals fall within the interval
            in_interval = (actual >= lower) & (actual <= upper)
            actual_coverage = np.mean(in_interval)

            calibration[stat][level] = actual_coverage

        if verbose:
            print(f"\n{stat}:")
            for level, coverage in calibration[stat].items():
                expected = level
                diff = coverage - expected
                status = "OK" if abs(diff) < 0.05 else ("HIGH" if diff > 0 else "LOW")
                print(f"  {int(level*100)}% interval: {coverage:.1%} coverage ({status})")

    return calibration


def analyze_by_player_type(
    df: pd.DataFrame,
    stats: List[str],
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Analyze prediction accuracy by player type.

    Player types:
    - Freshmen: Class_FR = True (if they somehow have data)
    - Returners (same team): Changed_Team = False and Prior_PA > 0
    - Transfers: Changed_Team = True

    Args:
        df: DataFrame with predictions and actuals
        stats: List of stats to analyze
        verbose: Whether to print results

    Returns:
        Dict mapping player_type -> stat -> R²
    """
    results = {}

    # Define player type groups
    groups = {
        'Returners (same team)': (df['Changed_Team'] == False) & (df['Prior_PA'] > 0),
        'Transfers': df['Changed_Team'] == True,
        'No prior data': df['Prior_PA'].isna() | (df['Prior_PA'] == 0)
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print("Analysis by Player Type")
        print("=" * 60)

    for group_name, mask in groups.items():
        subset = df[mask]

        if len(subset) < 20:
            if verbose:
                print(f"\n{group_name}: (n={len(subset)}, too few for reliable stats)")
            continue

        results[group_name] = {'n': len(subset)}

        if verbose:
            print(f"\n{group_name} (n={len(subset)}):")

        for stat in stats:
            actual = subset[stat].values
            predicted = subset[f'{stat}_mean'].values

            valid = ~(np.isnan(actual) | np.isnan(predicted))
            if valid.sum() < 10:
                continue

            r2 = calculate_r2(actual[valid], predicted[valid])
            results[group_name][stat] = r2

            if verbose:
                print(f"  {stat} R²: {r2:.4f}")

    return results


def flag_impossible_combinations(
    predictions_df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Flag predictions that have impossible or unlikely combinations.

    Rules:
    - K% > 50% AND BABIP > .400 is very unlikely
    - K% < 10% is rare but possible
    - ISO > 0.35 with K% > 40% is unusual
    - BB% > 25% is very rare

    Args:
        predictions_df: DataFrame with predicted means
        verbose: Whether to print flagged cases

    Returns:
        DataFrame of flagged predictions
    """
    flags = []

    for idx, row in predictions_df.iterrows():
        issues = []

        # High K% with high BABIP
        if row.get('K%_mean', 0) > 0.40 and row.get('BABIP_mean', 0) > 0.380:
            issues.append(f"High K% ({row['K%_mean']:.3f}) with high BABIP ({row['BABIP_mean']:.3f})")

        # Extreme ISO
        if row.get('ISO_mean', 0) > 0.35:
            issues.append(f"Extreme ISO ({row['ISO_mean']:.3f})")

        # Extreme BB%
        if row.get('BB%_mean', 0) > 0.22:
            issues.append(f"Very high BB% ({row['BB%_mean']:.3f})")

        # Very low K%
        if row.get('K%_mean', 0) < 0.08:
            issues.append(f"Very low K% ({row['K%_mean']:.3f})")

        if issues:
            flags.append({
                'Player': row.get('Player', ''),
                'Team': row.get('Team', ''),
                'Season': row.get('Season', ''),
                'Issues': '; '.join(issues),
                'K%_mean': row.get('K%_mean', 0),
                'BB%_mean': row.get('BB%_mean', 0),
                'ISO_mean': row.get('ISO_mean', 0),
                'BABIP_mean': row.get('BABIP_mean', 0)
            })

    flagged_df = pd.DataFrame(flags)

    if verbose and len(flagged_df) > 0:
        print(f"\n{'=' * 60}")
        print(f"Sanity Check: {len(flagged_df)} predictions flagged")
        print("=" * 60)
        print(flagged_df.head(10).to_string(index=False))

    return flagged_df


def check_transfer_regression(
    df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Verify that predictions for transfers show appropriate regression.

    Transfers should have:
    - More regression to the mean than same-team returners
    - Wider prediction intervals

    Args:
        df: Original data with Prior stats
        predictions_df: Predictions DataFrame
        verbose: Whether to print results

    Returns:
        Dict with regression statistics
    """
    # Merge predictions with original data
    merged = predictions_df.merge(
        df[['Season', 'Player', 'Team', 'Changed_Team', 'Prior_PA',
            'Prior_K%', 'Prior_BB%', 'Prior_ISO', 'Prior_BABIP']],
        on=['Season', 'Player', 'Team']
    )

    # Filter to players with prior data
    with_prior = merged[merged['Prior_PA'] > 100].copy()

    results = {}

    for stat in ['K%', 'BB%', 'ISO', 'BABIP']:
        prior_col = f'Prior_{stat}'
        pred_col = f'{stat}_mean'
        std_col = f'{stat}_std'

        if prior_col not in with_prior.columns:
            continue

        # Calculate regression amount (how much prediction differs from prior)
        with_prior[f'{stat}_regression'] = np.abs(
            with_prior[pred_col] - with_prior[prior_col]
        )

        # Compare transfers vs returners
        transfers = with_prior[with_prior['Changed_Team'] == True]
        returners = with_prior[with_prior['Changed_Team'] == False]

        if len(transfers) < 10 or len(returners) < 10:
            continue

        transfer_regression = transfers[f'{stat}_regression'].mean()
        returner_regression = returners[f'{stat}_regression'].mean()

        transfer_std = transfers[std_col].mean()
        returner_std = returners[std_col].mean()

        results[stat] = {
            'transfer_regression': transfer_regression,
            'returner_regression': returner_regression,
            'regression_ratio': transfer_regression / (returner_regression + 0.001),
            'transfer_uncertainty': transfer_std,
            'returner_uncertainty': returner_std,
            'uncertainty_ratio': transfer_std / (returner_std + 0.001)
        }

    if verbose:
        print(f"\n{'=' * 60}")
        print("Transfer vs Returner Regression Analysis")
        print("=" * 60)
        for stat, metrics in results.items():
            print(f"\n{stat}:")
            print(f"  Transfer regression amount: {metrics['transfer_regression']:.4f}")
            print(f"  Returner regression amount: {metrics['returner_regression']:.4f}")
            print(f"  Ratio (should be > 1.0): {metrics['regression_ratio']:.2f}")
            print(f"  Transfer uncertainty: {metrics['transfer_uncertainty']:.4f}")
            print(f"  Returner uncertainty: {metrics['returner_uncertainty']:.4f}")

    return results


def run_full_validation(
    df: pd.DataFrame,
    train_years: List[int] = [2018, 2019, 2021, 2022],
    test_years: List[int] = [2023, 2024],
    verbose: bool = True
) -> Tuple[ValidationResults, pd.DataFrame]:
    """
    Run the complete validation suite.

    Args:
        df: Full training dataset
        train_years: Years to use for training
        test_years: Years to use for testing
        verbose: Whether to print results

    Returns:
        Tuple of (ValidationResults, predictions_df)
    """
    print("=" * 70)
    print("BAYESIAN BATTING MODEL VALIDATION")
    print("=" * 70)
    print(f"\nTraining years: {train_years}")
    print(f"Test years: {test_years}")

    # Split data
    train_df = df[df['Season'].isin(train_years)].copy()
    test_df = df[df['Season'].isin(test_years)].copy()

    print(f"\nTraining set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # Run validation
    results = evaluate_preseason_predictions(train_df, test_df, verbose=verbose)

    # Calculate population means for generating predictions
    pop_means = calculate_population_means(train_df)
    predictions_df = build_player_priors(test_df, pop_means)

    # Sanity checks
    flagged = flag_impossible_combinations(predictions_df, verbose=verbose)

    # Transfer regression check
    transfer_analysis = check_transfer_regression(
        test_df, predictions_df, verbose=verbose
    )

    # Summary
    if verbose:
        print(f"\n{'=' * 70}")
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"\nTarget: R² > 0.25 for preseason predictions")
        print("\nResults:")
        for stat, r2 in results.r2_scores.items():
            status = "PASS" if r2 > 0.25 else "FAIL"
            print(f"  {stat}: R² = {r2:.4f} [{status}]")

        # Overall pass/fail
        all_pass = all(r2 > 0.25 for r2 in results.r2_scores.values())
        print(f"\nOverall: {'PASS' if all_pass else 'NEEDS IMPROVEMENT'}")

    return results, predictions_df
