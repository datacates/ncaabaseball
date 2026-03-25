"""
Validation Framework for Bayesian Pitching Model

This module provides:
1. Out-of-sample testing (R² for each component stat and FIP)
2. Calibration checks (prediction interval coverage)
3. Sanity checks (impossible combinations, bounds)
4. Reliability multiplier tuning via grid search
5. Transfer regression analysis
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from bayesian_batting.model import BetaComponentModel, NormalComponentModel
from .model import PitcherPosteriors
from .priors import (
    calculate_population_means,
    build_pitcher_priors,
    build_pitcher_prior,
    PitchingPopulationMeans,
    RELIABILITY_MULTIPLIERS,
)
from .aggregation import components_to_fip


@dataclass
class PitchingValidationResults:
    """Container for pitching validation metrics."""
    r2_scores: Dict[str, float]
    mae_scores: Dict[str, float]
    rmse_scores: Dict[str, float]
    calibration: Dict[str, Dict[str, float]]
    sample_size: int
    by_player_type: Dict[str, Dict[str, float]]
    fip_mae: Optional[float] = None
    fip_rmse: Optional[float] = None
    fip_r2: Optional[float] = None


def calculate_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean(np.abs(actual - predicted))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.sqrt(np.mean((actual - predicted) ** 2))


def evaluate_preseason_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    verbose: bool = True,
) -> PitchingValidationResults:
    """
    Evaluate preseason pitching predictions using out-of-sample testing.

    Uses minimum 30 IP for evaluation (vs 100 PA for batting).
    """
    pop_means = calculate_population_means(train_df)

    if verbose:
        print("=" * 60)
        print("Population Means (from training data)")
        print("=" * 60)
        print(f"\nK% by class: {pop_means.k_pct}")
        print(f"BB% by class: {pop_means.bb_pct}")
        print(f"HR/FB% by class: {pop_means.hr_fb_pct}")
        print(f"BABIP by class: {pop_means.babip}")
        print(f"\nHR/FB% variance: {pop_means.hr_fb_pct_variance:.6f}")
        print(f"BABIP variance: {pop_means.babip_variance:.6f}")
        print(f"FIP constant: {pop_means.fip_constant:.3f}")

    # Build priors for test data
    predictions_df = build_pitcher_priors(test_df, pop_means)

    # Merge with actual outcomes
    merge_cols = ['Season', 'Player', 'Team']
    actual_cols = ['K%', 'BB%', 'HR/FB%', 'BABIP', 'IP', 'FIP',
                   'Changed_Team', 'Class_FR', 'Class_SO', 'Class_JR',
                   'Class_SR', 'Class_GR', 'Prior_IP']
    available_actual_cols = [c for c in actual_cols if c in test_df.columns]

    merged = predictions_df.merge(
        test_df[merge_cols + available_actual_cols],
        on=merge_cols,
        suffixes=('_pred', '_actual'),
    )

    # Filter to pitchers with sufficient IP
    min_ip = 30
    merged_filtered = merged[merged['IP'] >= min_ip].copy()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Out-of-Sample Validation (test set)")
        print(f"{'=' * 60}")
        print(f"Test pitchers with IP >= {min_ip}: {len(merged_filtered)}")

    # Calculate metrics for each component stat
    stats_to_evaluate = ['K%', 'BB%', 'HR/FB%', 'BABIP']
    r2_scores = {}
    mae_scores = {}
    rmse_scores = {}

    for stat in stats_to_evaluate:
        if stat not in merged_filtered.columns:
            continue
        actual = merged_filtered[stat].values
        predicted = merged_filtered[f'{stat}_mean'].values

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

    # FIP prediction accuracy
    fip_mae = fip_rmse = fip_r2 = None
    if 'FIP' in merged_filtered.columns:
        # Calculate predicted FIP from component means
        pred_fip = merged_filtered.apply(
            lambda r: components_to_fip(
                r['K%_mean'], r['BB%_mean'], r['HR/FB%_mean'], r['BABIP_mean'],
                fip_constant=pop_means.fip_constant,
                fb_pct=pop_means.avg_fb_pct,
                ip_per_bf=pop_means.avg_ip_per_bf,
                hbp_rate=pop_means.avg_hbp_rate,
            ),
            axis=1,
        )
        actual_fip = merged_filtered['FIP'].values
        valid = ~(np.isnan(actual_fip) | np.isnan(pred_fip.values))

        fip_mae = calculate_mae(actual_fip[valid], pred_fip.values[valid])
        fip_rmse = calculate_rmse(actual_fip[valid], pred_fip.values[valid])
        fip_r2 = calculate_r2(actual_fip[valid], pred_fip.values[valid])

        if verbose:
            print(f"\nFIP (derived from components):")
            print(f"  R²:   {fip_r2:.4f}")
            print(f"  MAE:  {fip_mae:.4f}")
            print(f"  RMSE: {fip_rmse:.4f}")

    # Calibration checks
    calibration = check_calibration(merged_filtered, verbose=verbose)

    # By player type
    by_type = analyze_by_player_type(merged_filtered, stats_to_evaluate, verbose=verbose)

    return PitchingValidationResults(
        r2_scores=r2_scores,
        mae_scores=mae_scores,
        rmse_scores=rmse_scores,
        calibration=calibration,
        sample_size=len(merged_filtered),
        by_player_type=by_type,
        fip_mae=fip_mae,
        fip_rmse=fip_rmse,
        fip_r2=fip_r2,
    )


def check_calibration(
    df: pd.DataFrame,
    coverage_levels: List[float] = [0.50, 0.80, 0.90],
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Check if prediction intervals have correct coverage."""
    stats = ['K%', 'BB%', 'HR/FB%', 'BABIP']
    calibration = {}

    if verbose:
        print(f"\n{'=' * 60}")
        print("Calibration Check (Prediction Interval Coverage)")
        print("=" * 60)

    for stat in stats:
        if stat not in df.columns:
            continue
        calibration[stat] = {}

        for level in coverage_levels:
            pct = int(level * 100)
            lower_col = f'{stat}_p{(100 - pct) // 2}'
            upper_col = f'{stat}_p{100 - (100 - pct) // 2}'

            if lower_col not in df.columns or upper_col not in df.columns:
                continue

            actual = df[stat].values
            lower = df[lower_col].values
            upper = df[upper_col].values

            in_interval = (actual >= lower) & (actual <= upper)
            actual_coverage = np.mean(in_interval)
            calibration[stat][level] = actual_coverage

        if verbose:
            print(f"\n{stat}:")
            for level, coverage in calibration[stat].items():
                diff = coverage - level
                status = "OK" if abs(diff) < 0.05 else ("HIGH" if diff > 0 else "LOW")
                print(f"  {int(level*100)}% interval: {coverage:.1%} coverage ({status})")

    return calibration


def analyze_by_player_type(
    df: pd.DataFrame,
    stats: List[str],
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Analyze prediction accuracy by player type."""
    results = {}

    groups = {
        'Returners (same team)': (df['Changed_Team'] == False) & (df.get('Prior_IP', pd.Series(dtype=float)).notna()),
        'Transfers': df['Changed_Team'] == True,
    }
    if 'Prior_IP' in df.columns:
        groups['No prior data'] = df['Prior_IP'].isna() | (df['Prior_IP'] == 0)

    if verbose:
        print(f"\n{'=' * 60}")
        print("Analysis by Player Type")
        print("=" * 60)

    for group_name, mask in groups.items():
        subset = df[mask]
        if len(subset) < 20:
            if verbose:
                print(f"\n{group_name}: (n={len(subset)}, too few)")
            continue

        results[group_name] = {'n': len(subset)}
        if verbose:
            print(f"\n{group_name} (n={len(subset)}):")

        for stat in stats:
            if stat not in subset.columns or f'{stat}_mean' not in subset.columns:
                continue
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


def flag_impossible_pitching_combinations(
    predictions_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Flag predictions with impossible or unlikely combinations."""
    flags = []

    for idx, row in predictions_df.iterrows():
        issues = []

        if row.get('K%_mean', 0) < 0.12 and row.get('BB%_mean', 0) > 0.12:
            issues.append(f"Low K% ({row['K%_mean']:.3f}) with high BB% ({row['BB%_mean']:.3f})")

        if row.get('HR/FB%_mean', 0) > 0.20:
            issues.append(f"Extreme HR/FB% ({row['HR/FB%_mean']:.3f})")

        if row.get('BB%_mean', 0) > 0.18:
            issues.append(f"Very high BB% ({row['BB%_mean']:.3f})")

        if row.get('BABIP_mean', 0) < 0.255 or row.get('BABIP_mean', 0) > 0.345:
            issues.append(f"Unusual BABIP ({row['BABIP_mean']:.3f})")

        if issues:
            flags.append({
                'Player': row.get('Player', ''),
                'Team': row.get('Team', ''),
                'Season': row.get('Season', ''),
                'Issues': '; '.join(issues),
                'K%_mean': row.get('K%_mean', 0),
                'BB%_mean': row.get('BB%_mean', 0),
                'HR/FB%_mean': row.get('HR/FB%_mean', 0),
                'BABIP_mean': row.get('BABIP_mean', 0),
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
    verbose: bool = True,
) -> Dict[str, float]:
    """Verify that predictions for transfers show appropriate regression."""
    merged = predictions_df.merge(
        df[['Season', 'Player', 'Team', 'Changed_Team', 'Prior_IP',
            'Prior_K%', 'Prior_BB%', 'Prior_HR/FB%', 'Prior_BABIP']],
        on=['Season', 'Player', 'Team'],
    )

    with_prior = merged[merged['Prior_IP'] > 30].copy()
    results = {}

    for stat in ['K%', 'BB%', 'HR/FB%', 'BABIP']:
        prior_col = f'Prior_{stat}'
        pred_col = f'{stat}_mean'
        std_col = f'{stat}_std'

        if prior_col not in with_prior.columns:
            continue

        with_prior[f'{stat}_regression'] = np.abs(
            with_prior[pred_col] - with_prior[prior_col]
        )

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
        }

    if verbose:
        print(f"\n{'=' * 60}")
        print("Transfer vs Returner Regression Analysis")
        print("=" * 60)
        for stat, metrics in results.items():
            print(f"\n{stat}:")
            print(f"  Transfer regression: {metrics['transfer_regression']:.4f}")
            print(f"  Returner regression: {metrics['returner_regression']:.4f}")
            print(f"  Ratio (should be > 1.0): {metrics['regression_ratio']:.2f}")

    return results


def tune_reliability_multipliers(
    df: pd.DataFrame,
    train_years: List[int],
    test_years: List[int],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Grid search over reliability multiplier combinations to find optimal values.

    Optimizes composite score: weighted mean of component R² values.
    """
    import itertools

    grid = {
        'K%': [0.75, 0.80, 0.85, 0.90],
        'BB%': [0.55, 0.60, 0.65, 0.70],
        'HR/FB%': [0.20, 0.25, 0.30, 0.35],
        'BABIP': [0.03, 0.05, 0.07, 0.10],
    }

    train_df = df[df['Season'].isin(train_years)].copy()
    test_df = df[df['Season'].isin(test_years)].copy()

    pop_means = calculate_population_means(train_df)

    best_score = -np.inf
    best_multipliers = RELIABILITY_MULTIPLIERS.copy()
    all_results = []

    # Save original multipliers
    import bayesian_pitching.priors as priors_module
    original_multipliers = priors_module.RELIABILITY_MULTIPLIERS.copy()

    combos = list(itertools.product(
        grid['K%'], grid['BB%'], grid['HR/FB%'], grid['BABIP']
    ))

    if verbose:
        print(f"\nTuning reliability multipliers ({len(combos)} combinations)...")

    for k_m, bb_m, hr_m, babip_m in combos:
        # Temporarily set multipliers
        priors_module.RELIABILITY_MULTIPLIERS = {
            'K%': k_m, 'BB%': bb_m, 'HR/FB%': hr_m, 'BABIP': babip_m
        }

        try:
            predictions_df = build_pitcher_priors(test_df, pop_means)

            merged = predictions_df.merge(
                test_df[['Season', 'Player', 'Team', 'K%', 'BB%', 'HR/FB%', 'BABIP', 'IP']],
                on=['Season', 'Player', 'Team'],
                suffixes=('_pred', '_actual'),
            )
            merged = merged[merged['IP'] >= 30]

            r2_scores = {}
            for stat in ['K%', 'BB%', 'HR/FB%', 'BABIP']:
                if stat not in merged.columns:
                    continue
                actual = merged[stat].values
                predicted = merged[f'{stat}_mean'].values
                valid = ~(np.isnan(actual) | np.isnan(predicted))
                if valid.sum() < 20:
                    r2_scores[stat] = 0.0
                else:
                    r2_scores[stat] = calculate_r2(actual[valid], predicted[valid])

            # Composite score: weighted average (K% and BB% matter more for FIP)
            weights = {'K%': 0.35, 'BB%': 0.30, 'HR/FB%': 0.25, 'BABIP': 0.10}
            composite = sum(
                r2_scores.get(s, 0) * w for s, w in weights.items()
            )

            all_results.append({
                'K%': k_m, 'BB%': bb_m, 'HR/FB%': hr_m, 'BABIP': babip_m,
                'composite': composite,
                **{f'R2_{s}': r2_scores.get(s, 0) for s in ['K%', 'BB%', 'HR/FB%', 'BABIP']},
            })

            if composite > best_score:
                best_score = composite
                best_multipliers = {
                    'K%': k_m, 'BB%': bb_m, 'HR/FB%': hr_m, 'BABIP': babip_m
                }
        except Exception:
            continue

    # Restore original multipliers
    priors_module.RELIABILITY_MULTIPLIERS = original_multipliers

    if verbose:
        print(f"\nBest multipliers (composite R² = {best_score:.4f}):")
        for stat, val in best_multipliers.items():
            print(f"  {stat}: {val}")

        # Show top 5 combinations
        results_df = pd.DataFrame(all_results).sort_values('composite', ascending=False)
        print(f"\nTop 5 combinations:")
        print(results_df.head(5).to_string(index=False))

    return best_multipliers


def run_full_pitching_validation(
    df: pd.DataFrame,
    train_years: List[int] = [2018, 2019, 2021, 2022],
    test_years: List[int] = [2023, 2024],
    verbose: bool = True,
) -> Tuple[PitchingValidationResults, pd.DataFrame]:
    """Run the complete pitching validation suite."""
    print("=" * 70)
    print("BAYESIAN PITCHING MODEL VALIDATION")
    print("=" * 70)
    print(f"\nTraining years: {train_years}")
    print(f"Test years: {test_years}")

    train_df = df[df['Season'].isin(train_years)].copy()
    test_df = df[df['Season'].isin(test_years)].copy()

    print(f"\nTraining set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # Run validation
    results = evaluate_preseason_predictions(train_df, test_df, verbose=verbose)

    # Generate predictions for further analysis
    pop_means = calculate_population_means(train_df)
    predictions_df = build_pitcher_priors(test_df, pop_means)

    # Sanity checks
    flagged = flag_impossible_pitching_combinations(predictions_df, verbose=verbose)

    # Transfer regression
    transfer_analysis = check_transfer_regression(
        test_df, predictions_df, verbose=verbose
    )

    # Summary
    if verbose:
        print(f"\n{'=' * 70}")
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"\nTarget: R² > 0.15 for preseason pitching predictions")
        print("\nComponent Results:")
        for stat, r2 in results.r2_scores.items():
            status = "PASS" if r2 > 0.15 else "FAIL"
            print(f"  {stat}: R² = {r2:.4f} [{status}]")

        if results.fip_r2 is not None:
            status = "PASS" if results.fip_r2 > 0.10 else "FAIL"
            print(f"\nFIP: R² = {results.fip_r2:.4f} [{status}]")
            print(f"FIP MAE: {results.fip_mae:.3f}")

        all_pass = all(r2 > 0.15 for r2 in results.r2_scores.values())
        print(f"\nOverall: {'PASS' if all_pass else 'NEEDS IMPROVEMENT'}")

    return results, predictions_df
