#!/usr/bin/env python3
"""
Blended Priors ATS-Focused Tuning Script.

Optimizes blending schedule for 5+ Edge Win % vs Close Line using
Leave-One-Year-Out Cross Validation.

Primary target: Maximize 5+ Edge Win % vs Close Line
Secondary targets: ROI on 5+ edges, fold stability
Constraint: MAE within +0.5 of SP+-only baseline

Usage:
    python3 scripts/tune_blended_priors_ats.py [--quick] [--verbose]
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest import fetch_all_season_data, run_backtest
from src.models.blended_priors import BlendSchedule, create_blended_generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for name in ["src.models", "src.api", "scripts.backtest", "urllib3"]:
    logging.getLogger(name).setLevel(logging.WARNING)


@dataclass
class FoldResult:
    """Results from a single LOYO fold."""

    holdout_year: int
    games_total: int
    games_5plus: int
    wins_5plus: int
    losses_5plus: int
    win_pct_5plus: float
    wilson_lower: float
    wilson_upper: float
    roi_5plus: float
    mae: float
    mean_error: float

    # Edge buckets
    games_3to5: int = 0
    wins_3to5: int = 0
    games_5to7: int = 0
    wins_5to7: int = 0
    games_7plus: int = 0
    wins_7plus: int = 0

    # Open vs Close
    win_pct_5plus_open: float = 0.0
    win_pct_5plus_close: float = 0.0


@dataclass
class ConfigResult:
    """Results for a blending configuration across all folds."""

    config_name: str
    schedule_type: str
    params: dict

    # Aggregate metrics
    mean_win_pct_5plus: float = 0.0
    std_win_pct_5plus: float = 0.0
    mean_roi_5plus: float = 0.0
    mean_mae: float = 0.0
    total_games_5plus: int = 0
    total_wins_5plus: int = 0

    # Fold results
    fold_results: list = field(default_factory=list)

    # Acceptance flags
    passes_win_pct_threshold: bool = False
    passes_fold_stability: bool = False
    passes_fold_count: bool = False
    passes_mae_constraint: bool = False
    accepted: bool = False
    rejection_reason: str = ""


def wilson_confidence_interval(wins: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = wins / n

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    return (max(0, center - margin), min(1, center + margin))


def calculate_roi(wins: int, losses: int, juice: float = 0.10) -> float:
    """Calculate ROI assuming -110 juice on both sides."""
    if wins + losses == 0:
        return 0.0

    # Win pays 1.0, loss costs 1.1 (at -110)
    profit = wins * 1.0 - losses * 1.1
    total_wagered = (wins + losses) * 1.1  # Risk 1.1 per bet

    return profit / total_wagered * 100


class BlendScheduleGenerator:
    """Generates blending schedules from various functional forms."""

    @staticmethod
    def linear_decay(sp_start: float, sp_end: float, transition_week: int = 8) -> BlendSchedule:
        """Linear decay from sp_start to sp_end over weeks 1-transition_week."""
        week_weights = {}
        for week in range(1, transition_week + 1):
            t = (week - 1) / (transition_week - 1) if transition_week > 1 else 1.0
            w_sp = sp_start + t * (sp_end - sp_start)
            week_weights[week] = (w_sp, 1.0 - w_sp)

        return BlendSchedule(week_weights=week_weights, default_sp_weight=sp_end)

    @staticmethod
    def step_function(sp_early: float, sp_late: float, cutoff_week: int = 4) -> BlendSchedule:
        """Step function: sp_early for weeks 1-(cutoff-1), sp_late for week cutoff+."""
        week_weights = {}
        for week in range(1, cutoff_week):
            week_weights[week] = (sp_early, 1.0 - sp_early)

        return BlendSchedule(week_weights=week_weights, default_sp_weight=sp_late)

    @staticmethod
    def logistic_curve(sp_min: float, sp_max: float, midpoint: float = 4.0, steepness: float = 1.0) -> BlendSchedule:
        """Logistic curve transition from sp_min to sp_max."""
        week_weights = {}
        for week in range(1, 16):
            # Sigmoid: high at week 1 (sp_min), low by week 8+ (sp_max)
            # Inverted so SP+ weight increases over time
            t = 1 / (1 + np.exp(-steepness * (week - midpoint)))
            w_sp = sp_min + t * (sp_max - sp_min)
            week_weights[week] = (w_sp, 1.0 - w_sp)

        return BlendSchedule(week_weights=week_weights, default_sp_weight=sp_max)

    @staticmethod
    def piecewise(sp_phase1: float, sp_phase2: float, sp_phase3: float) -> BlendSchedule:
        """Piecewise: Phase 1 (weeks 1-3), Phase 2 (weeks 4-8), Phase 3 (weeks 9+)."""
        week_weights = {}
        for week in range(1, 4):
            week_weights[week] = (sp_phase1, 1.0 - sp_phase1)
        for week in range(4, 9):
            week_weights[week] = (sp_phase2, 1.0 - sp_phase2)

        return BlendSchedule(week_weights=week_weights, default_sp_weight=sp_phase3)


def generate_search_space(quick: bool = False) -> list[tuple[str, str, dict, BlendSchedule]]:
    """Generate the search space of blending configurations."""
    configs = []

    if quick:
        # Minimal search space for testing infrastructure
        # Just test a few piecewise configs that vary Phase 1 (weeks 1-3) weights
        for sp_p1 in [0.4, 0.6, 0.8, 1.0]:
            name = f"piecewise_sp{sp_p1:.1f}_0.8_1.0"
            params = {"sp_phase1": sp_p1, "sp_phase2": 0.8, "sp_phase3": 1.0}
            schedule = BlendScheduleGenerator.piecewise(sp_p1, 0.8, 1.0)
            configs.append((name, "piecewise", params, schedule))

        # Add baseline
        configs.append(("baseline_sp_only", "baseline", {}, BlendSchedule(week_weights={}, default_sp_weight=1.0)))

        logger.info(f"Generated {len(configs)} configurations (quick mode)")
        return configs

    # Full search space

    # 1. Linear decay configurations
    for sp_start, sp_end in product([0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0]):
        if sp_start < sp_end:  # SP+ weight should increase over time
            for trans_week in [4, 6, 8]:
                name = f"linear_sp{sp_start:.1f}_to_{sp_end:.1f}_w{trans_week}"
                params = {"sp_start": sp_start, "sp_end": sp_end, "transition_week": trans_week}
                schedule = BlendScheduleGenerator.linear_decay(sp_start, sp_end, trans_week)
                configs.append((name, "linear", params, schedule))

    # 2. Step function configurations
    for sp_early, sp_late in product([0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0]):
        if sp_early < sp_late:
            for cutoff in [3, 4, 5]:
                name = f"step_sp{sp_early:.1f}_to_{sp_late:.1f}_cut{cutoff}"
                params = {"sp_early": sp_early, "sp_late": sp_late, "cutoff_week": cutoff}
                schedule = BlendScheduleGenerator.step_function(sp_early, sp_late, cutoff)
                configs.append((name, "step", params, schedule))

    # 3. Logistic curve configurations
    for sp_min, sp_max in product([0.4, 0.5], [0.8, 0.9, 1.0]):
        for midpoint in [3.0, 4.0, 5.0]:
            for steepness in [0.5, 1.0, 1.5]:
                name = f"logistic_sp{sp_min:.1f}_to_{sp_max:.1f}_mid{midpoint:.0f}_k{steepness:.1f}"
                params = {"sp_min": sp_min, "sp_max": sp_max, "midpoint": midpoint, "steepness": steepness}
                schedule = BlendScheduleGenerator.logistic_curve(sp_min, sp_max, midpoint, steepness)
                configs.append((name, "logistic", params, schedule))

    # 4. Piecewise configurations (most interpretable)
    for sp_p1, sp_p2, sp_p3 in product([0.4, 0.5, 0.6], [0.6, 0.7, 0.8], [0.8, 0.9, 1.0]):
        if sp_p1 <= sp_p2 <= sp_p3:  # Monotonic increase
            name = f"piecewise_sp{sp_p1:.1f}_{sp_p2:.1f}_{sp_p3:.1f}"
            params = {"sp_phase1": sp_p1, "sp_phase2": sp_p2, "sp_phase3": sp_p3}
            schedule = BlendScheduleGenerator.piecewise(sp_p1, sp_p2, sp_p3)
            configs.append((name, "piecewise", params, schedule))

    # 5. Baseline: SP+-only (for comparison)
    configs.append(("baseline_sp_only", "baseline", {}, BlendSchedule(week_weights={}, default_sp_weight=1.0)))

    logger.info(f"Generated {len(configs)} configurations to test")
    return configs


def run_single_fold(
    holdout_year: int,
    schedule: BlendSchedule,
    end_week: int = 3,  # Focus on weeks 1-3 where priors matter
) -> Optional[FoldResult]:
    """Run backtest for a single LOYO fold with custom schedule."""
    try:
        # Run backtest with blended priors and custom schedule
        # Focus on weeks 1-3 (or specified end_week) where priors have most impact
        result = run_backtest(
            years=[holdout_year],
            start_week=1,
            end_week=end_week,
            use_priors=True,
            use_blended_priors=True,
            blend_schedule=schedule,
        )

        # Extract ATS results DataFrame (has vegas_spread) from result dict
        if isinstance(result, dict):
            results_df = result.get("ats_results")
        else:
            results_df = result

        if results_df is None or len(results_df) == 0:
            return None

        # Use the ATS results which has vegas_spread already merged
        df = results_df.copy()
        df['edge_close'] = np.abs(df['predicted_spread'] - df['vegas_spread'])
        df['ats_win_close'] = (
            ((df['edge_close'] > 0) & (df['predicted_spread'] > df['vegas_spread']) & (df['actual_margin'] > df['vegas_spread'])) |
            ((df['edge_close'] > 0) & (df['predicted_spread'] < df['vegas_spread']) & (df['actual_margin'] < df['vegas_spread']))
        ).astype(int)

        # Filter to games with valid spreads
        df = df[df['vegas_spread'].notna()]

        # 5+ edge games (close)
        mask_5plus = df['edge_close'] >= 5.0
        games_5plus = mask_5plus.sum()

        if games_5plus < 10:
            logger.warning(f"Fold {holdout_year}: Only {games_5plus} games at 5+ edge")
            return None

        # Calculate wins/losses for 5+ edge
        df_5plus = df[mask_5plus]

        # Proper ATS calculation
        home_wins = ((df_5plus['predicted_spread'] > df_5plus['vegas_spread']) &
                     (df_5plus['actual_margin'] > df_5plus['vegas_spread'])).sum()
        away_wins = ((df_5plus['predicted_spread'] < df_5plus['vegas_spread']) &
                     (df_5plus['actual_margin'] < df_5plus['vegas_spread'])).sum()
        wins_5plus = home_wins + away_wins

        # Pushes
        pushes = (df_5plus['actual_margin'] == df_5plus['vegas_spread']).sum()
        losses_5plus = games_5plus - wins_5plus - pushes

        # Exclude pushes from win %
        games_decided = wins_5plus + losses_5plus
        win_pct = wins_5plus / games_decided if games_decided > 0 else 0.5

        # Wilson CI
        wilson_lower, wilson_upper = wilson_confidence_interval(wins_5plus, games_decided)

        # ROI
        roi = calculate_roi(wins_5plus, losses_5plus)

        # MAE
        mae = np.abs(df['predicted_spread'] - df['actual_margin']).mean()
        mean_error = (df['predicted_spread'] - df['actual_margin']).mean()

        # Edge buckets
        mask_3to5 = (df['edge_close'] >= 3.0) & (df['edge_close'] < 5.0)
        mask_5to7 = (df['edge_close'] >= 5.0) & (df['edge_close'] < 7.0)
        mask_7plus = df['edge_close'] >= 7.0

        def bucket_wins(mask):
            df_bucket = df[mask]
            hw = ((df_bucket['predicted_spread'] > df_bucket['vegas_spread']) &
                  (df_bucket['actual_margin'] > df_bucket['vegas_spread'])).sum()
            aw = ((df_bucket['predicted_spread'] < df_bucket['vegas_spread']) &
                  (df_bucket['actual_margin'] < df_bucket['vegas_spread'])).sum()
            return hw + aw

        return FoldResult(
            holdout_year=holdout_year,
            games_total=len(df),
            games_5plus=games_decided,
            wins_5plus=wins_5plus,
            losses_5plus=losses_5plus,
            win_pct_5plus=win_pct,
            wilson_lower=wilson_lower,
            wilson_upper=wilson_upper,
            roi_5plus=roi,
            mae=mae,
            mean_error=mean_error,
            games_3to5=mask_3to5.sum(),
            wins_3to5=bucket_wins(mask_3to5),
            games_5to7=mask_5to7.sum(),
            wins_5to7=bucket_wins(mask_5to7),
            games_7plus=mask_7plus.sum(),
            wins_7plus=bucket_wins(mask_7plus),
        )

    except Exception as e:
        logger.error(f"Error in fold {holdout_year}: {e}")
        return None


def evaluate_configuration(
    config_name: str,
    schedule_type: str,
    params: dict,
    schedule: BlendSchedule,
    years: list[int],
    baseline_results: dict,
) -> ConfigResult:
    """Evaluate a single configuration across all LOYO folds."""
    result = ConfigResult(
        config_name=config_name,
        schedule_type=schedule_type,
        params=params,
    )

    fold_results = []

    for holdout_year in years:
        fold_result = run_single_fold(holdout_year, schedule)

        if fold_result is not None:
            fold_results.append(fold_result)

    if len(fold_results) < 3:
        result.rejection_reason = f"Only {len(fold_results)} valid folds (need ≥3)"
        return result

    result.fold_results = fold_results

    # Aggregate metrics
    win_pcts = [f.win_pct_5plus for f in fold_results]
    result.mean_win_pct_5plus = np.mean(win_pcts)
    result.std_win_pct_5plus = np.std(win_pcts)
    result.mean_roi_5plus = np.mean([f.roi_5plus for f in fold_results])
    result.mean_mae = np.mean([f.mae for f in fold_results])
    result.total_games_5plus = sum(f.games_5plus for f in fold_results)
    result.total_wins_5plus = sum(f.wins_5plus for f in fold_results)

    # Check acceptance criteria
    baseline_mean_win_pct = np.mean([baseline_results[y]['win_pct_5plus'] for y in years if y in baseline_results])
    baseline_mean_mae = np.mean([baseline_results[y]['mae'] for y in years if y in baseline_results])

    # 1. 5+ Edge Win % improves by at least +1.0%
    improvement = result.mean_win_pct_5plus - baseline_mean_win_pct
    result.passes_win_pct_threshold = improvement >= 0.01  # 1%

    # 2. Improvement in at least 3 of 4 folds
    folds_improved = sum(
        1 for f in fold_results
        if f.win_pct_5plus > baseline_results.get(f.holdout_year, {}).get('win_pct_5plus', 0.5)
    )
    result.passes_fold_count = folds_improved >= 3

    # 3. Variance across folds ≤ 2%
    result.passes_fold_stability = result.std_win_pct_5plus <= 0.02

    # 4. MAE within +0.5 of baseline
    mae_delta = result.mean_mae - baseline_mean_mae
    result.passes_mae_constraint = mae_delta <= 0.5

    # Overall acceptance
    if not result.passes_win_pct_threshold:
        result.rejection_reason = f"Win % improvement {improvement:.1%} < 1.0%"
    elif not result.passes_fold_count:
        result.rejection_reason = f"Only {folds_improved}/4 folds improved"
    elif not result.passes_fold_stability:
        result.rejection_reason = f"Fold variance {result.std_win_pct_5plus:.1%} > 2.0%"
    elif not result.passes_mae_constraint:
        result.rejection_reason = f"MAE delta {mae_delta:+.2f} > +0.5"
    else:
        result.accepted = True

    return result


def run_baseline_evaluation(years: list[int], end_week: int = 3) -> dict:
    """Run SP+-only baseline for comparison."""
    logger.info("Running SP+-only baseline evaluation...")
    baseline_results = {}

    for year in years:
        logger.info(f"  Baseline fold: holdout={year}")

        try:
            result = run_backtest(
                years=[year],
                start_week=1,
                end_week=end_week,
                use_priors=True,
                use_blended_priors=False,  # SP+-only
            )

            # Extract ATS results DataFrame (has vegas_spread already merged)
            if isinstance(result, dict):
                results_df = result.get("ats_results")
            else:
                results_df = result

            if results_df is None or len(results_df) == 0:
                continue

            df = results_df.copy()
            df = df[df['vegas_spread'].notna()]
            df['edge_close'] = np.abs(df['predicted_spread'] - df['vegas_spread'])

            mask_5plus = df['edge_close'] >= 5.0
            df_5plus = df[mask_5plus]

            home_wins = ((df_5plus['predicted_spread'] > df_5plus['vegas_spread']) &
                         (df_5plus['actual_margin'] > df_5plus['vegas_spread'])).sum()
            away_wins = ((df_5plus['predicted_spread'] < df_5plus['vegas_spread']) &
                         (df_5plus['actual_margin'] < df_5plus['vegas_spread'])).sum()
            wins = home_wins + away_wins

            pushes = (df_5plus['actual_margin'] == df_5plus['vegas_spread']).sum()
            games = len(df_5plus) - pushes

            baseline_results[year] = {
                'games_5plus': games,
                'wins_5plus': wins,
                'win_pct_5plus': wins / games if games > 0 else 0.5,
                'mae': np.abs(df['predicted_spread'] - df['actual_margin']).mean(),
            }

            logger.info(f"    {year}: 5+ Edge {wins}/{games} ({wins/games:.1%}), MAE {baseline_results[year]['mae']:.2f}")

        except Exception as e:
            logger.error(f"  Baseline error for {year}: {e}")

    return baseline_results


def print_fold_results(result: ConfigResult):
    """Print detailed fold results for a configuration."""
    print(f"\n{'='*80}")
    print(f"Configuration: {result.config_name}")
    print(f"Type: {result.schedule_type}")
    print(f"Params: {result.params}")
    print(f"{'='*80}")

    print("\n### Per-Fold Results")
    print(f"{'Year':<6} {'Games':<8} {'Wins':<6} {'Win %':<8} {'Wilson 95% CI':<18} {'ROI':<8} {'MAE':<6}")
    print("-" * 70)

    for fold in result.fold_results:
        print(f"{fold.holdout_year:<6} {fold.games_5plus:<8} {fold.wins_5plus:<6} "
              f"{fold.win_pct_5plus:.1%}    [{fold.wilson_lower:.1%}, {fold.wilson_upper:.1%}]     "
              f"{fold.roi_5plus:+.1f}%   {fold.mae:.2f}")

    print("-" * 70)
    print(f"{'TOTAL':<6} {result.total_games_5plus:<8} {result.total_wins_5plus:<6} "
          f"{result.mean_win_pct_5plus:.1%}    σ={result.std_win_pct_5plus:.1%}              "
          f"{result.mean_roi_5plus:+.1f}%   {result.mean_mae:.2f}")

    print("\n### Edge Buckets")
    print(f"{'Bucket':<10} {'Games':<8} {'Wins':<8} {'Win %':<10}")
    print("-" * 40)

    total_3to5 = sum(f.games_3to5 for f in result.fold_results)
    wins_3to5 = sum(f.wins_3to5 for f in result.fold_results)
    total_5to7 = sum(f.games_5to7 for f in result.fold_results)
    wins_5to7 = sum(f.wins_5to7 for f in result.fold_results)
    total_7plus = sum(f.games_7plus for f in result.fold_results)
    wins_7plus = sum(f.wins_7plus for f in result.fold_results)

    if total_3to5 > 0:
        print(f"{'3-5 pts':<10} {total_3to5:<8} {wins_3to5:<8} {wins_3to5/total_3to5:.1%}")
    if total_5to7 > 0:
        print(f"{'5-7 pts':<10} {total_5to7:<8} {wins_5to7:<8} {wins_5to7/total_5to7:.1%}")
    if total_7plus > 0:
        print(f"{'7+ pts':<10} {total_7plus:<8} {wins_7plus:<8} {wins_7plus/total_7plus:.1%}")

    print("\n### Acceptance Criteria")
    print(f"  Win % threshold (+1.0%): {'✓ PASS' if result.passes_win_pct_threshold else '✗ FAIL'}")
    print(f"  Fold count (≥3 improved): {'✓ PASS' if result.passes_fold_count else '✗ FAIL'}")
    print(f"  Fold stability (σ ≤ 2%): {'✓ PASS' if result.passes_fold_stability else '✗ FAIL'}")
    print(f"  MAE constraint (≤ +0.5): {'✓ PASS' if result.passes_mae_constraint else '✗ FAIL'}")

    if result.accepted:
        print(f"\n  ✓✓✓ ACCEPTED ✓✓✓")
    else:
        print(f"\n  ✗ REJECTED: {result.rejection_reason}")


def main():
    parser = argparse.ArgumentParser(description="Tune blended priors for ATS performance")
    parser.add_argument("--quick", action="store_true", help="Reduced search space for testing")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results for each config")
    parser.add_argument("--output", type=str, default="data/outputs/blended_tuning_ats.json",
                        help="Output file for results")
    args = parser.parse_args()

    years = [2022, 2023, 2024, 2025]

    print("=" * 80)
    print("BLENDED PRIORS ATS-FOCUSED TUNING")
    print("=" * 80)
    print(f"\nObjective: Maximize 5+ Edge Win % vs Close Line")
    print(f"Validation: Leave-One-Year-Out CV (years: {years})")
    print(f"Constraint: MAE within +0.5 of SP+-only baseline")
    print(f"\nAcceptance Criteria:")
    print(f"  - 5+ Edge Win % improves by ≥1.0%")
    print(f"  - Improvement in ≥3 of 4 folds")
    print(f"  - Fold variance ≤2%")
    print(f"  - MAE delta ≤ +0.5")
    print()

    # Run baseline
    baseline_results = run_baseline_evaluation(years)

    baseline_mean = np.mean([baseline_results[y]['win_pct_5plus'] for y in years if y in baseline_results])
    baseline_mae = np.mean([baseline_results[y]['mae'] for y in years if y in baseline_results])
    baseline_games = sum(baseline_results[y]['games_5plus'] for y in years if y in baseline_results)
    baseline_wins = sum(baseline_results[y]['wins_5plus'] for y in years if y in baseline_results)

    print(f"\n### Baseline (SP+-only)")
    print(f"  5+ Edge: {baseline_wins}/{baseline_games} ({baseline_mean:.1%})")
    print(f"  MAE: {baseline_mae:.2f}")
    print()

    # Generate and test configurations
    configs = generate_search_space(quick=args.quick)

    all_results = []
    accepted_configs = []

    for i, (name, schedule_type, params, schedule) in enumerate(configs):
        logger.info(f"Testing config {i+1}/{len(configs)}: {name}")

        result = evaluate_configuration(
            name, schedule_type, params, schedule, years, baseline_results
        )
        all_results.append(result)

        if result.accepted:
            accepted_configs.append(result)
            logger.info(f"  ✓ ACCEPTED: {result.mean_win_pct_5plus:.1%} (baseline: {baseline_mean:.1%})")
        else:
            logger.info(f"  ✗ Rejected: {result.rejection_reason}")

        if args.verbose and len(result.fold_results) >= 3:
            print_fold_results(result)

    # Summary
    print("\n" + "=" * 80)
    print("TUNING SUMMARY")
    print("=" * 80)

    print(f"\nConfigurations tested: {len(configs)}")
    print(f"Configurations accepted: {len(accepted_configs)}")

    if accepted_configs:
        print("\n### Accepted Configurations (sorted by Win %)")
        accepted_sorted = sorted(accepted_configs, key=lambda x: x.mean_win_pct_5plus, reverse=True)

        print(f"\n{'Rank':<5} {'Config':<45} {'Win %':<8} {'vs Base':<10} {'ROI':<8} {'MAE':<6}")
        print("-" * 90)

        for rank, result in enumerate(accepted_sorted[:10], 1):
            delta = result.mean_win_pct_5plus - baseline_mean
            print(f"{rank:<5} {result.config_name:<45} {result.mean_win_pct_5plus:.1%}    "
                  f"{delta:+.1%}     {result.mean_roi_5plus:+.1f}%   {result.mean_mae:.2f}")

        # Print details for top config
        best = accepted_sorted[0]
        print(f"\n### Best Configuration: {best.config_name}")
        print_fold_results(best)

    else:
        print("\n### NO CONFIGURATIONS ACCEPTED")
        print("\nBlended priors do not improve 5+ Edge ATS performance.")
        print("Recommendation: Keep SP+-only priors as default.")

        # Show top performers anyway for analysis
        all_with_folds = [r for r in all_results if len(r.fold_results) >= 3]
        if all_with_folds:
            top_performers = sorted(all_with_folds, key=lambda x: x.mean_win_pct_5plus, reverse=True)[:5]

            print("\n### Top 5 Performers (did not meet acceptance criteria)")
            print(f"\n{'Config':<45} {'Win %':<8} {'vs Base':<10} {'Rejection':<30}")
            print("-" * 100)

            for result in top_performers:
                delta = result.mean_win_pct_5plus - baseline_mean
                print(f"{result.config_name:<45} {result.mean_win_pct_5plus:.1%}    "
                      f"{delta:+.1%}     {result.rejection_reason[:30]}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "_meta": {
            "generated": datetime.now().isoformat(),
            "objective": "5+ Edge Win % vs Close Line",
            "years": years,
            "baseline_win_pct": baseline_mean,
            "baseline_mae": baseline_mae,
            "configs_tested": len(configs),
            "configs_accepted": len(accepted_configs),
        },
        "baseline": baseline_results,
        "accepted_configs": [
            {
                "name": r.config_name,
                "type": r.schedule_type,
                "params": r.params,
                "mean_win_pct": r.mean_win_pct_5plus,
                "std_win_pct": r.std_win_pct_5plus,
                "mean_roi": r.mean_roi_5plus,
                "mean_mae": r.mean_mae,
                "fold_results": [
                    {
                        "year": f.holdout_year,
                        "games": f.games_5plus,
                        "wins": f.wins_5plus,
                        "win_pct": f.win_pct_5plus,
                        "wilson_ci": [f.wilson_lower, f.wilson_upper],
                        "roi": f.roi_5plus,
                        "mae": f.mae,
                    }
                    for f in r.fold_results
                ],
            }
            for r in accepted_configs
        ],
        "recommendation": "ACCEPT" if accepted_configs else "REJECT - keep SP+-only",
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0 if accepted_configs else 1


if __name__ == "__main__":
    sys.exit(main())
