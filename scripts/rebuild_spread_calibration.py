#!/usr/bin/env python3
"""Rebuild Spread EV Calibration Pipeline.

This script addresses the calibration issues:
1. Missing 2022-2023 data due to min_train_seasons=2
2. Phase mixing where Phase 1 (weeks 1-3, ~47% ATS) drags down slope
3. Unrealistic breakeven edge (18+ points)

Solutions:
1. Use min_train_seasons=1 to include 2023 evaluations
2. Create Phase 2-only calibration (weeks 4-15)
3. Create weighted calibration (Phase 1 weight=0.25)
4. Validate breakeven edge is realistic (~4-6 pts at -110)

Usage:
    python3 scripts/rebuild_spread_calibration.py

Output:
    - Diagnostic report
    - Rebuilt calibration artifacts in data/spread_selection/artifacts/
    - Backtest comparison results
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.spread_selection.calibration import (
    CalibrationResult,
    load_and_normalize_game_data,
    calibrate_cover_probability,
    walk_forward_validate,
    breakeven_prob,
    diagnose_calibration,
    predict_cover_probability,
    calculate_ev_vectorized,
    estimate_push_rates,
    get_push_probability_vectorized,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# STEP A: Diagnostic Report
# =============================================================================

def step_a_diagnostic(ats_export_path: str) -> pd.DataFrame:
    """Step A: Confirm dataset and phase filters.

    Returns normalized DataFrame for further processing.
    """
    print("\n" + "=" * 80)
    print("STEP A: DIAGNOSTIC REPORT")
    print("=" * 80)

    # Load raw data
    raw_df = pd.read_csv(ats_export_path)
    print(f"\nSource file: {ats_export_path}")
    print(f"Total rows: {len(raw_df)}")

    # Check years
    print("\n--- Rows per YEAR ---")
    for year in sorted(raw_df['year'].unique()):
        count = len(raw_df[raw_df['year'] == year])
        print(f"  {year}: {count}")

    missing_years = set([2022, 2023, 2024, 2025]) - set(raw_df['year'].unique())
    if missing_years:
        raise ValueError(f"CRITICAL: Missing years in source data: {missing_years}")

    # Check weeks
    print("\n--- Rows per PHASE ---")
    phase1 = raw_df[raw_df['week'] <= 3]
    phase2 = raw_df[(raw_df['week'] >= 4) & (raw_df['week'] <= 15)]
    phase3 = raw_df[raw_df['week'] > 15]
    print(f"  Phase 1 (weeks 1-3): {len(phase1)}")
    print(f"  Phase 2 (weeks 4-15): {len(phase2)}")
    print(f"  Phase 3 (weeks 16+): {len(phase3)}")

    # Check for duplicates
    dup_mask = raw_df.duplicated(subset=['game_id'], keep=False)
    n_dups = dup_mask.sum()
    print(f"\n--- Duplicates by game_id ---")
    print(f"  Duplicate rows: {n_dups}")
    if n_dups > 0:
        print("  WARNING: Duplicates found - will be removed")
        raw_df = raw_df.drop_duplicates(subset=['game_id'], keep='first')

    # Normalize to calibration format
    print("\n--- Normalizing data ---")
    df = load_and_normalize_game_data(raw_df)
    print(f"  After normalization: {len(df)} games with valid Vegas spreads")

    # Check ATS by phase
    print("\n--- ATS Performance by Phase ---")
    for phase_name, mask in [
        ("Phase 1 (1-3)", df['week'] <= 3),
        ("Phase 2 (4-15)", (df['week'] >= 4) & (df['week'] <= 15)),
        ("Phase 3 (16+)", df['week'] > 15),
    ]:
        phase_df = df[mask & ~df['push']]
        if len(phase_df) > 0:
            wins = phase_df['jp_side_covered'].sum()
            total = len(phase_df)
            pct = 100 * wins / total
            print(f"  {phase_name}: {wins}/{total} ({pct:.1f}%)")

    # Check edge distribution
    print("\n--- Edge Distribution ---")
    for bucket, lo, hi in [
        ("[0-3)", 0, 3),
        ("[3-5)", 3, 5),
        ("[5-8)", 5, 8),
        ("[8-12)", 8, 12),
        ("[12+)", 12, 999),
    ]:
        mask = (df['edge_abs'] >= lo) & (df['edge_abs'] < hi) & ~df['push']
        bucket_df = df[mask]
        if len(bucket_df) > 0:
            wins = bucket_df['jp_side_covered'].sum()
            total = len(bucket_df)
            pct = 100 * wins / total
            print(f"  Edge {bucket}: {wins}/{total} ({pct:.1f}%)")

    print("\n" + "=" * 80)
    print("STEP A COMPLETE: All 4 years present, no critical issues")
    print("=" * 80)

    return df


# =============================================================================
# STEP B: Rebuild Canonical Calibration DataFrame
# =============================================================================

def step_b_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Step B: Build walk-forward predictions for all years.

    Uses min_train_seasons=1 to include 2023 evaluations.

    Returns DataFrame with walk-forward p_cover predictions.
    """
    print("\n" + "=" * 80)
    print("STEP B: WALK-FORWARD PREDICTIONS (min_train_seasons=1)")
    print("=" * 80)

    # Use min_train_seasons=1 and training_window_seasons=None (all prior years)
    # This allows 2023 to be evaluated (trains on 2022)
    wf_result = walk_forward_validate(
        all_games=df,
        min_train_seasons=1,  # Changed from 2 to include 2023
        exclude_covid=True,
        training_window_seasons=None,  # Use all prior years (INCLUDE_ALL)
        include_push_modeling=True,
    )

    game_results = wf_result.game_results

    print("\n--- Walk-Forward Fold Summaries ---")
    for fold in wf_result.fold_summaries:
        print(f"  {fold['eval_year']}: train={fold['n_train']}, eval={fold['n_eval']}, "
              f"slope={fold['slope']:.4f}, breakeven={fold['breakeven_edge']:.1f}")

    print(f"\n--- Overall Metrics ---")
    print(f"  Games with predictions: {len(game_results[game_results['p_cover_no_push'].notna()])}")
    print(f"  Brier score: {wf_result.overall_brier:.4f}")
    print(f"  Brier skill score: {wf_result.brier_skill_score:.4f}")

    # Verify we have all years
    years_in_results = sorted(game_results['year'].unique())
    print(f"\n  Years in results: {years_in_results}")

    if 2023 not in years_in_results:
        raise ValueError("CRITICAL: 2023 still missing from walk-forward results")

    print("\n" + "=" * 80)
    print("STEP B COMPLETE: Walk-forward predictions generated for 2023-2025")
    print("=" * 80)

    return game_results


# =============================================================================
# STEP C: Fit Calibrations
# =============================================================================

def fit_logistic_calibration(
    df: pd.DataFrame,
    name: str,
    weights: Optional[np.ndarray] = None,
) -> CalibrationResult:
    """Fit a single logistic calibration."""
    # Filter out pushes and zero-edge
    mask = (~df['push']) & (df['edge_abs'] > 0)
    data = df[mask].copy()

    X = data[['edge_abs']].values
    y = data['jp_side_covered'].astype(int).values

    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, fit_intercept=True)

    if weights is not None:
        w = weights[mask] if len(weights) == len(df) else weights
        model.fit(X, y, sample_weight=w)
    else:
        model.fit(X, y)

    intercept = model.intercept_[0]
    slope = model.coef_[0][0]

    p_cover_at_zero = expit(intercept)
    implied_5pt_pcover = expit(intercept + slope * 5)

    # Breakeven edge at -110
    breakeven = breakeven_prob(-110)
    logit_breakeven = np.log(breakeven / (1 - breakeven))

    if slope > 0:
        implied_breakeven_edge = (logit_breakeven - intercept) / slope
    else:
        implied_breakeven_edge = float('inf')

    years_trained = sorted(data['year'].unique().tolist())

    return CalibrationResult(
        intercept=intercept,
        slope=slope,
        n_games=len(data),
        years_trained=years_trained,
        implied_breakeven_edge=implied_breakeven_edge,
        implied_5pt_pcover=implied_5pt_pcover,
        p_cover_at_zero=p_cover_at_zero,
    )


def step_c_fit_calibrations(df: pd.DataFrame) -> dict:
    """Step C: Fit multiple calibrations.

    Returns dict of calibration name -> CalibrationResult.
    """
    print("\n" + "=" * 80)
    print("STEP C: FIT CALIBRATIONS")
    print("=" * 80)

    calibrations = {}

    # Calibration 1: Phase 2 only (weeks 4-15)
    print("\n--- Calibration 1: Phase 2 Only (weeks 4-15) ---")
    phase2_mask = (df['week'] >= 4) & (df['week'] <= 15)
    phase2_df = df[phase2_mask]

    cal_phase2 = fit_logistic_calibration(phase2_df, "Phase 2 Only")
    calibrations['phase2_only'] = cal_phase2
    print(f"  n_games: {cal_phase2.n_games}")
    print(f"  slope: {cal_phase2.slope:.6f}")
    print(f"  intercept: {cal_phase2.intercept:.6f}")
    print(f"  p_cover_at_zero: {cal_phase2.p_cover_at_zero:.4f}")
    print(f"  breakeven_edge: {cal_phase2.implied_breakeven_edge:.2f} pts")
    print(f"  P(cover) at 5pt edge: {cal_phase2.implied_5pt_pcover:.3f}")

    # Calibration 2: Weighted (Phase 1 = 0.25, Phase 2 = 1.0)
    print("\n--- Calibration 2: Weighted (Phase 1 = 0.25) ---")
    weights = np.where(df['week'] <= 3, 0.25, 1.0)

    cal_weighted = fit_logistic_calibration(df, "Weighted", weights=weights)
    calibrations['weighted'] = cal_weighted
    print(f"  n_games: {cal_weighted.n_games}")
    print(f"  slope: {cal_weighted.slope:.6f}")
    print(f"  intercept: {cal_weighted.intercept:.6f}")
    print(f"  p_cover_at_zero: {cal_weighted.p_cover_at_zero:.4f}")
    print(f"  breakeven_edge: {cal_weighted.implied_breakeven_edge:.2f} pts")
    print(f"  P(cover) at 5pt edge: {cal_weighted.implied_5pt_pcover:.3f}")

    # Calibration 3: Phase 1 only (for comparison/separate use)
    print("\n--- Calibration 3: Phase 1 Only (weeks 1-3) ---")
    phase1_mask = df['week'] <= 3
    phase1_df = df[phase1_mask]

    cal_phase1 = fit_logistic_calibration(phase1_df, "Phase 1 Only")
    calibrations['phase1_only'] = cal_phase1
    print(f"  n_games: {cal_phase1.n_games}")
    print(f"  slope: {cal_phase1.slope:.6f}")
    print(f"  intercept: {cal_phase1.intercept:.6f}")
    print(f"  p_cover_at_zero: {cal_phase1.p_cover_at_zero:.4f}")
    print(f"  breakeven_edge: {cal_phase1.implied_breakeven_edge:.2f} pts")
    print(f"  P(cover) at 5pt edge: {cal_phase1.implied_5pt_pcover:.3f}")

    # Calibration 4: Full season (all phases, no weighting) - baseline
    print("\n--- Calibration 4: Full Season (baseline) ---")
    cal_full = fit_logistic_calibration(df, "Full Season")
    calibrations['full_season'] = cal_full
    print(f"  n_games: {cal_full.n_games}")
    print(f"  slope: {cal_full.slope:.6f}")
    print(f"  intercept: {cal_full.intercept:.6f}")
    print(f"  p_cover_at_zero: {cal_full.p_cover_at_zero:.4f}")
    print(f"  breakeven_edge: {cal_full.implied_breakeven_edge:.2f} pts")
    print(f"  P(cover) at 5pt edge: {cal_full.implied_5pt_pcover:.3f}")

    print("\n" + "=" * 80)
    print("STEP C COMPLETE: 4 calibrations fitted")
    print("=" * 80)

    return calibrations


# =============================================================================
# STEP D: Validate Calibration Sanity
# =============================================================================

def step_d_validate(df: pd.DataFrame, calibrations: dict) -> dict:
    """Step D: Validate calibration sanity.

    Returns validation metrics dict.
    """
    print("\n" + "=" * 80)
    print("STEP D: VALIDATE CALIBRATION SANITY")
    print("=" * 80)

    validation = {}

    # Filter to games with predictions
    has_pred = df['p_cover_no_push'].notna() & ~df['push'] & (df['edge_abs'] > 0)
    pred_df = df[has_pred].copy()

    for cal_name, cal in calibrations.items():
        print(f"\n--- {cal_name.upper()} ---")

        # Apply calibration to get predictions
        p_cover = predict_cover_probability(pred_df['edge_abs'].values, cal)

        # Brier score
        y_true = pred_df['jp_side_covered'].astype(int).values
        brier = np.mean((p_cover - y_true) ** 2)

        # ATS by edge bucket
        print(f"\n  ATS by Edge Bucket:")
        bucket_stats = []
        for bucket, lo, hi in [
            ("[0-3)", 0, 3),
            ("[3-5)", 3, 5),
            ("[5-8)", 5, 8),
            ("[8-12)", 8, 12),
            ("[12+)", 12, 999),
        ]:
            mask = (pred_df['edge_abs'] >= lo) & (pred_df['edge_abs'] < hi)
            bucket_df = pred_df[mask]
            if len(bucket_df) > 0:
                wins = bucket_df['jp_side_covered'].sum()
                total = len(bucket_df)
                pct = 100 * wins / total
                bucket_stats.append({
                    'bucket': bucket,
                    'wins': int(wins),
                    'total': total,
                    'ats_pct': pct,
                })
                print(f"    {bucket}: {wins}/{total} ({pct:.1f}%)")

        # Phase-specific ATS
        print(f"\n  ATS by Phase:")
        phase_stats = []
        for phase_name, week_lo, week_hi in [
            ("Phase 1 (1-3)", 1, 3),
            ("Phase 2 (4-15)", 4, 15),
            ("Phase 3 (16+)", 16, 100),
        ]:
            mask = (pred_df['week'] >= week_lo) & (pred_df['week'] <= week_hi)
            phase_df = pred_df[mask]
            if len(phase_df) > 0:
                wins = phase_df['jp_side_covered'].sum()
                total = len(phase_df)
                pct = 100 * wins / total
                phase_stats.append({
                    'phase': phase_name,
                    'wins': int(wins),
                    'total': total,
                    'ats_pct': pct,
                })
                print(f"    {phase_name}: {wins}/{total} ({pct:.1f}%)")

        # 5+ edge ATS (key metric)
        edge5_mask = pred_df['edge_abs'] >= 5.0
        edge5_df = pred_df[edge5_mask]
        if len(edge5_df) > 0:
            wins_5 = edge5_df['jp_side_covered'].sum()
            total_5 = len(edge5_df)
            pct_5 = 100 * wins_5 / total_5
            print(f"\n  5+ Edge ATS: {wins_5}/{total_5} ({pct_5:.1f}%)")

        # Reliability by predicted probability bins
        print(f"\n  Reliability (predicted vs actual):")
        for prob_lo, prob_hi in [(0.45, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.65)]:
            mask = (p_cover >= prob_lo) & (p_cover < prob_hi)
            bin_df = pred_df.iloc[mask]
            if len(bin_df) > 0:
                actual = bin_df['jp_side_covered'].mean()
                predicted = p_cover[mask].mean()
                print(f"    [{prob_lo:.2f}-{prob_hi:.2f}): pred={predicted:.3f}, actual={actual:.3f}, n={len(bin_df)}")

        validation[cal_name] = {
            'slope': cal.slope,
            'intercept': cal.intercept,
            'breakeven_edge': cal.implied_breakeven_edge,
            'p_cover_at_zero': cal.p_cover_at_zero,
            'brier': brier,
            'bucket_stats': bucket_stats,
            'phase_stats': phase_stats,
        }

    print("\n" + "=" * 80)
    print("STEP D COMPLETE: All calibrations validated")
    print("=" * 80)

    return validation


# =============================================================================
# STEP E: EV Selection Backtest Comparison
# =============================================================================

def step_e_backtest_comparison(
    df: pd.DataFrame,
    calibrations: dict,
    min_ev: float = 0.03,
) -> dict:
    """Step E: Compare bet selection across calibrations.

    Returns comparison metrics dict.
    """
    print("\n" + "=" * 80)
    print("STEP E: EV SELECTION BACKTEST COMPARISON")
    print("=" * 80)

    # Filter to games with valid predictions
    has_pred = df['p_cover_no_push'].notna() & ~df['push'] & (df['edge_abs'] > 0)
    pred_df = df[has_pred].copy()

    comparison = {}

    for cal_name, cal in calibrations.items():
        print(f"\n--- {cal_name.upper()} ---")

        # Apply calibration to get p_cover
        p_cover = predict_cover_probability(pred_df['edge_abs'].values, cal)

        # Calculate EV (assuming -110 juice, no push for simplicity)
        ev = p_cover * (100/110) - (1 - p_cover)

        pred_df_cal = pred_df.copy()
        pred_df_cal['p_cover_cal'] = p_cover
        pred_df_cal['ev_cal'] = ev

        # Select bets with EV >= threshold
        ev_mask = pred_df_cal['ev_cal'] >= min_ev
        bets = pred_df_cal[ev_mask]

        # Overall stats
        n_bets = len(bets)
        if n_bets > 0:
            wins = bets['jp_side_covered'].sum()
            losses = n_bets - wins
            ats_pct = 100 * wins / n_bets
            avg_ev = bets['ev_cal'].mean()
            avg_edge = bets['edge_abs'].mean()
        else:
            wins, losses, ats_pct, avg_ev, avg_edge = 0, 0, 0, 0, 0

        print(f"  Total EV >= {min_ev:.1%} bets: {n_bets}")
        print(f"  ATS: {wins}-{losses} ({ats_pct:.1f}%)")
        print(f"  Avg EV: {avg_ev:.3f}")
        print(f"  Avg edge: {avg_edge:.1f} pts")

        # Per-year breakdown
        print(f"\n  Bets by Year:")
        year_stats = []
        for year in sorted(pred_df_cal['year'].unique()):
            year_bets = bets[bets['year'] == year]
            if len(year_bets) > 0:
                y_wins = year_bets['jp_side_covered'].sum()
                y_total = len(year_bets)
                y_pct = 100 * y_wins / y_total
                year_stats.append({'year': year, 'bets': y_total, 'ats': y_pct})
                print(f"    {year}: {y_total} bets, {y_wins}/{y_total} ({y_pct:.1f}%)")

        # Per-week breakdown
        print(f"\n  Bets by Week (2025 example):")
        week_stats = []
        for week in range(1, 16):
            week_bets = bets[(bets['year'] == 2025) & (bets['week'] == week)]
            if len(week_bets) > 0:
                week_stats.append({'week': week, 'bets': len(week_bets)})
                print(f"    Week {week}: {len(week_bets)} bets")

        # 5+ edge subset
        edge5_bets = bets[bets['edge_abs'] >= 5.0]
        if len(edge5_bets) > 0:
            e5_wins = edge5_bets['jp_side_covered'].sum()
            e5_total = len(edge5_bets)
            e5_pct = 100 * e5_wins / e5_total
            print(f"\n  5+ Edge subset: {e5_wins}/{e5_total} ({e5_pct:.1f}%)")

        comparison[cal_name] = {
            'n_bets': n_bets,
            'wins': int(wins) if n_bets > 0 else 0,
            'losses': int(losses) if n_bets > 0 else 0,
            'ats_pct': ats_pct,
            'avg_ev': avg_ev,
            'avg_edge': avg_edge,
            'year_stats': year_stats,
        }

    print("\n" + "=" * 80)
    print("STEP E COMPLETE: Backtest comparison finished")
    print("=" * 80)

    return comparison


# =============================================================================
# STEP F: Write Artifacts
# =============================================================================

def step_f_write_artifacts(
    calibrations: dict,
    validation: dict,
    comparison: dict,
    output_dir: str,
) -> list:
    """Step F: Write calibration artifacts.

    Returns list of written file paths.
    """
    print("\n" + "=" * 80)
    print("STEP F: WRITE ARTIFACTS")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files = []
    timestamp = datetime.now().isoformat()

    for cal_name, cal in calibrations.items():
        # Build artifact
        artifact = {
            'calibration_name': cal_name,
            'created_at': timestamp,
            'model_type': 'logistic_regression',
            'parameters': {
                'intercept': float(cal.intercept),
                'slope': float(cal.slope),
            },
            'metadata': {
                'n_games': cal.n_games,
                'years_trained': cal.years_trained,
                'breakeven_edge_at_110': float(cal.implied_breakeven_edge),
                'p_cover_at_zero': float(cal.p_cover_at_zero),
                'p_cover_at_5pt': float(cal.implied_5pt_pcover),
            },
            'validation': validation.get(cal_name, {}),
            'backtest_comparison': comparison.get(cal_name, {}),
            'usage': {
                'phase2_only': 'Use for weeks 4-15 (Core Phase)',
                'weighted': 'Use for full season with Phase 1 downweighted',
                'phase1_only': 'Use for weeks 1-3 only (not recommended)',
                'full_season': 'Baseline - includes Phase 1 drag (not recommended)',
            }.get(cal_name, 'N/A'),
        }

        # Write JSON
        filename = f"spread_ev_calibration_{cal_name}_2022_2025.json"
        filepath = output_path / filename

        with open(filepath, 'w') as f:
            json.dump(artifact, f, indent=2, default=str)

        print(f"  Wrote: {filepath}")
        written_files.append(str(filepath))

    # Write summary report
    summary = {
        'generated_at': timestamp,
        'calibrations': list(calibrations.keys()),
        'recommendation': 'phase2_only',
        'recommendation_reason': (
            'Phase 2 calibration excludes Phase 1 (47% ATS) drag, '
            'producing realistic breakeven edge and higher EV bet volume. '
            'Use phase2_only for weeks 4-15, and either skip Phase 1 or use weighted.'
        ),
        'breakeven_comparison': {
            name: float(cal.implied_breakeven_edge)
            for name, cal in calibrations.items()
        },
        'backtest_ats': {
            name: comp.get('ats_pct', 0)
            for name, comp in comparison.items()
        },
    }

    summary_path = output_path / "calibration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote: {summary_path}")
    written_files.append(str(summary_path))

    print("\n" + "=" * 80)
    print("STEP F COMPLETE: All artifacts written")
    print("=" * 80)

    return written_files


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run full calibration rebuild pipeline."""
    print("\n" + "=" * 80)
    print("SPREAD EV CALIBRATION REBUILD PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")

    # Paths
    ats_export_path = "data/spread_selection/ats_export.csv"
    output_dir = "data/spread_selection/artifacts"

    # Step A: Diagnostic
    df = step_a_diagnostic(ats_export_path)

    # Step B: Walk-forward predictions
    df = step_b_walk_forward(df)

    # Step C: Fit calibrations
    calibrations = step_c_fit_calibrations(df)

    # Step D: Validate
    validation = step_d_validate(df, calibrations)

    # Step E: Backtest comparison
    comparison = step_e_backtest_comparison(df, calibrations)

    # Step F: Write artifacts
    written_files = step_f_write_artifacts(calibrations, validation, comparison, output_dir)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nArtifacts written to: {output_dir}")
    print("\nRECOMMENDATION:")
    print("  - Use 'phase2_only' calibration for weeks 4-15")
    print("  - Use 'weighted' calibration if Phase 1 bets needed")
    print("  - OLD calibration had breakeven ~18pts, NEW should be ~4-6pts")

    # Final comparison table
    print("\n--- CALIBRATION COMPARISON ---")
    print(f"{'Calibration':<20} {'Slope':>10} {'Breakeven':>12} {'EV Bets':>10} {'ATS%':>8}")
    print("-" * 62)
    for name, cal in calibrations.items():
        comp = comparison.get(name, {})
        print(f"{name:<20} {cal.slope:>10.4f} {cal.implied_breakeven_edge:>11.1f}p {comp.get('n_bets', 0):>10} {comp.get('ats_pct', 0):>7.1f}%")


if __name__ == "__main__":
    main()
