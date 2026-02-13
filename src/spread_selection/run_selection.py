#!/usr/bin/env python3
"""CLI for calibrated spread betting selection validation.

Usage:
    # Validate with default ROLLING_2 mode
    python -m src.spread_selection.run_selection validate --csv data/spread_selection/ats_export.csv

    # Validate with specific training window
    python -m src.spread_selection.run_selection validate --training-window-seasons 3

    # Compare ROLLING_2 (primary) vs INCLUDE_ALL (ultra) modes
    python -m src.spread_selection.run_selection validate --compare-modes

    # Predict upcoming games (emit both primary and ultra lists)
    python -m src.spread_selection.run_selection predict --emit-modes primary,ultra

    # Sensitivity analysis (exclude 2022)
    python -m src.spread_selection.run_selection sensitivity --csv data/spread_selection/ats_export.csv
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.spread_selection.calibration import (
    CalibrationResult,
    PushRates,
    KEY_TICKS,
    load_and_normalize_game_data,
    calibrate_cover_probability,
    predict_cover_probability,
    walk_forward_validate,
    estimate_push_rates,
    get_push_probability_vectorized,
    breakeven_prob,
    calculate_ev,
    calculate_ev_vectorized,
    diagnose_calibration,
    diagnose_fold_stability,
    stratified_diagnostics,
    print_stratified_diagnostics,
    get_calibration_label,
    CALIBRATION_MODES,
    DEFAULT_CALIBRATION_MODE,
    DEFAULT_TRAINING_WINDOW_SEASONS,
)
from src.spread_selection.policies.phase1_sp_gate import (
    Phase1SPGateConfig,
    evaluate_single_game,
    fetch_sp_spreads_vegas,
    merge_gate_results_to_df,
)

logger = logging.getLogger(__name__)


def load_backtest_data(
    start_year: int = 2022,
    end_year: int = 2025,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load backtest data for calibration.

    Args:
        start_year: First year to include
        end_year: Last year to include
        csv_path: Path to exported ATS CSV (if None, runs backtest)

    Returns:
        DataFrame with required columns

    Note:
        To generate the ATS CSV, run:
        python3 scripts/backtest.py --export-ats data/spread_selection/ats_export.csv
    """
    if csv_path and Path(csv_path).exists():
        logger.info(f"Loading backtest data from {csv_path}")
        df = pd.read_csv(csv_path)

        # Validate required columns
        required = ["game_id", "year", "week", "home_team", "away_team",
                    "predicted_spread", "actual_margin"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # Check for Vegas spread columns
        if "spread_close" not in df.columns and "spread_open" not in df.columns:
            raise ValueError(
                "CSV must have spread_close or spread_open. "
                "Use --export-ats flag when running backtest."
            )
    else:
        # Run backtest to generate data
        logger.info(f"Running backtest for years {start_year}-{end_year}...")
        years = list(range(start_year, end_year + 1))
        df = run_backtest_for_calibration(years)

    # Filter to year range
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    logger.info(f"Loaded {len(df)} games from {start_year}-{end_year}")
    return df


def run_backtest_for_calibration(years: list[int]) -> pd.DataFrame:
    """Run backtest and return ATS results.

    This is a simplified backtest runner that extracts the data we need.
    """
    import subprocess
    import tempfile

    # Use subprocess to run backtest with export flag
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        export_path = f.name

    cmd = [
        "python3", "scripts/backtest.py",
        "--years", *[str(y) for y in years],
        "--export-ats", export_path,
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Backtest failed: {result.stderr}")
        raise RuntimeError(f"Backtest failed: {result.stderr}")

    # Load the exported CSV
    df = pd.read_csv(export_path)

    # Clean up
    Path(export_path).unlink(missing_ok=True)

    return df


def print_sample_games(df: pd.DataFrame, n: int = 10) -> None:
    """Print sample games for manual sign convention verification.

    Args:
        df: Normalized DataFrame
        n: Number of games to print
    """
    print("\n" + "=" * 80)
    print("SAMPLE GAMES FOR SIGN CONVENTION VERIFICATION")
    print("=" * 80)

    # Sample diverse games (different edge directions, cover outcomes)
    sample = df.sample(min(n, len(df)), random_state=42)

    for _, row in sample.iterrows():
        print(f"\n{row['away_team']} @ {row['home_team']} (Week {row['week']}, {row['year']})")
        print(f"  JP+ spread (Vegas conv): {row['jp_spread']:+.1f}")
        print(f"  Vegas spread:            {row['vegas_spread']:+.1f}")
        print(f"  Edge (JP - Vegas):       {row['edge_pts']:+.1f} (|edge|={row['edge_abs']:.1f})")
        print(f"  JP+ favored side:        {row['jp_favored_side']}")
        print(f"  Actual margin:           {row['actual_margin']:+.1f}")
        print(f"  Cover margin:            {row['cover_margin']:+.1f}")
        print(f"  Home covered:            {row['home_covered']}")
        print(f"  JP+ side covered:        {row['jp_side_covered']}")
        if "push" in row and row["push"]:
            print(f"  PUSH")

    print("\n" + "=" * 80)


def compare_strategies(games_with_pcov: pd.DataFrame) -> dict:
    """Compare selection methods on walk-forward results.

    Strategies:
        OLD: bet when edge_abs >= 5.0 (assume -110)
        NEW_EV3: bet when EV >= 0.03
        NEW_EV5: bet when EV >= 0.05
        NEW_EV8: bet when EV >= 0.08

    Args:
        games_with_pcov: DataFrame with p_cover_no_push, edge_abs, jp_side_covered

    Returns:
        Dictionary with strategy comparison results
    """
    # Filter to games with predictions (edge > 0, not push)
    df = games_with_pcov[
        games_with_pcov["p_cover_no_push"].notna() & ~games_with_pcov["push"]
    ].copy()

    # Use existing EV column if valid (preserves push-aware EV from walk_forward_validate)
    # Only recompute if ev column is missing or all-null
    if "ev" not in df.columns or df["ev"].isna().all():
        df["ev"] = calculate_ev_vectorized(df["p_cover_no_push"].values)

    strategies = {
        "OLD_5pt": {"filter": lambda x: x["edge_abs"] >= 5.0},
        "NEW_EV3": {"filter": lambda x: x["ev"] >= 0.03},
        "NEW_EV5": {"filter": lambda x: x["ev"] >= 0.05},
        "NEW_EV8": {"filter": lambda x: x["ev"] >= 0.08},
    }

    results = {}
    breakeven = breakeven_prob(-110)

    for name, config in strategies.items():
        mask = config["filter"](df)
        subset = df[mask]

        n_bets = len(subset)
        if n_bets == 0:
            results[name] = {"n_bets": 0, "error": "No bets selected"}
            continue

        wins = subset["jp_side_covered"].sum()
        losses = n_bets - wins  # No pushes in this filtered set

        cover_rate = wins / n_bets
        roi = (cover_rate * (100 / 110) - (1 - cover_rate)) * 100  # ROI as percentage

        avg_p_cover = subset["p_cover_no_push"].mean()
        avg_edge = subset["edge_abs"].mean()
        avg_ev = subset["ev"].mean()

        # Per-year breakdown
        yearly = []
        for year in sorted(subset["year"].unique()):
            year_df = subset[subset["year"] == year]
            year_wins = year_df["jp_side_covered"].sum()
            year_n = len(year_df)
            yearly.append({
                "year": year,
                "n": year_n,
                "wins": int(year_wins),
                "losses": year_n - int(year_wins),
                "cover_rate": year_wins / year_n,
            })

        results[name] = {
            "n_bets": n_bets,
            "wins": int(wins),
            "losses": int(losses),
            "cover_rate": cover_rate,
            "roi_pct": roi,
            "avg_p_cover": avg_p_cover,
            "avg_edge": avg_edge,
            "avg_ev": avg_ev,
            "yearly": yearly,
        }

    # Overlap analysis between OLD and NEW strategies
    old_mask = df["edge_abs"] >= 5.0
    new_ev3_mask = df["ev"] >= 0.03
    new_ev5_mask = df["ev"] >= 0.05

    overlap_old_ev3 = (old_mask & new_ev3_mask).sum()
    overlap_old_ev5 = (old_mask & new_ev5_mask).sum()

    # Games unique to each strategy
    old_only = (old_mask & ~new_ev3_mask).sum()
    ev3_only = (~old_mask & new_ev3_mask).sum()

    # Performance on overlap vs unique sets
    overlap_set = df[old_mask & new_ev3_mask]
    old_only_set = df[old_mask & ~new_ev3_mask]
    ev3_only_set = df[~old_mask & new_ev3_mask]

    overlap_metrics = {}
    for name, subset in [("overlap", overlap_set), ("old_only", old_only_set), ("ev3_only", ev3_only_set)]:
        if len(subset) > 0:
            wins = subset["jp_side_covered"].sum()
            n = len(subset)
            cover_rate = wins / n
            roi = (cover_rate * (100 / 110) - (1 - cover_rate)) * 100
            overlap_metrics[name] = {
                "n": n,
                "wins": int(wins),
                "cover_rate": cover_rate,
                "roi_pct": roi,
            }

    results["_overlap_analysis"] = {
        "old_total": int(old_mask.sum()),
        "ev3_total": int(new_ev3_mask.sum()),
        "ev5_total": int(new_ev5_mask.sum()),
        "overlap_old_ev3": int(overlap_old_ev3),
        "overlap_old_ev5": int(overlap_old_ev5),
        "old_only_vs_ev3": int(old_only),
        "ev3_only_vs_old": int(ev3_only),
        "set_metrics": overlap_metrics,
    }

    return results


def print_strategy_comparison(results: dict) -> None:
    """Print strategy comparison results."""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON (Walk-Forward Out-of-Fold)")
    print("=" * 80)

    print("\nSUMMARY:")
    print("-" * 80)
    print(f"{'Strategy':<12} | {'N Bets':>8} | {'W-L':>10} | {'Cover%':>8} | {'ROI%':>8} | {'Avg P(c)':>8} | {'Avg Edge':>8}")
    print("-" * 80)

    for name in ["OLD_5pt", "NEW_EV3", "NEW_EV5", "NEW_EV8"]:
        if name not in results:
            continue
        r = results[name]
        if "error" in r:
            print(f"{name:<12} | {'N/A':>8} | {r['error']}")
            continue

        wl = f"{r['wins']}-{r['losses']}"
        print(
            f"{name:<12} | {r['n_bets']:>8} | {wl:>10} | {r['cover_rate']*100:>7.1f}% | "
            f"{r['roi_pct']:>+7.1f}% | {r['avg_p_cover']:>8.3f} | {r['avg_edge']:>8.1f}"
        )

    print("-" * 80)

    # Yearly breakdown for OLD_5pt and NEW_EV5
    print("\nYEARLY BREAKDOWN:")
    print("-" * 80)

    for name in ["OLD_5pt", "NEW_EV5"]:
        if name not in results or "error" in results[name]:
            continue
        print(f"\n{name}:")
        for y in results[name]["yearly"]:
            wl = f"{y['wins']}-{y['losses']}"
            print(f"  {y['year']}: {wl:>10} ({y['cover_rate']*100:.1f}%)")

    # Overlap analysis
    if "_overlap_analysis" in results:
        oa = results["_overlap_analysis"]
        print("\n" + "-" * 80)
        print("OVERLAP ANALYSIS (OLD_5pt vs NEW_EV3):")
        print(f"  OLD_5pt selects:     {oa['old_total']} bets")
        print(f"  NEW_EV3 selects:     {oa['ev3_total']} bets")
        if oa['old_total'] > 0:
            overlap_pct = f"{oa['overlap_old_ev3']/oa['old_total']*100:.1f}%"
        else:
            overlap_pct = "N/A"
        print(f"  Overlap:             {oa['overlap_old_ev3']} bets ({overlap_pct} of OLD)")
        print(f"  OLD only:            {oa['old_only_vs_ev3']} bets")
        print(f"  EV3 only:            {oa['ev3_only_vs_old']} bets")

        if "set_metrics" in oa:
            print("\n  Performance by set:")
            for set_name, m in oa["set_metrics"].items():
                wl = f"{m['wins']}-{m['n']-m['wins']}"
                print(f"    {set_name:<12}: {wl:>10} ({m['cover_rate']*100:.1f}%, ROI {m['roi_pct']:+.1f}%)")

    print("\n" + "=" * 80)


def print_calibration_diagnostics(diag: dict) -> None:
    """Print calibration diagnostics."""
    print("\n" + "=" * 80)
    print("CALIBRATION DIAGNOSTICS")
    print("=" * 80)

    if "error" in diag:
        print(f"Error: {diag['error']}")
        return

    print(f"\nOverall: {diag['n_games']} games, empirical cover rate = {diag['empirical_cover_rate']*100:.1f}%")

    # Bucket stats
    print("\nSPREAD-BUCKET CALIBRATION:")
    print("-" * 80)
    print(f"{'Bucket':<10} | {'N':>6} | {'Empirical':>10} | {'Predicted':>10} | {'Wilson 95% CI':>18} | {'In CI':>6}")
    print("-" * 80)

    for b in diag["bucket_stats"]:
        ci = f"[{b['wilson_low']:.3f}, {b['wilson_high']:.3f}]"
        in_ci = "YES" if b["in_ci"] else "NO"
        print(
            f"{b['bucket']:<10} | {b['n']:>6} | {b['empirical_rate']*100:>9.1f}% | "
            f"{b['predicted_rate']*100:>9.1f}% | {ci:>18} | {in_ci:>6}"
        )

    print("-" * 80)

    # Monotonicity
    if diag["monotonic_violations"]:
        print("\nMONOTONICITY VIOLATIONS (>2pp decline):")
        for v in diag["monotonic_violations"]:
            print(f"  {v['from_bucket']} -> {v['to_bucket']}: {v['delta']*100:+.1f}pp")
    else:
        print("\nMonotonicity: OK (no violations)")

    # Scoring metrics
    print("\nSCORING METRICS:")
    print(f"  Brier (model):         {diag['brier_model']:.4f}")
    print(f"  Brier (constant 0.50): {diag['brier_constant_50']:.4f}")
    print(f"  Brier (constant 0.524):{diag['brier_constant_524']:.4f}")
    print(f"  Brier (best constant): {diag['brier_best_constant']:.4f}")
    print(f"  Brier Skill Score:     {diag['brier_skill_score']:.4f}")
    print(f"  Log Loss (model):      {diag['log_loss_model']:.4f}")
    print(f"  Log Loss (best const): {diag['log_loss_best_constant']:.4f}")

    beats_brier = "YES" if diag["beats_best_constant_brier"] else "NO"
    beats_log = "YES" if diag["beats_best_constant_log_loss"] else "NO"
    print(f"\n  Beats best-constant Brier?    {beats_brier}")
    print(f"  Beats best-constant Log Loss? {beats_log}")

    if not diag["beats_best_constant_log_loss"]:
        print("\n  *** WARNING: Model doesn't beat best-constant on log loss ***")
        print("  *** This suggests calibration may not be adding value ***")

    print("=" * 80)


def print_fold_stability(stability: dict) -> None:
    """Print fold stability diagnostics."""
    print("\n" + "=" * 80)
    print("FOLD STABILITY DIAGNOSTICS")
    print("=" * 80)

    if "error" in stability:
        print(f"Error: {stability['error']}")
        return

    print(f"\nAcross {stability['n_folds']} folds:")
    print(f"  Intercept:       {stability['intercept_mean']:.4f} +/- {stability['intercept_std']:.4f}")
    print(f"  Slope:           {stability['slope_mean']:.4f} +/- {stability['slope_std']:.4f} (median: {stability['slope_median']:.4f})")
    print(f"  P(cover) at 0:   {stability['p_cover_at_zero_mean']:.3f} +/- {stability['p_cover_at_zero_std']:.3f}")
    print(f"  Breakeven edge:  {stability['breakeven_edge_mean']:.2f} +/- {stability['breakeven_edge_std']:.2f}")

    # Per-fold details
    print("\nPER-FOLD DETAILS:")
    print("-" * 100)
    print(f"{'Year':>6} | {'N Train':>8} | {'Training Years':^25} | {'Intercept':>10} | {'Slope':>8} | {'P(c)@0':>8}")
    print("-" * 100)

    for f in stability["folds"]:
        # Get training years used (new field) or fall back to years_trained
        train_years = f.get("training_years_used", f.get("years_trained", []))
        train_years_str = ",".join(str(y) for y in train_years) if train_years else "N/A"
        print(
            f"{f['eval_year']:>6} | {f['n_train']:>8} | {train_years_str:^25} | "
            f"{f['intercept']:>10.4f} | {f['slope']:>8.4f} | {f['p_cover_at_zero']:>8.3f}"
        )

    print("-" * 80)

    # Warnings and flags
    if stability["intercept_warnings"]:
        print("\nINTERCEPT WARNINGS (p_cover@0 outside [0.47, 0.53]):")
        for w in stability["intercept_warnings"]:
            print(f"  {w['year']}: p_cover@0 = {w['p_cover_at_zero']:.3f}")

    if stability["intercept_flags"]:
        print("\nINTERCEPT FLAGS (p_cover@0 outside [0.45, 0.55]):")
        for f in stability["intercept_flags"]:
            print(f"  {f['year']}: p_cover@0 = {f['p_cover_at_zero']:.3f}")

    if stability["slope_violations"]:
        print("\nSLOPE VIOLATIONS (>30% from median):")
        for v in stability["slope_violations"]:
            print(f"  {v['year']}: slope = {v['slope']:.4f} (median: {v['median']:.4f}, {v['deviation_pct']:.1f}% off)")

    if not stability["intercept_warnings"] and not stability["intercept_flags"] and not stability["slope_violations"]:
        print("\nStability: OK (no warnings or flags)")

    print("=" * 80)


def run_validate(
    start_year: int,
    end_year: int,
    line_type: str = "close",
    csv_path: Optional[str] = None,
    training_window_seasons: int | None = DEFAULT_TRAINING_WINDOW_SEASONS,
    calibration_label: Optional[str] = None,
    exclude_years: list[int] | None = None,
    compare_modes: bool = False,
) -> None:
    """Run full validation pipeline.

    Args:
        start_year: First year
        end_year: Last year
        line_type: "close" or "open"
        csv_path: Path to exported backtest CSV
        training_window_seasons: Training window size (None = all prior, 2 = ROLLING_2)
        calibration_label: Optional label override (auto-derived if None)
        exclude_years: Years to exclude from training
        compare_modes: If True, run both primary and ultra modes and compare
    """
    if compare_modes:
        # Run mode comparison instead of single validation
        run_mode_comparison(start_year, end_year, csv_path)
        return

    # Derive calibration label
    if calibration_label is None:
        calibration_label = get_calibration_label(training_window_seasons)

    print(f"\n{'=' * 80}")
    print(f"CALIBRATED SPREAD SELECTION VALIDATION")
    print(f"{'=' * 80}")
    print(f"Calibration Mode:    {calibration_label}")
    print(f"Training Window:     {training_window_seasons if training_window_seasons else 'ALL prior seasons'}")
    print(f"Years:               {start_year}-{end_year}")
    print(f"Line Type:           {line_type}")
    if exclude_years:
        print(f"Excluded from Train: {exclude_years}")
    print("=" * 80)

    # Step 1: Load data
    raw_df = load_backtest_data(start_year, end_year, csv_path)

    # Step 2: Normalize
    normalized_df = load_and_normalize_game_data(raw_df, jp_convention="pos_home_favored")

    # Step 3: Print sample games for manual verification
    print_sample_games(normalized_df, n=10)

    # Step 4: Walk-forward validation
    print("\nRunning walk-forward validation...")
    wf_result = walk_forward_validate(
        normalized_df,
        min_train_seasons=2,
        exclude_covid=True,
        exclude_years_from_training=exclude_years,
        training_window_seasons=training_window_seasons,
    )

    print(f"\nWalk-forward complete: {len(wf_result.game_results)} games evaluated")
    print(f"  Brier score (model):       {wf_result.overall_brier:.4f}")
    print(f"  Brier (constant 0.5):      {wf_result.brier_vs_constant:.4f}")
    print(f"  Brier (best constant):     {wf_result.brier_vs_best_constant:.4f}")
    print(f"  Brier Skill Score:         {wf_result.brier_skill_score:.4f}")
    print(f"  Log Loss (model):          {wf_result.overall_log_loss:.4f}")

    # Step 5: Calibration diagnostics
    diag = diagnose_calibration(wf_result.game_results)
    print_calibration_diagnostics(diag)

    # Step 6: Fold stability
    stability = diagnose_fold_stability(wf_result.fold_summaries)
    print_fold_stability(stability)

    # Step 7: Strategy comparison
    strategy_results = compare_strategies(wf_result.game_results)
    print_strategy_comparison(strategy_results)

    # Step 8: GO/NO-GO Summary
    print("\n" + "=" * 80)
    print("GO/NO-GO SUMMARY")
    print("=" * 80)

    issues = []
    positives = []

    # Check key metrics
    if wf_result.brier_skill_score <= 0:
        issues.append(f"Brier Skill Score <= 0 ({wf_result.brier_skill_score:.4f}) - model not better than constant")

    if not diag["beats_best_constant_log_loss"]:
        issues.append("Model doesn't beat best-constant on log loss")

    if stability["intercept_flags"]:
        issues.append(f"Intercept stability flags: {len(stability['intercept_flags'])} folds")

    if stability["slope_violations"]:
        issues.append(f"Slope stability violations: {len(stability['slope_violations'])} folds")

    if diag["monotonic_violations"]:
        issues.append(f"Monotonicity violations: {len(diag['monotonic_violations'])}")

    # Check strategy comparison
    old_5pt = strategy_results.get("OLD_5pt", {})
    new_ev5 = strategy_results.get("NEW_EV5", {})

    if "error" not in old_5pt and "error" not in new_ev5:
        old_roi = old_5pt["roi_pct"]
        new_roi = new_ev5["roi_pct"]

        if new_roi >= old_roi:
            positives.append(f"NEW_EV5 ROI ({new_roi:+.1f}%) >= OLD_5pt ROI ({old_roi:+.1f}%)")
        else:
            issues.append(f"NEW_EV5 ROI ({new_roi:+.1f}%) < OLD_5pt ROI ({old_roi:+.1f}%)")

        if new_ev5["cover_rate"] >= old_5pt["cover_rate"]:
            positives.append(f"NEW_EV5 cover rate ({new_ev5['cover_rate']*100:.1f}%) >= OLD ({old_5pt['cover_rate']*100:.1f}%)")

    # Positive indicators
    if wf_result.brier_skill_score > 0:
        positives.append(f"Positive Brier Skill Score ({wf_result.brier_skill_score:.4f})")

    if diag["beats_best_constant_brier"]:
        positives.append("Model beats best-constant on Brier score")

    # Print summary
    print("\nPOSITIVE INDICATORS:")
    for p in positives:
        print(f"  + {p}")

    if issues:
        print("\nISSUES TO INVESTIGATE:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("\nNo major issues detected.")

    # Final verdict
    if len(issues) == 0 and len(positives) >= 2:
        print("\n>>> RECOMMENDATION: GO - Calibration appears valid <<<")
    elif len(issues) <= 1 and len(positives) >= 1:
        print("\n>>> RECOMMENDATION: CONDITIONAL GO - Minor issues, proceed with caution <<<")
    else:
        print("\n>>> RECOMMENDATION: NO-GO - Significant issues detected <<<")

    print("=" * 80)


def run_mode_comparison(
    start_year: int,
    end_year: int,
    csv_path: Optional[str] = None,
) -> None:
    """Compare PRIMARY (ROLLING_2) vs ULTRA (INCLUDE_ALL) calibration modes.

    Args:
        start_year: First year
        end_year: Last year
        csv_path: Path to exported backtest CSV
    """
    from scipy.special import expit

    print(f"\n{'=' * 80}")
    print("CALIBRATION MODE COMPARISON: PRIMARY vs ULTRA")
    print("=" * 80)
    print(f"PRIMARY: {CALIBRATION_MODES['primary']['label']} - {CALIBRATION_MODES['primary']['description']}")
    print(f"ULTRA:   {CALIBRATION_MODES['ultra']['label']} - {CALIBRATION_MODES['ultra']['description']}")
    print("=" * 80)

    # Load data
    raw_df = load_backtest_data(start_year, end_year, csv_path)
    normalized_df = load_and_normalize_game_data(raw_df, jp_convention="pos_home_favored")
    print(f"\nData: {len(normalized_df)} games total ({start_year}-{end_year})")

    # Run both modes
    modes = ["primary", "ultra"]
    results = {}

    for mode in modes:
        config = CALIBRATION_MODES[mode]
        print(f"\n--- Running: {config['label']} ---")

        wf_result = walk_forward_validate(
            normalized_df,
            min_train_seasons=2,
            exclude_covid=True,
            training_window_seasons=config["training_window_seasons"],
        )

        # Calculate EV
        df = wf_result.game_results.copy()
        df = df[df["p_cover_no_push"].notna() & ~df["push"]]
        df["ev"] = calculate_ev_vectorized(df["p_cover_no_push"].values)

        # Compute aggregate metrics
        avg_slope = np.mean([f["slope"] for f in wf_result.fold_summaries])
        avg_intercept = np.mean([f["intercept"] for f in wf_result.fold_summaries])
        p_at_5 = expit(avg_intercept + avg_slope * 5)
        p_at_10 = expit(avg_intercept + avg_slope * 10)

        be_prob = breakeven_prob(-110)
        logit_be = np.log(be_prob / (1 - be_prob))
        implied_be_edge = (logit_be - avg_intercept) / avg_slope if avg_slope > 0 else float("inf")

        # Strategy metrics
        strategies = {}
        for strat, ev_threshold in [("OLD_5pt", None), ("EV3", 0.03), ("EV5", 0.05)]:
            if ev_threshold is not None:
                mask = df["ev"] >= ev_threshold
            else:
                mask = df["edge_abs"] >= 5.0

            subset = df[mask]
            n = len(subset)
            if n > 0:
                wins = subset["jp_side_covered"].sum()
                cover_rate = wins / n
                roi = (cover_rate * (100 / 110) - (1 - cover_rate)) * 100
                avg_p = subset["p_cover_no_push"].mean()
                avg_ev = subset["ev"].mean()
            else:
                wins, cover_rate, roi, avg_p, avg_ev = 0, 0, 0, 0, 0

            strategies[strat] = {
                "n": n,
                "wins": int(wins),
                "cover_rate": cover_rate,
                "roi": roi,
                "avg_p_cover": avg_p,
                "avg_ev": avg_ev,
            }

        results[mode] = {
            "label": config["label"],
            "slope": avg_slope,
            "intercept": avg_intercept,
            "p_at_5": p_at_5,
            "p_at_10": p_at_10,
            "implied_be_edge": implied_be_edge,
            "brier": wf_result.overall_brier,
            "strategies": strategies,
            "fold_summaries": wf_result.fold_summaries,
        }

    # Print comparison table
    print("\n" + "=" * 80)
    print("CALIBRATION PARAMETERS")
    print("=" * 80)

    r_pri = results["primary"]
    r_ult = results["ultra"]

    print(f"\n| {'Metric':<20} | {'PRIMARY':>14} | {'ULTRA':>14} | {'Delta':>10} |")
    print(f"|{'-'*22}|{'-'*16}|{'-'*16}|{'-'*12}|")

    rows = [
        ("Slope", f"{r_pri['slope']:.5f}", f"{r_ult['slope']:.5f}",
         f"{r_ult['slope'] - r_pri['slope']:+.5f}"),
        ("P(cover) @ 5 pts", f"{r_pri['p_at_5']*100:.1f}%", f"{r_ult['p_at_5']*100:.1f}%",
         f"{(r_ult['p_at_5'] - r_pri['p_at_5'])*100:+.1f}%"),
        ("P(cover) @ 10 pts", f"{r_pri['p_at_10']*100:.1f}%", f"{r_ult['p_at_10']*100:.1f}%",
         f"{(r_ult['p_at_10'] - r_pri['p_at_10'])*100:+.1f}%"),
        ("Implied BE Edge", f"{r_pri['implied_be_edge']:.1f} pts", f"{r_ult['implied_be_edge']:.1f} pts",
         f"{r_ult['implied_be_edge'] - r_pri['implied_be_edge']:+.1f}"),
        ("Brier Score", f"{r_pri['brier']:.4f}", f"{r_ult['brier']:.4f}",
         f"{r_ult['brier'] - r_pri['brier']:+.4f}"),
    ]

    for metric, pri, ult, delta in rows:
        print(f"| {metric:<20} | {pri:>14} | {ult:>14} | {delta:>10} |")

    # Fold details
    print("\n" + "=" * 80)
    print("PER-FOLD TRAINING YEARS")
    print("=" * 80)

    print(f"\n| {'Year':>6} | {'PRIMARY':^30} | {'ULTRA':^30} |")
    print(f"|{'-'*8}|{'-'*32}|{'-'*32}|")

    # Align folds by eval_year instead of positional index
    pri_folds_by_year = {f["eval_year"]: f for f in r_pri["fold_summaries"]}
    ult_folds_by_year = {f["eval_year"]: f for f in r_ult["fold_summaries"]}
    common_years = sorted(set(pri_folds_by_year.keys()) & set(ult_folds_by_year.keys()))

    for eval_year in common_years:
        fold_pri = pri_folds_by_year[eval_year]
        fold_ult = ult_folds_by_year[eval_year]
        pri_years = ",".join(str(y) for y in fold_pri.get("training_years_used", fold_pri["years_trained"]))
        ult_years = ",".join(str(y) for y in fold_ult.get("training_years_used", fold_ult["years_trained"]))
        print(f"| {eval_year:>6} | {pri_years:^30} | {ult_years:^30} |")

    # Strategy comparison
    print("\n" + "=" * 80)
    print("STRATEGY METRICS COMPARISON")
    print("=" * 80)

    for strat in ["OLD_5pt", "EV3", "EV5"]:
        s_pri = r_pri["strategies"][strat]
        s_ult = r_ult["strategies"][strat]

        print(f"\n{strat}:")
        print(f"  {'':20} | {'PRIMARY':>12} | {'ULTRA':>12} | {'Delta':>10}")
        print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

        print(f"  {'N Bets':20} | {s_pri['n']:>12} | {s_ult['n']:>12} | {s_ult['n'] - s_pri['n']:>+10}")
        print(f"  {'W-L':20} | {s_pri['wins']:>5}-{s_pri['n']-s_pri['wins']:<5} | {s_ult['wins']:>5}-{s_ult['n']-s_ult['wins']:<5} |")
        print(f"  {'Cover Rate':20} | {s_pri['cover_rate']*100:>11.1f}% | {s_ult['cover_rate']*100:>11.1f}% | "
              f"{(s_ult['cover_rate'] - s_pri['cover_rate'])*100:>+9.1f}%")
        print(f"  {'ROI':20} | {s_pri['roi']:>+11.1f}% | {s_ult['roi']:>+11.1f}% | "
              f"{s_ult['roi'] - s_pri['roi']:>+9.1f}%")
        print(f"  {'Avg EV':20} | {s_pri['avg_ev']:>12.3f} | {s_ult['avg_ev']:>12.3f} | "
              f"{s_ult['avg_ev'] - s_pri['avg_ev']:>+10.3f}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    pri_ev5_roi = r_pri["strategies"]["EV5"]["roi"]
    ult_ev5_roi = r_ult["strategies"]["EV5"]["roi"]
    pri_ev5_n = r_pri["strategies"]["EV5"]["n"]
    ult_ev5_n = r_ult["strategies"]["EV5"]["n"]

    print(f"""
PRIMARY ({r_pri['label']}):
  - EV5 bets: {pri_ev5_n} at {pri_ev5_roi:+.1f}% ROI
  - Better for: Consistent volume, realistic betting cadence

ULTRA ({r_ult['label']}):
  - EV5 bets: {ult_ev5_n} at {ult_ev5_roi:+.1f}% ROI
  - Better for: High-conviction plays, low volume strategy

Production default: PRIMARY (training_window_seasons=2)
""")
    print("=" * 80)


def run_sensitivity_report(
    csv_path: Optional[str] = None,
    eval_years: list[int] | None = None,
) -> None:
    """Run sensitivity report comparing include vs exclude 2022 from training.

    Args:
        csv_path: Path to exported backtest CSV
        eval_years: Years to evaluate on (default: [2024, 2025])
    """
    from scipy.special import expit

    print("\n" + "=" * 80)
    print("SENSITIVITY REPORT: EXCLUDE 2022 FROM CALIBRATION TRAINING")
    print("=" * 80)

    if eval_years is None:
        eval_years = [2024, 2025]

    # Load data
    raw_df = load_backtest_data(2022, 2025, csv_path)
    normalized_df = load_and_normalize_game_data(raw_df, jp_convention="pos_home_favored")

    print(f"\nData: {len(normalized_df)} games total")
    print(f"Evaluation years: {eval_years}")

    # Run both configurations
    configs = [
        {"name": "INCLUDE 2022", "exclude_years": []},
        {"name": "EXCLUDE 2022", "exclude_years": [2022]},
    ]

    results = {}
    for config in configs:
        print(f"\n--- Running: {config['name']} ---")

        wf_result = walk_forward_validate(
            normalized_df,
            min_train_seasons=2,
            exclude_covid=True,
            exclude_years_from_training=config["exclude_years"],
        )

        # Filter to evaluation years only
        eval_df = wf_result.game_results[
            wf_result.game_results["year"].isin(eval_years)
        ].copy()

        # Filter to games with predictions
        eval_df = eval_df[eval_df["p_cover_no_push"].notna() & ~eval_df["push"]]

        # Calculate EV
        eval_df["ev"] = calculate_ev_vectorized(eval_df["p_cover_no_push"].values)

        # Compute metrics
        avg_slope = np.mean([f["slope"] for f in wf_result.fold_summaries])
        avg_intercept = np.mean([f["intercept"] for f in wf_result.fold_summaries])

        # P(cover) at various edge points
        p_at_5 = expit(avg_intercept + avg_slope * 5)
        p_at_7 = expit(avg_intercept + avg_slope * 7)
        p_at_10 = expit(avg_intercept + avg_slope * 10)

        # Implied breakeven edge
        be_prob = breakeven_prob(-110)
        logit_be = np.log(be_prob / (1 - be_prob))
        if avg_slope > 0:
            implied_be_edge = (logit_be - avg_intercept) / avg_slope
        else:
            implied_be_edge = float("inf")

        # Strategy metrics
        strategies = {}
        for name, ev_threshold in [("OLD_5pt", None), ("EV3", 0.03), ("EV5", 0.05)]:
            if ev_threshold is not None:
                mask = eval_df["ev"] >= ev_threshold
            else:
                mask = eval_df["edge_abs"] >= 5.0

            subset = eval_df[mask]
            n = len(subset)
            if n > 0:
                wins = subset["jp_side_covered"].sum()
                cover_rate = wins / n
                roi = (cover_rate * (100 / 110) - (1 - cover_rate)) * 100
                avg_p = subset["p_cover_no_push"].mean()
            else:
                wins, cover_rate, roi, avg_p = 0, 0, 0, 0

            strategies[name] = {
                "n": n,
                "wins": int(wins),
                "cover_rate": cover_rate,
                "roi": roi,
                "avg_p_cover": avg_p,
            }

        results[config["name"]] = {
            "slope": avg_slope,
            "intercept": avg_intercept,
            "p_at_5": p_at_5,
            "p_at_7": p_at_7,
            "p_at_10": p_at_10,
            "implied_be_edge": implied_be_edge,
            "n_eval": len(eval_df),
            "brier": wf_result.overall_brier,
            "strategies": strategies,
            "fold_details": [
                {"year": f["eval_year"], "slope": f["slope"], "n_train": f["n_train"]}
                for f in wf_result.fold_summaries
            ],
        }

    # Print comparison table
    print("\n" + "=" * 80)
    print("CALIBRATION PARAMETERS COMPARISON")
    print("=" * 80)

    print("\n| Metric           | INCLUDE 2022 | EXCLUDE 2022 | Delta    |")
    print("|------------------|--------------|--------------|----------|")

    r_inc = results["INCLUDE 2022"]
    r_exc = results["EXCLUDE 2022"]

    rows = [
        ("Avg Slope", f"{r_inc['slope']:.5f}", f"{r_exc['slope']:.5f}",
         f"{r_exc['slope'] - r_inc['slope']:+.5f}"),
        ("Avg Intercept", f"{r_inc['intercept']:.4f}", f"{r_exc['intercept']:.4f}",
         f"{r_exc['intercept'] - r_inc['intercept']:+.4f}"),
        ("P(cover) @ 5 pts", f"{r_inc['p_at_5']*100:.1f}%", f"{r_exc['p_at_5']*100:.1f}%",
         f"{(r_exc['p_at_5'] - r_inc['p_at_5'])*100:+.1f}%"),
        ("P(cover) @ 7 pts", f"{r_inc['p_at_7']*100:.1f}%", f"{r_exc['p_at_7']*100:.1f}%",
         f"{(r_exc['p_at_7'] - r_inc['p_at_7'])*100:+.1f}%"),
        ("P(cover) @ 10 pts", f"{r_inc['p_at_10']*100:.1f}%", f"{r_exc['p_at_10']*100:.1f}%",
         f"{(r_exc['p_at_10'] - r_inc['p_at_10'])*100:+.1f}%"),
        ("Implied BE Edge", f"{r_inc['implied_be_edge']:.1f} pts", f"{r_exc['implied_be_edge']:.1f} pts",
         f"{r_exc['implied_be_edge'] - r_inc['implied_be_edge']:+.1f}"),
        ("Brier Score", f"{r_inc['brier']:.4f}", f"{r_exc['brier']:.4f}",
         f"{r_exc['brier'] - r_inc['brier']:+.4f}"),
    ]

    for metric, inc, exc, delta in rows:
        print(f"| {metric:<16} | {inc:>12} | {exc:>12} | {delta:>8} |")

    # Fold details
    print("\n" + "=" * 80)
    print("PER-FOLD SLOPE COMPARISON")
    print("=" * 80)

    print("\n| Eval Year | INCLUDE 2022       | EXCLUDE 2022       |")
    print("|           | N Train | Slope    | N Train | Slope    |")
    print("|-----------|---------|----------|---------|----------|")

    for i, fold_inc in enumerate(r_inc["fold_details"]):
        fold_exc = r_exc["fold_details"][i]
        print(f"| {fold_inc['year']:>9} | {fold_inc['n_train']:>7} | {fold_inc['slope']:.6f} | "
              f"{fold_exc['n_train']:>7} | {fold_exc['slope']:.6f} |")

    # Strategy comparison
    print("\n" + "=" * 80)
    print("STRATEGY METRICS COMPARISON (on same eval years)")
    print("=" * 80)

    for strat in ["OLD_5pt", "EV3", "EV5"]:
        s_inc = r_inc["strategies"][strat]
        s_exc = r_exc["strategies"][strat]

        print(f"\n{strat}:")
        print(f"  {'':20} | {'INCLUDE 2022':>14} | {'EXCLUDE 2022':>14} | {'Delta':>10}")
        print(f"  {'-'*20}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}")

        print(f"  {'N Bets':20} | {s_inc['n']:>14} | {s_exc['n']:>14} | {s_exc['n'] - s_inc['n']:>+10}")
        print(f"  {'Wins':20} | {s_inc['wins']:>14} | {s_exc['wins']:>14} | {s_exc['wins'] - s_inc['wins']:>+10}")
        print(f"  {'Cover Rate':20} | {s_inc['cover_rate']*100:>13.1f}% | {s_exc['cover_rate']*100:>13.1f}% | "
              f"{(s_exc['cover_rate'] - s_inc['cover_rate'])*100:>+9.1f}%")
        print(f"  {'ROI':20} | {s_inc['roi']:>+13.1f}% | {s_exc['roi']:>+13.1f}% | "
              f"{s_exc['roi'] - s_inc['roi']:>+9.1f}%")
        print(f"  {'Avg P(cover)':20} | {s_inc['avg_p_cover']*100:>13.1f}% | {s_exc['avg_p_cover']*100:>13.1f}% | "
              f"{(s_exc['avg_p_cover'] - s_inc['avg_p_cover'])*100:>+9.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    slope_improvement = (r_exc["slope"] - r_inc["slope"]) / r_inc["slope"] * 100 if r_inc["slope"] > 0 else 0
    p5_improvement = (r_exc["p_at_5"] - r_inc["p_at_5"]) * 100
    be_improvement = r_inc["implied_be_edge"] - r_exc["implied_be_edge"]

    print(f"""
Excluding 2022 from training:
  - Slope increases by {slope_improvement:+.0f}% ({r_inc['slope']:.5f} -> {r_exc['slope']:.5f})
  - P(cover) at 5-pt edge increases by {p5_improvement:+.1f}pp ({r_inc['p_at_5']*100:.1f}% -> {r_exc['p_at_5']*100:.1f}%)
  - Implied breakeven edge decreases by {be_improvement:.1f} pts ({r_inc['implied_be_edge']:.1f} -> {r_exc['implied_be_edge']:.1f})
  - EV3 bet count: {r_inc['strategies']['EV3']['n']} -> {r_exc['strategies']['EV3']['n']}
  - EV5 bet count: {r_inc['strategies']['EV5']['n']} -> {r_exc['strategies']['EV5']['n']}
  - Brier score: {r_inc['brier']:.4f} -> {r_exc['brier']:.4f} ({'better' if r_exc['brier'] < r_inc['brier'] else 'worse'})

Key insight: 2022 had a NEGATIVE slope (-0.0119) when fit alone, dragging
down the cumulative training slope. Excluding it produces calibration
parameters more consistent with the 2023-2025 regime.

However, note that:
  - The empirical cover rate (55.4% at 5+ edge) is UNCHANGED by this setting
  - Only the calibrated P(cover) predictions change
  - Conservative (include 2022) vs aggressive (exclude 2022) is a risk choice
""")

    print("=" * 80)


def run_predict(
    slate_csv: str,
    historical_csv: str,
    emit_modes: list[str],
    primary_min_ev: float = 0.03,
    ultra_min_ev: float = 0.05,
    year: Optional[int] = None,
    week: Optional[int] = None,
    output_dir: str = "data/spread_selection/outputs",
    sp_gate_config: Optional[Phase1SPGateConfig] = None,
) -> None:
    """Generate predictions for upcoming games.

    Emits separate bet lists for each requested mode:
    - PRIMARY (ROLLING_2): Realistic volume, training_window_seasons=2
    - ULTRA (INCLUDE_ALL): High conviction, training_window_seasons=None

    Args:
        slate_csv: Path to CSV with upcoming slate
        historical_csv: Path to historical ATS CSV for calibration
        emit_modes: List of modes to emit ("primary", "ultra")
        primary_min_ev: Minimum EV for PRIMARY mode bets
        ultra_min_ev: Minimum EV for ULTRA mode bets
        year: Year for predictions (default: infer from slate)
        week: Week for predictions (default: infer from slate)
        output_dir: Output directory for prediction files
        sp_gate_config: Optional Phase 1 SP+ gate configuration (default OFF)
    """
    from scipy.special import expit

    print(f"\n{'=' * 80}")
    print("SPREAD SELECTION PREDICTION")
    print("=" * 80)
    print(f"Modes: {', '.join(emit_modes)}")
    print(f"PRIMARY min EV: {primary_min_ev:.2f}")
    print(f"ULTRA min EV:   {ultra_min_ev:.2f}")
    if sp_gate_config and sp_gate_config.enabled:
        print(f"SP+ Gate: ENABLED (mode={sp_gate_config.mode}, weeks={sp_gate_config.weeks})")
    else:
        print(f"SP+ Gate: DISABLED")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load slate
    print(f"\nLoading slate from {slate_csv}...")
    slate_df = pd.read_csv(slate_csv)
    print(f"  Loaded {len(slate_df)} games")

    # Validate required columns
    required = ["game_id", "home_team", "away_team"]
    missing = set(required) - set(slate_df.columns)
    if missing:
        raise ValueError(f"Slate CSV missing required columns: {missing}")

    # Need either jp_spread or predicted_spread
    if "jp_spread" not in slate_df.columns and "predicted_spread" not in slate_df.columns:
        raise ValueError("Slate CSV must have 'jp_spread' or 'predicted_spread' column")

    # Normalize JP spread to Vegas convention
    if "jp_spread" in slate_df.columns:
        # Already in Vegas convention
        pass
    else:
        # Internal convention: positive = home favored, flip it
        slate_df["jp_spread"] = -slate_df["predicted_spread"]

    # Need vegas_spread
    if "vegas_spread" not in slate_df.columns:
        if "spread_close" in slate_df.columns:
            slate_df["vegas_spread"] = slate_df["spread_close"]
        elif "spread_open" in slate_df.columns:
            slate_df["vegas_spread"] = slate_df["spread_open"]
        else:
            raise ValueError("Slate CSV must have 'vegas_spread', 'spread_close', or 'spread_open'")

    # Derive year/week if not provided
    if year is None:
        if "year" in slate_df.columns:
            year = int(slate_df["year"].iloc[0])
        else:
            year = datetime.now().year
    if week is None:
        if "week" in slate_df.columns:
            week = int(slate_df["week"].iloc[0])
        else:
            week = 0  # Unknown

    print(f"  Year: {year}, Week: {week}")

    # Load historical data for calibration fitting
    print(f"\nLoading historical data from {historical_csv}...")
    if not Path(historical_csv).exists():
        raise ValueError(
            f"Historical CSV not found: {historical_csv}. "
            "Run: python3 scripts/backtest.py --export-ats data/spread_selection/ats_export.csv"
        )

    historical_df = pd.read_csv(historical_csv)
    # Filter to years before prediction year
    historical_df = historical_df[historical_df["year"] < year]
    print(f"  Loaded {len(historical_df)} historical games (years < {year})")

    # Normalize historical data
    historical_normalized = load_and_normalize_game_data(
        historical_df, jp_convention="pos_home_favored"
    )

    # Calculate edge for slate
    slate_df["edge_pts"] = slate_df["jp_spread"] - slate_df["vegas_spread"]
    slate_df["edge_abs"] = np.abs(slate_df["edge_pts"])
    slate_df["jp_favored_side"] = np.where(slate_df["edge_pts"] < 0, "HOME", "AWAY")

    # Process each mode
    mode_configs = {
        "primary": {
            "training_window_seasons": CALIBRATION_MODES["primary"]["training_window_seasons"],
            "label": CALIBRATION_MODES["primary"]["label"],
            "min_ev": primary_min_ev,
        },
        "ultra": {
            "training_window_seasons": CALIBRATION_MODES["ultra"]["training_window_seasons"],
            "label": CALIBRATION_MODES["ultra"]["label"],
            "min_ev": ultra_min_ev,
        },
    }

    for mode in emit_modes:
        if mode not in mode_configs:
            print(f"WARNING: Unknown mode '{mode}', skipping")
            continue

        config = mode_configs[mode]
        print(f"\n--- Processing: {config['label']} mode ---")

        # Log calibration source
        logger.info(f"[PREDICT] Mode={mode}, source=REFIT from {historical_csv}")

        # Fit calibration using walk-forward (use most recent available fold)
        # Get years available for training
        train_years = sorted(historical_normalized["year"].unique())
        logger.info(f"[PREDICT] Seasons available in historical data: {list(train_years)}")

        # No explicit exclusions in predict (unlike validate which can exclude years)
        excluded_years = []  # Could be parameterized in future
        logger.info(f"[PREDICT] Seasons excluded from training: {excluded_years}")

        if config["training_window_seasons"] is not None:
            # Rolling window
            window_years = train_years[-config["training_window_seasons"]:]
        else:
            # All years
            window_years = train_years

        train_mask = historical_normalized["year"].isin(window_years)
        train_df = historical_normalized[train_mask]

        # Log detailed training info
        logger.info(f"[PREDICT] Final training_years_used: {list(window_years)}")

        # Count exclusions that calibrate_cover_probability will apply
        push_count = train_df["push"].sum() if "push" in train_df.columns else 0
        zero_edge_count = (train_df["edge_abs"] == 0).sum() if "edge_abs" in train_df.columns else 0
        n_after_exclusions = len(train_df) - push_count - zero_edge_count

        logger.info(f"[PREDICT] Training rows before exclusions: {len(train_df)}")
        logger.info(f"[PREDICT] Pushes to exclude: {push_count}")
        logger.info(f"[PREDICT] Zero-edge to exclude: {zero_edge_count}")
        logger.info(f"[PREDICT] Expected training rows after exclusions: ~{n_after_exclusions}")

        print(f"  Training years: {list(window_years)}")
        print(f"  Training games: {len(train_df)}")

        try:
            calibration = calibrate_cover_probability(train_df, min_games_warn=500)
        except ValueError as e:
            print(f"  ERROR: Calibration failed: {e}")
            logger.error(f"[PREDICT] Calibration failed: {e}")
            continue

        # Log fitted calibration parameters
        p_at_10 = expit(calibration.intercept + calibration.slope * 10)
        logger.info(f"[PREDICT] Fitted intercept: {calibration.intercept:.10f}")
        logger.info(f"[PREDICT] Fitted slope: {calibration.slope:.10f}")
        logger.info(f"[PREDICT] N training games (after exclusions): {calibration.n_games}")
        logger.info(f"[PREDICT] p_cover_at_zero: {calibration.p_cover_at_zero:.6f}")
        logger.info(f"[PREDICT] implied_5pt_pcover: {calibration.implied_5pt_pcover:.6f}")
        logger.info(f"[PREDICT] implied_10pt_pcover: {p_at_10:.6f}")
        logger.info(f"[PREDICT] implied_breakeven_edge: {calibration.implied_breakeven_edge:.4f}")

        print(f"  Calibration: slope={calibration.slope:.5f}, intercept={calibration.intercept:.4f}")
        print(f"  P(cover) @ 5 pts: {calibration.implied_5pt_pcover:.1%}")
        print(f"  P(cover) @ 10 pts: {p_at_10:.1%}")
        print(f"  Implied BE edge: {calibration.implied_breakeven_edge:.1f} pts")

        # Estimate push rates from training data (same pattern as walk_forward_validate)
        push_rates = None
        try:
            push_rates = estimate_push_rates(train_df)
            logger.info(f"[PREDICT] Push rates estimated: default_even={push_rates.default_even:.4f}")
        except Exception as e:
            logger.warning(f"[PREDICT] Push rate estimation failed: {e}. Falling back to p_push=0.")

        # Apply calibration to slate
        slate_pred = slate_df.copy()
        slate_pred["p_cover_no_push"] = predict_cover_probability(
            slate_pred["edge_abs"].values, calibration
        )

        # Calculate push-aware EV
        if push_rates is not None:
            # Get per-game push probabilities based on Vegas spread
            slate_pred["p_push"] = get_push_probability_vectorized(
                slate_pred["vegas_spread"].values, push_rates
            )
            # Push-aware EV formula: ev = p_win * (100/110) - p_lose
            # where p_win = p_cover_no_push * (1 - p_push), p_lose = (1 - p_cover_no_push) * (1 - p_push)
            p_cover = slate_pred["p_cover_no_push"].values
            p_push = slate_pred["p_push"].values
            p_win = p_cover * (1 - p_push)
            p_lose = (1 - p_cover) * (1 - p_push)
            slate_pred["ev"] = p_win * (100 / 110) - p_lose
        else:
            # Fallback to p_push=0
            slate_pred["p_push"] = 0.0
            slate_pred["ev"] = calculate_ev_vectorized(slate_pred["p_cover_no_push"].values)

        # Assign tiers
        min_ev = config["min_ev"]
        slate_pred["tier"] = "PASS"
        slate_pred.loc[slate_pred["ev"] >= min_ev, "tier"] = "BET"
        slate_pred.loc[slate_pred["ev"] >= min_ev + 0.02, "tier"] = "MED"
        slate_pred.loc[slate_pred["ev"] >= min_ev + 0.05, "tier"] = "HIGH"

        # Sort by EV descending
        slate_pred = slate_pred.sort_values("ev", ascending=False)

        # Extract bets BEFORE gate (baseline)
        bets_before_gate = slate_pred[slate_pred["tier"] != "PASS"].copy()

        # Apply Phase 1 SP+ Gate (post-selection filter)
        gate_results = []
        sp_spreads = {}
        if sp_gate_config and sp_gate_config.should_apply(week):
            print(f"\n  Applying SP+ Gate (mode={sp_gate_config.mode})...")

            # Fetch SP+ spreads
            try:
                from src.api.cfbd_client import CFBDClient
                client = CFBDClient()
                sp_spreads = fetch_sp_spreads_vegas(client, year, [week])
                print(f"    Fetched {len(sp_spreads)} SP+ spreads")
            except Exception as e:
                logger.warning(f"Failed to fetch SP+ spreads: {e}")
                sp_spreads = {}

            # Evaluate gate for each game in slate
            for _, row in slate_pred.iterrows():
                game_id = int(row['game_id'])
                sp_spread = sp_spreads.get(game_id)
                result = evaluate_single_game(
                    game_id=game_id,
                    jp_spread=row['jp_spread'],
                    vegas_spread=row['vegas_spread'],
                    sp_spread=sp_spread,
                    config=sp_gate_config,
                )
                gate_results.append(result)

            # Merge gate results into slate
            slate_pred = merge_gate_results_to_df(slate_pred, gate_results)

            # Filter bets based on gate (only for BET/MED/HIGH tiers)
            if sp_gate_config.candidate_basis == "ev":
                # Candidates are EV-based bets (tier != PASS)
                candidate_mask = slate_pred["tier"] != "PASS"
            else:
                # Candidates are edge-based
                candidate_mask = slate_pred["edge_abs"] >= sp_gate_config.jp_edge_min

            # Pass-through non-candidates, filter candidates by gate
            bets = slate_pred[
                ((candidate_mask) & (slate_pred["sp_gate_passed"] == True)) |
                (~candidate_mask)
            ].copy()
            bets = bets[bets["tier"] != "PASS"].copy()

            # Log gate summary
            n_before = len(bets_before_gate)
            n_after = len(bets)
            n_filtered = n_before - n_after
            print(f"    Gate result: {n_before} -> {n_after} bets ({n_filtered} filtered)")

            # Log category breakdown
            if gate_results:
                from collections import Counter
                cats = Counter(r.category.value for r in gate_results if
                               slate_pred[slate_pred['game_id'] == r.game_id]['tier'].iloc[0] != "PASS"
                               if len(slate_pred[slate_pred['game_id'] == r.game_id]) > 0)
                if cats:
                    print(f"    Categories: {dict(cats)}")
        else:
            bets = bets_before_gate.copy()
            # Add placeholder gate columns
            slate_pred['sp_spread'] = None
            slate_pred['sp_edge_pts'] = None
            slate_pred['sp_edge_abs'] = None
            slate_pred['sp_gate_category'] = None
            slate_pred['sp_gate_passed'] = True
            slate_pred['sp_gate_reason'] = None

        print(f"\n  Results ({config['label']}):")
        print(f"    Total games: {len(slate_pred)}")
        print(f"    Bets selected: {len(bets)}")
        print(f"    Tier breakdown: HIGH={len(bets[bets['tier']=='HIGH'])}, "
              f"MED={len(bets[bets['tier']=='MED'])}, "
              f"BET={len(bets[bets['tier']=='BET'])}")

        # Prepare output with metadata
        metadata = {
            "mode": mode,
            "label": config["label"],
            "training_window_seasons": config["training_window_seasons"],
            "training_years": window_years,
            "min_ev_threshold": min_ev,
            "calibration": {
                "intercept": calibration.intercept,
                "slope": calibration.slope,
                "implied_breakeven_edge": calibration.implied_breakeven_edge,
                "p_cover_at_zero": calibration.p_cover_at_zero,
                "implied_5pt_pcover": calibration.implied_5pt_pcover,
                "n_train_games": calibration.n_games,
            },
            "prediction_year": year,
            "prediction_week": week,
            "generated_at": datetime.now().isoformat(),
            "sp_gate": {
                "enabled": sp_gate_config.enabled if sp_gate_config else False,
                "mode": sp_gate_config.mode if sp_gate_config else None,
                "sp_edge_min": sp_gate_config.sp_edge_min if sp_gate_config else None,
                "jp_edge_min": sp_gate_config.jp_edge_min if sp_gate_config else None,
                "candidate_basis": sp_gate_config.candidate_basis if sp_gate_config else None,
                "n_sp_spreads_fetched": len(sp_spreads),
                "n_bets_before_gate": len(bets_before_gate),
                "n_bets_after_gate": len(bets),
            } if sp_gate_config and sp_gate_config.should_apply(week) else {"enabled": False},
        }

        # Write outputs
        # 1. Bets CSV
        bets_csv_path = output_path / f"bets_{mode}_{year}_week{week}.csv"
        bets.to_csv(bets_csv_path, index=False)
        print(f"  Wrote: {bets_csv_path}")

        # 2. Bets JSON (with metadata)
        bets_json_path = output_path / f"bets_{mode}_{year}_week{week}.json"
        bets_json = {
            "metadata": metadata,
            "bets": bets.to_dict(orient="records"),
        }
        with open(bets_json_path, "w") as f:
            json.dump(bets_json, f, indent=2, default=str)
        print(f"  Wrote: {bets_json_path}")

        # 3. Full slate CSV (debug)
        slate_csv_path = output_path / f"slate_{mode}_{year}_week{week}.csv"
        slate_pred.to_csv(slate_csv_path, index=False)
        print(f"  Wrote: {slate_csv_path}")

        # 4. Full slate JSON (debug, with metadata)
        slate_json_path = output_path / f"slate_{mode}_{year}_week{week}.json"
        slate_json = {
            "metadata": metadata,
            "games": slate_pred.to_dict(orient="records"),
        }
        with open(slate_json_path, "w") as f:
            json.dump(slate_json, f, indent=2, default=str)
        print(f"  Wrote: {slate_json_path}")

        # Print bet summary
        if len(bets) > 0:
            print(f"\n  TOP BETS ({config['label']}):")
            print(f"  {'-' * 70}")
            print(f"  {'Game':<40} | {'Side':<6} | {'Edge':>6} | {'P(c)':>6} | {'EV':>6} | {'Tier':<5}")
            print(f"  {'-' * 70}")
            for _, row in bets.head(10).iterrows():
                game = f"{row['away_team']} @ {row['home_team']}"
                if len(game) > 38:
                    game = game[:35] + "..."
                print(
                    f"  {game:<40} | {row['jp_favored_side']:<6} | "
                    f"{row['edge_abs']:>5.1f}p | {row['p_cover_no_push']:>5.1%} | "
                    f"{row['ev']:>+5.2f} | {row['tier']:<5}"
                )

    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE")
    print("=" * 80)


def run_backtest(
    csv_path: str,
    start_year: int = 2022,
    end_year: int = 2025,
    output_dir: str = "data/spread_selection/outputs",
    compare_modes: bool = True,
) -> None:
    """Run walk-forward backtest with push modeling for P&L analysis.

    Args:
        csv_path: Path to exported backtest CSV
        start_year: First year
        end_year: Last year
        output_dir: Output directory for results
        compare_modes: If True, run both PRIMARY and ULTRA modes
    """
    print(f"\n{'=' * 80}")
    print("WALK-FORWARD BACKTEST WITH PUSH MODELING")
    print("=" * 80)
    print(f"Data: {csv_path}")
    print(f"Years: {start_year}-{end_year}")
    print("=" * 80)

    # Load and normalize data
    raw_df = load_backtest_data(start_year, end_year, csv_path)
    normalized_df = load_and_normalize_game_data(raw_df, jp_convention="pos_home_favored")
    print(f"\nLoaded {len(normalized_df)} games total")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    modes = ["primary", "ultra"] if compare_modes else ["primary"]

    for mode in modes:
        config = CALIBRATION_MODES[mode]
        print(f"\n{'=' * 60}")
        print(f"MODE: {config['label']}")
        print("=" * 60)

        # Run walk-forward with push modeling
        wf_result = walk_forward_validate(
            normalized_df,
            min_train_seasons=2,
            exclude_covid=True,
            training_window_seasons=config["training_window_seasons"],
            include_push_modeling=True,
        )

        # Filter to games with predictions
        df = wf_result.game_results[
            wf_result.game_results["p_cover_no_push"].notna() & ~wf_result.game_results["push"]
        ].copy()

        print(f"\nGames with predictions: {len(df)}")

        # ----- Per-Year P&L with pushes -----
        print("\nPER-YEAR P&L (with push modeling):")
        print("-" * 70)
        print(f"{'Year':>6} | {'N':>6} | {'W-L-P':>12} | {'ATS%':>8} | {'ROI%':>8} | {'Avg EV':>8}")
        print("-" * 70)

        yearly_results = []
        for year in sorted(df["year"].unique()):
            year_df = df[df["year"] == year]

            # Calculate W-L-P
            # For backtest, we already have outcome data
            wins = year_df["jp_side_covered"].sum()
            n = len(year_df)
            losses = n - wins

            # Pushes would be 0 here since we filtered them out
            cover_rate = wins / n if n > 0 else 0

            # ROI at -110
            roi = (cover_rate * (100 / 110) - (1 - cover_rate)) * 100

            # Avg EV
            avg_ev = year_df["ev"].mean()

            yearly_results.append({
                "year": year,
                "n": n,
                "wins": int(wins),
                "losses": int(losses),
                "pushes": 0,
                "cover_rate": cover_rate,
                "roi": roi,
                "avg_ev": avg_ev,
            })

            wlp = f"{int(wins)}-{int(losses)}-0"
            print(f"{year:>6} | {n:>6} | {wlp:>12} | {cover_rate*100:>7.1f}% | {roi:>+7.1f}% | {avg_ev:>+7.4f}")

        print("-" * 70)

        # Totals
        total_wins = sum(r["wins"] for r in yearly_results)
        total_losses = sum(r["losses"] for r in yearly_results)
        total_n = sum(r["n"] for r in yearly_results)
        total_cover_rate = total_wins / total_n if total_n > 0 else 0
        total_roi = (total_cover_rate * (100 / 110) - (1 - total_cover_rate)) * 100
        total_avg_ev = df["ev"].mean()

        wlp = f"{total_wins}-{total_losses}-0"
        print(f"{'TOTAL':>6} | {total_n:>6} | {wlp:>12} | {total_cover_rate*100:>7.1f}% | {total_roi:>+7.1f}% | {total_avg_ev:>+7.4f}")
        print("-" * 70)

        # ----- Strategy Comparison -----
        print("\nSTRATEGY COMPARISON (EV thresholds):")
        print("-" * 70)
        print(f"{'Strategy':<12} | {'N':>6} | {'W-L':>10} | {'ATS%':>8} | {'ROI%':>8} | {'Avg EV':>8}")
        print("-" * 70)

        strategies = [
            ("OLD_5pt", df["edge_abs"] >= 5.0),
            ("EV3+", df["ev"] >= 0.03),
            ("EV5+", df["ev"] >= 0.05),
            ("EV8+", df["ev"] >= 0.08),
        ]

        for name, mask in strategies:
            subset = df[mask]
            n = len(subset)
            if n == 0:
                print(f"{name:<12} | {'N/A':>6} | No bets")
                continue

            wins = subset["jp_side_covered"].sum()
            losses = n - wins
            cover_rate = wins / n
            roi = (cover_rate * (100 / 110) - (1 - cover_rate)) * 100
            avg_ev = subset["ev"].mean()

            wl = f"{int(wins)}-{int(losses)}"
            print(f"{name:<12} | {n:>6} | {wl:>10} | {cover_rate*100:>7.1f}% | {roi:>+7.1f}% | {avg_ev:>+7.4f}")

        print("-" * 70)

        # ----- Push Rate Summary -----
        if wf_result.push_rate_summaries:
            print("\nPUSH RATE SUMMARY (per fold):")
            print("-" * 70)
            print(f"{'Fold':>6} | {'Default Even':>12} | {'Overall':>10} | Key Ticks (3/7/10/14)")
            print("-" * 70)
            for prs in wf_result.push_rate_summaries:
                key_rates = prs.get("key_tick_rates", {})
                key_str = ", ".join(f"{t//2}:{key_rates.get(t, 0):.2%}" for t in KEY_TICKS)
                print(f"{prs['eval_year']:>6} | {prs['default_even']:>11.2%} | {prs['default_overall']:>9.2%} | {key_str}")
            print("-" * 70)

        # ----- Stratified Diagnostics -----
        strat_diag = stratified_diagnostics(wf_result.game_results)
        print_stratified_diagnostics(strat_diag)

        # ----- Save results -----
        results_csv = output_path / f"backtest_{mode}_{start_year}-{end_year}.csv"
        df.to_csv(results_csv, index=False)
        print(f"\nSaved results to: {results_csv}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


def run_diagnose_full(
    csv_path: str,
    start_year: int = 2022,
    end_year: int = 2025,
) -> None:
    """Run full diagnostics including stratified analysis.

    Args:
        csv_path: Path to exported backtest CSV
        start_year: First year
        end_year: Last year
    """
    print(f"\n{'=' * 80}")
    print("FULL CALIBRATION DIAGNOSTICS")
    print("=" * 80)

    # Load and normalize data
    raw_df = load_backtest_data(start_year, end_year, csv_path)
    normalized_df = load_and_normalize_game_data(raw_df, jp_convention="pos_home_favored")
    print(f"Loaded {len(normalized_df)} games total")

    # Run walk-forward with push modeling
    print("\nRunning walk-forward validation with push modeling...")
    wf_result = walk_forward_validate(
        normalized_df,
        min_train_seasons=2,
        exclude_covid=True,
        training_window_seasons=2,  # PRIMARY mode
        include_push_modeling=True,
    )

    # Basic calibration diagnostics
    diag = diagnose_calibration(wf_result.game_results)
    print_calibration_diagnostics(diag)

    # Fold stability
    stability = diagnose_fold_stability(wf_result.fold_summaries)
    print_fold_stability(stability)

    # Stratified diagnostics
    strat_diag = stratified_diagnostics(wf_result.game_results)
    print_stratified_diagnostics(strat_diag)

    # Push rate analysis
    if wf_result.push_rate_summaries:
        print("\n" + "=" * 80)
        print("PUSH RATE ANALYSIS")
        print("=" * 80)
        for prs in wf_result.push_rate_summaries:
            print(f"\nFold {prs['eval_year']}:")
            print(f"  Default even: {prs['default_even']:.4f}")
            print(f"  Default overall: {prs['default_overall']:.4f}")
            print(f"  Key tick rates: {prs.get('key_tick_rates', {})}")

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


def parse_training_window(value: str) -> int | None:
    """Parse training window from CLI (handle 'none' as None)."""
    if value.lower() == "none":
        return None
    return int(value)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calibrated Spread Betting Selection Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate with default ROLLING_2 mode
  python -m src.spread_selection.run_selection validate --csv data/spread_selection/ats_export.csv

  # Compare PRIMARY vs ULTRA modes
  python -m src.spread_selection.run_selection validate --compare-modes

  # Validate with INCLUDE_ALL (ultra) mode
  python -m src.spread_selection.run_selection validate --training-window-seasons none

  # Predict upcoming games
  python -m src.spread_selection.run_selection predict --emit-modes primary,ultra
"""
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Shared calibration arguments
    def add_calibration_args(p):
        p.add_argument(
            "--training-window-seasons",
            type=parse_training_window,
            default=DEFAULT_TRAINING_WINDOW_SEASONS,
            metavar="N|none",
            help=f"Training window size. 'none' = all prior seasons (ULTRA). "
                 f"Default: {DEFAULT_TRAINING_WINDOW_SEASONS} (PRIMARY/ROLLING_2)"
        )
        p.add_argument(
            "--calibration-label",
            type=str,
            default=None,
            help="Override calibration label (auto-derived if not set)"
        )
        p.add_argument(
            "--exclude-years",
            type=int,
            nargs="+",
            default=None,
            help="Years to exclude from training (e.g., --exclude-years 2022)"
        )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Run full validation pipeline")
    validate_parser.add_argument(
        "--start-year", type=int, default=2022, help="First year (default: 2022)"
    )
    validate_parser.add_argument(
        "--end-year", type=int, default=2025, help="Last year (default: 2025)"
    )
    validate_parser.add_argument(
        "--line-type", type=str, default="close", choices=["close", "open"],
        help="Which line to use (default: close)"
    )
    validate_parser.add_argument(
        "--csv", type=str, default=None, help="Path to exported backtest CSV"
    )
    validate_parser.add_argument(
        "--compare-modes", action="store_true",
        help="Compare PRIMARY (ROLLING_2) vs ULTRA (INCLUDE_ALL) modes"
    )
    validate_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    add_calibration_args(validate_parser)

    # Diagnose command (alias for validate)
    diagnose_parser = subparsers.add_parser("diagnose", help="Run diagnostics (alias for validate)")
    diagnose_parser.add_argument(
        "--start-year", type=int, default=2022, help="First year (default: 2022)"
    )
    diagnose_parser.add_argument(
        "--end-year", type=int, default=2025, help="Last year (default: 2025)"
    )
    diagnose_parser.add_argument(
        "--csv", type=str, default=None, help="Path to exported backtest CSV"
    )
    diagnose_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    add_calibration_args(diagnose_parser)

    # Sensitivity report command
    sensitivity_parser = subparsers.add_parser(
        "sensitivity",
        help="Compare include vs exclude 2022 from calibration training"
    )
    sensitivity_parser.add_argument(
        "--csv", type=str, default=None, help="Path to exported backtest CSV"
    )
    sensitivity_parser.add_argument(
        "--eval-years", type=int, nargs="+", default=[2024, 2025],
        help="Years to evaluate on (default: 2024 2025)"
    )
    sensitivity_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate predictions for upcoming games"
    )
    predict_parser.add_argument(
        "--slate-csv", type=str, required=True,
        help="Path to CSV with upcoming slate (game_id, home_team, away_team, jp_spread, vegas_spread)"
    )
    predict_parser.add_argument(
        "--historical-csv", type=str, default="data/spread_selection/ats_export.csv",
        help="Path to historical ATS CSV for calibration fitting"
    )
    predict_parser.add_argument(
        "--emit-modes", type=str, default="primary",
        help="Comma-separated modes to emit: 'primary', 'ultra', or 'primary,ultra' (default: primary)"
    )
    predict_parser.add_argument(
        "--primary-min-ev", type=float, default=0.03,
        help="Minimum EV threshold for PRIMARY mode (default: 0.03)"
    )
    predict_parser.add_argument(
        "--ultra-min-ev", type=float, default=0.05,
        help="Minimum EV threshold for ULTRA mode (default: 0.05)"
    )
    predict_parser.add_argument(
        "--year", type=int, default=None,
        help="Year for predictions (default: current year from slate)"
    )
    predict_parser.add_argument(
        "--week", type=int, default=None,
        help="Week for predictions (default: from slate)"
    )
    predict_parser.add_argument(
        "--output-dir", type=str, default="data/spread_selection/outputs",
        help="Output directory for prediction files"
    )
    predict_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    # Phase 1 SP+ Gate arguments (defaults OFF)
    predict_parser.add_argument(
        "--sp-gate", action="store_true", default=False,
        help="Enable Phase 1 SP+ gate (weeks 1-3 only)"
    )
    predict_parser.add_argument(
        "--sp-gate-mode", type=str, default="confirm_only",
        choices=["confirm_only", "veto_opposes", "confirm_or_neutral"],
        help="SP+ gate mode (default: confirm_only)"
    )
    predict_parser.add_argument(
        "--sp-gate-sp-edge-min", type=float, default=2.0,
        help="Minimum SP+ edge for CONFIRM category (default: 2.0)"
    )
    predict_parser.add_argument(
        "--sp-gate-jp-edge-min", type=float, default=5.0,
        help="Minimum JP+ edge for candidate (when basis=edge) (default: 5.0)"
    )
    predict_parser.add_argument(
        "--sp-gate-missing-behavior", type=str, default="treat_neutral",
        choices=["treat_neutral", "reject"],
        help="How to handle games without SP+ prediction (default: treat_neutral)"
    )
    predict_parser.add_argument(
        "--sp-gate-candidate-basis", type=str, default="ev",
        choices=["ev", "edge"],
        help="Candidate selection basis: 'ev' for EV-based, 'edge' for edge-based (default: ev)"
    )

    # Backtest command (V2)
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run walk-forward backtest with push modeling and P&L analysis"
    )
    backtest_parser.add_argument(
        "--csv", type=str, default="data/spread_selection/ats_export.csv",
        help="Path to exported backtest CSV"
    )
    backtest_parser.add_argument(
        "--start-year", type=int, default=2022, help="First year (default: 2022)"
    )
    backtest_parser.add_argument(
        "--end-year", type=int, default=2025, help="Last year (default: 2025)"
    )
    backtest_parser.add_argument(
        "--output-dir", type=str, default="data/spread_selection/outputs",
        help="Output directory for results"
    )
    backtest_parser.add_argument(
        "--compare-modes", action="store_true", default=True,
        help="Compare PRIMARY and ULTRA modes (default: True)"
    )
    backtest_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.command == "validate":
        run_validate(
            args.start_year,
            args.end_year,
            args.line_type,
            args.csv,
            training_window_seasons=args.training_window_seasons,
            calibration_label=args.calibration_label,
            exclude_years=args.exclude_years,
            compare_modes=args.compare_modes,
        )
    elif args.command == "diagnose":
        # V2: Use run_diagnose_full for comprehensive stratified diagnostics
        if args.csv:
            run_diagnose_full(
                args.csv,
                args.start_year,
                args.end_year,
            )
        else:
            # Fallback to original validate behavior
            run_validate(
                args.start_year,
                args.end_year,
                "close",
                args.csv,
                training_window_seasons=args.training_window_seasons,
                calibration_label=args.calibration_label,
                exclude_years=args.exclude_years,
            )
    elif args.command == "sensitivity":
        run_sensitivity_report(args.csv, args.eval_years)
    elif args.command == "backtest":
        run_backtest(
            csv_path=args.csv,
            start_year=args.start_year,
            end_year=args.end_year,
            output_dir=args.output_dir,
            compare_modes=args.compare_modes,
        )
    elif args.command == "predict":
        # Build SP+ gate config
        sp_gate_config = None
        if args.sp_gate:
            sp_gate_config = Phase1SPGateConfig(
                enabled=True,
                mode=args.sp_gate_mode,
                sp_edge_min=args.sp_gate_sp_edge_min,
                jp_edge_min=args.sp_gate_jp_edge_min,
                missing_sp_behavior=args.sp_gate_missing_behavior,
                candidate_basis=args.sp_gate_candidate_basis,
            )

        run_predict(
            slate_csv=args.slate_csv,
            historical_csv=args.historical_csv,
            emit_modes=args.emit_modes.split(","),
            primary_min_ev=args.primary_min_ev,
            ultra_min_ev=args.ultra_min_ev,
            year=args.year,
            week=args.week,
            output_dir=args.output_dir,
            sp_gate_config=sp_gate_config,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
