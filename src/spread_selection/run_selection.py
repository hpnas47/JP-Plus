#!/usr/bin/env python3
"""CLI for calibrated spread betting selection validation.

Usage:
    python -m src.spread_selection.run_selection validate --start-year 2022 --end-year 2025
    python -m src.spread_selection.run_selection diagnose --start-year 2022 --end-year 2025
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.spread_selection.calibration import (
    CalibrationResult,
    load_and_normalize_game_data,
    calibrate_cover_probability,
    predict_cover_probability,
    walk_forward_validate,
    breakeven_prob,
    calculate_ev,
    calculate_ev_vectorized,
    diagnose_calibration,
    diagnose_fold_stability,
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

    # Calculate EV for each game
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
        print(f"  Overlap:             {oa['overlap_old_ev3']} bets ({oa['overlap_old_ev3']/oa['old_total']*100:.1f}% of OLD)")
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
    print("-" * 80)
    print(f"{'Year':>6} | {'N Train':>8} | {'Intercept':>10} | {'Slope':>8} | {'P(c)@0':>8} | {'BE Edge':>8}")
    print("-" * 80)

    for f in stability["folds"]:
        print(
            f"{f['eval_year']:>6} | {f['n_train']:>8} | {f['intercept']:>10.4f} | "
            f"{f['slope']:>8.4f} | {f['p_cover_at_zero']:>8.3f} | {f['breakeven_edge']:>8.2f}"
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
) -> None:
    """Run full validation pipeline.

    Args:
        start_year: First year
        end_year: Last year
        line_type: "close" or "open"
        csv_path: Path to exported backtest CSV
    """
    print(f"\nCalibrated Spread Selection Validation")
    print(f"Years: {start_year}-{end_year}, Line type: {line_type}")
    print("=" * 80)

    # Step 1: Load data
    raw_df = load_backtest_data(start_year, end_year, csv_path)

    # Step 2: Normalize
    normalized_df = load_and_normalize_game_data(raw_df, jp_convention="pos_home_favored")

    # Step 3: Print sample games for manual verification
    print_sample_games(normalized_df, n=10)

    # Step 4: Walk-forward validation
    print("\nRunning walk-forward validation...")
    wf_result = walk_forward_validate(normalized_df, min_train_seasons=2, exclude_covid=True)

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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calibrated Spread Betting Selection Validation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

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
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )

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

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.command == "validate":
        run_validate(args.start_year, args.end_year, args.line_type, args.csv)
    elif args.command == "diagnose":
        run_validate(args.start_year, args.end_year, "close", args.csv)
    elif args.command == "sensitivity":
        run_sensitivity_report(args.csv, args.eval_years)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
