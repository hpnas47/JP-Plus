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
from src.spread_selection.strategies.phase1_edge_baseline import (
    Phase1EdgeBaselineConfig,
    Phase1EdgeVetoConfig,
    Phase1EdgeRecommendation,
    Phase1EdgeResult,
    evaluate_slate_edge_baseline,
    recommendations_to_dataframe as edge_baseline_to_dataframe,
    summarize_recommendations as baseline_summarize_recommendations,
)

logger = logging.getLogger(__name__)


def _to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


# Team abbreviations for display formatting
TEAM_ABBREVS = {
    "Florida International": "FIU",
    "St. Francis (PA)": "St. Francis",
    "East Tennessee State": "ETSU",
    "Oklahoma State": "Okla St",
    "Sacramento State": "Sac State",
    "Western Michigan": "W. Michigan",
    "South Alabama": "S. Alabama",
    "Southern Illinois": "S. Illinois",
    "Bethune-Cookman": "B-Cookman",
    "East Texas A&M": "E. Texas A&M",
    "North Carolina": "UNC",
    "North Carolina A&T": "NC A&T",
    "Western Kentucky": "WKU",
    "Western Carolina": "W. Carolina",
    "Northern Illinois": "N. Illinois",
    "Eastern Washington": "E. Washington",
    "Jacksonville State": "Jax State",
    "Florida Atlantic": "FAU",
    "Washington State": "Wash St",
    "Central Michigan": "C. Michigan",
    "Mississippi State": "Miss St",
}


def _abbreviate_team(team: str) -> str:
    """Abbreviate team name for display."""
    return TEAM_ABBREVS.get(team, team)


def _format_matchup(away_team: str, home_team: str, max_len: int = 28) -> str:
    """Format matchup as 'Away @ Home', truncated to max_len."""
    away_abbrev = _abbreviate_team(away_team)
    home_abbrev = _abbreviate_team(home_team)
    matchup = f"{away_abbrev} @ {home_abbrev}"
    if len(matchup) > max_len:
        matchup = matchup[:max_len - 3] + "..."
    return matchup


def _format_bet_line(bet_team: str, spread: float) -> str:
    """Format bet as 'Team +/-X.X'."""
    team_abbrev = _abbreviate_team(bet_team)
    return f"{team_abbrev} {spread:+.1f}"


def format_primary_engine_table(
    bets_df: pd.DataFrame,
    results_df: Optional[pd.DataFrame] = None,
) -> str:
    """Format Primary Edge Execution Engine bets as display table.

    Args:
        bets_df: DataFrame with bets (columns: game_id, home_team, away_team,
                 edge_abs, ev, jp_favored_side, vegas_spread)
        results_df: Optional DataFrame with actual results (columns: game_id,
                    actual_margin, home_covered)

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 95)
    lines.append("Primary Edge Execution Engine (EV >= 3%)")
    lines.append("=" * 95)
    lines.append("Selection: EV calibrated on close | Bet execution: OPEN line")
    lines.append("-" * 95)
    lines.append(f"{'#':<3} {'Matchup':<30} | {'Edge':>6} | {'~EV':>6} | {'Bet (Open)':<18} | {'Result':<8}")
    lines.append("-" * 95)

    for i, (_, row) in enumerate(bets_df.iterrows(), 1):
        matchup = _format_matchup(row["away_team"], row["home_team"])

        # Determine bet team and spread for display
        if row["jp_favored_side"] == "HOME":
            bet_team = row["home_team"]
            # For home bets, use the negative of vegas_spread (home is favored)
            bet_spread = row.get("spread_open", row.get("vegas_spread", 0))
        else:
            bet_team = row["away_team"]
            # For away bets, use positive spread
            bet_spread = -row.get("spread_open", row.get("vegas_spread", 0))

        bet_line = _format_bet_line(bet_team, bet_spread)

        # Result (if available)
        result_str = ""
        if results_df is not None:
            game_result = results_df[results_df["game_id"] == row["game_id"]]
            if len(game_result) > 0:
                result_row = game_result.iloc[0]
                # Determine if bet covered
                if row["jp_favored_side"] == "HOME":
                    covered = result_row.get("home_covered", False)
                else:
                    # Away bet: cover_margin = actual_margin + vegas_spread
                    # If away bet, we bet on away, so: -actual_margin > spread_open
                    actual_margin = result_row.get("actual_margin", 0)
                    spread_open = row.get("spread_open", row.get("vegas_spread", 0))
                    covered = (-actual_margin) > spread_open
                result_str = "✓ WIN" if covered else "✗ LOSS"

        lines.append(f"{i:<3} {matchup:<30} | {row['edge_abs']:>5.1f}p | {row['ev']:>+5.2f} | {bet_line:<18} | {result_str:<8}")

    lines.append("-" * 95)
    return "\n".join(lines)


def format_phase1_edge_table(
    recs: list,
    results_df: Optional[pd.DataFrame] = None,
) -> str:
    """Format Phase 1 Edge >= 5.0 bets as display table.

    Args:
        recs: List of Phase1EdgeRecommendation objects
        results_df: Optional DataFrame with actual results

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 85)
    lines.append("Phase 1 Edge >= 5.0 vs Open (excluding Primary Engine)")
    lines.append("=" * 85)
    lines.append("Selection: |edge| >= 5.0 vs OPEN, NOT in Primary | Bet execution: OPEN line")
    lines.append("-" * 85)
    lines.append(f"{'#':<3} {'Matchup':<30} | {'Edge':>6} | {'Bet (Open)':<18} | {'Result':<8}")
    lines.append("-" * 85)

    for i, rec in enumerate(recs, 1):
        matchup = _format_matchup(rec.away_team, rec.home_team)

        # Determine bet spread for display
        if rec.bet_side == "HOME":
            bet_spread = rec.vegas_spread
        else:
            bet_spread = -rec.vegas_spread

        bet_line = _format_bet_line(rec.bet_team, bet_spread)

        # Result (if available)
        result_str = ""
        if results_df is not None:
            game_result = results_df[results_df["game_id"] == rec.game_id]
            if len(game_result) > 0:
                result_row = game_result.iloc[0]
                actual_margin = result_row.get("actual_margin", 0)
                if rec.bet_side == "HOME":
                    covered = actual_margin > -rec.vegas_spread
                else:
                    covered = (-actual_margin) > rec.vegas_spread
                result_str = "✓ WIN" if covered else "✗ LOSS"

        lines.append(f"{i:<3} {matchup:<30} | {rec.edge_abs:>5.1f}p | {bet_line:<18} | {result_str:<8}")

    lines.append("-" * 85)
    return "\n".join(lines)


def format_week_summary(
    week: int,
    primary_bets: int,
    primary_record: tuple[int, int],
    phase1_edge_bets: int,
    phase1_edge_record: tuple[int, int],
    total_fbs_games: int,
) -> str:
    """Format week summary block.

    Args:
        week: Week number
        primary_bets: Number of Primary Engine bets
        primary_record: (wins, losses) tuple for Primary
        phase1_edge_bets: Number of Phase 1 Edge bets
        phase1_edge_record: (wins, losses) tuple for Phase 1 Edge
        total_fbs_games: Total FBS vs FBS games that week

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 95)
    lines.append(f"WEEK {week} SUMMARY")
    lines.append("=" * 95)

    # Primary Engine
    p_w, p_l = primary_record
    p_total = p_w + p_l
    p_pct = 100 * p_w / p_total if p_total > 0 else 0
    lines.append(f"Primary Engine (EV >= 3%):           {primary_bets:>2} bets | {p_w}-{p_l} ({p_pct:.1f}% ATS)")

    # Phase 1 Edge
    e_w, e_l = phase1_edge_record
    e_total = e_w + e_l
    e_pct = 100 * e_w / e_total if e_total > 0 else 0
    lines.append(f"Phase 1 Edge >= 5.0 (non-Primary):   {phase1_edge_bets:>2} bets | {e_w}-{e_l} ({e_pct:.1f}% ATS)")

    # Combined
    combined_bets = primary_bets + phase1_edge_bets
    combined_w = p_w + e_w
    combined_l = p_l + e_l
    combined_total = combined_w + combined_l
    combined_pct = 100 * combined_w / combined_total if combined_total > 0 else 0
    lines.append(f"Combined (all edge >= 5.0):          {combined_bets:>2} bets | {combined_w}-{combined_l} ({combined_pct:.1f}% ATS)")

    lines.append(f"Total games that week:               {total_fbs_games:>2}")
    lines.append("=" * 95)

    return "\n".join(lines)


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

    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Backtest failed: {result.stderr}")
            raise RuntimeError(f"Backtest failed: {result.stderr}")

        # Load the exported CSV
        df = pd.read_csv(export_path)
        return df
    finally:
        # Clean up temp file even on failure
        Path(export_path).unlink(missing_ok=True)


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

        # Calculate EV (preserve existing if valid, same pattern as compare_strategies)
        df = wf_result.game_results.copy()
        df = df[df["p_cover_no_push"].notna() & ~df["push"]]
        if "ev" not in df.columns or df["ev"].isna().all():
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

    # Align folds by eval year instead of positional index
    inc_folds_by_year = {f["year"]: f for f in r_inc["fold_details"]}
    exc_folds_by_year = {f["year"]: f for f in r_exc["fold_details"]}
    common_years = sorted(set(inc_folds_by_year.keys()) & set(exc_folds_by_year.keys()))

    for eval_year in common_years:
        fold_inc = inc_folds_by_year[eval_year]
        fold_exc = exc_folds_by_year[eval_year]
        print(f"| {eval_year:>9} | {fold_inc['n_train']:>7} | {fold_inc['slope']:.6f} | "
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
    emit_phase1_edge_list: bool = True,
    phase1_edge_jp_min: float = 5.0,
    phase1_edge_veto_config: Optional[Phase1EdgeVetoConfig] = None,
    fbs_only: bool = True,
    distinct_lists: bool = True,
) -> None:
    """Generate predictions for upcoming games.

    Emits TWO separate bet lists in Phase 1 (weeks 1-3):
    - LIST A (ENGINE_EV PRIMARY/ULTRA): EV-based recommendations from calibrated engine
    - LIST B (PHASE1_EDGE): Edge-based Phase 1 list (auto-emitted in weeks 1-3)

    LIST A is the official engine output (execution_default=True for PRIMARY).
    LIST B is for visibility only (execution_default=False, is_official_engine=False).

    Args:
        slate_csv: Path to CSV with upcoming slate
        historical_csv: Path to historical ATS CSV for calibration
        emit_modes: List of modes to emit ("primary", "ultra")
        primary_min_ev: Minimum EV for PRIMARY mode bets
        ultra_min_ev: Minimum EV for ULTRA mode bets
        year: Year for predictions (default: infer from slate)
        week: Week for predictions (default: infer from slate)
        output_dir: Output directory for prediction files
        sp_gate_config: Optional Phase 1 SP+ gate for EV-based selection (default OFF)
        emit_phase1_edge_list: Whether to emit LIST B in Phase 1 (default True)
        phase1_edge_jp_min: JP+ edge threshold for LIST B (default 5.0)
        phase1_edge_veto_config: Optional HYBRID_VETO_2 config for LIST B (default OFF)
        fbs_only: If True (default), exclude FCS games from bet recommendations
        distinct_lists: If True (default), Phase 1 Edge excludes games in Primary Engine
    """
    from scipy.special import expit
    from src.api.cfbd_client import CFBDClient

    # Create client once for reuse across FBS filtering and SP+ spread fetching
    client = CFBDClient()

    print(f"\n{'=' * 80}")
    print("SPREAD SELECTION PREDICTION")
    print("=" * 80)
    print(f"LIST A (ENGINE_EV): {', '.join(emit_modes).upper()}")
    print(f"  PRIMARY min EV: {primary_min_ev:.2f}")
    print(f"  ULTRA min EV:   {ultra_min_ev:.2f}")
    if sp_gate_config and sp_gate_config.enabled:
        print(f"  SP+ Gate: ENABLED (mode={sp_gate_config.mode}) -- NOT RECOMMENDED")
    else:
        print(f"  SP+ Gate: DISABLED (recommended)")
    print(f"LIST B (PHASE1_EDGE): {'AUTO in weeks 1-3' if emit_phase1_edge_list else 'DISABLED'}")
    if emit_phase1_edge_list:
        print(f"  JP+ edge threshold: {phase1_edge_jp_min}")
        if phase1_edge_veto_config and phase1_edge_veto_config.enabled:
            print(f"  HYBRID_VETO_2: ENABLED (sp_oppose>={phase1_edge_veto_config.sp_oppose_min}, "
                  f"jp_band=[{phase1_edge_veto_config.jp_band_low},{phase1_edge_veto_config.jp_band_high}))")
        else:
            print(f"  HYBRID_VETO_2: DISABLED (default)")
    print(f"FBS-only filter: {'ENABLED' if fbs_only else 'DISABLED'}")
    print(f"Distinct lists: {'ENABLED' if distinct_lists else 'DISABLED'}")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load slate
    print(f"\nLoading slate from {slate_csv}...")
    slate_df = pd.read_csv(slate_csv)
    print(f"  Loaded {len(slate_df)} games")

    # FBS-only filter
    fbs_teams = set()
    if fbs_only:
        try:
            # Get year from slate or current year
            filter_year = year if year else (int(slate_df["year"].iloc[0]) if "year" in slate_df.columns else datetime.now().year)
            fbs_list = client.get_fbs_teams(year=filter_year)
            fbs_teams = {t.school for t in fbs_list}
            logger.info(f"[PREDICT] Fetched {len(fbs_teams)} FBS teams for {filter_year}")

            # Filter to FBS vs FBS games only
            pre_filter_count = len(slate_df)
            fbs_mask = slate_df["home_team"].isin(fbs_teams) & slate_df["away_team"].isin(fbs_teams)
            slate_df = slate_df[fbs_mask].copy()
            filtered_count = pre_filter_count - len(slate_df)
            if filtered_count > 0:
                print(f"  FBS filter: removed {filtered_count} FCS games, {len(slate_df)} FBS vs FBS remaining")
        except Exception as e:
            logger.warning(f"Failed to fetch FBS teams for filtering: {e}. Proceeding with all games.")
            print(f"  WARNING: FBS filter unavailable ({e}), using all games")

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

    # Track primary bets for distinct_lists filtering in LIST B
    primary_bets = None

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

            # Fetch SP+ spreads (reuse client from top of function)
            try:
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
                # Build set of non-PASS game_ids with consistent type (int)
                non_pass_game_ids = set(
                    int(gid) for gid in slate_pred[slate_pred["tier"] != "PASS"]["game_id"]
                )
                # Count categories only for gate results whose game_id is in the non-PASS set
                cats = Counter(
                    r.category.value for r in gate_results
                    if r.game_id in non_pass_game_ids
                )
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

        # Capture primary bets for distinct_lists filtering
        if mode == "primary":
            primary_bets = bets.copy()

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

        # Add required metadata fields to bets DataFrame (LIST A schema)
        is_primary = (mode == "primary")
        bets["list_family"] = "ENGINE_EV"
        bets["list_name"] = mode.upper()  # "PRIMARY" or "ULTRA"
        bets["selection_basis"] = "EV"
        bets["is_official_engine"] = is_primary
        bets["execution_default"] = is_primary
        bets["line_type"] = "close"  # List A uses close lines for EV
        bets["rationale"] = bets["ev"].apply(lambda x: f"EV>={min_ev:.2f}" if x >= min_ev else "PASS")

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
            json.dump(bets_json, f, indent=2, default=_to_native)
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
            json.dump(slate_json, f, indent=2, default=_to_native)
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

    # =========================================================================
    # LIST B: PHASE1_EDGE (Edge-based, auto-emitted in weeks 1-3)
    # This is SEPARATE from LIST A (EV-based) and does NOT use EV selection
    # =========================================================================
    list_b_selected = []
    list_b_vetoed = []
    list_b_candidates = []
    list_b_sp_spreads = {}

    phase1_edge_config = Phase1EdgeBaselineConfig(
        weeks=[1, 2, 3],
        jp_edge_min=phase1_edge_jp_min,
        line_type="open",
        veto_config=phase1_edge_veto_config,
    )

    if emit_phase1_edge_list and week in [1, 2, 3]:
        print(f"\n{'=' * 80}")
        print("LIST B: PHASE1_EDGE (auto-emitted in Phase 1)")
        print("=" * 80)
        veto_label = "ENABLED" if (phase1_edge_veto_config and phase1_edge_veto_config.enabled) else "DISABLED"
        print(f"Selection: |edge| >= {phase1_edge_jp_min} (OPEN line)")
        print(f"HYBRID_VETO_2: {veto_label}")
        print("execution_default=False, is_official_engine=False")
        print("=" * 80)

        # Fetch SP+ spreads for monitoring (and veto if enabled) - reuse client
        try:
            list_b_sp_spreads = fetch_sp_spreads_vegas(client, year, [week])
            print(f"  Fetched {len(list_b_sp_spreads)} SP+ spreads for week {week}")
        except Exception as e:
            logger.warning(f"Failed to fetch SP+ spreads for LIST B: {e}")
            list_b_sp_spreads = {}

        # Prepare slate for edge evaluation (use OPEN line)
        list_b_slate = slate_df.copy()
        if "season" not in list_b_slate.columns:
            list_b_slate["season"] = year

        # Run Phase1 Edge Baseline evaluation
        list_b_selected, list_b_vetoed, list_b_candidates = evaluate_slate_edge_baseline(
            list_b_slate,
            list_b_sp_spreads,
            phase1_edge_config,
        )

        # Apply distinct lists filter: exclude games already in Primary Engine
        if distinct_lists and primary_bets is not None and len(primary_bets) > 0:
            primary_game_ids = set(primary_bets['game_id'].values)
            pre_distinct_count = len(list_b_selected)
            list_b_selected = [rec for rec in list_b_selected if rec.game_id not in primary_game_ids]
            list_b_vetoed = [rec for rec in list_b_vetoed if rec.game_id not in primary_game_ids]
            list_b_candidates = [rec for rec in list_b_candidates if rec.game_id not in primary_game_ids]
            excluded_count = pre_distinct_count - len(list_b_selected)
            if excluded_count > 0:
                print(f"  Distinct lists: excluded {excluded_count} games already in Primary Engine")

        print(f"\n  Results (PHASE1_EDGE):")
        print(f"    Candidates (|edge| >= {phase1_edge_jp_min}): {len(list_b_candidates)}")
        print(f"    Selected: {len(list_b_selected)}")
        if phase1_edge_veto_config and phase1_edge_veto_config.enabled:
            print(f"    Vetoed by HYBRID_VETO_2: {len(list_b_vetoed)}")

        # Convert to DataFrame and save
        if list_b_selected or list_b_vetoed:
            # Include both selected and vetoed in output (vetoed marked as such)
            all_recs_for_output = list_b_selected + list_b_vetoed
            list_b_df = edge_baseline_to_dataframe(all_recs_for_output)

            # Determine list name based on veto config
            list_name_suffix = "edge_baseline" if not (phase1_edge_veto_config and phase1_edge_veto_config.enabled) else "edge_hybrid_veto_2"

            # Write LIST B outputs
            list_b_csv_path = output_path / f"bets_phase1_{list_name_suffix}_{year}_week{week}.csv"
            list_b_df.to_csv(list_b_csv_path, index=False)
            print(f"  Wrote: {list_b_csv_path}")

            list_b_json_path = output_path / f"bets_phase1_{list_name_suffix}_{year}_week{week}.json"
            list_b_json = {
                "metadata": {
                    "list_family": "PHASE1_EDGE",
                    "list_name": "EDGE_BASELINE" if not (phase1_edge_veto_config and phase1_edge_veto_config.enabled) else "EDGE_HYBRID_VETO_2",
                    "selection_basis": "EDGE",
                    "is_official_engine": False,
                    "execution_default": False,
                    "line_type": "open",
                    "jp_edge_min": phase1_edge_jp_min,
                    "veto_enabled": phase1_edge_veto_config.enabled if phase1_edge_veto_config else False,
                    "veto_sp_oppose_min": phase1_edge_veto_config.sp_oppose_min if phase1_edge_veto_config else None,
                    "veto_jp_band_high": phase1_edge_veto_config.jp_band_high if phase1_edge_veto_config else None,
                    "year": year,
                    "week": week,
                    "n_candidates": len(list_b_candidates),
                    "n_selected": len(list_b_selected),
                    "n_vetoed": len(list_b_vetoed),
                    "n_sp_spreads_fetched": len(list_b_sp_spreads),
                    "generated_at": datetime.now().isoformat(),
                },
                "bets": list_b_df.to_dict(orient="records"),
            }
            with open(list_b_json_path, "w") as f:
                json.dump(list_b_json, f, indent=2, default=_to_native)
            print(f"  Wrote: {list_b_json_path}")

            # Print bet summary (selected only)
            if list_b_selected:
                print(f"\n  TOP BETS (PHASE1_EDGE - selected):")
                print(f"  {'-' * 75}")
                print(f"  {'Game':<40} | {'Side':<6} | {'Edge':>7} | {'SP+ Edge':>8} | {'Veto':>5}")
                print(f"  {'-' * 75}")
                for rec in list_b_selected[:10]:
                    game = f"{rec.away_team} @ {rec.home_team}"
                    if len(game) > 38:
                        game = game[:35] + "..."
                    sp_edge_str = f"{rec.sp_edge_pts:+.1f}" if rec.sp_edge_pts is not None else "N/A"
                    veto_str = "YES" if rec.veto_applied else "NO"
                    print(f"  {game:<40} | {rec.bet_side:<6} | {rec.edge_pts:>+6.1f}p | {sp_edge_str:>8} | {veto_str:>5}")
        else:
            print(f"\n  No games meet edge threshold for LIST B")

    elif emit_phase1_edge_list and week not in [1, 2, 3]:
        print(f"\n  LIST B: Week {week} not in Phase 1 (weeks 1-3), skipping")

    # =========================================================================
    # WEEK SUMMARY + OVERLAP/CONFLICT REPORT (Phase 1 only)
    # =========================================================================
    if week in [1, 2, 3] and 'bets' in dir():
        # Get counts from primary mode
        engine_primary_count = len(bets) if 'primary' in emit_modes else 0
        engine_ultra_count = 0  # Will be updated if ultra was processed

        # Check if ultra was also processed (would need separate tracking)
        # For now, use emit_modes to infer

        phase1_edge_count = len(list_b_selected)
        phase1_vetoed_count = len(list_b_vetoed)

        # Write week summary JSON
        week_summary = {
            "year": year,
            "week": week,
            "generated_at": datetime.now().isoformat(),
            "engine_ev_primary_count": engine_primary_count,
            "engine_ev_ultra_count": engine_ultra_count,
            "phase1_edge_baseline_count": phase1_edge_count,
            "phase1_edge_vetoed_count": phase1_vetoed_count,
            "config": {
                "primary_min_ev": primary_min_ev,
                "ultra_min_ev": ultra_min_ev,
                "phase1_edge_jp_min": phase1_edge_jp_min,
                "phase1_edge_veto_enabled": phase1_edge_veto_config.enabled if phase1_edge_veto_config else False,
                "emit_modes": emit_modes,
            },
            "files": {
                "lista_primary": f"bets_primary_{year}_week{week}.csv" if "primary" in emit_modes else None,
                "lista_ultra": f"bets_ultra_{year}_week{week}.csv" if "ultra" in emit_modes else None,
                "listb_phase1_edge": f"bets_phase1_edge_baseline_{year}_week{week}.csv" if list_b_selected or list_b_vetoed else None,
            },
        }

        week_summary_path = output_path / f"week_summary_{year}_week{week}.json"
        with open(week_summary_path, "w") as f:
            json.dump(week_summary, f, indent=2, default=_to_native)
        print(f"\n  Wrote: {week_summary_path}")

        # Overlap/conflict report
        if len(bets) > 0 and len(list_b_selected) > 0:
            print(f"\n{'=' * 80}")
            print("LIST A vs LIST B OVERLAP/CONFLICT REPORT")
            print("=" * 80)

            # Build overlap DataFrame
            lista_games = set(int(gid) for gid in bets['game_id'].values)
            lista_sides = {int(row['game_id']): row['jp_favored_side'] for _, row in bets.iterrows()}
            lista_evs = {int(row['game_id']): row['ev'] for _, row in bets.iterrows()}
            lista_confidence = {int(row['game_id']): row.get('tier', 'BET') for _, row in bets.iterrows()}

            listb_games = {rec.game_id for rec in list_b_selected}
            listb_sides = {rec.game_id: rec.bet_side for rec in list_b_selected}
            listb_edges = {rec.game_id: rec.edge_abs for rec in list_b_selected}
            listb_vetoed = {rec.game_id: rec.veto_applied for rec in list_b_selected}

            all_games = lista_games | listb_games
            overlap_records = []

            for game_id in all_games:
                in_a = game_id in lista_games
                in_b = game_id in listb_games
                side_a = lista_sides.get(game_id)
                side_b = listb_sides.get(game_id)
                ev_a = lista_evs.get(game_id)
                edge_b = listb_edges.get(game_id)
                conf_a = lista_confidence.get(game_id)
                veto_b = listb_vetoed.get(game_id, False)

                if in_a and in_b:
                    agrees = side_a == side_b
                    conflict = side_a != side_b
                else:
                    agrees = None
                    conflict = None

                # Recommended resolution (informational only)
                if conflict:
                    resolution = "REVIEW: Lists disagree on side"
                elif in_a and in_b and agrees:
                    resolution = "CONSENSUS: Both lists agree"
                elif in_a and not in_b:
                    resolution = "LIST_A_ONLY: EV-based pick"
                elif in_b and not in_a:
                    resolution = "LIST_B_ONLY: Edge-based pick"
                else:
                    resolution = "UNKNOWN"

                overlap_records.append({
                    'game_id': game_id,
                    'in_engine_primary': in_a,
                    'engine_side': side_a,
                    'engine_ev': ev_a,
                    'engine_confidence': conf_a,
                    'in_phase1_edge': in_b,
                    'phase1_side': side_b,
                    'phase1_edge_abs': edge_b,
                    'veto_applied': veto_b,
                    'side_agrees': agrees,
                    'conflict': conflict,
                    'recommended_resolution': resolution,
                })

            overlap_df = pd.DataFrame(overlap_records)

            # Compute summary stats
            n_both = len([r for r in overlap_records if r['in_engine_primary'] and r['in_phase1_edge']])
            n_agrees = len([r for r in overlap_records if r['side_agrees'] is True])
            n_conflicts = len([r for r in overlap_records if r['conflict'] is True])
            n_only_a = len([r for r in overlap_records if r['in_engine_primary'] and not r['in_phase1_edge']])
            n_only_b = len([r for r in overlap_records if not r['in_engine_primary'] and r['in_phase1_edge']])

            print(f"  Engine bets: {len(lista_games)}")
            print(f"  Phase1 edge bets: {len(listb_games)}")
            print(f"  Overlap: {n_both}, Conflicts: {n_conflicts}")
            print(f"    - Side agrees: {n_agrees}")
            print(f"    - Side CONFLICTS: {n_conflicts}")
            print(f"  Engine only: {n_only_a}")
            print(f"  Phase1 edge only: {n_only_b}")

            if n_conflicts > 0:
                print(f"\n  *** WARNING: {n_conflicts} CONFLICTS DETECTED ***")
                print(f"  The betting engine should review these games before placing bets.")
                conflict_games = [r for r in overlap_records if r['conflict'] is True]
                for cg in conflict_games[:5]:
                    print(f"    game_id={cg['game_id']}: Engine says {cg['engine_side']}, Phase1 says {cg['phase1_side']}")

            # Write overlap CSV
            overlap_csv_path = output_path / f"overlap_engine_primary_vs_phase1_edge_{year}_week{week}.csv"
            overlap_df.to_csv(overlap_csv_path, index=False)
            print(f"\n  Wrote: {overlap_csv_path}")

    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE")
    print("=" * 80)


def run_backtest(
    csv_path: str,
    start_year: int = 2022,
    end_year: int = 2025,
    output_dir: str = "data/spread_selection/outputs",
    compare_modes: bool = True,
    save_artifacts: bool = False,
    freeze_artifacts: bool = False,
    artifact_description: str = "",
    rebuild_history: bool = False,
) -> None:
    """Run walk-forward backtest with push modeling for P&L analysis.

    Args:
        csv_path: Path to exported backtest CSV
        start_year: First year
        end_year: Last year
        output_dir: Output directory for results
        compare_modes: If True, run both PRIMARY and ULTRA modes
        save_artifacts: If True, save calibration artifacts for reproducibility
        freeze_artifacts: If True, mark artifacts as frozen for production
        artifact_description: Description to include in artifact metadata
        rebuild_history: If True, allow overwriting existing outputs with identical inputs
    """
    from .artifacts import (
        ArtifactStore,
        RunLog,
        RunMetadata,
        create_artifact_from_walk_forward,
        compute_file_hash,
        get_git_commit_hash,
    )
    from .reproducibility import (
        create_fingerprint,
        ReproducibilityGuard,
    )
    from datetime import datetime

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    modes = ["primary", "ultra"] if compare_modes else ["primary"]

    # Check reproducibility for each mode BEFORE loading data
    guard = ReproducibilityGuard()
    fingerprints = {}
    modes_to_skip = []
    for mode in modes:
        config = CALIBRATION_MODES[mode]
        fingerprint = create_fingerprint(
            data_path=csv_path,
            command="backtest",
            calibration_mode=mode.upper(),
            start_year=start_year,
            end_year=end_year,
            training_seasons=config["training_window_seasons"],
        )
        fingerprints[mode] = fingerprint

        # Determine output path for this mode
        results_csv = output_path / f"backtest_{mode}_{start_year}-{end_year}.csv"

        # Print fingerprint header
        fingerprint.print_header()

        # Print change detection (per-output tracking)
        guard.print_change_detection(results_csv, fingerprint)
        should_proceed, message = guard.check_output_exists(
            results_csv, fingerprint, rebuild_history
        )
        if not should_proceed:
            print(f"\n*** SKIPPING {mode.upper()} ***")
            print(f"    {message}")
            print(f"    Use --rebuild-history to force regeneration.")
            modes_to_skip.append(mode)

    # Remove skipped modes
    modes = [m for m in modes if m not in modes_to_skip]

    if not modes:
        print("\n*** ALL MODES SKIPPED — no work to do ***")
        return

    # Load and normalize data (only if we have modes to process)
    print(f"\n{'=' * 80}")
    print("WALK-FORWARD BACKTEST WITH PUSH MODELING")
    print("=" * 80)
    print(f"Data: {csv_path}")
    print(f"Years: {start_year}-{end_year}")
    print("=" * 80)

    raw_df = load_backtest_data(start_year, end_year, csv_path)
    normalized_df = load_and_normalize_game_data(raw_df, jp_convention="pos_home_favored")
    print(f"\nLoaded {len(normalized_df)} games total")

    for mode in modes:
        config = CALIBRATION_MODES[mode]
        fingerprint = fingerprints[mode]
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

        # Get all games with predictions (including pushes for counting)
        all_games = wf_result.game_results[
            wf_result.game_results["p_cover_no_push"].notna()
        ].copy()

        # Filter to non-push games for W-L calculations
        df = all_games[~all_games["push"]].copy()

        print(f"\nGames with predictions: {len(all_games)} ({len(df)} decided, {len(all_games) - len(df)} pushes)")

        # ----- Per-Year P&L with pushes -----
        print("\nPER-YEAR P&L (with push modeling):")
        print("-" * 70)
        print(f"{'Year':>6} | {'N':>6} | {'W-L-P':>12} | {'ATS%':>8} | {'ROI%':>8} | {'Avg EV':>8}")
        print("-" * 70)

        yearly_results = []
        for year in sorted(all_games["year"].unique()):
            year_all = all_games[all_games["year"] == year]
            year_df = df[df["year"] == year]

            # Calculate W-L-P from actual data
            wins = int(year_df["jp_side_covered"].sum())
            n_decided = len(year_df)
            losses = n_decided - wins
            pushes = int(year_all["push"].sum())

            cover_rate = wins / n_decided if n_decided > 0 else 0

            # ROI at -110 (pushes don't affect ROI calculation)
            roi = (cover_rate * (100 / 110) - (1 - cover_rate)) * 100

            # Avg EV
            avg_ev = year_df["ev"].mean() if len(year_df) > 0 else 0

            yearly_results.append({
                "year": year,
                "n": n_decided,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "cover_rate": cover_rate,
                "roi": roi,
                "avg_ev": avg_ev,
            })

            wlp = f"{wins}-{losses}-{pushes}"
            print(f"{year:>6} | {n_decided:>6} | {wlp:>12} | {cover_rate*100:>7.1f}% | {roi:>+7.1f}% | {avg_ev:>+7.4f}")

        print("-" * 70)

        # Totals
        total_wins = sum(r["wins"] for r in yearly_results)
        total_losses = sum(r["losses"] for r in yearly_results)
        total_pushes = sum(r["pushes"] for r in yearly_results)
        total_n = sum(r["n"] for r in yearly_results)
        total_cover_rate = total_wins / total_n if total_n > 0 else 0
        total_roi = (total_cover_rate * (100 / 110) - (1 - total_cover_rate)) * 100
        total_avg_ev = df["ev"].mean() if len(df) > 0 else 0

        wlp = f"{total_wins}-{total_losses}-{total_pushes}"
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

        # ----- Verify determinism -----
        is_deterministic, det_message = guard.verify_determinism(results_csv, fingerprint)
        if is_deterministic:
            print(f"  {det_message}")
        else:
            print(f"\n*** {det_message} ***")
            # This is a critical error - identical inputs produced different outputs
            raise RuntimeError(det_message)

        # ----- Record run -----
        guard.record_run(fingerprint, results_csv)
        print(f"  Run recorded for reproducibility tracking")

        # ----- Save calibration artifacts -----
        if save_artifacts:
            print(f"\n--- Saving calibration artifacts for {mode} ---")
            artifact = create_artifact_from_walk_forward(
                wf_result=wf_result,
                mode=mode,
                ats_export_path=csv_path,
                description=artifact_description or f"Backtest {start_year}-{end_year}",
            )

            store = ArtifactStore()
            artifact_path = store.save(artifact)
            print(f"Saved artifact: {artifact.artifact_id}")
            print(f"  Path: {artifact_path}")
            print(f"  Data hash: {artifact.ats_export_hash}")
            print(f"  Git commit: {artifact.git_commit or 'N/A'}")

            if freeze_artifacts:
                store.freeze(artifact.artifact_id)
                print(f"  Status: FROZEN for production")
            else:
                print(f"  Status: Not frozen (use --freeze-artifacts to lock)")

            # Log run metadata
            run_log = RunLog()
            run_metadata = RunMetadata(
                timestamp=datetime.now().isoformat(),
                command=f"backtest --csv {csv_path} --start-year {start_year} --end-year {end_year}",
                calibration_mode=mode,
                artifact_id=artifact.artifact_id,
                years_range=(start_year, end_year),
                ats_export_hash=artifact.ats_export_hash,
                git_commit=artifact.git_commit,
                frozen_mode=freeze_artifacts,
                output_files=[str(results_csv), str(artifact_path)],
                notes=artifact_description,
            )
            run_log.append(run_metadata)
            print(f"  Run logged to: {run_log.log_path}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


def run_replay(
    artifact_id: str | None,
    mode: str,
    year: int,
    week: int | None,
    csv_path: str,
    verify_hash: bool,
    output_dir: str | None,
) -> None:
    """Replay historical results using frozen calibration artifacts.

    This provides deterministic reproduction of historical predictions
    using the exact calibration parameters from a saved artifact.

    Args:
        artifact_id: Artifact ID to use (None = latest for mode)
        mode: Calibration mode ("primary" or "ultra")
        year: Year to replay
        week: Specific week to replay (None = all weeks for year)
        csv_path: Path to ATS export CSV
        verify_hash: If True, verify data hash matches artifact
        output_dir: Output directory for replay results
    """
    from .artifacts import (
        ArtifactStore,
        verify_data_integrity,
        RunLog,
        RunMetadata,
        compute_file_hash,
        get_git_commit_hash,
    )
    from datetime import datetime
    import numpy as np

    print(f"\n{'=' * 80}")
    print("DETERMINISTIC REPLAY FROM FROZEN ARTIFACTS")
    print("=" * 80)

    # Load artifact
    store = ArtifactStore()
    if artifact_id:
        artifact = store.load(artifact_id)
    else:
        artifact = store.load_latest(mode)
        if artifact is None:
            print(f"ERROR: No artifacts found for mode '{mode}'")
            print("Run 'backtest --save-artifacts' first to create artifacts.")
            return

    print(f"Artifact: {artifact.artifact_id}")
    print(f"Mode: {artifact.calibration_mode}")
    print(f"Created: {artifact.created_at}")
    print(f"Frozen: {'Yes' if artifact.frozen else 'No'}")
    print(f"Data hash: {artifact.ats_export_hash[:16]}...")

    # Verify data integrity
    if verify_hash:
        is_valid, message = verify_data_integrity(artifact, csv_path)
        if not is_valid:
            print(f"\nWARNING: {message}")
            print("Results may differ from original due to data changes.")
            print("Use --no-verify-hash to proceed anyway.")
            return
        else:
            print(f"Data integrity: VERIFIED")

    # Get fold calibration for the target year
    fold_cal = artifact.get_fold_calibration(year)
    if fold_cal is None:
        print(f"\nERROR: No calibration found for year {year}")
        print(f"Available years: {[fc.eval_year for fc in artifact.fold_calibrations]}")
        return

    # Get push rates for the target year
    fold_push = artifact.get_fold_push_rates(year)

    print(f"\n--- Calibration for {year} ---")
    print(f"Slope: {fold_cal.slope:.6f}")
    print(f"Intercept: {fold_cal.intercept:.6f}")
    print(f"Breakeven edge: {fold_cal.breakeven_edge:.2f}")
    print(f"P(cover) at edge=0: {fold_cal.p_cover_at_zero:.4f}")
    print(f"Training years: {fold_cal.training_years}")
    print(f"Training games: {fold_cal.n_train_games}")

    # Load data
    raw_df = load_backtest_data(year, year, csv_path)
    normalized_df = load_and_normalize_game_data(raw_df, jp_convention="pos_home_favored")

    # Filter to specific week if requested
    if week is not None:
        normalized_df = normalized_df[normalized_df["week"] == week].copy()
        print(f"\nFiltered to week {week}: {len(normalized_df)} games")
    else:
        print(f"\nAll weeks for {year}: {len(normalized_df)} games")

    if len(normalized_df) == 0:
        print("No games found for the specified criteria.")
        return

    # Apply frozen calibration parameters
    def compute_ev_frozen(row):
        """Compute EV using frozen calibration parameters."""
        edge_abs = row["edge_abs"]
        logit = fold_cal.intercept + fold_cal.slope * edge_abs
        p_cover = 1 / (1 + np.exp(-logit))

        # Get push rate if available
        push_rate = 0.0
        if fold_push is not None:
            vegas_spread = row.get("vegas_spread_close", row.get("vegas_spread", 0))
            tick = int(abs(vegas_spread * 2)) if not np.isnan(vegas_spread) else 0
            push_rate = fold_push.tick_rates.get(tick, fold_push.default_overall)

        # Adjust for push
        p_cover_adj = p_cover * (1 - push_rate)
        p_lose = (1 - p_cover) * (1 - push_rate)

        # EV at -110
        ev = p_cover_adj * (100 / 110) - p_lose
        return ev, p_cover, push_rate

    # Compute EV for all games
    results = []
    for _, row in normalized_df.iterrows():
        ev, p_cover, push_rate = compute_ev_frozen(row)

        # Get jp_side from jp_favored_side or compute from edge_pts
        jp_side = row.get("jp_favored_side", None)
        if jp_side is None:
            edge_pts = row["edge_pts"]
            jp_side = "HOME" if edge_pts < 0 else "AWAY"

        results.append({
            "game_id": row["game_id"],
            "year": row["year"],
            "week": row["week"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "jp_spread": row.get("jp_spread", -row.get("predicted_spread", 0)),
            "vegas_spread": row.get("vegas_spread_close", row.get("vegas_spread", None)),
            "edge_pts": row["edge_pts"],
            "edge_abs": row["edge_abs"],
            "jp_side": jp_side,
            "p_cover": p_cover,
            "push_rate": push_rate,
            "ev": ev,
            "fold_slope": fold_cal.slope,
            "fold_intercept": fold_cal.intercept,
            "artifact_id": artifact.artifact_id,
        })

    results_df = pd.DataFrame(results)

    # Filter to EV >= 3% (Primary threshold)
    ev_threshold = 0.03 if mode == "primary" else 0.05
    selected = results_df[results_df["ev"] >= ev_threshold].copy()

    print(f"\n--- Results (EV >= {ev_threshold*100:.0f}%) ---")
    print(f"Total games: {len(results_df)}")
    print(f"Selected: {len(selected)}")

    if len(selected) > 0:
        # Sort by EV descending
        selected = selected.sort_values("ev", ascending=False)

        print(f"\n{'#':<3} | {'Matchup':<35} | {'Edge':>6} | {'~EV':>8} | {'JP+ Line':<15} | {'Bet':<20}")
        print("-" * 100)

        for i, (_, row) in enumerate(selected.iterrows(), 1):
            matchup = f"{row['away_team']} @ {row['home_team']}"
            if len(matchup) > 35:
                matchup = matchup[:32] + "..."

            # Determine bet side and JP+ line
            if row["jp_side"] == "HOME":
                bet_team = row["home_team"]
                jp_line = f"{bet_team} {row['jp_spread']:+.1f}"
                vegas_spread = row["vegas_spread"] if pd.notna(row["vegas_spread"]) else 0
                bet_line = f"{bet_team} {vegas_spread:+.1f}"
            else:
                bet_team = row["away_team"]
                jp_line = f"{bet_team} {-row['jp_spread']:+.1f}"
                vegas_spread = row["vegas_spread"] if pd.notna(row["vegas_spread"]) else 0
                bet_line = f"{bet_team} {-vegas_spread:+.1f}"

            print(f"{i:<3} | {matchup:<35} | {row['edge_abs']:>5.1f} | {row['ev']:>+7.2%} | {jp_line:<15} | {bet_line:<20}")

        print("-" * 100)
        print(f"Average EV: {selected['ev'].mean():+.2%}")

    # Save results if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        week_str = f"_week{week}" if week else ""
        output_file = output_path / f"replay_{mode}_{year}{week_str}_{artifact.artifact_id}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved replay results to: {output_file}")

    # Log replay run
    run_log = RunLog()
    run_metadata = RunMetadata(
        timestamp=datetime.now().isoformat(),
        command=f"replay --mode {mode} --year {year}" + (f" --week {week}" if week else ""),
        calibration_mode=mode,
        artifact_id=artifact.artifact_id,
        years_range=(year, year),
        ats_export_hash=compute_file_hash(csv_path),
        git_commit=get_git_commit_hash(),
        frozen_mode=artifact.frozen,
        output_files=[str(output_file)] if output_dir else [],
        notes=f"Replay of {year}" + (f" week {week}" if week else ""),
    )
    run_log.append(run_metadata)

    print("\n" + "=" * 80)
    print("REPLAY COMPLETE")
    print("=" * 80)


def run_list_artifacts(
    mode: str | None,
    verbose: bool = False,
) -> None:
    """List available calibration artifacts.

    Args:
        mode: Filter by calibration mode (None = all)
        verbose: If True, show artifact details
    """
    from .artifacts import ArtifactStore

    print(f"\n{'=' * 80}")
    print("CALIBRATION ARTIFACTS")
    print("=" * 80)

    store = ArtifactStore()
    artifact_ids = store.list_artifacts(mode)

    if not artifact_ids:
        print(f"No artifacts found" + (f" for mode '{mode}'" if mode else ""))
        print("\nRun 'backtest --save-artifacts' to create artifacts.")
        return

    print(f"Found {len(artifact_ids)} artifact(s)")
    print()

    if verbose:
        # Show full details for each artifact
        for artifact_id in artifact_ids:
            artifact = store.load(artifact_id)
            print(f"{'=' * 60}")
            print(f"Artifact ID: {artifact.artifact_id}")
            print(f"Mode: {artifact.calibration_mode}")
            print(f"Created: {artifact.created_at}")
            print(f"Frozen: {'Yes' if artifact.frozen else 'No'}")
            print(f"Years: {artifact.years_range[0]}-{artifact.years_range[1]}")
            print(f"Data hash: {artifact.ats_export_hash[:16]}...")
            print(f"Git commit: {artifact.git_commit or 'N/A'}")
            if artifact.description:
                print(f"Description: {artifact.description}")
            print(f"\nFold calibrations:")
            for fc in artifact.fold_calibrations:
                print(f"  {fc.eval_year}: slope={fc.slope:.6f}, intercept={fc.intercept:.6f}, "
                      f"breakeven={fc.breakeven_edge:.2f}, n={fc.n_train_games}")
            print()
    else:
        # Summary table
        print(f"{'ID':<30} | {'Mode':<8} | {'Years':<10} | {'Frozen':<7} | {'Created':<20}")
        print("-" * 85)
        for artifact_id in artifact_ids:
            artifact = store.load(artifact_id)
            years_str = f"{artifact.years_range[0]}-{artifact.years_range[1]}"
            frozen_str = "Yes" if artifact.frozen else "No"
            created_short = artifact.created_at[:19] if len(artifact.created_at) >= 19 else artifact.created_at
            print(f"{artifact_id:<30} | {artifact.calibration_mode:<8} | {years_str:<10} | {frozen_str:<7} | {created_short:<20}")
        print("-" * 85)

    print("\nUse --verbose for full details")
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
    # Phase 1 Edge Baseline (LIST B) - auto-emitted in weeks 1-3 by default
    predict_parser.add_argument(
        "--no-phase1-edge-list", action="store_true", default=False,
        help="Disable automatic emission of Phase 1 Edge list (LIST B) in weeks 1-3"
    )
    predict_parser.add_argument(
        "--phase1-edge-jp-min", type=float, default=5.0,
        help="JP+ edge threshold for Phase 1 Edge baseline (default: 5.0)"
    )
    # HYBRID_VETO_2 overlay for LIST B (default OFF)
    predict_parser.add_argument(
        "--phase1-edge-veto", action="store_true", default=False,
        help="Enable HYBRID_VETO_2 overlay for Phase 1 Edge list (default: OFF)"
    )
    predict_parser.add_argument(
        "--phase1-edge-veto-sp-oppose-min", type=float, default=2.0,
        help="SP+ opposition threshold for HYBRID_VETO_2 (default: 2.0)"
    )
    predict_parser.add_argument(
        "--phase1-edge-veto-jp-band-high", type=float, default=8.0,
        help="Upper bound of JP+ marginal band for veto eligibility (default: 8.0)"
    )
    # FBS-only filter (default ON)
    predict_parser.add_argument(
        "--no-fbs-only", action="store_true", default=False,
        help="Disable FBS-only filter (include FCS games in recommendations)"
    )
    # Distinct lists (default ON)
    predict_parser.add_argument(
        "--no-distinct-lists", action="store_true", default=False,
        help="Disable distinct lists (Phase 1 Edge may overlap with Primary Engine)"
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
        "--compare-modes", action="store_true",
        help="Compare PRIMARY and ULTRA modes (default: True)"
    )
    backtest_parser.set_defaults(compare_modes=True)
    backtest_parser.add_argument(
        "--save-artifacts", action="store_true", default=False,
        help="Save calibration artifacts for reproducibility"
    )
    backtest_parser.add_argument(
        "--freeze-artifacts", action="store_true", default=False,
        help="Mark saved artifacts as frozen for production use"
    )
    backtest_parser.add_argument(
        "--artifact-description", type=str, default="",
        help="Description to include in artifact metadata"
    )
    backtest_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    backtest_parser.add_argument(
        "--rebuild-history", action="store_true", default=False,
        help="Allow overwriting existing output files even if inputs are identical"
    )

    # Replay command - deterministic reproduction from artifacts
    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay historical results using frozen calibration artifacts"
    )
    replay_parser.add_argument(
        "--artifact-id", type=str, default=None,
        help="Artifact ID to use (default: latest for mode)"
    )
    replay_parser.add_argument(
        "--mode", type=str, default="primary", choices=["primary", "ultra"],
        help="Calibration mode (default: primary)"
    )
    replay_parser.add_argument(
        "--year", type=int, required=True,
        help="Year to replay"
    )
    replay_parser.add_argument(
        "--week", type=int, default=None,
        help="Specific week to replay (default: all weeks)"
    )
    replay_parser.add_argument(
        "--csv", type=str, default="data/spread_selection/ats_export.csv",
        help="Path to ATS export CSV"
    )
    replay_parser.add_argument(
        "--verify-hash", action="store_true", default=True,
        help="Verify data hash matches artifact (default: True)"
    )
    replay_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for replay results"
    )
    replay_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )

    # List artifacts command
    list_artifacts_parser = subparsers.add_parser(
        "list-artifacts",
        help="List available calibration artifacts"
    )
    list_artifacts_parser.add_argument(
        "--mode", type=str, default=None, choices=["primary", "ultra"],
        help="Filter by calibration mode"
    )
    list_artifacts_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show artifact details"
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
            save_artifacts=args.save_artifacts,
            freeze_artifacts=args.freeze_artifacts,
            artifact_description=args.artifact_description,
            rebuild_history=args.rebuild_history,
        )
    elif args.command == "replay":
        run_replay(
            artifact_id=args.artifact_id,
            mode=args.mode,
            year=args.year,
            week=args.week,
            csv_path=args.csv,
            verify_hash=args.verify_hash,
            output_dir=args.output_dir,
        )
    elif args.command == "list-artifacts":
        run_list_artifacts(
            mode=args.mode,
            verbose=getattr(args, "verbose", False),
        )
    elif args.command == "predict":
        # Build SP+ gate config (applies to EV-based LIST A)
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

        # Build Phase1 Edge Veto config (HYBRID_VETO_2 - default OFF)
        phase1_edge_veto_config = None
        if args.phase1_edge_veto:
            phase1_edge_veto_config = Phase1EdgeVetoConfig(
                enabled=True,
                sp_oppose_min=args.phase1_edge_veto_sp_oppose_min,
                jp_band_low=args.phase1_edge_jp_min,  # Use same as edge min
                jp_band_high=args.phase1_edge_veto_jp_band_high,
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
            emit_phase1_edge_list=not args.no_phase1_edge_list,
            phase1_edge_jp_min=args.phase1_edge_jp_min,
            phase1_edge_veto_config=phase1_edge_veto_config,
            fbs_only=not args.no_fbs_only,
            distinct_lists=not args.no_distinct_lists,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
