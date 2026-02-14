#!/usr/bin/env python3
"""Backtest spread selection policies with parameter grid search.

This script runs walk-forward evaluation of different selection policies
over 2022-2025 data to find stable 2026 defaults.

Usage:
    # Fast run (no bootstrap)
    python scripts/backtest_spread_selection_policies.py --no-bootstrap

    # Default run (bootstrap top 10)
    python scripts/backtest_spread_selection_policies.py

    # Bootstrap only presets
    python scripts/backtest_spread_selection_policies.py --bootstrap-presets-only

    # Filter years and weeks
    python scripts/backtest_spread_selection_policies.py --years 2024,2025 --weeks 4-15

Output:
    - Grid search results ranked by stability score
    - Per-year breakdown for top configs
    - 2026 Preset Summary with grid ranks
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spread_selection.calibration import (
    load_and_normalize_game_data,
    calibrate_cover_probability,
    predict_cover_probability,
    calculate_ev_vectorized,
    PHASE2_WEEKS,
)
from src.spread_selection.selection_policy import (
    SelectionPolicyConfig,
    SelectionPolicy,
    apply_selection_policy,
    compute_selection_metrics,
    compute_stability_score,
    compute_max_drawdown,
    generate_policy_grid,
    config_to_label,
    SelectionMetrics,
    get_selection_policy_preset,
    configs_match,
    ALLOWED_PRESETS,
)
from src.spread_selection.stats_utils import wilson_ci, bootstrap_mean_ci

# ROI assumption tracking
ODDS_PLACEHOLDER_WARNING = (
    "ROI calculations assume -110 juice (historical odds are placeholders)."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Fixed random seed for reproducibility
BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 5000  # Reduced for grid search speed


@dataclass
class GridSearchResult:
    """Result for a single grid configuration."""

    config: SelectionPolicyConfig
    label: str
    metrics: SelectionMetrics
    stability_score: float
    max_drawdown: float
    wilson_ci_low: Optional[float]
    wilson_ci_high: Optional[float]
    roi_ci_low: Optional[float]
    roi_ci_high: Optional[float]
    per_year_ats: dict  # year -> ats%
    bootstrap_computed: bool = False


# =============================================================================
# ARGUMENT PARSING HELPERS
# =============================================================================

def parse_years(years_arg: str) -> list[int]:
    """Parse years argument: comma-separated or space-separated.

    Args:
        years_arg: String like "2024,2025" or "2022 2023 2024 2025"

    Returns:
        List of year integers
    """
    if "," in years_arg:
        return [int(y.strip()) for y in years_arg.split(",")]
    else:
        return [int(y) for y in years_arg.split()]


def parse_weeks(weeks_arg: str) -> tuple[int, int]:
    """Parse weeks argument: range like "4-15" or comma list like "4,5,6,7".

    Args:
        weeks_arg: String like "4-15" or "4,5,6"

    Returns:
        Tuple (min_week, max_week)
    """
    if "-" in weeks_arg:
        parts = weeks_arg.split("-")
        return (int(parts[0]), int(parts[1]))
    elif "," in weeks_arg:
        weeks = [int(w.strip()) for w in weeks_arg.split(",")]
        return (min(weeks), max(weeks))
    else:
        # Single week
        w = int(weeks_arg)
        return (w, w)


# =============================================================================
# DATA LOADING AND WALK-FORWARD EV
# =============================================================================

def load_and_prepare_data(
    data_path: str = "data/spread_selection/ats_export.csv",
    years: list[int] = None,
    weeks: tuple[int, int] = None,
) -> pd.DataFrame:
    """Load and prepare backtest data.

    Args:
        data_path: Path to ats_export.csv
        years: Filter to specific years (default: 2022-2025)
        weeks: Filter to week range (default: 4-15 for Phase 2)

    Returns:
        DataFrame with normalized data
    """
    if years is None:
        years = [2022, 2023, 2024, 2025]
    if weeks is None:
        weeks = PHASE2_WEEKS

    # Load raw data
    raw_df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(raw_df)} rows from {data_path}")

    # Load and normalize to Vegas convention
    df = load_and_normalize_game_data(raw_df)

    # Filter to specified years
    df = df[df["year"].isin(years)].copy()
    logger.info(f"Filtered to {len(df)} games for years {years}")

    # Filter to week range
    df = df[(df["week"] >= weeks[0]) & (df["week"] <= weeks[1])].copy()
    logger.info(f"Filtered to {len(df)} games for weeks {weeks[0]}-{weeks[1]}")

    return df


def compute_walk_forward_ev(
    df: pd.DataFrame,
    min_train_seasons: int = 1,
) -> pd.DataFrame:
    """Compute EV using walk-forward calibration (no leakage).

    For each evaluation year Y, trains on years < Y.

    Args:
        df: Normalized game data
        min_train_seasons: Minimum seasons for training

    Returns:
        DataFrame with ev_cal column added (walk-forward calibrated)
    """
    years = sorted(df["year"].unique())
    results = []

    for eval_year in years:
        # Training: years strictly before eval_year
        train_years = [y for y in years if y < eval_year]

        if len(train_years) < min_train_seasons:
            logger.debug(f"Skipping {eval_year}: insufficient training years")
            continue

        train_df = df[df["year"].isin(train_years)]
        eval_df = df[df["year"] == eval_year].copy()

        if len(train_df) < 500:
            logger.warning(f"Skipping {eval_year}: insufficient training data")
            continue

        # Fit calibration
        try:
            calibration = calibrate_cover_probability(train_df, min_games_warn=500)
        except ValueError as e:
            logger.warning(f"Calibration failed for {eval_year}: {e}")
            continue

        # Predict on eval set
        bet_mask = (eval_df["edge_abs"] > 0) & (~eval_df["push"])
        eval_df["ev_cal"] = np.nan

        if bet_mask.sum() > 0:
            edge_vals = eval_df.loc[bet_mask, "edge_abs"].values
            p_cover = predict_cover_probability(edge_vals, calibration)
            ev_vals = calculate_ev_vectorized(p_cover, p_push=0.0, juice=-110)
            eval_df.loc[bet_mask, "ev_cal"] = ev_vals

        results.append(eval_df)
        logger.debug(
            f"Year {eval_year}: {len(train_df)} train, {len(eval_df)} eval, "
            f"{bet_mask.sum()} bets, slope={calibration.slope:.4f}"
        )

    if not results:
        raise ValueError("No valid years for walk-forward evaluation")

    return pd.concat(results, ignore_index=True)


# =============================================================================
# CONFIG EVALUATION
# =============================================================================

def evaluate_single_config(
    df: pd.DataFrame,
    config: SelectionPolicyConfig,
    compute_bootstrap: bool = False,
) -> GridSearchResult:
    """Evaluate a single configuration.

    Args:
        df: DataFrame with ev_cal column from walk-forward
        config: Selection policy config
        compute_bootstrap: Whether to compute bootstrap CI (slower)

    Returns:
        GridSearchResult with metrics
    """
    # Filter to positive EV candidates
    candidates = df[df["ev_cal"] > 0].copy()
    candidates["ev"] = candidates["ev_cal"]  # Rename for policy function

    # Apply selection policy
    result = apply_selection_policy(
        candidates,
        config,
        ev_col="ev",
        year_col="year",
        week_col="week",
    )

    selected = result.selected_bets

    # Compute metrics
    metrics = compute_selection_metrics(
        selected,
        ev_col="ev",
        outcome_col="jp_side_covered",
        push_col="push",
    )

    # Compute stability score
    stability = compute_stability_score(metrics)

    # Compute max drawdown
    if len(selected) > 0:
        selected_sorted = selected.sort_values(["year", "week", "game_id"]).reset_index(drop=True)
        max_dd = compute_max_drawdown(selected_sorted)
    else:
        max_dd = 0.0

    # Wilson CI for ATS (always computed - fast)
    wilson_low, wilson_high = None, None
    if metrics.n_wins + metrics.n_losses > 0:
        wilson_low, wilson_high = wilson_ci(metrics.n_wins, metrics.n_wins + metrics.n_losses)
        wilson_low *= 100
        wilson_high *= 100

    # Bootstrap CI for ROI (optional - slow)
    roi_ci_low, roi_ci_high = None, None
    if compute_bootstrap and len(selected) >= 20:
        if "pnl" not in selected.columns:
            selected = selected.copy()
            payout = 100 / 110
            selected["pnl"] = selected.apply(
                lambda r: 0.0 if r["push"] else (payout if r["jp_side_covered"] else -1.0),
                axis=1,
            )
        returns = selected["pnl"].values
        try:
            roi_ci_low, roi_ci_high = bootstrap_mean_ci(
                returns,
                n_boot=BOOTSTRAP_N,
                seed=BOOTSTRAP_SEED,
            )
            roi_ci_low *= 100
            roi_ci_high *= 100
        except Exception:
            pass

    # Per-year ATS breakdown
    per_year_ats = {}
    for year in metrics.years_with_data:
        year_df = selected[selected["year"] == year]
        year_non_push = year_df[~year_df["push"]]
        if len(year_non_push) > 0:
            year_wins = year_non_push["jp_side_covered"].sum()
            per_year_ats[year] = year_wins / len(year_non_push) * 100
        else:
            per_year_ats[year] = 0.0

    return GridSearchResult(
        config=config,
        label=config_to_label(config),
        metrics=metrics,
        stability_score=stability,
        max_drawdown=max_dd,
        wilson_ci_low=wilson_low,
        wilson_ci_high=wilson_high,
        roi_ci_low=roi_ci_low,
        roi_ci_high=roi_ci_high,
        per_year_ats=per_year_ats,
        bootstrap_computed=compute_bootstrap,
    )


def run_grid_search(
    df: pd.DataFrame,
    configs: list[SelectionPolicyConfig] = None,
    compute_bootstrap_top_k: int = 0,
) -> list[GridSearchResult]:
    """Run grid search over all configurations.

    Args:
        df: DataFrame with ev_cal from walk-forward
        configs: List of configs to evaluate (default: generate_policy_grid())
        compute_bootstrap_top_k: Compute bootstrap CI for top K results only (0 = skip)

    Returns:
        List of GridSearchResult sorted by stability_score descending
    """
    if configs is None:
        configs = generate_policy_grid()

    logger.info(f"Running grid search over {len(configs)} configurations...")

    # First pass: evaluate all without bootstrap
    results = []
    for i, config in enumerate(configs):
        if (i + 1) % 20 == 0:
            logger.info(f"Progress: {i + 1}/{len(configs)} configs evaluated")
        try:
            result = evaluate_single_config(df, config, compute_bootstrap=False)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to evaluate {config_to_label(config)}: {e}")

    # Sort by stability score
    results.sort(key=lambda r: r.stability_score, reverse=True)

    # Second pass: compute bootstrap for top K (if requested)
    if compute_bootstrap_top_k > 0:
        logger.info(f"Computing bootstrap CIs for top {compute_bootstrap_top_k} configs...")
        for i in range(min(compute_bootstrap_top_k, len(results))):
            config = results[i].config
            try:
                results[i] = evaluate_single_config(df, config, compute_bootstrap=True)
            except Exception as e:
                logger.warning(f"Bootstrap failed for {results[i].label}: {e}")

    # Re-sort after bootstrap (stability unchanged, but for consistency)
    results.sort(key=lambda r: r.stability_score, reverse=True)

    return results


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_results_table(
    results: list[GridSearchResult],
    top_n: int = 10,
    show_per_year: bool = True,
    odds_placeholder: bool = True,
) -> None:
    """Print results table to console.

    Args:
        results: List of GridSearchResult sorted by stability
        top_n: Number of top results to show
        show_per_year: Whether to show per-year breakdown
        odds_placeholder: Whether odds are placeholders (affects ROI label)
    """
    print("\n" + "=" * 110)
    print("SPREAD SELECTION POLICY GRID SEARCH RESULTS")
    print("=" * 110)

    # ROI assumption warning
    if odds_placeholder:
        print(f"\n*** {ODDS_PLACEHOLDER_WARNING} ***\n")

    print(f"Top {min(top_n, len(results))} configurations by stability score:\n")

    # Header with ROI label
    roi_label = "ROI@-110" if odds_placeholder else "ROI%"
    header = (
        f"{'Rank':<5} | {'Policy':<42} | {'Bets':>5} | {'ATS%':>6} | "
        f"{'ATS CI':>14} | {roi_label:>9} | {'Stab':>6} | {'DD':>4}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results[:top_n]):
        # Format ATS CI (Wilson - always computed)
        if r.wilson_ci_low is not None and r.wilson_ci_high is not None:
            ats_ci = f"[{r.wilson_ci_low:.1f}, {r.wilson_ci_high:.1f}]"
        else:
            ats_ci = "—"

        print(
            f"{i + 1:<5} | {r.label:<42} | {r.metrics.n_bets:>5} | "
            f"{r.metrics.ats_pct:>5.1f}% | {ats_ci:>14} | "
            f"{r.metrics.roi_pct:>+8.1f}% | {r.stability_score:>+5.2f} | {r.max_drawdown:>4.1f}"
        )

    print("-" * len(header))

    # Per-year breakdown for top 5
    if show_per_year and len(results) > 0:
        print("\nPer-Year ATS% Breakdown (Top 5):\n")

        years = sorted(results[0].metrics.years_with_data) if results[0].metrics.years_with_data else []
        if years:
            year_header = f"{'Rank':<5} | {'Policy':<42} |"
            for year in years:
                year_header += f" {year}  |"
            print(year_header)
            print("-" * len(year_header))

            for i, r in enumerate(results[:5]):
                row = f"{i + 1:<5} | {r.label:<42} |"
                for year in years:
                    ats = r.per_year_ats.get(year, 0.0)
                    row += f" {ats:>4.1f}% |"
                print(row)

            print("-" * len(year_header))


# =============================================================================
# PRESET EVALUATION
# =============================================================================

def find_preset_in_grid(
    preset_name: str,
    results: list[GridSearchResult],
) -> tuple[Optional[int], Optional[GridSearchResult]]:
    """Find a preset's matching row in grid results.

    Args:
        preset_name: Preset name ("conservative", "balanced", "aggressive")
        results: Sorted grid search results

    Returns:
        (rank, result) if found, (None, None) if not in grid
    """
    preset_config = get_selection_policy_preset(preset_name)

    for i, r in enumerate(results):
        if configs_match(r.config, preset_config):
            return (i + 1, r)

    return (None, None)


def evaluate_presets(
    df: pd.DataFrame,
    results: list[GridSearchResult],
    compute_bootstrap: bool = False,
) -> dict[str, tuple[int, GridSearchResult]]:
    """Evaluate all presets and find their grid ranks.

    Args:
        df: DataFrame with walk-forward EV data
        results: Grid search results (for rank lookup)
        compute_bootstrap: Whether to compute bootstrap CI for presets

    Returns:
        Dict of preset_name -> (rank or None, GridSearchResult)
    """
    preset_results = {}

    for name in ALLOWED_PRESETS:
        rank, grid_result = find_preset_in_grid(name, results)

        if grid_result is not None:
            # Found in grid - use grid result (may or may not have bootstrap)
            # If we need bootstrap and grid didn't compute it, re-evaluate
            if compute_bootstrap and not grid_result.bootstrap_computed:
                logger.info(f"Computing bootstrap for preset '{name}'...")
                config = get_selection_policy_preset(name)
                result = evaluate_single_config(df, config, compute_bootstrap=True)
                preset_results[name] = (rank, result)
            else:
                preset_results[name] = (rank, grid_result)
        else:
            # Not in grid - evaluate explicitly
            logger.info(f"Preset '{name}' not in grid, evaluating explicitly...")
            config = get_selection_policy_preset(name)
            result = evaluate_single_config(df, config, compute_bootstrap=compute_bootstrap)
            preset_results[name] = (None, result)

    return preset_results


def print_preset_summary(
    preset_results: dict[str, tuple[int, GridSearchResult]],
    odds_placeholder: bool = True,
) -> None:
    """Print 2026 Preset Summary table.

    Args:
        preset_results: Dict from evaluate_presets
        odds_placeholder: Whether ROI is based on placeholder odds
    """
    print("\n" + "=" * 115)
    print("2026 PRESET SUMMARY")
    print("=" * 115)

    if odds_placeholder:
        print(f"\n*** {ODDS_PLACEHOLDER_WARNING} ***\n")

    # Header
    roi_label = "ROI@-110" if odds_placeholder else "ROI%"
    header = (
        f"{'Preset':<14} | {'Policy Config':<45} | {'Bets':>5} | "
        f"{'ATS%':>6} | {'ATS CI':>14} | {roi_label:>9} | {'Stab':>6} | {'Rank':>6}"
    )
    print(header)
    print("-" * len(header))

    # Print in order: conservative, balanced, aggressive
    for name in ["conservative", "balanced", "aggressive"]:
        if name not in preset_results:
            continue

        rank, r = preset_results[name]

        # Format ATS CI
        if r.wilson_ci_low is not None and r.wilson_ci_high is not None:
            ats_ci = f"[{r.wilson_ci_low:.1f}, {r.wilson_ci_high:.1f}]"
        else:
            ats_ci = "—"

        # Rank display
        rank_str = f"#{rank}" if rank else "N/A*"

        print(
            f"{name:<14} | {r.label:<45} | {r.metrics.n_bets:>5} | "
            f"{r.metrics.ats_pct:>5.1f}% | {ats_ci:>14} | "
            f"{r.metrics.roi_pct:>+8.1f}% | {r.stability_score:>+5.2f} | {rank_str:>6}"
        )

    print("-" * len(header))

    # Note if any preset not in grid
    not_in_grid = [name for name, (rank, _) in preset_results.items() if rank is None]
    if not_in_grid:
        print(f"\n* Presets not in grid (evaluated explicitly): {', '.join(not_in_grid)}")

    # Show preset definitions
    print("\nPreset Definitions:")
    print("  conservative: EV_THRESHOLD, ev_min=3%, max_bets_per_week=5")
    print("  balanced:     TOP_N_PER_WEEK, n=3, ev_floor=1%, max_bets_per_week=3")
    print("  aggressive:   TOP_N_PER_WEEK, n=5, ev_floor=1%, max_bets_per_week=5")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest spread selection policies with grid search"
    )
    parser.add_argument(
        "--data",
        default="data/spread_selection/ats_export.csv",
        help="Path to ATS export data",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top results to show (default: 10)",
    )
    parser.add_argument(
        "--policy",
        choices=["EV_THRESHOLD", "TOP_N_PER_WEEK", "HYBRID", "ALL"],
        default="ALL",
        help="Filter to specific policy type (default: ALL)",
    )

    # Years and weeks filters
    parser.add_argument(
        "--years",
        type=str,
        default="2022,2023,2024,2025",
        help="Years to include, comma-separated (default: 2022,2023,2024,2025)",
    )
    parser.add_argument(
        "--weeks",
        type=str,
        default="4-15",
        help="Weeks to include as range X-Y or comma list (default: 4-15)",
    )

    # Bootstrap controls
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip all bootstrap computations (fast mode)",
    )
    parser.add_argument(
        "--bootstrap-top-k",
        type=int,
        default=10,
        help="Compute bootstrap CI for top K grid configs (default: 10, ignored if --no-bootstrap)",
    )
    parser.add_argument(
        "--bootstrap-presets-only",
        action="store_true",
        help="Compute bootstrap CIs only for presets, not grid configs",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start_time = time.time()

    # Parse years and weeks
    years = parse_years(args.years)
    weeks = parse_weeks(args.weeks)

    # Determine bootstrap settings
    if args.no_bootstrap:
        bootstrap_top_k = 0
        bootstrap_presets = False
    elif args.bootstrap_presets_only:
        bootstrap_top_k = 0
        bootstrap_presets = True
    else:
        bootstrap_top_k = args.bootstrap_top_k
        bootstrap_presets = True

    # Track odds assumption (all historical CFBD data is placeholder -110)
    odds_placeholder = True

    # Load and prepare data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print(f"Years: {years}")
    print(f"Weeks: {weeks[0]}-{weeks[1]}")
    print(f"Bootstrap: {'disabled' if args.no_bootstrap else f'top-{bootstrap_top_k} grid' if not args.bootstrap_presets_only else 'presets only'}")

    df = load_and_prepare_data(args.data, years=years, weeks=weeks)

    # Compute walk-forward EV
    print("\n" + "=" * 80)
    print("COMPUTING WALK-FORWARD EV")
    print("=" * 80)

    df = compute_walk_forward_ev(df, min_train_seasons=1)
    logger.info(f"Walk-forward complete: {len(df)} games with EV computed")

    # Generate config grid
    configs = generate_policy_grid()

    # Filter by policy type if specified
    if args.policy != "ALL":
        configs = [c for c in configs if c.selection_policy == args.policy]
        logger.info(f"Filtered to {len(configs)} {args.policy} configs")

    # Run grid search
    print("\n" + "=" * 80)
    print("RUNNING GRID SEARCH")
    print("=" * 80)

    results = run_grid_search(df, configs, compute_bootstrap_top_k=bootstrap_top_k)

    # Print results
    print_results_table(results, top_n=args.top, show_per_year=True, odds_placeholder=odds_placeholder)

    # Evaluate presets and show where they land in grid
    print("\n" + "=" * 80)
    print("EVALUATING 2026 PRESETS")
    print("=" * 80)

    preset_results = evaluate_presets(df, results, compute_bootstrap=bootstrap_presets)
    print_preset_summary(preset_results, odds_placeholder=odds_placeholder)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"GRID SEARCH COMPLETE (elapsed: {elapsed:.1f}s)")
    print("=" * 80)

    return results, preset_results


if __name__ == "__main__":
    main()
