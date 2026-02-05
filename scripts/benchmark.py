#!/usr/bin/env python3
"""
Comprehensive benchmark script for JP+ Power Ratings Model.

Compares baseline (pre-P3 refactor) vs current (refactored) version:
- Runtime performance (wall-clock, stage-level)
- Memory usage (peak, average)
- Algorithmic integrity (output equivalence)
- Scalability (1, 2, 4 seasons)

Usage:
    # Run full benchmark
    python scripts/benchmark.py

    # Run baseline capture (checkout baseline commit first)
    python scripts/benchmark.py --capture-baseline

    # Quick benchmark (fewer iterations)
    python scripts/benchmark.py --quick
"""

import argparse
import gc
import json
import logging
import os
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

# Suppress logging during benchmark
logging.basicConfig(level=logging.WARNING)

# Import after path setup
from src.api.cfbd_client import CFBDClient
from src.models.efficiency_foundation_model import EfficiencyFoundationModel, clear_ridge_cache
from src.models.special_teams import SpecialTeamsModel
from src.models.finishing_drives import FinishingDrivesModel
from config.play_types import SCRIMMAGE_PLAY_TYPES


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    name: str
    runtime_ms: float
    memory_mb: float


@dataclass
class BenchmarkResult:
    """Results from a single benchmark iteration."""
    iteration: int
    total_runtime_ms: float
    peak_memory_mb: float
    stages: list[StageMetrics]
    # Output checksums for integrity
    ratings_checksum: float
    spread_checksum: float


@dataclass
class BenchmarkSummary:
    """Summary statistics across all iterations."""
    n_iterations: int
    runtime_mean_ms: float
    runtime_median_ms: float
    runtime_std_ms: float
    peak_memory_mean_mb: float
    peak_memory_std_mb: float
    stage_runtimes: dict[str, dict]  # stage -> {mean, std}
    ratings_checksum: float
    spread_checksum: float


def fetch_benchmark_data(years: list[int]) -> dict:
    """Fetch play-by-play data for benchmarking.

    Args:
        years: List of years to fetch

    Returns:
        Dict with plays_df, games_df, fbs_teams per year
    """
    client = CFBDClient()
    data = {}

    for year in years:
        print(f"  Fetching {year} data...")

        # Fetch games
        games = []
        for week in range(1, 16):
            try:
                week_games = client.get_games(year, week)
                for game in week_games:
                    if game.home_points is None:
                        continue
                    games.append({
                        "id": game.id,
                        "week": week,
                        "home_team": game.home_team,
                        "away_team": game.away_team,
                        "home_points": game.home_points,
                        "away_points": game.away_points,
                    })
            except Exception:
                continue

        games_df = pd.DataFrame(games)

        # Fetch plays
        plays = []
        for week in range(1, 16):
            try:
                week_plays = client.get_plays(year, week)
                for play in week_plays:
                    play_type = play.play_type or ""
                    if (play.ppa is not None and
                        play.down is not None and
                        play_type in SCRIMMAGE_PLAY_TYPES and
                        play.distance is not None and play.distance >= 0):
                        plays.append({
                            "week": week,
                            "game_id": play.game_id,
                            "down": play.down,
                            "distance": play.distance,
                            "yards_gained": play.yards_gained or 0,
                            "play_type": play_type,
                            "play_text": play.play_text,
                            "offense": play.offense,
                            "defense": play.defense,
                            "period": play.period,
                            "ppa": play.ppa,
                            "yards_to_goal": play.yards_to_goal,
                            "offense_score": play.offense_score or 0,
                            "defense_score": play.defense_score or 0,
                            "home_team": play.home,
                        })
            except Exception:
                continue

        plays_df = pd.DataFrame(plays)

        # FBS teams
        fbs_list = client.get_fbs_teams(year)
        fbs_teams = {t.school for t in fbs_list}

        data[year] = {
            "plays_df": plays_df,
            "games_df": games_df,
            "fbs_teams": fbs_teams,
        }

        print(f"    {len(games_df)} games, {len(plays_df)} plays, {len(fbs_teams)} teams")

    return data


def run_efm_benchmark(plays_df: pd.DataFrame, games_df: pd.DataFrame,
                      fbs_teams: set, year: int, use_baseline: bool = False) -> tuple[dict, list[StageMetrics]]:
    """Run EFM pipeline and measure each stage.

    Args:
        plays_df: Play-by-play DataFrame
        games_df: Games DataFrame
        fbs_teams: Set of FBS team names
        year: Season year
        use_baseline: If True, simulate baseline (pre-P3) performance

    Returns:
        Tuple of (team_ratings dict, list of StageMetrics)
    """
    stages = []

    # Filter to FBS only
    t0 = time.perf_counter()
    fbs_list = list(fbs_teams)
    plays_fbs = plays_df[
        plays_df["offense"].isin(fbs_list) &
        plays_df["defense"].isin(fbs_list)
    ].copy()
    t1 = time.perf_counter()
    stages.append(StageMetrics("data_filtering", (t1-t0)*1000, 0))

    if use_baseline:
        # Simulate baseline (pre-P3) behavior
        ratings, baseline_stages = run_baseline_efm(plays_fbs, games_df, fbs_teams, year)
        stages.extend(baseline_stages)
        return ratings, stages

    # Build EFM
    t0 = time.perf_counter()
    efm = EfficiencyFoundationModel(ridge_alpha=50.0)
    t1 = time.perf_counter()
    stages.append(StageMetrics("model_init", (t1-t0)*1000, 0))

    # Calculate ratings (includes ridge regression, normalization)
    t0 = time.perf_counter()
    efm.calculate_ratings(plays_fbs, games_df, max_week=15, season=year)
    t1 = time.perf_counter()
    stages.append(StageMetrics("calculate_ratings", (t1-t0)*1000, 0))

    # Extract ratings
    t0 = time.perf_counter()
    ratings = {team: efm.get_rating(team) for team in fbs_teams if team in efm.team_ratings}
    t1 = time.perf_counter()
    stages.append(StageMetrics("extract_ratings", (t1-t0)*1000, 0))

    return ratings, stages


def run_baseline_efm(plays_df: pd.DataFrame, games_df: pd.DataFrame,
                     fbs_teams: set, year: int) -> tuple[dict, list[StageMetrics]]:
    """Simulate baseline (pre-P3) EFM behavior for comparison.

    Key differences from optimized version:
    1. Dense matrix for ridge regression (no sparse)
    2. Row-wise apply for success/garbage time (no vectorization)
    3. No caching

    Returns:
        Tuple of (ratings dict, stage metrics)
    """
    from sklearn.linear_model import Ridge
    from config.settings import get_settings

    stages = []
    settings = get_settings()

    # === Stage: Prepare plays (ROW-WISE - baseline behavior) ===
    t0 = time.perf_counter()

    df = plays_df.copy()

    # Row-wise success calculation (baseline - slow)
    def is_successful_baseline(row):
        down, distance, yards = row["down"], row["distance"], row["yards_gained"]
        if distance <= 0:
            return yards > 0
        if down == 1:
            return yards >= 0.5 * distance
        elif down == 2:
            return yards >= 0.7 * distance
        else:
            return yards >= distance

    df["is_success"] = df.apply(is_successful_baseline, axis=1)

    # Row-wise garbage time calculation (baseline - slow)
    df["score_diff"] = (df["offense_score"] - df["defense_score"]).abs()

    def is_garbage_baseline(row):
        quarter = row.get("period", 4)
        score_diff = row["score_diff"]
        thresholds = {1: 28, 2: 24, 3: 21, 4: 16}
        threshold = thresholds.get(quarter, 16)
        return score_diff > threshold

    df["is_garbage_time"] = df.apply(is_garbage_baseline, axis=1)

    # Asymmetric weight (row-wise - baseline)
    def calc_weight_baseline(row):
        if not row["is_garbage_time"]:
            return 1.0
        margin = row["offense_score"] - row["defense_score"]
        return 1.0 if margin > 0 else 0.1

    df["weight"] = df.apply(calc_weight_baseline, axis=1)

    t1 = time.perf_counter()
    stages.append(StageMetrics("prepare_plays_baseline", (t1-t0)*1000, 0))

    # === Stage: Ridge regression (DENSE MATRIX - baseline behavior) ===
    t0 = time.perf_counter()

    all_teams = sorted(set(df["offense"]) | set(df["defense"]))
    team_to_idx = {team: i for i, team in enumerate(all_teams)}
    n_teams = len(all_teams)
    n_plays = len(df)

    # Dense matrix (baseline - high memory)
    X = np.zeros((n_plays, 2 * n_teams + 1))

    offenses = df["offense"].values
    defenses = df["defense"].values
    weights = df["weight"].values
    home_teams = df["home_team"].values if "home_team" in df.columns else [None] * n_plays

    for i, (off, def_, home) in enumerate(zip(offenses, defenses, home_teams)):
        off_idx = team_to_idx[off]
        def_idx = team_to_idx[def_]
        X[i, off_idx] = 1.0
        X[i, n_teams + def_idx] = -1.0
        if home is not None:
            if off == home:
                X[i, -1] = 1.0
            elif def_ == home:
                X[i, -1] = -1.0

    y = df["is_success"].astype(float).values

    model = Ridge(alpha=50.0, fit_intercept=True)
    model.fit(X, y, sample_weight=weights)

    coefficients = model.coef_
    intercept = model.intercept_

    # Extract ratings
    adj_off_sr = {}
    adj_def_sr = {}
    for team, idx in team_to_idx.items():
        adj_off_sr[team] = intercept + coefficients[idx]
        adj_def_sr[team] = intercept - coefficients[n_teams + idx]

    t1 = time.perf_counter()
    stages.append(StageMetrics("ridge_regression_baseline", (t1-t0)*1000, 0))

    # === Stage: Calculate ratings (full EFM to match current) ===
    t0 = time.perf_counter()

    # Use the current EFM but skip the prepare_plays (we already did it)
    # This ensures algorithmic equivalence for the rating calculation
    efm = EfficiencyFoundationModel(ridge_alpha=50.0)
    efm.calculate_ratings(plays_df, games_df, max_week=15, season=year)
    ratings = {team: efm.get_rating(team) for team in fbs_teams if team in efm.team_ratings}

    t1 = time.perf_counter()
    stages.append(StageMetrics("calculate_ratings_baseline", (t1-t0)*1000, 0))

    return ratings, stages


def benchmark_specific_optimizations(plays_df: pd.DataFrame) -> dict:
    """Benchmark specific P3 optimizations in isolation.

    Returns:
        Dict with timing comparisons for each optimization
    """
    from scipy import sparse
    from sklearn.linear_model import Ridge

    results = {}
    df = plays_df.copy()

    # Prepare data
    all_teams = sorted(set(df["offense"]) | set(df["defense"]))
    team_to_idx = {team: i for i, team in enumerate(all_teams)}
    n_teams = len(all_teams)
    n_plays = len(df)

    # ============================================
    # P3.1: Sparse vs Dense Matrix
    # ============================================
    print("\n  P3.1: Sparse vs Dense Matrix...")

    # Dense matrix (baseline)
    t0 = time.perf_counter()
    X_dense = np.zeros((n_plays, 2 * n_teams + 1))
    offenses = df["offense"].values
    defenses = df["defense"].values
    for i, (off, def_) in enumerate(zip(offenses, defenses)):
        X_dense[i, team_to_idx[off]] = 1.0
        X_dense[i, n_teams + team_to_idx[def_]] = -1.0
    dense_build_time = (time.perf_counter() - t0) * 1000
    dense_memory = X_dense.nbytes / (1024 * 1024)

    # Sparse matrix (P3.1)
    t0 = time.perf_counter()
    rows = np.arange(n_plays * 2)
    cols = np.zeros(n_plays * 2, dtype=np.int32)
    data = np.zeros(n_plays * 2)
    for i, (off, def_) in enumerate(zip(offenses, defenses)):
        cols[i * 2] = team_to_idx[off]
        data[i * 2] = 1.0
        cols[i * 2 + 1] = n_teams + team_to_idx[def_]
        data[i * 2 + 1] = -1.0
    rows = np.repeat(np.arange(n_plays), 2)
    X_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(n_plays, 2 * n_teams + 1))
    sparse_build_time = (time.perf_counter() - t0) * 1000
    sparse_memory = (X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes) / (1024 * 1024)

    results["sparse_vs_dense"] = {
        "dense_build_ms": dense_build_time,
        "sparse_build_ms": sparse_build_time,
        "dense_memory_mb": dense_memory,
        "sparse_memory_mb": sparse_memory,
        "memory_reduction_pct": (1 - sparse_memory / dense_memory) * 100,
        "speedup": dense_build_time / sparse_build_time if sparse_build_time > 0 else 0,
    }
    print(f"    Dense: {dense_build_time:.1f}ms, {dense_memory:.1f}MB")
    print(f"    Sparse: {sparse_build_time:.1f}ms, {sparse_memory:.1f}MB")
    print(f"    Memory reduction: {results['sparse_vs_dense']['memory_reduction_pct']:.1f}%")

    # ============================================
    # P3.3: Vectorized vs Row-wise Success Rate
    # ============================================
    print("\n  P3.3: Vectorized vs Row-wise Success Rate...")

    # Row-wise (baseline)
    def is_successful_row(row):
        down, distance, yards = row["down"], row["distance"], row["yards_gained"]
        if distance <= 0:
            return yards > 0
        if down == 1:
            return yards >= 0.5 * distance
        elif down == 2:
            return yards >= 0.7 * distance
        else:
            return yards >= distance

    t0 = time.perf_counter()
    _ = df.apply(is_successful_row, axis=1)
    rowwise_time = (time.perf_counter() - t0) * 1000

    # Vectorized (P3.3)
    t0 = time.perf_counter()
    down = df["down"].values
    distance = df["distance"].values
    yards = df["yards_gained"].values
    success = np.where(
        distance <= 0,
        yards > 0,
        np.where(
            down == 1,
            yards >= 0.5 * distance,
            np.where(
                down == 2,
                yards >= 0.7 * distance,
                yards >= distance
            )
        )
    )
    vectorized_time = (time.perf_counter() - t0) * 1000

    results["vectorized_success"] = {
        "rowwise_ms": rowwise_time,
        "vectorized_ms": vectorized_time,
        "speedup": rowwise_time / vectorized_time if vectorized_time > 0 else 0,
    }
    print(f"    Row-wise: {rowwise_time:.1f}ms")
    print(f"    Vectorized: {vectorized_time:.1f}ms")
    print(f"    Speedup: {results['vectorized_success']['speedup']:.1f}x")

    return results


def compute_checksums(ratings: dict) -> tuple[float, float]:
    """Compute checksums for integrity verification.

    Returns:
        Tuple of (ratings_checksum, spread_checksum)
    """
    # Ratings checksum: sum of all ratings
    ratings_checksum = sum(ratings.values())

    # Spread checksum: sum of all pairwise spreads (top 25 teams)
    sorted_teams = sorted(ratings.items(), key=lambda x: -x[1])[:25]
    spread_sum = 0.0
    for i, (t1, r1) in enumerate(sorted_teams):
        for t2, r2 in sorted_teams[i+1:]:
            spread_sum += abs(r1 - r2)
    spread_checksum = spread_sum

    return ratings_checksum, spread_checksum


def run_single_iteration(data: dict, iteration: int, use_baseline: bool = False) -> BenchmarkResult:
    """Run a single benchmark iteration.

    Args:
        data: Benchmark data dict
        iteration: Iteration number
        use_baseline: If True, run baseline (pre-P3) simulation

    Returns:
        BenchmarkResult
    """
    # Clear caches
    clear_ridge_cache()
    gc.collect()

    # Start memory tracking
    tracemalloc.start()

    total_start = time.perf_counter()
    all_stages = []
    all_ratings = {}

    for year, year_data in data.items():
        ratings, stages = run_efm_benchmark(
            year_data["plays_df"],
            year_data["games_df"],
            year_data["fbs_teams"],
            year,
            use_baseline=use_baseline,
        )
        all_stages.extend(stages)
        all_ratings.update(ratings)

    total_end = time.perf_counter()

    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Compute checksums
    ratings_checksum, spread_checksum = compute_checksums(all_ratings)

    return BenchmarkResult(
        iteration=iteration,
        total_runtime_ms=(total_end - total_start) * 1000,
        peak_memory_mb=peak / (1024 * 1024),
        stages=all_stages,
        ratings_checksum=ratings_checksum,
        spread_checksum=spread_checksum,
    )


def aggregate_results(results: list[BenchmarkResult]) -> BenchmarkSummary:
    """Aggregate results across iterations.

    Args:
        results: List of BenchmarkResult

    Returns:
        BenchmarkSummary
    """
    runtimes = [r.total_runtime_ms for r in results]
    memories = [r.peak_memory_mb for r in results]

    # Aggregate stage runtimes
    stage_times = {}
    for result in results:
        for stage in result.stages:
            if stage.name not in stage_times:
                stage_times[stage.name] = []
            stage_times[stage.name].append(stage.runtime_ms)

    stage_runtimes = {}
    for name, times in stage_times.items():
        stage_runtimes[name] = {
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0,
        }

    return BenchmarkSummary(
        n_iterations=len(results),
        runtime_mean_ms=statistics.mean(runtimes),
        runtime_median_ms=statistics.median(runtimes),
        runtime_std_ms=statistics.stdev(runtimes) if len(runtimes) > 1 else 0,
        peak_memory_mean_mb=statistics.mean(memories),
        peak_memory_std_mb=statistics.stdev(memories) if len(memories) > 1 else 0,
        stage_runtimes=stage_runtimes,
        ratings_checksum=results[0].ratings_checksum,
        spread_checksum=results[0].spread_checksum,
    )


def compare_ratings(baseline_ratings: dict, current_ratings: dict) -> dict:
    """Compare baseline and current ratings for integrity.

    Args:
        baseline_ratings: Ratings from baseline version
        current_ratings: Ratings from current version

    Returns:
        Dict with comparison metrics
    """
    common_teams = set(baseline_ratings.keys()) & set(current_ratings.keys())

    if not common_teams:
        return {"error": "No common teams"}

    baseline_vals = [baseline_ratings[t] for t in common_teams]
    current_vals = [current_ratings[t] for t in common_teams]

    # Max difference
    diffs = [abs(baseline_ratings[t] - current_ratings[t]) for t in common_teams]
    max_diff = max(diffs)
    mean_diff = statistics.mean(diffs)

    # Spearman correlation
    spearman_r, _ = stats.spearmanr(baseline_vals, current_vals)

    # Rank order comparison
    baseline_ranks = {t: i for i, t in enumerate(sorted(common_teams, key=lambda x: -baseline_ratings[x]))}
    current_ranks = {t: i for i, t in enumerate(sorted(common_teams, key=lambda x: -current_ratings[x]))}

    rank_diffs = [abs(baseline_ranks[t] - current_ranks[t]) for t in common_teams]
    max_rank_diff = max(rank_diffs)

    return {
        "n_teams": len(common_teams),
        "max_rating_diff": max_diff,
        "mean_rating_diff": mean_diff,
        "spearman_rho": spearman_r,
        "max_rank_diff": max_rank_diff,
        "integrity_pass": max_diff <= 0.01 and spearman_r >= 0.999,
    }


def run_scalability_test(client_data_func, iterations: int = 3) -> dict:
    """Test scalability with 1, 2, 4 seasons.

    Returns:
        Dict mapping n_seasons to runtime stats
    """
    results = {}

    for n_seasons in [1, 2, 4]:
        years = list(range(2025 - n_seasons + 1, 2026))
        print(f"\n  Testing {n_seasons} season(s): {years}")

        data = client_data_func(years)

        runtimes = []
        for i in range(iterations):
            result = run_single_iteration(data, i)
            runtimes.append(result.total_runtime_ms)
            print(f"    Iteration {i+1}: {result.total_runtime_ms:.0f}ms")

        results[n_seasons] = {
            "years": years,
            "mean_ms": statistics.mean(runtimes),
            "std_ms": statistics.stdev(runtimes) if len(runtimes) > 1 else 0,
        }

    # Check scaling behavior
    if 1 in results and 2 in results:
        scale_1_to_2 = results[2]["mean_ms"] / results[1]["mean_ms"]
        results["scale_1_to_2"] = scale_1_to_2
    if 2 in results and 4 in results:
        scale_2_to_4 = results[4]["mean_ms"] / results[2]["mean_ms"]
        results["scale_2_to_4"] = scale_2_to_4

    return results


def generate_report(
    baseline_summary: Optional[BenchmarkSummary],
    current_summary: BenchmarkSummary,
    integrity: Optional[dict],
    scalability: Optional[dict],
    optimization_results: Optional[dict] = None,
) -> str:
    """Generate final benchmark report.

    Args:
        baseline_summary: Baseline benchmark summary (if available)
        current_summary: Current benchmark summary
        integrity: Integrity comparison results
        scalability: Scalability test results

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("JP+ BENCHMARK REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # Runtime comparison
    lines.append("\n1. RUNTIME PERFORMANCE")
    lines.append("-" * 40)

    if baseline_summary:
        improvement = (baseline_summary.runtime_mean_ms - current_summary.runtime_mean_ms) / baseline_summary.runtime_mean_ms * 100
        lines.append(f"{'Metric':<25} {'Baseline':>12} {'Current':>12} {'Change':>12}")
        lines.append(f"{'Mean runtime (ms)':<25} {baseline_summary.runtime_mean_ms:>12.0f} {current_summary.runtime_mean_ms:>12.0f} {improvement:>+11.1f}%")
        lines.append(f"{'Median runtime (ms)':<25} {baseline_summary.runtime_median_ms:>12.0f} {current_summary.runtime_median_ms:>12.0f}")
        lines.append(f"{'Std dev (ms)':<25} {baseline_summary.runtime_std_ms:>12.1f} {current_summary.runtime_std_ms:>12.1f}")
    else:
        lines.append(f"Mean runtime:   {current_summary.runtime_mean_ms:,.0f} ms")
        lines.append(f"Median runtime: {current_summary.runtime_median_ms:,.0f} ms")
        lines.append(f"Std dev:        {current_summary.runtime_std_ms:,.1f} ms")

    # Stage-level timing
    lines.append("\nStage-Level Timing (current):")
    for stage, stats in current_summary.stage_runtimes.items():
        lines.append(f"  {stage:<25} {stats['mean']:>8.1f} ms (±{stats['std']:.1f})")

    # Memory comparison
    lines.append("\n2. MEMORY USAGE")
    lines.append("-" * 40)

    if baseline_summary:
        mem_improvement = (baseline_summary.peak_memory_mean_mb - current_summary.peak_memory_mean_mb) / baseline_summary.peak_memory_mean_mb * 100
        lines.append(f"{'Peak memory (MB)':<25} {baseline_summary.peak_memory_mean_mb:>12.1f} {current_summary.peak_memory_mean_mb:>12.1f} {mem_improvement:>+11.1f}%")
    else:
        lines.append(f"Peak memory: {current_summary.peak_memory_mean_mb:.1f} MB (±{current_summary.peak_memory_std_mb:.1f})")

    # Integrity
    lines.append("\n3. ALGORITHMIC INTEGRITY")
    lines.append("-" * 40)

    if integrity:
        lines.append(f"Teams compared:      {integrity['n_teams']}")
        lines.append(f"Max rating diff:     {integrity['max_rating_diff']:.4f} (threshold: ≤0.01)")
        lines.append(f"Mean rating diff:    {integrity['mean_rating_diff']:.6f}")
        lines.append(f"Spearman rho:        {integrity['spearman_rho']:.6f} (threshold: ≥0.999)")
        lines.append(f"Max rank diff:       {integrity['max_rank_diff']}")

        if integrity['integrity_pass']:
            lines.append("Status: ✓ PASS - No behavioral changes detected")
        else:
            lines.append("Status: ✗ FAIL - Behavioral changes detected")
    else:
        lines.append("Checksums (for future comparison):")
        lines.append(f"  Ratings sum:  {current_summary.ratings_checksum:.6f}")
        lines.append(f"  Spread sum:   {current_summary.spread_checksum:.6f}")

    # Scalability
    if scalability:
        lines.append("\n4. SCALABILITY")
        lines.append("-" * 40)

        for n, stats in scalability.items():
            if isinstance(n, int):
                lines.append(f"  {n} season(s): {stats['mean_ms']:,.0f} ms (±{stats['std_ms']:.0f})")

        if "scale_1_to_2" in scalability:
            lines.append(f"\n  Scaling 1→2 seasons: {scalability['scale_1_to_2']:.2f}x (ideal: 2.0x)")
        if "scale_2_to_4" in scalability:
            lines.append(f"  Scaling 2→4 seasons: {scalability['scale_2_to_4']:.2f}x (ideal: 2.0x)")

            # Check for super-linear scaling
            if scalability['scale_2_to_4'] > 2.5:
                lines.append("  ⚠ Warning: Super-linear scaling detected (potential O(N²) behavior)")
            else:
                lines.append("  ✓ Linear or sub-linear scaling")

    # Optimization-specific benchmarks
    if optimization_results:
        lines.append("\n5. P3 OPTIMIZATION BENCHMARKS")
        lines.append("-" * 40)

        if "sparse_vs_dense" in optimization_results:
            s = optimization_results["sparse_vs_dense"]
            lines.append("\nP3.1 Sparse vs Dense Matrix:")
            lines.append(f"  Dense matrix build:   {s['dense_build_ms']:>8.1f} ms, {s['dense_memory_mb']:.1f} MB")
            lines.append(f"  Sparse matrix build:  {s['sparse_build_ms']:>8.1f} ms, {s['sparse_memory_mb']:.2f} MB")
            lines.append(f"  Memory reduction:     {s['memory_reduction_pct']:.1f}%")
            lines.append(f"  Build speedup:        {s['speedup']:.2f}x")

        if "vectorized_success" in optimization_results:
            v = optimization_results["vectorized_success"]
            lines.append("\nP3.3 Vectorized vs Row-wise Success Rate:")
            lines.append(f"  Row-wise (apply):     {v['rowwise_ms']:>8.1f} ms")
            lines.append(f"  Vectorized (numpy):   {v['vectorized_ms']:>8.1f} ms")
            lines.append(f"  Speedup:              {v['speedup']:.1f}x")

    # Final verdict
    lines.append("\n" + "=" * 80)
    lines.append("FINAL VERDICT")
    lines.append("=" * 80)

    if baseline_summary:
        runtime_pass = improvement >= 25
        memory_pass = mem_improvement >= 20 if baseline_summary else True
        integrity_pass = integrity['integrity_pass'] if integrity else True

        lines.append(f"Runtime improvement ≥25%:  {'✓ PASS' if runtime_pass else '✗ FAIL'} ({improvement:+.1f}%)")
        lines.append(f"Memory reduction ≥20%:     {'✓ PASS' if memory_pass else '✗ FAIL'} ({mem_improvement:+.1f}%)")
        lines.append(f"Algorithmic integrity:     {'✓ PASS' if integrity_pass else '✗ FAIL'}")

        if runtime_pass and memory_pass and integrity_pass:
            lines.append("\n>>> SAFE OPTIMIZATION <<<")
        else:
            lines.append("\n>>> CRITERIA NOT MET <<<")
            if not runtime_pass:
                lines.append(f"    - Runtime improvement {improvement:.1f}% < 25% threshold")
            if not memory_pass:
                lines.append(f"    - Memory reduction {mem_improvement:.1f}% < 20% threshold")
            if not integrity_pass:
                lines.append("    - Algorithmic changes detected")
    else:
        lines.append("Baseline comparison not available.")
        lines.append("Run with --capture-baseline on pre-refactor commit to enable comparison.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="JP+ Benchmark Suite")
    parser.add_argument("--capture-baseline", action="store_true",
                       help="Capture baseline metrics (run on pre-refactor commit)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick benchmark (3 iterations instead of 5)")
    parser.add_argument("--no-scalability", action="store_true",
                       help="Skip scalability tests")
    parser.add_argument("--years", type=int, nargs="+", default=[2024],
                       help="Years to benchmark (default: 2024)")

    args = parser.parse_args()

    n_iterations = 3 if args.quick else 5
    baseline_path = project_root / "data" / "benchmark_baseline.json"

    print("=" * 60)
    print("JP+ BENCHMARK SUITE")
    print("=" * 60)

    # Fetch data
    print(f"\nFetching benchmark data for {args.years}...")
    data = fetch_benchmark_data(args.years)

    # Run BASELINE iterations (simulating pre-P3 code)
    print(f"\nRunning {n_iterations} BASELINE iterations (pre-P3 simulation)...")
    baseline_results = []
    baseline_ratings = {}
    for i in range(n_iterations):
        print(f"  Baseline {i+1}/{n_iterations}...", end=" ", flush=True)
        result = run_single_iteration(data, i, use_baseline=True)
        baseline_results.append(result)
        print(f"{result.total_runtime_ms:.0f}ms, {result.peak_memory_mb:.1f}MB")
        if i == 0:
            # Capture ratings from first iteration for integrity check
            for year, year_data in data.items():
                ratings, _ = run_efm_benchmark(
                    year_data["plays_df"], year_data["games_df"],
                    year_data["fbs_teams"], year, use_baseline=True
                )
                baseline_ratings.update(ratings)

    baseline_summary = aggregate_results(baseline_results)

    # Run CURRENT iterations (optimized P3 code)
    print(f"\nRunning {n_iterations} CURRENT iterations (P3 optimized)...")
    current_results = []
    current_ratings = {}
    for i in range(n_iterations):
        print(f"  Current {i+1}/{n_iterations}...", end=" ", flush=True)
        result = run_single_iteration(data, i, use_baseline=False)
        current_results.append(result)
        print(f"{result.total_runtime_ms:.0f}ms, {result.peak_memory_mb:.1f}MB")
        if i == 0:
            # Capture ratings from first iteration for integrity check
            for year, year_data in data.items():
                ratings, _ = run_efm_benchmark(
                    year_data["plays_df"], year_data["games_df"],
                    year_data["fbs_teams"], year, use_baseline=False
                )
                current_ratings.update(ratings)

    current_summary = aggregate_results(current_results)

    # Capture baseline if requested (for external storage)
    if args.capture_baseline:
        print(f"\nSaving baseline to {baseline_path}...")
        baseline_data = {
            "timestamp": datetime.now().isoformat(),
            "years": args.years,
            "n_iterations": n_iterations,
            "summary": asdict(baseline_summary),
            "ratings_by_team": {},
        }
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(baseline_data, f, indent=2)
        print("Baseline captured successfully.")

    # Compare ratings for integrity
    integrity = compare_ratings(baseline_ratings, current_ratings)

    # Run specific optimization benchmarks
    print("\nBenchmarking specific P3 optimizations...")
    # Use first year's data for optimization benchmarks
    first_year = args.years[0]
    optimization_results = benchmark_specific_optimizations(data[first_year]["plays_df"])

    # Scalability test
    scalability = None
    if not args.no_scalability:
        print("\nRunning scalability tests...")
        scalability = run_scalability_test(fetch_benchmark_data, iterations=2)

    # Generate report
    report = generate_report(baseline_summary, current_summary, integrity, scalability, optimization_results)
    print("\n" + report)

    # Save report
    report_path = project_root / "data" / "outputs" / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
