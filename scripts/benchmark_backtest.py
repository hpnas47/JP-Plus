#!/usr/bin/env python3
"""Quick benchmark for backtest performance (week 4-10 of 2024 only)."""

import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Reduce logging noise
import logging
logging.basicConfig(level=logging.WARNING)

from backtest import fetch_all_season_data, walk_forward_predict

def benchmark():
    """Run quick benchmark on 2024 weeks 4-10."""
    print("Starting benchmark: 2024 weeks 4-10")
    print("=" * 60)

    start_time = time.time()

    # Fetch data
    print("Fetching data...")
    fetch_start = time.time()
    season_data = fetch_all_season_data([2024], use_priors=True, use_portal=True)
    fetch_time = time.time() - fetch_start
    print(f"  Data fetch: {fetch_time:.2f}s")

    # Extract data for 2024
    (games_df, betting_df, plays_df, turnover_df, priors,
     efficiency_plays_df, fbs_teams, st_plays_df, historical_rankings) = season_data[2024]

    # Run backtest
    print("\nRunning backtest (weeks 4-10)...")
    backtest_start = time.time()
    results = walk_forward_predict(
        games_df=games_df,
        efficiency_plays_df=efficiency_plays_df,
        fbs_teams=fbs_teams,
        start_week=4,
        end_week=10,
        preseason_priors=priors,
        hfa_value=2.5,
        prior_weight=8,
        ridge_alpha=50.0,
        efficiency_weight=0.54,
        explosiveness_weight=0.36,
        turnover_weight=0.10,
        year=2024,
        st_plays_df=st_plays_df,
        historical_rankings=historical_rankings,
    )
    backtest_time = time.time() - backtest_start
    print(f"  Backtest: {backtest_time:.2f}s")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"TOTAL TIME: {total_time:.2f}s")
    print(f"  Data fetch: {fetch_time:.2f}s ({fetch_time/total_time*100:.1f}%)")
    print(f"  Backtest:   {backtest_time:.2f}s ({backtest_time/total_time*100:.1f}%)")
    print(f"\nPredictions generated: {len(results)}")
    print("=" * 60)

if __name__ == "__main__":
    benchmark()
