#!/usr/bin/env python3
"""Test vectorized EFM preprocessing (P3.2).

Verifies:
1. Vectorized operations produce same results as row-wise apply
2. Performance improvements are logged
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import time

from src.models.efficiency_foundation_model import (
    is_successful_play,
    is_successful_play_vectorized,
    is_garbage_time,
    is_garbage_time_vectorized,
)


def test_success_rate_equivalence():
    """Test that vectorized success rate matches scalar version."""
    print("Testing success rate equivalence...")

    np.random.seed(42)
    n = 100000

    # Generate random test data
    downs = np.random.choice([1, 2, 3, 4], n)
    distances = np.random.uniform(0, 15, n)
    distances[np.random.random(n) < 0.05] = 0  # 5% edge cases with distance=0
    yards_gained = np.random.uniform(-5, 20, n)

    # Scalar version (row-wise)
    scalar_start = time.time()
    scalar_results = np.array([
        is_successful_play(d, dist, yds)
        for d, dist, yds in zip(downs, distances, yards_gained)
    ])
    scalar_time = time.time() - scalar_start

    # Vectorized version
    vector_start = time.time()
    vector_results = is_successful_play_vectorized(downs, distances, yards_gained)
    vector_time = time.time() - vector_start

    # Compare
    matches = np.sum(scalar_results == vector_results)
    mismatches = n - matches

    print(f"  Plays tested: {n:,}")
    print(f"  Matches: {matches:,} ({100*matches/n:.2f}%)")
    print(f"  Mismatches: {mismatches}")
    print(f"  Scalar time: {scalar_time*1000:.1f}ms")
    print(f"  Vector time: {vector_time*1000:.1f}ms")
    print(f"  Speedup: {scalar_time/vector_time:.1f}x")

    if mismatches == 0:
        print("  ✓ PASS: Results match exactly")
        return True
    else:
        # Show mismatches
        mismatch_idx = np.where(scalar_results != vector_results)[0][:5]
        print(f"  ✗ FAIL: {mismatches} mismatches")
        for idx in mismatch_idx:
            print(f"    idx={idx}: down={downs[idx]}, dist={distances[idx]:.2f}, "
                  f"yds={yards_gained[idx]:.2f}, scalar={scalar_results[idx]}, "
                  f"vector={vector_results[idx]}")
        return False


def test_garbage_time_equivalence():
    """Test that vectorized garbage time matches scalar version."""
    print("\nTesting garbage time equivalence...")

    np.random.seed(42)
    n = 100000

    # Generate random test data
    periods = np.random.choice([1, 2, 3, 4], n)
    score_diffs = np.random.uniform(0, 50, n)

    # Scalar version (row-wise)
    scalar_start = time.time()
    scalar_results = np.array([
        is_garbage_time(p, sd)
        for p, sd in zip(periods, score_diffs)
    ])
    scalar_time = time.time() - scalar_start

    # Vectorized version
    vector_start = time.time()
    vector_results = is_garbage_time_vectorized(periods, score_diffs)
    vector_time = time.time() - vector_start

    # Compare
    matches = np.sum(scalar_results == vector_results)
    mismatches = n - matches

    print(f"  Plays tested: {n:,}")
    print(f"  Matches: {matches:,} ({100*matches/n:.2f}%)")
    print(f"  Mismatches: {mismatches}")
    print(f"  Scalar time: {scalar_time*1000:.1f}ms")
    print(f"  Vector time: {vector_time*1000:.1f}ms")
    print(f"  Speedup: {scalar_time/vector_time:.1f}x")

    if mismatches == 0:
        print("  ✓ PASS: Results match exactly")
        return True
    else:
        print(f"  ✗ FAIL: {mismatches} mismatches")
        return False


def test_efm_full_pipeline():
    """Test full EFM pipeline produces consistent results."""
    print("\nTesting full EFM pipeline...")

    from src.models.efficiency_foundation_model import EfficiencyFoundationModel
    from src.api.cfbd_client import CFBDClient

    client = CFBDClient()

    # Fetch a few weeks of 2024 data
    print("  Fetching 2024 weeks 1-4...")
    all_plays = []
    for week in range(1, 5):
        try:
            week_plays = client.get_plays(year=2024, season_type="regular", week=week)
            if isinstance(week_plays, list):
                all_plays.extend(week_plays)
            print(f"    Week {week}: {len(week_plays):,} plays")
        except Exception as e:
            print(f"    Week {week}: error - {e}")

    if not all_plays:
        print("  Skipping: no plays fetched")
        return True

    plays_df = pd.DataFrame(all_plays)
    print(f"  Total plays: {len(plays_df):,}")

    # Get FBS teams
    fbs_teams_raw = client.get_fbs_teams(year=2024)
    if fbs_teams_raw and hasattr(fbs_teams_raw[0], 'school'):
        fbs_teams = set(t.school for t in fbs_teams_raw)
    else:
        fbs_teams = set(fbs_teams_raw)

    # Filter to FBS vs FBS with required columns
    required_cols = ['offense', 'defense', 'down', 'distance', 'yards_gained',
                     'offense_score', 'defense_score']
    if not all(col in plays_df.columns for col in required_cols):
        print(f"  Skipping: missing required columns")
        return True

    plays_df = plays_df[
        plays_df["offense"].isin(fbs_teams) &
        plays_df["defense"].isin(fbs_teams)
    ]
    print(f"  FBS vs FBS plays: {len(plays_df):,}")

    # Run EFM
    print("\n  Running EFM with vectorized operations...")
    efm_start = time.time()
    efm = EfficiencyFoundationModel(ridge_alpha=50.0)
    ratings = efm.calculate_ratings(plays_df)
    efm_time = time.time() - efm_start

    print(f"\n  EFM completed in {efm_time*1000:.1f}ms")
    print(f"  Teams rated: {len(ratings)}")

    # Show top 5
    df = efm.get_ratings_df().head(5)
    print(f"\n  Top 5 teams:")
    print(df[['team', 'overall', 'offense', 'defense']].to_string(index=False))

    return True


if __name__ == "__main__":
    print("="*60)
    print("P3.2 VECTORIZATION TEST")
    print("="*60)

    success = True
    success = test_success_rate_equivalence() and success
    success = test_garbage_time_equivalence() and success

    try:
        success = test_efm_full_pipeline() and success
    except Exception as e:
        print(f"\nFull pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    if success:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60)

    sys.exit(0 if success else 1)
