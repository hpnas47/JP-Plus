#!/usr/bin/env python3
"""Unit tests for week-level caching."""

import sys
from pathlib import Path
import polars as pl
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.week_cache import WeekDataCache


def test_week_cache_basic():
    """Test basic week cache operations."""
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = WeekDataCache(cache_dir=tmpdir)

        # Test empty cache
        assert not cache.has_cached_week(2024, 1, "games")
        assert cache.load_week(2024, 1, "games") is None
        assert cache.get_cached_weeks(2024, "games") == []
        print("✓ Empty cache test passed")

        # Test save and load
        test_df = pl.DataFrame({
            "id": [1, 2, 3],
            "home_team": ["Georgia", "Alabama", "Ohio State"],
            "away_team": ["Florida", "LSU", "Michigan"],
        })

        cache.save_week(2024, 1, "games", test_df)
        assert cache.has_cached_week(2024, 1, "games")

        loaded_df = cache.load_week(2024, 1, "games")
        assert loaded_df is not None
        assert len(loaded_df) == 3
        assert loaded_df["home_team"].to_list() == ["Georgia", "Alabama", "Ohio State"]
        print("✓ Save/load test passed")

        # Test multiple weeks
        for week in range(1, 4):
            week_df = test_df.with_columns(
                pl.lit(week).alias("week")
            )
            cache.save_week(2024, week, "games", week_df)

        cached_weeks = cache.get_cached_weeks(2024, "games")
        assert cached_weeks == [1, 2, 3]
        print("✓ Multiple weeks test passed")

        # Test range loading
        combined_df = cache.load_weeks_range(2024, 1, 3, "games")
        assert combined_df is not None
        assert len(combined_df) == 9  # 3 rows × 3 weeks
        print("✓ Range loading test passed")

        # Test cache clearing
        cache.clear(year=2024, week=1)
        assert not cache.has_cached_week(2024, 1, "games")
        assert cache.has_cached_week(2024, 2, "games")
        print("✓ Cache clearing test passed")


def test_week_cache_missing_data():
    """Test handling of missing cache data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = WeekDataCache(cache_dir=tmpdir)

        # Save weeks 1 and 3 (skip week 2)
        test_df = pl.DataFrame({"id": [1]})
        cache.save_week(2024, 1, "games", test_df)
        cache.save_week(2024, 3, "games", test_df)

        # Range load should fail if any week missing
        combined = cache.load_weeks_range(2024, 1, 3, "games")
        assert combined is None  # Week 2 missing
        print("✓ Missing data handling test passed")


def test_week_cache_use_cache_flag():
    """Test use_cache flag functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = WeekDataCache(cache_dir=tmpdir)

        # Save data
        test_df = pl.DataFrame({"id": [1]})
        cache.save_week(2024, 1, "games", test_df)

        # With use_cache=True, should find cached data
        assert cache.has_cached_week(2024, 1, "games", use_cache=True)

        # With use_cache=False, should return False (forces refresh)
        assert not cache.has_cached_week(2024, 1, "games", use_cache=False)
        print("✓ use_cache flag test passed")


if __name__ == "__main__":
    print("\nRunning week cache tests...\n")

    try:
        test_week_cache_basic()
        test_week_cache_missing_data()
        test_week_cache_use_cache_flag()

        print("\n" + "=" * 50)
        print("✅ All week cache tests passed!")
        print("=" * 50 + "\n")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        sys.exit(1)
