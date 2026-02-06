"""Week-level caching for incremental data fetching.

Enables delta loads where only the current week is fetched from API,
and all historical weeks are loaded from locked cache. This dramatically
reduces API calls during production weekly runs.
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


class WeekDataCache:
    """Cache for week-level DataFrames to enable delta loading."""

    def __init__(self, cache_dir: str = ".cache/weeks"):
        """Initialize week data cache.

        Args:
            cache_dir: Directory to store cached week files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized WeekDataCache at {self.cache_dir.absolute()}")

    def _get_week_dir(self, year: int, week: int) -> Path:
        """Get directory for a specific week's cache files."""
        week_dir = self.cache_dir / str(year) / f"week_{week:02d}"
        week_dir.mkdir(parents=True, exist_ok=True)
        return week_dir

    def has_cached_week(
        self, year: int, week: int, data_type: str, use_cache: bool = True
    ) -> bool:
        """Check if a specific week's data is cached.

        Args:
            year: Season year
            week: Week number
            data_type: Type of data ("games", "betting", "plays", "turnovers", "st_plays")
            use_cache: If False, always return False (forces refresh)

        Returns:
            True if the week file exists
        """
        if not use_cache:
            return False

        week_dir = self._get_week_dir(year, week)
        cache_file = week_dir / f"{data_type}.parquet"
        return cache_file.exists()

    def load_week(
        self, year: int, week: int, data_type: str
    ) -> Optional[pl.DataFrame]:
        """Load cached data for a specific week.

        Args:
            year: Season year
            week: Week number
            data_type: Type of data to load

        Returns:
            DataFrame if cached, None otherwise
        """
        if not self.has_cached_week(year, week, data_type):
            return None

        week_dir = self._get_week_dir(year, week)
        cache_file = week_dir / f"{data_type}.parquet"

        try:
            df = pl.read_parquet(cache_file)
            logger.debug(
                f"Week cache HIT: {year} week {week} {data_type} ({len(df)} rows)"
            )
            return df
        except Exception as e:
            logger.warning(f"Failed to load cached week {year}-{week} {data_type}: {e}")
            return None

    def save_week(
        self, year: int, week: int, data_type: str, df: pl.DataFrame
    ):
        """Save data for a specific week to cache.

        Args:
            year: Season year
            week: Week number
            data_type: Type of data
            df: DataFrame to cache
        """
        week_dir = self._get_week_dir(year, week)
        cache_file = week_dir / f"{data_type}.parquet"

        try:
            df.write_parquet(cache_file)
            logger.debug(
                f"Week cache SAVE: {year} week {week} {data_type} ({len(df)} rows)"
            )
        except Exception as e:
            logger.warning(f"Failed to save week {year}-{week} {data_type}: {e}")

    def load_weeks_range(
        self, year: int, start_week: int, end_week: int, data_type: str
    ) -> Optional[pl.DataFrame]:
        """Load and concatenate multiple weeks.

        Args:
            year: Season year
            start_week: First week (inclusive)
            end_week: Last week (inclusive)
            data_type: Type of data to load

        Returns:
            Concatenated DataFrame if all weeks cached, None if any week missing
        """
        dfs = []
        for week in range(start_week, end_week + 1):
            df = self.load_week(year, week, data_type)
            if df is None:
                return None  # Missing week, can't use cache
            dfs.append(df)

        if not dfs:
            return pl.DataFrame()

        try:
            combined = pl.concat(dfs)
            logger.info(
                f"Week cache: Loaded weeks {start_week}-{end_week} {data_type} "
                f"({len(combined)} total rows)"
            )
            return combined
        except Exception as e:
            logger.warning(f"Failed to concatenate weeks {start_week}-{end_week}: {e}")
            return None

    def clear(self, year: Optional[int] = None, week: Optional[int] = None):
        """Clear cached weeks.

        Args:
            year: If provided, only clear this year (if week also provided, only that week)
            week: If provided with year, only clear this specific week
        """
        if year is not None and week is not None:
            week_dir = self._get_week_dir(year, week)
            if week_dir.exists():
                for file in week_dir.glob("*.parquet"):
                    file.unlink()
                logger.info(f"Cleared cache for {year} week {week}")
        elif year is not None:
            year_dir = self.cache_dir / str(year)
            if year_dir.exists():
                for file in year_dir.rglob("*.parquet"):
                    file.unlink()
                logger.info(f"Cleared cache for {year} season")
        else:
            for file in self.cache_dir.rglob("*.parquet"):
                file.unlink()
            logger.info("Cleared all week caches")

    def get_cached_weeks(self, year: int, data_type: str) -> list[int]:
        """Get list of weeks that are cached for a given year and data type.

        Args:
            year: Season year
            data_type: Type of data to check

        Returns:
            Sorted list of week numbers that are cached
        """
        year_dir = self.cache_dir / str(year)
        if not year_dir.exists():
            return []

        cached_weeks = []
        for week_dir in sorted(year_dir.glob("week_*")):
            cache_file = week_dir / f"{data_type}.parquet"
            if cache_file.exists():
                week_num = int(week_dir.name.split("_")[1])
                cached_weeks.append(week_num)

        return sorted(cached_weeks)
