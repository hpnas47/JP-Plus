"""High-level caching for complete season data.

Caches processed season DataFrames (games, plays, etc.) to eliminate
all API calls for historical seasons. This is more efficient than caching
individual API responses since processing happens once.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import polars as pl

logger = logging.getLogger(__name__)


class SeasonDataCache:
    """Cache for complete season DataFrames."""

    def __init__(self, cache_dir: str = ".cache/seasons"):
        """Initialize season data cache.

        Args:
            cache_dir: Directory to store cached season files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_year = datetime.now().year
        logger.debug(f"Initialized SeasonDataCache at {self.cache_dir.absolute()}")

    def _get_season_dir(self, year: int) -> Path:
        """Get directory for a specific season's cache files."""
        season_dir = self.cache_dir / str(year)
        season_dir.mkdir(parents=True, exist_ok=True)
        return season_dir

    def _is_historical(self, year: int) -> bool:
        """Check if a year is historical (completed season).

        Args:
            year: Season year

        Returns:
            True if year < current_year (historical data that won't change)
        """
        return year < self.current_year

    def has_cached_season(self, year: int, use_cache: bool = True) -> bool:
        """Check if a complete season is cached and valid.

        Args:
            year: Season year
            use_cache: If False, always return False (forces refresh)

        Returns:
            True if all season files exist and are valid
        """
        if not use_cache:
            return False

        # Current season: don't use cache (data changes weekly)
        # Historical seasons: use cache if available
        if not self._is_historical(year):
            return False

        season_dir = self._get_season_dir(year)
        required_files = [
            "games.parquet",
            "betting.parquet",
            "efficiency_plays.parquet",
            "turnover_plays.parquet",
            "st_plays.parquet",
        ]

        return all((season_dir / f).exists() for f in required_files)

    def load_season(self, year: int) -> Optional[Tuple[pl.DataFrame, ...]]:
        """Load a complete cached season.

        Args:
            year: Season year

        Returns:
            Tuple of (games_df, betting_df, efficiency_plays_df, turnover_plays_df, st_plays_df)
            or None if cache is invalid
        """
        if not self.has_cached_season(year):
            return None

        season_dir = self._get_season_dir(year)

        try:
            games_df = pl.read_parquet(season_dir / "games.parquet")
            betting_df = pl.read_parquet(season_dir / "betting.parquet")
            efficiency_plays_df = pl.read_parquet(season_dir / "efficiency_plays.parquet")
            turnover_plays_df = pl.read_parquet(season_dir / "turnover_plays.parquet")
            st_plays_df = pl.read_parquet(season_dir / "st_plays.parquet")

            logger.info(
                f"Cache HIT: Loaded {year} season "
                f"({len(games_df)} games, {len(efficiency_plays_df)} plays)"
            )
            return (games_df, betting_df, efficiency_plays_df, turnover_plays_df, st_plays_df)

        except Exception as e:
            logger.warning(f"Failed to load cached season {year}: {e}")
            return None

    def save_season(
        self,
        year: int,
        games_df: pl.DataFrame,
        betting_df: pl.DataFrame,
        efficiency_plays_df: pl.DataFrame,
        turnover_plays_df: pl.DataFrame,
        st_plays_df: pl.DataFrame,
    ):
        """Save a complete season to cache.

        Args:
            year: Season year
            games_df: Games DataFrame
            betting_df: Betting lines DataFrame
            efficiency_plays_df: Efficiency plays DataFrame
            turnover_plays_df: Turnover plays DataFrame
            st_plays_df: Special teams plays DataFrame
        """
        season_dir = self._get_season_dir(year)

        try:
            games_df.write_parquet(season_dir / "games.parquet")
            betting_df.write_parquet(season_dir / "betting.parquet")
            efficiency_plays_df.write_parquet(season_dir / "efficiency_plays.parquet")
            turnover_plays_df.write_parquet(season_dir / "turnover_plays.parquet")
            st_plays_df.write_parquet(season_dir / "st_plays.parquet")

            logger.info(
                f"Cache SAVE: Stored {year} season "
                f"({len(games_df)} games, {len(efficiency_plays_df)} plays)"
            )

        except Exception as e:
            logger.warning(f"Failed to save season {year} to cache: {e}")

    def clear(self, year: Optional[int] = None):
        """Clear cached seasons.

        Args:
            year: If provided, clear only this year. Otherwise clear all.
        """
        if year is not None:
            season_dir = self._get_season_dir(year)
            if season_dir.exists():
                for file in season_dir.glob("*.parquet"):
                    file.unlink()
                logger.info(f"Cleared cache for {year} season")
        else:
            for file in self.cache_dir.rglob("*.parquet"):
                file.unlink()
            logger.info("Cleared all season caches")

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache info per year and total size
        """
        stats = {
            "cache_dir": str(self.cache_dir.absolute()),
            "seasons": {},
            "total_size_mb": 0,
        }

        for year_dir in sorted(self.cache_dir.glob("*")):
            if not year_dir.is_dir():
                continue

            year = year_dir.name
            year_size = sum(f.stat().st_size for f in year_dir.glob("*.parquet"))
            file_count = len(list(year_dir.glob("*.parquet")))

            stats["seasons"][year] = {
                "files": file_count,
                "size_mb": year_size / (1024 * 1024),
                "complete": file_count >= 5,  # All 5 required files present
            }
            stats["total_size_mb"] += year_size / (1024 * 1024)

        return stats
