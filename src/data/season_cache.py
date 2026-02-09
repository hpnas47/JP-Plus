"""High-level caching for complete season data.

Caches processed season DataFrames (games, plays, etc.) to eliminate
all API calls for historical seasons. This is more efficient than caching
individual API responses since processing happens once.

Cache Integrity:
    Uses atomic write pattern to prevent corruption from interrupted writes:
    1. Write all files to .tmp/ subdirectory
    2. On success, move files to parent directory
    3. Write .complete marker file LAST
    4. has_cached_season() only returns True if .complete exists

    If interrupted mid-write, orphaned .tmp/ directories are cleaned on next run.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import polars as pl

logger = logging.getLogger(__name__)

# Completion marker file - written LAST after all data files are saved
COMPLETE_MARKER = ".complete"

# Required data files for a complete season cache
REQUIRED_FILES = [
    "games.parquet",
    "betting.parquet",
    "efficiency_plays.parquet",
    "turnover_plays.parquet",
    "st_plays.parquet",
]


class SeasonDataCache:
    """Cache for complete season DataFrames with atomic write guarantees."""

    def __init__(self, cache_dir: str = ".cache/seasons"):
        """Initialize season data cache.

        Args:
            cache_dir: Directory to store cached season files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_year = datetime.now().year

        # Clean up any orphaned .tmp directories from interrupted writes
        self._cleanup_orphaned_tmp()

        logger.debug(f"Initialized SeasonDataCache at {self.cache_dir.absolute()}")

    def _cleanup_orphaned_tmp(self):
        """Remove orphaned .tmp directories from interrupted writes."""
        for year_dir in self.cache_dir.glob("*"):
            if not year_dir.is_dir():
                continue
            tmp_dir = year_dir / ".tmp"
            if tmp_dir.exists():
                logger.warning(f"Cleaning orphaned temp directory: {tmp_dir}")
                shutil.rmtree(tmp_dir)

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

        A cache is considered valid only if:
        1. The .complete marker file exists (proves atomic write succeeded)
        2. All required parquet files exist
        3. The year is historical (current year data changes weekly)

        Args:
            year: Season year
            use_cache: If False, always return False (forces refresh)

        Returns:
            True if cache is complete and valid
        """
        if not use_cache:
            return False

        # Current season: don't use cache (data changes weekly)
        if not self._is_historical(year):
            return False

        season_dir = self._get_season_dir(year)

        # Check for completion marker FIRST (proves atomic write succeeded)
        if not (season_dir / COMPLETE_MARKER).exists():
            return False

        # Verify all required files exist
        return all((season_dir / f).exists() for f in REQUIRED_FILES)

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
            # Cache is corrupted - remove the completion marker
            complete_marker = season_dir / COMPLETE_MARKER
            if complete_marker.exists():
                complete_marker.unlink()
                logger.warning(f"Removed invalid completion marker for {year}")
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
        """Save a complete season to cache using atomic write pattern.

        Writes to a .tmp/ subdirectory first, then moves files to the
        final location and writes the .complete marker. This ensures
        that interrupted writes don't leave a corrupted cache.

        Args:
            year: Season year
            games_df: Games DataFrame
            betting_df: Betting lines DataFrame
            efficiency_plays_df: Efficiency plays DataFrame
            turnover_plays_df: Turnover plays DataFrame
            st_plays_df: Special teams plays DataFrame
        """
        season_dir = self._get_season_dir(year)
        tmp_dir = season_dir / ".tmp"

        # Clean any existing temp directory
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Write all files to temp directory
            games_df.write_parquet(tmp_dir / "games.parquet")
            betting_df.write_parquet(tmp_dir / "betting.parquet")
            efficiency_plays_df.write_parquet(tmp_dir / "efficiency_plays.parquet")
            turnover_plays_df.write_parquet(tmp_dir / "turnover_plays.parquet")
            st_plays_df.write_parquet(tmp_dir / "st_plays.parquet")

            # Step 2: Move files from temp to final location
            for filename in REQUIRED_FILES:
                src = tmp_dir / filename
                dst = season_dir / filename
                # Remove existing file if present (for force-refresh)
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))

            # Step 3: Remove temp directory
            shutil.rmtree(tmp_dir)

            # Step 4: Write completion marker LAST (atomic commit)
            complete_marker = season_dir / COMPLETE_MARKER
            complete_marker.write_text(
                f"Cached: {datetime.now().isoformat()}\n"
                f"Games: {len(games_df)}\n"
                f"Plays: {len(efficiency_plays_df)}\n"
            )

            logger.info(
                f"Cache SAVE: Stored {year} season "
                f"({len(games_df)} games, {len(efficiency_plays_df)} plays)"
            )

        except Exception as e:
            # Clean up on failure
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            logger.warning(f"Failed to save season {year} to cache: {e}")
            raise  # Re-raise so caller knows save failed

    def clear(self, year: Optional[int] = None):
        """Clear cached seasons.

        Args:
            year: If provided, clear only this year. Otherwise clear all.
        """
        if year is not None:
            season_dir = self._get_season_dir(year)
            if season_dir.exists():
                # Remove all parquet files and completion marker
                for file in season_dir.glob("*.parquet"):
                    file.unlink()
                complete_marker = season_dir / COMPLETE_MARKER
                if complete_marker.exists():
                    complete_marker.unlink()
                # Clean any orphaned temp directory
                tmp_dir = season_dir / ".tmp"
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                logger.info(f"Cleared cache for {year} season")
        else:
            for year_dir in self.cache_dir.glob("*"):
                if year_dir.is_dir():
                    shutil.rmtree(year_dir)
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

            # Check for completion marker (not just file count)
            has_marker = (year_dir / COMPLETE_MARKER).exists()
            is_complete = has_marker and file_count >= len(REQUIRED_FILES)

            stats["seasons"][year] = {
                "files": file_count,
                "size_mb": year_size / (1024 * 1024),
                "complete": is_complete,
                "has_marker": has_marker,
            }
            stats["total_size_mb"] += year_size / (1024 * 1024)

        return stats

    def migrate_legacy_cache(self) -> dict:
        """Migrate legacy cache entries by adding .complete markers.

        For caches created before the atomic write pattern was implemented,
        this method validates existing files and adds .complete markers
        if all required files exist and are readable.

        Returns:
            Dict with migration results per year
        """
        results = {"migrated": [], "failed": [], "skipped": []}

        for year_dir in sorted(self.cache_dir.glob("*")):
            if not year_dir.is_dir():
                continue

            year = year_dir.name
            complete_marker = year_dir / COMPLETE_MARKER

            # Skip if already has marker
            if complete_marker.exists():
                results["skipped"].append(year)
                continue

            # Check if all required files exist
            all_exist = all((year_dir / f).exists() for f in REQUIRED_FILES)
            if not all_exist:
                results["failed"].append(year)
                continue

            # Try to read each file to verify integrity
            try:
                games_df = pl.read_parquet(year_dir / "games.parquet")
                pl.read_parquet(year_dir / "betting.parquet")
                efficiency_plays_df = pl.read_parquet(year_dir / "efficiency_plays.parquet")
                pl.read_parquet(year_dir / "turnover_plays.parquet")
                pl.read_parquet(year_dir / "st_plays.parquet")

                # All files readable - add completion marker
                complete_marker.write_text(
                    f"Migrated: {datetime.now().isoformat()}\n"
                    f"Games: {len(games_df)}\n"
                    f"Plays: {len(efficiency_plays_df)}\n"
                )
                results["migrated"].append(year)
                logger.info(f"Migrated legacy cache for {year}")

            except Exception as e:
                logger.warning(f"Failed to migrate {year}: {e}")
                results["failed"].append(year)

        return results
