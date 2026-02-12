"""Cache for Preseason Priors API data.

Caches SP+, Talent Composite, Returning Production, and Transfer Portal
data to eliminate API calls for historical seasons. This data only changes
once per year (preseason), so historical years are cached permanently.

Cache Structure:
    .cache/priors/{year}/
        sp_ratings.parquet        - SP+ ratings from prior year
        talent.parquet            - Team talent composite
        returning_production.parquet - Percent PPA returning
        transfer_portal.parquet   - Transfer portal entries
        .complete                 - Marker for fully cached year

Follows the same atomic write pattern as SeasonDataCache for integrity.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PriorsDataCache:
    """Cache for preseason priors API data with atomic write guarantees."""

    def __init__(self, cache_dir: str = ".cache/priors"):
        """Initialize priors data cache.

        Args:
            cache_dir: Directory to store cached priors files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_year = datetime.now().year

        logger.debug(f"Initialized PriorsDataCache at {self.cache_dir.absolute()}")

    def _get_year_dir(self, year: int) -> Path:
        """Get directory for a specific year's cache files."""
        year_dir = self.cache_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        return year_dir

    def _is_historical(self, year: int) -> bool:
        """Check if a year is historical (completed preseason).

        Priors data is stable once the season starts, so we cache
        any year before the current year.
        """
        return year < self.current_year

    # ========================================================================
    # SP+ Ratings Cache
    # ========================================================================

    def has_sp_ratings(self, year: int) -> bool:
        """Check if SP+ ratings are cached for a year."""
        if not self._is_historical(year):
            return False
        year_dir = self._get_year_dir(year)
        return (year_dir / "sp_ratings.parquet").exists()

    def load_sp_ratings(self, year: int) -> Optional[dict[str, float]]:
        """Load cached SP+ ratings.

        Args:
            year: Season year (loads SP+ from year-1 as prior)

        Returns:
            Dict mapping team name to SP+ overall rating, or None if not cached
        """
        if not self.has_sp_ratings(year):
            return None

        year_dir = self._get_year_dir(year)
        try:
            df = pd.read_parquet(year_dir / "sp_ratings.parquet")
            result = dict(zip(df['team'], df['rating']))
            logger.info(f"Priors Cache HIT: Loaded {year} SP+ ratings ({len(result)} teams)")
            return result
        except Exception as e:
            logger.warning(f"Failed to load SP+ cache for {year}: {e}")
            return None

    def save_sp_ratings(self, year: int, ratings: dict[str, float]) -> None:
        """Save SP+ ratings to cache.

        Args:
            year: Season year
            ratings: Dict mapping team name to SP+ rating
        """
        if not self._is_historical(year):
            return

        year_dir = self._get_year_dir(year)
        try:
            df = pd.DataFrame([
                {'team': team, 'rating': rating}
                for team, rating in ratings.items()
            ])
            df.to_parquet(year_dir / "sp_ratings.parquet", index=False)
            logger.info(f"Priors Cache SAVE: {year} SP+ ratings ({len(ratings)} teams)")
        except Exception as e:
            logger.warning(f"Failed to save SP+ cache for {year}: {e}")

    # ========================================================================
    # Talent Composite Cache
    # ========================================================================

    def has_talent(self, year: int) -> bool:
        """Check if talent composite is cached for a year."""
        if not self._is_historical(year):
            return False
        year_dir = self._get_year_dir(year)
        return (year_dir / "talent.parquet").exists()

    def load_talent(self, year: int) -> Optional[dict[str, float]]:
        """Load cached talent composite.

        Args:
            year: Season year

        Returns:
            Dict mapping team name to talent score, or None if not cached
        """
        if not self.has_talent(year):
            return None

        year_dir = self._get_year_dir(year)
        try:
            df = pd.read_parquet(year_dir / "talent.parquet")
            result = dict(zip(df['team'], df['talent']))
            logger.info(f"Priors Cache HIT: Loaded {year} talent ({len(result)} teams)")
            return result
        except Exception as e:
            logger.warning(f"Failed to load talent cache for {year}: {e}")
            return None

    def save_talent(self, year: int, talent: dict[str, float]) -> None:
        """Save talent composite to cache.

        Args:
            year: Season year
            talent: Dict mapping team name to talent score
        """
        if not self._is_historical(year):
            return

        year_dir = self._get_year_dir(year)
        try:
            df = pd.DataFrame([
                {'team': team, 'talent': score}
                for team, score in talent.items()
            ])
            df.to_parquet(year_dir / "talent.parquet", index=False)
            logger.info(f"Priors Cache SAVE: {year} talent ({len(talent)} teams)")
        except Exception as e:
            logger.warning(f"Failed to save talent cache for {year}: {e}")

    # ========================================================================
    # Returning Production Cache
    # ========================================================================

    def has_returning_production(self, year: int) -> bool:
        """Check if returning production is cached for a year."""
        if not self._is_historical(year):
            return False
        year_dir = self._get_year_dir(year)
        return (year_dir / "returning_production.parquet").exists()

    def load_returning_production(self, year: int) -> Optional[dict[str, float]]:
        """Load cached returning production.

        Args:
            year: Season year

        Returns:
            Dict mapping team name to percent_ppa (0-1), or None if not cached
        """
        if not self.has_returning_production(year):
            return None

        year_dir = self._get_year_dir(year)
        try:
            df = pd.read_parquet(year_dir / "returning_production.parquet")
            result = dict(zip(df['team'], df['percent_ppa']))
            logger.info(f"Priors Cache HIT: Loaded {year} returning production ({len(result)} teams)")
            return result
        except Exception as e:
            logger.warning(f"Failed to load returning production cache for {year}: {e}")
            return None

    def save_returning_production(self, year: int, rp: dict[str, float]) -> None:
        """Save returning production to cache.

        Args:
            year: Season year
            rp: Dict mapping team name to percent_ppa
        """
        if not self._is_historical(year):
            return

        year_dir = self._get_year_dir(year)
        try:
            df = pd.DataFrame([
                {'team': team, 'percent_ppa': ppa}
                for team, ppa in rp.items()
            ])
            df.to_parquet(year_dir / "returning_production.parquet", index=False)
            logger.info(f"Priors Cache SAVE: {year} returning production ({len(rp)} teams)")
        except Exception as e:
            logger.warning(f"Failed to save returning production cache for {year}: {e}")

    # ========================================================================
    # Transfer Portal Cache
    # ========================================================================

    def has_transfer_portal(self, year: int) -> bool:
        """Check if transfer portal is cached for a year."""
        if not self._is_historical(year):
            return False
        year_dir = self._get_year_dir(year)
        return (year_dir / "transfer_portal.parquet").exists()

    def load_transfer_portal(self, year: int) -> Optional[pd.DataFrame]:
        """Load cached transfer portal data.

        Args:
            year: Season year

        Returns:
            DataFrame with transfer portal entries, or None if not cached
        """
        if not self.has_transfer_portal(year):
            return None

        year_dir = self._get_year_dir(year)
        try:
            df = pd.read_parquet(year_dir / "transfer_portal.parquet")
            logger.info(f"Priors Cache HIT: Loaded {year} transfer portal ({len(df)} entries)")
            return df
        except Exception as e:
            logger.warning(f"Failed to load transfer portal cache for {year}: {e}")
            return None

    def save_transfer_portal(self, year: int, df: pd.DataFrame) -> None:
        """Save transfer portal data to cache.

        Args:
            year: Season year
            df: DataFrame with transfer portal entries
        """
        if not self._is_historical(year):
            return

        year_dir = self._get_year_dir(year)
        try:
            df.to_parquet(year_dir / "transfer_portal.parquet", index=False)
            logger.info(f"Priors Cache SAVE: {year} transfer portal ({len(df)} entries)")
        except Exception as e:
            logger.warning(f"Failed to save transfer portal cache for {year}: {e}")

    # ========================================================================
    # Cache Management
    # ========================================================================

    def clear(self, year: Optional[int] = None) -> None:
        """Clear cached priors data.

        Args:
            year: If provided, clear only this year. Otherwise clear all.
        """
        if year is not None:
            year_dir = self._get_year_dir(year)
            if year_dir.exists():
                shutil.rmtree(year_dir)
                logger.info(f"Cleared priors cache for {year}")
        else:
            for year_dir in self.cache_dir.glob("*"):
                if year_dir.is_dir():
                    shutil.rmtree(year_dir)
            logger.info("Cleared all priors caches")

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache info per year
        """
        stats = {
            "cache_dir": str(self.cache_dir.absolute()),
            "years": {},
            "total_size_mb": 0,
        }

        for year_dir in sorted(self.cache_dir.glob("*")):
            if not year_dir.is_dir():
                continue

            year = year_dir.name
            year_size = sum(f.stat().st_size for f in year_dir.glob("*.parquet"))

            stats["years"][year] = {
                "has_sp_ratings": (year_dir / "sp_ratings.parquet").exists(),
                "has_talent": (year_dir / "talent.parquet").exists(),
                "has_returning_production": (year_dir / "returning_production.parquet").exists(),
                "has_transfer_portal": (year_dir / "transfer_portal.parquet").exists(),
                "size_mb": year_size / (1024 * 1024),
            }
            stats["total_size_mb"] += year_size / (1024 * 1024)

        return stats
