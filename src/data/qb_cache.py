"""Cache for QB Continuous data.

Caches QB PPA and pass attempt data to eliminate API calls for historical
weeks. Historical seasons are cached permanently; current season weeks are
cached individually as they complete.

Cache Structure:
    .cache/qb/{year}/
        week_{week}_ppa.parquet      - QB PPA data for week
        week_{week}_attempts.parquet - Pass attempts for week
        prior_season.parquet         - End-of-season PPA for prior year
        .complete_{week}             - Marker for completed week
        .complete_prior              - Marker for prior season data

Follows the same atomic write pattern as SeasonDataCache for integrity.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class QBDataCache:
    """Cache for QB Continuous API data with atomic write guarantees."""

    def __init__(self, cache_dir: str = ".cache/qb"):
        """Initialize QB data cache.

        Args:
            cache_dir: Directory to store cached QB files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_year = datetime.now().year

        logger.debug(f"Initialized QBDataCache at {self.cache_dir.absolute()}")

    def _get_year_dir(self, year: int) -> Path:
        """Get directory for a specific year's cache files."""
        year_dir = self.cache_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        return year_dir

    def _is_historical_week(self, year: int, week: int) -> bool:
        """Check if a year/week is historical (completed).

        A week is considered historical if:
        - Year is before current year (all weeks complete), OR
        - We're past week 15 of current year (regular season done)

        Args:
            year: Season year
            week: Week number

        Returns:
            True if this week's data won't change
        """
        if year < self.current_year:
            return True
        # Current year: only cache completed weeks (be conservative)
        # In practice, data for a week is final ~2 days after games
        return False  # Don't cache current year to be safe

    def _is_historical_season(self, year: int) -> bool:
        """Check if a full season is historical."""
        return year < self.current_year

    # ========================================================================
    # Per-Week Cache (PPA + Attempts)
    # ========================================================================

    def has_week_data(self, year: int, week: int) -> bool:
        """Check if week data is cached and valid.

        Args:
            year: Season year
            week: Week number

        Returns:
            True if both PPA and attempts data are cached for this week
        """
        if not self._is_historical_week(year, week):
            return False

        year_dir = self._get_year_dir(year)
        marker = year_dir / f".complete_week_{week}"

        if not marker.exists():
            return False

        # Verify both files exist
        ppa_file = year_dir / f"week_{week}_ppa.parquet"
        attempts_file = year_dir / f"week_{week}_attempts.parquet"

        return ppa_file.exists() and attempts_file.exists()

    def load_week_ppa(self, year: int, week: int) -> Optional[list[dict]]:
        """Load cached PPA data for a week.

        Args:
            year: Season year
            week: Week number

        Returns:
            List of PPA dicts (same format as _fetch_qb_ppa_for_week returns),
            or None if not cached
        """
        if not self.has_week_data(year, week):
            return None

        year_dir = self._get_year_dir(year)
        ppa_file = year_dir / f"week_{week}_ppa.parquet"

        try:
            df = pd.read_parquet(ppa_file)
            # Convert back to list of dicts
            records = df.to_dict('records')
            logger.debug(f"QB Cache HIT: Loaded {year} week {week} PPA ({len(records)} records)")
            return records
        except Exception as e:
            logger.warning(f"Failed to load QB PPA cache for {year} week {week}: {e}")
            return None

    def load_week_attempts(self, year: int, week: int) -> Optional[dict[tuple[str, str], int]]:
        """Load cached pass attempts for a week.

        Args:
            year: Season year
            week: Week number

        Returns:
            Dict mapping (team, player_name) -> pass_attempts,
            or None if not cached
        """
        if not self.has_week_data(year, week):
            return None

        year_dir = self._get_year_dir(year)
        attempts_file = year_dir / f"week_{week}_attempts.parquet"

        try:
            df = pd.read_parquet(attempts_file)
            # Convert back to dict with tuple keys
            results = {}
            for _, row in df.iterrows():
                key = (row['team'], row['player_name'])
                results[key] = row['pass_attempts']
            logger.debug(f"QB Cache HIT: Loaded {year} week {week} attempts ({len(results)} records)")
            return results
        except Exception as e:
            logger.warning(f"Failed to load QB attempts cache for {year} week {week}: {e}")
            return None

    def save_week_data(
        self,
        year: int,
        week: int,
        ppa_data: list[dict],
        attempts_data: dict[tuple[str, str], int],
    ) -> None:
        """Save week data to cache using atomic write pattern.

        Args:
            year: Season year
            week: Week number
            ppa_data: List of PPA dicts from API
            attempts_data: Dict of (team, player) -> attempts from API
        """
        # Only cache historical data
        if not self._is_historical_week(year, week):
            return

        year_dir = self._get_year_dir(year)
        tmp_dir = year_dir / ".tmp"

        # Clean any existing temp directory
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Write PPA data to temp
            if ppa_data:
                ppa_df = pd.DataFrame(ppa_data)
            else:
                ppa_df = pd.DataFrame(columns=['player_id', 'player_name', 'team', 'year', 'week', 'ppa_pass', 'ppa_all'])
            ppa_df.to_parquet(tmp_dir / f"week_{week}_ppa.parquet", index=False)

            # Step 2: Write attempts data to temp
            attempts_records = [
                {'team': team, 'player_name': name, 'pass_attempts': attempts}
                for (team, name), attempts in attempts_data.items()
            ]
            if attempts_records:
                attempts_df = pd.DataFrame(attempts_records)
            else:
                attempts_df = pd.DataFrame(columns=['team', 'player_name', 'pass_attempts'])
            attempts_df.to_parquet(tmp_dir / f"week_{week}_attempts.parquet", index=False)

            # Step 3: Move files to final location
            for filename in [f"week_{week}_ppa.parquet", f"week_{week}_attempts.parquet"]:
                src = tmp_dir / filename
                dst = year_dir / filename
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))

            # Step 4: Clean temp directory
            shutil.rmtree(tmp_dir)

            # Step 5: Write completion marker LAST
            marker = year_dir / f".complete_week_{week}"
            marker.write_text(f"Cached: {datetime.now().isoformat()}\n")

            logger.info(f"QB Cache SAVE: {year} week {week} ({len(ppa_data)} PPA, {len(attempts_data)} attempts)")

        except Exception as e:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            logger.warning(f"Failed to save QB week cache for {year} week {week}: {e}")

    # ========================================================================
    # Prior Season Cache
    # ========================================================================

    def has_prior_season(self, year: int) -> bool:
        """Check if prior season data is cached.

        Args:
            year: The prior year (e.g., 2023 for 2024 season priors)

        Returns:
            True if prior season data is cached
        """
        if not self._is_historical_season(year):
            return False

        year_dir = self._get_year_dir(year)
        marker = year_dir / ".complete_prior"
        data_file = year_dir / "prior_season.parquet"

        return marker.exists() and data_file.exists()

    def load_prior_season(self, year: int) -> Optional[list[dict]]:
        """Load cached prior season QB data.

        Args:
            year: The prior year

        Returns:
            List of dicts with QB season stats, or None if not cached
        """
        if not self.has_prior_season(year):
            return None

        year_dir = self._get_year_dir(year)
        data_file = year_dir / "prior_season.parquet"

        try:
            df = pd.read_parquet(data_file)
            records = df.to_dict('records')
            logger.debug(f"QB Cache HIT: Loaded {year} prior season ({len(records)} QBs)")
            return records
        except Exception as e:
            logger.warning(f"Failed to load QB prior season cache for {year}: {e}")
            return None

    def save_prior_season(self, year: int, data: list[dict]) -> None:
        """Save prior season data to cache.

        Args:
            year: The prior year
            data: List of dicts with QB season stats
        """
        if not self._is_historical_season(year):
            return

        year_dir = self._get_year_dir(year)
        tmp_dir = year_dir / ".tmp"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Write data to temp
            if data:
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(columns=['player_id', 'player_name', 'team', 'year', 'total_dropbacks', 'total_pass_ppa'])
            df.to_parquet(tmp_dir / "prior_season.parquet", index=False)

            # Move to final location
            dst = year_dir / "prior_season.parquet"
            if dst.exists():
                dst.unlink()
            shutil.move(str(tmp_dir / "prior_season.parquet"), str(dst))

            # Clean temp
            shutil.rmtree(tmp_dir)

            # Write marker
            marker = year_dir / ".complete_prior"
            marker.write_text(f"Cached: {datetime.now().isoformat()}\n")

            logger.info(f"QB Cache SAVE: {year} prior season ({len(data)} QBs)")

        except Exception as e:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            logger.warning(f"Failed to save QB prior season cache for {year}: {e}")

    # ========================================================================
    # Cache Management
    # ========================================================================

    def clear(self, year: Optional[int] = None) -> None:
        """Clear cached QB data.

        Args:
            year: If provided, clear only this year. Otherwise clear all.
        """
        if year is not None:
            year_dir = self._get_year_dir(year)
            if year_dir.exists():
                shutil.rmtree(year_dir)
                logger.info(f"Cleared QB cache for {year}")
        else:
            for year_dir in self.cache_dir.glob("*"):
                if year_dir.is_dir():
                    shutil.rmtree(year_dir)
            logger.info("Cleared all QB caches")

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

            # Count cached weeks
            week_markers = list(year_dir.glob(".complete_week_*"))
            cached_weeks = [int(m.name.split("_")[-1]) for m in week_markers]

            # Check prior season
            has_prior = (year_dir / ".complete_prior").exists()

            stats["years"][year] = {
                "cached_weeks": sorted(cached_weeks),
                "num_weeks": len(cached_weeks),
                "has_prior_season": has_prior,
                "size_mb": year_size / (1024 * 1024),
            }
            stats["total_size_mb"] += year_size / (1024 * 1024)

        return stats
