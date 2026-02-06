"""Local disk cache for CFBD API responses.

Provides a transparent caching layer that stores API responses to disk
to eliminate redundant network calls during backtest runs.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


class CFBDCache:
    """Disk-based cache for CFBD API responses."""

    def __init__(self, cache_dir: str = ".cache/cfbd"):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cache files (relative to project root)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized CFBD cache at {self.cache_dir.absolute()}")

    def _get_cache_key(self, endpoint: str, **params) -> str:
        """Generate a unique cache key for an API call.

        Args:
            endpoint: API endpoint name (e.g., "games", "plays", "sp_ratings")
            **params: Query parameters (year, week, team, etc.)

        Returns:
            MD5 hash of the endpoint and sorted params
        """
        # Sort params for deterministic key generation
        sorted_params = sorted(params.items())
        key_data = f"{endpoint}::{sorted_params}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, endpoint: str, file_format: str, **params) -> Path:
        """Get the file path for a cached response.

        Args:
            endpoint: API endpoint name
            file_format: File format ("parquet", "json")
            **params: Query parameters

        Returns:
            Path to cache file
        """
        cache_key = self._get_cache_key(endpoint, **params)
        # Create subdirectory by year if year param exists
        if "year" in params:
            subdir = self.cache_dir / str(params["year"])
            subdir.mkdir(parents=True, exist_ok=True)
            return subdir / f"{endpoint}_{cache_key}.{file_format}"
        return self.cache_dir / f"{endpoint}_{cache_key}.{file_format}"

    def _is_cache_valid(self, cache_path: Path, ttl_hours: Optional[float] = None) -> bool:
        """Check if a cache file exists and is not stale.

        Args:
            cache_path: Path to cache file
            ttl_hours: Time-to-live in hours (None = never expires)

        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False

        if ttl_hours is None:
            return True

        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(hours=ttl_hours)

    def get_dataframe(
        self,
        endpoint: str,
        year: int,
        week: Optional[int] = None,
        ttl_hours: Optional[float] = None,
        **extra_params
    ) -> Optional[pl.DataFrame]:
        """Retrieve a cached Polars DataFrame.

        Args:
            endpoint: API endpoint name
            year: Season year
            week: Week number (optional)
            ttl_hours: Cache TTL in hours (None = never expires)
            **extra_params: Additional query parameters

        Returns:
            Cached DataFrame if valid, None otherwise
        """
        params = {"year": year, **extra_params}
        if week is not None:
            params["week"] = week

        cache_path = self._get_cache_path(endpoint, "parquet", **params)

        if not self._is_cache_valid(cache_path, ttl_hours):
            return None

        try:
            df = pl.read_parquet(cache_path)
            logger.debug(f"Cache HIT: {endpoint} {params} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Cache read failed for {cache_path}: {e}")
            return None

    def save_dataframe(
        self,
        df: pl.DataFrame,
        endpoint: str,
        year: int,
        week: Optional[int] = None,
        **extra_params
    ):
        """Save a Polars DataFrame to cache.

        Args:
            df: DataFrame to cache
            endpoint: API endpoint name
            year: Season year
            week: Week number (optional)
            **extra_params: Additional query parameters
        """
        params = {"year": year, **extra_params}
        if week is not None:
            params["week"] = week

        cache_path = self._get_cache_path(endpoint, "parquet", **params)

        try:
            df.write_parquet(cache_path)
            logger.debug(f"Cache SAVE: {endpoint} {params} ({len(df)} rows)")
        except Exception as e:
            logger.warning(f"Cache write failed for {cache_path}: {e}")

    def get_json(
        self,
        endpoint: str,
        year: int,
        ttl_hours: Optional[float] = None,
        **extra_params
    ) -> Optional[Any]:
        """Retrieve cached JSON data.

        Args:
            endpoint: API endpoint name
            year: Season year
            ttl_hours: Cache TTL in hours (None = never expires)
            **extra_params: Additional query parameters

        Returns:
            Cached data if valid, None otherwise
        """
        params = {"year": year, **extra_params}
        cache_path = self._get_cache_path(endpoint, "json", **params)

        if not self._is_cache_valid(cache_path, ttl_hours):
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            logger.debug(f"Cache HIT: {endpoint} {params}")
            return data
        except Exception as e:
            logger.warning(f"Cache read failed for {cache_path}: {e}")
            return None

    def save_json(
        self,
        data: Any,
        endpoint: str,
        year: int,
        **extra_params
    ):
        """Save JSON data to cache.

        Args:
            data: Data to cache (must be JSON-serializable)
            endpoint: API endpoint name
            year: Season year
            **extra_params: Additional query parameters
        """
        params = {"year": year, **extra_params}
        cache_path = self._get_cache_path(endpoint, "json", **params)

        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            logger.debug(f"Cache SAVE: {endpoint} {params}")
        except Exception as e:
            logger.warning(f"Cache write failed for {cache_path}: {e}")

    def clear(self, year: Optional[int] = None):
        """Clear cache files.

        Args:
            year: If provided, only clear cache for this year. Otherwise clear all.
        """
        if year is not None:
            year_dir = self.cache_dir / str(year)
            if year_dir.exists():
                for file in year_dir.glob("*"):
                    file.unlink()
                logger.info(f"Cleared cache for year {year}")
        else:
            for file in self.cache_dir.rglob("*"):
                if file.is_file():
                    file.unlink()
            logger.info("Cleared all cache files")

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache size and file counts
        """
        total_size = 0
        file_count = 0

        for file in self.cache_dir.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
                file_count += 1

        return {
            "file_count": file_count,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir.absolute()),
        }
