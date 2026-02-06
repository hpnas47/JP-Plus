#!/usr/bin/env python3
"""
Pre-fetch and cache all CFB data for backtest range.

This script populates the local cache with all data needed for backtesting,
eliminating API calls during actual backtest runs.

Usage:
    python3 scripts/ensure_data.py                    # Cache 2022-2025
    python3 scripts/ensure_data.py --years 2024 2025  # Cache specific years
    python3 scripts/ensure_data.py --force-refresh    # Force re-download all data
    python3 scripts/ensure_data.py --year 2024        # Cache single year
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.season_cache import SeasonDataCache

# Import the backtest fetcher (which handles all the data processing)
from scripts.backtest import fetch_all_season_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_data(
    years: list[int],
    force_refresh: bool = False,
    use_priors: bool = True,
    use_portal: bool = True,
    portal_scale: float = 0.15,
):
    """Ensure all data for specified years is cached locally.

    Args:
        years: List of years to cache
        force_refresh: If True, re-download even if cached
        use_priors: Whether to build preseason priors
        use_portal: Whether to include portal data in priors
        portal_scale: Portal impact scale
    """
    cache = SeasonDataCache()

    logger.info("=" * 70)
    logger.info("CFB Data Pre-Fetch Utility")
    logger.info("=" * 70)
    logger.info(f"Target years:    {years}")
    logger.info(f"Force refresh:   {force_refresh}")
    logger.info(f"Preseason priors: {use_priors}")
    logger.info(f"Transfer portal:  {use_portal} (scale={portal_scale})")
    logger.info("=" * 70 + "\n")

    # Check current cache status
    logger.info("Current cache status:")
    stats = cache.get_stats()
    if stats["seasons"]:
        for year, info in sorted(stats["seasons"].items()):
            status = "✓ COMPLETE" if info["complete"] else "⚠ INCOMPLETE"
            logger.info(f"  {year}: {status} ({info['files']} files, {info['size_mb']:.1f} MB)")
    else:
        logger.info("  Cache is empty")
    logger.info(f"Total cache size: {stats['total_size_mb']:.1f} MB\n")

    # Determine which years need fetching
    years_to_fetch = []
    for year in years:
        if force_refresh or not cache.has_cached_season(year):
            years_to_fetch.append(year)
            logger.info(f"Will fetch {year} (force_refresh={force_refresh}, cached={cache.has_cached_season(year)})")
        else:
            logger.info(f"Skipping {year} (already cached, use --force-refresh to re-download)")

    if not years_to_fetch:
        logger.info("\n✓ All requested years are already cached. Nothing to do.")
        logger.info("  Use --force-refresh to re-download data.")
        return

    logger.info(f"\nFetching data for {len(years_to_fetch)} year(s): {years_to_fetch}")
    logger.info("This will take 30-60 seconds per year (API network latency)...\n")

    # Fetch and cache data
    try:
        season_data = fetch_all_season_data(
            years_to_fetch,
            use_priors=use_priors,
            use_portal=use_portal,
            portal_scale=portal_scale,
            use_cache=not force_refresh,  # If force_refresh, don't read from cache
            force_refresh=force_refresh,
        )

        logger.info("\n" + "=" * 70)
        logger.info("✓ Data fetch complete!")
        logger.info("=" * 70)

        # Print updated cache stats
        stats = cache.get_stats()
        logger.info(f"\nCache summary:")
        logger.info(f"  Total size:  {stats['total_size_mb']:.1f} MB")
        logger.info(f"  Cached years: {len(stats['seasons'])}")
        for year, info in sorted(stats["seasons"].items()):
            status = "✓ COMPLETE" if info["complete"] else "⚠ INCOMPLETE"
            logger.info(f"    {year}: {status} ({info['size_mb']:.1f} MB)")

        logger.info(f"\n✓ Cache ready for backtest use!")
        logger.info(f"  Run: python3 scripts/backtest.py --years {' '.join(map(str, years))}")

    except Exception as e:
        logger.error(f"\n✗ Data fetch failed: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-fetch and cache CFB data for backtest runs"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024, 2025],
        help="Years to cache (default: 2022 2023 2024 2025)",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Single year to cache (convenience shorthand)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download all data (ignores existing cache)",
    )
    parser.add_argument(
        "--no-priors",
        action="store_true",
        help="Skip preseason priors calculation (faster, but less complete)",
    )
    parser.add_argument(
        "--no-portal",
        action="store_true",
        help="Disable transfer portal adjustment in preseason priors",
    )
    parser.add_argument(
        "--portal-scale",
        type=float,
        default=0.15,
        help="Transfer portal impact scale (default: 0.15)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cached data and exit (does not fetch)",
    )

    args = parser.parse_args()

    # Handle --clear flag
    if args.clear:
        cache = SeasonDataCache()
        logger.info("Clearing all cached data...")
        cache.clear()
        logger.info("✓ Cache cleared.")
        return

    # Handle --year shorthand
    years = [args.year] if args.year else args.years

    # Run the data fetch
    ensure_data(
        years=years,
        force_refresh=args.force_refresh,
        use_priors=not args.no_priors,
        use_portal=not args.no_portal,
        portal_scale=args.portal_scale,
    )


if __name__ == "__main__":
    main()
