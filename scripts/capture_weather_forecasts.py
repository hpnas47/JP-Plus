#!/usr/bin/env python3
"""Capture weather forecasts for upcoming CFB games.

Fetches weather forecasts from tomorrow.io API for all outdoor FBS games
scheduled in the next 7 days. Stores forecasts in SQLite for use in
totals predictions.

RATE LIMITS (tomorrow.io free tier):
    - 25 calls/hour, 500 calls/day
    - For ~60-70 outdoor games, run in batches of 20 with --limit 20
    - Use --delay to increase wait time between calls (default: 3s)

Usage:
    # Capture forecasts for this week's games (respecting rate limits)
    python3 scripts/capture_weather_forecasts.py --limit 20

    # Capture for specific week
    python3 scripts/capture_weather_forecasts.py --year 2025 --week 10

    # Dry run (show games, don't fetch forecasts)
    python3 scripts/capture_weather_forecasts.py --dry-run

    # Refresh venue database from CFBD
    python3 scripts/capture_weather_forecasts.py --refresh-venues
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.cfbd_client import CFBDClient
from src.api.tomorrow_io import TomorrowIOClient, VenueLocation, is_dome_venue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Tomorrow.io API key
TOMORROW_IO_API_KEY = "kJlyTQ9lffZXbJlARF5zPwcMHymvxktU"


def refresh_venues(cfbd_client: CFBDClient, tomorrow_client: TomorrowIOClient) -> int:
    """Refresh venue database with coordinates from CFBD.

    Args:
        cfbd_client: CFBD API client
        tomorrow_client: Tomorrow.io client (for database access)

    Returns:
        Number of venues saved
    """
    logger.info("Fetching venues from CFBD API...")
    venues = cfbd_client.get_venues()

    count = 0
    for v in venues:
        # Extract coordinates directly from venue object
        lat = getattr(v, "latitude", None)
        lon = getattr(v, "longitude", None)

        if lat is None or lon is None:
            continue

        venue_name = v.name or "Unknown"
        cfbd_dome = getattr(v, "dome", False) or False

        venue = VenueLocation(
            venue_id=v.id,
            name=venue_name,
            city=v.city or "",
            state=v.state or "",
            latitude=lat,
            longitude=lon,
            dome=is_dome_venue(venue_name, cfbd_dome),  # Use fallback list for known domes
            elevation=getattr(v, "elevation", None),
            timezone=getattr(v, "timezone", None),
        )

        tomorrow_client.save_venue(venue)
        count += 1

    logger.info(f"Saved {count} venues with coordinates")
    return count


def get_upcoming_games(
    cfbd_client: CFBDClient,
    year: int,
    week: int,
) -> list:
    """Get upcoming games for a week.

    Args:
        cfbd_client: CFBD API client
        year: Season year
        week: Week number

    Returns:
        List of game objects with venue info
    """
    games = cfbd_client.get_games(year=year, week=week, season_type="regular")

    # Also check postseason if week > 15
    if week > 15:
        try:
            postseason = cfbd_client.get_games(
                year=year, week=week, season_type="postseason"
            )
            games.extend(postseason)
        except Exception:
            pass

    return games


def capture_forecasts(
    cfbd_client: CFBDClient,
    tomorrow_client: TomorrowIOClient,
    year: int,
    week: int,
    dry_run: bool = False,
    limit: int = 0,
) -> dict:
    """Capture weather forecasts for a week's games.

    Args:
        cfbd_client: CFBD API client
        tomorrow_client: Tomorrow.io client
        year: Season year
        week: Week number
        dry_run: If True, show games but don't fetch forecasts
        limit: Max forecasts to fetch (0 = no limit)

    Returns:
        Dict with capture statistics
    """
    games = get_upcoming_games(cfbd_client, year, week)
    logger.info(f"Found {len(games)} games for {year} week {week}")

    stats = {
        "total_games": len(games),
        "outdoor_games": 0,
        "indoor_games": 0,
        "forecasts_captured": 0,
        "forecasts_failed": 0,
        "rate_limited": 0,
        "venue_not_found": 0,
        "weather_concerns": [],
    }

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3  # Stop after 3 rate limits in a row

    for game in games:
        home_team = game.home_team
        away_team = game.away_team
        venue_id = getattr(game, "venue_id", None)
        venue_name = getattr(game, "venue", "Unknown")
        game_id = game.id

        # Parse game time
        start_date = getattr(game, "start_date", None)
        if start_date:
            if isinstance(start_date, str):
                try:
                    game_time = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                    game_time = game_time.replace(tzinfo=None)
                except ValueError:
                    game_time = datetime.now() + timedelta(days=2)
            else:
                game_time = start_date
                if hasattr(game_time, "tzinfo") and game_time.tzinfo:
                    game_time = game_time.replace(tzinfo=None)
        else:
            game_time = datetime.now() + timedelta(days=2)

        # Look up venue
        venue = tomorrow_client.get_venue(venue_id) if venue_id else None

        if venue is None:
            logger.warning(f"  {away_team} @ {home_team}: Venue not found (ID: {venue_id})")
            stats["venue_not_found"] += 1
            continue

        # Skip indoor games
        if venue.dome:
            logger.debug(f"  {away_team} @ {home_team}: Indoor (skipping)")
            stats["indoor_games"] += 1
            continue

        stats["outdoor_games"] += 1

        if dry_run:
            logger.info(
                f"  {away_team} @ {home_team} ({venue.name}) - "
                f"({venue.latitude:.4f}, {venue.longitude:.4f})"
            )
            continue

        # Check limit
        if limit > 0 and stats["forecasts_captured"] >= limit:
            logger.info(f"Reached limit of {limit} forecasts, stopping")
            break

        # Check for too many consecutive failures (likely rate limited)
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.error(
                f"Rate limit likely exhausted ({consecutive_failures} failures). "
                "Wait 1 hour or use --limit to batch requests."
            )
            stats["rate_limited"] = (
                stats["outdoor_games"] - stats["forecasts_captured"] - stats["forecasts_failed"]
            )
            break

        # Fetch forecast
        forecast = tomorrow_client.get_forecast(
            latitude=venue.latitude,
            longitude=venue.longitude,
            game_time=game_time,
            venue_id=venue.venue_id,
            venue_name=venue.name,
        )

        if forecast:
            # Mark as indoor if venue is dome
            forecast.is_indoor = venue.dome

            # Save to database
            tomorrow_client.save_forecast(forecast, game_id)
            stats["forecasts_captured"] += 1
            consecutive_failures = 0  # Reset on success

            # Check for weather concerns
            if tomorrow_client.is_weather_concern(forecast):
                concern = {
                    "game": f"{away_team} @ {home_team}",
                    "venue": venue.name,
                    "temp": forecast.temperature,
                    "wind": forecast.wind_speed,
                    "precip_prob": forecast.precipitation_probability,
                }
                stats["weather_concerns"].append(concern)
                logger.info(
                    f"  {away_team} @ {home_team}: WEATHER CONCERN - "
                    f"{forecast.temperature:.0f}°F, {forecast.wind_speed:.0f} mph wind, "
                    f"{forecast.precipitation_probability:.0f}% precip"
                )
            else:
                logger.info(
                    f"  {away_team} @ {home_team}: "
                    f"{forecast.temperature:.0f}°F, {forecast.wind_speed:.0f} mph wind"
                )
        else:
            stats["forecasts_failed"] += 1
            consecutive_failures += 1
            logger.warning(f"  {away_team} @ {home_team}: Failed to fetch forecast")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Capture weather forecasts for CFB games"
    )
    parser.add_argument(
        "--year", type=int, default=datetime.now().year,
        help="Season year (default: current year)"
    )
    parser.add_argument(
        "--week", type=int, default=None,
        help="Week number (default: auto-detect current week)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show games but don't fetch forecasts"
    )
    parser.add_argument(
        "--refresh-venues", action="store_true",
        help="Refresh venue database from CFBD"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max forecasts to fetch (0 = no limit, recommended: 20 per hour)"
    )
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="Seconds between API calls (default: 3.0, free tier safe)"
    )
    args = parser.parse_args()

    # Initialize clients
    cfbd_client = CFBDClient()
    tomorrow_client = TomorrowIOClient(
        api_key=TOMORROW_IO_API_KEY,
        rate_limit_delay=args.delay,
    )

    # Refresh venues if requested
    if args.refresh_venues:
        refresh_venues(cfbd_client, tomorrow_client)
        return

    # Auto-detect week if not specified
    if args.week is None:
        # Get current calendar week and estimate CFB week
        # CFB season typically starts late August (week ~1) through early January
        now = datetime.now()
        if now.month >= 8:
            # Fall season
            # Week 1 is usually last week of August / first week of September
            # Rough estimate: current week - 34 (week 35 = CFB week 1)
            week_of_year = now.isocalendar()[1]
            args.week = max(1, week_of_year - 34)
        else:
            # Bowl season / spring
            args.week = 1

        logger.info(f"Auto-detected week: {args.week}")

    # Capture forecasts
    print(f"\n{'='*60}")
    print(f"WEATHER FORECAST CAPTURE")
    print(f"{'='*60}")
    print(f"Year: {args.year}")
    print(f"Week: {args.week}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    if args.limit > 0:
        print(f"Limit: {args.limit} forecasts")
    print(f"Delay: {args.delay}s between calls")
    print(f"{'='*60}\n")

    stats = capture_forecasts(
        cfbd_client,
        tomorrow_client,
        args.year,
        args.week,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total games: {stats['total_games']}")
    print(f"Outdoor games: {stats['outdoor_games']}")
    print(f"Indoor games: {stats['indoor_games']}")
    print(f"Venues not found: {stats['venue_not_found']}")

    if not args.dry_run:
        print(f"Forecasts captured: {stats['forecasts_captured']}")
        print(f"Forecasts failed: {stats['forecasts_failed']}")
        if stats.get('rate_limited', 0) > 0:
            print(f"Rate limited (skipped): {stats['rate_limited']}")

        if stats["weather_concerns"]:
            print(f"\nWEATHER CONCERNS ({len(stats['weather_concerns'])} games):")
            for concern in stats["weather_concerns"]:
                print(
                    f"  - {concern['game']} @ {concern['venue']}: "
                    f"{concern['temp']:.0f}°F, {concern['wind']:.0f} mph, "
                    f"{concern['precip_prob']:.0f}% precip"
                )


if __name__ == "__main__":
    main()
