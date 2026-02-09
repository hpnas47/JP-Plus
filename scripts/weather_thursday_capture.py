#!/usr/bin/env python3
"""Thursday weather forecast capture for totals betting edge.

This script runs the full Thursday "Setup" capture workflow:
1. Captures forecasts for all outdoor games (batched for rate limits)
2. Identifies "Watchlist" games with high wind/weather concern
3. Saves results to database and logs watchlist to console

Schedule this to run Thursday mornings before limits increase at books.

Usage:
    # Manual run
    python3 scripts/weather_thursday_capture.py

    # With specific week (otherwise auto-detects)
    python3 scripts/weather_thursday_capture.py --week 14

    # Dry run (show what would be captured)
    python3 scripts/weather_thursday_capture.py --dry-run

Cron example (every Thursday at 6 AM):
    0 6 * * 4 cd /path/to/project && python3 scripts/weather_thursday_capture.py >> logs/weather_thursday.log 2>&1
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.cfbd_client import CFBDClient
from src.api.tomorrow_io import TomorrowIOClient, VenueLocation
from src.adjustments.weather import WeatherAdjuster, WeatherConditions

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "weather_thursday.log"),
    ]
)
logger = logging.getLogger(__name__)

# Tomorrow.io API key
TOMORROW_IO_API_KEY = "kJlyTQ9lffZXbJlARF5zPwcMHymvxktU"

# Rate limit settings
BATCH_SIZE = 20  # Games per batch (stay under 25/hour limit)
BATCH_DELAY_SECONDS = 3600  # 1 hour between batches (if needed)


def get_current_cfb_week() -> tuple[int, int]:
    """Auto-detect current CFB season year and week.

    Returns:
        Tuple of (year, week)
    """
    now = datetime.now()

    # CFB season: late August through early January
    if now.month >= 8:
        year = now.year
        # Week 1 is usually last week of August / first week of September
        # Rough estimate: calendar week - 34 (week 35 = CFB week 1)
        week_of_year = now.isocalendar()[1]
        week = max(1, min(week_of_year - 34, 15))
    elif now.month <= 1:
        # Bowl season - previous year's season
        year = now.year - 1
        week = 16  # Postseason
    else:
        # Off-season
        year = now.year
        week = 1

    return year, week


def capture_with_watchlist(
    cfbd_client: CFBDClient,
    tomorrow_client: TomorrowIOClient,
    year: int,
    week: int,
    dry_run: bool = False,
) -> dict:
    """Capture forecasts and build watchlist of weather-impacted games.

    Args:
        cfbd_client: CFBD API client
        tomorrow_client: Tomorrow.io client
        year: Season year
        week: Week number
        dry_run: If True, show games but don't fetch forecasts

    Returns:
        Dict with capture stats and watchlist
    """
    # Get games for the week
    games = cfbd_client.get_games(year=year, week=week, season_type="regular")

    # Get betting lines for over/under totals
    betting_lines = {}
    try:
        all_lines = cfbd_client.get_betting_lines(year=year, week=week, season_type="regular")
        for game_lines in (all_lines or []):
            # Find the best available line with over_under
            for line in (game_lines.lines or []):
                ou = getattr(line, 'over_under', None)
                if ou is not None:
                    betting_lines[game_lines.id] = ou
                    break
        logger.info(f"Found {len(betting_lines)} games with over/under totals")
    except Exception as e:
        logger.warning(f"Could not fetch betting lines: {e}")

    # Also check postseason if week > 15
    if week > 15:
        try:
            postseason = cfbd_client.get_games(year=year, week=week, season_type="postseason")
            games.extend(postseason)
        except Exception:
            pass

    logger.info(f"Found {len(games)} games for {year} week {week}")

    stats = {
        "year": year,
        "week": week,
        "capture_time": datetime.now().isoformat(),
        "total_games": len(games),
        "outdoor_games": 0,
        "indoor_games": 0,
        "forecasts_captured": 0,
        "forecasts_failed": 0,
        "venue_not_found": 0,
        "watchlist": [],  # Games with weather concerns
    }

    weather_adjuster = WeatherAdjuster()
    outdoor_games = []

    # First pass: identify outdoor games
    for game in games:
        venue_id = getattr(game, "venue_id", None)
        venue = tomorrow_client.get_venue(venue_id) if venue_id else None

        if venue is None:
            stats["venue_not_found"] += 1
            continue

        if venue.dome:
            stats["indoor_games"] += 1
            continue

        stats["outdoor_games"] += 1
        outdoor_games.append((game, venue))

    logger.info(f"Outdoor games to capture: {len(outdoor_games)}")

    if dry_run:
        for game, venue in outdoor_games:
            logger.info(f"  [DRY RUN] {game.away_team} @ {game.home_team} ({venue.name})")
        return stats

    # Capture forecasts (respecting rate limits)
    for i, (game, venue) in enumerate(outdoor_games):
        # Parse game time
        start_date = getattr(game, "start_date", None)
        if start_date:
            if isinstance(start_date, str):
                try:
                    game_time = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                    game_time = game_time.replace(tzinfo=None)
                except ValueError:
                    game_time = datetime.now()
            else:
                game_time = start_date
                if hasattr(game_time, "tzinfo") and game_time.tzinfo:
                    game_time = game_time.replace(tzinfo=None)
        else:
            game_time = datetime.now()

        # Fetch forecast
        forecast = tomorrow_client.get_forecast(
            latitude=venue.latitude,
            longitude=venue.longitude,
            game_time=game_time,
            venue_id=venue.venue_id,
            venue_name=venue.name,
        )

        if forecast:
            forecast.is_indoor = False
            tomorrow_client.save_forecast(forecast, game.id)
            stats["forecasts_captured"] += 1

            # Check if this is a watchlist game
            if tomorrow_client.is_weather_concern(forecast):
                # Calculate the adjustment
                conditions = tomorrow_client.forecast_to_weather_conditions(forecast, game.id)
                adjustment = weather_adjuster.calculate_adjustment(conditions)

                # Get Vegas total for this game
                vegas_total = betting_lines.get(game.id)
                adjusted_total = None
                if vegas_total is not None:
                    adjusted_total = vegas_total + adjustment.total_adjustment

                watchlist_entry = {
                    "game_id": game.id,
                    "matchup": f"{game.away_team} @ {game.home_team}",
                    "venue": venue.name,
                    "game_time": game_time.isoformat(),
                    "temperature": forecast.temperature,
                    "wind_speed": forecast.wind_speed,
                    "wind_gust": forecast.wind_gust,
                    "effective_wind": (
                        (forecast.wind_speed + forecast.wind_gust) / 2
                        if forecast.wind_gust else forecast.wind_speed
                    ),
                    "precip_prob": forecast.precipitation_probability,
                    "weather_code": forecast.weather_code,
                    "vegas_total": vegas_total,
                    "adjusted_total": adjusted_total,
                    "total_adjustment": adjustment.total_adjustment,
                    "wind_adjustment": adjustment.wind_adjustment,
                    "temp_adjustment": adjustment.temperature_adjustment,
                    "precip_adjustment": adjustment.precipitation_adjustment,
                    "confidence": forecast.confidence_factor,
                    "hours_until_game": forecast.hours_until_game,
                }
                stats["watchlist"].append(watchlist_entry)

                logger.info(
                    f"  🚨 WATCHLIST: {game.away_team} @ {game.home_team} | "
                    f"Wind: {forecast.wind_speed:.0f}/{forecast.wind_gust or 0:.0f} mph, "
                    f"Temp: {forecast.temperature:.0f}°F | "
                    f"Adjustment: {adjustment.total_adjustment:+.1f} pts"
                )
            else:
                logger.info(
                    f"  ✓ {game.away_team} @ {game.home_team}: "
                    f"{forecast.temperature:.0f}°F, {forecast.wind_speed:.0f} mph wind"
                )
        else:
            stats["forecasts_failed"] += 1
            logger.warning(f"  ✗ {game.away_team} @ {game.home_team}: Failed to fetch")

    return stats


def print_watchlist_report(stats: dict) -> None:
    """Print formatted watchlist report."""
    print("\n" + "=" * 80)
    print("🏈 THURSDAY WEATHER WATCHLIST")
    print("=" * 80)
    print(f"Captured: {stats['capture_time']}")
    print(f"Week: {stats['year']} Week {stats['week']}")
    print(f"Games: {stats['outdoor_games']} outdoor, {stats['indoor_games']} indoor")
    print(f"Forecasts: {stats['forecasts_captured']} captured, {stats['forecasts_failed']} failed")
    print("=" * 80)

    if not stats["watchlist"]:
        print("\n✅ No weather concerns this week. All games have normal conditions.")
    else:
        print(f"\n🚨 {len(stats['watchlist'])} GAMES WITH WEATHER CONCERNS:\n")

        # Sort by total adjustment (most negative first)
        sorted_watchlist = sorted(
            stats["watchlist"],
            key=lambda x: x["total_adjustment"]
        )

        for entry in sorted_watchlist:
            print(f"  {entry['matchup']}")
            print(f"    Venue: {entry['venue']}")
            print(f"    Wind: {entry['wind_speed']:.0f} mph (gust: {entry.get('wind_gust') or 'N/A'})")
            print(f"    Temp: {entry['temperature']:.0f}°F")

            # Show Vegas total and weather-adjusted total
            vegas_total = entry.get('vegas_total')
            adjusted_total = entry.get('adjusted_total')
            if vegas_total is not None and adjusted_total is not None:
                print(f"    📊 Vegas Total: {vegas_total:.1f} → Weather-Adjusted: {adjusted_total:.1f}")
            elif vegas_total is not None:
                print(f"    📊 Vegas Total: {vegas_total:.1f}")
            else:
                print(f"    📊 Vegas Total: N/A")

            print(f"    📉 UNDER ADJUSTMENT: {entry['total_adjustment']:+.1f} pts")
            print(f"       Wind: {entry['wind_adjustment']:+.1f}, Temp: {entry['temp_adjustment']:+.1f}, Precip: {entry['precip_adjustment']:+.1f}")
            print(f"    Confidence: {entry['confidence']:.0%} ({entry['hours_until_game']}h until game)")
            print()

        print("-" * 80)
        print("💡 ACTION: Consider betting UNDER on these games BEFORE market adjusts.")
        print("   Re-run Saturday morning for confirmation with higher confidence forecast.")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Thursday weather forecast capture for totals edge"
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Season year (default: auto-detect)"
    )
    parser.add_argument(
        "--week", type=int, default=None,
        help="Week number (default: auto-detect)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show games but don't fetch forecasts"
    )
    args = parser.parse_args()

    # Auto-detect year/week if not specified
    if args.year is None or args.week is None:
        auto_year, auto_week = get_current_cfb_week()
        args.year = args.year or auto_year
        args.week = args.week or auto_week
        logger.info(f"Auto-detected: {args.year} Week {args.week}")

    # Initialize clients
    cfbd_client = CFBDClient()
    tomorrow_client = TomorrowIOClient(
        api_key=TOMORROW_IO_API_KEY,
        rate_limit_delay=3.0,  # 3 seconds between calls
    )

    # Run capture
    logger.info(f"Starting Thursday weather capture for {args.year} Week {args.week}")

    stats = capture_with_watchlist(
        cfbd_client,
        tomorrow_client,
        args.year,
        args.week,
        dry_run=args.dry_run,
    )

    # Print report
    print_watchlist_report(stats)

    # Return exit code based on success
    if stats["forecasts_failed"] > stats["forecasts_captured"]:
        logger.error("More failures than successes - check rate limits")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
