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
from src.models.totals_model import TotalsModel
from scripts.backtest import fetch_season_data

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


def calculate_team_pass_rates(
    cfbd_client: CFBDClient,
    year: int,
    week: int,
) -> dict[str, float]:
    """Calculate pass rate for each team from play-by-play data.

    The "Passing Team" Multiplier: Wind hurts pass-heavy teams more.
    - Air Raid (Ole Miss): 60%+ pass rate → bigger wind penalty
    - Triple Option (Army): <40% pass rate → smaller wind penalty

    Args:
        cfbd_client: CFBD API client
        year: Season year
        week: Current week (use data from weeks < week)

    Returns:
        Dict mapping team name to pass rate (0-1)
    """
    logger.info(f"Calculating team pass rates for {year} weeks 1-{week-1}...")

    try:
        # Get play-by-play data
        from scripts.backtest import fetch_season_plays
        plays_df = fetch_season_plays(cfbd_client, year)
        plays = plays_df.to_pandas()

        # Filter to weeks < current week
        plays = plays[plays['week'] < week]

        # Only count scrimmage plays (pass/rush)
        plays = plays[plays['play_type'].notna()].copy()
        plays['play_type_lower'] = plays['play_type'].str.lower()

        # Identify pass and rush plays
        plays['is_pass'] = plays['play_type_lower'].str.contains('pass|sack|scramble', na=False)
        plays['is_rush'] = plays['play_type_lower'].str.contains('rush|run|kneel', na=False)
        plays['is_scrimmage'] = plays['is_pass'] | plays['is_rush']

        # Filter to scrimmage plays only
        scrimmage = plays[plays['is_scrimmage']]

        # Calculate pass rate by team
        team_stats = scrimmage.groupby('offense').agg(
            passes=('is_pass', 'sum'),
            total=('is_scrimmage', 'sum')
        ).reset_index()

        # Avoid division by zero
        team_stats['pass_rate'] = team_stats['passes'] / team_stats['total'].clip(lower=1)

        pass_rates = dict(zip(team_stats['offense'], team_stats['pass_rate']))
        logger.info(f"  Calculated pass rates for {len(pass_rates)} teams")

        # Log some examples
        if pass_rates:
            top_3 = sorted(pass_rates.items(), key=lambda x: x[1], reverse=True)[:3]
            bottom_3 = sorted(pass_rates.items(), key=lambda x: x[1])[:3]
            logger.info(f"  Highest pass rates: {top_3}")
            logger.info(f"  Lowest pass rates: {bottom_3}")

        return pass_rates

    except Exception as e:
        logger.warning(f"Could not calculate pass rates: {e}")
        return {}


def train_totals_model(
    cfbd_client: CFBDClient,
    year: int,
    week: int,
) -> tuple[TotalsModel, dict[str, float]]:
    """Train TotalsModel and calculate team pass rates.

    Args:
        cfbd_client: CFBD API client
        year: Season year
        week: Current week (train on weeks < week)

    Returns:
        Tuple of (Trained TotalsModel, dict of team pass rates)
    """
    logger.info(f"Training JP+ TotalsModel on {year} weeks 1-{week-1}...")

    # Fetch season data
    games_df, _ = fetch_season_data(cfbd_client, year)
    games = games_df.to_pandas()

    # Get FBS teams
    fbs_teams = cfbd_client.get_fbs_teams(year=year)
    fbs_set = {t.school for t in fbs_teams if t.school}

    # Filter to FBS vs FBS with scores
    games = games[
        games['home_team'].isin(fbs_set) &
        games['away_team'].isin(fbs_set) &
        games['home_points'].notna() &
        games['away_points'].notna()
    ].copy()

    # Add year column
    games['year'] = year

    # Train on weeks < current week (walk-forward)
    train_games = games[games['week'] < week]

    if len(train_games) < 20:
        logger.warning(f"Only {len(train_games)} training games available")

    # Create and train model
    model = TotalsModel(ridge_alpha=10.0, decay_factor=1.0)
    model.set_team_universe(fbs_set)
    model.train(train_games, fbs_set, max_week=week - 1)

    if model._trained:
        logger.info(f"  Trained on {len(train_games)} games, baseline={model.baseline:.1f}")
    else:
        logger.warning("  Model training failed")

    # Calculate pass rates
    pass_rates = calculate_team_pass_rates(cfbd_client, year, week)

    return model, pass_rates


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

    # Train JP+ TotalsModel and calculate pass rates
    totals_model = None
    team_pass_rates = {}
    if week > 1 and not dry_run:
        try:
            totals_model, team_pass_rates = train_totals_model(cfbd_client, year, week)
        except Exception as e:
            logger.warning(f"Could not train TotalsModel: {e}")

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
                # Calculate the adjustment with pass rate scaling
                conditions = tomorrow_client.forecast_to_weather_conditions(forecast, game.id)

                # Calculate combined pass rate for pass-rate multiplier
                home_pass_rate = team_pass_rates.get(game.home_team)
                away_pass_rate = team_pass_rates.get(game.away_team)
                combined_pass_rate = None
                if home_pass_rate is not None and away_pass_rate is not None:
                    combined_pass_rate = (home_pass_rate + away_pass_rate) / 2

                adjustment = weather_adjuster.calculate_adjustment(
                    conditions,
                    combined_pass_rate=combined_pass_rate,
                )

                # Get JP+ predicted total and Vegas total
                jp_total = None
                jp_weather_adjusted = None
                vegas_total = betting_lines.get(game.id)
                edge = None

                if totals_model and totals_model._trained:
                    pred = totals_model.predict_total(
                        game.home_team,
                        game.away_team,
                        year=year,
                    )
                    if pred:
                        jp_total = pred.predicted_total
                        jp_weather_adjusted = jp_total + adjustment.total_adjustment

                        # Edge: JP+ weather-adjusted vs Vegas
                        # Negative edge = JP+ says UNDER (we want this for weather games)
                        if vegas_total is not None:
                            edge = jp_weather_adjusted - vegas_total

                watchlist_entry = {
                    "game_id": game.id,
                    "matchup": f"{game.away_team} @ {game.home_team}",
                    "home_team": game.home_team,
                    "away_team": game.away_team,
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
                    "combined_pass_rate": combined_pass_rate,
                    "jp_total": jp_total,
                    "jp_weather_adjusted": jp_weather_adjusted,
                    "vegas_total": vegas_total,
                    "edge": edge,
                    "weather_adjustment": adjustment.total_adjustment,
                    "wind_adjustment": adjustment.wind_adjustment,
                    "temp_adjustment": adjustment.temperature_adjustment,
                    "precip_adjustment": adjustment.precipitation_adjustment,
                    "confidence": forecast.confidence_factor,
                    "hours_until_game": forecast.hours_until_game,
                }
                stats["watchlist"].append(watchlist_entry)

                edge_str = f"Edge: {edge:+.1f}" if edge is not None else "Edge: N/A"
                logger.info(
                    f"  🚨 WATCHLIST: {game.away_team} @ {game.home_team} | "
                    f"Wind: {forecast.wind_speed:.0f}/{forecast.wind_gust or 0:.0f} mph, "
                    f"Temp: {forecast.temperature:.0f}°F | "
                    f"Weather: {adjustment.total_adjustment:+.1f} pts | {edge_str}"
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

        # Sort by edge (most negative = strongest UNDER signal first)
        # If edge is None, use weather adjustment as fallback
        sorted_watchlist = sorted(
            stats["watchlist"],
            key=lambda x: x.get("edge") if x.get("edge") is not None else x.get("weather_adjustment", 0)
        )

        for entry in sorted_watchlist:
            print(f"  {entry['matchup']}")
            print(f"    Venue: {entry['venue']}")
            print(f"    Wind: {entry['wind_speed']:.0f} mph (gust: {entry.get('wind_gust') or 'N/A'})")
            print(f"    Temp: {entry['temperature']:.0f}°F")

            # Show JP+ prediction and weather adjustment
            jp_total = entry.get('jp_total')
            jp_weather = entry.get('jp_weather_adjusted')
            vegas_total = entry.get('vegas_total')
            edge = entry.get('edge')

            if jp_total is not None:
                print(f"    📊 JP+ Total: {jp_total:.1f} → Weather-Adjusted: {jp_weather:.1f}")
            else:
                print(f"    📊 JP+ Total: N/A (insufficient training data)")

            if vegas_total is not None:
                print(f"    🎰 Vegas Total: {vegas_total:.1f}")
            else:
                print(f"    🎰 Vegas Total: N/A")

            # Show the edge (JP+ weather-adjusted vs Vegas)
            if edge is not None:
                if edge < -3:
                    signal = "🔥 STRONG UNDER"
                elif edge < 0:
                    signal = "📉 LEAN UNDER"
                elif edge > 3:
                    signal = "⚠️ JP+ HIGHER THAN VEGAS"
                else:
                    signal = "➖ NEUTRAL"
                print(f"    💰 Edge: {edge:+.1f} pts ({signal})")

            print(f"    🌧️ Weather Adjustment: {entry['weather_adjustment']:+.1f} pts")
            print(f"       Wind: {entry['wind_adjustment']:+.1f}, Temp: {entry['temp_adjustment']:+.1f}, Precip: {entry['precip_adjustment']:+.1f}")

            # Show pass rate context if significant wind adjustment
            combined_pass_rate = entry.get('combined_pass_rate')
            if combined_pass_rate is not None and entry['wind_adjustment'] < 0:
                if combined_pass_rate >= 0.55:
                    style = "Pass-heavy matchup (wind hurts more)"
                elif combined_pass_rate <= 0.45:
                    style = "Run-heavy matchup (wind hurts less)"
                else:
                    style = "Balanced matchup"
                print(f"       📋 Pass Rate: {combined_pass_rate:.0%} ({style})")

            print(f"    Confidence: {entry['confidence']:.0%} ({entry['hours_until_game']}h until game)")
            print()

        print("-" * 80)
        print("💡 ACTION: Consider betting UNDER on these games BEFORE market adjusts.")
        print("   Re-run Saturday morning for confirmation with higher confidence forecast.")

    print("=" * 80 + "\n")


def print_saturday_confirmation_report(stats: dict, thursday_forecasts: dict) -> None:
    """Print Saturday confirmation report comparing to Thursday forecasts.

    Args:
        stats: Current capture stats with watchlist
        thursday_forecasts: Dict mapping game_id to Thursday's WeatherForecast
    """
    print("\n" + "=" * 80)
    print("🏈 SATURDAY CONFIRMATION REPORT (Final Model Input)")
    print("=" * 80)
    print(f"Captured: {stats['capture_time']}")
    print(f"Week: {stats['year']} Week {stats['week']}")
    print(f"Games: {stats['outdoor_games']} outdoor, {stats['indoor_games']} indoor")
    print(f"Forecasts: {stats['forecasts_captured']} captured")
    print("=" * 80)

    if not stats["watchlist"]:
        print("\n✅ No weather concerns this week. All games have normal conditions.")
        print("=" * 80 + "\n")
        return

    print(f"\n📊 {len(stats['watchlist'])} GAMES WITH WEATHER CONCERNS:\n")

    # Sort by edge (most negative first)
    sorted_watchlist = sorted(
        stats["watchlist"],
        key=lambda x: x.get("edge") if x.get("edge") is not None else x.get("weather_adjustment", 0)
    )

    for entry in sorted_watchlist:
        game_id = entry["game_id"]
        thursday = thursday_forecasts.get(game_id)

        print(f"  {entry['matchup']}")
        print(f"    Venue: {entry['venue']}")

        # Current conditions (Saturday)
        print(f"    📍 SATURDAY (Final):")
        print(f"       Wind: {entry['wind_speed']:.0f} mph (gust: {entry.get('wind_gust') or 'N/A'})")
        print(f"       Temp: {entry['temperature']:.0f}°F")

        # Compare to Thursday if available
        if thursday:
            wind_change = entry['wind_speed'] - thursday.wind_speed
            temp_change = entry['temperature'] - thursday.temperature

            print(f"    📅 THURSDAY (Previous):")
            print(f"       Wind: {thursday.wind_speed:.0f} mph → {entry['wind_speed']:.0f} mph ({wind_change:+.0f})")
            print(f"       Temp: {thursday.temperature:.0f}°F → {entry['temperature']:.0f}°F ({temp_change:+.0f})")

            # Flag significant changes
            if abs(wind_change) >= 5 or abs(temp_change) >= 10:
                print(f"    ⚠️  FORECAST CHANGED SIGNIFICANTLY")
        else:
            print(f"    📅 No Thursday forecast to compare")

        # Show JP+ prediction and edge
        jp_total = entry.get('jp_total')
        jp_weather = entry.get('jp_weather_adjusted')
        vegas_total = entry.get('vegas_total')
        edge = entry.get('edge')

        if jp_total is not None:
            print(f"    📊 JP+ Total: {jp_total:.1f} → Weather-Adjusted: {jp_weather:.1f}")
        if vegas_total is not None:
            print(f"    🎰 Vegas Total: {vegas_total:.1f}")

        if edge is not None:
            if edge < -3:
                signal = "🔥 STRONG UNDER — BET NOW"
            elif edge < 0:
                signal = "📉 LEAN UNDER"
            elif edge > 3:
                signal = "⚠️ JP+ HIGHER (skip)"
            else:
                signal = "➖ NEUTRAL (skip)"
            print(f"    💰 Edge: {edge:+.1f} pts ({signal})")

        print(f"    🌧️ Weather Adjustment: {entry['weather_adjustment']:+.1f} pts")
        print(f"    ✅ Confidence: {entry['confidence']:.0%} (FINAL - {entry['hours_until_game']}h until game)")
        print()

    print("-" * 80)
    print("🎯 SATURDAY CONFIRMATION COMPLETE")
    print("   Games with 🔥 STRONG UNDER are your highest-conviction bets.")
    print("   Forecasts are now 6-12h out — this is final model input.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Weather forecast capture for totals betting edge"
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
    parser.add_argument(
        "--saturday", action="store_true",
        help="Saturday confirmation run (compares to Thursday forecasts)"
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
    mode = "Saturday confirmation" if args.saturday else "Thursday"
    logger.info(f"Starting {mode} weather capture for {args.year} Week {args.week}")

    # If Saturday mode, load Thursday forecasts first for comparison
    thursday_forecasts = {}
    if args.saturday and not args.dry_run:
        logger.info("Loading Thursday forecasts for comparison...")
        # Get all games to find their IDs
        games = cfbd_client.get_games(year=args.year, week=args.week, season_type="regular")
        for game in games:
            # Load Thursday forecast (max_age_hours=96 to catch 4-day old forecasts)
            forecast = tomorrow_client.get_saved_forecast(game.id, max_age_hours=96)
            if forecast:
                thursday_forecasts[game.id] = forecast
        logger.info(f"  Loaded {len(thursday_forecasts)} Thursday forecasts")

    stats = capture_with_watchlist(
        cfbd_client,
        tomorrow_client,
        args.year,
        args.week,
        dry_run=args.dry_run,
    )

    # Print appropriate report
    if args.saturday:
        print_saturday_confirmation_report(stats, thursday_forecasts)
    else:
        print_watchlist_report(stats)

    # Return exit code based on success
    if stats["forecasts_failed"] > stats["forecasts_captured"]:
        logger.error("More failures than successes - check rate limits")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
