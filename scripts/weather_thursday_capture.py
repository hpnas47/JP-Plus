#!/usr/bin/env python3
"""Thursday weather forecast capture for totals betting edge.

This script runs the full Thursday "Setup" capture workflow:
1. Captures forecasts for outdoor games (rate-limited via --limit and --delay)
2. Identifies "Watchlist" games with high wind/weather concern
3. Saves results to database and logs watchlist to console

Rate Limiting:
    Free tier: 25 calls/hour, 500 calls/day
    Default: --limit 20 (stay under hourly cap), --delay 3.0s between calls
    For more games: run multiple times 1+ hour apart, or use --limit 0 carefully

Schedule this to run Thursday mornings before limits increase at books.

Environment:
    TOMORROW_IO_API_KEY: Tomorrow.io API key (required)
                         Get one at: https://www.tomorrow.io/

Usage:
    # Manual run (with env var set)
    export TOMORROW_IO_API_KEY="your_key_here"
    python3 scripts/weather_thursday_capture.py

    # Or pass API key directly
    python3 scripts/weather_thursday_capture.py --api-key your_key_here

    # With specific week (otherwise auto-detects)
    python3 scripts/weather_thursday_capture.py --week 14

    # Dry run (show what would be captured)
    python3 scripts/weather_thursday_capture.py --dry-run

Cron example (every Thursday at 6 AM):
    0 6 * * 4 cd /path/to/project && TOMORROW_IO_API_KEY=xxx python3 scripts/weather_thursday_capture.py
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.cfbd_client import CFBDClient
from src.api.tomorrow_io import TomorrowIOClient, VenueLocation, is_dome_venue
from src.adjustments.weather import WeatherAdjuster, WeatherConditions
from src.models.totals_model import TotalsModel
from scripts.backtest import fetch_season_data
from config.teams import normalize_team_name

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


class OffSeasonError(Exception):
    """Raised when called during off-season (no games scheduled)."""
    pass


def get_current_cfb_week(cfbd_client: CFBDClient) -> tuple[int, int]:
    """Auto-detect current CFB season year and week using CFBD calendar API.

    Finds the current or next upcoming week with scheduled games.
    Handles Week 0, regular season (1-15), and postseason (16+).

    Args:
        cfbd_client: CFBD API client

    Returns:
        Tuple of (year, week)

    Raises:
        OffSeasonError: If no games found within 14 days (off-season)
    """
    now = datetime.now()
    today = now.date()

    # Determine candidate years to check
    # Aug-Dec: current year season
    # Jan: previous year's postseason
    # Feb-Jul: off-season (will fail)
    if now.month >= 8:
        candidate_years = [now.year]
    elif now.month == 1:
        candidate_years = [now.year - 1, now.year]  # Check both for early Jan
    else:
        # Feb-Jul: true off-season
        raise OffSeasonError(
            f"Off-season detected ({now.strftime('%B %Y')}). "
            "CFB games run late August through early January. "
            "Use --year and --week to specify a historical period."
        )

    for year in candidate_years:
        try:
            calendar = cfbd_client.get_calendar(year)
            if not calendar:
                continue

            # Find current or next upcoming week
            best_week = None
            best_distance = float('inf')

            for week_info in calendar:
                # Parse week dates
                try:
                    start = datetime.strptime(
                        week_info.first_game_start[:10], "%Y-%m-%d"
                    ).date()
                    end = datetime.strptime(
                        week_info.last_game_start[:10], "%Y-%m-%d"
                    ).date()
                except (ValueError, TypeError, AttributeError):
                    continue

                # Check if we're currently in this week (with buffer for Sunday)
                end_with_buffer = end + timedelta(days=2)
                if start <= today <= end_with_buffer:
                    return year, week_info.week

                # Track nearest upcoming week (within 14 days)
                if start > today:
                    days_until = (start - today).days
                    if days_until <= 14 and days_until < best_distance:
                        best_distance = days_until
                        best_week = week_info.week

            # Return nearest upcoming week if found
            if best_week is not None:
                logger.info(f"Next CFB week in {best_distance} days")
                return year, best_week

        except Exception as e:
            logger.warning(f"Failed to fetch calendar for {year}: {e}")
            continue

    # No games found within window
    raise OffSeasonError(
        f"No CFB games found within 14 days of {today}. "
        "Use --year and --week to specify a target period."
    )


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
        scrimmage = plays[plays['is_scrimmage']].copy()

        # Normalize team names to match game.home_team/away_team from games endpoint
        scrimmage['offense_normalized'] = scrimmage['offense'].apply(normalize_team_name)

        # Calculate pass rate by team (using normalized names)
        team_stats = scrimmage.groupby('offense_normalized').agg(
            passes=('is_pass', 'sum'),
            total=('is_scrimmage', 'sum')
        ).reset_index()

        # Avoid division by zero
        team_stats['pass_rate'] = team_stats['passes'] / team_stats['total'].clip(lower=1)

        pass_rates = dict(zip(team_stats['offense_normalized'], team_stats['pass_rate']))
        logger.info(f"  Calculated pass rates for {len(pass_rates)} teams (normalized)")

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


def fetch_week_games(
    cfbd_client: CFBDClient,
    year: int,
    week: int,
) -> list:
    """Fetch all games for a week (regular + postseason if applicable).

    Args:
        cfbd_client: CFBD API client
        year: Season year
        week: Week number

    Returns:
        List of game objects
    """
    games = cfbd_client.get_games(year=year, week=week, season_type="regular")

    # Also check postseason if week > 15
    if week > 15:
        try:
            postseason = cfbd_client.get_games(year=year, week=week, season_type="postseason")
            games.extend(postseason)
            logger.info(f"Included {len(postseason)} postseason games")
        except Exception as e:
            logger.warning(f"Failed to fetch postseason games for {year} week {week}: {e}")

    return games


# Default thresholds for watchlist inclusion
DEFAULT_WATCHLIST_ADJ_THRESHOLD = 1.5  # Min |weather_adjustment| to include
DEFAULT_WATCHLIST_EDGE_THRESHOLD = 3.0  # Min |edge| to include (if edge available)


def capture_with_watchlist(
    cfbd_client: CFBDClient,
    tomorrow_client: TomorrowIOClient,
    year: int,
    week: int,
    dry_run: bool = False,
    limit: int = 0,
    games: list | None = None,
    watchlist_adj_threshold: float = DEFAULT_WATCHLIST_ADJ_THRESHOLD,
    watchlist_edge_threshold: float = DEFAULT_WATCHLIST_EDGE_THRESHOLD,
) -> dict:
    """Capture forecasts and build watchlist of weather-impacted games.

    Computes weather adjustments for ALL outdoor games, then builds watchlist
    based on configurable thresholds.

    Args:
        cfbd_client: CFBD API client
        tomorrow_client: Tomorrow.io client
        year: Season year
        week: Week number
        dry_run: If True, show games but don't fetch forecasts
        limit: Max forecasts to fetch (0 = no limit, recommended: 20 per hour)
        games: Optional pre-fetched games list (avoids duplicate API calls)
        watchlist_adj_threshold: Min |weather_adjustment| to include in watchlist
        watchlist_edge_threshold: Min |edge| to include (games meeting either threshold qualify)

    Returns:
        Dict with capture stats, all_adjustments, and watchlist
    """
    # Use provided games or fetch them
    if games is None:
        games = fetch_week_games(cfbd_client, year, week)

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
        "games_skipped_no_time": 0,
        "venue_not_found": 0,
        "all_adjustments": [],  # All outdoor games with computed adjustments
        "watchlist": [],  # Games exceeding thresholds
        "watchlist_adj_threshold": watchlist_adj_threshold,
        "watchlist_edge_threshold": watchlist_edge_threshold,
    }

    weather_adjuster = WeatherAdjuster()
    outdoor_games = []

    # Cache venue lookups - multiple games can share a venue (neutral sites, etc.)
    venue_cache: dict[int, object] = {}

    # First pass: identify outdoor games
    for game in games:
        venue_id = getattr(game, "venue_id", None)

        # Check cache first, then fetch if needed
        if venue_id is None:
            venue = None
        elif venue_id in venue_cache:
            venue = venue_cache[venue_id]
        else:
            venue = tomorrow_client.get_venue(venue_id)
            venue_cache[venue_id] = venue

        if venue is None:
            stats["venue_not_found"] += 1
            continue

        # Check if dome using both database flag and fallback list
        # (database may have stale data for some venues)
        if is_dome_venue(venue.name, venue.dome):
            stats["indoor_games"] += 1
            continue

        stats["outdoor_games"] += 1
        outdoor_games.append((game, venue))

    logger.info(f"Outdoor games to capture: {len(outdoor_games)}")
    if limit > 0:
        logger.info(f"Rate limit: capturing max {limit} forecasts (use --limit 0 for all)")

    # Log current hourly API usage
    hourly_calls = tomorrow_client.get_hourly_call_count()
    hourly_remaining = tomorrow_client.FREE_TIER_HOURLY_LIMIT - hourly_calls
    logger.info(
        f"Tomorrow.io hourly quota: {hourly_calls}/{tomorrow_client.FREE_TIER_HOURLY_LIMIT} "
        f"used, {hourly_remaining} remaining"
    )

    if dry_run:
        for game, venue in outdoor_games:
            logger.info(f"  [DRY RUN] {game.away_team} @ {game.home_team} ({venue.name})")
        return stats

    # Capture forecasts (respecting rate limits)
    for i, (game, venue) in enumerate(outdoor_games):
        # Check limit before each API call
        if limit > 0 and stats["forecasts_captured"] >= limit:
            remaining = len(outdoor_games) - i
            logger.info(f"Reached limit of {limit} forecasts. {remaining} games remaining.")
            logger.info("Run again in 1 hour to capture remaining games.")
            break

        matchup = f"{game.away_team} @ {game.home_team}"

        # Parse game time - must be valid and timezone-aware (UTC)
        start_date = getattr(game, "start_date", None)
        game_time: datetime | None = None

        if start_date:
            if isinstance(start_date, str):
                try:
                    # Parse ISO format, convert "Z" suffix to proper UTC offset
                    game_time = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                    # Ensure timezone-aware (UTC)
                    if game_time.tzinfo is None:
                        game_time = game_time.replace(tzinfo=timezone.utc)
                except ValueError as e:
                    logger.warning(
                        f"  ⏭ {matchup} (game_id={game.id}): "
                        f"Invalid start_date format '{start_date}': {e}"
                    )
                    stats["games_skipped_no_time"] += 1
                    continue
            else:
                # Already a datetime object
                game_time = start_date
                if game_time.tzinfo is None:
                    # Assume UTC if naive (CFBD times are UTC)
                    game_time = game_time.replace(tzinfo=timezone.utc)
        else:
            logger.warning(
                f"  ⏭ {matchup} (game_id={game.id}): "
                f"Missing start_date, skipping forecast capture"
            )
            stats["games_skipped_no_time"] += 1
            continue

        try:
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

                # Calculate adjustment for ALL outdoor games (not just flagged ones)
                conditions = tomorrow_client.forecast_to_weather_conditions(forecast, game.id)

                # Calculate combined pass rate for pass-rate multiplier
                # Default to 0.5 (neutral) if pass rate data unavailable
                # Normalize team names to match keys in team_pass_rates dict
                home_normalized = normalize_team_name(game.home_team)
                away_normalized = normalize_team_name(game.away_team)
                home_pass_rate = team_pass_rates.get(home_normalized)
                away_pass_rate = team_pass_rates.get(away_normalized)
                pass_rate_available = (
                    home_pass_rate is not None and away_pass_rate is not None
                )
                if pass_rate_available:
                    combined_pass_rate = (home_pass_rate + away_pass_rate) / 2
                else:
                    combined_pass_rate = 0.5  # Neutral default
                    missing_team = (
                        game.home_team if home_pass_rate is None else game.away_team
                    )
                    logger.debug(
                        f"  {matchup}: Pass-rate scaling inactive "
                        f"(missing data for {missing_team})"
                    )

                adjustment = weather_adjuster.calculate_adjustment(
                    conditions,
                    combined_pass_rate=combined_pass_rate,
                    confidence_factor=forecast.confidence_factor,
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

                game_entry = {
                    "game_id": game.id,
                    "matchup": matchup,
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
                    "pass_rate_available": pass_rate_available,
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
                    "high_variance": adjustment.high_variance,
                }

                # Store all adjustments
                stats["all_adjustments"].append(game_entry)

                # Check watchlist thresholds:
                # Include if |weather_adjustment| >= adj_threshold OR |edge| >= edge_threshold
                abs_adj = abs(adjustment.total_adjustment)
                abs_edge = abs(edge) if edge is not None else 0.0
                meets_adj_threshold = abs_adj >= watchlist_adj_threshold
                meets_edge_threshold = edge is not None and abs_edge >= watchlist_edge_threshold

                if meets_adj_threshold or meets_edge_threshold:
                    stats["watchlist"].append(game_entry)
                    edge_str = f"Edge: {edge:+.1f}" if edge is not None else "Edge: N/A"
                    logger.info(
                        f"  🚨 WATCHLIST: {matchup} | "
                        f"Wind: {forecast.wind_speed:.0f}/{forecast.wind_gust or 0:.0f} mph, "
                        f"Temp: {forecast.temperature:.0f}°F | "
                        f"Weather: {adjustment.total_adjustment:+.1f} pts | {edge_str}"
                    )
                else:
                    # Log non-watchlist games at debug level
                    logger.debug(
                        f"  ✓ {matchup}: "
                        f"{forecast.temperature:.0f}°F, {forecast.wind_speed:.0f} mph wind, "
                        f"adj={adjustment.total_adjustment:+.1f} (below threshold)"
                    )
            else:
                stats["forecasts_failed"] += 1
                logger.warning(f"  ✗ {matchup}: Forecast API returned no data")

        except Exception as e:
            stats["forecasts_failed"] += 1
            logger.error(
                f"  ✗ {matchup} (game_id={game.id}, venue={venue.name}): "
                f"Exception during processing: {type(e).__name__}: {e}"
            )
            continue

    # Log final hourly API usage
    final_hourly_calls = tomorrow_client.get_hourly_call_count()
    logger.info(
        f"Tomorrow.io hourly quota after capture: "
        f"{final_hourly_calls}/{tomorrow_client.FREE_TIER_HOURLY_LIMIT} used"
    )

    return stats


def print_watchlist_report(stats: dict) -> None:
    """Print formatted watchlist report."""
    print("\n" + "=" * 80)
    print("🏈 THURSDAY WEATHER WATCHLIST")
    print("=" * 80)
    print(f"Captured: {stats['capture_time']}")
    print(f"Week: {stats['year']} Week {stats['week']}")
    print(f"Games: {stats['outdoor_games']} outdoor, {stats['indoor_games']} indoor")
    skipped = stats.get('games_skipped_no_time', 0)
    skipped_str = f", {skipped} skipped (no time)" if skipped > 0 else ""
    print(f"Forecasts: {stats['forecasts_captured']} captured, {stats['forecasts_failed']} failed{skipped_str}")

    # Show thresholds and adjustment stats
    adj_threshold = stats.get('watchlist_adj_threshold', DEFAULT_WATCHLIST_ADJ_THRESHOLD)
    edge_threshold = stats.get('watchlist_edge_threshold', DEFAULT_WATCHLIST_EDGE_THRESHOLD)
    all_adj = stats.get('all_adjustments', [])
    print(f"Thresholds: |adj| >= {adj_threshold} OR |edge| >= {edge_threshold}")
    print(f"Adjustments computed: {len(all_adj)} games, {len(stats['watchlist'])} meet threshold")
    print("=" * 80)

    if not stats["watchlist"]:
        print("\n✅ No games exceed watchlist thresholds this week.")
    else:
        print(f"\n🚨 {len(stats['watchlist'])} GAMES EXCEED THRESHOLDS:\n")

        # Sort by edge (most negative = strongest UNDER signal first)
        # If edge is None, use weather adjustment as fallback
        # Sort by edge (most negative first), with edge=None entries last
        # Tuple key: (has_edge_flag, sort_value) where flag=0 for real edge, 1 for None
        sorted_watchlist = sorted(
            stats["watchlist"],
            key=lambda x: (
                (0, x["edge"]) if x.get("edge") is not None
                else (1, x.get("weather_adjustment", 0))
            )
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
            pass_rate_available = entry.get('pass_rate_available', True)
            if entry['wind_adjustment'] < 0:
                if not pass_rate_available:
                    print(f"       📋 Pass Rate: N/A (using neutral default, scaling inactive)")
                elif combined_pass_rate >= 0.55:
                    print(f"       📋 Pass Rate: {combined_pass_rate:.0%} (Pass-heavy, wind hurts more)")
                elif combined_pass_rate <= 0.45:
                    print(f"       📋 Pass Rate: {combined_pass_rate:.0%} (Run-heavy, wind hurts less)")
                else:
                    print(f"       📋 Pass Rate: {combined_pass_rate:.0%} (Balanced matchup)")

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

    # Show thresholds and adjustment stats
    adj_threshold = stats.get('watchlist_adj_threshold', DEFAULT_WATCHLIST_ADJ_THRESHOLD)
    edge_threshold = stats.get('watchlist_edge_threshold', DEFAULT_WATCHLIST_EDGE_THRESHOLD)
    all_adj = stats.get('all_adjustments', [])
    print(f"Thresholds: |adj| >= {adj_threshold} OR |edge| >= {edge_threshold}")
    print(f"Adjustments computed: {len(all_adj)} games, {len(stats['watchlist'])} meet threshold")
    print("=" * 80)

    if not stats["watchlist"]:
        print("\n✅ No games exceed watchlist thresholds this week.")
        print("=" * 80 + "\n")
        return

    print(f"\n📊 {len(stats['watchlist'])} GAMES WITH WEATHER CONCERNS:\n")

    # Sort by edge (most negative first)
    # Sort by edge (most negative first), with edge=None entries last
    # Tuple key: (has_edge_flag, sort_value) where flag=0 for real edge, 1 for None
    sorted_watchlist = sorted(
        stats["watchlist"],
        key=lambda x: (
            (0, x["edge"]) if x.get("edge") is not None
            else (1, x.get("weather_adjustment", 0))
        )
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
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Max forecasts to fetch per run (default: 20, free tier safe). Use 0 for no limit."
    )
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="Seconds between API calls (default: 3.0)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Tomorrow.io API key (or set TOMORROW_IO_API_KEY env var)"
    )
    parser.add_argument(
        "--adj-threshold", type=float, default=DEFAULT_WATCHLIST_ADJ_THRESHOLD,
        help=f"Min |weather_adjustment| for watchlist (default: {DEFAULT_WATCHLIST_ADJ_THRESHOLD})"
    )
    parser.add_argument(
        "--edge-threshold", type=float, default=DEFAULT_WATCHLIST_EDGE_THRESHOLD,
        help=f"Min |edge| for watchlist (default: {DEFAULT_WATCHLIST_EDGE_THRESHOLD})"
    )
    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("TOMORROW_IO_API_KEY")
    if not api_key:
        logger.error("API key required. Set TOMORROW_IO_API_KEY env var or use --api-key")
        sys.exit(1)

    # Initialize clients
    cfbd_client = CFBDClient()
    tomorrow_client = TomorrowIOClient(
        api_key=api_key,
        rate_limit_delay=args.delay,
    )

    # Auto-detect year/week if not specified (requires CFBD client)
    if args.year is None or args.week is None:
        try:
            auto_year, auto_week = get_current_cfb_week(cfbd_client)
            args.year = args.year or auto_year
            args.week = args.week or auto_week
            logger.info(f"Auto-detected: {args.year} Week {args.week}")
        except OffSeasonError as e:
            logger.error(str(e))
            sys.exit(1)

    # Run capture
    mode = "Saturday confirmation" if args.saturday else "Thursday"
    logger.info(f"Starting {mode} weather capture for {args.year} Week {args.week}")

    # Fetch games ONCE to ensure consistent game_ids across all operations
    games = fetch_week_games(cfbd_client, args.year, args.week)
    logger.info(f"Found {len(games)} games for {args.year} week {args.week}")

    # If Saturday mode, load Thursday forecasts first for comparison
    # Use EARLIEST forecast (not latest) to get the original Thursday morning capture
    # This prevents Thursday afternoon re-runs from overwriting the comparison baseline
    thursday_forecasts = {}
    if args.saturday and not args.dry_run:
        logger.info("Loading Thursday forecasts for comparison...")
        for game in games:
            # Get all forecasts and take the earliest (Thursday morning)
            all_forecasts = tomorrow_client.get_all_forecasts(game.id)
            if all_forecasts:
                # First forecast is oldest (Thursday morning capture)
                thursday_forecasts[game.id] = all_forecasts[0]
        logger.info(f"  Loaded {len(thursday_forecasts)} Thursday forecasts (earliest capture per game)")

    stats = capture_with_watchlist(
        cfbd_client,
        tomorrow_client,
        args.year,
        args.week,
        dry_run=args.dry_run,
        limit=args.limit,
        games=games,
        watchlist_adj_threshold=args.adj_threshold,
        watchlist_edge_threshold=args.edge_threshold,
    )

    # Save watchlist JSON for Discord bot / show_weather_alert.py
    import json
    weather_dir = Path(__file__).parent.parent / "data" / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)
    cache_file = weather_dir / f"watchlist_{args.year}_w{args.week}.json"
    with open(cache_file, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Saved watchlist to {cache_file}")

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
