#!/usr/bin/env python3
"""Populate week-level cache for delta loading.

This script fetches and caches data for individual weeks, enabling
future runs to only fetch the current week from the API.

Usage:
    # Cache all weeks for current season
    python3 scripts/populate_week_cache.py

    # Cache specific year
    python3 scripts/populate_week_cache.py --year 2024

    # Cache specific week range
    python3 scripts/populate_week_cache.py --year 2024 --weeks 1 5

    # Force refresh (bypass existing cache)
    python3 scripts/populate_week_cache.py --force-refresh
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.cfbd_client import CFBDClient
from src.data.week_cache import WeekDataCache
from config.play_types import TURNOVER_PLAY_TYPES, SCRIMMAGE_PLAY_TYPES
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def populate_week(
    client: CFBDClient,
    week_cache: WeekDataCache,
    year: int,
    week: int,
    force_refresh: bool = False,
):
    """Populate cache for a single week.

    Args:
        client: CFBD API client
        week_cache: Week cache instance
        year: Season year
        week: Week number
        force_refresh: If True, fetch even if cached
    """
    # Check if already cached
    if not force_refresh and week_cache.has_cached_week(year, week, "games"):
        logger.info(f"✓ {year} week {week} already cached, skipping")
        return

    logger.info(f"Fetching {year} week {week}...")

    # Fetch games
    games = []
    try:
        week_games = client.get_games(year, week)
        for game in week_games:
            if game.home_points is None:
                continue
            games.append({
                "id": game.id,
                "year": year,
                "week": week,
                "start_date": game.start_date,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "home_points": game.home_points,
                "away_points": game.away_points,
                "neutral_site": game.neutral_site or False,
            })
    except Exception as e:
        logger.warning(f"Failed to fetch games: {e}")
        return

    if not games:
        logger.warning(f"No completed games found for {year} week {week}")
        return

    games_df = pl.DataFrame(games)

    # Fetch betting lines
    betting = []
    preferred_providers = ["DraftKings", "ESPN Bet", "Bovada"]

    try:
        lines = client.get_betting_lines(year, week=week, season_type="regular")
        for game_lines in lines:
            if not game_lines.lines:
                continue

            selected_line = None
            for provider in preferred_providers:
                for line in game_lines.lines:
                    if line.provider and line.provider == provider:
                        selected_line = line
                        break
                if selected_line:
                    break

            if selected_line is None:
                selected_line = game_lines.lines[0] if game_lines.lines else None

            if selected_line and selected_line.spread is not None:
                betting.append({
                    "game_id": game_lines.id,
                    "home_team": game_lines.home_team,
                    "away_team": game_lines.away_team,
                    "spread_close": selected_line.spread,
                    "spread_open": selected_line.spread_open if selected_line.spread_open is not None else selected_line.spread,
                    "over_under": selected_line.over_under,
                    "provider": selected_line.provider,
                })
    except Exception as e:
        logger.warning(f"Failed to fetch betting lines: {e}")

    betting_df = pl.DataFrame(betting)

    # Fetch plays
    efficiency_plays = []
    turnover_plays = []
    st_plays = []

    try:
        plays = client.get_plays(year, week)
        for play in plays:
            play_type = play.play_type or ""

            # Turnovers
            if play_type in TURNOVER_PLAY_TYPES:
                turnover_plays.append({
                    "week": week,
                    "game_id": play.game_id,
                    "offense": play.offense,
                    "defense": play.defense,
                    "play_type": play_type,
                })

            # Special teams
            if any(st in play_type for st in ["Field Goal", "Punt", "Kickoff"]):
                st_plays.append({
                    "week": week,
                    "game_id": play.game_id,
                    "offense": play.offense,
                    "defense": play.defense,
                    "play_type": play_type,
                    "play_text": play.play_text,
                })

            # Efficiency plays
            if (play.ppa is not None and
                play.down is not None and
                play_type in SCRIMMAGE_PLAY_TYPES and
                play.distance is not None and play.distance >= 0):
                efficiency_plays.append({
                    "week": week,
                    "game_id": play.game_id,
                    "down": play.down,
                    "distance": play.distance,
                    "yards_gained": play.yards_gained or 0,
                    "play_type": play_type,
                    "play_text": play.play_text,
                    "offense": play.offense,
                    "defense": play.defense,
                    "period": play.period,
                    "ppa": play.ppa,
                    "yards_to_goal": play.yards_to_goal,
                    "offense_score": play.offense_score or 0,
                    "defense_score": play.defense_score or 0,
                    "home_team": play.home,
                })
    except Exception as e:
        logger.warning(f"Failed to fetch plays: {e}")

    efficiency_plays_df = pl.DataFrame(efficiency_plays)
    turnover_plays_df = pl.DataFrame(turnover_plays)
    st_plays_df = pl.DataFrame(st_plays)

    # Validate home_team in efficiency plays
    if len(efficiency_plays_df) > 0 and len(games_df) > 0:
        game_home = games_df.select([
            pl.col("id").alias("game_id"),
            pl.col("home_team").alias("validated_home_team"),
        ])
        efficiency_plays_df = efficiency_plays_df.join(
            game_home, on="game_id", how="left"
        )
        efficiency_plays_df = efficiency_plays_df.with_columns(
            pl.col("validated_home_team").alias("home_team")
        ).drop("validated_home_team")

    # Save to cache
    week_cache.save_week(year, week, "games", games_df)
    week_cache.save_week(year, week, "betting", betting_df)
    week_cache.save_week(year, week, "efficiency_plays", efficiency_plays_df)
    week_cache.save_week(year, week, "turnovers", turnover_plays_df)
    week_cache.save_week(year, week, "st_plays", st_plays_df)

    logger.info(
        f"✓ Cached {year} week {week}: {len(games_df)} games, "
        f"{len(efficiency_plays_df)} efficiency plays"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Populate week-level cache for delta loading"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Season year (default: current year)",
    )
    parser.add_argument(
        "--weeks",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Week range to cache (e.g., --weeks 1 5). Default: all regular season weeks (1-15)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh even if already cached",
    )

    args = parser.parse_args()

    # Determine year
    from datetime import datetime
    year = args.year or datetime.now().year

    # Determine week range
    if args.weeks:
        start_week, end_week = args.weeks
    else:
        start_week, end_week = 1, 15

    logger.info(
        f"Populating week cache for {year}, weeks {start_week}-{end_week}"
    )

    client = CFBDClient()
    week_cache = WeekDataCache()

    # Populate each week
    for week in range(start_week, end_week + 1):
        try:
            populate_week(client, week_cache, year, week, args.force_refresh)
        except Exception as e:
            logger.error(f"Failed to populate week {week}: {e}")
            continue

    # Show cache stats
    cached_weeks = week_cache.get_cached_weeks(year, "games")
    logger.info(
        f"\nCache complete: {len(cached_weeks)} weeks cached for {year}"
    )
    logger.info(f"Cached weeks: {cached_weeks}")


if __name__ == "__main__":
    main()
