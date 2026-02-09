#!/usr/bin/env python3
"""Link Odds API game IDs to CFBD game IDs.

This script matches captured odds records to CFBD games using team names
and game dates, then populates the cfbd_game_id column for fast lookups.

Usage:
    # Dry run - show what would be linked
    python scripts/link_odds_to_cfbd.py --year 2026 --dry-run

    # Actually link the records
    python scripts/link_odds_to_cfbd.py --year 2026

    # Link all unlinked records across all years
    python scripts/link_odds_to_cfbd.py --all
"""

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.cfbd_client import CFBDClient
from config.teams import normalize_team_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = project_root / "data" / "odds_api_lines.db"


def get_unlinked_games(conn: sqlite3.Connection, year: int = None) -> list[dict]:
    """Get odds records that don't have a cfbd_game_id.

    Args:
        conn: Database connection
        year: Optional year filter (extracted from commence_time)

    Returns:
        List of dicts with game info needing linking
    """
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if year:
        # Filter by year in commence_time (format: 2026-09-05T20:00:00Z)
        cursor.execute("""
            SELECT DISTINCT game_id, home_team, away_team, commence_time
            FROM odds_lines
            WHERE cfbd_game_id IS NULL
              AND commence_time LIKE ?
            ORDER BY commence_time
        """, (f"{year}-%",))
    else:
        cursor.execute("""
            SELECT DISTINCT game_id, home_team, away_team, commence_time
            FROM odds_lines
            WHERE cfbd_game_id IS NULL
            ORDER BY commence_time
        """)

    return [dict(row) for row in cursor.fetchall()]


def build_cfbd_game_index(client: CFBDClient, year: int) -> dict[tuple, int]:
    """Build an index of CFBD games for fast matching.

    Args:
        client: CFBD API client
        year: Season year

    Returns:
        Dict mapping (norm_home, norm_away, date_str) -> cfbd_game_id
    """
    index = {}

    # Fetch all games for the year
    games = client.get_games(year=year, season_type="both")

    for game in games:
        # Normalize team names for matching
        home_norm = normalize_team_name(game.home_team)
        away_norm = normalize_team_name(game.away_team)

        # Extract date (YYYY-MM-DD) from start_date
        if game.start_date:
            date_str = game.start_date[:10]  # "2026-09-05T20:00:00.000Z" -> "2026-09-05"
        else:
            continue

        # Primary key: normalized teams + date
        key = (home_norm, away_norm, date_str)
        index[key] = game.id

        # Also index with swapped teams for neutral site games
        # (Odds API might list teams in different order)
        key_swapped = (away_norm, home_norm, date_str)
        if key_swapped not in index:
            index[key_swapped] = game.id

    logger.info(f"Built CFBD index with {len(index)} game keys for {year}")
    return index


def match_game(
    odds_game: dict,
    cfbd_index: dict[tuple, int],
) -> int | None:
    """Try to match an Odds API game to a CFBD game.

    Args:
        odds_game: Dict with home_team, away_team, commence_time
        cfbd_index: Index from build_cfbd_game_index

    Returns:
        CFBD game_id if matched, None otherwise
    """
    # Normalize team names
    home_norm = normalize_team_name(odds_game['home_team'])
    away_norm = normalize_team_name(odds_game['away_team'])

    # Extract date from commence_time
    commence = odds_game['commence_time']
    if commence:
        date_str = commence[:10]  # "2026-09-05T20:00:00Z" -> "2026-09-05"
    else:
        return None

    # Try exact match
    key = (home_norm, away_norm, date_str)
    if key in cfbd_index:
        return cfbd_index[key]

    # Try swapped (for neutral sites or different home/away designation)
    key_swapped = (away_norm, home_norm, date_str)
    if key_swapped in cfbd_index:
        return cfbd_index[key_swapped]

    return None


def link_games(
    conn: sqlite3.Connection,
    year: int,
    client: CFBDClient,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Link unlinked odds records for a given year.

    Args:
        conn: Database connection
        year: Season year
        client: CFBD API client
        dry_run: If True, don't actually update database

    Returns:
        Tuple of (linked_count, unlinked_count)
    """
    # Get unlinked games
    unlinked = get_unlinked_games(conn, year)
    if not unlinked:
        logger.info(f"No unlinked games for {year}")
        return 0, 0

    logger.info(f"Found {len(unlinked)} unlinked games for {year}")

    # Build CFBD index
    cfbd_index = build_cfbd_game_index(client, year)

    linked = 0
    failed = []

    for game in unlinked:
        cfbd_id = match_game(game, cfbd_index)

        if cfbd_id:
            linked += 1
            if not dry_run:
                conn.execute("""
                    UPDATE odds_lines
                    SET cfbd_game_id = ?
                    WHERE game_id = ?
                """, (cfbd_id, game['game_id']))
            else:
                logger.debug(
                    f"Would link: {game['away_team']} @ {game['home_team']} "
                    f"({game['commence_time'][:10]}) -> CFBD {cfbd_id}"
                )
        else:
            failed.append(game)

    if not dry_run:
        conn.commit()

    # Log failures
    if failed:
        logger.warning(f"Failed to link {len(failed)} games:")
        for game in failed[:10]:  # Show first 10
            logger.warning(
                f"  - {game['away_team']} @ {game['home_team']} "
                f"({game['commence_time'][:10] if game['commence_time'] else 'no date'})"
            )
        if len(failed) > 10:
            logger.warning(f"  ... and {len(failed) - 10} more")

    action = "Would link" if dry_run else "Linked"
    logger.info(f"{action} {linked}/{len(unlinked)} games for {year}")

    return linked, len(failed)


def main():
    parser = argparse.ArgumentParser(description="Link Odds API games to CFBD game IDs")
    parser.add_argument("--year", type=int, help="Season year to link")
    parser.add_argument("--all", action="store_true", help="Link all unlinked records")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be linked")

    args = parser.parse_args()

    if not args.year and not args.all:
        parser.error("Must specify --year or --all")

    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    client = CFBDClient()

    total_linked = 0
    total_failed = 0

    try:
        if args.year:
            linked, failed = link_games(conn, args.year, client, args.dry_run)
            total_linked += linked
            total_failed += failed
        else:
            # Find all years with unlinked data
            cursor = conn.execute("""
                SELECT DISTINCT substr(commence_time, 1, 4) as year
                FROM odds_lines
                WHERE cfbd_game_id IS NULL
                  AND commence_time IS NOT NULL
                ORDER BY year
            """)
            years = [int(row[0]) for row in cursor.fetchall()]

            for year in years:
                linked, failed = link_games(conn, year, client, args.dry_run)
                total_linked += linked
                total_failed += failed

        logger.info(f"Total: linked {total_linked}, failed {total_failed}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
