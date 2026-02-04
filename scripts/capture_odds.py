#!/usr/bin/env python3
"""Capture and store betting odds from The Odds API.

This script captures opening and closing lines for NCAAF games and stores
them in a local SQLite database for use in backtesting.

Usage:
    # Check quota (free)
    python scripts/capture_odds.py --check-quota

    # Capture current odds (1 credit)
    python scripts/capture_odds.py --capture-current

    # Capture historical odds for a date (10 credits)
    python scripts/capture_odds.py --capture-historical 2024-09-07T12:00:00Z

    # Backfill missing opening lines from CFBD data
    python scripts/capture_odds.py --backfill --year 2024 --dry-run
    python scripts/capture_odds.py --backfill --year 2024

Environment:
    ODDS_API_KEY: Your Odds API key (required for API calls)
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.odds_api_client import OddsAPIClient, OddsSnapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = project_root / "data" / "odds_api_lines.db"


def init_database(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize the SQLite database for storing odds."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_type TEXT NOT NULL,  -- 'opening', 'closing', 'current'
            snapshot_time TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            credits_used INTEGER,
            UNIQUE(snapshot_type, snapshot_time)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS odds_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            sportsbook TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            spread_home REAL NOT NULL,
            spread_away REAL NOT NULL,
            price_home INTEGER,
            price_away INTEGER,
            commence_time TEXT,
            last_update TEXT,
            FOREIGN KEY (snapshot_id) REFERENCES odds_snapshots(id),
            UNIQUE(snapshot_id, game_id, sportsbook)
        )
    """)

    # Create indexes for fast lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_lines_game_id ON odds_lines(game_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_lines_teams ON odds_lines(home_team, away_team)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_type ON odds_snapshots(snapshot_type)
    """)

    conn.commit()
    return conn


def store_snapshot(
    conn: sqlite3.Connection,
    snapshot: OddsSnapshot,
    snapshot_type: str,
) -> int:
    """Store an odds snapshot in the database.

    Args:
        conn: Database connection
        snapshot: The OddsSnapshot to store
        snapshot_type: Type of snapshot ('opening', 'closing', 'current')

    Returns:
        The snapshot ID
    """
    cursor = conn.cursor()

    # Insert snapshot record
    cursor.execute("""
        INSERT OR REPLACE INTO odds_snapshots
        (snapshot_type, snapshot_time, captured_at, credits_used)
        VALUES (?, ?, ?, ?)
    """, (
        snapshot_type,
        snapshot.timestamp.isoformat(),
        datetime.utcnow().isoformat(),
        snapshot.credits_used,
    ))

    snapshot_id = cursor.lastrowid

    # Insert lines
    for line in snapshot.lines:
        cursor.execute("""
            INSERT OR REPLACE INTO odds_lines
            (snapshot_id, game_id, sportsbook, home_team, away_team,
             spread_home, spread_away, price_home, price_away,
             commence_time, last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot_id,
            line.game_id,
            line.sportsbook,
            line.home_team,
            line.away_team,
            line.spread_home,
            line.spread_away,
            line.price_home,
            line.price_away,
            line.commence_time.isoformat() if line.commence_time else None,
            line.last_update.isoformat() if line.last_update else None,
        ))

    conn.commit()
    logger.info(f"Stored {len(snapshot.lines)} lines in snapshot {snapshot_id}")
    return snapshot_id


def get_weeks_needing_backfill(year: int) -> list[dict]:
    """Identify weeks that need opening line backfill from CFBD data.

    Args:
        year: The year to analyze

    Returns:
        List of dicts with week info and first game date
    """
    from src.api.cfbd_client import CFBDClient

    client = CFBDClient()
    preferred_providers = ["DraftKings", "ESPN Bet", "Bovada"]

    # Get FBS teams
    fbs_teams_raw = client.get_fbs_teams(year=year)
    fbs_teams = set(t.school for t in fbs_teams_raw)

    # Get all games
    games = client.get_games(year=year, season_type='regular')
    games.extend(client.get_games(year=year, season_type='postseason') or [])

    # Build game info lookup (start_date is already a datetime object)
    game_info = {}
    for g in games:
        if g.home_team in fbs_teams or g.away_team in fbs_teams:
            game_info[g.id] = {
                'week': g.week,
                'date': g.start_date,  # Already a datetime object
                'home': g.home_team,
                'away': g.away_team,
            }

    # Analyze betting lines
    lines = client.get_betting_lines(year=year)
    week_stats = {}

    for game_lines in (lines or []):
        if game_lines.id not in game_info:
            continue

        info = game_info[game_lines.id]
        week = info['week']

        if week not in week_stats:
            week_stats[week] = {
                'total': 0,
                'missing': 0,
                'first_date': info['date'],  # datetime object
                'games': [],
            }

        # Track earliest game date for this week
        if info['date'] and (week_stats[week]['first_date'] is None or
                             info['date'] < week_stats[week]['first_date']):
            week_stats[week]['first_date'] = info['date']

        # Check for opening line
        selected_line = None
        for provider in preferred_providers:
            for line in game_lines.lines:
                if line.provider and line.provider == provider:
                    selected_line = line
                    break
            if selected_line:
                break

        if selected_line is None and game_lines.lines:
            selected_line = game_lines.lines[0]

        if selected_line and selected_line.spread is not None:
            week_stats[week]['total'] += 1
            if getattr(selected_line, 'spread_open', None) is None:
                week_stats[week]['missing'] += 1
                week_stats[week]['games'].append({
                    'home': info['home'],
                    'away': info['away'],
                    'date': info['date'],
                })

    # Return weeks with missing data
    weeks_needing = []
    for week, stats in sorted(week_stats.items()):
        if stats['missing'] > 0:
            weeks_needing.append({
                'year': year,
                'week': week,
                'missing': stats['missing'],
                'total': stats['total'],
                'first_date': stats['first_date'],
                'games': stats['games'],
            })

    return weeks_needing


def backfill_opening_lines(
    client: OddsAPIClient,
    conn: sqlite3.Connection,
    year: int,
    dry_run: bool = True,
) -> dict:
    """Backfill missing opening lines using historical API.

    For each week with missing opening lines, fetches historical odds
    from Sunday before the first game of that week.

    Args:
        client: The Odds API client
        conn: Database connection
        year: Year to backfill
        dry_run: If True, only report what would be done (no API calls)

    Returns:
        Summary dict with stats
    """
    weeks = get_weeks_needing_backfill(year)

    if not weeks:
        logger.info(f"No weeks need backfill for {year}")
        return {'year': year, 'weeks_processed': 0, 'credits_used': 0}

    total_missing = sum(w['missing'] for w in weeks)
    estimated_credits = len(weeks) * 10

    logger.info(f"\n{year} Backfill Summary:")
    logger.info(f"  Weeks needing data: {len(weeks)}")
    logger.info(f"  Total missing games: {total_missing}")
    logger.info(f"  Estimated credits: {estimated_credits}")

    if dry_run:
        logger.info("\n[DRY RUN] Would fetch historical odds for:")
        for w in weeks:
            # Calculate Sunday before first game
            if w['first_date']:
                first_date = w['first_date']  # Already a datetime object
                # Go back to Sunday (opening lines typically post Sunday/Monday)
                days_since_sunday = first_date.weekday() + 1  # Monday=0, so +1
                if days_since_sunday > 6:
                    days_since_sunday = 0  # Already Sunday
                opening_date = first_date - timedelta(days=days_since_sunday)
                opening_date = opening_date.replace(hour=18, minute=0, second=0)  # 6 PM ET
                date_str = opening_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                date_str = "unknown"

            logger.info(f"  Week {w['week']}: {w['missing']} games, query date: {date_str}")
            for g in w['games'][:3]:
                game_date = g['date'].strftime('%Y-%m-%d') if g['date'] else 'no date'
                logger.info(f"    - {g['away']} @ {g['home']} ({game_date})")
            if len(w['games']) > 3:
                logger.info(f"    ... and {len(w['games']) - 3} more")

        return {
            'year': year,
            'weeks_processed': 0,
            'credits_used': 0,
            'dry_run': True,
            'would_use_credits': estimated_credits,
        }

    # Actually fetch data
    credits_used = 0
    weeks_processed = 0

    for w in weeks:
        if not w['first_date']:
            logger.warning(f"Week {w['week']}: no date info, skipping")
            continue

        # Calculate query date (Sunday before first game, 6 PM ET)
        first_date = w['first_date']  # Already a datetime object
        days_since_sunday = first_date.weekday() + 1
        if days_since_sunday > 6:
            days_since_sunday = 0
        opening_date = first_date - timedelta(days=days_since_sunday)
        opening_date = opening_date.replace(hour=23, minute=0, second=0)  # 11 PM UTC (6 PM ET)
        date_str = opening_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(f"Fetching week {w['week']} opening lines (date: {date_str})...")

        try:
            snapshot = client.get_historical_odds(date=date_str)
            store_snapshot(conn, snapshot, f"opening_{year}_week{w['week']}")
            credits_used += 10
            weeks_processed += 1
            logger.info(f"  Got {len(snapshot.lines)} lines, credits remaining: {snapshot.credits_remaining}")
        except Exception as e:
            logger.error(f"  Error fetching week {w['week']}: {e}")

    return {
        'year': year,
        'weeks_processed': weeks_processed,
        'credits_used': credits_used,
    }


def main():
    parser = argparse.ArgumentParser(description="Capture and store betting odds")
    parser.add_argument("--check-quota", action="store_true",
                       help="Check API quota (free)")
    parser.add_argument("--capture-current", action="store_true",
                       help="Capture current odds (1 credit)")
    parser.add_argument("--capture-historical", type=str,
                       help="Capture historical odds for date (10 credits)")
    parser.add_argument("--backfill", action="store_true",
                       help="Backfill missing opening lines")
    parser.add_argument("--year", type=int, default=2024,
                       help="Year for backfill (default: 2024)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making API calls")
    parser.add_argument("--api-key", type=str,
                       help="Odds API key (or set ODDS_API_KEY env var)")

    args = parser.parse_args()

    # Initialize database
    conn = init_database()

    # Check if we need an API client
    needs_api = args.check_quota or args.capture_current or args.capture_historical
    needs_api = needs_api or (args.backfill and not args.dry_run)

    if needs_api:
        api_key = args.api_key or os.environ.get("ODDS_API_KEY")
        if not api_key:
            logger.error("API key required. Set ODDS_API_KEY or use --api-key")
            sys.exit(1)
        client = OddsAPIClient(api_key=api_key)
    else:
        client = None

    if args.check_quota:
        quota = client.check_quota()
        print(f"\nAPI Quota Status:")
        print(f"  Credits remaining: {quota['remaining']}")
        print(f"  Credits used: {quota['used']}")

    elif args.capture_current:
        snapshot = client.get_current_odds()
        store_snapshot(conn, snapshot, "current")
        print(f"\nCaptured {len(snapshot.lines)} current lines")
        print(f"Credits remaining: {snapshot.credits_remaining}")

    elif args.capture_historical:
        snapshot = client.get_historical_odds(date=args.capture_historical)
        store_snapshot(conn, snapshot, f"historical_{args.capture_historical}")
        print(f"\nCaptured {len(snapshot.lines)} historical lines")
        print(f"Snapshot timestamp: {snapshot.timestamp}")
        print(f"Credits remaining: {snapshot.credits_remaining}")

    elif args.backfill:
        result = backfill_opening_lines(
            client=client,
            conn=conn,
            year=args.year,
            dry_run=args.dry_run,
        )
        print(f"\nBackfill complete:")
        print(f"  Year: {result['year']}")
        if args.dry_run:
            print(f"  [DRY RUN] Would use {result.get('would_use_credits', 0)} credits")
        else:
            print(f"  Weeks processed: {result['weeks_processed']}")
            print(f"  Credits used: {result['credits_used']}")

    else:
        parser.print_help()

    conn.close()


if __name__ == "__main__":
    main()
