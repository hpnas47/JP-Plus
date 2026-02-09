#!/usr/bin/env python3
"""Weekly odds capture for opening and closing lines.

This script is designed to be run twice per week during CFB season:
1. Sunday morning - Capture opening lines (lines post by 8-10 AM ET)
2. Saturday morning - Capture closing lines (before games start)

IMPORTANT: --year and --week are REQUIRED for --opening/--closing to prevent
week mislabeling that corrupts opening-vs-closing analysis. The naive date-based
heuristic can assign Oct 31 → week 9 but Nov 2 → week 10 for the SAME CFB week.

Usage:
    # Capture opening lines (run Sunday ~8 AM ET)
    python scripts/weekly_odds_capture.py --opening --year 2026 --week 10

    # Capture closing lines (run Saturday ~9 AM ET)
    python scripts/weekly_odds_capture.py --closing --year 2026 --week 10

    # Check what's available without capturing (week args optional)
    python scripts/weekly_odds_capture.py --preview

Schedule with cron (example for 2026 Week 10):
    # Opening lines: Sunday 8 AM ET (1 PM UTC)
    0 13 * * 0 cd /path/to/project && python scripts/weekly_odds_capture.py --opening --year 2026 --week 10

    # Closing lines: Saturday 9 AM ET (2 PM UTC)
    0 14 * * 6 cd /path/to/project && python scripts/weekly_odds_capture.py --closing --year 2026 --week 10

    NOTE: Update --week each week, or use a wrapper script that computes the week.

Environment:
    ODDS_API_KEY: Your Odds API key (required)
"""

import argparse
import logging
import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.odds_api_client import OddsAPIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DB_PATH = project_root / "data" / "odds_api_lines.db"


def get_current_week() -> tuple[int, int]:
    """DEPRECATED: Estimate current CFB week based on date.

    WARNING: This heuristic can be off by 1-2 weeks, especially at month
    boundaries. If opening lines are captured Oct 31 (→ week 9) and closing
    lines Nov 2 (→ week 10), they get stored under different week labels for
    the SAME CFB week. This corrupts opening-vs-closing analysis.

    Use explicit --year and --week arguments instead. This function is only
    kept for --preview mode where week labeling doesn't affect stored data.

    Returns:
        Tuple of (year, week_number)
    """
    import warnings
    warnings.warn(
        "get_current_week() uses a naive heuristic that can mislabel weeks. "
        "Use --year and --week arguments for reliable labeling.",
        DeprecationWarning,
        stacklevel=2
    )

    now = datetime.now()
    year = now.year

    # CFB season typically runs late August through early January
    # Week 0: Late August
    # Week 1: First week of September
    # Weeks 2-13: Regular season
    # Weeks 14-15: Conference championships / bowl selection
    # Week 16+: Bowl season

    month = now.month
    day = now.day

    if month < 8:
        # Before season - return previous year's bowl season or upcoming season
        return (year, 0)
    elif month == 8:
        if day < 20:
            return (year, 0)  # Pre-season
        else:
            return (year, 0)  # Week 0
    elif month == 9:
        # Weeks 1-4 roughly
        week = (day - 1) // 7 + 1
        return (year, min(week, 4))
    elif month == 10:
        # Weeks 5-9 roughly
        week = (day - 1) // 7 + 5
        return (year, min(week, 9))
    elif month == 11:
        # Weeks 10-14 roughly
        week = (day - 1) // 7 + 10
        return (year, min(week, 14))
    elif month == 12:
        # Weeks 14-16+ (conference champs, bowl season)
        if day < 15:
            return (year, 15)
        else:
            return (year, 16)
    else:  # January
        # Bowl season continuation
        return (year - 1, 17)


def init_database() -> sqlite3.Connection:
    """Initialize database connection."""
    import re

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # P0.3: Enable foreign key enforcement (off by default in SQLite)
    conn.execute("PRAGMA foreign_keys = ON")

    # Ensure tables exist
    # P0: UNIQUE(snapshot_type, season, week) prevents duplicate captures
    # snapshot_type is just "opening" or "closing" (not composite label)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_type TEXT NOT NULL,
            snapshot_time TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            credits_used INTEGER,
            season INTEGER,
            week INTEGER,
            UNIQUE(snapshot_type, season, week)
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
            cfbd_game_id INTEGER,
            FOREIGN KEY (snapshot_id) REFERENCES odds_snapshots(id),
            UNIQUE(snapshot_id, game_id, sportsbook)
        )
    """)

    # Migration: Add columns if missing (for existing databases)
    for col, coltype in [("season", "INTEGER"), ("week", "INTEGER")]:
        try:
            cursor.execute(f"ALTER TABLE odds_snapshots ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    try:
        cursor.execute("ALTER TABLE odds_lines ADD COLUMN cfbd_game_id INTEGER")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # P0: Migration - convert composite snapshot_type labels to simple labels
    # Old format: "opening_2024_week5" -> New format: "opening" with season=2024, week=5
    cursor.execute("SELECT id, snapshot_type, season, week FROM odds_snapshots")
    rows = cursor.fetchall()
    migrated = 0
    for row_id, snap_type, season, week in rows:
        # Check if this is an old composite label
        match = re.match(r'^(opening|closing)_(\d{4})_week(\d+)$', snap_type)
        if match:
            new_type = match.group(1)  # "opening" or "closing"
            parsed_season = int(match.group(2))
            parsed_week = int(match.group(3))
            # Update to simple label and ensure season/week are set
            cursor.execute(
                "UPDATE odds_snapshots SET snapshot_type = ?, season = ?, week = ? WHERE id = ?",
                (new_type, parsed_season, parsed_week, row_id)
            )
            migrated += 1
    if migrated > 0:
        logger.info(f"Migrated {migrated} composite snapshot labels to simple format")

    # P0: Migration - update UNIQUE constraint from (type, time) to (type, season, week)
    # Check if old index exists and needs migration
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='odds_snapshots'")
    table_sql = cursor.fetchone()
    if table_sql and 'UNIQUE(snapshot_type, snapshot_time)' in table_sql[0]:
        # Old schema detected - need to recreate table with new constraint
        logger.info("Migrating odds_snapshots table to new UNIQUE constraint...")
        cursor.execute("ALTER TABLE odds_snapshots RENAME TO odds_snapshots_old")
        cursor.execute("""
            CREATE TABLE odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_type TEXT NOT NULL,
                snapshot_time TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                credits_used INTEGER,
                season INTEGER,
                week INTEGER,
                UNIQUE(snapshot_type, season, week)
            )
        """)
        cursor.execute("""
            INSERT INTO odds_snapshots (id, snapshot_type, snapshot_time, captured_at, credits_used, season, week)
            SELECT id, snapshot_type, snapshot_time, captured_at, credits_used, season, week
            FROM odds_snapshots_old
        """)
        cursor.execute("DROP TABLE odds_snapshots_old")
        logger.info("Migration complete: UNIQUE constraint updated to (snapshot_type, season, week)")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lines_game_id ON odds_lines(game_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lines_teams ON odds_lines(home_team, away_team)")
    conn.commit()
    return conn


def capture_odds(
    client: OddsAPIClient,
    conn: sqlite3.Connection,
    snapshot_type: str,
    year: int,
    week: int,
) -> dict:
    """Capture current odds and store them.

    Args:
        client: Odds API client
        conn: Database connection
        snapshot_type: 'opening' or 'closing'
        year: Season year (REQUIRED - no auto-detection)
        week: Week number (REQUIRED - no auto-detection)

    Returns:
        Summary dict

    Raises:
        ValueError: If year or week is None
    """
    # P0: Explicit year/week REQUIRED - heuristic removed to prevent week mislabeling
    if year is None or week is None:
        raise ValueError(
            "year and week are required. The auto-detection heuristic has been removed "
            "because it can mislabel weeks at month boundaries, corrupting opening-vs-closing analysis."
        )

    logger.info(f"Capturing {snapshot_type} lines for {year} week {week}...")

    snapshot = client.get_current_odds()

    if not snapshot.lines:
        logger.warning("No lines returned from API")
        return {
            'type': snapshot_type,
            'year': year,
            'week': week,
            'games': 0,
            'lines': 0,
            'credits_remaining': snapshot.credits_remaining,
        }

    # P0: Wrap entire insertion in explicit transaction for all-or-nothing semantics
    # If any insert fails, the entire batch is rolled back (no partial writes)
    cursor = conn.cursor()

    # Pre-filter lines with null spreads before starting transaction
    valid_lines = []
    null_spread_skipped = 0
    spread_anomalies = 0

    for line in snapshot.lines:
        # P0: Filter out lines with null spreads (table requires NOT NULL)
        if line.spread_home is None or line.spread_away is None:
            null_spread_skipped += 1
            logger.debug(
                f"Skipping line with null spread: {line.away_team} @ {line.home_team} "
                f"({line.sportsbook}): home={line.spread_home}, away={line.spread_away}"
            )
            continue

        # P1.3: Spread consistency check (home + away should be near-zero)
        if abs(line.spread_home + line.spread_away) > 0.5:
            spread_anomalies += 1
            logger.warning(
                f"Spread anomaly: {line.home_team} vs {line.away_team} "
                f"({line.sportsbook}): home={line.spread_home}, away={line.spread_away}, "
                f"sum={line.spread_home + line.spread_away:.1f}"
            )

        valid_lines.append(line)

    if null_spread_skipped > 0:
        logger.warning(f"P0: Filtered out {null_spread_skipped} lines with null spreads")

    # Begin explicit transaction
    try:
        cursor.execute("BEGIN")

        # Store snapshot
        # P0: UNIQUE(snapshot_type, season, week) prevents duplicate captures
        # snapshot_type is just "opening" or "closing" (not composite label)
        cursor.execute("""
            INSERT INTO odds_snapshots
            (snapshot_type, snapshot_time, captured_at, credits_used, season, week)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_type, season, week) DO UPDATE SET
                snapshot_time = excluded.snapshot_time,
                captured_at = excluded.captured_at,
                credits_used = excluded.credits_used
        """, (
            snapshot_type,  # Just "opening" or "closing"
            snapshot.timestamp.isoformat(),
            datetime.now(timezone.utc).isoformat(),
            snapshot.credits_used,
            year,
            week,
        ))

        # Retrieve the snapshot_id (lastrowid may be 0 on UPDATE)
        cursor.execute(
            "SELECT id FROM odds_snapshots WHERE snapshot_type = ? AND season = ? AND week = ?",
            (snapshot_type, year, week)
        )
        snapshot_id = cursor.fetchone()[0]

        # Store all valid lines
        games_seen = set()
        for line in valid_lines:
            games_seen.add(line.game_id)
            cursor.execute("""
                INSERT INTO odds_lines
                (snapshot_id, game_id, sportsbook, home_team, away_team,
                 spread_home, spread_away, price_home, price_away,
                 commence_time, last_update)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id, game_id, sportsbook) DO UPDATE SET
                    spread_home = excluded.spread_home,
                    spread_away = excluded.spread_away,
                    price_home = excluded.price_home,
                    price_away = excluded.price_away,
                    last_update = excluded.last_update
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

        # All inserts succeeded - commit transaction
        conn.commit()
        logger.info(f"Transaction committed: {len(valid_lines)} lines for {len(games_seen)} games")

    except Exception as e:
        # Any error - rollback entire transaction
        conn.rollback()
        logger.error(f"Transaction rolled back due to error: {e}")
        raise  # Re-raise so caller knows capture failed

    # Log summary (games_seen is set in the try block, only reached on success)
    if spread_anomalies > 0:
        logger.warning(f"P1.3: {spread_anomalies} spread anomalies detected (home+away != 0)")
    logger.info(f"Credits remaining: {snapshot.credits_remaining}")

    return {
        'type': snapshot_type,
        'year': year,
        'week': week,
        'games': len(games_seen),
        'lines': len(valid_lines),
        'lines_from_api': len(snapshot.lines),
        'null_spread_skipped': null_spread_skipped,
        'credits_remaining': snapshot.credits_remaining,
    }


def preview_odds(client: OddsAPIClient, year: int | None = None, week: int | None = None) -> None:
    """Preview available odds without storing (still uses 1 credit)."""
    if year is None or week is None:
        year, week = get_current_week()

    print(f"\nPreviewing NCAAF odds for {year} week {week}...")
    print("(This will use 1 credit)\n")

    snapshot = client.get_current_odds()

    if not snapshot.lines:
        print("No games currently available")
        return

    # P2.1: Group by game_id when available, fall back to team-name key
    games = {}
    for line in snapshot.lines:
        key = line.game_id if line.game_id else (line.home_team, line.away_team)
        if key not in games:
            games[key] = {
                'home': line.home_team,
                'away': line.away_team,
                'commence': line.commence_time,
                'lines': []
            }
        games[key]['lines'].append(line)

    print(f"Found {len(games)} games:\n")
    for _key, info in sorted(games.items(), key=lambda x: str(x[1]['commence'])):
        home, away = info['home'], info['away']
        commence = info['commence'].strftime('%Y-%m-%d %H:%M') if info['commence'] else 'TBD'
        print(f"{away} @ {home} ({commence})")
        for line in info['lines'][:3]:  # Show up to 3 books
            print(f"  {line.sportsbook}: {line.spread_home:+.1f} ({line.price_home})")
        if len(info['lines']) > 3:
            print(f"  ... and {len(info['lines']) - 3} more books")
        print()

    print(f"Credits remaining: {snapshot.credits_remaining}")


def main():
    parser = argparse.ArgumentParser(
        description="Weekly odds capture",
        epilog="IMPORTANT: --year and --week are REQUIRED for --opening/--closing to prevent "
               "week mislabeling that corrupts opening-vs-closing analysis."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--opening", action="store_true",
                      help="Capture opening lines (run Sunday)")
    group.add_argument("--closing", action="store_true",
                      help="Capture closing lines (run Saturday)")
    group.add_argument("--preview", action="store_true",
                      help="Preview available odds (week args optional)")
    parser.add_argument("--api-key", type=str,
                       help="Odds API key (or set ODDS_API_KEY)")
    # P0: Explicit year/week REQUIRED for capture to prevent week mislabeling
    parser.add_argument("--year", type=int, default=None,
                       help="Season year (REQUIRED for --opening/--closing)")
    parser.add_argument("--week", type=int, default=None,
                       help="Week number (REQUIRED for --opening/--closing)")

    args = parser.parse_args()

    # P0: Require --year and --week for capture operations to prevent week mislabeling
    # The naive get_current_week() heuristic can assign Oct 31 → week 9 but Nov 2 → week 10,
    # corrupting opening-vs-closing analysis for the same CFB week.
    if (args.opening or args.closing) and (args.year is None or args.week is None):
        parser.error(
            "--year and --week are REQUIRED for --opening/--closing.\n"
            "The auto-detection heuristic can mislabel weeks at month boundaries,\n"
            "causing opening and closing lines for the same CFB week to be stored\n"
            "under different week labels. This corrupts opening-vs-closing analysis.\n\n"
            "Example: python scripts/weekly_odds_capture.py --opening --year 2026 --week 10"
        )

    api_key = args.api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        logger.error("API key required. Set ODDS_API_KEY or use --api-key")
        sys.exit(1)

    client = OddsAPIClient(api_key=api_key)

    if args.preview:
        preview_odds(client, year=args.year, week=args.week)
    else:
        # P0: Use closing() to ensure conn.close() even if capture_odds() raises
        snapshot_type = "opening" if args.opening else "closing"
        with closing(init_database()) as conn:
            result = capture_odds(client, conn, snapshot_type, year=args.year, week=args.week)

        print(f"\n{snapshot_type.title()} lines captured:")
        print(f"  Season: {result['year']} Week {result['week']}")
        print(f"  Games: {result['games']}")
        print(f"  Lines stored: {result['lines']}")
        if result.get('null_spread_skipped', 0) > 0:
            print(f"  ⚠ Skipped (null spread): {result['null_spread_skipped']}")
        print(f"  Credits remaining: {result['credits_remaining']}")


if __name__ == "__main__":
    main()
