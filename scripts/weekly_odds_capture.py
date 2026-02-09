#!/usr/bin/env python3
"""Weekly odds capture for opening and closing lines.

This script is designed to be run twice per week during CFB season:
1. Sunday morning - Capture opening lines (lines post by 8-10 AM ET)
2. Saturday morning - Capture closing lines (before games start)

Usage:
    # Capture opening lines (run Sunday ~8 AM ET)
    python scripts/weekly_odds_capture.py --opening

    # Capture closing lines (run Saturday ~9 AM ET)
    python scripts/weekly_odds_capture.py --closing

    # Check what's available without capturing
    python scripts/weekly_odds_capture.py --preview

Schedule with cron (example):
    # Opening lines: Sunday 8 AM ET (1 PM UTC)
    0 13 * * 0 cd /path/to/project && python scripts/weekly_odds_capture.py --opening

    # Closing lines: Saturday 9 AM ET (2 PM UTC)
    0 14 * * 6 cd /path/to/project && python scripts/weekly_odds_capture.py --closing

Environment:
    ODDS_API_KEY: Your Odds API key (required)
"""

import argparse
import logging
import os
import sqlite3
import sys
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
    """Estimate current CFB week based on date.

    Returns:
        Tuple of (year, week_number)
    """
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
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # P0.3: Enable foreign key enforcement (off by default in SQLite)
    conn.execute("PRAGMA foreign_keys = ON")

    # Ensure tables exist
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
            cfbd_game_id INTEGER,
            FOREIGN KEY (snapshot_id) REFERENCES odds_snapshots(id),
            UNIQUE(snapshot_id, game_id, sportsbook)
        )
    """)
    # P1.1: Migrate existing tables — add new columns if missing
    for col, coltype in [("season", "INTEGER"), ("week", "INTEGER")]:
        try:
            cursor.execute(f"ALTER TABLE odds_snapshots ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    try:
        cursor.execute("ALTER TABLE odds_lines ADD COLUMN cfbd_game_id INTEGER")
    except sqlite3.OperationalError:
        pass  # Column already exists

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lines_game_id ON odds_lines(game_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lines_teams ON odds_lines(home_team, away_team)")
    conn.commit()
    return conn


def capture_odds(
    client: OddsAPIClient,
    conn: sqlite3.Connection,
    snapshot_type: str,
    year: int | None = None,
    week: int | None = None,
) -> dict:
    """Capture current odds and store them.

    Args:
        client: Odds API client
        conn: Database connection
        snapshot_type: 'opening' or 'closing'
        year: Explicit season year (preferred over heuristic)
        week: Explicit week number (preferred over heuristic)

    Returns:
        Summary dict
    """
    # P0.1: Prefer explicit year/week, fall back to heuristic with warning
    if year is not None and week is not None:
        logger.info(f"Using explicit year={year}, week={week}")
    else:
        year, week = get_current_week()
        logger.warning(
            f"P0.1: Using heuristic week estimation (year={year}, week={week}). "
            "For reliable labeling, use --year and --week arguments."
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

    # Store snapshot
    # P0.2: Use INSERT...ON CONFLICT DO UPDATE to preserve snapshot_id (avoids orphaned lines)
    # P1.1: Store snapshot_type as just "opening"/"closing" with season/week in dedicated columns
    cursor = conn.cursor()
    snapshot_label = f"{snapshot_type}_{year}_week{week}"

    cursor.execute("""
        INSERT INTO odds_snapshots
        (snapshot_type, snapshot_time, captured_at, credits_used, season, week)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(snapshot_type, snapshot_time) DO UPDATE SET
            captured_at = excluded.captured_at,
            credits_used = excluded.credits_used,
            season = excluded.season,
            week = excluded.week
    """, (
        snapshot_label,
        snapshot.timestamp.isoformat(),
        datetime.now(timezone.utc).isoformat(),  # P2.3: UTC-normalized timestamp
        snapshot.credits_used,
        year,
        week,
    ))

    # P0.2: Retrieve the actual snapshot_id (lastrowid may be 0 on UPDATE)
    cursor.execute(
        "SELECT id FROM odds_snapshots WHERE snapshot_type = ? AND snapshot_time = ?",
        (snapshot_label, snapshot.timestamp.isoformat())
    )
    snapshot_id = cursor.fetchone()[0]

    # Store lines
    # P0.2: Use INSERT...ON CONFLICT DO UPDATE to preserve row identity
    games_seen = set()
    spread_anomalies = 0
    for line in snapshot.lines:
        games_seen.add(line.game_id)

        # P1.3: Spread consistency check (home + away should be near-zero)
        if (line.spread_home is not None and line.spread_away is not None
                and abs(line.spread_home + line.spread_away) > 0.5):
            spread_anomalies += 1
            logger.warning(
                f"Spread anomaly: {line.home_team} vs {line.away_team} "
                f"({line.sportsbook}): home={line.spread_home}, away={line.spread_away}, "
                f"sum={line.spread_home + line.spread_away:.1f}"
            )
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

    conn.commit()

    logger.info(f"Stored {len(snapshot.lines)} lines for {len(games_seen)} games")
    if spread_anomalies > 0:
        logger.warning(f"P1.3: {spread_anomalies} spread anomalies detected (home+away != 0)")
    logger.info(f"Credits remaining: {snapshot.credits_remaining}")

    return {
        'type': snapshot_type,
        'year': year,
        'week': week,
        'games': len(games_seen),
        'lines': len(snapshot.lines),
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
    parser = argparse.ArgumentParser(description="Weekly odds capture")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--opening", action="store_true",
                      help="Capture opening lines (run Sunday)")
    group.add_argument("--closing", action="store_true",
                      help="Capture closing lines (run Saturday)")
    group.add_argument("--preview", action="store_true",
                      help="Preview available odds")
    parser.add_argument("--api-key", type=str,
                       help="Odds API key (or set ODDS_API_KEY)")
    # P0.1: Explicit year/week for reliable labeling
    parser.add_argument("--year", type=int, default=None,
                       help="Season year (recommended over auto-detection)")
    parser.add_argument("--week", type=int, default=None,
                       help="Week number (recommended over auto-detection)")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        logger.error("API key required. Set ODDS_API_KEY or use --api-key")
        sys.exit(1)

    client = OddsAPIClient(api_key=api_key)

    if args.preview:
        preview_odds(client, year=args.year, week=args.week)
    else:
        conn = init_database()
        snapshot_type = "opening" if args.opening else "closing"
        result = capture_odds(client, conn, snapshot_type, year=args.year, week=args.week)
        conn.close()

        print(f"\n{snapshot_type.title()} lines captured:")
        print(f"  Season: {result['year']} Week {result['week']}")
        print(f"  Games: {result['games']}")
        print(f"  Lines: {result['lines']}")
        print(f"  Credits remaining: {result['credits_remaining']}")


if __name__ == "__main__":
    main()
