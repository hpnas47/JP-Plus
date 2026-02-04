"""Unified betting lines interface.

Merges data from multiple sources:
1. The Odds API (primary for future games)
2. CFBD API (fallback and historical data)

This module provides a single interface for accessing betting lines
regardless of the underlying data source.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to The Odds API database
ODDS_API_DB = Path(__file__).parent.parent.parent / "data" / "odds_api_lines.db"

# Provider priority for The Odds API (FanDuel posts earliest on Sunday morning)
ODDS_API_PROVIDERS = ["fanduel", "draftkings", "betmgm", "caesars", "bovada"]

# Provider priority for CFBD (historical data - FanDuel not available)
CFBD_PROVIDERS = ["DraftKings", "ESPN Bet", "Bovada"]


@dataclass
class BettingLine:
    """Unified betting line representation."""

    game_id: str
    home_team: str
    away_team: str
    spread_open: Optional[float]  # Opening spread (home team perspective)
    spread_close: Optional[float]  # Closing spread (home team perspective)
    source: str  # 'odds_api' or 'cfbd'
    sportsbook: Optional[str] = None


def get_odds_api_lines(year: int) -> dict[str, BettingLine]:
    """Get betting lines from The Odds API database.

    Args:
        year: Season year

    Returns:
        Dict mapping game_id to BettingLine
    """
    if not ODDS_API_DB.exists():
        logger.debug("Odds API database not found")
        return {}

    conn = sqlite3.connect(ODDS_API_DB)
    conn.row_factory = sqlite3.Row

    lines = {}

    try:
        # Build provider priority CASE for ordering
        priority_case = "CASE LOWER(l.sportsbook) "
        for i, provider in enumerate(ODDS_API_PROVIDERS):
            priority_case += f"WHEN '{provider}' THEN {i} "
        priority_case += f"ELSE {len(ODDS_API_PROVIDERS)} END"

        # Get opening lines for this year, ordered by provider priority
        opening_cursor = conn.execute(f"""
            SELECT l.game_id, l.home_team, l.away_team, l.spread_home, l.sportsbook
            FROM odds_lines l
            JOIN odds_snapshots s ON l.snapshot_id = s.id
            WHERE s.snapshot_type LIKE ?
            ORDER BY l.game_id, {priority_case}
        """, (f"opening_{year}%",))

        for row in opening_cursor:
            game_id = row['game_id']
            # First match per game wins (highest priority provider)
            if game_id not in lines:
                lines[game_id] = BettingLine(
                    game_id=game_id,
                    home_team=row['home_team'],
                    away_team=row['away_team'],
                    spread_open=row['spread_home'],
                    spread_close=None,
                    source='odds_api',
                    sportsbook=row['sportsbook'],
                )

        # Get closing lines for this year, ordered by provider priority
        closing_cursor = conn.execute(f"""
            SELECT l.game_id, l.home_team, l.away_team, l.spread_home, l.sportsbook
            FROM odds_lines l
            JOIN odds_snapshots s ON l.snapshot_id = s.id
            WHERE s.snapshot_type LIKE ?
            ORDER BY l.game_id, {priority_case}
        """, (f"closing_{year}%",))

        for row in closing_cursor:
            game_id = row['game_id']
            if game_id in lines:
                lines[game_id].spread_close = row['spread_home']
            else:
                lines[game_id] = BettingLine(
                    game_id=game_id,
                    home_team=row['home_team'],
                    away_team=row['away_team'],
                    spread_open=None,
                    spread_close=row['spread_home'],
                    source='odds_api',
                    sportsbook=row['sportsbook'],
                )

        logger.info(f"Loaded {len(lines)} lines from Odds API for {year}")

    except Exception as e:
        logger.warning(f"Error loading Odds API lines: {e}")

    finally:
        conn.close()

    return lines


def get_cfbd_lines(
    year: int,
    cfbd_client=None,
) -> dict[str, BettingLine]:
    """Get betting lines from CFBD API.

    Args:
        year: Season year
        cfbd_client: Optional CFBD client instance

    Returns:
        Dict mapping game_id to BettingLine
    """
    if cfbd_client is None:
        from src.api.cfbd_client import CFBDClient
        cfbd_client = CFBDClient()

    preferred_providers = CFBD_PROVIDERS
    lines = {}

    try:
        betting_data = cfbd_client.get_betting_lines(year=year)

        for game_lines in (betting_data or []):
            game_id = str(game_lines.id)

            # Select best provider
            selected_line = None
            for provider in preferred_providers:
                for line in (game_lines.lines or []):
                    if line.provider == provider:
                        selected_line = line
                        break
                if selected_line:
                    break

            if selected_line is None and game_lines.lines:
                selected_line = game_lines.lines[0]

            if selected_line and selected_line.spread is not None:
                spread_open = getattr(selected_line, 'spread_open', None)
                lines[game_id] = BettingLine(
                    game_id=game_id,
                    home_team=game_lines.home_team,
                    away_team=game_lines.away_team,
                    spread_open=spread_open,
                    spread_close=selected_line.spread,
                    source='cfbd',
                    sportsbook=selected_line.provider,
                )

        logger.info(f"Loaded {len(lines)} lines from CFBD for {year}")

    except Exception as e:
        logger.warning(f"Error loading CFBD lines: {e}")

    return lines


def get_merged_lines(
    year: int,
    cfbd_client=None,
    prefer_odds_api: bool = True,
) -> dict[str, BettingLine]:
    """Get merged betting lines from all sources.

    Merges data from The Odds API and CFBD, with configurable priority.
    For each game:
    - Uses Odds API opening line if available (better coverage)
    - Falls back to CFBD opening line
    - Uses Odds API closing line if available
    - Falls back to CFBD closing line

    Args:
        year: Season year
        cfbd_client: Optional CFBD client instance
        prefer_odds_api: If True, prefer Odds API data when both available

    Returns:
        Dict mapping game_id to BettingLine with best available data
    """
    # Load from both sources
    odds_api_lines = get_odds_api_lines(year)
    cfbd_lines = get_cfbd_lines(year, cfbd_client)

    # Merge with priority
    merged = {}

    # Start with CFBD as base (broader historical coverage)
    for game_id, line in cfbd_lines.items():
        merged[game_id] = line

    # Overlay Odds API data
    for game_id, odds_line in odds_api_lines.items():
        if game_id in merged:
            cfbd_line = merged[game_id]
            # Merge: prefer Odds API for opening lines (better coverage)
            if prefer_odds_api:
                merged[game_id] = BettingLine(
                    game_id=game_id,
                    home_team=odds_line.home_team or cfbd_line.home_team,
                    away_team=odds_line.away_team or cfbd_line.away_team,
                    spread_open=odds_line.spread_open or cfbd_line.spread_open,
                    spread_close=odds_line.spread_close or cfbd_line.spread_close,
                    source='merged',
                    sportsbook=odds_line.sportsbook or cfbd_line.sportsbook,
                )
            else:
                # Only use Odds API if CFBD doesn't have opening line
                if cfbd_line.spread_open is None and odds_line.spread_open is not None:
                    merged[game_id].spread_open = odds_line.spread_open
                    merged[game_id].source = 'merged'
        else:
            # New game only in Odds API
            merged[game_id] = odds_line

    # Log merge statistics
    total = len(merged)
    from_odds_api = sum(1 for l in merged.values() if l.source == 'odds_api')
    from_cfbd = sum(1 for l in merged.values() if l.source == 'cfbd')
    from_merged = sum(1 for l in merged.values() if l.source == 'merged')
    has_open = sum(1 for l in merged.values() if l.spread_open is not None)
    has_close = sum(1 for l in merged.values() if l.spread_close is not None)

    logger.info(
        f"Merged {year} lines: {total} total "
        f"(odds_api={from_odds_api}, cfbd={from_cfbd}, merged={from_merged}), "
        f"open={has_open} ({100*has_open/total:.0f}%), "
        f"close={has_close} ({100*has_close/total:.0f}%)"
    )

    return merged


def get_line_for_game(
    year: int,
    home_team: str,
    away_team: str,
    cfbd_client=None,
) -> Optional[BettingLine]:
    """Get betting line for a specific game by team names.

    Args:
        year: Season year
        home_team: Home team name
        away_team: Away team name
        cfbd_client: Optional CFBD client instance

    Returns:
        BettingLine if found, None otherwise
    """
    lines = get_merged_lines(year, cfbd_client)

    # Search by team names (game_id might not match)
    for line in lines.values():
        if line.home_team == home_team and line.away_team == away_team:
            return line

    return None
