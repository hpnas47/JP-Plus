#!/usr/bin/env python3
"""
Weekly totals (O/U) prediction runner for CFB Power Ratings Model.

Primary use case: Run Sunday AM after Saturday games to find totals value.

Usage:
    python scripts/run_weekly_totals.py --year 2025 --week 6
    python scripts/run_weekly_totals.py --year 2025 --week 6 --export totals_week6.csv
    python scripts/run_weekly_totals.py --year 2025 --week 6 --min-edge 7
    python scripts/run_weekly_totals.py --year 2025 --week 6 --primary-only

Output Lists:
    - Primary EV Engine: Bets meeting EV threshold (default 2%)
    - 5+ Edge: High-edge bets that don't meet EV cut (diagnostic)

Display Format:
    | # | Matchup | JP+ Total | Bet (Line) | Edge | EV | Result |
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.api.cfbd_client import CFBDClient
from src.models.totals_model import TotalsModel
from src.spread_selection.totals_ev_engine import (
    TotalsEvent,
    TotalMarket,
    TotalsEVConfig,
    evaluate_totals_markets,
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CFB totals predictions for a given week",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard weekly run
    python scripts/run_weekly_totals.py --year 2025 --week 6

    # Export to CSV
    python scripts/run_weekly_totals.py --year 2025 --week 6 --export week6_totals.csv

    # High-conviction only (7+ edge)
    python scripts/run_weekly_totals.py --year 2025 --week 6 --min-edge 7

    # Show only Primary EV list (skip 5+ Edge diagnostic)
    python scripts/run_weekly_totals.py --year 2025 --week 6 --primary-only

    # Show all games (no filters)
    python scripts/run_weekly_totals.py --year 2025 --week 6 --show-all
"""
    )
    parser.add_argument(
        "--year", type=int, required=True,
        help="Season year"
    )
    parser.add_argument(
        "--week", type=int, required=True,
        help="Week number to predict"
    )
    parser.add_argument(
        "--min-edge", type=float, default=3.0,
        help="Minimum edge (pts) to display. Default: 3.0"
    )
    parser.add_argument(
        "--primary-only", action="store_true",
        help="Only show Primary EV list (skip 5+ Edge diagnostic list)"
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Show all evaluated games, not just those meeting thresholds"
    )
    parser.add_argument(
        "--export", type=str, default=None, metavar="FILE",
        help="Export results to CSV file"
    )
    parser.add_argument(
        "--ev-min", type=float, default=0.02,
        help="Minimum EV for Primary Engine qualification. Default: 0.02 (2%%)"
    )
    parser.add_argument(
        "--sigma", type=float, default=13.0,
        help="Standard deviation for probability model. Default: 13.0"
    )
    parser.add_argument(
        "--bankroll", type=float, default=1000.0,
        help="Bankroll for Kelly staking. Default: 1000"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output"
    )

    return parser.parse_args()


def fetch_games_and_lines(client: CFBDClient, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch games and betting lines for a season.

    Returns (games_df, betting_df).
    """
    # Get games
    games = client.get_games(year=year, season_type='regular')
    games_data = [
        {
            'id': g.id,
            'week': g.week,
            'home_team': g.home_team,
            'away_team': g.away_team,
            'home_points': g.home_points,
            'away_points': g.away_points,
        }
        for g in games
    ]
    games_df = pd.DataFrame(games_data)

    # Get betting lines
    betting_data = []
    try:
        lines = client.get_betting_lines(year=year, season_type='regular')
        for line in lines:
            if hasattr(line, 'lines') and line.lines:
                for book_line in line.lines:
                    if hasattr(book_line, 'over_under') and book_line.over_under:
                        betting_data.append({
                            'game_id': line.id,
                            'over_under': book_line.over_under,
                            'provider': getattr(book_line, 'provider', 'unknown'),
                        })
    except Exception as e:
        logger.warning(f"Could not fetch betting lines: {e}")

    betting_df = pd.DataFrame(betting_data) if betting_data else pd.DataFrame()

    return games_df, betting_df


def build_events(
    games_df: pd.DataFrame,
    betting_df: pd.DataFrame,
    week: int,
    year: int,
    fbs_set: set[str],
) -> list[TotalsEvent]:
    """Build TotalsEvent objects for a given week."""
    week_games = games_df[games_df['week'] == week].copy()

    # FBS filter
    week_games = week_games[
        (week_games['home_team'].isin(fbs_set)) &
        (week_games['away_team'].isin(fbs_set))
    ]

    if betting_df.empty:
        return []

    # Merge betting lines (take first line per game)
    lines_df = betting_df[['game_id', 'over_under']].drop_duplicates(subset=['game_id'])
    lines_df = lines_df.rename(columns={'over_under': 'total_line', 'game_id': 'id'})
    week_games = week_games.merge(lines_df, on='id', how='left')

    events = []
    for g in week_games.itertuples():
        line = getattr(g, 'total_line', None)
        if pd.isna(line):
            continue

        events.append(TotalsEvent(
            event_id=str(g.id),
            home_team=g.home_team,
            away_team=g.away_team,
            year=year,
            week=week,
            weather_adjustment=0.0,
            markets=[
                TotalMarket(
                    book="CFBD",
                    line=float(line),
                    odds_over=-110,
                    odds_under=-110,
                )
            ],
        ))

    return events


def format_matchup(away_team: str, home_team: str) -> str:
    """Format matchup as 'Away @ Home'."""
    return f"{away_team} @ {home_team}"


def format_bet(side: str, line: float) -> str:
    """Format bet recommendation."""
    return f"{side} {line}"


def print_recommendations(
    df: pd.DataFrame,
    title: str,
    show_ev: bool = True,
    min_edge: float = 0.0,
    results_df: Optional[pd.DataFrame] = None,
) -> int:
    """Print formatted bet recommendations with vertical separators.

    Format: # Matchup | JP+ Total | Edge | ~EV | Bet (Open) | Result

    Args:
        df: DataFrame with recommendations
        title: Section title
        show_ev: Whether to show EV column
        min_edge: Minimum edge to display
        results_df: Optional DataFrame with game results (must have 'id', 'home_points', 'away_points')

    Returns count of displayed bets.
    """
    if df.empty:
        print(f"\n{title}: No recommendations")
        return 0

    total_count = len(df)

    # Filter by min edge
    if min_edge > 0:
        df = df[df['edge_pts'].abs() >= min_edge].copy()

    if df.empty:
        print(f"\n{title}: No recommendations meeting {min_edge}+ edge threshold")
        return 0

    # Sort by EV descending
    df = df.sort_values('ev', ascending=False).reset_index(drop=True)

    # Check if we have results
    show_results = results_df is not None and not results_df.empty

    # Build results lookup if available
    results_lookup = {}
    if show_results:
        for _, row in results_df.iterrows():
            game_id = str(row['id'])
            home_pts = row.get('home_points')
            away_pts = row.get('away_points')
            if pd.notna(home_pts) and pd.notna(away_pts):
                results_lookup[game_id] = int(home_pts) + int(away_pts)

    # Calculate column widths
    matchups = [format_matchup(r['away_team'], r['home_team']) for _, r in df.iterrows()]
    max_matchup = max(len(m) for m in matchups)

    # Column widths
    w_num = 3
    w_matchup = max(max_matchup, 25)
    w_jp_total = 9
    w_vegas = 12
    w_edge = 6
    w_ev = 7
    w_bet = 12
    w_result = 8

    filtered_note = f" (showing {len(df)} of {total_count} with {min_edge}+ edge)" if min_edge > 0 and len(df) < total_count else ""
    print(f"\n{'=' * 115}")
    print(f"{title} ({len(df)} bets{filtered_note})")
    print('=' * 115)

    # Header
    if show_ev:
        if show_results:
            header = f"{'#':<{w_num}} {'Matchup':<{w_matchup}} | {'JP+ Total':^{w_jp_total}} | {'Vegas (Open)':^{w_vegas}} | {'Edge':^{w_edge}} | {'~EV':^{w_ev}} | {'Bet (Open)':^{w_bet}} | {'Result':^{w_result}}"
        else:
            header = f"{'#':<{w_num}} {'Matchup':<{w_matchup}} | {'JP+ Total':^{w_jp_total}} | {'Vegas (Open)':^{w_vegas}} | {'Edge':^{w_edge}} | {'~EV':^{w_ev}} | {'Bet (Open)':^{w_bet}}"
    else:
        header = f"{'#':<{w_num}} {'Matchup':<{w_matchup}} | {'JP+ Total':^{w_jp_total}} | {'Vegas (Open)':^{w_vegas}} | {'Edge':^{w_edge}} | {'Bet (Open)':^{w_bet}}"

    print(header)
    print('-' * len(header))

    # Rows
    for i, (_, r) in enumerate(df.iterrows()):
        matchup = format_matchup(r['away_team'], r['home_team'])
        jp_total = f"{r['mu_used']:.1f}"
        vegas_total = f"{r['line']:.1f}"
        edge = f"+{abs(r['edge_pts']):.1f}"
        ev = f"+{r['ev']*100:.1f}%" if r['ev'] > 0 else f"{r['ev']*100:.1f}%"
        bet = f"{r['side']} {r['line']}"

        # Result column
        result = ""
        if show_results:
            event_id = str(r['event_id'])
            if event_id in results_lookup:
                actual_total = results_lookup[event_id]
                line = r['line']
                side = r['side']

                if actual_total > line:
                    result = "Win ✓" if side == "OVER" else "Loss "
                elif actual_total < line:
                    result = "Win ✓" if side == "UNDER" else "Loss "
                else:
                    result = "Push "

        if show_ev:
            if show_results:
                row_str = f"{i+1:<{w_num}} {matchup:<{w_matchup}} | {jp_total:^{w_jp_total}} | {vegas_total:^{w_vegas}} | {edge:^{w_edge}} | {ev:^{w_ev}} | {bet:^{w_bet}} | {result:^{w_result}}"
            else:
                row_str = f"{i+1:<{w_num}} {matchup:<{w_matchup}} | {jp_total:^{w_jp_total}} | {vegas_total:^{w_vegas}} | {edge:^{w_edge}} | {ev:^{w_ev}} | {bet:^{w_bet}}"
        else:
            row_str = f"{i+1:<{w_num}} {matchup:<{w_matchup}} | {jp_total:^{w_jp_total}} | {vegas_total:^{w_vegas}} | {edge:^{w_edge}} | {bet:^{w_bet}}"

        print(row_str)

    return len(df)


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    year = args.year
    week = args.week

    print("=" * 80)
    print(f"JP+ TOTALS ENGINE - {year} Week {week}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Initialize client
    client = CFBDClient()

    # Get FBS teams
    print("\n[1/4] Fetching FBS teams...")
    fbs_teams = client.get_fbs_teams(year=year)
    fbs_set = {t.school for t in fbs_teams}
    print(f"      {len(fbs_set)} FBS teams")

    # Fetch data
    print("\n[2/4] Fetching games and betting lines...")
    games_df, betting_df = fetch_games_and_lines(client, year)
    print(f"      {len(games_df)} games, {len(betting_df)} betting lines")

    # Train model
    print(f"\n[3/4] Training TotalsModel through week {week - 1}...")
    train_games = games_df[games_df['week'] < week].copy()
    train_games = train_games[
        (train_games['home_team'].isin(fbs_set)) &
        (train_games['away_team'].isin(fbs_set))
    ]

    # Add actual_total for training
    train_games['actual_total'] = train_games['home_points'] + train_games['away_points']
    train_games = train_games.dropna(subset=['actual_total'])

    print(f"      Training on {len(train_games)} FBS games from weeks 1-{week - 1}")

    model = TotalsModel(ridge_alpha=10.0)
    model.set_team_universe(fbs_set)
    model.train(train_games, fbs_set, max_week=week - 1)

    if not model._trained:
        print("      ERROR: Model failed to train")
        sys.exit(1)

    print("      Model trained successfully")

    # Build events
    print(f"\n[4/4] Building predictions for week {week}...")
    events = build_events(games_df, betting_df, week, year, fbs_set)
    print(f"      {len(events)} games with betting lines")

    if not events:
        print("\nNo games found with betting lines for this week.")
        sys.exit(0)

    # Configure EV engine
    config = TotalsEVConfig(
        sigma_total=args.sigma,
        bankroll=args.bankroll,
        use_weather_adjustment=False,
        ev_min=args.ev_min,
        edge_pts_min=5.0,
    )

    # Evaluate
    primary_df, edge5_df = evaluate_totals_markets(
        model, events, config, n_train_games=len(train_games)
    )

    # Get results for this week (for historical data)
    week_results = games_df[games_df['week'] == week].copy()
    has_results = week_results['home_points'].notna().any()

    # Print results
    displayed_primary = 0
    displayed_edge5 = 0

    if args.show_all:
        # Combine and show all
        all_df = pd.concat([primary_df, edge5_df], ignore_index=True)
        all_df = all_df.drop_duplicates(subset=['event_id', 'side'])
        print_recommendations(
            all_df, "All Evaluated Games", show_ev=True, min_edge=0,
            results_df=week_results if has_results else None
        )
    else:
        # Primary EV Engine
        displayed_primary = print_recommendations(
            primary_df,
            "PRIMARY EV ENGINE",
            show_ev=True,
            min_edge=args.min_edge,
            results_df=week_results if has_results else None
        )

        # 5+ Edge (diagnostic)
        if not args.primary_only:
            displayed_edge5 = print_recommendations(
                edge5_df,
                "5+ EDGE (Below EV Cut)",
                show_ev=False,
                min_edge=max(args.min_edge, 5.0),
                results_df=week_results if has_results else None
            )

    # Summary stats
    print("\n" + "-" * 100)
    print("Summary")
    print("-" * 100)
    print(f"  Games evaluated: {len(events)}")
    print(f"  Total qualifying bets: {len(primary_df)} Primary, {len(edge5_df)} 5+ Edge")
    if args.min_edge > 0:
        print(f"  Displayed ({args.min_edge}+ edge): {displayed_primary} Primary, {displayed_edge5} 5+ Edge")
    print(f"  Config: sigma={args.sigma}, ev_min={args.ev_min * 100:.0f}%, bankroll=${args.bankroll:.0f}")

    # Record summary for historical data
    if has_results and len(primary_df) > 0:
        # Build results lookup from week_results
        results_lookup = {}
        for _, row in week_results.iterrows():
            game_id = str(row['id'])
            home_pts = row.get('home_points')
            away_pts = row.get('away_points')
            if pd.notna(home_pts) and pd.notna(away_pts):
                results_lookup[game_id] = int(home_pts) + int(away_pts)

        wins = 0
        losses = 0
        pushes = 0
        for _, r in primary_df.iterrows():
            event_id = str(r['event_id'])
            if event_id in results_lookup:
                actual_total = results_lookup[event_id]
                line = r['line']
                side = r['side']
                if actual_total > line:
                    if side == "OVER":
                        wins += 1
                    else:
                        losses += 1
                elif actual_total < line:
                    if side == "UNDER":
                        wins += 1
                    else:
                        losses += 1
                else:
                    pushes += 1

        total = wins + losses
        if total > 0:
            pct = wins / total * 100
            push_str = f", {pushes} Push" if pushes > 0 else ""
            print(f"\n  RECORD: {wins}-{losses}{push_str} ({pct:.1f}%)")

    # Export if requested
    if args.export:
        export_path = Path(args.export)
        all_df = pd.concat([primary_df, edge5_df], ignore_index=True)
        all_df['list'] = ['Primary'] * len(primary_df) + ['5+ Edge'] * len(edge5_df)
        all_df.to_csv(export_path, index=False)
        print(f"\n  Exported {len(all_df)} recommendations to: {export_path}")

    print()


if __name__ == "__main__":
    main()
