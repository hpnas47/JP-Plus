#!/usr/bin/env python3
"""Totals bet display script - uses cached FBS team data.

Selection criteria: 5+ point edge (2023-2025 backtest: 55.5% ATS, +6.0% ROI)
EV shown for reference but not used for filtering.
"""

import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.utils.display_helpers import get_abbrev, get_fbs_teams


# Normal CDF for EV calculation (matches totals_ev_engine.py)
def normal_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def calculate_totals_ev(edge: float, sigma: float = 16.4, odds: int = -110) -> tuple[float, float, float]:
    """Calculate EV for a totals bet using Normal CDF model.

    Args:
        edge: Signed edge (positive = OVER, negative = UNDER)
        sigma: Standard deviation of prediction errors (default 16.4 from 2023-2025)
        odds: American odds (default -110)

    Returns:
        (ev, p_win, p_push) tuple
    """
    z = abs(edge) / sigma
    p_push = normal_cdf(0.5 / sigma) - normal_cdf(-0.5 / sigma)
    p_win = normal_cdf(z) - p_push / 2
    p_lose = 1.0 - p_win - p_push

    if odds > 0:
        payout = odds / 100.0
    else:
        payout = 100.0 / abs(odds)

    ev = p_win * payout - p_lose
    return ev, p_win, p_push


def calc_play_vs_open(row) -> str:
    """Determine OVER/UNDER based on edge vs OPEN line."""
    if row['edge_open'] > 0:
        return 'OVER'
    elif row['edge_open'] < 0:
        return 'UNDER'
    return 'PASS'


def calc_result_vs_open(row) -> str:
    """Calculate WIN/LOSS/PUSH based on OPEN line (not CLOSE)."""
    if pd.isna(row['vegas_total_open']) or pd.isna(row['actual_total']):
        return 'NO_LINE'

    vegas = row['vegas_total_open']
    actual = row['actual_total']

    if actual == vegas:
        return 'PUSH'

    # JP+ says OVER if edge_open > 0, UNDER if edge_open < 0
    if row['edge_open'] > 0:  # JP+ says OVER
        return 'WIN' if actual > vegas else 'LOSS'
    elif row['edge_open'] < 0:  # JP+ says UNDER
        return 'WIN' if actual < vegas else 'LOSS'
    return 'PASS'


def format_result(result: str) -> str:
    """Format result for display."""
    if result == 'PUSH':
        return 'Push'
    elif result == 'WIN':
        return 'Win ✓'
    elif result == 'LOSS':
        return 'Loss'
    return '—'


def format_date(row) -> str:
    d = row.get('start_date')
    if pd.isna(d) or not d:
        return '—'
    from datetime import datetime
    dt = datetime.strptime(str(d)[:10], '%Y-%m-%d')
    return dt.strftime('%b %-d')


def format_score(row) -> str:
    if pd.isna(row['actual_total']):
        return '—'
    return f"{int(row['actual_total'])}"


def show_totals_bets(year: int, week: int):
    # Load pre-computed data (2023-2025 only, excludes 2022 transition year)
    data_path = Path(__file__).parent.parent / 'data/spread_selection/outputs/backtest_totals_2023-2025.csv'
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        print("Run: python3 scripts/backtest_totals.py --output data/spread_selection/outputs/backtest_totals_2023-2025.csv")
        return

    df = pd.read_csv(data_path)
    week_data = df[(df['year'] == year) & (df['week'] == week)]

    if len(week_data) == 0:
        print(f"No data found for {year} Week {week}")
        return

    # Filter to FBS vs FBS games only
    fbs_teams = get_fbs_teams(year)
    fbs_mask = week_data['home_team'].isin(fbs_teams) & week_data['away_team'].isin(fbs_teams)
    week_data = week_data[fbs_mask]

    if len(week_data) == 0:
        print(f"No FBS vs FBS games found for {year} Week {week}")
        return

    # Filter to games with lines
    week_data = week_data[week_data['vegas_total_open'].notna()].copy()

    # Calculate calibrated sigma from actual prediction errors
    SIGMA = df['jp_error'].std()  # ~16.4 from 2023-2025 backtest

    # Calculate EV using Normal CDF model (for display, not filtering)
    ev_data = week_data['edge_open'].apply(lambda e: calculate_totals_ev(e, SIGMA))
    week_data['ev'] = ev_data.apply(lambda x: x[0])
    week_data['p_win'] = ev_data.apply(lambda x: x[1])

    # Recalculate play and result vs OPEN line (CSV uses CLOSE, we bet on OPEN)
    week_data['play_open'] = week_data.apply(calc_play_vs_open, axis=1)
    week_data['result_open'] = week_data.apply(calc_result_vs_open, axis=1)

    # Primary: 5+ point edge (2023-2025 backtest: 55.5% ATS, +6.0% ROI)
    EDGE_MIN = 5.0
    primary = week_data[week_data['edge_open'].abs() >= EDGE_MIN].sort_values(
        'edge_open', key=abs, ascending=False
    )

    # Print Primary 5+ Edge
    has_dates = 'start_date' in week_data.columns and week_data['start_date'].notna().any()

    print(f"\n## {year} Week {week} — Totals Primary (5+ Edge)\n")
    if has_dates:
        print("| # | Date | Matchup | JP+ Total | Vegas (Open) | Side | Edge | ~EV | Final | Result |")
        print("|---|------|---------|-----------|--------------|------|------|-----|-------|--------|")
    else:
        print("| # | Matchup | JP+ Total | Vegas (Open) | Side | Edge | ~EV | Final | Result |")
        print("|---|---------|-----------|--------------|------|------|-----|-------|--------|")

    for i, (_, row) in enumerate(primary.iterrows(), 1):
        matchup = f"{row['away_team']} @ {row['home_team']}"
        jp_total = f"{row['adjusted_total']:.1f}"
        vegas_total = f"{row['vegas_total_open']:.1f}"
        side = row['play_open']
        edge = f"+{abs(row['edge_open']):.1f}"
        ev = f"+{row['ev']*100:.1f}%"
        final = format_score(row)
        result = format_result(row['result_open'])
        if has_dates:
            date = format_date(row)
            print(f"| {i} | {date} | {matchup} | {jp_total} | {vegas_total} | {side} | {edge} | {ev} | {final} | {result} |")
        else:
            print(f"| {i} | {matchup} | {jp_total} | {vegas_total} | {side} | {edge} | {ev} | {final} | {result} |")

    # Calculate record (using OPEN-based results)
    p_wins = len(primary[primary['result_open'] == 'WIN'])
    p_losses = len(primary[primary['result_open'] == 'LOSS'])
    p_pushes = len(primary[primary['result_open'] == 'PUSH'])

    if p_wins + p_losses > 0:
        if p_pushes > 0:
            print(f"\n**Record: {p_wins}-{p_losses}-{p_pushes} ({p_wins/(p_wins+p_losses)*100:.1f}%)**")
        else:
            print(f"\n**Record: {p_wins}-{p_losses} ({p_wins/(p_wins+p_losses)*100:.1f}%)**")
    else:
        print("\n**Record: No qualifying bets**")

    # Footnotes
    print("\n---")
    print(f"*Selection: 5+ point edge vs opening line. EV estimates use Normal CDF (sigma={SIGMA:.1f}).*")
    print("*2023-2025 backtest: 55.5% ATS, +6.0% ROI at 5+ edge (~15 games/week).*")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python show_totals_bets.py <year> <week>")
        sys.exit(1)

    year = int(sys.argv[1])
    week = int(sys.argv[2])
    show_totals_bets(year, week)
