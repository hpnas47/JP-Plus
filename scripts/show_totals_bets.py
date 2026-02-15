#!/usr/bin/env python3
"""Fast totals bet display script - no API calls, pre-computed data.

Selection criteria: 5+ point edge (2023-2025 backtest: 55.5% ATS, +6.0% ROI)
EV shown for reference but not used for filtering.
"""

import math
import sys
from pathlib import Path

import pandas as pd


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


# Team abbreviations (shared with show_spread_bets.py)
ABBREV = {
    'Alabama': 'ALA', 'Georgia': 'UGA', 'Ohio State': 'OSU', 'Texas': 'TEX',
    'Clemson': 'CLEM', 'Notre Dame': 'ND', 'Michigan': 'MICH', 'USC': 'USC',
    'Oregon': 'ORE', 'Penn State': 'PSU', 'Florida': 'FLA', 'LSU': 'LSU',
    'Oklahoma': 'OKLA', 'Tennessee': 'TENN', 'Auburn': 'AUB', 'Miami': 'MIA',
    'Florida State': 'FSU', 'Wisconsin': 'WIS', 'Iowa': 'IOWA', 'Utah': 'UTAH',
    'UCLA': 'UCLA', 'Washington': 'WASH', 'Texas A&M': 'TAMU', 'Ole Miss': 'MISS',
    'Arkansas': 'ARK', 'Kentucky': 'UK', 'South Carolina': 'SCAR', 'Missouri': 'MIZ',
    'NC State': 'NCST', 'Pittsburgh': 'PITT', 'Louisville': 'LOU', 'Virginia Tech': 'VT',
    'Duke': 'DUKE', 'Wake Forest': 'WAKE', 'Virginia': 'UVA', 'Boston College': 'BC',
    'Syracuse': 'SYR', 'Georgia Tech': 'GT', 'North Carolina': 'UNC', 'Stanford': 'STAN',
    'California': 'CAL', 'Arizona': 'ARIZ', 'Arizona State': 'ASU', 'Colorado': 'COLO',
    'Baylor': 'BAY', 'TCU': 'TCU', 'Kansas': 'KU', 'Kansas State': 'KSU',
    'Iowa State': 'ISU', 'Oklahoma State': 'OKST', 'West Virginia': 'WVU', 'Texas Tech': 'TTU',
    'Cincinnati': 'CIN', 'UCF': 'UCF', 'Houston': 'HOU', 'BYU': 'BYU',
    'Memphis': 'MEM', 'SMU': 'SMU', 'Tulane': 'TUL', 'Tulsa': 'TLSA',
    'San Diego State': 'SDSU', 'Fresno State': 'FRES', 'Boise State': 'BSU', 'Air Force': 'AFA',
    'Army': 'ARMY', 'Navy': 'NAVY', 'Marshall': 'MRSH', 'Appalachian State': 'APP', 'App State': 'APP',
    'Oregon State': 'ORST',
    'Coastal Carolina': 'CCU', 'James Madison': 'JMU', 'Liberty': 'LIB', 'Sam Houston': 'SHSU',
    'Minnesota': 'MINN', 'Illinois': 'ILL', 'Northwestern': 'NW', 'Purdue': 'PUR',
    'Indiana': 'IND', 'Nebraska': 'NEB', 'Michigan State': 'MSU', 'Rutgers': 'RUT', 'Maryland': 'UMD',
    'Mississippi State': 'MSST', 'Vanderbilt': 'VAN', 'Louisiana': 'ULL', 'Troy': 'TROY',
    'South Alabama': 'USA', 'Georgia Southern': 'GASO', 'Georgia State': 'GAST', 'UTSA': 'UTSA',
    'North Texas': 'UNT', 'Rice': 'RICE', 'Florida Atlantic': 'FAU', 'Charlotte': 'CLT',
    'East Carolina': 'ECU', 'Temple': 'TEM', 'Buffalo': 'BUFF', 'Ohio': 'OHIO',
    'Miami (OH)': 'M-OH', 'Bowling Green': 'BGSU', 'Kent State': 'KENT', 'Akron': 'AKR',
    'Ball State': 'BALL', 'Toledo': 'TOL', 'Central Michigan': 'CMU', 'Eastern Michigan': 'EMU',
    'Western Michigan': 'WMU', 'Northern Illinois': 'NIU', 'Nevada': 'NEV', 'UNLV': 'UNLV',
    'Wyoming': 'WYO', 'New Mexico': 'UNM', 'Utah State': 'USU', 'Colorado State': 'CSU',
    "Hawai'i": 'HAW', 'San José State': 'SJSU', 'Louisiana Tech': 'LT', 'UAB': 'UAB',
    'Middle Tennessee': 'MTSU', 'Western Kentucky': 'WKU', 'Old Dominion': 'ODU',
    'Southern Miss': 'USM', 'FIU': 'FIU', 'New Mexico State': 'NMSU', 'South Florida': 'USF',
    'Kennesaw State': 'KENST', 'Jacksonville State': 'JVST', 'Connecticut': 'CONN', 'UMass': 'MASS',
    'Arkansas State': 'ARST', 'Louisiana-Monroe': 'ULM', 'Texas State': 'TXST', 'UL Monroe': 'ULM',
}


def get_abbrev(team: str) -> str:
    return ABBREV.get(team, team[:4].upper())


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
    print(f"\n## {year} Week {week} — Totals Primary (5+ Edge)\n")
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
