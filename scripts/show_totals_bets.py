#!/usr/bin/env python3
"""Fast totals bet display script - no API calls, pre-computed data."""

import sys
from pathlib import Path

import pandas as pd

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


def get_result(row) -> str:
    result = row['result']
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
    # Load pre-computed data
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

    # Calculate EV estimate based on edge
    # At sigma=13, 5pt edge ≈ 65% win prob ≈ 24% EV
    # Roughly 4.8% EV per point of edge
    week_data['ev_est'] = week_data['edge_open'].abs() * 0.048

    # Primary: 5+ point edge (55.3% ATS historically)
    primary = week_data[week_data['edge_open'].abs() >= 5.0].sort_values('edge_open', key=abs, ascending=False)

    # Secondary: 3-5 point edge (54.5% ATS historically)
    edge3to5 = week_data[
        (week_data['edge_open'].abs() >= 3.0) &
        (week_data['edge_open'].abs() < 5.0)
    ].sort_values('edge_open', key=abs, ascending=False)

    # Print 5+ Edge (Primary)
    print(f"\n## {year} Week {week} — Totals 5+ Edge\n")
    print("| # | Matchup | JP+ Total | Vegas (Open) | Side | Edge | ~EV | Final | Result |")
    print("|---|---------|-----------|--------------|------|------|-----|-------|--------|")

    for i, (_, row) in enumerate(primary.iterrows(), 1):
        matchup = f"{row['away_team']} @ {row['home_team']}"
        jp_total = f"{row['adjusted_total']:.1f}"
        vegas_total = f"{row['vegas_total_open']:.1f}"
        side = row['play']
        edge = f"+{abs(row['edge_open']):.1f}"
        ev = f"+{row['ev_est']*100:.1f}%"
        final = format_score(row)
        result = get_result(row)
        print(f"| {i} | {matchup} | {jp_total} | {vegas_total} | {side} | {edge} | {ev} | {final} | {result} |")

    # Calculate record
    p_wins = len(primary[primary['result'] == 'WIN'])
    p_losses = len(primary[primary['result'] == 'LOSS'])
    p_pushes = len(primary[primary['result'] == 'PUSH'])

    if p_wins + p_losses > 0:
        if p_pushes > 0:
            print(f"\n**Record: {p_wins}-{p_losses}-{p_pushes} ({p_wins/(p_wins+p_losses)*100:.1f}%)**")
        else:
            print(f"\n**Record: {p_wins}-{p_losses} ({p_wins/(p_wins+p_losses)*100:.1f}%)**")
    else:
        print("\n**Record: No qualifying bets**")

    # Print 3-5 Edge (secondary list)
    if len(edge3to5) > 0:
        print(f"\n## 3-5 Point Edge (Secondary)\n")
        print("| # | Matchup | JP+ Total | Vegas (Open) | Side | Edge | Final | Result |")
        print("|---|---------|-----------|--------------|------|------|-------|--------|")

        for i, (_, row) in enumerate(edge3to5.iterrows(), 1):
            matchup = f"{row['away_team']} @ {row['home_team']}"
            jp_total = f"{row['adjusted_total']:.1f}"
            vegas_total = f"{row['vegas_total_open']:.1f}"
            side = row['play']
            edge = f"+{abs(row['edge_open']):.1f}"
            final = format_score(row)
            result = get_result(row)
            print(f"| {i} | {matchup} | {jp_total} | {vegas_total} | {side} | {edge} | {final} | {result} |")

        # Calculate record
        e_wins = len(edge3to5[edge3to5['result'] == 'WIN'])
        e_losses = len(edge3to5[edge3to5['result'] == 'LOSS'])
        e_pushes = len(edge3to5[edge3to5['result'] == 'PUSH'])

        if e_wins + e_losses > 0:
            if e_pushes > 0:
                print(f"\n**Record: {e_wins}-{e_losses}-{e_pushes} ({e_wins/(e_wins+e_losses)*100:.1f}%)**")
            else:
                print(f"\n**Record: {e_wins}-{e_losses} ({e_wins/(e_wins+e_losses)*100:.1f}%)**")

    # Footnotes
    print("\n---")
    print("*5+ Edge: High conviction bets (55.3% ATS historical).*")
    print("*3-5 Edge: Secondary opportunities (54.5% ATS historical).*")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python show_totals_bets.py <year> <week>")
        sys.exit(1)

    year = int(sys.argv[1])
    week = int(sys.argv[2])
    show_totals_bets(year, week)
