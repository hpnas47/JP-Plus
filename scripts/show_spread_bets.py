#!/usr/bin/env python3
"""Fast spread bet display script - uses cached FBS team data."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.api.cfbd_client import CFBDClient

# Cache FBS teams at module level (fetched once per session)
_fbs_teams_cache: dict[int, set[str]] = {}


def get_fbs_teams(year: int) -> set[str]:
    """Get FBS teams for a given year (cached)."""
    if year not in _fbs_teams_cache:
        client = CFBDClient()
        teams = client.get_fbs_teams(year=year)
        _fbs_teams_cache[year] = {t.school for t in teams}
    return _fbs_teams_cache[year]

# Team abbreviations
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
    """Get result vs OPEN line (matches Bet (Open) column)."""
    if row.get('ats_push_open', False) == True or row.get('ats_push_open', 0) == 1:
        return 'Push'
    elif row.get('ats_win_open', False) == True or row.get('ats_win_open', 0) == 1:
        return 'Win ✓'
    else:
        return 'Loss'


def format_date(row) -> str:
    d = row.get('start_date')
    if pd.isna(d) or not d:
        return '—'
    # "2022-09-03" -> "Sep 3"
    from datetime import datetime
    dt = datetime.strptime(str(d)[:10], '%Y-%m-%d')
    return dt.strftime('%b %-d')


def format_score(row) -> str:
    if pd.isna(row['home_points']) or pd.isna(row['away_points']):
        return '—'
    away_abbr = get_abbrev(row['away_team'])
    home_abbr = get_abbrev(row['home_team'])
    return f"{away_abbr} {int(row['away_points'])}, {home_abbr} {int(row['home_points'])}"


def format_jp_line(row) -> str:
    side = row['jp_favored_side']
    edge = row['edge_abs']
    spread = row['spread_open']
    if side == 'HOME':
        jp_spread = spread - edge
        return f"{row['home_team']} {jp_spread:+.1f}"
    else:
        jp_spread = -spread - edge
        return f"{row['away_team']} {jp_spread:+.1f}"


def format_bet_line(row) -> str:
    side = row['jp_favored_side']
    spread = row['spread_open']
    if side == 'HOME':
        return f"{row['home_team']} {spread:+.1f}"
    else:
        return f"{row['away_team']} {-spread:+.1f}"


def show_spread_bets(year: int, week: int):
    # Load data from appropriate source
    if year <= 2025:
        # Historical data (2022-2025)
        data_path = Path(__file__).parent.parent / 'data/spread_selection/outputs/backtest_primary_2022-2025_with_scores.csv'
        if not data_path.exists():
            print(f"Error: Historical data not found at {data_path}")
            return
        df = pd.read_csv(data_path)
        week_data = df[(df['year'] == year) & (df['week'] == week)]
    else:
        # Production data (2026+)
        data_path = Path(__file__).parent.parent / f'data/spread_selection/logs/spread_bets_{year}.csv'
        if not data_path.exists():
            print(f"No production data found for {year}. Run `python scripts/run_spread_weekly.py --year {year} --week {week}` first.")
            return
        df = pd.read_csv(data_path)
        week_data = df[df['week'] == week]

        # Map production columns to display columns
        if 'market_spread' in week_data.columns and 'spread_open' not in week_data.columns:
            week_data = week_data.copy()
            week_data['spread_open'] = week_data['market_spread']
        if 'edge_abs' not in week_data.columns and 'edge_pts' in week_data.columns:
            week_data = week_data.copy()
            week_data['edge_abs'] = week_data['edge_pts'].abs()
        if 'jp_favored_side' not in week_data.columns and 'side' in week_data.columns:
            week_data = week_data.copy()
            week_data['jp_favored_side'] = week_data['side'].str.upper()

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

    # Primary Engine: EV >= 3%
    primary = week_data[week_data['ev'] >= 0.03].sort_values('ev', ascending=False)

    # 5+ Edge Non-Primary
    edge5 = week_data[(week_data['edge_abs'] >= 5.0) & (week_data['ev'] < 0.03)].sort_values('edge_abs', ascending=False)

    # Print Primary EV Engine
    has_dates = 'start_date' in week_data.columns and week_data['start_date'].notna().any()

    print(f"\n## {year} Week {week} — Primary EV Engine\n")
    if has_dates:
        print("| # | Date | Matchup | JP+ Line | Bet (Open) | Edge | ~EV | Score | Result |")
        print("|---|------|---------|----------|------------|------|-----|-------|--------|")
    else:
        print("| # | Matchup | JP+ Line | Bet (Open) | Edge | ~EV | Score | Result |")
        print("|---|---------|----------|------------|------|-----|-------|--------|")

    for i, (_, row) in enumerate(primary.iterrows(), 1):
        away_abbr = get_abbrev(row['away_team'])
        home_abbr = get_abbrev(row['home_team'])
        matchup = f"{away_abbr} @ {home_abbr}"
        jp_line = format_jp_line(row)
        bet_line = format_bet_line(row)
        score = format_score(row)
        result = get_result(row)
        if has_dates:
            date = format_date(row)
            print(f"| {i} | {date} | {matchup} | {jp_line} | {bet_line} | {row['edge_abs']:.1f} | +{row['ev']*100:.1f}% | {score} | {result} |")
        else:
            print(f"| {i} | {matchup} | {jp_line} | {bet_line} | {row['edge_abs']:.1f} | +{row['ev']*100:.1f}% | {score} | {result} |")

    # Calculate record (vs OPEN line)
    p_wins = (primary['ats_win_open'] == 1).sum()
    p_pushes = (primary['ats_push_open'] == 1).sum()
    p_losses = len(primary) - p_wins - p_pushes

    if p_pushes > 0:
        print(f"\n**Record: {p_wins}-{p_losses}-{p_pushes} ({p_wins/(p_wins+p_losses)*100:.1f}%)**")
    else:
        pct = p_wins/(p_wins+p_losses)*100 if (p_wins+p_losses) > 0 else 0
        print(f"\n**Record: {p_wins}-{p_losses} ({pct:.1f}%)**")

    # Print 5+ Edge
    print(f"\n## 5+ Point Edge\n")
    if has_dates:
        print("| # | Date | Matchup | JP+ Line | Bet (Open) | Edge | Score | Result |")
        print("|---|------|---------|----------|------------|------|-------|--------|")
    else:
        print("| # | Matchup | JP+ Line | Bet (Open) | Edge | Score | Result |")
        print("|---|---------|----------|------------|------|-------|--------|")

    for i, (_, row) in enumerate(edge5.iterrows(), 1):
        away_abbr = get_abbrev(row['away_team'])
        home_abbr = get_abbrev(row['home_team'])
        matchup = f"{away_abbr} @ {home_abbr}"
        jp_line = format_jp_line(row)
        bet_line = format_bet_line(row)
        score = format_score(row)
        result = get_result(row)
        if has_dates:
            date = format_date(row)
            print(f"| {i} | {date} | {matchup} | {jp_line} | {bet_line} | {row['edge_abs']:.1f} | {score} | {result} |")
        else:
            print(f"| {i} | {matchup} | {jp_line} | {bet_line} | {row['edge_abs']:.1f} | {score} | {result} |")

    # Calculate record (vs OPEN line)
    e_wins = (edge5['ats_win_open'] == 1).sum()
    e_pushes = (edge5['ats_push_open'] == 1).sum()
    e_losses = len(edge5) - e_wins - e_pushes

    if e_pushes > 0:
        print(f"\n**Record: {e_wins}-{e_losses}-{e_pushes} ({e_wins/(e_wins+e_losses)*100:.1f}%)**")
    else:
        pct = e_wins/(e_wins+e_losses)*100 if (e_wins+e_losses) > 0 else 0
        print(f"\n**Record: {e_wins}-{e_losses} ({pct:.1f}%)**")

    # Footnotes
    print("\n---")
    print("*Primary EV Engine: Bets with EV >= 3% based on calibrated cover probability model.*")
    print("*5+ Edge: Games with 5+ point edge that didn't meet EV threshold.*")

    # Phase 1 warning (weeks 1-3)
    if week <= 3:
        print("")
        print("**⚠️ Phase 1 Warning:** EV calibration is less reliable in weeks 1-3 (44% ATS vs 55% in Core).")
        print("*Prior-driven predictions have higher variance. Consider half-stakes or 5+ Edge filter.*")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python show_spread_bets.py <year> <week>")
        sys.exit(1)

    year = int(sys.argv[1])
    week = int(sys.argv[2])
    show_spread_bets(year, week)
