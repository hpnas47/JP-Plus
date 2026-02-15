#!/usr/bin/env python3
"""Moneyline bet display script - shows ML EV recommendations from logged data.

Data sources:
- Historical (backtest): data/moneyline_selection/logs/moneyline_bets_{year}.csv
- Production (2026+):    data/moneyline_selection/logs/moneyline_bets_{year}.csv

Both use the same log format from run_moneyline_weekly.py.
"""

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


# Team abbreviations (shared across show_*_bets scripts)
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


def format_odds(odds) -> str:
    """Format American odds with sign."""
    odds = int(odds)
    return f"{odds:+d}"


def format_bet_team(row) -> str:
    """Return the team name for the bet side."""
    if row['side'] == 'HOME':
        return row['home_team']
    return row['away_team']


def format_result(row) -> str:
    """Format settlement result."""
    if pd.isna(row.get('covered')):
        return '—'
    if row['covered'] == 'W':
        return 'Win ✓'
    if row['covered'] == 'L':
        return 'Loss'
    return '—'


def format_profit(row) -> str:
    """Format profit/loss."""
    if pd.isna(row.get('profit_units')):
        return '—'
    p = row['profit_units']
    return f"{p:+.0f}" if p != 0 else "0"


# Confidence tiers based on Kelly fraction (composite of edge + odds quality).
# Tercile-based from 2022-2025 backtest distribution (N=1,794).
KELLY_HIGH = 0.06    # top ~34%
KELLY_MEDIUM = 0.025  # middle ~33%


def format_confidence(row) -> str:
    """Map kelly_f to High/Medium/Low confidence label."""
    kf = row.get('kelly_f')
    if pd.isna(kf):
        return '—'
    if kf >= KELLY_HIGH:
        return 'HIGH'
    if kf >= KELLY_MEDIUM:
        return 'MED'
    return 'LOW'


def show_moneyline_bets(year: int, week: int):
    # Load from log file (same path for historical and production)
    log_path = Path(__file__).parent.parent / f'data/moneyline_selection/logs/moneyline_bets_{year}.csv'
    if not log_path.exists():
        print(f"No moneyline data found for {year}.")
        print(f"Run: python3 scripts/run_moneyline_weekly.py --year {year} --week {week} --inputs-path <path>")
        return

    df = pd.read_csv(log_path)
    week_data = df[(df['year'] == year) & (df['week'] == week)]

    if len(week_data) == 0:
        print(f"No moneyline data found for {year} Week {week}")
        return

    # Filter to FBS vs FBS games only
    fbs_teams = get_fbs_teams(year)
    fbs_mask = week_data['home_team'].isin(fbs_teams) & week_data['away_team'].isin(fbs_teams)
    week_data = week_data[fbs_mask]

    if len(week_data) == 0:
        print(f"No FBS vs FBS moneyline games found for {year} Week {week}")
        return

    # Split into List A and List B
    list_a = week_data[week_data['list_type'] == 'A'].sort_values('ev', ascending=False)
    list_b = week_data[week_data['list_type'] == 'B'].sort_values(
        ['reason_code', 'ev'], ascending=[True, False]
    )

    # --- Config summary ---
    sigma = week_data['margin_sigma'].iloc[0] if 'margin_sigma' in week_data.columns else '?'
    ev_min = week_data['ev_min'].iloc[0] if 'ev_min' in week_data.columns else '?'
    min_disagree = week_data['min_disagreement_pts'].iloc[0] if 'min_disagreement_pts' in week_data.columns else '?'

    # --- List A: Actionable ML Bets ---
    print(f"\n## {year} Week {week} — Moneyline Bets (List A)\n")

    if list_a.empty:
        print("*No actionable moneyline bets this week.*\n")
    else:
        has_results = 'covered' in list_a.columns and list_a['covered'].notna().any()

        if has_results:
            print("| # | Matchup | Bet | Odds | p(Win) | EV | Disagree | Flip | Conf | Result |")
            print("|---|---------|-----|------|--------|-----|----------|------|------|--------|")
        else:
            print("| # | Matchup | Bet | Odds | p(Win) | EV | Disagree | Flip | Conf |")
            print("|---|---------|-----|------|--------|-----|----------|------|------|")

        for i, (_, row) in enumerate(list_a.iterrows(), 1):
            away_abbr = get_abbrev(row['away_team'])
            home_abbr = get_abbrev(row['home_team'])
            matchup = f"{away_abbr} @ {home_abbr}"
            bet_team = format_bet_team(row)
            odds = format_odds(row['odds_american'])
            p_win = f"{row['p_win']:.1%}"
            ev = f"+{row['ev']*100:.1f}%"
            disagree = f"{row['disagreement_pts']:.1f}"
            flip = "FLIP" if row['flip_flag'] else ""
            conf = format_confidence(row)

            if has_results:
                result = format_result(row)
                print(f"| {i} | {matchup} | {bet_team} ML | {odds} | {p_win} | {ev} | {disagree} | {flip} | {conf} | {result} |")
            else:
                print(f"| {i} | {matchup} | {bet_team} ML | {odds} | {p_win} | {ev} | {disagree} | {flip} | {conf} |")

        # Record
        if has_results:
            settled = list_a[list_a['covered'].notna()]
            wins = (settled['covered'] == 'W').sum()
            losses = (settled['covered'] == 'L').sum()
            pct = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            # Compute $100 flat-bet profit
            profit = 0.0
            for _, r in settled.iterrows():
                if r['covered'] == 'W':
                    odds = int(r['odds_american'])
                    if odds < 0:
                        profit += 100.0 * (100.0 / abs(odds))
                    else:
                        profit += 100.0 * (odds / 100.0)
                elif r['covered'] == 'L':
                    profit -= 100.0
            print(f"\n**Record: {wins}-{losses} ({pct:.1f}%) | $100 Flat P/L: {profit:+,.0f}**")
        else:
            print(f"\n**{len(list_a)} bet(s) | Avg EV: +{list_a['ev'].mean()*100:.1f}%**")

    # --- List B: Near-Misses ---
    print(f"\n## List B — Near-Misses / Diagnostics\n")

    if list_b.empty:
        print("*No near-miss games this week.*\n")
    else:
        print("| # | Matchup | Side | Odds | EV | Disagree | Flip | Reason |")
        print("|---|---------|------|------|----|----------|------|--------|")

        for i, (_, row) in enumerate(list_b.iterrows(), 1):
            away_abbr = get_abbrev(row['away_team'])
            home_abbr = get_abbrev(row['home_team'])
            matchup = f"{away_abbr} @ {home_abbr}"

            if pd.notna(row.get('side')):
                side_team = format_bet_team(row)
                side = f"{side_team} ML"
            else:
                side = "—"

            odds = format_odds(row['odds_american']) if pd.notna(row.get('odds_american')) else "—"
            ev = f"{row['ev']*100:+.1f}%" if pd.notna(row.get('ev')) else "—"
            disagree = f"{row['disagreement_pts']:.1f}" if pd.notna(row.get('disagreement_pts')) else "—"
            flip = "FLIP" if row.get('flip_flag') else ""
            reason = row['reason_code']
            print(f"| {i} | {matchup} | {side} | {odds} | {ev} | {disagree} | {flip} | {reason} |")

        # Reason code summary
        reasons = list_b['reason_code'].value_counts().to_dict()
        reason_parts = [f"{rc}: {ct}" for rc, ct in reasons.items()]
        print(f"\n*Reasons: {", ".join(reason_parts)}*")

    # Footnotes
    print("\n---")
    print(f"*Config: sigma={sigma}, ev_min={ev_min}, min_disagree={min_disagree}*")
    print("*ML EV = p(win) × (decimal_odds - 1) - p(loss). Sizing via quarter-Kelly.*")

    # Phase 1 warning
    if week <= 3:
        print("")
        print("**Phase 1 Warning:** Win probabilities are prior-driven in weeks 1-3. Consider half-stakes.")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python show_moneyline_bets.py <year> <week>")
        sys.exit(1)

    year = int(sys.argv[1])
    week = int(sys.argv[2])
    show_moneyline_bets(year, week)
