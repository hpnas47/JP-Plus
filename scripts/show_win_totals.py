#!/usr/bin/env python3
"""Fast win totals display script - reads from cached predictions CSV.

Usage:
    python3 scripts/show_win_totals.py 2025           # Top 25
    python3 scripts/show_win_totals.py 2025 50        # Top 50
    python3 scripts/show_win_totals.py 2025 all       # All teams
    python3 scripts/show_win_totals.py 2025 all SEC   # SEC only
    python3 scripts/show_win_totals.py 2025 all "Big Ten"  # Big Ten only

If predictions CSV doesn't exist, tells user to run predict first.
For completed seasons (year < current year), shows actual wins and result.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.teams import normalize_team_name
from src.api.cfbd_client import CFBDClient

# Max week for regular season (matches run_win_totals.py)
MAX_REGULAR_SEASON_WEEK = 15

# Conference alias mapping
CONF_ALIASES = {
    'sec': 'SEC',
    'big ten': 'Big Ten',
    'big 12': 'Big 12',
    'big12': 'Big 12',
    'acc': 'ACC',
    'aac': 'American Athletic',
    'american': 'American Athletic',
    'mw': 'Mountain West',
    'mountain west': 'Mountain West',
    'cusa': 'Conference USA',
    'conference usa': 'Conference USA',
    'mac': 'Mid-American',
    'mid-american': 'Mid-American',
    'sun belt': 'Sun Belt',
    'sunbelt': 'Sun Belt',
    'pac-12': 'Pac-12',
    'pac12': 'Pac-12',
    'ind': 'FBS Independents',
    'independent': 'FBS Independents',
    'independents': 'FBS Independents',
}


def resolve_conference(raw: str) -> str | None:
    """Resolve conference alias to full name."""
    key = raw.strip().lower()
    if key in CONF_ALIASES:
        return CONF_ALIASES[key]
    for alias, full in CONF_ALIASES.items():
        if key in alias or alias in key:
            return full
    return None


def get_team_conferences(year: int) -> dict[str, str]:
    """Get team -> conference mapping."""
    client = CFBDClient()
    teams = client.get_fbs_teams(year=year)
    return {normalize_team_name(t.school): t.conference for t in teams}


def get_actual_wins(year: int) -> dict[str, int]:
    """Get actual regular season win counts (week <= 15)."""
    client = CFBDClient()
    games = client.get_games(year=year, season_type='regular')
    wins: dict[str, int] = {}
    for g in games:
        if g.home_points is None or g.away_points is None:
            continue
        week = getattr(g, 'week', None)
        if week is not None and week > MAX_REGULAR_SEASON_WEEK:
            continue
        home = normalize_team_name(g.home_team)
        away = normalize_team_name(g.away_team)
        wins.setdefault(home, 0)
        wins.setdefault(away, 0)
        if g.home_points > g.away_points:
            wins[home] += 1
        elif g.away_points > g.home_points:
            wins[away] += 1
    return wins


def is_season_complete(year: int) -> bool:
    """A season is complete if we're past January of the following year."""
    return date.today() >= date(year + 1, 1, 15)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/show_win_totals.py <year> [top_n|all] [conference]")
        sys.exit(1)

    year = int(sys.argv[1])
    top_n_arg = sys.argv[2] if len(sys.argv) > 2 else '25'
    conf_arg = ' '.join(sys.argv[3:]) if len(sys.argv) > 3 else None

    # If second arg looks like a conference name (not a number or "all"), treat it as conf
    if top_n_arg.lower() not in ('all',) and not top_n_arg.isdigit():
        conf_arg = top_n_arg if not conf_arg else f"{top_n_arg} {conf_arg}"
        top_n_arg = 'all'

    csv_path = Path(f"data/win_totals/predictions_{year}.csv")
    if not csv_path.exists():
        print(f"No predictions found for {year}.")
        print(f"Run first: python3 -m src.win_totals.run_win_totals predict --year {year} --train-start 2015")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    conf_map = get_team_conferences(year)
    df['conference'] = df['team'].map(conf_map).fillna('Unknown')

    # Check if season is complete — fetch actual wins if so
    historical = is_season_complete(year)
    actual_wins_map: dict[str, int] = {}
    if historical:
        actual_wins_map = get_actual_wins(year)
        df['actual_wins'] = df['team'].map(actual_wins_map)

    # Filter by conference if requested
    conf_name = None
    if conf_arg:
        conf_name = resolve_conference(conf_arg)
        if conf_name is None:
            print(f"Unknown conference: '{conf_arg}'")
            print(f"Available: {', '.join(sorted(set(CONF_ALIASES.values())))}")
            sys.exit(1)
        df = df[df['conference'] == conf_name].copy()
        df['rank'] = range(1, len(df) + 1)

    # Apply top_n
    if top_n_arg.lower() == 'all':
        top_n = len(df)
    else:
        top_n = int(top_n_arg)

    df = df.head(top_n)

    if df.empty:
        print(f"No teams found for {year}" + (f" in {conf_name}" if conf_name else ""))
        sys.exit(0)

    # Build output
    title = f"## {year} JP+ Preseason Win Total Projections"
    if conf_name:
        title += f" — {conf_name}"
    print(title)
    print()

    show_conf_col = conf_name is None

    if historical:
        # Historical: Team (Conf) | E[W] | Actual | Result
        if show_conf_col:
            print("| Rank | Team | Conf | E[W] | Actual | Result |")
            print("|------|------|------|------|--------|--------|")
        else:
            print("| Rank | Team | E[W] | Actual | Result |")
            print("|------|------|------|--------|--------|")

        for _, row in df.iterrows():
            ew = row['expected_wins']
            actual = row.get('actual_wins')
            if pd.notna(actual):
                actual_int = int(actual)
                # Compare against expected wins rounded to nearest 0.5
                if actual_int > round(ew):
                    result = "Over"
                elif actual_int < round(ew):
                    result = "Under"
                else:
                    result = "Push"
                actual_str = str(actual_int)
            else:
                actual_str = "—"
                result = "—"

            if show_conf_col:
                print(f"| {int(row['rank'])} | **{row['team']}** | {row['conference']} | {ew:.1f} | {actual_str} | {result} |")
            else:
                print(f"| {int(row['rank'])} | **{row['team']}** | {ew:.1f} | {actual_str} | {result} |")
    else:
        # Future: Team (Conf) | E[W]
        if show_conf_col:
            print("| Rank | Team | Conf | E[W] |")
            print("|------|------|------|------|")
            for _, row in df.iterrows():
                print(f"| {int(row['rank'])} | **{row['team']}** | {row['conference']} | {row['expected_wins']:.1f} |")
        else:
            print("| Rank | Team | E[W] |")
            print("|------|------|------|")
            for _, row in df.iterrows():
                print(f"| {int(row['rank'])} | **{row['team']}** | {row['expected_wins']:.1f} |")


if __name__ == '__main__':
    main()
