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

from src.utils.display_helpers import get_abbrev, get_fbs_teams


def format_date(row) -> str:
    d = row.get('start_date')
    if pd.isna(d) or not d:
        return '—'
    from datetime import datetime
    dt = datetime.strptime(str(d)[:10], '%Y-%m-%d')
    return dt.strftime('%b %-d')


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
    # Load data from appropriate source
    if year <= 2025:
        # Historical data (2022-2025) from stable backtest artifact
        data_path = Path(__file__).parent.parent / 'data/moneyline_selection/outputs/backtest_moneyline_2022-2025.csv'
        if not data_path.exists():
            print(f"No historical moneyline data found at {data_path}")
            print("Run: python3 scripts/backfill_moneyline_log.py")
            return
        df = pd.read_csv(data_path)
        week_data = df[(df['year'] == year) & (df['week'] == week)]
    else:
        # Production data (2026+)
        data_path = Path(__file__).parent.parent / f'data/moneyline_selection/logs/moneyline_bets_{year}.csv'
        if not data_path.exists():
            print(f"No moneyline data found for {year}.")
            print(f"Run: python3 scripts/run_moneyline_weekly.py --year {year} --week {week} --inputs-path <path>")
            return
        df = pd.read_csv(data_path)
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

    # --- Actionable Bets ---
    print(f"\n## {year} Week {week} — Actionable Moneyline Bets\n")

    if list_a.empty:
        print("*No actionable moneyline bets this week.*\n")
    else:
        has_results = 'covered' in list_a.columns and list_a['covered'].notna().any()

        has_scores = 'home_points' in list_a.columns and list_a['home_points'].notna().any()
        has_dates = 'start_date' in list_a.columns and list_a['start_date'].notna().any()

        if has_results:
            if has_scores:
                if has_dates:
                    print("| # | Date | Matchup | Bet | Odds | p(W) | EV | Conf | Score | Result |")
                    print("|---|------|---------|-----|------|------|-----|------|-------|--------|")
                else:
                    print("| # | Matchup | Bet | Odds | p(W) | EV | Conf | Score | Result |")
                    print("|---|---------|-----|------|------|-----|------|-------|--------|")
            else:
                print("| # | Matchup | Bet | Odds | p(W) | EV | Conf | Result |")
                print("|---|---------|-----|------|------|-----|------|--------|")
        else:
            print("| # | Matchup | Bet | Odds | p(W) | EV | Conf |")
            print("|---|---------|-----|------|------|-----|------|")

        for i, (_, row) in enumerate(list_a.iterrows(), 1):
            matchup = f"{row['away_team']} @ {row['home_team']}"
            bet_abbr = get_abbrev(format_bet_team(row))
            odds = format_odds(row['odds_american'])
            p_win = f"{row['p_win']:.0%}"
            ev = f"+{row['ev']*100:.0f}%"
            conf = format_confidence(row)
            date = format_date(row) if has_dates else ''

            if has_results:
                result = format_result(row)
                if has_scores and pd.notna(row.get('home_points')):
                    score = f"{get_abbrev(row['away_team'])} {int(row['away_points'])}, {get_abbrev(row['home_team'])} {int(row['home_points'])}"
                    if has_dates:
                        print(f"| {i} | {date} | {matchup} | {bet_abbr} ML | {odds} | {p_win} | {ev} | {conf} | {score} | {result} |")
                    else:
                        print(f"| {i} | {matchup} | {bet_abbr} ML | {odds} | {p_win} | {ev} | {conf} | {score} | {result} |")
                elif has_scores:
                    if has_dates:
                        print(f"| {i} | {date} | {matchup} | {bet_abbr} ML | {odds} | {p_win} | {ev} | {conf} | — | {result} |")
                    else:
                        print(f"| {i} | {matchup} | {bet_abbr} ML | {odds} | {p_win} | {ev} | {conf} | — | {result} |")
                else:
                    print(f"| {i} | {matchup} | {bet_abbr} ML | {odds} | {p_win} | {ev} | {conf} | {result} |")
            else:
                print(f"| {i} | {matchup} | {bet_abbr} ML | {odds} | {p_win} | {ev} | {conf} |")

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

    # --- Watchlist ---
    print(f"\n## Watchlist\n")

    if list_b.empty:
        print("*No watchlist games this week.*\n")
    else:
        print("| # | Matchup | Side | Odds | EV | Disagree |")
        print("|---|---------|------|------|----|----------|")

        for i, (_, row) in enumerate(list_b.iterrows(), 1):
            matchup = f"{row['away_team']} @ {row['home_team']}"

            if pd.notna(row.get('side')):
                side_team = format_bet_team(row)
                side = f"{side_team} ML"
            else:
                side = "—"

            odds = format_odds(row['odds_american']) if pd.notna(row.get('odds_american')) else "—"
            ev = f"{row['ev']*100:+.1f}%" if pd.notna(row.get('ev')) else "—"
            disagree = f"{row['disagreement_pts']:.1f}" if pd.notna(row.get('disagreement_pts')) else "—"
            print(f"| {i} | {matchup} | {side} | {odds} | {ev} | {disagree} |")

        # Watchlist record
        if 'covered' in list_b.columns and list_b['covered'].notna().any():
            b_settled = list_b[list_b['covered'].notna()]
            b_wins = (b_settled['covered'] == 'W').sum()
            b_losses = (b_settled['covered'] == 'L').sum()
            b_pct = b_wins / (b_wins + b_losses) * 100 if (b_wins + b_losses) > 0 else 0
            print(f"\n**Watchlist Record: {b_wins}-{b_losses} ({b_pct:.1f}%)**")

            # Combined record
            a_settled = list_a[list_a['covered'].notna()] if 'covered' in list_a.columns else pd.DataFrame()
            a_wins = (a_settled['covered'] == 'W').sum() if len(a_settled) > 0 else 0
            a_losses = (a_settled['covered'] == 'L').sum() if len(a_settled) > 0 else 0
            c_wins = a_wins + b_wins
            c_losses = a_losses + b_losses
            c_pct = c_wins / (c_wins + c_losses) * 100 if (c_wins + c_losses) > 0 else 0
            print(f"\n**Combined Record: {c_wins}-{c_losses} ({c_pct:.1f}%)**")

    # Footnotes
    print("\n---")
    print(f"*Config: sigma={sigma}, ev_min={ev_min}, min_disagree={min_disagree}*")
    print("*ML EV = p(win) × (decimal_odds - 1) - p(loss). Sizing via quarter-Kelly.*")
    print("*Actionable bets capped at top 10 by EV.*")

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
