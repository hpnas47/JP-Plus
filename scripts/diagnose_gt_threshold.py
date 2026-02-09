#!/usr/bin/env python3
"""
Diagnostic: Garbage Time Threshold Analysis for Clock Rule Change

Hypothesis: The 2024 clock rule change (running clock on first downs) makes
leads safer. A 14-point Q4 lead in 2024-2025 is functionally equivalent to
a 16-point lead in 2022-2023.

Test: Compare prediction errors for games with Q4 plays in the 14-16 point
margin window, split by era (old rules vs new rules).

If 2024-25 shows systematic over-prediction of the trailing team (or under-
prediction of the leading team) that 2022-23 doesn't, the hypothesis is valid.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import polars as pl
from src.api.cfbd_client import CFBDClient
from scripts.backtest import fetch_season_data, fetch_season_plays, _remap_play_weeks


def identify_games_with_marginal_gt(plays_df: pd.DataFrame, year: int) -> dict:
    """
    Identify games that had Q4 plays in the 14-16 point margin window.

    Returns dict mapping game_id -> {
        'plays_in_window': count of plays with 14-16 pt margin in Q4,
        'total_q4_plays': total Q4 plays,
        'pct_in_window': percentage of Q4 plays in the marginal window,
        'home_leading': True if home team was leading in those plays
    }
    """
    # Filter to Q4 plays
    q4 = plays_df[plays_df['period'] == 4].copy()

    if len(q4) == 0:
        return {}

    # Calculate home/away scores from offense/defense perspective
    # offense_score is the score of the team on offense
    # If offense == home_team, then home_score = offense_score
    q4['home_score'] = np.where(
        q4['offense'] == q4['home_team'],
        q4['offense_score'],
        q4['defense_score']
    )
    q4['away_score'] = np.where(
        q4['offense'] == q4['home_team'],
        q4['defense_score'],
        q4['offense_score']
    )

    # Calculate margin (positive = home leading)
    q4['margin'] = q4['home_score'] - q4['away_score']
    q4['abs_margin'] = q4['margin'].abs()

    # Find plays in the 14-16 point window (the "marginal" garbage time zone)
    q4['in_marginal_window'] = (q4['abs_margin'] >= 14) & (q4['abs_margin'] <= 16)

    results = {}
    for game_id, group in q4.groupby('game_id'):
        plays_in_window = group['in_marginal_window'].sum()
        if plays_in_window > 0:
            # Determine which team was leading in those plays
            window_plays = group[group['in_marginal_window']]
            home_leading_plays = (window_plays['margin'] > 0).sum()

            results[game_id] = {
                'plays_in_window': int(plays_in_window),
                'total_q4_plays': len(group),
                'pct_in_window': plays_in_window / len(group) * 100,
                'home_leading_pct': home_leading_plays / plays_in_window * 100 if plays_in_window > 0 else 50,
            }

    return results


def run_diagnostic():
    """Run the GT threshold diagnostic."""
    client = CFBDClient()

    # Define eras
    old_rules_years = [2022, 2023]
    new_rules_years = [2024, 2025]

    print("=" * 80)
    print("GARBAGE TIME THRESHOLD DIAGNOSTIC: 14-16 Point Q4 Margin Analysis")
    print("=" * 80)
    print("\nHypothesis: 2024 clock rules make 14-16 pt leads 'safer' (more like old 16+ pt leads)")
    print("Test: Compare games with Q4 plays in 14-16 pt margin window across eras\n")

    all_results = []

    for year in old_rules_years + new_rules_years:
        era = "Old Rules" if year in old_rules_years else "New Rules"
        print(f"Processing {year} ({era})...")

        # Fetch data
        games_df, betting_df = fetch_season_data(client, year)
        _, _, efficiency_plays_df, _ = fetch_season_plays(client, year)

        # Create game_week_map for remapping
        if len(games_df) > 0:
            game_week_map = games_df.select([
                pl.col("id").alias("game_id"),
                pl.col("week").alias("game_week"),
            ])
            efficiency_plays_df = _remap_play_weeks(efficiency_plays_df, game_week_map)

        # Convert to pandas
        games = games_df.to_pandas()
        plays = efficiency_plays_df.to_pandas()
        betting = betting_df.to_pandas() if hasattr(betting_df, 'to_pandas') else betting_df

        # Get FBS teams
        fbs_teams = client.get_fbs_teams(year=year)
        fbs_set = {t.school for t in fbs_teams if t.school}

        # Filter to FBS vs FBS
        games = games[
            games['home_team'].isin(fbs_set) &
            games['away_team'].isin(fbs_set) &
            games['home_points'].notna() &
            games['away_points'].notna()
        ].copy()

        # Debug: print betting columns once
        if year == 2022:
            print(f"  Betting columns: {list(betting.columns)}")

        # Merge betting for spread (CFBD uses 'spread' column)
        spread_col = 'spread' if 'spread' in betting.columns else 'spread_open' if 'spread_open' in betting.columns else None
        if spread_col:
            betting_slim = betting[['game_id', spread_col]].drop_duplicates()
            betting_slim = betting_slim.rename(columns={'game_id': 'id', spread_col: 'vegas_spread'})
            games = games.merge(betting_slim, on='id', how='left')
            spread_coverage = games['vegas_spread'].notna().sum() / len(games) * 100
            print(f"  Spread coverage: {spread_coverage:.1f}% ({games['vegas_spread'].notna().sum()}/{len(games)} games)")
        else:
            print(f"  No spread column found in betting data")

        # Identify games with marginal GT plays
        marginal_games = identify_games_with_marginal_gt(plays, year)

        print(f"  Found {len(marginal_games)} games with Q4 plays in 14-16 pt margin window")

        # Analyze these games
        for game_id, gt_info in marginal_games.items():
            game = games[games['id'] == game_id]
            if len(game) == 0:
                continue
            game = game.iloc[0]

            # Calculate actual margin (positive = home won)
            actual_margin = game['home_points'] - game['away_points']
            vegas_spread = game['vegas_spread'] if 'vegas_spread' in game.index else np.nan

            # Determine if the leading team won/covered
            home_was_leading = gt_info['home_leading_pct'] > 50
            leader_won = (actual_margin > 0) == home_was_leading

            # For ATS: if home was leading in GT window, did home cover?
            if pd.notna(vegas_spread):
                # Vegas spread is negative if home favored
                home_covered = actual_margin > vegas_spread
                leader_covered = home_covered if home_was_leading else not home_covered
            else:
                home_covered = np.nan
                leader_covered = np.nan

            all_results.append({
                'year': year,
                'era': era,
                'game_id': game_id,
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_points': game['home_points'],
                'away_points': game['away_points'],
                'actual_margin': actual_margin,
                'vegas_spread': vegas_spread,
                'plays_in_window': gt_info['plays_in_window'],
                'total_q4_plays': gt_info['total_q4_plays'],
                'pct_in_window': gt_info['pct_in_window'],
                'home_leading_pct': gt_info['home_leading_pct'],
                'home_was_leading': home_was_leading,
                'leader_won': leader_won,
                'home_covered': home_covered,
                'leader_covered': leader_covered,
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("RESULTS BY ERA")
    print("=" * 80)

    for era in ["Old Rules", "New Rules"]:
        era_df = df[df['era'] == era]
        print(f"\n{era} ({', '.join(map(str, old_rules_years if era == 'Old Rules' else new_rules_years))}):")
        print(f"  Games with 14-16 pt Q4 margin: {len(era_df)}")
        print(f"  Avg plays in window: {era_df['plays_in_window'].mean():.1f}")
        print(f"  Avg % of Q4 in window: {era_df['pct_in_window'].mean():.1f}%")

        # Leader outcomes
        print(f"\n  Leader (team with 14-16 pt lead) outcomes:")
        print(f"    Won game: {era_df['leader_won'].mean()*100:.1f}%")

        valid_ats = era_df[era_df['leader_covered'].notna()]
        if len(valid_ats) > 0:
            print(f"    Covered spread: {valid_ats['leader_covered'].mean()*100:.1f}% ({int(valid_ats['leader_covered'].sum())}-{int((~valid_ats['leader_covered']).sum())})")

    # Key comparison: Did the leading team in marginal GT cover at different rates?
    print("\n" + "=" * 80)
    print("KEY DIAGNOSTIC: Leader ATS Performance in Marginal GT Window")
    print("=" * 80)

    old_df = df[(df['era'] == 'Old Rules') & df['leader_covered'].notna()]
    new_df = df[(df['era'] == 'New Rules') & df['leader_covered'].notna()]

    old_cover_rate = old_df['leader_covered'].mean() * 100 if len(old_df) > 0 else 0
    new_cover_rate = new_df['leader_covered'].mean() * 100 if len(new_df) > 0 else 0

    print(f"\nOld Rules (2022-23): Leader covered {old_cover_rate:.1f}% ({int(old_df['leader_covered'].sum())}-{int((~old_df['leader_covered']).sum())})")
    print(f"New Rules (2024-25): Leader covered {new_cover_rate:.1f}% ({int(new_df['leader_covered'].sum())}-{int((~new_df['leader_covered']).sum())})")
    print(f"Delta: {new_cover_rate - old_cover_rate:+.1f}%")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if new_cover_rate > old_cover_rate + 5:
        print("""
HYPOTHESIS SUPPORTED: Leaders in the 14-16 pt window cover MORE often under
new rules, suggesting the market hasn't fully adjusted to leads being "safer."
Lowering Q4 GT threshold from 16 to 14 for 2024+ could improve model accuracy.
""")
    elif new_cover_rate < old_cover_rate - 5:
        print("""
HYPOTHESIS REJECTED: Leaders in the 14-16 pt window cover LESS often under
new rules. The market may have OVER-adjusted. Current threshold is fine.
""")
    else:
        print("""
INCONCLUSIVE: No significant difference in leader cover rates between eras.
The 14-16 pt window behaves similarly under both rule sets, suggesting:
1. The market has correctly priced the rule change, OR
2. The rule change doesn't meaningfully affect this margin window

Recommendation: Keep current Q4 threshold at 16.
""")

    # Additional analysis: Final margins
    print("\n" + "-" * 80)
    print("SUPPLEMENTAL: Final Margin Distribution for Games with 14-16 pt Q4 Lead")
    print("-" * 80)

    for era in ["Old Rules", "New Rules"]:
        era_df = df[df['era'] == era]
        # From perspective of the leading team
        leader_margins = []
        for _, row in era_df.iterrows():
            if row['home_was_leading']:
                leader_margins.append(row['actual_margin'])
            else:
                leader_margins.append(-row['actual_margin'])

        if leader_margins:
            leader_margins = np.array(leader_margins)
            print(f"\n{era}:")
            print(f"  Mean final margin (leader perspective): {leader_margins.mean():+.1f}")
            print(f"  Std dev: {leader_margins.std():.1f}")
            print(f"  Leader won by 14+: {(leader_margins >= 14).mean()*100:.1f}%")
            print(f"  Leader won by 10-13: {((leader_margins >= 10) & (leader_margins < 14)).mean()*100:.1f}%")
            print(f"  Leader won by 1-9: {((leader_margins >= 1) & (leader_margins < 10)).mean()*100:.1f}%")
            print(f"  Leader lost: {(leader_margins < 0).mean()*100:.1f}%")

    return df


if __name__ == '__main__':
    run_diagnostic()
