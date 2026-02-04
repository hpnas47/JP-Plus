#!/usr/bin/env python3
"""
Calibrate situational adjustments using historical ATS data.

Measures the actual impact of:
- Bye week advantage
- Letdown spots (after beating top-15, facing unranked)
- Lookahead spots (rivalry or top-10 next week)
- Rivalry underdog boost

For each factor, calculates:
1. Number of games where factor applies
2. Average prediction error (our spread - actual margin)
3. ATS performance when factor applies
4. Recommended adjustment value
"""

import sys
sys.path.insert(0, '/Users/jason/Documents/CFB Power Ratings Model')

import argparse
import logging
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

from src.api.cfbd_client import CFBDClient
from src.models.preseason_priors import PreseasonPriors
from src.models.efficiency_foundation_model import EfficiencyFoundationModel
from src.adjustments.situational import SituationalAdjuster
from config.teams import is_rivalry_game

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

client = CFBDClient()


def fetch_season_data(year: int) -> tuple:
    """Fetch all data needed for a season."""
    logger.info(f"\nFetching data for {year}...")

    # FBS teams
    fbs_teams = {t.school for t in client.get_fbs_teams(year)}

    # Games
    games = []
    for week in range(1, 16):
        try:
            week_games = client.get_games(year, week)
            for g in week_games:
                if g.home_points is not None:
                    games.append({
                        'game_id': g.id,
                        'week': g.week,
                        'home_team': g.home_team,
                        'away_team': g.away_team,
                        'home_points': g.home_points,
                        'away_points': g.away_points,
                    })
        except:
            break
    games_df = pd.DataFrame(games)

    # Betting lines
    lines = []
    for week in range(1, 16):
        try:
            week_lines = client.get_betting_lines(year, week)
            for line in week_lines:
                if hasattr(line, 'lines') and line.lines:
                    for l in line.lines:
                        if hasattr(l, 'spread') and l.spread is not None:
                            lines.append({
                                'game_id': line.id,
                                'spread': l.spread,
                            })
                            break
        except:
            break
    lines_df = pd.DataFrame(lines)

    # Plays for EFM
    plays = []
    for week in range(1, 16):
        try:
            week_plays = client.get_plays(year, week)
            for p in week_plays:
                if p.ppa is not None and p.down is not None:
                    if p.offense in fbs_teams and p.defense in fbs_teams:
                        plays.append({
                            'week': week,
                            'game_id': p.game_id,
                            'down': p.down,
                            'distance': p.distance or 0,
                            'yards_gained': p.yards_gained or 0,
                            'play_type': p.play_type or '',
                            'offense': p.offense,
                            'defense': p.defense,
                            'period': p.period,
                            'ppa': p.ppa,
                            'offense_score': p.offense_score or 0,
                            'defense_score': p.defense_score or 0,
                        })
        except:
            break
    plays_df = pd.DataFrame(plays)

    return games_df, lines_df, plays_df, fbs_teams


def build_weekly_ratings(plays_df: pd.DataFrame, games_df: pd.DataFrame,
                         week: int, fbs_teams: set) -> dict:
    """Build EFM ratings using data up to (not including) specified week."""
    # Filter to games before this week
    train_plays = plays_df[plays_df['week'] < week].copy()
    train_games = games_df[games_df['week'] < week].copy()

    if len(train_plays) < 1000:  # Not enough data
        return {}

    efm = EfficiencyFoundationModel(
        ridge_alpha=50.0,
        efficiency_weight=0.54,
        explosiveness_weight=0.36,
        turnover_weight=0.10,
    )
    efm.calculate_ratings(train_plays, train_games)

    ratings_df = efm.get_ratings_df()
    fbs_ratings = ratings_df[ratings_df['team'].isin(fbs_teams)]

    # Normalize to SP+ scale
    mean_r = fbs_ratings['overall'].mean()
    std_r = fbs_ratings['overall'].std()
    if std_r > 0:
        scale = 12.0 / std_r
        normalized = {
            row['team']: (row['overall'] - mean_r) * scale
            for _, row in fbs_ratings.iterrows()
        }
        return normalized
    return {}


def identify_situational_factors(
    home_team: str,
    away_team: str,
    week: int,
    games_df: pd.DataFrame,
    rankings: dict,
) -> dict:
    """Identify which situational factors apply to a game."""
    factors = {
        'home_bye': False,
        'away_bye': False,
        'home_letdown': False,
        'away_letdown': False,
        'home_lookahead': False,
        'away_lookahead': False,
        'rivalry': False,
        'home_is_underdog': False,
    }

    # Check bye weeks
    last_week_games = games_df[games_df['week'] == week - 1]
    home_played_last = (
        (last_week_games['home_team'] == home_team) |
        (last_week_games['away_team'] == home_team)
    ).any()
    away_played_last = (
        (last_week_games['home_team'] == away_team) |
        (last_week_games['away_team'] == away_team)
    ).any()

    factors['home_bye'] = not home_played_last and week > 1
    factors['away_bye'] = not away_played_last and week > 1

    # Check letdown spots
    for team, is_home in [(home_team, True), (away_team, False)]:
        team_last_games = last_week_games[
            (last_week_games['home_team'] == team) |
            (last_week_games['away_team'] == team)
        ]
        if not team_last_games.empty:
            last_game = team_last_games.iloc[0]
            if last_game['home_team'] == team:
                won = last_game['home_points'] > last_game['away_points']
                last_opp = last_game['away_team']
            else:
                won = last_game['away_points'] > last_game['home_points']
                last_opp = last_game['home_team']

            if won:
                last_opp_rank = rankings.get(last_opp)
                current_opp = away_team if is_home else home_team
                current_opp_rank = rankings.get(current_opp)

                if last_opp_rank and last_opp_rank <= 15 and current_opp_rank is None:
                    if is_home:
                        factors['home_letdown'] = True
                    else:
                        factors['away_letdown'] = True

    # Check lookahead spots
    next_week_games = games_df[games_df['week'] == week + 1]
    for team, is_home in [(home_team, True), (away_team, False)]:
        team_next_games = next_week_games[
            (next_week_games['home_team'] == team) |
            (next_week_games['away_team'] == team)
        ]
        if not team_next_games.empty:
            next_game = team_next_games.iloc[0]
            next_opp = next_game['away_team'] if next_game['home_team'] == team else next_game['home_team']

            # Check if rivalry or top-10
            if is_rivalry_game(team, next_opp):
                if is_home:
                    factors['home_lookahead'] = True
                else:
                    factors['away_lookahead'] = True
            else:
                next_opp_rank = rankings.get(next_opp)
                if next_opp_rank and next_opp_rank <= 10:
                    if is_home:
                        factors['home_lookahead'] = True
                    else:
                        factors['away_lookahead'] = True

    # Check rivalry
    factors['rivalry'] = is_rivalry_game(home_team, away_team)

    return factors


def analyze_factor(games_with_factor: list, factor_name: str):
    """Analyze prediction error and ATS for games with a specific factor."""
    if not games_with_factor:
        return None

    df = pd.DataFrame(games_with_factor)

    # Prediction error: our_spread - actual_margin
    # Positive = we overestimated home team
    errors = df['our_spread'] - df['actual_margin']

    # ATS: home covers if actual_margin > -vegas_spread
    df['home_covers'] = df['actual_margin'] + df['vegas_spread'] > 0

    # Filter pushes
    df_no_push = df[abs(df['actual_margin'] + df['vegas_spread']) > 0.5]

    # Our edge: our_spread - vegas_spread
    # Negative = we like home more than Vegas
    df_no_push['edge'] = df_no_push['our_spread'] - df_no_push['vegas_spread']
    df_no_push['our_pick_covers'] = (
        (df_no_push['edge'] < 0) & df_no_push['home_covers']
    ) | (
        (df_no_push['edge'] > 0) & ~df_no_push['home_covers']
    )

    n_games = len(df)
    mean_error = errors.mean()
    std_error = errors.std()

    ats_wins = df_no_push['our_pick_covers'].sum() if len(df_no_push) > 0 else 0
    ats_total = len(df_no_push)
    ats_pct = ats_wins / ats_total if ats_total > 0 else 0

    # Recommended adjustment: negative of mean error
    # If we're overestimating by +2, we should adjust by -2
    recommended_adj = -mean_error

    # Statistical significance
    if n_games >= 10:
        t_stat, p_value = stats.ttest_1samp(errors, 0)
    else:
        t_stat, p_value = 0, 1.0

    return {
        'factor': factor_name,
        'n_games': n_games,
        'mean_error': mean_error,
        'std_error': std_error,
        'ats_record': f"{ats_wins}-{ats_total - ats_wins}",
        'ats_pct': ats_pct,
        'recommended_adj': recommended_adj,
        'p_value': p_value,
        'significant': p_value < 0.10,
    }


def main():
    parser = argparse.ArgumentParser(description='Calibrate situational adjustments')
    parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument('--start-week', type=int, default=5, help='First week to analyze')
    args = parser.parse_args()

    print("=" * 70)
    print("SITUATIONAL ADJUSTMENT CALIBRATION")
    print("=" * 70)
    print(f"Years: {args.years}")
    print(f"Start week: {args.start_week}")

    # Collect games by factor
    factor_games = defaultdict(list)
    all_games = []

    for year in args.years:
        games_df, lines_df, plays_df, fbs_teams = fetch_season_data(year)

        if games_df.empty or plays_df.empty:
            logger.warning(f"Insufficient data for {year}")
            continue

        # Merge games with lines
        merged = games_df.merge(lines_df, on='game_id', how='inner')

        for week in range(args.start_week, 16):
            # Build ratings using data up to this week
            ratings = build_weekly_ratings(plays_df, games_df, week, fbs_teams)

            if not ratings:
                continue

            # Build rankings
            sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
            rankings = {team: rank + 1 for rank, (team, _) in enumerate(sorted_ratings)}

            # Analyze games this week
            week_games = merged[merged['week'] == week]

            for _, game in week_games.iterrows():
                home = game['home_team']
                away = game['away_team']

                # Skip non-FBS matchups
                if home not in fbs_teams or away not in fbs_teams:
                    continue

                home_rating = ratings.get(home, 0)
                away_rating = ratings.get(away, 0)

                # Our spread (no situational adjustments)
                our_spread = away_rating - home_rating - 3.0  # -3 HFA

                actual_margin = game['home_points'] - game['away_points']
                vegas_spread = game['spread']

                # Identify situational factors
                factors = identify_situational_factors(
                    home, away, week, games_df, rankings
                )

                # Determine if home is underdog
                home_rank = rankings.get(home)
                away_rank = rankings.get(away)
                factors['home_is_underdog'] = home_rating < away_rating

                game_data = {
                    'year': year,
                    'week': week,
                    'home_team': home,
                    'away_team': away,
                    'our_spread': our_spread,
                    'vegas_spread': vegas_spread,
                    'actual_margin': actual_margin,
                    **factors,
                }

                all_games.append(game_data)

                # Categorize by factor
                if factors['home_bye']:
                    factor_games['home_bye'].append(game_data)
                if factors['away_bye']:
                    factor_games['away_bye'].append(game_data)
                if factors['home_letdown']:
                    factor_games['home_letdown'].append(game_data)
                if factors['away_letdown']:
                    factor_games['away_letdown'].append(game_data)
                if factors['home_lookahead']:
                    factor_games['home_lookahead'].append(game_data)
                if factors['away_lookahead']:
                    factor_games['away_lookahead'].append(game_data)
                if factors['rivalry']:
                    factor_games['rivalry'].append(game_data)
                if factors['rivalry'] and factors['home_is_underdog']:
                    factor_games['rivalry_home_underdog'].append(game_data)
                if factors['rivalry'] and not factors['home_is_underdog']:
                    factor_games['rivalry_away_underdog'].append(game_data)

    # Analyze each factor
    print(f"\n{'=' * 70}")
    print("FACTOR ANALYSIS")
    print("=" * 70)
    print(f"\nTotal games analyzed: {len(all_games)}")

    # Baseline (all games)
    baseline = analyze_factor(all_games, 'baseline')
    print(f"\nBaseline (all games): {baseline['n_games']} games, "
          f"ATS {baseline['ats_pct']:.1%}, Mean error {baseline['mean_error']:+.2f}")

    results = []

    factor_labels = {
        'home_bye': 'Home team off bye',
        'away_bye': 'Away team off bye',
        'home_letdown': 'Home in letdown spot',
        'away_letdown': 'Away in letdown spot',
        'home_lookahead': 'Home in lookahead spot',
        'away_lookahead': 'Away in lookahead spot',
        'rivalry': 'Rivalry game',
        'rivalry_home_underdog': 'Rivalry (home underdog)',
        'rivalry_away_underdog': 'Rivalry (away underdog)',
    }

    print(f"\n{'Factor':<30} {'N':>6} {'Mean Err':>10} {'ATS':>12} {'Rec Adj':>10} {'p-val':>8}")
    print("-" * 80)

    for factor_key, label in factor_labels.items():
        games = factor_games.get(factor_key, [])
        if not games:
            continue

        result = analyze_factor(games, factor_key)
        if result:
            results.append(result)
            sig = '*' if result['significant'] else ''
            print(f"{label:<30} {result['n_games']:>6} {result['mean_error']:>+10.2f} "
                  f"{result['ats_record']:>12} {result['recommended_adj']:>+10.2f} "
                  f"{result['p_value']:>7.3f}{sig}")

    # Summary and recommendations
    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS")
    print("=" * 70)

    print("\nCurrent values vs Recommended (based on historical error):")
    print(f"{'Factor':<25} {'Current':>10} {'Recommended':>12} {'Significant':>12}")
    print("-" * 60)

    current_values = {
        'home_bye': 1.5,
        'away_bye': -1.5,  # Helps home
        'home_letdown': -2.0,
        'away_letdown': 2.0,  # Helps home
        'home_lookahead': -1.5,
        'away_lookahead': 1.5,  # Helps home
        'rivalry_home_underdog': 1.0,
        'rivalry_away_underdog': -1.0,  # Helps home
    }

    for factor_key, current in current_values.items():
        result = next((r for r in results if r['factor'] == factor_key), None)
        if result:
            rec = result['recommended_adj']
            sig = 'Yes*' if result['significant'] else 'No'
            print(f"{factor_key:<25} {current:>+10.1f} {rec:>+12.2f} {sig:>12}")
        else:
            print(f"{factor_key:<25} {current:>+10.1f} {'N/A':>12} {'No data':>12}")

    print("\n* Statistically significant at p < 0.10")
    print("\nNote: Only adopt recommended values if significant AND sample size > 50")

    return results


if __name__ == '__main__':
    main()
