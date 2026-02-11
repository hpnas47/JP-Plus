#!/usr/bin/env python3
"""
Diagnose LSA feature coefficients by examining the underlying game data.

This script loads the training data and outputs detailed breakdowns for
problematic features (rivalry_underdog_home, short_week_away, game_shape_opener_away).

Usage:
    python scripts/diagnose_lsa_features.py --years 2022 2023 2024 2025
    python scripts/diagnose_lsa_features.py --feature rivalry_underdog_home
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import polars as pl

from config.dtypes import optimize_dtypes
from src.api.cfbd_client import CFBDClient
from src.models.preseason_priors import PreseasonPriors
from src.models.efficiency_foundation_model import EfficiencyFoundationModel, clear_ridge_cache
from src.models.special_teams import SpecialTeamsModel
from src.models.fcs_strength import FCSStrengthEstimator
from src.models.learned_situational import SituationalFeatures, FEATURE_NAMES
from src.adjustments.home_field import HomeFieldAdvantage
from src.adjustments.situational import (
    SituationalAdjuster,
    HistoricalRankings,
    precalculate_schedule_metadata,
)
from src.adjustments.travel import TravelAdjuster
from src.adjustments.altitude import AltitudeAdjuster
from src.predictions.spread_generator import SpreadGenerator
from scripts.backtest import fetch_all_season_data, SeasonData

logging.basicConfig(
    level=logging.WARNING,  # Suppress info logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def collect_feature_games(
    season_data: dict[int, SeasonData],
    years: list[int],
    start_week: int = 4,
) -> pd.DataFrame:
    """Collect all games with their feature values and residuals.

    Returns a DataFrame with all games, their features, and training targets.
    """
    all_games = []

    for year in sorted(years):
        data = season_data[year]
        games_df = data.games_df
        efficiency_plays_df = data.efficiency_plays_df
        fbs_teams = data.fbs_teams
        priors = data.priors
        historical_rankings = data.historical_rankings
        team_conferences = data.team_conferences
        st_plays_df = data.st_plays_df

        fbs_teams_list = list(fbs_teams)
        games_df_pd = optimize_dtypes(games_df.to_pandas())
        games_df_pd = precalculate_schedule_metadata(games_df_pd)

        # Adjusters
        travel_adjuster = TravelAdjuster()
        altitude_adjuster = AltitudeAdjuster()
        situational_adjuster = SituationalAdjuster()
        hfa = HomeFieldAdvantage(base_hfa=2.5, global_offset=0.5)

        max_week = games_df["week"].max()

        for pred_week in range(start_week, max_week + 1):
            # Training data
            train_games_pl = games_df.filter(pl.col("week") < pred_week)
            train_plays_pl = efficiency_plays_df.join(
                train_games_pl.select("id"), left_on="game_id", right_on="id", how="semi"
            ).filter(
                pl.col("offense").is_in(fbs_teams_list) &
                pl.col("defense").is_in(fbs_teams_list)
            )

            if len(train_plays_pl) < 5000:
                continue

            train_plays_pd = optimize_dtypes(train_plays_pl.to_pandas())
            train_games_pd = optimize_dtypes(train_games_pl.to_pandas())

            # Build EFM
            efm = EfficiencyFoundationModel(
                ridge_alpha=50.0,
                efficiency_weight=0.45,
                explosiveness_weight=0.45,
                turnover_weight=0.10,
            )
            efm.calculate_ratings(
                train_plays_pd, train_games_pd,
                max_week=pred_week - 1, season=year,
                team_conferences=team_conferences,
                fbs_teams=fbs_teams,
            )

            team_ratings = {
                team: efm.get_rating(team)
                for team in fbs_teams
                if team in efm.team_ratings
            }

            # Blend with priors
            if priors is not None and priors.preseason_ratings:
                games_played = pred_week - 1
                team_ratings = priors.blend_with_inseason(
                    team_ratings,
                    games_played=games_played,
                    games_for_full_weight=8,
                    talent_floor_weight=0.08,
                )

            # Rankings
            sorted_teams = sorted(team_ratings.items(), key=lambda x: (-x[1], x[0]))
            rankings = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}

            # Special teams
            special_teams = SpecialTeamsModel()
            train_st_pl = st_plays_df.filter(pl.col("week") < pred_week) if st_plays_df is not None else None
            if train_st_pl is not None and len(train_st_pl) > 0:
                train_st_pd = optimize_dtypes(train_st_pl.to_pandas())
                special_teams.calculate_all_st_ratings_from_plays(train_st_pd, max_week=pred_week - 1)

            # FCS estimator
            fcs_estimator = FCSStrengthEstimator()
            fcs_estimator.update_from_games(games_df, fbs_teams, max_week=pred_week - 1)

            # Build spread generator
            spread_gen = SpreadGenerator(
                ratings=team_ratings,
                special_teams=special_teams,
                home_field=hfa,
                situational=situational_adjuster,
                travel=travel_adjuster,
                altitude=altitude_adjuster,
                fbs_teams=fbs_teams,
                st_spread_cap=2.5,
                fcs_estimator=fcs_estimator,
            )

            # Predict this week's games
            week_games = games_df.filter(pl.col("week") == pred_week)

            for game in week_games.iter_rows(named=True):
                try:
                    pred = spread_gen.predict_spread(
                        home_team=game["home_team"],
                        away_team=game["away_team"],
                        neutral_site=game["neutral_site"],
                        week=pred_week,
                        schedule_df=games_df_pd,
                        rankings=rankings,
                        historical_rankings=historical_rankings,
                        game_date=game.get("start_date"),
                    )

                    actual_margin = game["home_points"] - game["away_points"]

                    # Get situational factors
                    prelim_spread = team_ratings.get(game["home_team"], 0.0) - team_ratings.get(game["away_team"], 0.0)
                    home_is_favorite = prelim_spread > 0

                    home_factors, away_factors = situational_adjuster.get_matchup_factors(
                        home_team=game["home_team"],
                        away_team=game["away_team"],
                        current_week=pred_week,
                        schedule_df=games_df_pd,
                        rankings=rankings,
                        home_is_favorite=home_is_favorite,
                        historical_rankings=historical_rankings,
                        game_date=game.get("start_date"),
                    )

                    # Convert to feature vector
                    features = SituationalFeatures.from_situational_factors(home_factors, away_factors)

                    # Compute residual (same as LSA training target)
                    base_margin_no_situ = pred.spread - pred.components.situational
                    residual = actual_margin - base_margin_no_situ

                    # Store game data
                    game_data = {
                        "year": year,
                        "week": pred_week,
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "actual_margin": actual_margin,
                        "predicted_spread": pred.spread,
                        "base_margin_no_situ": base_margin_no_situ,
                        "fixed_situational": pred.components.situational,
                        "residual": residual,
                        "home_rating": team_ratings.get(game["home_team"], 0.0),
                        "away_rating": team_ratings.get(game["away_team"], 0.0),
                        "home_is_favorite": home_is_favorite,
                    }

                    # Add all feature values
                    feature_array = features.to_array()
                    for i, name in enumerate(FEATURE_NAMES):
                        game_data[f"feat_{name}"] = feature_array[i]

                    # Add raw factor values for debugging
                    game_data["home_rivalry_boost"] = home_factors.rivalry_boost
                    game_data["away_rivalry_boost"] = away_factors.rivalry_boost
                    game_data["home_rest_days"] = home_factors.rest_days
                    game_data["away_rest_days"] = away_factors.rest_days
                    game_data["home_is_opener"] = home_factors.is_season_opener
                    game_data["away_is_opener"] = away_factors.is_season_opener

                    all_games.append(game_data)

                except Exception as e:
                    logger.debug(f"Error processing {game['away_team']} @ {game['home_team']}: {e}")

    return pd.DataFrame(all_games)


def analyze_feature(df: pd.DataFrame, feature_name: str, top_n: int = 30):
    """Analyze games where a specific feature is triggered."""
    feat_col = f"feat_{feature_name}"

    if feat_col not in df.columns:
        print(f"Feature {feature_name} not found in data")
        return

    # Filter to games where feature is triggered
    triggered = df[df[feat_col] > 0].copy()

    if len(triggered) == 0:
        print(f"No games found with {feature_name} triggered")
        return

    print(f"\n{'='*80}")
    print(f"FEATURE FORENSICS: {feature_name}")
    print(f"{'='*80}")
    print(f"\nTotal games with feature triggered: {len(triggered)}")

    # Summary statistics
    print(f"\n--- Summary Statistics ---")
    print(f"Mean residual when triggered: {triggered['residual'].mean():.2f}")
    print(f"Median residual: {triggered['residual'].median():.2f}")
    print(f"Std residual: {triggered['residual'].std():.2f}")
    print(f"Min residual: {triggered['residual'].min():.2f}")
    print(f"Max residual: {triggered['residual'].max():.2f}")

    # Compare to non-triggered games
    not_triggered = df[df[feat_col] == 0]
    print(f"\nMean residual when NOT triggered: {not_triggered['residual'].mean():.2f}")
    print(f"Difference (effect): {triggered['residual'].mean() - not_triggered['residual'].mean():.2f}")

    # The "learned coefficient" approximation
    # Ridge regression with alpha=10 finds the coefficient that minimizes ||y - X*beta||^2 + alpha*||beta||^2
    # For a single binary feature, this is approximately: beta = sum(y * x) / (sum(x^2) + alpha / n)
    n = len(df)
    x = df[feat_col].values
    y = df['residual'].values
    approx_coef = np.sum(x * y) / (np.sum(x**2) + 10.0 / n)
    print(f"\nApproximate learned coefficient: {approx_coef:.2f}")

    # Show extreme games
    print(f"\n--- Top {top_n} Most NEGATIVE Residuals (triggered games) ---")
    print("(Negative residual = actual margin was LOWER than base prediction)")
    worst = triggered.nsmallest(top_n, 'residual')[
        ['year', 'week', 'home_team', 'away_team', 'actual_margin',
         'base_margin_no_situ', 'residual', 'home_is_favorite']
    ]
    print(worst.to_string(index=False))

    print(f"\n--- Top {top_n} Most POSITIVE Residuals (triggered games) ---")
    print("(Positive residual = actual margin was HIGHER than base prediction)")
    best = triggered.nlargest(top_n, 'residual')[
        ['year', 'week', 'home_team', 'away_team', 'actual_margin',
         'base_margin_no_situ', 'residual', 'home_is_favorite']
    ]
    print(best.to_string(index=False))

    # Year breakdown
    print(f"\n--- By Year ---")
    year_stats = triggered.groupby('year').agg({
        'residual': ['mean', 'count', 'std']
    }).round(2)
    print(year_stats.to_string())

    # Return the triggered games for further analysis
    return triggered


def main():
    parser = argparse.ArgumentParser(description="Diagnose LSA feature coefficients")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024, 2025],
        help="Years to analyze",
    )
    parser.add_argument(
        "--start-week",
        type=int,
        default=4,
        help="First week to start analysis (default: 4)",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="Specific feature to analyze (default: all problematic features)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for all game data",
    )

    args = parser.parse_args()

    print("Loading season data...")
    clear_ridge_cache()
    season_data = fetch_all_season_data(
        args.years,
        use_priors=True,
        use_portal=True,
        portal_scale=0.15,
    )

    print("Collecting feature data from all games...")
    df = collect_feature_games(season_data, args.years, args.start_week)

    print(f"\nTotal games collected: {len(df)}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Full data saved to {args.output}")

    # Analyze problematic features
    features_to_analyze = [args.feature] if args.feature else [
        "rivalry_underdog_home",
        "rivalry_underdog_away",
        "short_week_away",
        "short_week_home",
        "game_shape_opener_away",
        "game_shape_opener_home",
    ]

    for feature in features_to_analyze:
        analyze_feature(df, feature)

    # Double-counting audit
    print("\n" + "="*80)
    print("DOUBLE-COUNTING AUDIT")
    print("="*80)

    # Check if rivalry games have systematic bias
    rivalry_home = df[df['feat_rivalry_underdog_home'] > 0]
    rivalry_away = df[df['feat_rivalry_underdog_away'] > 0]

    print(f"\n--- Rivalry Underdog Analysis ---")
    print(f"Home underdogs in rivalry: {len(rivalry_home)} games")
    if len(rivalry_home) > 0:
        print(f"  Mean fixed_situational applied: {rivalry_home['fixed_situational'].mean():.2f}")
        print(f"  Mean home_rivalry_boost: {rivalry_home['home_rivalry_boost'].mean():.2f}")
        print(f"  Mean residual: {rivalry_home['residual'].mean():.2f}")

    print(f"\nAway underdogs in rivalry: {len(rivalry_away)} games")
    if len(rivalry_away) > 0:
        print(f"  Mean fixed_situational applied: {rivalry_away['fixed_situational'].mean():.2f}")
        print(f"  Mean away_rivalry_boost: {rivalry_away['away_rivalry_boost'].mean():.2f}")
        print(f"  Mean residual: {rivalry_away['residual'].mean():.2f}")

    # The KEY insight: if rivalry_underdog is learning negative, it means
    # the base_margin_no_situ is OVER-PREDICTING for rivalry underdogs
    # This could mean:
    # 1. Rivalry underdogs actually underperform (the boost is wrong)
    # 2. Some other factor is already capturing rivalry effect

    print(f"\n--- Interpretation ---")
    if len(rivalry_home) > 0:
        mean_resid = rivalry_home['residual'].mean()
        if mean_resid < 0:
            print(f"HOME rivalry underdogs have negative mean residual ({mean_resid:.2f})")
            print("  -> They UNDERPERFORM vs base_margin_no_situ")
            print("  -> The fixed rivalry boost may be counter-productive")
            print("  -> LSA learns negative coefficient to SUBTRACT from predictions")
        else:
            print(f"HOME rivalry underdogs have positive mean residual ({mean_resid:.2f})")
            print("  -> They OVERPERFORM vs base_margin_no_situ")
            print("  -> The fixed rivalry boost may be too small")


if __name__ == "__main__":
    main()
