#!/usr/bin/env python3
"""
Backtesting script for CFB Power Ratings Model.

Performs walk-forward validation through historical seasons to measure:
- Mean Absolute Error (MAE) vs actual margins
- Win rate against the spread
- Simulated ROI

Usage:
    python scripts/backtest.py --years 2022 2023 2024
    python scripts/backtest.py --year 2024 --start-week 5
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from config.play_types import TURNOVER_PLAY_TYPES, POINTS_PER_TURNOVER, SCRIMMAGE_PLAY_TYPES
from src.api.cfbd_client import CFBDClient
from src.data.processors import DataProcessor, RecencyWeighter
from src.models.efficiency_foundation_model import EfficiencyFoundationModel
from src.models.preseason_priors import PreseasonPriors
from src.adjustments.home_field import HomeFieldAdvantage
from src.adjustments.situational import SituationalAdjuster
from src.adjustments.travel import TravelAdjuster
from src.adjustments.altitude import AltitudeAdjuster
# Legacy models (kept for comparison, EFM recommended instead)
from src.models.legacy.ridge_model import RidgeRatingsModel, TeamRatings
from src.models.legacy.luck_regression import LuckRegressor
from src.models.legacy.early_down_model import EarlyDownModel
from src.models.finishing_drives import FinishingDrivesModel
from src.models.special_teams import SpecialTeamsModel
from src.predictions.spread_generator import SpreadGenerator
from src.predictions.vegas_comparison import VegasComparison

import numpy as np
import pandas as pd
import polars as pl


def build_team_records(client: CFBDClient, years: list[int]) -> dict[str, dict[int, tuple[int, int]]]:
    """Build team win-loss records for trajectory calculation.

    Args:
        client: CFBD API client
        years: List of years to fetch records for

    Returns:
        Dict mapping team -> {year: (wins, losses)}
    """
    records = {}
    for year in years:
        try:
            games = client.get_games(year=year, season_type="regular")
            for game in games:
                if game.home_points is None or game.away_points is None:
                    continue

                home_won = game.home_points > game.away_points

                # Home team
                if game.home_team not in records:
                    records[game.home_team] = {}
                if year not in records[game.home_team]:
                    records[game.home_team][year] = (0, 0)
                w, l = records[game.home_team][year]
                records[game.home_team][year] = (w + (1 if home_won else 0), l + (0 if home_won else 1))

                # Away team
                if game.away_team not in records:
                    records[game.away_team] = {}
                if year not in records[game.away_team]:
                    records[game.away_team][year] = (0, 0)
                w, l = records[game.away_team][year]
                records[game.away_team][year] = (w + (0 if home_won else 1), l + (1 if home_won else 0))
        except Exception as e:
            logger.warning(f"Could not fetch games for {year}: {e}")

    return records


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# TURNOVER_PLAY_TYPES and POINTS_PER_TURNOVER imported from config.play_types


def fetch_season_data(
    client: CFBDClient,
    year: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch all games and betting lines for a season.

    Args:
        client: API client
        year: Season year

    Returns:
        Tuple of (games_df, betting_df) as Polars DataFrames
    """
    games = []
    betting = []
    failed_weeks = []
    successful_weeks = []

    for week in range(1, 16):
        try:
            week_games = client.get_games(year, week)
            week_game_count = 0
            for game in week_games:
                if game.home_points is None:
                    continue
                games.append({
                    "id": game.id,
                    "year": year,
                    "week": week,
                    "start_date": game.start_date,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "home_points": game.home_points,
                    "away_points": game.away_points,
                    "neutral_site": game.neutral_site or False,
                })
                week_game_count += 1
            if week_game_count > 0:
                successful_weeks.append(week)
        except Exception as e:
            failed_weeks.append((week, str(e)))
            logger.warning(f"Failed to fetch games for {year} week {week}: {e}")
            continue  # Continue to next week instead of breaking

    # Log fetch summary
    if failed_weeks:
        logger.warning(
            f"Games fetch for {year}: {len(successful_weeks)} weeks OK, "
            f"{len(failed_weeks)} weeks FAILED: {[w for w, _ in failed_weeks]}"
        )
    else:
        logger.debug(f"Games fetch for {year}: all {len(successful_weeks)} weeks OK")

    # Fetch betting lines
    # Prefer DraftKings for consistency, fall back to any available
    preferred_providers = ["DraftKings", "ESPN Bet", "Bovada"]
    try:
        lines = client.get_betting_lines(year)
        for game_lines in lines:
            if not game_lines.lines:
                continue

            # Find preferred provider line
            selected_line = None
            for provider in preferred_providers:
                for line in game_lines.lines:
                    if line.provider and line.provider == provider:
                        selected_line = line
                        break
                if selected_line:
                    break

            # Fall back to first available if no preferred provider found
            if selected_line is None:
                selected_line = game_lines.lines[0] if game_lines.lines else None

            if selected_line and selected_line.spread is not None:
                # CFBD spread is already from home team perspective
                # (negative = home favored, positive = away favored)
                # Capture both opening and closing lines
                betting.append({
                    "game_id": game_lines.id,
                    "home_team": game_lines.home_team,
                    "away_team": game_lines.away_team,
                    "spread_close": selected_line.spread,
                    "spread_open": selected_line.spread_open if selected_line.spread_open is not None else selected_line.spread,
                    "over_under": selected_line.over_under,
                    "provider": selected_line.provider,
                })
    except Exception as e:
        logger.warning(f"Error fetching betting lines: {e}")

    return pl.DataFrame(games), pl.DataFrame(betting)


def fetch_season_plays(
    client: CFBDClient,
    year: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Fetch all play-by-play data for a season.

    Args:
        client: API client
        year: Season year

    Returns:
        Tuple of (early_down_plays_df, turnover_plays_df, efficiency_plays_df, fg_plays_df) as Polars DataFrames
    """
    early_down_plays = []
    turnover_plays = []
    efficiency_plays = []  # All scrimmage plays with PPA for efficiency model
    fg_plays = []  # Field goal plays for special teams model
    failed_weeks = []
    successful_weeks = []

    for week in range(1, 16):
        try:
            plays = client.get_plays(year, week)
            week_play_count = 0
            for play in plays:
                play_type = play.play_type or ""

                # Collect turnover plays (offense = team that lost the ball)
                if play_type in TURNOVER_PLAY_TYPES:
                    turnover_plays.append({
                        "week": week,
                        "game_id": play.game_id,
                        "offense": play.offense,
                        "defense": play.defense,
                        "play_type": play_type,
                    })

                # Collect field goal plays
                if "Field Goal" in play_type:
                    fg_plays.append({
                        "week": week,
                        "game_id": play.game_id,
                        "offense": play.offense,
                        "play_type": play_type,
                        "play_text": play.play_text,
                    })

                # Collect early-down plays
                if play.down is not None and play.down in (1, 2):
                    early_down_plays.append({
                        "week": week,
                        "game_id": play.game_id,
                        "down": play.down,
                        "distance": play.distance,
                        "yards_gained": play.yards_gained or 0,
                        "play_type": play_type,
                        "offense": play.offense,
                        "defense": play.defense,
                        "period": play.period,
                        "offense_score": play.offense_score or 0,
                        "defense_score": play.defense_score or 0,
                    })

                # Collect scrimmage plays with PPA for efficiency model (P2.8 fix)
                # Filter: has PPA, has down, is scrimmage play type, valid distance
                if (play.ppa is not None and
                    play.down is not None and
                    play_type in SCRIMMAGE_PLAY_TYPES and
                    play.distance is not None and play.distance >= 0):
                    efficiency_plays.append({
                        "week": week,
                        "game_id": play.game_id,
                        "down": play.down,
                        "distance": play.distance,
                        "yards_gained": play.yards_gained or 0,
                        "play_type": play_type,
                        "play_text": play.play_text,  # Needed for TWP parsing
                        "offense": play.offense,
                        "defense": play.defense,
                        "period": play.period,
                        "ppa": play.ppa,
                        "yards_to_goal": play.yards_to_goal,
                        "offense_score": play.offense_score or 0,
                        "defense_score": play.defense_score or 0,
                        "home_team": play.home,  # For neutral-field ridge regression
                    })
                    week_play_count += 1

            if week_play_count > 0:
                successful_weeks.append(week)
        except Exception as e:
            failed_weeks.append((week, str(e)))
            logger.warning(f"Failed to fetch plays for {year} week {week}: {e}")
            continue  # Continue to next week instead of breaking

    # Log fetch summary
    if failed_weeks:
        logger.warning(
            f"Plays fetch for {year}: {len(successful_weeks)} weeks OK, "
            f"{len(failed_weeks)} weeks FAILED: {[w for w, _ in failed_weeks]}"
        )
    else:
        logger.debug(f"Plays fetch for {year}: all {len(successful_weeks)} weeks OK")

    logger.info(
        f"Fetched {len(early_down_plays)} early-down, {len(turnover_plays)} turnover, "
        f"{len(efficiency_plays)} efficiency, {len(fg_plays)} FG plays for {year}"
    )
    return (
        pl.DataFrame(early_down_plays),
        pl.DataFrame(turnover_plays),
        pl.DataFrame(efficiency_plays),
        pl.DataFrame(fg_plays),
    )


def build_game_turnovers(
    games_df: pl.DataFrame,
    turnover_plays_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build per-game net home turnover margin from play-by-play turnover data.

    Args:
        games_df: Games DataFrame with id, home_team, away_team columns
        turnover_plays_df: Turnover plays with game_id and offense columns

    Returns:
        Polars DataFrame with game_id and net_home_to_margin columns.
        Positive net_home_to_margin means home team gained more turnovers.
    """
    if turnover_plays_df is None or len(turnover_plays_df) == 0:
        return pl.DataFrame({"game_id": [], "net_home_to_margin": []})

    # Join turnover plays with games to get home_team
    turnovers_with_home = turnover_plays_df.join(
        games_df.select(["id", "home_team"]),
        left_on="game_id",
        right_on="id",
        how="left"
    )

    # Calculate home_lost (offense == home_team means home team lost the ball)
    result = (
        turnovers_with_home
        .with_columns([
            (pl.col("offense") == pl.col("home_team")).cast(pl.Int32).alias("home_lost")
        ])
        .group_by("game_id")
        .agg([
            pl.col("home_lost").sum().alias("home_turnovers"),
            pl.len().alias("total_turnovers")
        ])
        .with_columns([
            (pl.col("total_turnovers") - 2 * pl.col("home_turnovers")).alias("net_home_to_margin")
        ])
        .select(["game_id", "net_home_to_margin"])
    )

    return result


def walk_forward_predict(
    games_df: pd.DataFrame,
    start_week: int = 4,
    ridge_alpha: float = 150,
    preseason_priors: Optional[PreseasonPriors] = None,
    hfa_value: float = 2.8,
    decay_rate: float = 0.005,
    prior_weight: int = 8,
    margin_cap: int = 38,
    plays_df: Optional[pd.DataFrame] = None,
    turnover_df: Optional[pd.DataFrame] = None,
    to_scrub_factor: float = 0.5,
    fbs_teams: Optional[set[str]] = None,
    fcs_penalty_elite: float = 18.0,
    fcs_penalty_standard: float = 32.0,
) -> list[dict]:
    """Perform walk-forward prediction through a season.

    For each week from start_week onward:
    1. Train ridge model on all games through previous week
    2. Blend with preseason priors (weighted by games played)
    3. Generate predictions using SpreadGenerator which applies:
       - Base margin (ridge ratings)
       - Home field advantage
       - Luck adjustment (as differential)
       - Early-down efficiency (as differential)
       - Travel/altitude/situational adjustments
    4. Record predictions vs actual results

    NOTE: To avoid double-counting, luck and early-down are NOT baked into
    base ratings. They are applied as separate components at prediction time
    by SpreadGenerator.

    Args:
        games_df: All games for the season
        start_week: First week to start predictions (need data to train)
        ridge_alpha: Ridge regression alpha parameter
        preseason_priors: Preseason ratings to blend with in-season model
        hfa_value: Fixed home field advantage in points
        decay_rate: Recency decay rate for weighting games
        prior_weight: games_for_full_weight for preseason blending
        margin_cap: Cap game margins at this value (reduces blowout distortion)
        plays_df: Play-by-play data for early-down success rate model
        turnover_df: Per-game net home turnover margins from build_game_turnovers()
        to_scrub_factor: How much turnover noise to remove from margins (0-1).
                        0.0 = no scrub, 1.0 = full scrub.
        fbs_teams: Set of FBS team names for FCS detection
        fcs_penalty_elite: Points for elite FCS teams (default: 18.0)
        fcs_penalty_standard: Points for standard FCS teams (default: 32.0)

    Returns:
        List of prediction result dictionaries
    """
    results = []
    max_week = games_df["week"].max()

    for pred_week in range(start_week, max_week + 1):
        # Training data: all games before this week
        train_df = games_df[games_df["week"] < pred_week].copy()

        if len(train_df) < 50:
            logger.warning(f"Week {pred_week}: insufficient training data, skipping")
            continue

        # Compute raw margin
        if "margin" not in train_df.columns:
            train_df["margin"] = train_df["home_points"] - train_df["away_points"]

        # Scrub turnover noise from margins before capping
        if to_scrub_factor > 0 and turnover_df is not None and not turnover_df.empty:
            to_lookup = turnover_df.set_index("game_id")["net_home_to_margin"]
            scrub = train_df["id"].map(to_lookup).fillna(0) * POINTS_PER_TURNOVER * to_scrub_factor
            train_df["margin"] = train_df["margin"] - scrub

        # Cap blowout margins to reduce distortion from garbage time
        train_df["margin"] = train_df["margin"].clip(-margin_cap, margin_cap)
        # Adjust points to match capped margin (keep home_points, adjust away_points)
        train_df["away_points"] = train_df["home_points"] - train_df["margin"]

        # Process training data with specified decay rate
        recency_weighter = RecencyWeighter(decay_rate=decay_rate)
        processor = DataProcessor(recency_weighter=recency_weighter)
        processed = processor.process_games(train_df, apply_recency_weights=True)

        # Fit model with consistent HFA for both training and prediction
        model = RidgeRatingsModel(alpha=ridge_alpha, fixed_hfa=hfa_value)
        model.fit(processed)

        # Get team ratings
        ratings_df = model.get_all_ratings()
        team_ratings = dict(zip(ratings_df["team"], ratings_df["overall"]))

        # Blend with preseason priors if available
        if preseason_priors is not None and preseason_priors.preseason_ratings:
            games_played = pred_week - 1  # Weeks of data used for training
            blended = preseason_priors.blend_with_inseason(
                team_ratings, games_played, games_for_full_weight=prior_weight
            )
            # Update model ratings with blended values
            for team, rating in blended.items():
                model.ratings[team] = TeamRatings(
                    team=team, offense=rating, defense=0.0
                )
            logger.debug(
                f"Week {pred_week}: blended preseason "
                f"(preseason weight={1 - min(games_played/prior_weight, 1.0):.0%})"
            )

        # Use consistent HFA value for prediction
        hfa = HomeFieldAdvantage(base_hfa=hfa_value)
        for team in model.ratings.keys():
            hfa.team_hfa[team] = hfa_value

        # Calculate luck regression (applied as differential at prediction time, NOT baked into ratings)
        luck = LuckRegressor(skip_turnover_luck=(to_scrub_factor > 0))
        luck.calculate_all_teams(processed)

        # Calculate early-down model (applied as differential at prediction time, NOT baked into ratings)
        early_down = EarlyDownModel()
        if plays_df is not None and not plays_df.empty:
            train_plays = plays_df[plays_df["week"] < pred_week]
            if len(train_plays) > 0:
                early_down.calculate_all_teams(train_plays)

        # Build rankings from model ratings for situational adjustments
        team_ratings = {
            team: model.ratings[team].offense for team in model.ratings
        }
        sorted_teams = sorted(team_ratings.items(), key=lambda x: x[1], reverse=True)
        rankings = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}

        # Build spread generator - pass all component models
        # NOTE: Luck and early-down are applied as differentials at prediction time
        # to avoid double-counting (they are NOT in base ratings)
        situational = SituationalAdjuster()
        spread_gen = SpreadGenerator(
            ridge_model=model,
            luck_regressor=luck,
            early_down=early_down,
            home_field=hfa,
            situational=situational,
            travel=TravelAdjuster(),
            altitude=AltitudeAdjuster(),
            fbs_teams=fbs_teams,
            fcs_penalty_elite=fcs_penalty_elite,
            fcs_penalty_standard=fcs_penalty_standard,
        )

        # Predict this week's games
        week_games = games_df[games_df["week"] == pred_week]

        for _, game in week_games.iterrows():
            try:
                pred = spread_gen.predict_spread(
                    home_team=game["home_team"],
                    away_team=game["away_team"],
                    neutral_site=game["neutral_site"],
                    week=pred_week,
                    schedule_df=games_df,
                    rankings=rankings,
                )

                actual_margin = game["home_points"] - game["away_points"]

                results.append({
                    "game_id": game["id"],  # For reliable Vegas line matching
                    "year": game["year"],
                    "week": pred_week,
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "predicted_spread": pred.spread,
                    "actual_margin": actual_margin,
                    "error": pred.spread - actual_margin,
                    "abs_error": abs(pred.spread - actual_margin),
                })
            except Exception as e:
                logger.debug(f"Error predicting {game['away_team']} @ {game['home_team']}: {e}")

    return results


def walk_forward_predict_efm(
    games_df: pl.DataFrame,
    efficiency_plays_df: pl.DataFrame,
    fbs_teams: set[str],
    start_week: int = 4,
    preseason_priors: Optional[PreseasonPriors] = None,
    hfa_value: float = 2.5,
    prior_weight: int = 8,
    ridge_alpha: float = 50.0,  # Optimized via sweep
    efficiency_weight: float = 0.54,
    explosiveness_weight: float = 0.36,
    turnover_weight: float = 0.10,
    garbage_time_weight: float = 0.1,
    asymmetric_garbage: bool = True,
    team_records: Optional[dict[str, dict[int, tuple[int, int]]]] = None,
    year: int = 2024,
    fcs_penalty_elite: float = 18.0,
    fcs_penalty_standard: float = 32.0,
    fg_plays_df: Optional[pl.DataFrame] = None,
) -> list[dict]:
    """Perform walk-forward prediction using Efficiency Foundation Model.

    For each week from start_week onward:
    1. Train EFM on all plays from games before this week
    2. Rescale ratings to SP+ range
    3. Blend with preseason priors
    4. Generate predictions using SpreadGenerator
    5. Record predictions vs actual results

    Args:
        games_df: All games for the season (Polars DataFrame)
        efficiency_plays_df: Play-by-play data with PPA for efficiency model (Polars DataFrame)
        fbs_teams: Set of FBS team names
        start_week: First week to start predictions
        preseason_priors: Preseason ratings to blend with in-season model
        hfa_value: Fixed home field advantage in points
        prior_weight: games_for_full_weight for preseason blending
        ridge_alpha: Ridge alpha for EFM opponent adjustment
        efficiency_weight: Weight for success rate component (default 0.54)
        explosiveness_weight: Weight for IsoPPP component (default 0.36)
        turnover_weight: Weight for turnover margin component (default 0.10)
        garbage_time_weight: Weight for garbage time plays (default 0.1)
        asymmetric_garbage: Only penalize trailing team in garbage time (default True)
        team_records: Historical team records for trajectory calculation
        year: Current season year (for trajectory calculation)
        fcs_penalty_elite: Points for elite FCS teams (default 18.0)
        fcs_penalty_standard: Points for standard FCS teams (default 32.0)
        fg_plays_df: Field goal plays dataframe for FG efficiency calculation (Polars DataFrame)

    Returns:
        List of prediction result dictionaries
    """
    results = []
    max_week = games_df["week"].max()

    # Convert fbs_teams to list for Polars is_in()
    fbs_teams_list = list(fbs_teams)

    for pred_week in range(start_week, max_week + 1):
        # Training data: all plays from games before this week (Polars filtering - FAST)
        train_game_ids = games_df.filter(pl.col("week") < pred_week)["id"].to_list()

        # Filter plays to training games AND FBS-only (Polars filtering - 26x faster than pandas)
        train_plays_pl = efficiency_plays_df.filter(
            pl.col("game_id").is_in(train_game_ids) &
            pl.col("offense").is_in(fbs_teams_list) &
            pl.col("defense").is_in(fbs_teams_list)
        )

        if len(train_plays_pl) < 5000:
            logger.warning(f"Week {pred_week}: insufficient play data ({len(train_plays_pl)}), skipping")
            continue

        # Convert to pandas for EFM (sklearn needs pandas/numpy)
        train_plays_pd = train_plays_pl.to_pandas()
        train_games_pd = games_df.filter(pl.col("week") < pred_week).to_pandas()

        # Build EFM model
        efm = EfficiencyFoundationModel(
            ridge_alpha=ridge_alpha,
            efficiency_weight=efficiency_weight,
            explosiveness_weight=explosiveness_weight,
            turnover_weight=turnover_weight,
            garbage_time_weight=garbage_time_weight,
            asymmetric_garbage=asymmetric_garbage,
        )

        efm.calculate_ratings(train_plays_pd, train_games_pd)

        # Get ratings directly from EFM (full precision, already normalized to std=12)
        # IMPORTANT: Do NOT use get_ratings_df() here - it rounds to 1 decimal place
        # EFM._normalize_ratings() already scales to rating_std (default 12.0)
        # No second normalization needed - use ratings as-is
        team_ratings = {
            team: efm.get_rating(team)
            for team in fbs_teams
            if team in efm.team_ratings
        }

        # Blend with preseason priors if available
        if preseason_priors is not None and preseason_priors.preseason_ratings:
            games_played = pred_week - 1
            blended = preseason_priors.blend_with_inseason(
                team_ratings,
                games_played=games_played,
                games_for_full_weight=prior_weight,
                talent_floor_weight=0.08,
            )
            team_ratings = blended
            logger.debug(
                f"Week {pred_week}: blended preseason "
                f"(preseason weight={1 - min(games_played/prior_weight, 1.0):.0%})"
            )

        # Create a simple wrapper that mimics RidgeRatingsModel for SpreadGenerator
        class EFMWrapper:
            """Wrapper to make EFM compatible with SpreadGenerator."""

            def __init__(self, ratings: dict[str, float]):
                self.ratings = {
                    team: TeamRatings(team=team, offense=rating, defense=0.0)
                    for team, rating in ratings.items()
                }

            def predict_margin(
                self, home_team: str, away_team: str, neutral_site: bool = False
            ) -> float:
                home_r = self.ratings.get(home_team, TeamRatings(home_team, 0, 0)).offense
                away_r = self.ratings.get(away_team, TeamRatings(away_team, 0, 0)).offense
                return home_r - away_r

        efm_wrapper = EFMWrapper(team_ratings)

        # Initialize HFA with team-specific values and trajectory modifiers
        hfa = HomeFieldAdvantage(base_hfa=hfa_value)
        # Calculate trajectory modifiers if we have records
        if team_records:
            hfa.calculate_trajectory_modifiers(team_records, year)

        # Log HFA sources for this week's teams (first week only to avoid spam)
        if pred_week == start_week:
            # Get HFA breakdown for all FBS teams
            hfa_breakdown = hfa.get_hfa_breakdown(list(fbs_teams))
            source_counts = {"curated": 0, "dynamic": 0, "conference": 0, "fallback": 0}
            trajectory_count = 0
            for team, info in hfa_breakdown.items():
                source = info["source"]
                if source.startswith("curated"):
                    source_counts["curated"] += 1
                elif source.startswith("dynamic"):
                    source_counts["dynamic"] += 1
                elif source.startswith("conf:"):
                    source_counts["conference"] += 1
                else:
                    source_counts["fallback"] += 1
                if "+traj" in source:
                    trajectory_count += 1
            logger.info(
                f"HFA sources: curated={source_counts['curated']}, "
                f"fallback={source_counts['fallback']}, "
                f"with_trajectory={trajectory_count}"
            )

        # Build rankings for situational adjustments
        sorted_teams = sorted(team_ratings.items(), key=lambda x: x[1], reverse=True)
        rankings = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}

        # Calculate finishing drives with regression to mean
        finishing = FinishingDrivesModel(regress_to_mean=True, prior_strength=10)
        if "yards_to_goal" in train_plays_pd.columns:
            finishing.calculate_all_from_plays(train_plays_pd)

        # Calculate FG efficiency ratings from FG plays
        special_teams = SpecialTeamsModel()
        if fg_plays_df is not None and len(fg_plays_df) > 0:
            # Filter to FG plays before this week (Polars filtering)
            train_fg_pl = fg_plays_df.filter(pl.col("week") < pred_week)
            if len(train_fg_pl) > 0:
                train_fg_pd = train_fg_pl.to_pandas()
                special_teams.calculate_fg_ratings_from_plays(train_fg_pd)
                # Integrate special teams ratings into EFM for O/D/ST breakdown
                for team, st_rating in special_teams.team_ratings.items():
                    efm.set_special_teams_rating(team, st_rating.field_goal_rating)

        # Build spread generator with EFM as base model
        # NOTE: EFM is the foundation - no luck/early-down needed since it's built on efficiency
        # Finishing drives IS included to capture regressed RZ efficiency differential
        situational = SituationalAdjuster()
        spread_gen = SpreadGenerator(
            ridge_model=efm_wrapper,
            luck_regressor=LuckRegressor(),  # Empty - EFM doesn't need luck adjustment
            early_down=EarlyDownModel(),  # Empty - EFM already includes efficiency
            finishing_drives=finishing,  # Regressed RZ efficiency
            special_teams=special_teams,  # FG efficiency differential
            home_field=hfa,
            situational=situational,
            travel=TravelAdjuster(),
            altitude=AltitudeAdjuster(),
            fbs_teams=fbs_teams,
            fcs_penalty_elite=fcs_penalty_elite,
            fcs_penalty_standard=fcs_penalty_standard,
        )

        # Predict this week's games (Polars iteration is faster)
        week_games = games_df.filter(pl.col("week") == pred_week)

        # Convert games_df to pandas once for SpreadGenerator (it uses pandas internally)
        games_df_pd = games_df.to_pandas()

        for game in week_games.iter_rows(named=True):
            try:
                pred = spread_gen.predict_spread(
                    home_team=game["home_team"],
                    away_team=game["away_team"],
                    neutral_site=game["neutral_site"],
                    week=pred_week,
                    schedule_df=games_df_pd,
                    rankings=rankings,
                )

                actual_margin = game["home_points"] - game["away_points"]

                results.append({
                    "game_id": game["id"],  # For reliable Vegas line matching
                    "year": game["year"],
                    "week": pred_week,
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "predicted_spread": pred.spread,
                    "actual_margin": actual_margin,
                    "error": pred.spread - actual_margin,
                    "abs_error": abs(pred.spread - actual_margin),
                })
            except Exception as e:
                logger.debug(f"Error predicting {game['away_team']} @ {game['home_team']}: {e}")

    return results


def calculate_ats_results(
    predictions: list[dict],
    betting_df: pl.DataFrame,
    use_opening_line: bool = False,
) -> pd.DataFrame:
    """Calculate against-the-spread results.

    Uses game_id for reliable matching between predictions and betting lines.
    This avoids issues with team name drift, rematches, and neutral site swaps.

    Args:
        predictions: List of prediction dictionaries (must include game_id)
        betting_df: Polars DataFrame with Vegas lines (must have game_id, spread_close, spread_open)
        use_opening_line: If True, use opening lines; if False, use closing lines (default)

    Returns:
        Pandas DataFrame with ATS results
    """
    results = []
    unmatched_games = []
    spread_col = "spread_open" if use_opening_line else "spread_close"

    # Build lookup dict from betting_df for O(1) matching by game_id
    betting_lookup = {}
    for row in betting_df.iter_rows(named=True):
        betting_lookup[row["game_id"]] = row

    for pred in predictions:
        game_id = pred.get("game_id")

        # Match by game_id (reliable) rather than (home_team, away_team) (fragile)
        if game_id is None or game_id not in betting_lookup:
            # Track unmatched for sanity report
            unmatched_games.append({
                "game_id": game_id,
                "home_team": pred.get("home_team"),
                "away_team": pred.get("away_team"),
                "week": pred.get("week"),
                "year": pred.get("year"),
            })
            continue

        line_data = betting_lookup[game_id]
        vegas_spread = line_data[spread_col]

        if vegas_spread is None:
            unmatched_games.append({
                "game_id": game_id,
                "home_team": pred.get("home_team"),
                "away_team": pred.get("away_team"),
                "week": pred.get("week"),
                "year": pred.get("year"),
                "reason": f"missing {spread_col}",
            })
            continue

        actual_margin = pred["actual_margin"]  # Positive = home won
        model_spread = pred["predicted_spread"]  # Our model: positive = home favored

        # Convert our spread to Vegas convention for comparison
        # Our model: +10 means home favored by 10
        # Vegas: -10 means home favored by 10
        model_spread_vegas = -model_spread

        # Calculate edge (how much we differ from Vegas)
        edge = model_spread_vegas - vegas_spread
        # If edge < 0, our spread is more negative (we like home MORE than Vegas)
        # If edge > 0, our spread is less negative (we like home LESS than Vegas)
        model_pick_home = edge < 0  # Model likes home more than Vegas

        # Determine ATS result
        # home_cover > 0 means home beat the spread
        # Vegas spread is negative when home is favored (e.g., -7)
        # actual_margin + vegas_spread: e.g., home wins by 10, spread -7 => 10 + (-7) = 3 > 0 (covers)
        home_cover = actual_margin + vegas_spread
        if model_pick_home:
            ats_win = home_cover > 0
            ats_push = home_cover == 0
        else:
            ats_win = home_cover < 0
            ats_push = home_cover == 0

        results.append({
            **pred,
            "vegas_spread": vegas_spread,
            "edge": abs(edge),
            "pick": "HOME" if model_pick_home else "AWAY",
            "ats_win": ats_win,
            "ats_push": ats_push,
        })

    # Sanity report: log match rate and unmatched games
    total_predictions = len(predictions)
    matched = len(results)
    unmatched = len(unmatched_games)
    match_rate = matched / total_predictions if total_predictions > 0 else 0

    logger.info(
        f"ATS line matching: {matched}/{total_predictions} predictions matched ({match_rate:.1%}), "
        f"{unmatched} unmatched"
    )

    if unmatched_games and unmatched <= 20:
        # Log details for small numbers of unmatched
        for game in unmatched_games[:10]:
            reason = game.get("reason", "no betting line")
            logger.debug(
                f"  Unmatched: {game.get('away_team')} @ {game.get('home_team')} "
                f"(week {game.get('week')}, game_id={game.get('game_id')}) - {reason}"
            )
        if unmatched > 10:
            logger.debug(f"  ... and {unmatched - 10} more unmatched games")
    elif unmatched > 20:
        logger.warning(
            f"High unmatched rate: {unmatched} games without betting lines. "
            "Check game_id alignment between games and betting data."
        )

    return pd.DataFrame(results)


def calculate_metrics(
    predictions_df: pd.DataFrame,
    ats_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Calculate backtest performance metrics.

    Args:
        predictions_df: DataFrame with predictions
        ats_df: DataFrame with ATS results (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "total_games": len(predictions_df),
        "mae": predictions_df["abs_error"].mean(),
        "rmse": np.sqrt((predictions_df["error"] ** 2).mean()),
        "median_error": predictions_df["abs_error"].median(),
        "within_3": (predictions_df["abs_error"] <= 3).mean(),
        "within_7": (predictions_df["abs_error"] <= 7).mean(),
        "within_10": (predictions_df["abs_error"] <= 10).mean(),
    }

    if ats_df is not None and len(ats_df) > 0:
        # Overall ATS
        wins = ats_df["ats_win"].sum()
        pushes = ats_df["ats_push"].sum()
        losses = len(ats_df) - wins - pushes

        metrics["ats_record"] = f"{wins}-{losses}-{pushes}"
        metrics["ats_win_rate"] = wins / (wins + losses) if (wins + losses) > 0 else 0

        # ROI (assuming -110 odds)
        bet_amount = len(ats_df) - pushes  # Exclude pushes from bet count
        winnings = wins * 100 - losses * 110  # Win $100, lose $110
        metrics["roi"] = winnings / (bet_amount * 110) if bet_amount > 0 else 0

        # ATS by edge threshold
        for threshold in [2, 3, 5]:
            edge_df = ats_df[ats_df["edge"] >= threshold]
            if len(edge_df) > 0:
                edge_wins = edge_df["ats_win"].sum()
                edge_losses = len(edge_df) - edge_wins - edge_df["ats_push"].sum()
                edge_total = edge_wins + edge_losses
                metrics[f"ats_{threshold}pt_edge"] = (
                    f"{edge_wins}-{edge_losses} "
                    f"({edge_wins/edge_total:.1%})" if edge_total > 0 else "N/A"
                )

    return metrics


def fetch_all_season_data(
    years: list[int],
    use_priors: bool = True,
    use_portal: bool = True,
    portal_scale: float = 0.15,
) -> dict:
    """Fetch and cache all season data (games, betting, plays, turnovers, priors, fbs_teams).

    Args:
        years: List of years to fetch
        use_priors: Whether to build preseason priors
        use_portal: Whether to incorporate transfer portal data into priors
        portal_scale: How much to weight portal impact (default 0.15)

    Returns:
        Dict mapping year to (games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, fg_plays_df)
    """
    client = CFBDClient()
    season_data = {}

    for year in years:
        logger.info(f"\nFetching data for {year}...")

        games_df, betting_df = fetch_season_data(client, year)
        logger.info(f"Loaded {len(games_df)} games, {len(betting_df)} betting lines")

        plays_df, turnover_plays_df, efficiency_plays_df, fg_plays_df = fetch_season_plays(client, year)

        turnover_df = build_game_turnovers(games_df, turnover_plays_df)
        logger.info(f"Built turnover margins for {len(turnover_df)} games")

        # Sanity check: validate data completeness for determinism
        weeks_with_games = games_df["week"].unique().to_list() if len(games_df) > 0 else []
        weeks_with_plays = efficiency_plays_df["week"].unique().to_list() if len(efficiency_plays_df) > 0 else []
        expected_weeks = set(range(1, 16))
        missing_game_weeks = expected_weeks - set(weeks_with_games)
        missing_play_weeks = expected_weeks - set(weeks_with_plays)

        if missing_game_weeks or missing_play_weeks:
            logger.warning(
                f"Data completeness check for {year}: "
                f"games missing weeks {sorted(missing_game_weeks) if missing_game_weeks else 'none'}, "
                f"plays missing weeks {sorted(missing_play_weeks) if missing_play_weeks else 'none'}"
            )
        else:
            logger.info(f"Data completeness check for {year}: all weeks 1-15 present")

        # Fetch FBS teams for EFM filtering
        fbs_teams_list = client.get_fbs_teams(year)
        fbs_teams = {t.school for t in fbs_teams_list}
        logger.info(f"Loaded {len(fbs_teams)} FBS teams")

        priors = None
        if use_priors:
            try:
                priors = PreseasonPriors()
                priors.calculate_preseason_ratings(
                    year,
                    use_portal=use_portal,
                    portal_scale=portal_scale,
                )
                logger.info(
                    f"Loaded preseason priors for {len(priors.preseason_ratings)} teams"
                )
            except Exception as e:
                logger.warning(f"Could not load preseason priors for {year}: {e}")
                priors = None

        season_data[year] = (games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, fg_plays_df)

    return season_data


def run_backtest(
    years: list[int],
    start_week: int = 4,
    ridge_alpha: float = 50.0,
    use_priors: bool = True,
    hfa_value: float = 2.5,
    decay_rate: float = 0.005,
    prior_weight: int = 8,
    margin_cap: int = 38,
    season_data: Optional[dict] = None,
    to_scrub_factor: float = 0.5,
    use_efm: bool = False,
    efm_efficiency_weight: float = 0.54,
    efm_explosiveness_weight: float = 0.36,
    efm_turnover_weight: float = 0.10,
    efm_garbage_time_weight: float = 0.1,
    efm_asymmetric_garbage: bool = True,
    fcs_penalty_elite: float = 18.0,
    fcs_penalty_standard: float = 32.0,
    use_portal: bool = True,
    portal_scale: float = 0.15,
    use_opening_line: bool = False,
) -> dict:
    """Run full backtest across specified years.

    Args:
        years: List of years to backtest
        start_week: First week to start predictions
        ridge_alpha: Ridge regression alpha (for both ridge and EFM opponent adjustment)
        use_priors: Whether to use preseason priors
        hfa_value: Fixed home field advantage in points
        decay_rate: Recency decay rate (only for ridge model)
        prior_weight: games_for_full_weight for preseason blending
        margin_cap: Cap game margins at this value (only for ridge model)
        season_data: Pre-fetched season data (for sweep caching)
        to_scrub_factor: How much turnover noise to remove from margins (only for ridge model)
        use_efm: Use Efficiency Foundation Model instead of ridge
        efm_efficiency_weight: EFM success rate weight (default 0.54)
        efm_explosiveness_weight: EFM IsoPPP weight (default 0.36)
        efm_turnover_weight: EFM turnover margin weight (default 0.10)
        efm_garbage_time_weight: EFM garbage time play weight (default 0.1)
        efm_asymmetric_garbage: Only penalize trailing team in garbage time (default True)
        fcs_penalty_elite: Points for elite FCS teams (default 18.0, EFM only)
        fcs_penalty_standard: Points for standard FCS teams (default 32.0, EFM only)
        use_portal: Whether to incorporate transfer portal into preseason priors
        portal_scale: How much to weight portal impact (default 0.15)
        use_opening_line: If True, use opening lines for ATS; if False, use closing lines (default)

    Returns:
        Dictionary with backtest results
    """
    # Fetch data if not pre-cached
    if season_data is None:
        season_data = fetch_all_season_data(
            years,
            use_priors=use_priors,
            use_portal=use_portal,
            portal_scale=portal_scale,
        )

    # Build team records for trajectory calculation (need ~4 years before earliest year)
    if use_efm:
        client = CFBDClient()
        trajectory_years = list(range(min(years) - 4, max(years) + 1))
        team_records = build_team_records(client, trajectory_years)
        logger.info(f"Built team records for trajectory ({len(team_records)} teams, years {trajectory_years[0]}-{trajectory_years[-1]})")
    else:
        team_records = None

    all_predictions = []
    all_ats = []

    for year in years:
        logger.info(f"\nBacktesting {year} season {'(EFM)' if use_efm else '(Ridge)'}...")

        games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, fg_plays_df = season_data[year]

        if use_efm:
            # Walk-forward predictions using EFM
            predictions = walk_forward_predict_efm(
                games_df,
                efficiency_plays_df,
                fbs_teams,
                start_week,
                preseason_priors=priors,
                hfa_value=hfa_value,
                prior_weight=prior_weight,
                ridge_alpha=ridge_alpha,
                efficiency_weight=efm_efficiency_weight,
                explosiveness_weight=efm_explosiveness_weight,
                turnover_weight=efm_turnover_weight,
                garbage_time_weight=efm_garbage_time_weight,
                asymmetric_garbage=efm_asymmetric_garbage,
                team_records=team_records,
                year=year,
                fcs_penalty_elite=fcs_penalty_elite,
                fcs_penalty_standard=fcs_penalty_standard,
                fg_plays_df=fg_plays_df,
            )
        else:
            # Walk-forward predictions using ridge model (legacy - needs pandas)
            predictions = walk_forward_predict(
                games_df.to_pandas(),
                start_week,
                ridge_alpha,
                preseason_priors=priors,
                hfa_value=hfa_value,
                decay_rate=decay_rate,
                prior_weight=prior_weight,
                margin_cap=margin_cap,
                plays_df=plays_df.to_pandas() if plays_df is not None else None,
                turnover_df=turnover_df.to_pandas() if turnover_df is not None else None,
                to_scrub_factor=to_scrub_factor,
                fbs_teams=fbs_teams,
                fcs_penalty_elite=fcs_penalty_elite,
                fcs_penalty_standard=fcs_penalty_standard,
            )
        all_predictions.extend(predictions)

        # Calculate ATS
        if len(betting_df) > 0:
            ats_results = calculate_ats_results(predictions, betting_df, use_opening_line)
            all_ats.append(ats_results)

        # Year metrics
        year_df = pd.DataFrame(predictions)
        if not year_df.empty:
            logger.info(f"{year} MAE: {year_df['abs_error'].mean():.2f}")

    # Combine results
    predictions_df = pd.DataFrame(all_predictions)
    ats_df = pd.concat(all_ats, ignore_index=True) if all_ats else None

    # Calculate overall metrics
    metrics = calculate_metrics(predictions_df, ats_df)

    return {
        "predictions": predictions_df,
        "ats_results": ats_df,
        "metrics": metrics,
    }


def run_sweep(
    years: list[int],
    start_week: int = 4,
    use_priors: bool = True,
    use_efm: bool = False,
) -> pd.DataFrame:
    """Run grid search over key parameters.

    Tests combinations of alpha, decay, and hfa to find optimal settings.

    Args:
        years: List of years to backtest
        start_week: First week to start predictions
        use_priors: Whether to use preseason priors
        use_efm: Use Efficiency Foundation Model instead of ridge

    Returns:
        DataFrame with results for each parameter combination
    """
    from itertools import product

    if use_efm:
        # EFM sweep: alpha, hfa, efficiency_weight
        # Note: turnover_weight is fixed at 0.10, so efficiency + explosiveness = 0.90
        alphas = [50, 100, 200]
        hfas = [2.0, 2.5, 3.0]
        eff_weights = [0.50, 0.54, 0.60]  # These are efficiency weights (explosiveness = 0.90 - eff)
        turnover_weight = 0.10  # Fixed at 10%

        combos = list(product(alphas, hfas, eff_weights))
        total = len(combos)

        print(f"\nSweeping {total} EFM parameter combinations...")
        print(f"alpha:           {alphas}")
        print(f"hfa:             {hfas}")
        print(f"efficiency_wt:   {eff_weights}")
        print(f"turnover_wt:     {turnover_weight} (fixed)")
        print()

        # Pre-fetch all season data once
        cached_data = fetch_all_season_data(years, use_priors=use_priors)

        sweep_results = []

        for i, (alpha, hfa, eff_wt) in enumerate(combos, 1):
            try:
                result = run_backtest(
                    years=years,
                    start_week=start_week,
                    season_data=cached_data,
                    ridge_alpha=alpha,
                    use_priors=use_priors,
                    hfa_value=hfa,
                    use_efm=True,
                    efm_efficiency_weight=eff_wt,
                    efm_explosiveness_weight=0.90 - eff_wt,  # 0.90 = 1.0 - turnover_weight
                    efm_turnover_weight=turnover_weight,
                )
                metrics = result["metrics"]

                ats_w, ats_l, ats_pct, roi = 0, 0, 0.0, 0.0
                if "ats_record" in metrics:
                    parts = metrics["ats_record"].split("-")
                    ats_w = int(parts[0])
                    ats_l = int(parts[1])
                    ats_pct = metrics["ats_win_rate"]
                    roi = metrics["roi"]

                sweep_results.append({
                    "model": "EFM",
                    "alpha": alpha,
                    "hfa": hfa,
                    "eff_wt": eff_wt,
                    "MAE": round(metrics["mae"], 2),
                    "ATS_W": ats_w,
                    "ATS_L": ats_l,
                    "ATS%": round(ats_pct * 100, 1),
                    "ROI": round(roi * 100, 1),
                })

                print(
                    f"  [{i:3d}/{total}] alpha={alpha:>3.0f}  hfa={hfa:.1f}  "
                    f"eff_wt={eff_wt:.2f}  "
                    f"MAE={metrics['mae']:.2f}  "
                    f"ATS={ats_w}-{ats_l} ({ats_pct:.1%})  ROI={roi:.1%}"
                )
            except Exception as e:
                logger.warning(
                    f"Error with alpha={alpha}, hfa={hfa}, eff_wt={eff_wt}: {e}"
                )
    else:
        # Ridge sweep
        alphas = [10, 100, 300]
        decays = [0.005, 0.015]
        hfas = [2.5, 3.0, 3.5]
        to_scrub_factors = [0.0, 0.5, 1.0]

        combos = list(product(alphas, decays, hfas, to_scrub_factors))
        total = len(combos)

        print(f"\nSweeping {total} Ridge parameter combinations...")
        print(f"alpha:          {alphas}")
        print(f"decay:          {decays}")
        print(f"hfa:            {hfas}")
        print(f"to_scrub_factor: {to_scrub_factors}")
        print()

        # Pre-fetch all season data once (avoids re-fetching for each combo)
        cached_data = fetch_all_season_data(years, use_priors=use_priors)

        sweep_results = []

        for i, (alpha, decay, hfa, tosf) in enumerate(combos, 1):
            try:
                result = run_backtest(
                    years=years,
                    start_week=start_week,
                    season_data=cached_data,
                    ridge_alpha=alpha,
                    use_priors=use_priors,
                    hfa_value=hfa,
                    decay_rate=decay,
                    to_scrub_factor=tosf,
                    use_efm=False,
                )
                metrics = result["metrics"]

                ats_w, ats_l, ats_pct, roi = 0, 0, 0.0, 0.0
                if "ats_record" in metrics:
                    parts = metrics["ats_record"].split("-")
                    ats_w = int(parts[0])
                    ats_l = int(parts[1])
                    ats_pct = metrics["ats_win_rate"]
                    roi = metrics["roi"]

                sweep_results.append({
                    "model": "Ridge",
                    "alpha": alpha,
                    "decay": decay,
                    "hfa": hfa,
                    "to_scrub": tosf,
                    "MAE": round(metrics["mae"], 2),
                    "ATS_W": ats_w,
                    "ATS_L": ats_l,
                    "ATS%": round(ats_pct * 100, 1),
                    "ROI": round(roi * 100, 1),
                })

                print(
                    f"  [{i:3d}/{total}] alpha={alpha:>3.0f}  decay={decay:.3f}  "
                    f"hfa={hfa:.1f}  tosf={tosf:.1f}  "
                    f"MAE={metrics['mae']:.2f}  "
                    f"ATS={ats_w}-{ats_l} ({ats_pct:.1%})  ROI={roi:.1%}"
                )
            except Exception as e:
                logger.warning(
                    f"Error with alpha={alpha}, decay={decay}, hfa={hfa}, "
                    f"tosf={tosf}: {e}"
                )

    df = pd.DataFrame(sweep_results)
    df = df.sort_values("ATS%", ascending=False).reset_index(drop=True)
    return df


def print_results(results: dict) -> None:
    """Print backtest results to console.

    Args:
        results: Dictionary with backtest results from run_backtest
    """
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)

    metrics = results["metrics"]
    print(f"\nTotal games predicted: {metrics['total_games']}")
    print(f"\nError Metrics:")
    print(f"  MAE: {metrics['mae']:.2f} points")
    print(f"  RMSE: {metrics['rmse']:.2f} points")
    print(f"  Median Error: {metrics['median_error']:.2f} points")
    print(f"\nPrediction Accuracy:")
    print(f"  Within 3 pts: {metrics['within_3']:.1%}")
    print(f"  Within 7 pts: {metrics['within_7']:.1%}")
    print(f"  Within 10 pts: {metrics['within_10']:.1%}")

    if "ats_record" in metrics:
        print(f"\nAgainst the Spread:")
        print(f"  Record: {metrics['ats_record']}")
        print(f"  Win Rate: {metrics['ats_win_rate']:.1%}")
        print(f"  ROI: {metrics['roi']:.1%}")

        for threshold in [2, 3, 5]:
            key = f"ats_{threshold}pt_edge"
            if key in metrics:
                print(f"  {threshold}+ pt edge: {metrics[key]}")

    # Weekly MAE breakdown
    predictions_df = results["predictions"]
    if not predictions_df.empty:
        print(f"\nMAE by Week:")
        weekly = predictions_df.groupby("week").agg(
            games=("abs_error", "count"),
            mae=("abs_error", "mean"),
        )
        for week, row in weekly.iterrows():
            print(f"  Week {week:2d}: MAE={row['mae']:5.2f}  (n={int(row['games'])})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backtest CFB Power Ratings Model")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024, 2025],
        help="Years to backtest",
    )
    parser.add_argument(
        "--start-week",
        type=int,
        default=4,
        help="First week to start predictions",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=50.0,
        help="Ridge regression alpha parameter (default: 50.0)",
    )
    parser.add_argument(
        "--hfa",
        type=float,
        default=2.5,
        help="Home field advantage in points (default: 2.5)",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.005,
        help="Recency decay rate (default: 0.005)",
    )
    parser.add_argument(
        "--prior-weight",
        type=int,
        default=8,
        help="Games for full preseason prior weight (default: 8)",
    )
    parser.add_argument(
        "--margin-cap",
        type=int,
        default=38,
        help="Cap game margins at this value (default: 38, use 999 to disable)",
    )
    parser.add_argument(
        "--to-scrub-factor",
        type=float,
        default=0.5,
        help="Turnover margin scrub factor (default: 0.5 = half scrub, 0.0 = disabled, 1.0 = full)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for predictions",
    )
    parser.add_argument(
        "--no-priors",
        action="store_true",
        help="Disable preseason priors",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sweep (grid search over alpha, decay, hfa)",
    )
    parser.add_argument(
        "--use-efm",
        action="store_true",
        help="Use Efficiency Foundation Model instead of margin-based ridge",
    )
    parser.add_argument(
        "--efm-efficiency-weight",
        type=float,
        default=0.54,
        help="EFM: weight for success rate component (default: 0.54)",
    )
    parser.add_argument(
        "--efm-explosiveness-weight",
        type=float,
        default=0.36,
        help="EFM: weight for IsoPPP component (default: 0.36)",
    )
    parser.add_argument(
        "--efm-turnover-weight",
        type=float,
        default=0.10,
        help="EFM: weight for turnover margin component (default: 0.10)",
    )
    parser.add_argument(
        "--no-asymmetric-garbage",
        action="store_true",
        help="EFM: disable asymmetric garbage time (penalize both teams equally in garbage time)",
    )
    parser.add_argument(
        "--fcs-penalty-elite",
        type=float,
        default=18.0,
        help="EFM: points for elite FCS teams (default: 18.0)",
    )
    parser.add_argument(
        "--fcs-penalty-standard",
        type=float,
        default=32.0,
        help="EFM: points for standard FCS teams (default: 32.0)",
    )
    parser.add_argument(
        "--no-portal",
        action="store_true",
        help="Disable transfer portal adjustment in preseason priors",
    )
    parser.add_argument(
        "--portal-scale",
        type=float,
        default=0.15,
        help="How much to weight transfer portal impact (default: 0.15)",
    )
    parser.add_argument(
        "--opening-line",
        action="store_true",
        help="Use opening lines for ATS calculation instead of closing lines",
    )

    args = parser.parse_args()

    if args.sweep:
        sweep_df = run_sweep(
            years=args.years,
            start_week=args.start_week,
            use_priors=not args.no_priors,
            use_efm=args.use_efm,
        )
        print("\n" + "=" * 80)
        print(f"SWEEP RESULTS ({'EFM' if args.use_efm else 'Ridge'}) - sorted by ATS%")
        print("=" * 80)
        print(sweep_df.to_string(index=False))

        if args.output:
            sweep_df.to_csv(args.output, index=False)
            print(f"\nSweep results saved to {args.output}")
        return

    model_type = "EFM" if args.use_efm else "Ridge"
    # Print full config for transparency (Rule 8: Parameter Synchronization)
    print("\n" + "=" * 60)
    print("BACKTEST CONFIGURATION")
    print("=" * 60)
    print(f"  Model:              {model_type}")
    print(f"  Years:              {args.years}")
    print(f"  Start week:         {args.start_week}")
    print(f"  ATS line type:      {'opening' if args.opening_line else 'closing'}")
    print(f"  Ridge alpha:        {args.alpha}")
    print(f"  Preseason priors:   {'disabled' if args.no_priors else 'enabled'}")
    print(f"  Transfer portal:    {'disabled' if args.no_portal else f'enabled (scale={args.portal_scale})'}")
    if args.use_efm:
        print(f"  HFA:                team-specific (fallback={args.hfa})")
        print(f"  EFM weights:        SR={args.efm_efficiency_weight}, IsoPPP={args.efm_explosiveness_weight}, TO={args.efm_turnover_weight}")
        print(f"  Asymmetric GT:      {not args.no_asymmetric_garbage}")
        print(f"  FCS penalties:      elite={args.fcs_penalty_elite}, standard={args.fcs_penalty_standard}")
    else:
        print(f"  HFA:                {args.hfa} (flat)")
        print(f"  Decay:              {args.decay}")
        print(f"  Margin cap:         {args.margin_cap}")
        print(f"  TO scrub factor:    {args.to_scrub_factor}")
    print("=" * 60 + "\n")

    results = run_backtest(
        years=args.years,
        start_week=args.start_week,
        ridge_alpha=args.alpha,
        use_priors=not args.no_priors,
        hfa_value=args.hfa,
        decay_rate=args.decay,
        prior_weight=args.prior_weight,
        margin_cap=args.margin_cap,
        to_scrub_factor=args.to_scrub_factor,
        use_efm=args.use_efm,
        efm_efficiency_weight=args.efm_efficiency_weight,
        efm_explosiveness_weight=args.efm_explosiveness_weight,
        efm_turnover_weight=args.efm_turnover_weight,
        efm_asymmetric_garbage=not args.no_asymmetric_garbage,
        fcs_penalty_elite=args.fcs_penalty_elite,
        fcs_penalty_standard=args.fcs_penalty_standard,
        use_portal=not args.no_portal,
        portal_scale=args.portal_scale,
        use_opening_line=args.opening_line,
    )

    print_results(results)

    # Save to CSV if requested
    if args.output:
        results["predictions"].to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
