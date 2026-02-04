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
from config.dtypes import optimize_dtypes
from src.api.cfbd_client import CFBDClient
from src.data.processors import DataProcessor, RecencyWeighter
from src.models.efficiency_foundation_model import (
    EfficiencyFoundationModel,
    clear_ridge_cache,
    get_ridge_cache_stats,
)
from src.models.preseason_priors import PreseasonPriors
from src.adjustments.home_field import HomeFieldAdvantage
from src.adjustments.situational import SituationalAdjuster
from src.adjustments.travel import TravelAdjuster
from src.adjustments.altitude import AltitudeAdjuster
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

    # Fetch regular season games (weeks 1-15)
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

    # Fetch postseason games (bowl games and playoffs)
    try:
        postseason_games = client.get_games(year=year, season_type="postseason")
        postseason_count = 0
        for game in postseason_games:
            if game.home_points is None:
                continue
            # Assign postseason games to week 16+ based on start_date
            # CFBD labels all postseason as "week 1", so we remap
            games.append({
                "id": game.id,
                "year": year,
                "week": 16,  # All bowl games go to week 16 for simplicity
                "start_date": game.start_date,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "home_points": game.home_points,
                "away_points": game.away_points,
                "neutral_site": game.neutral_site or True,  # Bowl games are usually neutral
            })
            postseason_count += 1
        if postseason_count > 0:
            successful_weeks.append(16)
            logger.info(f"Fetched {postseason_count} postseason games for {year}")
    except Exception as e:
        logger.warning(f"Failed to fetch postseason games for {year}: {e}")

    # Log fetch summary
    if failed_weeks:
        logger.warning(
            f"Games fetch for {year}: {len(successful_weeks)} weeks OK, "
            f"{len(failed_weeks)} weeks FAILED: {[w for w, _ in failed_weeks]}"
        )
    else:
        logger.debug(f"Games fetch for {year}: all {len(successful_weeks)} weeks OK")

    # Fetch betting lines (regular season + postseason)
    # Prefer DraftKings for consistency, fall back to any available
    preferred_providers = ["DraftKings", "ESPN Bet", "Bovada"]

    def process_betting_lines(lines_list):
        """Process betting lines and append to betting list."""
        for game_lines in lines_list:
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
                betting.append({
                    "game_id": game_lines.id,
                    "home_team": game_lines.home_team,
                    "away_team": game_lines.away_team,
                    "spread_close": selected_line.spread,
                    "spread_open": selected_line.spread_open if selected_line.spread_open is not None else selected_line.spread,
                    "over_under": selected_line.over_under,
                    "provider": selected_line.provider,
                })

    try:
        # Regular season betting lines
        regular_lines = client.get_betting_lines(year, season_type="regular")
        process_betting_lines(regular_lines)

        # Postseason betting lines
        postseason_lines = client.get_betting_lines(year, season_type="postseason")
        process_betting_lines(postseason_lines)
        if postseason_lines:
            logger.info(f"Fetched {len([l for l in postseason_lines if l.lines])} postseason betting lines for {year}")
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
        Tuple of (early_down_plays_df, turnover_plays_df, efficiency_plays_df, st_plays_df) as Polars DataFrames
    """
    early_down_plays = []
    turnover_plays = []
    efficiency_plays = []  # All scrimmage plays with PPA for efficiency model
    st_plays = []  # Special teams plays (FG, Punt, Kickoff) for ST model
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

                # Collect special teams plays (FG, Punt, Kickoff) for ST model
                if any(st in play_type for st in ["Field Goal", "Punt", "Kickoff"]):
                    st_plays.append({
                        "week": week,
                        "game_id": play.game_id,
                        "offense": play.offense,
                        "defense": play.defense,
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

    # Fetch postseason plays (bowl games and playoffs)
    try:
        postseason_plays = client.get_plays(year, week=1, season_type="postseason")
        postseason_play_count = 0
        for play in postseason_plays:
            play_type = play.play_type or ""

            # Collect turnover plays
            if play_type in TURNOVER_PLAY_TYPES:
                turnover_plays.append({
                    "week": 16,  # Map to week 16 for consistency
                    "game_id": play.game_id,
                    "offense": play.offense,
                    "defense": play.defense,
                    "play_type": play_type,
                })

            # Collect special teams plays
            if any(st in play_type for st in ["Field Goal", "Punt", "Kickoff"]):
                st_plays.append({
                    "week": 16,
                    "game_id": play.game_id,
                    "offense": play.offense,
                    "defense": play.defense,
                    "play_type": play_type,
                    "play_text": play.play_text,
                })

            # Collect scrimmage plays with PPA
            if (play.ppa is not None and
                play.down is not None and
                play_type in SCRIMMAGE_PLAY_TYPES and
                play.distance is not None and play.distance >= 0):
                efficiency_plays.append({
                    "week": 16,
                    "game_id": play.game_id,
                    "down": play.down,
                    "distance": play.distance,
                    "yards_gained": play.yards_gained or 0,
                    "play_type": play_type,
                    "play_text": play.play_text,
                    "offense": play.offense,
                    "defense": play.defense,
                    "period": play.period,
                    "ppa": play.ppa,
                    "yards_to_goal": play.yards_to_goal,
                    "offense_score": play.offense_score or 0,
                    "defense_score": play.defense_score or 0,
                    "home_team": play.home,
                })
                postseason_play_count += 1

        if postseason_play_count > 0:
            successful_weeks.append(16)
            logger.info(f"Fetched {postseason_play_count} postseason efficiency plays for {year}")
    except Exception as e:
        logger.warning(f"Failed to fetch postseason plays for {year}: {e}")

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
        f"{len(efficiency_plays)} efficiency, {len(st_plays)} ST plays for {year}"
    )
    return (
        pl.DataFrame(early_down_plays),
        pl.DataFrame(turnover_plays),
        pl.DataFrame(efficiency_plays),
        pl.DataFrame(st_plays),
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
    games_df: pl.DataFrame,
    efficiency_plays_df: pl.DataFrame,
    fbs_teams: set[str],
    start_week: int = 1,
    end_week: Optional[int] = None,
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
    st_plays_df: Optional[pl.DataFrame] = None,
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
        st_plays_df: Field goal plays dataframe for FG efficiency calculation (Polars DataFrame)

    Returns:
        List of prediction result dictionaries
    """
    results = []
    max_week = games_df["week"].max()

    # Apply end_week limit if specified
    if end_week is not None:
        max_week = min(max_week, end_week)

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
        # P3.4: Apply optimized dtypes for memory efficiency
        train_plays_pd = optimize_dtypes(train_plays_pl.to_pandas())
        train_games_pd = optimize_dtypes(games_df.filter(pl.col("week") < pred_week).to_pandas())

        # DATA LEAKAGE GUARD: Verify no future data in training set
        if "week" in train_plays_pd.columns:
            max_train_week = train_plays_pd["week"].max()
            assert max_train_week < pred_week, (
                f"DATA LEAKAGE: Training plays include week {max_train_week} "
                f"but predicting week {pred_week}. Training data must be < pred_week."
            )
        if "week" in train_games_pd.columns:
            max_train_game_week = train_games_pd["week"].max()
            assert max_train_game_week < pred_week, (
                f"DATA LEAKAGE: Training games include week {max_train_game_week} "
                f"but predicting week {pred_week}. Training data must be < pred_week."
            )

        # Build EFM model
        efm = EfficiencyFoundationModel(
            ridge_alpha=ridge_alpha,
            efficiency_weight=efficiency_weight,
            explosiveness_weight=explosiveness_weight,
            turnover_weight=turnover_weight,
            garbage_time_weight=garbage_time_weight,
            asymmetric_garbage=asymmetric_garbage,
        )

        efm.calculate_ratings(train_plays_pd, train_games_pd, max_week=pred_week - 1, season=year)

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
        # DETERMINISM: Use (rating, team_name) as sort key for stable ordering when ratings tie
        sorted_teams = sorted(team_ratings.items(), key=lambda x: (-x[1], x[0]))
        rankings = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}

        # Calculate finishing drives with regression to mean
        finishing = FinishingDrivesModel(regress_to_mean=True, prior_strength=10)
        if "yards_to_goal" in train_plays_pd.columns:
            finishing.calculate_all_from_plays(train_plays_pd, max_week=pred_week - 1)

        # Calculate FG efficiency ratings from FG plays
        special_teams = SpecialTeamsModel()
        if st_plays_df is not None and len(st_plays_df) > 0:
            # Filter to ST plays before this week (Polars filtering)
            train_st_pl = st_plays_df.filter(pl.col("week") < pred_week)
            if len(train_st_pl) > 0:
                # P3.4: Apply optimized dtypes for memory efficiency
                train_st_pd = optimize_dtypes(train_st_pl.to_pandas())
                # DATA LEAKAGE GUARD: Verify ST plays are properly filtered
                if "week" in train_st_pd.columns:
                    max_st_week = train_st_pd["week"].max()
                    assert max_st_week < pred_week, (
                        f"DATA LEAKAGE: ST plays include week {max_st_week} "
                        f"but predicting week {pred_week}."
                    )
                # Calculate all ST ratings (FG + Punt + Kickoff)
                special_teams.calculate_all_st_ratings_from_plays(
                    train_st_pd, max_week=pred_week - 1
                )
                # Integrate special teams ratings into EFM for O/D/ST breakdown (diagnostic only)
                for team, st_rating in special_teams.team_ratings.items():
                    efm.set_special_teams_rating(team, st_rating.overall_rating)

        # Build spread generator with EFM ratings
        situational = SituationalAdjuster()
        spread_gen = SpreadGenerator(
            ratings=team_ratings,
            finishing_drives=finishing,  # Regressed RZ efficiency
            special_teams=special_teams,  # FG/Punt/Kickoff efficiency differential
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
        # P3.4: Apply optimized dtypes for memory efficiency
        games_df_pd = optimize_dtypes(games_df.to_pandas())

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
                    # P2.11: Track correlated adjustment stack (HFA + travel + altitude)
                    "correlated_stack": pred.components.correlated_stack,
                    "hfa": pred.components.home_field,
                    "travel": pred.components.travel,
                    "altitude": pred.components.altitude,
                    # P3.4: Track ratings for sanity check
                    "home_rating": team_ratings.get(game["home_team"], 0.0),
                    "away_rating": team_ratings.get(game["away_team"], 0.0),
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

        # Get both open and close spreads for P3.4 sanity report
        spread_open = line_data.get("spread_open", vegas_spread)
        spread_close = line_data.get("spread_close", vegas_spread)

        # Calculate CLV (Closing Line Value)
        # CLV = how many points better we got vs closing line
        # If betting home: we want line to move toward home (closing more negative)
        # If betting away: we want line to move toward away (closing less negative)
        if spread_open is not None and spread_close is not None:
            if model_pick_home:
                # We bet home at spread_open, it closed at spread_close
                # CLV = spread_open - spread_close (positive if line moved toward home)
                clv = spread_open - spread_close
            else:
                # We bet away at spread_open, it closed at spread_close
                # CLV = spread_close - spread_open (positive if line moved toward away)
                clv = spread_close - spread_open
        else:
            clv = None

        results.append({
            **pred,
            "vegas_spread": vegas_spread,
            "spread_open": spread_open,
            "spread_close": spread_close,
            "edge": abs(edge),
            "pick": "HOME" if model_pick_home else "AWAY",
            "ats_win": ats_win,
            "ats_push": ats_push,
            "clv": clv,
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

    # MAE vs Closing Spread (how close are we to the efficient market?)
    if ats_df is not None and len(ats_df) > 0 and "spread_close" in ats_df.columns:
        # Convert our spread to Vegas convention and compare to closing line
        # Our model: positive = home favored; Vegas: negative = home favored
        ats_with_close = ats_df[ats_df["spread_close"].notna()].copy()
        if len(ats_with_close) > 0:
            model_spread_vegas = -ats_with_close["predicted_spread"]
            closing_spread = ats_with_close["spread_close"]
            mae_vs_close = (model_spread_vegas - closing_spread).abs().mean()
            metrics["mae_vs_close"] = mae_vs_close

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


def analyze_stack_diagnostics(predictions_df: pd.DataFrame) -> dict:
    """Analyze adjustment stack patterns for P2.11 diagnostics.

    Evaluates whether correlated adjustments (HFA + travel + altitude) show
    systematic error patterns, especially for high-stack games.

    Args:
        predictions_df: DataFrame with predictions including correlated_stack column

    Returns:
        Dictionary with stack analysis results
    """
    if "correlated_stack" not in predictions_df.columns:
        return {"error": "No correlated_stack column in predictions"}

    HIGH_STACK_THRESHOLD = 5.0
    EXTREME_STACK_THRESHOLD = 7.0

    df = predictions_df.copy()
    df["is_high_stack"] = df["correlated_stack"].abs() > HIGH_STACK_THRESHOLD
    df["is_extreme_stack"] = df["correlated_stack"].abs() > EXTREME_STACK_THRESHOLD

    high_stack = df[df["is_high_stack"]]
    low_stack = df[~df["is_high_stack"]]

    result = {
        "total_games": len(df),
        "stack_mean": df["correlated_stack"].mean(),
        "stack_std": df["correlated_stack"].std(),
        "stack_min": df["correlated_stack"].min(),
        "stack_max": df["correlated_stack"].max(),
        "high_stack_count": len(high_stack),
        "high_stack_pct": len(high_stack) / len(df) * 100 if len(df) > 0 else 0,
        "extreme_stack_count": len(df[df["is_extreme_stack"]]),
    }

    # Compare errors between high-stack and low-stack games
    if len(high_stack) > 0:
        result["high_stack_mae"] = high_stack["abs_error"].mean()
        result["high_stack_me"] = high_stack["error"].mean()  # Positive = over-predicting home

    if len(low_stack) > 0:
        result["low_stack_mae"] = low_stack["abs_error"].mean()
        result["low_stack_me"] = low_stack["error"].mean()

    # Check for systematic bias in high-stack games
    if len(high_stack) >= 10 and "high_stack_me" in result:
        me_high = result["high_stack_me"]
        if me_high > 2.0:
            result["warning"] = (
                f"High-stack games show positive mean error ({me_high:.2f}), "
                "suggesting over-prediction of home advantage. Consider capping adjustments."
            )
        elif me_high < -2.0:
            result["warning"] = (
                f"High-stack games show negative mean error ({me_high:.2f}), "
                "suggesting under-prediction of home advantage."
            )
        else:
            result["status"] = "OK - no systematic bias detected in high-stack games"

    return result


def log_stack_diagnostics(predictions_df: pd.DataFrame) -> None:
    """Log adjustment stack diagnostics summary (P2.11).

    Args:
        predictions_df: DataFrame with predictions
    """
    analysis = analyze_stack_diagnostics(predictions_df)

    if "error" in analysis:
        logger.warning(f"Stack analysis failed: {analysis['error']}")
        return

    logger.info(
        f"P2.11 Stack diagnostics: {analysis['total_games']} games, "
        f"mean stack={analysis['stack_mean']:.2f}, "
        f"range=[{analysis['stack_min']:.1f}, {analysis['stack_max']:.1f}]"
    )

    if analysis["high_stack_count"] > 0:
        logger.info(
            f"  High-stack games (>{5}): {analysis['high_stack_count']} "
            f"({analysis['high_stack_pct']:.1f}%), "
            f"MAE={analysis.get('high_stack_mae', 0):.2f}, "
            f"ME={analysis.get('high_stack_me', 0):+.2f}"
        )

    if analysis.get("low_stack_mae"):
        logger.info(
            f"  Low-stack games: MAE={analysis['low_stack_mae']:.2f}, "
            f"ME={analysis['low_stack_me']:+.2f}"
        )

    if analysis["extreme_stack_count"] > 0:
        logger.warning(
            f"  Extreme stack games (>{7}): {analysis['extreme_stack_count']} - "
            "consider investigating"
        )

    if "warning" in analysis:
        logger.warning(f"  ⚠️  {analysis['warning']}")
    elif "status" in analysis:
        logger.info(f"  ✓ {analysis['status']}")


def calculate_clv_report(ats_df: pd.DataFrame) -> dict:
    """Calculate Closing Line Value (CLV) by edge bucket.

    CLV measures how many points better our entry was vs the closing line.
    Positive CLV = we beat the closing line (strong indicator of real edge).

    Args:
        ats_df: DataFrame with ATS results including 'clv', 'edge' columns

    Returns:
        Dict with CLV statistics by edge bucket
    """
    if ats_df is None or ats_df.empty or "clv" not in ats_df.columns:
        return {"error": "No CLV data available"}

    # Filter to games with CLV data (both open and close spreads available)
    df = ats_df[ats_df["clv"].notna()].copy()

    if len(df) == 0:
        return {"error": "No games with both opening and closing lines"}

    results = {
        "total_games_with_clv": len(df),
        "buckets": {},
    }

    # Calculate CLV for different edge buckets
    buckets = [
        ("all", df["edge"] >= 0),
        ("3+", df["edge"] >= 3),
        ("5+", df["edge"] >= 5),
        ("7+", df["edge"] >= 7),
    ]

    for name, mask in buckets:
        subset = df[mask]
        if len(subset) == 0:
            continue

        mean_clv = subset["clv"].mean()
        std_clv = subset["clv"].std()
        positive_clv_pct = (subset["clv"] > 0).mean() * 100
        ats_wins = subset["ats_win"].sum()
        ats_total = len(subset) - subset["ats_push"].sum()
        ats_pct = ats_wins / ats_total * 100 if ats_total > 0 else 0

        results["buckets"][name] = {
            "n": len(subset),
            "mean_clv": mean_clv,
            "std_clv": std_clv,
            "positive_clv_pct": positive_clv_pct,
            "ats_pct": ats_pct,
        }

    return results


def get_phase(week: int) -> str:
    """Categorize a week into a season phase.

    Args:
        week: Week number

    Returns:
        Phase name: 'Phase 1 (Calibration)', 'Phase 2 (Core)', or 'Phase 3 (Postseason)'
    """
    if week <= 3:
        return "Phase 1 (Calibration)"
    elif week <= 15:
        return "Phase 2 (Core)"
    else:
        return "Phase 3 (Postseason)"


def calculate_phase_metrics(df: pd.DataFrame, phase_col: str = "phase") -> pd.DataFrame:
    """Calculate metrics by phase.

    Args:
        df: DataFrame with predictions/ATS results
        phase_col: Column containing phase labels

    Returns:
        DataFrame with metrics per phase
    """
    results = []

    # Phase definitions with week ranges
    phase_info = [
        ("Phase 1 (Calibration)", "1-3"),
        ("Phase 2 (Core)", "4-15"),
        ("Phase 3 (Postseason)", "16+"),
    ]

    for phase, weeks in phase_info:
        phase_df = df[df[phase_col] == phase]

        if len(phase_df) == 0:
            continue

        metrics = {
            "Phase": phase,
            "Weeks": weeks,
            "Games": len(phase_df),
        }

        # MAE and RMSE (if abs_error column exists)
        if "abs_error" in phase_df.columns:
            metrics["MAE"] = phase_df["abs_error"].mean()
            metrics["RMSE"] = np.sqrt((phase_df["error"] ** 2).mean())

        # MAE vs Closing (if we have both predicted_spread and spread_close)
        if "predicted_spread" in phase_df.columns and "spread_close" in phase_df.columns:
            with_close = phase_df[phase_df["spread_close"].notna()]
            if len(with_close) > 0:
                model_vegas = -with_close["predicted_spread"]
                mae_vs_close = (model_vegas - with_close["spread_close"]).abs().mean()
                metrics["MAE vs Close"] = mae_vs_close

        # ATS metrics (if ats_win column exists)
        if "ats_win" in phase_df.columns:
            wins = phase_df["ats_win"].sum()
            pushes = phase_df["ats_push"].sum()
            losses = len(phase_df) - wins - pushes
            total = wins + losses
            metrics["ATS Record"] = f"{int(wins)}-{int(losses)}-{int(pushes)}"
            metrics["ATS %"] = wins / total * 100 if total > 0 else 0

            # 3+ edge subset
            edge_3 = phase_df[phase_df["edge"] >= 3]
            if len(edge_3) > 0:
                e3_wins = edge_3["ats_win"].sum()
                e3_losses = len(edge_3) - e3_wins - edge_3["ats_push"].sum()
                e3_total = e3_wins + e3_losses
                metrics["3+ Edge"] = f"{int(e3_wins)}-{int(e3_losses)} ({e3_wins/e3_total*100:.1f}%)" if e3_total > 0 else "N/A"

            # 5+ edge subset
            edge_5 = phase_df[phase_df["edge"] >= 5]
            if len(edge_5) > 0:
                e5_wins = edge_5["ats_win"].sum()
                e5_losses = len(edge_5) - e5_wins - edge_5["ats_push"].sum()
                e5_total = e5_wins + e5_losses
                metrics["5+ Edge"] = f"{int(e5_wins)}-{int(e5_losses)} ({e5_wins/e5_total*100:.1f}%)" if e5_total > 0 else "N/A"

        # CLV (if clv column exists)
        if "clv" in phase_df.columns:
            clv_df = phase_df[phase_df["clv"].notna()]
            if len(clv_df) > 0:
                metrics["Mean CLV"] = clv_df["clv"].mean()

        results.append(metrics)

    return pd.DataFrame(results)


def print_phase_report(ats_df: pd.DataFrame, predictions_df: pd.DataFrame = None) -> None:
    """Print phase-by-phase performance report.

    Args:
        ats_df: DataFrame with ATS results (has week column)
        predictions_df: DataFrame with prediction errors (optional, for MAE)
    """
    if ats_df is None or ats_df.empty:
        print("\nPhase Report: No ATS data available")
        return

    # Add phase column
    df = ats_df.copy()
    df["phase"] = df["week"].apply(get_phase)

    # If predictions_df provided, merge in error columns
    if predictions_df is not None and not predictions_df.empty:
        if "abs_error" not in df.columns and "abs_error" in predictions_df.columns:
            # Merge on game_id if available
            if "game_id" in df.columns and "game_id" in predictions_df.columns:
                error_cols = predictions_df[["game_id", "abs_error", "error"]].drop_duplicates()
                df = df.merge(error_cols, on="game_id", how="left")

    phase_metrics = calculate_phase_metrics(df)

    if phase_metrics.empty:
        print("\nPhase Report: No phase data available")
        return

    print("\n" + "=" * 90)
    print("PHASE-BY-PHASE PERFORMANCE")
    print("=" * 90)
    print()

    # Format and print table
    display_cols = ["Phase", "Weeks", "Games", "MAE", "MAE vs Close", "ATS %", "3+ Edge", "5+ Edge", "Mean CLV"]
    available_cols = [c for c in display_cols if c in phase_metrics.columns]

    # Format numeric columns
    for col in ["MAE", "MAE vs Close", "ATS %", "Mean CLV"]:
        if col in phase_metrics.columns:
            phase_metrics[col] = phase_metrics[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )

    print(phase_metrics[available_cols].to_string(index=False))
    print()


def print_clv_report(ats_df: pd.DataFrame) -> None:
    """Print CLV report to console.

    Args:
        ats_df: DataFrame with ATS results
    """
    report = calculate_clv_report(ats_df)

    if "error" in report:
        print(f"\nCLV Report: {report['error']}")
        return

    print(f"\nClosing Line Value (CLV) Report:")
    print(f"  Games with open+close lines: {report['total_games_with_clv']}")
    print()
    print(f"  {'Edge':<8} {'N':>6} {'Mean CLV':>10} {'CLV > 0':>10} {'ATS %':>8}")
    print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")

    for name, stats in report["buckets"].items():
        print(
            f"  {name:<8} {stats['n']:>6} {stats['mean_clv']:>+10.2f} "
            f"{stats['positive_clv_pct']:>9.1f}% {stats['ats_pct']:>7.1f}%"
        )

    # Interpretation
    print()
    clv_3plus = report["buckets"].get("3+", {}).get("mean_clv", 0)
    clv_5plus = report["buckets"].get("5+", {}).get("mean_clv", 0)

    if clv_5plus > 0.5:
        print("  ✓ Strong positive CLV at 5+ edge - edge appears REAL")
    elif clv_5plus > 0:
        print("  ~ Slight positive CLV at 5+ edge - edge may be real")
    elif clv_5plus > -0.5:
        print("  ~ Near-zero CLV at 5+ edge - edge is marginal")
    else:
        print("  ⚠ Negative CLV at 5+ edge - edge may be illusory (market moves against us)")


def print_data_sanity_report(season_data: dict, years: list[int]) -> None:
    """Print sanity report after data fetch (P3.4).

    Reports:
    - Expected vs actual game counts per year
    - FBS team counts
    - Betting line coverage rate
    - Week coverage

    Args:
        season_data: Dict from fetch_all_season_data
        years: List of years fetched
    """
    print("\n" + "=" * 60)
    print("DATA SANITY REPORT")
    print("=" * 60)

    # Expected games per season (roughly 850-900 for FBS)
    EXPECTED_GAMES_PER_YEAR = 870

    for year in years:
        games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, st_plays_df = season_data[year]

        n_games = len(games_df)
        n_betting = len(betting_df)
        n_efficiency = len(efficiency_plays_df)
        n_fbs = len(fbs_teams)

        # Game count check
        game_pct = n_games / EXPECTED_GAMES_PER_YEAR * 100
        game_status = "✓" if game_pct >= 95 else "⚠" if game_pct >= 80 else "✗"

        # Betting coverage
        betting_coverage = n_betting / n_games * 100 if n_games > 0 else 0
        betting_status = "✓" if betting_coverage >= 90 else "⚠" if betting_coverage >= 70 else "✗"

        # Week coverage
        weeks_with_games = set(games_df["week"].unique().to_list()) if n_games > 0 else set()
        weeks_with_plays = set(efficiency_plays_df["week"].unique().to_list()) if n_efficiency > 0 else set()
        expected_weeks = set(range(1, 16))
        missing_game_weeks = expected_weeks - weeks_with_games
        missing_play_weeks = expected_weeks - weeks_with_plays

        print(f"\n{year}:")
        print(f"  {game_status} Games: {n_games:,} (expected ~{EXPECTED_GAMES_PER_YEAR}, {game_pct:.0f}%)")
        print(f"  {betting_status} Betting lines: {n_betting:,} ({betting_coverage:.0f}% coverage)")
        print(f"    FBS teams: {n_fbs}")
        print(f"    Efficiency plays: {n_efficiency:,}")
        if missing_game_weeks:
            print(f"    ⚠ Missing game weeks: {sorted(missing_game_weeks)}")
        if missing_play_weeks:
            print(f"    ⚠ Missing play weeks: {sorted(missing_play_weeks)}")
        if priors:
            print(f"    Preseason priors: {len(priors.preseason_ratings)} teams")

    print()


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
        Dict mapping year to (games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, st_plays_df)
    """
    client = CFBDClient()
    season_data = {}

    for year in years:
        logger.info(f"\nFetching data for {year}...")

        games_df, betting_df = fetch_season_data(client, year)
        logger.info(f"Loaded {len(games_df)} games, {len(betting_df)} betting lines")

        plays_df, turnover_plays_df, efficiency_plays_df, st_plays_df = fetch_season_plays(client, year)

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

        season_data[year] = (games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, st_plays_df)

    return season_data


def run_backtest(
    years: list[int],
    start_week: int = 1,
    end_week: Optional[int] = None,
    ridge_alpha: float = 50.0,
    use_priors: bool = True,
    hfa_value: float = 2.5,
    prior_weight: int = 8,
    season_data: Optional[dict] = None,
    efficiency_weight: float = 0.54,
    explosiveness_weight: float = 0.36,
    turnover_weight: float = 0.10,
    garbage_time_weight: float = 0.1,
    asymmetric_garbage: bool = True,
    fcs_penalty_elite: float = 18.0,
    fcs_penalty_standard: float = 32.0,
    use_portal: bool = True,
    portal_scale: float = 0.15,
    use_opening_line: bool = False,
    clear_cache: bool = True,
) -> dict:
    """Run full backtest across specified years using EFM.

    Args:
        years: List of years to backtest
        start_week: First week to start predictions
        ridge_alpha: Ridge regression alpha for EFM opponent adjustment
        use_priors: Whether to use preseason priors
        hfa_value: Base home field advantage in points
        prior_weight: games_for_full_weight for preseason blending
        season_data: Pre-fetched season data (for sweep caching)
        efficiency_weight: EFM success rate weight (default 0.54)
        explosiveness_weight: EFM IsoPPP weight (default 0.36)
        turnover_weight: EFM turnover margin weight (default 0.10)
        garbage_time_weight: EFM garbage time play weight (default 0.1)
        asymmetric_garbage: Only penalize trailing team in garbage time (default True)
        fcs_penalty_elite: Points for elite FCS teams (default 18.0)
        fcs_penalty_standard: Points for standard FCS teams (default 32.0)
        use_portal: Whether to incorporate transfer portal into preseason priors
        portal_scale: How much to weight portal impact (default 0.15)
        use_opening_line: If True, use opening lines for ATS; if False, use closing lines (default)
        clear_cache: Whether to clear ridge adjustment cache at start (default True)

    Returns:
        Dictionary with backtest results
    """
    # Optionally clear ridge adjustment cache at start of backtest
    # This ensures deterministic results across runs while enabling
    # within-run caching for repeated (season, week, metric) lookups
    # Set clear_cache=False when called from sweep to preserve cache across runs
    if clear_cache:
        clear_ridge_cache()

    # Fetch data if not pre-cached
    if season_data is None:
        season_data = fetch_all_season_data(
            years,
            use_priors=use_priors,
            use_portal=use_portal,
            portal_scale=portal_scale,
        )

    # Build team records for trajectory calculation (need ~4 years before earliest year)
    client = CFBDClient()
    trajectory_years = list(range(min(years) - 4, max(years) + 1))
    team_records = build_team_records(client, trajectory_years)
    logger.info(f"Built team records for trajectory ({len(team_records)} teams, years {trajectory_years[0]}-{trajectory_years[-1]})")

    all_predictions = []
    all_ats = []

    for year in years:
        logger.info(f"\nBacktesting {year} season...")

        games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, st_plays_df = season_data[year]

        # Walk-forward predictions using EFM
        predictions = walk_forward_predict(
            games_df,
            efficiency_plays_df,
            fbs_teams,
            start_week,
            end_week=end_week,
            preseason_priors=priors,
            hfa_value=hfa_value,
            prior_weight=prior_weight,
            ridge_alpha=ridge_alpha,
            efficiency_weight=efficiency_weight,
            explosiveness_weight=explosiveness_weight,
            turnover_weight=turnover_weight,
            garbage_time_weight=garbage_time_weight,
            asymmetric_garbage=asymmetric_garbage,
            team_records=team_records,
            year=year,
            fcs_penalty_elite=fcs_penalty_elite,
            fcs_penalty_standard=fcs_penalty_standard,
            st_plays_df=st_plays_df,
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

    # Log ridge adjustment cache statistics
    cache_stats = get_ridge_cache_stats()
    logger.info(
        f"Ridge cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses, "
        f"{cache_stats['size']} entries (hit rate: {cache_stats['hit_rate']:.1%})"
    )

    return {
        "predictions": predictions_df,
        "ats_results": ats_df,
        "metrics": metrics,
    }


def run_sweep(
    years: list[int],
    start_week: int = 1,
    use_priors: bool = True,
) -> pd.DataFrame:
    """Run grid search over EFM parameters.

    Tests combinations of alpha, hfa, and efficiency_weight to find optimal settings.

    Args:
        years: List of years to backtest
        start_week: First week to start predictions
        use_priors: Whether to use preseason priors

    Returns:
        DataFrame with results for each parameter combination
    """
    from itertools import product

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

    # Clear ridge cache once at start of sweep (cache is keyed by alpha, so
    # different parameter combinations won't collide)
    clear_ridge_cache()

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
                efficiency_weight=eff_wt,
                explosiveness_weight=0.90 - eff_wt,  # 0.90 = 1.0 - turnover_weight
                turnover_weight=turnover_weight,
                clear_cache=False,  # Preserve cache across sweep iterations
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

    df = pd.DataFrame(sweep_results)
    df = df.sort_values("ATS%", ascending=False).reset_index(drop=True)

    # Log ridge cache statistics for sweep
    cache_stats = get_ridge_cache_stats()
    print(
        f"\nRidge cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses, "
        f"{cache_stats['size']} entries (hit rate: {cache_stats['hit_rate']:.1%})"
    )

    return df


def print_results(
    results: dict,
    ats_df: pd.DataFrame = None,
    diagnostics: bool = True,
) -> None:
    """Print backtest results to console.

    Args:
        results: Dictionary with backtest results from run_backtest
        ats_df: Optional ATS results DataFrame for sanity reporting
        diagnostics: Whether to print detailed diagnostics (stack, CLV, phase reports).
                    Set False for faster output in sweeps. Default True.
    """
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)

    metrics = results["metrics"]
    print(f"\nTotal games predicted: {metrics['total_games']}")
    print(f"\nError Metrics:")
    print(f"  MAE (vs actual):  {metrics['mae']:.2f} points")
    if "mae_vs_close" in metrics:
        print(f"  MAE (vs closing): {metrics['mae_vs_close']:.2f} points")
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

    # P3.7: Diagnostic reports are optional (skip with --no-diagnostics for faster sweeps)
    if diagnostics:
        # P2.11: Stack diagnostics (HFA + travel + altitude)
        if "correlated_stack" in predictions_df.columns:
            print(f"\nP2.11 Adjustment Stack Diagnostics:")
            log_stack_diagnostics(predictions_df)

        # CLV Report (Closing Line Value)
        if ats_df is not None and not ats_df.empty:
            print_clv_report(ats_df)

        # Phase-by-Phase Report
        if ats_df is not None and not ats_df.empty:
            print_phase_report(ats_df, predictions_df)

    # P3.4: Sanity Report (always run - lightweight validation)
    print_prediction_sanity_report(results, ats_df)


def print_prediction_sanity_report(results: dict, ats_df: pd.DataFrame = None) -> None:
    """Print prediction sanity metrics (P3.4).

    Reports:
    - Predictions per week
    - Line match rate
    - Open vs close line difference rate
    - Rating distribution stats

    Args:
        results: Dictionary with backtest results
        ats_df: Optional ATS results DataFrame
    """
    predictions_df = results["predictions"]
    if predictions_df.empty:
        return

    print("\n" + "-" * 50)
    print("SANITY CHECK")
    print("-" * 50)

    # Predictions per week
    print("\nPredictions per week:")
    week_counts = predictions_df.groupby("week").size()
    weeks_low = week_counts[week_counts < 30].index.tolist()
    weeks_high = week_counts[week_counts > 80].index.tolist()
    print(f"  Total: {len(predictions_df):,} predictions across {len(week_counts)} weeks")
    print(f"  Range: {week_counts.min()}-{week_counts.max()} per week")
    if weeks_low:
        print(f"  ⚠ Low weeks (<30): {weeks_low}")

    # Line match rate (from ATS results)
    if ats_df is not None and not ats_df.empty:
        matched_games = len(ats_df)
        total_predictions = len(predictions_df)
        match_rate = matched_games / total_predictions * 100 if total_predictions > 0 else 0
        match_status = "✓" if match_rate >= 95 else "⚠" if match_rate >= 80 else "✗"
        print(f"\nBetting line matching:")
        print(f"  {match_status} Match rate: {matched_games:,}/{total_predictions:,} ({match_rate:.1f}%)")

        # Open vs close line difference
        if "spread_open" in ats_df.columns and "spread_close" in ats_df.columns:
            line_diffs = (ats_df["spread_close"] - ats_df["spread_open"]).abs()
            lines_moved = (line_diffs > 0.5).sum()
            lines_moved_pct = lines_moved / len(ats_df) * 100
            avg_movement = line_diffs.mean()
            print(f"  Line movement: {lines_moved:,} games moved >0.5 pts ({lines_moved_pct:.1f}%)")
            print(f"  Avg movement: {avg_movement:.2f} pts")

    # Rating distribution (from predictions - home/away ratings)
    if "home_rating" in predictions_df.columns and "away_rating" in predictions_df.columns:
        all_ratings = pd.concat([predictions_df["home_rating"], predictions_df["away_rating"]])
        rating_mean = all_ratings.mean()
        rating_std = all_ratings.std()
        rating_min = all_ratings.min()
        rating_max = all_ratings.max()
        print(f"\nRating distribution:")
        print(f"  Mean: {rating_mean:.2f} (expected ~0)")
        print(f"  Std:  {rating_std:.2f} (expected ~12)")
        print(f"  Range: [{rating_min:.1f}, {rating_max:.1f}]")
        # Warn if mean is far from 0 or std is far from 12
        if abs(rating_mean) > 1.0:
            print(f"  ⚠ Mean is {abs(rating_mean):.2f} away from 0 - check normalization")
        if abs(rating_std - 12.0) > 2.0:
            print(f"  ⚠ Std is {abs(rating_std - 12.0):.2f} away from 12 - check scaling")

    # Spread distribution
    if "predicted_spread" in predictions_df.columns:
        spread_mean = predictions_df["predicted_spread"].mean()
        spread_std = predictions_df["predicted_spread"].std()
        spread_min = predictions_df["predicted_spread"].min()
        spread_max = predictions_df["predicted_spread"].max()
        print(f"\nSpread distribution:")
        print(f"  Mean: {spread_mean:+.2f} (expected ~+2-3 for HFA)")
        print(f"  Std:  {spread_std:.2f}")
        print(f"  Range: [{spread_min:.1f}, {spread_max:.1f}]")

    # Error distribution check
    if "error" in predictions_df.columns:
        mean_error = predictions_df["error"].mean()
        print(f"\nBias check:")
        print(f"  Mean error: {mean_error:+.2f} pts")
        if abs(mean_error) > 2.0:
            print(f"  ⚠ Systematic bias detected - mean error should be ~0")
        else:
            print(f"  ✓ No systematic bias (mean error within ±2 pts)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backtest CFB Power Ratings Model (EFM)")
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
        default=1,
        help="First week to start predictions (default: 1 for full season)",
    )
    parser.add_argument(
        "--end-week",
        type=int,
        default=None,
        help="Last week to predict (default: None = all weeks)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=50.0,
        help="Ridge regression alpha for opponent adjustment (default: 50.0)",
    )
    parser.add_argument(
        "--hfa",
        type=float,
        default=2.5,
        help="Base home field advantage in points (default: 2.5)",
    )
    parser.add_argument(
        "--prior-weight",
        type=int,
        default=8,
        help="Games for full preseason prior weight (default: 8)",
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
        help="Run parameter sweep (grid search over alpha, hfa, efficiency weight)",
    )
    parser.add_argument(
        "--efficiency-weight",
        type=float,
        default=0.54,
        help="Weight for success rate component (default: 0.54)",
    )
    parser.add_argument(
        "--explosiveness-weight",
        type=float,
        default=0.36,
        help="Weight for IsoPPP component (default: 0.36)",
    )
    parser.add_argument(
        "--turnover-weight",
        type=float,
        default=0.10,
        help="Weight for turnover margin component (default: 0.10)",
    )
    parser.add_argument(
        "--no-asymmetric-garbage",
        action="store_true",
        help="Disable asymmetric garbage time (penalize both teams equally)",
    )
    parser.add_argument(
        "--fcs-penalty-elite",
        type=float,
        default=18.0,
        help="Points for elite FCS teams (default: 18.0)",
    )
    parser.add_argument(
        "--fcs-penalty-standard",
        type=float,
        default=32.0,
        help="Points for standard FCS teams (default: 32.0)",
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
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip diagnostic reports (stack analysis, phase metrics, CLV). Faster for sweeps.",
    )

    args = parser.parse_args()

    if args.sweep:
        sweep_df = run_sweep(
            years=args.years,
            start_week=args.start_week,
            use_priors=not args.no_priors,
        )
        print("\n" + "=" * 80)
        print("SWEEP RESULTS (EFM) - sorted by ATS%")
        print("=" * 80)
        print(sweep_df.to_string(index=False))

        if args.output:
            sweep_df.to_csv(args.output, index=False)
            print(f"\nSweep results saved to {args.output}")
        return

    # Fetch data first for sanity reporting (P3.4)
    season_data = fetch_all_season_data(
        args.years,
        use_priors=not args.no_priors,
        use_portal=not args.no_portal,
        portal_scale=args.portal_scale,
    )

    # P3.4: Print data sanity report
    print_data_sanity_report(season_data, args.years)

    # Print full config for transparency
    print("\n" + "=" * 60)
    print("BACKTEST CONFIGURATION (EFM)")
    print("=" * 60)
    print(f"  Years:              {args.years}")
    print(f"  Week range:         {args.start_week} - {args.end_week if args.end_week else 'end'}")
    print(f"  ATS line type:      {'opening' if args.opening_line else 'closing'}")
    print(f"  Ridge alpha:        {args.alpha}")
    print(f"  Preseason priors:   {'disabled' if args.no_priors else 'enabled'}")
    print(f"  Transfer portal:    {'disabled' if args.no_portal else f'enabled (scale={args.portal_scale})'}")
    print(f"  HFA:                team-specific (fallback={args.hfa})")
    print(f"  EFM weights:        SR={args.efficiency_weight}, IsoPPP={args.explosiveness_weight}, TO={args.turnover_weight}")
    print(f"  Asymmetric GT:      {not args.no_asymmetric_garbage}")
    print(f"  FCS penalties:      elite={args.fcs_penalty_elite}, standard={args.fcs_penalty_standard}")
    print("=" * 60 + "\n")

    results = run_backtest(
        years=args.years,
        start_week=args.start_week,
        end_week=args.end_week,
        ridge_alpha=args.alpha,
        use_priors=not args.no_priors,
        hfa_value=args.hfa,
        prior_weight=args.prior_weight,
        efficiency_weight=args.efficiency_weight,
        explosiveness_weight=args.explosiveness_weight,
        turnover_weight=args.turnover_weight,
        asymmetric_garbage=not args.no_asymmetric_garbage,
        fcs_penalty_elite=args.fcs_penalty_elite,
        fcs_penalty_standard=args.fcs_penalty_standard,
        use_portal=not args.no_portal,
        portal_scale=args.portal_scale,
        use_opening_line=args.opening_line,
        season_data=season_data,  # Use pre-fetched data
    )

    # P3.4: Print results with ATS data for sanity report
    # P3.7: Skip detailed diagnostics if --no-diagnostics flag is set
    print_results(
        results,
        ats_df=results.get("ats_results"),
        diagnostics=not args.no_diagnostics,
    )

    # Save to CSV if requested
    if args.output:
        results["predictions"].to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
