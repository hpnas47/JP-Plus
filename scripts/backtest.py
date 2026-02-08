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
import os
import sys
from concurrent.futures import ProcessPoolExecutor
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
from src.models.efficiency_foundation_model import (
    EfficiencyFoundationModel,
    clear_ridge_cache,
    get_ridge_cache_stats,
)
from src.models.preseason_priors import PreseasonPriors
from src.adjustments.home_field import HomeFieldAdvantage
from src.adjustments.situational import SituationalAdjuster, HistoricalRankings, precalculate_schedule_metadata
from src.adjustments.travel import TravelAdjuster
from src.adjustments.altitude import AltitudeAdjuster
from src.models.finishing_drives import FinishingDrivesModel
from src.models.special_teams import SpecialTeamsModel
from src.predictions.spread_generator import SpreadGenerator
import numpy as np
import pandas as pd
import polars as pl


def build_team_records(
    client: CFBDClient,
    years: list[int],
    season_data: dict | None = None,
) -> dict[str, dict[int, tuple[int, int]]]:
    """Build team win-loss records for trajectory calculation.

    Uses cached games DataFrames from season_data when available,
    falling back to API calls only for years not in the cache.

    Args:
        client: CFBD API client
        years: List of years to fetch records for
        season_data: Optional dict from fetch_all_season_data(), keyed by year.
                     Each value is a tuple where index 0 is a Polars games DataFrame.

    Returns:
        Dict mapping team -> {year: (wins, losses)}
    """
    records = {}

    for year in years:
        # Try cached games DataFrame first (index 0 in the season_data tuple)
        if season_data is not None and year in season_data:
            games_df = season_data[year][0]  # Polars DataFrame
            # Filter to regular season (weeks 1-15) with completed scores
            regular = games_df.filter(
                (pl.col("week") <= 15)
                & pl.col("home_points").is_not_null()
                & pl.col("away_points").is_not_null()
            )
            for row in regular.iter_rows(named=True):
                home_won = row["home_points"] > row["away_points"]
                for team, is_home in [(row["home_team"], True), (row["away_team"], False)]:
                    if team not in records:
                        records[team] = {}
                    if year not in records[team]:
                        records[team][year] = (0, 0)
                    w, l = records[team][year]
                    won = home_won if is_home else not home_won
                    records[team][year] = (w + int(won), l + int(not won))
            continue

        # Fallback: fetch from API for years not in cache
        try:
            games = client.get_games(year=year, season_type="regular")
            for game in games:
                if game.home_points is None or game.away_points is None:
                    continue
                home_won = game.home_points > game.away_points
                for team, is_home in [(game.home_team, True), (game.away_team, False)]:
                    if team not in records:
                        records[team] = {}
                    if year not in records[team]:
                        records[team][year] = (0, 0)
                    w, l = records[team][year]
                    won = home_won if is_home else not home_won
                    records[team][year] = (w + int(won), l + int(not won))
        except Exception as e:
            logger.warning(f"Could not fetch games for {year}: {e}")

    return records


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# TURNOVER_PLAY_TYPES and POINTS_PER_TURNOVER imported from config.play_types


def _assign_postseason_pseudo_weeks(games: list[dict]) -> list[dict]:
    """Assign sequential pseudo-weeks to postseason games based on start_date.

    P0.1 Fix: Prevents walk-forward chronology violations by ensuring
    postseason games played on different dates get different week numbers.
    Games on the same date share a pseudo-week.

    Args:
        games: List of game dictionaries with 'week' and 'start_date' keys

    Returns:
        Same list with postseason game weeks remapped to 16, 17, 18, etc.
    """
    regular = [g for g in games if g["week"] <= 15]
    postseason = [g for g in games if g["week"] >= 16]

    if not postseason:
        return games

    # Sort by start_date for chronological ordering
    # start_date may be datetime.datetime or string depending on API version
    postseason.sort(key=lambda g: str(g.get("start_date") or "9999"))

    # Assign pseudo-weeks: each unique date gets a new week number
    current_pseudo_week = 16
    prev_date = None
    for game in postseason:
        raw_date = game.get("start_date")
        game_date = str(raw_date)[:10] if raw_date is not None else ""  # Extract YYYY-MM-DD
        if prev_date is not None and game_date != prev_date:
            current_pseudo_week += 1
        game["week"] = current_pseudo_week
        prev_date = game_date

    max_pseudo = max(g["week"] for g in postseason)
    logger.info(
        f"P0.1: Mapped {len(postseason)} postseason games to pseudo-weeks 16-{max_pseudo}"
    )

    return regular + postseason


def _remap_play_weeks(plays_df: pl.DataFrame, game_week_map: pl.DataFrame) -> pl.DataFrame:
    """Remap play week assignments using game_id -> week mapping from games.

    P0.1 Fix: Ensures play weeks match the pseudo-week assigned to their game,
    so walk-forward data leakage guards work correctly for postseason.

    Args:
        plays_df: Play DataFrame with game_id and week columns
        game_week_map: DataFrame with game_id and game_week columns

    Returns:
        DataFrame with week column remapped via game_id join
    """
    if len(plays_df) == 0 or "game_id" not in plays_df.columns:
        return plays_df

    remapped = plays_df.join(game_week_map, on="game_id", how="left")
    remapped = remapped.with_columns(
        pl.when(pl.col("game_week").is_not_null())
        .then(pl.col("game_week"))
        .otherwise(pl.col("week"))
        .alias("week")
    ).drop("game_week")
    return remapped


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
                    "home_line_scores": game.home_line_scores,
                    "away_line_scores": game.away_line_scores,
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
                "home_line_scores": game.home_line_scores,
                "away_line_scores": game.away_line_scores,
                "neutral_site": game.neutral_site or True,  # Bowl games are usually neutral
            })
            postseason_count += 1
        if postseason_count > 0:
            successful_weeks.append(16)
            # P3.9: Debug level for quiet runs
            logger.debug(f"Fetched {postseason_count} postseason games for {year}")
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

    def process_betting_lines(lines_list: list) -> None:
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
                ou_open = getattr(selected_line, 'over_under_open', None)
                betting.append({
                    "game_id": game_lines.id,
                    "home_team": game_lines.home_team,
                    "away_team": game_lines.away_team,
                    "spread_close": selected_line.spread,
                    "spread_open": selected_line.spread_open if selected_line.spread_open is not None else selected_line.spread,
                    "over_under": selected_line.over_under,
                    "over_under_open": ou_open if ou_open is not None else selected_line.over_under,
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
            # P3.9: Debug level for quiet runs
            logger.debug(f"Fetched {len([l for l in postseason_lines if l.lines])} postseason betting lines for {year}")
    except Exception as e:
        logger.warning(f"Error fetching betting lines: {e}")

    # P0.1: Assign sequential pseudo-weeks to postseason games by date
    games = _assign_postseason_pseudo_weeks(games)

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
                        "drive_id": play.drive_id,  # For red zone trip calculation
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

    # P0.2: Fetch postseason plays - try multiple weeks for robustness
    # CFBD labels all postseason plays as week=1, but we try weeks 1-5
    # to guard against API behavior changes.
    postseason_play_count = 0
    postseason_game_ids_with_plays = set()
    for ps_week in range(1, 6):
        try:
            postseason_plays = client.get_plays(year, week=ps_week, season_type="postseason")
            if not postseason_plays:
                break  # No more postseason weeks
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
                        "drive_id": play.drive_id,  # For red zone trip calculation
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
                    postseason_game_ids_with_plays.add(play.game_id)
        except Exception as e:
            logger.warning(f"Failed to fetch postseason plays for {year} week {ps_week}: {e}")
            break

    if postseason_play_count > 0:
        successful_weeks.append(16)
        logger.debug(
            f"Fetched {postseason_play_count} postseason efficiency plays "
            f"across {len(postseason_game_ids_with_plays)} games for {year}"
        )

    # Log fetch summary
    if failed_weeks:
        logger.warning(
            f"Plays fetch for {year}: {len(successful_weeks)} weeks OK, "
            f"{len(failed_weeks)} weeks FAILED: {[w for w, _ in failed_weeks]}"
        )
    else:
        logger.debug(f"Plays fetch for {year}: all {len(successful_weeks)} weeks OK")

    # P3.9: Debug level for quiet runs
    logger.debug(
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
    efficiency_weight: float = 0.45,
    explosiveness_weight: float = 0.45,
    turnover_weight: float = 0.10,
    garbage_time_weight: float = 0.1,
    asymmetric_garbage: bool = True,
    team_records: Optional[dict[str, dict[int, tuple[int, int]]]] = None,
    year: int = 2024,
    fcs_penalty_elite: float = 18.0,
    fcs_penalty_standard: float = 32.0,
    st_plays_df: Optional[pl.DataFrame] = None,
    historical_rankings: Optional[HistoricalRankings] = None,
    team_conferences: Optional[dict[str, str]] = None,
    hfa_global_offset: float = 0.0,
    ooc_credibility_weight: float = 0.0,
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
        efficiency_weight: Weight for success rate component (default 0.45)
        explosiveness_weight: Weight for IsoPPP component (default 0.45)
        turnover_weight: Weight for turnover margin component (default 0.10)
        garbage_time_weight: Weight for garbage time plays (default 0.1)
        asymmetric_garbage: Only penalize trailing team in garbage time (default True)
        team_records: Historical team records for trajectory calculation
        year: Current season year (for trajectory calculation)
        fcs_penalty_elite: Points for elite FCS teams (default 18.0)
        fcs_penalty_standard: Points for standard FCS teams (default 32.0)
        st_plays_df: Field goal plays dataframe for FG efficiency calculation (Polars DataFrame)
        historical_rankings: Week-by-week AP poll rankings for letdown spot detection
        team_conferences: Dict mapping team name to conference name for conference strength anchor
        hfa_global_offset: Points subtracted from ALL HFA values (for sweep testing). Default 0.0.

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

    # P1.1: Convert full schedule to pandas once (reused every week for SpreadGenerator)
    games_df_pd = optimize_dtypes(games_df.to_pandas())
    # Pre-calculate rest days + schedule metadata (vectorized, runs once)
    games_df_pd = precalculate_schedule_metadata(games_df_pd)

    for pred_week in range(start_week, max_week + 1):
        # Training data: all plays from games before this week
        # P1.2: Use Polars semi-join instead of materializing game_id list to Python
        train_games_pl = games_df.filter(pl.col("week") < pred_week)
        train_plays_pl = efficiency_plays_df.join(
            train_games_pl.select("id"), left_on="game_id", right_on="id", how="semi"
        ).filter(
            pl.col("offense").is_in(fbs_teams_list) &
            pl.col("defense").is_in(fbs_teams_list)
        )

        if len(train_plays_pl) < 5000:
            logger.warning(f"Week {pred_week}: insufficient play data ({len(train_plays_pl)}), skipping")
            continue

        # Convert to pandas for EFM (sklearn needs pandas/numpy)
        # P3.4: Apply optimized dtypes for memory efficiency
        train_plays_pd = optimize_dtypes(train_plays_pl.to_pandas())
        train_games_pd = optimize_dtypes(train_games_pl.to_pandas())

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

        # Initialize HFA with team-specific values and trajectory modifiers
        hfa = HomeFieldAdvantage(base_hfa=hfa_value, global_offset=hfa_global_offset)
        # Calculate trajectory modifiers if we have records
        if team_records:
            hfa.calculate_trajectory_modifiers(team_records, year)

        # Build team-specific HFA lookup for EFM fraud tax (uses full priority chain)
        hfa_lookup = {
            team: hfa.get_hfa_value(team)
            for team in fbs_teams
        }

        # Build EFM model
        efm = EfficiencyFoundationModel(
            ridge_alpha=ridge_alpha,
            efficiency_weight=efficiency_weight,
            explosiveness_weight=explosiveness_weight,
            turnover_weight=turnover_weight,
            garbage_time_weight=garbage_time_weight,
            asymmetric_garbage=asymmetric_garbage,
            ooc_credibility_weight=ooc_credibility_weight,
        )

        efm.calculate_ratings(
            train_plays_pd, train_games_pd,
            max_week=pred_week - 1, season=year,
            team_conferences=team_conferences,
            hfa_lookup=hfa_lookup,
        )

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
                    game_date=game.get("start_date"),  # For rest day calculation
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

    P3.8: Vectorized implementation using pandas merge + numpy operations.
    Uses game_id for reliable matching between predictions and betting lines.

    Args:
        predictions: List of prediction dictionaries (must include game_id)
        betting_df: Polars DataFrame with Vegas lines (must have game_id, spread_close, spread_open)
        use_opening_line: If True, use opening lines; if False, use closing lines (default)

    Returns:
        Pandas DataFrame with ATS results
    """
    if not predictions:
        return pd.DataFrame()

    spread_col = "spread_open" if use_opening_line else "spread_close"

    # P3.8: Convert to DataFrames for vectorized operations
    pred_df = pd.DataFrame(predictions)
    betting_pd = betting_df.to_pandas()

    # Merge predictions with betting data on game_id (left join to identify unmatched)
    merged = pred_df.merge(
        betting_pd[["game_id", "spread_open", "spread_close"]],
        on="game_id",
        how="left",
        suffixes=("", "_bet"),
    )

    # P0.4: Identify unmatched games from missing spread fields
    # (game_id is always present after left join; unmatched = missing betting columns)
    vegas_spread = merged[spread_col]
    unmatched_mask = vegas_spread.isna()
    matched_mask = ~unmatched_mask

    # Log unmatched games (preserve per-game logging)
    unmatched_df = merged[unmatched_mask]
    if len(unmatched_df) > 0:
        unmatched_games = unmatched_df[["game_id", "home_team", "away_team", "week", "year"]].to_dict("records")
    else:
        unmatched_games = []

    # Filter to matched games for vectorized ATS calculation
    df = merged[matched_mask].copy()

    if len(df) == 0:
        logger.warning("No predictions matched with betting lines")
        return pd.DataFrame()

    # P3.8: Vectorized ATS calculations
    actual_margin = df["actual_margin"].values
    model_spread = df["predicted_spread"].values
    vegas_spread_vals = df[spread_col].values

    # Convert our spread to Vegas convention for comparison
    model_spread_vegas = -model_spread

    # Calculate edge (how much we differ from Vegas)
    edge = model_spread_vegas - vegas_spread_vals
    model_pick_home = edge < 0  # Model likes home more than Vegas

    # Determine ATS result (vectorized)
    home_cover = actual_margin + vegas_spread_vals
    # ats_win: if picking home, home must cover (home_cover > 0)
    #          if picking away, home must NOT cover (home_cover < 0)
    ats_win = np.where(model_pick_home, home_cover > 0, home_cover < 0)
    ats_push = home_cover == 0

    # Get both open and close spreads
    spread_open = df["spread_open"].values
    spread_close = df["spread_close"].values

    # Calculate CLV (Closing Line Value) - vectorized
    # If betting home: CLV = spread_open - spread_close (positive if line moved toward home)
    # If betting away: CLV = spread_close - spread_open (positive if line moved toward away)
    clv = np.where(model_pick_home, spread_open - spread_close, spread_close - spread_open)
    # Handle cases where open/close are missing
    clv_mask = pd.isna(spread_open) | pd.isna(spread_close)
    clv = np.where(clv_mask, np.nan, clv)

    # Build result DataFrame with vectorized column assignment
    df["vegas_spread"] = vegas_spread_vals
    df["edge"] = np.abs(edge)
    df["pick"] = np.where(model_pick_home, "HOME", "AWAY")
    df["ats_win"] = ats_win
    df["ats_push"] = ats_push
    df["clv"] = clv

    # Sanity report: log match rate and unmatched games
    total_predictions = len(predictions)
    matched = len(df)
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

    return df


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

    print("\n### Phase-by-Phase Performance\n")

    # Format and print table
    display_cols = ["Phase", "Weeks", "Games", "MAE", "MAE vs Close", "ATS %", "3+ Edge", "5+ Edge", "Mean CLV"]
    available_cols = [c for c in display_cols if c in phase_metrics.columns]

    # Format numeric columns
    for col in ["MAE", "MAE vs Close", "ATS %", "Mean CLV"]:
        if col in phase_metrics.columns:
            phase_metrics[col] = phase_metrics[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )

    # Print as Markdown table
    header = "| " + " | ".join(available_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(available_cols)) + " |"
    print(header)
    print(sep)
    for _, row in phase_metrics[available_cols].iterrows():
        print("| " + " | ".join(str(row[c]) for c in available_cols) + " |")
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

    print(f"\n### Closing Line Value (CLV)\n")
    print(f"Games with open+close lines: {report['total_games_with_clv']}\n")
    print("| Edge | N | Mean CLV | CLV > 0 | ATS % |")
    print("|------|---|----------|---------|-------|")

    for name, stats in report["buckets"].items():
        print(
            f"| {name} | {stats['n']} | {stats['mean_clv']:+.2f} "
            f"| {stats['positive_clv_pct']:.1f}% | {stats['ats_pct']:.1f}% |"
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


def print_data_sanity_report(season_data: dict, years: list[int], verbose: bool = False) -> None:
    """Print sanity report after data fetch (P3.4).

    Reports:
    - Expected vs actual game counts per year
    - FBS team counts
    - Betting line coverage rate
    - Week coverage

    P3.9: Default is compact summary; use verbose=True for per-year details.

    Args:
        season_data: Dict from fetch_all_season_data
        years: List of years fetched
        verbose: Whether to print per-year breakdown (default False for quiet runs)
    """
    # Expected games per season (roughly 850-900 for FBS)
    EXPECTED_GAMES_PER_YEAR = 870

    # Aggregate statistics across years
    total_games = 0
    total_betting = 0
    total_plays = 0
    warnings = []

    for year in years:
        # Unpack season data (historical_rankings added for letdown spot detection)
        season_tuple = season_data[year]
        if len(season_tuple) >= 10:
            games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, st_plays_df, historical_rankings, team_conferences = season_tuple
        elif len(season_tuple) == 9:
            games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, st_plays_df, historical_rankings = season_tuple
        else:
            # Backward compatibility for old 8-tuple format
            games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, st_plays_df = season_tuple

        n_games = len(games_df)
        n_betting = len(betting_df)
        n_efficiency = len(efficiency_plays_df)
        n_fbs = len(fbs_teams)

        total_games += n_games
        total_betting += n_betting
        total_plays += n_efficiency

        # Game count check
        game_pct = n_games / EXPECTED_GAMES_PER_YEAR * 100
        game_status = "✓" if game_pct >= 95 else "⚠" if game_pct >= 80 else "✗"

        # Betting coverage
        betting_coverage = n_betting / n_games * 100 if n_games > 0 else 0
        betting_status = "✓" if betting_coverage >= 90 else "⚠" if betting_coverage >= 70 else "✗"

        # Week coverage — P2.1: separate regular season (1-15) from postseason (16+)
        weeks_with_games = set(games_df["week"].unique().to_list()) if n_games > 0 else set()
        weeks_with_plays = set(efficiency_plays_df["week"].unique().to_list()) if n_efficiency > 0 else set()
        regular_expected = set(range(1, 16))
        missing_game_weeks = regular_expected - weeks_with_games
        missing_play_weeks = regular_expected - weeks_with_plays
        postseason_game_weeks = {w for w in weeks_with_games if w >= 16}
        postseason_play_weeks = {w for w in weeks_with_plays if w >= 16}

        # Track warnings
        if game_pct < 95:
            warnings.append(f"{year}: {game_status} Games {game_pct:.0f}%")
        if betting_coverage < 90:
            warnings.append(f"{year}: {betting_status} Betting {betting_coverage:.0f}%")
        if missing_game_weeks:
            warnings.append(f"{year}: Missing regular-season game weeks {sorted(missing_game_weeks)}")
        if missing_play_weeks:
            warnings.append(f"{year}: Missing regular-season play weeks {sorted(missing_play_weeks)}")
        if postseason_game_weeks and not postseason_play_weeks:
            warnings.append(f"{year}: Postseason games found (weeks {sorted(postseason_game_weeks)}) but no postseason plays")

        # P3.9: Per-year details only in verbose mode
        if verbose:
            if year == years[0]:
                print("\n" + "=" * 60)
                print("DATA SANITY REPORT")
                print("=" * 60)

            print(f"\n{year}:")
            print(f"  {game_status} Games: {n_games:,} (expected ~{EXPECTED_GAMES_PER_YEAR}, {game_pct:.0f}%)")
            print(f"  {betting_status} Betting lines: {n_betting:,} ({betting_coverage:.0f}% coverage)")
            print(f"    FBS teams: {n_fbs}")
            print(f"    Efficiency plays: {n_efficiency:,}")
            if missing_game_weeks:
                print(f"    ⚠ Missing regular-season game weeks: {sorted(missing_game_weeks)}")
            if missing_play_weeks:
                print(f"    ⚠ Missing regular-season play weeks: {sorted(missing_play_weeks)}")
            if postseason_game_weeks:
                print(f"    Postseason: {len(postseason_game_weeks)} pseudo-week(s), plays in {len(postseason_play_weeks)} week(s)")
            if priors:
                print(f"    Preseason priors: {len(priors.preseason_ratings)} teams")

    # P3.9: Compact summary for non-verbose (warnings only)
    if not verbose:
        if warnings:
            print(f"\nData warnings: {len(warnings)} issues")
            for w in warnings[:5]:  # Show first 5 warnings
                print(f"  ⚠ {w}")
            if len(warnings) > 5:
                print(f"  ... and {len(warnings) - 5} more (use --verbose for details)")
        else:
            betting_rate = total_betting / total_games * 100 if total_games > 0 else 0
            print(f"\nData: {total_games:,} games, {total_plays:,} plays, {betting_rate:.0f}% betting coverage")
    else:
        print()


def fetch_week_data_delta(
    client: CFBDClient,
    year: int,
    target_week: int,
    use_cache: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Fetch data for a single week, using cache for historical weeks.

    This enables delta loading where only the current week needs to be fetched
    from the API, while all previous weeks are loaded from disk cache.

    Args:
        client: CFBD API client
        year: Season year
        target_week: Week number to fetch (1-16)
        use_cache: Whether to use week-level cache

    Returns:
        Tuple of (games_df, betting_df, efficiency_plays_df, turnover_plays_df, st_plays_df)
        for the target week only
    """
    from src.data.week_cache import WeekDataCache

    week_cache = WeekDataCache()

    # Try to load from week cache first
    if use_cache:
        games_df = week_cache.load_week(year, target_week, "games")
        betting_df = week_cache.load_week(year, target_week, "betting")
        efficiency_plays_df = week_cache.load_week(year, target_week, "efficiency_plays")
        turnover_plays_df = week_cache.load_week(year, target_week, "turnovers")
        st_plays_df = week_cache.load_week(year, target_week, "st_plays")

        if all(
            df is not None
            for df in [games_df, betting_df, efficiency_plays_df, turnover_plays_df, st_plays_df]
        ):
            logger.info(f"Using cached data for {year} week {target_week}")
            return games_df, betting_df, efficiency_plays_df, turnover_plays_df, st_plays_df

    # Fetch from API
    logger.info(f"Fetching {year} week {target_week} from API")

    # Fetch games for this week only
    games = []
    try:
        week_games = client.get_games(year, target_week)
        for game in week_games:
            if game.home_points is None:
                continue
            games.append({
                "id": game.id,
                "year": year,
                "week": target_week,
                "start_date": game.start_date,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "home_points": game.home_points,
                "away_points": game.away_points,
                "neutral_site": game.neutral_site or False,
            })
    except Exception as e:
        logger.warning(f"Failed to fetch games for {year} week {target_week}: {e}")

    games_df = pl.DataFrame(games)

    # Fetch betting lines for this week
    betting = []
    preferred_providers = ["DraftKings", "ESPN Bet", "Bovada"]

    try:
        lines = client.get_betting_lines(year, week=target_week, season_type="regular")
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

            # Fall back to first available
            if selected_line is None:
                selected_line = game_lines.lines[0] if game_lines.lines else None

            if selected_line and selected_line.spread is not None:
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
        logger.warning(f"Failed to fetch betting lines for {year} week {target_week}: {e}")

    betting_df = pl.DataFrame(betting)

    # Fetch plays for this week
    efficiency_plays = []
    turnover_plays = []
    st_plays = []

    try:
        plays = client.get_plays(year, target_week)
        for play in plays:
            play_type = play.play_type or ""

            # Turnovers
            if play_type in TURNOVER_PLAY_TYPES:
                turnover_plays.append({
                    "week": target_week,
                    "game_id": play.game_id,
                    "offense": play.offense,
                    "defense": play.defense,
                    "play_type": play_type,
                })

            # Special teams
            if any(st in play_type for st in ["Field Goal", "Punt", "Kickoff"]):
                st_plays.append({
                    "week": target_week,
                    "game_id": play.game_id,
                    "offense": play.offense,
                    "defense": play.defense,
                    "play_type": play_type,
                    "play_text": play.play_text,
                })

            # Efficiency plays
            if (play.ppa is not None and
                play.down is not None and
                play_type in SCRIMMAGE_PLAY_TYPES and
                play.distance is not None and play.distance >= 0):
                efficiency_plays.append({
                    "week": target_week,
                    "game_id": play.game_id,
                    "drive_id": play.drive_id,  # For red zone trip calculation
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
    except Exception as e:
        logger.warning(f"Failed to fetch plays for {year} week {target_week}: {e}")

    efficiency_plays_df = pl.DataFrame(efficiency_plays)
    turnover_plays_df = pl.DataFrame(turnover_plays)
    st_plays_df = pl.DataFrame(st_plays)

    # Validate home_team in efficiency plays
    if len(efficiency_plays_df) > 0 and len(games_df) > 0:
        game_home = games_df.select([
            pl.col("id").alias("game_id"),
            pl.col("home_team").alias("validated_home_team"),
        ])
        efficiency_plays_df = efficiency_plays_df.join(
            game_home, on="game_id", how="left"
        )
        efficiency_plays_df = efficiency_plays_df.with_columns(
            pl.col("validated_home_team").alias("home_team")
        ).drop("validated_home_team")

    # Save to week cache
    if use_cache:
        week_cache.save_week(year, target_week, "games", games_df)
        week_cache.save_week(year, target_week, "betting", betting_df)
        week_cache.save_week(year, target_week, "efficiency_plays", efficiency_plays_df)
        week_cache.save_week(year, target_week, "turnovers", turnover_plays_df)
        week_cache.save_week(year, target_week, "st_plays", st_plays_df)

    return games_df, betting_df, efficiency_plays_df, turnover_plays_df, st_plays_df


def fetch_season_data_with_delta(
    client: CFBDClient,
    year: int,
    current_week: Optional[int] = None,
    use_cache: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Fetch season data using delta loading strategy.

    If current_week is provided, only fetches that week from API and loads
    all previous weeks from cache. If cache is incomplete, falls back to
    full season fetch.

    Args:
        client: CFBD API client
        year: Season year
        current_week: Current week number (if None, fetches full season)
        use_cache: Whether to use week-level cache

    Returns:
        Tuple of (games_df, betting_df, efficiency_plays_df, turnover_plays_df, st_plays_df)
        for weeks 1 through current_week
    """
    from src.data.week_cache import WeekDataCache

    # If no current week specified, use full season fetch
    if current_week is None:
        logger.info(f"No current_week specified, fetching full season for {year}")
        games_df, betting_df = fetch_season_data(client, year)
        _, turnover_plays_df, efficiency_plays_df, st_plays_df = fetch_season_plays(client, year)

        # Apply postseason remapping
        if len(games_df) > 0:
            game_week_map = games_df.select([
                pl.col("id").alias("game_id"),
                pl.col("week").alias("game_week"),
            ])
            efficiency_plays_df = _remap_play_weeks(efficiency_plays_df, game_week_map)
            turnover_plays_df = _remap_play_weeks(turnover_plays_df, game_week_map)
            st_plays_df = _remap_play_weeks(st_plays_df, game_week_map)

        # Validate home_team
        if len(efficiency_plays_df) > 0 and len(games_df) > 0:
            game_home = games_df.select([
                pl.col("id").alias("game_id"),
                pl.col("home_team").alias("validated_home_team"),
            ])
            efficiency_plays_df = efficiency_plays_df.join(
                game_home, on="game_id", how="left"
            )
            efficiency_plays_df = efficiency_plays_df.with_columns(
                pl.col("validated_home_team").alias("home_team")
            ).drop("validated_home_team")

        return games_df, betting_df, efficiency_plays_df, turnover_plays_df, st_plays_df

    # Delta loading: load weeks 1 to current_week-1 from cache, fetch current_week
    week_cache = WeekDataCache()

    # Check if we have all historical weeks cached
    historical_weeks = range(1, current_week)
    all_cached = all(
        week_cache.has_cached_week(year, week, "games", use_cache)
        for week in historical_weeks
    )

    if not all_cached and use_cache:
        logger.info(
            f"Week cache incomplete for {year} weeks 1-{current_week-1}, "
            "falling back to full season fetch"
        )
        # Fall back to full season fetch and populate cache
        games_df, betting_df = fetch_season_data(client, year)
        _, turnover_plays_df, efficiency_plays_df, st_plays_df = fetch_season_plays(client, year)

        # Apply postseason remapping
        if len(games_df) > 0:
            game_week_map = games_df.select([
                pl.col("id").alias("game_id"),
                pl.col("week").alias("game_week"),
            ])
            efficiency_plays_df = _remap_play_weeks(efficiency_plays_df, game_week_map)
            turnover_plays_df = _remap_play_weeks(turnover_plays_df, game_week_map)
            st_plays_df = _remap_play_weeks(st_plays_df, game_week_map)

        # Validate home_team
        if len(efficiency_plays_df) > 0 and len(games_df) > 0:
            game_home = games_df.select([
                pl.col("id").alias("game_id"),
                pl.col("home_team").alias("validated_home_team"),
            ])
            efficiency_plays_df = efficiency_plays_df.join(
                game_home, on="game_id", how="left"
            )
            efficiency_plays_df = efficiency_plays_df.with_columns(
                pl.col("validated_home_team").alias("home_team")
            ).drop("validated_home_team")

        # Save each week to cache for future delta loads
        if use_cache:
            for week in range(1, 16):
                week_games = games_df.filter(pl.col("week") == week)
                if len(week_games) > 0:
                    week_betting = betting_df.filter(
                        pl.col("game_id").is_in(week_games["id"])
                    )
                    week_efficiency = efficiency_plays_df.filter(pl.col("week") == week)
                    week_turnovers = turnover_plays_df.filter(pl.col("week") == week)
                    week_st = st_plays_df.filter(pl.col("week") == week)

                    week_cache.save_week(year, week, "games", week_games)
                    week_cache.save_week(year, week, "betting", week_betting)
                    week_cache.save_week(year, week, "efficiency_plays", week_efficiency)
                    week_cache.save_week(year, week, "turnovers", week_turnovers)
                    week_cache.save_week(year, week, "st_plays", week_st)

        return games_df, betting_df, efficiency_plays_df, turnover_plays_df, st_plays_df

    # Load historical weeks from cache
    logger.info(f"Loading {year} weeks 1-{current_week-1} from cache")
    historical_dfs = {
        "games": [],
        "betting": [],
        "efficiency_plays": [],
        "turnovers": [],
        "st_plays": [],
    }

    for week in historical_weeks:
        for data_type in historical_dfs.keys():
            df = week_cache.load_week(year, week, data_type)
            if df is not None:
                historical_dfs[data_type].append(df)

    # Fetch current week from API
    logger.info(f"Fetching {year} week {current_week} from API (delta load)")
    current_games, current_betting, current_efficiency, current_turnovers, current_st = (
        fetch_week_data_delta(client, year, current_week, use_cache)
    )

    # Combine historical + current week
    def safe_concat(dfs_list: list[pl.DataFrame]) -> pl.DataFrame:
        """Safely concatenate Polars DataFrames."""
        valid_dfs = [df for df in dfs_list if len(df) > 0]
        if not valid_dfs:
            return pl.DataFrame()
        return pl.concat(valid_dfs)

    games_df = safe_concat(historical_dfs["games"] + [current_games])
    betting_df = safe_concat(historical_dfs["betting"] + [current_betting])
    efficiency_plays_df = safe_concat(historical_dfs["efficiency_plays"] + [current_efficiency])
    turnover_plays_df = safe_concat(historical_dfs["turnovers"] + [current_turnovers])
    st_plays_df = safe_concat(historical_dfs["st_plays"] + [current_st])

    logger.info(
        f"Delta load complete: {len(games_df)} games, "
        f"{len(efficiency_plays_df)} efficiency plays through week {current_week}"
    )

    return games_df, betting_df, efficiency_plays_df, turnover_plays_df, st_plays_df


def fetch_all_season_data(
    years: list[int],
    use_priors: bool = True,
    use_portal: bool = True,
    portal_scale: float = 0.15,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> dict:
    """Fetch and cache all season data (games, betting, plays, turnovers, priors, fbs_teams, historical_rankings).

    Args:
        years: List of years to fetch
        use_priors: Whether to build preseason priors
        use_portal: Whether to incorporate transfer portal data into priors
        portal_scale: How much to weight portal impact (default 0.15)
        use_cache: Whether to use cached data if available (default True)
        force_refresh: If True, bypass cache and force fresh API calls (default False)

    Returns:
        Dict mapping year to (games_df, betting_df, plays_df, turnover_df, priors,
                              efficiency_plays_df, fbs_teams, st_plays_df, historical_rankings)
    """
    from src.data.season_cache import SeasonDataCache

    client = CFBDClient()
    season_cache = SeasonDataCache()
    season_data = {}

    # If force_refresh, don't use cache at all
    effective_use_cache = use_cache and not force_refresh

    for year in years:
        # Try to load from cache first
        cached_season = None
        if effective_use_cache:
            cached_season = season_cache.load_season(year)

        if cached_season is not None:
            # Use cached data (already processed: remap + home_team validated before save)
            games_df, betting_df, efficiency_plays_df, turnover_plays_df, st_plays_df = cached_season
            logger.info(f"Using cached data for {year}")
            plays_df = pl.DataFrame()  # Empty placeholder
        else:
            # Fetch from API
            logger.debug(f"Fetching data for {year} from API...")

            games_df, betting_df = fetch_season_data(client, year)
            logger.debug(f"Loaded {len(games_df)} games, {len(betting_df)} betting lines")

            plays_df, turnover_plays_df, efficiency_plays_df, st_plays_df = fetch_season_plays(client, year)

            # P0.1: Remap postseason play weeks to match game pseudo-weeks
            if len(games_df) > 0:
                game_week_map = games_df.select([
                    pl.col("id").alias("game_id"),
                    pl.col("week").alias("game_week"),
                ])
                efficiency_plays_df = _remap_play_weeks(efficiency_plays_df, game_week_map)
                turnover_plays_df = _remap_play_weeks(turnover_plays_df, game_week_map)
                st_plays_df = _remap_play_weeks(st_plays_df, game_week_map)

            # P0.3: Validate home_team in efficiency plays by joining to game records
            # play.home may not be the home team string; game.home_team is reliable
            if len(efficiency_plays_df) > 0 and len(games_df) > 0:
                game_home = games_df.select([
                    pl.col("id").alias("game_id"),
                    pl.col("home_team").alias("validated_home_team"),
                ])
                efficiency_plays_df = efficiency_plays_df.join(
                    game_home, on="game_id", how="left"
                )
                efficiency_plays_df = efficiency_plays_df.with_columns(
                    pl.col("validated_home_team").alias("home_team")
                ).drop("validated_home_team")

                null_count = efficiency_plays_df["home_team"].null_count()
                total = len(efficiency_plays_df)
                if null_count > 0:
                    logger.warning(
                        f"P0.3: home_team coverage {total - null_count}/{total} "
                        f"({(total - null_count) / total * 100:.1f}%) after game join"
                    )
                else:
                    logger.debug(
                        f"P0.3: home_team validated via game join ({total} plays, 100% coverage)"
                    )

        turnover_df = build_game_turnovers(games_df, turnover_plays_df)
        logger.debug(f"Built turnover margins for {len(turnover_df)} games")

        # P2.1: Sanity check — validate regular-season (1-15) completeness; report postseason separately
        weeks_with_games = set(games_df["week"].unique().to_list()) if len(games_df) > 0 else set()
        weeks_with_plays = set(efficiency_plays_df["week"].unique().to_list()) if len(efficiency_plays_df) > 0 else set()
        regular_expected = set(range(1, 16))
        missing_game_weeks = regular_expected - weeks_with_games
        missing_play_weeks = regular_expected - weeks_with_plays
        postseason_game_weeks = sorted(w for w in weeks_with_games if w >= 16)
        postseason_play_weeks = sorted(w for w in weeks_with_plays if w >= 16)

        if missing_game_weeks or missing_play_weeks:
            logger.warning(
                f"Data completeness for {year}: "
                f"regular-season games missing weeks {sorted(missing_game_weeks) if missing_game_weeks else 'none'}, "
                f"plays missing weeks {sorted(missing_play_weeks) if missing_play_weeks else 'none'}"
            )
        else:
            logger.debug(f"Data completeness for {year}: all regular-season weeks 1-15 present")

        if postseason_game_weeks:
            logger.debug(
                f"Postseason for {year}: {len(postseason_game_weeks)} game pseudo-weeks, "
                f"{len(postseason_play_weeks)} play pseudo-weeks"
            )

        # P0.2: Postseason play coverage check
        # Compare postseason games (from games_df) to games with efficiency plays
        if len(games_df) > 0 and len(efficiency_plays_df) > 0:
            postseason_game_ids = set(
                games_df.filter(pl.col("week") >= 16)["id"].to_list()
            )
            postseason_play_game_ids = set(
                efficiency_plays_df.filter(pl.col("week") >= 16)["game_id"].unique().to_list()
            )
            if postseason_game_ids:
                missing_play_games = postseason_game_ids - postseason_play_game_ids
                coverage_pct = (
                    len(postseason_play_game_ids & postseason_game_ids) / len(postseason_game_ids) * 100
                )
                # P2.2: Report postseason play count and warn if suspiciously low
                n_postseason_plays = len(efficiency_plays_df.filter(pl.col("week") >= 16))
                avg_plays_per_game = n_postseason_plays / len(postseason_game_ids) if postseason_game_ids else 0

                if missing_play_games:
                    logger.warning(
                        f"Postseason play coverage {coverage_pct:.0f}% "
                        f"({len(postseason_game_ids) - len(missing_play_games)}/{len(postseason_game_ids)} games, "
                        f"{n_postseason_plays} plays, {avg_plays_per_game:.0f} plays/game) for {year}"
                    )
                elif avg_plays_per_game < 80:
                    logger.warning(
                        f"Postseason play count suspiciously low: {n_postseason_plays} plays "
                        f"across {len(postseason_game_ids)} games ({avg_plays_per_game:.0f} plays/game) for {year}"
                    )
                else:
                    logger.debug(
                        f"Postseason play coverage 100% "
                        f"({len(postseason_game_ids)} games, {n_postseason_plays} plays, "
                        f"{avg_plays_per_game:.0f} plays/game) for {year}"
                    )

            # Save processed season to cache (only if we fetched from API)
            if effective_use_cache and cached_season is None:
                season_cache.save_season(
                    year,
                    games_df,
                    betting_df,
                    efficiency_plays_df,
                    turnover_plays_df,
                    st_plays_df,
                )

        # Fetch FBS teams for EFM filtering
        fbs_teams_list = client.get_fbs_teams(year)
        fbs_teams = {t.school for t in fbs_teams_list}
        logger.debug(f"Loaded {len(fbs_teams)} FBS teams")

        # Build conference map for Conference Strength Anchor (year-appropriate)
        team_conferences = {}
        for t in fbs_teams_list:
            if t.school and t.conference:
                team_conferences[t.school] = t.conference
        logger.debug(f"Built conference map: {len(team_conferences)} teams across "
                     f"{len(set(team_conferences.values()))} conferences for {year}")

        # Load historical AP rankings (for letdown spot detection)
        historical_rankings = HistoricalRankings("AP Top 25")
        try:
            historical_rankings.load_from_api(client, year)
        except Exception as e:
            logger.warning(f"Could not load historical rankings for {year}: {e}")

        priors = None
        if use_priors:
            try:
                priors = PreseasonPriors(client)
                priors.calculate_preseason_ratings(
                    year,
                    use_portal=use_portal,
                    portal_scale=portal_scale,
                )
                logger.debug(
                    f"Loaded preseason priors for {len(priors.preseason_ratings)} teams"
                )
            except Exception as e:
                logger.warning(f"Could not load preseason priors for {year}: {e}")
                priors = None

        season_data[year] = (games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, st_plays_df, historical_rankings, team_conferences)

    return season_data


def _process_single_season(
    year: int,
    season_tuple: tuple,
    team_records: dict,
    start_week: int,
    end_week: Optional[int],
    hfa_value: float,
    prior_weight: int,
    ridge_alpha: float,
    efficiency_weight: float,
    explosiveness_weight: float,
    turnover_weight: float,
    garbage_time_weight: float,
    asymmetric_garbage: bool,
    fcs_penalty_elite: float,
    fcs_penalty_standard: float,
    use_opening_line: bool,
    hfa_global_offset: float = 0.0,
    ooc_credibility_weight: float = 0.0,
) -> tuple:
    """Process a single season in a worker process for parallel backtesting.

    Top-level function required for pickle serialization with ProcessPoolExecutor.
    Each spawned worker gets its own module-level state (ridge cache, GT thresholds).
    """
    # Configure logging in worker process (spawn mode starts with no handlers)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Each worker starts with a fresh ridge cache
    clear_ridge_cache()

    # Unpack season data
    if len(season_tuple) >= 10:
        games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, st_plays_df, historical_rankings, team_conferences = season_tuple
    elif len(season_tuple) == 9:
        games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, st_plays_df, historical_rankings = season_tuple
        team_conferences = None
    else:
        games_df, betting_df, plays_df, turnover_df, priors, efficiency_plays_df, fbs_teams, st_plays_df = season_tuple
        historical_rankings = None
        team_conferences = None

    # Walk-forward predictions
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
        historical_rankings=historical_rankings,
        team_conferences=team_conferences,
        hfa_global_offset=hfa_global_offset,
        ooc_credibility_weight=ooc_credibility_weight,
    )

    # Calculate ATS
    ats_results = None
    if len(betting_df) > 0:
        ats_results = calculate_ats_results(predictions, betting_df, use_opening_line)

    # Year MAE for logging
    year_df = pd.DataFrame(predictions)
    year_mae = year_df["abs_error"].mean() if not year_df.empty else None

    return year, predictions, ats_results, year_mae


def run_backtest(
    years: list[int],
    start_week: int = 1,
    end_week: Optional[int] = None,
    ridge_alpha: float = 50.0,
    use_priors: bool = True,
    hfa_value: float = 2.5,
    prior_weight: int = 8,
    season_data: Optional[dict] = None,
    efficiency_weight: float = 0.45,
    explosiveness_weight: float = 0.45,
    turnover_weight: float = 0.10,
    garbage_time_weight: float = 0.1,
    asymmetric_garbage: bool = True,
    fcs_penalty_elite: float = 18.0,
    fcs_penalty_standard: float = 32.0,
    use_portal: bool = True,
    portal_scale: float = 0.15,
    use_opening_line: bool = False,
    clear_cache: bool = True,
    use_season_cache: bool = True,
    force_refresh: bool = False,
    hfa_global_offset: float = 0.0,
    ooc_credibility_weight: float = 0.0,
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
        efficiency_weight: EFM success rate weight (default 0.45)
        explosiveness_weight: EFM IsoPPP weight (default 0.45)
        turnover_weight: EFM turnover margin weight (default 0.10)
        garbage_time_weight: EFM garbage time play weight (default 0.1)
        asymmetric_garbage: Only penalize trailing team in garbage time (default True)
        fcs_penalty_elite: Points for elite FCS teams (default 18.0)
        fcs_penalty_standard: Points for standard FCS teams (default 32.0)
        use_portal: Whether to incorporate transfer portal into preseason priors
        portal_scale: How much to weight portal impact (default 0.15)
        use_opening_line: If True, use opening lines for ATS; if False, use closing lines (default)
        clear_cache: Whether to clear ridge adjustment cache at start (default True)
        use_season_cache: Whether to use disk cache for season data (default True)
        force_refresh: If True, bypass all caches and force fresh API calls (default False)
        hfa_global_offset: Points subtracted from ALL HFA values (for sweep testing). Default 0.0.

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
            use_cache=use_season_cache,
            force_refresh=force_refresh,
        )

    # Build team records for trajectory calculation (need ~4 years before earliest year)
    # Pre-cache trajectory years not already in season_data to avoid extra API calls
    client = CFBDClient()
    trajectory_years = list(range(min(years) - 4, max(years) + 1))
    uncached_traj_years = [y for y in trajectory_years if y not in season_data]
    if uncached_traj_years:
        traj_data = fetch_all_season_data(
            uncached_traj_years,
            use_priors=False,  # Only need games, not priors
            use_cache=use_season_cache,
            force_refresh=force_refresh,
        )
        # Merge into season_data so build_team_records can use cached DataFrames
        season_data.update(traj_data)
        logger.info(f"Pre-cached {len(uncached_traj_years)} trajectory years from disk: {uncached_traj_years}")
    team_records = build_team_records(client, trajectory_years, season_data=season_data)
    logger.info(f"Built team records for trajectory ({len(team_records)} teams, years {trajectory_years[0]}-{trajectory_years[-1]})")

    all_predictions = []
    all_ats = []

    # Strip non-picklable client references from priors for multiprocessing
    # (client is only used during data fetching, which is already complete)
    for year in years:
        season_tuple = season_data[year]
        if len(season_tuple) >= 5:
            priors = season_tuple[4]
            if priors is not None and hasattr(priors, "client"):
                priors.client = None

    # Common kwargs for _process_single_season
    season_kwargs = dict(
        team_records=team_records,
        start_week=start_week,
        end_week=end_week,
        hfa_value=hfa_value,
        prior_weight=prior_weight,
        ridge_alpha=ridge_alpha,
        efficiency_weight=efficiency_weight,
        explosiveness_weight=explosiveness_weight,
        turnover_weight=turnover_weight,
        garbage_time_weight=garbage_time_weight,
        asymmetric_garbage=asymmetric_garbage,
        fcs_penalty_elite=fcs_penalty_elite,
        fcs_penalty_standard=fcs_penalty_standard,
        use_opening_line=use_opening_line,
        hfa_global_offset=hfa_global_offset,
        ooc_credibility_weight=ooc_credibility_weight,
    )

    if len(years) > 1:
        # Parallel execution: each season runs in its own process
        n_workers = min(len(years), os.cpu_count() or 4)
        logger.info(f"Running {len(years)} seasons in parallel ({n_workers} workers)")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_season,
                    year=year,
                    season_tuple=season_data[year],
                    **season_kwargs,
                ): year
                for year in years
            }

            for future in futures:
                try:
                    year_val, predictions, ats_results, year_mae = future.result()
                    all_predictions.extend(predictions)
                    if ats_results is not None:
                        all_ats.append(ats_results)
                    if year_mae is not None:
                        logger.debug(f"{year_val} MAE: {year_mae:.2f}")
                except Exception as e:
                    failed_year = futures[future]
                    logger.error(f"Season {failed_year} failed: {e}")
                    raise
    else:
        # Single year: sequential execution (no multiprocessing overhead)
        for year in years:
            logger.debug(f"Backtesting {year} season...")
            year_val, predictions, ats_results, year_mae = _process_single_season(
                year=year,
                season_tuple=season_data[year],
                **season_kwargs,
            )
            all_predictions.extend(predictions)
            if ats_results is not None:
                all_ats.append(ats_results)
            if year_mae is not None:
                logger.debug(f"{year_val} MAE: {year_mae:.2f}")

    # Sort predictions for deterministic output across parallel/sequential runs
    all_predictions.sort(key=lambda p: (p["year"], p["week"], p["game_id"]))

    # Combine results
    predictions_df = pd.DataFrame(all_predictions)
    ats_df = pd.concat(all_ats, ignore_index=True) if all_ats else None
    if ats_df is not None:
        ats_df = ats_df.sort_values(["year", "week", "game_id"]).reset_index(drop=True)

    # Calculate overall metrics
    metrics = calculate_metrics(predictions_df, ats_df)

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
    eff_weights = [0.40, 0.45, 0.50]  # These are efficiency weights (explosiveness = 0.90 - eff)
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
    verbose: bool = False,
) -> None:
    """Print backtest results to console.

    Args:
        results: Dictionary with backtest results from run_backtest
        ats_df: Optional ATS results DataFrame for sanity reporting
        diagnostics: Whether to print detailed diagnostics (stack, CLV, phase reports).
                    Set False for faster output in sweeps. Default True.
        verbose: Whether to print per-week breakdowns and detailed logs. Default False.
    """
    print("\n## BACKTEST RESULTS\n")

    metrics = results["metrics"]
    print(f"Total games predicted: {metrics['total_games']}\n")

    # Error Metrics table
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| MAE (vs actual) | {metrics['mae']:.2f} pts |")
    if "mae_vs_close" in metrics:
        print(f"| MAE (vs closing) | {metrics['mae_vs_close']:.2f} pts |")
    print(f"| RMSE | {metrics['rmse']:.2f} pts |")
    print(f"| Median Error | {metrics['median_error']:.2f} pts |")
    print(f"| Within 3 pts | {metrics['within_3']:.1%} |")
    print(f"| Within 7 pts | {metrics['within_7']:.1%} |")
    print(f"| Within 10 pts | {metrics['within_10']:.1%} |")

    if "ats_record" in metrics:
        print(f"\n### Against the Spread\n")
        print("| Edge | Record | Win % | ROI |")
        print("|------|--------|-------|-----|")
        print(f"| All | {metrics['ats_record']} | {metrics['ats_win_rate']:.1%} | {metrics['roi']:.1%} |")

        for threshold in [3, 5]:
            key = f"ats_{threshold}pt_edge"
            if key in metrics:
                print(f"| {threshold}+ pts | {metrics[key]} |  |  |")

    # Weekly MAE breakdown (P3.9: gated behind --verbose for faster default runs)
    predictions_df = results["predictions"]
    if verbose and not predictions_df.empty:
        print(f"\n### MAE by Week\n")
        print("| Week | Games | MAE |")
        print("|------|-------|-----|")
        # Collapse postseason pseudo-weeks (16+) into a single "Post" row
        display_week = predictions_df["week"].clip(upper=16)
        weekly = predictions_df.assign(display_week=display_week).groupby("display_week").agg(
            games=("abs_error", "count"),
            mae=("abs_error", "mean"),
        )
        for week, row in weekly.iterrows():
            label = "Post" if int(week) == 16 else f"{int(week):3d} "
            print(f"| {label} | {int(row['games'])} | {row['mae']:.2f} |")

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

    print("\n### Sanity Check\n")

    # Predictions per week (collapse postseason pseudo-weeks into single bucket)
    print("\nPredictions per week:")
    display_week = predictions_df["week"].clip(upper=16)
    week_counts = display_week.value_counts().sort_index()
    n_regular = (predictions_df["week"] <= 15).sum()
    n_post = (predictions_df["week"] >= 16).sum()
    reg_weeks = predictions_df[predictions_df["week"] <= 15]["week"].nunique()
    print(f"  Total: {len(predictions_df):,} predictions ({reg_weeks} regular-season weeks + postseason)")
    print(f"  Regular season: {n_regular:,} games | Postseason: {n_post:,} games")
    reg_counts = week_counts[week_counts.index < 16]
    if len(reg_counts) > 0:
        print(f"  Per-week range (regular): {reg_counts.min()}-{reg_counts.max()}")

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
        "--hfa-offset",
        type=float,
        default=0.50,
        help="Points subtracted from ALL HFA values globally (default: 0.50, calibrated Feb 2026)",
    )
    parser.add_argument(
        "--ooc-cred-weight",
        type=float,
        default=0.0,
        help="OOC Credibility Anchor scale for intra-conf play weighting (default: 0.0 = disabled, REJECTED: degraded 5+ Edge)",
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
        default=0.45,
        help="Weight for success rate component (default: 0.45)",
    )
    parser.add_argument(
        "--explosiveness-weight",
        type=float,
        default=0.45,
        help="Weight for IsoPPP component (default: 0.45)",
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (per-week MAE, detailed logging). Default is quiet summary.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable season data caching (always fetch from API)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh all data from API (bypasses cache completely)",
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
        use_cache=not args.no_cache,
        force_refresh=args.force_refresh,
    )

    # P3.4: Print data sanity report
    # P3.9: Pass verbose flag for per-year breakdown
    print_data_sanity_report(season_data, args.years, verbose=args.verbose)

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
    print(f"  HFA:                team-specific (fallback={args.hfa}, global_offset={args.hfa_offset})")
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
        hfa_global_offset=args.hfa_offset,
        ooc_credibility_weight=args.ooc_cred_weight,
    )

    # P3.4: Print results with ATS data for sanity report
    # P3.7: Skip detailed diagnostics if --no-diagnostics flag is set
    # P3.9: Use --verbose for per-week breakdown (default is quiet summary)
    print_results(
        results,
        ats_df=results.get("ats_results"),
        diagnostics=not args.no_diagnostics,
        verbose=args.verbose,
    )

    # Save to CSV if requested
    if args.output:
        results["predictions"].to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
