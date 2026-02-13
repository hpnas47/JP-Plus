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
from typing import NamedTuple, Optional

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
from src.models.blended_priors import create_blended_generator, BlendSchedule
from src.adjustments.home_field import HomeFieldAdvantage
from src.adjustments.situational import SituationalAdjuster, HistoricalRankings, precalculate_schedule_metadata
from src.adjustments.travel import TravelAdjuster
from src.adjustments.altitude import AltitudeAdjuster
# FinishingDrivesModel import removed — shelved after 4 backtest rejections (70-80% overlap with EFM)
# Infrastructure preserved in src/models/finishing_drives.py for future reactivation
from src.models.special_teams import SpecialTeamsModel
from src.models.fcs_strength import FCSStrengthEstimator
from src.adjustments.qb_continuous import QBContinuousAdjuster
from src.models.learned_situational import (
    LearnedSituationalModel,
    SituationalFeatures,
    compute_situational_residual,
)
from src.predictions.spread_generator import SpreadGenerator
import numpy as np
import pandas as pd
import polars as pl


class SeasonData(NamedTuple):
    """Typed container for all season data required by backtest.

    Using NamedTuple instead of raw tuple:
    - Named field access prevents positional unpacking bugs
    - Picklable for multiprocessing (unlike some dataclasses)
    - Adding new fields is safe (access by name, not position)
    """

    games_df: pl.DataFrame
    betting_df: pl.DataFrame
    plays_df: pl.DataFrame  # Turnover plays (legacy name kept for compatibility)
    turnover_df: pl.DataFrame
    priors: Optional["PreseasonPriors"]
    efficiency_plays_df: pl.DataFrame
    fbs_teams: set[str]
    st_plays_df: pl.DataFrame
    historical_rankings: Optional["HistoricalRankings"]
    team_conferences: Optional[dict[str, str]]


def build_team_records(
    client: CFBDClient,
    years: list[int],
    season_data: Optional[dict[int, SeasonData]] = None,
) -> dict[str, dict[int, tuple[int, int]]]:
    """Build team win-loss records for trajectory calculation.

    Uses cached games DataFrames from season_data when available,
    falling back to API calls only for years not in the cache.

    Args:
        client: CFBD API client
        years: List of years to fetch records for
        season_data: Optional dict from fetch_all_season_data(), keyed by year.

    Returns:
        Dict mapping team -> {year: (wins, losses)}
    """
    records = {}

    for year in years:
        # Try cached games DataFrame first
        if season_data is not None and year in season_data:
            games_df = season_data[year].games_df
            # Filter to regular season (weeks 1-15) with completed scores
            regular = games_df.filter(
                (pl.col("week") <= 15)
                & pl.col("home_points").is_not_null()
                & pl.col("away_points").is_not_null()
            )
            # P3.4: Vectorized aggregation instead of Python row iteration
            regular = regular.with_columns(
                (pl.col("home_points") > pl.col("away_points")).alias("home_won")
            )
            # Home team stats: wins when home_won=True, losses when home_won=False
            home_agg = regular.group_by("home_team").agg([
                pl.col("home_won").sum().alias("wins"),
                (~pl.col("home_won")).sum().alias("losses"),
            ])
            # Away team stats: wins when home_won=False, losses when home_won=True
            away_agg = regular.group_by("away_team").agg([
                (~pl.col("home_won")).sum().alias("wins"),
                pl.col("home_won").sum().alias("losses"),
            ])
            # Merge into records dict
            for row in home_agg.iter_rows(named=True):
                team = row["home_team"]
                if team not in records:
                    records[team] = {}
                records[team][year] = (row["wins"], row["losses"])
            for row in away_agg.iter_rows(named=True):
                team = row["away_team"]
                if team not in records:
                    records[team] = {}
                if year in records[team]:
                    w, l = records[team][year]
                    records[team][year] = (w + row["wins"], l + row["losses"])
                else:
                    records[team][year] = (row["wins"], row["losses"])
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
                "neutral_site": game.neutral_site if game.neutral_site is not None else True,  # Default to neutral if API returns None
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
    k_int: float = 10.0,
    k_fumble: float = 30.0,
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
    st_spread_cap: Optional[float] = 2.5,
    st_early_weight: float = 1.0,
    fcs_estimator: Optional[FCSStrengthEstimator] = None,
    fcs_static: bool = False,
    fcs_use_deviation: bool = True,  # Use FBS team's deviation from FBS mean for FCS games
    # Learned Situational Adjustment (LSA) parameters
    lsa_model: Optional[LearnedSituationalModel] = None,
    # LSA training data sources (for turnover adjustment and Vegas filter)
    turnover_df: Optional[pl.DataFrame] = None,
    betting_df: Optional[pl.DataFrame] = None,
    # QB Continuous Rating parameters
    use_qb_continuous: bool = False,
    qb_shrinkage_k: float = 200.0,
    qb_cap: float = 3.0,
    qb_scale: float = 4.0,
    qb_prior_decay: float = 0.3,
    qb_use_prior_season: bool = True,
    qb_phase1_only: bool = False,
    qb_fix_misattribution: bool = False,
    lock_prior_weeks: int = 0,  # Force 100% prior weight for weeks <= this value
    phase1_shrinkage: float = 0.90,  # Shrinkage for Phase 1 spread predictions
) -> tuple[list[dict], list[tuple]]:
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
        fcs_penalty_elite: Points for elite FCS teams (default 18.0, used if fcs_static=True)
        fcs_penalty_standard: Points for standard FCS teams (default 32.0, used if fcs_static=True)
        st_plays_df: Field goal plays dataframe for FG efficiency calculation (Polars DataFrame)
        historical_rankings: Week-by-week AP poll rankings for letdown spot detection
        team_conferences: Dict mapping team name to conference name for conference strength anchor
        hfa_global_offset: Points subtracted from ALL HFA values (for sweep testing). Default 0.0.
        fcs_estimator: Dynamic FCS strength estimator (if provided and fcs_static=False, used for penalties)
        fcs_static: If True, use static elite list instead of dynamic estimator. Default False.
        lsa_model: Learned Situational Model for ridge regression on situational residuals.
            If provided and trained, replaces fixed situational constants.

    Returns:
        Tuple of (prediction_results, lsa_training_data):
            - prediction_results: List of prediction result dictionaries
            - lsa_training_data: List of (features_array, residual) tuples for LSA training
    """
    results = []
    lsa_training_data = []  # Collect (features_array, residual) tuples for LSA
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

    # Stateless adjusters: construct once, reuse every week
    # These use static lookup tables (team locations, altitudes, rivalry config)
    # and don't depend on training data, so they're safe to reuse.
    travel_adjuster = TravelAdjuster()
    altitude_adjuster = AltitudeAdjuster()
    situational_adjuster = SituationalAdjuster()

    # QB Continuous: Initialize once, build data incrementally each week
    # The adjuster caches QB data by week and builds incrementally
    qb_continuous_adjuster = None
    if use_qb_continuous:
        logger.info(f"QB Continuous enabled: initializing for {year} season")
        qb_continuous_adjuster = QBContinuousAdjuster(
            year=year,
            shrinkage_k=qb_shrinkage_k,
            qb_cap=qb_cap,
            qb_scale=qb_scale,
            prior_decay=qb_prior_decay,
            use_prior_season=qb_use_prior_season,
            phase1_only=qb_phase1_only,
            fix_misattribution=qb_fix_misattribution,
        )

    # LSA: If model provided, train after each week's games
    use_learned_situ = lsa_model is not None

    # P3.1: Initialize HFA once (depends only on team_records and year, constant for season)
    hfa = HomeFieldAdvantage(base_hfa=hfa_value, global_offset=hfa_global_offset)
    if team_records:
        hfa.calculate_trajectory_modifiers(team_records, year)
    # Build HFA lookup once (reused every week)
    hfa_lookup = {team: hfa.get_hfa_value(team) for team in fbs_teams}

    # P3.2: Pre-convert Polars DataFrames to Pandas by week
    # Week-keyed dict structure guarantees walk-forward correctness (no future data possible)
    plays_by_week_pd = {}
    games_by_week_pd = {}
    for week in range(1, max_week + 1):
        # Filter games for this week
        week_games = games_df.filter(pl.col("week") == week)
        if len(week_games) > 0:
            games_by_week_pd[week] = optimize_dtypes(week_games.to_pandas())

        # Filter plays for this week (FBS-only via semi-join + filter)
        week_plays = efficiency_plays_df.join(
            week_games.select("id"), left_on="game_id", right_on="id", how="semi"
        ).filter(
            pl.col("offense").is_in(fbs_teams_list) &
            pl.col("defense").is_in(fbs_teams_list)
        )
        if len(week_plays) > 0:
            plays_by_week_pd[week] = optimize_dtypes(week_plays.to_pandas())

    # P3.3: Pre-convert ST plays by week
    st_by_week_pd = {}
    if st_plays_df is not None and len(st_plays_df) > 0:
        for week in range(1, max_week + 1):
            week_st = st_plays_df.filter(pl.col("week") == week)
            if len(week_st) > 0:
                st_by_week_pd[week] = optimize_dtypes(week_st.to_pandas())

    for pred_week in range(start_week, max_week + 1):
        # Training data: concat pre-converted week slices for weeks < pred_week
        # Week-keyed dict guarantees no future data inclusion (walk-forward safe)
        train_plays_pd = pd.concat(
            [plays_by_week_pd[w] for w in range(1, pred_week) if w in plays_by_week_pd],
            ignore_index=True
        ) if any(w in plays_by_week_pd for w in range(1, pred_week)) else pd.DataFrame()

        train_games_pd = pd.concat(
            [games_by_week_pd[w] for w in range(1, pred_week) if w in games_by_week_pd],
            ignore_index=True
        ) if any(w in games_by_week_pd for w in range(1, pred_week)) else pd.DataFrame()

        # P1.1 FIX: Explicit walk-forward chronology assertion
        # Validates no future data leaked into training set
        if len(train_plays_pd) > 0:
            max_train_week = train_plays_pd["week"].max()
            assert max_train_week < pred_week, (
                f"WALK-FORWARD VIOLATION: Training data contains week {max_train_week} "
                f"but predicting week {pred_week}"
            )

        # Check if we have enough training data
        use_pure_priors = False
        train_play_count = sum(len(plays_by_week_pd[w]) for w in range(1, pred_week) if w in plays_by_week_pd)
        if train_play_count < 5000:
            # Not enough plays to train EFM - can we use pure preseason priors?
            if preseason_priors is not None and preseason_priors.preseason_ratings:
                use_pure_priors = True
                logger.info(
                    f"Week {pred_week}: no training data, using 100% preseason priors"
                )
            else:
                logger.warning(
                    f"Week {pred_week}: insufficient play data ({train_play_count}) "
                    f"and no preseason priors, skipping"
                )
                continue

        if use_pure_priors:
            # Week 1 (or any week with no training data): use 100% preseason priors
            # Handle both PreseasonPriors (combined_rating) and BlendedRating (blended_rating)
            team_ratings = {
                team: getattr(rating, 'combined_rating', getattr(rating, 'blended_rating', 0))
                for team, rating in preseason_priors.preseason_ratings.items()
            }
            logger.debug(f"Week {pred_week}: using {len(team_ratings)} preseason ratings")
        else:
            # Normal case: train EFM on available data
            # (train_plays_pd and train_games_pd already built via pd.concat above)

            # Build EFM model
            efm = EfficiencyFoundationModel(
                ridge_alpha=ridge_alpha,
                efficiency_weight=efficiency_weight,
                explosiveness_weight=explosiveness_weight,
                turnover_weight=turnover_weight,
                k_int=k_int,
                k_fumble=k_fumble,
                garbage_time_weight=garbage_time_weight,
                asymmetric_garbage=asymmetric_garbage,
                ooc_credibility_weight=ooc_credibility_weight,
            )

            efm.calculate_ratings(
                train_plays_pd, train_games_pd,
                max_week=pred_week - 1, season=year,
                team_conferences=team_conferences,
                hfa_lookup=hfa_lookup,
                fbs_teams=fbs_teams,  # P0: Exclude FCS teams from normalization
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
                # Phase 1 Purity Test: Lock to 100% prior if pred_week <= lock_prior_weeks
                if lock_prior_weeks > 0 and pred_week <= lock_prior_weeks:
                    games_played = 0  # Forces 100% prior weight
                    logger.debug(f"Week {pred_week}: LOCKED to 100% prior (lock_prior_weeks={lock_prior_weeks})")
                else:
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

        # Calculate FG efficiency ratings from FG plays
        special_teams = SpecialTeamsModel()
        if st_by_week_pd:
            # P3.3: Concat pre-converted ST week slices (walk-forward safe via week-keyed dict)
            train_st_pd = pd.concat(
                [st_by_week_pd[w] for w in range(1, pred_week) if w in st_by_week_pd],
                ignore_index=True
            ) if any(w in st_by_week_pd for w in range(1, pred_week)) else pd.DataFrame()
            if len(train_st_pd) > 0:
                # Calculate all ST ratings (FG + Punt + Kickoff)
                special_teams.calculate_all_st_ratings_from_plays(
                    train_st_pd, max_week=pred_week - 1
                )
                # Integrate special teams ratings into EFM for O/D/ST breakdown (diagnostic only)
                if not use_pure_priors:
                    for team, st_rating in special_teams.team_ratings.items():
                        efm.set_special_teams_rating(team, st_rating.overall_rating)

        # Update FCS estimator with games from prior weeks (walk-forward safe)
        # Only update if we have an estimator and not using static mode
        active_fcs_estimator = None
        if fcs_estimator is not None and not fcs_static:
            fcs_estimator.update_from_games(games_df, fbs_teams, through_week=pred_week - 1)
            active_fcs_estimator = fcs_estimator

        # QB Continuous: Build data through pred_week - 1 (walk-forward safe)
        if qb_continuous_adjuster is not None:
            qb_continuous_adjuster.build_qb_data(through_week=pred_week - 1)

        # Build spread generator with EFM ratings
        # Note: travel/altitude/situational are stateless (reused from outside loop)
        # special_teams is retrained each week (fresh instance)
        # finishing_drives removed — shelved (hardcoded to 0.0 in SpreadGenerator)
        spread_gen = SpreadGenerator(
            ratings=team_ratings,
            special_teams=special_teams,  # FG/Punt/Kickoff efficiency (retrained each week)
            home_field=hfa,
            situational=situational_adjuster,  # Stateless (reused)
            travel=travel_adjuster,  # Stateless (reused)
            altitude=altitude_adjuster,  # Stateless (reused)
            fbs_teams=fbs_teams,
            fcs_penalty_elite=fcs_penalty_elite,
            fcs_penalty_standard=fcs_penalty_standard,
            st_spread_cap=st_spread_cap,  # Margin-level ST capping (Approach B)
            st_early_season_weight=st_early_weight,  # Early-season ST weighting (Approach C)
            fcs_estimator=active_fcs_estimator,  # Dynamic FCS penalties (None = use static)
            qb_continuous=qb_continuous_adjuster,  # Continuous QB rating adjustment
            use_fbs_deviation_for_fcs=fcs_use_deviation,  # FBS team deviation for FCS games
        )

        # Predict this week's games (Polars iteration is faster)
        week_games = games_df.filter(pl.col("week") == pred_week)

        # LSA: Train model before predictions for this week if we have enough data
        if use_learned_situ:
            lsa_model.train(max_week=pred_week - 1)

        for game in week_games.iter_rows(named=True):
            try:
                # Standard prediction (uses fixed situational constants)
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
                final_spread = pred.spread

                # LSA: Collect training data and optionally apply learned adjustment
                if use_learned_situ:
                    # Determine favorite for rivalry boost
                    prelim_spread = team_ratings.get(game["home_team"], 0.0) - team_ratings.get(game["away_team"], 0.0)
                    home_is_favorite = prelim_spread > 0

                    # Get situational factors (same as SpreadGenerator does internally)
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

                    # Compute residual for training
                    residual = compute_situational_residual(
                        actual_margin=actual_margin,
                        predicted_spread=pred.spread,
                        fixed_situational=pred.components.situational,
                    )

                    # Look up turnover margin for this game (for LSA turnover adjustment)
                    turnover_margin = None
                    if turnover_df is not None and len(turnover_df) > 0:
                        game_to_row = turnover_df.filter(pl.col("game_id") == game["id"])
                        if len(game_to_row) > 0:
                            turnover_margin = game_to_row["net_home_to_margin"][0]

                    # Look up Vegas spread for this game (for LSA filtering/weighting)
                    vegas_spread = None
                    if betting_df is not None and len(betting_df) > 0:
                        game_betting = betting_df.filter(pl.col("game_id") == game["id"])
                        if len(game_betting) > 0:
                            # Use closing spread (spread column), not opening
                            if "spread" in game_betting.columns:
                                vegas_spread = game_betting["spread"][0]

                    # Add to training data (extended format with weight and vegas)
                    weight = lsa_model._compute_sample_weight(vegas_spread) if lsa_model else 1.0
                    lsa_training_data.append((features.to_array(), residual, weight, vegas_spread))
                    lsa_model.add_training_game(features, residual, turnover_margin, vegas_spread)

                    # If model is trained, apply learned adjustment instead of fixed
                    if lsa_model.is_trained():
                        learned_situ = lsa_model.predict(features)
                        # Replace fixed situational with learned
                        final_spread = pred.spread - pred.components.situational + learned_situ

                # Phase 1 Shrinkage: Apply spread compression for weeks 1-3
                # Formula: NewSpread = (OldSpread - HFA) * Shrinkage + HFA
                # This reduces overconfidence in prior-driven predictions while preserving HFA
                if phase1_shrinkage < 1.0 and pred_week <= 3:
                    hfa_value = pred.components.home_field
                    rating_diff = final_spread - hfa_value
                    final_spread = rating_diff * phase1_shrinkage + hfa_value

                results.append({
                    "game_id": game["id"],  # For reliable Vegas line matching
                    "year": game["year"],
                    "week": pred_week,
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "predicted_spread": final_spread,
                    "actual_margin": actual_margin,
                    "error": final_spread - actual_margin,
                    "abs_error": abs(final_spread - actual_margin),
                    # P2.11: Track correlated adjustment stack (HFA + travel + altitude)
                    "correlated_stack": pred.components.correlated_stack,
                    "hfa": pred.components.home_field,
                    "travel": pred.components.travel,
                    "altitude": pred.components.altitude,
                    # P3.4: Track ratings for sanity check
                    "home_rating": team_ratings.get(game["home_team"], 0.0),
                    "away_rating": team_ratings.get(game["away_team"], 0.0),
                    # Dual-cap mode: raw component values for spread reassembly
                    "special_teams_raw": pred.components.special_teams_raw,
                    "home_field_raw": pred.components.home_field_raw,
                    "base_margin": pred.components.base_margin,
                    "situational": pred.components.situational,
                    "fcs_adj": pred.components.fcs_adjustment,
                    "pace_adj": pred.components.pace_adjustment,
                    "qb_adj": pred.components.qb_adjustment,
                    "special_teams": pred.components.special_teams,
                    "env_score": pred.components.env_score,
                })
            except Exception as e:
                logger.debug(f"Error predicting {game['away_team']} @ {game['home_team']}: {e}")

    return results, lsa_training_data


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


def assemble_spread(
    base_margin: float,
    home_field_raw: float,
    situational: float,
    travel: float,
    altitude: float,
    special_teams_raw: float,
    fcs_adj: float,
    pace_adj: float,
    qb_adj: float,
    env_score: float,
    st_cap: Optional[float] = 2.5,
    hfa_offset: float = 0.50,
) -> float:
    """Assemble spread from raw components with given caps/offsets.

    This is a pure function for dual-cap mode: given raw component values,
    reassemble the spread with different ST cap and HFA offset parameters.

    Args:
        base_margin: EFM ratings differential (home - away)
        home_field_raw: Raw HFA before global offset
        situational: Situational adjustment (bye weeks, letdown, etc.)
        travel: Travel penalty (already smoothed via aggregator)
        altitude: Altitude penalty (already smoothed via aggregator)
        special_teams_raw: Raw ST differential before cap
        fcs_adj: FCS team penalty
        pace_adj: Triple-option pace compression
        qb_adj: QB injury adjustment
        env_score: Full environmental score from aggregator
        st_cap: Cap on ST differential (None or 0 = no cap)
        hfa_offset: Points subtracted from raw HFA (global offset)

    Returns:
        Assembled spread (positive = home favored)
    """
    # Apply HFA offset (with floor at 0.5)
    hfa = max(0.5, home_field_raw - hfa_offset) if home_field_raw > 0 else 0.0

    # Apply ST cap
    if st_cap and st_cap > 0:
        st = max(-st_cap, min(st_cap, special_teams_raw))
    else:
        st = special_teams_raw

    # Reassemble spread
    # Note: env_score includes HFA + travel + altitude (already smoothed)
    # We need to compute the delta from using different HFA offset
    # The original spread used: env_score (which includes smoothed HFA from home_field_raw - offset)
    # For reassembly, we substitute the new HFA value
    #
    # Original spread = base_margin + env_score + situational + st + fcs + pace + qb
    # where env_score = smoothed(hfa + travel + altitude) + rest + consec
    #
    # For dual-cap, we can't perfectly reconstruct the smoothing without the aggregator,
    # so we use a simpler approach: just adjust the HFA and ST components relative to
    # what was already in the spread.
    #
    # Actually, the cleanest approach: since we store all components, we rebuild from scratch
    # But we need the smoothing factor from aggregator... which we don't have.
    #
    # Simpler: the spread is already computed with the original cap/offset.
    # For dual-cap reassembly, we need to store the original predicted_spread and adjust it.
    #
    # Delta approach:
    # new_spread = original_spread - old_st + new_st - old_hfa + new_hfa
    #
    # This function is called from vectorized code, so we need a different approach.
    # Let's just compute a simple sum (ignoring smoothing for now, as it's a minor effect):

    return (
        base_margin
        + hfa
        + situational
        + travel
        + altitude
        + st
        + fcs_adj
        + pace_adj
        + qb_adj
    )


def assemble_spread_vectorized(
    pred_df: pd.DataFrame,
    st_cap: Optional[float] = 2.5,
    hfa_offset: float = 0.50,
) -> np.ndarray:
    """Vectorized spread assembly for dual-cap mode.

    Given a DataFrame with raw component columns, reassemble spreads with
    the specified ST cap and HFA offset.

    The challenge: original spread includes smoothed env_score from aggregator,
    which we can't recompute here. Instead, we compute the delta from changing
    the ST cap and HFA offset, and apply that to the original spread.

    Delta = (new_st - old_st) + (new_hfa - old_hfa)
    new_spread = original_spread + delta

    Args:
        pred_df: DataFrame with columns: predicted_spread, special_teams_raw,
                 special_teams, home_field_raw, hfa
        st_cap: Cap on ST differential (None or 0 = no cap)
        hfa_offset: Points subtracted from raw HFA

    Returns:
        Array of reassembled spreads
    """
    # Get raw and original values
    st_raw = pred_df["special_teams_raw"].values
    st_original = pred_df["special_teams"].values
    hfa_raw = pred_df["home_field_raw"].values
    hfa_original = pred_df["hfa"].values
    original_spread = pred_df["predicted_spread"].values

    # Compute new ST (with new cap)
    if st_cap and st_cap > 0:
        st_new = np.clip(st_raw, -st_cap, st_cap)
    else:
        st_new = st_raw

    # Compute new HFA (with new offset)
    # Apply floor at 0.5 for non-neutral games (hfa_raw > 0)
    hfa_new = np.where(
        hfa_raw > 0,
        np.maximum(0.5, hfa_raw - hfa_offset),
        0.0
    )

    # Compute deltas
    st_delta = st_new - st_original
    hfa_delta = hfa_new - hfa_original

    # Apply deltas to original spread
    return original_spread + st_delta + hfa_delta


def calculate_dual_ats_results(
    predictions: list[dict],
    betting_df: pl.DataFrame,
    st_cap_open: float,
    st_cap_close: float,
    hfa_offset_open: float,
    hfa_offset_close: float,
) -> pd.DataFrame:
    """Calculate ATS for all 4 timing combinations in dual-cap mode.

    Evaluates spreads reassembled with different ST cap and HFA offset
    parameters against both opening and closing lines.

    Combinations evaluated:
    1. open-tuned spread vs open line (primary)
    2. close-tuned spread vs close line (primary)
    3. open-tuned spread vs close line (diagnostic: does open-tuning beat sharp close?)
    4. close-tuned spread vs open line (diagnostic: cross-validation)

    Args:
        predictions: List of prediction dicts with raw component values
        betting_df: Polars DataFrame with Vegas lines
        st_cap_open: ST cap for open-tuned spread
        st_cap_close: ST cap for close-tuned spread
        hfa_offset_open: HFA offset for open-tuned spread
        hfa_offset_close: HFA offset for close-tuned spread

    Returns:
        DataFrame with ATS results for all 4 combinations
    """
    if not predictions:
        return pd.DataFrame()

    # Convert to DataFrames
    pred_df = pd.DataFrame(predictions)
    betting_pd = betting_df.to_pandas()

    # Merge predictions with betting data
    merged = pred_df.merge(
        betting_pd[["game_id", "spread_open", "spread_close"]],
        on="game_id",
        how="left",
    )

    # Filter to games with both lines
    has_open = merged["spread_open"].notna()
    has_close = merged["spread_close"].notna()
    df = merged[has_open & has_close].copy()

    if len(df) == 0:
        logger.warning("No predictions matched with both open and close lines for dual-cap eval")
        return pd.DataFrame()

    # Reassemble spreads with open-tuned and close-tuned parameters
    spread_open_tuned = assemble_spread_vectorized(df, st_cap_open, hfa_offset_open)
    spread_close_tuned = assemble_spread_vectorized(df, st_cap_close, hfa_offset_close)

    # Store reassembled spreads
    df["spread_open_tuned"] = spread_open_tuned
    df["spread_close_tuned"] = spread_close_tuned

    # Get Vegas spreads
    vegas_open = df["spread_open"].values
    vegas_close = df["spread_close"].values
    actual_margin = df["actual_margin"].values

    # Convert model spreads to Vegas convention (our: +home, Vegas: -home)
    model_open_vegas = -spread_open_tuned
    model_close_vegas = -spread_close_tuned

    # Calculate edges for all 4 combinations
    # Edge = model_spread - vegas_spread (positive = model picks away more than Vegas)
    edge_open_vs_open = model_open_vegas - vegas_open
    edge_close_vs_close = model_close_vegas - vegas_close
    edge_open_vs_close = model_open_vegas - vegas_close  # Diagnostic
    edge_close_vs_open = model_close_vegas - vegas_open  # Diagnostic

    # Determine picks (edge < 0 = model likes home more)
    pick_open_open = edge_open_vs_open < 0  # Home pick
    pick_close_close = edge_close_vs_close < 0
    pick_open_close = edge_open_vs_close < 0
    pick_close_open = edge_close_vs_open < 0

    # Calculate ATS results for each combination
    # Home covers if actual_margin + vegas_spread > 0
    home_cover_open = actual_margin + vegas_open
    home_cover_close = actual_margin + vegas_close

    # ATS win: if picking home, home must cover; if picking away, home must NOT cover
    df["ats_open_open"] = np.where(pick_open_open, home_cover_open > 0, home_cover_open < 0)
    df["ats_close_close"] = np.where(pick_close_close, home_cover_close > 0, home_cover_close < 0)
    df["ats_open_close"] = np.where(pick_open_close, home_cover_close > 0, home_cover_close < 0)
    df["ats_close_open"] = np.where(pick_close_open, home_cover_open > 0, home_cover_open < 0)

    # Track pushes
    df["push_open"] = home_cover_open == 0
    df["push_close"] = home_cover_close == 0

    # Store edges for filtering
    df["edge_open_open"] = np.abs(edge_open_vs_open)
    df["edge_close_close"] = np.abs(edge_close_vs_close)
    df["edge_open_close"] = np.abs(edge_open_vs_close)
    df["edge_close_open"] = np.abs(edge_close_vs_open)

    # Store picks
    df["pick_open_open"] = np.where(pick_open_open, "HOME", "AWAY")
    df["pick_close_close"] = np.where(pick_close_close, "HOME", "AWAY")

    return df


def print_dual_cap_report(
    dual_df: pd.DataFrame,
    st_cap_open: float,
    st_cap_close: float,
    hfa_offset_open: float,
    hfa_offset_close: float,
    start_week: int = 4,
    end_week: int = 15,
) -> None:
    """Print dual-cap ATS report with all 4 timing combinations.

    Args:
        dual_df: DataFrame from calculate_dual_ats_results
        st_cap_open/close: ST cap parameters
        hfa_offset_open/close: HFA offset parameters
        start_week/end_week: Filter to Core phase
    """
    # Filter to Core weeks
    core_df = dual_df[(dual_df["week"] >= start_week) & (dual_df["week"] <= end_week)]

    if len(core_df) == 0:
        print("No games in Core phase for dual-cap report")
        return

    print("\n" + "=" * 80)
    print("DUAL-CAP MODE ATS RESULTS (Core Weeks 4-15)")
    print("=" * 80)
    print(f"  Open-tuned params:  ST cap = {st_cap_open}, HFA offset = {hfa_offset_open}")
    print(f"  Close-tuned params: ST cap = {st_cap_close}, HFA offset = {hfa_offset_close}")
    print("-" * 80)

    def calc_wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
        """Calculate Wilson score confidence interval."""
        if total == 0:
            return (0.0, 0.0)
        p = wins / total
        denom = 1 + z * z / total
        center = (p + z * z / (2 * total)) / denom
        spread = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
        return (center - spread, center + spread)

    def calc_roi(wins: int, losses: int) -> float:
        """Calculate ROI assuming -110 odds."""
        if wins + losses == 0:
            return 0.0
        profit = wins * 100 - losses * 110
        wagered = (wins + losses) * 110
        return (profit / wagered) * 100 if wagered > 0 else 0.0

    def print_combo_stats(df: pd.DataFrame, edge_col: str, ats_col: str, push_col: str, label: str):
        """Print ATS stats for one combination at multiple edge thresholds."""
        print(f"\n{label}:")
        for edge_thresh in [0, 3, 5]:
            if edge_thresh > 0:
                subset = df[df[edge_col] >= edge_thresh]
                label_str = f"  {edge_thresh}+ Edge"
            else:
                subset = df
                label_str = "  All Picks"

            wins = subset[ats_col].sum()
            pushes = subset[push_col].sum()
            losses = len(subset) - wins - pushes
            total = wins + losses
            pct = wins / total * 100 if total > 0 else 0.0
            roi = calc_roi(wins, losses)
            ci_low, ci_high = calc_wilson_ci(wins, total)

            print(f"{label_str:12} | {wins:3}-{losses:3} ({pct:5.1f}%) | ROI: {roi:+5.1f}% | 95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%] | N={total}")

    # Primary combinations
    print_combo_stats(core_df, "edge_open_open", "ats_open_open", "push_open",
                      "Open-Tuned vs Open Line (PRIMARY)")
    print_combo_stats(core_df, "edge_close_close", "ats_close_close", "push_close",
                      "Close-Tuned vs Close Line (PRIMARY)")

    # Diagnostic combinations
    print_combo_stats(core_df, "edge_open_close", "ats_open_close", "push_close",
                      "Open-Tuned vs Close Line (DIAG)")
    print_combo_stats(core_df, "edge_close_open", "ats_close_open", "push_open",
                      "Close-Tuned vs Open Line (DIAG)")

    print("\n" + "-" * 80)
    print("INTERPRETATION:")
    print("  - If open-tuned beats close line better than close-tuned, timing split justified")
    print("  - If close-tuned beats open line better than open-tuned, timing split NOT justified")
    print("=" * 80)


def run_dual_cap_sweep(
    all_predictions: list[dict],
    betting_df: pl.DataFrame,
    years: list[int],
    st_cap_grid: list,
    hfa_offset: float = 0.50,
    start_week: int = 4,
    end_week: int = 15,
    min_bets: int = 30,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run LOO-CV sweep to find optimal per-timing ST caps.

    For each holdout year, use the other years to find the best ST cap,
    then evaluate on the holdout year. This prevents overfitting to
    a single year's quirks.

    Args:
        all_predictions: List of prediction dicts from run_backtest
        betting_df: Polars DataFrame with betting lines
        years: List of years in the backtest
        st_cap_grid: List of ST cap values to try (None = no cap)
        hfa_offset: Fixed HFA offset (0.50 default)
        start_week: Start of Core phase
        end_week: End of Core phase
        min_bets: Minimum bets at 5+ edge to be selectable

    Returns:
        (sweep_results_df, stability_analysis, best_params)
    """
    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(all_predictions)
    betting_pd = betting_df.to_pandas()

    # Merge with betting data once
    merged = pred_df.merge(
        betting_pd[["game_id", "spread_open", "spread_close"]],
        on="game_id",
        how="left",
    )
    # Filter to games with both lines and Core weeks
    has_lines = merged["spread_open"].notna() & merged["spread_close"].notna()
    core_mask = (merged["week"] >= start_week) & (merged["week"] <= end_week)
    df = merged[has_lines & core_mask].copy()

    if len(df) == 0:
        logger.warning("No games matched for dual-cap sweep")
        return pd.DataFrame(), {}, {}

    results = []
    stability = {"open": {}, "close": {}}

    for holdout_year in years:
        # Split data
        train_mask = df["year"] != holdout_year
        holdout_mask = df["year"] == holdout_year
        train_df = df[train_mask]
        holdout_df = df[holdout_mask].copy()

        if len(holdout_df) < 50:
            logger.info(f"Skipping {holdout_year}: only {len(holdout_df)} holdout games")
            continue

        # Sweep ST caps on training data
        # P3.5: Decoupled search - best open cap is independent of best close cap
        # Reduces from O(grid²) to O(grid) calls to assemble_spread_vectorized
        best_open_cap = 2.5
        best_close_cap = 2.5
        best_open_roi = -float("inf")
        best_close_roi = -float("inf")

        # Pre-extract Vegas lines and actual margins (constant across cap values)
        vegas_open = train_df["spread_open"].values
        vegas_close = train_df["spread_close"].values
        actual_margin = train_df["actual_margin"].values

        for st_cap in st_cap_grid:
            # Reassemble spreads with this cap (single call per cap value)
            spread_reassembled = assemble_spread_vectorized(train_df, st_cap, hfa_offset)

            # Evaluate vs OPEN line
            edge_open = np.abs(-spread_reassembled - vegas_open)
            mask_5_open = edge_open >= 5
            if mask_5_open.sum() >= min_bets:
                home_cover = actual_margin[mask_5_open] + vegas_open[mask_5_open]
                pick_home = (-spread_reassembled[mask_5_open] - vegas_open[mask_5_open]) < 0
                ats_win = np.where(pick_home, home_cover > 0, home_cover < 0)
                wins = ats_win.sum()
                losses = mask_5_open.sum() - wins - (home_cover == 0).sum()
                roi_open = (wins * 100 - losses * 110) / ((wins + losses) * 110) * 100 if (wins + losses) > 0 else 0

                if roi_open > best_open_roi:
                    best_open_roi = roi_open
                    best_open_cap = st_cap

            # Evaluate vs CLOSE line (same reassembled spread)
            edge_close = np.abs(-spread_reassembled - vegas_close)
            mask_5_close = edge_close >= 5
            if mask_5_close.sum() >= min_bets:
                home_cover = actual_margin[mask_5_close] + vegas_close[mask_5_close]
                pick_home = (-spread_reassembled[mask_5_close] - vegas_close[mask_5_close]) < 0
                ats_win = np.where(pick_home, home_cover > 0, home_cover < 0)
                wins = ats_win.sum()
                losses = mask_5_close.sum() - wins - (home_cover == 0).sum()
                roi_close = (wins * 100 - losses * 110) / ((wins + losses) * 110) * 100 if (wins + losses) > 0 else 0

                if roi_close > best_close_roi:
                    best_close_roi = roi_close
                    best_close_cap = st_cap

        # Track stability
        stability["open"][holdout_year] = best_open_cap
        stability["close"][holdout_year] = best_close_cap

        # Evaluate on holdout with best caps found from training
        spread_open_holdout = assemble_spread_vectorized(holdout_df, best_open_cap, hfa_offset)
        spread_close_holdout = assemble_spread_vectorized(holdout_df, best_close_cap, hfa_offset)

        vegas_open_h = holdout_df["spread_open"].values
        vegas_close_h = holdout_df["spread_close"].values
        actual_margin_h = holdout_df["actual_margin"].values

        # Evaluate open-tuned vs open on holdout
        edge_open_h = np.abs(-spread_open_holdout - vegas_open_h)
        mask_5_open_h = edge_open_h >= 5
        if mask_5_open_h.sum() > 0:
            home_cover = actual_margin_h[mask_5_open_h] + vegas_open_h[mask_5_open_h]
            pick_home = (-spread_open_holdout[mask_5_open_h] - vegas_open_h[mask_5_open_h]) < 0
            ats_win = np.where(pick_home, home_cover > 0, home_cover < 0)
            wins_open = ats_win.sum()
            losses_open = mask_5_open_h.sum() - wins_open - (home_cover == 0).sum()
        else:
            wins_open, losses_open = 0, 0

        # Evaluate close-tuned vs close on holdout
        edge_close_h = np.abs(-spread_close_holdout - vegas_close_h)
        mask_5_close_h = edge_close_h >= 5
        if mask_5_close_h.sum() > 0:
            home_cover = actual_margin_h[mask_5_close_h] + vegas_close_h[mask_5_close_h]
            pick_home = (-spread_close_holdout[mask_5_close_h] - vegas_close_h[mask_5_close_h]) < 0
            ats_win = np.where(pick_home, home_cover > 0, home_cover < 0)
            wins_close = ats_win.sum()
            losses_close = mask_5_close_h.sum() - wins_close - (home_cover == 0).sum()
        else:
            wins_close, losses_close = 0, 0

        results.append({
            "holdout_year": holdout_year,
            "best_open_cap": best_open_cap if best_open_cap is not None else "none",
            "best_close_cap": best_close_cap if best_close_cap is not None else "none",
            "open_5plus_wins": wins_open,
            "open_5plus_losses": losses_open,
            "open_5plus_pct": wins_open / (wins_open + losses_open) * 100 if (wins_open + losses_open) > 0 else 0,
            "close_5plus_wins": wins_close,
            "close_5plus_losses": losses_close,
            "close_5plus_pct": wins_close / (wins_close + losses_close) * 100 if (wins_close + losses_close) > 0 else 0,
        })

    results_df = pd.DataFrame(results)

    # Determine overall best params (most common across folds or with best holdout performance)
    open_caps = [r["best_open_cap"] for r in results if r["best_open_cap"]]
    close_caps = [r["best_close_cap"] for r in results if r["best_close_cap"]]

    from collections import Counter
    best_open = Counter(open_caps).most_common(1)[0][0] if open_caps else 2.5
    best_close = Counter(close_caps).most_common(1)[0][0] if close_caps else 2.5

    # Check stability
    open_stable = len(set(open_caps)) <= 2  # At most 2 different values
    close_stable = len(set(close_caps)) <= 2

    best_params = {
        "st_cap_open": best_open,
        "st_cap_close": best_close,
        "open_stable": open_stable,
        "close_stable": close_stable,
    }

    return results_df, stability, best_params


def print_dual_cap_sweep_report(
    sweep_df: pd.DataFrame,
    stability: dict,
    best_params: dict,
) -> None:
    """Print dual-cap sweep results."""
    print("\n" + "=" * 80)
    print("DUAL-CAP LOO-CV SWEEP RESULTS")
    print("=" * 80)

    if len(sweep_df) == 0:
        print("No sweep results to display")
        return

    print("\nPER-FOLD RESULTS (5+ Edge):")
    print("-" * 80)
    print(f"{'Year':6} | {'Open Cap':8} | {'Open ATS':12} | {'Close Cap':9} | {'Close ATS':12}")
    print("-" * 80)

    for _, row in sweep_df.iterrows():
        open_ats = f"{row['open_5plus_wins']:.0f}-{row['open_5plus_losses']:.0f} ({row['open_5plus_pct']:.1f}%)"
        close_ats = f"{row['close_5plus_wins']:.0f}-{row['close_5plus_losses']:.0f} ({row['close_5plus_pct']:.1f}%)"
        print(f"{row['holdout_year']:6} | {str(row['best_open_cap']):8} | {open_ats:12} | {str(row['best_close_cap']):9} | {close_ats:12}")

    print("-" * 80)

    # Summary
    total_open_wins = sweep_df["open_5plus_wins"].sum()
    total_open_losses = sweep_df["open_5plus_losses"].sum()
    total_close_wins = sweep_df["close_5plus_wins"].sum()
    total_close_losses = sweep_df["close_5plus_losses"].sum()

    open_pct = total_open_wins / (total_open_wins + total_open_losses) * 100 if (total_open_wins + total_open_losses) > 0 else 0
    close_pct = total_close_wins / (total_close_wins + total_close_losses) * 100 if (total_close_wins + total_close_losses) > 0 else 0

    print(f"\nAGGREGATE LOO-CV PERFORMANCE:")
    print(f"  Open 5+ Edge:  {total_open_wins:.0f}-{total_open_losses:.0f} ({open_pct:.1f}%)")
    print(f"  Close 5+ Edge: {total_close_wins:.0f}-{total_close_losses:.0f} ({close_pct:.1f}%)")

    print(f"\nRECOMMENDED PARAMS:")
    print(f"  ST cap (open):  {best_params['st_cap_open']} {'(STABLE)' if best_params['open_stable'] else '(UNSTABLE - varies across folds)'}")
    print(f"  ST cap (close): {best_params['st_cap_close']} {'(STABLE)' if best_params['close_stable'] else '(UNSTABLE - varies across folds)'}")

    # Stability detail
    if not best_params['open_stable'] or not best_params['close_stable']:
        print(f"\nSTABILITY DETAIL:")
        print(f"  Open caps by year:  {stability.get('open', {})}")
        print(f"  Close caps by year: {stability.get('close', {})}")

    print("=" * 80)


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
    display_cols = ["Phase", "Weeks", "Games", "MAE", "RMSE", "MAE vs Close", "ATS %", "3+ Edge", "5+ Edge", "Mean CLV"]
    available_cols = [c for c in display_cols if c in phase_metrics.columns]

    # Format numeric columns
    for col in ["MAE", "RMSE", "MAE vs Close", "ATS %", "Mean CLV"]:
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
        # Extract fields from SeasonData (named access prevents positional bugs)
        sd = season_data[year]
        games_df = sd.games_df
        betting_df = sd.betting_df
        efficiency_plays_df = sd.efficiency_plays_df
        fbs_teams = sd.fbs_teams

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
            if sd.priors:
                print(f"    Preseason priors: {len(sd.priors.preseason_ratings)} teams")

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
    use_blended_priors: bool = False,
    blend_schedule: Optional[BlendSchedule] = None,
    credible_rebuild_enabled: bool = True,
) -> dict[int, SeasonData]:
    """Fetch and cache all season data for backtest.

    Args:
        years: List of years to fetch
        use_priors: Whether to build preseason priors
        use_portal: Whether to incorporate transfer portal data into priors
        portal_scale: How much to weight portal impact (default 0.15)
        use_cache: Whether to use cached data if available (default True)
        force_refresh: If True, bypass cache and force fresh API calls (default False)
        use_blended_priors: Whether to use blended SP+/own-prior system
        blend_schedule: Custom BlendSchedule for blended priors (uses default if None)
        credible_rebuild_enabled: Whether to apply credible rebuild relief (default True)

    Returns:
        Dict mapping year to SeasonData namedtuple with all required DataFrames.
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
                if use_blended_priors:
                    # Load historical JP+ ratings for blended priors
                    import json
                    hist_ratings_path = Path("data/historical_jp_ratings.json")
                    if hist_ratings_path.exists():
                        with open(hist_ratings_path) as f:
                            historical_jp_ratings = json.load(f)
                        # Use provided schedule or default
                        if blend_schedule is None:
                            schedule = BlendSchedule(
                                week_weights={
                                    1: (0.0, 1.0),   # Week 1: 100% own-prior
                                    2: (0.4, 0.6),   # Week 2: 60% own, 40% SP+
                                    3: (0.0, 1.0),   # Week 3: 100% own-prior
                                },
                                default_sp_weight=0.8,  # Week 4+: 80% SP+
                            )
                        else:
                            schedule = blend_schedule
                        priors = create_blended_generator(
                            client, historical_jp_ratings, schedule
                        )
                        priors.calculate_blended_ratings(year, week=1)
                        logger.info(
                            f"Loaded BLENDED priors for {len(priors.blended_ratings)} teams"
                        )
                    else:
                        logger.warning(
                            "Historical ratings not found, falling back to SP+-only"
                        )
                        priors = PreseasonPriors(
                            client,
                            credible_rebuild_enabled=credible_rebuild_enabled,
                        )
                        priors.calculate_preseason_ratings(
                            year, use_portal=use_portal, portal_scale=portal_scale
                        )
                else:
                    priors = PreseasonPriors(
                        client,
                        credible_rebuild_enabled=credible_rebuild_enabled,
                    )
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

        season_data[year] = SeasonData(
            games_df=games_df,
            betting_df=betting_df,
            plays_df=plays_df,
            turnover_df=turnover_df,
            priors=priors,
            efficiency_plays_df=efficiency_plays_df,
            fbs_teams=fbs_teams,
            st_plays_df=st_plays_df,
            historical_rankings=historical_rankings,
            team_conferences=team_conferences,
        )

    return season_data


def _process_single_season(
    year: int,
    season_data: SeasonData,
    team_records: dict,
    start_week: int,
    end_week: Optional[int],
    hfa_value: float,
    prior_weight: int,
    ridge_alpha: float,
    efficiency_weight: float,
    explosiveness_weight: float,
    turnover_weight: float,
    k_int: float,  # Bayesian shrinkage for INT (skill-based)
    k_fumble: float,  # Bayesian shrinkage for fumbles (luck-based)
    garbage_time_weight: float,
    asymmetric_garbage: bool,
    fcs_penalty_elite: float,
    fcs_penalty_standard: float,
    use_opening_line: bool,
    hfa_global_offset: float = 0.0,
    ooc_credibility_weight: float = 0.0,
    st_shrink_enabled: bool = False,
    st_k_fg: float = 6.0,
    st_k_punt: float = 6.0,
    st_k_ko: float = 6.0,
    st_spread_cap: Optional[float] = 2.5,
    st_early_weight: float = 1.0,
    fcs_static: bool = False,
    fcs_k: float = 8.0,
    fcs_baseline: float = -28.0,
    fcs_min_pen: float = 10.0,
    fcs_max_pen: float = 45.0,
    fcs_slope: float = 0.8,
    fcs_intercept: float = 10.0,
    fcs_hfa: float = 0.0,
    # Learned Situational Adjustment (LSA) parameters
    use_learned_situ: bool = False,
    lsa_alpha: float = 300.0,
    lsa_min_games: int = 150,
    lsa_ema: float = 0.3,
    lsa_clamp_max: Optional[float] = 4.0,
    lsa_prior_data: Optional[list] = None,  # Prior years' training data
    # LSA training signal quality parameters
    lsa_adjust_turnovers: bool = False,
    lsa_turnover_value: float = 4.0,
    lsa_filter_vegas: Optional[float] = None,
    lsa_weighted_training: bool = False,
    lsa_weight_spread: float = 17.0,
    lsa_min_training_games: int = 30,
    # QB Continuous Rating parameters
    use_qb_continuous: bool = False,
    qb_shrinkage_k: float = 200.0,
    qb_cap: float = 3.0,
    qb_scale: float = 4.0,
    qb_prior_decay: float = 0.3,
    qb_use_prior_season: bool = True,
    qb_phase1_only: bool = False,
    qb_fix_misattribution: bool = False,
    fcs_use_deviation: bool = True,  # Use FBS team's deviation from FBS mean for FCS games
    lock_prior_weeks: int = 0,  # Force 100% prior weight for weeks <= this value
    phase1_shrinkage: float = 0.90,  # Shrinkage for Phase 1 spread predictions
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

    # Configure ST shrinkage settings for this worker process
    # Each spawned worker has its own settings singleton instance
    settings = get_settings()
    settings.st_shrink_enabled = st_shrink_enabled
    settings.st_k_fg = st_k_fg
    settings.st_k_punt = st_k_punt
    settings.st_k_ko = st_k_ko

    # Extract fields from SeasonData (named access prevents positional bugs)
    games_df = season_data.games_df
    betting_df = season_data.betting_df
    turnover_df = season_data.turnover_df
    efficiency_plays_df = season_data.efficiency_plays_df
    fbs_teams = season_data.fbs_teams
    st_plays_df = season_data.st_plays_df
    priors = season_data.priors
    historical_rankings = season_data.historical_rankings
    team_conferences = season_data.team_conferences

    # Create FCS estimator for this season (if not using static mode)
    fcs_estimator = None
    if not fcs_static:
        fcs_estimator = FCSStrengthEstimator(
            hfa_value=fcs_hfa,
            k_fcs=fcs_k,
            baseline_margin=fcs_baseline,
            min_penalty=fcs_min_pen,
            max_penalty=fcs_max_pen,
            slope=fcs_slope,
            intercept=fcs_intercept,
        )

    # Create LSA model if enabled
    lsa_model = None
    if use_learned_situ:
        from pathlib import Path
        persist_dir = Path(__file__).parent.parent / "data" / "learned_situ_coefficients"
        lsa_model = LearnedSituationalModel(
            ridge_alpha=lsa_alpha,
            min_games=lsa_min_games,
            ema_beta=lsa_ema,
            clamp_max=lsa_clamp_max,
            persist_dir=persist_dir,
            # Training signal quality parameters
            adjust_for_turnovers=lsa_adjust_turnovers,
            turnover_point_value=lsa_turnover_value,
            max_abs_vegas_spread=lsa_filter_vegas,
            use_sample_weights=lsa_weighted_training,
            weight_spread_threshold=lsa_weight_spread,
            min_training_games=lsa_min_training_games,
        )
        lsa_model.reset(year)
        # Seed with prior years' training data
        if lsa_prior_data:
            lsa_model.seed_with_prior_data(lsa_prior_data)
            logger.debug(f"LSA seeded with {len(lsa_prior_data)} prior games for {year}")

    # Walk-forward predictions
    predictions, lsa_training_data = walk_forward_predict(
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
        k_int=k_int,
        k_fumble=k_fumble,
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
        st_spread_cap=st_spread_cap,
        st_early_weight=st_early_weight,
        fcs_estimator=fcs_estimator,
        fcs_static=fcs_static,
        fcs_use_deviation=fcs_use_deviation,
        lsa_model=lsa_model,
        turnover_df=turnover_df,
        betting_df=betting_df,
        # QB Continuous Rating params
        use_qb_continuous=use_qb_continuous,
        qb_shrinkage_k=qb_shrinkage_k,
        qb_cap=qb_cap,
        qb_scale=qb_scale,
        qb_prior_decay=qb_prior_decay,
        qb_use_prior_season=qb_use_prior_season,
        qb_phase1_only=qb_phase1_only,
        qb_fix_misattribution=qb_fix_misattribution,
        lock_prior_weeks=lock_prior_weeks,
        phase1_shrinkage=phase1_shrinkage,
    )

    # Calculate ATS
    ats_results = None
    if len(betting_df) > 0:
        ats_results = calculate_ats_results(predictions, betting_df, use_opening_line)

    # Year MAE for logging
    year_df = pd.DataFrame(predictions)
    year_mae = year_df["abs_error"].mean() if not year_df.empty else None

    return year, predictions, ats_results, year_mae, lsa_training_data


def run_backtest(
    years: list[int],
    start_week: int = 1,
    end_week: Optional[int] = None,
    ridge_alpha: float = 50.0,
    use_priors: bool = True,
    use_blended_priors: bool = False,
    hfa_value: float = 2.5,
    prior_weight: int = 8,
    season_data: Optional[dict] = None,
    efficiency_weight: float = 0.45,
    explosiveness_weight: float = 0.45,
    turnover_weight: float = 0.10,
    k_int: float = 10.0,  # Bayesian shrinkage for INT (skill-based)
    k_fumble: float = 30.0,  # Bayesian shrinkage for fumbles (luck-based)
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
    st_shrink_enabled: bool = False,
    st_k_fg: float = 6.0,
    st_k_punt: float = 6.0,
    st_k_ko: float = 6.0,
    st_spread_cap: Optional[float] = 2.5,
    st_early_weight: float = 1.0,
    fcs_static: bool = False,
    fcs_k: float = 8.0,
    fcs_baseline: float = -28.0,
    fcs_min_pen: float = 10.0,
    fcs_max_pen: float = 45.0,
    fcs_slope: float = 0.8,
    fcs_intercept: float = 10.0,
    fcs_hfa: float = 0.0,
    fcs_use_deviation: bool = True,  # Use FBS team's deviation from FBS mean for FCS games
    # Learned Situational Adjustment (LSA) parameters
    use_learned_situ: bool = False,
    lsa_alpha: float = 300.0,
    lsa_min_games: int = 150,
    lsa_ema: float = 0.3,
    lsa_clamp_max: Optional[float] = 4.0,
    # LSA training signal quality parameters
    lsa_adjust_turnovers: bool = False,
    lsa_turnover_value: float = 4.0,
    lsa_filter_vegas: Optional[float] = None,
    lsa_weighted_training: bool = False,
    lsa_weight_spread: float = 17.0,
    lsa_min_training_games: int = 30,
    # QB Continuous Rating parameters
    use_qb_continuous: bool = False,
    qb_shrinkage_k: float = 200.0,
    qb_cap: float = 3.0,
    qb_scale: float = 4.0,
    qb_prior_decay: float = 0.3,
    qb_use_prior_season: bool = True,
    qb_phase1_only: bool = False,
    qb_fix_misattribution: bool = False,
    # Blended prior schedule
    blend_schedule: Optional[BlendSchedule] = None,
    # Credible Rebuild adjustment
    credible_rebuild_enabled: bool = True,
    # Phase 1 Purity Test
    lock_prior_weeks: int = 0,  # Force 100% prior weight for weeks <= this value
    # Phase 1 Spread Shrinkage
    phase1_shrinkage: float = 0.90,  # Shrinkage for weeks 1-3 predictions
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
        credible_rebuild_enabled: Apply credible rebuild relief for low-RP teams (default True)
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
        st_shrink_enabled: Whether to apply Empirical Bayes shrinkage to ST ratings (default True)
        st_k_fg: Shrinkage k for FG component (attempts needed to trust rating fully). Default 20.0.
        st_k_punt: Shrinkage k for punt component. Default 15.0.
        st_k_ko: Shrinkage k for kickoff component. Default 15.0.
        fcs_static: If True, use static elite list instead of dynamic estimator. Default False.
        fcs_k: FCS shrinkage k (games for 50% trust). Default 8.0.
        fcs_baseline: Prior margin for unknown FCS teams (FCS - FBS). Default -28.0.
        fcs_min_pen: Minimum penalty for elite FCS. Default 10.0.
        fcs_max_pen: Maximum penalty for weak FCS. Default 45.0.
        fcs_slope: Penalty increase per point of avg loss. Default 0.8.
        fcs_intercept: Base penalty (elite FCS with 0 avg loss). Default 10.0.
        use_learned_situ: Enable learned situational adjustment via ridge regression. Default False.
        lsa_alpha: LSA ridge regularization strength. Default 10.0.
        lsa_min_games: Minimum games before LSA is used. Default 150.
        lsa_ema: LSA coefficient smoothing factor. Default 0.3.

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
            use_blended_priors=use_blended_priors,
            blend_schedule=blend_schedule,
            credible_rebuild_enabled=credible_rebuild_enabled,
        )
    else:
        # Shallow copy to avoid mutating caller's dict (sweep reuses cached_data)
        season_data = {**season_data}

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
        priors = season_data[year].priors
        if priors is not None:
            if hasattr(priors, "client"):
                priors.client = None
            # Also strip clients from BlendedPriorGenerator's internal generators
            if hasattr(priors, "sp_gen") and hasattr(priors.sp_gen, "client"):
                priors.sp_gen.client = None
            if hasattr(priors, "own_gen") and hasattr(priors.own_gen, "client"):
                priors.own_gen.client = None

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
        k_int=k_int,
        k_fumble=k_fumble,
        garbage_time_weight=garbage_time_weight,
        asymmetric_garbage=asymmetric_garbage,
        fcs_penalty_elite=fcs_penalty_elite,
        fcs_penalty_standard=fcs_penalty_standard,
        use_opening_line=use_opening_line,
        hfa_global_offset=hfa_global_offset,
        ooc_credibility_weight=ooc_credibility_weight,
        st_shrink_enabled=st_shrink_enabled,
        st_k_fg=st_k_fg,
        st_k_punt=st_k_punt,
        st_k_ko=st_k_ko,
        st_spread_cap=st_spread_cap,
        st_early_weight=st_early_weight,
        # FCS dynamic estimator params
        fcs_static=fcs_static,
        fcs_k=fcs_k,
        fcs_baseline=fcs_baseline,
        fcs_min_pen=fcs_min_pen,
        fcs_max_pen=fcs_max_pen,
        fcs_slope=fcs_slope,
        fcs_intercept=fcs_intercept,
        fcs_hfa=fcs_hfa,
        fcs_use_deviation=fcs_use_deviation,
        # LSA params
        use_learned_situ=use_learned_situ,
        lsa_alpha=lsa_alpha,
        lsa_min_games=lsa_min_games,
        lsa_ema=lsa_ema,
        lsa_clamp_max=lsa_clamp_max,
        # LSA training signal quality params
        lsa_adjust_turnovers=lsa_adjust_turnovers,
        lsa_turnover_value=lsa_turnover_value,
        lsa_filter_vegas=lsa_filter_vegas,
        lsa_weighted_training=lsa_weighted_training,
        lsa_weight_spread=lsa_weight_spread,
        lsa_min_training_games=lsa_min_training_games,
        # QB Continuous Rating params
        use_qb_continuous=use_qb_continuous,
        qb_shrinkage_k=qb_shrinkage_k,
        qb_cap=qb_cap,
        qb_scale=qb_scale,
        qb_prior_decay=qb_prior_decay,
        qb_use_prior_season=qb_use_prior_season,
        qb_phase1_only=qb_phase1_only,
        qb_fix_misattribution=qb_fix_misattribution,
        # Phase 1 Purity Test
        lock_prior_weeks=lock_prior_weeks,
        # Phase 1 Spread Shrinkage
        phase1_shrinkage=phase1_shrinkage,
    )

    # LSA requires sequential processing: prior years' data feeds later years
    if use_learned_situ:
        logger.info(f"LSA enabled: processing {len(years)} seasons sequentially for multi-year pooling")
        lsa_prior_data = []  # Accumulate training data across years

        for year in sorted(years):
            logger.debug(f"Backtesting {year} season (LSA prior data: {len(lsa_prior_data)} games)...")
            year_val, predictions, ats_results, year_mae, year_lsa_data = _process_single_season(
                year=year,
                season_data=season_data[year],
                lsa_prior_data=lsa_prior_data,  # Pass accumulated data
                **season_kwargs,
            )
            all_predictions.extend(predictions)
            if ats_results is not None:
                all_ats.append(ats_results)
            if year_mae is not None:
                logger.debug(f"{year_val} MAE: {year_mae:.2f}")
            # Accumulate this year's training data for next year
            lsa_prior_data.extend(year_lsa_data)

    elif len(years) > 1:
        # Parallel execution: each season runs in its own process
        n_workers = min(len(years), os.cpu_count() or 4)
        logger.info(f"Running {len(years)} seasons in parallel ({n_workers} workers)")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_season,
                    year=year,
                    season_data=season_data[year],
                    **season_kwargs,
                ): year
                for year in years
            }

            for future in futures:
                try:
                    year_val, predictions, ats_results, year_mae, _ = future.result()
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
            year_val, predictions, ats_results, year_mae, _ = _process_single_season(
                year=year,
                season_data=season_data[year],
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

    # Combine betting data from all years for dual-cap mode
    all_betting = []
    for year in years:
        if year in season_data and season_data[year].betting_df is not None:
            all_betting.append(season_data[year].betting_df)
    combined_betting_df = pl.concat(all_betting) if all_betting else None

    # Calculate overall metrics
    metrics = calculate_metrics(predictions_df, ats_df)

    return {
        "predictions": predictions_df,
        "ats_results": ats_df,
        "metrics": metrics,
        "all_predictions": all_predictions,  # For dual-cap mode
        "betting_df": combined_betting_df,  # For dual-cap mode
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
        "--export-ats",
        type=str,
        default=None,
        help="Export ATS results CSV (includes Vegas spreads) for calibration module",
    )
    parser.add_argument(
        "--no-priors",
        action="store_true",
        help="Disable preseason priors",
    )
    parser.add_argument(
        "--blended-priors",
        action="store_true",
        help="Use blended SP+/own-prior system instead of SP+-only",
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
    # INT/Fumble separate shrinkage parameters
    parser.add_argument(
        "--k-int",
        type=float,
        default=10.0,
        help="Bayesian shrinkage k for interceptions (skill-based, moderate shrinkage). Default: 10.0",
    )
    parser.add_argument(
        "--k-fumble",
        type=float,
        default=30.0,
        help="Bayesian shrinkage k for fumbles (luck-based, strong shrinkage). Default: 30.0",
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
        help="Points for elite FCS teams (default: 18.0, used if --fcs-static)",
    )
    parser.add_argument(
        "--fcs-penalty-standard",
        type=float,
        default=32.0,
        help="Points for standard FCS teams (default: 32.0, used if --fcs-static)",
    )
    # FCS Dynamic Estimator parameters
    parser.add_argument(
        "--fcs-static",
        action="store_true",
        help="Use static elite FCS list instead of dynamic estimator (baseline comparison)",
    )
    parser.add_argument(
        "--fcs-k",
        type=float,
        default=8.0,
        help="FCS shrinkage k (games for 50%% trust in data). Default: 8.0",
    )
    parser.add_argument(
        "--fcs-baseline",
        type=float,
        default=-28.0,
        help="Prior margin for unknown FCS teams (FCS - FBS, negative = FCS loses). Default: -28.0",
    )
    parser.add_argument(
        "--fcs-min-pen",
        type=float,
        default=10.0,
        help="Minimum FCS penalty (for elite FCS). Default: 10.0",
    )
    parser.add_argument(
        "--fcs-max-pen",
        type=float,
        default=45.0,
        help="Maximum FCS penalty (for weak FCS). Default: 45.0",
    )
    parser.add_argument(
        "--fcs-slope",
        type=float,
        default=0.8,
        help="FCS penalty increase per point of avg loss. Default: 0.8",
    )
    parser.add_argument(
        "--fcs-intercept",
        type=float,
        default=10.0,
        help="Base FCS penalty (elite FCS with 0 avg loss). Default: 10.0",
    )
    parser.add_argument(
        "--fcs-hfa",
        type=float,
        default=0.0,
        help="HFA value to neutralize in FCS margin calculations. 0=disabled. Default: 0.0",
    )
    parser.add_argument(
        "--fcs-legacy-base",
        action="store_true",
        help="Use legacy FCS base_margin=0 (treats all FBS teams identically vs FCS). Default: False (use FBS deviation).",
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
    # ST Shrinkage parameters (REJECTED: disabled by default, infrastructure preserved)
    parser.add_argument(
        "--st-shrink",
        action="store_true",
        help="Enable ST Empirical Bayes shrinkage (REJECTED: degrades ATS, disabled by default)",
    )
    parser.add_argument(
        "--st-k-fg",
        type=float,
        default=6.0,
        help="ST shrinkage k for FG (attempts needed to trust rating fully). Default: 6.0",
    )
    parser.add_argument(
        "--st-k-punt",
        type=float,
        default=6.0,
        help="ST shrinkage k for punt (punts needed to trust rating fully). Default: 6.0",
    )
    parser.add_argument(
        "--st-k-ko",
        type=float,
        default=6.0,
        help="ST shrinkage k for kickoff (events needed to trust rating fully). Default: 6.0",
    )
    parser.add_argument(
        "--st-spread-cap",
        type=float,
        default=2.5,
        help="Cap ST differential's effect on spread. Default: 2.5 (APPROVED). Use --st-spread-cap 0 to disable.",
    )
    parser.add_argument(
        "--st-early-weight",
        type=float,
        default=1.0,
        help="Weight for ST impact in weeks 1-3 (Approach C). 1.0 = full, 0.5 = half. Default: 1.0",
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
    # Learned Situational Adjustment (LSA) parameters
    parser.add_argument(
        "--learned-situ",
        action="store_true",
        help="Use learned situational via ridge regression (experimental, default OFF = fixed baseline)",
    )
    parser.add_argument(
        "--lsa-alpha",
        type=float,
        default=300.0,
        help="LSA ridge alpha for regularization. Default: 300.0",
    )
    parser.add_argument(
        "--lsa-min-games",
        type=int,
        default=150,
        help="LSA minimum training games before switching from fixed. Default: 150",
    )
    parser.add_argument(
        "--lsa-ema",
        type=float,
        default=0.3,
        help="LSA EMA smoothing beta (0.3 = 30%% new, 70%% prior). Default: 0.3",
    )
    parser.add_argument(
        "--lsa-clamp-max",
        type=float,
        default=4.0,
        help="LSA coefficient clamp magnitude (e.g., 3.0 clamps to [-3, +3]). Default: 4.0",
    )
    # LSA training signal quality parameters
    parser.add_argument(
        "--lsa-adjust-turnovers",
        action="store_true",
        help="Adjust LSA training residuals to remove turnover-driven noise",
    )
    parser.add_argument(
        "--lsa-turnover-value",
        type=float,
        default=4.0,
        help="Points per turnover for target adjustment. Default: 4.0",
    )
    parser.add_argument(
        "--lsa-filter-vegas",
        type=float,
        default=None,
        help="Filter LSA training games with |vegas_spread| > threshold (e.g., 21.0). None disables.",
    )
    parser.add_argument(
        "--lsa-weighted-training",
        action="store_true",
        help="Use smooth Cauchy sample weighting instead of hard Vegas filter",
    )
    parser.add_argument(
        "--lsa-weight-spread",
        type=float,
        default=17.0,
        help="Spread threshold for Cauchy weight decay. Default: 17.0",
    )
    parser.add_argument(
        "--lsa-min-training-games",
        type=int,
        default=30,
        help="Minimum games after filtering; relax filter if below. Default: 30",
    )

    # Dual-Cap Mode: per-timing ST cap and HFA offset
    parser.add_argument(
        "--dual-cap-mode",
        action="store_true",
        help="Enable per-timing ST cap and HFA offset (evaluates both open and close lines)",
    )
    parser.add_argument(
        "--st-cap-open",
        type=float,
        default=2.5,
        help="ST spread cap for open line evaluation. Default: 2.5",
    )
    parser.add_argument(
        "--st-cap-close",
        type=float,
        default=2.5,
        help="ST spread cap for close line evaluation. Default: 2.5",
    )
    parser.add_argument(
        "--hfa-offset-open",
        type=float,
        default=0.50,
        help="HFA offset for open line evaluation. Default: 0.50",
    )
    parser.add_argument(
        "--hfa-offset-close",
        type=float,
        default=0.50,
        help="HFA offset for close line evaluation. Default: 0.50",
    )
    # QB Continuous Rating System (DEFAULT: enabled with phase1-only)
    parser.add_argument(
        "--qb-continuous",
        action="store_true",
        default=True,
        help="Enable continuous QB rating system. Default: ENABLED",
    )
    parser.add_argument(
        "--no-qb-continuous",
        action="store_true",
        help="Disable QB continuous rating system",
    )
    parser.add_argument(
        "--qb-shrinkage-k",
        type=float,
        default=200.0,
        help="QB shrinkage parameter (higher = more shrinkage). Default: 200",
    )
    parser.add_argument(
        "--qb-cap",
        type=float,
        default=3.0,
        help="QB point adjustment cap. Default: 3.0",
    )
    parser.add_argument(
        "--qb-scale",
        type=float,
        default=5.0,
        help="QB PPA-to-points scaling factor. Default: 5.0",
    )
    parser.add_argument(
        "--qb-prior-decay",
        type=float,
        default=0.3,
        help="Prior season decay factor (0-1). Default: 0.3",
    )
    parser.add_argument(
        "--no-qb-prior-season",
        action="store_true",
        help="Disable prior season data for QB Week 1 projections",
    )
    parser.add_argument(
        "--qb-phase1-only",
        action="store_true",
        default=True,
        help="Only apply QB adjustment for weeks 1-3 (skip Core weeks 4+). Default: ENABLED",
    )
    parser.add_argument(
        "--no-qb-phase1-only",
        action="store_true",
        help="Apply QB adjustment for ALL weeks (not just Phase 1). Warning: may degrade Core ATS.",
    )
    parser.add_argument(
        "--qb-fix-misattribution",
        action="store_true",
        help="Zero out Week 1 QB adjustment for unverified starters. Prevents misattribution of departed QB data.",
    )
    # Credible Rebuild Adjustment (reduces extra regression for low-RP teams with quality priors)
    parser.add_argument(
        "--no-credible-rebuild",
        action="store_true",
        help="Disable credible rebuild adjustment (reduced regression for low-RP teams with high talent/portal)",
    )
    # Dual-Cap Sweep Mode
    parser.add_argument(
        "--sweep-dual-st-cap",
        action="store_true",
        help="Run LOO-CV sweep to find optimal per-timing ST caps",
    )
    parser.add_argument(
        "--st-cap-grid",
        type=str,
        default="1.5,2.0,2.5,3.0,3.5,none",
        help="Comma-separated ST cap values to sweep. 'none' = no cap. Default: 1.5,2.0,2.5,3.0,3.5,none",
    )
    parser.add_argument(
        "--sweep-dual-hfa",
        action="store_true",
        help="Run LOO-CV sweep to find optimal per-timing HFA offsets",
    )
    parser.add_argument(
        "--hfa-grid",
        type=str,
        default="0.25,0.50,0.75,1.00",
        help="Comma-separated HFA offset values to sweep. Default: 0.25,0.50,0.75,1.00",
    )
    # Phase 1 Purity Test: Force 100% prior weight for early weeks
    parser.add_argument(
        "--lock-prior-weeks",
        type=int,
        default=0,
        help="Force 100%% prior weight (0%% in-season) for weeks <= this value. Default: 0 (disabled). "
             "Use 3 for Phase 1 purity test.",
    )
    # Phase 1 Spread Shrinkage (APPROVED 2026-02-12)
    parser.add_argument(
        "--phase1-shrinkage",
        type=float,
        default=0.90,
        help="Shrinkage factor for Phase 1 (weeks 1-3) spread predictions. "
             "Formula: NewSpread = (OldSpread - HFA) * Shrinkage + HFA. "
             "Default: 0.90 (APPROVED: +0.6%% 5+ Edge). Use 1.0 to disable.",
    )
    parser.add_argument(
        "--no-phase1-shrinkage",
        action="store_true",
        help="Disable Phase 1 spread shrinkage (use 1.0 factor)",
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

    # Parse ST cap grid for sweep mode
    def parse_cap_grid(grid_str: str) -> list:
        """Parse comma-separated cap values. 'none' = None (no cap)."""
        caps = []
        for val in grid_str.split(","):
            val = val.strip().lower()
            if val == "none":
                caps.append(None)
            else:
                caps.append(float(val))
        return caps

    # Handle dual-cap sweep mode
    if args.sweep_dual_st_cap:
        print("\n" + "=" * 80)
        print("DUAL-CAP LOO-CV SWEEP MODE")
        print("=" * 80)
        print(f"  Years: {args.years}")
        print(f"  ST cap grid: {args.st_cap_grid}")
        print(f"  HFA offset: {args.hfa_offset}")
        print(f"  Week range: {args.start_week} - {args.end_week if args.end_week else 15}")
        print("=" * 80 + "\n")

        st_cap_grid = parse_cap_grid(args.st_cap_grid)

        # First, run full backtest to get predictions with raw components
        print("Running full backtest to collect predictions...")
        season_data = fetch_all_season_data(
            args.years,
            use_priors=not args.no_priors,
            use_portal=not args.no_portal,
            portal_scale=args.portal_scale,
            use_cache=not args.no_cache,
            force_refresh=args.force_refresh,
            use_blended_priors=args.blended_priors,
            credible_rebuild_enabled=not args.no_credible_rebuild,
        )

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
            k_int=args.k_int,
            k_fumble=args.k_fumble,
            asymmetric_garbage=not args.no_asymmetric_garbage,
            fcs_penalty_elite=args.fcs_penalty_elite,
            fcs_penalty_standard=args.fcs_penalty_standard,
            use_portal=not args.no_portal,
            portal_scale=args.portal_scale,
            use_opening_line=args.opening_line,
            season_data=season_data,
            hfa_global_offset=args.hfa_offset,
            ooc_credibility_weight=args.ooc_cred_weight,
            st_shrink_enabled=args.st_shrink,
            st_k_fg=args.st_k_fg,
            st_k_punt=args.st_k_punt,
            st_k_ko=args.st_k_ko,
            st_spread_cap=args.st_spread_cap,
            st_early_weight=args.st_early_weight,
            fcs_static=args.fcs_static,
            fcs_k=args.fcs_k,
            fcs_baseline=args.fcs_baseline,
            fcs_min_pen=args.fcs_min_pen,
            fcs_max_pen=args.fcs_max_pen,
            fcs_slope=args.fcs_slope,
            fcs_intercept=args.fcs_intercept,
            fcs_hfa=args.fcs_hfa,
            fcs_use_deviation=not args.fcs_legacy_base,
            use_learned_situ=args.learned_situ,
            lsa_alpha=args.lsa_alpha,
            lsa_min_games=args.lsa_min_games,
            lsa_ema=args.lsa_ema,
            lsa_clamp_max=args.lsa_clamp_max,
            lsa_adjust_turnovers=args.lsa_adjust_turnovers,
            lsa_turnover_value=args.lsa_turnover_value,
            lsa_filter_vegas=args.lsa_filter_vegas,
            lsa_weighted_training=args.lsa_weighted_training,
            lsa_weight_spread=args.lsa_weight_spread,
            lsa_min_training_games=args.lsa_min_training_games,
        )

        # Run sweep
        all_predictions = results["all_predictions"]
        betting_df = results["betting_df"]

        sweep_df, stability, best_params = run_dual_cap_sweep(
            all_predictions=all_predictions,
            betting_df=betting_df,
            years=args.years,
            st_cap_grid=st_cap_grid,
            hfa_offset=args.hfa_offset,
            start_week=args.start_week,
            end_week=args.end_week if args.end_week else 15,
        )

        print_dual_cap_sweep_report(sweep_df, stability, best_params)
        return

    # Fetch data first for sanity reporting (P3.4)
    season_data = fetch_all_season_data(
        args.years,
        use_priors=not args.no_priors,
        use_portal=not args.no_portal,
        portal_scale=args.portal_scale,
        use_cache=not args.no_cache,
        force_refresh=args.force_refresh,
        use_blended_priors=args.blended_priors,
        credible_rebuild_enabled=not args.no_credible_rebuild,
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
    prior_status = 'disabled' if args.no_priors else ('blended SP+/own' if args.blended_priors else 'SP+-only')
    print(f"  Preseason priors:   {prior_status}")
    print(f"  Transfer portal:    {'disabled' if args.no_portal else f'enabled (scale={args.portal_scale})'}")
    print(f"  HFA:                team-specific (fallback={args.hfa}, global_offset={args.hfa_offset})")
    print(f"  EFM weights:        SR={args.efficiency_weight}, IsoPPP={args.explosiveness_weight}, TO={args.turnover_weight}")
    print(f"  TO shrinkage:       k_int={args.k_int}, k_fumble={args.k_fumble} (INT=skill, fumble=luck)")
    print(f"  Asymmetric GT:      {not args.no_asymmetric_garbage}")
    if args.fcs_static:
        print(f"  FCS penalties:      static (elite={args.fcs_penalty_elite}, standard={args.fcs_penalty_standard})")
    else:
        print(f"  FCS penalties:      dynamic (k={args.fcs_k}, baseline={args.fcs_baseline}, range=[{args.fcs_min_pen}, {args.fcs_max_pen}])")
    st_shrink_status = f"enabled (k_fg={args.st_k_fg}, k_punt={args.st_k_punt}, k_ko={args.st_k_ko})" if args.st_shrink else "disabled (default)"
    print(f"  ST shrinkage:       {st_shrink_status}")
    st_cap_status = f"±{args.st_spread_cap} pts" if args.st_spread_cap and args.st_spread_cap > 0 else "disabled"
    print(f"  ST spread cap:      {st_cap_status}")
    lsa_status = f"enabled (alpha={args.lsa_alpha}, min_games={args.lsa_min_games}, ema={args.lsa_ema})" if args.learned_situ else "disabled (fixed baseline)"
    print(f"  Learned situational: {lsa_status}")
    qb_status = f"enabled (k={args.qb_shrinkage_k}, cap=±{args.qb_cap}, scale={args.qb_scale})" if args.qb_continuous else "disabled"
    print(f"  QB continuous:      {qb_status}")
    print("=" * 60 + "\n")

    results = run_backtest(
        years=args.years,
        start_week=args.start_week,
        end_week=args.end_week,
        ridge_alpha=args.alpha,
        use_priors=not args.no_priors,
        use_blended_priors=args.blended_priors,
        hfa_value=args.hfa,
        prior_weight=args.prior_weight,
        efficiency_weight=args.efficiency_weight,
        explosiveness_weight=args.explosiveness_weight,
        turnover_weight=args.turnover_weight,
        k_int=args.k_int,
        k_fumble=args.k_fumble,
        asymmetric_garbage=not args.no_asymmetric_garbage,
        fcs_penalty_elite=args.fcs_penalty_elite,
        fcs_penalty_standard=args.fcs_penalty_standard,
        use_portal=not args.no_portal,
        portal_scale=args.portal_scale,
        use_opening_line=args.opening_line,
        season_data=season_data,  # Use pre-fetched data
        hfa_global_offset=args.hfa_offset,
        ooc_credibility_weight=args.ooc_cred_weight,
        st_shrink_enabled=args.st_shrink,
        st_k_fg=args.st_k_fg,
        st_k_punt=args.st_k_punt,
        st_k_ko=args.st_k_ko,
        st_spread_cap=args.st_spread_cap,
        st_early_weight=args.st_early_weight,
        # FCS dynamic estimator params
        fcs_static=args.fcs_static,
        fcs_k=args.fcs_k,
        fcs_baseline=args.fcs_baseline,
        fcs_min_pen=args.fcs_min_pen,
        fcs_max_pen=args.fcs_max_pen,
        fcs_slope=args.fcs_slope,
        fcs_intercept=args.fcs_intercept,
        fcs_hfa=args.fcs_hfa,
        fcs_use_deviation=not args.fcs_legacy_base,  # Default True unless --fcs-legacy-base
        # Learned Situational Adjustment (LSA) params
        use_learned_situ=args.learned_situ,
        lsa_alpha=args.lsa_alpha,
        lsa_min_games=args.lsa_min_games,
        lsa_ema=args.lsa_ema,
        lsa_clamp_max=args.lsa_clamp_max,
        # LSA training signal quality params
        lsa_adjust_turnovers=args.lsa_adjust_turnovers,
        lsa_turnover_value=args.lsa_turnover_value,
        lsa_filter_vegas=args.lsa_filter_vegas,
        lsa_weighted_training=args.lsa_weighted_training,
        lsa_weight_spread=args.lsa_weight_spread,
        lsa_min_training_games=args.lsa_min_training_games,
        # QB Continuous Rating params
        use_qb_continuous=args.qb_continuous and not args.no_qb_continuous,
        qb_shrinkage_k=args.qb_shrinkage_k,
        qb_cap=args.qb_cap,
        qb_scale=args.qb_scale,
        qb_prior_decay=args.qb_prior_decay,
        qb_use_prior_season=not args.no_qb_prior_season,
        qb_phase1_only=args.qb_phase1_only and not args.no_qb_phase1_only,
        qb_fix_misattribution=args.qb_fix_misattribution,
        # Credible Rebuild adjustment
        credible_rebuild_enabled=not args.no_credible_rebuild,
        # Phase 1 Purity Test
        lock_prior_weeks=args.lock_prior_weeks,
        # Phase 1 Spread Shrinkage
        phase1_shrinkage=1.0 if args.no_phase1_shrinkage else args.phase1_shrinkage,
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

    # Dual-cap mode: evaluate with per-timing parameters
    if args.dual_cap_mode:
        all_predictions = results.get("all_predictions", [])
        betting_df = results.get("betting_df")

        if all_predictions and betting_df is not None:
            dual_df = calculate_dual_ats_results(
                predictions=all_predictions,
                betting_df=betting_df,
                st_cap_open=args.st_cap_open,
                st_cap_close=args.st_cap_close,
                hfa_offset_open=args.hfa_offset_open,
                hfa_offset_close=args.hfa_offset_close,
            )

            if len(dual_df) > 0:
                print_dual_cap_report(
                    dual_df=dual_df,
                    st_cap_open=args.st_cap_open,
                    st_cap_close=args.st_cap_close,
                    hfa_offset_open=args.hfa_offset_open,
                    hfa_offset_close=args.hfa_offset_close,
                    start_week=args.start_week,
                    end_week=args.end_week if args.end_week else 15,
                )
        else:
            logger.warning("Dual-cap mode: missing all_predictions or betting_df in results")

    # Save to CSV if requested
    if args.output:
        results["predictions"].to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")

    # Export ATS results CSV for calibration module
    if args.export_ats:
        ats_df = results.get("ats_results")
        if ats_df is not None and len(ats_df) > 0:
            # Ensure all required columns are present
            export_cols = [
                "game_id", "year", "week", "home_team", "away_team",
                "predicted_spread", "actual_margin", "spread_open", "spread_close",
            ]
            # Add optional columns if present
            optional_cols = ["vegas_spread", "edge", "pick", "ats_win", "ats_push", "clv"]
            for col in optional_cols:
                if col in ats_df.columns:
                    export_cols.append(col)

            # Filter to available columns only
            available_cols = [c for c in export_cols if c in ats_df.columns]
            ats_df[available_cols].to_csv(args.export_ats, index=False)
            logger.info(f"ATS results saved to {args.export_ats} ({len(ats_df)} games)")
        else:
            logger.warning("No ATS results to export")


if __name__ == "__main__":
    main()
