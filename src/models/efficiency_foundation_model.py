"""Efficiency Foundation Model (EFM) - built on Success Rate, not margins.

This model builds power ratings from play-by-play efficiency metrics:
1. Foundation = Efficiency (Success Rate) + Explosiveness (IsoPPP)
2. Opponent adjustment via Ridge regression on Success Rate
3. DO NOT regress on margins - regress on efficiency metrics

Key insight: "Do not regress on the final score. Regress on the Success Rate
per Game so that we are measuring efficiency, not just the scoreboard outcome."
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config.settings import get_settings
from config.play_types import (
    TURNOVER_PLAY_TYPES,
    POINTS_PER_TURNOVER,
    SCRIMMAGE_PLAY_TYPES,
    NON_SCRIMMAGE_PLAY_TYPES,
)

logger = logging.getLogger(__name__)


def is_garbage_time(quarter: int, score_diff: int) -> bool:
    """Check if play is in garbage time.

    Uses thresholds from Settings (single source of truth).
    A play is garbage time if the score differential exceeds the
    threshold for that quarter.

    Args:
        quarter: Game quarter (1-4)
        score_diff: Absolute score differential

    Returns:
        True if garbage time

    Thresholds (from config/settings.py):
        Q1: 28 pts, Q2: 24 pts, Q3: 21 pts, Q4: 16 pts
    """
    settings = get_settings()
    thresholds = {
        1: settings.garbage_time_q1,
        2: settings.garbage_time_q2,
        3: settings.garbage_time_q3,
        4: settings.garbage_time_q4,
    }
    threshold = thresholds.get(quarter, settings.garbage_time_q4)
    return score_diff > threshold


# Success rate thresholds
def is_successful_play(down: int, distance: float, yards_gained: float) -> bool:
    """Determine if play was successful.

    Standard success rate definition:
    - 1st down: Gain at least 50% of yards needed
    - 2nd down: Gain at least 70% of yards needed
    - 3rd/4th down: Gain 100% of yards needed (first down or TD)

    Edge case handling:
    - distance <= 0: Treat as goal-to-go or data error. Require positive yards
      to be successful (any gain from the goal line is good).

    Args:
        down: Current down (1-4)
        distance: Yards to go (may be 0 at goal line or due to data issues)
        yards_gained: Yards gained on play

    Returns:
        True if successful
    """
    # Handle distance=0 edge case (goal line or data error)
    # Require positive yards to avoid auto-success on 0-gain plays
    if distance <= 0:
        return yards_gained > 0

    if down == 1:
        return yards_gained >= 0.5 * distance  # 50% of yards needed
    elif down == 2:
        return yards_gained >= 0.7 * distance  # 70% of yards needed
    else:  # 3rd or 4th down
        return yards_gained >= distance  # 100% of yards needed


@dataclass
class TeamEFMRating:
    """Container for Efficiency Foundation Model team ratings."""
    team: str

    # Raw metrics (before opponent adjustment)
    raw_success_rate: float
    raw_isoppp: float  # EPA on successful plays

    # Opponent-adjusted metrics
    adj_success_rate: float
    adj_isoppp: float

    # Component ratings (in points)
    efficiency_rating: float  # From success rate (combined O+D)
    explosiveness_rating: float  # From IsoPPP (combined O+D)

    # Separate O/D/ST ratings (in points, relative to average)
    offensive_rating: float  # Higher = better offense
    defensive_rating: float  # Higher = better defense (fewer points allowed)
    special_teams_rating: float  # FG efficiency - DIAGNOSTIC ONLY, not in overall (P2.7)
    turnover_rating: float  # Turnover margin contribution (higher = more takeaways)

    # Combined
    overall_rating: float

    # Sample sizes
    off_plays: int
    def_plays: int


class EfficiencyFoundationModel:
    """Efficiency Foundation Model: regress on Success Rate, not margins.

    The key insight: build ratings from play-by-play efficiency,
    then convert to point differential for predictions.

    Rating Components:
    - overall_rating = offensive_rating + defensive_rating
    - offensive_rating = efficiency + explosiveness + ball_security (turnovers)
    - defensive_rating = efficiency + explosiveness + takeaways (turnovers)

    Special Teams Integration (P2.7):
    - special_teams_rating is stored for DIAGNOSTIC/REPORTING purposes only
    - It is NOT included in overall_rating to avoid double-counting
    - SpreadGenerator applies ST as a separate adjustment layer using
      SpecialTeamsModel.get_matchup_differential()
    - This follows SP+ methodology: ST is a game-level adjustment, not a base rating
    - If you need ST in ratings, use set_special_teams_rating() for reporting,
      but the spread calculation handles ST separately
    """

    # Conversion factors (empirically derived)
    # These convert efficiency metrics to point equivalents
    SUCCESS_RATE_TO_POINTS = 80.0  # 1% SR difference ≈ 0.8 points
    ISOPPP_TO_POINTS = 15.0  # 0.1 IsoPPP difference ≈ 1.5 points

    # League averages (approximate FBS averages)
    LEAGUE_AVG_SUCCESS_RATE = 0.42
    LEAGUE_AVG_ISOPPP = 0.30  # ~0.3 EPA per successful play

    # Minimum plays for reliable rating
    MIN_PLAYS = 100

    # TURNOVER_PLAY_TYPES and POINTS_PER_TURNOVER imported from config.play_types

    def __init__(
        self,
        ridge_alpha: float = 50.0,  # Optimized via sweep (was 100.0)
        efficiency_weight: float = 0.54,  # Reduced from 0.60 to make room for turnovers
        explosiveness_weight: float = 0.36,  # Reduced from 0.40 to make room for turnovers
        turnover_weight: float = 0.10,  # 10% weight for turnovers (like SP+)
        turnover_prior_strength: float = 10.0,  # Bayesian shrinkage for turnover margin
        garbage_time_weight: float = 0.1,  # Weight for garbage time plays (0 to discard)
        rating_std: float = 12.0,  # Target std for ratings (SP+ uses ~12)
        asymmetric_garbage: bool = True,  # Only penalize trailing team in garbage time
        time_decay: float = 1.0,  # Per-week decay factor (1.0 = no decay, 0.95 = 5% per week)
    ):
        """Initialize Efficiency Foundation Model.

        Args:
            ridge_alpha: Regularization for opponent adjustment
            efficiency_weight: Weight for success rate component (default 0.54)
            explosiveness_weight: Weight for IsoPPP component (default 0.36)
            turnover_weight: Weight for turnover margin component (default 0.10, like SP+)
            turnover_prior_strength: Bayesian shrinkage for turnover margin (default 10).
                                    Equivalent to 10 games of 0 margin prior data. Higher = more
                                    regression toward 0. Prevents overweighting small-sample TO luck.
            garbage_time_weight: Weight for garbage time plays (0.1 recommended, 0 to discard)
            rating_std: Target standard deviation for ratings. Set to 12.0 for SP+-like scale
                       where Team A - Team B = expected spread. Higher = more spread between teams.
            asymmetric_garbage: If True, only trailing team's garbage time plays are down-weighted.
                              Leading team keeps full weight (they earned the blowout through efficiency).
            time_decay: Per-week decay factor for play weights. 1.0 = no decay (all weeks equal).
                       0.95 = 5% decay per week (Week 1 plays get ~0.54 weight by Week 12).
                       Formula: weight *= decay ^ (max_week - play_week)
        """
        self.ridge_alpha = ridge_alpha
        self.efficiency_weight = efficiency_weight
        self.rating_std = rating_std
        self.explosiveness_weight = explosiveness_weight
        self.turnover_weight = turnover_weight
        self.turnover_prior_strength = turnover_prior_strength
        self.garbage_time_weight = garbage_time_weight
        self.asymmetric_garbage = asymmetric_garbage
        self.time_decay = time_decay

        self.team_ratings: dict[str, TeamEFMRating] = {}

        # Store opponent-adjusted values
        self.off_success_rate: dict[str, float] = {}
        self.def_success_rate: dict[str, float] = {}
        self.off_isoppp: dict[str, float] = {}
        self.def_isoppp: dict[str, float] = {}

        # Turnover stats (P2.6: split into O/D components)
        self.turnovers_lost: dict[str, float] = {}  # Per-game turnovers lost (ball security)
        self.turnovers_forced: dict[str, float] = {}  # Per-game turnovers forced (takeaways)
        self.turnover_margin: dict[str, float] = {}  # Per-game net margin (for backward compat)
        self.team_games_played: dict[str, int] = {}  # Games played per team (for TO shrinkage)

        # Learned implicit HFA from ridge regression (for validation/logging)
        self.learned_hfa_sr: Optional[float] = None  # Implicit HFA in success rate
        self.learned_hfa_isoppp: Optional[float] = None  # Implicit HFA in IsoPPP

    def _prepare_plays(self, plays_df: pd.DataFrame) -> pd.DataFrame:
        """Filter and prepare plays for analysis.

        Filters:
        1. Non-scrimmage plays (special teams, penalties, period markers)
        2. Plays with missing/invalid data

        Args:
            plays_df: Raw play-by-play DataFrame

        Returns:
            Filtered DataFrame with success and garbage time flags
        """
        initial_count = len(plays_df)
        df = plays_df.copy()

        # Filter to scrimmage plays only (P2.8 fix)
        # This excludes special teams, penalties, period markers, etc.
        if "play_type" in df.columns:
            scrimmage_mask = df["play_type"].isin(SCRIMMAGE_PLAY_TYPES)
            non_scrimmage_count = (~scrimmage_mask).sum()
            if non_scrimmage_count > 0:
                # Log what's being filtered
                filtered_types = df[~scrimmage_mask]["play_type"].value_counts()
                logger.debug(
                    f"Filtering {non_scrimmage_count} non-scrimmage plays: "
                    f"{dict(filtered_types.head(5))}"
                )
            df = df[scrimmage_mask]

        # Filter plays with invalid distance (P2.8 fix)
        if "distance" in df.columns:
            invalid_distance = df["distance"].isna() | (df["distance"] < 0)
            invalid_count = invalid_distance.sum()
            if invalid_count > 0:
                logger.debug(f"Filtering {invalid_count} plays with invalid distance")
            df = df[~invalid_distance]

        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.info(
                f"Prepared {len(df)} plays for EFM "
                f"(filtered {filtered_count} non-scrimmage/invalid plays)"
            )
        else:
            logger.info(f"Prepared {len(df)} plays for EFM")

        # Calculate success for each play
        df["is_success"] = df.apply(
            lambda r: is_successful_play(r["down"], r["distance"], r["yards_gained"]),
            axis=1
        )

        # Calculate garbage time flag
        df["score_diff"] = (df["offense_score"] - df["defense_score"]).abs()
        df["is_garbage_time"] = df.apply(
            lambda r: is_garbage_time(r.get("period", 1), r["score_diff"]),
            axis=1
        )

        # Apply garbage time weighting
        if self.garbage_time_weight == 0:
            # Discard garbage time plays entirely
            df = df[~df["is_garbage_time"]]
            df["weight"] = 1.0
        elif self.asymmetric_garbage:
            # Asymmetric: only penalize TRAILING team's garbage time plays
            # Leading team keeps full weight - they earned the blowout through efficiency
            def calc_asymmetric_weight(row):
                if not row["is_garbage_time"]:
                    return 1.0
                # In garbage time: is offense winning or losing?
                margin = row["offense_score"] - row["defense_score"]
                if margin > 0:
                    # Offense is winning big - keep full weight
                    return 1.0
                else:
                    # Offense is trailing - this is garbage time noise
                    return self.garbage_time_weight

            df["weight"] = df.apply(calc_asymmetric_weight, axis=1)
        else:
            # Symmetric: weight ALL garbage time plays at reduced value
            df["weight"] = df["is_garbage_time"].apply(
                lambda gt: self.garbage_time_weight if gt else 1.0
            )

        # Apply time decay if enabled (decay < 1.0)
        if self.time_decay < 1.0 and "week" in df.columns:
            max_week = df["week"].max()
            # Weight = decay ^ (max_week - play_week)
            # Recent plays (max_week) get weight 1.0, older plays get less
            df["time_weight"] = self.time_decay ** (max_week - df["week"])
            df["weight"] = df["weight"] * df["time_weight"]

        return df

    def _calculate_raw_metrics(
        self, plays_df: pd.DataFrame
    ) -> tuple[dict, dict, dict, dict]:
        """Calculate raw (unadjusted) success rate and IsoPPP for all teams.

        Both SR and IsoPPP use the same play weighting scheme (garbage time + time decay).
        This ensures consistency: if a play is down-weighted for SR, it's also
        down-weighted for IsoPPP.

        Args:
            plays_df: Prepared plays DataFrame (must have 'weight' column)

        Returns:
            Tuple of (off_sr, def_sr, off_isoppp, def_isoppp) dicts
        """
        all_teams = set(plays_df["offense"]) | set(plays_df["defense"])

        off_sr = {}
        def_sr = {}
        off_isoppp = {}
        def_isoppp = {}

        for team in all_teams:
            # Offensive plays
            off_plays = plays_df[plays_df["offense"] == team]
            if len(off_plays) >= self.MIN_PLAYS:
                # Weighted success rate
                weights = off_plays["weight"]
                successes = off_plays["is_success"]
                off_sr[team] = (successes * weights).sum() / weights.sum()

                # IsoPPP: EPA on successful plays only, WEIGHTED (P2.3 fix)
                # Use same play weights as SR for consistency
                successful = off_plays[off_plays["is_success"]]
                if len(successful) > 20 and "ppa" in successful.columns:
                    succ_weights = successful["weight"]
                    succ_ppa = successful["ppa"]
                    off_isoppp[team] = (succ_ppa * succ_weights).sum() / succ_weights.sum()
                else:
                    off_isoppp[team] = self.LEAGUE_AVG_ISOPPP
            else:
                off_sr[team] = self.LEAGUE_AVG_SUCCESS_RATE
                off_isoppp[team] = self.LEAGUE_AVG_ISOPPP

            # Defensive plays (what they allow)
            def_plays = plays_df[plays_df["defense"] == team]
            if len(def_plays) >= self.MIN_PLAYS:
                weights = def_plays["weight"]
                successes = def_plays["is_success"]
                def_sr[team] = (successes * weights).sum() / weights.sum()

                # IsoPPP: EPA on successful plays only, WEIGHTED (P2.3 fix)
                successful = def_plays[def_plays["is_success"]]
                if len(successful) > 20 and "ppa" in successful.columns:
                    succ_weights = successful["weight"]
                    succ_ppa = successful["ppa"]
                    def_isoppp[team] = (succ_ppa * succ_weights).sum() / succ_weights.sum()
                else:
                    def_isoppp[team] = self.LEAGUE_AVG_ISOPPP
            else:
                def_sr[team] = self.LEAGUE_AVG_SUCCESS_RATE
                def_isoppp[team] = self.LEAGUE_AVG_ISOPPP

        return off_sr, def_sr, off_isoppp, def_isoppp

    def _ridge_adjust_metric(
        self,
        plays_df: pd.DataFrame,
        metric_col: str,
    ) -> tuple[dict[str, float], dict[str, float], Optional[float]]:
        """Use ridge regression to opponent-adjust a metric.

        Per spec: Y = metric value, X = sparse Team/Opponent IDs

        NEUTRAL-FIELD REGRESSION: If home_team column is present, we add a home
        field indicator to the design matrix. This allows the model to separately
        learn:
        1. Team strength (neutral-field) - the team coefficients
        2. Implicit home field advantage - the home indicator coefficient

        Without this, team coefficients contain implicit HFA from the data
        (EPA/success rates are higher for home teams), causing double-counting
        when SpreadGenerator adds explicit HFA.

        Args:
            plays_df: Prepared plays with the metric column
            metric_col: Name of column to adjust (e.g., 'is_success')

        Returns:
            Tuple of (off_adjusted, def_adjusted, learned_hfa) dicts
            learned_hfa is the implicit HFA coefficient (None if no home_team data)
        """
        # Get all teams
        all_teams = sorted(set(plays_df["offense"]) | set(plays_df["defense"]))
        team_to_idx = {team: i for i, team in enumerate(all_teams)}
        n_teams = len(all_teams)
        n_plays = len(plays_df)

        if n_plays == 0:
            return {}, {}, None

        # Check if we have home team info for neutral-field regression
        has_home_info = "home_team" in plays_df.columns

        # Build design matrix
        # Columns: [off_team_0, ..., off_team_n, def_team_0, ..., def_team_n, (home_indicator)]
        n_cols = 2 * n_teams + (1 if has_home_info else 0)
        X = np.zeros((n_plays, n_cols))

        offenses = plays_df["offense"].values
        defenses = plays_df["defense"].values
        weights = plays_df["weight"].values if "weight" in plays_df.columns else np.ones(n_plays)

        if has_home_info:
            home_teams = plays_df["home_team"].values
        else:
            home_teams = [None] * n_plays

        for i, (off, def_, home) in enumerate(zip(offenses, defenses, home_teams)):
            off_idx = team_to_idx[off]
            def_idx = team_to_idx[def_]
            X[i, off_idx] = 1.0  # Offense contributes positively
            X[i, n_teams + def_idx] = -1.0  # Defense reduces (good D = lower success)

            # Home field indicator: +1 if offense is home, -1 if away, 0 if neutral
            if has_home_info and home is not None:
                if off == home:
                    X[i, -1] = 1.0  # Offense is home team
                elif def_ == home:
                    X[i, -1] = -1.0  # Defense is home team (offense is away)
                # else: neutral site or unknown, leave as 0

        # Target: the metric value (e.g., success = 1/0)
        if metric_col == "is_success":
            y = plays_df[metric_col].astype(float).values
        else:
            y = plays_df[metric_col].values

        # Fit weighted ridge regression
        model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        model.fit(X, y, sample_weight=weights)

        # Extract coefficients
        coefficients = model.coef_
        intercept = model.intercept_

        # Separate home coefficient from team coefficients
        learned_hfa = None
        if has_home_info:
            learned_hfa = coefficients[-1]
            team_coefficients = coefficients[:-1]
            logger.info(
                f"Ridge adjust {metric_col}: intercept={intercept:.4f}, "
                f"implicit_HFA={learned_hfa:.4f} (neutral-field regression)"
            )
        else:
            team_coefficients = coefficients
            logger.info(f"Ridge adjust {metric_col}: intercept={intercept:.4f} (no home info)")

        off_adjusted = {}
        def_adjusted = {}

        for team, idx in team_to_idx.items():
            # Offensive rating: intercept + off_coef (now neutral-field)
            off_adjusted[team] = intercept + team_coefficients[idx]
            # Defensive rating: intercept - def_coef (negative because good D has negative coef)
            def_adjusted[team] = intercept - team_coefficients[n_teams + idx]

        return off_adjusted, def_adjusted, learned_hfa

    def _calculate_turnover_stats(
        self,
        plays_df: pd.DataFrame,
        games_df: Optional[pd.DataFrame] = None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Calculate per-game turnover stats for each team (P2.6: split O/D).

        Returns separate stats for ball security (turnovers lost) and
        takeaways (turnovers forced) to enable O/D-specific turnover ratings.

        Args:
            plays_df: Play-by-play data with 'play_type', 'offense', 'defense' columns
            games_df: Games data for counting games played

        Returns:
            Tuple of (lost_per_game, forced_per_game, margin_per_game) dicts
            - lost_per_game: Turnovers lost per game (lower = better ball security)
            - forced_per_game: Turnovers forced per game (higher = better takeaways)
            - margin_per_game: Net margin (forced - lost) for backward compat
        """
        empty_result = ({}, {}, {})

        if "play_type" not in plays_df.columns:
            logger.warning("No play_type column for turnover calculation")
            return empty_result

        # Find turnover plays
        turnover_plays = plays_df[plays_df["play_type"].isin(TURNOVER_PLAY_TYPES)]

        if len(turnover_plays) == 0:
            logger.warning("No turnover plays found")
            return empty_result

        # Count turnovers lost (offense = team that lost the ball)
        turnovers_lost = turnover_plays.groupby("offense").size()

        # Count turnovers forced (defense = team that forced it)
        turnovers_forced = turnover_plays.groupby("defense").size()

        # Get all teams
        all_teams = set(turnovers_lost.index) | set(turnovers_forced.index)

        # Count games per team - MUST have reliable count for shrinkage calculation
        # P2.10: No arbitrary defaults; compute from data or fail loudly
        games_played = {}
        games_source = None

        if games_df is not None and len(games_df) > 0:
            # Primary: count from games_df (most reliable)
            games_source = "games_df"
            for team in all_teams:
                n_games = len(games_df[
                    (games_df["home_team"] == team) | (games_df["away_team"] == team)
                ])
                games_played[team] = max(n_games, 1)
        elif "game_id" in plays_df.columns:
            # Fallback: count unique game_ids from plays (reliable if game_id exists)
            games_source = "plays_df.game_id"
            for team in all_teams:
                team_plays = plays_df[
                    (plays_df["offense"] == team) | (plays_df["defense"] == team)
                ]
                n_games = team_plays["game_id"].nunique()
                if n_games == 0:
                    logger.warning(
                        f"Team {team} has turnover data but 0 games found via game_id; "
                        "this may indicate data issues"
                    )
                    n_games = 1  # Minimum to avoid division by zero
                games_played[team] = n_games
        else:
            # P2.10: Fail loudly - cannot compute reliable games count
            raise ValueError(
                "Cannot compute games played for turnover shrinkage: "
                "games_df not provided and plays_df has no 'game_id' column. "
                "Provide games_df or ensure plays_df contains game_id."
            )

        logger.info(f"Games played computed from {games_source} for {len(games_played)} teams")

        # Calculate per-game stats (P2.6: separate lost/forced for O/D split)
        lost_per_game = {}
        forced_per_game = {}
        margin_per_game = {}

        for team in all_teams:
            lost = turnovers_lost.get(team, 0)
            forced = turnovers_forced.get(team, 0)
            games = games_played[team]

            lost_per_game[team] = lost / games
            forced_per_game[team] = forced / games
            margin_per_game[team] = (forced - lost) / games

        # Store games played for Bayesian shrinkage in calculate_ratings
        self.team_games_played = games_played

        # Log summary stats
        avg_lost = np.mean(list(lost_per_game.values()))
        avg_forced = np.mean(list(forced_per_game.values()))
        logger.info(
            f"Calculated turnover stats for {len(all_teams)} teams: "
            f"avg lost={avg_lost:.2f}/game, avg forced={avg_forced:.2f}/game"
        )

        return lost_per_game, forced_per_game, margin_per_game

    def calculate_ratings(
        self,
        plays_df: pd.DataFrame,
        games_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, TeamEFMRating]:
        """Calculate efficiency-based ratings for all teams.

        Args:
            plays_df: Play-by-play data
            games_df: Optional games data for sample size info

        Returns:
            Dict mapping team name to TeamEFMRating
        """
        # Prepare plays
        prepared = self._prepare_plays(plays_df)
        logger.info(f"Prepared {len(prepared)} plays for EFM")

        # Calculate raw metrics
        raw_off_sr, raw_def_sr, raw_off_isoppp, raw_def_isoppp = \
            self._calculate_raw_metrics(prepared)

        # Opponent-adjust Success Rate via ridge regression
        # This is the key: regress on SUCCESS RATE, not margins
        # NEUTRAL-FIELD: If home_team is present, regression separates team skill from HFA
        logger.info("Ridge adjusting Success Rate...")
        adj_off_sr, adj_def_sr, self.learned_hfa_sr = self._ridge_adjust_metric(
            prepared, "is_success"
        )

        # Opponent-adjust IsoPPP (EPA on successful plays)
        # Only use successful plays for this
        successful_plays = prepared[prepared["is_success"]].copy()
        if len(successful_plays) > 1000 and "ppa" in successful_plays.columns:
            logger.info("Ridge adjusting IsoPPP...")
            adj_off_isoppp, adj_def_isoppp, self.learned_hfa_isoppp = self._ridge_adjust_metric(
                successful_plays, "ppa"
            )
        else:
            logger.warning("Insufficient successful plays for IsoPPP adjustment, using raw")
            adj_off_isoppp = raw_off_isoppp
            adj_def_isoppp = raw_def_isoppp
            self.learned_hfa_isoppp = None

        # Store adjusted values
        self.off_success_rate = adj_off_sr
        self.def_success_rate = adj_def_sr
        self.off_isoppp = adj_off_isoppp
        self.def_isoppp = adj_def_isoppp

        # Calculate turnover stats if turnover_weight > 0 (P2.6: split O/D)
        if self.turnover_weight > 0 and "play_type" in plays_df.columns:
            logger.info("Calculating turnover stats...")
            self.turnovers_lost, self.turnovers_forced, self.turnover_margin = \
                self._calculate_turnover_stats(plays_df, games_df)
        else:
            self.turnovers_lost = {}
            self.turnovers_forced = {}
            self.turnover_margin = {}

        # Build team ratings
        all_teams = set(adj_off_sr.keys()) | set(adj_def_sr.keys())

        # Calculate league averages from adjusted values
        avg_sr = np.mean(list(adj_off_sr.values()))
        avg_isoppp = np.mean([v for v in adj_off_isoppp.values() if v != self.LEAGUE_AVG_ISOPPP])
        if np.isnan(avg_isoppp):
            avg_isoppp = self.LEAGUE_AVG_ISOPPP

        # Calculate league average turnover rates for O/D split (P2.6)
        avg_lost = np.mean(list(self.turnovers_lost.values())) if self.turnovers_lost else 0.0
        avg_forced = np.mean(list(self.turnovers_forced.values())) if self.turnovers_forced else 0.0

        for team in all_teams:
            # Get adjusted metrics
            off_sr = adj_off_sr.get(team, avg_sr)
            def_sr = adj_def_sr.get(team, avg_sr)
            off_iso = adj_off_isoppp.get(team, avg_isoppp)
            def_iso = adj_def_isoppp.get(team, avg_isoppp)

            # Convert to point equivalents
            # Offensive efficiency: how much better than average
            off_eff_pts = (off_sr - avg_sr) * self.SUCCESS_RATE_TO_POINTS
            # Defensive efficiency: how much better than average (lower = better)
            def_eff_pts = (avg_sr - def_sr) * self.SUCCESS_RATE_TO_POINTS

            # Explosiveness
            off_exp_pts = (off_iso - avg_isoppp) * self.ISOPPP_TO_POINTS
            def_exp_pts = (avg_isoppp - def_iso) * self.ISOPPP_TO_POINTS

            # Combine efficiency and explosiveness
            efficiency_rating = off_eff_pts + def_eff_pts
            explosiveness_rating = off_exp_pts + def_exp_pts

            # P2.6: Split turnovers into offensive (ball security) and defensive (takeaways)
            # Apply Bayesian shrinkage to each component separately
            # Shrinkage: games / (games + prior_strength). E.g., 15-game team keeps 60% of raw value.
            # P2.10: No arbitrary defaults - must have reliable games count
            if team not in self.team_games_played:
                # Team has efficiency data but no turnover data - compute games from plays
                if "game_id" in prepared.columns:
                    team_plays = prepared[
                        (prepared["offense"] == team) | (prepared["defense"] == team)
                    ]
                    games = max(team_plays["game_id"].nunique(), 1)
                    logger.debug(
                        f"Team {team} not in turnover stats; computed {games} games from plays"
                    )
                else:
                    raise ValueError(
                        f"Cannot determine games played for {team}: not in turnover stats "
                        "and no game_id column in plays. Provide games_df to calculate_ratings."
                    )
            else:
                games = self.team_games_played[team]
            shrinkage = games / (games + self.turnover_prior_strength)

            # Offensive turnover: ball security (fewer lost = better)
            # Relative to average: (avg_lost - team_lost) * shrinkage * points_per_to
            # Positive when team loses fewer than average
            raw_lost = self.turnovers_lost.get(team, avg_lost)
            off_to_pts = (avg_lost - raw_lost) * shrinkage * POINTS_PER_TURNOVER

            # Defensive turnover: takeaways (more forced = better)
            # Relative to average: (team_forced - avg_forced) * shrinkage * points_per_to
            # Positive when team forces more than average
            raw_forced = self.turnovers_forced.get(team, avg_forced)
            def_to_pts = (raw_forced - avg_forced) * shrinkage * POINTS_PER_TURNOVER

            # Combined turnover rating for backward compat (should equal off_to + def_to)
            turnover_rating = off_to_pts + def_to_pts

            # Separate offensive and defensive ratings (P2.6: now includes turnovers)
            # offensive_rating = efficiency + explosiveness + ball_security
            # defensive_rating = efficiency + explosiveness + takeaways
            offensive_rating = (
                self.efficiency_weight * off_eff_pts +
                self.explosiveness_weight * off_exp_pts +
                self.turnover_weight * off_to_pts
            )
            defensive_rating = (
                self.efficiency_weight * def_eff_pts +
                self.explosiveness_weight * def_exp_pts +
                self.turnover_weight * def_to_pts
            )

            # Overall rating = O + D (turnovers now inside O/D, not separate)
            overall = offensive_rating + defensive_rating

            # Get sample sizes
            off_plays = len(prepared[prepared["offense"] == team])
            def_plays = len(prepared[prepared["defense"] == team])

            self.team_ratings[team] = TeamEFMRating(
                team=team,
                raw_success_rate=raw_off_sr.get(team, avg_sr),
                raw_isoppp=raw_off_isoppp.get(team, avg_isoppp),
                adj_success_rate=off_sr,
                adj_isoppp=off_iso,
                efficiency_rating=efficiency_rating,
                explosiveness_rating=explosiveness_rating,
                offensive_rating=offensive_rating,
                defensive_rating=defensive_rating,
                special_teams_rating=0.0,  # Set separately via set_special_teams_rating()
                turnover_rating=turnover_rating,
                overall_rating=overall,
                off_plays=off_plays,
                def_plays=def_plays,
            )

        logger.info(f"Calculated EFM ratings for {len(self.team_ratings)} teams")

        # Normalize ratings to target standard deviation
        # This ensures Team A rating - Team B rating = expected spread
        # Note: all_teams from CFBD API is FBS teams only
        self._normalize_ratings(all_teams)

        return self.team_ratings

    def _normalize_ratings(self, fbs_teams: set[str]) -> None:
        """Normalize ratings to target standard deviation.

        Centers each component (O/D/TO/efficiency/explosiveness) by its own mean,
        then scales all components uniformly. This ensures:
        - Overall rating has mean=0, std=rating_std (for spread calculation)
        - Each component is properly centered by its own mean
        - Relationship overall = off + def is preserved (P2.6: turnovers now inside O/D)
        - Components remain interpretable (mean=0 for each)

        Math verification (P2.6 update):
        - mean(overall) = mean(off) + mean(def) by linearity
        - After centering each component: new_overall = new_off + new_def
        - Turnover effects are embedded in O (ball security) and D (takeaways)
        - turnover_rating is kept as diagnostic (off_to + def_to)

        Args:
            fbs_teams: Set of FBS team names (normalization based on these)
        """
        if not self.team_ratings:
            return

        # Get current FBS ratings for normalization stats
        fbs_ratings = [r for team, r in self.team_ratings.items() if team in fbs_teams]

        if not fbs_ratings:
            return

        # Calculate component means from FBS teams (P2.5 fix: each component by its own mean)
        overall_values = [r.overall_rating for r in fbs_ratings]
        offense_values = [r.offensive_rating for r in fbs_ratings]
        defense_values = [r.defensive_rating for r in fbs_ratings]
        turnover_values = [r.turnover_rating for r in fbs_ratings]
        efficiency_values = [r.efficiency_rating for r in fbs_ratings]
        explosiveness_values = [r.explosiveness_rating for r in fbs_ratings]

        overall_mean = np.mean(overall_values)
        overall_std = np.std(overall_values)
        offense_mean = np.mean(offense_values)
        defense_mean = np.mean(defense_values)
        turnover_mean = np.mean(turnover_values)
        efficiency_mean = np.mean(efficiency_values)
        explosiveness_mean = np.mean(explosiveness_values)

        if overall_std == 0:
            return

        # Calculate scale factor from overall rating
        scale = self.rating_std / overall_std

        logger.info(
            f"Normalizing ratings: mean {overall_mean:.2f} → 0, "
            f"std {overall_std:.2f} → {self.rating_std:.1f} (scale={scale:.2f}x)"
        )
        logger.debug(
            f"Component means: off={offense_mean:.2f}, def={defense_mean:.2f}, "
            f"to={turnover_mean:.2f}, eff={efficiency_mean:.2f}, exp={explosiveness_mean:.2f}"
        )

        # Apply normalization to all teams
        # Each component is centered by its own mean, then scaled uniformly
        for team, rating in self.team_ratings.items():
            new_offense = (rating.offensive_rating - offense_mean) * scale
            new_defense = (rating.defensive_rating - defense_mean) * scale
            new_turnover = (rating.turnover_rating - turnover_mean) * scale
            new_efficiency = (rating.efficiency_rating - efficiency_mean) * scale
            new_explosiveness = (rating.explosiveness_rating - explosiveness_mean) * scale

            # Overall = off + def (P2.6: turnovers now inside O/D, not separate)
            new_overall = new_offense + new_defense

            # Update the rating object
            self.team_ratings[team] = TeamEFMRating(
                team=rating.team,
                raw_success_rate=rating.raw_success_rate,
                raw_isoppp=rating.raw_isoppp,
                adj_success_rate=rating.adj_success_rate,
                adj_isoppp=rating.adj_isoppp,
                efficiency_rating=new_efficiency,
                explosiveness_rating=new_explosiveness,
                offensive_rating=new_offense,
                defensive_rating=new_defense,
                special_teams_rating=rating.special_teams_rating,
                turnover_rating=new_turnover,
                overall_rating=new_overall,
                off_plays=rating.off_plays,
                def_plays=rating.def_plays,
            )

    def get_rating(self, team: str) -> float:
        """Get overall rating for a team.

        Args:
            team: Team name

        Returns:
            Overall rating (0.0 if unknown)
        """
        if team in self.team_ratings:
            return self.team_ratings[team].overall_rating
        return 0.0

    def get_offensive_rating(self, team: str) -> float:
        """Get offensive rating for a team.

        Args:
            team: Team name

        Returns:
            Offensive rating (0.0 if unknown). Higher = better offense.
        """
        if team in self.team_ratings:
            return self.team_ratings[team].offensive_rating
        return 0.0

    def get_defensive_rating(self, team: str) -> float:
        """Get defensive rating for a team.

        Args:
            team: Team name

        Returns:
            Defensive rating (0.0 if unknown). Higher = better defense.
        """
        if team in self.team_ratings:
            return self.team_ratings[team].defensive_rating
        return 0.0

    def get_special_teams_rating(self, team: str) -> float:
        """Get special teams rating for a team (DIAGNOSTIC ONLY).

        NOTE (P2.7): This rating is NOT included in overall_rating.
        Special teams is applied as a separate adjustment layer in SpreadGenerator
        using SpecialTeamsModel.get_matchup_differential(). This method exists
        for diagnostic/reporting purposes only.

        Args:
            team: Team name

        Returns:
            Special teams rating (0.0 if unknown). Currently FG efficiency only.
        """
        if team in self.team_ratings:
            return self.team_ratings[team].special_teams_rating
        return 0.0

    def set_special_teams_rating(self, team: str, rating: float) -> None:
        """Set special teams rating for a team (DIAGNOSTIC ONLY).

        NOTE (P2.7): This rating is stored for reporting but NOT included in
        overall_rating. SpreadGenerator handles ST as a separate adjustment layer
        using SpecialTeamsModel.get_matchup_differential() to avoid double-counting.

        Use this method to populate ST ratings for display in get_ratings_df(),
        but do NOT expect it to affect spread predictions (those come from
        SpecialTeamsModel directly in SpreadGenerator).

        Args:
            team: Team name
            rating: Special teams rating (per-game point value)
        """
        if team in self.team_ratings:
            # Update the rating object
            old_rating = self.team_ratings[team]
            self.team_ratings[team] = TeamEFMRating(
                team=old_rating.team,
                raw_success_rate=old_rating.raw_success_rate,
                raw_isoppp=old_rating.raw_isoppp,
                adj_success_rate=old_rating.adj_success_rate,
                adj_isoppp=old_rating.adj_isoppp,
                efficiency_rating=old_rating.efficiency_rating,
                explosiveness_rating=old_rating.explosiveness_rating,
                offensive_rating=old_rating.offensive_rating,
                defensive_rating=old_rating.defensive_rating,
                special_teams_rating=rating,
                turnover_rating=old_rating.turnover_rating,
                overall_rating=old_rating.overall_rating,
                off_plays=old_rating.off_plays,
                def_plays=old_rating.def_plays,
            )

    def get_ratings_df(self) -> pd.DataFrame:
        """Get ratings as DataFrame sorted by overall rating.

        Returns:
            DataFrame with all team ratings including O/D/ST breakdown
        """
        if not self.team_ratings:
            return pd.DataFrame()

        data = [
            {
                "team": r.team,
                "overall": round(r.overall_rating, 1),
                "offense": round(r.offensive_rating, 1),
                "defense": round(r.defensive_rating, 1),
                "special_teams": round(r.special_teams_rating, 2),
                "efficiency": round(r.efficiency_rating, 1),
                "explosiveness": round(r.explosiveness_rating, 1),
                "adj_sr": round(r.adj_success_rate, 3),
                "adj_isoppp": round(r.adj_isoppp, 3),
                "raw_sr": round(r.raw_success_rate, 3),
                "off_plays": r.off_plays,
            }
            for r in self.team_ratings.values()
        ]

        df = pd.DataFrame(data)
        return df.sort_values("overall", ascending=False).reset_index(drop=True)

    def predict_margin(
        self,
        home_team: str,
        away_team: str,
        neutral_site: bool = False,
        hfa: float = 2.5,
    ) -> float:
        """Predict point margin for a game.

        Args:
            home_team: Home team name
            away_team: Away team name
            neutral_site: Whether game is at neutral site
            hfa: Home field advantage in points

        Returns:
            Predicted margin (positive = home favored)
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        margin = home_rating - away_rating

        if not neutral_site:
            margin += hfa

        return margin
