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

    Args:
        down: Current down (1-4)
        distance: Yards to go
        yards_gained: Yards gained on play

    Returns:
        True if successful
    """
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
    special_teams_rating: float  # FG efficiency (PAAE per game)
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

    # Turnover play types (offense = team that lost the ball)
    # Verified against CFBD API play_type values (2024 data)
    TURNOVER_PLAY_TYPES = frozenset({
        "Fumble Recovery (Opponent)",
        "Pass Interception Return",
        "Interception",
        "Fumble Return Touchdown",
        "Interception Return Touchdown",
    })

    # Points per turnover (empirical value)
    POINTS_PER_TURNOVER = 4.5

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
        self.turnover_margin: dict[str, float] = {}  # Per-game turnover margin (raw, before shrinkage)
        self.team_games_played: dict[str, int] = {}  # Games played per team (for TO shrinkage)

        # Learned implicit HFA from ridge regression (for validation/logging)
        self.learned_hfa_sr: Optional[float] = None  # Implicit HFA in success rate
        self.learned_hfa_isoppp: Optional[float] = None  # Implicit HFA in IsoPPP

    def _prepare_plays(self, plays_df: pd.DataFrame) -> pd.DataFrame:
        """Filter and prepare plays for analysis.

        Args:
            plays_df: Raw play-by-play DataFrame

        Returns:
            Filtered DataFrame with success and garbage time flags
        """
        df = plays_df.copy()

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

    def _calculate_turnover_margin(
        self,
        plays_df: pd.DataFrame,
        games_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, float]:
        """Calculate per-game turnover margin for each team.

        Args:
            plays_df: Play-by-play data with 'play_type', 'offense', 'defense' columns
            games_df: Games data for counting games played

        Returns:
            Dict mapping team to per-game turnover margin (positive = more takeaways)
        """
        if "play_type" not in plays_df.columns:
            logger.warning("No play_type column for turnover calculation")
            return {}

        # Find turnover plays
        turnover_plays = plays_df[plays_df["play_type"].isin(self.TURNOVER_PLAY_TYPES)]

        if len(turnover_plays) == 0:
            logger.warning("No turnover plays found")
            return {}

        # Count turnovers lost (offense = team that lost the ball)
        turnovers_lost = turnover_plays.groupby("offense").size()

        # Count turnovers forced (defense = team that forced it)
        turnovers_forced = turnover_plays.groupby("defense").size()

        # Get all teams
        all_teams = set(turnovers_lost.index) | set(turnovers_forced.index)

        # Count games per team (approximate from play data if games_df not provided)
        if games_df is not None:
            games_played = {}
            for team in all_teams:
                n_games = len(games_df[
                    (games_df["home_team"] == team) | (games_df["away_team"] == team)
                ])
                games_played[team] = max(n_games, 1)
        else:
            # Approximate from unique game_ids in plays
            games_played = {}
            for team in all_teams:
                team_plays = plays_df[
                    (plays_df["offense"] == team) | (plays_df["defense"] == team)
                ]
                n_games = team_plays["game_id"].nunique() if "game_id" in team_plays.columns else 10
                games_played[team] = max(n_games, 1)

        # Calculate per-game turnover margin
        turnover_margins = {}
        for team in all_teams:
            lost = turnovers_lost.get(team, 0)
            forced = turnovers_forced.get(team, 0)
            margin = forced - lost
            per_game = margin / games_played[team]
            turnover_margins[team] = per_game

        # Store games played for Bayesian shrinkage in calculate_ratings
        self.team_games_played = games_played

        logger.info(f"Calculated turnover margins for {len(turnover_margins)} teams")

        return turnover_margins

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

        # Calculate turnover margins if turnover_weight > 0
        if self.turnover_weight > 0 and "play_type" in plays_df.columns:
            logger.info("Calculating turnover margins...")
            self.turnover_margin = self._calculate_turnover_margin(plays_df, games_df)
        else:
            self.turnover_margin = {}

        # Build team ratings
        all_teams = set(adj_off_sr.keys()) | set(adj_def_sr.keys())

        # Calculate league averages from adjusted values
        avg_sr = np.mean(list(adj_off_sr.values()))
        avg_isoppp = np.mean([v for v in adj_off_isoppp.values() if v != self.LEAGUE_AVG_ISOPPP])
        if np.isnan(avg_isoppp):
            avg_isoppp = self.LEAGUE_AVG_ISOPPP

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

            # Turnover rating: convert per-game margin to points with Bayesian shrinkage
            # Positive margin = more takeaways = good
            # Apply shrinkage toward 0 based on games played (turnover margin is ~50-70% luck)
            # With prior_strength=10, a 15-game team keeps 60% of their margin (15/(15+10))
            raw_to_margin = self.turnover_margin.get(team, 0.0)
            games = self.team_games_played.get(team, 10)
            shrinkage = games / (games + self.turnover_prior_strength)
            shrunk_to_margin = raw_to_margin * shrinkage
            turnover_rating = shrunk_to_margin * self.POINTS_PER_TURNOVER

            # Separate offensive and defensive ratings (weighted combo of eff + exp)
            # Note: turnover is a combined O+D metric, not split
            offensive_rating = (
                self.efficiency_weight * off_eff_pts +
                self.explosiveness_weight * off_exp_pts
            )
            defensive_rating = (
                self.efficiency_weight * def_eff_pts +
                self.explosiveness_weight * def_exp_pts
            )

            # Overall rating (weighted combination) = O + D + Turnovers
            # Weights: efficiency + explosiveness + turnover = 1.0
            overall = offensive_rating + defensive_rating + self.turnover_weight * turnover_rating

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

        Scales all ratings so that:
        - Mean of FBS teams = 0
        - Std of FBS teams = self.rating_std (default 12.0)

        This allows direct spread calculation: Team A - Team B = expected margin.

        Args:
            fbs_teams: Set of FBS team names (normalization based on these)
        """
        if not self.team_ratings:
            return

        # Get current FBS ratings
        fbs_overalls = [
            r.overall_rating for team, r in self.team_ratings.items()
            if team in fbs_teams
        ]

        if not fbs_overalls:
            return

        current_mean = np.mean(fbs_overalls)
        current_std = np.std(fbs_overalls)

        if current_std == 0:
            return

        # Calculate scale factor
        scale = self.rating_std / current_std

        logger.info(
            f"Normalizing ratings: mean {current_mean:.2f} → 0, "
            f"std {current_std:.2f} → {self.rating_std:.1f} (scale={scale:.2f}x)"
        )

        # Apply normalization to all teams
        for team, rating in self.team_ratings.items():
            # Scale and center
            # Note on O/D centering: We use current_mean/2 for both offense and defense.
            # This works because ridge regression with balanced data produces mean(off) ≈ mean(def) ≈ 0
            # pre-normalization, so current_mean ≈ 0.1*mean(turnover). The /2 split correctly
            # distributes any residual mean while maintaining overall = off + def + 0.1*TO.
            # Verified empirically: formula holds exactly post-normalization with zero error.
            new_overall = (rating.overall_rating - current_mean) * scale
            new_offense = (rating.offensive_rating - current_mean / 2) * scale
            new_defense = (rating.defensive_rating - current_mean / 2) * scale
            new_efficiency = rating.efficiency_rating * scale
            new_explosiveness = rating.explosiveness_rating * scale
            new_turnover = rating.turnover_rating * scale

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
        """Get special teams rating for a team.

        Args:
            team: Team name

        Returns:
            Special teams rating (0.0 if unknown). Currently FG efficiency only.
        """
        if team in self.team_ratings:
            return self.team_ratings[team].special_teams_rating
        return 0.0

    def set_special_teams_rating(self, team: str, rating: float) -> None:
        """Set special teams rating for a team.

        This is used to integrate FG efficiency from SpecialTeamsModel.

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
