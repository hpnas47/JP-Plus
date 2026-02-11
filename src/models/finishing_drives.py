"""Finishing drives model for red zone and scoring efficiency."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FinishingDrivesRating:
    """Container for team finishing drives ratings."""

    team: str
    red_zone_td_rate: float  # TD% in red zone
    red_zone_scoring_rate: float  # Any score % in red zone
    points_per_trip: float  # Average points per red zone trip
    goal_to_go_conversion: float  # TD% in goal-to-go situations
    overall_rating: float  # Combined efficiency score


class FinishingDrivesModel:
    """
    Model for evaluating red zone and scoring efficiency.

    Focuses on:
    - Red zone TD% (touchdowns vs field goals)
    - Points per red zone opportunity
    - Goal-to-go conversion rate
    - Scoring opportunity conversion
    """

    # Expected values (FBS averages)
    EXPECTED_RZ_TD_RATE = 0.58  # ~58% of RZ trips end in TD
    EXPECTED_RZ_SCORING_RATE = 0.85  # ~85% score something
    EXPECTED_POINTS_PER_TRIP = 4.05  # Fallback constant (used only if dynamic mean unavailable)
    EXPECTED_GOAL_TO_GO = 0.65  # 65% TD rate in goal-to-go

    # PBTA scaling: maximum matchup differential from FD component
    # Caps total swing to ±1.5 points to prevent overwhelming EFM signal
    MAX_MATCHUP_DIFFERENTIAL = 1.5  # points

    # Point values for conversion
    TD_POINTS = 7.0  # Including typical PAT
    FG_POINTS = 3.0

    # Prior strength for Bayesian regression (equivalent to ~10 RZ trips of prior data)
    # Reduced from 20 to 10 to trust actual data more at end of season
    # With 150+ RZ plays per team, raw data is reliable
    PRIOR_RZ_TRIPS = 10

    def __init__(self, regress_to_mean: bool = True, prior_strength: Optional[int] = None) -> None:
        """Initialize the finishing drives model.

        Args:
            regress_to_mean: Whether to apply Bayesian regression toward expected values
            prior_strength: Number of prior RZ trips to use for regression (default: 10)
        """
        self.team_ratings: dict[str, FinishingDrivesRating] = {}
        self.regress_to_mean = regress_to_mean
        self.prior_strength = prior_strength or self.PRIOR_RZ_TRIPS
        self._seasonal_mean_ppt: Optional[float] = None  # Dynamic baseline computed from training data

    def calculate_team_rating(
        self,
        team: str,
        rz_touchdowns: int,
        rz_field_goals: int,
        rz_turnovers: int,
        rz_failed: int,  # Downs/other
        goal_to_go_tds: Optional[int] = None,
        goal_to_go_attempts: Optional[int] = None,
        games_played: Optional[int] = None,
    ) -> FinishingDrivesRating:
        """Calculate finishing drives rating for a team.

        Args:
            team: Team name
            rz_touchdowns: Red zone touchdowns scored
            rz_field_goals: Red zone field goals made
            rz_turnovers: Red zone turnovers
            rz_failed: Other failed red zone trips (4th down stops, etc.)
            goal_to_go_tds: TDs in goal-to-go situations (optional)
            goal_to_go_attempts: Total goal-to-go attempts (optional)
            games_played: Number of games played (optional, used to normalize per-game)

        Returns:
            FinishingDrivesRating for the team
        """
        total_rz_trips = rz_touchdowns + rz_field_goals + rz_turnovers + rz_failed

        if total_rz_trips == 0:
            return FinishingDrivesRating(
                team=team,
                red_zone_td_rate=self.EXPECTED_RZ_TD_RATE,
                red_zone_scoring_rate=self.EXPECTED_RZ_SCORING_RATE,
                points_per_trip=self.EXPECTED_POINTS_PER_TRIP,
                goal_to_go_conversion=self.EXPECTED_GOAL_TO_GO,
                overall_rating=0.0,
            )

        # Calculate rates with optional Bayesian regression toward the mean
        if self.regress_to_mean:
            # Bayesian regression: (observed + prior_mean * prior_strength) / (n + prior_strength)
            prior = self.prior_strength

            # Regressed RZ TD rate
            prior_tds = self.EXPECTED_RZ_TD_RATE * prior
            rz_td_rate = (rz_touchdowns + prior_tds) / (total_rz_trips + prior)

            # Regressed RZ scoring rate
            prior_scores = self.EXPECTED_RZ_SCORING_RATE * prior
            rz_scoring_rate = (rz_touchdowns + rz_field_goals + prior_scores) / (total_rz_trips + prior)

            # Regressed points per trip
            total_points = (rz_touchdowns * self.TD_POINTS) + (rz_field_goals * self.FG_POINTS)
            prior_points = self.EXPECTED_POINTS_PER_TRIP * prior
            points_per_trip = (total_points + prior_points) / (total_rz_trips + prior)
        else:
            # Raw rates (no regression)
            rz_td_rate = rz_touchdowns / total_rz_trips
            rz_scoring_rate = (rz_touchdowns + rz_field_goals) / total_rz_trips
            total_points = (rz_touchdowns * self.TD_POINTS) + (rz_field_goals * self.FG_POINTS)
            points_per_trip = total_points / total_rz_trips

        # Goal-to-go conversion (also regressed if enabled)
        if goal_to_go_attempts and goal_to_go_attempts > 0:
            if self.regress_to_mean:
                prior_gtg = self.EXPECTED_GOAL_TO_GO * (self.prior_strength / 2)  # Fewer GTG situations
                goal_to_go_rate = (goal_to_go_tds + prior_gtg) / (goal_to_go_attempts + self.prior_strength / 2)
            else:
                goal_to_go_rate = goal_to_go_tds / goal_to_go_attempts
        else:
            # Estimate from RZ TD rate
            goal_to_go_rate = min(rz_td_rate + 0.05, 1.0)

        # Overall rating: raw difference from dynamic seasonal mean (PBTA units: points per trip)
        # Formula: points_per_trip - seasonal_mean_ppt
        # No multiplier needed — this directly measures efficiency deviation
        # Dynamic baseline prevents systematic bias from year-to-year PPT variation (3.25-3.78)

        # Use dynamic seasonal mean if available, otherwise fall back to constant
        baseline_ppt = self._seasonal_mean_ppt if self._seasonal_mean_ppt is not None else self.EXPECTED_POINTS_PER_TRIP
        overall = points_per_trip - baseline_ppt

        rating = FinishingDrivesRating(
            team=team,
            red_zone_td_rate=rz_td_rate,
            red_zone_scoring_rate=rz_scoring_rate,
            points_per_trip=points_per_trip,
            goal_to_go_conversion=goal_to_go_rate,
            overall_rating=overall,
        )

        self.team_ratings[team] = rating
        return rating

    def calculate_from_drives(
        self,
        team: str,
        drives_df: pd.DataFrame,
    ) -> FinishingDrivesRating:
        """Calculate finishing drives rating from drive-level data.

        P2.2: SECONDARY pathway — use when drive-level data is available but
        play-by-play is not. Preferred over calculate_from_game_stats().

        Hierarchy: calculate_all_from_plays() > calculate_from_drives() > calculate_from_game_stats()

        Args:
            team: Team name
            drives_df: DataFrame with drive data including:
                - offense: Team on offense
                - start_yards_to_goal: Starting field position
                - end_yards_to_goal: Ending field position
                - drive_result: Result (TD, FG, Punt, Turnover, etc.)

        Returns:
            FinishingDrivesRating for the team
        """
        logger.debug(f"FD pathway: drive-level for {team}")
        # Filter to team's offensive drives that entered red zone
        team_drives = drives_df[drives_df["offense"] == team]

        # Red zone = inside 20 yards
        rz_drives = team_drives[
            (team_drives["end_yards_to_goal"] <= 20)
            | (team_drives["start_yards_to_goal"] <= 20)
        ]

        if len(rz_drives) == 0:
            return self.calculate_team_rating(team, 0, 0, 0, 0)

        # Count outcomes
        result_col = "drive_result" if "drive_result" in rz_drives.columns else "result"

        rz_tds = len(rz_drives[rz_drives[result_col].str.contains("TD", case=False, na=False)])
        rz_fgs = len(rz_drives[rz_drives[result_col].str.contains("FG", case=False, na=False)])
        rz_turnovers = len(
            rz_drives[
                rz_drives[result_col].str.contains(
                    "INT|FUMBLE|TURNOVER", case=False, na=False
                )
            ]
        )
        rz_failed = len(rz_drives) - rz_tds - rz_fgs - rz_turnovers

        # Goal-to-go situations (inside 10)
        goal_to_go = team_drives[
            team_drives["start_yards_to_goal"] <= 10
        ]
        gtg_tds = len(
            goal_to_go[goal_to_go[result_col].str.contains("TD", case=False, na=False)]
        )
        gtg_attempts = len(goal_to_go)

        # Derive games_played from unique game_ids
        games_played = team_drives["game_id"].nunique() if "game_id" in team_drives.columns else None

        return self.calculate_team_rating(
            team=team,
            rz_touchdowns=rz_tds,
            rz_field_goals=rz_fgs,
            rz_turnovers=rz_turnovers,
            rz_failed=rz_failed,
            goal_to_go_tds=gtg_tds,
            goal_to_go_attempts=gtg_attempts,
            games_played=games_played,
        )

    def calculate_from_game_stats(
        self,
        team: str,
        games_df: pd.DataFrame,
    ) -> FinishingDrivesRating:
        """Calculate finishing drives rating from game-level statistics.

        P2.2: TERTIARY pathway (last resort) — uses scoring proxies when
        neither play-by-play nor drive-level data is available.

        Hierarchy: calculate_all_from_plays() > calculate_from_drives() > calculate_from_game_stats()

        Args:
            team: Team name
            games_df: DataFrame with game stats

        Returns:
            FinishingDrivesRating for the team
        """
        logger.debug(f"FD pathway: game-stats fallback for {team}")
        home_games = games_df[games_df["home_team"] == team]
        away_games = games_df[games_df["away_team"] == team]

        # Aggregate red zone stats if available (VECTORIZED)
        # P3.3: Replaced iterrows with .sum() for ~10x speedup
        rz_attempts = 0
        rz_tds = 0

        if "home_rz_attempts" in games_df.columns:
            rz_attempts += home_games["home_rz_attempts"].fillna(0).sum()
            rz_tds += home_games["home_rz_tds"].fillna(0).sum()

        if "away_rz_attempts" in games_df.columns:
            rz_attempts += away_games["away_rz_attempts"].fillna(0).sum()
            rz_tds += away_games["away_rz_tds"].fillna(0).sum()

        # Derive games_played
        games_played = len(home_games) + len(away_games)

        if rz_attempts == 0:
            # Use scoring as proxy (VECTORIZED)
            total_points = (
                home_games["home_points"].sum() +
                away_games["away_points"].sum()
            )

            if games_played == 0:
                return self.calculate_team_rating(team, 0, 0, 0, 0, games_played=0)

            # Estimate RZ trips from total scoring
            avg_points = total_points / games_played
            estimated_rz_trips = avg_points / self.EXPECTED_POINTS_PER_TRIP
            estimated_tds = int(estimated_rz_trips * self.EXPECTED_RZ_TD_RATE)
            estimated_fgs = int(estimated_rz_trips * 0.25)

            return self.calculate_team_rating(
                team=team,
                rz_touchdowns=estimated_tds * games_played,
                rz_field_goals=estimated_fgs * games_played,
                rz_turnovers=int(estimated_rz_trips * 0.08 * games_played),
                rz_failed=int(estimated_rz_trips * 0.07 * games_played),
                games_played=games_played,
            )

        # Calculate from actual RZ data
        rz_fgs = int(rz_attempts * 0.25)  # Estimate
        rz_turnovers = int(rz_attempts * 0.08)
        rz_failed = rz_attempts - rz_tds - rz_fgs - rz_turnovers

        return self.calculate_team_rating(
            team=team,
            rz_touchdowns=rz_tds,
            rz_field_goals=max(0, rz_fgs),
            rz_turnovers=max(0, rz_turnovers),
            rz_failed=max(0, rz_failed),
            games_played=games_played,
        )

    def calculate_all_from_game_stats(
        self,
        teams: set[str],
        games_df: pd.DataFrame,
    ) -> dict[str, FinishingDrivesRating]:
        """Calculate finishing drives ratings for all teams in a single pass.

        Batch version of calculate_from_game_stats() that uses groupby instead of
        per-team DataFrame filtering. ~130x faster for FBS (130 teams).

        Args:
            teams: Set of team names to calculate ratings for
            games_df: DataFrame with game-level stats

        Returns:
            Dict mapping team name to FinishingDrivesRating
        """
        logger.debug(f"FD batch: calculating ratings for {len(teams)} teams from game stats")

        # Check which columns are available
        has_rz = 'home_rz_attempts' in games_df.columns and 'away_rz_attempts' in games_df.columns

        # Initialize accumulators for each team
        team_stats = {team: {
            'rz_attempts': 0, 'rz_tds': 0,
            'total_points': 0, 'n_games': 0
        } for team in teams}

        # Single pass: aggregate home game stats by home_team
        home_agg = {'home_points': 'sum'}
        if has_rz:
            home_agg['home_rz_attempts'] = 'sum'
            home_agg['home_rz_tds'] = 'sum'

        home_grouped = games_df.groupby('home_team').agg(home_agg).fillna(0)

        for team in teams:
            if team in home_grouped.index:
                row = home_grouped.loc[team]
                team_stats[team]['total_points'] += row.get('home_points', 0)
                if has_rz:
                    team_stats[team]['rz_attempts'] += row.get('home_rz_attempts', 0)
                    team_stats[team]['rz_tds'] += row.get('home_rz_tds', 0)

        # Single pass: aggregate away game stats by away_team
        away_agg = {'away_points': 'sum'}
        if has_rz:
            away_agg['away_rz_attempts'] = 'sum'
            away_agg['away_rz_tds'] = 'sum'

        away_grouped = games_df.groupby('away_team').agg(away_agg).fillna(0)

        for team in teams:
            if team in away_grouped.index:
                row = away_grouped.loc[team]
                team_stats[team]['total_points'] += row.get('away_points', 0)
                if has_rz:
                    team_stats[team]['rz_attempts'] += row.get('away_rz_attempts', 0)
                    team_stats[team]['rz_tds'] += row.get('away_rz_tds', 0)

        # Count games per team (single pass each)
        home_counts = games_df.groupby('home_team').size()
        away_counts = games_df.groupby('away_team').size()
        for team in teams:
            team_stats[team]['n_games'] = (
                home_counts.get(team, 0) + away_counts.get(team, 0)
            )

        # Calculate ratings for all teams
        results = {}
        for team in teams:
            stats = team_stats[team]
            games_played = stats['n_games']

            if stats['rz_attempts'] == 0:
                # Use scoring as proxy (same logic as calculate_from_game_stats)
                if games_played == 0:
                    rating = self.calculate_team_rating(team, 0, 0, 0, 0, games_played=0)
                else:
                    avg_points = stats['total_points'] / games_played
                    estimated_rz_trips = avg_points / self.EXPECTED_POINTS_PER_TRIP
                    estimated_tds = int(estimated_rz_trips * self.EXPECTED_RZ_TD_RATE)
                    estimated_fgs = int(estimated_rz_trips * 0.25)

                    rating = self.calculate_team_rating(
                        team=team,
                        rz_touchdowns=estimated_tds * games_played,
                        rz_field_goals=estimated_fgs * games_played,
                        rz_turnovers=int(estimated_rz_trips * 0.08 * games_played),
                        rz_failed=int(estimated_rz_trips * 0.07 * games_played),
                        games_played=games_played,
                    )
            else:
                # Calculate from actual RZ data
                rz_tds = int(stats['rz_tds'])
                rz_attempts = int(stats['rz_attempts'])
                rz_fgs = int(rz_attempts * 0.25)
                rz_turnovers = int(rz_attempts * 0.08)
                rz_failed = rz_attempts - rz_tds - rz_fgs - rz_turnovers

                rating = self.calculate_team_rating(
                    team=team,
                    rz_touchdowns=rz_tds,
                    rz_field_goals=max(0, rz_fgs),
                    rz_turnovers=max(0, rz_turnovers),
                    rz_failed=max(0, rz_failed),
                    games_played=games_played,
                )

            results[team] = rating

        return results

    def get_rating(self, team: str) -> Optional[FinishingDrivesRating]:
        """Get finishing drives rating for a team.

        Args:
            team: Team name

        Returns:
            FinishingDrivesRating or None if not calculated
        """
        return self.team_ratings.get(team)

    def get_matchup_differential(
        self, team_a: str, team_b: str
    ) -> float:
        """Get finishing drives point differential in a matchup.

        Applies magnitude cap to prevent FD component from overwhelming EFM signal.
        Maximum swing is ±MAX_MATCHUP_DIFFERENTIAL points.

        Args:
            team_a: First team
            team_b: Second team

        Returns:
            Expected point advantage for team_a from finishing drives (capped)
        """
        rating_a = self.team_ratings.get(team_a)
        rating_b = self.team_ratings.get(team_b)

        overall_a = rating_a.overall_rating if rating_a else 0.0
        overall_b = rating_b.overall_rating if rating_b else 0.0

        differential = overall_a - overall_b

        # Apply magnitude cap to prevent overwhelming EFM signal
        return np.clip(differential, -self.MAX_MATCHUP_DIFFERENTIAL, self.MAX_MATCHUP_DIFFERENTIAL)

    def calculate_all_from_plays(
        self, plays_df: pd.DataFrame, max_week: int | None = None
    ) -> None:
        """Calculate finishing drives ratings for all teams from play-by-play data.

        P2.2: PRIMARY pathway — preferred for production and backtests.
        Identifies red zone plays (yards_to_goal <= 20) and tracks scoring outcomes
        at drive level using game_id + drive_id.

        Hierarchy: calculate_all_from_plays() > calculate_from_drives() > calculate_from_game_stats()

        Args:
            plays_df: Play-by-play DataFrame with yards_to_goal, offense, play_type columns
            max_week: Maximum week allowed in training data (for data leakage prevention).
                      If provided, asserts that no plays exceed this week.
        """
        # DATA LEAKAGE GUARD: Verify no future weeks in training data (check FIRST)
        if max_week is not None and "week" in plays_df.columns and not plays_df.empty:
            actual_max = plays_df["week"].max()
            assert actual_max <= max_week, (
                f"DATA LEAKAGE in FinishingDrives: plays include week {actual_max} "
                f"but max_week={max_week}. Filter plays before calling."
            )

        if plays_df.empty or "yards_to_goal" not in plays_df.columns:
            logger.warning("No yards_to_goal data available for finishing drives calculation")
            return

        # Filter to red zone plays
        # P3.5: No .copy() needed - rz_plays is only read, not modified
        rz_plays = plays_df[plays_df["yards_to_goal"] <= 20]

        if rz_plays.empty:
            return

        # Require drive identifiers for trip-level analysis
        if "drive_id" not in rz_plays.columns or "game_id" not in rz_plays.columns:
            logger.warning(
                "Cannot compute RZ trips: missing drive_id or game_id columns. "
                "Skipping all teams."
            )
            return

        # --- VECTORIZED TRIP CLASSIFICATION ---
        # Extract the last play of every RZ drive in one operation (replaces per-team for-loop).
        # Uses .tail(1) on the groupby to get the final play per (offense, game_id, drive_id).
        last_plays = rz_plays.groupby(["offense", "game_id", "drive_id"]).tail(1).copy()

        # Lowercase play_type once for all classification
        pt_lower = last_plays["play_type"].fillna("").str.lower()

        # Filter out non-competitive trips: kneeldowns, end-of-game, timeouts
        non_competitive = pt_lower.str.contains("kneel|end of|timeout", na=False)
        last_plays = last_plays[~non_competitive]
        pt_lower = pt_lower[~non_competitive]

        # Classify outcomes via vectorized string matching
        is_td = pt_lower.str.contains("touchdown|rushing td|passing td", na=False)
        is_fg = pt_lower.str.contains("field goal good", na=False)
        is_to = pt_lower.str.contains("interception|fumble recovery \\(opponent\\)", na=False)
        # FAILED = everything else (4th down stops, punts, etc.)

        last_plays["_outcome"] = "FAILED"
        last_plays.loc[is_td, "_outcome"] = "TD"
        last_plays.loc[is_fg, "_outcome"] = "FG"
        last_plays.loc[is_to, "_outcome"] = "TO"

        # Aggregate counts per team in one groupby
        outcome_counts = (
            last_plays.groupby(["offense", "_outcome"])
            .size()
            .unstack(fill_value=0)
        )
        # Ensure all outcome columns exist
        for col in ("TD", "FG", "TO", "FAILED"):
            if col not in outcome_counts.columns:
                outcome_counts[col] = 0

        # --- VECTORIZED GOAL-TO-GO (inside 10) ---
        # Get last play per GTG drive, classify TDs
        gtg_plays = rz_plays[rz_plays["yards_to_goal"] <= 10]
        if not gtg_plays.empty:
            gtg_last = gtg_plays.groupby(["offense", "game_id", "drive_id"]).tail(1)
            gtg_pt_lower = gtg_last["play_type"].fillna("").str.lower()
            gtg_is_td = gtg_pt_lower.str.contains("touchdown", na=False)

            gtg_trip_counts = gtg_last.groupby("offense").size()
            gtg_td_counts = gtg_last[gtg_is_td].groupby("offense").size()
        else:
            gtg_trip_counts = pd.Series(dtype=int)
            gtg_td_counts = pd.Series(dtype=int)

        # Games played per team (for PBTA normalization)
        games_per_team = rz_plays.groupby("offense")["game_id"].nunique()

        # DETERMINISM: Sort for consistent iteration order
        all_teams = sorted(outcome_counts.index)

        for team in all_teams:
            row = outcome_counts.loc[team]
            rz_tds = int(row["TD"])
            rz_fgs = int(row["FG"])
            rz_turnovers = int(row["TO"])
            rz_failed = int(row["FAILED"])

            gtg_tds = int(gtg_td_counts.get(team, 0))
            gtg_attempts = int(gtg_trip_counts.get(team, 0))
            games_played = int(games_per_team.get(team, 0))

            self.calculate_team_rating(
                team=team,
                rz_touchdowns=rz_tds,
                rz_field_goals=rz_fgs,
                rz_turnovers=rz_turnovers,
                rz_failed=rz_failed,
                goal_to_go_tds=gtg_tds,
                goal_to_go_attempts=gtg_attempts,
                games_played=games_played,
            )

        # DYNAMIC BASELINE: Compute FBS-wide mean PPT from all teams in training window
        # This prevents systematic bias from year-to-year PPT variation (empirical range: 3.25-3.78)
        # Walk-forward chronology guaranteed: backtest engine filters plays to weeks < prediction_week
        # before calling this method, so this mean only uses prior data.
        if self.team_ratings:
            ppt_values = [rating.points_per_trip for rating in self.team_ratings.values()]
            self._seasonal_mean_ppt = float(np.mean(ppt_values))

            # Recalculate all overall ratings using dynamic baseline
            for team, rating in self.team_ratings.items():
                new_overall = rating.points_per_trip - self._seasonal_mean_ppt
                # Create updated rating with new overall score
                updated_rating = FinishingDrivesRating(
                    team=rating.team,
                    red_zone_td_rate=rating.red_zone_td_rate,
                    red_zone_scoring_rate=rating.red_zone_scoring_rate,
                    points_per_trip=rating.points_per_trip,
                    goal_to_go_conversion=rating.goal_to_go_conversion,
                    overall_rating=new_overall,
                )
                self.team_ratings[team] = updated_rating

            logger.info(
                f"Finishing drives: computed dynamic baseline PPT = {self._seasonal_mean_ppt:.3f} "
                f"from {len(self.team_ratings)} teams"
            )

        # P3.9: Debug level for per-week logging
        logger.debug(f"Calculated finishing drives ratings for {len(self.team_ratings)} teams")

    def get_summary_df(self) -> pd.DataFrame:
        """Get summary of all team ratings as a DataFrame.

        Returns:
            DataFrame with finishing drives ratings
        """
        if not self.team_ratings:
            return pd.DataFrame()

        data = [
            {
                "team": r.team,
                "rz_td_rate": r.red_zone_td_rate,
                "rz_scoring_rate": r.red_zone_scoring_rate,
                "points_per_trip": r.points_per_trip,
                "goal_to_go": r.goal_to_go_conversion,
                "overall": r.overall_rating,
            }
            for r in self.team_ratings.values()
        ]

        df = pd.DataFrame(data)
        return df.sort_values("overall", ascending=False).reset_index(drop=True)
