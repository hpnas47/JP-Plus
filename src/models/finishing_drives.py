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
    EXPECTED_POINTS_PER_TRIP = 4.05  # Average points per RZ trip (empirical FBS mean: ~4.0-4.1 pts/trip)
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

        # Overall rating: points above expected, normalized to per-game basis
        # Formula: (points_per_trip - expected) * avg_trips_per_game
        # This keeps the rating in PBTA units (points per game)
        expected_points = self.EXPECTED_POINTS_PER_TRIP

        # Normalize RZ trips to per-game basis to prevent cumulative inflation
        if games_played and games_played > 0:
            avg_rz_trips_per_game = total_rz_trips / games_played
        else:
            # Fallback: estimate ~2.5 RZ trips per game (FBS average)
            # This prevents division by zero and maintains backwards compatibility
            avg_rz_trips_per_game = total_rz_trips / max(1, total_rz_trips // 2.5)

        overall = (points_per_trip - expected_points) * avg_rz_trips_per_game

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

        # DETERMINISM: Sort for consistent iteration order
        all_teams = sorted(set(rz_plays["offense"].dropna()))

        for team in all_teams:
            team_rz = rz_plays[rz_plays["offense"] == team]

            if team_rz.empty:
                continue

            # P0.1 FIX: Count RED ZONE TRIPS at drive level, not play level
            # A trip = one distinct possession that entered the red zone
            # Use game_id + drive_id to uniquely identify trips

            if "drive_id" not in team_rz.columns or "game_id" not in team_rz.columns:
                # Fallback: cannot compute trips without drive identifiers
                logger.warning(
                    f"Cannot compute RZ trips for {team}: missing drive_id or game_id columns. "
                    "Skipping team."
                )
                continue

            # P0.1: Count DISTINCT drives that entered red zone
            rz_trips = team_rz.groupby(["game_id", "drive_id"])

            # Classify each trip by outcome (look at last play of trip in RZ)
            # For each trip, get the last play's play_type to determine outcome
            trip_outcomes = []

            for (game_id, drive_id), trip_plays in rz_trips:
                # Sort by play order (use index as proxy if no explicit play number)
                # Last play in RZ determines trip outcome
                last_play = trip_plays.iloc[-1]
                play_type_lower = str(last_play.get("play_type", "")).lower()

                # Classify outcome
                if "touchdown" in play_type_lower or "rushing td" in play_type_lower or "passing td" in play_type_lower:
                    outcome = "TD"
                elif "field goal good" in play_type_lower:
                    outcome = "FG"
                elif "interception" in play_type_lower or "fumble recovery (opponent)" in play_type_lower:
                    outcome = "TO"
                elif last_play.get("down") == 4:
                    # 4th down stop (failed conversion)
                    outcome = "FAILED"
                else:
                    # Other failures (punt, turnover on downs without 4th down marker, etc.)
                    outcome = "FAILED"

                trip_outcomes.append(outcome)

            # Count trips by outcome
            rz_tds = trip_outcomes.count("TD")
            rz_fgs = trip_outcomes.count("FG")
            rz_turnovers = trip_outcomes.count("TO")
            rz_failed = trip_outcomes.count("FAILED")

            # P0.1: Goal-to-go (inside 10) - also count TRIPS not plays
            # A GTG trip = one distinct possession that started inside 10
            gtg_mask = team_rz["yards_to_goal"] <= 10
            if gtg_mask.any():
                gtg_trips = team_rz[gtg_mask].groupby(["game_id", "drive_id"])

                gtg_td_count = 0
                gtg_trip_count = 0

                for (game_id, drive_id), trip_plays in gtg_trips:
                    gtg_trip_count += 1
                    last_play = trip_plays.iloc[-1]
                    play_type_lower = str(last_play.get("play_type", "")).lower()
                    if "touchdown" in play_type_lower:
                        gtg_td_count += 1

                gtg_tds = gtg_td_count
                gtg_attempts = gtg_trip_count
            else:
                gtg_tds = 0
                gtg_attempts = 0

            # Derive games_played from unique game_ids in team's plays
            # This ensures per-game normalization for PBTA calculation
            games_played = team_rz["game_id"].nunique()

            self.calculate_team_rating(
                team=team,
                rz_touchdowns=int(rz_tds),
                rz_field_goals=int(rz_fgs),
                rz_turnovers=int(rz_turnovers),
                rz_failed=int(rz_failed),
                goal_to_go_tds=int(gtg_tds),
                goal_to_go_attempts=int(gtg_attempts),
                games_played=games_played,
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
