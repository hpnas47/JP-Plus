"""Situational adjustments for bye weeks, letdown/lookahead spots, and rivalries."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from config.settings import get_settings
from config.teams import is_rivalry_game

logger = logging.getLogger(__name__)


@dataclass
class SituationalFactors:
    """Container for situational adjustment factors."""

    team: str
    bye_week_advantage: float = 0.0
    letdown_penalty: float = 0.0
    lookahead_penalty: float = 0.0
    rivalry_boost: float = 0.0
    total_adjustment: float = 0.0

    def __post_init__(self):
        self.total_adjustment = (
            self.bye_week_advantage
            + self.letdown_penalty
            + self.lookahead_penalty
            + self.rivalry_boost
        )


class SituationalAdjuster:
    """
    Calculate situational adjustments for matchups.

    Factors:
    - Bye week advantage: Team coming off bye week gets boost
    - Letdown spot: Team coming off big win vs top-15 now facing unranked
    - Look-ahead spot: Team with rival/top-10 opponent next week
    - Rivalry: Underdog gets small boost in rivalry games
    """

    def __init__(
        self,
        bye_advantage: Optional[float] = None,
        letdown_penalty: Optional[float] = None,
        lookahead_penalty: Optional[float] = None,
        rivalry_boost: Optional[float] = None,
    ):
        """Initialize situational adjuster.

        Args:
            bye_advantage: Points for bye week advantage
            letdown_penalty: Points penalty for letdown spot
            lookahead_penalty: Points penalty for look-ahead spot
            rivalry_boost: Points boost for underdog in rivalry
        """
        settings = get_settings()

        self.bye_advantage = (
            bye_advantage if bye_advantage is not None else settings.bye_week_advantage
        )
        self.letdown_penalty = (
            letdown_penalty
            if letdown_penalty is not None
            else settings.letdown_penalty
        )
        self.lookahead_penalty = (
            lookahead_penalty
            if lookahead_penalty is not None
            else settings.lookahead_penalty
        )
        self.rivalry_boost = (
            rivalry_boost
            if rivalry_boost is not None
            else settings.rivalry_underdog_boost
        )

    def check_bye_week(
        self,
        team: str,
        current_week: int,
        schedule_df: pd.DataFrame,
    ) -> bool:
        """Check if team is coming off a bye week.

        Args:
            team: Team name
            current_week: Current week number
            schedule_df: DataFrame with schedule (must have week, home_team, away_team)

        Returns:
            True if team had bye last week
        """
        if current_week <= 1:
            return False

        # Check if team played last week
        last_week_games = schedule_df[schedule_df["week"] == current_week - 1]
        team_played = (
            (last_week_games["home_team"] == team)
            | (last_week_games["away_team"] == team)
        ).any()

        return not team_played

    def check_letdown_spot(
        self,
        team: str,
        current_week: int,
        opponent: str,
        schedule_df: pd.DataFrame,
        rankings: Optional[dict[str, int]] = None,
    ) -> bool:
        """Check if team is in a letdown spot.

        Letdown: Coming off a win vs top-15 team, now facing unranked.

        Args:
            team: Team name
            current_week: Current week number
            opponent: Current opponent
            schedule_df: DataFrame with schedule and results
            rankings: Dict of team -> ranking (None if unranked)

        Returns:
            True if team is in letdown spot
        """
        if current_week <= 1 or rankings is None:
            return False

        # Check last week's game
        last_week = schedule_df[schedule_df["week"] == current_week - 1]
        team_games = last_week[
            (last_week["home_team"] == team) | (last_week["away_team"] == team)
        ]

        if team_games.empty:
            return False  # Had bye week

        last_game = team_games.iloc[0]

        # Determine if they won
        if last_game["home_team"] == team:
            won = last_game.get("home_points", 0) > last_game.get("away_points", 0)
            last_opponent = last_game["away_team"]
        else:
            won = last_game.get("away_points", 0) > last_game.get("home_points", 0)
            last_opponent = last_game["home_team"]

        if not won:
            return False

        # Check if last opponent was top-15
        last_opp_rank = rankings.get(last_opponent)
        if last_opp_rank is None or last_opp_rank > 15:
            return False

        # Check if current opponent is unranked
        current_opp_rank = rankings.get(opponent)
        return current_opp_rank is None

    def check_lookahead_spot(
        self,
        team: str,
        current_week: int,
        schedule_df: pd.DataFrame,
        rankings: Optional[dict[str, int]] = None,
    ) -> bool:
        """Check if team is in a look-ahead spot.

        Look-ahead: Next week is vs rival or top-10 team.

        Args:
            team: Team name
            current_week: Current week number
            schedule_df: DataFrame with schedule
            rankings: Dict of team -> ranking

        Returns:
            True if team is in look-ahead spot
        """
        # Check next week's game
        next_week = schedule_df[schedule_df["week"] == current_week + 1]
        team_games = next_week[
            (next_week["home_team"] == team) | (next_week["away_team"] == team)
        ]

        if team_games.empty:
            return False

        next_game = team_games.iloc[0]

        # Get next opponent
        if next_game["home_team"] == team:
            next_opponent = next_game["away_team"]
        else:
            next_opponent = next_game["home_team"]

        # Check if it's a rivalry game
        if is_rivalry_game(team, next_opponent):
            return True

        # Check if next opponent is top-10
        if rankings:
            next_opp_rank = rankings.get(next_opponent)
            if next_opp_rank is not None and next_opp_rank <= 10:
                return True

        return False

    def check_rivalry(
        self,
        team_a: str,
        team_b: str,
    ) -> bool:
        """Check if two teams are rivals.

        Args:
            team_a: First team
            team_b: Second team

        Returns:
            True if teams are rivals
        """
        return is_rivalry_game(team_a, team_b)

    def calculate_factors(
        self,
        team: str,
        opponent: str,
        current_week: int,
        schedule_df: pd.DataFrame,
        rankings: Optional[dict[str, int]] = None,
        team_is_favorite: bool = True,
    ) -> SituationalFactors:
        """Calculate all situational factors for a team in a matchup.

        Args:
            team: Team to calculate factors for
            opponent: Opponent team
            current_week: Current week number
            schedule_df: DataFrame with schedule and results
            rankings: Dict of team -> ranking
            team_is_favorite: Whether team is favored (for rivalry boost)

        Returns:
            SituationalFactors for the team
        """
        factors = SituationalFactors(team=team)

        # Bye week advantage
        if self.check_bye_week(team, current_week, schedule_df):
            factors.bye_week_advantage = self.bye_advantage
            logger.debug(f"{team} coming off bye: +{self.bye_advantage}")

        # Letdown spot
        if self.check_letdown_spot(team, current_week, opponent, schedule_df, rankings):
            factors.letdown_penalty = self.letdown_penalty
            logger.debug(f"{team} in letdown spot: {self.letdown_penalty}")

        # Look-ahead spot
        if self.check_lookahead_spot(team, current_week, schedule_df, rankings):
            factors.lookahead_penalty = self.lookahead_penalty
            logger.debug(f"{team} in look-ahead spot: {self.lookahead_penalty}")

        # Rivalry boost (for underdog only)
        if self.check_rivalry(team, opponent) and not team_is_favorite:
            factors.rivalry_boost = self.rivalry_boost
            logger.debug(f"{team} rivalry underdog boost: +{self.rivalry_boost}")

        # Recalculate total
        factors.total_adjustment = (
            factors.bye_week_advantage
            + factors.letdown_penalty
            + factors.lookahead_penalty
            + factors.rivalry_boost
        )

        return factors

    def get_matchup_adjustment(
        self,
        home_team: str,
        away_team: str,
        current_week: int,
        schedule_df: pd.DataFrame,
        rankings: Optional[dict[str, int]] = None,
        home_is_favorite: bool = True,
    ) -> tuple[float, dict]:
        """Get net situational adjustment for a matchup.

        Args:
            home_team: Home team name
            away_team: Away team name
            current_week: Current week number
            schedule_df: Schedule DataFrame
            rankings: Team rankings
            home_is_favorite: Whether home team is favored

        Returns:
            Tuple of (net adjustment favoring home, breakdown dict)
        """
        home_factors = self.calculate_factors(
            team=home_team,
            opponent=away_team,
            current_week=current_week,
            schedule_df=schedule_df,
            rankings=rankings,
            team_is_favorite=home_is_favorite,
        )

        away_factors = self.calculate_factors(
            team=away_team,
            opponent=home_team,
            current_week=current_week,
            schedule_df=schedule_df,
            rankings=rankings,
            team_is_favorite=not home_is_favorite,
        )

        net_adjustment = home_factors.total_adjustment - away_factors.total_adjustment

        breakdown = {
            "home_bye": home_factors.bye_week_advantage,
            "away_bye": away_factors.bye_week_advantage,
            "home_letdown": home_factors.letdown_penalty,
            "away_letdown": away_factors.letdown_penalty,
            "home_lookahead": home_factors.lookahead_penalty,
            "away_lookahead": away_factors.lookahead_penalty,
            "home_rivalry": home_factors.rivalry_boost,
            "away_rivalry": away_factors.rivalry_boost,
            "net": net_adjustment,
        }

        return net_adjustment, breakdown
