"""Luck regression model to regress outlier statistics toward expected values."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LuckMetrics:
    """Container for luck-adjusted metrics."""

    team: str
    raw_turnover_margin: float
    adjusted_turnover_margin: float
    raw_fumble_recovery_rate: float
    adjusted_fumble_recovery_rate: float
    raw_close_game_record: float  # Win % in close games
    adjusted_close_game_record: float
    total_luck_adjustment: float  # Points adjustment


class LuckRegressor:
    """
    Regress luck-influenced statistics toward expected values.

    Key stats with low year-to-year correlation that indicate luck:
    - Turnover margin (R² ≈ 0.026 year-to-year)
    - Fumble recovery rate (expected: 50%)
    - Record in close games (expected: 50%)
    """

    # Expected values (regression targets)
    EXPECTED_FUMBLE_RECOVERY = 0.50
    EXPECTED_CLOSE_GAME_WIN_PCT = 0.50
    EXPECTED_TURNOVER_MARGIN = 0.0

    # Points per turnover (approximate EPA impact)
    POINTS_PER_TURNOVER = 4.5

    def __init__(
        self,
        regression_factor: Optional[float] = None,
        skip_turnover_luck: bool = False,
    ):
        """Initialize the luck regressor.

        Args:
            regression_factor: How much to regress toward mean (0-1).
                             0 = no regression, 1 = fully regress to mean.
                             If None, uses settings default (0.5).
            skip_turnover_luck: When True, zero out turnover luck component.
                              Use when turnover noise is already pre-scrubbed
                              from margins to avoid double-counting.
        """
        if regression_factor is None:
            settings = get_settings()
            regression_factor = settings.luck_regression_factor

        self.regression_factor = regression_factor
        self.skip_turnover_luck = skip_turnover_luck
        self.team_metrics: dict[str, LuckMetrics] = {}

    def _regress_to_mean(
        self, value: float, expected: float, factor: Optional[float] = None
    ) -> float:
        """Regress a value toward its expected value.

        Args:
            value: Observed value
            expected: Expected (mean) value
            factor: Regression factor (uses self.regression_factor if None)

        Returns:
            Regressed value
        """
        if factor is None:
            factor = self.regression_factor
        return value + factor * (expected - value)

    def calculate_turnover_luck(
        self,
        turnovers_gained: int,
        turnovers_lost: int,
        games_played: int,
        factor: Optional[float] = None,
    ) -> tuple[float, float, float]:
        """Calculate turnover margin luck adjustment.

        Args:
            turnovers_gained: Total turnovers forced
            turnovers_lost: Total turnovers committed
            games_played: Number of games played
            factor: Regression factor override (uses self.regression_factor if None)

        Returns:
            Tuple of (raw_margin_per_game, adjusted_margin_per_game, luck_points)
        """
        raw_margin = (turnovers_gained - turnovers_lost) / max(games_played, 1)
        adjusted_margin = self._regress_to_mean(
            raw_margin, self.EXPECTED_TURNOVER_MARGIN, factor=factor
        )

        # Convert margin difference to points
        # positive = unlucky (raw margin worse than adjusted → rating should go up)
        margin_diff = adjusted_margin - raw_margin
        luck_points = margin_diff * self.POINTS_PER_TURNOVER

        return raw_margin, adjusted_margin, luck_points

    def calculate_fumble_luck(
        self,
        fumbles_recovered: int,
        total_fumbles: int,
    ) -> tuple[float, float, float]:
        """Calculate fumble recovery luck adjustment.

        Args:
            fumbles_recovered: Own fumbles recovered
            total_fumbles: Total fumbles (by both teams in games)

        Returns:
            Tuple of (raw_rate, adjusted_rate, luck_points)
        """
        if total_fumbles == 0:
            return 0.5, 0.5, 0.0

        raw_rate = fumbles_recovered / total_fumbles
        adjusted_rate = self._regress_to_mean(raw_rate, self.EXPECTED_FUMBLE_RECOVERY)

        # Convert to expected fumbles lost difference
        fumbles_diff = (adjusted_rate - raw_rate) * total_fumbles
        luck_points = fumbles_diff * self.POINTS_PER_TURNOVER * 0.5  # Partial impact

        return raw_rate, adjusted_rate, luck_points

    def calculate_close_game_luck(
        self,
        close_wins: int,
        close_games: int,
        points_threshold: int = 7,
        factor: Optional[float] = None,
    ) -> tuple[float, float, float]:
        """Calculate close game luck adjustment.

        Close games are defined as games decided by <= points_threshold.

        Args:
            close_wins: Wins in close games
            close_games: Total close games played
            points_threshold: Margin threshold for "close" game
            factor: Regression factor override (uses self.regression_factor if None)

        Returns:
            Tuple of (raw_win_pct, adjusted_win_pct, luck_points)
        """
        if close_games == 0:
            return 0.5, 0.5, 0.0

        raw_win_pct = close_wins / close_games
        adjusted_win_pct = self._regress_to_mean(
            raw_win_pct, self.EXPECTED_CLOSE_GAME_WIN_PCT, factor=factor
        )

        # Expected wins difference -> points adjustment
        # positive = unlucky (raw win pct worse than adjusted → rating should go up)
        expected_win_diff = (adjusted_win_pct - raw_win_pct) * close_games
        # Each "lucky" win is worth approximately 3-4 points in margin
        luck_points = expected_win_diff * 3.5

        return raw_win_pct, adjusted_win_pct, luck_points

    def calculate_team_luck(
        self,
        team: str,
        games_df: pd.DataFrame,
    ) -> LuckMetrics:
        """Calculate comprehensive luck metrics for a team.

        Args:
            team: Team name
            games_df: DataFrame with game data including:
                - home_team, away_team
                - home_turnovers, away_turnovers (optional)
                - home_points, away_points
                - margin (or will be calculated)

        Returns:
            LuckMetrics for the team
        """
        # Filter to team's games
        home_games = games_df[games_df["home_team"] == team]
        away_games = games_df[games_df["away_team"] == team]

        games_played = len(home_games) + len(away_games)

        if games_played == 0:
            return LuckMetrics(
                team=team,
                raw_turnover_margin=0.0,
                adjusted_turnover_margin=0.0,
                raw_fumble_recovery_rate=0.5,
                adjusted_fumble_recovery_rate=0.5,
                raw_close_game_record=0.5,
                adjusted_close_game_record=0.5,
                total_luck_adjustment=0.0,
            )

        # Scale regression factor with sample size: regress more early, less late
        dynamic_factor = self.regression_factor * max(1.0 - games_played / 20.0, 0.2)

        # Calculate turnover margin if data available
        turnovers_gained = 0
        turnovers_lost = 0
        turnover_data_available = (
            "home_turnovers" in games_df.columns
            and "away_turnovers" in games_df.columns
        )

        if turnover_data_available:
            for _, game in home_games.iterrows():
                turnovers_gained += game.get("away_turnovers", 0)
                turnovers_lost += game.get("home_turnovers", 0)
            for _, game in away_games.iterrows():
                turnovers_gained += game.get("home_turnovers", 0)
                turnovers_lost += game.get("away_turnovers", 0)

        raw_to, adj_to, to_luck = self.calculate_turnover_luck(
            turnovers_gained, turnovers_lost, games_played,
            factor=dynamic_factor,
        )

        if self.skip_turnover_luck:
            to_luck = 0.0

        # Calculate close game record
        close_wins = 0
        close_games_count = 0
        close_threshold = 7

        for _, game in home_games.iterrows():
            margin = game["home_points"] - game["away_points"]
            if abs(margin) <= close_threshold:
                close_games_count += 1
                if margin > 0:
                    close_wins += 1

        for _, game in away_games.iterrows():
            margin = game["away_points"] - game["home_points"]
            if abs(margin) <= close_threshold:
                close_games_count += 1
                if margin > 0:
                    close_wins += 1

        raw_close, adj_close, close_luck = self.calculate_close_game_luck(
            close_wins, close_games_count, factor=dynamic_factor,
        )

        # Fumble luck (approximation if detailed data not available)
        raw_fumble = 0.5
        adj_fumble = 0.5
        fumble_luck = 0.0

        total_luck = to_luck + close_luck + fumble_luck

        metrics = LuckMetrics(
            team=team,
            raw_turnover_margin=raw_to,
            adjusted_turnover_margin=adj_to,
            raw_fumble_recovery_rate=raw_fumble,
            adjusted_fumble_recovery_rate=adj_fumble,
            raw_close_game_record=raw_close,
            adjusted_close_game_record=adj_close,
            total_luck_adjustment=total_luck,
        )

        self.team_metrics[team] = metrics
        return metrics

    def calculate_all_teams(self, games_df: pd.DataFrame) -> dict[str, LuckMetrics]:
        """Calculate luck metrics for all teams in the dataset.

        Args:
            games_df: DataFrame with game data

        Returns:
            Dictionary mapping team names to LuckMetrics
        """
        all_teams = set(games_df["home_team"]) | set(games_df["away_team"])

        for team in all_teams:
            self.calculate_team_luck(team, games_df)

        logger.info(f"Calculated luck metrics for {len(self.team_metrics)} teams")
        return self.team_metrics

    def get_luck_adjustment(self, team: str) -> float:
        """Get the luck points adjustment for a team.

        Positive adjustment means team has been unlucky (should add points to rating).
        Negative adjustment means team has been lucky (should subtract points).

        Args:
            team: Team name

        Returns:
            Points adjustment for luck
        """
        metrics = self.team_metrics.get(team)
        if metrics is None:
            return 0.0
        return metrics.total_luck_adjustment

    def get_summary_df(self) -> pd.DataFrame:
        """Get summary of all team luck metrics as a DataFrame.

        Returns:
            DataFrame with luck metrics for all teams
        """
        if not self.team_metrics:
            return pd.DataFrame()

        data = [
            {
                "team": m.team,
                "raw_to_margin": m.raw_turnover_margin,
                "adj_to_margin": m.adjusted_turnover_margin,
                "raw_close_pct": m.raw_close_game_record,
                "adj_close_pct": m.adjusted_close_game_record,
                "luck_adjustment": m.total_luck_adjustment,
            }
            for m in self.team_metrics.values()
        ]

        df = pd.DataFrame(data)
        return df.sort_values("luck_adjustment", ascending=False).reset_index(drop=True)
