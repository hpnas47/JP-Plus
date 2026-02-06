"""Data validation utilities."""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.api.cfbd_client import CFBDClient

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a data validation check."""

    is_valid: bool
    message: str
    details: Optional[dict] = None


class DataValidator:
    """Validate data completeness and quality."""

    def __init__(self, client: CFBDClient):
        """Initialize validator with API client.

        Args:
            client: CFBD API client instance
        """
        self.client = client

    def validate_week_data(self, year: int, week: int) -> ValidationResult:
        """Validate that data for a week is complete.

        Args:
            year: Season year
            week: Week number

        Returns:
            ValidationResult with status and details
        """
        try:
            games = self.client.get_games(year, week)

            if not games:
                return ValidationResult(
                    is_valid=False,
                    message=f"No games found for {year} week {week}",
                )

            total_games = len(games)
            completed_games = len([g for g in games if g.home_points is not None])
            completion_rate = completed_games / total_games

            details = {
                "total_games": total_games,
                "completed_games": completed_games,
                "completion_rate": completion_rate,
            }

            if completion_rate < 0.8:
                return ValidationResult(
                    is_valid=False,
                    message=f"Only {completed_games}/{total_games} games have scores",
                    details=details,
                )

            return ValidationResult(
                is_valid=True,
                message=f"Data complete: {completed_games}/{total_games} games",
                details=details,
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Validation error: {str(e)}",
            )

    def validate_betting_data(self, year: int, week: int) -> ValidationResult:
        """Validate betting lines availability.

        Args:
            year: Season year
            week: Week number

        Returns:
            ValidationResult with status and details
        """
        try:
            lines = self.client.get_betting_lines(year, week)

            if not lines:
                return ValidationResult(
                    is_valid=False,
                    message=f"No betting lines found for {year} week {week}",
                )

            # Count games with consensus lines
            games_with_lines = len(lines)
            consensus_lines = sum(
                1
                for line in lines
                if any(l.provider == "consensus" for l in (line.lines or []))
            )

            details = {
                "games_with_lines": games_with_lines,
                "consensus_lines": consensus_lines,
            }

            return ValidationResult(
                is_valid=True,
                message=f"Found {games_with_lines} games with betting lines",
                details=details,
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Betting data validation error: {str(e)}",
            )

    def validate_team_coverage(
        self, year: int, expected_teams: Optional[list[str]] = None
    ) -> ValidationResult:
        """Validate that we have data for expected teams.

        Args:
            year: Season year
            expected_teams: List of teams to check (or None for all FBS)

        Returns:
            ValidationResult with status and details
        """
        try:
            if expected_teams is None:
                fbs_teams = self.client.get_fbs_teams(year)
                expected_teams = [t.school for t in fbs_teams]

            games = self.client.get_games(year)

            # Get all teams that appear in games
            teams_in_games = set()
            for game in games:
                teams_in_games.add(game.home_team)
                teams_in_games.add(game.away_team)

            # Check coverage
            missing_teams = [t for t in expected_teams if t not in teams_in_games]

            details = {
                "expected_teams": len(expected_teams),
                "teams_found": len(teams_in_games),
                "missing_teams": missing_teams[:10],  # First 10
            }

            if missing_teams:
                return ValidationResult(
                    is_valid=False,
                    message=f"Missing data for {len(missing_teams)} teams",
                    details=details,
                )

            return ValidationResult(
                is_valid=True,
                message=f"All {len(expected_teams)} teams have game data",
                details=details,
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Team coverage validation error: {str(e)}",
            )

    def validate_games_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a games DataFrame has required columns and data quality.

        Args:
            df: Games DataFrame to validate

        Returns:
            ValidationResult with status and details
        """
        required_columns = [
            "home_team",
            "away_team",
            "home_points",
            "away_points",
        ]

        # Check columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return ValidationResult(
                is_valid=False,
                message=f"Missing required columns: {missing_columns}",
            )

        # Check for nulls in required fields
        null_counts = df[required_columns].isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0].to_dict()

        if columns_with_nulls:
            return ValidationResult(
                is_valid=False,
                message=f"Null values in required columns",
                details={"null_counts": columns_with_nulls},
            )

        # Check for reasonable score values
        invalid_scores = (
            (df["home_points"] < 0)
            | (df["away_points"] < 0)
            | (df["home_points"] > 100)
            | (df["away_points"] > 100)
        )

        if invalid_scores.any():
            return ValidationResult(
                is_valid=False,
                message=f"Found {invalid_scores.sum()} games with invalid scores",
            )

        return ValidationResult(
            is_valid=True,
            message=f"DataFrame valid: {len(df)} games",
            details={"game_count": len(df)},
        )

    def validate_all(self, year: int, week: int) -> dict[str, ValidationResult]:
        """Run all validations for a year/week.

        Args:
            year: Season year
            week: Week number

        Returns:
            Dictionary of validation results by check name
        """
        return {
            "week_data": self.validate_week_data(year, week),
            "betting_data": self.validate_betting_data(year, week),
            "team_coverage": self.validate_team_coverage(year),
        }
