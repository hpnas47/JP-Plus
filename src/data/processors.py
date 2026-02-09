"""Data processing utilities including garbage time filtering and recency weighting."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GarbageTimeThresholds:
    """Thresholds for garbage time detection by quarter."""

    q1: int = 28
    q2: int = 24
    q3: int = 21
    q4: int = 16


class GarbageTimeFilter:
    """Filter out garbage time plays based on score differential."""

    def __init__(self, thresholds: Optional[GarbageTimeThresholds] = None):
        """Initialize with thresholds.

        Args:
            thresholds: Custom thresholds or None to use defaults from settings
        """
        if thresholds is None:
            settings = get_settings()
            thresholds = GarbageTimeThresholds(
                q1=settings.garbage_time_q1,
                q2=settings.garbage_time_q2,
                q3=settings.garbage_time_q3,
                q4=settings.garbage_time_q4,
            )
        self.thresholds = thresholds

    def is_garbage_time(
        self, quarter: int, score_diff: int, time_remaining: Optional[float] = None
    ) -> bool:
        """Determine if a play occurred in garbage time.

        Args:
            quarter: Quarter number (1-4, 5+ for OT)
            score_diff: Absolute score differential
            time_remaining: Minutes remaining in quarter (optional refinement)

        Returns:
            True if play is in garbage time
        """
        # Overtime is never garbage time
        if quarter > 4:
            return False

        threshold = {
            1: self.thresholds.q1,
            2: self.thresholds.q2,
            3: self.thresholds.q3,
            4: self.thresholds.q4,
        }.get(quarter, self.thresholds.q4)

        return abs(score_diff) >= threshold

    def filter_plays(self, plays_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out garbage time plays from a DataFrame.

        Expected columns:
            - period: Quarter number
            - home_score: Home team score
            - away_score: Away team score

        Args:
            plays_df: DataFrame with play-by-play data

        Returns:
            DataFrame with garbage time plays removed
        """
        if plays_df.empty:
            return plays_df

        # Calculate score differential
        plays_df = plays_df.copy()
        plays_df["score_diff"] = abs(
            plays_df["home_score"] - plays_df["away_score"]
        )

        # Apply garbage time filter
        mask = ~plays_df.apply(
            lambda row: self.is_garbage_time(row["period"], row["score_diff"]),
            axis=1,
        )

        filtered = plays_df[mask].drop(columns=["score_diff"])
        removed_count = len(plays_df) - len(filtered)

        if removed_count > 0:
            logger.debug(f"Filtered {removed_count} garbage time plays")

        return filtered

class RecencyWeighter:
    """Apply exponential decay weighting to game data based on recency."""

    def __init__(self, decay_rate: Optional[float] = None):
        """Initialize with decay rate.

        Args:
            decay_rate: Decay parameter (xi). If None, uses settings default.
                       Higher values = faster decay (more weight to recent games)
        """
        if decay_rate is None:
            settings = get_settings()
            decay_rate = settings.recency_decay
        self.decay_rate = decay_rate

    def calculate_weight(self, days_ago: float) -> float:
        """Calculate weight for a game based on days since it occurred.

        Args:
            days_ago: Number of days since the game

        Returns:
            Weight between 0 and 1
        """
        return np.exp(-self.decay_rate * days_ago)

    def calculate_half_life_days(self) -> float:
        """Calculate the half-life in days (when weight = 0.5)."""
        return np.log(2) / self.decay_rate

    def add_weights(
        self,
        games_df: pd.DataFrame,
        date_column: str = "start_date",
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Add recency weights to a games DataFrame.

        Args:
            games_df: DataFrame with game data
            date_column: Name of the date column
            reference_date: Reference date for calculating days ago.
                           If None, uses today.

        Returns:
            DataFrame with 'recency_weight' column added
        """
        if games_df.empty:
            games_df["recency_weight"] = []
            return games_df

        games_df = games_df.copy()

        if reference_date is None:
            reference_date = datetime.now()

        # Parse dates if needed
        if games_df[date_column].dtype == object:
            games_df[date_column] = pd.to_datetime(games_df[date_column])

        # Handle timezone mismatches
        if games_df[date_column].dt.tz is None:
            # Dates are naive, make reference naive too
            if hasattr(reference_date, "tzinfo") and reference_date.tzinfo is not None:
                reference_date = reference_date.replace(tzinfo=None)
        else:
            # Dates are tz-aware, make reference tz-aware or convert dates to naive
            if not hasattr(reference_date, "tzinfo") or reference_date.tzinfo is None:
                # Convert dates to naive for comparison
                games_df[date_column] = games_df[date_column].dt.tz_localize(None)

        # Calculate days ago
        games_df["days_ago"] = (reference_date - games_df[date_column]).dt.days

        # Apply exponential decay
        games_df["recency_weight"] = games_df["days_ago"].apply(self.calculate_weight)

        # Drop intermediate column
        games_df = games_df.drop(columns=["days_ago"])

        return games_df

    def weighted_mean(
        self, values: np.ndarray, weights: np.ndarray
    ) -> float:
        """Calculate weighted mean.

        Args:
            values: Array of values
            weights: Array of weights

        Returns:
            Weighted mean
        """
        if len(values) == 0:
            return 0.0
        return np.average(values, weights=weights)


class DataProcessor:
    """Main data processor combining garbage time filter and recency weighting."""

    def __init__(
        self,
        garbage_filter: Optional[GarbageTimeFilter] = None,
        recency_weighter: Optional[RecencyWeighter] = None,
    ):
        """Initialize data processor.

        Args:
            garbage_filter: Custom garbage time filter or None for default
            recency_weighter: Custom recency weighter or None for default
        """
        self.garbage_filter = garbage_filter or GarbageTimeFilter()
        self.recency_weighter = recency_weighter or RecencyWeighter()

    def process_games(
        self,
        games_df: pd.DataFrame,
        apply_garbage_filter: bool = True,
        apply_recency_weights: bool = True,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Process games data with filtering and weighting.

        Args:
            games_df: DataFrame with game data
            apply_garbage_filter: Whether to filter garbage time
            apply_recency_weights: Whether to add recency weights
            reference_date: Reference date for recency calculation

        Returns:
            Processed DataFrame
        """
        if games_df.empty:
            return games_df

        processed = games_df.copy()

        # Note: Garbage time filter requires play-by-play data
        # For game-level stats, we'd need to flag games with significant garbage time

        if apply_recency_weights:
            processed = self.recency_weighter.add_weights(
                processed, reference_date=reference_date
            )

        return processed

    def process_plays(
        self,
        plays_df: pd.DataFrame,
        apply_garbage_filter: bool = True,
    ) -> pd.DataFrame:
        """Process play-by-play data.

        Args:
            plays_df: DataFrame with play-by-play data
            apply_garbage_filter: Whether to filter garbage time

        Returns:
            Processed DataFrame
        """
        if plays_df.empty:
            return plays_df

        processed = plays_df.copy()

        if apply_garbage_filter:
            processed = self.garbage_filter.filter_plays(processed)

        return processed

    def aggregate_team_stats(
        self,
        games_df: pd.DataFrame,
        team: str,
        stat_columns: list[str],
        use_weights: bool = True,
    ) -> dict:
        """Aggregate statistics for a team across games.

        Args:
            games_df: DataFrame with processed game data
            team: Team name to aggregate for
            stat_columns: List of statistic columns to aggregate
            use_weights: Whether to use recency weights

        Returns:
            Dictionary of aggregated statistics
        """
        # Filter to team's games
        team_games = games_df[
            (games_df["home_team"] == team) | (games_df["away_team"] == team)
        ]

        if team_games.empty:
            return {col: 0.0 for col in stat_columns}

        result = {}
        weights = (
            team_games["recency_weight"].values
            if use_weights and "recency_weight" in team_games.columns
            else np.ones(len(team_games))
        )

        for col in stat_columns:
            if col in team_games.columns:
                values = team_games[col].values
                result[col] = self.recency_weighter.weighted_mean(values, weights)
            else:
                result[col] = 0.0

        return result
