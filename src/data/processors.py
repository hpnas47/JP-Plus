"""Data processing utilities for garbage time filtering."""

import logging
from dataclasses import dataclass
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

        # Vectorized garbage time detection (10-50x faster than .apply)
        period = plays_df["period"].values
        score_diff = np.abs(
            plays_df["home_score"].values - plays_df["away_score"].values
        )

        # Map quarters to thresholds (OT periods > 4 get Q4 threshold, then excluded)
        thresholds = np.select(
            [period == 1, period == 2, period == 3, period >= 4],
            [self.thresholds.q1, self.thresholds.q2, self.thresholds.q3, self.thresholds.q4],
            default=self.thresholds.q4,
        )

        # Garbage time: score_diff >= threshold AND not overtime
        is_garbage = (score_diff >= thresholds) & (period <= 4)

        # Keep non-garbage plays
        filtered = plays_df[~is_garbage]
        removed_count = len(plays_df) - len(filtered)

        if removed_count > 0:
            logger.debug(f"Filtered {removed_count} garbage time plays")

        return filtered
