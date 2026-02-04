"""Early-down success rate model.

Isolates 1st and 2nd down performance as a stable predictor of future outcomes.
Early-down success rate is more predictive than 3rd down conversion rate because:
- Larger sample size (more 1st/2nd down plays than 3rd down)
- Less variance (3rd down outcomes are heavily influenced by distance)
- Better proxy for sustainable offensive/defensive quality
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.play_types import SCRIMMAGE_PLAY_TYPES
from config.settings import get_settings

logger = logging.getLogger(__name__)


# SCRIMMAGE_PLAY_TYPES imported from config.play_types (single source of truth)


@dataclass
class EarlyDownRating:
    """Container for team early-down success rate ratings."""

    team: str
    off_success_rate: float  # Offensive early-down SR
    def_success_rate: float  # Defensive early-down SR (opponent's SR against this team)
    off_plays: int  # Sample size (offensive)
    def_plays: int  # Sample size (defensive)
    overall_rating: float  # Points above/below average


class EarlyDownModel:
    """
    Model for evaluating early-down (1st & 2nd down) success rate.

    Success definitions:
    - 1st down: gain >= 50% of distance (e.g., 5+ yards on 1st & 10)
    - 2nd down: gain >= 70% of distance (e.g., 4+ yards on 2nd & 6)

    Early-down success rate is one of the most stable and predictive
    metrics in college football analytics.
    """

    EXPECTED_SUCCESS_RATE = 0.42  # FBS average early-down SR

    # Points per 1% of success rate above/below average.
    # Empirically, a 10% SR advantage ~ 4-5 points per game.
    POINTS_PER_SR_PCT = 0.45

    # Plays needed for full confidence in observed rate
    FULL_WEIGHT_PLAYS = 200

    def __init__(self):
        """Initialize the early-down model."""
        self.team_ratings: dict[str, EarlyDownRating] = {}

    def calculate_all_teams(self, plays_df: pd.DataFrame) -> dict[str, EarlyDownRating]:
        """Calculate early-down ratings for all teams using vectorized operations.

        Args:
            plays_df: DataFrame with play-by-play data (columns: down, distance,
                      yards_gained, offense, defense, play_type, period,
                      offense_score, defense_score)

        Returns:
            Dictionary mapping team names to EarlyDownRating
        """
        if plays_df.empty:
            return self.team_ratings

        # Vectorized filtering: early downs + scrimmage plays
        df = plays_df[
            plays_df["down"].isin([1, 2])
            & plays_df["play_type"].isin(SCRIMMAGE_PLAY_TYPES)
        ]

        if df.empty:
            return self.team_ratings

        # Vectorized garbage time filter (use Settings as source of truth)
        settings = get_settings()
        gt_thresholds = np.array([
            settings.garbage_time_q1,
            settings.garbage_time_q2,
            settings.garbage_time_q3,
            settings.garbage_time_q4,
        ])
        score_diff = (df["offense_score"] - df["defense_score"]).abs()
        period = df["period"].values
        # Map period to threshold; periods > 4 get threshold=999 (never garbage)
        gt_thresh = np.where(
            (period >= 1) & (period <= 4),
            gt_thresholds[np.clip(period - 1, 0, 3)],
            999,
        )
        df = df[score_diff.values < gt_thresh]

        if df.empty:
            return self.team_ratings

        # Vectorized success calculation
        down = df["down"].values
        distance = df["distance"].values
        yards = df["yards_gained"].values
        success = np.where(
            down == 1,
            yards >= 0.5 * distance,
            yards >= 0.7 * distance,
        )

        # Add success column for groupby
        df = df.assign(success=success)

        # Offensive stats: group by offense team
        off_stats = df.groupby("offense").agg(
            off_plays=("success", "count"),
            off_successes=("success", "sum"),
        )
        off_stats["off_sr"] = off_stats["off_successes"] / off_stats["off_plays"]

        # Defensive stats: group by defense team
        def_stats = df.groupby("defense").agg(
            def_plays=("success", "count"),
            def_successes=("success", "sum"),
        )
        def_stats["def_sr"] = def_stats["def_successes"] / def_stats["def_plays"]

        # Build ratings for all teams
        all_teams = set(off_stats.index) | set(def_stats.index)
        expected = self.EXPECTED_SUCCESS_RATE
        fw = self.FULL_WEIGHT_PLAYS

        for team in all_teams:
            if team in off_stats.index:
                off_sr_raw = off_stats.at[team, "off_sr"]
                off_n = int(off_stats.at[team, "off_plays"])
            else:
                off_sr_raw = expected
                off_n = 0

            if team in def_stats.index:
                def_sr_raw = def_stats.at[team, "def_sr"]
                def_n = int(def_stats.at[team, "def_plays"])
            else:
                def_sr_raw = expected
                def_n = 0

            # Regress toward mean
            off_w = min(off_n / fw, 1.0)
            def_w = min(def_n / fw, 1.0)
            off_sr = off_w * off_sr_raw + (1 - off_w) * expected
            def_sr = def_w * def_sr_raw + (1 - def_w) * expected

            # Convert to points
            off_edge = (off_sr - expected) * 100 * self.POINTS_PER_SR_PCT
            def_edge = (expected - def_sr) * 100 * self.POINTS_PER_SR_PCT
            overall = off_edge + def_edge

            self.team_ratings[team] = EarlyDownRating(
                team=team,
                off_success_rate=off_sr,
                def_success_rate=def_sr,
                off_plays=off_n,
                def_plays=def_n,
                overall_rating=overall,
            )

        logger.info(
            f"Calculated early-down ratings for {len(self.team_ratings)} teams"
        )
        return self.team_ratings

    def get_matchup_differential(self, team_a: str, team_b: str) -> float:
        """Get early-down point differential in a matchup.

        Args:
            team_a: First team (home)
            team_b: Second team (away)

        Returns:
            Expected point advantage for team_a from early-down efficiency
        """
        rating_a = self.team_ratings.get(team_a)
        rating_b = self.team_ratings.get(team_b)

        overall_a = rating_a.overall_rating if rating_a else 0.0
        overall_b = rating_b.overall_rating if rating_b else 0.0

        return overall_a - overall_b

    def get_summary_df(self) -> pd.DataFrame:
        """Get summary of all team ratings as a DataFrame.

        Returns:
            DataFrame with early-down ratings for all teams
        """
        if not self.team_ratings:
            return pd.DataFrame()

        data = [
            {
                "team": r.team,
                "off_sr": round(r.off_success_rate, 3),
                "def_sr": round(r.def_success_rate, 3),
                "off_plays": r.off_plays,
                "def_plays": r.def_plays,
                "overall": round(r.overall_rating, 2),
            }
            for r in self.team_ratings.values()
        ]

        df = pd.DataFrame(data)
        return df.sort_values("overall", ascending=False).reset_index(drop=True)
