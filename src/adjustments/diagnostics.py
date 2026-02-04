"""Adjustment stack diagnostics for detecting correlated adjustment outliers (P2.11).

This module identifies games where multiple correlated adjustments stack
(HFA + travel + altitude), evaluates error patterns, and optionally applies
soft caps to prevent outliers.

The concern: HFA, travel, and altitude adjustments are correlated because they
all favor the home team in the same scenarios. A sea-level team traveling
cross-country to a high-altitude venue gets hit by all three simultaneously,
which may over-penalize them if effects are partially captured in each other.

Typical ranges:
- HFA: 1.5-4.5 pts (curated values + trajectory)
- Travel: 0-3 pts (up to 5 for Hawaii: 3 tz × 0.5 + 1.0 distance + 2.0 Hawaii special)
- Altitude: 0-3 pts (Wyoming 3.0, Air Force 2.5, Colorado 2.0, etc.)

Combined "correlated stack" (HFA + travel + altitude):
- Normal: 2-5 pts
- High: 5-7 pts
- Extreme: 7+ pts (rare, should be investigated)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AdjustmentStack:
    """Breakdown of stacked adjustments for a game."""

    home_team: str
    away_team: str

    # Correlated adjustments (all favor home in same scenarios)
    hfa: float = 0.0
    travel: float = 0.0
    altitude: float = 0.0

    # Other adjustments (less correlated)
    situational: float = 0.0
    special_teams: float = 0.0
    finishing_drives: float = 0.0
    early_down: float = 0.0
    luck: float = 0.0
    fcs: float = 0.0
    pace: float = 0.0
    qb: float = 0.0

    # Computed properties
    @property
    def correlated_stack(self) -> float:
        """Sum of correlated adjustments (HFA + travel + altitude)."""
        return self.hfa + self.travel + self.altitude

    @property
    def total_adjustment(self) -> float:
        """Sum of all adjustments."""
        return (
            self.hfa
            + self.travel
            + self.altitude
            + self.situational
            + self.special_teams
            + self.finishing_drives
            + self.early_down
            + self.luck
            + self.fcs
            + self.pace
            + self.qb
        )

    @property
    def is_high_stack(self) -> bool:
        """Check if correlated stack is unusually high (>5 pts)."""
        return abs(self.correlated_stack) > 5.0

    @property
    def is_extreme_stack(self) -> bool:
        """Check if correlated stack is extreme (>7 pts)."""
        return abs(self.correlated_stack) > 7.0

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "hfa": self.hfa,
            "travel": self.travel,
            "altitude": self.altitude,
            "correlated_stack": self.correlated_stack,
            "situational": self.situational,
            "special_teams": self.special_teams,
            "finishing_drives": self.finishing_drives,
            "early_down": self.early_down,
            "luck": self.luck,
            "fcs": self.fcs,
            "pace": self.pace,
            "qb": self.qb,
            "total_adjustment": self.total_adjustment,
            "is_high_stack": self.is_high_stack,
            "is_extreme_stack": self.is_extreme_stack,
        }


# Default thresholds for stack warnings
HIGH_STACK_THRESHOLD = 5.0  # pts
EXTREME_STACK_THRESHOLD = 7.0  # pts

# Soft cap configuration
# When correlated stack exceeds cap_start, scale down the excess by cap_factor
DEFAULT_CAP_START = 6.0  # Start capping above 6 pts
DEFAULT_CAP_FACTOR = 0.5  # Reduce excess by 50%


def calculate_soft_cap(
    correlated_stack: float,
    cap_start: float = DEFAULT_CAP_START,
    cap_factor: float = DEFAULT_CAP_FACTOR,
) -> float:
    """Apply soft cap to correlated stack to reduce outliers.

    Uses a piecewise linear function:
    - Below cap_start: no change
    - Above cap_start: excess is multiplied by cap_factor

    Example with cap_start=6, cap_factor=0.5:
    - Stack of 5 → 5 (no change)
    - Stack of 7 → 6 + (7-6)*0.5 = 6.5
    - Stack of 9 → 6 + (9-6)*0.5 = 7.5

    Args:
        correlated_stack: The correlated adjustment stack (HFA + travel + altitude)
        cap_start: Point value where capping begins
        cap_factor: Factor to multiply excess by (0.5 = 50% reduction)

    Returns:
        Capped correlated stack value
    """
    if abs(correlated_stack) <= cap_start:
        return correlated_stack

    sign = 1 if correlated_stack > 0 else -1
    abs_stack = abs(correlated_stack)
    excess = abs_stack - cap_start
    capped_excess = excess * cap_factor

    return sign * (cap_start + capped_excess)


def calculate_stack_adjustment(
    original_stack: float,
    cap_start: float = DEFAULT_CAP_START,
    cap_factor: float = DEFAULT_CAP_FACTOR,
) -> float:
    """Calculate the adjustment needed to apply soft cap.

    Returns the delta to add to the original spread to apply the cap.
    This is (capped_value - original_value).

    Args:
        original_stack: Original correlated stack value
        cap_start: Point value where capping begins
        cap_factor: Factor to multiply excess by

    Returns:
        Adjustment to add to spread (negative if capping reduced the stack)
    """
    capped = calculate_soft_cap(original_stack, cap_start, cap_factor)
    return capped - original_stack


class AdjustmentStackDiagnostics:
    """Diagnostics for adjustment stacking patterns.

    Tracks games with high correlated adjustment stacks and can evaluate
    whether these games have systematic prediction errors.
    """

    def __init__(
        self,
        cap_enabled: bool = False,
        cap_start: float = DEFAULT_CAP_START,
        cap_factor: float = DEFAULT_CAP_FACTOR,
    ):
        """Initialize diagnostics.

        Args:
            cap_enabled: Whether to recommend/apply soft caps
            cap_start: Point value where capping begins
            cap_factor: Factor to multiply excess by
        """
        self.cap_enabled = cap_enabled
        self.cap_start = cap_start
        self.cap_factor = cap_factor

        # Track all games for analysis
        self.stacks: list[AdjustmentStack] = []

    def add_game(self, stack: AdjustmentStack) -> None:
        """Add a game's adjustment stack to diagnostics."""
        self.stacks.append(stack)

    def get_high_stack_games(self) -> list[AdjustmentStack]:
        """Get games with high correlated stacks (>5 pts)."""
        return [s for s in self.stacks if s.is_high_stack]

    def get_extreme_stack_games(self) -> list[AdjustmentStack]:
        """Get games with extreme correlated stacks (>7 pts)."""
        return [s for s in self.stacks if s.is_extreme_stack]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all stacks to DataFrame for analysis."""
        if not self.stacks:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self.stacks])

    def get_stack_distribution(self) -> dict:
        """Get distribution statistics for correlated stacks."""
        if not self.stacks:
            return {}

        stacks = [s.correlated_stack for s in self.stacks]
        df = pd.Series(stacks)

        return {
            "count": len(stacks),
            "mean": df.mean(),
            "std": df.std(),
            "min": df.min(),
            "max": df.max(),
            "median": df.median(),
            "pct_high": sum(1 for s in stacks if abs(s) > HIGH_STACK_THRESHOLD)
            / len(stacks)
            * 100,
            "pct_extreme": sum(1 for s in stacks if abs(s) > EXTREME_STACK_THRESHOLD)
            / len(stacks)
            * 100,
        }

    def log_summary(self) -> None:
        """Log summary of adjustment stack patterns."""
        if not self.stacks:
            logger.info("No adjustment stacks recorded")
            return

        stats = self.get_stack_distribution()
        high_games = self.get_high_stack_games()
        extreme_games = self.get_extreme_stack_games()

        logger.info(
            f"Adjustment stack summary: {stats['count']} games, "
            f"mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
            f"range=[{stats['min']:.1f}, {stats['max']:.1f}]"
        )

        if high_games:
            logger.info(
                f"High stack games (>5 pts): {len(high_games)} "
                f"({stats['pct_high']:.1f}%)"
            )
            # Log top 5 extreme cases
            sorted_high = sorted(
                high_games, key=lambda s: abs(s.correlated_stack), reverse=True
            )[:5]
            for s in sorted_high:
                logger.info(
                    f"  {s.away_team} @ {s.home_team}: "
                    f"stack={s.correlated_stack:.1f} "
                    f"(HFA={s.hfa:.1f}, travel={s.travel:.1f}, alt={s.altitude:.1f})"
                )

        if extreme_games:
            logger.warning(
                f"Extreme stack games (>7 pts): {len(extreme_games)} "
                f"({stats['pct_extreme']:.1f}%) - consider investigating"
            )

    def evaluate_errors(
        self,
        predictions_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ) -> dict:
        """Evaluate prediction errors for high-stack vs low-stack games.

        This helps determine if stacking creates systematic bias.

        Args:
            predictions_df: DataFrame with predictions (must have home_team, away_team, spread)
            results_df: DataFrame with actual results (must have home_team, away_team, home_points, away_points)

        Returns:
            Dict with error analysis comparing high-stack vs low-stack games
        """
        if not self.stacks:
            return {"error": "No stacks recorded"}

        # Build lookup for stacks
        stack_lookup = {
            (s.home_team, s.away_team): s.correlated_stack for s in self.stacks
        }

        # Merge predictions with results
        errors = []
        for _, pred in predictions_df.iterrows():
            home = pred.get("home_team")
            away = pred.get("away_team")
            spread = pred.get("spread_raw", pred.get("spread"))

            # Find matching result
            result = results_df[
                (results_df["home_team"] == home) & (results_df["away_team"] == away)
            ]
            if result.empty:
                continue

            actual_margin = (
                result.iloc[0]["home_points"] - result.iloc[0]["away_points"]
            )
            error = spread - actual_margin
            stack = stack_lookup.get((home, away), 0)

            errors.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "stack": stack,
                    "spread": spread,
                    "actual": actual_margin,
                    "error": error,
                    "abs_error": abs(error),
                    "is_high_stack": abs(stack) > HIGH_STACK_THRESHOLD,
                }
            )

        if not errors:
            return {"error": "No matching predictions/results"}

        df = pd.DataFrame(errors)

        # Compare high-stack vs low-stack
        high_mask = df["is_high_stack"]
        low_mask = ~high_mask

        high_errors = df[high_mask]
        low_errors = df[low_mask]

        result = {
            "total_games": len(df),
            "overall_mae": df["abs_error"].mean(),
            "overall_me": df["error"].mean(),
        }

        if len(high_errors) > 0:
            result["high_stack"] = {
                "count": len(high_errors),
                "mae": high_errors["abs_error"].mean(),
                "me": high_errors["error"].mean(),  # Positive = over-predicting home
                "avg_stack": high_errors["stack"].mean(),
            }

        if len(low_errors) > 0:
            result["low_stack"] = {
                "count": len(low_errors),
                "mae": low_errors["abs_error"].mean(),
                "me": low_errors["error"].mean(),
            }

        # Check for systematic bias
        if "high_stack" in result and result["high_stack"]["count"] >= 10:
            me_high = result["high_stack"]["me"]
            if me_high > 2.0:
                result["warning"] = (
                    f"High-stack games show positive mean error ({me_high:.1f}), "
                    "suggesting over-prediction of home advantage. Consider capping."
                )
            elif me_high < -2.0:
                result["warning"] = (
                    f"High-stack games show negative mean error ({me_high:.1f}), "
                    "suggesting under-prediction of home advantage."
                )

        return result

    def log_error_analysis(
        self,
        predictions_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ) -> None:
        """Log error analysis for high-stack vs low-stack games."""
        analysis = self.evaluate_errors(predictions_df, results_df)

        if "error" in analysis:
            logger.warning(f"Stack error analysis failed: {analysis['error']}")
            return

        logger.info(
            f"Stack error analysis: {analysis['total_games']} games, "
            f"overall MAE={analysis['overall_mae']:.2f}, ME={analysis['overall_me']:.2f}"
        )

        if "high_stack" in analysis:
            hs = analysis["high_stack"]
            logger.info(
                f"  High-stack ({hs['count']} games, avg stack={hs['avg_stack']:.1f}): "
                f"MAE={hs['mae']:.2f}, ME={hs['me']:+.2f}"
            )

        if "low_stack" in analysis:
            ls = analysis["low_stack"]
            logger.info(f"  Low-stack ({ls['count']} games): MAE={ls['mae']:.2f}, ME={ls['me']:+.2f}")

        if "warning" in analysis:
            logger.warning(f"  ⚠️  {analysis['warning']}")


def extract_stack_from_prediction(prediction) -> AdjustmentStack:
    """Extract adjustment stack from a PredictedSpread object.

    Args:
        prediction: PredictedSpread object with components

    Returns:
        AdjustmentStack with all adjustments populated
    """
    c = prediction.components
    return AdjustmentStack(
        home_team=prediction.home_team,
        away_team=prediction.away_team,
        hfa=c.home_field,
        travel=c.travel,
        altitude=c.altitude,
        situational=c.situational,
        special_teams=c.special_teams,
        finishing_drives=c.finishing_drives,
        early_down=c.early_down,
        luck=c.luck_adjustment,
        fcs=c.fcs_adjustment,
        pace=c.pace_adjustment,
        qb=c.qb_adjustment,
    )
