"""Adjustment aggregator with unified environmental stack smoothing.

This module consolidates all adjustment smoothing into a single location to prevent
double-counting when correlated factors stack on the same team.

Architecture:
-------------
1. Environmental Stack (Single-Layer Soft Cap):
   - Raw inputs: HFA + Travel + Altitude + Rest + Consecutive Road
   - Linear sum first, then apply soft cap for extreme stacks only
   - Threshold: 5.0 pts, Excess weight: 60%
   - This replicates the original smooth_correlated_stack() behavior

2. Mental Bucket (Standard Smoothing):
   - letdown, lookahead, sandwich penalties
   - Smoothing: largest at 100%, second at 50%, others at 25%

3. Boosts Bucket (Linear Sum):
   - rivalry_boost, positive rest (bye week advantage)
   - No dampening for positive factors

Final: total = env_score + mental + boosts, capped at ±7.0

Global Cap: ±7.0 points to prevent unrealistic adjustments.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.adjustments.situational import SituationalFactors

logger = logging.getLogger(__name__)


@dataclass
class TravelBreakdown:
    """Container for raw travel-related adjustments."""

    travel_penalty: float = 0.0  # Distance + timezone penalty (negative value)
    altitude_penalty: float = 0.0  # Altitude penalty (negative value)


@dataclass
class AggregatedAdjustments:
    """Result of adjustment aggregation with bucket breakdown."""

    # Final values after smoothing
    net_adjustment: float = 0.0  # Total adjustment favoring home team

    # Bucket breakdowns (for diagnostics)
    raw_env_stack: float = 0.0  # Raw linear sum: HFA + travel + altitude + rest + consecutive_road
    env_score: float = 0.0  # Environmental stack after soft cap
    mental_bucket: float = 0.0  # letdown + lookahead + sandwich (smoothed)
    boosts_bucket: float = 0.0  # rivalry + positive rest (linear)

    # Individual raw values (for detailed diagnostics)
    raw_hfa: float = 0.0
    raw_travel: float = 0.0
    raw_altitude: float = 0.0
    raw_rest: float = 0.0
    raw_consecutive_road: float = 0.0

    # Pre-cap raw total (for debugging)
    raw_total: float = 0.0
    env_was_capped: bool = False  # True if soft cap was applied to env stack
    was_capped: bool = False  # True if global cap was applied

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "net_adjustment": self.net_adjustment,
            "raw_env_stack": self.raw_env_stack,
            "env_score": self.env_score,
            "mental_bucket": self.mental_bucket,
            "boosts_bucket": self.boosts_bucket,
            "raw_hfa": self.raw_hfa,
            "raw_travel": self.raw_travel,
            "raw_altitude": self.raw_altitude,
            "env_was_capped": self.env_was_capped,
            "raw_total": self.raw_total,
            "was_capped": self.was_capped,
        }


class AdjustmentAggregator:
    """Consolidates all game adjustments with four-bucket smoothing.

    This class is the single point where all adjustments are combined and smoothed.
    It replaces the separate smoothing in:
    - smooth_correlated_stack() in spread_generator.py (HFA + travel + altitude)
    - SituationalFactors.__post_init__() (rest + letdown + lookahead + etc.)

    The four-bucket design groups adjustments by their correlation structure:
    - Venue (HFA) stands alone
    - Physical factors (travel, altitude, consecutive road, short week) are highly correlated
    - Mental factors (letdown, lookahead, sandwich) are moderately correlated
    - Boosts (rivalry, bye week rest) are rare positive factors that stack linearly
    """

    # Smoothing constants
    GLOBAL_CAP: float = 7.0
    ENV_SOFT_CAP: float = 5.0  # Threshold where soft cap kicks in for environmental stack
    ENV_EXCESS_WEIGHT: float = 0.60  # Weight for excess above soft cap (tuned for ~0 mean error)
    MENTAL_SECOND_WEIGHT: float = 0.50  # Weight for second-largest mental factor
    MENTAL_OTHER_WEIGHT: float = 0.25  # Weight for remaining mental factors

    def __init__(
        self,
        global_cap: Optional[float] = None,
        env_soft_cap: Optional[float] = None,
        env_excess_weight: Optional[float] = None,
        mental_second_weight: Optional[float] = None,
        mental_other_weight: Optional[float] = None,
    ):
        """Initialize the aggregator with smoothing parameters.

        Args:
            global_cap: Maximum total adjustment (default: 7.0)
            env_soft_cap: Threshold where soft cap kicks in for env stack (default: 5.0)
            env_excess_weight: Weight for excess above soft cap (default: 0.60)
            mental_second_weight: Weight for second-largest mental factor
            mental_other_weight: Weight for remaining mental factors
        """
        self.global_cap = global_cap if global_cap is not None else self.GLOBAL_CAP
        self.env_soft_cap = (
            env_soft_cap if env_soft_cap is not None else self.ENV_SOFT_CAP
        )
        self.env_excess_weight = (
            env_excess_weight if env_excess_weight is not None else self.ENV_EXCESS_WEIGHT
        )
        self.mental_second_weight = (
            mental_second_weight
            if mental_second_weight is not None
            else self.MENTAL_SECOND_WEIGHT
        )
        self.mental_other_weight = (
            mental_other_weight
            if mental_other_weight is not None
            else self.MENTAL_OTHER_WEIGHT
        )

    def aggregate(
        self,
        raw_hfa: float,
        travel_breakdown: TravelBreakdown,
        home_factors: SituationalFactors,
        away_factors: SituationalFactors,
    ) -> AggregatedAdjustments:
        """Aggregate all adjustments with unified environmental stack smoothing.

        Uses single-layer soft cap on the environmental stack (HFA + travel + altitude
        + rest + consecutive_road) to avoid double-damping. This replicates the original
        smooth_correlated_stack() behavior that achieved MAE 12.21 and ATS 58.9%.

        Args:
            raw_hfa: Raw home field advantage (positive value)
            travel_breakdown: Travel and altitude penalties for away team
            home_factors: Raw situational factors for home team
            away_factors: Raw situational factors for away team

        Returns:
            AggregatedAdjustments with smoothed values and bucket breakdowns
        """
        result = AggregatedAdjustments()

        # =====================================================================
        # STEP 1: Collect Raw Environmental Factors (NO smoothing yet)
        # These all relate to the physical/venue environment of the game
        # =====================================================================

        # HFA (positive = favors home)
        result.raw_hfa = raw_hfa

        # Travel penalty (positive = favors home, away team traveled)
        result.raw_travel = abs(travel_breakdown.travel_penalty)

        # Altitude penalty (positive = favors home, away team at altitude disadvantage)
        result.raw_altitude = abs(travel_breakdown.altitude_penalty)

        # Rest advantage (positive = home has more rest, negative = home has less rest)
        # This includes both bye week advantage AND short week penalty
        result.raw_rest = home_factors.rest_advantage

        # Consecutive road penalty for away team (positive = favors home)
        # Home team consecutive road is rare but would be negative
        consecutive_away = abs(away_factors.consecutive_road_penalty)
        consecutive_home = abs(home_factors.consecutive_road_penalty)
        result.raw_consecutive_road = consecutive_away - consecutive_home

        # =====================================================================
        # STEP 2: Calculate Raw Environmental Stack (Linear Sum)
        # =====================================================================
        result.raw_env_stack = (
            result.raw_hfa
            + result.raw_travel
            + result.raw_altitude
            + result.raw_rest
            + result.raw_consecutive_road
        )

        # =====================================================================
        # STEP 3: Apply Single-Layer Soft Cap to Environmental Stack
        # This is the ONLY smoothing applied to environmental factors
        # =====================================================================
        raw_stack = result.raw_env_stack

        if abs(raw_stack) <= self.env_soft_cap:
            # Standard game - use linear sum as-is (no damping)
            result.env_score = raw_stack
            result.env_was_capped = False
        else:
            # Extreme stack - apply soft cap
            # env_score = (cap + excess * weight) * sign
            sign = 1 if raw_stack > 0 else -1
            excess = abs(raw_stack) - self.env_soft_cap
            result.env_score = (
                self.env_soft_cap + excess * self.env_excess_weight
            ) * sign
            result.env_was_capped = True

            logger.debug(
                f"Env soft cap: raw={raw_stack:.2f} -> capped={result.env_score:.2f} "
                f"(threshold={self.env_soft_cap}, excess={excess:.2f})"
            )

        # =====================================================================
        # STEP 4: Mental Bucket (Standard Smoothing)
        # Components: letdown, lookahead, sandwich
        # Net = (away penalties favor home) - (home penalties hurt home)
        # =====================================================================
        # Away team mental penalties (favor home)
        away_mental = []
        if away_factors.letdown_penalty != 0:
            away_mental.append(("letdown", abs(away_factors.letdown_penalty)))
        if away_factors.lookahead_penalty != 0:
            away_mental.append(("lookahead", abs(away_factors.lookahead_penalty)))
        if away_factors.sandwich_penalty != 0:
            away_mental.append(("sandwich", abs(away_factors.sandwich_penalty)))

        away_mental_smoothed = self._smooth_mental(away_mental)

        # Home team mental penalties (hurt home)
        home_mental = []
        if home_factors.letdown_penalty != 0:
            home_mental.append(("letdown", abs(home_factors.letdown_penalty)))
        if home_factors.lookahead_penalty != 0:
            home_mental.append(("lookahead", abs(home_factors.lookahead_penalty)))
        if home_factors.sandwich_penalty != 0:
            home_mental.append(("sandwich", abs(home_factors.sandwich_penalty)))

        home_mental_smoothed = self._smooth_mental(home_mental)

        # Net mental: away penalties favor home, home penalties hurt home
        result.mental_bucket = away_mental_smoothed - home_mental_smoothed

        # =====================================================================
        # STEP 5: Boosts Bucket (Linear Sum, No Dampening)
        # Components: rivalry_boost only (rest is now in env stack)
        # =====================================================================
        result.boosts_bucket = 0.0

        # Home team rivalry boost (favor home)
        if home_factors.rivalry_boost > 0:
            result.boosts_bucket += home_factors.rivalry_boost

        # Away team rivalry boost (favor away, hurt home)
        if away_factors.rivalry_boost > 0:
            result.boosts_bucket -= away_factors.rivalry_boost

        # =====================================================================
        # STEP 6: Final Calculation
        # total = env_score + mental + boosts
        # =====================================================================
        raw_total = (
            result.env_score
            + result.mental_bucket
            + result.boosts_bucket
        )
        result.raw_total = raw_total

        # Apply global cap (±7.0)
        if abs(raw_total) > self.global_cap:
            result.net_adjustment = max(-self.global_cap, min(self.global_cap, raw_total))
            result.was_capped = True
            logger.debug(
                f"Global cap applied: {raw_total:.2f} -> {result.net_adjustment:.2f}"
            )
        else:
            result.net_adjustment = raw_total

        return result

    def _smooth_mental(self, factors: list[tuple[str, float]]) -> float:
        """Apply standard smoothing to mental factors.

        Largest at 100%, second at 50%, others at 25%.

        Args:
            factors: List of (name, value) tuples (all positive values)

        Returns:
            Smoothed sum of mental factors
        """
        if not factors:
            return 0.0

        # Sort by value descending (largest first)
        sorted_factors = sorted(factors, key=lambda x: x[1], reverse=True)

        # Largest at 100%
        smoothed = sorted_factors[0][1]

        # Second at 50%
        if len(sorted_factors) > 1:
            smoothed += sorted_factors[1][1] * self.mental_second_weight

        # Others at 25%
        for _, value in sorted_factors[2:]:
            smoothed += value * self.mental_other_weight

        return smoothed
