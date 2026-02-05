"""Adjustment aggregator with consolidated four-bucket smoothing.

This module consolidates all adjustment smoothing into a single location to prevent
double-counting when correlated factors stack on the same team.

Four-Bucket Architecture:
------------------------
Bucket A (Venue): HFA - raw value
Bucket B (Physical/Fatigue): travel, altitude, consecutive_road, negative rest
    - Aggressive smoothing: largest at 100%, all others at 25%
    - Travel-consecutive correlation: when travel > 1.5, reduce consecutive_road by 50%
Bucket C (Mental/Focus): letdown, lookahead, sandwich
    - Standard smoothing: largest at 100%, second at 50%, others at 25%
Bucket D (Boosts): rivalry_boost, positive rest
    - Linear sum (no dampening for positive factors)

Venue-Physical Integration (Soft Cap):
--------------------------------------
Linear sum for standard games, soft cap for extreme stacks only.
This preserves full HFA+Travel for most matchups while dampening only the
extreme "super stacks" (>6.0 pts combined) that cause MAE regression.

    - If |venue + physical| <= 6.0: Use linear sum (no damping)
    - If |venue + physical| > 6.0: Apply soft cap
        integrated = (6.0 + excess * 0.5) * sign(raw_stack)

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
    venue_bucket: float = 0.0  # Bucket A: HFA (raw)
    physical_bucket: float = 0.0  # Bucket B: travel + altitude + consecutive_road + negative rest
    integrated_env: float = 0.0  # Venue + Physical after integration smoothing
    mental_bucket: float = 0.0  # Bucket C: letdown + lookahead + sandwich
    boosts_bucket: float = 0.0  # Bucket D: rivalry + positive rest

    # Individual smoothed values (for detailed diagnostics)
    smoothed_hfa: float = 0.0
    smoothed_travel: float = 0.0
    smoothed_altitude: float = 0.0
    smoothed_consecutive_road: float = 0.0
    smoothed_rest: float = 0.0
    smoothed_letdown: float = 0.0
    smoothed_lookahead: float = 0.0
    smoothed_sandwich: float = 0.0
    smoothed_rivalry: float = 0.0

    # Pre-smoothing raw values (for debugging)
    raw_total: float = 0.0
    venue_physical_aligned: bool = False  # True if venue and physical had same sign
    was_capped: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "net_adjustment": self.net_adjustment,
            "venue_bucket": self.venue_bucket,
            "physical_bucket": self.physical_bucket,
            "integrated_env": self.integrated_env,
            "mental_bucket": self.mental_bucket,
            "boosts_bucket": self.boosts_bucket,
            "smoothed_hfa": self.smoothed_hfa,
            "smoothed_travel": self.smoothed_travel,
            "smoothed_altitude": self.smoothed_altitude,
            "venue_physical_aligned": self.venue_physical_aligned,
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
    TRAVEL_CONSECUTIVE_THRESHOLD: float = 1.5  # Travel above this reduces consecutive_road
    PHYSICAL_SECONDARY_WEIGHT: float = 0.25  # Weight for non-largest physical factors
    MENTAL_SECOND_WEIGHT: float = 0.50  # Weight for second-largest mental factor
    MENTAL_OTHER_WEIGHT: float = 0.25  # Weight for remaining mental factors
    VENUE_PHYSICAL_SOFT_CAP: float = 6.0  # Threshold where soft cap kicks in for venue+physical stack
    VENUE_PHYSICAL_EXCESS_WEIGHT: float = 0.50  # Weight for excess above soft cap

    def __init__(
        self,
        global_cap: Optional[float] = None,
        travel_consecutive_threshold: Optional[float] = None,
        physical_secondary_weight: Optional[float] = None,
        mental_second_weight: Optional[float] = None,
        mental_other_weight: Optional[float] = None,
        venue_physical_soft_cap: Optional[float] = None,
        venue_physical_excess_weight: Optional[float] = None,
    ):
        """Initialize the aggregator with smoothing parameters.

        Args:
            global_cap: Maximum total adjustment (default: 7.0)
            travel_consecutive_threshold: Travel penalty above which consecutive_road is reduced
            physical_secondary_weight: Weight for non-largest physical factors
            mental_second_weight: Weight for second-largest mental factor
            mental_other_weight: Weight for remaining mental factors
            venue_physical_soft_cap: Threshold where soft cap kicks in (default: 4.5)
            venue_physical_excess_weight: Weight for excess above soft cap (default: 0.50)
        """
        self.global_cap = global_cap if global_cap is not None else self.GLOBAL_CAP
        self.travel_consecutive_threshold = (
            travel_consecutive_threshold
            if travel_consecutive_threshold is not None
            else self.TRAVEL_CONSECUTIVE_THRESHOLD
        )
        self.physical_secondary_weight = (
            physical_secondary_weight
            if physical_secondary_weight is not None
            else self.PHYSICAL_SECONDARY_WEIGHT
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
        self.venue_physical_soft_cap = (
            venue_physical_soft_cap
            if venue_physical_soft_cap is not None
            else self.VENUE_PHYSICAL_SOFT_CAP
        )
        self.venue_physical_excess_weight = (
            venue_physical_excess_weight
            if venue_physical_excess_weight is not None
            else self.VENUE_PHYSICAL_EXCESS_WEIGHT
        )

    def aggregate(
        self,
        raw_hfa: float,
        travel_breakdown: TravelBreakdown,
        home_factors: SituationalFactors,
        away_factors: SituationalFactors,
    ) -> AggregatedAdjustments:
        """Aggregate all adjustments with four-bucket smoothing.

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
        # BUCKET A: Venue (No Smoothing)
        # HFA stands alone - it's the baseline home advantage
        # =====================================================================
        result.venue_bucket = raw_hfa
        result.smoothed_hfa = raw_hfa

        # =====================================================================
        # BUCKET B: Physical/Fatigue (Aggressive Smoothing)
        # Components: travel, altitude, consecutive_road (away), negative rest (home)
        # All factors that cause physical fatigue - highly correlated
        # =====================================================================
        physical_factors = []

        # Travel penalty (favors home, so positive contribution)
        travel_penalty = abs(travel_breakdown.travel_penalty)
        if travel_penalty > 0:
            physical_factors.append(("travel", travel_penalty))

        # Altitude penalty (favors home)
        altitude_penalty = abs(travel_breakdown.altitude_penalty)
        if altitude_penalty > 0:
            physical_factors.append(("altitude", altitude_penalty))

        # Consecutive road penalty for away team (favors home)
        consecutive_away = abs(away_factors.consecutive_road_penalty)
        # Apply travel-consecutive correlation: reduce if travel is significant
        if travel_penalty > self.travel_consecutive_threshold and consecutive_away > 0:
            consecutive_away *= 0.5
            logger.debug(
                f"Travel-consecutive correlation: reduced consecutive_road from "
                f"{abs(away_factors.consecutive_road_penalty):.2f} to {consecutive_away:.2f}"
            )
        if consecutive_away > 0:
            physical_factors.append(("consecutive_away", consecutive_away))

        # Note: Home team consecutive road is rare (they're home), but handle if present
        consecutive_home = abs(home_factors.consecutive_road_penalty)
        if consecutive_home > 0:
            physical_factors.append(("consecutive_home", -consecutive_home))  # Hurts home

        # Short week penalty (negative rest_advantage means home is disadvantaged)
        # Positive rest_advantage goes to boosts bucket
        # Keep the negative sign - this hurts home team
        if home_factors.rest_advantage < 0:
            physical_factors.append(("short_week", home_factors.rest_advantage))  # Negative, hurts home

        # Apply physical smoothing: largest at 100%, others at 25%
        result.physical_bucket = self._smooth_physical(physical_factors)

        # Store individual smoothed values for diagnostics
        for name, _ in physical_factors:
            if name == "travel":
                result.smoothed_travel = travel_penalty
            elif name == "altitude":
                result.smoothed_altitude = altitude_penalty
            elif name == "consecutive_away":
                result.smoothed_consecutive_road = consecutive_away

        # =====================================================================
        # BUCKET C: Mental/Focus (Standard Smoothing)
        # Components: letdown, lookahead, sandwich
        # Net = (away penalties) - (home penalties) since penalties hurt the team
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

        # Store smoothed values (use away team's values since they're typically the affected team)
        for name, val in away_mental:
            if name == "letdown":
                result.smoothed_letdown = val
            elif name == "lookahead":
                result.smoothed_lookahead = val
            elif name == "sandwich":
                result.smoothed_sandwich = val

        # =====================================================================
        # BUCKET D: Boosts (Linear Sum)
        # Components: rivalry_boost, positive rest_advantage (bye week)
        # Rare positive factors - no dampening
        # =====================================================================
        result.boosts_bucket = 0.0

        # Home team boosts (favor home)
        if home_factors.rivalry_boost > 0:
            result.boosts_bucket += home_factors.rivalry_boost
            result.smoothed_rivalry = home_factors.rivalry_boost
        if home_factors.rest_advantage > 0:
            result.boosts_bucket += home_factors.rest_advantage
            result.smoothed_rest = home_factors.rest_advantage

        # Away team boosts (hurt home, i.e., favor away)
        if away_factors.rivalry_boost > 0:
            result.boosts_bucket -= away_factors.rivalry_boost
        # Note: away rest_advantage is typically 0 since rest_advantage is
        # calculated as home - away differential and assigned to home

        # =====================================================================
        # VENUE-PHYSICAL INTEGRATION (Soft Cap)
        # Linear sum for standard games, soft cap for extreme stacks.
        # This preserves full HFA+Travel for most games while dampening only
        # the extreme "super stacks" (>4.5 pts) that cause MAE regression.
        # =====================================================================
        venue = result.venue_bucket
        physical = result.physical_bucket

        # Sum linearly first
        raw_stack = venue + physical

        # Check if stack exceeds soft cap threshold
        if abs(raw_stack) <= self.venue_physical_soft_cap:
            # Standard game - use linear sum as-is
            result.integrated_env = raw_stack
            result.venue_physical_aligned = False  # No damping applied
        else:
            # Extreme stack - apply soft cap
            # integrated = (cap + excess * weight) * sign
            sign = 1 if raw_stack > 0 else -1
            excess = abs(raw_stack) - self.venue_physical_soft_cap
            result.integrated_env = (
                self.venue_physical_soft_cap + excess * self.venue_physical_excess_weight
            ) * sign
            result.venue_physical_aligned = True  # Damping was applied

            logger.debug(
                f"Venue-Physical soft cap: raw_stack={raw_stack:.2f} "
                f"-> integrated={result.integrated_env:.2f} "
                f"(cap={self.venue_physical_soft_cap}, excess={excess:.2f})"
            )

        # =====================================================================
        # FINAL CALCULATION
        # net = integrated_env + mental + boosts
        # =====================================================================
        raw_total = (
            result.integrated_env
            + result.mental_bucket
            + result.boosts_bucket
        )
        result.raw_total = raw_total

        # Apply global cap
        if abs(raw_total) > self.global_cap:
            result.net_adjustment = max(-self.global_cap, min(self.global_cap, raw_total))
            result.was_capped = True
            logger.debug(
                f"Global cap applied: {raw_total:.2f} -> {result.net_adjustment:.2f}"
            )
        else:
            result.net_adjustment = raw_total

        return result

    def _smooth_physical(self, factors: list[tuple[str, float]]) -> float:
        """Apply aggressive smoothing to physical factors.

        Factors that favor home (positive) and hurt home (negative) are smoothed
        separately within each group, then summed. This ensures correlated factors
        are smoothed together while opposing factors sum linearly.

        Within each group: Largest at 100%, all others at 25%.

        Args:
            factors: List of (name, value) tuples (positive = favors home, negative = hurts home)

        Returns:
            Smoothed sum of physical factors
        """
        if not factors:
            return 0.0

        # Separate positive (favor home) and negative (hurt home) factors
        positive_factors = [(n, v) for n, v in factors if v > 0]
        negative_factors = [(n, v) for n, v in factors if v < 0]

        # Smooth positive factors (favor home)
        positive_smoothed = 0.0
        if positive_factors:
            sorted_pos = sorted(positive_factors, key=lambda x: x[1], reverse=True)
            positive_smoothed = sorted_pos[0][1]  # Largest at 100%
            for _, value in sorted_pos[1:]:
                positive_smoothed += value * self.physical_secondary_weight

        # Smooth negative factors (hurt home)
        negative_smoothed = 0.0
        if negative_factors:
            sorted_neg = sorted(negative_factors, key=lambda x: x[1])  # Most negative first
            negative_smoothed = sorted_neg[0][1]  # Largest magnitude at 100%
            for _, value in sorted_neg[1:]:
                negative_smoothed += value * self.physical_secondary_weight

        # Sum the two groups (opposing factors sum linearly)
        return positive_smoothed + negative_smoothed

    def _smooth_mental(self, factors: list[tuple[str, float]]) -> float:
        """Apply standard smoothing to mental factors.

        Largest at 100%, second at 50%, others at 25%.

        Args:
            factors: List of (name, value) tuples

        Returns:
            Smoothed sum of mental factors
        """
        if not factors:
            return 0.0

        # Sort by absolute value descending
        sorted_factors = sorted(factors, key=lambda x: abs(x[1]), reverse=True)

        # Largest at 100%
        smoothed = sorted_factors[0][1]

        # Second at 50%
        if len(sorted_factors) > 1:
            smoothed += sorted_factors[1][1] * self.mental_second_weight

        # Others at 25%
        for _, value in sorted_factors[2:]:
            smoothed += value * self.mental_other_weight

        return smoothed
