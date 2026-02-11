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

Independence Assumptions:
-------------------------
HFA and Travel are intentionally NOT given interaction effects because they measure
orthogonal phenomena:

- **HFA** measures what the HOME team GAINS: crowd energy, venue familiarity,
  favorable officiating tendencies. LSU gets its 4.0 pt HFA regardless of whether
  the opponent traveled 200 miles or 2000 miles.

- **Travel** measures what the AWAY team LOSES: jet lag, circadian disruption,
  physical fatigue from travel. A 3-timezone trip hurts equally at LSU or Vanderbilt.

- **Altitude** measures oxygen debt for visiting sea-level teams. Independent of
  crowd noise or travel distance per se.

These factors stack additively because they represent genuinely independent mechanisms.
The environmental soft cap (5.0 pts, 60% excess) protects against over-prediction
in extreme scenarios without artificially dampening independent effects.

Interaction Effects (where double-counting IS a risk):
- Travel × Consecutive Road: When travel > TRAVEL_INTERACTION_THRESHOLD (1.5 pts),
  consecutive road reduced by CONSECUTIVE_ROAD_INTERACTION_DAMPENING (50%)
  (both measure physical fatigue from being away from home)
- Travel × Altitude: When travel > TRAVEL_INTERACTION_THRESHOLD (1.5 pts),
  altitude reduced by ALTITUDE_INTERACTION_DAMPENING (30%)
  (stacking physical stressors on already-fatigued team; affects ~5 games/year)
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.adjustments.situational import SituationalFactors

logger = logging.getLogger(__name__)


@dataclass
class TravelBreakdown:
    """Container for raw travel-related adjustments.

    Sign Convention:
        Both fields are **positive magnitudes** representing the advantage
        the home team gains from the away team's travel burden.

        - travel_penalty: How much the away team is disadvantaged by travel
          (distance + timezone). E.g., 1.5 means home gains 1.5 pts.
        - altitude_penalty: How much the away team is disadvantaged by altitude.
          E.g., 2.0 means home gains 2.0 pts from altitude advantage.

        These values come directly from TravelAdjuster.get_total_travel_adjustment()
        and AltitudeAdjuster.get_altitude_adjustment(), both of which return
        positive values favoring the home team.

    Validation:
        The aggregator asserts these are non-negative to catch upstream sign errors.
    """

    travel_penalty: float = 0.0  # Positive magnitude: away team's travel disadvantage
    altitude_penalty: float = 0.0  # Positive magnitude: away team's altitude disadvantage

    def __post_init__(self):
        """Validate sign convention."""
        if self.travel_penalty < 0:
            raise ValueError(
                f"travel_penalty must be >= 0 (positive = home advantage), "
                f"got {self.travel_penalty}. Check TravelAdjuster output."
            )
        if self.altitude_penalty < 0:
            raise ValueError(
                f"altitude_penalty must be >= 0 (positive = home advantage), "
                f"got {self.altitude_penalty}. Check AltitudeAdjuster output."
            )


@dataclass
class AggregatedAdjustments:
    """Result of adjustment aggregation with bucket breakdown."""

    # Final values after smoothing
    net_adjustment: float = 0.0  # Total adjustment favoring home team

    # Bucket breakdowns (for diagnostics)
    raw_venue_stack: float = 0.0  # Raw linear sum: HFA + travel + altitude (venue-only, for smoothing)
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

    # Pro-rata smoothing support (for accurate per-bucket reporting)
    # venue_smoothing_factor: Applied to HFA, travel, altitude only (the correlated venue stack)
    # Rest and consecutive_road are added linearly after venue smoothing
    venue_smoothing_factor: float = 1.0  # venue_score / raw_venue_stack (for HFA/travel/altitude)
    env_smoothing_factor: float = 1.0  # Deprecated: use venue_smoothing_factor for components
    situational_score: float = 0.0  # Pure situational: rest + consec + mental + boosts (no venue smoothing)

    # Diagnostic totals showing smoothing progression:
    # 1. raw_sum_all: Linear sum of ALL raw values (zero smoothing applied)
    # 2. pre_global_cap_total: After env soft cap + mental smoothing (only pre-global-cap)
    # 3. net_adjustment: Final value after global cap
    # Use (raw_sum_all - pre_global_cap_total) to see how much smoothing reduced the stack.
    raw_sum_all: float = 0.0  # True linear sum: raw_env_stack + raw_mental + raw_boosts
    raw_mental_sum: float = 0.0  # Unsmoothed mental penalty sum (for raw_sum_all calc)
    pre_global_cap_total: float = 0.0  # env_score + mental_bucket + boosts (post-smoothing, pre-cap)
    env_was_capped: bool = False  # True if soft cap was applied to env stack
    was_capped: bool = False  # True if global cap was applied

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "net_adjustment": self.net_adjustment,
            "raw_venue_stack": self.raw_venue_stack,
            "raw_env_stack": self.raw_env_stack,
            "env_score": self.env_score,
            "venue_smoothing_factor": self.venue_smoothing_factor,
            "situational_score": self.situational_score,
            "mental_bucket": self.mental_bucket,
            "boosts_bucket": self.boosts_bucket,
            "raw_hfa": self.raw_hfa,
            "raw_travel": self.raw_travel,
            "raw_altitude": self.raw_altitude,
            "env_was_capped": self.env_was_capped,
            "raw_sum_all": self.raw_sum_all,
            "raw_mental_sum": self.raw_mental_sum,
            "pre_global_cap_total": self.pre_global_cap_total,
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

    # Interaction effect constants (prevent double-counting correlated fatigue)
    TRAVEL_INTERACTION_THRESHOLD: float = 1.5  # Travel penalty above which interactions apply
    ALTITUDE_INTERACTION_DAMPENING: float = 0.70  # Altitude reduced to 70% when travel > threshold
    CONSECUTIVE_ROAD_INTERACTION_DAMPENING: float = 0.50  # Consec road reduced to 50% when travel > threshold

    def __init__(
        self,
        global_cap: Optional[float] = None,
        env_soft_cap: Optional[float] = None,
        env_excess_weight: Optional[float] = None,
        mental_second_weight: Optional[float] = None,
        mental_other_weight: Optional[float] = None,
        travel_interaction_threshold: Optional[float] = None,
        altitude_interaction_dampening: Optional[float] = None,
        consecutive_road_interaction_dampening: Optional[float] = None,
    ):
        """Initialize the aggregator with smoothing parameters.

        Args:
            global_cap: Maximum total adjustment (default: 7.0)
            env_soft_cap: Threshold where soft cap kicks in for env stack (default: 5.0)
            env_excess_weight: Weight for excess above soft cap (default: 0.60)
            mental_second_weight: Weight for second-largest mental factor
            mental_other_weight: Weight for remaining mental factors
            travel_interaction_threshold: Travel penalty above which interactions apply (default: 1.5)
            altitude_interaction_dampening: Altitude multiplier when travel > threshold (default: 0.70)
            consecutive_road_interaction_dampening: Consecutive road multiplier when travel > threshold (default: 0.50)
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
        self.travel_interaction_threshold = (
            travel_interaction_threshold
            if travel_interaction_threshold is not None
            else self.TRAVEL_INTERACTION_THRESHOLD
        )
        self.altitude_interaction_dampening = (
            altitude_interaction_dampening
            if altitude_interaction_dampening is not None
            else self.ALTITUDE_INTERACTION_DAMPENING
        )
        self.consecutive_road_interaction_dampening = (
            consecutive_road_interaction_dampening
            if consecutive_road_interaction_dampening is not None
            else self.CONSECUTIVE_ROAD_INTERACTION_DAMPENING
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
        # abs() is a safety belt - TravelBreakdown validates non-negative in __post_init__
        result.raw_travel = abs(travel_breakdown.travel_penalty)

        # Altitude penalty (positive = favors home, away team at altitude disadvantage)
        # abs() is a safety belt - TravelBreakdown validates non-negative in __post_init__
        altitude = abs(travel_breakdown.altitude_penalty)

        # Travel/Altitude Interaction: When travel exceeds threshold, reduce altitude
        # to prevent over-stacking physical stressors (affects ~5 games/year)
        if result.raw_travel > self.travel_interaction_threshold and altitude > 0:
            altitude *= self.altitude_interaction_dampening

        result.raw_altitude = altitude

        # Rest advantage (positive = home has more rest, negative = home has less rest)
        # This includes both bye week advantage AND short week penalty.
        #
        # NOTE: rest_advantage intentionally has NO interaction with travel.
        #
        # Why consecutive_road interacts with travel (50% reduction):
        #   Consecutive road = "you've been traveling for 2+ weeks" (travel fatigue)
        #   Travel penalty = "this trip's fatigue" (also travel fatigue)
        #   → Same mechanism, so we dampen to avoid double-counting.
        #
        # Why altitude interacts with travel (30% reduction):
        #   Altitude = acute physical stress (oxygen debt)
        #   Travel = acute physical stress (jet lag)
        #   → Both are game-day physical stressors, some overlap.
        #
        # Why short-week does NOT interact with travel:
        #   Short-week = incomplete RECOVERY from previous game:
        #     - Fewer practices (preparation)
        #     - Less film study (preparation)
        #     - Not fully healed from injuries (chronic fatigue)
        #     - Mental fatigue from quick turnaround
        #   Travel = acute JOURNEY stress (jet lag, disrupted sleep)
        #
        #   These are orthogonal mechanisms:
        #     - A team on short week AT HOME still has recovery/prep penalties
        #     - A team with normal rest traveling far still has journey fatigue
        #   They stack linearly because they measure different things.
        #   The env soft cap (5.0 pts) handles extreme stacks adequately.
        result.raw_rest = home_factors.rest_advantage

        # Consecutive road penalty: positive magnitude on the penalized team.
        # SituationalFactors stores this as a positive value (e.g., 1.5) on whichever
        # team is playing their 2nd straight road game. The away team having this penalty
        # favors home; the home team having it (rare) hurts home.
        # abs() is a safety belt - SituationalFactors should always store positive values.
        consecutive_away = abs(away_factors.consecutive_road_penalty)
        consecutive_home = abs(home_factors.consecutive_road_penalty)

        # Travel/Consecutive Road Interaction: When travel exceeds threshold, reduce
        # consecutive road to prevent double-counting the fatigue component
        if result.raw_travel > self.travel_interaction_threshold:
            consecutive_away *= self.consecutive_road_interaction_dampening
            consecutive_home *= self.consecutive_road_interaction_dampening

        result.raw_consecutive_road = consecutive_away - consecutive_home

        # =====================================================================
        # STEP 2: Calculate Raw Venue Stack and Raw Environmental Stack
        # Venue stack = HFA + travel + altitude (correlated venue factors, subject to soft cap)
        # Env stack = Venue + rest + consecutive_road (full environmental, for diagnostics)
        # =====================================================================
        result.raw_venue_stack = (
            result.raw_hfa
            + result.raw_travel
            + result.raw_altitude
        )
        result.raw_env_stack = (
            result.raw_venue_stack
            + result.raw_rest
            + result.raw_consecutive_road
        )

        # =====================================================================
        # STEP 3: Apply Single-Layer Soft Cap to VENUE Stack Only
        # Venue = HFA + travel + altitude (the correlated components)
        # Rest and consecutive_road are added linearly AFTER venue smoothing
        #
        # DESIGN DECISION: Soft cap on venue stack, not full env stack
        # ---------------------------------------------------------------
        # The soft cap exists to prevent extreme venue stacking (e.g., altitude
        # game after cross-country flight). Rest and consecutive_road are
        # situational factors that should NOT trigger venue smoothing.
        #
        # Example of the bug this fixes:
        #   HFA=3.5, travel=2.0, altitude=1.5, rest=-2.0 → raw_env=5.0, factor=1.0
        #   HFA=3.5, travel=2.0, altitude=1.5, rest=+0.5 → raw_env=7.5, factor=0.67
        # The rest value changing shouldn't affect HFA/travel/altitude smoothing.
        #
        # Empirical analysis (2023-2025, 1,824 games):
        #   - Venue stack cap trigger (>5.0): ~13% of games
        #   - Negative venue stack (<0): Never occurs (all three are always positive)
        # =====================================================================
        venue_stack = result.raw_venue_stack

        if abs(venue_stack) <= self.env_soft_cap:
            # Standard game - use linear sum as-is (no damping)
            venue_score = venue_stack
            result.env_was_capped = False
        else:
            # Extreme venue stack - apply soft cap
            # venue_score = (cap + excess * weight) * sign
            sign = 1 if venue_stack > 0 else -1
            excess = abs(venue_stack) - self.env_soft_cap
            venue_score = (
                self.env_soft_cap + excess * self.env_excess_weight
            ) * sign
            result.env_was_capped = True

            logger.debug(
                f"Venue soft cap: raw={venue_stack:.2f} -> capped={venue_score:.2f} "
                f"(threshold={self.env_soft_cap}, excess={excess:.2f})"
            )

        # Calculate venue smoothing factor for HFA/travel/altitude allocation
        # This allows SpreadGenerator to report accurate post-smoothing values
        if abs(venue_stack) > 0.001:
            result.venue_smoothing_factor = venue_score / venue_stack
        else:
            result.venue_smoothing_factor = 1.0

        # env_score = smoothed venue + linear rest + linear consecutive_road
        result.env_score = venue_score + result.raw_rest + result.raw_consecutive_road

        # Deprecated: env_smoothing_factor kept for backward compatibility
        # Use venue_smoothing_factor for HFA/travel/altitude components
        result.env_smoothing_factor = result.venue_smoothing_factor

        # =====================================================================
        # STEP 4: Mental Bucket (Standard Smoothing)
        # Components: letdown, lookahead, sandwich, game_shape
        # Net = (away penalties favor home) - (home penalties hurt home)
        #
        # DESIGN DECISION: Per-Team Smoothing Before Netting
        # -------------------------------------------------
        # We smooth each team's mental factors INDEPENDENTLY, then net the results.
        # This is intentional and models diminishing marginal psychological impact.
        #
        # Example asymmetry (intentional):
        #   Team A: letdown 3.5 only           → smoothed = 3.5
        #   Team B: letdown 2.0 + lookahead 1.5 → smoothed = 2.0 + 0.75 = 2.75
        #   Net: 3.5 - 2.75 = 0.75 (even though raw sums both equal 3.5)
        #
        # Why this is correct:
        # 1. One overwhelming distraction is worse than two smaller ones of equal
        #    total magnitude. A team in a SINGLE massive spot is fully distracted.
        # 2. Multiple smaller concerns "mask" each other - coaching can only focus
        #    on so many messages, players have limited capacity for additional load.
        # 3. The alternative (net raw values first, then smooth) would lose this
        #    per-team psychological reality.
        #
        # The raw_mental_sum diagnostic field captures unsmoothed net for comparison.
        # =====================================================================
        # Away team mental penalties (favor home)
        away_mental = []
        if away_factors.letdown_penalty != 0:
            away_mental.append(("letdown", abs(away_factors.letdown_penalty)))
        if away_factors.lookahead_penalty != 0:
            away_mental.append(("lookahead", abs(away_factors.lookahead_penalty)))
        if away_factors.sandwich_penalty != 0:
            away_mental.append(("sandwich", abs(away_factors.sandwich_penalty)))
        if away_factors.game_shape_penalty != 0:
            away_mental.append(("game_shape", abs(away_factors.game_shape_penalty)))

        away_mental_smoothed = self._smooth_mental(away_mental)

        # Home team mental penalties (hurt home)
        home_mental = []
        if home_factors.letdown_penalty != 0:
            home_mental.append(("letdown", abs(home_factors.letdown_penalty)))
        if home_factors.lookahead_penalty != 0:
            home_mental.append(("lookahead", abs(home_factors.lookahead_penalty)))
        if home_factors.sandwich_penalty != 0:
            home_mental.append(("sandwich", abs(home_factors.sandwich_penalty)))
        if home_factors.game_shape_penalty != 0:
            home_mental.append(("game_shape", abs(home_factors.game_shape_penalty)))

        home_mental_smoothed = self._smooth_mental(home_mental)

        # Net mental: away penalties favor home, home penalties hurt home
        result.mental_bucket = away_mental_smoothed - home_mental_smoothed

        # Track raw (unsmoothed) mental sum for diagnostic comparison
        away_mental_raw = sum(v for _, v in away_mental)
        home_mental_raw = sum(v for _, v in home_mental)
        result.raw_mental_sum = away_mental_raw - home_mental_raw

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
        # STEP 5.5: Calculate Pure Situational Score
        # This is what SpreadGenerator should report as "situational"
        # Includes: rest, consecutive_road (linear), mental (smoothed), boosts (linear)
        # Excludes: HFA, travel, altitude (those are separate components)
        #
        # NOTE: Rest and consecutive_road are NOT smoothed by venue factor.
        # They're situational factors independent of venue stacking.
        # =====================================================================
        result.situational_score = (
            result.raw_rest
            + result.raw_consecutive_road
            + result.mental_bucket
            + result.boosts_bucket
        )

        # =====================================================================
        # STEP 6: Final Calculation
        # Compute diagnostic totals showing smoothing progression:
        # - raw_sum_all: True linear sum (zero smoothing)
        # - pre_global_cap_total: After env soft cap + mental smoothing
        # - net_adjustment: After global cap
        # =====================================================================

        # raw_sum_all: What if we just added everything linearly?
        # Uses raw_env_stack (not env_score) and raw_mental_sum (not mental_bucket)
        result.raw_sum_all = (
            result.raw_env_stack
            + result.raw_mental_sum
            + result.boosts_bucket  # Boosts have no smoothing
        )

        # pre_global_cap_total: Post-smoothing, pre-global-cap
        pre_global_cap_total = (
            result.env_score  # Post-env-soft-cap
            + result.mental_bucket  # Post-mental-smoothing
            + result.boosts_bucket
        )
        result.pre_global_cap_total = pre_global_cap_total

        # Apply global cap (±7.0)
        #
        # DESIGN DECISION: Global Cap Priority Ordering
        # ---------------------------------------------
        # When the cap is hit, ALL buckets are proportionally reduced - there's no
        # explicit priority. However, because env+boosts are added first and are
        # typically larger than mental factors, mental factors have less marginal
        # effect in extreme scenarios.
        #
        # Example: Utah home (4.5 HFA + 1.5 travel + 2.0 altitude = 8.0 raw)
        #   - Env soft cap: 5.0 + (8.0-5.0)*0.6 = 6.8
        #   - Rivalry boost: +1.0 → pre-mental = 7.8
        #   - Letdown: +2.0 → would be 9.8, capped to 7.0
        #   - Mental's marginal contribution: only 0.2 of its 2.0 value
        #
        # This is INTENTIONAL. Rationale:
        # 1. Global cap triggers rarely (~1% of games per stack diagnostics)
        # 2. Environmental factors (HFA, travel, altitude) are objectively measurable
        # 3. Mental factors (letdown, lookahead) are more speculative/psychological
        # 4. When caps are hit, trust "harder" physical signals over "softer" ones
        #
        # Alternatives considered but rejected:
        # - Per-bucket caps: Complex, hard to calibrate, doesn't solve the problem
        # - Reserved mental minimum: Arbitrary, adds complexity
        # - Proportional reduction: Already happens implicitly (all excess is clipped)
        if abs(pre_global_cap_total) > self.global_cap:
            result.net_adjustment = max(-self.global_cap, min(self.global_cap, pre_global_cap_total))
            result.was_capped = True
            logger.debug(
                f"Global cap applied: {pre_global_cap_total:.2f} -> {result.net_adjustment:.2f}"
            )
        else:
            result.net_adjustment = pre_global_cap_total
            # Explicit False for defensive programming (matches env_was_capped pattern)
            result.was_capped = False

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
