"""Weather adjustments for totals prediction.

This module provides weather-based adjustments for over/under (totals) betting.
Weather significantly impacts scoring, particularly:
- Wind: Reduces passing efficiency and field goal accuracy
- Cold: Slows game pace, affects ball handling
- Precipitation: Impacts ball security, passing, and kicking

Data source: Tomorrow.io forecast API (see src/api/tomorrow_io.py)
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WeatherConditions:
    """Weather conditions for a game."""

    game_id: int
    home_team: str
    away_team: str
    venue: str
    game_indoors: bool
    temperature: Optional[float]  # Fahrenheit
    wind_speed: Optional[float]  # MPH
    wind_gust: Optional[float] = None  # MPH (peak gusts)
    wind_direction: Optional[int] = None  # Degrees (0-360)
    precipitation: Optional[float] = None  # Inches
    snowfall: Optional[float] = None  # Inches
    humidity: Optional[int] = None  # Percentage
    weather_condition: Optional[str] = None  # Text description


@dataclass
class WeatherAdjustment:
    """Weather adjustment breakdown for totals.

    The adjustment is SCALED by confidence_factor:
        total_adjustment = raw_adjustment * confidence_factor

    This means a 72h forecast (0.65 confidence) with -6.0 raw adjustment
    becomes -3.9 dampened adjustment. As forecast confidence improves
    (closer to game time), the full adjustment is applied.

    HIGH_VARIANCE flag: If confidence < 0.75 AND raw_adjustment > 3.0,
    the game is flagged as high variance. RULE: Never bet OVER on these
    games, even if the edge looks huge — the weather is uncertain but
    potentially severe.
    """

    total_adjustment: float  # Dampened adjustment (raw * confidence)
    raw_adjustment: float  # Pre-dampened adjustment (for display)
    wind_adjustment: float  # Raw wind component
    temperature_adjustment: float  # Raw temp component
    precipitation_adjustment: float  # Raw precip component
    is_indoor: bool
    conditions: Optional[WeatherConditions] = None
    confidence_factor: float = 1.0  # Forecast confidence (0.0-1.0)
    high_variance: bool = False  # True if uncertain severe weather — NO OVERS

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/display."""
        return {
            "total_adjustment": self.total_adjustment,
            "raw_adjustment": self.raw_adjustment,
            "wind_adj": self.wind_adjustment,
            "temp_adj": self.temperature_adjustment,
            "precip_adj": self.precipitation_adjustment,
            "is_indoor": self.is_indoor,
            "confidence": self.confidence_factor,
            "high_variance": self.high_variance,
        }


class WeatherAdjuster:
    """
    Calculate weather adjustments for game totals.

    Weather affects scoring through multiple mechanisms. Uses NON-LINEAR
    thresholds based on sharp betting research (wind is king of unders).

    1. WIND (most significant for totals):
       Wind impact is non-linear. 10 mph does nothing. 20 mph destroys passing.
       Uses average of wind_speed and wind_gust for effective wind.
       - < 12 mph: No impact (0.0 pts)
       - 12-15 mph: Slight deep passing degradation (-1.5 pts)
       - 15-20 mph: Kicking range reduced, deep ball erased (-4.0 pts)
       - > 20 mph: Run-only game profiles, clock runs constantly (-6.0+ pts)

    2. TEMPERATURE:
       Extreme cold turns the ball into a rock (harder to catch/kick).
       - > 32°F: No significant impact (0.0 pts)
       - 20-32°F: Slight reduction in FG% and catch rate (-1.0 pts)
       - < 20°F: Significant impact on mechanics (-3.0 pts)

    3. PRECIPITATION:
       Light rain does NOT hurt totals (can cause missed tackles = more scores).
       Only HEAVY rain/snow causes conservative playcalling.
       - Light rain (< 0.1 in/hr): No adjustment (the "slick trap")
       - Heavy rain (> 0.3 in/hr): Ball security issues (-2.5 pts)
       - Snow with high intensity: Visual impairment, footing (-3.0 pts)

    4. INDOOR GAMES:
       - No weather adjustment (controlled environment)
       - Identified via game_indoors flag

    Edge Theory: Capture forecasts Thursday/Saturday BEFORE market moves the line.
    The value is in timing, not the weather signal itself.
    """

    # Wind thresholds (non-linear tiers)
    WIND_TIER_1 = 12.0  # MPH - below this = no impact
    WIND_TIER_2 = 15.0  # MPH - moderate impact
    WIND_TIER_3 = 20.0  # MPH - severe impact

    WIND_ADJ_TIER_1 = -1.5  # 12-15 mph
    WIND_ADJ_TIER_2 = -4.0  # 15-20 mph
    WIND_ADJ_TIER_3 = -6.0  # 20+ mph (and cap)

    # Temperature thresholds (non-linear tiers)
    TEMP_TIER_1 = 32.0  # Freezing - below this = impact starts
    TEMP_TIER_2 = 20.0  # Severe cold

    TEMP_ADJ_TIER_1 = -1.0  # 20-32°F
    TEMP_ADJ_TIER_2 = -3.0  # < 20°F

    # Precipitation thresholds
    # Light rain < 2.5 mm/hr = ~0.1 in/hr: NO adjustment (the "slick trap")
    # Heavy rain > 7.6 mm/hr = ~0.3 in/hr: adjustment applies
    PRECIP_LIGHT_THRESHOLD = 0.1  # Inches/hr - below this = no adjustment
    PRECIP_HEAVY_THRESHOLD = 0.3  # Inches/hr - heavy precipitation
    PRECIP_HEAVY_PENALTY = -2.5  # Heavy rain penalty
    SNOW_PENALTY = -3.0  # Snow with accumulation

    def __init__(self):
        """Initialize weather adjuster with non-linear thresholds.

        Uses class constants for thresholds. No parameters needed - the
        non-linear tiers are based on sharp betting research and should
        not be casually adjusted.
        """
        pass  # All config is in class constants

    # Pass rate multiplier for wind adjustment (The "Passing Team" Multiplier)
    # Air Raid teams (60%+ pass rate) suffer more; Option teams (<40%) barely care
    PASS_RATE_HIGH_THRESHOLD = 0.55  # Combined pass rate above this = pass-heavy
    PASS_RATE_LOW_THRESHOLD = 0.45   # Combined pass rate below this = run-heavy
    PASS_RATE_HIGH_MULTIPLIER = 1.25  # Pass-heavy teams get 25% more wind penalty
    PASS_RATE_LOW_MULTIPLIER = 0.50   # Run-heavy teams get 50% less wind penalty

    def _get_effective_wind(
        self,
        wind_speed: Optional[float],
        wind_gust: Optional[float] = None,
    ) -> float:
        """Calculate effective wind from speed and gust."""
        if wind_speed is None:
            return 0.0
        if wind_gust is not None:
            return (wind_speed + wind_gust) / 2
        return wind_speed

    def _calculate_wind_adjustment(
        self,
        wind_speed: Optional[float],
        wind_gust: Optional[float] = None,
        combined_pass_rate: Optional[float] = None,
    ) -> float:
        """Calculate wind-based adjustment using non-linear tiers.

        Uses average of wind_speed and wind_gust for effective wind,
        since gusts matter for passing and kicking.

        The "Passing Team" Multiplier: Wind hurts pass-heavy teams more.
        - Air Raid (Ole Miss): 60%+ combined pass rate → 1.25x adjustment
        - Balanced: 45-55% combined pass rate → 1.0x adjustment
        - Triple Option (Army): <45% combined pass rate → 0.5x adjustment

        Args:
            wind_speed: Sustained wind speed in MPH
            wind_gust: Peak wind gust in MPH (optional)
            combined_pass_rate: (home_pass_rate + away_pass_rate) / 2 (optional, 0-1)

        Returns:
            Points adjustment (negative = lower total)
        """
        effective_wind = self._get_effective_wind(wind_speed, wind_gust)
        if effective_wind == 0.0:
            return 0.0

        # Non-linear tiers
        if effective_wind < self.WIND_TIER_1:
            base_adj = 0.0
        elif effective_wind < self.WIND_TIER_2:
            base_adj = self.WIND_ADJ_TIER_1  # -1.5
        elif effective_wind < self.WIND_TIER_3:
            base_adj = self.WIND_ADJ_TIER_2  # -4.0
        else:
            base_adj = self.WIND_ADJ_TIER_3  # -6.0

        # Apply pass rate multiplier if provided
        if combined_pass_rate is not None and base_adj != 0.0:
            if combined_pass_rate >= self.PASS_RATE_HIGH_THRESHOLD:
                base_adj *= self.PASS_RATE_HIGH_MULTIPLIER  # More penalty for pass-heavy
            elif combined_pass_rate <= self.PASS_RATE_LOW_THRESHOLD:
                base_adj *= self.PASS_RATE_LOW_MULTIPLIER  # Less penalty for run-heavy

        return base_adj

    def _calculate_temperature_adjustment(
        self, temperature: Optional[float]
    ) -> float:
        """Calculate temperature-based adjustment using non-linear tiers.

        The "rock effect" - cold balls are harder to catch and kick.

        Args:
            temperature: Temperature in Fahrenheit

        Returns:
            Points adjustment (negative = lower total)
        """
        if temperature is None or temperature >= self.TEMP_TIER_1:
            return 0.0

        # Non-linear tiers
        if temperature >= self.TEMP_TIER_2:
            return self.TEMP_ADJ_TIER_1  # 20-32°F: -1.0
        else:
            return self.TEMP_ADJ_TIER_2  # < 20°F: -3.0

    # Weather conditions that indicate HEAVY precipitation (penalty applies)
    HEAVY_RAIN_CONDITIONS = frozenset({
        "Heavy Rain",
        "Thunderstorm",
    })

    SNOW_CONDITIONS = frozenset({
        "Snow",
        "Snowfall",
        "Heavy Snow",
        "Sleet",
        "Freezing Rain",
        "Wintry Mix",
    })

    # Light conditions - the "slick trap" - NO penalty
    LIGHT_PRECIP_CONDITIONS = frozenset({
        "Rain",
        "Light Rain",
        "Rain Shower",
        "Drizzle",
        "Flurries",
        "Light Snow",
    })

    def _calculate_precipitation_adjustment(
        self,
        precipitation: Optional[float],
        snowfall: Optional[float] = None,
        weather_condition: Optional[str] = None,
        effective_wind: float = 0.0,
    ) -> float:
        """Calculate precipitation-based adjustment.

        THE "SLICK TRAP": Light rain does NOT hurt totals. Defenders slip,
        miss tackles, and games can go OVER. Only HEAVY rain/snow with
        conservative playcalling reduces scoring.

        THE "SNOW OVERREACTION FADE": Public loves betting "Snow Unders" but
        snow without wind often goes OVER. Defenders slip, receivers know
        their routes. Only bet Snow Under if wind is ALSO present (wind makes
        snow swirl, disrupts passing lanes).

        Args:
            precipitation: Rain intensity in inches/hr
            snowfall: Snowfall in inches (accumulation)
            weather_condition: Text description (Rain, Snow, etc.)
            effective_wind: Effective wind speed in MPH (for snow+wind check)

        Returns:
            Points adjustment (negative = lower total)
        """
        # Snow: THE "OVERREACTION FADE"
        # Snow without wind = NO penalty (sharps often bet OVER)
        # Snow with significant wind = apply penalty
        if weather_condition in self.SNOW_CONDITIONS:
            if (snowfall or 0.0) > 0.1:  # Accumulating snow
                if effective_wind >= self.WIND_TIER_1:  # Wind >= 12 mph
                    return self.SNOW_PENALTY  # -3.0 (wind makes snow bad)
                else:
                    logger.debug(f"Snow without wind ({effective_wind:.0f} mph) - no penalty (overreaction fade)")
                    return 0.0  # Snow alone = defenders slip, game can go OVER
            return 0.0

        # Light rain/drizzle = NO penalty (the slick trap)
        if weather_condition in self.LIGHT_PRECIP_CONDITIONS:
            return 0.0

        # Heavy rain - check intensity
        rain_intensity = precipitation or 0.0

        if rain_intensity < self.PRECIP_LIGHT_THRESHOLD:
            return 0.0  # < 0.1 in/hr = light, no penalty

        if rain_intensity >= self.PRECIP_HEAVY_THRESHOLD:
            return self.PRECIP_HEAVY_PENALTY  # > 0.3 in/hr = heavy, -2.5

        # Moderate rain (0.1 - 0.3 in/hr) - small penalty only if condition says heavy
        if weather_condition in self.HEAVY_RAIN_CONDITIONS:
            return self.PRECIP_HEAVY_PENALTY

        return 0.0

    # Confidence thresholds
    # 0.75 = ~48h out (NAM/HRRR mesoscale models reliable)
    # 0.85 = ~24h out
    # 0.90 = ~12h out (maximum confidence, fire limit bets)
    HIGH_VARIANCE_CONFIDENCE = 0.75  # Below this + severe weather = HIGH_VARIANCE flag
    HIGH_VARIANCE_RAW_THRESHOLD = 3.0  # Raw adjustment that triggers HIGH_VARIANCE

    def calculate_adjustment(
        self,
        conditions: WeatherConditions,
        combined_pass_rate: Optional[float] = None,
        confidence_factor: float = 1.0,
    ) -> WeatherAdjustment:
        """Calculate total weather adjustment for a game.

        CONFIDENCE SCALING: Adjustment is scaled by confidence_factor.
            dampened_adjustment = raw_adjustment * confidence_factor

        This means:
        - 72h forecast (0.65): -6.0 raw → -3.9 dampened (scouting value)
        - 48h forecast (0.75): -6.0 raw → -4.5 dampened (green light)
        - 12h forecast (0.90): -6.0 raw → -5.4 dampened (max limit)

        HIGH_VARIANCE FLAG: If confidence < 0.75 AND abs(raw) > 3.0,
        the game is flagged high_variance=True. RULE: Never bet OVER
        on these games — weather is uncertain but potentially severe.

        Args:
            conditions: WeatherConditions dataclass with game weather
            combined_pass_rate: (home_pass_rate + away_pass_rate) / 2, range 0-1
                               Pass-heavy matchups (>55%) get bigger wind penalty
                               Run-heavy matchups (<45%) get smaller wind penalty
            confidence_factor: Forecast confidence (0.0-1.0), from hours_until_game

        Returns:
            WeatherAdjustment with raw and dampened adjustments
        """
        # Indoor games get no adjustment
        if conditions.game_indoors:
            return WeatherAdjustment(
                total_adjustment=0.0,
                raw_adjustment=0.0,
                wind_adjustment=0.0,
                temperature_adjustment=0.0,
                precipitation_adjustment=0.0,
                is_indoor=True,
                conditions=conditions,
                confidence_factor=confidence_factor,
            )

        # Get effective wind (needed for snow+wind check)
        effective_wind = self._get_effective_wind(
            conditions.wind_speed,
            conditions.wind_gust,
        )

        # Calculate individual RAW adjustments
        wind_adj = self._calculate_wind_adjustment(
            conditions.wind_speed,
            conditions.wind_gust,
            combined_pass_rate=combined_pass_rate,
        )
        temp_adj = self._calculate_temperature_adjustment(conditions.temperature)
        precip_adj = self._calculate_precipitation_adjustment(
            conditions.precipitation,
            conditions.snowfall,
            conditions.weather_condition,
            effective_wind=effective_wind,
        )

        # Combine RAW adjustments (they stack)
        raw_adj = wind_adj + temp_adj + precip_adj

        # Scale by confidence (the "dampening")
        dampened_adj = raw_adj * confidence_factor

        # HIGH_VARIANCE flag: uncertain severe weather — NO OVERS
        # If forecast is low confidence BUT predicting severe weather,
        # flag it so we don't bet OVER (weather might be worse than model shows)
        high_variance = (
            confidence_factor < self.HIGH_VARIANCE_CONFIDENCE and
            abs(raw_adj) > self.HIGH_VARIANCE_RAW_THRESHOLD
        )

        if high_variance:
            logger.info(
                f"HIGH_VARIANCE: {conditions.home_team} vs {conditions.away_team} - "
                f"raw={raw_adj:.1f}, conf={confidence_factor:.2f} — NO OVERS"
            )

        return WeatherAdjustment(
            total_adjustment=dampened_adj,
            raw_adjustment=raw_adj,
            wind_adjustment=wind_adj,
            temperature_adjustment=temp_adj,
            precipitation_adjustment=precip_adj,
            is_indoor=False,
            conditions=conditions,
            confidence_factor=confidence_factor,
            high_variance=high_variance,
        )

    def get_weather_summary(self, conditions: WeatherConditions) -> str:
        """Get human-readable weather summary.

        Args:
            conditions: WeatherConditions for a game

        Returns:
            Summary string describing conditions
        """
        if conditions.game_indoors:
            return "Indoor (dome)"

        parts = []

        if conditions.temperature is not None:
            parts.append(f"{conditions.temperature:.0f}°F")

        if conditions.wind_speed is not None and conditions.wind_speed > 5:
            parts.append(f"wind {conditions.wind_speed:.0f} mph")

        if conditions.weather_condition:
            if conditions.weather_condition not in ["Clear", "Fair"]:
                parts.append(conditions.weather_condition)

        return ", ".join(parts) if parts else "Unknown conditions"

    def is_weather_concern(self, conditions: WeatherConditions) -> bool:
        """Check if weather is significant enough to warrant attention.

        Uses the new non-linear thresholds. A "concern" means the weather
        would trigger a material adjustment (> 1 point).

        Args:
            conditions: WeatherConditions for a game

        Returns:
            True if weather may significantly impact totals
        """
        if conditions.game_indoors:
            return False

        # Wind concern: effective wind > 12 mph (TIER_1)
        effective_wind = conditions.wind_speed or 0.0
        if conditions.wind_gust:
            effective_wind = (effective_wind + conditions.wind_gust) / 2
        high_wind = effective_wind >= self.WIND_TIER_1

        # Temperature concern: < 32°F (TIER_1)
        cold = (
            conditions.temperature is not None
            and conditions.temperature < self.TEMP_TIER_1
        )

        # Precipitation concern: heavy rain OR snow
        heavy_precip = (
            conditions.weather_condition in self.HEAVY_RAIN_CONDITIONS
            or conditions.weather_condition in self.SNOW_CONDITIONS
        )

        return high_wind or cold or heavy_precip

    def get_parameter_summary(self) -> dict:
        """Get current parameter settings for transparency.

        Returns:
            Dict of all configurable parameters (non-linear tiers)
        """
        return {
            "wind_tiers_mph": [self.WIND_TIER_1, self.WIND_TIER_2, self.WIND_TIER_3],
            "wind_adjustments_pts": [self.WIND_ADJ_TIER_1, self.WIND_ADJ_TIER_2, self.WIND_ADJ_TIER_3],
            "temp_tiers_f": [self.TEMP_TIER_1, self.TEMP_TIER_2],
            "temp_adjustments_pts": [self.TEMP_ADJ_TIER_1, self.TEMP_ADJ_TIER_2],
            "precip_light_threshold_in_hr": self.PRECIP_LIGHT_THRESHOLD,
            "precip_heavy_threshold_in_hr": self.PRECIP_HEAVY_THRESHOLD,
            "precip_heavy_penalty_pts": self.PRECIP_HEAVY_PENALTY,
            "snow_penalty_pts": self.SNOW_PENALTY,
        }
