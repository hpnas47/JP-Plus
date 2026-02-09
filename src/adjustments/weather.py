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
    """Weather adjustment breakdown for totals."""

    total_adjustment: float  # Combined adjustment to predicted total
    wind_adjustment: float
    temperature_adjustment: float
    precipitation_adjustment: float
    is_indoor: bool
    conditions: Optional[WeatherConditions] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/display."""
        return {
            "total_adjustment": self.total_adjustment,
            "wind_adj": self.wind_adjustment,
            "temp_adj": self.temperature_adjustment,
            "precip_adj": self.precipitation_adjustment,
            "is_indoor": self.is_indoor,
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

    def _calculate_wind_adjustment(
        self,
        wind_speed: Optional[float],
        wind_gust: Optional[float] = None,
    ) -> float:
        """Calculate wind-based adjustment using non-linear tiers.

        Uses average of wind_speed and wind_gust for effective wind,
        since gusts matter for passing and kicking.

        Args:
            wind_speed: Sustained wind speed in MPH
            wind_gust: Peak wind gust in MPH (optional)

        Returns:
            Points adjustment (negative = lower total)
        """
        if wind_speed is None:
            return 0.0

        # Use average of speed and gust if gust available
        if wind_gust is not None:
            effective_wind = (wind_speed + wind_gust) / 2
        else:
            effective_wind = wind_speed

        # Non-linear tiers
        if effective_wind < self.WIND_TIER_1:
            return 0.0
        elif effective_wind < self.WIND_TIER_2:
            return self.WIND_ADJ_TIER_1  # -1.5
        elif effective_wind < self.WIND_TIER_3:
            return self.WIND_ADJ_TIER_2  # -4.0
        else:
            return self.WIND_ADJ_TIER_3  # -6.0

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
    ) -> float:
        """Calculate precipitation-based adjustment.

        THE "SLICK TRAP": Light rain does NOT hurt totals. Defenders slip,
        miss tackles, and games can go OVER. Only HEAVY rain/snow with
        conservative playcalling reduces scoring.

        Args:
            precipitation: Rain intensity in inches/hr
            snowfall: Snowfall in inches (accumulation)
            weather_condition: Text description (Rain, Snow, etc.)

        Returns:
            Points adjustment (negative = lower total)
        """
        # Snow gets its own penalty (visual impairment, footing)
        if weather_condition in self.SNOW_CONDITIONS:
            if (snowfall or 0.0) > 0.1:  # Accumulating snow
                return self.SNOW_PENALTY  # -3.0
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

    def calculate_adjustment(
        self,
        conditions: WeatherConditions,
    ) -> WeatherAdjustment:
        """Calculate total weather adjustment for a game.

        Args:
            conditions: WeatherConditions dataclass with game weather

        Returns:
            WeatherAdjustment with breakdown of all adjustments
        """
        # Indoor games get no adjustment
        if conditions.game_indoors:
            return WeatherAdjustment(
                total_adjustment=0.0,
                wind_adjustment=0.0,
                temperature_adjustment=0.0,
                precipitation_adjustment=0.0,
                is_indoor=True,
                conditions=conditions,
            )

        # Calculate individual adjustments
        wind_adj = self._calculate_wind_adjustment(
            conditions.wind_speed,
            conditions.wind_gust,
        )
        temp_adj = self._calculate_temperature_adjustment(conditions.temperature)
        precip_adj = self._calculate_precipitation_adjustment(
            conditions.precipitation,
            conditions.snowfall,
            conditions.weather_condition,
        )

        # Combine adjustments (they stack)
        total_adj = wind_adj + temp_adj + precip_adj

        return WeatherAdjustment(
            total_adjustment=total_adj,
            wind_adjustment=wind_adj,
            temperature_adjustment=temp_adj,
            precipitation_adjustment=precip_adj,
            is_indoor=False,
            conditions=conditions,
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
