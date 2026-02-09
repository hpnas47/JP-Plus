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
    wind_direction: Optional[int]  # Degrees (0-360)
    precipitation: Optional[float]  # Inches
    snowfall: Optional[float]  # Inches
    humidity: Optional[int]  # Percentage
    weather_condition: Optional[str]  # Text description


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

    Weather affects scoring through multiple mechanisms:

    1. WIND (most significant for totals):
       - High wind (15+ mph) reduces passing efficiency
       - Affects field goal accuracy, especially longer kicks
       - Research shows ~0.3 points reduction per mph over 10 mph threshold

    2. TEMPERATURE:
       - Cold weather (<40°F) reduces scoring
       - Ball is harder to grip, affects throwing/catching
       - Players tire faster in extreme cold
       - ~0.15 points reduction per degree below 40°F

    3. PRECIPITATION:
       - Rain significantly impacts ball security
       - Reduces passing attempts, increases run-heavy game scripts
       - Heavy rain (>0.02 inches) warrants flat penalty

    4. INDOOR GAMES:
       - No weather adjustment (controlled environment)
       - Identified via game_indoors flag from API

    Default parameters are conservative estimates based on:
    - NFL weather studies (Marek 2015, Kacsmar 2016)
    - CFB-specific analysis showing similar effects
    - Backtesting against historical totals results

    NOTE: These defaults should be validated against ATS/totals performance
    before relying on them for betting decisions.
    """

    # Wind thresholds and adjustments
    DEFAULT_WIND_THRESHOLD = 10.0  # MPH - adjustment starts above this
    DEFAULT_WIND_COEFFICIENT = -0.3  # Points per MPH above threshold
    DEFAULT_WIND_CAP = -6.0  # Maximum wind adjustment (prevents extreme values)

    # Temperature thresholds and adjustments
    DEFAULT_TEMP_THRESHOLD = 40.0  # Fahrenheit - adjustment starts below this
    DEFAULT_TEMP_COEFFICIENT = -0.15  # Points per degree below threshold
    DEFAULT_TEMP_CAP = -4.0  # Maximum cold adjustment

    # Precipitation adjustments
    DEFAULT_PRECIP_THRESHOLD = 0.02  # Inches - significant precipitation
    DEFAULT_PRECIP_PENALTY = -3.0  # Flat penalty for rain/snow games
    DEFAULT_HEAVY_PRECIP_THRESHOLD = 0.05  # Heavy precipitation
    DEFAULT_HEAVY_PRECIP_PENALTY = -5.0  # Additional penalty for heavy precip

    def __init__(
        self,
        wind_threshold: float = DEFAULT_WIND_THRESHOLD,
        wind_coefficient: float = DEFAULT_WIND_COEFFICIENT,
        wind_cap: float = DEFAULT_WIND_CAP,
        temp_threshold: float = DEFAULT_TEMP_THRESHOLD,
        temp_coefficient: float = DEFAULT_TEMP_COEFFICIENT,
        temp_cap: float = DEFAULT_TEMP_CAP,
        precip_threshold: float = DEFAULT_PRECIP_THRESHOLD,
        precip_penalty: float = DEFAULT_PRECIP_PENALTY,
        heavy_precip_threshold: float = DEFAULT_HEAVY_PRECIP_THRESHOLD,
        heavy_precip_penalty: float = DEFAULT_HEAVY_PRECIP_PENALTY,
    ):
        """Initialize weather adjuster with configurable parameters.

        Args:
            wind_threshold: Wind speed (mph) where adjustment begins
            wind_coefficient: Points adjustment per mph above threshold
            wind_cap: Maximum wind adjustment (should be negative)
            temp_threshold: Temperature (F) where cold adjustment begins
            temp_coefficient: Points adjustment per degree below threshold
            temp_cap: Maximum temperature adjustment (should be negative)
            precip_threshold: Precipitation (inches) for rain penalty
            precip_penalty: Flat penalty for precipitation games
            heavy_precip_threshold: Threshold for heavy precipitation
            heavy_precip_penalty: Additional penalty for heavy precip
        """
        self.wind_threshold = wind_threshold
        self.wind_coefficient = wind_coefficient
        self.wind_cap = wind_cap
        self.temp_threshold = temp_threshold
        self.temp_coefficient = temp_coefficient
        self.temp_cap = temp_cap
        self.precip_threshold = precip_threshold
        self.precip_penalty = precip_penalty
        self.heavy_precip_threshold = heavy_precip_threshold
        self.heavy_precip_penalty = heavy_precip_penalty

    def _calculate_wind_adjustment(self, wind_speed: Optional[float]) -> float:
        """Calculate wind-based adjustment.

        Args:
            wind_speed: Wind speed in MPH

        Returns:
            Points adjustment (negative = lower total)
        """
        if wind_speed is None or wind_speed <= self.wind_threshold:
            return 0.0

        excess_wind = wind_speed - self.wind_threshold
        adjustment = self.wind_coefficient * excess_wind

        # Apply cap to prevent extreme adjustments
        return max(adjustment, self.wind_cap)

    def _calculate_temperature_adjustment(
        self, temperature: Optional[float]
    ) -> float:
        """Calculate temperature-based adjustment.

        Args:
            temperature: Temperature in Fahrenheit

        Returns:
            Points adjustment (negative = lower total)
        """
        if temperature is None or temperature >= self.temp_threshold:
            return 0.0

        degrees_below = self.temp_threshold - temperature
        adjustment = self.temp_coefficient * degrees_below

        # Apply cap to prevent extreme adjustments
        return max(adjustment, self.temp_cap)

    # Weather conditions that indicate actual precipitation
    PRECIPITATION_CONDITIONS = frozenset({
        "Rain",
        "Light Rain",
        "Heavy Rain",
        "Rain Shower",
        "Drizzle",
        "Thunderstorm",
        "Snow",
        "Snowfall",
        "Light Snow",
        "Heavy Snow",
        "Sleet",
        "Freezing Rain",
        "Wintry Mix",
    })

    def _calculate_precipitation_adjustment(
        self,
        precipitation: Optional[float],
        snowfall: Optional[float] = None,
        weather_condition: Optional[str] = None,
    ) -> float:
        """Calculate precipitation-based adjustment.

        Only applies penalty when both:
        1. Numeric precipitation value exceeds threshold
        2. Weather condition indicates actual rain/snow (not just fog/humidity)

        Args:
            precipitation: Precipitation in inches
            snowfall: Snowfall in inches (added to total precip)
            weather_condition: Text description (Rain, Snow, Fog, etc.)

        Returns:
            Points adjustment (negative = lower total)
        """
        total_precip = (precipitation or 0.0) + (snowfall or 0.0)

        if total_precip < self.precip_threshold:
            return 0.0

        # Only apply penalty if condition indicates actual precipitation
        # This avoids penalizing fog/humidity that may register small precip values
        if weather_condition and weather_condition not in self.PRECIPITATION_CONDITIONS:
            return 0.0

        if total_precip >= self.heavy_precip_threshold:
            return self.heavy_precip_penalty

        return self.precip_penalty

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
        wind_adj = self._calculate_wind_adjustment(conditions.wind_speed)
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

        Args:
            conditions: WeatherConditions for a game

        Returns:
            True if weather may significantly impact totals
        """
        if conditions.game_indoors:
            return False

        # Check each factor
        high_wind = (
            conditions.wind_speed is not None
            and conditions.wind_speed > self.wind_threshold
        )
        cold = (
            conditions.temperature is not None
            and conditions.temperature < self.temp_threshold
        )
        # Only flag precipitation if condition indicates actual rain/snow
        has_precip_condition = (
            conditions.weather_condition in self.PRECIPITATION_CONDITIONS
            if conditions.weather_condition
            else False
        )
        precip = has_precip_condition and (
            (
                conditions.precipitation is not None
                and conditions.precipitation > self.precip_threshold
            )
            or (
                conditions.snowfall is not None
                and conditions.snowfall > self.precip_threshold
            )
        )

        return high_wind or cold or precip

    def get_parameter_summary(self) -> dict:
        """Get current parameter settings for transparency.

        Returns:
            Dict of all configurable parameters
        """
        return {
            "wind_threshold_mph": self.wind_threshold,
            "wind_coefficient_pts_per_mph": self.wind_coefficient,
            "wind_cap_pts": self.wind_cap,
            "temp_threshold_f": self.temp_threshold,
            "temp_coefficient_pts_per_degree": self.temp_coefficient,
            "temp_cap_pts": self.temp_cap,
            "precip_threshold_inches": self.precip_threshold,
            "precip_penalty_pts": self.precip_penalty,
            "heavy_precip_threshold_inches": self.heavy_precip_threshold,
            "heavy_precip_penalty_pts": self.heavy_precip_penalty,
        }
