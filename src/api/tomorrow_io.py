"""Tomorrow.io Weather Forecast API client.

Provides weather forecasts for CFB game venues using the Tomorrow.io API.
Used for operational totals predictions where we need forecasts BEFORE game time.

FORECAST DISCIPLINE (Lookahead Bias Prevention):
    The betting edge in weather comes from TIMING — capturing forecasts Thursday
    before the market adjusts lines. Using later, more accurate forecasts would
    be lookahead bias (using information you wouldn't have at decision time).

    Rules:
    1. BETTING DECISIONS: Use get_betting_forecast() — returns FIRST capture
    2. POSITION MANAGEMENT: Use get_latest_forecast() — for confirmation only
    3. BACKTESTS: Should NOT use weather (CFBD actuals = perfect hindsight)
       - Backtest showed: weather improves MAE by 0.04 but ATS by 0%
       - Market already prices weather; we gain nothing from hindsight

    The database stores multiple forecasts per game (Thursday + Saturday).
    Always be explicit about which forecast you're using and why.

API Documentation: https://docs.tomorrow.io/reference/weather-forecast
"""

import logging
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "weather_forecasts.db"


@dataclass
class WeatherForecast:
    """Weather forecast for a game venue."""

    venue_id: int
    venue_name: str
    latitude: float
    longitude: float
    forecast_time: datetime  # When forecast was captured
    game_time: datetime  # When game is scheduled
    hours_until_game: int  # Forecast horizon

    # Core weather fields
    temperature: Optional[float]  # Fahrenheit
    wind_speed: Optional[float]  # MPH
    wind_gust: Optional[float]  # MPH
    wind_direction: Optional[int]  # Degrees 0-360
    precipitation_probability: Optional[float]  # 0-100%
    rain_intensity: Optional[float]  # mm/hr
    snow_intensity: Optional[float]  # mm/hr
    humidity: Optional[int]  # 0-100%
    cloud_cover: Optional[int]  # 0-100%
    weather_code: Optional[int]  # Tomorrow.io weather code

    # Metadata
    is_indoor: bool = False
    confidence_factor: float = 1.0  # Reduce for longer forecast horizons


@dataclass
class VenueLocation:
    """Venue with geographic coordinates."""

    venue_id: int
    name: str
    city: str
    state: str
    latitude: float
    longitude: float
    dome: bool  # True if indoor stadium
    elevation: Optional[float] = None
    timezone: Optional[str] = None


# Known domes fallback list - CFBD dome field is sometimes inaccurate or missing
# This list provides a safety net for known indoor stadiums
# Venue names are partial matches (case-insensitive) to handle naming variations
KNOWN_DOMES = {
    # FBS Team Home Stadiums
    "carrier dome",  # Syracuse (now JMA Wireless Dome)
    "jma wireless dome",  # Syracuse (new name)
    "alamodome",  # UTSA
    "ford field",  # Detroit (MAC Championship, Quick Lane Bowl)
    "u.s. bank stadium",  # Minnesota Vikings (bowl games)
    "lucas oil stadium",  # Indianapolis (Big Ten Championship)
    "caesars superdome",  # New Orleans (Sugar Bowl, Tulane sometimes)
    "mercedes-benz superdome",  # New Orleans (old name)
    "louisiana superdome",  # New Orleans (older name)

    # Major Bowl Game Venues (retractable roofs typically closed for CFB)
    "at&t stadium",  # Arlington TX (Cotton Bowl, various games)
    "mercedes-benz stadium",  # Atlanta (SEC Championship, Peach Bowl, CFP)
    "nrg stadium",  # Houston (Texas Bowl)
    "state farm stadium",  # Glendale AZ (Fiesta Bowl, CFP)
    "sofi stadium",  # LA (Pac-12 Championship, various bowls)
    "allegiant stadium",  # Las Vegas (Vegas Bowl, Pac-12 Championship)

    # Other indoor venues that occasionally host CFB
    "tropicana field",  # St. Petersburg (various games)
    "metrodome",  # Minneapolis (demolished, but for historical data)
}


def is_dome_venue(venue_name: str, cfbd_dome_flag: bool) -> bool:
    """Check if a venue is a dome using CFBD flag + fallback list.

    Args:
        venue_name: Name of the venue
        cfbd_dome_flag: Dome flag from CFBD API (may be inaccurate)

    Returns:
        True if venue is known to be a dome/indoor stadium
    """
    if cfbd_dome_flag:
        return True

    # Check fallback list (case-insensitive partial match)
    venue_lower = venue_name.lower()
    for known_dome in KNOWN_DOMES:
        if known_dome in venue_lower:
            return True

    return False


class TomorrowIOClient:
    """Client for Tomorrow.io Weather Forecast API.

    Usage:
        client = TomorrowIOClient(api_key="your_key")
        forecast = client.get_forecast(latitude=40.7128, longitude=-74.0060, game_time=dt)
    """

    BASE_URL = "https://api.tomorrow.io/v4/weather/forecast"

    # Weather codes that indicate precipitation
    # https://docs.tomorrow.io/reference/data-layers-weather-codes
    PRECIPITATION_CODES = {
        4000,  # Drizzle
        4001,  # Rain
        4200,  # Light Rain
        4201,  # Heavy Rain
        5000,  # Snow
        5001,  # Flurries
        5100,  # Light Snow
        5101,  # Heavy Snow
        6000,  # Freezing Drizzle
        6001,  # Freezing Rain
        6200,  # Light Freezing Rain
        6201,  # Heavy Freezing Rain
        7000,  # Ice Pellets
        7101,  # Heavy Ice Pellets
        7102,  # Light Ice Pellets
        8000,  # Thunderstorm
    }

    # Free tier limit: 25 requests per hour
    FREE_TIER_HOURLY_LIMIT = 25

    def __init__(
        self,
        api_key: str,
        db_path: Path = DEFAULT_DB_PATH,
        units: str = "imperial",
        rate_limit_delay: float = 3.0,  # Seconds between API calls
        max_retries: int = 3,  # Max retries on rate limit
        max_backoff_seconds: float = 60.0,  # Max wait on exponential backoff
    ):
        """Initialize Tomorrow.io client.

        Args:
            api_key: Tomorrow.io API key
            db_path: Path to SQLite database for caching forecasts
            units: "imperial" (Fahrenheit, MPH) or "metric" (Celsius, m/s)
            rate_limit_delay: Seconds to wait between API calls (free tier: 25/hr)
            max_retries: Maximum retries on 429 rate limit errors
            max_backoff_seconds: Maximum wait time for exponential backoff
        """
        self.api_key = api_key
        self.db_path = db_path
        self.units = units
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.max_backoff_seconds = max_backoff_seconds
        self._last_call_time = 0.0
        # Hourly rate limit tracking
        self._hourly_calls: list[float] = []  # Timestamps of calls within current hour
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER,
                    venue_id INTEGER,
                    venue_name TEXT,
                    latitude REAL,
                    longitude REAL,
                    forecast_time TEXT,
                    game_time TEXT,
                    hours_until_game INTEGER,
                    temperature REAL,
                    wind_speed REAL,
                    wind_gust REAL,
                    wind_direction INTEGER,
                    precipitation_probability REAL,
                    rain_intensity REAL,
                    snow_intensity REAL,
                    humidity INTEGER,
                    cloud_cover INTEGER,
                    weather_code INTEGER,
                    is_indoor INTEGER,
                    confidence_factor REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(game_id, forecast_time)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS venues (
                    venue_id INTEGER PRIMARY KEY,
                    name TEXT,
                    city TEXT,
                    state TEXT,
                    latitude REAL,
                    longitude REAL,
                    dome INTEGER,
                    elevation REAL,
                    timezone TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Index for efficient game lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_forecasts_game
                ON forecasts(game_id, forecast_time)
            """)

    def _prune_hourly_calls(self) -> None:
        """Remove call timestamps older than 1 hour."""
        cutoff = time.time() - 3600  # 1 hour ago
        self._hourly_calls = [t for t in self._hourly_calls if t > cutoff]

    def get_hourly_call_count(self) -> int:
        """Return number of API calls made in the last hour."""
        self._prune_hourly_calls()
        return len(self._hourly_calls)

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        game_time: datetime,
        venue_id: Optional[int] = None,
        venue_name: Optional[str] = None,
    ) -> Optional[WeatherForecast]:
        """Get weather forecast for a location and time.

        Args:
            latitude: Venue latitude
            longitude: Venue longitude
            game_time: Scheduled game start time
            venue_id: Optional venue ID for caching
            venue_name: Optional venue name

        Returns:
            WeatherForecast or None if API call fails
        """
        location_ctx = f"venue={venue_name or 'Unknown'} ({latitude:.4f}, {longitude:.4f})"

        # Check hourly rate limit before attempting
        self._prune_hourly_calls()
        if len(self._hourly_calls) >= self.FREE_TIER_HOURLY_LIMIT:
            oldest_call = min(self._hourly_calls)
            wait_until = oldest_call + 3600
            wait_remaining = wait_until - time.time()
            logger.warning(
                f"Hourly rate limit reached ({self.FREE_TIER_HOURLY_LIMIT}/hr). "
                f"Next slot available in {wait_remaining / 60:.1f} minutes. "
                f"Skipping {location_ctx}"
            )
            return None

        try:
            # Build request
            params = {
                "location": f"{latitude},{longitude}",
                "apikey": self.api_key,
                "units": self.units,
                "timesteps": "1h",  # Hourly forecasts
            }

            headers = {
                "accept": "application/json",
                "accept-encoding": "gzip",
            }

            # Retry loop with exponential backoff + jitter for rate limits
            data = None
            for attempt in range(self.max_retries + 1):
                # Rate limiting - wait between calls
                elapsed = time.time() - self._last_call_time
                if elapsed < self.rate_limit_delay:
                    time.sleep(self.rate_limit_delay - elapsed)
                self._last_call_time = time.time()

                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=30,
                )

                if response.status_code == 429:
                    # Rate limited - exponential backoff with jitter
                    if attempt < self.max_retries:
                        # Base: 5s, 10s, 20s... capped at max_backoff_seconds
                        base_wait = min((2 ** attempt) * 5, self.max_backoff_seconds)
                        # Add jitter: ±25% randomization to prevent thundering herd
                        jitter = base_wait * random.uniform(-0.25, 0.25)
                        wait_time = base_wait + jitter
                        logger.warning(
                            f"Rate limited (429) for {location_ctx}. "
                            f"Retry {attempt + 1}/{self.max_retries} in {wait_time:.1f}s"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(
                            f"Max retries ({self.max_retries}) exceeded on rate limit "
                            f"for {location_ctx}"
                        )
                        return None

                response.raise_for_status()
                data = response.json()
                # Record successful API call for hourly tracking
                self._hourly_calls.append(time.time())
                break

            if data is None:
                return None

            # Find the forecast closest to game time
            forecast_data = self._find_forecast_for_time(data, game_time)
            if not forecast_data:
                logger.warning(f"No forecast found for game time {game_time}")
                return None

            # Calculate hours until game (use UTC for consistency)
            # game_time should be UTC-aware; if naive, assume UTC
            now_utc = datetime.now(timezone.utc)
            if game_time.tzinfo is None:
                # Naive datetime - assume it represents UTC
                game_time_utc = game_time.replace(tzinfo=timezone.utc)
            else:
                game_time_utc = game_time
            forecast_time = now_utc.replace(tzinfo=None)  # Store as naive for DB compatibility
            hours_until = int((game_time_utc - now_utc).total_seconds() / 3600)

            # Calculate confidence factor based on forecast horizon
            # Shorter horizons = higher confidence
            if hours_until <= 6:
                confidence = 0.95
            elif hours_until <= 12:
                confidence = 0.90
            elif hours_until <= 24:
                confidence = 0.85
            elif hours_until <= 48:
                confidence = 0.75
            else:
                confidence = 0.65

            values = forecast_data.get("values", {})

            return WeatherForecast(
                venue_id=venue_id or 0,
                venue_name=venue_name or "Unknown",
                latitude=latitude,
                longitude=longitude,
                forecast_time=forecast_time,
                game_time=game_time,
                hours_until_game=hours_until,
                temperature=values.get("temperature"),
                wind_speed=values.get("windSpeed"),
                wind_gust=values.get("windGust"),
                wind_direction=values.get("windDirection"),
                precipitation_probability=values.get("precipitationProbability"),
                rain_intensity=values.get("rainIntensity"),
                snow_intensity=values.get("snowIntensity"),
                humidity=values.get("humidity"),
                cloud_cover=values.get("cloudCover"),
                weather_code=values.get("weatherCode"),
                is_indoor=False,
                confidence_factor=confidence,
            )

        except requests.RequestException as e:
            logger.error(f"Tomorrow.io API error for {location_ctx}: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing Tomorrow.io response for {location_ctx}: {e}")
            return None

    def _find_forecast_for_time(
        self, data: dict, target_time: datetime
    ) -> Optional[dict]:
        """Find the hourly forecast closest to target time.

        Args:
            data: Tomorrow.io API response
            target_time: Target game time

        Returns:
            Forecast data dict or None
        """
        timelines = data.get("timelines", {})
        hourly = timelines.get("hourly", [])

        if not hourly:
            return None

        # Find closest forecast to target time
        best_match = None
        min_diff = float("inf")

        # Normalize target_time to UTC for comparison
        if target_time.tzinfo is None:
            # Naive datetime - assume UTC
            target_utc = target_time.replace(tzinfo=timezone.utc)
        else:
            target_utc = target_time

        for forecast in hourly:
            time_str = forecast.get("time", "")
            try:
                # Parse ISO format: 2024-01-15T14:00:00Z (Tomorrow.io returns UTC)
                forecast_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

                diff = abs((forecast_time - target_utc).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    best_match = forecast
            except ValueError:
                continue

        return best_match

    def save_forecast(
        self,
        forecast: WeatherForecast,
        game_id: int,
    ) -> None:
        """Save forecast to database.

        Args:
            forecast: WeatherForecast to save
            game_id: CFBD game ID
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO forecasts (
                    game_id, venue_id, venue_name, latitude, longitude,
                    forecast_time, game_time, hours_until_game,
                    temperature, wind_speed, wind_gust, wind_direction,
                    precipitation_probability, rain_intensity, snow_intensity,
                    humidity, cloud_cover, weather_code,
                    is_indoor, confidence_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    forecast.venue_id,
                    forecast.venue_name,
                    forecast.latitude,
                    forecast.longitude,
                    forecast.forecast_time.isoformat(),
                    forecast.game_time.isoformat(),
                    forecast.hours_until_game,
                    forecast.temperature,
                    forecast.wind_speed,
                    forecast.wind_gust,
                    forecast.wind_direction,
                    forecast.precipitation_probability,
                    forecast.rain_intensity,
                    forecast.snow_intensity,
                    forecast.humidity,
                    forecast.cloud_cover,
                    forecast.weather_code,
                    1 if forecast.is_indoor else 0,
                    forecast.confidence_factor,
                ),
            )

    def _row_to_forecast(self, row: sqlite3.Row) -> WeatherForecast:
        """Convert database row to WeatherForecast object."""
        return WeatherForecast(
            venue_id=row["venue_id"],
            venue_name=row["venue_name"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            forecast_time=datetime.fromisoformat(row["forecast_time"]),
            game_time=datetime.fromisoformat(row["game_time"]),
            hours_until_game=row["hours_until_game"],
            temperature=row["temperature"],
            wind_speed=row["wind_speed"],
            wind_gust=row["wind_gust"],
            wind_direction=row["wind_direction"],
            precipitation_probability=row["precipitation_probability"],
            rain_intensity=row["rain_intensity"],
            snow_intensity=row["snow_intensity"],
            humidity=row["humidity"],
            cloud_cover=row["cloud_cover"],
            weather_code=row["weather_code"],
            is_indoor=bool(row["is_indoor"]),
            confidence_factor=row["confidence_factor"],
        )

    def get_betting_forecast(
        self,
        game_id: int,
    ) -> Optional[WeatherForecast]:
        """Get FIRST captured forecast for a game — use for betting decisions.

        IMPORTANT: This returns the earliest forecast, which represents the
        information available when the betting edge was captured (Thursday).
        Using later forecasts would be lookahead bias.

        For position management or confirmation, use get_latest_forecast().

        Args:
            game_id: CFBD game ID

        Returns:
            WeatherForecast from first capture, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM forecasts
                WHERE game_id = ?
                ORDER BY forecast_time ASC
                LIMIT 1
                """,
                (game_id,),
            ).fetchone()

            if not row:
                return None

            return self._row_to_forecast(row)

    def get_latest_forecast(
        self,
        game_id: int,
        max_age_hours: int = 24,
    ) -> Optional[WeatherForecast]:
        """Get most recent forecast for a game — use for confirmation only.

        WARNING: Do NOT use this for betting decisions or backtests.
        Later forecasts are more accurate but represent information you
        wouldn't have had at bet decision time. Use get_betting_forecast()
        for actual betting.

        Args:
            game_id: CFBD game ID
            max_age_hours: Maximum age of forecast to consider valid

        Returns:
            WeatherForecast or None if not found/stale
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM forecasts
                WHERE game_id = ?
                ORDER BY forecast_time DESC
                LIMIT 1
                """,
                (game_id,),
            ).fetchone()

            if not row:
                return None

            # Check if forecast is stale (stored times are UTC)
            forecast_time = datetime.fromisoformat(row["forecast_time"])
            if forecast_time.tzinfo is None:
                forecast_time = forecast_time.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - forecast_time).total_seconds() / 3600
            if age_hours > max_age_hours:
                return None

            return self._row_to_forecast(row)

    def get_saved_forecast(
        self,
        game_id: int,
        max_age_hours: int = 24,
    ) -> Optional[WeatherForecast]:
        """Alias for get_latest_forecast() — prefer explicit method names.

        DEPRECATED: Use get_betting_forecast() for betting decisions
        or get_latest_forecast() for confirmation/position management.
        """
        return self.get_latest_forecast(game_id, max_age_hours)

    def get_all_forecasts(
        self,
        game_id: int,
    ) -> list[WeatherForecast]:
        """Get all captured forecasts for a game, ordered by capture time.

        Use this for forecast comparison (Thursday vs Saturday) or debugging.
        Returns forecasts in chronological order (oldest first).

        Args:
            game_id: CFBD game ID

        Returns:
            List of WeatherForecast objects, oldest first
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM forecasts
                WHERE game_id = ?
                ORDER BY forecast_time ASC
                """,
                (game_id,),
            ).fetchall()

            return [self._row_to_forecast(row) for row in rows]

    def save_venue(self, venue: VenueLocation) -> None:
        """Save venue location to database.

        Args:
            venue: VenueLocation to save
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO venues (
                    venue_id, name, city, state, latitude, longitude,
                    dome, elevation, timezone, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    venue.venue_id,
                    venue.name,
                    venue.city,
                    venue.state,
                    venue.latitude,
                    venue.longitude,
                    1 if venue.dome else 0,
                    venue.elevation,
                    venue.timezone,
                ),
            )

    def get_venue(self, venue_id: int) -> Optional[VenueLocation]:
        """Get venue location from database.

        Args:
            venue_id: Venue ID

        Returns:
            VenueLocation or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM venues WHERE venue_id = ?",
                (venue_id,),
            ).fetchone()

            if not row:
                return None

            return VenueLocation(
                venue_id=row["venue_id"],
                name=row["name"],
                city=row["city"],
                state=row["state"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                dome=bool(row["dome"]),
                elevation=row["elevation"],
                timezone=row["timezone"],
            )

    def is_weather_concern(self, forecast: WeatherForecast) -> bool:
        """Check if forecast indicates weather that may impact totals.

        Uses non-linear thresholds matching WeatherAdjuster:
        - Effective wind (avg of speed + gust) >= 12 mph
        - Temperature < 32°F
        - Heavy precipitation (> 60% probability with high intensity)

        Args:
            forecast: WeatherForecast to check

        Returns:
            True if weather warrants attention for totals betting
        """
        if forecast.is_indoor:
            return False

        # Wind: use average of speed and gust, threshold at 12 mph
        effective_wind = forecast.wind_speed or 0.0
        if forecast.wind_gust:
            effective_wind = (effective_wind + forecast.wind_gust) / 2
        high_wind = effective_wind >= 12.0

        freezing = (
            forecast.temperature is not None
            and forecast.temperature < 32
        )

        likely_precip = (
            forecast.precipitation_probability is not None
            and forecast.precipitation_probability > 60
        )

        return high_wind or freezing or likely_precip

    def forecast_to_weather_conditions(
        self, forecast: WeatherForecast, game_id: int
    ):
        """Convert forecast to WeatherConditions for use with WeatherAdjuster.

        Args:
            forecast: WeatherForecast from tomorrow.io
            game_id: Game ID

        Returns:
            WeatherConditions compatible with existing WeatherAdjuster
        """
        # Import here to avoid circular dependency
        from src.adjustments.weather import WeatherConditions

        # Convert rain/snow intensity from mm/hr to inches
        # (Tomorrow.io uses mm/hr even in imperial mode for intensity)
        rain_inches = (forecast.rain_intensity or 0) / 25.4
        snow_inches = (forecast.snow_intensity or 0) / 25.4

        # Map weather code to condition string
        weather_condition = self._weather_code_to_condition(forecast.weather_code)

        return WeatherConditions(
            game_id=game_id,
            home_team="",  # Not available from forecast
            away_team="",
            venue=forecast.venue_name,
            game_indoors=forecast.is_indoor,
            temperature=forecast.temperature,
            wind_speed=forecast.wind_speed,
            wind_gust=forecast.wind_gust,
            wind_direction=forecast.wind_direction,
            precipitation=rain_inches,
            snowfall=snow_inches,
            humidity=forecast.humidity,
            weather_condition=weather_condition,
        )

    def _weather_code_to_condition(self, code: Optional[int]) -> Optional[str]:
        """Convert Tomorrow.io weather code to condition string.

        Args:
            code: Tomorrow.io weather code

        Returns:
            Condition string compatible with WeatherAdjuster
        """
        if code is None:
            return None

        # Map Tomorrow.io codes to our condition strings
        # https://docs.tomorrow.io/reference/data-layers-weather-codes
        code_map = {
            0: "Unknown",
            1000: "Clear",
            1100: "Mostly Clear",
            1101: "Partly Cloudy",
            1102: "Mostly Cloudy",
            1001: "Cloudy",
            2000: "Fog",
            2100: "Light Fog",
            4000: "Drizzle",
            4001: "Rain",
            4200: "Light Rain",
            4201: "Heavy Rain",
            5000: "Snow",
            5001: "Flurries",
            5100: "Light Snow",
            5101: "Heavy Snow",
            6000: "Freezing Drizzle",
            6001: "Freezing Rain",
            6200: "Light Freezing Rain",
            6201: "Heavy Freezing Rain",
            7000: "Sleet",
            7101: "Heavy Ice Pellets",
            7102: "Light Ice Pellets",
            8000: "Thunderstorm",
        }

        return code_map.get(code, "Unknown")
