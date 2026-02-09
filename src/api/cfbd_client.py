"""CFBD API client with retry logic and error handling."""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import cfbd
from cfbd.rest import ApiException

from config.settings import get_settings

logger = logging.getLogger(__name__)


class DataNotAvailableError(Exception):
    """Raised when requested data is not yet available in the API."""

    pass


class APIRateLimitError(Exception):
    """Raised when API rate limit is exceeded."""

    pass


class CFBDClient:
    """Wrapper for CFBD API with retry logic and convenience methods."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the CFBD client.

        Args:
            api_key: CFBD API key. If not provided, uses settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.cfbd_api_key

        if not self.api_key:
            raise ValueError("CFBD API key is required")

        # Configure API client
        self.configuration = cfbd.Configuration()
        self.configuration.access_token = self.api_key

        # Initialize API instances
        self._games_api: Optional[cfbd.GamesApi] = None
        self._betting_api: Optional[cfbd.BettingApi] = None
        self._metrics_api: Optional[cfbd.MetricsApi] = None
        self._plays_api: Optional[cfbd.PlaysApi] = None
        self._teams_api: Optional[cfbd.TeamsApi] = None
        self._stats_api: Optional[cfbd.StatsApi] = None
        self._ratings_api: Optional[cfbd.RatingsApi] = None
        self._rankings_api: Optional[cfbd.RankingsApi] = None
        self._players_api: Optional[cfbd.PlayersApi] = None
        self._venues_api: Optional[cfbd.VenuesApi] = None

        # Retry settings
        self.max_retries = settings.max_retries
        self.retry_base_delay = 1.0  # seconds for rate limit retries

        # Session-level cache for frequently called endpoints
        self._fbs_teams_cache: dict[int, list] = {}  # year -> teams list

    @property
    def games_api(self) -> cfbd.GamesApi:
        if self._games_api is None:
            self._games_api = cfbd.GamesApi(cfbd.ApiClient(self.configuration))
        return self._games_api

    @property
    def betting_api(self) -> cfbd.BettingApi:
        if self._betting_api is None:
            self._betting_api = cfbd.BettingApi(cfbd.ApiClient(self.configuration))
        return self._betting_api

    @property
    def metrics_api(self) -> cfbd.MetricsApi:
        if self._metrics_api is None:
            self._metrics_api = cfbd.MetricsApi(cfbd.ApiClient(self.configuration))
        return self._metrics_api

    @property
    def plays_api(self) -> cfbd.PlaysApi:
        if self._plays_api is None:
            self._plays_api = cfbd.PlaysApi(cfbd.ApiClient(self.configuration))
        return self._plays_api

    @property
    def teams_api(self) -> cfbd.TeamsApi:
        if self._teams_api is None:
            self._teams_api = cfbd.TeamsApi(cfbd.ApiClient(self.configuration))
        return self._teams_api

    @property
    def stats_api(self) -> cfbd.StatsApi:
        if self._stats_api is None:
            self._stats_api = cfbd.StatsApi(cfbd.ApiClient(self.configuration))
        return self._stats_api

    @property
    def ratings_api(self) -> cfbd.RatingsApi:
        if self._ratings_api is None:
            self._ratings_api = cfbd.RatingsApi(cfbd.ApiClient(self.configuration))
        return self._ratings_api

    @property
    def rankings_api(self) -> cfbd.RankingsApi:
        if self._rankings_api is None:
            self._rankings_api = cfbd.RankingsApi(cfbd.ApiClient(self.configuration))
        return self._rankings_api

    @property
    def players_api(self) -> cfbd.PlayersApi:
        if self._players_api is None:
            self._players_api = cfbd.PlayersApi(cfbd.ApiClient(self.configuration))
        return self._players_api

    @property
    def venues_api(self) -> cfbd.VenuesApi:
        if self._venues_api is None:
            self._venues_api = cfbd.VenuesApi(cfbd.ApiClient(self.configuration))
        return self._venues_api

    def _call_with_retry(self, func: callable, *args, **kwargs) -> Any:
        """Execute API call with exponential backoff retry on rate limits."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except ApiException as e:
                if e.status == 429:  # Rate limited
                    delay = self.retry_base_delay * (2**attempt)
                    logger.warning(f"Rate limited. Waiting {delay}s before retry...")
                    time.sleep(delay)
                    last_exception = APIRateLimitError(str(e))
                elif e.status == 404:
                    # Data not found - might not be available yet
                    raise DataNotAvailableError(f"Data not found: {e}")
                else:
                    raise
            except Exception as e:
                logger.error(f"API call failed: {e}")
                raise

        raise last_exception or Exception("Max retries exceeded")

    def get_games(
        self,
        year: int,
        week: Optional[int] = None,
        season_type: str = "regular",
        team: Optional[str] = None,
        classification: str = "fbs",
    ) -> list:
        """Get games for a given year/week.

        Args:
            year: Season year
            week: Week number (optional)
            season_type: 'regular' or 'postseason'
            team: Filter by team name (optional)
            classification: 'fbs' or 'fcs'

        Returns:
            List of game objects
        """
        kwargs = {"year": year, "season_type": season_type}
        if week is not None:
            kwargs["week"] = week
        if team is not None:
            kwargs["team"] = team
        if classification is not None:
            kwargs["classification"] = classification

        return self._call_with_retry(self.games_api.get_games, **kwargs)

    def get_game_results(
        self,
        year: int,
        week: Optional[int] = None,
        season_type: str = "regular",
    ) -> list:
        """Get completed game results with scores."""
        games = self.get_games(year, week, season_type)
        # Filter to completed games
        return [g for g in games if g.home_points is not None]

    def get_betting_lines(
        self,
        year: int,
        week: Optional[int] = None,
        season_type: str = "regular",
        team: Optional[str] = None,
    ) -> list:
        """Get betting lines for games.

        Args:
            year: Season year
            week: Week number (optional)
            team: Filter by team (optional)

        Returns:
            List of betting line objects
        """
        kwargs = {"year": year, "season_type": season_type}
        if week is not None:
            kwargs["week"] = week
        if team is not None:
            kwargs["team"] = team

        return self._call_with_retry(self.betting_api.get_lines, **kwargs)

    def get_advanced_box_score(self, game_id: int) -> Any:
        """Get advanced box score for a specific game."""
        return self._call_with_retry(
            self.games_api.get_advanced_box_score, game_id=game_id
        )

    def get_team_game_stats(
        self,
        year: int,
        week: Optional[int] = None,
        season_type: str = "regular",
        team: Optional[str] = None,
    ) -> list:
        """Get team stats for games."""
        kwargs = {"year": year, "season_type": season_type}
        if week is not None:
            kwargs["week"] = week
        if team is not None:
            kwargs["team"] = team

        return self._call_with_retry(self.games_api.get_team_game_stats, **kwargs)

    def get_ppa_games(
        self,
        year: int,
        week: Optional[int] = None,
        team: Optional[str] = None,
        season_type: str = "regular",
    ) -> list:
        """Get Predicted Points Added (PPA) data by game.

        Args:
            year: Season year
            week: Week number (optional)
            team: Filter by team (optional)
            season_type: 'regular' or 'postseason'

        Returns:
            List of PPA game objects
        """
        kwargs = {"year": year, "season_type": season_type}
        if week is not None:
            kwargs["week"] = week
        if team is not None:
            kwargs["team"] = team

        return self._call_with_retry(self.metrics_api.get_game_ppa, **kwargs)

    def get_ppa_season(
        self,
        year: int,
        team: Optional[str] = None,
    ) -> list:
        """Get season-level PPA data."""
        kwargs = {"year": year}
        if team is not None:
            kwargs["team"] = team

        return self._call_with_retry(self.metrics_api.get_team_ppa, **kwargs)

    def get_plays(
        self,
        year: int,
        week: int,
        team: Optional[str] = None,
        season_type: str = "regular",
    ) -> list:
        """Get play-by-play data.

        Args:
            year: Season year
            week: Week number
            team: Filter by team (optional)
            season_type: 'regular' or 'postseason'

        Returns:
            List of play objects
        """
        kwargs = {"year": year, "week": week, "season_type": season_type}
        if team is not None:
            kwargs["team"] = team

        return self._call_with_retry(self.plays_api.get_plays, **kwargs)

    def get_drive_data(
        self,
        year: int,
        week: Optional[int] = None,
        team: Optional[str] = None,
        season_type: str = "regular",
    ) -> list:
        """Get drive-level data.

        Args:
            year: Season year
            week: Week number (optional)
            team: Filter by team (optional)
            season_type: 'regular' or 'postseason'

        Returns:
            List of drive objects
        """
        kwargs = {"year": year, "season_type": season_type}
        if week is not None:
            kwargs["week"] = week
        if team is not None:
            kwargs["team"] = team

        return self._call_with_retry(self.games_api.get_drives, **kwargs)

    def get_fbs_teams(self, year: Optional[int] = None) -> list:
        """Get list of FBS teams.

        Uses session-level caching to avoid redundant API calls.
        Same (year) within a session returns cached result.

        Args:
            year: Season year (None for current teams)

        Returns:
            List of team objects
        """
        # Use year as cache key (None for current season)
        cache_key = year if year is not None else -1

        if cache_key in self._fbs_teams_cache:
            logger.debug(f"Cache hit: get_fbs_teams(year={year})")
            return self._fbs_teams_cache[cache_key]

        kwargs = {}
        if year is not None:
            kwargs["year"] = year

        result = self._call_with_retry(self.teams_api.get_fbs_teams, **kwargs)
        self._fbs_teams_cache[cache_key] = result
        logger.debug(f"Cache miss: get_fbs_teams(year={year}), cached {len(result)} teams")

        return result

    def get_team_records(self, year: int, team: Optional[str] = None) -> list:
        """Get team records for a season."""
        kwargs = {"year": year}
        if team is not None:
            kwargs["team"] = team
        return self._call_with_retry(self.games_api.get_team_records, **kwargs)

    def get_pregame_win_probabilities(
        self,
        year: int,
        week: Optional[int] = None,
        team: Optional[str] = None,
        season_type: str = "regular",
    ) -> list:
        """Get pregame win probability predictions."""
        kwargs = {"year": year, "season_type": season_type}
        if week is not None:
            kwargs["week"] = week
        if team is not None:
            kwargs["team"] = team

        return self._call_with_retry(self.metrics_api.get_pregame_win_probabilities, **kwargs)

    def get_calendar(self, year: int) -> list:
        """Get season calendar with week dates."""
        return self._call_with_retry(self.games_api.get_calendar, year=year)

    def get_current_week(self, year: int) -> int:
        """Determine current week based on calendar and today's date."""
        calendar = self.get_calendar(year)
        today = datetime.now().date()

        for week_info in calendar:
            # Parse week start/end dates
            start = datetime.strptime(
                week_info.first_game_start[:10], "%Y-%m-%d"
            ).date()
            end = datetime.strptime(week_info.last_game_start[:10], "%Y-%m-%d").date()

            # Add buffer for Sunday morning runs
            end_with_buffer = end + timedelta(days=2)

            if start <= today <= end_with_buffer:
                return week_info.week

        # Default to last week if past season
        return calendar[-1].week if calendar else 1

    def check_data_availability(self, year: int, week: int) -> bool:
        """Check if data for a given week is available.

        This checks if game results have been populated for the specified week.
        Useful for Sunday morning runs when Saturday data may still be loading.

        Returns:
            True if data appears complete, False otherwise
        """
        try:
            games = self.get_games(year, week)
            if not games:
                return False

            # Check if at least some games have scores
            completed = [g for g in games if g.home_points is not None]

            # Consider data available if >80% of games have scores
            return len(completed) / len(games) > 0.8
        except (DataNotAvailableError, ApiException):
            return False

    def wait_for_data(
        self,
        year: int,
        week: int,
        max_wait_hours: float = 8.0,
        check_interval: float = 300.0,
    ) -> bool:
        """Wait for week data to become available.

        Args:
            year: Season year
            week: Week number
            max_wait_hours: Maximum time to wait in hours
            check_interval: Time between checks in seconds

        Returns:
            True if data became available, False if timed out
        """
        settings = get_settings()
        check_interval = check_interval or settings.data_check_interval
        max_wait_seconds = max_wait_hours * 3600

        start_time = time.time()

        while time.time() - start_time < max_wait_seconds:
            if self.check_data_availability(year, week):
                logger.info(f"Data available for {year} week {week}")
                return True

            elapsed = (time.time() - start_time) / 60
            logger.info(
                f"Data not ready for {year} week {week}. "
                f"Waited {elapsed:.1f} min. Checking again in {check_interval/60:.1f} min..."
            )
            time.sleep(check_interval)

        logger.warning(f"Timed out waiting for {year} week {week} data")
        return False

    def get_fpi_ratings(self, year: int) -> list:
        """Get ESPN FPI ratings for teams."""
        return self._call_with_retry(self.ratings_api.get_fpi, year=year)

    def get_rankings(
        self,
        year: int,
        week: Optional[int] = None,
        season_type: str = "regular",
    ) -> list:
        """Get poll rankings (AP, Coaches, CFP, etc.) for a season/week.

        Args:
            year: Season year
            week: Week number (if None, returns all weeks)
            season_type: Season type ("regular", "postseason", "both")

        Returns:
            List of poll week objects containing poll rankings
        """
        kwargs = {"year": year, "season_type": season_type}
        if week is not None:
            kwargs["week"] = week
        return self._call_with_retry(self.rankings_api.get_rankings, **kwargs)

    def get_team_talent(self, year: int) -> list:
        """Get team talent composite rankings."""
        return self._call_with_retry(self.teams_api.get_talent, year=year)

    def get_havoc_stats(
        self,
        year: int,
        week: Optional[int] = None,
        team: Optional[str] = None,
        season_type: str = "regular",
    ) -> list:
        """Get havoc statistics (sacks, TFLs, PBUs) by game.

        Args:
            year: Season year
            week: Week number (optional)
            team: Filter by team (optional)
            season_type: 'regular' or 'postseason'

        Returns:
            List of GameHavocStats objects with:
            - team, opponent, gameId, week
            - offense.havocRate, offense.frontSevenHavocRate, offense.dbHavocRate
            - defense.havocRate, defense.frontSevenHavocRate, defense.dbHavocRate
        """
        kwargs = {"year": year, "season_type": season_type}
        if week is not None:
            kwargs["week"] = week
        if team is not None:
            kwargs["team"] = team

        return self._call_with_retry(self.stats_api.get_game_havoc_stats, **kwargs)

    def get_weather(
        self,
        year: int,
        week: Optional[int] = None,
        team: Optional[str] = None,
        conference: Optional[str] = None,
        season_type: str = "regular",
        game_id: Optional[int] = None,
    ) -> list:
        """Get weather data for games.

        Weather data is essential for totals prediction. Key factors:
        - Wind speed: 15+ mph significantly affects passing game
        - Temperature: Cold weather (<40°F) reduces scoring
        - Precipitation: Rain affects ball security and passing
        - Indoor games: No weather adjustment needed

        Args:
            year: Season year
            week: Week number (optional)
            team: Filter by team (optional)
            conference: Filter by conference (optional)
            season_type: 'regular' or 'postseason'
            game_id: Specific game ID (optional)

        Returns:
            List of GameWeather objects with:
            - id, season, week, season_type, start_time
            - home_team, away_team, venue, venue_id
            - game_indoors: Boolean for dome games
            - temperature: Fahrenheit
            - wind_speed: MPH
            - wind_direction: Degrees (0-360)
            - precipitation: Inches
            - snowfall: Inches
            - humidity: Percentage (0-100)
            - dew_point: Fahrenheit
            - pressure: Millibars
            - weather_condition: Text (Clear, Cloudy, Rain, etc.)
            - weather_condition_code: Numeric code
        """
        kwargs = {"year": year, "season_type": season_type}
        if week is not None:
            kwargs["week"] = week
        if team is not None:
            kwargs["team"] = team
        if conference is not None:
            kwargs["conference"] = conference
        if game_id is not None:
            kwargs["game_id"] = game_id

        return self._call_with_retry(self.games_api.get_weather, **kwargs)

    def get_sp_ratings(self, year: int) -> list:
        """Get SP+ ratings for a given year.

        Args:
            year: Season year

        Returns:
            List of SP+ rating objects
        """
        return self._call_with_retry(self.ratings_api.get_sp, year=year)

    def get_transfer_portal(self, year: int) -> list:
        """Get transfer portal entries for a given year.

        Args:
            year: Season year (transfers FOR this year)

        Returns:
            List of transfer portal entry objects
        """
        return self._call_with_retry(self.players_api.get_transfer_portal, year=year)

    def get_player_usage(self, year: int) -> list:
        """Get player usage stats (includes PPA) for a given year.

        Args:
            year: Season year

        Returns:
            List of player usage objects
        """
        return self._call_with_retry(self.players_api.get_player_usage, year=year)

    def get_returning_production(self, year: int, team: Optional[str] = None) -> list:
        """Get returning production metrics for a given year.

        Args:
            year: Season year
            team: Filter by team (optional)

        Returns:
            List of returning production objects
        """
        kwargs = {"year": year}
        if team is not None:
            kwargs["team"] = team
        return self._call_with_retry(
            self.players_api.get_returning_production, **kwargs
        )

    def get_venues(self) -> list:
        """Get all college football venues with location data.

        Returns:
            List of venue objects with:
            - id: Venue ID
            - name: Stadium name
            - city, state, zip, country_code: Location
            - location: Dict with latitude/longitude coordinates
            - elevation: Elevation in feet
            - capacity: Seating capacity
            - year_constructed: Year built
            - grass: Boolean for natural grass (vs turf)
            - dome: Boolean for indoor stadium
            - timezone: Timezone string
        """
        return self._call_with_retry(self.venues_api.get_venues)
