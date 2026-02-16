"""CFBD API client with retry logic and error handling."""

import logging
import time
import warnings
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


def _parse_date_safe(value: Any) -> Optional[datetime]:
    """Parse a date value that may be None, a datetime, or an ISO string.

    Returns None if parsing fails (caller should skip/warn).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        # dateutil-style parse for ISO strings (handles 'Z', offsets, etc.)
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


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

        # Single shared ApiClient for all API instances (avoids per-property overhead)
        self._api_client = cfbd.ApiClient(self.configuration)

        # Lazy-init API instances (all share _api_client)
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
        self._coaches_api: Optional[cfbd.CoachesApi] = None
        self._drives_api: Optional[cfbd.DrivesApi] = None  # cfbd 5.x moved drives here

        # Retry settings
        self.max_retries = settings.max_retries
        self.retry_base_delay = 1.0  # seconds for rate limit retries

        # Session-level cache for frequently called endpoints
        self._fbs_teams_cache: dict[int, list] = {}  # year -> teams list

    # --- Lazy API accessors (shared ApiClient) ---

    @property
    def games_api(self) -> cfbd.GamesApi:
        if self._games_api is None:
            self._games_api = cfbd.GamesApi(self._api_client)
        return self._games_api

    @property
    def betting_api(self) -> cfbd.BettingApi:
        if self._betting_api is None:
            self._betting_api = cfbd.BettingApi(self._api_client)
        return self._betting_api

    @property
    def metrics_api(self) -> cfbd.MetricsApi:
        if self._metrics_api is None:
            self._metrics_api = cfbd.MetricsApi(self._api_client)
        return self._metrics_api

    @property
    def plays_api(self) -> cfbd.PlaysApi:
        if self._plays_api is None:
            self._plays_api = cfbd.PlaysApi(self._api_client)
        return self._plays_api

    @property
    def teams_api(self) -> cfbd.TeamsApi:
        if self._teams_api is None:
            self._teams_api = cfbd.TeamsApi(self._api_client)
        return self._teams_api

    @property
    def stats_api(self) -> cfbd.StatsApi:
        if self._stats_api is None:
            self._stats_api = cfbd.StatsApi(self._api_client)
        return self._stats_api

    @property
    def ratings_api(self) -> cfbd.RatingsApi:
        if self._ratings_api is None:
            self._ratings_api = cfbd.RatingsApi(self._api_client)
        return self._ratings_api

    @property
    def rankings_api(self) -> cfbd.RankingsApi:
        if self._rankings_api is None:
            self._rankings_api = cfbd.RankingsApi(self._api_client)
        return self._rankings_api

    @property
    def players_api(self) -> cfbd.PlayersApi:
        if self._players_api is None:
            self._players_api = cfbd.PlayersApi(self._api_client)
        return self._players_api

    @property
    def venues_api(self) -> cfbd.VenuesApi:
        if self._venues_api is None:
            self._venues_api = cfbd.VenuesApi(self._api_client)
        return self._venues_api

    @property
    def coaches_api(self) -> cfbd.CoachesApi:
        if self._coaches_api is None:
            self._coaches_api = cfbd.CoachesApi(self._api_client)
        return self._coaches_api

    @property
    def drives_api(self) -> cfbd.DrivesApi:
        """DrivesApi — cfbd 5.x moved get_drives here from GamesApi."""
        if self._drives_api is None:
            self._drives_api = cfbd.DrivesApi(self._api_client)
        return self._drives_api

    def _call_with_retry(self, func: callable, *args, **kwargs) -> Any:
        """Execute API call with exponential backoff retry on rate limits.

        Makes up to (max_retries + 1) total attempts. On 429 responses,
        respects the Retry-After header if present, otherwise uses exponential
        backoff. Raises APIRateLimitError (not bare Exception) after exhaustion.
        """
        last_exception: Optional[ApiException] = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except ApiException as e:
                if e.status == 429:  # Rate limited
                    last_exception = e
                    # Respect Retry-After header if present, else exponential backoff
                    retry_after = None
                    if hasattr(e, "headers") and e.headers:
                        retry_after = e.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            delay = float(retry_after)
                        except (ValueError, TypeError):
                            delay = self.retry_base_delay * (2 ** attempt)
                    else:
                        delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Rate limited (attempt %d/%d). Waiting %.1fs before retry...",
                        attempt + 1, self.max_retries + 1, delay,
                    )
                    time.sleep(delay)
                elif e.status == 404:
                    raise DataNotAvailableError(f"Data not found: {e}") from e
                else:
                    raise
            except Exception as e:
                logger.error(f"API call failed: {e}")
                raise

        # Exhausted all retries — raise specific error with original context
        raise APIRateLimitError(
            f"Rate limit exceeded after {self.max_retries + 1} attempts"
        ) from last_exception

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
        # cfbd 5.x param is 'id', not 'game_id'
        return self._call_with_retry(
            self.games_api.get_advanced_box_score, id=game_id
        )

    def get_team_game_stats(
        self,
        year: int,
        week: Optional[int] = None,
        season_type: str = "regular",
        team: Optional[str] = None,
    ) -> list:
        """Get team stats for games.

        Uses StatsApi.get_advanced_game_stats (cfbd 5.x moved this from GamesApi).
        """
        kwargs = {"year": year, "season_type": season_type}
        if week is not None:
            kwargs["week"] = week
        if team is not None:
            kwargs["team"] = team

        return self._call_with_retry(self.stats_api.get_advanced_game_stats, **kwargs)

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

        # cfbd 5.x renamed get_game_ppa -> get_predicted_points_added_by_game
        return self._call_with_retry(
            self.metrics_api.get_predicted_points_added_by_game, **kwargs
        )

    def get_ppa_season(
        self,
        year: int,
        team: Optional[str] = None,
    ) -> list:
        """Get season-level PPA data."""
        kwargs = {"year": year}
        if team is not None:
            kwargs["team"] = team

        # cfbd 5.x renamed get_team_ppa -> get_predicted_points_added_by_team
        return self._call_with_retry(
            self.metrics_api.get_predicted_points_added_by_team, **kwargs
        )

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

        # cfbd 5.x moved get_drives from GamesApi to DrivesApi
        return self._call_with_retry(self.drives_api.get_drives, **kwargs)

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
        # cfbd 5.x renamed get_team_records -> get_records
        return self._call_with_retry(self.games_api.get_records, **kwargs)

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
        """Determine current week based on calendar and today's date.

        Handles first_game_start/last_game_start being None, datetime objects,
        or ISO strings. Skips unparseable entries with a warning.
        """
        calendar = self.get_calendar(year)
        if not calendar:
            return 1

        today = datetime.now().date()

        for week_info in calendar:
            start_dt = _parse_date_safe(week_info.first_game_start)
            end_dt = _parse_date_safe(week_info.last_game_start)

            if start_dt is None or end_dt is None:
                # Skip weeks with missing/unparseable dates
                warnings.warn(
                    f"Skipping week {week_info.week}: unparseable dates "
                    f"(first={week_info.first_game_start!r}, last={week_info.last_game_start!r})"
                )
                continue

            start = start_dt.date()
            end = end_dt.date()

            # Add buffer for Sunday morning runs
            end_with_buffer = end + timedelta(days=2)

            if start <= today <= end_with_buffer:
                return week_info.week

        # Default to last week if past season
        return calendar[-1].week

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
        check_interval: Optional[float] = None,
    ) -> bool:
        """Wait for week data to become available.

        Args:
            year: Season year
            week: Week number
            max_wait_hours: Maximum time to wait in hours (must be > 0)
            check_interval: Time between checks in seconds (must be > 0).
                            If None, uses settings.data_check_interval.

        Returns:
            True if data became available, False if timed out

        Raises:
            DataNotAvailableError: If week has no scheduled games (invalid week)
            ValueError: If check_interval or max_wait_hours are invalid
        """
        settings = get_settings()
        # Use settings default only when caller didn't provide a value
        if check_interval is None:
            check_interval = getattr(settings, "data_check_interval", 300.0)

        # Validate parameters — don't allow sleep(0), sleep(negative), or sleep(None)
        if not isinstance(check_interval, (int, float)) or check_interval <= 0:
            raise ValueError(
                f"check_interval must be a positive number, got {check_interval!r}"
            )
        if not isinstance(max_wait_hours, (int, float)) or max_wait_hours <= 0:
            raise ValueError(
                f"max_wait_hours must be a positive number, got {max_wait_hours!r}"
            )

        max_wait_seconds = max_wait_hours * 3600

        # P0: Fail fast if week has no games at all (prevents infinite polling)
        # This catches requests for weeks beyond the season (e.g., week 19+)
        try:
            games = self.get_games(year, week)
            if not games:
                logger.error(
                    f"No games scheduled for {year} week {week}. "
                    "This week may not exist (CFB season ends at week 17)."
                )
                raise DataNotAvailableError(
                    f"No games found for {year} week {week}. "
                    "The CFB season typically ends at week 17 (national championship)."
                )
        except ApiException as e:
            logger.error(f"API error checking {year} week {week}: {e}")
            raise DataNotAvailableError(f"Cannot fetch games for {year} week {week}: {e}") from e

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

    def get_coaches(
        self,
        year: Optional[int] = None,
        team: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> list:
        """Get head coach information and records.

        Args:
            year: Filter by specific season
            team: Filter by team name
            min_year: Filter by minimum year
            max_year: Filter by maximum year

        Returns:
            List of Coach objects with:
            - first_name, last_name: Coach name
            - seasons: List of season records with team, year, wins, losses, etc.
        """
        kwargs = {}
        if year is not None:
            kwargs["year"] = year
        if team is not None:
            kwargs["team"] = team
        if min_year is not None:
            kwargs["min_year"] = min_year
        if max_year is not None:
            kwargs["max_year"] = max_year

        return self._call_with_retry(self.coaches_api.get_coaches, **kwargs)
