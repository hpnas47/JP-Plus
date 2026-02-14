"""The Odds API client for fetching betting lines.

This client interfaces with The Odds API (https://the-odds-api.com) to fetch
current and historical betting odds for college football games.

Credit costs:
- Current odds: 1 credit per market per region
- Historical odds: 10 credits per market per region
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# API Configuration
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_ncaaf"
DEFAULT_REGION = "us"
DEFAULT_MARKET = "spreads"


@dataclass
class OddsLine:
    """A single betting line from a sportsbook."""

    game_id: str
    sportsbook: str
    home_team: str
    away_team: str
    spread_home: float  # Negative means home is favored
    spread_away: float
    price_home: int  # American odds
    price_away: int
    commence_time: datetime
    last_update: datetime
    # Rotation numbers (Nevada standard betting IDs)
    home_rotation: Optional[int] = None  # Even number
    away_rotation: Optional[int] = None  # Odd number


@dataclass
class OddsSnapshot:
    """A snapshot of odds at a point in time."""

    timestamp: datetime
    lines: list[OddsLine]
    credits_used: int
    credits_remaining: int


class OddsAPIClient:
    """Client for The Odds API.

    Usage:
        client = OddsAPIClient(api_key="your_key")

        # Get current odds (1 credit)
        snapshot = client.get_current_odds()

        # Get historical odds (10 credits)
        snapshot = client.get_historical_odds(date="2024-09-07T12:00:00Z")
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the client.

        Args:
            api_key: The Odds API key. If not provided, reads from
                     ODDS_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key or set ODDS_API_KEY env var."
            )

        self.session = requests.Session()
        self._last_credits_remaining = None
        self._last_credits_used = None

    @property
    def credits_remaining(self) -> Optional[int]:
        """Credits remaining after last request."""
        return self._last_credits_remaining

    @property
    def credits_used(self) -> Optional[int]:
        """Credits used in last request."""
        return self._last_credits_used

    def _update_credits(self, response: requests.Response) -> None:
        """Update credit tracking from response headers."""
        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")

        if remaining is not None:
            self._last_credits_remaining = int(remaining)
        if used is not None:
            self._last_credits_used = int(used)

    def _parse_odds_response(
        self,
        data: list[dict],
        timestamp: datetime,
        market_key: str = "spreads",
    ) -> list[OddsLine]:
        """Parse API response into OddsLine objects.

        Args:
            data: Raw API response data
            timestamp: Snapshot timestamp
            market_key: Market type to parse ("spreads", "totals", "h2h").
                       Note: Only "spreads" parsing is fully implemented.
                       Other markets will filter correctly but may not parse
                       due to different outcome structures.
        """
        lines = []

        for game in data:
            game_id = game.get("id")
            home_team = game.get("home_team")
            away_team = game.get("away_team")
            commence_time_str = game.get("commence_time")
            # Rotation numbers (may be null for new/far-out games)
            home_rotation = game.get("home_rotation_number")
            away_rotation = game.get("away_rotation_number")

            if commence_time_str:
                commence_time = datetime.fromisoformat(
                    commence_time_str.replace("Z", "+00:00")
                )
            else:
                commence_time = None

            for bookmaker in game.get("bookmakers", []):
                sportsbook = bookmaker.get("key")
                last_update_str = bookmaker.get("last_update")

                if last_update_str:
                    last_update = datetime.fromisoformat(
                        last_update_str.replace("Z", "+00:00")
                    )
                else:
                    last_update = timestamp

                for market in bookmaker.get("markets", []):
                    if market.get("key") != market_key:
                        continue

                    outcomes = market.get("outcomes", [])
                    if len(outcomes) != 2:
                        continue

                    # Find home and away outcomes
                    home_outcome = None
                    away_outcome = None
                    for outcome in outcomes:
                        if outcome.get("name") == home_team:
                            home_outcome = outcome
                        elif outcome.get("name") == away_team:
                            away_outcome = outcome

                    if not home_outcome or not away_outcome:
                        continue

                    lines.append(OddsLine(
                        game_id=game_id,
                        sportsbook=sportsbook,
                        home_team=home_team,
                        away_team=away_team,
                        spread_home=home_outcome.get("point", 0),
                        spread_away=away_outcome.get("point", 0),
                        price_home=home_outcome.get("price", -110),
                        price_away=away_outcome.get("price", -110),
                        commence_time=commence_time,
                        last_update=last_update,
                        home_rotation=home_rotation,
                        away_rotation=away_rotation,
                    ))

        return lines

    def get_current_odds(
        self,
        region: str = DEFAULT_REGION,
        market: str = DEFAULT_MARKET,
        bookmakers: Optional[list[str]] = None,
    ) -> OddsSnapshot:
        """Fetch current odds for all NCAAF games.

        Cost: 1 credit per market per region.

        Args:
            region: Region code (us, uk, au, eu). Default: us
            market: Market type (spreads, h2h, totals). Default: spreads
            bookmakers: Optional list of specific bookmakers to include.
                       If provided, overrides region.

        Returns:
            OddsSnapshot with current lines.
        """
        url = f"{BASE_URL}/sports/{SPORT_KEY}/odds"

        params = {
            "apiKey": self.api_key,
            "markets": market,
            "oddsFormat": "american",
            "dateFormat": "iso",
            "includeRotationNumbers": "true",  # Request rotation numbers
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        else:
            params["regions"] = region

        logger.info(f"Fetching current NCAAF odds (region={region}, market={market})")

        response = self.session.get(url, params=params)
        self._update_credits(response)

        if response.status_code != 200:
            logger.error(f"API error: {response.status_code} - {response.text}")
            response.raise_for_status()

        data = response.json()
        timestamp = datetime.utcnow()
        lines = self._parse_odds_response(data, timestamp, market_key=market)

        logger.info(
            f"Fetched {len(lines)} lines for {len(data)} games. "
            f"Credits remaining: {self._last_credits_remaining}"
        )

        return OddsSnapshot(
            timestamp=timestamp,
            lines=lines,
            credits_used=1,  # 1 market × 1 region
            credits_remaining=self._last_credits_remaining or 0,
        )

    def get_historical_odds(
        self,
        date: str,
        region: str = DEFAULT_REGION,
        market: str = DEFAULT_MARKET,
        bookmakers: Optional[list[str]] = None,
    ) -> OddsSnapshot:
        """Fetch historical odds snapshot for a specific date/time.

        Cost: 10 credits per market per region.

        Args:
            date: ISO 8601 timestamp (e.g., "2024-09-07T12:00:00Z").
                  Returns closest snapshot at or before this time.
            region: Region code (us, uk, au, eu). Default: us
            market: Market type (spreads, h2h, totals). Default: spreads
            bookmakers: Optional list of specific bookmakers.

        Returns:
            OddsSnapshot with historical lines.
        """
        url = f"{BASE_URL}/historical/sports/{SPORT_KEY}/odds"

        params = {
            "apiKey": self.api_key,
            "date": date,
            "markets": market,
            "oddsFormat": "american",
            "dateFormat": "iso",
            "includeRotationNumbers": "true",  # Request rotation numbers
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        else:
            params["regions"] = region

        logger.info(f"Fetching historical NCAAF odds for {date}")

        response = self.session.get(url, params=params)
        self._update_credits(response)

        if response.status_code != 200:
            logger.error(f"API error: {response.status_code} - {response.text}")
            response.raise_for_status()

        result = response.json()

        # Historical endpoint wraps data differently
        timestamp_str = result.get("timestamp")
        if timestamp_str:
            timestamp = datetime.fromisoformat(
                timestamp_str.replace("Z", "+00:00")
            )
        else:
            timestamp = datetime.utcnow()

        data = result.get("data", [])
        lines = self._parse_odds_response(data, timestamp, market_key=market)

        logger.info(
            f"Fetched {len(lines)} historical lines for {len(data)} games "
            f"(snapshot: {timestamp_str}). Credits remaining: {self._last_credits_remaining}"
        )

        return OddsSnapshot(
            timestamp=timestamp,
            lines=lines,
            credits_used=10,  # 10 × 1 market × 1 region
            credits_remaining=self._last_credits_remaining or 0,
        )

    def check_quota(self) -> dict:
        """Check current API quota without using credits.

        Makes a free request to /sports endpoint.

        Returns:
            Dict with 'remaining' and 'used' credit counts.
        """
        url = f"{BASE_URL}/sports"
        params = {"apiKey": self.api_key}

        response = self.session.get(url, params=params)
        self._update_credits(response)

        return {
            "remaining": self._last_credits_remaining,
            "used": self._last_credits_used,
        }
