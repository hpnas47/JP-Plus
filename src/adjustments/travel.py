"""Travel adjustments based on distance and timezone changes."""

import logging
from typing import Optional

from geopy.distance import geodesic

from config.settings import get_settings
from config.teams import TEAM_LOCATIONS, get_timezone_difference

logger = logging.getLogger(__name__)


class TravelAdjuster:
    """
    Calculate travel-related adjustments.

    Factors:
    - Timezone crossings: ~0.5 points per timezone crossed
    - Long-distance travel: Additional fatigue factor for very long trips
    - West-to-east travel: Slightly harder than east-to-west
    """

    # Distance thresholds (miles)
    SHORT_TRIP = 300
    MEDIUM_TRIP = 1000
    LONG_TRIP = 2000

    def __init__(
        self,
        timezone_adjustment: Optional[float] = None,
    ):
        """Initialize travel adjuster.

        Args:
            timezone_adjustment: Points per timezone crossed
        """
        settings = get_settings()
        self.timezone_adjustment = (
            timezone_adjustment
            if timezone_adjustment is not None
            else settings.timezone_adjustment
        )

    def get_distance(
        self,
        team_a: str,
        team_b: str,
    ) -> Optional[float]:
        """Get distance between two teams' home venues in miles.

        Args:
            team_a: First team
            team_b: Second team

        Returns:
            Distance in miles or None if location data unavailable
        """
        loc_a = TEAM_LOCATIONS.get(team_a)
        loc_b = TEAM_LOCATIONS.get(team_b)

        if loc_a is None or loc_b is None:
            return None

        point_a = (loc_a["lat"], loc_a["lon"])
        point_b = (loc_b["lat"], loc_b["lon"])

        return geodesic(point_a, point_b).miles

    def get_timezone_adjustment(
        self,
        away_team: str,
        home_team: str,
    ) -> float:
        """Calculate timezone adjustment for away team traveling.

        Args:
            away_team: Traveling team
            home_team: Host team

        Returns:
            Points adjustment (negative = penalty for away team)
        """
        tz_diff = get_timezone_difference(away_team, home_team)

        if tz_diff == 0:
            return 0.0

        # Base adjustment per timezone
        base_adj = tz_diff * self.timezone_adjustment

        # West-to-east is harder (losing time)
        away_loc = TEAM_LOCATIONS.get(away_team)
        home_loc = TEAM_LOCATIONS.get(home_team)

        if away_loc and home_loc:
            # Traveling east = going to smaller (less negative) longitude
            if away_loc["lon"] < home_loc["lon"]:
                # Traveling west (easier)
                return -base_adj * 0.8
            else:
                # Traveling east (harder)
                return -base_adj

        return -base_adj

    def get_distance_adjustment(
        self,
        away_team: str,
        home_team: str,
    ) -> float:
        """Calculate adjustment based on travel distance.

        Args:
            away_team: Traveling team
            home_team: Host team

        Returns:
            Points adjustment (negative = penalty for away team)
        """
        distance = self.get_distance(away_team, home_team)

        if distance is None:
            return 0.0

        # No adjustment for short trips
        if distance < self.SHORT_TRIP:
            return 0.0

        # Small adjustment for medium trips
        if distance < self.MEDIUM_TRIP:
            return -0.25

        # Moderate adjustment for long trips
        if distance < self.LONG_TRIP:
            return -0.5

        # Larger adjustment for very long trips (Hawaii, cross-country)
        return -1.0

    def get_total_travel_adjustment(
        self,
        home_team: str,
        away_team: str,
    ) -> tuple[float, dict]:
        """Get total travel adjustment for a matchup.

        The adjustment is from the home team's perspective (positive = favors home).

        Args:
            home_team: Home team
            away_team: Away team

        Returns:
            Tuple of (total adjustment favoring home, breakdown dict)
        """
        tz_adj = self.get_timezone_adjustment(away_team, home_team)
        dist_adj = self.get_distance_adjustment(away_team, home_team)

        # These are penalties on away team, so flip sign for home perspective
        total = -(tz_adj + dist_adj)

        distance = self.get_distance(away_team, home_team)
        tz_diff = get_timezone_difference(away_team, home_team)

        breakdown = {
            "distance_miles": distance,
            "timezone_diff": tz_diff,
            "timezone_penalty": -tz_adj,  # As home team advantage
            "distance_penalty": -dist_adj,  # As home team advantage
            "total_home_advantage": total,
        }

        return total, breakdown

    def get_hawaii_adjustment(
        self,
        away_team: str,
        home_team: str,
    ) -> float:
        """Special case adjustment for games involving Hawaii.

        Hawaii games have unique travel challenges:
        - 5-6 hour time difference
        - 2500+ miles from nearest continental team
        - Jet lag recovery time

        Args:
            away_team: Traveling team
            home_team: Host team

        Returns:
            Additional points adjustment (negative = penalty for traveler)
        """
        if home_team == "Hawaii" and away_team != "Hawaii":
            # Mainland team traveling to Hawaii
            return -2.0  # Significant travel penalty
        elif away_team == "Hawaii" and home_team != "Hawaii":
            # Hawaii traveling to mainland
            return -1.5  # Slightly less due to familiarity with travel
        return 0.0
