"""Altitude adjustments for high-elevation venues."""

import logging
from typing import Optional

from config.teams import (
    ALTITUDE_VENUES,
    HIGH_ALTITUDE_TEAMS,
    get_altitude_adjustment,
    get_team_altitude,
    normalize_team_name,
    safe_is_high_altitude,
)

logger = logging.getLogger(__name__)


class AltitudeAdjuster:
    """
    Calculate altitude adjustments for games at high-elevation venues.

    Teams from sea level visiting high-altitude venues face:
    - Reduced oxygen availability
    - Ball travels differently (kicks, throws)
    - Increased fatigue

    Teams already acclimatized to altitude don't face these challenges.
    """

    # Elevation thresholds (feet)
    MODERATE_ALTITUDE = 3500
    HIGH_ALTITUDE = 5000
    VERY_HIGH_ALTITUDE = 6000

    def __init__(self):
        """Initialize altitude adjuster."""
        self.venue_adjustments = ALTITUDE_VENUES.copy()

    def get_venue_elevation(self, team: str) -> int:
        """Get venue elevation for a team.

        Args:
            team: Team name

        Returns:
            Elevation in feet (0 for sea-level teams)
        """
        return get_team_altitude(team)

    def is_high_altitude_team(self, team: str) -> bool:
        """Check if team plays at high altitude.

        Uses team name normalization (P2.13) to handle CFBD naming variations.

        Args:
            team: Team name

        Returns:
            True if team's home venue is above 3000 feet
        """
        return safe_is_high_altitude(team)

    def get_altitude_adjustment(
        self,
        home_team: str,
        away_team: str,
    ) -> float:
        """Calculate altitude adjustment for a matchup.

        Adjustment favors the home team when:
        - Home venue is at high altitude
        - Away team is from sea level

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Points adjustment (positive = favors home team)
        """
        return get_altitude_adjustment(home_team, away_team)

    def get_detailed_adjustment(
        self,
        home_team: str,
        away_team: str,
    ) -> tuple[float, dict]:
        """Get altitude adjustment with detailed breakdown.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Tuple of (adjustment, breakdown dict)
        """
        home_elevation = self.get_venue_elevation(home_team)
        away_elevation = self.get_venue_elevation(away_team)

        # Get base adjustment
        adjustment = self.get_altitude_adjustment(home_team, away_team)

        # Determine reason for adjustment (or lack thereof)
        if adjustment == 0:
            if home_elevation < self.MODERATE_ALTITUDE:
                reason = "Home venue at low altitude"
            elif away_team in HIGH_ALTITUDE_TEAMS:
                reason = "Away team acclimatized to altitude"
            else:
                reason = "No altitude advantage applicable"
        else:
            reason = f"Sea-level team visiting {home_elevation}ft elevation"

        breakdown = {
            "home_team": home_team,
            "home_elevation_ft": home_elevation,
            "away_team": away_team,
            "away_elevation_ft": away_elevation,
            "away_is_high_altitude": away_team in HIGH_ALTITUDE_TEAMS,
            "adjustment": adjustment,
            "reason": reason,
        }

        return adjustment, breakdown

    def get_all_high_altitude_venues(self) -> list[dict]:
        """Get list of all high-altitude venues with details.

        Returns:
            List of venue dictionaries sorted by elevation
        """
        venues = [
            {
                "team": team,
                "elevation": info["elevation"],
                "adjustment": info["adjustment"],
            }
            for team, info in self.venue_adjustments.items()
        ]

        return sorted(venues, key=lambda x: x["elevation"], reverse=True)

    def estimate_performance_impact(
        self,
        visitor_elevation: int,
        host_elevation: int,
    ) -> dict:
        """Estimate physiological impacts of altitude difference.

        This provides context on why altitude matters.

        Args:
            visitor_elevation: Visitor's home elevation (feet)
            host_elevation: Host venue elevation (feet)

        Returns:
            Dict with estimated impacts
        """
        elevation_diff = host_elevation - visitor_elevation

        if elevation_diff <= 0:
            return {
                "oxygen_reduction": "0%",
                "fatigue_increase": "None",
                "ball_carry_increase": "0%",
                "recommendation": "No altitude concerns",
            }

        # Rough estimates based on sports science
        # Oxygen availability decreases ~3% per 1000 feet above sea level
        oxygen_reduction = min((host_elevation / 1000) * 3, 25)

        # Fatigue increase
        if elevation_diff < 2000:
            fatigue = "Minimal"
        elif elevation_diff < 4000:
            fatigue = "Moderate"
        else:
            fatigue = "Significant"

        # Ball carries farther in thin air (~2% per 1000 feet for kicks)
        ball_carry = min((host_elevation / 1000) * 2, 15)

        # Recommendation
        if elevation_diff < 2000:
            rec = "Minor altitude adjustment needed"
        elif elevation_diff < 4000:
            rec = "Consider extra conditioning, hydration focus"
        else:
            rec = "Significant altitude challenge, early arrival recommended"

        return {
            "oxygen_reduction": f"{oxygen_reduction:.1f}%",
            "fatigue_increase": fatigue,
            "ball_carry_increase": f"{ball_carry:.1f}%",
            "recommendation": rec,
        }
