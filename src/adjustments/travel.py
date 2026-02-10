"""Travel adjustments based on distance and timezone changes.

DST Policy (P2.12):
Timezone offsets in config/teams.py represent effective differences during DST,
since ~70% of CFB regular season is during DST (weeks 0-10). Arizona and Hawaii
(which don't observe DST) use DST-era offsets by default.

DST-Aware Mode:
When game_date is provided to timezone functions, offsets are adjusted for teams
that don't observe DST (Arizona, Arizona State, Hawaii):
- During DST (Mar-Nov): Arizona/ASU = 3 hrs behind ET, Hawaii = 6 hrs
- After DST ends: Arizona/ASU = 2 hrs behind ET, Hawaii = 5 hrs

If game_date is not provided, the default (DST-era) offsets are used.
"""

import logging
from functools import lru_cache
from typing import Optional

from geopy.distance import geodesic

from config.settings import get_settings
from config.teams import (
    TEAM_LOCATIONS,
    get_timezone_difference,
    get_directed_timezone_change,
    safe_get_location,
    normalize_team_name,
)

logger = logging.getLogger(__name__)

# Track teams we've already warned about missing location data (avoid log spam)
_warned_missing_location: set[str] = set()


def _warn_missing_location_once(team: str, normalized: str) -> None:
    """Log a warning for missing location data, but only once per team."""
    if normalized not in _warned_missing_location:
        _warned_missing_location.add(normalized)
        logger.warning(
            f"Team '{team}' (normalized: '{normalized}') missing location data in TEAM_LOCATIONS. "
            f"Distance-based travel adjustment unavailable."
        )


def clear_missing_location_warnings() -> None:
    """Clear the warned teams set. Use for testing or new backtest runs."""
    _warned_missing_location.clear()


# Cache sizing: 136 FBS teams = 9,180 unique unordered pairs. With symmetric
# key ordering, we need at most ~10K entries (including FCS opponents). Using
# 8192 (power of 2) covers nearly all pairs while bounding memory in long-running
# processes. LRU eviction handles overflow gracefully.
@lru_cache(maxsize=8192)
def _cached_geodesic_distance_impl(team_a: str, team_b: str) -> Optional[float]:
    """Internal implementation with normalized, ordered keys. Use wrapper below."""
    loc_a = safe_get_location(team_a)
    loc_b = safe_get_location(team_b)

    if loc_a is None or loc_b is None:
        return None

    point_a = (loc_a["lat"], loc_a["lon"])
    point_b = (loc_b["lat"], loc_b["lon"])

    return geodesic(point_a, point_b).miles


def _cached_geodesic_distance(
    team_a: str,
    team_b: str,
    warn_on_missing: bool = True,
) -> Optional[float]:
    """Cached geodesic distance between two teams' venues (miles).

    Normalizes team names and orders them lexicographically so (A,B) and (B,A)
    hit the same cache entry (distance is symmetric). Module-level cache ensures
    distances are computed once per team pair across the entire backtest run.

    Args:
        team_a: First team name
        team_b: Second team name
        warn_on_missing: If True, log warning (once per team) when location data missing

    Returns:
        Distance in miles, or None if location data unavailable for either team
    """
    # Normalize team names to handle CFBD naming variations
    norm_a = normalize_team_name(team_a)
    norm_b = normalize_team_name(team_b)

    # Check for missing location data and warn (before ordering for cache)
    if warn_on_missing:
        loc_a = safe_get_location(norm_a)
        loc_b = safe_get_location(norm_b)
        if loc_a is None:
            _warn_missing_location_once(team_a, norm_a)
        if loc_b is None:
            _warn_missing_location_once(team_b, norm_b)

    # Order lexicographically for symmetric cache hits
    if norm_a > norm_b:
        norm_a, norm_b = norm_b, norm_a

    return _cached_geodesic_distance_impl(norm_a, norm_b)


def clear_geodesic_cache() -> None:
    """Clear the geodesic distance cache. Use for long-running processes or testing."""
    _cached_geodesic_distance_impl.cache_clear()


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

        Delegates to module-level cached function to avoid repeated geodesic
        calculations for the same team pair across the backtest loop.

        Args:
            team_a: First team
            team_b: Second team

        Returns:
            Distance in miles or None if location data unavailable
        """
        return _cached_geodesic_distance(team_a, team_b)

    def get_timezone_adjustment(
        self,
        away_team: str,
        home_team: str,
        game_date=None,
    ) -> float:
        """Calculate timezone-based home advantage from away team travel fatigue.

        Convention: Positive = favors home team (consistent with all adjusters).

        Args:
            away_team: Traveling team
            home_team: Host team
            game_date: Optional date for DST-accurate calculations. If None, assumes
                       DST is active (accurate for ~70% of CFB season, weeks 0-10).

        Returns:
            Home advantage in points (positive = home benefits from away travel)
        """
        # Get directed timezone change (positive = east, negative = west)
        directed_tz = get_directed_timezone_change(away_team, home_team, game_date)

        if directed_tz == 0:
            return 0.0

        # Base adjustment per timezone crossed
        base_adj = abs(directed_tz) * self.timezone_adjustment

        # Direction affects severity:
        # - Traveling EAST (positive directed_tz): harder, losing time → full advantage
        # - Traveling WEST (negative directed_tz): easier, gaining time → 0.8x advantage
        if directed_tz > 0:
            # Away traveling east (harder) → home gains full advantage
            return base_adj
        else:
            # Away traveling west (easier) → home gains reduced advantage
            return base_adj * 0.8

    def get_distance_adjustment(
        self,
        away_team: str,
        home_team: str,
        distance: float | None = None,
    ) -> float:
        """Calculate distance-based home advantage from away team travel fatigue.

        Convention: Positive = favors home team (consistent with all adjusters).

        Args:
            away_team: Traveling team
            home_team: Host team
            distance: Pre-computed distance in miles (optional, avoids duplicate lookup)

        Returns:
            Home advantage in points (positive = home benefits from away travel)
        """
        if distance is None:
            distance = self.get_distance(away_team, home_team)

        if distance is None:
            return 0.0

        # No adjustment for short trips
        if distance < self.SHORT_TRIP:
            return 0.0

        # Small adjustment for medium trips
        if distance < self.MEDIUM_TRIP:
            return 0.25

        # Moderate adjustment for long trips
        if distance < self.LONG_TRIP:
            return 0.5

        # Larger adjustment for very long trips (Hawaii, cross-country)
        return 1.0

    def get_total_travel_adjustment(
        self,
        home_team: str,
        away_team: str,
        game_date=None,
    ) -> tuple[float, dict]:
        """Get total travel-based home advantage for a matchup.

        Convention: Positive = favors home team (consistent with all adjusters).

        Note: Timezone advantage is reduced for short distances (<700 miles) because
        analysis shows these "regional" games with timezone differences (e.g., due to
        DST quirks or CT/ET border) are over-predicted when given full TZ advantage.
        The 500-800mi TZ games showed +3.83 pts mean error vs -0.87 for no-TZ games.

        Args:
            home_team: Home team
            away_team: Away team
            game_date: Optional date for DST-accurate timezone calculations. If None,
                       assumes DST is active (accurate for ~70% of CFB season).

        Returns:
            Tuple of (total home advantage in points, breakdown dict)
        """
        # Compute distance once and reuse
        distance = self.get_distance(away_team, home_team)

        # All sub-methods return positive = home advantage
        tz_home_adv_raw = self.get_timezone_adjustment(away_team, home_team, game_date)
        distance_home_adv = self.get_distance_adjustment(away_team, home_team, distance=distance)

        # Reduce timezone advantage for short distances
        # Rationale: DST quirks and CT/ET border crossings shouldn't inflate
        # home advantage for truly regional games where travel fatigue is minimal
        tz_home_adv = tz_home_adv_raw
        if distance is not None and tz_home_adv_raw > 0:
            if distance < 400:
                # Very short trips: eliminate TZ advantage entirely
                tz_home_adv = 0.0
            elif distance < 700:
                # Short-medium trips: reduce TZ advantage by 50%
                tz_home_adv = tz_home_adv_raw * 0.5

        # All values already in home advantage convention - just sum
        total = tz_home_adv + distance_home_adv

        tz_diff = get_timezone_difference(away_team, home_team, game_date)

        # Breakdown dict uses unambiguous "home_adv" suffix
        # All values positive = favors home team
        # distance_available=False means location data missing for one/both teams
        breakdown = {
            "distance_miles": distance,
            "distance_available": distance is not None,
            "timezone_diff": tz_diff,
            "timezone_home_adv_raw": tz_home_adv_raw,  # Before distance dampening
            "timezone_home_adv": tz_home_adv,  # After distance dampening
            "distance_home_adv": distance_home_adv,  # 0.0 if distance unavailable
            "total_home_adv": total,
            "game_date_provided": game_date is not None,  # False = using DST-era default
        }

        return total, breakdown

