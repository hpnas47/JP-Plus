"""Preseason priors for early-season predictions.

Uses previous year's ratings and team talent to create preseason expectations.
These are blended with in-season data based on games played.

Key features:
- **Asymmetric Regression**: Teams far from the mean (elite or terrible) regress less
  toward average. This preserves the true spread between top and bottom teams,
  improving accuracy for blowout games where traditional regression compresses
  ratings too much.

- **Extremity-Weighted Talent**: For extreme teams (20+ pts from mean), talent weight
  is reduced to 50% of normal. This trusts proven performance over talent projections
  for outlier teams.

- **Coaching Change Adjustment**: When a new HC arrives at an underperforming team,
  the model dampens prior year drag and weights talent more heavily.

- **Transfer Portal Integration**: Adjusts returning production based on net portal
  gains/losses.
"""

import heapq
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from config.teams import TRIPLE_OPTION_TEAMS
from src.data.priors_cache import PriorsDataCache

if TYPE_CHECKING:
    from src.api.cfbd_client import CFBDClient

logger = logging.getLogger(__name__)


# =============================================================================
# COACHING CHANGE DATA
# =============================================================================

# Coach pedigree multipliers for forget factor (DATA-DRIVEN)
# Calculated from: career win %, P5 HC experience years, total HC years
# Higher = more confidence in new coach = forget more of underperformance
# Formula: base 1.0 + win% bonus (-0.15 to +0.25) + P5 exp (0.03/yr, max 0.12) + longevity (0.03 if 5+ yrs)
#
# IMPORTANT: Only coaches with PRIOR HC experience should be in this dict.
# First-time HCs should NOT be listed - they get excluded from adjustment entirely.
# The pedigree score should reflect record BEFORE the hire, not including current job.
COACH_PEDIGREE = {
    # ==========================================================================
    # ELITE TIER (1.27 - 1.35): 65%+ win rate, significant P5/NFL HC experience
    # ==========================================================================
    "Bill Belichick": 1.35,     # NFL legend (302-163), 6x Super Bowl
    "Brian Kelly": 1.30,        # Notre Dame (113-40) before LSU
    "Lincoln Riley": 1.30,      # Oklahoma (55-10) before USC
    "Luke Fickell": 1.30,       # Cincinnati (57-18) before Wisconsin
    "Lane Kiffin": 1.30,        # FAU, Ole Miss success before LSU
    "Hugh Freeze": 1.27,        # Ole Miss, Liberty before Auburn

    # ==========================================================================
    # STRONG TIER (1.20 - 1.26): Proven P5/G5 HC success
    # ==========================================================================
    "Kalen DeBoer": 1.25,       # Fresno State, Washington success before Alabama
    "Billy Napier": 1.25,       # Louisiana (40-12) before Florida
    "Curt Cignetti": 1.25,      # JMU dominance (19-4) before Indiana
    "Jon Sumrall": 1.25,        # Tulane (32-9) before Florida
    "Dan Mullen": 1.25,         # Florida (34-15), Mississippi State (69-46) before UNLV
    "Sonny Dykes": 1.25,        # TCU (29-16, national runner-up), SMU before TCU
    "Bronco Mendenhall": 1.22,  # BYU (99-43), Virginia (36-38) before New Mexico
    "Willie Fritz": 1.22,       # Tulane (55-37), Georgia Southern before Houston
    "Tom Herman": 1.20,         # Texas (32-18), Houston (22-4) before FAU
    "Mario Cristobal": 1.20,    # Oregon (35-13) before Miami
    "Manny Diaz": 1.20,         # Miami (21-15) before Duke
    "Jeff Brohm": 1.20,         # Purdue (36-34) before Louisville return
    "Ryan Silverfield": 1.21,   # Memphis (42-20) before Arkansas
    "Jamey Chadwell": 1.21,     # Coastal Carolina (40-13) before Liberty
    "Bill O'Brien": 1.20,       # Penn State (29-11), NFL HC before Boston College

    # ==========================================================================
    # ABOVE AVERAGE TIER (1.10 - 1.19): Some HC success, or elite coordinator
    # ==========================================================================
    "Barry Odom": 1.18,         # Missouri (25-25), UNLV success before current
    "Marcus Freeman": 1.15,     # Elite Notre Dame DC, limited HC but elite trajectory
    "Matt Rhule": 1.15,         # Temple (28-23), Baylor (19-20) before Nebraska
    "Mike Elko": 1.15,          # Duke (17-18), elite DC pedigree before Texas A&M
    "Jonathan Smith": 1.15,     # Oregon State (28-34) rebuild before Michigan State
    "Jim Mora": 1.15,           # UCLA (46-30), NFL HC before UConn
    "Derek Mason": 1.12,        # Vanderbilt (27-55), elite DC before Middle Tennessee
    "Ken Niumatalolo": 1.12,    # Navy (109-83), triple-option specialist
    "Joey McGuire": 1.10,       # Limited HC before Texas Tech, strong recruiter
    "Brent Venables": 1.10,     # First P5 HC but elite Clemson DC pedigree
    "Tony Elliott": 1.10,       # Elite Clemson OC before Virginia
    "Jeff Tedford": 1.08,       # Cal (82-57), Fresno return
    "Jerry Kill": 1.08,         # Minnesota (29-29), Northern Illinois before NMSU

    # ==========================================================================
    # AVERAGE TIER (1.00 - 1.09): Mixed results or lateral moves
    # ==========================================================================
    "Rhett Lashlee": 1.05,      # SMU (29-19), coordinator background
    "Scott Satterfield": 1.05,  # App State (51-24), struggled at Louisville (25-24)
    "Ryan Walters": 1.05,       # Illinois DC success before Purdue
    "G.J. Kinne": 1.05,         # Texas State turnaround before current
    "Eric Morris": 1.05,        # North Texas, offensive mind
    "Tim Beck": 1.05,           # Coastal Carolina, coordinator background
    "Trent Dilfer": 1.05,       # UAB, NFL QB turned coach
    "Kevin Wilson": 1.05,       # Indiana (26-47), Tulsa after OC stint
    "Jay Norvell": 1.05,        # Nevada (33-26) before Colorado State
    "Joe Moorhead": 1.02,       # Mississippi State (14-12) before Akron
    "Clay Helton": 1.00,        # USC (46-24) before Georgia Southern
    "Stan Drayton": 1.00,       # First HC at Temple, NFL RB coach
    "Major Applewhite": 1.00,   # Houston (15-11), OC background
    "Scotty Walden": 1.00,      # Austin Peay, UTEP
    "Chad Lunsford": 1.00,      # Georgia Southern (28-21) before FAU
    "Sean Lewis": 1.00,         # Kent State (24-31) before San Diego State
    "Tony Sanchez": 0.98,       # UNLV (19-40) before New Mexico State

    # ==========================================================================
    # BELOW AVERAGE TIER (0.85 - 0.99): Concerning prior HC record
    # ==========================================================================
    "Jedd Fisch": 0.95,         # Below .500 at Arizona (16-16) before Washington
    "Rich Rodriguez": 0.95,     # Michigan (15-22), Arizona (43-35), mixed bag
    "Scott Frost": 0.85,        # UCF (19-7) but Nebraska disaster (16-31)
    "Jeff Lebby": 0.90,         # Limited/poor HC record
    # Note: First-time HCs go in FIRST_TIME_HCS, not here with pedigree 0
}

# First-time HCs - explicitly excluded from coaching change adjustment
# These coaches have no prior HC record, so we have no basis to predict improvement
# Categories: Internal promotions, first P5 HC jobs, FCS-only backgrounds
FIRST_TIME_HCS = {
    # 2022 hires
    "Dan Lanning",          # First HC job at Oregon - elite DC but no HC record
    "Brent Pry",            # First HC at Virginia Tech - elite DC (Penn State, Clemson)
    "Jake Dickert",         # Internal promotion at Washington State
    "Michael Desormeaux",   # First HC at Louisiana - internal promotion

    # 2023 hires
    "Deion Sanders",        # No prior FBS HC record before Colorado (HBCU only)
    "Brent Key",            # Internal promotion at Georgia Tech
    "Kenny Dillingham",     # First HC job at Arizona State
    "Troy Taylor",          # FCS only (Sacramento State) before Stanford
    "David Braun",          # Internal promotion at Northwestern
    "Spencer Danielson",    # Internal promotion at Boise State

    # 2024 hires
    "Sherrone Moore",       # Internal promotion at Michigan
    "Deshaun Foster",       # Internal promotion at UCLA - no HC experience
    "Trent Bray",           # Internal promotion at Oregon State
    "Fran Brown",           # First HC at Syracuse - elite recruiter, no HC record
    "Brent Brennan",        # San Jose State (FCS-level results) before Arizona
    "Pete Kaligis",         # Internal/interim at Washington State
    "Freddie Kitchens",     # NFL OC, no college HC before North Carolina

    # 2025 hires
    "Pete Golding",         # First HC job - DC background
    "Frank Reich",          # NFL HC but no college HC before Stanford
    "Eddie George",         # Tennessee State (FCS) before Bowling Green
    "Dowell Loggains",      # First HC at App State - NFL OC background
    "Blake Harrell",        # Internal promotion at East Carolina

    # Interim coaches (excluded - too short tenure to evaluate)
    "Jim Leonhard",         # Wisconsin interim (2022)
    "Cadillac Williams",    # Auburn interim (2022)
    "Mickey Joseph",        # Nebraska interim (2022)
    "Elijah Robinson",      # Texas A&M interim (2023)
    "Nunzio Campanile",     # Syracuse interim (2023)
    "Greg Knox",            # Mississippi State interim (2023)
    "Slade Nagle",          # Tulane interim (2023)
    "Trooper Taylor",       # Duke interim (2023)
    "Gerad Parker",         # Troy interim (2024)
    "Reed Stringer",        # Southern Miss interim (2024)
    "Everett Withers",      # Temple interim (2024)
    "Chad Scott",           # West Virginia interim (2024)
    "Jay Sawvel",           # Wyoming interim (2024)
    "Nate Dreiling",        # Utah State interim (2024)
}

# Manual coaching change overrides: {year: {team: coach_name}}
# These are used as fallback when API data is unavailable or as explicit overrides.
# The auto-detection via detect_coaching_changes() is preferred for current data.
COACHING_CHANGES_MANUAL = {
    # Historical records (API may not have complete data for older years)
    2022: {
        "LSU": "Brian Kelly",
        "USC": "Lincoln Riley",
        "Miami": "Mario Cristobal",
        "Florida": "Billy Napier",
        "Texas Tech": "Joey McGuire",
    },
    2023: {
        "Wisconsin": "Luke Fickell",
        "Nebraska": "Matt Rhule",
        "Auburn": "Hugh Freeze",
    },
    2024: {
        "Washington": "Jedd Fisch",
        "Alabama": "Kalen DeBoer",
        "Duke": "Manny Diaz",
        "Indiana": "Curt Cignetti",
        "Mississippi State": "Jeff Lebby",
    },
    # Future years (API won't have data yet)
    2026: {
        "LSU": "Lane Kiffin",
        "Florida": "Jon Sumrall",
        "Arkansas": "Ryan Silverfield",
    },
}

# Cache for auto-detected coaching changes
_coaching_changes_cache: dict[int, dict[str, str]] = {}

# Manual overrides for talent rank (if API data is inconsistent)
# Format: {year: {team: talent_rank}}
TALENT_RANK_OVERRIDES = {}


# =============================================================================
# TRIPLE-OPTION TEAM ADJUSTMENT
# =============================================================================

# Triple-option teams are systematically underrated by efficiency metrics like SP+
# because EPA calculations don't capture their scheme's value properly.
# Analysis of 2024 backtest showed Navy/Army rated as underdogs when Vegas had
# them as 7-11 pt favorites. Adding a rating boost corrects this systematic bias.
#
# Boost magnitude based on observed discrepancy:
# - Navy vs Temple: Ours +4.1, Vegas -11.5 → 15.6 pt gap
# - Army vs Rice: Ours +8.8, Vegas -7.0 → 15.8 pt gap
# - Navy vs Memphis: Ours +21.4, Vegas +9.5 → 11.9 pt gap
# Using 8.0 pt boost as conservative estimate (half of observed gap)

# TRIPLE_OPTION_TEAMS imported from config.teams (single source of truth)

TRIPLE_OPTION_RATING_BOOST = 6.0  # Points to add to raw SP+ rating (conservative)


def get_triple_option_boost(team: str) -> float:
    """Get rating boost for triple-option teams.

    Args:
        team: Team name

    Returns:
        Rating boost (positive points to add to raw SP+ rating)
    """
    if team in TRIPLE_OPTION_TEAMS:
        return TRIPLE_OPTION_RATING_BOOST
    return 0.0


def get_coach_pedigree(coach_name: str) -> Optional[float]:
    """Get pedigree multiplier for a coach.

    Args:
        coach_name: Coach's name

    Returns:
        Pedigree multiplier (0.90 - 1.30), 0 if first-time HC (explicitly excluded),
        or None if coach is unlisted (data entry error)
    """
    # First-time HCs are explicitly excluded (return 0 sentinel)
    if coach_name in FIRST_TIME_HCS:
        return 0

    # Check if coach has explicit pedigree entry
    if coach_name in COACH_PEDIGREE:
        return COACH_PEDIGREE[coach_name]

    # Unlisted coach - return None to signal data entry error
    return None


def detect_coaching_changes(client, year: int) -> dict[str, str]:
    """Auto-detect coaching changes for a season using CFBD API.

    Compares head coaches between year and year-1 to identify changes.
    Results are cached to avoid repeated API calls.

    Args:
        client: CFBDClient instance
        year: Season year to detect changes for

    Returns:
        Dict mapping team -> new coach name for teams with coaching changes
    """
    global _coaching_changes_cache

    # Return cached result if available
    if year in _coaching_changes_cache:
        return _coaching_changes_cache[year]

    changes: dict[str, str] = {}

    try:
        # Get coaches for current year and prior year
        current_coaches = client.get_coaches(year=year)
        prior_coaches = client.get_coaches(year=year - 1)

        if not current_coaches or not prior_coaches:
            logger.warning(
                f"Incomplete coach data for {year}/{year-1}, falling back to manual dict"
            )
            return COACHING_CHANGES_MANUAL.get(year, {})

        # Build lookup: team -> coach name for each year
        def build_coach_map(coaches_list) -> dict[str, str]:
            coach_map = {}
            for coach in coaches_list:
                if coach.seasons:
                    for season in coach.seasons:
                        team = season.school
                        coach_name = f"{coach.first_name} {coach.last_name}"
                        coach_map[team] = coach_name
            return coach_map

        current_map = build_coach_map(current_coaches)
        prior_map = build_coach_map(prior_coaches)

        # Find teams where coach changed
        for team, current_coach in current_map.items():
            prior_coach = prior_map.get(team)
            if prior_coach and prior_coach != current_coach:
                changes[team] = current_coach
                logger.debug(
                    f"Detected coaching change: {team} - {prior_coach} -> {current_coach}"
                )

        logger.info(f"Auto-detected {len(changes)} coaching changes for {year}")

        # Cache the result
        _coaching_changes_cache[year] = changes
        return changes

    except Exception as e:
        logger.warning(f"Error detecting coaching changes for {year}: {e}")
        logger.warning("Falling back to manual coaching changes dict")
        return COACHING_CHANGES_MANUAL.get(year, {})


def get_coaching_changes(client, year: int) -> dict[str, str]:
    """Get coaching changes for a year, using auto-detection with manual fallback.

    Args:
        client: CFBDClient instance (can be None to use manual dict only)
        year: Season year

    Returns:
        Dict mapping team -> new coach name
    """
    # Try auto-detection first if client is provided
    if client is not None:
        detected = detect_coaching_changes(client, year)
        if detected:
            return detected

    # Fall back to manual dict
    return COACHING_CHANGES_MANUAL.get(year, {})


def is_new_coach(
    team: str, year: int, client=None
) -> tuple[bool, Optional[str]]:
    """Check if team has a new head coach for the given year.

    Uses auto-detection via CFBD API if client is provided, otherwise
    falls back to manual COACHING_CHANGES_MANUAL dict.

    Args:
        team: Team name
        year: Season year
        client: Optional CFBDClient for auto-detection

    Returns:
        Tuple of (is_new_coach, coach_name or None)
    """
    year_changes = get_coaching_changes(client, year)
    if team in year_changes:
        return True, year_changes[team]
    return False, None


@dataclass
class PreseasonRating:
    """Container for preseason team rating."""

    team: str
    prior_rating: float  # Previous year's rating (or talent-based estimate)
    talent_score: float  # Recruiting talent composite (raw API value, ~500-1000)
    talent_rating_normalized: float  # P0.1: Talent on SP+ rating scale (z-score * 12)
    returning_ppa: float  # Percentage of PPA returning (0-1)
    combined_rating: float  # Blended preseason rating
    confidence: float  # How confident we are in this prior (0-1)
    new_coach: bool = False  # Whether team has new head coach
    coach_name: Optional[str] = None  # Name of new coach (if applicable)
    coaching_adjustment: float = 0.0  # Rating adjustment from coaching change
    portal_adjustment: float = 0.0  # Transfer portal impact on returning production
    talent_discount: float = 1.0  # Reserved for future per-team talent floor scaling
    # Credible Rebuild fields (for week-tapered relief in blend_with_inseason)
    rebuild_credibility: float = 0.0  # 0-1 score based on talent + portal signals
    rebuild_relief_delta: float = 0.0  # Point gain from reduced regression (applied with week taper)
    # Raw SP+ prior for delta-shrink experiment (Phase 1 lambda sweep)
    sp_prior: float = 0.0  # Raw SP+ from prior year (no RP/portal/talent adjustments)


class PreseasonPriors:
    """
    Generate preseason team ratings using:
    1. Previous year's SP+ ratings (regressed toward mean)
    2. Team talent composite (recruiting)
    3. Returning production (if available)

    These priors help early-season predictions when game data is limited.
    """

    def __init__(
        self,
        client: "CFBDClient",
        prior_year_weight: float = 0.6,
        talent_weight: float = 0.4,
        regression_factor: float = 0.3,
        use_talent_decay: bool = True,
        # Credible Rebuild config (reduces extra regression for low-RP teams with quality priors)
        credible_rebuild_enabled: bool = True,
        credible_rebuild_rp_cutoff: float = 0.35,  # Only trigger if RP <= this
        credible_rebuild_max_relief: float = 0.25,  # Max 25% regression reduction (conservative)
        credible_rebuild_week_end: int = 5,  # Linear taper to 0 by this week
        credible_rebuild_talent_threshold: float = 0.65,  # Min talent_norm to qualify
        credible_rebuild_portal_threshold: float = 0.65,  # Min portal_norm to qualify
        credible_rebuild_use_coach: bool = False,  # Include coach pedigree (experimental)
    ) -> None:
        """Initialize preseason priors calculator.

        Args:
            client: CFBDClient instance for API access
            prior_year_weight: Weight for previous year's rating (0-1)
            talent_weight: Weight for talent composite (0-1)
            regression_factor: How much to regress prior ratings toward mean (0-1)
            use_talent_decay: If True, decay talent_floor_weight from 0.08→0.03 over weeks 0-10.
                If False, use static talent_floor_weight all season (legacy behavior).
            credible_rebuild_enabled: Enable credible rebuild adjustment for low-RP teams
            credible_rebuild_rp_cutoff: RP threshold to consider for rebuild relief (0.35 = 35%)
            credible_rebuild_max_relief: Maximum regression reduction (0.25 = 25% less regression)
            credible_rebuild_week_end: Week by which relief tapers to 0 (linear decay, protects Core)
            credible_rebuild_talent_threshold: Min talent_norm to qualify (0.65 default)
            credible_rebuild_portal_threshold: Min portal_norm to qualify (0.65 default)
            credible_rebuild_use_coach: Include coach pedigree in credibility (experimental)
        """
        self.client = client
        self.prior_year_weight = prior_year_weight
        self.talent_weight = talent_weight
        self.regression_factor = regression_factor
        self.use_talent_decay = use_talent_decay

        # Credible Rebuild config
        self.credible_rebuild_enabled = credible_rebuild_enabled
        self.credible_rebuild_rp_cutoff = credible_rebuild_rp_cutoff
        self.credible_rebuild_max_relief = credible_rebuild_max_relief
        self.credible_rebuild_week_end = credible_rebuild_week_end
        self.credible_rebuild_talent_threshold = credible_rebuild_talent_threshold
        self.credible_rebuild_portal_threshold = credible_rebuild_portal_threshold
        self.credible_rebuild_use_coach = credible_rebuild_use_coach

        self.preseason_ratings: dict[str, PreseasonRating] = {}

        # Disk cache for API responses (eliminates HTTP calls for historical data)
        self._cache = PriorsDataCache()

    def fetch_prior_year_sp(self, year: int) -> dict[str, float]:
        """Fetch SP+ ratings from the previous year.

        Args:
            year: The CURRENT season year (will fetch year-1)

        Returns:
            Dictionary mapping team name to SP+ rating
        """
        prior_year = year - 1

        # Try cache first
        cached = self._cache.load_sp_ratings(prior_year)
        if cached is not None:
            return cached

        # Cache miss - fetch from API
        try:
            sp_ratings = self.client.get_sp_ratings(year=prior_year)
            ratings = {}
            for team in sp_ratings:
                if team.rating is not None:
                    ratings[team.team] = team.rating
            logger.info(f"Fetched {len(ratings)} SP+ ratings from {prior_year}")

            # Save to cache for future runs
            self._cache.save_sp_ratings(prior_year, ratings)

            return ratings
        except Exception as e:
            logger.warning(f"Could not fetch SP+ ratings for {prior_year}: {e}")
            return {}

    def fetch_talent(self, year: int) -> dict[str, float]:
        """Fetch team talent composite for the current year.

        Args:
            year: Season year

        Returns:
            Dictionary mapping team name to talent score
        """
        # Try cache first
        cached = self._cache.load_talent(year)
        if cached is not None:
            return cached

        # Cache miss - fetch from API
        try:
            talent = self.client.get_team_talent(year=year)
            scores = {}
            for team in talent:
                scores[team.team] = team.talent
            logger.info(f"Fetched talent scores for {len(scores)} teams")

            # Save to cache for future runs
            self._cache.save_talent(year, scores)

            return scores
        except Exception as e:
            logger.warning(f"Could not fetch talent data for {year}: {e}")
            return {}

    def fetch_returning_production(self, year: int) -> dict[str, float]:
        """Fetch returning production (percent PPA returning) for each team.

        Args:
            year: Season year (fetches returning production FOR this year)

        Returns:
            Dictionary mapping team name to percent_ppa (0-1 scale)
        """
        # Try cache first
        cached = self._cache.load_returning_production(year)
        if cached is not None:
            return cached

        # Cache miss - fetch from API
        try:
            rp_data = self.client.get_returning_production(year=year)
            returning = {}
            for team in rp_data:
                if team.percent_ppa is not None:
                    returning[team.team] = team.percent_ppa
            logger.info(f"Fetched returning production for {len(returning)} teams")

            # Save to cache for future runs
            self._cache.save_returning_production(year, returning)

            return returning
        except Exception as e:
            logger.warning(f"Could not fetch returning production for {year}: {e}")
            return {}

    def fetch_transfer_portal(self, year: int) -> pd.DataFrame:
        """Fetch transfer portal data for a given year.

        Args:
            year: Season year (fetches transfers FOR this year)

        Returns:
            DataFrame with transfer portal entries
        """
        # Try cache first
        cached = self._cache.load_transfer_portal(year)
        if cached is not None:
            return cached

        # Cache miss - fetch from API
        try:
            transfers = self.client.get_transfer_portal(year=year)
            data = []
            for t in transfers:
                # Handle nested objects - origin/destination can be str or object
                origin = t.origin
                destination = t.destination

                origin_team = origin if isinstance(origin, str) else (
                    origin.team if origin and hasattr(origin, 'team') else None
                )
                dest_team = destination if isinstance(destination, str) else (
                    destination.team if destination and hasattr(destination, 'team') else None
                )

                data.append({
                    'first_name': t.first_name,
                    'last_name': t.last_name,
                    'full_name': f"{t.first_name} {t.last_name}",
                    'position': t.position if isinstance(t.position, str) else (
                        t.position.position if hasattr(t.position, 'position') else str(t.position)
                    ),
                    'origin': origin_team,
                    'destination': dest_team,
                    'rating': t.rating if hasattr(t, 'rating') else None,
                    'stars': t.stars if hasattr(t, 'stars') else None,
                })

            df = pd.DataFrame(data)
            logger.info(f"Fetched {len(df)} transfer portal entries for {year}")

            # Save to cache for future runs
            self._cache.save_transfer_portal(year, df)

            return df

        except Exception as e:
            logger.warning(f"Could not fetch transfer portal for {year}: {e}")
            return pd.DataFrame()

    def fetch_player_usage(self, year: int) -> pd.DataFrame:
        """Fetch player usage stats (includes PPA) for a given year.

        Args:
            year: Season year

        Returns:
            DataFrame with player usage data
        """
        try:
            usage = self.client.get_player_usage(year=year)
            data = []
            for u in usage:
                ppa = u.usage.overall if hasattr(u.usage, 'overall') else None
                if ppa is not None:
                    data.append({
                        'name': u.name,
                        'team': u.team,
                        'position': u.position,
                        'total_ppa': ppa,
                    })

            df = pd.DataFrame(data)
            logger.info(f"Fetched player usage for {len(df)} players in {year}")
            return df

        except Exception as e:
            logger.warning(f"Could not fetch player usage for {year}: {e}")
            return pd.DataFrame()

    # Position group mapping for transfer portal analysis
    # Split OT from IOL, and iDL from EDGE for scarcity-based weighting
    POSITION_GROUPS = {
        'QB': ['QB'],
        'OT': ['OT', 'T'],           # Offensive Tackles - premium scarcity
        'IOL': ['OG', 'OC', 'OL', 'G', 'C', 'IOL'],  # Interior OL
        'EDGE': ['EDGE', 'DE'],       # Edge rushers
        'IDL': ['DL', 'DT', 'NT'],    # Interior defensive line
        'LB': ['LB', 'ILB', 'OLB', 'MLB'],
        'RB': ['RB', 'FB', 'APB'],
        'WR': ['WR'],
        'CB': ['CB'],                 # Split CB from safety for skill tier
        'S': ['S', 'FS', 'SS', 'SAF', 'DB'],  # Safeties
        'TE': ['TE'],
        'ST': ['K', 'P', 'LS', 'PK'],
        'ATH': ['ATH', 'ATHLETE', 'PRO'],
    }

    # Scarcity-Based Position Weights (2026 market reality)
    # Reflects that elite trench play is the primary driver of rating stability
    POSITION_WEIGHTS = {
        # Premium Tier (0.85+): Franchise-altering positions
        'QB': 1.00,    # Signal caller - highest impact
        'OT': 0.90,    # Elite blindside protector - 90% of QB value

        # Anchor Tier (0.75): Trench dominance
        'EDGE': 0.75,  # Premium pass rushers
        'IDL': 0.75,   # Interior defensive line - run stuffers + interior pressure

        # Support Tier (0.55-0.60): Important but deeper talent pools
        'IOL': 0.60,   # Interior OL - guards/centers
        'LB': 0.55,    # Run defense, coverage
        'S': 0.55,     # Safety - deep coverage, run support
        'TE': 0.50,    # Hybrid role, increasing value

        # Skill/Floor Tier (0.40-0.50): High plug-and-play success
        'WR': 0.45,    # Receiving weapons - higher replacement rate
        'CB': 0.45,    # Corners - athletic translation
        'RB': 0.40,    # Most replaceable skill position

        # Depth Tier
        'ST': 0.15,    # Limited snaps
        'ATH': 0.40,   # Unknown role, average skill value
    }

    # Power 4 Conferences (for level-up discount logic)
    # Note: Conference lookup is now year-appropriate via get_fbs_teams(year=year)
    # IMPORTANT: 'FBS Independents' is NOT included — it contains non-P4 teams like
    # UConn, UMass. P4-level independents are listed explicitly in P4_TEAMS.
    P4_CONFERENCES = {'SEC', 'Big Ten', 'Big 12', 'ACC'}

    # P4-level teams that may not be in a P4 conference in a given year
    # With year-appropriate conference lookup (P2.1 fix), this is now minimal:
    # - Notre Dame: Independent but plays P4-level schedule (ACC for most games)
    # - Army: Independent since 2024, competitive with P4 (service academy constraints)
    # - Navy: Independent, competitive with P4 (service academy constraints)
    # Note: BYU joined Big 12 in 2023, so year-appropriate lookup handles it.
    # UConn, UMass are NOT P4 level despite being independents.
    P4_TEAMS = {
        'Notre Dame',
        'Army',   # Competitive independent, service academy recruiting limits
        'Navy',   # Competitive independent, service academy recruiting limits
    }

    # High-contact positions for physicality tax (G5→P4 transfers)
    HIGH_CONTACT_POSITIONS = {'OT', 'IOL', 'IDL', 'LB', 'EDGE'}

    # Skill positions for athleticism discount (G5→P4 transfers)
    SKILL_POSITIONS = {'WR', 'RB', 'CB', 'S'}

    # Level-up discount factors
    PHYSICALITY_TAX = 0.75      # 25% discount for trench players G5→P4
    ATHLETICISM_DISCOUNT = 0.90  # 10% discount for skill players G5→P4

    # Continuity retention: fraction of value retained when losing an incumbent
    # Value 0.90 means departing players retain 90% effectiveness → dividing by 0.90
    # amplifies loss by ~11% (1/0.90 = 1.111). This reflects hidden costs of turnover:
    # scheme fit, leadership, chemistry that a raw "replacement" doesn't capture.
    # E.g., if a team loses 5.0 of outgoing value: effective loss = 5.0 / 0.90 = 5.56
    CONTINUITY_RETENTION = 0.90

    def _get_position_group(self, position: str) -> str:
        """Map a position to its position group.

        Args:
            position: Raw position string from API

        Returns:
            Position group name (QB, OT, IOL, etc.) or 'OTHER' if unknown
        """
        if not position or pd.isna(position):
            return 'OTHER'

        pos_upper = str(position).upper().strip()
        for group, positions in self.POSITION_GROUPS.items():
            if pos_upper in positions:
                return group
        return 'OTHER'

    def _is_p4_team(self, team: str, team_conferences: dict[str, str]) -> bool:
        """Determine if a team is Power 4 level.

        Args:
            team: Team name
            team_conferences: Dict mapping team name to conference

        Returns:
            True if team is P4 level, False if G5/FCS
        """
        if not team or pd.isna(team):
            return False

        # Check explicit P4 teams list (independents, recent movers)
        if team in self.P4_TEAMS:
            return True

        # Check conference
        conf = team_conferences.get(team, '')
        return conf in self.P4_CONFERENCES

    def _get_level_up_discount(
        self,
        position_group: str,
        origin_is_p4: bool,
        dest_is_p4: bool,
    ) -> float:
        """Calculate the level-up discount for G5→P4 transfers.

        Reflects the physicality gap and adjustment curve when players
        move from G5 to P4 competition.

        Args:
            position_group: Player's position group
            origin_is_p4: Whether origin school is P4
            dest_is_p4: Whether destination school is P4

        Returns:
            Discount multiplier (1.0 = no discount, 0.75 = 25% discount)
        """
        # P4→P4 or G5→G5: No discount
        if origin_is_p4 == dest_is_p4:
            return 1.0

        # P4→G5: Player is "stepping down" - slight boost (proven at higher level)
        if origin_is_p4 and not dest_is_p4:
            return 1.10  # 10% boost

        # G5→P4: Apply level-up discount based on position
        if position_group in self.HIGH_CONTACT_POSITIONS:
            # Physicality Tax: 25% discount for trench players
            # Steep curve in trench physicality at P4 level
            return self.PHYSICALITY_TAX
        elif position_group in self.SKILL_POSITIONS:
            # Athleticism Discount: 10% discount for skill players
            # High-end speed translates more easily
            return self.ATHLETICISM_DISCOUNT
        else:
            # Other positions (QB, TE, ST): Moderate discount
            return 0.85

    def _calculate_quality_factor(
        self,
        stars: Optional[float],
        rating: Optional[float],
    ) -> float:
        """Calculate raw quality factor from stars and rating.

        Args:
            stars: Player's star rating (1-5, or None)
            rating: Player's 247 rating (0-1, or None)

        Returns:
            Quality factor (0.1 to 1.0)
        """
        if stars is not None and not pd.isna(stars):
            # Normalize stars: 2->0.1, 3->0.33, 4->0.67, 5->1.0
            stars_factor = (float(stars) - 2) / 3
            stars_factor = max(0.1, min(1.0, stars_factor))

            if rating is not None and not pd.isna(rating):
                # Normalize rating: 0.77->0.1, 0.85->0.5, 0.99->1.0
                rating_factor = (float(rating) - 0.77) / 0.22
                rating_factor = max(0.1, min(1.0, rating_factor))
                # Blend: 60% rating, 40% stars
                return 0.6 * rating_factor + 0.4 * stars_factor
            return stars_factor

        if rating is not None and not pd.isna(rating):
            rating_factor = (float(rating) - 0.77) / 0.22
            return max(0.1, min(1.0, rating_factor))

        # No quality data - assume average (3-star equivalent)
        return 0.33

    def _calculate_player_value(
        self,
        stars: Optional[float],
        rating: Optional[float],
        position_group: str,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        team_conferences: Optional[dict[str, str]] = None,
    ) -> float:
        """Calculate a player's transfer value based on quality, position, and level jump.

        Formula: position_weight × quality_factor × level_up_discount

        Args:
            stars: Player's star rating (1-5, or None)
            rating: Player's 247 rating (0-1, or None)
            position_group: The player's position group
            origin: Origin team (for level-up discount)
            destination: Destination team (for level-up discount)
            team_conferences: Dict mapping team to conference

        Returns:
            Weighted player value (higher = more valuable transfer)
        """
        # Get position weight (default to 0.35 for unknown positions)
        pos_weight = self.POSITION_WEIGHTS.get(position_group, 0.35)

        # Calculate quality factor
        quality_factor = self._calculate_quality_factor(stars, rating)

        # Calculate level-up discount if conference info available
        level_discount = 1.0
        if team_conferences and origin and destination:
            origin_is_p4 = self._is_p4_team(origin, team_conferences)
            dest_is_p4 = self._is_p4_team(destination, team_conferences)
            level_discount = self._get_level_up_discount(
                position_group, origin_is_p4, dest_is_p4
            )

        return pos_weight * quality_factor * level_discount

    def fetch_fbs_teams(self, year: int) -> set[str]:
        """Fetch the set of FBS team names for a given year.

        Uses session-level cache via CFBDClient.

        Args:
            year: Season year

        Returns:
            Set of FBS team names
        """
        try:
            teams = self.client.get_fbs_teams(year=year)
            fbs_set = {t.school for t in teams}
            logger.debug(f"Fetched {len(fbs_set)} FBS teams for {year}")
            return fbs_set
        except Exception as e:
            logger.warning(f"Could not fetch FBS teams for {year}: {e}")
            return set()

    def _fetch_team_conferences(self, year: int) -> dict[str, str]:
        """Fetch conference affiliation for all FBS teams for a specific year.

        Uses get_fbs_teams(year=year) to get year-appropriate conference data,
        which correctly handles realignment (e.g., USC/UCLA to Big Ten in 2024,
        Texas/Oklahoma to SEC in 2024).

        Uses session-level cache via CFBDClient.

        Args:
            year: Season year (conference affiliations as of this year)

        Returns:
            Dict mapping team name to conference name
        """
        try:
            # P2.1 FIX: Use get_fbs_teams with year parameter to get
            # year-appropriate conference affiliations instead of get_teams()
            # which returns current (potentially future) affiliations
            teams = self.client.get_fbs_teams(year=year)
            conf_map = {}
            for t in teams:
                if t.school and t.conference:
                    conf_map[t.school] = t.conference
            logger.debug(f"Fetched {year} conference data for {len(conf_map)} FBS teams")
            return conf_map
        except Exception as e:
            logger.warning(f"Could not fetch team conferences for {year}: {e}")
            return {}

    def calculate_portal_impact(
        self,
        year: int,
        portal_scale: float = 0.15,
        fbs_only: bool = True,
        impact_cap: float = 0.12,
    ) -> dict[str, float]:
        """Calculate net transfer portal impact for each team using unit-level analysis.

        Uses scarcity-based position weights, level-up discounts for G5→P4 moves,
        and continuity tax for losing incumbents. Reflects 2026 market reality
        where elite trench play is the primary driver of rating stability.

        Args:
            year: Season year (fetches transfers FOR this year)
            portal_scale: How much to scale portal impact (default 0.15, matches production caller)
            fbs_only: If True, only include FBS teams in results (default True)
            impact_cap: Cap on portal impact (default ±12%)

        Returns:
            Dictionary mapping team name to adjusted returning production modifier
            Positive = net gain from portal, Negative = net loss
        """
        # Fetch transfer portal entries for this year
        transfers_df = self.fetch_transfer_portal(year)
        if transfers_df.empty:
            logger.warning(f"No transfer portal data for {year}")
            return {}

        # Fetch FBS teams and conferences in ONE API call (avoids redundant get_fbs_teams)
        # Always fetch for conference data (needed for level-up discount logic)
        fbs_teams: set[str] = set()
        team_conferences: dict[str, str] = {}
        try:
            teams = self.client.get_fbs_teams(year=year)
            for t in teams:
                if t.school:
                    fbs_teams.add(t.school)
                    if t.conference:
                        team_conferences[t.school] = t.conference
            logger.debug(f"Fetched {len(fbs_teams)} FBS teams for {year}")
        except Exception as e:
            logger.warning(f"Could not fetch FBS teams for {year}: {e}")
            if fbs_only:
                logger.warning("Cannot filter to FBS-only, using all teams")
                fbs_only = False

        # Filter to FBS-relevant transfers (origin OR destination is FBS)
        if fbs_only:
            original_count = len(transfers_df)
            fbs_mask = (
                transfers_df['origin'].isin(fbs_teams) |
                transfers_df['destination'].isin(fbs_teams)
            )
            transfers_df = transfers_df[fbs_mask].copy()
            logger.info(
                f"Filtered to FBS-relevant transfers: {len(transfers_df)}/{original_count} "
                f"({len(transfers_df)/original_count:.1%})"
            )

        # P1.1: Vectorized position group mapping (replaces per-row apply)
        pos_to_group = {}
        for group, positions in self.POSITION_GROUPS.items():
            for pos in positions:
                pos_to_group[pos] = group
        transfers_df['pos_group'] = (
            transfers_df['position']
            .fillna('')
            .str.upper()
            .str.strip()
            .map(pos_to_group)
            .fillna('OTHER')
        )

        # Log OTHER position group transfers (helps identify unmapped positions)
        other_count = (transfers_df['pos_group'] == 'OTHER').sum()
        if other_count > 0:
            other_positions = (
                transfers_df[transfers_df['pos_group'] == 'OTHER']['position']
                .fillna('UNKNOWN')
                .value_counts()
                .head(10)
                .to_dict()
            )
            logger.warning(
                f"{other_count} transfers ({other_count/len(transfers_df):.1%}) mapped to OTHER. "
                f"Top unmapped positions: {other_positions}"
            )

        # VECTORIZED player value calculation (replaces 2x .apply() over ~2000 rows)
        # Formula: pos_weight × quality_factor × level_discount

        # 1. Position weights (vectorized map)
        pos_weight = transfers_df['pos_group'].map(self.POSITION_WEIGHTS).fillna(0.35)

        # 2. Quality factor (vectorized np.where chains)
        stars = pd.to_numeric(transfers_df['stars'], errors='coerce')
        rating = pd.to_numeric(transfers_df['rating'], errors='coerce')

        # Stars factor: (stars - 2) / 3, clamped to [0.1, 1.0]
        stars_factor = ((stars - 2) / 3).clip(0.1, 1.0)
        # Rating factor: (rating - 0.77) / 0.22, clamped to [0.1, 1.0]
        rating_factor = ((rating - 0.77) / 0.22).clip(0.1, 1.0)

        has_stars = stars.notna()
        has_rating = rating.notna()

        # Blend: 60% rating + 40% stars when both present
        quality_factor = np.where(
            has_stars & has_rating,
            0.6 * rating_factor + 0.4 * stars_factor,
            np.where(
                has_stars,
                stars_factor,
                np.where(
                    has_rating,
                    rating_factor,
                    0.33  # Default: 3-star equivalent
                )
            )
        )

        # 3. Level-up discount (vectorized based on P4 status and position)
        level_discount = np.ones(len(transfers_df))  # Default: no discount

        if team_conferences:
            # Pre-compute P4 set for vectorized lookup
            p4_set = self.P4_TEAMS | {
                t for t, c in team_conferences.items() if c in self.P4_CONFERENCES
            }
            origin_is_p4 = transfers_df['origin'].isin(p4_set)
            dest_is_p4 = transfers_df['destination'].isin(p4_set)

            # G5→P4: Apply position-based discount
            g5_to_p4 = ~origin_is_p4 & dest_is_p4
            is_high_contact = transfers_df['pos_group'].isin(self.HIGH_CONTACT_POSITIONS)
            is_skill = transfers_df['pos_group'].isin(self.SKILL_POSITIONS)

            level_discount = np.where(
                g5_to_p4 & is_high_contact,
                self.PHYSICALITY_TAX,  # 0.75 for trench
                np.where(
                    g5_to_p4 & is_skill,
                    self.ATHLETICISM_DISCOUNT,  # 0.90 for skill
                    np.where(
                        g5_to_p4,
                        0.85,  # Other positions (QB, TE, ST)
                        np.where(
                            origin_is_p4 & ~dest_is_p4,
                            1.10,  # P4→G5: 10% boost
                            1.0   # Same level: no discount
                        )
                    )
                )
            )

        # 4. Final values: ASYMMETRIC treatment for level-up transfers
        # Outgoing value: full value regardless of where player goes
        # The origin school lost this player at their competition level (no discount)
        transfers_df['outgoing_value'] = pos_weight * quality_factor

        # Incoming value: discounted for G5→P4 level-up adjustment
        # G5→P4 players face physicality/scheme adjustment at the higher level
        transfers_df['incoming_value'] = pos_weight * quality_factor * level_discount

        # Log position group distribution
        pos_dist = transfers_df['pos_group'].value_counts()
        logger.info(f"Transfer portal {year}: {len(transfers_df)} FBS-relevant transfers")
        logger.debug(f"Position distribution: {pos_dist.to_dict()}")

        # Calculate outgoing value per team (players who left)
        # Apply continuity retention: losing an incumbent hurts even if "replaced"
        # With ASYMMETRIC level-up discounts, net impact is:
        #   Same-level: origin loses value×1.11, dest gains value×1.0, net = -0.11 (continuity only)
        #   G5→P4: origin loses value×1.11 (FULL value), dest gains value×0.75 (discounted)
        #          net = -0.36 to -0.46 depending on position (continuity + discount asymmetry)
        #   P4→G5: origin loses value×1.11, dest gains value×1.10, net ≈ 0 (slight boost)
        outgoing_raw = transfers_df.groupby('origin')['outgoing_value'].sum()
        outgoing = outgoing_raw / self.CONTINUITY_RETENTION  # Amplify loss (divide by 0.90 = ~11%)

        # Calculate incoming value per team (players who arrived)
        # Level-up discounts already applied to incoming_value (G5→P4 discounted)
        incoming_df = transfers_df[transfers_df['destination'].notna()]
        incoming = incoming_df.groupby('destination')['incoming_value'].sum()

        # Log coverage and level-up stats
        has_dest = len(incoming_df)
        logger.info(
            f"Portal coverage: {has_dest}/{len(transfers_df)} ({has_dest/len(transfers_df):.1%}) "
            f"have destinations"
        )

        # P1.1: Vectorized G5→P4 transfer count (replaces per-row apply)
        if team_conferences:
            p4_set = self.P4_TEAMS | {
                t for t, c in team_conferences.items() if c in self.P4_CONFERENCES
            }
            origin_is_p4 = incoming_df['origin'].isin(p4_set)
            dest_is_p4 = incoming_df['destination'].isin(p4_set)
            g5_to_p4 = int((~origin_is_p4 & dest_is_p4).sum())
            logger.info(f"G5→P4 transfers (discounted): {g5_to_p4}")

        # Calculate net impact for each team
        # DETERMINISM: Sort for consistent iteration order
        all_teams = sorted(set(outgoing.index) | set(incoming.index))

        # Filter to FBS teams only for the final results
        if fbs_only and fbs_teams:
            all_teams = [t for t in all_teams if t in fbs_teams]

        portal_impact = {}

        for team in all_teams:
            out_val = outgoing.get(team, 0.0)
            in_val = incoming.get(team, 0.0)
            net_val = in_val - out_val

            # Scale the impact and cap to ±12%
            scaled_impact = net_val * portal_scale
            scaled_impact = max(-impact_cap, min(impact_cap, scaled_impact))

            portal_impact[team] = scaled_impact

        # Log top winners/losers (P3.9: debug level for quiet runs)
        sorted_impact = sorted(portal_impact.items(), key=lambda x: (-x[1], x[0]))
        if sorted_impact:
            logger.debug("Top portal winners (position-weighted value):")
            for team, impact in sorted_impact[:5]:
                logger.debug(f"  {team}: {impact:+.1%}")
            logger.debug("Top portal losers:")
            for team, impact in sorted_impact[-5:]:
                logger.debug(f"  {team}: {impact:+.1%}")

        return portal_impact

    def _normalize_talent(self, talent_scores: dict[str, float]) -> dict[str, float]:
        """Normalize talent scores to be on similar scale as SP+ ratings.

        SP+ ranges roughly from -20 to +35.
        Talent scores range from ~500 to ~1000.

        Args:
            talent_scores: Raw talent scores

        Returns:
            Normalized talent scores (roughly -20 to +35 scale)
        """
        if not talent_scores:
            return {}

        values = list(talent_scores.values())
        mean_talent = np.mean(values)
        std_talent = np.std(values)

        # Normalize to z-scores then scale to SP+ range
        # SP+ roughly has std of ~12 points
        sp_scale = 12.0
        normalized = {}

        for team, talent in talent_scores.items():
            z_score = (talent - mean_talent) / std_talent if std_talent > 0 else 0
            normalized[team] = z_score * sp_scale

        return normalized

    def _get_regression_factor(
        self,
        returning_ppa: Optional[float],
        raw_prior: Optional[float] = None,
        mean_prior: float = 0.0,
    ) -> float:
        """Calculate regression factor based on returning production and rating extremity.

        Teams with high returning production should regress less toward mean.
        Teams with low returning production (roster turnover) regress more.

        ASYMMETRIC REGRESSION: Teams far from the mean regress less.
        This preserves the true spread between elite and weak teams, preventing
        artificial compression that hurts betting accuracy for blowout games.

        Formula:
        1. Base regression = 0.5 - (0.4 * percent_ppa)
           - At 0% returning: 0.5 (heavy regression)
           - At 50% returning: 0.3 (baseline)
           - At 100% returning: 0.1 (minimal regression)

        2. Extremity multiplier scales down regression for extreme teams:
           - Within ±10 of mean: full regression (multiplier = 1.0)
           - 10-25 from mean: linear scale down (1.0 -> 0.5)
           - 25+ from mean: minimal regression (multiplier = 0.5)

        3. Final regression = base_regression * extremity_multiplier

        Args:
            returning_ppa: Percentage of PPA returning (0-1), or None if unknown
            raw_prior: The raw prior rating before regression (for extremity calc)
            mean_prior: The mean prior rating to measure distance from

        Returns:
            Regression factor (0-1)
        """
        # Calculate base regression from returning production
        if returning_ppa is None:
            base_regression = self.regression_factor
        else:
            # Clamp to valid range
            pct = max(0.0, min(1.0, returning_ppa))
            # Linear interpolation: high returning = less regression
            base_regression = 0.5 - (0.4 * pct)

        # Apply asymmetric regression based on distance from mean
        # This preserves the true spread - bad teams stay bad, good teams stay good
        if raw_prior is not None:
            distance_from_mean = abs(raw_prior - mean_prior)

            # Scale down regression for extreme teams:
            # - Within ±8 of mean: full regression (multiplier = 1.0)
            # - 8-20 from mean: linear scale down (1.0 -> 0.33)
            # - 20+ from mean: minimal regression (multiplier = 0.33)
            #
            # More aggressive than before to preserve full spread between
            # elite teams (+30) and terrible teams (-25)
            if distance_from_mean <= 8:
                extremity_multiplier = 1.0
            elif distance_from_mean >= 20:
                extremity_multiplier = 0.33
            else:
                # Linear interpolation between 8 and 20
                extremity_multiplier = 1.0 - 0.67 * (distance_from_mean - 8) / 12

            return base_regression * extremity_multiplier

        return base_regression

    def _compute_credible_rebuild_relief(
        self,
        team: str,
        returning_ppa: Optional[float],
        talent_normalized: float,
        portal_impact: float,
        raw_prior: Optional[float],
        mean_prior: float,
        coach_pedigree: Optional[float] = None,
    ) -> tuple[float, float, float, dict]:
        """Compute credible rebuild relief for low-RP teams with quality priors.

        For teams with extremely low returning production (RP <= cutoff), if they have
        high talent and/or strong portal impact, we reduce the extra regression caused
        by low RP. This prevents overly penalizing credible rebuilds in early weeks.

        The relief reduces regression but is BOUNDED:
        - reg_new >= reg_cutoff (can't be better than teams at the RP cutoff)
        - max_relief caps the percentage reduction

        Args:
            team: Team name (for logging)
            returning_ppa: Returning production percentage (0-1), or None
            talent_normalized: Normalized talent score on SP+ scale
            portal_impact: Net portal impact (positive = gained talent)
            raw_prior: Raw prior rating (for regression calculation)
            mean_prior: Mean prior rating
            coach_pedigree: Optional coach pedigree score (if use_coach enabled)

        Returns:
            Tuple of (credibility, relief_delta, new_regression, diagnostics_dict)
            - credibility: 0-1 score based on talent + portal signals
            - relief_delta: Point gain from reduced regression (0 if not triggered)
            - new_regression: Adjusted regression factor
            - diagnostics: Dict with intermediate values for logging
        """
        diag = {
            "triggered": False,
            "reason": "not_low_rp",
            "rp": returning_ppa,
            "talent_raw": talent_normalized,
            "talent_norm": 0.0,
            "portal_raw": portal_impact,
            "portal_norm": 0.0,
            "credibility": 0.0,
            "relief": 0.0,
            "reg_base": 0.0,
            "reg_cutoff": 0.0,
            "reg_new": 0.0,
        }

        # Not enabled or RP above cutoff - no adjustment
        if not self.credible_rebuild_enabled:
            diag["reason"] = "disabled"
            reg_base = self._get_regression_factor(returning_ppa, raw_prior, mean_prior)
            return 0.0, 0.0, reg_base, diag

        if returning_ppa is None or returning_ppa > self.credible_rebuild_rp_cutoff:
            reg_base = self._get_regression_factor(returning_ppa, raw_prior, mean_prior)
            diag["reg_base"] = reg_base
            diag["reg_new"] = reg_base
            if returning_ppa is not None and returning_ppa > self.credible_rebuild_rp_cutoff:
                diag["reason"] = "rp_above_cutoff"
            return 0.0, 0.0, reg_base, diag

        # RP is low - compute credibility from talent + portal signals
        diag["reason"] = "low_rp"

        # Compute base regression and regression at cutoff (for clamping)
        reg_base = self._get_regression_factor(returning_ppa, raw_prior, mean_prior)
        reg_cutoff = self._get_regression_factor(self.credible_rebuild_rp_cutoff, raw_prior, mean_prior)
        diag["reg_base"] = reg_base
        diag["reg_cutoff"] = reg_cutoff

        # Normalize talent to [0, 1] using logistic of z-score
        # talent_normalized is on SP+ scale (z-score * 12), so divide by 12 to get z-score
        # Then use logistic: 1 / (1 + exp(-k * z)) with k=2 for reasonable spread
        # z=0 (average) -> 0.5, z=1.5 (top ~7%) -> 0.95, z=-1.5 -> 0.05
        talent_z = talent_normalized / 12.0 if talent_normalized else 0.0
        talent_norm = 1.0 / (1.0 + np.exp(-2.0 * talent_z))
        diag["talent_norm"] = talent_norm

        # Normalize portal to [0, 1]
        # portal_impact is already scaled (typically -0.12 to +0.12 after cap)
        # Map: 0.0 -> 0.5, +0.12 -> 1.0, -0.12 -> 0.0
        portal_norm = 0.5 + (portal_impact / 0.24) if portal_impact else 0.5
        portal_norm = max(0.0, min(1.0, portal_norm))
        diag["portal_norm"] = portal_norm

        # EXPLICIT THRESHOLD CHECK: Must have RP <= cutoff AND (talent >= threshold OR portal >= threshold)
        # This ensures we only trigger on tail-case teams with real quality signals
        talent_qualifies = talent_norm >= self.credible_rebuild_talent_threshold
        portal_qualifies = portal_norm >= self.credible_rebuild_portal_threshold

        if not (talent_qualifies or portal_qualifies):
            # Team is low-RP but doesn't meet quality thresholds - no relief
            diag["reason"] = "below_quality_threshold"
            diag["credibility"] = 0.0
            diag["reg_new"] = reg_base
            return 0.0, 0.0, reg_base, diag

        # Compute credibility (0.5 * talent + 0.5 * portal)
        credibility = 0.5 * talent_norm + 0.5 * portal_norm

        # Optionally incorporate coach pedigree (small tiebreaker)
        if self.credible_rebuild_use_coach and coach_pedigree is not None and coach_pedigree > 1.0:
            # Coach adds up to 10% boost to credibility (scaled by pedigree - 1.0)
            coach_boost = min(0.10, (coach_pedigree - 1.0) * 0.33)
            credibility = min(1.0, credibility + coach_boost)

        diag["credibility"] = credibility
        diag["triggered"] = True
        diag["reason"] = "credible_rebuild"

        # Compute relief: max_relief * credibility
        # But bound so reg_new >= reg_cutoff (can't be better than cutoff teams)
        relief = self.credible_rebuild_max_relief * credibility
        reg_candidate = reg_base * (1 - relief)
        reg_new = max(reg_candidate, reg_cutoff)

        # STABILITY ASSERTIONS: Ensure no inversion
        # 1. reg_new must be less than reg_base (we're reducing regression, not increasing)
        # 2. reg_new must be >= reg_cutoff (can't be better than teams at cutoff)
        assert reg_new <= reg_base, (
            f"STABILITY VIOLATION: reg_new ({reg_new:.4f}) > reg_base ({reg_base:.4f}) "
            f"for {team}. Relief should reduce, not increase regression."
        )
        assert reg_new >= reg_cutoff - 1e-9, (
            f"STABILITY VIOLATION: reg_new ({reg_new:.4f}) < reg_cutoff ({reg_cutoff:.4f}) "
            f"for {team}. Triggered team can't have better regression than cutoff team."
        )

        # Actual effective relief after clamping
        actual_relief = 1 - (reg_new / reg_base) if reg_base > 0 else 0
        diag["relief"] = actual_relief
        diag["reg_new"] = reg_new

        # Compute point delta from relief
        # With regression, rating = raw_prior * (1 - reg) + mean_prior * reg
        # Delta = (raw_prior - mean_prior) * (reg_base - reg_new)
        relief_delta = 0.0
        rating_baseline = 0.0
        rating_adjusted = 0.0
        if raw_prior is not None:
            relief_delta = (raw_prior - mean_prior) * (reg_base - reg_new)
            # Log rating impact for debugging
            rating_baseline = raw_prior * (1 - reg_base) + mean_prior * reg_base
            rating_adjusted = raw_prior * (1 - reg_new) + mean_prior * reg_new
            diag["rating_baseline"] = rating_baseline
            diag["rating_adjusted"] = rating_adjusted
            diag["rating_delta"] = relief_delta

        return credibility, relief_delta, reg_new, diag

    def _calculate_coaching_change_weights(
        self,
        team: str,
        year: int,
        talent_rank: int,
        performance_rank: int,
    ) -> tuple[float, float, float, str]:
        """Calculate adjusted weights for teams with new head coaches.

        When a new HC arrives at an underperforming team (talent > performance),
        we dampen the prior year's drag and weight talent more heavily.

        The "forget factor" determines how much to forget prior underperformance:
        - Based on talent-performance gap (bigger gap = more forgetting)
        - Multiplied by coach pedigree (proven coaches get more benefit)
        - Capped at 50% (never fully ignore prior)

        Args:
            team: Team name
            year: Season year
            talent_rank: Team's talent ranking (lower = better)
            performance_rank: Team's prior year performance ranking (lower = better)

        Returns:
            Tuple of (prior_weight, talent_weight, forget_factor, coach_name)
        """
        new_coach, coach_name = is_new_coach(team, year, client=self.client)

        if not new_coach:
            # No coaching change - use default weights
            return self.prior_year_weight, self.talent_weight, 0.0, ""

        # Calculate talent-performance gap (positive = underperformer)
        # Lower rank = better, so underperformer has talent_rank < performance_rank
        talent_gap = performance_rank - talent_rank

        if talent_gap <= 0:
            # Team is at or above expectations - apply mild uncertainty
            # but don't pull them down aggressively
            if talent_gap < -20:
                # Significant overperformer losing their coach - expect regression
                # toward talent (DECREASE prior_weight to trust talent more)
                forget_factor = 0.15  # Mild forget factor
                prior_weight = self.prior_year_weight * (1 - forget_factor)
                talent_weight = 1 - prior_weight
                logger.debug(
                    f"{team}: Overperformer losing coach, expect regression "
                    f"(talent #{talent_rank}, perf #{performance_rank})"
                )
            else:
                # At expectation - just add uncertainty, no directional change
                return self.prior_year_weight, self.talent_weight, 0.0, coach_name
        else:
            # Underperformer with new coach - apply forget factor
            # Get coach pedigree: None=unlisted (error), 0=first-time HC (excluded), >0=valid
            pedigree = get_coach_pedigree(coach_name) if coach_name else 0

            if pedigree is None:
                # Unlisted coach - data entry error, warn and skip adjustment
                logger.warning(
                    f"{team}: Coach '{coach_name}' detected as new coach but not in "
                    f"COACH_PEDIGREE or FIRST_TIME_HCS - add to one of these dicts"
                )
                return self.prior_year_weight, self.talent_weight, 0.0, coach_name

            if pedigree == 0:
                # First-time HC - explicitly excluded, no prior record to base adjustment on
                logger.debug(f"{team}: First-time HC {coach_name}, no adjustment")
                return self.prior_year_weight, self.talent_weight, 0.0, coach_name

            # Scale: gap of 60 spots -> 50% forget (capped)
            base_forget = min(0.5, talent_gap / 60.0)

            # Apply coach pedigree multiplier
            forget_factor = min(0.5, base_forget * pedigree)

            # Adjust weights
            prior_weight = self.prior_year_weight * (1 - forget_factor)
            talent_weight = 1 - prior_weight

            logger.info(
                f"{team}: New coach adjustment - {coach_name} (pedigree {pedigree:.2f}), "
                f"talent #{talent_rank}, perf #{performance_rank}, "
                f"gap={talent_gap}, forget={forget_factor:.2f}, "
                f"weights: prior={prior_weight:.2f}, talent={talent_weight:.2f}"
            )

        return prior_weight, talent_weight, forget_factor, coach_name or ""

    def _validate_data_quality(
        self,
        prior_sp: dict[str, float],
        talent: dict[str, float],
        talent_normalized: dict[str, float],
        returning_prod: dict[str, float],
        portal_impact: dict[str, float],
    ) -> None:
        """P0.2: Validate dataset intersections and rank direction assumptions.

        Checks:
        - Dataset intersection sizes (how many teams have each data source)
        - Talent score direction (higher raw score = better, known elites should rank high)
        - SP+ direction (higher rating = better)
        """
        # Log dataset intersection sizes
        sp_teams = set(prior_sp.keys())
        talent_teams = set(talent.keys())
        ret_teams = set(returning_prod.keys())
        portal_teams = set(portal_impact.keys())

        logger.info(
            f"P0.2 data coverage: SP+={len(sp_teams)}, talent={len(talent_teams)}, "
            f"returning={len(ret_teams)}, portal={len(portal_teams)}"
        )
        all_four = sp_teams & talent_teams & ret_teams & portal_teams
        logger.debug(
            f"  SP+ ∩ talent: {len(sp_teams & talent_teams)}, "
            f"SP+ ∩ returning: {len(sp_teams & ret_teams)}, "
            f"all four: {len(all_four)}"
        )

        # P2.2: Report teams missing from key datasets (SP+ is the reference set)
        if sp_teams:
            missing_talent = sp_teams - talent_teams
            missing_ret = sp_teams - ret_teams
            if missing_talent:
                logger.debug(
                    f"P2.2: {len(missing_talent)} SP+ teams missing talent data: "
                    f"{sorted(missing_talent)[:10]}{'...' if len(missing_talent) > 10 else ''}"
                )
            if missing_ret:
                logger.debug(
                    f"P2.2: {len(missing_ret)} SP+ teams missing returning production: "
                    f"{sorted(missing_ret)[:10]}{'...' if len(missing_ret) > 10 else ''}"
                )

        # Validate talent direction: known elite programs should have high raw scores
        # Higher talent score = better recruiting = should rank near top
        elite_programs = ["Alabama", "Georgia", "Ohio State", "Texas", "LSU"]
        if talent:
            # O(n) top-k extraction instead of O(n log n) full sort
            top_20 = heapq.nlargest(20, talent.items(), key=lambda x: x[1])
            top_20_teams = {t for t, _ in top_20}
            elite_in_top20 = [t for t in elite_programs if t in top_20_teams]
            elite_present = [t for t in elite_programs if t in talent]

            if elite_present and len(elite_in_top20) < len(elite_present) // 2:
                logger.warning(
                    f"P0.2 RANK DIRECTION CHECK FAILED: Only {len(elite_in_top20)}/{len(elite_present)} "
                    f"elite programs in talent top-20. Talent scoring may be inverted!"
                )
            else:
                logger.debug(
                    f"P0.2 rank direction OK: {len(elite_in_top20)}/{len(elite_present)} "
                    f"elite programs in talent top-20"
                )

        # Validate SP+ direction: same elite programs should have positive ratings
        if prior_sp:
            elite_sp = {t: prior_sp[t] for t in elite_programs if t in prior_sp}
            positive_count = sum(1 for v in elite_sp.values() if v > 0)
            if elite_sp and positive_count < len(elite_sp) // 2:
                logger.warning(
                    f"P0.2 SP+ DIRECTION CHECK FAILED: Only {positive_count}/{len(elite_sp)} "
                    f"elite programs have positive SP+ ratings. Rating scale may be inverted!"
                )
            else:
                logger.debug(
                    f"P0.2 SP+ direction OK: {positive_count}/{len(elite_sp)} "
                    f"elite programs positive"
                )

        # Validate normalized talent is consistent with raw talent direction
        if talent_normalized and talent:
            # Top raw talent team should also be top normalized
            top_raw = max(talent, key=talent.get)
            top_norm = max(talent_normalized, key=talent_normalized.get)
            if top_raw != top_norm:
                logger.warning(
                    f"P0.2: Top raw talent ({top_raw}) != top normalized ({top_norm}) - check normalization"
                )

    def calculate_preseason_ratings(
        self,
        year: int,
        use_portal: bool = True,
        portal_scale: float = 0.15,
    ) -> dict[str, PreseasonRating]:
        """Calculate preseason ratings for all teams.

        Uses:
        - Returning production to adjust regression toward mean
        - Transfer portal impact to modify effective returning production
        - Coaching change adjustment for underperforming teams with new HC

        Args:
            year: Season year to generate priors for
            use_portal: Whether to incorporate transfer portal data (default True)
            portal_scale: How much to weight portal impact (default 0.15)

        Returns:
            Dictionary mapping team name to PreseasonRating
        """
        # Fetch data
        prior_sp = self.fetch_prior_year_sp(year)
        talent = self.fetch_talent(year)
        talent_normalized = self._normalize_talent(talent)
        returning_prod = self.fetch_returning_production(year)

        # Fetch transfer portal impact if enabled
        portal_impact = {}
        if use_portal:
            portal_impact = self.calculate_portal_impact(
                year,
                portal_scale=portal_scale,
            )

        # P0.2: Validate data quality and rank direction
        self._validate_data_quality(prior_sp, talent, talent_normalized, returning_prod, portal_impact)

        # Get all teams - DETERMINISM: Sort for consistent iteration order
        all_teams = sorted(set(prior_sp.keys()) | set(talent.keys()))

        # Calculate mean prior rating for regression
        if prior_sp:
            mean_prior = np.mean(list(prior_sp.values()))
        else:
            mean_prior = 0.0

        # Build rankings for coaching change calculation
        # Talent ranking (lower = better)
        talent_ranked = sorted(talent.items(), key=lambda x: -x[1])
        talent_ranks = {team: rank + 1 for rank, (team, _) in enumerate(talent_ranked)}

        # Performance ranking from prior SP+ (lower = better)
        perf_ranked = sorted(prior_sp.items(), key=lambda x: -x[1])
        perf_ranks = {team: rank + 1 for rank, (team, _) in enumerate(perf_ranked)}

        self.preseason_ratings = {}
        coaching_adjustments = []  # Track for logging

        for team in all_teams:
            # Get returning production for this team
            ret_ppa = returning_prod.get(team)

            # Portal impact is a directional modifier: positive = net talent gain
            portal_adj = portal_impact.get(team, 0.0)

            # Get raw prior rating first (needed for asymmetric regression)
            raw_prior = prior_sp.get(team)

            # Apply triple-option boost to raw prior
            # These teams are systematically underrated by SP+ efficiency metrics
            triple_option_boost = get_triple_option_boost(team)
            if raw_prior is not None and triple_option_boost > 0:
                raw_prior += triple_option_boost
                logger.debug(f"{team}: Applied triple-option boost of +{triple_option_boost:.1f} pts")

            # Calculate team-specific regression factor based on:
            # 1. RAW returning production (roster continuity — higher = less regression)
            # 2. Distance from mean (farther = less regression — asymmetric)
            # 3. Credible Rebuild relief for low-RP teams with quality priors
            #
            # IMPORTANT: Use raw ret_ppa here, NOT portal-adjusted.
            # Portal impact is a directional talent change, not a continuity signal.
            # Adding portal talent to a bad team should HELP them (move toward mean),
            # not anchor them to their bad prior by reducing regression.
            # Portal effect is applied as a direct rating adjustment below.
            talent_norm_value = talent_normalized.get(team, 0.0)

            # Compute credible rebuild relief (returns adjusted regression factor)
            # Note: coach_pedigree is passed as None here; use_coach is disabled by default
            # If enabled in future, this computation would need to move after coach lookup
            rebuild_cred, rebuild_delta, team_regression, rebuild_diag = self._compute_credible_rebuild_relief(
                team=team,
                returning_ppa=ret_ppa,
                talent_normalized=talent_norm_value,
                portal_impact=portal_adj,
                raw_prior=raw_prior,
                mean_prior=mean_prior,
                coach_pedigree=None,  # Coach lookup happens later; disabled by default
            )

            # Log triggered credible rebuilds
            if rebuild_diag["triggered"]:
                logger.debug(
                    f"CREDIBLE REBUILD: {team} year={year} "
                    f"rp={ret_ppa:.1%} reg_base={rebuild_diag['reg_base']:.3f} "
                    f"reg_new={rebuild_diag['reg_new']:.3f} "
                    f"talent_norm={rebuild_diag['talent_norm']:.2f} "
                    f"portal_norm={rebuild_diag['portal_norm']:.2f} "
                    f"credibility={rebuild_cred:.2f} relief={rebuild_diag['relief']:.1%}"
                )

            # Get prior year rating (regressed toward mean)
            # NOTE: rebuild_delta is applied in blend_with_inseason with week taper
            # Here we use the BASE regression (not relieved) for the stored rating
            # The relief_delta is stored separately and applied with week taper
            base_regression = self._get_regression_factor(ret_ppa, raw_prior, mean_prior)
            if raw_prior is not None:
                regressed_prior = (
                    raw_prior * (1 - base_regression)
                    + mean_prior * base_regression
                )
                prior_confidence = 0.8
            else:
                regressed_prior = 0.0  # Average team
                prior_confidence = 0.2

            # Get talent score (both raw and normalized)
            if team in talent_normalized:
                talent_score = talent_normalized[team]  # On SP+ scale (z-score * 12)
                talent_raw = talent[team]  # Raw API value (~500-1000)
                talent_confidence = 0.6
            else:
                talent_score = 0.0
                talent_raw = 0.0
                talent_confidence = 0.1

            # Check for coaching change adjustment
            talent_rank = talent_ranks.get(team, 70)  # Default to middle if unknown
            perf_rank = perf_ranks.get(team, 70)

            prior_wt, talent_wt, forget_factor, coach_name = self._calculate_coaching_change_weights(
                team, year, talent_rank, perf_rank
            )

            is_new_hc = forget_factor > 0 or coach_name != ""
            coaching_adj = 0.0

            # Default extremity scale (used for coaching adjustment tracking)
            # Must be initialized before branching to avoid NameError if triple-option
            # team has a coaching change (triple-option branch doesn't set this)
            extremity_talent_scale = 1.0

            # Calculate combined rating
            if team in prior_sp and team in talent_normalized:
                # Have both sources - apply potentially adjusted weights
                #
                # SPECIAL CASE: Triple-option teams (service academies)
                # These teams have very low recruiting talent due to unique constraints
                # (service commitment, physical requirements) but the triple-option
                # scheme lets them punch way above their talent weight.
                # Use 100% prior rating, no talent blend.
                if team in TRIPLE_OPTION_TEAMS:
                    final_prior_wt = 1.0
                    final_talent_wt = 0.0
                    logger.debug(f"{team}: Triple-option team, using 100% prior rating")
                # For extreme teams, reduce talent weight to preserve proven performance
                # Talent regression hurts accuracy for outlier teams (very good or very bad)
                # because talent doesn't capture scheme advantages or systemic issues
                elif raw_prior is not None:
                    distance_from_mean = abs(raw_prior - mean_prior)
                    if distance_from_mean >= 20:
                        # Very extreme team - trust prior more, talent less
                        extremity_talent_scale = 0.5  # Halve talent weight
                    elif distance_from_mean >= 12:
                        # Moderately extreme - gradual reduction
                        extremity_talent_scale = 1.0 - 0.5 * (distance_from_mean - 12) / 8
                    else:
                        extremity_talent_scale = 1.0

                    # Apply extremity scaling to coaching-adjusted weights
                    final_talent_wt = talent_wt * extremity_talent_scale
                    final_prior_wt = 1.0 - final_talent_wt
                else:
                    final_prior_wt = prior_wt
                    final_talent_wt = talent_wt

                combined = (
                    regressed_prior * final_prior_wt
                    + talent_score * final_talent_wt
                )

                # Track coaching adjustment impact (compared to default weights)
                if is_new_hc and forget_factor > 0:
                    # Calculate what rating would be with default weights
                    default_talent_scale = extremity_talent_scale if raw_prior is not None else 1.0
                    default_talent_wt = self.talent_weight * default_talent_scale
                    default_prior_wt = 1.0 - default_talent_wt
                    default_combined = (
                        regressed_prior * default_prior_wt
                        + talent_score * default_talent_wt
                    )
                    coaching_adj = combined - default_combined
                    if abs(coaching_adj) > 0.1:
                        coaching_adjustments.append((team, coach_name, coaching_adj, talent_rank, perf_rank))

                confidence = min(prior_confidence, talent_confidence) + 0.2
            elif team in prior_sp:
                # Only have prior rating
                combined = regressed_prior
                confidence = prior_confidence
            elif team in talent_normalized:
                # Only have talent
                combined = talent_score
                confidence = talent_confidence
            else:
                # No data - assume average
                combined = 0.0
                confidence = 0.1

            # Apply portal impact as a direct rating adjustment
            # Portal is a directional talent change: positive portal = team got better.
            # Decoupled from regression factor to avoid the "bad team anchor" bug where
            # adding portal talent to a bad team reduced regression and locked them into
            # their bad prior instead of helping them improve.
            # Scale: portal_adj (±0.12 max) * PORTAL_TO_POINTS (12) ≈ ±1.4 pts max
            PORTAL_TO_POINTS = 12.0  # Matches rating_std normalization target
            if portal_adj != 0.0:
                portal_rating_bonus = portal_adj * PORTAL_TO_POINTS
                combined += portal_rating_bonus
                if abs(portal_rating_bonus) > 0.3:
                    logger.debug(
                        f"  {team}: Portal rating adjustment {portal_rating_bonus:+.1f} pts "
                        f"(portal_adj={portal_adj:+.3f})"
                    )

            # Boost confidence if we have returning production data
            if ret_ppa is not None:
                confidence = min(confidence + 0.1, 1.0)

            # Store raw SP+ from prior year (before any JP+ adjustments)
            # This is used for the Phase 1 delta-shrink experiment
            sp_prior_raw = prior_sp.get(team, 0.0)

            self.preseason_ratings[team] = PreseasonRating(
                team=team,
                prior_rating=regressed_prior,
                talent_score=talent_raw,
                talent_rating_normalized=talent_score,  # P0.1: on SP+ scale
                returning_ppa=ret_ppa if ret_ppa is not None else 0.5,
                combined_rating=combined,
                confidence=min(confidence, 1.0),
                new_coach=is_new_hc,
                coach_name=coach_name if is_new_hc else None,
                coaching_adjustment=coaching_adj,
                portal_adjustment=portal_adj,
                talent_discount=1.0,
                rebuild_credibility=rebuild_cred,
                rebuild_relief_delta=rebuild_delta,
                sp_prior=sp_prior_raw,  # Raw SP+ for delta-shrink experiment
            )

        logger.info(f"Generated preseason ratings for {len(self.preseason_ratings)} teams")

        # Log returning production range
        if returning_prod:
            high_ret = max(returning_prod.items(), key=lambda x: x[1])
            low_ret = min(returning_prod.items(), key=lambda x: x[1])
            logger.info(
                f"Returning production range: {low_ret[0]} ({low_ret[1]:.0%}) to "
                f"{high_ret[0]} ({high_ret[1]:.0%})"
            )

        # Log coaching adjustments (P3.9: per-team details at debug level)
        if coaching_adjustments:
            logger.info(f"Applied coaching change adjustments to {len(coaching_adjustments)} teams")
            for team, coach, adj, t_rank, p_rank in sorted(coaching_adjustments, key=lambda x: -abs(x[2])):
                direction = "↑" if adj > 0 else "↓"
                logger.debug(
                    f"  {team} ({coach}): {direction}{abs(adj):.1f} pts "
                    f"(talent #{t_rank}, perf #{p_rank})"
                )

        # Log credible rebuild adjustments with comprehensive diagnostics
        if self.credible_rebuild_enabled:
            # Count all teams with RP data for trigger rate calculation
            teams_with_rp = [
                r for r in self.preseason_ratings.values()
                if r.returning_ppa is not None
            ]
            total_evaluated = len(teams_with_rp)

            # Identify triggered teams (those with non-zero relief)
            rebuild_teams = [
                r for r in self.preseason_ratings.values()
                if r.rebuild_credibility > 0 and r.rebuild_relief_delta > 0.01
            ]
            triggered_count = len(rebuild_teams)
            trigger_rate = triggered_count / total_evaluated if total_evaluated > 0 else 0

            if rebuild_teams:
                # Compute mean stats for triggered teams
                mean_rp = sum(r.returning_ppa for r in rebuild_teams) / triggered_count
                mean_relief = sum(r.rebuild_relief_delta for r in rebuild_teams) / triggered_count

                # Log summary with trigger rate
                logger.info(
                    f"Credible Rebuild triggered for {triggered_count}/{total_evaluated} teams "
                    f"({trigger_rate:.1%}) — RP <= {self.credible_rebuild_rp_cutoff:.0%}, "
                    f"talent >= {self.credible_rebuild_talent_threshold:.0%} OR portal >= {self.credible_rebuild_portal_threshold:.0%}"
                )
                logger.info(
                    f"  Mean RP: {mean_rp:.1%}, Mean relief: +{mean_relief:.2f} pts"
                )

                # Warn if trigger rate exceeds 20%
                if trigger_rate > 0.20:
                    logger.warning(
                        f"CREDIBLE REBUILD WARNING: Trigger rate {trigger_rate:.1%} exceeds 20% target. "
                        f"Consider tightening thresholds (talent={self.credible_rebuild_talent_threshold}, "
                        f"portal={self.credible_rebuild_portal_threshold})."
                    )

                # Log individual teams with rating deltas
                for r in sorted(rebuild_teams, key=lambda x: -x.rebuild_relief_delta):
                    logger.debug(
                        f"  {r.team}: RP={r.returning_ppa:.0%}, cred={r.rebuild_credibility:.2f}, "
                        f"relief=+{r.rebuild_relief_delta:.2f} pts"
                    )
            else:
                logger.debug(
                    f"Credible Rebuild: 0/{total_evaluated} teams triggered "
                    f"(no low-RP teams met quality thresholds)"
                )

        return self.preseason_ratings

    def get_preseason_rating(self, team: str) -> float:
        """Get preseason rating for a team.

        Args:
            team: Team name

        Returns:
            Preseason rating (0.0 if unknown)
        """
        if team in self.preseason_ratings:
            return self.preseason_ratings[team].combined_rating
        return 0.0

    @staticmethod
    def calculate_decayed_talent_weight(
        games_played: int,
        w_base: float = 0.08,
        w_min: float = 0.03,
        target_week: int = 10,
    ) -> float:
        """Calculate temporally-decayed talent floor weight.

        Replaces the static talent_floor_weight (0.08) with a dynamic value
        that decays linearly as in-season data becomes more robust.

        Rationale: Early in the season, recruiting talent is a strong prior
        signal (Talent > Performance when N_games < 4). By mid-season, actual
        play-by-play efficiency is far more predictive, and the talent floor
        should shrink to avoid inflating teams whose on-field results diverge
        from their recruiting profile (the "Talent Mirage").

        Formula: W(t) = max(W_base - t * decay_rate, W_min)
        where decay_rate = (W_base - W_min) / target_week

        Args:
            games_played: Number of games played (pred_week - 1)
            w_base: Starting weight at week 0 (default 0.08)
            w_min: Minimum floor weight (default 0.03)
            target_week: Week at which decay reaches w_min (default 10)

        Returns:
            Decayed talent weight (between w_min and w_base)

        Examples:
            Week 0 (preseason): 0.080
            Week 3:             0.065
            Week 5:             0.055
            Week 10+:           0.030
        """
        if games_played <= 0:
            return w_base
        decay_rate = (w_base - w_min) / target_week
        return max(w_base - games_played * decay_rate, w_min)

    def blend_with_inseason(
        self,
        inseason_ratings: dict[str, float],
        games_played: int,
        games_for_full_weight: int = 9,
        talent_floor_weight: float = 0.08,
    ) -> dict[str, float]:
        """Blend preseason ratings with in-season ratings (SP+ style).

        Uses non-linear fade matching SP+ methodology:
        - Weeks 0-3: Priors dominant (~95% -> ~65%)
        - Weeks 4-5: Tipping point (~50%)
        - Weeks 8-9: Priors nearly gone (~5%)

        The talent floor decays from talent_floor_weight (0.08) at week 0
        down to 0.03 by week 10, preventing late-season inflation of
        high-talent teams with poor on-field efficiency.

        Args:
            inseason_ratings: Current in-season ratings
            games_played: Average games played per team (or weeks into season)
            games_for_full_weight: Games needed before in-season dominates (default 9)
            talent_floor_weight: Base talent weight at week 0 (default 0.08 = 8%).
                Decays to 0.03 by week 10 via calculate_decayed_talent_weight().

        Returns:
            Blended ratings dictionary
        """
        # Apply temporal decay to talent floor weight (if enabled)
        if self.use_talent_decay:
            effective_talent_weight = self.calculate_decayed_talent_weight(
                games_played, w_base=talent_floor_weight,
            )
        else:
            effective_talent_weight = talent_floor_weight

        # Non-linear fade curve matching SP+ methodology
        # Uses sigmoid-like curve: slower fade early, faster in middle, levels off
        if games_played <= 0:
            # Week 0: 100% preseason (prior + talent), 0% in-season
            prior_weight = 1.0 - effective_talent_weight
            inseason_weight = 0.0
        elif games_played >= games_for_full_weight:
            prior_weight = 0.05  # Small residual prior weight even late
            inseason_weight = 1.0 - prior_weight - effective_talent_weight
        else:
            # Sigmoid-style curve for smoother transition
            # At week 3: ~65%, week 5: ~50%, week 7: ~25%, week 9: ~5%
            t = games_played / games_for_full_weight
            # Modified sigmoid: steeper in middle, flatter at ends
            prior_weight = 0.92 * (1.0 - t ** 1.5) ** 1.2
            prior_weight = max(prior_weight, 0.05)
            inseason_weight = 1.0 - prior_weight - effective_talent_weight

        # Invariant: weights must sum to 1.0 for proper blending
        weight_sum = prior_weight + inseason_weight + effective_talent_weight
        assert abs(weight_sum - 1.0) < 1e-9, (
            f"Blend weights must sum to 1.0, got {weight_sum:.6f} "
            f"(prior={prior_weight:.4f}, inseason={inseason_weight:.4f}, "
            f"talent={effective_talent_weight:.4f})"
        )

        logger.debug(
            f"Blending week {games_played}: prior={prior_weight:.1%}, "
            f"inseason={inseason_weight:.1%}, "
            f"talent_floor={effective_talent_weight:.1%} "
            f"(base={talent_floor_weight:.1%})"
        )

        # DETERMINISM: Sort for consistent iteration order
        all_teams = sorted(set(inseason_ratings.keys()) | set(self.preseason_ratings.keys()))
        blended = {}

        # Track high-talent outlier impact for diagnostics
        high_talent_reductions = []

        # Credible Rebuild: compute week taper (1.0 at week 0, 0.0 at week_end)
        # Relief is strongest early season, fades to protect Core phase
        rebuild_taper = max(0.0, 1.0 - games_played / self.credible_rebuild_week_end)
        rebuild_applied = []  # Track for diagnostics

        for team in all_teams:
            preseason = self.get_preseason_rating(team)
            inseason = inseason_ratings.get(team, 0.0)

            # P0.1: Use normalized talent (already on SP+ rating scale) for persistent floor
            talent_rating = 0.0
            rebuild_delta = 0.0
            if team in self.preseason_ratings:
                talent_rating = self.preseason_ratings[team].talent_rating_normalized
                # Apply week-tapered credible rebuild relief
                if self.credible_rebuild_enabled and rebuild_taper > 0:
                    rebuild_delta = (
                        self.preseason_ratings[team].rebuild_relief_delta * rebuild_taper
                    )
                    if rebuild_delta > 0.1:  # Only track significant adjustments
                        rebuild_applied.append((team, rebuild_delta))

            # Apply blended rating with optional rebuild relief
            # The relief_delta adjusts the preseason component (reduces regression penalty)
            blended[team] = (
                (preseason + rebuild_delta) * prior_weight
                + inseason * inseason_weight
                + talent_rating * effective_talent_weight
            )

            # Log reduction for high-talent outliers (talent_composite > 90th pct)
            # Normalized talent > ~15 indicates elite recruiting (top ~15 programs)
            if talent_rating > 15.0 and effective_talent_weight < talent_floor_weight:
                static_contribution = talent_rating * talent_floor_weight
                decayed_contribution = talent_rating * effective_talent_weight
                reduction = static_contribution - decayed_contribution
                high_talent_reductions.append((team, talent_rating, reduction))

        # Log high-talent outlier reductions
        if high_talent_reductions:
            high_talent_reductions.sort(key=lambda x: -x[2])
            logger.debug(
                f"Talent floor decay shaved points from {len(high_talent_reductions)} "
                f"high-talent teams (week {games_played}):"
            )
            for team, talent, reduction in high_talent_reductions[:10]:
                logger.debug(
                    f"  {team}: talent={talent:.1f}, "
                    f"reduction={reduction:-.2f} pts "
                    f"(static={talent * talent_floor_weight:.2f} → "
                    f"decayed={talent * effective_talent_weight:.2f})"
                )

        # Log Credible Rebuild applications (reduced regression for low-RP + quality priors)
        if rebuild_applied and rebuild_taper > 0:
            rebuild_applied.sort(key=lambda x: -x[1])  # Sort by relief magnitude
            logger.debug(
                f"Credible Rebuild: {len(rebuild_applied)} teams received relief "
                f"(week {games_played}, taper={rebuild_taper:.2f}):"
            )
            for team, delta in rebuild_applied[:10]:
                pr = self.preseason_ratings.get(team)
                if pr:
                    logger.debug(
                        f"  {team}: +{delta:.2f} pts "
                        f"(cred={pr.rebuild_credibility:.2f}, "
                        f"RP={pr.returning_ppa:.2f})"
                    )

        return blended

    def blend_with_inseason_sp(
        self,
        inseason_ratings: dict[str, float],
        games_played: int,
        games_for_full_weight: int = 9,
    ) -> dict[str, float]:
        """Blend raw SP+ priors with in-season ratings (no JP+ adjustments).

        Used for Phase 1 delta-shrink experiment to isolate priors delta.
        Uses same blend curve as blend_with_inseason but:
        - No talent floor component
        - No credible rebuild relief
        - Uses raw SP+ (sp_prior) instead of combined_rating

        Args:
            inseason_ratings: Current in-season ratings
            games_played: Average games played per team
            games_for_full_weight: Games needed before in-season dominates

        Returns:
            Blended ratings using raw SP+ priors
        """
        # Same blend curve as JP+ version but no talent floor
        if games_played <= 0:
            prior_weight = 1.0
            inseason_weight = 0.0
        elif games_played >= games_for_full_weight:
            prior_weight = 0.05
            inseason_weight = 0.95
        else:
            t = games_played / games_for_full_weight
            prior_weight = 0.92 * (1.0 - t ** 1.5) ** 1.2
            prior_weight = max(prior_weight, 0.05)
            inseason_weight = 1.0 - prior_weight

        logger.debug(
            f"SP+ blend week {games_played}: prior={prior_weight:.1%}, "
            f"inseason={inseason_weight:.1%} (no talent floor)"
        )

        all_teams = sorted(set(inseason_ratings.keys()) | set(self.preseason_ratings.keys()))
        blended = {}

        for team in all_teams:
            # Use raw SP+ prior (not JP+ combined_rating)
            sp_prior = 0.0
            if team in self.preseason_ratings:
                sp_prior = self.preseason_ratings[team].sp_prior
            inseason = inseason_ratings.get(team, 0.0)

            # Simple blend: no talent floor, no rebuild delta
            blended[team] = sp_prior * prior_weight + inseason * inseason_weight

        return blended

    def get_ratings_dataframe(self) -> pd.DataFrame:
        """Get preseason ratings as DataFrame.

        Returns:
            DataFrame with preseason ratings sorted by combined rating
        """
        if not self.preseason_ratings:
            return pd.DataFrame()

        data = [
            {
                "team": r.team,
                "prior_rating": r.prior_rating,
                "talent_score": r.talent_score,
                "returning_ppa": r.returning_ppa,
                "combined_rating": r.combined_rating,
                "confidence": r.confidence,
                "new_coach": r.new_coach,
                "coach_name": r.coach_name,
                "coaching_adjustment": r.coaching_adjustment,
                "portal_adjustment": r.portal_adjustment,
            }
            for r in self.preseason_ratings.values()
        ]

        df = pd.DataFrame(data)
        return df.sort_values("combined_rating", ascending=False).reset_index(drop=True)
