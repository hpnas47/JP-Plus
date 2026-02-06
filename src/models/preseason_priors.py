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

import logging
from dataclasses import dataclass
from typing import Optional

import cfbd
import numpy as np
import pandas as pd

from config.settings import get_settings

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
    # Elite tier (1.27 - 1.30): 65%+ win rate with prior HC experience
    "Brian Kelly": 1.30,        # Notre Dame (113-40) before LSU
    "Lincoln Riley": 1.30,      # Oklahoma (55-10) before USC
    "Luke Fickell": 1.30,       # Cincinnati (57-18) before Wisconsin
    "Lane Kiffin": 1.30,        # FAU, Ole Miss success before LSU
    "Hugh Freeze": 1.27,        # Ole Miss, Liberty before Auburn
    "Kalen DeBoer": 1.25,       # Fresno State success before Washington/Alabama
    "Billy Napier": 1.25,       # Louisiana (40-12) before Florida
    "Mario Cristobal": 1.20,    # Oregon (35-13) before Miami

    # Strong tier (1.20 - 1.25): Proven G5 HC success
    "Curt Cignetti": 1.25,      # JMU dominance (19-4) before Indiana
    "Jon Sumrall": 1.25,        # Tulane success (32-9) before Florida
    "Manny Diaz": 1.20,         # Miami experience before Duke
    "Ryan Silverfield": 1.21,   # Memphis (42-20) before Arkansas
    "Jamey Chadwell": 1.21,     # Coastal Carolina before Liberty
    "Joey McGuire": 1.10,       # Limited HC experience before Texas Tech

    # Above average tier (1.10 - 1.19): Some HC experience
    "Matt Rhule": 1.15,         # Temple, Baylor before Nebraska
    "Brent Venables": 1.10,     # First P5 HC job but DC pedigree - borderline

    # Below average tier: Concerning prior HC record
    "Jedd Fisch": 0.95,         # Below .500 at Arizona before Washington
    "Jeff Lebby": 0.90,         # Limited/poor HC record

    # Default: 0 means first-time HC - will be EXCLUDED from adjustment
    "default": 0,
}

# First-time HCs - explicitly excluded from coaching change adjustment
# These coaches have no prior HC record, so we have no basis to predict improvement
FIRST_TIME_HCS = {
    "Dan Lanning",      # First HC job at Oregon (2022) - elite DC but no HC record
    "Deion Sanders",    # No prior FBS HC record before Colorado
    "Brent Key",        # Internal promotion at Georgia Tech
    "Kenny Dillingham", # First HC job at Arizona State
    "Troy Taylor",      # FCS only before Stanford
    "Sherrone Moore",   # Internal promotion at Michigan
    "Pete Golding",     # First HC job
}

# Coaching changes by year: {year: {team: coach_name}}
# Only track NEW head coaches (first year at program) who have PRIOR HC experience.
# First-time HCs are listed in FIRST_TIME_HCS and excluded from adjustment.
COACHING_CHANGES = {
    2022: {
        "LSU": "Brian Kelly",           # Notre Dame success
        "USC": "Lincoln Riley",         # Oklahoma success
        "Miami": "Mario Cristobal",     # Oregon success
        "Florida": "Billy Napier",      # Louisiana success
        # P0.3: Dan Lanning removed - first-time HC (in FIRST_TIME_HCS), no prior HC record
        "Texas Tech": "Joey McGuire",   # First P5 HC but had HC experience
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
    2025: {
        "LSU": "Lane Kiffin",
        "Florida": "Jon Sumrall",
        "Arkansas": "Ryan Silverfield",
    },
}

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

TRIPLE_OPTION_TEAMS = frozenset({
    "Army",
    "Navy",
    "Air Force",
    "Kennesaw State",
})

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


def get_coach_pedigree(coach_name: str) -> float:
    """Get pedigree multiplier for a coach.

    Args:
        coach_name: Coach's name

    Returns:
        Pedigree multiplier (0.90 - 1.30), or 0 if first-time HC (excluded)
    """
    # First-time HCs are explicitly excluded
    if coach_name in FIRST_TIME_HCS:
        return 0

    return COACH_PEDIGREE.get(coach_name, COACH_PEDIGREE["default"])


def is_new_coach(team: str, year: int) -> tuple[bool, Optional[str]]:
    """Check if team has a new head coach for the given year.

    Args:
        team: Team name
        year: Season year

    Returns:
        Tuple of (is_new_coach, coach_name or None)
    """
    year_changes = COACHING_CHANGES.get(year, {})
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
        prior_year_weight: float = 0.6,
        talent_weight: float = 0.4,
        regression_factor: float = 0.3,
    ):
        """Initialize preseason priors calculator.

        Args:
            prior_year_weight: Weight for previous year's rating (0-1)
            talent_weight: Weight for talent composite (0-1)
            regression_factor: How much to regress prior ratings toward mean (0-1)
        """
        self.prior_year_weight = prior_year_weight
        self.talent_weight = talent_weight
        self.regression_factor = regression_factor

        settings = get_settings()
        self.configuration = cfbd.Configuration()
        self.configuration.access_token = settings.cfbd_api_key

        self.ratings_api = cfbd.RatingsApi(cfbd.ApiClient(self.configuration))
        self.teams_api = cfbd.TeamsApi(cfbd.ApiClient(self.configuration))
        self.players_api = cfbd.PlayersApi(cfbd.ApiClient(self.configuration))

        self.preseason_ratings: dict[str, PreseasonRating] = {}

    def fetch_prior_year_sp(self, year: int) -> dict[str, float]:
        """Fetch SP+ ratings from the previous year.

        Args:
            year: The CURRENT season year (will fetch year-1)

        Returns:
            Dictionary mapping team name to SP+ rating
        """
        prior_year = year - 1
        try:
            sp_ratings = self.ratings_api.get_sp(year=prior_year)
            ratings = {}
            for team in sp_ratings:
                if team.rating is not None:
                    ratings[team.team] = team.rating
            logger.info(f"Fetched {len(ratings)} SP+ ratings from {prior_year}")
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
        try:
            talent = self.teams_api.get_talent(year=year)
            scores = {}
            for team in talent:
                scores[team.team] = team.talent
            logger.info(f"Fetched talent scores for {len(scores)} teams")
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
        try:
            rp_data = self.players_api.get_returning_production(year=year)
            returning = {}
            for team in rp_data:
                if team.percent_ppa is not None:
                    returning[team.team] = team.percent_ppa
            logger.info(f"Fetched returning production for {len(returning)} teams")
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
        try:
            transfers = self.players_api.get_transfer_portal(year=year)
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
            usage = self.players_api.get_player_usage(year=year)
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
    P4_CONFERENCES = {
        'SEC', 'Big Ten', 'Big 12', 'ACC',
        # Include independents that play P4-level schedules
        'Notre Dame', 'FBS Independents',
    }

    # Teams that are P4 level (for edge cases and independents)
    P4_TEAMS = {
        'Notre Dame', 'BYU', 'USC', 'UCLA', 'Oregon', 'Washington',
        'Colorado', 'Arizona', 'Arizona State', 'Utah',
        # Any other P4-level independents
    }

    # High-contact positions for physicality tax (G5→P4 transfers)
    HIGH_CONTACT_POSITIONS = {'OT', 'IOL', 'IDL', 'LB', 'EDGE'}

    # Skill positions for athleticism discount (G5→P4 transfers)
    SKILL_POSITIONS = {'WR', 'RB', 'CB', 'S'}

    # Level-up discount factors
    PHYSICALITY_TAX = 0.75      # 25% discount for trench players G5→P4
    ATHLETICISM_DISCOUNT = 0.90  # 10% discount for skill players G5→P4
    CONTINUITY_TAX = 0.90       # 11% penalty for losing incumbent (Option A calibration)

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

        Args:
            year: Season year

        Returns:
            Set of FBS team names
        """
        try:
            teams = self.teams_api.get_fbs_teams(year=year)
            fbs_set = {t.school for t in teams}
            logger.debug(f"Fetched {len(fbs_set)} FBS teams for {year}")
            return fbs_set
        except Exception as e:
            logger.warning(f"Could not fetch FBS teams for {year}: {e}")
            return set()

    def _fetch_team_conferences(self, year: int) -> dict[str, str]:
        """Fetch conference affiliation for all teams.

        Args:
            year: Season year

        Returns:
            Dict mapping team name to conference name
        """
        try:
            teams = self.teams_api.get_teams()
            conf_map = {}
            for t in teams:
                if t.school and t.conference:
                    conf_map[t.school] = t.conference
            logger.debug(f"Fetched conference data for {len(conf_map)} teams")
            return conf_map
        except Exception as e:
            logger.warning(f"Could not fetch team conferences: {e}")
            return {}

    def calculate_portal_impact(
        self,
        year: int,
        portal_scale: float = 0.06,
        fbs_only: bool = True,
        impact_cap: float = 0.12,
    ) -> dict[str, float]:
        """Calculate net transfer portal impact for each team using unit-level analysis.

        Uses scarcity-based position weights, level-up discounts for G5→P4 moves,
        and continuity tax for losing incumbents. Reflects 2026 market reality
        where elite trench play is the primary driver of rating stability.

        Args:
            year: Season year (fetches transfers FOR this year)
            portal_scale: How much to scale portal impact (default 0.06)
            fbs_only: If True, only include FBS teams in results (default True)
            impact_cap: Maximum team-wide portal impact (default ±12%)

        Returns:
            Dictionary mapping team name to adjusted returning production modifier
            Positive = net gain from portal, Negative = net loss
        """
        # Fetch transfer portal entries for this year
        transfers_df = self.fetch_transfer_portal(year)
        if transfers_df.empty:
            logger.warning(f"No transfer portal data for {year}")
            return {}

        # Fetch FBS teams for filtering
        fbs_teams = self.fetch_fbs_teams(year) if fbs_only else set()
        if fbs_only and not fbs_teams:
            logger.warning(f"Could not fetch FBS teams for {year}, using all teams")
            fbs_only = False

        # Fetch conference data for level-up discount logic
        team_conferences = self._fetch_team_conferences(year)

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

        # Add position group column
        transfers_df['pos_group'] = transfers_df['position'].apply(self._get_position_group)

        # Calculate OUTGOING value (what team loses - no level discount, full value)
        # This is the "raw" value of the player leaving
        transfers_df['outgoing_value'] = transfers_df.apply(
            lambda row: self._calculate_player_value(
                row.get('stars'),
                row.get('rating'),
                row['pos_group'],
                origin=None,  # No level discount for outgoing
                destination=None,
                team_conferences=None,
            ),
            axis=1
        )

        # Calculate INCOMING value (what team gains - WITH level-up discount)
        # G5→P4 transfers get discounted based on physicality/athleticism
        transfers_df['incoming_value'] = transfers_df.apply(
            lambda row: self._calculate_player_value(
                row.get('stars'),
                row.get('rating'),
                row['pos_group'],
                origin=row.get('origin'),
                destination=row.get('destination'),
                team_conferences=team_conferences,
            ),
            axis=1
        )

        # Log position group distribution
        pos_dist = transfers_df['pos_group'].value_counts()
        logger.info(f"Transfer portal {year}: {len(transfers_df)} FBS-relevant transfers")
        logger.debug(f"Position distribution: {pos_dist.to_dict()}")

        # Calculate outgoing value per team (players who left)
        # Apply CONTINUITY_TAX: losing an incumbent hurts even if "replaced"
        outgoing_raw = transfers_df.groupby('origin')['outgoing_value'].sum()
        outgoing = outgoing_raw / self.CONTINUITY_TAX  # Amplify loss (divide by 0.85 = ~18% boost)

        # Calculate incoming value per team (players who arrived)
        # Level-up discounts already applied in incoming_value
        incoming_df = transfers_df[transfers_df['destination'].notna()]
        incoming = incoming_df.groupby('destination')['incoming_value'].sum()

        # Log coverage and level-up stats
        has_dest = len(incoming_df)
        logger.info(
            f"Portal coverage: {has_dest}/{len(transfers_df)} ({has_dest/len(transfers_df):.1%}) "
            f"have destinations"
        )

        # Log G5→P4 transfer count
        if team_conferences:
            g5_to_p4 = incoming_df.apply(
                lambda row: (
                    not self._is_p4_team(row.get('origin'), team_conferences) and
                    self._is_p4_team(row.get('destination'), team_conferences)
                ),
                axis=1
            ).sum()
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

            # Scale the impact and apply tighter cap (±12%)
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
        new_coach, coach_name = is_new_coach(team, year)

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
                # Significant overperformer with new coach - slight regression
                forget_factor = 0.15  # Mild forget factor for prior success
                prior_weight = self.prior_year_weight * (1 + forget_factor * 0.5)
                talent_weight = 1 - prior_weight
                logger.debug(
                    f"{team}: Overperformer with new coach, mild regression "
                    f"(talent #{talent_rank}, perf #{performance_rank})"
                )
            else:
                # At expectation - just add uncertainty, no directional change
                return self.prior_year_weight, self.talent_weight, 0.0, coach_name
        else:
            # Underperformer with new coach - apply forget factor
            # Get coach pedigree first - if 0, coach is first-time HC and excluded
            pedigree = get_coach_pedigree(coach_name) if coach_name else 0

            if pedigree == 0:
                # First-time HC - no prior record to base adjustment on
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
        logger.debug(
            f"  SP+ ∩ talent: {len(sp_teams & talent_teams)}, "
            f"SP+ ∩ returning: {len(sp_teams & ret_teams)}, "
            f"all four: {len(sp_teams & talent_teams & ret_teams & portal_teams)}"
        )

        # Validate talent direction: known elite programs should have high raw scores
        # Higher talent score = better recruiting = should rank near top
        elite_programs = ["Alabama", "Georgia", "Ohio State", "Texas", "LSU"]
        if talent:
            talent_sorted = sorted(talent.items(), key=lambda x: -x[1])
            top_20_teams = {t for t, _ in talent_sorted[:20]}
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
            portal_impact = self.calculate_portal_impact(year, portal_scale=portal_scale)

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

            # Adjust returning production by transfer portal impact
            # Portal impact is a modifier: positive = net gain, negative = net loss
            portal_adj = portal_impact.get(team, 0.0)
            if ret_ppa is not None:
                # Effective returning = base returning + portal adjustment
                # Portal adjustment is already scaled and capped at ±15%
                effective_ret_ppa = max(0.0, min(1.0, ret_ppa + portal_adj))
            else:
                effective_ret_ppa = None

            # Get raw prior rating first (needed for asymmetric regression)
            raw_prior = prior_sp.get(team)

            # Apply triple-option boost to raw prior
            # These teams are systematically underrated by SP+ efficiency metrics
            triple_option_boost = get_triple_option_boost(team)
            if raw_prior is not None and triple_option_boost > 0:
                raw_prior += triple_option_boost
                logger.debug(f"{team}: Applied triple-option boost of +{triple_option_boost:.1f} pts")

            # Calculate team-specific regression factor based on:
            # 1. Effective returning production (higher = less regression)
            # 2. Distance from mean (farther = less regression - asymmetric)
            team_regression = self._get_regression_factor(
                effective_ret_ppa,
                raw_prior=raw_prior,
                mean_prior=mean_prior,
            )

            # Get prior year rating (regressed toward mean)
            if raw_prior is not None:
                regressed_prior = (
                    raw_prior * (1 - team_regression)
                    + mean_prior * team_regression
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

            # Boost confidence if we have returning production data
            if ret_ppa is not None:
                confidence = min(confidence + 0.1, 1.0)

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

        Additionally, talent composite persists as a floor all year.
        This prevents elite-talent teams from dropping too far with bad performance.

        Args:
            inseason_ratings: Current in-season ratings
            games_played: Average games played per team (or weeks into season)
            games_for_full_weight: Games needed before in-season dominates (default 9)
            talent_floor_weight: Persistent talent weight all year (default 0.08 = 8%)

        Returns:
            Blended ratings dictionary
        """
        # Non-linear fade curve matching SP+ methodology
        # Uses sigmoid-like curve: slower fade early, faster in middle, levels off
        if games_played <= 0:
            prior_weight = 0.95 - talent_floor_weight  # Adjust for talent floor
            inseason_weight = 0.0
        elif games_played >= games_for_full_weight:
            prior_weight = 0.05  # Small residual prior weight even late
            inseason_weight = 1.0 - prior_weight - talent_floor_weight
        else:
            # Sigmoid-style curve for smoother transition
            # At week 3: ~65%, week 5: ~50%, week 7: ~25%, week 9: ~5%
            t = games_played / games_for_full_weight
            # Modified sigmoid: steeper in middle, flatter at ends
            prior_weight = 0.92 * (1.0 - t ** 1.5) ** 1.2
            prior_weight = max(prior_weight, 0.05)
            inseason_weight = 1.0 - prior_weight - talent_floor_weight

        logger.debug(
            f"Blending week {games_played}: prior={prior_weight:.1%}, "
            f"inseason={inseason_weight:.1%}, talent_floor={talent_floor_weight:.1%}"
        )

        # DETERMINISM: Sort for consistent iteration order
        all_teams = sorted(set(inseason_ratings.keys()) | set(self.preseason_ratings.keys()))
        blended = {}

        for team in all_teams:
            preseason = self.get_preseason_rating(team)
            inseason = inseason_ratings.get(team, 0.0)

            # P0.1: Use normalized talent (already on SP+ rating scale) for persistent floor
            # Previously used ad hoc (raw - 750) / 25.0 which was on a different scale
            talent_rating = 0.0
            if team in self.preseason_ratings:
                talent_rating = self.preseason_ratings[team].talent_rating_normalized

            # Blend: prior + in-season + persistent talent floor
            blended[team] = (
                preseason * prior_weight
                + inseason * inseason_weight
                + talent_rating * talent_floor_weight
            )

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
