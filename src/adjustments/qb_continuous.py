"""Continuous QB Rating System for JP+ Spreads.

This module provides always-on, walk-forward-safe QB quality estimates that:
1. Adjust predicted spreads based on QB quality (Phase A)
2. Compute QB uncertainty for future bet gating (Phase B - logging only)

Walk-Forward Safety:
- For predicting week W games, only uses QB data from weeks < W
- Preseason priors from prior season with decay
- Conservative starter identification (last known starter)

Key Concepts:
- Dropbacks = pass_attempts + sacks_taken (estimated from completions/attempts)
- PPA = Predicted Points Added per dropback (from CFBD)
- Uncertainty = multiplicative composition of sample size, starter change, unknown starter
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cfbd
import pandas as pd
import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Constants
# ============================================================================

# Shrinkage parameter: higher = more shrinkage, slower confidence buildup
# At K=200, ~250 dropbacks → ~55% weight on raw signal
DEFAULT_SHRINKAGE_K = 200

# QB point adjustment cap (except for manual --qb-out override)
DEFAULT_QB_CAP = 3.0

# QB scaling factor: converts PPA difference to points
# Calibrated so 90th percentile reaches ~60-70% of cap by week 8
DEFAULT_QB_SCALE = 4.0

# Prior season decay factor
DEFAULT_PRIOR_DECAY = 0.3

# Minimum dropbacks for starter identification
MIN_RECENT_DB = 15  # Minimum dropbacks in most recent game to trigger switch
MIN_CUM_DB = 50  # Minimum cumulative dropbacks for season to be considered starter

# FBS average PPA per pass play (used as baseline for mean-centering)
# Calibrated from 2022-2024 data
FBS_AVG_PPA = 0.10

# Baking Factor Parameters
# The EFM "bakes in" QB quality through efficiency metrics as the season progresses.
# Early season: QB adjustment is additive (EFM doesn't yet reflect current QB)
# Core season: QB adjustment becomes residual (only adjust for deviation from team norm)
BAKING_WEEK_START = 1  # Week at which baking begins
BAKING_WEEK_MID = 3    # Week at which baking reaches mid-point (0.5)
BAKING_WEEK_FULL = 4   # Week at which baking reaches 1.0 (fully baked by week 4)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class QBGameStats:
    """QB statistics for a single game."""
    player_id: str
    player_name: str
    team: str
    year: int
    week: int
    pass_attempts: int
    completions: int
    ppa_pass: Optional[float]  # Average PPA on pass plays
    ppa_all: Optional[float]  # Average PPA on all plays


@dataclass
class QBSeasonStats:
    """Cumulative QB statistics for a season (through a given week)."""
    player_id: str
    player_name: str
    team: str
    year: int
    through_week: int  # Stats are cumulative through this week
    total_dropbacks: int
    total_pass_ppa: float  # Sum of PPA (not average)
    games_played: int

    @property
    def avg_ppa(self) -> float:
        """Average PPA per dropback."""
        if self.total_dropbacks == 0:
            return 0.0
        return self.total_pass_ppa / self.total_dropbacks


@dataclass
class QBProjection:
    """Projected starter and quality for a team at a given week."""
    team: str
    year: int
    pred_week: int  # Week we're predicting

    # Projected starter (based on data through pred_week - 1)
    projected_starter_id: Optional[str] = None
    projected_starter_name: Optional[str] = None
    unknown_starter: bool = True

    # Current season data
    current_season_dropbacks: int = 0
    current_season_ppa: float = 0.0  # Average PPA

    # Prior season data (if applicable)
    prior_season_dropbacks: int = 0
    prior_season_ppa: float = 0.0  # Average PPA
    prior_season_used: bool = False

    # Starter change detection
    starter_changed: bool = False  # True if starter changed recently

    @property
    def n_effective_db(self) -> float:
        """Effective dropbacks (current + decayed prior)."""
        prior_contribution = self.prior_season_dropbacks * DEFAULT_PRIOR_DECAY if self.prior_season_used else 0
        return self.current_season_dropbacks + prior_contribution


@dataclass
class QBQuality:
    """QB quality estimate with uncertainty."""
    team: str
    year: int
    pred_week: int

    # Quality metrics
    qb_value_raw: float = 0.0  # Mean-centered PPA (vs FBS average)
    qb_value_shrunk: float = 0.0  # After shrinkage
    qb_points: float = 0.0  # Final point adjustment (capped)
    qb_points_effective: float = 0.0  # After uncertainty dampening

    # Uncertainty
    qb_uncertainty: float = 1.0  # 0 = certain, 1 = very uncertain

    # Projection details (for logging)
    projection: Optional[QBProjection] = None


@dataclass
class QBMatchupDiagnostics:
    """Full diagnostics for a game's QB adjustment."""
    home_team: str
    away_team: str
    year: int
    pred_week: int

    # Per-team quality
    home_quality: Optional[QBQuality] = None
    away_quality: Optional[QBQuality] = None

    # Game-level
    spread_qb_adj: float = 0.0  # Net adjustment (positive = helps home)
    qb_uncertainty_game: float = 1.0  # Average of home/away uncertainty

    # Override status
    manual_override_home: bool = False
    manual_override_away: bool = False


# ============================================================================
# Main Class
# ============================================================================

class QBContinuousAdjuster:
    """
    Continuous QB rating system for JP+ spreads.

    Provides always-on QB quality estimates that:
    1. Are walk-forward safe (only uses data from prior weeks)
    2. Apply heavy shrinkage for low-sample QBs
    3. Detect starter changes conservatively
    4. Incorporate prior season data with decay

    Usage:
        adjuster = QBContinuousAdjuster(api_key="...", year=2024)
        adjuster.build_qb_data(through_week=3)  # Prepare data
        adjustment = adjuster.get_adjustment(home_team, away_team, pred_week=4)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        year: int = 2024,
        shrinkage_k: float = DEFAULT_SHRINKAGE_K,
        qb_cap: float = DEFAULT_QB_CAP,
        qb_scale: float = DEFAULT_QB_SCALE,
        prior_decay: float = DEFAULT_PRIOR_DECAY,
        use_prior_season: bool = True,
        fbs_avg_ppa: float = FBS_AVG_PPA,
        phase1_only: bool = False,
    ):
        """Initialize QB continuous adjuster.

        Args:
            api_key: CFBD API key (uses settings if not provided)
            year: Current season year
            shrinkage_k: Shrinkage parameter (higher = more shrinkage)
            qb_cap: Maximum QB point adjustment
            qb_scale: Scaling factor for PPA to points
            prior_decay: Decay factor for prior season data
            use_prior_season: Whether to use prior season data for Week 1
            fbs_avg_ppa: FBS average PPA for mean-centering
            phase1_only: If True, only apply QB adjustment for weeks 1-3 (skip Core)
        """
        settings = get_settings()
        self.api_key = api_key or settings.cfbd_api_key
        self.year = year
        self.shrinkage_k = shrinkage_k
        self.qb_cap = qb_cap
        self.qb_scale = qb_scale
        self.prior_decay = prior_decay
        self.use_prior_season = use_prior_season
        self.fbs_avg_ppa = fbs_avg_ppa
        self.phase1_only = phase1_only  # If True, only apply QB adjustment for weeks 1-3

        # API clients
        self._metrics_api: Optional[cfbd.MetricsApi] = None
        self._games_api: Optional[cfbd.GamesApi] = None

        # Data caches (populated by build_qb_data)
        self._game_stats: dict[tuple[str, int, int], list[QBGameStats]] = {}  # (team, year, week) -> stats
        self._season_stats: dict[tuple[str, int, int], dict[str, QBSeasonStats]] = {}  # (team, year, through_week) -> {player_id: stats}
        self._projections: dict[tuple[str, int, int], QBProjection] = {}  # (team, year, pred_week) -> projection
        self._qualities: dict[tuple[str, int, int], QBQuality] = {}  # (team, year, pred_week) -> quality

        # Prior season data
        self._prior_season_stats: dict[tuple[str, int], QBSeasonStats] = {}  # (player_id, year) -> end-of-season stats

        # Team average QB value (for residual adjustment)
        # Maps (team, year, through_week) -> weighted average qb_points of all QBs who played
        self._team_avg_qb: dict[tuple[str, int, int], float] = {}

        # Manual overrides (teams flagged as QB out)
        self._qb_out_teams: set[str] = set()
        self._qb_out_adjustments: dict[str, float] = {}  # team -> manual adjustment

        # Diagnostics storage
        self._diagnostics: list[QBMatchupDiagnostics] = []

        # Data availability tracking
        self._data_built_through_week: int = 0
        self._prior_season_loaded: bool = False

    @property
    def metrics_api(self) -> cfbd.MetricsApi:
        if self._metrics_api is None:
            config = cfbd.Configuration()
            config.access_token = self.api_key
            self._metrics_api = cfbd.MetricsApi(cfbd.ApiClient(config))
        return self._metrics_api

    @property
    def games_api(self) -> cfbd.GamesApi:
        if self._games_api is None:
            config = cfbd.Configuration()
            config.access_token = self.api_key
            self._games_api = cfbd.GamesApi(cfbd.ApiClient(config))
        return self._games_api

    # ========================================================================
    # Data Fetching
    # ========================================================================

    def _fetch_qb_ppa_for_week(self, year: int, week: int) -> list[dict]:
        """Fetch QB PPA data for a specific week."""
        try:
            ppa_data = self.metrics_api.get_predicted_points_added_by_player_game(
                year=year, week=week, position="QB"
            )
            results = []
            for p in ppa_data:
                d = p.to_dict()
                avg_ppa = d.get('averagePPA', {}) or {}
                results.append({
                    'player_id': str(d.get('id', '')),
                    'player_name': d.get('name', ''),
                    'team': d.get('team', ''),
                    'year': year,
                    'week': week,
                    'ppa_pass': avg_ppa.get('pass'),
                    'ppa_all': avg_ppa.get('all'),
                })
            return results
        except Exception as e:
            logger.warning(f"Error fetching QB PPA for {year} week {week}: {e}")
            return []

    def _fetch_pass_attempts_for_week(self, year: int, week: int) -> dict[tuple[str, str], int]:
        """Fetch pass attempts by (team, player_name) for a week.

        Returns dict mapping (team, player_name) -> pass_attempts
        """
        try:
            stats = self.games_api.get_game_player_stats(
                year=year, week=week, category="passing"
            )
            results = {}
            for game in stats:
                d = game.to_dict()
                for team_data in d.get('teams', []):
                    team = team_data.get('team', '')
                    for cat in team_data.get('categories', []):
                        if cat.get('name') == 'passing':
                            for typ in cat.get('types', []):
                                if typ.get('name') == 'C/ATT':
                                    for ath in typ.get('athletes', []):
                                        name = ath.get('name', '')
                                        stat = ath.get('stat', '0/0')
                                        # Parse "17/21" format
                                        try:
                                            comp, att = stat.split('/')
                                            attempts = int(att)
                                            results[(team, name)] = results.get((team, name), 0) + attempts
                                        except (ValueError, AttributeError):
                                            pass
            return results
        except Exception as e:
            logger.warning(f"Error fetching pass attempts for {year} week {week}: {e}")
            return {}

    def _build_game_stats_for_week(self, year: int, week: int) -> dict[str, list[QBGameStats]]:
        """Build QBGameStats for all teams for a week.

        Returns dict mapping team -> list of QBGameStats for that team.
        """
        # Fetch PPA and attempts
        ppa_data = self._fetch_qb_ppa_for_week(year, week)
        attempts_data = self._fetch_pass_attempts_for_week(year, week)

        # Group PPA by team
        team_qbs: dict[str, list[dict]] = {}
        for ppa in ppa_data:
            team = ppa['team']
            if team not in team_qbs:
                team_qbs[team] = []
            team_qbs[team].append(ppa)

        # Build stats
        results: dict[str, list[QBGameStats]] = {}
        for team, qbs in team_qbs.items():
            team_stats = []
            for qb in qbs:
                name = qb['player_name']
                attempts = attempts_data.get((team, name), 0)

                # Skip QBs with no attempts (defensive players, etc.)
                if attempts == 0 and qb['ppa_pass'] is None:
                    continue

                # Estimate dropbacks from attempts (proxy for now)
                # In ideal world, we'd add sacks_taken but that data is harder to get per-player
                dropbacks = attempts

                stat = QBGameStats(
                    player_id=qb['player_id'],
                    player_name=name,
                    team=team,
                    year=year,
                    week=week,
                    pass_attempts=attempts,
                    completions=0,  # Not tracked for now
                    ppa_pass=qb['ppa_pass'],
                    ppa_all=qb['ppa_all'],
                )
                team_stats.append(stat)

            if team_stats:
                results[team] = team_stats

        return results

    def build_qb_data(self, through_week: int) -> None:
        """Build QB data through a given week.

        This is the main data preparation method. Call before get_adjustment().

        Args:
            through_week: Build data through this week (for predicting through_week + 1)
        """
        if through_week <= self._data_built_through_week:
            return  # Already built

        logger.info(f"Building QB data for {self.year} through week {through_week}")

        # Load prior season if needed and not already loaded
        if self.use_prior_season and not self._prior_season_loaded:
            self._load_prior_season(self.year - 1)

        # Build game stats for each week
        start_week = max(1, self._data_built_through_week + 1)
        for week in range(start_week, through_week + 1):
            week_stats = self._build_game_stats_for_week(self.year, week)
            for team, stats in week_stats.items():
                self._game_stats[(team, self.year, week)] = stats

        # Build cumulative season stats
        self._build_season_stats(through_week)

        self._data_built_through_week = through_week
        logger.info(f"QB data built through week {through_week}")

    def _load_prior_season(self, prior_year: int) -> None:
        """Load prior season QB data for preseason priors."""
        logger.info(f"Loading prior season QB data from {prior_year}")

        # Fetch end-of-season PPA for prior year (all weeks)
        try:
            ppa_data = self.metrics_api.get_predicted_points_added_by_player_season(
                year=prior_year, position="QB"
            )

            for p in ppa_data:
                d = p.to_dict()
                player_id = str(d.get('id', ''))
                name = d.get('name', '')
                team = d.get('team', '')

                avg_ppa = d.get('averagePPA', {}) or {}
                total_ppa = d.get('totalPPA', {}) or {}

                pass_ppa = avg_ppa.get('pass', 0) or 0
                total_pass_ppa = total_ppa.get('pass', 0) or 0

                # Estimate dropbacks from total/avg
                if pass_ppa != 0:
                    est_dropbacks = abs(total_pass_ppa / pass_ppa)
                else:
                    est_dropbacks = 0

                if est_dropbacks > 10:  # Minimum threshold
                    stats = QBSeasonStats(
                        player_id=player_id,
                        player_name=name,
                        team=team,
                        year=prior_year,
                        through_week=99,  # End of season
                        total_dropbacks=int(est_dropbacks),
                        total_pass_ppa=total_pass_ppa,
                        games_played=0,  # Not tracked
                    )
                    self._prior_season_stats[(player_id, prior_year)] = stats

            logger.info(f"Loaded {len(self._prior_season_stats)} prior season QB records")
            self._prior_season_loaded = True

        except Exception as e:
            logger.warning(f"Error loading prior season QB data: {e}")
            self._prior_season_loaded = True  # Mark as loaded to avoid retries

    def _build_season_stats(self, through_week: int) -> None:
        """Build cumulative season stats through each week."""
        # Group game stats by team
        team_weeks: dict[str, dict[int, list[QBGameStats]]] = {}
        for (team, year, week), stats in self._game_stats.items():
            if year != self.year:
                continue
            if team not in team_weeks:
                team_weeks[team] = {}
            team_weeks[team][week] = stats

        # Build cumulative stats for each team
        for team, weeks in team_weeks.items():
            for pred_week in range(1, through_week + 2):  # +2 because we build for predicting pred_week
                # Accumulate stats through pred_week - 1
                qb_cumulative: dict[str, dict] = {}

                for week in range(1, pred_week):
                    if week not in weeks:
                        continue
                    for stat in weeks[week]:
                        pid = stat.player_id
                        if pid not in qb_cumulative:
                            qb_cumulative[pid] = {
                                'player_name': stat.player_name,
                                'total_dropbacks': 0,
                                'total_ppa': 0.0,
                                'games': 0,
                                'recent_week': 0,
                                'recent_dropbacks': 0,
                            }

                        qb_cumulative[pid]['total_dropbacks'] += stat.pass_attempts
                        if stat.ppa_pass is not None:
                            # Approximate total PPA from average * attempts
                            qb_cumulative[pid]['total_ppa'] += stat.ppa_pass * stat.pass_attempts
                        qb_cumulative[pid]['games'] += 1

                        # Track most recent game
                        if week > qb_cumulative[pid]['recent_week']:
                            qb_cumulative[pid]['recent_week'] = week
                            qb_cumulative[pid]['recent_dropbacks'] = stat.pass_attempts

                # Store season stats
                team_stats: dict[str, QBSeasonStats] = {}
                for pid, data in qb_cumulative.items():
                    if data['total_dropbacks'] > 0:
                        team_stats[pid] = QBSeasonStats(
                            player_id=pid,
                            player_name=data['player_name'],
                            team=team,
                            year=self.year,
                            through_week=pred_week - 1,
                            total_dropbacks=data['total_dropbacks'],
                            total_pass_ppa=data['total_ppa'],
                            games_played=data['games'],
                        )

                if team_stats:
                    self._season_stats[(team, self.year, pred_week)] = team_stats

    # ========================================================================
    # Starter Identification
    # ========================================================================

    def _identify_projected_starter(
        self, team: str, pred_week: int
    ) -> QBProjection:
        """Identify projected starter for a team at a given prediction week.

        Uses conservative logic:
        - Default = QB with most cumulative dropbacks through prior week
        - Switch only if new QB led team in most recent game AND has sufficient volume
        """
        key = (team, self.year, pred_week)
        if key in self._projections:
            return self._projections[key]

        proj = QBProjection(
            team=team,
            year=self.year,
            pred_week=pred_week,
        )

        # Get season stats through pred_week - 1
        season_stats = self._season_stats.get((team, self.year, pred_week), {})

        if not season_stats:
            # No data - check prior season
            if self.use_prior_season:
                proj = self._apply_prior_season(proj)
            self._projections[key] = proj
            return proj

        # Find QB with most cumulative dropbacks
        sorted_qbs = sorted(
            season_stats.values(),
            key=lambda s: s.total_dropbacks,
            reverse=True
        )

        if not sorted_qbs:
            if self.use_prior_season:
                proj = self._apply_prior_season(proj)
            self._projections[key] = proj
            return proj

        current_leader = sorted_qbs[0]

        # Check for starter change
        # Find who led in most recent completed week
        most_recent_week = max(
            w for (t, y, w), _ in self._game_stats.items()
            if t == team and y == self.year and w < pred_week
        ) if any(
            t == team and y == self.year and w < pred_week
            for (t, y, w) in self._game_stats.keys()
        ) else 0

        recent_leader = None
        recent_leader_db = 0
        if most_recent_week > 0:
            recent_stats = self._game_stats.get((team, self.year, most_recent_week), [])
            for stat in recent_stats:
                if stat.pass_attempts > recent_leader_db:
                    recent_leader_db = stat.pass_attempts
                    recent_leader = stat.player_id

        # Determine projected starter with conservative switch logic
        projected_starter = current_leader
        starter_changed = False

        if recent_leader and recent_leader != current_leader.player_id:
            # Different QB led in most recent game
            recent_stats_obj = season_stats.get(recent_leader)
            if recent_stats_obj:
                # Check switch conditions
                if (recent_leader_db >= MIN_RECENT_DB or
                    recent_stats_obj.total_dropbacks >= MIN_CUM_DB):
                    projected_starter = recent_stats_obj
                    starter_changed = True

        # Populate projection
        proj.projected_starter_id = projected_starter.player_id
        proj.projected_starter_name = projected_starter.player_name
        proj.unknown_starter = False
        proj.current_season_dropbacks = projected_starter.total_dropbacks
        proj.current_season_ppa = projected_starter.avg_ppa
        proj.starter_changed = starter_changed

        # Apply prior season data if available
        if self.use_prior_season:
            proj = self._apply_prior_season(proj)

        self._projections[key] = proj
        return proj

    def _apply_prior_season(self, proj: QBProjection) -> QBProjection:
        """Apply prior season data to projection if available."""
        if not proj.projected_starter_id:
            # Unknown starter - try to find prior season data for team
            for (pid, year), stats in self._prior_season_stats.items():
                if stats.team == proj.team and year == self.year - 1:
                    if stats.total_dropbacks > proj.prior_season_dropbacks:
                        proj.projected_starter_id = pid
                        proj.projected_starter_name = stats.player_name
                        proj.prior_season_dropbacks = stats.total_dropbacks
                        proj.prior_season_ppa = stats.avg_ppa
                        proj.prior_season_used = True
                        # Still mark as unknown since we're using prior year
                        proj.unknown_starter = proj.current_season_dropbacks == 0
            return proj

        # Have current starter - check for prior season data
        prior_stats = self._prior_season_stats.get(
            (proj.projected_starter_id, self.year - 1)
        )
        if prior_stats and prior_stats.team == proj.team:
            proj.prior_season_dropbacks = prior_stats.total_dropbacks
            proj.prior_season_ppa = prior_stats.avg_ppa
            proj.prior_season_used = True

        return proj

    # ========================================================================
    # Quality & Uncertainty Computation
    # ========================================================================

    def _compute_qb_quality(self, team: str, pred_week: int) -> QBQuality:
        """Compute QB quality estimate for a team at prediction week."""
        key = (team, self.year, pred_week)
        if key in self._qualities:
            return self._qualities[key]

        proj = self._identify_projected_starter(team, pred_week)

        quality = QBQuality(
            team=team,
            year=self.year,
            pred_week=pred_week,
            projection=proj,
        )

        # Compute weighted average PPA
        if proj.current_season_dropbacks > 0 or proj.prior_season_used:
            current_weight = proj.current_season_dropbacks
            prior_weight = proj.prior_season_dropbacks * self.prior_decay if proj.prior_season_used else 0
            total_weight = current_weight + prior_weight

            if total_weight > 0:
                weighted_ppa = (
                    proj.current_season_ppa * current_weight +
                    proj.prior_season_ppa * prior_weight
                ) / total_weight
            else:
                weighted_ppa = self.fbs_avg_ppa
        else:
            weighted_ppa = self.fbs_avg_ppa

        # Mean-center (vs FBS average)
        quality.qb_value_raw = weighted_ppa - self.fbs_avg_ppa

        # Apply shrinkage
        n_eff = proj.n_effective_db
        shrink_factor = n_eff / (n_eff + self.shrinkage_k)
        quality.qb_value_shrunk = quality.qb_value_raw * shrink_factor

        # Scale to points and cap
        qb_points_uncapped = quality.qb_value_shrunk * self.qb_scale
        quality.qb_points = max(-self.qb_cap, min(self.qb_cap, qb_points_uncapped))

        # Compute uncertainty
        quality.qb_uncertainty = self._compute_uncertainty(proj)

        # Apply uncertainty dampening
        dampening_factor = 0.5  # How much uncertainty reduces adjustment
        quality.qb_points_effective = quality.qb_points * (1.0 - dampening_factor * quality.qb_uncertainty)

        self._qualities[key] = quality
        return quality

    def _compute_uncertainty(self, proj: QBProjection) -> float:
        """Compute QB uncertainty from projection.

        Uses multiplicative composition:
        - sample_confidence = n_eff / (n_eff + K)
        - switch_factor = 0.5 if starter changed, else 1.0
        - known_factor = 0.3 if unknown starter, else 1.0

        Returns uncertainty in [0, 1] where 1 = very uncertain.
        """
        n_eff = proj.n_effective_db
        sample_confidence = n_eff / (n_eff + self.shrinkage_k)

        switch_factor = 0.5 if proj.starter_changed else 1.0
        known_factor = 0.3 if proj.unknown_starter else 1.0

        qb_confidence = sample_confidence * switch_factor * known_factor
        return 1.0 - qb_confidence

    # ========================================================================
    # Residual Adjustment Logic
    # ========================================================================

    def _compute_baking_factor(self, pred_week: int) -> float:
        """Compute the baking factor for a given prediction week.

        The baking factor represents how much of the QB quality is already
        "baked into" the team's efficiency rating through EFM.

        Returns:
            Factor in [0, 1] where:
            - 0 = QB quality not yet baked in (use full additive adjustment)
            - 1 = QB quality fully baked in (use residual adjustment only)

        Schedule:
            Weeks 1-3: Ramp 0.0 → 1.0 (additive early, fully baked by week 4)
            Week 4+:   1.0 (fully baked, residual only)
        """
        if pred_week <= BAKING_WEEK_START:
            return 0.0
        elif pred_week <= BAKING_WEEK_MID:
            # Ramp from 0.0 to ~0.5 over weeks 1-3
            progress = (pred_week - BAKING_WEEK_START) / (BAKING_WEEK_MID - BAKING_WEEK_START)
            return 0.5 * progress
        elif pred_week <= BAKING_WEEK_FULL:
            # Ramp from 0.5 to 1.0 over weeks 3-4
            progress = (pred_week - BAKING_WEEK_MID) / (BAKING_WEEK_FULL - BAKING_WEEK_MID)
            return 0.5 + 0.5 * progress
        else:
            return 1.0

    def _compute_team_avg_qb_value(self, team: str, pred_week: int) -> float:
        """Compute the dropback-weighted average QB value for a team.

        This represents the "QB quality baked into the team rating" - the weighted
        average of all QBs who have played for the team, weighted by dropbacks.

        Args:
            team: Team name
            pred_week: The prediction week (stats are through pred_week - 1)

        Returns:
            Weighted average qb_points (after shrinkage and scaling) for the team.
            Returns 0.0 if no data available.
        """
        key = (team, self.year, pred_week)
        if key in self._team_avg_qb:
            return self._team_avg_qb[key]

        # Get all QBs who played for this team through pred_week - 1
        # Note: _season_stats is keyed by (team, year, pred_week) but contains data through pred_week - 1
        season_stats = self._season_stats.get((team, self.year, pred_week), {})

        if not season_stats:
            self._team_avg_qb[key] = 0.0
            return 0.0

        # Compute weighted average of QB values
        total_db = 0
        weighted_sum = 0.0

        for player_id, stats in season_stats.items():
            db = stats.total_dropbacks
            if db == 0:
                continue

            # Compute this QB's value (same logic as _compute_qb_quality but per-QB)
            avg_ppa = stats.avg_ppa
            qb_value_raw = avg_ppa - self.fbs_avg_ppa

            # Apply shrinkage based on this QB's sample size
            shrink_factor = db / (db + self.shrinkage_k)
            qb_value_shrunk = qb_value_raw * shrink_factor

            # Scale to points (no cap for team average)
            qb_points = qb_value_shrunk * self.qb_scale

            # Weight by dropbacks
            total_db += db
            weighted_sum += qb_points * db

        if total_db == 0:
            self._team_avg_qb[key] = 0.0
            return 0.0

        team_avg = weighted_sum / total_db
        self._team_avg_qb[key] = team_avg
        return team_avg

    # ========================================================================
    # Manual Override Interface
    # ========================================================================

    def flag_qb_out(self, team: str, adjustment: Optional[float] = None) -> None:
        """Flag a team's QB as out (manual override).

        Args:
            team: Team name
            adjustment: Optional custom adjustment (uses default if None)
        """
        self._qb_out_teams.add(team)
        if adjustment is not None:
            self._qb_out_adjustments[team] = adjustment
        logger.info(f"Manual QB override: {team} flagged as QB out")

    def clear_qb_out(self, team: Optional[str] = None) -> None:
        """Clear QB out flag(s)."""
        if team:
            self._qb_out_teams.discard(team)
            self._qb_out_adjustments.pop(team, None)
        else:
            self._qb_out_teams.clear()
            self._qb_out_adjustments.clear()

    def _get_manual_adjustment(self, team: str) -> Optional[float]:
        """Get manual QB adjustment if team is flagged."""
        if team not in self._qb_out_teams:
            return None

        if team in self._qb_out_adjustments:
            return self._qb_out_adjustments[team]

        # Default: use current starter quality as basis for drop-off estimate
        quality = self._compute_qb_quality(team, self._data_built_through_week + 1)
        # Assume backup is ~0.2 PPA worse
        backup_drop = 0.2 * self.qb_scale
        return -min(self.qb_cap, backup_drop + quality.qb_points_effective)

    # ========================================================================
    # Main Interface
    # ========================================================================

    def get_adjustment(
        self,
        home_team: str,
        away_team: str,
        pred_week: Optional[int] = None,
    ) -> float:
        """Get net QB adjustment for a matchup.

        This is the main interface for SpreadGenerator.

        Args:
            home_team: Home team name
            away_team: Away team name
            pred_week: Prediction week (defaults to data_built_through_week + 1)

        Returns:
            Net point adjustment (positive = favors home)
        """
        if pred_week is None:
            pred_week = self._data_built_through_week + 1

        # Phase 1 only mode: skip QB adjustment for Core weeks (4+)
        if self.phase1_only and pred_week >= 4:
            return 0.0

        # Check for manual overrides
        home_manual = self._get_manual_adjustment(home_team)
        away_manual = self._get_manual_adjustment(away_team)

        # Get continuous quality estimates
        home_quality = self._compute_qb_quality(home_team, pred_week)
        away_quality = self._compute_qb_quality(away_team, pred_week)

        # Compute baking factor and team averages for residual adjustment
        baking_factor = self._compute_baking_factor(pred_week)
        home_team_avg = self._compute_team_avg_qb_value(home_team, pred_week)
        away_team_avg = self._compute_team_avg_qb_value(away_team, pred_week)

        # Compute adjustments using residual logic:
        # residual = raw_qb_points - (team_avg * baking_factor)
        # Early season (baking_factor ≈ 0): keeps additive boost
        # Core season (baking_factor → 1): adjustment → 0 for stable starter (no double-count)
        # Injury case: current << team_avg → negative adjustment (correct penalty)
        #
        # Use qb_points (not qb_points_effective) for both raw and team_avg to compare apples-to-apples.
        # Apply uncertainty dampening to the final residual.
        if home_manual is not None:
            home_adj = home_manual
            logger.debug(
                f"Manual QB override for {home_team}: continuous disabled, "
                f"using manual adjustment of {home_adj:.1f} pts"
            )
        else:
            home_raw = home_quality.qb_points  # Use raw, not effective
            home_baked = home_team_avg * baking_factor
            home_residual = home_raw - home_baked
            # Apply uncertainty dampening to residual
            dampening_factor = 0.5
            home_adj = home_residual * (1.0 - dampening_factor * home_quality.qb_uncertainty)

        if away_manual is not None:
            away_adj = away_manual
            logger.debug(
                f"Manual QB override for {away_team}: continuous disabled, "
                f"using manual adjustment of {away_adj:.1f} pts"
            )
        else:
            away_raw = away_quality.qb_points  # Use raw, not effective
            away_baked = away_team_avg * baking_factor
            away_residual = away_raw - away_baked
            # Apply uncertainty dampening to residual
            dampening_factor = 0.5
            away_adj = away_residual * (1.0 - dampening_factor * away_quality.qb_uncertainty)

        # Net adjustment (positive = favors home)
        spread_qb_adj = home_adj - away_adj

        # Store diagnostics
        diag = QBMatchupDiagnostics(
            home_team=home_team,
            away_team=away_team,
            year=self.year,
            pred_week=pred_week,
            home_quality=home_quality,
            away_quality=away_quality,
            spread_qb_adj=spread_qb_adj,
            qb_uncertainty_game=(home_quality.qb_uncertainty + away_quality.qb_uncertainty) / 2,
            manual_override_home=home_manual is not None,
            manual_override_away=away_manual is not None,
        )
        self._diagnostics.append(diag)

        return spread_qb_adj

    def get_diagnostics(self, home_team: str, away_team: str, pred_week: int) -> Optional[QBMatchupDiagnostics]:
        """Get full diagnostics for a matchup."""
        for diag in reversed(self._diagnostics):
            if (diag.home_team == home_team and
                diag.away_team == away_team and
                diag.pred_week == pred_week):
                return diag
        return None

    # ========================================================================
    # Logging & Reporting
    # ========================================================================

    def log_weekly_summary(self, pred_week: int) -> None:
        """Log weekly QB summary at INFO level."""
        # Collect stats for this week's predictions
        week_diagnostics = [d for d in self._diagnostics if d.pred_week == pred_week]

        if not week_diagnostics:
            logger.info(f"No QB diagnostics for week {pred_week}")
            return

        # Count unknown starters
        unknown_count = sum(
            1 for d in week_diagnostics
            if (d.home_quality and d.home_quality.projection and d.home_quality.projection.unknown_starter) or
               (d.away_quality and d.away_quality.projection and d.away_quality.projection.unknown_starter)
        )

        # Uncertainty distribution
        uncertainties = [d.qb_uncertainty_game for d in week_diagnostics]
        if uncertainties:
            unc_min = min(uncertainties)
            unc_p25 = np.percentile(uncertainties, 25)
            unc_med = np.median(uncertainties)
            unc_p75 = np.percentile(uncertainties, 75)
            unc_max = max(uncertainties)
        else:
            unc_min = unc_p25 = unc_med = unc_p75 = unc_max = 0

        # Top/bottom qb_points_effective
        all_qualities = []
        for d in week_diagnostics:
            if d.home_quality:
                all_qualities.append((d.home_team, d.home_quality.qb_points_effective))
            if d.away_quality:
                all_qualities.append((d.away_team, d.away_quality.qb_points_effective))

        all_qualities.sort(key=lambda x: x[1], reverse=True)
        top_3 = all_qualities[:3]
        bottom_3 = all_qualities[-3:]

        # Count starter changes
        starter_changes = sum(
            1 for d in week_diagnostics
            if (d.home_quality and d.home_quality.projection and d.home_quality.projection.starter_changed) or
               (d.away_quality and d.away_quality.projection and d.away_quality.projection.starter_changed)
        )

        # Count meaningful adjustments
        meaningful_count = sum(
            1 for d in week_diagnostics if abs(d.spread_qb_adj) > 1.0
        )

        logger.info(f"=== QB Weekly Summary: Week {pred_week} ===")
        logger.info(f"Teams with unknown starter: {unknown_count}")
        logger.info(f"Uncertainty distribution: min={unc_min:.2f}, p25={unc_p25:.2f}, "
                   f"median={unc_med:.2f}, p75={unc_p75:.2f}, max={unc_max:.2f}")
        logger.info(f"Top 3 QB ratings: {', '.join(f'{t}: {v:+.2f}' for t, v in top_3)}")
        logger.info(f"Bottom 3 QB ratings: {', '.join(f'{t}: {v:+.2f}' for t, v in bottom_3)}")
        logger.info(f"Starter changes detected: {starter_changes}")
        logger.info(f"Games with |QB adj| > 1.0: {meaningful_count}")

    def get_calibration_data(self) -> pd.DataFrame:
        """Get calibration data for all computed qualities.

        Returns DataFrame with qb_value_shrunk by team-week for analysis.
        """
        data = []
        for (team, year, pred_week), quality in self._qualities.items():
            data.append({
                'team': team,
                'year': year,
                'pred_week': pred_week,
                'qb_value_raw': quality.qb_value_raw,
                'qb_value_shrunk': quality.qb_value_shrunk,
                'qb_points': quality.qb_points,
                'qb_points_effective': quality.qb_points_effective,
                'qb_uncertainty': quality.qb_uncertainty,
                'n_effective_db': quality.projection.n_effective_db if quality.projection else 0,
                'unknown_starter': quality.projection.unknown_starter if quality.projection else True,
                'starter_changed': quality.projection.starter_changed if quality.projection else False,
                'prior_used': quality.projection.prior_season_used if quality.projection else False,
            })
        return pd.DataFrame(data)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._game_stats.clear()
        self._season_stats.clear()
        self._projections.clear()
        self._qualities.clear()
        self._team_avg_qb.clear()
        self._diagnostics.clear()
        self._data_built_through_week = 0
