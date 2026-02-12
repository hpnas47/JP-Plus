"""Efficiency Foundation Model (EFM) - built on Success Rate, not margins.

This model builds power ratings from play-by-play efficiency metrics:
1. Foundation = Efficiency (Success Rate) + Explosiveness (IsoPPP)
2. Opponent adjustment via Ridge regression on Success Rate
3. DO NOT regress on margins - regress on efficiency metrics

Key insight: "Do not regress on the final score. Regress on the Success Rate
per Game so that we are measuring efficiency, not just the scoreboard outcome."
"""

import hashlib
import logging
import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd
from scipy import linalg

from config.settings import get_settings
from config.play_types import (
    TURNOVER_PLAY_TYPES,
    INTERCEPTION_PLAY_TYPES,
    FUMBLE_PLAY_TYPES,
    POINTS_PER_TURNOVER,
    SCRIMMAGE_PLAY_TYPES,
    NON_SCRIMMAGE_PLAY_TYPES,
)
from config.dtypes import optimize_dtypes, FLOAT32_COLUMNS

logger = logging.getLogger(__name__)


# =============================================================================
# OPPONENT-ADJUSTED METRIC CACHE
# =============================================================================
# Module-level cache for ridge regression results to avoid redundant computation.
# Key: (season, eval_week, metric_name, ridge_alpha)
# Value: (off_adjusted, def_adjusted, learned_hfa)
#
# WHY CACHING IS SAFE:
# Ridge regression on (season, week, metric) is deterministic given:
# 1. Same input plays (filtered by season and week)
# 2. Same ridge_alpha hyperparameter
# 3. Same metric column (is_success or ppa)
#
# The cache invalidates naturally via key mismatch when any parameter changes.
# A data_hash is also included in the key to guard against edge cases where
# the same (season, week) combination has different underlying data.
#
# LRU Eviction: Cache is bounded to prevent unbounded growth during sweeps.
# Uses OrderedDict with move-to-end on hit, pop-from-front when full.
# =============================================================================
from collections import OrderedDict

_RIDGE_CACHE_MAX_SIZE = 500  # ~4 years × 15 weeks × 2 metrics × 4 hyperparameter combos
_RIDGE_ADJUST_CACHE: OrderedDict[tuple, tuple[dict, dict, Optional[float]]] = OrderedDict()
_CACHE_STATS = {"hits": 0, "misses": 0, "evictions": 0, "eviction_warned": False}


def clear_ridge_cache() -> dict:
    """Clear the ridge adjustment cache and return stats.

    Returns:
        Dict with cache statistics (hits, misses, evictions) before clearing.
    """
    global _RIDGE_ADJUST_CACHE, _CACHE_STATS
    stats = _CACHE_STATS.copy()
    _RIDGE_ADJUST_CACHE.clear()
    _CACHE_STATS = {"hits": 0, "misses": 0, "evictions": 0, "eviction_warned": False}
    logger.info(f"Ridge cache cleared. Previous stats: {stats}")
    return stats


def get_ridge_cache_stats() -> dict:
    """Get current cache statistics.

    Returns:
        Dict with hits, misses, evictions, size, max_size, and hit_rate.
    """
    total = _CACHE_STATS["hits"] + _CACHE_STATS["misses"]
    hit_rate = _CACHE_STATS["hits"] / total if total > 0 else 0.0
    return {
        "hits": _CACHE_STATS["hits"],
        "misses": _CACHE_STATS["misses"],
        "evictions": _CACHE_STATS["evictions"],
        "size": len(_RIDGE_ADJUST_CACHE),
        "max_size": _RIDGE_CACHE_MAX_SIZE,
        "hit_rate": hit_rate,
    }




def _compute_data_hash(plays_df: pd.DataFrame, metric_col: str) -> str:
    """Compute a robust hash of play data for cache key verification.

    P0.3 Fix: Uses sampled team sequences and multiple metric statistics
    to dramatically reduce collision probability vs the old first/last-only hash.

    Args:
        plays_df: Prepared plays DataFrame
        metric_col: The metric column being adjusted

    Returns:
        Short hash string (first 12 chars of SHA256)
    """
    n_plays = len(plays_df)
    if n_plays == 0:
        return "empty"

    # Sample for large datasets to keep hashing fast
    if n_plays > 10000:
        sample_idx = np.linspace(0, n_plays - 1, 1000, dtype=int)
        sample = plays_df.iloc[sample_idx]
    else:
        sample = plays_df

    # P0.3: Robust hash using team sequences + metric stats + weight stats
    # P3.6: Use numpy tobytes() instead of str.cat() for faster hashing
    # Note: np.asarray() handles ArrowStringArray (pyarrow backend)
    hash_parts = [
        str(n_plays),
        hashlib.md5(np.asarray(sample["offense"]).astype(str).tobytes()).hexdigest()[:8],
        hashlib.md5(np.asarray(sample["defense"]).astype(str).tobytes()).hexdigest()[:8],
        f"{sample[metric_col].sum():.8f}",
        f"{sample[metric_col].mean():.8f}",
    ]
    if len(sample) > 1:
        hash_parts.append(f"{sample[metric_col].std():.8f}")
    if "weight" in sample.columns:
        hash_parts.append(f"{sample['weight'].sum():.8f}")

    hash_data = "|".join(hash_parts)
    return hashlib.sha256(hash_data.encode()).hexdigest()[:12]


def is_garbage_time(quarter: int, score_diff: int) -> bool:
    """Check if play is in garbage time (scalar version for single plays).

    Uses thresholds from Settings (single source of truth).
    A play is garbage time if the score differential exceeds the
    threshold for that quarter.

    Args:
        quarter: Game quarter (1-4)
        score_diff: Absolute score differential

    Returns:
        True if garbage time

    Thresholds (from config/settings.py):
        Q1: 28 pts, Q2: 24 pts, Q3: 21 pts, Q4: 16 pts
    """
    settings = get_settings()
    thresholds = {
        1: settings.garbage_time_q1,
        2: settings.garbage_time_q2,
        3: settings.garbage_time_q3,
        4: settings.garbage_time_q4,
    }
    threshold = thresholds.get(quarter, settings.garbage_time_q4)
    return score_diff > threshold


def is_garbage_time_vectorized(
    periods: np.ndarray,
    score_diffs: np.ndarray,
    year: int | None = None
) -> np.ndarray:
    """Vectorized garbage time detection for entire DataFrame.

    P3.2 optimization: Replaces row-wise apply with vectorized numpy operations.
    P2.2: Uses module-level cached thresholds to avoid repeated Settings lookups.

    Note on year-conditional thresholds (Feb 2026 investigation):
    The 2024 clock rule change makes leads objectively safer (68% vs 63.2% "won by 14+").
    However, lowering Q4 threshold from 16→14 for 2024+ DEGRADED model performance
    (5+ Edge 54.7%→54.6%, 3+ Edge 54.0%→53.7%). Reason: play-level efficiency data
    is still informative even when game outcomes are harder to achieve. The trailing
    team's plays in a 14-15 pt Q4 deficit still reflect their true capability.
    Year parameter preserved for future experimentation but not currently used.

    Args:
        periods: Array of quarter/period values (1-4)
        score_diffs: Array of absolute score differentials
        year: Season year (currently unused, preserved for future experiments)

    Returns:
        Boolean array indicating garbage time plays
    """
    # P0: Read directly from settings each time (cheap dict access)
    # Avoids stale cache issue when sweeping GT thresholds between iterations
    settings = get_settings()
    gt_q1 = settings.garbage_time_q1
    gt_q2 = settings.garbage_time_q2
    gt_q3 = settings.garbage_time_q3
    gt_q4 = settings.garbage_time_q4

    # Year-conditional thresholds tested but REJECTED (see docstring)
    # Keeping year parameter for future experimentation

    # Build threshold array based on period
    # Default to Q4 threshold for periods outside 1-4 (OT, etc.)
    thresholds = np.full(len(periods), gt_q4, dtype=np.float64)
    thresholds[periods == 1] = gt_q1
    thresholds[periods == 2] = gt_q2
    thresholds[periods == 3] = gt_q3
    thresholds[periods == 4] = gt_q4

    return score_diffs > thresholds


# Success rate thresholds
def is_successful_play(down: int, distance: float, yards_gained: float) -> bool:
    """Determine if play was successful (scalar version for single plays).

    Standard success rate definition:
    - 1st down: Gain at least 50% of yards needed
    - 2nd down: Gain at least 70% of yards needed
    - 3rd/4th down: Gain 100% of yards needed (first down or TD)

    Edge case handling:
    - distance <= 0: Treat as goal-to-go or data error. Require positive yards
      to be successful (any gain from the goal line is good).

    Args:
        down: Current down (1-4)
        distance: Yards to go (may be 0 at goal line or due to data issues)
        yards_gained: Yards gained on play

    Returns:
        True if successful
    """
    # Handle distance=0 edge case (goal line or data error)
    # Require positive yards to avoid auto-success on 0-gain plays
    if distance <= 0:
        return yards_gained > 0

    if down == 1:
        return yards_gained >= 0.5 * distance  # 50% of yards needed
    elif down == 2:
        return yards_gained >= 0.7 * distance  # 70% of yards needed
    else:  # 3rd or 4th down
        return yards_gained >= distance  # 100% of yards needed


def is_successful_play_vectorized(
    downs: np.ndarray, distances: np.ndarray, yards_gained: np.ndarray
) -> np.ndarray:
    """Vectorized success rate calculation for entire DataFrame.

    P3.2 optimization: Replaces row-wise apply with vectorized numpy operations.

    Args:
        downs: Array of down values (1-4)
        distances: Array of yards to go
        yards_gained: Array of yards gained

    Returns:
        Boolean array indicating successful plays
    """
    # Handle distance <= 0 edge case: require positive yards
    zero_distance_mask = distances <= 0
    zero_distance_success = yards_gained > 0

    # Calculate required yards based on down
    # 1st down: 50%, 2nd down: 70%, 3rd/4th down: 100%
    required_pct = np.where(downs == 1, 0.5, np.where(downs == 2, 0.7, 1.0))
    required_yards = distances * required_pct

    # Success if yards_gained >= required_yards
    normal_success = yards_gained >= required_yards

    # Combine: use zero_distance logic where distance <= 0, else normal logic
    return np.where(zero_distance_mask, zero_distance_success, normal_success)


@dataclass
class TeamEFMRating:
    """Container for Efficiency Foundation Model team ratings."""
    team: str

    # Raw metrics (before opponent adjustment)
    raw_success_rate: float
    raw_isoppp: float  # EPA on successful plays

    # Opponent-adjusted metrics
    adj_success_rate: float
    adj_isoppp: float

    # Component ratings (in points)
    efficiency_rating: float  # From success rate (combined O+D)
    explosiveness_rating: float  # From IsoPPP (combined O+D)

    # Separate O/D/ST ratings (in points, relative to average)
    offensive_rating: float  # Higher = better offense
    defensive_rating: float  # Higher = better defense (fewer points allowed)
    special_teams_rating: float  # FG efficiency - DIAGNOSTIC ONLY, not in overall (P2.7)
    turnover_rating: float  # DIAGNOSTIC ONLY: ball_security + takeaways (already embedded in O/D, NOT additive to overall)

    # Combined
    overall_rating: float

    # Sample sizes
    off_plays: int
    def_plays: int


class EfficiencyFoundationModel:
    """Efficiency Foundation Model: regress on Success Rate, not margins.

    The key insight: build ratings from play-by-play efficiency,
    then convert to point differential for predictions.

    Rating Components:
    - overall_rating = offensive_rating + defensive_rating
    - offensive_rating = efficiency + explosiveness + ball_security (turnovers)
    - defensive_rating = efficiency + explosiveness + takeaways (turnovers)

    FIELD POSITION INDEPENDENCE (No Double-Counting with ST):
    - EFM measures PLAY EFFICIENCY: Success Rate and IsoPPP
    - Success Rate = % of plays meeting down-specific yardage thresholds
    - IsoPPP = EPA per successful play (explosiveness)
    - Neither metric uses starting field position (yards_to_goal)
    - This is intentional: field position value is captured by SpecialTeamsModel
    - ST uses YARDS_TO_POINTS = 0.04 to convert field position to points
    - EFM and ST are independent metrics combined additively in SpreadGenerator
    - No double-counting because they measure different things:
      * EFM: "How efficiently does this team move the ball per play?"
      * ST: "How much field position advantage does this team create?"

    Special Teams Integration (P2.7):
    - special_teams_rating is stored for DIAGNOSTIC/REPORTING purposes only
    - It is NOT included in overall_rating to avoid double-counting
    - SpreadGenerator applies ST as a separate adjustment layer using
      SpecialTeamsModel.get_matchup_differential()
    - This follows SP+ methodology: ST is a game-level adjustment, not a base rating
    - If you need ST in ratings, use set_special_teams_rating() for reporting,
      but the spread calculation handles ST separately
    """

    # Conversion factors (empirically derived)
    # These convert efficiency metrics to point equivalents
    SUCCESS_RATE_TO_POINTS = 80.0  # 1% SR difference ≈ 0.8 points
    ISOPPP_TO_POINTS = 15.0  # 0.1 IsoPPP difference ≈ 1.5 points

    # League averages (approximate FBS averages)
    LEAGUE_AVG_SUCCESS_RATE = 0.42
    LEAGUE_AVG_ISOPPP = 0.30  # ~0.3 EPA per successful play

    # Minimum plays for reliable rating
    MIN_PLAYS = 100

    # TURNOVER_PLAY_TYPES and POINTS_PER_TURNOVER imported from config.play_types

    def __init__(
        self,
        ridge_alpha: float = 50.0,  # Optimized via sweep (was 100.0)
        efficiency_weight: float = 0.45,  # Equal SR/IsoPPP weighting (Explosiveness Uplift)
        explosiveness_weight: float = 0.45,  # Equal SR/IsoPPP weighting (Explosiveness Uplift)
        turnover_weight: float = 0.10,  # 10% weight for turnovers (like SP+)
        turnover_prior_strength: float = 10.0,  # DEPRECATED: Use k_int/k_fumble instead
        k_int: float = 10.0,  # Bayesian shrinkage for interceptions (skill-based, moderate shrinkage)
        k_fumble: float = 30.0,  # Bayesian shrinkage for fumbles (luck-based, strong shrinkage)
        garbage_time_weight: float = 0.1,  # Weight for garbage time plays (0 to discard)
        rating_std: float = 12.0,  # Target std for ratings (SP+ uses ~12)
        asymmetric_garbage: bool = True,  # Only penalize trailing team in garbage time
        leading_garbage_weight: float = 1.0,  # Weight for leading team's garbage time plays
        time_decay: float = 1.0,  # Per-week decay factor (1.0 = no decay, 0.95 = 5% per week)
        mov_weight: float = 0.0,  # Weight for MOV calibration layer (0.0 to disable) - DEPRECATED
        mov_cap: float = 5.0,  # Cap for MOV adjustment in points (prevents extreme swings) - DEPRECATED
        fraud_tax_enabled: bool = False,  # Enable "Efficiency Fraud" tax (one-way penalty) - DISABLED: degraded 5+ edge by 0.2%
        fraud_tax_penalty: float = 2.0,  # Fixed penalty in rating points for efficiency frauds
        fraud_tax_threshold: float = 2.5,  # Win gap threshold to trigger penalty
        rz_leverage_enabled: bool = True,  # Enable Red Zone leverage weighting
        rz_weight_20: float = 1.5,  # Weight multiplier for plays inside the 20-yard line
        rz_weight_10: float = 2.0,  # Weight multiplier for plays inside the 10-yard line
        empty_yards_weight: float = 0.7,  # Weight for "empty" successful plays between opp 40-20
        money_down_weight: float = 1.0,  # Weight multiplier for 3rd/4th down plays (LASR - dormant, 1.0 = disabled)
        empty_success_weight: float = 1.0,  # Penalty for successful 3rd/4th down plays that don't convert (dormant)
        def_efficiency_weight: float = None,  # Defensive SR weight (None = use efficiency_weight)
        def_explosiveness_weight: float = None,  # Defensive IsoPPP weight (None = use explosiveness_weight)
        def_turnover_weight: float = None,  # Defensive TO weight (None = use turnover_weight)
        ooc_credibility_weight: float = 0.0,  # OOC Credibility Anchor scale (0.0 = disabled) - REJECTED: monotonic 5+ Edge degradation
    ):
        """Initialize Efficiency Foundation Model.

        Args:
            ridge_alpha: Regularization for opponent adjustment
            efficiency_weight: Weight for success rate component (default 0.45)
            explosiveness_weight: Weight for IsoPPP component (default 0.45)
            turnover_weight: Weight for turnover margin component (default 0.10, like SP+)
            turnover_prior_strength: DEPRECATED - Use k_int/k_fumble for separate shrinkage.
            k_int: Bayesian shrinkage strength for interceptions (default 10.0).
                   INTs are more skill-based (QB decisions, defensive scheme) so use moderate shrinkage.
                   shrink = games / (games + k_int). At k=10, 10-game team keeps 50% of raw value.
            k_fumble: Bayesian shrinkage strength for fumbles (default 30.0).
                      Fumbles are more luck-based (ball bounces randomly) so use strong shrinkage.
                      shrink = games / (games + k_fumble). At k=30, 10-game team keeps only 25% of raw value.
            garbage_time_weight: Weight for garbage time plays (0.1 recommended, 0 to discard)
            rating_std: Target standard deviation for ratings. Set to 12.0 for SP+-like scale
                       where Team A - Team B = expected spread. Higher = more spread between teams.
            asymmetric_garbage: If True, trailing team's garbage time plays are down-weighted.
                              Leading team gets partial credit (leading_garbage_weight, default 0.5).
            leading_garbage_weight: Weight for leading team's plays during garbage time (default 1.0).
                                  1.0 = full credit (best ATS), 0.5 = half credit. Tested 0.5/0.7/symmetric - all degraded 5+ edge.
            time_decay: Per-week decay factor for play weights. 1.0 = no decay (all weeks equal).
                       0.95 = 5% decay per week (Week 1 plays get ~0.54 weight by Week 12).
                       Formula: weight *= decay ^ (max_week - play_week)
            mov_weight: DEPRECATED - Weight for MOV calibration layer (default 0.0, disabled).
                       Use fraud_tax_enabled instead for asymmetric penalty.
            mov_cap: DEPRECATED - Cap for MOV adjustment in points (default 5.0).
            fraud_tax_enabled: Enable "Efficiency Fraud" tax (default True).
                              Applies a one-way penalty to teams with high efficiency but poor wins.
            fraud_tax_penalty: Fixed penalty in rating points (default 2.0).
                              Applied when expected_wins - actual_wins > threshold.
            fraud_tax_threshold: Win gap threshold to trigger penalty (default 2.5 wins).
            rz_leverage_enabled: Enable Red Zone leverage weighting (default True).
                                Up-weight plays near the goal line to emphasize finishing efficiency.
            rz_weight_20: Weight multiplier for plays inside the 20-yard line (default 1.5).
            rz_weight_10: Weight multiplier for plays inside the 10-yard line (default 2.0).
            empty_yards_weight: Weight for "empty" successful plays between opponent 40-20 (default 0.7).
                               Devalues successful plays that stall without entering the red zone or scoring.
            money_down_weight: Weight multiplier for 3rd/4th down plays (default 2.0).
                              LASR (Late And Short Runs) weighting - emphasizes conversion situations.
            empty_success_weight: Penalty for successful 3rd/4th down plays that don't convert (default 0.5).
                                 Applies to plays that meet success criteria but fail to gain first down.
                                 Stacks multiplicatively with money_down_weight (net 1.0x for empty successes).
        """
        self.ridge_alpha = ridge_alpha
        self.efficiency_weight = efficiency_weight
        self.rating_std = rating_std
        self.explosiveness_weight = explosiveness_weight
        self.turnover_weight = turnover_weight
        self.turnover_prior_strength = turnover_prior_strength  # DEPRECATED
        self.k_int = k_int
        self.k_fumble = k_fumble
        self.garbage_time_weight = garbage_time_weight
        self.asymmetric_garbage = asymmetric_garbage
        self.leading_garbage_weight = leading_garbage_weight
        self.time_decay = time_decay
        self.mov_weight = mov_weight
        self.mov_cap = mov_cap
        self.fraud_tax_enabled = fraud_tax_enabled
        self.fraud_tax_penalty = fraud_tax_penalty
        self.fraud_tax_threshold = fraud_tax_threshold
        self.rz_leverage_enabled = rz_leverage_enabled
        self.rz_weight_20 = rz_weight_20
        self.rz_weight_10 = rz_weight_10
        self.empty_yards_weight = empty_yards_weight
        self.money_down_weight = money_down_weight
        self.empty_success_weight = empty_success_weight
        self.ooc_credibility_weight = ooc_credibility_weight
        self.def_efficiency_weight = def_efficiency_weight if def_efficiency_weight is not None else self.efficiency_weight
        self.def_explosiveness_weight = def_explosiveness_weight if def_explosiveness_weight is not None else self.explosiveness_weight
        self.def_turnover_weight = def_turnover_weight if def_turnover_weight is not None else self.turnover_weight

        self.team_ratings: dict[str, TeamEFMRating] = {}

        # Store opponent-adjusted values
        self.off_success_rate: dict[str, float] = {}
        self.def_success_rate: dict[str, float] = {}
        self.off_isoppp: dict[str, float] = {}
        self.def_isoppp: dict[str, float] = {}

        # Turnover stats (P2.6: split into O/D components, now with INT/fumble separation)
        self.turnovers_lost: dict[str, float] = {}  # Per-game turnovers lost (ball security)
        self.turnovers_forced: dict[str, float] = {}  # Per-game turnovers forced (takeaways)
        self.turnover_margin: dict[str, float] = {}  # Per-game net margin (for backward compat)
        self.team_games_played: dict[str, int] = {}  # Games played per team (for TO shrinkage)
        # INT/Fumble split for separate shrinkage (INTs are skill, fumbles are luck)
        self.ints_thrown: dict[str, float] = {}  # Per-game INTs thrown (offense lost)
        self.ints_forced: dict[str, float] = {}  # Per-game INTs forced (defense gained)
        self.fumbles_lost: dict[str, float] = {}  # Per-game fumbles lost (offense lost)
        self.fumbles_recovered: dict[str, float] = {}  # Per-game fumbles recovered (defense gained)

        # Learned implicit HFA from ridge regression (for validation/logging)
        self.learned_hfa_sr: Optional[float] = None  # Implicit HFA in success rate
        self.learned_hfa_isoppp: Optional[float] = None  # Implicit HFA in IsoPPP

        # P3.6: Canonical team index - computed once, reused across pipeline
        # Eliminates redundant sorted(set(...) | set(...)) calls
        self._canonical_teams: Optional[list[str]] = None
        self._team_to_idx: Optional[dict[str, int]] = None

    def _validate_and_normalize_plays(self, plays_df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns and normalize optional columns (P2.9).

        Required columns (fail loudly if missing):
        - offense, defense: Team identifiers
        - down, distance, yards_gained: For success rate calculation
        - offense_score, defense_score: For garbage time detection

        Optional columns (safe fallback with warning):
        - play_type: For scrimmage filtering (if missing, all plays used)
        - period/quarter: For garbage time (normalized to 'period', default 1 with warning)
        - ppa: For IsoPPP (NaN values dropped with warning)
        - week: For time decay (only needed if time_decay < 1.0)
        - game_id: For games played count
        - home_team: For neutral-field ridge regression

        Args:
            plays_df: Raw play-by-play DataFrame

        Returns:
            Normalized DataFrame with validated columns

        Raises:
            ValueError: If required columns are missing
        """
        df = plays_df.copy()

        # Define required columns
        required_cols = ["offense", "defense", "down", "distance", "yards_gained",
                        "offense_score", "defense_score"]
        missing_required = [col for col in required_cols if col not in df.columns]

        if missing_required:
            raise ValueError(
                f"EFM requires columns {missing_required} but they are missing. "
                f"Available columns: {list(df.columns)}"
            )

        # Check for unexpected NaN in required columns
        for col in required_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(
                    f"Column '{col}' has {nan_count} NaN values ({nan_count/len(df)*100:.1f}%); "
                    "these plays will be excluded from analysis"
                )

        # Normalize period/quarter column (P2.9: consistent naming)
        if "period" not in df.columns:
            if "quarter" in df.columns:
                df["period"] = df["quarter"]
                logger.debug("Normalized 'quarter' column to 'period'")
            else:
                # No period info - this will effectively disable garbage time detection
                logger.warning(
                    "No 'period' or 'quarter' column found; defaulting to period=1. "
                    "This will DISABLE garbage time detection for all plays. "
                    "Provide period/quarter data for accurate garbage time weighting."
                )
                df["period"] = 1

        # Validate period values (should be 1-4 for regulation, 5+ for OT)
        invalid_periods = ~df["period"].isin([1, 2, 3, 4, 5, 6, 7, 8])
        invalid_count = invalid_periods.sum()
        if invalid_count > 0:
            logger.warning(
                f"{invalid_count} plays have invalid period values; will be treated as period=1"
            )
            df.loc[invalid_periods, "period"] = 1

        # Handle NaN in PPA column (P2.9: prevent NaN propagation)
        if "ppa" in df.columns:
            ppa_nan_count = df["ppa"].isna().sum()
            if ppa_nan_count > 0:
                ppa_nan_pct = ppa_nan_count / len(df) * 100
                logger.warning(
                    f"Column 'ppa' has {ppa_nan_count} NaN values ({ppa_nan_pct:.1f}%); "
                    "these will be excluded from IsoPPP calculations"
                )
                # Don't drop rows here - just warn. NaN PPA will be handled in weighted calcs.

        # Log optional column status
        optional_cols = {
            "play_type": "scrimmage filtering",
            "week": "time decay weighting",
            "game_id": "games played counting",
            "home_team": "neutral-field ridge regression",
        }
        missing_optional = []
        for col, purpose in optional_cols.items():
            if col not in df.columns:
                missing_optional.append(f"{col} ({purpose})")

        if missing_optional:
            logger.info(f"Optional columns not present: {', '.join(missing_optional)}")

        # Special warning for time_decay without week column
        if self.time_decay < 1.0 and "week" not in df.columns:
            logger.warning(
                f"time_decay={self.time_decay} but no 'week' column; "
                "time decay will not be applied"
            )

        # P3.9: Debug level for per-week logging
        logger.debug(
            f"Validated {len(df)} plays: "
            f"required columns OK, period column {'present' if 'period' in plays_df.columns or 'quarter' in plays_df.columns else 'MISSING (defaulted)'}"
        )

        return df

    def _prepare_plays(
        self, plays_df: pd.DataFrame, max_week: int | None = None,
        team_conferences: Optional[dict[str, str]] = None,
        season: int | None = None
    ) -> pd.DataFrame:
        """Filter and prepare plays for analysis.

        Filters:
        1. Non-scrimmage plays (special teams, penalties, period markers)
        2. Plays with missing/invalid data

        Args:
            plays_df: Raw play-by-play DataFrame
            max_week: Maximum week allowed in training data (for data leakage prevention).
                      If provided, asserts no plays exceed this week and uses it for time_decay.
            team_conferences: Optional dict mapping team name to conference name.
                            If provided, applies 1.5x weight to non-conference games.
            season: Season year for year-conditional garbage time thresholds.
                    2024+ uses Q4=14 (clock rule change), earlier uses Q4=16.

        Returns:
            Filtered DataFrame with success and garbage time flags
        """
        # P2.9: Validate and normalize input data first
        df = self._validate_and_normalize_plays(plays_df)
        initial_count = len(df)

        # DATA LEAKAGE GUARD: Verify no future weeks in training data
        # BUG FIX: Use explicit if/raise instead of assert (asserts can be disabled with -O)
        if max_week is not None and "week" in df.columns:
            actual_max = df["week"].max()
            if actual_max > max_week:
                raise ValueError(
                    f"DATA LEAKAGE in EFM: plays include week {actual_max} "
                    f"but max_week={max_week}. Filter plays before calling calculate_ratings()."
                )

        # P3.5: Build combined filter mask to avoid multiple intermediate DataFrames
        # This reduces peak memory by filtering once instead of chaining df = df[mask]
        keep_mask = np.ones(len(df), dtype=bool)

        # Filter to scrimmage plays only (P2.8 fix)
        # This excludes special teams, penalties, period markers, etc.
        if "play_type" in df.columns:
            scrimmage_mask = df["play_type"].isin(SCRIMMAGE_PLAY_TYPES).values
            non_scrimmage_count = (~scrimmage_mask).sum()
            if non_scrimmage_count > 0:
                # Log what's being filtered (before applying mask)
                filtered_types = df.loc[~scrimmage_mask, "play_type"].value_counts()
                logger.debug(
                    f"Filtering {non_scrimmage_count} non-scrimmage plays: "
                    f"{dict(filtered_types.head(5))}"
                )
            keep_mask &= scrimmage_mask

        # Filter plays with invalid distance (P2.8 fix)
        if "distance" in df.columns:
            valid_distance = df["distance"].notna().values & (df["distance"].values >= 0)
            invalid_count = (~valid_distance).sum()
            if invalid_count > 0:
                logger.debug(f"Filtering {invalid_count} plays with invalid distance")
            keep_mask &= valid_distance

        # P2.9: Filter plays with NaN in required columns for success calculation
        required_for_success = ["down", "distance", "yards_gained", "offense_score", "defense_score"]
        nan_mask = df[required_for_success].isna().any(axis=1).values
        nan_count = nan_mask.sum()
        if nan_count > 0:
            logger.debug(f"Filtering {nan_count} plays with NaN in required columns")
        keep_mask &= ~nan_mask

        # Apply combined filter once (avoids multiple intermediate DataFrames)
        # P2.1: Explicit .copy() to prevent SettingWithCopy warnings on subsequent column assignments
        df = df[keep_mask].copy()

        # P3.9: Debug level for per-week logging (runs many times during backtest)
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.debug(
                f"Prepared {len(df)} plays for EFM "
                f"(filtered {filtered_count} non-scrimmage/invalid/NaN plays)"
            )
        else:
            logger.debug(f"Prepared {len(df)} plays for EFM")

        # P3.2: Vectorized success calculation (replaces row-wise apply)
        prep_start = time.time()

        df["is_success"] = is_successful_play_vectorized(
            df["down"].values,
            df["distance"].values,
            df["yards_gained"].values,
        )

        # P3.2: Vectorized garbage time detection (replaces row-wise apply)
        # Year-conditional threshold: Q4=14 for 2024+ (clock rule change), Q4=16 for earlier
        df["score_diff"] = (df["offense_score"] - df["defense_score"]).abs()
        df["is_garbage_time"] = is_garbage_time_vectorized(
            df["period"].values,
            df["score_diff"].values,
            year=season,
        )

        # P3.2: Vectorized weight calculation (replaces row-wise apply)
        if self.garbage_time_weight == 0:
            # Discard garbage time plays entirely
            # BUG FIX: Use .copy() to avoid SettingWithCopyWarning and ensure assignment sticks
            df = df[~df["is_garbage_time"]].copy()
            df["weight"] = 1.0
        elif self.asymmetric_garbage:
            # Three-tier garbage time: leading team gets partial credit,
            # trailing team heavily down-weighted, non-GT plays full weight.
            # Prevents stat-padding inflation while still crediting dominance.
            offense_margin = df["offense_score"].values - df["defense_score"].values
            is_gt = df["is_garbage_time"].values
            is_trailing = offense_margin <= 0
            is_leading = offense_margin > 0

            df["weight"] = np.where(
                is_gt & is_trailing,
                self.garbage_time_weight,           # 0.1 for trailing team
                np.where(
                    is_gt & is_leading,
                    self.leading_garbage_weight,     # 0.5 for leading team
                    1.0                              # full weight outside GT
                )
            )
        else:
            # Symmetric: weight ALL garbage time plays at reduced value
            df["weight"] = np.where(
                df["is_garbage_time"].values,
                self.garbage_time_weight,
                1.0
            )

        # Apply non-conference game weighting (1.5x boost for OOC matchups)
        # This reduces conference circularity in ridge regression by anchoring
        # conference ratings to external benchmarks. Only boosts FBS-vs-FBS OOC games.
        if team_conferences is not None and "offense" in df.columns and "defense" in df.columns:
            # Map teams to conferences (vectorized lookup)
            off_conf = df["offense"].map(team_conferences)
            def_conf = df["defense"].map(team_conferences)

            # Non-conference game: both teams are FBS (in conference dict) and different conferences
            # .notna() ensures both teams are FBS (missing conf = FCS team, don't boost)
            is_ooc_fbs = off_conf.notna() & def_conf.notna() & (off_conf != def_conf)
            ooc_count = is_ooc_fbs.sum()

            if ooc_count > 0:
                df["weight"] = np.where(is_ooc_fbs, df["weight"] * 1.5, df["weight"])
                logger.debug(f"  Applied 1.5x weight to {ooc_count:,} non-conference FBS plays")

            # OOC Credibility Anchor: weight intra-conference plays by opponent's OOC exposure
            # Teams with more OOC games are better-calibrated by external data, so plays AGAINST
            # them carry more information for Ridge regression. This improves relative ordering
            # within conferences (spread) rather than conference-level shifts.
            # Z_opponent = ooc_games / league_avg_ooc_games, clamped to [0.5, 1.5]
            # W_play *= 1.0 + ooc_credibility_weight * (Z_opponent - 1.0)
            if self.ooc_credibility_weight > 0:
                # Count OOC FBS games per team (unique game_ids where team played OOC)
                if "game_id" in df.columns:
                    # Build per-team OOC game count from plays
                    ooc_plays = df[is_ooc_fbs]
                    off_ooc_games = ooc_plays.groupby("offense")["game_id"].nunique()
                    def_ooc_games = ooc_plays.groupby("defense")["game_id"].nunique()
                    # Combine: total unique OOC games per team
                    all_teams = set(off_ooc_games.index) | set(def_ooc_games.index)
                    team_ooc_counts = {}
                    for team in all_teams:
                        # A team's OOC games appear in both offense and defense rows
                        # Use max of the two since same game appears in both
                        off_n = off_ooc_games.get(team, 0)
                        def_n = def_ooc_games.get(team, 0)
                        team_ooc_counts[team] = max(off_n, def_n)

                    if team_ooc_counts:
                        avg_ooc = np.mean(list(team_ooc_counts.values()))
                        if avg_ooc > 0:
                            # Compute Z for each team: ratio to league average, clamped
                            team_z = {
                                team: np.clip(count / avg_ooc, 0.5, 1.5)
                                for team, count in team_ooc_counts.items()
                            }

                            # For intra-conference plays, multiply weight by credibility of OPPONENT
                            # (defense column = opponent when offense has the ball)
                            is_intra_conf = off_conf.notna() & def_conf.notna() & (off_conf == def_conf)
                            intra_count = is_intra_conf.sum()

                            if intra_count > 0:
                                # Map opponent (defense) Z values for each play
                                opp_z = df["defense"].map(team_z).fillna(1.0)
                                # Credibility multiplier: 1.0 + scale * (Z - 1.0)
                                cred_mult = 1.0 + self.ooc_credibility_weight * (opp_z - 1.0)
                                df["weight"] = np.where(
                                    is_intra_conf,
                                    df["weight"] * cred_mult,
                                    df["weight"]
                                )
                                # Log stats
                                z_values = [team_z.get(t, 1.0) for t in team_ooc_counts]
                                logger.debug(
                                    f"  OOC Credibility Anchor: {intra_count:,} intra-conf plays weighted, "
                                    f"avg Z={np.mean(z_values):.3f}, "
                                    f"range [{min(z_values):.2f}, {max(z_values):.2f}], "
                                    f"avg OOC games={avg_ooc:.1f}"
                                )

        # Apply Red Zone Leverage weighting (up-weight plays near the goal line)
        # Hypothesis: Finishing efficiency in the red zone is more revealing than field-position efficiency.
        # A team that succeeds between the 20s but fails inside the 20 is a "paper tiger".
        if self.rz_leverage_enabled and "yards_to_goal" in df.columns:
            # Inside the 10-yard line: 2.0x weight (critical scoring zone)
            inside_10 = df["yards_to_goal"] <= 10
            # Inside the Red Zone (20-yard line): 1.5x weight
            inside_20 = (df["yards_to_goal"] <= 20) & ~inside_10

            inside_10_count = inside_10.sum()
            inside_20_count = inside_20.sum()

            df["weight"] = np.where(inside_10, df["weight"] * self.rz_weight_10, df["weight"])
            df["weight"] = np.where(inside_20, df["weight"] * self.rz_weight_20, df["weight"])

            if inside_10_count > 0 or inside_20_count > 0:
                total_plays = len(df)
                logger.debug(
                    f"  Red Zone Leverage: {inside_10_count:,} plays inside 10 ({inside_10_count/total_plays*100:.1f}%, "
                    f"{self.rz_weight_10}x weight), "
                    f"{inside_20_count:,} plays 10-20 yds ({inside_20_count/total_plays*100:.1f}%, {self.rz_weight_20}x weight)"
                )

        # Apply Empty Yards Filter (devalue successful plays that stall between opponent 40-20)
        # Hypothesis: "Paper tigers" rack up yards between the 20s but can't finish drives.
        # A successful play at the 30 that gains 4 yards (still at 26) = empty success.
        if self.rz_leverage_enabled and "yards_to_goal" in df.columns and "is_success" in df.columns:
            # SAFETY CHECK 1: Exclude plays with nonsensical field position data
            # If yards_to_goal + yards_gained > 100, the field position is flipped or corrupted
            # P3.5: Use numpy arrays instead of pd.Series for boolean masks
            valid_field_position = np.ones(len(df), dtype=bool)
            if "yards_gained" in df.columns:
                valid_field_position = (df["yards_to_goal"].values + df["yards_gained"].values) <= 100
                invalid_count = (~valid_field_position).sum()
                if invalid_count > 0:
                    logger.debug(
                        f"  Empty Yards Filter: {invalid_count:,} plays excluded due to invalid field position "
                        f"(yards_to_goal + yards_gained > 100)"
                    )

            # Plays in the "empty yards" zone: opponent 40 to 20
            in_empty_zone = (df["yards_to_goal"] >= 20) & (df["yards_to_goal"] <= 40) & valid_field_position

            # Successful plays in this zone that DON'T enter the red zone or score
            # A play enters the RZ if: yards_to_goal - yards_gained < 20
            # P3.5: Use numpy arrays instead of pd.Series for boolean masks
            if "yards_gained" in df.columns:
                enters_rz = (df["yards_to_goal"].values - df["yards_gained"].values) < 20
            else:
                enters_rz = np.zeros(len(df), dtype=bool)

            # Check for scoring plays
            # P3.5: Use numpy arrays instead of pd.Series for boolean masks
            if "play_type" in df.columns:
                scoring_types = ["Passing Touchdown", "Rushing Touchdown", "Field Goal Good"]
                is_scoring = df["play_type"].isin(scoring_types).values
            else:
                is_scoring = np.zeros(len(df), dtype=bool)

            # Apply 0.7x weight to successful plays in empty zone that don't enter RZ or score
            # P3.5: Convert remaining Series to numpy arrays for consistent boolean operations
            is_empty_success = (
                in_empty_zone.values &
                (df["is_success"].values == 1) &
                ~enters_rz &
                ~is_scoring
            )

            empty_count = is_empty_success.sum()
            if empty_count > 0:
                df["weight"] = np.where(is_empty_success, df["weight"] * self.empty_yards_weight, df["weight"])
                total_plays = len(df)
                logger.debug(
                    f"  Empty Yards Filter: {empty_count:,} plays down-weighted ({empty_count/total_plays*100:.1f}%, "
                    f"{self.empty_yards_weight}x weight)"
                )

            # SAFETY CHECK 2 & 3: Per-team empty yards coverage with warning threshold
            # Uses single groupby instead of O(teams × plays) per-team loop (7.6x faster)
            if empty_count > 0 and "offense" in df.columns:
                # Build stats in one pass via groupby
                successful_mask = df["is_success"] == 1
                if successful_mask.any():
                    # P3.5: is_empty_success is now numpy array, index directly
                    stats_df = pd.DataFrame({
                        "offense": df.loc[successful_mask, "offense"],
                        "is_empty": is_empty_success[successful_mask.values],
                    })
                    team_stats = stats_df.groupby("offense").agg(
                        success_count=("offense", "count"),
                        empty_count=("is_empty", "sum"),
                    )
                    team_stats["empty_pct"] = (team_stats["empty_count"] / team_stats["success_count"]) * 100

                    # Per-team debug logs only if DEBUG enabled (avoids iteration overhead otherwise)
                    if logger.isEnabledFor(logging.DEBUG):
                        for team, row in team_stats.iterrows():
                            logger.debug(
                                f"  Empty Yards Filter applied to {row['empty_pct']:.1f}% of successful plays for {team}"
                            )

                    # Always check for anomalous teams (>15% threshold)
                    high_empty = team_stats[team_stats["empty_pct"] > 15.0]
                    for team, row in high_empty.iterrows():
                        logger.warning(
                            f"{team} has {row['empty_pct']:.1f}% successful plays in Empty Yards zone — "
                            f"possible field position data error"
                        )

        # Apply Money Down weighting (LASR - Late And Short Runs)
        # Hypothesis: 3rd/4th down conversion ability is a persistent trait that reveals
        # coaching quality, scheme adaptability, and execution under pressure.
        if self.money_down_weight != 1.0 and "down" in df.columns:
            is_money_down = df["down"].isin([3, 4])
            money_down_count = is_money_down.sum()

            if money_down_count > 0:
                # Apply base money down weight
                df["weight"] = np.where(is_money_down, df["weight"] * self.money_down_weight, df["weight"])

                # Apply Empty Success penalty: successful 3rd/4th down plays that don't convert
                # A play is an "empty success" if it meets success rate criteria (50%/70%/100% of distance)
                # but fails to gain enough yards for a first down (yards_gained < distance)
                if self.empty_success_weight != 1.0 and "is_success" in df.columns and "distance" in df.columns and "yards_gained" in df.columns:
                    # Empty success: successful play on money down that didn't convert
                    # yards_gained < distance means no first down (didn't gain full distance)
                    is_empty_success = (
                        is_money_down &
                        (df["is_success"] == 1) &
                        (df["yards_gained"] < df["distance"])
                    )

                    empty_success_count = is_empty_success.sum()

                    if empty_success_count > 0:
                        # Apply penalty (stacks with money_down_weight: 2.0 × 0.5 = 1.0 net)
                        df["weight"] = np.where(is_empty_success, df["weight"] * self.empty_success_weight, df["weight"])

                        total_plays = len(df)
                        logger.debug(
                            f"  LASR: {money_down_count:,} money down plays ({money_down_count/total_plays*100:.1f}%, "
                            f"{self.money_down_weight}x weight), "
                            f"{empty_success_count:,} empty successes ({empty_success_count/total_plays*100:.1f}%, "
                            f"{self.empty_success_weight}x penalty)"
                        )
                else:
                    total_plays = len(df)
                    logger.debug(
                        f"  LASR: {money_down_count:,} money down plays ({money_down_count/total_plays*100:.1f}%, "
                        f"{self.money_down_weight}x weight)"
                    )

        # Store base weight (before time decay) for X_base caching.
        # base_weight = GT × OOC × RZ × empty_yards (all non-temporal weights).
        # _ridge_adjust_metric() uses base_weight and applies time decay dynamically
        # per eval_week, enabling X_base reuse when only the reference week changes.
        df["base_weight"] = df["weight"].values.copy()

        # Apply time decay if enabled (decay < 1.0).
        # Moved to end of weight pipeline so base_weight captures all non-temporal weights.
        # weight = base_weight × time_decay^(ref_week - play_week)
        if self.time_decay < 1.0 and "week" in df.columns:
            decay_reference_week = max_week if max_week is not None else df["week"].max()
            df["time_weight"] = self.time_decay ** (decay_reference_week - df["week"])
            df["weight"] = df["base_weight"] * df["time_weight"]

        prep_time = time.time() - prep_start
        logger.debug(f"  Vectorized preprocessing: {prep_time*1000:.1f}ms for {len(df):,} plays")

        # P2.1: Verify expected columns exist after preprocessing
        expected_cols = {"is_success", "weight", "base_weight", "offense", "defense", "ppa"}
        missing_cols = expected_cols - set(df.columns)
        assert not missing_cols, f"EFM preprocessing missing columns: {missing_cols}"

        return df

    def _calculate_raw_metrics(
        self, plays_df: pd.DataFrame
    ) -> tuple[dict, dict, dict, dict]:
        """Calculate raw (unadjusted) success rate and IsoPPP for all teams.

        P3.2 optimization: Uses groupby aggregation instead of per-team loops.
        This reduces complexity from O(T×N) to O(N) where T=teams, N=plays.

        Both SR and IsoPPP use the same play weighting scheme (garbage time + time decay).
        This ensures consistency: if a play is down-weighted for SR, it's also
        down-weighted for IsoPPP.

        Args:
            plays_df: Prepared plays DataFrame (must have 'weight' column)

        Returns:
            Tuple of (off_sr, def_sr, off_isoppp, def_isoppp, off_isoppp_real, def_isoppp_real)
            where *_real are sets of teams with play-derived (non-sentinel) IsoPPP values
        """
        agg_start = time.time()

        # BUG FIX: Compute weighted success as local array instead of mutating input DataFrame
        # This prevents cross-run contamination if caller reuses/caches the DataFrame
        weighted_success = plays_df["is_success"].astype(float).values * plays_df["weight"].values

        # ===== OFFENSIVE METRICS (groupby offense) =====
        # Use assign() to create temporary column for aggregation without mutating input
        off_grouped = plays_df.assign(weighted_success=weighted_success).groupby("offense").agg(
            play_count=("weight", "size"),
            weight_sum=("weight", "sum"),
            weighted_success_sum=("weighted_success", "sum"),
        )

        # Vectorized SR: compute ratio, mask below threshold, convert to dict in one pass
        sufficient_off = off_grouped["play_count"] >= self.MIN_PLAYS
        off_grouped["sr"] = np.where(
            sufficient_off,
            off_grouped["weighted_success_sum"] / off_grouped["weight_sum"],
            self.LEAGUE_AVG_SUCCESS_RATE,
        )
        off_sr = off_grouped["sr"].to_dict()

        # ===== DEFENSIVE METRICS (groupby defense) =====
        def_grouped = plays_df.assign(weighted_success=weighted_success).groupby("defense").agg(
            play_count=("weight", "size"),
            weight_sum=("weight", "sum"),
            weighted_success_sum=("weighted_success", "sum"),
        )

        sufficient_def = def_grouped["play_count"] >= self.MIN_PLAYS
        def_grouped["sr"] = np.where(
            sufficient_def,
            def_grouped["weighted_success_sum"] / def_grouped["weight_sum"],
            self.LEAGUE_AVG_SUCCESS_RATE,
        )
        def_sr = def_grouped["sr"].to_dict()

        # ===== ISOPPP (successful plays only, with valid PPA) =====
        off_isoppp = {}
        def_isoppp = {}
        # P1.2: Track which teams have play-derived IsoPPP (not sentinel defaults)
        off_isoppp_real = set()
        def_isoppp_real = set()

        if "ppa" in plays_df.columns:
            # Filter to successful plays with valid PPA
            successful_plays = plays_df[plays_df["is_success"] & plays_df["ppa"].notna()].copy()

            if len(successful_plays) > 0:
                successful_plays["weighted_ppa"] = successful_plays["ppa"] * successful_plays["weight"]

                # Offensive IsoPPP - vectorized
                off_isoppp_grouped = successful_plays.groupby("offense").agg(
                    succ_count=("weight", "size"),
                    weight_sum=("weight", "sum"),
                    weighted_ppa_sum=("weighted_ppa", "sum"),
                )
                # Join play_count from off_grouped for threshold check
                off_isoppp_grouped = off_isoppp_grouped.join(off_grouped["play_count"], how="left")
                off_isoppp_grouped["play_count"] = off_isoppp_grouped["play_count"].fillna(0)

                # Compound mask: succ_count >= 10 AND play_count >= MIN_PLAYS
                off_valid = (off_isoppp_grouped["succ_count"] >= 10) & (off_isoppp_grouped["play_count"] >= self.MIN_PLAYS)
                off_isoppp_grouped["isoppp"] = np.where(
                    off_valid,
                    off_isoppp_grouped["weighted_ppa_sum"] / off_isoppp_grouped["weight_sum"],
                    self.LEAGUE_AVG_ISOPPP,
                )
                off_isoppp = off_isoppp_grouped["isoppp"].to_dict()
                off_isoppp_real = set(off_isoppp_grouped.index[off_valid])

                # Defensive IsoPPP - vectorized
                def_isoppp_grouped = successful_plays.groupby("defense").agg(
                    succ_count=("weight", "size"),
                    weight_sum=("weight", "sum"),
                    weighted_ppa_sum=("weighted_ppa", "sum"),
                )
                # Join play_count from def_grouped for threshold check
                def_isoppp_grouped = def_isoppp_grouped.join(def_grouped["play_count"], how="left")
                def_isoppp_grouped["play_count"] = def_isoppp_grouped["play_count"].fillna(0)

                def_valid = (def_isoppp_grouped["succ_count"] >= 10) & (def_isoppp_grouped["play_count"] >= self.MIN_PLAYS)
                def_isoppp_grouped["isoppp"] = np.where(
                    def_valid,
                    def_isoppp_grouped["weighted_ppa_sum"] / def_isoppp_grouped["weight_sum"],
                    self.LEAGUE_AVG_ISOPPP,
                )
                def_isoppp = def_isoppp_grouped["isoppp"].to_dict()
                def_isoppp_real = set(def_isoppp_grouped.index[def_valid])

        # Fill in missing teams with league average
        # P3.6: Use canonical team index if available, otherwise compute
        all_teams = self._canonical_teams if self._canonical_teams else sorted(
            set(off_grouped.index) | set(def_grouped.index)
        )
        for team in all_teams:
            if team not in off_sr:
                off_sr[team] = self.LEAGUE_AVG_SUCCESS_RATE
            if team not in def_sr:
                def_sr[team] = self.LEAGUE_AVG_SUCCESS_RATE
            if team not in off_isoppp:
                off_isoppp[team] = self.LEAGUE_AVG_ISOPPP
            if team not in def_isoppp:
                def_isoppp[team] = self.LEAGUE_AVG_ISOPPP

        agg_time = time.time() - agg_start
        logger.debug(
            f"  Groupby aggregation: {agg_time*1000:.1f}ms for {len(all_teams)} teams, "
            f"{len(plays_df):,} plays"
        )

        return off_sr, def_sr, off_isoppp, def_isoppp, off_isoppp_real, def_isoppp_real

    def _ridge_solve_cholesky(
        self,
        off_idx: np.ndarray,
        def_idx: np.ndarray,
        home_signs: Optional[np.ndarray],
        y: np.ndarray,
        weights: np.ndarray,
        n_teams: int,
        alpha: float,
    ) -> tuple[np.ndarray, float]:
        """Direct Cholesky solve for opponent-adjusted Ridge regression.

        Exploits the known structure of the design matrix (team indicators + HFA)
        to compute X^T W X and X^T W y analytically via np.bincount, bypassing
        sparse matrix construction entirely.

        This is both faster and more accurate than sklearn's iterative sparse_cg solver:
        - Complexity: O(n_plays) for Gram matrix + O(p^3/3) for Cholesky where p ~ 261
        - Accuracy: 1e-14 vs sparse_cg's 1e-6 (direct method vs iterative)

        Args:
            off_idx: Team indices for offensive team (0 to n_teams-1)
            def_idx: Team indices for defensive team (0 to n_teams-1)
            home_signs: +1 if offense is home, -1 if defense is home, 0 if neutral (or None)
            y: Target metric values (e.g., success = 1/0 or PPA)
            weights: Sample weights (incorporating time decay, garbage time, etc.)
            n_teams: Number of teams
            alpha: Ridge regularization strength

        Returns:
            (beta, intercept) where beta = [off_coefs..., def_coefs..., (hfa)]
        """
        n_plays = len(y)
        has_hfa = home_signs is not None
        n_cols = 2 * n_teams + (1 if has_hfa else 0)

        # sklearn normalizes weights so sum(w) = n_samples
        # We replicate this for exact equivalence
        w_sum = weights.sum()
        w_norm = weights * (n_plays / w_sum)

        # --- Weighted column means (for centering) ---
        # X_offset[i] = weighted_mean(X[:, i]) for each column
        off_w = np.bincount(off_idx, weights=w_norm, minlength=n_teams)
        def_w = np.bincount(def_idx, weights=w_norm, minlength=n_teams)
        X_offset = np.empty(n_cols)
        X_offset[:n_teams] = off_w / n_plays
        X_offset[n_teams : 2 * n_teams] = -def_w / n_plays
        if has_hfa:
            X_offset[2 * n_teams] = np.dot(w_norm, home_signs) / n_plays

        # --- Weighted mean of y ---
        y_offset = np.dot(w_norm, y) / n_plays
        y_c = y - y_offset
        wy = w_norm * y_c

        # --- Gram matrix G = X^T W X (analytical via bincount) ---
        G = np.zeros((n_cols, n_cols))

        # Diagonal: offense block (sum of weights per offense team)
        G[np.arange(n_teams), np.arange(n_teams)] = off_w

        # Diagonal: defense block (sum of weights per defense team)
        G[np.arange(n_teams, 2 * n_teams), np.arange(n_teams, 2 * n_teams)] = def_w

        # Off-diagonal: matchup cross-block (off vs def interactions)
        # G[i, n+j] = -sum(w) for plays where off=i AND def=j
        matchup_idx = off_idx * n_teams + def_idx  # unique matchup encoding
        matchup_w = np.bincount(
            matchup_idx, weights=w_norm, minlength=n_teams * n_teams
        ).reshape(n_teams, n_teams)
        G[:n_teams, n_teams : 2 * n_teams] = -matchup_w
        G[n_teams : 2 * n_teams, :n_teams] = -matchup_w.T

        if has_hfa:
            hc = 2 * n_teams
            wh = w_norm * home_signs
            # HFA interactions with offense teams
            off_hfa = np.bincount(off_idx, weights=wh, minlength=n_teams)
            # HFA interactions with defense teams (negative because def has -1 in X)
            def_hfa = np.bincount(def_idx, weights=-wh, minlength=n_teams)
            G[:n_teams, hc] = off_hfa
            G[hc, :n_teams] = off_hfa
            G[n_teams : 2 * n_teams, hc] = def_hfa
            G[hc, n_teams : 2 * n_teams] = def_hfa
            G[hc, hc] = np.dot(w_norm, home_signs**2)

        # --- Centering correction ---
        # After centering: G_centered = G - n * X_offset @ X_offset^T
        G -= n_plays * np.outer(X_offset, X_offset)

        # --- Regularization (DO NOT regularize intercept) ---
        # Add alpha * I to the centered Gram matrix
        np.fill_diagonal(G, G.diagonal() + alpha)

        # --- Right-hand side: X^T W y_c ---
        Xty = np.empty(n_cols)
        Xty[:n_teams] = np.bincount(off_idx, weights=wy, minlength=n_teams)
        Xty[n_teams : 2 * n_teams] = -np.bincount(def_idx, weights=wy, minlength=n_teams)
        if has_hfa:
            Xty[2 * n_teams] = np.dot(home_signs, wy)

        # --- Safety check: matrix must be symmetric and positive definite ---
        max_asymmetry = np.max(np.abs(G - G.T))
        if max_asymmetry > 1e-10:
            logger.error(
                f"Gram matrix is not symmetric: max asymmetry = {max_asymmetry:.2e}. "
                "This indicates a bug in the analytical Gram matrix computation."
            )
            raise ValueError(f"Non-symmetric Gram matrix (asymmetry={max_asymmetry:.2e})")

        # --- Cholesky solve ---
        try:
            cho = linalg.cho_factor(G, lower=False)
            beta = linalg.cho_solve(cho, Xty)
        except linalg.LinAlgError as e:
            logger.warning(
                f"Cholesky factorization failed for alpha={alpha}, "
                f"falling back to LU decomposition. "
                f"n_teams={n_teams}, n_plays={n_plays}. Original error: {e}"
            )
            try:
                beta = linalg.solve(G, Xty)
            except linalg.LinAlgError:
                logger.error(
                    "LU fallback also failed. Gram matrix is singular. "
                    f"n_teams={n_teams}, n_plays={n_plays}, alpha={alpha}."
                )
                raise

        # --- Reconstruct intercept (sklearn convention) ---
        intercept = y_offset - X_offset @ beta

        return beta, intercept

    def _ridge_adjust_metric(
        self,
        plays_df: pd.DataFrame,
        metric_col: str,
        season: Optional[int] = None,
        eval_week: Optional[int] = None,
    ) -> tuple[dict[str, float], dict[str, float], Optional[float]]:
        """Use ridge regression to opponent-adjust a metric.

        Per spec: Y = metric value, X = Team/Opponent IDs

        CACHING: Results are cached by (season, eval_week, metric_col, ridge_alpha, data_hash)
        to avoid redundant computation across backtest iterations. Cache is safe because
        ridge regression is deterministic given the same inputs.

        NEUTRAL-FIELD REGRESSION: If home_team column is present, we add a home
        field indicator to the design matrix. This allows the model to separately
        learn:
        1. Team strength (neutral-field) - the team coefficients
        2. Implicit home field advantage - the home indicator coefficient

        Without this, team coefficients contain implicit HFA from the data
        (EPA/success rates are higher for home teams), causing double-counting
        when SpreadGenerator adds explicit HFA.

        CHOLESKY OPTIMIZATION: Uses analytical Gram matrix computation via np.bincount
        to bypass sparse matrix construction entirely. Computes X^T W X and X^T W y
        directly from team index arrays, then solves via Cholesky decomposition.
        This is both faster (~3x) and more accurate (1e-14 vs 1e-6) than iterative sparse_cg.

        Args:
            plays_df: Prepared plays with the metric column
            metric_col: Name of column to adjust (e.g., 'is_success')
            season: Season year for cache key (optional, but recommended for caching)
            eval_week: Evaluation week for cache key (optional, but recommended for caching)

        Returns:
            Tuple of (off_adjusted, def_adjusted, learned_hfa) dicts
            learned_hfa is the implicit HFA coefficient (None if no home_team data)
        """
        global _RIDGE_ADJUST_CACHE, _CACHE_STATS

        n_plays = len(plays_df)
        if n_plays == 0:
            return {}, {}, None

        # =================================================================
        # CACHE LOOKUP
        # =================================================================
        # Check cache if season and eval_week are provided
        cache_key = None
        if season is not None and eval_week is not None:
            data_hash = _compute_data_hash(plays_df, metric_col)
            cache_key = (season, eval_week, metric_col, self.ridge_alpha, self.time_decay, data_hash)

            if cache_key in _RIDGE_ADJUST_CACHE:
                _CACHE_STATS["hits"] += 1
                # LRU: Move accessed key to end (most recently used)
                _RIDGE_ADJUST_CACHE.move_to_end(cache_key)
                cached_result = _RIDGE_ADJUST_CACHE[cache_key]
                logger.debug(
                    f"Cache HIT for ridge adjust: season={season}, week={eval_week}, "
                    f"metric={metric_col} (hits={_CACHE_STATS['hits']})"
                )
                return cached_result

            _CACHE_STATS["misses"] += 1
            logger.debug(
                f"Cache MISS for ridge adjust: season={season}, week={eval_week}, "
                f"metric={metric_col}"
            )

        start_time = time.time()

        # P3.6: Use canonical team index if available (ensures consistent dimensions)
        # This is critical for IsoPPP adjustment which uses a subset of plays (successful only)
        # but still needs the same team dimensions as SR adjustment for consistency
        if self._canonical_teams and self._team_to_idx:
            all_teams = self._canonical_teams
            team_to_idx = self._team_to_idx
        else:
            all_teams = sorted(set(plays_df["offense"]) | set(plays_df["defense"]))
            team_to_idx = {team: i for i, team in enumerate(all_teams)}
        n_teams = len(all_teams)

        # =================================================================
        # Construct team index arrays for analytical Cholesky solve
        # =================================================================
        offenses = plays_df["offense"].values
        defenses = plays_df["defense"].values

        # Vectorized team index lookup via pd.Categorical (1.5x faster than dict comprehension)
        # Using preset categories ensures indices match team_to_idx ordering
        off_idx = pd.Categorical(offenses, categories=all_teams, ordered=False).codes.astype(np.int32)
        def_idx = pd.Categorical(defenses, categories=all_teams, ordered=False).codes.astype(np.int32)

        # BUG FIX: Validate team indices - pd.Categorical returns -1 for unknown categories
        # This can happen with NaN teams, trailing spaces, name mismatches, or subset filtering
        invalid_off = (off_idx == -1)
        invalid_def = (def_idx == -1)
        if invalid_off.any() or invalid_def.any():
            n_invalid_off = invalid_off.sum()
            n_invalid_def = invalid_def.sum()
            # Get examples of unknown teams for debugging
            unknown_off = set(offenses[invalid_off][:5]) if n_invalid_off > 0 else set()
            unknown_def = set(defenses[invalid_def][:5]) if n_invalid_def > 0 else set()
            raise ValueError(
                f"Invalid team indices in ridge regression: "
                f"{n_invalid_off} unknown offenses (e.g., {unknown_off}), "
                f"{n_invalid_def} unknown defenses (e.g., {unknown_def}). "
                f"Teams not in all_teams set. Check for NaN, typos, or filtering mismatches."
            )

        # Home field signs: +1 if offense is home, -1 if defense is home, 0 if neutral
        has_home_info = "home_team" in plays_df.columns
        if has_home_info:
            home_teams = plays_df["home_team"].values
            home_valid = sum(1 for h in home_teams if pd.notna(h))
            home_pct = home_valid / n_plays * 100
            if home_pct < 90:
                logger.warning(
                    f"Home team coverage low: {home_valid}/{n_plays} ({home_pct:.1f}%). "
                    "Neutral-field regression may be unreliable."
                )
            else:
                logger.debug(f"Home team coverage: {home_valid}/{n_plays} ({home_pct:.1f}%)")

            # Compute home signs vectorized
            off_str = np.asarray(offenses, dtype=object)
            def_str = np.asarray(defenses, dtype=object)
            home_str = np.asarray(home_teams, dtype=object)
            # P3.4: Use vectorized pd.notna instead of list comprehension
            home_valid_mask = pd.notna(plays_df["home_team"]).values
            off_is_home = (off_str == home_str) & home_valid_mask
            def_is_home = (def_str == home_str) & home_valid_mask

            home_signs = np.zeros(n_plays, dtype=np.float64)
            home_signs[off_is_home] = 1.0
            home_signs[def_is_home] = -1.0
        else:
            home_signs = None

        # =================================================================
        # W: Sample weights with dynamic time decay
        # =================================================================
        # base_weight = GT × OOC × RZ × empty_yards
        # Time decay applied dynamically per eval_week
        base_weights = (
            plays_df["base_weight"].values
            if "base_weight" in plays_df.columns
            else plays_df["weight"].values
            if "weight" in plays_df.columns
            else np.ones(n_plays)
        )

        # Apply time decay dynamically (varies per eval_week)
        if self.time_decay < 1.0 and "week" in plays_df.columns:
            decay_ref = eval_week if eval_week is not None else plays_df["week"].max()
            time_weights = self.time_decay ** (decay_ref - plays_df["week"].values)
            weights = base_weights * time_weights
        else:
            weights = base_weights

        # Target: the metric value (e.g., success = 1/0)
        if metric_col == "is_success":
            y = plays_df[metric_col].astype(float).values
        else:
            y = plays_df[metric_col].values

        # P2.1: Guard against NaN in ridge inputs
        nan_in_y = np.isnan(y).sum()
        nan_in_w = np.isnan(weights).sum()
        if nan_in_y > 0 or nan_in_w > 0:
            logger.warning(
                f"NaN detected before ridge fit: {nan_in_y} in target, {nan_in_w} in weights. "
                "Dropping NaN rows to prevent ridge failure."
            )
            valid = ~(np.isnan(y) | np.isnan(weights))
            off_idx = off_idx[valid]
            def_idx = def_idx[valid]
            if home_signs is not None:
                home_signs = home_signs[valid]
            y = y[valid]
            weights = weights[valid]
            n_plays = len(y)  # Update count after filtering

        # =================================================================
        # Analytical Cholesky Ridge solve
        # =================================================================
        fit_start = time.time()
        coefficients, intercept = self._ridge_solve_cholesky(
            off_idx, def_idx, home_signs, y, weights, n_teams, self.ridge_alpha
        )
        fit_time = time.time() - fit_start

        # Separate home coefficient from team coefficients
        learned_hfa = None
        if has_home_info:
            learned_hfa = coefficients[-1]
            team_coefficients = coefficients[:-1]
            # P3.9: Debug level for per-week ridge stats (runs many times during backtest)
            logger.debug(
                f"Ridge adjust {metric_col}: intercept={intercept:.4f}, "
                f"implicit_HFA={learned_hfa:.4f} (neutral-field regression)"
            )
        else:
            team_coefficients = coefficients
            logger.debug(f"Ridge adjust {metric_col}: intercept={intercept:.4f} (no home info)")

        total_time = time.time() - start_time
        n_cols = 2 * n_teams + (1 if has_home_info else 0)
        logger.debug(
            f"  Cholesky solve: {n_plays:,} plays × {n_cols} teams+HFA, "
            f"{fit_time*1000:.1f}ms solve, {total_time*1000:.1f}ms total"
        )

        # P1.1: Post-center ridge coefficients for identifiability
        # Ridge regularization doesn't enforce sum-to-zero on coefficient groups.
        # Without centering, mean(off_coefs) ≠ 0 and mean(def_coefs) ≠ 0, causing
        # the intercept to NOT equal the true league average. Post-centering fixes
        # this without changing spread predictions (differences are invariant).
        off_coefs = team_coefficients[:n_teams]
        def_coefs = team_coefficients[n_teams:2 * n_teams]

        off_coef_mean = np.mean(off_coefs)
        def_coef_mean = np.mean(def_coefs)

        # Center coefficients to mean-zero and absorb means into intercept
        off_coefs_centered = off_coefs - off_coef_mean
        def_coefs_centered = def_coefs - def_coef_mean

        # New baselines: off_baseline absorbs offense mean, def_baseline absorbs defense mean
        off_baseline = intercept + off_coef_mean
        def_baseline = intercept + def_coef_mean

        if abs(off_coef_mean) > 0.0005 or abs(def_coef_mean) > 0.0005:
            logger.debug(
                f"  P1.1 post-centering ({metric_col}): "
                f"off_mean_coef={off_coef_mean:.5f}, def_mean_coef={def_coef_mean:.5f}, "
                f"raw_intercept={intercept:.4f} → off_base={off_baseline:.4f}, def_base={def_baseline:.4f}"
            )

        off_adjusted = {}
        def_adjusted = {}

        for team, idx in team_to_idx.items():
            # Offensive rating: baseline + centered coef
            # After centering: mean(off_adjusted) = off_baseline exactly
            off_adjusted[team] = off_baseline + off_coefs_centered[idx]
            # Defensive rating: baseline - centered coef
            # After centering: mean(def_adjusted) = def_baseline exactly
            def_adjusted[team] = def_baseline - def_coefs_centered[idx]

        # P0.1 + P1.1: Validate ridge baseline (should now pass exactly)
        mean_off = np.mean(list(off_adjusted.values()))
        mean_def = np.mean(list(def_adjusted.values()))

        logger.debug(
            f"  Ridge baseline check ({metric_col}): "
            f"off_baseline={off_baseline:.4f}, mean_off={mean_off:.4f} (Δ={abs(mean_off - off_baseline):.6f}), "
            f"def_baseline={def_baseline:.4f}, mean_def={mean_def:.4f} (Δ={abs(mean_def - def_baseline):.6f})"
        )

        # Post-centering should make drift essentially zero (floating point only)
        max_drift = max(abs(mean_off - off_baseline), abs(mean_def - def_baseline))
        if max_drift > 1e-6:
            logger.warning(
                f"Post-centering failed for {metric_col}: drift={max_drift:.8f}. "
                f"This indicates a bug in the centering logic."
            )

        # D.1: Consolidated ridge sanity summary (debug-level)
        off_std = np.std(list(off_adjusted.values()))
        def_std = np.std(list(def_adjusted.values()))
        hfa_str = f"{learned_hfa:.4f}" if learned_hfa is not None else "N/A"
        logger.debug(
            f"  Ridge sanity ({metric_col}): intercept={intercept:.4f}, "
            f"HFA={hfa_str}, "
            f"off mean={mean_off:.4f} std={off_std:.4f}, "
            f"def mean={mean_def:.4f} std={def_std:.4f}, "
            f"n_teams={n_teams}, n_plays={n_plays:,}"
        )

        # Log coefficient magnitude for tracking feature importance
        # This helps identify if new features (like RZ scoring) add signal
        coef_magnitude = np.mean(np.abs(off_coefs_centered))
        logger.info(
            f"Ridge ({metric_col}): mean|coef|={coef_magnitude:.4f}, "
            f"intercept={intercept:.4f}, HFA={hfa_str}"
        )

        # =================================================================
        # CACHE STORAGE (with LRU eviction)
        # =================================================================
        result = (off_adjusted, def_adjusted, learned_hfa)
        if cache_key is not None:
            _RIDGE_ADJUST_CACHE[cache_key] = result

            # LRU eviction: remove oldest entries if cache exceeds max size
            while len(_RIDGE_ADJUST_CACHE) > _RIDGE_CACHE_MAX_SIZE:
                evicted_key = next(iter(_RIDGE_ADJUST_CACHE))
                _RIDGE_ADJUST_CACHE.pop(evicted_key)
                _CACHE_STATS["evictions"] += 1

                # Warn once per session when eviction starts
                if not _CACHE_STATS["eviction_warned"]:
                    logger.warning(
                        f"Ridge cache exceeded max size ({_RIDGE_CACHE_MAX_SIZE}), "
                        f"LRU eviction active. Consider clearing cache between sweeps."
                    )
                    _CACHE_STATS["eviction_warned"] = True

            logger.debug(
                f"Cached ridge adjust result: season={season}, week={eval_week}, "
                f"metric={metric_col} (cache size={len(_RIDGE_ADJUST_CACHE)})"
            )

        return result

    def _calculate_turnover_stats(
        self,
        plays_df: pd.DataFrame,
        games_df: Optional[pd.DataFrame] = None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Calculate per-game turnover stats for each team (P2.6: split O/D, now INT/fumble).

        Returns separate stats for ball security (turnovers lost) and
        takeaways (turnovers forced) to enable O/D-specific turnover ratings.
        Also calculates INT and fumble breakdown for separate shrinkage.

        Args:
            plays_df: Play-by-play data with 'play_type', 'offense', 'defense' columns
            games_df: Games data for counting games played

        Returns:
            Tuple of (lost_per_game, forced_per_game, margin_per_game) dicts
            - lost_per_game: Turnovers lost per game (lower = better ball security)
            - forced_per_game: Turnovers forced per game (higher = better takeaways)
            - margin_per_game: Net margin (forced - lost) for backward compat
            Also populates self.ints_thrown, self.ints_forced, self.fumbles_lost, self.fumbles_recovered
        """
        empty_result = ({}, {}, {})

        if "play_type" not in plays_df.columns:
            logger.warning("No play_type column for turnover calculation")
            return empty_result

        # Find turnover plays by type
        # P3.3: Consolidate 6 separate groupby().size() calls into 2 groupby().agg() calls
        to_plays = plays_df[plays_df["play_type"].isin(TURNOVER_PLAY_TYPES)].copy()

        if len(to_plays) == 0:
            logger.warning("No turnover plays found")
            return empty_result

        # Add boolean columns for INT/FUM classification
        to_plays["is_int"] = to_plays["play_type"].isin(INTERCEPTION_PLAY_TYPES)
        to_plays["is_fum"] = to_plays["play_type"].isin(FUMBLE_PLAY_TYPES)

        # Count turnovers lost (offense = team that lost the ball)
        lost_agg = to_plays.groupby("offense").agg(
            total=("is_int", "count"),
            ints=("is_int", "sum"),
            fums=("is_fum", "sum"),
        )
        turnovers_lost = lost_agg["total"]
        ints_thrown_count = lost_agg["ints"]
        fumbles_lost_count = lost_agg["fums"]

        # Count turnovers forced (defense = team that forced it)
        forced_agg = to_plays.groupby("defense").agg(
            total=("is_int", "count"),
            ints=("is_int", "sum"),
            fums=("is_fum", "sum"),
        )
        turnovers_forced = forced_agg["total"]
        ints_forced_count = forced_agg["ints"]
        fumbles_rec_count = forced_agg["fums"]

        # Get all teams - DETERMINISM: Sort for consistent iteration order
        all_teams = sorted(
            set(turnovers_lost.index) | set(turnovers_forced.index) |
            set(ints_thrown_count.index) | set(ints_forced_count.index) |
            set(fumbles_lost_count.index) | set(fumbles_rec_count.index)
        )

        # Count games per team - MUST have reliable count for shrinkage calculation
        # P2.10: No arbitrary defaults; compute from data or fail loudly
        games_played = {}
        games_source = None

        if games_df is not None and len(games_df) > 0:
            # Primary: count from games_df (most reliable)
            # P3.2: Use two groupbys instead of per-team O(N) filtering
            games_source = "games_df"
            home_counts = games_df.groupby("home_team").size()
            away_counts = games_df.groupby("away_team").size()
            for team in all_teams:
                games_played[team] = max(
                    home_counts.get(team, 0) + away_counts.get(team, 0), 1
                )
        elif "game_id" in plays_df.columns:
            # Fallback: count unique game_ids from plays (reliable if game_id exists)
            games_source = "plays_df.game_id"
            for team in all_teams:
                team_plays = plays_df[
                    (plays_df["offense"] == team) | (plays_df["defense"] == team)
                ]
                n_games = team_plays["game_id"].nunique()
                if n_games == 0:
                    logger.warning(
                        f"Team {team} has turnover data but 0 games found via game_id; "
                        "this may indicate data issues"
                    )
                    n_games = 1  # Minimum to avoid division by zero
                games_played[team] = n_games
        else:
            # P2.10: Fail loudly - cannot compute reliable games count
            raise ValueError(
                "Cannot compute games played for turnover shrinkage: "
                "games_df not provided and plays_df has no 'game_id' column. "
                "Provide games_df or ensure plays_df contains game_id."
            )

        # P3.9: Debug level for per-week logging
        logger.debug(f"Games played computed from {games_source} for {len(games_played)} teams")

        # Calculate per-game stats (P2.6: separate lost/forced for O/D split)
        lost_per_game = {}
        forced_per_game = {}
        margin_per_game = {}
        ints_thrown_pg = {}
        ints_forced_pg = {}
        fumbles_lost_pg = {}
        fumbles_rec_pg = {}

        for team in all_teams:
            lost = turnovers_lost.get(team, 0)
            forced = turnovers_forced.get(team, 0)
            games = games_played[team]

            lost_per_game[team] = lost / games
            forced_per_game[team] = forced / games
            margin_per_game[team] = (forced - lost) / games

            # INT/Fumble breakdown
            ints_thrown_pg[team] = ints_thrown_count.get(team, 0) / games
            ints_forced_pg[team] = ints_forced_count.get(team, 0) / games
            fumbles_lost_pg[team] = fumbles_lost_count.get(team, 0) / games
            fumbles_rec_pg[team] = fumbles_rec_count.get(team, 0) / games

        # Store games played for Bayesian shrinkage in calculate_ratings
        self.team_games_played = games_played
        # Store INT/fumble breakdown for separate shrinkage
        self.ints_thrown = ints_thrown_pg
        self.ints_forced = ints_forced_pg
        self.fumbles_lost = fumbles_lost_pg
        self.fumbles_recovered = fumbles_rec_pg

        # Log summary stats (P3.9: debug level for per-week logging)
        avg_lost = np.mean(list(lost_per_game.values()))
        avg_forced = np.mean(list(forced_per_game.values()))
        avg_ints_thrown = np.mean(list(ints_thrown_pg.values())) if ints_thrown_pg else 0.0
        avg_ints_forced = np.mean(list(ints_forced_pg.values())) if ints_forced_pg else 0.0
        avg_fum_lost = np.mean(list(fumbles_lost_pg.values())) if fumbles_lost_pg else 0.0
        avg_fum_rec = np.mean(list(fumbles_rec_pg.values())) if fumbles_rec_pg else 0.0
        logger.debug(
            f"Calculated turnover stats for {len(all_teams)} teams: "
            f"avg lost={avg_lost:.2f}/game, avg forced={avg_forced:.2f}/game, "
            f"INT={avg_ints_thrown:.2f}/{avg_ints_forced:.2f}, FUM={avg_fum_lost:.2f}/{avg_fum_rec:.2f}"
        )

        return lost_per_game, forced_per_game, margin_per_game

    def calculate_ratings(
        self,
        plays_df: pd.DataFrame,
        games_df: Optional[pd.DataFrame] = None,
        max_week: int | None = None,
        season: int | None = None,
        team_conferences: Optional[dict[str, str]] = None,
        hfa_lookup: Optional[dict[str, float]] = None,
        fbs_teams: Optional[set[str]] = None,
    ) -> dict[str, TeamEFMRating]:
        """Calculate efficiency-based ratings for all teams.

        DATA FLOW AND NORMALIZATION ORDER
        =================================
        The pipeline processes data in this specific order:

        1. RAW METRICS (metric scale)
           - Success Rate: 0.0-1.0 (proportion)
           - IsoPPP: ~0.3 (EPA per successful play)

        2. OPPONENT ADJUSTMENT via Ridge Regression (still metric scale)
           - Ridge regression on metric-scale data (NOT normalized)
           - Intercept = league average (interpretable on metric scale)
           - Team coefficients = deviation from average
           - Implicit HFA extracted separately (neutral-field regression)

        3. POINT CONVERSION
           - SR points = (adj_sr - avg) × 80.0
           - IsoPPP points = (adj_iso - avg) × 15.0
           - Turnovers = margin × shrinkage × 4.5 pts/turnover

        4. FINAL NORMALIZATION (mean=0, std=12)
           - Each component centered by its own mean
           - All components scaled uniformly by overall std

        WHY THIS ORDER IS MATHEMATICALLY CORRECT
        ========================================
        Normalizing BEFORE opponent adjustment would be WRONG because:

        1. Ridge regression is scale-sensitive. The regularization penalty
           ||β||² depends on coefficient scale. Ridge alpha (50.0) is tuned
           for metric-scale inputs (SR in 0-1, IsoPPP in ~0.3).

        2. The intercept has natural interpretation on metric scale:
           intercept ≈ 0.42 (league average success rate)
           If data were pre-normalized, intercept would be ~0 (meaningless).

        3. Implicit HFA extraction requires metric-scale regression:
           HFA coefficient ≈ 0.03 (3% higher SR at home)
           This is interpretable and correctly separates from team skill.

        4. Point conversion factors (80.0, 15.0) are calibrated for
           opponent-adjusted metrics, not normalized values.

        The final normalization step is a simple linear transformation
        (center + scale) that preserves all inter-team relationships while
        mapping to SP+-compatible scale for spread calculation.

        MATHEMATICAL PROOF (additivity preservation):
        - Before: overall = off + def
        - After centering: new_off = (off - off_mean), new_def = (def - def_mean)
        - After scaling: new_off × scale, new_def × scale
        - Result: new_overall = new_off + new_def (relationship preserved)

        CACHING: If season and max_week are provided, ridge regression results
        are cached by (season, max_week, metric) to avoid redundant computation.
        See module-level _RIDGE_ADJUST_CACHE for details.

        Args:
            plays_df: Play-by-play data
            games_df: Optional games data for sample size info
            max_week: Maximum week allowed in training data (for data leakage prevention).
                      If provided, asserts no plays exceed this week. Also used as cache key.
            season: Season year for cache key. If provided with max_week, enables caching
                    of ridge regression results.
            team_conferences: Optional dict mapping team name to conference name.
                            If provided, applies 1.5x weight to non-conference games.
            hfa_lookup: Optional dict of team name to HFA values for efficiency fraud tax.
            fbs_teams: Optional set of FBS team names for normalization. If provided,
                       normalization uses only FBS teams (excludes FCS outliers). If None,
                       falls back to all teams in play data (legacy behavior).

        Returns:
            Dict mapping team name to TeamEFMRating
        """
        # Early return for empty plays (Week 1 with no prior game data)
        # SpreadGenerator will use mean rating (0.0) for all teams, relying on priors from other sources
        if plays_df.empty:
            logger.warning(
                "Empty plays DataFrame - no efficiency data available. "
                "Returning empty ratings (week 1 priors-only mode)."
            )
            self.team_ratings = {}
            self._canonical_teams = []
            self._team_to_idx = {}
            return {}

        # Prepare plays (with data leakage guard if max_week provided)
        # Season passed for year-conditional garbage time thresholds (2024+ clock rule change)
        prepared = self._prepare_plays(plays_df, max_week=max_week, team_conferences=team_conferences, season=season)
        # P3.9: Debug level for per-week logging
        logger.debug(f"Prepared {len(prepared)} plays for EFM")

        # P3.6: Build canonical team index ONCE, reuse throughout pipeline
        # This eliminates redundant sorted(set(...) | set(...)) calls
        self._canonical_teams = sorted(set(prepared["offense"]) | set(prepared["defense"]))
        self._team_to_idx = {team: i for i, team in enumerate(self._canonical_teams)}
        logger.debug(f"Canonical team index: {len(self._canonical_teams)} teams")

        # Calculate raw metrics
        raw_off_sr, raw_def_sr, raw_off_isoppp, raw_def_isoppp, \
            off_isoppp_real, def_isoppp_real = \
            self._calculate_raw_metrics(prepared)

        # Opponent-adjust Success Rate via ridge regression
        # This is the key: regress on SUCCESS RATE, not margins
        # NEUTRAL-FIELD: If home_team is present, regression separates team skill from HFA
        # CACHING: Results cached by (season, max_week, metric) if both are provided
        # P3.9: Debug level for per-week logging
        logger.debug("Ridge adjusting Success Rate...")
        adj_off_sr, adj_def_sr, self.learned_hfa_sr = self._ridge_adjust_metric(
            prepared, "is_success", season=season, eval_week=max_week
        )

        # Opponent-adjust IsoPPP (EPA on successful plays)
        # Only use successful plays for this
        # Note: Uses different metric_col ("ppa") so cached separately from SR
        # P3.5: No .copy() needed - _ridge_adjust_metric() only reads from DataFrame
        successful_plays = prepared[prepared["is_success"]]
        if len(successful_plays) > 1000 and "ppa" in successful_plays.columns:
            logger.debug("Ridge adjusting IsoPPP...")
            adj_off_isoppp, adj_def_isoppp, self.learned_hfa_isoppp = self._ridge_adjust_metric(
                successful_plays, "ppa", season=season, eval_week=max_week
            )
        else:
            logger.warning("Insufficient successful plays for IsoPPP adjustment, using raw")
            adj_off_isoppp = raw_off_isoppp
            adj_def_isoppp = raw_def_isoppp
            self.learned_hfa_isoppp = None

        # Store adjusted values
        self.off_success_rate = adj_off_sr
        self.def_success_rate = adj_def_sr
        self.off_isoppp = adj_off_isoppp
        self.def_isoppp = adj_def_isoppp

        # Calculate turnover stats if turnover_weight > 0 (P2.6: split O/D)
        if self.turnover_weight > 0 and "play_type" in plays_df.columns:
            logger.debug("Calculating turnover stats...")
            self.turnovers_lost, self.turnovers_forced, self.turnover_margin = \
                self._calculate_turnover_stats(plays_df, games_df)
        else:
            self.turnovers_lost = {}
            self.turnovers_forced = {}
            self.turnover_margin = {}

        # Build team ratings
        # P3.6: Use canonical team index (already sorted, computed once at start)
        all_teams = self._canonical_teams

        # P3.1: Pre-compute play counts via groupby instead of per-team O(N) filtering
        off_play_counts = prepared.groupby("offense").size()
        def_play_counts = prepared.groupby("defense").size()

        # Calculate league averages from adjusted values
        # P3.6: np.mean() is order-independent, no need to sort for determinism
        avg_sr = np.mean(list(adj_off_sr.values()))
        # P1.2: Use play-derived tracking sets instead of sentinel equality check.
        # Old approach: exclude teams where adj_isoppp == LEAGUE_AVG_ISOPPP, which also
        # excludes real teams that happen to be near-average. New approach: only include
        # teams that had enough successful plays for a real IsoPPP computation.
        valid_isoppp = [adj_off_isoppp[t] for t in off_isoppp_real if t in adj_off_isoppp]
        avg_isoppp = np.mean(valid_isoppp) if valid_isoppp else self.LEAGUE_AVG_ISOPPP

        # Calculate league average turnover rates for O/D split (P2.6)
        # P3.6: np.mean() is order-independent, no need to sort
        avg_lost = np.mean(list(self.turnovers_lost.values())) if self.turnovers_lost else 0.0
        avg_forced = np.mean(list(self.turnovers_forced.values())) if self.turnovers_forced else 0.0
        # League averages for INT/Fumble split (for separate shrinkage)
        avg_ints_thrown = np.mean(list(self.ints_thrown.values())) if self.ints_thrown else 0.0
        avg_ints_forced = np.mean(list(self.ints_forced.values())) if self.ints_forced else 0.0
        avg_fum_lost = np.mean(list(self.fumbles_lost.values())) if self.fumbles_lost else 0.0
        avg_fum_rec = np.mean(list(self.fumbles_recovered.values())) if self.fumbles_recovered else 0.0

        for team in all_teams:
            # Get adjusted metrics
            off_sr = adj_off_sr.get(team, avg_sr)
            def_sr = adj_def_sr.get(team, avg_sr)
            off_iso = adj_off_isoppp.get(team, avg_isoppp)
            def_iso = adj_def_isoppp.get(team, avg_isoppp)

            # Convert to point equivalents
            # Offensive efficiency: how much better than average
            off_eff_pts = (off_sr - avg_sr) * self.SUCCESS_RATE_TO_POINTS
            # Defensive efficiency: how much better than average (lower = better)
            def_eff_pts = (avg_sr - def_sr) * self.SUCCESS_RATE_TO_POINTS

            # Explosiveness (IsoPPP only; RZ finishing captured via RZ Leverage play weighting)
            off_exp_pts = (off_iso - avg_isoppp) * self.ISOPPP_TO_POINTS
            def_exp_pts = (avg_isoppp - def_iso) * self.ISOPPP_TO_POINTS

            efficiency_rating = off_eff_pts + def_eff_pts
            explosiveness_rating = off_exp_pts + def_exp_pts

            # P2.6: Split turnovers into offensive (ball security) and defensive (takeaways)
            # Now with separate shrinkage for INTs (skill) vs fumbles (luck)
            # Shrinkage: games / (games + k). E.g., 10-game team with k=10 keeps 50% of raw value.
            # P2.10: No arbitrary defaults - must have reliable games count
            if team not in self.team_games_played:
                # Team has efficiency data but no turnover data - compute games from plays
                if "game_id" in prepared.columns:
                    team_plays = prepared[
                        (prepared["offense"] == team) | (prepared["defense"] == team)
                    ]
                    games = max(team_plays["game_id"].nunique(), 1)
                    logger.debug(
                        f"Team {team} not in turnover stats; computed {games} games from plays"
                    )
                else:
                    raise ValueError(
                        f"Cannot determine games played for {team}: not in turnover stats "
                        "and no game_id column in plays. Provide games_df to calculate_ratings."
                    )
            else:
                games = self.team_games_played[team]

            # Separate shrinkage for INT (skill-based) and fumbles (luck-based)
            shrink_int = games / (games + self.k_int)
            shrink_fum = games / (games + self.k_fumble)

            # ---- OFFENSIVE TURNOVERS: Ball security (fewer lost = better) ----
            # INT thrown: skill (moderate shrinkage)
            raw_ints_thrown = self.ints_thrown.get(team, avg_ints_thrown)
            int_off_pts = (avg_ints_thrown - raw_ints_thrown) * shrink_int * POINTS_PER_TURNOVER
            # Fumbles lost: luck (strong shrinkage)
            raw_fum_lost = self.fumbles_lost.get(team, avg_fum_lost)
            fum_off_pts = (avg_fum_lost - raw_fum_lost) * shrink_fum * POINTS_PER_TURNOVER
            off_to_pts = int_off_pts + fum_off_pts

            # ---- DEFENSIVE TURNOVERS: Takeaways (more forced = better) ----
            # INT forced: skill (moderate shrinkage)
            raw_ints_forced = self.ints_forced.get(team, avg_ints_forced)
            int_def_pts = (raw_ints_forced - avg_ints_forced) * shrink_int * POINTS_PER_TURNOVER
            # Fumbles recovered: luck (strong shrinkage)
            raw_fum_rec = self.fumbles_recovered.get(team, avg_fum_rec)
            fum_def_pts = (raw_fum_rec - avg_fum_rec) * shrink_fum * POINTS_PER_TURNOVER
            def_to_pts = int_def_pts + fum_def_pts

            # Combined turnover rating for backward compat (should equal off_to + def_to)
            turnover_rating = off_to_pts + def_to_pts

            # Separate offensive and defensive ratings (P2.6: now includes turnovers)
            # offensive_rating = efficiency + explosiveness + ball_security
            # defensive_rating = efficiency + explosiveness + takeaways
            offensive_rating = (
                self.efficiency_weight * off_eff_pts +
                self.explosiveness_weight * off_exp_pts +
                self.turnover_weight * off_to_pts
            )
            defensive_rating = (
                self.efficiency_weight * def_eff_pts +
                self.explosiveness_weight * def_exp_pts +
                self.turnover_weight * def_to_pts
            )

            # Overall rating = O + D (P1.3: turnovers are inside O/D, NOT additive)
            # turnover_rating is diagnostic only — do not add it to overall
            overall = offensive_rating + defensive_rating

            # Get sample sizes (P3.1: use pre-computed groupby counts)
            off_plays = off_play_counts.get(team, 0)
            def_plays = def_play_counts.get(team, 0)

            self.team_ratings[team] = TeamEFMRating(
                team=team,
                raw_success_rate=raw_off_sr.get(team, avg_sr),
                raw_isoppp=raw_off_isoppp.get(team, avg_isoppp),
                adj_success_rate=off_sr,
                adj_isoppp=off_iso,
                efficiency_rating=efficiency_rating,
                explosiveness_rating=explosiveness_rating,
                offensive_rating=offensive_rating,
                defensive_rating=defensive_rating,
                special_teams_rating=0.0,  # Set separately via set_special_teams_rating()
                turnover_rating=turnover_rating,
                overall_rating=overall,
                off_plays=off_plays,
                def_plays=def_plays,
            )

        # P3.9: Debug level for per-week logging
        logger.debug(f"Calculated EFM ratings for {len(self.team_ratings)} teams")

        # Conference Strength Anchor: adjust ratings based on non-conference performance
        # Applied AFTER Ridge regression but BEFORE normalization to break conference
        # circularity. Only activates when team_conferences and games_df are both provided.
        if team_conferences is not None and games_df is not None:
            conf_anchor, conf_splits = self._calculate_conference_anchor(
                games_df, team_conferences, max_week=max_week
            )
            if conf_anchor:
                for team, rating in self.team_ratings.items():
                    adj = conf_anchor.get(team, 0.0)
                    if abs(adj) > 0.001:
                        # Separate O/D anchors: offense and defense get independent
                        # adjustments based on OOC scoring vs allowing
                        off_adj, def_adj = conf_splits.get(team, (adj / 2, adj / 2))
                        new_off = rating.offensive_rating + off_adj
                        new_def = rating.defensive_rating + def_adj
                        self.team_ratings[team] = replace(
                            rating,
                            offensive_rating=new_off,
                            defensive_rating=new_def,
                            overall_rating=new_off + new_def,
                        )

        # MOV Calibration: DEPRECATED - symmetric adjustment degraded performance
        # Kept for backward compatibility but defaults to disabled (mov_weight=0.0)
        if self.mov_weight > 0.0:
            self._apply_mov_calibration(games_df, max_week=max_week)

        # Efficiency Fraud Tax: One-way penalty for teams with high efficiency but poor wins
        # Addresses the "UCF Paradox" (4-8 teams rating like 8-4 teams) without dragging down elite teams
        # Applied after conference anchor, before normalization (same timing as MOV)
        if self.fraud_tax_enabled:
            self._apply_efficiency_fraud_tax(games_df, max_week=max_week, hfa_lookup=hfa_lookup)

        # Normalize ratings to target standard deviation
        # This ensures Team A rating - Team B rating = expected spread
        # P0: Use fbs_teams if provided (excludes FCS outliers from mean/std),
        # otherwise fall back to all_teams from play data (includes FCS - legacy behavior)
        normalization_teams = fbs_teams if fbs_teams is not None else set(all_teams)
        self._normalize_ratings(normalization_teams)

        return self.team_ratings

    def _calculate_conference_anchor(
        self,
        games_df: pd.DataFrame,
        team_conferences: dict[str, str],
        max_week: int | None = None,
        anchor_scale: float = 0.08,
        prior_games: int = 30,
        max_adjustment: float = 2.0,
    ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        """Calculate conference strength adjustment from non-conference game performance.

        Addresses conference circularity in Ridge regression: teams in a weak conference
        inflate each other's opponent-adjusted metrics because the Ridge system is nearly
        closed for conference games. This anchor uses out-of-conference (OOC) results as
        an external reference point.

        Algorithm:
        1. Identify non-conference FBS-vs-FBS games from games_df
        2. Calculate average scoring margin per conference in OOC games
        3. Apply Bayesian shrinkage: n_games / (n_games + prior_games) to suppress
           early-season noise when sample sizes are small
        4. Scale the deviation to produce a per-team adjustment (capped at +-max_adjustment)
        5. Compute O/D-specific split ratios from Ridge-derived ratings to allocate
           more anchor correction to the weaker side (offense or defense)

        The adjustment is applied to overall_rating BEFORE normalization, so it
        affects the relative ordering of teams across conferences while preserving
        intra-conference relationships from Ridge regression.

        Args:
            games_df: Games DataFrame with home_team, away_team, home_points, away_points
            team_conferences: Dict mapping team name to conference name
            max_week: If provided, only use games through this week
            anchor_scale: Converts margin deviation to rating points (default 0.08).
                         With Bayesian shrinkage, this controls the maximum impact.
                         E.g., margin=+10, full shrinkage -> adj = 10 * 0.08 = 0.8 pts
            prior_games: Bayesian prior equivalent games for shrinkage (default 30).
                        With 15 OOC games, shrinkage = 15/(15+30) = 0.33.
                        With 60 OOC games, shrinkage = 60/(60+30) = 0.67.
            max_adjustment: Hard cap on adjustment magnitude (default 2.0 pts)

        Returns:
            Tuple of:
            - Dict mapping team name to conference anchor adjustment (in pre-normalization points)
            - Dict mapping team name to (off_fraction, def_fraction) split ratios
              where off_fraction + def_fraction = 1.0 and each >= 0.3
        """
        if games_df is None or len(games_df) == 0 or not team_conferences:
            return {}, {}

        df = games_df.copy()

        # Filter to games within the training window
        if max_week is not None and "week" in df.columns:
            df = df[df["week"] <= max_week]

        # Need scoring data
        required = ["home_team", "away_team", "home_points", "away_points"]
        if not all(col in df.columns for col in required):
            logger.debug("Conference anchor: missing required columns in games_df")
            return {}, {}

        # Drop games without scores
        df = df.dropna(subset=["home_points", "away_points"])

        # Map teams to conferences
        df["home_conf"] = df["home_team"].map(team_conferences)
        df["away_conf"] = df["away_team"].map(team_conferences)

        # Non-conference = both teams are FBS (have conference) and different conferences
        ooc_mask = (
            df["home_conf"].notna() &
            df["away_conf"].notna() &
            (df["home_conf"] != df["away_conf"])
        )
        ooc_games = df[ooc_mask].copy()

        if len(ooc_games) < 10:
            logger.debug(f"Conference anchor: only {len(ooc_games)} OOC games, skipping")
            return {}, {}

        # Compute SEPARATE offensive and defensive conference anchors
        # from OOC points scored (offensive signal) and points allowed (defensive signal).
        # A conference can have a positive offensive anchor (good offenses) but negative
        # defensive anchor (weak defenses), which a single composite margin would hide.
        home_points = ooc_games["home_points"].values
        away_points = ooc_games["away_points"].values
        home_confs = ooc_games["home_conf"].values
        away_confs = ooc_games["away_conf"].values

        # Build (conference, pts_scored, pts_allowed) for all team appearances
        confs = np.concatenate([home_confs, away_confs])
        scored = np.concatenate([home_points, away_points])
        allowed = np.concatenate([away_points, home_points])

        ooc_df = pd.DataFrame({
            "conference": confs,
            "scored": scored,
            "allowed": allowed,
            "margin": scored - allowed,
        })

        # Aggregate per conference
        conf_stats = ooc_df.groupby("conference").agg(
            mean_scored=("scored", "mean"),
            mean_allowed=("allowed", "mean"),
            mean_margin=("margin", "mean"),
            n_games=("margin", "count"),
        )

        # FBS-wide averages (should be symmetric for FBS-vs-FBS)
        fbs_avg_scored = ooc_df["scored"].mean()
        fbs_avg_allowed = ooc_df["allowed"].mean()

        # Calculate per-conference SEPARATE O and D adjustments with Bayesian shrinkage
        conf_off_adjustments = {}
        conf_def_adjustments = {}
        conf_adjustments = {}  # composite for backward-compatible total
        for conf, row in conf_stats.iterrows():
            n = row["n_games"]
            shrinkage = n / (n + prior_games)

            # Offensive anchor: conference scores MORE than average = positive
            off_deviation = row["mean_scored"] - fbs_avg_scored
            raw_off_adj = shrinkage * anchor_scale * off_deviation
            off_adj = np.clip(raw_off_adj, -max_adjustment, max_adjustment)

            # Defensive anchor: conference ALLOWS LESS than average = positive
            # (allowing fewer points is good for defense)
            def_deviation = fbs_avg_allowed - row["mean_allowed"]
            raw_def_adj = shrinkage * anchor_scale * def_deviation
            def_adj = np.clip(raw_def_adj, -max_adjustment, max_adjustment)

            conf_off_adjustments[conf] = off_adj
            conf_def_adjustments[conf] = def_adj
            conf_adjustments[conf] = off_adj + def_adj

        # Log conference anchor adjustments (separate O and D)
        sorted_confs = sorted(conf_adjustments.items(), key=lambda x: x[1])
        n_nonzero = sum(1 for _, a in conf_adjustments.items() if abs(a) > 0.01)
        logger.info(
            f"Conference Anchor ({len(ooc_games)} OOC games, "
            f"FBS avg scored={fbs_avg_scored:.1f} allowed={fbs_avg_allowed:.1f}, "
            f"{n_nonzero} confs adjusted):"
        )
        for conf, adj in sorted_confs:
            n = conf_stats.loc[conf, "n_games"] if conf in conf_stats.index else 0
            shrink = n / (n + prior_games)
            off_a = conf_off_adjustments.get(conf, 0.0)
            def_a = conf_def_adjustments.get(conf, 0.0)
            if abs(adj) > 0.01 or abs(off_a - def_a) > 0.02:
                scored_v = conf_stats.loc[conf, "mean_scored"]
                allowed_v = conf_stats.loc[conf, "mean_allowed"]
                logger.info(
                    f"  {conf}: scored={scored_v:.1f} allowed={allowed_v:.1f} "
                    f"({n} games, shrink={shrink:.2f}) -> "
                    f"off={off_a:+.2f} def={def_a:+.2f} total={adj:+.2f}"
                )

        # Map conference adjustments to individual teams (separate O and D)
        team_adjustments = {}
        team_splits = {}
        for team, conf in team_conferences.items():
            off_a = conf_off_adjustments.get(conf, 0.0)
            def_a = conf_def_adjustments.get(conf, 0.0)
            team_adjustments[team] = off_a + def_a
            team_splits[team] = (off_a, def_a)  # separate O and D anchor values

        return team_adjustments, team_splits

    def _apply_mov_calibration(
        self,
        games_df: pd.DataFrame,
        max_week: int | None = None,
    ) -> None:
        """Apply Margin of Victory calibration layer to ratings.

        This method addresses the "UCF Paradox": EFM is outcome-blind (based only on
        play efficiency), so a 4-8 team can rate nearly identically to an 8-4 team
        if their success rates are similar. This calibration grounds efficiency ratings
        in actual game outcomes.

        Algorithm:
        1. For each team, calculate expected margin based on current EFM ratings:
           expected_margin = sum(team_rating - opponent_rating) / n_games
        2. Calculate actual margin from game results:
           actual_margin = sum(points_for - points_against) / n_games
        3. Calculate residual (actual outperformance vs expectation):
           residual = actual_margin - expected_margin
        4. Scale residual to be on same magnitude as ratings (≈1 residual point = 1 rating point)
        5. Cap residual to prevent extreme swings from blowouts: clamp to [-mov_cap, +mov_cap]
        6. Blend into ratings: final_rating = efm_rating + mov_weight × residual_scaled

        WALK-FORWARD SAFETY:
        - Only uses games from weeks ≤ max_week (same as training data)
        - Expected margins computed from current EFM ratings (already opponent-adjusted)
        - No future data leakage

        EARLY SEASON HANDLING:
        - Residuals are per-game averages (not cumulative), so work with 1+ games
        - Small sample sizes naturally result in smaller adjustments due to fewer data points

        INTERPRETATION:
        - Positive residual: Team wins by more than their efficiency predicts (e.g., clutch finishing)
        - Negative residual: Team wins by less than expected (e.g., close losses, bad luck)
        - Zero residual: Team performs exactly as their efficiency suggests

        Args:
            games_df: Games DataFrame with columns: home_team, away_team, home_points, away_points
            max_week: Maximum week to include (for walk-forward chronology)

        Side Effects:
            Updates self.team_ratings with MOV-calibrated ratings
        """
        if self.mov_weight == 0.0:
            logger.debug("MOV calibration disabled (mov_weight=0.0)")
            return

        if games_df is None or len(games_df) == 0:
            logger.warning("MOV calibration: no games_df provided, skipping")
            return

        # Validate required columns
        required = ["home_team", "away_team", "home_points", "away_points"]
        if not all(col in games_df.columns for col in required):
            logger.warning(f"MOV calibration: missing required columns {required}, skipping")
            return

        df = games_df.copy()

        # Filter to training window
        if max_week is not None and "week" in df.columns:
            df = df[df["week"] <= max_week]

        # Drop games without scores
        df = df.dropna(subset=["home_points", "away_points"])

        if len(df) == 0:
            logger.warning("MOV calibration: no complete games in training window, skipping")
            return

        # Get current EFM ratings (before MOV adjustment)
        # These are PRE-NORMALIZATION ratings (still in raw point scale)
        team_efm_ratings = {
            team: rating.overall_rating
            for team, rating in self.team_ratings.items()
        }

        # Calculate per-team statistics
        team_stats = {}

        for team in team_efm_ratings.keys():
            # Get all games involving this team
            home_games = df[df["home_team"] == team].copy()
            away_games = df[df["away_team"] == team].copy()

            n_home = len(home_games)
            n_away = len(away_games)
            n_games = n_home + n_away

            if n_games == 0:
                continue  # Skip teams with no games in training window

            # Calculate actual margin (points_for - points_against)
            home_actual = (home_games["home_points"] - home_games["away_points"]).sum()
            away_actual = (away_games["away_points"] - away_games["home_points"]).sum()
            total_actual_margin = home_actual + away_actual
            actual_margin_per_game = total_actual_margin / n_games

            # Calculate expected margin based on EFM ratings
            # expected_margin = team_rating - opponent_rating for each game
            home_expected = 0.0
            for _, game in home_games.iterrows():
                opp = game["away_team"]
                if opp in team_efm_ratings:
                    home_expected += team_efm_ratings[team] - team_efm_ratings[opp]
                # If opponent not in ratings (FCS, etc.), assume 0 differential

            away_expected = 0.0
            for _, game in away_games.iterrows():
                opp = game["home_team"]
                if opp in team_efm_ratings:
                    away_expected += team_efm_ratings[team] - team_efm_ratings[opp]

            total_expected_margin = home_expected + away_expected
            expected_margin_per_game = total_expected_margin / n_games

            # Calculate residual (actual - expected)
            residual = actual_margin_per_game - expected_margin_per_game

            team_stats[team] = {
                "n_games": n_games,
                "actual_margin_per_game": actual_margin_per_game,
                "expected_margin_per_game": expected_margin_per_game,
                "residual": residual,
            }

        # Calculate scaling factor
        # The residual is already in points per game, which is naturally on the same scale
        # as rating points (since ratings predict point differential). So scale = 1.0.
        # However, we apply a cap to prevent extreme swings from blowout schedules.

        # Apply MOV adjustment to ratings
        n_adjusted = 0
        total_adjustment = 0.0

        for team, rating in self.team_ratings.items():
            if team not in team_stats:
                continue  # No games, no adjustment

            stats = team_stats[team]
            residual = stats["residual"]

            # Cap the residual to prevent extreme swings
            capped_residual = np.clip(residual, -self.mov_cap, self.mov_cap)

            # Calculate adjustment: mov_weight × capped_residual
            adjustment = self.mov_weight * capped_residual

            # Apply to overall rating (split evenly between offense and defense)
            # This preserves the overall = off + def invariant
            new_off = rating.offensive_rating + adjustment / 2
            new_def = rating.defensive_rating + adjustment / 2
            self.team_ratings[team] = replace(
                rating,
                offensive_rating=new_off,
                defensive_rating=new_def,
                overall_rating=new_off + new_def,
            )

            n_adjusted += 1
            total_adjustment += adjustment

        if n_adjusted > 0:
            avg_adjustment = total_adjustment / n_adjusted
            logger.info(
                f"MOV calibration applied to {n_adjusted} teams "
                f"(weight={self.mov_weight:.2f}, cap=±{self.mov_cap:.1f}pts, "
                f"avg_adjustment={avg_adjustment:+.2f}pts)"
            )
        else:
            logger.warning("MOV calibration: no teams adjusted (no game data)")

    def _apply_efficiency_fraud_tax(
        self,
        games_df: pd.DataFrame,
        max_week: int | None = None,
        hfa_lookup: Optional[dict[str, float]] = None,
    ) -> None:
        """Apply asymmetric "Efficiency Fraud" tax to ratings.

        PROBLEM: The "UCF Paradox" - Teams with high efficiency metrics but terrible
        win-loss records (e.g., UCF 2024: 4-8 record but rated #15 in efficiency).
        EFM is outcome-blind, so a team that loses close games can rate as high as
        a team that wins them.

        SOLUTION: One-way penalty for teams whose actual wins significantly lag their
        expected wins based on EFM ratings. This is NOT a symmetric MOV calibration -
        we NEVER boost over-performers, only penalize under-performers.

        Algorithm:
        1. Calculate Expected Win Total for each team:
           - For each game: P(win) = 1 / (1 + exp(-(team_rating - opp_rating + HFA) / 7.0))
           - Sum P(win) across all games
        2. Calculate Actual Win Total from game results
        3. Identify "Efficiency Frauds": win_gap = expected_wins - actual_wins
        4. Apply fixed penalty if win_gap > threshold:
           - Rating adjustment: -fraud_tax_penalty (default -2.0 points)
           - Split evenly between offense and defense to preserve overall = off + def
        5. NEVER apply positive adjustments (asymmetric by design)

        WALK-FORWARD SAFETY:
        - Only uses games from weeks ≤ max_week (same as training data)
        - Expected wins computed from current EFM ratings (already opponent-adjusted)
        - No future data leakage

        DESIGN CHOICES:
        - Fixed penalty (not proportional): Simple, predictable, prevents cascading effects
        - One-way only: Never reward over-performance, only penalize efficiency fraud
        - Win probability logistic: 7.0-point spread ≈ 1 std dev of game outcomes

        Args:
            games_df: Games DataFrame with columns: home_team, away_team, home_points, away_points
            max_week: Maximum week to include (for walk-forward chronology)
            hfa_lookup: Optional dict mapping team names to team-specific HFA values.
                If provided, uses team-specific HFA for expected win calculations.
                Falls back to 2.5 for teams not in the lookup.

        Side Effects:
            Updates self.team_ratings with fraud tax penalties applied
        """
        if not self.fraud_tax_enabled:
            logger.debug("Efficiency Fraud tax disabled")
            return

        if games_df is None or len(games_df) == 0:
            logger.warning("Efficiency Fraud tax: no games_df provided, skipping")
            return

        # Validate required columns
        required = ["home_team", "away_team", "home_points", "away_points"]
        if not all(col in games_df.columns for col in required):
            logger.warning(f"Efficiency Fraud tax: missing required columns {required}, skipping")
            return

        df = games_df.copy()

        # Filter to training window
        if max_week is not None and "week" in df.columns:
            df = df[df["week"] <= max_week]

        # Drop games without scores
        df = df.dropna(subset=["home_points", "away_points"])

        if len(df) == 0:
            logger.warning("Efficiency Fraud tax: no complete games in training window, skipping")
            return

        # Get current EFM ratings (before fraud tax)
        # These are PRE-NORMALIZATION ratings (still in raw point scale)
        team_efm_ratings = {
            team: rating.overall_rating
            for team, rating in self.team_ratings.items()
        }

        # Default HFA value for expected win calculation (roughly league average)
        DEFAULT_HFA = 2.5

        # Logistic function parameter: 7.0 points ≈ 1 std dev of game outcomes
        LOGISTIC_SCALE = 7.0

        # =================================================================
        # VECTORIZED: Compute expected/actual wins in one pass via groupby
        # Replaces O(teams × games) per-team filtering with O(games) merge
        # =================================================================

        # Create ratings Series for vectorized lookup
        ratings_series = pd.Series(team_efm_ratings, name="rating")

        # Merge ratings onto games (home and away)
        df["home_rating"] = df["home_team"].map(ratings_series)
        df["away_rating"] = df["away_team"].map(ratings_series)

        # HFA lookup (vectorized)
        if hfa_lookup:
            hfa_series = pd.Series(hfa_lookup, name="hfa")
            df["home_hfa"] = df["home_team"].map(hfa_series).fillna(DEFAULT_HFA)
        else:
            df["home_hfa"] = DEFAULT_HFA

        # For unrated opponents (FCS), use 0 rating (team's rating becomes the diff)
        df["home_rating"] = df["home_rating"].fillna(0)
        df["away_rating"] = df["away_rating"].fillna(0)

        # Compute win probabilities vectorized
        # Home team perspective: rating_diff = home_rating - away_rating + home_hfa
        home_diff = df["home_rating"] - df["away_rating"] + df["home_hfa"]
        df["home_win_prob"] = 1.0 / (1.0 + np.exp(-home_diff / LOGISTIC_SCALE))
        df["away_win_prob"] = 1.0 - df["home_win_prob"]

        # Actual wins (1 if won, 0 otherwise)
        df["home_won"] = (df["home_points"] > df["away_points"]).astype(float)
        df["away_won"] = (df["away_points"] > df["home_points"]).astype(float)

        # Aggregate per team using two groupbys (home games and away games)
        home_stats = df.groupby("home_team").agg(
            home_expected=("home_win_prob", "sum"),
            home_actual=("home_won", "sum"),
            home_games=("home_team", "count"),
        )
        away_stats = df.groupby("away_team").agg(
            away_expected=("away_win_prob", "sum"),
            away_actual=("away_won", "sum"),
            away_games=("away_team", "count"),
        )

        # Combine home and away stats
        # P3.7: Replace per-team conditional .loc lookups with a single join
        combined_stats = home_stats.join(away_stats, how="outer").fillna(0)
        combined_stats["n_games"] = combined_stats["home_games"] + combined_stats["away_games"]
        combined_stats["expected_wins"] = combined_stats["home_expected"] + combined_stats["away_expected"]
        combined_stats["actual_wins"] = combined_stats["home_actual"] + combined_stats["away_actual"]
        combined_stats["win_gap"] = combined_stats["expected_wins"] - combined_stats["actual_wins"]
        combined_stats = combined_stats[combined_stats["n_games"] > 0]
        team_win_stats = combined_stats[["n_games", "actual_wins", "expected_wins", "win_gap"]].to_dict("index")

        # Apply fraud tax to under-performers
        n_penalized = 0
        fraud_teams = []

        for team, rating in self.team_ratings.items():
            if team not in team_win_stats:
                continue  # No games, no adjustment

            stats = team_win_stats[team]
            win_gap = stats["win_gap"]

            # Only penalize if win_gap exceeds threshold (one-way asymmetric)
            if win_gap > self.fraud_tax_threshold:
                # Apply fixed penalty (split evenly between O/D to preserve overall = off + def)
                penalty = -self.fraud_tax_penalty
                new_off = rating.offensive_rating + penalty / 2
                new_def = rating.defensive_rating + penalty / 2
                self.team_ratings[team] = replace(
                    rating,
                    offensive_rating=new_off,
                    defensive_rating=new_def,
                    overall_rating=new_off + new_def,
                )

                n_penalized += 1
                fraud_teams.append(
                    f"{team}: {stats['actual_wins']:.0f}-? (exp={stats['expected_wins']:.1f}, "
                    f"gap={win_gap:+.1f})"
                )

        if n_penalized > 0:
            logger.info(
                f"Efficiency Fraud tax applied to {n_penalized} teams "
                f"(penalty=-{self.fraud_tax_penalty:.1f}pts, threshold={self.fraud_tax_threshold:.1f} wins)"
            )
            for fraud_info in fraud_teams:
                logger.info(f"  FRAUD: {fraud_info}")
        else:
            logger.debug("Efficiency Fraud tax: no teams penalized (all within threshold)")

    def _normalize_ratings(self, fbs_teams: set[str]) -> None:
        """Normalize ratings to target standard deviation.

        IMPORTANT: This is the FINAL step after opponent adjustment and point
        conversion. Normalization must happen AFTER ridge regression because:
        1. Ridge regression needs metric-scale input (not normalized)
        2. Normalization is a linear transform that preserves all relationships
        3. This order ensures team coefficients have proper interpretation

        Centers each component (O/D/TO/efficiency/explosiveness) by its own mean,
        then scales all components uniformly. This ensures:
        - Overall rating has mean=0, std=rating_std (for spread calculation)
        - Each component is properly centered by its own mean
        - Relationship overall = off + def is preserved (P2.6: turnovers now inside O/D)
        - Components remain interpretable (mean=0 for each)

        Math verification (P2.6 update):
        - mean(overall) = mean(off) + mean(def) by linearity
        - After centering each component: new_overall = new_off + new_def
        - Turnover effects are embedded in O (ball security) and D (takeaways)
        - turnover_rating is kept as diagnostic (off_to + def_to)

        Args:
            fbs_teams: Set of FBS team names (normalization based on these)
        """
        if not self.team_ratings:
            return

        # Get current FBS ratings for normalization stats
        fbs_ratings = [r for team, r in self.team_ratings.items() if team in fbs_teams]

        if not fbs_ratings:
            return

        # Calculate component means from FBS teams (P2.5 fix: each component by its own mean)
        overall_values = [r.overall_rating for r in fbs_ratings]
        offense_values = [r.offensive_rating for r in fbs_ratings]
        defense_values = [r.defensive_rating for r in fbs_ratings]
        turnover_values = [r.turnover_rating for r in fbs_ratings]
        efficiency_values = [r.efficiency_rating for r in fbs_ratings]
        explosiveness_values = [r.explosiveness_rating for r in fbs_ratings]

        overall_mean = np.mean(overall_values)
        overall_std = np.std(overall_values)
        offense_mean = np.mean(offense_values)
        defense_mean = np.mean(defense_values)
        turnover_mean = np.mean(turnover_values)
        efficiency_mean = np.mean(efficiency_values)
        explosiveness_mean = np.mean(explosiveness_values)

        if overall_std == 0:
            return

        # Calculate scale factor from overall rating
        scale = self.rating_std / overall_std

        # P3.9: Debug level for per-week logging
        logger.debug(
            f"Normalizing ratings: mean {overall_mean:.2f} → 0, "
            f"std {overall_std:.2f} → {self.rating_std:.1f} (scale={scale:.2f}x)"
        )
        logger.debug(
            f"Component means: off={offense_mean:.2f}, def={defense_mean:.2f}, "
            f"to={turnover_mean:.2f}, eff={efficiency_mean:.2f}, exp={explosiveness_mean:.2f}"
        )

        # Apply normalization to all teams
        # Each component is centered by its own mean, then scaled uniformly
        for team, rating in self.team_ratings.items():
            new_offense = (rating.offensive_rating - offense_mean) * scale
            new_defense = (rating.defensive_rating - defense_mean) * scale
            new_turnover = (rating.turnover_rating - turnover_mean) * scale
            new_efficiency = (rating.efficiency_rating - efficiency_mean) * scale
            new_explosiveness = (rating.explosiveness_rating - explosiveness_mean) * scale

            # Overall = off + def (P1.3: turnover_rating is diagnostic, NOT additive)
            self.team_ratings[team] = replace(
                rating,
                efficiency_rating=new_efficiency,
                explosiveness_rating=new_explosiveness,
                offensive_rating=new_offense,
                defensive_rating=new_defense,
                turnover_rating=new_turnover,
                overall_rating=new_offense + new_defense,
            )

    def get_rating(self, team: str) -> float:
        """Get overall rating for a team.

        Args:
            team: Team name

        Returns:
            Overall rating (0.0 if unknown)
        """
        if team in self.team_ratings:
            return self.team_ratings[team].overall_rating
        return 0.0

    def get_offensive_rating(self, team: str) -> float:
        """Get offensive rating for a team.

        Args:
            team: Team name

        Returns:
            Offensive rating (0.0 if unknown). Higher = better offense.
        """
        if team in self.team_ratings:
            return self.team_ratings[team].offensive_rating
        return 0.0

    def get_defensive_rating(self, team: str) -> float:
        """Get defensive rating for a team.

        Args:
            team: Team name

        Returns:
            Defensive rating (0.0 if unknown). Higher = better defense.
        """
        if team in self.team_ratings:
            return self.team_ratings[team].defensive_rating
        return 0.0

    def get_special_teams_rating(self, team: str) -> float:
        """Get special teams rating for a team (DIAGNOSTIC ONLY).

        NOTE (P2.7): This rating is NOT included in overall_rating.
        Special teams is applied as a separate adjustment layer in SpreadGenerator
        using SpecialTeamsModel.get_matchup_differential(). This method exists
        for diagnostic/reporting purposes only.

        Args:
            team: Team name

        Returns:
            Special teams rating (0.0 if unknown). Currently FG efficiency only.
        """
        if team in self.team_ratings:
            return self.team_ratings[team].special_teams_rating
        return 0.0

    def set_special_teams_rating(self, team: str, rating: float) -> None:
        """Set special teams rating for a team (DIAGNOSTIC ONLY).

        NOTE (P2.7): This rating is stored for reporting but NOT included in
        overall_rating. SpreadGenerator handles ST as a separate adjustment layer
        using SpecialTeamsModel.get_matchup_differential() to avoid double-counting.

        Use this method to populate ST ratings for display in get_ratings_df(),
        but do NOT expect it to affect spread predictions (those come from
        SpecialTeamsModel directly in SpreadGenerator).

        Args:
            team: Team name
            rating: Special teams rating (per-game point value)
        """
        if team in self.team_ratings:
            self.team_ratings[team] = replace(
                self.team_ratings[team],
                special_teams_rating=rating,
            )

    def get_ratings_df(self) -> pd.DataFrame:
        """Get ratings as DataFrame sorted by overall rating.

        Columns follow the JP+ Power Ratings Display Protocol (see CLAUDE.md):
        Rank | Team | Overall | Offense (rank) | Defense (rank) | Special Teams (rank)

        Returns:
            DataFrame with all team ratings including O/D/ST breakdown
        """
        if not self.team_ratings:
            return pd.DataFrame()

        data = [
            {
                "team": r.team,
                "overall": round(r.overall_rating, 1),
                "offense": round(r.offensive_rating, 1),
                "defense": round(r.defensive_rating, 1),
                "special_teams": round(r.special_teams_rating, 2),
                "efficiency": round(r.efficiency_rating, 1),
                "explosiveness": round(r.explosiveness_rating, 1),
                "adj_sr": round(r.adj_success_rate, 3),
                "adj_isoppp": round(r.adj_isoppp, 3),
                "raw_sr": round(r.raw_success_rate, 3),
                "off_plays": r.off_plays,
            }
            for r in self.team_ratings.values()
        ]

        df = pd.DataFrame(data)
        return df.sort_values("overall", ascending=False).reset_index(drop=True)

    def predict_margin(
        self,
        home_team: str,
        away_team: str,
        neutral_site: bool = False,
        hfa: float = 2.5,
    ) -> float:
        """Predict point margin for a game.

        Args:
            home_team: Home team name
            away_team: Away team name
            neutral_site: Whether game is at neutral site
            hfa: Home field advantage in points

        Returns:
            Predicted margin (positive = home favored)
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        margin = home_rating - away_rating

        if not neutral_site:
            margin += hfa

        return margin
