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
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from config.settings import get_settings
from config.play_types import (
    TURNOVER_PLAY_TYPES,
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
# =============================================================================

_RIDGE_ADJUST_CACHE: dict[tuple, tuple[dict, dict, Optional[float]]] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def clear_ridge_cache() -> dict:
    """Clear the ridge adjustment cache and return stats.

    Returns:
        Dict with cache statistics (hits, misses) before clearing.
    """
    global _RIDGE_ADJUST_CACHE, _CACHE_STATS
    stats = _CACHE_STATS.copy()
    _RIDGE_ADJUST_CACHE.clear()
    _CACHE_STATS = {"hits": 0, "misses": 0}
    logger.info(f"Ridge cache cleared. Previous stats: {stats}")
    return stats


def get_ridge_cache_stats() -> dict:
    """Get current cache statistics.

    Returns:
        Dict with hits, misses, size, and hit_rate.
    """
    total = _CACHE_STATS["hits"] + _CACHE_STATS["misses"]
    hit_rate = _CACHE_STATS["hits"] / total if total > 0 else 0.0
    return {
        "hits": _CACHE_STATS["hits"],
        "misses": _CACHE_STATS["misses"],
        "size": len(_RIDGE_ADJUST_CACHE),
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
    hash_parts = [
        str(n_plays),
        hashlib.md5(sample["offense"].str.cat(sep=",").encode()).hexdigest()[:8],
        hashlib.md5(sample["defense"].str.cat(sep=",").encode()).hexdigest()[:8],
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


def is_garbage_time_vectorized(periods: np.ndarray, score_diffs: np.ndarray) -> np.ndarray:
    """Vectorized garbage time detection for entire DataFrame.

    P3.2 optimization: Replaces row-wise apply with vectorized numpy operations.

    Args:
        periods: Array of quarter/period values (1-4)
        score_diffs: Array of absolute score differentials

    Returns:
        Boolean array indicating garbage time plays
    """
    settings = get_settings()

    # Build threshold array based on period
    # Default to Q4 threshold for periods outside 1-4 (OT, etc.)
    thresholds = np.full(len(periods), settings.garbage_time_q4, dtype=np.float64)
    thresholds[periods == 1] = settings.garbage_time_q1
    thresholds[periods == 2] = settings.garbage_time_q2
    thresholds[periods == 3] = settings.garbage_time_q3
    thresholds[periods == 4] = settings.garbage_time_q4

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
    turnover_rating: float  # Turnover margin contribution (higher = more takeaways)

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
        efficiency_weight: float = 0.54,  # Reduced from 0.60 to make room for turnovers
        explosiveness_weight: float = 0.36,  # Reduced from 0.40 to make room for turnovers
        turnover_weight: float = 0.10,  # 10% weight for turnovers (like SP+)
        turnover_prior_strength: float = 10.0,  # Bayesian shrinkage for turnover margin
        garbage_time_weight: float = 0.1,  # Weight for garbage time plays (0 to discard)
        rating_std: float = 12.0,  # Target std for ratings (SP+ uses ~12)
        asymmetric_garbage: bool = True,  # Only penalize trailing team in garbage time
        time_decay: float = 1.0,  # Per-week decay factor (1.0 = no decay, 0.95 = 5% per week)
    ):
        """Initialize Efficiency Foundation Model.

        Args:
            ridge_alpha: Regularization for opponent adjustment
            efficiency_weight: Weight for success rate component (default 0.54)
            explosiveness_weight: Weight for IsoPPP component (default 0.36)
            turnover_weight: Weight for turnover margin component (default 0.10, like SP+)
            turnover_prior_strength: Bayesian shrinkage for turnover margin (default 10).
                                    Equivalent to 10 games of 0 margin prior data. Higher = more
                                    regression toward 0. Prevents overweighting small-sample TO luck.
            garbage_time_weight: Weight for garbage time plays (0.1 recommended, 0 to discard)
            rating_std: Target standard deviation for ratings. Set to 12.0 for SP+-like scale
                       where Team A - Team B = expected spread. Higher = more spread between teams.
            asymmetric_garbage: If True, only trailing team's garbage time plays are down-weighted.
                              Leading team keeps full weight (they earned the blowout through efficiency).
            time_decay: Per-week decay factor for play weights. 1.0 = no decay (all weeks equal).
                       0.95 = 5% decay per week (Week 1 plays get ~0.54 weight by Week 12).
                       Formula: weight *= decay ^ (max_week - play_week)
        """
        self.ridge_alpha = ridge_alpha
        self.efficiency_weight = efficiency_weight
        self.rating_std = rating_std
        self.explosiveness_weight = explosiveness_weight
        self.turnover_weight = turnover_weight
        self.turnover_prior_strength = turnover_prior_strength
        self.garbage_time_weight = garbage_time_weight
        self.asymmetric_garbage = asymmetric_garbage
        self.time_decay = time_decay

        self.team_ratings: dict[str, TeamEFMRating] = {}

        # Store opponent-adjusted values
        self.off_success_rate: dict[str, float] = {}
        self.def_success_rate: dict[str, float] = {}
        self.off_isoppp: dict[str, float] = {}
        self.def_isoppp: dict[str, float] = {}

        # Turnover stats (P2.6: split into O/D components)
        self.turnovers_lost: dict[str, float] = {}  # Per-game turnovers lost (ball security)
        self.turnovers_forced: dict[str, float] = {}  # Per-game turnovers forced (takeaways)
        self.turnover_margin: dict[str, float] = {}  # Per-game net margin (for backward compat)
        self.team_games_played: dict[str, int] = {}  # Games played per team (for TO shrinkage)

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
        self, plays_df: pd.DataFrame, max_week: int | None = None
    ) -> pd.DataFrame:
        """Filter and prepare plays for analysis.

        Filters:
        1. Non-scrimmage plays (special teams, penalties, period markers)
        2. Plays with missing/invalid data

        Args:
            plays_df: Raw play-by-play DataFrame
            max_week: Maximum week allowed in training data (for data leakage prevention).
                      If provided, asserts no plays exceed this week and uses it for time_decay.

        Returns:
            Filtered DataFrame with success and garbage time flags
        """
        # P2.9: Validate and normalize input data first
        df = self._validate_and_normalize_plays(plays_df)
        initial_count = len(df)

        # DATA LEAKAGE GUARD: Verify no future weeks in training data
        if max_week is not None and "week" in df.columns:
            actual_max = df["week"].max()
            assert actual_max <= max_week, (
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
        df = df[keep_mask]

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
        df["score_diff"] = (df["offense_score"] - df["defense_score"]).abs()
        df["is_garbage_time"] = is_garbage_time_vectorized(
            df["period"].values,
            df["score_diff"].values,
        )

        # P3.2: Vectorized weight calculation (replaces row-wise apply)
        if self.garbage_time_weight == 0:
            # Discard garbage time plays entirely
            df = df[~df["is_garbage_time"]]
            df["weight"] = 1.0
        elif self.asymmetric_garbage:
            # Asymmetric: only penalize TRAILING team's garbage time plays
            # Leading team keeps full weight - they earned the blowout through efficiency
            # Vectorized: weight = 1.0 unless (garbage_time AND offense trailing)
            offense_margin = df["offense_score"].values - df["defense_score"].values
            is_gt = df["is_garbage_time"].values
            is_trailing = offense_margin <= 0

            # Full weight unless in garbage time AND trailing
            df["weight"] = np.where(
                is_gt & is_trailing,
                self.garbage_time_weight,
                1.0
            )
        else:
            # Symmetric: weight ALL garbage time plays at reduced value
            df["weight"] = np.where(
                df["is_garbage_time"].values,
                self.garbage_time_weight,
                1.0
            )

        # Apply time decay if enabled (decay < 1.0)
        if self.time_decay < 1.0 and "week" in df.columns:
            # Use explicit max_week if provided (prevents data leakage),
            # otherwise fall back to df["week"].max()
            decay_reference_week = max_week if max_week is not None else df["week"].max()
            # Weight = decay ^ (reference_week - play_week)
            # Recent plays (reference_week) get weight 1.0, older plays get less
            df["time_weight"] = self.time_decay ** (decay_reference_week - df["week"])
            df["weight"] = df["weight"] * df["time_weight"]

        prep_time = time.time() - prep_start
        logger.debug(f"  Vectorized preprocessing: {prep_time*1000:.1f}ms for {len(df):,} plays")

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
            Tuple of (off_sr, def_sr, off_isoppp, def_isoppp) dicts
        """
        agg_start = time.time()

        # P3.5: Compute weighted success without copying - caller passes a copy from _prepare_plays()
        # Pre-compute weighted success column for efficiency (in-place assignment is safe here)
        plays_df["weighted_success"] = plays_df["is_success"].astype(float) * plays_df["weight"]

        # ===== OFFENSIVE METRICS (groupby offense) =====
        off_grouped = plays_df.groupby("offense").agg(
            play_count=("weight", "size"),
            weight_sum=("weight", "sum"),
            weighted_success_sum=("weighted_success", "sum"),
        )

        # Calculate weighted SR for teams with enough plays
        off_sr = {}
        for team in off_grouped.index:
            row = off_grouped.loc[team]
            if row["play_count"] >= self.MIN_PLAYS:
                off_sr[team] = row["weighted_success_sum"] / row["weight_sum"]
            else:
                off_sr[team] = self.LEAGUE_AVG_SUCCESS_RATE

        # ===== DEFENSIVE METRICS (groupby defense) =====
        def_grouped = plays_df.groupby("defense").agg(
            play_count=("weight", "size"),
            weight_sum=("weight", "sum"),
            weighted_success_sum=("weighted_success", "sum"),
        )

        def_sr = {}
        for team in def_grouped.index:
            row = def_grouped.loc[team]
            if row["play_count"] >= self.MIN_PLAYS:
                def_sr[team] = row["weighted_success_sum"] / row["weight_sum"]
            else:
                def_sr[team] = self.LEAGUE_AVG_SUCCESS_RATE

        # ===== ISOPPP (successful plays only, with valid PPA) =====
        off_isoppp = {}
        def_isoppp = {}

        if "ppa" in plays_df.columns:
            # Filter to successful plays with valid PPA
            successful_plays = plays_df[plays_df["is_success"] & plays_df["ppa"].notna()].copy()

            if len(successful_plays) > 0:
                successful_plays["weighted_ppa"] = successful_plays["ppa"] * successful_plays["weight"]

                # Offensive IsoPPP
                off_isoppp_grouped = successful_plays.groupby("offense").agg(
                    succ_count=("weight", "size"),
                    weight_sum=("weight", "sum"),
                    weighted_ppa_sum=("weighted_ppa", "sum"),
                )

                for team in off_isoppp_grouped.index:
                    row = off_isoppp_grouped.loc[team]
                    # Need at least 10 successful plays with valid PPA
                    if row["succ_count"] >= 10 and team in off_sr and off_grouped.loc[team, "play_count"] >= self.MIN_PLAYS:
                        off_isoppp[team] = row["weighted_ppa_sum"] / row["weight_sum"]
                    else:
                        off_isoppp[team] = self.LEAGUE_AVG_ISOPPP

                # Defensive IsoPPP
                def_isoppp_grouped = successful_plays.groupby("defense").agg(
                    succ_count=("weight", "size"),
                    weight_sum=("weight", "sum"),
                    weighted_ppa_sum=("weighted_ppa", "sum"),
                )

                for team in def_isoppp_grouped.index:
                    row = def_isoppp_grouped.loc[team]
                    if row["succ_count"] >= 10 and team in def_sr and def_grouped.loc[team, "play_count"] >= self.MIN_PLAYS:
                        def_isoppp[team] = row["weighted_ppa_sum"] / row["weight_sum"]
                    else:
                        def_isoppp[team] = self.LEAGUE_AVG_ISOPPP

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

        return off_sr, def_sr, off_isoppp, def_isoppp

    def _ridge_adjust_metric(
        self,
        plays_df: pd.DataFrame,
        metric_col: str,
        season: Optional[int] = None,
        eval_week: Optional[int] = None,
    ) -> tuple[dict[str, float], dict[str, float], Optional[float]]:
        """Use ridge regression to opponent-adjust a metric.

        Per spec: Y = metric value, X = sparse Team/Opponent IDs

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

        P3.1 OPTIMIZATION: Uses sparse CSR matrix for the design matrix.
        Each play has only 2-3 non-zero entries out of ~280 columns, so
        sparse representation reduces memory by ~99% and speeds up fitting.

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
            cache_key = (season, eval_week, metric_col, self.ridge_alpha, data_hash)

            if cache_key in _RIDGE_ADJUST_CACHE:
                _CACHE_STATS["hits"] += 1
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

        # P3.6: Use canonical team index if available (ensures consistent sparse matrix dimensions)
        # This is critical for IsoPPP adjustment which uses a subset of plays (successful only)
        # but still needs the same team dimensions as SR adjustment for consistency
        if self._canonical_teams and self._team_to_idx:
            all_teams = self._canonical_teams
            team_to_idx = self._team_to_idx
        else:
            all_teams = sorted(set(plays_df["offense"]) | set(plays_df["defense"]))
            team_to_idx = {team: i for i, team in enumerate(all_teams)}
        n_teams = len(all_teams)

        # Check if we have home team info for neutral-field regression
        has_home_info = "home_team" in plays_df.columns

        # P3.1: Build sparse design matrix using COO format (efficient for construction)
        # Columns: [off_team_0, ..., off_team_n, def_team_0, ..., def_team_n, (home_indicator)]
        # Each row has exactly 2-3 non-zero entries:
        #   - 1 for offense team (+1)
        #   - 1 for defense team (-1)
        #   - optionally 1 for home indicator (+1 or -1)
        n_cols = 2 * n_teams + (1 if has_home_info else 0)

        offenses = plays_df["offense"].values
        defenses = plays_df["defense"].values
        weights = plays_df["weight"].values if "weight" in plays_df.columns else np.ones(n_plays)

        if has_home_info:
            home_teams = plays_df["home_team"].values
            # P0.2: Validate home_team coverage (handle both None and NaN)
            home_valid = sum(1 for h in home_teams if pd.notna(h))
            home_pct = home_valid / n_plays * 100
            if home_pct < 90:
                logger.warning(
                    f"Home team coverage low: {home_valid}/{n_plays} ({home_pct:.1f}%). "
                    "Neutral-field regression may be unreliable."
                )
            else:
                logger.debug(f"Home team coverage: {home_valid}/{n_plays} ({home_pct:.1f}%)")
        else:
            home_teams = None

        # Pre-allocate arrays for COO sparse matrix
        # Max entries: 2 per row (offense + defense) + 1 per row for home (if applicable)
        max_nnz = 3 * n_plays if has_home_info else 2 * n_plays
        row_indices = np.empty(max_nnz, dtype=np.int32)
        col_indices = np.empty(max_nnz, dtype=np.int32)
        data_values = np.empty(max_nnz, dtype=np.float64)

        nnz = 0  # Number of non-zero entries

        for i in range(n_plays):
            off = offenses[i]
            def_ = defenses[i]
            off_idx = team_to_idx[off]
            def_idx = team_to_idx[def_]

            # Offense contributes positively
            row_indices[nnz] = i
            col_indices[nnz] = off_idx
            data_values[nnz] = 1.0
            nnz += 1

            # Defense reduces (good D = lower success)
            row_indices[nnz] = i
            col_indices[nnz] = n_teams + def_idx
            data_values[nnz] = -1.0
            nnz += 1

            # Home field indicator: +1 if offense is home, -1 if away, 0 if neutral
            # P0.2: Use pd.notna() to catch both None and np.nan
            if has_home_info and pd.notna(home_teams[i]):
                home = home_teams[i]
                if off == home:
                    row_indices[nnz] = i
                    col_indices[nnz] = n_cols - 1
                    data_values[nnz] = 1.0
                    nnz += 1
                elif def_ == home:
                    row_indices[nnz] = i
                    col_indices[nnz] = n_cols - 1
                    data_values[nnz] = -1.0
                    nnz += 1

        # Trim arrays to actual size and create sparse matrix
        row_indices = row_indices[:nnz]
        col_indices = col_indices[:nnz]
        data_values = data_values[:nnz]

        X_sparse = sparse.csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(n_plays, n_cols),
            dtype=np.float64
        )

        # Calculate memory usage for logging
        dense_memory_mb = (n_plays * n_cols * 8) / (1024 * 1024)  # 8 bytes per float64
        sparse_memory_mb = (X_sparse.data.nbytes + X_sparse.indices.nbytes +
                           X_sparse.indptr.nbytes) / (1024 * 1024)
        memory_savings_pct = (1 - sparse_memory_mb / dense_memory_mb) * 100

        build_time = time.time() - start_time

        # Target: the metric value (e.g., success = 1/0)
        if metric_col == "is_success":
            y = plays_df[metric_col].astype(float).values
        else:
            y = plays_df[metric_col].values

        # Fit weighted ridge regression (sklearn Ridge supports sparse matrices)
        fit_start = time.time()
        model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        model.fit(X_sparse, y, sample_weight=weights)
        fit_time = time.time() - fit_start

        # Extract coefficients
        coefficients = model.coef_
        intercept = model.intercept_

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
        logger.debug(
            f"  Sparse matrix: {n_plays:,} plays × {n_cols} cols, "
            f"{nnz:,} non-zeros ({100*nnz/(n_plays*n_cols):.2f}% density)"
        )
        logger.debug(
            f"  Memory: {sparse_memory_mb:.2f} MB sparse vs {dense_memory_mb:.2f} MB dense "
            f"({memory_savings_pct:.1f}% savings)"
        )
        logger.debug(
            f"  Time: {build_time*1000:.1f}ms build + {fit_time*1000:.1f}ms fit = "
            f"{total_time*1000:.1f}ms total"
        )

        off_adjusted = {}
        def_adjusted = {}

        for team, idx in team_to_idx.items():
            # Offensive rating: intercept + off_coef (now neutral-field)
            # Intercept = league average metric (e.g., ~0.42 for SR)
            # Team coef = deviation from average (positive = better than average)
            off_adjusted[team] = intercept + team_coefficients[idx]
            # Defensive rating: intercept - def_coef (negative because good D has negative coef)
            # Good defense has negative coef (reduces opponent success rate)
            # So intercept - def_coef gives higher value for good defenses
            def_adjusted[team] = intercept - team_coefficients[n_teams + idx]

        # P0.1: Validate ridge baseline interpretation via invariants
        # Mean of adjusted metrics should be close to intercept (league average)
        # This confirms that team coefficients represent deviations from baseline
        mean_off = np.mean(list(off_adjusted.values()))
        mean_def = np.mean(list(def_adjusted.values()))

        # Log deviations from expected baseline
        # For Success Rate: intercept ≈ 0.42, mean should match within ~0.01
        # For IsoPPP: intercept ≈ 0.30, mean should match within ~0.02
        off_baseline_error = abs(mean_off - intercept)
        def_baseline_error = abs(mean_def - intercept)

        logger.debug(
            f"  Ridge baseline check ({metric_col}): "
            f"intercept={intercept:.4f}, "
            f"mean_off={mean_off:.4f} (Δ={off_baseline_error:.4f}), "
            f"mean_def={mean_def:.4f} (Δ={def_baseline_error:.4f})"
        )

        # Warn if baseline drift is large (>5% of intercept magnitude)
        max_acceptable_drift = abs(intercept) * 0.05 if intercept != 0 else 0.05
        if off_baseline_error > max_acceptable_drift or def_baseline_error > max_acceptable_drift:
            logger.warning(
                f"Ridge baseline drift detected for {metric_col}: "
                f"mean_off and mean_def should equal intercept±{max_acceptable_drift:.4f}, "
                f"but deviations are off={off_baseline_error:.4f}, def={def_baseline_error:.4f}. "
                f"This may indicate numerical instability or data issues."
            )

        # =================================================================
        # CACHE STORAGE
        # =================================================================
        result = (off_adjusted, def_adjusted, learned_hfa)
        if cache_key is not None:
            _RIDGE_ADJUST_CACHE[cache_key] = result
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
        """Calculate per-game turnover stats for each team (P2.6: split O/D).

        Returns separate stats for ball security (turnovers lost) and
        takeaways (turnovers forced) to enable O/D-specific turnover ratings.

        Args:
            plays_df: Play-by-play data with 'play_type', 'offense', 'defense' columns
            games_df: Games data for counting games played

        Returns:
            Tuple of (lost_per_game, forced_per_game, margin_per_game) dicts
            - lost_per_game: Turnovers lost per game (lower = better ball security)
            - forced_per_game: Turnovers forced per game (higher = better takeaways)
            - margin_per_game: Net margin (forced - lost) for backward compat
        """
        empty_result = ({}, {}, {})

        if "play_type" not in plays_df.columns:
            logger.warning("No play_type column for turnover calculation")
            return empty_result

        # Find turnover plays
        turnover_plays = plays_df[plays_df["play_type"].isin(TURNOVER_PLAY_TYPES)]

        if len(turnover_plays) == 0:
            logger.warning("No turnover plays found")
            return empty_result

        # Count turnovers lost (offense = team that lost the ball)
        turnovers_lost = turnover_plays.groupby("offense").size()

        # Count turnovers forced (defense = team that forced it)
        turnovers_forced = turnover_plays.groupby("defense").size()

        # Get all teams - DETERMINISM: Sort for consistent iteration order
        all_teams = sorted(set(turnovers_lost.index) | set(turnovers_forced.index))

        # Count games per team - MUST have reliable count for shrinkage calculation
        # P2.10: No arbitrary defaults; compute from data or fail loudly
        games_played = {}
        games_source = None

        if games_df is not None and len(games_df) > 0:
            # Primary: count from games_df (most reliable)
            games_source = "games_df"
            for team in all_teams:
                n_games = len(games_df[
                    (games_df["home_team"] == team) | (games_df["away_team"] == team)
                ])
                games_played[team] = max(n_games, 1)
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

        for team in all_teams:
            lost = turnovers_lost.get(team, 0)
            forced = turnovers_forced.get(team, 0)
            games = games_played[team]

            lost_per_game[team] = lost / games
            forced_per_game[team] = forced / games
            margin_per_game[team] = (forced - lost) / games

        # Store games played for Bayesian shrinkage in calculate_ratings
        self.team_games_played = games_played

        # Log summary stats (P3.9: debug level for per-week logging)
        avg_lost = np.mean(list(lost_per_game.values()))
        avg_forced = np.mean(list(forced_per_game.values()))
        logger.debug(
            f"Calculated turnover stats for {len(all_teams)} teams: "
            f"avg lost={avg_lost:.2f}/game, avg forced={avg_forced:.2f}/game"
        )

        return lost_per_game, forced_per_game, margin_per_game

    def calculate_ratings(
        self,
        plays_df: pd.DataFrame,
        games_df: Optional[pd.DataFrame] = None,
        max_week: int | None = None,
        season: int | None = None,
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

        Returns:
            Dict mapping team name to TeamEFMRating
        """
        # Prepare plays (with data leakage guard if max_week provided)
        prepared = self._prepare_plays(plays_df, max_week=max_week)
        # P3.9: Debug level for per-week logging
        logger.debug(f"Prepared {len(prepared)} plays for EFM")

        # P3.6: Build canonical team index ONCE, reuse throughout pipeline
        # This eliminates redundant sorted(set(...) | set(...)) calls
        self._canonical_teams = sorted(set(prepared["offense"]) | set(prepared["defense"]))
        self._team_to_idx = {team: i for i, team in enumerate(self._canonical_teams)}
        logger.debug(f"Canonical team index: {len(self._canonical_teams)} teams")

        # Calculate raw metrics
        raw_off_sr, raw_def_sr, raw_off_isoppp, raw_def_isoppp = \
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

        # Calculate league averages from adjusted values
        # P3.6: np.mean() is order-independent, no need to sort for determinism
        avg_sr = np.mean(list(adj_off_sr.values()))
        valid_isoppp = [v for v in adj_off_isoppp.values() if v != self.LEAGUE_AVG_ISOPPP]
        avg_isoppp = np.mean(valid_isoppp) if valid_isoppp else self.LEAGUE_AVG_ISOPPP

        # Calculate league average turnover rates for O/D split (P2.6)
        # P3.6: np.mean() is order-independent, no need to sort
        avg_lost = np.mean(list(self.turnovers_lost.values())) if self.turnovers_lost else 0.0
        avg_forced = np.mean(list(self.turnovers_forced.values())) if self.turnovers_forced else 0.0

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

            # Explosiveness
            off_exp_pts = (off_iso - avg_isoppp) * self.ISOPPP_TO_POINTS
            def_exp_pts = (avg_isoppp - def_iso) * self.ISOPPP_TO_POINTS

            # Combine efficiency and explosiveness
            efficiency_rating = off_eff_pts + def_eff_pts
            explosiveness_rating = off_exp_pts + def_exp_pts

            # P2.6: Split turnovers into offensive (ball security) and defensive (takeaways)
            # Apply Bayesian shrinkage to each component separately
            # Shrinkage: games / (games + prior_strength). E.g., 15-game team keeps 60% of raw value.
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
            shrinkage = games / (games + self.turnover_prior_strength)

            # Offensive turnover: ball security (fewer lost = better)
            # Relative to average: (avg_lost - team_lost) * shrinkage * points_per_to
            # Positive when team loses fewer than average
            raw_lost = self.turnovers_lost.get(team, avg_lost)
            off_to_pts = (avg_lost - raw_lost) * shrinkage * POINTS_PER_TURNOVER

            # Defensive turnover: takeaways (more forced = better)
            # Relative to average: (team_forced - avg_forced) * shrinkage * points_per_to
            # Positive when team forces more than average
            raw_forced = self.turnovers_forced.get(team, avg_forced)
            def_to_pts = (raw_forced - avg_forced) * shrinkage * POINTS_PER_TURNOVER

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

            # Overall rating = O + D (turnovers now inside O/D, not separate)
            overall = offensive_rating + defensive_rating

            # Get sample sizes
            off_plays = len(prepared[prepared["offense"] == team])
            def_plays = len(prepared[prepared["defense"] == team])

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

        # Normalize ratings to target standard deviation
        # This ensures Team A rating - Team B rating = expected spread
        # Note: all_teams from CFBD API is FBS teams only
        self._normalize_ratings(all_teams)

        return self.team_ratings

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

            # Overall = off + def (P2.6: turnovers now inside O/D, not separate)
            new_overall = new_offense + new_defense

            # Update the rating object
            self.team_ratings[team] = TeamEFMRating(
                team=rating.team,
                raw_success_rate=rating.raw_success_rate,
                raw_isoppp=rating.raw_isoppp,
                adj_success_rate=rating.adj_success_rate,
                adj_isoppp=rating.adj_isoppp,
                efficiency_rating=new_efficiency,
                explosiveness_rating=new_explosiveness,
                offensive_rating=new_offense,
                defensive_rating=new_defense,
                special_teams_rating=rating.special_teams_rating,
                turnover_rating=new_turnover,
                overall_rating=new_overall,
                off_plays=rating.off_plays,
                def_plays=rating.def_plays,
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
            # Update the rating object
            old_rating = self.team_ratings[team]
            self.team_ratings[team] = TeamEFMRating(
                team=old_rating.team,
                raw_success_rate=old_rating.raw_success_rate,
                raw_isoppp=old_rating.raw_isoppp,
                adj_success_rate=old_rating.adj_success_rate,
                adj_isoppp=old_rating.adj_isoppp,
                efficiency_rating=old_rating.efficiency_rating,
                explosiveness_rating=old_rating.explosiveness_rating,
                offensive_rating=old_rating.offensive_rating,
                defensive_rating=old_rating.defensive_rating,
                special_teams_rating=rating,
                turnover_rating=old_rating.turnover_rating,
                overall_rating=old_rating.overall_rating,
                off_plays=old_rating.off_plays,
                def_plays=old_rating.def_plays,
            )

    def get_ratings_df(self) -> pd.DataFrame:
        """Get ratings as DataFrame sorted by overall rating.

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
