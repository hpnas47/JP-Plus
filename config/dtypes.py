"""Centralized dtype configuration for memory-efficient DataFrame operations.

P3.4: Explicit dtype specifications for all DataFrame columns to reduce memory usage
and improve cache efficiency.

DESIGN DECISIONS:
=================

1. CATEGORICAL COLUMNS (category dtype)
   - Team identifiers (home_team, away_team, offense, defense): ~130 FBS teams
   - Position groups: ~15 categories
   - Conferences: ~15 categories
   - Memory savings: ~8x for string columns with repeated values
   - Caveat: Category dtype uses integer codes internally, so equality comparisons
     work but string methods require .astype(str) first

2. SMALL INTEGERS (int8/int16)
   - week: 1-16 → int8 (range: -128 to 127)
   - season/year: 2015-2030 → int16 (range: -32768 to 32767)
   - down: 1-4 → int8
   - period/quarter: 1-5 → int8
   - Small counters (attempts, makes): int16 (rarely exceed 1000)
   - Memory savings: 4-8x vs int64

3. FLOAT32 vs FLOAT64
   - Ratings and normalized metrics: float32 (7 significant digits)
   - Raw EPA/PPA values: float32 (precision adequate for ~0.001 resolution)
   - Probability values: float32 (0.0-1.0 with adequate precision)
   - Sparse matrix data: Keep float64 (sklearn requirement)
   - Ridge regression outputs: Keep float64 (numerical stability)
   - Memory savings: 2x for float columns

4. BOOLEAN COLUMNS (bool)
   - Already optimal, no changes needed
   - Examples: neutral_site, is_success, is_garbage_time, made

5. COLUMNS TO KEEP AS-IS
   - game_id: int64 (large unique identifiers)
   - String columns that aren't repeated: object (player names)
   - Coordinates/timestamps: native types

IMPLEMENTATION NOTES:
====================
- Apply dtypes at DataFrame creation time where possible
- For Polars→Pandas conversion, apply dtypes after conversion
- Category columns must be created BEFORE filtering operations that might
  introduce new values not in the category set
- When merging DataFrames, ensure category columns have compatible categories
"""

import numpy as np
import pandas as pd

# =============================================================================
# DTYPE SPECIFICATIONS BY COLUMN NAME
# =============================================================================

# Categorical columns (team identifiers and labels)
CATEGORICAL_COLUMNS = {
    # Team identifiers
    "team",
    "home_team",
    "away_team",
    "offense",
    "defense",
    "origin",  # Transfer portal origin
    "destination",  # Transfer portal destination
    # Position/conference labels
    "pos_group",
    "position",
    "conference",
    "play_type",
}

# Small integer columns (int8: -128 to 127)
INT8_COLUMNS = {
    "week",
    "down",
    "period",
    "quarter",
}

# Medium integer columns (int16: -32768 to 32767)
INT16_COLUMNS = {
    "year",
    "season",
    "distance",  # Yards to go (0-99)
    "yards_to_goal",  # Field position (0-100)
    "stars",  # Recruit stars (2-5)
    "rating",  # Recruit rating (0-100)
    "attempts",
    "makes",
    "punt_count",
    "fg_attempts",
    "off_plays",
    "def_plays",
}

# Float32 columns (adequate precision for ratings/metrics)
FLOAT32_COLUMNS = {
    # Ratings (typically -40 to +40 range)
    "overall_rating",
    "offensive_rating",
    "defensive_rating",
    "efficiency_rating",
    "explosiveness_rating",
    "special_teams_rating",
    "field_goal_rating",
    "punt_rating",
    "kickoff_rating",
    "turnover_rating",
    "prior_rating",
    "talent_score",
    "combined_rating",
    # Normalized metrics (0-1 range)
    "adj_sr",
    "adj_isoppp",
    "raw_sr",
    "success_rate",
    "weight",
    "time_weight",
    # Point values and margins
    "predicted_spread",
    "actual_margin",
    "error",
    "abs_error",
    "home_margin",
    "expected_margin",
    "adjusted_margin",
    "residual",
    "paae",
    "punt_value",
    "net_yards",
    "gross_yards",
    "return_yards",
    # Adjustments
    "hfa",
    "travel",
    "altitude",
    "coaching_adjustment",
    "portal_adjustment",
    # Intermediate calculations
    "weighted_success",
    "weighted_ppa",
    "total_paae",
    "total_value",
}

# Columns to keep as float64 (numerical precision required)
FLOAT64_COLUMNS = {
    # Ridge regression I/O (sklearn requirement)
    "ppa",  # Raw EPA values used in ridge regression
    # Large accumulators where precision matters
    "home_points",
    "away_points",
    "offense_score",
    "defense_score",
    "yards_gained",
}

# Columns to keep as int64 (large identifiers)
INT64_COLUMNS = {
    "id",
    "game_id",
}


# =============================================================================
# DTYPE MAPPING DICTIONARY
# =============================================================================

def get_dtype_map() -> dict:
    """Get complete dtype mapping for all known columns.

    Returns:
        Dict mapping column names to numpy/pandas dtypes
    """
    dtype_map = {}

    for col in CATEGORICAL_COLUMNS:
        dtype_map[col] = "category"

    for col in INT8_COLUMNS:
        dtype_map[col] = np.int8

    for col in INT16_COLUMNS:
        dtype_map[col] = np.int16

    for col in FLOAT32_COLUMNS:
        dtype_map[col] = np.float32

    for col in FLOAT64_COLUMNS:
        dtype_map[col] = np.float64

    for col in INT64_COLUMNS:
        dtype_map[col] = np.int64

    return dtype_map


# Pre-computed dtype map for fast lookup
DTYPE_MAP = get_dtype_map()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def optimize_dtypes(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Apply optimal dtypes to a DataFrame based on column names.

    Args:
        df: DataFrame to optimize
        inplace: If True, modify df in place (default False)

    Returns:
        DataFrame with optimized dtypes
    """
    if not inplace:
        df = df.copy()

    for col in df.columns:
        if col in DTYPE_MAP:
            target_dtype = DTYPE_MAP[col]
            current_dtype = df[col].dtype

            # Skip if already correct dtype
            if current_dtype == target_dtype:
                continue

            # Handle category conversion
            if target_dtype == "category":
                df[col] = df[col].astype("category")

            # Handle numeric conversions with NaN handling
            elif target_dtype in (np.int8, np.int16, np.int64):
                if df[col].isna().any():
                    # Use nullable integer type for columns with NaN
                    nullable_map = {
                        np.int8: "Int8",
                        np.int16: "Int16",
                        np.int64: "Int64",
                    }
                    df[col] = df[col].astype(nullable_map[target_dtype])
                else:
                    df[col] = df[col].astype(target_dtype)

            # Handle float conversions
            elif target_dtype in (np.float32, np.float64):
                df[col] = df[col].astype(target_dtype)

    return df


def get_memory_usage(df: pd.DataFrame) -> dict:
    """Get memory usage breakdown by column.

    Args:
        df: DataFrame to analyze

    Returns:
        Dict with total_mb, per_column breakdown, and dtype summary
    """
    memory = df.memory_usage(deep=True)
    total_bytes = memory.sum()

    # Per-column breakdown
    per_column = {}
    for col in df.columns:
        col_bytes = memory[col]
        per_column[col] = {
            "bytes": col_bytes,
            "mb": col_bytes / (1024 * 1024),
            "dtype": str(df[col].dtype),
        }

    # Dtype summary
    dtype_totals = {}
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if dtype_str not in dtype_totals:
            dtype_totals[dtype_str] = 0
        dtype_totals[dtype_str] += memory[col]

    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "per_column": per_column,
        "by_dtype": {k: v / (1024 * 1024) for k, v in dtype_totals.items()},
    }


def estimate_savings(df: pd.DataFrame) -> dict:
    """Estimate memory savings from dtype optimization.

    Args:
        df: DataFrame to analyze

    Returns:
        Dict with current_mb, optimized_mb, savings_mb, savings_pct
    """
    current = get_memory_usage(df)
    optimized_df = optimize_dtypes(df)
    optimized = get_memory_usage(optimized_df)

    savings_mb = current["total_mb"] - optimized["total_mb"]
    savings_pct = (savings_mb / current["total_mb"]) * 100 if current["total_mb"] > 0 else 0

    return {
        "current_mb": current["total_mb"],
        "optimized_mb": optimized["total_mb"],
        "savings_mb": savings_mb,
        "savings_pct": savings_pct,
    }
