"""Rating normalization utilities.

Provides functions to normalize team ratings to PBTA (Points Better Than Average)
convention with proper scaling for each component.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_ratings(
    df: pd.DataFrame,
    off_col: str,
    def_col: str,
    st_col: str,
    team_col: str = "team",
    fbs_teams: Optional[set] = None,
    target_total_std: float = 13.5,
    target_st_std: float = 1.5,
    defense_is_points_allowed: bool = False,
) -> pd.DataFrame:
    """
    Normalize ratings to PBTA (Points Better Than Average) convention.

    Sign Convention:
    - Offense: Positive (+) = Scores more points than average (good)
    - Defense: Positive (+) = Prevents more points than average (good)
    - ST: Positive (+) = Better special teams than average (good)
    - Total = Offense + Defense (ST displayed separately)

    Zero-Center Logic:
    - Calculate mean for each component using only FBS teams
    - Shift all ratings so FBS average is exactly 0.0

    Scaling:
    - Total (Off + Def) scaled to target_total_std (default 13.5)
    - ST scaled separately to target_st_std (default 1.5)
    - This keeps ST impact realistic (~±3 pts covers 95% of teams)

    Args:
        df: DataFrame with team ratings
        off_col: Column name for raw offense rating
        def_col: Column name for raw defense rating
        st_col: Column name for raw special teams rating
        team_col: Column name for team identifier
        fbs_teams: Set of FBS team names (for zero-centering). If None, uses all teams.
        target_total_std: Target standard deviation for Total rating (~13.5 for SP+ scale)
        target_st_std: Target standard deviation for ST rating (~1.5 for realistic impact)
        defense_is_points_allowed: If True, invert defense sign (lower = better -> higher = better)

    Returns:
        DataFrame with new columns: norm_off, norm_def, norm_st, norm_total

    Example:
        >>> normalized = normalize_ratings(
        ...     ratings_df,
        ...     off_col='raw_off',
        ...     def_col='raw_def',
        ...     st_col='raw_st',
        ...     fbs_teams=fbs_team_set,
        ...     target_total_std=13.5,
        ...     target_st_std=1.5,
        ... )
        >>> # Top teams will be ~+30, bottom teams ~-30
        >>> # Total = norm_off + norm_def
        >>> # ST shown separately with realistic ±3 pt range
    """
    df = df.copy()

    # Extract raw values
    raw_off = df[off_col].values.copy()
    raw_def = df[def_col].values.copy()
    raw_st = df[st_col].values.copy()

    # Invert defense if it's "points allowed" convention (lower = better)
    if defense_is_points_allowed:
        raw_def = -raw_def
        logger.info("Inverted defense sign (was points allowed, now points prevented)")

    # Identify FBS teams mask
    if fbs_teams is not None:
        fbs_mask = df[team_col].isin(fbs_teams)
    else:
        fbs_mask = np.ones(len(df), dtype=bool)

    fbs_count = fbs_mask.sum()
    logger.debug(f"Normalizing over {fbs_count} teams")

    # Step 1: Center O and D by FBS means
    fbs_off_mean = raw_off[fbs_mask].mean()
    fbs_def_mean = raw_def[fbs_mask].mean()
    centered_off = raw_off - fbs_off_mean
    centered_def = raw_def - fbs_def_mean

    # Step 2: Calculate base total (O + D only)
    base_total = centered_off + centered_def

    # Step 3: Scale O and D to target total std
    current_total_std = base_total[fbs_mask].std()
    if current_total_std == 0:
        od_scale = 1.0
    else:
        od_scale = target_total_std / current_total_std

    norm_off = centered_off * od_scale
    norm_def = centered_def * od_scale
    norm_total = norm_off + norm_def

    # Step 4: Scale ST separately to realistic range
    fbs_st_mean = raw_st[fbs_mask].mean()
    centered_st = raw_st - fbs_st_mean
    current_st_std = centered_st[fbs_mask].std()
    if current_st_std == 0:
        st_scale = 1.0
    else:
        st_scale = target_st_std / current_st_std

    norm_st = centered_st * st_scale

    # Add normalized columns
    df["norm_off"] = norm_off
    df["norm_def"] = norm_def
    df["norm_st"] = norm_st
    df["norm_total"] = norm_total

    # Log summary stats
    logger.info(
        f"Normalization complete: "
        f"O/D scale={od_scale:.3f}x, ST scale={st_scale:.3f}x, "
        f"Total std={norm_total[fbs_mask].std():.2f}, "
        f"Total range=[{norm_total[fbs_mask].min():.1f}, {norm_total[fbs_mask].max():.1f}]"
    )

    return df


def verify_normalization(
    df: pd.DataFrame,
    team_col: str = "team",
    fbs_teams: Optional[set] = None,
) -> dict:
    """
    Verify that normalization was applied correctly.

    Args:
        df: DataFrame with normalized columns (norm_off, norm_def, norm_st, norm_total)
        team_col: Column name for team identifier
        fbs_teams: Set of FBS team names

    Returns:
        Dict with verification results
    """
    if fbs_teams is not None:
        fbs_mask = df[team_col].isin(fbs_teams)
    else:
        fbs_mask = np.ones(len(df), dtype=bool)

    results = {
        "off_mean": df.loc[fbs_mask, "norm_off"].mean(),
        "def_mean": df.loc[fbs_mask, "norm_def"].mean(),
        "st_mean": df.loc[fbs_mask, "norm_st"].mean(),
        "total_mean": df.loc[fbs_mask, "norm_total"].mean(),
        "off_std": df.loc[fbs_mask, "norm_off"].std(),
        "def_std": df.loc[fbs_mask, "norm_def"].std(),
        "st_std": df.loc[fbs_mask, "norm_st"].std(),
        "total_std": df.loc[fbs_mask, "norm_total"].std(),
        "total_min": df.loc[fbs_mask, "norm_total"].min(),
        "total_max": df.loc[fbs_mask, "norm_total"].max(),
        "relationship_error": (
            df["norm_total"] - (df["norm_off"] + df["norm_def"])
        ).abs().max(),
    }

    # Check if all means are ~0
    results["means_centered"] = all(
        abs(results[f"{c}_mean"]) < 0.001 for c in ["off", "def", "st", "total"]
    )

    # Check if Total = Off + Def
    results["relationship_valid"] = results["relationship_error"] < 1e-6

    return results
