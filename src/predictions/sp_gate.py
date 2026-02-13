"""SP+ Agreement Gate for Phase 1 value plays.

This module implements a gating mechanism that uses SP+ pregame predictions
to filter JP+ value plays in weeks 1-3, reducing overconfident bets.

Sign Conventions:
- Internal model spread: positive = home favored
- SP+ raw (CFBD API): negative = home favored
- Vegas spread: negative = home favored
- Edge = model_spread_internal - (-vegas_open) = model_spread + vegas_open
  Positive edge = model likes home more than Vegas
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.api.cfbd_client import CFBDClient

logger = logging.getLogger(__name__)


class SPGateMode(Enum):
    """SP+ gating modes."""
    CONFIRM_ONLY = "confirm_only"  # Keep only where SP+ confirms
    VETO_OPPOSES = "veto_opposes"  # Keep confirms + neutral, reject opposes
    CONFIRM_OR_NEUTRAL = "confirm_or_neutral"  # Same as veto_opposes


class SPGateCategory(Enum):
    """SP+ agreement categories."""
    CONFIRMS = "confirms"  # Same side AND |edge_sp| >= threshold
    NEUTRAL = "neutral"  # Same side but |edge_sp| < threshold, or missing
    OPPOSES = "opposes"  # Opposite side
    MISSING = "missing"  # No SP+ prediction available


@dataclass
class SPGateConfig:
    """Configuration for SP+ gating."""
    enabled: bool = False
    sp_edge_min: float = 2.0  # Min SP+ edge for "confirms"
    jp_edge_min: float = 5.0  # Min JP+ edge for candidate bet
    mode: str = "confirm_only"

    @property
    def gate_mode(self) -> SPGateMode:
        return SPGateMode(self.mode)


@dataclass
class SPGateResult:
    """Result of SP+ gating for a single game."""
    game_id: int
    sp_spread_internal: Optional[float]  # Converted to internal convention
    sp_edge: Optional[float]  # vs Vegas
    jp_edge: float
    category: SPGateCategory
    passed_gate: bool


def fetch_sp_predictions(
    client: CFBDClient,
    year: int,
    weeks: list[int],
) -> dict[int, float]:
    """
    Fetch SP+ pregame predictions for specified weeks.

    Args:
        client: CFBD API client
        year: Season year
        weeks: List of week numbers

    Returns:
        Dict mapping game_id to SP+ spread (internal convention: positive = home favored)
    """
    sp_spreads = {}

    for week in weeks:
        try:
            probs = client.get_pregame_win_probabilities(year, week=week)
            for p in probs:
                if p.spread is not None:
                    # Convert from Vegas convention (negative = home favored)
                    # to internal convention (positive = home favored)
                    sp_spreads[p.game_id] = -p.spread
        except Exception as e:
            logger.warning(f"Could not fetch SP+ predictions for {year} week {week}: {e}")

    logger.info(f"Fetched {len(sp_spreads)} SP+ predictions for weeks {weeks}")
    return sp_spreads


def compute_sp_edge(sp_spread_internal: float, vegas_open: float) -> float:
    """
    Compute SP+ edge vs Vegas.

    Uses same convention as VegasComparison:
    edge = model_spread_internal - (-vegas_open) = model_spread + vegas_open

    Args:
        sp_spread_internal: SP+ spread (positive = home favored)
        vegas_open: Vegas opening line (negative = home favored)

    Returns:
        Edge in points (positive = model likes home more than Vegas)
    """
    return sp_spread_internal + vegas_open


def categorize_sp_agreement(
    jp_edge: float,
    sp_edge: Optional[float],
    sp_edge_min: float,
) -> SPGateCategory:
    """
    Categorize SP+ agreement with JP+.

    Args:
        jp_edge: JP+ edge vs Vegas
        sp_edge: SP+ edge vs Vegas (or None if missing)
        sp_edge_min: Minimum |sp_edge| for "confirms"

    Returns:
        SPGateCategory enum value
    """
    if sp_edge is None:
        return SPGateCategory.MISSING

    jp_side = np.sign(jp_edge)
    sp_side = np.sign(sp_edge)

    if jp_side == 0:
        # JP+ has no edge, treat SP as neutral
        return SPGateCategory.NEUTRAL

    if sp_side == 0:
        # SP+ has no edge, treat as neutral
        return SPGateCategory.NEUTRAL

    if jp_side == sp_side:
        # Same side
        if abs(sp_edge) >= sp_edge_min:
            return SPGateCategory.CONFIRMS
        else:
            return SPGateCategory.NEUTRAL
    else:
        # Opposite sides
        return SPGateCategory.OPPOSES


def gate_passes(category: SPGateCategory, mode: SPGateMode) -> bool:
    """
    Determine if a bet passes the gate.

    Args:
        category: SP+ agreement category
        mode: Gating mode

    Returns:
        True if bet should be kept, False if filtered
    """
    if mode == SPGateMode.CONFIRM_ONLY:
        return category == SPGateCategory.CONFIRMS

    elif mode in (SPGateMode.VETO_OPPOSES, SPGateMode.CONFIRM_OR_NEUTRAL):
        # Keep confirms + neutral + missing, reject opposes
        return category != SPGateCategory.OPPOSES

    return True  # Unknown mode, don't filter


def apply_sp_gate(
    value_plays_df: pd.DataFrame,
    sp_spreads: dict[int, float],
    vegas_opens: dict[int, float],
    config: SPGateConfig,
) -> pd.DataFrame:
    """
    Apply SP+ gating to value plays DataFrame.

    Args:
        value_plays_df: DataFrame with value plays (must have game_id, edge_signed or edge)
        sp_spreads: Dict mapping game_id to SP+ spread (internal convention)
        vegas_opens: Dict mapping game_id to Vegas opening line
        config: SP+ gate configuration

    Returns:
        Filtered DataFrame with additional columns:
        - sp_spread: SP+ spread (internal convention)
        - sp_edge: SP+ edge vs Vegas
        - sp_gate_category: SPGateCategory string
        - phase1_sp_gate_passed: bool
    """
    if not config.enabled:
        # Gate disabled, return original with placeholder columns
        df = value_plays_df.copy()
        df['sp_spread'] = None
        df['sp_edge'] = None
        df['sp_gate_category'] = None
        df['phase1_sp_gate_passed'] = True
        return df

    mode = config.gate_mode

    results = []
    for _, row in value_plays_df.iterrows():
        game_id = row['game_id']

        # Get JP+ edge (try edge_signed first, then edge)
        if 'edge_signed' in row:
            jp_edge = row['edge_signed']
        elif 'edge_recommended' in row:
            jp_edge = row['edge_recommended']
        else:
            # Compute from model_spread and vegas_spread
            jp_edge = row.get('model_spread', 0) + row.get('vegas_spread', 0)

        # Get SP+ spread and edge
        sp_spread = sp_spreads.get(game_id)
        vegas_open = vegas_opens.get(game_id)

        sp_edge = None
        if sp_spread is not None and vegas_open is not None:
            sp_edge = compute_sp_edge(sp_spread, vegas_open)

        # Categorize
        category = categorize_sp_agreement(jp_edge, sp_edge, config.sp_edge_min)

        # Apply gate
        passes = gate_passes(category, mode)

        results.append({
            'game_id': game_id,
            'sp_spread': sp_spread,
            'sp_edge': sp_edge,
            'sp_gate_category': category.value,
            'phase1_sp_gate_passed': passes,
        })

    # Create results DataFrame and merge
    gate_df = pd.DataFrame(results)
    df = value_plays_df.merge(gate_df, on='game_id', how='left')

    # Log summary
    n_total = len(df)
    n_confirms = (df['sp_gate_category'] == SPGateCategory.CONFIRMS.value).sum()
    n_neutral = (df['sp_gate_category'] == SPGateCategory.NEUTRAL.value).sum()
    n_opposes = (df['sp_gate_category'] == SPGateCategory.OPPOSES.value).sum()
    n_missing = (df['sp_gate_category'] == SPGateCategory.MISSING.value).sum()
    n_passed = df['phase1_sp_gate_passed'].sum()

    logger.info(
        f"SP+ Gate ({mode.value}): {n_total} candidates → "
        f"confirms={n_confirms}, neutral={n_neutral}, opposes={n_opposes}, missing={n_missing} → "
        f"{n_passed} passed"
    )

    # Filter based on gate
    df = df[df['phase1_sp_gate_passed'] == True].copy()

    return df.reset_index(drop=True)


def apply_sp_gate_to_comparison_df(
    comparison_df: pd.DataFrame,
    sp_spreads: dict[int, float],
    config: SPGateConfig,
) -> pd.DataFrame:
    """
    Apply SP+ gating annotations to full comparison DataFrame.

    This adds SP+ columns to ALL games (not just value plays) for reporting.

    Args:
        comparison_df: Full comparison DataFrame with all predictions
        sp_spreads: Dict mapping game_id to SP+ spread (internal convention)
        config: SP+ gate configuration

    Returns:
        DataFrame with additional columns:
        - sp_spread: SP+ spread (internal convention)
        - sp_edge: SP+ edge vs Vegas
        - sp_gate_category: Category string
        - phase1_sp_gate_passed: bool (based on config.mode)
    """
    df = comparison_df.copy()

    # Add SP+ spread
    df['sp_spread'] = df['game_id'].map(sp_spreads)

    # Compute SP+ edge using vegas_open if available, else vegas_spread
    vegas_col = 'vegas_open' if 'vegas_open' in df.columns else 'vegas_spread'

    def get_sp_edge(row):
        sp = row.get('sp_spread')
        vegas = row.get(vegas_col)
        if pd.isna(sp) or pd.isna(vegas):
            return None
        return sp + vegas

    df['sp_edge'] = df.apply(get_sp_edge, axis=1)

    # Compute SP+ edge abs
    df['sp_edge_abs'] = df['sp_edge'].abs()

    # Get JP+ edge
    edge_col = 'edge_recommended' if 'edge_recommended' in df.columns else 'edge'

    # Categorize and check gate
    def categorize_row(row):
        jp_edge = row.get(edge_col, 0)
        sp_edge = row.get('sp_edge')
        return categorize_sp_agreement(jp_edge, sp_edge, config.sp_edge_min).value

    df['sp_gate_category'] = df.apply(categorize_row, axis=1)

    mode = config.gate_mode

    def check_gate(row):
        cat = SPGateCategory(row['sp_gate_category'])
        return gate_passes(cat, mode)

    df['phase1_sp_gate_passed'] = df.apply(check_gate, axis=1)

    return df
