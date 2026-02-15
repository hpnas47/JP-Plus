"""Selection policy system for spread EV betting.

This module provides configurable bet selection policies that determine which
bets to recommend from the pool of EV-positive candidates.

EV Units Contract
-----------------
All ``ev`` / ``ev_col`` values consumed by this module MUST be **expected
profit per unit staked** (ROI units), consistent with the engine formula::

    EV = p_win * b - (1 - p_win)

where *b* is the decimal payout minus 1 (e.g., +100 American → b = 1.0).
A value of 0.03 means "+3% expected return per bet", NOT "3 percentage-point
probability edge over breakeven."  All thresholds (``ev_min``, ``ev_floor``)
are in these same ROI units.

Policy Behavior:
- EV_THRESHOLD: Keep all bets with EV >= ev_min (ROI units), then cap per week
- TOP_N_PER_WEEK: Keep top N by EV per week, optionally filter by ev_floor (ROI units)
- HYBRID: Keep candidates >= ev_floor (ROI units), top N per week, with hard cap

All policies enforce:
- One bet per game/event (already enforced upstream)
- Deterministic sort order: EV desc, then edge_abs desc, then game_id asc
- Max bets per week cap after selection
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SelectionPolicy(str, Enum):
    """Bet selection policy types."""
    EV_THRESHOLD = "EV_THRESHOLD"
    TOP_N_PER_WEEK = "TOP_N_PER_WEEK"
    HYBRID = "HYBRID"


@dataclass
class SelectionPolicyConfig:
    """Configuration for spread bet selection policy.

    Attributes:
        selection_policy: Which selection algorithm to use
        ev_min: Minimum EV (ROI units) for EV_THRESHOLD policy (default 0.03 = +3% ROI)
        ev_floor: Minimum EV (ROI units) for TOP_N_PER_WEEK and HYBRID (default 0.0)
        top_n_per_week: Number of top bets per week (default 10)
        max_bets_per_week: Hard cap on bets per week (default 10)
        phase1_policy: How to handle weeks 0-3 ("skip", "apply", default "skip")
        phase2_weeks: Tuple of (start, end) for Phase 2 (default (4, 15))

    Tie-breaking order (deterministic):
        1. EV descending (highest EV first)
        2. edge_abs descending (largest edge as tiebreaker)
        3. game_id ascending (alphabetical for final determinism)
    """

    selection_policy: Literal["EV_THRESHOLD", "TOP_N_PER_WEEK", "HYBRID"] = "EV_THRESHOLD"
    ev_min: float = 0.03  # 3% EV minimum for EV_THRESHOLD
    ev_floor: float = 0.0  # Floor for TOP_N and HYBRID (0 = no floor beyond 0%)
    top_n_per_week: int = 10  # Top N per week for TOP_N and HYBRID
    max_bets_per_week: int = 10  # Hard cap for all policies
    phase1_policy: str = "skip"  # "skip" or "apply" for weeks 0-3
    phase2_weeks: tuple[int, int] = (4, 15)

    def __post_init__(self):
        """Validate configuration."""
        if self.ev_min < 0:
            raise ValueError(f"ev_min must be >= 0, got {self.ev_min}")
        if self.ev_floor < 0:
            raise ValueError(f"ev_floor must be >= 0, got {self.ev_floor}")
        if self.top_n_per_week < 1:
            raise ValueError(f"top_n_per_week must be >= 1, got {self.top_n_per_week}")
        if self.max_bets_per_week < 1:
            raise ValueError(f"max_bets_per_week must be >= 1, got {self.max_bets_per_week}")
        if self.selection_policy not in ("EV_THRESHOLD", "TOP_N_PER_WEEK", "HYBRID"):
            raise ValueError(f"Unknown policy: {self.selection_policy}")


@dataclass
class SelectionResult:
    """Result of applying selection policy.

    Attributes:
        selected_bets: DataFrame of selected bets
        n_candidates: Number of candidates before any filtering
        n_selected: Number of bets selected
        n_phase1_skipped: Number excluded by phase1_policy="skip" (weeks 0-3)
        n_filtered_by_ev: Number filtered by EV threshold/floor (after phase1 skip)
        n_filtered_by_cap: Number removed by weekly cap (after EV filtering)
        bets_per_week: Dict of (year, week) -> bet count
        config: The config used for selection
    """

    selected_bets: pd.DataFrame
    n_candidates: int
    n_selected: int
    n_phase1_skipped: int
    n_filtered_by_ev: int
    n_filtered_by_cap: int
    bets_per_week: dict
    config: SelectionPolicyConfig


def _sort_deterministic(df: pd.DataFrame, ev_col: str = "ev") -> pd.DataFrame:
    """Sort candidates deterministically for tie-breaking.

    Order: EV desc, edge_abs desc, game_id asc

    Args:
        df: DataFrame with candidates
        ev_col: Name of EV column (default "ev")

    Returns:
        Sorted DataFrame (copy)
    """
    if len(df) == 0:
        return df.copy()

    # Create sort keys
    sort_cols = []
    ascending = []

    # Primary: EV descending
    if ev_col in df.columns:
        sort_cols.append(ev_col)
        ascending.append(False)

    # Secondary: edge_abs descending
    if "edge_abs" in df.columns:
        sort_cols.append("edge_abs")
        ascending.append(False)

    # Tertiary: game_id ascending for final determinism
    if "game_id" in df.columns:
        sort_cols.append("game_id")
        ascending.append(True)

    if sort_cols:
        return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return df.copy()


def apply_selection_policy(
    candidates: pd.DataFrame,
    config: SelectionPolicyConfig,
    ev_col: str = "ev",
    year_col: str = "year",
    week_col: str = "week",
) -> SelectionResult:
    """Apply selection policy to candidate bets.

    The ``ev_col`` column must contain EV in ROI units (expected profit per
    unit staked), **not** probability edge vs breakeven.  See module docstring.

    Args:
        candidates: DataFrame with candidate bets (must have ev, year, week, game_id, edge_abs)
        config: SelectionPolicyConfig with policy parameters
        ev_col: Name of EV column (must be ROI units)
        year_col: Name of year column
        week_col: Name of week column

    Returns:
        SelectionResult with selected bets and accurate accounting

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = [ev_col, year_col, week_col, "game_id"]
    missing = set(required_cols) - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(candidates) == 0:
        return SelectionResult(
            selected_bets=candidates.copy(),
            n_candidates=0,
            n_selected=0,
            n_phase1_skipped=0,
            n_filtered_by_ev=0,
            n_filtered_by_cap=0,
            bets_per_week={},
            config=config,
        )

    # Start with a copy
    df = candidates.copy()
    n_candidates = len(df)

    # EV units sanity check: if thresholds are in typical ROI range but
    # the EV distribution looks like probability-edge, warn the user.
    ev_threshold = max(config.ev_min, config.ev_floor)
    if ev_threshold <= 0.05 and len(df) > 10:
        ev_vals = df[ev_col].dropna()
        if len(ev_vals) > 0 and ev_vals.abs().max() < 0.005:
            logger.warning(
                "EV values appear very small (max |ev| < 0.005) while thresholds "
                "are in ROI range (ev_min=%.4f, ev_floor=%.4f). Ensure ev_col "
                "contains ROI units (p_win*b - p_lose), not probability edge "
                "(p_win - p_breakeven).",
                config.ev_min, config.ev_floor,
            )

    # Phase 1 handling: optionally filter out weeks 0-3
    n_phase1_skipped = 0
    if config.phase1_policy == "skip":
        phase1_mask = df[week_col] <= 3
        n_phase1_skipped = int(phase1_mask.sum())
        if n_phase1_skipped > 0:
            logger.debug(f"Skipping {n_phase1_skipped} Phase 1 candidates (weeks 0-3)")
            df = df[~phase1_mask].copy()

    # Apply policy-specific selection; each returns (selected_df, n_ev_filtered, n_cap_filtered)
    policy = config.selection_policy

    if policy == "EV_THRESHOLD":
        selected, n_filtered_by_ev, n_filtered_by_cap = _apply_ev_threshold(df, config, ev_col, year_col, week_col)
    elif policy == "TOP_N_PER_WEEK":
        selected, n_filtered_by_ev, n_filtered_by_cap = _apply_top_n_per_week(df, config, ev_col, year_col, week_col)
    elif policy == "HYBRID":
        selected, n_filtered_by_ev, n_filtered_by_cap = _apply_hybrid(df, config, ev_col, year_col, week_col)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    n_selected = len(selected)

    # Compute bets per week
    if n_selected > 0:
        bets_per_week = selected.groupby([year_col, week_col]).size().to_dict()
    else:
        bets_per_week = {}

    return SelectionResult(
        selected_bets=selected,
        n_candidates=n_candidates,
        n_selected=n_selected,
        n_phase1_skipped=n_phase1_skipped,
        n_filtered_by_ev=n_filtered_by_ev,
        n_filtered_by_cap=n_filtered_by_cap,
        bets_per_week=bets_per_week,
        config=config,
    )


def _apply_ev_threshold(
    df: pd.DataFrame,
    config: SelectionPolicyConfig,
    ev_col: str,
    year_col: str,
    week_col: str,
) -> tuple[pd.DataFrame, int, int]:
    """EV_THRESHOLD policy: keep EV >= ev_min (ROI units), then cap per week.

    Returns:
        (selected_df, n_filtered_by_ev, n_filtered_by_cap)
    """
    # Filter by ev_min
    mask = df[ev_col] >= config.ev_min
    filtered = df[mask].copy()
    n_filtered_by_ev = int((~mask).sum())

    if len(filtered) == 0:
        return filtered, n_filtered_by_ev, 0

    # Sort deterministically
    filtered = _sort_deterministic(filtered, ev_col)

    # Apply max_bets_per_week cap
    n_filtered_by_cap = 0
    result_rows = []
    for (year, week), group in filtered.groupby([year_col, week_col]):
        group_sorted = _sort_deterministic(group, ev_col)
        capped = group_sorted.head(config.max_bets_per_week)
        n_filtered_by_cap += len(group_sorted) - len(capped)
        result_rows.append(capped)

    if result_rows:
        result = pd.concat(result_rows, ignore_index=True)
        return _sort_deterministic(result, ev_col), n_filtered_by_ev, n_filtered_by_cap
    return pd.DataFrame(columns=filtered.columns), n_filtered_by_ev, n_filtered_by_cap


def _apply_top_n_per_week(
    df: pd.DataFrame,
    config: SelectionPolicyConfig,
    ev_col: str,
    year_col: str,
    week_col: str,
) -> tuple[pd.DataFrame, int, int]:
    """TOP_N_PER_WEEK policy: top N by EV per week, optionally filtered by ev_floor (ROI units).

    Returns:
        (selected_df, n_filtered_by_ev, n_filtered_by_cap)
    """
    if len(df) == 0:
        return df.copy(), 0, 0

    n_filtered_by_ev = 0
    n_filtered_by_cap = 0
    result_rows = []

    for (year, week), group in df.groupby([year_col, week_col]):
        group_sorted = _sort_deterministic(group, ev_col)

        # Take top N (excess beyond top_n is cap-filtered)
        top_n = group_sorted.head(config.top_n_per_week)
        n_filtered_by_cap += len(group_sorted) - len(top_n)

        # Filter by ev_floor
        if config.ev_floor > 0:
            before_floor = len(top_n)
            top_n = top_n[top_n[ev_col] >= config.ev_floor]
            n_filtered_by_ev += before_floor - len(top_n)

        # Apply max_bets_per_week (redundant if >= top_n_per_week)
        before_cap = len(top_n)
        top_n = top_n.head(config.max_bets_per_week)
        n_filtered_by_cap += before_cap - len(top_n)

        if len(top_n) > 0:
            result_rows.append(top_n)

    if result_rows:
        result = pd.concat(result_rows, ignore_index=True)
        return _sort_deterministic(result, ev_col), n_filtered_by_ev, n_filtered_by_cap
    return pd.DataFrame(columns=df.columns), n_filtered_by_ev, n_filtered_by_cap


def _apply_hybrid(
    df: pd.DataFrame,
    config: SelectionPolicyConfig,
    ev_col: str,
    year_col: str,
    week_col: str,
) -> tuple[pd.DataFrame, int, int]:
    """HYBRID policy: take candidates >= ev_floor (ROI units), top N per week, with hard cap.

    Key difference from TOP_N: filters BEFORE taking top N (not after).

    Returns:
        (selected_df, n_filtered_by_ev, n_filtered_by_cap)
    """
    if len(df) == 0:
        return df.copy(), 0, 0

    n_filtered_by_ev = 0
    n_filtered_by_cap = 0
    result_rows = []

    for (year, week), group in df.groupby([year_col, week_col]):
        # First filter by ev_floor
        if config.ev_floor > 0:
            before_floor = len(group)
            group = group[group[ev_col] >= config.ev_floor]
            n_filtered_by_ev += before_floor - len(group)

        if len(group) == 0:
            continue

        group_sorted = _sort_deterministic(group, ev_col)

        # Take top N, then apply hard cap
        effective_cap = min(config.top_n_per_week, config.max_bets_per_week)
        top_n = group_sorted.head(effective_cap)
        n_filtered_by_cap += len(group_sorted) - len(top_n)

        if len(top_n) > 0:
            result_rows.append(top_n)

    if result_rows:
        result = pd.concat(result_rows, ignore_index=True)
        return _sort_deterministic(result, ev_col), n_filtered_by_ev, n_filtered_by_cap
    return pd.DataFrame(columns=df.columns), n_filtered_by_ev, n_filtered_by_cap


# =============================================================================
# EVALUATION METRICS
# =============================================================================

@dataclass
class SelectionMetrics:
    """Metrics for evaluating a selection policy result.

    Attributes:
        n_bets: Total number of bets
        n_wins: Number of wins (jp_side_covered)
        n_losses: Number of losses
        n_pushes: Number of pushes
        ats_pct: ATS win percentage (wins / (wins + losses))
        roi_pct: ROI assuming -110 juice
        avg_ev: Average EV of selected bets
        avg_edge: Average absolute edge

        bets_per_week_mean: Mean bets per week
        bets_per_week_p50: Median bets per week
        bets_per_week_p90: 90th percentile bets per week
        bets_per_week_std: Std dev of bets per week

        bets_per_year: Dict of year -> total bets
        years_with_data: List of years with bets
    """

    n_bets: int
    n_wins: int
    n_losses: int
    n_pushes: int
    ats_pct: float
    roi_pct: float
    avg_ev: float
    avg_edge: float

    bets_per_week_mean: float
    bets_per_week_p50: float
    bets_per_week_p90: float
    bets_per_week_std: float

    bets_per_year: dict
    years_with_data: list


def compute_selection_metrics(
    selected_bets: pd.DataFrame,
    ev_col: str = "ev",
    outcome_col: str = "jp_side_covered",
    push_col: str = "push",
    year_col: str = "year",
    week_col: str = "week",
    juice: int = -110,
) -> SelectionMetrics:
    """Compute evaluation metrics for selected bets.

    Args:
        selected_bets: DataFrame of selected bets
        ev_col: EV column name
        outcome_col: Win/loss outcome column (bool)
        push_col: Push indicator column (bool)
        year_col, week_col: Time columns
        juice: American odds for ROI calculation

    Returns:
        SelectionMetrics with all computed values
    """
    if len(selected_bets) == 0:
        return SelectionMetrics(
            n_bets=0,
            n_wins=0,
            n_losses=0,
            n_pushes=0,
            ats_pct=0.0,
            roi_pct=0.0,
            avg_ev=0.0,
            avg_edge=0.0,
            bets_per_week_mean=0.0,
            bets_per_week_p50=0.0,
            bets_per_week_p90=0.0,
            bets_per_week_std=0.0,
            bets_per_year={},
            years_with_data=[],
        )

    df = selected_bets.copy()

    # Ensure push column exists
    if push_col not in df.columns:
        df[push_col] = False

    # Guard: filter out rows with NaN outcomes (unsettled bets)
    n_total = len(df)
    if outcome_col in df.columns:
        settled_mask = df[outcome_col].notna() & df[push_col].notna()
        n_unsettled = int((~settled_mask).sum())
        if n_unsettled > 0:
            logger.warning(
                "Dropping %d rows with NaN in %s/%s (unsettled bets)",
                n_unsettled, outcome_col, push_col,
            )
            df = df[settled_mask].copy()

    # Basic counts
    n_bets = len(df)
    n_pushes = int(df[push_col].sum()) if push_col in df.columns else 0
    n_wins = int(df[~df[push_col]][outcome_col].sum()) if outcome_col in df.columns else 0
    n_losses = n_bets - n_pushes - n_wins

    # ATS percentage (exclude pushes)
    decided = n_wins + n_losses
    ats_pct = (n_wins / decided * 100) if decided > 0 else 0.0

    # ROI assuming -110 juice
    # Win: +100/110, Loss: -1
    if juice < 0:
        payout = 100 / abs(juice)
    else:
        payout = juice / 100
    profit = n_wins * payout - n_losses * 1.0
    roi_pct = (profit / n_bets * 100) if n_bets > 0 else 0.0

    # Average EV and edge
    avg_ev = df[ev_col].mean() if ev_col in df.columns else 0.0
    avg_edge = df["edge_abs"].mean() if "edge_abs" in df.columns else 0.0

    # Bets per week distribution
    weekly_counts = df.groupby([year_col, week_col]).size()
    if len(weekly_counts) > 0:
        bets_per_week_mean = float(weekly_counts.mean())
        bets_per_week_p50 = float(weekly_counts.median())
        bets_per_week_p90 = float(weekly_counts.quantile(0.9))
        # std() returns NaN if only one group - handle gracefully
        std_val = weekly_counts.std()
        bets_per_week_std = 0.0 if (len(weekly_counts) < 2 or pd.isna(std_val)) else float(std_val)
    else:
        bets_per_week_mean = 0.0
        bets_per_week_p50 = 0.0
        bets_per_week_p90 = 0.0
        bets_per_week_std = 0.0

    # Bets per year
    bets_per_year = df.groupby(year_col).size().to_dict()
    years_with_data = sorted(df[year_col].unique().tolist())

    return SelectionMetrics(
        n_bets=n_bets,
        n_wins=n_wins,
        n_losses=n_losses,
        n_pushes=n_pushes,
        ats_pct=ats_pct,
        roi_pct=roi_pct,
        avg_ev=avg_ev,
        avg_edge=avg_edge,
        bets_per_week_mean=bets_per_week_mean,
        bets_per_week_p50=bets_per_week_p50,
        bets_per_week_p90=bets_per_week_p90,
        bets_per_week_std=bets_per_week_std,
        bets_per_year=bets_per_year,
        years_with_data=years_with_data,
    )


# =============================================================================
# STABILITY SCORING
# =============================================================================

def compute_stability_score(
    metrics: SelectionMetrics,
    min_bets: int = 50,
    target_ats: float = 52.4,
    weekly_variance_penalty: float = 0.5,
) -> float:
    """Compute stability score for ranking configurations.

    The stability score balances:
    - ATS performance (higher is better)
    - Bet volume (penalize too few bets)
    - Weekly variance (penalize high variance in weekly bet counts)

    Formula:
        stability = (ats_pct - target_ats) * volume_factor - weekly_variance_penalty * weekly_cv

    where:
        volume_factor = min(1.0, n_bets / min_bets)
        weekly_cv = bets_per_week_std / bets_per_week_mean (coefficient of variation)

    Args:
        metrics: SelectionMetrics from compute_selection_metrics
        min_bets: Minimum bets for full credit (default 50)
        target_ats: Breakeven ATS% (default 52.4 for -110)
        weekly_variance_penalty: Penalty weight for weekly variance (default 0.5)

    Returns:
        Stability score (higher is better, can be negative)
    """
    if metrics.n_bets == 0:
        return -100.0  # Heavily penalize no bets

    # Volume factor: full credit if >= min_bets, linear penalty below
    volume_factor = min(1.0, metrics.n_bets / min_bets)

    # ATS edge over breakeven
    ats_edge = metrics.ats_pct - target_ats

    # Weekly coefficient of variation (penalize high variance)
    if metrics.bets_per_week_mean > 0 and not np.isnan(metrics.bets_per_week_std):
        weekly_cv = metrics.bets_per_week_std / metrics.bets_per_week_mean
    else:
        weekly_cv = 0.0  # No variance if single week or no mean

    # Stability score
    stability = (ats_edge * volume_factor) - (weekly_variance_penalty * weekly_cv * 10)

    return stability


def compute_max_drawdown(
    selected_bets: pd.DataFrame,
    outcome_col: str = "jp_side_covered",
    push_col: str = "push",
    juice: int = -110,
) -> float:
    """Compute maximum drawdown from peak cumulative return.

    The DataFrame is sorted by (year, week, game_id) internally to ensure
    deterministic results regardless of input order.

    Args:
        selected_bets: DataFrame with year, week, game_id columns
        outcome_col: Win/loss outcome column
        push_col: Push indicator column
        juice: American odds

    Returns:
        Maximum drawdown as positive number (e.g., 5.0 = 5 unit drawdown)
    """
    if len(selected_bets) == 0:
        return 0.0

    df = selected_bets.copy()

    # Enforce chronological sort for meaningful drawdown
    sort_cols = [c for c in ("year", "week", "game_id") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Ensure push column
    if push_col not in df.columns:
        df[push_col] = False

    # Drop unsettled bets (NaN outcomes would be counted as wins via bool(NaN)==True)
    if outcome_col in df.columns:
        settled_mask = df[outcome_col].notna()
        if push_col in df.columns:
            settled_mask = settled_mask & df[push_col].notna()
        n_unsettled = int((~settled_mask).sum())
        if n_unsettled > 0:
            logger.warning(
                "compute_max_drawdown: dropping %d rows with NaN outcomes (unsettled bets)",
                n_unsettled,
            )
            df = df[settled_mask].copy()

    if len(df) == 0:
        return 0.0

    # Compute per-bet P&L
    if juice < 0:
        payout = 100 / abs(juice)
    else:
        payout = juice / 100

    def bet_pnl(row):
        if row[push_col]:
            return 0.0
        elif row[outcome_col]:
            return payout
        else:
            return -1.0

    df["pnl"] = df.apply(bet_pnl, axis=1)

    # Cumulative return
    cumulative = df["pnl"].cumsum()

    # Running max
    running_max = cumulative.cummax()

    # Drawdown at each point
    drawdown = running_max - cumulative

    return float(drawdown.max())


# =============================================================================
# GRID SEARCH HELPERS
# =============================================================================

def generate_policy_grid() -> list[SelectionPolicyConfig]:
    """Generate default parameter grid for backtest sweep.

    Grid dimensions:
        EV_THRESHOLD: ev_min × max_bets_per_week
        TOP_N_PER_WEEK: top_n × ev_floor × max_bets_per_week
        HYBRID: top_n × ev_floor × max_bets_per_week

    Note: max_bets_per_week includes 3, 5, 10 to match preset configs:
        - balanced preset: max_bets=3
        - aggressive preset: max_bets=5
        - grid default: max_bets=10

    Returns:
        List of SelectionPolicyConfig objects to evaluate
    """
    configs = []

    # EV_THRESHOLD variants: 4 ev_min × 3 max_bets = 12 configs
    for ev_min in [0.01, 0.02, 0.03, 0.05]:
        for max_bets in [5, 10, 15]:
            configs.append(SelectionPolicyConfig(
                selection_policy="EV_THRESHOLD",
                ev_min=ev_min,
                max_bets_per_week=max_bets,
            ))

    # TOP_N_PER_WEEK variants: 4 top_n × 3 ev_floor × 3 max_bets = 36 configs
    # (reduced ev_floor options to keep grid manageable)
    for top_n in [3, 5, 8, 10]:
        for ev_floor in [0.0, 0.01, 0.03]:  # Reduced from 4 to 3 options
            for max_bets in [3, 5, 10]:  # Include preset values 3 and 5
                configs.append(SelectionPolicyConfig(
                    selection_policy="TOP_N_PER_WEEK",
                    top_n_per_week=top_n,
                    ev_floor=ev_floor,
                    max_bets_per_week=max_bets,
                ))

    # HYBRID variants: 4 top_n × 3 ev_floor × 3 max_bets = 36 configs
    for top_n in [3, 5, 8, 10]:
        for ev_floor in [0.01, 0.03, 0.05]:  # Reduced from 4 to 3 options
            for max_bets in [3, 5, 10]:  # Include preset values 3 and 5
                configs.append(SelectionPolicyConfig(
                    selection_policy="HYBRID",
                    top_n_per_week=top_n,
                    ev_floor=ev_floor,
                    max_bets_per_week=max_bets,
                ))

    return configs


def config_to_label(config: SelectionPolicyConfig) -> str:
    """Generate human-readable label for a config.

    Args:
        config: SelectionPolicyConfig

    Returns:
        Label like "EV_THRESH(ev>=3%,max=5)" or "TOP_N(n=3,floor=1%,max=3)"
    """
    policy = config.selection_policy

    if policy == "EV_THRESHOLD":
        return f"EV_THRESH(ev>={config.ev_min*100:.0f}%,max={config.max_bets_per_week})"
    elif policy == "TOP_N_PER_WEEK":
        floor_str = f",floor={config.ev_floor*100:.0f}%" if config.ev_floor > 0 else ""
        return f"TOP_N(n={config.top_n_per_week}{floor_str},max={config.max_bets_per_week})"
    elif policy == "HYBRID":
        return f"HYBRID(n={config.top_n_per_week},floor={config.ev_floor*100:.0f}%,max={config.max_bets_per_week})"
    else:
        return f"{policy}(?)"


# =============================================================================
# 2026 PRESET CONFIGURATIONS
# =============================================================================

# Preset definitions - these are fixed, evidence-based configurations
# grounded in the 2022-2025 backtest grid search results.
PRESET_CONFIGS = {
    "conservative": SelectionPolicyConfig(
        selection_policy="EV_THRESHOLD",
        ev_min=0.03,
        ev_floor=0.0,
        top_n_per_week=10,
        max_bets_per_week=5,
        phase1_policy="skip",
    ),
    "balanced": SelectionPolicyConfig(
        selection_policy="TOP_N_PER_WEEK",
        top_n_per_week=3,
        ev_min=0.03,
        ev_floor=0.01,
        max_bets_per_week=3,
        phase1_policy="skip",
    ),
    "aggressive": SelectionPolicyConfig(
        selection_policy="TOP_N_PER_WEEK",
        top_n_per_week=5,
        ev_min=0.03,
        ev_floor=0.01,
        max_bets_per_week=5,
        phase1_policy="skip",
    ),
}

# Allowed preset names (for validation)
ALLOWED_PRESETS = frozenset(PRESET_CONFIGS.keys())


def get_selection_policy_preset(name: str) -> SelectionPolicyConfig:
    """Get a named preset selection policy configuration.

    Presets are evidence-based configurations grounded in the 2022-2025
    backtest grid search results. They represent different risk/volume
    tradeoffs for production betting.

    Available presets:
        "conservative":
            - EV_THRESHOLD policy with ev_min=3%, max=5 bets/week
            - Prioritizes high-confidence bets with strict EV floor
            - Backtest: ~100 bets/season, ~58-59% ATS, ~12% ROI
            - Best for: Minimizing variance, capital preservation

        "balanced":
            - TOP_N_PER_WEEK with n=3, floor=1%
            - Takes top 3 by EV each week, requires 1% floor
            - Backtest: ~97 bets/season, ~58-59% ATS, ~12% ROI
            - Best for: Balanced volume and edge

        "aggressive":
            - TOP_N_PER_WEEK with n=5, floor=1%
            - Takes top 5 by EV each week, requires 1% floor
            - Backtest: ~150-200 bets/season, ~55-57% ATS, ~7-10% ROI
            - Best for: Higher volume, still disciplined

    All presets use phase1_policy="skip" to avoid low-confidence
    early-season bets (weeks 0-3).

    Args:
        name: Preset name ("conservative", "balanced", or "aggressive")

    Returns:
        SelectionPolicyConfig for the requested preset

    Raises:
        ValueError: If name is not a valid preset

    Examples:
        >>> config = get_selection_policy_preset("conservative")
        >>> config.selection_policy
        'EV_THRESHOLD'
        >>> config.ev_min
        0.03

        >>> config = get_selection_policy_preset("balanced")
        >>> config.top_n_per_week
        3
    """
    name_lower = name.lower().strip()

    if name_lower not in ALLOWED_PRESETS:
        allowed = ", ".join(sorted(ALLOWED_PRESETS))
        raise ValueError(
            f"Unknown preset '{name}'. Valid presets are: {allowed}"
        )

    # Return a copy to prevent modification of global preset
    preset = PRESET_CONFIGS[name_lower]
    return SelectionPolicyConfig(
        selection_policy=preset.selection_policy,
        ev_min=preset.ev_min,
        ev_floor=preset.ev_floor,
        top_n_per_week=preset.top_n_per_week,
        max_bets_per_week=preset.max_bets_per_week,
        phase1_policy=preset.phase1_policy,
        phase2_weeks=preset.phase2_weeks,
    )


def configs_match(a: SelectionPolicyConfig, b: SelectionPolicyConfig) -> bool:
    """Check if two configs have matching policy parameters.

    Compares: selection_policy, ev_min, ev_floor, top_n_per_week,
    max_bets_per_week, phase1_policy.

    Args:
        a, b: SelectionPolicyConfig objects to compare

    Returns:
        True if all relevant fields match
    """
    return (
        a.selection_policy == b.selection_policy
        and a.ev_min == b.ev_min
        and a.ev_floor == b.ev_floor
        and a.top_n_per_week == b.top_n_per_week
        and a.max_bets_per_week == b.max_bets_per_week
        and a.phase1_policy == b.phase1_policy
    )
