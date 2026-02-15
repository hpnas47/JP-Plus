#!/usr/bin/env python3
"""
Weekly spread EV betting runner with monitoring and settlement.

Production use: Generate spread bet recommendations with calibrated EV selection.

Usage:
    # Generate bets for upcoming week (auto-detect)
    python scripts/run_spread_weekly.py

    # Generate bets for specific week
    python scripts/run_spread_weekly.py --year 2026 --week 10

    # Use different preset
    python scripts/run_spread_weekly.py --preset aggressive

    # Override specific parameters
    python scripts/run_spread_weekly.py --ev-min 0.05 --max-bets 3

    # Phase 1 policy (weeks 1-3 only)
    python scripts/run_spread_weekly.py --year 2026 --week 2 --phase1-policy weighted
    python scripts/run_spread_weekly.py --year 2026 --week 2 --phase1-policy skip

    # Settle past bets (update results)
    python scripts/run_spread_weekly.py --settle --year 2026 --week 10

    # Dry run (historical validation)
    python scripts/run_spread_weekly.py --year 2025 --week 10 --dry-run

2026 Production Defaults:
    - Preset: balanced (TOP_N_PER_WEEK, n=3, ev_floor=1%, max_bets=3)
    - Phase 1 (weeks 1-3): weighted calibration, conservative constraints (n=2, ev_floor=2%, max_bets=2)
    - Phase 2 (weeks 4-15): phase2_only calibration, balanced preset
    - Phase 3 (weeks 16+): phase2_only calibration, balanced preset

Phase 1 Conservative Controls:
    - Calibration: weighted (blends phase1 + phase2 data)
    - Selection: TOP_N_PER_WEEK, n=2, ev_floor=0.02, max_bets=2
    - Stake multiplier: 0.5x (reduced risk)

Output:
    - List A: EV-qualified recommendations (calibrated selection policy)
    - List B: 5+ edge disagreements that fail EV threshold (diagnostic)
    - CSV log: data/spread_selection/logs/spread_bets_{year}.csv
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd

# Add project root to path
# NOTE: EV helpers defined after imports block (see american_to_b, ev_units)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.spread_selection import (
    # Calibration
    get_spread_calibration_for_week,
    predict_cover_probability,
    PHASE1_WEEKS,
    PHASE2_WEEKS,
    # Selection policy
    SelectionPolicyConfig,
    SelectionPolicy,
    apply_selection_policy,
    compute_selection_metrics,
    get_selection_policy_preset,
    config_to_label,
    ALLOWED_PRESETS,
)
from src.api.cfbd_client import CFBDClient

# =============================================================================
# EV / ODDS HELPERS
# =============================================================================

def american_to_b(odds_american: float) -> float:
    """Convert American odds to net win multiple per 1 unit staked.

    Examples:
        -110 -> 100/110 = 0.90909...
        +150 -> 1.5
    """
    if odds_american < 0:
        return 100.0 / abs(odds_american)
    else:
        return odds_american / 100.0


def implied_prob_from_american(odds_american: float) -> float:
    """Convert American odds to implied probability (no-vig single side).

    Examples:
        -110 -> 110/210 = 0.52381
        +150 -> 100/250 = 0.40
    """
    if odds_american < 0:
        return abs(odds_american) / (abs(odds_american) + 100.0)
    else:
        return 100.0 / (odds_american + 100.0)


def ev_units(p_win: float, odds_american: float) -> float:
    """Compute expected value in betting units (ROI per 1 unit staked).

    EV = p_win * b - (1 - p_win)

    Examples:
        p=0.55, -110: 0.55 * (100/110) - 0.45 = 0.05
        p=0.60, -110: 0.60 * (100/110) - 0.40 = 0.14545
    """
    b = american_to_b(odds_american)
    return p_win * b - (1.0 - p_win)


# Cache FBS teams to avoid repeated API calls
_fbs_teams_cache: dict[int, set[str]] = {}


def get_fbs_teams(year: int) -> set[str]:
    """Get set of FBS team names for a given year.

    Uses cached values to avoid repeated API calls.

    Args:
        year: Season year

    Returns:
        Set of FBS team names
    """
    if year not in _fbs_teams_cache:
        try:
            client = CFBDClient()
            teams = client.get_fbs_teams(year=int(year))  # Ensure Python int
            # Handle both object and dict responses
            if teams and hasattr(teams[0], "school"):
                _fbs_teams_cache[year] = {t.school for t in teams}
            else:
                _fbs_teams_cache[year] = {t["school"] for t in teams}
            logger.debug(f"Loaded {len(_fbs_teams_cache[year])} FBS teams for {year}")
        except Exception as e:
            logger.warning(f"Could not load FBS teams for {year}: {e}")
            # Fallback: return empty set (no filtering)
            return set()
    return _fbs_teams_cache[year]


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Default production preset for 2026+
DEFAULT_PRESET = "balanced"

# Phase 1 policy options
PHASE1_POLICY_SKIP = "skip"
PHASE1_POLICY_WEIGHTED = "weighted"
PHASE1_POLICY_PHASE1_ONLY = "phase1_only"
ALLOWED_PHASE1_POLICIES = frozenset([PHASE1_POLICY_SKIP, PHASE1_POLICY_WEIGHTED, PHASE1_POLICY_PHASE1_ONLY])

# Default Phase 1 policy for 2026+
DEFAULT_PHASE1_POLICY = PHASE1_POLICY_WEIGHTED

# Phase 1 stake multiplier (reduced risk)
DEFAULT_PHASE1_STAKE_MULTIPLIER = 0.5

# Log file paths
LOGS_DIR = project_root / "data" / "spread_selection" / "logs"

# Guardrail codes
GUARDRAIL_OK = "OK"
GUARDRAIL_PHASE1_SKIP = "PHASE1_SKIP"
GUARDRAIL_LOW_TRAIN_GAMES = "LOW_TRAIN_GAMES"
GUARDRAIL_BELOW_EV_THRESHOLD = "BELOW_EV_THRESHOLD"
GUARDRAIL_BELOW_EV_FLOOR = "BELOW_EV_FLOOR"
GUARDRAIL_CAP_EXCEEDED = "CAP_EXCEEDED"


# =============================================================================
# PHASE CONFIGURATION
# =============================================================================

@dataclass
class PhaseConfig:
    """Phase-specific configuration for selection policy and staking.

    Attributes:
        phase: Phase number (1, 2, or 3)
        calibration_name: Name of calibration used (weighted, phase2_only, etc.)
        top_n_per_week: Override for top N per week
        ev_floor: Override for EV floor
        max_bets_per_week: Override for max bets per week
        stake_multiplier: Multiplier for stake (1.0 = full, 0.5 = half)
        selection_policy: Selection policy to use (EV_THRESHOLD or TOP_N_PER_WEEK)
    """
    phase: int
    calibration_name: str
    top_n_per_week: int
    ev_floor: float
    max_bets_per_week: int
    stake_multiplier: float
    selection_policy: str = "TOP_N_PER_WEEK"  # Default to TOP_N


# Phase 1: EV_THRESHOLD policy (take ALL bets above 2% EV, no cap)
# Note: Calibration less reliable in weeks 1-3, so 0.5x stake
PHASE1_CONSERVATIVE_CONFIG = PhaseConfig(
    phase=1,
    calibration_name="weighted",
    top_n_per_week=99,  # Not used with EV_THRESHOLD
    ev_floor=0.02,  # 2% EV floor (stricter than Phase 2)
    max_bets_per_week=99,  # No cap with EV_THRESHOLD
    stake_multiplier=0.5,  # 50% stake (calibration less reliable)
    selection_policy="EV_THRESHOLD",  # Take ALL above threshold
)

# Phase 2 Default Config (uses balanced preset)
PHASE2_DEFAULT_CONFIG = PhaseConfig(
    phase=2,
    calibration_name="phase2_only",
    top_n_per_week=3,
    ev_floor=0.01,
    max_bets_per_week=3,
    stake_multiplier=1.0,  # Full stake
    selection_policy="TOP_N_PER_WEEK",
)

# Phase 3 (Postseason) - same as Phase 2
PHASE3_DEFAULT_CONFIG = PhaseConfig(
    phase=3,
    calibration_name="phase2_only",
    top_n_per_week=3,
    ev_floor=0.01,
    max_bets_per_week=3,
    stake_multiplier=1.0,
    selection_policy="TOP_N_PER_WEEK",
)


def get_phase_config(
    week: int,
    phase1_policy: str = DEFAULT_PHASE1_POLICY,
    phase1_stake_multiplier: float = DEFAULT_PHASE1_STAKE_MULTIPLIER,
) -> PhaseConfig:
    """Get phase-specific configuration for a given week.

    Args:
        week: Week number
        phase1_policy: Policy for Phase 1 ("skip", "weighted", "phase1_only")
        phase1_stake_multiplier: Stake multiplier for Phase 1

    Returns:
        PhaseConfig with appropriate settings for the week
    """
    if week <= PHASE1_WEEKS[1]:
        # Phase 1 (weeks 1-3)
        if phase1_policy == PHASE1_POLICY_SKIP:
            # Skip mode: return config but with signal to skip List A
            # Note: top_n_per_week=1 to pass validation, but calibration_name="skip"
            # tells build_candidates to produce empty List A
            return PhaseConfig(
                phase=1,
                calibration_name="skip",
                top_n_per_week=1,  # Minimum valid, but won't be used (skip mode)
                ev_floor=1.0,  # Effectively filter everything
                max_bets_per_week=1,  # Minimum valid
                stake_multiplier=0.0,
                selection_policy="EV_THRESHOLD",
            )
        else:
            # weighted or phase1_only mode: use EV_THRESHOLD (take all above 2%)
            return PhaseConfig(
                phase=1,
                calibration_name=phase1_policy,
                top_n_per_week=PHASE1_CONSERVATIVE_CONFIG.top_n_per_week,
                ev_floor=PHASE1_CONSERVATIVE_CONFIG.ev_floor,
                max_bets_per_week=PHASE1_CONSERVATIVE_CONFIG.max_bets_per_week,
                stake_multiplier=phase1_stake_multiplier,
                selection_policy=PHASE1_CONSERVATIVE_CONFIG.selection_policy,
            )
    elif week <= PHASE2_WEEKS[1]:
        # Phase 2 (weeks 4-15)
        return PHASE2_DEFAULT_CONFIG
    else:
        # Phase 3 (weeks 16+)
        return PHASE3_DEFAULT_CONFIG


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SpreadBetLog:
    """Log entry for a spread bet recommendation.

    This is the schema for CSV logging with all required fields.
    """
    # Timestamps
    run_timestamp: str

    # Game identifiers
    year: int
    week: int
    game_id: str

    # Teams
    home_team: str
    away_team: str

    # Market data
    market_spread: float  # Vegas spread (negative = home favored)
    model_spread: float   # JP+ predicted spread
    edge_pts: float       # Model - Market (signed)
    edge_abs: float       # Absolute edge

    # Bet details
    side: str  # "home" or "away"
    bet_spread: float  # The spread we're betting

    # Odds (placeholder if not available)
    odds_american: int
    odds_placeholder: bool  # True if using assumed -110
    implied_prob: float

    # Calibration outputs
    p_cover: float  # Calibrated probability of covering
    ev: float       # Expected value

    # Phase routing
    phase: int  # 1, 2, or 3
    calibration_name: str  # "weighted", "phase2_only", "skip", etc.

    # Selection policy
    preset_name: str
    selection_policy: str
    ev_min: float
    ev_floor: float
    top_n_per_week: int
    max_bets_per_week: int

    # Staking
    stake: float  # Units to stake (0 if skipped)
    stake_multiplier_used: float  # Multiplier applied (e.g., 0.5 for Phase 1)
    guardrail_reason: str

    # List membership
    list_type: str  # "A" (EV primary) or "B" (5+ edge)

    # Results (nullable - filled in by settlement)
    actual_margin: Optional[float] = None
    covered: Optional[str] = None  # "W", "L", "P"
    profit_units: Optional[float] = None
    settled_timestamp: Optional[str] = None


# =============================================================================
# PRODUCTION CONFIG
# =============================================================================

def get_production_config(
    preset: str = DEFAULT_PRESET,
    ev_min: Optional[float] = None,
    ev_floor: Optional[float] = None,
    top_n_per_week: Optional[int] = None,
    max_bets_per_week: Optional[int] = None,
) -> SelectionPolicyConfig:
    """Get production selection policy config with optional overrides.

    Args:
        preset: Preset name (conservative, balanced, aggressive)
        ev_min: Override ev_min (for EV_THRESHOLD)
        ev_floor: Override ev_floor (for TOP_N, HYBRID)
        top_n_per_week: Override top N per week
        max_bets_per_week: Override max bets per week

    Returns:
        SelectionPolicyConfig with preset defaults and overrides applied

    Raises:
        ValueError: If preset is invalid
    """
    # Start with preset
    config = get_selection_policy_preset(preset)

    # Apply overrides (create new config with overrides)
    if ev_min is not None or ev_floor is not None or top_n_per_week is not None or max_bets_per_week is not None:
        config = SelectionPolicyConfig(
            selection_policy=config.selection_policy,
            ev_min=ev_min if ev_min is not None else config.ev_min,
            ev_floor=ev_floor if ev_floor is not None else config.ev_floor,
            top_n_per_week=top_n_per_week if top_n_per_week is not None else config.top_n_per_week,
            max_bets_per_week=max_bets_per_week if max_bets_per_week is not None else config.max_bets_per_week,
            phase1_policy=config.phase1_policy,
            phase2_weeks=config.phase2_weeks,
        )

    return config


# =============================================================================
# DATA LOADING
# =============================================================================

def load_backtest_data(year: int, week: int) -> Optional[pd.DataFrame]:
    """Load backtest data for a specific year/week from ATS export.

    This is used for historical validation / dry-run mode.
    Filters to FBS-only games (excludes any game involving FCS teams).

    Args:
        year: Season year
        week: Week number

    Returns:
        DataFrame with game data or None if not found
    """
    ats_path = project_root / "data" / "spread_selection" / "ats_export.csv"

    if not ats_path.exists():
        logger.warning(f"ATS export not found: {ats_path}")
        return None

    df = pd.read_csv(ats_path)

    # Filter to year/week
    mask = (df["year"] == year) & (df["week"] == week)
    week_data = df[mask].copy()

    if len(week_data) == 0:
        logger.warning(f"No data found for {year} week {week}")
        return None

    # Filter to FBS-only games (exclude any game involving FCS teams)
    fbs_teams = get_fbs_teams(year)
    if fbs_teams:
        pre_filter = len(week_data)
        fbs_mask = (
            week_data["home_team"].isin(fbs_teams) &
            week_data["away_team"].isin(fbs_teams)
        )
        week_data = week_data[fbs_mask].copy()
        filtered = pre_filter - len(week_data)
        if filtered > 0:
            logger.debug(f"Filtered {filtered} FCS games from {year} week {week}")

    return week_data


def load_games_for_settlement(year: int) -> pd.DataFrame:
    """Load game results for settlement from CFBD.

    Args:
        year: Season year

    Returns:
        DataFrame with game_id, actual_margin columns
    """
    # Try to load from ATS export first (has actual_margin)
    ats_path = project_root / "data" / "spread_selection" / "ats_export.csv"

    if ats_path.exists():
        df = pd.read_csv(ats_path)
        df = df[df["year"] == year].copy()
        if "actual_margin" in df.columns:
            # ATS export has different column names
            result = df[["game_id", "actual_margin"]].copy()
            # Compute home_covered from actual_margin and vegas_spread
            if "vegas_spread" in df.columns:
                result["home_covered"] = df["actual_margin"] > -df["vegas_spread"]
            else:
                result["home_covered"] = df["actual_margin"] > 0
            # Compute push (exact spread)
            if "vegas_spread" in df.columns:
                result["push"] = np.abs(df["actual_margin"] + df["vegas_spread"]) < 0.01
            else:
                result["push"] = False
            return result

    # Fallback: Could call CFBD API here if needed
    logger.warning(f"No game results found for {year}")
    return pd.DataFrame(columns=["game_id", "actual_margin", "home_covered", "push"])


# =============================================================================
# CANDIDATE BUILDING
# =============================================================================

def build_candidates(
    week_data: pd.DataFrame,
    year: int,
    week: int,
    config: SelectionPolicyConfig,
    phase_config: PhaseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build candidate bets for List A (EV) and List B (5+ edge).

    Args:
        week_data: DataFrame with game predictions and Vegas lines
        year: Season year
        week: Week number
        config: Selection policy config
        phase_config: Phase-specific configuration

    Returns:
        Tuple of (list_a_candidates, list_b_candidates)
        - List A: All EV-positive candidates (before selection policy)
        - List B: 5+ edge games that don't meet EV threshold
    """
    df = week_data.copy()

    # Handle column name variations (backtest export vs live data)
    # Backtest: predicted_spread, edge
    # Live: jp_spread, edge_pts
    if "predicted_spread" in df.columns and "jp_spread" not in df.columns:
        df["jp_spread"] = df["predicted_spread"]
    if "edge" in df.columns and "edge_pts" not in df.columns:
        df["edge_pts"] = df["edge"]

    # Required columns
    required = ["game_id", "home_team", "away_team", "jp_spread", "vegas_spread", "edge_pts"]
    missing = set(required) - set(df.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return pd.DataFrame(), pd.DataFrame()

    # Compute absolute edge
    if "edge_abs" not in df.columns:
        df["edge_abs"] = df["edge_pts"].abs()

    # Add year/week if not present
    df["year"] = year
    df["week"] = week

    # Determine side (home or away)
    # If "pick" column exists (from backtest), use it directly
    # Otherwise, derive from edge direction
    if "pick" in df.columns:
        df["jp_favored_side"] = df["pick"].str.upper()
    else:
        # Negative edge = JP+ likes home more than Vegas
        df["jp_favored_side"] = df["edge_pts"].apply(lambda x: "HOME" if x < 0 else "AWAY")

    # Add outcome columns if available (for backtest data)
    if "ats_win" in df.columns:
        df["jp_side_covered"] = df["ats_win"]
    if "ats_push" in df.columns:
        df["push"] = df["ats_push"]

    # Check if Phase 1 skip mode
    if phase_config.calibration_name == "skip":
        # Phase 1 skip: No List A candidates, but still compute List B
        list_a_candidates = pd.DataFrame(columns=df.columns.tolist() + ["ev", "p_cover"])
    else:
        # Compute EV using phase-appropriate calibration
        try:
            # Map phase config calibration name to get_spread_calibration_for_week params
            if phase_config.phase == 1:
                # Phase 1: use the specified calibration (weighted or phase1_only)
                calibration_result = get_spread_calibration_for_week(
                    week,
                    phase1_policy="apply",  # Enable Phase 1
                    phase_mode=f"force_{phase_config.calibration_name}",
                )
            else:
                # Phase 2/3: use phase2_only
                calibration_result = get_spread_calibration_for_week(
                    week,
                    phase1_policy="skip",  # Skip Phase 1 logic (we're in Phase 2/3)
                )

            # Odds for EV computation (placeholder -110 until real odds wired)
            odds_col = -110
            if calibration_result is None:
                logger.warning("Could not load calibration, using default EV estimation")
                # Fallback: Simple edge-based EV proxy
                df["p_cover"] = 0.5 + (df["edge_abs"] / 100)  # Rough approximation
            else:
                # Use calibration to compute p_cover
                # edge_abs is always positive, calibration gives P(cover)
                df["p_cover"] = predict_cover_probability(
                    df["edge_abs"].values, calibration_result
                )
            # Compute EV consistently using correct ROI formula for both paths
            df["ev"] = df["p_cover"].apply(lambda p: ev_units(p, odds_col))
        except Exception as e:
            logger.warning(f"Error computing calibration: {e}, using fallback")
            df["p_cover"] = 0.5 + (df["edge_abs"] / 100)
            df["ev"] = df["p_cover"].apply(lambda p: ev_units(p, -110))

        # List A candidates: EV > 0 (will be filtered by selection policy)
        list_a_candidates = df[df["ev"] > 0].copy()

    # List B: 5+ edge games that fail EV threshold
    # These are kept regardless of phase for diagnostic purposes
    edge5_mask = df["edge_abs"] >= 5.0

    # Use phase-specific ev_floor for List B exclusion
    ev_threshold = phase_config.ev_floor

    if len(list_a_candidates) > 0:
        # Exclude games already in List A with ev >= ev_floor
        list_a_game_ids = set(list_a_candidates[list_a_candidates["ev"] >= ev_threshold]["game_id"])
        list_b_mask = edge5_mask & ~df["game_id"].isin(list_a_game_ids)
    else:
        list_b_mask = edge5_mask

    list_b_candidates = df[list_b_mask].copy()

    return list_a_candidates, list_b_candidates


# =============================================================================
# BET LOGGING
# =============================================================================

def get_log_path(year: int) -> Path:
    """Get path to bet log CSV for a year."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / f"spread_bets_{year}.csv"


def load_bet_log(year: int) -> pd.DataFrame:
    """Load existing bet log for a year.

    Returns empty DataFrame with correct schema if file doesn't exist.
    """
    log_path = get_log_path(year)

    if log_path.exists():
        return pd.read_csv(log_path)

    # Return empty DataFrame with schema
    return pd.DataFrame(columns=[
        "run_timestamp", "year", "week", "game_id", "home_team", "away_team",
        "market_spread", "model_spread", "edge_pts", "edge_abs",
        "side", "bet_spread", "odds_american", "odds_placeholder", "implied_prob",
        "p_cover", "ev", "phase", "calibration_name",
        "preset_name", "selection_policy", "ev_min", "ev_floor",
        "top_n_per_week", "max_bets_per_week", "stake", "stake_multiplier_used",
        "guardrail_reason", "list_type",
        "actual_margin", "covered", "profit_units", "settled_timestamp",
    ])


def save_bet_log(df: pd.DataFrame, year: int) -> None:
    """Save bet log for a year.

    Args:
        df: DataFrame with bet log entries
        year: Season year
    """
    log_path = get_log_path(year)
    df.to_csv(log_path, index=False)
    logger.info(f"Saved bet log to {log_path}")


def append_bets_to_log(
    bets: list[SpreadBetLog],
    year: int,
    week: int,
    dedupe: bool = True,
) -> int:
    """Append new bets to the log, with deduplication.

    Args:
        bets: List of SpreadBetLog entries
        year: Season year
        week: Week number
        dedupe: If True, skip bets already in log for this (year, week, game_id, side)

    Returns:
        Number of new bets appended
    """
    if not bets:
        return 0

    # Load existing log
    existing = load_bet_log(year)

    # Convert bets to DataFrame
    new_rows = pd.DataFrame([asdict(b) for b in bets])

    if dedupe and len(existing) > 0:
        # Create dedup key
        existing["_dedup_key"] = (
            existing["year"].astype(str) + "_" +
            existing["week"].astype(str) + "_" +
            existing["game_id"].astype(str) + "_" +
            existing["side"].astype(str)
        )
        new_rows["_dedup_key"] = (
            new_rows["year"].astype(str) + "_" +
            new_rows["week"].astype(str) + "_" +
            new_rows["game_id"].astype(str) + "_" +
            new_rows["side"].astype(str)
        )

        # Filter out duplicates
        existing_keys = set(existing["_dedup_key"])
        new_rows = new_rows[~new_rows["_dedup_key"].isin(existing_keys)]

        # Drop dedup key
        existing = existing.drop(columns=["_dedup_key"])
        if len(new_rows) > 0:
            new_rows = new_rows.drop(columns=["_dedup_key"])

    if len(new_rows) == 0:
        logger.info(f"No new bets to append (all duplicates)")
        return 0

    # Append and save
    combined = pd.concat([existing, new_rows], ignore_index=True)
    save_bet_log(combined, year)

    return len(new_rows)


# =============================================================================
# RECOMMENDATION GENERATION
# =============================================================================

def generate_recommendations(
    year: int,
    week: int,
    config: SelectionPolicyConfig,
    preset_name: str,
    phase_config: PhaseConfig,
    dry_run: bool = False,
) -> tuple[list[SpreadBetLog], list[SpreadBetLog], dict]:
    """Generate spread bet recommendations for a week.

    Args:
        year: Season year
        week: Week number
        config: Selection policy config (base preset, may be overridden by phase)
        preset_name: Name of preset used
        phase_config: Phase-specific configuration
        dry_run: If True, load from historical data

    Returns:
        Tuple of (list_a_bets, list_b_bets, summary_stats)
    """
    run_timestamp = datetime.now().isoformat()

    # Load data
    if dry_run:
        week_data = load_backtest_data(year, week)
    else:
        # TODO: For live production, load from weekly slate export
        # For now, fall back to backtest data
        week_data = load_backtest_data(year, week)

    if week_data is None or len(week_data) == 0:
        logger.error(f"No data available for {year} week {week}")
        return [], [], {"error": "No data available"}

    # Create phase-adjusted selection config
    # Phase 1 uses EV_THRESHOLD (all above 2%), Phase 2 uses TOP_N_PER_WEEK
    # For EV_THRESHOLD policy (Phase 1), ev_min IS the threshold — wire from phase ev_floor
    effective_ev_min = (
        phase_config.ev_floor
        if phase_config.selection_policy == "EV_THRESHOLD"
        else config.ev_min
    )
    phase_adjusted_config = SelectionPolicyConfig(
        selection_policy=phase_config.selection_policy,  # Use phase policy
        ev_min=effective_ev_min,
        ev_floor=phase_config.ev_floor,  # Use phase ev_floor
        top_n_per_week=phase_config.top_n_per_week,  # Use phase top_n
        max_bets_per_week=phase_config.max_bets_per_week,  # Use phase max_bets
        phase1_policy="apply" if phase_config.calibration_name != "skip" else "skip",
        phase2_weeks=config.phase2_weeks,
    )

    # Build candidates with phase config
    list_a_candidates, list_b_candidates = build_candidates(
        week_data, year, week, phase_adjusted_config, phase_config
    )

    # Check if Phase 1 skip mode
    phase1_skipped = phase_config.calibration_name == "skip"

    # Apply selection policy to List A
    list_a_bets = []

    if not phase1_skipped and len(list_a_candidates) > 0:
        result = apply_selection_policy(list_a_candidates, phase_adjusted_config)
        selected = result.selected_bets

        # Convert to bet logs
        for _, row in selected.iterrows():
            # Determine bet side and spread
            side = "home" if row["jp_favored_side"] == "HOME" else "away"
            bet_spread = row.get("vegas_spread", 0) if side == "home" else -row.get("vegas_spread", 0)

            # Odds (placeholder for now)
            odds_american = -110
            odds_placeholder = True
            implied_prob = implied_prob_from_american(odds_american)

            # Apply phase stake multiplier
            base_stake = 1.0
            actual_stake = base_stake * phase_config.stake_multiplier

            bet = SpreadBetLog(
                run_timestamp=run_timestamp,
                year=year,
                week=week,
                game_id=str(row["game_id"]),
                home_team=row["home_team"],
                away_team=row["away_team"],
                market_spread=row.get("vegas_spread", 0),
                model_spread=row.get("jp_spread", 0),
                edge_pts=row.get("edge_pts", 0),
                edge_abs=row.get("edge_abs", 0),
                side=side,
                bet_spread=bet_spread,
                odds_american=odds_american,
                odds_placeholder=odds_placeholder,
                implied_prob=implied_prob,
                p_cover=row.get("p_cover", 0.5),
                ev=row.get("ev", 0),
                phase=phase_config.phase,
                calibration_name=phase_config.calibration_name,
                preset_name=preset_name,
                selection_policy=phase_adjusted_config.selection_policy,
                ev_min=phase_adjusted_config.ev_min,
                ev_floor=phase_adjusted_config.ev_floor,
                top_n_per_week=phase_adjusted_config.top_n_per_week,
                max_bets_per_week=phase_adjusted_config.max_bets_per_week,
                stake=actual_stake,
                stake_multiplier_used=phase_config.stake_multiplier,
                guardrail_reason=GUARDRAIL_OK,
                list_type="A",
            )
            list_a_bets.append(bet)

    # List B: 5+ edge bets (diagnostic, no stake)
    list_b_bets = []

    for _, row in list_b_candidates.iterrows():
        side = "home" if row.get("jp_favored_side", "HOME") == "HOME" else "away"
        bet_spread = row.get("vegas_spread", 0) if side == "home" else -row.get("vegas_spread", 0)

        # Determine guardrail reason
        if phase1_skipped:
            guardrail = GUARDRAIL_PHASE1_SKIP
        else:
            guardrail = GUARDRAIL_BELOW_EV_THRESHOLD

        bet = SpreadBetLog(
            run_timestamp=run_timestamp,
            year=year,
            week=week,
            game_id=str(row["game_id"]),
            home_team=row["home_team"],
            away_team=row["away_team"],
            market_spread=row.get("vegas_spread", 0),
            model_spread=row.get("jp_spread", 0),
            edge_pts=row.get("edge_pts", 0),
            edge_abs=row.get("edge_abs", 0),
            side=side,
            bet_spread=bet_spread,
            odds_american=-110,
            odds_placeholder=True,
            implied_prob=110 / 210,
            p_cover=row.get("p_cover", 0.5),
            ev=row.get("ev", 0),
            phase=phase_config.phase,
            calibration_name=phase_config.calibration_name,
            preset_name=preset_name,
            selection_policy=phase_adjusted_config.selection_policy,
            ev_min=phase_adjusted_config.ev_min,
            ev_floor=phase_adjusted_config.ev_floor,
            top_n_per_week=phase_adjusted_config.top_n_per_week,
            max_bets_per_week=phase_adjusted_config.max_bets_per_week,
            stake=0.0,  # No stake for List B
            stake_multiplier_used=0.0,
            guardrail_reason=guardrail,
            list_type="B",
        )
        list_b_bets.append(bet)

    # Summary stats
    summary = {
        "year": year,
        "week": week,
        "games_evaluated": len(week_data),
        "list_a_count": len(list_a_bets),
        "list_b_count": len(list_b_bets),
        "total_stake": sum(b.stake for b in list_a_bets),
        "avg_edge": np.mean([b.edge_abs for b in list_a_bets]) if list_a_bets else 0.0,
        "avg_ev": np.mean([b.ev for b in list_a_bets]) if list_a_bets else 0.0,
        "home_count": sum(1 for b in list_a_bets if b.side == "home"),
        "away_count": sum(1 for b in list_a_bets if b.side == "away"),
        # Phase info
        "phase": phase_config.phase,
        "calibration_name": phase_config.calibration_name,
        "stake_multiplier": phase_config.stake_multiplier,
        "phase1_skipped": phase1_skipped,
        # Selection constraints used
        "selection_top_n": phase_config.top_n_per_week,
        "selection_ev_floor": phase_config.ev_floor,
        "selection_max_bets": phase_config.max_bets_per_week,
        # Base config
        "preset": preset_name,
        "config_label": config_to_label(phase_adjusted_config),
    }

    return list_a_bets, list_b_bets, summary


# =============================================================================
# PROGRAMMATIC API
# =============================================================================

def run_week(
    year: int,
    week: int,
    preset: str = DEFAULT_PRESET,
    phase1_policy: str = DEFAULT_PHASE1_POLICY,
    phase1_stake_multiplier: float = DEFAULT_PHASE1_STAKE_MULTIPLIER,
    ev_min: Optional[float] = None,
    ev_floor: Optional[float] = None,
    top_n_per_week: Optional[int] = None,
    max_bets_per_week: Optional[int] = None,
    write_csv: bool = False,
    log_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run weekly recommendations programmatically.

    This function exposes the runner logic for programmatic use,
    returning DataFrames instead of just printing/writing.

    Args:
        year: Season year
        week: Week number
        preset: Selection policy preset
        phase1_policy: Phase 1 calibration policy
        phase1_stake_multiplier: Stake multiplier for Phase 1
        ev_min: Override ev_min
        ev_floor: Override ev_floor
        top_n_per_week: Override top N per week
        max_bets_per_week: Override max bets per week
        write_csv: If True, write to CSV log
        log_path: Custom log path (if None, uses default)

    Returns:
        Tuple of (list_a_df, list_b_df, summary_dict)
        - list_a_df: DataFrame of List A bets
        - list_b_df: DataFrame of List B bets
        - summary_dict: Summary statistics
    """
    # Build base config from preset
    config = get_production_config(
        preset=preset,
        ev_min=ev_min,
        ev_floor=ev_floor,
        top_n_per_week=top_n_per_week,
        max_bets_per_week=max_bets_per_week,
    )

    # Get phase-specific config
    phase_config = get_phase_config(
        week=week,
        phase1_policy=phase1_policy,
        phase1_stake_multiplier=phase1_stake_multiplier,
    )

    # Generate recommendations
    list_a_bets, list_b_bets, summary = generate_recommendations(
        year=year,
        week=week,
        config=config,
        preset_name=preset,
        phase_config=phase_config,
        dry_run=True,  # Always use historical data for programmatic access
    )

    # Convert to DataFrames
    if list_a_bets:
        list_a_df = pd.DataFrame([asdict(b) for b in list_a_bets])
    else:
        list_a_df = pd.DataFrame()

    if list_b_bets:
        list_b_df = pd.DataFrame([asdict(b) for b in list_b_bets])
    else:
        list_b_df = pd.DataFrame()

    # Write CSV if requested
    if write_csv:
        all_bets = list_a_bets + list_b_bets
        if log_path is not None:
            # Write to custom path
            if all_bets:
                df = pd.DataFrame([asdict(b) for b in all_bets])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(log_path, index=False)
        else:
            # Write to default log
            append_bets_to_log(all_bets, year, week)

    return list_a_df, list_b_df, summary


# =============================================================================
# SETTLEMENT
# =============================================================================

def settle_bets(year: int, week: int) -> dict:
    """Settle bets for a specific week by updating results.

    Updates bet log in-place with actual_margin, covered, profit_units.

    Args:
        year: Season year
        week: Week number

    Returns:
        Settlement summary dict
    """
    # Load bet log
    log_df = load_bet_log(year)

    if len(log_df) == 0:
        logger.warning(f"No bets found in log for {year}")
        return {"error": "No bets in log"}

    # Filter to unsettled bets for this week
    week_mask = (log_df["week"] == week) & log_df["settled_timestamp"].isna()
    unsettled = log_df[week_mask].copy()

    if len(unsettled) == 0:
        logger.info(f"No unsettled bets for {year} week {week}")
        return {"settled": 0, "already_settled": len(log_df[log_df["week"] == week])}

    # Load game results
    results = load_games_for_settlement(year)

    if len(results) == 0:
        logger.error(f"No game results available for settlement")
        return {"error": "No game results"}

    # Merge results
    settle_timestamp = datetime.now().isoformat()
    settled_count = 0

    # Convert nullable columns to object type to allow mixed values
    log_df["covered"] = log_df["covered"].astype(object)
    log_df["settled_timestamp"] = log_df["settled_timestamp"].astype(object)

    for idx in log_df[week_mask].index:
        game_id = str(log_df.loc[idx, "game_id"])
        side = log_df.loc[idx, "side"]
        bet_spread = log_df.loc[idx, "bet_spread"]
        stake = log_df.loc[idx, "stake"]
        odds_am = log_df.loc[idx, "odds_american"]

        # Find game result
        game_result = results[results["game_id"].astype(str) == game_id]

        if len(game_result) == 0:
            logger.warning(f"No result found for game {game_id}")
            continue

        result_row = game_result.iloc[0]
        actual_margin = result_row["actual_margin"]  # Positive = home won

        # Determine cover
        # If betting home: home_margin > -spread (home covered)
        # If betting away: -home_margin > spread (away covered)
        if side == "home":
            cover_margin = actual_margin + bet_spread  # bet_spread is negative for home favorite
            covered = cover_margin > 0
        else:
            # Away bet: bet_spread is +vegas_spread (positive when home favored)
            # Away covers when: away_margin > -bet_spread, i.e. bet_spread - actual_margin > 0
            cover_margin = bet_spread - actual_margin
            covered = cover_margin > 0

        # Handle push
        b = american_to_b(odds_am)
        if abs(cover_margin) < 0.01:
            covered_str = "P"
            profit = 0.0
        elif covered:
            covered_str = "W"
            profit = stake * b
        else:
            covered_str = "L"
            profit = -stake

        # Update log
        log_df.loc[idx, "actual_margin"] = actual_margin
        log_df.loc[idx, "covered"] = covered_str
        log_df.loc[idx, "profit_units"] = profit
        log_df.loc[idx, "settled_timestamp"] = settle_timestamp

        settled_count += 1

    # Save updated log
    save_bet_log(log_df, year)

    # Summary - only count List A (staked) bets in record
    settled_bets = log_df[(log_df["week"] == week) & log_df["settled_timestamp"].notna()]
    staked_bets = settled_bets[settled_bets["stake"] > 0]
    wins = (staked_bets["covered"] == "W").sum()
    losses = (staked_bets["covered"] == "L").sum()
    pushes = (staked_bets["covered"] == "P").sum()
    profit = staked_bets["profit_units"].sum()

    # List B results (diagnostic only)
    list_b_bets = settled_bets[settled_bets["stake"] == 0]
    list_b_wins = (list_b_bets["covered"] == "W").sum()
    list_b_losses = (list_b_bets["covered"] == "L").sum()

    return {
        "settled": settled_count,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "profit_units": profit,
        "ats_pct": (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0,
        "list_b_wins": list_b_wins,
        "list_b_losses": list_b_losses,
    }


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def print_weekly_summary(
    list_a_bets: list[SpreadBetLog],
    list_b_bets: list[SpreadBetLog],
    summary: dict,
) -> None:
    """Print concise weekly summary to console."""
    print()
    print("=" * 70)
    print(f"SPREAD EV WEEKLY SUMMARY: {summary['year']} Week {summary['week']}")
    print("=" * 70)

    # Phase routing info
    phase = summary.get("phase", 2)
    calibration = summary.get("calibration_name", "phase2_only")
    stake_mult = summary.get("stake_multiplier", 1.0)

    print(f"\n  Phase:               Phase {phase}" + (" (SKIP MODE)" if summary.get("phase1_skipped") else ""))
    print(f"  Calibration:         {calibration}")
    print(f"  Stake multiplier:    {stake_mult:.1f}x")

    # Selection constraints
    print(f"\n  Selection Constraints:")
    print(f"    top_n_per_week:    {summary.get('selection_top_n', 3)}")
    print(f"    ev_floor:          {summary.get('selection_ev_floor', 0.01)*100:.0f}%")
    print(f"    max_bets_per_week: {summary.get('selection_max_bets', 3)}")

    print(f"\n  Games evaluated:     {summary['games_evaluated']}")
    print(f"  Preset:              {summary['preset']} ({summary['config_label']})")
    print()

    # List A
    print(f"  LIST A (EV Primary): {summary['list_a_count']} bets")
    if summary["list_a_count"] > 0:
        print(f"    Total stake:       {summary['total_stake']:.2f} units")
        print(f"    Avg edge:          {summary['avg_edge']:.1f} pts")
        print(f"    Avg ROI:           {summary['avg_ev']*100:.1f}%")
        print(f"    Home/Away split:   {summary['home_count']} / {summary['away_count']}")

    # List B
    print(f"\n  LIST B (5+ Edge):    {summary['list_b_count']} games (diagnostic, no stake)")

    print()
    print("-" * 70)

    # Detail tables
    if list_a_bets:
        print("\nLIST A - EV Qualified Bets:")
        print(f"{'#':<3} {'Matchup':<30} {'Edge':>6} {'ROI':>7} {'Stake':>6} {'Bet':<18}")
        print("-" * 70)
        for i, bet in enumerate(list_a_bets, 1):
            matchup = f"{bet.away_team[:12]} @ {bet.home_team[:12]}"
            bet_str = f"{bet.home_team[:10] if bet.side == 'home' else bet.away_team[:10]} {bet.bet_spread:+.1f}"
            print(f"{i:<3} {matchup:<30} {bet.edge_abs:>5.1f}p {bet.ev*100:>+5.1f}% {bet.stake:>5.2f}u {bet_str:<18}")

    if list_b_bets:
        print("\nLIST B - 5+ Edge (No EV Qualification):")
        print(f"{'#':<3} {'Matchup':<30} {'Edge':>6} {'Reason':<20}")
        print("-" * 70)
        for i, bet in enumerate(list_b_bets[:5], 1):  # Show top 5
            matchup = f"{bet.away_team[:12]} @ {bet.home_team[:12]}"
            print(f"{i:<3} {matchup:<30} {bet.edge_abs:>5.1f}p {bet.guardrail_reason:<20}")
        if len(list_b_bets) > 5:
            print(f"    ... and {len(list_b_bets) - 5} more")

    print()


def print_settlement_summary(result: dict, year: int, week: int) -> None:
    """Print settlement summary to console."""
    print()
    print("=" * 70)
    print(f"SETTLEMENT SUMMARY: {year} Week {week}")
    print("=" * 70)

    if "error" in result:
        print(f"\n  ERROR: {result['error']}")
        return

    print(f"\n  Bets settled:        {result['settled']}")
    print(f"\n  LIST A (Staked):")
    print(f"    Record:            {result['wins']}-{result['losses']}-{result['pushes']}")
    print(f"    ATS%:              {result['ats_pct']:.1f}%")
    print(f"    Profit/Loss:       {result['profit_units']:+.2f} units")

    if result.get("list_b_wins", 0) + result.get("list_b_losses", 0) > 0:
        print(f"\n  LIST B (Diagnostic, no stake):")
        print(f"    Record:            {result.get('list_b_wins', 0)}-{result.get('list_b_losses', 0)}")

    print()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Weekly spread EV betting runner"
    )

    # Mode selection
    parser.add_argument(
        "--settle",
        action="store_true",
        help="Settlement mode: update results for past bets",
    )

    # Time selection
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Season year (default: current year)",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Week number (required for --settle, auto-detect otherwise)",
    )

    # Preset selection
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        choices=list(ALLOWED_PRESETS),
        help=f"Selection policy preset (default: {DEFAULT_PRESET})",
    )

    # Parameter overrides
    parser.add_argument(
        "--ev-min",
        type=float,
        default=None,
        help="Override minimum EV threshold",
    )
    parser.add_argument(
        "--ev-floor",
        type=float,
        default=None,
        help="Override EV floor for TOP_N/HYBRID",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Override top N per week",
    )
    parser.add_argument(
        "--max-bets",
        type=int,
        default=None,
        help="Override max bets per week",
    )

    # Phase 1 routing (weeks 1-3 only)
    parser.add_argument(
        "--phase1-policy",
        type=str,
        default=DEFAULT_PHASE1_POLICY,
        choices=list(ALLOWED_PHASE1_POLICIES),
        help=f"Phase 1 (weeks 1-3) calibration policy. "
             f"'weighted' uses blended calibration with conservative constraints, "
             f"'phase1_only' uses Phase 1 data only, "
             f"'skip' disables List A for weeks 1-3. "
             f"(default: {DEFAULT_PHASE1_POLICY})",
    )
    parser.add_argument(
        "--phase1-stake-multiplier",
        type=float,
        default=DEFAULT_PHASE1_STAKE_MULTIPLIER,
        help=f"Stake multiplier for Phase 1 (weeks 1-3). "
             f"0.5 = half stakes, 1.0 = full stakes. (default: {DEFAULT_PHASE1_STAKE_MULTIPLIER})",
    )

    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: use historical data, don't log bets",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip writing to bet log CSV",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine year
    year = args.year or datetime.now().year

    # Settlement mode
    if args.settle:
        if args.week is None:
            print("ERROR: --week is required for settlement mode")
            sys.exit(1)

        result = settle_bets(year, args.week)
        print_settlement_summary(result, year, args.week)
        return

    # Recommendation mode
    week = args.week
    if week is None:
        # Auto-detect: This would query the API for current week
        # For now, default to a reasonable test value
        print("ERROR: --week is required (auto-detect not yet implemented)")
        sys.exit(1)

    # Build base config from preset
    config = get_production_config(
        preset=args.preset,
        ev_min=args.ev_min,
        ev_floor=args.ev_floor,
        top_n_per_week=args.top_n,
        max_bets_per_week=args.max_bets,
    )

    # Get phase-specific config
    phase_config = get_phase_config(
        week=week,
        phase1_policy=args.phase1_policy,
        phase1_stake_multiplier=args.phase1_stake_multiplier,
    )

    logger.info(f"Using config: {config_to_label(config)}")
    logger.info(f"Phase {phase_config.phase}: calibration={phase_config.calibration_name}, "
                f"stake_mult={phase_config.stake_multiplier}")

    # Generate recommendations
    list_a_bets, list_b_bets, summary = generate_recommendations(
        year=year,
        week=week,
        config=config,
        preset_name=args.preset,
        phase_config=phase_config,
        dry_run=args.dry_run,
    )

    # Print summary
    print_weekly_summary(list_a_bets, list_b_bets, summary)

    # Log bets (unless dry-run or --no-log)
    if not args.dry_run and not args.no_log:
        all_bets = list_a_bets + list_b_bets
        n_appended = append_bets_to_log(all_bets, year, week)
        print(f"  Logged {n_appended} new bets to {get_log_path(year)}")
    elif args.dry_run:
        print("  [DRY RUN - bets not logged]")


if __name__ == "__main__":
    main()
