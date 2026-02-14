"""Phase 1 SP+ Gate Policy Layer.

This module implements a gating mechanism that uses SP+ pregame predictions
to filter EV-based betting recommendations in weeks 0-3 (Phase 1).

This is a POST-SELECTION policy filter:
- Applied AFTER BetRecommendation objects are produced with EV/confidence
- Does NOT modify calibration, p_cover computation, push rates, or EV logic
- Only filters the final BET list based on SP+ agreement

Sign Conventions (Vegas convention throughout):
- All spreads: negative = home favored
- edge_pts = jp_spread - vegas_spread
  - Negative edge = JP+ likes HOME more than Vegas
  - Positive edge = JP+ likes AWAY more than Vegas
- SP+ from CFBD API: already in Vegas convention (negative = home favored)

Gate Categories:
- CONFIRM: SP+ and JP+ agree on side AND |sp_edge| >= sp_edge_min
- NEUTRAL: SP+ and JP+ agree on side BUT |sp_edge| < sp_edge_min
- OPPOSE: SP+ and JP+ disagree on side
- MISSING: No SP+ prediction available for game
- NO_BET_SP: SP+ has zero edge (no directional signal)

Gate Modes:
- confirm_only: Keep only CONFIRM games
- veto_opposes: Keep CONFIRM + NEUTRAL + MISSING, drop OPPOSE
- confirm_or_neutral: Same as veto_opposes

Candidate Basis:
- "ev": Candidates are games where EV >= threshold and confidence != PASS
- "edge": Candidates are games where |edge_pts| >= jp_edge_min (original research)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from src.api.cfbd_client import CFBDClient

logger = logging.getLogger(__name__)


class SPGateCategory(Enum):
    """SP+ agreement categories."""
    CONFIRM = "confirm"  # Same side AND |sp_edge| >= threshold
    NEUTRAL = "neutral"  # Same side but |sp_edge| < threshold
    OPPOSE = "oppose"    # Opposite side
    MISSING = "missing"  # No SP+ prediction available
    NO_BET_SP = "no_bet_sp"  # SP+ edge is exactly 0


class SPGateMode(Enum):
    """SP+ gating modes."""
    CONFIRM_ONLY = "confirm_only"
    VETO_OPPOSES = "veto_opposes"
    CONFIRM_OR_NEUTRAL = "confirm_or_neutral"


@dataclass
class Phase1SPGateConfig:
    """Configuration for Phase 1 SP+ Gate.

    All defaults are OFF to preserve existing behavior.
    """
    # Master enable flag
    enabled: bool = False

    # Gate mode
    mode: str = "confirm_only"

    # Thresholds
    sp_edge_min: float = 2.0  # Min |SP+ edge| for CONFIRM
    jp_edge_min: float = 5.0  # Min |JP+ edge| for candidate (when basis="edge")

    # Weeks to apply gate (includes CFB week 0)
    weeks: list[int] = field(default_factory=lambda: [0, 1, 2, 3])

    # Missing SP+ behavior: "treat_neutral" or "reject"
    missing_sp_behavior: str = "treat_neutral"

    # Candidate basis: "ev" or "edge"
    # "ev": Candidates are EV-based BET recommendations
    # "edge": Candidates are games with |edge_pts| >= jp_edge_min
    candidate_basis: str = "ev"

    @property
    def gate_mode(self) -> SPGateMode:
        return SPGateMode(self.mode)

    def should_apply(self, week: int) -> bool:
        """Check if gate should be applied for given week."""
        return self.enabled and week in self.weeks


@dataclass
class Phase1SPGateResult:
    """Result of SP+ gating for a single game.

    All spreads in Vegas convention (negative = home favored).
    """
    game_id: int

    # SP+ data (Vegas convention)
    sp_spread: Optional[float]  # SP+ spread (negative = home favored)
    sp_edge_pts: Optional[float]  # sp_spread - vegas_spread (signed)
    sp_edge_abs: Optional[float]  # |sp_edge_pts|

    # JP+ data (Vegas convention)
    jp_spread: float  # JP+ spread (negative = home favored)
    jp_edge_pts: float  # jp_spread - vegas_spread (signed)
    vegas_spread: float

    # Sides
    jp_side: str  # "HOME", "AWAY", or "NO_BET"
    sp_side: Optional[str]  # "HOME", "AWAY", "NO_BET", or None (missing)

    # Gate result
    category: SPGateCategory
    passed_gate: bool
    gate_reason: str


def _compute_side(edge_pts: float) -> str:
    """Determine which side the edge favors.

    Args:
        edge_pts: signed edge (negative = favors HOME, positive = favors AWAY)

    Returns:
        "HOME", "AWAY", or "NO_BET"
    """
    if edge_pts < 0:
        return "HOME"
    elif edge_pts > 0:
        return "AWAY"
    else:
        return "NO_BET"


def fetch_sp_spreads_vegas(
    client: CFBDClient,
    year: int,
    weeks: list[int],
) -> dict[int, float]:
    """Fetch SP+ pregame predictions in Vegas convention.

    Args:
        client: CFBD API client
        year: Season year
        weeks: List of week numbers to fetch

    Returns:
        Dict mapping game_id -> SP+ spread (Vegas convention: negative = home favored)
    """
    sp_spreads = {}

    for week in weeks:
        try:
            probs = client.get_pregame_win_probabilities(year, week=week)
            for p in probs:
                if p.spread is not None:
                    # CFBD returns Vegas convention (negative = home favored)
                    # Keep as-is
                    sp_spreads[p.game_id] = p.spread
        except Exception as e:
            logger.warning(f"Could not fetch SP+ for {year} week {week}: {e}")

    logger.info(f"Fetched {len(sp_spreads)} SP+ spreads (Vegas conv) for weeks {weeks}")
    return sp_spreads


def _categorize_agreement(
    jp_side: str,
    sp_side: Optional[str],
    sp_edge_abs: Optional[float],
    sp_edge_min: float,
) -> tuple[SPGateCategory, str]:
    """Categorize SP+ agreement with JP+.

    Returns:
        (category, reason) tuple
    """
    # Handle missing SP+
    if sp_side is None:
        return SPGateCategory.MISSING, "No SP+ prediction available"

    # Handle NO_BET cases
    if jp_side == "NO_BET":
        return SPGateCategory.NEUTRAL, "JP+ has zero edge"

    if sp_side == "NO_BET":
        return SPGateCategory.NO_BET_SP, "SP+ has zero edge"

    # Check agreement
    if jp_side == sp_side:
        # Same side - check edge magnitude
        if sp_edge_abs is not None and sp_edge_abs >= sp_edge_min:
            return SPGateCategory.CONFIRM, f"SP+ confirms {jp_side} with {sp_edge_abs:.1f}pt edge"
        else:
            edge_str = f"{sp_edge_abs:.1f}" if sp_edge_abs is not None else "N/A"
            return SPGateCategory.NEUTRAL, f"SP+ agrees on {jp_side} but edge {edge_str} < {sp_edge_min}"
    else:
        # Opposite sides
        return SPGateCategory.OPPOSE, f"SP+ opposes: JP+ likes {jp_side}, SP+ likes {sp_side}"


def _gate_passes(
    category: SPGateCategory,
    mode: SPGateMode,
    missing_behavior: str,
) -> tuple[bool, str]:
    """Determine if game passes the gate.

    Returns:
        (passes, reason) tuple
    """
    if mode == SPGateMode.CONFIRM_ONLY:
        if category == SPGateCategory.CONFIRM:
            return True, "CONFIRM: SP+ confirms bet"
        elif category == SPGateCategory.MISSING:
            if missing_behavior == "treat_neutral":
                return False, "REJECT (confirm_only): Missing SP+ treated as non-confirm"
            else:
                return False, "REJECT (confirm_only): Missing SP+ rejected"
        else:
            return False, f"REJECT (confirm_only): Category is {category.value}"

    elif mode in (SPGateMode.VETO_OPPOSES, SPGateMode.CONFIRM_OR_NEUTRAL):
        if category == SPGateCategory.OPPOSE:
            return False, "REJECT (veto_opposes): SP+ opposes bet"
        elif category == SPGateCategory.MISSING:
            if missing_behavior == "treat_neutral":
                return True, "PASS: Missing SP+ treated as neutral"
            else:
                return False, "REJECT: Missing SP+ rejected"
        else:
            return True, f"PASS: Category is {category.value}"

    # Unknown mode - pass by default
    return True, "PASS: Unknown mode"


def evaluate_single_game(
    game_id: int,
    jp_spread: float,
    vegas_spread: float,
    sp_spread: Optional[float],
    config: Phase1SPGateConfig,
) -> Phase1SPGateResult:
    """Evaluate SP+ gate for a single game.

    All spreads in Vegas convention (negative = home favored).

    Args:
        game_id: Unique game identifier
        jp_spread: JP+ spread (Vegas convention)
        vegas_spread: Vegas spread
        sp_spread: SP+ spread (Vegas convention) or None if missing
        config: Gate configuration

    Returns:
        Phase1SPGateResult with gate decision and metadata
    """
    # Compute edges
    jp_edge_pts = jp_spread - vegas_spread
    jp_side = _compute_side(jp_edge_pts)

    if sp_spread is not None:
        sp_edge_pts = sp_spread - vegas_spread
        sp_edge_abs = abs(sp_edge_pts)
        sp_side = _compute_side(sp_edge_pts)
    else:
        sp_edge_pts = None
        sp_edge_abs = None
        sp_side = None

    # Categorize
    category, cat_reason = _categorize_agreement(
        jp_side, sp_side, sp_edge_abs, config.sp_edge_min
    )

    # Check gate
    passed, gate_reason = _gate_passes(category, config.gate_mode, config.missing_sp_behavior)

    return Phase1SPGateResult(
        game_id=game_id,
        sp_spread=sp_spread,
        sp_edge_pts=sp_edge_pts,
        sp_edge_abs=sp_edge_abs,
        jp_spread=jp_spread,
        jp_edge_pts=jp_edge_pts,
        vegas_spread=vegas_spread,
        jp_side=jp_side,
        sp_side=sp_side,
        category=category,
        passed_gate=passed,
        gate_reason=gate_reason,
    )


def apply_phase1_sp_gate(
    recommendations: list,
    sp_spreads: dict[int, float],
    config: Phase1SPGateConfig,
    week: int,
) -> tuple[list, list[Phase1SPGateResult]]:
    """Apply Phase 1 SP+ Gate to BetRecommendation list.

    This is a POST-SELECTION filter that does NOT modify calibration or EV.
    It only filters which bets to actually place based on SP+ confirmation.

    Args:
        recommendations: List of BetRecommendation objects
        sp_spreads: Dict mapping game_id -> SP+ spread (Vegas convention)
        config: Gate configuration
        week: Current week number

    Returns:
        (filtered_recommendations, all_gate_results) tuple
        - filtered_recommendations: BetRecommendation list with non-passing filtered out
        - all_gate_results: Phase1SPGateResult for EVERY game (for audit trail)
    """
    # Check if gate should be applied
    if not config.should_apply(week):
        # Return all recommendations unchanged, with empty gate results
        return recommendations, []

    all_gate_results = []
    passed_recommendations = []

    # Determine which recommendations are candidates
    for rec in recommendations:
        game_id = int(rec.game_id) if hasattr(rec, 'game_id') else int(rec.get('game_id', 0))

        # Extract spreads (already in Vegas convention from selection.py)
        jp_spread = rec.jp_spread if hasattr(rec, 'jp_spread') else rec.get('jp_spread', 0)
        vegas_spread = rec.vegas_spread if hasattr(rec, 'vegas_spread') else rec.get('vegas_spread', 0)

        # Get SP+ spread
        sp_spread = sp_spreads.get(game_id)

        # Evaluate gate
        result = evaluate_single_game(
            game_id=game_id,
            jp_spread=jp_spread,
            vegas_spread=vegas_spread,
            sp_spread=sp_spread,
            config=config,
        )
        all_gate_results.append(result)

        # Check if this is a candidate for gating
        confidence = rec.confidence if hasattr(rec, 'confidence') else rec.get('confidence', '')

        if config.candidate_basis == "ev":
            # Candidates are BET/MED/HIGH (not PASS/NO_BET)
            is_candidate = confidence not in ("PASS", "NO_BET")
        else:  # "edge"
            edge_abs = rec.edge_abs if hasattr(rec, 'edge_abs') else rec.get('edge_abs', 0)
            is_candidate = abs(edge_abs) >= config.jp_edge_min

        # Non-candidates pass through unchanged
        if not is_candidate:
            passed_recommendations.append(rec)
            continue

        # Candidates must pass gate
        if result.passed_gate:
            passed_recommendations.append(rec)

    # Log summary
    n_total = len(recommendations)
    n_candidates = sum(
        1 for rec in recommendations
        if (rec.confidence if hasattr(rec, 'confidence') else rec.get('confidence', ''))
           not in ("PASS", "NO_BET")
    ) if config.candidate_basis == "ev" else sum(
        1 for rec in recommendations
        if abs(rec.edge_abs if hasattr(rec, 'edge_abs') else rec.get('edge_abs', 0))
           >= config.jp_edge_min
    )

    n_passed = sum(1 for r in all_gate_results if r.passed_gate)
    n_confirm = sum(1 for r in all_gate_results if r.category == SPGateCategory.CONFIRM)
    n_neutral = sum(1 for r in all_gate_results if r.category == SPGateCategory.NEUTRAL)
    n_oppose = sum(1 for r in all_gate_results if r.category == SPGateCategory.OPPOSE)
    n_missing = sum(1 for r in all_gate_results if r.category == SPGateCategory.MISSING)

    logger.info(
        f"SP+ Gate (week {week}, {config.mode}): "
        f"{n_total} games, {n_candidates} candidates -> "
        f"confirm={n_confirm}, neutral={n_neutral}, oppose={n_oppose}, missing={n_missing} -> "
        f"{len(passed_recommendations)} passed"
    )

    return passed_recommendations, all_gate_results


def gate_results_to_dataframe(results: list[Phase1SPGateResult]) -> pd.DataFrame:
    """Convert gate results to DataFrame for reporting.

    Args:
        results: List of Phase1SPGateResult objects

    Returns:
        DataFrame with gate metadata columns
    """
    records = []
    for r in results:
        records.append({
            'game_id': r.game_id,
            'sp_spread': r.sp_spread,
            'sp_edge_pts': r.sp_edge_pts,
            'sp_edge_abs': r.sp_edge_abs,
            'jp_spread': r.jp_spread,
            'jp_edge_pts': r.jp_edge_pts,
            'vegas_spread': r.vegas_spread,
            'jp_side': r.jp_side,
            'sp_side': r.sp_side,
            'sp_gate_category': r.category.value,
            'sp_gate_passed': r.passed_gate,
            'sp_gate_reason': r.gate_reason,
        })
    return pd.DataFrame(records)


def merge_gate_results_to_df(
    df: pd.DataFrame,
    results: list[Phase1SPGateResult],
) -> pd.DataFrame:
    """Merge gate results into existing DataFrame.

    Args:
        df: DataFrame with game_id column
        results: List of Phase1SPGateResult objects

    Returns:
        DataFrame with gate metadata columns added
    """
    if not results:
        # Add placeholder columns
        df = df.copy()
        df['sp_spread'] = None
        df['sp_edge_pts'] = None
        df['sp_edge_abs'] = None
        df['sp_gate_category'] = None
        df['sp_gate_passed'] = True  # Default to pass when gate disabled
        df['sp_gate_reason'] = None
        return df

    gate_df = gate_results_to_dataframe(results)

    # Merge on game_id
    df = df.merge(
        gate_df[['game_id', 'sp_spread', 'sp_edge_pts', 'sp_edge_abs',
                 'sp_gate_category', 'sp_gate_passed', 'sp_gate_reason']],
        on='game_id',
        how='left'
    )

    # Fill NaN for games without gate results
    df['sp_gate_passed'] = df['sp_gate_passed'].fillna(True)

    return df
