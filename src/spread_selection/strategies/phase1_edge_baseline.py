"""Phase 1 Edge Baseline Strategy with optional HYBRID_VETO_2.

A Phase 1-only edge-based betting strategy that:
- Selects bets where |edge_jp| >= 5.0 (using OPEN lines)
- Optionally applies HYBRID_VETO_2 overlay (default OFF)

This is LIST B: PHASE1_EDGE, separate from the EV-based LIST A.

HYBRID_VETO_2 Rule (when enabled):
VETO if ALL of:
  - oppose is True (SP+ and JP+ disagree)
  - |sp_edge| >= sp_oppose_min (default 2.0)
  - jp_band_low <= |jp_edge| < jp_band_high (default 5.0 <= |jp_edge| < 8.0)

NEVER veto if |jp_edge| >= jp_band_high (high-conviction bets protected).

Based on backtest validation (2026-02-13):
- HYBRID_VETO_2 passes 2025 guardrail (vetoed 55.6% < kept 60.2%)
- Removes bad bets (vetoed ATS 43.5% vs kept 50.3%)
- ROI improvement: -3.5% → -2.7%

Sign Conventions (Vegas convention throughout):
- All spreads: negative = home favored
- edge_pts = jp_spread - vegas_spread
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Phase1EdgeResult(Enum):
    """Result of Phase 1 edge evaluation."""
    SELECTED = "selected"           # Passed edge threshold (and veto if enabled)
    REJECTED_EDGE = "rejected_edge" # Did not meet edge threshold
    VETOED = "vetoed"              # Met edge but vetoed by HYBRID_VETO_2
    NOT_PHASE1 = "not_phase1"      # Week not in Phase 1


@dataclass
class Phase1EdgeVetoConfig:
    """Configuration for HYBRID_VETO_2 overlay.

    Default OFF - must be explicitly enabled.
    """
    # Master enable flag
    enabled: bool = False

    # SP+ opposition threshold
    sp_oppose_min: float = 2.0

    # JP+ edge band for veto eligibility (marginal bets only)
    jp_band_low: float = 5.0
    jp_band_high: float = 8.0

    def should_veto(
        self,
        jp_edge_abs: float,
        sp_edge_abs: Optional[float],
        is_oppose: bool,
    ) -> tuple[bool, str]:
        """Check if bet should be vetoed.

        Args:
            jp_edge_abs: Absolute JP+ edge
            sp_edge_abs: Absolute SP+ edge (None if SP+ unavailable)
            is_oppose: True if SP+ and JP+ disagree on side

        Returns:
            (should_veto, reason) tuple
        """
        if not self.enabled:
            return False, ""

        if sp_edge_abs is None:
            return False, ""  # No SP+ data, cannot veto

        # Check if in marginal band
        in_band = self.jp_band_low <= jp_edge_abs < self.jp_band_high
        if not in_band:
            # High-conviction bets (>= jp_band_high) are NEVER vetoed
            return False, ""

        # Check if strongly opposed
        if is_oppose and sp_edge_abs >= self.sp_oppose_min:
            return True, f"HYBRID_VETO_2: oppose AND |sp_edge|={sp_edge_abs:.1f}>=2.0 AND |jp_edge|={jp_edge_abs:.1f} in [5,8)"

        return False, ""


@dataclass
class Phase1EdgeBaselineConfig:
    """Configuration for Phase 1 Edge Baseline strategy (LIST B).

    Defaults are set for automatic emission in Phase 1.
    """
    # Weeks to apply (Phase 1 only)
    weeks: list[int] = field(default_factory=lambda: [1, 2, 3])

    # JP+ edge threshold for candidacy
    jp_edge_min: float = 5.0

    # Line type to use (OPEN for Phase 1)
    line_type: str = "open"

    # Optional HYBRID_VETO_2 overlay (default OFF)
    veto_config: Optional[Phase1EdgeVetoConfig] = None

    def should_apply(self, week: int) -> bool:
        """Check if strategy should be applied for given week."""
        return week in self.weeks


@dataclass
class Phase1EdgeRecommendation:
    """A betting recommendation from the Phase 1 Edge Baseline strategy.

    All spreads in Vegas convention (negative = home favored).
    """
    # Game identification
    game_id: int
    season: int
    week: int
    home_team: str
    away_team: str

    # Spreads (Vegas convention)
    jp_spread: float
    vegas_spread: float  # OPEN line
    edge_pts: float      # Signed (negative = HOME)
    edge_abs: float      # Absolute value

    # Bet side
    bet_side: str  # "HOME" or "AWAY"
    bet_team: str

    # SP+ data (for veto evaluation and monitoring)
    sp_spread: Optional[float] = None
    sp_edge_pts: Optional[float] = None
    sp_edge_abs: Optional[float] = None
    sp_side: Optional[str] = None
    is_oppose: bool = False

    # Veto fields (always present)
    veto_applied: bool = False
    veto_reason: str = ""

    # Result status
    result: Phase1EdgeResult = Phase1EdgeResult.SELECTED

    # Strategy metadata (required by schema)
    list_family: str = "PHASE1_EDGE"
    list_name: str = "EDGE_BASELINE"
    selection_basis: str = "EDGE"
    is_official_engine: bool = False
    execution_default: bool = False
    line_type: str = "open"
    rationale: str = ""

    # Config used
    jp_edge_min: float = 5.0


def _compute_side(edge_pts: float) -> str:
    """Determine which side the edge favors."""
    if edge_pts < 0:
        return "HOME"
    elif edge_pts > 0:
        return "AWAY"
    else:
        return "NO_BET"


def evaluate_game_edge_baseline(
    game_id: int,
    season: int,
    week: int,
    home_team: str,
    away_team: str,
    jp_spread: float,
    vegas_spread_open: float,
    sp_spread: Optional[float],
    config: Phase1EdgeBaselineConfig,
) -> Optional[Phase1EdgeRecommendation]:
    """Evaluate a single game for Phase 1 Edge Baseline selection.

    Args:
        game_id: Game identifier
        season: Year
        week: Week number
        home_team: Home team name
        away_team: Away team name
        jp_spread: JP+ spread (Vegas convention)
        vegas_spread_open: Vegas OPEN spread
        sp_spread: SP+ spread (Vegas convention, optional)
        config: Strategy configuration

    Returns:
        Phase1EdgeRecommendation if game meets edge threshold (may be vetoed),
        None if doesn't meet threshold
    """
    # Compute edge
    edge_pts = jp_spread - vegas_spread_open
    edge_abs = abs(edge_pts)

    # Check edge threshold
    if edge_abs < config.jp_edge_min:
        return None

    # Determine bet side
    jp_side = _compute_side(edge_pts)
    if jp_side == "NO_BET":
        return None

    bet_team = home_team if jp_side == "HOME" else away_team

    # Compute SP+ fields (for monitoring and veto)
    sp_edge_pts = None
    sp_edge_abs = None
    sp_side = None
    is_oppose = False

    if sp_spread is not None:
        sp_edge_pts = sp_spread - vegas_spread_open
        sp_edge_abs = abs(sp_edge_pts)
        sp_side = _compute_side(sp_edge_pts)
        is_oppose = (jp_side != sp_side) and (sp_side != "NO_BET")

    # Check for veto
    veto_applied = False
    veto_reason = ""
    result = Phase1EdgeResult.SELECTED
    list_name = "EDGE_BASELINE"

    if config.veto_config is not None:
        veto_applied, veto_reason = config.veto_config.should_veto(
            edge_abs, sp_edge_abs, is_oppose
        )
        if veto_applied:
            result = Phase1EdgeResult.VETOED
            list_name = "EDGE_HYBRID_VETO_2"  # Mark that veto logic was applied

    # Build rationale
    rationale = f"|edge|>={config.jp_edge_min} open"
    if veto_applied:
        rationale += " [VETOED]"

    return Phase1EdgeRecommendation(
        game_id=game_id,
        season=season,
        week=week,
        home_team=home_team,
        away_team=away_team,
        jp_spread=jp_spread,
        vegas_spread=vegas_spread_open,
        edge_pts=edge_pts,
        edge_abs=edge_abs,
        bet_side=jp_side,
        bet_team=bet_team,
        sp_spread=sp_spread,
        sp_edge_pts=sp_edge_pts,
        sp_edge_abs=sp_edge_abs,
        sp_side=sp_side,
        is_oppose=is_oppose,
        veto_applied=veto_applied,
        veto_reason=veto_reason,
        result=result,
        list_name=list_name,
        rationale=rationale,
        jp_edge_min=config.jp_edge_min,
    )


def evaluate_slate_edge_baseline(
    slate_df: pd.DataFrame,
    sp_spreads: dict[int, float],
    config: Phase1EdgeBaselineConfig,
) -> tuple[list[Phase1EdgeRecommendation], list[Phase1EdgeRecommendation], list[Phase1EdgeRecommendation]]:
    """Evaluate a full slate for Phase 1 Edge Baseline.

    Args:
        slate_df: DataFrame with game_id, season, week, home_team, away_team,
                  jp_spread, spread_open (or vegas_spread)
        sp_spreads: Dict mapping game_id -> SP+ spread (Vegas convention)
        config: Strategy configuration

    Returns:
        (selected, vetoed, all_candidates) tuple of recommendation lists
    """
    # Determine Vegas open column
    if "spread_open" in slate_df.columns:
        vegas_col = "spread_open"
    elif "vegas_spread" in slate_df.columns:
        vegas_col = "vegas_spread"
    else:
        raise ValueError("slate_df must have 'spread_open' or 'vegas_spread' column")

    # Determine season column
    if "season" not in slate_df.columns:
        if "year" in slate_df.columns:
            slate_df = slate_df.copy()
            slate_df["season"] = slate_df["year"]
        else:
            raise ValueError("slate_df must have 'season' or 'year' column")

    selected = []
    vetoed = []
    all_candidates = []

    for _, row in slate_df.iterrows():
        game_id = row["game_id"]  # Keep as-is (can be int or str)
        week = int(row["week"])

        # Get Vegas OPEN spread
        vegas_open = row[vegas_col]
        if pd.isna(vegas_open):
            continue

        # Get JP+ spread (Vegas convention: negative = home favored)
        if "jp_spread" in row:
            jp_spread = row["jp_spread"]
        elif "predicted_spread" in row:
            # predicted_spread is in INTERNAL convention (positive = home favored)
            # Convert to Vegas convention by negating
            # See CLAUDE.md Sign Conventions and calibration.py load_and_normalize_game_data()
            jp_spread = -row["predicted_spread"]
            logger.debug(f"game_id={game_id}: using predicted_spread fallback (internal->Vegas conversion)")
        else:
            continue

        if pd.isna(jp_spread):
            continue

        # Get SP+ spread
        sp_spread = sp_spreads.get(game_id)

        rec = evaluate_game_edge_baseline(
            game_id=game_id,
            season=int(row["season"]),
            week=week,
            home_team=row["home_team"],
            away_team=row["away_team"],
            jp_spread=jp_spread,
            vegas_spread_open=vegas_open,
            sp_spread=sp_spread,
            config=config,
        )

        if rec is not None:
            all_candidates.append(rec)
            if rec.result == Phase1EdgeResult.SELECTED:
                selected.append(rec)
            elif rec.result == Phase1EdgeResult.VETOED:
                vetoed.append(rec)

    # Log summary
    n_selected = len(selected)
    n_vetoed = len(vetoed)
    n_candidates = len(all_candidates)
    logger.info(f"Phase1 Edge Baseline: {n_candidates} candidates, {n_selected} selected, {n_vetoed} vetoed")

    return selected, vetoed, all_candidates


def recommendations_to_dataframe(recs: list[Phase1EdgeRecommendation]) -> pd.DataFrame:
    """Convert recommendations to DataFrame with all required fields."""
    if not recs:
        return pd.DataFrame()

    records = []
    for rec in recs:
        records.append({
            "game_id": rec.game_id,
            "season": rec.season,
            "week": rec.week,
            "home_team": rec.home_team,
            "away_team": rec.away_team,
            "jp_spread": rec.jp_spread,
            "vegas_spread": rec.vegas_spread,
            "edge_pts": rec.edge_pts,
            "edge_abs": rec.edge_abs,
            "bet_side": rec.bet_side,
            "bet_team": rec.bet_team,
            "sp_spread": rec.sp_spread,
            "sp_edge_pts": rec.sp_edge_pts,
            "sp_edge_abs": rec.sp_edge_abs,
            "sp_side": rec.sp_side,
            "is_oppose": rec.is_oppose,
            "veto_applied": rec.veto_applied,
            "veto_reason": rec.veto_reason,
            "result": rec.result.value,
            "list_family": rec.list_family,
            "list_name": rec.list_name,
            "selection_basis": rec.selection_basis,
            "is_official_engine": rec.is_official_engine,
            "execution_default": rec.execution_default,
            "line_type": rec.line_type,
            "rationale": rec.rationale,
            "jp_edge_min": rec.jp_edge_min,
        })

    return pd.DataFrame(records)


def summarize_recommendations(
    selected: list[Phase1EdgeRecommendation],
    vetoed: list[Phase1EdgeRecommendation],
    all_candidates: list[Phase1EdgeRecommendation],
) -> dict:
    """Generate summary statistics for recommendations."""
    return {
        "n_candidates": len(all_candidates),
        "n_selected": len(selected),
        "n_vetoed": len(vetoed),
        "retention_pct": 100 * len(selected) / len(all_candidates) if all_candidates else 0,
        "avg_edge_abs_selected": np.mean([r.edge_abs for r in selected]) if selected else None,
        "avg_edge_abs_vetoed": np.mean([r.edge_abs for r in vetoed]) if vetoed else None,
    }
