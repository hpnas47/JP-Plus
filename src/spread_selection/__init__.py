"""Calibrated spread betting selection layer.

This module provides calibrated probability estimation for spread bets,
EV-based selection, and strategy comparison tools.

V1 Scope:
- Logistic regression P(cover) from walk-forward out-of-fold predictions
- EV calculation assuming constant -110 juice, no push modeling
- Strategy comparison: fixed threshold vs EV-based selection

V2 Additions:
- Push rate estimation with tick-based keys for float safety
- Full selection engine (BetRecommendation, evaluate_game, evaluate_slate)
- Push-aware walk-forward backtest (no leakage for both calibration AND push rates)
- Stratified diagnostics (vegas_total terciles, spread buckets, week buckets)

V3 Additions:
- Policy layers (post-selection filters) in policies/
- Phase1SPGate: SP+ confirmation gating for EV-based selections (default OFF)

V4 Additions:
- Separate strategies (distinct bet lists) in strategies/
- Phase1EdgeBaseline: Edge-based selection for Phase 1 (|edge| >= 5.0 vs OPEN)
- Produces LIST B separate from EV-based LIST A (no merging)

V5 Additions (2026 Production):
- FBS-only filter: Exclude FCS games from bet recommendations (default ON)
- Distinct lists: Phase 1 Edge excludes games already in Primary Engine (default ON)
- Display formatters: format_primary_engine_table, format_phase1_edge_table, format_week_summary

Calibration Modes:
- PRIMARY (ROLLING_2): training_window_seasons=2, realistic volume
- ULTRA (INCLUDE_ALL): training_window_seasons=None, high conviction

Production Output:
- LIST A (PRIMARY): EV-based recommendations from calibrated selection engine
- LIST B (PHASE1_EDGE): Edge-based Phase 1 bets NOT in Primary (distinct list)
"""

from .calibration import (
    # Core classes
    CalibrationResult,
    WalkForwardResult,
    # V2: Push rates
    PushRates,
    KEY_TICKS,
    # Data loading
    load_and_normalize_game_data,
    # Calibration
    calibrate_cover_probability,
    predict_cover_probability,
    walk_forward_validate,
    # V2: Push rate functions
    estimate_push_rates,
    get_push_probability,
    get_push_probability_vectorized,
    # EV calculation
    breakeven_prob,
    calculate_ev,
    calculate_ev_vectorized,
    # Diagnostics
    diagnose_calibration,
    diagnose_fold_stability,
    stratified_diagnostics,
    print_stratified_diagnostics,
    # Mode configuration
    get_calibration_label,
    CALIBRATION_MODES,
    DEFAULT_CALIBRATION_MODE,
    DEFAULT_TRAINING_WINDOW_SEASONS,
)

from .selection import (
    BetRecommendation,
    evaluate_game,
    evaluate_slate,
    summarize_slate,
    calculate_ev_with_push,
)

# V3: Policy layers (post-selection filters)
from .policies import (
    Phase1SPGateConfig,
    Phase1SPGateResult,
    SPGateCategory,
    SPGateMode,
    apply_phase1_sp_gate,
    fetch_sp_spreads_vegas,
)

# V4: Separate strategies (distinct bet lists, NOT merged with EV)
from .strategies import (
    Phase1EdgeBaselineConfig,
    Phase1EdgeVetoConfig,
    Phase1EdgeRecommendation,
    Phase1EdgeResult,
    evaluate_game_edge_baseline,
    evaluate_slate_edge_baseline,
    recommendations_to_dataframe as edge_baseline_to_dataframe,
    summarize_recommendations as summarize_edge_baseline,
)

# V5: Display formatters
from .run_selection import (
    format_primary_engine_table,
    format_phase1_edge_table,
    format_week_summary,
    TEAM_ABBREVS,
)

__all__ = [
    # Core classes
    "CalibrationResult",
    "WalkForwardResult",
    # V2: Push rates
    "PushRates",
    "KEY_TICKS",
    # Data loading
    "load_and_normalize_game_data",
    # Calibration
    "calibrate_cover_probability",
    "predict_cover_probability",
    "walk_forward_validate",
    # V2: Push rate functions
    "estimate_push_rates",
    "get_push_probability",
    "get_push_probability_vectorized",
    # EV calculation
    "breakeven_prob",
    "calculate_ev",
    "calculate_ev_vectorized",
    "calculate_ev_with_push",
    # Diagnostics
    "diagnose_calibration",
    "diagnose_fold_stability",
    "stratified_diagnostics",
    "print_stratified_diagnostics",
    # Mode configuration
    "get_calibration_label",
    "CALIBRATION_MODES",
    "DEFAULT_CALIBRATION_MODE",
    "DEFAULT_TRAINING_WINDOW_SEASONS",
    # V2: Selection engine
    "BetRecommendation",
    "evaluate_game",
    "evaluate_slate",
    "summarize_slate",
    # V3: Policy layers
    "Phase1SPGateConfig",
    "Phase1SPGateResult",
    "SPGateCategory",
    "SPGateMode",
    "apply_phase1_sp_gate",
    "fetch_sp_spreads_vegas",
    # V4: Separate strategies
    "Phase1EdgeBaselineConfig",
    "Phase1EdgeVetoConfig",
    "Phase1EdgeRecommendation",
    "Phase1EdgeResult",
    "evaluate_game_edge_baseline",
    "evaluate_slate_edge_baseline",
    "edge_baseline_to_dataframe",
    "summarize_edge_baseline",
    # V5: Display formatters
    "format_primary_engine_table",
    "format_phase1_edge_table",
    "format_week_summary",
    "TEAM_ABBREVS",
]
