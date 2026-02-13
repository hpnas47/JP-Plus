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

Calibration Modes:
- PRIMARY (ROLLING_2): training_window_seasons=2, realistic volume
- ULTRA (INCLUDE_ALL): training_window_seasons=None, high conviction
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
]
