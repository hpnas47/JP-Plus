"""Calibrated spread betting selection layer.

This module provides calibrated probability estimation for spread bets,
EV-based selection, and strategy comparison tools.

V1 Scope:
- Logistic regression P(cover) from walk-forward out-of-fold predictions
- EV calculation assuming constant -110 juice, no push modeling
- Strategy comparison: fixed threshold vs EV-based selection
"""

from .calibration import (
    CalibrationResult,
    load_and_normalize_game_data,
    calibrate_cover_probability,
    predict_cover_probability,
    walk_forward_validate,
    breakeven_prob,
    calculate_ev,
)

__all__ = [
    "CalibrationResult",
    "load_and_normalize_game_data",
    "calibrate_cover_probability",
    "predict_cover_probability",
    "walk_forward_validate",
    "breakeven_prob",
    "calculate_ev",
]
