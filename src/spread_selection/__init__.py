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

V6 Additions (Totals EV Engine):
- TotalsEVConfig: Configuration for totals betting EV evaluation
- TotalsBetRecommendation: Full recommendation for totals bets
- evaluate_totals_markets: Main interface for evaluating totals bets
- Normal CDF probability model with push-awareness
- Kelly staking with three-outcome formula

Calibration Modes:
- PRIMARY (ROLLING_2): training_window_seasons=2, realistic volume
- ULTRA (INCLUDE_ALL): training_window_seasons=None, high conviction

Production Output:
- LIST A (PRIMARY): EV-based recommendations from calibrated selection engine
- LIST B (PHASE1_EDGE): Edge-based Phase 1 bets NOT in Primary (distinct list)

V8 Additions (Selection Policy System):
- SelectionPolicyConfig: Configuration for bet selection policy
- SelectionPolicy: Enum for policy types (EV_THRESHOLD, TOP_N_PER_WEEK, HYBRID)
- apply_selection_policy: Main policy application function
- compute_selection_metrics: Compute ATS, ROI, and other metrics
- compute_stability_score: Score for ranking configurations
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
    # Artifact loading
    load_spread_calibration_from_artifact,
    get_default_spread_calibration,
    # Phase-aware routing
    get_spread_calibration_for_week,
    PHASE1_WEEKS,
    PHASE2_WEEKS,
    PHASE3_WEEKS,
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
# SP+ gate removed (2026-02-14) - research showed unstable year-to-year results
# See docs/PHASE1_SP_POLICY.md for rationale

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

# V6: Totals EV Engine
from .totals_ev_engine import (
    # Data structures
    TotalMarket,
    TotalsEvent,
    TotalsEVConfig,
    TotalsBetRecommendation,
    MuOverrideFn,
    # Guardrail constants
    GUARDRAIL_OK,
    GUARDRAIL_LOW_TRAIN_GAMES,
    GUARDRAIL_BASELINE_OUT_OF_RANGE,
    GUARDRAIL_DIAGNOSTIC_ONLY_FORCED,
    # Math utilities
    american_to_decimal,
    decimal_to_implied_prob,
    normal_cdf,
    estimate_sigma_from_backtest,
    # Phase 1 baseline blending
    compute_baseline_blend_weight,
    compute_baseline_shift,
    check_guardrails,
    # Phase 2 sigma calibration
    compute_game_reliability,
    compute_calibrated_sigma,
    get_sigma_for_week_bucket,
    get_effective_ev_min,
    # Probability model
    calculate_totals_probabilities,
    # EV + Kelly
    calculate_ev_totals,
    calculate_kelly_stake,
    # Main interface
    evaluate_totals_markets,
    recommendations_to_dataframe as totals_recommendations_to_dataframe,
    summarize_totals_ev,
)

# V7: Totals Calibration Module
from .totals_calibration import (
    # Config
    TotalsCalibrationConfig,
    SigmaEstimate,
    IntervalCoverageResult,
    CalibrationReport,
    # Residual collection
    collect_walk_forward_residuals,
    # Sigma estimators
    estimate_sigma_global,
    estimate_sigma_robust,
    estimate_sigma_by_week_bucket,
    compute_all_sigma_estimates,
    # Coverage
    evaluate_interval_coverage,
    compute_coverage_score,
    # Tuning
    tune_sigma_for_coverage,
    tune_sigma_for_roi,
    compute_week_bucket_multipliers,
    run_full_calibration,
    # Load/Save
    get_calibration_artifact_path,
    save_calibration,
    load_calibration,
    select_calibration_for_runtime,
    # Runtime helpers
    get_sigma_for_game,
)

# V8: Selection Policy System
from .selection_policy import (
    SelectionPolicyConfig,
    SelectionPolicy,
    SelectionResult,
    SelectionMetrics,
    apply_selection_policy,
    compute_selection_metrics,
    compute_stability_score,
    compute_max_drawdown,
    generate_policy_grid,
    config_to_label,
    # Presets
    get_selection_policy_preset,
    configs_match,
    PRESET_CONFIGS,
    ALLOWED_PRESETS,
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
    # Artifact loading
    "load_spread_calibration_from_artifact",
    "get_default_spread_calibration",
    # Phase-aware routing
    "get_spread_calibration_for_week",
    "PHASE1_WEEKS",
    "PHASE2_WEEKS",
    "PHASE3_WEEKS",
    # V2: Selection engine
    "BetRecommendation",
    "evaluate_game",
    "evaluate_slate",
    "summarize_slate",
    # V3: Policy layers (SP+ gate removed 2026-02-14)
    # V4: Separate strategies
    "Phase1EdgeBaselineConfig",
    "Phase1EdgeVetoConfig",
    "Phase1EdgeRecommendation",
    "Phase1EdgeResult",
    "evaluate_game_edge_baseline",
    "evaluate_slate_edge_baseline",
    "edge_baseline_to_dataframe",
    "summarize_edge_baseline",
    # V6: Totals EV Engine
    "TotalMarket",
    "TotalsEvent",
    "TotalsEVConfig",
    "TotalsBetRecommendation",
    "MuOverrideFn",
    # Guardrail constants
    "GUARDRAIL_OK",
    "GUARDRAIL_LOW_TRAIN_GAMES",
    "GUARDRAIL_BASELINE_OUT_OF_RANGE",
    "GUARDRAIL_DIAGNOSTIC_ONLY_FORCED",
    # Math utilities
    "american_to_decimal",
    "decimal_to_implied_prob",
    "normal_cdf",
    "estimate_sigma_from_backtest",
    # Phase 1 baseline blending
    "compute_baseline_blend_weight",
    "compute_baseline_shift",
    "check_guardrails",
    # Phase 2 sigma calibration
    "compute_game_reliability",
    "compute_calibrated_sigma",
    "get_sigma_for_week_bucket",
    "get_effective_ev_min",
    # Probability/EV
    "calculate_totals_probabilities",
    "calculate_ev_totals",
    "calculate_kelly_stake",
    "evaluate_totals_markets",
    "totals_recommendations_to_dataframe",
    "summarize_totals_ev",
    # V7: Totals Calibration Module
    "TotalsCalibrationConfig",
    "SigmaEstimate",
    "IntervalCoverageResult",
    "CalibrationReport",
    "collect_walk_forward_residuals",
    "estimate_sigma_global",
    "estimate_sigma_robust",
    "estimate_sigma_by_week_bucket",
    "compute_all_sigma_estimates",
    "evaluate_interval_coverage",
    "compute_coverage_score",
    "tune_sigma_for_coverage",
    "tune_sigma_for_roi",
    "compute_week_bucket_multipliers",
    "run_full_calibration",
    "get_calibration_artifact_path",
    "save_calibration",
    "load_calibration",
    "select_calibration_for_runtime",
    "get_sigma_for_game",
    # V8: Selection Policy System
    "SelectionPolicyConfig",
    "SelectionPolicy",
    "SelectionResult",
    "SelectionMetrics",
    "apply_selection_policy",
    "compute_selection_metrics",
    "compute_stability_score",
    "compute_max_drawdown",
    "generate_policy_grid",
    "config_to_label",
    # Presets
    "get_selection_policy_preset",
    "configs_match",
    "PRESET_CONFIGS",
    "ALLOWED_PRESETS",
]
