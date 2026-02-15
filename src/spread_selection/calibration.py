"""Calibrated spread betting selection: P(cover) estimation and EV calculation.

This module provides:
1. Data loading and normalization to Vegas convention
2. Logistic regression calibration for P(cover | edge)
3. Walk-forward validation for out-of-sample estimates
4. EV calculation for bet selection

Sign Convention (Single Convention Throughout):
    All spreads normalized to Vegas convention (negative = home favored):

    | Component          | Convention                              | Example                      |
    |--------------------|----------------------------------------|------------------------------|
    | jp_spread (after)  | Negative = home favored                | -7 = home by 7               |
    | vegas_spread       | Negative = home favored                | -7 = home by 7               |
    | edge_pts           | Negative = JP+ likes HOME more         | -3 = JP+ likes home 3 pts    |
    |                    | Formula: jp_spread - vegas_spread       | JP+=-10, Vegas=-7 → edge=-3  |
    | actual_margin      | Positive = home won                    | +10 = home won by 10         |
    | cover_margin       | Positive = home covered                | +3 = home beat spread by 3   |

V1 Assumptions:
    - Constant juice: -110
    - No push modeling: p_push = 0
    - Pure selection layer: does NOT modify JP+ predictions
    - Both negative and positive American odds are supported in EV/breakeven calculations.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit  # Logistic function
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# ============================================================================
# CALIBRATION MODE CONSTANTS
# ============================================================================
# Primary mode: ROLLING_2 (most recent 2 seasons) - good volume, solid signal
# Ultra mode: INCLUDE_ALL (all prior seasons) - low volume, high conviction

CALIBRATION_MODES = {
    "primary": {
        "training_window_seasons": 2,
        "label": "ROLLING_2",
        "description": "Most recent 2 seasons (realistic volume)",
        "default_min_ev": 0.03,
    },
    "ultra": {
        "training_window_seasons": None,  # All prior seasons
        "label": "INCLUDE_ALL",
        "description": "All prior seasons (high conviction, low volume)",
        "default_min_ev": 0.05,
    },
}

# Default mode for production
DEFAULT_CALIBRATION_MODE = "primary"
DEFAULT_TRAINING_WINDOW_SEASONS = CALIBRATION_MODES["primary"]["training_window_seasons"]

# Shared edge_abs bucket definitions for calibration diagnostics.
# Both diagnose_calibration and stratified_diagnostics must use these.
EDGE_BUCKETS = [
    ("[0,3)", 0, 3),
    ("[3,5)", 3, 5),
    ("[5,7)", 5, 7),
    ("[7,10)", 7, 10),
    ("[10,+)", 10, float("inf")),
]

# ============================================================================
# PUSH RATE CONSTANTS (V2)
# ============================================================================
# Push rate tick-based keys for float safety
# tick = int(round(abs(spread) * 2))
# KEY_TICKS correspond to key integer spreads: 3, 7, 10, 14
KEY_TICKS = [6, 14, 20, 28]  # spread * 2


@dataclass
class PushRates:
    """Push rate estimates by tick bucket.

    Ticks are computed as: tick = int(round(abs(spread) * 2))
    - Odd ticks (half-point spreads like 3.5) => p_push = 0.0 by definition
    - Even ticks (integer spreads) => use empirical rate or default

    Attributes:
        tick_rates: tick -> p_push for key ticks and integer ticks with n>=min_games
        default_even: Fallback for even ticks (integer spreads) with insufficient data
        default_overall: Overall average push rate (rarely used)
        n_games_by_tick: Sample size per tick (for diagnostics)
        years_trained: Years used for estimation (for auditability)
    """

    tick_rates: dict[int, float]
    default_even: float
    default_overall: float
    n_games_by_tick: dict[int, int]
    years_trained: list[int]

    def __repr__(self) -> str:
        key_rates = {t: self.tick_rates.get(t, self.default_even) for t in KEY_TICKS}
        return (
            f"PushRates(key_ticks={key_rates}, default_even={self.default_even:.4f}, "
            f"n_ticks={len(self.tick_rates)}, years={self.years_trained})"
        )


def estimate_push_rates(
    historical_games: pd.DataFrame,
    min_games_per_tick: int = 50,
) -> PushRates:
    """Estimate push rates from historical games.

    Args:
        historical_games: DataFrame with vegas_spread, push columns
        min_games_per_tick: Minimum games to get own bucket (except KEY_TICKS)

    Returns:
        PushRates with tick-based lookup

    Logic:
        1. Convert vegas_spread to tick = int(round(abs(spread) * 2))
        2. Odd ticks (half-point spreads) => p_push = 0.0 by definition
        3. KEY_TICKS always get own bucket regardless of sample size
        4. Non-key even ticks with n >= min_games_per_tick get own bucket
        5. Non-key even ticks with n < min_games_per_tick use default_even
        6. Compute default_even as weighted avg of all even-tick games
        7. Compute default_overall as fallback
    """
    if "vegas_spread" not in historical_games.columns:
        raise ValueError("historical_games must have 'vegas_spread' column")
    if "push" not in historical_games.columns:
        raise ValueError("historical_games must have 'push' column")

    # Filter to games with valid vegas_spread
    df = historical_games[historical_games["vegas_spread"].notna()].copy()

    if len(df) == 0:
        raise ValueError("No games with valid vegas_spread for push rate estimation")

    # Compute tick: int(round(abs(spread) * 2))
    df["tick"] = (df["vegas_spread"].abs() * 2).round().astype(int)

    # Overall push rate
    default_overall = df["push"].mean()

    # Even-tick games only (integer spreads that can push)
    even_mask = df["tick"] % 2 == 0
    even_df = df[even_mask]

    if len(even_df) > 0:
        default_even = even_df["push"].mean()
    else:
        default_even = default_overall

    # Compute per-tick rates
    tick_rates = {}
    n_games_by_tick = {}

    for tick in df["tick"].unique():
        tick_df = df[df["tick"] == tick]
        n = len(tick_df)
        n_games_by_tick[tick] = n

        # Odd ticks (half-point spreads) => p_push = 0.0
        if tick % 2 == 1:
            tick_rates[tick] = 0.0
            continue

        # Even ticks
        push_rate = tick_df["push"].mean()

        # KEY_TICKS always get own bucket
        if tick in KEY_TICKS:
            tick_rates[tick] = push_rate
        elif n >= min_games_per_tick:
            tick_rates[tick] = push_rate
        # else: will use default_even

    # Get years for auditability
    if "year" in df.columns:
        years_trained = sorted(df["year"].unique().tolist())
    else:
        years_trained = []

    return PushRates(
        tick_rates=tick_rates,
        default_even=default_even,
        default_overall=default_overall,
        n_games_by_tick=n_games_by_tick,
        years_trained=years_trained,
    )


def get_push_probability(
    vegas_spread: float,
    push_rates: PushRates,
) -> float:
    """Get push probability for a given Vegas spread.

    Args:
        vegas_spread: Vegas spread (any sign, uses abs())
        push_rates: PushRates from estimate_push_rates

    Returns:
        p_push: Probability of push for this spread

    Logic:
        tick = int(round(abs(vegas_spread) * 2))
        if tick is odd: return 0.0 (half-point spreads can't push)
        if tick in tick_rates: return tick_rates[tick]
        else: return default_even (or default_overall if needed)
    """
    tick = int(round(abs(vegas_spread) * 2))

    # Odd ticks (half-point spreads) can't push
    if tick % 2 == 1:
        return 0.0

    # Look up in tick_rates
    if tick in push_rates.tick_rates:
        return push_rates.tick_rates[tick]

    # Fallback to default_even for integer spreads
    return push_rates.default_even


def get_push_probability_vectorized(
    vegas_spreads: np.ndarray,
    push_rates: PushRates,
) -> np.ndarray:
    """Vectorized push probability lookup.

    Args:
        vegas_spreads: Array of Vegas spreads
        push_rates: PushRates from estimate_push_rates

    Returns:
        Array of p_push values
    """
    ticks = np.round(np.abs(vegas_spreads) * 2).astype(int)

    result = np.zeros(len(ticks))

    for i, tick in enumerate(ticks):
        # Odd ticks can't push
        if tick % 2 == 1:
            result[i] = 0.0
        elif tick in push_rates.tick_rates:
            result[i] = push_rates.tick_rates[tick]
        else:
            result[i] = push_rates.default_even

    return result


def get_calibration_label(training_window_seasons: int | None) -> str:
    """Get human-readable label for calibration mode.

    Args:
        training_window_seasons: Training window size (None = all prior)

    Returns:
        Label like "ROLLING_2" or "INCLUDE_ALL"
    """
    if training_window_seasons is None:
        return "INCLUDE_ALL"
    else:
        return f"ROLLING_{training_window_seasons}"


@dataclass
class CalibrationResult:
    """Result of P(cover) calibration via logistic regression."""

    intercept: float  # logit(P(cover|edge=0))
    slope: float  # coefficient on edge_abs (should be positive)
    n_games: int  # Number of games used in training
    years_trained: list[int]  # Years included in training
    implied_breakeven_edge: float  # edge where P(cover) = 0.524 (breakeven at -110)
    implied_5pt_pcover: float  # P(cover) at edge_abs=5
    p_cover_at_zero: float  # expit(intercept) - extrapolated baseline

    def __repr__(self) -> str:
        return (
            f"CalibrationResult(slope={self.slope:.4f}, intercept={self.intercept:.4f}, "
            f"n={self.n_games}, p_cover@0={self.p_cover_at_zero:.3f}, "
            f"p_cover@5={self.implied_5pt_pcover:.3f}, breakeven_edge={self.implied_breakeven_edge:.2f})"
        )


@dataclass
class WalkForwardResult:
    """Result of walk-forward validation.

    V2 additions:
    - game_results now includes p_push, p_cover (unconditional), ev columns
    - fold_summaries now includes push_rates info
    - push_rate_summaries: Per-fold push rate diagnostics
    """

    game_results: pd.DataFrame  # Per-game results with p_cover_no_push, p_push, p_cover, ev
    fold_summaries: list[dict]  # Per-fold calibration summaries
    overall_brier: float  # Brier score vs actual outcomes
    overall_log_loss: float  # Log loss vs actual outcomes
    brier_vs_constant: float  # Brier score of constant 0.5 baseline
    brier_vs_best_constant: float  # Brier score of best constant (empirical mean)
    brier_skill_score: float  # 1 - Brier_model / Brier_best_constant
    # V2 additions
    push_rate_summaries: list[dict] = field(default_factory=list)  # Per-fold push rate diagnostics
    include_push_modeling: bool = False  # Whether push modeling was used


def load_and_normalize_game_data(
    backtest_df: pd.DataFrame,
    jp_convention: str = "pos_home_favored",
) -> pd.DataFrame:
    """Load backtest results and normalize to Vegas convention.

    Required input columns:
        game_id, year, week, home_team, away_team,
        predicted_spread (JP+ internal), actual_margin,
        spread_open, spread_close

    Derived columns added:
        jp_spread (Vegas convention), vegas_spread, cover_margin,
        home_covered, away_covered, push, edge_pts, edge_abs,
        jp_favored_side, jp_side_covered

    Args:
        backtest_df: DataFrame with backtest results
        jp_convention: Convention for predicted_spread input
            "pos_home_favored": Internal JP+ (positive = home favored) -> needs flip
            "neg_home_favored": Already Vegas convention -> no flip

    Returns:
        DataFrame with normalized spreads and derived columns
    """
    required_cols = [
        "game_id", "year", "week", "home_team", "away_team",
        "predicted_spread", "actual_margin",
    ]
    missing = set(required_cols) - set(backtest_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = backtest_df.copy()

    # Normalize JP+ spread to Vegas convention (negative = home favored)
    if jp_convention == "pos_home_favored":
        df["jp_spread"] = -df["predicted_spread"]
    else:
        df["jp_spread"] = df["predicted_spread"]

    # Vegas spread - use close by default, fallback to open
    if "spread_close" in df.columns:
        df["vegas_spread"] = df["spread_close"]
    elif "spread_open" in df.columns:
        df["vegas_spread"] = df["spread_open"]
    else:
        raise ValueError("Need either spread_close or spread_open in input")

    # Filter to games with vegas spread
    valid_mask = df["vegas_spread"].notna()
    df = df[valid_mask].copy()
    logger.info(f"Filtered to {len(df)} games with valid Vegas spreads")

    # Cover margin: actual_margin + vegas_spread
    # Positive = home covered, Negative = away covered, Zero = push
    df["cover_margin"] = df["actual_margin"] + df["vegas_spread"]

    # Binary outcomes
    df["home_covered"] = df["cover_margin"] > 0
    df["away_covered"] = df["cover_margin"] < 0
    df["push"] = df["cover_margin"] == 0

    # Edge calculation (Vegas convention)
    # edge_pts = jp_spread - vegas_spread
    # Negative = JP+ likes HOME more than Vegas
    # Positive = JP+ likes AWAY more than Vegas
    df["edge_pts"] = df["jp_spread"] - df["vegas_spread"]
    df["edge_abs"] = np.abs(df["edge_pts"])

    # Which side does JP+ favor?
    df["jp_favored_side"] = np.where(df["edge_pts"] < 0, "HOME", "AWAY")

    # Did the JP+ favored side cover?
    df["jp_side_covered"] = np.where(
        df["jp_favored_side"] == "HOME",
        df["home_covered"],
        df["away_covered"]
    )

    # Zero-edge games: JP+ has no opinion — must not be included in calibration or evaluation.
    zero_edge_mask = df["edge_pts"] == 0
    if zero_edge_mask.any():
        df["jp_side_covered"] = df["jp_side_covered"].astype(object)
        df.loc[zero_edge_mask, "jp_favored_side"] = "NO_BET"
        df.loc[zero_edge_mask, "jp_side_covered"] = np.nan

    return df


def calibrate_cover_probability(
    historical_df: pd.DataFrame,
    min_games_warn: int = 2000,
) -> CalibrationResult:
    """Fit logistic regression: P(jp_side_covered) ~ edge_abs.

    Training filters:
    - Exclude pushes (cover_margin == 0)
    - Exclude edge_abs == 0 (NO_BET games)

    Args:
        historical_df: DataFrame with jp_side_covered, edge_abs columns
        min_games_warn: Warn if training on fewer games

    Returns:
        CalibrationResult with fitted parameters
    """
    # Filter out pushes and zero-edge games
    mask = (~historical_df["push"]) & (historical_df["edge_abs"] > 0)
    df = historical_df[mask].copy()

    if len(df) < min_games_warn:
        logger.warning(
            f"Calibrating on only {len(df)} games (threshold: {min_games_warn})"
        )

    if len(df) < 100:
        raise ValueError(f"Insufficient data for calibration: {len(df)} games")

    # Prepare features and target
    X = df[["edge_abs"]].values
    y = df["jp_side_covered"].astype(int).values

    # Fit logistic regression
    # No standardization - we want raw edge_abs coefficients
    # Use C=1.0 for L2 regularization (equivalent to penalty="l2" which is deprecated)
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        fit_intercept=True,
    )
    model.fit(X, y)

    intercept = model.intercept_[0]
    slope = model.coef_[0][0]

    # Derived quantities
    p_cover_at_zero = expit(intercept)

    # P(cover) at edge_abs = 5
    implied_5pt_pcover = expit(intercept + slope * 5)

    # Breakeven edge: find edge where P(cover) = 0.524 (breakeven at -110)
    # 0.524 = expit(intercept + slope * edge)
    # logit(0.524) = intercept + slope * edge
    # edge = (logit(0.524) - intercept) / slope
    breakeven_prob_val = breakeven_prob(-110)
    logit_breakeven = np.log(breakeven_prob_val / (1 - breakeven_prob_val))

    if slope > 0:
        implied_breakeven_edge = (logit_breakeven - intercept) / slope
    else:
        # Slope should be positive; if not, something is wrong
        logger.warning(f"Negative slope ({slope:.4f}) - calibration may be invalid")
        implied_breakeven_edge = float("inf")

    years_trained = sorted(df["year"].unique().tolist())

    return CalibrationResult(
        intercept=intercept,
        slope=slope,
        n_games=len(df),
        years_trained=years_trained,
        implied_breakeven_edge=implied_breakeven_edge,
        implied_5pt_pcover=implied_5pt_pcover,
        p_cover_at_zero=p_cover_at_zero,
    )


def predict_cover_probability(
    edge_abs: np.ndarray,
    calibration: CalibrationResult,
    clamp_min: float = 0.01,
    clamp_max: float = 0.99,
) -> np.ndarray:
    """Predict P(cover | no push) from edge_abs.

    Args:
        edge_abs: Array of absolute edge values
        calibration: CalibrationResult from calibrate_cover_probability
        clamp_min: Minimum probability (avoid log(0))
        clamp_max: Maximum probability

    Returns:
        Array of P(cover | no push), clamped to [clamp_min, clamp_max]

    Note:
        This is CONDITIONAL P(cover | no push). In V1, p_push=0 so equals unconditional.
    """
    logits = calibration.intercept + calibration.slope * edge_abs
    p_cover = expit(logits)
    return np.clip(p_cover, clamp_min, clamp_max)


def walk_forward_validate(
    all_games: pd.DataFrame,
    min_train_seasons: int = 2,
    exclude_covid: bool = True,
    exclude_years_from_training: list[int] | None = None,
    training_window_seasons: int | None = None,
    include_push_modeling: bool = True,
) -> WalkForwardResult:
    """Season-level walk-forward calibration.

    For each evaluation season Y:
        1. Training: seasons < Y (optionally excluding specified years, optionally windowed)
        2. Filter training: exclude pushes and edge_abs == 0
        3. Fit calibration on training
        4. [V2] Estimate push rates from same training set
        5. Apply to season Y with push-aware EV
        6. Track p_cover_no_push, p_push, p_cover, EV

    Args:
        all_games: DataFrame with all games (normalized)
        min_train_seasons: Minimum seasons required for training
        exclude_covid: If True, exclude 2020 from evaluation folds
        exclude_years_from_training: Years to exclude from training data
            (e.g., [2022] to exclude 2022's negative-slope data)
        training_window_seasons: If set, only use the most recent N seasons
            prior to evaluation year (e.g., 2 means use Y-2, Y-1).
            If None, use all prior seasons. Default production mode is 2 (ROLLING_2).
        include_push_modeling: If True (V2 default), estimate push rates and
            compute push-aware p_cover and EV. If False, use p_push=0.

    Returns:
        WalkForwardResult with per-game predictions and diagnostics
    """
    years = sorted(all_games["year"].unique())
    exclude_from_train = set(exclude_years_from_training or [])

    if exclude_covid and 2020 in years:
        eval_years = [y for y in years if y != 2020]
    else:
        eval_years = years

    # Determine first year we can evaluate (need min_train_seasons prior)
    first_eval_year = years[0] + min_train_seasons

    results = []
    fold_summaries = []
    push_rate_summaries = []

    for eval_year in eval_years:
        if eval_year < first_eval_year:
            continue

        # Training: years strictly before eval_year
        # Build list of eligible training years (excluding specified exclusions)
        eligible_train_years = [y for y in years if y < eval_year and y not in exclude_from_train]

        if training_window_seasons is not None:
            # Rolling window: only use most recent N seasons
            if len(eligible_train_years) < training_window_seasons:
                # Guardrail: insufficient data for requested window
                logger.warning(
                    f"Fold {eval_year}: Only {len(eligible_train_years)} eligible training years "
                    f"for window={training_window_seasons}. Using all {len(eligible_train_years)} years."
                )
                window_years = eligible_train_years  # Use all available
            else:
                window_years = eligible_train_years[-training_window_seasons:]  # Most recent N
            train_mask = all_games["year"].isin(window_years)
            actual_train_years = window_years
        else:
            # All prior years (excluding specified)
            train_mask = (all_games["year"] < eval_year) & (~all_games["year"].isin(exclude_from_train))
            actual_train_years = eligible_train_years

        train_df = all_games[train_mask]

        # Evaluation: eval_year only
        eval_mask = all_games["year"] == eval_year
        eval_df = all_games[eval_mask].copy()

        if len(train_df) < 500:
            logger.warning(
                f"Skipping {eval_year}: insufficient training data ({len(train_df)} games)"
            )
            continue

        # Fit calibration on training data
        try:
            calibration = calibrate_cover_probability(train_df, min_games_warn=1000)
        except ValueError as e:
            logger.warning(f"Calibration failed for {eval_year}: {e}")
            continue

        # [V2] Estimate push rates from same training set (no leakage)
        push_rates = None
        push_rate_summary = None
        if include_push_modeling:
            try:
                push_rates = estimate_push_rates(train_df, min_games_per_tick=50)
                push_rate_summary = {
                    "eval_year": eval_year,
                    "default_even": push_rates.default_even,
                    "default_overall": push_rates.default_overall,
                    "n_ticks": len(push_rates.tick_rates),
                    "key_tick_rates": {t: push_rates.tick_rates.get(t, push_rates.default_even) for t in KEY_TICKS},
                    "years_trained": push_rates.years_trained,
                }
                push_rate_summaries.append(push_rate_summary)
            except ValueError as e:
                logger.warning(f"Push rate estimation failed for {eval_year}: {e}")
                # Fall back to no push modeling for this fold

        # Predict on eval set
        # Only games with edge_abs > 0 and not push get predictions
        bet_mask = (eval_df["edge_abs"] > 0) & (~eval_df["push"])
        eval_df["p_cover_no_push"] = np.nan
        eval_df["p_push"] = np.nan
        eval_df["p_cover"] = np.nan
        eval_df["ev"] = np.nan

        if bet_mask.sum() > 0:
            edge_vals = eval_df.loc[bet_mask, "edge_abs"].values
            p_cover_no_push = predict_cover_probability(edge_vals, calibration)
            eval_df.loc[bet_mask, "p_cover_no_push"] = p_cover_no_push

            # [V2] Add push probability and push-aware metrics
            if push_rates is not None:
                vegas_vals = eval_df.loc[bet_mask, "vegas_spread"].values
                p_push_vals = get_push_probability_vectorized(vegas_vals, push_rates)
                eval_df.loc[bet_mask, "p_push"] = p_push_vals

                # Unconditional P(cover) = P(cover | no push) * (1 - p_push)
                p_cover_vals = p_cover_no_push * (1 - p_push_vals)
                eval_df.loc[bet_mask, "p_cover"] = p_cover_vals

                # EV computed via canonical function to ensure consistency
                ev_vals = calculate_ev_vectorized(p_cover_no_push, p_push=p_push_vals)
                eval_df.loc[bet_mask, "ev"] = ev_vals
            else:
                # No push modeling - p_push = 0
                eval_df.loc[bet_mask, "p_push"] = 0.0
                eval_df.loc[bet_mask, "p_cover"] = p_cover_no_push
                # EV computed via canonical function to ensure consistency
                ev_vals = calculate_ev_vectorized(p_cover_no_push, p_push=0.0)
                eval_df.loc[bet_mask, "ev"] = ev_vals

        # Add fold metadata
        eval_df["fold_intercept"] = calibration.intercept
        eval_df["fold_slope"] = calibration.slope
        eval_df["fold_breakeven_edge"] = calibration.implied_breakeven_edge
        eval_df["fold_p_cover_at_zero"] = calibration.p_cover_at_zero

        results.append(eval_df)

        # Fold summary - include actual training years for auditability
        fold_summaries.append({
            "eval_year": eval_year,
            "n_train": calibration.n_games,
            "n_eval": len(eval_df),
            "n_bets": bet_mask.sum(),
            "intercept": calibration.intercept,
            "slope": calibration.slope,
            "p_cover_at_zero": calibration.p_cover_at_zero,
            "implied_5pt_pcover": calibration.implied_5pt_pcover,
            "breakeven_edge": calibration.implied_breakeven_edge,
            "years_trained": calibration.years_trained,
            "training_years_used": actual_train_years,  # Actual years used for this fold
            "training_window_seasons": training_window_seasons,  # Config for auditability
            "push_modeling": include_push_modeling and push_rates is not None,  # V2
            "push_rate_summary": push_rate_summary,  # V2
        })

    if not results:
        raise ValueError("No valid folds for walk-forward validation")

    game_results = pd.concat(results, ignore_index=True)

    # Calculate overall metrics (on games with predictions and outcomes)
    eval_mask = game_results["p_cover_no_push"].notna() & (~game_results["push"])
    eval_df = game_results[eval_mask]

    if len(eval_df) == 0:
        raise ValueError("No games with valid predictions for scoring")

    y_true = eval_df["jp_side_covered"].astype(int).values
    y_pred = eval_df["p_cover_no_push"].values

    # Brier score
    overall_brier = np.mean((y_pred - y_true) ** 2)

    # Log loss
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    overall_log_loss = -np.mean(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )

    # Baseline comparisons
    # Constant 0.5
    brier_vs_constant = np.mean((0.5 - y_true) ** 2)

    # Best constant (empirical mean)
    best_constant = y_true.mean()
    brier_vs_best_constant = np.mean((best_constant - y_true) ** 2)

    # Brier skill score
    brier_skill_score = 1 - overall_brier / brier_vs_best_constant

    return WalkForwardResult(
        game_results=game_results,
        fold_summaries=fold_summaries,
        overall_brier=overall_brier,
        overall_log_loss=overall_log_loss,
        brier_vs_constant=brier_vs_constant,
        brier_vs_best_constant=brier_vs_best_constant,
        brier_skill_score=brier_skill_score,
        push_rate_summaries=push_rate_summaries,
        include_push_modeling=include_push_modeling,
    )


def breakeven_prob(juice: float = -110) -> float:
    """Breakeven probability for given juice.

    At -110 (negative): risk 110 to win 100 → 110/210 = 52.38%
    At +150 (positive): risk 100 to win 150 → 100/250 = 40.00%

    Args:
        juice: American odds (e.g., -110 or +150). Zero is invalid.

    Returns:
        Breakeven probability as decimal (e.g., 0.5238)
    """
    if juice == 0:
        raise ValueError("American odds of 0 are undefined")
    if juice < 0:
        risk = abs(juice)
        win = 100
        return risk / (risk + win)
    else:
        # Positive odds: risk 100 to win juice
        return 100 / (100 + juice)


def calculate_ev(
    p_cover_no_push: float,
    p_push: float = 0.0,
    juice: float = -110,
) -> float:
    """Expected value as fraction of stake.

    EV = P(win) * payout - P(lose) * stake

    At -110:
        Payout = 100/110 = 0.909
        Stake = 1.0
        EV = p * (100/110) - (1-p) * 1.0

    Args:
        p_cover_no_push: P(cover | no push), from calibration
        p_push: P(push) - V1 always 0
        juice: American odds (e.g., -110 or +150). Zero is invalid.

    Returns:
        EV as fraction of stake (e.g., 0.03 = +3% EV)
    """
    if juice == 0:
        raise ValueError("American odds of 0 are undefined")

    # Adjust for push probability
    # P(win) = P(cover | no push) * (1 - p_push)
    # P(lose) = P(not cover | no push) * (1 - p_push)
    # P(push) = p_push (returns stake, no win/loss)
    p_win = p_cover_no_push * (1 - p_push)
    p_lose = (1 - p_cover_no_push) * (1 - p_push)

    # Payout per unit risked
    if juice < 0:
        payout = 100 / abs(juice)
    else:
        payout = juice / 100

    ev = p_win * payout - p_lose * 1.0
    return ev


def calculate_ev_vectorized(
    p_cover_no_push: np.ndarray,
    p_push: float | np.ndarray = 0.0,
    juice: float = -110,
) -> np.ndarray:
    """Vectorized EV calculation.

    Args:
        p_cover_no_push: Array of P(cover | no push)
        p_push: P(push) as scalar or per-game array. NumPy broadcasting
            handles both cases correctly.
        juice: American odds (e.g., -110 or +150). Zero is invalid.

    Returns:
        Array of EV values
    """
    if juice == 0:
        raise ValueError("American odds of 0 are undefined")

    p_win = p_cover_no_push * (1 - p_push)
    p_lose = (1 - p_cover_no_push) * (1 - p_push)

    if juice < 0:
        payout = 100 / abs(juice)
    else:
        payout = juice / 100

    return p_win * payout - p_lose


def get_spread_bucket(edge_abs: float) -> str:
    """Categorize edge_abs into diagnostic buckets using EDGE_BUCKETS.

    Args:
        edge_abs: Absolute edge value

    Returns:
        Bucket label from EDGE_BUCKETS (e.g., "[0,3)", "[3,5)", etc.)
    """
    for label, low, high in EDGE_BUCKETS:
        if low <= edge_abs < high:
            return label
    return EDGE_BUCKETS[-1][0]  # Fallback to last bucket


def diagnose_calibration(
    game_results: pd.DataFrame,
    calibration: Optional[CalibrationResult] = None,
) -> dict:
    """Run calibration diagnostics.

    Checks:
    - Spread-bucket calibration with Wilson CI
    - Monotonicity with 2pp tolerance
    - Model vs baseline Brier/log loss

    Args:
        game_results: DataFrame with p_cover_no_push, jp_side_covered, edge_abs
        calibration: Optional CalibrationResult for prediction

    Returns:
        Dictionary with diagnostic results
    """
    from scipy.stats import norm

    # Filter to games with predictions
    df = game_results[game_results["p_cover_no_push"].notna() & ~game_results["push"]].copy()

    if len(df) == 0:
        return {"error": "No games with predictions"}

    # Add buckets
    df["bucket"] = df["edge_abs"].apply(get_spread_bucket)

    # Per-bucket statistics
    bucket_stats = []
    bucket_order = [label for label, _, _ in EDGE_BUCKETS]

    for bucket in bucket_order:
        bucket_df = df[df["bucket"] == bucket]
        n = len(bucket_df)

        if n == 0:
            continue

        # Empirical win rate
        wins = bucket_df["jp_side_covered"].sum()
        empirical_rate = wins / n

        # Wilson score 95% CI
        z = norm.ppf(0.975)
        denom = 1 + z**2 / n
        center = (empirical_rate + z**2 / (2 * n)) / denom
        spread = z * np.sqrt(empirical_rate * (1 - empirical_rate) / n + z**2 / (4 * n**2)) / denom
        wilson_low = center - spread
        wilson_high = center + spread

        # Midpoint edge and predicted P(cover) at midpoint
        midpoint = bucket_df["edge_abs"].median()
        predicted_at_mid = bucket_df["p_cover_no_push"].mean()

        # Check if prediction is within CI
        in_ci = wilson_low <= predicted_at_mid <= wilson_high

        bucket_stats.append({
            "bucket": bucket,
            "n": n,
            "wins": int(wins),
            "empirical_rate": empirical_rate,
            "predicted_rate": predicted_at_mid,
            "wilson_low": wilson_low,
            "wilson_high": wilson_high,
            "midpoint_edge": midpoint,
            "in_ci": in_ci,
        })

    # Check monotonicity (empirical rate should increase with edge, 2pp tolerance)
    monotonic_violations = []
    for i in range(1, len(bucket_stats)):
        prev_rate = bucket_stats[i - 1]["empirical_rate"]
        curr_rate = bucket_stats[i]["empirical_rate"]
        if curr_rate < prev_rate - 0.02:  # 2pp tolerance
            monotonic_violations.append({
                "from_bucket": bucket_stats[i - 1]["bucket"],
                "to_bucket": bucket_stats[i]["bucket"],
                "delta": curr_rate - prev_rate,
            })

    # Overall metrics
    y_true = df["jp_side_covered"].astype(int).values
    y_pred = df["p_cover_no_push"].values

    brier_model = np.mean((y_pred - y_true) ** 2)
    brier_constant_50 = np.mean((0.5 - y_true) ** 2)
    brier_constant_524 = np.mean((breakeven_prob(-110) - y_true) ** 2)
    brier_best_constant = np.mean((y_true.mean() - y_true) ** 2)

    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    log_loss_model = -np.mean(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )

    # Constant baselines for log loss
    best_const = y_true.mean()
    best_const_clipped = np.clip(best_const, eps, 1 - eps)
    log_loss_best_constant = -np.mean(
        y_true * np.log(best_const_clipped) + (1 - y_true) * np.log(1 - best_const_clipped)
    )

    return {
        "n_games": len(df),
        "empirical_cover_rate": y_true.mean(),
        "bucket_stats": bucket_stats,
        "monotonic_violations": monotonic_violations,
        "brier_model": brier_model,
        "brier_constant_50": brier_constant_50,
        "brier_constant_524": brier_constant_524,
        "brier_best_constant": brier_best_constant,
        "brier_skill_score": 1 - brier_model / brier_best_constant,
        "log_loss_model": log_loss_model,
        "log_loss_best_constant": log_loss_best_constant,
        "beats_best_constant_brier": brier_model < brier_best_constant,
        "beats_best_constant_log_loss": log_loss_model < log_loss_best_constant,
    }


def diagnose_fold_stability(fold_summaries: list[dict]) -> dict:
    """Check calibration stability across folds.

    Flags:
    - Intercept: WARN if p_cover_at_zero outside [0.47, 0.53], FLAG if outside [0.45, 0.55]
    - Slope: stable within 30% of median across folds
    - Breakeven edge: log per fold, flag large swings

    Args:
        fold_summaries: List of fold summary dicts from walk_forward_validate

    Returns:
        Dictionary with stability diagnostics
    """
    if not fold_summaries:
        return {"error": "No fold summaries"}

    intercepts = [f["intercept"] for f in fold_summaries]
    slopes = [f["slope"] for f in fold_summaries]
    p_cover_zeros = [f["p_cover_at_zero"] for f in fold_summaries]
    breakeven_edges = [f["breakeven_edge"] for f in fold_summaries]

    # Intercept stability
    intercept_warnings = []
    intercept_flags = []
    for i, p0 in enumerate(p_cover_zeros):
        year = fold_summaries[i]["eval_year"]
        if p0 < 0.45 or p0 > 0.55:
            intercept_flags.append({"year": year, "p_cover_at_zero": p0})
        elif p0 < 0.47 or p0 > 0.53:
            intercept_warnings.append({"year": year, "p_cover_at_zero": p0})

    # Slope stability (within 30% of median)
    median_slope = np.median(slopes)
    slope_violations = []
    for i, s in enumerate(slopes):
        year = fold_summaries[i]["eval_year"]
        if abs(s - median_slope) > 0.3 * abs(median_slope):
            slope_violations.append({
                "year": year,
                "slope": s,
                "median": median_slope,
                "deviation_pct": abs(s - median_slope) / abs(median_slope) * 100,
            })

    # Breakeven edge stability
    be_mean = np.mean(breakeven_edges)
    be_std = np.std(breakeven_edges)

    return {
        "n_folds": len(fold_summaries),
        "intercept_mean": np.mean(intercepts),
        "intercept_std": np.std(intercepts),
        "slope_mean": np.mean(slopes),
        "slope_median": median_slope,
        "slope_std": np.std(slopes),
        "breakeven_edge_mean": be_mean,
        "breakeven_edge_std": be_std,
        "p_cover_at_zero_mean": np.mean(p_cover_zeros),
        "p_cover_at_zero_std": np.std(p_cover_zeros),
        "intercept_warnings": intercept_warnings,
        "intercept_flags": intercept_flags,
        "slope_violations": slope_violations,
        "folds": fold_summaries,
    }


def stratified_diagnostics(
    game_results: pd.DataFrame,
    vegas_total: Optional[pd.Series] = None,
) -> dict:
    """Stratified calibration diagnostics.

    Returns diagnostics stratified by:
        1. vegas_total terciles (low/med/high) - if vegas_total provided
        2. abs(vegas_spread) buckets from EDGE_BUCKETS: [0,3), [3,5), [5,7), [7,10), [10,+)
        3. week buckets: weeks 0-3, 4-8, 9-15, 16+

    Per stratum:
        - N games
        - Empirical cover rate
        - Average predicted p_cover
        - Brier score
        - ATS% if bet selected

    Args:
        game_results: DataFrame with p_cover_no_push, jp_side_covered, edge_abs, week, vegas_spread
        vegas_total: Optional Series of total lines (for tercile analysis)

    Returns:
        Dictionary with stratified diagnostics
    """
    # Filter to games with predictions (edge > 0, not push)
    df = game_results[
        game_results["p_cover_no_push"].notna() & ~game_results["push"]
    ].copy()

    if len(df) == 0:
        return {"error": "No games with predictions"}

    result = {
        "n_games_total": len(df),
        "spread_strata": [],
        "week_strata": [],
        "total_terciles": [],
    }

    # ----- Spread buckets (shared with diagnose_calibration) -----
    for label, low, high in EDGE_BUCKETS:
        mask = (df["edge_abs"] >= low) & (df["edge_abs"] < high)
        stratum = df[mask]
        n = len(stratum)

        if n == 0:
            continue

        y_true = stratum["jp_side_covered"].astype(int).values
        y_pred = stratum["p_cover_no_push"].values

        result["spread_strata"].append({
            "bucket": label,
            "n": n,
            "empirical_rate": y_true.mean(),
            "avg_predicted": y_pred.mean(),
            "brier": np.mean((y_pred - y_true) ** 2),
            "wins": int(y_true.sum()),
            "losses": n - int(y_true.sum()),
        })

    # ----- Week buckets -----
    week_buckets = [
        ("weeks 0-3", 0, 3),  # Phase 1 (includes CFB week 0)
        ("weeks 4-8", 4, 8),
        ("weeks 9-15", 9, 15),
        ("weeks 16+", 16, 100),  # Postseason
    ]

    for label, low, high in week_buckets:
        mask = (df["week"] >= low) & (df["week"] <= high)
        stratum = df[mask]
        n = len(stratum)

        if n == 0:
            continue

        y_true = stratum["jp_side_covered"].astype(int).values
        y_pred = stratum["p_cover_no_push"].values

        result["week_strata"].append({
            "bucket": label,
            "n": n,
            "empirical_rate": y_true.mean(),
            "avg_predicted": y_pred.mean(),
            "brier": np.mean((y_pred - y_true) ** 2),
            "wins": int(y_true.sum()),
            "losses": n - int(y_true.sum()),
        })

    # ----- Total terciles (if provided) -----
    if vegas_total is not None and len(vegas_total) == len(game_results):
        # Align vegas_total with filtered df
        df_with_total = df.copy()
        df_with_total["vegas_total"] = vegas_total.loc[df.index].values

        # Filter to games with valid totals
        df_with_total = df_with_total[df_with_total["vegas_total"].notna()]

        if len(df_with_total) >= 30:
            # Compute tercile thresholds
            t1 = df_with_total["vegas_total"].quantile(0.33)
            t2 = df_with_total["vegas_total"].quantile(0.67)

            tercile_defs = [
                ("low (< {:.1f})".format(t1), 0, t1),
                ("med ({:.1f}-{:.1f})".format(t1, t2), t1, t2),
                ("high (>= {:.1f})".format(t2), t2, float("inf")),
            ]

            for label, low, high in tercile_defs:
                mask = (df_with_total["vegas_total"] >= low) & (df_with_total["vegas_total"] < high)
                stratum = df_with_total[mask]
                n = len(stratum)

                if n == 0:
                    continue

                y_true = stratum["jp_side_covered"].astype(int).values
                y_pred = stratum["p_cover_no_push"].values

                result["total_terciles"].append({
                    "bucket": label,
                    "n": n,
                    "empirical_rate": y_true.mean(),
                    "avg_predicted": y_pred.mean(),
                    "brier": np.mean((y_pred - y_true) ** 2),
                    "wins": int(y_true.sum()),
                    "losses": n - int(y_true.sum()),
                })

    return result


def print_stratified_diagnostics(diag: dict) -> None:
    """Print stratified diagnostics in a readable format.

    Args:
        diag: Output from stratified_diagnostics()
    """
    if "error" in diag:
        print(f"Error: {diag['error']}")
        return

    print("\n" + "=" * 80)
    print("STRATIFIED CALIBRATION DIAGNOSTICS")
    print("=" * 80)
    print(f"Total games: {diag['n_games_total']}")

    # Spread strata
    print("\nSPREAD BUCKETS:")
    print("-" * 70)
    print(f"{'Bucket':<12} | {'N':>6} | {'Emp%':>7} | {'Pred%':>7} | {'Brier':>7} | {'W-L':>10}")
    print("-" * 70)
    for s in diag["spread_strata"]:
        wl = f"{s['wins']}-{s['losses']}"
        print(f"{s['bucket']:<12} | {s['n']:>6} | {s['empirical_rate']*100:>6.1f}% | "
              f"{s['avg_predicted']*100:>6.1f}% | {s['brier']:>7.4f} | {wl:>10}")
    print("-" * 70)

    # Week strata
    print("\nWEEK BUCKETS:")
    print("-" * 70)
    print(f"{'Bucket':<12} | {'N':>6} | {'Emp%':>7} | {'Pred%':>7} | {'Brier':>7} | {'W-L':>10}")
    print("-" * 70)
    for s in diag["week_strata"]:
        wl = f"{s['wins']}-{s['losses']}"
        print(f"{s['bucket']:<12} | {s['n']:>6} | {s['empirical_rate']*100:>6.1f}% | "
              f"{s['avg_predicted']*100:>6.1f}% | {s['brier']:>7.4f} | {wl:>10}")
    print("-" * 70)

    # Total terciles (if available)
    if diag["total_terciles"]:
        print("\nTOTAL TERCILES:")
        print("-" * 70)
        print(f"{'Bucket':<25} | {'N':>6} | {'Emp%':>7} | {'Pred%':>7} | {'Brier':>7}")
        print("-" * 70)
        for s in diag["total_terciles"]:
            print(f"{s['bucket']:<25} | {s['n']:>6} | {s['empirical_rate']*100:>6.1f}% | "
                  f"{s['avg_predicted']*100:>6.1f}% | {s['brier']:>7.4f}")
        print("-" * 70)

    print("=" * 80)


# =============================================================================
# CALIBRATION ARTIFACT LOADING
# =============================================================================

def load_spread_calibration_from_artifact(
    artifact_path,
) -> CalibrationResult:
    """Load a spread calibration from a JSON artifact file.

    Args:
        artifact_path: Path to JSON artifact file

    Returns:
        CalibrationResult with loaded parameters

    Example:
        >>> cal = load_spread_calibration_from_artifact(
        ...     "data/spread_selection/artifacts/spread_ev_calibration_phase2_only_2022_2025.json"
        ... )
        >>> print(cal.implied_breakeven_edge)  # Should be ~5.3 pts
    """
    from pathlib import Path
    import json

    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration artifact not found: {artifact_path}")

    with open(path, 'r') as f:
        data = json.load(f)

    # Extract parameters
    params = data.get('parameters', {})
    metadata = data.get('metadata', {})

    return CalibrationResult(
        intercept=params.get('intercept', 0.0),
        slope=params.get('slope', 0.0),
        n_games=metadata.get('n_games', 0),
        years_trained=metadata.get('years_trained', []),
        implied_breakeven_edge=metadata.get('breakeven_edge_at_110', float('inf')),
        implied_5pt_pcover=metadata.get('p_cover_at_5pt', 0.5),
        p_cover_at_zero=metadata.get('p_cover_at_zero', 0.5),
    )


def get_default_spread_calibration(
    phase: str = "phase2",
    artifact_dir: str = "data/spread_selection/artifacts",
) -> CalibrationResult:
    """Get default spread calibration for production use.

    Args:
        phase: Which calibration to use:
            - "phase2": Weeks 4-15 calibration (recommended for Core Phase)
            - "weighted": Full season with Phase 1 downweighted
            - "phase1": Weeks 1-3 only (not recommended)
            - "full": Full season baseline (not recommended)
        artifact_dir: Directory containing calibration artifacts

    Returns:
        CalibrationResult for the requested phase

    Example:
        >>> cal = get_default_spread_calibration("phase2")
        >>> print(cal.implied_breakeven_edge)  # ~5.3 pts
    """
    from pathlib import Path

    phase_to_file = {
        "phase2": "spread_ev_calibration_phase2_only_2022_2025.json",
        "weighted": "spread_ev_calibration_weighted_2022_2025.json",
        "phase1": "spread_ev_calibration_phase1_only_2022_2025.json",
        "full": "spread_ev_calibration_full_season_2022_2025.json",
    }

    if phase not in phase_to_file:
        raise ValueError(f"Unknown phase '{phase}'. Choose from: {list(phase_to_file.keys())}")

    artifact_path = Path(artifact_dir) / phase_to_file[phase]

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Calibration artifact not found: {artifact_path}. "
            "Run scripts/rebuild_spread_calibration.py to generate artifacts."
        )

    return load_spread_calibration_from_artifact(artifact_path)


# =============================================================================
# PHASE-AWARE ROUTING (Production)
# =============================================================================

# Phase week boundaries
PHASE1_WEEKS = (1, 3)
PHASE2_WEEKS = (4, 15)
PHASE3_WEEKS = (16, 99)


def get_spread_calibration_for_week(
    week: int,
    phase_mode: str = "auto",
    phase1_policy: str = "skip",
    phase2_policy: str = "phase2_only",
    phase3_policy: str = "phase2",
    artifact_dir: str = "data/spread_selection/artifacts",
) -> Optional[CalibrationResult]:
    """Get the correct calibration for a given week with phase routing.

    This is the production entrypoint for calibration selection.

    Args:
        week: Game week (1-15 regular, 16+ postseason)
        phase_mode: Routing mode:
            - "auto": Use policy args to determine calibration
            - "force_phase2": Always use Phase 2 calibration
            - "force_weighted": Always use weighted calibration
        phase1_policy: How to handle weeks 0-3:
            - "skip": Return None (engine should not generate EV bets)
            - "weighted": Use weighted calibration
            - "phase1_only": Use Phase 1-only calibration (not recommended)
        phase2_policy: How to handle weeks 4-15:
            - "phase2_only": Use Phase 2-only calibration (default)
            - "weighted": Use weighted calibration
        phase3_policy: How to handle weeks 16+ (postseason):
            - "phase2": Treat as Phase 2 (use phase2_only)
            - "skip": Return None (skip postseason)
            - "weighted": Use weighted calibration
        artifact_dir: Directory containing calibration artifacts

    Returns:
        CalibrationResult for the week, or None if policy is "skip"

    Raises:
        ValueError: If invalid policy or phase_mode

    Examples:
        >>> # Phase 2 game (week 10)
        >>> cal = get_spread_calibration_for_week(10)
        >>> cal.implied_breakeven_edge  # ~5.3 pts

        >>> # Phase 1 game (week 2) with skip policy
        >>> cal = get_spread_calibration_for_week(2, phase1_policy="skip")
        >>> cal is None  # True - engine should skip EV bets

        >>> # Force weighted for all weeks
        >>> cal = get_spread_calibration_for_week(2, phase_mode="force_weighted")
        >>> cal is not None  # True - weighted calibration returned
    """
    # Determine phase
    if week <= PHASE1_WEEKS[1]:
        phase = "phase1"
    elif week <= PHASE2_WEEKS[1]:
        phase = "phase2"
    else:
        phase = "phase3"

    # Handle force modes
    if phase_mode == "force_phase2":
        return get_default_spread_calibration("phase2", artifact_dir)
    elif phase_mode == "force_weighted":
        return get_default_spread_calibration("weighted", artifact_dir)
    elif phase_mode != "auto":
        raise ValueError(f"Unknown phase_mode: {phase_mode}")

    # Route based on phase and policy
    if phase == "phase1":
        if phase1_policy == "skip":
            return None
        elif phase1_policy == "weighted":
            return get_default_spread_calibration("weighted", artifact_dir)
        elif phase1_policy == "phase1_only":
            return get_default_spread_calibration("phase1", artifact_dir)
        else:
            raise ValueError(f"Unknown phase1_policy: {phase1_policy}")

    elif phase == "phase2":
        if phase2_policy == "phase2_only":
            return get_default_spread_calibration("phase2", artifact_dir)
        elif phase2_policy == "weighted":
            return get_default_spread_calibration("weighted", artifact_dir)
        else:
            raise ValueError(f"Unknown phase2_policy: {phase2_policy}")

    else:  # phase3 (postseason)
        if phase3_policy == "phase2":
            return get_default_spread_calibration("phase2", artifact_dir)
        elif phase3_policy == "skip":
            return None
        elif phase3_policy == "weighted":
            return get_default_spread_calibration("weighted", artifact_dir)
        else:
            raise ValueError(f"Unknown phase3_policy: {phase3_policy}")
