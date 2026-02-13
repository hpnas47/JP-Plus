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
    | actual_margin      | Positive = home won                    | +10 = home won by 10         |
    | cover_margin       | Positive = home covered                | +3 = home beat spread by 3   |

V1 Assumptions:
    - Constant juice: -110
    - No push modeling: p_push = 0
    - Pure selection layer: does NOT modify JP+ predictions
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit  # Logistic function
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


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
    """Result of walk-forward validation."""

    game_results: pd.DataFrame  # Per-game results with p_cover_no_push
    fold_summaries: list[dict]  # Per-fold calibration summaries
    overall_brier: float  # Brier score vs actual outcomes
    overall_log_loss: float  # Log loss vs actual outcomes
    brier_vs_constant: float  # Brier score of constant 0.5 baseline
    brier_vs_best_constant: float  # Brier score of best constant (empirical mean)
    brier_skill_score: float  # 1 - Brier_model / Brier_best_constant


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
) -> WalkForwardResult:
    """Season-level walk-forward calibration.

    For each evaluation season Y:
        1. Training: seasons < Y (optionally excluding specified years)
        2. Filter training: exclude pushes and edge_abs == 0
        3. Fit calibration on training
        4. Apply to season Y games with edge_abs > 0
        5. For NO_BET games: p_cover = None

    Args:
        all_games: DataFrame with all games (normalized)
        min_train_seasons: Minimum seasons required for training
        exclude_covid: If True, exclude 2020 from evaluation folds
        exclude_years_from_training: Years to exclude from training data
            (e.g., [2022] to exclude 2022's negative-slope data)

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

    for eval_year in eval_years:
        if eval_year < first_eval_year:
            continue

        # Training: all years strictly before eval_year, excluding specified years
        train_mask = (all_games["year"] < eval_year) & (~all_games["year"].isin(exclude_from_train))
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

        # Predict on eval set
        # Only games with edge_abs > 0 and not push get predictions
        bet_mask = (eval_df["edge_abs"] > 0) & (~eval_df["push"])
        eval_df["p_cover_no_push"] = np.nan

        if bet_mask.sum() > 0:
            edge_vals = eval_df.loc[bet_mask, "edge_abs"].values
            p_cover = predict_cover_probability(edge_vals, calibration)
            eval_df.loc[bet_mask, "p_cover_no_push"] = p_cover

        # Add fold metadata
        eval_df["fold_intercept"] = calibration.intercept
        eval_df["fold_slope"] = calibration.slope
        eval_df["fold_breakeven_edge"] = calibration.implied_breakeven_edge
        eval_df["fold_p_cover_at_zero"] = calibration.p_cover_at_zero

        results.append(eval_df)

        # Fold summary
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
    )


def breakeven_prob(juice: float = -110) -> float:
    """Breakeven probability for given juice.

    At -110: Need to risk 110 to win 100
    Breakeven = risk / (risk + win) = 110 / 210 = 52.38%

    Args:
        juice: American odds (negative, e.g., -110)

    Returns:
        Breakeven probability as decimal (e.g., 0.5238)
    """
    if juice >= 0:
        raise ValueError(f"Expected negative juice, got {juice}")
    risk = abs(juice)
    win = 100
    return risk / (risk + win)


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
        juice: American odds (negative, e.g., -110)

    Returns:
        EV as fraction of stake (e.g., 0.03 = +3% EV)
    """
    if juice >= 0:
        raise ValueError(f"Expected negative juice, got {juice}")

    # Adjust for push probability
    # P(win) = P(cover | no push) * (1 - p_push)
    # P(lose) = P(not cover | no push) * (1 - p_push)
    # P(push) = p_push (returns stake, no win/loss)
    p_win = p_cover_no_push * (1 - p_push)
    p_lose = (1 - p_cover_no_push) * (1 - p_push)

    # Payout and stake
    risk = abs(juice)
    payout = 100 / risk  # Win per dollar risked

    ev = p_win * payout - p_lose * 1.0
    return ev


def calculate_ev_vectorized(
    p_cover_no_push: np.ndarray,
    p_push: float = 0.0,
    juice: float = -110,
) -> np.ndarray:
    """Vectorized EV calculation.

    Args:
        p_cover_no_push: Array of P(cover | no push)
        p_push: P(push) - V1 always 0
        juice: American odds

    Returns:
        Array of EV values
    """
    if juice >= 0:
        raise ValueError(f"Expected negative juice, got {juice}")

    p_win = p_cover_no_push * (1 - p_push)
    p_lose = (1 - p_cover_no_push) * (1 - p_push)

    risk = abs(juice)
    payout = 100 / risk

    return p_win * payout - p_lose


def get_spread_bucket(edge_abs: float) -> str:
    """Categorize edge_abs into diagnostic buckets.

    Args:
        edge_abs: Absolute edge value

    Returns:
        Bucket label: "[0,3)", "[3,5)", "[5,7)", "[7,10)", "[10,+)"
    """
    if edge_abs < 3:
        return "[0,3)"
    elif edge_abs < 5:
        return "[3,5)"
    elif edge_abs < 7:
        return "[5,7)"
    elif edge_abs < 10:
        return "[7,10)"
    else:
        return "[10,+)"


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
    bucket_order = ["[0,3)", "[3,5)", "[5,7)", "[7,10)", "[10,+)"]

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
