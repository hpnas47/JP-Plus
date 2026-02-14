"""Phase 2 Calibration for Totals EV Engine.

This module provides calibration of sigma_total, reliability scaling, and
EV/Kelly parameters using historical walk-forward residuals and betting outcomes.

Key Components:
1. Residual collection from walk-forward backtest
2. Multiple sigma estimators (global, robust, week-bucket, reliability-scaled)
3. Probability calibration via interval coverage
4. EV/Kelly backtest with historical lines (when available)
5. Parameter tuning framework

Usage:
    from src.spread_selection.totals_calibration import (
        collect_walk_forward_residuals,
        calibrate_sigma,
        evaluate_interval_coverage,
        TotalsCalibrationConfig,
        load_calibration,
    )
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm as scipy_norm
    from scipy.stats import median_abs_deviation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TotalsCalibrationConfig:
    """Configuration for calibrated totals EV evaluation.

    This is the output of calibration - the parameters to use in production.

    Calibration Modes:
        - "model_only": Sigma calibrated on mu_model - actual_total residuals
        - "weather_adjusted": Sigma calibrated on (mu_model + weather_adj) - actual_total

    For 2026 production:
        - If betting WITH weather adjustments, use weather_adjusted sigma if available
        - If weather_adjusted not available, use model_only sigma * fallback_multiplier
    """
    # Calibration mode: which mu definition was used for residuals
    calibration_mode: str = "model_only"  # "model_only" or "weather_adjusted"

    # Sigma settings
    sigma_mode: str = "fixed"  # "fixed", "week_bucket", "reliability_scaled"
    sigma_base: float = 13.0

    # Week bucket multipliers (sigma = sigma_base * multiplier)
    # Keys: "1-2", "3-5", "6-9", "10-14", "15+"
    week_bucket_multipliers: dict = field(default_factory=lambda: {
        "1-2": 1.3,
        "3-5": 1.1,
        "6-9": 1.0,
        "10-14": 1.0,
        "15+": 1.1,
    })

    # Reliability scaling: sigma_used = sigma_base * (1 + k*(1 - rel_game))
    reliability_k: float = 0.5  # Up to 50% inflation when reliability=0
    reliability_sigma_min: float = 10.0
    reliability_sigma_max: float = 25.0

    # EV thresholds
    ev_min: float = 0.02
    ev_min_phase1: float = 0.05  # Higher threshold for Phase 1

    # Kelly settings
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.02

    # Calibration metadata
    years_used: list = field(default_factory=lambda: [2022, 2023, 2024, 2025])
    n_games_calibrated: int = 0
    calibration_date: str = ""
    has_weather_data: bool = False  # Whether historical weather was available

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TotalsCalibrationConfig":
        """Create from dictionary, ignoring unknown keys for forward compatibility."""
        valid_keys = {f.name for f in fields(cls)}
        unknown_keys = set(d.keys()) - valid_keys
        if unknown_keys:
            logger.warning(
                f"TotalsCalibrationConfig.from_dict: ignoring unknown keys {unknown_keys} "
                f"(may be from newer code version)"
            )
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class SigmaEstimate:
    """Result of a sigma estimation method."""
    name: str
    sigma: float
    method: str
    n_games: int
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IntervalCoverageResult:
    """Result of interval coverage evaluation."""
    target_coverage: float
    empirical_coverage: float
    error: float  # empirical - target
    n_games: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibrationReport:
    """Full calibration report."""
    sigma_estimates: list  # List of SigmaEstimate
    coverage_results: dict  # sigma -> list of IntervalCoverageResult
    best_sigma: float
    best_sigma_method: str
    recommended_config: TotalsCalibrationConfig

    # ROI metrics (if available)
    roi_by_sigma: dict = field(default_factory=dict)  # sigma -> ROI stats

    def to_dict(self) -> dict:
        return {
            "sigma_estimates": [s.to_dict() for s in self.sigma_estimates],
            "coverage_results": {
                str(k): [r.to_dict() for r in v]
                for k, v in self.coverage_results.items()
            },
            "best_sigma": self.best_sigma,
            "best_sigma_method": self.best_sigma_method,
            "recommended_config": self.recommended_config.to_dict(),
            "roi_by_sigma": self.roi_by_sigma,
        }


# =============================================================================
# Residual Collection
# =============================================================================

def collect_walk_forward_residuals(
    preds_df: pd.DataFrame,
    calibration_mode: str = "model_only",
) -> pd.DataFrame:
    """Prepare residuals DataFrame from walk-forward predictions.

    Args:
        preds_df: DataFrame from backtest_totals with columns:
            - year, week, home_team, away_team
            - predicted_total (raw model output, mu_model)
            - weather_adj (optional, for weather-adjusted mode)
            - actual_total
            - vegas_total_close, vegas_total_open (optional)
        calibration_mode:
            - "model_only": error = predicted_total - actual_total
            - "weather_adjusted": error = (predicted_total + weather_adj) - actual_total

    Returns:
        DataFrame with residual analysis columns added:
            - mu_model: raw model prediction
            - weather_adj: weather adjustment (0.0 if not available)
            - mu_used: mu_model + weather_adj (matches what we bet with)
            - error: mu_used - actual_total (signed)
            - error_model_only: mu_model - actual_total (for comparison)
            - abs_error: |error|
            - week_bucket: categorized week
            - phase: Phase 1/2/3
    """
    df = preds_df.copy()

    # Identify mu_model column
    if 'mu_model' in df.columns:
        mu_model_col = 'mu_model'
    elif 'predicted_total' in df.columns:
        mu_model_col = 'predicted_total'
    elif 'adjusted_total' in df.columns:
        # Legacy: adjusted_total was used before weather separation
        mu_model_col = 'adjusted_total'
        logger.warning(
            "Using 'adjusted_total' as mu_model - this may include weather adjustments. "
            "For proper calibration, use 'predicted_total' column."
        )
    else:
        raise ValueError("preds_df must have 'predicted_total' or 'mu_model' column")

    df['mu_model'] = df[mu_model_col]

    # Get weather adjustment (default to 0.0 if not available)
    if 'weather_adj' in df.columns:
        df['weather_adj'] = df['weather_adj'].fillna(0.0)
        has_weather = df['weather_adj'].abs().sum() > 0
    elif 'weather_adjustment' in df.columns:
        df['weather_adj'] = df['weather_adjustment'].fillna(0.0)
        has_weather = df['weather_adj'].abs().sum() > 0
    else:
        df['weather_adj'] = 0.0
        has_weather = False

    # Compute mu_used based on calibration mode
    if calibration_mode == "weather_adjusted":
        if not has_weather:
            logger.warning(
                "calibration_mode='weather_adjusted' but no weather data found. "
                "Falling back to model_only calibration."
            )
        df['mu_used'] = df['mu_model'] + df['weather_adj']
    else:  # model_only
        df['mu_used'] = df['mu_model']

    # Compute error columns
    df['error'] = df['mu_used'] - df['actual_total']
    df['error_model_only'] = df['mu_model'] - df['actual_total']
    df['abs_error'] = df['error'].abs()

    # Assign week buckets
    def get_week_bucket(week: int) -> str:
        if week <= 2:
            return "1-2"
        elif week <= 5:
            return "3-5"
        elif week <= 9:
            return "6-9"
        elif week <= 14:
            return "10-14"
        else:
            return "15+"

    df['week_bucket'] = df['week'].apply(get_week_bucket)

    # Assign phase
    def get_phase(week: int) -> str:
        if week <= 3:
            return "Phase 1"
        elif week <= 15:
            return "Phase 2"
        else:
            return "Phase 3"

    df['phase'] = df['week'].apply(get_phase)

    return df


# =============================================================================
# Sigma Estimators
# =============================================================================

def estimate_sigma_global(residuals: pd.DataFrame) -> SigmaEstimate:
    """Global sigma = std(error)."""
    sigma = float(residuals['error'].std())
    return SigmaEstimate(
        name="global",
        sigma=sigma,
        method="std(error)",
        n_games=len(residuals),
        description="Standard deviation of all residuals",
    )


def estimate_sigma_robust(residuals: pd.DataFrame, winsorize_pct: float = 0.05) -> SigmaEstimate:
    """Robust sigma using MAD or winsorized std.

    Args:
        residuals: DataFrame with 'error' column
        winsorize_pct: Percentile to winsorize at each tail (e.g., 0.05 = 5%)
    """
    errors = residuals['error'].values

    if SCIPY_AVAILABLE:
        # Use median absolute deviation (robust)
        mad = median_abs_deviation(errors, scale='normal')
        sigma = float(mad)
        method = "MAD (scaled)"
    else:
        # Fallback: winsorized std
        lower = np.percentile(errors, winsorize_pct * 100)
        upper = np.percentile(errors, (1 - winsorize_pct) * 100)
        clipped = np.clip(errors, lower, upper)
        sigma = float(clipped.std())
        method = f"winsorized std ({winsorize_pct:.0%})"

    return SigmaEstimate(
        name="robust",
        sigma=sigma,
        method=method,
        n_games=len(residuals),
        description="Robust sigma estimate less sensitive to outliers",
    )


def estimate_sigma_by_week_bucket(residuals: pd.DataFrame) -> dict[str, SigmaEstimate]:
    """Compute sigma for each week bucket.

    Returns:
        Dict mapping bucket name to SigmaEstimate
    """
    results = {}

    for bucket, group in residuals.groupby('week_bucket'):
        sigma = float(group['error'].std())
        results[bucket] = SigmaEstimate(
            name=f"week_bucket_{bucket}",
            sigma=sigma,
            method="std(error) by week bucket",
            n_games=len(group),
            description=f"Sigma for weeks {bucket}",
        )

    return results


def estimate_sigma_by_phase(residuals: pd.DataFrame) -> dict[str, SigmaEstimate]:
    """Compute sigma for each phase."""
    results = {}

    for phase, group in residuals.groupby('phase'):
        sigma = float(group['error'].std())
        results[phase] = SigmaEstimate(
            name=f"phase_{phase.replace(' ', '_')}",
            sigma=sigma,
            method="std(error) by phase",
            n_games=len(group),
            description=f"Sigma for {phase}",
        )

    return results


def compute_all_sigma_estimates(residuals: pd.DataFrame) -> list[SigmaEstimate]:
    """Compute all sigma estimates.

    Returns:
        List of SigmaEstimate objects for comparison
    """
    estimates = []

    # Global
    estimates.append(estimate_sigma_global(residuals))

    # Robust
    estimates.append(estimate_sigma_robust(residuals))

    # By week bucket
    week_estimates = estimate_sigma_by_week_bucket(residuals)
    estimates.extend(week_estimates.values())

    # By phase
    phase_estimates = estimate_sigma_by_phase(residuals)
    estimates.extend(phase_estimates.values())

    return estimates


# =============================================================================
# Probability Calibration (Interval Coverage)
# =============================================================================

def normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    if SCIPY_AVAILABLE:
        return float(scipy_norm.cdf(x))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def evaluate_interval_coverage(
    residuals: pd.DataFrame,
    sigma: float,
    targets: list[float] = None,
) -> list[IntervalCoverageResult]:
    """Evaluate interval coverage for a given sigma.

    For each target coverage (e.g., 50%, 68%, 80%, 90%), compute:
    - What fraction of actual totals fall within the predicted interval

    Args:
        residuals: DataFrame with 'error' column (predicted - actual)
        sigma: Standard deviation to use for intervals
        targets: List of target coverage levels (default: [0.5, 0.68, 0.80, 0.90])

    Returns:
        List of IntervalCoverageResult for each target
    """
    if targets is None:
        targets = [0.50, 0.68, 0.80, 0.90, 0.95]

    errors = residuals['error'].values
    n = len(errors)

    results = []
    for target in targets:
        # For a Normal distribution, the z-score for coverage c is:
        # z = Phi^(-1)((1 + c) / 2)
        # Using symmetry: |error| < z * sigma covers fraction c
        if SCIPY_AVAILABLE:
            z = scipy_norm.ppf((1 + target) / 2)
        else:
            # Approximation for common values
            z_lookup = {0.50: 0.674, 0.68: 1.0, 0.80: 1.282, 0.90: 1.645, 0.95: 1.96}
            z = z_lookup.get(target, 1.645)

        threshold = z * sigma
        empirical = (np.abs(errors) <= threshold).mean()

        results.append(IntervalCoverageResult(
            target_coverage=target,
            empirical_coverage=float(empirical),
            error=float(empirical - target),
            n_games=n,
        ))

    return results


def compute_coverage_score(coverage_results: list[IntervalCoverageResult]) -> float:
    """Compute total miscoverage score (lower is better).

    Sum of squared errors between target and empirical coverage.
    """
    return sum(r.error ** 2 for r in coverage_results)


# =============================================================================
# Reliability Scaling
# =============================================================================

def compute_game_reliability(
    home_games_played: int,
    away_games_played: int,
    max_games: int = 8,
) -> float:
    """Compute reliability score for a game based on team games played.

    Args:
        home_games_played: Number of games home team has played
        away_games_played: Number of games away team has played
        max_games: Number of games for full reliability (default: 8)

    Returns:
        Reliability score in [0, 1]
    """
    rel_home = min(1.0, max(0.0, (home_games_played - 1) / (max_games - 1)))
    rel_away = min(1.0, max(0.0, (away_games_played - 1) / (max_games - 1)))

    # Use minimum (most conservative) or average
    return min(rel_home, rel_away)


def compute_scaled_sigma(
    sigma_base: float,
    reliability: float,
    k: float = 0.5,
    sigma_min: float = 10.0,
    sigma_max: float = 25.0,
) -> float:
    """Compute reliability-scaled sigma.

    Formula: sigma_used = sigma_base * (1 + k * (1 - reliability))

    Args:
        sigma_base: Base sigma estimate
        reliability: Game reliability in [0, 1]
        k: Scaling factor (0.5 = up to 50% inflation at reliability=0)
        sigma_min: Minimum allowed sigma
        sigma_max: Maximum allowed sigma

    Returns:
        Scaled sigma, bounded by [sigma_min, sigma_max]
    """
    sigma = sigma_base * (1 + k * (1 - reliability))
    return max(sigma_min, min(sigma_max, sigma))


def get_sigma_for_week_bucket(
    week: int,
    sigma_base: float,
    multipliers: dict[str, float],
) -> float:
    """Get sigma for a specific week using bucket multipliers.

    Args:
        week: Week number
        sigma_base: Base sigma
        multipliers: Dict mapping bucket name to multiplier

    Returns:
        sigma_base * multiplier for the appropriate bucket
    """
    if week <= 2:
        bucket = "1-2"
    elif week <= 5:
        bucket = "3-5"
    elif week <= 9:
        bucket = "6-9"
    elif week <= 14:
        bucket = "10-14"
    else:
        bucket = "15+"

    multiplier = multipliers.get(bucket, 1.0)
    return sigma_base * multiplier


# =============================================================================
# EV/ROI Backtest (with historical lines)
# =============================================================================

def american_to_decimal(odds: int | float) -> float:
    """Convert American odds to decimal odds."""
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / abs(odds)


def calculate_totals_probabilities(
    mu: float,
    line: float,
    sigma: float,
    side: str,
) -> tuple[float, float, float]:
    """Calculate win/loss/push probabilities for totals bet.

    Args:
        mu: Model predicted total
        line: Vegas line (integer or half-point)
        sigma: Standard deviation (must be positive)
        side: "OVER" or "UNDER"

    Returns:
        (p_win, p_loss, p_push)

    Raises:
        ValueError: If sigma is not positive
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    is_half_point = (line % 1.0) != 0.0

    if is_half_point:
        z = (line - mu) / sigma
        p_under = normal_cdf(z)
        p_over = 1.0 - p_under
        p_push = 0.0
    else:
        z_low = (line - 0.5 - mu) / sigma
        z_high = (line + 0.5 - mu) / sigma
        p_under = normal_cdf(z_low)
        p_over = 1.0 - normal_cdf(z_high)
        p_push = normal_cdf(z_high) - normal_cdf(z_low)

    if side.upper() == "OVER":
        return (p_over, p_under, p_push)
    else:
        return (p_under, p_over, p_push)


def _vectorized_normal_cdf(x: np.ndarray) -> np.ndarray:
    """Vectorized standard normal CDF for arrays."""
    if SCIPY_AVAILABLE:
        return scipy_norm.cdf(x)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def backtest_ev_roi(
    preds_df: pd.DataFrame,
    sigma: float,
    ev_min: float = 0.02,
    kelly_fraction: float = 0.25,
    max_bet_fraction: float = 0.02,
    default_odds: int = -110,
    mu_column: Optional[str] = None,
) -> dict:
    """Backtest EV-based betting with historical results (vectorized).

    Args:
        preds_df: DataFrame with predicted_total, actual_total, vegas_total_close
        sigma: Sigma to use for probability calculations
        ev_min: Minimum EV to bet
        kelly_fraction: Fractional Kelly
        max_bet_fraction: Max stake as fraction of bankroll
        default_odds: Default American odds if not in data
        mu_column: Explicit column to use for mu. If None, falls back to
            'mu_used' -> 'adjusted_total' -> 'predicted_total' for backward compat.

    Returns:
        Dict with ROI metrics
    """
    # Use closing line for backtest
    vegas_col = 'vegas_total_close'
    if vegas_col not in preds_df.columns or preds_df[vegas_col].isna().all():
        return {"error": "No Vegas lines available"}

    # Resolve mu column
    if mu_column is not None:
        if mu_column not in preds_df.columns:
            return {"error": f"Specified mu_column '{mu_column}' not found"}
        pred_col = mu_column
    elif 'mu_used' in preds_df.columns:
        pred_col = 'mu_used'
    elif 'adjusted_total' in preds_df.columns:
        pred_col = 'adjusted_total'
    else:
        pred_col = 'predicted_total'

    valid = preds_df[preds_df[vegas_col].notna()].copy()
    if len(valid) == 0:
        return {"error": "No valid games with lines"}

    # Extract arrays for vectorized computation
    mu = valid[pred_col].values
    line = valid[vegas_col].values
    actual = valid['actual_total'].values
    years = valid['year'].values if 'year' in valid.columns else np.zeros(len(valid))
    weeks = valid['week'].values if 'week' in valid.columns else np.zeros(len(valid))

    # Determine best side (vectorized)
    edge_over = mu - line
    edge_under = line - mu
    pick_over = np.abs(edge_over) > np.abs(edge_under)
    side = np.where(pick_over, "OVER", "UNDER")
    edge = np.where(pick_over, edge_over, edge_under)

    # Filter: skip tiny edges
    edge_mask = np.abs(edge) >= 0.5
    if not edge_mask.any():
        return {"error": "No qualifying bets"}

    # Calculate probabilities (vectorized)
    is_half_point = (line % 1.0) != 0.0

    # For half-point lines
    z_half = (line - mu) / sigma
    p_under_half = _vectorized_normal_cdf(z_half)
    p_over_half = 1.0 - p_under_half
    p_push_half = np.zeros_like(mu)

    # For integer lines
    z_low = (line - 0.5 - mu) / sigma
    z_high = (line + 0.5 - mu) / sigma
    p_under_int = _vectorized_normal_cdf(z_low)
    p_over_int = 1.0 - _vectorized_normal_cdf(z_high)
    p_push_int = _vectorized_normal_cdf(z_high) - _vectorized_normal_cdf(z_low)

    # Select based on line type
    p_under = np.where(is_half_point, p_under_half, p_under_int)
    p_over = np.where(is_half_point, p_over_half, p_over_int)
    p_push = np.where(is_half_point, p_push_half, p_push_int)

    # Win/loss probabilities based on side
    p_win = np.where(pick_over, p_over, p_under)
    p_loss = np.where(pick_over, p_under, p_over)

    # Calculate EV (vectorized)
    odds_decimal = american_to_decimal(default_odds)
    b = odds_decimal - 1.0
    ev = p_win * b - p_loss

    # Filter: EV >= ev_min
    ev_mask = ev >= ev_min

    # Kelly stake (vectorized)
    numerator = p_win * b - p_loss
    denominator = b * p_win + p_loss
    kelly_mask = (denominator > 0) & (numerator > 0)

    f_star = np.where(kelly_mask, numerator / denominator, 0.0)
    f = np.minimum(kelly_fraction * f_star, max_bet_fraction)

    # Actual results (vectorized)
    is_push = actual == line
    is_win = np.where(
        pick_over,
        actual > line,
        actual < line
    )

    profit = np.where(
        is_push,
        0.0,
        np.where(is_win, f * b, -f)
    )

    outcome = np.where(
        is_push,
        "push",
        np.where(is_win, "win", "loss")
    )

    at_cap = f >= max_bet_fraction - 0.0001

    # Combined filter mask
    final_mask = edge_mask & ev_mask & kelly_mask

    if not final_mask.any():
        return {"error": "No qualifying bets"}

    # Build results DataFrame from filtered arrays
    results_df = pd.DataFrame({
        "year": years[final_mask],
        "week": weeks[final_mask],
        "side": side[final_mask],
        "edge": edge[final_mask],
        "ev": ev[final_mask],
        "stake_frac": f[final_mask],
        "outcome": outcome[final_mask],
        "profit": profit[final_mask],
        "at_cap": at_cap[final_mask],
    })

    # Compute metrics
    n_bets = len(results_df)
    total_stake = results_df['stake_frac'].sum()
    total_profit = results_df['profit'].sum()
    roi = total_profit / total_stake if total_stake > 0 else 0

    wins = (results_df['outcome'] == 'win').sum()
    losses = (results_df['outcome'] == 'loss').sum()
    pushes = (results_df['outcome'] == 'push').sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    cap_hit_rate = results_df['at_cap'].mean()

    # EV calibration: bin by EV decile and check monotonicity
    results_df['ev_decile'] = pd.qcut(results_df['ev'], q=5, labels=False, duplicates='drop')
    ev_by_decile = results_df.groupby('ev_decile').agg({
        'profit': 'sum',
        'stake_frac': 'sum',
        'ev': 'mean',
    })
    ev_by_decile['realized_roi'] = ev_by_decile['profit'] / ev_by_decile['stake_frac']

    return {
        "n_bets": n_bets,
        "n_bets_per_season": n_bets / len(preds_df['year'].unique()),
        "total_stake": total_stake,
        "total_profit": total_profit,
        "roi": roi,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "cap_hit_rate": cap_hit_rate,
        "mean_ev": results_df['ev'].mean(),
        "mean_edge": results_df['edge'].abs().mean(),
        "ev_by_decile": ev_by_decile.to_dict() if len(ev_by_decile) > 0 else {},
    }


# =============================================================================
# Parameter Tuning
# =============================================================================

def tune_sigma_for_coverage(
    residuals: pd.DataFrame,
    sigma_candidates: list[float] = None,
    targets: list[float] = None,
) -> tuple[float, dict]:
    """Find sigma that minimizes interval coverage error.

    Args:
        residuals: DataFrame with 'error' column
        sigma_candidates: List of sigma values to try (all must be positive)
        targets: Coverage targets to evaluate

    Returns:
        (best_sigma, coverage_results_by_sigma)

    Raises:
        ValueError: If any sigma candidate is not positive
    """
    if sigma_candidates is None:
        sigma_candidates = [s * 0.5 + 10.0 for s in range(21)]  # 10.0 to 20.0 step 0.5

    # Validate all sigma candidates are positive
    invalid = [s for s in sigma_candidates if s <= 0]
    if invalid:
        raise ValueError(f"All sigma candidates must be positive, got invalid values: {invalid}")

    if targets is None:
        targets = [0.50, 0.68, 0.80, 0.90, 0.95]

    results = {}
    best_sigma = sigma_candidates[0]
    best_score = float('inf')

    for sigma in sigma_candidates:
        coverage = evaluate_interval_coverage(residuals, sigma, targets)
        score = compute_coverage_score(coverage)
        results[sigma] = coverage

        if score < best_score:
            best_score = score
            best_sigma = sigma

    return best_sigma, results


def tune_sigma_for_roi(
    preds_df: pd.DataFrame,
    sigma_candidates: list[float] = None,
    ev_min: float = 0.02,
    mu_column: Optional[str] = None,
) -> tuple[float, dict]:
    """Find sigma that maximizes ROI (with constraints).

    Args:
        preds_df: DataFrame with predictions and lines
        sigma_candidates: List of sigma values to try (all must be positive)
        ev_min: Minimum EV threshold
        mu_column: Explicit mu column to use in backtest_ev_roi

    Returns:
        (best_sigma, roi_results_by_sigma)
        If no valid candidate found, roi_results will contain "no_valid_candidate": True

    Raises:
        ValueError: If any sigma candidate is not positive
    """
    if sigma_candidates is None:
        sigma_candidates = [s * 0.5 + 10.0 for s in range(21)]

    # Validate all sigma candidates are positive
    invalid = [s for s in sigma_candidates if s <= 0]
    if invalid:
        raise ValueError(f"All sigma candidates must be positive, got invalid values: {invalid}")

    results = {}
    best_sigma = sigma_candidates[0]
    best_roi = -float('inf')

    for sigma in sigma_candidates:
        roi_result = backtest_ev_roi(preds_df, sigma, ev_min=ev_min, mu_column=mu_column)
        results[sigma] = roi_result

        if "error" in roi_result:
            continue

        # Constraints
        if roi_result.get("cap_hit_rate", 1.0) > 0.30:
            continue  # Too many capped bets
        if roi_result.get("n_bets_per_season", 0) < 50:
            continue  # Not enough bets

        if roi_result.get("roi", -1) > best_roi:
            best_roi = roi_result["roi"]
            best_sigma = sigma

    # Check if any valid candidate was found
    if best_roi == -float('inf'):
        logger.warning(
            f"tune_sigma_for_roi: All {len(sigma_candidates)} sigma candidates rejected. "
            f"Constraints: cap_hit_rate <= 0.30, n_bets_per_season >= 50. "
            f"Returning first candidate ({sigma_candidates[0]}) as fallback."
        )
        results["no_valid_candidate"] = True

    return best_sigma, results


def compute_week_bucket_multipliers(
    residuals: pd.DataFrame,
    min_multiplier: float = 0.8,
    max_multiplier: float = 1.5,
) -> dict[str, float]:
    """Compute week bucket multipliers relative to global sigma.

    Returns multipliers such that sigma_bucket = sigma_global * multiplier.
    Multipliers are clamped to [min_multiplier, max_multiplier] to prevent
    extreme values from noisy buckets (especially "15+" with few games).

    Args:
        residuals: DataFrame with 'error' and 'week_bucket' columns
        min_multiplier: Minimum allowed multiplier (default 0.8)
        max_multiplier: Maximum allowed multiplier (default 1.5)

    Returns:
        Dict mapping bucket name to clamped multiplier
    """
    global_sigma = float(residuals['error'].std())
    bucket_estimates = estimate_sigma_by_week_bucket(residuals)

    multipliers = {}
    for bucket, estimate in bucket_estimates.items():
        raw_mult = estimate.sigma / global_sigma if global_sigma > 0 else 1.0
        clamped_mult = max(min_multiplier, min(max_multiplier, raw_mult))

        if clamped_mult != raw_mult:
            logger.warning(
                f"Week bucket '{bucket}' multiplier clamped: {raw_mult:.3f} -> {clamped_mult:.3f} "
                f"(n_games={estimate.n_games})"
            )

        multipliers[bucket] = float(clamped_mult)  # Ensure native Python float

    return multipliers


# =============================================================================
# Full Calibration
# =============================================================================

def run_full_calibration(
    preds_df: pd.DataFrame,
    sigma_candidates: list[float] = None,
    calibration_mode: str = "model_only",
) -> CalibrationReport:
    """Run full calibration and produce report.

    Args:
        preds_df: DataFrame from backtest_totals with predictions and results
        sigma_candidates: List of sigma values to try
        calibration_mode: "model_only" or "weather_adjusted"
            - model_only: calibrate on predicted_total - actual_total
            - weather_adjusted: calibrate on (predicted_total + weather_adj) - actual_total

    Returns:
        CalibrationReport with all results
    """
    if sigma_candidates is None:
        sigma_candidates = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 18.0, 19.0, 20.0]

    # Collect residuals with specified calibration mode
    # This resolves the mu column and stores it in 'mu_used'
    residuals = collect_walk_forward_residuals(preds_df, calibration_mode=calibration_mode)

    # Check if weather data was available
    has_weather = 'weather_adj' in residuals.columns and residuals['weather_adj'].abs().sum() > 0

    # Compute all sigma estimates
    sigma_estimates = compute_all_sigma_estimates(residuals)

    # Tune sigma for coverage
    best_sigma_coverage, coverage_results = tune_sigma_for_coverage(
        residuals, sigma_candidates
    )

    # Check if we have lines for ROI tuning
    has_lines = 'vegas_total_close' in residuals.columns and residuals['vegas_total_close'].notna().any()

    roi_by_sigma = {}
    best_sigma_roi = None
    roi_tuning_skipped = False
    if has_lines:
        # CRITICAL: Pass residuals (which has 'mu_used') and specify the mu_column
        # This ensures ROI backtest uses the same mu as sigma calibration
        best_sigma_roi, roi_by_sigma = tune_sigma_for_roi(
            residuals,
            sigma_candidates,
            mu_column='mu_used',  # Matches what collect_walk_forward_residuals computed
        )
        # Check if all candidates were rejected
        if roi_by_sigma.get("no_valid_candidate", False):
            roi_tuning_skipped = True
            logger.info(
                "ROI tuning found no valid candidates; falling back to coverage-optimized sigma"
            )

    # Choose best sigma (prefer ROI-tuned if available and better)
    if (best_sigma_roi is not None
        and not roi_tuning_skipped
        and roi_by_sigma.get(best_sigma_roi, {}).get("roi", -1) > 0):
        best_sigma = best_sigma_roi
        best_method = "ROI-optimized"
    else:
        best_sigma = best_sigma_coverage
        best_method = "coverage-optimized"

    # Compute week bucket multipliers
    week_multipliers = compute_week_bucket_multipliers(residuals)

    # Build recommended config
    recommended_config = TotalsCalibrationConfig(
        calibration_mode=calibration_mode,
        sigma_mode="fixed",  # Default; can upgrade to week_bucket or reliability_scaled
        sigma_base=best_sigma,
        week_bucket_multipliers=week_multipliers,
        years_used=sorted(preds_df['year'].unique().tolist()),
        n_games_calibrated=len(preds_df),
        calibration_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        has_weather_data=has_weather,
    )

    # Convert coverage results for serialization
    coverage_results_clean = {
        sigma: results for sigma, results in coverage_results.items()
    }

    return CalibrationReport(
        sigma_estimates=sigma_estimates,
        coverage_results=coverage_results_clean,
        best_sigma=best_sigma,
        best_sigma_method=best_method,
        recommended_config=recommended_config,
        roi_by_sigma={str(k): v for k, v in roi_by_sigma.items()},
    )


# =============================================================================
# Load/Save Calibration
# =============================================================================

def get_calibration_artifact_path(
    base_dir: str | Path = "artifacts",
    years: list[int] = None,
    calibration_mode: str = "model_only",
) -> Path:
    """Get the standard artifact path for a calibration config.

    Args:
        base_dir: Base directory for artifacts
        years: List of years used for calibration
        calibration_mode: "model_only" or "weather_adjusted"

    Returns:
        Path like: artifacts/totals_sigma_model_only_2022_2025.json
    """
    base_dir = Path(base_dir)
    if years is None:
        years = [2022, 2023, 2024, 2025]
    year_range = f"{min(years)}_{max(years)}"
    return base_dir / f"totals_sigma_{calibration_mode}_{year_range}.json"


def save_calibration(
    config: TotalsCalibrationConfig,
    path: str | Path = None,
) -> Path:
    """Save calibration config to JSON.

    Args:
        config: Calibration config to save
        path: Output path. If None, uses standard naming convention.

    Returns:
        Path where config was saved
    """
    if path is None:
        path = get_calibration_artifact_path(
            calibration_mode=config.calibration_mode,
            years=config.years_used,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    logger.info(f"Saved calibration to {path} (mode={config.calibration_mode})")
    return path


def load_calibration(path: str | Path) -> TotalsCalibrationConfig:
    """Load calibration config from JSON."""
    path = Path(path)

    with open(path, 'r') as f:
        data = json.load(f)

    return TotalsCalibrationConfig.from_dict(data)


def select_calibration_for_runtime(
    use_weather: bool = True,
    base_dir: str | Path = "artifacts",
    years: list[int] = None,
    fallback_multiplier: float = 1.05,
) -> tuple[TotalsCalibrationConfig, str]:
    """Select appropriate calibration config for runtime based on weather usage.

    Args:
        use_weather: Whether weather adjustments will be applied at runtime
        base_dir: Directory containing calibration artifacts
        years: Years to look for (for naming convention)
        fallback_multiplier: If betting with weather but only model_only calibration
            exists, inflate sigma by this factor as conservative fallback

    Returns:
        Tuple of (config, selection_reason):
        - config: The calibration config to use
        - selection_reason: Explanation of why this config was selected

    Raises:
        FileNotFoundError: If no calibration artifacts found
    """
    base_dir = Path(base_dir)
    if years is None:
        years = [2022, 2023, 2024, 2025]

    weather_path = get_calibration_artifact_path(base_dir, years, "weather_adjusted")
    model_path = get_calibration_artifact_path(base_dir, years, "model_only")

    if use_weather:
        # Prefer weather-adjusted calibration when betting with weather
        if weather_path.exists():
            config = load_calibration(weather_path)
            return (config, "weather_adjusted calibration available and selected")
        elif model_path.exists():
            # Fallback: use model_only but inflate sigma
            config = load_calibration(model_path)
            original_sigma = config.sigma_base
            config.sigma_base = config.sigma_base * fallback_multiplier
            logger.warning(
                f"Betting with weather but only model_only calibration available. "
                f"Inflating sigma {original_sigma:.2f} -> {config.sigma_base:.2f} "
                f"(multiplier={fallback_multiplier:.2f})"
            )
            return (config, f"model_only with {fallback_multiplier}x sigma inflation (no weather calibration)")
        else:
            raise FileNotFoundError(
                f"No calibration artifacts found at {weather_path} or {model_path}"
            )
    else:
        # Use model_only calibration when not using weather
        if model_path.exists():
            config = load_calibration(model_path)
            return (config, "model_only calibration selected (no weather)")
        elif weather_path.exists():
            # Weather calibration exists but we're not using weather - still usable
            config = load_calibration(weather_path)
            logger.info(
                "Using weather_adjusted calibration even though weather not applied "
                "(slightly conservative but acceptable)"
            )
            return (config, "weather_adjusted calibration used without weather (conservative)")
        else:
            raise FileNotFoundError(
                f"No calibration artifacts found at {model_path} or {weather_path}"
            )


# =============================================================================
# Runtime Integration Helpers
# =============================================================================

def get_sigma_for_game(
    config: TotalsCalibrationConfig,
    week: int,
    home_games_played: int = 8,
    away_games_played: int = 8,
) -> float:
    """Get appropriate sigma for a game based on config mode.

    Args:
        config: Calibration config
        week: Game week
        home_games_played: Home team's games played
        away_games_played: Away team's games played

    Returns:
        Sigma to use for this game
    """
    if config.sigma_mode == "fixed":
        return config.sigma_base

    elif config.sigma_mode == "week_bucket":
        return get_sigma_for_week_bucket(
            week, config.sigma_base, config.week_bucket_multipliers
        )

    elif config.sigma_mode == "reliability_scaled":
        reliability = compute_game_reliability(home_games_played, away_games_played)
        return compute_scaled_sigma(
            config.sigma_base,
            reliability,
            config.reliability_k,
            config.reliability_sigma_min,
            config.reliability_sigma_max,
        )

    else:
        logger.warning(f"Unknown sigma_mode: {config.sigma_mode}, using fixed")
        return config.sigma_base


# =============================================================================
# CLI Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Totals Calibration Module - Self Test")
    print("=" * 60)

    # Create synthetic residuals for testing
    np.random.seed(42)
    n = 500
    errors = np.random.normal(0, 13.0, n)  # True sigma = 13

    test_df = pd.DataFrame({
        'year': np.random.choice([2022, 2023, 2024, 2025], n),
        'week': np.random.randint(1, 16, n),
        'adjusted_total': 50 + errors,
        'actual_total': 50.0,
        'vegas_total_close': 50.0,
    })

    # Run calibration
    print("\nRunning calibration on synthetic data...")
    report = run_full_calibration(test_df)

    print(f"\nSigma Estimates:")
    for est in report.sigma_estimates:
        print(f"  {est.name}: {est.sigma:.2f} ({est.n_games} games)")

    print(f"\nBest Sigma: {report.best_sigma:.2f} ({report.best_sigma_method})")

    print(f"\nInterval Coverage at sigma={report.best_sigma:.1f}:")
    for result in report.coverage_results.get(report.best_sigma, []):
        print(f"  {result.target_coverage:.0%}: empirical={result.empirical_coverage:.1%} (error={result.error:+.1%})")

    print(f"\nRecommended Config:")
    print(f"  sigma_mode: {report.recommended_config.sigma_mode}")
    print(f"  sigma_base: {report.recommended_config.sigma_base:.2f}")
    print(f"  week_bucket_multipliers: {report.recommended_config.week_bucket_multipliers}")

    print("\n" + "=" * 60)
    print("Self Test Complete")
    print("=" * 60)
