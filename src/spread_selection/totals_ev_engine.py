"""Totals EV Execution Engine for over/under betting.

This module provides:
1. TotalsBetRecommendation - full recommendation for a single totals bet
2. evaluate_totals_markets - evaluate all markets for a slate of games
3. Normal CDF probability modeling with push-awareness (whole-number totals)
4. Kelly staking with push-adjusted formula

Output Lists:
- Primary Edge Execution Engine: EV-qualified bets (EV >= threshold)
- 5+ Edge: 5+ point edge bets that don't meet EV cut (diagnostic)

Key Differences from Spread Engine:
- Uses analytical Normal CDF (not historical logistic calibration)
- Includes Kelly staking (spread engine lacks this)
- Push probability is analytical (from Normal CDF) not empirical
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

# Try scipy for accurate Normal CDF, fallback to math.erf
try:
    from scipy.stats import norm as scipy_norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TotalMarket:
    """Market data for a single total line at one book."""
    book: str
    line: float  # Typically whole number, but allow .5 for half-point lines
    odds_over: int | float  # American odds for Over
    odds_under: int | float  # American odds for Under


@dataclass
class TotalsEvent:
    """Event with multiple market options (books/lines)."""
    event_id: str
    home_team: str
    away_team: str
    year: Optional[int] = None
    week: Optional[int] = None
    weather_adjustment: float = 0.0
    markets: list[TotalMarket] = field(default_factory=list)


@dataclass
class TotalsEVConfig:
    """Configuration for totals EV evaluation.

    CFB-SPECIFIC DEFAULTS:
        These defaults are calibrated for college football:
        - sigma_total=13.0: Standard deviation of PREDICTION ERRORS (model predicted
          total minus actual total), not raw game total variance. Typical CFB backtest
          residuals show ~11-14pt std dev. This uncalibrated default matches
          totals_calibration.py. Replace with calibrated sigma when available.
        - baseline_prior_per_team=30.0: CFB average is ~30 PPG per team (~60 total).
        - baseline_sanity_min/max=7.0/55.0: CFB per-team totals routinely range from
          low-scoring defensive games (~35 total) to Big 12 shootouts (80+ total).
        - enable_baseline_blend=False: Global game counts don't reflect per-team sample
          stability across 134 heterogeneous teams. Disabled by default; model
          predictions are trusted directly without regression to prior.
        - baseline_blend_n0=500: Safety fallback if blending is ever re-enabled.
          High value ensures minimal prior influence given CFB's team heterogeneity.

    Attributes:
        sigma_total: Standard deviation of PREDICTION ERRORS (model - actual), NOT
            raw game total variance. Default 13.0 matches totals_calibration.py.
            Should be replaced by calibrated sigma from totals_calibration when available.
        use_adjusted_total: DEPRECATED - use use_weather_adjustment instead.

        ev_min: Minimum EV for Primary Edge Execution Engine qualification.
        edge_pts_min: Minimum edge in points for 5+ Edge list qualification.

        kelly_fraction: Fraction of full Kelly to use (e.g., 0.25 = quarter Kelly).
        max_bet_fraction: Maximum stake as fraction of bankroll.
        bankroll: Bankroll size for stake calculation.
        min_bet: Minimum bet size (stakes below this become 0).
        round_to: Round stakes to nearest this value (e.g., 1.0 = nearest dollar).

        one_bet_per_market: Pick best side (Over/Under) per (event, book, line).
        one_bet_per_event: Further filter to single best EV per event.

        listB_include_positive_ev_below_cut: Include EV > 0 but < ev_min in 5+ Edge list.

        allow_scipy: Whether to use scipy.stats.norm.cdf if available.
        sort_key: Sort key for output ("ev" or "edge_pts").

    Weather Adjustment (2026 Production):
        use_weather_adjustment: Whether to apply external weather adjustments (default True).
        weather_max_adjustment: Maximum absolute weather adjustment in points (guardrail).
            Set to 10.0 to prevent extreme adjustments. Can be overridden.
        weather_sigma_fallback_multiplier: If betting with weather but only have model-only
            sigma calibration, inflate sigma by this factor as conservative fallback.

    Phase 1 Baseline Blending:
        enable_baseline_blend: Enable baseline prior blending for early-season stability.
            Default False for CFB - global game counts don't reflect per-team stability
            across 134 heterogeneous teams. When False, blend weight is always 0.0
            (fully trust model, no regression to prior).
        baseline_prior_per_team: Prior expectation for per-team PPG (default 30.0 for CFB).
        baseline_blend_n0: Sample size parameter for blend decay (default 500 for CFB).
            High value ensures minimal prior influence if blending is ever re-enabled.
        baseline_blend_mode: "rational" (n0/(n0+n)) or "exp" (exp(-n/n0)).

    Phase 1 Guardrails:
        min_train_games_for_staking: Minimum training games to generate stakes.
        baseline_sanity_min: Minimum acceptable per-team baseline (7.0 for CFB).
        baseline_sanity_max: Maximum acceptable per-team baseline (55.0 for CFB).
        diagnostic_only_mode: Force stake=0 for all bets (diagnostic output only).
        auto_diagnostic_if_guardrail_hit: Auto-enable diagnostic mode on guardrail violation.

    Mu Composition (Single Source of Truth):
        mu_model = TotalsModel.predict_total(...).predicted_total (raw model output)
        weather_adj = external adjustment (negative = lower scoring expected)
        mu_used = mu_model + weather_adj + baseline_shift (after all adjustments)

        Weather adjustment is applied AFTER model prediction but BEFORE baseline blending.
        Sign convention: negative weather_adj means "lower scoring expected" (reduce total).
    """
    # Model parameters (CFB defaults - see docstring for rationale)
    sigma_total: float = 13.0  # Prediction error std dev (matches totals_calibration.py)
    use_adjusted_total: bool = True  # DEPRECATED: kept for backwards compatibility

    # Weather adjustment (2026 production)
    use_weather_adjustment: bool = True
    weather_max_adjustment: float = 10.0  # Guardrail cap
    weather_sigma_fallback_multiplier: float = 1.05  # Inflate sigma if no weather-adjusted calibration

    # EV thresholds
    ev_min: float = 0.02
    edge_pts_min: float = 5.0

    # Kelly staking
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.02
    bankroll: float = 1000.0
    min_bet: float = 0.0
    round_to: float = 1.0

    # Selection options
    one_bet_per_market: bool = True
    one_bet_per_event: bool = False

    # 5+ Edge list behavior
    listB_include_positive_ev_below_cut: bool = True

    # Implementation options
    allow_scipy: bool = True
    sort_key: str = "ev"  # "ev" or "edge_pts"

    # Phase 1: Baseline Prior Blending (CFB defaults)
    enable_baseline_blend: bool = False  # Disabled for CFB - 134 teams too heterogeneous
    baseline_prior_per_team: float = 30.0  # CFB average ~30 PPG per team
    baseline_blend_n0: float = 500.0  # High value if blending ever re-enabled
    baseline_blend_mode: str = "rational"  # "rational" or "exp"

    # Phase 1: Guardrails (CFB defaults)
    min_train_games_for_staking: int = 80
    baseline_sanity_min: float = 7.0  # CFB can have very low-scoring games
    baseline_sanity_max: float = 55.0  # CFB shootouts can exceed 40 PPG per team
    diagnostic_only_mode: bool = False
    auto_diagnostic_if_guardrail_hit: bool = True

    # Phase 2: Sigma Calibration
    sigma_mode: str = "fixed"  # "fixed", "week_bucket", "reliability_scaled"

    # Week bucket multipliers (sigma = sigma_total * multiplier)
    # Keys must match totals_calibration.py: "0-2", "3-5", "6-9", "10-14", "15+"
    week_bucket_multipliers: dict = field(default_factory=lambda: {
        "0-2": 1.3,  # Includes CFB week 0 (pre-Labor Day games)
        "3-5": 1.1,
        "6-9": 1.0,
        "10-14": 1.0,
        "15+": 1.1,
    })

    # Reliability scaling: sigma_used = sigma_total * (1 + k*(1 - rel_game))
    reliability_k: float = 0.5  # Up to 50% inflation when reliability=0
    reliability_sigma_min: float = 10.0
    reliability_sigma_max: float = 25.0
    reliability_max_games: int = 10  # Games for full reliability (CFB: 10 of 12-13 regular season)

    # EV threshold override for Phase 1
    ev_min_phase1: float = 0.05  # Higher threshold for weeks 1-3


# Guardrail reason codes
GUARDRAIL_OK = "OK"
GUARDRAIL_LOW_TRAIN_GAMES = "LOW_TRAIN_GAMES"
GUARDRAIL_BASELINE_OUT_OF_RANGE = "BASELINE_OUT_OF_RANGE"
GUARDRAIL_DIAGNOSTIC_ONLY_FORCED = "DIAGNOSTIC_ONLY_FORCED"


@dataclass
class TotalsBetRecommendation:
    """Full recommendation for a single totals bet.

    Mu Composition (Single Source of Truth):
        mu_model = TotalsModel.predict_total(...).predicted_total
        weather_adj = external adjustment (negative = lower scoring)
        mu_raw = mu_model + weather_adj (before baseline blend)
        mu_used = mu_raw + baseline_shift (final for probability calculations)

    Attributes:
        event_id: Unique game identifier
        home_team: Home team name
        away_team: Away team name
        book: Sportsbook name
        line: Total line
        side: "OVER" or "UNDER"

        mu_model: Raw model prediction from TotalsModel.predict_total().predicted_total
        weather_adj: Weather adjustment applied (negative = lower scoring expected)
        mu_raw: mu_model + weather_adj (before baseline blend)
        baseline_shift: Shift from baseline blending
        mu_used: Final mu for probability calculations (mu_raw + baseline_shift)
        adjusted_total: Legacy field (equals mu_used for compatibility)
        model_total: Legacy alias for mu_model (backwards compatibility)
        home_expected: Expected home team points (from model)
        away_expected: Expected away team points (from model)
        baseline: Model baseline (per-team)

        edge_pts: Edge in points (positive = bet side favored)
        p_win: Probability of winning
        p_loss: Probability of losing
        p_push: Probability of push

        odds_american: American odds for this side
        odds_decimal: Decimal odds for this side
        implied_prob: Implied probability from odds

        ev: Expected value as fraction of stake
        edge_prob: p_win - implied_prob
        kelly_f: Optimal Kelly fraction (before cap)
        stake: Recommended stake (after cap and guardrails)

        sigma_total: Standard deviation used for calculation
        ev_min: EV threshold used for qualification
        guardrail_reason: Reason code if guardrail triggered
    """
    # Identifiers
    event_id: str
    home_team: str
    away_team: str
    book: str
    line: float
    side: str  # "OVER" or "UNDER"

    # Mu composition (single source of truth)
    mu_model: float  # Raw from TotalsModel.predict_total().predicted_total
    weather_adj: float  # External weather adjustment (negative = lower scoring)
    mu_raw: float  # mu_model + weather_adj (before baseline blend)
    baseline_shift: float  # Shift from baseline blending
    mu_used: float  # Final mu for probability (mu_raw + baseline_shift)
    adjusted_total: float  # Legacy (equals mu_used)
    model_total: float  # Legacy alias for mu_model (backwards compatibility)
    home_expected: float
    away_expected: float
    baseline: float

    # Edge
    edge_pts: float

    # Probabilities
    p_win: float
    p_loss: float
    p_push: float

    # Pricing
    odds_american: float
    odds_decimal: float
    implied_prob: float

    # EV + Staking
    ev: float
    edge_prob: float
    kelly_f: float
    stake: float

    # Diagnostics
    sigma_total: float
    ev_min: float
    guardrail_reason: str = GUARDRAIL_OK

    # Phase 2: Calibration tracking
    sigma_used: Optional[float] = None  # Actual sigma used (may differ from sigma_total due to calibration)
    rel_game: float = 1.0  # Game reliability (0-1)
    sigma_mode: str = "fixed"  # Calibration mode used

    # Optional extras
    year: Optional[int] = None
    week: Optional[int] = None

    @property
    def weather_adjustment(self) -> float:
        """Legacy alias for weather_adj."""
        return self.weather_adj

    def __repr__(self) -> str:
        mode = f" [{self.guardrail_reason}]" if self.guardrail_reason != GUARDRAIL_OK else ""
        return (
            f"TotalsBetRecommendation({self.away_team}@{self.home_team}, "
            f"{self.side} {self.line} @ {self.book}, "
            f"edge={self.edge_pts:+.1f}, EV={self.ev:+.3f}, stake=${self.stake:.0f}{mode})"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "book": self.book,
            "line": self.line,
            "side": self.side,
            # Mu composition (single source of truth)
            "mu_model": self.mu_model,
            "weather_adj": self.weather_adj,
            "mu_raw": self.mu_raw,
            "baseline_shift": self.baseline_shift,
            "mu_used": self.mu_used,
            # Legacy fields (backwards compatibility)
            "model_total": self.model_total,
            "adjusted_total": self.adjusted_total,
            "weather_adjustment": self.weather_adj,  # Legacy alias
            # Model breakdown
            "home_expected": self.home_expected,
            "away_expected": self.away_expected,
            "baseline": self.baseline,
            # Edge
            "edge_pts": self.edge_pts,
            # Probabilities
            "p_win": self.p_win,
            "p_loss": self.p_loss,
            "p_push": self.p_push,
            # Pricing
            "odds_american": self.odds_american,
            "odds_decimal": self.odds_decimal,
            "implied_prob": self.implied_prob,
            # EV + Staking
            "ev": self.ev,
            "edge_prob": self.edge_prob,
            "kelly_f": self.kelly_f,
            "stake": self.stake,
            # Calibration
            "sigma_total": self.sigma_total,
            "sigma_used": self.sigma_used,
            "rel_game": self.rel_game,
            "sigma_mode": self.sigma_mode,
            "ev_min": self.ev_min,
            "guardrail_reason": self.guardrail_reason,
            # Metadata
            "year": self.year,
            "week": self.week,
        }


# =============================================================================
# Math Utilities
# =============================================================================

def normal_cdf(x: float, use_scipy: bool = True) -> float:
    """Standard normal CDF: P(Z <= x).

    Args:
        x: Value to evaluate
        use_scipy: Whether to use scipy if available

    Returns:
        Probability P(Z <= x)
    """
    if use_scipy and SCIPY_AVAILABLE:
        return float(scipy_norm.cdf(x))
    # Fallback using error function
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def american_to_decimal(odds: int | float) -> float:
    """Convert American odds to decimal odds.

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Decimal odds (e.g., 1.909, 2.5)

    Raises:
        ValueError: If odds is zero

    Examples:
        >>> american_to_decimal(-110)
        1.909...
        >>> american_to_decimal(+150)
        2.5
    """
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / abs(odds)


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability.

    Args:
        decimal_odds: Decimal odds (e.g., 1.909)

    Returns:
        Implied probability (e.g., 0.5238 for -110)
    """
    if decimal_odds <= 0:
        raise ValueError(f"Decimal odds must be positive, got {decimal_odds}")
    return 1.0 / decimal_odds


def estimate_sigma_from_backtest(preds_df: pd.DataFrame) -> float:
    """Estimate sigma_total from backtest residuals.

    Args:
        preds_df: DataFrame with 'error' column (predicted - actual)

    Returns:
        Standard deviation of errors
    """
    if 'error' not in preds_df.columns:
        raise ValueError("preds_df must have 'error' column (predicted - actual)")
    return float(preds_df['error'].std())


# =============================================================================
# Phase 1: Baseline Prior Blending
# =============================================================================

def compute_baseline_blend_weight(
    n_train_games: int,
    n0: float,
    mode: str = "rational",
) -> float:
    """Compute blend weight for baseline prior vs fitted baseline.

    The weight decays from 1.0 (full prior) to 0.0 (full fitted) as
    training sample size increases.

    Args:
        n_train_games: Number of training games
        n0: Decay parameter (half-life for rational, scale for exp)
        mode: "rational" (n0/(n0+n)) or "exp" (exp(-n/n0))

    Returns:
        Weight w in [0, 1] where:
        - w=1 means use prior only
        - w=0 means use fitted only
        - w=0.5 at n_train_games=n0 (for rational mode)
    """
    if n_train_games <= 0:
        return 1.0

    if mode == "rational":
        return n0 / (n0 + n_train_games)
    elif mode == "exp":
        return math.exp(-n_train_games / n0)
    else:
        raise ValueError(f"Unknown blend mode: {mode}")


def compute_baseline_shift(
    baseline_fit: float,
    baseline_prior: float,
    n_train_games: int,
    config: TotalsEVConfig,
) -> float:
    """Compute total shift to apply to mu_raw based on baseline blending.

    Args:
        baseline_fit: Fitted baseline per-team from model
        baseline_prior: Prior baseline per-team expectation
        n_train_games: Number of training games
        config: EV configuration with blend parameters

    Returns:
        Total shift to add to mu_raw (can be positive or negative).
        Multiply by 2 because baseline is per-team and total is two teams.
    """
    if not config.enable_baseline_blend:
        return 0.0

    w = compute_baseline_blend_weight(
        n_train_games,
        config.baseline_blend_n0,
        config.baseline_blend_mode,
    )

    # Blend the baselines
    baseline_used = w * baseline_prior + (1 - w) * baseline_fit

    # Compute shift: difference between used and fitted, times 2 for total
    shift = 2 * (baseline_used - baseline_fit)

    logger.debug(
        f"Baseline blend: n={n_train_games}, w={w:.3f}, "
        f"fit={baseline_fit:.2f}, prior={baseline_prior:.2f}, "
        f"used={baseline_used:.2f}, shift={shift:+.2f}"
    )

    return shift


def check_guardrails(
    n_train_games: int,
    baseline_fit: float,
    config: TotalsEVConfig,
) -> tuple[bool, str]:
    """Check Phase 1 guardrails and return diagnostic status.

    Args:
        n_train_games: Number of training games
        baseline_fit: Fitted baseline per-team from model
        config: EV configuration with guardrail parameters

    Returns:
        Tuple of (is_diagnostic_mode, reason_code):
        - is_diagnostic_mode: True if stakes should be forced to 0
        - reason_code: One of GUARDRAIL_* constants
    """
    # Check forced diagnostic mode first
    if config.diagnostic_only_mode:
        return (True, GUARDRAIL_DIAGNOSTIC_ONLY_FORCED)

    if not config.auto_diagnostic_if_guardrail_hit:
        return (False, GUARDRAIL_OK)

    # Check training games threshold
    if n_train_games < config.min_train_games_for_staking:
        return (True, GUARDRAIL_LOW_TRAIN_GAMES)

    # Check baseline sanity bounds
    if baseline_fit < config.baseline_sanity_min or baseline_fit > config.baseline_sanity_max:
        return (True, GUARDRAIL_BASELINE_OUT_OF_RANGE)

    return (False, GUARDRAIL_OK)


# =============================================================================
# Phase 2: Sigma Calibration
# =============================================================================

def compute_game_reliability(
    home_games_played: int,
    away_games_played: int,
    max_games: int = 10,
) -> float:
    """Compute reliability score for a game based on team games played.

    Args:
        home_games_played: Number of games home team has COMPLETED before the
            game being predicted. Week 1 games should have 0 games played,
            giving reliability 0.0.
        away_games_played: Number of games away team has COMPLETED before the
            game being predicted.
        max_games: Number of games for full reliability. Default 10 is tuned
            for CFB's 12-13 game regular season, allowing reliability scaling
            to affect predictions through week 11.

    Returns:
        Reliability score in [0, 1]
    """
    if max_games <= 1:
        max_games = 10

    rel_home = min(1.0, max(0.0, (home_games_played - 1) / (max_games - 1)))
    rel_away = min(1.0, max(0.0, (away_games_played - 1) / (max_games - 1)))

    # Use minimum (most conservative)
    return min(rel_home, rel_away)


def get_sigma_for_week_bucket(
    week: int,
    sigma_base: float,
    multipliers: dict,
) -> float:
    """Get sigma for a specific week using bucket multipliers.

    Args:
        week: Week number (0-15+ for CFB)
        sigma_base: Base sigma
        multipliers: Dict mapping bucket name to multiplier.
            Keys must match totals_calibration.py: "0-2", "3-5", "6-9", "10-14", "15+"

    Returns:
        sigma_base * multiplier for the appropriate bucket
    """
    if week <= 2:
        bucket = "0-2"  # Includes CFB week 0
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


def compute_calibrated_sigma(
    config: TotalsEVConfig,
    week: int,
    home_games_played: int = 8,
    away_games_played: int = 8,
) -> tuple[float, float, str]:
    """Compute sigma based on calibration mode.

    Args:
        config: EV configuration with calibration settings
        week: Game week
        home_games_played: Home team's games played
        away_games_played: Away team's games played

    Returns:
        Tuple of (sigma_used, rel_game, sigma_mode)
    """
    sigma_base = config.sigma_total
    sigma_mode = config.sigma_mode

    if sigma_mode == "fixed":
        return (sigma_base, 1.0, "fixed")

    elif sigma_mode == "week_bucket":
        sigma = get_sigma_for_week_bucket(week, sigma_base, config.week_bucket_multipliers)
        return (sigma, 1.0, "week_bucket")

    elif sigma_mode == "reliability_scaled":
        rel_game = compute_game_reliability(
            home_games_played, away_games_played, config.reliability_max_games
        )

        # sigma_used = sigma_base * (1 + k * (1 - rel_game))
        sigma = sigma_base * (1 + config.reliability_k * (1 - rel_game))

        # Apply bounds
        sigma = max(config.reliability_sigma_min, min(config.reliability_sigma_max, sigma))

        return (sigma, rel_game, "reliability_scaled")

    else:
        logger.warning(f"Unknown sigma_mode: {sigma_mode}, using fixed")
        return (sigma_base, 1.0, "fixed")


def get_effective_ev_min(config: TotalsEVConfig, week: int) -> float:
    """Get effective EV minimum based on week (Phase 1 has higher threshold).

    Args:
        config: EV configuration
        week: Game week

    Returns:
        Effective ev_min threshold
    """
    if week <= 3:
        return max(config.ev_min, config.ev_min_phase1)
    return config.ev_min


# =============================================================================
# Probability Model (Push-Aware Normal CDF)
# =============================================================================

def calculate_totals_probabilities(
    model_total: float,
    line: float,
    sigma: float,
    side: str,
    use_scipy: bool = True,
) -> tuple[float, float, float]:
    """Calculate win/loss/push probabilities for a totals bet.

    For whole-number line L and model prediction mu:
        p_under_strict = Phi((L - 0.5 - mu) / sigma)   # T < L - 0.5
        p_over_strict  = 1 - Phi((L + 0.5 - mu) / sigma)  # T > L + 0.5
        p_push = Phi((L + 0.5 - mu) / sigma) - Phi((L - 0.5 - mu) / sigma)

    For half-point lines (e.g., 50.5):
        p_push = 0 (no push possible)
        p_under = Phi((L - mu) / sigma)
        p_over = 1 - Phi((L - mu) / sigma)

    Args:
        model_total: Model's predicted total
        line: Market line (e.g., 50, 50.5)
        sigma: Standard deviation of prediction errors
        side: "OVER" or "UNDER"
        use_scipy: Whether to use scipy for CDF

    Returns:
        Tuple of (p_win, p_loss, p_push) for the specified side
    """
    # Detect half-point line (no push possible)
    is_half_point = (line % 1.0) != 0.0

    if is_half_point:
        # No push possible
        z = (line - model_total) / sigma
        p_under = normal_cdf(z, use_scipy)
        p_over = 1.0 - p_under
        p_push = 0.0
    else:
        # Whole number line - push is possible
        z_low = (line - 0.5 - model_total) / sigma
        z_high = (line + 0.5 - model_total) / sigma

        p_under = normal_cdf(z_low, use_scipy)  # T < L - 0.5
        p_over = 1.0 - normal_cdf(z_high, use_scipy)  # T > L + 0.5
        p_push = normal_cdf(z_high, use_scipy) - normal_cdf(z_low, use_scipy)

    # Sanity check: probabilities should sum to 1
    total_prob = p_over + p_under + p_push
    if not (0.99 < total_prob < 1.01):
        logger.warning(f"Probabilities sum to {total_prob:.4f}, expected ~1.0")

    # Return based on side
    if side.upper() == "OVER":
        return (p_over, p_under, p_push)
    elif side.upper() == "UNDER":
        return (p_under, p_over, p_push)
    else:
        raise ValueError(f"side must be 'OVER' or 'UNDER', got {side}")


# =============================================================================
# EV Calculation
# =============================================================================

def calculate_ev_totals(
    p_win: float,
    p_loss: float,
    p_push: float,
    odds_decimal: float,
) -> float:
    """Calculate expected value for a totals bet.

    Formula:
        profit_multiple b = decimal_odds - 1
        EV = p_win * b - p_loss * 1.0 + p_push * 0
           = p_win * b - p_loss

    Args:
        p_win: Probability of winning
        p_loss: Probability of losing
        p_push: Probability of push (stake returned, 0 profit)
        odds_decimal: Decimal odds (e.g., 1.909 for -110)

    Returns:
        EV as fraction of stake
    """
    b = odds_decimal - 1.0  # Profit if win (per unit stake)
    ev = p_win * b - p_loss
    return ev


# =============================================================================
# Kelly Staking (Three-Outcome Formula)
# =============================================================================

def calculate_kelly_stake(
    p_win: float,
    p_loss: float,
    p_push: float,
    odds_decimal: float,
    config: TotalsEVConfig,
) -> tuple[float, float]:
    """Calculate Kelly stake for a three-outcome bet.

    Three-outcome Kelly derivation (maximizes E[log(wealth)]):
        E[log(W)] = p_win * log(1 + b*f) + p_loss * log(1 - f) + p_push * log(1)

        Taking derivative and setting to zero:
        f_star = (p_win * b - p_loss) / (b * (p_win + p_loss))

    Where b = decimal_odds - 1 (profit per unit).

    When p_push = 0 and p_loss = 1 - p_win, this reduces to the standard
    two-outcome Kelly formula: f_star = (p*b - q) / b.

    Args:
        p_win: Probability of winning
        p_loss: Probability of losing
        p_push: Probability of push (stake returned)
        odds_decimal: Decimal odds
        config: Configuration with Kelly parameters

    Returns:
        Tuple of (kelly_fraction, stake_amount)
    """
    b = odds_decimal - 1.0

    # Three-outcome Kelly formula
    numerator = p_win * b - p_loss
    denominator = b * (p_win + p_loss)

    if denominator <= 0 or numerator <= 0:
        # No edge or degenerate case
        return (0.0, 0.0)

    f_star = numerator / denominator

    # Apply fractional Kelly
    f = config.kelly_fraction * f_star
    f = max(0.0, f)  # No negative stakes

    # Cap at max_bet_fraction
    f = min(f, config.max_bet_fraction)

    # Calculate stake
    stake = config.bankroll * f

    # Round down to nearest unit
    stake = math.floor(stake / config.round_to) * config.round_to

    # Enforce minimum bet
    if stake < config.min_bet:
        stake = 0.0

    return (f, stake)


# =============================================================================
# Single Market Evaluation
# =============================================================================

def evaluate_single_market(
    event: TotalsEvent,
    market: TotalMarket,
    mu_model: float,
    weather_adj: float,
    mu_raw: float,
    mu_used: float,
    baseline_shift: float,
    home_expected: float,
    away_expected: float,
    baseline: float,
    config: TotalsEVConfig,
    guardrail_reason: str = GUARDRAIL_OK,
    sigma_used: float = None,
    rel_game: float = 1.0,
    sigma_mode: str = "fixed",
) -> list[TotalsBetRecommendation]:
    """Evaluate both sides of a single market.

    Args:
        event: Event data
        market: Market data (book, line, odds)
        mu_model: Raw model prediction from TotalsModel.predict_total().predicted_total
        weather_adj: External weather adjustment (negative = lower scoring)
        mu_raw: mu_model + weather_adj (before baseline blend)
        mu_used: Final mu for probability calculations (mu_raw + baseline_shift)
        baseline_shift: Shift applied from baseline blending
        home_expected: Expected home points (from model)
        away_expected: Expected away points (from model)
        baseline: Model baseline (per-team)
        config: EV configuration
        guardrail_reason: Reason code if guardrail triggered
        sigma_used: Calibrated sigma (default: config.sigma_total)
        rel_game: Game reliability score (0-1)
        sigma_mode: Sigma calibration mode used

    Returns:
        List of recommendations (0-2 items depending on config)
    """
    # Use calibrated sigma or fall back to config
    if sigma_used is None:
        sigma_used = config.sigma_total

    recommendations = []
    is_diagnostic = guardrail_reason != GUARDRAIL_OK

    # Get effective EV minimum (Phase 1 may have higher threshold)
    week = event.week or 10  # Default to mid-season if not specified
    effective_ev_min = get_effective_ev_min(config, week)

    for side, odds_american in [("OVER", market.odds_over), ("UNDER", market.odds_under)]:
        # Convert odds
        try:
            odds_decimal = american_to_decimal(odds_american)
            implied_prob = decimal_to_implied_prob(odds_decimal)
        except ValueError as e:
            logger.warning(f"Invalid odds for {event.event_id} {side}: {e}")
            continue

        # Calculate probabilities using mu_used and calibrated sigma
        p_win, p_loss, p_push = calculate_totals_probabilities(
            model_total=mu_used,
            line=market.line,
            sigma=sigma_used,
            side=side,
            use_scipy=config.allow_scipy and SCIPY_AVAILABLE,
        )

        # Calculate edge in points (using mu_used)
        if side == "OVER":
            edge_pts = mu_used - market.line  # Positive = model higher than line
        else:
            edge_pts = market.line - mu_used  # Positive = model lower than line

        # Calculate EV
        ev = calculate_ev_totals(p_win, p_loss, p_push, odds_decimal)

        # Calculate Kelly stake
        kelly_f, stake = calculate_kelly_stake(
            p_win, p_loss, p_push, odds_decimal, config
        )

        # Force stake=0 if in diagnostic mode
        if is_diagnostic:
            stake = 0.0

        # Edge probability
        edge_prob = p_win - implied_prob

        rec = TotalsBetRecommendation(
            event_id=event.event_id,
            home_team=event.home_team,
            away_team=event.away_team,
            book=market.book,
            line=market.line,
            side=side,
            # Mu composition (single source of truth)
            mu_model=mu_model,
            weather_adj=weather_adj,
            mu_raw=mu_raw,
            baseline_shift=baseline_shift,
            mu_used=mu_used,
            # Legacy fields
            adjusted_total=mu_used,
            model_total=mu_model,  # Legacy alias
            # Model breakdown
            home_expected=home_expected,
            away_expected=away_expected,
            baseline=baseline,
            # Edge
            edge_pts=edge_pts,
            # Probabilities
            p_win=p_win,
            p_loss=p_loss,
            p_push=p_push,
            # Pricing
            odds_american=odds_american,
            odds_decimal=odds_decimal,
            implied_prob=implied_prob,
            # EV + Staking
            ev=ev,
            edge_prob=edge_prob,
            kelly_f=kelly_f,
            stake=stake,
            # Calibration
            sigma_total=config.sigma_total,
            ev_min=effective_ev_min,
            guardrail_reason=guardrail_reason,
            sigma_used=sigma_used,
            rel_game=rel_game,
            sigma_mode=sigma_mode,
            # Metadata
            year=event.year,
            week=event.week,
        )
        recommendations.append(rec)

    # NOTE: one_bet_per_market filtering is now done in evaluate_totals_markets
    # after the List A / List B split. This ensures List B candidates are not
    # prematurely dropped when one side has higher EV but fails the EV cut.

    return recommendations


# =============================================================================
# Main Evaluation Interface
# =============================================================================

# Type alias for mu override function
# Signature: fn(mu_raw: float, line: float, event: TotalsEvent, market: TotalMarket) -> float
MuOverrideFn = Optional[Callable[[float, float, "TotalsEvent", "TotalMarket"], float]]


def evaluate_totals_markets(
    model,  # TotalsModel - avoid circular import with duck typing
    events: list[dict] | list[TotalsEvent],
    config: TotalsEVConfig,
    mu_override_fn: MuOverrideFn = None,
    n_train_games: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all totals markets for a slate of events.

    Args:
        model: TotalsModel instance (must have predict_total method)
        events: List of events with market data (dicts or TotalsEvent objects)
        config: EV configuration
        mu_override_fn: Optional function to override mu per-market.
            Signature: fn(mu_raw: float, line: float, event: TotalsEvent, market: TotalMarket) -> float
            If provided, the returned value is used as the model mean for probability calculations.
            The original mu is preserved in model_total field for reference.
            NOTE: If baseline blending is enabled, mu_override_fn takes precedence.
        n_train_games: Number of training games used (for guardrails/blend).
            If None, attempts to infer from model (not always reliable).

    Returns:
        Tuple of (primary_df, edge5_df):
        - primary_df: Primary Edge Execution Engine - EV-qualified bets, sorted by EV desc
        - edge5_df: 5+ Edge - Edge >= 5 but fails EV cut, sorted by edge_pts desc
    """
    all_recommendations: list[TotalsBetRecommendation] = []

    # Get model baseline for guardrails and blend
    baseline_fit = getattr(model, 'baseline', 26.0)

    # Infer n_train_games if not provided
    if n_train_games is None:
        # Try to get from model (TotalsModel tracks this internally)
        n_train_games = getattr(model, '_last_train_game_count', 0)
        if n_train_games == 0:
            # Fallback: estimate from team_ratings
            n_teams = len(getattr(model, 'team_ratings', {}))
            n_train_games = n_teams * 2  # Rough estimate

    # Check guardrails
    is_diagnostic, guardrail_reason = check_guardrails(n_train_games, baseline_fit, config)
    if is_diagnostic:
        logger.warning(
            f"Phase 1 guardrail triggered: {guardrail_reason} "
            f"(n_train={n_train_games}, baseline={baseline_fit:.2f}). "
            f"Stakes will be forced to 0."
        )

    # Compute baseline shift (for all markets, since it's model-level)
    baseline_shift = compute_baseline_shift(
        baseline_fit=baseline_fit,
        baseline_prior=config.baseline_prior_per_team,
        n_train_games=n_train_games,
        config=config,
    )

    for event_data in events:
        # Convert dict to TotalsEvent if needed
        if isinstance(event_data, dict):
            markets = [
                TotalMarket(**m) if isinstance(m, dict) else m
                for m in event_data.get("markets", [])
            ]
            event = TotalsEvent(
                event_id=event_data.get("event_id", ""),
                home_team=event_data.get("home_team", ""),
                away_team=event_data.get("away_team", ""),
                year=event_data.get("year"),
                week=event_data.get("week"),
                weather_adjustment=event_data.get("weather_adjustment", 0.0),
                markets=markets,
            )
        else:
            event = event_data

        # Get model prediction
        pred = model.predict_total(
            home_team=event.home_team,
            away_team=event.away_team,
            weather_adjustment=event.weather_adjustment,
            year=event.year,
        )

        if pred is None:
            logger.debug(f"No prediction for {event.event_id}: {event.away_team} @ {event.home_team}")
            continue

        # Get games played for reliability calculation
        home_rating = model.team_ratings.get(event.home_team) if hasattr(model, 'team_ratings') else None
        away_rating = model.team_ratings.get(event.away_team) if hasattr(model, 'team_ratings') else None
        home_games = home_rating.games_played if home_rating else 8
        away_games = away_rating.games_played if away_rating else 8

        # Compute calibrated sigma
        sigma_used, rel_game, sigma_mode = compute_calibrated_sigma(
            config=config,
            week=event.week or 10,
            home_games_played=home_games,
            away_games_played=away_games,
        )

        # Evaluate each market
        for market in event.markets:
            # Mu composition (single source of truth):
            # mu_model = TotalsModel.predict_total().predicted_total (raw from model)
            # weather_adj = external adjustment (negative = lower scoring)
            # mu_raw = mu_model + weather_adj (before baseline blend)
            # mu_used = mu_raw + baseline_shift (final for probability calculations)

            # CRITICAL: Always use pred.predicted_total, NEVER pred.adjusted_total.
            # pred.adjusted_total adds weather_adjustment again, causing double-counting.
            # The config.use_adjusted_total flag is DEPRECATED and ignored.
            mu_model = pred.predicted_total

            # Apply weather adjustment with guardrail cap
            raw_weather_adj = event.weather_adjustment if config.use_weather_adjustment else 0.0
            if raw_weather_adj != 0.0:
                # Guardrail: cap absolute weather adjustment
                capped_adj = max(-config.weather_max_adjustment,
                                 min(config.weather_max_adjustment, raw_weather_adj))
                if abs(raw_weather_adj) > config.weather_max_adjustment:
                    logger.warning(
                        f"Weather adjustment capped: {raw_weather_adj:+.1f} -> {capped_adj:+.1f} "
                        f"for {event.event_id}"
                    )
                weather_adj = capped_adj
            else:
                weather_adj = 0.0

            # mu_raw = mu_model + weather_adj (before baseline blend)
            mu_raw = mu_model + weather_adj

            # Compute mu_used: mu_override_fn takes precedence, else apply baseline blend
            if mu_override_fn is not None:
                mu_used = mu_override_fn(mu_raw, market.line, event, market)
                actual_shift = mu_used - mu_raw
            else:
                mu_used = mu_raw + baseline_shift
                actual_shift = baseline_shift

            recs = evaluate_single_market(
                event=event,
                market=market,
                mu_model=mu_model,
                weather_adj=weather_adj,
                mu_raw=mu_raw,
                mu_used=mu_used,
                baseline_shift=actual_shift,
                home_expected=pred.home_expected,
                away_expected=pred.away_expected,
                baseline=baseline_fit,
                config=config,
                guardrail_reason=guardrail_reason,
                sigma_used=sigma_used,
                rel_game=rel_game,
                sigma_mode=sigma_mode,
            )
            all_recommendations.extend(recs)

    if not all_recommendations:
        empty_df = pd.DataFrame()
        return (empty_df, empty_df)

    # Split into Primary Edge Execution Engine (List A) and 5+ Edge (List B) FIRST,
    # BEFORE applying one_bet_per_event or one_bet_per_market filters.
    # This ensures List B candidates are not prematurely dropped.
    primary: list[TotalsBetRecommendation] = []
    edge5: list[TotalsBetRecommendation] = []

    for rec in all_recommendations:
        # Primary Edge Execution Engine: EV >= effective ev_min AND (stake > 0 OR diagnostic mode)
        # Note: rec.ev_min is already the effective ev_min for this game's week
        # Diagnostic mode bets (stake=0 due to guardrails) stay in List A with guardrail_reason explaining why
        if rec.ev >= rec.ev_min and (rec.stake > 0 or rec.guardrail_reason != GUARDRAIL_OK):
            primary.append(rec)
        # 5+ Edge: edge_pts >= edge_pts_min AND fails EV cut
        # Note: edge_pts is directional (positive = bet side favored), so we check >= not abs()
        elif rec.edge_pts >= config.edge_pts_min:
            # Check what qualifies as "fails EV cut"
            if config.listB_include_positive_ev_below_cut:
                # Include if EV < ev_min OR stake == 0
                if rec.ev < rec.ev_min or rec.stake == 0:
                    edge5.append(rec)
            else:
                # Only include if EV <= 0 OR stake == 0
                if rec.ev <= 0 or rec.stake == 0:
                    edge5.append(rec)

    # Helper for deterministic tie-breaking: (EV desc, stake desc, book, line, side)
    def _comparison_key(rec: TotalsBetRecommendation) -> tuple:
        """Deterministic comparison key for filtering ties.

        Higher EV wins, then higher stake, then deterministic by book/line/side.
        """
        return (-rec.ev, -rec.stake, rec.book, rec.line, rec.side)

    # Apply one_bet_per_market filter to List A only (keep best side per event+book+line)
    if config.one_bet_per_market:
        by_market: dict[tuple[str, str, float], TotalsBetRecommendation] = {}
        for rec in primary:
            key = (rec.event_id, rec.book, rec.line)
            if key not in by_market or _comparison_key(rec) < _comparison_key(by_market[key]):
                by_market[key] = rec
        primary = list(by_market.values())

    # Apply one_bet_per_event filter to List A only (keep best EV per event)
    if config.one_bet_per_event:
        by_event: dict[str, TotalsBetRecommendation] = {}
        for rec in primary:
            if rec.event_id not in by_event or _comparison_key(rec) < _comparison_key(by_event[rec.event_id]):
                by_event[rec.event_id] = rec
        primary = list(by_event.values())

    # Sort results
    if config.sort_key == "ev":
        primary.sort(key=lambda r: -r.ev)
        edge5.sort(key=lambda r: -abs(r.edge_pts))
    else:  # edge_pts
        primary.sort(key=lambda r: -abs(r.edge_pts))
        edge5.sort(key=lambda r: -abs(r.edge_pts))

    # Convert to DataFrames
    primary_df = pd.DataFrame([r.to_dict() for r in primary])
    edge5_df = pd.DataFrame([r.to_dict() for r in edge5])

    return (primary_df, edge5_df)


def recommendations_to_dataframe(
    recommendations: list[TotalsBetRecommendation],
) -> pd.DataFrame:
    """Convert list of recommendations to DataFrame.

    Args:
        recommendations: List of TotalsBetRecommendation objects

    Returns:
        DataFrame with all recommendation fields
    """
    if not recommendations:
        return pd.DataFrame()
    return pd.DataFrame([r.to_dict() for r in recommendations])


def summarize_totals_ev(
    primary_df: pd.DataFrame,
    edge5_df: pd.DataFrame,
    config: TotalsEVConfig,
) -> dict:
    """Summarize totals EV evaluation results.

    Args:
        primary_df: Primary Edge Execution Engine (EV-qualified) DataFrame
        edge5_df: 5+ Edge (edge-only) DataFrame
        config: Configuration used

    Returns:
        Summary statistics dictionary (with native Python types for JSON)
    """
    summary = {
        "config": {
            "sigma_total": config.sigma_total,
            "ev_min": config.ev_min,
            "edge_pts_min": config.edge_pts_min,
            "kelly_fraction": config.kelly_fraction,
            "bankroll": config.bankroll,
        },
        "primary": {
            "count": len(primary_df),
            "total_stake": float(primary_df["stake"].sum()) if len(primary_df) > 0 else 0.0,
            "avg_ev": float(primary_df["ev"].mean()) if len(primary_df) > 0 else 0.0,
            "avg_edge_pts": float(primary_df["edge_pts"].abs().mean()) if len(primary_df) > 0 else 0.0,
        },
        "edge5": {
            "count": len(edge5_df),
            "avg_edge_pts": float(edge5_df["edge_pts"].abs().mean()) if len(edge5_df) > 0 else 0.0,
        },
    }

    # Side breakdown for Primary
    if len(primary_df) > 0:
        summary["primary"]["over_count"] = int(len(primary_df[primary_df["side"] == "OVER"]))
        summary["primary"]["under_count"] = int(len(primary_df[primary_df["side"] == "UNDER"]))

    return summary


# =============================================================================
# Example / Sanity Test
# =============================================================================

def _run_sanity_test() -> None:
    """Run sanity tests on the EV engine."""
    print("=" * 60)
    print("Totals EV Engine Sanity Tests")
    print("=" * 60)

    # Test 1: Normal CDF
    print("\n1. Normal CDF tests:")
    assert abs(normal_cdf(0) - 0.5) < 0.001, "CDF(0) should be 0.5"
    assert abs(normal_cdf(1.96) - 0.975) < 0.001, "CDF(1.96) should be ~0.975"
    print("   PASS: Normal CDF working correctly")

    # Test 2: Odds conversion
    print("\n2. Odds conversion tests:")
    assert abs(american_to_decimal(-110) - 1.909) < 0.01, "-110 -> ~1.909"
    assert abs(american_to_decimal(+150) - 2.5) < 0.01, "+150 -> 2.5"
    assert abs(decimal_to_implied_prob(1.909) - 0.524) < 0.01, "1.909 -> ~52.4%"
    print("   PASS: Odds conversion working correctly")

    # Test 3: Probabilities for symmetric case
    print("\n3. Symmetric probability test (mu == line):")
    p_win, p_loss, p_push = calculate_totals_probabilities(
        model_total=50.0, line=50, sigma=13.0, side="OVER"
    )
    print(f"   mu=50, line=50: p_over={p_win:.3f}, p_under={p_loss:.3f}, p_push={p_push:.3f}")
    assert abs(p_win - p_loss) < 0.01, "Over and Under should be symmetric"
    assert p_push > 0.01, "Push probability should be positive for whole number line"
    assert abs(p_win + p_loss + p_push - 1.0) < 0.001, "Probabilities should sum to 1"
    print("   PASS: Symmetric case correct")

    # Test 4: Half-point line (no push)
    print("\n4. Half-point line test (no push):")
    p_win, p_loss, p_push = calculate_totals_probabilities(
        model_total=50.0, line=50.5, sigma=13.0, side="OVER"
    )
    print(f"   mu=50, line=50.5: p_over={p_win:.3f}, p_under={p_loss:.3f}, p_push={p_push:.3f}")
    assert p_push == 0.0, "No push for half-point line"
    print("   PASS: Half-point line has no push")

    # Test 5: EV calculation
    print("\n5. EV calculation test:")
    ev = calculate_ev_totals(p_win=0.55, p_loss=0.45, p_push=0.0, odds_decimal=1.909)
    print(f"   p_win=0.55, odds=-110: EV = {ev:+.4f}")
    expected_ev = 0.55 * 0.909 - 0.45  # ~0.05
    assert abs(ev - expected_ev) < 0.001, f"EV should be ~{expected_ev:.4f}"
    print("   PASS: EV calculation correct")

    # Test 6: Kelly staking with numerical verification
    print("\n6. Kelly staking test:")
    config = TotalsEVConfig(kelly_fraction=0.25, bankroll=1000.0, round_to=1.0)
    p_win, p_loss, p_push = 0.55, 0.45, 0.0
    b = 0.909  # decimal 1.909 - 1
    kelly_f, stake = calculate_kelly_stake(
        p_win=p_win, p_loss=p_loss, p_push=p_push, odds_decimal=1.909, config=config
    )
    print(f"   p_win=0.55, odds=-110, quarter Kelly: f={kelly_f:.4f}, stake=${stake:.0f}")

    # Verify against correct Kelly formula: f* = (p*b - q) / (b * (p + q))
    # For two outcomes (p_push=0): denominator = b * (0.55 + 0.45) = b * 1.0 = b
    # Full Kelly: (0.55 * 0.909 - 0.45) / 0.909 = 0.04995 / 0.909 ≈ 0.05495
    # Quarter Kelly: 0.25 * 0.05495 ≈ 0.01374
    expected_full_kelly = (p_win * b - p_loss) / (b * (p_win + p_loss))
    expected_quarter_kelly = 0.25 * expected_full_kelly
    print(f"   Expected full Kelly: {expected_full_kelly:.5f}, quarter: {expected_quarter_kelly:.5f}")
    assert abs(kelly_f - expected_quarter_kelly) < 0.001, \
        f"Kelly fraction should be ~{expected_quarter_kelly:.5f}, got {kelly_f:.5f}"

    assert kelly_f > 0, "Kelly fraction should be positive with edge"
    assert stake > 0, "Stake should be positive"
    assert stake <= config.max_bet_fraction * config.bankroll, "Stake should be capped"
    print("   PASS: Kelly staking correct (verified numerically)")

    # Test 7: Edge case - model agrees with line
    print("\n7. Edge case - model == line at -110/-110:")
    config = TotalsEVConfig(sigma_total=13.0)
    p_win, p_loss, p_push = calculate_totals_probabilities(
        model_total=50.0, line=50, sigma=13.0, side="OVER"
    )
    ev = calculate_ev_totals(p_win, p_loss, p_push, odds_decimal=1.909)
    print(f"   EV for Over when mu=line: {ev:+.4f}")
    # Both sides should have slight negative EV due to vig
    assert ev < 0, "Both sides should be -EV when mu == line with vig"
    print("   PASS: No edge when model == line")

    # Test 8: Directional edge
    print("\n8. Directional edge test:")
    p_win_over, p_loss_over, p_push = calculate_totals_probabilities(
        model_total=55.0, line=50, sigma=13.0, side="OVER"
    )
    ev_over = calculate_ev_totals(p_win_over, p_loss_over, p_push, odds_decimal=1.909)

    p_win_under, p_loss_under, _ = calculate_totals_probabilities(
        model_total=55.0, line=50, sigma=13.0, side="UNDER"
    )
    ev_under = calculate_ev_totals(p_win_under, p_loss_under, p_push, odds_decimal=1.909)

    print(f"   mu=55, line=50: EV_over={ev_over:+.4f}, EV_under={ev_under:+.4f}")
    assert ev_over > ev_under, "Over should have higher EV when mu > line"
    assert ev_over > 0, "Over should be +EV when mu >> line"
    print("   PASS: Directional edge correct")

    print("\n" + "=" * 60)
    print("All sanity tests PASSED")
    print("=" * 60)


def _run_example():
    """Run example with mock model."""
    print("\n" + "=" * 60)
    print("Totals EV Engine Example")
    print("=" * 60)

    # Create a mock model
    class MockTotalsModel:
        def predict_total(self, home_team, away_team, weather_adjustment=0.0, year=None):
            # Simple mock: always predict 52 points
            from dataclasses import dataclass

            @dataclass
            class MockPrediction:
                home_team: str
                away_team: str
                predicted_total: float
                home_expected: float
                away_expected: float
                baseline: float
                weather_adjustment: float

                @property
                def adjusted_total(self):
                    return self.predicted_total + self.weather_adjustment

            return MockPrediction(
                home_team=home_team,
                away_team=away_team,
                predicted_total=52.0,
                home_expected=27.0,
                away_expected=25.0,
                baseline=26.0,
                weather_adjustment=weather_adjustment,
            )

    model = MockTotalsModel()
    config = TotalsEVConfig(
        sigma_total=13.0,
        ev_min=0.02,
        edge_pts_min=5.0,
        kelly_fraction=0.25,
        bankroll=1000.0,
    )

    # Create test events
    events = [
        TotalsEvent(
            event_id="game_1",
            home_team="Georgia",
            away_team="Alabama",
            year=2025,
            week=10,
            markets=[
                TotalMarket(book="DraftKings", line=48, odds_over=-110, odds_under=-110),
                TotalMarket(book="FanDuel", line=47, odds_over=-105, odds_under=-115),
            ],
        ),
        TotalsEvent(
            event_id="game_2",
            home_team="Ohio State",
            away_team="Michigan",
            year=2025,
            week=10,
            markets=[
                TotalMarket(book="DraftKings", line=45, odds_over=-110, odds_under=-110),
            ],
        ),
    ]

    # Evaluate
    primary_df, edge5_df = evaluate_totals_markets(model, events, config)

    print(f"\nPrimary Edge Execution Engine: {len(primary_df)} bets")
    if len(primary_df) > 0:
        print(primary_df[["home_team", "away_team", "book", "line", "side", "edge_pts", "ev", "stake"]].to_string())

    print(f"\n5+ Edge: {len(edge5_df)} bets")
    if len(edge5_df) > 0:
        print(edge5_df[["home_team", "away_team", "book", "line", "side", "edge_pts", "ev"]].to_string())

    # Summary
    summary = summarize_totals_ev(primary_df, edge5_df, config)
    print(f"\nSummary: {summary}")


# =============================================================================
# Production Integration Helpers (2026+)
# =============================================================================

def apply_weather_to_events(
    events: list[TotalsEvent],
    weather_df: pd.DataFrame,
    config: TotalsEVConfig,
) -> list[TotalsEvent]:
    """Apply weather adjustments to events from external weather data.

    Args:
        events: List of TotalsEvent objects (weather_adjustment may be 0)
        weather_df: DataFrame with columns:
            - game_id (or event_id): Unique identifier matching events
            - weather_adj: Weather adjustment (negative = lower scoring)
        config: EV configuration with guardrails

    Returns:
        List of TotalsEvent objects with weather_adjustment populated
    """
    if weather_df is None or len(weather_df) == 0:
        return events

    # Build lookup: game_id/event_id -> weather_adj
    id_col = 'game_id' if 'game_id' in weather_df.columns else 'event_id'
    weather_lookup = dict(zip(
        weather_df[id_col].astype(str),
        weather_df['weather_adj'].fillna(0.0)
    ))

    updated_events = []
    for event in events:
        adj = weather_lookup.get(str(event.event_id), 0.0)

        if not config.use_weather_adjustment:
            adj = 0.0

        # Apply guardrail cap
        if abs(adj) > config.weather_max_adjustment:
            logger.warning(
                f"Weather adjustment capped for {event.event_id}: "
                f"{adj:+.1f} -> {max(-config.weather_max_adjustment, min(config.weather_max_adjustment, adj)):+.1f}"
            )
            adj = max(-config.weather_max_adjustment, min(config.weather_max_adjustment, adj))

        # Create new event with weather applied
        updated = TotalsEvent(
            event_id=event.event_id,
            home_team=event.home_team,
            away_team=event.away_team,
            year=event.year,
            week=event.week,
            weather_adjustment=adj,
            markets=event.markets,
        )
        updated_events.append(updated)

    return updated_events


def create_production_config(
    use_weather: bool = True,
    calibration_path: str = None,
    fallback_sigma: float = 13.0,
    fallback_multiplier: float = 1.05,
) -> TotalsEVConfig:
    """Create production-ready EV configuration.

    Args:
        use_weather: Whether weather adjustments will be applied
        calibration_path: Path to calibration JSON (optional, uses defaults if None)
        fallback_sigma: Sigma to use if no calibration file
        fallback_multiplier: Sigma inflation if betting with weather but only model_only calibration

    Returns:
        TotalsEVConfig configured for production
    """
    config = TotalsEVConfig(
        use_weather_adjustment=use_weather,
        weather_max_adjustment=10.0,
        weather_sigma_fallback_multiplier=fallback_multiplier,
    )

    if calibration_path:
        try:
            from .totals_calibration import load_calibration, select_calibration_for_runtime
            from pathlib import Path

            calib_dir = Path(calibration_path).parent
            calib, reason = select_calibration_for_runtime(
                use_weather=use_weather,
                base_dir=calib_dir,
                fallback_multiplier=fallback_multiplier,
            )

            # Copy calibrated values to config
            config.sigma_total = calib.sigma_base
            config.sigma_mode = calib.sigma_mode
            config.week_bucket_multipliers = calib.week_bucket_multipliers
            config.ev_min = calib.ev_min
            config.ev_min_phase1 = calib.ev_min_phase1
            config.kelly_fraction = calib.kelly_fraction
            config.max_bet_fraction = calib.max_bet_fraction

            logger.info(f"Loaded calibration: {reason}, sigma={config.sigma_total:.2f}")

        except Exception as e:
            logger.warning(f"Could not load calibration: {e}. Using defaults.")
            config.sigma_total = fallback_sigma
    else:
        config.sigma_total = fallback_sigma

    return config


if __name__ == "__main__":
    _run_sanity_test()
    _run_example()
