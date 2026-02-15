"""Win probability calibration and Monte Carlo season simulation.

Pure math module — no API dependencies. Provides:
- Poisson binomial PMF for independent game win probabilities
- Logistic win probability from spread
- Monte Carlo simulation with latent team shocks (correlated uncertainty)
- Walk-forward calibration of win probability and tau parameters
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator

logger = logging.getLogger(__name__)

# Default HFA in points (conservative, used when no calibration available)
DEFAULT_HFA = 3.0

# Default logistic calibration slope. Historical CFB spread std ≈ 14 pts,
# logistic scale = 1/slope ≈ 8.3, consistent with implied_sigma range [5, 20].
DEFAULT_CALIBRATION_SLOPE = 0.12

# Naive fallback regression factor (regress toward mean by 30%)
NAIVE_FALLBACK_REGRESSION = 0.30


@dataclass
class ScheduledGame:
    """A single game on a team's schedule."""
    opponent: str
    is_home: bool
    is_neutral: bool = False
    opponent_rating: float = 0.0


@dataclass
class GameProjection:
    """Projected outcome for a single game."""
    opponent: str
    is_home: bool
    is_neutral: bool
    spread: float  # positive = team favored
    win_prob: float


@dataclass
class WinProbCalibration:
    """Calibration parameters for logistic win probability model.

    Metadata tracks data provenance for auditing:
    - n_games_primary: Games from OOF predictions (honest signal)
    - n_games_fallback: Games supplemented via naive baseline (weaker signal)
    - calibration_source: 'primary', 'primary+fallback', or 'naive_default'
    - used_fallback_years: List of years where naive fallback was used
    """
    intercept: float = 0.0
    slope: float = DEFAULT_CALIBRATION_SLOPE
    n_games: int = 0
    implied_sigma: float = 0.0  # 1 / slope, interpretable as spread std
    n_games_primary: int = 0
    n_games_fallback: int = 0
    years: list[int] = field(default_factory=list)  # All years contributing data
    calibration_source: str = 'naive_default'
    used_fallback_years: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.slope > 0:
            self.implied_sigma = 1.0 / self.slope


@dataclass
class WinTotalDistribution:
    """Full win total distribution for a team-season."""
    team: str
    year: int
    predicted_rating: float
    expected_wins: float
    win_probs: np.ndarray  # P(wins = k) for k = 0..n_games
    game_projections: list[GameProjection] = field(default_factory=list)

    @property
    def n_games(self) -> int:
        return len(self.win_probs) - 1

    def prob_over(self, line: float) -> float:
        """P(wins > line). Half-win lines have no push; integer lines can push."""
        # For half-integer lines (8.5): over = P(wins >= 9)
        # For integer lines (8): over = P(wins >= 9), push = P(wins == 8)
        return float(np.sum(self.win_probs[int(line) + 1:]))

    def prob_under(self, line: float) -> float:
        """P(wins < line). Half-win lines have no push; integer lines can push."""
        if line != int(line):
            # Half-integer (8.5): under = P(wins <= 8)
            return float(np.sum(self.win_probs[:int(line) + 1]))
        else:
            # Integer (8): under = P(wins <= 7), push = P(wins == 8)
            return float(np.sum(self.win_probs[:int(line)]))

    def prob_push(self, line: float) -> float:
        """P(wins == line). Zero for half-integer lines."""
        if line != int(line):
            return 0.0
        k = int(line)
        if 0 <= k <= self.n_games:
            return float(self.win_probs[k])
        return 0.0

    def prob_exactly(self, n: int) -> float:
        """P(wins == n)."""
        if 0 <= n <= self.n_games:
            return float(self.win_probs[n])
        return 0.0


def compute_spread(
    team_rating: float,
    opponent_rating: float,
    is_home: bool,
    is_neutral: bool = False,
    hfa: float = DEFAULT_HFA,
) -> float:
    """Compute projected spread (positive = team favored).

    Args:
        team_rating: Team's predicted SP+ rating
        opponent_rating: Opponent's predicted SP+ rating
        is_home: Whether team is home
        is_neutral: Whether game is at neutral site (overrides is_home)
        hfa: Home field advantage in points

    Returns:
        Projected spread in points (positive = team favored)
    """
    spread = team_rating - opponent_rating
    if is_neutral:
        return spread
    if is_home:
        spread += hfa
    else:
        spread -= hfa
    return spread


def game_win_probability(
    spread: float,
    calibration: WinProbCalibration | None = None,
) -> float:
    """Convert spread to win probability via logistic function.

    P(win) = 1 / (1 + exp(-(intercept + slope * spread)))

    Args:
        spread: Projected spread (positive = team favored)
        calibration: Logistic calibration params. Uses defaults if None.

    Returns:
        Win probability in [0.01, 0.99]
    """
    if calibration is None:
        intercept = 0.0
        slope = DEFAULT_CALIBRATION_SLOPE
    else:
        intercept = calibration.intercept
        slope = calibration.slope

    logit = intercept + slope * spread
    # Clip logit to prevent overflow, then clamp prob to [0.01, 0.99]
    logit = np.clip(logit, -20.0, 20.0)
    p = float(1.0 / (1.0 + np.exp(-logit)))
    return float(np.clip(p, 0.01, 0.99))


def poisson_binomial_pmf(probs: list[float] | np.ndarray) -> np.ndarray:
    """Exact Poisson binomial PMF via dynamic programming.

    Computes P(X = k) for k = 0, 1, ..., n where X is the sum of n
    independent Bernoulli trials with potentially different probabilities.

    Note: This is the zero-tau (independent games) special case. For
    production EV calculations, use simulate_season_with_shocks() which
    adds correlated team-level uncertainty and produces wider distributions.
    Using this function directly will underestimate distribution width.

    Args:
        probs: Win probabilities for each game (length n)

    Returns:
        Array of length n+1 with P(X = k) for k = 0..n
    """
    probs = np.asarray(probs, dtype=np.float64)
    n = len(probs)
    if n == 0:
        return np.array([1.0])

    dp = np.zeros(n + 1)
    dp[0] = 1.0 - probs[0]
    dp[1] = probs[0]

    for i in range(1, n):
        p = probs[i]
        new_dp = np.zeros(n + 1)
        new_dp[0] = dp[0] * (1.0 - p)
        for k in range(1, i + 2):
            new_dp[k] = dp[k] * (1.0 - p) + dp[k - 1] * p
        dp = new_dp

    return dp


def simulate_season_with_shocks(
    game_spreads: list[float],
    tau: float,
    n_sims: int = 10000,
    rng: Generator | None = None,
    calibration: WinProbCalibration | None = None,
) -> np.ndarray:
    """Monte Carlo season simulation with latent team shocks.

    Each simulation draws a single team-level shock delta ~ N(0, tau^2),
    which shifts ALL game spreads for that team. This creates correlated
    uncertainty (a team is consistently better/worse than predicted).

    Note on calibration interaction: The logistic calibration slope is fitted
    on out-of-fold predicted spreads, which already embed prediction error.
    This means the fitted slope is slightly flatter than the true-spread slope.
    When tau-shifted spreads are evaluated through this logistic, there is mild
    double-counting of uncertainty. This is accepted as conservative: it widens
    the distribution slightly, making extreme win totals marginally more likely,
    which errs toward underconfidence rather than overconfidence in EV estimates.

    Args:
        game_spreads: Projected spreads for each game (positive = favored)
        tau: Std dev of latent team shock (from walk-forward residuals)
        n_sims: Number of Monte Carlo simulations
        rng: NumPy random generator (for reproducibility)
        calibration: Win probability calibration params

    Returns:
        Array of length n_games+1 with P(wins = k) for k = 0..n_games
    """
    if rng is None:
        rng = np.random.default_rng()

    spreads = np.asarray(game_spreads, dtype=np.float64)
    n_games = len(spreads)

    if n_games == 0:
        return np.array([1.0])

    deltas = rng.normal(0.0, tau, size=n_sims)
    shifted = spreads[np.newaxis, :] + deltas[:, np.newaxis]

    if calibration is None:
        intercept = 0.0
        slope = DEFAULT_CALIBRATION_SLOPE
    else:
        intercept = calibration.intercept
        slope = calibration.slope

    logits = intercept + slope * shifted
    logits = np.clip(logits, -20.0, 20.0)
    win_probs = 1.0 / (1.0 + np.exp(-logits))
    # Clamp to [0.01, 0.99] to prevent degenerate simulations
    win_probs = np.clip(win_probs, 0.01, 0.99)

    uniforms = rng.random(size=(n_sims, n_games))
    wins = (uniforms < win_probs).astype(np.int32)
    win_counts = wins.sum(axis=1)

    pmf = np.zeros(n_games + 1)
    for k in range(n_games + 1):
        pmf[k] = np.mean(win_counts == k)

    return pmf


def calibrate_win_probability(
    spreads: np.ndarray,
    outcomes: np.ndarray,
    min_games: int = 1500,
    fallback_spreads: np.ndarray | None = None,
    fallback_outcomes: np.ndarray | None = None,
    fallback_years: list[int] | None = None,
    primary_years: list[int] | None = None,
) -> WinProbCalibration:
    """Fit logistic regression: P(win) = logistic(intercept + slope * spread).

    Uses out-of-fold predicted spreads and actual outcomes to calibrate.
    If primary data < min_games, supplements with fallback data (naive
    baseline ratings regressed 30% toward mean).

    LogisticRegression config: penalty='l2', C=1.0, solver='lbfgs', max_iter=1000.

    Args:
        spreads: Primary predicted spreads (from OOF model predictions)
        outcomes: Binary outcomes (1 = win, 0 = loss)
        min_games: Minimum games required for reliable calibration
        fallback_spreads: Supplemental spreads from naive baseline
        fallback_outcomes: Supplemental outcomes for fallback spreads
        fallback_years: Years that contributed fallback data

    Returns:
        WinProbCalibration with fitted params and provenance metadata
    """
    spreads = np.asarray(spreads, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    n_primary = len(spreads)

    # Try to supplement with fallback if insufficient primary data
    if n_primary < min_games and fallback_spreads is not None:
        fb_spreads = np.asarray(fallback_spreads, dtype=np.float64)
        fb_outcomes = np.asarray(fallback_outcomes, dtype=np.float64)
        n_fallback = len(fb_spreads)

        combined_spreads = np.concatenate([spreads, fb_spreads])
        combined_outcomes = np.concatenate([outcomes, fb_outcomes])

        if len(combined_spreads) >= min_games:
            logger.info(
                f"Supplementing {n_primary} primary games with {n_fallback} "
                f"fallback games to reach {min_games} minimum"
            )
            all_years = sorted(set((primary_years or []) + (fallback_years or [])))
            return _fit_logistic(
                combined_spreads, combined_outcomes,
                n_games_primary=n_primary,
                n_games_fallback=n_fallback,
                calibration_source='primary+fallback',
                years=all_years,
                used_fallback_years=fallback_years or [],
            )
        else:
            logger.warning(
                f"Only {len(combined_spreads)} total games (primary={n_primary}, "
                f"fallback={n_fallback}), still under {min_games}. Using naive default."
            )
            return WinProbCalibration(
                intercept=0.0,
                slope=DEFAULT_CALIBRATION_SLOPE,
                n_games=len(combined_spreads),
                n_games_primary=n_primary,
                n_games_fallback=n_fallback,
                years=sorted(set((primary_years or []) + (fallback_years or []))),
                calibration_source='naive_default',
                used_fallback_years=fallback_years or [],
            )

    if n_primary < min_games:
        logger.warning(
            f"Only {n_primary} games for calibration (need {min_games}), "
            "no fallback available. Using naive logistic calibration."
        )
        return WinProbCalibration(
            intercept=0.0,
            slope=DEFAULT_CALIBRATION_SLOPE,
            n_games=n_primary,
            n_games_primary=n_primary,
            n_games_fallback=0,
            years=primary_years or [],
            calibration_source='naive_default',
        )

    return _fit_logistic(
        spreads, outcomes,
        n_games_primary=n_primary,
        n_games_fallback=0,
        calibration_source='primary',
        years=primary_years or [],
    )


def _fit_logistic(
    spreads: np.ndarray,
    outcomes: np.ndarray,
    n_games_primary: int,
    n_games_fallback: int,
    calibration_source: str,
    years: list[int] | None = None,
    used_fallback_years: list[int] | None = None,
) -> WinProbCalibration:
    """Internal: fit LogisticRegression with exact spec params."""
    from sklearn.linear_model import LogisticRegression

    X = spreads.reshape(-1, 1)
    y = outcomes

    lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
    lr.fit(X, y)

    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])

    cal = WinProbCalibration(
        intercept=intercept,
        slope=slope,
        n_games=len(spreads),
        n_games_primary=n_games_primary,
        n_games_fallback=n_games_fallback,
        years=years or [],
        calibration_source=calibration_source,
        used_fallback_years=used_fallback_years or [],
    )

    logger.info(
        f"Win prob calibration ({calibration_source}): "
        f"intercept={intercept:.4f}, slope={slope:.4f}, "
        f"implied_sigma={cal.implied_sigma:.2f}, "
        f"n_primary={n_games_primary}, n_fallback={n_games_fallback}"
    )

    if cal.implied_sigma < 5.0 or cal.implied_sigma > 20.0:
        logger.warning(
            f"Implied sigma {cal.implied_sigma:.2f} outside expected range [5.0, 20.0]!"
        )

    return cal


def calibrate_tau(residuals: np.ndarray) -> float:
    """Compute tau (team-level uncertainty) from walk-forward residuals.

    Tau is the standard deviation of (predicted_rating - actual_rating)
    across walk-forward folds.

    Args:
        residuals: Array of (predicted - actual) SP+ rating differences

    Returns:
        tau (std dev of residuals)
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    # ddof=1 (Bessel's correction): ~0.2-0.4% overestimate at N=130-260,
    # negligible at typical N=400-600. Accepted as mildly conservative.
    tau = float(np.std(residuals, ddof=1))
    logger.info(f"Calibrated tau = {tau:.3f} from {len(residuals)} residuals")
    return tau


def project_season(
    team: str,
    year: int,
    team_rating: float,
    schedule: list[ScheduledGame],
    tau: float,
    calibration: WinProbCalibration | None = None,
    hfa: float = DEFAULT_HFA,
    n_sims: int = 10000,
    rng: Generator | None = None,
) -> WinTotalDistribution:
    """Project a team's full season win total distribution.

    Args:
        team: Team name
        year: Season year
        team_rating: Predicted SP+ rating for the team
        schedule: List of scheduled games with opponent ratings
        tau: Team-level uncertainty (from calibrate_tau)
        calibration: Win probability calibration
        hfa: Home field advantage in points
        n_sims: Monte Carlo simulations
        rng: Random generator

    Returns:
        WinTotalDistribution with full PMF and game-level projections
    """
    game_spreads = []
    game_projections = []

    for game in schedule:
        spread = compute_spread(
            team_rating, game.opponent_rating,
            game.is_home, game.is_neutral, hfa
        )
        win_prob = game_win_probability(spread, calibration)

        game_spreads.append(spread)
        game_projections.append(GameProjection(
            opponent=game.opponent,
            is_home=game.is_home,
            is_neutral=game.is_neutral,
            spread=spread,
            win_prob=win_prob,
        ))

    win_pmf = simulate_season_with_shocks(
        game_spreads, tau, n_sims=n_sims, rng=rng, calibration=calibration
    )

    expected_wins = float(np.sum(np.arange(len(win_pmf)) * win_pmf))

    return WinTotalDistribution(
        team=team,
        year=year,
        predicted_rating=team_rating,
        expected_wins=expected_wins,
        win_probs=win_pmf,
        game_projections=game_projections,
    )
