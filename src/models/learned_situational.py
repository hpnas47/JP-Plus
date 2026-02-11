"""Learned Situational Adjustment (LSA) Model.

Replaces fixed situational constants (bye_week_advantage=1.5, letdown_penalty=-2.0, etc.)
with coefficients learned via walk-forward ridge regression on prediction residuals.

For each week W, trains on games from weeks < W, learning optimal weights for 16
situational features. Multi-year pooling provides ~3,000+ training samples by week 4.

Walk-forward safe: Only trains on games before prediction week.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge

from src.adjustments.situational import SituationalFactors

logger = logging.getLogger(__name__)


# Feature names in order (must match to_array() ordering)
FEATURE_NAMES = [
    "rest_differential",
    "short_week_home",
    "short_week_away",
    "bye_week_home",
    "bye_week_away",
    "letdown_home",
    "letdown_away",
    "lookahead_home",
    "lookahead_away",
    "sandwich_home",
    "sandwich_away",
    "consecutive_road_away",
    "rivalry_underdog_home",
    "rivalry_underdog_away",
    "game_shape_opener_home",
    "game_shape_opener_away",
]

# Expected signs for each feature (positive = favors home team)
EXPECTED_SIGNS = {
    "rest_differential": "+",
    "short_week_home": "-",
    "short_week_away": "+",
    "bye_week_home": "+",
    "bye_week_away": "-",
    "letdown_home": "-",
    "letdown_away": "+",
    "lookahead_home": "-",
    "lookahead_away": "+",
    "sandwich_home": "-",
    "sandwich_away": "+",
    "consecutive_road_away": "+",
    "rivalry_underdog_home": "+",
    "rivalry_underdog_away": "-",
    "game_shape_opener_home": "-",
    "game_shape_opener_away": "+",
}


@dataclass
class SituationalFeatures:
    """Feature vector for situational adjustment learning.

    All features are from home team perspective (positive = favors home).
    """
    # Rest differential (continuous): home_rest_days - away_rest_days
    rest_differential: float = 0.0

    # Short week (binary): rest_days <= 5
    short_week_home: float = 0.0
    short_week_away: float = 0.0

    # Bye week (binary): rest_days >= 14
    bye_week_home: float = 0.0
    bye_week_away: float = 0.0

    # Letdown (binary): coming off big win vs unranked
    letdown_home: float = 0.0
    letdown_away: float = 0.0

    # Lookahead (binary): big game next week
    lookahead_home: float = 0.0
    lookahead_away: float = 0.0

    # Sandwich (binary): both letdown AND lookahead
    sandwich_home: float = 0.0
    sandwich_away: float = 0.0

    # Consecutive road (binary): 2nd straight road game for away team
    consecutive_road_away: float = 0.0

    # Rivalry underdog boost (binary)
    rivalry_underdog_home: float = 0.0
    rivalry_underdog_away: float = 0.0

    # Game shape opener (binary): team playing first game of season
    game_shape_opener_home: float = 0.0
    game_shape_opener_away: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ridge regression."""
        return np.array([
            self.rest_differential,
            self.short_week_home,
            self.short_week_away,
            self.bye_week_home,
            self.bye_week_away,
            self.letdown_home,
            self.letdown_away,
            self.lookahead_home,
            self.lookahead_away,
            self.sandwich_home,
            self.sandwich_away,
            self.consecutive_road_away,
            self.rivalry_underdog_home,
            self.rivalry_underdog_away,
            self.game_shape_opener_home,
            self.game_shape_opener_away,
        ])

    @classmethod
    def from_situational_factors(
        cls,
        home_factors: SituationalFactors,
        away_factors: SituationalFactors,
    ) -> "SituationalFeatures":
        """Create feature vector from home and away SituationalFactors.

        Converts the raw factor magnitudes into binary/continuous features
        suitable for ridge regression learning.
        """
        # Rest differential (continuous, from home perspective)
        rest_diff = home_factors.rest_days - away_factors.rest_days

        # Short week: rest_days <= 5
        short_week_home = 1.0 if home_factors.rest_days <= 5 else 0.0
        short_week_away = 1.0 if away_factors.rest_days <= 5 else 0.0

        # Bye week: rest_days >= 14
        bye_week_home = 1.0 if home_factors.rest_days >= 14 else 0.0
        bye_week_away = 1.0 if away_factors.rest_days >= 14 else 0.0

        # Letdown: penalty magnitude > 0 indicates letdown spot
        letdown_home = 1.0 if home_factors.letdown_penalty > 0 else 0.0
        letdown_away = 1.0 if away_factors.letdown_penalty > 0 else 0.0

        # Lookahead: penalty magnitude > 0
        lookahead_home = 1.0 if home_factors.lookahead_penalty > 0 else 0.0
        lookahead_away = 1.0 if away_factors.lookahead_penalty > 0 else 0.0

        # Sandwich: both letdown AND lookahead (penalty > 0 means triggered)
        sandwich_home = 1.0 if home_factors.sandwich_penalty > 0 else 0.0
        sandwich_away = 1.0 if away_factors.sandwich_penalty > 0 else 0.0

        # Consecutive road: away team penalty > 0
        # Note: home team can't have consecutive road penalty by definition
        consecutive_road_away = 1.0 if away_factors.consecutive_road_penalty > 0 else 0.0

        # Rivalry underdog boost: boost > 0 means team is underdog in rivalry
        rivalry_underdog_home = 1.0 if home_factors.rivalry_boost > 0 else 0.0
        rivalry_underdog_away = 1.0 if away_factors.rivalry_boost > 0 else 0.0

        # Game shape opener: is_season_opener flag
        game_shape_opener_home = 1.0 if home_factors.is_season_opener else 0.0
        game_shape_opener_away = 1.0 if away_factors.is_season_opener else 0.0

        return cls(
            rest_differential=rest_diff,
            short_week_home=short_week_home,
            short_week_away=short_week_away,
            bye_week_home=bye_week_home,
            bye_week_away=bye_week_away,
            letdown_home=letdown_home,
            letdown_away=letdown_away,
            lookahead_home=lookahead_home,
            lookahead_away=lookahead_away,
            sandwich_home=sandwich_home,
            sandwich_away=sandwich_away,
            consecutive_road_away=consecutive_road_away,
            rivalry_underdog_home=rivalry_underdog_home,
            rivalry_underdog_away=rivalry_underdog_away,
            game_shape_opener_home=game_shape_opener_home,
            game_shape_opener_away=game_shape_opener_away,
        )


@dataclass
class LearnedSituationalCoefficients:
    """Container for learned situational coefficients with metadata."""
    year: int
    max_week: int
    n_games: int
    alpha: float
    intercept: float
    coefficients: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "year": self.year,
            "max_week": self.max_week,
            "n_games": self.n_games,
            "alpha": self.alpha,
            "intercept": self.intercept,
            "coefficients": self.coefficients,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearnedSituationalCoefficients":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            year=data["year"],
            max_week=data["max_week"],
            n_games=data["n_games"],
            alpha=data["alpha"],
            intercept=data["intercept"],
            coefficients=data["coefficients"],
        )


class LearnedSituationalModel:
    """Learned Situational Adjustment model using ridge regression.

    Trains on residuals (actual_margin - base_margin_no_situ) to learn
    optimal weights for 16 situational features. Uses multi-year pooling
    for stable coefficient estimates.

    Key features:
    - Walk-forward safe: Only trains on completed games
    - EMA smoothing: Prevents coefficient instability week-to-week
    - Sign validation: Logs warnings when learned signs flip from expected
    - Coefficient persistence: Saves to JSON for analysis/debugging
    """

    def __init__(
        self,
        ridge_alpha: float = 300.0,
        min_games: int = 150,
        ema_beta: float = 0.3,
        clamp_max: Optional[float] = 4.0,
        persist_dir: Optional[Path] = None,
    ):
        """Initialize LSA model.

        Args:
            ridge_alpha: Ridge regularization strength (higher = more shrinkage). Default 300.0.
            min_games: Minimum training games required; falls back to fixed if fewer
            ema_beta: EMA smoothing factor (0.3 = 30% new, 70% prior)
            clamp_max: Maximum coefficient magnitude (e.g., 4.0 clamps to [-4, +4]). Default 4.0.
            persist_dir: Optional directory for coefficient JSON persistence
        """
        self.ridge_alpha = ridge_alpha
        self.min_games = min_games
        self.ema_beta = ema_beta
        self.clamp_max = clamp_max
        self.persist_dir = persist_dir

        # Training data: list of (features_array, residual) tuples
        self._training_data: list[tuple[np.ndarray, float]] = []

        # Current learned coefficients (dict[feature_name -> coefficient])
        self._coefficients: Optional[dict[str, float]] = None
        self._intercept: float = 0.0

        # Previous coefficients for EMA smoothing
        self._prev_coefficients: Optional[dict[str, float]] = None

        # Metadata for persistence
        self._year: Optional[int] = None
        self._max_week: int = 0
        self._is_trained: bool = False

    def reset(self, year: int) -> None:
        """Reset model state for a new season.

        Note: Training data from prior seasons should be passed via
        add_training_game() before reset() is called. This method only
        clears the current season's state, not the accumulated training data.

        Args:
            year: Current season year
        """
        self._year = year
        self._max_week = 0
        self._is_trained = False
        # Don't clear training data or coefficients - they persist across seasons

    def seed_with_prior_data(
        self,
        prior_training_data: list[tuple[np.ndarray, float]],
    ) -> None:
        """Seed model with training data from prior completed seasons.

        Called at start of each season with pooled data from all prior years.
        This provides ~3,000+ training samples even in week 1.

        Args:
            prior_training_data: List of (features_array, residual) tuples
        """
        self._training_data = list(prior_training_data)  # Copy to avoid mutation
        logger.debug(f"LSA seeded with {len(self._training_data)} prior games")

    def add_training_game(
        self,
        features: SituationalFeatures,
        residual: float,
    ) -> None:
        """Add a completed game to training data.

        Args:
            features: Situational feature vector for the game
            residual: actual_margin - base_margin_no_situ
        """
        self._training_data.append((features.to_array(), residual))

    def train(self, max_week: int) -> Optional[LearnedSituationalCoefficients]:
        """Train ridge regression on accumulated training data.

        Should be called after each week's games are added. If fewer than
        min_games are available, returns None (caller should use fixed constants).

        Args:
            max_week: Maximum week included in training data

        Returns:
            LearnedSituationalCoefficients if trained, None if insufficient data
        """
        self._max_week = max_week

        n_games = len(self._training_data)
        if n_games < self.min_games:
            logger.debug(
                f"LSA: {n_games} games < min {self.min_games}, skipping training"
            )
            self._is_trained = False
            return None

        # Build feature matrix and target vector
        X = np.array([fd[0] for fd in self._training_data])
        y = np.array([fd[1] for fd in self._training_data])

        # Fit ridge regression
        model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        model.fit(X, y)

        # Extract coefficients
        raw_coefs = {
            name: float(coef)
            for name, coef in zip(FEATURE_NAMES, model.coef_)
        }

        # Apply EMA smoothing if we have prior coefficients
        if self._prev_coefficients is not None:
            smoothed_coefs = {}
            for name in FEATURE_NAMES:
                prev = self._prev_coefficients.get(name, 0.0)
                curr = raw_coefs[name]
                smoothed_coefs[name] = (1 - self.ema_beta) * prev + self.ema_beta * curr
            self._coefficients = smoothed_coefs
        else:
            self._coefficients = raw_coefs

        # Apply coefficient clamping if configured
        if self.clamp_max is not None:
            clamped_count = 0
            for name in FEATURE_NAMES:
                raw_val = self._coefficients[name]
                clamped_val = max(min(raw_val, self.clamp_max), -self.clamp_max)
                if clamped_val != raw_val:
                    clamped_count += 1
                    logger.debug(
                        f"LSA clamp: {name} {raw_val:.3f} -> {clamped_val:.3f}"
                    )
                self._coefficients[name] = clamped_val
            if clamped_count > 0:
                logger.info(f"LSA: clamped {clamped_count} coefficients to +/-{self.clamp_max}")

        self._intercept = float(model.intercept_)
        self._prev_coefficients = dict(self._coefficients)  # Save for next round
        self._is_trained = True

        # Check for sign flips
        self._check_sign_flips()

        # Log summary
        nonzero_count = sum(1 for v in self._coefficients.values() if abs(v) > 0.1)
        logger.debug(
            f"LSA trained: {n_games} games, week {max_week}, "
            f"{nonzero_count} features > 0.1 magnitude"
        )

        # Build result object
        result = LearnedSituationalCoefficients(
            year=self._year or 0,
            max_week=max_week,
            n_games=n_games,
            alpha=self.ridge_alpha,
            intercept=self._intercept,
            coefficients=dict(self._coefficients),
        )

        # Persist to disk if configured
        if self.persist_dir is not None:
            self._persist_coefficients(result)

        return result

    def predict(self, features: SituationalFeatures) -> float:
        """Predict situational adjustment for a matchup.

        Returns the learned situational adjustment from home team perspective.
        Positive = favors home team.

        Args:
            features: Situational feature vector for the matchup

        Returns:
            Predicted situational adjustment in points

        Raises:
            ValueError: If model has not been trained
        """
        if not self._is_trained or self._coefficients is None:
            raise ValueError("LSA model not trained - use is_trained() to check")

        X = features.to_array()
        adjustment = self._intercept
        for i, name in enumerate(FEATURE_NAMES):
            adjustment += self._coefficients[name] * X[i]

        return adjustment

    def is_trained(self) -> bool:
        """Check if model is ready for prediction."""
        return self._is_trained and self._coefficients is not None

    def get_coefficients(self) -> Optional[dict[str, float]]:
        """Get current learned coefficients."""
        return dict(self._coefficients) if self._coefficients else None

    def get_training_data(self) -> list[tuple[np.ndarray, float]]:
        """Get accumulated training data (for passing to next season)."""
        return list(self._training_data)

    def _check_sign_flips(self) -> None:
        """Check for coefficient sign flips vs expected and log warnings."""
        if self._coefficients is None:
            return

        flips = []
        for name, expected in EXPECTED_SIGNS.items():
            coef = self._coefficients.get(name, 0.0)
            if abs(coef) < 0.1:
                continue  # Ignore tiny coefficients

            actual = "+" if coef > 0 else "-"
            if expected != actual:
                flips.append((name, coef, expected))

        if flips:
            for name, coef, expected in flips:
                logger.warning(
                    f"LSA SIGN FLIP: {name} = {coef:.3f} (expected {expected})"
                )

    def _persist_coefficients(self, result: LearnedSituationalCoefficients) -> None:
        """Save coefficients to JSON file."""
        if self.persist_dir is None:
            return

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        filename = f"lsa_{result.year}_week{result.max_week:02d}.json"
        filepath = self.persist_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.debug(f"LSA coefficients saved to {filepath}")


def compute_situational_residual(
    actual_margin: float,
    predicted_spread: float,
    fixed_situational: float,
) -> float:
    """Compute the residual for LSA training.

    The residual is computed by removing the fixed situational adjustment
    from the prediction and comparing to actual margin.

    Args:
        actual_margin: Actual home margin (home_points - away_points)
        predicted_spread: Full predicted spread including situational
        fixed_situational: Fixed situational adjustment that was applied

    Returns:
        Residual = actual_margin - base_margin_no_situ
    """
    # base_margin_no_situ = predicted_spread - fixed_situational
    base_margin_no_situ = predicted_spread - fixed_situational
    return actual_margin - base_margin_no_situ
