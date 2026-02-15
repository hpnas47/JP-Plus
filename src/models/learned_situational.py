"""Learned Situational Adjustment (LSA) Model.

Replaces fixed situational constants (bye_week_advantage=1.5, letdown_penalty=-2.0, etc.)
with coefficients learned via walk-forward ridge regression on prediction residuals.

For each week W, trains on games from weeks < W, learning optimal weights for 16
situational features. Multi-year pooling provides ~3,000+ training samples by week 4.

Walk-forward safe: Only trains on games before prediction week.
"""

import dataclasses
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
        """Convert to numpy array for ridge regression.

        Order is determined by FEATURE_NAMES to ensure consistency.
        """
        return np.array([getattr(self, name) for name in FEATURE_NAMES])

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
        # Guard against None values (defensive - coalesce to neutral defaults)
        home_rest = home_factors.rest_days if home_factors.rest_days is not None else 7
        away_rest = away_factors.rest_days if away_factors.rest_days is not None else 7

        # Rest differential (continuous, from home perspective)
        rest_diff = home_rest - away_rest

        # Short week: rest_days <= 5
        short_week_home = 1.0 if home_rest <= 5 else 0.0
        short_week_away = 1.0 if away_rest <= 5 else 0.0

        # Bye week: rest_days >= 14
        bye_week_home = 1.0 if home_rest >= 14 else 0.0
        bye_week_away = 1.0 if away_rest >= 14 else 0.0

        # Letdown: penalty magnitude > 0 indicates letdown spot
        letdown_home = 1.0 if (home_factors.letdown_penalty or 0.0) > 0 else 0.0
        letdown_away = 1.0 if (away_factors.letdown_penalty or 0.0) > 0 else 0.0

        # Lookahead: penalty magnitude > 0
        lookahead_home = 1.0 if (home_factors.lookahead_penalty or 0.0) > 0 else 0.0
        lookahead_away = 1.0 if (away_factors.lookahead_penalty or 0.0) > 0 else 0.0

        # Sandwich: both letdown AND lookahead (penalty > 0 means triggered)
        sandwich_home = 1.0 if (home_factors.sandwich_penalty or 0.0) > 0 else 0.0
        sandwich_away = 1.0 if (away_factors.sandwich_penalty or 0.0) > 0 else 0.0

        # Consecutive road: away team penalty > 0
        # Note: home team can't have consecutive road penalty by definition
        consecutive_road_away = 1.0 if (away_factors.consecutive_road_penalty or 0.0) > 0 else 0.0

        # Rivalry underdog boost: boost > 0 means team is underdog in rivalry
        rivalry_underdog_home = 1.0 if (home_factors.rivalry_boost or 0.0) > 0 else 0.0
        rivalry_underdog_away = 1.0 if (away_factors.rivalry_boost or 0.0) > 0 else 0.0

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


# Validate FEATURE_NAMES matches SituationalFeatures fields at import time.
# Uses explicit if/raise instead of assert to prevent optimization removal (python -O).
_FEATURE_FIELDS = {f.name for f in dataclasses.fields(SituationalFeatures)}
for _name in FEATURE_NAMES:
    if _name not in _FEATURE_FIELDS:
        raise ValueError(f"FEATURE_NAMES entry '{_name}' not in SituationalFeatures")
if len(FEATURE_NAMES) != len(_FEATURE_FIELDS):
    raise ValueError(
        f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries but SituationalFeatures "
        f"has {len(_FEATURE_FIELDS)} fields"
    )
del _FEATURE_FIELDS, _name  # Clean up module namespace


@dataclass
class LearnedSituationalCoefficients:
    """Container for learned situational coefficients with metadata."""
    year: int
    max_week: int
    n_games: int
    alpha: float
    intercept: float
    coefficients: dict[str, float] = field(default_factory=dict)
    # Training configuration and statistics (for diagnostics)
    training_config: dict = field(default_factory=dict)
    training_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "year": self.year,
            "max_week": self.max_week,
            "n_games": self.n_games,
            "alpha": self.alpha,
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "training_config": self.training_config,
            "training_stats": self.training_stats,
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
            training_config=data.get("training_config", {}),
            training_stats=data.get("training_stats", {}),
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
        # Turnover adjustment parameters
        adjust_for_turnovers: bool = False,
        turnover_point_value: float = 4.0,
        # Vegas spread filter parameters
        max_abs_vegas_spread: Optional[float] = None,
        # Sample weighting parameters
        use_sample_weights: bool = False,
        weight_spread_threshold: Optional[float] = None,
        # Minimum sample safeguard
        min_training_games: int = 30,
    ):
        """Initialize LSA model.

        Args:
            ridge_alpha: Ridge regularization strength (higher = more shrinkage). Default 300.0.
            min_games: Minimum training games required; falls back to fixed if fewer
            ema_beta: EMA smoothing factor (0.3 = 30% new, 70% prior)
            clamp_max: Maximum coefficient magnitude (e.g., 4.0 clamps to [-4, +4]). Default 4.0.
            persist_dir: Optional directory for coefficient JSON persistence
            adjust_for_turnovers: If True, adjust residuals to remove turnover noise. Default False.
            turnover_point_value: Points per turnover for target adjustment. Default 4.0.
            max_abs_vegas_spread: Filter training games with |vegas_spread| > threshold. None disables.
            use_sample_weights: If True, use smooth Cauchy weighting instead of hard filter. Default False.
            weight_spread_threshold: Spread threshold for Cauchy decay (default 17.0 if weights enabled).
            min_training_games: Minimum games after filtering; relax filter if below. Default 30.
        """
        self.ridge_alpha = ridge_alpha
        self.min_games = min_games
        self.ema_beta = ema_beta
        self.clamp_max = clamp_max
        self.persist_dir = persist_dir

        # Turnover adjustment
        self.adjust_for_turnovers = adjust_for_turnovers
        self.turnover_point_value = turnover_point_value

        # Vegas spread filter
        self.max_abs_vegas_spread = max_abs_vegas_spread

        # Sample weighting
        self.use_sample_weights = use_sample_weights
        self.weight_spread_threshold = weight_spread_threshold if weight_spread_threshold is not None else 17.0
        self.min_training_games = min_training_games

        # Log if both weighting and filter are enabled (weighting takes precedence)
        if use_sample_weights and max_abs_vegas_spread is not None:
            logger.info(
                "LSA: Both weighted training and Vegas filter specified; using weighting only."
            )

        # Training data: parallel lists for efficient numpy stacking
        self._X_train: list[np.ndarray] = []  # Feature rows
        self._y_train: list[float] = []  # Residuals (post-turnover-adjustment)
        self._sample_weights: list[float] = []  # Sample weights for weighted Ridge
        self._vegas_spreads: list[Optional[float]] = []  # For filtering at train time
        self._weeks: list[Optional[int]] = []  # Week per sample (None = prior-year pooled)

        # Diagnostic counters (reset each season)
        self._n_games_total: int = 0
        self._n_games_turnover_adjusted: int = 0
        self._n_games_turnover_missing: int = 0
        self._n_games_filtered_vegas: int = 0

        # Current learned coefficients (dict[feature_name -> coefficient])
        self._coefficients: Optional[dict[str, float]] = None
        self._intercept: float = 0.0

        # Previous coefficients for EMA smoothing
        self._prev_coefficients: Optional[dict[str, float]] = None

        # Metadata for persistence
        self._year: Optional[int] = None
        self._max_week: int = 0
        self._is_trained: bool = False
        self._seeded: bool = False  # Guard against duplicate seed_with_prior_data calls

    def reset(self, year: int) -> None:
        """Reset model state for a new season.

        Clears ALL state including training data. After reset(), call
        seed_with_prior_data() to load prior seasons, then add_training_game()
        for current season games.

        Args:
            year: Current season year
        """
        self._year = year
        self._max_week = 0
        self._is_trained = False
        # Clear training data to prevent duplicate accumulation across seasons
        self._X_train.clear()
        self._y_train.clear()
        self._sample_weights.clear()
        self._vegas_spreads.clear()
        self._weeks.clear()
        # Reset diagnostic counters
        self._n_games_total = 0
        self._n_games_turnover_adjusted = 0
        self._n_games_turnover_missing = 0
        self._n_games_filtered_vegas = 0
        # Clear EMA state so first train() uses raw coefficients without smoothing.
        # This is correct because training data already includes prior seasons
        # via seed_with_prior_data() - no need for additional cross-season EMA bleed.
        self._prev_coefficients = None
        # Clear seeded flag for duplicate prevention
        self._seeded = False

    def seed_with_prior_data(
        self,
        prior_training_data: list[tuple],
    ) -> None:
        """Seed model with training data from prior completed seasons.

        Called ONCE at start of each season with pooled data from all prior years.
        This provides ~3,000+ training samples even in week 1.

        Calling multiple times without reset() in between will raise an error
        to prevent duplicate data accumulation.

        Args:
            prior_training_data: List of tuples. Supports formats:
                - Legacy: (features_array, residual)
                - Extended: (features_array, residual, weight, vegas_spread)
                - V2 extended: (features_array, residual, weight, vegas_spread, week)
                Prior-season data should use week=None (always included in training).

        Raises:
            RuntimeError: If called multiple times without reset()
        """
        if self._seeded:
            raise RuntimeError(
                "seed_with_prior_data() already called this season. "
                "Call reset(year) first to clear state before re-seeding."
            )

        # Prepend prior data, preserving any current-season games already added
        prior_X = []
        prior_y = []
        prior_weights = []
        prior_vegas = []
        prior_weeks = []

        for t in prior_training_data:
            prior_X.append(t[0])
            prior_y.append(t[1])
            # Handle extended format with weights and vegas spreads
            if len(t) >= 3:
                prior_weights.append(t[2])
            else:
                prior_weights.append(1.0)  # Default weight
            if len(t) >= 4:
                prior_vegas.append(t[3])
            else:
                prior_vegas.append(None)  # Unknown vegas spread
            if len(t) >= 5:
                prior_weeks.append(t[4])
            else:
                prior_weeks.append(None)  # Prior-year pooled data (always included)

        self._X_train = prior_X + self._X_train
        self._y_train = prior_y + self._y_train
        self._sample_weights = prior_weights + self._sample_weights
        self._vegas_spreads = prior_vegas + self._vegas_spreads
        self._weeks = prior_weeks + self._weeks
        self._seeded = True
        logger.debug(
            "LSA seeded with %d prior games (%d total)",
            len(prior_training_data),
            len(self._X_train),
        )

    def add_training_game(
        self,
        features: SituationalFeatures,
        residual: float,
        turnover_margin: Optional[float] = None,
        vegas_spread: Optional[float] = None,
        week: Optional[int] = None,
    ) -> None:
        """Add a completed game to training data.

        Args:
            features: Situational feature vector for the game
            residual: actual_margin - base_margin_no_situ (raw, before turnover adjustment)
            turnover_margin: Home takeaways - home giveaways (positive = home gained turnovers).
                            If None and adjust_for_turnovers=True, uses 0 and logs warning.
            vegas_spread: Vegas closing spread for the game (for filtering/weighting).
                          Convention: negative = home favored (Vegas standard).
            week: Game week (for walk-forward filtering in train()). If None, always included.
        """
        self._n_games_total += 1

        # Apply turnover adjustment to residual if enabled
        adjusted_residual = residual
        if self.adjust_for_turnovers:
            if turnover_margin is not None:
                # Remove turnover-driven noise from residual
                # Positive turnover_margin = home gained turnovers = home scored more than expected
                # We remove this from residual to isolate situational effects
                adjustment = turnover_margin * self.turnover_point_value
                adjusted_residual = residual - adjustment
                self._n_games_turnover_adjusted += 1
            else:
                # No turnover data - use unadjusted residual and log warning
                self._n_games_turnover_missing += 1
                if self._n_games_turnover_missing <= 5:  # Limit log spam
                    logger.warning(
                        "LSA: Missing turnover data for game; using unadjusted residual"
                    )

        # Compute sample weight based on Vegas spread
        weight = self._compute_sample_weight(vegas_spread)

        self._X_train.append(features.to_array())
        self._y_train.append(adjusted_residual)
        self._sample_weights.append(weight)
        self._vegas_spreads.append(vegas_spread)
        self._weeks.append(week)

    def _compute_sample_weight(self, vegas_spread: Optional[float]) -> float:
        """Compute sample weight using Cauchy-style decay for large spreads.

        Args:
            vegas_spread: Vegas spread (negative = home favored)

        Returns:
            Weight in (0, 1]. Returns 1.0 if weighting disabled or spread unknown.
        """
        if not self.use_sample_weights:
            return 1.0
        if vegas_spread is None:
            return 1.0
        if self.weight_spread_threshold is None or self.weight_spread_threshold <= 0:
            return 1.0

        # Use absolute value of spread (ignore sign convention)
        abs_spread = abs(vegas_spread)
        # Cauchy-style decay: 1 / (1 + (x/threshold)^2)
        return 1.0 / (1.0 + (abs_spread / self.weight_spread_threshold) ** 2)

    def train(self, max_week: int) -> Optional[LearnedSituationalCoefficients]:
        """Train ridge regression on accumulated training data.

        Should be called after each week's games are added. If fewer than
        min_games are available, returns None (caller should use fixed constants).

        Filtering/weighting logic:
        - If use_sample_weights=True: uses Cauchy-weighted Ridge (no hard filter)
        - Elif max_abs_vegas_spread set: filters out large-spread games (hard filter)
        - If filtering would drop below min_training_games: relaxes filter

        Args:
            max_week: Maximum week included in training data

        Returns:
            LearnedSituationalCoefficients if trained, None if insufficient data
        """
        self._max_week = max_week

        # Walk-forward safety: filter samples by week
        # Prior-year pooled data (week=None) is always included.
        # Current-season samples are only included if week <= max_week.
        n_excluded_future = 0
        walk_forward_mask = []
        for w in self._weeks:
            if w is None or w <= max_week:
                walk_forward_mask.append(True)
            else:
                walk_forward_mask.append(False)
                n_excluded_future += 1

        if n_excluded_future > 0:
            logger.debug(
                "LSA: excluded %d current-season games with week > %d",
                n_excluded_future, max_week,
            )

        X_list = [x for x, m in zip(self._X_train, walk_forward_mask) if m]
        y_list = [y for y, m in zip(self._y_train, walk_forward_mask) if m]
        weights_list = [w for w, m in zip(self._sample_weights, walk_forward_mask) if m]
        vegas_list = [v for v, m in zip(self._vegas_spreads, walk_forward_mask) if m]

        n_games_total = len(X_list)
        if n_games_total < self.min_games:
            logger.debug(
                "LSA: %d games < min %d, skipping training", n_games_total, self.min_games
            )
            self._is_trained = False
            return None

        n_filtered_vegas = 0
        use_filter = False

        # Apply Vegas spread filter if enabled (and NOT using weighted training)
        if self.max_abs_vegas_spread is not None and not self.use_sample_weights:
            # Build mask for games passing the filter
            pass_filter = []
            for vs in vegas_list:
                if vs is None:
                    # Unknown spread - do NOT filter (passes)
                    pass_filter.append(True)
                elif abs(vs) <= self.max_abs_vegas_spread:
                    pass_filter.append(True)
                else:
                    pass_filter.append(False)

            n_filtered = sum(1 for p in pass_filter if not p)
            n_after_filter = sum(pass_filter)

            # Check minimum training sample safeguard
            if n_after_filter < self.min_training_games:
                logger.warning(
                    "LSA: Vegas spread filter disabled for week %d: would reduce training "
                    "games from %d to %d (below minimum %d)",
                    max_week, n_games_total, n_after_filter, self.min_training_games
                )
                # Relax filter - use all games
                pass_filter = [True] * n_games_total
                n_filtered = 0
            else:
                use_filter = True
                n_filtered_vegas = n_filtered

            # Apply filter
            if use_filter:
                X_list = [x for x, p in zip(self._X_train, pass_filter) if p]
                y_list = [y for y, p in zip(self._y_train, pass_filter) if p]
                weights_list = [w for w, p in zip(self._sample_weights, pass_filter) if p]

        # Build feature matrix and target vector (efficient stack)
        X = np.vstack(X_list)
        y = np.array(y_list)
        n_games_used = len(y_list)

        # Determine sample weights for Ridge
        sample_weight = None
        if self.use_sample_weights:
            sample_weight = np.array(weights_list)

        # Fit ridge regression
        model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        model.fit(X, y, sample_weight=sample_weight)

        # Extract coefficients
        raw_coefs = {
            name: float(coef)
            for name, coef in zip(FEATURE_NAMES, model.coef_)
        }

        # 1. Clamp raw coefficients FIRST (before EMA)
        # This prevents the ratchet effect where clamped values anchor EMA at ceiling
        if self.clamp_max is not None:
            clamped_count = 0
            for name in FEATURE_NAMES:
                raw_val = raw_coefs[name]
                clamped_val = max(min(raw_val, self.clamp_max), -self.clamp_max)
                if clamped_val != raw_val:
                    clamped_count += 1
                    logger.debug(
                        "LSA clamp: %s %.3f -> %.3f", name, raw_val, clamped_val
                    )
                raw_coefs[name] = clamped_val
            if clamped_count > 0:
                logger.info(
                    "LSA: clamped %d coefficients to +/-%.1f", clamped_count, self.clamp_max
                )

        # 2. Apply EMA smoothing AFTER clamping
        # This allows coefficients to recover from ceiling when raw values drop
        if self._prev_coefficients is not None:
            for name in FEATURE_NAMES:
                prev = self._prev_coefficients.get(name, 0.0)
                curr = raw_coefs[name]
                raw_coefs[name] = (1 - self.ema_beta) * prev + self.ema_beta * curr

        self._coefficients = raw_coefs

        self._intercept = float(model.intercept_)
        self._prev_coefficients = dict(self._coefficients)  # Save for next round
        self._is_trained = True
        self._n_games_filtered_vegas = n_filtered_vegas

        # Check for sign flips
        self._check_sign_flips()

        # Log summary with diagnostics
        pct_filtered = (n_games_total - n_games_used) / n_games_total * 100 if n_games_total > 0 else 0
        if pct_filtered > 25:
            logger.warning(
                "LSA: High filter rate: %.1f%% of games filtered (%d -> %d)",
                pct_filtered, n_games_total, n_games_used
            )

        nonzero_count = sum(1 for v in self._coefficients.values() if abs(v) > 0.1)
        logger.debug(
            "LSA trained: %d/%d games (%.1f%% filtered), week %d, %d features > 0.1 magnitude",
            n_games_used, n_games_total, pct_filtered, max_week, nonzero_count,
        )

        # Validate year before persisting
        if self._year is None:
            logger.warning(
                "LSA train() called before reset() - year not set, skipping persistence"
            )
            result_year = 0
        else:
            result_year = self._year

        # Build training config and stats for persistence
        training_config = {
            "adjust_for_turnovers": self.adjust_for_turnovers,
            "turnover_point_value": self.turnover_point_value,
            "max_abs_vegas_spread": self.max_abs_vegas_spread,
            "use_sample_weights": self.use_sample_weights,
            "weight_spread_threshold": self.weight_spread_threshold if self.use_sample_weights else None,
        }
        training_stats = {
            "n_games_total": n_games_total,
            "n_games_used": n_games_used,
            "n_games_turnover_adjusted": self._n_games_turnover_adjusted,
            "n_games_turnover_missing": self._n_games_turnover_missing,
            "n_games_filtered_vegas": n_filtered_vegas,
            "pct_filtered": round(pct_filtered, 2),
        }

        # Build result object
        result = LearnedSituationalCoefficients(
            year=result_year,
            max_week=max_week,
            n_games=n_games_used,
            alpha=self.ridge_alpha,
            intercept=self._intercept,
            coefficients=dict(self._coefficients),
            training_config=training_config,
            training_stats=training_stats,
        )

        # Persist to disk if configured (only if year is valid)
        if self.persist_dir is not None and self._year is not None:
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

        x = features.to_array()
        coef_array = np.array([self._coefficients[name] for name in FEATURE_NAMES])
        return float(self._intercept + np.dot(coef_array, x))

    def is_trained(self) -> bool:
        """Check if model is ready for prediction."""
        return self._is_trained and self._coefficients is not None

    def get_coefficients(self) -> Optional[dict[str, float]]:
        """Get current learned coefficients."""
        return dict(self._coefficients) if self._coefficients else None

    def get_training_data(self) -> list[tuple]:
        """Get accumulated training data (for passing to next season).

        Returns V2 extended tuple format: (features_array, residual, weight, vegas_spread, week)
        Compatible with seed_with_prior_data() which handles legacy, extended, and V2 formats.
        """
        return list(zip(self._X_train, self._y_train, self._sample_weights, self._vegas_spreads, self._weeks))

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
    predicted_spread_includes_situational: bool = True,
) -> float:
    """Compute the residual for LSA training.

    The residual is computed by removing the fixed situational adjustment
    from the prediction and comparing to actual margin.

    Sign conventions (all from home perspective):
        actual_margin = home_points - away_points (positive = home win)
        predicted_spread = positive means model favors home
        fixed_situational = positive means situational favored home

    The residual captures what the base model (without situational) missed:
        residual = actual_margin - base_margin_no_situ

    If residual is positive, the actual margin exceeded the base prediction,
    suggesting situational factors helped the home team more than expected.

    Args:
        actual_margin: Actual home margin (home_points - away_points)
        predicted_spread: Full predicted spread (includes or excludes situational per flag)
        fixed_situational: Fixed situational adjustment that was applied
        predicted_spread_includes_situational: If True (default), predicted_spread includes
            the fixed situational adjustment and it will be subtracted to recover the base.
            If False, predicted_spread is already the base (no subtraction needed).

    Returns:
        Residual = actual_margin - base_margin_no_situ
    """
    if predicted_spread_includes_situational:
        base_margin_no_situ = predicted_spread - fixed_situational
    else:
        base_margin_no_situ = predicted_spread
    return actual_margin - base_margin_no_situ
