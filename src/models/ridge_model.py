"""Ridge regression model for opponent-adjusted team ratings."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TeamRatings:
    """Container for team offensive and defensive ratings."""

    team: str
    offense: float
    defense: float
    overall: float = field(init=False)

    def __post_init__(self):
        # Overall = offense - defense (higher is better for both components)
        self.overall = self.offense - self.defense


class RidgeRatingsModel:
    """
    Ridge regression model for calculating opponent-adjusted team ratings.

    Creates a design matrix where:
    - Each game is a row
    - Columns represent team offenses and defenses
    - Home field advantage is captured in a separate column
    - Target variable is EPA per play (or point margin)

    The ridge penalty prevents overfitting to small sample sizes and handles
    the collinearity inherent in rating systems.
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        use_epa: bool = True,
        fixed_hfa: Optional[float] = None,
    ):
        """Initialize the model.

        Args:
            alpha: Ridge regularization strength. If None, uses settings default.
            use_epa: If True, uses EPA per play as target. If False, uses point margin.
            fixed_hfa: If provided, subtract this HFA from target before fitting
                       (don't learn HFA from data). Recommended: 2.5-3.0 for CFB.
        """
        if alpha is None:
            settings = get_settings()
            alpha = settings.ridge_alpha

        self.alpha = alpha
        self.use_epa = use_epa
        self.fixed_hfa = fixed_hfa
        self.model: Optional[Ridge] = None
        self.teams: list[str] = []
        self.team_to_idx: dict[str, int] = {}
        self.hfa_coefficient: float = 0.0
        self.ratings: dict[str, TeamRatings] = {}

    def _create_design_matrix(
        self,
        games_df: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create the design matrix for ridge regression.

        Uses a single rating per team (not separate offense/defense) since
        margin data alone cannot distinguish between the two.

        Args:
            games_df: DataFrame with game data
            weights: Optional sample weights

        Returns:
            Tuple of (X design matrix, y target, weights)
        """
        # Get unique teams
        all_teams = set(games_df["home_team"]) | set(games_df["away_team"])
        self.teams = sorted(list(all_teams))
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
        n_teams = len(self.teams)

        # Design matrix: [team_ratings] or [team_ratings, hfa]
        # For each game predicting margin = home_points - away_points:
        # - Home team: +1 (their rating contributes positively)
        # - Away team: -1 (their rating contributes negatively)
        # - HFA: +1 (only if not using fixed HFA)

        n_games = len(games_df)

        # If using fixed HFA, don't include HFA column
        if self.fixed_hfa is not None:
            X = np.zeros((n_games, n_teams))
        else:
            X = np.zeros((n_games, n_teams + 1))

        for i, (_, game) in enumerate(games_df.iterrows()):
            home_idx = self.team_to_idx[game["home_team"]]
            away_idx = self.team_to_idx[game["away_team"]]

            # Team rating columns
            X[i, home_idx] = 1  # Home team rating (adds to margin)
            X[i, away_idx] = -1  # Away team rating (subtracts from margin)

            # HFA column (only if not using fixed HFA)
            if self.fixed_hfa is None:
                X[i, n_teams] = 1

        # Target: home margin (home points - away points or EPA differential)
        if self.use_epa and "home_epa" in games_df.columns:
            y = games_df["home_epa"].values - games_df["away_epa"].values
        else:
            y = (games_df["home_points"].values - games_df["away_points"].values).astype(float)

        # If using fixed HFA, subtract it from target for non-neutral games
        if self.fixed_hfa is not None:
            if "neutral_site" in games_df.columns:
                neutral = games_df["neutral_site"].fillna(False).values
                y = y - np.where(neutral, 0, self.fixed_hfa)
            else:
                y = y - self.fixed_hfa

        # Weights
        if weights is None:
            if "recency_weight" in games_df.columns:
                weights = games_df["recency_weight"].values
            else:
                weights = np.ones(n_games)

        return X, y, weights

    def fit(
        self,
        games_df: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
    ) -> "RidgeRatingsModel":
        """Fit the model to game data.

        Args:
            games_df: DataFrame with columns:
                - home_team: Home team name
                - away_team: Away team name
                - home_points: Home team score
                - away_points: Away team score
                - home_epa: Home team EPA (optional, for EPA mode)
                - away_epa: Away team EPA (optional, for EPA mode)
                - recency_weight: Game weight (optional)
            weights: Optional explicit sample weights

        Returns:
            Self for chaining
        """
        logger.info(f"Fitting ridge model with alpha={self.alpha}")

        X, y, sample_weights = self._create_design_matrix(games_df, weights)

        # Fit ridge regression
        self.model = Ridge(alpha=self.alpha, fit_intercept=False)
        self.model.fit(X, y, sample_weight=sample_weights)

        # Extract coefficients
        n_teams = len(self.teams)
        team_ratings = self.model.coef_[:n_teams]

        # HFA coefficient (from model or fixed)
        if self.fixed_hfa is not None:
            self.hfa_coefficient = self.fixed_hfa
        else:
            self.hfa_coefficient = self.model.coef_[n_teams]

        # Create team ratings (store overall in offense, set defense to 0)
        # This maintains compatibility with existing code while using single rating
        self.ratings = {}
        for i, team in enumerate(self.teams):
            self.ratings[team] = TeamRatings(
                team=team,
                offense=team_ratings[i],
                defense=0.0,  # Not used in single-rating model
            )

        logger.info(
            f"Model fit complete. HFA: {self.hfa_coefficient:.2f} points. "
            f"Teams rated: {len(self.ratings)}"
        )

        return self

    def get_rating(self, team: str) -> Optional[TeamRatings]:
        """Get ratings for a specific team.

        Args:
            team: Team name

        Returns:
            TeamRatings object or None if team not found
        """
        return self.ratings.get(team)

    def get_all_ratings(self) -> pd.DataFrame:
        """Get all team ratings as a DataFrame.

        Returns:
            DataFrame with team ratings sorted by overall rating
        """
        if not self.ratings:
            return pd.DataFrame()

        data = [
            {
                "team": r.team,
                "offense": r.offense,
                "defense": r.defense,
                "overall": r.overall,
            }
            for r in self.ratings.values()
        ]

        df = pd.DataFrame(data)
        return df.sort_values("overall", ascending=False).reset_index(drop=True)

    def predict_margin(
        self,
        home_team: str,
        away_team: str,
        neutral_site: bool = False,
    ) -> float:
        """Predict point margin for a matchup.

        Args:
            home_team: Home team name
            away_team: Away team name
            neutral_site: If True, don't apply HFA

        Returns:
            Predicted margin (positive = home team favored)
        """
        home_rating = self.ratings.get(home_team)
        away_rating = self.ratings.get(away_team)

        if home_rating is None:
            logger.warning(f"No rating for {home_team}, using 0")
            home_overall = 0.0
        else:
            home_overall = home_rating.overall

        if away_rating is None:
            logger.warning(f"No rating for {away_team}, using 0")
            away_overall = 0.0
        else:
            away_overall = away_rating.overall

        # Simple model: margin = home_rating - away_rating + HFA
        margin = home_overall - away_overall

        if not neutral_site:
            margin += self.hfa_coefficient

        return margin

    def cross_validate(
        self,
        games_df: pd.DataFrame,
        cv_folds: int = 5,
        alphas: Optional[list[float]] = None,
    ) -> dict[float, float]:
        """Perform cross-validation to tune alpha.

        Args:
            games_df: DataFrame with game data
            cv_folds: Number of CV folds
            alphas: List of alpha values to try

        Returns:
            Dictionary mapping alpha to mean CV score (negative MSE)
        """
        if alphas is None:
            alphas = [50, 100, 150, 200, 250, 300]

        X, y, weights = self._create_design_matrix(games_df)
        results = {}

        for alpha in alphas:
            model = Ridge(alpha=alpha, fit_intercept=False)
            scores = cross_val_score(
                model, X, y, cv=cv_folds, scoring="neg_mean_squared_error"
            )
            results[alpha] = scores.mean()
            logger.info(f"Alpha {alpha}: Mean CV Score = {scores.mean():.4f}")

        return results

    def tune_alpha(
        self,
        games_df: pd.DataFrame,
        cv_folds: int = 5,
        alpha_range: tuple[float, float] = (50, 300),
        n_trials: int = 10,
    ) -> float:
        """Find optimal alpha via cross-validation.

        Args:
            games_df: DataFrame with game data
            cv_folds: Number of CV folds
            alpha_range: (min, max) range for alpha search
            n_trials: Number of alpha values to try

        Returns:
            Optimal alpha value
        """
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_trials)
        results = self.cross_validate(games_df, cv_folds, list(alphas))

        # Find alpha with best (least negative) score
        best_alpha = max(results, key=results.get)
        logger.info(f"Optimal alpha: {best_alpha}")

        return best_alpha
