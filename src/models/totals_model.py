"""
Totals Prediction Model: Opponent-adjusted scoring for over/under betting.

This model predicts game totals using team-level points scored/allowed,
adjusted for opponent strength via Ridge regression. It's separate from
the efficiency-based spread model (EFM) and is designed specifically for
totals betting.

Architecture:
- Uses game outcomes (points scored/allowed) not play-level efficiency
- Ridge regression solves for "true" offensive/defensive scoring vs average opponents
- Walk-forward training: only uses games from weeks < prediction_week

Formula:
    home_expected = baseline + (home_off_adj + away_def_adj) / 2
    away_expected = baseline + (away_off_adj + home_def_adj) / 2
    total = home_expected + away_expected

Where:
    - baseline = average points per team in FBS (typically ~26-27)
    - off_adj = team's offensive adjustment (+ = scores more than avg)
    - def_adj = team's defensive adjustment (+ = allows more than avg, i.e. worse defense)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


@dataclass
class TotalsRating:
    """Team's totals-specific ratings."""
    team: str
    adj_off_ppg: float  # Expected points scored vs average defense
    adj_def_ppg: float  # Expected points allowed vs average offense
    off_adjustment: float  # Offensive adjustment from baseline
    def_adjustment: float  # Defensive adjustment from baseline (+ = worse D)
    games_played: int


@dataclass
class TotalsPrediction:
    """Predicted game total with breakdown."""
    home_team: str
    away_team: str
    predicted_total: float
    home_expected: float
    away_expected: float
    baseline: float
    # Optional adjustments (weather, etc.)
    weather_adjustment: float = 0.0

    @property
    def adjusted_total(self) -> float:
        """Total after all adjustments."""
        return self.predicted_total + self.weather_adjustment


class TotalsModel:
    """Opponent-adjusted totals prediction model.

    Uses Ridge regression on game-level points to solve for team offensive
    and defensive adjustments relative to FBS average.

    Args:
        ridge_alpha: Regularization strength for Ridge regression.
            Lower = less regularization = more extreme adjustments.
            Recommended range: 5-15. Default 10.0 based on 2022-2025 backtest
            (Core 5+ Edge: 52.8% at alpha=10).
    """

    def __init__(self, ridge_alpha: float = 10.0):
        self.ridge_alpha = ridge_alpha
        self.team_ratings: dict[str, TotalsRating] = {}
        self.baseline: float = 26.0  # Will be set by training
        self._team_to_idx: dict[str, int] = {}
        self._trained = False

    def train(
        self,
        games_df: pd.DataFrame | pl.DataFrame,
        fbs_teams: set[str],
        max_week: Optional[int] = None,
    ) -> None:
        """Train the model on historical games.

        Args:
            games_df: DataFrame with columns: home_team, away_team, home_points,
                      away_points, week
            fbs_teams: Set of FBS team names to include
            max_week: Only use games from weeks <= max_week (for walk-forward).
                      If None, uses all games.
        """
        # Convert to pandas if needed
        if isinstance(games_df, pl.DataFrame):
            games = games_df.to_pandas()
        else:
            games = games_df.copy()

        # Filter to FBS vs FBS
        games = games[
            games['home_team'].isin(fbs_teams) &
            games['away_team'].isin(fbs_teams)
        ]

        # Filter to games with scores
        games = games[
            games['home_points'].notna() &
            games['away_points'].notna()
        ]

        # Walk-forward filter
        if max_week is not None:
            games = games[games['week'] <= max_week]

        if len(games) < 50:
            logger.warning(f"Only {len(games)} games for training - ratings may be unstable")
            if len(games) < 10:
                logger.error("Insufficient games for training")
                return

        # Get all teams
        teams = sorted(set(games['home_team'].unique()) | set(games['away_team'].unique()))
        self._team_to_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        logger.info(f"Training totals model: {len(games)} games, {n_teams} teams, max_week={max_week}")

        # Build design matrix
        # Each game contributes two rows:
        # Row 1: home_team scores home_points against away_team defense
        # Row 2: away_team scores away_points against home_team defense

        X = []
        y = []

        # Track games per team
        games_per_team = {t: 0 for t in teams}

        for _, g in games.iterrows():
            home = g['home_team']
            away = g['away_team']
            home_pts = g['home_points']
            away_pts = g['away_points']

            if home not in self._team_to_idx or away not in self._team_to_idx:
                continue

            home_idx = self._team_to_idx[home]
            away_idx = self._team_to_idx[away]

            games_per_team[home] += 1
            games_per_team[away] += 1

            # Home team offense vs Away team defense
            row1 = np.zeros(n_teams * 2)
            row1[home_idx] = 1  # Home offense
            row1[n_teams + away_idx] = 1  # Away defense (+ = allows more = worse)
            X.append(row1)
            y.append(home_pts)

            # Away team offense vs Home team defense
            row2 = np.zeros(n_teams * 2)
            row2[away_idx] = 1  # Away offense
            row2[n_teams + home_idx] = 1  # Home defense
            X.append(row2)
            y.append(away_pts)

        X = np.array(X)
        y = np.array(y)

        # Fit Ridge regression
        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        ridge.fit(X, y)

        # Extract coefficients
        self.baseline = ridge.intercept_
        off_coefs = ridge.coef_[:n_teams]
        def_coefs = ridge.coef_[n_teams:]

        logger.info(f"Baseline PPG: {self.baseline:.1f}")
        logger.info(f"Offensive adjustments: [{off_coefs.min():.1f}, {off_coefs.max():.1f}]")
        logger.info(f"Defensive adjustments: [{def_coefs.min():.1f}, {def_coefs.max():.1f}]")

        # Build ratings dict
        self.team_ratings = {}
        for team, idx in self._team_to_idx.items():
            self.team_ratings[team] = TotalsRating(
                team=team,
                adj_off_ppg=self.baseline + off_coefs[idx],
                adj_def_ppg=self.baseline + def_coefs[idx],
                off_adjustment=off_coefs[idx],
                def_adjustment=def_coefs[idx],
                games_played=games_per_team[team],
            )

        self._trained = True

    def predict_total(
        self,
        home_team: str,
        away_team: str,
        weather_adjustment: float = 0.0,
    ) -> Optional[TotalsPrediction]:
        """Predict the total for a game.

        Args:
            home_team: Home team name
            away_team: Away team name
            weather_adjustment: Optional adjustment for weather (negative = lower total)

        Returns:
            TotalsPrediction with breakdown, or None if teams not found
        """
        if not self._trained:
            logger.warning("Model not trained")
            return None

        home = self.team_ratings.get(home_team)
        away = self.team_ratings.get(away_team)

        if not home:
            logger.warning(f"Team not found: {home_team}")
            return None
        if not away:
            logger.warning(f"Team not found: {away_team}")
            return None

        # SP+-style formula:
        # Each team's expected points = (their offense adj + opposing defense adj) / 2 + baseline
        home_expected = self.baseline + (home.off_adjustment + away.def_adjustment) / 2
        away_expected = self.baseline + (away.off_adjustment + home.def_adjustment) / 2

        predicted_total = home_expected + away_expected

        return TotalsPrediction(
            home_team=home_team,
            away_team=away_team,
            predicted_total=predicted_total,
            home_expected=home_expected,
            away_expected=away_expected,
            baseline=self.baseline,
            weather_adjustment=weather_adjustment,
        )

    def get_ratings_df(self) -> pd.DataFrame:
        """Get ratings as a sorted DataFrame."""
        if not self.team_ratings:
            return pd.DataFrame()

        rows = []
        for team, r in self.team_ratings.items():
            rows.append({
                'team': team,
                'adj_off_ppg': r.adj_off_ppg,
                'adj_def_ppg': r.adj_def_ppg,
                'off_adjustment': r.off_adjustment,
                'def_adjustment': r.def_adjustment,
                'games_played': r.games_played,
            })

        df = pd.DataFrame(rows)
        # Sort by offensive rating descending
        df = df.sort_values('adj_off_ppg', ascending=False).reset_index(drop=True)
        return df


def walk_forward_totals_backtest(
    games_df: pd.DataFrame | pl.DataFrame,
    fbs_teams: set[str],
    start_week: int = 4,
    ridge_alpha: float = 5.0,
) -> dict:
    """Run walk-forward backtest for totals model.

    For each week W >= start_week, trains on weeks 1 to W-1 and predicts week W.

    Args:
        games_df: All games for the season
        fbs_teams: Set of FBS team names
        start_week: First week to predict (need enough training data)
        ridge_alpha: Ridge regularization strength

    Returns:
        Dict with predictions, errors, and summary metrics
    """
    # Convert to pandas
    if isinstance(games_df, pl.DataFrame):
        games = games_df.to_pandas()
    else:
        games = games_df.copy()

    # Filter to FBS
    games = games[
        games['home_team'].isin(fbs_teams) &
        games['away_team'].isin(fbs_teams) &
        games['home_points'].notna() &
        games['away_points'].notna()
    ]

    max_week = int(games['week'].max())

    predictions = []

    for pred_week in range(start_week, max_week + 1):
        # Train on weeks < pred_week
        model = TotalsModel(ridge_alpha=ridge_alpha)
        model.train(games, fbs_teams, max_week=pred_week - 1)

        if not model._trained:
            continue

        # Predict games in pred_week
        week_games = games[games['week'] == pred_week]

        for _, g in week_games.iterrows():
            pred = model.predict_total(g['home_team'], g['away_team'])
            if pred:
                actual_total = g['home_points'] + g['away_points']
                predictions.append({
                    'week': pred_week,
                    'home_team': g['home_team'],
                    'away_team': g['away_team'],
                    'predicted_total': pred.predicted_total,
                    'actual_total': actual_total,
                    'error': pred.predicted_total - actual_total,
                    'abs_error': abs(pred.predicted_total - actual_total),
                })

    if not predictions:
        return {'predictions': [], 'mae': None, 'rmse': None, 'mean_error': None}

    preds_df = pd.DataFrame(predictions)

    return {
        'predictions': predictions,
        'mae': preds_df['abs_error'].mean(),
        'rmse': np.sqrt((preds_df['error'] ** 2).mean()),
        'mean_error': preds_df['error'].mean(),
        'n_games': len(predictions),
    }
