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
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge

from src.adjustments.weather import WeatherAdjuster, WeatherConditions, WeatherAdjustment

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

    def __init__(self, ridge_alpha: float = 10.0, decay_factor: float = 1.0):
        self.ridge_alpha = ridge_alpha
        self.decay_factor = decay_factor  # Within-season recency weight (1.0 = no decay)
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

        # Build design matrix using sparse COO construction (vectorized)
        # Each game contributes two rows:
        # Row 1: home_team scores home_points against away_team defense
        # Row 2: away_team scores away_points against home_team defense

        # Map team names to indices
        home_idx = games['home_team'].map(self._team_to_idx)
        away_idx = games['away_team'].map(self._team_to_idx)

        # Filter out games where either team is not in index (shouldn't happen after FBS filter)
        valid_mask = home_idx.notna() & away_idx.notna()
        games = games[valid_mask].reset_index(drop=True)
        home_idx = home_idx[valid_mask].astype(int).values
        away_idx = away_idx[valid_mask].astype(int).values

        n_games = len(games)
        home_pts = games['home_points'].values
        away_pts = games['away_points'].values
        game_weeks = games['week'].values

        # Track games per team
        games_per_team = {t: 0 for t in teams}
        for h, a in zip(games['home_team'], games['away_team']):
            games_per_team[h] += 1
            games_per_team[a] += 1

        # Build sparse matrix: each game = 2 rows, each row has 2 non-zero entries
        # Row 2i:   home_off (col=home_idx), away_def (col=n_teams+away_idx)
        # Row 2i+1: away_off (col=away_idx), home_def (col=n_teams+home_idx)
        n_rows = 2 * n_games
        n_cols = 2 * n_teams

        # COO matrix arrays: 4 entries per game (2 rows × 2 cols each)
        row_indices = np.empty(4 * n_games, dtype=np.int32)
        col_indices = np.empty(4 * n_games, dtype=np.int32)
        data = np.ones(4 * n_games, dtype=np.float64)

        game_range = np.arange(n_games)

        # Row 2i: home offense + away defense
        row_indices[0::4] = 2 * game_range          # home offense entry
        col_indices[0::4] = home_idx                # offense column
        row_indices[1::4] = 2 * game_range          # away defense entry (same row)
        col_indices[1::4] = n_teams + away_idx      # defense column

        # Row 2i+1: away offense + home defense
        row_indices[2::4] = 2 * game_range + 1      # away offense entry
        col_indices[2::4] = away_idx                # offense column
        row_indices[3::4] = 2 * game_range + 1      # home defense entry (same row)
        col_indices[3::4] = n_teams + home_idx      # defense column

        X = coo_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols)).tocsr()

        # Build y vector: [home_pts_0, away_pts_0, home_pts_1, away_pts_1, ...]
        y = np.empty(n_rows, dtype=np.float64)
        y[0::2] = home_pts
        y[1::2] = away_pts

        # Build weeks vector for recency weighting
        weeks = np.empty(n_rows, dtype=np.float64)
        weeks[0::2] = game_weeks
        weeks[1::2] = game_weeks

        # Compute sample weights for recency (decay_factor^weeks_ago)
        # pred_week = max_week + 1 (we predict the week after training data)
        pred_week = (max_week or int(weeks.max())) + 1
        weeks_ago = pred_week - weeks
        sample_weights = self.decay_factor ** weeks_ago

        # Fit Ridge regression with sample weights
        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        ridge.fit(X, y, sample_weight=sample_weights)

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

    def predict_total_with_weather(
        self,
        home_team: str,
        away_team: str,
        weather_data,
        weather_adjuster: Optional[WeatherAdjuster] = None,
    ) -> Optional[TotalsPrediction]:
        """Predict total with automatic weather adjustment.

        Args:
            home_team: Home team name
            away_team: Away team name
            weather_data: CFBD GameWeather object or WeatherConditions
            weather_adjuster: WeatherAdjuster instance (uses default if None)

        Returns:
            TotalsPrediction with weather adjustment applied
        """
        if weather_adjuster is None:
            weather_adjuster = WeatherAdjuster()

        # Convert API object to WeatherConditions if needed
        if hasattr(weather_data, 'game_indoors'):
            # It's a CFBD API GameWeather object
            weather_adj = weather_adjuster.calculate_adjustment_from_api(weather_data)
        elif isinstance(weather_data, WeatherConditions):
            weather_adj = weather_adjuster.calculate_adjustment(weather_data)
        else:
            logger.warning(f"Unknown weather data type: {type(weather_data)}")
            weather_adj = WeatherAdjustment(
                total_adjustment=0.0,
                wind_adjustment=0.0,
                temperature_adjustment=0.0,
                precipitation_adjustment=0.0,
                is_indoor=False,
            )

        return self.predict_total(
            home_team=home_team,
            away_team=away_team,
            weather_adjustment=weather_adj.total_adjustment,
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
