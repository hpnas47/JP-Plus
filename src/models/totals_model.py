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
- Learned HFA: Home field advantage is learned from data (typically +3-4 pts)
- Year intercepts: Each year gets its own baseline to handle scoring environment shifts

Formula:
    home_expected = year_baseline + (home_off_adj + away_def_adj) / 2 + hfa_coef
    away_expected = year_baseline + (away_off_adj + home_def_adj) / 2
    total = home_expected + away_expected

Where:
    - year_baseline = year-specific average points per team (handles 57→53 PPG trend)
    - off_adj = team's offensive adjustment (+ = scores more than avg)
    - def_adj = team's defensive adjustment (+ = allows more than avg, i.e. worse defense)
    - hfa_coef = learned home field advantage (typically +3-4 pts for home team)

Performance Note:
    Incremental matrix caching was evaluated (Feb 2026) to avoid rebuilding the
    sparse matrix for games already processed in walk-forward iterations. However,
    profiling showed Ridge.fit() dominates runtime (0.5ms vs 0.08ms for matrix ops),
    and cache overhead (vstack, concatenation) actually made training SLOWER.
    The simple rebuild-each-time approach is preferred for clarity.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import coo_matrix
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

    def __repr__(self) -> str:
        return (
            f"TotalsRating({self.team}: off={self.adj_off_ppg:.1f}, "
            f"def={self.adj_def_ppg:.1f}, games={self.games_played})"
        )


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

    def __repr__(self) -> str:
        if self.weather_adjustment != 0.0:
            return (
                f"TotalsPrediction({self.away_team} @ {self.home_team}: "
                f"base={self.predicted_total:.1f}, weather={self.weather_adjustment:+.1f}, "
                f"adj={self.adjusted_total:.1f})"
            )
        return (
            f"TotalsPrediction({self.away_team} @ {self.home_team}: "
            f"total={self.predicted_total:.1f})"
        )


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

    def __init__(self, ridge_alpha: float = 10.0, decay_factor: float = 1.0, use_year_intercepts: bool = False):
        # Validate decay_factor: must be in (0, 1.0]
        # Values > 1.0 would UPWEIGHT old games (opposite of intended recency bias)
        if not (0 < decay_factor <= 1.0):
            raise ValueError(
                f"decay_factor must be in (0, 1.0], got {decay_factor}. "
                f"Values > 1.0 upweight old games instead of recent ones."
            )
        self.ridge_alpha = ridge_alpha
        self.decay_factor = decay_factor  # Within-season recency weight (1.0 = no decay)
        self.use_year_intercepts = use_year_intercepts  # Per-year baselines for multi-year training
        self.team_ratings: dict[str, TotalsRating] = {}
        self.baseline: float = 26.0  # Will be set by training
        self.year_baselines: dict[int, float] = {}  # Year-specific baselines (if enabled)
        self.hfa_coef: float = 0.0  # Learned home field advantage
        self._team_to_idx: dict[str, int] = {}
        self._n_teams: int = 0  # Cached team count for column layout
        self._team_universe_set: bool = False  # Lock flag for team universe
        self._year_to_idx: dict[int, int] = {}  # Year to column index mapping
        self._trained = False

    def set_team_universe(self, fbs_teams: set[str]) -> None:
        """Set the team universe once for reuse across train() calls.

        This locks the team-to-index mapping so column layout is consistent
        across walk-forward weeks. Teams that haven't played yet get zero
        columns in the design matrix (Ridge shrinks their coefficients to 0).

        Args:
            fbs_teams: Full set of FBS team names for this season
        """
        if self._team_universe_set:
            return  # Already set, no-op for idempotence
        teams = sorted(fbs_teams)
        self._team_to_idx = {t: i for i, t in enumerate(teams)}
        self._n_teams = len(teams)
        self._team_universe_set = True

    def reset(self) -> None:
        """Reset model state for reuse with a new season/team universe.

        Clears the team universe lock, ratings, and trained state while
        preserving configuration (ridge_alpha, decay_factor, use_year_intercepts).

        Use this when:
        - Starting a new season with different FBS members (conference realignment)
        - Switching between single-season and multi-season training
        - Clearing state for a fresh training run
        """
        self.team_ratings = {}
        self.baseline = 26.0
        self.year_baselines = {}
        self.hfa_coef = 0.0
        self._team_to_idx = {}
        self._n_teams = 0
        self._team_universe_set = False
        self._year_to_idx = {}
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
        # Convert to pandas if needed (Polars for ingest, pandas for sklearn is the project pattern)
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
            # Track for external leakage verification
            self._last_train_max_week = max_week
            # DATA LEAKAGE GUARD: Verify no future weeks slipped through
            actual_max = int(games['week'].max()) if len(games) > 0 else 0
            if actual_max > max_week:
                raise ValueError(
                    f"DATA LEAKAGE in TotalsModel: games include week {actual_max} "
                    f"but max_week={max_week}. Check filtering logic."
                )
        else:
            self._last_train_max_week = int(games['week'].max()) if len(games) > 0 else 0

        if len(games) < 50:
            logger.warning(f"Only {len(games)} games for training - ratings may be unstable")
            if len(games) < 10:
                logger.error("Insufficient games for training")
                return

        # Set team universe if not already set (allows reuse across train() calls)
        if not self._team_universe_set:
            self.set_team_universe(fbs_teams)
        n_teams = self._n_teams

        logger.info(f"Training totals model: {len(games)} games, {n_teams} teams, max_week={max_week}")

        # Build design matrix using sparse COO construction (vectorized)
        # Each game contributes two rows:
        # Row 1: home_team scores home_points against away_team defense
        # Row 2: away_team scores away_points against home_team defense

        # Filter out games where either team is not in index (shouldn't happen after FBS filter)
        valid_mask = (
            games['home_team'].isin(self._team_to_idx) &
            games['away_team'].isin(self._team_to_idx)
        )
        games = games[valid_mask].reset_index(drop=True)

        # Map team names to indices AFTER filter/reset for robust alignment
        home_idx = games['home_team'].map(self._team_to_idx).astype(int).values
        away_idx = games['away_team'].map(self._team_to_idx).astype(int).values

        n_games = len(games)
        home_pts = games['home_points'].values
        away_pts = games['away_points'].values
        game_weeks = games['week'].values

        # Track games per team (vectorized - use full universe so teams without games get 0)
        home_counts = games['home_team'].value_counts()
        away_counts = games['away_team'].value_counts()
        all_counts = home_counts.add(away_counts, fill_value=0).astype(int)
        games_per_team = {t: int(all_counts.get(t, 0)) for t in self._team_to_idx}

        # Get years for year intercepts (handles scoring environment shift)
        # Auto-detect: if multiple years present, FORCE year intercepts to avoid era-averaging bug
        if 'year' in games.columns:
            game_years = games['year'].values
            years = sorted(set(game_years))

            # Auto-enable year intercepts for multi-year training (fixes "Year Leakage" bug)
            # Without this, Ridge learns a single intercept averaging 2022 (~27 PPG) with 2024 (~26 PPG)
            if len(years) > 1 and not self.use_year_intercepts:
                logger.info(f"Auto-enabling year intercepts for multi-year training ({years})")
                self.use_year_intercepts = True

            if self.use_year_intercepts:
                self._year_to_idx = {y: i for i, y in enumerate(years)}
                n_years = len(years)
            else:
                years = []
                n_years = 0
        else:
            years = []
            n_years = 0
            game_years = None

        # Build sparse matrix: each game = 2 rows
        # Columns: [0..n_teams-1] = offense, [n_teams..2*n_teams-1] = defense, [2*n_teams] = HFA
        # Optional: [2*n_teams+1..] = year indicators (if use_year_intercepts)
        n_rows = 2 * n_games
        n_cols = 2 * n_teams + 1 + n_years  # +1 HFA + n_years for year intercepts

        game_range = np.arange(n_games)

        if n_years > 0:
            # With year intercepts: 7 entries per game
            year_col_base = 2 * n_teams + 1
            game_year_cols = np.array([self._year_to_idx[y] for y in game_years], dtype=np.int32)

            n_entries = 7 * n_games
            row_indices = np.empty(n_entries, dtype=np.int32)
            col_indices = np.empty(n_entries, dtype=np.int32)
            data = np.ones(n_entries, dtype=np.float64)

            # Row 2i (home): offense + defense + HFA + year
            row_indices[0::7] = 2 * game_range
            col_indices[0::7] = home_idx
            row_indices[1::7] = 2 * game_range
            col_indices[1::7] = n_teams + away_idx
            row_indices[2::7] = 2 * game_range
            col_indices[2::7] = 2 * n_teams  # HFA
            row_indices[3::7] = 2 * game_range
            col_indices[3::7] = year_col_base + game_year_cols

            # Row 2i+1 (away): offense + defense + year (no HFA)
            row_indices[4::7] = 2 * game_range + 1
            col_indices[4::7] = away_idx
            row_indices[5::7] = 2 * game_range + 1
            col_indices[5::7] = n_teams + home_idx
            row_indices[6::7] = 2 * game_range + 1
            col_indices[6::7] = year_col_base + game_year_cols
        else:
            # Without year intercepts: 5 entries per game (original approach)
            n_entries = 5 * n_games
            row_indices = np.empty(n_entries, dtype=np.int32)
            col_indices = np.empty(n_entries, dtype=np.int32)
            data = np.ones(n_entries, dtype=np.float64)

            # Row 2i (home): offense + defense + HFA
            row_indices[0::5] = 2 * game_range
            col_indices[0::5] = home_idx
            row_indices[1::5] = 2 * game_range
            col_indices[1::5] = n_teams + away_idx
            row_indices[2::5] = 2 * game_range
            col_indices[2::5] = 2 * n_teams  # HFA

            # Row 2i+1 (away): offense + defense (no HFA)
            row_indices[3::5] = 2 * game_range + 1
            col_indices[3::5] = away_idx
            row_indices[4::5] = 2 * game_range + 1
            col_indices[4::5] = n_teams + home_idx

        X = coo_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols)).tocsr()

        # Build y vector: [home_pts_0, away_pts_0, home_pts_1, away_pts_1, ...]
        y = np.empty(n_rows, dtype=np.float64)
        y[0::2] = home_pts
        y[1::2] = away_pts

        # Compute sample weights for recency (decay_factor^weeks_ago)
        # Short-circuit when decay_factor == 1.0 (no decay = uniform weights)
        if self.decay_factor == 1.0:
            sample_weights = None  # Ridge uses uniform weights when None
        else:
            # Guard: recency weighting is only valid for single-year training
            # With multi-year data, week numbers repeat (2022 week 9 vs 2024 week 9)
            # and weeks_ago would incorrectly treat them as equally recent.
            if n_years > 1:
                raise ValueError(
                    f"decay_factor={self.decay_factor} is incompatible with multi-year training "
                    f"({n_years} years). Week-based recency doesn't account for year boundaries. "
                    f"Use decay_factor=1.0 (no decay) for multi-year training."
                )
            # Build weeks vector for recency weighting (single-year only)
            weeks = np.empty(n_rows, dtype=np.float64)
            weeks[0::2] = game_weeks
            weeks[1::2] = game_weeks
            # pred_week = max_week + 1 (we predict the week after training data)
            pred_week = (max_week or int(weeks.max())) + 1
            weeks_ago = pred_week - weeks
            sample_weights = self.decay_factor ** weeks_ago

        # Fit Ridge regression with sample weights
        # fit_intercept: True unless using year intercepts (which serve as baselines)
        # solver='sparse_cg': Conjugate gradient for sparse CSR, deterministic, no densification
        use_intercept = not (n_years > 0)
        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=use_intercept, solver='sparse_cg')
        ridge.fit(X, y, sample_weight=sample_weights)

        # Extract coefficients
        off_coefs = ridge.coef_[:n_teams]
        def_coefs = ridge.coef_[n_teams:2 * n_teams]
        self.hfa_coef = ridge.coef_[2 * n_teams]  # HFA column

        if n_years > 0:
            # Year intercepts mode
            year_coefs = ridge.coef_[2 * n_teams + 1:]
            self.year_baselines = {year: year_coefs[idx] for year, idx in self._year_to_idx.items()}
            most_recent_year = max(years)
            self.baseline = self.year_baselines[most_recent_year]
            year_str = ", ".join(f"{y}: {self.year_baselines[y]:.1f}" for y in sorted(years))
            logger.info(f"Year baselines: {year_str}")
        else:
            # Standard mode with intercept
            self.baseline = ridge.intercept_
            self.year_baselines = {}
            logger.info(f"Baseline PPG: {self.baseline:.1f}")

        logger.info(f"Offensive adjustments: [{off_coefs.min():.1f}, {off_coefs.max():.1f}]")
        logger.info(f"Defensive adjustments: [{def_coefs.min():.1f}, {def_coefs.max():.1f}]")
        logger.info(f"Learned HFA: {self.hfa_coef:+.1f} pts")

        # Build ratings dict (only for teams with games - preserves prediction behavior)
        self.team_ratings = {}
        for team, idx in self._team_to_idx.items():
            if games_per_team[team] > 0:  # Only include teams that have played
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
        year: Optional[int] = None,
    ) -> Optional[TotalsPrediction]:
        """Predict the total for a game.

        Args:
            home_team: Home team name
            away_team: Away team name
            weather_adjustment: Optional adjustment for weather (negative = lower total)
            year: Optional year for year-specific baseline (uses most recent if not provided)

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

        # P2.1 FIX: Validate ratings are finite (catch NaN propagation)
        if not np.isfinite(home.off_adjustment) or not np.isfinite(home.def_adjustment):
            logger.warning(
                f"Non-finite rating for {home_team}: off={home.off_adjustment}, def={home.def_adjustment}"
            )
            return None
        if not np.isfinite(away.off_adjustment) or not np.isfinite(away.def_adjustment):
            logger.warning(
                f"Non-finite rating for {away_team}: off={away.off_adjustment}, def={away.def_adjustment}"
            )
            return None

        # Get year-specific baseline (handles scoring environment shift)
        if year is not None and year in self.year_baselines:
            baseline = self.year_baselines[year]
        else:
            baseline = self.baseline  # Most recent year's baseline

        # Opponent-adjustment formula with 0.5x shrinkage on adjustments:
        # The /2 is intentional shrinkage that improves ATS despite hurting MAE.
        # Without /2: MAE 12.90 (better) but 5+ Edge 53.3% (worse)
        # With /2:    MAE 13.09 (worse) but 5+ Edge 54.5% (better)
        # Since 5+ Edge is the binding constraint, we keep the shrinkage.
        home_expected = baseline + (home.off_adjustment + away.def_adjustment) / 2 + self.hfa_coef
        away_expected = baseline + (away.off_adjustment + home.def_adjustment) / 2

        # Sanity bounds for degenerate cases (early season, bad data)
        # Per-team floor: no team scores negative points
        # Total floor: 21 (lowest FBS totals are ~23)
        # Total ceiling: 105 (highest FBS totals are ~100)
        TEAM_FLOOR = 0.0
        TOTAL_FLOOR = 21.0
        TOTAL_CEILING = 105.0

        home_expected = max(TEAM_FLOOR, home_expected)
        away_expected = max(TEAM_FLOOR, away_expected)

        # predicted_total is the base model prediction BEFORE weather adjustments
        # Weather adjustment is stored separately and applied in adjusted_total property
        predicted_total = home_expected + away_expected
        predicted_total = max(TOTAL_FLOOR, min(TOTAL_CEILING, predicted_total))

        return TotalsPrediction(
            home_team=home_team,
            away_team=away_team,
            predicted_total=predicted_total,
            home_expected=home_expected,
            away_expected=away_expected,
            baseline=baseline,
            weather_adjustment=weather_adjustment,
        )

    def get_ratings_df(self, min_games: int = 0) -> pd.DataFrame:
        """Get ratings as a sorted DataFrame.

        Args:
            min_games: Minimum games played to include (0 = all teams with any games).
                       Use 3-4 to filter out teams with unreliable early-season ratings.

        Returns:
            DataFrame with columns: team, adj_off_ppg, adj_def_ppg, off_adjustment,
            def_adjustment, games_played, reliability.

            Reliability is a 0-1 score based on games_played:
            - 0.0: 1 game (very unreliable)
            - 0.5: 4 games (moderate)
            - 1.0: 8+ games (fully reliable)
        """
        if not self.team_ratings:
            return pd.DataFrame()

        rows = []
        for team, r in self.team_ratings.items():
            if r.games_played < min_games:
                continue
            # Reliability: linear ramp from 0 (1 game) to 1.0 (8+ games)
            reliability = min(1.0, max(0.0, (r.games_played - 1) / 7))
            rows.append({
                'team': team,
                'adj_off_ppg': r.adj_off_ppg,
                'adj_def_ppg': r.adj_def_ppg,
                'off_adjustment': r.off_adjustment,
                'def_adjustment': r.def_adjustment,
                'games_played': r.games_played,
                'reliability': round(reliability, 2),
            })

        df = pd.DataFrame(rows)
        # Sort by offensive rating descending
        df = df.sort_values('adj_off_ppg', ascending=False).reset_index(drop=True)
        return df


def walk_forward_totals_backtest(
    games_df: pd.DataFrame | pl.DataFrame,
    fbs_teams: set[str],
    start_week: int = 4,
    ridge_alpha: float = 10.0,  # Matches TotalsModel default; 10.0 optimal for 5+ Edge
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
    # Convert to pandas if needed (Polars for ingest, pandas for sklearn is the project pattern)
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

    # Create model once and lock team universe for consistent column layout
    # This avoids redundant set_team_universe() calls inside train()
    model = TotalsModel(ridge_alpha=ridge_alpha)
    model.set_team_universe(fbs_teams)

    for pred_week in range(start_week, max_week + 1):
        # Retrain on weeks < pred_week (model state is properly reset in train())
        model.train(games, fbs_teams, max_week=pred_week - 1)

        if not model._trained:
            continue

        # Predict games in pred_week
        week_games = games[games['week'] == pred_week]

        for g in week_games.itertuples():
            # Pass year for correct year-specific baseline (P0.1: multi-year backtest fix)
            game_year = getattr(g, 'year', None)
            pred = model.predict_total(g.home_team, g.away_team, year=game_year)
            if pred:
                actual_total = g.home_points + g.away_points
                predictions.append({
                    'year': game_year,
                    'week': pred_week,
                    'home_team': g.home_team,
                    'away_team': g.away_team,
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
