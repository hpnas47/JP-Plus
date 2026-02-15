"""Ridge regression model for preseason SP+ prediction.

Walk-forward validation:
- Train on years [start, year-1], predict year
- Alpha selected via walk-forward MAE (manual sweep, not cross-validation)
- Out-of-fold predictions cached for honest calibration

Alpha selection protocol:
  For each fold year Y, train Ridge(alpha) on years < Y for each candidate
  alpha, compute MAE on year Y. The production alpha is the alpha with
  lowest average MAE across all folds.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.win_totals.features import PreseasonFeatureBuilder, FEATURE_METADATA

logger = logging.getLogger(__name__)

ALPHA_CANDIDATES = [0.1, 1.0, 10.0, 50.0, 100.0, 200.0]


def _impute_nan_with_means(
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Replace NaN with training column means.

    Returns copies; originals are not modified.

    Args:
        X_train: Training features (may contain NaN)
        X_test: Test features (may contain NaN), or None

    Returns:
        (X_train_imputed, X_test_imputed_or_None, col_means)
    """
    col_means = np.nanmean(X_train, axis=0)
    # If an entire column is NaN, fall back to 0.0
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    X_train_out = X_train.copy()
    for j in range(X_train_out.shape[1]):
        mask = np.isnan(X_train_out[:, j])
        X_train_out[mask, j] = col_means[j]

    X_test_out = None
    if X_test is not None:
        X_test_out = X_test.copy()
        for j in range(X_test_out.shape[1]):
            mask = np.isnan(X_test_out[:, j])
            X_test_out[mask, j] = col_means[j]

    return X_train_out, X_test_out, col_means


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    year: int
    fold_alpha: float  # Best alpha for this fold
    mae: float
    rmse: float
    n_teams: int
    predictions: pd.DataFrame  # team, year, predicted_sp, actual_sp
    coefficients: np.ndarray
    feature_names: list[str]
    intercept: float


@dataclass
class ValidationResult:
    """Aggregate walk-forward validation results."""
    folds: list[FoldResult] = field(default_factory=list)
    tau: float = 0.0  # calibrated from production_alpha residuals
    production_alpha: float = 0.0  # alpha with lowest avg MAE across folds
    alpha_mae_table: dict[float, float] = field(default_factory=dict)  # alpha -> avg MAE
    production_residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    _production_predictions: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def per_fold_optimized_mae(self) -> float:
        """MAE from fold predictions where each fold used its own best alpha.

        Mildly optimistic — each fold selected the alpha that performed best
        on that specific fold's test data. Use production_mae for honest estimates.
        """
        all_preds = pd.concat([f.predictions for f in self.folds])
        return float(np.mean(np.abs(all_preds['predicted_sp'] - all_preds['actual_sp'])))

    @property
    def per_fold_optimized_rmse(self) -> float:
        """RMSE from fold predictions where each fold used its own best alpha."""
        all_preds = pd.concat([f.predictions for f in self.folds])
        return float(np.sqrt(np.mean((all_preds['predicted_sp'] - all_preds['actual_sp']) ** 2)))

    @property
    def production_mae(self) -> float:
        """Honest MAE from production_alpha residuals (consistent alpha across all folds)."""
        if len(self.production_residuals) > 0:
            return float(np.mean(np.abs(self.production_residuals)))
        return self.per_fold_optimized_mae

    @property
    def production_rmse(self) -> float:
        """Honest RMSE from production_alpha residuals (consistent alpha across all folds)."""
        if len(self.production_residuals) > 0:
            return float(np.sqrt(np.mean(self.production_residuals ** 2)))
        return self.per_fold_optimized_rmse

    @property
    def production_predictions(self) -> pd.DataFrame:
        """OOF predictions generated with production_alpha (consistent across folds).

        Used for calibration to ensure spread scale matches production model.
        Falls back to per-fold-optimized predictions if production run wasn't done.
        """
        if self._production_predictions is not None:
            return self._production_predictions
        return self.all_predictions

    @property
    def all_predictions(self) -> pd.DataFrame:
        """Per-fold-optimized OOF predictions (each fold used its own best alpha)."""
        return pd.concat([f.predictions for f in self.folds], ignore_index=True)

    @property
    def all_residuals(self) -> np.ndarray:
        preds = self.all_predictions
        return (preds['predicted_sp'] - preds['actual_sp']).values

    def feature_importance(self) -> pd.DataFrame:
        """Average absolute coefficients across folds."""
        if not self.folds:
            return pd.DataFrame()

        names = self.folds[0].feature_names
        coef_matrix = np.array([f.coefficients for f in self.folds])
        avg_abs = np.mean(np.abs(coef_matrix), axis=0)
        avg_raw = np.mean(coef_matrix, axis=0)

        df = pd.DataFrame({
            'feature': names,
            'avg_abs_coef': avg_abs,
            'avg_coef': avg_raw,
            'std_coef': np.std(coef_matrix, axis=0),
        })
        return df.sort_values('avg_abs_coef', ascending=False).reset_index(drop=True)


class PreseasonModel:
    """Ridge regression for preseason SP+ prediction.

    IMPORTANT: Alpha selection is done via manual walk-forward, not CV.
    explicit walk-forward: for each fold year Y, try all candidate alphas,
    pick best MAE on year Y. Production alpha = lowest avg MAE across folds.
    """

    def __init__(
        self,
        alpha_candidates: list[float] | None = None,
    ):
        self.alpha_candidates = alpha_candidates or ALPHA_CANDIDATES
        self.scaler: StandardScaler | None = None
        self.model: Ridge | None = None
        self.feature_names: list[str] = []
        self.selected_alpha: float = 0.0
        self._col_means: np.ndarray | None = None  # for consistent NaN imputation

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Get feature columns (exclude team, year, target)."""
        exclude = {'team', 'year', 'target_sp'}
        return [c for c in df.columns if c not in exclude]

    def _fit_and_predict_fold(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_names: list[str],
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, Ridge, StandardScaler, np.ndarray]:
        """Fit Ridge on train, predict on test. Shared preprocessing logic.

        Returns:
            (predictions, col_means, fitted_ridge, fitted_scaler, y_test)
        """
        X_train_raw = train_df[feature_names].values.astype(np.float64)
        y_train = train_df['target_sp'].values.astype(np.float64)
        X_test_raw = test_df[feature_names].values.astype(np.float64)
        y_test = test_df['target_sp'].values.astype(np.float64)

        X_train, X_test, col_means = _impute_nan_with_means(X_train_raw, X_test_raw)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        preds = ridge.predict(X_test_scaled)

        return preds, col_means, ridge, scaler, y_test

    def train(
        self,
        train_df: pd.DataFrame,
        alpha: float | None = None,
    ) -> tuple[Ridge, StandardScaler]:
        """Train Ridge model on provided data with a specific alpha.

        Args:
            train_df: Training data with features + target_sp
            alpha: Ridge alpha. If None, uses self.selected_alpha or 10.0 default.

        Returns:
            (fitted model, fitted scaler)
        """
        if alpha is None:
            alpha = self.selected_alpha if self.selected_alpha > 0 else 10.0

        self.feature_names = self._get_feature_cols(train_df)
        X = train_df[self.feature_names].values.astype(np.float64)
        y = train_df['target_sp'].values.astype(np.float64)

        X, _, self._col_means = _impute_nan_with_means(X)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = Ridge(alpha=alpha)
        self.model.fit(X_scaled, y)
        self.selected_alpha = alpha

        logger.info(f"Trained Ridge: alpha={alpha:.1f}, "
                     f"n={len(y)}, features={len(self.feature_names)}")

        return self.model, self.scaler

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict SP+ ratings for given features.

        Args:
            df: DataFrame with feature columns

        Returns:
            Array of predicted SP+ ratings
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = df[self.feature_names].values.astype(np.float64)
        # Impute NaN with training column means for consistent z-scores
        col_means = getattr(self, '_col_means', None)
        if col_means is not None:
            for j in range(X.shape[1]):
                mask = np.isnan(X[:, j])
                X[mask, j] = col_means[j]
        else:
            X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def validate(
        self,
        dataset: pd.DataFrame,
        start_year: int | None = None,
        end_year: int | None = None,
        min_train_years: int = 3,
    ) -> ValidationResult:
        """Walk-forward validation with per-fold alpha selection.

        For each year Y in [start_year, end_year]:
        - Train on all data from years < Y
        - For each candidate alpha, fit Ridge(alpha) and compute MAE on year Y
        - Record fold-best alpha and predictions using that alpha
        After all folds: production_alpha = alpha with lowest average MAE.
        Then re-run all folds with production_alpha for consistent tau calibration,
        and retrain on full dataset with production_alpha for self.model.

        Args:
            dataset: Full training dataset from build_training_dataset()
            start_year: First year to predict (default: first year with >= min_train_years prior)
            end_year: Last year to predict (default: max year)
            min_train_years: Minimum years of training data required

        Returns:
            ValidationResult with all fold results and production alpha
        """
        years = sorted(dataset['year'].unique())

        if start_year is None:
            # Find first year with enough prior years of data
            for y in years:
                if sum(1 for yy in years if yy < y) >= min_train_years:
                    start_year = y
                    break
            else:
                start_year = years[-1]  # fallback
        if end_year is None:
            end_year = years[-1]

        result = ValidationResult()

        # Track MAE per alpha across folds for production alpha selection
        alpha_fold_maes: dict[float, list[float]] = {
            a: [] for a in self.alpha_candidates
        }

        # Store fold data splits for production_alpha re-run
        fold_splits: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []

        for pred_year in range(start_year, end_year + 1):
            if pred_year not in years:
                continue

            train_mask = dataset['year'] < pred_year
            test_mask = dataset['year'] == pred_year

            train_df = dataset[train_mask]
            test_df = dataset[test_mask]

            if len(train_df) < 50:
                logger.warning(f"Skipping {pred_year}: only {len(train_df)} training samples")
                continue

            fold_splits.append((pred_year, train_df, test_df))

            feature_names = self._get_feature_cols(train_df)

            # Try each alpha candidate on this fold
            best_alpha = self.alpha_candidates[0]
            best_mae = float('inf')
            best_preds = None
            best_model = None

            for alpha in self.alpha_candidates:
                preds, _, ridge, _, y_test = self._fit_and_predict_fold(
                    train_df, test_df, feature_names, alpha
                )
                mae = float(np.mean(np.abs(preds - y_test)))
                alpha_fold_maes[alpha].append(mae)

                if mae < best_mae:
                    best_mae = mae
                    best_alpha = alpha
                    best_preds = preds
                    best_model = ridge

            y_test = test_df['target_sp'].values.astype(np.float64)
            pred_df = pd.DataFrame({
                'team': test_df['team'].values,
                'year': test_df['year'].values,
                'predicted_sp': best_preds,
                'actual_sp': y_test,
            })

            residuals = best_preds - y_test
            rmse = float(np.sqrt(np.mean(residuals ** 2)))

            fold = FoldResult(
                year=pred_year,
                fold_alpha=best_alpha,
                mae=best_mae,
                rmse=rmse,
                n_teams=len(test_df),
                predictions=pred_df,
                coefficients=best_model.coef_.copy(),
                feature_names=feature_names.copy(),
                intercept=float(best_model.intercept_),
            )
            result.folds.append(fold)

            logger.info(f"Fold {pred_year}: MAE={best_mae:.3f}, RMSE={rmse:.3f}, "
                         f"fold_alpha={best_alpha:.1f}, n={len(test_df)}")

        # Select production alpha = lowest avg MAE across all folds
        result.alpha_mae_table = {
            a: float(np.mean(maes)) for a, maes in alpha_fold_maes.items() if maes
        }
        if result.alpha_mae_table:
            result.production_alpha = min(
                result.alpha_mae_table, key=result.alpha_mae_table.get
            )
            logger.info(f"Production alpha: {result.production_alpha:.1f} "
                         f"(avg MAE={result.alpha_mae_table[result.production_alpha]:.3f})")
            for a, m in sorted(result.alpha_mae_table.items()):
                logger.info(f"  alpha={a:.1f}: avg MAE={m:.3f}")

        # Re-run all folds with production_alpha for consistent tau calibration
        # and production_predictions (used for calibration spread scale matching)
        if result.folds and result.production_alpha > 0:
            from src.win_totals.schedule import calibrate_tau

            all_production_residuals = []
            all_production_pred_dfs = []
            feature_names = result.folds[0].feature_names

            for pred_year, train_df, test_df in fold_splits:
                preds, _, _, _, y_test = self._fit_and_predict_fold(
                    train_df, test_df, feature_names, result.production_alpha
                )
                all_production_residuals.append(preds - y_test)
                all_production_pred_dfs.append(pd.DataFrame({
                    'team': test_df['team'].values,
                    'year': test_df['year'].values,
                    'predicted_sp': preds,
                    'actual_sp': y_test,
                }))

            result.production_residuals = np.concatenate(all_production_residuals)
            result._production_predictions = pd.concat(
                all_production_pred_dfs, ignore_index=True
            )
            result.tau = calibrate_tau(result.production_residuals)

            # Retrain final production model on full dataset
            self.train(dataset, alpha=result.production_alpha)

            logger.info(f"Production model retrained on full dataset "
                         f"(alpha={result.production_alpha:.1f}, n={len(dataset)})")

        logger.info(f"Walk-forward complete: MAE={result.production_mae:.3f}, "
                     f"RMSE={result.production_rmse:.3f}, tau={result.tau:.3f}")

        return result

    @staticmethod
    def naive_baseline(dataset: pd.DataFrame) -> float:
        """Naive baseline: prior SP+ predicts current SP+.

        Returns in-sample MAE (no fitted parameters, but uses full dataset).
        """
        residuals = dataset['prior_sp_overall'] - dataset['target_sp']
        return float(np.mean(np.abs(residuals)))

    @staticmethod
    def talent_baseline(dataset: pd.DataFrame) -> float:
        """Talent baseline: talent_composite as sole predictor.

        Returns in-sample MAE (fitted on all years simultaneously).
        Not comparable to walk-forward MAE — use only as a sanity check.
        """
        from sklearn.linear_model import LinearRegression
        talent = dataset['talent_composite']
        target = dataset['target_sp']
        X = talent.values.reshape(-1, 1)
        y = target.values
        valid = ~(np.isnan(X.ravel()) | np.isnan(y))
        lr = LinearRegression()
        lr.fit(X[valid], y[valid])
        preds = lr.predict(X)
        return float(np.mean(np.abs(preds - y)))
