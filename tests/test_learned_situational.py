"""Tests for Learned Situational Adjustment (LSA) model."""

import numpy as np
import pytest

from src.models.learned_situational import (
    LearnedSituationalModel,
    SituationalFeatures,
    compute_situational_residual,
    FEATURE_NAMES,
)


def _make_features(**kwargs) -> SituationalFeatures:
    return SituationalFeatures(**kwargs)


def _make_feature_array(**kwargs) -> np.ndarray:
    return _make_features(**kwargs).to_array()


class TestComputeSituationalResidual:
    """Residual sign/definition correctness."""

    def test_includes_situ_default(self):
        """Default behavior: predicted_spread includes situational."""
        # predicted_spread=10 includes fixed_situational=2 → base=8
        # actual=12 → residual = 12 - 8 = 4
        r = compute_situational_residual(
            actual_margin=12.0,
            predicted_spread=10.0,
            fixed_situational=2.0,
        )
        assert r == pytest.approx(4.0)

    def test_includes_situ_explicit_true(self):
        """Same as default when flag explicitly True."""
        r = compute_situational_residual(
            actual_margin=12.0,
            predicted_spread=10.0,
            fixed_situational=2.0,
            predicted_spread_includes_situational=True,
        )
        assert r == pytest.approx(4.0)

    def test_excludes_situ(self):
        """When predicted_spread is already base (no situational included)."""
        # predicted_spread=8 IS the base → residual = 12 - 8 = 4
        r = compute_situational_residual(
            actual_margin=12.0,
            predicted_spread=8.0,
            fixed_situational=2.0,
            predicted_spread_includes_situational=False,
        )
        assert r == pytest.approx(4.0)

    def test_wrong_flag_produces_bias(self):
        """If caller uses wrong flag, residual is biased by fixed_situational."""
        # Correct (includes=True): base = 10 - 2 = 8, residual = 12 - 8 = 4
        r_correct = compute_situational_residual(12.0, 10.0, 2.0, True)
        # Wrong (includes=False when it does include): base = 10, residual = 12 - 10 = 2
        r_wrong = compute_situational_residual(12.0, 10.0, 2.0, False)
        assert r_correct != r_wrong
        assert r_correct - r_wrong == pytest.approx(2.0)  # Bias = fixed_situational


class TestWalkForwardFiltering:
    """train() must filter out future-week samples."""

    def _build_model_with_samples(self, weeks_and_residuals, prior_data=None):
        """Helper: build model with samples at given weeks."""
        model = LearnedSituationalModel(
            ridge_alpha=300.0,
            min_games=2,  # Low threshold for testing
        )
        model.reset(2024)

        if prior_data:
            model.seed_with_prior_data(prior_data)

        for week, residual in weeks_and_residuals:
            features = _make_features(bye_week_home=1.0)
            model.add_training_game(features, residual, week=week)

        return model

    def test_train_filters_future_weeks(self):
        """Samples with week > max_week should be excluded from training."""
        # Add 200 samples at week 3 (residual=5) and 200 at week 5 (residual=-5)
        samples = [(3, 5.0)] * 200 + [(5, -5.0)] * 200
        model = self._build_model_with_samples(samples)

        # Train with max_week=3: only week 3 samples (residual=5)
        result = model.train(max_week=3)
        assert result is not None
        assert result.n_games == 200  # Only week 3

        # Train with max_week=5: all 400 samples
        result_all = model.train(max_week=5)
        assert result_all is not None
        assert result_all.n_games == 400

    def test_seeded_samples_not_filtered(self):
        """Prior-year data (week=None) is always included regardless of max_week."""
        # 200 prior-year samples (week=None)
        prior = [(_make_feature_array(bye_week_home=1.0), 3.0, 1.0, None)] * 200

        # 10 current-season at week 8 (will be filtered when max_week=3)
        model = self._build_model_with_samples(
            [(8, -10.0)] * 10,
            prior_data=prior,
        )

        result = model.train(max_week=3)
        assert result is not None
        # Should include 200 prior + 0 current-season (week 8 > 3)
        assert result.n_games == 200

    def test_week_none_always_included(self):
        """Games added with week=None are always included."""
        model = LearnedSituationalModel(ridge_alpha=300.0, min_games=2)
        model.reset(2024)

        for _ in range(200):
            model.add_training_game(_make_features(), 1.0, week=None)

        result = model.train(max_week=1)
        assert result is not None
        assert result.n_games == 200


class TestSeedBackwardCompatibility:
    """seed_with_prior_data handles legacy, extended, and V2 formats."""

    def test_legacy_format(self):
        """(features, residual) tuples work."""
        model = LearnedSituationalModel(min_games=2)
        model.reset(2024)
        prior = [(_make_feature_array(), 1.0)] * 200
        model.seed_with_prior_data(prior)
        assert len(model._X_train) == 200
        assert all(w is None for w in model._weeks)  # Legacy → week=None

    def test_extended_format(self):
        """(features, residual, weight, vegas_spread) tuples work."""
        model = LearnedSituationalModel(min_games=2)
        model.reset(2024)
        prior = [(_make_feature_array(), 1.0, 0.8, -7.0)] * 200
        model.seed_with_prior_data(prior)
        assert len(model._X_train) == 200
        assert all(w is None for w in model._weeks)  # No week → None

    def test_v2_format_with_week(self):
        """(features, residual, weight, vegas_spread, week) tuples work."""
        model = LearnedSituationalModel(min_games=2)
        model.reset(2024)
        prior = [(_make_feature_array(), 1.0, 0.8, -7.0, None)] * 200
        model.seed_with_prior_data(prior)
        assert len(model._weeks) == 200
        assert all(w is None for w in model._weeks)


class TestGetTrainingDataRoundTrip:
    """get_training_data returns V2 format that seeds correctly."""

    def test_round_trip(self):
        model = LearnedSituationalModel(min_games=2)
        model.reset(2024)
        model.add_training_game(_make_features(bye_week_home=1.0), 2.0, week=5)
        model.add_training_game(_make_features(letdown_home=1.0), -1.0, week=7)

        data = model.get_training_data()
        assert len(data) == 2
        assert len(data[0]) == 5  # V2 format includes week
        assert data[0][4] == 5
        assert data[1][4] == 7

        # Re-seed into new model
        model2 = LearnedSituationalModel(min_games=2)
        model2.reset(2025)
        model2.seed_with_prior_data(data)
        assert model2._weeks == [5, 7]
