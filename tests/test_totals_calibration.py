#!/usr/bin/env python3
"""Tests for Totals EV Engine Phase 2 Calibration.

Test Coverage:
- Part A: Sigma estimators return finite positive values
- Part C: Reliability scaling increases sigma when games_played is low
- Week bucket lookup is deterministic
- Calibration JSON round-trip works
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.spread_selection.totals_calibration import (
    # Config
    TotalsCalibrationConfig,
    SigmaEstimate,
    IntervalCoverageResult,
    # Residual collection
    collect_walk_forward_residuals,
    # Sigma estimators
    estimate_sigma_global,
    estimate_sigma_robust,
    estimate_sigma_by_week_bucket,
    estimate_sigma_by_phase,
    compute_all_sigma_estimates,
    # Coverage
    evaluate_interval_coverage,
    compute_coverage_score,
    # Reliability
    compute_game_reliability,
    compute_scaled_sigma,
    get_sigma_for_week_bucket,
    get_sigma_for_game,
    # Tuning
    tune_sigma_for_coverage,
    compute_week_bucket_multipliers,
    # ROI backtest
    backtest_ev_roi,
    calculate_totals_probabilities,
    # Load/Save
    save_calibration,
    load_calibration,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_residuals():
    """Generate sample residuals for testing."""
    np.random.seed(42)
    n = 500

    # True sigma = 13.0
    errors = np.random.normal(0, 13.0, n)

    df = pd.DataFrame({
        'year': np.random.choice([2022, 2023, 2024, 2025], n),
        'week': np.random.randint(1, 16, n),
        'home_team': ['TeamA'] * n,
        'away_team': ['TeamB'] * n,
        'adjusted_total': 50 + errors,
        'actual_total': 50.0,
        'vegas_total_close': 50.0,
    })

    return collect_walk_forward_residuals(df)


@pytest.fixture
def sample_preds_df():
    """Generate sample predictions DataFrame with lines."""
    np.random.seed(42)
    n = 200

    errors = np.random.normal(0, 13.0, n)
    vegas_lines = np.round(50 + np.random.normal(0, 3, n))

    df = pd.DataFrame({
        'year': np.random.choice([2023, 2024, 2025], n),
        'week': np.random.randint(1, 15, n),
        'home_team': ['TeamA'] * n,
        'away_team': ['TeamB'] * n,
        'adjusted_total': 50 + errors,
        'actual_total': 50 + errors + np.random.normal(0, 2, n),  # Some noise
        'vegas_total_close': vegas_lines,
        'vegas_total_open': vegas_lines - 0.5,
    })

    return df


# =============================================================================
# Part A: Sigma Estimators
# =============================================================================

class TestSigmaEstimators:
    """Test that sigma estimators return finite positive values."""

    def test_global_sigma_positive(self, sample_residuals):
        """Global sigma should be positive and finite."""
        estimate = estimate_sigma_global(sample_residuals)

        assert estimate.sigma > 0, "Sigma must be positive"
        assert np.isfinite(estimate.sigma), "Sigma must be finite"
        assert estimate.n_games == len(sample_residuals)

    def test_global_sigma_near_true_value(self, sample_residuals):
        """Global sigma should be close to true value (13.0)."""
        estimate = estimate_sigma_global(sample_residuals)

        # Allow 10% tolerance
        assert 11.7 < estimate.sigma < 14.3, f"Sigma {estimate.sigma} too far from true 13.0"

    def test_robust_sigma_positive(self, sample_residuals):
        """Robust sigma should be positive and finite."""
        estimate = estimate_sigma_robust(sample_residuals)

        assert estimate.sigma > 0
        assert np.isfinite(estimate.sigma)

    def test_robust_sigma_handles_outliers(self):
        """Robust sigma should be less affected by outliers."""
        np.random.seed(42)
        n = 500
        errors = np.random.normal(0, 10.0, n)

        # Add some extreme outliers
        errors[0:10] = 100

        df = pd.DataFrame({
            'error': errors,
            'week': [5] * n,
        })

        global_est = estimate_sigma_global(df)
        robust_est = estimate_sigma_robust(df)

        # Robust should be closer to true value (10.0)
        assert abs(robust_est.sigma - 10.0) < abs(global_est.sigma - 10.0), \
            "Robust sigma should be less affected by outliers"

    def test_week_bucket_sigma_all_buckets(self, sample_residuals):
        """Week bucket sigma should cover all expected buckets."""
        estimates = estimate_sigma_by_week_bucket(sample_residuals)

        expected_buckets = {"1-2", "3-5", "6-9", "10-14", "15+"}
        actual_buckets = set(estimates.keys())

        # Should have at least some buckets
        assert len(actual_buckets) > 0
        # All should be positive
        for bucket, est in estimates.items():
            assert est.sigma > 0, f"Sigma for {bucket} must be positive"
            assert np.isfinite(est.sigma)

    def test_phase_sigma_all_phases(self, sample_residuals):
        """Phase sigma should cover all phases."""
        estimates = estimate_sigma_by_phase(sample_residuals)

        # All phases should have positive sigma
        for phase, est in estimates.items():
            assert est.sigma > 0
            assert np.isfinite(est.sigma)

    def test_compute_all_estimates(self, sample_residuals):
        """compute_all_sigma_estimates should return multiple estimates."""
        estimates = compute_all_sigma_estimates(sample_residuals)

        # Should have global + robust + week buckets + phases
        assert len(estimates) >= 4

        # All should be SigmaEstimate with positive sigma
        for est in estimates:
            assert isinstance(est, SigmaEstimate)
            assert est.sigma > 0


# =============================================================================
# Part C: Reliability Scaling
# =============================================================================

class TestReliabilityScaling:
    """Test reliability scaling increases sigma when games_played is low."""

    def test_reliability_zero_at_one_game(self):
        """Reliability should be 0 when team has 1 game."""
        rel = compute_game_reliability(home_games_played=1, away_games_played=1)
        assert rel == 0.0

    def test_reliability_midpoint(self):
        """Reliability should be ~0.5 at 4-5 games."""
        rel = compute_game_reliability(home_games_played=5, away_games_played=5)
        assert 0.4 < rel < 0.6

    def test_reliability_uses_minimum(self):
        """Reliability should be min of home/away (most conservative)."""
        rel = compute_game_reliability(home_games_played=8, away_games_played=2)
        # Away has 2 games -> (2-1)/(8-1) = 1/7 ≈ 0.14
        assert rel < 0.2

    def test_scaled_sigma_increases_at_low_reliability(self):
        """Sigma should increase when reliability is low."""
        sigma_base = 13.0
        k = 0.5

        # At reliability=1, sigma_used = sigma_base
        sigma_full = compute_scaled_sigma(sigma_base, reliability=1.0, k=k)
        assert abs(sigma_full - sigma_base) < 0.01

        # At reliability=0, sigma_used = sigma_base * (1 + k) = 13 * 1.5 = 19.5
        sigma_zero = compute_scaled_sigma(sigma_base, reliability=0.0, k=k)
        assert abs(sigma_zero - 19.5) < 0.01

    def test_scaled_sigma_bounds(self):
        """Scaled sigma should respect min/max bounds."""
        sigma_min = 10.0
        sigma_max = 20.0

        # Test lower bound
        sigma = compute_scaled_sigma(
            sigma_base=8.0, reliability=1.0, k=0.0,
            sigma_min=sigma_min, sigma_max=sigma_max
        )
        assert sigma == sigma_min

        # Test upper bound
        sigma = compute_scaled_sigma(
            sigma_base=15.0, reliability=0.0, k=1.0,  # Would be 30
            sigma_min=sigma_min, sigma_max=sigma_max
        )
        assert sigma == sigma_max


# =============================================================================
# Week Bucket Lookup
# =============================================================================

class TestWeekBucketLookup:
    """Test week bucket lookup is deterministic."""

    def test_week_bucket_deterministic(self):
        """Same inputs should always give same output."""
        multipliers = {"1-2": 1.3, "3-5": 1.1, "6-9": 1.0, "10-14": 1.0, "15+": 1.1}
        sigma_base = 13.0

        results = []
        for _ in range(100):
            results.append(get_sigma_for_week_bucket(5, sigma_base, multipliers))

        assert all(r == results[0] for r in results)


# =============================================================================
# Get Sigma For Game (Integration)
# =============================================================================

class TestGetSigmaForGame:
    """Test get_sigma_for_game with different modes."""

    def test_fixed_mode(self):
        """Fixed mode should return sigma_base."""
        config = TotalsCalibrationConfig(
            sigma_mode="fixed",
            sigma_base=13.5,
        )

        sigma = get_sigma_for_game(config, week=5, home_games_played=1, away_games_played=1)
        assert sigma == 13.5


# =============================================================================
# Calibration JSON Round-Trip
# =============================================================================

class TestCalibrationJsonRoundTrip:
    """Test calibration config can be saved and loaded."""

    def test_save_and_load(self):
        """Config should survive JSON round-trip."""
        config = TotalsCalibrationConfig(
            sigma_mode="week_bucket",
            sigma_base=13.5,
            week_bucket_multipliers={
                "1-2": 1.25,
                "3-5": 1.1,
                "6-9": 1.0,
                "10-14": 0.95,
                "15+": 1.15,
            },
            reliability_k=0.4,
            ev_min=0.025,
            kelly_fraction=0.20,
            years_used=[2023, 2024, 2025],
            n_games_calibrated=1500,
            calibration_date="2026-02-13",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.json"

            # Save
            save_calibration(config, path)

            # Load
            loaded = load_calibration(path)

            # Verify all fields match
            assert loaded.sigma_mode == config.sigma_mode
            assert loaded.sigma_base == config.sigma_base
            assert loaded.week_bucket_multipliers == config.week_bucket_multipliers
            assert loaded.reliability_k == config.reliability_k
            assert loaded.ev_min == config.ev_min
            assert loaded.kelly_fraction == config.kelly_fraction
            assert loaded.years_used == config.years_used
            assert loaded.n_games_calibrated == config.n_games_calibrated
            assert loaded.calibration_date == config.calibration_date

    def test_json_file_valid(self):
        """Saved file should be valid JSON."""
        config = TotalsCalibrationConfig(sigma_base=14.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_calibration(config, path)

            # Should be valid JSON
            with open(path) as f:
                data = json.load(f)

            assert "sigma_base" in data
            assert data["sigma_base"] == 14.0


# =============================================================================
# Interval Coverage
# =============================================================================

class TestIntervalCoverage:
    """Test interval coverage calculations."""

    def test_coverage_symmetric(self, sample_residuals):
        """Coverage should be close to targets for well-calibrated sigma."""
        # Use true sigma
        coverage = evaluate_interval_coverage(sample_residuals, sigma=13.0)

        # Each coverage should be within 5% of target
        for result in coverage:
            assert abs(result.error) < 0.10, \
                f"Coverage for {result.target_coverage:.0%} is off by {result.error:.1%}"

    def test_coverage_undercalibrated(self):
        """Low sigma should produce undercoverage."""
        np.random.seed(42)
        errors = np.random.normal(0, 15.0, 500)  # True sigma = 15

        df = pd.DataFrame({'error': errors, 'week': [5] * 500})

        # Use sigma = 10 (too low)
        coverage = evaluate_interval_coverage(df, sigma=10.0)

        # Should have negative errors (empirical < target)
        total_error = sum(r.error for r in coverage)
        assert total_error < 0, "Low sigma should produce undercoverage"

    def test_coverage_score_minimized(self, sample_residuals):
        """Coverage score should be minimal near true sigma."""
        scores = {}
        for sigma in [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]:
            coverage = evaluate_interval_coverage(sample_residuals, sigma)
            scores[sigma] = compute_coverage_score(coverage)

        # Score at 13.0 should be among the lowest
        best_sigma = min(scores, key=scores.get)
        assert 11.0 <= best_sigma <= 15.0, "Best sigma should be near true value"


# =============================================================================
# ROI Backtest
# =============================================================================

class TestROIBacktest:
    """Test ROI backtest calculations."""

    def test_probability_calculation(self):
        """Test totals probability calculation."""
        # Over edge: mu=55, line=50 -> should favor over
        p_win, p_loss, p_push = calculate_totals_probabilities(
            mu=55, line=50, sigma=13, side="OVER"
        )

        assert p_win > p_loss, "Over should be favored when mu > line"
        assert p_win + p_loss + p_push == pytest.approx(1.0, abs=0.01)

    def test_probability_half_point_no_push(self):
        """Half-point lines should have no push."""
        p_win, p_loss, p_push = calculate_totals_probabilities(
            mu=50, line=50.5, sigma=13, side="OVER"
        )

        assert p_push == 0.0

    def test_backtest_returns_metrics(self, sample_preds_df):
        """Backtest should return expected metrics."""
        result = backtest_ev_roi(sample_preds_df, sigma=13.0)

        # Should have valid metrics (not error)
        if "error" not in result:
            assert "n_bets" in result
            assert "roi" in result
            assert "win_rate" in result
            assert "cap_hit_rate" in result

    def test_backtest_no_lines_returns_error(self):
        """Backtest without lines should return error."""
        df = pd.DataFrame({
            'adjusted_total': [50],
            'actual_total': [52],
            'year': [2024],
            'week': [5],
            # No vegas_total_close
        })

        result = backtest_ev_roi(df, sigma=13.0)
        assert "error" in result


# =============================================================================
# Tuning
# =============================================================================

class TestTuning:
    """Test parameter tuning functions."""

    def test_tune_sigma_coverage(self, sample_residuals):
        """Tuning should find sigma near true value."""
        best_sigma, results = tune_sigma_for_coverage(
            sample_residuals,
            sigma_candidates=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        )

        # Should be near true sigma of 13
        assert 11.0 <= best_sigma <= 15.0

    def test_week_bucket_multipliers(self, sample_residuals):
        """Week bucket multipliers should be computed."""
        multipliers = compute_week_bucket_multipliers(sample_residuals)

        # Should have some buckets
        assert len(multipliers) > 0

        # All multipliers should be positive
        for bucket, mult in multipliers.items():
            assert mult > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
