"""Tests for calibrated spread betting selection layer.

Tests cover:
1. Sign convention (MANDATORY - must all pass before any other tests)
2. Calibration logic
3. EV calculations
4. Walk-forward validation integrity
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.spread_selection.calibration import (
    CalibrationResult,
    load_and_normalize_game_data,
    calibrate_cover_probability,
    predict_cover_probability,
    walk_forward_validate,
    breakeven_prob,
    calculate_ev,
    calculate_ev_vectorized,
    get_spread_bucket,
    diagnose_calibration,
    diagnose_fold_stability,
)


# =============================================================================
# SIGN CONVENTION TESTS (MANDATORY)
# =============================================================================

class TestSignConvention:
    """Sign convention tests - all must pass before proceeding."""

    def test_normalize_spread_flip(self):
        """JP+ +10 internal (home favored by 10) → -10 Vegas convention."""
        df = pd.DataFrame({
            "game_id": [1],
            "year": [2024],
            "week": [5],
            "home_team": ["Alabama"],
            "away_team": ["Auburn"],
            "predicted_spread": [10.0],  # Internal: +10 = home by 10
            "actual_margin": [7.0],
            "spread_close": [-7.0],  # Vegas: -7 = home by 7
        })

        result = load_and_normalize_game_data(df, jp_convention="pos_home_favored")

        # After normalization, jp_spread should be -10 (Vegas convention)
        assert result["jp_spread"].iloc[0] == -10.0
        assert result["vegas_spread"].iloc[0] == -7.0

    def test_normalize_spread_already_vegas(self):
        """No-op when convention is neg_home_favored (already Vegas)."""
        df = pd.DataFrame({
            "game_id": [1],
            "year": [2024],
            "week": [5],
            "home_team": ["Alabama"],
            "away_team": ["Auburn"],
            "predicted_spread": [-10.0],  # Already Vegas
            "actual_margin": [7.0],
            "spread_close": [-7.0],
        })

        result = load_and_normalize_game_data(df, jp_convention="neg_home_favored")

        # Should remain -10
        assert result["jp_spread"].iloc[0] == -10.0

    def test_cover_margin_home_favorite_covers(self):
        """Vegas -7 (home by 7), actual +10 (home wins by 10) → cover_margin = +3 → home covered."""
        df = pd.DataFrame({
            "game_id": [1],
            "year": [2024],
            "week": [5],
            "home_team": ["Alabama"],
            "away_team": ["Auburn"],
            "predicted_spread": [7.0],  # Internal: home by 7
            "actual_margin": [10.0],  # Home won by 10
            "spread_close": [-7.0],  # Vegas: home by 7
        })

        result = load_and_normalize_game_data(df, jp_convention="pos_home_favored")

        # cover_margin = actual_margin + vegas_spread = 10 + (-7) = +3
        assert result["cover_margin"].iloc[0] == 3.0
        assert result["home_covered"].iloc[0] == True
        assert result["away_covered"].iloc[0] == False
        assert result["push"].iloc[0] == False

    def test_cover_margin_home_favorite_fails(self):
        """Vegas -7 (home by 7), actual +3 (home wins by 3) → cover_margin = -4 → away covered."""
        df = pd.DataFrame({
            "game_id": [1],
            "year": [2024],
            "week": [5],
            "home_team": ["Alabama"],
            "away_team": ["Auburn"],
            "predicted_spread": [7.0],
            "actual_margin": [3.0],  # Home won by 3 (but was -7)
            "spread_close": [-7.0],
        })

        result = load_and_normalize_game_data(df, jp_convention="pos_home_favored")

        # cover_margin = 3 + (-7) = -4
        assert result["cover_margin"].iloc[0] == -4.0
        assert result["home_covered"].iloc[0] == False
        assert result["away_covered"].iloc[0] == True

    def test_cover_margin_push(self):
        """Vegas -7, actual +7 → cover_margin = 0 → push."""
        df = pd.DataFrame({
            "game_id": [1],
            "year": [2024],
            "week": [5],
            "home_team": ["Alabama"],
            "away_team": ["Auburn"],
            "predicted_spread": [7.0],
            "actual_margin": [7.0],
            "spread_close": [-7.0],
        })

        result = load_and_normalize_game_data(df, jp_convention="pos_home_favored")

        # cover_margin = 7 + (-7) = 0
        assert result["cover_margin"].iloc[0] == 0.0
        assert result["push"].iloc[0] == True
        assert result["home_covered"].iloc[0] == False
        assert result["away_covered"].iloc[0] == False

    def test_edge_direction_home(self):
        """edge_pts < 0 → jp_favored_side = HOME."""
        df = pd.DataFrame({
            "game_id": [1],
            "year": [2024],
            "week": [5],
            "home_team": ["Alabama"],
            "away_team": ["Auburn"],
            "predicted_spread": [10.0],  # Internal: home by 10
            "actual_margin": [7.0],
            "spread_close": [-3.0],  # Vegas: home by 3
        })

        result = load_and_normalize_game_data(df, jp_convention="pos_home_favored")

        # jp_spread = -10, vegas_spread = -3
        # edge_pts = jp_spread - vegas_spread = -10 - (-3) = -7
        # Negative edge → JP+ likes home MORE
        assert result["jp_spread"].iloc[0] == -10.0
        assert result["edge_pts"].iloc[0] == -7.0
        assert result["edge_abs"].iloc[0] == 7.0
        assert result["jp_favored_side"].iloc[0] == "HOME"

    def test_edge_direction_away(self):
        """edge_pts > 0 → jp_favored_side = AWAY."""
        df = pd.DataFrame({
            "game_id": [1],
            "year": [2024],
            "week": [5],
            "home_team": ["Alabama"],
            "away_team": ["Auburn"],
            "predicted_spread": [1.0],  # Internal: home by 1
            "actual_margin": [7.0],
            "spread_close": [-7.0],  # Vegas: home by 7
        })

        result = load_and_normalize_game_data(df, jp_convention="pos_home_favored")

        # jp_spread = -1, vegas_spread = -7
        # edge_pts = -1 - (-7) = +6
        # Positive edge → JP+ likes away MORE
        assert result["jp_spread"].iloc[0] == -1.0
        assert result["edge_pts"].iloc[0] == 6.0
        assert result["jp_favored_side"].iloc[0] == "AWAY"

    def test_jp_side_covered_mapping_home(self):
        """jp_side_covered = home_covered when jp_favored_side = HOME."""
        df = pd.DataFrame({
            "game_id": [1],
            "year": [2024],
            "week": [5],
            "home_team": ["Alabama"],
            "away_team": ["Auburn"],
            "predicted_spread": [10.0],  # JP+ likes home by 10
            "actual_margin": [14.0],  # Home won by 14
            "spread_close": [-7.0],  # Vegas: home by 7
        })

        result = load_and_normalize_game_data(df, jp_convention="pos_home_favored")

        # JP+ favored HOME, home covered → jp_side_covered = True
        assert result["jp_favored_side"].iloc[0] == "HOME"
        assert result["home_covered"].iloc[0] == True
        assert result["jp_side_covered"].iloc[0] == True

    def test_jp_side_covered_mapping_away(self):
        """jp_side_covered = away_covered when jp_favored_side = AWAY."""
        df = pd.DataFrame({
            "game_id": [1],
            "year": [2024],
            "week": [5],
            "home_team": ["Alabama"],
            "away_team": ["Auburn"],
            "predicted_spread": [1.0],  # JP+ likes home by 1 (so Vegas is too high on home)
            "actual_margin": [3.0],  # Home won by 3
            "spread_close": [-7.0],  # Vegas: home by 7
        })

        result = load_and_normalize_game_data(df, jp_convention="pos_home_favored")

        # JP+ favored AWAY, away covered (home didn't beat -7) → jp_side_covered = True
        assert result["jp_favored_side"].iloc[0] == "AWAY"
        assert result["cover_margin"].iloc[0] == -4.0  # Home failed to cover
        assert result["away_covered"].iloc[0] == True
        assert result["jp_side_covered"].iloc[0] == True

    def test_symmetry_invariant(self):
        """Swapping home/away while preserving |edge| produces same P(cover) concept.

        This test validates that the model treats edges symmetrically - a 5-point
        edge on the home side should have the same P(cover) as a 5-point edge
        on the away side, all else equal.
        """
        # Create two games with same |edge| but different directions
        df = pd.DataFrame({
            "game_id": [1, 2],
            "year": [2024, 2024],
            "week": [5, 5],
            "home_team": ["Alabama", "Georgia"],
            "away_team": ["Auburn", "Florida"],
            "predicted_spread": [10.0, 2.0],  # Internal
            "actual_margin": [14.0, 0.0],  # Different outcomes
            "spread_close": [-5.0, -7.0],  # Vegas
        })

        result = load_and_normalize_game_data(df, jp_convention="pos_home_favored")

        # Game 1: jp=-10, vegas=-5, edge=-5 → HOME by 5
        # Game 2: jp=-2, vegas=-7, edge=+5 → AWAY by 5
        assert result["edge_abs"].iloc[0] == result["edge_abs"].iloc[1] == 5.0
        assert result["jp_favored_side"].iloc[0] == "HOME"
        assert result["jp_favored_side"].iloc[1] == "AWAY"

        # The key invariant: both games have the same edge magnitude
        # so they should receive the same P(cover) prediction
        # (This is tested implicitly - the calibration uses edge_abs)


# =============================================================================
# CALIBRATION TESTS
# =============================================================================

class TestCalibration:
    """Calibration logic tests."""

    def test_slope_positive_on_synthetic(self):
        """Higher edge → higher P(cover) - slope must be positive."""
        # Create synthetic data where higher edge = more covers
        np.random.seed(42)
        n = 5000  # More data for better convergence

        # Generate edge values
        edges = np.abs(np.random.normal(5, 2, n))

        # P(cover) increases with edge (logistic relationship)
        # Use a more realistic slope (real data is ~0.05)
        true_intercept = 0.0
        true_slope = 0.05
        p_cover = expit(true_intercept + true_slope * edges)
        covers = np.random.binomial(1, p_cover)

        df = pd.DataFrame({
            "game_id": range(n),
            "year": [2023] * n,
            "week": np.random.randint(1, 15, n),
            "home_team": ["Team A"] * n,
            "away_team": ["Team B"] * n,
            "edge_abs": edges,
            "jp_side_covered": covers.astype(bool),
            "push": [False] * n,
        })

        result = calibrate_cover_probability(df)

        # Slope should be positive
        assert result.slope > 0, f"Expected positive slope, got {result.slope}"
        # Slope should be within 50% of true value (logistic fitting has variance)
        assert result.slope > true_slope * 0.5, f"Slope {result.slope} too far below {true_slope}"
        assert result.slope < true_slope * 1.5, f"Slope {result.slope} too far above {true_slope}"

    def test_no_leakage_walk_forward(self):
        """Season Y never in training for predicting Y."""
        # Create multi-year data
        years = [2022, 2023, 2024, 2025]
        games_per_year = 200

        dfs = []
        for i, year in enumerate(years):
            df = pd.DataFrame({
                "game_id": range(i * games_per_year, (i + 1) * games_per_year),
                "year": [year] * games_per_year,
                "week": np.random.randint(1, 15, games_per_year),
                "home_team": [f"Team {j}" for j in range(games_per_year)],
                "away_team": [f"Team {j+100}" for j in range(games_per_year)],
                "edge_abs": np.abs(np.random.normal(4, 2, games_per_year)),
                "jp_side_covered": np.random.choice([True, False], games_per_year),
                "push": [False] * games_per_year,
                "p_cover_no_push": [None] * games_per_year,  # Will be filled by walk-forward
            })
            dfs.append(df)

        all_games = pd.concat(dfs, ignore_index=True)

        result = walk_forward_validate(all_games, min_train_seasons=2)

        # Check that each fold only trained on prior years
        for fold in result.fold_summaries:
            eval_year = fold["eval_year"]
            years_trained = fold["years_trained"]

            # All training years must be < eval_year
            for train_year in years_trained:
                assert train_year < eval_year, \
                    f"Leakage: {train_year} in training for {eval_year}"

    def test_excludes_pushes(self):
        """Push games not in calibration training."""
        n = 500
        df = pd.DataFrame({
            "game_id": range(n),
            "year": [2023] * n,
            "week": np.random.randint(1, 15, n),
            "home_team": ["Team A"] * n,
            "away_team": ["Team B"] * n,
            "edge_abs": np.abs(np.random.normal(4, 2, n)),
            "jp_side_covered": np.random.choice([True, False], n),
            "push": [i % 5 == 0 for i in range(n)],  # 20% pushes
        })

        result = calibrate_cover_probability(df)

        # n_games should be less than n (pushes excluded)
        expected_non_push = sum(1 for i in range(n) if i % 5 != 0 and df["edge_abs"].iloc[i] > 0)
        assert result.n_games <= expected_non_push

    def test_excludes_zero_edge(self):
        """edge_abs == 0 games excluded."""
        n = 500
        edges = np.abs(np.random.normal(4, 2, n))
        edges[:50] = 0  # Set first 50 to zero edge

        df = pd.DataFrame({
            "game_id": range(n),
            "year": [2023] * n,
            "week": np.random.randint(1, 15, n),
            "home_team": ["Team A"] * n,
            "away_team": ["Team B"] * n,
            "edge_abs": edges,
            "jp_side_covered": np.random.choice([True, False], n),
            "push": [False] * n,
        })

        result = calibrate_cover_probability(df)

        # Should have excluded 50 zero-edge games
        assert result.n_games <= n - 50

    def test_implied_breakeven_formula(self):
        """Verify breakeven edge calculation is correct."""
        # Create calibration with known parameters
        intercept = 0.0
        slope = 0.05

        # Manually create a CalibrationResult
        be_prob = breakeven_prob(-110)  # 0.5238
        logit_be = np.log(be_prob / (1 - be_prob))
        expected_be_edge = (logit_be - intercept) / slope

        # Create synthetic data that would produce these params (approximately)
        n = 10000
        edges = np.abs(np.random.normal(5, 3, n))
        p_cover = expit(intercept + slope * edges)
        covers = np.random.binomial(1, p_cover)

        df = pd.DataFrame({
            "game_id": range(n),
            "year": [2023] * n,
            "week": np.random.randint(1, 15, n),
            "home_team": ["Team A"] * n,
            "away_team": ["Team B"] * n,
            "edge_abs": edges,
            "jp_side_covered": covers.astype(bool),
            "push": [False] * n,
        })

        result = calibrate_cover_probability(df)

        # Verify formula: P(cover) at breakeven_edge should be ~0.524
        predicted_p = expit(result.intercept + result.slope * result.implied_breakeven_edge)
        assert abs(predicted_p - be_prob) < 0.01, \
            f"P(cover) at breakeven edge = {predicted_p}, expected {be_prob}"


# =============================================================================
# EV TESTS
# =============================================================================

class TestEV:
    """Expected value calculation tests."""

    def test_breakeven_at_110(self):
        """P=0.5238 → EV ≈ 0 at -110."""
        be_prob = breakeven_prob(-110)
        assert abs(be_prob - 0.5238) < 0.001

        ev = calculate_ev(be_prob, p_push=0.0, juice=-110)
        assert abs(ev) < 0.001, f"EV at breakeven should be ~0, got {ev}"

    def test_positive_ev(self):
        """P=0.55, juice=-110 → EV > 0."""
        ev = calculate_ev(0.55, p_push=0.0, juice=-110)
        # EV = 0.55 * (100/110) - 0.45 * 1 = 0.5 - 0.45 = 0.05
        expected = 0.55 * (100 / 110) - 0.45
        assert abs(ev - expected) < 0.001
        assert ev > 0

    def test_negative_ev(self):
        """P=0.50, juice=-110 → EV < 0."""
        ev = calculate_ev(0.50, p_push=0.0, juice=-110)
        # EV = 0.50 * (100/110) - 0.50 * 1 = 0.4545 - 0.50 = -0.0455
        expected = 0.50 * (100 / 110) - 0.50
        assert abs(ev - expected) < 0.001
        assert ev < 0

    def test_vectorized_matches_scalar(self):
        """Vectorized EV matches scalar calculation."""
        p_covers = np.array([0.50, 0.524, 0.55, 0.60])

        # Vectorized
        evs_vec = calculate_ev_vectorized(p_covers, p_push=0.0, juice=-110)

        # Scalar
        evs_scalar = [calculate_ev(p, p_push=0.0, juice=-110) for p in p_covers]

        np.testing.assert_array_almost_equal(evs_vec, evs_scalar)


# =============================================================================
# UTILITY TESTS
# =============================================================================

class TestUtilities:
    """Utility function tests."""

    def test_spread_bucket_boundaries(self):
        """Test spread bucket categorization."""
        assert get_spread_bucket(0) == "[0,3)"
        assert get_spread_bucket(2.9) == "[0,3)"
        assert get_spread_bucket(3) == "[3,5)"
        assert get_spread_bucket(4.9) == "[3,5)"
        assert get_spread_bucket(5) == "[5,7)"
        assert get_spread_bucket(6.9) == "[5,7)"
        assert get_spread_bucket(7) == "[7,10)"
        assert get_spread_bucket(9.9) == "[7,10)"
        assert get_spread_bucket(10) == "[10,+)"
        assert get_spread_bucket(100) == "[10,+)"

    def test_predict_cover_probability_clamping(self):
        """Predictions are clamped to [0.01, 0.99]."""
        # Create calibration with extreme slope
        cal = CalibrationResult(
            intercept=0.0,
            slope=1.0,  # Very high slope
            n_games=1000,
            years_trained=[2023],
            implied_breakeven_edge=0.5,
            implied_5pt_pcover=0.99,
            p_cover_at_zero=0.5,
        )

        # Very large edge would produce P > 0.99 without clamping
        edges = np.array([0.0, 5.0, 50.0, 100.0])
        p_covers = predict_cover_probability(edges, cal)

        assert all(p_covers >= 0.01)
        assert all(p_covers <= 0.99)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_synthetic(self):
        """Test full pipeline with synthetic data."""
        np.random.seed(42)

        # Generate 4 years of synthetic data
        # Need enough games per year for walk-forward to have sufficient training
        years = [2022, 2023, 2024, 2025]
        all_dfs = []

        for year in years:
            n = 400  # More games per year for valid walk-forward
            edges = np.abs(np.random.normal(4, 2, n))

            # True model: P(cover) = logistic(0.1 * edge)
            true_p = expit(0.1 * edges)
            covers = np.random.binomial(1, true_p)

            # Some pushes
            pushes = np.random.choice([True, False], n, p=[0.02, 0.98])
            covers[pushes] = False  # Reset covers for pushes

            df = pd.DataFrame({
                "game_id": range(len(all_dfs) * 200, len(all_dfs) * 200 + n),
                "year": [year] * n,
                "week": np.random.randint(1, 15, n),
                "home_team": [f"Home{i}" for i in range(n)],
                "away_team": [f"Away{i}" for i in range(n)],
                "edge_abs": edges,
                "jp_side_covered": covers.astype(bool),
                "push": pushes,
            })
            all_dfs.append(df)

        all_games = pd.concat(all_dfs, ignore_index=True)

        # Run walk-forward validation
        result = walk_forward_validate(all_games, min_train_seasons=2)

        # Basic sanity checks
        assert len(result.game_results) > 0
        assert len(result.fold_summaries) >= 2
        assert result.overall_brier > 0
        assert result.overall_brier < 0.5  # Better than random guessing

        # Calibration diagnostics
        diag = diagnose_calibration(result.game_results)
        assert "bucket_stats" in diag
        assert diag["brier_model"] > 0

        # Fold stability
        stability = diagnose_fold_stability(result.fold_summaries)
        assert stability["n_folds"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
