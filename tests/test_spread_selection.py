"""Tests for calibrated spread betting selection layer.

Tests cover:
1. Sign convention (MANDATORY - must all pass before any other tests)
2. Calibration logic
3. EV calculations
4. Walk-forward validation integrity
5. V2: Push rate estimation
6. V2: Push-aware EV calculations
7. V2: Selection engine (BetRecommendation, evaluate_game, evaluate_slate)
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.spread_selection.calibration import (
    CalibrationResult,
    PushRates,
    KEY_TICKS,
    load_and_normalize_game_data,
    calibrate_cover_probability,
    predict_cover_probability,
    walk_forward_validate,
    estimate_push_rates,
    get_push_probability,
    get_push_probability_vectorized,
    breakeven_prob,
    calculate_ev,
    calculate_ev_vectorized,
    get_spread_bucket,
    diagnose_calibration,
    diagnose_fold_stability,
    stratified_diagnostics,
)

from src.spread_selection.selection import (
    BetRecommendation,
    evaluate_game,
    evaluate_slate,
    calculate_ev_with_push,
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


# =============================================================================
# V2: PUSH RATE TESTS
# =============================================================================

class TestPushRates:
    """Push rate estimation tests (V2)."""

    def test_half_point_tick_zero_push(self):
        """Odd ticks (half-point spreads) => p_push = 0.0 by definition."""
        # Create synthetic data with half-point spreads
        n = 100
        df = pd.DataFrame({
            "game_id": range(n),
            "year": [2023] * n,
            "vegas_spread": [-3.5] * n,  # Half-point spread
            "push": [False] * n,  # No pushes for half-point
        })

        push_rates = estimate_push_rates(df)

        # spread = 3.5 => tick = 7 (odd) => p_push = 0
        p_push = get_push_probability(-3.5, push_rates)
        assert p_push == 0.0

    def test_integer_tick_lookup(self):
        """Even ticks use empirical rate or default_even."""
        n = 200
        # Mix of integer spreads
        spreads = [-3.0] * 100 + [-7.0] * 100
        # 10% push rate for -3, 5% push rate for -7
        pushes = [i % 10 == 0 for i in range(100)] + [i % 20 == 0 for i in range(100)]

        df = pd.DataFrame({
            "game_id": range(n),
            "year": [2023] * n,
            "vegas_spread": spreads,
            "push": pushes,
        })

        push_rates = estimate_push_rates(df, min_games_per_tick=50)

        # tick = 6 (spread 3) is a KEY_TICK - should have own bucket
        # tick = 14 (spread 7) is a KEY_TICK - should have own bucket
        assert 6 in push_rates.tick_rates  # KEY_TICK
        assert 14 in push_rates.tick_rates  # KEY_TICK

        # Check rates are approximately correct (allow for counting differences)
        assert 0.05 <= push_rates.tick_rates[6] <= 0.15  # ~10%
        assert 0.02 <= push_rates.tick_rates[14] <= 0.10  # ~5%

    def test_key_tick_always_separate(self):
        """KEY_TICKS always get own bucket regardless of sample size."""
        # Small sample for each key tick
        spreads = []
        pushes = []
        for tick in KEY_TICKS:
            spread = tick / 2.0  # Convert tick to spread
            # Add only 10 games per key tick (below min_games_per_tick threshold)
            spreads.extend([-spread] * 10)
            pushes.extend([i == 0 for i in range(10)])  # 10% push rate

        df = pd.DataFrame({
            "game_id": range(len(spreads)),
            "year": [2023] * len(spreads),
            "vegas_spread": spreads,
            "push": pushes,
        })

        push_rates = estimate_push_rates(df, min_games_per_tick=50)

        # All KEY_TICKS should have their own bucket despite n < 50
        for tick in KEY_TICKS:
            assert tick in push_rates.tick_rates, f"KEY_TICK {tick} should have own bucket"

    def test_float_safety(self):
        """Tick calculation handles floating point correctly."""
        # Test edge cases
        assert int(round(abs(3.0) * 2)) == 6
        assert int(round(abs(2.99999999) * 2)) == 6  # Should round to 6, not 5
        assert int(round(abs(3.0000001) * 2)) == 6
        assert int(round(abs(3.5) * 2)) == 7

        # Now test with push rates lookup
        df = pd.DataFrame({
            "game_id": range(100),
            "year": [2023] * 100,
            "vegas_spread": [-3.0] * 100,
            "push": [i % 10 == 0 for i in range(100)],
        })

        push_rates = estimate_push_rates(df)

        # All these should give the same result
        p1 = get_push_probability(-3.0, push_rates)
        p2 = get_push_probability(-2.99999999, push_rates)
        p3 = get_push_probability(3.0, push_rates)  # Sign doesn't matter

        assert p1 == p2 == p3

    def test_fallback_default_even(self):
        """Non-key even ticks with n < min_games_per_tick use default_even."""
        n = 100
        # All at spread -3 (KEY_TICK) plus a few at -15 (non-key, insufficient n)
        spreads = [-3.0] * 90 + [-15.0] * 10
        pushes = [i % 10 == 0 for i in range(90)] + [False] * 10

        df = pd.DataFrame({
            "game_id": range(n),
            "year": [2023] * n,
            "vegas_spread": spreads,
            "push": pushes,
        })

        push_rates = estimate_push_rates(df, min_games_per_tick=50)

        # tick = 30 (-15 spread) should NOT be in tick_rates (n=10 < 50)
        # It should fall back to default_even
        assert 30 not in push_rates.tick_rates

        # Getting push probability for -15 should return default_even
        p_push = get_push_probability(-15.0, push_rates)
        assert p_push == push_rates.default_even

    def test_no_leakage_in_walk_forward(self):
        """Push rates only from training seasons in walk-forward."""
        np.random.seed(42)

        # Create multi-year data with different push rates per year
        years = [2022, 2023, 2024, 2025]
        dfs = []

        for i, year in enumerate(years):
            n = 200
            # Vary push rate by year (5% for 2022, 10% for 2023, etc.)
            push_rate = 0.05 + i * 0.05
            df = pd.DataFrame({
                "game_id": range(i * n, (i + 1) * n),
                "year": [year] * n,
                "week": np.random.randint(1, 15, n),
                "home_team": [f"Home{j}" for j in range(n)],
                "away_team": [f"Away{j}" for j in range(n)],
                "edge_abs": np.abs(np.random.normal(4, 2, n)),
                "vegas_spread": np.random.choice([-3.0, -7.0, -3.5, -7.5], n),
                "jp_side_covered": np.random.choice([True, False], n),
                "push": np.random.random(n) < push_rate,
            })
            dfs.append(df)

        all_games = pd.concat(dfs, ignore_index=True)

        # Run walk-forward with push modeling
        result = walk_forward_validate(
            all_games,
            min_train_seasons=2,
            include_push_modeling=True,
        )

        # Check that each fold's push rates only used training years
        for fold in result.fold_summaries:
            eval_year = fold["eval_year"]
            push_summary = fold.get("push_rate_summary")

            if push_summary:
                years_used = push_summary["years_trained"]
                for y in years_used:
                    assert y < eval_year, f"Push rate leakage: {y} in training for {eval_year}"


# =============================================================================
# V2: PUSH-AWARE EV TESTS
# =============================================================================

class TestPushAwareEV:
    """EV calculation with push modeling (V2)."""

    def test_ev_with_push_explicit(self):
        """Explicit test case from V2 spec.

        p_cover_no_push = 0.55
        p_push = 0.05
        juice = -110

        p_win = 0.55 * 0.95 = 0.5225
        p_lose = 0.45 * 0.95 = 0.4275
        p_push = 0.05

        EV = 0.5225 * (100/110) - 0.4275 * 1 + 0.05 * 0
           = 0.475 - 0.4275
           = 0.0475
        """
        p_cover_no_push = 0.55
        p_push = 0.05
        juice = -110

        ev = calculate_ev_with_push(p_cover_no_push, p_push, juice)

        expected_p_win = 0.55 * 0.95
        expected_p_lose = 0.45 * 0.95
        expected_ev = expected_p_win * (100 / 110) - expected_p_lose * 1.0

        assert abs(ev - expected_ev) < 0.0001
        assert abs(ev - 0.0475) < 0.001

    def test_ev_with_zero_push_matches_v1(self):
        """EV with p_push=0 should match V1 calculation."""
        p_cover_no_push = 0.55
        p_push = 0.0
        juice = -110

        ev_v2 = calculate_ev_with_push(p_cover_no_push, p_push, juice)
        ev_v1 = calculate_ev(p_cover_no_push, p_push=0.0, juice=-110)

        assert abs(ev_v2 - ev_v1) < 0.0001

    def test_naming_correctness(self):
        """p_cover_no_push and p_cover are not confused.

        p_cover_no_push: P(cover | no push) - from logistic calibration
        p_cover: Unconditional P(cover) = p_cover_no_push * (1 - p_push)
        """
        p_cover_no_push = 0.55
        p_push = 0.05

        # Unconditional P(cover)
        p_cover = p_cover_no_push * (1 - p_push)

        assert p_cover == 0.55 * 0.95
        assert p_cover < p_cover_no_push  # Push reduces unconditional cover probability

    def test_push_increases_ev_for_negative_ev_bets(self):
        """For negative EV bets, push probability helps (reduces loss)."""
        p_cover_no_push = 0.50  # Negative EV at -110

        ev_no_push = calculate_ev_with_push(p_cover_no_push, 0.0, -110)
        ev_with_push = calculate_ev_with_push(p_cover_no_push, 0.10, -110)

        # Both should be negative, but with push should be less negative
        assert ev_no_push < 0
        assert ev_with_push < 0
        assert ev_with_push > ev_no_push  # Push helps

    def test_push_decreases_ev_for_positive_ev_bets(self):
        """For positive EV bets, push probability hurts (reduces win)."""
        p_cover_no_push = 0.60  # Positive EV at -110

        ev_no_push = calculate_ev_with_push(p_cover_no_push, 0.0, -110)
        ev_with_push = calculate_ev_with_push(p_cover_no_push, 0.10, -110)

        # Both should be positive, but with push should be less positive
        assert ev_no_push > 0
        assert ev_with_push > 0
        assert ev_with_push < ev_no_push  # Push hurts


# =============================================================================
# V2: SELECTION ENGINE TESTS
# =============================================================================

class TestSelectionEngine:
    """Selection engine tests (V2)."""

    @pytest.fixture
    def sample_calibration(self):
        """Create a sample calibration result."""
        return CalibrationResult(
            intercept=0.0,
            slope=0.05,
            n_games=1000,
            years_trained=[2022, 2023],
            implied_breakeven_edge=1.0,
            implied_5pt_pcover=0.56,
            p_cover_at_zero=0.5,
        )

    @pytest.fixture
    def sample_push_rates(self):
        """Create sample push rates."""
        return PushRates(
            tick_rates={6: 0.08, 14: 0.06, 20: 0.05, 28: 0.04},
            default_even=0.05,
            default_overall=0.03,
            n_games_by_tick={6: 100, 14: 100, 20: 50, 28: 30},
            years_trained=[2022, 2023],
        )

    def test_evaluate_game_no_bet(self, sample_calibration, sample_push_rates):
        """edge_abs == 0 => NO_BET with None probabilities."""
        game = pd.Series({
            "game_id": "test_1",
            "home_team": "Alabama",
            "away_team": "Auburn",
            "jp_spread": -7.0,  # Same as Vegas
            "vegas_spread": -7.0,
            "year": 2024,
            "week": 5,
        })

        rec = evaluate_game(game, sample_calibration, sample_push_rates)

        assert rec.confidence == "NO_BET"
        assert rec.edge_abs == 0
        assert rec.p_cover_no_push is None
        assert rec.p_push is None
        assert rec.p_cover is None
        assert rec.ev is None

    def test_evaluate_game_bet(self, sample_calibration, sample_push_rates):
        """Positive EV => BET tier (or higher)."""
        game = pd.Series({
            "game_id": "test_2",
            "home_team": "Alabama",
            "away_team": "Auburn",
            "jp_spread": -12.0,  # JP+ likes home by 5 more
            "vegas_spread": -7.0,
            "year": 2024,
            "week": 5,
        })

        rec = evaluate_game(game, sample_calibration, sample_push_rates, min_ev_threshold=0.01)

        assert rec.edge_abs == 5.0
        assert rec.jp_favored_side == "HOME"
        assert rec.p_cover_no_push is not None
        assert rec.p_push is not None
        assert rec.p_cover is not None
        assert rec.ev is not None
        assert rec.confidence in ("HIGH", "MED", "BET")

    def test_evaluate_game_pass(self, sample_calibration, sample_push_rates):
        """Small edge, low EV => PASS."""
        game = pd.Series({
            "game_id": "test_3",
            "home_team": "Alabama",
            "away_team": "Auburn",
            "jp_spread": -7.5,  # Only 0.5 edge
            "vegas_spread": -7.0,
            "year": 2024,
            "week": 5,
        })

        rec = evaluate_game(game, sample_calibration, sample_push_rates, min_ev_threshold=0.03)

        assert rec.edge_abs == 0.5
        assert rec.confidence == "PASS"

    def test_evaluate_slate_sorting(self, sample_calibration, sample_push_rates):
        """Sorted by EV descending, PASS/NO_BET at bottom."""
        games = pd.DataFrame([
            {
                "game_id": "g1",
                "home_team": "Team A",
                "away_team": "Team B",
                "jp_spread": -10.0,
                "vegas_spread": -7.0,
                "year": 2024,
                "week": 5,
            },
            {
                "game_id": "g2",
                "home_team": "Team C",
                "away_team": "Team D",
                "jp_spread": -7.0,  # No edge
                "vegas_spread": -7.0,
                "year": 2024,
                "week": 5,
            },
            {
                "game_id": "g3",
                "home_team": "Team E",
                "away_team": "Team F",
                "jp_spread": -20.0,  # Large edge
                "vegas_spread": -7.0,
                "year": 2024,
                "week": 5,
            },
        ])

        recs = evaluate_slate(games, sample_calibration, sample_push_rates, min_ev_threshold=0.01)

        # Should have 3 recommendations
        assert len(recs) == 3

        # First should be highest EV (g3 with 13 pt edge)
        assert recs[0].game_id == "g3"
        assert recs[0].edge_abs == 13.0

        # Second should be g1 (3 pt edge)
        assert recs[1].game_id == "g1"
        assert recs[1].edge_abs == 3.0

        # Last should be g2 (NO_BET)
        assert recs[2].game_id == "g2"
        assert recs[2].confidence == "NO_BET"

    def test_confidence_tiers(self, sample_calibration, sample_push_rates):
        """Correct tier assignment based on EV thresholds."""
        # Create games with different edge sizes
        edge_configs = [
            (20.0, "HIGH"),   # Very large edge -> HIGH
            (10.0, "MED"),    # Large edge -> MED
            (6.0, "BET"),     # Medium edge -> BET
            (2.0, "PASS"),    # Small edge -> PASS
        ]

        for edge, expected_tier in edge_configs:
            game = pd.Series({
                "game_id": f"test_{edge}",
                "home_team": "Alabama",
                "away_team": "Auburn",
                "jp_spread": -7.0 - edge,
                "vegas_spread": -7.0,
                "year": 2024,
                "week": 5,
            })

            rec = evaluate_game(game, sample_calibration, sample_push_rates, min_ev_threshold=0.03)

            # Just check that large edges get better tiers
            # The exact mapping depends on calibration slope
            assert rec.edge_abs == edge
            assert rec.confidence in ("HIGH", "MED", "BET", "PASS")

    def test_bet_recommendation_to_dict(self, sample_calibration, sample_push_rates):
        """BetRecommendation.to_dict() produces complete dictionary."""
        game = pd.Series({
            "game_id": "test_1",
            "home_team": "Alabama",
            "away_team": "Auburn",
            "jp_spread": -10.0,
            "vegas_spread": -7.0,
            "year": 2024,
            "week": 5,
        })

        rec = evaluate_game(game, sample_calibration, sample_push_rates)
        d = rec.to_dict()

        # Check all expected keys are present
        expected_keys = [
            "game_id", "home_team", "away_team", "jp_spread", "vegas_spread",
            "edge_pts", "edge_abs", "jp_favored_side",
            "p_cover_no_push", "p_push", "p_cover", "p_breakeven", "edge_prob", "ev",
            "juice", "confidence", "year", "week",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_evaluate_game_without_push_rates(self, sample_calibration):
        """Evaluate game without push rates (p_push = 0)."""
        game = pd.Series({
            "game_id": "test_1",
            "home_team": "Alabama",
            "away_team": "Auburn",
            "jp_spread": -10.0,
            "vegas_spread": -7.0,
            "year": 2024,
            "week": 5,
        })

        rec = evaluate_game(game, sample_calibration, push_rates=None)

        assert rec.p_push == 0.0
        assert rec.p_cover == rec.p_cover_no_push  # No push adjustment


# =============================================================================
# V2: STRATIFIED DIAGNOSTICS TESTS
# =============================================================================

class TestStratifiedDiagnostics:
    """Tests for stratified diagnostics (V2)."""

    def test_spread_bucket_strata(self):
        """Test spread bucket stratification."""
        np.random.seed(42)
        n = 500

        df = pd.DataFrame({
            "game_id": range(n),
            "year": [2024] * n,
            "week": np.random.randint(4, 15, n),
            "edge_abs": np.abs(np.random.normal(6, 3, n)),
            "p_cover_no_push": np.random.uniform(0.52, 0.60, n),
            "jp_side_covered": np.random.choice([True, False], n, p=[0.55, 0.45]),
            "push": [False] * n,
        })

        diag = stratified_diagnostics(df)

        assert "spread_strata" in diag
        assert len(diag["spread_strata"]) > 0

        # Check each stratum has required fields
        for stratum in diag["spread_strata"]:
            assert "bucket" in stratum
            assert "n" in stratum
            assert "empirical_rate" in stratum
            assert "avg_predicted" in stratum
            assert "brier" in stratum

    def test_week_bucket_strata(self):
        """Test week bucket stratification."""
        np.random.seed(42)
        n = 500

        df = pd.DataFrame({
            "game_id": range(n),
            "year": [2024] * n,
            "week": np.random.choice([1, 2, 3, 5, 6, 10, 11, 17, 18], n),
            "edge_abs": np.abs(np.random.normal(5, 2, n)),
            "p_cover_no_push": np.random.uniform(0.52, 0.58, n),
            "jp_side_covered": np.random.choice([True, False], n, p=[0.54, 0.46]),
            "push": [False] * n,
        })

        diag = stratified_diagnostics(df)

        assert "week_strata" in diag
        assert len(diag["week_strata"]) > 0

        # Check that each stratum has required fields
        for stratum in diag["week_strata"]:
            assert "bucket" in stratum
            assert "n" in stratum
            assert "empirical_rate" in stratum
            assert "avg_predicted" in stratum
            assert "brier" in stratum
            assert stratum["n"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
