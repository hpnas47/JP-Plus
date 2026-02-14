#!/usr/bin/env python3
"""Validation tests for Totals EV Engine Phase 1 baseline blending and guardrails.

Test Suite:
- Test A: Baseline blend fades out correctly
- Test B: Baseline correction direction matches counterfactual
- Test C: Guardrails enforce stake=0
- Test D: No change when blend disabled
"""

import pytest
from dataclasses import dataclass

from src.spread_selection.totals_ev_engine import (
    TotalsEVConfig,
    TotalsEvent,
    TotalMarket,
    compute_baseline_blend_weight,
    compute_baseline_shift,
    check_guardrails,
    evaluate_totals_markets,
    GUARDRAIL_OK,
    GUARDRAIL_LOW_TRAIN_GAMES,
    GUARDRAIL_BASELINE_OUT_OF_RANGE,
    GUARDRAIL_DIAGNOSTIC_ONLY_FORCED,
)


# =============================================================================
# Mock Model for Testing
# =============================================================================

@dataclass
class MockTotalsPrediction:
    """Mock prediction from TotalsModel."""
    home_team: str
    away_team: str
    predicted_total: float
    home_expected: float
    away_expected: float
    baseline: float
    weather_adjustment: float

    @property
    def adjusted_total(self):
        return self.predicted_total + self.weather_adjustment


class MockTotalsModel:
    """Mock TotalsModel for testing baseline blend."""

    def __init__(self, baseline: float = 26.0, total: float = 50.0):
        self.baseline = baseline
        self._total = total
        self._last_train_game_count = 0
        self.team_ratings = {}

    def predict_total(self, home_team, away_team, weather_adjustment=0.0, year=None):
        return MockTotalsPrediction(
            home_team=home_team,
            away_team=away_team,
            predicted_total=self._total,
            home_expected=self._total / 2,
            away_expected=self._total / 2,
            baseline=self.baseline,
            weather_adjustment=weather_adjustment,
        )


# =============================================================================
# Test A: Baseline Blend Fades Out Correctly
# =============================================================================

class TestBaselineBlendFadeout:
    """Test that baseline blend weight decays from 1.0 to 0.0 as training data grows."""

    def test_zero_games_uses_full_prior(self):
        """With n_train_games=0, weight should be 1.0 (full prior)."""
        w = compute_baseline_blend_weight(n_train_games=0, n0=80.0, mode="rational")
        assert w == 1.0, f"Expected weight=1.0, got {w}"

    def test_n0_games_uses_half_blend(self):
        """With n_train_games=n0, weight should be 0.5 (50/50 blend)."""
        w = compute_baseline_blend_weight(n_train_games=80, n0=80.0, mode="rational")
        assert abs(w - 0.5) < 0.001, f"Expected weight=0.5, got {w}"

    def test_high_games_near_zero_weight(self):
        """With n_train_games >> n0, weight should approach 0."""
        w = compute_baseline_blend_weight(n_train_games=800, n0=80.0, mode="rational")
        assert w < 0.1, f"Expected weight<0.1, got {w}"

        w2 = compute_baseline_blend_weight(n_train_games=8000, n0=80.0, mode="rational")
        assert w2 < 0.01, f"Expected weight<0.01, got {w2}"

    def test_exponential_mode(self):
        """Exponential mode should also decay properly."""
        w0 = compute_baseline_blend_weight(n_train_games=0, n0=80.0, mode="exp")
        assert w0 == 1.0

        w_n0 = compute_baseline_blend_weight(n_train_games=80, n0=80.0, mode="exp")
        assert abs(w_n0 - 0.368) < 0.01  # e^(-1) ≈ 0.368

    def test_baseline_shift_at_zero_games(self):
        """With n=0, shift should be 2*(prior - fit)."""
        config = TotalsEVConfig(
            enable_baseline_blend=True,
            baseline_prior_per_team=26.0,
            baseline_blend_n0=80.0,
            baseline_blend_mode="rational",
        )
        shift = compute_baseline_shift(
            baseline_fit=20.0,  # Low fitted baseline
            baseline_prior=26.0,
            n_train_games=0,
            config=config,
        )
        # w=1.0, baseline_used=26.0, shift=2*(26-20)=12.0
        assert abs(shift - 12.0) < 0.01, f"Expected shift=12.0, got {shift}"

    def test_baseline_shift_at_high_games(self):
        """With n >> n0, shift should approach 0."""
        config = TotalsEVConfig(
            enable_baseline_blend=True,
            baseline_prior_per_team=26.0,
            baseline_blend_n0=80.0,
            baseline_blend_mode="rational",
        )
        shift = compute_baseline_shift(
            baseline_fit=20.0,
            baseline_prior=26.0,
            n_train_games=8000,
            config=config,
        )
        # With n=8000, w≈0.01, shift should be small
        assert abs(shift) < 0.2, f"Expected shift≈0, got {shift}"


# =============================================================================
# Test B: Baseline Correction Direction Matches Counterfactual
# =============================================================================

class TestBaselineCorrectionDirection:
    """Test that correction shifts mu in the right direction."""

    def test_low_baseline_shifts_up(self):
        """When fitted baseline < prior, mu should shift UP."""
        config = TotalsEVConfig(
            enable_baseline_blend=True,
            baseline_prior_per_team=26.0,  # Higher than fit
            baseline_blend_n0=80.0,
        )
        shift = compute_baseline_shift(
            baseline_fit=20.0,  # Low baseline
            baseline_prior=26.0,
            n_train_games=60,
            config=config,
        )
        assert shift > 0, f"Expected positive shift (shift up), got {shift}"

    def test_high_baseline_shifts_down(self):
        """When fitted baseline > prior, mu should shift DOWN."""
        config = TotalsEVConfig(
            enable_baseline_blend=True,
            baseline_prior_per_team=26.0,  # Lower than fit
            baseline_blend_n0=80.0,
        )
        shift = compute_baseline_shift(
            baseline_fit=30.0,  # High baseline
            baseline_prior=26.0,
            n_train_games=60,
            config=config,
        )
        assert shift < 0, f"Expected negative shift (shift down), got {shift}"

    def test_under_bias_flips_with_correction(self):
        """Low baseline should favor UNDERs; correction should flip to balanced."""
        # Model with low baseline (predicts low totals, favors UNDERs)
        model = MockTotalsModel(baseline=20.0, total=45.0)  # Low total

        events = [
            TotalsEvent(
                event_id="test_1",
                home_team="TeamA",
                away_team="TeamB",
                markets=[TotalMarket(book="Test", line=50, odds_over=-110, odds_under=-110)],
            )
        ]

        # Without blend: mu=45 < line=50, should favor UNDER
        config_no_blend = TotalsEVConfig(
            enable_baseline_blend=False,
            auto_diagnostic_if_guardrail_hit=False,
        )
        primary_no, _ = evaluate_totals_markets(model, events, config_no_blend, n_train_games=60)

        # With blend: should shift mu up, changing recommendation
        config_blend = TotalsEVConfig(
            enable_baseline_blend=True,
            baseline_prior_per_team=26.0,
            baseline_blend_n0=80.0,
            auto_diagnostic_if_guardrail_hit=False,
        )
        primary_blend, _ = evaluate_totals_markets(model, events, config_blend, n_train_games=60)

        # Both should have recommendations
        assert len(primary_no) > 0, "Expected recommendation without blend"
        assert len(primary_blend) > 0, "Expected recommendation with blend"

        # Check that mu_used shifted up with blend
        if len(primary_blend) > 0:
            rec = primary_blend.iloc[0]
            assert rec['baseline_shift'] > 0, "Expected positive baseline shift"


# =============================================================================
# Test C: Guardrails Enforce stake=0
# =============================================================================

class TestGuardrailsEnforceStakeZero:
    """Test that guardrails force stake=0 when triggered."""

    def test_low_train_games_triggers_guardrail(self):
        """With n_train_games < min, guardrail should trigger."""
        config = TotalsEVConfig(min_train_games_for_staking=80)

        is_diag, reason = check_guardrails(n_train_games=60, baseline_fit=26.0, config=config)
        assert is_diag is True
        assert reason == GUARDRAIL_LOW_TRAIN_GAMES

    def test_baseline_out_of_range_triggers_guardrail(self):
        """With baseline outside [22, 30], guardrail should trigger."""
        config = TotalsEVConfig(baseline_sanity_min=22.0, baseline_sanity_max=30.0)

        # Baseline too low
        is_diag, reason = check_guardrails(n_train_games=100, baseline_fit=18.0, config=config)
        assert is_diag is True
        assert reason == GUARDRAIL_BASELINE_OUT_OF_RANGE

        # Baseline too high
        is_diag, reason = check_guardrails(n_train_games=100, baseline_fit=35.0, config=config)
        assert is_diag is True
        assert reason == GUARDRAIL_BASELINE_OUT_OF_RANGE

    def test_forced_diagnostic_mode(self):
        """diagnostic_only_mode should always force stake=0."""
        config = TotalsEVConfig(diagnostic_only_mode=True)

        is_diag, reason = check_guardrails(n_train_games=1000, baseline_fit=26.0, config=config)
        assert is_diag is True
        assert reason == GUARDRAIL_DIAGNOSTIC_ONLY_FORCED

    def test_guardrail_disables_staking(self):
        """When guardrail triggers, all stakes should be 0."""
        model = MockTotalsModel(baseline=18.0, total=50.0)  # Bad baseline

        events = [
            TotalsEvent(
                event_id="test_1",
                home_team="TeamA",
                away_team="TeamB",
                markets=[TotalMarket(book="Test", line=45, odds_over=-110, odds_under=-110)],
            )
        ]

        config = TotalsEVConfig(
            baseline_sanity_min=22.0,
            baseline_sanity_max=30.0,
            auto_diagnostic_if_guardrail_hit=True,
        )

        primary, _ = evaluate_totals_markets(model, events, config, n_train_games=100)

        if len(primary) > 0:
            assert all(primary['stake'] == 0), "All stakes should be 0 when guardrail triggers"
            assert all(primary['guardrail_reason'] == GUARDRAIL_BASELINE_OUT_OF_RANGE)

    def test_sufficient_games_passes_guardrail(self):
        """With n_train_games >= min and good baseline, no guardrail."""
        config = TotalsEVConfig(
            min_train_games_for_staking=80,
            baseline_sanity_min=22.0,
            baseline_sanity_max=30.0,
        )

        is_diag, reason = check_guardrails(n_train_games=100, baseline_fit=26.0, config=config)
        assert is_diag is False
        assert reason == GUARDRAIL_OK


# =============================================================================
# Test D: No Change When Blend Disabled
# =============================================================================

class TestBlendDisabled:
    """Test that disabling blend produces identical results to original behavior."""

    def test_shift_is_zero_when_disabled(self):
        """With enable_baseline_blend=False, shift should be 0."""
        config = TotalsEVConfig(enable_baseline_blend=False)

        shift = compute_baseline_shift(
            baseline_fit=20.0,  # Would normally produce large shift
            baseline_prior=26.0,
            n_train_games=0,
            config=config,
        )
        assert shift == 0.0, f"Expected shift=0 when disabled, got {shift}"

    def test_mu_unchanged_when_disabled(self):
        """With blend disabled, mu_used should equal mu_raw."""
        model = MockTotalsModel(baseline=20.0, total=45.0)

        events = [
            TotalsEvent(
                event_id="test_1",
                home_team="TeamA",
                away_team="TeamB",
                markets=[TotalMarket(book="Test", line=50, odds_over=-110, odds_under=-110)],
            )
        ]

        config = TotalsEVConfig(
            enable_baseline_blend=False,
            auto_diagnostic_if_guardrail_hit=False,
        )

        primary, _ = evaluate_totals_markets(model, events, config, n_train_games=60)

        assert len(primary) > 0, "Expected recommendation"
        rec = primary.iloc[0]
        assert rec['baseline_shift'] == 0.0, "Shift should be 0 when disabled"
        assert rec['mu_raw'] == rec['mu_used'], "mu_used should equal mu_raw when disabled"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
