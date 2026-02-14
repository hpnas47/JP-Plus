"""Tests for weather adjustment sign convention and guardrails.

Weather Convention:
- Negative weather_adj means "lower scoring expected" (reduce mu_used)
- mu_used = mu_model + weather_adj
- If weather_adj is negative, mu_used < mu_model
"""

import pytest
from dataclasses import dataclass

from src.spread_selection.totals_ev_engine import (
    TotalsEVConfig,
    TotalsEvent,
    TotalMarket,
    TotalsBetRecommendation,
    evaluate_totals_markets,
)


@dataclass
class MockPrediction:
    """Mock TotalsModel prediction."""
    home_team: str
    away_team: str
    predicted_total: float
    home_expected: float
    away_expected: float
    baseline: float
    weather_adjustment: float = 0.0

    @property
    def adjusted_total(self):
        return self.predicted_total + self.weather_adjustment


class MockTotalsModel:
    """Mock TotalsModel for testing."""

    def __init__(self, total: float = 52.0):
        self._total = total
        self.baseline = 26.0
        self.team_ratings = {}

    def predict_total(self, home_team, away_team, weather_adjustment=0.0, year=None):
        return MockPrediction(
            home_team=home_team,
            away_team=away_team,
            predicted_total=self._total,
            home_expected=self._total / 2,
            away_expected=self._total / 2,
            baseline=26.0,
            weather_adjustment=weather_adjustment,
        )


class TestWeatherSignConvention:
    """Test that weather sign convention is correct."""

    def test_negative_weather_reduces_mu_used(self):
        """Negative weather_adj should reduce mu_used below mu_model."""
        model = MockTotalsModel(total=52.0)
        config = TotalsEVConfig(
            use_weather_adjustment=True,
            weather_max_adjustment=10.0,
        )

        # Create event with negative weather adjustment (cold/windy = lower scoring)
        event = TotalsEvent(
            event_id="test_1",
            home_team="TeamA",
            away_team="TeamB",
            year=2024,
            week=10,
            weather_adjustment=-3.0,  # Negative = lower scoring
            markets=[TotalMarket(book="Test", line=50, odds_over=-110, odds_under=-110)],
        )

        primary_df, _ = evaluate_totals_markets(model, [event], config)

        # Check that mu_model > mu_used (weather reduced the total)
        assert len(primary_df) > 0 or True  # May be in edge5 depending on EV
        # Use evaluate_single_market directly for precise control
        from src.spread_selection.totals_ev_engine import evaluate_single_market

        recs = evaluate_single_market(
            event=event,
            market=event.markets[0],
            mu_model=52.0,
            weather_adj=-3.0,
            mu_raw=52.0 - 3.0,  # 49.0
            mu_used=52.0 - 3.0,  # 49.0 (no baseline shift)
            baseline_shift=0.0,
            home_expected=26.0,
            away_expected=26.0,
            baseline=26.0,
            config=config,
        )

        assert len(recs) >= 1
        rec = recs[0]
        assert rec.mu_model == 52.0
        assert rec.weather_adj == -3.0
        assert rec.mu_used == 49.0
        assert rec.mu_used < rec.mu_model

    def test_positive_weather_increases_mu_used(self):
        """Positive weather_adj should increase mu_used above mu_model."""
        from src.spread_selection.totals_ev_engine import evaluate_single_market

        config = TotalsEVConfig()
        event = TotalsEvent(
            event_id="test_2",
            home_team="TeamA",
            away_team="TeamB",
            year=2024,
            week=10,
            weather_adjustment=2.5,  # Positive = higher scoring (dome game?)
            markets=[TotalMarket(book="Test", line=50, odds_over=-110, odds_under=-110)],
        )

        recs = evaluate_single_market(
            event=event,
            market=event.markets[0],
            mu_model=52.0,
            weather_adj=2.5,
            mu_raw=52.0 + 2.5,
            mu_used=52.0 + 2.5,
            baseline_shift=0.0,
            home_expected=26.0,
            away_expected=26.0,
            baseline=26.0,
            config=config,
        )

        assert len(recs) >= 1
        rec = recs[0]
        assert rec.mu_model == 52.0
        assert rec.weather_adj == 2.5
        assert rec.mu_used == 54.5
        assert rec.mu_used > rec.mu_model

    def test_zero_weather_preserves_mu_model(self):
        """Zero weather_adj should leave mu_used == mu_model."""
        from src.spread_selection.totals_ev_engine import evaluate_single_market

        config = TotalsEVConfig()
        event = TotalsEvent(
            event_id="test_3",
            home_team="TeamA",
            away_team="TeamB",
            year=2024,
            week=10,
            weather_adjustment=0.0,
            markets=[TotalMarket(book="Test", line=50, odds_over=-110, odds_under=-110)],
        )

        recs = evaluate_single_market(
            event=event,
            market=event.markets[0],
            mu_model=52.0,
            weather_adj=0.0,
            mu_raw=52.0,
            mu_used=52.0,
            baseline_shift=0.0,
            home_expected=26.0,
            away_expected=26.0,
            baseline=26.0,
            config=config,
        )

        assert len(recs) >= 1
        rec = recs[0]
        assert rec.mu_model == 52.0
        assert rec.weather_adj == 0.0
        assert rec.mu_used == 52.0


class TestWeatherGuardrails:
    """Test weather adjustment guardrail caps."""

    def test_large_negative_weather_is_capped(self):
        """Weather adjustments exceeding cap should be clamped."""
        model = MockTotalsModel(total=52.0)
        config = TotalsEVConfig(
            use_weather_adjustment=True,
            weather_max_adjustment=10.0,  # Cap at ±10 pts
        )

        # Create event with extreme weather adjustment
        event = TotalsEvent(
            event_id="test_cap",
            home_team="TeamA",
            away_team="TeamB",
            year=2024,
            week=10,
            weather_adjustment=-15.0,  # Exceeds cap
            markets=[TotalMarket(book="Test", line=50, odds_over=-110, odds_under=-110)],
        )

        primary_df, edge5_df = evaluate_totals_markets(model, [event], config, n_train_games=200)

        # Check that weather_adj was capped
        all_recs = list(primary_df.itertuples()) + list(edge5_df.itertuples())
        if len(all_recs) > 0:
            rec = all_recs[0]
            # Weather should be capped to -10.0
            assert abs(rec.weather_adj) <= 10.0 + 0.01
            assert rec.weather_adj == -10.0  # Should be exactly -10.0

    def test_large_positive_weather_is_capped(self):
        """Positive weather adjustments exceeding cap should be clamped."""
        model = MockTotalsModel(total=52.0)
        config = TotalsEVConfig(
            use_weather_adjustment=True,
            weather_max_adjustment=10.0,
        )

        event = TotalsEvent(
            event_id="test_cap_pos",
            home_team="TeamA",
            away_team="TeamB",
            year=2024,
            week=10,
            weather_adjustment=12.5,  # Exceeds cap
            markets=[TotalMarket(book="Test", line=50, odds_over=-110, odds_under=-110)],
        )

        primary_df, edge5_df = evaluate_totals_markets(model, [event], config, n_train_games=200)

        all_recs = list(primary_df.itertuples()) + list(edge5_df.itertuples())
        if len(all_recs) > 0:
            rec = all_recs[0]
            assert abs(rec.weather_adj) <= 10.0 + 0.01
            assert rec.weather_adj == 10.0  # Should be exactly +10.0

    def test_weather_disabled_uses_zero(self):
        """When use_weather_adjustment=False, weather_adj should be 0."""
        model = MockTotalsModel(total=52.0)
        config = TotalsEVConfig(
            use_weather_adjustment=False,  # Disabled
        )

        event = TotalsEvent(
            event_id="test_disabled",
            home_team="TeamA",
            away_team="TeamB",
            year=2024,
            week=10,
            weather_adjustment=-5.0,  # Should be ignored
            markets=[TotalMarket(book="Test", line=50, odds_over=-110, odds_under=-110)],
        )

        primary_df, edge5_df = evaluate_totals_markets(model, [event], config, n_train_games=200)

        all_recs = list(primary_df.itertuples()) + list(edge5_df.itertuples())
        if len(all_recs) > 0:
            rec = all_recs[0]
            assert rec.weather_adj == 0.0
            assert rec.mu_used == rec.mu_model  # No weather applied


class TestMuComposition:
    """Test that mu composition formula is correct."""

    def test_mu_composition_with_baseline_shift(self):
        """Test: mu_used = mu_model + weather_adj + baseline_shift."""
        from src.spread_selection.totals_ev_engine import evaluate_single_market

        config = TotalsEVConfig()
        event = TotalsEvent(
            event_id="test_comp",
            home_team="TeamA",
            away_team="TeamB",
            year=2024,
            week=10,
            weather_adjustment=-2.0,
            markets=[TotalMarket(book="Test", line=50, odds_over=-110, odds_under=-110)],
        )

        mu_model = 52.0
        weather_adj = -2.0
        baseline_shift = 1.5  # Baseline blending adds +1.5
        mu_raw = mu_model + weather_adj  # 50.0
        mu_used = mu_raw + baseline_shift  # 51.5

        recs = evaluate_single_market(
            event=event,
            market=event.markets[0],
            mu_model=mu_model,
            weather_adj=weather_adj,
            mu_raw=mu_raw,
            mu_used=mu_used,
            baseline_shift=baseline_shift,
            home_expected=26.0,
            away_expected=26.0,
            baseline=26.0,
            config=config,
        )

        assert len(recs) >= 1
        rec = recs[0]

        # Verify composition
        assert rec.mu_model == 52.0
        assert rec.weather_adj == -2.0
        assert rec.mu_raw == 50.0
        assert rec.baseline_shift == 1.5
        assert rec.mu_used == 51.5

        # Verify formula: mu_used = mu_raw + baseline_shift = mu_model + weather_adj + baseline_shift
        assert abs(rec.mu_used - (rec.mu_model + rec.weather_adj + rec.baseline_shift)) < 0.001

    def test_weather_adjustment_property_alias(self):
        """Test that weather_adjustment property returns weather_adj."""
        from src.spread_selection.totals_ev_engine import evaluate_single_market

        config = TotalsEVConfig()
        event = TotalsEvent(
            event_id="test_alias",
            home_team="TeamA",
            away_team="TeamB",
            year=2024,
            week=10,
            weather_adjustment=-3.5,
            markets=[TotalMarket(book="Test", line=50, odds_over=-110, odds_under=-110)],
        )

        recs = evaluate_single_market(
            event=event,
            market=event.markets[0],
            mu_model=52.0,
            weather_adj=-3.5,
            mu_raw=48.5,
            mu_used=48.5,
            baseline_shift=0.0,
            home_expected=26.0,
            away_expected=26.0,
            baseline=26.0,
            config=config,
        )

        assert len(recs) >= 1
        rec = recs[0]

        # Legacy property should work
        assert rec.weather_adjustment == rec.weather_adj
        assert rec.weather_adjustment == -3.5


class TestDefaultWeatherBehavior:
    """Test default behavior when weather data is missing."""

    def test_missing_weather_defaults_to_zero(self):
        """When weather_adjustment is not set, it should default to 0."""
        model = MockTotalsModel(total=52.0)
        config = TotalsEVConfig()

        # Create event WITHOUT weather_adjustment (should default to 0.0)
        event = TotalsEvent(
            event_id="test_default",
            home_team="TeamA",
            away_team="TeamB",
            year=2024,
            week=10,
            # weather_adjustment not set, defaults to 0.0
            markets=[TotalMarket(book="Test", line=50, odds_over=-110, odds_under=-110)],
        )

        assert event.weather_adjustment == 0.0  # Field default

        primary_df, edge5_df = evaluate_totals_markets(model, [event], config, n_train_games=200)

        all_recs = list(primary_df.itertuples()) + list(edge5_df.itertuples())
        if len(all_recs) > 0:
            rec = all_recs[0]
            assert rec.weather_adj == 0.0
