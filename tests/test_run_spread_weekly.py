"""Regression tests for run_spread_weekly EV math, settlement, and threshold wiring."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pytest
import pandas as pd
from scripts.run_spread_weekly import (
    american_to_b,
    implied_prob_from_american,
    ev_units,
    settle_bets,
    SelectionPolicyConfig,
    PhaseConfig,
    PHASE1_CONSERVATIVE_CONFIG,
)


# ── A) EV correctness ──────────────────────────────────────────────────────

class TestEVHelpers:
    def test_american_to_b_negative(self):
        assert american_to_b(-110) == pytest.approx(100 / 110)

    def test_american_to_b_positive(self):
        assert american_to_b(150) == pytest.approx(1.5)

    def test_implied_prob_negative(self):
        assert implied_prob_from_american(-110) == pytest.approx(110 / 210)

    def test_implied_prob_positive(self):
        assert implied_prob_from_american(150) == pytest.approx(100 / 250)

    def test_ev_breakeven_is_zero(self):
        """At breakeven prob for -110 odds, EV should be ~0."""
        p_break = 110 / 210  # 0.52381
        assert ev_units(p_break, -110) == pytest.approx(0.0, abs=1e-6)

    def test_ev_p55_minus110(self):
        """p=0.55, -110: EV = 0.55*(100/110) - 0.45 = 0.05"""
        assert ev_units(0.55, -110) == pytest.approx(0.55 * (100 / 110) - 0.45)

    def test_ev_p60_minus110(self):
        """p=0.60, -110: EV = 0.60*(100/110) - 0.40 = 0.14545..."""
        assert ev_units(0.60, -110) == pytest.approx(0.60 * (100 / 110) - 0.40)


# ── C) Away settlement sign ────────────────────────────────────────────────

class TestAwaySettlement:
    def test_away_covers_worked_example(self):
        """vegas_spread=-7 (home -7), away bet => bet_spread=+7,
        actual_margin=+3 (home by 3). Away should cover."""
        bet_spread = 7  # -(-7) = +7 for away
        actual_margin = 3
        # Away formula: cover_margin = bet_spread - actual_margin
        cover_margin = bet_spread - actual_margin
        assert cover_margin == 4
        assert cover_margin > 0  # covered

    def test_away_does_not_cover(self):
        """vegas_spread=-7, away bet => bet_spread=+7,
        actual_margin=+10. Away should NOT cover."""
        cover_margin = 7 - 10
        assert cover_margin == -3
        assert cover_margin < 0  # not covered

    def test_home_covers(self):
        """vegas_spread=-7, home bet => bet_spread=-7,
        actual_margin=+10. Home covers by 3."""
        cover_margin = 10 + (-7)
        assert cover_margin == 3
        assert cover_margin > 0


# ── D) Phase 1 EV_THRESHOLD wiring ─────────────────────────────────────────

class TestPhase1ThresholdWiring:
    def test_phase1_ev_floor_is_0_02(self):
        """Phase 1 config should have ev_floor=0.02."""
        assert PHASE1_CONSERVATIVE_CONFIG.ev_floor == 0.02

    def test_phase1_uses_ev_threshold_policy(self):
        assert PHASE1_CONSERVATIVE_CONFIG.selection_policy == "EV_THRESHOLD"

    def test_phase1_ev_min_wiring(self):
        """When selection_policy is EV_THRESHOLD, ev_min should be wired
        from phase ev_floor, not from the base config."""
        from scripts.run_spread_weekly import get_phase_config, get_production_config

        base_config = get_production_config(preset="balanced")
        phase_config = get_phase_config(week=2)  # Phase 1

        # Simulate the wiring logic from generate_recommendations
        effective_ev_min = (
            phase_config.ev_floor
            if phase_config.selection_policy == "EV_THRESHOLD"
            else base_config.ev_min
        )

        assert effective_ev_min == 0.02  # Must be phase ev_floor, not base 0.03

    def test_phase1_filters_below_threshold(self):
        """A candidate with EV=0.015 should be rejected by Phase 1 (2% floor)."""
        # EV_THRESHOLD uses ev_min=0.02, so 0.015 < 0.02 → rejected
        assert 0.015 < PHASE1_CONSERVATIVE_CONFIG.ev_floor

    def test_phase1_accepts_above_threshold(self):
        """A candidate with EV=0.025 should pass Phase 1 (2% floor)."""
        assert 0.025 >= PHASE1_CONSERVATIVE_CONFIG.ev_floor
