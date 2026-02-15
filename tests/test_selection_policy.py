"""Tests for spread selection policy system.

Tests verify:
1. Policy selection behavior (EV_THRESHOLD, TOP_N_PER_WEEK, HYBRID)
2. Deterministic tie-breaking
3. Max bets per week cap
4. Phase 1 skip behavior
5. Edge cases (no candidates, no qualifying bets)
"""

import pytest
import pandas as pd
import numpy as np

from src.spread_selection.selection_policy import (
    SelectionPolicyConfig,
    SelectionPolicy,
    apply_selection_policy,
    compute_selection_metrics,
    compute_stability_score,
    _sort_deterministic,
    config_to_label,
    get_selection_policy_preset,
    configs_match,
    ALLOWED_PRESETS,
    PRESET_CONFIGS,
)


@pytest.fixture
def sample_candidates():
    """Create sample candidate bets DataFrame."""
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"],
        "year": [2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024],
        "week": [4, 4, 4, 4, 5, 5, 5, 5],
        "ev": [0.08, 0.05, 0.03, 0.01, 0.10, 0.04, 0.02, 0.015],
        "edge_abs": [8.0, 5.0, 3.0, 1.0, 10.0, 4.0, 2.0, 1.5],
        "jp_side_covered": [True, True, False, True, True, False, True, False],
        "push": [False, False, False, False, False, False, False, False],
    })


@pytest.fixture
def sample_candidates_with_ties():
    """Create candidates with tied EV values for determinism testing."""
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3", "g4"],
        "year": [2024, 2024, 2024, 2024],
        "week": [4, 4, 4, 4],
        "ev": [0.05, 0.05, 0.05, 0.05],  # All tied
        "edge_abs": [6.0, 8.0, 4.0, 8.0],  # g2 and g4 tie on edge
        "jp_side_covered": [True, True, False, True],
        "push": [False, False, False, False],
    })


class TestSelectionPolicyConfig:
    """Tests for SelectionPolicyConfig validation."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = SelectionPolicyConfig()
        assert config.selection_policy == "EV_THRESHOLD"
        assert config.ev_min == 0.03
        assert config.ev_floor == 0.0
        assert config.top_n_per_week == 10
        assert config.max_bets_per_week == 10
        assert config.phase1_policy == "skip"

    def test_invalid_ev_min_raises(self):
        """Negative ev_min raises ValueError."""
        with pytest.raises(ValueError, match="ev_min must be >= 0"):
            SelectionPolicyConfig(ev_min=-0.01)

    def test_invalid_top_n_raises(self):
        """top_n_per_week < 1 raises ValueError."""
        with pytest.raises(ValueError, match="top_n_per_week must be >= 1"):
            SelectionPolicyConfig(top_n_per_week=0)

    def test_invalid_policy_raises(self):
        """Unknown policy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown policy"):
            SelectionPolicyConfig(selection_policy="INVALID")


class TestDeterministicSort:
    """Tests for deterministic sorting."""

    def test_sort_by_ev_desc(self, sample_candidates):
        """Sorts by EV descending."""
        sorted_df = _sort_deterministic(sample_candidates)
        assert list(sorted_df["ev"]) == sorted(sample_candidates["ev"], reverse=True)

    def test_sort_ties_by_edge(self, sample_candidates_with_ties):
        """When EV ties, sorts by edge_abs descending."""
        sorted_df = _sort_deterministic(sample_candidates_with_ties)
        # All EV = 0.05, so sort by edge: 8.0, 8.0, 6.0, 4.0
        assert sorted_df.iloc[0]["edge_abs"] == 8.0
        assert sorted_df.iloc[1]["edge_abs"] == 8.0
        assert sorted_df.iloc[2]["edge_abs"] == 6.0
        assert sorted_df.iloc[3]["edge_abs"] == 4.0

    def test_sort_ties_by_game_id(self, sample_candidates_with_ties):
        """When EV and edge tie, sorts by game_id ascending."""
        sorted_df = _sort_deterministic(sample_candidates_with_ties)
        # g2 and g4 both have edge=8.0, so g2 < g4 alphabetically
        tied_8 = sorted_df[sorted_df["edge_abs"] == 8.0]
        assert list(tied_8["game_id"]) == ["g2", "g4"]

    def test_empty_df(self):
        """Empty DataFrame returns empty."""
        empty = pd.DataFrame(columns=["game_id", "year", "week", "ev", "edge_abs"])
        result = _sort_deterministic(empty)
        assert len(result) == 0


class TestEVThresholdPolicy:
    """Tests for EV_THRESHOLD selection policy."""

    def test_keeps_above_threshold(self, sample_candidates):
        """Keeps only bets with EV >= ev_min."""
        config = SelectionPolicyConfig(
            selection_policy="EV_THRESHOLD",
            ev_min=0.03,
            max_bets_per_week=10,
        )
        result = apply_selection_policy(sample_candidates, config)

        # EV >= 0.03: 0.08, 0.05, 0.03, 0.10, 0.04 = 5 bets
        assert result.n_selected == 5
        assert all(result.selected_bets["ev"] >= 0.03)

    def test_respects_max_bets_cap(self, sample_candidates):
        """max_bets_per_week limits selections per week."""
        config = SelectionPolicyConfig(
            selection_policy="EV_THRESHOLD",
            ev_min=0.01,  # Very low threshold to get all
            max_bets_per_week=2,
        )
        result = apply_selection_policy(sample_candidates, config)

        # Check each week has at most 2 bets
        weekly = result.selected_bets.groupby("week").size()
        assert all(weekly <= 2)

    def test_takes_highest_ev_when_capped(self, sample_candidates):
        """When capped, keeps highest EV bets."""
        config = SelectionPolicyConfig(
            selection_policy="EV_THRESHOLD",
            ev_min=0.01,
            max_bets_per_week=2,
        )
        result = apply_selection_policy(sample_candidates, config)

        # Week 4 should have g1 (0.08) and g2 (0.05)
        week4 = result.selected_bets[result.selected_bets["week"] == 4]
        assert set(week4["game_id"]) == {"g1", "g2"}


class TestTopNPerWeekPolicy:
    """Tests for TOP_N_PER_WEEK selection policy."""

    def test_takes_top_n(self, sample_candidates):
        """Takes exactly top N per week."""
        config = SelectionPolicyConfig(
            selection_policy="TOP_N_PER_WEEK",
            top_n_per_week=2,
            ev_floor=0.0,
            max_bets_per_week=10,
        )
        result = apply_selection_policy(sample_candidates, config)

        # 2 per week × 2 weeks = 4 bets
        assert result.n_selected == 4

        # Check per week
        weekly = result.selected_bets.groupby("week").size()
        assert weekly[4] == 2
        assert weekly[5] == 2

    def test_respects_ev_floor(self, sample_candidates):
        """ev_floor filters out low EV after taking top N."""
        config = SelectionPolicyConfig(
            selection_policy="TOP_N_PER_WEEK",
            top_n_per_week=3,
            ev_floor=0.03,
            max_bets_per_week=10,
        )
        result = apply_selection_policy(sample_candidates, config)

        # All selected should have EV >= 0.03
        assert all(result.selected_bets["ev"] >= 0.03)

    def test_takes_fewer_if_not_enough(self, sample_candidates):
        """Takes fewer than N if not enough candidates in week."""
        config = SelectionPolicyConfig(
            selection_policy="TOP_N_PER_WEEK",
            top_n_per_week=10,  # More than available
            ev_floor=0.0,
            max_bets_per_week=10,
        )
        result = apply_selection_policy(sample_candidates, config)

        # Should get all 8 (4 per week)
        assert result.n_selected == 8


class TestHybridPolicy:
    """Tests for HYBRID selection policy."""

    def test_filters_before_top_n(self, sample_candidates):
        """HYBRID filters by ev_floor BEFORE taking top N."""
        config = SelectionPolicyConfig(
            selection_policy="HYBRID",
            top_n_per_week=10,
            ev_floor=0.03,
            max_bets_per_week=10,
        )
        result = apply_selection_policy(sample_candidates, config)

        # Only bets with EV >= 0.03
        assert all(result.selected_bets["ev"] >= 0.03)

    def test_returns_zero_when_none_meet_floor(self):
        """Returns zero bets when no candidates meet ev_floor."""
        candidates = pd.DataFrame({
            "game_id": ["g1", "g2"],
            "year": [2024, 2024],
            "week": [4, 4],
            "ev": [0.01, 0.02],  # Both below 0.03 floor
            "edge_abs": [1.0, 2.0],
            "jp_side_covered": [True, False],
            "push": [False, False],
        })

        config = SelectionPolicyConfig(
            selection_policy="HYBRID",
            top_n_per_week=10,
            ev_floor=0.03,
            max_bets_per_week=10,
        )
        result = apply_selection_policy(candidates, config)

        assert result.n_selected == 0
        assert len(result.selected_bets) == 0

    def test_hard_cap_enforced(self, sample_candidates):
        """max_bets_per_week enforced as hard cap."""
        config = SelectionPolicyConfig(
            selection_policy="HYBRID",
            top_n_per_week=10,
            ev_floor=0.01,
            max_bets_per_week=2,
        )
        result = apply_selection_policy(sample_candidates, config)

        # Each week should have at most 2
        weekly = result.selected_bets.groupby("week").size()
        assert all(weekly <= 2)


class TestPhase1Skip:
    """Tests for Phase 1 (weeks 1-3) skip behavior."""

    def test_skips_phase1_by_default(self):
        """phase1_policy='skip' excludes weeks 1-3."""
        candidates = pd.DataFrame({
            "game_id": ["g1", "g2", "g3", "g4"],
            "year": [2024, 2024, 2024, 2024],
            "week": [2, 3, 4, 5],  # 2, 3 are Phase 1
            "ev": [0.10, 0.10, 0.05, 0.05],
            "edge_abs": [10.0, 10.0, 5.0, 5.0],
            "jp_side_covered": [True, True, True, True],
            "push": [False, False, False, False],
        })

        config = SelectionPolicyConfig(
            selection_policy="EV_THRESHOLD",
            ev_min=0.01,
            phase1_policy="skip",
        )
        result = apply_selection_policy(candidates, config)

        # Should only have weeks 4 and 5
        assert set(result.selected_bets["week"]) == {4, 5}
        assert result.n_selected == 2

    def test_includes_phase1_when_apply(self):
        """phase1_policy='apply' includes weeks 1-3."""
        candidates = pd.DataFrame({
            "game_id": ["g1", "g2", "g3", "g4"],
            "year": [2024, 2024, 2024, 2024],
            "week": [2, 3, 4, 5],
            "ev": [0.10, 0.10, 0.05, 0.05],
            "edge_abs": [10.0, 10.0, 5.0, 5.0],
            "jp_side_covered": [True, True, True, True],
            "push": [False, False, False, False],
        })

        config = SelectionPolicyConfig(
            selection_policy="EV_THRESHOLD",
            ev_min=0.01,
            phase1_policy="apply",
        )
        result = apply_selection_policy(candidates, config)

        # Should include all weeks
        assert set(result.selected_bets["week"]) == {2, 3, 4, 5}
        assert result.n_selected == 4


class TestSelectionMetrics:
    """Tests for compute_selection_metrics."""

    def test_computes_ats_correctly(self, sample_candidates):
        """ATS% computed correctly."""
        # sample_candidates: 5 wins (g1,g2,g4,g5,g7), 3 losses
        metrics = compute_selection_metrics(sample_candidates)
        assert metrics.n_wins == 5
        assert metrics.n_losses == 3
        assert metrics.ats_pct == pytest.approx(5 / 8 * 100, rel=0.01)

    def test_handles_pushes(self):
        """Pushes excluded from ATS calculation."""
        df = pd.DataFrame({
            "game_id": ["g1", "g2", "g3"],
            "year": [2024, 2024, 2024],
            "week": [4, 4, 4],
            "ev": [0.05, 0.05, 0.05],
            "edge_abs": [5.0, 5.0, 5.0],
            "jp_side_covered": [True, False, False],
            "push": [False, False, True],  # g3 is push
        })

        metrics = compute_selection_metrics(df)
        assert metrics.n_wins == 1
        assert metrics.n_losses == 1
        assert metrics.n_pushes == 1
        assert metrics.ats_pct == 50.0  # 1 / (1+1) * 100

    def test_empty_df(self):
        """Empty DataFrame returns zero metrics."""
        empty = pd.DataFrame(columns=["game_id", "year", "week", "ev", "edge_abs", "jp_side_covered", "push"])
        metrics = compute_selection_metrics(empty)
        assert metrics.n_bets == 0
        assert metrics.ats_pct == 0.0


class TestStabilityScore:
    """Tests for compute_stability_score."""

    def test_positive_ats_positive_score(self):
        """Above-breakeven ATS gives positive score."""
        metrics = compute_selection_metrics(pd.DataFrame({
            "game_id": ["g1", "g2", "g3", "g4", "g5"],
            "year": [2024] * 5,
            "week": [4, 4, 4, 5, 5],
            "ev": [0.05] * 5,
            "edge_abs": [5.0] * 5,
            "jp_side_covered": [True, True, True, False, False],  # 3-2 = 60%
            "push": [False] * 5,
        }))

        score = compute_stability_score(metrics, min_bets=5)
        assert score > 0  # 60% > 52.4% breakeven

    def test_low_volume_penalty(self):
        """Few bets penalized via volume factor."""
        # Same ATS, different volumes
        few_bets = pd.DataFrame({
            "game_id": ["g1", "g2"],
            "year": [2024, 2024],
            "week": [4, 4],
            "ev": [0.05, 0.05],
            "edge_abs": [5.0, 5.0],
            "jp_side_covered": [True, True],  # 100% but N=2
            "push": [False, False],
        })

        many_bets = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(20)],
            "year": [2024] * 20,
            "week": [4] * 20,
            "ev": [0.05] * 20,
            "edge_abs": [5.0] * 20,
            "jp_side_covered": [True] * 20,  # 100% N=20
            "push": [False] * 20,
        })

        metrics_few = compute_selection_metrics(few_bets)
        metrics_many = compute_selection_metrics(many_bets)

        score_few = compute_stability_score(metrics_few, min_bets=50)
        score_many = compute_stability_score(metrics_many, min_bets=50)

        # More bets should have better (or equal) score
        assert score_many >= score_few


class TestConfigToLabel:
    """Tests for config_to_label."""

    def test_ev_threshold_label(self):
        """EV_THRESHOLD label format."""
        config = SelectionPolicyConfig(
            selection_policy="EV_THRESHOLD",
            ev_min=0.03,
            max_bets_per_week=10,
        )
        label = config_to_label(config)
        assert "EV_THRESH" in label
        assert "3%" in label
        assert "max=10" in label

    def test_top_n_label(self):
        """TOP_N_PER_WEEK label format."""
        config = SelectionPolicyConfig(
            selection_policy="TOP_N_PER_WEEK",
            top_n_per_week=5,
            ev_floor=0.02,
        )
        label = config_to_label(config)
        assert "TOP_N" in label
        assert "n=5" in label
        assert "floor=2%" in label

    def test_hybrid_label(self):
        """HYBRID label format."""
        config = SelectionPolicyConfig(
            selection_policy="HYBRID",
            top_n_per_week=8,
            ev_floor=0.05,
        )
        label = config_to_label(config)
        assert "HYBRID" in label
        assert "n=8" in label
        assert "floor=5%" in label


class TestPresets:
    """Tests for preset helper functions."""

    def test_get_preset_conservative(self):
        """Conservative preset returns correct config."""
        config = get_selection_policy_preset("conservative")
        assert config.selection_policy == "EV_THRESHOLD"
        assert config.ev_min == 0.03
        assert config.max_bets_per_week == 5
        assert config.phase1_policy == "skip"

    def test_get_preset_balanced(self):
        """Balanced preset returns correct config."""
        config = get_selection_policy_preset("balanced")
        assert config.selection_policy == "TOP_N_PER_WEEK"
        assert config.top_n_per_week == 3
        assert config.ev_floor == 0.01
        assert config.max_bets_per_week == 3
        assert config.phase1_policy == "skip"

    def test_get_preset_aggressive(self):
        """Aggressive preset returns correct config."""
        config = get_selection_policy_preset("aggressive")
        assert config.selection_policy == "TOP_N_PER_WEEK"
        assert config.top_n_per_week == 5
        assert config.ev_floor == 0.01
        assert config.max_bets_per_week == 5
        assert config.phase1_policy == "skip"

    def test_invalid_preset_raises(self):
        """Invalid preset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_selection_policy_preset("invalid_preset")

    def test_preset_case_insensitive(self):
        """Preset lookup is case-insensitive."""
        config1 = get_selection_policy_preset("CONSERVATIVE")
        config2 = get_selection_policy_preset("Conservative")
        config3 = get_selection_policy_preset("conservative")

        assert configs_match(config1, config2)
        assert configs_match(config2, config3)

    def test_preset_configs_are_deterministic(self):
        """Same preset returns identical configs every time."""
        for name in ALLOWED_PRESETS:
            config1 = get_selection_policy_preset(name)
            config2 = get_selection_policy_preset(name)

            assert configs_match(config1, config2)
            assert config1.selection_policy == config2.selection_policy
            assert config1.ev_min == config2.ev_min
            assert config1.ev_floor == config2.ev_floor
            assert config1.top_n_per_week == config2.top_n_per_week
            assert config1.max_bets_per_week == config2.max_bets_per_week

    def test_preset_returns_copy_not_reference(self):
        """get_selection_policy_preset returns a copy, not the global preset."""
        config1 = get_selection_policy_preset("conservative")
        config2 = get_selection_policy_preset("conservative")

        # Should be equal but not the same object
        assert configs_match(config1, config2)
        assert config1 is not config2

    def test_allowed_presets_contains_all_three(self):
        """ALLOWED_PRESETS has exactly the three expected presets."""
        assert ALLOWED_PRESETS == {"conservative", "balanced", "aggressive"}

    def test_configs_match_same_config(self):
        """configs_match returns True for identical configs."""
        config1 = SelectionPolicyConfig(
            selection_policy="EV_THRESHOLD",
            ev_min=0.03,
            ev_floor=0.0,
            top_n_per_week=10,
            max_bets_per_week=5,
        )
        config2 = SelectionPolicyConfig(
            selection_policy="EV_THRESHOLD",
            ev_min=0.03,
            ev_floor=0.0,
            top_n_per_week=10,
            max_bets_per_week=5,
        )
        assert configs_match(config1, config2)

    def test_configs_match_different_policy(self):
        """configs_match returns False for different policies."""
        config1 = SelectionPolicyConfig(selection_policy="EV_THRESHOLD")
        config2 = SelectionPolicyConfig(selection_policy="TOP_N_PER_WEEK")
        assert not configs_match(config1, config2)

    def test_configs_match_different_ev_min(self):
        """configs_match returns False for different ev_min."""
        config1 = SelectionPolicyConfig(ev_min=0.03)
        config2 = SelectionPolicyConfig(ev_min=0.05)
        assert not configs_match(config1, config2)


# =============================================================================
# NEW TESTS: Accounting accuracy, NaN handling, drawdown sort
# =============================================================================

from src.spread_selection.selection_policy import compute_max_drawdown


class TestAccountingAccuracy:
    """Verify n_phase1_skipped, n_filtered_by_ev, n_filtered_by_cap are exact."""

    def _make_candidates(self):
        """10 candidates: 2 phase1 (weeks 2,3), 8 phase2 (weeks 4,5)."""
        return pd.DataFrame({
            "game_id": [f"g{i}" for i in range(10)],
            "year": [2024] * 10,
            "week": [2, 3, 4, 4, 4, 4, 5, 5, 5, 5],
            "ev": [0.10, 0.10, 0.08, 0.06, 0.04, 0.02, 0.09, 0.07, 0.03, 0.01],
            "edge_abs": [10, 10, 8, 6, 4, 2, 9, 7, 3, 1],
            "jp_side_covered": [True] * 10,
            "push": [False] * 10,
        })

    def test_ev_threshold_accounting(self):
        """EV_THRESHOLD: 10 cands, 2 phase1, ev_min=0.05 filters 3, cap=2 removes 1."""
        df = self._make_candidates()
        config = SelectionPolicyConfig(
            selection_policy="EV_THRESHOLD",
            ev_min=0.05,
            max_bets_per_week=2,
            phase1_policy="skip",
        )
        result = apply_selection_policy(df, config)

        assert result.n_candidates == 10
        assert result.n_phase1_skipped == 2
        # After phase1: 8 candidates. ev>=0.05: g2(0.08), g3(0.06), g6(0.09), g7(0.07) = 4
        assert result.n_filtered_by_ev == 4  # 8 - 4 = 4 below threshold
        # 4 pass EV, cap=2/week: wk4 has 2 (g2,g3), wk5 has 2 (g6,g7) → 0 capped
        assert result.n_filtered_by_cap == 0
        assert result.n_selected == 4
        # Verify: phase1 + ev + cap + selected = candidates
        assert (result.n_phase1_skipped + result.n_filtered_by_ev +
                result.n_filtered_by_cap + result.n_selected) == result.n_candidates

    def test_ev_threshold_cap_fires(self):
        """EV_THRESHOLD with tight cap: verify cap count is exact."""
        df = self._make_candidates()
        config = SelectionPolicyConfig(
            selection_policy="EV_THRESHOLD",
            ev_min=0.01,  # keep almost all
            max_bets_per_week=1,
            phase1_policy="skip",
        )
        result = apply_selection_policy(df, config)

        assert result.n_candidates == 10
        assert result.n_phase1_skipped == 2
        assert result.n_filtered_by_ev == 0  # all 8 pass ev_min=0.01
        assert result.n_filtered_by_cap == 6  # 8 - 2 (1 per week × 2 weeks)
        assert result.n_selected == 2

    def test_top_n_accounting(self):
        """TOP_N_PER_WEEK: exact counts for top_n + floor + cap."""
        df = self._make_candidates()
        config = SelectionPolicyConfig(
            selection_policy="TOP_N_PER_WEEK",
            top_n_per_week=3,
            ev_floor=0.05,
            max_bets_per_week=10,
            phase1_policy="skip",
        )
        result = apply_selection_policy(df, config)

        assert result.n_candidates == 10
        assert result.n_phase1_skipped == 2
        # wk4: top3 = [0.08, 0.06, 0.04], floor=0.05 keeps [0.08, 0.06] → 1 ev-filtered
        # wk5: top3 = [0.09, 0.07, 0.03], floor=0.05 keeps [0.09, 0.07] → 1 ev-filtered
        assert result.n_filtered_by_ev == 2
        # wk4: 4 candidates, top3 → 1 cap-filtered; wk5: same → 1 cap-filtered
        assert result.n_filtered_by_cap == 2
        assert result.n_selected == 4

    def test_hybrid_accounting(self):
        """HYBRID: exact counts for floor-first + top_n + cap."""
        df = self._make_candidates()
        config = SelectionPolicyConfig(
            selection_policy="HYBRID",
            top_n_per_week=2,
            ev_floor=0.03,
            max_bets_per_week=10,
            phase1_policy="skip",
        )
        result = apply_selection_policy(df, config)

        assert result.n_candidates == 10
        assert result.n_phase1_skipped == 2
        # wk4: floor filters 0.02 → 1 ev-filtered; wk5: floor filters 0.01 → 1 ev-filtered
        assert result.n_filtered_by_ev == 2
        # wk4: 3 pass floor, top2 → 1 cap; wk5: 3 pass floor, top2 → 1 cap
        assert result.n_filtered_by_cap == 2
        assert result.n_selected == 4

    def test_accounting_sums_to_candidates(self):
        """For all policies, phase1 + ev + cap + selected = candidates."""
        df = self._make_candidates()
        for policy in ["EV_THRESHOLD", "TOP_N_PER_WEEK", "HYBRID"]:
            config = SelectionPolicyConfig(
                selection_policy=policy,
                ev_min=0.03,
                ev_floor=0.03,
                top_n_per_week=2,
                max_bets_per_week=2,
                phase1_policy="skip",
            )
            result = apply_selection_policy(df, config)
            total = (result.n_phase1_skipped + result.n_filtered_by_ev +
                     result.n_filtered_by_cap + result.n_selected)
            assert total == result.n_candidates, (
                f"{policy}: {result.n_phase1_skipped}+{result.n_filtered_by_ev}+"
                f"{result.n_filtered_by_cap}+{result.n_selected} != {result.n_candidates}"
            )


class TestNaNOutcomes:
    """Verify compute_selection_metrics handles NaN outcomes gracefully."""

    def test_nan_outcomes_not_counted_as_losses(self):
        """NaN in outcome_col should be dropped, not treated as a loss."""
        df = pd.DataFrame({
            "game_id": ["g1", "g2", "g3"],
            "year": [2024, 2024, 2024],
            "week": [4, 4, 4],
            "ev": [0.05, 0.05, 0.05],
            "edge_abs": [5.0, 5.0, 5.0],
            "jp_side_covered": [True, False, np.nan],  # g3 unsettled
            "push": [False, False, False],
        })
        metrics = compute_selection_metrics(df)
        # g3 dropped → 2 bets: 1 win, 1 loss
        assert metrics.n_bets == 2
        assert metrics.n_wins == 1
        assert metrics.n_losses == 1
        assert metrics.ats_pct == 50.0


class TestMaxDrawdownSort:
    """Verify compute_max_drawdown is stable regardless of input order."""

    def test_shuffled_input_same_drawdown(self):
        """Shuffled input produces same drawdown as sorted input."""
        df = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(10)],
            "year": [2024] * 10,
            "week": [4, 4, 4, 5, 5, 5, 6, 6, 6, 6],
            "jp_side_covered": [True, True, False, False, False, True, True, False, True, True],
            "push": [False] * 10,
        })

        dd_sorted = compute_max_drawdown(df)

        # Shuffle and compute again
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        dd_shuffled = compute_max_drawdown(df_shuffled)

        assert dd_sorted == dd_shuffled
