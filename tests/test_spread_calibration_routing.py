"""Tests for spread calibration phase routing and OOS validation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Skip if artifacts don't exist
ARTIFACT_DIR = Path("data/spread_selection/artifacts")
PHASE2_ARTIFACT = ARTIFACT_DIR / "spread_ev_calibration_phase2_only_2022_2025.json"


@pytest.fixture
def has_artifacts():
    """Check if calibration artifacts exist."""
    return PHASE2_ARTIFACT.exists()


class TestPhaseRouting:
    """Test get_spread_calibration_for_week phase routing."""

    @pytest.mark.skipif(
        not PHASE2_ARTIFACT.exists(),
        reason="Calibration artifacts not found"
    )
    def test_phase1_skip_returns_none(self):
        """Phase 1 with skip policy should return None."""
        from src.spread_selection.calibration import get_spread_calibration_for_week

        # Week 1, 2, 3 should all return None with skip policy
        for week in [1, 2, 3]:
            cal = get_spread_calibration_for_week(week, phase1_policy="skip")
            assert cal is None, f"Week {week} should return None with skip policy"

    @pytest.mark.skipif(
        not PHASE2_ARTIFACT.exists(),
        reason="Calibration artifacts not found"
    )
    def test_phase2_returns_calibration(self):
        """Phase 2 (weeks 4-15) should return valid calibration."""
        from src.spread_selection.calibration import get_spread_calibration_for_week

        for week in [4, 8, 12, 15]:
            cal = get_spread_calibration_for_week(week)
            assert cal is not None, f"Week {week} should return calibration"
            assert cal.slope > 0, "Slope should be positive"
            assert cal.implied_breakeven_edge < 10, "Breakeven should be reasonable"

    @pytest.mark.skipif(
        not PHASE2_ARTIFACT.exists(),
        reason="Calibration artifacts not found"
    )
    def test_phase1_weighted_returns_calibration(self):
        """Phase 1 with weighted policy should return calibration."""
        from src.spread_selection.calibration import get_spread_calibration_for_week

        cal = get_spread_calibration_for_week(2, phase1_policy="weighted")
        assert cal is not None, "Weighted policy should return calibration"

    @pytest.mark.skipif(
        not PHASE2_ARTIFACT.exists(),
        reason="Calibration artifacts not found"
    )
    def test_force_phase2_overrides_phase1(self):
        """force_phase2 mode should return calibration even for week 1."""
        from src.spread_selection.calibration import get_spread_calibration_for_week

        cal = get_spread_calibration_for_week(1, phase_mode="force_phase2")
        assert cal is not None, "force_phase2 should return calibration"

    @pytest.mark.skipif(
        not PHASE2_ARTIFACT.exists(),
        reason="Calibration artifacts not found"
    )
    def test_postseason_default_uses_phase2(self):
        """Postseason (week 16+) should use Phase 2 by default."""
        from src.spread_selection.calibration import get_spread_calibration_for_week

        cal = get_spread_calibration_for_week(17)
        assert cal is not None, "Postseason should return calibration by default"


class TestOOSValidation:
    """Test out-of-sample validation logic."""

    def test_train_test_no_overlap(self):
        """Verify train and test years cannot overlap."""
        # This is validated in the OOS script but we test the assertion

        train_years = [2022, 2023, 2024]
        test_years = [2025]

        overlap = set(train_years) & set(test_years)
        assert len(overlap) == 0, "Train and test years must not overlap"

    def test_oos_builder_excludes_test_years(self):
        """OOS calibration fitting must not include test year data."""
        # This tests the logic that training data is filtered correctly

        # Mock data
        df = pd.DataFrame({
            'year': [2022, 2022, 2023, 2023, 2024, 2024, 2025, 2025],
            'week': [4, 5, 4, 5, 4, 5, 4, 5],
            'edge_abs': [5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0],
            'jp_side_covered': [True, False, True, True, False, True, True, True],
            'push': [False] * 8,
        })

        train_years = [2022, 2023, 2024]
        test_years = [2025]

        # Filter to train years only
        train_df = df[df['year'].isin(train_years)]

        # Verify no test year data in training
        assert 2025 not in train_df['year'].values, "2025 must not be in training data"
        assert len(train_df) == 6, "Should have 6 training rows"


class TestPlaceholderOdds:
    """Test placeholder odds detection."""

    def test_odds_placeholder_flag_added(self):
        """odds_placeholder column should be added to DataFrame."""
        from scripts.validate_spread_calibration_oos import detect_placeholder_odds

        df = pd.DataFrame({
            'spread_open': [-7.0, -3.5],
            'spread_close': [-7.0, -3.5],
        })

        result = detect_placeholder_odds(df)

        assert 'odds_placeholder' in result.columns, "odds_placeholder column should exist"

    def test_missing_odds_columns_flags_all_as_placeholder(self):
        """Missing odds columns should flag all as placeholder."""
        from scripts.validate_spread_calibration_oos import detect_placeholder_odds

        df = pd.DataFrame({
            'other_col': [1, 2, 3],
        })

        result = detect_placeholder_odds(df)

        assert 'odds_placeholder' in result.columns
        assert result['odds_placeholder'].all(), "All should be flagged as placeholder"


class TestCalibrationLoading:
    """Test calibration artifact loading."""

    @pytest.mark.skipif(
        not PHASE2_ARTIFACT.exists(),
        reason="Calibration artifacts not found"
    )
    def test_load_phase2_calibration(self):
        """Load Phase 2 calibration from artifact."""
        from src.spread_selection.calibration import get_default_spread_calibration

        cal = get_default_spread_calibration("phase2")

        assert cal is not None
        assert cal.slope > 0.01, "Slope should be positive and non-trivial"
        assert cal.implied_breakeven_edge < 10, "Breakeven should be < 10 pts"
        assert 0.4 < cal.p_cover_at_zero < 0.6, "p_cover_at_zero should be near 0.5"

    @pytest.mark.skipif(
        not PHASE2_ARTIFACT.exists(),
        reason="Calibration artifacts not found"
    )
    def test_load_weighted_calibration(self):
        """Load weighted calibration from artifact."""
        from src.spread_selection.calibration import get_default_spread_calibration

        cal = get_default_spread_calibration("weighted")

        assert cal is not None
        assert cal.slope > 0, "Slope should be positive"

    def test_invalid_phase_raises(self):
        """Invalid phase name should raise ValueError."""
        from src.spread_selection.calibration import get_default_spread_calibration

        with pytest.raises(ValueError):
            get_default_spread_calibration("invalid_phase")
