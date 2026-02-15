"""Tests for Dynamic FCS Strength Estimator."""

import logging

import polars as pl
import pytest

from src.models.fcs_strength import FCSStrengthEstimator, FCSTeamStrength


# Helper to build a games DataFrame
def _games_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


FBS_TEAMS = {"Alabama", "Georgia", "Ohio State", "Florida State", "Kent State"}


class TestWalkForwardSafety:
    """Postseason and future games must be excluded."""

    def test_season_type_excludes_postseason(self):
        """Games with season_type != 'regular' are excluded."""
        games = _games_df([
            {"week": 2, "home_team": "Alabama", "away_team": "FCS Team A",
             "home_points": 42, "away_points": 10, "season_type": "regular"},
            {"week": 16, "home_team": "Georgia", "away_team": "FCS Team A",
             "home_points": 35, "away_points": 7, "season_type": "postseason"},
        ])
        est = FCSStrengthEstimator()
        est.update_from_games(games, FBS_TEAMS, through_week=20)

        strength = est.get_strength("FCS Team A")
        assert strength is not None
        assert strength.n_games == 1  # Only the regular season game

    def test_week_cap_without_season_type(self):
        """Without season_type, weeks > 15 are excluded by default."""
        games = _games_df([
            {"week": 2, "home_team": "Alabama", "away_team": "FCS Team A",
             "home_points": 42, "away_points": 10},
            {"week": 17, "home_team": "Georgia", "away_team": "FCS Team A",
             "home_points": 35, "away_points": 7},
        ])
        est = FCSStrengthEstimator()
        est.update_from_games(games, FBS_TEAMS, through_week=20)

        strength = est.get_strength("FCS Team A")
        assert strength is not None
        assert strength.n_games == 1  # Week 17 excluded

    def test_week_cap_logs_warning(self, caplog):
        """Warn when games beyond max_regular_season_week would be included."""
        games = _games_df([
            {"week": 16, "home_team": "Alabama", "away_team": "FCS Team A",
             "home_points": 42, "away_points": 10},
        ])
        est = FCSStrengthEstimator()
        with caplog.at_level(logging.WARNING):
            est.update_from_games(games, FBS_TEAMS, through_week=20)
        assert "postseason leakage" in caplog.text.lower()


class TestTeamNameNormalization:
    """Team name normalization and mismatch diagnostics."""

    def test_normalize_fixes_alias(self):
        """Normalization hook resolves aliases for FBS classification."""
        alias_map = {"Florida St.": "Florida State"}

        games = _games_df([
            {"week": 1, "home_team": "Florida St.", "away_team": "FCS Team B",
             "home_points": 45, "away_points": 14},
        ])

        est = FCSStrengthEstimator(
            normalize_team_name=lambda t: alias_map.get(t, t)
        )
        est.update_from_games(games, FBS_TEAMS, through_week=3)

        # Should be classified as FBS vs FCS
        strength = est.get_strength("FCS Team B")
        assert strength is not None
        assert strength.n_games == 1

    def test_mismatch_without_normalization_warns(self, caplog):
        """Unmatched FBS name logs warning and FCS team gets baseline fallback."""
        games = _games_df([
            {"week": 1, "home_team": "Florida St.", "away_team": "FCS Team B",
             "home_points": 45, "away_points": 14},
        ])

        est = FCSStrengthEstimator()
        with caplog.at_level(logging.WARNING):
            est.update_from_games(games, FBS_TEAMS, through_week=3)

        # Neither team matched FBS → warning
        assert "neither team" in caplog.text.lower() or "unmatched" in caplog.text.lower()

        # FCS Team B not observed → baseline fallback
        assert est.get_strength("FCS Team B") is None
        penalty = est.get_penalty("FCS Team B")
        assert penalty == est.baseline_penalty


class TestColumnValidation:
    """Required columns must exist."""

    def test_missing_columns_raises(self):
        """Missing required columns raise ValueError."""
        games = pl.DataFrame({"week": [1], "home_team": ["A"]})
        est = FCSStrengthEstimator()
        with pytest.raises(ValueError, match="missing required columns"):
            est.update_from_games(games, FBS_TEAMS, through_week=3)


class TestHFANeutralization:
    """HFA neutralization sign convention."""

    def test_hfa_neutralization_home_fbs(self):
        """When home is FBS and HFA is applied, margin should decrease."""
        games = _games_df([
            {"week": 1, "home_team": "Alabama", "away_team": "FCS Team C",
             "home_points": 42, "away_points": 10},
        ])

        est_no_hfa = FCSStrengthEstimator(hfa_value=0.0)
        est_no_hfa.update_from_games(games, FBS_TEAMS, through_week=3)

        est_with_hfa = FCSStrengthEstimator(hfa_value=3.0)
        est_with_hfa.update_from_games(games, FBS_TEAMS, through_week=3)

        s_no = est_no_hfa.get_strength("FCS Team C")
        s_with = est_with_hfa.get_strength("FCS Team C")

        # margin = fcs_pts - (fbs_pts - hfa). With hfa=3, FBS score reduced by 3,
        # so FCS margin is LESS negative (closer to 0) → lower penalty
        assert s_with.raw_margin > s_no.raw_margin
        assert s_with.penalty < s_no.penalty

    def test_hfa_neutralization_home_fcs(self):
        """When home is FCS and HFA is applied, margin should decrease."""
        games = _games_df([
            {"week": 1, "home_team": "FCS Team C", "away_team": "Alabama",
             "home_points": 10, "away_points": 42},
        ])

        est_no_hfa = FCSStrengthEstimator(hfa_value=0.0)
        est_no_hfa.update_from_games(games, FBS_TEAMS, through_week=3)

        est_with_hfa = FCSStrengthEstimator(hfa_value=3.0)
        est_with_hfa.update_from_games(games, FBS_TEAMS, through_week=3)

        s_no = est_no_hfa.get_strength("FCS Team C")
        s_with = est_with_hfa.get_strength("FCS Team C")

        # margin = (fcs_pts - hfa) - fbs_pts. With hfa=3, FCS score reduced by 3,
        # so FCS margin is MORE negative → higher penalty
        assert s_with.raw_margin < s_no.raw_margin
        assert s_with.penalty > s_no.penalty


class TestBaselineBehavior:
    """Core estimator math (existing behavior, regression guard)."""

    def test_unknown_team_gets_baseline_penalty(self):
        est = FCSStrengthEstimator()
        assert est.get_penalty("Unknown FCS Team") == est.baseline_penalty

    def test_shrinkage_moves_toward_baseline(self):
        """With few games, shrunk margin should be closer to baseline than raw."""
        games = _games_df([
            {"week": 1, "home_team": "Alabama", "away_team": "Elite FCS",
             "home_points": 20, "away_points": 17},  # Close game
        ])
        est = FCSStrengthEstimator()
        est.update_from_games(games, FBS_TEAMS, through_week=3)

        s = est.get_strength("Elite FCS")
        # Raw margin = 17 - 20 = -3. Baseline = -28.
        # Shrunk should be between -3 and -28
        assert s.raw_margin == pytest.approx(-3.0)
        assert s.shrunk_margin < s.raw_margin  # Pulled toward -28
        assert s.shrunk_margin > est.baseline_margin  # But not all the way
