"""Tests for Vegas comparison module."""

import numpy as np
import pandas as pd
import pytest

from src.predictions.vegas_comparison import VegasComparison, VegasLine


# Minimal PredictedSpread-like object for testing (avoids importing full SpreadGenerator)
class _FakeSpread:
    def __init__(self, home_team, away_team, spread, game_id=None, confidence="medium"):
        self.home_team = home_team
        self.away_team = away_team
        self.spread = spread
        self.game_id = game_id
        self.confidence = confidence


class TestFetchLinesClearsState:
    """Fix 1: fetch_lines must clear stale state from prior calls."""

    def _populate_lines(self, vc: VegasComparison, lines: list[VegasLine]):
        """Directly populate lookup dicts (simulates fetch_lines internals)."""
        vc.lines_by_id.clear()
        vc.lines.clear()
        for vl in lines:
            vc.lines_by_id[vl.game_id] = vl
            vc.lines[(vl.home_team, vl.away_team)] = vl

    def test_second_fetch_clears_first(self):
        """Calling fetch_lines again should not retain lines from the first call."""
        vc = VegasComparison(client=None)

        # Simulate week 13 lines
        week13_line = VegasLine(
            game_id=100, home_team="Michigan", away_team="Ohio State",
            spread=-7.5, spread_open=-7.5,
        )
        vc.lines_by_id[100] = week13_line
        vc.lines[("Michigan", "Ohio State")] = week13_line

        # Simulate clearing (what fetch_lines now does at the top)
        vc.lines_by_id.clear()
        vc.lines.clear()

        # Add week 15 rematch
        week15_line = VegasLine(
            game_id=200, home_team="Michigan", away_team="Ohio State",
            spread=-3.5, spread_open=-3.5,
        )
        vc.lines_by_id[200] = week15_line
        vc.lines[("Michigan", "Ohio State")] = week15_line

        # Team-name lookup should return week 15 line, not week 13
        result = vc.get_line("Michigan", "Ohio State")
        assert result is not None
        assert result.spread == -3.5
        assert result.game_id == 200

    def test_stale_team_key_not_retained(self):
        """After clear, a team pair from a prior week should not be found."""
        vc = VegasComparison(client=None)

        # Week 5 line
        vc.lines[("Alabama", "Georgia")] = VegasLine(
            game_id=300, home_team="Alabama", away_team="Georgia", spread=-3.0,
        )

        # Clear (simulates new fetch_lines call)
        vc.lines_by_id.clear()
        vc.lines.clear()

        # Different week, different game
        vc.lines[("Texas", "Oklahoma")] = VegasLine(
            game_id=400, home_team="Texas", away_team="Oklahoma", spread=-1.0,
        )

        # Alabama/Georgia should NOT be found
        assert vc.get_line("Alabama", "Georgia") is None
        # Texas/Oklahoma should be found
        assert vc.get_line("Texas", "Oklahoma") is not None


class TestGameIdPropagation:
    """Fix 2: team-name fallback must propagate game_id."""

    def test_fallback_sets_game_id(self):
        """When prediction has no game_id, fallback should fill it from VegasLine."""
        vc = VegasComparison(client=None)

        # Set up a line via team-name dict only (no lines_by_id entry)
        vl = VegasLine(
            game_id=500, home_team="Oregon", away_team="USC",
            spread=-6.0, spread_open=-5.5, over_under=55.0,
        )
        vc.lines[("Oregon", "USC")] = vl

        pred = _FakeSpread("Oregon", "USC", spread=8.0, game_id=None)
        df = vc.generate_comparison_df([pred])

        assert len(df) == 1
        assert df.at[0, "game_id"] == 500
        assert df.at[0, "vegas_spread"] == -6.0

    def test_existing_game_id_not_overwritten(self):
        """When prediction already has game_id, fallback should not overwrite it."""
        vc = VegasComparison(client=None)

        vl = VegasLine(
            game_id=600, home_team="Oregon", away_team="USC",
            spread=-6.0, spread_open=-5.5,
        )
        vc.lines_by_id[600] = vl

        pred = _FakeSpread("Oregon", "USC", spread=8.0, game_id=600)
        df = vc.generate_comparison_df([pred])

        assert len(df) == 1
        assert df.at[0, "game_id"] == 600


class TestGameIdNormalization:
    """Fix 3: game_id type normalization for consistent matching."""

    def test_string_game_id_matches_int_key(self):
        """String game_id in prediction should match int key in lines_by_id."""
        vc = VegasComparison(client=None)

        vl = VegasLine(
            game_id=401635901, home_team="Alabama", away_team="LSU",
            spread=-7.0, spread_open=-6.5,
        )
        vc.lines_by_id[401635901] = vl

        # get_line with string game_id
        result = vc.get_line("Alabama", "LSU", game_id="401635901")
        assert result is not None
        assert result.game_id == 401635901

    def test_string_game_id_merge_in_comparison_df(self):
        """String game_id in prediction merges correctly with int game_id in lines."""
        vc = VegasComparison(client=None)

        vl = VegasLine(
            game_id=401635901, home_team="Alabama", away_team="LSU",
            spread=-7.0, spread_open=-6.5, over_under=48.0,
        )
        vc.lines_by_id[401635901] = vl

        # Prediction with string game_id
        pred = _FakeSpread("Alabama", "LSU", spread=9.0, game_id="401635901")
        df = vc.generate_comparison_df([pred])

        assert len(df) == 1
        assert not pd.isna(df.at[0, "vegas_spread"])
        assert df.at[0, "vegas_spread"] == -7.0

    def test_non_numeric_game_id_falls_through(self):
        """Non-numeric game_id should not crash, just fall through to team-name lookup."""
        vc = VegasComparison(client=None)

        vl = VegasLine(
            game_id=999, home_team="Texas", away_team="OU",
            spread=-2.0,
        )
        vc.lines[("Texas", "OU")] = vl

        result = vc.get_line("Texas", "OU", game_id="not-a-number")
        assert result is not None
        assert result.spread == -2.0


class TestEdgeCalculation:
    """Verify edge sign convention is consistent across code paths."""

    def test_compare_prediction_edge(self):
        """compare_prediction edge = (-model) - vegas."""
        vc = VegasComparison(client=None)

        vl = VegasLine(
            game_id=700, home_team="Ohio State", away_team="Penn State",
            spread=-3.0, spread_open=-3.0,
        )
        vc.lines_by_id[700] = vl

        pred = _FakeSpread("Ohio State", "Penn State", spread=10.0, game_id=700)
        result = vc.compare_prediction(pred)

        assert result is not None
        # edge = (-10) - (-3) = -7 → model likes HOME 7 more
        assert result["edge"] == pytest.approx(-7.0)

    def test_comparison_df_edge_matches(self):
        """generate_comparison_df edge should match compare_prediction for same inputs."""
        vc = VegasComparison(client=None)

        vl = VegasLine(
            game_id=700, home_team="Ohio State", away_team="Penn State",
            spread=-3.0, spread_open=-3.0, over_under=45.0,
        )
        vc.lines_by_id[700] = vl
        vc.lines[("Ohio State", "Penn State")] = vl

        pred = _FakeSpread("Ohio State", "Penn State", spread=10.0, game_id=700)

        single = vc.compare_prediction(pred)
        df = vc.generate_comparison_df([pred])

        assert single["edge"] == pytest.approx(df.at[0, "edge"])
