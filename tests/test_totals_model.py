"""Regression tests for TotalsModel correctness fixes.

Tests:
1. Walk-forward leak-proof: training with through_week excludes that week+1
2. Multi-year universe: warns/drops when teams missing from universe
3. Zero-game teams: predict_total works for bye-week teams
"""

import numpy as np
import pandas as pd
import pytest

from src.models.totals_model import TotalsModel


TEAMS_8 = [f'Team{c}' for c in 'ABCDEFGH']


def _make_games(weeks, teams=None, year=2024):
    """Build a games DataFrame with full round-robin each week.

    With 8 teams, each week produces 4 games → 10 weeks = 40 games,
    comfortably above the min_games=10 threshold.
    """
    teams = teams or TEAMS_8
    rows = []
    rng = np.random.default_rng(42)
    for w in weeks:
        # Round-robin pairs: (0,1), (2,3), (4,5), (6,7)
        for i in range(0, len(teams) - 1, 2):
            rows.append({
                'home_team': teams[i],
                'away_team': teams[i + 1],
                'home_points': float(rng.integers(14, 42)),
                'away_points': float(rng.integers(14, 42)),
                'week': w,
                'year': year,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Fix #1: Walk-forward semantics — through_week is inclusive upper bound
# =============================================================================

class TestWalkForwardLeakProof:
    """through_week=W must include week W and exclude week W+1."""

    def test_through_week_excludes_next_week(self):
        """Train for predicting week 4: no week-4 rows in training."""
        fbs = set(TEAMS_8)
        games = _make_games(range(1, 5))  # weeks 1-4, 4 games/week = 16 games

        model = TotalsModel(ridge_alpha=10.0)
        model.train(games, fbs, through_week=3)

        assert model._last_train_through_week == 3
        # 3 weeks × 4 games/week = 12 games; each team plays once/week = 3 games
        for team in fbs:
            assert model.team_ratings[team].games_played == 3

    def test_through_week_includes_that_week(self):
        """through_week=3 must include week 3 games."""
        fbs = set(TEAMS_8)
        games = _make_games(range(1, 5))

        model = TotalsModel(ridge_alpha=10.0)
        model.train(games, fbs, through_week=3)

        for team in fbs:
            assert model.team_ratings[team].games_played == 3

    def test_week_4_data_not_leaked(self):
        """Key regression test: wrong operator (<= vs <) would leak the pred week."""
        fbs = set(TEAMS_8)
        games = _make_games(range(1, 4))  # weeks 1-3

        # Add week 4 with extreme scores that would distort ratings
        week4 = pd.DataFrame([
            {'home_team': 'TeamA', 'away_team': 'TeamB',
             'home_points': 100.0, 'away_points': 0.0, 'week': 4, 'year': 2024},
            {'home_team': 'TeamC', 'away_team': 'TeamD',
             'home_points': 0.0, 'away_points': 100.0, 'week': 4, 'year': 2024},
            {'home_team': 'TeamE', 'away_team': 'TeamF',
             'home_points': 100.0, 'away_points': 0.0, 'week': 4, 'year': 2024},
            {'home_team': 'TeamG', 'away_team': 'TeamH',
             'home_points': 0.0, 'away_points': 100.0, 'week': 4, 'year': 2024},
        ])
        all_games = pd.concat([games, week4], ignore_index=True)

        model_clean = TotalsModel(ridge_alpha=10.0)
        model_clean.train(games, fbs, through_week=3)

        model_test = TotalsModel(ridge_alpha=10.0)
        model_test.train(all_games, fbs, through_week=3)

        for team in fbs:
            assert model_test.team_ratings[team].off_adjustment == pytest.approx(
                model_clean.team_ratings[team].off_adjustment, abs=1e-10
            ), f"{team} off_adjustment differs — week 4 data leaked!"

    def test_deprecated_max_week_still_works(self):
        """max_week= kwarg should work as alias for through_week."""
        fbs = set(TEAMS_8)
        games = _make_games(range(1, 5))

        model = TotalsModel(ridge_alpha=10.0)
        model.train(games, fbs, max_week=3)
        assert model._last_train_through_week == 3

    def test_both_through_and_max_week_raises(self):
        """Cannot specify both through_week and max_week."""
        fbs = set(TEAMS_8)
        games = _make_games(range(1, 5))

        model = TotalsModel(ridge_alpha=10.0)
        with pytest.raises(ValueError, match="Cannot specify both"):
            model.train(games, fbs, through_week=3, max_week=3)


# =============================================================================
# Fix #2: Multi-year team universe — warn on dropped teams
# =============================================================================

class TestMultiYearTeamUniverse:
    """Universe mismatch must warn, not silently drop."""

    def test_warns_on_missing_teams(self, caplog):
        """Teams not in universe should trigger a warning with team names."""
        games = _make_games(range(1, 5))
        # Add a game with a team NOT in the pre-set universe
        extra = pd.DataFrame([{
            'home_team': 'NewTeam', 'away_team': 'TeamA',
            'home_points': 21.0, 'away_points': 28.0, 'week': 2, 'year': 2024,
        }])
        games = pd.concat([games, extra], ignore_index=True)

        # fbs_teams includes NewTeam (so FBS filter passes), but universe doesn't
        fbs_with_new = set(TEAMS_8) | {'NewTeam'}
        universe_without_new = set(TEAMS_8)

        model = TotalsModel(ridge_alpha=10.0)
        model.set_team_universe(universe_without_new)

        import logging
        with caplog.at_level(logging.WARNING):
            model.train(games, fbs_with_new, through_week=4)

        assert any('NewTeam' in msg for msg in caplog.messages), (
            "Expected warning mentioning 'NewTeam' in dropped games"
        )

    def test_no_warning_when_all_teams_present(self, caplog):
        """No warning when all game teams are in universe."""
        fbs = set(TEAMS_8)
        games = _make_games(range(1, 5))

        model = TotalsModel(ridge_alpha=10.0)
        import logging
        with caplog.at_level(logging.WARNING):
            model.train(games, fbs, through_week=4)

        drop_warnings = [m for m in caplog.messages if 'Dropping' in m and 'games' in m]
        assert len(drop_warnings) == 0


# =============================================================================
# Fix #3: Zero-game teams get baseline predictions
# =============================================================================

class TestZeroGameTeams:
    """Teams with 0 games should still be predictable."""

    def _train_with_idle_teams(self):
        """Train on 8 active teams, with 2 extra idle teams in universe."""
        fbs = set(TEAMS_8) | {'IdleX', 'IdleY'}
        games = _make_games(range(1, 5))  # 16 games, only TEAMS_8 play
        model = TotalsModel(ridge_alpha=10.0)
        model.train(games, fbs, through_week=4)
        return model

    def test_predict_total_for_bye_week_team(self):
        """A team with 0 games should return a valid prediction, not None."""
        model = self._train_with_idle_teams()

        assert 'IdleX' in model.team_ratings
        assert model.team_ratings['IdleX'].games_played == 0
        assert model.team_ratings['IdleX'].off_adjustment == pytest.approx(0.0, abs=0.1)

        pred = model.predict_total('IdleX', 'IdleY')
        assert pred is not None
        assert pred.predicted_total > 0

    def test_zero_game_team_gets_baseline_rating(self):
        """Zero-game team adjustments should be ~0 (Ridge shrinkage)."""
        model = self._train_with_idle_teams()

        tc = model.team_ratings['IdleX']
        assert abs(tc.off_adjustment) < 1.0
        assert abs(tc.def_adjustment) < 1.0
        assert tc.adj_off_ppg == pytest.approx(model.baseline, abs=1.0)

    def test_reliability_zero_for_zero_games(self):
        """get_ratings_df should show reliability=0.0 for 0-game teams."""
        model = self._train_with_idle_teams()

        df = model.get_ratings_df(min_games=0)
        idle_row = df[df['team'] == 'IdleX']
        assert len(idle_row) == 1
        assert idle_row.iloc[0]['reliability'] == 0.0
        assert idle_row.iloc[0]['games_played'] == 0
