"""Tests for EFM ridge regression fixes."""

import numpy as np
import pandas as pd
import pytest

from src.models.efficiency_foundation_model import EfficiencyFoundationModel


class TestHFANotRegularized:
    """HFA coefficient should not be shrunk by ridge penalty."""

    def _make_synthetic_data(self, n_per_matchup=500, home_effect=0.15):
        """Create synthetic plays with a clear home advantage effect.

        Four teams with varied matchups. When offense is home,
        success rate is base + home_effect; when away, just base.
        """
        rng = np.random.RandomState(42)
        base_sr = 0.40
        teams = ["TeamA", "TeamB", "TeamC", "TeamD"]

        offenses, defenses, signs, ys = [], [], [], []

        for i, off_team in enumerate(teams):
            for j, def_team in enumerate(teams):
                if i == j:
                    continue
                n = n_per_matchup
                n_home = n // 2
                n_away = n - n_home

                # Offense is home
                offenses.extend([off_team] * n_home)
                defenses.extend([def_team] * n_home)
                signs.extend([1.0] * n_home)
                ys.extend((rng.rand(n_home) < base_sr + home_effect).astype(float).tolist())

                # Offense is away (defense is home)
                offenses.extend([off_team] * n_away)
                defenses.extend([def_team] * n_away)
                signs.extend([-1.0] * n_away)
                ys.extend((rng.rand(n_away) < base_sr).astype(float).tolist())

        home_signs = np.array(signs)
        y = np.array(ys)

        team_to_idx = {t: i for i, t in enumerate(teams)}
        off_idx = np.array([team_to_idx[offenses[k]] for k in range(len(offenses))], dtype=np.int32)
        def_idx = np.array([team_to_idx[defenses[k]] for k in range(len(defenses))], dtype=np.int32)
        weights = np.ones(len(y))

        return off_idx, def_idx, home_signs, y, weights, len(teams)

    def test_huge_alpha_preserves_hfa(self):
        """With very large alpha, team coefs shrink to ~0 but HFA stays non-zero."""
        off_idx, def_idx, home_signs, y, weights, n_teams = self._make_synthetic_data()

        efm = EfficiencyFoundationModel(ridge_alpha=50.0)

        # Fit with huge alpha — team coefs should shrink, HFA should not
        beta_huge, _ = efm._ridge_solve_cholesky(
            off_idx, def_idx, home_signs, y, weights, n_teams, alpha=1e6,
        )
        team_coefs = beta_huge[:2 * n_teams]
        learned_hfa = beta_huge[2 * n_teams]

        # Team coefs should be near zero with huge regularization
        assert np.mean(np.abs(team_coefs)) < 0.01, (
            f"Team coefs should shrink with huge alpha, got mean abs {np.mean(np.abs(team_coefs)):.4f}"
        )

        # HFA should remain materially non-zero (true home effect is ~0.15)
        assert abs(learned_hfa) > 0.05, (
            f"HFA should NOT be regularized, got {learned_hfa:.4f} (expected ~0.15)"
        )

        # HFA should have the correct sign (positive = home advantage)
        assert learned_hfa > 0, f"HFA should be positive, got {learned_hfa:.4f}"

    def test_moderate_vs_huge_alpha_hfa_stable(self):
        """HFA should be similar across alpha values since it's not penalized."""
        off_idx, def_idx, home_signs, y, weights, n_teams = self._make_synthetic_data()

        efm = EfficiencyFoundationModel()

        beta_mod, _ = efm._ridge_solve_cholesky(
            off_idx, def_idx, home_signs, y, weights, n_teams, alpha=50.0,
        )
        beta_huge, _ = efm._ridge_solve_cholesky(
            off_idx, def_idx, home_signs, y, weights, n_teams, alpha=1e6,
        )

        hfa_mod = beta_mod[2 * n_teams]
        hfa_huge = beta_huge[2 * n_teams]

        # HFA should be roughly similar regardless of alpha (within 50% of each other)
        assert abs(hfa_mod - hfa_huge) / max(abs(hfa_mod), 1e-6) < 0.5, (
            f"HFA should be stable across alpha: moderate={hfa_mod:.4f}, huge={hfa_huge:.4f}"
        )


class TestTimeDecayFutureWeekGuard:
    """Time decay must raise on future-week plays."""

    def test_future_week_raises(self):
        """Plays with week > eval_week should raise ValueError when time_decay < 1."""
        efm = EfficiencyFoundationModel(ridge_alpha=50.0, time_decay=0.95)

        # Create minimal plays_df with a future week
        plays_df = pd.DataFrame({
            "offense": ["TeamA", "TeamB", "TeamA"],
            "defense": ["TeamB", "TeamA", "TeamB"],
            "home_team": ["TeamA", "TeamB", "TeamA"],
            "is_success": [1, 0, 1],
            "base_weight": [1.0, 1.0, 1.0],
            "week": [3, 4, 6],  # week 6 > eval_week=5
        })

        with pytest.raises(ValueError, match="DATA LEAKAGE"):
            efm._ridge_adjust_metric(plays_df, "is_success", season=2024, eval_week=5)

    @staticmethod
    def _make_plays_df(n=100, weeks=None):
        """Create a minimal plays_df with 4 teams and mixed home/away."""
        rng = np.random.RandomState(99)
        teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
        n_matchups = len(teams) * (len(teams) - 1)
        per = max(1, n // n_matchups)

        offenses, defenses, homes, weeks_col = [], [], [], []
        for off in teams:
            for defe in teams:
                if off == defe:
                    continue
                # Half home, half away
                n_h = per // 2
                n_a = per - n_h
                offenses.extend([off] * n_h + [off] * n_a)
                defenses.extend([defe] * n_h + [defe] * n_a)
                homes.extend([off] * n_h + [defe] * n_a)
                if weeks is not None:
                    weeks_col.extend(rng.choice(weeks, size=n_h + n_a).tolist())

        total = len(offenses)
        data = {
            "offense": offenses,
            "defense": defenses,
            "home_team": homes,
            "is_success": rng.randint(0, 2, total).tolist(),
            "base_weight": [1.0] * total,
        }
        if weeks_col:
            data["week"] = weeks_col
        return pd.DataFrame(data)

    def test_no_future_week_no_error(self):
        """Plays within eval_week should not raise."""
        efm = EfficiencyFoundationModel(ridge_alpha=50.0, time_decay=0.95)
        plays_df = self._make_plays_df(n=200, weeks=[3, 4, 5])

        # Should not raise — all weeks <= eval_week
        result = efm._ridge_adjust_metric(plays_df, "is_success", season=2024, eval_week=5)
        assert result is not None

    def test_decay_disabled_no_guard(self):
        """With time_decay=1.0, future weeks should NOT raise (guard is skipped)."""
        efm = EfficiencyFoundationModel(ridge_alpha=50.0, time_decay=1.0)
        plays_df = self._make_plays_df(n=200, weeks=[3, 10])

        # Should not raise — time_decay=1.0 means guard is not triggered
        result = efm._ridge_adjust_metric(plays_df, "is_success", season=2024, eval_week=5)
        assert result is not None
