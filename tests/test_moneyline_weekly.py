"""Tests for moneyline weekly runner helpers."""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.spread_selection.moneyline_ev_engine import COLUMNS as ENGINE_COLUMNS, MoneylineEVConfig
from src.spread_selection.moneyline_weekly import (
    ALL_LOG_COLUMNS,
    DEDUP_KEY,
    append_to_log,
    build_config,
    load_sigma_artifact,
    run_recommend,
    save_sigma_artifact,
    settle_week,
)


def _make_inputs_csv(path: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_scores_csv(path: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _sample_events():
    return [
        {
            "year": 2025, "week": 8, "game_id": "g1",
            "home_team": "Alabama", "away_team": "Tennessee",
            "model_spread": 10.0, "market_spread": -3.0,
            "ml_odds_home": -130, "ml_odds_away": 110,
        },
        {
            "year": 2025, "week": 8, "game_id": "g2",
            "home_team": "Ohio State", "away_team": "Penn State",
            "model_spread": 7.0, "market_spread": -4.0,
            "ml_odds_home": -180, "ml_odds_away": 155,
        },
        {
            "year": 2025, "week": 8, "game_id": "g3",
            "home_team": "Georgia", "away_team": "Florida",
            "model_spread": 3.0, "market_spread": 1.0,
            "ml_odds_home": -200, "ml_odds_away": 170,
        },
    ]


def _cfg(**kw):
    return MoneylineEVConfig(margin_sigma=13.5, **kw)


# ---- 1. Dedupe prevents duplicate append ----

def test_dedupe_prevents_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "test.csv"
        from src.spread_selection.moneyline_ev_engine import evaluate_moneylines
        events = _sample_events()[:1]
        cfg = _cfg(min_disagreement_pts=5.0)
        a, b = evaluate_moneylines(events, cfg)

        # First write
        n1, s1 = append_to_log(a, b, cfg, log_path)
        assert n1 > 0
        assert s1 == 0

        # Second write — should skip all
        n2, s2 = append_to_log(a, b, cfg, log_path)
        assert n2 == 0
        assert s2 > 0

        # File should have same row count as first write
        df = pd.read_csv(log_path)
        assert len(df) == n1


# ---- 2. Schema evolution (union of columns) ----

def test_schema_evolution():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "test.csv"
        # Write a CSV with a subset of columns
        initial = pd.DataFrame({
            "year": [2025], "week": [8], "game_id": ["g_old"],
            "side": ["HOME"], "list_type": ["A"],
            "old_column": ["x"],
        })
        initial.to_csv(log_path, index=False)

        from src.spread_selection.moneyline_ev_engine import evaluate_moneylines
        cfg = _cfg(min_disagreement_pts=5.0)
        events = _sample_events()[:1]
        a, b = evaluate_moneylines(events, cfg)
        append_to_log(a, b, cfg, log_path)

        df = pd.read_csv(log_path)
        assert "old_column" in df.columns
        assert "ev" in df.columns


# ---- 3. Dry run does not write file ----

def test_dry_run_no_write():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "test.csv"
        from src.spread_selection.moneyline_ev_engine import evaluate_moneylines
        cfg = _cfg(min_disagreement_pts=5.0)
        events = _sample_events()[:1]
        a, b = evaluate_moneylines(events, cfg)

        n, s = append_to_log(a, b, cfg, log_path, dry_run=True)
        assert n > 0
        assert not log_path.exists()


# ---- 4. Settlement updates only unsettled List A rows ----

def test_settlement_updates_unsettled_only():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "test.csv"
        scores_path = Path(tmp) / "scores.csv"

        # Create a log with 2 List A rows (one already settled) + 1 List B
        log_data = pd.DataFrame({
            "year": [2025, 2025, 2025],
            "week": [8, 8, 8],
            "game_id": ["g1", "g2", "g1"],
            "home_team": ["A", "C", "A"],
            "away_team": ["B", "D", "B"],
            "side": ["HOME", "HOME", "HOME"],
            "list_type": ["A", "A", "B"],
            "odds_american": [-130, -150, -130],
            "stake": [20.0, 15.0, 10.0],
            "settled_timestamp": ["2025-01-01T00:00:00", None, None],
            "actual_margin": [7.0, None, None],
            "covered": ["W", None, None],
            "profit_units": [15.38, None, None],
        })
        log_data.to_csv(log_path, index=False)

        _make_scores_csv(str(scores_path), [
            {"year": 2025, "week": 8, "game_id": "g2", "home_points": 28, "away_points": 21},
        ])

        settled, warns, already = settle_week(log_path, str(scores_path), 2025, 8)
        assert settled == 1  # only g2
        assert already == 1  # g1 was pre-settled

        df = pd.read_csv(log_path)
        assert len(df) == 3  # row count preserved
        g2_row = df[df["game_id"] == "g2"]
        assert g2_row.iloc[0]["covered"] == "W"


# ---- 5. Profit calculation for + and - odds ----

def test_profit_negative_odds():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "test.csv"
        scores_path = Path(tmp) / "scores.csv"

        # -200 odds: d=1.5, b=0.5. Win => profit = 0.5 * 20 = 10
        log_data = pd.DataFrame({
            "year": [2025], "week": [8], "game_id": ["g1"],
            "home_team": ["A"], "away_team": ["B"],
            "side": ["HOME"], "list_type": ["A"],
            "odds_american": [-200], "stake": [20.0],
            "settled_timestamp": [None], "actual_margin": [None],
            "covered": [None], "profit_units": [None],
        })
        log_data.to_csv(log_path, index=False)
        _make_scores_csv(str(scores_path), [
            {"year": 2025, "week": 8, "game_id": "g1", "home_points": 30, "away_points": 20},
        ])

        settle_week(log_path, str(scores_path), 2025, 8)
        df = pd.read_csv(log_path)
        assert df.iloc[0]["profit_units"] == pytest.approx(10.0)


def test_profit_positive_odds():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "test.csv"
        scores_path = Path(tmp) / "scores.csv"

        # +150 odds: d=2.5, b=1.5. Loss (away side, home won) => profit = -20
        log_data = pd.DataFrame({
            "year": [2025], "week": [8], "game_id": ["g1"],
            "home_team": ["A"], "away_team": ["B"],
            "side": ["AWAY"], "list_type": ["A"],
            "odds_american": [150], "stake": [20.0],
            "settled_timestamp": [None], "actual_margin": [None],
            "covered": [None], "profit_units": [None],
        })
        log_data.to_csv(log_path, index=False)
        _make_scores_csv(str(scores_path), [
            {"year": 2025, "week": 8, "game_id": "g1", "home_points": 30, "away_points": 20},
        ])

        settle_week(log_path, str(scores_path), 2025, 8)
        df = pd.read_csv(log_path)
        assert df.iloc[0]["profit_units"] == pytest.approx(-20.0)
        assert df.iloc[0]["covered"] == "L"


# ---- 6. Sigma artifact fallback triggers warning ----

def test_sigma_fallback_on_missing(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        inputs_path = Path(tmp) / "inputs.csv"
        _make_inputs_csv(str(inputs_path), _sample_events())

        a, b = run_recommend(
            year=2025, week=8,
            inputs_path=str(inputs_path),
            dry_run=True,
            sigma_artifact_path=Path(tmp) / "nonexistent.json",
        )
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "fallback" in captured.out


# ---- 7. End-to-end fake slate integration ----

def test_end_to_end_fake_slate():
    with tempfile.TemporaryDirectory() as tmp:
        inputs_path = Path(tmp) / "inputs.csv"
        log_path = Path(tmp) / "ml_bets_2025.csv"

        # 3 games: game 1 big disagree + good odds => List A
        # game 2: big disagree + expensive odds => List B
        # game 3: small disagree => excluded
        events = [
            {
                "year": 2025, "week": 8, "game_id": "g1",
                "home_team": "Alabama", "away_team": "Tennessee",
                "model_spread": 10.0, "market_spread": -3.0,
                "ml_odds_home": -130, "ml_odds_away": 110,
            },
            {
                "year": 2025, "week": 8, "game_id": "g2",
                "home_team": "Ohio State", "away_team": "Penn State",
                "model_spread": 7.0, "market_spread": -4.0,
                "ml_odds_home": -250, "ml_odds_away": 200,
            },
            {
                "year": 2025, "week": 8, "game_id": "g3",
                "home_team": "Georgia", "away_team": "Florida",
                "model_spread": 3.0, "market_spread": 1.0,
                "ml_odds_home": -200, "ml_odds_away": 170,
            },
        ]
        _make_inputs_csv(str(inputs_path), events)

        # Write sigma artifact
        sigma_path = Path(tmp) / "sigma.json"
        save_sigma_artifact(13.5, [2022, 2023, 2024, 2025], 4, sigma_path)

        a, b = run_recommend(
            year=2025, week=8,
            inputs_path=str(inputs_path),
            log_path=str(log_path),
            dry_run=False,
            sigma_artifact_path=sigma_path,
        )

        assert log_path.exists()
        log_df = pd.read_csv(log_path)

        # Verify columns present
        assert "ev" in log_df.columns
        assert "list_type" in log_df.columns
        assert "run_timestamp" in log_df.columns

        # g1 should be List A (big disagree, favorable odds)
        list_a_rows = log_df[log_df["list_type"] == "A"]
        list_b_rows = log_df[log_df["list_type"] == "B"]
        assert len(list_a_rows) >= 1
        # g3 should not appear at all (gate failure)
        assert "g3" not in log_df["game_id"].values


# ---- 8. Sigma artifact round-trip ----

def test_sigma_artifact_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sigma.json"
        save_sigma_artifact(13.42, [2022, 2023], 4, path)
        loaded = load_sigma_artifact(path)
        assert loaded == pytest.approx(13.42)

        with open(path) as f:
            data = json.load(f)
        assert data["train_years"] == [2022, 2023]
        assert data["week_min"] == 4
