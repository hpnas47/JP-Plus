"""Unit tests for moneyline_ev_engine."""

import math

import pandas as pd
import pytest

from src.spread_selection.moneyline_ev_engine import (
    COLUMNS,
    MoneylineEVConfig,
    american_to_decimal,
    evaluate_moneylines,
    estimate_margin_sigma_from_backtest,
    _phi,
)


def _cfg(**kw):
    return MoneylineEVConfig(margin_sigma=13.5, **kw)


# ---- american_to_decimal ----

def test_american_to_decimal_negative():
    assert american_to_decimal(-200) == pytest.approx(1.5)


def test_american_to_decimal_positive():
    assert american_to_decimal(150) == pytest.approx(2.5)


def test_american_to_decimal_zero_raises():
    with pytest.raises(ValueError):
        american_to_decimal(0)


# ---- EV sanity: pick'em at -110/-110 => negative EV both sides ----

def test_pickem_symmetric_110_negative_ev():
    events = [{
        "year": 2025, "week": 5, "game_id": "g1",
        "home_team": "A", "away_team": "B",
        "model_spread": 0.0, "market_spread": 0.0,
        "ml_odds_home": -110, "ml_odds_away": -110,
    }]
    # disagreement=0 < 5 => gate fails, but let's test with gate disabled
    cfg = _cfg(min_disagreement_pts=0.0)
    a, b = evaluate_moneylines(events, cfg)
    # p=0.5, d=1.909, b=0.909 => EV = 0.5*0.909 - 0.5 = -0.045
    assert a.empty
    assert not b.empty
    assert (b["ev"] < 0).all()


# ---- Flip detection ----

def test_flip_detected():
    events = [{
        "year": 2025, "week": 5, "game_id": "g1",
        "home_team": "A", "away_team": "B",
        "model_spread": 7.0, "market_spread": -4.0,
        "ml_odds_home": -180, "ml_odds_away": 155,
    }]
    cfg = _cfg()
    a, b = evaluate_moneylines(events, cfg)
    combined = pd.concat([a, b])
    assert combined.iloc[0]["flip_flag"] == True


def test_no_flip_same_side():
    events = [{
        "year": 2025, "week": 5, "game_id": "g1",
        "home_team": "A", "away_team": "B",
        "model_spread": 7.0, "market_spread": 1.0,
        "ml_odds_home": -300, "ml_odds_away": 250,
    }]
    cfg = _cfg(min_disagreement_pts=5.0)
    a, b = evaluate_moneylines(events, cfg)
    combined = pd.concat([a, b])
    assert combined.iloc[0]["flip_flag"] == False


# ---- Gating truth table ----

def _gate_event(model_spread, market_spread):
    return {
        "year": 2025, "week": 5, "game_id": "g1",
        "home_team": "A", "away_team": "B",
        "model_spread": model_spread, "market_spread": market_spread,
        "ml_odds_home": -150, "ml_odds_away": 130,
    }


def test_gate_and_default_disagree_only():
    """Default AND with require_flip=False: gate = disagree_check only."""
    # disagreement = 2 < 5 => gate fails
    cfg = _cfg(listA_gate_logic="AND", require_flip=False, min_disagreement_pts=5.0)
    a, b = evaluate_moneylines([_gate_event(3.0, 1.0)], cfg)
    assert a.empty and b.empty


def test_gate_and_with_require_flip():
    """AND with require_flip=True: need BOTH flip AND disagree."""
    # flip=False (both HOME), disagree=6 >= 5 => gate = False AND True = False
    cfg = _cfg(listA_gate_logic="AND", require_flip=True, min_disagreement_pts=5.0)
    a, b = evaluate_moneylines([_gate_event(7.0, 1.0)], cfg)
    assert a.empty and b.empty


def test_gate_or_disagree_sufficient():
    """OR: disagreement alone is sufficient even without flip."""
    cfg = _cfg(listA_gate_logic="OR", require_flip=False, min_disagreement_pts=5.0)
    a, b = evaluate_moneylines([_gate_event(7.0, 1.0)], cfg)
    assert not (a.empty and b.empty)


# ---- Weekly cap to List B ----

def test_weekly_cap_moves_to_listB():
    events = [
        {
            "year": 2025, "week": 5, "game_id": f"g{i}",
            "home_team": f"H{i}", "away_team": f"A{i}",
            "model_spread": 10.0, "market_spread": -3.0,
            "ml_odds_home": -130, "ml_odds_away": 110,
        }
        for i in range(5)
    ]
    cfg = _cfg(max_bets_per_week=2, weekly_cap_to_listB=True)
    a, b = evaluate_moneylines(events, cfg)
    cap_rows = b[b["reason_code"] == "WEEKLY_CAP_EXCEEDED"]
    assert len(a) == 2
    assert len(cap_rows) == 3


def test_weekly_cap_silent_drop():
    events = [
        {
            "year": 2025, "week": 5, "game_id": f"g{i}",
            "home_team": f"H{i}", "away_team": f"A{i}",
            "model_spread": 10.0, "market_spread": -3.0,
            "ml_odds_home": -130, "ml_odds_away": 110,
        }
        for i in range(5)
    ]
    cfg = _cfg(max_bets_per_week=2, weekly_cap_to_listB=False)
    a, b = evaluate_moneylines(events, cfg)
    assert len(a) == 2
    assert b[b["reason_code"] == "WEEKLY_CAP_EXCEEDED"].empty


# ---- Empty outputs preserve schema ----

def test_empty_output_schema():
    a, b = evaluate_moneylines([], _cfg())
    assert list(a.columns) == COLUMNS
    assert list(b.columns) == COLUMNS
    assert a.empty and b.empty


# ---- Missing odds ----

def test_missing_odds_skip_by_default():
    events = [{
        "year": 2025, "week": 5, "game_id": "g1",
        "home_team": "A", "away_team": "B",
        "model_spread": 10.0, "market_spread": -3.0,
        "ml_odds_home": None, "ml_odds_away": 200,
    }]
    a, b = evaluate_moneylines(events, _cfg())
    assert a.empty and b.empty


def test_missing_odds_in_listB_when_enabled():
    events = [{
        "year": 2025, "week": 5, "game_id": "g1",
        "home_team": "A", "away_team": "B",
        "model_spread": 10.0, "market_spread": -3.0,
        "ml_odds_home": None, "ml_odds_away": 200,
    }]
    cfg = _cfg(include_missing_odds_in_listB=True)
    a, b = evaluate_moneylines(events, cfg)
    assert a.empty
    assert len(b) == 1
    assert b.iloc[0]["reason_code"] == "MISSING_ODDS"
    assert pd.isna(b.iloc[0]["side"])


# ---- Sigma calibration ----

def test_estimate_sigma():
    df = pd.DataFrame({
        "week": [4, 5, 6, 7] * 10,
        "actual_margin": [10.0] * 40,
        "model_spread": [7.0] * 40,
    })
    sigma = estimate_margin_sigma_from_backtest(df)
    assert sigma == pytest.approx(0.0, abs=1e-10)


def test_estimate_sigma_insufficient_rows():
    df = pd.DataFrame({
        "week": [4, 5],
        "actual_margin": [10.0, 12.0],
        "model_spread": [7.0, 9.0],
    })
    with pytest.raises(ValueError, match="Insufficient"):
        estimate_margin_sigma_from_backtest(df)
