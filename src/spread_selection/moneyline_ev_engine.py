"""
NCAA FBS Moneyline EV Engine
============================

Standalone moneyline expected-value engine using JP+ spread predictions as the
core signal.  Converts a predicted home margin (model_spread) into win
probabilities via a Normal CDF, then evaluates moneyline odds for +EV
opportunities.

**Default mode (disagreement-gated):**
    Games must show >= min_disagreement_pts between JP+ and the market spread
    to be considered.  This targets the highest-conviction spots where the
    model and market materially disagree.

**General-purpose EV-only mode:**
    Set ``listA_gate_logic="OR"`` with ``require_flip=False`` to evaluate every
    game purely on EV without disagreement gating.

Sign conventions (consistent with JP+ codebase):
    model_spread > 0  =>  JP+ favors home
    market_spread > 0 =>  market favors home
    (home_points - away_points)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Normal CDF — scipy preferred, pure-Python fallback
# ---------------------------------------------------------------------------
try:
    from scipy.stats import norm as _norm

    def _phi(x: float) -> float:
        return float(_norm.cdf(x))

except ImportError:

    def _phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Column schema (single source of truth)
# ---------------------------------------------------------------------------
COLUMNS: list[str] = [
    "year", "week", "game_id", "home_team", "away_team",
    "model_spread", "market_spread", "disagreement_pts", "flip_flag",
    "model_fav", "market_fav",
    "side",
    "odds_american", "odds_decimal", "b",
    "p_win", "implied_prob", "edge_prob",
    "ev",
    "kelly_f", "stake",
    "reason_code",
]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MoneylineEVConfig:
    margin_sigma: float  # required — no default
    bankroll: float = 1000.0
    ev_min: float = 0.02
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.02
    min_bet: float = 0.0
    round_to: float = 1.0
    rounding_mode: Literal["floor", "nearest"] = "floor"
    one_bet_per_game: bool = True
    max_bets_per_week: int | None = None
    require_flip: bool = False
    min_disagreement_pts: float = 5.0
    listA_gate_logic: Literal["OR", "AND"] = "AND"
    include_missing_odds_in_listB: bool = False
    include_positive_ev_loser_in_listB: bool = False
    weekly_cap_to_listB: bool = True


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------
def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal.  Raises on odds == 0."""
    if odds == 0:
        raise ValueError("American odds of 0 are undefined")
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0 + odds / 100.0


def implied_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


# ---------------------------------------------------------------------------
# Per-side evaluation (internal)
# ---------------------------------------------------------------------------
def _eval_side(
    p_win: float,
    odds_american: int,
    config: MoneylineEVConfig,
) -> dict:
    d = american_to_decimal(odds_american)
    b = d - 1.0
    ev = p_win * b - (1.0 - p_win)
    imp = implied_prob(d)
    edge_prob = p_win - imp

    # Kelly
    f_star = (p_win * b - (1.0 - p_win)) / b if b > 0 else 0.0
    f = max(0.0, config.kelly_fraction * f_star)
    stake_raw = config.bankroll * min(f, config.max_bet_fraction)

    # Rounding
    if config.round_to > 0:
        if config.rounding_mode == "floor":
            stake = math.floor(stake_raw / config.round_to) * config.round_to
        else:
            stake = round(stake_raw / config.round_to) * config.round_to
    else:
        stake = stake_raw

    if stake < config.min_bet:
        stake = 0.0

    return {
        "odds_american": odds_american,
        "odds_decimal": d,
        "b": b,
        "p_win": p_win,
        "implied_prob": imp,
        "edge_prob": edge_prob,
        "ev": ev,
        "kelly_f": f,
        "stake": stake,
    }


# ---------------------------------------------------------------------------
# Disagreement / flip logic
# ---------------------------------------------------------------------------
def _fav_label(spread: float) -> str:
    if spread > 0:
        return "HOME"
    elif spread < 0:
        return "AWAY"
    return "PICK"


def _compute_gate(
    model_spread: float,
    market_spread: float,
    config: MoneylineEVConfig,
) -> tuple[bool, float, bool, str, str]:
    model_fav = _fav_label(model_spread)
    market_fav = _fav_label(market_spread)
    flip_flag = (
        model_fav != "PICK"
        and market_fav != "PICK"
        and model_fav != market_fav
    )
    disagreement_pts = abs(model_spread - market_spread)

    flip_check = flip_flag if config.require_flip else True
    disagree_check = disagreement_pts >= config.min_disagreement_pts

    if config.listA_gate_logic == "OR":
        base_gate = flip_check or disagree_check
    else:
        base_gate = flip_check and disagree_check

    return base_gate, disagreement_pts, flip_flag, model_fav, market_fav


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
def evaluate_moneylines(
    events: list[dict],
    config: MoneylineEVConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate moneyline EV for a list of game events.

    Returns (list_a_df, list_b_df) — both with the full COLUMNS schema,
    even when empty.
    """
    list_a_rows: list[dict] = []
    list_b_rows: list[dict] = []

    for ev in events:
        year = ev["year"]
        week = ev["week"]
        game_id = ev["game_id"]
        home_team = ev["home_team"]
        away_team = ev["away_team"]
        model_spread = float(ev["model_spread"])
        market_spread = float(ev["market_spread"])
        ml_home = ev.get("ml_odds_home")
        ml_away = ev.get("ml_odds_away")

        base_gate, disagreement_pts, flip_flag, model_fav, market_fav = (
            _compute_gate(model_spread, market_spread, config)
        )

        base_row = {
            "year": year,
            "week": week,
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "model_spread": model_spread,
            "market_spread": market_spread,
            "disagreement_pts": disagreement_pts,
            "flip_flag": flip_flag,
            "model_fav": model_fav,
            "market_fav": market_fav,
        }

        # --- Missing odds handling ---
        if ml_home is None or ml_away is None:
            if config.include_missing_odds_in_listB and base_gate:
                row = {c: None for c in COLUMNS}
                row.update(base_row)
                row["side"] = None
                row["reason_code"] = "MISSING_ODDS"
                list_b_rows.append(row)
            continue

        # --- Win probabilities ---
        p_home = _phi(model_spread / config.margin_sigma)
        p_away = 1.0 - p_home

        # --- Evaluate both sides ---
        home_eval = _eval_side(p_home, int(ml_home), config)
        away_eval = _eval_side(p_away, int(ml_away), config)

        home_eval["side"] = "HOME"
        away_eval["side"] = "AWAY"

        # --- One-bet-per-game selection ---
        if config.one_bet_per_game:
            if home_eval["ev"] >= away_eval["ev"]:
                winner, loser = home_eval, away_eval
            else:
                winner, loser = away_eval, home_eval
            candidates = [winner]

            # Optionally surface the loser in List B
            if (
                config.include_positive_ev_loser_in_listB
                and loser["ev"] >= config.ev_min
                and base_gate
            ):
                row = {c: None for c in COLUMNS}
                row.update(base_row)
                row.update(loser)
                row["reason_code"] = "ONE_BET_CONSTRAINT"
                list_b_rows.append(row)
        else:
            candidates = [home_eval, away_eval]

        # --- Gate + EV/stake classification ---
        for cand in candidates:
            if not base_gate:
                continue  # excluded entirely

            row = {c: None for c in COLUMNS}
            row.update(base_row)
            row.update(cand)

            if cand["ev"] >= config.ev_min and cand["stake"] > 0:
                row["reason_code"] = "OK"
                list_a_rows.append(row)
            elif cand["ev"] < config.ev_min:
                row["reason_code"] = "EV_BELOW_MIN"
                list_b_rows.append(row)
            else:
                # stake == 0 (rounded below min_bet)
                row["reason_code"] = "STAKE_ZERO"
                list_b_rows.append(row)

    # --- Build preliminary List A ---
    list_a_df = pd.DataFrame(list_a_rows, columns=COLUMNS) if list_a_rows else _empty_df()
    list_b_df = pd.DataFrame(list_b_rows, columns=COLUMNS) if list_b_rows else _empty_df()

    # --- Weekly cap ---
    if config.max_bets_per_week is not None and len(list_a_df) > config.max_bets_per_week:
        list_a_df = list_a_df.sort_values(
            by=["ev", "stake", "game_id", "side"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)

        keep = list_a_df.iloc[: config.max_bets_per_week].copy()
        overflow = list_a_df.iloc[config.max_bets_per_week :].copy()

        if config.weekly_cap_to_listB:
            overflow["reason_code"] = "WEEKLY_CAP_EXCEEDED"
            list_b_df = pd.concat([list_b_df, overflow], ignore_index=True)

        list_a_df = keep.reset_index(drop=True)

    # --- Deterministic sort ---
    if not list_a_df.empty:
        list_a_df = list_a_df.sort_values(
            by=["ev", "stake", "game_id", "side"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)

    if not list_b_df.empty:
        list_b_df = list_b_df.sort_values(
            by=["reason_code", "game_id", "side"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    return list_a_df, list_b_df


# ---------------------------------------------------------------------------
# Sigma calibration helper
# ---------------------------------------------------------------------------
def estimate_margin_sigma_from_backtest(
    df: pd.DataFrame,
    week_min: int = 4,
) -> float:
    """Estimate margin sigma from backtest residuals.

    Parameters
    ----------
    df : DataFrame with columns ``actual_margin``, ``model_spread``, ``week``.
    week_min : Exclude weeks < this value (prior-heavy early weeks inflate sigma).

    Returns
    -------
    float : sample standard deviation (ddof=1) of residuals.
    """
    filtered = df[df["week"] >= week_min].copy()
    if len(filtered) < 30:
        raise ValueError(
            f"Insufficient rows ({len(filtered)}) after filtering week >= {week_min}. "
            "Need at least 30."
        )
    residuals = filtered["actual_margin"] - filtered["model_spread"]
    return float(residuals.std(ddof=1))


# ---------------------------------------------------------------------------
# Smoke example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = MoneylineEVConfig(
        margin_sigma=13.5,
        listA_gate_logic="AND",
        require_flip=False,
        min_disagreement_pts=5.0,
        ev_min=0.02,
        one_bet_per_game=True,
    )

    events = [
        # Game 1a: flip + disagreement, expensive ML => neg EV => List B
        {
            "year": 2025, "week": 8, "game_id": "401635901",
            "home_team": "Alabama", "away_team": "Tennessee",
            "model_spread": 7.0, "market_spread": -4.0,
            "ml_odds_home": -250, "ml_odds_away": 200,
        },
        # Game 1b: flip + disagreement, favorable ML => pos EV => List A
        {
            "year": 2025, "week": 8, "game_id": "401635902",
            "home_team": "Ohio State", "away_team": "Penn State",
            "model_spread": 7.0, "market_spread": -4.0,
            "ml_odds_home": -180, "ml_odds_away": 155,
        },
        # Game 2: small disagreement (2 pts < 5) => excluded entirely
        {
            "year": 2025, "week": 8, "game_id": "401635903",
            "home_team": "Georgia", "away_team": "Florida",
            "model_spread": 3.0, "market_spread": 1.0,
            "ml_odds_home": -200, "ml_odds_away": 170,
        },
    ]

    list_a, list_b = evaluate_moneylines(events, cfg)

    print("=" * 80)
    print("LIST A (Actionable Bets)")
    print("=" * 80)
    if list_a.empty:
        print("  (none)")
    else:
        for _, r in list_a.iterrows():
            print(
                f"  {r['home_team']} vs {r['away_team']} | "
                f"Side={r['side']} | EV={r['ev']:.4f} | "
                f"Stake=${r['stake']:.0f} | p_win={r['p_win']:.3f} | "
                f"odds={r['odds_american']}"
            )

    print()
    print("=" * 80)
    print("LIST B (Near-Misses / Diagnostics)")
    print("=" * 80)
    if list_b.empty:
        print("  (none)")
    else:
        for _, r in list_b.iterrows():
            print(
                f"  {r['home_team']} vs {r['away_team']} | "
                f"Side={r['side']} | EV={r['ev'] if pd.notna(r['ev']) else 'N/A'} | "
                f"Reason={r['reason_code']}"
            )
