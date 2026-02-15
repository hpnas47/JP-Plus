"""
Moneyline Weekly Runner — helper module
========================================

Keeps the CLI script thin by housing all logic for:
- sigma artifact load/save
- recommendation generation with logging
- settlement
- console summary printing
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.spread_selection.moneyline_ev_engine import (
    COLUMNS as ENGINE_COLUMNS,
    MoneylineEVConfig,
    american_to_decimal,
    estimate_margin_sigma_from_backtest,
    evaluate_moneylines,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_LOG_DIR = Path("data/moneyline_selection/logs")
DEFAULT_ARTIFACT_DIR = Path("data/moneyline_selection/artifacts")
DEFAULT_SIGMA_ARTIFACT = DEFAULT_ARTIFACT_DIR / "margin_sigma_2022_2025.json"

# ---------------------------------------------------------------------------
# Log columns (engine columns + metadata)
# ---------------------------------------------------------------------------
CONFIG_SNAPSHOT_COLS = [
    "margin_sigma", "ev_min", "require_flip", "gate_logic",
    "min_disagreement_pts", "one_bet_per_game", "max_bets_per_week",
    "rounding_mode", "round_to", "bankroll", "kelly_fraction",
    "max_bet_fraction", "min_bet",
]

SETTLEMENT_COLS = [
    "actual_margin", "covered", "profit_units", "settled_timestamp",
]

LOG_EXTRA_COLS = [
    "run_timestamp", "list_type", "odds_placeholder",
] + CONFIG_SNAPSHOT_COLS + SETTLEMENT_COLS

ALL_LOG_COLUMNS = ENGINE_COLUMNS + LOG_EXTRA_COLS

DEDUP_KEY = ["year", "week", "game_id", "side", "list_type"]

REQUIRED_INPUT_COLS = {
    "year", "week", "game_id", "home_team", "away_team",
    "model_spread", "market_spread", "ml_odds_home", "ml_odds_away",
}

REQUIRED_SCORES_COLS = {"year", "week", "game_id", "home_points", "away_points"}


# ---------------------------------------------------------------------------
# Sigma artifact
# ---------------------------------------------------------------------------
def load_sigma_artifact(path: str | Path = DEFAULT_SIGMA_ARTIFACT) -> float | None:
    """Load margin_sigma from JSON artifact. Returns None if missing/invalid."""
    try:
        with open(path) as f:
            data = json.load(f)
        return float(data["margin_sigma"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError, ValueError):
        return None


def save_sigma_artifact(
    sigma: float,
    train_years: list[int],
    week_min: int,
    path: str | Path = DEFAULT_SIGMA_ARTIFACT,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "margin_sigma": round(sigma, 4),
        "train_years": train_years,
        "week_min": week_min,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def fit_sigma(backtest_path: str, week_min: int = 4, artifact_out: str | None = None) -> float:
    """Load backtest CSV, compute sigma, save artifact."""
    df = pd.read_csv(backtest_path)
    sigma = estimate_margin_sigma_from_backtest(df, week_min=week_min)

    out = Path(artifact_out) if artifact_out else DEFAULT_SIGMA_ARTIFACT
    # Infer years from data
    years = sorted(df["year"].unique().tolist()) if "year" in df.columns else []
    save_sigma_artifact(sigma, years, week_min, out)

    print(f"margin_sigma = {sigma:.4f}")
    print(f"  week_min = {week_min}")
    print(f"  years = {years}")
    print(f"  artifact written to {out}")
    return sigma


# ---------------------------------------------------------------------------
# Input loading + validation
# ---------------------------------------------------------------------------
def load_inputs(path: str, year: int, week: int) -> list[dict]:
    """Load inputs CSV, filter to (year, week), validate columns, return events."""
    df = pd.read_csv(path)
    missing = REQUIRED_INPUT_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df = df[(df["year"] == year) & (df["week"] == week)].copy()
    if df.empty:
        return []

    # Convert NaN odds to None
    events = []
    for _, row in df.iterrows():
        ev = row.to_dict()
        for k in ("ml_odds_home", "ml_odds_away"):
            if pd.isna(ev.get(k)):
                ev[k] = None
            else:
                ev[k] = int(ev[k])
        events.append(ev)
    return events


def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_SCORES_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Scores CSV missing required columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------
def build_config(sigma: float, **overrides) -> MoneylineEVConfig:
    """Build MoneylineEVConfig from sigma + CLI overrides (None values ignored)."""
    kw: dict[str, Any] = {"margin_sigma": sigma}
    # Map CLI arg names to config field names
    field_map = {
        "bankroll": "bankroll",
        "ev_min": "ev_min",
        "max_bets_per_week": "max_bets_per_week",
        "min_disagreement": "min_disagreement_pts",
        "require_flip": "require_flip",
        "gate": "listA_gate_logic",
        "round_to": "round_to",
        "rounding_mode": "rounding_mode",
        "one_bet_per_game": "one_bet_per_game",
        "include_missing_odds_in_listB": "include_missing_odds_in_listB",
        "kelly_fraction": "kelly_fraction",
        "max_bet_fraction": "max_bet_fraction",
        "min_bet": "min_bet",
    }
    for cli_name, cfg_name in field_map.items():
        val = overrides.get(cli_name)
        if val is not None:
            kw[cfg_name] = val
    return MoneylineEVConfig(**kw)


# ---------------------------------------------------------------------------
# Log append with dedupe + schema evolution
# ---------------------------------------------------------------------------
def _annotate_df(
    df: pd.DataFrame,
    list_type: str,
    config: MoneylineEVConfig,
    timestamp: str,
) -> pd.DataFrame:
    """Add metadata columns to a list A or B dataframe."""
    df = df.copy()
    df["run_timestamp"] = timestamp
    df["list_type"] = list_type
    df["odds_placeholder"] = False
    df["margin_sigma"] = config.margin_sigma
    df["ev_min"] = config.ev_min
    df["require_flip"] = config.require_flip
    df["gate_logic"] = config.listA_gate_logic
    df["min_disagreement_pts"] = config.min_disagreement_pts
    df["one_bet_per_game"] = config.one_bet_per_game
    df["max_bets_per_week"] = config.max_bets_per_week
    df["rounding_mode"] = config.rounding_mode
    df["round_to"] = config.round_to
    df["bankroll"] = config.bankroll
    df["kelly_fraction"] = config.kelly_fraction
    df["max_bet_fraction"] = config.max_bet_fraction
    df["min_bet"] = config.min_bet
    # Settlement columns (empty on creation)
    for col in SETTLEMENT_COLS:
        if col not in df.columns:
            df[col] = None
    return df


def _append_to_log_impl(
    new_rows: pd.DataFrame,
    log_path: Path,
) -> tuple[int, int]:
    """Inner append logic separated for testability. Returns (appended, skipped)."""
    if new_rows.empty:
        return 0, 0

    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
        existing = pd.read_csv(log_path)
        all_cols = list(dict.fromkeys(list(existing.columns) + list(new_rows.columns)))
        existing = existing.reindex(columns=all_cols)
        new_rows = new_rows.reindex(columns=all_cols)

        def _dk(df):
            return df[DEDUP_KEY].astype(str).apply(tuple, axis=1)

        existing_keys = set(_dk(existing))
        mask = ~_dk(new_rows).isin(existing_keys)
        to_append = new_rows[mask]
        skipped = int((~mask).sum())

        if skipped > 0:
            print(f"  skipped {skipped} duplicates")
            print(
                "  NOTE: If a bet moved between List A/B due to odds changes, "
                "both rows can coexist (different list_type)."
            )
            print(
                "  NOTE: Dedup assumes consistent config. If config changed, "
                "use a different --log-path or delete that week's rows."
            )

        if to_append.empty:
            return 0, skipped

        combined = pd.concat([existing, to_append], ignore_index=True)
        appended = len(to_append)
    else:
        combined = new_rows
        skipped = 0
        appended = len(new_rows)

    combined.to_csv(log_path, index=False)
    return appended, skipped


# Rewrite append_to_log to use the clean impl
def append_to_log(
    list_a: pd.DataFrame,
    list_b: pd.DataFrame,
    config: MoneylineEVConfig,
    log_path: str | Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Append annotated rows to CSV log with dedupe.

    Returns (rows_appended, rows_skipped).
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    a_ann = _annotate_df(list_a, "A", config, timestamp)
    b_ann = _annotate_df(list_b, "B", config, timestamp)
    new_rows = pd.concat([a_ann, b_ann], ignore_index=True)

    if new_rows.empty:
        return 0, 0

    log_p = Path(log_path)

    # Warn if config differs from previous run for the same (year, week)
    if log_p.exists() and not new_rows.empty:
        existing = pd.read_csv(log_p)
        sample_row = new_rows.iloc[0]
        yr, wk = sample_row.get("year"), sample_row.get("week")
        prev = existing[(existing["year"] == yr) & (existing["week"] == wk)]
        if not prev.empty:
            for col, curr_val in [
                ("margin_sigma", config.margin_sigma),
                ("ev_min", config.ev_min),
                ("gate_logic", config.listA_gate_logic),
            ]:
                if col in prev.columns:
                    prev_val = prev[col].iloc[0]
                    if pd.notna(prev_val) and prev_val != curr_val:
                        print(
                            f"  WARNING: Config differs from previous run for year={yr} week={wk}. "
                            f"{col}: previous={prev_val}, current={curr_val}. "
                            f"New bets matching existing (game_id, side, list_type) will be deduped. "
                            f"Use --log-path to write to a separate file if this is intentional."
                        )
                        break

    if dry_run:
        # Run dedup logic to give accurate preview, but don't write
        if log_p.exists():
            existing = pd.read_csv(log_p)
            def _dk(df):
                return df[DEDUP_KEY].astype(str).apply(tuple, axis=1)
            existing_keys = set(_dk(existing))
            mask = ~_dk(new_rows).isin(existing_keys)
            would_append = int(mask.sum())
            would_skip = int((~mask).sum())
            return would_append, would_skip
        return len(new_rows), 0

    return _append_to_log_impl(new_rows, log_p)


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------
def settle_week(
    log_path: str | Path,
    scores_path: str,
    year: int,
    week: int,
) -> tuple[int, int, int]:
    """Settle List A bets for (year, week).

    Settlement loads the full CSV into memory and rewrites the full file.
    This is not concurrent-safe with a simultaneous recommend append.
    Assume these are run sequentially.

    Returns (settled_count, warnings_count, already_settled_count).
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    log = pd.read_csv(log_path)
    # Ensure settlement columns exist (handles logs from older schema versions)
    for col in SETTLEMENT_COLS:
        if col not in log.columns:
            log[col] = None
    # Ensure settlement columns are object-typed to accept mixed str/NaN
    for col in ("covered", "settled_timestamp"):
        log[col] = log[col].astype(object)
    scores = load_scores(scores_path)

    # Filter to target rows: List A, this (year, week), not yet settled
    target_mask = (
        (log["year"] == year)
        & (log["week"] == week)
        & (log["list_type"] == "A")
        & (log["settled_timestamp"].isna())
    )
    already_settled = (
        (log["year"] == year)
        & (log["week"] == week)
        & (log["list_type"] == "A")
        & (log["settled_timestamp"].notna())
    ).sum()

    target_idx = log[target_mask].index
    if len(target_idx) == 0:
        return 0, 0, int(already_settled)

    # Merge scores — normalize game_id to string without float suffix
    scores_week = scores[(scores["year"] == year) & (scores["week"] == week)]
    def _norm_gid(gid) -> str:
        s = str(gid)
        return s.split(".")[0] if "." in s else s
    scores_map = {
        _norm_gid(row["game_id"]): (row["home_points"], row["away_points"])
        for _, row in scores_week.iterrows()
    }

    settled = 0
    warn_count = 0
    ts = datetime.now(timezone.utc).isoformat()

    for idx in target_idx:
        row = log.loc[idx]
        gid = _norm_gid(row["game_id"])
        try:
            if gid not in scores_map:
                print(f"  WARNING: game_id={gid} ({row.get('home_team', '?')} vs {row.get('away_team', '?')}) not found in scores file — skipping")
                warn_count += 1
                continue

            hp, ap = scores_map[gid]
            actual_margin = hp - ap

            # In FBS, margin == 0 should not occur
            if actual_margin == 0:
                warnings.warn(
                    f"actual_margin==0 for game_id={gid} (year={year} week={week}). "
                    "Treating as data error; row left unsettled."
                )
                warn_count += 1
                continue

            side = row["side"]
            if side == "HOME":
                covered = "W" if actual_margin > 0 else "L"
            elif side == "AWAY":
                covered = "W" if actual_margin < 0 else "L"
            else:
                print(f"  WARNING: game_id={gid} has invalid side='{side}' — skipping")
                warn_count += 1
                continue

            odds_am = row["odds_american"]
            if pd.isna(odds_am):
                print(f"  WARNING: game_id={gid} has NaN odds_american — skipping")
                warn_count += 1
                continue
            d = american_to_decimal(int(odds_am))

            stake_val = row["stake"]
            if pd.isna(stake_val):
                print(f"  WARNING: game_id={gid} has NaN stake — skipping")
                warn_count += 1
                continue
            stake = float(stake_val)

            profit = (d - 1.0) * stake if covered == "W" else -stake

            log.at[idx, "actual_margin"] = actual_margin
            log.at[idx, "covered"] = covered
            log.at[idx, "profit_units"] = round(profit, 2)
            log.at[idx, "settled_timestamp"] = ts
            settled += 1
        except Exception as e:
            print(f"  WARNING: game_id={gid} settlement failed: {e} — skipping")
            warn_count += 1
            continue

    log.to_csv(log_path, index=False)
    return settled, warn_count, int(already_settled)


# ---------------------------------------------------------------------------
# Console summaries
# ---------------------------------------------------------------------------
def print_recommend_summary(
    year: int,
    week: int,
    n_games: int,
    n_with_odds: int,
    config: MoneylineEVConfig,
    list_a: pd.DataFrame,
    list_b: pd.DataFrame,
) -> None:
    print(f"\n{'='*70}")
    print(f"  MONEYLINE EV — Week {week}, {year}")
    print(f"{'='*70}")
    print(f"  Games loaded:     {n_games}")
    print(f"  Games w/ ML odds: {n_with_odds}")
    print(f"  Config:")
    print(f"    sigma={config.margin_sigma:.2f}  ev_min={config.ev_min:.3f}")
    print(f"    gate={config.listA_gate_logic}  min_disagree={config.min_disagreement_pts}")
    print(f"    require_flip={config.require_flip}  one_bet_per_game={config.one_bet_per_game}")
    print(f"    max_bets_per_week={config.max_bets_per_week}")
    print()

    if list_a.empty:
        print("  List A: 0 bets")
    else:
        home_ct = (list_a["side"] == "HOME").sum()
        away_ct = (list_a["side"] == "AWAY").sum()
        print(f"  List A: {len(list_a)} bet(s)")
        print(f"    Total stake:  ${list_a['stake'].sum():.0f}")
        print(f"    Avg EV:       {list_a['ev'].mean():.4f}")
        print(f"    Avg disagree: {list_a['disagreement_pts'].mean():.1f} pts")
        print(f"    Home/Away:    {home_ct}/{away_ct}")
        print()
        print("  Top bets:")
        for _, r in list_a.head(5).iterrows():
            print(
                f"    {r['home_team']:20s} vs {r['away_team']:20s} | "
                f"Side={r['side']:4s} | EV={r['ev']:.4f} | "
                f"Stake=${r['stake']:.0f} | odds={int(r['odds_american'])}"
            )

    print(f"\n  List B: {len(list_b)} row(s)")
    if not list_b.empty:
        reasons = list_b["reason_code"].value_counts().to_dict()
        for rc, ct in reasons.items():
            print(f"    {rc}: {ct}")
    print(f"{'='*70}\n")


def print_settle_summary(
    year: int,
    week: int,
    log_path: str | Path,
    settled: int,
    warn_count: int,
    already_settled: int,
) -> None:
    print(f"\n{'='*70}")
    print(f"  MONEYLINE SETTLEMENT — Week {week}, {year}")
    print(f"{'='*70}")
    print(f"  Settled this run: {settled}")
    print(f"  Already settled:  {already_settled}")
    print(f"  Warnings:         {warn_count}")

    # Load log and compute summary for settled rows
    log = pd.read_csv(log_path)
    settled_rows = log[
        (log["year"] == year)
        & (log["week"] == week)
        & (log["list_type"] == "A")
        & (log["settled_timestamp"].notna())
    ]

    if settled_rows.empty:
        print("  No settled bets to summarize.")
    else:
        wins = (settled_rows["covered"] == "W").sum()
        losses = (settled_rows["covered"] == "L").sum()
        total_stake = settled_rows["stake"].sum()
        total_profit = settled_rows["profit_units"].sum()
        roi = (total_profit / total_stake * 100) if total_stake > 0 else 0.0
        print(f"  Record:           {wins}W - {losses}L")
        print(f"  Total stake:      ${total_stake:.0f}")
        print(f"  Total profit:     ${total_profit:.2f}")
        print(f"  ROI:              {roi:.1f}%")
        if already_settled > 0 and settled > 0:
            print(f"  Note: W-L and ROI above are cumulative for the week (includes {already_settled} previously settled bets).")

    unsettled = log[
        (log["year"] == year)
        & (log["week"] == week)
        & (log["list_type"] == "A")
        & (log["settled_timestamp"].isna())
    ]
    print(f"  Unresolved:       {len(unsettled)}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Orchestrator (called by CLI)
# ---------------------------------------------------------------------------
def run_recommend(
    year: int,
    week: int,
    inputs_path: str,
    log_path: str | Path | None = None,
    dry_run: bool = False,
    sigma_artifact_path: str | Path = DEFAULT_SIGMA_ARTIFACT,
    **config_overrides,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full recommend pipeline. Returns (list_a, list_b)."""
    # Sigma
    sigma = load_sigma_artifact(sigma_artifact_path)
    if sigma is None:
        sigma = 13.5
        print(f"  WARNING: sigma artifact not found at {sigma_artifact_path}; using fallback {sigma}")

    config = build_config(sigma, **config_overrides)

    # Load events
    events = load_inputs(inputs_path, year, week)
    n_games = len(events)

    if n_games == 0:
        print(f"  No games found for year={year} week={week} in {inputs_path}")
        return pd.DataFrame(columns=ENGINE_COLUMNS), pd.DataFrame(columns=ENGINE_COLUMNS)

    n_with_odds = sum(1 for e in events if e.get("ml_odds_home") is not None and e.get("ml_odds_away") is not None)

    if n_with_odds == 0:
        print(f"  NO MONEYLINE ODDS AVAILABLE for year={year} week={week}; nothing to do.")
        return pd.DataFrame(columns=ENGINE_COLUMNS), pd.DataFrame(columns=ENGINE_COLUMNS)

    list_a, list_b = evaluate_moneylines(events, config)

    # Log path
    if log_path is None:
        log_path = DEFAULT_LOG_DIR / f"moneyline_bets_{year}.csv"

    appended, skipped = append_to_log(list_a, list_b, config, log_path, dry_run=dry_run)
    if not dry_run:
        print(f"  Wrote {appended} rows to {log_path}" + (f" (skipped {skipped} dupes)" if skipped else ""))
    else:
        print(f"  DRY RUN: would write {appended} rows")

    print_recommend_summary(year, week, n_games, n_with_odds, config, list_a, list_b)
    return list_a, list_b


def run_settle(
    year: int,
    week: int,
    scores_path: str,
    log_path: str | Path | None = None,
) -> None:
    if log_path is None:
        log_path = DEFAULT_LOG_DIR / f"moneyline_bets_{year}.csv"

    settled, warn_count, already = settle_week(log_path, scores_path, year, week)
    print_settle_summary(year, week, log_path, settled, warn_count, already)
