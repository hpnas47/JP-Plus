"""Persistence for bot state via JSON file."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from bot.config import STATE_FILE


def _load() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save(data: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(data, indent=2, default=str))


def record_pipeline_run(year: int, week: int, results: dict) -> None:
    data = _load()
    data["last_pipeline"] = {
        "year": year,
        "week": week,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    _save(data)


def pipeline_completed(year: int, week: int) -> bool:
    data = _load()
    last = data.get("last_pipeline", {})
    return last.get("year") == year and last.get("week") == week


def record_error(error: str) -> None:
    data = _load()
    errors = data.get("recent_errors", [])
    errors.insert(0, {"timestamp": datetime.now().isoformat(), "error": error[:500]})
    data["recent_errors"] = errors[:10]
    _save(data)


def get_status() -> dict:
    return _load()


def get_season_record(year: int) -> dict:
    """Aggregate W-L records from spread/totals/ML log CSVs."""
    import pandas as pd

    base = Path(__file__).parent.parent / "data"
    records: dict[str, dict] = {}

    # Spread bets
    spread_path = base / f"spread_selection/logs/spread_bets_{year}.csv"
    if spread_path.exists():
        df = pd.read_csv(spread_path)
        if "ats_win_open" in df.columns:
            w = int((df["ats_win_open"] == 1).sum())
            l = int(len(df) - w - (df.get("ats_push_open", 0) == 1).sum())
            records["spreads"] = {"wins": w, "losses": l, "total": len(df)}

    # Totals bets
    totals_path = base / f"spread_selection/logs/totals_bets_{year}.csv"
    if totals_path.exists():
        df = pd.read_csv(totals_path)
        if "result_open" in df.columns:
            w = int((df["result_open"] == "WIN").sum())
            l = int((df["result_open"] == "LOSS").sum())
            records["totals"] = {"wins": w, "losses": l, "total": len(df)}

    # Moneyline bets
    ml_path = base / f"moneyline_selection/logs/moneyline_bets_{year}.csv"
    if ml_path.exists():
        df = pd.read_csv(ml_path)
        if "covered" in df.columns:
            w = int((df["covered"] == "W").sum())
            l = int((df["covered"] == "L").sum())
            records["moneyline"] = {"wins": w, "losses": l, "total": len(df)}

    return records
