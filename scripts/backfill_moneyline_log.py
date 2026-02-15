#!/usr/bin/env python3
"""One-time backfill: fetch ML odds from CFBD for 2022-2025 backtest games,
cache them locally, run through the ML EV engine, and populate the moneyline
log CSVs with settled results.

The ML odds are cached to data/moneyline_selection/cache/ml_odds_2022_2025.csv
so this only hits the API once. Subsequent runs use the cache.

Usage:
    python3 scripts/backfill_moneyline_log.py
    python3 scripts/backfill_moneyline_log.py --refresh-cache  # force re-fetch
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.api.cfbd_client import CFBDClient
from src.spread_selection.moneyline_ev_engine import (
    MoneylineEVConfig,
    evaluate_moneylines,
)
from src.spread_selection.moneyline_weekly import (
    append_to_log,
    settle_week,
    DEFAULT_LOG_DIR,
)

CACHE_DIR = Path("data/moneyline_selection/cache")
CACHE_PATH = CACHE_DIR / "ml_odds_2022_2025.csv"

# CFBD provider priority (match our spread pipeline)
PROVIDER_PRIORITY = ["DraftKings", "ESPN Bet", "Bovada", "consensus"]

YEARS = [2022, 2023, 2024, 2025]


def fetch_and_cache_ml_odds(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch ML odds from CFBD for all backtest games, cache to CSV."""
    client = CFBDClient()
    rows = []

    for year in YEARS:
        year_df = backtest_df[backtest_df["year"] == year]
        weeks = sorted(year_df["week"].unique())
        print(f"  Fetching {year} ({len(weeks)} weeks)...")

        for week in weeks:
            week = int(week)
            try:
                games = client.get_betting_lines(year=year, week=week)
            except Exception as e:
                print(f"    WARNING: API error for {year} week {week}: {e}")
                continue

            for game in games:
                game_id = str(game.id)
                if not game.lines:
                    continue

                # Pick best provider with ML odds
                home_ml, away_ml, provider_used = None, None, None
                for provider in PROVIDER_PRIORITY:
                    for line in game.lines:
                        if (line.provider == provider
                                and line.home_moneyline is not None
                                and line.away_moneyline is not None):
                            home_ml = int(line.home_moneyline)
                            away_ml = int(line.away_moneyline)
                            provider_used = provider
                            break
                    if home_ml is not None:
                        break

                # Fallback: any provider
                if home_ml is None:
                    for line in game.lines:
                        if (line.home_moneyline is not None
                                and line.away_moneyline is not None):
                            home_ml = int(line.home_moneyline)
                            away_ml = int(line.away_moneyline)
                            provider_used = line.provider
                            break

                if home_ml is not None:
                    rows.append({
                        "game_id": game_id,
                        "year": year,
                        "week": week,
                        "ml_odds_home": home_ml,
                        "ml_odds_away": away_ml,
                        "provider": provider_used,
                    })

            time.sleep(0.3)  # rate limit

    cache_df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(CACHE_PATH, index=False)
    print(f"  Cached {len(cache_df)} ML odds rows to {CACHE_PATH}")
    return cache_df


def load_cached_odds() -> pd.DataFrame:
    """Load cached ML odds."""
    return pd.read_csv(CACHE_PATH, dtype={"game_id": str})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Force re-fetch ML odds from CFBD API")
    args = parser.parse_args()

    backtest_path = Path("data/spread_selection/outputs/backtest_primary_2022-2025_with_scores.csv")
    if not backtest_path.exists():
        print(f"ERROR: {backtest_path} not found")
        return 1

    df = pd.read_csv(backtest_path, dtype={"game_id": str})

    # --- Cache ML odds ---
    if args.refresh_cache or not CACHE_PATH.exists():
        print("Fetching ML odds from CFBD API...")
        odds_df = fetch_and_cache_ml_odds(df)
    else:
        print(f"Using cached ML odds from {CACHE_PATH}")
        odds_df = load_cached_odds()
        print(f"  {len(odds_df)} cached odds rows")

    # Build lookup: game_id -> (home_ml, away_ml)
    odds_map = {
        str(row["game_id"]): (int(row["ml_odds_home"]), int(row["ml_odds_away"]))
        for _, row in odds_df.iterrows()
    }

    # --- Run engine + populate logs ---
    config = MoneylineEVConfig(margin_sigma=13.5)

    total_list_a = 0
    total_list_b = 0
    total_games = 0
    total_with_odds = 0

    for year in YEARS:
        year = int(year)
        log_path = DEFAULT_LOG_DIR / f"moneyline_bets_{year}.csv"

        # Remove existing log to avoid stale data on re-run
        if log_path.exists():
            log_path.unlink()

        year_df = df[df["year"] == year]
        year_a = 0
        year_b = 0

        print(f"\n{'='*60}")
        print(f"  {year} — {len(year_df)} backtest games")
        print(f"{'='*60}")

        for week in sorted(year_df["week"].unique()):
            week = int(week)
            week_df = year_df[year_df["week"] == week]

            # Build events
            events = []
            for _, row in week_df.iterrows():
                gid = str(row["game_id"])
                home_ml, away_ml = odds_map.get(gid, (None, None))

                events.append({
                    "year": year,
                    "week": week,
                    "game_id": gid,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "model_spread": float(row["predicted_spread"]),
                    "market_spread": float(row["spread_open"]) if pd.notna(row["spread_open"]) else 0.0,
                    "ml_odds_home": home_ml,
                    "ml_odds_away": away_ml,
                })

            n_with_odds = sum(1 for e in events if e["ml_odds_home"] is not None)
            total_games += len(events)
            total_with_odds += n_with_odds

            if n_with_odds == 0:
                print(f"  Week {week:2d}: {len(events):3d} games, 0 with ML odds — skipped")
                continue

            # Run engine
            list_a, list_b = evaluate_moneylines(events, config)

            # Append to log
            appended, skipped = append_to_log(list_a, list_b, config, log_path)

            year_a += len(list_a)
            year_b += len(list_b)

            print(f"  Week {week:2d}: {len(events):3d} games, {n_with_odds:3d} ML odds | "
                  f"A={len(list_a):2d} B={len(list_b):2d} | wrote {appended}")

            # Settle with actual scores
            if not list_a.empty:
                scores_rows = []
                for _, row in week_df.iterrows():
                    if pd.notna(row["home_points"]) and pd.notna(row["away_points"]):
                        scores_rows.append({
                            "year": year,
                            "week": week,
                            "game_id": str(row["game_id"]),
                            "home_points": row["home_points"],
                            "away_points": row["away_points"],
                        })

                if scores_rows:
                    scores_path = Path(f"/tmp/ml_scores_{year}_{week}.csv")
                    pd.DataFrame(scores_rows).to_csv(scores_path, index=False)
                    settled, warns, _ = settle_week(log_path, str(scores_path), year, week)
                    if settled > 0:
                        print(f"           settled {settled} bets")

        total_list_a += year_a
        total_list_b += year_b
        print(f"\n  {year} totals: List A={year_a}, List B={year_b}")

    print(f"\n{'='*60}")
    print(f"  BACKFILL COMPLETE")
    print(f"  Total games: {total_games} ({total_with_odds} with ML odds)")
    print(f"  Total List A: {total_list_a}")
    print(f"  Total List B: {total_list_b}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
