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

import numpy as np
import pandas as pd
from scipy.stats import norm

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


def synthesize_ml_odds_from_spread(spread_internal: float, sigma: float = 13.5) -> tuple[int, int]:
    """Synthesize American ML odds from an opening spread using Normal CDF.

    Converts spread to win probability, then to vigorous American odds
    (standard -110/-110 vig structure applied proportionally).

    Args:
        spread_internal: Opening spread in internal convention (positive = home favored)
        sigma: Margin sigma for CDF conversion

    Returns:
        (home_ml_american, away_ml_american)
    """
    p_home = norm.cdf(spread_internal / sigma)
    p_away = 1.0 - p_home

    def _prob_to_american_with_vig(p: float) -> int:
        """Convert fair probability to American odds with ~4.5% total vig."""
        # Apply vig: multiply fair prob by ~1.045 (standard -110/-110 overround)
        p_vig = min(p * 1.045, 0.99)  # cap to avoid extreme odds
        p_vig = max(p_vig, 0.01)
        if p_vig >= 0.5:
            return int(round(-p_vig / (1.0 - p_vig) * 100))
        else:
            return int(round((1.0 - p_vig) / p_vig * 100))

    home_ml = _prob_to_american_with_vig(p_home)
    away_ml = _prob_to_american_with_vig(p_away)
    return home_ml, away_ml


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

    # --- ML odds: synthesize from opening spread ---
    # CFBD only provides closing ML odds, but we need opening ML odds
    # to be consistent with opening spread disagreement. Synthesize from
    # the opening spread using Normal CDF + standard vig.
    # Use MARKET sigma (15.4) not model sigma (13.5) — we're estimating
    # what the market would price, not our model's win probability.
    MARKET_SIGMA = 15.4
    MODEL_SIGMA = 13.5
    print(f"Synthesizing ML odds from opening spreads (market_sigma={MARKET_SIGMA})")

    # --- Run engine + populate logs ---
    config = MoneylineEVConfig(margin_sigma=MODEL_SIGMA)

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

                # Synthesize opening ML odds from opening spread
                if pd.notna(row["spread_open"]):
                    spread_open_internal = -float(row["spread_open"])  # Vegas→internal
                    home_ml, away_ml = synthesize_ml_odds_from_spread(spread_open_internal, MARKET_SIGMA)
                else:
                    home_ml, away_ml = None, None

                events.append({
                    "year": year,
                    "week": week,
                    "game_id": gid,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "model_spread": float(row["predicted_spread"]),
                    "market_spread": spread_open_internal if pd.notna(row["spread_open"]) else 0.0,
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

    # --- Consolidate per-year logs into a single backtest artifact ---
    output_dir = Path("data/moneyline_selection/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    consolidated_path = output_dir / "backtest_moneyline_2022-2025.csv"

    year_dfs = []
    for year in YEARS:
        log_path = DEFAULT_LOG_DIR / f"moneyline_bets_{year}.csv"
        if log_path.exists():
            year_dfs.append(pd.read_csv(log_path))
    if year_dfs:
        consolidated = pd.concat(year_dfs, ignore_index=True)
        consolidated.to_csv(consolidated_path, index=False)
        print(f"\n  Consolidated artifact: {consolidated_path} ({len(consolidated)} rows)")

    print(f"\n{'='*60}")
    print(f"  BACKFILL COMPLETE")
    print(f"  Total games: {total_games} ({total_with_odds} with ML odds)")
    print(f"  Total List A: {total_list_a}")
    print(f"  Total List B: {total_list_b}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
