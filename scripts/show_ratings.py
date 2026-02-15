#!/usr/bin/env python3
"""Generate and display JP+ power ratings for a given year.

Usage:
    python3 scripts/show_ratings.py 2025                # Top 25 (from cache)
    python3 scripts/show_ratings.py 2025 50              # Top 50
    python3 scripts/show_ratings.py 2025 all             # All FBS teams
    python3 scripts/show_ratings.py 2025 --refresh       # Force recompute + cache
    python3 scripts/show_ratings.py 2025 --week 10       # Ratings through Week 10
    python3 scripts/show_ratings.py 2025 25 --week 14    # Top 25 through Week 14

Reads from cached CSV when available. Use --refresh to recompute.
Uses src.ratings.generate (single source of truth) for all computation.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.ratings.generate import generate_ratings

logging.basicConfig(level=logging.WARNING)

CACHE_DIR = Path(__file__).parent.parent / "data" / "ratings"


def _cache_path(year: int, week: int | None = None) -> Path:
    if week is not None:
        return CACHE_DIR / f"ratings_{year}_w{week}.csv"
    return CACHE_DIR / f"ratings_{year}.csv"


def compute_and_cache(year: int, week: int | None = None) -> pd.DataFrame:
    """Compute ratings via canonical generate_ratings() and save to cache CSV."""
    label = f"{year} Week {week}" if week else f"{year} full season"
    print(f"Computing {label} ratings...", file=sys.stderr)

    ratings, meta = generate_ratings(year, week=week)

    # Convert to DataFrame for caching and display
    ratings_df = pd.DataFrame(ratings).rename(columns={"st": "special_teams"})
    ratings_df = ratings_df.sort_values("overall", ascending=False).reset_index(drop=True)

    computed_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    ratings_df.attrs["n_regular"] = meta["n_regular"]
    ratings_df.attrs["n_postseason"] = meta["n_postseason"]
    ratings_df.attrs["n_fbs"] = meta["n_fbs"]
    ratings_df.attrs["computed_at"] = computed_at
    ratings_df.attrs["through_week"] = week or "all"

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_df = ratings_df.copy()
    cache_df["n_regular"] = meta["n_regular"]
    cache_df["n_postseason"] = meta["n_postseason"]
    cache_df["n_fbs"] = meta["n_fbs"]
    cache_df["computed_at"] = computed_at
    cache_df["through_week"] = week or "all"
    cache_df.to_csv(_cache_path(year, week), index=False)
    print(f"Cached to {_cache_path(year, week)}", file=sys.stderr)

    return ratings_df


def show_ratings(year: int, top_n: int | None = 25, refresh: bool = False, week: int | None = None):
    cache = _cache_path(year, week)

    if not refresh and cache.exists():
        ratings_df = pd.read_csv(cache)
        n_regular = int(ratings_df["n_regular"].iloc[0])
        n_postseason = int(ratings_df["n_postseason"].iloc[0])
        n_fbs = int(ratings_df["n_fbs"].iloc[0])
        computed_at = str(ratings_df["computed_at"].iloc[0])
        through_week = ratings_df["through_week"].iloc[0]
        ratings_df = ratings_df.drop(columns=["n_regular", "n_postseason", "n_fbs", "computed_at", "through_week"])
    else:
        ratings_df = compute_and_cache(year, week)
        n_regular = ratings_df.attrs.get("n_regular", 0)
        n_postseason = ratings_df.attrs.get("n_postseason", 0)
        n_fbs = ratings_df.attrs.get("n_fbs", 0)
        computed_at = ratings_df.attrs.get("computed_at", "")
        through_week = ratings_df.attrs.get("through_week", "all")

    if top_n is not None:
        ratings_df = ratings_df.head(top_n)

    # Build header
    if week is not None:
        title = f"{year} JP+ Power Ratings — Through Week {through_week}"
    else:
        title = f"{year} JP+ Power Ratings — Full Season"
    print(f"\n## {title} (Top {len(ratings_df)})\n")

    print("| Rank | Team | Overall | Offense | Defense | Special Teams |")
    print("|------|------|---------|---------|---------|---------------|")

    for _, row in ratings_df.iterrows():
        print(
            f"| {int(row['rank'])} "
            f"| {row['team']} "
            f"| {row['overall']:+.1f} "
            f"| {row['offense']:+.1f} ({int(row['off_rank'])}) "
            f"| {row['defense']:+.1f} ({int(row['def_rank'])}) "
            f"| {row['special_teams']:+.2f} ({int(row['st_rank'])}) |"
        )

    games_note = f"{n_regular} regular season"
    if n_postseason > 0:
        games_note += f" + {n_postseason} postseason"
    print(f"\n*{n_fbs} FBS teams. {games_note} games. Updated {computed_at}.*")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/show_ratings.py <year> [top_n|all] [--week N] [--refresh]")
        sys.exit(1)

    refresh = "--refresh" in sys.argv
    week_val = None
    args_clean = []
    skip_next = False
    for i, a in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if a == "--refresh":
            continue
        if a == "--week" and i < len(sys.argv) - 1:
            week_val = int(sys.argv[i + 1])
            skip_next = True
            continue
        args_clean.append(a)

    year = int(args_clean[0])
    top_n_arg = args_clean[1] if len(args_clean) > 1 else "25"

    if top_n_arg.lower() == "all":
        top_n = None
    else:
        top_n = int(top_n_arg)

    show_ratings(year, top_n, refresh=refresh, week=week_val)
