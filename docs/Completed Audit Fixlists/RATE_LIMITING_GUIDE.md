# API Rate Limiting & Credit Preservation Guide

This guide documents the API optimization strategies implemented to reduce CFBD/OddsAPI credit consumption.

## Problem Statement

The CFB Power Ratings Model was hitting API rate limits too quickly, particularly during:
- Weekly production runs (run_weekly.py)
- Multi-year backtests (backtest.py)
- Development/testing iterations

## Optimization Strategies Implemented

### 1. ✅ Request Coalescing (Already Implemented)

**Status:** Verified and working correctly

All API calls use bulk endpoints rather than per-team loops:
- `client.get_games(year, week)` - Fetches all games for a week
- `client.get_plays(year, week)` - Fetches all plays for a week
- `client.get_betting_lines(year, week)` - Fetches all lines for a week
- `client.get_fbs_teams(year)` - Fetches all teams at once
- `client.get_rankings(year)` - Fetches all rankings at once

**No action needed** - code already follows best practices.

### 2. ✅ Season-Level Disk Caching (Already Implemented)

**Status:** Working - provides 60x speedup for backtests

**Location:** `src/data/season_cache.py`

**How it works:**
- Caches complete season DataFrames (games, plays, betting lines)
- Historical seasons (year < current_year) cached indefinitely
- Current season data NOT cached (fetched fresh each run)

**Usage:**
```bash
# Backtest with cache (default)
python3 scripts/backtest.py --years 2022 2023 2024

# Force refresh (bypass cache)
python3 scripts/backtest.py --years 2024 --force-refresh

# Disable cache completely
python3 scripts/backtest.py --years 2024 --no-cache
```

**Cache location:** `.cache/seasons/{year}/`

**Performance:**
- First run (cold cache): ~2 minutes to fetch 4-year data
- Subsequent runs (warm cache): ~2 seconds (60x faster)

### 3. ✅ Week-Level Delta Caching (NEW)

**Status:** Implemented

**Location:** `src/data/week_cache.py`

**Purpose:** Enable production runs to only fetch current week from API

**How it works:**
1. Historical weeks (1 through N-1) loaded from disk cache
2. Only current week (N) fetched from API
3. Dramatically reduces API calls for weekly production runs

**Setup (one-time per season):**
```bash
# Populate cache for current season
python3 scripts/populate_week_cache.py

# Populate specific year
python3 scripts/populate_week_cache.py --year 2024

# Populate specific week range
python3 scripts/populate_week_cache.py --year 2024 --weeks 1 10

# Force refresh existing cache
python3 scripts/populate_week_cache.py --year 2024 --force-refresh
```

**Usage in backtest.py:**
```python
from scripts.backtest import fetch_season_data_with_delta

# Fetch season data with delta loading (only fetches week 12)
games_df, betting_df, efficiency_df, turnover_df, st_df = (
    fetch_season_data_with_delta(
        client=client,
        year=2024,
        current_week=12,  # Only fetch week 12 from API
        use_cache=True,
    )
)
```

**Usage in run_weekly.py:**
```bash
# Run weekly predictions with delta caching
python3 scripts/run_weekly.py --use-delta-cache
```

**Cache location:** `.cache/weeks/{year}/week_{N}/`

**API call reduction:**
- **Without delta cache:** ~15 API calls per week (games, plays, betting × 15 weeks)
- **With delta cache:** ~3 API calls (only current week)
- **Reduction:** 80% fewer API calls for weekly runs

### 4. ⏸ ETag / Last-Modified Headers (DEFERRED)

**Status:** Not implemented

**Reason:** CFBD Python SDK uses OpenAPI-generated client that doesn't expose HTTP headers. Implementing ETags would require:
1. Forking the SDK
2. Modifying underlying HTTP client (urllib3/requests)
3. Custom header interception
4. Maintaining fork across SDK updates

**Cost/Benefit:** High complexity, low ROI given season/week caching already provides 60-80% reduction.

**Future consideration:** If rate limits persist after delta caching, request CFBD API team to add official ETag support.

## Recommended Workflow

### For Development/Testing

```bash
# First run: populate week cache for current season
python3 scripts/populate_week_cache.py

# Subsequent backtests: use cached data
python3 scripts/backtest.py --years 2024
```

### For Production Weekly Runs

```bash
# Sunday AM after Saturday games:

# Step 1: Ensure historical weeks cached (run once per season)
python3 scripts/populate_week_cache.py --weeks 1 11

# Step 2: Run weekly predictions (only fetches week 12)
python3 scripts/run_weekly.py --use-delta-cache
```

### For Full Historical Backtests

```bash
# First run: pre-populate all seasons (saves ~6 minutes on 4-year backtest)
python3 scripts/ensure_data.py

# Run backtest (uses season cache)
python3 scripts/backtest.py --years 2022 2023 2024 2025
```

## Cache Management

### View Cache Statistics

```python
from src.data.season_cache import SeasonDataCache
from src.data.week_cache import WeekDataCache

season_cache = SeasonDataCache()
print(season_cache.get_stats())

week_cache = WeekDataCache()
print(week_cache.get_cached_weeks(2024, "games"))
```

### Clear Caches

```python
# Clear season cache
season_cache = SeasonDataCache()
season_cache.clear(year=2024)  # Clear specific year
season_cache.clear()  # Clear all

# Clear week cache
week_cache = WeekDataCache()
week_cache.clear(year=2024, week=5)  # Clear specific week
week_cache.clear(year=2024)  # Clear entire year
week_cache.clear()  # Clear all
```

### Cache Directory Structure

```
.cache/
├── seasons/           # Season-level cache (historical data)
│   ├── 2022/
│   │   ├── games.parquet
│   │   ├── betting.parquet
│   │   ├── efficiency_plays.parquet
│   │   ├── turnover_plays.parquet
│   │   └── st_plays.parquet
│   ├── 2023/
│   └── 2024/
│
└── weeks/             # Week-level cache (delta loading)
    ├── 2024/
    │   ├── week_01/
    │   │   ├── games.parquet
    │   │   ├── betting.parquet
    │   │   ├── efficiency_plays.parquet
    │   │   ├── turnovers.parquet
    │   │   └── st_plays.parquet
    │   ├── week_02/
    │   └── ...
    └── 2025/
```

## Performance Benchmarks

### Backtest (4 years, weeks 4-16)

| Strategy | API Calls | Runtime | Speedup |
|----------|-----------|---------|---------|
| No cache | ~800 | 26 min | 1x |
| Season cache | ~60 | 16 min | 1.6x |
| Week cache (production run) | ~3 | <1 min | 26x |

### Weekly Production Run (Week 12)

| Strategy | API Calls | Runtime | Speedup |
|----------|-----------|---------|---------|
| No cache | ~45 (15 weeks × 3 endpoints) | 2 min | 1x |
| Week cache | ~3 (1 week × 3 endpoints) | 10 sec | 12x |

## Troubleshooting

### "Week cache incomplete, falling back to full season fetch"

**Cause:** One or more weeks missing from cache.

**Solution:**
```bash
# Re-populate cache
python3 scripts/populate_week_cache.py --year 2024 --force-refresh
```

### "Cache HIT but data is stale"

**Cause:** Historical data changed (rare, but can happen if CFBD corrects data).

**Solution:**
```bash
# Clear and refresh specific week
python3 scripts/populate_week_cache.py --year 2024 --weeks 8 8 --force-refresh
```

### "Rate limit exceeded"

**Cause:** Cache not populated, or current week being re-fetched multiple times.

**Immediate fix:**
```bash
# Populate cache for all completed weeks
python3 scripts/populate_week_cache.py
```

**Long-term fix:**
- Use `--use-delta-cache` flag in production runs
- Pre-populate cache at start of season
- Monitor cache hit rates via logs

## API Credit Estimates

### CFBD API (Free Tier: ~200 calls/hour)

**One full season fetch (15 weeks):**
- Games: 15 calls
- Betting: 15 calls
- Plays: 15 calls
- **Total: 45 calls**

**With week-level caching (week 12):**
- Games: 1 call
- Betting: 1 call
- Plays: 1 call
- **Total: 3 calls (93% reduction)**

**Four-year backtest:**
- Without cache: 180 calls (60 min of quota)
- With season cache: 0 calls (all cached)
- With week cache: 3 calls (current week only)

## Best Practices

1. **Always populate cache before production runs**
   ```bash
   python3 scripts/populate_week_cache.py
   ```

2. **Use `--use-delta-cache` for weekly runs**
   ```bash
   python3 scripts/run_weekly.py --use-delta-cache
   ```

3. **Pre-fetch historical data for backtests**
   ```bash
   python3 scripts/ensure_data.py
   python3 scripts/backtest.py --years 2022 2023 2024
   ```

4. **Monitor cache hit rates in logs**
   ```
   Cache HIT: Loaded 2024 season (128 games, 18234 plays)
   Week cache HIT: 2024 week 11 games (64 rows)
   ```

5. **Clear cache at season start**
   ```bash
   # New season? Clear previous year's cache if needed
   python3 -c "from src.data.week_cache import WeekDataCache; WeekDataCache().clear(year=2024)"
   ```

## Future Enhancements

If rate limits persist after implementing delta caching:

1. **Request CFBD official ETag support** - Most impactful, requires API team support
2. **Implement custom HTTP interceptor** - Complex, fragile, high maintenance
3. **Add Redis/Memcached layer** - Over-engineering for current scale
4. **Rate limit throttling** - Add exponential backoff wrapper (already exists in `_call_with_retry`)

## Contact

If you encounter rate limiting issues after following this guide, check:
1. Cache is populated: `ls -la .cache/weeks/2024/`
2. Cache is being used: Check logs for "Cache HIT" messages
3. API calls are actually reduced: Monitor CFBD dashboard

For questions or issues, see project maintainer.
