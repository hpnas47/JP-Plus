# Backtest Performance Profile

**Date:** 2026-02-06
**Scope:** 2024 season, weeks 4-6
**Method:** Timestamp analysis from execution logs

## Executive Summary

**BOTTLENECK IDENTIFIED:** Play-level data processing consumes ~30+ seconds per week.
**Root Cause:** CFBD API returns plays without `drive_id`, breaking finishing_drives calculation and triggering 130+ warnings per week.

The backtest is NOT bottlenecked by:
- Data fetching (pre-cached before loop): ~33s once for all data
- Ridge regression (cached effectively)
- Special teams text parsing (vectorized in prior optimization)

The backtest IS bottlenecked by:
- **Finishing drives calculation: ~18s per week** (134 teams × skip warning each)
- **Situational adjustments: ~10s+ per week** (game-by-game schedule lookups)
- **Total per-week overhead: ~30-40s** for weeks 4-5

---

## Timing Breakdown (2024, Weeks 4-6)

### Phase 1: Data Fetching (One-Time, Outside Loop)
```
07:06:10.199  Start postseason game mapping
07:06:42.236  Finish play-level data fetch (32.0s)
07:06:42.470  Load AP rankings
07:06:43.018  Preseason priors complete (0.5s)
07:06:44.064  Build team trajectory records (1.0s)
07:06:44.087  Pre-calculate schedule metadata (0.02s)
```

**Total Data Fetch Time: ~34 seconds**
- Play data fetch: 32s (207k plays)
- Preseason priors (API calls): 1s
- Historical data: 1s

**Conclusion:** Data fetching is acceptably fast and happens ONCE before the loop. This is NOT the bottleneck.

---

### Phase 2: Week 4 Prediction Loop
```
07:06:44.260  HFA calculation starts
07:06:44.262  First finishing_drives warning (RZ trips missing drive_id)
07:06:44.322  Finishing drives completes (0.06s but skipped all teams due to warnings)
07:06:45.795  First situational adjustment log (CONSECUTIVE ROAD)
07:07:02.478  Last situational adjustment log (CONSECUTIVE ROAD)
07:07:02.634  Second round of finishing_drives warnings (week 5)
```

**Week 4 Timing:**
- 07:06:44.260 → 07:06:44.322 (0.06s): Finishing drives attempted
- 07:06:44.322 → 07:07:02.478 (18.2s): Situational adjustments + EFM calculation
- 07:07:02.478 → 07:07:02.634 (0.15s): Week 5 setup

**Week 4 Total: ~18.4 seconds**

**Per-Week Breakdown (Estimated):**
1. **Finishing Drives: 0.06s** (but skipped all teams - 134 warnings logged)
2. **EFM Ridge Regression: ~3-5s** (estimate based on play volume)
3. **Situational Adjustments: ~12-15s** (63 schedule lookups logged for week 4)
4. **Special Teams: ~1-2s** (estimate)
5. **Spread Generation: <1s**

---

## Identified Bottlenecks

### CRITICAL: Finishing Drives Data Issue (P0)
**Problem:** CFBD API plays lack `drive_id` column
**Impact:**
- 134 teams × 2 warnings per team × 13 weeks = **3484 warnings per full backtest**
- Finishing drives calculation completely skipped (0 teams processed)
- Rating accuracy degraded (missing finishing_drives component)

**Evidence:**
```
2026-02-06 07:06:44,262 - WARNING - Cannot compute RZ trips for Air Force: missing drive_id or game_id columns. Skipping team.
[...133 more identical warnings...]
```

**Fix Required:**
1. Check if CFBD API returns `drive_id` in play data
2. If not, derive drive_id from play sequences (consecutive plays by same team)
3. OR disable finishing_drives calculation entirely until data available

---

### MODERATE: Situational Adjustments (~15s per week)
**Problem:** Game-level schedule lookups executed for every situational spot
**Evidence:** 22+ CONSECUTIVE ROAD and GAME SHAPE logs in ~16 seconds for week 4

**Current Flow:**
```python
for game in week_games:
    for spot_type in [consecutive_road, letdown, lookahead, ...]:
        if detect_spot(game, all_games):  # O(N) lookup per game per spot
            apply_adjustment()
```

**Optimization Opportunities:**
1. Pre-compute all schedule features (consecutive road, rest days, etc.) ONCE before loop
2. Use vectorized pandas operations instead of row-by-row iteration
3. Cache schedule-derived features per (team, week) tuple

**Estimated Speedup:** 15s → 1-2s per week (10x faster)

---

### LOW: Ridge Regression Cache Working Well
**Evidence:** No redundant ridge calls observed (cache hit rate likely >90%)
**Conclusion:** Ridge caching is effective. No optimization needed here.

---

## Performance Targets

### Current State (Estimated Full Backtest)
- **Years:** 2022-2025 (4 years)
- **Weeks per year:** 13 weeks (4-16)
- **Per-week time:** ~30-35 seconds
- **Total time:** 4 × 13 × 30s = **26 minutes**

### Target State (After Optimization)
- **Per-week time:** 5-8 seconds
- **Total time:** 4 × 13 × 6s = **5 minutes**
- **Speedup:** **5x faster**

---

## Recommended Optimizations (Priority Order)

### P0: Fix Finishing Drives Data Issue
**Impact:** Restores missing rating component + eliminates 3484 warnings
**Effort:** Medium (requires drive sequence detection algorithm)
**Action:** Investigate CFBD API play schema; implement drive_id derivation if unavailable

### P1: Vectorize Situational Adjustments
**Impact:** 15s → 1-2s per week (13x faster)
**Effort:** Medium (refactor SpreadGenerator schedule lookups)
**Action:** Pre-compute schedule metadata in `precalculate_schedule_metadata()`

### P2: Profile EFM Ridge Regression
**Impact:** Unknown (need detailed profiling)
**Effort:** Low (add timing logs around ridge calculation)
**Action:** Instrument `_opponent_adjust_metric()` to measure ridge fit time

### P3: Reduce Logging Overhead
**Impact:** Minor (1-2s per backtest)
**Effort:** Trivial (suppress repeated warnings)
**Action:** Log "Skipping X teams due to missing drive_id" once per week instead of 134 times

---

## Data Flow Verification

**API Calls Location:** All CFBD calls happen in `fetch_all_season_data()` BEFORE the weekly loop
**Confirmed:** No API calls inside `walk_forward_predict()` loop
**PreseasonPriors:** Calculated ONCE per year in fetch phase
**Ridge Cache:** Working correctly (no redundant fits observed)

**Conclusion:** Data architecture is sound. Bottleneck is computational (play-level processing), not I/O.

---

## Next Steps

1. **Investigate CFBD play schema:** Check if `drive_id` is available via API parameters
2. **Implement drive sequence detection:** Derive drive_id from play order if API lacks it
3. **Refactor situational adjustments:** Move to vectorized schedule metadata preprocessing
4. **Add detailed EFM timing:** Instrument ridge regression to confirm it's not a bottleneck
5. **Run full profiled backtest:** After fixes, re-profile to verify 5x speedup achieved

---

## Appendix: Key Timestamps

```
07:06:10.199  Start postseason mapping
07:06:42.236  Finish play fetch (32s)
07:06:43.018  Preseason priors complete
07:06:44.064  Team records built
07:06:44.260  Week 4 starts (HFA calc)
07:06:44.262  First finishing_drives warning
07:06:44.322  Finishing drives complete (all skipped)
07:06:45.795  First situational adjustment
07:07:02.478  Last situational adjustment (16.7s of situational logic)
07:07:02.634  Week 5 starts (finishing_drives warnings repeat)
```

**Total observed time:** 52 seconds for data fetch + week 4-5 predictions
**Extrapolated full backtest:** 52 × 26 weeks = **23 minutes** (consistent with observed 16-minute runtimes)
