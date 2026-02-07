# Non-Conference Game Weighting Implementation

## Summary

Implemented 1.5x weight multiplier for non-conference FBS games in the Efficiency Foundation Model (EFM) to address conference rating circularity in Ridge regression.

## Problem Statement

**Issue:** Big 12 teams were massively over-rated compared to SP+:
- UCF: JP+ #13 vs SP+ #62 (49-spot difference, 4-8 record)
- Baylor: JP+ #16 vs SP+ #38 (22-spot difference)
- Colorado: JP+ #12 vs SP+ #39 (27-spot difference)

**Root Cause:** Ridge regression operates in a nearly closed system for conference games. Big 12 has the lowest rating standard deviation (6.77) of any major conference. Without strong cross-conference anchors, teams inflate each other's opponent-adjusted metrics through circular reasoning.

**Failed Approach:** Tested garbage time weighting variants (leading team at 0.5x, 0.7x, and symmetric) - all degraded 5+ edge ATS by 1.0-1.3%.

## Solution

Apply 1.5x weight multiplier to plays from non-conference FBS-vs-FBS games. This increases the influence of out-of-conference results in the Ridge regression, anchoring conference ratings to external benchmarks.

## Implementation

### Files Modified

1. **`src/models/efficiency_foundation_model.py`**
   - Line 1260: Added `team_conferences` parameter to `calculate_ratings()`
   - Line 577: Added `team_conferences` parameter to `_prepare_plays()`
   - Lines 711-724: Non-conference weighting logic

2. **`scripts/backtest.py`**
   - Lines 50-80: Added `_fetch_team_conferences()` helper function
   - Line 713: Fetch conference data and pass to EFM

### Code Details

**EFM Non-Conference Weighting Logic** (`_prepare_plays()` lines 711-724):

```python
# Apply non-conference game weighting (1.5x boost for OOC matchups)
# This reduces conference circularity in ridge regression by anchoring
# conference ratings to external benchmarks. Only boosts FBS-vs-FBS OOC games.
if team_conferences is not None and "offense" in df.columns and "defense" in df.columns:
    # Map teams to conferences (vectorized lookup)
    off_conf = df["offense"].map(team_conferences)
    def_conf = df["defense"].map(team_conferences)

    # Non-conference game: both teams are FBS (in conference dict) and different conferences
    # .notna() ensures both teams are FBS (missing conf = FCS team, don't boost)
    is_ooc_fbs = off_conf.notna() & def_conf.notna() & (off_conf != def_conf)
    ooc_count = is_ooc_fbs.sum()

    if ooc_count > 0:
        df["weight"] = np.where(is_ooc_fbs, df["weight"] * 1.5, df["weight"])
        logger.debug(f"  Applied 1.5x weight to {ooc_count:,} non-conference FBS plays")
```

**Conference Data Fetching** (`backtest.py` lines 50-80):

```python
def _fetch_team_conferences(year: int) -> dict[str, str]:
    """Fetch conference affiliation for all FBS teams for a specific year.

    Uses get_fbs_teams(year=year) to get year-appropriate conference data,
    which correctly handles realignment (e.g., USC/UCLA to Big Ten in 2024,
    Texas/Oklahoma to SEC in 2024).

    Uses CFBDClient session-level cache.

    Args:
        year: Season year (conference affiliations as of this year)

    Returns:
        Dict mapping team name to conference name
    """
    try:
        client = CFBDClient()
        teams = client.get_fbs_teams(year=year)
        conf_map = {}
        for t in teams:
            if t.school and t.conference:
                conf_map[t.school] = t.conference
        logging.getLogger(__name__).debug(
            f"Fetched {year} conference data for {len(conf_map)} FBS teams"
        )
        return conf_map
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Could not fetch team conferences for {year}: {e}"
        )
        return {}
```

## Impact

**Test Case: 2024 Weeks 1-3**
- Total plays: 47,759
- Non-conference FBS plays: 15,804 (33.1%)
- All 15,804 plays received 1.5x weight multiplier

**Example OOC Games Boosted:**
- Nevada (Mountain West) vs SMU (ACC)
- LSU (SEC) vs USC (Big Ten)
- Wyoming (Mountain West) vs Arizona State (Big 12)
- Coastal Carolina (Sun Belt) vs Jacksonville State (C-USA)

## Key Design Decisions

1. **Only boost FBS-vs-FBS games**: FCS games are excluded using `.notna()` check on both offense and defense conferences.

2. **Conference data via `get_fbs_teams(year=year)`**: Uses year-specific conference affiliations to correctly handle realignment (USC/UCLA to Big Ten in 2024, Texas/Oklahoma to SEC in 2024).

3. **Client-side caching**: `CFBDClient` caches `get_fbs_teams()` results at session level, so conference data is fetched only once per year per backtest run.

4. **Vectorized implementation**: Uses pandas `.map()` and `np.where()` for efficient processing.

5. **Applied after garbage time weighting**: OOC multiplier is applied to the `weight` column after garbage time and time decay adjustments, so it compounds with those weights.

6. **Debug logging only**: The "Applied 1.5x weight" message logs at DEBUG level to avoid cluttering INFO-level backtest output.

## Rationale

### Why 1.5x?

- **Conservative anchor**: Strong enough to reduce circularity without overwhelming conference-play data
- **Balance**: Early season has ~33% OOC games, so 1.5x boost makes OOC games roughly equal weight to conference games in aggregate
- **Tunable**: Parameter can be adjusted if testing shows 1.5x is too strong/weak

### Why weight-based instead of post-regression adjustment?

- **Integrated approach**: Weighting affects the Ridge regression directly, so OOC results influence the learned team coefficients
- **Preserves HFA estimation**: Ridge regression still learns home field advantage from all games
- **Simpler**: No need for separate conference strength calculation or post-hoc adjustments

## Testing

**Syntax Validation:**
```bash
python3 -m py_compile src/models/efficiency_foundation_model.py
python3 -m py_compile scripts/backtest.py
```

**Functional Test:**
```bash
python3 scripts/backtest.py --years 2024 --start-week 4 --end-week 6
```

Result: Backtest completed successfully with non-conference weighting applied.

## Full Backtest Results (2022-2025): APPROVED

The OOC weighting was committed as part of the Conference Strength Anchor feature (`f7bf815`), which combined two mechanisms:
1. **OOC play weighting (1.5x)** — this document's implementation
2. **Post-Ridge Bayesian conference anchors** — separate O/D anchors from OOC scoring margin (scale=0.08, prior_games=30, max_adjustment=±2.0)

### Backtest Impact (Combined Feature, vs Pre-Anchor Baseline)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Core MAE (4-15) | 12.52 | 12.49 | -0.03 |
| Core ATS (Close) | 52.0% | 52.4% | +0.4% |
| Core 3+ Edge (Close) | 52.3% | 52.9% | +0.6% |
| **Core 5+ Edge (Close)** | **53.5%** | **54.8%** | **+1.3%** |

### Anchor Scale Sweep (Post-Approval)

Tested 4 anchor_scale variants to optimize the Bayesian anchor strength:

| Scale | 3+ Edge | 5+ Edge | MAE | Verdict |
|-------|---------|---------|-----|---------|
| **0.08** | 52.9% | **54.8%** | 12.49 | **KEPT (production)** |
| 0.12 | 53.6% (+0.7%) | 54.5% (-0.3%) | 12.50 | Reverted — 5+ Edge degraded |
| 0.15 | 53.4% | 54.2% | 12.52 | Rejected |
| 0.20 | 53.1% | 53.8% | 12.55 | Rejected |

**Decision:** 5+ Edge (~2% over vig) is the binding constraint. Scale 0.08 preserved best 5+ Edge. Scale 0.12 improved 3+ Edge but degraded the higher-conviction bets.

### Big 12 Impact

- UCF dropped from #13 → #19 (still inflated vs SP+ #62, but meaningful improvement)
- Composite anchor CANNOT fully fix Colorado/Baylor because Big 12 OOC margin is positive overall — the conference performs reasonably well OOC, the problem is intra-conference circularity
- Conference rating std deviation increased (reduced artificial compression)

### Garbage Time Variants (Tested Alongside, All Rejected)

| GT Variant | 5+ Edge | Verdict |
|------------|---------|---------|
| Baseline (asymmetric, leading=1.0) | 53.5% | — |
| Leading=0.5 | 52.5% | REJECTED |
| Leading=0.7 | 52.2% | REJECTED |
| Symmetric | 52.3% | REJECTED |

Asymmetric GT (leading=1.0) confirmed as optimal. Leading team plays ARE informative; down-weighting them removes real signal.

## Implementation Date

2026-02-06

## Current Status

**APPROVED and in production.** OOC 1.5x weighting is part of the Conference Strength Anchor feature. Parameters reverted to conservative settings (0.08/30/2.0) on 2026-02-07 after aggressive params (0.12/20/3.0) were found to degrade 5+ Edge.

## Related Issues

- Big 12 rating inflation (UCF #13, Baylor #16, Colorado #12 vs SP+ reality)
- Ridge regression conference circularity
- Garbage time weighting experiments (rejected due to ATS degradation)
- Conference Anchor param revert (57f228a) — preserved O/D split architecture
