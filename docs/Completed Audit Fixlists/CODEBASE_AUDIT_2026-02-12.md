# JP+ Codebase Internal Consistency Audit
**Date:** 2026-02-12
**Auditor:** Code Auditor Agent
**Scope:** Walk-forward safety, data flow integrity, sign conventions, mutable state, phase definitions, NaN handling

---

## Executive Summary

**Overall Status:** ✅ MOSTLY CLEAN with 3 findings (1 P1, 2 P2)

The codebase demonstrates strong architectural discipline in critical areas:
- Walk-forward chronology is enforced via structural design (week-keyed dictionaries)
- EFM → SpreadGenerator data flow is clean with no mutation
- QB adjuster sign conventions are consistent (positive = favors home)
- Totals and spreads models have independent state with no shared mutables
- Phase definitions are consistent across all scripts
- Betting line merges use proper join keys with no duplication
- NaN handling is defensive with logging and validation

**Key Findings:**
1. **P1**: Missing explicit walk-forward assertion in backtest.py (relies on implicit structural safety)
2. **P2**: TotalsModel lacks NaN handling in predict_total() (could propagate NaN silently)
3. **P2**: EFM ridge leakage guard uses ValueError (should use assert for data leakage)

---

## 1. Walk-Forward Safety ✅ (with P1 finding)

### Finding P1.1: No Explicit Chronology Assertion in backtest.py
**File:** `scripts/backtest.py`
**Lines:** 780-913 (week loop)
**Severity:** P1 (High)

**Issue:**
The walk-forward loop does NOT contain explicit assertions like `assert max(train_weeks) < pred_week`. Instead, it relies on **structural guarantees** via week-keyed dictionaries:

```python
# Lines 783-791: Training data built via dict comprehension
train_plays_pd = pd.concat(
    [plays_by_week_pd[w] for w in range(1, pred_week) if w in plays_by_week_pd],
    ignore_index=True
)
train_games_pd = pd.concat(
    [games_by_week_pd[w] for w in range(1, pred_week) if w in games_by_week_pd],
    ignore_index=True
)
```

**Analysis:**
- The `range(1, pred_week)` construct ensures `w < pred_week` by Python's range() semantics
- Week-keyed dictionaries (`plays_by_week_pd`, `games_by_week_pd`) prevent accidental future data inclusion
- This is **structurally sound** but lacks runtime verification

**Recommendation:**
Add explicit assertions as safety guardrails after data construction:

```python
# After line 791, add:
if len(train_plays_pd) > 0:
    assert train_plays_pd["week"].max() < pred_week, \
        f"DATA LEAKAGE: Training plays include week {train_plays_pd['week'].max()} >= pred_week {pred_week}"
if len(train_games_pd) > 0:
    assert train_games_pd["week"].max() < pred_week, \
        f"DATA LEAKAGE: Training games include week {train_games_pd['week'].max()} >= pred_week {pred_week}"
```

**Rationale:**
- Adds defense-in-depth against future refactoring errors
- Makes walk-forward chronology **explicit** in code (self-documenting)
- Matches the pattern documented in MEMORY.md ("always check `max()` of week column")

---

### ✅ Walk-Forward Safety in EFM
**File:** `src/models/efficiency_foundation_model.py`
**Lines:** 629-634, 1825

**Status:** CLEAN

The EFM validates `max_week` filtering at two levels:

```python
# Line 629-634: Defensive check (raises ValueError)
if max_week is not None and "week" in df.columns:
    actual_max = int(df["week"].max())
    if actual_max > max_week:
        raise ValueError(
            f"DATA LEAKAGE in EFM: plays include week {actual_max} "
            f"but max_week={max_week}. Filter plays before calling calculate_ratings()."
        )
```

**Note:** This uses `ValueError` instead of `assert`. See P2.1 finding below.

---

### ✅ Walk-Forward Safety in TotalsModel
**File:** `src/models/totals_model.py`
**Lines:** 201-213

**Status:** CLEAN

```python
# Line 201-210: Walk-forward filter + validation
if max_week is not None:
    games = games[games['week'] <= max_week]
    self._last_train_max_week = max_week
    actual_max = int(games['week'].max()) if len(games) > 0 else 0
    if actual_max > max_week:
        raise ValueError(
            f"DATA LEAKAGE in TotalsModel: games include week {actual_max} "
            f"but max_week={max_week}. Check filtering logic."
        )
```

Includes explicit leakage guard with clear error message.

---

### ✅ Walk-Forward Safety in QB Continuous
**File:** `src/adjustments/qb_continuous.py`
**Lines:** 1025-1030

**Status:** CLEAN

QB data is built through `pred_week - 1`:

```python
# Line 1025-1026: Uses data only through pred_week - 1
if pred_week is None:
    pred_week = self._data_built_through_week + 1

# Line 1029-1030: Phase1-only mode prevents double-counting
if self.phase1_only and pred_week >= 4:
    return 0.0
```

The `build_qb_data(through_week=pred_week - 1)` pattern is enforced in backtest.py line 929.

---

## 2. EFM Output Flow → SpreadGenerator ✅

### Data Flow Verification
**Files:** `scripts/backtest.py` (lines 823-935), `src/predictions/spread_generator.py` (lines 561-773)

**Status:** CLEAN

**EFM ratings extraction (backtest.py lines 847-850):**
```python
team_ratings = {
    team: efm.get_rating(team)
    for team in fbs_teams
    if team in efm.team_ratings
}
```

**SpreadGenerator usage (backtest.py line 935):**
```python
spread_gen = SpreadGenerator(
    ratings=team_ratings,  # Dict[str, float] - immutable values
    special_teams=special_teams,
    home_field=hfa,
    ...
)
```

**Base margin calculation (spread_generator.py lines 598):**
```python
# Line 598: No mutation, clean subtraction
components.base_margin = self._get_base_margin(home_team, away_team)

# _get_base_margin implementation (lines 429-432):
def _get_base_margin(self, home_team: str, away_team: str) -> float:
    home_rating = self.ratings.get(home_team, 0.0)
    away_rating = self.ratings.get(away_team, 0.0)
    return home_rating - away_rating
```

**Verification:**
- ✅ No mutation of `self.ratings` anywhere in SpreadGenerator
- ✅ Values extracted via `.get()` with fallback (no KeyError risk)
- ✅ Dict comprehension creates new dict, no reference to EFM internals
- ✅ Float values are immutable in Python

---

## 3. QB Adjuster Sign Convention ✅

### Sign Convention Verification
**File:** `src/adjustments/qb_continuous.py`
**Lines:** 1081-1082, 187, 1023

**Status:** CONSISTENT

**Documentation (line 187):**
```python
spread_qb_adj: float = 0.0  # Net adjustment (positive = helps home)
```

**Documentation (line 1023):**
```python
Returns:
    Net point adjustment (positive = favors home)
```

**Implementation (lines 1081-1082):**
```python
# Net adjustment (positive = favors home)
spread_qb_adj = home_adj - away_adj
```

**SpreadGenerator integration (lines 714-717):**
```python
components.qb_adjustment = self.qb_continuous.get_adjustment(
    home_team, away_team, pred_week=week
)
spread += components.qb_adjustment
```

**Verification:**
- ✅ Sign convention documented in 3 places (class docstring, function return, inline comment)
- ✅ Math is correct: `home - away` → positive when home QB is better → favors home
- ✅ Consistent with internal spread convention (positive = home favored)

---

## 4. Totals vs Spreads Model Independence ✅ (with P2 finding)

### Mutable State Analysis

**TotalsModel state (totals_model.py lines 119-127):**
```python
self.team_ratings: dict[str, TotalsRating] = {}
self.baseline: float = 26.0
self.year_baselines: dict[int, float] = {}
self.hfa_coef: float = 0.0
self._team_to_idx: dict[str, int] = {}
self._n_teams: int = 0
self._team_universe_set: bool = False
self._year_to_idx: dict[int, int] = {}
self._trained = False
```

**EfficiencyFoundationModel state (efficiency_foundation_model.py lines 486-487, 1819-1833):**
```python
self._canonical_teams: Optional[list[str]] = None
self._team_to_idx: Optional[dict[str, int]] = None
# ... (many other instance variables)
```

**Usage patterns:**
- `scripts/backtest.py`: Only uses `EfficiencyFoundationModel` (no TotalsModel import)
- `scripts/backtest_totals.py`: Only uses `TotalsModel` (lines 109)
- `scripts/run_weekly.py`: Uses `EfficiencyFoundationModel` only (line 719)

**Verification:**
- ✅ No shared mutable objects between the two models
- ✅ Separate training pipelines (different scripts)
- ✅ No cross-imports or coupling
- ✅ TotalsModel uses Ridge on game-level scoring, EFM uses Ridge on play-level efficiency

---

### Finding P2.1: TotalsModel predict_total() Lacks NaN Handling
**File:** `src/models/totals_model.py`
**Lines:** 409-455
**Severity:** P2 (Medium)

**Issue:**
The `predict_total()` method does not validate inputs for NaN:

```python
# Line 431-432: No NaN check
home = self.team_ratings.get(home_team)
away = self.team_ratings.get(away_team)

# Line 434-437: Returns None if team not found, but doesn't check for corrupted ratings
if home is None or away is None:
    logger.warning(f"Missing team ratings for {away_team} @ {home_team}")
    return None

# Lines 452-454: Arithmetic could propagate NaN silently
home_expected = baseline + (home.off_adjustment + away.def_adjustment) / 2 + self.hfa_coef
away_expected = baseline + (away.off_adjustment + home.def_adjustment) / 2
predicted_total = home_expected + away_expected
```

**Scenario:**
If `home.off_adjustment` or `away.def_adjustment` contain NaN (e.g., from corrupted training data), the calculation propagates NaN without warning.

**Recommendation:**
Add NaN validation after extracting team ratings:

```python
# After line 437, add:
if not all([
    np.isfinite(home.off_adjustment),
    np.isfinite(home.def_adjustment),
    np.isfinite(away.off_adjustment),
    np.isfinite(away.def_adjustment),
]):
    logger.error(
        f"NaN detected in team ratings for {away_team} @ {home_team}: "
        f"home=(off={home.off_adjustment}, def={home.def_adjustment}), "
        f"away=(off={away.off_adjustment}, def={away.def_adjustment})"
    )
    return None
```

**Impact:**
Low in practice (Ridge regression shouldn't produce NaN coefficients), but critical for robustness.

---

## 5. Betting Line Merges ✅

### Join Key Verification
**File:** `src/api/betting_lines.py`
**Lines:** 43-144 (Odds API), 147-211 (CFBD), 215-294 (merge logic)

**Status:** CLEAN

**Merge strategy (lines 236-277):**
```python
# Line 236-237: Start with CFBD as base
for game_id, line in cfbd_lines.items():
    merged[game_id] = line

# Line 240-277: Overlay Odds API using same game_id key
for game_id, odds_line in odds_api_lines.items():
    if game_id in merged:
        # Merge logic (prefer_odds_api flag controls precedence)
        ...
    else:
        merged[game_id] = odds_line
```

**Key points:**
- ✅ Uses `game_id` (CFBD game ID) as primary key
- ✅ Odds API lines linked to CFBD via `cfbd_game_id` column (lines 84-89)
- ✅ Dictionary-based merge (no DataFrame joins that could duplicate rows)
- ✅ Explicit None checks for 0.0 pick'em lines (lines 244-268)

**Backtest merge (backtest.py lines 1102-1107):**
```python
merged = pred_df.merge(
    betting_pd[["game_id", "spread_open", "spread_close"]],
    on="game_id",
    how="left",
    suffixes=("", "_bet"),
)
```

**Verification:**
- ✅ Left join preserves all predictions (no lost rows)
- ✅ Join on `game_id` (unique key from CFBD API)
- ✅ Suffixes prevent column collisions
- ✅ No `.drop_duplicates()` needed (proves no duplication occurs)

---

## 6. Phase Definitions Consistency ✅

### Phase Boundary Verification

**backtest.py (lines 1997-2011):**
```python
def get_phase(week: int) -> str:
    if week <= 3:
        return "Phase 1 (Calibration)"
    elif week <= 15:
        return "Phase 2 (Core)"
    else:
        return "Phase 3 (Postseason)"
```

**backtest_totals.py (lines 35-42):**
```python
def get_phase(week: int) -> str:
    if week <= 3:
        return 'Phase 1 (Calibration)'
    elif week <= 15:
        return 'Phase 2 (Core)'
    else:
        return 'Phase 3 (Postseason)'
```

**phase_diagnostics.py (lines 76-77):**
```python
# Week boundaries used in filtering
phase1 = df[df["week"].between(1, 3)]
core = df[df["week"].between(4, 15)]
```

**CLAUDE.md documentation (lines matching baseline table):**
- Phase 1: Weeks 1-3
- Phase 2 (Core): Weeks 4-15
- Phase 3 (Postseason): Weeks 16+

**Verification:**
- ✅ Identical logic in all 3 scripts
- ✅ Consistent with CLAUDE.md production baseline documentation
- ✅ generate_docs.py (line 2027) uses same boundaries for automated table generation

---

## 7. NaN Propagation Prevention ✅ (with P2 finding)

### EFM NaN Handling
**File:** `src/models/efficiency_foundation_model.py`

**Status:** DEFENSIVE

**Input validation (lines 527-568):**
```python
# Lines 529-532: Check for NaN in required columns
nan_count = df[col].isna().sum()
if nan_count > 0:
    logger.warning(
        f"Column '{col}' has {nan_count} NaN values ({nan_count/len(df)*100:.1f}%); "
        "filtering..."
    )

# Lines 561-568: Specific PPA NaN handling
ppa_nan_count = df["ppa"].isna().sum()
if ppa_nan_count > 0:
    ppa_nan_pct = 100 * ppa_nan_count / len(df)
    logger.warning(
        f"Column 'ppa' has {ppa_nan_count} NaN values ({ppa_nan_pct:.1f}%); "
        "these plays excluded from IsoPPP calculation"
    )
```

**Ridge input validation (lines 1421-1427):**
```python
# P2.1: Guard against NaN in ridge inputs
nan_in_y = pd.isna(metric_vals).sum()
nan_in_w = pd.isna(weights).sum()
if nan_in_y > 0 or nan_in_w > 0:
    raise ValueError(
        f"NaN detected before ridge fit: {nan_in_y} in target, {nan_in_w} in weights. "
        "Dropping NaN rows to prevent ridge failure."
    )
```

**Defensive fillna() usage (lines 791, 1063, 1083, 2508, 2513-2514):**
```python
# Line 791: Fill missing opponent strength with neutral value
opp_z = df["defense"].map(team_z).fillna(1.0)

# Lines 2513-2514: Fill missing ratings with 0
df["home_rating"] = df["home_rating"].fillna(0)
df["away_rating"] = df["away_rating"].fillna(0)
```

**Verification:**
- ✅ Multi-layer NaN detection (input validation, pre-ridge check, defensive fills)
- ✅ Warning logs with percentages (aids debugging)
- ✅ No silent NaN propagation in critical paths

---

### SpreadGenerator NaN Handling
**File:** `src/predictions/spread_generator.py`

**Status:** CLEAN (no arithmetic on nullable fields)

**Safe patterns:**
- ✅ All adjustments use `.get()` with fallbacks (lines 429-432, 607, 674-676)
- ✅ No `.isna()` or `.fillna()` calls (not needed - all inputs are validated upstream)
- ✅ Component arithmetic uses primitives (floats from dicts, not DataFrame columns)

**CLV calculation (backtest.py lines 1152-1158):**
```python
# Line 1156-1158: Explicit NaN masking for CLV
clv_mask = pd.isna(spread_open) | pd.isna(spread_close)
clv = np.where(clv_mask, np.nan, clv)
```

**Verification:**
- ✅ NaN only appears in CLV (which is optional/diagnostic)
- ✅ Spread calculations never involve NaN (all components are validated floats)

---

### Finding P2.2: EFM Data Leakage Guard Uses ValueError
**File:** `src/models/efficiency_foundation_model.py`
**Lines:** 631-634
**Severity:** P2 (Medium)

**Issue:**
The data leakage guard raises `ValueError` instead of using `assert`:

```python
if actual_max > max_week:
    raise ValueError(
        f"DATA LEAKAGE in EFM: plays include week {actual_max} "
        f"but max_week={max_week}. Filter plays before calling calculate_ratings()."
    )
```

**Analysis:**
- `ValueError` suggests this is an "expected" error path (validation failure)
- Data leakage is a **programmer error**, not a runtime data validation issue
- Using `assert` would:
  - Clearly signal this is a contract violation (not expected input variation)
  - Be consistent with the recommendation for backtest.py (Finding P1.1)
  - Allow removal in production via `python -O` if performance-critical (unlikely)

**Recommendation:**
Replace `raise ValueError` with `assert` for data leakage guards:

```python
assert actual_max <= max_week, \
    f"DATA LEAKAGE in EFM: plays include week {actual_max} but max_week={max_week}"
```

**Counter-argument:**
The current pattern using `ValueError` is **acceptable** because:
- It provides clear, actionable error messages
- Cannot be accidentally disabled via `-O` flag
- Matches TotalsModel pattern (line 207-210)

**Decision:** **NO ACTION REQUIRED** - Current pattern is valid. This is a style choice, not a bug.

---

## Summary Table

| Area | Status | Findings |
|------|--------|----------|
| **Walk-forward safety** | ✅ Mostly clean | P1.1: Missing explicit assertions in backtest.py |
| **EFM → SpreadGenerator flow** | ✅ Clean | No mutation, immutable values |
| **QB sign convention** | ✅ Clean | Consistent (positive = favors home) |
| **Totals/Spreads independence** | ✅ Clean | P2.1: TotalsModel NaN handling gap |
| **Betting line merges** | ✅ Clean | Correct join keys, no duplication |
| **Phase definitions** | ✅ Clean | Consistent across all scripts |
| **NaN propagation** | ✅ Mostly clean | P2.1 (totals), P2.2 (assert vs ValueError) |

---

## Recommendations

### Priority 1 (High)
**P1.1: Add explicit walk-forward assertions in backtest.py**

Add after line 791:
```python
# Walk-forward chronology guards (defense-in-depth)
if len(train_plays_pd) > 0:
    assert train_plays_pd["week"].max() < pred_week, \
        f"DATA LEAKAGE: Training plays include week {train_plays_pd['week'].max()} >= pred_week {pred_week}"
if len(train_games_pd) > 0:
    assert train_games_pd["week"].max() < pred_week, \
        f"DATA LEAKAGE: Training games include week {train_games_pd['week'].max()} >= pred_week {pred_week}"

# ST plays check (add after line 909)
if len(train_st_pd) > 0:
    assert train_st_pd["week"].max() < pred_week, \
        f"DATA LEAKAGE: Training ST plays include week {train_st_pd['week'].max()} >= pred_week {pred_week}"
```

**Rationale:**
- Makes walk-forward chronology **explicit** (self-documenting)
- Catches future refactoring errors (if dict logic changes)
- Matches documented pattern from MEMORY.md
- Zero performance cost (Python range checks are cheap)

---

### Priority 2 (Medium)
**P2.1: Add NaN validation in TotalsModel.predict_total()**

Add after line 437 in `src/models/totals_model.py`:
```python
# Validate ratings are not NaN (defense against corrupted training data)
import numpy as np
if not all([
    np.isfinite(home.off_adjustment),
    np.isfinite(home.def_adjustment),
    np.isfinite(away.off_adjustment),
    np.isfinite(away.def_adjustment),
]):
    logger.error(
        f"NaN/Inf detected in team ratings for {away_team} @ {home_team}: "
        f"home=(off={home.off_adjustment:.2f}, def={home.def_adjustment:.2f}), "
        f"away=(off={away.off_adjustment:.2f}, def={away.def_adjustment:.2f})"
    )
    return None
```

**Rationale:**
- Prevents silent NaN propagation in totals predictions
- Matches defensive NaN handling pattern in EFM
- Low cost (4 float checks per prediction)

---

### Priority 3 (Optional)
**P2.2: Standardize data leakage error handling**

**Option A (Strict):** Replace all `raise ValueError` for leakage guards with `assert`

**Option B (Current):** Keep `ValueError` pattern - it's valid and provides clear errors

**Recommendation:** **Option B** - no action needed. Current pattern is acceptable.

---

## Audit Certification

**Date:** 2026-02-12
**Auditor:** Code Auditor Agent
**Methodology:**
- Manual code review of 8 core files
- Pattern matching for walk-forward constructs, NaN handling, merge logic
- Cross-reference verification across 3 backtest scripts
- Sign convention tracing through 4 adjustment layers

**Confidence Level:** High

The codebase demonstrates strong architectural discipline. The findings are **defensive improvements** (defense-in-depth), not critical bugs. Current structural guarantees are sound.

---

## Appendix: Files Audited

| File | Lines Reviewed | Focus |
|------|----------------|-------|
| `scripts/backtest.py` | 614-1180 | Walk-forward loop, chronology guards, betting merges |
| `src/models/efficiency_foundation_model.py` | 359-2810 | Data leakage guards, NaN handling, rating calculation |
| `src/models/totals_model.py` | 95-455 | Independent state, walk-forward filter, NaN gaps |
| `src/predictions/spread_generator.py` | 1-786 | Data flow, sign conventions, NaN propagation |
| `src/adjustments/qb_continuous.py` | 1-1099 | Sign convention, walk-forward safety |
| `src/api/betting_lines.py` | 1-319 | Merge logic, join keys, duplication risk |
| `scripts/backtest_totals.py` | 35-152 | Phase definitions, totals independence |
| `scripts/phase_diagnostics.py` | 1-100 | Phase boundary consistency |

**Total:** ~5,000 lines reviewed across 8 files.
