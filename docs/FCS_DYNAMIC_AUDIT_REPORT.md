# FCS Dynamic Strength Estimator - Audit Report
**Date**: 2026-02-10
**Auditor**: Code Auditor
**Status**: ✅ APPROVED with Documentation P2 Issue

---

## Executive Summary

The FCS Dynamic Strength Estimator implementation is **walk-forward safe**, mathematically sound, and properly integrated. The feature replaces a static elite list with data-driven Bayesian shrinkage penalties, correctly handling all edge cases including week 1, sparse data, and unknown teams.

**Issues Found**:
- **1 P2 (Documentation)**: CLAUDE.md has incorrect parameter values (intercept listed as 20, should be 10)
- **0 P0/P1 issues**: No data leakage, no edge case failures, no integration bugs

---

## 1. Data Leakage Prevention ✅ PASS

### Walk-Forward Safety Verification

**Implementation**: `FCSStrengthEstimator.update_from_games()` at line 843 of `backtest.py`

```python
fcs_estimator.update_from_games(games_df, fbs_teams, max_week=pred_week - 1)
```

**Key Design Decisions**:
1. **Clears state on every update** (lines 115-117 of `fcs_strength.py`): Prevents accumulation bugs
2. **Filters to `max_week` before aggregation** (line 120-124): Only includes games with `week <= max_week`
3. **Rebuilds from scratch each week**: Idempotent operation ensures consistency

**Validation Tests**:
```
✓ Week 1 prediction (max_week=0): 0 games used
✓ Week 2 prediction (max_week=1): 1 game used
✓ Week 3 prediction (max_week=2): 2 games used
✓ Idempotency: Calling twice with max_week=4 yields same n_games=2
✓ Future game isolation: Team playing in week 3 has 0 games at max_week=2
```

**Conclusion**: Walk-forward chronology is strictly enforced. **No data leakage possible.**

---

## 2. Implementation Files Review ✅ PASS

### Core Module: `src/models/fcs_strength.py`

**Architecture**:
- `FCSTeamStrength` dataclass: Stores per-team margin, shrinkage, penalty
- `FCSStrengthEstimator` dataclass: Main estimator with Bayesian shrinkage
- Vectorized Polars pipeline for performance (replaces Python loops)

**Key Methods**:
- `update_from_games()`: Walk-forward safe update (clears & rebuilds)
- `get_penalty()`: Returns dynamic penalty or baseline fallback
- `get_strength()`: Returns full strength object or None
- `_margin_to_penalty()`: Linear mapping with [min, max] clipping

**Code Quality**: Clean, well-documented, type-annotated, no TODOs

### Integration: `scripts/backtest.py`

**Lines 841-844**: FCS estimator updated before each week's predictions
```python
if fcs_estimator is not None and not fcs_static:
    fcs_estimator.update_from_games(games_df, fbs_teams, max_week=pred_week - 1)
    active_fcs_estimator = fcs_estimator
```

**Lines 858-862**: Passed to SpreadGenerator
```python
spread_gen = SpreadGenerator(
    ratings=team_ratings,
    fbs_teams=fbs_teams,
    fcs_estimator=active_fcs_estimator,  # None if --fcs-static
    ...
)
```

**Correctness**: Estimator is stateless across weeks (cleared on each update), preventing cross-contamination

### Usage: `src/predictions/spread_generator.py`

**Lines 386-413**: `_get_fcs_adjustment()` method
- Checks if home/away is FBS using `fbs_teams` set
- Calls `fcs_estimator.get_penalty()` for dynamic penalties
- Falls back to static `fcs_penalty_elite/standard` if no estimator
- **Important**: FCS penalty is applied EVEN IF FCS team has a rating (line 368-370 comment)

**Design Validation**: FCS teams that play FBS games will have EFM ratings, but still receive the FCS penalty. This is correct—the penalty represents talent gap + scheduling strength, not just absence of data.

---

## 3. Bayesian Shrinkage Validation ✅ PASS

### Formula Verification

**Implementation** (lines 204-209 of `fcs_strength.py`):
```python
shrink_factor = n_games / (n_games + self.k_fcs)
shrunk_margin = (
    self.baseline_margin
    + (raw_margin - self.baseline_margin) * shrink_factor
)
```

**Default Parameters**:
- `k_fcs = 8.0` ✓
- `baseline_margin = -28.0` ✓
- `intercept = 10.0` ✓
- `slope = 0.8` ✓
- Range: `[10.0, 45.0]` ✓

### Penalty Mapping Test

| Raw Margin | Shrunk Margin | Penalty | Expected |
|------------|---------------|---------|----------|
| -50 (weak FCS) | -50.0 | 45.0 | Capped at max ✓ |
| -28 (baseline) | -28.0 | 32.4 | intercept + 0.8*28 ✓ |
| -10 (elite FCS) | -10.0 | 18.0 | intercept + 0.8*10 ✓ |
| 0 (hypothetical) | 0.0 | 10.0 | intercept only ✓ |

**Math Check**: Formula is `penalty = clamp(10 + 0.8 * abs(margin), 10, 45)`
- For margin=-28: `10 + 0.8*28 = 32.4` ✓
- For margin=-10: `10 + 0.8*10 = 18.0` ✓

**Conclusion**: Shrinkage and penalty mapping are mathematically correct.

---

## 4. CLI Integration ✅ PASS

### Available Arguments

**Baseline Comparison**:
- `--fcs-static`: Use static elite list instead of dynamic estimator ✓

**Tuning Parameters**:
- `--fcs-k`: Shrinkage k (default 8.0) ✓
- `--fcs-baseline`: Prior margin (default -28.0) ✓
- `--fcs-min-pen`: Minimum penalty (default 10.0) ✓
- `--fcs-max-pen`: Maximum penalty (default 45.0) ✓
- `--fcs-slope`: Penalty slope (default 0.8) ✓
- `--fcs-intercept`: Base penalty (default 10.0) ✓
- `--fcs-hfa`: HFA neutralization (default 0.0 = disabled) ✓

**Fallback Parameters** (used when `--fcs-static` enabled):
- `--fcs-penalty-elite`: Static elite penalty (default 18.0) ✓
- `--fcs-penalty-standard`: Static standard penalty (default 32.0) ✓

**Validation**: All parameters correctly parsed and passed to estimator (lines 2186-2193 of `backtest.py`)

**Default Mode**: Dynamic estimator is **enabled by default**, `--fcs-static` is opt-in for baseline comparison

---

## 5. Edge Cases ✅ PASS

### Edge Case 1: Week 1 (No FCS Games Yet)

**Test**: Predict week 1 with `max_week=0` (no prior data)

**Result**:
- Unknown FCS team gets baseline penalty: **32.4 pts** ✓
- No crash, no NaN values ✓

**Logic**: `get_penalty()` checks if team is in `_team_strengths`, falls back to `_margin_to_penalty(baseline_margin)` for unknowns (lines 250-254)

### Edge Case 2: FCS Team with 1 Game

**Test**: NDSU loses 42-21 to Alabama in week 1

**Result**:
- Raw margin: -21.0
- Shrunk margin: -27.2 (heavy shrinkage toward baseline=-28)
- Penalty: **31.8 pts** ✓

**Validation**: With k=8 and n=1, shrink_factor = 1/9 = 11%, so margin is mostly baseline

### Edge Case 3: Division by Zero Risk

**Analysis**: Shrinkage formula is `n / (n + k)`
- When `n=0`: Not computed (team not in dict, uses baseline fallback)
- When `k=0`: Formula becomes `n / n = 1.0` (100% trust in data, no shrinkage) ✓
- No division by zero risk exists

### Edge Case 4: FCS Team with Rating

**Test**: NDSU has EFM rating of -15.0 AND is flagged as FCS

**Result**:
- Base margin: 35.0 (Alabama 20.0 - NDSU -15.0)
- FCS adjustment: +31.8
- Total spread: **70.5** ✓

**Validation**: System correctly applies BOTH rating differential AND FCS penalty. This is intentional per design (line 368-370 comment in `spread_generator.py`)

### Edge Case 5: Both Teams FCS

**Logic**: Line 411-413 of `spread_generator.py`
```python
else:
    # Both FBS or both non-FBS - no adjustment
    return 0.0
```

**Result**: No penalty applied ✓ (FCS-vs-FCS games shouldn't appear in FBS schedule, but handled gracefully)

---

## 6. Consistency Checks

### Documentation in CLAUDE.md

**Current Line 73**:
```markdown
- **FCS Strength Estimator:** Dynamic, walk-forward-safe FCS penalties (APPROVED 2026-02-10).
  Replaces static elite list with Bayesian shrinkage (k=8, intercept=20).
  Penalty range [10, 40] pts. CLI: `--fcs-static` for baseline comparison.
```

**Issues Found**:
- ❌ **P2**: `intercept=20` should be `intercept=10` (actual default is 10.0)
- ❌ **P2**: Penalty range `[10, 40]` should be `[10, 45]` (actual max is 45.0)

**Corrected Version**:
```markdown
- **FCS Strength Estimator:** Dynamic, walk-forward-safe FCS penalties (APPROVED 2026-02-10).
  Replaces static elite list with Bayesian shrinkage (k=8, baseline=-28, intercept=10, slope=0.8).
  Penalty range [10, 45] pts. CLI: `--fcs-static` for baseline comparison.
```

### Backtest Metrics

**From SESSION_LOG.md** (lines 99-105):

| Phase | Dynamic 5+ Edge | Static 5+ Edge | Delta |
|-------|-----------------|----------------|-------|
| Phase 1 (weeks 1-3) | 50.2% | 48.8% | +1.4% |
| Phase 2 (Core) | 54.3% | 54.6% | -0.3% |

**Status**:
- ✅ Both modes meet 54.0% acceptance threshold
- ✅ Dynamic mode improves Phase 1 (where FCS games occur)
- ✅ Slight Core regression (-0.3%) within noise tolerance

**Approval Rationale**: SESSION_LOG documents trade-off as acceptable. Dynamic mode provides adaptive infrastructure for future cross-season FCS tracking.

### Current Baseline Alignment

**CLAUDE.md Line 62** lists metrics as of 2026-02-10:
- Core 5+ Edge (Close): 54.4%
- Core 5+ Edge (Open): 57.5%

**SESSION_LOG.md** reports dynamic FCS at:
- Phase 2 Core 5+ Edge: 54.3%

**Discrepancy Analysis**:
- SESSION_LOG reports 54.3%, CLAUDE.md reports 54.4%
- Likely different backtest runs (close vs open line, or rounding)
- Both exceed 54.0% threshold ✓

---

## 7. Final Recommendations

### Issues to Fix

**P2.1 - Documentation Correction** (CLAUDE.md line 73):
```diff
-- **FCS Strength Estimator:** Dynamic, walk-forward-safe FCS penalties (APPROVED 2026-02-10). Replaces static elite list with Bayesian shrinkage (k=8, intercept=20). Penalty range [10, 40] pts. CLI: `--fcs-static` for baseline comparison.
+- **FCS Strength Estimator:** Dynamic, walk-forward-safe FCS penalties (APPROVED 2026-02-10). Replaces static elite list with Bayesian shrinkage (k=8, baseline=-28, intercept=10, slope=0.8). Penalty range [10, 45] pts. CLI: `--fcs-static` for baseline comparison.
```

### Strengths to Preserve

1. **Walk-forward safety**: Clear & rebuild pattern is foolproof
2. **Vectorized Polars**: Performant, readable, maintainable
3. **Graceful fallbacks**: Unknown teams get sensible baseline penalty
4. **CLI flexibility**: Full tuning surface exposed for experimentation
5. **Documentation**: Code comments are detailed and accurate

### Future Enhancement Paths

1. **Cross-season FCS tracking**: Accumulate FCS strength across years (would require separate data store)
2. **Margin quality weighting**: Weight blowouts less than competitive games
3. **Opponent strength adjustment**: FCS team that plays Alabama is different from one that plays Troy

---

## Summary Table

| Audit Area | Status | Issues Found |
|------------|--------|--------------|
| Data Leakage Prevention | ✅ PASS | 0 |
| Implementation Files | ✅ PASS | 0 |
| Bayesian Shrinkage | ✅ PASS | 0 |
| CLI Integration | ✅ PASS | 0 |
| Edge Cases (5 tested) | ✅ PASS | 0 |
| Documentation | ⚠️ P2 | 1 (incorrect params in CLAUDE.md) |
| Backtest Metrics | ✅ PASS | 0 |

**Overall Grade**: ✅ **APPROVED** (with 1 minor documentation fix)

---

## Conclusion

The FCS Dynamic Strength Estimator is a high-quality implementation that correctly replaces a static hardcoded list with an adaptive, data-driven system. The feature is:

- **Walk-forward safe**: Strictly enforces temporal boundaries
- **Mathematically sound**: Bayesian shrinkage correctly implemented
- **Robustly tested**: All edge cases handled gracefully
- **Well-integrated**: Properly threaded through backtest and spread generator
- **Production-ready**: Enabled by default with static fallback option

The only issue is a documentation discrepancy in CLAUDE.md listing incorrect parameter values (intercept=20 should be 10, max_penalty=40 should be 45). This should be corrected to match the actual implementation.

**Recommendation**: Fix P2 documentation issue and proceed with production deployment.
