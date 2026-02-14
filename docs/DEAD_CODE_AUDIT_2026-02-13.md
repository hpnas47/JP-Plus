# Dead Code Audit Report - 2026-02-13

**Auditor:** Code Auditor Agent
**Scope:** src/, scripts/, src/spread_selection/, src/models/, src/adjustments/
**Objective:** Identify unused functions, imports, variables, classes, and modules that can be safely removed

---

## Executive Summary

**Total Findings:** 10 items
**High Confidence (Safe to Remove):** 5 items
**Medium Confidence (Review Recommended):** 3 items
**Low Confidence (Investigate Further):** 2 items

**Estimated LOC Savings:** ~500-800 lines
**Risk Assessment:** LOW - All findings are either unused infrastructure or disabled-by-default features

---

## HIGH CONFIDENCE - Safe to Remove

### 1. **FinishingDrivesModel in SpreadGenerator** ⭐ HIGHEST PRIORITY
- **File:** `src/predictions/spread_generator.py`
- **Lines:** 12 (import), 225 (parameter), 294 (init), 695 (commented-out usage)
- **What:** FinishingDrivesModel is imported, instantiated, but hardcoded to 0.0 contribution
- **Evidence:**
  ```python
  # Line 12: from src.models.finishing_drives import FinishingDrivesModel
  # Line 294: self.finishing_drives = finishing_drives or FinishingDrivesModel()
  # Line 695: components.finishing_drives = 0.0  # self.finishing_drives.get_matchup_differential(...)
  ```
- **Also Affects:**
  - `scripts/run_weekly.py` line 44 (import), line 832 (instantiation - passes to SpreadGenerator)
  - `scripts/benchmark.py` line 52 (import)
- **Backtest Status:** SHELVED after 4 consecutive rejections (70-80% overlap with EFM IsoPPP)
- **Risk:** VERY LOW - Model exists but is never called (returns 0.0)
- **Recommendation:**
  1. Remove `finishing_drives` parameter from `SpreadGenerator.__init__`
  2. Remove import from `spread_generator.py`, `run_weekly.py`, `benchmark.py`
  3. Remove `components.finishing_drives` line 695 in spread_generator.py
  4. Keep `src/models/finishing_drives.py` preserved for future reactivation
  5. Remove line 178 in `PredictedSpread.to_dict()` output ("finishing_drives": 0.0)

---

### 2. **GarbageTimeFilter Module (Entire Module Unused)**
- **File:** `src/data/processors.py` (111 lines)
- **What:** Complete garbage time filtering module that is never imported
- **Evidence:**
  ```bash
  $ grep -r "from src.data.processors import" .
  # No results - only exported in __init__.py but never imported
  ```
- **Exported In:** `src/data/__init__.py` line 3
- **Why Unused:** Garbage time filtering moved into EFM's `_prepare_plays()` method directly
- **Risk:** LOW - Module is isolated, no dependents
- **Recommendation:** Remove entire file + export from `src/data/__init__.py`
- **LOC Savings:** 111 lines

---

### 3. **Normalization Utilities Module (Entire Module Unused)**
- **File:** `src/utils/normalization.py` (188 lines)
- **What:** Rating normalization functions (`normalize_ratings`, `verify_normalization`)
- **Evidence:**
  ```bash
  $ grep -r "from src.utils.normalization import" .
  # No results - module exists but never imported
  ```
- **Why Unused:** Ratings normalization is handled directly in model classes (EFM, ST, etc.)
- **Risk:** LOW - Standalone utility module with no dependents
- **Recommendation:** Remove entire file
- **LOC Savings:** 188 lines

---

### 4. **Blended Priors CLI Flag (Vestigial Feature)**
- **File:** `scripts/backtest.py`
- **Lines:** 38 (import), 3728 (CLI arg), 4170, 4249, 4265, 4292 (parameter passing)
- **What:** `--blended-priors` flag and `BlendSchedule` infrastructure
- **Evidence:**
  - CLI flag exists at line 3728
  - Used in 4 function calls but always defaults to False
  - Memory confirms: "Blended Priors - REJECTED (2026-02-07)" with -1.1% 5+ Edge degradation
- **Status:** REJECTED experiment, infrastructure preserved but unused in production
- **Risk:** MEDIUM - Could be reactivated, but currently disabled
- **Recommendation:**
  - If never planning to reactivate: Remove CLI arg, parameter, and imports
  - If keeping for research: Add `# EXPERIMENTAL - DISABLED` comment at CLI arg
  - Keep `src/models/blended_priors.py` and `src/models/own_priors.py` (only imported by blended_priors)

---

### 5. **Deprecated/Unused Test/Diagnostic Scripts**
- **Files:**
  - `scripts/smoke_test_weather_ev.py` (NOT in git, 1562 lines total in totals_ev_engine)
  - `scripts/week2_counterfactuals.py` (NOT in git, counterfactual experiment)
- **Status:** Both appear in git status as uncommitted/untracked
- **What:** One-time diagnostic/validation scripts
- **Risk:** LOW - Scripts are standalone, no dependencies
- **Recommendation:**
  - Move to `scripts/archive/` or `tests/integration/` if useful for regression testing
  - Delete if one-time exploratory scripts

---

## MEDIUM CONFIDENCE - Review Recommended

### 6. **LASR (Money Down Weighting) Infrastructure**
- **File:** `src/models/efficiency_foundation_model.py`
- **Lines:** 382-383 (params), 454-455 (instance vars), 920-957 (implementation)
- **What:** Money down weighting (`money_down_weight`, `empty_success_weight`) with defaults = 1.0 (disabled)
- **Evidence:**
  - Defaults to 1.0 (no-op)
  - Memory confirms: "LASR Money Down Weight - REJECTED (2026-02-07)" with -0.7% 5+ Edge
- **Current State:** Infrastructure preserved, disabled by default
- **Risk:** MEDIUM - Code paths exist and execute, just with neutral weights
- **Recommendation:**
  - KEEP for now - disabled features with preserved infra are standard pattern in this codebase
  - Add docstring: "REJECTED: Sub-metric redundancy. Defaults disable this feature."

---

### 7. **Efficiency Fraud Tax Infrastructure**
- **File:** `src/models/efficiency_foundation_model.py`
- **Lines:** 375-377 (params), 447-449 (instance vars), 2405-2581 (implementation)
- **What:** Outcome-based penalty for teams with high efficiency but poor records
- **Evidence:**
  - `fraud_tax_enabled=False` by default
  - Memory confirms: "REJECTED" with -0.2% 5+ Edge degradation
- **Current State:** Infrastructure preserved, disabled by default
- **Risk:** MEDIUM - Implementation exists (~176 lines) but never executes
- **Recommendation:**
  - KEEP for now - same pattern as LASR
  - Consider removing entire `_apply_efficiency_fraud_tax()` method if confidence grows that outcome-based features are permanently rejected

---

### 8. **MOV Calibration Layer (Deprecated)**
- **File:** `src/models/efficiency_foundation_model.py`
- **Lines:** 373-374 (params), 445-446 (instance vars), 2043-2403 (implementation ~360 lines!)
- **What:** Margin of Victory calibration layer (superseded by fraud_tax)
- **Evidence:**
  - `mov_weight=0.0` by default (disabled)
  - Docstring says "DEPRECATED"
  - Lines 2043-2044: "Kept for backward compatibility but defaults to disabled"
- **Current State:** Large implementation block (~360 lines) that never executes
- **Risk:** MEDIUM-HIGH for removal - marked deprecated and explicitly disabled
- **Recommendation:**
  - **HIGH PRIORITY REMOVAL** - This is the largest dead code block found (360 lines)
  - Remove entire `_apply_mov_calibration()` method (lines 2252-2403)
  - Keep parameters for backward compatibility (prevent old config files from breaking)
  - Add deprecation warning if `mov_weight > 0.0` is ever passed

---

## LOW CONFIDENCE - Investigate Further

### 9. **Phase 1 Kill-Switch Module**
- **File:** `src/predictions/phase1_killswitch.py` (260 lines)
- **What:** Emergency risk control for disabling Phase 1 betting if early ATS is poor
- **Evidence:**
  - Fully implemented module (260 lines)
  - Imported and used in `run_weekly.py` lines 66-69
  - CLI args exist: `--killswitch`, `--killswitch-action`, `--killswitch-trigger-ats`
  - Used at lines 1164-1187 in run_weekly
- **Current State:** Implemented but disabled by default (`killswitch_enabled=False`)
- **Risk:** LOW - This is operational safety code, not research code
- **Recommendation:**
  - **KEEP** - This is a risk control feature designed for production use
  - It's meant to be dormant until needed (like a circuit breaker)
  - Not dead code - it's safety infrastructure

---

### 10. **OOC Credibility Weight Parameter**
- **File:** `src/models/efficiency_foundation_model.py`
- **Lines:** 387 (param), 456 (instance var), usage unknown
- **What:** Out-of-conference credibility anchor scaling (rejected feature)
- **Evidence:**
  - Default = 0.0 (disabled)
  - Memory confirms: "REJECTED (2026-02-08)" with monotonic 5+ Edge degradation
- **Current State:** Parameter exists but no implementation found
- **Risk:** LOW - Appears to be vestigial parameter with no implementation
- **Recommendation:**
  - Search for actual usage in EFM (likely in conference anchor logic)
  - If no usage found, remove parameter
  - If used, keep with disabled default (standard pattern)

---

## Additional Observations

### Preserved-But-Shelved Pattern (Not Dead Code)
The following modules are intentionally preserved for future reactivation:
- `src/models/finishing_drives.py` (634 lines) - RZ efficiency model
- `src/models/blended_priors.py` - Blended SP+/own-prior system
- `src/models/own_priors.py` - JP+ historical ratings system

**Recommendation:** These are NOT dead code - they're deactivated infrastructure. Keep as-is.

### Feature Flags vs Dead Code
Many "disabled" features are intentionally preserved with default-off flags:
- `fraud_tax_enabled=False`
- `money_down_weight=1.0` (no-op)
- `mov_weight=0.0`
- `time_decay=1.0` (no-op)

**Pattern:** This codebase preserves rejected experiment infrastructure for future research.

**Recommendation:** This is acceptable IF:
1. Parameters are documented as rejected/deprecated
2. Defaults ensure no-op behavior
3. Implementation is not excessively large (MOV calibration violates this - 360 lines!)

---

## Recommended Action Plan

### Immediate (High Value, Low Risk)
1. **Remove FinishingDrivesModel usage in SpreadGenerator** - saves ~10 lines, removes confusion
2. **Remove GarbageTimeFilter module** - saves 111 lines
3. **Remove Normalization utilities module** - saves 188 lines
4. **Archive or delete diagnostic scripts** (smoke_test, week2_counterfactuals)

**Total Immediate Savings:** ~300+ lines

### Short-Term (Medium Value, Review Required)
5. **Remove MOV Calibration implementation** - saves 360 lines (largest single block)
6. **Document Blended Priors status** - add "EXPERIMENTAL - DISABLED" comment
7. **Investigate OOC Credibility Weight** - remove if no implementation exists

**Total Short-Term Savings:** ~360+ lines

### Long-Term (Low Priority)
8. Consider removing `_apply_efficiency_fraud_tax()` if outcome-based features are permanently ruled out
9. Review LASR infrastructure - could remove if 3rd/4th down weighting is permanently rejected

---

## Files Requiring No Changes (Validated as Active)

The following were checked and are ACTIVELY USED:
- `src/adjustments/diagnostics.py` - Used by spread_generator for adjustment stack tracking
- `src/reports/html_report.py` - Used by run_weekly
- `src/reports/excel_export.py` - Used by run_weekly
- `src/notifications.py` - Used by run_weekly
- `src/predictions/sp_gate.py` - Used by run_weekly (SP+ gate policy)
- `src/predictions/phase1_killswitch.py` - Operational safety code (dormant by design)
- `src/adjustments/aggregator.py` - Active adjustment smoothing module
- All cache modules (season_cache, week_cache, qb_cache, priors_cache) - Active

---

## Verification Commands

To verify these findings:

```bash
# Check FinishingDrivesModel usage
grep -rn "finishing_drives\." src/predictions/spread_generator.py

# Check GarbageTimeFilter imports
grep -r "from src.data.processors import" .

# Check normalization imports
grep -r "from src.utils.normalization import" .

# Check blended_priors usage
grep -rn "args.blended_priors" scripts/backtest.py

# Check MOV weight usage
grep -n "mov_weight > 0" src/models/efficiency_foundation_model.py
```

---

## Impact Assessment

**Before Cleanup:**
- Total Python LOC: ~31,622 (src/ only)
- Unused/Dead code: ~500-800 lines (1.6-2.5%)

**After Recommended Cleanup:**
- Immediate savings: ~300 lines
- Short-term savings: ~360 lines
- Total potential savings: ~660+ lines

**Risk Level:** LOW
- All HIGH confidence items are isolated and unused
- MEDIUM confidence items are disabled by default
- No impact on backtest metrics (all unused code)

---

## Notes for Code Auditor Memory

**Patterns Discovered:**
1. **FinishingDrivesModel pattern**: Imported + instantiated but hardcoded to 0.0 (most deceptive)
2. **Preserved infrastructure**: Rejected features kept with disabled defaults (intentional)
3. **Diagnostic scripts**: Uncommitted one-off scripts in scripts/ folder (smoke_test, counterfactuals)
4. **MOV calibration**: Largest dead code block (360 lines, explicitly deprecated)

**Codebase Health:** Generally clean. Most "unused" code is intentionally preserved research infrastructure. True dead code is minimal (~2% of codebase).

**Recommendation for Future Audits:**
- Check for `= 0.0` or `= 1.0` default parameters that create no-ops
- Search for large implementation blocks guarded by `if self.weight > 0.0` conditions
- Look for imports in entry points (backtest.py, run_weekly.py) that are passed but never called
