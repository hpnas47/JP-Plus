# Metrics Audit Report - 2026-02-07

## Audit Scope
Complete audit of all documentation files and Python docstrings for stale or incorrect hardcoded metrics following the Explosiveness Uplift (45/45/10 EFM weights).

## Current Production Baseline (2022-2025 Backtest)

| Slice | Weeks | Games | MAE | RMSE | ATS (Close) | ATS (Open) |
|-------|-------|-------|-----|------|-------------|------------|
| Full | 1–Post | 3,273 | 13.02 | 16.50 | 51.1% | 52.7% |
| Phase 1 | 1–3 | 597 | 14.94 | — | 47.1% | 48.6% |
| **Phase 2 (Core)** | **4–15** | **2,485** | **12.52** | **15.84** | **52.4%** | **54.0%** |
| Phase 3 (Post) | 16+ | 176 | 13.43 | — | 47.4% | 48.3% |
| 3+ Edge (Core) | 4–15 | 1,433 | — | — | 53.3% (764-669) | 55.5% (811-650) |
| **5+ Edge (Core)** | **4–15** | **866** | — | — | **54.6% (473-393)** | **56.9% (525-397)** |

**Key Parameters:**
- EFM Weights: SR=45%, IsoPPP=45%, Turnovers=10%
- Ridge alpha: 50.0
- Conference anchor: scale=0.08, prior_games=30, max_adjustment=2.0
- Garbage time: asymmetric, leading=1.0, trailing=0.1
- RZ leverage: rz_weight_20=1.5, rz_weight_10=2.0, empty_yards_weight=0.7

## Files Audited

### Documentation Files (**.md)
- ✅ `CLAUDE.md` - CORRECT (previously verified)
- ✅ `docs/MODEL_EXPLAINER.md` - CORRECT
- ✅ `docs/MODEL_ARCHITECTURE.md` - CORRECT
- ✅ `docs/PROJECT_MAP.md` - CORRECT
- ✅ `docs/SESSION_LOG.md` - CORRECT (historical notes properly dated)
- ✅ `docs/PERFORMANCE_AUDIT_2026-02-05.md` - CORRECT (historical baseline)
- ✅ `docs/Completed Audit Fixlists/*.md` - CORRECT (historical documentation)
- ✅ `.claude/agent-memory/model-strategist/MEMORY.md` - CORRECT
- ✅ `.claude/agent-memory/quant-auditor/explosiveness-uplift-backtest.md` - CORRECT
- ⚠️ `.claude/agent-memory/quant-auditor/MEMORY.md` - **FIXED**

### Python Files (**.py)
- ✅ `scripts/backtest.py` - No stale metrics in docstrings
- ✅ `src/models/efficiency_foundation_model.py` - No stale metrics in docstrings
- ✅ `config/settings.py` - No stale metrics in docstrings
- ✅ All other Python files - No stale weight references (54/36 or 60/40) found

## Issues Found and Fixed

### 1. Quant Auditor Memory (FIXED)

**File:** `.claude/agent-memory/quant-auditor/MEMORY.md`

**Issue:** Baseline metrics section referenced stale data from "Portal Decoupling" variant with incorrect numbers:
- Full MAE: 13.00 → **13.02** (correct)
- Core MAE: 12.51 → **12.52** (correct)
- ATS percentages and game counts all updated to match current baseline

**Fix Applied:** Updated all baseline metrics to reflect current 45/45/10 Explosiveness Uplift production baseline. Consolidated Conference Anchor and RZ Leverage sections to note they are now integrated into baseline (not separate experiments).

## Verification

All metrics in documentation now match the production baseline from:
```bash
python3 scripts/backtest.py --start-week 4
```

Run date: 2026-02-07
Commit: dd2c56c (Apply Explosiveness Uplift: equal SR/IsoPPP weights)

## Conclusion

✅ **AUDIT COMPLETE**

All documentation is now synchronized with the current production baseline. The only stale metrics found were in the Quant Auditor's memory file, which has been corrected.

No code changes were required - all Python files use dynamic configuration values from `config/settings.py` or pass parameters explicitly, preventing hardcoded metric drift.

---

**Auditor:** Code Auditor Agent
**Date:** 2026-02-07
**Files Scanned:** 67 markdown files, 15 Python files
**Issues Found:** 1
**Issues Fixed:** 1
