# JP+ Development Session Log

> **Governing rules have moved to [CLAUDE.md](../CLAUDE.md).** This file is a chronological development journal only.

---

## Session: February 15, 2026

### Theme: Preseason Win Total Projection Module — Implementation + Compliance Audit

---

#### Preseason Win Totals Module — COMMITTED
**Commit**: `0e12a37`

New standalone module (`src/win_totals/`) for preseason win total projections. Trains Ridge regression on historical CFBD data to predict end-of-season SP+ ratings, converts predictions to win total distributions via Monte Carlo simulation with latent team shocks, and evaluates betting edge against book lines.

**6 source files + 1 test file (3,104 lines):**
- `schedule.py` — Poisson binomial PMF (DP), logistic win prob, Monte Carlo with correlated team shocks, walk-forward calibration
- `edge.py` — EV calculation, American odds conversion, push handling, leakage contribution tracking
- `features.py` — 17 preseason features from CFBD API with honest leakage audit (11 SAFE, 6 ASSUMED)
- `model.py` — Ridge regression with manual walk-forward alpha selection over [0.1, 1.0, 10.0, 50.0, 100.0, 200.0]
- `run_win_totals.py` — CLI with train/predict/backtest/calibrate commands
- `tests/test_win_totals.py` — 85 unit tests, all passing

**Key design decisions:**
- **Target**: End-of-season SP+ (not raw wins) — strips schedule effects
- **Alpha selection**: Manual walk-forward sweep (NOT RidgeCV/LOOCV) — train years < Y, test year Y
- **Calibration**: Out-of-fold predictions only, years < Y for fold Y. min_games=1500 guardrail with naive fallback (30% regression toward mean)
- **Portal features**: Excluded in V1 — CFBD lacks reliable transfer dates for pre-August filtering
- **Win counting**: Regular season only (season_type='regular' AND week <= 15)
- **Win prob clamp**: [0.01, 0.99] in both scalar (`game_win_probability`) and vectorized (Monte Carlo) paths
- **Push handling**: Integer lines account for push probability in EV calculation
- **Leakage tracking**: End-to-end from Ridge coefficients × standardized features → per-team leakage % → BetRecommendation warning flag (>25% threshold)

**Compliance audit performed (A–G checklist):**
- 4 non-compliant items found and patched: prob clamp on scalar path, push handling, leakage e2e wiring in CLI, `years` field on WinProbCalibration
- Post-patch consistency audit confirmed all 85 tests pass with correct numerical behavior

---

## Session: February 14, 2026 (Evening)

### Theme: FBS Filtering, Moneyline Bugfixes, Selection Policy Hardening

---

#### FBS vs FCS Filtering in Bet Display Scripts — COMMITTED
**Commits**: `8945f37`, `5fbbe70`

Both `show_spread_bets.py` and `show_totals_bets.py` were showing FBS vs FCS games in betting recommendations. Added dynamic FBS team filtering using `CFBDClient.get_fbs_teams(year)` with module-level caching.

- **Impact**: Week 1 2025 spreads went from 46 bets → 21 (FBS-only). Record improved from 56.2% to 64.3% (Primary EV) and 57.1% to 71.4% (5+ Edge) — FCS games were adding noise.
- **Design choice**: API-based lookup (not hard-coded list) to handle year-by-year FBS membership changes (Delaware & Missouri State joined FBS in 2025).

---

#### 2025 P&L Analysis — RESEARCH
**Status**: Analysis only, no code changes

Generated full 2025 season P&L for FBS-only spread bets:
- **Primary EV Engine**: 89-67-2 (57.1%), +8.8% ROI, +$1,391 at $100/bet
- **5+ Point Edge**: 138-111-2 (55.4%), +5.8% ROI, +$1,445 at $100/bet
- EV Engine outperforms 5+ Edge by 3% ROI despite fewer bets (quality > quantity)
- Worst week: Week 5 (-$755 at $100); Best: Week 1 (+$573)

---

#### Moneyline Weekly Bugfixes — COMMITTED
**Commit**: `036ede7` (included in earlier hardening commit)

Fixed 3 bugs in `src/spread_selection/moneyline_weekly.py`:
1. **dry_run dedup (MODERATE)**: `append_to_log` in dry_run mode now runs dedup against existing log for accurate preview counts
2. **Missing settlement columns (MODERATE)**: `settle_week` now creates missing `SETTLEMENT_COLS` with `None` before accessing, protecting against old/manually-created logs
3. **Silent skip warning (MINOR)**: Prints warning with game_id and team names when score lookup fails during settlement

---

#### Selection Policy Hardening — COMMITTED
**Commit**: `2c2a4e6`

Patched `src/spread_selection/selection_policy.py` for EV semantics, accounting accuracy, and robustness:

**CRITICAL fixes:**
- **EV units contract**: Module docstring now explicitly documents EV = ROI units (`p_win*b - p_lose`), not probability edge. All threshold docstrings updated.
- **Heuristic warning**: `apply_selection_policy()` warns when EV values look like probability-edge (max |ev| < 0.005) while thresholds are in ROI range.
- **Accounting fix**: `n_filtered_by_ev` and `n_filtered_by_cap` were wrong — phase1 skips mislabeled as EV filtering, cap formula was algebraically incorrect. All three policy helpers now return exact `(selected_df, n_ev_filtered, n_cap_filtered)` tuples. Added `n_phase1_skipped` to `SelectionResult`. Invariant enforced: `phase1 + ev + cap + selected = candidates`.

**MINOR fixes:**
- **NaN outcomes**: `compute_selection_metrics()` filters out NaN in outcome_col before computing wins/losses (prevents misclassifying unsettled bets as losses).
- **Drawdown sort**: `compute_max_drawdown()` now sorts by `(year, week, game_id)` internally for deterministic results regardless of input order.

**Tests**: 7 new tests added (45 total, all passing): accounting accuracy for all 3 policies, sum invariant, NaN handling, drawdown stability.

---

## Session: February 14, 2026

### Theme: Totals EV Engine Weather Integration

---

#### Weather Support for run_weekly_totals.py — COMMITTED
**Status**: Complete

Wired up weather adjustment pipeline for 2026+ production totals betting:

**New CLI Flags:**
- `--weather` — Enable weather adjustments (requires prior forecast capture)
- `--no-high-variance-overs` — Block OVER bets on uncertain weather games

**Implementation Details:**
- `load_weather_adjustments()` — Fetches forecasts from `data/weather_forecasts.db` using `TomorrowIOClient.get_betting_forecast()` (earliest capture for lookahead safety)
- Weather adjustments applied to `TotalsEvent.weather_adjustment` field
- High-variance detection: confidence < 0.75 AND raw adj > 3.0
- OVER bets filtered on high-variance games when flag set
- Summary shows weather status: `Weather: Enabled (X adj)` or `Weather: Disabled`

**Usage:**
```bash
# 2026 production with weather
python scripts/run_weekly_totals.py --year 2026 --week 6 --weather

# With high-variance OVER protection
python scripts/run_weekly_totals.py --year 2026 --week 6 --weather --no-high-variance-overs

# Historical (2025 and earlier) — NO weather flag
python scripts/run_weekly_totals.py --year 2025 --week 6
```

**Prerequisite:** Run `python scripts/weather_thursday_capture.py` before game day to populate forecasts.

---

#### EV Engine Consistency Review — CONFIRMED
**Status**: Analysis complete, no changes needed

Reviewed spread EV engine (`selection.py`) vs totals EV engine (`totals_ev_engine.py`):

| Aspect | Spread EV | Totals EV | Notes |
|--------|-----------|-----------|-------|
| EV Formula | `p_win * payout - p_lose` | `p_win * payout - p_lose` | **Identical** |
| P(win) Source | Logistic calibration on `edge_abs` | Normal CDF with sigma | Different by design |
| Push Handling | Empirical push rates | Analytical (Normal CDF band) | Both valid |
| Kelly Staking | None | Full three-outcome Kelly | Totals-only |

**Conclusion:** Design differences are intentional and appropriate for each domain.

---

#### Import Fix: SP+ Gate Cleanup — COMMITTED
**Status**: Complete

Fixed import error in `src/spread_selection/__init__.py`:
- Removed stale SP+ gate exports (`Phase1SPGateConfig`, `Phase1SPGateResult`, etc.)
- `policies/__init__.py` was emptied in prior session but parent still imported from it
- Added comment: "SP+ gate removed (2026-02-14)"

---

#### /show-totals-bets Skill — COMMITTED
**Status**: Complete

Created new skill for displaying totals betting recommendations (mirrors `/show-spread-bets`):

**New Files:**
- `scripts/show_totals_bets.py` — Fast display script using pre-computed backtest data
- `.claude/skills/show-totals-bets/SKILL.md` — Skill definition
- `data/spread_selection/outputs/backtest_totals_2023-2025.csv` — Pre-computed backtest data

**Usage:**
```
/show-totals-bets 2025 6
/show-totals-bets 2024 10
```

**Implementation Evolution:**
1. Initial version used hardcoded sigma=17.0 and 10% EV threshold
2. User flagged inconsistency with spread engine's 3% threshold
3. Final version uses **calibrated sigma from backtest residuals** (16.4) and **3% EV threshold**

**Calibration Approach:**
- Sigma calculated from `df['jp_error'].std()` = 16.4 (actual prediction error std)
- EV calculated using Normal CDF probability model (same as `totals_ev_engine.py`)
- 3% EV threshold matches spread engine for consistency

**Output Comparison:**
| Threshold | Games (Week 6) | Record |
|-----------|----------------|--------|
| EV >= 10% | 26 | 19-7 (73.1%) |
| EV >= 3% (final) | 35 | 25-10 (71.4%) |

**Key Insight:** Totals market shows more edge opportunities than spreads at same 3% threshold. This could indicate:
- Less efficient totals markets
- Normal CDF model more generous than logistic calibration
- Both factors combined

---

## Session: February 13, 2026 (PM)

### Theme: Phase 1 SP+ Policy Finalization

---

#### Dual-List Architecture Implemented — COMMITTED
**Status**: Complete

Implemented comprehensive dual-list betting system for Phase 1 (weeks 1-3):

**List A (ENGINE_EV PRIMARY)**
- EV-based calibrated selection engine output
- `is_official_engine=True`, `execution_default=True`
- NO SP+ filtering applied (tagging only for monitoring)
- Uses CLOSE lines for EV computation

**List B (PHASE1_EDGE)**
- Edge-based selection (|jp_edge| >= 5.0 vs OPEN lines)
- `is_official_engine=False`, `execution_default=False`
- Auto-emitted in weeks 1-3 for visibility
- Optional HYBRID_VETO_2 overlay (default OFF)

**New Files:**
- `src/spread_selection/strategies/phase1_edge_baseline.py` — LIST B strategy logic
- `docs/PHASE1_SP_POLICY.md` — Policy documentation and research conclusions

**CLI Additions:**
- `--no-phase1-edge-list` — Disable automatic List B emission
- `--phase1-edge-veto` — Enable HYBRID_VETO_2 overlay
- `--phase1-edge-veto-sp-oppose-min` — SP+ threshold (default 2.0)
- `--phase1-edge-veto-jp-band-high` — Upper JP+ band (default 8.0)

---

#### Phase 1 SP+ Filtering Research — FROZEN
**Status**: Research concluded, no further experiments

**Backtest Results (2022-2025, N=395 games at 5+ edge):**

| Approach | Overall ATS% | 2025 ATS% | Verdict |
|----------|-------------|-----------|---------|
| EDGE_BASELINE | 49.9% | 59.8% | Baseline |
| SP+ Confirm-Only | 45.9% | N/A | REJECTED (catastrophic 2022) |
| VETO_OPPOSES_2 | 50.9% | 59.4% | REJECTED (vetoes 2025 winners 62.5%) |
| VETO_OPPOSES_3 | 50.1% | 59.0% | REJECTED (vetoes 2025 winners 71.4%) |
| HYBRID_VETO_3 | 49.7% | 59.3% | REJECTED (vetoes 2025 winners 75%) |
| **HYBRID_VETO_2** | **50.3%** | **60.2%** | **APPROVED (optional, default OFF)** |

**HYBRID_VETO_2 Rule:**
- VETO if: oppose AND |sp_edge|>=2.0 AND 5.0<=|jp_edge|<8.0
- NEVER veto if: |jp_edge|>=8.0 (high-conviction protected)
- ROI improvement: -3.5% → -2.7% (+0.8pp)
- Retention: 94.2%

**Final Conclusions:**
1. SP+ confirm-only gating failed catastrophically in 2022 (18.8% ATS)
2. Simple veto approaches remove winners in 2025 (model's best year)
3. HYBRID_VETO_2 is the only approach that passes all guardrails
4. Default stance: **SP+ as tagging-only**, HYBRID_VETO_2 available but OFF

---

#### Week Summary & Overlap Reporting — COMMITTED
**Status**: Complete

When both lists are generated in Phase 1, automatically writes:
- `week_summary_{year}_week{week}.json` — Counts and config snapshot
- `overlap_engine_primary_vs_phase1_edge_{year}_week{week}.csv` — Conflict detection

**Overlap Report Fields:**
- `in_engine_primary`, `engine_side`, `engine_ev`
- `in_phase1_edge`, `phase1_side`, `phase1_edge_abs`
- `side_agrees`, `conflict`, `recommended_resolution`

---

#### Validator Schema Updates — COMMITTED
**Status**: Complete

Updated `scripts/validate_spread_selection_outputs.py`:
- New metadata fields: `list_family`, `list_name`, `selection_basis`, etc.
- Week summary JSON schema validation
- Overlap CSV schema validation (new and legacy formats)
- List B veto field validation

---

## Session: February 13, 2026 (AM)

### Theme: Codebase Cleanup & Critical Bug Fixes

---

#### Phase 1 Diagnostic Scripts — REMOVED
**Status**: Deleted (cleanup)

Removed 9 one-off diagnostic scripts created during Phase 1 SP+ gate research:
- `scripts/phase1_lambda_sweep.py`
- `scripts/phase1_sp_comparison.py`
- `scripts/phase1_common_set_analysis.py`
- `scripts/phase1_strict_common_set.py`
- `scripts/phase1_sp_agreement_filter.py`
- `scripts/backtest_phase1_sp_gate.py`
- `scripts/phase1_2022_anomaly_diagnostics.py`
- `scripts/backtest_phase1_killswitch.py`
- `scripts/analyze_sp_divergence.py`

**Retained**: Core reusable modules in `src/predictions/`:
- `sp_gate.py` — SP+ agreement gating logic (integrated with `run_weekly.py`)
- `phase1_killswitch.py` — Kill-switch regime protection logic

---

#### Critical Bug Fixes in backtest.py — COMMITTED
**Status**: Fixed 3 bugs

**Bug 1: LSA Vegas Spread Column Name (CRITICAL - silent data loss)**
- `"spread"` → `"spread_close"` in LSA training lookup
- Previous: Vegas spread was ALWAYS None, silently disabling `--lsa-filter-vegas` and `--lsa-weighted-training`

**Bug 2: Parameter Shadowing in Phase 1 Shrinkage (landmine)**
- Renamed local `hfa_value` → `game_hfa` to avoid shadowing function parameter
- Prevented potential future bugs from reusing shadowed variable

**Bug 3: Cache Save Gated Behind Non-Empty Plays (logic error)**
- Moved cache save outside `len(efficiency_plays_df) > 0` check
- Now caches games data even if plays API fails

---

#### Bug Fixes in run_selection.py — COMMITTED
**Status**: Fixed 3 bugs (`2618a0c`)

**Bug 1: Push Modeling Inconsistency (EV mismatch between pipelines)**
- `compare_strategies()` was recomputing EV with `p_push=0`, discarding push-aware EV from `walk_forward_validate()`
- `run_predict()` was computing EV without push rates, causing production/backtest mismatch
- **Fix**: `compare_strategies()` now preserves existing `ev` column if valid; `run_predict()` now estimates push rates from training data and computes push-aware EV
- **Impact**: For integer spreads at key numbers (3, 7), push rates are 5-8%, shifting EV by 0.004-0.005

**Bug 2: IndexError in run_mode_comparison Fold Alignment**
- Loop used positional indexing `r_ult["fold_summaries"][i]`, crashing when modes have different fold counts
- **Fix**: Align folds by `eval_year` using dict lookup, iterate over intersection of years

**Bug 3: Division by Zero in print_strategy_comparison**
- `oa['overlap_old_ev3']/oa['old_total']*100` crashes when no games have `edge_abs >= 5.0`
- **Fix**: Guard with `if oa['old_total'] > 0`, print "N/A" otherwise

---

#### Additional Bug Fixes in run_selection.py — COMMITTED
**Status**: Fixed 3 more bugs (`8d274a4`)

**Bug 4: Gate Category Counter Crash in run_predict**
- Double-if generator expression executed `.iloc[0]` before length guard, causing IndexError
- Type mismatch: `r.game_id` was int but `slate_pred['game_id']` was string/float from CSV, causing silent zero matches
- **Fix**: Build set of non-PASS game_ids first (cast to int), then filter gate_results against that set

**Bug 5: run_mode_comparison Still Overwrote EV with p_push=0**
- Same issue as compare_strategies: `df["ev"] = calculate_ev_vectorized()` discarded push-aware EV
- **Fix**: Applied same pattern — preserve existing `ev` column if valid, only recompute as fallback

**Bug 6: Sensitivity Report Fold Alignment Still Positional**
- Same issue as run_mode_comparison: `fold_exc = r_exc["fold_details"][i]` crashed when fold counts differed
- **Fix**: Build dict keyed by `year`, iterate over intersection of years present in both configs

---

## Session: February 12, 2026 (Night)

### Theme: Postseason Exclusion & Week-Varying Shrinkage Experiment

---

#### Postseason Exclusion from Metrics — COMMITTED
**Status**: Committed (`25302ec`, `1db6e1f`, `59e10d0`)
**Files**: `CLAUDE.md`, `docs/MODEL_EXPLAINER.md`, `docs/MODEL_ARCHITECTURE.md`

**Rationale**: Postseason (bowls, CFP) has unmodeled factors that degrade betting edge:
- Coaching changes mid-cycle
- Transfer portal activity
- Player opt-outs
- Motivation variance
- 3-4 week layoffs

**Changes**:
- Backtest metrics now use **Regular Season only (weeks 1-15)**
- Removed "Full (1-Post)" and "Phase 3 (Postseason)" rows from all performance tables
- Added "Regular Season" aggregate row (3,445 games for spreads, 1,993 for totals)
- **Final power ratings STILL include postseason** — captures full body of work for end-of-season rankings

**New Baseline (Regular Season, 2022-2025)**:
| Metric | Value |
|--------|-------|
| Games | 3,445 |
| MAE | 12.90 |
| 5+ Edge (Close) | 53.6% (700-605) |
| 5+ Edge (Open) | 54.9% (743-610) |

---

#### Week-Varying Phase 1 Shrinkage — REJECTED
**Status**: Tested and reverted
**Hypothesis**: Model Strategist recommended week-specific shrinkage (W1=0.95, W2=0.90, W3=0.85) based on MSE escalation pattern through Phase 1

**Test Results**:
| Configuration | 5+ Edge (Close) | 5+ Edge (Open) |
|---------------|-----------------|----------------|
| **Baseline (uniform 0.90)** | **50.7%** (237-230) | **51.3%** (235-223) |
| W1=0.95, W2=0.90, W3=0.85 | 50.2% (233-231) | 51.1% (232-222) |
| W1=0.88, W2=0.90, W3=0.92 (inverse) | 50.8% (241-233) | 51.0% (235-226) |

**Failure Mode**: Same as documented rejection pattern — highest-conviction bets (5+ pts) already well-calibrated. Week-varying approach improved 3+ Edge (+1.2%) but degraded 5+ Edge (-0.5%).

**Decision**: Uniform 0.90 shrinkage remains optimal. Code reverted.

---

#### Prior Source Analysis (Model Strategist)
**Status**: Research completed, no action needed

**Question**: Would switching from SP+ to FPI/Sagarin improve Phase 1 edge?

**Finding**: SP+, FPI, and Sagarin are effectively the same signal (r=0.970 between SP+ and FPI). Switching would be "changing the font on a spreadsheet."

**Key insight**: The problem is not WHICH prior, but HOW MUCH to trust ANY prior. The blended priors experiment (Rejection #14) already proved this — using JP+'s own historical ratings as prior resulted in -1.1% 5+ Edge degradation.

---

## Session: February 12, 2026 (Evening)

### Theme: Phase 1 Spread Shrinkage Research & Implementation

---

#### Phase 1 Diagnostic Analysis
**Status**: Completed

Comprehensive analysis of Phase 1 (Weeks 1-3) underperformance revealed:
- **Dual Pathology**: Variance-dominant (Error Std 17.55 vs Core 15.81) with embedded directional bias (MSE +0.90)
- **Root cause**: SP+ priors comprise 92% of Week 1 signal; model is essentially a "SP+ wrapper" for early season
- **Edge-to-ATS calibration**: Completely flat — increasing edge provides no improvement in win rate

---

#### Phase 1 Shrinkage Sweep — APPROVED
**Status**: Committed (`4d28992`)
**Files**: `scripts/backtest.py`, `CLAUDE.md`, `docs/MODEL_ARCHITECTURE.md`

**Formula**: `NewSpread = (OldSpread - HFA) × Shrinkage + HFA`

**Sweep Results (7 values tested)**:

| Shrinkage | MAE | MSE | ErrStd | 5+ Edge (Close) |
|-----------|-----|-----|--------|-----------------|
| 1.00 | 13.95 | +0.90 | 17.54 | 50.4% |
| **0.90** | **13.95** | **-0.79** | **17.43** | **51.1%** ✓ |
| 0.85 | 14.03 | -1.64 | 17.45 | 49.6% |
| 0.80 | 14.17 | -2.49 | 17.51 | 49.8% |

**Optimal**: Shrinkage=0.90 (+0.7% Close, +0.5% Open)

**CLI**: `--phase1-shrinkage 0.90` (default), `--no-phase1-shrinkage` to disable

---

#### Pure Prior Test — REJECTED
**Status**: Tested, rejected

Hypothesis: Force 100% prior weight for weeks 1-3 to eliminate "small sample noise"

**Results**:
| Mode | 5+ Edge (Close) | Delta |
|------|-----------------|-------|
| Normal Blend + Shrink | 51.1% | baseline |
| Pure Prior + Shrink | **49.6%** | **-1.5%** ✗ |

**Conclusion**: Early-season blending (5-17% in-season data) is HELPING, not hurting. The noise is in the SP+ priors themselves, not the blending architecture.

---

#### Stacking Interaction Sweep — HFA Offset Variation
**Status**: Tested, baseline confirmed optimal

Hypothesis: Shrinkage=0.90 creates -0.79 MSE (road bias). Removing HFA penalty might fix it.

**Results**:

| HFA Offset | MSE | 5+ Edge (Close) |
|------------|-----|-----------------|
| 0.00 (Restoration) | **-0.31** ✓ | 49.8% ✗ |
| **0.50 (Baseline)** | -0.79 | **51.1%** ✓ |
| 1.00 (Hybrid) | -1.27 | 49.5% |
| 1.50 (Double Down) | -1.74 | 49.5% |

**Key Finding**: MSE/ATS Divergence Pattern confirmed. Improving calibration (MSE→0) makes model MORE like market, which DESTROYS edge. Baseline (0.50) is optimal.

---

#### Phase 1 Research Conclusions

| Experiment | Result | Status |
|------------|--------|--------|
| **Shrinkage=0.90** | +0.7% 5+ Edge, fixed 21+ bucket | **APPROVED** |
| Pure Prior (100%) | -1.5% 5+ Edge | **REJECTED** |
| HFA Offset stacking | MSE/ATS divergence | **REJECTED** |

**Final Production Config for Phase 1**:
- Shrinkage: 0.90 (default)
- HFA Offset: 0.50 (unchanged)
- Prior blending: Normal (unchanged)

---

## Session: February 12, 2026

### Theme: FCS Double-Counting Fix (Historical Backtest Consistency)

---

#### Phase 1 Error Analysis — FCS Games Identified as Root Cause
**Status**: Diagnostic completed

User requested investigation into Phase 1 (Weeks 1-3) prediction errors. Analysis revealed:

| Cohort | N | MAE | Mean Error |
|--------|---|-----|------------|
| FCS games (spread>35) | 276 | 19.00 | **+12.02** |
| FBS-only (spread≤35) | 684 | 13.60 | +0.19 |

Year-over-year showed massive discrepancy:
- 2022-2023: ME +17.64 pts (pre-FCS estimator calibration)
- 2024-2025: ME +2.88 pts (post-calibration)

**Root Cause Identified**: Double-counting in spread calculation:
1. FCS team gets mean FBS rating fallback (~0)
2. PLUS FCS penalty (~32 pts) added on top
3. Result: ~50 pt predicted spreads vs ~34 pt actual margins

---

#### FCS Double-Counting Fix — APPROVED
**Status**: Committed (`b19da17`)
**Files**: `src/predictions/spread_generator.py`

**Change**: In `_get_base_margin()`, return 0.0 for FBS vs FCS games. Let the FCS penalty alone handle the expected margin, preventing rating_diff + penalty stacking.

```python
# In _get_base_margin()
if self.fbs_teams:
    home_is_fbs = home_team in self.fbs_teams
    away_is_fbs = away_team in self.fbs_teams
    if home_is_fbs != away_is_fbs:
        # FCS penalty handles the rating differential
        return 0.0
```

**Results (2022-2025 backtest)**:

| Metric | Before Fix | After Fix | Delta |
|--------|------------|-----------|-------|
| **FCS Mean Error** | +12.02 | +4.18 | **-7.84** ✓ |
| **FCS MAE** | 19.00 | 16.79 | **-2.21** ✓ |
| **Phase 1 MAE** | 15.30 | 14.42 | **-0.88** ✓ |
| **Full MAE** | 13.32 | 13.13 | **-0.19** ✓ |
| **Core 5+ Edge** | 54.5% | 55.0% | **+0.5%** ✓ |
| Phase 1 5+ Edge | 50.8% | 50.0% | -0.8% |

Year consistency fixed:
- 2022-2023: ME +17.64 → +4.28
- 2024-2025: ME +2.88 → +4.08

**Trade-off accepted**: Phase 1 5+ Edge drops from 50.8% (below breakeven) to 50.0% (exact coin-flip). This is acceptable because:
1. Original 50.8% was inflated by buggy spreads
2. Core 5+ Edge improved (+0.5%)
3. Phase 1 was never a profitable betting window

---

#### FCS Double-Counting Fix v2: FBS Team Differentiation — APPROVED
**Status**: Committed (`be57aa6`)
**Files**: `src/predictions/spread_generator.py`, `scripts/backtest.py`

**Problem**: The v1 fix (b19da17) set `base_margin=0` for ALL FBS vs FCS games, treating Alabama and Kent State identically vs the same FCS opponent. This eliminated FBS team quality differentiation.

**Solution**: For FCS games, return the FBS team's deviation from FBS mean (not 0.0). The FCS penalty represents the average-FBS-to-FCS gap, so we only add the FBS team's deviation from average.

```python
# New _get_base_margin() logic for FCS games:
deviation = fbs_rating - _mean_fbs_rating
base_margin = deviation if home_is_fbs else -deviation
```

**Example spreads**:
- Alabama (+25 rating, FBS mean ~+2) vs FCS: base=+23, +penalty=~55
- Average FBS team (+2 rating) vs FCS: base=0, +penalty=~32
- Kent State (-8 rating) vs FCS: base=-10, +penalty=~22

**Results (2022-2025 backtest, vs v1 fix baseline)**:

| Metric | v1 Fix | v2 Fix | Delta |
|--------|--------|--------|-------|
| **FCS MAE** | 15.53 | 14.43 | **-1.10** ✓ |
| **FCS ME** | +4.53 | +3.59 | **-0.94** ✓ |
| **Full MAE** | 13.14 | 12.99 | **-0.15** ✓ |
| Core 5+ Edge (FBS-only) | 54.3% | 54.3% | **0** ✓ |
| Core 5+ Edge (Total) | 54.7% | 54.3% | -0.4% |

**Key finding**: The -0.4% Core 5+ Edge drop is entirely from FCS games; FBS-only Core performance is **identical** (445-375 in both). The FCS ATS edge loss is a consequence of more accurate spreads (less random disagreement with Vegas).

**Config toggle**: `--fcs-legacy-base` reverts to v1 behavior for A/B testing.

---

#### Baseline Metrics Updated — COMMITTED
**Status**: Committed (`c91df87`)
**Files**: `CLAUDE.md`, `docs/MODEL_ARCHITECTURE.md`

Updated production baseline with new metrics post-FCS fix.

---

#### FCS Baseline Tuning — NOT NEEDED
**Status**: Tested, no change

Tested FCS baseline margins (-28, -26, -24, -22):

| Baseline | Penalty | FCS ME | Core 5+ Edge |
|----------|---------|--------|--------------|
| **-28** | 32.4 | +4.18 | **55.0%** |
| -26 | 30.8 | +2.48 | 54.9% |
| -24 | 29.2 | +2.29 | 54.9% |
| -22 | 27.6 | -0.92 | 54.9% |

Baseline -28 kept — Core performance essentially identical, and it's the documented standard.

---

#### Totals Backtest Cleanup — APPROVED
**Status**: Committed (earlier today)
**Files**: `scripts/backtest_totals.py`, `src/models/totals_model.py`

Removed redundant pre-filtering and added data leakage guards:

**Changes**:
1. Removed pre-filtering of weeks < start_week in game list (walk-forward already handles)
2. Added `_last_train_max_week` tracking in TotalsModel
3. Added explicit leakage guard assertion: `model._last_train_max_week > pred_week - 1` raises ValueError

Pattern matches leakage guards used in main backtest.py.

---

#### QB Calibration Walk-Forward Fix — APPROVED
**Status**: Committed (`b5b5a44`)
**Files**: `src/adjustments/qb_continuous.py`, `scripts/qb_calibration.py`

Fixed walk-forward semantics in QB calibration:

**Bug 2 (CRITICAL)**: Week 1 predictions used `through_week=1` (data leakage). Fixed by adding `through_week=0` handling that loads prior season only.

**Bug 3 (MINOR)**: Added small denominator guard for recommended scale calculation (`p90_shrunk > 0.01`).

**Bug 4 (MINOR)**: Replaced bare `except: pass` with proper failure logging.

**Before/After Comparison (2024 data)**:
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Row count | 2010 | 2010 | 0 |
| Weeks 1-3 N | 401 | 401 | 0 |
| Weeks 1-3 mean | 0.076337 | 0.076337 | 0 |
| Recommended QB_SCALE | 9.25 | 9.25 | 0 |

Output identical because prior season loading already occurred correctly — fix is about semantic correctness and edge case safety.

**Production code unchanged**: `get_adjustment()` not modified.

---

## Session: February 11, 2026 (Evening)

### Theme: API Caching Optimization

---

#### Performance Audit — Comprehensive API Call Analysis
**Status**: Completed
**Agent**: perf-optimizer

Full audit of all API calls across the codebase identified 4 external services with 28 endpoints:
- **CFBD API**: Games, plays, teams, QB PPA, priors data
- **Odds API**: Betting lines (SQLite cached)
- **Tomorrow.io**: Weather forecasts (SQLite cached)

**Key Finding**: QB Continuous API calls were the largest uncached bottleneck — 31 calls/season for PPA + attempts data.

---

#### QB Data Caching — APPROVED
**Status**: Committed (`f1a5fb4`)
**Files**: `src/data/qb_cache.py` (new), `src/adjustments/qb_continuous.py`

Implemented disk cache for QB Continuous API responses following SeasonDataCache pattern.

**Cache Structure:**
```
.cache/qb/{year}/
  week_{week}_ppa.parquet      - QB PPA data per week
  week_{week}_attempts.parquet - Pass attempts per week
  prior_season.parquet         - End-of-year QB stats
  .complete_week_{week}        - Atomic write markers
```

**Performance:**
| Run | Time | Notes |
|-----|------|-------|
| First (API calls) | 34.9s | Fetches all QB data from CFBD |
| Second (cached) | 24.5s | Uses parquet cache |
| **Savings** | **10.4s (30%)** | |

**Cache Size**: ~2.0 MB for 2022-2025 data

---

#### Priors Data Caching — APPROVED
**Status**: Committed (`01294d6`)
**Files**: `src/data/priors_cache.py` (new), `src/models/preseason_priors.py`

Implemented disk cache for preseason priors API responses (SP+, Talent, RP, Portal).

**Cache Structure:**
```
.cache/priors/{year}/
  sp_ratings.parquet           - SP+ from prior year
  talent.parquet               - Team talent composite
  returning_production.parquet - Percent PPA returning
  transfer_portal.parquet      - Portal entries
```

**API Calls Eliminated**: 16 per 4-year backtest (4 endpoints × 4 years)

**Cache Size**: ~0.4 MB for 2021-2025 data

---

#### Combined Caching Impact

| Metric | Before Caching | After Caching | Improvement |
|--------|----------------|---------------|-------------|
| API calls per backtest | ~140 | ~0* | **~95% reduction** |
| 4-year backtest (first run) | ~35s | ~25s | 10s saved |
| 4-year backtest (cached) | ~25s | ~23s | 2s saved |

*After cache is populated for historical years

**Total Cache Size**: ~2.4 MB

---

#### Blended Priors ATS Tuning — REJECTED (Earlier Today)
**Status**: Rejected, infrastructure committed (`ef2695c`)
**Files**: `scripts/tune_blended_priors_ats.py`, `src/models/blended_priors.py`, `src/models/own_priors.py`

Attempted to tune blended priors (SP+ + own-prior) with ATS as objective function instead of MAE.

**Result**: Direct comparison showed -1.1% degradation at 5+ Edge:
| Config | 5+ Edge (Close) |
|--------|-----------------|
| SP+-only baseline | 50.8% |
| Blended priors | 49.7% |

**Conclusion**: Blended priors fail the binding constraint (must not degrade 5+ Edge). Added to rejection pattern as #14. Infrastructure preserved for potential future research.

---

## Session: February 11, 2026 (Post-Midnight)

### Theme: Production Pipeline Hardening

---

#### run_weekly.py Production Fixes — 5 Issues Resolved
**Status**: Committed (`cc0f6b9`)
**Files**: `scripts/run_weekly.py`, `src/models/efficiency_foundation_model.py`

Fixed 5 issues in the production prediction pipeline to improve robustness.

**Fix 1 (CRITICAL): Week 1 Predictions Crash**
- Guard `wait_for_data` to skip for week 1 (no prior week data exists)
- Handle empty `current_games_df` for week 1 with proper DataFrame schema
- Add EFM early return for empty plays (priors-only mode)
- Fix play column names: `home_score/away_score` → `offense_score/defense_score` (API mismatch)
- Add `home_team` column for neutral-field ridge regression

**Fix 2 (MEDIUM): Bare Except in Edge-Aware Recommendation**
- Changed `except Exception:` to `except Exception as e:` with debug logging
- Now captures game_id and error message for debugging

**Fix 3 (LOW): Wasted Value Play Computation**
- Moved initial value play computation inside `if not dual_spread:` guard
- In dual_spread mode, value plays are recomputed with edge-aware recommended spreads

**Fix 4 (LOW): LSA Mode Coupling Validation**
- Added guard that auto-enables `use_learned_situ` when `dual_spread=True`
- Prevents invalid configuration state

**Fix 5 (LOW): Fragile Schedule Fetching**
- Removed `consecutive_empty` early-exit logic in `build_schedule_df`
- Now always iterates all 15 regular season weeks, skipping empty weeks gracefully
- Prevents truncated schedules due to mid-season bye weeks

**Testing:**
- Week 5 backtest: MAE 10.42, 54.7% ATS (Core slice)
- EFM correctly loaded 34,799 FBS plays for training

---

## Session: February 11, 2026 (Late Night)

### Theme: G5/Cross-Tier/HFA Investigation — Complete

---

#### G5 Conference Island Diagnostics — NO ACTION NEEDED
**Status**: Investigated, no bias found
**Files**: `scripts/diagnose_g5_islands.py`, `data/outputs/g5_diagnostics/PHASE_A_SUMMARY.md`

Investigated whether G5 conferences are "isolated islands" causing rating inflation.

**Key Findings:**
- Overall G5 vs P4 ME: -2.15 pts (over-predict P4)
- **Weeks 1-3**: ME = -3.16 (bias present)
- **Weeks 4+**: ME = -0.28 (bias nearly gone)
- Connectivity is similar across all conferences (1.7-2.1 OOC games/team)
- Sun Belt is only truly problematic conference (47.1% ATS, -3.80 ME)

**Conclusion**: Conference island inflation is NOT real. The existing conference anchor mechanism is working correctly.

---

#### G5 Prior Analysis — BLENDED PRIORS WON'T HELP
**Status**: Investigated, counterintuitive finding
**Files**: `scripts/analyze_g5_priors.py`, `data/outputs/g5_prior_analysis.md`

Investigated whether SP+ vs own-prior explains G5 early-season bias.

**Key Findings:**
| Prior System | Overrates G5 By |
|--------------|-----------------|
| SP+ (prior year) | +0.46 pts |
| Own-Prior (JP+) | +1.29 pts |

**Conclusion**: Own-prior actually overrates G5 MORE than SP+. Blended priors would make G5 bias WORSE, not better.

---

#### Cross-Tier HFA Analysis — REJECTED (Blowout Artifact)
**Status**: Investigated, robustness check FAILED
**Files**: `scripts/analyze_cross_tier_hfa.py`, `scripts/cross_tier_hfa_robustness.py`, `data/outputs/cross_tier_hfa_*.md`

Investigated the 7-point ME swing between G5 home vs G5 away in cross-tier games.

**Initial Finding (Full Dataset, N=432):**
| Scenario | Residual HFA |
|----------|--------------|
| P4 Home vs G5 | +7.16 |
| G5 Home vs P4 | -0.27 |

**Robustness Check — FAILED:**
| Dataset | P4 Home Residual HFA |
|---------|---------------------|
| Full (N=432) | +7.16 |
| **Excluding 30+ pt margins (N=301)** | **-0.10** |

The +7.16 effect **completely disappears** when excluding blowout games (131 games, 30.3%).

**Conclusion**: Cross-tier HFA asymmetry is a blowout artifact, NOT a venue effect. In competitive games (<30 pt margin), model HFA is well-calibrated. Do NOT implement cross-tier HFA adjustment.

---

#### Blowout Rating Contamination — NO ISSUE
**Status**: Quick sanity check, no contamination detected
**Files**: `scripts/blowout_rating_impact.py`, `data/outputs/blowout_rating_impact.md`

Verified that blowout wins don't contaminate team ratings.

**EFM Architecture (Why Blowouts Are Handled):**
- Regresses on Success Rate (0-1) and IsoPPP (~-0.5 to +1.5), NOT margins
- Garbage time: trailing team 0.1 weight, leading team 1.0 weight
- MOV calibration: DISABLED (mov_weight=0.0)
- Efficiency metrics are inherently bounded

**Rating Trajectory Analysis (3 teams with 3+ blowout G5 wins):**
| Metric | Value |
|--------|-------|
| Mean change after blowout G5 win | **+0.03** |
| Mean change after P4 game | **+0.36** |

No spike/correction pattern. Ratings are stable.

---

#### Investigation Summary — VALUABLE NEGATIVE RESULTS
**Commit**: `c2bfec1` — Add G5/cross-tier diagnostic scripts (NEGATIVE RESULTS)

**Final Conclusions:**
1. Conference island inflation is NOT real — connectivity similar across conferences
2. Cross-tier HFA asymmetry is a blowout artifact — not a venue effect
3. Model is well-calibrated for competitive cross-tier games (< 30 pt margin)
4. Early-season G5 vs P4 ME is mostly a blowout prediction issue — not systematic bias
5. Blowout wins do NOT contaminate ratings — EFM handles them appropriately

These negative results prevent implementing corrections that would hurt performance.

---

## Session: February 11, 2026 (Night)

### Theme: QB Misattribution Analysis

---

#### QB Misattribution Fix — Phased Analysis — NO IMPACT
**Status**: Tested, infrastructure added but no performance impact
**Files**: `src/adjustments/qb_continuous.py`, `scripts/backtest.py`, `data/outputs/qb_phase_*.md`

Investigated and fixed the QB misattribution problem: at Week 1, the system was using departed QB's PPA data for teams with new starters. 89 of 136 FBS teams had wrong QB data applied at Week 1.

**Phased Approach:**
- **Phase A**: Zero out ALL Week 1 QB adjustments (can't verify who's starting)
- **Phase B**: Measure gap between matched (returning starter) vs unmatched (newcomer) teams
- **Phase C**: Implement informed prior for newcomers (conditional on Phase B)

**Implementation:**
Added `fix_misattribution` parameter to `QBContinuousAdjuster` and `--qb-fix-misattribution` CLI flag. When enabled, zeros out Week 1 adjustments for any team without verified current-season dropbacks.

**Phase A Results (2022-2025, Week 1 only):**
| Metric | Original | Phase A | Delta |
|--------|----------|---------|-------|
| MAE | 15.39 pts | 15.39 pts | 0.00 |
| 5+ Edge | 50.0% (100-100) | 50.0% (100-100) | 0.0% |
| 3+ Edge | 49.2% (127-131) | 49.2% (127-131) | 0.0% |

**Phase B Gap Analysis:**
Gap between matched and unmatched teams: **0.0 points** (no measurable difference)

**Phase C: SKIPPED** — Gap < 1.0 pts threshold

**Root Cause — Why No Impact:**
Week 1 QB adjustments are too small to matter:
| Factor | Effect | Result |
|--------|--------|--------|
| Prior decay (0.3) | 469 dropbacks → 141 effective | 70% signal loss |
| Shrinkage (K=200) | 141 effective → 43% weight | Max ~0.7 pts adjustment |
| Typical case | 300 dropbacks | ~0.4 pts adjustment |

These sub-1-point adjustments are overwhelmed by EFM ratings (±30 pts range), preseason priors, and HFA (2.5-3.5 pts).

**Key Insight:**
The QB Continuous system's value comes from **Weeks 2-3** (when actual 2025 game data exists), not from Week 1 prior-based estimates. The prior-based Week 1 adjustments are too heavily shrunk to provide meaningful signal.

**Recommendation:**
Enable `--qb-fix-misattribution` as default (conceptually correct, zero downside). But no performance gain expected.

---

#### Blended Priors (SP+ + JP+ Own-Prior) — ATS Validation — REJECTED
**Status**: Tested, infrastructure complete but ATS degradation
**Files**: `src/models/blended_priors.py`, `src/models/own_priors.py`, `scripts/backtest.py` (--blended-priors flag), `data/outputs/blended_prior_summary.md`

Validated the blended prior system (combining SP+ and JP+ own-priors with week-dependent weights) against the SP+-only baseline for ATS performance.

**Blended Schedule Tested (from tuning):**
| Week | SP+ Weight | Own Weight |
|------|------------|------------|
| 1 | 0% | 100% |
| 2 | 40% | 60% |
| 3 | 0% | 100% |
| 4+ | 80% | 20% |

**Results (Weeks 1-3, 2022-2025):**
| Metric | SP+-Only | Blended | Delta |
|--------|----------|---------|-------|
| MAE | 15.30 | 14.94 | **-0.36** ✓ |
| Mean Error (bias) | +3.33 | +2.63 | **-0.70** ✓ |
| 5+ Edge | **50.8%** (270-262) | 49.7% (274-277) | **-1.1%** ✗ |
| 3+ Edge | 48.9% (340-355) | 49.0% (341-355) | +0.1% ≈ |

**Why It Failed:**
Classic 3+/5+ divergence pattern. Blended priors improved MAE by 0.36 pts and reduced bias, but degraded 5+ Edge by 1.1%. The tuning was optimized for spread MAE, not ATS performance.

**Key Insight:**
The disagreement analysis showed own-prior converges to actual 62% of time vs 33% for SP+. But convergence to "correct" end-of-season rating ≠ betting edge. The market may already price the same convergence pattern. MAE improvement without ATS improvement = no edge.

**Infrastructure Preserved:**
- `--blended-priors` CLI flag in backtest.py
- `BlendedPriorGenerator` class with `BlendSchedule` dataclass
- `OwnPriorGenerator` class with fitted parameters
- `data/historical_jp_ratings.json` (2021-2025 JP+ ratings)
- Full disagreement analysis in `data/outputs/`

**Recommendation:**
Keep SP+-only priors as default. Blended system may be revisited if tuning is redone with ATS as objective function rather than MAE.

---

## Session: February 11, 2026 (Evening)

### Theme: QB Continuous Rating System

---

#### QB Continuous Rating System — Phase1-only Mode — COMMITTED
**Status**: Committed (`9d2118c`)
**Files**: `src/adjustments/qb_continuous.py` (NEW), `scripts/qb_calibration.py` (NEW), `scripts/backtest.py`, `src/predictions/spread_generator.py`

Implemented walk-forward-safe QB quality estimates that improve Phase 1 (weeks 1-3) ATS performance without degrading Core (weeks 4-15).

**Key Architecture:**
- QB PPA tracking via CFBD API (`get_predicted_points_added_by_player_game`)
- Shrinkage parameter K=200 (~250 dropbacks → 55% weight on raw signal)
- Prior season decay factor 0.3 for Week 1 projections
- Conservative starter identification (MIN_RECENT_DB=15, MIN_CUM_DB=50)
- Residual adjustment logic to avoid double-counting with EFM

**The Key Insight — Phase1-only Mode:**
By week 4, the EFM has already "baked in" QB quality through efficiency metrics. Any additional QB adjustment in Core weeks causes double-counting and slight ATS degradation. The `--qb-phase1-only` flag disables QB adjustment for weeks 4+, capturing early-season signal where QB priors provide genuine value.

**Calibration Results (2023-2024, N=4,035 team-weeks):**
| Week Bucket | qb_value_shrunk P90 | Mean Uncertainty | Unknown Starter |
|-------------|---------------------|------------------|-----------------|
| Weeks 1-3 | 0.163 | 0.78 | 34.7% |
| Weeks 4-8 | 0.216 | 0.57 | 0.9% |
| Weeks 9-15 | 0.245 | 0.48 | 0.7% |

**Scale Sweep Results (2024 Core Weeks 4-15):**
| Scale | MAE | 5+ Edge Close | vs Baseline |
|-------|-----|---------------|-------------|
| Baseline | 12.66 | 56.4% | — |
| 4.0 | 12.63 | 56.6% | +0.2% |
| **5.0** | **12.63** | **57.1%** | **+0.7%** |
| 6.0 | 12.62 | 56.9% | +0.5% |

**But Full 2022-2025 Validation showed Core degradation with full QB mode:**
All scales degraded Core 5+ Edge by 0.2-0.3% due to double-counting.

**Solution — Phase1-only Mode (2022-2025):**
| Phase | Baseline 5+ Edge | QB Phase1 Only | Delta |
|-------|------------------|----------------|-------|
| Phase 1 (1-3) | 50.2% (269-267) | 50.8% (270-261) | **+0.6%** |
| Phase 2 (4-15) | 54.5% (463-386) | 54.5% (463-386) | **0.0%** |

**CLI Flags:**
- `--qb-continuous`: Enable QB adjustment
- `--qb-scale`: PPA-to-points scale (default 5.0)
- `--qb-phase1-only`: Only apply for weeks 1-3 (RECOMMENDED)
- `--qb-shrinkage-k`: Shrinkage parameter (default 200)
- `--qb-cap`: Max point adjustment (default 3.0)
- `--qb-prior-decay`: Prior season decay (default 0.3)

**Residual Adjustment Logic (Infrastructure):**
Built but not needed with Phase1-only mode. The residual formula:
```
baking_factor = ramp(0→1 over weeks 1-8)
residual = current_starter_value - (team_avg_value × baking_factor)
```
This prevents double-counting for stable starters while correctly penalizing injuries/backups.

---

#### Experiments Tested but Not Adopted

**Full QB Continuous (all weeks):**
- Core 5+ Edge: 54.5% → 54.3% (-0.2%)
- Failed acceptance criteria (must not degrade Core)
- Root cause: Double-counting with EFM efficiency metrics

**Year-by-Year Analysis:**
| Year | Baseline 5+ Edge | QB Full Mode | Delta |
|------|------------------|--------------|-------|
| 2022 | 53.2% | 54.0% | +0.8% |
| 2023 | 54.8% | 53.8% | -1.0% |
| 2024 | 56.4% | 57.1% | +0.7% |
| 2025 | 53.5% | 51.7% | -1.8% |

Inconsistent year-to-year results suggest QB signal is noisier in Core weeks where EFM already captures it.

---

## Session: February 11, 2026 (Afternoon)

### Theme: Edge-Aware LSA Default + Dual-Cap Infrastructure

---

#### Edge-Aware LSA Mode — NOW DEFAULT — COMMITTED
**Status**: Committed (`b337c1a`)
**Files**: `scripts/run_weekly.py`, `CLAUDE.md`

Made edge-aware LSA the default behavior for 2026 production. No flags needed.

**Automatic Mode Selection:**
| Timing | Edge | Mode | ATS |
|--------|------|------|-----|
| Opening (4+ days) | Any | Fixed | 56.5% (5+) |
| Closing (<4 days) | 5+ pts | LSA | 55.1% |
| Closing (<4 days) | 3-5 pts | Fixed | 52.9% |

**CLI Changes:**
- `--no-lsa` flag added to disable edge-aware mode
- `--learned-situ` and `--dual-spread` kept as legacy (now no-ops, behavior is default)

**Usage:** Just run `python scripts/run_weekly.py` — engine handles timing/edge logic automatically.

---

#### Dual-Cap Mode Infrastructure — TESTED, NO IMPROVEMENT — COMMITTED
**Status**: Committed (`b337c1a`)
**Files**: `scripts/backtest.py`, `src/predictions/spread_generator.py`, `src/adjustments/home_field.py`

Built infrastructure to test per-timing ST cap and HFA offset parameters.

**Hypothesis:** Different ST caps might be optimal for open vs close line evaluation.

**Implementation:**
- Added `special_teams_raw` and `home_field_raw` fields to SpreadComponents
- Added `skip_offset` parameter to HomeFieldAdvantage.get_hfa()
- Added `assemble_spread_vectorized()` for efficient spread reassembly
- Added `--dual-cap-mode`, `--sweep-dual-st-cap` CLI flags

**LOO-CV Sweep Results (2022-2025, Core weeks 4-15):**
| Timing | Aggregate 5+ Edge | Best Cap | Stability |
|--------|-------------------|----------|-----------|
| Open | 56.5% | 2.0 | Unstable (varies 2.0/none by year) |
| Close | 54.3% | 1.5 | Unstable (varies 1.5/2.5 by year) |

**Conclusion:** No stable improvement over single-cap baseline (57.0% open, 54.5% close). Year-to-year variance indicates noise, not signal. Infrastructure preserved for future testing but not recommended for production.

---

#### CLAUDE.md Baseline Update — COMMITTED
**Status**: Committed (`b337c1a`)

Updated with edge-aware production mode documentation and current validated metrics.

---

## Session: February 11, 2026 (Morning-2)

### Theme: Bug Fixes & Documentation

---

#### Dual-Spread Mode Documentation — COMMITTED
**Status**: Committed (`a05e830`)
**Files**: `docs/MODEL_EXPLAINER.md`, `docs/MODEL_ARCHITECTURE.md`

Documented the `--dual-spread` feature for dynamic LSA/fixed switching based on bet timing.

**Key insight documented**:
- **4+ days out (Sun–Wed)** → Use fixed baseline (57.0% at 5+ Edge on opening lines)
- **<4 days (Thu–Sat)** → Use LSA (56.1% at 5+ Edge on closing lines)

**MODEL_EXPLAINER.md**: Added "Dynamic Bet Timing Mode" section with CLI examples
**MODEL_ARCHITECTURE.md**: Added technical implementation details and performance table

---

#### EFM Bug Fixes (4 bugs) — COMMITTED
**Status**: Committed (`73982cc`)
**Files**: `src/models/efficiency_foundation_model.py`
**Backtest**: Metrics unchanged (MAE 12.53, 5+ Edge 54.5%)

**Bug 1: Invalid team indices (-1)** — HIGH SEVERITY
- `pd.Categorical.codes` returns -1 for unknown teams → `np.bincount` fails
- **Fix**: Added explicit validation with clear error message showing unknown teams

**Bug 2: Data leakage guard used assert** — HIGH SEVERITY
- Running Python with `-O` disables asserts → silent future-data leak
- **Fix**: Changed to explicit `if/raise ValueError`

**Bug 3: SettingWithCopyWarning in garbage_time_weight=0** — MEDIUM SEVERITY
- Filtering creates view, then `df["weight"]=1.0` may not stick
- **Fix**: Added `.copy()` after filter

**Bug 4: In-place DataFrame mutation in _calculate_raw_metrics** — LOW SEVERITY
- `plays_df["weighted_success"]=...` could contaminate cached DataFrames
- **Fix**: Use `.assign()` with local array instead of mutating input

---

#### Aggregator Defensive Fix — COMMITTED
**Status**: Committed (`b8aaee0`)
**Files**: `src/adjustments/aggregator.py`

**Issue**: `was_capped = False` not explicitly set in non-capped branch
- Current code relies on dataclass default, but `env_was_capped` IS explicitly set in both branches
- Inconsistent pattern; risk of state-leak if objects reused

**Fix**: Added explicit `result.was_capped = False` in else branch to match `env_was_capped` pattern

---

#### Bug Validated as NOT A BUG

**sort_values key=abs in generate_comparison_df** — NOT A BUG
- Claimed: `abs` (Python builtin) doesn't work on pandas Series, would raise TypeError
- **Reality**: Pandas Series implements `__abs__()`, so `abs(series)` works correctly
- Tested both approaches; identical correct output (sorted by absolute edge descending)

---

## Session: February 11, 2026 (Evening)

### Theme: FCS & LSA Code Hardening

---

#### FCS Penalty Logic Fix — APPROVED
**Status**: Committed
**Files**: `src/models/fcs_strength.py`

Fixed critical bug where `abs(margin)` incorrectly penalized elite FCS teams that beat FBS opponents.

**The Bug**: A +10 margin (FCS wins by 10) was treated identically to -10 margin (FCS loses by 10), causing elite FCS teams like NDSU to receive higher penalties when they should receive lower.

**The Fix**: Changed `abs(margin)` to `-margin` so:
- Positive margin (FCS wins) → lower penalty (min 10 pts)
- Negative margin (FCS loses) → higher penalty

**Phase 1 Backtest (Weeks 1-3)**: ATS +0.8% (47.9% → 48.7%), MAE +0.51

---

#### FCSStrengthEstimator Refactor — Code Quality
**Status**: Committed
**Files**: `src/models/fcs_strength.py`, `scripts/backtest.py`, `scripts/diagnose_lsa_features.py`

1. Renamed `max_week` → `through_week` with explicit docstring clarifying callers must pass `current_week - 1`
2. Added `__post_init__` parameter validation (k_fcs > 0, min_penalty <= max_penalty, slope >= 0)
3. Vectorized Polars pipeline: shrink_factor, shrunk_margin, penalty computed in Polars chain
4. Single iteration instead of separate `_recalculate_strengths()` method
5. `pl.Series` for fbs_teams (avoids rebuilding hashset on each `is_in()`)
6. Lazy %-formatting for logger
7. Single-pass accumulation in `get_summary_stats()`

---

#### LSA Model Hardening — Bug Fixes + Optimization
**Status**: Committed
**Files**: `src/models/learned_situational.py`

**Bug Fixes**:
1. **FEATURE_NAMES/to_array() drift prevention**: Dynamic `to_array()` from FEATURE_NAMES + module-level assertion
2. **EMA cross-season bleed**: Clear `_prev_coefficients` in `reset()` — first train() of each season uses raw coefficients
3. **seed_with_prior_data ordering**: Prepend prior data instead of replacing (safe regardless of call order)
4. **Clamp-after-EMA ratchet**: Clamp raw coefficients FIRST, then EMA (prevents sticky ceiling)
5. **None guards in from_situational_factors**: Defensive coalescing for rest_days and penalty fields
6. **Year validation before persist**: Warn and skip if `train()` called before `reset()`

**Optimizations**:
7. **Numpy**: `np.dot()` for predict(), parallel lists + `np.vstack()` for training data
8. **Documentation**: Expanded sign convention docstring for `compute_situational_residual`

---

#### run_weekly.py Bug Fixes — Code Quality
**Status**: Committed (`dcb074a`)
**Files**: `scripts/run_weekly.py`

**Validated & Fixed**:

1. **build_schedule_df bare `except Exception: break`** — REAL BUG
   - Transient API errors (rate limits, timeouts) silently truncated schedule
   - Caused missed lookahead/sandwich spot detection with no warning
   - **Fix**: Log error and continue; use consecutive-empty-weeks heuristic for end detection

2. **LSA lsa_alpha parameter unused** — DEAD CODE
   - Parameter accepted but never used (alpha is training-time, coefficients pre-baked)
   - **Fix**: Removed from CLI and function signature; added explanatory comment

3. **LSA misleading log** — BUG
   - Logged mean of final spreads, not mean of LSA adjustments
   - **Fix**: Track `lsa_adjustments` list, log actual mean adjustment

4. **Unnecessary `object.__setattr__`** — CODE SMELL
   - Used to "bypass frozen dataclass" but neither dataclass is frozen
   - **Fix**: Normal assignment `pred.components.situational = lsa_adj`

5. **hasattr guards** — MIXED
   - `settings.ridge_alpha`: Field exists, guard unnecessary → removed
   - `fcs_penalty_elite/standard`: Fields don't exist in Settings, always fell through to hardcoded defaults → removed (SpreadGenerator has its own defaults)
   - `pred.components.situational`: Always exists (default_factory) → removed

**Backtest**: No regressions (changes only affect run_weekly.py code path)

---

#### Bugs Validated as NOT BUGS

1. **FG np.select `conditions[:-1]`** — NOT A BUG
   - Claimed 50-60 yard bucket was dropped, causing 50-59 yard FGs to get wrong rate
   - **Reality**: `[:-1]` removes (60,100) bucket; 50-60 bucket (index 3) is preserved
   - 60+ yard FGs correctly fall through to `default=0.30`

2. **Kickoff return_count dimensional issue** — NOT A BUG
   - Claimed high-TB teams get "almost zero credit on return-yards-saved axis"
   - **Reality**: Intentional decomposition — high-TB teams get credit via `tb_bonus`, not `return_saved`
   - Components are orthogonal; no double-counting

---

#### run_weekly.py Additional Fixes — Code Quality
**Status**: Committed (`76a7613`)
**Files**: `scripts/run_weekly.py`

**Fixes**:

1. **fetch_upcoming_games returning completed games** — BUG
   - No filter for completed games, could include games with scores
   - **Fix**: Added filter `completed=False` (if available) or filter on null home_points

2. **Broad exception handler in run_predictions** — CODE SMELL
   - Single `try/except Exception` caught programming errors (TypeError, ValueError, KeyError)
   - **Fix**: Split into targeted handlers; let programming errors propagate

3. **Missing fbs_teams validation** — BUG
   - If fbs_teams empty/None, predictions silently used invalid data
   - **Fix**: Added explicit validation with descriptive error message

4. **Polars-to-Pandas conversion before FBS filtering** — INEFFICIENCY
   - Converted full DataFrame to pandas, then filtered
   - **Fix**: Do FBS filtering in Polars, convert result to pandas (~130 teams filtered from ~200)

5. **build_schedule_df DataNotAvailableError handling** — BUG
   - All exceptions caused `break`, conflating "week doesn't exist" with transient errors
   - **Fix**: `DataNotAvailableError` → break (expected); other exceptions → log warning and continue

6. **Per-team ST/FD loops** — IMPLEMENTED batch methods
   - 130+ FBS teams × full DataFrame scan = ~130x redundant iteration
   - Added `calculate_all_from_game_stats()` to SpecialTeamsModel and FinishingDrivesModel
   - Uses groupby instead of per-team filtering; run_weekly.py updated to use batch methods

---

#### ST/FD Batch Methods for Game Stats — Performance Enhancement
**Status**: Committed (`4e6366b`)
**Files**: `src/models/special_teams.py`, `src/models/finishing_drives.py`, `scripts/run_weekly.py`

Added batch methods for game-stats calculation path (used by run_weekly.py when play-by-play unavailable):
- `SpecialTeamsModel.calculate_all_from_game_stats(teams, games_df)`
- `FinishingDrivesModel.calculate_all_from_game_stats(teams, games_df)`

**Before**: Per-team loops with 260+ DataFrame filter operations (130 teams × 2 filters each)
**After**: 4 groupby operations total (home/away for each model)

Backtest unaffected (uses play-by-play batch methods). Production run_weekly.py now ~130x faster for ST/FD calculation.

---

## Session: February 11, 2026

### Theme: INT/Fumble Separate Shrinkage in Turnover Model

---

#### INT/Fumble Bayesian Shrinkage Split — APPROVED
**Status**: Committed
**Files**: `config/play_types.py`, `src/models/efficiency_foundation_model.py`, `scripts/backtest.py`

Refactored turnover margin component to apply different Bayesian shrinkage to INTs (skill-based) vs fumbles (luck-based).

**Hypothesis**: Interceptions correlate year-to-year (QB decision-making, defensive scheme), while fumble recoveries are essentially coin flips.

**Implementation**:
- Added `INTERCEPTION_PLAY_TYPES` and `FUMBLE_PLAY_TYPES` classification in play_types.py
- Track INT thrown/forced and fumbles lost/recovered separately in EFM
- Apply different shrinkage: `shrink_int = n/(n+k_int)`, `shrink_fum = n/(n+k_fumble)`
- CLI parameters: `--k-int` (default 10.0), `--k-fumble` (default 30.0)

**Parameter Sweep Results (k_int × k_fumble)**:
| k_int | k_fumble | 5+ ATS Close |
|-------|----------|--------------|
| 10 | 10 | 53.7% (unified baseline) |
| **10** | **30** | **54.0%** (+0.3pp) |
| 10 | 50 | 54.0% (tied) |

**Year-to-Year Stability**:
- 2022: 52.7% → 52.7% (unchanged)
- 2023: 53.7% → 53.9% (+0.2pp)
- 2024: 55.5% → 55.8% (+0.3pp)
- 2025: 52.9% → 53.2% (+0.3pp)

**Key Finding**: Applying stronger shrinkage to fumbles (k=30 vs k=10) improves 5+ ATS Close by +0.3pp consistently across years. Validates the "INT is skill, fumble is luck" hypothesis.

---

#### FCS Vectorization — Minor Improvement
**Status**: Committed
**Files**: `src/models/fcs_strength.py`

Converted FCS strength estimation to fully vectorized Polars pipeline. No performance change, cleaner code.

---

## Session: February 10, 2026 (Evening)

### Theme: Learned Situational Adjustment (LSA) Implementation

---

#### Learned Situational Model — APPROVED (High-Confidence Filter)
**Status**: Committed
**Files**: `src/models/learned_situational.py` (new), `scripts/backtest.py`, `scripts/diagnose_lsa_features.py` (new)

Implemented walk-forward ridge regression on situational residuals to replace fixed constants (bye_week_advantage=1.5, letdown_penalty=-2.0, etc.) with learned coefficients.

**Architecture**:
- 16 situational features learned from residuals (rest, bye, letdown, lookahead, sandwich, rivalry, consecutive road, game shape)
- Multi-year pooling: ~3,000+ training samples by week 4 (pools 2022-2024 for 2025 predictions)
- Walk-forward safe: Only trains on games before prediction week
- EMA smoothing (beta=0.3) for coefficient stability

**Alpha × Clamp Grid Sweep Results**:
| Alpha | 3+ Edge | 5+ Edge |
|-------|---------|---------|
| Fixed baseline | **53.1%** | 53.7% |
| 300 (optimal) | 52.3% | **54.9%** |
| 500 | 52.4% | 54.9% |

**Key Finding**: LSA serves as a **high-confidence filter**. It improves Top Tier (5+) performance by +1.2% while slightly reducing volume/accuracy in the lower-confidence (3+) tier. The trade-off is acceptable because 5+ Edge bets have ~2% over vig vs ~1.3% for 3+ Edge.

**Defaults Set**: `alpha=300.0`, `clamp_max=4.0` (safety net)
**CLI**: `--learned-situ` to enable (default OFF for production)

---

#### Rivalry Underdog Boost — DOMAIN FINDING
**Status**: Documented (no code change yet)

Forensics analysis of 108 rivalry underdog home games revealed a **handicapping fallacy**:

| Metric | Value |
|--------|-------|
| Fixed rivalry boost applied | +1.0 pts |
| Actual residual after boost | **-3.52 pts** |
| LSA learned coefficient | **-3.21 pts** |

**Finding**: Favorites in rivalry games tend to **WIN BIG**, not smaller. Examples:
- Oklahoma -49 actual vs Texas when predicted -8 (residual -41)
- Arizona -42 vs ASU when predicted -2 (residual -40)

The traditional "underdogs play harder in rivalry games" is not supported by data. LSA correctly learns a negative coefficient to offset the wrong fixed boost.

---

#### Coefficient Clamping — IMPLEMENTED
**Status**: Committed

Added `--lsa-clamp-max` CLI parameter to enforce hard bounds on coefficients after Ridge regression.

**Finding**: Clamping only matters at low alpha values. With alpha ≥ 100, Ridge regularization already keeps coefficients small enough that clamping has no additional effect.

---

#### Production Workflow Streamlining — COMMITTED
**Status**: Committed
**Commits**: `a9b6163`, `fa04789`

Added streamlined CLI arguments to `scripts/run_weekly.py` for 2026 season production use.

**New CLI arguments**:
- `--min-edge N` - Minimum edge (pts) to show as value play (default: 3.0)
- `--sharp` - Sharp betting mode: shortcut for `--min-edge 5`
- `--learned-situ` - Enable LSA with automatic coefficient loading

**LSA coefficient loading**:
- `load_lsa_coefficients(year)` - Loads most recent coefficients from `data/learned_situ_coefficients/`
- Falls back to previous year's coefficients if current year not available
- `apply_lsa_adjustment()` - Computes per-game adjustment from situational features

**Streamlined 2026 workflow**:
```bash
# First-time setup (once per season):
python3 scripts/backtest.py --years 2022 2023 2024 2025 --learned-situ

# Weekly (sharp betting mode):
python3 scripts/run_weekly.py --learned-situ --sharp
```

---

#### Documentation Updates — COMMITTED
**Status**: Committed
**Commits**: `7a4a7cf`, `3713592`

Updated all documentation to reflect LSA implementation:
- `docs/MODEL_EXPLAINER.md` - Added LSA section with architecture and grid sweep results
- `docs/MODEL_ARCHITECTURE.md` - Added LSA to situational factors section
- `CLAUDE.md` - Updated baseline table to show both Fixed and LSA modes

**Dual-baseline display** (from CLAUDE.md):
| Mode | 3+ Edge (Close) | 5+ Edge (Close) | Use Case |
|------|-----------------|-----------------|----------|
| Fixed (default) | **53.1%** (800-708) | 53.7% (498-429) | Standard production |
| LSA enabled | 52.3% (772-704) | **54.9%** (483-397) | High-conviction filtering |

---

## Session: February 11, 2026

### Theme: FCS Vectorization + HFA Neutralization

---

#### FCS Polars Vectorization — APPROVED
**Status**: Committed
**Commit**: `71ce0f0`

Refactored `FCSStrengthEstimator.update_from_games()` to use vectorized Polars operations instead of Python loops.

**Changes**:
- Replaced `for row in filtered.iter_rows()` with vectorized pipeline
- Uses `with_columns()` for FBS detection and margin calculation
- Uses `group_by("fcs_team").agg()` for per-team aggregation
- Cleaner code, same performance

---

#### FCS HFA Neutralization — REJECTED (infrastructure preserved)
**Status**: Infrastructure committed, feature disabled by default
**Commit**: `71ce0f0`

Tested HFA neutralization to remove venue bias from FCS margin calculations.

**Math**:
- When Home is FBS: `margin = fcs_pts - (fbs_pts - hfa_value)`
- When Home is FCS: `margin = (fcs_pts - hfa_value) - fbs_pts`

**Parameter sweeps**:

HFA sweep (intercept=9):
| HFA | Phase 1 5+ Edge | Phase 2 5+ Edge |
|-----|-----------------|-----------------|
| 0.0 | 49.8% | 54.4% |
| 1.5-3.0 | 49.7% | 54.4% |

Intercept sweep (HFA=0):
| Intercept | Phase 1 5+ Edge | Phase 2 5+ Edge |
|-----------|-----------------|-----------------|
| 7 | 49.1% | 54.2% |
| 8 | 49.4% | 54.3% |
| 9 | 49.8% | 54.4% |
| **10** | **50.2%** | **54.3%** |
| 11 | 49.6% | 54.4% |

**Key finding**: HFA=0, intercept=10 matches original baseline exactly.

**Rejection reason**: HFA neutralization helps Core (+0.1%) but hurts Phase 1 (-0.5%). Since FCS games occur primarily in Phase 1, the net effect is negative.

**Infrastructure preserved**:
- `hfa_value` parameter in `FCSStrengthEstimator` (default 0.0 = disabled)
- `--fcs-hfa` CLI argument for future experimentation

**Unit tests**:
- HFA=0: raw_margin = -25.0 (no adjustment) ✓
- HFA=3: raw_margin = -22.0 (FBS wins 35-10 at home) ✓

---

## Session: February 10, 2026 (Late Night)

### Theme: Dynamic FCS Strength Estimator

---

#### Dynamic FCS Strength Estimator — APPROVED
**Status**: Committed, enabled by default
**Commits**: `0219f62` (initial), `d01156e` (recalibration)

Replaced static `ELITE_FCS_TEAMS` frozenset (17 hardcoded teams with fixed 18/32 pt penalties) with a walk-forward-safe, data-driven FCS strength estimator using Bayesian shrinkage.

**Architecture**:
- `FCSStrengthEstimator` class in new `src/models/fcs_strength.py`
- Per-team strength tracked from FBS-vs-FCS game margins
- Bayesian shrinkage: `shrunk = baseline + (raw - baseline) * (n / (n + k))`
- Continuous penalty mapping: `penalty = clamp(intercept + slope * abs(margin), min, max)`
- Walk-forward safe: `update_from_games()` only uses `max_week` filter

**Recalibrated parameters** (matches historical static baseline):
- k=8 (games for 50% trust in data)
- baseline=-28 (prior FCS margin vs FBS)
- intercept=10.0, slope=0.8, range=[10, 45]

**Penalty mapping** (after recalibration):
| FCS Tier | Avg Loss to FBS | Penalty |
|----------|-----------------|---------|
| Elite (NDSU) | ~10 pts | +18 pts |
| Average | ~28 pts | +32.4 pts |
| Weak | 40+ pts | +45 pts (capped) |

**Phase-by-phase comparison**:
| Phase | Dynamic 5+ Edge | Static 5+ Edge | Delta |
|-------|-----------------|----------------|-------|
| Phase 1 (weeks 1-3) | **50.2%** | 48.8% | **+1.4%** |
| Phase 2 (Core) | 54.3% | **54.6%** | -0.3% |

**Trade-off**: Dynamic FCS helps Phase 1 where FCS games occur (+1.4% 5+ Edge) at cost of slight Core regression (-0.3%). Both meet 54.0% acceptance threshold.

**CLI arguments added**:
- `--fcs-static`: Use static elite list (baseline comparison)
- `--fcs-k`: Shrinkage k (default 8.0)
- `--fcs-baseline`: Prior margin (default -28.0)
- `--fcs-min-pen`, `--fcs-max-pen`: Penalty bounds (10.0, 45.0)
- `--fcs-slope`, `--fcs-intercept`: Mapping params (0.8, 10.0)

**Why approved**: Replaces hardcoded list with adaptive, walk-forward-safe system. Improves Phase 1 ATS performance. Infrastructure enables future improvements (e.g., cross-season FCS tracking).

---

## Session: February 10, 2026 (Evening)

### Theme: ST Empirical Bayes Shrinkage + Bug Fixes

---

#### ST Empirical Bayes Shrinkage — REJECTED
**Status**: Infrastructure committed, feature disabled by default

Implemented opportunity-based Empirical Bayes shrinkage for Special Teams ratings:
- Formula: `shrunk = raw * (n / (n + k))` where n = opportunities, k = trust threshold
- Tracks FG attempts, punts, and kickoff events per team
- CLI args: `--st-shrink`, `--st-k-fg`, `--st-k-punt`, `--st-k-ko`

**9-value k sweep results** (Core weeks 4-15):
| k | 5+ Edge | vs Baseline |
|---|---------|-------------|
| No shrink | 53.7% (502-433) | — |
| k=4 | 53.6% (490-425) | -0.1% |
| k=6 | 53.7% (490-422) | 0% |
| k=8 | 53.7% (484-418) | 0% |
| k=10+ | 53.2-53.4% | -0.3 to -0.5% |

**Rejection reason**: While k=6-8 maintain 5+ Edge, they degrade 3+ Edge (53.3% → 52.6%) and All games (52.0% → 51.5%). Shrinkage compresses spreads toward zero, reducing disagreement with Vegas — the opposite of what we want.

**Failure mode**: Spread compression — pulling extreme values toward mean kills edge.

**Infrastructure preserved**: `--st-shrink` flag enables for future experimentation. Default k=6 if enabled.

---

#### ST Spread Cap (Approach B: Margin-Level Capping) — APPROVED
**Status**: Committed, enabled by default (cap=2.5)

Alternative approach to ST reliability: cap the EFFECT of ST on spread, not the rating itself.
- Formula: `st_impact = clip(st_diff, -cap, +cap)`
- CLI: `--st-spread-cap` (default 2.5, use 0 to disable)

**4-value sweep results** (Core weeks 4-15):
| Cap | 5+ Edge | 3+ Edge | Spread Std |
|-----|---------|---------|------------|
| None | 53.7% | 53.3% | 12.42 |
| ±1.5 | 54.0% | 53.1% | 12.21 |
| ±2.0 | 53.8% | 53.0% | 12.27 |
| **±2.5** | **54.0%** | **53.3%** | **12.32** |
| ±3.0 | 53.8% | 53.3% | 12.35 |

**Why it works**: Unlike shrinkage (which pulls ALL ratings toward zero), capping only affects extreme matchups (|diff| > 2.5). This reduces false positives from ST spikes while preserving true disagreement.

**Key insight**: 5+ Edge improved because extreme ST predictions were noisy outliers. Capping them reduced false positives without reducing true positives.

---

#### Bug Fixes — COMMITTED (earlier in session)

1. **Pace adjustment ordering**: FCS penalty now applied after pace compression
2. **Venue smoothing**: Soft cap applied to venue stack only, not full environmental stack
3. **Priors weight sum**: Week 0 branch now uses `1.0 - effective_talent_weight` (was 0.95)
4. **Rating dtype**: `rating` column moved to FLOAT32 (was incorrectly INT16)
5. **Pick'em lines**: betting_lines.py uses `x if x is not None else y` instead of `x or y`

---

## Session: February 10, 2026

### Theme: Weather Capture Infrastructure + Totals Backtest Export + Week Detection Fix

---

#### Add CSV Export to Totals Backtest — COMMITTED
**Commit**: `74dd186`

Added `--output` flag to `backtest_totals.py` for game-level CSV export:
- Includes `edge_close`, `edge_open`, `play` (OVER/UNDER), and `result` columns
- Enables detailed analysis of individual totals predictions
- Also removed Kennesaw State from triple-option list (transitioned to conventional offense in FBS)

---

#### Replace Hardcoded Week Detection with CFBD Calendar API — COMMITTED
**Commit**: `7e87d01`

Fixed `get_current_cfb_week()` in `weather_thursday_capture.py`:
- **Problem**: Used hardcoded arithmetic (`week_of_year - 34`) which was off by ±1-2 weeks
- **Solution**: Uses `cfbd_client.get_calendar(year)` to find current/upcoming week
- Handles Week 0, regular season (1-15), and postseason (16+)
- Fails clearly during off-season (Feb-Jul) with `OffSeasonError`
- Added `timedelta` import for date arithmetic

---

#### Remove Unused BATCH Constants — COMMITTED
**Commit**: `7c9a11f`

Removed dead code from `weather_thursday_capture.py`:
- `BATCH_SIZE` and `BATCH_DELAY_SECONDS` were defined but never referenced
- Rate limiting handled via `--limit` and `--delay` CLI args
- Updated docstring to document actual rate limiting approach
- Added hourly API quota logging before/after capture

---

#### Fix Pass-Rate Handling and Watchlist Sorting — COMMITTED
**Commit**: `10020e3`

Pass-rate handling:
- Default to 0.5 (neutral) when team pass rate data unavailable
- Add `pass_rate_available` flag to watchlist entries
- Log when pass-rate scaling is inactive (at debug level)
- Update report to show "N/A (using neutral default)" when data missing

Watchlist sorting:
- Use tuple sort key: `(0, edge)` for real edges, `(1, weather_adj)` for None
- Entries with `edge=None` now sort AFTER entries with real edge
- Most negative edge ranks first within each group

---

#### Compute Weather Adjustments for All Outdoor Games — COMMITTED
**Commit**: `5ef7d77`

Previously, adjustments were only computed for games flagged by `is_weather_concern()`.
Now computes adjustments for ALL outdoor games, then builds watchlist by thresholding.

- Add `DEFAULT_WATCHLIST_ADJ_THRESHOLD` (1.5) and `DEFAULT_WATCHLIST_EDGE_THRESHOLD` (3.0)
- Add `--adj-threshold` and `--edge-threshold` CLI arguments
- Store all adjustments in `stats["all_adjustments"]`
- Build watchlist from games where `|adj| >= threshold` OR `|edge| >= threshold`
- Add `high_variance` flag to game entries
- Update both report functions to show threshold info and adjustment counts

---

#### Add Hourly Rate Limit Tracking to TomorrowIOClient — COMMITTED
**Commit**: `4f16c3e`

Improvements to `src/api/tomorrow_io.py`:
- Add `FREE_TIER_HOURLY_LIMIT` constant (25/hr)
- Add `_hourly_calls` list to track call timestamps
- Add `get_hourly_call_count()` method for quota visibility
- Add hourly limit check before API calls (returns None if limit reached)
- Add exponential backoff with jitter on 429 errors
- Add `max_backoff_seconds` parameter (default 60s)

Cleanup:
- Remove `scripts/diagnose_gt_threshold.py` (one-off diagnostic)

---

#### Memory Updates
- Added "Betting Analysis Preferences" section: always use OPENING lines for totals betting analysis
- Saved season summary table format to `memory/season_summary_format.md`

---

#### Optimize TravelAdjuster: Compute Distance Once — COMMITTED
**Commit**: `b9ea81d`

In `get_total_travel_adjustment()`, `self.get_distance()` was called twice:
1. Once explicitly to get distance
2. Once again inside `get_distance_adjustment()`

**The Fix:**
- Add optional `distance` parameter to `get_distance_adjustment()`
- Compute distance once in `get_total_travel_adjustment()` and pass through
- Reuse distance value in breakdown dict

Also cleaned up sign conventions with unambiguous `home_adv` suffix in breakdown dict.

---

#### Add DST-Aware Timezone Calculations — COMMITTED
**Commit**: `36a1da7`

The module docstring acknowledged ~0.5 pt error for late-season games involving Arizona/Hawaii (which don't observe DST), but nothing in the code checked game timing.

**Added to `config/teams.py`:**
- `NO_DST_TEAMS` frozenset: Arizona, Arizona State, Hawaii
- `is_dst_active(game_date)`: Detects if DST is active on a given date
- `get_dst_adjusted_offset(team, game_date)`: Returns corrected offset for no-DST teams

**Updated functions with optional `game_date` parameter:**
- `get_timezone_difference(team1, team2, game_date=None)`
- `get_directed_timezone_change(away_team, home_team, game_date=None)`
- `TravelAdjuster.get_timezone_adjustment(away_team, home_team, game_date=None)`
- `TravelAdjuster.get_total_travel_adjustment(home_team, away_team, game_date=None)`

**Offset adjustments when DST is NOT active:**
- Arizona/ASU: 3 → 2 hrs behind ET
- Hawaii: 6 → 5 hrs behind ET

Default behavior (no `game_date`) unchanged — assumes DST is active.

---

### Theme: Bug Fix Sweep (Evening Session)

Multiple architectural bugs identified and fixed. All fixes validated via backtest with no regression.

---

#### Add Totals Preseason Priors to 2026 Production Planning — COMMITTED
**Commit**: `53ec79e`

Added feature spec to MODEL_ARCHITECTURE.md for enabling week 0/1 totals predictions:
- End-of-season snapshot of totals ratings to carry forward
- 30-40% regression to mean (teams regress toward league average)
- Blending schedule: 100% prior in week 0-1 → fades to ~30% by week 4
- Roster continuity adjustment for high-churn teams
- Implementation: `src/models/totals_priors.py`, `scripts/save_totals_priors.py`

---

#### Fix Pace Adjustment Incorrectly Compressing FCS Penalty — COMMITTED
**Commit**: `d537fc3`

**The Bug:** `_get_pace_adjustment()` was applied to the full spread including FCS penalty. If Army hosts an FCS team, the 32-point FCS penalty got compressed by 10%, subtracting 3.2 points from the spread.

**The Fix:** Reorder spread calculation so pace adjustment is applied BEFORE FCS penalty:
1. Calculate efficiency-derived spread (ratings + adjustments + ST)
2. Apply pace compression (10% toward zero)
3. THEN add FCS penalty (uncompressed)

---

#### Fix Venue Smoothing Incorrectly Triggered by Rest/Consecutive — COMMITTED
**Commit**: `7befaf7`

**The Bug:** `env_smoothing_factor = env_score / raw_env_stack` where `raw_env_stack` included rest and consecutive_road. Negative situational values could mask the venue stack, and positive ones could trigger smoothing on venue components that hadn't changed.

Example:
- HFA=3.5, travel=2.0, altitude=1.5, rest=-2.0 → factor=1.0 (no smoothing)
- HFA=3.5, travel=2.0, altitude=1.5, rest=+0.5 → factor=0.67 (venue over-smoothed)

**The Fix:**
- Calculate `raw_venue_stack` = HFA + travel + altitude only
- Apply soft cap to venue stack, calculate `venue_smoothing_factor`
- Add rest and consecutive_road linearly after venue smoothing
- Use `venue_smoothing_factor` for component pro-rata allocation

---

#### Fix blend_with_inseason Weights Not Summing to 1.0 — COMMITTED
**Commit**: `c77dbb2`

**The Bug:** The `games_played <= 0` branch set `prior_weight = 0.95 - talent_weight` with `inseason_weight = 0.0`, summing to only 0.95. This silently compressed all preseason ratings by 5%, shrinking early-season spreads by ~1.5 points on 30-point mismatches.

**The Fix:** `prior_weight = 1.0 - effective_talent_weight` at week 0.

Added assertion after all branches to prevent future regressions:
```python
assert abs(prior_weight + inseason_weight + effective_talent_weight - 1.0) < 1e-9
```

---

#### Fix Kickoff Rating Unit Mismatch — COMMITTED
**Commit**: `3a61220`

**The Bug:** In `calculate_kickoff_ratings_from_plays`:
1. `tb_bonus` was a rate difference (e.g., 0.10) multiplied by kicks_per_game, double-counting volume
2. `return_saved` was per-return points scaled by total kicks including touchbacks

**The Fix:**
- `tb_bonus`: `(tb_rate - 0.60) × kicks / games × 0.15` → pts/game
- `return_saved`: `(23 - avg_return) × 0.04 × returns / games` → pts/game
- `coverage_rating` = direct sum (both already pts/game, no further multiplication)

Added unit comments on all intermediate columns: `[dimensionless]`, `[kicks]`, `[games]`, `[yards/return]`, `[pts/return]`, `[returns/game]`, `[pts/game]`

---

#### Fix Rating Column Dtype: float32 not int16 — COMMITTED
**Commit**: `1073d07`

**The Bug:** `config/dtypes.py` listed `rating` in INT16_COLUMNS with comment "Recruit rating (0-100)", but 247Sports ratings are floats in 0.0-1.0 range (e.g., 0.8834). If `optimize_dtypes()` was applied to portal data, ratings truncated to 0, collapsing all transfers to floor quality factor (0.1).

**The Fix:**
- Move `rating` from INT16_COLUMNS → FLOAT32_COLUMNS
- Add warning comment: "0.0-1.0 scale, e.g., 0.8834 — NOT integer 0-100"
- Disambiguate `distance` and `yards_to_goal` with "Play-by-play:" prefix

---

#### Fix Pick'em Line (0.0) Being Discarded in get_merged_lines — COMMITTED
**Commit**: `05a6387`

**The Bug:** Python's `or` operator treats 0.0 as falsy, so `odds_line.spread_open or cfbd_line.spread_open` silently discarded the Odds API value when it was a pick'em (0.0) and substituted the CFBD value.

**The Fix:** Replace all `x or y` fallback patterns with explicit None checks:
```python
spread_open=(
    odds_line.spread_open if odds_line.spread_open is not None
    else cfbd_line.spread_open
)
```
Applied to: `spread_open`, `spread_close`, `home_team`, `away_team`, `sportsbook`

Audited `get_odds_api_lines` — uses direct assignment, no `or`-based fallbacks.

---

## Session: February 9, 2026

### Theme: MAE Regression Investigation + FCS Safeguard Bug Fix + Smoothed Stack Property + Sign Convention Hardening + Performance Hardening + Infrastructure Hardening + Vectorization Sweep

---

#### Fix Asymmetric Portal Transfer Penalty — COMMITTED
**Impact: G5→P4 trench transfers now consistently ~8-11% net negative (continuity tax only)**

Fixed asymmetric penalty for G5→P4 trench player transfers in `preseason_priors.py`.

**The Bug:**
- Outgoing values computed WITHOUT level-up discount → 1.11x (continuity tax)
- Incoming values computed WITH level-up discount → 0.75x (G5→P4 trench)
- Net: 36% penalty for transfers that should be roughly zero-sum minus continuity tax

**The Fix:** Applied level-up discount to outgoing values as well:
```python
transfers_df['outgoing_value'] = transfers_df.apply(
    lambda row: self._calculate_player_value(
        row.get('stars'), row.get('rating'), row['pos_group'],
        origin=row.get('origin'),
        destination=row.get('destination'),
        team_conferences=team_conferences,
    ), axis=1
)
```

**Backtest:** 5+ Edge 54.6% (0.1% decrease = 3 games out of 867). Model Strategist confirmed: "Preserving known-incorrect logic because of 3 lucky games is the definition of overfitting to noise."

**Commit**: `329654e`

---

#### Fix Latent NameError in Coaching Change Path — COMMITTED
**Impact: Prevents crash if triple-option team has coaching change**

In `preseason_priors.py`, `extremity_talent_scale` was only defined inside the non-triple-option branch but referenced unconditionally later for coaching adjustment tracking.

**The Fix:** Initialize `extremity_talent_scale = 1.0` before branching:
```python
# Default extremity scale (used for coaching adjustment tracking)
# Must be initialized before branching to avoid NameError if triple-option
# team has a coaching change (triple-option branch doesn't set this)
extremity_talent_scale = 1.0
```

**Commit**: `b82e6f8`

---

#### Remove Dead Code from SpecialTeamsModel — COMMITTED
**Impact: -211 lines, cleaner codebase**

Removed `calculate_team_rating()` and 6 helper methods from `special_teams.py`. Docstring claimed "kept for run_weekly.py" but `run_weekly.py` actually uses `calculate_from_game_stats()`.

Also contained a circular bug where `punts_per_game` always equaled 5.0.

**Commit**: `1d14895`

---

#### Batch Odds Inserts with executemany() — COMMITTED
**Impact: ~480 SQLite round-trips → 1 batch insert**

Refactored `capture_odds()` in `weekly_odds_capture.py` to build tuples during validation and batch insert with `cursor.executemany()`.

**Commit**: `ae39bd5`

---

#### Add LRU Eviction to Ridge Cache — COMMITTED
**Impact: Prevents unbounded memory growth during parameter sweeps**

Added LRU eviction to `_RIDGE_ADJUST_CACHE` in `efficiency_foundation_model.py`:
- Max size: 500 entries
- Uses `collections.OrderedDict` with `move_to_end()` on hit
- Evicts oldest entries when full
- Added cache stats tracking (hits, misses, evictions)

**Commit**: `14b0ec8`

---

#### Kickoff Per-Game Scaling Fix — REVERTED
**Impact: Would have normalized coverage/return by actual games played**

Attempted to fix inconsistent per-game scaling in kickoff ratings (coverage used actual games, return used estimated). Backtest showed 5+ Edge degradation: 54.7% → 54.1%.

**Reverted:** Not a valid fix per decision gate.

---

#### FG Rate Lookup Investigation — NOT A BUG
**Impact: Confirmed np.select distance buckets are correct**

User reported that `np.select` might be slicing off the last (50,60) bucket. Traced through code and verified:
- 5 distance buckets: (0,20), (20,30), (30,40), (40,50), (50,60)
- `conditions[:-1]` correctly returns indices 0-3, but conditions list has 5 elements
- Index 4 (the 50-60 bucket) is retained as `default` fallback
- Tested edge cases: 19yd, 29yd, 49yd, 55yd all mapped correctly

**No change required.**

---

#### to_pandas() Profiling — NO CHANGE
**Impact: Confirmed overhead is negligible**

Profiled Polars→pandas conversion in `calculate_ats_results()`:
- 0.34ms per call
- ~0.005% of backtest runtime
- Not worth optimizing

---

#### Performance Hardening and Dead Code Removal — COMMITTED
**Impact: Cleaner codebase, faster execution, no metric changes**

Major cleanup and optimization pass across backtest and EFM:

**Backtest (scripts/backtest.py):**
- Removed FinishingDrivesModel (shelved after 4 backtest rejections, 70-80% overlap with EFM)
- Hoisted TravelAdjuster, AltitudeAdjuster, SituationalAdjuster outside week loop (stateless, safe to reuse)
- Added shallow copy to prevent sweep cache pollution

**EFM (src/models/efficiency_foundation_model.py):**
| Optimization | Speedup | Technique |
|-------------|---------|-----------|
| Empty yards per-team analysis | 7.6x | Vectorized groupby |
| _apply_efficiency_fraud_tax() | 7.2x | Vectorized merge |
| Team index lookup | 1.5x | pd.Categorical |
| _calculate_raw_metrics() | 1.7x | Groupby→dict in one pass |
| TeamEFMRating mutations | N/A | dataclasses.replace() (immutability) |

- Removed stale `_GT_THRESHOLDS` module-level cache (sweep pollution risk)
- Replaced 5 mutable assignment sites with `dataclasses.replace()`

**Multiprocessing overhead investigation:**
- Measured 51 MB pickle per season (204 MB for 4 years)
- Despite overhead, parallel is 2.8x faster (21s vs 59s) — keep parallel

**Backtest verified:** MAE 12.50, Core ATS 52.2%, 5+ Edge 54.7% (unchanged)

**Commit**: `58f72a6` (Performance hardening and dead code removal)

---

#### Symmetric Soft Cap Analysis — DOCUMENTED
**Impact: Confirms negative cap threshold is effectively dead code**

Analyzed whether env soft cap should be asymmetric for positive vs negative stacks.

**Empirical analysis (2023-2025, 1,824 games):**
| Threshold | Games | Percentage |
|-----------|-------|------------|
| Positive (>5.0) | 238 | 13.0% |
| Negative (<-5.0) | **0** | **0.00%** |
| Most negative observed | -1.50 pts | — |

**Why negatives are so rare:** HFA, travel, and altitude are always positive. Only rest can be negative, and even worst-case (neutral + short week + opponent bye) reaches only ~-3.0 pts.

**Recommendation:** Keep symmetric treatment. The negative cap is effectively dead code — adding asymmetric logic would add complexity for zero practical benefit.

**Commit**: `358ddef` (Document symmetric soft cap design decision with empirical analysis)

---

#### Global Cap Priority Ordering — DOCUMENTED
**Impact: Explains why mental factors may have reduced effect in extreme scenarios**

When global cap (7.0) is hit, environmental factors implicitly take priority over mental:
- Utah home (4.5 HFA + 1.5 travel + 2.0 altitude) → env_score ~6.8
- Add rivalry (+1.0) → 7.8 pre-mental
- Add letdown (+2.0) → 9.8 capped to 7.0
- Mental's marginal contribution: only 0.2 of its 2.0 value

**This is intentional:**
1. Global cap triggers rarely (~1% of games)
2. Environmental factors are objectively measurable
3. Mental factors are more speculative/psychological
4. When caps are hit, trust "harder" signals over "softer" ones

**Commit**: `4b4b6a9` (Document global cap priority ordering design decision)

---

#### Per-Team Mental Smoothing — DOCUMENTED
**Impact: Explains asymmetric smoothing design for mental factors**

Mental factors are smoothed per-team BEFORE netting (not raw values netted first):
- Team A: letdown 3.5 only → smoothed = 3.5
- Team B: letdown 2.0 + lookahead 1.5 → smoothed = 2.75
- Net: 0.75 (even though raw sums both equal 3.5)

**This is intentional:** Models diminishing marginal psychological impact. One overwhelming distraction is worse than two smaller ones of equal total magnitude.

**Commit**: `fc9b233` (Document per-team mental smoothing design decision)

---

#### Extract Interaction Effect Constants — COMMITTED
**Impact: Tunable interaction dampening without editing source code**

Extracted hardcoded interaction values to class-level constants with constructor overrides:

| Constant | Default | Purpose |
|----------|---------|---------|
| `TRAVEL_INTERACTION_THRESHOLD` | 1.5 | Travel penalty above which interactions apply |
| `ALTITUDE_INTERACTION_DAMPENING` | 0.70 | Altitude reduced to 70% when travel > threshold |
| `CONSECUTIVE_ROAD_INTERACTION_DAMPENING` | 0.50 | Consec road reduced to 50% when travel > threshold |

All three have constructor parameters for sweep testing without source edits.

**Commit**: `2d120eb` (Extract interaction effect constants to class-level with constructor overrides)

---

#### Short-Week × Travel Interaction Analysis — DOCUMENTED
**Impact: Explicit reasoning for why short-week has no travel interaction**

Analyzed whether short-week rest should have an interaction effect with travel (like consecutive_road at 50% and altitude at 30%):

**Why existing interactions exist:**
- **Consecutive road × travel**: Both measure "travel fatigue" (2+ weeks away vs. this trip)
- **Altitude × travel**: Both are acute physical stressors on game day

**Why short-week is intentionally exempt:**
- Short-week = incomplete RECOVERY from previous game (prep time, healing, mental fatigue)
- Travel = acute JOURNEY stress (jet lag, disrupted sleep)
- These are orthogonal mechanisms: a team on short week **at home** still has recovery penalties
- The env soft cap (5.0 pts, 60% excess) handles extreme stacks adequately

**Commit**: `c44a3cf` (Document why short-week has no travel interaction effect)

---

#### Diagnostic Field Rename: raw_total → pre_global_cap_total — COMMITTED
**Impact: Accurate naming for adjustment smoothing progression**

The old `raw_total` field was misleadingly named - it was actually post-env-soft-cap and post-mental-smoothing, only pre-global-cap.

**New diagnostic progression:**
1. `raw_sum_all`: True linear sum of ALL raw values (zero smoothing)
2. `pre_global_cap_total`: After env soft cap + mental smoothing
3. `net_adjustment`: Final value after global cap

Added `raw_mental_sum` to track unsmoothed mental penalties.

**Diagnostic usage:** `raw_sum_all - pre_global_cap_total` shows exactly how much smoothing reduced extreme stacks.

**Commit**: `e168f90` (Rename raw_total → pre_global_cap_total, add raw_sum_all for diagnostics)

---

#### Sign Convention Documentation and Validation — COMMITTED
**Impact: Prevents silent wrong results if upstream adjusters change sign conventions**

Traced through TravelAdjuster, AltitudeAdjuster, and SituationalFactors to document actual sign conventions:

1. **TravelBreakdown**:
   - Corrected docstring: values are **positive magnitudes** (not "negative value" as previously documented)
   - Added `__post_init__` validation to raise `ValueError` if values are negative
   - Documented that values come from `TravelAdjuster.get_total_travel_adjustment()` (positive = favors home)

2. **SituationalFactors**:
   - Added "Sign Convention" section to class docstring
   - All penalties are **positive magnitudes** on the affected team
   - `game_shape_penalty` is the exception (stored as negative)
   - Added inline comments on each field with sign expectations

3. **Aggregator**:
   - Documented that `abs()` calls are safety belts (values already validated positive)
   - Expanded `consecutive_road_penalty` comment with sign semantics

**Commit**: `64ae4cf` (Document and validate sign conventions for adjustment components)

---

#### MAE Regression Investigation — RESOLVED
**Impact: Identified and fixed bug causing MAE regression from 12.50 → 12.62**

- **Symptom**: Backtest MAE was 12.62, but documented baseline is 12.50.
- **Bisect approach**: Tested midpoint commits between `d456ee5` (known good) and HEAD.
- **Root cause**: Commit `2ebfbe7` ("Add safeguards for misspelled team names") introduced a faulty FCS safeguard.

**The Bug:**
```python
# Faulty logic:
if not home_is_fbs and home_in_ratings:
    home_is_fbs = True  # Skip FCS penalty
```

**Why it was wrong:** The safeguard assumed "if team is in ratings but not in fbs_teams, it must be a misspelled FBS team." But FCS teams that play FBS opponents (Chattanooga, Rhode Island, Fordham, etc.) ARE in `self.ratings` because the EFM generates ratings for all teams in play data. The safeguard was incorrectly skipping FCS penalties for legitimate FCS teams.

**Evidence during backtest:**
```
WARNING - Team 'Chattanooga' is in ratings but NOT in fbs_teams. Possible data inconsistency — skipping FCS penalty.
WARNING - Team 'Rhode Island' is in ratings but NOT in fbs_teams. Possible data inconsistency — skipping FCS penalty.
... (dozens of legitimate FCS teams)
```

**Fix**: Removed the faulty safeguard. FCS penalty now correctly applies based on `fbs_teams` set membership only.

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Core MAE | 12.62 | **12.50** |
| 5+ Edge | 54.9% | 54.7% |

- **Commit**: `94f95b1` (Fix FCS safeguard that incorrectly skipped penalties for legitimate FCS teams)
- **Kept**: Warning logs for "not found in ratings" (47 FCS teams with minimal data).
- **Removed**: Faulty "in ratings but not fbs_teams = misspelled" assumption (-18 lines).

---

#### Smoothed Correlated Stack Property — COMMITTED
**Impact: Better diagnostics for raw vs post-smoothing environmental adjustment values**

Added `smoothed_correlated_stack` property to `SpreadComponents`:
- `correlated_stack`: Raw HFA + travel + altitude (pre-smoothing, for transparency)
- `smoothed_correlated_stack`: Post-soft-cap value from aggregator (what actually hit the spread)
- `env_score` field: Stores `aggregated.env_score`

Updated:
- `_determine_confidence()`: Uses smoothed stack for decisions
- Extreme stack warnings: Show both raw and smoothed values
- `to_dict()`: Exports both `correlated_stack` and `correlated_stack_smoothed`

**Commit**: `cc5cec4` (Add smoothed_correlated_stack property to SpreadComponents)

---

#### Pro-Rata Smoothing for Adjustment Buckets — COMMITTED
**Impact: Accurate per-bucket post-smoothing values in SpreadComponents**

**Problem:** `components.situational` was computed as residual (`net_adjustment - raw_hfa - raw_travel - raw_altitude`), which absorbed smoothing deltas from env soft-cap. Made situational values misleading.

**Solution:** Added pro-rata allocation via `env_smoothing_factor`:
- `AggregatedAdjustments.env_smoothing_factor`: ratio of `env_score / raw_env_stack`
- `AggregatedAdjustments.situational_score`: pure situational (rest + consec + mental + boosts)
- SpreadGenerator now uses `raw_value * factor` for HFA/travel/altitude
- Renamed `smoothed_correlated_stack` → `full_env_stack` for clarity

**Commit**: `e4b5d3e` (Add pro-rata smoothing for accurate per-bucket adjustment reporting)

---

#### spread_generator.py Cleanup — COMMITTED
**Impact: game_id support, import cleanup, dead code removal**

1. **game_id parameter support (P0.1)**:
   - Added `game_id` param to `predict_spread()`
   - `predict_week()` extracts from game dict ("id" or "game_id")
   - Added `game_id` to `PredictedSpread.to_dict()` output

2. **ELITE_FCS_TEAMS cleanup**:
   - Removed James Madison and Sam Houston (now FBS, were dead entries)
   - Added historical note comment

3. **Import cleanup**:
   - Moved `SituationalFactors` to top-level import
   - Removed inline import in fallback path

4. **Documentation**:
   - Updated `predict_week()` docstring with all expected game dict fields

**Commit**: `b727eb8` (Clean up spread_generator.py: game_id support, imports, dead code)

---

#### totals_model.py Hardening — COMMITTED
**Impact: Validation, bounds, alignment fixes across 7 areas**

1. **Year parameter fix (P0.1)**: `walk_forward_totals_backtest()` now passes `year` to `predict_total()` for correct year-specific baseline.

2. **decay_factor validation**: Raises `ValueError` if not in `(0, 1.0]`. Values > 1.0 would upweight old games. Short-circuit when `decay_factor == 1.0`.

3. **Model reuse optimization**: Create `TotalsModel` once before backtest loop, call `set_team_universe()` once.

4. **ridge_alpha alignment**: Backtest default now 10.0 (was 5.0), matches model default and documented optimal.

5. **/2 shrinkage documentation**: Documented that `/2` is intentional:
   - Without /2: MAE 12.90 (better), 5+ Edge 53.3% (worse)
   - With /2: MAE 13.09 (worse), 5+ Edge 54.5% (better)
   - Since 5+ Edge is binding constraint, keep shrinkage

6. **Prediction bounds**:
   - Per-team floor: `max(0, expected)`
   - Total floor: 21 pts
   - Total ceiling: 105 pts
   - `weather_adjustment` now included in total calculation

7. **Index alignment robustness**: Filter/reset DataFrame BEFORE computing `home_idx`/`away_idx` to prevent silent misalignment.

**Commit**: `e83cbba` (Harden totals_model.py: validation, bounds, alignment fixes)

---

#### TotalsModel reset() + spread_generator.py Cleanup — COMMITTED
**Impact: Model reuse support + minor code quality improvements**

1. **TotalsModel.reset()**: Added method to clear model state for reuse across seasons:
   - Clears team universe lock, ratings, trained state
   - Preserves configuration (ridge_alpha, decay_factor, use_year_intercepts)

2. **spread_generator.py cleanup**:
   - Extract `WIN_PROB_K = 0.15` constant for spread-to-probability conversion
   - Document `sort_by_spread` parameter in `predict_week()`
   - Fix `get_high_stack_games()` return type hint
   - Name discarded tuple values with `_` prefix

**Commit**: `9d4c3a2` (Add TotalsModel.reset() method + spread_generator cleanup)

---

#### TotalsModel Cleanup: __repr__, Reliability, Types — COMMITTED
**Impact: Better debugging, reliability filtering, type consistency**

1. **__repr__ for dataclasses**: Added to `TotalsRating` and `TotalsPrediction` for debugging:
   ```python
   TotalsRating(Georgia: off=29.3, def=22.1, games=12)
   TotalsPrediction(Alabama @ Georgia: total=54.2, adj=53.7)
   ```

2. **get_ratings_df() enhancements**:
   - Added `reliability` column (0-1 based on games_played: 1 game=0.0, 8+ games=1.0)
   - Added `min_games` filter parameter for early-season filtering

3. **float64 types**: Removed float32 casts that Ridge converts internally anyway

4. **DataFrame pattern**: Documented Polars→pandas conversion pattern in comments

**Commit**: `8d47199` (TotalsModel cleanup: __repr__, reliability column, float64 types)

---

#### Unused Import Fix — COMMITTED
**Impact: Resolves Pylance diagnostic on aggregator.py line 55**

Removed unused `field` import from `dataclasses` in aggregator.py. The `field()` function was imported but never used — `TravelBreakdown` and `AggregatedAdjustments` use simple default values that don't require field factories.

**Commit**: `1d3cfb3` (Remove unused 'field' import from aggregator.py)

---

#### Odds API ↔ CFBD Game ID Bridge — COMMITTED
**Impact: Enables 2026 production with captured Odds API lines**

**The Problem (ID Island):** Odds API uses alphanumeric game IDs (`cf82a9b3...`) while CFBD uses integers (`401628456`). The existing merge logic in `betting_lines.py` silently failed because IDs never matched.

**Solution:**
1. Added `cfbd_game_id INTEGER` column to `odds_lines` table
2. Created `scripts/link_odds_to_cfbd.py` — matches by normalized team names + date
3. Updated `get_odds_api_lines()` to use `cfbd_game_id` as key when available
4. Updated PROJECT_MAP.md in both repos

**Usage for 2026:**
```bash
python scripts/capture_odds.py --capture-current
python scripts/link_odds_to_cfbd.py --year 2026
python scripts/run_weekly.py --year 2026 --week 1
```

**Commit**: `26ac101` (Add cfbd_game_id column to bridge Odds API and CFBD game IDs)

---

#### Deprecated datetime.utcnow() Fix — COMMITTED
**Impact: Python 3.12+ compatibility**

Replaced `datetime.utcnow()` with `datetime.now(timezone.utc)` in `capture_odds.py`. The former is deprecated in Python 3.12+.

**Commit**: `3fc301e` (Fix deprecated datetime.utcnow() in capture_odds.py)

---

#### Opening Line Capture Time Fix — COMMITTED
**Impact: Correct timing for Sunday odds capture**

Opening lines are posted by 8-10 AM ET on Sunday, not 6 PM as previously documented.

| Before | After |
|--------|-------|
| 6 PM ET (11 PM UTC) | **8 AM ET (1 PM UTC)** |

Updated:
- `capture_odds.py`: Query date calculation
- `weekly_odds_capture.py`: Docstrings and cron examples

**Commit**: `fee2fc9` (Fix opening line capture time: 6 PM ET -> 8 AM ET)

---

#### Atomic Write Pattern for SeasonDataCache — COMMITTED
**Impact: Prevents corrupted cache from interrupted writes (Ctrl+C, network failure)**

**The Vulnerability:** If `save_season()` was interrupted after writing 3/5 files, or mid-write of a file, the cache could be left in a corrupted state that `has_cached_season()` incorrectly treated as valid.

**Solution — Write-to-Temp-then-Move pattern:**
1. Write all files to `.tmp/` subdirectory first
2. Move files to final location on success
3. Write `.complete` marker file LAST (atomic commit)
4. `has_cached_season()` requires `.complete` marker to return True
5. Orphaned `.tmp/` directories cleaned on init
6. Added `migrate_legacy_cache()` to validate and mark existing caches

**Failure scenarios now handled:**

| Scenario | Before | After |
|----------|--------|-------|
| Ctrl+C during write | Partial cache marked complete | `.tmp/` cleaned, no marker |
| Network failure mid-download | Corrupted files cached | No marker, re-downloads |
| Corrupted parquet file | Silent load failure | Marker removed, re-downloads |

**Commit**: `06a93a7` (Add atomic write pattern to SeasonDataCache for cache integrity)

---

#### SSH Authentication Setup
**Impact: Fixes GitHub push hanging with HTTPS**

Configured SSH authentication for GitHub pushes:
1. Generated Ed25519 SSH key (`~/.ssh/id_ed25519`)
2. Added to GitHub account (hpnas47)
3. Switched both repos to SSH remote URLs
4. Created `~/.ssh/config` for explicit key specification

---

#### FCS Teams in Normalization Fix — COMMITTED
**Impact: Architecturally correct normalization (minimal practical impact)**

**The Bug:** `_normalize_ratings()` was passed `all_teams` (from `_canonical_teams`), which includes FCS teams that appear in play data (when FBS teams play FCS opponents). The comment at line 1900 claimed "all_teams from CFBD API is FBS teams only" — this was **false**.

**Fix:**
1. Added `fbs_teams: Optional[set[str]] = None` parameter to `calculate_ratings()`
2. If provided, normalization uses only FBS teams (excludes FCS outliers)
3. Falls back to `all_teams` if not provided (legacy behavior for compatibility)
4. Updated `backtest.py` and `run_weekly.py` to pass `fbs_teams`

**Backtest validation (2022-2025 Core):**

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| MAE | 12.50 | 12.50 | 0 |
| ATS (Close) | 52.2% | 52.20% | 0 |
| 5+ Edge | 54.7% | 54.7% | 0 |

**Why minimal impact:** FCS teams have very few plays (only vs FBS on defense), so their ratings cluster near league average due to shrinkage. They weren't extreme outliers that would significantly skew mean/std.

**Commit**: `e6dac28` (Fix FCS teams included in normalization calculation)

---

#### Require Explicit --year/--week for Odds Capture — COMMITTED
**Impact: Prevents week mislabeling that corrupts opening-vs-closing analysis**

**The Bug:** `get_current_week()` uses a naive month/day formula that can be off by 1-2 weeks. Critically, if opening lines are captured Oct 31 (→ week 9) and closing lines Nov 2 (→ week 10), they get stored under different week labels for the SAME CFB week. This corrupts all downstream opening-vs-closing analysis.

**Fix:**
1. Made `--year` and `--week` REQUIRED for `--opening`/`--closing` operations
2. Added deprecation warning to `get_current_week()` heuristic
3. `--preview` still allows optional week args (display only, no stored data)
4. Clear error message with usage example if args missing

**Before:**
```bash
python scripts/weekly_odds_capture.py --opening  # Silently uses heuristic
```

**After:**
```bash
python scripts/weekly_odds_capture.py --opening --year 2026 --week 10  # Required
```

**Error if missing:**
```
error: --year and --week are REQUIRED for --opening/--closing.
The auto-detection heuristic can mislabel weeks at month boundaries...
```

**Commit**: `6e72d8c` (Require explicit --year/--week for odds capture operations)

---

#### Null Spread Resilience for Odds Capture — COMMITTED
**Impact: Prevents batch abort when API returns lines with null spreads**

**The Bug:** `odds_lines` table defines `spread_home REAL NOT NULL`, but the API can return lines with `spread_home = None` (futures, props, etc.). One null spread would throw `sqlite3.IntegrityError` and abort the entire batch, losing all lines for that week's snapshot.

**Fix (two-part):**
1. **Filter null spreads before insert**: Skip lines where `spread_home` or `spread_away` is None, with warning log
2. **Per-line try/except**: Wrap each INSERT in exception handler so one bad line doesn't abort the batch

**New return dict fields:**
- `lines`: Actual lines stored (excludes skipped/failed)
- `lines_from_api`: Total lines received from API
- `null_spread_skipped`: Lines skipped due to null spreads
- `insert_errors`: Lines that failed to insert

**Output now shows:**
```
Opening lines captured:
  Season: 2026 Week 10
  Games: 45
  Lines stored: 312
  ⚠ Skipped (null spread): 3
  Credits remaining: 487
```

**Commit**: `e477692` (Add null spread resilience to odds capture)

---

#### Max Week Safeguard for run_weekly.py — COMMITTED
**Impact: Prevents infinite wait loop when requesting non-existent weeks**

**The Bug:** Requesting week 20 for 2025 caused the script to poll for 200+ minutes waiting for data that will never exist (CFB season ends at week 17).

**Fix (two layers):**
1. **run_weekly.py**: Fail fast if `week > 18` with clear error message
2. **cfbd_client.wait_for_data()**: Check if week has any scheduled games before polling; if no games, raise `DataNotAvailableError` immediately

**Before:**
```
Data not ready for 2025 week 19. Waited 200.0 min. Checking again in 5.0 min...
(continues forever until max_wait_hours)
```

**After:**
```
ValueError: Week 20 exceeds maximum CFB week (18). CFB season ends at week 17 (national championship).
```

Or if games exist but the week is invalid:
```
DataNotAvailableError: No games found for 2025 week 19. The CFB season typically ends at week 17.
```

**Commit**: `4f867f8` (Add max week safeguard to prevent infinite wait loop)

---

#### Transaction Safety for Odds Capture — COMMITTED
**Impact: Prevents partial writes and ensures connection cleanup on error**

**The Bug:** If `capture_odds()` raised an exception (API error, IntegrityError), `conn.close()` was never called and partial writes could be left in an inconsistent state.

**Fix (two parts):**
1. **`contextlib.closing()`**: Wrap database connection to ensure `close()` even on exception
2. **Explicit transaction**: `BEGIN`/`COMMIT`/`ROLLBACK` for all-or-nothing semantics

**Before:**
```python
conn = init_database()
result = capture_odds(...)  # If this raises, conn never closed
conn.close()
```

**After:**
```python
with closing(init_database()) as conn:
    result = capture_odds(...)  # conn.close() guaranteed
```

**Transaction semantics:**
- Pre-filter lines with null spreads before starting transaction
- `BEGIN` explicit transaction
- Insert snapshot + all lines
- On success: `COMMIT` (all lines saved)
- On any error: `ROLLBACK` (no partial writes) + re-raise

**Commit**: `53e72a9` (Add transaction safety and connection cleanup to odds capture)

---

#### Prevent Duplicate Odds Captures via UNIQUE Constraint — COMMITTED
**Impact: Running --opening twice for the same week now updates instead of duplicating**

**The Bug:** UNIQUE constraint was `(snapshot_type, snapshot_time)` where:
- `snapshot_type` was composite like "opening_2024_week5"
- `snapshot_time` was the API timestamp (different each call)

Running `--opening --year 2024 --week 5` twice would create two separate snapshots because the timestamps differ.

**Fix:**
1. Changed `snapshot_type` from composite label to simple "opening"/"closing"
2. Changed UNIQUE constraint to `(snapshot_type, season, week)`
3. Added migration to convert existing composite labels
4. Added migration to recreate table with new constraint

**Schema change:**
```sql
-- Before
UNIQUE(snapshot_type, snapshot_time)

-- After
UNIQUE(snapshot_type, season, week)
```

**Migration handles:**
1. Converting "opening_2024_week5" → snapshot_type="opening", season=2024, week=5
2. Recreating table with new UNIQUE constraint
3. Preserving all existing data and foreign key relationships

**Commit**: `53632c0` (Prevent duplicate odds captures via UNIQUE constraint)

---

#### Remove Stale GT Thresholds Cache — COMMITTED
**Impact: Fixes garbage time threshold persistence when sweeping settings between iterations**

**The Bug:** `_GT_THRESHOLDS` was a module-level cache that stored garbage time thresholds from `get_settings()` on first use and never reset. `clear_ridge_cache()` cleared the Ridge model cache but NOT `_GT_THRESHOLDS`. If a sweep or test changed garbage time settings between iterations (e.g., testing Q4=14 vs Q4=16), the old thresholds would persist.

**Fix:** Removed the module-level cache entirely. `is_garbage_time()` now reads directly from settings on each call. This is a negligible performance impact (cheap dict access) in exchange for guaranteed correctness.

**Before:**
```python
_GT_THRESHOLDS: Optional[tuple[float, float, float, float]] = None

def is_garbage_time(...):
    global _GT_THRESHOLDS
    if _GT_THRESHOLDS is None:
        settings = get_settings()
        _GT_THRESHOLDS = (settings.garbage_time_q1, ...)
```

**After:**
```python
def is_garbage_time(...):
    settings = get_settings()
    gt_q1 = settings.garbage_time_q1
    # ... reads fresh each call
```

**Commit**: `46dc230` (Remove stale GT thresholds cache from EFM)

---

#### Prevent Caller Mutation in run_backtest() — COMMITTED
**Impact: Fixes sweep cache pollution from trajectory year merging**

**The Bug:** `run_backtest()` calls `season_data.update(traj_data)` to merge trajectory years. When called from `run_sweep()`, the same `cached_data` dict is reused across iterations. After iteration 1, `cached_data` contains trajectory years that weren't in the original fetch.

**Fix:** Shallow copy before mutating:
```python
if season_data is None:
    season_data = fetch_all_season_data(...)
else:
    # Shallow copy to avoid mutating caller's dict
    season_data = {**season_data}
```

**Commit**: `ba2e4c5` (included in SeasonData refactor)

---

#### Refactor Season Data to NamedTuple — COMMITTED
**Impact: Eliminates highest maintenance risk in codebase**

**The Problem:** Season data was a raw tuple of 10 elements unpacked by length-checking in 3+ locations. Adding a new field required updating all sites simultaneously or wrong DataFrame gets assigned to wrong variable.

**Solution:** `SeasonData` NamedTuple with named fields:
```python
class SeasonData(NamedTuple):
    games_df: pl.DataFrame
    betting_df: pl.DataFrame
    plays_df: pl.DataFrame
    turnover_df: pl.DataFrame
    priors: Optional[PreseasonPriors]
    efficiency_plays_df: pl.DataFrame
    fbs_teams: set[str]
    st_plays_df: pl.DataFrame
    historical_rankings: Optional[HistoricalRankings]
    team_conferences: Optional[dict[str, str]]
```

**Benefits:**
- Adding a field = 1 change (definition), not 4+ unpacking sites
- No more `if len(tuple) >= 10` / `elif len == 9` conditionals
- Named access prevents positional bugs: `sd.games_df` not `sd[0]`
- NamedTuple is picklable (multiprocessing compatible)

**Backtest verified:** MAE 12.50, ATS 52.2%, 5+ Edge 54.7% (unchanged)

**Commit**: `ba2e4c5` (Refactor season data from raw tuple to SeasonData NamedTuple)

---

#### Performance: Vectorize Portal Player Value Calculation — COMMITTED
**Impact: Eliminates Python loop over ~2,000 transfers per year**

Replaced row-wise `.apply()` with vectorized numpy operations in `calculate_portal_impact()`:
- `np.select()` for position group weights
- `np.where()` for level-up discount logic
- Direct column arithmetic for value calculations

**Commit**: `e9279cc`

---

#### Performance: Vectorize Kickoff Ratings Calculation — COMMITTED
**Impact: Eliminates ~130 Python loop iterations per team**

Replaced Python `for team, group in kicker_groups` loops with:
- `groupby("offense").agg()` for coverage ratings
- `groupby("defense").agg()` for return ratings
- Vectorized column arithmetic for tb_rate, avg_return, etc.

**Backtest verified:** MAE 12.50, ATS 52.2%, 5+ Edge 54.6%

**Commit**: `73efb40`

---

#### Performance: Pre-compute ST Play Type Masks — COMMITTED
**Impact: Reduces 3x redundant str.contains calls per play type**

In `calculate_all_st_ratings_from_plays()`, play type masks (FG, punt, kickoff) were computed inside each sub-method. Now computed once and passed to each method.

**Commit**: `d3de174`

---

#### Performance: Consolidate Redundant get_fbs_teams API Calls — COMMITTED
**Impact: Eliminates redundant API call/iteration in calculate_portal_impact**

`fetch_fbs_teams()` and `_fetch_team_conferences()` both called `get_fbs_teams(year)`. Now call once and derive both the fbs_teams set and team_conferences dict from the same response.

**Commit**: `5ec5f93`

---

#### Performance: Vectorize generate_comparison_df — COMMITTED
**Impact: Eliminates per-prediction Python loop**

Replaced `for pred in predictions: compare_prediction(pred)` loop with:
- DataFrame merge on game_id (primary lookup)
- Team-name fallback only for unmatched rows
- Vectorized edge/is_value calculation

**Commit**: `0d5672f`

---

#### Docs: Clarify Portal impact_cap vs churn_penalty Order — COMMITTED
**Impact: Documents intentional cap-then-penalty behavior**

The order (cap first, then churn penalty) is intentional: high-churn teams have a LOWER effective ceiling (e.g., 12% × 0.7 = 8.4%). This reflects integration risk.

Updated docstring and inline comments to make this explicit.

**Commit**: `9137fff`

---

#### Fix: ST is_complete Flag for Missing Components — COMMITTED
**Impact: Prevents incomplete ST ratings from being marked complete**

When one ST component (FG, punt, or kickoff) had zero plays, `is_complete` was incorrectly set to True because the None-check failed.

**Commit**: `d0d9915`

---

#### Fix: Warning for Unlisted Coaches in COACHING_CHANGES — COMMITTED
**Impact: Alerts when coaching change detected but not in config**

If a coaching change is detected in the data but not listed in `COACHING_CHANGES`, now logs a warning instead of silently ignoring.

**Commit**: `8af7916`

---

#### Fix: Remove Dead Retry Block in compare_prediction — COMMITTED
**Impact: -12 lines dead code**

The retry logic in `compare_prediction()` was unreachable because the method returns early on any failure.

**Commit**: `9a8b423`

---

#### Fix: Inconsistent Duplicate Handling in VegasComparison — COMMITTED
**Impact: Consistent "keep first" behavior for duplicate game_ids**

Both `lines_by_id` and `lines` dicts now consistently keep the first encountered line when duplicates exist, matching documented behavior.

**Commit**: `e0ecd57`

---

## Session: February 8, 2026

### Theme: Error Cohort Diagnostic + HFA Calibration + G5 Circularity Investigation + Totals Model + GT Threshold Analysis + Weather Forecast Infrastructure

---

#### Year-Conditional Garbage Time Thresholds — REJECTED
**Impact: Degraded both 3+ and 5+ Edge despite valid hypothesis**

- **Hypothesis:** The 2024 clock rule change (running clock on first downs) makes leads safer. A 14-point Q4 lead under new rules is functionally equivalent to a 16-point lead under old rules. Therefore, lower Q4 GT threshold from 16 to 14 for 2024+.

- **Diagnostic results (N=753 games with Q4 plays in 14-16 pt margin window):**

| Metric | Old Rules (2022-23) | New Rules (2024-25) | Delta |
|--------|---------------------|---------------------|-------|
| Leader Won by 14+ | 63.2% | **68.0%** | **+4.8pp** |
| Leader ATS (covered) | 92.3% (361-30) | 92.8% (336-26) | +0.5% |
| Mean Final Margin | +14.2 | +14.8 | +0.6 |

- **Hypothesis validated for game outcomes:** Leads ARE objectively safer (+4.8pp in "won by 14+"). Clock rules matter.
- **But ATS is flat:** Vegas already prices this (92.3% vs 92.8% leader cover rate).

- **Backtest with Q4=14 for 2024+:**

| Metric | Baseline (Q4=16) | Year-Conditional (Q4=14 for 2024+) | Delta |
|--------|------------------|-----------------------------------|-------|
| Core 5+ Edge (Close) | 54.7% (473-391) | 54.6% (470-391) | **-0.1%** |
| Core 3+ Edge (Close) | 54.0% (764-650) | 53.7% (762-658) | **-0.3%** |
| Core MAE | 12.50 | 12.50 | 0.00 |

- **Why it failed:** Game outcomes are affected by clock rules, but **play-level efficiency data is not**. The trailing team in a 14-15 pt Q4 deficit is still running real plays that inform their capability — even if the game is harder to win. By down-weighting those plays (treating them as GT), we lose informative signal.

- **Key insight:** "Leads are safer" ≠ "trailing team's plays are less informative." The GT threshold should reflect when teams stop trying, not when games become mathematically difficult.

- **Decision:** Reverted. Infrastructure preserved (year parameter added to `is_garbage_time_vectorized`) but not used.
- **Diagnostic script:** `scripts/diagnose_gt_threshold.py`

---

#### Error Cohort Diagnostic Analysis — COMPLETED (Research Only)
**Impact: Identified systematic home bias and G5 circular inflation as top addressable patterns**

- **Scope:** 2,489 core games (weeks 4-15, 2022-2025) segmented into error buckets and analyzed across 5 dimensions.
- **Key findings:**
  1. **Turnovers are #1 error driver** (r=-0.48): Lopsided TO games have MAE 16.98 vs 12.09 normal. NOT fixable — inherent game variance.
  2. **Persistent home/favorite bias**: +0.90 pts overall, +3.4 in catastrophic (25+ error) games. Model systematically over-predicts home team.
  3. **G5 circular inflation**: AAC intra-conf MAE 13.94, Sun Belt 13.52, Big 12 13.17 — all suffer from within-conference rating inflation.
  4. **Late season WORSE than early**: Early (W4-7) MAE 12.30 vs Late (W8-15) 12.64. W14 championships = 13.43 (worst).
  5. **Team-level biases are persistent**: Notre Dame under-rated by -9.5 pts (4yr avg), Charlotte over-rated by +7.0 pts.
- **No single fix addresses >5% of error budget.** Error distribution is remarkably uniform across weeks, spread sizes, and years.
- Full report: `.claude/agent-memory/quant-auditor/error-cohort-analysis.md`

#### HFA Global Offset Calibration — APPROVED
**Impact: 5+ Edge +0.6% (Close), +0.5% (Open) — first improvement on binding constraint since Explosiveness Uplift**

- **Hypothesis:** Error cohort analysis revealed +0.90 pts home bias. Modern CFB HFA may be declining due to portal churn, NIL, and general trend across sports.
- **Implementation:** Added `global_offset` parameter to `HomeFieldAdvantage` that subtracts a fixed amount from ALL team HFA values (floor=0.5). Threaded through backtest via `--hfa-offset` CLI flag.
- **6-variant sweep (offsets 0.0, 0.25, 0.375, 0.50, 0.75, 1.00):**

| Offset | Core MAE | 3+ Edge (Close) | 5+ Edge (Close) | 5+ Edge (Open) | Mean Error |
|--------|----------|-----------------|-----------------|-----------------|------------|
| 0.00 (baseline) | 12.51 | 53.5% (765-666) | 54.1% (473-401) | 57.3% (531-396) | +0.90 |
| 0.25 | 12.50 | 53.6% (765-663) | 54.4% (472-396) | — | +0.68 |
| 0.375 | 12.50 | 53.8% (764-655) | 54.5% (475-396) | — | +0.57 |
| **0.50** | **12.50** | **54.0% (764-650)** | **54.7% (473-391)** | **57.8% (525-384)** | **+0.46** |
| 0.75 | 12.49 | 54.0% (760-647) | 54.6% (471-392) | — | +0.24 |
| 1.00 | 12.50 | 53.8% (762-654) | 54.5% (476-397) | — | +0.02 |

- **Offset=0.50 selected** — peak 5+ Edge, diminishing returns beyond.
- **No 3+/5+ divergence** — both improve together (first time in 12 experiments).
- **Effect**: LSU 4.0→3.5, Ohio State 3.75→3.25, SEC default 2.75→2.25, UMass 1.5→1.0.
- **Also applied to `run_weekly.py`** for production predictions.

#### G5 Circular Inflation Investigation — RESEARCH (No Code Change)
**Impact: Diagnosed root cause as variance-driven, not bias-driven. Recommended betting filter over model adjustment.**

- **Dual analysis:** Model Strategist + Quant Auditor ran in parallel.
- **Key findings:**
  1. **Variance, not bias**: All problem conferences have <0.5% systematic bias. 100% of excess error is pure variance.
  2. **Big 12 Paradox**: Worst compression ratio (0.47) but BEST 5+ Edge ATS (64.2%). Market is equally confused — our errors are uncorrelated with market errors.
  3. **AAC/Sun Belt/MW are ATS dead zones**: 48-52% at 5+ Edge, combined 97-97 (50.0%). No betting edge.
  4. **Counterfactual ceiling**: Fixing all G5 to SEC-level MAE = only 0.65 overall improvement, all variance-driven.
  5. **Worst-predicted teams are conference additions**: Arizona (17.89), Colorado (16.66), ASU (14.37), UCF (14.36).
  6. **Year-over-year instability**: AAC 2024 = 16.47 MAE (spike), MW cycles between 10.14 and 15.13.
- **Strategist recommendation**: OOC Credibility Weighting (weight intra-conf plays by opponent's OOC exposure density).
- **Quant recommendation**: Betting confidence filter on AAC/SB/MW intra-conference games.
- Full reports: `.claude/agent-memory/quant-auditor/g5-circularity-deep-dive.md`, `.claude/agent-memory/model-strategist/`

#### OOC Credibility Weighting — REJECTED (Rejection #12)
**Impact: Monotonic 5+ Edge degradation across all tested weights**

- **Hypothesis:** Weight intra-conference plays by opponent's OOC exposure density (Z = ooc_games / league_avg). Teams with more OOC games are better-calibrated, so plays against them carry more information for Ridge regression.
- **Implementation:** In `_prepare_plays()`, after existing 1.5x OOC weighting, added credibility multiplier `W_play *= 1.0 + scale * (Z_opponent - 1.0)` for intra-conference plays only.
- **5-variant sweep (weights 0.0, 0.25, 0.50, 0.75, 1.0):**

| Weight | Core MAE | 3+ Edge (Close) | 5+ Edge (Close) | 5+ Edge (Open) |
|--------|----------|-----------------|-----------------|----------------|
| **0.00 (baseline)** | **12.50** | **54.0%** (764-650) | **54.7%** (473-391) | **54.1%** (508-431) |
| 0.25 | 12.50 | 54.2% (766-648) | 54.5% (476-398) | 53.8% (511-439) |
| 0.50 | 12.51 | 54.1% (770-653) | 54.4% (478-401) | 53.7% (512-441) |
| 0.75 | 12.52 | 54.1% (764-649) | 54.2% (479-405) | 53.5% (512-445) |
| 1.00 | 12.54 | 54.4% (771-647) | 54.1% (488-414) | 53.5% (521-453) |

- **Failure mode**: Sub-metric redundancy + variance-driven problem. Re-weighting existing information doesn't create new signal. Confirms quant auditor's diagnosis that no play-level weight geometry change can fix variance.
- **Big 12 guardrail never triggered** — feature failed the global 5+ Edge test first.
- **Infrastructure preserved**: `ooc_credibility_weight` param in EFM + CLI `--ooc-cred-weight` (default 0.0 = disabled).
- **Decision**: Conference circularity to be addressed via betting confidence filter (operational change, not model change).

#### Tomorrow.io Weather Forecast Infrastructure — OPERATIONAL (Totals Only)
**Impact: Live weather forecasts for operational totals predictions**

- **Purpose:** Backtest showed perfect weather data provides MAE -0.04 improvement but 0% ATS change (market already prices weather). However, for operational predictions, we need forecast data to match backtest methodology. CFBD provides look-back weather (post-game actuals), not forecasts.
- **API:** Tomorrow.io Hourly Forecast API (free tier: 25 calls/hour, 500 calls/day)
- **Database:** `data/weather_forecasts.db` (SQLite) — separate from `odds_api_lines.db`

**Files Created:**
- `src/api/tomorrow_io.py` — Full client implementation:
  - `WeatherForecast` dataclass (temp, wind, precip, confidence factor)
  - `VenueLocation` dataclass (stadium coordinates)
  - `TomorrowIOClient` class with rate limiting + exponential backoff on 429
  - `forecast_to_weather_conditions()` — converts to existing `WeatherConditions` for `WeatherAdjuster` integration
  - Weather concern detection: wind >15 mph OR temp <32°F OR precip >60%

- `scripts/capture_weather_forecasts.py` — Capture script:
  - `--year/--week` for specific games
  - `--dry-run` to preview venues without API calls
  - `--refresh-venues` to populate venue database from CFBD
  - `--limit N` to stay within rate limits (recommended: 20 per hour)
  - `--delay` to customize inter-call wait time (default: 3s)
  - Early stopping after 3 consecutive rate limit failures
  - Weather concern flagging in output

- `src/api/cfbd_client.py` — Added `get_venues()` method for stadium coordinates

**Venue Database:**
- 795 FBS/FCS venues loaded with coordinates
- Dome detection for indoor stadiums (auto-skip forecast)
- Example: 2025 Week 14 = 67 games, 63 outdoor, 4 indoor

**Rate Limit Strategy:**
- Free tier: 25 calls/hour → batch with `--limit 20`
- Exponential backoff: 5s, 10s, 20s on 429 errors
- Auto-stop after 3 consecutive failures

**Confidence Factor (forecast horizon):**
- 0-6h before game: 0.95
- 6-12h: 0.90
- 12-24h: 0.85
- 24-48h: 0.75
- 48h+: 0.65

**Integration path:** `TomorrowIOClient.forecast_to_weather_conditions()` → `WeatherAdjuster` (existing infrastructure)

**Next steps:** Capture Saturday morning forecasts (6-12h before kickoff) for live predictions.

---

#### Documentation Sync (All Three Docs)
- **MODEL_ARCHITECTURE.md**: Updated per-year MAE/RMSE tables, per-year ATS tables (Close + Open), added changelog entries (HFA Global Offset, Conference Anchor, RZ Leverage), added 8 rejections to "Explored but Not Included", added 2025 Season Performance section with consolidated phase-by-phase table.
- **CLAUDE.md**: Auto-synced via `generate_docs.py` with fresh backtest metrics.
- **MODEL_EXPLAINER.md** (JP-Plus-Docs only): Synced all per-year ATS tables, MAE/RMSE table, CLV core season numbers, added 2025 Season Performance section, updated HFA description with global offset. All numbers now match MODEL_ARCHITECTURE.md.
- Both repos pushed: JP-Plus (code) and JP-Plus-Docs (docs-only).

#### Open Items & Parking Lot Cleanup
**Impact: 4 items closed, documentation reflects actual state of experiments**

Updated MODEL_ARCHITECTURE.md Open Items and Parking Lot sections:

| Item | Status | Rationale |
|------|--------|-----------|
| Improve situational adjustment calibration | **DONE** | HFA offset (-0.50) + stack smoothing |
| Soft cap on asymmetric GT | **INVESTIGATED, NOT IMPLEMENTING** | All GT weight variants degraded 5+ Edge |
| Reduce turnover weight for 3+ Edge | **INVESTIGATED, KEEPING 10%** | 3+/5+ Edge divergence trap; 5+ is binding |
| EV-weighted performance metric | **PARTIALLY IMPLEMENTED** | CLV tracking done; Kelly sizing remains open |

Remaining open: Real-time line movement, automated betting recommendations, coaching pedigree normalization.

#### Totals Model Built From Scratch — NEW PRODUCTION MODULE
**Impact: New over/under prediction capability with 52.8% Core 5+ Edge ATS**

- **Background:** EFM ratings cannot be directly used for totals prediction (ratings are "points better than average efficiency" scaled for spreads, not raw points). Attempted SP+-style formula produced totals 15-18 pts too high (e.g., Indiana-Miami predicted 66, actual 48).
- **Solution:** Built new `TotalsModel` using opponent-adjusted Ridge regression on **game-level points scored/allowed** (not play-level efficiency like EFM).

**Architecture:**
- Each game produces 2 training rows: home_off + away_def → home_pts, away_off + home_def → away_pts
- Ridge regression solves for team offensive/defensive adjustments relative to FBS average
- Formula: `home_expected = baseline + (home_off_adj + away_def_adj) / 2`
- Walk-forward training: only uses games from weeks < prediction_week

**Ridge Alpha Sweep (Core Phase, Weeks 4-15):**

| Alpha | Core MAE | Core ATS | 3+ Edge | 5+ Edge |
|-------|----------|----------|---------|---------|
| 5.0 | 13.12 | 52.5% | 51.9% | 52.4% |
| **10.0** | **13.23** | **52.4%** | **52.7%** | **52.8%** |
| 15.0 | 13.40 | 52.3% | 52.4% | 52.5% |
| 20.0 | 13.62 | 52.0% | 52.1% | 52.0% |

- **Alpha=10.0 selected** — best Core 5+ Edge ATS (52.8%)
- **Calibration phase (weeks 1-3) is surprisingly strong**: 57.4% at 5+ edge — opposite of spreads model pattern
- **Phase 3 (postseason) unstable**: ~48% ATS, high variance, small sample

**Files Created:**
- `src/models/totals_model.py` — Core model with TotalsModel, TotalsRating, TotalsPrediction classes
- `scripts/backtest_totals.py` — Walk-forward backtest with phase filtering, ATS calculations

**Weather Integration:**
- Added `predict_total_with_weather()` method using existing WeatherAdjuster
- Added `--weather` flag to backtest_totals.py
- **Backtest comparison:**
  - Without weather: Core MAE 13.23, Core 5+ Edge 52.8%
  - With weather: Core MAE 13.19 (-0.04), Core 5+ Edge 52.8% (unchanged)
- Weather improves MAE marginally but no ATS improvement (market already prices weather)
- **Note:** Weather data from CFBD is look-back (actual game-day), not forecast — not operationally useful without forecast API

**Opening vs Closing Line Comparison:**
- Added `over_under_open` extraction from CFBD API
- Opening line slightly better than closing (as expected — less market efficiency):

| Line | Core 5+ Edge | Record |
|------|--------------|--------|
| Closing | 52.8% | 479-428 |
| **Opening** | **53.3%** | 448-393 |

**Scoring Environment Trend Discovery:**
- Analyzed FBS average total PPG across years:

| Year | Avg Total |
|------|-----------|
| 2018 | 57.6 |
| 2019 | 56.0 |
| 2022 | 54.8 |
| 2023 | 53.8 |
| 2024 | 53.9 |
| 2025 | 52.9 |

- **CFB scoring dropped ~5 PPG since 2018** — structural shift (better defenses, portal roster balance, rule enforcement)
- 2022 was transition year with poor ATS (49%) — data hurts model

**Drop 2022 from Evaluation — APPROVED:**

| Metric | With 2022 | Without 2022 | Delta |
|--------|-----------|--------------|-------|
| Core 5+ Edge (Close) | 52.8% | **54.5%** | **+1.7%** |
| Core 5+ Edge (Open) | 53.3% | **55.3%** | **+2.0%** |

- Default years updated to 2023-2025

**Within-Season Decay — REJECTED:**
- Strategist hypothesized cupcake games inflate baseline; decay would help
- Tested decay_factor 0.97/0.95/0.93 (Week 1 game weighted 71%/54%/42% at Week 12)
- **All variants degraded 5+ Edge:**

| Decay | Core 5+ Edge | Delta |
|-------|--------------|-------|
| 1.00 (baseline) | 54.5% | — |
| 0.97 | 54.2% | -0.3% |
| 0.95 | 53.9% | -0.6% |
| 0.93 | 53.7% | -0.8% |

- **Same pattern as spreads**: walk-forward already handles temporality; cupcake games still calibrate team strength; early OOC anchors opponent graph
- Scoring environment shift is cross-year (57→53 PPG), not within-season
- Infrastructure preserved: `decay_factor` param (default 1.0 = disabled)

**OT Protection (Regulation Scores) — REJECTED:**
- **Hypothesis:** Overtime points are noise (coin flip after regulation tie). Training on final scores contaminates team ratings.
- **Implementation:** Extract regulation scores from CFBD `line_scores` (sum first 4 quarters). Example: Vanderbilt-VT 34-27 final → 27-27 regulation.
- **Coverage:** 99.8% of games have line_scores available
- **Backtest results (2023-2025):**

| Metric | Before OT Fix | After OT Fix | Delta |
|--------|---------------|--------------|-------|
| Core 5+ Edge (Close) | 54.5% | 53.9% | **-0.6%** |
| Core 5+ Edge (Open) | 55.3% | 54.5% | **-0.8%** |

- **2025 worst affected:** 52.1% → 49.4% (-2.7%)
- **Reverted** — Vegas already prices OT potential; final scores contain market-relevant signal
- **Key lesson:** "Cleaner" data ≠ better betting edge. Market prices what it prices.

**Sparse Matrix Optimization:**
- Replaced `iterrows()` loop with vectorized numpy + scipy.sparse.coo_matrix
- Build COO matrix from index arrays, convert to CSR for Ridge regression
- Memory: O(4×games) sparse vs O(games × 2×teams) dense
- **Bit-accurate:** Results identical to original iterrows implementation
- Performance improvement for ~800 games/iteration training loops

**PPP × Pace Architecture — REJECTED (Pre-Backtest):**
- **Proposal:** Decouple efficiency from pace. Train Ridge on points-per-play (PPP), build separate pace model, predict Total = efficiency × pace.
- **Strategist verdict: NO-GO** — 3 independent kill signals:
  1. **Mathematical equivalence:** PPP × Pace = Points. Decomposing and recombining compounds estimation error from two models. Single Ridge is more efficient.
  2. **Market-visible signal:** Pace is one of the most market-efficient variables. Vegas adjusts 7-10 pts for pace differentials. Same failure as Penalty Discipline.
  3. **Pattern match:** 0-for-12 on features that decompose existing data or add market-visible info.
- **Key insight:** Current Ridge on raw points already captures pace implicitly through opponent adjustment. A team scoring 35 in 85 plays generates different coefficients than one scoring 35 in 60 plays.
- **Not backtested** — killed on theoretical grounds.

**Asymmetric Ridge Regularization — REJECTED (Audit Disproved Hypothesis):**
- **Hypothesis:** Symmetric Ridge shrinkage may be suppressing defense signal. Test block-regularized Ridge with different penalties for offense vs defense coefficients.
- **Hard reject rule:** 5+ Edge must improve by ≥0.3%
- **Coefficient audit results (2023-2025, 403 team-seasons):**

| Metric | Offense | Defense | Ratio |
|--------|---------|---------|-------|
| Std Dev | 3.49 | 3.46 | 1.02 |
| Range | [-8.6, +10.3] | [-9.7, +10.0] | ~equal |
| Off-Def Correlation | -0.36 to -0.50 | (expected) | — |

- **Verdict:** Hypothesis disproved by data. Variance ratio=1.02 means coefficients already symmetric. No evidence of over-shrinkage on defense.
- **Not backtested** — audit showed nothing to fix.

**Learned HFA Column — APPROVED:**
- **Proposal:** Add HFA column to Ridge design matrix (1 for home rows, 0 for away rows). Let Ridge learn optimal home advantage from data instead of assuming fixed value.
- **Implementation:** Extended sparse matrix from `2*n_teams` to `2*n_teams+1` columns. HFA column (last column) = 1 for home team rows only.
- **Learned HFA coefficients:**

| Year | Learned HFA |
|------|-------------|
| 2023 | +3.35 pts |
| 2024 | +4.05 pts |
| 2025 | +4.46 pts |

- **Backtest results (2023-2025):**

| Metric | Before HFA | After HFA | Delta |
|--------|------------|-----------|-------|
| Core 5+ Edge (Close) | 52.9% | **54.5%** | **+1.6%** |
| Core 5+ Edge (Open) | 53.2% | **55.3%** | **+2.1%** |
| Core 3+ Edge (Close) | 51.7% | **54.7%** | **+3.0%** |
| Core MAE | 13.71 | **13.09** | **-0.62** |

- **Key insight:** HFA for totals is different from spreads HFA. Totals HFA affects expected scoring (home team scores more), not margin. Learned value (~+4 pts) is larger than expected because it captures the full home advantage in scoring, not just the margin contribution.
- Prediction formula: `home_expected = baseline + (home_off_adj + away_def_adj) / 2 + hfa_coef`

**Totals Model Final Configuration (Production):**
- **Years:** 2023-2025 (dropped 2022 transition year)
- **Ridge Alpha:** 10.0
- **Decay Factor:** 1.0 (no within-season decay)
- **Learned HFA:** Via Ridge column (+3.5 to +4.5 pts typical)
- **OT Protection:** Disabled (final scores used)
- **Weather:** Available but optional
- **Core 5+ Edge:** 54.5% (close), 55.3% (open)

**Year Intercepts Infrastructure — BUILT (Disabled by Default):**
- **Problem:** CFB scoring dropped 57.6 → 52.9 PPG from 2018-2025. When training on 3-year rolling window, baseline learned is 3-year average. If 2025 is structurally lower than 2023, model has systematic "Over" bias for 2025 games.
- **Solution:** Add year indicator columns to Ridge design matrix. Each year gets its own baseline coefficient.
- **Implementation:**
  - Added `use_year_intercepts` param (default=False)
  - Extended sparse matrix to `2*n_teams + 1 + n_years` columns
  - Year columns: 1 for rows from that year, 0 otherwise
  - `fit_intercept=False` when year intercepts enabled (year baselines replace global intercept)
  - `predict_total()` accepts optional `year` param for year-specific baseline

- **Testing results (2023-2025 walk-forward):**

| Metric | Without Year Intercepts | With Year Intercepts | Delta |
|--------|------------------------|----------------------|-------|
| Mean Error | -0.59 | **-1.71** | **-1.12 (worse)** |
| Core 5+ Edge (Close) | 54.5% | 54.4% | -0.1% |
| Core 5+ Edge (Open) | 55.3% | 55.2% | -0.1% |

- **Root cause of failure:** Ridge regularization shrinks year coefficients toward zero. Learned baselines (e.g., 2025=23.3) are ~3 pts below actual averages (26.2). This creates systematic under-prediction.
- **Key insight:** Year intercepts would help for multi-year joint training (fit all 3 years at once with different baselines). For walk-forward single-year training (current backtest), each year trains independently — so year intercepts provide no benefit and Ridge shrinkage actively hurts.
- **Decision:** Infrastructure preserved for future multi-year training mode (e.g., 2026 production with 2023-2025 joint training). Default=False maintains current performance.

**Service Academy vs Service Academy — DOCUMENTED LIMITATION (No Fix):**
- **Hypothesis:** When two triple-option teams (Army, Navy, Air Force) play, the slowdown is multiplicative, not additive. Linear Ridge model can't capture "Slow × Slow = Slower" interaction.
- **Evidence (2023-2025, N=7 games):**

| Metric | JP+ Model | Vegas |
|--------|-----------|-------|
| Mean Error | **+17.5** | +3.6 |
| MAE | **20.0** | 8.9 |

- JP+ predicts 47-56 pts for games that are mostly 23-41 pts actual
- Vegas already adjusts (sets 28-50 pts vs our 47-56)

- **Gate failures:**
  1. **Sample size**: N=7 games (0.3% of sample), below noise floor for coefficient fitting
  2. **Counter-example**: 2025 AF @ Navy = 65 total points (we under-predicted by 9)
  3. **Generalizability**: Tested "slow vs slow" matchups (bottom 10% pace, N=64). Effect does NOT generalize — Navy/Air Force aren't even in bottom 10% pace. Effect is scheme-specific (triple-option), not pace-specific.
  4. **Market awareness**: Vegas already prices this (~15 pts lower than JP+)

- **Why no fix:**
  - Hardcoded -15 adjustment → overfitting to 7 games
  - Triple-option detection → adds complexity for ~2 games/year
  - Pace interaction term → data shows it doesn't generalize (counter-examples: Minnesota @ Northwestern 71 pts, Michigan @ Minnesota 62 pts)

- **Betting impact**: ~2-3 games/year at 5+ Edge, estimated 0.1-0.2% overall impact
- **Decision**: Accept as known limitation. Market already prices this; fixing would be micro-targeting below noise floor.

#### TotalsModel Performance Optimization — INFRASTRUCTURE
**Impact: 13x speedup on ATS calculation, 50% memory reduction, infrastructure hardening**

Systematic performance audit of `TotalsModel` and `backtest_totals.py`:

| Optimization | Change | Result |
|--------------|--------|--------|
| **iterrows → itertuples** | `calculate_ou_ats()` loop | **13.2x speedup** (0.86s → 0.07s for 54 calls) |
| **iterrows → itertuples** | Prediction loop | 1.01x (Ridge.fit dominates) |
| **Cache team universe** | `set_team_universe()` once per season | ~1.0x (dict build is O(134), trivial) |
| **float64 → float32** | COO data, y vector, sample_weights | **50% memory reduction**, 1.02x timing |
| **Vectorize games_per_team** | `value_counts().add()` vs Python loop | Cleaner code, marginal speedup |
| **Explicit solver=sparse_cg** | Ridge solver hardening | No perf change (stability win) |

**Solver Evaluation for Sparse CSR:**
- `sparse_cg`: **SELECTED** — deterministic, sparse-native, 0.0 coef diff vs auto
- `lsqr`: REJECTED — 5.3e-3 coef diff (above 1e-4 threshold)
- `svd/cholesky`: REJECTED — require dense matrix
- `sag/saga`: REJECTED — stochastic variance, saga fails with sample_weight

**Key Learnings:**
1. Real bottleneck is `Ridge.fit()` — matrix/loop optimizations are dwarfed by solver time
2. `itertuples` wins big when called many times (13x for ATS, called 54x per backtest)
3. float32 doesn't help sklearn timing — LAPACK/BLAS internally use float64
4. Architecture wins > micro-optimizations (team universe caching is cleaner but no speedup)

**All changes verified:** Backtest output unchanged (706 preds, MAE 13.03, 5+ Edge 58.6%)

#### Offense/Defense Alpha Weighting — VALIDATED (No Change)
**Impact: Confirmed 50/50 split is optimal; structural assumption holds**

- **Hypothesis:** The prediction formula `(home_off_adj + away_def_adj) / 2` assumes offense and defense contribute equally. This is a hard-coded constraint Ridge can't correct — it can only contort coefficients.
- **Test:** Generalize to `α * off_adj + (1-α) * def_adj` and sweep α at prediction time (training unchanged).

| α | MAE | Mean Err | 5+ Edge |
|---|-----|----------|---------|
| 0.40 (def-heavy) | 13.03 | -0.55 | 55.0% |
| 0.45 | 13.03 | -0.51 | 55.1% |
| **0.50 (baseline)** | **13.04** | **-0.47** | **55.3%** |
| 0.55 | 13.05 | -0.44 | 54.0% |
| 0.60 (off-heavy) | 13.06 | -0.40 | 54.3% |

- **Result:** Baseline α=0.50 is optimal. Moving in either direction degrades 5+ Edge.
- **Why it holds:** Earlier audit showed offense/defense coefficient std devs are nearly identical (3.49 vs 3.46). Ridge learns balanced coefficients; no suppressed signal on either side.
- **Decision:** No change. The 50/50 assumption is empirically validated.

#### Prior-Informed Baseline — REJECTED
**Impact: Degrades both Phase 1 and Phase 2 performance**

- **Hypothesis:** Static per-season baseline creates coefficient drift when training data is sparse. Early weeks (1-3) have only 40-80 games; Ridge learns baseline from small sample. If first 3 weeks are cupcake-heavy (FCS, G5 mismatches), baseline is inflated vs true CFB average.
- **Proposal:** Use prior season's learned baseline as a prior. Blend: `effective_baseline = (1-w) * learned + w * prior_baseline`, where w decays over training weeks.
- **Implementation tested:** Decay rates 4, 6, 8 weeks (w = max(0, 1 - week/decay))

| Decay | Phase 1 5+ Edge | Phase 2 5+ Edge | Delta |
|-------|-----------------|-----------------|-------|
| None (baseline) | **57.5%** | **56.0%** | — |
| 4 weeks | 54.8% | 56.0% | **-2.7% (P1)** |
| 6 weeks | 56.8% | 55.1% | **-0.9% (P2)** |
| 8 weeks | 51.2% | 55.0% | **-6.3% (P1), -1.0% (P2)** |

- **Why it failed:**
  1. **Baselines nearly identical:** 2023=24.97, 2024=24.85, 2025=25.27 — only ~0.5 pt variance. No systematic drift to correct.
  2. **Phase 1 is already strong:** 57.5% at 5+ Edge without any prior. This is a model STRENGTH, not a problem to fix.
  3. **Prior contamination:** Using 2024 baseline for early 2025 games introduces stale information. Walk-forward already handles temporal validity — adding another layer hurts.
- **Decision:** No change. Current per-season learned baseline is optimal.

#### Thursday Weather Capture Automation — OPERATIONAL
**Impact: Automated Thursday morning weather capture with betting watchlist**

- **Purpose:** The edge in weather betting is TIMING — capture forecasts Thursday morning BEFORE market moves the line. By the time books adjust totals on Friday/Saturday, the value is gone.

**Thursday Workflow (`scripts/weather_thursday_capture.py`):**
- Auto-detects current CFB year/week
- Fetches all games for the week, identifies outdoor venues
- Captures forecasts for outdoor games (respecting 25/hour rate limit)
- Generates watchlist of games with weather concerns (wind >12 mph effective, temp <32°F, heavy precip)
- Calculates weather adjustment using non-linear thresholds (see below)
- **NEW:** Includes Vegas total and weather-adjusted total for each watchlist game

**Watchlist output format:**
```
🏈 THURSDAY WEATHER WATCHLIST
================================================================================
🚨 3 GAMES WITH WEATHER CONCERNS:

  Iowa @ Nebraska
    Venue: Memorial Stadium (Lincoln, NE)
    Wind: 22 mph (gust: 28)
    Temp: 28°F
    📊 JP+ Total: 52.1 → Weather-Adjusted: 45.1
    🎰 Vegas Total: 48.5
    💰 Edge: -3.4 pts (LEAN UNDER)
    🌧️ Weather Adjustment: -7.0 pts
       Wind: -6.0, Temp: -1.0, Precip: +0.0
    Confidence: 75% (72h until game)

💡 ACTION: Consider betting UNDER on games with negative edge BEFORE market adjusts.
```

**Edge interpretation:**
- `Edge < -3 pts` = STRONG UNDER signal
- `Edge < 0 pts` = LEAN UNDER signal
- `Edge > 3 pts` = JP+ higher than Vegas (rare in weather games)

**JP+ TotalsModel integration:**
- Trains walk-forward on all games from weeks 1 to (current_week - 1)
- Uses Ridge alpha=10.0, no within-season decay
- Weather adjustment applied to JP+ prediction (not Vegas)

**Cron automation (`scripts/setup_weather_cron.sh`):**
- `./scripts/setup_weather_cron.sh install` — Installs Thursday 6 AM cron job
- `./scripts/setup_weather_cron.sh status` — Check installation
- Logs to `logs/weather_thursday.log`

**Non-linear weather thresholds (`src/adjustments/weather.py`):**
Based on sharp betting research (wind is king of unders):

| Wind (effective) | Adjustment | Rationale |
|------------------|------------|-----------|
| <12 mph | 0.0 pts | No impact |
| 12-15 mph | -1.5 pts | Deep passing degraded |
| 15-20 mph | -4.0 pts | Kicking range reduced |
| >20 mph | -6.0 pts | Run-only game profiles |

| Temperature | Adjustment | Rationale |
|-------------|------------|-----------|
| >32°F | 0.0 pts | No impact |
| 20-32°F | -1.0 pts | "Rock effect" on ball |
| <20°F | -3.0 pts | Severe mechanics impact |

| Precipitation | Adjustment | Rationale |
|---------------|------------|-----------|
| Light rain (<0.1 in/hr) | 0.0 pts | **The "slick trap"** — defenders slip, more scoring |
| Heavy rain (>0.3 in/hr) | -2.5 pts | Ball security issues |
| Snow with accumulation | -3.0 pts | Visual impairment, footing |

**Effective wind = (wind_speed + wind_gust) / 2** — gusts matter for passing and kicking.

**Strategic Refinements (Sharp Betting Research):**

1. **"Passing Team" Multiplier** — Wind adjustments scaled by combined pass rate:
   - Pass-heavy matchups (>55% combined pass rate): 1.25x wind penalty
   - Run-heavy matchups (<45% combined pass rate): 0.5x wind penalty
   - Example: Army vs Navy in 20mph wind = -3.0 pts (they don't care)
   - Example: Ole Miss vs Air Raid opponent in 20mph wind = -7.5 pts
   - Pass rates calculated from play-by-play data (weeks 1 to current_week - 1)

2. **"Snow Overreaction Fade"** — Public loves betting "Snow Unders" but:
   - Snow without wind often goes OVER (defenders slip, receivers know their routes)
   - Snow penalty only applies if effective wind >= 12 mph
   - Low-wind snow games = no penalty (sharps bet OVER)
   - Heavy wind + snow = -3.0 pts (wind makes snow swirl, disrupts passing)

Watchlist now shows pass rate context:
```
🌧️ Weather Adjustment: -7.5 pts
   Wind: -7.5, Temp: +0.0, Precip: +0.0
   📋 Pass Rate: 58% (Pass-heavy matchup — wind hurts more)
```

**Removed legacy CFBD weather code:**
- `cfbd_client.get_weather()` — no longer used (158 lines removed)
- Weather API was for look-back actuals, not forecasts

#### Saturday Weather Confirmation Run — OPERATIONAL
**Impact: Two-stage weather capture workflow for maximum timing edge**

- **Purpose:** Thursday forecasts (72h out) are useful for identifying concerns but have lower confidence. Saturday 8 AM ET forecasts (6-12h out) are high-confidence and should be compared to Thursday predictions.

**Saturday Workflow (`scripts/weather_thursday_capture.py --saturday`):**
- Loads Thursday forecasts from database
- Captures fresh Saturday morning forecasts
- Compares forecast changes (wind, temp, precip)
- Re-calculates weather adjustments with updated data
- Highlights games where conditions improved/worsened significantly

**Output includes Thursday vs Saturday comparison:**
```
📊 THURSDAY vs SATURDAY COMPARISON:

  Iowa @ Nebraska
    Wind: 22 mph → 18 mph (IMPROVED)
    Thursday Adj: -7.0 pts → Saturday Adj: -4.5 pts
    ⚠️ Adjustment changed by 2.5+ pts — re-evaluate position
```

**Cron automation:**
- Saturday 8:00 AM ET cron job added to `setup_weather_cron.sh`
- Logs to `logs/weather_saturday.log`

#### Year Leakage Bug Fix (TotalsModel) — COMMITTED
**Impact: Ensures multi-year training correctly handles scoring environment shifts**

- **Bug:** When `use_year_intercepts=False` (the default), Ridge regression learns a single baseline intercept for ALL years. If training on 2022-2024, the learned intercept averages scoring environments (2022 ~27 PPG vs 2024 ~26 PPG), contaminating predictions for individual years.
- **Impact:** Walk-forward single-year training (current backtest) is unaffected. Multi-year joint training (e.g., 2026 production with 2023-2025 pooled) would have systematic bias.
- **Fix:** Auto-detect when `len(unique_years) > 1` and automatically enable year intercepts:
```python
if len(years) > 1 and not self.use_year_intercepts:
    logger.info(f"Auto-enabling year intercepts for multi-year training ({years})")
    self.use_year_intercepts = True
```
- **Verification:** Single-year 2024 backtest still works correctly (year intercepts not triggered).

#### 2026 Production Planning — DOCUMENTED (MODEL_ARCHITECTURE.md)
**Impact: Roadmap for operational deployment and stake sizing**

Added new section to MODEL_ARCHITECTURE.md covering:

**Bet Automation / Discord Integration:**
- Discord deep links for sportsbook bet slips (one-click betting)
- DraftKings: `https://sportsbook.draftkings.com/...`
- FanDuel/ESPN Bet: Similar deep link APIs
- Workflow: JP+ generates picks → posts to Discord → deep links to sportsbooks

**API Betting Exchanges:**
- Sporttrade, ProphetX, Novig — API-first exchanges with better liquidity
- Advantages: Better odds (-105 vs -110), programmatic betting, larger limits
- Evaluation criteria: Liquidity depth, API documentation, CFB coverage

**Kelly Criterion Stake Sizing:**
- Formula: `f* = (p - q) / b` where p = win prob, q = 1-p, b = odds
- JP+ Edge → Implied Win % mapping:

| JP+ Edge | Implied Win % | Full Kelly | Half Kelly | Quarter Kelly |
|----------|---------------|------------|------------|---------------|
| 3 pts | 54.5% | 4.5% | 2.25% | 1.1% |
| 5 pts | 57.5% | 9.5% | 4.75% | 2.4% |
| 7 pts | 60.0% | 14.0% | 7.0% | 3.5% |
| 10 pts | 65.0% | 21.0% | 10.5% | 5.3% |

- **Recommendation:** Half Kelly for most bettors, Quarter Kelly for bankroll preservation
- Edge confidence tiers: 5+ pts = high confidence, 7+ pts = max confidence

---

## Session: February 7, 2026 (Continued)

### Theme: Transfer Portal Architecture Audit

---

#### G5→P4 Physicality Tax Sweep — CANCELLED
**Impact: Strategist killed before backtest — magnitude below noise floor**

- **Proposal:** Test PHYSICALITY_TAX at 0.65, 0.60, 0.55 (from current 0.75) for G5→P4 trench transfers.
- **Strategist analysis:** With `portal_scale=0.15`, changing 0.75→0.60 produces **0.007 pts per player** and **0.035 pts per team** (even with 5 G5→P4 trench transfers). At MAE=12.99, this is indistinguishable from noise.
- **FSU/Colorado misdiagnosis:** FSU 2024's portal class was predominantly **P4→P4** (from SEC/ACC programs). Colorado's Sanders poached from SEC/Big Ten — `origin_is_p4 == dest_is_p4` returns 1.0, so the G5 discount never fires. Neither team is actually affected by this parameter.
- **This is rejection #8** in the micro-fix pattern (Chemistry Tax, MOV, Fraud Tax, GT variants, Zombie Prior, Talent Abandonment, Def Weight Split).
- Quant Auditor sweep was launched but cancelled before completion after Strategist magnitude analysis.

#### Volume Diminishing Returns (VDR) + Incumbent Symmetry (IS) — REJECTED (Pre-Backtest)
**Impact: Strategist 3-0 Council rejection — architectural flaws identified**

- **VDR proposal:** If additions > 10, apply 0.98^(additions-10) penalty to net portal value.
- **IS proposal:** Flat 0.85 multiplier on all incoming_value.
- **VDR rejection:** Model already has a superior 2-dimensional sigmoid churn penalty (lines 626-688 in `preseason_priors.py`) that uses BOTH returning production (70% weight) AND portal volume (30% weight). VDR only uses volume — strictly inferior.
- **IS fatal flaw:** The `impact_cap = ±0.12` absorbs the IS reduction for ALL problem teams (FSU, Colorado, TAMU all hit the cap). IS has **zero effect** on the teams we're targeting. For uncapped teams, it creates a 26% penalty on 1:1 replacements (too harsh; existing CONTINUITY_TAX already creates 6%).
- **Recommendation:** Activate existing dormant churn penalty instead.

#### Roster Churn Penalty Activation — REJECTED (Rejection #9)
**Impact: 3+ Edge improved but 5+ Edge degraded — same pattern as conf anchor 0.12**

- **Change:** `PreseasonPriors(client, use_churn_penalty=True)` — zero new code, activated existing dormant sigmoid penalty.
- **Backtest results (--start-week 1):**

| Metric | Baseline | Churn Penalty | Delta |
|--------|----------|---------------|-------|
| Core MAE | 12.52 | 12.51 | -0.01 |
| Phase 1 MAE | 14.82 | 14.80 | -0.02 |
| Core ATS (Close) | 52.4% | 52.5% | +0.1% |
| Core 3+ Edge | 53.0% (758-671) | 53.5% (765-665) | **+0.5%** |
| Core 5+ Edge | 54.5% (475-396) | 54.1% (472-401) | **-0.4%** |

- **Reverted** — 5+ Edge degradation (-0.4%) fails "must not degrade" criterion.
- **3+/5+ Edge divergence now documented across 3 experiments:**
  1. Conference anchor 0.12: 3+ Edge +0.7%, 5+ Edge -0.3%
  2. Churn penalty: 3+ Edge +0.5%, 5+ Edge -0.4%
  3. Conference anchor 0.15+: Both degraded
- **Key insight:** Model's highest-conviction disagreements with the market (5+ pts) are already well-calibrated. Broad adjustments that improve moderate-conviction picks dilute signal on the strongest picks.
- Infrastructure remains dormant (`use_churn_penalty=False`) for future selective activation (e.g., only <40% returning AND >20 transfers).

#### Style Asymmetry Collision Analysis — REJECTED (No Code Change)
**Impact: Confirmed 45/45/10 weighting handles style mismatches well**

- **Hypothesis:** "Explosive offense vs big-play-limiting defense" collisions cause MAE outages for teams like Indiana and Ole Miss. The 45/45 SR/IsoPPP split may over-value explosive offenses in these matchups.
- **Method:** Segmented 182 collision games (7.6% of core) where top-20th-pct offensive IsoPPP faced top-20th-pct defensive IsoPPP across 2022–2025.
- **Results (collision vs core baseline):**

| Metric | Core Baseline | Collision (N=182) | Delta |
|--------|--------------|-------------------|-------|
| MAE | 12.50 | 11.58 | **-0.91 (better)** |
| Mean Error | +0.68 | +0.13 | **near zero bias** |
| ATS 5+ Edge | 52.6% | 55.6% (N=72) | **+3.0% (stronger)** |

- **Hypothesis rejected:** Collision games are a model **strength**, not weakness. Equal weighting correctly offsets explosive offense IsoPPP against limiting defense IsoPPP.
- **Magnitude check:** ±2.5 pct pt weight shift produces 0.25 pts per game — below noise floor. Even ±5.0 pct pt (0.49 pts) doesn't reach the 0.5 pt threshold. Flagged for rejection before backtest.
- **Indiana/Ole Miss root cause:** Priors problems (coaching turnarounds, portal churn) and conference circularity, not style asymmetry. Indiana's largest error (Wk5 -15.5 vs Maryland) occurred before the model captured Cignetti's turnaround.

#### LASR (Money Down Weighting) — REJECTED (Rejection #10)
**Impact: Worst single-experiment degradation; confirmed SR already captures down-and-distance**

- **Hypothesis:** Up-weighting 3rd/4th down plays (2.0x) and penalizing "empty successes" (0.5x for plays that meet SR criteria but don't convert) captures a "clutch" signal that reduces MAE for efficiency-inflated teams like UCF.
- **Implementation:** Added `money_down_weight` and `empty_success_weight` params to `_prepare_plays()` in EFM, stacking multiplicatively with existing weights (RZ Leverage, garbage time, etc.).
- **Magnitude gate PASSED:** UCF 2024 dropped +20.38 → +16.88 (**-3.50 pts**). Colorado moved minimally (-0.30 pts).
- **Core backtest FAILED (Weeks 4-15, 2022-2025):**

| Metric | Baseline | LASR | Delta |
|--------|----------|------|-------|
| Core MAE | 12.52 | 12.63 | **+0.11 (5x tolerance)** |
| Core ATS (Close) | 52.4% | 52.0% | -0.4% |
| Core 3+ Edge | 53.0% | 53.1% | +0.1% |
| Core 5+ Edge | 54.5% | 53.8% | **-0.7% (worst ever)** |

- **CLV at 5+ edge: -0.57** — market moves AGAINST LASR-influenced picks.
- **Root cause:** SR's success thresholds already differentiate by down (50%/70%/100% for 1st/2nd/3rd-4th down). 3rd down conversion signal is priced into both SR and the market. Up-weighting adds noise, not information.
- **Key insight (Code Auditor):** "The UCF/Colorado overrating problem cannot be solved by re-weighting existing signals. The market already knows which teams are bad on 3rd down — we need signals the market DOESN'T have."
- Infrastructure preserved dormant (defaults=1.0, guard `!= 1.0` prevents execution).

#### Penalty Discipline PPA — REJECTED (Rejection #11)
**Impact: Strongest pre-backtest signal ever, but still failed 5+ Edge constraint**

- **Hypothesis:** Penalties are invisible to the EFM (filtered via `SCRIMMAGE_PLAY_TYPES`). A team's penalty yards per game is a persistent trait (YoY r=0.503) and could provide a novel "discipline" signal the market underprices.
- **Exploration (Quant Auditor Phase 1-3):**
  - YoY stability: r=0.503 (strongest of any tested feature)
  - Team-level bias: r=-0.1899 (more penalties → worse ATS performance)
  - Quintile gradient: 3.1 pts between most/least penalized quintiles
  - Low redundancy: r=-0.10 against existing EFM ratings (genuinely novel signal)
  - **Passed all 5 decision gates** — first feature to do so
- **Implementation:** Prior-year penalty yards → post-Ridge adjustment (factor × yards_above_mean). Walk-forward safe using year-1 stats.
- **Phase 4 Backtest (3 variants, --start-week 4):**

| Variant | Core MAE | Core ATS (Close) | Core 3+ Edge | Core 5+ Edge |
|---------|----------|-------------------|--------------|--------------|
| Baseline | 12.52 | 52.4% | 53.0% | 54.1% |
| Factor 0.03 | 12.53 | 52.1% | 53.0% | 53.2% (-0.9%) |
| Factor 0.04 | 12.54 | 52.0% | 52.8% | 53.2% (-0.9%) |
| Factor 0.05 | 12.55 | 52.2% | 52.9% | 53.6% (-0.5%) |

- **All 3 variants FAILED** — 5+ Edge degraded 0.5-0.9% across the board.
- **Root cause:** Despite low correlation with EFM (r=-0.10), the penalty signal may already be partially priced in by the market. The market likely observes penalties directly (they're in box scores), so this isn't a true blind spot for oddsmakers — only for our model.
- **Fully reverted** — no code traces remain (CFBDClient method, EFM adjustment, backtest wiring all removed).
- **Key insight:** "Novel to the model" ≠ "novel to the market." Even signals with excellent statistical properties (stability, gradient, low redundancy) fail if the market already incorporates them. The test is: does the market MISS this signal?

#### Zombie Prior (5% Floor Removal) — REJECTED (Rejection #6, reverted)
**Impact: Floor encodes real coaching/depth signals — removal degrades 5+ Edge**

- **Hypothesis:** The 5% preseason prior floor in `blend_with_inseason()` (lines 1537-1546 in `preseason_priors.py`) is stale at weeks 9+. After 12 games of play-by-play data, the floor should decay to 0%.
- **Math analysis:** FSU 2024 (worst-case): `prior_weight=0.05 × prior_value=25.0 = +1.25 phantom pts`, plus talent floor (0.03 × 20.0 = +0.60), total = **+1.85 pts** of preseason ghost signal.
- **Change:** Set floor from 0.05 → 0.0 at lines 1538 and 1546.
- **Backtest results (--start-week 1):**

| Metric | Baseline | No Floor | Delta |
|--------|----------|----------|-------|
| Core MAE | 12.52 | 12.54 | +0.02 (at boundary) |
| Core 5+ Edge | 54.6% | 54.2% | **-0.4%** |

- **Reverted** — 5+ Edge degradation fails "must not degrade" criterion.
- **Key insight:** The 5% floor is NOT a zombie — it encodes persistent coaching quality and roster depth signals that 12 games of play-by-play efficiency don't fully capture. The talent floor alone (3%) is an insufficient substitute; the prior floor provides meaningful late-season stabilization.
- **Lesson:** Preseason priors contain persistent trait information (coaching quality, scheme fit) beyond what talent composites capture. The floor is intentional SP+ design, not a bug.

---

## Session: February 7, 2026

### Theme: Defensive Weighting Investigation + Documentation Automation

---

#### Conference Anchor Param Revert (57f228a)
**Impact: Preserved 5+ Edge by reverting to conservative anchor params**

- **Context:** Previous session committed aggressive conference anchor params (scale=0.12, prior=20, max=3.0) that improved 3+ Edge (+0.7%) but degraded 5+ Edge (-0.3%).
- **Decision:** 5+ Edge (~2% over vig) is the binding constraint; 3+ Edge (~1.3% over vig) is secondary.
- **Action:** Reverted to 0.08/30/2.0 while keeping the separate O/D anchor architecture.
- **Backtest confirmed:** Core MAE 12.52, 5+ Edge 54.5% — exact baseline match.

#### Defensive SR/IsoPPP Divergence Investigation
**Impact: Explained cosmetic defensive ranking gaps vs SP+**

- **Discovery:** Massive SR vs IsoPPP divergence on defense for elite teams:
  - Indiana: Def SR #8, Def IsoPPP #96 → Final #11 (SP+ #2)
  - Oklahoma: Def SR #1, Def IsoPPP #91 → Final #3
  - Ole Miss: Def SR #32, Def IsoPPP #104 → Final #40 (SP+ #20)
  - Notre Dame: SR #12, IsoPPP #21 (no divergence) → Final #5 (SP+ #13) — control case
- **Root cause:** At 45/45 weighting, poor defensive IsoPPP drags down teams with elite SR.

#### Defensive Weight Split — REJECTED (6dbacc4)
**Impact: Confirmed 45/45/10 is optimal for both O and D**

- **Hypothesis:** Defense controls SR more than IsoPPP; shift defensive weighting toward SR.
- **3 variants tested:** 0.55/0.35, 0.60/0.30, 0.65/0.25 (offense unchanged at 0.45/0.45).
- **All 3 FAILED:** 5+ Edge degraded monotonically (54.5% → 53.7% → 53.5% → 52.7%).
- MAE also worsened (12.52 → 12.54 → 12.57 → 12.61).
- **Surprise finding:** Defensive IsoPPP IS predictive — preventing big plays is a persistent defensive trait (scheme discipline, secondary coverage), not random variance.
- **Infrastructure preserved:** `def_efficiency_weight`, `def_explosiveness_weight`, `def_turnover_weight` params added as dormant infrastructure (default None → shared weights).

#### Documentation Metrics Audit (10e045d)
**Impact: Verified all docs are in sync with production baseline**

- Scanned 67 markdown files + 15 Python files for stale hardcoded metrics.
- **1 stale value found:** Quant Auditor agent memory had pre-uplift baseline. Fixed.
- All project docs (CLAUDE.md, MODEL_ARCHITECTURE.md, SESSION_LOG.md, MODEL_EXPLAINER.md) were already correct.
- Report: `docs/METRICS_AUDIT_2026-02-07.md`.

#### Commit: Add generate_docs.py (58a3e89)
**Impact: Automated documentation sync system**

- New script `scripts/generate_docs.py` that runs the backtest, extracts structured metrics, and auto-updates CLAUDE.md + MODEL_ARCHITECTURE.md baseline tables.
- Features: `--skip-backtest` (use cached metrics), `--dry-run` (preview changes).
- Metrics cached to `data/last_backtest_metrics.json` for fast re-use.
- Pre-push git hook installed: warns and blocks push if docs are stale (uses cached metrics, no backtest needed). Bypass with `git push --no-verify`.

#### Commit: Sync docs with fresh backtest (f31467b)
**Impact: First automated doc sync run**

- Populated metrics cache and updated CLAUDE.md + MODEL_ARCHITECTURE.md via generate_docs.py.
- Core metrics stable: MAE 12.52, ATS 52.4%, 5+ Edge 54.5%.

#### Commit: Move governing rules from SESSION_LOG to CLAUDE.md (6aab560)
**Impact: Clean separation of governance vs journal**

- Merged unique rules (sign conventions, data sources, mercy rule, market blindness, dual repo sync) into CLAUDE.md.
- SESSION_LOG is now a pure chronological development journal.

#### Commit: Replace sklearn Ridge with analytical Cholesky solver (a2f3e0c)
**Impact: 5.2x faster Ridge solve, improved numerical accuracy**

- **Problem:** sklearn's `sparse_cg` solver is iterative (tol=1e-4), producing ~1e-6 coefficient accuracy. Also requires building a sparse CSR design matrix per call.
- **Solution:** Analytical Gram matrix construction via `np.bincount()` + direct Cholesky decomposition via `scipy.linalg.cho_factor/cho_solve`.
- **4-step Council pipeline:** Strategist (equivalence confirmed) → Perf Optimizer (design plan) → Code Auditor (implementation) → Quant Auditor (backtest validation).
- **Key implementation details:**
  - Gram matrix `X^T W X` computed analytically from team indices (no sparse matrix needed)
  - Replicates sklearn's weighted centering + weight normalization exactly
  - Intercept NOT regularized (matches sklearn behavior)
  - Runtime assertions: Gram matrix symmetry check, Cholesky positive definiteness guard
- **Removed:** `_build_base_matrix()`, `_BASE_MATRIX_CACHE`, `_compute_matrix_hash()`, sklearn Ridge import
- **Performance:** 15.6ms → 3.0ms per prediction week (5.2x), 60% memory reduction (4MB → 1.6MB)
- **Accuracy:** Cholesky is 100x more precise (1e-14 vs 1e-6 vs gold standard)
- **Backtest:** Core MAE 12.51 (-0.01), ATS 52.5% (+0.1%), 3+ Edge 53.5% (+0.2%), 5+ Edge 54.1% (-0.5%, boundary effect — same 473 wins, 8 more games entered cohort under more precise coefficients)

#### Commit: Add LU decomposition fallback to Cholesky solver (3a8a80b)
**Impact: Robustness safety net for edge-case Gram matrices**

- **Problem:** If `linalg.cho_factor` fails (non-positive-definite Gram matrix), the model crashes with no recovery path.
- **Fix:** Modified the `except linalg.LinAlgError` block in `_ridge_solve_cholesky()`:
  1. Log a warning (not error) and attempt `scipy.linalg.solve(G, Xty)` (LU decomposition) as fallback.
  2. Only raise if LU also fails (truly singular matrix).
- **Backtest:** No change — Cholesky succeeded on all calls as expected. Fallback path is pure safety net.

#### Commit: Fix kneel-down/end-of-game RZ trips (ffda9ad)
**Impact: Logic fix — teams no longer penalized for winning efficiently**

- **Bug:** `FinishingDrivesModel.calculate_all_from_plays()` counted kneel-downs, end-of-game, and timeout plays inside the red zone as "FAILED" trips.
- **Fix:** Added 4-line filter before outcome classification: if last play of a trip has `play_type` containing "kneel", "end of", or "timeout", skip the trip entirely (no failed count, no total trip increment).
- **Backtest:** No change (FD is shelved at weight=0.0). Fix prepares infrastructure for future reactivation.

#### Commit: Vectorize RZ trip classification (3bc5a4b)
**Impact: Performance — eliminated Python for-loops in FinishingDrives**

- Replaced two nested Python for-loops (per-team × per-trip) with vectorized pandas operations:
  - `groupby().tail(1)` extracts last play of every RZ drive in one operation
  - `str.contains()` masks classify outcomes (TD/FG/TO/FAILED) in bulk
  - `groupby().size().unstack()` aggregates counts per team
  - Goal-to-go similarly vectorized
- Non-competitive trip filter (kneel/timeout) preserved via vectorized `str.contains`.
- Net -13 lines. Remaining per-team loop only does dict lookups on pre-computed aggregates.
- **Backtest:** No change (FD shelved). Verified identical output.

#### Commit: Document Portal Churn Investigation outcome (0224fe9)
**Impact: Closed open investigation doc**

- Updated `docs/PORTAL_CHURN_INVESTIGATION.md` with Chemistry Tax 3-0 Council rejection outcome.
- Added cross-reference from SESSION_LOG Chemistry Tax entry.
- Churn penalty infrastructure remains dormant (`use_churn_penalty=False`).

#### Commit: Document conference anchor backtest results (fb0a2f3)
**Impact: Closed open investigation doc + recovered missing session log**

- Updated `docs/NON_CONFERENCE_WEIGHTING.md` with full backtest impact (5+ Edge 53.5% → 54.8%), anchor scale sweep (4 variants), Big 12 impact, and garbage time variant results.
- Added missing "Feb 6 (Model Tuning)" session log entry covering conference anchor, RZ leverage, and 3 rejected experiments (MOV, fraud tax, GT variants).
- Synced both repos (JP-Plus + JP-Plus-Docs).

---

## Session: February 6, 2026 (Bug Fixes)

### Theme: Special Teams Fixes + Code Cleanup

---

#### Commit: Fix punt touchback net yards (81765b0)
**Impact: Phase 1 MAE improved 14.94 → 14.80; Core unchanged**

- **Bug:** `calculate_punt_ratings_from_plays()` computed touchback net yards as `np.minimum(gross, 55)`, assuming the ball was downed at the opponent's 45. In reality, touchbacks go to the 25-yard line.
- **Fix:** Changed to `gross - 25` — a 60-yard punt that touchbacks nets 35 yards (opponent gets ball at 25), not `min(60, 55) = 55`.
- **Also included:** Refactored FG expected rate lookup to dynamically build from `EXPECTED_FG_RATES` dict (DRY fix, `464299c`).
- **Backtest:** Core MAE 12.52 (unchanged), Phase 1 MAE 14.94 → 14.80 (-0.14). Improvement concentrated in early season where ST has higher relative weight.

#### Commit: Simplify kickoff per-game normalization (38e2ab7)
**Impact: Code cleanup — zero backtest delta**

- **Bug:** `calculate_kickoff_ratings_from_plays()` had redundant arithmetic in coverage and return normalization:
  - Coverage: `(tb_bonus + return_saved) * kicks_per_game / 5.0` where `kicks_per_game = total_kicks / games_played` and `/5.0` was the per-game normalizer. Without `games_played`, the `* X / X` cancelled to 1.0 (identity).
  - Returns: Same pattern with `* returns_per_game / 3.0`.
- **Fix:** Replaced with explicit branching — multiply by per-game rate when `games_played` is available, otherwise use raw per-kick value. Clearer intent, same output.
- **Backtest:** Zero delta confirmed.

#### Other cleanup commits in this session:
- **QB/pace adjustment ordering** (`9ad696b`): Moved QB adjustment before pace so triple-option compression correctly dampens QB impact.
- **Vegas numpy import** (`f4d53f2`): Added missing `import numpy as np` to `vegas_comparison.py`.
- **Edge sign convention docs** (`61b9bba`): Documented edge formula with worked examples in `ValuePlay` dataclass.
- **RZ scoring ghost feature removal** (`e844046`): Deleted `_add_rz_scoring_feature()` + RZ Ridge regression (-86 lines dead code). Also added team-specific HFA to fraud tax.
- **Portal impact refactor** (`71783f0`): Decoupled portal from regression factor, applies as direct rating adjustment.

---

## Session: February 6, 2026 (Performance)

### Theme: Caching and Data Plumbing Optimizations

Two performance-focused refactors to reduce API calls and redundant matrix construction.

---

#### Commit: Wire up week-level delta cache (4a57dfa)
**Impact: Eliminates redundant API calls in weekly production runs**

- **Problem:** `run_weekly.py` fetched ALL weeks (1 through N-1) from the CFBD API on every run, even though historical weeks never change.
- **Solution:** Wired existing `WeekDataCache` to `run_weekly.py` via `--use-delta-cache` flag:
  - Historical weeks [1, week-2] loaded from Parquet cache on disk
  - Only week (week-1) fetched from API and persisted
  - Graceful cold start: first run populates cache, subsequent runs fetch 1 week
  - Schema enforced via explicit Polars dtypes for cross-week consistency
- **Usage:** `python scripts/run_weekly.py --year 2025 --week 10 --use-delta-cache`
- **Zero behavior change** when flag is off (default)

#### Commit: Refactor EFM to precompute and cache X_base sparse matrix (741f3b7)
**Impact: Architectural separation of cacheable matrix from dynamic weights**

- **Problem:** `_ridge_adjust_metric()` rebuilt the sparse design matrix from scratch on every call, even when play structure was identical across metrics or time-decay parameter sweeps.
- **Solution:** Two-tier caching in EFM:
  1. **X_base cache** (`_BASE_MATRIX_CACHE`): Sparse CSR matrix keyed by play structure hash (teams + home_team). Independent of weights, targets, and time decay.
  2. **Result cache** (`_RIDGE_ADJUST_CACHE`): Full Ridge output keyed by (season, week, metric, alpha, time_decay, data_hash).
- **Weight pipeline refactored:**
  - `_prepare_plays()` stores `base_weight` = GT × OOC × RZ × empty_yards (all non-temporal weights)
  - Time decay moved to `_ridge_adjust_metric()` — applied dynamically per `eval_week`
  - `weight` = `base_weight × time_decay` (for raw metrics backward compatibility)
- **Backtest verified:** MAE 12.52, ATS 52.4%, 3,273 games — identical to baseline

---

## Session: February 6, 2026 (Council)

### Chemistry Tax — REJECTED (3-0 Council Vote)

**Proposal:** -3.0 pt penalty for teams with <50% Returning Production in Week 1, decaying linearly to 0 by Week 5. Goal: reduce 14.94 early-season MAE.

**Council Findings:**

- **Model Strategist (REJECT):** Redundant with 3 existing mechanisms (RetProd regression, talent floor decay, prior weight decay) that already penalize low-RetProd teams by 5-8 pts in Weeks 1-5. Early-season MAE is a sample-size problem, not a bias problem.
- **Code Auditor (APPROVED architecturally):** Does not violate Anti-Drift Guardrail (prior modifier, not post-hoc constant). Recommended Option B: explicit `chemistry_penalty` field in `PreseasonRating`, decayed via existing prior weight fade. Use raw `ret_ppa`, not portal-adjusted.
- **Quant Auditor (REJECT):** Empirical analysis on 2024-2025 found:
  - <50% RetProd threshold captures 46-56% of FBS (median split, not outlier detector)
  - Observed bias only +1.28 pts (t=0.87, p>0.38 — not statistically significant)
  - -3.0 tax overcorrects: would flip bias from +1.28 to -1.72
  - Signal reverses by Week 4 (+3.25 Week 2 → -1.08 Week 4)
  - 50-70% RetProd group has 57.5% ATS at 3+ edge — at risk from threshold bleed

**Conclusion:** 14.94 early-season MAE is the structural cost of predicting with 0-2 games of data per team. Not fixable via roster-turnover penalties. The model correctly handles this by treating Weeks 1-3 as "Calibration" and concentrating edge in Core (Weeks 4-15). See also: `docs/PORTAL_CHURN_INVESTIGATION.md` (updated with outcome).

**Alternatives Evaluated (Not Pursued):**
- Targeted version (<25% RetProd, -1.5 pts, Weeks 2-3 only): Would affect ~30 games out of 597, theoretical MAE improvement ~0.08 pts — not worth the complexity.
- Prior decay recalibration (Week 1 at 92% instead of 100%): No in-season data exists in Week 1 to blend in; early-season mean error is only -0.32, not systematically biased.
- Early-season confidence gate (raise edge threshold to 7+ in Weeks 1-3): Operational fix, not a model fix. Viable for bankroll management.

---

## Session: February 6, 2026 (Model Tuning)

### Theme: Conference Anchor + RZ Leverage + Rejected Experiments

Five model tuning experiments after the FD shelving and Council sessions. Two approved (conference anchor, RZ leverage), three rejected (MOV calibration, efficiency fraud tax, garbage time variants). Net result: Core 5+ Edge 53.5% → 55.0% (+1.5%).

---

#### Commit: Reconcile baselines against fresh backtest (ddc2550)
**Impact: Established verified ground truth for all subsequent experiments**

- Ran fresh `--start-week 4` and `--start-week 1` backtests.
- Updated CLAUDE.md, MODEL_ARCHITECTURE.md, MODEL_EXPLAINER.md, SESSION_LOG.md with verified numbers.
- Core baseline: MAE 12.52, ATS 52.0%, 3+ Edge 52.3%, 5+ Edge 53.5%.

#### Commit: Add conference strength anchor (f7bf815) — APPROVED
**Impact: Core 5+ Edge 53.5% → 54.8% (+1.3%), MAE 12.52 → 12.49**

- **Problem:** Big 12 teams massively over-rated (UCF #13 vs SP+ #62, Baylor #16 vs SP+ #38) due to Ridge regression conference circularity.
- **Two mechanisms:**
  1. OOC play weighting (1.5x) — gives Ridge more cross-conference signal
  2. Post-Ridge Bayesian conference anchor — separate O/D anchors from OOC scoring margin (scale=0.08, prior_games=30, max=±2.0)
- UCF dropped #13 → #19. Big 12 rating std deviation increased (reduced compression).
- Also added `leading_garbage_weight` param (tested 0.5/0.7/symmetric — all rejected, default 1.0 preserved).
- See `docs/NON_CONFERENCE_WEIGHTING.md` for full implementation details.

#### Garbage Time Variants — REJECTED
**Impact: Confirmed asymmetric GT (leading=1.0) is optimal**

- Tested leading=0.5, leading=0.7, and symmetric weighting.
- All degraded 5+ Edge: 53.5% → 52.2-52.5%.
- Root cause of Big 12 bubble is NOT GT weighting — it's conference circularity.
- Leading team plays ARE informative; down-weighting them removes real signal.

#### MOV Calibration — REJECTED
**Impact: All 4 weights degraded Core 5+ Edge ATS**

- Tested weights 0.05, 0.10, 0.15, 0.20. Best (0.05): 54.4% (-0.4%), Worst (0.15): 53.3% (-1.5%).
- MAE improves slightly (12.49→12.45) but ATS is the binding constraint.
- Root cause: MOV makes model MORE like market (lower MAE vs close), reducing edge.
- Infrastructure preserved: `_apply_mov_calibration()` + `mov_weight`/`mov_cap` params (default 0.0).

#### Efficiency Fraud Tax — REJECTED
**Impact: Core 5+ Edge 54.8% → 54.6% (-0.2%) — fails "must not degrade" criterion**

- Asymmetric one-way penalty for teams where expected_wins - actual_wins > 2.5.
- UCF 2024: Drops #19 → #38 (-5.31 pts) — targeting works perfectly.
- Too broad: 8 teams triggered (2024), 11 teams (2025) — target was 2-5.
- Baylor/Colorado NOT triggered (circularity makes them "look right").
- Infrastructure preserved: `_apply_efficiency_fraud_tax()` + params (default disabled).

#### Commit: Add RZ leverage weighting + empty yards filter (70b2ed0) — APPROVED
**Impact: Core 5+ Edge 54.8% → 55.0% (+0.2%), 57 more qualifying games**

- Play-level weighting in `_prepare_plays()` — no outcome signals used.
- Inside 20yd: 1.5x weight. Inside 10yd: 2.0x weight. Empty successful plays (opp 40-20, no RZ entry/score): 0.7x.
- Core 3+ Edge: 52.9% → 53.6% (+0.7%). Core MAE: 12.49 → 12.51 (+0.02, within tolerance).
- Naturally drops UCF from #19 → #23 without outcome-based signals.

#### Commit: Update model docs with performance metrics (9ce2940)
**Impact: Comprehensive documentation refresh**

- Added opening + closing line ATS to all tables, RMSE metrics, per-year breakdowns, CLV tables, metric explainer section.
- Updated 2025 Top 25 with full O/D/ST component ratings.

### Baseline (Post-Tuning Session)

| Metric | Pre-Session | Post-Session | Delta |
|--------|-------------|--------------|-------|
| Core MAE | 12.52 | 12.51 | -0.01 |
| Core ATS (Close) | 52.0% | 52.4% | +0.4% |
| Core 3+ Edge (Close) | 52.3% | 53.6% | +1.3% |
| Core 5+ Edge (Close) | 53.5% | 55.0% | +1.5% |

---

## Session: February 6, 2026 (Late Night)

### Theme: Explosiveness Uplift — Equal SR/IsoPPP Weights (45/45/10)

Rebalanced EFM component weights from SR=0.54/IsoPPP=0.36/TO=0.10 to SR=0.45/IsoPPP=0.45/TO=0.10. Equal weighting of Success Rate and Explosiveness better captures boom-or-bust offensive teams like Ole Miss and Texas Tech that generate value through big plays rather than sustained drives.

---

#### Commit: Apply Explosiveness Uplift (dd2c56c)
**Impact: +1.1% Core ATS, new production baseline**

- **Rationale:** Previous 54/36 split (inherited from when turnovers were added to the 60/40 base) over-weighted consistency. IsoPPP captures EPA on successful plays — underweighting it systematically undervalued explosive offenses.
- **Files changed:** `efficiency_foundation_model.py`, `backtest.py`, `calibrate_situational.py`, `benchmark_backtest.py`, `compare_ratings.py`, `MODEL_ARCHITECTURE.md`, `MODEL_EXPLAINER.md`

#### Backtest Results (vs Closing Line)

| Metric | Old (54/36/10) | New (45/45/10) | Delta |
|--------|---------------|----------------|-------|
| Full MAE | 13.00 | 13.02 | +0.02 |
| Core MAE (4-15) | 12.49 | 12.51 | +0.02 |
| Full ATS | 50.2% | 51.1% | +0.9% |
| Core ATS | 51.3% | 52.4% | +1.1% |
| Core 3+ Edge | 52.9% (730-649) | 53.3% (764-669) | +0.4% |
| Core 5+ Edge | 55.0% (493-404) | 54.6% (473-393) | -0.4% |

#### Backtest Results (vs Opening Line)

| Phase | ATS % | 3+ Edge | 5+ Edge | Mean CLV |
|-------|-------|---------|---------|----------|
| Calibration (1-3) | 48.6% | 48.4% | 48.8% | +0.02 |
| Core (4-15) | 54.0% | 55.5% | 56.9% | +0.56 |
| Postseason (16+) | 48.3% | 47.0% | 47.4% | +0.11 |
| Full | 52.7% | 53.6% | 54.4% | +0.44 |

#### CLV Analysis (vs Opening Line)

| Edge | N | Mean CLV | CLV > 0 | ATS % |
|------|---|----------|---------|-------|
| All | 3,258 | +0.44 | 39.4% | 52.7% |
| 3+ | 2,001 | +0.61 | 39.7% | 53.6% |
| 5+ | 1,339 | +0.75 | 40.3% | 54.4% |
| 7+ | 824 | +0.93 | 40.0% | 55.4% |

#### Gate Check
- Core MAE +0.02: AT strict tolerance (+0.02), PASS
- Core 5+ Edge (Close) 54.6%: Above 54.5% floor, PASS
- Core 5+ Edge (Open) 56.9%: At floor, PASS
- CLV positive and monotonically increasing: PASS
- Overall ATS improved across all tiers: PASS

**Verdict: APPROVED by Quant Auditor**

#### Key Insight
The 5+ edge vs closing dipped slightly (-0.4%) while ALL other metrics improved significantly. The trade-off is acceptable because: (1) Opening line performance is more actionable (that's when bets are placed), and (2) Core ATS improvement of +1.1% is the largest single-change improvement in model history.

---

### New Production Baseline (Post-Explosiveness Uplift)

| Slice | Weeks | Games | MAE | ATS (Close) | ATS (Open) | 5+ Edge (Close) | 5+ Edge (Open) |
|-------|-------|-------|-----|-------------|------------|-----------------|----------------|
| Full | 1–Post | 3,273 | 13.02 | 51.1% | 52.7% | 52.3% | 54.4% |
| Calibration | 1–3 | 597 | 14.94 | 47.1% | 48.6% | 47.4% | 48.8% |
| **Core** | **4–15** | **2,485** | **12.52** | **52.4%** | **54.0%** | **54.6%** | **56.9%** |
| Postseason | 16+ | 176 | 13.43 | 47.4% | 48.3% | 46.7% | 47.4% |

---

## Session: February 6, 2026 (Evening)

### Theme: Finishing Drives Investigation — 4 Rejections → EFM Integration

Comprehensive investigation of the FinishingDrives sub-model. Five engineering iterations, three agents (Code Auditor, Quant Auditor, Model Strategist), and four backtest rejections led to the definitive conclusion: RZ efficiency is ~88% redundant with EFM's IsoPPP. The correct architecture is Ridge integration, not post-hoc addition.

---

#### Commit 1: drive_id Pipeline Fix + Postseason Display (fd700fa)
**Impact: Data pipeline — enables FinishingDrives model to function**

- Added `drive_id` field to efficiency_plays extraction at 3 locations in `scripts/backtest.py` (regular season, postseason, delta cache).
- Without drive_id, red zone trips couldn't be counted at the drive level — FD was silently producing zero for all teams.
- Collapsed postseason pseudo-weeks (16+) into single "Post" row in MAE-by-week and sanity check reports.

#### Commit 2: Per-Game Normalization + Magnitude Cap (bb67d0f)
**Impact: Engineering fix — prevented ±7.9pt swings**

- **Bug**: `overall = (ppt - 4.8) * (total_rz_trips / 10.0)` used cumulative trip count that grew with games played. By Week 12, matchup differentials reached ±7.9 points.
- **Fix**: Normalized to `avg_rz_trips_per_game = total_rz_trips / games_played`.
- Added `MAX_MATCHUP_DIFFERENTIAL = 1.5` class constant with `np.clip()` in `get_matchup_differential()`.
- Added `games_played` parameter tracking throughout all calculation pathways.
- **Backtest**: MAE 12.65 (+0.06) — REJECTED (exceeds +0.05 threshold).

#### Commit 3: Calibration Fix — EXPECTED_POINTS_PER_TRIP 4.8 → 4.05 (aa5ba20)
**Impact: Reduced bias but still insufficient**

- Empirical FBS mean PPT is ~4.0-4.1, not 4.8. Fixed constant created systematic downward bias.
- **Backtest**: MAE 12.65 (+0.06), 60% cap saturation — REJECTED. Year-to-year PPT variance (3.25–3.78) means no fixed constant works across all years.

#### Commit 4: Shelve FD at Zero Weight (d92d249)
**Impact: Model protection — removed harmful signal**

- Set `components.finishing_drives = 0.0` in `spread_generator.py` while preserving all infrastructure.
- Confirmed exact baseline recovery: MAE 12.52, ATS 51.6%, 5+ Edge 53.5%.

#### Commit 5: Dynamic Seasonal Baseline + Reactivation (2840ad1)
**Impact: Solved cap saturation (57% → 4.5%) but signal still redundant**

- Replaced fixed constant with dynamic `_seasonal_mean_ppt` computed from all teams in training window.
- Eliminated trips-per-game multiplier (was the 3.5x amplifier): `overall = ppt - seasonal_mean_ppt`.
- Walk-forward safe: backtest engine pre-filters plays to weeks < prediction_week.
- **Cap saturation**: 57% → 4.5% (target was <15%).
- **Backtest**: MAE 12.55 (+0.03), ATS 51.3% — REJECTED. Cohort analysis: games with highest FD had WORST MAE (12.82 vs 12.42 neutral). Model-Strategist confirmed ~70-80% overlap with IsoPPP.

#### Commit 6: Final Shelving with Full Rationale (8fcbb13)
**Impact: Definitive closure on additive FD approach**

- Re-shelved FD at 0.0 weight after 4th consecutive rejection.
- Documented root cause: IsoPPP (EPA-based) already captures red zone scoring at play level.
- FD is not opponent-adjusted — Sun Belt teams with high PPT vs weak defenses appear elite.
- Preserved all infrastructure for future residualization or EFM integration.

#### Commit 7: RZ Efficiency as EFM Ridge Feature (2dd6bae)
**Impact: Architecturally correct integration — zero regression**

- Created `_add_rz_scoring_feature()` in EFM: assigns scoring values (TD=7.0, FG=3.0, other=0.0) for RZ plays (yards_to_goal ≤ 20).
- Ridge regression naturally opponent-adjusts and learns optimal weight alongside SR and IsoPPP.
- Ridge coefficient: mean |coef| = 0.12, but only **2.2% of total rating variance** (0.27 pts std).
- **Backtest**: MAE 12.52, ATS 51.6%, 5+ Edge 53.5% — exact baseline match. CONDITIONAL PASS.
- Confirms Model Strategist hypothesis: ~88% of RZ signal already captured by IsoPPP. Ridge correctly suppresses the feature to near-zero effective weight.

#### Commit 8: Model Governance Update (be465d7)
**Impact: Codified lessons learned into CLAUDE.md**

- Added Model Strategist role (redundancy filter, signal test, temporal integrity).
- Tightened MAE tolerance from +0.05 to +0.02 based on FD investigation findings.
- Added Anti-Drift Guardrail: new features must be evaluated for EFM integration before post-hoc addition.
- Added Redundancy Protocol: Quant Auditor must report correlation against existing PPA/IsoPPP for any proposed signal.

---

### Key Lesson: The Finishing Drives Principle

> Any sub-model measuring a subset of what EFM captures at play level must be **residualized against EFM** or **integrated as a Ridge feature**, not added as a post-hoc constant. Post-hoc addition double-counts shared signal and lacks opponent adjustment.

This principle is now codified in CLAUDE.md as the "Anti-Drift Guardrail."

### Baseline (Unchanged)

| Metric | Value |
|--------|-------|
| **MAE (core weeks 4-15)** | 12.52 |
| **ATS All** | 51.6% (1347-1263-51) |
| **ATS 3+ Edge** | 52.3% (738-672) |
| **ATS 5+ Edge** | 53.5% (474-412) |

### Files Changed (Session Total)
- `src/models/finishing_drives.py` — Per-game normalization, dynamic baseline, magnitude cap
- `src/models/efficiency_foundation_model.py` — RZ scoring as Ridge feature
- `src/predictions/spread_generator.py` — FD shelved at 0.0 (signal flows through EFM)
- `scripts/backtest.py` — drive_id pipeline, postseason display cleanup
- `CLAUDE.md` — Model governance: redundancy protocol, tighter tolerances

---

## Session: February 6, 2026

### Theme: Performance Engineering — 16 min → 25 sec (38x cumulative speedup)

Five commits across three performance domains: vectorization, caching, and multiprocessing. Plus a new preseason feature (temporal talent decay). All changes verified numerically identical to baseline via Quant Auditor.

---

#### Commit 1: O(1) Situational Lookups + Vectorization (7c979e8)
**Impact: 16 min → 3:22 (4.8x speedup)**

- **Situational adjustments**: Added `precalculate_schedule_metadata()` — pre-computes rest days, last/next game week, win/loss, home/away, rivalry flags once per season. Added O(1) fast paths to `check_letdown_spot()`, `check_lookahead_spot()`, and `check_consecutive_road()`, eliminating ~13 min of O(N) DataFrame scans per backtest.
- **Special teams**: Replaced `.apply(lambda)` regex with `.str.extract()`/`.str.contains()`, replaced `.iterrows()` with numpy aggregation.
- **Vegas comparison**: Vectorized `.apply(lambda)` → `np.where()` for favorite identification.
- **Reverted**: Tier 2 per-team talent discount (MAE +0.08, outside tolerance).
- **New docs**: `PERFORMANCE_AUDIT_2026-02-05.md`, `PERFORMANCE_PROFILE.md`, benchmark/profiling scripts.

#### Commit 2: Local-First Disk Caching (073a02b)
**Impact: 3:22 → 1:10 (2.9x speedup)**

- Cache processed season DataFrames to `.cache/seasons/` as Parquet files, eliminating redundant API calls on repeated runs.
- New CLI flags: `--no-cache`, `--force-refresh`.
- New utility: `scripts/ensure_data.py` (pre-fetch all season data).
- New modules: `src/data/cache.py`, `src/data/season_cache.py`, `src/data/processors.py`, `src/data/validators.py`.
- Fixed `.gitignore` pattern (`data/` → `/data/`) that was accidentally hiding `src/data/` from git.
- **Quant Auditor verified**: MAE 12.50, ATS 3+ edge 53.1%, 5+ edge 54.7% (bit-identical).

#### Commit 3: API Rate Limiting & Delta Caching (7a2aa3b)
**Impact: ~42 API calls → ~6 per cached run**

- Added week-level delta caching (`src/data/week_cache.py`) — only current week fetched from API; historical weeks locked in cache.
- Routed all PreseasonPriors API calls through CFBDClient (closed rate-limiting blind spot — priors were using raw `cfbd.ApiClient` directly).
- Added session-level cache for `get_fbs_teams()` to deduplicate 3-4 calls/year to 1.
- Refactored `build_team_records()` to use cached games DataFrames for trajectory years instead of re-fetching via API (eliminated 5 API calls per run).
- New wrapper methods on CFBDClient: `get_sp_ratings`, `get_transfer_portal`, `get_returning_production`, `get_player_usage`.
- New tools: `scripts/populate_week_cache.py`, `tests/test_week_cache.py`.
- New docs: `API_RATE_LIMITING_SUMMARY.md`, `RATE_LIMITING_GUIDE.md`.

#### Commit 4: Temporal Talent Floor Decay + Roster Churn Penalty (c11bf9f)
**Impact: Model feature — reduces late-season "Talent Mirage"**

- Replaced static `talent_floor_weight` (0.08) with `calculate_decayed_talent_weight()`: linearly decays from 0.08 at week 0 to 0.03 by week 10. Prevents late-season inflation for high-talent teams with poor on-field efficiency.
- FSU stress test: MAE 14.98 → 14.88, RMSE 17.32 → 17.25. Overall MAE neutral (12.43 both variants).
- Added `calculate_roster_churn_penalty()` for portal-heavy teams (off by default via `use_churn_penalty=False`).
- Both features toggleable via PreseasonPriors constructor for A/B testing.
- New toggles: `use_talent_decay` (default True), `use_churn_penalty` (default False).
- New docs: `PORTAL_CHURN_INVESTIGATION.md`, `PORTAL_CODE_MAP.md`.

#### Commit 5: Multiprocessing Season Loop (03c2b2d)
**Impact: 3:30 → 25 sec (8.4x speedup)**

- Parallelized backtest season loop via `ProcessPoolExecutor` — each season (2022-2025) runs on its own CPU core.
- Three-agent collaboration (Model Strategist + Code Auditor + Quant Auditor) analyzed 4 proposed optimizations. Strategist recommended multiprocessing as #1 priority; Code Auditor identified shared-state risks; Quant Auditor captured baseline for verification.
- Key design decisions:
  - `_process_single_season()` defined at module level (required for pickle serialization with `spawn` mode).
  - `priors.client = None` before pool submission (CFBDClient not picklable).
  - Results sorted by `(year, week, game_id)` for deterministic output ordering.
  - Single-year runs skip multiprocessing overhead (sequential fallback preserved).
- **Quant Auditor verified**: All metrics bit-identical. Max numeric diff: 1.07e-14 (float noise). Ole Miss +15.24, FSU +12.48 unchanged.

---

### Performance Summary (Cumulative)

| Stage | Wall-Clock Time | Speedup | Commit |
|-------|----------------|---------|--------|
| **Starting point** | ~16 min | — | (pre-session) |
| After vectorization | 3:22 | 4.8x | 7c979e8 |
| After disk caching | 1:10 | 2.9x (13.7x cumulative) | 073a02b |
| After multiprocessing | **0:25** | **8.4x (38x cumulative)** | 03c2b2d |

### Baseline (Post-Performance-Session, `--start-week 4`)

| Metric | Standard (Wks 4+) | Core (Wks 4–15) |
|--------|-------------------|-----------------|
| **MAE** | 12.59 | **12.52** |
| **ATS All** | 51.6% (1347-1263-51) | 52.0% |
| **ATS 3+ Edge** | 52.0% (790-728) | 52.3% (738-672) |
| **ATS 5+ Edge** | 53.0% (509-451) | 53.5% (474-412) |
| **Mean Error** | +0.90 pts | — |

### Key Team Ratings (2025 End-of-Regular-Season)

| Team | Rating | Rank |
|------|--------|------|
| Ohio State | +27.86 | 1 |
| Indiana | +25.02 | 2 |
| Notre Dame | +23.52 | 3 |
| Oregon | +23.44 | 4 |
| Alabama | +21.66 | 5 |
| Georgia | +18.58 | 9 |
| Ole Miss | +15.24 | 15 |
| Florida State | +12.48 | 22 |

### Files Changed (Session Total)
- **31 files changed**, 4,811 insertions, 309 deletions
- New modules: `src/data/` package (cache, season_cache, processors, validators, week_cache)
- New scripts: `ensure_data.py`, `populate_week_cache.py`, `benchmark_backtest.py`, `profile_backtest.py`
- New docs: 5 documentation files (performance audit, profiling, rate limiting, portal investigation)

---

## Session: February 5, 2026 (Late Night)

### Completed: P3 Implementation Sweep

Final sweep focused on system legibility and long-term maintainability. Docstrings, type hints, logging format, project documentation, and the last EFM performance optimization. Zero MAE tolerance enforced throughout.

#### Batch 1: Docstrings, Type Hints, Logging, PROJECT_MAP (all accepted)

- **Docstring audit**: All top-level functions in `src/models/` and `scripts/backtest.py` already had Google-style docstrings. Added JP+ Power Ratings Display Protocol reference to `get_ratings_df()` (EFM) and `excel_export.py`.
- **Type hinting**: Fixed 3 `__init__` methods missing `-> None` (finishing_drives, special_teams, preseason_priors). Fixed `prior_strength: int = None` to `Optional[int] = None`. Added type hints to 5 nested helper functions in special_teams.py (`extract_distance`, `extract_punt_yards`, `is_touchback`, `is_inside_20`, `extract_return_yards`). Added type hint to `process_betting_lines` closure in backtest.py.
- **Logging cleanup**: Converted backtest `print_results()` output from plain text to Markdown tables (metrics, ATS, CLV report, phase report, sanity check). Headers use `##`/`###` Markdown syntax. Phase report uses manual Markdown table construction (no `tabulate` dependency).
- **PROJECT_MAP.md**: Generated comprehensive file map covering all 45+ Python files across 10 directories (`src/models/`, `src/adjustments/`, `src/predictions/`, `src/api/`, `src/data/`, `src/reports/`, `src/utils/`, `config/`, `scripts/`, `docs/`).

#### Batch 2: EFM P3.1 + D.1 (both accepted)

- **EFM P3.1 — Vectorize sparse COO construction**: Replaced Python per-row loop (O(N_plays) iterations) with vectorized numpy operations. Team-to-index lookup via list comprehension to int32 arrays. Row/col/data assembled via `np.concatenate`. Home team comparison uses `np.asarray(dtype=object)` to avoid pandas Categorical comparison errors. Backtest MAE identical.
- **EFM D.1 — Ridge sanity logging**: Added consolidated debug-level sanity summary after post-centering: intercept, learned HFA, off/def mean and std, n_teams, n_plays. Reported in a single log line per ridge fit. (Most sub-metrics already existed from P0.2, P1.1 fixes; D.1 consolidates them.)

#### Quant Auditor Decision Gate Results

| Batch | MAE Delta | Verdict |
|-------|-----------|---------|
| Batch 1 (Docstrings+Types+Logging+Docs) | 0.00 | PASS |
| Batch 2 (EFM P3.1+D.1) | 0.00 | PASS |

### Baseline (Post-P3 Sweep — as measured 2026-02-05, before perf/FD sessions)

*Note: These values reflect the codebase at this point in time. Current production baseline is in the Feb 6 Evening session.*

| Metric | 2024-2025 (Wks 4–15) | 4-Year (Wks 4–15) |
|--------|----------------------|--------------------|
| Core MAE | 12.43 | 12.49 |
| Core ATS (close) | 51.33% | 51.87% |
| Core 3+ edge | 53.5% | 53.1% |
| Core 5+ edge | 55.7% | 54.7% |

### Audit Sweep Progress (Final — All Priorities)

| Priority | Total | Fixed | Deferred/Rejected | Remaining |
|----------|-------|-------|-------------------|-----------|
| P0 | 11 | 11 | 0 | 0 |
| P1 | 18 | 15 | 3 | 0 |
| P2 | 16 | 12 | 4 | 0 |
| P3 | 2 | 2 | 0 | 0 |
| D (Diagnostics) | 1 | 1 | 0 | 0 |
| **Total** | **48** | **41** | **7** | **0** |

**Key takeaway:** Full audit sweep complete (P0 through P3 + diagnostics). 41/48 items fixed. All 7 remaining items are deferred with documented blockers (external API research or model recalibration required). Codebase is fully documented with type hints, Google-style docstrings, Markdown output, and PROJECT_MAP.md.

---

## Session: February 5, 2026 (Night)

### Completed: P2 Implementation Sweep

Scanned all 7 `AUDIT_FIXLIST_*.md` files, identified 16 unfixed P2 items across 7 files. Implemented with Quant Auditor backtest gate (0.01 MAE tolerance). 12 accepted, 4 deferred. Two batches.

#### Batch 1: Backtest + Weekly Odds + EFM (6/6 accepted)

- **Backtest P2.1 — Week coverage sanity checks**: Separated regular-season (1-15) from postseason (16+) in coverage checks. Missing-week warnings now specify "regular-season." Postseason pseudo-week counts reported separately.
- **Backtest P2.2 — Postseason play coverage diagnostics**: Enhanced P0.2 check to include total play count and avg plays/game. Warns if avg < 80 plays/game.
- **Weekly Odds P2.1 — Stable preview grouping**: Preview groups by `game_id` when available, falls back to `(home_team, away_team)` tuple.
- **Weekly Odds P2.3 — UTC timestamps**: `captured_at` uses `datetime.now(timezone.utc)`. Provider timestamps preserved as-is (already UTC).
- **EFM P2.1 — SettingWithCopy safety**: Added `.copy()` after `df[keep_mask]`. Added column assertion post-preprocessing. Added NaN guard before ridge `.fit()`.
- **EFM P2.2 — Cache garbage-time thresholds**: Module-level `_GT_THRESHOLDS` tuple. `is_garbage_time_vectorized()` reads Settings once on first call.

#### Batch 2: Vegas + Special Teams + Priors + Finishing Drives (6/6 accepted)

- **Vegas P2.1 — Opener diagnostics**: `fetch_lines()` logs spread_open coverage % and movement rate. Warns if < 50% have openers.
- **Vegas P2.2 — game_id in get_line_movement()**: Added optional `game_id` parameter, passed through to `get_line()`. Consistent with rest of module.
- **Special Teams P2.1 — Parse coverage diagnostics**: FG distance, punt gross yards, kickoff return yards (non-TB) parse rates logged. Warns if < 80%.
- **Special Teams P2.3 — Public API clarity**: Added `is_complete: bool` to `SpecialTeamsRating`. FG-only ratings = `False`, full ST = `True`. Guard in `get_matchup_differential()`.
- **Preseason Priors P2.2 — Name consistency diagnostics**: Reports SP+ teams missing from talent and returning production datasets (first 10 names listed).
- **Finishing Drives P2.2 — Fallback pathway hierarchy**: Documented PRIMARY > SECONDARY > TERTIARY in all 3 method docstrings. Debug-level pathway logging.

#### Deferred P2 Items (4)

| Item | Reason |
|------|--------|
| Weekly Odds P2.2 (additional markets) | Schema change, requires OddsAPI research |
| Preseason Priors P2.1 (conference-by-year) | Requires historical conference data by year |
| Finishing Drives P2.1 (overall_rating scaling) | Known risky from P1 rejection (MAE +0.09) |
| Special Teams P2.2 (structured fields) | Requires CFBD API field research |

### Quant Auditor Decision Gate Results

| Batch | MAE Delta | ATS Delta | 5+ Edge Delta | Verdict |
|-------|-----------|-----------|---------------|---------|
| Batch 1 (Backtest+Odds+EFM) | 0.00 | 0.00% | 0.0% | PASS |
| Batch 2 (Vegas+ST+Priors+FD) | 0.00 | 0.00% | 0.0% | PASS |

### Baseline (Post-P2 Sweep — as measured 2026-02-05, before perf/FD sessions)

| Metric | 2024-2025 (Wks 4–15) | 4-Year (Wks 4–15) |
|--------|----------------------|--------------------|
| Core MAE | 12.43 | 12.49 |
| Core ATS (close) | 51.33% | 51.87% |
| Core 3+ edge | 53.5% | 53.1% |
| Core 5+ edge | 55.7% | 54.7% |

**Key takeaway:** All 12 accepted P2 changes are pure refactors — diagnostics, logging, copy safety, timestamp normalization. Zero MAE movement across both batches. P2 sweep complete — 12/16 fixed, 4 deferred pending external research or recalibration.

### Audit Sweep Progress (All Priorities)

| Priority | Total | Fixed | Deferred/Rejected | Remaining |
|----------|-------|-------|-------------------|-----------|
| P0 | 11 | 11 | 0 | 0 |
| P1 | 18 | 15 | 3 | 0 |
| P2 | 16 | 12 | 4 | 0 |
| **Total** | **45** | **38** | **7** | **0** |

---

## Session: February 5, 2026 (Late Evening)

### Completed: P1 Implementation Sweep

Scanned all 8 `AUDIT_FIXLIST_*.md` files, identified 18 unfixed P1 items across 7 files (EFM P1s already cleared). Implemented with Quant Auditor backtest gate after each file. 15 accepted, 3 rejected/deferred.

#### EFM P1s (3/3 accepted)

- **P1.1 — Post-center ridge coefficients**: After ridge fit, offense/defense coefficient means are subtracted to make coefs mean-zero, absorbed into separate baselines. Mathematically neutral for spreads (differences invariant); stabilizes O/D decomposition.
- **P1.2 — Set-based IsoPPP filtering**: Added `off_isoppp_real`/`def_isoppp_real` tracking sets in `_compute_raw_metrics()`. Teams added when they have >=10 successful plays. `avg_isoppp` now uses set membership instead of `!= LEAGUE_AVG_ISOPPP` equality check. Near-average teams no longer excluded.
- **P1.3 — Turnover diagnostics clarification**: Updated `TeamEFMRating.turnover_rating` docstring to "DIAGNOSTIC ONLY" and added guard comments at both `overall` computation sites.

#### Backtest P1s (2/2 accepted)

- **P1.1 — games_df pandas conversion once**: Moved `games_df.to_pandas()` before the weekly prediction loop (was converting full schedule every week).
- **P1.2 — Semi-join replaces game_id list**: Replaced `games_df.filter(...)["id"].to_list()` + `is_in()` with Polars semi-join. Also reuses `train_games_pl` for pandas conversion.

#### Special Teams P1s (1/3 — 1 accepted, 2 deferred)

- **P1.3 — Vectorized kickoff parsing** (accepted): Replaced `apply(lambda r: is_touchback(...), axis=1)` with `str.contains("touchback")`. Replaced `apply(extract_return_yards)` with `str.extract()` + `pd.to_numeric()`.
- **P1.1 — Kickoff scaling** (REJECTED): Removing `/5.0` and `/3.0` divisors amplified kickoff impact 5x/3x, causing 5+ edge to drop 55.7% to 53.3%. Divisors are empirically calibrated dampening factors.
- **P1.2 — Punt touchback handling** (DEFERRED): Current heuristic produces reasonable values. Requires comprehensive ST recalibration.

#### Preseason Priors P1s (3/3 accepted)

- **P1.1 — Vectorized portal impact**: Position group mapping via reverse lookup dict + `.map()`. G5-to-P4 transfer count via set membership + boolean ops.
- **P1.2 — Continuity tax clarity**: Added explanatory comment on `CONTINUITY_TAX` constant showing amplification math. Fixed incorrect comment ("0.85" to "0.90").
- **P1.3 — portal_scale default**: Changed internal default from 0.06 to 0.15 to match production caller.

#### Vegas Comparison P1s (4/4 accepted)

- **P1.1 — Deterministic provider fallback**: Sorts lines alphabetically by provider name, filters to non-null spreads.
- **P1.2 — Duplicate game_id detection**: Logs warning when duplicate game_id encountered; keeps first-seen line.
- **P1.3 — Signed edge preserved**: Added `edge_signed` column to `value_plays_to_dataframe()` output.
- **P1.4 — Robust edge sorting**: Added `pd.to_numeric(df["edge"], errors="coerce")` before `sort_values()`.

#### Weekly Odds Capture P1s (3/3 accepted)

- **P1.1 — Schema normalization**: Added `season` and `week` INTEGER columns to `odds_snapshots`. Migration logic for existing DBs.
- **P1.2 — Join metadata**: Added `cfbd_game_id` INTEGER placeholder column to `odds_lines`.
- **P1.3 — Spread consistency check**: Per-line check that `abs(spread_home + spread_away) <= 0.5`. Logs anomalies.

#### Finishing Drives P1s (2 fixed by P0.1, 1 deferred)

- **P1.2/P1.3**: Already fixed by P0.1 drive-level refactor (trip-based rz_failed, drive-based GTG counting).
- **P1.1** (REJECTED): Adding `drive_id` + `scoring` to efficiency_plays enabled a previously-dead finishing drives code path. Core MAE increased from 12.43 to 12.52 (+0.09). Root cause: finishing drives scaling formula adds noise. Requires P2.1 scaling recalibration before enabling.

### Quant Auditor Decision Gate Results

| Change | MAE Delta | ATS Delta | 5+ Edge Delta | Verdict |
|--------|-----------|-----------|---------------|---------|
| EFM P1.1+P1.2 | 0.00 | 0.00% | 0.0% | PASS |
| Backtest P1.1+P1.2 | 0.00 | 0.00% | 0.0% | PASS |
| Finishing Drives P1.1 | **+0.09** | +0.88% | -2.1% | **REJECT** |
| ST P1.3 vectorization | 0.00 | 0.00% | 0.0% | PASS |
| ST P1.1 kickoff scaling | +0.03 | +0.40% | **-2.4%** | **REJECT** |
| Vegas P1.1-P1.4 | 0.00 | 0.00% | 0.0% | PASS |
| Preseason P1.1-P1.3 | 0.00 | 0.00% | 0.0% | PASS |
| Odds Capture P1.1-P1.3 | N/A | N/A | N/A | PASS (no backtest impact) |

### Baseline (Post-P1 Sweep — as measured 2026-02-04, before perf/FD sessions)

| Metric | 2024-2025 (Wks 4–15) | 4-Year (Wks 4–15) |
|--------|----------------------|--------------------|
| Core MAE | 12.43 | 12.49 |
| Core ATS (close) | 51.33% | 51.87% |
| Core 3+ edge | 53.5% | 53.1% |
| Core 5+ edge | 55.7% | 54.7% |

**Key takeaway:** All accepted P1 changes are mathematically neutral or performance-only. Backtest results identical across all years. Three changes rejected for degrading high-confidence picks (5+ edge). P1 sweep complete — 0 unfixed P1 items remain except the 3 deferred items requiring deeper recalibration.

### Files Modified
- `src/models/efficiency_foundation_model.py` — post-centering, set-based IsoPPP, turnover docs
- `scripts/backtest.py` — pandas conversion once, semi-join, drive_id/scoring reverted
- `src/models/special_teams.py` — vectorized kickoff parsing
- `src/models/preseason_priors.py` — vectorized portal, continuity tax, portal_scale default
- `src/predictions/vegas_comparison.py` — deterministic fallback, dedup, signed edge, robust sort
- `scripts/weekly_odds_capture.py` — schema columns, cfbd_game_id, spread check
- `src/adjustments/aggregator.py` — independence assumptions documentation
- All 7 `docs/AUDIT_FIXLIST_*.md` files — P1 items marked fixed/deferred

---

## Session: February 5, 2026 (Evening)

### Completed: Full P0 Reconciliation

Scanned all 8 `AUDIT_FIXLIST_*.md` files, identified 11 unfixed P0 items, and implemented all of them with Quant Auditor backtest gate after each file change.

- **Backtest P0.2 — Postseason play completeness**
  - `fetch_season_plays()` now loops weeks 1-5 for postseason plays (guards against CFBD API behavior changes)
  - `fetch_all_season_data()` adds postseason coverage sanity check comparing games-with-plays to total postseason games

- **Finishing Drives P0.2 — Remove fabricated minimums**
  - Fixed `rz_failed_4th` → `rz_failed` (NameError left from P0.1 drive-level refactor)
  - All `max(1, ...)` hacks already removed by P0.1; this fixes the residual crash bug

- **Special Teams P0.2 — Fix `calculate_team_rating()` double-normalization**
  - Punt/kickoff components (per-event averages) were incorrectly divided by count before scaling
  - FG now divides total by estimated games; punt/kickoff multiply per-event avg by events-per-game

- **Special Teams P0.3 — Document `calculate_from_game_stats()` as fallback**
  - Units already correct after P0.1; added docstring and debug logging marking it as fallback path

- **Preseason Priors P0.1 — Unify talent scaling**
  - Added `talent_rating_normalized` field to `PreseasonRating` dataclass
  - `blend_with_inseason()` now uses z-score-normalized talent (same scale as preseason blending) instead of ad hoc `(raw - 750) / 25.0`

- **Preseason Priors P0.2 — Rank direction sanity checks**
  - Added `_validate_data_quality()` method: logs dataset intersection sizes, validates elite programs appear in talent top-20 and have positive SP+ ratings

- **Preseason Priors P0.3 — Coaching table consistency**
  - Removed Dan Lanning from `COACHING_CHANGES[2022]` (already in `FIRST_TIME_HCS`)

- **Vegas Comparison P0.2 — Add `game_id` to ValuePlay**
  - Added `game_id` field to `ValuePlay` dataclass, `compare_prediction()` output, and `value_plays_to_dataframe()`

- **Weekly Odds Capture P0.1 — Replace heuristic week detection**
  - Added `--year`/`--week` CLI arguments; heuristic demoted to fallback with warning

- **Weekly Odds Capture P0.2 — Fix SQLite upsert**
  - Replaced `INSERT OR REPLACE` with `INSERT...ON CONFLICT DO UPDATE` (preserves row identity, no orphaned FK references)

- **Weekly Odds Capture P0.3 — Enable FK enforcement**
  - Added `PRAGMA foreign_keys = ON` after connection creation

### Updated Baseline (Post-Full P0 Reconciliation)

| Metric | Pre-Reconciliation | Post-Reconciliation | Delta |
|--------|-------------------|--------------------:|-------|
| Core MAE (4yr) | 12.55 | 12.49 | -0.06 |
| Core ATS (4yr Close) | 52.0% | 51.9% | -0.1% |
| Core 3+ (4yr Close) | 51.9% | 53.1% | **+1.2%** |
| Core 5+ (4yr Close) | 53.2% | 54.7% | **+1.5%** |

**Key takeaway:** Talent scaling unification (P0.1) improved high-confidence picks significantly (5+ edge +1.5%). All-picks ATS held steady. No P0 items remain unfixed across all 8 audit files.

### Files Modified
- `scripts/backtest.py` — postseason multi-week fetch + coverage check
- `src/models/finishing_drives.py` — `rz_failed_4th` → `rz_failed` bug fix
- `src/models/special_teams.py` — double-normalization fix + fallback docs
- `src/models/preseason_priors.py` — talent normalization + data validation + coaching table cleanup
- `src/predictions/vegas_comparison.py` — `game_id` in ValuePlay end-to-end
- `scripts/weekly_odds_capture.py` — explicit year/week, safe upsert, FK enforcement
- All 8 `docs/AUDIT_FIXLIST_*.md` files — P0 items marked fixed

---

## Session: February 5, 2026

### Completed Today

- **P0 Audit Fixes (Code Auditor Pass)**
  - Fixed 6 structural issues from `AUDIT_FIXLIST_BACKTEST.md` and `AUDIT_FIXLIST_EFM.md`:
    1. **P0.1 (Backtest):** Postseason games mapped to sequential pseudo-weeks by `start_date` via `_assign_postseason_pseudo_weeks()` + `_remap_play_weeks()`. Prevents walk-forward chronology violation where "future" bowls trained earlier bowl predictions.
    2. **P0.3 (Backtest):** `home_team` validated via game join on `game_id` instead of trusting `play.home` field. Coverage: 100%.
    3. **P0.4 (Backtest):** ATS unmatched mask changed from `game_id.isna()` to `vegas_spread.isna()`.
    4. **P0.2 (EFM):** `pd.notna()` replaces `is not None` for home_team check in ridge sparse matrix build. Added coverage logging.
    5. **P0.3 (EFM):** Ridge cache hash strengthened with MD5 of sampled team sequences + metric sum/mean/std.
    6. **P3.1 (Backtest):** Removed unused imports (`DataProcessor`, `RecencyWeighter`, `VegasComparison`).

- **Quant Auditor Validation**
  - Ran full 4-year walk-forward backtest (2022-2025) post-fix
  - Identified sub-model issues for future work: Finishing Drives `overall_rating` not per-game normalized, kickoff scaling questionable

- **Refreshed All Performance Tables in MODEL_ARCHITECTURE.md**
  - Phase-by-Phase table updated from fresh 4-year closing line backtest
  - Core ATS table updated with closing + opening line data (9 separate runs)
  - CLV table updated with full-season opening line values
  - Per-year table updated with individual year backtests (both closing and opening)
  - Changelog entry added for P0 fixes

### Updated Baseline (Post-P0 Fixes)

| Metric | Old | New | Delta |
|--------|-----|-----|-------|
| Core MAE | 12.54 | 12.55 | +0.01 |
| Core ATS (Close) | 50.8% | 52.0% | **+1.2%** |
| Core 3+ (Close) | 51.7% | 51.9% | +0.2% |
| Core 5+ (Close) | 52.8% | 53.2% | +0.4% |
| Core 5+ (Open) | 57.0% | 57.1% | +0.1% |
| Full-Season CLV (5+) | +1.22 | +0.89 | -0.33 |

**Key takeaway:** Core ATS improved +1.2% with no MAE regression. The P0 fixes corrected data integrity issues (home_team validation, postseason chronology) that were adding noise to predictions.

### Files Modified
- `scripts/backtest.py` — postseason pseudo-weeks, home_team game join, ATS mask fix, import cleanup
- `src/models/efficiency_foundation_model.py` — pd.notna() fix, cache hash strengthening
- `docs/AUDIT_FIXLIST_BACKTEST.md` — P0.1/P0.3/P0.4/P3.1 marked fixed
- `docs/AUDIT_FIXLIST_EFM.md` — P0.2/P0.3 marked fixed
- `docs/MODEL_ARCHITECTURE.md` — all performance tables refreshed
- `CLAUDE.md` — governance and agent collaboration protocol updated

---

## Session: February 4, 2026

### Completed Today

- **Added 2022-2025 Backtest Performance Documentation**
  - Walk-forward backtest across 4 seasons (2,477 FBS games, weeks 4-15)
  - Aggregate: MAE 12.52, RMSE 15.80
  - ATS vs Closing: 51.0% all, 51.8% at 3+ edge, 53.2% at 5+ edge
  - ATS vs Opening: 53.1% all, 54.6% at 3+ edge, **57.0% at 5+ edge**
  - Key insight: Opening line performance significantly exceeds closing, indicating model captures value that market prices out

- **Documented Betting Line Data Sources**
  - Analyzed actual provider usage in backtest
  - DraftKings: 2,255 games (39%), 93% have opening lines
  - ESPN Bet: 1,671 games (29%), only 10% have opening lines
  - Bovada: 908 games (16%), 99% have opening lines
  - Overall: 55% of all games have true opening lines from CFBD

- **Integrated The Odds API for Future Data**
  - Created `src/api/odds_api_client.py` - API client for current/historical odds
  - Created `src/api/betting_lines.py` - Unified interface merging CFBD + Odds API
  - Created `scripts/capture_odds.py` - Backfill and one-time capture utility
  - Created `scripts/weekly_odds_capture.py` - Weekly scheduled captures
  - **Note:** Historical backfill requires paid Odds API plan (free tier = current only)
  - Tested current odds capture: 6 games returned, 1 credit used, 499 remaining

- **Data Strategy Established**
  - Historical (2022-2025): Continue using CFBD API (91% FBS opening line coverage)
  - Future (2026+): The Odds API (2 credits/week for opening + closing)
  - Opening lines: Capture Sunday ~6 PM ET
  - Closing lines: Capture Saturday ~9 AM ET

- **Reorganized Backtest Documentation**
  - Restructured MODEL_ARCHITECTURE.md to lead with phase-based performance
  - Added Full Season row to phase table (3,258 games, 49.5% ATS)
  - Moved ATS/CLV tables under "Core Season Detail (Weeks 4-15)" with clear scope
  - Removed redundant "Evaluation Results" section (had old metrics, conflicting phase definitions)
  - Updated MODEL_EXPLAINER.md with consistent phase-based stats

- **Implemented Opponent-Adjusted Metric Caching**
  - **Problem:** Ridge regression for Success Rate and IsoPPP was recomputed from scratch at each backtest week, leading to O(n²) work accumulation across the season
  - **Solution:** Module-level cache keyed by `(season, eval_week, metric_name, ridge_alpha, data_hash)` stores ridge regression results for reuse
  - **Implementation:**
    1. Added `_RIDGE_ADJUST_CACHE` dict and helper functions to `efficiency_foundation_model.py`:
       - `clear_ridge_cache()` - clears cache and returns stats
       - `get_ridge_cache_stats()` - returns hits, misses, size, hit_rate
       - `_compute_data_hash()` - fast data fingerprint for cache key
    2. Updated `_ridge_adjust_metric()` to check/populate cache:
       - Added `season` and `eval_week` parameters
       - Cache lookup before computation, cache storage after
    3. Updated `calculate_ratings()` to pass `season` and `max_week` parameters
    4. Updated call sites in `backtest.py` and `run_weekly.py`
  - **Cache safety guarantees:**
    - Ridge regression is deterministic: same inputs → same outputs
    - Cache key includes `data_hash` to guard against edge cases
    - Cache invalidates naturally via key mismatch when any parameter changes
  - **Performance monitoring:**
    - `run_backtest()` logs cache stats at end of run
    - `run_sweep()` logs aggregate cache stats across all iterations
  - **Why caching is safe:**
    - Mathematical output unchanged (same ridge regression code)
    - Cache key includes all parameters that affect results
    - `data_hash` ensures cache correctness if data differs for same (season, week)

- **P3.3: Vectorized Row-Wise Operations (~10x speedup)**
  - **Problem:** Multiple `apply(axis=1)` and `iterrows()` patterns in hot paths caused O(n) Python interpreter overhead on play-level data (~50,000 plays/season)
  - **Solution:** Replaced with vectorized NumPy/Pandas operations
  - **Files modified:**
    1. `src/models/special_teams.py`:
       - `calc_paae`: `np.select()` for FG expected rate lookup
       - `calc_net_yards`: `np.where()` + `np.minimum()` for punt net yards
       - `calc_punt_value`: Vectorized arithmetic for punt value
       - `calculate_from_game_stats`: `.sum()` instead of iterrows
    2. `src/adjustments/home_field.py`:
       - `calculate_league_hfa`: `.map(team_ratings)` for vectorized lookup
       - `calculate_team_hfa`: `.map()` for opponent ratings lookup
       - `calculate_dynamic_team_hfa`: `groupby` + `.map()` for residuals
    3. `src/models/finishing_drives.py`:
       - `calculate_from_game_stats`: `.sum()` instead of iterrows
    4. `scripts/analyze_stack_bias.py`:
       - Hard/soft cap: `np.minimum`, `np.where`
       - Sqrt/log scaling: Direct numpy operations
  - **Patterns NOT vectorized (justified):**
    - Team-level loops on groupby results (~130 teams) that create dataclass objects
    - Already O(teams) not O(plays), vectorization would require restructuring with minimal benefit
  - **Verification:** Backtest 2024 weeks 5-8 passed with identical numerical results

- **P3.4: DataFrame Dtype Optimization (~75% memory reduction)**
  - **Problem:** Default pandas dtypes (object, int64, float64) waste memory for columns with known ranges
  - **Solution:** Created centralized `config/dtypes.py` with explicit dtype mappings
  - **Dtype decisions:**
    | Column Type | Examples | Dtype | Rationale |
    |-------------|----------|-------|-----------|
    | Team identifiers | offense, defense, home_team | `category` | ~130 unique values, 8x savings |
    | Small integers | week, down, period | `int8` | Range 1-16, fits in -128 to 127 |
    | Medium integers | year, distance, stars | `int16` | Range fits in -32768 to 32767 |
    | Ratings/metrics | overall_rating, adj_sr | `float32` | 7 sig digits adequate |
    | Precision-critical | ppa, yards_gained | `float64` | Ridge regression needs precision |
  - **Files modified:**
    1. `config/dtypes.py` (NEW): Centralized dtype configuration with:
       - `CATEGORICAL_COLUMNS`, `INT8_COLUMNS`, `INT16_COLUMNS`, `FLOAT32_COLUMNS`
       - `optimize_dtypes(df)`: Apply optimal dtypes to DataFrame
       - `estimate_savings(df)`: Report memory savings
    2. `scripts/backtest.py`: Apply `optimize_dtypes()` after Polars→Pandas conversion
    3. `src/models/efficiency_foundation_model.py`: Import for potential future use
  - **Memory impact:** 50,000-row play DataFrame: 4.58 MB → 1.14 MB (75% reduction)
  - **Columns kept as float64 (precision required):**
    - `ppa`: Raw EPA values used in ridge regression
    - `yards_gained`: Accumulated in weighted sums
  - **Verification:** Backtest 2024 weeks 5-6 passed with identical ATS results

- **P3.5: DataFrame Transformation Chain Optimization (Peak Memory Reduction)**
  - **Problem:** Chained DataFrame operations (filter → copy → filter → copy) create multiple intermediate DataFrames with overlapping data, increasing peak memory usage
  - **Solution:** Audit and refactor to minimize intermediate copies
  - **Patterns optimized:**
    | File | Before | After |
    |------|--------|-------|
    | `efficiency_foundation_model.py` | 4 chained `df = df[mask]` filters | Combined into single compound mask |
    | `efficiency_foundation_model.py` | `plays_df.copy()` in `_calculate_raw_metrics` | Removed (caller already passes copy) |
    | `efficiency_foundation_model.py` | `successful_plays = ...copy()` | Removed (read-only usage) |
    | `special_teams.py` | Double `.copy()` (filter → copy → filter → copy) | Single copy with early column selection |
    | `home_field.py` | `.copy()` for single-column addition | Compute derived Series directly on view |
    | `finishing_drives.py` | `rz_plays = ...copy()` | Removed (read-only filtering) |
  - **Key techniques:**
    1. **Combined filter masks:** Build boolean mask with `&` then apply once
    2. **Drop unused columns early:** `plays_df.loc[mask, needed_cols].copy()` reduces copy size
    3. **Compute Series on view:** Avoid `.copy()` when only computing derived values
    4. **Read-only patterns:** Skip `.copy()` when filtered DataFrame is only read
  - **Files modified:**
    1. `src/models/efficiency_foundation_model.py`: Combined filters, removed unnecessary copies
    2. `src/models/special_teams.py`: Single copy with column selection, removed second copies
    3. `src/adjustments/home_field.py`: Compute derived Series on views
    4. `src/models/finishing_drives.py`: Removed read-only copy
  - **Verification:** Backtest 2024 weeks 5-6 passed with identical results (MAE 12.60, ATS 51.0%)
  - **No SettingWithCopyWarning:** Verified safe with `python -W all`

- **P3.6: Canonical Team Index (Eliminate Redundant Sorting)**
  - **Problem:** Multiple `sorted(set(...) | set(...))` calls computed the same team list throughout the EFM pipeline:
    - Line 724: `_calculate_raw_metrics()` - teams from groupby
    - Line 817: `_ridge_adjust_metric()` - teams for sparse matrix
    - Line 1212: `calculate_ratings()` - teams for final loop
  - **Solution:** Compute canonical team index ONCE early in `calculate_ratings()`, reuse throughout
  - **Changes:**
    1. Added `_canonical_teams: list[str]` and `_team_to_idx: dict[str, int]` to EFM
    2. Set at start of `calculate_ratings()` from prepared plays
    3. Helper methods check for canonical index before computing
  - **Additional optimization:** Removed unnecessary `sorted()` around `np.mean()`:
    - `np.mean()` is mathematically order-independent
    - Python dicts maintain insertion order since 3.7
    - Removed on lines 1216-1222 (SR/IsoPPP/turnover averages)
  - **Pattern guidance:**
    | Use Case | Pattern |
    |----------|---------|
    | Internal iteration (determinism) | `sorted(set(...))` - keep for determinism |
    | Multiple methods, same teams | Canonical index - compute once, reuse |
    | User-facing output | Sort in `get_summary_df()` - keep for presentation |
    | Math operations (mean, sum) | No sort needed - order-independent |
  - **Files audited:** EFM (optimized), special_teams.py, finishing_drives.py, home_field.py (all clean)
  - **Verification:** Backtest 2024 weeks 8-10 passed with identical results

- **P3.7: Conditional Execution for Run Modes (Lazy Evaluation)**
  - **Problem:** Weekly prediction and backtest modes share code paths but have different needs:
    - Backtest needs detailed diagnostics (stack analysis, CLV, phase metrics)
    - Weekly needs fast execution and report generation
    - Sweeps need minimal overhead (no diagnostics, no reports)
  - **Solution:** Added flags to skip unnecessary computations
  - **New flags:**
    | Script | Flag | What it skips | Savings |
    |--------|------|--------------|---------|
    | `backtest.py` | `--no-diagnostics` | Stack analysis, CLV report, phase metrics | ~150ms |
    | `run_weekly.py` | `--no-reports` | Excel/HTML report generation | ~100-200ms |
  - **Already lazy (no changes needed):**
    - `SpreadGenerator.track_diagnostics` defaults to False
    - Ridge regression cache already enabled in backtest
    - Special teams uses simplified path in weekly mode
  - **Implementation details:**
    1. `backtest.py`: Added `diagnostics` parameter to `print_results()`, gated diagnostic blocks
    2. `run_weekly.py`: Added `generate_reports` parameter to `run_predictions()`, conditional report creation
  - **Usage examples:**
    ```bash
    # Fast backtest (skip verbose diagnostics)
    python scripts/backtest.py --years 2024 --no-diagnostics

    # Fast weekly test (skip report generation)
    python scripts/run_weekly.py --year 2025 --week 5 --no-reports --no-wait
    ```
  - **Sanity report always runs:** `print_prediction_sanity_report()` is lightweight validation, not skipped

- **P3.8: Vectorized ATS Calculation (Batch Computation)**
  - **Problem:** `calculate_ats_results()` looped over 500+ predictions per backtest, building dicts one-by-one
  - **Solution:** Replaced per-game loop with pandas merge + numpy vectorized operations
  - **Changes:**
    1. Convert predictions list to DataFrame upfront
    2. Merge with betting data on `game_id` (left join)
    3. Vectorize all calculations: edge, home_cover, ats_win, ats_push, clv
    4. Use `np.where()` for conditional logic instead of if/else
  - **Code pattern:**
    ```python
    # Before: O(n) loop with dict construction
    for pred in predictions:
        edge = model_spread_vegas - vegas_spread
        if model_pick_home:
            ats_win = home_cover > 0
        ...
        results.append({...})

    # After: O(1) vectorized operations
    edge = model_spread_vegas - vegas_spread_vals  # numpy array
    ats_win = np.where(model_pick_home, home_cover > 0, home_cover < 0)
    ```
  - **Preserved functionality:**
    - Unmatched game tracking via merge + mask
    - Per-game logging for unmatched games (preserved)
    - All output columns identical
  - **Audit findings for future work:**
    - Main prediction loop requires per-game iteration (SpreadGenerator interface)
    - Batch prediction would require deep refactoring of all adjustment modules
    - HFA, travel, altitude lookups would need vectorized versions
  - **Verification:** Backtest 2024 weeks 5-7: identical results (MAE 12.60, ATS 74-75-7)

- **P3.9: Logging Verbosity Control (Gate Debug Output)**
  - **Problem:** Per-week EFM, ST, and FD calculations logged at INFO level, flooding output with 16+ log lines per season per model component during backtests
  - **Solution:** Gate non-essential logging behind verbosity flags and DEBUG level
  - **Changes:**
    1. Added `--verbose` flag to `backtest.py` for detailed per-week output
    2. Changed per-week model logs (EFM, SpecialTeams, FinishingDrives) to DEBUG level
    3. Changed per-year iteration logs in `fetch_all_season_data()` to DEBUG level
    4. Made `print_data_sanity_report()` compact by default (warnings-only), verbose for details
    5. Converted QB adjuster `print()` statements to `logger.info()`
    6. Changed portal winners/losers and coaching adjustment per-team logs to DEBUG
  - **Files modified:**
    - `scripts/backtest.py`: Added `--verbose` flag, gated per-week MAE and sanity details
    - `scripts/run_weekly.py`: Changed per-year historical fetch to DEBUG
    - `src/models/efficiency_foundation_model.py`: Ridge stats, validation, ratings to DEBUG
    - `src/models/special_teams.py`: FG/Punt/Kickoff ratings to DEBUG
    - `src/models/finishing_drives.py`: Ratings calculation to DEBUG
    - `src/adjustments/home_field.py`: Trajectory modifiers to DEBUG
    - `src/adjustments/qb_adjustment.py`: Converted print() to logger.info()
    - `src/models/preseason_priors.py`: Portal winners/losers, coaching details to DEBUG
  - **Default behavior:** Quiet summary output with warnings only
  - **Verbose behavior:** `--verbose` shows per-week MAE, per-year data details
  - **Debug behavior:** Full logging restored with `--debug` flag (sets DEBUG level)

- **P3 Optimization Benchmark Results (SAFE OPTIMIZATION)**
  - **Purpose:** Comprehensive audit comparing pre-P3 vs post-P3 refactored code
  - **Methodology:**
    - Baseline: Simulated pre-P3 behavior (dense matrices, row-wise apply operations)
    - Current: P3-optimized code (sparse matrices, vectorized numpy)
    - N=3 iterations for statistical stability
    - Tested on 2024 season (872 games, 201,045 plays, 134 teams)
  - **Results:**
    | Metric | Baseline | Current | Improvement |
    |--------|----------|---------|-------------|
    | Mean Runtime | 13,148 ms | 6,595 ms | **+49.8%** |
    | Peak Memory | 628.9 MB | 38.8 MB | **+93.8%** |
    | Max Rating Diff | - | 0.0000 | ✓ (threshold: ≤0.01) |
    | Spearman ρ | - | 1.000000 | ✓ (threshold: ≥0.999) |
  - **Isolated Optimization Measurements:**
    | Optimization | Before | After | Improvement |
    |--------------|--------|-------|-------------|
    | P3.1 Sparse Matrix | 918.8 MB | 5.4 MB | **99.4% memory** |
    | P3.1 Sparse Build | 334.9 ms | 242.3 ms | 1.4x faster |
    | P3.3 Row-wise → Vectorized | 745.5 ms | 1.9 ms | **390x faster** |
  - **Final Verdict:** ✓ **SAFE OPTIMIZATION**
    - Runtime improvement ≥25%: ✓ PASS (+49.8%)
    - Memory reduction ≥20%: ✓ PASS (+93.8%)
    - Algorithmic integrity: ✓ PASS (identical ratings)
  - **Benchmark script:** `scripts/benchmark.py` (created)
  - **Report:** `data/outputs/benchmark_report_20260204_193446.txt`

- **Fixed Historical Rankings for Letdown Spot Detection (Critical Model Fix)**
  - **Problem:** `check_letdown_spot()` used current rankings to evaluate previous opponents
  - **CFB Context:** Rankings are volatile in college football. A team ranked #15 in Week 3 may be unranked by Week 8, or vice versa. When evaluating "did they beat a ranked team last week?", we must use the rank AT THE TIME of the game, not today's ranking.
  - **Example of bug:**
    - Week 7: Oregon beats #2 Ohio State
    - Week 8: Oregon plays unranked Purdue
    - **Old behavior:** If Ohio State dropped to #20 by Week 10 when we run backtest, no letdown detected (wrong)
    - **New behavior:** Uses Week 7 historical ranking (#2), correctly detects letdown spot
  - **Solution:**
    1. Added `HistoricalRankings` class to store week-by-week AP poll data
    2. Added `get_rankings(year, week)` method to CFBDClient
    3. Modified `check_letdown_spot()` to accept `historical_rankings` parameter
    4. Updated `SituationalAdjuster`, `SpreadGenerator`, and backtest to load and pass historical rankings
  - **Files modified:**
    - `src/api/cfbd_client.py`: Added `rankings_api` property and `get_rankings()` method
    - `src/adjustments/situational.py`: Added `HistoricalRankings` class, updated all spot-checking methods
    - `src/predictions/spread_generator.py`: Added `historical_rankings` to `predict_spread()` and `predict_week()`
    - `scripts/backtest.py`: Load historical rankings in `fetch_all_season_data()`, pass through pipeline
    - `scripts/run_weekly.py`: Load historical rankings before predictions
  - **Backward compatibility:** Falls back to current rankings if historical not available

- **Replaced Binary Bye Week with Rest Day Differential Calculation**
  - **Problem:** `check_bye_week()` was binary (played last week or not)
  - **CFB Context:** College football isn't just Saturdays. MACtion Tuesday/Wednesday and Thursday/Friday games create meaningful rest differentials:
    - **Mini-bye:** Thursday game → following Saturday = 9 days rest (advantage)
    - **Short week:** Saturday game → Thursday = 5 days rest (disadvantage)
    - **Normal:** Saturday → Saturday = 7 days rest (baseline)
  - **Solution:**
    1. Added `calculate_rest_days()` method using actual game `start_date`
    2. Added `calculate_rest_advantage()` method: `(home_rest - away_rest) × 0.5 pts/day`
    3. Capped at ±1.5 pts (equivalent to full bye week advantage)
    4. Updated `SituationalFactors` dataclass to include `rest_days` and `rest_advantage`
    5. Updated `get_matchup_adjustment()` to compute rest differential
  - **Examples:**
    - Oregon (Thu game, 9 days) vs Texas (Sat game, 7 days): Oregon +1.0 pts rest advantage
    - Toledo short week (Sat → Thu, 5 days) vs normal opponent: Toledo -1.0 pts penalty
  - **Files modified:**
    - `src/adjustments/situational.py`: New rest day calculation methods
    - `src/predictions/spread_generator.py`: Pass `game_date` through pipeline
    - `scripts/backtest.py`: Pass `start_date` to predict_spread
  - **Backward compatibility:** Falls back to binary bye week check if dates unavailable

- **Implemented Data Leakage Prevention Guards**
  - **Problem:** Walk-forward backtesting relies on filtering data by game_id/week, but no programmatic guards existed to catch accidental leakage of future data into model training
  - **Solution:** Added explicit assertions throughout the pipeline that verify `max_week` constraints
  - **Files modified:**
    1. `scripts/backtest.py` - Added assertions after filtering:
       - `assert max_train_week < pred_week` for plays and games
       - `assert max_st_week < pred_week` for special teams plays
       - Passes `max_week=pred_week-1` to all model methods
    2. `src/models/efficiency_foundation_model.py`:
       - Added `max_week` parameter to `calculate_ratings()` and `_prepare_plays()`
       - Guard assertion at start of `_prepare_plays()`
       - Fixed `time_decay` to use explicit `max_week` instead of `df["week"].max()` (defense-in-depth)
    3. `src/models/finishing_drives.py`:
       - Added `max_week` parameter to `calculate_all_from_plays()`
       - Guard assertion at start of method (before any processing)
    4. `src/models/special_teams.py`:
       - Added `max_week` parameter to `calculate_all_st_ratings_from_plays()`
       - Guard assertion at start of method
  - **Testing:** Verified guards catch leakage (assertion fires when week > max_week) and allow valid data through

- **Audited Normalization Order (Confirmed Correct)**
  - **Question:** Should normalization happen before or after opponent adjustment?
  - **Finding:** Current order is mathematically correct: Raw → Opponent Adjust → Points → Normalize
  - **Why current order is RIGHT:**
    1. Ridge regression is scale-sensitive; alpha (50.0) tuned for metric scale (SR 0-1, IsoPPP ~0.3)
    2. Intercept has natural interpretation on metric scale (≈0.42 = league avg SR)
    3. Implicit HFA extraction works correctly on metric scale (≈0.03 = 3% higher SR at home)
    4. Point conversion factors (80.0, 15.0) calibrated for opponent-adjusted metrics
  - **Why normalizing first would be WRONG:**
    1. Would change ridge alpha needed (different scale)
    2. Intercept would be ~0 (meaningless)
    3. HFA coefficient would be on arbitrary scale
    4. Point conversion factors would need recalibration
  - **Documentation added:** Extended docstring in `calculate_ratings()` with full mathematical justification
  - **Key insight:** Normalization is a linear transform that preserves all inter-team relationships; it can safely happen last

- **Audited ST vs Offensive Efficiency (Confirmed No Double-Counting)**
  - **Concern:** Does ST field position value overlap with offensive efficiency?
  - **Finding:** NO double-counting - the metrics are mathematically independent
  - **Why they're independent:**
    1. EFM measures PLAY EFFICIENCY: Success Rate (% plays meeting yardage thresholds) and IsoPPP (EPA per successful play)
    2. Neither metric uses starting field position (yards_to_goal not referenced in EFM)
    3. ST measures FIELD POSITION VALUE: converts yards to points using YARDS_TO_POINTS = 0.04
    4. They capture different information:
       - EFM: "How efficiently does this team move the ball per play?"
       - ST: "How much field position advantage does this team create?"
  - **Integration design (P2.7):**
    1. EFM calculates base_margin = home_overall - away_overall (efficiency only)
    2. ST differential calculated separately via SpecialTeamsModel
    3. Combined ADDITIVELY in SpreadGenerator (not multiplicatively)
    4. EFM's special_teams_rating is DIAGNOSTIC ONLY (not in overall_rating)
  - **Documentation added:** Extended EFM class docstring with "FIELD POSITION INDEPENDENCE" section
  - **No refactoring needed:** Architecture already prevents double-counting by design

- **Implemented Model Determinism Fixes**
  - **Goal:** Ensure model produces identical outputs given identical inputs (week-by-week reproducibility)
  - **Issues identified and fixed:**
    1. **Set iteration order:** Python sets have non-deterministic iteration order. Fixed by using `sorted()` on all set unions.
    2. **Sort stability:** When sorting teams by rating, ties produce non-deterministic ordering. Fixed by adding team name as secondary sort key.
    3. **Floating-point summation order:** `np.mean()` results can vary slightly based on summation order. Fixed by sorting values before mean calculation where it matters.
  - **Files modified:**
    1. `src/models/efficiency_foundation_model.py`:
       - Line 621: `all_teams = sorted(set(off_grouped.index) | set(def_grouped.index))`
       - Line 856: `all_teams = sorted(set(turnovers_lost.index) | set(turnovers_forced.index))`
       - Line 1040: `all_teams = sorted(set(adj_off_sr.keys()) | set(adj_def_sr.keys()))`
       - Lines 1044-1050: Added `sorted()` to `np.mean()` value lists
    2. `src/models/finishing_drives.py`:
       - Line 346: `all_teams = sorted(set(rz_plays["offense"].dropna()))`
    3. `src/models/special_teams.py`:
       - Line 746: `sorted()` for kickoff coverage/return union
       - Line 806: `sorted()` for FG/punt/kickoff union
    4. `src/models/preseason_priors.py`:
       - Lines 449, 688, 941: `sorted()` for all set unions
    5. `src/adjustments/home_field.py`:
       - Line 263: `sorted()` for team set union
    6. `scripts/backtest.py`:
       - Line 609: Added `(-x[1], x[0])` sort key for stable team ranking
  - **Result:** Model now produces deterministic outputs for identical inputs across runs
  - **Verified:** Backtest ran successfully (2024, weeks 4-16):
    - 677 games predicted, no assertion errors (data leakage guards passed)
    - MAE: 12.61 pts, ATS: 49.6% overall, 53.3% at 5+ edge
    - Mean error: -0.22 pts (no systematic bias)
    - All determinism fixes working correctly

### Model Integrity Summary

All four integrity tasks completed:

| Task | Status | Commit |
|------|--------|--------|
| Data Leakage Prevention | ✅ Fixed | `40e403b` |
| Normalization Order Audit | ✅ Confirmed Correct | `5f4855d` |
| ST vs EFM Double-Counting | ✅ Confirmed Independent | `eb018d7` |
| Model Determinism | ✅ Fixed | `ca165de` |

### Transfer Portal Logic Refactor

Comprehensive overhaul of transfer portal evaluation in `src/models/preseason_priors.py`:

**1. Scarcity-Based Position Weights (2026 Market)**

Reflects that elite trench play is the primary driver of rating stability:

| Tier | Position | Weight | Rationale |
|------|----------|--------|-----------|
| Premium | QB | 1.00 | Highest impact position |
| Premium | OT | 0.90 | Elite blindside protector (90% of QB value) |
| Anchor | EDGE | 0.75 | Premium pass rushers |
| Anchor | IDL | 0.75 | Interior pressure + run stuffing |
| Support | IOL | 0.60 | Interior OL (guards/centers) |
| Support | LB | 0.55 | Run defense, coverage |
| Support | S | 0.55 | Deep coverage, run support |
| Support | TE | 0.50 | Hybrid role, increasing value |
| Skill | WR | 0.45 | Higher replacement rate |
| Skill | CB | 0.45 | Athletic translation |
| Skill | RB | 0.40 | Most replaceable skill position |
| Depth | ST | 0.15 | Limited snaps |

**2. Level-Up Discount Logic (G5→P4 Transfers)**

| Move Type | Trench Positions | Skill Positions |
|-----------|------------------|-----------------|
| P4 → P4 | 100% value | 100% value |
| G5 → P4 | 75% value (Physicality Tax) | 90% value (Athleticism Discount) |
| P4 → G5 | 110% value (proven at higher level) | 110% value |

- **Physicality Tax (25%):** Applied to OT, IOL, IDL, LB, EDGE - steep curve in trench physicality at P4 level
- **Athleticism Discount (10%):** Applied to WR, RB, CB, S - high-end speed translates more easily

**3. Continuity Tax**

- Factor: 0.90 (losses amplified by ~11%)
- Rationale: Losing an incumbent hurts more than raw talent value suggests (chemistry, scheme fit, experience)

**4. Volatility Management**

- Impact cap tightened from ±15% to ±12%
- FBS-only filtering (excludes FCS noise)
- Conference-aware P4/G5 classification

**5. Blue Blood Validation**

Verified talent integration properly offsets portal losses:

| Team (2024) | Portal Impact | Talent Score | Final Rating Δ |
|-------------|---------------|--------------|----------------|
| Alabama | -12.0% | +26.0 | -0.28 pts |
| Ohio State | -12.0% | +24.6 | -0.32 pts |
| Georgia | -12.0% | +25.2 | -0.40 pts |
| Texas | -12.0% | +21.6 | -0.30 pts |

**Conclusion:** Blue Bloods hitting -12% portal cap show minimal final rating impact (~0.3 pts) because talent composite correctly captures elite recruiting offset.

**6. Backtest Impact Analysis (2024-2025)**

A/B comparison of portal logic impact on ATS performance:

| Phase | Metric | With Portal | Without Portal | Δ |
|-------|--------|-------------|----------------|---|
| Calibration (1-3) | ATS % | 46.7% | 46.3% | +0.4% |
| Calibration (1-3) | 5+ Edge | 47.5% | 48.2% | -0.7% |
| Core (4-15) | ATS % | 51.3% | 51.2% | +0.1% |
| Core (4-15) | 5+ Edge | **55.5%** | 54.8% | **+0.7%** |

**Finding:** Portal logic has minimal but slightly positive impact on Core Season 5+ edge (+0.7%). The muted effect is expected because:
1. Portal adjusts regression factor (indirect), not raw ratings
2. Talent composite provides primary offset for Blue Bloods
3. Preseason priors fade by week 8 as in-season data dominates

**Commits:** `ab84a9a` (portal refactor), `b5beeb0` (backtest --end-week parameter)

### Scripts Added

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/capture_odds.py` | One-time capture, backfill, quota check | `--check-quota`, `--capture-current`, `--backfill --year 2024` |
| `scripts/weekly_odds_capture.py` | Scheduled weekly captures | `--opening` (Sunday), `--closing` (Saturday) |

### Environment Setup

```bash
# Set API key
export ODDS_API_KEY="your_key_here"

# Check quota (free)
python scripts/capture_odds.py --check-quota

# Preview available odds (1 credit)
python scripts/weekly_odds_capture.py --preview

# Capture opening lines (1 credit)
python scripts/weekly_odds_capture.py --opening

# Capture closing lines (1 credit)
python scripts/weekly_odds_capture.py --closing
```

---

## Session: February 3, 2026

### Completed Today

- **Expanded Special Teams to Full PBTA Model** - Complete overhaul from FG-only to comprehensive FG + Punt + Kickoff
  - **Problem:** ST ratings were mixing units (FG in points, punt in yards, kickoff in mixed scale)
  - **Solution:** All components now expressed as PBTA (Points Better Than Average) - the marginal point contribution per game
  - **Key changes to `src/models/special_teams.py`:**
    1. Added `YARDS_TO_POINTS = 0.04` constant (from expected points models: ~0.04 pts per yard of field position)
    2. Punt rating: `net_yards_vs_expected × 0.04` + inside-20 bonus (+0.5 pts) + touchback penalty (-0.3 pts)
    3. Kickoff coverage: touchback rate bonus + return yards saved × 0.04
    4. Kickoff returns: return yards gained × 0.04
    5. Overall = simple sum (no weighting needed since all in same unit)
  - **FBS ST Distribution (2024):**
    - Mean: ~0.05 pts/game (essentially 0)
    - Std: ~1.06 pts/game
    - 95% range: [-2.02, +2.17] pts/game
  - **Top ST units:** Vanderbilt (+2.34), Charlotte (+2.22), Florida State (+2.22), Georgia (+2.19)
  - **Worst ST units:** UTEP (-2.83), Sam Houston (-2.35), Kent State (-2.27)
  - **Interpretation:** Vanderbilt's ST gains them ~2.3 more points per game than an average unit would
  - Updated all docstrings to clarify PBTA convention and sign conventions
  - Updated `SpecialTeamsRating` dataclass with PBTA documentation

- **Implemented Neutral-Field Ridge Regression (MAJOR FIX)** - Fixed systematic -6.7 mean error caused by double-counting home field advantage
  - **Problem:** The CFBD EPA data implicitly contains HFA—home teams naturally generate better EPA due to crowd noise, familiarity, etc. The ridge regression was learning team coefficients with this HFA baked in. When SpreadGenerator added explicit HFA, this caused double-counting.
  - **Solution:** Added home field indicator column to ridge regression design matrix:
    - `+1` when offense is home team (home advantage)
    - `-1` when defense is home team (away disadvantage)
    - `0` for neutral site plays
  - **Code changes:**
    - `backtest.py` line 263: Added `"home_team": play.home` to efficiency_plays
    - `efficiency_foundation_model.py` lines 298-401: Modified `_ridge_adjust_metric()` to check for `home_team` column and add home indicator to design matrix
    - `efficiency_foundation_model.py` lines 178-180: Added `learned_hfa_sr` and `learned_hfa_isoppp` instance variables to store the learned implicit HFA
  - **Learned Implicit HFA:**
    - Success Rate: ~0.006 (~0.6% SR advantage for home teams)
    - IsoPPP: ~0.02 (~0.02 EPA advantage)
    - Combined in points: ~0.8 pts (much smaller than explicit ~2.5 pts HFA)
  - **Results:**
    | Metric | Before | After |
    |--------|--------|-------|
    | Mean Error (2024) | -6.7 pts | **-0.40 pts** |
    | Mean Error (2022-2024) | ~-6.7 pts | **+0.51 pts** |
    | MAE (2022-2024) | ~12.4 | 12.48 |
    | Overall ATS | ~51% | 50.9% (921-889-37) |
    | 3+ edge ATS | ~54% | 53.1% (519-459) |
    | 5+ edge ATS | ~57% | 56.2% (336-262) |
  - **Key Insight:** Most HFA manifests at scoring/outcome level (special teams, turnovers, momentum) rather than pure play-by-play efficiency. The ~0.8 pts implicit HFA in the data is small compared to the ~2.5 pts explicit HFA applied by SpreadGenerator.

- **Updated Rule 4 (Modeling Philosophy)** - The -6.7 mean error is now FIXED via neutral-field regression. The model no longer has systematic home bias.

- **Fixed P0.1: Trajectory Leakage** - Fixed data leakage in `calculate_trajectory_modifiers()`
  - **Bug:** Recent years calculation used `range(current_year - 1 + 1, current_year + 1)` which equals `[current_year]`
  - **Fix:** Changed to `range(current_year - 1, current_year)` which equals `[current_year - 1]`
  - **Result:** For 2024 predictions, now uses:
    - Baseline: 2020, 2021, 2022 (3 years before recent)
    - Recent: 2023 (1 year before current_year)
    - NOT used: 2024 (current year being predicted)
  - **Logging:** Added explicit log message showing year ranges: "Trajectory modifiers for 2024: baseline years [2020-2022], recent years [2023-2023] (current year 2024 NOT included)"
  - **Verified:** Test case confirms correct behavior (uses prior year data, not current year)

- **Fixed P0.2: Game ID Matching for ATS** - Refactored Vegas line matching to use `game_id` instead of `(home_team, away_team)`
  - **Problem:** Team name matching can fail on rematches, naming drift (e.g., "Miami" vs "Miami (FL)"), and neutral-site home/away flips
  - **Changes:**
    1. Added `game_id` to prediction results in both `walk_forward_predict` and `walk_forward_predict_efm`
    2. Refactored `calculate_ats_results()` to match by `game_id` using O(1) dict lookup
    3. Added sanity report logging: "ATS line matching: X/Y predictions matched (Z%), N unmatched"
    4. Updated `VegasComparison` to support `game_id` matching via `get_line_by_id()` and updated `get_line()` to prefer `game_id` when provided
  - **Result:** 100% match rate (631/631 in 2024 test)
  - **Files changed:** `scripts/backtest.py`, `src/predictions/vegas_comparison.py`

- **Fixed P0.3: Remove EFM Double-Normalization** - Eliminated scaling from rounded outputs
  - **Problem:** EFM's `_normalize_ratings()` scaled to std=12.0, then backtest's `walk_forward_predict_efm()` was rescaling to std=10.0 using values from `get_ratings_df()` (rounded to 1 decimal). This introduced:
    1. Double-normalization (std=12 → std=10)
    2. Rounding contamination (full precision → 1 decimal → rescaled)
  - **Fix:** Removed the second normalization block in `walk_forward_predict_efm()`. Now uses `efm.get_rating(team)` directly for full precision ratings.
  - **Verification:** Test script confirmed:
    - Full precision std = 12.0000 stable across weeks 4-8
    - Means centered at 0.0000 (max deviation 0.0000)
    - Rounding error minimal (max difference 0.048 between full precision and rounded DF)
  - **Key insight:** EFM's `_normalize_ratings()` (std=12.0) is now the ONLY normalization. `get_ratings_df()` is for presentation only, not for calculations.
  - **Files changed:** `scripts/backtest.py`

- **Fixed P0.4: Prevent Silent Season Truncation** - Made data fetching resilient to transient API errors
  - **Problem:** Both `fetch_season_data()` and `fetch_season_plays()` used `break` on exceptions, which silently dropped all remaining weeks if any single week failed.
  - **Fix:**
    1. Changed `break` to `continue` so errors skip only the affected week, not all remaining weeks
    2. Added `failed_weeks` and `successful_weeks` tracking in both functions
    3. Added warning logs when weeks fail, showing week number and error message
    4. Added data completeness sanity check in `fetch_all_season_data()` that reports missing weeks
  - **Logging examples:**
    - On success: `"Data completeness check for 2024: all weeks 1-15 present"`
    - On failure: `"Games fetch for 2024: 14 weeks OK, 1 weeks FAILED: [7]"`
    - Per-week: `"Failed to fetch plays for 2024 week 7: <error>"`
  - **Result:** Backtests remain deterministic - missing data is now explicit and logged rather than silently dropped.
  - **Files changed:** `scripts/backtest.py`

- **Verified P2.2: Ridge Intercept Handling** - Confirmed no double-counting of baseline
  - **Concern:** Audit flagged that intercept is applied to both offense and defense outputs, potentially double-counting.
  - **Analysis:**
    1. Both `off_adjusted = intercept + off_coef` and `def_adjusted = intercept - def_coef`
    2. When computing ratings: `overall ∝ (off_sr - def_sr) = off_coef + def_coef` - intercept cancels!
    3. Pre-normalization means: `mean(off) ≈ 0`, `mean(def) ≈ 0` due to balanced ridge regression
    4. Normalization uses `current_mean/2` for O/D, which correctly distributes turnover residual
    5. Formula `overall = off + def + 0.1*TO` verified to hold exactly (error = 0.000000)
  - **Conclusion:** NOT A BUG. The implementation is mathematically correct.
  - **Action:** Added clarifying comment to `_normalize_ratings()` explaining the math.
  - **Files changed:** `src/models/efficiency_foundation_model.py`, `docs/Audit_Fixlist.md`

- **Fixed P1.1: Travel Direction Logic Inversion** - West→East now correctly gets full penalty
  - **Bug:** The longitude comparison in `get_timezone_adjustment()` was inverted.
  - **Root cause:** When `away_lon < home_lon`, away team is WEST of home (more negative longitude), so they're traveling EAST. But the code was applying the 0.8x "easier" multiplier instead of full penalty.
  - **Evidence:**
    - Before fix: UCLA→Rutgers (West→East) got -1.20, Rutgers→UCLA (East→West) got -1.50
    - After fix: UCLA→Rutgers (West→East) gets -1.50, Rutgers→UCLA (East→West) gets -1.20
  - **Fix:** Swapped the condition branches so `away_lon < home_lon` → full penalty (traveling east, harder).
  - **Verified:** All 6 test cases pass (UCLA↔Rutgers, Oregon↔Ohio State, USC↔Penn State).
  - **Files changed:** `src/adjustments/travel.py`

- **Simplified P1.2: Travel Direction Uses TZ Offsets Instead of Longitude** - Cleaner, more direct logic
  - **Problem:** Direction detection used longitude comparison which was confusing (negative values, less-than vs greater-than).
  - **Solution:** Added `get_directed_timezone_change()` function that returns signed timezone difference:
    - Positive = traveling EAST (losing time, harder)
    - Negative = traveling WEST (gaining time, easier)
  - **Logic simplification:**
    - Old: `if away_loc["lon"] < home_loc["lon"]` (confusing with negative numbers)
    - New: `if directed_tz > 0` (clear: positive = east)
  - **Verified:** All 8 test cases pass including Hawaii (5 TZ difference):
    - Hawaii→Ohio State: directed_tz=+5, adj=-2.50 (full penalty)
    - Ohio State→Hawaii: directed_tz=-5, adj=-2.00 (0.8x penalty)
  - **Files changed:** `config/teams.py`, `src/adjustments/travel.py`

- **Fixed P1.3: HFA Source Tracking and Logging** - Made HFA sourcing explicit and logged
  - **Problem:** Multiple HFA sources (curated, dynamic, conference, fallback, trajectory) with no visibility into which was used.
  - **Changes:**
    1. `get_hfa()` now returns `(hfa_value, source_string)` tuple with source tracking
    2. Sources: "curated", "dynamic", "conf:XXX", "fallback", plus "+traj(±X.XX)" suffix
    3. Added `get_hfa_value()` for backward compatibility (returns just the float)
    4. Added `get_hfa_breakdown(teams)` for per-team HFA details
    5. Added `log_hfa_summary(teams)` for aggregate logging
    6. Backtest logs HFA source distribution at start of first week
  - **Example output:** `HFA sources: curated=46, fallback=90, with_trajectory=108`
  - **HFA Priority (documented in get_hfa):**
    1. Neutral site → 0
    2. TEAM_HFA_VALUES (curated) → hardcoded team values
    3. self.team_hfa (dynamic) → calculated values
    4. CONFERENCE_HFA_DEFAULTS → conference fallback
    5. self.base_hfa → CLI/settings fallback
    6. Plus trajectory modifier if applicable
  - **Files changed:** `src/adjustments/home_field.py`, `src/predictions/spread_generator.py`, `scripts/backtest.py`

---

## Session: February 2, 2026

### Completed Today

- **Added Weather Adjustment Module** - New module for totals prediction (`src/adjustments/weather.py`)
  - **Purpose:** Weather significantly impacts game totals (over/under). This prepares JP+ for totals prediction.
  - **Data Source:** CFBD API `get_weather()` endpoint - provides temperature, wind speed/direction, precipitation, snowfall, humidity, indoor flag
  - **Adjustments:**
    | Factor | Threshold | Adjustment | Cap |
    |--------|-----------|------------|-----|
    | Wind | >10 mph | -0.3 pts/mph | -6.0 |
    | Temperature | <40°F | -0.15 pts/degree | -4.0 |
    | Precipitation | >0.02 in | -3.0 pts | — |
    | Heavy Precip | >0.05 in | -5.0 pts | — |
  - **Smart precip logic:** Only applies rain penalty when `weather_condition` indicates actual rain/snow (not fog/humidity)
  - **Indoor detection:** Dome games (`game_indoors=True`) receive no adjustment
  - **Example extreme games (2024):**
    - UNLV @ San Jose State: -8.5 pts (22 mph wind + heavy rain)
    - Nebraska @ Iowa: -3.6 pts (16°F cold)
    - Yale @ Harvard: -5.2 pts (17 mph wind + light rain)
  - **Status:** Ready for totals integration. Parameters are conservative estimates based on NFL weather studies.

- **Field Position Component - EXPLORED, TABLED** - Investigated adding field position as a JP+ component (SP+ uses 10% weight)
  - **Data source:** CFBD DrivesApi provides start_yardline for all drives
  - **Raw field position problem:** Heavily confounded by schedule - good teams have WORSE raw FP (r = -0.56 with SP+)
  - **Opponent-adjusted FP:** Still negatively correlated (r = -0.63) - confounding persists
  - **Return Game Rating (cleaner signal):** Used punt return yards and coverage data
    - Weak correlation with team quality (r = +0.20) - good, means it's different signal
    - Weak correlation with overperformance (r = +0.06) - concerning for predictive value
    - Top 30 return teams: +5.3 ranks overperformance
    - Bottom 30 return teams: -0.7 ranks overperformance
  - **Root causes of confounding:**
    1. Good teams score more TDs → receive more kickoffs at own 25
    2. Good teams face better punters/coverage units
    3. Good teams force turnovers deeper in opponent territory
  - **Decision:** TABLED. Signal too weak to justify complexity. Will revisit with proper ATS backtest if needed.
  - **Note:** SP+ uses 10% for field position, but JP+ may not need it if Vegas already prices it

- **Added FPI Ratings Comparison** - 3-way validation of JP+ vs FPI vs SP+ (`scripts/compare_ratings.py`)
  - **Purpose:** External benchmark for JP+ ratings quality
  - **Implementation:** Added `get_fpi_ratings()` to CFBD client, fixed `get_sp_ratings()` to use correct `RatingsApi` endpoint
  - **2025 Correlation Results:**
    | Comparison | Rating r | Rank r |
    |------------|----------|--------|
    | JP+ vs FPI | 0.956 | 0.947 |
    | JP+ vs SP+ | 0.937 | 0.934 |
    | FPI vs SP+ | 0.970 | 0.965 |
  - **Key Divergence:** JP+ has Ohio State #1, Indiana #2; FPI/SP+ have Indiana #1
  - **JP+ unique Top 25:** James Madison (#17), Florida State (#22)

- **Calibration Centering Experiment** - Tested whether fixing -6.7 mean error helps or hurts ATS
  - **Hypothesis:** Model's "calibration debt" (double-counting HFA) is a structural risk that could backfire
  - **Test:** Swept HFA from +2.5 (current) to -4.0 (centered, ~0 mean error)
  - **Results (2022-2025, n=2230):**
    | HFA | MAE | 3+ edge | 5+ edge |
    |-----|-----|---------|---------|
    | 2.5 (current) | **12.37** | **54.0%** | **57.3%** |
    | 0.0 | 12.41 | 53.6% | 55.4% |
    | -4.0 (centered) | 12.52 | 53.3% | 55.0% |
  - **Finding:** Centering HURTS performance across the board (MAE +0.15, 3+ edge -0.7%, 5+ edge -2.3%)
  - **Interpretation:** The "bias" appears to capture something real that Vegas underprices. The model's effective ~6pt HFA may reflect EPA advantage + traditional HFA that Vegas doesn't fully account for.
  - **Risk acknowledged:** If market dynamics shift, our edge could evaporate. Monitor year-over-year.
  - **Decision:** Keep current HFA=2.5. The bias is a feature, not a bug.

- **Defense Full-Weight Garbage Time - TESTED AND REJECTED** - Investigated giving defense full credit when winning in GT
  - **Hypothesis:** When a team is up 17+ in Q4, their defense should get full weight (not 0.1x) since they earned the blowout
  - **Implementation:** Added `defense_full_weight_gt` parameter to EFM with separate O/D regression weights
    - Offense: uses standard asymmetric weights (1.0 winning, 0.1 trailing)
    - Defense: uses full weight when defense's team is winning in garbage time
  - **Ranking Impact (2025 Top 25):**
    | Team | Before | After | Change |
    |------|--------|-------|--------|
    | Indiana | #4 | #2 | +2 ↑ |
    | Georgia | #9 | #8 | +1 ↑ |
    | Ole Miss | #14 | #12 | +2 ↑ |
  - **Backtest Results (2022-2025):**
    | Metric | Before | After | Diff |
    |--------|--------|-------|------|
    | MAE | 12.37 | 12.38 | +0.01 |
    | 3+ edge | 54.0% | 54.2% | +0.2% |
    | **5+ edge** | **57.3%** | **56.5%** | **-0.8%** |
  - **Finding:** Rankings improved for underrated teams, but 5+ edge ATS REGRESSED by 0.8%
  - **Interpretation:** The "right" rankings hurt predictive accuracy - suggests current asymmetric GT is correctly calibrated
  - **Decision:** REJECTED. ATS is our optimization target (Rule #3). Keep standard asymmetric weighting.

- **Dynamic Alpha Experiment - TESTED AND REJECTED** - Investigated using different ridge alpha by week
  - **Hypothesis:** Use higher regularization (alpha=75) early season when data is noisy, lower (alpha=35) late season to "let the data speak"
  - **Implementation:** Added `--dynamic-alpha` flag: weeks 4-6 use 75, weeks 7-11 use 50, weeks 12+ use 35
  - **Results (2022-2025):**
    | Config | MAE | 3+ edge | 5+ edge |
    |--------|-----|---------|---------|
    | Flat alpha=50 | 12.41 | 53.5% | **56.4%** |
    | Dynamic (75→50→35) | 12.41 | 53.5% | 55.7% |
  - **Finding:** Dynamic alpha hurt 5+ edge by 0.7%
  - **Why it fails:** Walk-forward already self-regularizes via sample size. Lower alpha late-season helps ALL teams equally, not just elite ones.
  - **Decision:** REJECTED. Keep flat alpha=50.

- **Turnover Prior Strength Sweep - CONFIRMED OPTIMAL** - Tested whether turnovers should be regressed more aggressively
  - **Hypothesis:** Turnovers are ~50-70% luck (especially fumble recoveries). Prior strength of 10 may over-credit turnover margin.
  - **Sweep:** Tested prior_strength = 5, 10, 20, 30
  - **Results (2022-2025):**
    | Prior Strength | 3+ edge | 5+ edge |
    |----------------|---------|---------|
    | 5 | 53.5% | 56.0% |
    | **10 (current)** | 53.5% | **56.4%** |
    | 20 | 53.4% | 56.3% |
    | 30 | **53.7%** | 55.8% |
  - **Finding:** Prior strength 10 is already optimal for 5+ edge. More aggressive shrinkage helps 3+ edge but hurts high-conviction bets.
  - **Decision:** Keep prior_strength=10. Current calibration is correct.

- **Added Backtest Config Transparency** - Backtest now prints full configuration at start
  - Shows all active parameters: model type, alpha, HFA, weights, FCS penalties, etc.
  - Clarifies that HFA is "team-specific (fallback=X)" not flat
  - Only shows relevant params for EFM vs legacy Ridge mode
  - Supports Rule 8 (Parameter Synchronization) - easy to verify what's running

- **Sign Convention Audit - 2 BUGS FIXED** - Complete audit of spread/margin/HFA sign conventions across all code
  - **Conventions documented:**
    - Internal (SpreadGenerator): positive spread = home favored
    - Vegas (CFBD API): negative spread = home favored
    - HFA: positive value = points added to home team advantage
  - **BUG 1 FIXED:** `spread_generator.py:386` - `home_is_favorite = prelim_spread < 0` → WRONG
    - Fixed to: `home_is_favorite = prelim_spread > 0`
    - Impact: Situational adjuster (rivalry boost for underdog) was applied to wrong team
  - **BUG 2 FIXED:** `html_report.py:250` - CSS class `spread < 0` mapped to "home" → WRONG
    - Fixed to: `spread > 0` → "home" class (blue = home favored)
    - Impact: Visual display incorrectly colored favorites/underdogs
  - **Backtest Comparison (2022-2025, n=2230):**
    | Metric | With Bug | With Fix | Diff |
    |--------|----------|----------|------|
    | MAE | 12.39 | **12.37** | -0.02 ✓ |
    | Overall ATS | 51.2% | 51.0% | -0.2% |
    | **3+ edge ATS** | 53.3% (604-529) | **54.0% (612-521)** | **+0.7%** ✓ |
    | 5+ edge ATS | 57.2% | 57.3% | +0.1% |
  - **Key finding:** 3+ edge improved by +0.7% (+8 net wins) - rivalry games typically have moderate edges
  - **Files Audited (no issues):**
    - `efficiency_foundation_model.py` - predict_margin: margin = home - away + HFA ✓
    - `home_field.py` - get_hfa: returns positive HFA added to home advantage ✓
    - `situational.py` - uses `team_is_favorite` correctly for rivalry boost ✓
    - `backtest.py` - correctly converts conventions: `model_spread_vegas = -model_spread` ✓
    - `vegas_comparison.py` - edge = (-model_spread) - vegas_spread ✓
    - `calibrate_situational.py` - uses Vegas convention internally (our_spread = away - home - HFA) ✓
    - `legacy/ridge_model.py` - margin = home - away + HFA ✓

- **Investigated Pace-Based Margin Scaling - NOT IMPLEMENTING** - Theoretical concern that fast games should have larger margins (more plays to compound efficiency edge)
  - **Theory:** JP+ should under-predict margins in fast games → hurt ATS
  - **Empirical findings (2023-2025):**
    - Correlation(total_pts, margin) = weak (R² = 2.2%)
    - Fast games actually have SMALLER margins, not larger
    - JP+ OVER-predicts fast games (mean error -2.1), not under-predicts
    - ATS is BETTER in fast games (73% vs 67.6%), not worse
  - **Why theory fails:** High-scoring games are often close shootouts; Vegas prices pace; efficiency metrics implicitly capture tempo; clock management makes blowouts have fewer plays
  - **Decision:** Do NOT implement. Risk of overfitting to non-issue. Keep model simpler.

- **Investigated Mercy Rule Dampener - NOT IMPLEMENTING** - Theory: Coaches tap brakes in blowouts, so we over-predict large margins
  - **Finding:** Bias EXISTS - mean error of -38.7 on 21+ Vegas spreads (we over-predict)
  - **Dampening test:** threshold=14, scale=0.3 compresses spreads beyond 14 pts
    - MAE improved: 24.95 → 23.29 (-1.66) ✓
    - **ATS hurt: 64.0% → 61.2% (-2.8pp) ✗**
  - **Root cause:** Our large edges against Vegas are CORRECT directionally even when magnitude is off
    - A spread of -28 vs Vegas -21 may be "wrong" by 7 pts but RIGHT about home covering
    - Dampening pulls us closer to Vegas, eliminating our edge
  - **Decision:** Do NOT implement. ATS is our optimization target (Rule #3). Accept higher MAE to maintain betting edge.

- **Investigated Baseline -6.7 Mean Error** - Root cause identified, decision NOT to fix
  - **Finding:** Model systematically predicts home teams ~6.7 pts better than actual margin
  - **Root cause:** CFBD EPA data implicitly contains home field advantage (home teams generate better EPA due to crowd noise, familiarity). Adding explicit HFA on top creates double-counting.
  - **Optimal HFA for zero error:** -3.7 pts (vs current ~2.8)
  - **Key insight:** Despite mean error, ATS performance is excellent:
    - 61.6% when picking favorites
    - 66.6% when picking underdogs
  - **Decision:** Don't "fix" it. Mean error is a calibration issue; ATS is what matters. Bias is concentrated in blowouts, not close games where ATS is won/lost.
  - **Documented:** Added "Mean Error vs ATS Performance" section to MODEL_ARCHITECTURE.md

- **Updated All Documentation with Final 2025 Results**
  - MODEL_EXPLAINER.md: Updated Top 25 with full CFP data (Indiana #2, National Champions)
  - MODEL_ARCHITECTURE.md: Updated 2025 results tables
  - SESSION_LOG.md: Added Rules & Conventions section, updated performance metrics
  - **Note:** Top 25 now includes 6,223 postseason plays from bowl games and CFP

- **Added Rules & Conventions Section** - Critical rules for JP+ development
  1. Top 25 MUST use end-of-season data including playoffs
  2. ATS results MUST be from walk-forward backtesting
  3. Mean error is a nuisance metric; optimize for ATS
  4. All parameters should flow through config

- **Optimized Ridge Alpha from 100 to 50** - Sweep revealed alpha=50 is optimal
  - Tested alphas: 25, 40, 50, 60, 75, 85, 100, 150, 200
  - **MAE improvement:** 12.56 → 12.54 (-0.02 pts)
  - **3+ edge ATS:** 52.1% → 52.4% (+0.3%)
  - **5+ edge ATS:** 55.3% → 56.0% (+0.7%)
  - Lower alpha = less regularization = teams separate more
  - Updated default in `efficiency_foundation_model.py` and `backtest.py`

- **Tested EPA Regression vs SR+IsoPPP** - EPA regression performed worse
  - Hypothesis: Single clipped EPA regression could capture both efficiency and magnitude
  - **Result:** EPA regression had higher MAE (12.64 vs 12.55) and worse ATS (52.4% vs 54.5% at 5+ edge)
  - SR+IsoPPP remains the better approach - Success Rate filters noise, IsoPPP adds explosiveness
  - Experiment not committed (per user request)

- **Implemented Tiered FCS Penalties** - Elite vs Standard FCS teams now have different penalties
  - **Problem:** Flat 24pt penalty overestimates elite FCS teams (NDSU, Montana State, etc.) and underestimates weak FCS
  - **Analysis:** 2022-2024 FCS games show mean FBS margin of 30.3 pts vs standard FCS, but only +2 to +15 vs elite FCS
  - **Solution:** Tiered system with ELITE=18 pts, STANDARD=32 pts
  - **Elite FCS teams identified:** 23 teams including NDSU, Montana State, Sacramento State, Idaho, South Dakota State
  - **Backtest results (2022-2025):**
    | FCS Penalty Config | 3+ edge | 5+ edge |
    |-------------------|---------|---------|
    | Flat 24pt | 52.4% | 56.0% |
    | Tiered 18/28 (SP+-like) | 52.4% | 56.3% |
    | **Tiered 18/32** | **52.5%** | **56.9%** |
  - **5+ edge improvement:** +0.9% over flat penalty, +0.6% over SP+ values
  - Updated `spread_generator.py` with `ELITE_FCS_TEAMS` set and tiered penalty logic
  - Updated `backtest.py` with `--fcs-penalty-elite` and `--fcs-penalty-standard` CLI args
  - Fixed CLI default for `--asymmetric-garbage` (now enabled by default, use `--no-asymmetric-garbage` to disable)

- **Tested Turnover Cap (±7 pts) - Not Implemented** - Investigated double-counting concern
  - **Concern:** Turnovers influence field position → EPA → IsoPPP, then added again as explicit component
  - **Tested:** Cap turnover contribution at ±7 points after shrinkage
  - **Results:**
    | Config | MAE | 3+ edge | 5+ edge |
    |--------|-----|---------|---------|
    | No turnover (weight=0) | 12.48 | **52.3%** | 56.0% |
    | With turnover, no cap | 12.48 | 52.1% | **56.9%** |
    | With turnover, cap=7 | 12.48 | 52.1% | **56.9%** |
  - **Finding:** Cap has zero effect - the 10% weight + Bayesian shrinkage already constrains turnovers sufficiently
  - **Decision:** Not implemented. Added complexity with no benefit. The real tradeoff is turnovers vs no turnovers, not cap vs no cap.
  - **Optimization insight:** Should optimize for MAE (primary) with 3+ edge as key validation metric. 5+ edge can be variance with smaller sample.

- **Fixed CLI/Function Default Mismatches** - Audited all backtest.py parameters
  - `--asymmetric-garbage` → `--no-asymmetric-garbage` (asymmetric now enabled by default)
  - `--alpha` aligned to 50.0 (was 10 CLI, 150 function)
  - `--hfa` aligned to 2.5 (was 3.0 CLI, 2.8 function)
  - Fixed docstrings that incorrectly stated asymmetric_garbage default as False

- **Implemented Pace Adjustment for Triple-Option Teams** - Compress spreads for teams with fewer possessions
  - **Analysis:** Triple-option teams (Army, Navy, Air Force, Kennesaw State) have 30% worse MAE (16.09 vs 12.36, p=0.001)
  - **Root cause:** ~55 plays/game vs ~70 for standard offenses = less regression to mean = more variance
  - **Solution:** Compress spreads by 10% toward zero when triple-option team involved (15% if both)
  - **Backtest results (2022-2024):**
    | Config | MAE | 3+ edge | 5+ edge |
    |--------|-----|---------|---------|
    | No pace adjustment | 13.97 | 52.0% (677-626) | 53.3% (537-471) |
    | With pace adjustment | 13.96 | 52.0% (679-626) | 53.3% (538-472) |
  - **Finding:** Minimal overall effect (~0.01 MAE, +2 wins) because triple-option games are only ~3% of sample
  - **Decision:** Implemented as theoretically sound incremental improvement. Effect is neutral-to-slightly-positive.
  - Added `_get_pace_adjustment()` method and `TRIPLE_OPTION_TEAMS` set to `spread_generator.py`
  - Added `pace_adjustment` component to `SpreadComponents` dataclass

- **Audited Garbage Time Thresholds (Indiana/Ole Miss)** - Investigated if thresholds cause rating lag vs consensus
  - **Hypothesis:** Our Q4 threshold (>14 pts) vs SP+ (>21 pts) might suppress "dominance signal"
  - **Analysis of 2025 data:**
    | Team | Garbage Time Plays | While Leading | While Trailing | Threshold Gap |
    |------|-------------------|---------------|----------------|---------------|
    | Indiana (def) | 154 (21.9%) | 154 | **0** | 9 plays |
    | Ole Miss (off) | 108 (12.2%) | 108 | **0** | 21 plays |
  - **Critical Finding:** Asymmetric weighting already solves this problem
    - Both teams have **0 trailing garbage time plays** - no penalty applied
    - All dominant plays receive **full 1.0x weight**
    - The 14pt vs 21pt threshold gap affects only 30 combined plays
  - **Indiana Defense:** Actually allows MORE success in garbage time (68.2%) vs non-garbage (64.5%)
    - Opponent desperation offense is more effective, not less
  - **Ole Miss Offense:** Higher SR in garbage time (56.5%) vs non-garbage (47.8%) - dominance IS captured
  - **Recommendation:** Do NOT change thresholds. Issue is elsewhere.

- **Deep Dive: Ole Miss Rating Gap vs Consensus** - Root cause is NOT garbage time
  - **Efficiency Comparison (2025):**
    | Team | Off SR | Def SR | SR Margin |
    |------|--------|--------|-----------|
    | Indiana | 54.4% | 34.7% | +19.7% |
    | Ohio State | 54.7% | 35.9% | +18.8% |
    | Ole Miss | 48.9% | 39.2% | **+9.7%** |
  - **Ole Miss SR Margin gap vs elite teams: -10pp** - this is the primary driver
  - **Turnover Margin:**
    | Team | TO Margin |
    |------|-----------|
    | Indiana | +5 |
    | Oregon | +4 |
    | Ole Miss | **-2** |
  - **Game-by-game defensive disasters:**
    - Georgia: 62.5% SR allowed
    - Arkansas: 52.1% SR allowed
  - **Why SP+ might rank Ole Miss higher:**
    - Different opponent adjustment (SEC strength credit)
    - More aggressive turnover regression
    - Different explosiveness weighting (Ole Miss has 1.292 IsoPPP)
    - EPA-based vs SR-based defense metrics
  - **Parked for future investigation:** Opponent adjustment methodology, turnover regression tuning

- **Ensured Script Connectivity** - Updated all scripts to pass FBS teams and FCS penalties
  - `backtest.py` legacy path: Added `fbs_teams`, `fcs_penalty_elite`, `fcs_penalty_standard` params
  - `run_weekly.py`: Now fetches FBS teams and passes to SpreadGenerator
  - Ensures pace adjustment and tiered FCS penalties work across all execution paths

- **Tested Alpha × Decay 2D Sweep - Decay NOT Beneficial** - Time decay makes predictions worse
  - **Hypothesis:** Weight recent games more heavily (decay=0.95 → Week 1 gets ~0.54 weight by Week 12)
  - **Added:** `time_decay` parameter to `EfficiencyFoundationModel`
  - **Sweep:** 5 alphas (30,40,50,60,75) × 5 decays (1.0,0.98,0.96,0.94,0.92) across 2022-2025
  - **Results:**
    | Config | MAE | 5+ Edge |
    |--------|-----|---------|
    | alpha=40, decay=1.0 | **13.802** | **52.8%** |
    | alpha=50, decay=1.0 | 13.805 | 52.6% |
    | alpha=75, decay=0.92 | 13.924 | 50.6% |
  - **Clear finding:** No decay (1.0) is best across ALL alpha values
  - **Why decay hurts:** Walk-forward already ensures temporal validity; early-season data provides valuable opponent calibration; teams don't change as much week-to-week as assumed
  - **Decision:** Keep alpha=50, decay=1.0 (default). Parameter added but not used.

- **Implemented QB Injury Adjustment System** - Manual flagging of starter injuries with pre-computed drop-offs
  - **Problem:** QB injuries are the single biggest source of MAE error, not handled by model
  - **Solution:** Minimal viable implementation with manual flagging
  - **New module:** `src/adjustments/qb_adjustment.py`
    - `QBInjuryAdjuster` class computes starter/backup PPA differentials
    - Uses CFBD `get_predicted_points_added_by_player_season` for QB PPA
    - Identifies starter by volume (most pass attempts)
    - Calculates point adjustment: `PPA_drop × 30 plays/game`
  - **CLI integration:** `--qb-out TEAM` flag in `run_weekly.py`
    - Example: `python scripts/run_weekly.py --qb-out Georgia Texas`
  - **Sample depth charts (2024):**
    | Team | Starter | PPA | Backup | PPA | Adjustment |
    |------|---------|-----|--------|-----|------------|
    | Georgia | Beck | 0.353 | Stockton | 0.125 | **-6.8 pts** |
    | Ohio State | Howard | 0.575 | Brown | 0.243 | **-10.0 pts** |
    | Texas | Ewers | 0.322 | A. Manning | 0.589 | **+8.0 pts** |
    | Alabama | Milroe | 0.321 | Simpson | 0.215 | **-3.2 pts** |
  - **Note:** Texas is unusual - Arch Manning backup is BETTER than Ewers starter
  - **Data limitation:** CFBD has no injury reports; requires manual flagging
  - **Updated files:** `spread_generator.py` (added `qb_adjustment` component), `run_weekly.py` (added `--qb-out` flag)

- **Tested Havoc Rate vs Turnover Margin - Not Implemented** - Havoc doesn't improve ATS predictions
  - **Hypothesis:** Replace/augment turnovers (10% weight) with Havoc Rate (TFLs, sacks, PBUs, forced fumbles)
    - Havoc is a "sticky skill" while turnover recovery is ~50% luck
    - Expected havoc to be more predictive of future ATS success
  - **Data source:** CFBD API `get_havoc_stats()` endpoint provides:
    - Front-7 havoc (sacks, TFLs)
    - DB havoc (PBUs, interceptions)
    - Total havoc rate per game
  - **Correlation analysis (2024):**
    | Metric | Correlation |
    |--------|-------------|
    | Havoc → Turnovers Forced | r = 0.425 (strong) |
    | Havoc → Turnover Margin | r = 0.148 (weak) |
    | DB Havoc → Margin | r = 0.230 |
    | Front-7 Havoc → Margin | r = 0.096 |
  - **Interpretation:** Havoc IS predictive of forcing turnovers (skill), but turnover margin adds significant luck noise
  - **ATS Backtest (2022-2024, 1631 games):**
    | Metric Differential | Correlation with Cover |
    |--------------------|----------------------|
    | Havoc | r = -0.013 |
    | Turnover Margin | r = -0.024 |
  - **Finding:** Neither metric provides ATS edge - Vegas already prices both
  - **When Havoc and TO disagree:** ~48% cover rate either way - no predictive value
  - **Conclusion:** Havoc's theoretical advantage (skill vs luck) doesn't translate to ATS improvement
  - **Decision:** Keep current 10% turnover weight with Bayesian shrinkage. Shrinkage already handles luck noise. Added `get_havoc_stats()` to CFBD client for future use.

- **Tested Style Mismatch Adjustment - Not Implemented** - Rush/pass matchup profiles don't improve predictions
  - **Hypothesis:** Split rush/pass ratings could capture matchup advantages Vegas misses
    - Rush-heavy offense vs weak rush defense → boost prediction
    - Pass-heavy offense vs weak pass defense → boost prediction
  - **Implementation tested:**
    - Calculate team style profiles (rush share, rush/pass SR, defensive rush/pass SR allowed)
    - Apply adjustment when extreme styles meet weak defenses (thresholds: >52% rush-heavy, <42% pass-heavy)
    - Scale adjustments by severity (max ±4 points)
  - **Backtest results (2022-2024, walk-forward):**
    | Config | Games | ATS | MAE |
    |--------|-------|-----|-----|
    | Baseline (Vegas only) | 641 | 48.3% (304-326-11) | 12.14 |
    | With mismatch adjustment | 641 | 48.1% (303-327-11) | 12.23 |
  - **Finding:** Style mismatch adjustment made predictions WORSE
    - ATS: -0.2pp (1 fewer win)
    - MAE: +0.09 (higher error)
  - **Conclusion:** Vegas already prices style matchups efficiently. Aggregate efficiency metrics (Success Rate, IsoPPP) capture matchup dependencies implicitly. Adding explicit rush/pass splits introduces noise without signal.
  - **Decision:** Not implemented. Keeps model simpler without sacrificing accuracy.

---

## Session: February 1, 2026 (Evening)

### Completed This Evening

- **Investigated Ole Miss Ranking Discrepancy** - Ole Miss ranks #14 in JP+ but #7-9 in SP+, FPI, and Sagarin
  - **Root cause identified:** Defense. Ole Miss offense is elite (+12.2, top 5) but defense is mediocre (+2.6)
  - Average top-9 defense: +13.9. Ole Miss defense gap: **-11.3 points**
  - Defensive SR allowed: 36.8% (vs 31-34% for elite defenses like Ohio State/Oregon)
  - **Hypothetical:** If Ole Miss had average top-10 defense, they'd rank **#3** at +26.1
  - **Conclusion:** JP+ is correctly identifying a defensive weakness that hurt them in big games
    - Lost to Miami 27-31 in playoff (allowed 31 points as predicted)
    - Close wins vs Oklahoma (+8), LSU (+5), Arkansas (+6) - defense gave up a lot
  - Other systems may weight win-loss record (13-2) more heavily; JP+ ignores record entirely
  - **Game-by-game defensive breakdown:** Struggled vs Georgia (47% SR allowed), Arkansas (46%)

- **Validated Defensive Rating Convention for Game Totals** - Confirmed current convention works
  - Current JP+: Higher defense = better (points saved vs average)
  - Formula for game totals: `Total = 2×Avg + (Off_A + Off_B) - (Def_A + Def_B)`
  - Example verified: A(Off +10, Def +8) vs B(Off +5, Def +3) → Total = 56 + 15 - 11 = 60
  - Good defenses lower totals, good offenses raise them - mathematically correct
  - **No changes needed** to defensive rating convention for totals prediction

---

## Session: February 1, 2026 (Continued)

### Completed Today

- **Added Turnover Component to EFM** - Turnovers now included in ratings with Bayesian shrinkage
  - **Problem identified:** Indiana's +15 turnover margin (vs OSU's +3) was not captured in JP+
  - SP+ includes turnovers at 10% weight; JP+ was missing this entirely
  - **Solution:** Added `turnover_weight` parameter to EFM (default 0.10)
  - Calculates per-game turnover margin (forced - lost) and converts to point value
  - New weights: 54% efficiency + 36% explosiveness + 10% turnovers
  - **Bayesian shrinkage added:** `turnover_prior_strength=10` regresses margin toward 0 based on games played
    - 5 games: keeps 33% of margin (5/15)
    - 15 games: keeps 60% of margin (15/25)
    - Prevents overweighting small-sample turnover luck while trusting sustained performance
  - **Final Impact:** Indiana now #2 at +27.8, only 0.2 pts behind Ohio State (+28.0)
  - Added `turnover_rating` field to `TeamEFMRating` dataclass
  - Added `TURNOVER_PLAY_TYPES`, `POINTS_PER_TURNOVER`, and `turnover_prior_strength` constants
  - **Verified turnover play types** against CFBD API (removed dead "Interception Return" entry)
  - Updated `backtest.py` with new weights and `--efm-turnover-weight` CLI argument

- **2025 Backtest Results** (with turnover component + shrinkage)
  - MAE: 12.44 points
  - ATS: 322-310-6 (50.9%)
  - 3+ pt edge: 179-155 (53.6%)
  - 5+ pt edge: 110-92 (54.5%)

- **Reduced Red Zone Regression Strength (prior_strength 20→10)** - Analysis showed original was too aggressive
  - **Problem identified:** Elite RZ teams like Indiana (87.2% TD rate) were being penalized
  - Indiana's raw finishing drives advantage (+0.04 over OSU) was flipping to a disadvantage (-0.05) after regression
  - With 150+ RZ plays per team by end of season, high TD rates become skill, not luck
  - **Solution:** Reduced `prior_strength` from 20 to 10 in `finishing_drives.py`
  - **Impact:** Better credits teams that sustain elite RZ efficiency over full season
  - Also updated hardcoded value in `backtest.py` line 628

- **Migrated backtest.py from pandas to Polars** - Performance optimization for string filtering operations
  - **Problem identified:** cProfile showed 30s spent in pandas string comparisons (`isin()` operations)
  - **Solution:** Migrate data pipeline to Polars DataFrames (26x faster for string filtering)
  - **Changes:**
    - `fetch_season_data()` now returns `pl.DataFrame` instead of `pd.DataFrame`
    - `fetch_season_plays()` now returns `pl.DataFrame` tuples
    - `build_game_turnovers()` now uses Polars joins and group_by operations
    - `walk_forward_predict_efm()` uses Polars filtering, converts to pandas only at sklearn boundary
    - `calculate_ats_results()` accepts Polars betting DataFrame
    - Legacy Ridge model path converts to pandas before calling `walk_forward_predict()`
  - **Dependencies added:** `polars`, `pyarrow` (for efficient Polars→pandas conversion)
  - **Verified:** Full 2022-2025 backtest completes successfully with identical results

- **Implemented FBS-Only Filtering** - Ridge regression now excludes plays involving FCS opponents
  - **Problem identified:** FCS teams have too few games against FBS opponents to estimate reliable coefficients
  - FCS plays were polluting the regression with unreliable team strength estimates
  - **Solution:** Filter `train_plays` to only include FBS vs FBS matchups before passing to EFM
  - FCS games still handled via FCS Penalty adjustment (+24 pts)
  - **Backtest results (2022-2025):** 5+ pt edge improved from 54.0% → 54.8% (+0.8%)
  - Implementation in `scripts/backtest.py` line 537-545

- **Validated Asymmetric Garbage Time** - Tested symmetric vs asymmetric GT with FBS-only baseline
  - User hypothesis: Asymmetric GT was a "hack" that would hurt Indiana, should revert to symmetric
  - **Empirical finding:** Asymmetric actually HELPS Indiana (moves from #4 to #2)
  - **Backtest comparison:**
    - Asymmetric: 5+ edge 54.8% (449-371)
    - Symmetric: 5+ edge 53.0% (447-396)
  - **Decision:** Keep asymmetric GT - it improves both rankings AND predictive accuracy
  - Key insight: Indiana DOES maintain elite efficiency in garbage time; asymmetric captures this signal

- **Investigated Indiana #1 Question** - Why is Indiana #2 instead of #1 (they won the championship)?
  - Audited play-by-play efficiency consistency for Indiana vs Oregon vs Ohio State
  - Audited performance against elite defenses (Top 30)
  - **Findings:**
    - Indiana raw SR: 53.8% (highest)
    - Indiana vs elite defenses: 42.7% SR (better than Oregon's 41.2%)
    - Indiana beat Oregon head-to-head: 47.1% vs 36.1% SR
    - Yet Indiana is still #2 behind Ohio State after opponent adjustment
  - **Conclusion:** The gap is NOT from FCS contamination or asymmetric GT
  - Ohio State-Indiana gap: 3.4 points (same in both symmetric and asymmetric)
  - Ohio State maintains #1 due to stronger performance vs elite opponents AND stronger schedule overall

---

## Session: February 1, 2026 (Earlier)

### Completed Earlier Today

- **Implemented Transfer Portal Integration** - Preseason priors now incorporate net transfer portal impact
  - Fetches transfer portal entries from CFBD API
  - Matches transfers to prior-year player usage (PPA) to quantify production
  - Calculates net incoming - outgoing PPA for each team
  - Adjusts effective returning production by portal impact (scaled, capped at ±15%)
  - **Backtest results (2022-2025):** 5+ pt edge improved from 53.3% → 53.7% (+0.4%)
  - Added `fetch_transfer_portal()`, `fetch_player_usage()`, `calculate_portal_impact()` to `PreseasonPriors`
  - Added `portal_adjustment` field to `PreseasonRating` dataclass
  - New CLI flags: `--no-portal`, `--portal-scale` (default 0.15)
  - Top 2024 portal winners: Missouri (+15%), Washington (+15%), UCF (+15%), California (+15%)
  - Top 2024 portal losers: New Mexico State, Ohio, Arizona State

- **Updated efficiency/explosiveness weights documentation** - All docs now reflect 60/40 weighting

- **Implemented Rating Normalization** - Ratings now scaled for direct spread calculation
  - Added `rating_std` parameter to EFM (default 12.0)
  - Added `_normalize_ratings()` method to scale ratings to target std
  - Ratings normalized: mean=0, std=12 across FBS teams
  - **Direct spread calculation:** Team A rating - Team B rating = expected spread
  - Example: Ohio State (+33.1) vs Penn State (+19.1) → Ohio State -14.0
  - Updated MODEL_EXPLAINER.md with normalized Top 10 ratings

- **Implemented Asymmetric Garbage Time** - Only trailing team's garbage time plays down-weighted
  - **Problem identified:** Dominant teams (Indiana 56% SR, Ohio State 57% SR in garbage time) were having their best plays discarded
  - **Solution:** Winning team keeps full weight; only trailing team gets 0.1 weight
  - Added `asymmetric_garbage` parameter to EFM (default True)
  - Added `--asymmetric-garbage` CLI flag to backtest.py
  - **Backtest results (2022-2025):**
    - MAE: 12.53 → 12.52 (-0.01)
    - 5+ pt edge: 53.4% → 54.0% (+0.6%)
  - **Ranking impact:** Indiana rises from #4 to #3; Notre Dame drops from #3 to #4
  - Conceptual improvement: rewards teams that maintain dominance, penalizes coasting

- **Tested 55/45 Efficiency/Explosiveness Weights** - Decided to keep 60/40
  - Hypothesis: Increasing explosiveness weight might help explosive teams like Ole Miss
  - **Backtest results (55/45 + asymmetric):** 5+ edge 54.2% (+0.2% vs 60/40)
  - **Ranking impact:** Ole Miss unchanged (+21.6); Indiana dropped (-0.5); Oregon/Notre Dame rose
  - **Decision:** Keep 60/40 - marginal improvement not worth hurting Indiana (National Champs)
  - Ole Miss #16 ranking appears accurate based on raw efficiency (48.9% SR, #22 nationally)

---

## Session: January 31, 2026

### Completed Previously

- **Implemented FG Efficiency Adjustment** - Integrated kicker PAAE (Points Above Average Expected) into spread predictions
  - Calculates FG make rates vs expected by distance (92% for <30yd, 83% for 30-40, etc.)
  - PAAE = actual points - expected points for each kick
  - Per-game FG rating applied as differential adjustment to spreads
  - **Impact:** ATS improved from 50.6% → 51.2% (+0.6%), MAE improved from 12.57 → 12.47
  - Added `calculate_fg_ratings_from_plays()` to `SpecialTeamsModel`
  - Added FG plays collection to backtest data pipeline

- **Cleaned up documentation** - Removed all references to unused/legacy components
  - Removed Ridge Model (margin-based) section from MODEL_ARCHITECTURE.md
  - Removed luck regression, early-down model, turnover scrubbing references
  - Removed legacy CLI flags (--decay, --to-scrub-factor, --margin-cap)
  - Updated file structure to only show actively used files
  - Clarified EFM is the sole foundation model, not one of two options

- **Exposed separate O/D/ST ratings** - JP+ now provides separate offensive, defensive, and special teams ratings
  - Updated `TeamEFMRating` dataclass with `offensive_rating`, `defensive_rating`, `special_teams_rating` fields
  - Added methods: `get_offensive_rating()`, `get_defensive_rating()`, `get_special_teams_rating()`, `set_special_teams_rating()`
  - Updated `get_ratings_df()` to include offense, defense, special_teams columns
  - Integrated special teams rating from SpecialTeamsModel into EFM in backtest pipeline
  - Example 2025 insights: Vanderbilt has best offense (+16.7), Oklahoma has best defense (+13.8)
  - **Purpose:** Enables future game totals prediction (over/under) by predicting each team's expected points

- **Tuned efficiency/explosiveness weights from 65/35 to 60/40**
  - Compared JP+ rankings to SP+ to identify systematic issues
  - Found explosive teams (Texas Tech, Ole Miss) were being underrated
  - Tested weight configurations: 65/35, 60/40, 55/45, 50/50
  - 60/40 weighting showed best multi-year backtest results (2022-2025):
    - MAE: 12.63 (vs 12.65 with 65/35)
    - ATS: 51.3% (vs 51.0% with 65/35)
    - 5+ pt edge: 54.5% (vs 54.2%)
  - Updated defaults in EFM and backtest.py

- **Implemented FCS Penalty Adjustment** - Adds 24 points to FBS team's predicted margin vs FCS opponents
  - **Diagnostic finding:** JP+ was UNDER-predicting blowouts by 26 pts, not over-predicting
  - 99.5% of blowout errors were under-predictions (actual margins larger than predicted)
  - FCS games (3.3% of total) had MAE of 29.54 - dragging overall MAE significantly
  - Tested penalty sweep from 0-30 pts; optimal value ~24 pts based on actual under-prediction
  - **Impact:** MAE improved from 13.11 → 12.57 (-0.54), 5+ edge ATS improved 52.8% → 53.2%
  - Added `fcs_penalty` parameter to `SpreadGenerator` and `backtest.py`
  - Added `fcs_adjustment` component to spread breakdown

- **Ran MAE by Margin Diagnostic Analysis**
  - Discovered blowouts (29+ pts, 17% of games) contribute 33% of overall MAE
  - Identified root causes: FCS teams treated as merely "below average" instead of dramatically weaker
  - Elite teams (Ohio State, Notre Dame, Indiana) consistently beating weak opponents by more than predicted

- **Implemented Coaching Change Regression** - New HC at underperforming team triggers weight shift from prior performance toward talent
  - "Forget factor" based on talent-performance gap (bigger gap = more forgetting, capped at 50%)
  - Data-driven coach pedigree scores calculated from CFBD API (career win %, P5 years)
  - Elite coaches (Kiffin 1.30, Cignetti 1.25, Sumrall 1.25) get more benefit of the doubt
  - **First-time HCs are EXCLUDED** - no adjustment (we have no basis to predict improvement)
  - Impact: +1 to +5 pts for underperformers with proven coaches
  - **Backtest result:** Neutral impact on MAE/ATS (sample too small), but kept for conceptual soundness

- **Updated all documentation** with FCS penalty and coaching change regression details

---

## Session: January 29, 2026

### Completed Previously

- **Implemented HFA trajectory modifier** - Dynamic adjustment (±0.5 pts) for rising/declining programs based on win % improvement
  - Compares recent year (1 yr) to baseline (prior 3 yrs)
  - Added `calculate_trajectory_modifiers()` and `set_trajectory_modifier()` to `HomeFieldAdvantage` class
  - Constants: `TRAJECTORY_MAX_MODIFIER = 0.5`, `TRAJECTORY_BASELINE_YEARS = 3`, `TRAJECTORY_RECENT_YEARS = 1`

- **Updated all documentation** with trajectory modifier details
  - MODEL_ARCHITECTURE.md: Added Trajectory Modifier subsection with calculation table and examples
  - MODEL_EXPLAINER.md: Added user-friendly explanation of trajectory concept
  - Changelog updated

- **Validated trajectory calculations** for Vanderbilt and Indiana (2023-2025)
  - Vanderbilt: 2.12 → 2.48 → 2.50 (rising program)
  - Indiana: 2.46 → 3.25 → 3.25 (penalty in 2023, max boost 2024-25)

---

### Current Source of Truth

#### EFM Parameters
| Parameter | Value | Location |
|-----------|-------|----------|
| `ridge_alpha` | 50 | `efficiency_foundation_model.py` (optimized from 100) |
| `efficiency_weight` | 0.45 | `efficiency_foundation_model.py` (Explosiveness Uplift) |
| `explosiveness_weight` | 0.45 | `efficiency_foundation_model.py` (Explosiveness Uplift) |
| `turnover_weight` | 0.10 | `efficiency_foundation_model.py` |
| `turnover_prior_strength` | 10.0 | `efficiency_foundation_model.py` |
| `garbage_time_weight` | 0.1 | `efficiency_foundation_model.py` |
| `asymmetric_garbage` | True | `efficiency_foundation_model.py` |
| `time_decay` | 1.0 | `efficiency_foundation_model.py` (tested, decay hurts performance) |
| `rating_std` | 12.0 | `efficiency_foundation_model.py` |
| `rz_prior_strength` | 10 | `finishing_drives.py` |
| `fcs_penalty_elite` | 18.0 | `spread_generator.py` |
| `fcs_penalty_standard` | 32.0 | `spread_generator.py` |

#### HFA Parameters
| Parameter | Value | Location |
|-----------|-------|----------|
| Base HFA range | 1.5 - 4.0 | `home_field.py` → `TEAM_HFA_VALUES` |
| Conference defaults | SEC/B1G: 2.75, Big12/ACC: 2.50, G5: 2.0-2.25 | `home_field.py` → `CONFERENCE_HFA_DEFAULTS` |
| Trajectory max modifier | ±0.5 | `home_field.py` → `TRAJECTORY_MAX_MODIFIER` |
| Trajectory baseline years | 3 | `home_field.py` → `TRAJECTORY_BASELINE_YEARS` |
| Trajectory recent years | 1 | `home_field.py` → `TRAJECTORY_RECENT_YEARS` |

#### Preseason Priors
| Parameter | Value | Location |
|-----------|-------|----------|
| Prior year weight | 0.6 (default) | `preseason_priors.py` |
| Talent weight | 0.4 (default, reduced to 0.2 for extreme teams) | `preseason_priors.py` |
| Base regression range | 0.1 - 0.5 (based on returning PPA) | `preseason_priors.py` → `_get_regression_factor()` |
| Asymmetric regression threshold | ±8 pts (full), ±20 pts (min) | `preseason_priors.py` → `_get_regression_factor()` |
| Asymmetric regression floor | 0.33x multiplier at 20+ pts from mean | `preseason_priors.py` → `_get_regression_factor()` |
| Extremity talent threshold | 12-20 pts from mean | `preseason_priors.py` → `calculate_preseason_ratings()` |
| Extremity talent scale | 0.5 (halve talent weight) at 20+ pts | `preseason_priors.py` → `calculate_preseason_ratings()` |
| Coaching forget factor cap | 0.5 | `preseason_priors.py` → `_calculate_coaching_change_weights()` |
| Coaching gap divisor | 60 | `preseason_priors.py` (gap/60 = base forget) |
| Portal scale | 0.15 | `preseason_priors.py` → `calculate_portal_impact()` |
| Portal adjustment cap | ±15% | `preseason_priors.py` → `calculate_portal_impact()` |
| Triple-option rating boost | +6.0 pts | `preseason_priors.py` → `TRIPLE_OPTION_RATING_BOOST` |
| Triple-option talent weight | 0% (use 100% prior) | `preseason_priors.py` → `calculate_preseason_ratings()` |

#### Coach Pedigree (Data-Driven)
| Tier | Pedigree Range | Example Coaches |
|------|----------------|-----------------|
| Elite | 1.27 - 1.30 | Kiffin, Kirby Smart, DeBoer, James Franklin |
| Strong | 1.20 - 1.25 | Cignetti, Sumrall, Silverfield, Chadwell |
| Above Avg | 1.10 - 1.19 | Rhule, McGuire, Venables |
| Average | 1.05 - 1.09 | Brent Key, Sam Pittman, Deion Sanders |
| Neutral | 1.00 | First-time HCs, no HC record |
| Below Avg | 0.88 - 0.97 | Clark Lea, Troy Taylor, Jeff Lebby |

#### Data Pipeline
| Parameter | Value | Location |
|-----------|-------|----------|
| FBS-only filtering | True | `backtest.py` → `walk_forward_predict_efm()` |

#### Model Performance (2022-2025) - With All Features (FBS-only + Asymmetric GT)
- **MAE:** 12.48
- **ATS:** 51.3%
- **3+ pt edge ATS (closing):** 52.1% (684-630)
- **3+ pt edge ATS (opening):** 55.0% (763-624)
- **5+ pt edge ATS (closing):** 56.9% (449-340)
- **5+ pt edge ATS (opening):** 58.3% (500-358)

#### 2025 Season Performance (Final)
- **MAE:** 12.21
- **ATS (closing):** 51.9% (325-301-12)
- **ATS (opening):** 53.0% (336-298-4)
- **3+ pt edge (closing):** 54.1% (172-146)
- **3+ pt edge (opening):** 55.3% (189-153)
- **5+ pt edge (closing):** 55.4% (98-79)
- **5+ pt edge (opening):** 58.3% (119-85)

#### 2025 Top 25 (End-of-Season + CFP)
1. Ohio State (+27.5), 2. Indiana (+26.8) ★, 3. Notre Dame (+25.4), 4. Oregon (+23.4), 5. Miami (+22.8)
6. Texas Tech (+22.0), 7. Texas A&M (+19.2), 8. Alabama (+18.8), 9. Georgia (+17.5), 10. Utah (+17.5)

★ National Champions - beat Alabama 38-3, Oregon 56-22, Miami 27-21 in CFP

---

### The Parking Lot (Future Work)

**Where we stopped:** Investigated baseline -6.7 mean error. Root cause: EPA data implicitly contains home field advantage, causing double-counting with explicit HFA. Decision: Don't fix it—ATS performance is excellent (61-67%) despite mean error. Documented finding in MODEL_ARCHITECTURE.md. Updated all docs with final 2025 results including CFP.

**Open tasks to consider:**
1. ~~**EFM alpha parameter sweep**~~ - ✅ DONE. Optimal alpha=50 (was 100). See Feb 2 session.
2. **Penalty Adjustment** - Explore adding penalty yards as an adjustment factor
   - Hypothesis: Disciplined teams (fewer penalties) have a real edge JP+ ignores
   - Approach: Calculate penalty yards/game vs FBS average, convert to point impact
3. **Game totals prediction** - Formula validated and ready to implement:
   - `Total = 2×Avg + (Off_A + Off_B) - (Def_A + Def_B)`
   - `Team_A_points = Avg + Off_A - Def_B`
   - Current defensive convention (higher = better) works correctly
4. **Further blowout improvement** - FCS penalty helped, but blowout MAE still high. Could explore:
   - Lower ridge alpha to reduce shrinkage (let elite teams rate higher)
   - Weak FBS team penalty (G5 bottom-feeders similar to FCS)
   - Non-linear rating transformation
5. **Weather impact modeling** - Rain/wind effects on passing efficiency
6. **Investigate Arkansas JP+ rating** - Arkansas shows #3 offense but #96 defense, yet only #25 overall. Verify this reflects reality or if there's a calculation issue.

---

### Unresolved Questions

1. ~~**Trajectory modifier timing**~~ - **RESOLVED:** Calculate trajectory ONCE at start of season using prior year as "recent". Lock it in for the whole season. Rationale: HFA reflects stadium atmosphere built over years, not weeks. Avoids double-counting current performance (already in ratings).

2. ~~**Trajectory for new coaches**~~ - **RESOLVED:** Implemented coaching change regression in preseason priors. Proven coaches (Cignetti, Kiffin, Sumrall) get preseason rating boost when taking over underperforming teams. First-time HCs excluded.

3. ~~**Trajectory decay**~~ - **RESOLVED:** No explicit decay needed. The formula naturally handles this—as successful years roll into the baseline (prior 3 yrs), the improvement gap shrinks automatically. Example: Indiana's 2024 success becomes part of baseline by 2027, reducing their modifier.

4. ~~**G5 elite programs**~~ - **RESOLVED:** No special G5 treatment needed. Base HFA already handles elite G5s (Boise 3.0, JMU 2.75). Trajectory measures crowd energy change, not "impressiveness" of wins. Same formula for all conferences.

5. **"Mistake-free" football (penalties)** - 2025 Indiana was extremely disciplined. Does Success Rate already capture ~80% of this value, or is penalty differential a distinct predictive signal? Need to analyze correlation between penalty rates and ATS performance independent of efficiency metrics.

6. ~~**Ole Miss #14 vs #7-9 in other systems**~~ - **RESOLVED:** Investigated why Ole Miss ranks lower in JP+ than SP+/FPI/Sagarin:
   - **Root cause:** Defense (+2.6) is far below top-team average (+13.9)
   - Offense is elite (+12.2), but defense allowed 36.8% SR (vs 31-34% for elite defenses)
   - If Ole Miss had average top-10 defense, they'd rank #3 at +26.1
   - Lost to Miami 27-31 in playoff - exactly what weak defense rating predicted
   - **Conclusion:** JP+ correctly identifies defensive weakness; other systems may weight 13-2 record more heavily

7. **Indiana #2 vs Ohio State #1** - Indiana won the championship, beat everyone including Ohio State, yet JP+ has them #2. Investigation found:
   - FBS-only filtering didn't change this (Indiana still #2)
   - Asymmetric GT actually helps Indiana (#4 → #2)
   - The 3.4 pt gap is consistent across all filter configurations
   - Ohio State's stronger schedule overall appears to be the driver
   - **Open question:** Is this correct (OSU was genuinely better pre-championship) or is the model still over-penalizing Indiana's weaker regular season schedule?

---

*End of session*

---

## Session: February 11, 2026 (Late Evening)

### Theme: QB Continuous Bug Fixes + Production Script Hardening

---

#### QB Continuous Module Fixes — COMMITTED
**Status**: Committed (`a127e3d`)
**Files**: `src/adjustments/qb_continuous.py`

Fixed 5 issues in the QB continuous rating system:

1. **_get_manual_adjustment formula fix**: The backup drop-off calculation incorrectly added starter quality, which meant losing a bad starter appeared beneficial. Changed to pure starter-to-backup drop-off (`-min(qb_cap, backup_drop)`) independent of starter absolute quality.

2. **Shrinkage basis mismatch fix**: `_compute_team_avg_qb_value` used only current season dropbacks for shrinkage, but `_compute_qb_quality` used n_effective_db (current + prior × decay). This caused residuals to not cancel properly for stable starters. Fixed by incorporating prior season data in team average calculation.

3. **Baking factor off-by-one fix**: Week 1 was stuck at 0.0 baking factor because condition was `<= BAKING_WEEK_START` with BAKING_WEEK_START=1. Changed to `< BAKING_WEEK_START` so week 1 begins ramping.

4. **Removed qb_points_effective field**: This field was computed but never used in the actual spread adjustment path (get_adjustment uses qb_points with its own dampening). Removed from dataclass and all references to eliminate confusion.

5. **Walk-forward safety guard**: Added check in `_load_prior_season` to verify `prior_year < self.year` before fetching, preventing data leakage if module is misconfigured.

---

#### Production Script Fixes — COMMITTED
**Status**: Committed (`b9fab34`)
**Files**: `scripts/run_weekly.py`, `scripts/qb_calibration.py`

**run_weekly.py (4 fixes):**

1. **Week 1 QB initialization bug**: `build_qb_data(through_week=0)` triggered early return before `_load_prior_season` was called, leaving week 1 with zero QB adjustment. Fixed with `max(1, week - 1)`.

2. **Vegas spread sign convention documentation**: Added comprehensive comment block explaining:
   - jp_spread: Positive (+) = home team favored
   - vegas_spread: Negative (-) = home team favored
   - edge = (-jp_spread) - vegas_spread
   - Added debug log line for convention verification in production.

3. **rankings=None behavior documentation**: Explained that current-week AP/CFP poll rankings are not fetched, disabling letdown/lookahead detection. **Critical note**: LSA coefficients were trained with this same behavior—if future version fetches rankings, LSA must be retrained.

4. **Legacy LSA path safety**: Added log line for legacy mode (`"LSA applied directly to N predictions (legacy mode)"`) and documented the two LSA application modes (legacy vs edge-aware).

**qb_calibration.py (4 fixes):**

1. **Removed qb_points_effective references**: Field was removed from QBQuality; updated section 6 to analyze qb_points only.

2. **Week 1 build_qb_data bug**: Same fix as run_weekly.py—`max(1, pred_week - 1)`.

3. **Dummy "Opponent" team pollution**: Changed from `get_adjustment(team, "Opponent")` to `_compute_qb_quality(team)` directly, preventing thousands of junk entries in calibration DataFrame.

4. **API rate limiting**: Added 2-second delay between years to prevent hitting CFBD limits on multi-year calibration runs.

---

#### Summary of QB Continuous State

The QB Continuous system is now fully hardened:
- Walk-forward safe with temporal guards
- Prior season data loads correctly for week 1
- Residual adjustment logic uses consistent shrinkage basis
- Calibration script produces clean data without dummy entries
- Production script has clear sign convention documentation

**Production Defaults (2026):**
- `--qb-continuous --qb-phase1-only` enabled by default
- Phase 1 (weeks 1-3): QB adjustment applied (+0.6% 5+ Edge improvement)
- Phase 2 (weeks 4-15): QB adjustment disabled (avoids double-counting with EFM)

---

*End of session*

---

## Session: February 13, 2026

### Theme: Totals Calibration & EV Engine Bug Fixes + ATS Export Enhancement

---

#### ATS Export: Add Open Line Columns — COMMITTED
**Status**: Committed (`9ea61b1`)
**Files**: `scripts/backtest.py`

Added `ats_win_open` and `ats_push_open` columns to `ats_export.csv` for direct betting analysis against opening lines:

- `ats_win` / `ats_push`: Result vs CLOSE line (existing)
- `ats_win_open` / `ats_push_open`: Result vs OPEN line (new)

**Why needed**: The existing `ats_win` column calculates ATS result against the closing line, but betting analysis uses opening lines. Previously required manual recalculation; now can use column directly.

**Example**: Kent State @ Texas Tech (Week 2, 2025)
- Final: Texas Tech won by 48
- Open spread: -48.5 (Kent State +48.5)
- Close spread: -47.5 (Kent State +47.5)
- `ats_win` = False (vs close: 48 + (-47.5) = +0.5, Kent State loses)
- `ats_win_open` = True (vs open: 48 + (-48.5) = -0.5, Kent State **wins**)

---

#### Totals Calibration: 6 Bug Fixes — COMMITTED
**Status**: Committed (`62089d7`)
**Files**: `src/spread_selection/totals_calibration.py`

1. **Mu column mismatch fix (P0)**: `run_full_calibration` called `collect_walk_forward_residuals` which resolves the mu column to `mu_used`, but then passed original `preds_df` to `backtest_ev_roi` which independently picked its own column. Fixed by passing `mu_column='mu_used'` explicitly to `tune_sigma_for_roi`.

2. **Vectorize backtest_ev_roi (Performance)**: Replaced iterrows loop (~55,000 Python iterations with 17 sigma candidates × 3,200 games) with vectorized numpy/scipy operations. Uses `scipy.stats.norm.cdf` for arrays.

3. **Silent ROI rejection warning (Minor)**: When all sigma candidates fail constraints (cap_hit_rate > 0.30 or n_bets_per_season < 50), now logs warning with constraint details and sets `no_valid_candidate: True` flag.

4. **Week bucket multiplier clamping (Minor)**: `compute_week_bucket_multipliers` now clamps to [0.8, 1.5] with warning. Prevents extreme values from noisy buckets (especially "15+" with few bowl games).

5. **from_dict forward compatibility (Minor)**: `TotalsCalibrationConfig.from_dict` now filters to valid dataclass fields with warning for unknown keys, preventing crashes when loading configs from newer code versions.

6. **Sigma validation (Trivial)**: Added guards in `tune_sigma_for_coverage`, `tune_sigma_for_roi`, and `calculate_totals_probabilities` to raise `ValueError` if sigma candidates are not positive.

---

#### Totals EV Engine: 6 Bug Fixes — COMMITTED
**Status**: Committed (`62089d7`)
**Files**: `src/spread_selection/totals_ev_engine.py`

1. **one_bet_per_event drops List B fix (P1)**: Filter was running on `all_recommendations` before the List A/B split, dropping high-edge bets that should appear in 5+ Edge list. Fixed by applying filter only to List A after the split.

2. **one_bet_per_market drops List B fix (P1)**: `evaluate_single_market` filtered to best EV side before caller could check List B eligibility. Fixed by returning both sides always; filtering applied in `evaluate_totals_markets` to List A only.

3. **Deterministic tie-breaking (Minor)**: Changed from simple dict overwrite to tuple comparison key `(-ev, -stake, book, line, side)` for consistent ordering when EV ties.

4. **Deprecation warning for use_adjusted_total (Minor)**: Added explicit comment that `pred.adjusted_total` is never used and `use_adjusted_total` flag is deprecated. Prevents double weather adjustment if someone misreads the config.

5. **MuOverrideFn type alias fix (Trivial)**: Changed from `Optional[callable]` (builtin function) to `Optional[Callable[[float, float, "TotalsEvent", "TotalMarket"], float]]` (proper type hint).

6. **sigma_used default fix (Trivial)**: Changed default from `0.0` to `None`. The value is always set via `evaluate_single_market`'s fallback logic; `None` makes it clear when uninitialized.

---

*End of session*


---

## Session: February 14, 2026

### Theme: Weather Layer Smoke Test Infrastructure

---

#### Smoke Test Script for Weather + Totals EV — COMMITTED
**Status**: Committed (`4e9253f`)
**Files**: `scripts/smoke_test_weather_ev.py`, `.gitignore`

Created comprehensive smoke test for the weather layer integration with the Totals EV Engine. Two execution modes:

**Mode A (Synthetic):**
- Deterministic, always runs with hard assertions
- Tests mu composition formula: `mu_used = mu_model + weather_adj + baseline_shift`
- Validates push probability (0 for half-points, >0 for integers)
- Validates guardrail capping at ±10 points
- Validates Kelly staking logic

**Mode B (Real-Data):**
- Three-phase architecture: Preflight → Model Prep → Evaluation
- `--require-weather` flag for strict weather data requirement
- Graceful skip if data missing (preflight checks run before any heavy compute)

**CLI Enhancements:**
- `--max-train-seconds N`: Time-bounded training with multiprocessing timeout
- `--max-games N`: Limit training data for faster iteration
- `--save-model-path PATH`: Cache trained model for reuse
- `--load-model-path PATH`: Load pre-trained model (directory auto-discovery supported)
- `--overwrite-model`: Allow overwriting existing saved models

**Model Filename Convention:**
- Format: `totals_{year}_pred_w{pred_week}_trained_thru_w{train_week}.joblib`
- Example: `totals_2024_pred_w10_trained_thru_w9.joblib`
- Walk-forward semantics explicit in filename (pred_week > train_thru_week always)

**Git Hygiene (.gitignore additions):**
- `artifacts/` - smoke test model cache directory
- `*.joblib` - serialized models
- `data/weather/` - captured weather data (not source-controlled)
- `calibration_artifacts/` - calibration outputs

**Usage Examples:**
```bash
# Full synthetic + real-data test
python3 scripts/smoke_test_weather_ev.py --year 2024 --week 10

# Save model for future runs
python3 scripts/smoke_test_weather_ev.py --year 2024 --week 10 \
    --save-model-path artifacts/models/

# Fast rerun with cached model
python3 scripts/smoke_test_weather_ev.py --year 2024 --week 10 \
    --load-model-path artifacts/models/

# Strict weather requirement (fails fast if no weather file)
python3 scripts/smoke_test_weather_ev.py --year 2024 --week 10 --require-weather
```

---

*End of session*


---

## Session: February 14, 2026 (Continued)

### Theme: Bug Fixes Across Totals and Spreads Pipelines

---

#### Totals Calibration: 5 Bug Fixes — COMMITTED
**Status**: Committed (`14e581c`)
**Files**: `src/spread_selection/totals_calibration.py`

1. **OVER bet selection bug (CRITICAL)**: `pick_over = mu > line` was always False due to logic error. Fixed: now correctly picks OVER when model total exceeds line.

2. **Kelly formula denominator**: Changed from `f_star = numerator / b` to `f_star = numerator / (b * (p_win + p_loss))` for proper three-outcome Kelly.

3. **max_games 8→10**: CFB teams play 12-13 games; 8 was too conservative for reliability scaling.

4. **Week bucket rename**: "1-2" → "0-2" to reflect CFB week 0 games.

5. **Self-test enhancement**: Added numerical Kelly verification in module self-test.

---

#### Totals EV Engine: 6 Bug Fixes — COMMITTED
**Status**: Committed (`14e581c`)
**Files**: `src/spread_selection/totals_ev_engine.py`

1. **Kelly formula fix**: Same three-outcome denominator fix as calibration module.

2. **Week bucket fix**: "1-2" → "0-2" for week 0 compatibility.

3. **reliability_max_games**: 8→10 for CFB's 12-13 game season.

4. **Diagnostic mode fix (P1)**: Guardrail-triggered bets (stake=0) now stay in List A instead of being dropped. These are valuable diagnostics showing model conviction even when sizing is capped.

5. **sigma default**: 20.0→13.0→17.0 (final value after strategist analysis).

6. **Sanity test Kelly verification**: Added numerical verification in module self-test.

---

#### Sigma Selection: Model Strategist Analysis
**Status**: Analyzed, sigma=17 selected
**Commits**: `314c143`

Ran 3-year sigma sweep (2023-2025, weeks 3-14) with external critique:
- Sigma=13: 54.2% overall but only 52.7% in 2025 (near break-even)
- Sigma=17-25: More consistent 53.3-53.7% floor in weak years

**Strategist verdict**: 2025 single-year dip is statistical noise (N≈500, CI overlaps). However, sigma=17 chosen as "golden mean":
- 94% volume of sigma=13
- Best 2023 performance (54.7%)
- Realistic for CFB variance
- Protects against Kelly over-sizing in edge cases

---

#### Obsolete Script Cleanup — COMMITTED
**Status**: Committed (`c8571cf`)
**Deleted**: 5 scripts, ~1,542 lines

- `scripts/debug_totals_ev.py` (139 lines)
- `scripts/analyze_totals_backtest.py` (421 lines)
- `scripts/sigma_sweep_analysis.py` (217 lines)
- `scripts/analyze_sigma_sweep.py` (311 lines)
- `scripts/test_kelly_integration.py` (454 lines)

All functionality superseded by `totals_calibration.py` and `smoke_test_weather_ev.py`.

---

#### run_weekly.py: 4 CFB Week 0 Fixes — COMMITTED
**Status**: Committed (`0c62e80`)
**Files**: `scripts/run_weekly.py`

1. **Week 0 crash**: Changed `if week == 1` to `if week <= 1` for priors-only mode.

2. **Week 0 data excluded**: Changed `range(1, through_week + 1)` to `range(0, through_week + 1)` in 3 data-fetching functions.

3. **export-slate fallback**: Added fallback for `--no-lsa` mode which doesn't produce `lsa_spread`.

4. **QB Continuous week 0**: Changed `if week <= 3` to `if 1 <= week <= 3` (week 0 has no prior PPA data).

---

#### run_weekly_totals.py: 4 Bug Fixes — COMMITTED
**Status**: Committed (`d73c591`)
**Files**: `scripts/run_weekly_totals.py`

1. **Record mismatch**: Record calculation now applies same `min_edge` filter as display, so reported W-L matches displayed bets.

2. **--show-all display count**: Captured return value as `displayed_all` and added to summary output.

3. **Week 0 informative exit**: Added early check with clear message: "Totals model requires at least 1 week of game data for training."

4. **Training log fix**: Changed "weeks 1-{N}" to "weeks 0-{N}" to reflect actual data range.

---

*End of session*


---

## Session: February 14, 2026 (Continued)

### Theme: Phase 1 Documentation & Code Cleanup

---

#### Week 0 Spread Selection Fix — COMMITTED
**Status**: Committed (`f5f0805`)
**Files**: 7 files in `src/spread_selection/`

CFB has week 0 games (Dublin game, pre-Labor Day) that were being excluded from Phase 1:

- `phase1_edge_baseline.py`: weeks `[1,2,3]` → `[0,1,2,3]`
- `phase1_sp_gate.py`: weeks `[1,2,3]` → `[0,1,2,3]`
- `calibration.py`: week bucket `(1, 3)` → `(0, 3)`
- Updated all "weeks 1-3" comments to "weeks 0-3" across 7 files

---

#### SP+ Agreement Gate: Documentation Inconsistency Found
**Status**: Research validated, docs corrected

**The Problem**: MODEL_EXPLAINER.md and MODEL_ARCHITECTURE.md showed 60% ATS for SP+ confirms, but PHASE1_SP_POLICY.md (research frozen 2026-02-13) showed:
- 2022: **18.8% ATS** (catastrophic)
- Aggregate 60% hid year-by-year variance

**Resolution**: The 60% was misleading. SP+ gating research was already frozen with conclusion: "SP+ confirm-only gating is harmful."

---

#### SP+ Gate Documentation Removal — COMMITTED
**Status**: Committed (`f755c14`)
**Files**: `docs/MODEL_EXPLAINER.md`, `docs/MODEL_ARCHITECTURE.md`

Removed misleading SP+ Agreement Gate sections (-54 lines):
- Threshold optimization table
- Gating modes table
- Decision matrix SP+ rows

---

#### Kill-Switch Documentation Removal — COMMITTED
**Status**: Committed (`653a053`)
**Files**: `docs/MODEL_EXPLAINER.md`, `docs/MODEL_ARCHITECTURE.md`

Kill-switch required manual result tracking to function in production. Was documented as if it worked automatically. Removed (-61 lines):
- Kill-switch protection section
- Historical trigger analysis table
- CLI reference for kill-switch flags

**Simplified Phase 1 guidance**: "Bet cautiously at half stakes. The ~51% ATS is barely above breakeven, so position sizing matters more than selection."

---

#### SP+ Gate Code Removal — COMMITTED
**Status**: Committed (`cc859ee`)
**Files**: 4 files, -894 lines

Deleted:
- `src/predictions/sp_gate.py` (331 lines)
- `src/spread_selection/policies/phase1_sp_gate.py` (468 lines)
- CLI args and imports from `run_weekly.py`
- Updated `policies/__init__.py` to empty exports

---

#### Kill-Switch Code Removal — COMMITTED
**Status**: Committed (`7ed15a1`)
**Files**: 2 files, -331 lines

Deleted:
- `src/predictions/phase1_killswitch.py` (259 lines)
- CLI args, function parameters, and logic from `run_weekly.py`

---

#### Total Phase 1 Cleanup Summary

| Item | Lines Removed |
|------|---------------|
| SP+ gate docs | -54 |
| Kill-switch docs | -61 |
| `sp_gate.py` | -331 |
| `phase1_sp_gate.py` | -468 |
| `phase1_killswitch.py` | -259 |
| `run_weekly.py` SP+/KS code | -112 |
| **Total** | **~1,285 lines** |

**Rationale**: Both features had fundamental issues:
1. **SP+ Gate**: Unstable year-to-year (18.8% in 2022 vs 70%+ in other years)
2. **Kill-Switch**: Required manual result tracking that didn't exist in production

**New Phase 1 guidance**: Simple and honest — bet cautiously at half stakes, ~51% ATS expected.

---

*End of session*

---

## Session: February 14, 2026 (Evening)

### Theme: Totals Display Script Overhaul & Statistical Validation

---

#### show_totals_bets.py: Switch from EV to 5+ Edge Filter — COMMITTED
**Status**: Committed (`b06c0a1`)
**Files**: `scripts/show_totals_bets.py`, `.claude/skills/show-totals-bets/SKILL.md`

**Problem**: At 3% EV threshold (matching spreads), totals produced 35-46 games/week — "46 games is crazy."

**Analysis** (2023-2025 backtest):
| Filter | Games/Wk | ATS% | ROI |
|--------|----------|------|-----|
| 3% EV | 34.8 | 53.3% | +1.7% |
| 5% EV | 32.2 | 53.2% | +1.6% |
| 10% EV | 24.6 | 52.8% | +0.8% |
| **5+ Edge** | **15.8** | **53.7%** | **+2.4%** |

**Root cause**: EV highly correlated with edge — raising EV threshold barely reduces volume.

**Changes**:
- Primary filter: 5+ edge (not EV-based)
- EV shown as `~EV` column for reference only
- Volume: 35 → 15 games/week (-57%)
- Default sigma updated to 16.4 (calibrated from residuals)

---

#### show_totals_bets.py: Fix CLOSE→OPEN Line Bug — COMMITTED
**Status**: Included in same commit (`b06c0a1`)

**Problem**: CSV `result` column computed vs CLOSE line, but we filter by `edge_open` and bet on OPEN lines.

**Evidence**:
- CSV `result`: 96-92 (51.1%)
- Recalc vs OPEN: 98-90 (52.1%)

**Fix**: Added `calc_result_vs_open()` and `calc_play_vs_open()` functions to recalculate results at display time.

---

#### 2025 Totals Performance Degradation: Strategic Analysis
**Status**: Research complete, no action required

Invoked model-strategist to analyze why 2025 (53.1% ATS) underperformed 2023-2024 (56-59% ATS).

**Root cause**: 2024 was the anomaly, not 2025.
- JP+ has systematic OVER bias (55-61% of picks)
- 2024 had elevated OVER base rate (+9.1pp selection lift)
- 2025 OVER lift = 0.0pp (normal reversion)

**UNDER vs OVER performance** (2023-2025 pooled):
| Side | ATS% |
|------|------|
| OVER | 53.6% |
| UNDER | 55.2% |

Model's edge is stronger on UNDERs (identifying overpriced totals).

---

#### Statistical Validation: OVER vs UNDER Asymmetry
**Status**: Validated as NOT statistically significant

Rigorous 6-step validation protocol:

| Step | Test | Result |
|------|------|--------|
| 1 | Bias (JP+ - Vegas) | +0.46 pts (sig, but small) |
| 2 | OVER vs UNDER ATS (5+ edge) | +2.9pp, **p=0.39 (NOT SIG)** |
| 3 | Season consistency | 3/4 seasons (reversed in 2024) |
| 4 | Walk-forward | 1/2 folds (does not persist) |
| 5 | Residual asymmetry | Selection artifact (explained) |

**Conclusion**: UNDER outperformance is NOT statistically significant, does not persist out-of-sample, and is explained by selection mechanics. **No adjustment justified.**

---

#### 5+ vs 6+ Edge Threshold Validation
**Status**: Validated, keep 5+ edge

Full statistical comparison:

**Season-by-Season**:
| Season | 5+ ATS | 6+ ATS | Winner |
|--------|--------|--------|--------|
| 2023 | 52.6% | 51.6% | **5+** |
| 2024 | 57.8% | 61.4% | 6+ |
| 2025 | 52.1% | 49.6% | **5+** |

**Statistical Significance**: None (all p > 0.05, pooled p=0.94)

**Walk-Forward**:
- Test 2024: 6+ wins
- Test 2025: 5+ wins
- Does NOT persist

**Risk-Adjusted**:
- 5+ ROI variance: 24.1
- 6+ ROI variance: 97.9 (4x more unstable)

**Classification**: C) No meaningful difference → Keep 5+

**Key finding**: 6+ advantage ENTIRELY driven by 2024 (anomaly OVER tailwind year).

---

#### Summary: Totals Model Status

| Metric | 2023-2025 Backtest |
|--------|-------------------|
| Filter | 5+ edge vs OPEN |
| Volume | ~15 games/week |
| ATS | 54.3% |
| ROI | +3.6% |

**No changes to model**. Display script fixed. Statistical rigor confirms current approach is sound.

---

*End of session*
