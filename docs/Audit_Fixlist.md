```md
# AUDIT_FIXLIST.md — JP+ Codebase Audit & Fix List

**Last updated:** 2026-02-03  
**Scope:** JP+ backtest + EFM + SpreadGenerator + adjustments + metadata

This checklist consolidates all audit findings into one canonical backlog (deduped).  
Each item includes an **AI nudge prompt** (non-prescriptive) you can paste into Claude Code.

---

## How to use this file

- Work top-down (P0 → P3).  
- After each fix, re-run:
  - `python scripts/backtest.py --use-efm ...` (your standard years)
  - `python scripts/backtest.py --use-efm --opening-line` (if you track open)
- Keep PRs small: **one checkbox = one PR**.
- For each PR, record “before vs after” metrics (MAE, ATS, 3+/5+ edge) and attach output CSV diffs if possible.

---

## P0 — Must Fix (can change ATS materially / leakage risk)

- [x] **P0.1 Fix trajectory leakage (current-year record used)** ✅ COMPLETE
  - **Files:** `src/adjustments/home_field.py`
  - **Issue:** `calculate_trajectory_modifiers()` includes `current_year` in "recent" window; leaks info into walk-forward.
  - **Acceptance criteria:** Trajectory for season `Y` uses only years `< Y`. Add guard/log proving it.
  - **AI nudge prompt:**
    > Audit `HomeFieldAdvantage.calculate_trajectory_modifiers()` end-to-end and ensure trajectory modifiers for a season are computed using only seasons strictly prior to that season. Make the implementation match the documented intent ("calculate once at season start using prior year as recent"), and add a guard/log that proves no current-year results were used.
  - **Notes:** FIXED 2026-02-03. Changed recent years range from `[current_year]` to `[current_year - 1]`. For 2024 predictions: baseline=2020-2022, recent=2023 (NOT 2024). Added logging that explicitly shows year ranges and confirms current year is NOT included. Verified with test showing correct behavior.

- [x] **P0.2 Key ATS + Vegas lookups by `game_id` (not team names)** ✅ COMPLETE
  - **Files:** `scripts/backtest.py`, `src/predictions/vegas_comparison.py`
  - **Issue:** `(home_team, away_team)` matching can mis-assign lines (rematches, naming drift, neutral-site home/away flips).
  - **Acceptance criteria:** Predictions carry `game_id`; lines matched by `game_id`; report match rate and list unmatched.
  - **AI nudge prompt:**
    > Refactor all Vegas line lookups and ATS evaluation to be keyed on `game_id` rather than `(home_team, away_team)`. Ensure predictions carry `game_id` through the entire pipeline (predictions output, ATS results, value plays). Add a sanity report showing the % of predictions successfully matched to a betting line and list unmatched games.
  - **Notes:** FIXED 2026-02-03. Added `game_id` to prediction results in both `walk_forward_predict` and `walk_forward_predict_efm`. Refactored `calculate_ats_results()` to match by `game_id` using O(1) dict lookup. Added sanity report logging match rate and unmatched games. Updated `VegasComparison` to support `game_id` matching via `get_line_by_id()` and updated `get_line()` to prefer `game_id` when provided. Test shows 100% match rate (631/631).

- [x] **P0.3 Remove EFM double-normalization / scaling from rounded outputs** ✅ COMPLETE
  - **Files:** `scripts/backtest.py` (EFM path), `src/models/efficiency_foundation_model.py`
  - **Issue:** Backtest rescales EFM ratings using `get_ratings_df()` (rounded) and EFM already normalizes internally.
  - **Acceptance criteria:** No scaling uses rounded values; only one normalization path exists; weekly rating scale is stable.
  - **AI nudge prompt:**
    > Review how EFM ratings are normalized/scaled across `EfficiencyFoundationModel` and the EFM walk-forward backtest. Eliminate double-normalization and ensure numeric scaling never uses rounded/presentation outputs. Confirm weekly rating scale and predicted spreads are stable and comparable week-to-week.
  - **Notes:** FIXED 2026-02-03. Removed second normalization in `walk_forward_predict_efm()` that was rescaling from std=12 to std=10 using rounded DataFrame values. Now uses `efm.get_rating(team)` directly for full precision. EFM's `_normalize_ratings()` (std=12.0) is now the ONLY normalization. Test verified: full precision std=12.0000 stable across weeks 4-8, means centered at 0, rounding error minimal (<0.05).

- [x] **P0.4 Prevent silent season truncation on exceptions** ✅ COMPLETE
  - **Files:** `scripts/backtest.py` (`fetch_season_data`, `fetch_season_plays`)
  - **Issue:** Week loop `break` on exception drops remaining weeks silently.
  - **Acceptance criteria:** Robust fetching; exceptions don't truncate season; log missing weeks and reasons.
  - **AI nudge prompt:**
    > Make season data fetching resilient: transient API errors in a single week should not truncate the rest of the season. Add logging that clearly reports which weeks were missing/skipped and why, and ensure backtests remain deterministic.
  - **Notes:** FIXED 2026-02-03. Changed `break` to `continue` in both `fetch_season_data()` and `fetch_season_plays()` so transient API errors skip only the affected week, not all remaining weeks. Added per-week failure tracking with warning logs showing which weeks failed and why. Added data completeness sanity check in `fetch_all_season_data()` that reports missing weeks for games and plays. Test verified: all 15 weeks fetched successfully for 2024.

---

## P1 — Correctness / consistency bugs (skew MAE/ATS)

- [x] **P1.1 Fix travel direction logic (longitude sign inversion)** ✅ COMPLETE
  - **Files:** `src/adjustments/travel.py`, `config/teams.py`
  - **Issue:** West→East vs East→West detection is inverted due to negative longitude conventions.
  - **Acceptance criteria:** Validated with real examples (e.g., UCLA→Rutgers, Rutgers→UCLA).
  - **AI nudge prompt:**
    > Validate the west-to-east vs east-to-west travel logic in `TravelAdjuster.get_timezone_adjustment()` using real team examples from `config/teams.py`. Correct any inverted direction detection so "west-to-east is harder" is applied to the correct cases.
  - **Notes:** FIXED 2026-02-03. The longitude comparison logic was inverted. When `away_lon < home_lon`, the away team is WEST of home (traveling EAST), but the code was applying the 0.8x "easier" multiplier instead of full penalty. Swapped the conditions so West→East gets full penalty and East→West gets 0.8x. Verified with UCLA→Rutgers (-1.50) vs Rutgers→UCLA (-1.20).

- [x] **P1.2 Simplify travel direction logic (prefer tz offsets over longitude heuristics)** ✅ COMPLETE
  - **Files:** `src/adjustments/travel.py`, `config/teams.py`
  - **Issue:** You already store tz offsets; direction inference should be consistent and non-contradictory.
  - **Acceptance criteria:** One clear rule for direction; consistent application; documented.
  - **AI nudge prompt:**
    > Simplify travel direction detection so it uses a single consistent source of truth (timezone offsets and/or kickoff context). Remove redundant/contradictory longitude heuristics and ensure west-to-east penalties trigger correctly in all relevant cases.
  - **Notes:** FIXED 2026-02-03. Added `get_directed_timezone_change()` function that returns signed timezone difference (positive = east, negative = west). Refactored `get_timezone_adjustment()` to use this instead of longitude comparisons. Logic is now simpler: `directed_tz > 0` → traveling east (full penalty), `directed_tz < 0` → traveling west (0.8x penalty). Verified with 8 test cases including Hawaii.

- [x] **P1.3 Make baseline HFA sourcing explicit and consistent** ✅ COMPLETE
  - **Files:** `config/settings.py`, `src/adjustments/home_field.py`, `scripts/backtest.py`
  - **Issue:** Multiple "baselines" (settings default vs CLI vs curated TEAM_HFA_VALUES) can cause instability/confusion.
  - **Acceptance criteria:** Document and log final HFA per team (or per game) in backtests; predictable behavior.
  - **AI nudge prompt:**
    > Audit how baseline HFA is sourced and applied (settings defaults, CLI fallback, curated team HFA table, dynamic/team_hfa). Make behavior explicit and consistent, and add logging that reports the final HFA used (at least per team, ideally per game) during backtests.
  - **Notes:** FIXED 2026-02-03. Added source tracking to `get_hfa()` which now returns `(hfa_value, source_string)`. Sources are: "curated" (TEAM_HFA_VALUES), "dynamic" (calculated), "conf:XXX" (conference default), "fallback" (base_hfa). Trajectory modifiers append "+traj(±X.XX)" to source. Added `get_hfa_value()` for backward compatibility, `get_hfa_breakdown()` for per-team details, and `log_hfa_summary()` for aggregate stats. Backtest now logs: "HFA sources: curated=46, fallback=90, with_trajectory=108".

- [ ] **P1.4 De-duplicate rivalry list + add validation**
  - **Files:** `config/teams.py`
  - **Issue:** Duplicates in rivalry list don’t break runtime but signal drift and risk.
  - **Acceptance criteria:** Clean unique list; validator asserts no duplicates after normalization.
  - **AI nudge prompt:**
    > Clean up rivalry metadata so the source list contains no duplicates or mirrored pairs. Add a validation check/test that asserts there are no duplicate rivalry definitions after normalization.
  - **Notes:**

- [ ] **P1.5 Unify turnover play type definitions across repo**
  - **Files:** `src/models/efficiency_foundation_model.py`, `scripts/backtest.py`, any turnover/FD logic
  - **Issue:** Different turnover play-type sets are used in different places.
  - **Acceptance criteria:** Single shared source of truth; small test/assertion ensures all modules use it.
  - **AI nudge prompt:**
    > Consolidate turnover play type definitions into a single shared source of truth used consistently by EFM turnover calculation, backtest turnover scrubbing, and any other turnover-related logic. Add a small test/assertion that fails if modules diverge.
  - **Notes:**

- [x] **P1.6 Keep full precision spreads internally (don't round before evaluation)** ✅ COMPLETE
  - **Files:** `src/predictions/spread_generator.py`
  - **Issue:** Rounding spreads before MAE/edge bucketing changes metrics.
  - **Acceptance criteria:** Store float spread internally; round only in report/DF output.
  - **AI nudge prompt:**
    > Ensure spreads retain full numeric precision internally and are only rounded at reporting/UI boundaries. Confirm MAE and ATS edge bucketing operate on unrounded values.
  - **Notes:** FIXED 2026-02-03. `PredictedSpread.spread` and `home_win_probability` now store full precision values internally. Added `spread_display` and `win_prob_display` properties for rounded presentation. `to_dict()` now includes both `spread` (rounded to 0.5 for display) and `spread_raw` (full precision for analysis). Backtest verified: 191/196 predictions have fractional precision, MAE diff ~0.004 pts, edge bucketing correctly uses full precision (107 vs 110 games at 3+ pt threshold).

---

## P2 — Engine correctness & feature completeness (improves MAE / long-run ATS)

- [x] **P2.1 Add home/away context for EFM ridge (neutral-field coefficients)** ✅ COMPLETE
  - **Files:** `src/models/efficiency_foundation_model.py`, `scripts/backtest.py` (play ingestion)
  - **Issue:** EFM ridge doesn't isolate implicit HFA; team strength becomes home-contaminated.
  - **Acceptance criteria:** Ridge separates home effect from team strength; play rows contain home/away context.
  - **AI nudge prompt:**
    > Update EFM opponent adjustment so team strength is estimated as a neutral-field latent parameter and home/away effects are separated. Ensure play ingestion provides enough context to determine whether the offense is home or away for each play, and add diagnostics proving the home effect is being captured separately.
  - **Notes:** FIXED 2026-02-03. Added `home_team` field to efficiency_plays in backtest.py. Modified `_ridge_adjust_metric()` to add home indicator column to design matrix (+1 home, -1 away, 0 neutral). Model now learns implicit HFA separately (~0.006 SR, ~0.02 IsoPPP ≈ 0.8 pts). Mean error improved from -6.7 to ~0. Commit: 1b2164e.

- [x] **P2.2 Fix ridge intercept handling to avoid double-counting baseline** ✅ VERIFIED (NOT A BUG)
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** Intercept is effectively applied to both offense and defense outputs.
  - **Acceptance criteria:** Clear interpretation; combining O/D doesn't double-count baseline.
  - **AI nudge prompt:**
    > Review how ridge regression coefficients and intercept are interpreted/extracted for offense and defense adjusted values. Ensure the baseline/intercept is handled consistently so combining offense and defense does not double-count any shared baseline component.
  - **Notes:** VERIFIED 2026-02-03. Analysis confirms intercept handling is correct:
    1. Both `off_adjusted` and `def_adjusted` include intercept, but it cancels when computing ratings: `overall ∝ (off_coef + def_coef)` with no intercept term.
    2. Pre-normalization: `mean(off) ≈ 0`, `mean(def) ≈ 0` due to balanced ridge regression.
    3. Normalization uses `current_mean/2` for O/D which correctly distributes residual mean from turnover component.
    4. Formula `overall = off + def + 0.1*TO` holds exactly post-normalization (verified: error = 0.000000).
    5. Added clarifying comment to `_normalize_ratings()` explaining why this works.

- [x] **P2.3 Make IsoPPP computation consistent with weights** ✅ COMPLETE
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** SR is weighted; IsoPPP uses unweighted mean.
  - **Acceptance criteria:** IsoPPP uses the same play weights (GT/time decay) as SR.
  - **AI nudge prompt:**
    > Make IsoPPP (EPA/PPA on successful plays) computation consistent with the play weighting scheme used for success rate (garbage time weighting and any time decay). If a play is down-weighted for SR, it should also be down-weighted for IsoPPP.
  - **Notes:** FIXED 2026-02-03. In `_calculate_raw_metrics()`, changed IsoPPP calculation from `successful["ppa"].mean()` to weighted mean `(succ_ppa * succ_weights).sum() / succ_weights.sum()`. Now both SR and IsoPPP use identical play weighting (garbage time + time decay). Verified: synthetic test with inflated garbage time EPA shows weighted IsoPPP (0.34) properly lower than unweighted (0.52). Ridge adjustment already used weights correctly. Backtest unchanged (MAE 12.52, ATS 51.5%).

- [x] **P2.4 Consolidate garbage time thresholds (Settings → EFM)** ✅ COMPLETE
  - **Files:** `config/settings.py`, `src/models/efficiency_foundation_model.py`
  - **Issue:** Settings thresholds don't control EFM; EFM has redundant logic.
  - **Acceptance criteria:** Single source of truth; clean non-overlapping logic; test coverage.
  - **AI nudge prompt:**
    > Consolidate garbage time threshold configuration so EFM uses a single source of truth (preferably Settings). Remove redundant/overlapping garbage time logic and add a test verifying garbage time classification behaves as intended by quarter.
  - **Notes:** FIXED 2026-02-03. Refactored `is_garbage_time()` to use Settings thresholds (Q1:28, Q2:24, Q3:21, Q4:16) instead of hardcoded overlapping conditions. Old logic had no Q1/Q2 detection and Q4>14; new uses per-quarter thresholds from Settings. Clean single-source lookup: `thresholds.get(quarter)`. Verified with 12 test cases (all boundaries + interior). Backtest: MAE 12.58, ATS 52.2% (improved from 51.5%), 3+ edge 53.7%.

- [ ] **P2.5 Fix normalization of offense/defense components**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** O/D centered using `overall_mean/2`, which is not generally valid.
  - **Acceptance criteria:** Components normalized consistently; intended relationships preserved.
  - **AI nudge prompt:**
    > Fix the rating normalization logic so offense and defense components are centered/scaled in a mathematically consistent way (not assuming overall_mean/2). Ensure component relationships remain interpretable and consistent after normalization.
  - **Notes:**

- [ ] **P2.6 Split turnovers into offense vs defense components**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** Turnovers are only in overall; O/D ratings omit turnover effects (hurts totals modeling).
  - **Acceptance criteria:** O reflects ball security; D reflects takeaways; overall remains consistent.
  - **AI nudge prompt:**
    > Refactor turnover handling so turnover effects are represented in offense and defense components (ball security vs takeaways) while keeping overall rating consistent. Ensure totals/matchup computations can use O/D ratings without missing turnover effects.
  - **Notes:**

- [ ] **P2.7 Clarify special teams integration (avoid double-counting or no-counting)**
  - **Files:** `src/models/efficiency_foundation_model.py`, `src/predictions/spread_generator.py`, `src/models/special_teams.py`
  - **Issue:** ST stored in EFM but not in overall; also applied in SpreadGenerator as differential.
  - **Acceptance criteria:** One explicit integration strategy; documentation; guard against double-counting.
  - **AI nudge prompt:**
    > Clarify and enforce a single integration strategy for special teams: either treat ST purely as an adjustment layer in SpreadGenerator or incorporate it into base ratings—never both. Add documentation and a safeguard to prevent accidental double-counting.
  - **Notes:**

- [ ] **P2.8 Filter non-scrimmage plays for efficiency dataset + fix distance=0**
  - **Files:** `scripts/backtest.py` (play ingestion), `src/models/efficiency_foundation_model.py`
  - **Issue:** Efficiency plays may include non-scrimmage/weird play types; distance=0 can auto-success.
  - **Acceptance criteria:** Clear filter rules; logs removed plays count; success logic handles edge cases.
  - **AI nudge prompt:**
    > Ensure efficiency modeling uses only meaningful scrimmage plays and that success-rate logic handles edge cases like distance=0. Add logging on how many plays are filtered and why, and ensure changes are consistent across backtest and weekly runs.
  - **Notes:**

- [ ] **P2.9 Add validation + NaN handling + period/quarter normalization**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** Missing columns cause KeyErrors; NaN PPA can propagate; quarter naming mismatch can silently disable GT.
  - **Acceptance criteria:** Fail loudly when required schema missing; safe fallback when optional data missing; warnings/logs.
  - **AI nudge prompt:**
    > Add robust input validation and schema normalization for EFM training data, including NaN handling for PPA and consistent quarter/period handling. Ensure failures are explicit and logged instead of silently producing incorrect behavior.
  - **Notes:**

- [ ] **P2.10 Fix games-played assumptions for turnover shrinkage**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** Arbitrary fallback (e.g., default 10 games) biases shrinkage.
  - **Acceptance criteria:** Reliable games played count or explicit failure; log assumptions.
  - **AI nudge prompt:**
    > Make turnover shrinkage depend on a reliable games-played count. Avoid arbitrary defaults; if games cannot be counted from the inputs, fail loudly or compute from available identifiers, and log the assumptions used.
  - **Notes:**

- [ ] **P2.11 Add adjustment-stack diagnostics (travel + altitude + HFA)**
  - **Files:** `src/predictions/spread_generator.py` + adjustment modules
  - **Issue:** Additive correlated adjustments can create outliers and hidden double-counting.
  - **Acceptance criteria:** Report games with large combined adjustments; evaluate error patterns; consider caps if needed.
  - **AI nudge prompt:**
    > Add diagnostics that identify games where multiple correlated adjustments stack (travel + altitude + HFA + situational). Report these cases and evaluate whether they have systematic prediction errors. If stacking creates outliers, consider reasonable caps or scaling.
  - **Notes:**

- [ ] **P2.12 Evaluate timezone offsets/DST policy (Hawaii/Arizona edge cases)**
  - **Files:** `config/teams.py`, `src/adjustments/travel.py`
  - **Issue:** Fixed tz offsets ignore DST; may be wrong for some weeks.
  - **Acceptance criteria:** Document policy; validate edge cases; optionally use kickoff local time/timezone if available.
  - **AI nudge prompt:**
    > Decide and document a policy for timezone differences with DST effects (especially Hawaii and Arizona). Validate current offsets against the season calendar and either improve the calculation or add tests/documentation explaining the approximation.
  - **Notes:**

- [ ] **P2.13 Add centralized team-name normalization/aliasing**
  - **Files:** `config/teams.py`, any module doing lookups by team string
  - **Issue:** Exact string match required for rivalries/altitude/locations/HFA tables; CFBD naming drift can silently break.
  - **Acceptance criteria:** One normalization layer used before all lookups; log unknown team names; add coverage report.
  - **AI nudge prompt:**
    > Add a centralized team-name normalization/aliasing approach used before all metadata lookups (locations, altitude, rivalries, curated HFA, FBS sets). Add logging for unknown team names and a coverage report showing which teams are missing from metadata.
  - **Notes:**

---

## P3 — Performance / maintainability

- [ ] **P3.1 Use sparse matrix for ridge opponent adjustment**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** Dense X wastes memory; scales poorly.
  - **Acceptance criteria:** Sparse design matrix; results within tolerance; faster/more memory efficient.
  - **AI nudge prompt:**
    > Improve ridge opponent adjustment scalability by using an appropriate sparse representation for the design matrix. Ensure results match prior behavior within tolerance and add a quick benchmark/log showing memory/runtime improvements.
  - **Notes:**

- [ ] **P3.2 Vectorize play preprocessing & avoid O(T×N) filtering**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** `df.apply(axis=1)` and per-team slicing loops are slow.
  - **Acceptance criteria:** Vectorized computations / groupby aggregation; same outputs; improved runtime.
  - **AI nudge prompt:**
    > Optimize EFM preprocessing and raw metric aggregation by removing row-wise apply and repeated per-team dataframe scans. Use vectorized operations/grouped aggregation while preserving correctness and outputs.
  - **Notes:**

- [ ] **P3.3 Separate legacy ridge path from EFM in sweeps and reporting**
  - **Files:** `scripts/backtest.py`
  - **Issue:** Risk of confusing results between model types.
  - **Acceptance criteria:** Explicit labeling; prevent mixing incompatible settings; clear output.
  - **AI nudge prompt:**
    > Improve separation between legacy ridge and EFM backtests in CLI output and sweep logic. Make it difficult to conflate results, and ensure each mode reports its relevant parameters clearly.
  - **Notes:**

- [ ] **P3.4 Add backtest sanity report**
  - **Files:** `scripts/backtest.py`
  - **Include:** game counts, predictions per week, line-match rate, open≠close rate, rating mean/std.
  - **AI nudge prompt:**
    > Add a concise sanity report after each data fetch and backtest run: expected vs actual game counts, predictions per week, betting line match rate, open≠close rate, and rating distribution stats. The goal is to detect silent truncation, join failures, and scaling artifacts immediately.
  - **Notes:**

---

## Suggested fix order (Top 10)

1. ~~P0.1 Trajectory leakage~~ ✅ DONE
2. ~~P0.2 game_id joins for ATS + VegasComparison~~ ✅ DONE
3. ~~P0.3 EFM scaling/double-normalization + rounding contamination~~ ✅ DONE
4. ~~P0.4 robust season data fetching (no silent truncation)~~ ✅ DONE
5. P1.1/P1.2 travel direction + tz logic
6. P1.5 unify turnover play types
7. ~~P2.1 home/away context + isolate implicit HFA in EFM ridge~~ ✅ DONE
8. P2.2 intercept handling
9. P2.3/P2.8 IsoPPP weighting + play filtering + distance=0 fix
10. P2.5 normalization of O/D components

---

## Optional: Repo validator script (recommended)

Create a lightweight validator that checks:
- duplicates in rivalry list
- missing team locations for teams encountered in games/plays
- missing HFA entries / unexpected fallbacks
- mismatch between FBS team set and metadata coverage
- any team name appearing in games/plays not recognized by normalization layer
- ATS line match rate by game_id

---
```