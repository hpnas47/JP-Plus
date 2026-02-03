# AUDIT_FIXLIST.md — JP+ Codebase Audit & Fix List

**Last updated:** 2026-02-03  
**Scope:** JP+ backtest + EFM + SpreadGenerator + adjustments + metadata

This checklist consolidates all audit findings into one canonical backlog (deduped).  
Use it to track fixes and prevent regressions.

---

## How to use this file

- Work top-down (P0 → P3).  
- After each fix, re-run:
  - `python scripts/backtest.py --use-efm ...` (your standard years)
  - `python scripts/backtest.py --use-efm --opening-line` (if you track open)
- Add a short note under each checkbox with:
  - what changed,
  - backtest diffs (MAE, ATS, 3+/5+ edge),
  - any newly added tests/validators.

---

## P0 — Must Fix (can change ATS materially / leakage risk)

- [ ] **P0.1 Fix trajectory leakage (current-year record used)**
  - **Files:** `src/adjustments/home_field.py`
  - **Issue:** `calculate_trajectory_modifiers()` includes `current_year` in “recent” window; leaks info into walk-forward.
  - **Acceptance criteria:** Trajectory for season Y uses only years `< Y`. Add guard/log proving it.
  - **Notes:**

- [ ] **P0.2 Key ATS + Vegas lookups by `game_id` (not team names)**
  - **Files:** `scripts/backtest.py`, `src/predictions/vegas_comparison.py`
  - **Issue:** `(home_team, away_team)` matching can mis-assign lines (rematches, naming drift, neutral-site home/away flips).
  - **Acceptance criteria:** Predictions carry `game_id`; lines matched by `game_id`; report match rate and list unmatched.
  - **Notes:**

- [ ] **P0.3 Remove EFM double-normalization / scaling from rounded outputs**
  - **Files:** `scripts/backtest.py` (EFM path), `src/models/efficiency_foundation_model.py`
  - **Issue:** Backtest rescales EFM ratings using `get_ratings_df()` (rounded) and EFM already normalizes internally.
  - **Acceptance criteria:** No scaling uses rounded values; only one normalization path exists; weekly rating scale is stable.
  - **Notes:**

- [ ] **P0.4 Prevent silent season truncation on exceptions**
  - **Files:** `scripts/backtest.py` (`fetch_season_data`, `fetch_season_plays`)
  - **Issue:** Week loop `break` on exception drops remaining weeks silently.
  - **Acceptance criteria:** Robust fetching; exceptions don’t truncate season; log missing weeks and reasons.
  - **Notes:**

---

## P1 — Correctness / consistency bugs (skew MAE/ATS)

- [ ] **P1.1 Fix travel direction logic (longitude sign inversion)**
  - **Files:** `src/adjustments/travel.py`, `config/teams.py`
  - **Issue:** West→East vs East→West detection is inverted due to negative longitude conventions.
  - **Acceptance criteria:** Validated with real examples (e.g., UCLA→Rutgers, Rutgers→UCLA).
  - **Notes:**

- [ ] **P1.2 Simplify travel direction logic (prefer tz offsets over longitude heuristics)**
  - **Files:** `src/adjustments/travel.py`, `config/teams.py`
  - **Issue:** You already store tz offsets; direction inference should be consistent and non-contradictory.
  - **Acceptance criteria:** One clear rule for direction; consistent application; documented.
  - **Notes:**

- [ ] **P1.3 Make baseline HFA sourcing explicit and consistent**
  - **Files:** `config/settings.py`, `src/adjustments/home_field.py`, `scripts/backtest.py`
  - **Issue:** Multiple “baselines” (settings default vs CLI vs curated TEAM_HFA_VALUES) can cause instability/confusion.
  - **Acceptance criteria:** Document and log final HFA per team (or per game) in backtests; predictable behavior.
  - **Notes:**

- [ ] **P1.4 De-duplicate rivalry list + add validation**
  - **Files:** `config/teams.py`
  - **Issue:** Duplicates in rivalry list don’t break runtime but signal drift and risk.
  - **Acceptance criteria:** Clean unique list; validator asserts no duplicates after normalization.
  - **Notes:**

- [ ] **P1.5 Unify turnover play type definitions across repo**
  - **Files:** `src/models/efficiency_foundation_model.py`, `scripts/backtest.py`, any turnover/FD logic
  - **Issue:** Different turnover play-type sets are used in different places.
  - **Acceptance criteria:** Single shared source of truth; small test/assertion ensures all modules use it.
  - **Notes:**

- [ ] **P1.6 Keep full precision spreads internally (don’t round before evaluation)**
  - **Files:** `src/predictions/spread_generator.py`
  - **Issue:** Rounding spreads before MAE/edge bucketing changes metrics.
  - **Acceptance criteria:** Store float spread internally; round only in report/DF output.
  - **Notes:**

---

## P2 — Engine correctness & feature completeness (improves MAE / long-run ATS)

- [ ] **P2.1 Add home/away context for EFM ridge (neutral-field coefficients)**
  - **Files:** `src/models/efficiency_foundation_model.py`, `scripts/backtest.py` (play ingestion)
  - **Issue:** EFM ridge doesn’t isolate implicit HFA; team strength becomes home-contaminated.
  - **Acceptance criteria:** Ridge can separate home effect from team strength; play rows contain home/away context.
  - **Notes:**

- [ ] **P2.2 Fix ridge intercept handling to avoid double-counting baseline**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** Intercept is effectively applied to both offense and defense outputs.
  - **Acceptance criteria:** Clear interpretation; combining O/D doesn’t double-count baseline.
  - **Notes:**

- [ ] **P2.3 Make IsoPPP computation consistent with weights**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** SR is weighted; IsoPPP uses unweighted mean.
  - **Acceptance criteria:** IsoPPP uses the same play weights (GT/time decay) as SR.
  - **Notes:**

- [ ] **P2.4 Consolidate garbage time thresholds (Settings → EFM)**
  - **Files:** `config/settings.py`, `src/models/efficiency_foundation_model.py`
  - **Issue:** Settings thresholds don’t control EFM; EFM has redundant logic.
  - **Acceptance criteria:** Single source of truth; clean non-overlapping logic; test coverage.
  - **Notes:**

- [ ] **P2.5 Fix normalization of offense/defense components**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** O/D centered using `overall_mean/2`, which is not generally valid.
  - **Acceptance criteria:** Components normalized consistently; intended relationships preserved.
  - **Notes:**

- [ ] **P2.6 Split turnovers into offense vs defense components**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** Turnovers are only in overall; O/D ratings omit turnover effects (hurts totals modeling).
  - **Acceptance criteria:** O rating reflects ball security; D rating reflects takeaways; overall remains consistent.
  - **Notes:**

- [ ] **P2.7 Clarify special teams integration (avoid double-counting or no-counting)**
  - **Files:** `src/models/efficiency_foundation_model.py`, `src/predictions/spread_generator.py`, `src/models/special_teams.py`
  - **Issue:** ST stored in EFM but not in overall; also applied in SpreadGenerator as differential.
  - **Acceptance criteria:** One explicit integration strategy; documentation; guard against double-counting.
  - **Notes:**

- [ ] **P2.8 Filter non-scrimmage plays for efficiency dataset + fix distance=0**
  - **Files:** `scripts/backtest.py` (play ingestion), `src/models/efficiency_foundation_model.py`
  - **Issue:** Efficiency plays may include non-scrimmage/weird play types; distance=0 can auto-success.
  - **Acceptance criteria:** Clear filter rules; logs removed plays count; success logic handles edge cases.
  - **Notes:**

- [ ] **P2.9 Add validation + NaN handling + period/quarter normalization**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** Missing columns cause KeyErrors; NaN PPA can propagate; quarter naming mismatch can silently disable GT.
  - **Acceptance criteria:** Fail loudly when required schema missing; safe fallback when optional data missing; warnings/logs.
  - **Notes:**

- [ ] **P2.10 Fix games-played assumptions for turnover shrinkage**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** Arbitrary fallback (e.g., default 10 games) biases shrinkage.
  - **Acceptance criteria:** Reliable games played count or explicit failure; log assumptions.
  - **Notes:**

- [ ] **P2.11 Add adjustment-stack diagnostics (travel + altitude + HFA)**
  - **Files:** `src/predictions/spread_generator.py` + adjustment modules
  - **Issue:** Additive correlated adjustments can create outliers and hidden double-counting.
  - **Acceptance criteria:** Report games with large combined adjustments; evaluate error patterns; consider caps if needed.
  - **Notes:**

- [ ] **P2.12 Evaluate timezone offsets/DST policy (Hawaii/Arizona edge cases)**
  - **Files:** `config/teams.py`, `src/adjustments/travel.py`
  - **Issue:** Fixed tz offsets ignore DST; may be wrong for some weeks.
  - **Acceptance criteria:** Document policy; validate edge cases; optionally use kickoff local time/timezone if available.
  - **Notes:**

- [ ] **P2.13 Add centralized team-name normalization/aliasing**
  - **Files:** `config/teams.py`, any module doing lookups by team string
  - **Issue:** Exact string match required for rivalries/altitude/locations/HFA tables; CFBD naming drift can silently break.
  - **Acceptance criteria:** One normalization layer used before all lookups; log unknown team names; add coverage report.
  - **Notes:**

---

## P3 — Performance / maintainability

- [ ] **P3.1 Use sparse matrix for ridge opponent adjustment**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** Dense X wastes memory; scales poorly.
  - **Acceptance criteria:** Sparse design matrix; results within tolerance; faster/more memory efficient.
  - **Notes:**

- [ ] **P3.2 Vectorize play preprocessing & avoid O(T×N) filtering**
  - **Files:** `src/models/efficiency_foundation_model.py`
  - **Issue:** `df.apply(axis=1)` and per-team slicing loops are slow.
  - **Acceptance criteria:** Vectorized computations / groupby aggregation; same outputs; improved runtime.
  - **Notes:**

- [ ] **P3.3 Separate legacy ridge path from EFM in sweeps and reporting**
  - **Files:** `scripts/backtest.py`
  - **Issue:** Risk of confusing results between model types.
  - **Acceptance criteria:** Explicit labeling; prevent mixing incompatible settings; clear output.
  - **Notes:**

- [ ] **P3.4 Add backtest sanity report**
  - **Files:** `scripts/backtest.py`
  - **Include:** game counts, predictions per week, line-match rate, open≠close rate, rating mean/std.
  - **Notes:**

---

## Suggested fix order (Top 10)

1. P0.1 Trajectory leakage  
2. P0.2 game_id joins for ATS + VegasComparison  
3. P0.3 EFM scaling/double-normalization + rounding contamination  
4. P0.4 robust season data fetching (no silent truncation)  
5. P1.1/P1.2 travel direction + tz logic  
6. P1.5 unify turnover play types  
7. P2.1 home/away context + isolate implicit HFA in EFM ridge  
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
