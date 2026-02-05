# AUDIT_FIXLIST_EFM.md

**Target:** `src/models/efficiency_foundation_model.py`  
**Goal:** Ensure EFM opponent adjustment is statistically sound, caching is safe, preprocessing is robust, and diagnostics are reliable for backtests.

---

## Summary (what’s good)

- Vectorized preprocessing for success rate + garbage time weighting.
- Scrimmage-only filtering (prevents SR contamination).
- Neutral-field ridge regression support via home indicator.
- Turnover split into offense (ball security) and defense (takeaways).
- Sparse matrix ridge design (major memory win).
- Module-level caching to reduce repeated ridge fits.

---

## P0 — Must Fix (correctness; can materially change ratings)

- [x] **P0.1 Ridge intercept / coefficient interpretation is not clean** -- FIXED 2026-02-05
  - **Issue:** Ridge extraction builds "adjusted levels" as:
    - `off_adjusted = intercept + off_coef`
    - `def_adjusted = intercept - def_coef`
    This works operationally but can be hard to interpret and can embed intercept/baseline ambiguities into O/D decomposition.
  - **Acceptance criteria:**
    - Offense and defense adjusted metrics have a clearly defined baseline interpretation (e.g., "predicted league-average level + deviation" OR "deviation-only with explicit intercept").
    - Add invariants/logging that confirm:
      - `mean(adj_off_sr)` and `mean(adj_def_sr)` behave as expected relative to intercept.
  - **Fix applied:** Added invariant validation after ridge extraction - logs mean_off/mean_def vs intercept and warns if drift exceeds 5% tolerance. Comments clarify interpretation: intercept = league average, coefficients = deviations.

---

- [x] **P0.2 `home_team` missingness check is not robust (`None` vs `NaN`)** -- FIXED 2026-02-05
  - **Issue:** Ridge build uses:
    - `if home_teams[i] is not None`
    In pandas, missing values are often `np.nan`, so this check can silently treat missingness incorrectly and cause home indicator to be 0 for many plays.
  - **Acceptance criteria:**
    - Explicitly detect NaN/missing values in `home_team`.
    - Log missing rate and warn if home_info coverage is low (since it disables neutral-field regression).
  - **Claude nudge prompt:**
    > Make home/away indicator handling robust to pandas NaN values and add diagnostics on home_team coverage. Ensure neutral-field ridge regression is actually active when expected and warn if home-team data is missing.

---

- [x] **P0.3 Ridge cache data hash is too weak (collision risk)** -- FIXED 2026-02-05
  - **Issue:** `_compute_data_hash()` uses only a few summary fields (first/last team strings, sum(metric), n_plays). Different datasets can collide.
  - **Acceptance criteria:**
    - Data hash incorporates enough information to avoid accidental collisions (e.g., sample-based stable hash of offense/defense/metric/weights).
    - Optional: debug assertion verifying cached results align with expected dataset summaries.
  - **Claude nudge prompt:**
    > Strengthen the ridge cache key hash so cached ridge results cannot be reused for different underlying play datasets. Use a more robust hash strategy and add a lightweight consistency check for cached reuse.

---

## P1 — High impact (statistical stability / weekly consistency)

- [ ] **P1.1 Add explicit identifiability stabilization (centering) for ridge coefficients**
  - **Issue:** Offense/defense terms + intercept are partially confounded; ridge regularization helps but decomposition can drift.
  - **Acceptance criteria:**
    - Add explicit post-centering (or equivalent stabilization) so offense and defense coefficient sets are mean-zero (or otherwise constrained) and intercept retains league-average meaning.
    - Confirm coefficients are stable week-to-week in backtest diagnostics.
  - **Claude nudge prompt:**
    > Add an explicit identifiability/stabilization step to ridge opponent adjustment outputs (e.g., post-centering offense and defense coefficient sets) so decomposition and intercept interpretation remain stable across weeks.

---

- [ ] **P1.2 `avg_isoppp` mean computation uses sentinel filtering**
  - **Issue:** Excluding values equal to `LEAGUE_AVG_ISOPPP` can drop real teams near average and bias the mean.
  - **Acceptance criteria:**
    - Compute `avg_isoppp` using a missingness-aware method (e.g., compute mean across all teams or across valid play-derived teams without relying on equality checks).
    - Weekly `avg_isoppp` should not jump due to sentinel collision.
  - **Claude nudge prompt:**
    > Remove sentinel-value filtering from IsoPPP league-average computation and replace with a safer missingness strategy. Ensure near-average teams are not excluded and weekly averages remain stable.

---

- [ ] **P1.3 Clarify turnover diagnostics semantics post P2.6**
  - **Issue:** `turnover_rating` is now embedded inside O/D but also normalized separately for diagnostics. Downstream code may assume it adds to overall.
  - **Acceptance criteria:**
    - Make it explicit in naming/docs that turnover_rating is diagnostic after O/D embedding.
    - Add a note or guard preventing misuse of turnover_rating as an additive component.
  - **Claude nudge prompt:**
    > Clarify turnover component semantics after the O/D split (ball security vs takeaways). Ensure diagnostic turnover_rating cannot be mistakenly treated as a separate additive component to overall.

---

## P2 — Engineering correctness & robustness

- [ ] **P2.1 Avoid pandas SettingWithCopy pitfalls after filtering**
  - **Issue:** `df = df[keep_mask]` then assigning new columns can cause SettingWithCopy issues in pandas.
  - **Acceptance criteria:**
    - Ensure the filtered DataFrame is writable (explicit copy once after filtering).
    - Add an assertion verifying expected columns exist after preprocessing.
  - **Claude nudge prompt:**
    > Make preprocessing robust to pandas view/copy pitfalls. Ensure assignments after filtering operate on a guaranteed writable DataFrame, and add a small assertion/test verifying expected columns (is_success, weight, etc.) exist.

---

- [ ] **P2.2 Cache `settings` thresholds to reduce repeated calls**
  - **Issue:** `get_settings()` is called inside vectorized garbage-time functions; not expensive per call, but repeated across backtest weeks.
  - **Acceptance criteria:**
    - Cache garbage-time thresholds (and any frequently used settings) on the model instance or module level for repeated runs.
  - **Claude nudge prompt:**
    > Reduce repeated configuration lookups during repeated backtest calls by caching frequently used settings (e.g., garbage time thresholds) in the model instance.

---

## P3 — Performance (nice-to-have; improves sweep/backtest runtime)

- [ ] **P3.1 Vectorize sparse matrix COO construction**
  - **Issue:** Sparse matrix build loop is still Python-level O(N_plays).
  - **Acceptance criteria:**
    - Replace per-row loop with vectorized construction (using categorical codes or vectorized indexing).
    - Confirm results match baseline within tolerance.
  - **Claude nudge prompt:**
    > Optimize ridge design-matrix construction by vectorizing sparse COO assembly (avoid Python loops over plays). Preserve correctness and verify coefficients match baseline within tolerance.

---

## Diagnostics to add (high ROI)

- [ ] **D.1 Ridge sanity logging (debug-level)**
  - Include:
    - intercept, learned_hfa
    - mean(adj_off_sr), mean(adj_def_sr)
    - home_team missing rate (if used)
    - cache hit/miss stats (already partially done)
  - **Claude nudge prompt:**
    > Add lightweight ridge sanity logs that confirm neutral-field regression is working as intended (intercept meaning, learned HFA sign/magnitude, and mean adjusted values). Keep logs debug-level to avoid noise.

---

## Suggested fix order

1) P0.2 home_team missingness handling  
2) P0.3 cache hash strengthening  
3) P0.1 intercept/baseline interpretation cleanup + invariants  
4) P1.2 IsoPPP mean computation fix  
5) P2.1 pandas copy safety after filtering  
6) P1.1 identifiability stabilization (post-centering)  
7) P3.1 vectorize sparse matrix build (optional)

---

## Backtest Validation (2026-02-05, post P0.2/P0.3 fixes)

Baseline confirmed after EFM P0 structural fixes:
- **pd.notna()** now correctly handles home_team None/NaN in ridge regression
- **Cache hash** strengthened with team sequence + metric stats
- **Home_team coverage:** 100% after game join in backtest.py

---

## Definition of "done"

- Neutral-field ridge regression actually uses home indicators (when present) and logs coverage.
- Cached ridge results are safe against dataset collisions.
- Adjusted metric baselines are interpretable and stable across weeks.
- IsoPPP averages are computed without sentinel hacks.
- Preprocessing is pandas-safe and deterministic.
- Backtest results are stable run-to-run and diagnostics confirm expected behavior.