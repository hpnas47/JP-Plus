# AUDIT_FIXLIST_BACKTEST.md

**Target:** `scripts/backtest.py`  
**Goal:** Ensure the walk-forward backtest is chronologically correct (especially postseason), data joins are reliable, and runtime is efficient/deterministic.

---

## Summary (what’s already strong)

- Uses `game_id` for ATS joins (major correctness win).
- Avoids rounded EFM outputs and uses full-precision ratings.
- Scrimmage play filtering is applied at ingest.
- Data leakage guards (`assert max_train_week < pred_week`) are excellent.
- CLV reporting + sanity checks are valuable.

---

## P0 — Must Fix (correctness; can materially change evaluation)

- [x] **P0.1 Fix postseason week mapping (don't lump everything into week=16 if doing walk-forward)** -- FIXED 2026-02-05
  - **Issue:** All postseason games are assigned to week 16. This breaks chronology (training can include “future” bowls while predicting earlier bowls).
  - **Acceptance criteria:**
    - Postseason games are either:
      - excluded from walk-forward, OR
      - mapped into sequential pseudo-weeks based on date ordering (week 16/17/18...).
    - No “future postseason game” is used to train earlier postseason predictions.
  - **Claude nudge prompt:**
    > Audit postseason handling and week assignment. Ensure postseason games are not all lumped into a single week in a way that breaks walk-forward chronology. Either exclude postseason from walk-forward prediction or map postseason into sequential weeks based on start_date so training always precedes prediction.

---

- [x] **P0.2 Ensure postseason plays are fully fetched (current method may be incomplete)** -- FIXED 2026-02-05
  - **Issue:** `get_plays(year, week=1, season_type="postseason")` may not return all postseason plays depending on CFBD behavior.
  - **Acceptance criteria:**
    - Postseason play coverage is validated (e.g., ratio of postseason games to postseason plays).
    - If needed, fetch postseason plays by iterating postseason game_ids or a supported postseason structure.
  - **Fix applied:** `fetch_season_plays()` now loops through weeks 1-5 for postseason plays (guarding against API behavior changes). `fetch_all_season_data()` adds a postseason coverage sanity check comparing games with plays to total postseason games, logging warnings when coverage is incomplete.

---

- [x] **P0.3 Validate `home_team` field on play rows used for neutral-field ridge regression** -- FIXED 2026-02-05
  - **Issue:** `home_team` in play ingest is derived from `play.home`, which may not be the home team string. If wrong, EFM home-indicator regression silently fails.
  - **Acceptance criteria:**
    - Play rows contain a correct `home_team` string matching the game’s home_team.
    - Add validation (assert or warning) if home_team coverage is missing or mismatched.
    - Prefer joining plays to games by `game_id` to populate home_team reliably.
  - **Claude nudge prompt:**
    > Audit the play ingestion schema for the `home_team` field used in neutral-field ridge regression. Ensure it is the actual home team name string and not a boolean/other flag. Prefer populating home_team via join to games on game_id and add validation of coverage/mismatch rate.

---

- [x] **P0.4 Fix unmatched ATS mask logic (minor but misleading)** -- FIXED 2026-02-05
  - **Issue:** `merged["game_id"].isna()` is not meaningful after merge because predictions always have game_id; unmatched should be determined by missing Vegas columns.
  - **Acceptance criteria:**
    - Unmatched games are correctly detected from missing spread fields after merge.
    - Unmatched logging remains accurate.
  - **Claude nudge prompt:**
    > Clean up ATS merge unmatched detection so it correctly identifies unmatched rows based on missing betting-line fields rather than game_id NaNs. Preserve the current behavior but make the logic correct and maintainable.

---

## P1 — High impact (performance + determinism)

- [x] **P1.1 Convert `games_df` to pandas once per season (not inside weekly loop)** -- FIXED 2026-02-05
  - **Issue:** `games_df.to_pandas()` is performed inside the prediction loop each week.
  - **Acceptance criteria:**
    - Convert once outside the weekly loop and reuse.
    - Ensure outputs are unchanged.
  - **Fix applied:** Moved `games_df_pd = optimize_dtypes(games_df.to_pandas())` before the weekly loop. Removed per-week conversion at old line 756. Backtest results identical.

---

- [x] **P1.2 Avoid materializing `train_game_ids` Python lists if possible** -- FIXED 2026-02-05
  - **Issue:** Building a large Python list of game_ids each week can be slower than a join/filter.
  - **Acceptance criteria:**
    - Prefer joining plays to games and filtering by week in Polars (or another efficient approach).
    - Keep behavior identical (training games strictly < pred_week).
  - **Fix applied:** Replaced `games_df.filter(...)["id"].to_list()` + `is_in(train_game_ids)` with Polars semi-join: `efficiency_plays_df.join(train_games_pl.select("id"), left_on="game_id", right_on="id", how="semi")`. Also reuses `train_games_pl` for pandas conversion instead of re-filtering. Backtest results identical.

---

## P2 — Evaluation quality improvements (trust the results)

- [x] **P2.1 Update week coverage sanity checks to account for postseason or explicitly exclude it** -- FIXED 2026-02-05
  - **Issue:** Sanity checks expect only weeks 1–15 but data may include week 16 postseason.
  - **Fix applied:** `print_data_sanity_report()` and `fetch_all_season_data()` now separate regular-season (1-15) from postseason (16+) in coverage checks. Missing-week warnings specify "regular-season." Postseason pseudo-week count reported separately. Warning added when postseason games exist but no postseason plays found.

---

- [x] **P2.2 Add explicit sanity checks for postseason play coverage** -- FIXED 2026-02-05
  - **Issue:** It's easy to include postseason games but miss postseason plays.
  - **Fix applied:** Enhanced P0.2 postseason coverage check to include total play count and average plays/game. Warns if avg plays/game < 80 (suspiciously low). Debug log includes complete postseason statistics (games, plays, plays/game).

---

## P3 — Maintainability / cleanup

- [x] **P3.1 Remove unused imports to reduce cognitive load** -- FIXED 2026-02-05
  - **Issue:** Some imports are unused (e.g., DataProcessor/RecencyWeighter, VegasComparison).
  - **Acceptance criteria:** Lint clean (or at least obvious unused imports removed).
  - **Claude nudge prompt:**
    > Clean up unused imports and dead code paths in backtest.py to reduce cognitive overhead and avoid confusion about which components are active.

---

## Suggested implementation order

1) P0.3 home_team validation/join  
2) P0.2 postseason plays completeness  
3) P0.1 postseason week mapping / chronology  
4) P1.1 pandas conversion once per season  
5) P0.4 ATS unmatched detection cleanup  
6) P1.2 training filter optimization  
7) P2.1/P2.2 sanity checks for postseason  
8) P3.1 cleanup

---

## Backtest Validation (2026-02-05, post P0 structural fixes)

Baseline metrics after P0.1/P0.3/P0.4/P3.1 fixes (no weight changes):
- **MAE (vs actual):** 12.88 | **MAE (vs closing):** 4.89
- **Overall ATS:** 1221-1193-45 (50.6%)
- **Phase 2 (Core, wk 4-15) ATS:** 52.0% | **5+ edge:** 55.3%
- **Postseason ATS:** 40.9% (needs investigation)
- **Postseason pseudo-weeks:** 16-37 (correctly mapped by date)
- **Home_team coverage:** 100% via game join

---

## Definition of "done"

- Walk-forward chronology is correct for all included games (regular + postseason if enabled).
- Postseason data ingestion is complete or explicitly excluded.
- Play `home_team` field is validated and correct for neutral-field ridge.
- ATS matching and sanity logging remain accurate.
- Backtest runtime is improved by eliminating repeated conversions and large list materialization.