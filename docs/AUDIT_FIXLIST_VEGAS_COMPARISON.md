
# AUDIT_FIXLIST_VEGAS_COMPARISON.md

**Target:** `src/predictions/vegas_comparison.py`  
**Goal:** Make Vegas line comparison/value-play identification reliable, deterministic, and diagnostics-friendly.

---

## P0 — Must fix (can change ATS/value-play correctness)

- [x] **P0.1 Use `game_id` in `compare_prediction()` (currently unused)** -- FIXED 2026-02-05
  - **Issue:** `compare_prediction()` calls `get_line(home_team, away_team)` and ignores the game_id-capable lookup path. This keeps the system vulnerable to rematches and naming mismatches.
  - **Acceptance criteria:**
    - If prediction has `game_id`, line matching uses `lines_by_id` (preferred) via `get_line(..., game_id=...)`.
    - Only fall back to `(home_team, away_team)` when `game_id` is missing.
    - Log (or expose) how often fallback matching is used.
  - **Fix applied:** compare_prediction() now checks prediction.game_id first, uses lines_by_id lookup if available. Falls back to team-name matching with warning log when game_id lookup fails. Added game_id field to PredictedSpread dataclass in spread_generator.py.

---

- [x] **P0.2 Add `game_id` to ValuePlay outputs** -- FIXED 2026-02-05
  - **Issue:** `ValuePlay` does not include `game_id`, making it hard to join to outcomes, lines, CLV, or movement without team-name matching.
  - **Acceptance criteria:**
    - `ValuePlay` includes `game_id` when available.
    - DataFrame outputs include `game_id`.
  - **Fix applied:** Added `game_id` field to ValuePlay dataclass. `identify_value_plays()` passes game_id from prediction. `value_plays_to_dataframe()` includes game_id column. `compare_prediction()` returns game_id from matched VegasLine.

---

## P1 — High impact correctness & determinism

- [x] **P1.1 Deterministic provider fallback (avoid reliance on list order)** -- FIXED 2026-02-05
  - **Issue:** If the configured provider line isn't found, code falls back to `game.lines[0]`. If CFBD line order is not stable, results can change between runs.
  - **Fix applied:** Fallback now sorts available lines alphabetically by provider name (lowercased) and filters to those with non-null spreads, taking the first. Deterministic across runs.

---

- [x] **P1.2 Duplicate line handling for a single `game_id`** -- FIXED 2026-02-05
  - **Issue:** `self.lines_by_id[game_id] = vl` silently overwrites if multiple line entries appear for the same game.
  - **Fix applied:** Added duplicate detection with warning log. First-encountered line is kept; subsequent duplicates are logged but not overwritten.

---

- [x] **P1.3 Preserve signed edge alongside absolute edge in outputs** -- FIXED 2026-02-05
  - **Issue:** `value_plays_to_dataframe()` outputs `edge = abs(vp.edge)`, losing sign and making bias diagnostics harder.
  - **Fix applied:** Added `edge_signed` column to `value_plays_to_dataframe()` output preserving the raw signed edge (negative = model favors home more). Absolute `edge` column retained for sorting/display.

---

- [x] **P1.4 Make edge sorting robust when edge is missing** -- FIXED 2026-02-05
  - **Issue:** `sort_values("edge", key=abs, ...)` can behave inconsistently if `edge` is not guaranteed numeric/NaN.
  - **Fix applied:** Added `pd.to_numeric(df["edge"], errors="coerce")` before sorting in `generate_comparison_df()`. Non-numeric values become NaN, and `na_position="last"` ensures they sort to the bottom.

---

## P2 — Diagnostics & data quality checks (helps trust the results)

- [ ] **P2.1 Opener data reliability diagnostics**
  - **Issue:** `spread_open` is surfaced but its quality varies; users may treat it as a true opener.
  - **Acceptance criteria:**
    - When fetching lines, record/report:
      - % of games with `spread_open` present
      - % where `spread_open != spread_close` by > 0.5
    - Warn if opener appears unreliable.
  - **Claude nudge prompt:**
    > Add lightweight diagnostics to assess opener quality. Report how often spread_open exists and how often it differs meaningfully from spread_close, and warn if opener data appears unreliable.

---

- [ ] **P2.2 Improve `get_line_movement()` to support `game_id`**
  - **Issue:** `get_line_movement()` currently uses team-name matching only.
  - **Acceptance criteria:**
    - If `game_id` is available, movement lookup prefers it.
    - Fallback to team names only if needed.
  - **Claude nudge prompt:**
    > Make line movement retrieval consistent with the rest of the module by supporting `game_id` matching and using team-name matching only as fallback.

---

## Notes on sign convention (keep consistent)
- **Model spread (JP+ internal):** positive = home favored  
- **Vegas spread (CFBD):** negative = home favored  
- **Edge (Vegas convention):** `edge = (-model_spread) - vegas_spread`
  - edge < 0 ⇒ model likes **home** more than Vegas
  - edge > 0 ⇒ model likes **away** more than Vegas

---

## Suggested implementation order
1. P0.1 Use `game_id` in compare_prediction  
2. P0.2 Add `game_id` to ValuePlay outputs  
3. P1.1 Deterministic provider fallback  
4. P1.2 Duplicate handling by `game_id`  
5. P1.3 Preserve signed edge  
6. P1.4 Robust sorting  
7. P2.1 Opener diagnostics  
8. P2.2 game_id support in line movement

```
