
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

- [ ] **P1.1 Deterministic provider fallback (avoid reliance on list order)**
  - **Issue:** If the configured provider line isn’t found, code falls back to `game.lines[0]`. If CFBD line order is not stable, results can change between runs.
  - **Acceptance criteria:**
    - When provider is missing, fallback uses an explicit deterministic rule (e.g., priority list or stable sorting).
    - Behavior is documented and logged once per fetch.
  - **Claude nudge prompt:**
    > Make provider selection deterministic. If the configured provider line is missing, choose a fallback line using an explicit and stable rule rather than relying on list order.

---

- [ ] **P1.2 Duplicate line handling for a single `game_id`**
  - **Issue:** `self.lines_by_id[game_id] = vl` silently overwrites if multiple line entries appear for the same game.
  - **Acceptance criteria:**
    - Detect duplicate `game_id` entries and select consistently.
    - Warn or log if duplicates occur.
  - **Claude nudge prompt:**
    > Add duplicate-handling for betting lines keyed by `game_id`. If multiple candidate lines exist for the same game, select one deterministically and log the duplicate situation for transparency.

---

- [ ] **P1.3 Preserve signed edge alongside absolute edge in outputs**
  - **Issue:** `value_plays_to_dataframe()` outputs `edge = abs(vp.edge)`, losing sign and making bias diagnostics harder.
  - **Acceptance criteria:**
    - Outputs include both `edge_signed` and `edge_abs` (or similar).
    - Sorting still uses absolute edge.
  - **Claude nudge prompt:**
    > Preserve both signed and absolute edge in outputs. Use absolute edge for sorting/thresholding, but keep signed edge for diagnostics (home/away bias, directionality).

---

- [ ] **P1.4 Make edge sorting robust when edge is missing**
  - **Issue:** `sort_values("edge", key=abs, ...)` can behave inconsistently if `edge` is not guaranteed numeric/NaN.
  - **Acceptance criteria:**
    - `edge` column is numeric with NaNs for missing values.
    - Sorting never throws and correctly places missing values last.
  - **Claude nudge prompt:**
    > Ensure comparison DataFrame sorting by abs(edge) is robust even when edge is missing. Enforce numeric edge dtype with NaNs and verify missing values sort last reliably.

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
