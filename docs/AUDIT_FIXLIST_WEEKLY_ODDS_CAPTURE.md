# AUDIT_FIXLIST_WEEKLY_ODDS_CAPTURE.md

**Target:** `scripts/weekly_odds_capture.py`  
**Goal:** Capture reliable opening/closing lines for later backtests/CLV with correct season/week labeling, deterministic storage, and strong DB integrity.

---

## Summary (current risks)

The script works for basic capturing, but there are several issues that will cause pain later:

- Week detection is heuristic and can mislabel data.
- SQLite upsert uses `INSERT OR REPLACE`, which can delete rows and orphan line records.
- Snapshot schema mixes “type” with “label,” complicating queries and reuse.
- Stored IDs likely won’t match CFBD game IDs, so you need better join metadata.

---

## P0 — Must Fix (correctness / data integrity)

- [x] **P0.1 Replace heuristic `get_current_week()` for storage keys** -- FIXED 2026-02-05
  - **Issue:** Current week calculation based on month/day is not reliable and will mislabel weeks around boundaries, Week 0/1, and postseason.
  - **Acceptance criteria:**
    - Captures can be run with explicit `--year` and `--week` (preferred), OR
    - Week is derived robustly from games/commence_time and an authoritative schedule source.
    - Stored snapshot includes correct season/week for later joins.
  - **Fix applied:** Added `--year` and `--week` CLI arguments. When provided, these are used directly. Heuristic `get_current_week()` is kept as fallback but now logs a warning recommending explicit args. Both `capture_odds()` and `preview_odds()` accept explicit year/week.

---

- [x] **P0.2 Fix SQLite upsert logic (avoid `INSERT OR REPLACE`)** -- FIXED 2026-02-05
  - **Issue:** `INSERT OR REPLACE` can delete the old snapshot row and insert a new one, breaking foreign-key relationships (`odds_lines.snapshot_id`) and causing orphaned rows.
  - **Acceptance criteria:**
    - Use a safe upsert strategy that preserves snapshot identity.
    - After insert/update, snapshot_id is correct and stable.
    - No orphan `odds_lines` remain after repeated captures.
  - **Fix applied:** Replaced both `INSERT OR REPLACE` with `INSERT...ON CONFLICT DO UPDATE`. Snapshots: on conflict updates captured_at/credits_used, then retrieves actual id via SELECT. Lines: on conflict updates spread/price/last_update fields. Row identity preserved across re-runs.

---

- [x] **P0.3 Enable and enforce foreign keys** -- FIXED 2026-02-05
  - **Issue:** SQLite foreign keys are off by default; your schema declares FKs but they may not be enforced.
  - **Acceptance criteria:**
    - `PRAGMA foreign_keys = ON;` is applied for the connection.
    - Inserting lines with invalid snapshot_id is prevented or logged.
  - **Fix applied:** Added `PRAGMA foreign_keys = ON` immediately after connection creation in `init_database()`. FK violations will now raise errors instead of silently inserting orphaned records.

---

## P1 — High Impact (schema clarity + future usability)

- [x] **P1.1 Normalize snapshot schema fields** -- FIXED 2026-02-05
  - **Issue:** `snapshot_type` column stores `opening_YYYY_weekW` labels, but the name implies the value should just be `opening`/`closing`.
  - **Fix applied:** Added `season` and `week` INTEGER columns to `odds_snapshots` table. INSERT now populates these from explicit year/week args. Migration logic adds columns to existing DBs via ALTER TABLE (silently skips if already present). `snapshot_type` still stores the compound label for backward compat with UNIQUE constraint; season/week columns enable clean queries.

---

- [x] **P1.2 Store join metadata for mapping Odds API games to CFBD games** -- FIXED 2026-02-05
  - **Issue:** Odds API `game_id` will not match CFBD game IDs; later reconciliation requires more keys.
  - **Fix applied:** Added `cfbd_game_id` INTEGER column (nullable placeholder) to `odds_lines` table for future CFBD reconciliation. Existing columns `commence_time`, `home_team`, `away_team` already serve as join candidates. Migration logic adds column to existing DBs.

---

- [x] **P1.3 Add basic invariants/sanity checks when inserting lines** -- FIXED 2026-02-05
  - **Issue:** You store both spread_home and spread_away but don't check they are consistent.
  - **Fix applied:** Added per-line consistency check: if `abs(spread_home + spread_away) > 0.5`, logs a warning with game/sportsbook details. After insertion loop, logs total anomaly count. Detects bad provider data early without blocking insertion.

---

## P2 — Operational / quality improvements

- [x] **P2.1 Make preview grouping stable (prefer game_id)** -- FIXED 2026-02-05
  - **Issue:** Preview groups by `(home_team, away_team)` which can fail if home/away swaps for neutral-site listings.
  - **Fix applied:** Preview now groups by `game_id` when available, falling back to `(home_team, away_team)` tuple key when `game_id` is missing. Game info (home, away, commence) stored in the group dict for display.

---

- [ ] **P2.2 Store additional markets if available (totals, moneyline)**
  - **Issue:** Future model evaluation may need totals and ML, but schema doesn’t store them.
  - **Acceptance criteria:** If OddsAPI returns these markets, store them with prices.
  - **Claude nudge prompt:**
    > If OddsAPI provides totals and moneylines, extend the schema to store them now. This improves long-term usefulness and allows more complete market comparisons.

---

- [x] **P2.3 Normalize timestamps to UTC** -- FIXED 2026-02-05
  - **Issue:** `datetime.now()` is local; provider timestamp may be UTC; stored ISO strings may be inconsistent.
  - **Fix applied:** `captured_at` now uses `datetime.now(timezone.utc).isoformat()` for UTC-normalized timestamps. Provider `snapshot_time` and `commence_time` are preserved as-is from the API (already UTC). `get_current_week()` retains local time (appropriate for "what day is it?" logic).

---

## Suggested implementation order

1) P0.2 safe upsert (avoid REPLACE)  
2) P0.3 enforce foreign keys  
3) P0.1 explicit year/week or robust derivation  
4) P1.1 snapshot schema normalization  
5) P1.2 join metadata storage  
6) P1.3 invariants/sanity checks  
7) P2.1 preview grouping  
8) P2.2 additional markets  
9) P2.3 UTC timestamps

---

## Definition of “done”

- Captures are labeled with correct season/week and are reproducible.
- Re-running captures does not orphan line rows or corrupt snapshot references.
- Data stored is sufficient to reconcile Odds API games with CFBD games later.
- Logs clearly report what was captured and any anomalies.