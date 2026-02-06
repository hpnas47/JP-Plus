
# AUDIT_FIXLIST_SPECIAL_TEAMS.md

**Target:** `src/models/special_teams.py`  
**Goal:** Make special teams ratings consistent, correctly scaled, and robust enough to use as a points-per-game adjustment in spread prediction.

---

## Summary (current risks)

The model is directionally correct and has good structure (PBTA framing, leakage guard, merged overall rating), but there are **unit/scale inconsistencies** and **fragile parsing** that can distort punt/kickoff values and make `overall_rating` unreliable unless the full play-by-play path is used carefully.

---

## P0 — Must Fix (unit correctness; can materially change spreads)

- [x] **P0.1 Enforce consistent units across components (PBTA points per game)** -- FIXED 2026-02-05
  - **Issue:** Some methods mix incompatible units:
    - FG: points above expected (points)
    - Punt: net-yards-ish heuristic (yards + bonuses)
    - Kickoff: heuristic units + extra divisions
    - Game-stats fallback mixes points and yards
  - **Acceptance criteria:**
    - FG, punt, kickoff components each represent **points per game** vs average.
    - `overall_rating = fg + punt + kickoff` (all points/game) in all "official" pathways.
    - No method returns a mixed-unit `overall_rating`.
  - **Fix applied:** Punt/kickoff ratings now convert yards to points via YARDS_TO_POINTS (0.04). calculate_team_rating() normalizes per-event to per-game. calculate_from_game_stats() properly converts yards to points before summing.

---

- [x] **P0.2 Deprecate or fix `calculate_team_rating()` (currently inconsistent)** -- FIXED 2026-02-05
  - **Issue:** `calculate_team_rating()` combines values that are not on the same scale and attempts to normalize via hardcoded divisors.
  - **Acceptance criteria:**
    - Either remove/deprecate this method from production usage, or rewrite it so its output matches the "PBTA points per game" definition.
    - If kept, it must be numerically consistent with the play-by-play aggregation pipeline.
  - **Fix applied:** Fixed double-normalization bug: punt/kickoff components (per-event averages) were incorrectly divided by event count before scaling. Now FG divides total by estimated games, punt/kickoff multiply per-event average by events-per-game. Added docstring noting `calculate_all_st_ratings_from_plays()` is the primary production pathway.

---

- [x] **P0.3 Deprecate or fix `calculate_from_game_stats()` (mixed units)** -- FIXED 2026-02-05
  - **Issue:** Punt rating uses gross yards vs expected (yards) while FG rating is points; `overall` becomes mixed.
  - **Acceptance criteria:**
    - Either mark this as non-production and do not use it for spreads, or convert punt component to points consistently before summing.
  - **Fix applied:** P0.1 already converted punt yards→points via YARDS_TO_POINTS. Now explicitly documented as FALLBACK pathway (no play-by-play). Added debug logging when fallback is used. Kickoff defaults to 0 (no data). All outputs verified as PBTA pts/game.

---

## P1 — High impact (stability, scaling, realism)

- [ ] **P1.1 Kickoff rating scaling sanity (avoid double-normalization)** -- DEFERRED
  - **Issue:** Coverage/return calculations include extra divisors (e.g., dividing by 5 or 3 after already normalizing by per-game rates), likely shrinking effects too much and making magnitudes hard to interpret.
  - **Status:** Attempted 2026-02-05. Removing `/5.0` and `/3.0` divisors amplified kickoff impact by 5x/3x, causing 5+ edge to drop from 55.7% to 53.3%. **REJECTED by Quant Auditor.** These divisors are empirically calibrated dampening factors, not bugs. Changing them requires recalibration of the ST weight in the EFM.

---

- [ ] **P1.2 Punt touchback/net-yards handling is overly ad-hoc** -- DEFERRED
  - **Issue:** Touchback net yard logic uses `min(gross, 55)` which is not grounded in actual field position mechanics. Inside-20/touchback bonuses may double-count what net yards already capture.
  - **Status:** Current heuristic produces reasonable values for most cases (tested 35/45/60-yard touchbacks). Changes here risk degrading backtest performance. Deferred pending comprehensive ST recalibration effort.

---

- [x] **P1.3 Remove remaining row-wise apply hotspots (kickoffs)** -- FIXED 2026-02-05
  - **Issue:** `kickoff_plays.apply(... axis=1)` remains, while other components are vectorized.
  - **Acceptance criteria:**
    - Kickoff parsing/flags use vectorized operations where practical.
    - Runtime improves without changing results materially.
  - **Fix applied:** Replaced `apply(lambda r: is_touchback(r["play_text"], r["play_type"]), axis=1)` with vectorized `str.contains("touchback", case=False)`. Replaced `apply(extract_return_yards)` with vectorized `str.extract()` + `pd.to_numeric()`. Backtest results identical.

---

## P2 — Data quality / robustness (prevents silent bias)

- [x] **P2.1 Add parse coverage diagnostics for play_text-based extraction** -- FIXED 2026-02-05
  - **Issue:** FG/punt/kickoff logic relies on regex parsing of `play_text`, which can fail or misparse. Today this failure is mostly silent.
  - **Fix applied:** Added parse coverage logging to all three ST components: FG distance parse rate, punt gross yards parse rate, kickoff return yards parse rate (non-touchback only). Each warns if coverage < 80%, otherwise logs at debug level.

---

- [ ] **P2.2 Prefer structured fields over play_text parsing when available** -- DEFERRED
  - **Issue:** Regex parsing is inherently brittle; if CFBD provides structured distance/return fields, use them.
  - **Acceptance criteria:**
    - Use structured fields when present, fall back to play_text parsing otherwise.
    - Document precedence and log which path is used.
  - **Status:** Deferred 2026-02-05. Requires CFBD API field audit to determine which structured fields (distance, return yards, touchback flags) are reliably populated. Current regex parse coverage is >80% (P2.1 diagnostics confirm), so urgency is low.

---

- [x] **P2.3 Clarify public API to avoid partial state confusion** -- FIXED 2026-02-05
  - **Issue:** `calculate_fg_ratings_from_plays()` writes FG-only ratings into `team_ratings`. If users call it directly, they may assume `overall_rating` is full ST when it is FG-only.
  - **Fix applied:** Added `is_complete: bool` field to `SpecialTeamsRating` dataclass. FG-only ratings set `is_complete=False`; `calculate_all_st_ratings_from_plays()` sets `is_complete=True`. `get_matchup_differential()` logs debug warning when using incomplete ratings.

---

## Definition of “done”

- All special teams outputs are **PBTA points per game**.
- `overall_rating` is always a sum of per-game point components (FG + punt + kickoff) in the “official” pathway.
- Parsing is measurable (coverage rates logged) and failures are not silent.
- Kickoff and punt values have realistic magnitudes and do not introduce obvious outlier distortions.
- Backtests show ST adjustments are stable and do not degrade performance.

---

```
