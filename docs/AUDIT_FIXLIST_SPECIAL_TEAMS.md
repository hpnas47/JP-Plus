
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

- [ ] **P0.2 Deprecate or fix `calculate_team_rating()` (currently inconsistent)**
  - **Issue:** `calculate_team_rating()` combines values that are not on the same scale and attempts to normalize via hardcoded divisors.
  - **Acceptance criteria:**
    - Either remove/deprecate this method from production usage, or rewrite it so its output matches the “PBTA points per game” definition.
    - If kept, it must be numerically consistent with the play-by-play aggregation pipeline.
  - **Claude nudge prompt:**
    > Review `calculate_team_rating()` and either (a) deprecate it in favor of the play-by-play aggregation pipeline, or (b) make it produce true PBTA points-per-game consistent with the rest of the model.

---

- [ ] **P0.3 Deprecate or fix `calculate_from_game_stats()` (mixed units)**
  - **Issue:** Punt rating uses gross yards vs expected (yards) while FG rating is points; `overall` becomes mixed.
  - **Acceptance criteria:**
    - Either mark this as non-production and do not use it for spreads, or convert punt component to points consistently before summing.
  - **Claude nudge prompt:**
    > Audit `calculate_from_game_stats()` for unit correctness. Ensure it does not mix yards with points in the returned `overall_rating`. If it cannot be made accurate, treat it as a fallback-only path and prevent it from being used for spread adjustments.

---

## P1 — High impact (stability, scaling, realism)

- [ ] **P1.1 Kickoff rating scaling sanity (avoid double-normalization)**
  - **Issue:** Coverage/return calculations include extra divisors (e.g., dividing by 5 or 3 after already normalizing by per-game rates), likely shrinking effects too much and making magnitudes hard to interpret.
  - **Acceptance criteria:**
    - Kickoff coverage and return components are expressed as **points per game** with realistic magnitudes.
    - Provide a distribution sanity output (mean/std/min/max across teams).
  - **Claude nudge prompt:**
    > Re-evaluate kickoff rating scaling to avoid double-normalization by plays-per-game. Ensure coverage and return components produce realistic per-game point magnitudes, and add a distribution report (mean/std/min/max) across teams.

---

- [ ] **P1.2 Punt touchback/net-yards handling is overly ad-hoc**
  - **Issue:** Touchback net yard logic uses `min(gross, 55)` which is not grounded in actual field position mechanics. Inside-20/touchback bonuses may double-count what net yards already capture.
  - **Acceptance criteria:**
    - Punt value calculation is defensible and stable (net + field position bonuses/penalties without double-counting).
    - Touchback handling reflects realistic field position impact.
  - **Claude nudge prompt:**
    > Review punt modeling (net yards, touchbacks, inside-20) for realism and to avoid double-counting. Ensure punt_value reflects field position impact in points per game with stable magnitude across teams.

---

- [ ] **P1.3 Remove remaining row-wise apply hotspots (kickoffs)**
  - **Issue:** `kickoff_plays.apply(... axis=1)` remains, while other components are vectorized.
  - **Acceptance criteria:**
    - Kickoff parsing/flags use vectorized operations where practical.
    - Runtime improves without changing results materially.
  - **Claude nudge prompt:**
    > Optimize kickoff play processing to avoid row-wise `apply(axis=1)` where possible. Maintain identical outputs but improve performance and reduce overhead.

---

## P2 — Data quality / robustness (prevents silent bias)

- [ ] **P2.1 Add parse coverage diagnostics for play_text-based extraction**
  - **Issue:** FG/punt/kickoff logic relies on regex parsing of `play_text`, which can fail or misparse. Today this failure is mostly silent.
  - **Acceptance criteria:**
    - Log parse coverage rates:
      - % of FG plays with distance parsed
      - % of punts with gross parsed
      - % of kickoffs with return yards parsed (or defaulted)
    - Warn if coverage is below a threshold (e.g., <80%).
  - **Claude nudge prompt:**
    > Add diagnostics and safeguards around `play_text` parsing for FG/punt/kickoffs. Report parse success rates and warn when coverage is low. Ensure parsing failures do not silently bias team ratings.

---

- [ ] **P2.2 Prefer structured fields over play_text parsing when available**
  - **Issue:** Regex parsing is inherently brittle; if CFBD provides structured distance/return fields, use them.
  - **Acceptance criteria:**
    - Use structured fields when present, fall back to play_text parsing otherwise.
    - Document precedence and log which path is used.
  - **Claude nudge prompt:**
    > Where possible, prefer structured play fields (distance, return yards, touchback indicators) over regex parsing of play_text. Keep play_text parsing as a fallback and log which data path is used.

---

- [ ] **P2.3 Clarify public API to avoid partial state confusion**
  - **Issue:** `calculate_fg_ratings_from_plays()` writes FG-only ratings into `team_ratings`. If users call it directly, they may assume `overall_rating` is full ST when it is FG-only.
  - **Acceptance criteria:**
    - Public API makes it hard to accidentally use partial ST ratings.
    - Either:
      - treat FG-only calc as internal helper, or
      - clearly label/encode partial ratings and require merge step for overall.
  - **Claude nudge prompt:**
    > Clarify the SpecialTeamsModel public API so callers can’t easily mistake FG-only ratings for full special teams ratings. Make “FG-only” vs “full ST” explicit and enforceable.

---

## Definition of “done”

- All special teams outputs are **PBTA points per game**.
- `overall_rating` is always a sum of per-game point components (FG + punt + kickoff) in the “official” pathway.
- Parsing is measurable (coverage rates logged) and failures are not silent.
- Kickoff and punt values have realistic magnitudes and do not introduce obvious outlier distortions.
- Backtests show ST adjustments are stable and do not degrade performance.

---

```
