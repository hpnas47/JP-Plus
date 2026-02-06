# AUDIT_FIXLIST_PRESEASON_PRIORS.md

**Target:** `src/models/preseason_priors.py`  
**Goal:** Make preseason priors correct, internally consistent (especially talent scaling), robust to historical seasons, and efficient enough for sweeps/backtests.

---

## Summary (current risks)

This module is feature-rich (asymmetric regression, coaching change, portal impact, returning production) but has several high-impact risks:

- Talent is scaled in **two incompatible ways** (normalized z-score vs ad hoc linear transform).
- Coaching change inputs and “first-time HC exclusion” rules are **internally inconsistent** in the hardcoded tables.
- Portal impact logic is **slow** due to repeated `apply(axis=1)` and may be historically inconsistent due to conference realignment.
- Missing data and naming mismatches can silently degrade priors without clear diagnostics.

---

## P0 — Must Fix (correctness / consistency)

- [x] **P0.1 Unify talent scaling across the entire class** -- FIXED 2026-02-05
  - **Issue:** Talent is used on a normalized rating scale in preseason blending, but later "talent floor" uses a different ad hoc scaling derived from raw talent.
  - **Acceptance criteria:**
    - Store both `talent_raw` (API value) and `talent_rating_normalized` (rating-scale value).
    - Use a single normalized talent rating scale consistently for:
      - preseason talent blend
      - persistent talent floor
    - Remove or clearly deprecate the ad hoc `(raw_talent - 750) / 25.0` mapping.
  - **Fix applied:** Added `talent_rating_normalized` field to PreseasonRating dataclass. `blend_with_inseason()` now uses the z-score-normalized talent (same scale as preseason blending) instead of the ad hoc `(raw - 750) / 25.0`. Backtest 5+ edge improved from 53.2% to 54.7%.

---

- [x] **P0.2 Add sanity checks to ensure ranking direction and gaps behave as intended** -- FIXED 2026-02-05
  - **Issue:** Coaching-change logic depends on ranks meaning "lower = better." If talent data is ever rank-like (lower=better) rather than score-like (higher=better), everything flips.
  - **Acceptance criteria:**
    - Add a sanity/validation routine that checks:
      - known elite talent teams appear near the top of the talent ranks
      - known poor talent teams appear near the bottom
      - reported `talent_gap` sign matches documentation for a few examples
    - Log intersection sizes between SP+/talent/returning production/portal datasets.
  - **Fix applied:** Added `_validate_data_quality()` method that logs dataset intersection sizes and validates rank direction using known elite programs (Alabama, Georgia, Ohio State, Texas, LSU). Checks talent top-20 presence, SP+ sign direction, and normalized vs raw talent consistency. Called at start of `calculate_preseason_ratings()`.

---

- [x] **P0.3 Align coaching-change tables with documented rules** -- FIXED 2026-02-05
  - **Issue:** Comments say COACHING_CHANGES should contain only coaches with prior HC experience, but the dict includes first-time HCs (even though they are later excluded).
  - **Acceptance criteria:**
    - Make the data structures consistent with your stated policy:
      - either remove first-time HCs from COACHING_CHANGES, or
      - rename/repurpose the table and rely on pedigree logic explicitly
    - Avoid ambiguous states where a coach is "listed but excluded."
  - **Fix applied:** Removed Dan Lanning from COACHING_CHANGES[2022] (he's already in FIRST_TIME_HCS). Cross-referenced all COACHING_CHANGES entries against FIRST_TIME_HCS — no other conflicts found. Added comment explaining the removal.

---

## P1 — High impact (MAE/ATS quality + runtime)

- [ ] **P1.1 Vectorize portal impact calculations (reduce DataFrame apply usage)**
  - **Issue:** Portal impact uses multiple `apply(axis=1)` passes (outgoing, incoming, and G5→P4 counts). This will be slow and dominate runtime in sweeps.
  - **Acceptance criteria:**
    - Reduce row-wise apply usage where practical.
    - Preserve outputs within tolerance (or document expected small differences).
    - Keep deterministic ordering and reproducibility.
  - **Claude nudge prompt:**
    > Refactor portal impact computation to reduce row-wise DataFrame apply usage. Preserve the same logic and outputs, but improve runtime and determinism for backtests and sweeps.

---

- [ ] **P1.2 Clarify and standardize “continuity tax” semantics**
  - **Issue:** Continuity tax is applied by dividing outgoing losses by 0.90 (amplifying losses). The naming is easy to misread and encourages sign mistakes later.
  - **Acceptance criteria:**
    - Rename constants or restructure math so it is obvious this amplifies outgoing loss.
    - Add a comment explaining the intended magnitude and sign.
  - **Claude nudge prompt:**
    > Make continuity-tax handling easier to reason about. Ensure the naming and math clearly convey that outgoing losses are amplified (if that’s intended), so future edits don’t accidentally invert the effect.

---

- [ ] **P1.3 Resolve `portal_scale` default confusion**
  - **Issue:** `calculate_portal_impact()` defaults to 0.06 but `calculate_preseason_ratings()` passes 0.15; the internal default is misleading.
  - **Acceptance criteria:**
    - Ensure portal_scale is configured in one place (settings/constructor) and passed consistently.
    - Document the intended default and remove unused/confusing defaults.
  - **Claude nudge prompt:**
    > Standardize how portal_scale is defined and passed so defaults are consistent and documented. Avoid having multiple “defaults” that differ across functions.

---

## P2 — Robustness & historical correctness

- [ ] **P2.1 Conference-by-year semantics for P4/G5 classification**
  - **Issue:** Portal “level-up discount” uses conference data that may reflect current alignment rather than the season being modeled (realignment years).
  - **Acceptance criteria:**
    - If possible, use conference affiliation appropriate for the modeled year.
    - If not possible, document the approximation and add warnings for known realignment seasons/teams.
  - **Claude nudge prompt:**
    > Review conference and P4/G5 classification used for portal level-up discounts. Ensure classification aligns with the modeled season year (especially across realignment). If exact-year affiliation isn’t available, document and guard against mismatches.

---

- [ ] **P2.2 Improve metadata/name consistency diagnostics**
  - **Issue:** Team naming mismatches across SP+, talent, returning production, and portal data can silently reduce coverage.
  - **Acceptance criteria:**
    - Log intersection counts:
      - teams in SP+
      - teams in talent
      - teams in returning production
      - teams with portal impact
    - Report missing teams and consider a name normalization/alias layer if needed.
  - **Claude nudge prompt:**
    > Add diagnostics for team-name alignment across SP+/talent/returning production/portal datasets. Log intersection sizes and identify missing teams, and consider using a normalization/alias layer to reduce silent coverage loss.

---

## Suggested fix order

1) P0.1 talent scaling unification  
2) P0.2 sanity checks for rank direction + dataset intersections  
3) P0.3 coaching-change metadata cleanup  
4) P1.1 portal computation vectorization  
5) P1.2 continuity-tax clarity  
6) P1.3 portal_scale defaults cleanup  
7) P2.1 conference-by-year semantics  
8) P2.2 name alignment diagnostics

---

## Definition of “done”

- Talent is represented on a consistent rating scale across the entire module.
- Coaching change tables and rules match documentation and are testable.
- Portal impact calculations are reasonably fast and deterministic.
- The module reports coverage/intersections and warns when priors are built on weak/mismatched inputs.
- Backtest results remain stable and interpretability improves (no silent flips in sign or scaling).