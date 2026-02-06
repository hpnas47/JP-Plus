
# AUDIT_FIXLIST_FINISHING_DRIVES.md

**Target:** `src/models/finishing_drives.py`  
**Goal:** Ensure FinishingDrivesModel measures *true red zone trips* and produces a stable, low-noise finishing drives adjustment suitable for spread prediction.

---

## Summary (what’s wrong today)

The Bayesian regression logic in `calculate_team_rating()` is reasonable, but the play-by-play pathway `calculate_all_from_plays()` **does not actually compute red zone trips**. It counts certain play outcomes (TD/FG/turnover/failed 4th) and treats that as “trips,” which will systematically overcount/undercount and produce noisy or mis-scaled `overall_rating`.

If this adjustment is applied to spreads, the current implementation can introduce error rather than reduce it.

---

## P0 — Must Fix (correctness; can materially change model outputs)

- [x] **P0.1 Red zone trips must be computed at the drive/trip level (not by counting plays)** -- FIXED 2026-02-05
  - **Issue:** `calculate_all_from_plays()` estimates `total_trips` as:
    - RZ TD plays + RZ FG plays + RZ turnovers + `max(1, failed_4th)`
    - This is not equivalent to "red zone trips."
  - **Acceptance criteria:**
    - Red zone trips correspond to distinct offensive possessions/drive entries into the red zone.
    - One trip is counted once per drive, not once per play.
    - No fabricated minimum trips (no `max(1, ...)`).
  - **Fix applied:** RZ trips now count distinct (game_id, drive_id) combinations. Outcomes determined by last play of each drive. Requires drive_id + game_id columns (warns and skips team if missing). Removed all max(1,...) hacks.

---

- [x] **P0.2 Remove fabricated minimums that create fake opportunities** -- FIXED 2026-02-05
  - **Issue:** `max(1, rz_failed_4th)` and `max(1, len(gtg_plays)//3)` create fake trips/attempts.
  - **Acceptance criteria:**
    - If a team has 0 RZ trips or 0 GTG situations, totals remain 0 and regression handles small samples.
    - No forced minimum counts.
  - **Fix applied:** P0.1 refactor already removed all `max(1, ...)` hacks by switching to drive-level trip counting. Fixed residual bug: `rz_failed_4th` (undefined after P0.1) replaced with `rz_failed` (correctly counted per-trip failures). Teams with 0 RZ trips return expected-value defaults; Bayesian prior handles small samples.

---

## P1 — High impact (reduces noise / improves measurement fidelity)

- [ ] **P1.1 TD/FG/turnover detection must not rely on `play_type` containing "touchdown"** -- DEFERRED
  - **Issue:** TD detection via `play_type` substring search is unreliable in CFBD; many scoring plays are encoded differently and TD indication may live in play text or score deltas.
  - **Acceptance criteria:**
    - TD/FG/turnover outcomes are detected using a reliable signal (drive result, scoring flags, score delta, or play_text parsing if needed).
    - Add validation that counted TDs roughly reconcile with actual scoring totals (within tolerance).
  - **Status:** Attempted 2026-02-05. Adding `drive_id` + `scoring` to efficiency_plays enabled the finishing drives model (which was previously a no-op due to missing `drive_id`). This increased Core MAE from 12.43 to 12.52 (+0.09, exceeds 0.05 gate). **REJECTED by Quant Auditor.** Root cause: the finishing drives scaling formula (`(ppt - expected) * trips/10`) adds noise. Must recalibrate P2.1 scaling before enabling this code path.

---

- [x] **P1.2 `rz_failed` must represent failed *trips*, not failed 4th-down plays** -- FIXED by P0.1 (2026-02-05)
  - **Issue:** Current code passes `rz_failed=int(rz_failed_4th)` into `calculate_team_rating()`, which misrepresents failures (many failed trips end before 4th down).
  - **Fix:** P0.1 refactored to drive-level trip counting. `rz_failed` is now `trip_outcomes.count("FAILED")` — the count of RZ trips that did not end in TD, FG, or turnover. Semantics match acceptance criteria.

---

- [x] **P1.3 Goal-to-go calculation should be drive/opportunity-based (or removed if too noisy)** -- FIXED by P0.1 (2026-02-05)
  - **Issue:** GTG attempts estimated by `len(gtg_plays)//3` is arbitrary and style-dependent.
  - **Fix:** P0.1 refactored GTG to drive-level counting: `team_rz[gtg_mask].groupby(["game_id", "drive_id"])`. Each GTG trip is a distinct possession inside 10 yards. Arbitrary `//3` divisor removed.

---

## P2 — Calibration & stability improvements (important for spreads)

- [ ] **P2.1 Revisit `overall_rating` scaling to ensure stable magnitude**
  - **Issue:** `overall = (points_per_trip - expected_points) * (total_rz_trips / 10.0)` makes adjustment magnitude depend on trip count. If trips are noisy, magnitude becomes noisy.
  - **Acceptance criteria:**
    - `overall_rating` has a stable, interpretable scale.
    - Add diagnostics for typical ranges and identify outliers.
  - **Claude nudge prompt:**
    > Re-evaluate the scaling of finishing drives `overall_rating` so that the adjustment magnitude is stable and interpretable. Add diagnostics that report the distribution of trips and overall ratings, and ensure outliers are explainable.

---

- [x] **P2.2 Strengthen and clarify fallback pathways (drives > plays > game stats)** -- FIXED 2026-02-05
  - **Issue:** `calculate_from_game_stats()` uses proxies (points/game → estimated trips) which can be misleading.
  - **Fix applied:** Added hierarchy documentation to all three method docstrings: PRIMARY (calculate_all_from_plays) > SECONDARY (calculate_from_drives) > TERTIARY (calculate_from_game_stats). Added debug-level pathway logging to secondary and tertiary methods so callers can confirm which pathway was used.

---

## Existing good practice (keep)

- ✅ Data leakage guard in `calculate_all_from_plays()` is good—keep it (and extend to any new data pathway).
- ✅ Bayesian regression in `calculate_team_rating()` is conceptually sound.

---

## Suggested implementation order

1) P0.1 (true trip counting)  
2) P0.2 (remove fabricated minimums)  
3) P1.1 (robust scoring detection)  
4) P1.2 (failed trips semantics)  
5) P1.3 (GTG reliability or disable)  
6) P2.1 (scale diagnostics)  
7) P2.2 (fallback clarity + logging)

---

## Definition of “done”

- Trips are counted once per drive entry into the red zone.
- TD/FG/TO outcomes are assigned per trip reliably.
- No fabricated minimums.
- Output `overall_rating` distribution is stable (no unexplained extreme values).
- The adjustment improves or at least does not degrade backtest MAE/ATS when enabled.

```
