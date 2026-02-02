# JP+ Development Session Log

---

## Session: February 2, 2026

### Completed Today

- **Optimized Ridge Alpha from 100 to 50** - Sweep revealed alpha=50 is optimal
  - Tested alphas: 25, 40, 50, 60, 75, 85, 100, 150, 200
  - **MAE improvement:** 12.56 → 12.54 (-0.02 pts)
  - **3+ edge ATS:** 52.1% → 52.4% (+0.3%)
  - **5+ edge ATS:** 55.3% → 56.0% (+0.7%)
  - Lower alpha = less regularization = teams separate more
  - Updated default in `efficiency_foundation_model.py` and `backtest.py`

- **Tested EPA Regression vs SR+IsoPPP** - EPA regression performed worse
  - Hypothesis: Single clipped EPA regression could capture both efficiency and magnitude
  - **Result:** EPA regression had higher MAE (12.64 vs 12.55) and worse ATS (52.4% vs 54.5% at 5+ edge)
  - SR+IsoPPP remains the better approach - Success Rate filters noise, IsoPPP adds explosiveness
  - Experiment not committed (per user request)

- **Implemented Tiered FCS Penalties** - Elite vs Standard FCS teams now have different penalties
  - **Problem:** Flat 24pt penalty overestimates elite FCS teams (NDSU, Montana State, etc.) and underestimates weak FCS
  - **Analysis:** 2022-2024 FCS games show mean FBS margin of 30.3 pts vs standard FCS, but only +2 to +15 vs elite FCS
  - **Solution:** Tiered system with ELITE=18 pts, STANDARD=32 pts
  - **Elite FCS teams identified:** 23 teams including NDSU, Montana State, Sacramento State, Idaho, South Dakota State
  - **Backtest results (2022-2025):**
    | FCS Penalty Config | 3+ edge | 5+ edge |
    |-------------------|---------|---------|
    | Flat 24pt | 52.4% | 56.0% |
    | Tiered 18/28 (SP+-like) | 52.4% | 56.3% |
    | **Tiered 18/32** | **52.5%** | **56.9%** |
  - **5+ edge improvement:** +0.9% over flat penalty, +0.6% over SP+ values
  - Updated `spread_generator.py` with `ELITE_FCS_TEAMS` set and tiered penalty logic
  - Updated `backtest.py` with `--fcs-penalty-elite` and `--fcs-penalty-standard` CLI args
  - Fixed CLI default for `--asymmetric-garbage` (now enabled by default, use `--no-asymmetric-garbage` to disable)

---

## Session: February 1, 2026 (Evening)

### Completed This Evening

- **Investigated Ole Miss Ranking Discrepancy** - Ole Miss ranks #14 in JP+ but #7-9 in SP+, FPI, and Sagarin
  - **Root cause identified:** Defense. Ole Miss offense is elite (+12.2, top 5) but defense is mediocre (+2.6)
  - Average top-9 defense: +13.9. Ole Miss defense gap: **-11.3 points**
  - Defensive SR allowed: 36.8% (vs 31-34% for elite defenses like Ohio State/Oregon)
  - **Hypothetical:** If Ole Miss had average top-10 defense, they'd rank **#3** at +26.1
  - **Conclusion:** JP+ is correctly identifying a defensive weakness that hurt them in big games
    - Lost to Miami 27-31 in playoff (allowed 31 points as predicted)
    - Close wins vs Oklahoma (+8), LSU (+5), Arkansas (+6) - defense gave up a lot
  - Other systems may weight win-loss record (13-2) more heavily; JP+ ignores record entirely
  - **Game-by-game defensive breakdown:** Struggled vs Georgia (47% SR allowed), Arkansas (46%)

- **Validated Defensive Rating Convention for Game Totals** - Confirmed current convention works
  - Current JP+: Higher defense = better (points saved vs average)
  - Formula for game totals: `Total = 2×Avg + (Off_A + Off_B) - (Def_A + Def_B)`
  - Example verified: A(Off +10, Def +8) vs B(Off +5, Def +3) → Total = 56 + 15 - 11 = 60
  - Good defenses lower totals, good offenses raise them - mathematically correct
  - **No changes needed** to defensive rating convention for totals prediction

---

## Session: February 1, 2026 (Continued)

### Completed Today

- **Added Turnover Component to EFM** - Turnovers now included in ratings with Bayesian shrinkage
  - **Problem identified:** Indiana's +15 turnover margin (vs OSU's +3) was not captured in JP+
  - SP+ includes turnovers at 10% weight; JP+ was missing this entirely
  - **Solution:** Added `turnover_weight` parameter to EFM (default 0.10)
  - Calculates per-game turnover margin (forced - lost) and converts to point value
  - New weights: 54% efficiency + 36% explosiveness + 10% turnovers
  - **Bayesian shrinkage added:** `turnover_prior_strength=10` regresses margin toward 0 based on games played
    - 5 games: keeps 33% of margin (5/15)
    - 15 games: keeps 60% of margin (15/25)
    - Prevents overweighting small-sample turnover luck while trusting sustained performance
  - **Final Impact:** Indiana now #2 at +27.8, only 0.2 pts behind Ohio State (+28.0)
  - Added `turnover_rating` field to `TeamEFMRating` dataclass
  - Added `TURNOVER_PLAY_TYPES`, `POINTS_PER_TURNOVER`, and `turnover_prior_strength` constants
  - **Verified turnover play types** against CFBD API (removed dead "Interception Return" entry)
  - Updated `backtest.py` with new weights and `--efm-turnover-weight` CLI argument

- **2025 Backtest Results** (with turnover component + shrinkage)
  - MAE: 12.44 points
  - ATS: 322-310-6 (50.9%)
  - 3+ pt edge: 179-155 (53.6%)
  - 5+ pt edge: 110-92 (54.5%)

- **Reduced Red Zone Regression Strength (prior_strength 20→10)** - Analysis showed original was too aggressive
  - **Problem identified:** Elite RZ teams like Indiana (87.2% TD rate) were being penalized
  - Indiana's raw finishing drives advantage (+0.04 over OSU) was flipping to a disadvantage (-0.05) after regression
  - With 150+ RZ plays per team by end of season, high TD rates become skill, not luck
  - **Solution:** Reduced `prior_strength` from 20 to 10 in `finishing_drives.py`
  - **Impact:** Better credits teams that sustain elite RZ efficiency over full season
  - Also updated hardcoded value in `backtest.py` line 628

- **Migrated backtest.py from pandas to Polars** - Performance optimization for string filtering operations
  - **Problem identified:** cProfile showed 30s spent in pandas string comparisons (`isin()` operations)
  - **Solution:** Migrate data pipeline to Polars DataFrames (26x faster for string filtering)
  - **Changes:**
    - `fetch_season_data()` now returns `pl.DataFrame` instead of `pd.DataFrame`
    - `fetch_season_plays()` now returns `pl.DataFrame` tuples
    - `build_game_turnovers()` now uses Polars joins and group_by operations
    - `walk_forward_predict_efm()` uses Polars filtering, converts to pandas only at sklearn boundary
    - `calculate_ats_results()` accepts Polars betting DataFrame
    - Legacy Ridge model path converts to pandas before calling `walk_forward_predict()`
  - **Dependencies added:** `polars`, `pyarrow` (for efficient Polars→pandas conversion)
  - **Verified:** Full 2022-2025 backtest completes successfully with identical results

- **Implemented FBS-Only Filtering** - Ridge regression now excludes plays involving FCS opponents
  - **Problem identified:** FCS teams have too few games against FBS opponents to estimate reliable coefficients
  - FCS plays were polluting the regression with unreliable team strength estimates
  - **Solution:** Filter `train_plays` to only include FBS vs FBS matchups before passing to EFM
  - FCS games still handled via FCS Penalty adjustment (+24 pts)
  - **Backtest results (2022-2025):** 5+ pt edge improved from 54.0% → 54.8% (+0.8%)
  - Implementation in `scripts/backtest.py` line 537-545

- **Validated Asymmetric Garbage Time** - Tested symmetric vs asymmetric GT with FBS-only baseline
  - User hypothesis: Asymmetric GT was a "hack" that would hurt Indiana, should revert to symmetric
  - **Empirical finding:** Asymmetric actually HELPS Indiana (moves from #4 to #2)
  - **Backtest comparison:**
    - Asymmetric: 5+ edge 54.8% (449-371)
    - Symmetric: 5+ edge 53.0% (447-396)
  - **Decision:** Keep asymmetric GT - it improves both rankings AND predictive accuracy
  - Key insight: Indiana DOES maintain elite efficiency in garbage time; asymmetric captures this signal

- **Investigated Indiana #1 Question** - Why is Indiana #2 instead of #1 (they won the championship)?
  - Audited play-by-play efficiency consistency for Indiana vs Oregon vs Ohio State
  - Audited performance against elite defenses (Top 30)
  - **Findings:**
    - Indiana raw SR: 53.8% (highest)
    - Indiana vs elite defenses: 42.7% SR (better than Oregon's 41.2%)
    - Indiana beat Oregon head-to-head: 47.1% vs 36.1% SR
    - Yet Indiana is still #2 behind Ohio State after opponent adjustment
  - **Conclusion:** The gap is NOT from FCS contamination or asymmetric GT
  - Ohio State-Indiana gap: 3.4 points (same in both symmetric and asymmetric)
  - Ohio State maintains #1 due to stronger performance vs elite opponents AND stronger schedule overall

---

## Session: February 1, 2026 (Earlier)

### Completed Earlier Today

- **Implemented Transfer Portal Integration** - Preseason priors now incorporate net transfer portal impact
  - Fetches transfer portal entries from CFBD API
  - Matches transfers to prior-year player usage (PPA) to quantify production
  - Calculates net incoming - outgoing PPA for each team
  - Adjusts effective returning production by portal impact (scaled, capped at ±15%)
  - **Backtest results (2022-2025):** 5+ pt edge improved from 53.3% → 53.7% (+0.4%)
  - Added `fetch_transfer_portal()`, `fetch_player_usage()`, `calculate_portal_impact()` to `PreseasonPriors`
  - Added `portal_adjustment` field to `PreseasonRating` dataclass
  - New CLI flags: `--no-portal`, `--portal-scale` (default 0.15)
  - Top 2024 portal winners: Missouri (+15%), Washington (+15%), UCF (+15%), California (+15%)
  - Top 2024 portal losers: New Mexico State, Ohio, Arizona State

- **Updated efficiency/explosiveness weights documentation** - All docs now reflect 60/40 weighting

- **Implemented Rating Normalization** - Ratings now scaled for direct spread calculation
  - Added `rating_std` parameter to EFM (default 12.0)
  - Added `_normalize_ratings()` method to scale ratings to target std
  - Ratings normalized: mean=0, std=12 across FBS teams
  - **Direct spread calculation:** Team A rating - Team B rating = expected spread
  - Example: Ohio State (+33.1) vs Penn State (+19.1) → Ohio State -14.0
  - Updated MODEL_EXPLAINER.md with normalized Top 10 ratings

- **Implemented Asymmetric Garbage Time** - Only trailing team's garbage time plays down-weighted
  - **Problem identified:** Dominant teams (Indiana 56% SR, Ohio State 57% SR in garbage time) were having their best plays discarded
  - **Solution:** Winning team keeps full weight; only trailing team gets 0.1 weight
  - Added `asymmetric_garbage` parameter to EFM (default True)
  - Added `--asymmetric-garbage` CLI flag to backtest.py
  - **Backtest results (2022-2025):**
    - MAE: 12.53 → 12.52 (-0.01)
    - 5+ pt edge: 53.4% → 54.0% (+0.6%)
  - **Ranking impact:** Indiana rises from #4 to #3; Notre Dame drops from #3 to #4
  - Conceptual improvement: rewards teams that maintain dominance, penalizes coasting

- **Tested 55/45 Efficiency/Explosiveness Weights** - Decided to keep 60/40
  - Hypothesis: Increasing explosiveness weight might help explosive teams like Ole Miss
  - **Backtest results (55/45 + asymmetric):** 5+ edge 54.2% (+0.2% vs 60/40)
  - **Ranking impact:** Ole Miss unchanged (+21.6); Indiana dropped (-0.5); Oregon/Notre Dame rose
  - **Decision:** Keep 60/40 - marginal improvement not worth hurting Indiana (National Champs)
  - Ole Miss #16 ranking appears accurate based on raw efficiency (48.9% SR, #22 nationally)

---

## Session: January 31, 2026

### Completed Previously

- **Implemented FG Efficiency Adjustment** - Integrated kicker PAAE (Points Above Average Expected) into spread predictions
  - Calculates FG make rates vs expected by distance (92% for <30yd, 83% for 30-40, etc.)
  - PAAE = actual points - expected points for each kick
  - Per-game FG rating applied as differential adjustment to spreads
  - **Impact:** ATS improved from 50.6% → 51.2% (+0.6%), MAE improved from 12.57 → 12.47
  - Added `calculate_fg_ratings_from_plays()` to `SpecialTeamsModel`
  - Added FG plays collection to backtest data pipeline

- **Cleaned up documentation** - Removed all references to unused/legacy components
  - Removed Ridge Model (margin-based) section from MODEL_ARCHITECTURE.md
  - Removed luck regression, early-down model, turnover scrubbing references
  - Removed legacy CLI flags (--decay, --to-scrub-factor, --margin-cap)
  - Updated file structure to only show actively used files
  - Clarified EFM is the sole foundation model, not one of two options

- **Exposed separate O/D/ST ratings** - JP+ now provides separate offensive, defensive, and special teams ratings
  - Updated `TeamEFMRating` dataclass with `offensive_rating`, `defensive_rating`, `special_teams_rating` fields
  - Added methods: `get_offensive_rating()`, `get_defensive_rating()`, `get_special_teams_rating()`, `set_special_teams_rating()`
  - Updated `get_ratings_df()` to include offense, defense, special_teams columns
  - Integrated special teams rating from SpecialTeamsModel into EFM in backtest pipeline
  - Example 2025 insights: Vanderbilt has best offense (+16.7), Oklahoma has best defense (+13.8)
  - **Purpose:** Enables future game totals prediction (over/under) by predicting each team's expected points

- **Tuned efficiency/explosiveness weights from 65/35 to 60/40**
  - Compared JP+ rankings to SP+ to identify systematic issues
  - Found explosive teams (Texas Tech, Ole Miss) were being underrated
  - Tested weight configurations: 65/35, 60/40, 55/45, 50/50
  - 60/40 weighting showed best multi-year backtest results (2022-2025):
    - MAE: 12.63 (vs 12.65 with 65/35)
    - ATS: 51.3% (vs 51.0% with 65/35)
    - 5+ pt edge: 54.5% (vs 54.2%)
  - Updated defaults in EFM and backtest.py

- **Implemented FCS Penalty Adjustment** - Adds 24 points to FBS team's predicted margin vs FCS opponents
  - **Diagnostic finding:** JP+ was UNDER-predicting blowouts by 26 pts, not over-predicting
  - 99.5% of blowout errors were under-predictions (actual margins larger than predicted)
  - FCS games (3.3% of total) had MAE of 29.54 - dragging overall MAE significantly
  - Tested penalty sweep from 0-30 pts; optimal value ~24 pts based on actual under-prediction
  - **Impact:** MAE improved from 13.11 → 12.57 (-0.54), 5+ edge ATS improved 52.8% → 53.2%
  - Added `fcs_penalty` parameter to `SpreadGenerator` and `backtest.py`
  - Added `fcs_adjustment` component to spread breakdown

- **Ran MAE by Margin Diagnostic Analysis**
  - Discovered blowouts (29+ pts, 17% of games) contribute 33% of overall MAE
  - Identified root causes: FCS teams treated as merely "below average" instead of dramatically weaker
  - Elite teams (Ohio State, Notre Dame, Indiana) consistently beating weak opponents by more than predicted

- **Implemented Coaching Change Regression** - New HC at underperforming team triggers weight shift from prior performance toward talent
  - "Forget factor" based on talent-performance gap (bigger gap = more forgetting, capped at 50%)
  - Data-driven coach pedigree scores calculated from CFBD API (career win %, P5 years)
  - Elite coaches (Kiffin 1.30, Cignetti 1.25, Sumrall 1.25) get more benefit of the doubt
  - **First-time HCs are EXCLUDED** - no adjustment (we have no basis to predict improvement)
  - Impact: +1 to +5 pts for underperformers with proven coaches
  - **Backtest result:** Neutral impact on MAE/ATS (sample too small), but kept for conceptual soundness

- **Updated all documentation** with FCS penalty and coaching change regression details

---

## Session: January 29, 2026

### Completed Previously

- **Implemented HFA trajectory modifier** - Dynamic adjustment (±0.5 pts) for rising/declining programs based on win % improvement
  - Compares recent year (1 yr) to baseline (prior 3 yrs)
  - Added `calculate_trajectory_modifiers()` and `set_trajectory_modifier()` to `HomeFieldAdvantage` class
  - Constants: `TRAJECTORY_MAX_MODIFIER = 0.5`, `TRAJECTORY_BASELINE_YEARS = 3`, `TRAJECTORY_RECENT_YEARS = 1`

- **Updated all documentation** with trajectory modifier details
  - MODEL_ARCHITECTURE.md: Added Trajectory Modifier subsection with calculation table and examples
  - MODEL_EXPLAINER.md: Added user-friendly explanation of trajectory concept
  - Changelog updated

- **Validated trajectory calculations** for Vanderbilt and Indiana (2023-2025)
  - Vanderbilt: 2.12 → 2.48 → 2.50 (rising program)
  - Indiana: 2.46 → 3.25 → 3.25 (penalty in 2023, max boost 2024-25)

---

### Current Source of Truth

#### EFM Parameters
| Parameter | Value | Location |
|-----------|-------|----------|
| `ridge_alpha` | 50 | `efficiency_foundation_model.py` (optimized from 100) |
| `efficiency_weight` | 0.54 | `efficiency_foundation_model.py` |
| `explosiveness_weight` | 0.36 | `efficiency_foundation_model.py` |
| `turnover_weight` | 0.10 | `efficiency_foundation_model.py` |
| `turnover_prior_strength` | 10.0 | `efficiency_foundation_model.py` |
| `garbage_time_weight` | 0.1 | `efficiency_foundation_model.py` |
| `asymmetric_garbage` | True | `efficiency_foundation_model.py` |
| `rating_std` | 12.0 | `efficiency_foundation_model.py` |
| `rz_prior_strength` | 10 | `finishing_drives.py` |
| `fcs_penalty` | 24.0 | `spread_generator.py` |

#### HFA Parameters
| Parameter | Value | Location |
|-----------|-------|----------|
| Base HFA range | 1.5 - 4.0 | `home_field.py` → `TEAM_HFA_VALUES` |
| Conference defaults | SEC/B1G: 2.75, Big12/ACC: 2.50, G5: 2.0-2.25 | `home_field.py` → `CONFERENCE_HFA_DEFAULTS` |
| Trajectory max modifier | ±0.5 | `home_field.py` → `TRAJECTORY_MAX_MODIFIER` |
| Trajectory baseline years | 3 | `home_field.py` → `TRAJECTORY_BASELINE_YEARS` |
| Trajectory recent years | 1 | `home_field.py` → `TRAJECTORY_RECENT_YEARS` |

#### Preseason Priors
| Parameter | Value | Location |
|-----------|-------|----------|
| Prior year weight | 0.6 (default) | `preseason_priors.py` |
| Talent weight | 0.4 (default) | `preseason_priors.py` |
| Regression range | 0.1 - 0.5 (based on returning PPA) | `preseason_priors.py` → `_get_regression_factor()` |
| Coaching forget factor cap | 0.5 | `preseason_priors.py` → `_calculate_coaching_change_weights()` |
| Coaching gap divisor | 60 | `preseason_priors.py` (gap/60 = base forget) |
| Portal scale | 0.15 | `preseason_priors.py` → `calculate_portal_impact()` |
| Portal adjustment cap | ±15% | `preseason_priors.py` → `calculate_portal_impact()` |

#### Coach Pedigree (Data-Driven)
| Tier | Pedigree Range | Example Coaches |
|------|----------------|-----------------|
| Elite | 1.27 - 1.30 | Kiffin, Kirby Smart, DeBoer, James Franklin |
| Strong | 1.20 - 1.25 | Cignetti, Sumrall, Silverfield, Chadwell |
| Above Avg | 1.10 - 1.19 | Rhule, McGuire, Venables |
| Average | 1.05 - 1.09 | Brent Key, Sam Pittman, Deion Sanders |
| Neutral | 1.00 | First-time HCs, no HC record |
| Below Avg | 0.88 - 0.97 | Clark Lea, Troy Taylor, Jeff Lebby |

#### Data Pipeline
| Parameter | Value | Location |
|-----------|-------|----------|
| FBS-only filtering | True | `backtest.py` → `walk_forward_predict_efm()` |

#### Model Performance (2022-2025) - With All Features (FBS-only + Asymmetric GT)
- **MAE:** 12.54
- **ATS:** 51.3% (1255-1190-40)
- **3+ pt edge ATS:** 52.8% (724-646)
- **5+ pt edge ATS:** 54.8% (449-371)

---

### The Parking Lot (Future Work)

**Where we stopped:** Optimized ridge alpha from 100 to 50 (+0.7% ATS at 5+ edge). Tested EPA regression vs SR+IsoPPP (SR+IsoPPP wins). Game totals formula validated and ready for implementation.

**Open tasks to consider:**
1. ~~**EFM alpha parameter sweep**~~ - ✅ DONE. Optimal alpha=50 (was 100). See Feb 2 session.
2. **Penalty Adjustment** - Explore adding penalty yards as an adjustment factor
   - Hypothesis: Disciplined teams (fewer penalties) have a real edge JP+ ignores
   - Approach: Calculate penalty yards/game vs FBS average, convert to point impact
3. **Game totals prediction** - Formula validated and ready to implement:
   - `Total = 2×Avg + (Off_A + Off_B) - (Def_A + Def_B)`
   - `Team_A_points = Avg + Off_A - Def_B`
   - Current defensive convention (higher = better) works correctly
4. **Further blowout improvement** - FCS penalty helped, but blowout MAE still high. Could explore:
   - Lower ridge alpha to reduce shrinkage (let elite teams rate higher)
   - Weak FBS team penalty (G5 bottom-feeders similar to FCS)
   - Non-linear rating transformation
5. **Weather impact modeling** - Rain/wind effects on passing efficiency

---

### Unresolved Questions

1. ~~**Trajectory modifier timing**~~ - **RESOLVED:** Calculate trajectory ONCE at start of season using prior year as "recent". Lock it in for the whole season. Rationale: HFA reflects stadium atmosphere built over years, not weeks. Avoids double-counting current performance (already in ratings).

2. ~~**Trajectory for new coaches**~~ - **RESOLVED:** Implemented coaching change regression in preseason priors. Proven coaches (Cignetti, Kiffin, Sumrall) get preseason rating boost when taking over underperforming teams. First-time HCs excluded.

3. ~~**Trajectory decay**~~ - **RESOLVED:** No explicit decay needed. The formula naturally handles this—as successful years roll into the baseline (prior 3 yrs), the improvement gap shrinks automatically. Example: Indiana's 2024 success becomes part of baseline by 2027, reducing their modifier.

4. ~~**G5 elite programs**~~ - **RESOLVED:** No special G5 treatment needed. Base HFA already handles elite G5s (Boise 3.0, JMU 2.75). Trajectory measures crowd energy change, not "impressiveness" of wins. Same formula for all conferences.

5. **"Mistake-free" football (penalties)** - 2025 Indiana was extremely disciplined. Does Success Rate already capture ~80% of this value, or is penalty differential a distinct predictive signal? Need to analyze correlation between penalty rates and ATS performance independent of efficiency metrics.

6. ~~**Ole Miss #14 vs #7-9 in other systems**~~ - **RESOLVED:** Investigated why Ole Miss ranks lower in JP+ than SP+/FPI/Sagarin:
   - **Root cause:** Defense (+2.6) is far below top-team average (+13.9)
   - Offense is elite (+12.2), but defense allowed 36.8% SR (vs 31-34% for elite defenses)
   - If Ole Miss had average top-10 defense, they'd rank #3 at +26.1
   - Lost to Miami 27-31 in playoff - exactly what weak defense rating predicted
   - **Conclusion:** JP+ correctly identifies defensive weakness; other systems may weight 13-2 record more heavily

7. **Indiana #2 vs Ohio State #1** - Indiana won the championship, beat everyone including Ohio State, yet JP+ has them #2. Investigation found:
   - FBS-only filtering didn't change this (Indiana still #2)
   - Asymmetric GT actually helps Indiana (#4 → #2)
   - The 3.4 pt gap is consistent across all filter configurations
   - Ohio State's stronger schedule overall appears to be the driver
   - **Open question:** Is this correct (OSU was genuinely better pre-championship) or is the model still over-penalizing Indiana's weaker regular season schedule?

---

*End of session*
