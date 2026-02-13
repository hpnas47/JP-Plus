# 🏈 JP+ CFB Power Ratings: Model Governance

## Project Overview
- This project is a Python-based CFB power ratings model designed to generate an analytical edge against market lines. Key files include the backtest pipeline, efficiency model, ratings engine, and config. The primary language is Python. Documentation is maintained in Markdown. Always check that column names, feature names, and config flags match across files after any refactor.

## Model Development Rules
- **Backtest Before Commit:** After any code change to the model pipeline, always run a backtest before committing to verify MAE, correlation, and ATS metrics haven't regressed. Never commit model changes without backtest validation.
- **Multi-Year Validation:** When adding a new feature, always validate across multiple years (not just one season) before keeping it. If single-year results look good but multi-year degrades MAE, remove the feature.
- **Edge > Accuracy:** Optimize parameters to maximize ATS Profit (Win % on 3+/5+ pt edge), not to minimize MAE. ATS performance is what matters for betting edge.
- **Market Blindness:** Never use the Vegas line as a training target or feature input. Disagreement with Vegas is the goal, not an error.
- **Process Over Resume:** Rankings must be derived from Efficiency (Success Rate), not Outcomes (Points/Wins). Exception: Turnover Margin (10%) as regressed modifier.
- **No Mercy Rule:** Never dampen predicted margins solely to lower MAE in blowouts. We model team capability, not coaching psychology.
- **Regular Season for Metrics:** Backtest metrics use regular season only (weeks 1-15). Postseason excluded due to unmodeled factors: coaching changes, opt-outs, transfer portal, motivation variance.
- **Full Season for Final Ratings:** End-of-season power ratings include all games through National Championship to capture each team's full body of work.

## Sign Conventions (Immutable)
- **Internal (SpreadGenerator):** Positive (+) = Home Team Favored
- **Vegas (CFBD API):** Negative (-) = Home Team Favored
- **HFA:** Positive (+) = Points added to Home Team
- **Edge:** Negative = JP+ likes Home more than Vegas; Positive = JP+ likes Away more
- **Actual Margin:** Positive (+) = Home Team Won
- **Conversion:** `vegas_spread = -internal_spread`

## Data Sources (Betting Lines)
- **Historical (2022-2025):** CFBD API — 91% FBS opening line coverage
- **Future (2026+):** The Odds API — capture opening (Sunday) and closing (Saturday) lines
- **Priority order:** DraftKings > ESPN Bet > Bovada > fallback
- **Storage:** Odds API lines stored in `data/odds_api_lines.db` (SQLite)
- **Merge logic:** `src/api/betting_lines.py` combines both sources, preferring Odds API when available

## Data Hygiene
- **FCS:** Efficiency metrics must be trained on FBS vs. FBS data only. FCS plays dropped before Ridge Regression.
- **Garbage Time:** Asymmetric filtering. Winner keeps full weight (signal); Loser gets down-weighted (noise).

## Git Workflow
- **Atomic Commits:** Never commit unrelated changes alongside a targeted fix. Keep commits atomic and scoped to the task at hand.
- **Staging Protocol:** Always confirm with the user before staging files beyond the current task.
- **Documentation Sync:** All documentation changes must be pushed to BOTH repositories:
  1. `hpnas47/JP-Plus` (main code repo) — `docs/` directory
  2. `hpnas47/JP-Plus-Docs` (docs-only repo) — root directory
- **Session Log:** Update `docs/SESSION_LOG.md` at end of every session with ALL changes made AND tested-but-rejected experiments.

## 🛠️ Environment & Technical Context
- **Python:** `python3` (v3.10+)
- **Database:** SQLite located at `data/cfb_model.db`
- **Execution:** All scripts MUST be run from the project root.
- **Dependencies:** `Code Auditor` is authorized to `pip install` missing packages to resolve environment drift.

## 📁 Key File Map (Source of Truth)
- **Priors Engine:** `src/models/preseason_priors.py` (Talent, Portal, Recruiting Offsets)
- **Core EFM Logic:** `src/models/efficiency_foundation_model.py` (Ridge Regression, HFA, SOS)
- **Sub-Models:** `src/models/special_teams.py` (FG + punt + kickoff)
- **QB Continuous:** `src/adjustments/qb_continuous.py` (Walk-forward QB quality estimates)
- **Backtest Engine:** `scripts/backtest.py` (The Validator)
- **Market Data:** `scripts/weekly_odds_capture.py` (OddsAPI/Market Snapshots)
- **Full File Map:** `docs/PROJECT_MAP.md`

## ✅ Current Production Baseline (2022-2025 backtest, as of 2026-02-12)

**With QB Continuous Phase1-only + Phase 1 Shrinkage enabled**

| Slice | Weeks | Games | MAE | RMSE | ATS (Close) | ATS (Open) |
|-------|-------|-------|-----|------|-------------|------------|
| Phase 1 (Calibration) | 1–3 | 960 | 14.01 | 17.51 | 46.9% | 47.1% |
| **Phase 2 (Core)** | **4–15** | **2,485** | **12.51** | **15.82** | **51.7%** | **53.0%** |
| **Regular Season** | **1–15** | **3,445** | **12.90** | **16.27** | **50.3%** | **51.5%** |
| 3+ Edge (Regular) | 1–15 | 2,009 | — | — | 51.5% (1034-975) | 53.5% (1095-952) |
| 5+ Edge (Regular) | 1–15 | 1,305 | — | — | 53.6% (700-605) | 54.9% (743-610) |
| 3+ Edge (Core) | 4–15 | 1,389 | — | — | 53.1% (737-652) | 55.4% (791-637) |
| 5+ Edge (Core) | 4–15 | 841 | — | — | 55.1% (463-378) | 57.0% (509-384) |

**Phase 1 Improvement with Shrinkage=0.90:**
| Metric | Without Shrinkage | With Shrinkage | Delta |
|--------|-------------------|----------------|-------|
| 5+ Edge (Close) | 50.4% (225-221) | **51.1%** (237-227) | **+0.7%** |
| 5+ Edge (Open) | 50.4% (225-221) | **50.9%** (234-226) | **+0.5%** |
| RMSE | 17.57 | 17.45 | -0.12 |

Core (weeks 4-15) performance is **unchanged** with Phase 1 shrinkage.

### Edge-Aware Production Mode (DEFAULT in 2026)

The prediction engine automatically selects Fixed or LSA based on timing and edge magnitude:

| Timing | Edge | Mode | ATS | Rationale |
|--------|------|------|-----|-----------|
| **Opening** (4+ days out) | Any | Fixed | **56.5%** (5+) | Fixed dominates on less-efficient opening lines |
| **Closing** (<4 days) | 3-5 pts | Fixed | **52.9%** | LSA degrades 3+ edge by 0.9% |
| **Closing** (<4 days) | 5+ pts | LSA | **55.1%** | LSA improves 5+ edge by 1.1% |

**Full LSA vs Fixed Comparison (Core Weeks 4-15):**

| Mode | 3+ Edge (Close) | 3+ Edge (Open) | 5+ Edge (Close) | 5+ Edge (Open) |
|------|-----------------|----------------|-----------------|----------------|
| Fixed | **52.9%** (795-707) | **55.1%** (844-688) | 54.0% (498-425) | **56.5%** (551-425) |
| LSA | 52.0% (768-708) | 54.9% (825-678) | **55.1%** (480-391) | 55.9% (523-413) |

**LSA Config:** `alpha=300.0`, `clamp_max=4.0`, `min_games=150`, `ema=0.3`, `adjust_for_turnovers=True`

### Model Configuration

- **Audit:** 41/48 items fixed (P0-P3). 7 deferred. Fixlists archived in `docs/Completed Audit Fixlists/`.
- **EFM Weights:** SR=45%, IsoPPP=45%, Turnovers=10% (Explosiveness Uplift from 54/36/10).
- **HFA Global Offset:** -0.50 pts applied to all HFA values (calibrated Feb 2026). Reduces systematic home bias from +0.90 to +0.46.
- **RZ Leverage:** Play-level weighting in EFM (2.0x inside 10, 1.5x inside 20). Replaces shelved Finishing Drives model (4 rejections, 70-80% overlap with IsoPPP).
- **Conference Anchor:** OOC game weighting (1.5x) + separate O/D Bayesian conference anchors (scale=0.08, prior=30, max=2.0). Fixes inter-conference bias; Big 12 intra-conference circularity remains.
- **ST Spread Cap:** ±2.5 pts (APPROVED 2026-02-10). Caps ST differential's effect on spread without shrinking ratings toward zero.
- **FCS Strength Estimator:** Dynamic, walk-forward-safe FCS penalties (APPROVED 2026-02-10). Replaces static elite list with Bayesian shrinkage (k=8, baseline=-28, intercept=10, slope=0.8). Penalty range [10, 45] pts. CLI: `--fcs-static` for baseline comparison.
- **LSA Edge-Aware Mode:** DEFAULT for 2026 production. Automatically uses LSA for 5+ edge closing bets, Fixed otherwise. No flags needed — `run_weekly.py` handles timing/edge logic automatically. CLI: `--no-lsa` to force Fixed-only mode.
- **QB Continuous Rating:** DEFAULT for 2026. Walk-forward-safe QB quality estimates using PPA data with shrinkage (K=200) and prior season decay (0.3). **Phase1-only mode** applies QB adjustment only for weeks 1-3 where EFM hasn't yet captured QB quality. CLI: `--no-qb-continuous` to disable.
- **Phase 1 Spread Shrinkage:** DEFAULT for 2026 (APPROVED 2026-02-12). Applies 0.90x shrinkage to rating differential for weeks 1-3, preserving HFA. Formula: `NewSpread = (OldSpread - HFA) * 0.90 + HFA`. Reduces overconfidence in prior-driven predictions. Improves Phase 1 5+ Edge by +0.7% (Close) and +0.5% (Open). CLI: `--no-phase1-shrinkage` to disable, `--phase1-shrinkage 0.85` to adjust.

## ✅ Totals Model Baseline (2023-2025 backtest, as of 2026-02-08)

| Slice | Weeks | Games | MAE | ATS (Close) | ATS (Open) |
|-------|-------|-------|-----|-------------|------------|
| Phase 1 (Calibration) | 1–3 | 169 | 12.42 | 57.9% | 55.4% |
| **Phase 2 (Core)** | **4–15** | **1,824** | **13.09** | **53.9%** | **53.4%** |
| **Regular Season** | **1–15** | **1,993** | **13.03** | **54.3%** | **53.6%** |
| 3+ Edge (Core) | 4–15 | 985 | — | 54.7% (539-446) | 54.2% (528-447) |
| 5+ Edge (Core) | 4–15 | 613 | — | 54.5% (334-279) | 55.3% (330-267) |

- **Years:** 2023-2025 only (2022 dropped — scoring environment transition year, 49% ATS).
- **Ridge Alpha:** 10.0 (optimal for 5+ Edge).
- **Architecture:** Separate from EFM. Ridge regression on game-level points scored/allowed (not play-level efficiency).
- **Learned HFA:** Home field advantage learned via Ridge column (+3.5 to +4.5 pts typical). Improved 5+ Edge by +1.6% vs fixed/no HFA.
- **Decay Factor:** 1.0 (no within-season decay — walk-forward handles temporality).
- **OT Protection:** Disabled (final scores used — Vegas prices OT potential).
- **Weather:** Available but optional (no ATS improvement — market already prices weather).

---

## 📊 JP+ Power Ratings Display Protocol
1. **Components:** Show O/D/ST (Offense/Defense/Special Teams) component ratings.
2. **Rankings:** Include component rankings in parentheses (e.g., "16.3 (1)"). 1 = Best.
3. **Table Columns:** `Rank | Team | Overall | Offense (rank) | Defense (rank) | Special Teams (rank)`

## 📈 Backtest Reporting Protocol
- **Range:** 2022–2025.
- **Markets:** Performance vs. BOTH Opening Line and Closing Line.
- **Thresholds:** Report results for 3+ point edge and 5+ point edge categories.

---

# 🏛️ The Audit Council: Model Governance

## 🧠 Model Strategist (The Architect)
- **Role:** High-level logic validation and feature-trait differentiation.
- **Primary Objective:** Protect the model from "Redundancy Rot" and "Overfitting to Noise."
- **The Signal Test:** Before coding begins, must determine if a proposed feature is a **Persistent Trait** (coaching/talent) or a **High-Variance Event** (luck/turnovers).
- **Redundancy Filter:** Evaluate if new features (e.g., Finishing Drives) are already "priced in" via existing PPA/Success Rate metrics. If overlap > 60%, mandate **EFM Integration** or **Residualization**.
- **Temporal Integrity:** Ensure that "Priors" (recruiting/portal) are correctly decayed as "Live Data" (on-field performance) takes over mid-season.
- **Strategic Guardrail:** Prevent the model from becoming a "Spread Follower." If the model moves toward market consensus without improving MAE, flag as a loss of predictive edge.

---

## 🛡️ Code Auditor (The Safety)
- **Status:** Maintenance Mode (P0-P3 sweeps complete).
- **Role:** Enforce data integrity, walk-forward chronology, and **feature architectural purity**.
- **Constraints:**
    - Strictly enforce `training_max_week < prediction_week` (Zero Data Leakage).
    - All sub-model outputs must be rendered as **PBTA Points Per Game**.
    - **Anti-Drift Guardrail:** Prevent "Additive Drift." Ensure new features are evaluated for integration into the **EFM Ridge Regression** before being added as post-hoc constants.
- **On New Code:** Run `/audit-logic` to check for regressions and **double-counting** before merging.

---

## 📊 Quant Auditor (The Analyst)
- **Role:** Weight optimization, MAE/ATS validation, and **redundancy detection**.
- **Success Metrics (Core Phase, Weeks 4–15, 2,485 games):**
    - **MAE Baseline:** 12.51 (Strict Tolerance: +0.02).
    - **ATS Target (Core):** > 52.0%. **5+ Edge Target:** > 54.0%.
- **Redundancy Protocol:** For any new signal, you must report the **Correlation Coefficient** against existing PPA/IsoPPP metrics.
- **Validation Slices (Mandatory):**
    - **EFM/In-Season:** `python backtest.py --start-week 4`. Focus on Weeks 4-15 to isolate in-season signal from preseason noise.
    - **Priors/Portal/Talent:** `python backtest.py --start-week 1`. Full Season validation for Recruiting Offset and Portal Continuity Tax.
- **Sanity Check:** Must report rating stability for **High Variance Cohorts** (High Churn/Portal teams) alongside Blue Bloods (ALA, UGA, OSU, TEX, ORE, ND).

<!-- Last validated: 2026-02-12 by generate_docs.py -->
