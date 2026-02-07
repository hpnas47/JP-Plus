# 🏈 JP+ CFB Power Ratings: Model Governance

## Project Overview
- This project is a Python-based CFB power ratings model designed to generate an analytical edge against market lines. Key files include the backtest pipeline, efficiency model, ratings engine, and config. The primary language is Python. Documentation is maintained in Markdown. Always check that column names, feature names, and config flags match across files after any refactor.

## Model Development Rules
- Backtest Before Commit: After any code change to the model pipeline, always run a backtest before committing to verify MAE, correlation, and ATS metrics haven't regressed. Never commit model changes without backtest validation.

- Sign Convention Verification: When fixing spread/line calculations, always verify the sign convention (home-team perspective vs. away-team perspective) before and after the change. Double-check that normalization doesn't flip already-correct values.

- Multi-Year Validation: When adding a new feature to the model (e.g., TWP, early-down success rate, red zone regression), always validate across multiple years (not just one season) before keeping it. If single-year results look good but multi-year degrades MAE, remove the feature.

## Git Workflow
- Atomic Commits: Never commit unrelated changes alongside a targeted fix. Keep commits atomic and scoped to the task at hand.

- Staging Protocol: Always confirm with the user before staging files beyond the current task.

## 🛠️ Environment & Technical Context
- **Python:** `python3` (v3.10+)
- **Database:** SQLite located at `data/cfb_model.db`
- **Execution:** All scripts MUST be run from the project root.
- **Dependencies:** `Code Auditor` is authorized to `pip install` missing packages to resolve environment drift.

## 📁 Key File Map (Source of Truth)
- **Priors Engine:** `src/models/preseason_priors.py` (Talent, Portal, Recruiting Offsets)
- **Core EFM Logic:** `src/models/efficiency_foundation_model.py` (Ridge Regression, HFA, SOS)
- **Sub-Models:** `src/models/finishing_drives.py`, `src/models/special_teams.py`
- **Backtest Engine:** `scripts/backtest.py` (The Validator)
- **Market Data:** `scripts/weekly_odds_capture.py` (OddsAPI/Market Snapshots)
- **Full File Map:** `docs/PROJECT_MAP.md`

## ✅ Current Production Baseline (2022-2025 backtest, as of 2026-02-07)

| Slice | Weeks | Games | MAE | RMSE | ATS (Close) | ATS (Open) |
|-------|-------|-------|-----|------|-------------|------------|
| **Full (`--start-week 1`)** | 1–Post | 3,273 | 13.02 | 16.50 | 51.1% | 52.7% |
| Phase 1 (Calibration) | 1–3 | 597 | 14.94 | — | 47.1% | 48.6% |
| **Phase 2 (Core)** | **4–15** | **2,485** | **12.53** | **15.91** | **52.5%** | **54.0%** |
| Phase 3 (Postseason) | 16+ | 176 | 13.49 | — | 47.4% | 48.3% |
| 3+ Edge (Core) | 4–15 | 1,412 | — | — | 53.7% (758-654) | 55.5% (811-650) |
| 5+ Edge (Core) | 4–15 | 875 | — | — | 54.2% (474-401) | 56.9% (525-397) |

- **Audit:** 41/48 items fixed (P0-P3). 7 deferred. Fixlists archived in `docs/Completed Audit Fixlists/`.
- **EFM Weights:** SR=45%, IsoPPP=45%, Turnovers=10% (Explosiveness Uplift from 54/36/10).
- **Finishing Drives:** Shelved as post-hoc component (4 rejections). RZ efficiency integrated as EFM Ridge feature (2.2% of variance).
- **Conference Anchor:** OOC game weighting (1.5x) + separate O/D Bayesian conference anchors (scale=0.12, prior=20, max=3.0). Fixes inter-conference defensive inflation; Big 12 intra-conference circularity partially addressed.

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
    - **MAE Baseline:** 12.52 (Strict Tolerance: +0.02).
    - **ATS Target (Core):** > 52.0%. **5+ Edge Target:** > 53.5%.
- **Redundancy Protocol:** For any new signal, you must report the **Correlation Coefficient** against existing PPA/IsoPPP metrics.
- **Validation Slices (Mandatory):**
    - **EFM/In-Season:** `python backtest.py --start-week 4`. Focus on Weeks 4-15 to isolate in-season signal from preseason noise.
    - **Priors/Portal/Talent:** `python backtest.py --start-week 1`. Full Season validation for Recruiting Offset and Portal Continuity Tax.
- **Sanity Check:** Must report rating stability for **High Variance Cohorts** (High Churn/Portal teams) alongside Blue Bloods (ALA, UGA, OSU, TEX, ORE, ND).

