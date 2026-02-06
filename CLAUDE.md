# 🏈 JP+ CFB Power Ratings: Model Governance

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

## ✅ Audit Status (Complete as of 2026-02-05)
- **Result:** 41/48 items fixed across P0-P3 + diagnostics. 7 deferred with documented blockers.
- **Baseline:** Core MAE 12.49 | ATS 51.87% | 3+ Edge 53.1% | 5+ Edge 54.7%
- **Archived:** All audit fixlists in `docs/Completed Audit Fixlists/`
- **Deferred items** require external API research (CFBD structured fields, OddsAPI schema) or model recalibration (Finishing Drives scaling).

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
- **Success Metrics:**
    - **MAE Baseline:** 12.52 (Strict Tolerance: +0.02).
    - **ATS Target:** > 52.3%.
- **Redundancy Protocol:** For any new signal, you must report the **Correlation Coefficient** against existing PPA/IsoPPP metrics.
- **Validation Slices (Mandatory):**
    - **EFM/In-Season:** `python backtest.py --start-week 4`. Focus on Weeks 4-15 to isolate in-season signal from preseason noise.
    - **Priors/Portal/Talent:** `python backtest.py --start-week 1`. Full Season validation for Recruiting Offset and Portal Continuity Tax.
- **Sanity Check:** Must report rating stability for **High Variance Cohorts** (High Churn/Portal teams) alongside Blue Bloods (ALA, UGA, OSU, TEX, ORE, ND).