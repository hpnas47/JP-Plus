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

# 🤝 Agent Collaboration Protocol

## 🛡️ Code Auditor (The Safety)
- **Status:** Audit sweep complete (P0-P3). Now in maintenance mode.
- **Role:** Enforce data integrity, walk-forward chronology, and code quality for new changes.
- **Constraints:** Strictly enforce `training_max_week < prediction_week`. All sub-model outputs must be **PBTA Points Per Game**.
- **On new code:** Run `/audit-logic` to check for regressions before merging.

## 📊 Quant Auditor (The Analyst)
- **Role:** Weight optimization and MAE/ATS performance validation.
- **Success Metric:** MAE must stay stable (Tolerance: +0.05). ATS baseline: > 52.3%.
- **Validation Slices (Mandatory):**
    - **EFM/In-Season Tuning:** Run `python backtest.py --start-week 4`. Focus on Weeks 4-15 to isolate in-season signal from preseason noise.
    - **Priors/Portal/Talent Tuning:** Run `python backtest.py --start-week 1`. Use the Full Season to validate the "Blue Blood" Recruiting Offset and Portal Continuity Tax.
- **Sanity Check:** Must specifically report rating stability for ALA, UGA, OSU, TEX, ORE, and ND.

---

## ⌨️ Custom Workflow Shortcuts
- `/audit-logic`: Invoke `🛡️ Code Auditor` to scan for data leakage or P0 bugs.
- `/audit-math`: Invoke `📊 Quant Auditor` to run the 3rd-year backtest sweep.
- `/show-ratings`: Generate the JP+ ratings table for the current week.