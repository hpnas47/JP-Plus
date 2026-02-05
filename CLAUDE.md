# Claude Code Memory - CFB Power Ratings Model

## Power Ratings Display Preferences

When asked to produce JP+ power ratings:
1. Show O/D/ST (Offense/Defense/Special Teams) component ratings
2. Include each component's ranking in parentheses (e.g., if Indiana has the #1 offense, display as "16.3 (1)")
3. Rankings should show 1 = best for all components
4. Format columns: Rank, Team, Overall, Offense (rank), Defense (rank), Special Teams (rank)

## Backtest Display Preferences

When presenting backtest results:
1. Always include 2025 in the year range (2022-2025)
2. Always show BOTH vs Opening Line AND vs Closing Line performance
3. Always show 3+ edge AND 5+ edge results

# 🤝 Agent Collaboration Protocol

## 🛡️ Code Auditor (The Safety)
- **Invoke for:** Refactoring, fixing "P0/P1" audit issues, and data-integrity checks.
- **Key Files:** `scripts/backtest.py`, `src/models/efficiency_foundation_model.py`.
- **Constraint:** Ensure walk-forward chronology (training < prediction).

## 📊 Quant Auditor (The Analyst)
- **Invoke for:** Tuning weights (HFA, SOS, Portal), validating offsets, and performance reviews.
- **Constraint:** Must run `python backtest.py` before approving any logic change.
- **Success Metric:** MAE must remain stable or improve; ATS win rate > 52.3%.
