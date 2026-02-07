# PROJECT_MAP.md

Reference map of every file in the JP+ CFB Power Ratings Model.

---

## Core Models (`src/models/`)

| File | Purpose |
|------|---------|
| `efficiency_foundation_model.py` | **EFM** - Ridge regression on opponent-adjusted success rate + IsoPPP. Produces team offensive/defensive ratings. Primary engine for spread prediction. |
| `preseason_priors.py` | Preseason rating engine. Blends prior-year SP+, talent composite, returning production, transfer portal impact, and coaching change adjustments. |
| `special_teams.py` | Special teams model (FG, punt, kickoff). All outputs are PBTA points per game. |
| `finishing_drives.py` | Red zone / finishing drives model. Bayesian regression on drive-level RZ trip outcomes. |

---

## Adjustments (`src/adjustments/`)

| File | Purpose |
|------|---------|
| `home_field.py` | Home field advantage calculator. Dynamic HFA based on venue, crowd, elevation. |
| `travel.py` | Travel fatigue adjustment based on distance between teams. |
| `altitude.py` | Altitude adjustment for high-elevation venues (e.g., Wyoming, Air Force, Colorado). |
| `qb_adjustment.py` | QB depth chart adjustment for injuries / transfers. |
| `situational.py` | Situational adjustments (rivalry, rest days, etc.). |
| `weather.py` | Weather impact adjustments (wind, rain, temperature). |
| `aggregator.py` | Combines all adjustment components into a single stack. |
| `diagnostics.py` | Adjustment stack diagnostics and correlation analysis. |

---

## Predictions (`src/predictions/`)

| File | Purpose |
|------|---------|
| `spread_generator.py` | Generates predicted spreads from EFM ratings + adjustment stack. |
| `vegas_comparison.py` | Compares model predictions to Vegas lines. Identifies value plays via edge analysis. |

---

## API Layer (`src/api/`)

| File | Purpose |
|------|---------|
| `cfbd_client.py` | CFBD API client. Fetches games, plays, drives, team stats, talent, SP+, and betting lines. |
| `betting_lines.py` | Betting lines data structures and utilities. |
| `odds_api_client.py` | OddsAPI client for live market odds capture. |

---

## Data Processing (`src/data/`)

| File | Purpose |
|------|---------|
| `processors.py` | Data transformation and preprocessing utilities. |
| `validators.py` | Data validation routines (schema checks, NaN detection). |

---

## Configuration (`config/`)

| File | Purpose |
|------|---------|
| `settings.py` | Model hyperparameters, weights, and feature toggles. Central settings singleton. |
| `dtypes.py` | Column dtype definitions for Polars/pandas DataFrames. |
| `play_types.py` | Play type classification mappings (rush, pass, penalty, etc.). |
| `teams.py` | Team name normalization, conference mappings, P4/G5 classification. |

---

## Reports (`src/reports/`)

| File | Purpose |
|------|---------|
| `excel_export.py` | Excel workbook export with Power Ratings sheet (follows Display Protocol). |
| `html_report.py` | HTML report generation for web display. |

---

## Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `backtest.py` | **The Validator.** Walk-forward backtest engine (2022-2025). Trains on weeks < N, predicts week N. Reports MAE, ATS, CLV. |
| `run_weekly.py` | Weekly prediction pipeline. Fetches current data, runs EFM, generates spreads. |
| `weekly_odds_capture.py` | Captures live odds snapshots from OddsAPI to SQLite. |
| `capture_odds.py` | Odds capture scheduling / orchestration. |
| `compare_ratings.py` | Compare JP+ ratings across weeks or seasons. |
| `benchmark.py` | Performance benchmarking for model computation. |
| `calibrate_situational.py` | Calibration sweep for situational adjustment weights. |
| `analyze_stack_bias.py` | Analyzes adjustment stack for systematic bias. |

---

## Utilities

| File | Purpose |
|------|---------|
| `src/notifications.py` | Alert/notification system for model runs. |
| `src/utils/normalization.py` | Rating normalization utilities (z-score, min-max). |

---

## Documentation (`docs/`)

| File | Purpose |
|------|---------|
| `MODEL_ARCHITECTURE.md` | Detailed technical architecture of the JP+ model. |
| `MODEL_EXPLAINER.md` | High-level model explainer for non-technical readers. |
| `SESSION_LOG.md` | Development session log with change history. |
| `AUDIT_FIXLIST_*.md` | Per-module audit fix lists (P0/P1/P2/P3 items). |
| `PROJECT_MAP.md` | This file. |

---

## Key Data

| Path | Purpose |
|------|---------|
| `data/cfb_model.db` | SQLite database with captured odds, cached API data. |
| `CLAUDE.md` | Agent governance rules, Display Protocol, Backtest Reporting Protocol. |
