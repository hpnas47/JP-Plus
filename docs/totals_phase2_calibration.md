# Totals EV Engine: Phase 2 Calibration

## Overview

Phase 2 calibration tunes the distribution assumptions (sigma) and execution parameters (EV thresholds, Kelly sizing) for the Totals EV Engine. This ensures:

1. **Well-calibrated probabilities** - Predicted intervals match empirical coverage
2. **Sensible bet counts** - Not over-betting or under-betting
3. **Realistic ROI** - EV predictions align with realized outcomes
4. **Appropriate confidence** - Lower confidence early season when data is limited

## Quick Start

### Run Calibration (Annual Offseason Task)

```bash
# Full calibration on 2023-2025 (recommended for 2026 season)
python3 scripts/calibrate_totals_engine.py

# Include 2022 (transition year - optional)
python3 scripts/calibrate_totals_engine.py --years 2022 2023 2024 2025

# Faster iteration: reuse existing predictions
python3 scripts/calibrate_totals_engine.py --from-csv data/totals_calibration_preds.csv
```

### Output Files

- `artifacts/totals_calibration_2022_2025.json` - Production config (load in TotalsEVConfig)
- `artifacts/totals_calibration_2022_2025.report.json` - Full calibration report
- `data/totals_calibration_preds.csv` - Walk-forward predictions (for faster iteration)

## Calibration Approach

### Part A: Sigma Calibration from Residuals

We calibrate `sigma_total` (the Normal distribution standard deviation used to convert point predictions to probabilities) using walk-forward residuals.

#### Method

1. Run walk-forward backtest: for each week W, train TotalsModel on weeks 1 to W-1, predict week W
2. Collect residuals: `error = predicted_total - actual_total`
3. Compute multiple sigma estimates:
   - **Global sigma**: `std(error)` across all games
   - **Robust sigma**: MAD-based (less sensitive to outliers)
   - **Week bucket sigma**: Separate sigma for weeks 1-2, 3-5, 6-9, 10-14, 15+
   - **Phase sigma**: Separate sigma for Phase 1 (weeks 1-3), Phase 2 (4-15), Phase 3 (16+)

4. Evaluate interval coverage for each sigma candidate:
   - 50%, 68%, 80%, 90%, 95% prediction intervals
   - Score = sum of squared errors between target and empirical coverage
   - Best sigma = lowest score

#### Expected Results (2023-2025 baseline)

| Phase | Games | Sigma | RMSE | MAE |
|-------|-------|-------|------|-----|
| Phase 1 | 169 | 14.5 | 15.2 | 12.4 |
| Phase 2 | 1,824 | 12.8 | 13.1 | 10.3 |
| Phase 3 | ~50 | 13.5 | 14.0 | 11.2 |
| **Overall** | ~2,000 | **13.0** | 13.3 | 10.5 |

### Part B: ROI Backtest (When Historical Lines Available)

If historical O/U lines are available (they are via CFBD API), we backtest the full EV/Kelly system:

1. For each game with a closing line:
   - Calculate P(over), P(under), P(push) using candidate sigma
   - Calculate EV for each side at -110 odds
   - Apply Kelly sizing with caps
   - Record profit/loss based on actual result

2. Evaluate:
   - **ROI**: Total profit / Total stake
   - **Win rate**: Wins / (Wins + Losses)
   - **Cap-hit rate**: % of bets hitting max stake cap
   - **EV calibration**: Does higher predicted EV produce higher realized ROI?

3. Constraints for valid config:
   - Cap-hit rate < 30% (avoid over-concentration)
   - Bets per season: 100-400 (reasonable volume)

### Part C: Reliability Scaling

Early-season and when teams have few games, we want wider confidence intervals (higher sigma).

#### Formula

```
sigma_used = sigma_base * (1 + k * (1 - reliability))
```

Where:
- `reliability = min(rel_home, rel_away)`
- `rel_team = clamp((games_played - 1) / 7, 0, 1)`
- `k` = tunable parameter (default 0.5 = up to 50% sigma inflation)

#### Example

| Games Played | Reliability | Sigma (k=0.5, base=13) |
|--------------|-------------|------------------------|
| 1 | 0.0 | 19.5 |
| 2 | 0.14 | 18.6 |
| 4 | 0.43 | 16.7 |
| 6 | 0.71 | 14.9 |
| 8+ | 1.0 | 13.0 |

### Part D: Week Bucket Multipliers

Alternative to reliability scaling: use week-based multipliers.

```
sigma_used = sigma_base * week_bucket_multiplier
```

| Week Bucket | Typical Multiplier | Rationale |
|-------------|-------------------|-----------|
| 1-2 | 1.3 | Very early, unstable ratings |
| 3-5 | 1.1 | Still calibrating |
| 6-9 | 1.0 | Full confidence |
| 10-14 | 1.0 | Full confidence |
| 15+ | 1.1 | Postseason uncertainty |

## Configuration Options

### TotalsCalibrationConfig

```python
@dataclass
class TotalsCalibrationConfig:
    # Sigma settings
    sigma_mode: str = "fixed"  # "fixed", "week_bucket", "reliability_scaled"
    sigma_base: float = 13.0

    # Week bucket multipliers
    week_bucket_multipliers: dict = {
        "1-2": 1.3, "3-5": 1.1, "6-9": 1.0, "10-14": 1.0, "15+": 1.1
    }

    # Reliability scaling
    reliability_k: float = 0.5
    reliability_sigma_min: float = 10.0
    reliability_sigma_max: float = 25.0

    # EV thresholds
    ev_min: float = 0.02
    ev_min_phase1: float = 0.05  # Higher threshold for Phase 1

    # Kelly settings
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.02
```

### Integration with TotalsEVConfig

Load calibration in production:

```python
from src.spread_selection.totals_calibration import load_calibration, get_sigma_for_game

# Load calibration
calib = load_calibration("artifacts/totals_calibration_2022_2025.json")

# Get sigma for a specific game
sigma = get_sigma_for_game(
    config=calib,
    week=5,
    home_games_played=4,
    away_games_played=3,
)

# Use in TotalsEVConfig
from src.spread_selection import TotalsEVConfig

config = TotalsEVConfig(
    sigma_total=sigma,  # Or calib.sigma_base for fixed mode
    ev_min=calib.ev_min,
    kelly_fraction=calib.kelly_fraction,
    max_bet_fraction=calib.max_bet_fraction,
)
```

## Metrics Reference

### Interval Coverage

| Target | Z-Score | Interpretation |
|--------|---------|----------------|
| 50% | ±0.67σ | Half of games should be within ±8.7 pts (σ=13) |
| 68% | ±1.00σ | Two-thirds within ±13 pts |
| 80% | ±1.28σ | 80% within ±16.6 pts |
| 90% | ±1.65σ | 90% within ±21.4 pts |
| 95% | ±1.96σ | 95% within ±25.5 pts |

**Good calibration**: Empirical coverage ≈ target (within ±3%)

**Under-calibrated** (sigma too low): Empirical < target (intervals too narrow)

**Over-calibrated** (sigma too high): Empirical > target (intervals too wide)

### ROI Metrics

| Metric | Target | Warning |
|--------|--------|---------|
| ROI | > 0% | Negative = losing money |
| Win Rate | > 52.4% | Below breakeven at -110 |
| Cap-Hit Rate | < 30% | High = over-concentration |
| Bets/Season | 100-400 | Low = missing opportunities; High = overconfident |

## Rerunning Calibration

### When to Recalibrate

1. **Annually** before each new season (offseason task)
2. **After major changes** to TotalsModel architecture
3. **If ROI degrades significantly** mid-season (optional mid-season check)

### Annual Calibration Workflow

```bash
# 1. Generate fresh walk-forward predictions
python3 scripts/calibrate_totals_engine.py --years 2023 2024 2025 --output-csv data/totals_calibration_preds_2026.csv

# 2. Review calibration report
# (printed to stdout)

# 3. Verify JSON was saved
cat artifacts/totals_calibration_2022_2025.json

# 4. Commit updated calibration
git add artifacts/totals_calibration_2022_2025.json
git commit -m "Update totals calibration for 2026 season"
```

### Command Reference

```bash
# Default calibration
python3 scripts/calibrate_totals_engine.py

# Custom years
python3 scripts/calibrate_totals_engine.py --years 2022 2023 2024 2025

# Reuse predictions (faster)
python3 scripts/calibrate_totals_engine.py --from-csv data/totals_calibration_preds.csv

# Custom sigma range
python3 scripts/calibrate_totals_engine.py --sigma-min 10 --sigma-max 18 --sigma-step 0.5

# Custom output path
python3 scripts/calibrate_totals_engine.py --output-config artifacts/my_config.json
```

## Design Decisions

### Why Fixed Sigma as Default?

Despite having week-bucket and reliability-scaled options, **fixed sigma is recommended** because:

1. **Simplicity**: One parameter to track and explain
2. **Robustness**: Week buckets can overfit to historical patterns
3. **Phase 1 guardrails**: Separate Phase 1 protections (baseline blending, higher EV threshold) already handle early-season uncertainty

Use `sigma_mode="reliability_scaled"` if you want adaptive confidence without Phase 1 guardrails.

### Why Not Tune EV Threshold?

The EV threshold (`ev_min`) is set conservatively (2%) because:

1. **Vig cushion**: At -110, breakeven is ~52.4%; we want extra margin
2. **Calibration error**: Small EV edges can be noise
3. **Phase 1**: Use higher threshold (`ev_min_phase1=0.05`) when data is limited

### Why Separate Phase 1 Protection?

Phase 1 (weeks 1-3) has structural challenges:
- TotalsModel baseline is unstable (small training set)
- Team ratings are prior-dominated
- Market lines reflect preseason expectations that may be wrong

We handle this with:
1. **Baseline blending** (Phase 1 feature from V1)
2. **Higher sigma** (via week bucket or reliability scaling)
3. **Higher EV threshold** (`ev_min_phase1`)
4. **Diagnostic mode** when guardrails trigger

## File Structure

```
src/spread_selection/
├── totals_ev_engine.py      # Core EV engine (V1)
├── totals_calibration.py    # Calibration module (V2)
└── __init__.py              # Exports

scripts/
├── backtest_totals.py       # Walk-forward backtest
└── calibrate_totals_engine.py  # Calibration CLI

artifacts/
└── totals_calibration_2022_2025.json  # Production config

docs/
└── totals_phase2_calibration.md  # This file
```

## Troubleshooting

### "No Vegas lines available"

The CFBD API provides O/U lines. Check:
1. `fetch_season_data()` is returning betting data
2. `over_under` column exists in betting DataFrame
3. Year/week has betting data (some games may not)

### Coverage score not improving

If coverage is consistently off:
1. Check for outliers in residuals (use robust sigma)
2. Verify predictions are walk-forward (no leakage)
3. Consider week-bucket sigma for phase-specific calibration

### ROI negative despite positive EV

Possible causes:
1. **Sigma too low**: Over-confident probabilities
2. **EV threshold too low**: Including marginal bets
3. **Variance**: Small sample size (need 500+ bets for reliable ROI)

---

*Last updated: 2026-02-13*
*Calibration baseline: 2023-2025 (1,993 regular season games)*
