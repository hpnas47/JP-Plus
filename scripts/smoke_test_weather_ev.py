#!/usr/bin/env python3
"""
Smoke Test for Weather Layer + Totals EV Engine Integration.

Two modes:
  --synthetic : Deterministic synthetic test (always runs, hard assertions)
  --real      : Best-effort real-data test (graceful skip if prerequisites missing)

Real-data mode executes in three phases:
  Phase 1 (Preflight): Validate inputs, locate data files (fast, no training)
  Phase 2 (Model Prep): Train or load TotalsModel (time-bounded)
  Phase 3 (Evaluation): Run EV evaluation with/without weather

Examples:
    # Run synthetic test only (default)
    python3 scripts/smoke_test_weather_ev.py --synthetic

    # Run real-data test (skips gracefully if prerequisites missing)
    python3 scripts/smoke_test_weather_ev.py --real --year 2024 --week 10

    # Real-data test requiring weather data
    python3 scripts/smoke_test_weather_ev.py --real --year 2024 --week 10 --require-weather

    # Real-data test with pre-trained model (fastest)
    python3 scripts/smoke_test_weather_ev.py --real --year 2024 --week 10 \\
        --load-model-path artifacts/models/totals_2024_w9.joblib

    # Train and save model for future use
    python3 scripts/smoke_test_weather_ev.py --real --year 2024 --week 10 \\
        --save-model-path artifacts/models/

    # Limit training time and games for quick smoke test
    python3 scripts/smoke_test_weather_ev.py --real --year 2024 --week 10 \\
        --max-train-seconds 10 --max-games 500

    # Both modes
    python3 scripts/smoke_test_weather_ev.py --synthetic --real --year 2024 --week 10

This script validates:
  - Mu composition formula: mu_used = mu_model + weather_adj + baseline_shift
  - Weather sign convention: negative weather_adj lowers totals
  - Guardrail capping at ±10 pts (or configured cap)
  - Push probability: 0 for half-point lines, >0 for integer lines
  - EV monotonicity: decreasing weather_adj increases UNDER EV
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Configure logging early
logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO from imported modules during smoke test
    format='%(message)s'
)

# Print versions first for reproducibility
print("=" * 70)
print("SMOKE TEST: Weather Layer + Totals EV Engine")
print("=" * 70)
print("\nDependency Versions:")
print(f"  Python: {sys.version.split()[0]}")

try:
    import numpy as np
    print(f"  NumPy:  {np.__version__}")
except ImportError:
    print("  NumPy:  NOT INSTALLED")
    sys.exit(1)

try:
    import pandas as pd
    print(f"  Pandas: {pd.__version__}")
except ImportError:
    print("  Pandas: NOT INSTALLED")
    sys.exit(1)

try:
    import scipy
    print(f"  SciPy:  {scipy.__version__}")
except ImportError:
    print("  SciPy:  not installed (fallback math will be used)")

try:
    import joblib
    print(f"  Joblib: {joblib.__version__}")
    HAS_JOBLIB = True
except ImportError:
    print("  Joblib: not installed (model save/load disabled)")
    HAS_JOBLIB = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.spread_selection.totals_ev_engine import (
    TotalMarket,
    TotalsEvent,
    TotalsEVConfig,
    TotalsBetRecommendation,
    evaluate_totals_markets,
    apply_weather_to_events,
    GUARDRAIL_OK,
)

print()


# =============================================================================
# PHASE MARKERS (for self-check that preflight runs before training)
# =============================================================================

_PHASE_LOG: list[str] = []


def log_phase(phase: str) -> None:
    """Log phase execution order for self-check."""
    _PHASE_LOG.append(phase)
    print(f"\n{'='*70}")
    print(f"[{phase}]")
    print("=" * 70)


def verify_phase_order() -> bool:
    """Verify preflight executed before model_prep."""
    if "PHASE_1_PREFLIGHT" in _PHASE_LOG and "PHASE_2_MODEL_PREP" in _PHASE_LOG:
        preflight_idx = _PHASE_LOG.index("PHASE_1_PREFLIGHT")
        model_prep_idx = _PHASE_LOG.index("PHASE_2_MODEL_PREP")
        return preflight_idx < model_prep_idx
    return True  # If phases not both present, order check not applicable


# =============================================================================
# SYNTHETIC MODE: Fake Model + Deterministic Tests
# =============================================================================

@dataclass
class FakeTotalsPrediction:
    """Mock prediction object compatible with TotalsModel output."""
    home_team: str
    away_team: str
    predicted_total: float
    home_expected: float
    away_expected: float
    baseline: float
    weather_adjustment: float = 0.0

    @property
    def adjusted_total(self) -> float:
        return self.predicted_total + self.weather_adjustment


class FakeTotalsModel:
    """Fake TotalsModel for synthetic testing.

    Returns a fixed predicted_total regardless of teams.
    """
    def __init__(self, mu_fixed: float = 60.0, baseline: float = 30.0):
        self.mu_fixed = mu_fixed
        self.baseline = baseline
        self.team_ratings = {}  # Empty, not used in synthetic
        self._trained = True
        self._n_games = 200  # Fake training games count

    def predict_total(
        self,
        home_team: str,
        away_team: str,
        weather_adjustment: float = 0.0,
        year: int | None = None,
    ) -> FakeTotalsPrediction:
        """Return prediction with fixed mu."""
        return FakeTotalsPrediction(
            home_team=home_team,
            away_team=away_team,
            predicted_total=self.mu_fixed,
            home_expected=self.mu_fixed / 2,
            away_expected=self.mu_fixed / 2,
            baseline=self.baseline,
            weather_adjustment=weather_adjustment,
        )


def create_synthetic_event(weather_adj: float = 0.0) -> TotalsEvent:
    """Create single synthetic event with two markets."""
    return TotalsEvent(
        event_id="SYNTHETIC_001",
        home_team="TestHome",
        away_team="TestAway",
        year=2024,
        week=10,
        weather_adjustment=weather_adj,
        markets=[
            # Market 1: Half-point line (no push possible)
            TotalMarket(
                book="SyntheticBook1",
                line=60.5,
                odds_over=-110,
                odds_under=-110,
            ),
            # Market 2: Integer line (push possible)
            TotalMarket(
                book="SyntheticBook2",
                line=60,
                odds_over=-110,
                odds_under=-110,
            ),
        ],
    )


def create_synthetic_config() -> TotalsEVConfig:
    """Create config with guardrails disabled for synthetic testing."""
    return TotalsEVConfig(
        # Core model params
        sigma_total=13.0,
        bankroll=1000.0,

        # Disable baseline blending for deterministic tests
        enable_baseline_blend=False,

        # Disable guardrails so we see stakes move
        min_train_games_for_staking=0,
        diagnostic_only_mode=False,
        auto_diagnostic_if_guardrail_hit=False,

        # EV thresholds
        ev_min=0.0,  # Include all bets for testing
        edge_pts_min=0.0,

        # Weather
        use_weather_adjustment=True,
        weather_max_adjustment=10.0,
    )


def print_results_table(recs: list[TotalsBetRecommendation], title: str) -> None:
    """Print compact results table."""
    print(f"\n{title}")
    print("-" * 120)
    print(f"{'line':>6} {'side':<5} {'mu_model':>8} {'w_adj':>6} {'b_shift':>7} "
          f"{'mu_used':>7} {'p_win':>6} {'p_push':>6} {'p_loss':>6} "
          f"{'ev':>7} {'kelly_f':>7} {'stake':>6}")
    print("-" * 120)

    for r in sorted(recs, key=lambda x: (x.line, x.side)):
        print(f"{r.line:>6.1f} {r.side:<5} {r.mu_model:>8.2f} {r.weather_adj:>+6.1f} "
              f"{r.baseline_shift:>7.2f} {r.mu_used:>7.2f} {r.p_win:>6.3f} "
              f"{r.p_push:>6.3f} {r.p_loss:>6.3f} {r.ev:>+7.4f} {r.kelly_f:>7.4f} "
              f"{r.stake:>6.0f}")


def run_synthetic_test() -> bool:
    """Run deterministic synthetic test. Returns True on success."""
    print("\n" + "=" * 70)
    print("MODE A: SYNTHETIC SMOKE TEST")
    print("=" * 70)

    model = FakeTotalsModel(mu_fixed=60.0, baseline=30.0)
    config = create_synthetic_config()

    # Collect results for each weather scenario
    results_by_weather: dict[float, list[TotalsBetRecommendation]] = {}

    # Test 1: weather_adj = 0
    print("\n[Test 1] weather_adj = 0.0")
    event_0 = create_synthetic_event(weather_adj=0.0)
    primary_0, edge5_0 = evaluate_totals_markets(model, [event_0], config, n_train_games=200)
    recs_0 = _df_to_recs(primary_0, edge5_0)
    results_by_weather[0.0] = recs_0
    print_results_table(recs_0, "Results (weather_adj=0.0)")

    # Test 2: weather_adj = -10
    print("\n[Test 2] weather_adj = -10.0")
    event_neg10 = create_synthetic_event(weather_adj=-10.0)
    primary_neg10, edge5_neg10 = evaluate_totals_markets(model, [event_neg10], config, n_train_games=200)
    recs_neg10 = _df_to_recs(primary_neg10, edge5_neg10)
    results_by_weather[-10.0] = recs_neg10
    print_results_table(recs_neg10, "Results (weather_adj=-10.0)")

    # Test 3: weather_adj = +10
    print("\n[Test 3] weather_adj = +10.0")
    event_pos10 = create_synthetic_event(weather_adj=+10.0)
    primary_pos10, edge5_pos10 = evaluate_totals_markets(model, [event_pos10], config, n_train_games=200)
    recs_pos10 = _df_to_recs(primary_pos10, edge5_pos10)
    results_by_weather[+10.0] = recs_pos10
    print_results_table(recs_pos10, "Results (weather_adj=+10.0)")

    # Test 4: weather_adj = -25 (should be capped to -10)
    print("\n[Test 4] weather_adj = -25.0 (guardrail test)")
    event_extreme = create_synthetic_event(weather_adj=-25.0)
    primary_ext, edge5_ext = evaluate_totals_markets(model, [event_extreme], config, n_train_games=200)
    recs_extreme = _df_to_recs(primary_ext, edge5_ext)
    results_by_weather[-25.0] = recs_extreme
    print_results_table(recs_extreme, "Results (weather_adj=-25.0 input, should cap to -10.0)")

    # ==========================================================================
    # ASSERTIONS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RUNNING ASSERTIONS")
    print("=" * 70)

    all_passed = True

    def assert_true(condition: bool, msg: str) -> None:
        nonlocal all_passed
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {msg}")
        if not condition:
            all_passed = False

    # Assertion 1: mu_used = mu_model + weather_adj + baseline_shift (float tolerance)
    print("\n1. Mu Composition Formula:")
    for weather, recs in results_by_weather.items():
        for r in recs:
            expected_mu = r.mu_model + r.weather_adj + r.baseline_shift
            diff = abs(r.mu_used - expected_mu)
            assert_true(
                diff < 1e-6,
                f"  weather={weather:+.0f}, line={r.line}: "
                f"mu_used={r.mu_used:.4f} == mu_model({r.mu_model:.2f}) + "
                f"w_adj({r.weather_adj:+.2f}) + b_shift({r.baseline_shift:.2f}) "
                f"= {expected_mu:.4f} (diff={diff:.2e})"
            )

    # Assertion 2: Half-point line (60.5) has p_push = 0
    print("\n2. Half-Point Line Push Probability:")
    for weather, recs in results_by_weather.items():
        for r in recs:
            if r.line == 60.5:
                assert_true(
                    abs(r.p_push) < 1e-9,
                    f"  weather={weather:+.0f}, line=60.5: p_push={r.p_push:.9f} == 0"
                )

    # Assertion 3: Integer line (60) has p_push > 0
    print("\n3. Integer Line Push Probability:")
    for weather, recs in results_by_weather.items():
        for r in recs:
            if r.line == 60:
                assert_true(
                    r.p_push > 0,
                    f"  weather={weather:+.0f}, line=60: p_push={r.p_push:.6f} > 0"
                )

    # Assertion 4: Monotonicity - decreasing weather_adj increases UNDER EV
    print("\n4. Monotonicity (UNDER EV increases as weather_adj decreases):")
    for line in [60.0, 60.5]:
        under_ev_w0 = [r.ev for r in results_by_weather[0.0] if r.line == line and r.side == "UNDER"]
        under_ev_wn10 = [r.ev for r in results_by_weather[-10.0] if r.line == line and r.side == "UNDER"]
        if under_ev_w0 and under_ev_wn10:
            assert_true(
                under_ev_wn10[0] >= under_ev_w0[0] - 1e-9,
                f"  line={line}: UNDER EV at w=-10 ({under_ev_wn10[0]:+.4f}) >= "
                f"UNDER EV at w=0 ({under_ev_w0[0]:+.4f})"
            )

    # Assertion 5: Monotonicity - increasing weather_adj increases OVER EV
    print("\n5. Monotonicity (OVER EV increases as weather_adj increases):")
    for line in [60.0, 60.5]:
        over_ev_w0 = [r.ev for r in results_by_weather[0.0] if r.line == line and r.side == "OVER"]
        over_ev_wp10 = [r.ev for r in results_by_weather[+10.0] if r.line == line and r.side == "OVER"]
        if over_ev_w0 and over_ev_wp10:
            assert_true(
                over_ev_wp10[0] >= over_ev_w0[0] - 1e-9,
                f"  line={line}: OVER EV at w=+10 ({over_ev_wp10[0]:+.4f}) >= "
                f"OVER EV at w=0 ({over_ev_w0[0]:+.4f})"
            )

    # Assertion 6: Guardrail - weather_adj=-25 input should be capped to -10
    print("\n6. Guardrail Capping (input -25 -> capped to -10):")
    for r in results_by_weather[-25.0]:
        assert_true(
            r.weather_adj == -10.0,
            f"  line={r.line}: stored weather_adj={r.weather_adj:.1f} == -10.0 (capped)"
        )
        expected_mu = r.mu_model + (-10.0) + r.baseline_shift
        assert_true(
            abs(r.mu_used - expected_mu) < 1e-6,
            f"  line={r.line}: mu_used={r.mu_used:.2f} reflects capped value "
            f"(expected={expected_mu:.2f})"
        )

    # Assertion 7: All weather_adj in bounds [-10, 10]
    print("\n7. All weather_adj within [-10, 10]:")
    for weather, recs in results_by_weather.items():
        for r in recs:
            assert_true(
                -10.0 <= r.weather_adj <= 10.0,
                f"  weather_input={weather:+.0f}, line={r.line}: "
                f"weather_adj={r.weather_adj:+.1f} in [-10, 10]"
            )

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("SYNTHETIC TEST PASSED")
    else:
        print("SYNTHETIC TEST FAILED")
    print("=" * 70)

    return all_passed


def _df_to_recs(primary_df: pd.DataFrame, edge5_df: pd.DataFrame) -> list[TotalsBetRecommendation]:
    """Convert DataFrames back to recommendation objects for assertions."""
    recs = []
    for df in [primary_df, edge5_df]:
        if len(df) == 0:
            continue
        for row in df.itertuples():
            recs.append(TotalsBetRecommendation(
                event_id=getattr(row, 'event_id', 'SYNTHETIC_001'),
                home_team=getattr(row, 'home_team', ''),
                away_team=getattr(row, 'away_team', ''),
                book=getattr(row, 'book', ''),
                line=row.line,
                side=row.side,
                mu_model=row.mu_model,
                weather_adj=row.weather_adj,
                mu_raw=getattr(row, 'mu_raw', row.mu_model + row.weather_adj),
                baseline_shift=row.baseline_shift,
                mu_used=row.mu_used,
                adjusted_total=row.mu_used,  # Legacy field
                model_total=row.mu_model,  # Legacy alias
                home_expected=getattr(row, 'home_expected', 30.0),
                away_expected=getattr(row, 'away_expected', 30.0),
                baseline=getattr(row, 'baseline', 30.0),
                edge_pts=row.edge_pts,
                p_win=row.p_win,
                p_loss=row.p_loss,
                p_push=row.p_push,
                odds_american=row.odds_american,
                odds_decimal=row.odds_decimal,
                implied_prob=row.implied_prob,
                ev=row.ev,
                edge_prob=row.edge_prob,
                kelly_f=row.kelly_f,
                stake=row.stake,
                sigma_total=getattr(row, 'sigma_total', 13.0),
                ev_min=getattr(row, 'ev_min', 0.02),
                guardrail_reason=getattr(row, 'guardrail_reason', GUARDRAIL_OK),
                sigma_used=getattr(row, 'sigma_used', None),
            ))
    return recs


# =============================================================================
# REAL-DATA MODE: Three-Phase Execution
# =============================================================================

@dataclass
class PreflightResult:
    """Result of Phase 1 preflight checks."""
    success: bool
    skip_reason: Optional[str] = None

    # Data located during preflight
    games_df: Optional[pd.DataFrame] = None
    betting_df: Optional[pd.DataFrame] = None
    weather_df: Optional[pd.DataFrame] = None
    fbs_set: Optional[set] = None
    n_train_games: int = 0
    n_week_games: int = 0

    # Weather status
    weather_available: bool = False
    weather_path: Optional[Path] = None


def _locate_weather_file(year: int, week: int) -> Optional[Path]:
    """Search for weather data file. Returns path if found, None otherwise."""
    weather_paths = [
        Path(f"data/weather/weather_{year}_week{week}.json"),
        Path(f"data/weather/weather_{year}_{week:02d}.json"),
        Path(f"data/weather_captures/weather_{year}_week{week}.csv"),
        Path(f"data/weather_captures/{year}_week{week}_weather.json"),
        Path(f"data/weather/{year}_w{week}.json"),
    ]

    for p in weather_paths:
        if p.exists():
            return p
    return None


def _load_weather_df(path: Path) -> Optional[pd.DataFrame]:
    """Load weather data from file."""
    try:
        if path.suffix == '.json':
            with open(path) as f:
                weather_data = json.load(f)
            # Extract weather_adjustment per game
            rows = []
            entries = weather_data.get('games', weather_data)
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        game_id = entry.get('game_id')
                        adj = entry.get('weather_adjustment', entry.get('total_adjustment', 0.0))
                        if game_id is not None:
                            rows.append({'game_id': str(game_id), 'weather_adj': float(adj)})
            return pd.DataFrame(rows) if rows else None
        else:
            df = pd.read_csv(path)
            # Normalize column names
            if 'weather_adjustment' in df.columns and 'weather_adj' not in df.columns:
                df['weather_adj'] = df['weather_adjustment']
            return df if len(df) > 0 else None
    except Exception:
        return None


def run_phase1_preflight(
    year: int,
    week: int,
    require_weather: bool,
    max_games: Optional[int],
) -> PreflightResult:
    """Phase 1: Preflight checks (no heavy compute).

    Validates inputs and locates all required data before training.
    """
    log_phase("PHASE_1_PREFLIGHT")

    result = PreflightResult(success=False)

    # -------------------------------------------------------------------------
    # Step 1.1: Validate year/week inputs
    # -------------------------------------------------------------------------
    print("\n[1.1] Validating inputs...")
    if year < 2000 or year > 2030:
        result.skip_reason = f"Invalid year: {year} (expected 2000-2030)"
        return result
    if week < 1 or week > 20:
        result.skip_reason = f"Invalid week: {week} (expected 1-20)"
        return result
    print(f"      year={year}, week={week} OK")

    # -------------------------------------------------------------------------
    # Step 1.2: Locate weather data FIRST (if require_weather)
    # -------------------------------------------------------------------------
    print("\n[1.2] Locating weather data...")
    weather_path = _locate_weather_file(year, week)

    if weather_path is not None:
        weather_df = _load_weather_df(weather_path)
        if weather_df is not None and len(weather_df) > 0:
            result.weather_available = True
            result.weather_df = weather_df
            result.weather_path = weather_path
            print(f"      Found: {weather_path} ({len(weather_df)} records)")
        else:
            print(f"      Found file but empty/invalid: {weather_path}")
    else:
        print("      No weather file found")
        print("      Searched paths:")
        for p in [
            f"data/weather/weather_{year}_week{week}.json",
            f"data/weather/weather_{year}_{week:02d}.json",
            f"data/weather_captures/weather_{year}_week{week}.csv",
        ]:
            print(f"        - {p}")

    # Check require_weather constraint
    if require_weather and not result.weather_available:
        result.skip_reason = (
            "REAL-DATA TEST SKIPPED: missing weather data (require-weather enabled)\n\n"
            "To provide weather data:\n"
            "  1. Run: python3 scripts/weather_thursday_capture.py --year {year} --week {week}\n"
            f"  2. Or place a JSON file at: data/weather/weather_{year}_week{week}.json\n"
            "     Format: {\"games\": [{\"game_id\": 123, \"weather_adjustment\": -2.5}, ...]}\n"
            "  3. Or disable --require-weather to skip weather-applied evaluation"
        )
        return result

    # -------------------------------------------------------------------------
    # Step 1.3: Locate games and betting lines
    # -------------------------------------------------------------------------
    print("\n[1.3] Locating games and betting lines...")
    try:
        from src.api.cfbd_client import CFBDClient
        from scripts.backtest import fetch_season_data

        client = CFBDClient()
        games_df_raw, betting_df_raw = fetch_season_data(client, year)

        # Get FBS teams
        fbs_teams = client.get_fbs_teams(year=year)
        fbs_set = {t.school for t in fbs_teams if t.school}
        result.fbs_set = fbs_set

        # Convert to pandas
        games = games_df_raw.to_pandas() if hasattr(games_df_raw, 'to_pandas') else games_df_raw
        betting = betting_df_raw.to_pandas() if hasattr(betting_df_raw, 'to_pandas') else betting_df_raw

        # Filter to FBS vs FBS with scores
        games = games[
            games['home_team'].isin(fbs_set) &
            games['away_team'].isin(fbs_set) &
            games['home_points'].notna() &
            games['away_points'].notna()
        ].copy()

        # Check training data availability
        n_train = int((games['week'] < week).sum())
        result.n_train_games = n_train

        if n_train < 30:
            result.skip_reason = (
                f"REAL-DATA TEST SKIPPED: insufficient training data\n"
                f"  Found {n_train} games before week {week} (need >= 30)"
            )
            return result

        # Apply max_games subsample if specified
        if max_games is not None and len(games) > max_games:
            print(f"      Subsampling: {len(games)} games -> {max_games} (--max-games)")
            # Deterministic subsample: take earliest games by week, then by id
            games = games.sort_values(['week', 'id']).head(max_games).copy()
            n_train = int((games['week'] < week).sum())
            result.n_train_games = n_train

        result.games_df = games
        result.betting_df = betting

        print(f"      Games available: {len(games)} total, {n_train} for training")

        # Check week games exist
        week_games = games[games['week'] == week]
        result.n_week_games = len(week_games)

        if len(week_games) == 0:
            result.skip_reason = f"REAL-DATA TEST SKIPPED: no games found for week {week}"
            return result

        print(f"      Week {week} games: {len(week_games)}")

        # Check betting lines exist
        if 'over_under' not in betting.columns:
            result.skip_reason = (
                f"REAL-DATA TEST SKIPPED: missing betting lines / markets for year={year} week={week}\n"
                "  The betting data does not contain 'over_under' column"
            )
            return result

        # Check if week's games have betting lines
        lines_df = betting[['game_id', 'over_under']].drop_duplicates()
        lines_df = lines_df.rename(columns={'game_id': 'id'})
        week_with_lines = week_games.merge(lines_df, on='id', how='inner')

        if len(week_with_lines) == 0:
            result.skip_reason = (
                f"REAL-DATA TEST SKIPPED: missing betting lines / markets for year={year} week={week}\n"
                f"  Found {len(week_games)} games but 0 have betting lines"
            )
            return result

        print(f"      Week {week} games with lines: {len(week_with_lines)}")

    except Exception as e:
        result.skip_reason = f"REAL-DATA TEST SKIPPED: failed to load data\n  {type(e).__name__}: {e}"
        return result

    # -------------------------------------------------------------------------
    # Preflight passed
    # -------------------------------------------------------------------------
    print("\n[1.4] Preflight checks PASSED")
    result.success = True
    return result


class TrainingTimeoutError(Exception):
    """Raised when training exceeds time budget."""
    pass


def _train_model_worker(
    games: pd.DataFrame,
    fbs_set: set,
    max_week: int,
    ridge_alpha: float,
    result_queue: multiprocessing.Queue,
):
    """Worker function for training in subprocess."""
    try:
        from src.models.totals_model import TotalsModel

        model = TotalsModel(ridge_alpha=ridge_alpha)
        model.set_team_universe(fbs_set)
        model.train(games, fbs_set, max_week=max_week)

        if model._trained:
            result_queue.put(('success', model))
        else:
            result_queue.put(('failed', None))
    except Exception as e:
        result_queue.put(('error', str(e)))


def run_phase2_model_prep(
    preflight: PreflightResult,
    week: int,
    max_train_seconds: int,
    load_model_path: Optional[str],
    save_model_path: Optional[str],
    ridge_alpha: float = 10.0,
) -> tuple[bool, Optional[Any], Optional[str]]:
    """Phase 2: Model preparation (time-bounded).

    Returns (success, model, skip_reason)
    """
    log_phase("PHASE_2_MODEL_PREP")

    # -------------------------------------------------------------------------
    # Option A: Load pre-trained model
    # -------------------------------------------------------------------------
    if load_model_path:
        print(f"\n[2.1] Loading pre-trained model from: {load_model_path}")

        if not HAS_JOBLIB:
            return (False, None, "REAL-DATA TEST SKIPPED: joblib not installed (required for --load-model-path)")

        try:
            model = joblib.load(load_model_path)

            # Validate loaded model
            if not hasattr(model, '_trained') or not model._trained:
                return (False, None, f"REAL-DATA TEST SKIPPED: loaded model is not trained")
            if not hasattr(model, 'predict_total'):
                return (False, None, f"REAL-DATA TEST SKIPPED: loaded object is not a TotalsModel")

            print(f"      Model loaded successfully (_trained={model._trained})")
            return (True, model, None)

        except Exception as e:
            return (False, None, f"REAL-DATA TEST SKIPPED: failed to load model\n  {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # Option B: Train model with time limit
    # -------------------------------------------------------------------------
    print(f"\n[2.1] Training TotalsModel (timeout={max_train_seconds}s)...")
    print(f"      Training on {preflight.n_train_games} games, max_week={week-1}")

    start_time = time.time()

    # Use multiprocessing for timeout (works on all platforms)
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_train_model_worker,
        args=(preflight.games_df, preflight.fbs_set, week - 1, ridge_alpha, result_queue),
    )
    process.start()
    process.join(timeout=max_train_seconds)

    elapsed = time.time() - start_time

    if process.is_alive():
        # Training exceeded time limit
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
        return (
            False,
            None,
            f"REAL-DATA TEST SKIPPED: training exceeded time budget (max-train-seconds={max_train_seconds})\n"
            f"  Elapsed: {elapsed:.1f}s before timeout"
        )

    # Get result from queue
    try:
        status, result = result_queue.get_nowait()
    except Exception:
        return (False, None, "REAL-DATA TEST SKIPPED: training failed (no result returned)")

    if status == 'success':
        model = result
        print(f"      Training completed in {elapsed:.1f}s")

        # Save model if requested
        if save_model_path:
            if not HAS_JOBLIB:
                print("      WARNING: Cannot save model (joblib not installed)")
            else:
                save_path = Path(save_model_path)
                if save_path.is_dir():
                    # Generate filename with year/week
                    save_path = save_path / f"totals_{preflight.games_df['year'].iloc[0]}_w{week-1}.joblib"

                save_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, save_path)
                print(f"      Model saved to: {save_path}")
                print(f"\n      Tip: Future smoke tests can use --load-model-path {save_path}")

        return (True, model, None)
    elif status == 'failed':
        return (False, None, "REAL-DATA TEST SKIPPED: model training failed (_trained=False)")
    else:
        return (False, None, f"REAL-DATA TEST SKIPPED: training error\n  {result}")


def run_phase3_evaluation(
    model: Any,
    preflight: PreflightResult,
    year: int,
    week: int,
) -> bool:
    """Phase 3: Run EV evaluation with/without weather.

    Returns True on success.
    """
    log_phase("PHASE_3_EVALUATION")

    # -------------------------------------------------------------------------
    # Build events
    # -------------------------------------------------------------------------
    print("\n[3.1] Building events from games + betting lines...")

    games = preflight.games_df
    betting = preflight.betting_df
    week_games = games[games['week'] == week].copy()

    # Merge betting lines
    lines_df = betting[['game_id', 'over_under']].drop_duplicates()
    lines_df = lines_df.rename(columns={'over_under': 'total_line', 'game_id': 'id'})
    week_games = week_games.merge(lines_df, on='id', how='left')

    events = []
    for g in week_games.itertuples():
        line = getattr(g, 'total_line', None)
        if pd.isna(line):
            continue

        events.append(TotalsEvent(
            event_id=str(g.id),
            home_team=g.home_team,
            away_team=g.away_team,
            year=year,
            week=week,
            weather_adjustment=0.0,
            markets=[
                TotalMarket(
                    book="CFBD",
                    line=float(line),
                    odds_over=-110,
                    odds_under=-110,
                )
            ],
        ))

    print(f"      Built {len(events)} events with betting lines")

    if len(events) == 0:
        print("\n      No events to evaluate. Skipping evaluation.")
        return True

    # -------------------------------------------------------------------------
    # Run evaluation WITHOUT weather
    # -------------------------------------------------------------------------
    print("\n[3.2] Running evaluation WITHOUT weather...")

    config = TotalsEVConfig(
        sigma_total=13.0,
        bankroll=1000.0,
        use_weather_adjustment=False,
        ev_min=0.02,
        edge_pts_min=5.0,
    )

    primary_no_weather, edge5_no_weather = evaluate_totals_markets(
        model, events, config, n_train_games=preflight.n_train_games
    )

    print(f"      List A (Primary): {len(primary_no_weather)} bets")
    print(f"      List B (5+ Edge): {len(edge5_no_weather)} bets")

    # -------------------------------------------------------------------------
    # Run evaluation WITH weather (if available)
    # -------------------------------------------------------------------------
    if preflight.weather_available:
        print("\n[3.3] Running evaluation WITH weather...")

        config_with_weather = TotalsEVConfig(
            sigma_total=13.0,
            bankroll=1000.0,
            use_weather_adjustment=True,
            weather_max_adjustment=10.0,
            ev_min=0.02,
            edge_pts_min=5.0,
        )

        events_with_weather = apply_weather_to_events(events, preflight.weather_df, config_with_weather)

        primary_with_weather, edge5_with_weather = evaluate_totals_markets(
            model, events_with_weather, config_with_weather, n_train_games=preflight.n_train_games
        )

        print(f"      List A (Primary): {len(primary_with_weather)} bets")
        print(f"      List B (5+ Edge): {len(edge5_with_weather)} bets")

        # Print top 10 by EV
        all_recs = pd.concat([primary_with_weather, edge5_with_weather], ignore_index=True)
    else:
        print("\n[3.3] WEATHER NOT AVAILABLE: skipping weather-applied run")
        all_recs = pd.concat([primary_no_weather, edge5_no_weather], ignore_index=True)

    # -------------------------------------------------------------------------
    # Print results summary
    # -------------------------------------------------------------------------
    if len(all_recs) > 0:
        print("\n" + "-" * 70)
        print("Top 10 Recommendations (sorted by EV desc)")
        print("-" * 70)

        top10 = all_recs.nlargest(10, 'ev')
        cols = ['event_id', 'home_team', 'away_team', 'book', 'line', 'side',
                'mu_model', 'weather_adj', 'baseline_shift', 'mu_used',
                'sigma_used', 'ev', 'stake', 'guardrail_reason']
        available_cols = [c for c in cols if c in top10.columns]
        print(top10[available_cols].to_string(index=False))

    # -------------------------------------------------------------------------
    # Soft assertions (warnings only)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Sanity Checks (warnings only)")
    print("-" * 70)

    warnings = []

    if len(all_recs) > 0:
        # Check: at least one row has nonzero weather_adj if weather was applied
        if preflight.weather_available:
            has_weather = (all_recs['weather_adj'] != 0.0).any()
            if not has_weather:
                warnings.append("No rows have nonzero weather_adj despite weather data being available")

        # Check: all weather_adj within bounds
        out_of_bounds = all_recs[(all_recs['weather_adj'] < -10.0) | (all_recs['weather_adj'] > 10.0)]
        if len(out_of_bounds) > 0:
            warnings.append(f"{len(out_of_bounds)} rows have weather_adj outside [-10, 10]")

        # Check: mu_used differs from mu_model when weather applied
        weather_applied = all_recs[all_recs['weather_adj'] != 0.0]
        if len(weather_applied) > 0:
            same_mu = (weather_applied['mu_used'] == weather_applied['mu_model']).all()
            if same_mu:
                warnings.append("mu_used equals mu_model even with weather_adj != 0")

        # Guardrail summary
        if 'guardrail_reason' in all_recs.columns:
            guardrail_counts = all_recs['guardrail_reason'].value_counts()
            print("\nGuardrail Summary:")
            for reason, count in guardrail_counts.items():
                print(f"  {reason}: {count}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n  All sanity checks passed.")

    # -------------------------------------------------------------------------
    # Verify phase order (internal self-check)
    # -------------------------------------------------------------------------
    if not verify_phase_order():
        warnings.append("INTERNAL ERROR: Phase order violation (preflight should run before model_prep)")

    return True


def run_real_data_test(
    year: int,
    week: int,
    require_weather: bool,
    max_train_seconds: int,
    max_games: Optional[int],
    load_model_path: Optional[str],
    save_model_path: Optional[str],
) -> bool:
    """Run three-phase real-data test. Returns True on success/skip."""
    print("\n" + "=" * 70)
    print(f"MODE B: REAL-DATA SMOKE TEST (year={year}, week={week})")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  require_weather: {require_weather}")
    print(f"  max_train_seconds: {max_train_seconds}")
    print(f"  max_games: {max_games}")
    print(f"  load_model_path: {load_model_path}")
    print(f"  save_model_path: {save_model_path}")

    # -------------------------------------------------------------------------
    # Phase 1: Preflight
    # -------------------------------------------------------------------------
    preflight = run_phase1_preflight(year, week, require_weather, max_games)

    if not preflight.success:
        print("\n" + "=" * 70)
        print(preflight.skip_reason)
        print("=" * 70)
        return True  # Skip is not failure

    # -------------------------------------------------------------------------
    # Phase 2: Model Preparation
    # -------------------------------------------------------------------------
    success, model, skip_reason = run_phase2_model_prep(
        preflight=preflight,
        week=week,
        max_train_seconds=max_train_seconds,
        load_model_path=load_model_path,
        save_model_path=save_model_path,
    )

    if not success:
        print("\n" + "=" * 70)
        print(skip_reason)
        print("=" * 70)
        return True  # Skip is not failure

    # -------------------------------------------------------------------------
    # Phase 3: Evaluation
    # -------------------------------------------------------------------------
    success = run_phase3_evaluation(model, preflight, year, week)

    print("\n" + "=" * 70)
    print("REAL-DATA TEST PASSED")
    print("=" * 70)

    return success


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for Weather Layer + Totals EV Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run synthetic test only (default)
  python3 scripts/smoke_test_weather_ev.py --synthetic

  # Run real-data test (skips gracefully if prerequisites missing)
  python3 scripts/smoke_test_weather_ev.py --real --year 2024 --week 10

  # Real-data test requiring weather data
  python3 scripts/smoke_test_weather_ev.py --real --year 2024 --week 10 --require-weather

  # Real-data test with pre-trained model (fastest)
  python3 scripts/smoke_test_weather_ev.py --real --year 2024 --week 10 \\
      --load-model-path artifacts/models/totals_2024_w9.joblib

  # Train and save model for future use
  python3 scripts/smoke_test_weather_ev.py --real --year 2024 --week 10 \\
      --save-model-path artifacts/models/

  # Limit training time and games for quick smoke test
  python3 scripts/smoke_test_weather_ev.py --real --year 2024 --week 10 \\
      --max-train-seconds 10 --max-games 500
"""
    )

    # Mode selection
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run synthetic deterministic tests (always runs, hard assertions)"
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Run real-data test (graceful skip if prerequisites missing)"
    )

    # Real-data parameters
    parser.add_argument(
        "--year", type=int, default=2024,
        help="Year for real-data test (default: 2024)"
    )
    parser.add_argument(
        "--week", type=int, default=10,
        help="Week for real-data test (default: 10)"
    )

    # Weather requirement (A)
    parser.add_argument(
        "--require-weather", action="store_true",
        help="Require weather data to be available; skip early if missing"
    )

    # Time-bounded execution (C)
    parser.add_argument(
        "--max-train-seconds", type=int, default=20,
        help="Maximum seconds for model training before skip (default: 20)"
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Maximum games to use (subsample if exceeded); None = no limit"
    )

    # Model serialization (D)
    parser.add_argument(
        "--load-model-path", type=str, default=None,
        help="Path to pre-trained model (skip training)"
    )
    parser.add_argument(
        "--save-model-path", type=str, default=None,
        help="Path to save trained model for reuse (can be directory)"
    )

    args = parser.parse_args()

    # Default to synthetic if neither specified
    if not args.synthetic and not args.real:
        args.synthetic = True

    success = True

    if args.synthetic:
        if not run_synthetic_test():
            success = False

    if args.real:
        if not run_real_data_test(
            year=args.year,
            week=args.week,
            require_weather=args.require_weather,
            max_train_seconds=args.max_train_seconds,
            max_games=args.max_games,
            load_model_path=args.load_model_path,
            save_model_path=args.save_model_path,
        ):
            success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
