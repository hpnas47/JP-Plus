"""CLI entry point for preseason win total projections.

Usage:
    python3 -m src.win_totals.run_win_totals train --start-year 2015 --end-year 2025
    python3 -m src.win_totals.run_win_totals predict --year 2026 --book-lines data/win_totals/2026_book_lines.csv
    python3 -m src.win_totals.run_win_totals backtest --start-year 2020 --end-year 2025
    python3 -m src.win_totals.run_win_totals calibrate --start-year 2015 --end-year 2025

Win counting convention:
    Regular season only. CCG, bowls, and CFP games are EXCLUDED.
    Safeguard: season_type='regular' AND week <= 15.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.teams import normalize_team_name
from src.api.cfbd_client import CFBDClient
from src.win_totals.edge import (
    BookLine,
    compute_leakage_contribution,
    evaluate_all,
    generate_report,
)
from src.win_totals.features import PreseasonFeatureBuilder, FEATURE_METADATA
from src.win_totals.model import PreseasonModel
from src.win_totals.schedule import (
    NAIVE_FALLBACK_REGRESSION,
    ScheduledGame,
    calibrate_tau,
    calibrate_win_probability,
    project_season,
)

logger = logging.getLogger(__name__)

DEFAULT_HFA = 3.0
DEFAULT_N_SIMS = 50000
# Max week for regular season games (excludes CCG, bowls, CFP)
MAX_REGULAR_SEASON_WEEK = 15
# Default SP+ rating for FCS opponents (~90-95% FBS win rate historically).
# The main JP+ pipeline has a dynamic FCS estimator (src/models/fcs_strength.py)
# that uses Bayesian shrinkage on in-season game margins, but it requires
# completed game data (Polars DataFrame + walk-forward week). For preseason
# win totals where no games have been played, a static default is appropriate.
# The -17.0 value is consistent with the estimator's baseline_margin of -28
# (game-level margin including noise) mapped to a persistent team quality scale.
FCS_DEFAULT_RATING = -17.0


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Preseason Win Total Projection System'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_p = subparsers.add_parser('train', help='Walk-forward validation')
    train_p.add_argument('--start-year', type=int, default=2015)
    train_p.add_argument('--end-year', type=int, default=2025)
    train_p.add_argument('--exclude-covid', action='store_true', default=True)
    train_p.add_argument('--verbose', '-v', action='store_true')

    pred_p = subparsers.add_parser('predict', help='Generate predictions for a year')
    pred_p.add_argument('--year', type=int, required=True)
    pred_p.add_argument('--book-lines', type=str, help='CSV path for book lines')
    pred_p.add_argument('--train-start', type=int, default=2015)
    pred_p.add_argument('--n-sims', type=int, default=DEFAULT_N_SIMS)
    pred_p.add_argument('--min-ev', type=float, default=0.02)
    pred_p.add_argument('--output-csv', type=str)
    pred_p.add_argument('--verbose', '-v', action='store_true')

    bt_p = subparsers.add_parser('backtest', help='Historical backtest with book lines')
    bt_p.add_argument('--start-year', type=int, default=2020)
    bt_p.add_argument('--end-year', type=int, default=2025)
    bt_p.add_argument('--train-start', type=int, default=2015)
    bt_p.add_argument('--min-ev', type=float, default=0.02)
    bt_p.add_argument('--verbose', '-v', action='store_true')

    cal_p = subparsers.add_parser('calibrate', help='Calibrate win probability + tau')
    cal_p.add_argument('--start-year', type=int, default=2015)
    cal_p.add_argument('--end-year', type=int, default=2025)
    cal_p.add_argument('--verbose', '-v', action='store_true')

    return parser.parse_args(argv)


def load_book_lines(path: str) -> list[BookLine]:
    """Load book lines from CSV.

    Expected columns: team, year, line, over_odds, under_odds
    """
    lines = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lines.append(BookLine(
                team=normalize_team_name(row['team']),
                year=int(row['year']),
                line=float(row['line']),
                over_odds=int(row.get('over_odds', -110)),
                under_odds=int(row.get('under_odds', -110)),
            ))
    return lines


def cmd_train(args):
    """Walk-forward validation of the preseason model."""
    exclude = {2020} if args.exclude_covid else set()

    builder = PreseasonFeatureBuilder()
    dataset = builder.build_training_dataset(
        args.start_year, args.end_year, exclude_years=exclude
    )

    model = PreseasonModel()
    result = model.validate(dataset)

    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 70)

    for fold in result.folds:
        print(f"  {fold.year}: MAE={fold.mae:.3f}  RMSE={fold.rmse:.3f}  "
              f"fold_alpha={fold.fold_alpha:.0f}  n={fold.n_teams}")

    print(f"\n  Production MAE (honest):  {result.production_mae:.3f}")
    print(f"  Production RMSE (honest): {result.production_rmse:.3f}")
    print(f"  Per-fold optimized MAE (diagnostic only): {result.per_fold_optimized_mae:.3f}")
    print(f"  Tau:          {result.tau:.3f}")
    print(f"  Production alpha: {result.production_alpha:.1f}")

    # Alpha MAE table
    print("\n  Alpha sweep (avg MAE across folds):")
    for alpha, mae in sorted(result.alpha_mae_table.items()):
        marker = " <-- SELECTED" if alpha == result.production_alpha else ""
        print(f"    alpha={alpha:>6.1f}: MAE={mae:.3f}{marker}")

    # Baselines
    naive = PreseasonModel.naive_baseline(dataset)
    print(f"\n  Naive baseline (in-sample, no fit): MAE={naive:.3f}")

    # Feature importance
    fi = result.feature_importance()
    print("\nFeature Importance (avg |coef|):")
    for _, row in fi.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['avg_abs_coef']:>8.3f}  "
              f"(raw: {row['avg_coef']:>+7.3f} ± {row['std_coef']:.3f})")

    # Feature status summary
    from src.win_totals.features import feature_status_counts
    counts = feature_status_counts()
    print(f"\nFeature leakage audit: {counts}")


def cmd_predict(args):
    """Generate predictions and bet recommendations for a year."""
    builder = PreseasonFeatureBuilder()
    client = builder.client

    dataset = builder.build_training_dataset(
        args.train_start, args.year - 1, exclude_years={2020}
    )

    model = PreseasonModel()
    # validate() internally retrains with production_alpha on full dataset
    val_result = model.validate(dataset)

    features = builder.build_features(args.year)
    predictions = model.predict(features)
    features['predicted_sp'] = predictions

    games = client.get_games(year=args.year, season_type='regular')
    fbs_teams = builder._get_fbs_teams(args.year)
    pred_by_team = dict(zip(features['team'], predictions))

    rng = np.random.default_rng(seed=42)
    tau = val_result.tau

    # Calibrate win probability from prior-folds-only OOF predictions
    cal_spreads, cal_outcomes, fb_spreads, fb_outcomes, fb_years, primary_years = (
        _build_calibration_data_prior_folds(
            val_result, client, builder, max_year=args.year - 1
        )
    )
    calibration = calibrate_win_probability(
        cal_spreads, cal_outcomes,
        fallback_spreads=fb_spreads,
        fallback_outcomes=fb_outcomes,
        fallback_years=fb_years,
        primary_years=primary_years,
    )

    distributions = []
    for team in sorted(fbs_teams):
        if team not in pred_by_team:
            continue
        schedule = _build_schedule(team, games, pred_by_team, fbs_teams)
        if not schedule:
            continue
        dist = project_season(
            team=team, year=args.year, team_rating=pred_by_team[team],
            schedule=schedule, tau=tau, calibration=calibration,
            n_sims=args.n_sims, rng=rng,
        )
        distributions.append(dist)

    print(f"\nGenerated distributions for {len(distributions)} teams")

    distributions.sort(key=lambda d: d.expected_wins, reverse=True)
    print(f"\n{'Rank':>4}  {'Team':<25} {'SP+':>6} {'E[W]':>5}")
    print("-" * 45)
    for i, d in enumerate(distributions[:25], 1):
        print(f"{i:>4}  {d.team:<25} {d.predicted_rating:>+6.1f} {d.expected_wins:>5.1f}")

    # Save all predictions to CSV artifact for fast display script
    pred_csv_path = Path(f"data/win_totals/predictions_{args.year}.csv")
    pred_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, d in enumerate(distributions, 1):
        rows.append({
            'rank': i, 'team': d.team, 'year': d.year,
            'sp_plus': round(d.predicted_rating, 1),
            'expected_wins': round(d.expected_wins, 1),
            'n_games': d.n_games,
            'win_probs': ','.join(f'{p:.6f}' for p in d.win_probs),
        })
    pd.DataFrame(rows).to_csv(pred_csv_path, index=False)
    print(f"\nSaved {len(rows)} predictions to {pred_csv_path}")

    if args.book_lines:
        book_lines = load_book_lines(args.book_lines)
        leakage_pcts = _compute_leakage_pcts(model, features)
        recs = evaluate_all(
            distributions, book_lines, min_ev=args.min_ev,
            leakage_pcts=leakage_pcts,
        )
        report = generate_report(recs, csv_path=args.output_csv)
        print(f"\n{'=' * 70}")
        print("BET RECOMMENDATIONS")
        print(f"{'=' * 70}")
        print(report)


def cmd_backtest(args):
    """Historical backtest using saved book lines."""
    builder = PreseasonFeatureBuilder()

    total_bets = 0
    total_wins = 0
    total_pushes = 0
    total_payout = 0.0  # sum of payouts for ROI calculation

    for year in range(args.start_year, args.end_year + 1):
        lines_path = Path(f"data/win_totals/{year}_book_lines.csv")
        if not lines_path.exists():
            logger.warning(f"No book lines for {year}, skipping")
            continue

        book_lines = load_book_lines(str(lines_path))
        # Build lookup for per-bet odds
        book_line_lookup = {bl.team: bl for bl in book_lines}

        dataset = builder.build_training_dataset(
            args.train_start, year - 1, exclude_years={2020}
        )
        model = PreseasonModel()
        # validate() internally retrains with production_alpha on full dataset
        val_result = model.validate(dataset)

        features = builder.build_features(year)
        predictions = model.predict(features)
        pred_by_team = dict(zip(features['team'], predictions))

        actual_wins = _get_actual_wins(year, builder.client)

        games = builder.client.get_games(year=year, season_type='regular')
        fbs_teams = builder._get_fbs_teams(year)
        rng = np.random.default_rng(seed=42)
        tau = val_result.tau

        # Build calibration for this backtest year (matching predict path)
        cal_spreads, cal_outcomes, fb_spreads, fb_outcomes, fb_years, primary_years = (
            _build_calibration_data_prior_folds(
                val_result, builder.client, builder, max_year=year
            )
        )
        calibration = calibrate_win_probability(
            cal_spreads, cal_outcomes,
            fallback_spreads=fb_spreads,
            fallback_outcomes=fb_outcomes,
            fallback_years=fb_years,
            primary_years=primary_years,
        )

        distributions = []
        for team in sorted(fbs_teams):
            if team not in pred_by_team:
                continue
            schedule = _build_schedule(team, games, pred_by_team, fbs_teams)
            if not schedule:
                continue
            dist = project_season(
                team=team, year=year, team_rating=pred_by_team[team],
                schedule=schedule, tau=tau, calibration=calibration,
                n_sims=DEFAULT_N_SIMS, rng=rng,
            )
            distributions.append(dist)

        leakage_pcts = _compute_leakage_pcts(model, features)
        recs = evaluate_all(
            distributions, book_lines, min_ev=args.min_ev,
            leakage_pcts=leakage_pcts,
        )

        year_wins = 0
        year_bets = 0
        year_pushes = 0
        year_payout = 0.0
        for rec in recs:
            actual = actual_wins.get(rec.team)
            if actual is None:
                continue
            year_bets += 1
            won = False
            pushed = False
            if rec.side == "Over" and actual > rec.line:
                won = True
            elif rec.side == "Under" and actual < rec.line:
                won = True
            elif actual == rec.line and rec.line == int(rec.line):
                pushed = True

            if won:
                year_wins += 1
                # Compute payout from actual odds
                odds = rec.odds
                if odds < 0:
                    year_payout += 100.0 / abs(odds)
                else:
                    year_payout += odds / 100.0
            elif pushed:
                year_pushes += 1
                # Push = refund, no payout change
            else:
                year_payout -= 1.0

        total_bets += year_bets
        total_wins += year_wins
        total_pushes += year_pushes
        total_payout += year_payout

        decided = year_bets - year_pushes
        win_pct = year_wins / decided * 100 if decided > 0 else 0
        print(f"{year}: {year_wins}/{decided} ({win_pct:.1f}%), "
              f"{year_pushes} pushes, {len(recs)} recs, {len(distributions)} teams")

    if total_bets > 0:
        decided = total_bets - total_pushes
        overall_pct = total_wins / decided * 100 if decided > 0 else 0
        print(f"\nOverall: {total_wins}/{decided} ({overall_pct:.1f}%), {total_pushes} pushes")
        if decided > 0:
            roi = total_payout / decided * 100
            print(f"ROI (actual odds): {roi:+.1f}%")


def cmd_calibrate(args):
    """Calibrate win probability and tau parameters."""
    builder = PreseasonFeatureBuilder()
    dataset = builder.build_training_dataset(
        args.start_year, args.end_year, exclude_years={2020}
    )

    model = PreseasonModel()
    result = model.validate(dataset)

    print(f"\nTau (team uncertainty): {result.tau:.3f}")
    print(f"Walk-forward MAE (production): {result.production_mae:.3f}")
    print(f"Walk-forward RMSE (production): {result.production_rmse:.3f}")
    print(f"Production alpha: {result.production_alpha:.1f}")

    cal_spreads, cal_outcomes, fb_spreads, fb_outcomes, fb_years, primary_years = (
        _build_calibration_data_prior_folds(
            result, builder.client, builder, max_year=args.end_year
        )
    )

    if len(cal_spreads) > 0:
        cal = calibrate_win_probability(
            cal_spreads, cal_outcomes,
            fallback_spreads=fb_spreads,
            fallback_outcomes=fb_outcomes,
            fallback_years=fb_years,
            primary_years=primary_years,
        )
        print(f"\nWin Probability Calibration:")
        print(f"  Source:        {cal.calibration_source}")
        print(f"  Intercept:     {cal.intercept:.4f}")
        print(f"  Slope:         {cal.slope:.4f}")
        print(f"  Implied sigma: {cal.implied_sigma:.2f}")
        print(f"  N primary:     {cal.n_games_primary}")
        print(f"  N fallback:    {cal.n_games_fallback}")
        print(f"  Years:         {cal.years}")
        if cal.used_fallback_years:
            print(f"  Fallback years: {cal.used_fallback_years}")


def _compute_leakage_pcts(
    model: PreseasonModel,
    features: pd.DataFrame,
) -> dict[str, float]:
    """Compute per-team leakage contribution percentage.

    Uses the fitted model's coefficients and scaler to compute standardized
    feature contributions, then measures what fraction of |signal| comes
    from LEAKAGE_RISK features.
    """
    if model.model is None or model.scaler is None:
        return {}

    feature_cols = model.feature_names
    X = features[feature_cols].values.astype(np.float64)
    # Impute NaN with training column means (consistent with model.py)
    col_means = getattr(model, '_col_means', None)
    if col_means is not None:
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_means[j]
    else:
        X = np.nan_to_num(X, nan=0.0)
    X_scaled = model.scaler.transform(X)
    coefficients = model.model.coef_

    result = {}
    for i, team in enumerate(features['team'].values):
        pct = compute_leakage_contribution(
            coefficients, X_scaled[i], feature_cols, FEATURE_METADATA
        )
        result[str(team)] = pct
    return result


def _build_schedule(
    team: str,
    games: list,
    pred_by_team: dict[str, float],
    fbs_teams: set[str],
) -> list[ScheduledGame]:
    """Build a team's regular-season schedule from game objects.

    Safeguard: Only includes games with week <= MAX_REGULAR_SEASON_WEEK
    to exclude CCG/bowls even if API misclassifies them.
    """
    schedule = []
    for g in games:
        # Week safeguard: exclude CCG and postseason
        week = getattr(g, 'week', None)
        if week is not None and week > MAX_REGULAR_SEASON_WEEK:
            continue

        home = normalize_team_name(g.home_team)
        away = normalize_team_name(g.away_team)

        if home == team:
            opp = away
            is_home = True
        elif away == team:
            opp = home
            is_home = False
        else:
            continue

        opp_rating = pred_by_team.get(opp, FCS_DEFAULT_RATING if opp not in fbs_teams else 0.0)
        is_neutral = bool(getattr(g, 'neutral_site', False))

        schedule.append(ScheduledGame(
            opponent=opp,
            is_home=is_home,
            is_neutral=is_neutral,
            opponent_rating=opp_rating,
        ))

    return schedule


def _build_calibration_data_prior_folds(
    val_result,
    client: CFBDClient,
    builder: PreseasonFeatureBuilder,
    max_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, list[int] | None, list[int]]:
    """Build calibration data using ONLY prior folds (years < max_year).

    For predicting year Y, calibration uses OOF predictions from
    years start..Y-1 ONLY (not including Y itself).

    Uses production_predictions (consistent production_alpha across all folds)
    to ensure calibration spread scale matches production model.

    If insufficient primary data (< 1500 games), supplements with
    earlier years using naive baseline ratings regressed 30% toward mean.

    Returns:
        (primary_spreads, primary_outcomes,
         fallback_spreads, fallback_outcomes, fallback_years, primary_years)
    """
    spreads_list = []
    outcomes_list = []

    # Use production_predictions for consistent alpha across folds
    all_preds = val_result.production_predictions
    # Only use predictions from years strictly before max_year
    prior_preds = all_preds[all_preds['year'] < max_year]

    for year in sorted(prior_preds['year'].unique()):
        year_preds = prior_preds[prior_preds['year'] == year]
        pred_by_team = dict(zip(year_preds['team'], year_preds['predicted_sp']))

        try:
            games = client.get_games(year=int(year), season_type='regular')
        except Exception as e:
            logger.warning(f"Failed to get games for {year}: {e}")
            continue

        fbs_teams = builder._get_fbs_teams(int(year))

        for g in games:
            if g.home_points is None or g.away_points is None:
                continue
            # Week safeguard
            week = getattr(g, 'week', None)
            if week is not None and week > MAX_REGULAR_SEASON_WEEK:
                continue

            home = normalize_team_name(g.home_team)
            away = normalize_team_name(g.away_team)

            if home not in pred_by_team or away not in pred_by_team:
                continue
            if home not in fbs_teams or away not in fbs_teams:
                continue

            # Skip tied games (shouldn't happen post-1996, but guard against bad data)
            if g.home_points == g.away_points:
                logger.warning(f"Skipping tied game {home} vs {away} ({year} week {week})")
                continue

            spread = pred_by_team[home] - pred_by_team[away] + DEFAULT_HFA
            if getattr(g, 'neutral_site', False):
                spread = pred_by_team[home] - pred_by_team[away]

            outcome = 1.0 if g.home_points > g.away_points else 0.0

            spreads_list.append(spread)
            outcomes_list.append(outcome)

    primary_spreads = np.array(spreads_list) if spreads_list else np.array([])
    primary_outcomes = np.array(outcomes_list) if outcomes_list else np.array([])

    # Build fallback data if needed (naive baseline regressed 30% toward mean)
    fb_spreads_list = []
    fb_outcomes_list = []
    fb_years = []

    if len(primary_spreads) < 1500:
        # Get years that are NOT in val_result (too early for walk-forward)
        oof_years = set(prior_preds['year'].unique())

        # Try to find earlier years not covered by OOF
        earliest_oof = min(oof_years) if oof_years else max_year
        for year in range(earliest_oof - 5, earliest_oof):
            if year < 2005:
                continue
            try:
                prev_count = len(fb_spreads_list)
                sp_prior = {}
                raw = client.get_sp_ratings(year)
                for r in raw:
                    team = normalize_team_name(r.team)
                    rating = getattr(r, 'rating', None) or 0.0
                    # Regress 30% toward mean (mean SP+ ~ 0)
                    sp_prior[team] = rating * (1.0 - NAIVE_FALLBACK_REGRESSION)

                games = client.get_games(year=int(year), season_type='regular')
                fbs_teams = builder._get_fbs_teams(int(year))

                for g in games:
                    if g.home_points is None or g.away_points is None:
                        continue
                    week = getattr(g, 'week', None)
                    if week is not None and week > MAX_REGULAR_SEASON_WEEK:
                        continue

                    home = normalize_team_name(g.home_team)
                    away = normalize_team_name(g.away_team)

                    if home not in sp_prior or away not in sp_prior:
                        continue
                    if home not in fbs_teams or away not in fbs_teams:
                        continue

                    # Skip tied games
                    if g.home_points == g.away_points:
                        continue

                    spread = sp_prior[home] - sp_prior[away] + DEFAULT_HFA
                    if getattr(g, 'neutral_site', False):
                        spread = sp_prior[home] - sp_prior[away]

                    outcome = 1.0 if g.home_points > g.away_points else 0.0
                    fb_spreads_list.append(spread)
                    fb_outcomes_list.append(outcome)

                new_games = len(fb_spreads_list) - prev_count
                if new_games > 0:
                    fb_years.append(year)
                    logger.info(f"Fallback calibration: {year} contributed "
                                f"{new_games} games (naive baseline, 30% regressed)")

            except Exception as e:
                logger.warning(f"Failed to build fallback for {year}: {e}")
                continue

    fallback_spreads = np.array(fb_spreads_list) if fb_spreads_list else None
    fallback_outcomes = np.array(fb_outcomes_list) if fb_outcomes_list else None

    primary_years = sorted(int(y) for y in prior_preds['year'].unique())
    return primary_spreads, primary_outcomes, fallback_spreads, fallback_outcomes, fb_years or None, primary_years


def _get_actual_wins(year: int, client: CFBDClient) -> dict[str, int]:
    """Get actual regular season win counts.

    Regular season only: season_type='regular' AND week <= 15.
    CCG, bowls, and CFP games are EXCLUDED.
    """
    games = client.get_games(year=year, season_type='regular')
    wins: dict[str, int] = {}

    for g in games:
        if g.home_points is None or g.away_points is None:
            continue
        # Week safeguard
        week = getattr(g, 'week', None)
        if week is not None and week > MAX_REGULAR_SEASON_WEEK:
            continue

        home = normalize_team_name(g.home_team)
        away = normalize_team_name(g.away_team)

        wins.setdefault(home, 0)
        wins.setdefault(away, 0)

        if g.home_points > g.away_points:
            wins[home] += 1
        elif g.away_points > g.home_points:
            wins[away] += 1

    return wins


def main(argv=None):
    args = parse_args(argv)
    setup_logging(getattr(args, 'verbose', False))

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'backtest':
        cmd_backtest(args)
    elif args.command == 'calibrate':
        cmd_calibrate(args)


if __name__ == '__main__':
    main()
