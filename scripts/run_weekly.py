#!/usr/bin/env python3
"""
Weekly prediction runner for CFB Power Ratings Model.

Primary use case: Run Sunday AM after Saturday games.

Usage:
    python scripts/run_weekly.py
    python scripts/run_weekly.py --year 2025 --week 5
    python scripts/run_weekly.py --no-wait  # Skip data availability check
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from src.api.cfbd_client import CFBDClient, DataNotAvailableError
from src.data.processors import DataProcessor
from src.data.validators import DataValidator
from src.models.special_teams import SpecialTeamsModel
from src.models.finishing_drives import FinishingDrivesModel
from src.models.efficiency_foundation_model import EfficiencyFoundationModel
from src.models.preseason_priors import PreseasonPriors
from src.adjustments.home_field import HomeFieldAdvantage
from src.adjustments.situational import SituationalAdjuster, HistoricalRankings
from src.adjustments.travel import TravelAdjuster
from src.adjustments.altitude import AltitudeAdjuster
from src.adjustments.qb_adjustment import QBInjuryAdjuster
from src.predictions.spread_generator import SpreadGenerator
from src.predictions.vegas_comparison import VegasComparison
from src.reports.excel_export import ExcelExporter
from src.reports.html_report import HTMLReporter
from src.notifications import Notifier

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "data" / "outputs" / "run.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CFB Power Ratings weekly predictions"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Season year (default: current year from settings)",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Week number (default: auto-detect current week)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Skip waiting for data availability",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip sending notifications",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--qb-out",
        nargs="+",
        default=[],
        metavar="TEAM",
        help="Teams whose starting QB is out (e.g., --qb-out 'Georgia' 'Texas')",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip Excel/HTML report generation (faster for testing/scripting)",
    )
    return parser.parse_args()


def fetch_games_data(
    client: CFBDClient,
    year: int,
    through_week: int,
) -> pd.DataFrame:
    """Fetch and combine all games data for the season.

    Args:
        client: API client
        year: Season year
        through_week: Include games through this week

    Returns:
        DataFrame with all game data
    """
    all_games = []

    for week in range(1, through_week + 1):
        try:
            games = client.get_games(year, week)
            for game in games:
                if game.home_points is None:
                    continue  # Skip unplayed games

                all_games.append({
                    "id": game.id,
                    "week": game.week,
                    "start_date": game.start_date,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "home_points": game.home_points,
                    "away_points": game.away_points,
                    "neutral_site": game.neutral_site,
                    "conference_game": game.conference_game,
                })
        except Exception as e:
            logger.warning(f"Error fetching week {week}: {e}")

    df = pd.DataFrame(all_games)
    logger.info(f"Fetched {len(df)} completed games through week {through_week}")
    return df


def fetch_upcoming_games(
    client: CFBDClient,
    year: int,
    week: int,
) -> list[dict]:
    """Fetch upcoming games for prediction.

    Args:
        client: API client
        year: Season year
        week: Week to predict

    Returns:
        List of game dictionaries
    """
    games = client.get_games(year, week)
    upcoming = []

    for game in games:
        # Include games without scores (not yet played)
        upcoming.append({
            "id": game.id,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "neutral_site": game.neutral_site or False,
            "start_date": game.start_date,
        })

    logger.info(f"Found {len(upcoming)} games for week {week}")
    return upcoming


def fetch_historical_data(
    client: CFBDClient,
    years: tuple,
) -> pd.DataFrame:
    """Fetch multi-year historical data for HFA and model training.

    Args:
        client: API client
        years: Tuple of years to include

    Returns:
        DataFrame with historical game data
    """
    all_games = []

    for year in years:
        # P3.9: Per-year progress at debug level for quiet runs
        logger.debug(f"Fetching {year} season data...")
        try:
            games = client.get_games(year)
            for game in games:
                if game.home_points is None:
                    continue

                all_games.append({
                    "season": year,
                    "week": game.week,
                    "start_date": game.start_date,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "home_points": game.home_points,
                    "away_points": game.away_points,
                    "neutral_site": game.neutral_site,
                })
        except Exception as e:
            logger.warning(f"Error fetching {year}: {e}")

    df = pd.DataFrame(all_games)
    logger.info(f"Fetched {len(df)} historical games across {len(years)} seasons")
    return df


def build_schedule_df(
    client: CFBDClient,
    year: int,
) -> pd.DataFrame:
    """Build full season schedule for situational analysis.

    Args:
        client: API client
        year: Season year

    Returns:
        DataFrame with full schedule
    """
    all_games = []

    # Fetch all regular season weeks
    for week in range(1, 16):
        try:
            games = client.get_games(year, week)
            for game in games:
                all_games.append({
                    "week": game.week,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "home_points": game.home_points,
                    "away_points": game.away_points,
                })
        except Exception:
            break  # Stop if week doesn't exist

    return pd.DataFrame(all_games)


def run_predictions(
    year: int,
    week: int,
    wait_for_data: bool = True,
    send_notifications: bool = True,
    qb_out_teams: list[str] = None,
    generate_reports: bool = True,
) -> dict:
    """Run the full prediction pipeline.

    Args:
        year: Season year
        week: Week to predict
        wait_for_data: Whether to wait for data availability
        send_notifications: Whether to send notifications
        qb_out_teams: List of teams whose starting QB is out
        generate_reports: Whether to generate Excel/HTML reports. Set False for
                         faster execution in testing/scripting. Default True.

    Returns:
        Dictionary with results summary
    """
    qb_out_teams = qb_out_teams or []
    settings = get_settings()
    notifier = Notifier() if send_notifications else None

    # Validate settings
    errors = settings.validate()
    if errors:
        for error in errors:
            logger.error(error)
        if notifier:
            notifier.notify_failure("\n".join(errors), year, week)
        return {"success": False, "errors": errors}

    # Ensure output directory exists
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize API client
        logger.info("Initializing API client...")
        client = CFBDClient()

        # Wait for data if requested
        if wait_for_data:
            logger.info(f"Checking data availability for {year} week {week}...")
            if not client.check_data_availability(year, week - 1):
                logger.info("Data not ready, waiting...")
                if not client.wait_for_data(year, week - 1, max_wait_hours=4):
                    raise DataNotAvailableError(
                        f"Data not available after waiting for {year} week {week-1}"
                    )

        # Fetch FBS teams for FCS detection (needed before EFM)
        logger.info("Fetching FBS teams...")
        fbs_teams_list = client.get_fbs_teams(year)
        fbs_teams = {t.school for t in fbs_teams_list}
        logger.info(f"Loaded {len(fbs_teams)} FBS teams")

        # Fetch current season data
        logger.info("Fetching current season game data...")
        current_games_df = fetch_games_data(client, year, week - 1)

        if current_games_df.empty:
            raise ValueError(f"No game data found for {year} through week {week-1}")

        # Process data
        logger.info("Processing game data...")
        processor = DataProcessor()
        processed_games = processor.process_games(
            current_games_df,
            apply_recency_weights=True,
        )

        # Fetch historical data for HFA (use current year + historical)
        logger.info("Fetching historical data for HFA calculation...")
        historical_years = [y for y in settings.historical_years if y < year]
        if historical_years:
            historical_df = fetch_historical_data(client, tuple(historical_years))
            combined_for_hfa = pd.concat([historical_df, current_games_df], ignore_index=True)
        else:
            combined_for_hfa = current_games_df

        # Fetch play-by-play data for EFM
        logger.info("Fetching play-by-play data for efficiency model...")
        plays_data = []
        for w in range(1, week):
            try:
                week_plays = client.get_plays(year, w)
                for play in week_plays:
                    plays_data.append({
                        "game_id": play.game_id,
                        "offense": play.offense,
                        "defense": play.defense,
                        "down": play.down,
                        "distance": play.distance,
                        "yards_gained": play.yards_gained,
                        "ppa": play.ppa,
                        "play_type": play.play_type,
                        "period": play.period,
                        "home_score": play.home_score,
                        "away_score": play.away_score,
                        "yards_to_goal": getattr(play, "yards_to_goal", None),
                    })
            except Exception as e:
                logger.warning(f"Error fetching plays for week {w}: {e}")

        plays_df = pd.DataFrame(plays_data)
        logger.info(f"Fetched {len(plays_df)} plays for EFM training")

        # Build EFM model
        logger.info("Fitting Efficiency Foundation Model...")
        efm = EfficiencyFoundationModel(
            ridge_alpha=settings.ridge_alpha if hasattr(settings, 'ridge_alpha') else 50.0,
        )
        efm.calculate_ratings(plays_df, current_games_df, max_week=week - 1, season=year)

        # Get team ratings from EFM
        team_ratings = {
            team: efm.get_rating(team)
            for team in fbs_teams
            if team in efm.team_ratings
        }
        ratings_df = efm.get_ratings_df()

        # Calculate HFA
        logger.info("Calculating home field advantages...")
        hfa = HomeFieldAdvantage()
        hfa.calculate_all_team_hfa(combined_for_hfa, team_ratings)

        # Special teams (simplified without detailed data)
        logger.info("Calculating special teams ratings...")
        special_teams = SpecialTeamsModel()
        for team in team_ratings.keys():
            special_teams.calculate_from_game_stats(team, processed_games)

        # Finishing drives (simplified)
        logger.info("Calculating finishing drives ratings...")
        finishing = FinishingDrivesModel()
        for team in team_ratings.keys():
            finishing.calculate_from_game_stats(team, processed_games)

        # Initialize adjusters
        situational = SituationalAdjuster()
        travel = TravelAdjuster()
        altitude = AltitudeAdjuster()

        # Initialize QB injury adjuster if any QBs flagged as out
        qb_adjuster = None
        if qb_out_teams:
            logger.info(f"Initializing QB injury adjuster for: {qb_out_teams}")
            qb_adjuster = QBInjuryAdjuster(
                api_key=settings.cfbd_api_key,
                year=year,
            )
            for team in qb_out_teams:
                qb_adjuster.flag_qb_out(team)
            qb_adjuster.print_depth_charts()

        # Build spread generator
        spread_gen = SpreadGenerator(
            ratings=team_ratings,
            special_teams=special_teams,
            finishing_drives=finishing,
            home_field=hfa,
            situational=situational,
            travel=travel,
            altitude=altitude,
            fbs_teams=fbs_teams,
            fcs_penalty_elite=settings.fcs_penalty_elite if hasattr(settings, 'fcs_penalty_elite') else 18.0,
            fcs_penalty_standard=settings.fcs_penalty_standard if hasattr(settings, 'fcs_penalty_standard') else 32.0,
            qb_adjuster=qb_adjuster,
        )

        # Fetch upcoming games
        logger.info(f"Fetching week {week} games for prediction...")
        upcoming_games = fetch_upcoming_games(client, year, week)

        if not upcoming_games:
            raise ValueError(f"No games found for {year} week {week}")

        # Build schedule for situational analysis
        schedule_df = build_schedule_df(client, year)

        # Load historical AP rankings for letdown spot detection
        historical_rankings = HistoricalRankings("AP Top 25")
        try:
            historical_rankings.load_from_api(client, year)
        except Exception as e:
            logger.warning(f"Could not load historical rankings: {e}")
            historical_rankings = None

        # Generate predictions
        logger.info(f"Generating predictions for {len(upcoming_games)} games...")
        predictions = spread_gen.predict_week(
            games=upcoming_games,
            week=week,
            schedule_df=schedule_df,
            rankings=None,  # Current rankings (could also fetch AP/CFP here)
            historical_rankings=historical_rankings,
        )

        predictions_df = spread_gen.predictions_to_dataframe(predictions)

        # Vegas comparison
        logger.info("Fetching Vegas lines and comparing...")
        vegas = VegasComparison(client=client)
        vegas.fetch_lines(year, week)

        comparison_df = vegas.generate_comparison_df(predictions)
        value_plays = vegas.identify_value_plays(predictions)
        value_plays_df = vegas.value_plays_to_dataframe(value_plays)

        # P3.7: Generate reports (skip with --no-reports for faster testing)
        excel_path = None
        html_path = None

        if generate_reports:
            logger.info("Generating reports...")

            # Excel report
            excel_exporter = ExcelExporter()
            excel_path = excel_exporter.export(
                predictions_df=comparison_df,
                value_plays_df=value_plays_df,
                ratings_df=ratings_df,
                year=year,
                week=week,
            )

            # HTML report
            html_reporter = HTMLReporter()
            html_path = html_reporter.generate(
                predictions_df=comparison_df,
                value_plays_df=value_plays_df,
                ratings_df=ratings_df,
                year=year,
                week=week,
            )

            logger.info(f"Excel report: {excel_path}")
            logger.info(f"HTML report: {html_path}")
        else:
            logger.info("Skipping report generation (--no-reports)")

        # Send success notifications
        if notifier:
            notifier.notify_success(
                year=year,
                week=week,
                games_predicted=len(predictions),
                value_plays=len(value_plays),
                report_path=excel_path,
            )

        logger.info("Prediction run complete!")
        logger.info(f"Games predicted: {len(predictions)}")
        logger.info(f"Value plays: {len(value_plays)}")

        return {
            "success": True,
            "year": year,
            "week": week,
            "games_predicted": len(predictions),
            "value_plays": len(value_plays),
            "excel_path": str(excel_path) if excel_path else None,
            "html_path": str(html_path) if html_path else None,
        }

    except Exception as e:
        logger.exception(f"Prediction run failed: {e}")
        if notifier:
            notifier.notify_failure(str(e), year, week)
        return {
            "success": False,
            "error": str(e),
        }


def main():
    """Main entry point."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    settings = get_settings()

    # Determine year and week
    year = args.year or settings.current_year

    if args.week:
        week = args.week
    else:
        # Auto-detect week - predict next week's games
        try:
            client = CFBDClient()
            current_week = client.get_current_week(year)
            week = current_week + 1  # Predict upcoming week
        except Exception as e:
            logger.error(f"Could not auto-detect week: {e}")
            logger.info("Please specify --week argument")
            sys.exit(1)

    logger.info(f"Running predictions for {year} Week {week}")

    results = run_predictions(
        year=year,
        week=week,
        wait_for_data=not args.no_wait,
        send_notifications=not args.no_notify,
        qb_out_teams=args.qb_out,
        generate_reports=not args.no_reports,
    )

    if results["success"]:
        print(f"\nPredictions complete for {year} Week {week}")
        print(f"Games: {results['games_predicted']}")
        print(f"Value plays: {results['value_plays']}")
        if results.get("excel_path") and results.get("html_path"):
            print(f"\nReports saved to:")
            print(f"  Excel: {results['excel_path']}")
            print(f"  HTML: {results['html_path']}")
    else:
        print(f"\nPrediction run failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
