#!/usr/bin/env python3
"""
Weekly prediction runner for CFB Power Ratings Model.

Primary use case: Run Sunday AM after Saturday games.

Usage:
    python scripts/run_weekly.py                    # Default: edge-aware LSA mode
    python scripts/run_weekly.py --year 2025 --week 5
    python scripts/run_weekly.py --no-wait          # Skip data availability check
    python scripts/run_weekly.py --sharp            # High-conviction mode (5+ edge only)
    python scripts/run_weekly.py --min-edge 7       # Custom edge threshold
    python scripts/run_weekly.py --no-lsa           # Disable LSA, use Fixed only

Edge-Aware LSA Mode (DEFAULT for 2026):
    The prediction engine automatically selects Fixed or LSA based on timing and edge:
    - Opening lines (4+ days out): Fixed (56.5% at 5+ edge)
    - Closing lines, 5+ edge: LSA (55.1% at 5+ edge)
    - Closing lines, 3-5 edge: Fixed (52.9%)

    No flags needed - this is the default behavior.

    First-time setup (pre-compute LSA coefficients):
        python scripts/backtest.py --years 2022 2023 2024 2025 --learned-situ --lsa-adjust-turnovers
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

import requests

from config.settings import get_settings
from src.api.cfbd_client import CFBDClient, DataNotAvailableError, APIRateLimitError
from src.data.validators import DataValidator
from src.models.special_teams import SpecialTeamsModel
from src.models.finishing_drives import FinishingDrivesModel
from src.models.efficiency_foundation_model import EfficiencyFoundationModel
from src.models.preseason_priors import PreseasonPriors
from src.models.learned_situational import (
    LearnedSituationalModel,
    SituationalFeatures,
    FEATURE_NAMES,
)
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
from src.data.week_cache import WeekDataCache

import pandas as pd
import polars as pl

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
    parser.add_argument(
        "--use-delta-cache",
        action="store_true",
        help="Use week-level delta caching (only fetch current week from API, load historical from cache)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=3.0,
        help="Minimum edge (pts) to show as value play. Default: 3.0",
    )
    parser.add_argument(
        "--sharp",
        action="store_true",
        help="Sharp betting mode: only show 5+ edge picks (shortcut for --min-edge 5)",
    )
    # Edge-aware LSA is DEFAULT for 2026 production
    # Automatically uses LSA for 5+ edge closing bets, Fixed otherwise
    parser.add_argument(
        "--no-lsa",
        action="store_true",
        help="Disable edge-aware LSA mode. Use Fixed situational for all bets.",
    )
    parser.add_argument(
        "--lsa-threshold-days",
        type=int,
        default=4,
        help="Days before game to switch from fixed to LSA. Default: 4 (use fixed Sun-Tue, LSA Wed-Sat).",
    )
    # Legacy flags (kept for backward compatibility, but edge-aware is now default)
    parser.add_argument(
        "--learned-situ",
        action="store_true",
        help="(Legacy) Explicitly enable LSA - now default behavior.",
    )
    parser.add_argument(
        "--dual-spread",
        action="store_true",
        help="(Legacy) Enable dual-spread mode - now default behavior.",
    )
    return parser.parse_args()


def load_lsa_coefficients(year: int) -> dict[str, float] | None:
    """Load the most recent LSA coefficients for a given year.

    Looks for coefficients in data/learned_situ_coefficients/ and returns
    the highest week number available for the specified year.

    Args:
        year: Season year to load coefficients for

    Returns:
        Dictionary of {feature_name: coefficient} or None if not found
    """
    import json
    import re
    coef_dir = project_root / "data" / "learned_situ_coefficients"

    if not coef_dir.exists():
        logger.warning("LSA coefficients directory not found. Run backtest with --learned-situ first.")
        return None

    def extract_week_number(filepath):
        """Extract week number from filename for numeric sorting."""
        match = re.search(r'week(\d+)', filepath.name)
        return int(match.group(1)) if match else 0

    # Find all coefficient files for this year, sorted numerically by week
    pattern = f"lsa_{year}_week*.json"
    files = sorted(coef_dir.glob(pattern), key=extract_week_number)

    if not files:
        # Try previous year's final coefficients
        prev_year = year - 1
        pattern = f"lsa_{prev_year}_week*.json"
        files = sorted(coef_dir.glob(pattern), key=extract_week_number)
        if files:
            logger.info(f"Using {prev_year} coefficients (no {year} data yet)")

    if not files:
        logger.warning(f"No LSA coefficients found for {year} or {year-1}. Run backtest with --learned-situ first.")
        return None

    # Use the most recent (highest week number)
    latest_file = files[-1]
    logger.info(f"Loading LSA coefficients from {latest_file.name}")

    with open(latest_file) as f:
        data = json.load(f)

    return data.get("coefficients", None)


def apply_lsa_adjustment(
    prediction,
    lsa_coefficients: dict[str, float],
    situational: SituationalAdjuster,
    schedule_df: pd.DataFrame,
    rankings: dict,
    historical_rankings,
    week: int,
) -> float:
    """Compute LSA adjustment for a single prediction.

    Args:
        prediction: PredictedSpread object
        lsa_coefficients: Dict of {feature_name: coefficient}
        situational: SituationalAdjuster for computing raw factors
        schedule_df: Schedule DataFrame for situational lookup
        rankings: Current rankings dict
        historical_rankings: Historical rankings object
        week: Current week number

    Returns:
        LSA adjustment in points (positive = favors home)
    """
    # Get base spread to determine who is favorite
    home_is_favorite = prediction.spread > 0

    # Get situational factors
    home_factors, away_factors = situational.get_matchup_factors(
        home_team=prediction.home_team,
        away_team=prediction.away_team,
        current_week=week,
        schedule_df=schedule_df,
        rankings=rankings,
        home_is_favorite=home_is_favorite,
        historical_rankings=historical_rankings,
    )

    # Convert to feature vector
    features = SituationalFeatures.from_situational_factors(home_factors, away_factors)
    feature_array = features.to_array()

    # Compute learned adjustment
    adjustment = 0.0
    for i, name in enumerate(FEATURE_NAMES):
        adjustment += lsa_coefficients.get(name, 0.0) * feature_array[i]

    return adjustment


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

    Only returns games that haven't been played yet (no scores).
    Completed games are skipped to avoid predicting already-known outcomes.

    Args:
        client: API client
        year: Season year
        week: Week to predict

    Returns:
        List of game dictionaries (unplayed games only)
    """
    games = client.get_games(year, week)
    upcoming = []
    skipped = 0

    for game in games:
        # Skip completed games (already have scores)
        if game.home_points is not None and game.away_points is not None:
            skipped += 1
            continue

        upcoming.append({
            "id": game.id,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "neutral_site": game.neutral_site or False,
            "start_date": game.start_date,
        })

    if skipped > 0:
        logger.debug(f"Skipped {skipped} completed games for week {week}")
    logger.info(f"Found {len(upcoming)} upcoming games for week {week}")
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

    Fetches all regular season weeks (1-15). Needed for lookahead and sandwich
    spot detection, which requires visibility into future opponents.

    Args:
        client: API client
        year: Season year

    Returns:
        DataFrame with full schedule
    """
    all_games = []
    consecutive_empty = 0

    # Fetch all regular season weeks
    for week in range(1, 16):
        try:
            games = client.get_games(year, week)
            if not games:
                # Empty response = week doesn't exist yet (mid-season)
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    # Two consecutive empty weeks = reached end of scheduled games
                    break
                continue
            consecutive_empty = 0  # Reset on successful fetch
            for game in games:
                all_games.append({
                    "week": game.week,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "home_points": game.home_points,
                    "away_points": game.away_points,
                })
        except DataNotAvailableError:
            # Week doesn't exist yet - stop fetching
            break
        except Exception as e:
            # Log transient errors but continue fetching remaining weeks
            # (rate limits, timeouts should not truncate the schedule)
            logger.warning(f"Error fetching schedule week {week}: {e}")
            continue

    if all_games:
        max_week = max(g["week"] for g in all_games)
        logger.debug(f"Built schedule with {len(all_games)} games through week {max_week}")

    return pd.DataFrame(all_games)


PLAY_COLUMNS = [
    "game_id", "offense", "defense", "down", "distance",
    "yards_gained", "ppa", "play_type", "period",
    "home_score", "away_score", "yards_to_goal",
]

# Polars schema for play-by-play data (used for empty DataFrame creation)
PLAY_SCHEMA = {
    "game_id": pl.Int64,
    "offense": pl.Utf8,
    "defense": pl.Utf8,
    "down": pl.Int64,
    "distance": pl.Float64,
    "yards_gained": pl.Float64,
    "ppa": pl.Float64,
    "play_type": pl.Utf8,
    "period": pl.Int64,
    "home_score": pl.Int64,
    "away_score": pl.Int64,
    "yards_to_goal": pl.Float64,
}


def _fetch_week_plays_from_api(client: CFBDClient, year: int, w: int) -> pl.DataFrame:
    """Fetch a single week's plays from the API and return as Polars DataFrame."""
    plays_data = []
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
    return pl.DataFrame(plays_data, schema=PLAY_SCHEMA)


def _fetch_plays(
    client: CFBDClient, year: int, week: int, use_delta_cache: bool
) -> pl.DataFrame:
    """Fetch play-by-play data, using week-level cache when enabled.

    Returns a Polars DataFrame for efficient filtering before Pandas conversion.

    When use_delta_cache=True:
        - Loads weeks [1, week-2] from disk cache (cache hits)
        - Fetches ONLY week (week-1) from the CFBD API
        - Saves the newly fetched week to cache
    When use_delta_cache=False:
        - Fetches all weeks [1, week-1] from API (original behavior)
    """
    training_weeks = list(range(1, week))  # weeks 1 through week-1

    if not use_delta_cache:
        # Original behavior: fetch every week from API
        all_dfs = []
        for w in training_weeks:
            try:
                df = _fetch_week_plays_from_api(client, year, w)
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Error fetching plays for week {w}: {e}")
        if all_dfs:
            return pl.concat(all_dfs)
        return pl.DataFrame(schema=PLAY_SCHEMA)

    # --- Delta cache path ---
    cache = WeekDataCache()
    cached_weeks = set(cache.get_cached_weeks(year, "plays"))
    historical_weeks = training_weeks[:-1]  # weeks [1, week-2]
    fetch_week = training_weeks[-1] if training_weeks else None  # week-1

    # Load historical weeks from cache
    all_dfs: list[pl.DataFrame] = []
    missing_historical = []
    for w in historical_weeks:
        if w in cached_weeks:
            df = cache.load_week(year, w, "plays")
            if df is not None:
                all_dfs.append(df)
                continue
        missing_historical.append(w)

    # If any historical weeks are missing, fetch them from API and cache
    if missing_historical:
        logger.info(
            f"Delta cache: {len(missing_historical)} historical week(s) not cached, "
            f"fetching from API: {missing_historical}"
        )
        for w in missing_historical:
            try:
                df = _fetch_week_plays_from_api(client, year, w)
                cache.save_week(year, w, "plays", df)
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Error fetching plays for week {w}: {e}")
    else:
        if historical_weeks:
            logger.info(
                f"Delta cache: Loaded weeks 1-{historical_weeks[-1]} from cache "
                f"({sum(len(d) for d in all_dfs):,} plays)"
            )

    # Fetch the current week from API (always fresh)
    if fetch_week is not None:
        try:
            logger.info(f"Delta cache: Fetching week {fetch_week} from API...")
            df = _fetch_week_plays_from_api(client, year, fetch_week)
            cache.save_week(year, fetch_week, "plays", df)
            all_dfs.append(df)
            logger.info(f"Delta cache: Week {fetch_week} fetched and cached ({len(df):,} plays)")
        except Exception as e:
            logger.warning(f"Error fetching plays for week {fetch_week}: {e}")

    if all_dfs:
        return pl.concat(all_dfs)
    return pl.DataFrame(schema=PLAY_SCHEMA)


def run_predictions(
    year: int,
    week: int,
    wait_for_data: bool = True,
    send_notifications: bool = True,
    qb_out_teams: list[str] = None,
    generate_reports: bool = True,
    use_delta_cache: bool = False,
    min_edge: float = 3.0,
    use_learned_situ: bool = False,
    dual_spread: bool = False,
    lsa_threshold_days: int = 4,
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
        min_edge: Minimum edge (pts) to qualify as value play. Default 3.0.
                  Use 5.0 for high-conviction filtering (--sharp mode).
        use_learned_situ: Whether to use LSA (Learned Situational Adjustment).
        dual_spread: Output both fixed and LSA spreads with timing recommendation.
        lsa_threshold_days: Days before game to switch from fixed to LSA recommendation.

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

    # P0: Fail fast for invalid weeks (prevents infinite wait loop)
    # CFB season: weeks 0-15 regular, 16-17 postseason (bowls + CFP)
    MAX_CFB_WEEK = 18  # Week 17 is national championship, week 18+ doesn't exist
    if week > MAX_CFB_WEEK:
        raise ValueError(
            f"Week {week} exceeds maximum CFB week ({MAX_CFB_WEEK}). "
            f"CFB season ends at week 17 (national championship). "
            f"Did you mean a different week?"
        )

    # Initialize API client
    logger.info("Initializing API client...")
    client = CFBDClient()

    # === DATA FETCHING SECTION ===
    # Wrap API calls in narrow exception handler for network/availability errors.
    # Programming errors (TypeError, KeyError, etc.) propagate naturally.
    try:
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

        if not fbs_teams:
            raise ValueError(f"No FBS teams returned for {year}. Check API key and year.")
        logger.info(f"Loaded {len(fbs_teams)} FBS teams")

        # Fetch current season data
        logger.info("Fetching current season game data...")
        current_games_df = fetch_games_data(client, year, week - 1)

        if current_games_df.empty:
            raise ValueError(f"No game data found for {year} through week {week-1}")

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
        plays_pl = _fetch_plays(client, year, week, use_delta_cache)
        # Filter to FBS-only plays in Polars (more efficient than Pandas)
        # then convert to Pandas for sklearn compatibility
        pre_filter = plays_pl.height
        fbs_teams_list = list(fbs_teams)  # Convert set to list for Polars is_in()
        plays_df = (
            plays_pl
            .filter(
                pl.col("offense").is_in(fbs_teams_list)
                & pl.col("defense").is_in(fbs_teams_list)
            )
            .to_pandas()
        )
        logger.info(
            f"Loaded {len(plays_df)} FBS plays for EFM training "
            f"({pre_filter - len(plays_df)} FCS plays excluded)"
        )

        # Build EFM model
        logger.info("Fitting Efficiency Foundation Model...")
        efm = EfficiencyFoundationModel(
            ridge_alpha=settings.ridge_alpha,
        )
        efm.calculate_ratings(
            plays_df, current_games_df, max_week=week - 1, season=year,
            fbs_teams=fbs_teams,  # P0: Exclude FCS teams from normalization
        )

        # Get team ratings from EFM
        team_ratings = {
            team: efm.get_rating(team)
            for team in fbs_teams
            if team in efm.team_ratings
        }
        ratings_df = efm.get_ratings_df()

        # Calculate HFA
        logger.info("Calculating home field advantages...")
        hfa = HomeFieldAdvantage(global_offset=0.50)
        hfa.calculate_all_team_hfa(combined_for_hfa, team_ratings)

        # Special teams (simplified without detailed data)
        # Uses batch method for single-pass groupby instead of per-team filtering
        logger.info("Calculating special teams ratings...")
        special_teams = SpecialTeamsModel()
        special_teams.calculate_all_from_game_stats(set(team_ratings.keys()), current_games_df)

        # Finishing drives (simplified)
        # Uses batch method for single-pass groupby instead of per-team filtering
        logger.info("Calculating finishing drives ratings...")
        finishing = FinishingDrivesModel()
        finishing.calculate_all_from_game_stats(set(team_ratings.keys()), current_games_df)

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
        # Note: fcs_penalty_elite/standard use SpreadGenerator's defaults (18.0/32.0)
        # These are fallbacks when dynamic FCS estimator is unavailable.
        spread_gen = SpreadGenerator(
            ratings=team_ratings,
            special_teams=special_teams,
            finishing_drives=finishing,
            home_field=hfa,
            situational=situational,
            travel=travel,
            altitude=altitude,
            fbs_teams=fbs_teams,
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

    except (DataNotAvailableError, APIRateLimitError, requests.RequestException) as e:
        # Handle data availability and network errors gracefully
        logger.error(f"Data fetching failed: {e}")
        if notifier:
            notifier.notify_failure(f"Data fetching failed: {e}", year, week)
        return {"success": False, "error": str(e)}

    # === MODEL FITTING AND PREDICTION SECTION ===
    # No broad exception handling here - programming errors (TypeError, KeyError, etc.)
    # should propagate naturally so they surface as real bugs.

    # Generate predictions
    logger.info(f"Generating predictions for {len(upcoming_games)} games...")
    predictions = spread_gen.predict_week(
        games=upcoming_games,
        week=week,
        schedule_df=schedule_df,
        rankings=None,  # Current rankings (could also fetch AP/CFP here)
        historical_rankings=historical_rankings,
    )

    # Store fixed spreads before any LSA modification (for dual-spread mode)
    fixed_spreads = {pred.game_id: pred.spread for pred in predictions}

    # Dual-spread mode: compute BOTH fixed and LSA spreads
    lsa_spreads = {}
    lsa_adjustments = []  # Track adjustments for proper logging
    if dual_spread or use_learned_situ:
        lsa_coefficients = load_lsa_coefficients(year)
        if lsa_coefficients:
            logger.info(f"Computing LSA adjustments for {len(predictions)} predictions...")
            for pred in predictions:
                # Compute LSA adjustment
                lsa_adj = apply_lsa_adjustment(
                    prediction=pred,
                    lsa_coefficients=lsa_coefficients,
                    situational=situational,
                    schedule_df=schedule_df,
                    rankings=None,
                    historical_rankings=historical_rankings,
                    week=week,
                )
                lsa_adjustments.append(lsa_adj)

                # Calculate LSA spread: replace fixed situational with learned
                fixed_situ = pred.components.situational
                lsa_spread = pred.spread - fixed_situ + lsa_adj
                lsa_spreads[pred.game_id] = lsa_spread

                # If use_learned_situ (not dual), apply LSA to prediction
                if use_learned_situ and not dual_spread:
                    pred.spread = lsa_spread
                    pred.components.situational = lsa_adj

            # Log mean of actual LSA adjustments (not mean of final spreads)
            mean_lsa = sum(lsa_adjustments) / len(lsa_adjustments) if lsa_adjustments else 0.0
            if use_learned_situ and not dual_spread:
                logger.info(f"LSA adjustments applied (mean adjustment: {mean_lsa:+.2f} pts)")
            else:
                logger.info(f"LSA spreads computed for dual-spread mode (mean adjustment: {mean_lsa:+.2f} pts)")
        else:
            logger.warning("LSA coefficients not found - using fixed situational only")
            # Fill lsa_spreads with fixed values as fallback
            lsa_spreads = fixed_spreads.copy()

    predictions_df = spread_gen.predictions_to_dataframe(predictions)

    # Store dual-spread data for later (after Vegas comparison)
    # Edge-aware recommendation requires knowing the edge, which needs Vegas lines
    if dual_spread:
        predictions_df['jp_spread_fixed'] = predictions_df['game_id'].map(fixed_spreads)
        predictions_df['jp_spread_lsa'] = predictions_df['game_id'].map(lsa_spreads)

    # Vegas comparison
    logger.info(f"Fetching Vegas lines and comparing (min_edge={min_edge})...")
    vegas = VegasComparison(client=client, value_threshold=min_edge)
    vegas.fetch_lines(year, week)

    comparison_df = vegas.generate_comparison_df(predictions)
    value_plays = vegas.identify_value_plays(predictions)
    value_plays_df = vegas.value_plays_to_dataframe(value_plays)

    # Add edge-aware dual-spread recommendations (requires Vegas lines for edge calculation)
    if dual_spread:
        from datetime import datetime

        # Merge dual-spread columns into comparison_df
        dual_cols = predictions_df[['game_id', 'jp_spread_fixed', 'jp_spread_lsa']].copy()
        comparison_df = comparison_df.merge(dual_cols, on='game_id', how='left')

        # Add game dates from upcoming_games for timing calculation
        game_dates = {g['id']: g.get('start_date') for g in upcoming_games}
        comparison_df['start_date'] = comparison_df['game_id'].map(game_dates)

        # Calculate edge for both fixed and LSA spreads
        # Sign convention: edge = (-model_spread) - vegas_spread
        comparison_df['edge_fixed'] = (-comparison_df['jp_spread_fixed']) - comparison_df['vegas_spread']
        comparison_df['edge_lsa'] = (-comparison_df['jp_spread_lsa']) - comparison_df['vegas_spread']

        # Edge-aware recommendation logic:
        # - Opening line (4+ days): Always use Fixed (57.0% at 5+ edge)
        # - Closing line, 5+ edge: Use LSA (55.8% at 5+ edge)
        # - Closing line, 3-5 edge: Use Fixed (53.4% vs 52.5% LSA)
        today = datetime.now().date()

        def get_edge_aware_recommendation(row):
            try:
                # Determine days until game
                days_until = 99  # Default to opening if unknown
                if pd.notna(row.get('start_date')):
                    game_date = pd.to_datetime(row['start_date']).date()
                    days_until = (game_date - today).days

                # Opening line (4+ days): Always use Fixed
                if days_until >= lsa_threshold_days:
                    return 'fixed'

                # Closing line (<4 days): LSA only for 5+ edge
                edge_lsa = row.get('edge_lsa')
                if pd.notna(edge_lsa) and abs(edge_lsa) >= 5.0:
                    return 'lsa'
                else:
                    return 'fixed'  # 3-5 pt edge at closing: use fixed (53.4% > 52.5%)
            except Exception:
                return 'fixed'

        comparison_df['bet_timing_rec'] = comparison_df.apply(get_edge_aware_recommendation, axis=1)
        comparison_df['jp_spread_recommended'] = comparison_df.apply(
            lambda row: row['jp_spread_fixed'] if row['bet_timing_rec'] == 'fixed' else row['jp_spread_lsa'],
            axis=1
        )
        comparison_df['edge_recommended'] = comparison_df.apply(
            lambda row: row['edge_fixed'] if row['bet_timing_rec'] == 'fixed' else row['edge_lsa'],
            axis=1
        )

        # Log summary
        n_fixed = sum(comparison_df['bet_timing_rec'] == 'fixed')
        n_lsa = sum(comparison_df['bet_timing_rec'] == 'lsa')
        logger.info(f"Dual-spread mode: {n_fixed} games recommend fixed, {n_lsa} recommend LSA "
                   f"(threshold: {lsa_threshold_days} days, LSA only for 5+ edge at closing)")

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
        # Auto-detect week - predict current week's games
        try:
            client = CFBDClient()
            week = client.get_current_week(year)
        except Exception as e:
            logger.error(f"Could not auto-detect week: {e}")
            logger.info("Please specify --week argument")
            sys.exit(1)

    # Determine min_edge threshold
    min_edge = args.min_edge
    if args.sharp:
        min_edge = 5.0
        logger.info("Sharp mode enabled: filtering to 5+ edge picks only")

    # Edge-aware LSA mode (DEFAULT for 2026 production)
    # Uses LSA for 5+ edge closing bets, Fixed otherwise
    if args.no_lsa:
        use_learned_situ = False
        dual_spread = False
        logger.info("LSA disabled (--no-lsa): using Fixed situational for all bets")
    else:
        # Edge-aware mode is now the default
        use_learned_situ = True
        dual_spread = True
        logger.info(f"Edge-aware LSA mode (default): Fixed for opening, LSA for 5+ edge closing (threshold: {args.lsa_threshold_days} days)")

    logger.info(f"Running predictions for {year} Week {week} (min_edge={min_edge})")

    results = run_predictions(
        year=year,
        week=week,
        wait_for_data=not args.no_wait,
        send_notifications=not args.no_notify,
        qb_out_teams=args.qb_out,
        generate_reports=not args.no_reports,
        use_delta_cache=args.use_delta_cache,
        min_edge=min_edge,
        use_learned_situ=use_learned_situ,
        dual_spread=dual_spread,
        lsa_threshold_days=args.lsa_threshold_days,
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
