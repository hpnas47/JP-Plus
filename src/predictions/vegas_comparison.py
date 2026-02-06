"""Vegas line comparison and value play identification."""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config.settings import get_settings
from src.api.cfbd_client import CFBDClient
from src.predictions.spread_generator import PredictedSpread

logger = logging.getLogger(__name__)


@dataclass
class VegasLine:
    """Container for Vegas betting line data."""

    game_id: int
    home_team: str
    away_team: str
    spread: float  # From home team perspective (negative = home favored)
    spread_open: Optional[float] = None
    over_under: Optional[float] = None
    provider: str = "consensus"


@dataclass
class ValuePlay:
    """Container for identified value play."""

    home_team: str
    away_team: str
    model_spread: float
    vegas_spread: float
    edge: float  # Model vs Vegas in Vegas convention (negative = model likes home more)
    side: str  # "HOME" or "AWAY"
    confidence: str
    analysis: str
    game_id: Optional[int] = None  # P0.2: CFBD game_id for reliable joins


class VegasComparison:
    """
    Compare model predictions to Vegas lines and identify value plays.

    Value plays are games where model spread differs from Vegas by a
    significant margin (default: 3+ points).
    """

    def __init__(
        self,
        client: Optional[CFBDClient] = None,
        value_threshold: Optional[float] = None,
        provider: Optional[str] = None,
    ):
        """Initialize Vegas comparison.

        Args:
            client: CFBD API client
            value_threshold: Minimum edge to flag as value play
            provider: Vegas line provider (default: consensus)
        """
        settings = get_settings()

        self.client = client
        self.value_threshold = (
            value_threshold
            if value_threshold is not None
            else settings.value_threshold
        )
        self.provider = provider or settings.vegas_provider

        # Lines indexed by game_id (preferred) and by (home_team, away_team) (fallback)
        self.lines_by_id: dict[int, VegasLine] = {}
        self.lines: dict[tuple[str, str], VegasLine] = {}  # Legacy fallback

    def fetch_lines(
        self,
        year: int,
        week: int,
        season_type: str = "regular",
    ) -> list[VegasLine]:
        """Fetch Vegas lines for a week.

        Args:
            year: Season year
            week: Week number
            season_type: 'regular' or 'postseason'

        Returns:
            List of VegasLine objects
        """
        if self.client is None:
            logger.warning("No API client provided, cannot fetch Vegas lines")
            return []

        raw_lines = self.client.get_betting_lines(year, week, season_type)
        vegas_lines = []

        for game in raw_lines:
            if not game.lines:
                continue

            # Find consensus or specified provider line
            provider_line = None
            for line in game.lines:
                if line.provider and line.provider.lower() == self.provider.lower():
                    provider_line = line
                    break

            # Fall back to first available line
            if provider_line is None and game.lines:
                provider_line = game.lines[0]

            if provider_line is None:
                continue

            # CFBD spread is already from home team perspective
            # (negative = home favored, positive = away favored)
            spread = provider_line.spread
            if spread is None:
                continue

            vl = VegasLine(
                game_id=game.id,
                home_team=game.home_team,
                away_team=game.away_team,
                spread=spread,
                spread_open=provider_line.spread_open,
                over_under=provider_line.over_under,
                provider=provider_line.provider or self.provider,
            )

            vegas_lines.append(vl)
            self.lines_by_id[vl.game_id] = vl  # Primary: by game_id
            self.lines[(game.home_team, game.away_team)] = vl  # Fallback: by team names

        logger.info(f"Fetched {len(vegas_lines)} Vegas lines for {year} week {week}")
        return vegas_lines

    def get_line_by_id(self, game_id: int) -> Optional[VegasLine]:
        """Get Vegas line by game_id (preferred method).

        Args:
            game_id: CFBD game ID

        Returns:
            VegasLine or None if not found
        """
        return self.lines_by_id.get(game_id)

    def get_line(self, home_team: str, away_team: str, game_id: Optional[int] = None) -> Optional[VegasLine]:
        """Get Vegas line for a specific matchup.

        Prefers game_id matching if provided, falls back to team name matching.

        Args:
            home_team: Home team name
            away_team: Away team name
            game_id: Optional game_id for reliable matching

        Returns:
            VegasLine or None if not found
        """
        # Prefer game_id matching (reliable)
        if game_id is not None and game_id in self.lines_by_id:
            return self.lines_by_id[game_id]

        # Fall back to team name matching (legacy)
        return self.lines.get((home_team, away_team))

    def compare_prediction(
        self,
        prediction: PredictedSpread,
    ) -> Optional[dict]:
        """Compare a single prediction to Vegas line.

        P0.1 FIX: Uses game_id for reliable matching when available,
        falls back to team name matching for backward compatibility.

        Args:
            prediction: Model prediction

        Returns:
            Comparison dict or None if no Vegas line available
        """
        # P0.1: Prefer game_id matching (reliable), fallback to team names
        vegas = self.get_line(
            prediction.home_team,
            prediction.away_team,
            game_id=prediction.game_id if hasattr(prediction, 'game_id') else None
        )

        # P0.1: Log diagnostic when falling back to team name matching
        if vegas is None and hasattr(prediction, 'game_id') and prediction.game_id is not None:
            logger.debug(
                f"No Vegas line found via game_id={prediction.game_id} for "
                f"{prediction.home_team} vs {prediction.away_team}, trying team name fallback"
            )
            # Retry without game_id as explicit fallback
            vegas = self.get_line(prediction.home_team, prediction.away_team, game_id=None)

        if vegas is None:
            return None

        # Convert model spread to Vegas convention (negate) before comparing
        # Model: positive = home favored; Vegas: negative = home favored
        edge = (-prediction.spread) - vegas.spread

        return {
            "game_id": vegas.game_id,  # P0.2: include for reliable joins
            "home_team": prediction.home_team,
            "away_team": prediction.away_team,
            "model_spread": prediction.spread,
            "vegas_spread": vegas.spread,
            "vegas_open": vegas.spread_open,
            "edge": edge,
            "over_under": vegas.over_under,
            "is_value": abs(edge) >= self.value_threshold,
            "confidence": prediction.confidence,
        }

    def identify_value_plays(
        self,
        predictions: list[PredictedSpread],
    ) -> list[ValuePlay]:
        """Identify value plays from a list of predictions.

        Args:
            predictions: List of model predictions

        Returns:
            List of ValuePlay objects for games meeting threshold
        """
        value_plays = []

        for pred in predictions:
            comparison = self.compare_prediction(pred)

            if comparison is None:
                continue

            if not comparison["is_value"]:
                continue

            edge = comparison["edge"]

            # Determine which side has value
            if edge < 0:
                # Model likes home more than Vegas (take home team)
                side = "HOME"
                analysis = (
                    f"Model has {pred.home_team} as {abs(edge):.1f} points better "
                    f"than Vegas ({pred.spread:.1f} vs {comparison['vegas_spread']:.1f})"
                )
            else:
                # Model likes away more than Vegas (take away team)
                side = "AWAY"
                analysis = (
                    f"Model has {pred.away_team} as {abs(edge):.1f} points better "
                    f"than Vegas ({pred.spread:.1f} vs {comparison['vegas_spread']:.1f})"
                )

            vp = ValuePlay(
                home_team=pred.home_team,
                away_team=pred.away_team,
                model_spread=pred.spread,
                vegas_spread=comparison["vegas_spread"],
                edge=edge,
                side=side,
                confidence=comparison["confidence"],
                analysis=analysis,
                game_id=getattr(pred, 'game_id', None),  # P0.2: carry game_id
            )

            value_plays.append(vp)

        # Sort by edge magnitude
        value_plays.sort(key=lambda x: abs(x.edge), reverse=True)

        logger.info(f"Identified {len(value_plays)} value plays")
        return value_plays

    def generate_comparison_df(
        self,
        predictions: list[PredictedSpread],
    ) -> pd.DataFrame:
        """Generate full comparison DataFrame.

        Args:
            predictions: List of model predictions

        Returns:
            DataFrame with model vs Vegas comparison
        """
        comparisons = []

        for pred in predictions:
            comp = self.compare_prediction(pred)

            if comp is None:
                # Include prediction without Vegas line
                comp = {
                    "home_team": pred.home_team,
                    "away_team": pred.away_team,
                    "model_spread": pred.spread,
                    "vegas_spread": None,
                    "vegas_open": None,
                    "edge": None,
                    "over_under": None,
                    "is_value": False,
                    "confidence": pred.confidence,
                }

            comparisons.append(comp)

        df = pd.DataFrame(comparisons)

        # Add additional columns
        df["model_favorite"] = df.apply(
            lambda r: r["home_team"] if r["model_spread"] > 0 else r["away_team"],
            axis=1,
        )

        df["vegas_favorite"] = df.apply(
            lambda r: (
                r["home_team"]
                if r["vegas_spread"] is not None and r["vegas_spread"] < 0
                else (r["away_team"] if r["vegas_spread"] is not None else None)
            ),
            axis=1,
        )

        # Sort by edge
        df = df.sort_values("edge", key=abs, ascending=False, na_position="last")

        return df.reset_index(drop=True)

    def value_plays_to_dataframe(
        self,
        value_plays: list[ValuePlay],
    ) -> pd.DataFrame:
        """Convert value plays to DataFrame.

        Args:
            value_plays: List of value plays

        Returns:
            DataFrame with value play details
        """
        data = [
            {
                "game_id": vp.game_id,  # P0.2: include for reliable joins
                "home_team": vp.home_team,
                "away_team": vp.away_team,
                "side": vp.side,
                "team": vp.home_team if vp.side == "HOME" else vp.away_team,
                "model_spread": vp.model_spread,
                "vegas_spread": vp.vegas_spread,
                "edge": abs(vp.edge),
                "confidence": vp.confidence,
                "analysis": vp.analysis,
            }
            for vp in value_plays
        ]

        return pd.DataFrame(data)

    def get_line_movement(
        self,
        home_team: str,
        away_team: str,
    ) -> Optional[dict]:
        """Get line movement information for a game.

        Args:
            home_team: Home team
            away_team: Away team

        Returns:
            Dict with line movement info or None
        """
        vegas = self.get_line(home_team, away_team)

        if vegas is None or vegas.spread_open is None:
            return None

        movement = vegas.spread - vegas.spread_open

        return {
            "open": vegas.spread_open,
            "current": vegas.spread,
            "movement": movement,
            "direction": "toward_home" if movement < 0 else "toward_away",
            "moved_points": abs(movement),
        }
