"""Vegas line comparison and value play identification."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
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
    """Container for identified value play.

    Edge Sign Convention (DO NOT CHANGE):
        edge = (-model_spread) - vegas_spread

        Negative edge: Model favors HOME more than Vegas → bet HOME.
        Positive edge: Model favors AWAY more than Vegas → bet AWAY.

        Example (bet HOME): Model spread = +10 (home by 10), Vegas = -3 (home by 3).
            edge = (-10) - (-3) = -7 → model likes HOME 7 pts more than Vegas.
        Example (bet AWAY): Model spread = +1 (home by 1), Vegas = -7 (home by 7).
            edge = (-1) - (-7) = +6 → model likes AWAY 6 pts more than Vegas.
    """

    home_team: str
    away_team: str
    model_spread: float
    vegas_spread: float
    edge: float  # Negative = model favors HOME more; Positive = model favors AWAY more
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

            # P1.1: Deterministic provider fallback — sort by provider name for stability
            if provider_line is None and game.lines:
                sorted_lines = sorted(
                    [l for l in game.lines if l.spread is not None],
                    key=lambda l: (l.provider or "").lower()
                )
                provider_line = sorted_lines[0] if sorted_lines else None

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
            # P1.2: Warn on duplicate game_id, keep first encountered
            if vl.game_id in self.lines_by_id:
                logger.warning(
                    f"Duplicate game_id {vl.game_id}: "
                    f"{self.lines_by_id[vl.game_id].home_team} vs {vl.home_team}. Keeping first."
                )
            else:
                self.lines_by_id[vl.game_id] = vl
            # Fallback dict: also keep first for consistency with lines_by_id
            team_key = (game.home_team, game.away_team)
            if team_key not in self.lines:
                self.lines[team_key] = vl

        # P2.1: Opener data reliability diagnostics
        n_lines = len(vegas_lines)
        if n_lines > 0:
            n_with_open = sum(1 for vl in vegas_lines if vl.spread_open is not None)
            open_pct = n_with_open / n_lines * 100
            if n_with_open > 0:
                n_moved = sum(
                    1 for vl in vegas_lines
                    if vl.spread_open is not None and abs(vl.spread - vl.spread_open) > 0.5
                )
                moved_pct = n_moved / n_with_open * 100
                logger.debug(
                    f"Opener quality: {open_pct:.0f}% have spread_open, "
                    f"{moved_pct:.0f}% moved >0.5 pts from open"
                )
            if open_pct < 50:
                logger.warning(
                    f"Opener data unreliable: only {open_pct:.0f}% of lines have spread_open"
                )

        logger.info(f"Fetched {n_lines} Vegas lines for {year} week {week}")
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

        Edge Sign Convention:
            edge = (-model_spread) - vegas_spread
            Negative edge → model favors HOME more than Vegas (bet HOME).
            Positive edge → model favors AWAY more than Vegas (bet AWAY).

        Args:
            prediction: Model prediction

        Returns:
            Comparison dict or None if no Vegas line available
        """
        # Prefer game_id matching (reliable), fallback to team names handled internally by get_line
        vegas = self.get_line(
            prediction.home_team,
            prediction.away_team,
            game_id=prediction.game_id if hasattr(prediction, 'game_id') else None
        )

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

        PERFORMANCE: Vectorized merge replaces per-prediction loop for 10-100x speedup.

        Args:
            predictions: List of model predictions

        Returns:
            DataFrame with model vs Vegas comparison
        """
        if not predictions:
            return pd.DataFrame()

        # Build predictions DataFrame
        pred_data = [
            {
                "game_id": getattr(p, "game_id", None),
                "home_team": p.home_team,
                "away_team": p.away_team,
                "model_spread": p.spread,
                "confidence": p.confidence,
            }
            for p in predictions
        ]
        df = pd.DataFrame(pred_data)

        # Build lines DataFrame from game_id dict (primary)
        if self.lines_by_id:
            lines_data = [
                {
                    "game_id": vl.game_id,
                    "vegas_spread": vl.spread,
                    "vegas_open": vl.spread_open,
                    "over_under": vl.over_under,
                }
                for vl in self.lines_by_id.values()
            ]
            lines_df = pd.DataFrame(lines_data)

            # Merge on game_id (left join)
            df = df.merge(lines_df, on="game_id", how="left")
        else:
            # No lines by ID, add empty columns
            df["vegas_spread"] = np.nan
            df["vegas_open"] = np.nan
            df["over_under"] = np.nan

        # Fallback: for rows without Vegas line, try team-name lookup
        missing_mask = df["vegas_spread"].isna()
        if missing_mask.any() and self.lines:
            for idx in df.index[missing_mask]:
                key = (df.at[idx, "home_team"], df.at[idx, "away_team"])
                vl = self.lines.get(key)
                if vl is not None:
                    df.at[idx, "vegas_spread"] = vl.spread
                    df.at[idx, "vegas_open"] = vl.spread_open
                    df.at[idx, "over_under"] = vl.over_under

        # Vectorized edge calculation: edge = (-model_spread) - vegas_spread
        df["edge"] = (-df["model_spread"]) - df["vegas_spread"]

        # Vectorized is_value: abs(edge) >= threshold, False if edge is NaN
        df["is_value"] = df["edge"].abs() >= self.value_threshold

        # Model favorite: home if model_spread > 0, else away
        df["model_favorite"] = np.where(
            df["model_spread"] > 0,
            df["home_team"],
            df["away_team"]
        )

        # Vegas favorite: home if vegas_spread < 0, away if > 0, None if missing
        df["vegas_favorite"] = np.where(
            df["vegas_spread"].isna(),
            None,
            np.where(df["vegas_spread"] < 0, df["home_team"], df["away_team"])
        )

        # Sort by absolute edge, NaN last
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
                "edge": abs(vp.edge),  # Absolute edge for sorting/display
                "edge_signed": vp.edge,  # P1.3: Signed edge for diagnostics (neg=home, pos=away)
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
        game_id: Optional[int] = None,
    ) -> Optional[dict]:
        """Get line movement information for a game.

        P2.2: Supports game_id matching (preferred), falls back to team names.

        Args:
            home_team: Home team
            away_team: Away team
            game_id: Optional CFBD game_id for reliable matching

        Returns:
            Dict with line movement info or None
        """
        vegas = self.get_line(home_team, away_team, game_id=game_id)

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
