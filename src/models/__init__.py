"""Model components package.

Active models (EFM pipeline):
- EfficiencyFoundationModel: Core rating engine using success rate, IsoPPP, turnover margin, and RZ Leverage
- SpecialTeamsModel: Field goal, punt, and kickoff efficiency
- PreseasonPriors: Preseason ratings and blending
- TotalsModel: Opponent-adjusted scoring for over/under betting

Shelved models (kept for backward compatibility):
- FinishingDrivesModel: Replaced by RZ Leverage weighting in EFM (4 backtest rejections)
"""

from .efficiency_foundation_model import EfficiencyFoundationModel
from .finishing_drives import FinishingDrivesModel
from .special_teams import SpecialTeamsModel
from .preseason_priors import PreseasonPriors
from .totals_model import TotalsModel, TotalsRating, TotalsPrediction

__all__ = [
    "EfficiencyFoundationModel",
    "FinishingDrivesModel",
    "SpecialTeamsModel",
    "PreseasonPriors",
    "TotalsModel",
    "TotalsRating",
    "TotalsPrediction",
]
