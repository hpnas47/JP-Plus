"""Model components package.

Active models (EFM pipeline):
- EfficiencyFoundationModel: Core rating engine using success rate, IsoPPP, and turnover margin
- FinishingDrivesModel: Red zone efficiency
- SpecialTeamsModel: Field goal, punt, and kickoff efficiency
- PreseasonPriors: Preseason ratings and blending
- TotalsModel: Opponent-adjusted scoring for over/under betting
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
