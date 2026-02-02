"""Model components package.

Active models (EFM pipeline):
- EfficiencyFoundationModel: Core rating engine
- FinishingDrivesModel: Red zone efficiency
- SpecialTeamsModel: Field goal efficiency
- PreseasonPriors: Preseason ratings and adjustments

Legacy models available in src.models.legacy:
- RidgeRatingsModel, LuckRegressor, EarlyDownModel
"""

from .efficiency_foundation_model import EfficiencyFoundationModel
from .finishing_drives import FinishingDrivesModel
from .special_teams import SpecialTeamsModel
from .preseason_priors import PreseasonPriors

__all__ = [
    "EfficiencyFoundationModel",
    "FinishingDrivesModel",
    "SpecialTeamsModel",
    "PreseasonPriors",
]
