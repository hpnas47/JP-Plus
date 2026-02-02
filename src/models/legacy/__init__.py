"""Legacy models - kept for comparison but not recommended for use.

These models were part of the original margin-based approach before the
Efficiency Foundation Model (EFM) was implemented. The EFM approach is
more accurate and should be used instead.

Models:
- ridge_model.py: Original margin-based ridge regression
- luck_regression.py: Turnover luck adjustment for ridge model
- early_down_model.py: Early-down success rate (EFM captures this directly)

To use these (not recommended):
    from src.models.legacy.ridge_model import RidgeRatingsModel
    from src.models.legacy.luck_regression import LuckRegressor
    from src.models.legacy.early_down_model import EarlyDownModel
"""

from .ridge_model import RidgeRatingsModel, TeamRatings
from .luck_regression import LuckRegressor
from .early_down_model import EarlyDownModel

__all__ = [
    "RidgeRatingsModel",
    "TeamRatings",
    "LuckRegressor",
    "EarlyDownModel",
]
