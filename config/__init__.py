"""Configuration package for CFB Power Ratings Model."""

from .settings import Settings, get_settings
from .teams import ALTITUDE_VENUES, RIVALRIES, get_team_altitude

__all__ = [
    "Settings",
    "get_settings",
    "ALTITUDE_VENUES",
    "RIVALRIES",
    "get_team_altitude",
]
