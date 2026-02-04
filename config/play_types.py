"""Play type definitions - single source of truth for the entire codebase.

All modules that need to identify play types (turnovers, non-scrimmage, etc.)
should import from this file to ensure consistency.
"""

# Turnover play types where the offense lost the ball
# These are the CFBD API play_type values that represent turnovers
# The "offense" field in these plays is the team that LOST the ball
TURNOVER_PLAY_TYPES = frozenset({
    # Fumbles recovered by opponent
    "Fumble Recovery (Opponent)",
    "Fumble Return Touchdown",
    # Interceptions
    "Interception",
    "Interception Return",
    "Interception Return Touchdown",
    "Pass Interception Return",
})

# Points value per turnover (empirical, used for margin scrubbing and ratings)
POINTS_PER_TURNOVER = 4.5


def validate_turnover_play_types():
    """Validate that turnover play types are properly defined.

    Call this at module load or in tests to catch any issues early.

    Raises:
        AssertionError: If validation fails
    """
    # Ensure we have the core turnover types
    required_types = {
        "Fumble Recovery (Opponent)",
        "Interception",
    }
    missing = required_types - TURNOVER_PLAY_TYPES
    assert not missing, f"Missing required turnover types: {missing}"

    # Ensure all types are non-empty strings
    for pt in TURNOVER_PLAY_TYPES:
        assert isinstance(pt, str) and pt.strip(), f"Invalid play type: {pt!r}"

    return True


# Validate on import
validate_turnover_play_types()
