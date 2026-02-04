"""Play type definitions - single source of truth for the entire codebase.

All modules that need to identify play types (turnovers, non-scrimmage, etc.)
should import from this file to ensure consistency.
"""

# Scrimmage play types - actual offensive plays that should be used for efficiency metrics
# These represent meaningful offensive attempts where success rate applies
SCRIMMAGE_PLAY_TYPES = frozenset({
    # Standard plays
    "Rush",
    "Pass Reception",
    "Pass Incompletion",
    "Sack",
    "Rushing Touchdown",
    "Passing Touchdown",
    # Turnovers that happen on scrimmage plays (offense lost the ball)
    "Fumble Recovery (Own)",
    "Fumble Recovery (Opponent)",
    "Fumble Return Touchdown",
    "Interception",
    "Interception Return",
    "Interception Return Touchdown",
    "Pass Interception Return",
    # Safety can occur on scrimmage plays
    "Safety",
})

# Non-scrimmage play types that should be EXCLUDED from efficiency analysis
# These have PPA/down in CFBD data but don't represent offensive attempts
NON_SCRIMMAGE_PLAY_TYPES = frozenset({
    # Period markers (not plays)
    "End Period",
    "End of Half",
    "End of Game",
    "Timeout",
    # Special teams plays
    "Punt",
    "Punt Return Touchdown",
    "Blocked Punt",
    "Blocked Punt Touchdown",
    "Kickoff",
    "Kickoff Return Touchdown",
    "Field Goal Good",
    "Field Goal Missed",
    "Blocked Field Goal",
    "Blocked Field Goal Touchdown",
    # PAT/2PC (special situation, not normal downs)
    "Extra Point Good",
    "Extra Point Missed",
    "Two-Point Conversion",
    "Defensive 2pt Conversion",
    # Penalties without a play
    "Penalty",
})

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


def validate_play_types():
    """Validate that play type definitions are properly defined and consistent.

    Call this at module load or in tests to catch any issues early.

    Raises:
        AssertionError: If validation fails
    """
    # Ensure we have the core turnover types
    required_turnover_types = {
        "Fumble Recovery (Opponent)",
        "Interception",
    }
    missing = required_turnover_types - TURNOVER_PLAY_TYPES
    assert not missing, f"Missing required turnover types: {missing}"

    # Ensure turnovers are subset of scrimmage plays
    turnover_not_in_scrimmage = TURNOVER_PLAY_TYPES - SCRIMMAGE_PLAY_TYPES
    assert not turnover_not_in_scrimmage, (
        f"Turnover types not in scrimmage plays: {turnover_not_in_scrimmage}"
    )

    # Ensure no overlap between scrimmage and non-scrimmage
    overlap = SCRIMMAGE_PLAY_TYPES & NON_SCRIMMAGE_PLAY_TYPES
    assert not overlap, f"Overlap between scrimmage and non-scrimmage: {overlap}"

    # Ensure all types are non-empty strings
    for pt in TURNOVER_PLAY_TYPES | SCRIMMAGE_PLAY_TYPES | NON_SCRIMMAGE_PLAY_TYPES:
        assert isinstance(pt, str) and pt.strip(), f"Invalid play type: {pt!r}"

    return True


# Validate on import
validate_play_types()
