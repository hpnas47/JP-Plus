"""Team metadata including altitude venues and rivalries."""

# High altitude venues with elevation (feet) and point adjustment for visiting sea-level teams
ALTITUDE_VENUES: dict[str, dict] = {
    "Wyoming": {"elevation": 7220, "adjustment": 3.0},
    "Air Force": {"elevation": 6621, "adjustment": 2.5},
    "Colorado": {"elevation": 5430, "adjustment": 2.0},
    "New Mexico": {"elevation": 5312, "adjustment": 2.0},
    "Utah": {"elevation": 4637, "adjustment": 1.5},
    "BYU": {"elevation": 4549, "adjustment": 1.5},
    "Colorado State": {"elevation": 4982, "adjustment": 1.5},
    "Nevada": {"elevation": 4505, "adjustment": 1.5},
    "UTEP": {"elevation": 3740, "adjustment": 1.0},
    "Boise State": {"elevation": 2730, "adjustment": 0.5},
}

# Teams considered high-altitude (don't get penalty when visiting other high-altitude venues)
HIGH_ALTITUDE_TEAMS = set(ALTITUDE_VENUES.keys())

# Rivalry matchups (team1, team2) - order doesn't matter
RIVALRIES: list[tuple[str, str]] = [
    # SEC
    ("Alabama", "Auburn"),
    ("Alabama", "Tennessee"),
    ("Georgia", "Florida"),
    ("Georgia", "Auburn"),
    ("Georgia", "Georgia Tech"),
    ("LSU", "Alabama"),
    ("LSU", "Texas A&M"),
    ("Ole Miss", "Mississippi State"),
    ("South Carolina", "Clemson"),
    ("Kentucky", "Louisville"),
    ("Florida", "Florida State"),
    ("Florida", "Miami"),
    ("Tennessee", "Vanderbilt"),
    ("Arkansas", "Texas A&M"),
    ("Arkansas", "LSU"),
    ("Missouri", "Kansas"),
    ("Texas", "Oklahoma"),
    ("Texas", "Texas A&M"),
    ("Oklahoma", "Oklahoma State"),

    # Big Ten
    ("Michigan", "Ohio State"),
    ("Michigan", "Michigan State"),
    ("Ohio State", "Penn State"),
    ("Wisconsin", "Minnesota"),
    ("Iowa", "Iowa State"),
    ("Iowa", "Nebraska"),
    ("Nebraska", "Colorado"),
    ("Purdue", "Indiana"),
    ("Illinois", "Northwestern"),
    ("USC", "UCLA"),
    ("Oregon", "Oregon State"),
    ("Oregon", "Washington"),
    ("Washington", "Washington State"),
    ("Rutgers", "Maryland"),

    # Big 12
    ("Oklahoma State", "Texas Tech"),
    ("Kansas State", "Kansas"),
    ("Baylor", "TCU"),
    ("West Virginia", "Pitt"),
    ("Iowa State", "Kansas State"),
    ("BYU", "Utah"),
    ("BYU", "Utah State"),
    ("Colorado", "Colorado State"),
    ("Arizona", "Arizona State"),
    ("Cincinnati", "Louisville"),
    ("UCF", "South Florida"),
    ("Houston", "Rice"),

    # ACC
    ("Clemson", "South Carolina"),
    ("North Carolina", "Duke"),
    ("North Carolina", "NC State"),
    ("NC State", "Wake Forest"),
    ("Virginia", "Virginia Tech"),
    ("Boston College", "Notre Dame"),
    ("Miami", "Florida State"),
    ("Pitt", "Penn State"),
    ("Georgia Tech", "Georgia"),
    ("Louisville", "Kentucky"),
    ("Syracuse", "Boston College"),
    ("Stanford", "Cal"),
    ("SMU", "TCU"),

    # Group of 5 notable rivalries
    ("Army", "Navy"),
    ("Army", "Air Force"),
    ("Navy", "Air Force"),
    ("Boise State", "Fresno State"),
    ("San Diego State", "Fresno State"),
    ("Memphis", "Cincinnati"),
    ("Tulane", "Tulsa"),
    ("Hawaii", "San Diego State"),
    ("UNLV", "Nevada"),
    ("New Mexico", "New Mexico State"),
    ("North Texas", "SMU"),
    ("East Carolina", "NC State"),
    ("Marshall", "Ohio"),
    ("Toledo", "Bowling Green"),
    ("Miami (OH)", "Cincinnati"),
    ("Kent State", "Akron"),
    ("Wyoming", "Colorado State"),
    ("Utah State", "BYU"),
    ("Troy", "South Alabama"),
    ("Louisiana", "Louisiana Tech"),
    ("Appalachian State", "Georgia Southern"),
]

# Build lookup set for faster rivalry checking
_RIVALRY_SET: set[frozenset] = {frozenset(r) for r in RIVALRIES}


def is_rivalry_game(team1: str, team2: str) -> bool:
    """Check if two teams are rivals."""
    return frozenset([team1, team2]) in _RIVALRY_SET


def get_team_altitude(team: str) -> int:
    """Get team's home venue elevation in feet. Returns 0 for sea-level teams."""
    return ALTITUDE_VENUES.get(team, {}).get("elevation", 0)


def get_altitude_adjustment(home_team: str, away_team: str) -> float:
    """
    Calculate altitude adjustment for away team visiting high-altitude venue.

    Returns positive adjustment favoring home team if away team is from low altitude.
    """
    if home_team not in ALTITUDE_VENUES:
        return 0.0

    # High-altitude teams don't get penalized visiting other high-altitude venues
    if away_team in HIGH_ALTITUDE_TEAMS:
        return 0.0

    return ALTITUDE_VENUES[home_team]["adjustment"]


# Team location data for travel calculations (latitude, longitude, timezone offset from ET)
# Major FBS teams - timezone offset is hours behind Eastern Time
TEAM_LOCATIONS: dict[str, dict] = {
    # SEC
    "Alabama": {"lat": 33.2084, "lon": -87.5503, "tz_offset": 1},
    "Arkansas": {"lat": 36.0686, "lon": -94.1778, "tz_offset": 1},
    "Auburn": {"lat": 32.6028, "lon": -85.4897, "tz_offset": 1},
    "Florida": {"lat": 29.6500, "lon": -82.3486, "tz_offset": 0},
    "Georgia": {"lat": 33.9500, "lon": -83.3736, "tz_offset": 0},
    "Kentucky": {"lat": 38.0281, "lon": -84.5050, "tz_offset": 0},
    "LSU": {"lat": 30.4122, "lon": -91.1836, "tz_offset": 1},
    "Mississippi State": {"lat": 33.4556, "lon": -88.7953, "tz_offset": 1},
    "Missouri": {"lat": 38.9358, "lon": -92.3281, "tz_offset": 1},
    "Oklahoma": {"lat": 35.2058, "lon": -97.4456, "tz_offset": 1},
    "Ole Miss": {"lat": 34.3622, "lon": -89.5344, "tz_offset": 1},
    "South Carolina": {"lat": 33.9731, "lon": -81.0197, "tz_offset": 0},
    "Tennessee": {"lat": 35.9550, "lon": -83.9256, "tz_offset": 0},
    "Texas": {"lat": 30.2836, "lon": -97.7322, "tz_offset": 1},
    "Texas A&M": {"lat": 30.6103, "lon": -96.3397, "tz_offset": 1},
    "Vanderbilt": {"lat": 36.1447, "lon": -86.8094, "tz_offset": 1},

    # Big Ten
    "Illinois": {"lat": 40.0992, "lon": -88.2361, "tz_offset": 1},
    "Indiana": {"lat": 39.1803, "lon": -86.5264, "tz_offset": 0},
    "Iowa": {"lat": 41.6589, "lon": -91.5508, "tz_offset": 1},
    "Maryland": {"lat": 38.9869, "lon": -76.9426, "tz_offset": 0},
    "Michigan": {"lat": 42.2656, "lon": -83.7486, "tz_offset": 0},
    "Michigan State": {"lat": 42.7284, "lon": -84.4839, "tz_offset": 0},
    "Minnesota": {"lat": 44.9764, "lon": -93.2247, "tz_offset": 1},
    "Nebraska": {"lat": 40.8206, "lon": -96.7056, "tz_offset": 1},
    "Northwestern": {"lat": 42.0656, "lon": -87.6836, "tz_offset": 1},
    "Ohio State": {"lat": 40.0017, "lon": -83.0197, "tz_offset": 0},
    "Oregon": {"lat": 44.0583, "lon": -123.0681, "tz_offset": 3},
    "Penn State": {"lat": 40.8122, "lon": -77.8561, "tz_offset": 0},
    "Purdue": {"lat": 40.4406, "lon": -86.9136, "tz_offset": 0},
    "Rutgers": {"lat": 40.5139, "lon": -74.4653, "tz_offset": 0},
    "UCLA": {"lat": 34.1614, "lon": -118.1681, "tz_offset": 3},
    "USC": {"lat": 34.0224, "lon": -118.2851, "tz_offset": 3},
    "Washington": {"lat": 47.6503, "lon": -122.3019, "tz_offset": 3},
    "Wisconsin": {"lat": 43.0700, "lon": -89.4111, "tz_offset": 1},

    # Big 12
    "Arizona": {"lat": 32.2289, "lon": -110.9486, "tz_offset": 2},
    "Arizona State": {"lat": 33.4255, "lon": -111.9325, "tz_offset": 2},
    "Baylor": {"lat": 31.5586, "lon": -97.1153, "tz_offset": 1},
    "BYU": {"lat": 40.2568, "lon": -111.6547, "tz_offset": 2},
    "Cincinnati": {"lat": 39.1317, "lon": -84.5153, "tz_offset": 0},
    "Colorado": {"lat": 40.0092, "lon": -105.2669, "tz_offset": 2},
    "Houston": {"lat": 29.7214, "lon": -95.3431, "tz_offset": 1},
    "Iowa State": {"lat": 42.0140, "lon": -93.6358, "tz_offset": 1},
    "Kansas": {"lat": 38.9583, "lon": -95.2522, "tz_offset": 1},
    "Kansas State": {"lat": 39.2017, "lon": -96.5936, "tz_offset": 1},
    "Oklahoma State": {"lat": 36.1269, "lon": -97.0683, "tz_offset": 1},
    "TCU": {"lat": 32.7097, "lon": -97.3678, "tz_offset": 1},
    "Texas Tech": {"lat": 33.5906, "lon": -101.8725, "tz_offset": 1},
    "UCF": {"lat": 28.6024, "lon": -81.2001, "tz_offset": 0},
    "Utah": {"lat": 40.7600, "lon": -111.8489, "tz_offset": 2},
    "West Virginia": {"lat": 39.6500, "lon": -79.9556, "tz_offset": 0},

    # ACC
    "Boston College": {"lat": 42.3355, "lon": -71.1685, "tz_offset": 0},
    "Cal": {"lat": 37.8708, "lon": -122.2506, "tz_offset": 3},
    "Clemson": {"lat": 34.6783, "lon": -82.8436, "tz_offset": 0},
    "Duke": {"lat": 36.0011, "lon": -78.9422, "tz_offset": 0},
    "Florida State": {"lat": 30.4383, "lon": -84.3044, "tz_offset": 0},
    "Georgia Tech": {"lat": 33.7722, "lon": -84.3928, "tz_offset": 0},
    "Louisville": {"lat": 38.2131, "lon": -85.7586, "tz_offset": 0},
    "Miami": {"lat": 25.7211, "lon": -80.2789, "tz_offset": 0},
    "NC State": {"lat": 35.7872, "lon": -78.6711, "tz_offset": 0},
    "North Carolina": {"lat": 35.9050, "lon": -79.0469, "tz_offset": 0},
    "Notre Dame": {"lat": 41.6983, "lon": -86.2339, "tz_offset": 0},
    "Pitt": {"lat": 40.4444, "lon": -79.9608, "tz_offset": 0},
    "SMU": {"lat": 32.8417, "lon": -96.7839, "tz_offset": 1},
    "Stanford": {"lat": 37.4346, "lon": -122.1609, "tz_offset": 3},
    "Syracuse": {"lat": 43.0361, "lon": -76.1364, "tz_offset": 0},
    "Virginia": {"lat": 38.0314, "lon": -78.5131, "tz_offset": 0},
    "Virginia Tech": {"lat": 37.2200, "lon": -80.4181, "tz_offset": 0},
    "Wake Forest": {"lat": 36.1344, "lon": -80.2500, "tz_offset": 0},

    # Independents
    "Army": {"lat": 41.3917, "lon": -73.9583, "tz_offset": 0},
    "UConn": {"lat": 41.8078, "lon": -72.2539, "tz_offset": 0},
    "UMass": {"lat": 42.3903, "lon": -72.5267, "tz_offset": 0},

    # Group of 5 (selected)
    "Air Force": {"lat": 38.9983, "lon": -104.8617, "tz_offset": 2},
    "Boise State": {"lat": 43.6036, "lon": -116.1972, "tz_offset": 2},
    "Fresno State": {"lat": 36.8139, "lon": -119.7558, "tz_offset": 3},
    "Hawaii": {"lat": 21.2969, "lon": -157.8171, "tz_offset": 5},
    "Memphis": {"lat": 35.1186, "lon": -89.9372, "tz_offset": 1},
    "Navy": {"lat": 38.9833, "lon": -76.4867, "tz_offset": 0},
    "Nevada": {"lat": 39.5461, "lon": -119.8172, "tz_offset": 3},
    "New Mexico": {"lat": 35.0853, "lon": -106.6189, "tz_offset": 2},
    "San Diego State": {"lat": 32.7758, "lon": -117.0700, "tz_offset": 3},
    "San Jose State": {"lat": 37.3353, "lon": -121.8814, "tz_offset": 3},
    "UNLV": {"lat": 36.1089, "lon": -115.1428, "tz_offset": 3},
    "Wyoming": {"lat": 41.3142, "lon": -105.5672, "tz_offset": 2},
    "Tulane": {"lat": 29.9428, "lon": -90.1172, "tz_offset": 1},
    "South Florida": {"lat": 28.0639, "lon": -82.4128, "tz_offset": 0},
    "East Carolina": {"lat": 35.6042, "lon": -77.3722, "tz_offset": 0},
    "Appalachian State": {"lat": 36.2139, "lon": -81.6806, "tz_offset": 0},
    "Coastal Carolina": {"lat": 33.7944, "lon": -79.0186, "tz_offset": 0},
    "James Madison": {"lat": 38.4350, "lon": -78.8733, "tz_offset": 0},
    "Liberty": {"lat": 37.3522, "lon": -79.1792, "tz_offset": 0},
    "Marshall": {"lat": 38.4236, "lon": -82.4256, "tz_offset": 0},
    "Old Dominion": {"lat": 36.8861, "lon": -76.3061, "tz_offset": 0},
    "Georgia Southern": {"lat": 32.4217, "lon": -81.7839, "tz_offset": 0},
    "Georgia State": {"lat": 33.7508, "lon": -84.3869, "tz_offset": 0},
    "Louisiana": {"lat": 30.2139, "lon": -92.0178, "tz_offset": 1},
    "Louisiana Tech": {"lat": 32.5267, "lon": -92.6447, "tz_offset": 1},
    "Troy": {"lat": 31.7994, "lon": -85.9622, "tz_offset": 1},
    "South Alabama": {"lat": 30.6944, "lon": -88.1772, "tz_offset": 1},
    "Arkansas State": {"lat": 35.8428, "lon": -90.6764, "tz_offset": 1},
    "Texas State": {"lat": 29.8886, "lon": -97.9383, "tz_offset": 1},
    "UTSA": {"lat": 29.5828, "lon": -98.6206, "tz_offset": 1},
    "Rice": {"lat": 29.7167, "lon": -95.4019, "tz_offset": 1},
    "Tulsa": {"lat": 36.1519, "lon": -95.9458, "tz_offset": 1},
    "North Texas": {"lat": 33.2078, "lon": -97.1528, "tz_offset": 1},
    "UTEP": {"lat": 31.7694, "lon": -106.5022, "tz_offset": 2},
    "Charlotte": {"lat": 35.3047, "lon": -80.7328, "tz_offset": 0},
    "FAU": {"lat": 26.3708, "lon": -80.1019, "tz_offset": 0},
    "FIU": {"lat": 25.7558, "lon": -80.3753, "tz_offset": 0},
    "Middle Tennessee": {"lat": 35.8489, "lon": -86.3658, "tz_offset": 1},
    "Western Kentucky": {"lat": 36.9867, "lon": -86.4586, "tz_offset": 1},
    "MTSU": {"lat": 35.8489, "lon": -86.3658, "tz_offset": 1},
    "UAB": {"lat": 33.5017, "lon": -86.8089, "tz_offset": 1},
    "Temple": {"lat": 39.9817, "lon": -75.1508, "tz_offset": 0},
    "Akron": {"lat": 41.0764, "lon": -81.5117, "tz_offset": 0},
    "Ball State": {"lat": 40.2083, "lon": -85.4097, "tz_offset": 0},
    "Bowling Green": {"lat": 41.3786, "lon": -83.6336, "tz_offset": 0},
    "Buffalo": {"lat": 42.9992, "lon": -78.7861, "tz_offset": 0},
    "Central Michigan": {"lat": 43.5922, "lon": -84.7744, "tz_offset": 0},
    "Eastern Michigan": {"lat": 42.2500, "lon": -83.6236, "tz_offset": 0},
    "Kent State": {"lat": 41.1492, "lon": -81.3414, "tz_offset": 0},
    "Miami (OH)": {"lat": 39.5089, "lon": -84.7353, "tz_offset": 0},
    "Northern Illinois": {"lat": 41.9347, "lon": -88.7703, "tz_offset": 1},
    "Ohio": {"lat": 39.3253, "lon": -82.1014, "tz_offset": 0},
    "Toledo": {"lat": 41.6586, "lon": -83.6128, "tz_offset": 0},
    "Western Michigan": {"lat": 42.2833, "lon": -85.6144, "tz_offset": 0},
    "Colorado State": {"lat": 40.5758, "lon": -105.0842, "tz_offset": 2},
    "Utah State": {"lat": 41.7500, "lon": -111.8128, "tz_offset": 2},
    "Washington State": {"lat": 46.7319, "lon": -117.1542, "tz_offset": 3},
    "Oregon State": {"lat": 44.5597, "lon": -123.2781, "tz_offset": 3},
    "Jacksonville State": {"lat": 33.8236, "lon": -85.7661, "tz_offset": 1},
    "Sam Houston": {"lat": 30.7133, "lon": -95.5497, "tz_offset": 1},
    "Kennesaw State": {"lat": 34.0378, "lon": -84.5817, "tz_offset": 0},
}


def get_team_location(team: str) -> dict | None:
    """Get team location data. Returns None if team not found."""
    return TEAM_LOCATIONS.get(team)


def get_timezone_difference(team1: str, team2: str) -> int:
    """Get timezone difference in hours between two teams."""
    loc1 = TEAM_LOCATIONS.get(team1)
    loc2 = TEAM_LOCATIONS.get(team2)

    if loc1 is None or loc2 is None:
        return 0

    return abs(loc1["tz_offset"] - loc2["tz_offset"])
