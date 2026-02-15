"""Shared display helpers for bet recommendation scripts."""

from src.api.cfbd_client import CFBDClient

# Cache FBS teams at module level (fetched once per session)
_fbs_teams_cache: dict[int, set[str]] = {}


def get_fbs_teams(year: int) -> set[str]:
    """Get FBS teams for a given year (cached)."""
    if year not in _fbs_teams_cache:
        client = CFBDClient()
        teams = client.get_fbs_teams(year=year)
        _fbs_teams_cache[year] = {t.school for t in teams}
    return _fbs_teams_cache[year]


# Team abbreviations
ABBREV = {
    'Alabama': 'ALA', 'Georgia': 'UGA', 'Ohio State': 'OSU', 'Texas': 'TEX',
    'Clemson': 'CLEM', 'Notre Dame': 'ND', 'Michigan': 'MICH', 'USC': 'USC',
    'Oregon': 'ORE', 'Penn State': 'PSU', 'Florida': 'FLA', 'LSU': 'LSU',
    'Oklahoma': 'OKLA', 'Tennessee': 'TENN', 'Auburn': 'AUB', 'Miami': 'MIA',
    'Florida State': 'FSU', 'Wisconsin': 'WIS', 'Iowa': 'IOWA', 'Utah': 'UTAH',
    'UCLA': 'UCLA', 'Washington': 'WASH', 'Texas A&M': 'TAMU', 'Ole Miss': 'MISS',
    'Arkansas': 'ARK', 'Kentucky': 'UK', 'South Carolina': 'SCAR', 'Missouri': 'MIZ',
    'NC State': 'NCST', 'Pittsburgh': 'PITT', 'Louisville': 'LOU', 'Virginia Tech': 'VT',
    'Duke': 'DUKE', 'Wake Forest': 'WAKE', 'Virginia': 'UVA', 'Boston College': 'BC',
    'Syracuse': 'SYR', 'Georgia Tech': 'GT', 'North Carolina': 'UNC', 'Stanford': 'STAN',
    'California': 'CAL', 'Arizona': 'ARIZ', 'Arizona State': 'ASU', 'Colorado': 'COLO',
    'Baylor': 'BAY', 'TCU': 'TCU', 'Kansas': 'KU', 'Kansas State': 'KSU',
    'Iowa State': 'ISU', 'Oklahoma State': 'OKST', 'West Virginia': 'WVU', 'Texas Tech': 'TTU',
    'Cincinnati': 'CIN', 'UCF': 'UCF', 'Houston': 'HOU', 'BYU': 'BYU',
    'Memphis': 'MEM', 'SMU': 'SMU', 'Tulane': 'TUL', 'Tulsa': 'TLSA',
    'San Diego State': 'SDSU', 'Fresno State': 'FRES', 'Boise State': 'BSU', 'Air Force': 'AFA',
    'Army': 'ARMY', 'Navy': 'NAVY', 'Marshall': 'MRSH', 'Appalachian State': 'APP', 'App State': 'APP',
    'Oregon State': 'ORST',
    'Coastal Carolina': 'CCU', 'James Madison': 'JMU', 'Liberty': 'LIB', 'Sam Houston': 'SHSU',
    'Minnesota': 'MINN', 'Illinois': 'ILL', 'Northwestern': 'NW', 'Purdue': 'PUR',
    'Indiana': 'IND', 'Nebraska': 'NEB', 'Michigan State': 'MSU', 'Rutgers': 'RUT', 'Maryland': 'UMD',
    'Mississippi State': 'MSST', 'Vanderbilt': 'VAN', 'Louisiana': 'ULL', 'Troy': 'TROY',
    'South Alabama': 'USA', 'Georgia Southern': 'GASO', 'Georgia State': 'GAST', 'UTSA': 'UTSA',
    'North Texas': 'UNT', 'Rice': 'RICE', 'Florida Atlantic': 'FAU', 'Charlotte': 'CLT',
    'East Carolina': 'ECU', 'Temple': 'TEM', 'Buffalo': 'BUFF', 'Ohio': 'OHIO',
    'Miami (OH)': 'M-OH', 'Bowling Green': 'BGSU', 'Kent State': 'KENT', 'Akron': 'AKR',
    'Ball State': 'BALL', 'Toledo': 'TOL', 'Central Michigan': 'CMU', 'Eastern Michigan': 'EMU',
    'Western Michigan': 'WMU', 'Northern Illinois': 'NIU', 'Nevada': 'NEV', 'UNLV': 'UNLV',
    'Wyoming': 'WYO', 'New Mexico': 'UNM', 'Utah State': 'USU', 'Colorado State': 'CSU',
    "Hawai'i": 'HAW', 'San José State': 'SJSU', 'Louisiana Tech': 'LT', 'UAB': 'UAB',
    'Middle Tennessee': 'MTSU', 'Western Kentucky': 'WKU', 'Old Dominion': 'ODU',
    'Southern Miss': 'USM', 'FIU': 'FIU', 'New Mexico State': 'NMSU', 'South Florida': 'USF',
    'Kennesaw State': 'KENST', 'Jacksonville State': 'JVST', 'Connecticut': 'CONN', 'UMass': 'MASS',
    'Arkansas State': 'ARST', 'Louisiana-Monroe': 'ULM', 'Texas State': 'TXST', 'UL Monroe': 'ULM',
}


def get_abbrev(team: str) -> str:
    return ABBREV.get(team, team[:4].upper())
