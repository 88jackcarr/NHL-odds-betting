"""Team names and abbreviations, used to line up NHL API games with sportsbook odds."""
import unicodedata

TEAM_NAMES = {
    "ANA": "Anaheim Ducks", "ARI": "Arizona Coyotes", "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres", "CGY": "Calgary Flames", "CAR": "Carolina Hurricanes",
    "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche", "CBJ": "Columbus Blue Jackets",
    "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers", "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens", "NSH": "Nashville Predators", "NJD": "New Jersey Devils",
    "NYI": "New York Islanders", "NYR": "New York Rangers", "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks",
    "SEA": "Seattle Kraken", "STL": "St Louis Blues", "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs", "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights", "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}

# Other names sportsbooks use.
ALIASES = {
    "utah hockey club": "UTA", "utah": "UTA", "st. louis blues": "STL",
    "montréal canadiens": "MTL",
}

# MoneyPuck uses a few different abbreviations.
MONEYPUCK_ABBREV = {"LAK": "L.A", "NJD": "N.J", "SJS": "S.J", "TBL": "T.B"}


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(name.lower().replace(".", "").split())


_LOOKUP = {_norm(v): k for k, v in TEAM_NAMES.items()}
_LOOKUP.update({_norm(k): v for k, v in ALIASES.items()})


def abbrev_for(name: str):
    """Sportsbook team name -> NHL abbreviation (None if unknown)."""
    return _LOOKUP.get(_norm(name))
