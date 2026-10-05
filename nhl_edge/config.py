"""Settings for the NHL edge system. Tweak these, not the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SITE_DATA_DIR = ROOT / "docs" / "data"

GAMES_CSV = DATA_DIR / "games.csv"            # every game, cached from the NHL API
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"  # model vs market for every game, every day
BETS_CSV = DATA_DIR / "bets.csv"              # flagged bets, closing lines, results

# How far back to pull history (more seasons = better-trained model).
HISTORY_START = "2021-10-01"

# --- Model ---
ELO_K = 8.0              # how fast Elo ratings move
ELO_HOME_ADV = 35.0      # home-ice advantage in Elo points
ELO_SEASON_CARRY = 0.70  # share of last season's rating kept over the summer
SHRINK_GAMES = 8         # early-season stats are pulled toward average until ~this many games
BURN_IN_GAMES = 300      # skip the first games of history while ratings settle

# --- Betting ---
# Blend model with market: 1.0 = trust the model completely, 0.0 = trust the market.
# Markets are sharp, so trusting the model ~50% avoids chasing fake edges.
MODEL_WEIGHT = 0.50
MIN_EV = 0.03            # only flag bets with at least +3% expected value
MAX_DOG_ODDS = 250       # skip big underdogs (above +250); too noisy
KELLY_FRACTION = 0.25    # quarter Kelly
MAX_STAKE_UNITS = 2.0    # never more than 2 units (2% of bankroll) on one game

# --- Optional expected-goals data (MoneyPuck). Skipped automatically if unavailable. ---
MONEYPUCK_TEAM_URL = (
    "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/regular/teams/{team}.csv"
)

ODDS_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
