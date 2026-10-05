"""Pulls the schedule and results from the free NHL API and keeps a local cache."""
import time
from datetime import date, timedelta

import pandas as pd
import requests

from . import config

BASE = "https://api-web.nhle.com/v1"
FINAL_STATES = {"FINAL", "OFF"}
COLUMNS = ["game_id", "season", "game_type", "date", "start_utc", "state",
           "home", "away", "home_score", "away_score", "last_period"]


def _get(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"NHL API request failed: {url}")


def _parse_game(day, g):
    home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
    return {
        "game_id": g["id"],
        "season": g.get("season"),
        "game_type": g.get("gameType"),
        "date": day,
        "start_utc": g.get("startTimeUTC"),
        "state": g.get("gameState"),
        "home": home.get("abbrev"),
        "away": away.get("abbrev"),
        "home_score": home.get("score"),
        "away_score": away.get("score"),
        "last_period": (g.get("gameOutcome") or {}).get("lastPeriodType"),
    }


def fetch_range(start: str, end: str) -> pd.DataFrame:
    """All regular-season and playoff games from start to end (YYYY-MM-DD)."""
    rows, cursor = [], start
    while cursor and cursor <= end:
        data = _get(f"{BASE}/schedule/{cursor}")
        for day in data.get("gameWeek", []):
            for g in day.get("games", []):
                if g.get("gameType") in (2, 3):  # 2 = regular season, 3 = playoffs
                    rows.append(_parse_game(day["date"], g))
        nxt = data.get("nextStartDate")
        if not nxt or nxt <= cursor:
            break
        cursor = nxt
    return pd.DataFrame(rows, columns=COLUMNS)


def update_games(today: date) -> pd.DataFrame:
    """Refresh the cache: re-pull anything not final yet, plus the next few days."""
    end = (today + timedelta(days=3)).isoformat()
    if config.GAMES_CSV.exists():
        cached = pd.read_csv(config.GAMES_CSV)
        recent = cached["date"] >= (today - timedelta(days=10)).isoformat()
        unfinished = cached[recent & ~cached["state"].isin(FINAL_STATES)]
        start = unfinished["date"].min() if len(unfinished) else cached["date"].max()
        start = min(start, (today - timedelta(days=2)).isoformat())
    else:
        cached = pd.DataFrame(columns=COLUMNS)
        start = config.HISTORY_START

    fresh = fetch_range(start, end)
    games = pd.concat([cached, fresh]).drop_duplicates("game_id", keep="last")
    games = games.sort_values(["date", "start_utc", "game_id"]).reset_index(drop=True)
    config.DATA_DIR.mkdir(exist_ok=True)
    games.to_csv(config.GAMES_CSV, index=False)
    return games
