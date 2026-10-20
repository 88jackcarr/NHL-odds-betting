"""Sportsbook moneyline odds from The Odds API, with the bookmaker's margin (vig) removed."""
import os
from statistics import median

import requests

from . import config
from .diag import note
from .teams import abbrev_for


def american_to_decimal(a: float) -> float:
    return 1 + (a / 100 if a > 0 else 100 / -a)


def implied(a: float) -> float:
    return 1 / american_to_decimal(a)


def fetch_odds():
    """Returns (list of games with odds, credits remaining) or ([], None) without a key."""
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        note("Odds: no ODDS_API_KEY secret reached the run. Check the secret's exact name "
             "under Settings > Secrets and variables > Actions > Repository secrets.")
        return [], None
    try:
        r = requests.get(config.ODDS_URL, timeout=30, params={
            "apiKey": key, "regions": "us", "markets": "h2h", "oddsFormat": "american"})
    except requests.RequestException as e:
        note(f"Odds: couldn't reach The Odds API ({type(e).__name__}).")
        return [], None
    if r.status_code != 200:
        note(f"Odds: The Odds API returned error {r.status_code}: {r.text[:200].replace(key, '***')} "
             f"(key length {len(key)}).")
        return [], None
    events = r.json()
    games = [g for g in (_parse(e) for e in events) if g]
    note(f"Odds: loaded {len(games)} NHL games with prices ({len(events)} listed by the API).")
    return games, r.headers.get("x-requests-remaining")


def _parse(event):
    home, away = abbrev_for(event["home_team"]), abbrev_for(event["away_team"])
    if not home or not away:
        note(f"Odds: unknown team name {event['home_team']} / {event['away_team']}")
        return None
    books = []
    for b in event.get("bookmakers", []):
        for m in b.get("markets", []):
            if m["key"] != "h2h":
                continue
            prices = {o["name"]: o["price"] for o in m["outcomes"]}
            hp, ap = prices.get(event["home_team"]), prices.get(event["away_team"])
            if hp is None or ap is None:
                continue
            ih, ia = implied(hp), implied(ap)
            books.append({"book": b["title"], "home": hp, "away": ap,
                          "fair_home": ih / (ih + ia)})   # vig removed
    if not books:
        return None
    best_home = max(books, key=lambda x: american_to_decimal(x["home"]))
    best_away = max(books, key=lambda x: american_to_decimal(x["away"]))
    return {
        "home": home, "away": away, "commence": event["commence_time"],
        "market_home": median(b["fair_home"] for b in books),  # consensus fair probability
        "best_home": best_home["home"], "best_home_book": best_home["book"],
        "best_away": best_away["away"], "best_away_book": best_away["book"],
        "n_books": len(books),
    }
