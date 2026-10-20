"""Starting goalies: who started past games (NHL API box scores) and who starts today.

Today's starters come from Daily Faceoff when posted ("Confirmed" / "Likely"). If that
page is unavailable or a game isn't listed, the starter is projected from recent starts:
the team's most-used goalie, or the backup on the second night of a back-to-back.
"""
import json
import time
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import requests

from . import config
from .diag import note
from .teams import abbrev_for

BOX_URL = "https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore"
DF_URL = "https://www.dailyfaceoff.com/starting-goalies/{day}"
UA = {"User-Agent": "Mozilla/5.0 (nhl-edge research script)"}
COLS = ["game_id", "team", "goalie_id", "name", "starter", "toi_sec", "shots", "ga"]


def _toi(s):
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except ValueError:
        return 0


_session = requests.Session()
_session.headers.update(UA)


def _box(gid):
    data = None
    for attempt in range(5):
        try:
            r = _session.get(BOX_URL.format(gid=gid), timeout=30)
            if r.status_code == 200:
                data = r.json()
                break
            if r.status_code == 404:
                return None
            wait = float(r.headers.get("Retry-After", 0) or 0)
            time.sleep(max(wait, 2 * (attempt + 1)))   # throttled: back off
        except Exception:
            time.sleep(2 * (attempt + 1))
    if data is None:
        return None
    rows = []
    stats = data.get("playerByGameStats", {})
    for side in ("homeTeam", "awayTeam"):
        team = (data.get(side) or {}).get("abbrev")
        goalies = (stats.get(side) or {}).get("goalies", [])
        if goalies and not any(g.get("starter") for g in goalies):
            top = max(goalies, key=lambda g: _toi(g.get("toi")))   # fallback: most ice time
            top["starter"] = True
        for g in goalies:
            toi = _toi(g.get("toi"))
            if toi == 0 and not g.get("starter"):
                continue
            shots = g.get("shotsAgainst")
            if shots is None:   # older format: "saves/shots"
                try:
                    shots = int(str(g.get("saveShotsAgainst", "0/0")).split("/")[1])
                except (IndexError, ValueError):
                    shots = 0
            rows.append({"game_id": gid, "team": team, "goalie_id": g.get("playerId"),
                         "name": (g.get("name") or {}).get("default", ""),
                         "starter": bool(g.get("starter")), "toi_sec": toi,
                         "shots": int(shots or 0), "ga": int(g.get("goalsAgainst") or 0)})
    return rows


def update_goalie_games(games: pd.DataFrame) -> pd.DataFrame:
    """Box-score goalie lines for every finished game, cached in data/goalie_games.csv."""
    path = config.DATA_DIR / "goalie_games.csv"
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)
    done = games[games["state"].isin(["FINAL", "OFF"])]["game_id"].astype(int)
    have = set(cached["game_id"].astype(int))
    todo = [g for g in done if g not in have]
    if todo:
        print(f"  Fetching {len(todo)} box scores for goalie starts...")
        failed = 0
        for i in range(0, len(todo), 300):
            chunk = todo[i:i + 300]
            with ThreadPoolExecutor(max_workers=3) as ex:
                results = list(ex.map(_box, chunk))
            new = [r for rows in results if rows for r in rows]
            failed += sum(1 for r in results if not r)
            if new:
                cached = pd.concat([cached, pd.DataFrame(new, columns=COLS)], ignore_index=True)
                cached.to_csv(path, index=False)
            print(f"    {min(i + 300, len(todo))}/{len(todo)} done, {failed} failed")
        if failed:
            note(f"Goalies: {failed} box scores couldn't be loaded (will retry next run).")
    cached["starter"] = cached["starter"].astype(str).str.lower().eq("true")
    return cached


# ---------- Today's starters ----------

def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s).strip()


def _last(name):
    parts = _norm(name).split()
    return parts[-1] if parts else ""


def _find(d, *needles):
    """Value of the first key containing all needles (case-insensitive)."""
    for k, v in d.items():
        kl = k.lower()
        if all(n in kl for n in needles):
            return v
    return None


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def daily_faceoff(day: date) -> dict:
    """{team_abbrev: (goalie name, status)} from Daily Faceoff, or {} if unavailable."""
    try:
        r = requests.get(DF_URL.format(day=day.isoformat()), timeout=30, headers=UA)
        r.raise_for_status()
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
        if not m:
            raise ValueError("page format changed")
        data = json.loads(m.group(1))
    except Exception as e:
        note(f"Goalies: Daily Faceoff unavailable ({type(e).__name__}); using projected starters.")
        return {}
    out = {}
    for d in _walk(data):
        for side in ("home", "away"):
            gname = _find(d, side, "goalie", "name")
            tname = _find(d, side, "team", "name") or _find(d, side, "team", "slug")
            if not isinstance(gname, str) or not isinstance(tname, str):
                continue
            team = abbrev_for(tname.replace("-", " "))
            status = _find(d, side, "strength", "name") or _find(d, side, "status") or ""
            if team:
                out[team] = (gname, str(status).title() or "Reported")
    note(f"Goalies: Daily Faceoff listed starters for {len(out)} teams.")
    return out


def todays_starters(today_games: pd.DataFrame, goalie_games: pd.DataFrame,
                    games: pd.DataFrame, today: date) -> dict:
    """{(game_id, team): {"goalie_id", "name", "status"}} for today's games."""
    reported = daily_faceoff(today)
    starts = goalie_games[goalie_games["starter"].astype(bool)].merge(
        games[["game_id", "date", "season"]], on="game_id")
    starts = starts.sort_values("date")
    out = {}
    for g in today_games.itertuples(index=False):
        for team in (g.home, g.away):
            mine = starts[starts["team"] == team]
            recent = mine.tail(10)
            pick, status = None, "Projected"
            if team in reported:
                name, status = reported[team]
                pool = mine.tail(60).drop_duplicates("goalie_id", keep="last")
                match = pool[pool["name"].map(_last) == _last(name)]
                if len(match):
                    pick = (int(match.iloc[-1]["goalie_id"]), name)
                else:
                    pick = (None, name)   # call-up with no NHL starts: average rating
            if pick is None and len(recent):
                counts = recent["goalie_id"].value_counts()
                gid = int(counts.index[0])
                last = mine.iloc[-1]
                played_yesterday = (date.fromisoformat(last["date"]) - today).days == -1
                if played_yesterday and int(last["goalie_id"]) == gid and len(counts) > 1:
                    gid = int(counts.index[1])   # back-to-back: backup usually starts
                    status = "Projected (B2B backup)"
                name = recent[recent["goalie_id"] == gid]["name"].iloc[-1]
                pick = (gid, name)
            if pick:
                out[(int(g.game_id), team)] = {"goalie_id": pick[0], "name": pick[1],
                                               "status": status}
    return out
