"""Optional: game-by-game expected goals (xG) from MoneyPuck.

xG measures shot quality, and it predicts future results better than goals do.
If MoneyPuck is unreachable or changes its file format, the model just runs without it.
"""
import io

import pandas as pd
import requests

from . import config
from .diag import note
from .teams import MONEYPUCK_ABBREV


def _fetch(url):
    """MoneyPuck is flaky around game nights: retry a few times before giving up."""
    import time
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60,
                             headers={"User-Agent": "Mozilla/5.0 (nhl-edge research script)"})
            if r.status_code == 200 and "gameId" in r.text[:2000]:
                return r   # a real CSV, not an error page
            if r.status_code == 404:
                return r
        except requests.RequestException:
            pass
        time.sleep(3 * (attempt + 1))
    return None


def load_xg(teams) -> pd.DataFrame:
    """Fresh MoneyPuck data, falling back to the saved copy for any team that fails."""
    cache_path = config.DATA_DIR / "xg_cache.csv"
    cols = ["game_id", "team", "xgf", "xga", "xgf5", "xga5"]
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=cols)
    fresh = _load_fresh(teams)
    got = set(fresh["team"]) if len(fresh) else set()
    stale = cache[~cache["team"].isin(got)]
    used_cache = sorted(set(stale["team"]) & set(teams))
    if used_cache:
        note(f"xG: used saved data for {len(used_cache)} teams ({', '.join(used_cache[:6])}"
             f"{'...' if len(used_cache) > 6 else ''}).")
    out = pd.concat([fresh, stale], ignore_index=True) if len(stale) else fresh
    if len(out):
        out = out.drop_duplicates(["game_id", "team"], keep="first")
        out = out[out["game_id"].astype(int) >= int(config.HISTORY_START[:4]) * 1000000]
        out.to_csv(cache_path, index=False)
    return out if len(out) else pd.DataFrame(columns=cols)


def _load_fresh(teams) -> pd.DataFrame:
    """Returns columns: game_id, team, xgf, xga, xgf5, xga5 (empty frame if unavailable)."""
    frames, errors = [], []
    for team in sorted(set(teams)):
        try:
            df = None
            # Try both spellings (e.g. LAK and L.A) and keep whichever file is most recent.
            for mp in dict.fromkeys([team, MONEYPUCK_ABBREV.get(team, team)]):
                r = _fetch(config.MONEYPUCK_TEAM_URL.format(team=mp))
                if r is None or r.status_code != 200:
                    continue
                cand = pd.read_csv(io.StringIO(r.text))
                if df is None or cand["gameId"].max() > df["gameId"].max():
                    df = cand
            if df is None:
                raise RuntimeError("no MoneyPuck file found")
            df = df[df["situation"].isin(["all", "5on5"])]
            wide = df.pivot_table(index="gameId", columns="situation",
                                  values=["xGoalsFor", "xGoalsAgainst"], aggfunc="first")
            out = pd.DataFrame({
                "game_id": wide.index.astype(int),
                "team": team,
                "xgf": wide[("xGoalsFor", "all")].values,
                "xga": wide[("xGoalsAgainst", "all")].values,
                "xgf5": wide[("xGoalsFor", "5on5")].values,
                "xga5": wide[("xGoalsAgainst", "5on5")].values,
            })
            frames.append(out)
        except Exception as e:  # missing team file, format change, network error
            errors.append(f"{team}: {type(e).__name__} {str(e)[:80]}")
    if errors:
        note(f"xG: MoneyPuck failed for {len(errors)} teams, e.g. {errors[0]}")
    if frames:
        latest = pd.concat(frames)["game_id"].max()
        note(f"xG: loaded MoneyPuck data for {len(frames)} teams (latest game id {latest}).")
    if not frames:
        return pd.DataFrame(columns=["game_id", "team", "xgf", "xga", "xgf5", "xga5"])
    return pd.concat(frames, ignore_index=True)
