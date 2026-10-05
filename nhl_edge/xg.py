"""Optional: game-by-game expected goals (xG) from MoneyPuck.

xG measures shot quality, and it predicts future results better than goals do.
If MoneyPuck is unreachable or changes its file format, the model just runs without it.
"""
import io

import pandas as pd
import requests

from . import config
from .teams import MONEYPUCK_ABBREV


def load_xg(teams) -> pd.DataFrame:
    """Returns columns: game_id, team, xgf, xga, xgf5, xga5 (empty frame if unavailable)."""
    frames = []
    for team in sorted(set(teams)):
        mp = MONEYPUCK_ABBREV.get(team, team)
        try:
            r = requests.get(config.MONEYPUCK_TEAM_URL.format(team=mp), timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
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
            print(f"  xG unavailable for {team}: {type(e).__name__}")
    if not frames:
        return pd.DataFrame(columns=["game_id", "team", "xgf", "xga", "xgf5", "xga5"])
    return pd.concat(frames, ignore_index=True)
