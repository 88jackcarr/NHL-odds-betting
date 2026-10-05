"""Builds pre-game features for every game, using only information from earlier games.

Walking through games in date order and computing features *before* updating with each
result is what keeps the backtest honest: the model never sees the future.
"""
import math
from collections import defaultdict, deque
from datetime import date

import numpy as np
import pandas as pd

from . import config
from .nhl_api import FINAL_STATES

BASE_FEATURES = ["elo_diff", "gd_season_diff", "gd_l10_diff", "win_l10_diff",
                 "streak_diff", "rest_diff", "home_b2b", "away_b2b"]
XG_FEATURES = ["xg_season_diff", "xg_l10_diff", "xg5_l10_diff"]


class TeamState:
    def __init__(self, elo=1500.0):
        self.elo = elo
        self.season = None
        self.reset_season()

    def reset_season(self):
        self.n = 0
        self.gd_sum = 0.0
        self.last10 = deque(maxlen=10)       # (goal diff, win)
        self.streak = 0                      # +3 = won 3 straight, -2 = lost 2 straight
        self.last_date = None
        self.xg_for = self.xg_against = 0.0
        self.xg_n = 0
        self.xg10 = deque(maxlen=10)         # (xgf, xga, xgf5, xga5)

    def snapshot(self, game_date):
        k = config.SHRINK_GAMES
        rest = 4 if self.last_date is None else min((game_date - self.last_date).days, 4)
        l10 = list(self.last10)
        xg10 = list(self.xg10)
        def share(f, a):
            return 0.5 if f + a <= 0 else f / (f + a)
        return {
            "elo": self.elo,
            "gd_season": self.gd_sum / (self.n + k),          # shrunk toward 0 early on
            "gd_l10": (sum(d for d, _ in l10)) / (len(l10) + 3),
            "win_l10": (sum(w for _, w in l10) + 0.5 * 3) / (len(l10) + 3),
            "streak": self.streak,
            "rest": rest,
            "b2b": int(rest == 1),
            # xG share, pulled toward 50% early in the season
            "xg_season": share(self.xg_for + 2.5 * k, self.xg_against + 2.5 * k),
            "xg_l10": share(sum(x[0] for x in xg10) + 7.5, sum(x[1] for x in xg10) + 7.5),
            "xg5_l10": share(sum(x[2] for x in xg10) + 5, sum(x[3] for x in xg10) + 5),
        }

    def update(self, game_date, gf, ga, xg=None):
        win = int(gf > ga)
        self.n += 1
        self.gd_sum += gf - ga
        self.last10.append((gf - ga, win))
        self.streak = (max(self.streak, 0) + 1) if win else (min(self.streak, 0) - 1)
        self.last_date = game_date
        if xg is not None and not any(pd.isna(v) for v in xg):
            self.xg_for += xg[0]
            self.xg_against += xg[1]
            self.xg_n += 1
            self.xg10.append(xg)


def _elo_update(h, a, home_win, gd, overtime):
    expected = 1 / (1 + 10 ** (-(h.elo + config.ELO_HOME_ADV - a.elo) / 400))
    mov = math.log(abs(gd) + 1) + 0.5
    k = config.ELO_K * mov * (0.5 if overtime else 1.0)  # OT/shootout wins are near coin flips
    delta = k * (home_win - expected)
    h.elo += delta
    a.elo -= delta


def build_features(games: pd.DataFrame, xg: pd.DataFrame | None = None) -> pd.DataFrame:
    xg_lookup = {}
    if xg is not None and len(xg):
        for r in xg.itertuples(index=False):
            xg_lookup[(int(r.game_id), r.team)] = (r.xgf, r.xga, r.xgf5, r.xga5)

    teams = defaultdict(TeamState)
    rows = []
    for g in games.sort_values(["date", "start_utc", "game_id"]).itertuples(index=False):
        gdate = date.fromisoformat(g.date)
        h, a = teams[g.home], teams[g.away]
        for t in (h, a):
            if t.season != g.season:   # new season: regress Elo toward average, reset stats
                t.elo = config.ELO_SEASON_CARRY * t.elo + (1 - config.ELO_SEASON_CARRY) * 1500
                t.season = g.season
                t.reset_season()

        hs, as_ = h.snapshot(gdate), a.snapshot(gdate)
        row = {
            "game_id": g.game_id, "season": g.season, "date": g.date, "start_utc": g.start_utc,
            "state": g.state, "home": g.home, "away": g.away,
            "home_elo": hs["elo"], "away_elo": as_["elo"],
            "home_streak": hs["streak"], "away_streak": as_["streak"],
            "elo_diff": hs["elo"] - as_["elo"],
            "gd_season_diff": hs["gd_season"] - as_["gd_season"],
            "gd_l10_diff": hs["gd_l10"] - as_["gd_l10"],
            "win_l10_diff": hs["win_l10"] - as_["win_l10"],
            "streak_diff": float(np.clip(hs["streak"], -6, 6) - np.clip(as_["streak"], -6, 6)),
            "rest_diff": hs["rest"] - as_["rest"],
            "home_b2b": hs["b2b"], "away_b2b": as_["b2b"],
            "xg_season_diff": hs["xg_season"] - as_["xg_season"],
            "xg_l10_diff": hs["xg_l10"] - as_["xg_l10"],
            "xg5_l10_diff": hs["xg5_l10"] - as_["xg5_l10"],
            "home_win": np.nan,
        }

        if g.state in FINAL_STATES and not pd.isna(g.home_score):
            hg, ag = int(g.home_score), int(g.away_score)
            row["home_win"] = int(hg > ag)
            ot = g.last_period in ("OT", "SO")
            _elo_update(h, a, int(hg > ag), hg - ag, ot)
            h.update(gdate, hg, ag, xg_lookup.get((int(g.game_id), g.home)))
            a.update(gdate, ag, hg, xg_lookup.get((int(g.game_id), g.away)))
        rows.append(row)
    return pd.DataFrame(rows)
