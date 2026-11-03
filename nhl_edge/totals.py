"""Over/under model.

Instead of assuming total goals follow a textbook distribution (NHL totals are more tightly
clustered than Poisson), it fits one calibrated model per threshold: P(total >= k) for
k = 3..10, then reads any line off that curve. Half lines (6.5) never push; whole lines (6)
push when the total lands exactly on the number.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import TOTAL_BASE, TOTAL_XG

KS = list(range(3, 11))


def choose_total_features(xg_used: bool) -> list[str]:
    return TOTAL_BASE + (TOTAL_XG if xg_used else [])


def _rows(feats: pd.DataFrame) -> pd.DataFrame:
    return feats.dropna(subset=["total_goals"]).iloc[config.BURN_IN_GAMES:]


class TotalsModel:
    def fit(self, train: pd.DataFrame, features: list[str]):
        self.features = features
        self.models = {}
        for k in KS:
            y = (train["total_goals"] >= k).astype(int)
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
            self.models[k] = m.fit(train[features], y)
        self.mean = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=500))
        self.mean.fit(train[features], train["total_goals"])
        return self

    def curve(self, rows: pd.DataFrame) -> np.ndarray:
        """P(total >= k) for each row and k in KS, forced to decrease with k."""
        c = np.column_stack([self.models[k].predict_proba(rows[self.features])[:, 1] for k in KS])
        return np.minimum.accumulate(c, axis=1)

    def expected(self, rows: pd.DataFrame) -> np.ndarray:
        return self.mean.predict(rows[self.features])


def at_least(curve_row, k: int) -> float:
    if k <= KS[0]:
        return float(curve_row[0]) if k == KS[0] else 1.0
    if k > KS[-1]:
        return float(curve_row[-1]) * 0.5 ** (k - KS[-1])   # rare; tail fade
    return float(curve_row[KS.index(k)])


def line_probs(curve_row, line: float) -> tuple[float, float, float]:
    """(P over, P under, P push) for a total line like 6.5 or 6."""
    if line % 1:   # half line
        over = at_least(curve_row, int(np.ceil(line)))
        return over, 1 - over, 0.0
    L = int(line)
    over = at_least(curve_row, L + 1)
    under = 1 - at_least(curve_row, L)
    return over, under, max(1 - over - under, 0.0)


def backtest(feats: pd.DataFrame, features: list[str]) -> dict:
    """Walk-forward check: does the model beat 'every game is average' at 5.5 and 6.5?"""
    data = _rows(feats)
    seasons = sorted(data["season"].unique())
    if len(seasons) < 2:
        return {"note": "Need at least two seasons of history."}
    out_lines = {}
    curves, ys, bases = [], [], []
    for s in seasons[1:]:
        train, test = data[data["season"] < s], data[data["season"] == s]
        if len(test) < 50:
            continue
        m = TotalsModel().fit(train, features)
        curves.append(m.curve(test))
        ys.append(test["total_goals"].values)
        bases.append(np.tile([(train["total_goals"] >= k).mean() for k in KS], (len(test), 1)))
    c, y, b = np.vstack(curves), np.concatenate(ys), np.vstack(bases)
    for line in (5.5, 6.5):
        k = int(np.ceil(line))
        i = KS.index(k)
        over = (y >= k).astype(int)
        p = c[:, i]
        bins = pd.qcut(p, 5, duplicates="drop")
        cal = (pd.DataFrame({"p": p, "o": over}).groupby(bins, observed=True)
               .agg(predicted=("p", "mean"), actual=("o", "mean"), games=("o", "size")))
        out_lines[str(line)] = {
            "log_loss": round(float(log_loss(over, p)), 4),
            "log_loss_baseline": round(float(log_loss(over, b[:, i])), 4),
            "over_rate": round(float(over.mean()), 3),
            "calibration": [{k2: (round(float(v), 3) if k2 != "games" else int(v))
                             for k2, v in r.items()} for r in cal.to_dict("records")],
        }
    return {"features": features, "games": int(len(y)),
            "avg_total": round(float(y.mean()), 2), "lines": out_lines,
            "note": ("Totals are harder to predict than winners. The baseline guesses the "
                     "league-average over rate for every game; beating it by even a little is "
                     "about as good as public models do.")}
