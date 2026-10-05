"""Win-probability model plus an honest walk-forward backtest."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import BASE_FEATURES, XG_FEATURES

STREAK = ["streak_diff"]


def _new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000))


def training_rows(feats: pd.DataFrame) -> pd.DataFrame:
    done = feats.dropna(subset=["home_win"])
    return done.iloc[config.BURN_IN_GAMES:]


def choose_features(feats: pd.DataFrame, xg_coverage: float) -> list[str]:
    return BASE_FEATURES + (XG_FEATURES if xg_coverage >= 0.9 else [])


def fit(train: pd.DataFrame, features: list[str]):
    m = _new_model()
    m.fit(train[features], train["home_win"].astype(int))
    return m


def predict(model, rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(rows[features])[:, 1]


def _score(y, p):
    return {"log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "accuracy": round(float(np.mean((p > 0.5) == y)), 4)}


def backtest(feats: pd.DataFrame, features: list[str]) -> dict:
    """Each season is predicted by a model trained only on earlier seasons."""
    data = training_rows(feats)
    seasons = sorted(data["season"].unique())
    if len(seasons) < 2:
        return {"note": "Need at least two seasons of history to backtest."}

    per_season, all_y, all_p, all_elo, all_base, all_nostreak = [], [], [], [], [], []
    for s in seasons[1:]:
        train, test = data[data["season"] < s], data[data["season"] == s]
        if len(test) < 50:
            continue
        y = test["home_win"].astype(int).values
        p = predict(fit(train, features), test, features)
        p_elo = predict(fit(train, ["elo_diff"]), test, ["elo_diff"])
        no_streak = [f for f in features if f not in STREAK]
        p_ns = predict(fit(train, no_streak), test, no_streak)
        p_base = np.full(len(y), train["home_win"].mean())
        per_season.append({"season": f"{str(s)[:4]}-{str(s)[6:]}", "games": int(len(y)),
                           **_score(y, p)})
        all_y.append(y); all_p.append(p); all_elo.append(p_elo)
        all_base.append(p_base); all_nostreak.append(p_ns)

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    overall = _score(y, p)

    # Calibration: when the model says 60%, does the team win ~60% of the time?
    bins = np.clip((p * 10).astype(int), 0, 9)
    calibration = [{"predicted": round(float(p[bins == b].mean()), 3),
                    "actual": round(float(y[bins == b].mean()), 3),
                    "games": int((bins == b).sum())}
                   for b in range(10) if (bins == b).sum() >= 30]

    # Which inputs matter? Coefficients of a model trained on everything (standardized).
    full = fit(data, features)
    coefs = full.named_steps["logisticregression"].coef_[0]
    importance = sorted(({"feature": f, "weight": round(float(c), 4)}
                         for f, c in zip(features, coefs)),
                        key=lambda d: -abs(d["weight"]))

    ll_ns = log_loss(y, np.concatenate(all_nostreak))
    return {
        "features": features,
        "overall": overall,
        "baselines": {
            "always_home_rate": _score(y, np.concatenate(all_base)),
            "elo_only": _score(y, np.concatenate(all_elo)),
        },
        "streak_effect": {
            "log_loss_without_streaks": round(float(ll_ns), 4),
            "log_loss_with_streaks": overall["log_loss"],
            "helps": bool(overall["log_loss"] < ll_ns - 0.0005),
        },
        "per_season": per_season,
        "calibration": calibration,
        "importance": importance,
        "note": ("This measures prediction quality, not betting ROI. Free historical odds "
                 "aren't available, so real ROI and closing line value are tracked going "
                 "forward in the bet log."),
    }
