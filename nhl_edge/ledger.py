"""Bet log: records flagged bets, near-closing lines (for CLV), and results."""
import pandas as pd

from . import config
from .nhl_api import FINAL_STATES
from .odds import american_to_decimal

BET_COLS = ["logged_at", "date", "game_id", "matchup", "pick", "side", "book", "odds",
            "model_prob", "market_prob", "prob", "ev", "stake", "close_fair", "clv",
            "result", "profit"]


def load_bets() -> pd.DataFrame:
    if config.BETS_CSV.exists():
        return pd.read_csv(config.BETS_CSV)
    return pd.DataFrame(columns=BET_COLS)


def save_bets(bets: pd.DataFrame):
    bets[BET_COLS].to_csv(config.BETS_CSV, index=False)


def add_bets(bets: pd.DataFrame, new: list[dict]) -> pd.DataFrame:
    """Logs new picks once; a pick already logged for that game keeps its original odds."""
    have = set(zip(bets["game_id"].astype(int), bets["side"])) if len(bets) else set()
    fresh = [b for b in new if (int(b["game_id"]), b["side"]) not in have]
    if fresh:
        bets = pd.concat([bets, pd.DataFrame(fresh)], ignore_index=True)
    return bets


def update_closing(bets: pd.DataFrame, odds_by_game: dict, started: set) -> pd.DataFrame:
    """For games not started yet, store the latest consensus fair price and the CLV."""
    for i, b in bets.iterrows():
        gid = int(b["game_id"])
        if gid in started or gid not in odds_by_game or not pd.isna(b["result"]):
            continue
        o = odds_by_game[gid]
        fair = o["market_home"] if b["side"] == "home" else 1 - o["market_home"]
        bets.at[i, "close_fair"] = round(fair, 4)
        bets.at[i, "clv"] = round(american_to_decimal(b["odds"]) * fair - 1, 4)
    return bets


def grade(bets: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    finals = games[games["state"].isin(FINAL_STATES)].set_index("game_id")
    for i, b in bets.iterrows():
        gid = int(b["game_id"])
        if not pd.isna(b["result"]) or gid not in finals.index:
            continue
        g = finals.loc[gid]
        home_won = g["home_score"] > g["away_score"]
        won = home_won if b["side"] == "home" else not home_won
        bets.at[i, "result"] = "W" if won else "L"
        profit = b["stake"] * (american_to_decimal(b["odds"]) - 1) if won else -b["stake"]
        bets.at[i, "profit"] = round(profit, 3)
    return bets


def summary(bets: pd.DataFrame) -> dict:
    done = bets.dropna(subset=["result"])
    clv = bets.dropna(subset=["clv"])
    staked = float(done["stake"].sum()) if len(done) else 0.0
    profit = float(done["profit"].sum()) if len(done) else 0.0
    return {
        "bets": int(len(done)),
        "pending": int(len(bets) - len(done)),
        "wins": int((done["result"] == "W").sum()),
        "losses": int((done["result"] == "L").sum()),
        "units_staked": round(staked, 2),
        "units_profit": round(profit, 2),
        "roi": round(profit / staked, 4) if staked else None,
        "avg_clv": round(float(clv["clv"].mean()), 4) if len(clv) else None,
        "beat_close_pct": round(float((clv["clv"] > 0).mean()), 3) if len(clv) else None,
    }


def log_predictions(rows: list[dict]):
    if not rows:
        return
    new = pd.DataFrame(rows)
    if config.PREDICTIONS_CSV.exists():
        new = pd.concat([pd.read_csv(config.PREDICTIONS_CSV), new])
    new = new.drop_duplicates(["game_id", "run"], keep="last")
    new.to_csv(config.PREDICTIONS_CSV, index=False)
