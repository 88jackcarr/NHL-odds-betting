"""Bet log: records flagged bets, near-closing lines (for CLV), and results."""
import pandas as pd

from . import config
from .nhl_api import FINAL_STATES
from .odds import american_to_decimal

BET_COLS = ["logged_at", "date", "game_id", "matchup", "market", "line", "pick", "side",
            "book", "odds",
            "model_prob", "market_prob", "prob", "ev", "stake", "goalies", "close_fair", "clv",
            "result", "profit"]


TEXT_COLS = ["logged_at", "date", "matchup", "market", "pick", "side", "book", "goalies", "result"]


def _types(bets: pd.DataFrame) -> pd.DataFrame:
    bets = bets.reindex(columns=BET_COLS)
    for c in TEXT_COLS:   # text columns that may be all-blank must not be read as numbers
        bets[c] = bets[c].astype(object)
    for c in ("line", "close_fair", "clv", "profit"):
        bets[c] = pd.to_numeric(bets[c], errors="coerce")
    bets["market"] = bets["market"].where(bets["market"].notna(), "ML")   # older bets
    return bets


def load_bets() -> pd.DataFrame:
    if config.BETS_CSV.exists():
        return _types(pd.read_csv(config.BETS_CSV))
    return _types(pd.DataFrame(columns=BET_COLS))


def save_bets(bets: pd.DataFrame):
    bets.reindex(columns=BET_COLS).to_csv(config.BETS_CSV, index=False)


def add_bets(bets: pd.DataFrame, new: list[dict]) -> pd.DataFrame:
    """Logs one pick per game per market; a logged pick keeps its original odds."""
    have = set(zip(bets["game_id"].astype(int), bets["market"])) if len(bets) else set()
    fresh = [b for b in new if (int(b["game_id"]), b.get("market", "ML")) not in have]
    if fresh:
        bets = _types(pd.concat([bets, _types(pd.DataFrame(fresh))], ignore_index=True))
    return bets


def _total_fair_at(bet_line, side, close, curve):
    """Market's fair probability for our side at OUR line, using the latest market line.

    If the line moved (say 6.5 -> 6), shift the market's probability by how much our
    model's curve says that half-goal is worth for this game.
    """
    from .totals import line_probs
    p_over_close = close["market_over"]
    if close["line"] != bet_line:
        if curve is None:
            return None
        o_b, u_b, _ = line_probs(curve, bet_line)
        o_c, u_c, _ = line_probs(curve, close["line"])
        p_over_close = min(max(p_over_close + (o_b / (o_b + u_b) - o_c / (o_c + u_c)), 0.01), 0.99)
    return p_over_close if side == "over" else 1 - p_over_close


def update_closing(bets: pd.DataFrame, odds_by_game: dict, started: set,
                   curves: dict | None = None) -> pd.DataFrame:
    """For games not started yet, store the latest consensus fair price and the CLV."""
    curves = curves or {}
    for i, b in bets.iterrows():
        gid = int(b["game_id"])
        if gid in started or gid not in odds_by_game or not pd.isna(b["result"]):
            continue
        o = odds_by_game[gid]
        if b["market"] == "Total":
            if not o.get("total"):
                continue
            fair = _total_fair_at(b["line"], b["side"], o["total"], curves.get(gid))
            if fair is None:
                continue
            bets.at[i, "close_fair"] = round(fair, 4)
            bets.at[i, "clv"] = round(american_to_decimal(b["odds"]) * fair - 1, 4)
            continue
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
        if b["market"] == "Total":
            total = g["home_score"] + g["away_score"]
            if total == b["line"]:
                bets.at[i, "result"], bets.at[i, "profit"] = "P", 0.0   # push: stake back
                continue
            won = (total > b["line"]) == (b["side"] == "over")
        else:
            home_won = g["home_score"] > g["away_score"]
            won = home_won if b["side"] == "home" else not home_won
        bets.at[i, "result"] = "W" if won else "L"
        profit = b["stake"] * (american_to_decimal(b["odds"]) - 1) if won else -b["stake"]
        bets.at[i, "profit"] = round(profit, 3)
    return bets


def summary_by_market(bets: pd.DataFrame) -> dict:
    out = {"all": summary(bets)}
    for m, label in (("ML", "moneyline"), ("Total", "totals")):
        out[label] = summary(bets[bets["market"] == m])
    return out


def summary(bets: pd.DataFrame) -> dict:
    done = bets.dropna(subset=["result"])
    clv = bets.dropna(subset=["clv"])
    staked = float(done.loc[done["result"] != "P", "stake"].sum()) if len(done) else 0.0
    profit = float(done["profit"].sum()) if len(done) else 0.0
    return {
        "bets": int(len(done)),
        "pending": int(len(bets) - len(done)),
        "wins": int((done["result"] == "W").sum()),
        "pushes": int((done["result"] == "P").sum()),
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
