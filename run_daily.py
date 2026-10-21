"""Daily NHL edge run.

  python run_daily.py picks   # morning: update data, retrain, find value, log new bets
  python run_daily.py close   # pre-game: grab near-closing lines for CLV, refresh the site

Set ODDS_API_KEY in your environment (GitHub Actions reads it from repo secrets).
"""
import json
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from nhl_edge import config, diag, ledger, model
from nhl_edge.features import build_features
from nhl_edge.goalies import todays_starters, update_goalie_games
from nhl_edge.nhl_api import update_games
from nhl_edge.odds import american_to_decimal, fetch_odds
from nhl_edge.xg import load_xg

ET, CT = ZoneInfo("America/New_York"), ZoneInfo("America/Chicago")


def _match_odds(row, odds):
    start = pd.Timestamp(row["start_utc"])
    for o in odds:
        if o["home"] == row["home"] and o["away"] == row["away"]:
            if abs((pd.Timestamp(o["commence"]) - start).total_seconds()) < 12 * 3600:
                return o
    return None


def _evaluate(p_side, american):
    d = american_to_decimal(american)
    ev = p_side * d - 1
    kelly = max(ev / (d - 1), 0)
    stake = min(config.KELLY_FRACTION * kelly * 100, config.MAX_STAKE_UNITS)
    return ev, round(stake, 1)


def main(mode: str):
    now = datetime.now(timezone.utc)
    today = now.astimezone(ET).date()
    print(f"[{mode}] {now.astimezone(CT):%Y-%m-%d %H:%M} CT")

    print("Updating games from the NHL API...")
    games = update_games(today)

    print("Loading expected-goals data...")
    done = games[games["state"].isin(["FINAL", "OFF"])]
    xg = load_xg(set(games["home"]) | set(games["away"]))
    reg = done[(done["game_type"] == 2) & (done["season"] >= done["season"].max() - 10001)]
    covered = set(zip(xg["game_id"].astype(int), xg["team"])) if len(xg) else set()
    side_rows = [(int(g), t) for g, h, a in zip(reg["game_id"], reg["home"], reg["away"])
                 for t in (h, a)]
    xg_cov = sum(k in covered for k in side_rows) / max(len(side_rows), 1)
    by_team = pd.Series([k in covered for k in side_rows],
                        index=[t for _, t in side_rows]).groupby(level=0).mean().sort_values()
    if len(by_team) and by_team.iloc[0] < 0.9:
        worst = ", ".join(f"{t} {v:.0%}" for t, v in by_team.head(5).items())
        diag.note(f"xG: lowest team coverage: {worst}")
    diag.note(f"xG coverage of recent games: {xg_cov:.0%} (needs 75% to be used).")

    print("Loading goalie starts...")
    goalie_games = update_goalie_games(games)
    starter_keys = set(zip(goalie_games.loc[goalie_games["starter"], "game_id"].astype(int),
                           goalie_games.loc[goalie_games["starter"], "team"]))
    g_cov = sum(k in starter_keys for k in side_rows) / max(len(side_rows), 1)
    diag.note(f"Goalies: starter known for {g_cov:.0%} of recent games (needs 75% to be used).")
    today_games = games[games["date"] == today.isoformat()]
    starters = todays_starters(today_games, goalie_games, games, today) if len(today_games) else {}

    feats = build_features(games, xg, goalie_games, starters)
    features = model.choose_features(feats, xg_cov, g_cov)
    train = model.training_rows(feats)
    print(f"Training on {len(train)} games with {len(features)} features...")
    clf = model.fit(train, features)
    bt = model.backtest(feats, features)

    upcoming = feats[(feats["date"] == today.isoformat())].copy()
    upcoming["model_home"] = model.predict(clf, upcoming, features) if len(upcoming) else []

    print("Fetching odds...")
    odds, credits = fetch_odds()

    games_out, new_bets, pred_rows, odds_by_game = [], [], [], {}
    for _, r in upcoming.iterrows():
        o = _match_odds(r, odds)
        start_ct = pd.Timestamp(r["start_utc"]).tz_convert(CT)
        g = {"game_id": int(r["game_id"]), "time": start_ct.strftime("%-I:%M %p"),
             "home": r["home"], "away": r["away"], "state": r["state"],
             "home_streak": int(r["home_streak"]), "away_streak": int(r["away_streak"]),
             "model_home": round(float(r["model_home"]), 4), "pick": None}
        for side in ("home", "away"):
            st = starters.get((int(r["game_id"]), r[side]), {})
            g[f"{side}_goalie"] = st.get("name")
            g[f"{side}_goalie_status"] = st.get("status")
            g[f"{side}_goalie_gsax"] = round(float(r[f"{side}_goalie"]), 3)
            g[f"{side}_goalie_swap"] = round(float(r[f"{side}_goalie_swap"]), 3)
        pred = {"date": r["date"], "game_id": int(r["game_id"]), "run": mode,
                "home": r["home"], "away": r["away"], "model_home": g["model_home"],
                "market_home": None}
        if o:
            odds_by_game[int(r["game_id"])] = o
            w = config.MODEL_WEIGHT
            p_home = w * r["model_home"] + (1 - w) * o["market_home"]
            ev_h, stake_h = _evaluate(p_home, o["best_home"])
            ev_a, stake_a = _evaluate(1 - p_home, o["best_away"])
            g.update({"market_home": round(o["market_home"], 4), "blend_home": round(p_home, 4),
                      "best_home": o["best_home"], "best_home_book": o["best_home_book"],
                      "best_away": o["best_away"], "best_away_book": o["best_away_book"],
                      "ev_home": round(ev_h, 4), "ev_away": round(ev_a, 4)})
            pred["market_home"] = round(o["market_home"], 4)

            side, ev, stake = ("home", ev_h, stake_h) if ev_h >= ev_a else ("away", ev_a, stake_a)
            price = o["best_home"] if side == "home" else o["best_away"]
            if (ev >= config.MIN_EV and price <= config.MAX_DOG_ODDS and stake >= 0.1
                    and r["state"] in ("FUT", "PRE")):
                team = r[side]
                g["pick"] = {"team": team, "side": side, "odds": int(price), "ev": round(ev, 4),
                             "stake": stake}
                new_bets.append({
                    "logged_at": now.astimezone(CT).strftime("%Y-%m-%d %H:%M"),
                    "date": r["date"], "game_id": int(r["game_id"]),
                    "matchup": f"{r['away']} @ {r['home']}", "pick": team, "side": side,
                    "book": o[f"best_{side}_book"], "odds": int(price),
                    "model_prob": round(r["model_home"] if side == "home" else 1 - r["model_home"], 4),
                    "market_prob": round(o["market_home"] if side == "home" else 1 - o["market_home"], 4),
                    "prob": round(p_home if side == "home" else 1 - p_home, 4),
                    "ev": round(ev, 4), "stake": stake,
                    "goalies": f"{g['away_goalie'] or '?'} ({g['away_goalie_status'] or '-'}) @ "
                               f"{g['home_goalie'] or '?'} ({g['home_goalie_status'] or '-'})"})
        games_out.append(g)
        pred_rows.append(pred)

    bets = ledger.load_bets()
    if mode == "picks":
        bets = ledger.add_bets(bets, new_bets)
        print(f"  {len(new_bets)} value bets today")
    started = {int(g) for g, s in zip(games["game_id"], games["state"])
               if s not in ("FUT", "PRE")}
    for step in (lambda b: ledger.update_closing(b, odds_by_game, started),
                 lambda b: ledger.grade(b, games)):
        try:
            bets = step(bets)
        except Exception as e:   # never let one step block the rest of the run
            diag.note(f"Bet log step failed: {type(e).__name__}: {str(e)[:150]}")
    ledger.save_bets(bets)
    ledger.log_predictions(pred_rows)

    recent_bets = bets.tail(60).iloc[::-1].astype(object).where(bets.tail(60).iloc[::-1].notna(), None)
    site = {
        "updated": now.astimezone(CT).strftime("%a %b %-d, %-I:%M %p CT"),
        "run": mode, "date": today.isoformat(),
        "has_odds": bool(odds),
        "diagnostics": diag.MESSAGES, "odds_credits_left": credits,
        "xg_used": any(f.startswith("xg") for f in features),
        "goalies_used": "goalie_swap_diff" in features,
        "settings": {"model_weight": config.MODEL_WEIGHT, "min_ev": config.MIN_EV,
                     "kelly_fraction": config.KELLY_FRACTION,
                     "max_stake": config.MAX_STAKE_UNITS},
        "games": games_out,
        "record": ledger.summary(bets),
        "bets": recent_bets.to_dict("records"),
        "backtest": bt,
    }
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.SITE_DATA_DIR / "site.json").write_text(json.dumps(site, indent=1, default=str))
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "picks")
