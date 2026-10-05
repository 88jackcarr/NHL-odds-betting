# NHL Edge

A daily NHL moneyline model that compares its win probabilities to sportsbook odds,
flags bets with positive expected value, and tracks results and closing line value (CLV)
on a website that updates itself.

## How it works

1. **Data** — schedule and results from the free NHL API, plus game-by-game expected
   goals (xG) from MoneyPuck when available.
2. **Features** — computed only from games played *before* each game, so nothing leaks
   from the future: Elo rating, goal differential (season and last 10), win % last 10,
   win/loss streak, rest days and back-to-backs, and xG share (season, last 10, 5-on-5).
3. **Model** — logistic regression trained on every game since 2021, retrained daily.
4. **Odds** — moneylines from The Odds API. The vig is removed to get the market's fair
   probability, and the best available price across books is used.
5. **Value** — the model is blended 50/50 with the market (markets are sharp; this avoids
   chasing fake edges). A bet is flagged when expected value is at least +3%.
6. **Sizing** — quarter Kelly, capped at 2 units (1 unit = 1% of bankroll).
7. **Tracking** — every flagged bet is logged with its odds, a near-closing line, CLV, and
   result. Positive average CLV over a few hundred bets is the best sign of a real edge.

## Schedule (GitHub Actions)

- **11:00 AM CT** — retrain, pull odds, log value bets.
- **5:45 PM CT** — pull near-closing odds for CLV, refresh the site.
- Results are graded automatically on the next run.

Run it by hand anytime: **Actions → Daily NHL picks → Run workflow**.

## Setup

1. Add your Odds API key: **Settings → Secrets and variables → Actions → New repository
   secret**, name `ODDS_API_KEY`.
2. Turn on the website: **Settings → Pages → Build and deployment → Deploy from a branch**,
   branch `main`, folder `/docs`.
3. Run the workflow once from the Actions tab. The first run pulls several seasons of
   history and takes a few minutes.

## Tuning

All the knobs are in `nhl_edge/config.py`: model vs. market weight, minimum EV,
Kelly fraction, max stake, and the biggest underdog it will bet.

## Honest limits

- The backtest measures prediction quality, not ROI. Free historical odds don't exist,
  so real ROI and CLV are measured going forward.
- Starting goalies aren't in the model yet. The free NHL API doesn't publish confirmed
  starters before games, and goalie news moves lines. This is the most valuable upgrade.
- The 5:45 PM snapshot is "near close" for most games, but not for late West Coast games.
- Expect long losing streaks even with a real edge. Hockey is high-variance.
