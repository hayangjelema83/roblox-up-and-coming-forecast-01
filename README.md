# Roblox Up-and-Coming Top 10 Forecaster

A momentum Monte-Carlo forecaster for the make.fun bounty
"Predict Roblox's Up-and-Coming top 10" (`roblox-up-and-coming-forecast-01`).

It reads `/input/history.ndjson` (hourly snapshots of Roblox's Up-and-Coming
chart: rank, universe_id, name, playing counts) and `/input/task.json`, and
writes `/output/forecast.json`: for each of the top 10 places, a probability
for each game it names being at that place, plus its guess at the top 10 in
order.

## How it works

The chart is sorted by user growth and moves fast: its #1 changed 13 times
in the archive's 45 reads, and the top 10 turns over about half its games in
3 days. Persistence (betting the chart will not move) therefore loses most of
its places by the later hours, and the log score punishes a confident wrong
call at -9.21 a place. So this forecaster does not bet on one order.

For every game it fits a rank random walk from the history — a recent rank
velocity (a weighted least-squares slope over the last ~18 hours, damped to
a fifth because velocity is noisy and mean-reverts, decaying with a 24-hour
half-life) plus a per-game noise (from the game's own rank changes, shrunk
toward the median) — and simulates 2,000 paths of the next 72 hours,
re-sorting the games every hour. The probability a game holds a place is its
share of (path, hour) pairs at that place: a time average over the window,
tight early and spread late. A quarter of every place is left unnamed, shared
by games the history never saw (a brand-new game took #1 in the practice
window).

No network, no randomness, no clock: the same history always gives the same
bytes (seeded PRNG, mulberry32).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only.
