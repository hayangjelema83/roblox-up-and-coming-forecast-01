// roblox-up-and-coming-forecast-01 entry (Node 22, standard library only).
//
// A momentum Monte-Carlo forecaster for Roblox's Up-and-Coming chart. It reads
// /input/history.ndjson (one JSON object per read of the chart, oldest first)
// and /input/task.json, and writes /output/forecast.json: for each of the top 10
// places, a probability for each game it names being at that place, plus its
// guess at the top 10 in order. No network, no randomness, no clock: the same
// history always gives the same bytes.
//
// The idea, in brief. The chart is sorted by user growth and moves fast: its
// #1 changed 13 times in the archive's 45 reads, and the top 10 turns over
// about half its games in 3 days. Persistence (betting the chart will not
// move) therefore loses most of its places by the later hours, and the log
// score punishes a confident wrong call at -9.21 a place. So this forecaster
// does not bet on one order. For every game it fits a rank random walk from
// the history — a recent rank velocity (a weighted least-squares slope over
// the last ~18 hours, damped to a fifth because velocity is noisy and
// mean-reverts, decaying with a 24-hour half-life) plus a per-game noise
// (from the game's own rank changes, shrunk toward the median, scaled to
// 0.15 because the raw changes overstate the hourly noise) — and simulates
// 2,000 paths of the next 72 hours, re-sorting the games every hour. The
// probability a game holds a place is its share of (path, hour) pairs at
// that place: a time average over the window, tight early (the chart has
// barely moved) and spread late (it has had time to turn over). Games the
// history never saw still reach the top 10 — a brand-new game took #1 in
// the practice window — so a quarter of every place is left unnamed, shared
// by every game on a read's chart the forecast does not name there.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Paths simulated of the window. */
const PATHS = 2000;
/** Fixed seed: the same history gives the same forecast, run after run. */
const SEED = 12345;
/** Rank velocity is noisy and mean-reverts: keep this much of the fitted slope. */
const DRIFT_DAMP = 0.2;
/** The fitted slope's half-life in hours. */
const DRIFT_HALFLIFE_H = 24;
/** Hours of history the slope fit weights (exponential, half-life tau). */
const TAU_H = 18;
/** The raw rank changes overstate the hourly noise: scale the fitted sigma. */
const SIGMA_SCALE = 0.15;
/** A place's mass left unnamed, shared by the games not named there. */
const UNNAMED_FLOOR = 0.25;
/** Only name a game at a place when its probability is at least this. */
const CUTOFF = 0.003;
/** At most this many games named at one place. */
const MAX_PER_PLACE = 40;
/** A game out of the last read but top-40 in the last 3 reads may return. */
const RECENT_TOP = 40;
const RECENT_SNAPS = 3;
/** A returning game starts this many places below its last rank. */
const RETURN_PENALTY = 12;
/** Ranks are clipped to this range in the simulation. */
const RANK_MAX = 140;

interface SnapRow {
  rank: number;
  universe_id: number;
}

interface Snap {
  at: string;
  up_and_coming: SnapRow[];
}

/** mulberry32: a small seeded PRNG, deterministic across runs. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** One standard normal from the seeded PRNG (Box-Muller, cos branch). */
function gauss(rnd: () => number): number {
  const u1 = Math.max(rnd(), 1e-12);
  const u2 = rnd();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  window?: { slots?: number };
};
const PLACES = task.places;
const HOURS = task.window?.slots ?? 72;
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const snaps = lines.map((l) => JSON.parse(l) as Snap);
const tLast = Date.parse(snaps[snaps.length - 1]!.at) / 1000;
const lastRank = new Map<number, number>();
for (const r of snaps[snaps.length - 1]!.up_and_coming) lastRank.set(r.universe_id, r.rank);

// Per-game rank series: [time seconds, rank].
const series = new Map<number, Array<[number, number]>>();
for (const s of snaps) {
  const t = Date.parse(s.at) / 1000;
  for (const r of s.up_and_coming) {
    const pts = series.get(r.universe_id);
    if (pts) pts.push([t, r.rank]);
    else series.set(r.universe_id, [[t, r.rank]]);
  }
}

// Candidate games: the last chart, plus recent top-40 games that may return.
const cand = new Set<number>(lastRank.keys());
for (const s of snaps.slice(-RECENT_SNAPS)) {
  for (const r of s.up_and_coming.slice(0, RECENT_TOP)) cand.add(r.universe_id);
}

interface Game {
  uid: number;
  r0: number;
  drift: number;
  sig: number;
  /** How far to shrink this game's sigma toward the median (0 to 1). */
  shrink: number;
}

const games: Game[] = [];
const sigs: number[] = [];
for (const uid of cand) {
  const pts = series.get(uid);
  if (!pts || pts.length === 0) continue;
  const r0 = lastRank.get(uid) ?? pts[pts.length - 1]![1] + RETURN_PENALTY;
  // Weighted least-squares slope of rank against time (hours since the last
  // read), weights exp(h / TAU_H) with h <= 0.
  let sw = 0;
  let mx = 0;
  let my = 0;
  const w: number[] = [];
  const x: number[] = [];
  const y: number[] = [];
  for (const [t, r] of pts) {
    const h = (t - tLast) / 3600;
    const wi = h > -200 ? Math.exp(h / TAU_H) : 0;
    w.push(wi);
    x.push(h);
    y.push(r);
    sw += wi;
    mx += wi * h;
    my += wi * r;
  }
  if (sw <= 0) continue;
  mx /= sw;
  my /= sw;
  let den = 0;
  let num = 0;
  for (let i = 0; i < pts.length; i++) {
    den += w[i]! * (x[i]! - mx) * (x[i]! - mx);
    num += w[i]! * (x[i]! - mx) * (y[i]! - my);
  }
  const slope = den > 1e-9 ? num / den : 0;
  // Noise from consecutive rank changes, per sqrt(hour).
  let ss = 0;
  let nn = 0;
  for (let i = 1; i < pts.length; i++) {
    const dt = (pts[i]![0] - pts[i - 1]![0]) / 3600;
    if (dt > 0) {
      const d = (pts[i]![1] - pts[i - 1]![1]) / Math.sqrt(dt);
      ss += d * d;
      nn++;
    }
  }
  const sig = nn > 0 ? Math.sqrt(ss / nn) : 1;
  sigs.push(sig);
  // Games seen only once or twice have a sigma from a single noisy diff:
  // shrink it three-quarters of the way to the median instead of half.
  const shrink = pts.length < 4 ? 0.75 : 0.5;
  games.push({ uid, r0, drift: slope, sig, shrink });
}
sigs.sort((a, b) => a - b);
const medSig = sigs.length > 0 ? sigs[Math.floor(sigs.length / 2)]! : 1;
for (const g of games) {
  g.sig =
    Math.max(0.4, Math.min(2.5, g.sig * (1 - g.shrink) + medSig * g.shrink)) *
    SIGMA_SCALE;
  g.drift = Math.max(-1.5, Math.min(1.5, g.drift)) * DRIFT_DAMP;
}

// Simulate PATHS paths of HOURS hours; count (game, place) pairs.
const G = games.length;
const r0 = games.map((g) => g.r0);
const drift = games.map((g) => g.drift);
const sig = games.map((g) => g.sig);
const counts: number[][] = Array.from({ length: G }, () => new Array<number>(PLACES).fill(0));
const rnd = mulberry32(SEED);
const idx = Array.from({ length: G }, (_, i) => i);
for (let p = 0; p < PATHS; p++) {
  const ranks = r0.slice();
  for (let h = 1; h <= HOURS; h++) {
    const decay = Math.pow(0.5, h / DRIFT_HALFLIFE_H);
    for (let i = 0; i < G; i++) {
      ranks[i] = ranks[i]! + drift[i]! * decay + sig[i]! * gauss(rnd);
      if (ranks[i]! < 1) ranks[i] = 1;
      if (ranks[i]! > RANK_MAX) ranks[i] = RANK_MAX;
    }
    const order = idx.slice().sort((a, b) => {
      const ka = ranks[a]! + 1e-6 * r0[a]!;
      const kb = ranks[b]! + 1e-6 * r0[b]!;
      return ka - kb;
    });
    for (let k = 0; k < PLACES; k++) counts[order[k]!]![k]++;
  }
}

const total = PATHS * HOURS;
const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
const top10: number[] = [];
for (let k = 0; k < PLACES; k++) {
  const col: Array<[number, number]> = []; // [p, game index]
  for (let i = 0; i < G; i++) {
    const p = counts[i]![k]! / total;
    if (p >= CUTOFF) col.push([p, i]);
  }
  col.sort((a, b) => b[0] - a[0]);
  const named = col.slice(0, MAX_PER_PLACE);
  const s = named.reduce((a, [p]) => a + p, 0);
  const un = Math.max(UNNAMED_FLOOR, 1 - s);
  const scale = s > 0 ? (1 - un) / s : 0;
  for (const [p, i] of named) {
    probabilities.push({
      universe_id: games[i]!.uid,
      place: k + 1,
      p: Math.round(p * scale * 1e6) / 1e6,
    });
  }
  // The top 10 is the most likely game at each place, in order, deduplicated.
  const byP = Array.from({ length: G }, (_, i) => i).sort(
    (a, b) => counts[b]![k]! - counts[a]![k]!,
  );
  for (const i of byP) {
    const uid = games[i]!.uid;
    if (!top10.includes(uid)) {
      top10.push(uid);
      break;
    }
  }
}
for (const r of snaps[snaps.length - 1]!.up_and_coming) {
  if (top10.length >= PLACES) break;
  if (!top10.includes(r.universe_id)) top10.push(r.universe_id);
}

writeFileSync(
  join(OUTPUT_DIR, task.output),
  `${JSON.stringify({ probabilities, top10: top10.slice(0, PLACES) })}\n`,
);
