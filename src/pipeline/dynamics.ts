/**
 * Gain-curve computation for the dynamics stages. Everything here works on
 * per-hop level measurements (one value per HOP samples) of the whole file,
 * so each stage can look both backwards and forwards in time.
 *
 * All levels and gains are in dB.
 */

export const HOP_SECONDS = 0.005;
const HOPS_PER_SECOND = 1 / HOP_SECONDS;
const FLOOR_DB = -120;

export function powerToDb(p: number): number {
  return p > 1e-12 ? 10 * Math.log10(p) : FLOOR_DB;
}

const s2h = (s: number) => Math.max(1, Math.round(s * HOPS_PER_SECOND));

// ---------------------------------------------------------------------------
// Generic helpers

/** Sliding minimum over [i - before, i + after]. */
export function minFilter(x: Float32Array, before: number, after: number): Float32Array {
  const n = x.length;
  const out = new Float32Array(n);
  const dq = new Int32Array(n + 1);
  let head = 0, tail = 0;
  let next = 0; // next index to add
  for (let i = 0; i < n; i++) {
    const hi = Math.min(n - 1, i + after);
    while (next <= hi) {
      while (tail > head && x[dq[tail - 1]] >= x[next]) tail--;
      dq[tail++] = next++;
    }
    while (dq[head] < i - before) head++;
    out[i] = x[dq[head]];
  }
  return out;
}

/** Centered moving average with half-width `half` (window 2·half+1, edges shrink). */
export function boxSmooth(x: Float32Array, half: number): Float32Array {
  const n = x.length;
  const pre = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + x[i];
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - half), b = Math.min(n, i + half + 1);
    out[i] = (pre[b] - pre[a]) / (b - a);
  }
  return out;
}

/**
 * Smooths a gain-reduction curve without letting it rise above the input
 * anywhere: running min over ±half, then a ±half box average.
 */
export function smoothReduction(x: Float32Array, half: number): Float32Array {
  return boxSmooth(minFilter(x, half, half), half);
}

/** Zero-phase one-pole smoothing (forward then backward). */
export function zeroPhaseSmooth(x: Float32Array, tauHops: number): Float32Array {
  const a = 1 - Math.exp(-1 / tauHops);
  const out = new Float32Array(x);
  for (let i = 1; i < out.length; i++) out[i] = out[i - 1] + a * (out[i] - out[i - 1]);
  for (let i = out.length - 2; i >= 0; i--) out[i] = out[i + 1] + a * (out[i] - out[i + 1]);
  return out;
}

const BIN_MIN = -120;
const BIN_STEP = 0.5;
const BINS = 300; // −120 … +30 dB

function bin(v: number): number {
  return Math.max(0, Math.min(BINS - 1, Math.round((v - BIN_MIN) / BIN_STEP)));
}

/**
 * Percentile `q` (0..1) of `x` over hops where `mask` is set, in a window of
 * ±half hops. Returns NaN where fewer than `minCount` hops qualify.
 */
export function slidingPercentile(
  x: Float32Array,
  mask: Uint8Array,
  half: number,
  q: number,
  minCount: number,
  stride = 4,
): Float32Array {
  const n = x.length;
  const hist = new Int32Array(BINS);
  let count = 0;
  const out = new Float32Array(n).fill(NaN);
  let lo = 0, hi = 0; // window [lo, hi)
  const query = (): number => {
    if (count < minCount) return NaN;
    const target = q * (count - 1);
    let acc = 0;
    for (let b = 0; b < BINS; b++) {
      acc += hist[b];
      if (acc > target) return BIN_MIN + b * BIN_STEP;
    }
    return BIN_MIN + (BINS - 1) * BIN_STEP;
  };
  const points: number[] = [];
  for (let i = 0; i < n; i += stride) points.push(i);
  if (points[points.length - 1] !== n - 1) points.push(n - 1);
  for (const i of points) {
    const a = Math.max(0, i - half), b = Math.min(n, i + half + 1);
    while (hi < b) {
      if (mask[hi]) { hist[bin(x[hi])]++; count++; }
      hi++;
    }
    while (lo < a) {
      if (mask[lo]) { hist[bin(x[lo])]--; count--; }
      lo++;
    }
    out[i] = query();
  }
  // Linear interpolation between query points.
  for (let k = 0; k + 1 < points.length; k++) {
    const i0 = points[k], i1 = points[k + 1];
    const v0 = out[i0], v1 = out[i1];
    for (let i = i0 + 1; i < i1; i++) {
      if (Number.isNaN(v0) || Number.isNaN(v1)) out[i] = Number.isNaN(v0) ? v1 : v0;
      else out[i] = v0 + ((v1 - v0) * (i - i0)) / (i1 - i0);
    }
  }
  return out;
}

export function percentile(x: Float32Array, mask: Uint8Array | null, q: number): number {
  const vals: number[] = [];
  for (let i = 0; i < x.length; i++) if (!mask || mask[i]) vals.push(x[i]);
  if (!vals.length) return NaN;
  vals.sort((a, b) => a - b);
  return vals[Math.min(vals.length - 1, Math.floor(q * (vals.length - 1) + 0.5))];
}

/** Fills NaN entries by linear interpolation from valid neighbours. */
export function fillGaps(x: Float32Array, fallback: number): Float32Array {
  const out = new Float32Array(x);
  let last = -1;
  for (let i = 0; i <= out.length; i++) {
    if (i < out.length && Number.isNaN(out[i])) continue;
    const v1 = i < out.length ? out[i] : last >= 0 ? out[last] : fallback;
    const v0 = last >= 0 ? out[last] : v1;
    for (let k = last + 1; k < i; k++) out[k] = last < 0 ? v1 : v0 + ((v1 - v0) * (k - last)) / (i - last);
    last = i;
  }
  if (last < 0) out.fill(fallback);
  return out;
}

// ---------------------------------------------------------------------------
// Speech detection

/**
 * Speech mask from the noise-reduction model's voice probability, with
 * short gaps bridged so consonants inside words count as speech.
 */
export function speechMask(vad: Float32Array, levelDb: Float32Array): Uint8Array {
  const n = vad.length;
  const m = new Uint8Array(n);
  for (let i = 0; i < n; i++) m[i] = vad[i] >= 0.5 && levelDb[i] > -70 ? 1 : 0;
  // Close gaps up to 60 ms.
  const gap = s2h(0.06);
  let lastOn = -Infinity;
  for (let i = 0; i < n; i++) {
    if (m[i]) {
      if (i - lastOn > 1 && i - lastOn <= gap + 1) for (let k = lastOn + 1; k < i; k++) m[k] = 1;
      lastOn = i;
    }
  }
  return m;
}

// ---------------------------------------------------------------------------
// Stage 1: sudden loud noises

export interface SpikeResult {
  gain: Float32Array;
  events: number;
}

/**
 * Finds short bursts (< 1 s) far above the loud end of the surrounding
 * speech and pulls them down to that level.
 *
 * `levelDb`: unweighted 5 ms power. `speech`: speech mask.
 */
export function spikeGains(levelDb: Float32Array, speech: Uint8Array): SpikeResult {
  const n = levelDb.length;
  const gain = new Float32Array(n);
  let globalRef = percentile(levelDb, speech, 0.95);
  if (Number.isNaN(globalRef)) globalRef = percentile(levelDb, null, 0.9);
  if (Number.isNaN(globalRef)) return { gain, events: 0 };
  const ref = fillGaps(slidingPercentile(levelDb, speech, s2h(3), 0.95, s2h(0.5)), globalRef);
  const over = 6; // dB above the loud end of nearby speech that counts as a spike
  const flagged = new Uint8Array(n);
  for (let i = 0; i < n; i++) flagged[i] = levelDb[i] > ref[i] + over ? 1 : 0;
  // Group into events, bridging gaps ≤ 20 ms.
  const bridge = s2h(0.02);
  const maxLen = s2h(1.0);
  let events = 0;
  let i = 0;
  const target = new Float32Array(n);
  while (i < n) {
    if (!flagged[i]) { i++; continue; }
    let end = i;
    let k = i + 1;
    while (k < n && k - end <= bridge + 1) {
      if (flagged[k]) end = k;
      k++;
    }
    if (end - i + 1 <= maxLen) {
      events++;
      for (let j = i; j <= end; j++) {
        // Reduce to 3 dB above the loud end of speech.
        const g = Math.min(0, ref[j] + 3 - levelDb[j]);
        target[j] = Math.max(-30, g);
      }
    }
    i = end + 1;
  }
  if (!events) return { gain, events };
  // Reach full reduction slightly before the burst and hold briefly after.
  const ahead = minFilter(target, s2h(0.03), s2h(0.01));
  return { gain: boxSmooth(ahead, s2h(0.01)), events };
}

// ---------------------------------------------------------------------------
// Stage 2: leveler (even out speakers)

export interface LevelerResult {
  gain: Float32Array;
  /** Overall speech level the leveler aims for (K-weighted dB). */
  target: number;
}

/**
 * Rides the level of each stretch of speech (≈3 s windows) towards the
 * median speech level of the whole recording, like a careful engineer on a
 * fader. Pauses take the gain of the speech around them, with boost capped
 * so room tone isn't lifted.
 */
export function levelerGains(levelDb: Float32Array, speech: Uint8Array): LevelerResult {
  const n = levelDb.length;
  const half = s2h(1.5);
  const minCount = s2h(0.3);
  const phrase = new Float32Array(n).fill(NaN);
  let sum = 0, cnt = 0;
  let lo = 0, hi = 0;
  const pw = (i: number) => Math.pow(10, levelDb[i] / 10);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - half), b = Math.min(n, i + half + 1);
    while (hi < b) { if (speech[hi]) { sum += pw(hi); cnt++; } hi++; }
    while (lo < a) { if (speech[lo]) { sum -= pw(lo); cnt--; } lo++; }
    if (cnt >= minCount && sum > 0) phrase[i] = 10 * Math.log10(sum / cnt);
  }
  const speechPhrase = new Uint8Array(n);
  for (let i = 0; i < n; i++) speechPhrase[i] = speech[i] && !Number.isNaN(phrase[i]) ? 1 : 0;
  const target = percentile(phrase, speechPhrase, 0.5);
  if (Number.isNaN(target)) return { gain: new Float32Array(n), target: percentile(levelDb, null, 0.9) };
  const raw = new Float32Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    if (!Number.isNaN(phrase[i])) raw[i] = Math.max(-12, Math.min(12, target - phrase[i]));
  }
  const filled = fillGaps(raw, 0);
  // Cap boost where nobody is talking.
  const quietHalf = s2h(0.4);
  const nearSpeech = minFilter(Float32Array.from(speech, (v) => 1 - v), quietHalf, quietHalf); // 1 = no speech within ±0.4 s
  for (let i = 0; i < n; i++) if (nearSpeech[i] === 1 && filled[i] > 3) filled[i] = 3;
  return { gain: zeroPhaseSmooth(filled, s2h(0.25)), target };
}

// ---------------------------------------------------------------------------
// Stage 3: gentle compression (inside phrases)

export function compressorGains(levelDb: Float32Array, threshold: number, ratio = 2.5, knee = 6): Float32Array {
  const n = levelDb.length;
  // 30 ms RMS detector.
  const pw = Float32Array.from(levelDb, (d) => Math.pow(10, d / 10));
  const det = boxSmooth(pw, 3);
  const target = new Float32Array(n);
  const slope = 1 - 1 / ratio;
  for (let i = 0; i < n; i++) {
    const x = powerToDb(det[i]);
    const over = x - threshold;
    let g = 0;
    if (over >= knee / 2) g = -slope * over;
    else if (over > -knee / 2) g = (-slope * (over + knee / 2) ** 2) / (2 * knee);
    target[i] = g;
  }
  // 10 ms look-ahead attack, 120 ms release.
  const ahead = minFilter(target, 0, s2h(0.01));
  const rel = 1 - Math.exp(-1 / s2h(0.12));
  const out = new Float32Array(n);
  let y = 0;
  for (let i = 0; i < n; i++) {
    const g = ahead[i];
    y = g < y ? g : y + rel * (g - y);
    out[i] = y;
  }
  return boxSmooth(out, s2h(0.01));
}

// ---------------------------------------------------------------------------
// Stage 4: de-esser

/**
 * Reduces the band above ~5 kHz when it is both dominant (a sibilant, not a
 * vowel) and loud relative to the speech level.
 *
 * `hfDb`: 5 ms power above the crossover, after the broadband gains.
 * `fullDb`: 5 ms broadband power, after the broadband gains.
 */
export function deEssGains(hfDb: Float32Array, fullDb: Float32Array, speechLevel: number): Float32Array {
  const n = hfDb.length;
  const thr = speechLevel - 6;
  const target = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const dominance = hfDb[i] - fullDb[i]; // ≤ 0
    if (dominance < -8) continue;
    const over = hfDb[i] - thr;
    if (over <= 0) continue;
    target[i] = Math.max(-8, -over * (1 - 1 / 3));
  }
  return smoothReduction(target, s2h(0.01));
}
