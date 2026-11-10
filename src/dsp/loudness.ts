import { Biquad, Cascade } from "./biquad.ts";

/**
 * K-weighting filter from ITU-R BS.1770 for any sample rate.
 * Coefficients follow the analogue-prototype derivation used by libebur128,
 * which reproduces the reference 48 kHz coefficients exactly.
 */
export function kWeighting(fs: number): Cascade {
  // Stage 1: high-shelf (head effects).
  let f0 = 1681.974450955533;
  const G = 3.999843853973347;
  let Q = 0.7071752369554196;
  let K = Math.tan((Math.PI * f0) / fs);
  const Vh = Math.pow(10, G / 20);
  const Vb = Math.pow(Vh, 0.4996667741545416);
  let a0 = 1 + K / Q + K * K;
  const shelf = new Biquad(
    (Vh + (Vb * K) / Q + K * K) / a0,
    (2 * (K * K - Vh)) / a0,
    (Vh - (Vb * K) / Q + K * K) / a0,
    (2 * (K * K - 1)) / a0,
    (1 - K / Q + K * K) / a0,
  );
  // Stage 2: RLB high-pass.
  f0 = 38.13547087602444;
  Q = 0.5003270373238773;
  K = Math.tan((Math.PI * f0) / fs);
  a0 = 1 + K / Q + K * K;
  const hp = new Biquad(1, -2, 1, (2 * (K * K - 1)) / a0, (1 - K / Q + K * K) / a0);
  return new Cascade([shelf, hp]);
}

export function powerToLufs(p: number): number {
  return p > 0 ? -0.691 + 10 * Math.log10(p) : -Infinity;
}

export function lufsToPower(l: number): number {
  return Math.pow(10, (l + 0.691) / 10);
}

/** Growable Float64 array. */
export class F64List {
  data = new Float64Array(1024);
  length = 0;
  push(v: number): void {
    if (this.length === this.data.length) {
      const d = new Float64Array(this.data.length * 2);
      d.set(this.data);
      this.data = d;
    }
    this.data[this.length++] = v;
  }
  view(): Float64Array {
    return this.data.subarray(0, this.length);
  }
}

/**
 * Integrated loudness per ITU-R BS.1770-5 (400 ms blocks, 75 % overlap,
 * −70 LKFS absolute gate, −10 LU relative gate).
 *
 * `dualMono`: a single-channel signal is measured as if it were played on
 * both left and right speakers (each with weight 1.0), which is how podcast
 * players reproduce mono files.
 */
export class LoudnessMeter {
  private filters: Cascade[];
  private weights: number[];
  private segLen: number;
  private segPos = 0;
  private segAcc = 0;
  private tmp = new Float64Array(0);
  /** Channel-weighted sum of squares for each 100 ms segment. */
  readonly segments = new F64List();

  constructor(
    readonly sampleRate: number,
    readonly channels: number,
    dualMono = true,
  ) {
    this.filters = Array.from({ length: channels }, () => kWeighting(sampleRate));
    this.weights = Array.from({ length: channels }, () => (channels === 1 && dualMono ? 2 : 1));
    this.segLen = Math.round(sampleRate / 10);
  }

  /** Adds planar audio. All channel arrays must have the same length. */
  push(planar: Float32Array[]): void {
    const n = planar[0].length;
    if (this.tmp.length < n) this.tmp = new Float64Array(n);
    const sq = new Float64Array(n);
    for (let c = 0; c < this.channels; c++) {
      const t = this.tmp.subarray(0, n);
      t.set(planar[c]);
      this.filters[c].process64(t);
      const w = this.weights[c];
      for (let i = 0; i < n; i++) sq[i] += w * t[i] * t[i];
    }
    let i = 0;
    while (i < n) {
      const take = Math.min(n - i, this.segLen - this.segPos);
      let acc = this.segAcc;
      for (let k = i; k < i + take; k++) acc += sq[k];
      this.segAcc = acc;
      this.segPos += take;
      i += take;
      if (this.segPos === this.segLen) {
        this.segments.push(this.segAcc);
        this.segAcc = 0;
        this.segPos = 0;
      }
    }
  }

  /** Mean-square power of each gating block (400 ms, step 100 ms). */
  blockPowers(): Float64Array {
    const s = this.segments.view();
    const nb = Math.max(0, s.length - 3);
    const out = new Float64Array(nb);
    const denom = 4 * this.segLen;
    for (let j = 0; j < nb; j++) out[j] = (s[j] + s[j + 1] + s[j + 2] + s[j + 3]) / denom;
    return out;
  }

  integrated(): number {
    return gatedLoudness(this.blockPowers());
  }

  /** Highest short-term (3 s) loudness. */
  maxShortTerm(): number {
    const s = this.segments.view();
    let best = -Infinity;
    let acc = 0;
    for (let j = 0; j < s.length; j++) {
      acc += s[j];
      if (j >= 30) acc -= s[j - 30];
      if (j >= 29) best = Math.max(best, powerToLufs(acc / (30 * this.segLen)));
    }
    return best;
  }
}

export function gatedLoudness(blocks: Float64Array): number {
  const absGate = lufsToPower(-70);
  let sum = 0;
  let count = 0;
  for (const p of blocks) {
    if (p > absGate) {
      sum += p;
      count++;
    }
  }
  if (count === 0) return -Infinity;
  const relGate = lufsToPower(powerToLufs(sum / count) - 10);
  let sum2 = 0;
  let count2 = 0;
  for (const p of blocks) {
    if (p > absGate && p > relGate) {
      sum2 += p;
      count2++;
    }
  }
  return count2 ? powerToLufs(sum2 / count2) : -Infinity;
}
