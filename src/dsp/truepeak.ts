/**
 * True-peak measurement (ITU-R BS.1770 Annex 2: ≥4× oversampling) and a
 * look-ahead true-peak limiter.
 */

const OVERSAMPLE = 4;
const TAPS = 32; // per phase; even
const HALF = TAPS / 2;

function besselI0(x: number): number {
  let sum = 1, term = 1;
  const q = (x * x) / 4;
  for (let k = 1; k < 64; k++) {
    term *= q / (k * k);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  return sum;
}

/**
 * coeffs[phase][j] weights x[n - HALF + 1 + j] to give the value at n + phase/4.
 * Phase 0 is the sample itself and is not stored.
 */
function phaseCoeffs(): Float64Array[] {
  const rho = 0.94;
  const beta = 8;
  const i0b = besselI0(beta);
  const out: Float64Array[] = [];
  for (let ph = 1; ph < OVERSAMPLE; ph++) {
    const c = new Float64Array(TAPS);
    let sum = 0;
    for (let j = 0; j < TAPS; j++) {
      const t = j - HALF + 1 - ph / OVERSAMPLE;
      const x = Math.PI * rho * t;
      const sinc = t === 0 ? 1 : Math.sin(x) / x;
      const r = t / (HALF + 1);
      const w = Math.abs(r) < 1 ? besselI0(beta * Math.sqrt(1 - r * r)) / i0b : 0;
      c[j] = rho * sinc * w;
      sum += c[j];
    }
    for (let j = 0; j < TAPS; j++) c[j] /= sum;
    out.push(c);
  }
  return out;
}

const COEFFS = phaseCoeffs();
/** Upper bound on |interpolated| / max|x| over the filter window. */
const GAIN_BOUND = Math.max(1, ...COEFFS.map((c) => c.reduce((s, v) => s + Math.abs(v), 0)));

/** Streaming true-peak meter. */
export class TruePeakMeter {
  private hist: Float32Array[];
  private filled = 0;
  max = 0;

  constructor(readonly channels: number) {
    // Linear history; we shift by chunk which keeps the inner loop simple.
    this.hist = Array.from({ length: channels }, () => new Float32Array(TAPS - 1));
  }

  push(planar: Float32Array[]): void {
    const n = planar[0].length;
    for (let c = 0; c < this.channels; c++) {
      const x = new Float32Array(TAPS - 1 + n);
      x.set(this.hist[c]);
      x.set(planar[c], TAPS - 1);
      let m = this.max;
      // Sample n_out = k + HALF - 1 (index into x) has its full window when k + TAPS - 1 < x.length.
      for (let k = 0; k + TAPS <= x.length; k++) {
        const center = x[k + HALF - 1];
        const a = center < 0 ? -center : center;
        if (a > m) m = a;
        for (let p = 0; p < COEFFS.length; p++) {
          const cf = COEFFS[p];
          let acc = 0;
          for (let j = 0; j < TAPS; j++) acc += x[k + j] * cf[j];
          if (acc > m) m = acc;
          else if (-acc > m) m = -acc;
        }
      }
      this.max = m;
      this.hist[c] = x.slice(x.length - (TAPS - 1));
    }
    this.filled += n;
  }

  /** Processes the trailing samples. */
  flush(): void {
    this.push(Array.from({ length: this.channels }, () => new Float32Array(TAPS)));
  }

  get dbtp(): number {
    return this.max > 0 ? 20 * Math.log10(this.max) : -Infinity;
  }
}

/**
 * Look-ahead true-peak limiter with linked channels.
 *
 * gain[n] is the box-average over `attack` samples of a release-smoothed
 * running minimum of the required gain over the next `attack` samples, which
 * guarantees gain[n] ≤ required[n] at every sample while ramping smoothly.
 */
export class TruePeakLimiter {
  private ceiling: number;
  private attack: number;
  private releaseCoef: number;
  private readonly delay: number;
  // Rings
  private size: number;
  private xr: Float32Array[]; // input audio ring
  private reqRing: Float64Array; // required gain ring (for windowed min)
  private boxRing: Float64Array; // smoothed min values (for box average)
  private boxSum = 0;
  private boxCount = 0;
  private dq: Int32Array; // monotonic deque of indices into reqRing
  private dqHead = 0;
  private dqTail = 0;
  private released = 1;
  // Sliding max of |x| (over channels) across the true-peak window, used to
  // skip the oversampling filter where no peak can reach the ceiling.
  private absRing: Float32Array;
  private amDq: Int32Array;
  private amHead = 0;
  private amTail = 0;
  private inIdx = 0; // samples pushed
  private outIdx = 0; // samples emitted
  maxReduction = 0; // dB, positive

  constructor(
    readonly sampleRate: number,
    readonly channels: number,
    ceilingDb: number,
    attackMs = 3,
    releaseMs = 80,
  ) {
    this.ceiling = Math.pow(10, ceilingDb / 20);
    this.attack = Math.max(2, Math.round((attackMs / 1000) * sampleRate));
    this.releaseCoef = 1 - Math.exp(-1 / ((releaseMs / 1000) * sampleRate));
    this.delay = HALF + this.attack - 1;
    this.size = 1;
    while (this.size < TAPS + 2 * this.attack + 8) this.size *= 2;
    this.xr = Array.from({ length: channels }, () => new Float32Array(this.size));
    this.reqRing = new Float64Array(this.size);
    this.boxRing = new Float64Array(this.size);
    this.dq = new Int32Array(this.size);
    this.absRing = new Float32Array(this.size);
    this.amDq = new Int32Array(this.size);
  }

  /** Records sample inIdx (already written to xr) in the sliding-max deque. */
  private trackAbs(): void {
    const mask = this.size - 1;
    const i = this.inIdx;
    let a = 0;
    for (let c = 0; c < this.channels; c++) {
      const v = this.xr[c][i & mask];
      const av = v < 0 ? -v : v;
      if (av > a) a = av;
    }
    this.absRing[i & mask] = a;
    while (this.amTail > this.amHead && this.absRing[this.amDq[(this.amTail - 1) & mask] & mask] <= a) this.amTail--;
    this.amDq[this.amTail++ & mask] = i;
  }

  /** Peak (incl. inter-sample) magnitude around sample index i, which needs x up to i + HALF. */
  private peakAt(i: number): number {
    const mask = this.size - 1;
    let m = 0;
    for (let c = 0; c < this.channels; c++) {
      const r = this.xr[c];
      const v = r[i & mask];
      const a = v < 0 ? -v : v;
      if (a > m) m = a;
      const base = i - HALF + 1;
      for (let p = 0; p < COEFFS.length; p++) {
        const cf = COEFFS[p];
        let acc = 0;
        for (let j = 0; j < TAPS; j++) acc += r[(base + j) & mask] * cf[j];
        if (acc > m) m = acc;
        else if (-acc > m) m = -acc;
      }
    }
    return m;
  }

  private step(out: Float32Array[], outPos: number): boolean {
    // Called after sample inIdx-1 was written. Sample j = inIdx-1-HALF now has its TP window.
    const mask = this.size - 1;
    const j = this.inIdx - 1 - HALF;
    if (j < 0) return false;
    while (this.amDq[this.amHead & mask] < j - HALF + 1) this.amHead++;
    const bound = this.absRing[this.amDq[this.amHead & mask] & mask] * GAIN_BOUND;
    let req = 1;
    if (bound > this.ceiling) {
      const p = this.peakAt(j);
      if (p > this.ceiling) req = this.ceiling / p;
    }
    this.reqRing[j & mask] = req;
    // Monotonic deque for min over [j - attack + 1, j].
    while (this.dqTail > this.dqHead && this.reqRing[this.dq[(this.dqTail - 1) & mask] & mask] >= req) this.dqTail--;
    this.dq[this.dqTail++ & mask] = j;
    // Window [k, j] is complete → running min for index k. For k < 0 the
    // window reaches before the start, where the required gain is 1.
    const k = j - this.attack + 1;
    while (this.dq[this.dqHead & mask] < k) this.dqHead++;
    const m = this.reqRing[this.dq[this.dqHead & mask] & mask];
    // Release smoothing (instant attack).
    this.released = m < this.released ? m : this.released + (m - this.released) * this.releaseCoef;
    if (this.released > m) this.released = m;
    // Box average over the last `attack` smoothed values gives gain for sample k.
    this.boxRing[k & mask] = this.released;
    this.boxSum += this.released;
    this.boxCount++;
    if (this.boxCount > this.attack) {
      this.boxSum -= this.boxRing[(k - this.attack) & mask];
      this.boxCount = this.attack;
    }
    if (k < 0) return false;
    // Box average for k uses smoothed values k-attack+1..k (all exist now).
    const g = this.boxSum / this.attack;
    const db = -20 * Math.log10(g);
    if (db > this.maxReduction) this.maxReduction = db;
    for (let c = 0; c < this.channels; c++) out[c][outPos] = this.xr[c][k & mask] * g;
    this.outIdx = k + 1;
    return true;
  }

  push(planar: Float32Array[]): Float32Array[] {
    const n = planar[0].length;
    const mask = this.size - 1;
    const out = Array.from({ length: this.channels }, () => new Float32Array(n));
    let w = 0;
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < this.channels; c++) this.xr[c][this.inIdx & mask] = planar[c][i];
      this.trackAbs();
      this.inIdx++;
      if (this.step(out, w)) w++;
    }
    return out.map((o) => o.subarray(0, w));
  }

  flush(): Float32Array[] {
    const total = this.inIdx;
    const pad = this.delay + 1;
    const out = Array.from({ length: this.channels }, () => new Float32Array(pad));
    const mask = this.size - 1;
    let w = 0;
    for (let i = 0; i < pad && this.outIdx < total; i++) {
      for (let c = 0; c < this.channels; c++) this.xr[c][this.inIdx & mask] = 0;
      this.trackAbs();
      this.inIdx++;
      if (this.step(out, w)) w++;
    }
    return out.map((o) => o.subarray(0, w));
  }
}
