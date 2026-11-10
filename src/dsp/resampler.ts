/**
 * Streaming band-limited resampler (Kaiser-windowed sinc) for arbitrary
 * rate ratios. Stopband ≈ −100 dB, passband flat to 92 % of the lower
 * Nyquist frequency (≈ 20.3 kHz when the lower rate is 44.1 kHz).
 */

const PHASES = 1024; // table resolution per unit of filter time
const HALF_WIDTH = 80; // zero crossings each side, in filter-time units
const RHO = 0.92; // cutoff relative to the lower Nyquist
const BETA = 10.06; // Kaiser β for ≈100 dB attenuation

function besselI0(x: number): number {
  let sum = 1;
  let term = 1;
  const q = (x * x) / 4;
  for (let k = 1; k < 64; k++) {
    term *= q / (k * k);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  return sum;
}

let kernelTable: Float64Array | null = null;
function kernel(): Float64Array {
  if (kernelTable) return kernelTable;
  const n = HALF_WIDTH * PHASES + 2;
  const tab = new Float64Array(n);
  const i0b = besselI0(BETA);
  for (let j = 0; j < n; j++) {
    const t = j / PHASES;
    if (t >= HALF_WIDTH) {
      tab[j] = 0;
      continue;
    }
    const x = Math.PI * RHO * t;
    const sinc = t === 0 ? 1 : Math.sin(x) / x;
    const r = t / HALF_WIDTH;
    const w = besselI0(BETA * Math.sqrt(1 - r * r)) / i0b;
    tab[j] = RHO * sinc * w;
  }
  kernelTable = tab;
  return tab;
}

export class Resampler {
  readonly ratio: number; // input samples per output sample
  private scale: number;
  private reach: number; // half-width in input samples
  private buf: Float32Array[];
  private bufLen: number;
  private dropped = 0; // input samples discarded from the front of buf
  private outCount = 0;
  private inCount = 0;
  private coeffs: Float64Array;
  private tab = kernel();

  constructor(
    readonly inRate: number,
    readonly outRate: number,
    readonly channels: number,
  ) {
    this.ratio = inRate / outRate;
    this.scale = Math.min(1, outRate / inRate);
    this.reach = Math.ceil(HALF_WIDTH / this.scale) + 1;
    // Start with `reach` zeros so the first outputs see silence before time 0.
    this.bufLen = this.reach;
    this.buf = Array.from({ length: channels }, () => new Float32Array(Math.max(4096, this.reach * 4)));
    this.coeffs = new Float64Array(2 * this.reach + 2);
  }

  get passthrough(): boolean {
    return this.inRate === this.outRate;
  }

  private ensure(extra: number): void {
    const need = this.bufLen + extra;
    if (need <= this.buf[0].length) return;
    let cap = this.buf[0].length;
    while (cap < need) cap *= 2;
    this.buf = this.buf.map((b) => {
      const nb = new Float32Array(cap);
      nb.set(b.subarray(0, this.bufLen));
      return nb;
    });
  }

  private append(planar: Float32Array[], n: number): void {
    this.ensure(n);
    for (let c = 0; c < this.channels; c++) {
      if (planar[c]) this.buf[c].set(planar[c].subarray(0, n), this.bufLen);
      else this.buf[c].fill(0, this.bufLen, this.bufLen + n);
    }
    this.bufLen += n;
  }

  /** Produces as many output samples as possible. `limit` caps total output. */
  private produce(limit: number): Float32Array[] {
    const startPos = (n: number) => n * this.ratio - this.dropped + this.reach;
    // Count outputs we can make: need buf up to floor(pos + reach).
    let n = this.outCount;
    const outs: number[] = [];
    while (n < limit) {
      const pos = startPos(n);
      if (Math.floor(pos) + this.reach >= this.bufLen) break;
      outs.push(pos);
      n++;
    }
    const count = outs.length;
    const out = Array.from({ length: this.channels }, () => new Float32Array(count));
    const tab = this.tab;
    const scale = this.scale;
    const coeffs = this.coeffs;
    const reach = this.reach;
    const tabMax = HALF_WIDTH * PHASES;
    for (let k = 0; k < count; k++) {
      const pos = outs[k];
      const first = Math.floor(pos) - reach + 1;
      const taps = 2 * reach;
      for (let j = 0; j < taps; j++) {
        const u = Math.abs(scale * (first + j - pos)) * PHASES;
        if (u >= tabMax) {
          coeffs[j] = 0;
          continue;
        }
        const idx = u | 0;
        const fr = u - idx;
        coeffs[j] = tab[idx] + fr * (tab[idx + 1] - tab[idx]);
      }
      for (let c = 0; c < this.channels; c++) {
        const b = this.buf[c];
        let acc = 0;
        for (let j = 0; j < taps; j++) acc += b[first + j] * coeffs[j];
        out[c][k] = acc * scale;
      }
    }
    this.outCount = n;
    // Discard input that no future output needs.
    const keepFrom = Math.max(0, Math.floor(startPos(n)) - reach - 1);
    if (keepFrom > 0) {
      for (let c = 0; c < this.channels; c++) this.buf[c].copyWithin(0, keepFrom, this.bufLen);
      this.bufLen -= keepFrom;
      this.dropped += keepFrom;
    }
    return out;
  }

  push(planar: Float32Array[]): Float32Array[] {
    if (this.passthrough) return planar.map((p) => p.slice());
    const n = planar[0].length;
    this.inCount += n;
    this.append(planar, n);
    return this.produce(Number.MAX_SAFE_INTEGER);
  }

  /** Emits the remaining output; total output length is round(in × outRate / inRate). */
  flush(): Float32Array[] {
    if (this.passthrough) return Array.from({ length: this.channels }, () => new Float32Array(0));
    const total = Math.round((this.inCount * this.outRate) / this.inRate);
    const zeros = Array.from({ length: this.channels }, () => new Float32Array(this.reach * 2 + 2));
    this.append(zeros, zeros[0].length);
    return this.produce(total);
  }
}
