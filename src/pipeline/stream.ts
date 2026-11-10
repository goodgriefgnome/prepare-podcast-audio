import { LR4Crossover, type Cascade } from "../dsp/biquad.ts";
import { kWeighting } from "../dsp/loudness.ts";
import type { Planar } from "../io/source.ts";
import type { Denoiser } from "./denoise.ts";

/** Sample FIFO. */
export class Fifo {
  private buf = new Float32Array(4096);
  private start = 0;
  private end = 0;
  get length() {
    return this.end - this.start;
  }
  push(x: Float32Array): void {
    if (this.end + x.length > this.buf.length) {
      const len = this.length;
      let cap = this.buf.length;
      while (cap < len + x.length) cap *= 2;
      const nb = cap === this.buf.length ? this.buf : new Float32Array(cap);
      nb.set(this.buf.subarray(this.start, this.end), 0);
      this.buf = nb;
      this.start = 0;
      this.end = len;
    }
    this.buf.set(x, this.end);
    this.end += x.length;
  }
  shift(n: number): Float32Array {
    const out = this.buf.slice(this.start, this.start + n);
    this.start += n;
    return out;
  }
}

/** Growable Float32 array. */
export class F32List {
  data = new Float32Array(4096);
  length = 0;
  push(v: number): void {
    if (this.length === this.data.length) {
      const d = new Float32Array(this.data.length * 2);
      d.set(this.data);
      this.data = d;
    }
    this.data[this.length++] = v;
  }
  view(): Float32Array {
    return this.data.subarray(0, this.length);
  }
}

/**
 * Streams audio through the noise-reduction model (one instance per channel),
 * compensating for its delay so output lines up with input. Also records
 * voice activity per model frame, from the model when noise reduction is on,
 * or from a single instance on the mix when it's off (audio passes through).
 */
export class DenoiseStream {
  readonly vad = new F32List();
  readonly frameSize: number;
  private wet: Fifo[];
  private dry: Fifo[];
  private pending: Fifo[];
  private skip: number;
  private inCount = 0;
  private outCount = 0;
  private frame: Float32Array;
  private mixFrame: Float32Array;
  private before: number[] = [];
  private after: number[] = [];
  /** Linear mix of the original kept under the model output. */
  private dryMix: number;

  constructor(
    private insts: Denoiser[],
    private channels: number,
    private enabled: boolean,
    maxReductionDb = 18,
  ) {
    this.frameSize = insts[0].frameSize;
    this.skip = enabled ? insts[0].delay : 0;
    this.wet = Array.from({ length: channels }, () => new Fifo());
    this.dry = Array.from({ length: channels }, () => new Fifo());
    this.pending = Array.from({ length: channels }, () => new Fifo());
    this.frame = new Float32Array(this.frameSize);
    this.mixFrame = new Float32Array(this.frameSize);
    this.dryMix = Math.pow(10, -maxReductionDb / 20);
  }

  private processFrame(frames: Float32Array[]): void {
    const F = this.frameSize;
    const C = this.channels;
    const mix = this.mixFrame;
    mix.fill(0);
    for (let c = 0; c < C; c++) for (let i = 0; i < F; i++) mix[i] += frames[c][i] / C;
    let eIn = 0;
    for (let i = 0; i < F; i++) eIn += mix[i] * mix[i];
    if (!this.enabled) {
      this.vad.push(this.insts[0].process(mix));
      for (let c = 0; c < C; c++) this.wet[c].push(frames[c]);
      return;
    }
    let v = 0;
    let eOut = 0;
    for (let c = 0; c < C; c++) {
      const f = this.frame;
      f.set(frames[c]);
      this.dry[c].push(frames[c]);
      v = Math.max(v, this.insts[c].process(f));
      this.wet[c].push(this.skip >= F ? f.subarray(F) : f.subarray(this.skip));
      for (let i = 0; i < F; i++) eOut += (f[i] * f[i]) / C;
    }
    this.skip = Math.max(0, this.skip - F);
    this.vad.push(v);
    if (v < 0.1 && eIn > 1e-10) {
      const m = this.dryMix;
      this.before.push(10 * Math.log10(eIn / F));
      this.after.push(10 * Math.log10(Math.max(1e-12, (eOut / F) * (1 - m) ** 2 + (eIn / F) * m * m)));
    }
  }

  private drain(limit: number): Planar {
    const C = this.channels;
    const dryLen = this.enabled ? this.dry[0].length : Infinity;
    const n = Math.max(0, Math.min(this.wet[0].length, dryLen, limit - this.outCount));
    const out: Planar = [];
    for (let c = 0; c < C; c++) {
      const w = this.wet[c].shift(n);
      if (this.enabled) {
        const d = this.dry[c].shift(n);
        const m = this.dryMix;
        for (let i = 0; i < n; i++) w[i] = w[i] * (1 - m) + d[i] * m;
      }
      out.push(w);
    }
    this.outCount += n;
    return out;
  }

  private run(): void {
    const F = this.frameSize;
    while (this.pending[0].length >= F) {
      this.processFrame(this.pending.map((p) => p.shift(F)));
    }
  }

  push(p: Planar): Planar {
    for (let c = 0; c < this.channels; c++) this.pending[c].push(p[c]);
    this.inCount += p[0].length;
    this.run();
    return this.drain(this.inCount);
  }

  flush(): Planar {
    const F = this.frameSize;
    // Pad the last partial frame and push enough silence to flush the delay.
    const pad = (F - (this.pending[0].length % F)) % F + (this.enabled ? Math.ceil(this.insts[0].delay / F) * F : 0);
    for (let c = 0; c < this.channels; c++) this.pending[c].push(new Float32Array(pad));
    this.run();
    return this.drain(this.inCount);
  }

  /** Median change in background level between words, in dB (negative = quieter). */
  noiseReductionDb(): number | null {
    if (!this.enabled || this.before.length < 20) return null;
    const med = (x: number[]) => x.slice().sort((p, q) => p - q)[Math.floor(x.length / 2)];
    return med(this.after) - med(this.before);
  }
}

/** Per-hop level features used to compute the dynamics gain curves. */
export class HopFeatures {
  private kw: Cascade[];
  private xo: LR4Crossover[];
  private kAcc: F32List = new F32List();
  private rAcc: F32List = new F32List();
  private hAcc: F32List = new F32List();
  private pos = 0; // samples into the current hop
  private k = 0;
  private r = 0;
  private h = 0;
  private tmp = new Float64Array(0);
  private lo = new Float32Array(0);
  private hi = new Float32Array(0);

  constructor(
    sampleRate: number,
    private channels: number,
    private hop: number,
    crossoverHz: number,
  ) {
    this.kw = Array.from({ length: channels }, () => kWeighting(sampleRate));
    this.xo = Array.from({ length: channels }, () => new LR4Crossover(crossoverHz, sampleRate));
  }

  push(p: Planar): void {
    const n = p[0].length;
    if (this.tmp.length < n) {
      this.tmp = new Float64Array(n);
      this.lo = new Float32Array(n);
      this.hi = new Float32Array(n);
    }
    const kSq = new Float64Array(n), rSq = new Float64Array(n), hSq = new Float64Array(n);
    for (let c = 0; c < this.channels; c++) {
      const t = this.tmp.subarray(0, n);
      t.set(p[c]);
      this.kw[c].process64(t);
      const hi = this.hi.subarray(0, n);
      this.xo[c].split(p[c], this.lo.subarray(0, n), hi);
      const x = p[c];
      for (let i = 0; i < n; i++) {
        kSq[i] += t[i] * t[i];
        rSq[i] += x[i] * x[i];
        hSq[i] += hi[i] * hi[i];
      }
    }
    const norm = 1 / (this.hop * this.channels);
    for (let i = 0; i < n; i++) {
      this.k += kSq[i];
      this.r += rSq[i];
      this.h += hSq[i];
      if (++this.pos === this.hop) {
        this.kAcc.push(this.k * norm);
        this.rAcc.push(this.r * norm);
        this.hAcc.push(this.h * norm);
        this.k = this.r = this.h = 0;
        this.pos = 0;
      }
    }
  }

  /** Mean-square power per hop: K-weighted, unweighted, and above the crossover. */
  finish(): { k: Float32Array; raw: Float32Array; hf: Float32Array } {
    if (this.pos > 0) {
      const norm = 1 / (this.pos * this.channels);
      this.kAcc.push(this.k * norm);
      this.rAcc.push(this.r * norm);
      this.hAcc.push(this.h * norm);
      this.pos = 0;
    }
    return { k: this.kAcc.view(), raw: this.rAcc.view(), hf: this.hAcc.view() };
  }
}
