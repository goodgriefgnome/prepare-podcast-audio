import { butterworthHighPass } from "../dsp/biquad.ts";
import { LoudnessMeter } from "../dsp/loudness.ts";
import { Resampler } from "../dsp/resampler.ts";
import type { Planar } from "../io/source.ts";
import type { DenoiserFactory } from "./denoise.ts";
import { DenoiseStream, HopFeatures } from "./stream.ts";

/**
 * The front end (resample → high-pass → noise reduction → level features)
 * runs on blocks of the recording that can be processed independently, so
 * several can run at once on separate threads.
 *
 * Each block starts LEAD_SECONDS early so the filters and the noise model
 * have settled by the time it reaches the part it owns, and reads
 * TAIL_SECONDS past its end. Block edges are on whole seconds, which keeps
 * every rate (input samples, 48 kHz samples, noise-model frames, 5 ms hops
 * and 100 ms loudness segments) aligned.
 *
 * The noise model has a long memory, so a block never matches exactly what
 * one continuous run would have produced: with a 3 s lead-in the two differ
 * by about −40 to −50 dB (relative to the average level) at the join, and
 * now and then by more later on. Both are equally valid results; to keep the
 * join seamless, each block also returns FADE_SECONDS of audio past its end,
 * which is crossfaded into the start of the next block.
 */
export const WORK_RATE = 48000;
export const HOP = 240; // 5 ms at 48 kHz
export const CROSSOVER_HZ = 5000; // de-esser band split
export const BLOCK_SECONDS = 30;
export const LEAD_SECONDS = 3;
export const FADE_SECONDS = 1;
export const TAIL_SECONDS = FADE_SECONDS + 1;
const SEGMENT = WORK_RATE / 10; // 100 ms loudness segment

export interface FrontEndOptions {
  channels: 1 | 2;
  rumble: boolean;
  noise: boolean;
  /** Sample rate of the block's input. */
  inRate: number;
}

export interface Block {
  /** First second of the recording this block is responsible for. */
  start: number;
  /** Second it stops at, or null for the last block (runs to the end of the input). */
  end: number | null;
  /** Second the input starts at (≤ start; the lead-in is processed and discarded). */
  from: number;
  /** Input audio, already mixed to the output channels, at `inRate`. */
  input: Planar;
}

/** Everything from `start` (to `end`), each at its own rate. */
export interface BlockResult {
  /** Cleaned audio at 48 kHz. */
  audio: Planar;
  /** The FADE_SECONDS after `end` (empty for the last block), to crossfade into the next block. */
  fadeOut: Planar;
  /** Voice probability per noise-model frame (10 ms). */
  vad: Float32Array;
  /** Per 5 ms hop: K-weighted, unweighted and above-crossover mean-square power. */
  k: Float32Array;
  raw: Float32Array;
  hf: Float32Array;
  /** Input loudness: channel-weighted sum of squares per 100 ms segment. */
  loudness: Float64Array;
  /** Background level (dB) between words before and after noise reduction. */
  nrBefore: Float32Array;
  nrAfter: Float32Array;
}

export interface FrontEnd {
  /** How many blocks can usefully be in progress at once. */
  readonly concurrency: number;
  run(block: Block, opts: FrontEndOptions): Promise<BlockResult>;
}

/** Runs blocks one at a time on this thread. */
export function localFrontEnd(denoiser: DenoiserFactory): FrontEnd {
  return { concurrency: 1, run: (b, o) => runBlock(b, o, denoiser) };
}

/** Collects the samples of a stream that fall in [from, to). */
class Window {
  private pos = 0;
  private parts: Float32Array[] = [];
  constructor(
    private from: number,
    private to: number,
  ) {}
  push(x: Float32Array): void {
    const a = Math.max(this.from - this.pos, 0);
    const b = Math.min(this.to - this.pos, x.length);
    if (b > a) this.parts.push(x.slice(a, b));
    this.pos += x.length;
  }
  get received(): number {
    return this.pos;
  }
  result(): Float32Array {
    const n = this.parts.reduce((s, p) => s + p.length, 0);
    const out = new Float32Array(n);
    let o = 0;
    for (const p of this.parts) {
      out.set(p, o);
      o += p.length;
    }
    return out;
  }
}

const range = <T extends Float32Array | Float64Array>(x: T, a: number, b: number): T => x.slice(a, Math.min(b, x.length)) as T;

export async function runBlock(b: Block, o: FrontEndOptions, denoiser: DenoiserFactory): Promise<BlockResult> {
  const C = o.channels;
  const last = b.end === null;
  // Positions within this block's 48 kHz output.
  const skip = (b.start - b.from) * WORK_RATE;
  const stop = last ? Infinity : (b.end! - b.from) * WORK_RATE;

  const toWork = new Resampler(o.inRate, WORK_RATE, C);
  const hp = Array.from({ length: C }, () =>
    o.rumble ? butterworthHighPass(4, 80, WORK_RATE) : butterworthHighPass(2, 20, WORK_RATE),
  );
  const meter = new LoudnessMeter(WORK_RATE, C);
  const insts = await Promise.all(Array.from({ length: o.noise ? C : 1 }, () => denoiser.create()));
  const F = insts[0].frameSize;
  const denoise = new DenoiseStream(insts, C, o.noise, 18, skip / F, stop / F);
  const features = new HopFeatures(WORK_RATE, C, HOP, CROSSOVER_HZ);
  const out = Array.from({ length: C }, () => new Window(skip, stop));
  const fade = Array.from({ length: C }, () => new Window(stop, stop + FADE_SECONDS * WORK_RATE));

  const afterDenoise = (p: Planar) => {
    if (!p[0].length) return;
    features.push(p);
    for (let c = 0; c < C; c++) {
      out[c].push(p[c]);
      fade[c].push(p[c]);
    }
  };
  const afterResample = (p: Planar) => {
    if (!p[0].length) return;
    meter.push(p);
    for (let c = 0; c < C; c++) hp[c].process(p[c]);
    afterDenoise(denoise.push(p));
  };

  try {
    const n = b.input[0]?.length ?? 0;
    const step = 4 * o.inRate;
    for (let i = 0; i < n; i += step) {
      afterResample(toWork.push(b.input.map((ch) => ch.subarray(i, Math.min(n, i + step)))));
    }
    if (last) {
      afterResample(toWork.flush());
      afterDenoise(denoise.flush());
      features.finish();
    }
  } finally {
    for (const d of insts) d.destroy();
  }
  if (!last && out[0].received < stop + FADE_SECONDS * WORK_RATE) throw new Error("Front-end block ended early (not enough input).");

  const end = (per: number) => (last ? Infinity : stop / per);
  return {
    audio: out.map((w) => w.result()),
    fadeOut: fade.map((w) => w.result()),
    vad: range(denoise.vad.view(), skip / F, end(F)),
    k: range(features.kAcc.view(), skip / HOP, end(HOP)),
    raw: range(features.rAcc.view(), skip / HOP, end(HOP)),
    hf: range(features.hAcc.view(), skip / HOP, end(HOP)),
    loudness: range(meter.segments.view(), skip / SEGMENT, end(SEGMENT)),
    nrBefore: denoise.before.view().slice(),
    nrAfter: denoise.after.view().slice(),
  };
}

/** Transferable buffers of a result, for posting between threads. */
export function transferables(r: BlockResult): ArrayBuffer[] {
  return [...r.audio, ...r.fadeOut, r.vad, r.k, r.raw, r.hf, r.loudness, r.nrBefore, r.nrAfter].map((a) => a.buffer as ArrayBuffer);
}
