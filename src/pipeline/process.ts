import { createMp3Encoder } from "wasm-media-encoders";
import { MPEGDecoder } from "mpg123-decoder";
import { LR4Crossover } from "../dsp/biquad.ts";
import { gatedLoudness, LoudnessMeter } from "../dsp/loudness.ts";
import { Resampler } from "../dsp/resampler.ts";
import { TruePeakLimiter, TruePeakMeter } from "../dsp/truepeak.ts";
import type { AudioSource, Planar } from "../io/source.ts";
import { readBytes } from "../io/source.ts";
import type { Storage } from "../io/store.ts";
import {
  BLOCK_SECONDS,
  type BlockResult,
  CROSSOVER_HZ,
  type FrontEnd,
  type FrontEndOptions,
  HOP,
  LEAD_SECONDS,
  TAIL_SECONDS,
  WORK_RATE,
} from "./frontend.ts";
import { PLAN_CONTEXT_HOPS, planGains } from "./plan.ts";
import { F32List, Fifo, noiseReductionDb } from "./stream.ts";

export { WORK_RATE };
export const OUT_RATE = 44100;
export const TARGET_LUFS = -16;
export const MAX_TRUE_PEAK = -1;
/** How far from the target the finished MP3 may be before it's reported as a miss (LU). */
export const LOUDNESS_TOLERANCE = 0.5;
const SEGMENT_HOPS = 4000; // gains are planned 20 s at a time
const VAD_FRAME = 480; // noise-model frame, 10 ms

export interface Options {
  channels: 1 | 2;
  rumble: boolean;
  noise: boolean;
  spikes: boolean;
  sibilance: boolean;
  level: boolean;
  /**
   * Output level change in dB (≤ 0), applied after the peak limiter as the
   * very last step, so it lowers loudness and peaks together.
   */
  outputTrimDb: number;
}

/**
 * process: the single pass — decode, clean up, balance, set loudness,
 * limit peaks and encode the MP3.
 * verify: decode the MP3 and measure what listeners will hear.
 */
export type Stage = "process" | "verify";

export interface Progress {
  stage: Stage;
  /** 0–1 overall. */
  fraction: number;
  /** Seconds of finished MP3 written so far (it can be played while processing goes on). */
  outputSeconds: number;
}

export interface Result {
  outputName: string;
  bytes: number;
  duration: number;
  bitrate: number;
  /** 1 = mono sound (written as joint stereo with identical channels). */
  channels: 1 | 2;
  targetLufs: number;
  inputLufs: number;
  outputLufs: number;
  outputTruePeak: number;
  /** Whether the decoded MP3 is within LOUDNESS_TOLERANCE of the target. */
  loudnessOk: boolean;
  /** Whether the decoded MP3's true peak is at or below the ceiling. */
  peakOk: boolean;
  noiseReductionDb: number | null;
  spikesTamed: number;
}

export interface Deps {
  storage: Storage;
  /** Runs the front end (resample, high-pass, noise reduction, features). */
  frontEnd: FrontEnd;
  onProgress(p: Progress): void;
  /** Throws if the run was cancelled. */
  checkCancel(): void;
  /** Name for the output file in storage (default out.mp3). */
  outputName?: string;
  /** Awaited between chunks; resolves when processing may continue (used for pausing). */
  gate?: () => Promise<void>;
}

const WEIGHTS: Record<Stage, number> = { process: 92, verify: 8 };

/** Disk space needed for a recording's output, in bytes. */
export function workingBytes(durationSeconds: number, channels: number): number {
  return (durationSeconds * (channels === 1 ? 128000 : 192000)) / 8;
}

const yieldToEvents = () => new Promise<void>((r) => setTimeout(r, 0));

/**
 * One pass over the input. Audio waits in a delay line while gains are
 * planned from the level features of the minute or so around it, then is
 * processed and encoded straight away. Memory stays bounded by that delay
 * plus a few small per-5 ms arrays; the only file written is the MP3.
 */
export async function processAudio(source: AudioSource, opts: Options, deps: Deps): Promise<Result> {
  const { storage, checkCancel } = deps;
  const C = opts.channels;
  const outName = deps.outputName ?? "out.mp3";
  const bitrate = C === 1 ? 128 : 192;
  const trimDb = Math.min(0, opts.outputTrimDb || 0);
  // LAME's output measures slightly quieter than its input; these were
  // measured on speech for each bitrate (joint stereo, 44.1 kHz).
  const encodeLoss = bitrate === 128 ? 0.42 : 0.25;
  // Encoding can push true peaks up a little, so limit with some margin.
  const ceilingDb = MAX_TRUE_PEAK - 0.2;

  let lastReport = 0;
  const report = (stage: Stage, f: number) => {
    const now = Date.now();
    if (now - lastReport < 100 && f < 1) return;
    lastReport = now;
    const base = stage === "verify" ? WEIGHTS.process : 0;
    deps.onProgress({
      stage,
      fraction: (base + WEIGHTS[stage] * Math.min(1, Math.max(0, f))) / 100,
      outputSeconds: (out.size * 8) / (bitrate * 1000),
    });
  };

  const R = source.info.sampleRate;
  const expected = Math.max(1, source.info.duration * WORK_RATE);

  // ---- Front end, in blocks that may run in parallel (see frontend.ts).
  // Results are taken in order and assembled here.
  const fe = deps.frontEnd;
  const feOpts: FrontEndOptions = { channels: C, rumble: opts.rumble, noise: opts.noise, inRate: R };
  const vad = new F32List(); // voice probability per 10 ms frame
  const features = { kAcc: new F32List(), rAcc: new F32List(), hAcc: new F32List() };
  const inputSegments: Float64Array[] = [];
  const nrBefore: Float32Array[] = [];
  const nrAfter: Float32Array[] = [];
  const delay = Array.from({ length: C }, () => new Fifo());
  let cleaned = 0; // samples that went into the delay line
  // ---- Gain plan, finalized a segment at a time.
  const gb = new F32List(); // linear broadband gain per hop
  const gh = new F32List(); // linear gain above the crossover per hop
  let spikesTamed = 0;
  const vadAt = (h: number) => {
    const v = vad.view();
    return v[Math.min(v.length - 1, Math.floor((h * HOP) / VAD_FRAME))] ?? 0;
  };
  const planUpTo = (final: boolean) => {
    const avail = features.kAcc.length;
    while (gb.length < avail) {
      const h0 = gb.length;
      const h1 = Math.min(avail, h0 + SEGMENT_HOPS);
      if (!final && h1 + PLAN_CONTEXT_HOPS > avail) break;
      const w0 = Math.max(0, h0 - PLAN_CONTEXT_HOPS);
      const w1 = Math.min(avail, h1 + PLAN_CONTEXT_HOPS);
      const vad = new Float32Array(w1 - w0);
      for (let h = w0; h < w1; h++) vad[h - w0] = vadAt(h);
      const plan = planGains(
        { k: features.kAcc.view().slice(w0, w1), raw: features.rAcc.view().slice(w0, w1), hf: features.hAcc.view().slice(w0, w1), vad },
        { spikes: opts.spikes, sibilance: opts.sibilance, level: opts.level, targetLufs: TARGET_LUFS + encodeLoss, channelFactor: 2, ceilingDb },
      );
      for (let h = h0; h < h1; h++) {
        gb.push(Math.pow(10, plan.broadband[h - w0] / 20));
        gh.push(plan.hf ? Math.pow(10, plan.hf[h - w0] / 20) : 1);
        spikesTamed += plan.spikeHops[h - w0];
      }
    }
  };

  // ---- Back end: gains → 44.1 kHz → limiter → output level → MP3.
  const xo = Array.from({ length: C }, () => new LR4Crossover(CROSSOVER_HZ, WORK_RATE));
  const toOut = new Resampler(WORK_RATE, OUT_RATE, C);
  const lim = new TruePeakLimiter(OUT_RATE, C, ceilingDb);
  const trim = Math.pow(10, trimDb / 20);
  const preEncodeMeter = new LoudnessMeter(OUT_RATE, C);
  const encoder = await createMp3Encoder();
  encoder.configure({ channels: 2, sampleRate: OUT_RATE, bitrate: bitrate as 128 | 192 });
  const out = await storage.bytes(outName);
  let applied = 0; // samples taken out of the delay line
  const encode = (p: Planar) => {
    if (!p[0].length) return;
    if (trim !== 1) for (const ch of p) for (let i = 0; i < ch.length; i++) ch[i] *= trim;
    preEncodeMeter.push(p);
    out.append(encoder.encode(C === 1 ? [p[0], p[0]] : p).slice());
  };
  const apply = (final: boolean) => {
    // A sample needs the gains of the hops on both sides of it.
    const ready = final ? cleaned : Math.min(cleaned, (gb.length - 1) * HOP + HOP / 2);
    const n = ready - applied;
    if (n <= 0) return;
    const p = delay.map((d) => d.shift(n));
    const g = gb.view(), hfg = gh.view();
    const last = g.length - 1;
    const lo = new Float32Array(n), hi = new Float32Array(n);
    for (let c = 0; c < C; c++) {
      const x = p[c];
      if (opts.sibilance) xo[c].split(x, lo, hi);
      for (let i = 0; i < n; i++) {
        // Gains are defined at hop centres; interpolate linearly between them.
        const t = (applied + i - HOP / 2) / HOP;
        let a: number, b: number, fr: number;
        if (t <= 0) { a = b = 0; fr = 0; }
        else {
          a = t | 0;
          if (a >= last) { a = b = last; fr = 0; }
          else { b = a + 1; fr = t - a; }
        }
        const gBroad = g[a] + fr * (g[b] - g[a]);
        x[i] = opts.sibilance ? (lo[i] + hi[i] * (hfg[a] + fr * (hfg[b] - hfg[a]))) * gBroad : x[i] * gBroad;
      }
    }
    applied += n;
    encode(lim.push(toOut.push(p)));
  };

  let fadeIn: Planar | null = null; // previous block's audio past its end
  const consume = (r: BlockResult) => {
    if (fadeIn) {
      // Linear crossfade: the two versions are almost identical (correlated).
      const n = Math.min(fadeIn[0].length, r.audio[0].length);
      for (let c = 0; c < C; c++) {
        const a = fadeIn[c], b = r.audio[c];
        for (let i = 0; i < n; i++) {
          const w = (i + 0.5) / n;
          b[i] = a[i] * (1 - w) + b[i] * w;
        }
      }
    }
    fadeIn = r.fadeOut[0].length ? r.fadeOut : null;
    for (let c = 0; c < C; c++) delay[c].push(r.audio[c]);
    cleaned += r.audio[0].length;
    for (const v of r.vad) vad.push(v);
    for (let i = 0; i < r.k.length; i++) {
      features.kAcc.push(r.k[i]);
      features.rAcc.push(r.raw[i]);
      features.hAcc.push(r.hf[i]);
    }
    inputSegments.push(r.loudness);
    nrBefore.push(r.nrBefore);
    nrAfter.push(r.nrAfter);
  };

  // Mixed input not yet handed to a block; `bufStart` is its first sample.
  const buf = Array.from({ length: C }, () => new Fifo());
  let bufStart = 0;
  let nextStart = 0; // second the next block starts at
  const inflight: Promise<BlockResult>[] = [];
  const submit = (end: number | null) => {
    const from = Math.max(0, nextStart - LEAD_SECONDS);
    const a = from * R - bufStart;
    const b = end === null ? buf[0].length : (end + TAIL_SECONDS) * R - bufStart;
    const input = buf.map((f) => f.peek(a, b - a));
    const p = fe.run({ start: nextStart, end, from, input }, feOpts);
    p.catch(() => {}); // awaited in order below
    inflight.push(p);
    if (end !== null) {
      const drop = (end - LEAD_SECONDS) * R - bufStart;
      if (drop > 0) {
        for (const f of buf) f.shift(drop);
        bufStart += drop;
      }
      nextStart = end;
    }
  };
  const takeOldest = async () => {
    consume(await inflight.shift()!);
    checkCancel();
    planUpTo(false);
    apply(false);
    report("process", cleaned / expected);
  };

  // At most one block waits in the queue for each thread.
  const maxInflight = fe.concurrency + 1;
  for await (const chunk of source.chunks()) {
    checkCancel();
    const m = mix(chunk, C);
    for (let c = 0; c < C; c++) buf[c].push(m[c]);
    // Hand over a block once there's input past its tail (so it isn't the last).
    while (bufStart + buf[0].length > (nextStart + BLOCK_SECONDS + TAIL_SECONDS) * R) {
      submit(nextStart + BLOCK_SECONDS);
      while (inflight.length >= maxInflight) await takeOldest();
    }
    await yieldToEvents();
    if (deps.gate) await deps.gate();
  }
  submit(null);
  while (inflight.length) await takeOldest();

  if (cleaned < WORK_RATE * 0.5) throw new Error("This recording is too short to process (under half a second).");
  planUpTo(true);
  apply(true);
  encode(lim.push(toOut.flush()));
  encode(lim.flush());
  report("process", 1);
  const common = {
    outputName: outName,
    duration: cleaned / WORK_RATE,
    bitrate,
    channels: C,
    targetLufs: TARGET_LUFS + trimDb,
    inputLufs: gatedLoudness(blockPowers(inputSegments)),
    noiseReductionDb: noiseReductionDb(concat(nrBefore), concat(nrAfter)),
    spikesTamed,
  };
  out.append(encoder.finalize().slice());
  const bytes = out.size;
  await storage.close(outName);

  // ---- Verify: decode the MP3 and measure what listeners will get. A miss
  // is reported, not corrected: it means the processing above has a bug.
  const v = await verify(await storage.blob(outName), checkCancel, (x) => report("verify", x));
  if ((globalThis as { DEBUG_PPA?: boolean }).DEBUG_PPA) {
    console.log("encode", { preEncode: preEncodeMeter.integrated(), post: v.lufs, tp: v.tp, limiterMaxDb: lim.maxReduction });
  }
  report("verify", 1);
  return {
    ...common,
    bytes,
    outputLufs: v.lufs,
    outputTruePeak: v.tp,
    loudnessOk: Math.abs(v.lufs - (TARGET_LUFS + trimDb)) <= LOUDNESS_TOLERANCE,
    peakOk: v.tp <= MAX_TRUE_PEAK + trimDb,
  };
}

function concat(parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

/** BS.1770 gating blocks (400 ms, step 100 ms) from 100 ms segment sums at 48 kHz. */
function blockPowers(parts: Float64Array[]): Float64Array {
  const s = Float64Array.from(parts.flatMap((p) => Array.from(p)));
  const nb = Math.max(0, s.length - 3);
  const out = new Float64Array(nb);
  const denom = (4 * WORK_RATE) / 10;
  for (let j = 0; j < nb; j++) out[j] = (s[j] + s[j + 1] + s[j + 2] + s[j + 3]) / denom;
  return out;
}

/** Mixes or duplicates channels to `C`. */
function mix(p: Planar, C: number): Planar {
  if (p.length === C) return p;
  if (C === 1) {
    const out = new Float32Array(p[0].length);
    const k = 1 / p.length;
    for (const ch of p) for (let i = 0; i < out.length; i++) out[i] += ch[i] * k;
    return [out];
  }
  if (p.length === 1) return [p[0], p[0].slice()];
  return p.slice(0, C);
}

/** Decodes the (always two-channel) MP3 and measures it as a listener's player would. */
async function verify(
  mp3: Blob,
  checkCancel: () => void,
  progress: (f: number) => void,
): Promise<{ lufs: number; tp: number }> {
  const dec = new MPEGDecoder();
  await dec.ready;
  const C = 2;
  const meter = new LoudnessMeter(OUT_RATE, C);
  const tp = new TruePeakMeter(C);
  try {
    const step = 1 << 20;
    for (let o = 0; o < mp3.size; o += step) {
      checkCancel();
      const r = dec.decode(await readBytes(mp3, o, step));
      if (r.samplesDecoded) {
        const p = Array.from({ length: C }, (_, c) => (r.channelData[c] ?? r.channelData[0]).slice(0, r.samplesDecoded));
        meter.push(p);
        tp.push(p);
      }
      progress(o / mp3.size);
      await yieldToEvents();
    }
  } finally {
    dec.free();
  }
  tp.flush();
  return { lufs: meter.integrated(), tp: tp.dbtp };
}
