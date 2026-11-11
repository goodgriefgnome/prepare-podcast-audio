import { createMp3Encoder } from "wasm-media-encoders";
import { MPEGDecoder } from "mpg123-decoder";
import { butterworthHighPass, LR4Crossover } from "../dsp/biquad.ts";
import { LoudnessMeter } from "../dsp/loudness.ts";
import { Resampler } from "../dsp/resampler.ts";
import { TruePeakLimiter, TruePeakMeter } from "../dsp/truepeak.ts";
import type { AudioSource, Planar } from "../io/source.ts";
import { readBytes } from "../io/source.ts";
import type { Storage } from "../io/store.ts";
import type { DenoiserFactory } from "./denoise.ts";
import { PLAN_CONTEXT_HOPS, planGains } from "./plan.ts";
import { DenoiseStream, F32List, Fifo, HopFeatures } from "./stream.ts";

export const WORK_RATE = 48000;
export const OUT_RATE = 44100;
export const TARGET_LUFS = -16;
export const MAX_TRUE_PEAK = -1;
/** How far from the target the finished MP3 may be before it's reported as a miss (LU). */
export const LOUDNESS_TOLERANCE = 0.5;
const HOP = 240; // 5 ms at 48 kHz
const SEGMENT_HOPS = 4000; // gains are planned 20 s at a time
const CROSSOVER_HZ = 5000; // de-esser band split

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
  denoiser: DenoiserFactory;
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

  const expected = Math.max(1, source.info.duration * source.info.sampleRate);

  // ---- Front end: decode → mix → 48 kHz → high-pass → noise reduction.
  const toWork = new Resampler(source.info.sampleRate, WORK_RATE, C);
  const hp = Array.from({ length: C }, () =>
    opts.rumble ? butterworthHighPass(4, 80, WORK_RATE) : butterworthHighPass(2, 20, WORK_RATE),
  );
  const inputMeter = new LoudnessMeter(WORK_RATE, C);
  const insts = await Promise.all(Array.from({ length: opts.noise ? C : 1 }, () => deps.denoiser.create()));
  const denoise = new DenoiseStream(insts, C, opts.noise);
  const features = new HopFeatures(WORK_RATE, C, HOP, CROSSOVER_HZ);
  const delay = Array.from({ length: C }, () => new Fifo());
  let cleaned = 0; // samples that went into the delay line

  // ---- Gain plan, finalized a segment at a time.
  const gb = new F32List(); // linear broadband gain per hop
  const gh = new F32List(); // linear gain above the crossover per hop
  let spikesTamed = 0;
  const vadAt = (h: number) => {
    const v = denoise.vad.view();
    return v[Math.min(v.length - 1, Math.floor((h * HOP) / denoise.frameSize))] ?? 0;
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

  const afterResample = (p: Planar) => {
    if (!p[0].length) return;
    inputMeter.push(p);
    for (let c = 0; c < C; c++) hp[c].process(p[c]);
    afterDenoise(denoise.push(p));
  };
  const afterDenoise = (p: Planar) => {
    if (!p[0].length) return;
    features.push(p);
    for (let c = 0; c < C; c++) delay[c].push(p[c]);
    cleaned += p[0].length;
  };

  let decoded = 0;
  try {
    for await (const chunk of source.chunks()) {
      checkCancel();
      decoded += chunk[0].length;
      afterResample(toWork.push(mix(chunk, C)));
      planUpTo(false);
      apply(false);
      report("process", decoded / expected);
      await yieldToEvents();
      if (deps.gate) await deps.gate();
    }
    afterResample(toWork.flush());
    afterDenoise(denoise.flush());
  } finally {
    for (const d of insts) d.destroy();
  }
  if (cleaned < WORK_RATE * 0.5) throw new Error("This recording is too short to process (under half a second).");
  features.finish();
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
    inputLufs: inputMeter.integrated(),
    noiseReductionDb: denoise.noiseReductionDb(),
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
