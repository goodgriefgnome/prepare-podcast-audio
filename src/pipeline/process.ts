import { createMp3Encoder } from "wasm-media-encoders";
import { MPEGDecoder } from "mpg123-decoder";
import { butterworthHighPass, LR4Crossover } from "../dsp/biquad.ts";
import { LoudnessMeter } from "../dsp/loudness.ts";
import { Resampler } from "../dsp/resampler.ts";
import { TruePeakLimiter, TruePeakMeter } from "../dsp/truepeak.ts";
import type { AudioSource, Planar } from "../io/source.ts";
import { readBytes } from "../io/source.ts";
import type { SampleStore, Storage } from "../io/store.ts";
import type { DenoiserFactory } from "./denoise.ts";
import {
  compressorGains,
  deEssGains,
  levelerGains,
  percentile,
  powerToDb,
  speechMask,
  spikeGains,
} from "./dynamics.ts";
import { DenoiseStream, HopFeatures } from "./stream.ts";

export const WORK_RATE = 48000;
export const OUT_RATE = 44100;
export const TARGET_LUFS = -16;
export const MAX_TRUE_PEAK = -1;
/** How far from the target the finished MP3 may be before it's reported as a miss (LU). */
export const LOUDNESS_TOLERANCE = 0.5;
const HOP = 240; // 5 ms at 48 kHz
const CHUNK = 48000; // samples per read of the working file
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
 * clean: decode, filter and reduce noise (writes the one working file).
 * balance: work out and measure the level changes.
 * encode: apply them, set loudness, limit peaks and encode the MP3.
 * verify: decode the MP3 and measure what listeners will hear.
 */
export type Stage = "clean" | "balance" | "encode" | "verify";

export interface Progress {
  stage: Stage;
  /** 0–1 overall. */
  fraction: number;
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
}

/** Relative cost of each stage, for the progress bar. */
const WEIGHTS: Record<Stage, number> = { clean: 55, balance: 12, encode: 25, verify: 8 };
const TOTAL_WEIGHT = Object.values(WEIGHTS).reduce((a, b) => a + b, 0);
const ORDER: Stage[] = ["clean", "balance", "encode", "verify"];

/** Temporary disk space needed for a recording, in bytes. */
export function workingBytes(durationSeconds: number, channels: number): number {
  return durationSeconds * WORK_RATE * channels * 4 + (durationSeconds * 192000) / 8;
}

const yieldToEvents = () => new Promise<void>((r) => setTimeout(r, 0));

export async function processAudio(source: AudioSource, opts: Options, deps: Deps): Promise<Result> {
  const { storage, checkCancel } = deps;
  const C = opts.channels;
  let shown = 0;
  let lastReport = 0;
  const report = (stage: Stage, f: number) => {
    const now = Date.now();
    if (now - lastReport < 100 && f < 1) return;
    lastReport = now;
    const before = ORDER.slice(0, ORDER.indexOf(stage)).reduce((s, k) => s + WEIGHTS[k], 0);
    // Never move backwards (a retry of the encode stage keeps the bar where it is).
    shown = Math.max(shown, (before + WEIGHTS[stage] * Math.min(1, Math.max(0, f))) / TOTAL_WEIGHT);
    deps.onProgress({ stage, fraction: shown });
  };

  // ---- Clean: decode → mix → 48 kHz → high-pass → noise reduction → working file.
  const work = await storage.samples("work.f32", C);
  const toWork = new Resampler(source.info.sampleRate, WORK_RATE, C);
  const hp = Array.from({ length: C }, () =>
    opts.rumble ? butterworthHighPass(4, 80, WORK_RATE) : butterworthHighPass(2, 20, WORK_RATE),
  );
  const inputMeter = new LoudnessMeter(WORK_RATE, C);
  const insts = await Promise.all(Array.from({ length: opts.noise ? C : 1 }, () => deps.denoiser.create()));
  const denoise = new DenoiseStream(insts, C, opts.noise);
  const features = new HopFeatures(WORK_RATE, C, HOP, CROSSOVER_HZ);
  const expected = Math.max(1, source.info.duration * source.info.sampleRate);
  let decoded = 0;
  const afterResample = (p: Planar) => {
    if (!p[0].length) return;
    inputMeter.push(p);
    for (let c = 0; c < C; c++) hp[c].process(p[c]);
    afterDenoise(denoise.push(p));
  };
  const afterDenoise = (p: Planar) => {
    if (!p[0].length) return;
    features.push(p);
    work.append(p);
  };
  try {
    for await (const chunk of source.chunks()) {
      checkCancel();
      decoded += chunk[0].length;
      afterResample(toWork.push(mix(chunk, C)));
      report("clean", decoded / expected);
      await yieldToEvents();
    }
    afterResample(toWork.flush());
    afterDenoise(denoise.flush());
  } finally {
    for (const d of insts) d.destroy();
  }
  const frames = work.frames;
  if (frames < WORK_RATE * 0.5) throw new Error("This recording is too short to process (under half a second).");
  const inputLufs = inputMeter.integrated();
  report("clean", 1);

  // ---- Balance: gain curves from the whole recording.
  const f = features.finish();
  const hops = f.k.length;
  const kDb = Float32Array.from(f.k, powerToDb);
  const rawDb = Float32Array.from(f.raw, powerToDb);
  const hfDb = Float32Array.from(f.hf, powerToDb);
  const vadFrames = denoise.vad.view();
  const vad = new Float32Array(hops);
  for (let h = 0; h < hops; h++) vad[h] = vadFrames[Math.min(vadFrames.length - 1, Math.floor((h * HOP) / denoise.frameSize))] ?? 0;
  const speech = speechMask(vad, kDb);

  const broadband = new Float32Array(hops);
  let spikesTamed = 0;
  if (opts.spikes) {
    const s = spikeGains(rawDb, speech);
    spikesTamed = s.events;
    for (let h = 0; h < hops; h++) broadband[h] += s.gain[h];
  }
  const after = (x: Float32Array) => Float32Array.from(x, (v, h) => v + broadband[h]);
  let speechLevel = percentile(after(kDb), speech, 0.5);
  if (opts.level) {
    const l = levelerGains(after(kDb), speech);
    for (let h = 0; h < hops; h++) broadband[h] += l.gain[h];
    speechLevel = l.target;
    const comp = compressorGains(after(kDb), l.target + 3);
    for (let h = 0; h < hops; h++) broadband[h] += comp[h];
  }
  const hfGain = opts.sibilance && !Number.isNaN(speechLevel) ? deEssGains(after(hfDb), after(rawDb), speechLevel) : null;
  if ((globalThis as { DEBUG_PPA?: boolean }).DEBUG_PPA) {
    const stat = (x: Float32Array | null, mask?: Uint8Array) => {
      if (!x) return null;
      let n = 0, active = 0, min = 0, sum = 0;
      for (let h = 0; h < x.length; h++) {
        if (mask && !mask[h]) continue;
        n++;
        sum += x[h];
        if (x[h] < -1) active++;
        min = Math.min(min, x[h]);
      }
      return { activePct: +((100 * active) / Math.max(1, n)).toFixed(1), meanDb: +(sum / Math.max(1, n)).toFixed(2), minDb: +min.toFixed(1) };
    };
    let sp = 0;
    for (const v of speech) sp += v;
    console.log("analysis", {
      speechPct: +((100 * sp) / hops).toFixed(1),
      speechLevel,
      broadbandInSpeech: stat(broadband, speech),
      deEssInSpeech: stat(hfGain, speech),
    });
  }
  const gains: Gains = {
    broadband: Float32Array.from(broadband, (d) => Math.pow(10, d / 20)),
    hf: hfGain ? Float32Array.from(hfGain, (d) => Math.pow(10, d / 20)) : null,
  };
  await yieldToEvents();

  // Measure the balanced audio's loudness (nothing is written).
  const measured = await render(work, gains, null, deps, (x) => report("balance", x));
  report("balance", 1);

  // ---- Encode and verify, correcting until the MP3 itself is on target.
  // Both choices are written as joint-stereo MP3s; for mono sound the two
  // channels are identical, which joint stereo stores almost for free.
  const bitrate = C === 1 ? 128 : 192;
  const trim = Math.min(0, opts.outputTrimDb || 0);
  // LAME's output measures slightly quieter than its input; these were
  // measured on speech for each bitrate (joint stereo, 44.1 kHz).
  const encodeLoss = bitrate === 128 ? 0.42 : 0.25;
  const gainDb = Number.isFinite(measured.lufs) ? TARGET_LUFS + encodeLoss - measured.lufs : 0;
  // Encoding can push true peaks up a little, so limit with some margin.
  const ceiling = MAX_TRUE_PEAK - 0.2;
  const enc = await render(work, gains, { gainDb, ceilingDb: ceiling, trimDb: trim, bitrate, storage }, deps, (x) => report("encode", x));
  // Decode the MP3 and measure what listeners will get. A miss is reported,
  // not corrected: it means the processing above has a bug.
  const v = await verify(await storage.blob("out.mp3"), checkCancel, (x) => report("verify", x));
  if ((globalThis as { DEBUG_PPA?: boolean }).DEBUG_PPA) console.log("encode", { gainDb, ceiling, trim, pre: enc.lufs, post: v.lufs, tp: v.tp });
  const final = { lufs: v.lufs, tp: v.tp, bytes: enc.bytes };
  report("verify", 1);
  await storage.remove("work.f32");

  return {
    outputName: "out.mp3",
    bytes: final.bytes,
    duration: frames / WORK_RATE,
    bitrate,
    channels: C,
    targetLufs: TARGET_LUFS + trim,
    inputLufs,
    outputLufs: final.lufs,
    outputTruePeak: final.tp,
    loudnessOk: Math.abs(final.lufs - (TARGET_LUFS + trim)) <= LOUDNESS_TOLERANCE,
    peakOk: final.tp <= MAX_TRUE_PEAK + trim,
    noiseReductionDb: denoise.noiseReductionDb(),
    spikesTamed,
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

interface Gains {
  /** Linear broadband gain per hop. */
  broadband: Float32Array;
  /** Linear gain above the crossover per hop (de-esser), or null. */
  hf: Float32Array | null;
}

interface EncodeSettings {
  gainDb: number;
  ceilingDb: number;
  /** Final level change after the limiter, in dB. */
  trimDb: number;
  bitrate: number;
  storage: Storage;
}

/**
 * Reads the working file, applies the gain curves and resamples to 44.1 kHz.
 * Without `enc` it only measures loudness. With it, it also applies the
 * loudness gain and true-peak limiter and encodes out.mp3.
 */
async function render(
  work: SampleStore,
  gains: Gains,
  enc: EncodeSettings | null,
  deps: Deps,
  progress: (f: number) => void,
): Promise<{ lufs: number; bytes: number }> {
  const C = work.channels;
  const frames = work.frames;
  const xo = Array.from({ length: C }, () => new LR4Crossover(CROSSOVER_HZ, WORK_RATE));
  const toOut = new Resampler(WORK_RATE, OUT_RATE, C);
  const meter = new LoudnessMeter(OUT_RATE, C);
  const lim = enc ? new TruePeakLimiter(OUT_RATE, C, enc.ceilingDb) : null;
  const g = enc ? Math.pow(10, enc.gainDb / 20) : 1;
  const trim = enc ? Math.pow(10, enc.trimDb / 20) : 1;
  let encoder: Awaited<ReturnType<typeof createMp3Encoder>> | null = null;
  let out: Awaited<ReturnType<Storage["bytes"]>> | null = null;
  if (enc) {
    encoder = await createMp3Encoder();
    encoder.configure({ channels: 2, sampleRate: OUT_RATE, bitrate: enc.bitrate as 128 | 192 });
    out = await enc.storage.bytes("out.mp3");
  }
  const lo = new Float32Array(CHUNK);
  const hi = new Float32Array(CHUNK);
  const gb = gains.broadband;
  const gh = gains.hf;
  const last = gb.length - 1;

  const emit = (p: Planar) => {
    if (!p[0].length) return;
    if (trim !== 1) for (const ch of p) for (let i = 0; i < ch.length; i++) ch[i] *= trim;
    meter.push(p);
    if (encoder && out) out.append(encoder.encode(C === 1 ? [p[0], p[0]] : p).slice());
  };
  const finalStage = (p: Planar) => {
    if (!p[0].length) return;
    if (lim) {
      if (g !== 1) for (const ch of p) for (let i = 0; i < ch.length; i++) ch[i] *= g;
      emit(lim.push(p));
    } else emit(p);
  };

  for (let s = 0; s < frames; s += CHUNK) {
    deps.checkCancel();
    const p = work.read(s, CHUNK);
    const n = p[0].length;
    for (let c = 0; c < C; c++) {
      const x = p[c];
      if (gh) xo[c].split(x, lo.subarray(0, n), hi.subarray(0, n));
      for (let i = 0; i < n; i++) {
        // Gains are defined at hop centres; interpolate linearly between them.
        const t = (s + i - HOP / 2) / HOP;
        let a: number, b: number, fr: number;
        if (t <= 0) { a = b = 0; fr = 0; }
        else {
          a = t | 0;
          if (a >= last) { a = b = last; fr = 0; }
          else { b = a + 1; fr = t - a; }
        }
        const gBroad = gb[a] + fr * (gb[b] - gb[a]);
        x[i] = gh ? (lo[i] + hi[i] * (gh[a] + fr * (gh[b] - gh[a]))) * gBroad : x[i] * gBroad;
      }
    }
    finalStage(toOut.push(p));
    progress(s / frames);
    await yieldToEvents();
  }
  finalStage(toOut.flush());
  if (lim) emit(lim.flush());
  let bytes = 0;
  if (encoder && out && enc) {
    out.append(encoder.finalize().slice());
    bytes = out.size;
    await enc.storage.close("out.mp3");
  }
  return { lufs: meter.integrated(), bytes };
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
