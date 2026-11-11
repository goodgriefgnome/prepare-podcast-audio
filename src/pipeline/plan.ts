import {
  compressorGains,
  deEssGains,
  HOP_SECONDS,
  LOCAL_HALF_SECONDS,
  levelerGains,
  normalizerGains,
  percentile,
  powerToDb,
  speechMask,
  spikeGains,
} from "./dynamics.ts";

export interface PlanOptions {
  spikes: boolean;
  sibilance: boolean;
  level: boolean;
  /** Loudness the gains should produce before the limiter (LUFS). */
  targetLufs: number;
  /** Channels the output plays on, for loudness (2 for joint stereo). */
  channelFactor: number;
  /** Limiter ceiling (dBTP), so the normalizer can allow for its effect. */
  ceilingDb: number;
}

/** Per-hop features (mean-square power, voice probability). */
export interface Features {
  k: Float32Array;
  raw: Float32Array;
  hf: Float32Array;
  vad: Float32Array;
}

export interface Plan {
  /** Broadband gain per hop, dB. */
  broadband: Float32Array;
  /** Gain above the de-esser crossover per hop, dB, or null if off. */
  hf: Float32Array | null;
  /** Number of spike events whose peak hop is in the window. */
  spikeHops: Uint8Array;
}

/**
 * How far either side of a hop the plan looks. The normalizer's ±30 s window
 * reads levels the leveler set using its own ±30 s window, plus a few
 * seconds for the shorter stages and smoothing.
 */
export const PLAN_CONTEXT_SECONDS = 2 * LOCAL_HALF_SECONDS + 5;
export const PLAN_CONTEXT_HOPS = Math.round(PLAN_CONTEXT_SECONDS / HOP_SECONDS);

/** Computes all gain curves for a window of features. */
export function planGains(f: Features, opts: PlanOptions): Plan {
  const n = f.k.length;
  const kDb = Float32Array.from(f.k, powerToDb);
  const rawDb = Float32Array.from(f.raw, powerToDb);
  const hfDb = Float32Array.from(f.hf, powerToDb);
  const speech = speechMask(f.vad, kDb);

  const broadband = new Float32Array(n);
  const spikeHops = new Uint8Array(n);
  if (opts.spikes) {
    const s = spikeGains(rawDb, speech);
    for (let h = 0; h < n; h++) {
      broadband[h] += s.gain[h];
      // Mark the deepest point of each reduction so events can be counted once.
      if (s.gain[h] < -3 && (h === 0 || s.gain[h - 1] >= -3)) spikeHops[h] = 1;
    }
  }
  const after = (x: Float32Array) => Float32Array.from(x, (v, h) => v + broadband[h]);
  let speechLevel: Float32Array;
  if (opts.level) {
    const l = levelerGains(after(kDb), speech);
    for (let h = 0; h < n; h++) broadband[h] += l.gain[h];
    speechLevel = l.target;
    const comp = compressorGains(after(kDb), Float32Array.from(l.target, (t) => t + 3));
    for (let h = 0; h < n; h++) broadband[h] += comp[h];
  } else {
    speechLevel = new Float32Array(n).fill(percentile(after(kDb), speech, 0.5));
  }
  const hf = opts.sibilance && !Number.isNaN(speechLevel[0]) ? deEssGains(after(hfDb), after(rawDb), speechLevel) : null;
  // Power after all gains so far, including the de-esser's cut to the top
  // band (which K-weighting boosts by about 4 dB, i.e. ×2.5 in power).
  const kPow = new Float32Array(n);
  const rawPow = new Float32Array(n);
  for (let h = 0; h < n; h++) {
    const g2 = Math.pow(10, broadband[h] / 10);
    const hfLoss = hf ? f.hf[h] * (1 - Math.pow(10, hf[h] / 10)) : 0;
    kPow[h] = Math.max(0, f.k[h] - 2.5 * hfLoss) * g2;
    rawPow[h] = Math.max(0, f.raw[h] - hfLoss) * g2;
  }
  // A 5 ms RMS level within ~3 dB of the ceiling means the limiter will act.
  const norm = normalizerGains(kPow, rawPow, opts.targetLufs, opts.ceilingDb - 3, opts.channelFactor);
  for (let h = 0; h < n; h++) broadband[h] += norm[h];
  return { broadband, hf, spikeHops };
}
