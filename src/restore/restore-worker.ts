/// <reference lib="webworker" />
/**
 * Runs speech restoration models on an excerpt, for the comparison page.
 *
 * Models (ONNX, exported by restore-models/export_*.py, read from files the
 * person picks): Sidon, PASE and FlowSE. Plus the app's own noise reduction
 * (RNNoise) as the baseline.
 *
 * Big sessions are created one stage at a time and released straight after,
 * to stay inside WebAssembly's 4 GB memory.
 */
import * as ort from "onnxruntime-web/webgpu";
import { LoudnessMeter } from "../dsp/loudness.ts";
import { Resampler } from "../dsp/resampler.ts";
import { openAudio } from "../io/mp3.ts";
import { rnnoiseFactory } from "../pipeline/denoise.ts";
import { runBlock, WORK_RATE } from "../pipeline/frontend.ts";
import { w2vBertFeatures } from "./w2vbert.ts";

export type Method = "rnnoise" | "sidon" | "pase" | "flowse" | "flowse_text";
/** Versions shown on the page: the input and each method. */
export type VersionKey = Method | "original";

export type RestoreRequest = {
  type: "run";
  file: File;
  /** Excerpt, in seconds. */
  start: number;
  length: number;
  methods: Method[];
  /** Run the models on WebGPU (if available). Off by default: unverified, and VoiceFixer produced noise there. */
  gpu: boolean;
  flowse: FlowseSettings;
  /** Model files the person chose from their disk, by file name. */
  localFiles: Record<string, File>;
  seed: number;
};

export interface FlowseSettings {
  /** Euler steps (FlowSE's default is 32; each runs the transformer twice). */
  steps: number;
  /** What's said in the excerpt, for the "with transcript" version. */
  transcript: string;
}

export type RestoreResponse =
  | { type: "backend"; text: string }
  | { type: "status"; method: VersionKey | "input"; text: string; fraction: number }
  | { type: "result"; method: VersionKey; audio: Float32Array; seconds: number; lufs: number }
  | { type: "error"; method: VersionKey | "input"; message: string }
  | { type: "done" };

declare const self: DedicatedWorkerGlobalScope;
const post = (r: RestoreResponse, transfer: Transferable[] = []) => self.postMessage(r, transfer);

const SR = 44100;

// ------------------------------------------------------------------ helpers

/** Seeded uniform PRNG (mulberry32) and Gaussian samples (Box–Muller). */
function gaussian(seed: number) {
  let a = seed >>> 0;
  const uni = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return (n: number) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 2) {
      const u = Math.max(uni(), 1e-12), v = uni();
      const r = Math.sqrt(-2 * Math.log(u));
      out[i] = r * Math.cos(2 * Math.PI * v);
      if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
    }
    return out;
  };
}

const tensor = (data: Float32Array, dims: number[]) => new ort.Tensor("float32", data, dims);

let backend: { gpu: boolean; cpu: string } | null = null;
async function pickBackend() {
  if (backend) return backend;
  const threads = self.crossOriginIsolated ? Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 2) - 1)) : 1;
  ort.env.wasm.numThreads = threads;
  let gpu = false;
  try {
    const nav = navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } };
    gpu = !!(nav.gpu && (await nav.gpu.requestAdapter()));
  } catch {
    gpu = false;
  }
  const cpu = `CPU, ${threads} thread${threads === 1 ? "" : "s"}${self.crossOriginIsolated ? "" : " (multithreading unavailable)"}`;
  backend = { gpu, cpu };
  return backend;
}

/**
 * Creates a session on WebGPU when the model is known to run correctly there
 * (`gpuOk`) and the browser has it, otherwise on the CPU.
 */
async function session(req: RestoreRequest, method: Method, name: string, what: string, gpuOk: boolean): Promise<ort.InferenceSession> {
  const b = await pickBackend();
  const eps = gpuOk && b.gpu ? ["webgpu", "wasm"] : ["wasm"];
  post({ type: "backend", text: eps[0] === "webgpu" ? `WebGPU (CPU fallback: ${b.cpu})` : b.cpu });
  const local = req.localFiles[name];
  if (!local) throw new Error(`${name} hasn’t been chosen`);
  post({ type: "status", method, text: `Reading ${what} from your disk…`, fraction: 0 });
  const bytes = await local.arrayBuffer();
  post({ type: "status", method, text: `Loading ${what}…`, fraction: 0 });
  try {
    return await ort.InferenceSession.create(bytes, { executionProviders: eps, graphOptimizationLevel: "all" });
  } catch (err) {
    if (eps[0] !== "webgpu") throw err;
    // Some GPUs can't run every operator; fall back to the CPU for this model.
    console.warn("WebGPU failed, using CPU", err);
    return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  }
}

/** Overlap-adds equal-length pieces with linear crossfades of `overlap` samples. */
function crossfadeJoin(pieces: Float32Array[], hop: number, overlap: number, length: number): Float32Array {
  const out = new Float32Array(length);
  pieces.forEach((p, i) => {
    const at = i * hop;
    for (let j = 0; j < p.length && at + j < length; j++) {
      let w = 1;
      if (i > 0 && j < overlap) w *= (j + 0.5) / overlap;
      if (i < pieces.length - 1 && j >= hop) w *= 1 - (j - hop + 0.5) / overlap;
      out[at + j] += p[j] * w;
    }
  });
  return out;
}

// ------------------------------------------------------------------ input

async function readExcerpt(file: File, start: number, length: number): Promise<Float32Array> {
  const src = await openAudio(file, file.name);
  const R = src.info.sampleRate;
  const a = Math.floor(start * R), b = Math.floor((start + length) * R);
  const out = new Float32Array(b - a);
  let pos = 0, n = 0;
  for await (const chunk of src.chunks()) {
    const len = chunk[0].length;
    const lo = Math.max(a, pos), hi = Math.min(b, pos + len);
    for (let i = lo; i < hi; i++) {
      let s = 0;
      for (const ch of chunk) s += ch[i - pos];
      out[i - a] = s / chunk.length;
    }
    n = Math.max(n, hi - a);
    pos += len;
    if (pos >= b) break;
  }
  const mono = out.subarray(0, n);
  if (R === SR) return mono.slice();
  const rs = new Resampler(R, SR, 1);
  const parts = [rs.push([mono])[0], rs.flush()[0]];
  return concat(parts);
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

function loudness(x: Float32Array): number {
  const m = new LoudnessMeter(SR, 1);
  m.push([x]);
  return m.integrated();
}

// ------------------------------------------------------------------ methods

async function runRnnoise(x: Float32Array): Promise<Float32Array> {
  post({ type: "status", method: "rnnoise", text: "Reducing noise…", fraction: 0 });
  const r = await runBlock({ start: 0, end: null, from: 0, input: [x] }, { channels: 1, rumble: true, noise: true, inRate: SR }, rnnoiseFactory);
  const rs = new Resampler(WORK_RATE, SR, 1);
  return concat([rs.push(r.audio)[0], rs.flush()[0]]).subarray(0, x.length).slice();
}

// ------------------------------------------------------------------ models from files on disk

function resample(x: Float32Array, from: number, to: number): Float32Array {
  const n = Math.round((x.length * to) / from);
  if (from === to) return x.slice();
  const rs = new Resampler(from, to, 1);
  return fit(concat([rs.push([x])[0], rs.flush()[0]]), n);
}

/** Pads with zeros or trims to n samples. */
function fit(y: Float32Array, n: number): Float32Array {
  if (y.length === n) return y;
  const out = new Float32Array(n);
  out.set(y.subarray(0, n));
  return out;
}

/** Splits x into windows of `seconds` overlapping by `overlap` seconds (joined back with crossfades). */
function windows(x: Float32Array, rate: number, seconds: number, overlap: number) {
  const CH = Math.round(seconds * rate), OV = Math.round(overlap * rate), HOP = CH - OV;
  const segs: Float32Array[] = [];
  for (let s = 0; s < x.length; s += HOP) {
    segs.push(x.subarray(s, s + CH));
    if (s + CH >= x.length) break;
  }
  return { segs, join: (ys: Float32Array[]) => crossfadeJoin(ys.map((y, i) => fit(y, segs[i].length)), HOP, OV, x.length) };
}

/** torchaudio.functional.highpass_biquad (Q = 0.707), output clamped to ±1 as torchaudio's lfilter does. */
function highpass(x: Float32Array, rate: number, cutoff: number): Float32Array {
  const w0 = (2 * Math.PI * cutoff) / rate, alpha = Math.sin(w0) / (2 * 0.707), c = Math.cos(w0);
  const a0 = 1 + alpha;
  const b0 = (1 + c) / 2 / a0, b1 = -(1 + c) / a0, b2 = b0, a1 = (-2 * c) / a0, a2 = (1 - alpha) / a0;
  const y = new Float32Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = v;
    y[i] = Math.max(-1, Math.min(1, v));
  }
  return y;
}

/**
 * Sidon v0.1, following its official demo: peak-normalise to 0.9, high-pass at
 * 50 Hz, 16 kHz, w2v-BERT features, cleansed by the fine-tuned w2v-BERT
 * predictor, then the vocoder (48 kHz). The demo runs 96 s chunks; here 30 s
 * windows are crossfaded, to keep the attention within WebAssembly's memory.
 */
async function runSidon(req: RestoreRequest, x: Float32Array): Promise<Float32Array> {
  const say = (text: string, f: number) => post({ type: "status", method: "sidon", text, fraction: f });
  let peak = 1e-9;
  for (const v of x) peak = Math.max(peak, Math.abs(v));
  const scaled = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) scaled[i] = (0.9 * x[i]) / peak;
  const x16 = resample(highpass(scaled, SR, 50), SR, 16000);
  const { segs, join } = windows(x16, 16000, 30, 1);
  const feats: { data: Float32Array; frames: number }[] = [];
  let s = await session(req, "sidon", "sidon_predictor.f16.onnx", "Sidon, part 1 of 2 (398 MB)", req.gpu);
  try {
    for (let i = 0; i < segs.length; i++) {
      say(`Cleaning the speech features: ${i + 1} of ${segs.length}…`, (i / segs.length) * 0.5);
      const padded = new Float32Array(segs[i].length + 320);
      padded.set(segs[i], 160);
      const f = w2vBertFeatures(padded);
      const o = await s.run({ input_features: tensor(f.data, [1, f.frames, 160]) });
      feats.push({ data: (o.features.data as Float32Array).slice(), frames: o.features.dims[1] });
      o.features.dispose();
    }
  } finally {
    await s.release();
  }
  s = await session(req, "sidon", "sidon_vocoder.f16.onnx", "Sidon, part 2 of 2 (105 MB)", req.gpu);
  const ys: Float32Array[] = [];
  try {
    for (let i = 0; i < feats.length; i++) {
      say(`Synthesising audio: ${i + 1} of ${feats.length}…`, 0.5 + (i / feats.length) * 0.5);
      const o = await s.run({ features: tensor(feats[i].data, [1, feats[i].frames, 1024]) });
      // 48 kHz; back to 16 kHz to crossfade at the windows' own rate.
      ys.push(resample(o.audio.data as Float32Array, 48000, 16000));
      o.audio.dispose();
    }
  } finally {
    await s.release();
  }
  return resample(join(ys), 16000, SR);
}

/** PASE's padding: to 320k + 80 samples (trimming if that's shorter). */
const paseLength = (n: number) => (n % 320 === 80 ? n : Math.floor(n / 320) * 320 + 80);

/**
 * PASE (16 kHz): DeWavLM layers 1 and 24, then the dual-stream vocoder; output
 * scaled to the input's peak, as PASE's inference script does. 30 s windows.
 */
async function runPase(req: RestoreRequest, x: Float32Array): Promise<Float32Array> {
  const say = (text: string, f: number) => post({ type: "status", method: "pase", text, fraction: f });
  const x16 = resample(x, SR, 16000);
  const { segs, join } = windows(x16, 16000, 30, 1);
  const feats: { a: Float32Array; p: Float32Array; T: number }[] = [];
  let s = await session(req, "pase", "pase_encoder.f16.onnx", "PASE, part 1 of 2 (632 MB)", req.gpu);
  try {
    for (let i = 0; i < segs.length; i++) {
      say(`Cleaning the speech features: ${i + 1} of ${segs.length}…`, (i / segs.length) * 0.7);
      const w = fit(segs[i].slice(), paseLength(segs[i].length));
      const o = await s.run({ wav: tensor(w, [1, w.length]) });
      feats.push({ a: (o.feat_a.data as Float32Array).slice(), p: (o.feat_p.data as Float32Array).slice(), T: o.feat_a.dims[1] });
      for (const t of Object.values(o)) t.dispose();
    }
  } finally {
    await s.release();
  }
  s = await session(req, "pase", "pase_vocoder.f16.onnx", "PASE, part 2 of 2 (137 MB)", req.gpu);
  const ys: Float32Array[] = [];
  try {
    for (let i = 0; i < feats.length; i++) {
      say(`Synthesising audio: ${i + 1} of ${feats.length}…`, 0.7 + (i / feats.length) * 0.3);
      const f = feats[i];
      const o = await s.run({ feat_p: tensor(f.p, [1, f.T, 1024]), feat_a: tensor(f.a, [1, f.T, 1024]) });
      const y = fit((o.wav.data as Float32Array).slice(), segs[i].length);
      o.wav.dispose();
      let pin = 1e-9, pout = 1e-8;
      for (const v of segs[i]) pin = Math.max(pin, Math.abs(v));
      for (const v of y) pout = Math.max(pout, Math.abs(v));
      for (let j = 0; j < y.length; j++) y[j] *= pin / pout;
      ys.push(y);
    }
  } finally {
    await s.release();
  }
  return resample(join(ys), 16000, SR);
}

/**
 * FlowSE: conditional flow matching on 24 kHz log mels (the noisy mels are the
 * condition), Euler steps with classifier-free guidance 1.0 as in its
 * inference code, then the Vocos vocoder. Its training audio was 16 kHz, so
 * the input goes through 16 kHz first, as in FlowSE's inference script.
 * 30 s windows (its text positions only reach about 44 s).
 */
async function runFlowse(req: RestoreRequest, method: "flowse" | "flowse_text", x: Float32Array): Promise<Float32Array> {
  const say = (text: string, f: number) => post({ type: "status", method, text, fraction: f });
  const steps = req.flowse.steps;
  const randn = gaussian(req.seed ^ (method === "flowse" ? 0 : 0x9e3779b9));
  const x24 = resample(resample(x, SR, 16000), 16000, 24000);
  const { segs, join } = windows(x24, 24000, 30, 1);
  let vocab: Map<string, number> | null = null;
  if (method === "flowse_text") {
    const file = req.localFiles["flowse_vocab.txt"];
    if (!file) throw new Error("flowse_vocab.txt is missing");
    const lines = (await file.text()).split("\n");
    if (lines[lines.length - 1] === "") lines.pop();
    vocab = new Map(lines.map((c, i) => [c, i]));
    if (segs.length > 1) throw new Error("The transcript can only be used on excerpts of 30 seconds or less");
  }
  // 1. Log mels of each window.
  let s = await session(req, method, "flowse_mel.onnx", "FlowSE, part 1 of 3 (4 MB)", false);
  const mels: { data: Float32Array; N: number }[] = [];
  try {
    for (const seg of segs) {
      const o = await s.run({ wav: tensor(seg.slice(), [1, seg.length]) });
      mels.push({ data: (o.mel.data as Float32Array).slice(), N: o.mel.dims[1] });
      o.mel.dispose();
    }
  } finally {
    await s.release();
  }
  // 2. The ODE from noise to clean mels.
  s = await session(req, method, "flowse_velocity.f16.onnx", "FlowSE, part 2 of 3 (680 MB)", req.gpu);
  const outs: { data: Float32Array; N: number }[] = [];
  try {
    for (let i = 0; i < mels.length; i++) {
      const { data: cond, N } = mels[i];
      const size = N * 100;
      const condB = new Float32Array(2 * size);
      condB.set(cond); // second half zero: the unconditional ("null") pass
      const text = new BigInt64Array(2 * N);
      if (vocab) {
        const chars = Array.from(req.flowse.transcript);
        for (let k = 0; k < Math.min(N, chars.length); k++) text[k] = BigInt((vocab.get(chars[k]) ?? 0) + 1);
      }
      const condT = tensor(condB, [2, N, 100]);
      const textT = new ort.Tensor("int64", text, [2, N]);
      let y = randn(size);
      for (let k = 0; k < steps; k++) {
        say(`Generating ${i + 1} of ${mels.length}: step ${k + 1} of ${steps}…`, ((i + k / steps) / mels.length) * 0.95);
        const t = k / steps, dt = 1 / steps;
        const xb = new Float32Array(2 * size);
        xb.set(y);
        xb.set(y, size);
        const o = await s.run({ x: tensor(xb, [2, N, 100]), cond: condT, text: textT, t: tensor(new Float32Array([t, t]), [2]) });
        const v = o.v.data as Float32Array;
        const next = new Float32Array(size);
        for (let j = 0; j < size; j++) next[j] = y[j] + dt * (2 * v[j] - v[size + j]);
        o.v.dispose();
        y = next;
      }
      outs.push({ data: y, N });
    }
  } finally {
    await s.release();
  }
  // 3. Vocoder (24 kHz).
  s = await session(req, method, "flowse_vocoder.f16.onnx", "FlowSE, part 3 of 3 (29 MB)", req.gpu);
  const ys: Float32Array[] = [];
  try {
    for (const m of outs) {
      say("Synthesising audio…", 0.97);
      const o = await s.run({ mel: tensor(m.data, [1, m.N, 100]) });
      ys.push((o.wav.data as Float32Array).slice());
      o.wav.dispose();
    }
  } finally {
    await s.release();
  }
  return resample(join(ys), 24000, SR);
}

// ------------------------------------------------------------------ driver

// Resemble Enhance (since removed) kept 730 MB of models in this cache.
if (self.name !== "em-pthread") caches?.delete("ppa-restore-models-v1").catch(() => {});

// In a production build ONNX Runtime is bundled into this script, so its
// WebAssembly threads ("em-pthread" workers) load this file too; they must
// keep ONNX Runtime's own message handler.
if (self.name !== "em-pthread") self.onmessage = async (e: MessageEvent<RestoreRequest>) => {
  const req = e.data;
  let x: Float32Array;
  try {
    post({ type: "status", method: "input", text: "Reading the excerpt…", fraction: 0 });
    x = await readExcerpt(req.file, req.start, req.length);
    if (x.length < SR) throw new Error("The excerpt is shorter than a second. Pick an earlier start.");
  } catch (err) {
    post({ type: "error", method: "input", message: err instanceof Error ? err.message : String(err) });
    post({ type: "done" });
    return;
  }
  const send = (method: VersionKey, y: Float32Array) => {
    const copy = y.slice();
    post({ type: "result", method, audio: copy, seconds: copy.length / SR, lufs: loudness(copy) }, [copy.buffer]);
  };
  send("original", x);
  for (const m of req.methods) {
    const t0 = performance.now();
    try {
      const y =
        m === "rnnoise" ? await runRnnoise(x)
        : m === "sidon" ? await runSidon(req, x)
        : m === "pase" ? await runPase(req, x)
        : await runFlowse(req, m, x);
      post({ type: "status", method: m, text: `Done in ${((performance.now() - t0) / 1000).toFixed(0)} s`, fraction: 1 });
      send(m, y);
    } catch (err) {
      console.error(err);
      post({ type: "error", method: m, message: err instanceof Error ? err.message : String(err) });
    }
  }
  post({ type: "done" });
};
