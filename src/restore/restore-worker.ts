/// <reference lib="webworker" />
/**
 * Runs speech restoration models on an excerpt, for the comparison page.
 *
 * Models (ONNX, exported by restore/export_*.py):
 *  - VoiceFixer (Liu et al. 2022): one graph, 30 s segments.
 *  - Resemble Enhance (Resemble AI 2023), enhancer: front (mels, denoiser, latent),
 *    velocity (flow-matching ODE, run here), decode, vocoder (run on short windows).
 * Plus the app's own noise reduction (RNNoise) as the baseline.
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
import { toneMatch } from "./tone.ts";

export type Method = "rnnoise" | "voicefixer" | "resemble";
/** Versions shown on the page: the input, each method, and VoiceFixer with the recording's tone put back. */
export type VersionKey = Method | "original" | "vftone";

export interface ResembleSettings {
  /** Velocity evaluations (midpoint solver: two per step). */
  nfe: number;
  /** How much the condition leans on the denoised recording (0–1). */
  lambd: number;
  /** Prior temperature: how much of the start point is noise rather than the recording (0–1). */
  tau: number;
}

export type RestoreRequest = {
  type: "run";
  file: File;
  /** Excerpt, in seconds. */
  start: number;
  length: number;
  methods: Method[];
  resemble: ResembleSettings;
  /** Where the .onnx files are (absolute URL, ending in /). */
  modelBase: string;
  seed: number;
};

export type RestoreResponse =
  | { type: "backend"; text: string }
  | { type: "status"; method: VersionKey | "input"; text: string; fraction: number }
  | { type: "result"; method: VersionKey; audio: Float32Array; seconds: number; lufs: number }
  | { type: "error"; method: VersionKey | "input"; message: string }
  | { type: "done" };

declare const self: DedicatedWorkerGlobalScope;
const post = (r: RestoreResponse, transfer: Transferable[] = []) => self.postMessage(r, transfer);

const SR = 44100;
export const MODEL_FILES: Record<Exclude<Method, "rnnoise">, string[]> = {
  voicefixer: ["vf.f16.onnx"],
  resemble: ["re_front.f16.onnx", "re_velocity.f16.onnx", "re_decode.f16.onnx", "re_vocoder.f16.onnx"],
};

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

const CACHE = "ppa-restore-models-v1";

/** index.json (written by restore-models/index.py): each model's parts, size and hash. */
type ModelIndex = Record<string, { parts: string[]; size: number; sha256: string }>;
let index: Promise<ModelIndex> | null = null;

/**
 * Fetches a model, from the browser's cache when it's been downloaded before.
 * Models are stored in parts under 100 MB (see restore-models/build.sh).
 */
async function modelBytes(base: string, name: string, onProgress: (loaded: number, total: number) => void): Promise<ArrayBuffer> {
  index ??= fetch(new URL("index.json", base)).then((r) => {
    if (!r.ok) throw new Error(`Couldn't find the models (${r.status})`);
    return r.json() as Promise<ModelIndex>;
  });
  const entry = (await index)[name];
  if (!entry) throw new Error(`${name} isn't available`);
  // The hash is part of the key, so a rebuilt model is downloaded again.
  const key = new URL(`${name}?sha256=${entry.sha256}`, base).href;
  const cache = await caches.open(CACHE).catch(() => null);
  const hit = await cache?.match(key);
  if (hit) return hit.arrayBuffer();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (const part of entry.parts) {
    const res = await fetch(new URL(part, base));
    if (!res.ok || !res.body) throw new Error(`Couldn't download ${part} (${res.status})`);
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress(loaded, entry.size);
    }
  }
  if (loaded !== entry.size) throw new Error(`${name} downloaded incompletely (${loaded} of ${entry.size} bytes)`);
  const blob = new Blob(chunks as BlobPart[]);
  // Drop older versions of this model before keeping the new one.
  for (const old of (await cache?.keys()) ?? []) if (old.url.includes(`/${name}?`)) await cache?.delete(old);
  await cache?.put(key, new Response(blob, { headers: { "content-type": "application/octet-stream" } })).catch(() => {});
  return blob.arrayBuffer();
}

/**
 * Creates a session on WebGPU when the model is known to run correctly there
 * (`gpuOk`) and the browser has it, otherwise on the CPU.
 */
async function session(req: RestoreRequest, method: Method, name: string, what: string, gpuOk: boolean): Promise<ort.InferenceSession> {
  const b = await pickBackend();
  const eps = gpuOk && b.gpu ? ["webgpu", "wasm"] : ["wasm"];
  post({ type: "backend", text: eps[0] === "webgpu" ? `WebGPU (CPU fallback: ${b.cpu})` : b.cpu });
  const bytes = await modelBytes(req.modelBase, name, (l, t) =>
    post({ type: "status", method, text: `Downloading ${what}: ${(l / 1e6).toFixed(0)}${t ? ` of ${(t / 1e6).toFixed(0)}` : ""} MB`, fraction: t ? l / t : 0 }),
  );
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

async function runRnnoise(x: Float32Array, report: VersionKey = "rnnoise"): Promise<Float32Array> {
  post({ type: "status", method: report, text: "Reducing noise…", fraction: 0 });
  const r = await runBlock({ start: 0, end: null, from: 0, input: [x] }, { channels: 1, rumble: true, noise: true, inRate: SR }, rnnoiseFactory);
  const rs = new Resampler(WORK_RATE, SR, 1);
  return concat([rs.push(r.audio)[0], rs.flush()[0]]).subarray(0, x.length).slice();
}

async function runVoiceFixer(req: RestoreRequest, x: Float32Array): Promise<Float32Array> {
  const SEG = 30 * SR, OV = SR, HOP = SEG - OV;
  // ONNX Runtime's WebGPU backend computes VoiceFixer wrongly (the output is
  // noise), so it always runs on the CPU, where it matches the original.
  const s = await session(req, "voicefixer", "vf.f16.onnx", "VoiceFixer (216 MB)", false);
  try {
    const pieces: Float32Array[] = [];
    const count = Math.max(1, Math.ceil((x.length - OV) / HOP));
    for (let i = 0; i < count; i++) {
      post({ type: "status", method: "voicefixer", text: `Restoring part ${i + 1} of ${count}…`, fraction: i / count });
      const seg = new Float32Array(SEG);
      seg.set(x.subarray(i * HOP, i * HOP + SEG));
      const out = (await s.run({ wav: tensor(seg, [1, 1, SEG]) })).out;
      const y = out.data as Float32Array;
      // The output is a little longer than the input; VoiceFixer trims it centred.
      const off = Math.floor((y.length - SEG) / 2);
      pieces.push(y.slice(off, off + SEG));
      out.dispose();
    }
    return crossfadeJoin(pieces, HOP, OV, x.length);
  } finally {
    await s.release();
  }
}

/** Solves h(1/n) = 1/2 for Resemble's exponential time mapping h(t) = (aᵗ − 1)/(a − 1). */
function timeGrid(steps: number, n = 4): number[] {
  // With b = a^(1/n): (b − 1)/(bⁿ − 1) = 1/2. Bisection on b in (0, 1).
  const f = (b: number) => (b - 1) / (Math.pow(b, n) - 1) - 0.5;
  let lo = 1e-9, hi = 1 - 1e-9;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (f(lo) * f(mid) <= 0) hi = mid;
    else lo = mid;
  }
  const a = Math.pow((lo + hi) / 2, n);
  return Array.from({ length: steps + 1 }, (_, i) => (Math.pow(a, i / steps) - 1) / (a - 1));
}

async function runResemble(req: RestoreRequest, x: Float32Array): Promise<Float32Array> {
  const { nfe, lambd, tau } = req.resemble;
  const randn = gaussian(req.seed);
  const CH = 30 * SR, OV = SR, HOP = CH - OV, FR = 420;
  const starts: number[] = [];
  for (let s = 0; s < x.length; s += HOP) {
    starts.push(s);
    if (s + CH >= x.length) break;
  }
  const n = starts.length;
  type Part = { len: number; peak: number; cond?: Float32Array; psi?: Float32Array; F: number; h?: Float32Array; y?: Float32Array };
  const parts: Part[] = [];
  const say = (text: string, f: number) => post({ type: "status", method: "resemble", text, fraction: f });

  // 1. Mels, denoiser and the starting latent.
  let s = await session(req, "resemble", "re_front.f16.onnx", "Resemble Enhance, part 1 of 4 (143 MB)", true);
  try {
    for (let i = 0; i < n; i++) {
      say(`Analysing ${i + 1} of ${n}…`, (i / n) * 0.05);
      const seg = x.subarray(starts[i], starts[i] + CH);
      let peak = 1e-7;
      for (const v of seg) peak = Math.max(peak, Math.abs(v));
      const w = new Float32Array(seg.length + 441);
      for (let j = 0; j < seg.length; j++) w[j] = seg[j] / peak;
      const o = await s.run({ wav: tensor(w, [1, w.length]) });
      const mo = o.mel_orig.data as Float32Array, md = o.mel_den.data as Float32Array, z0 = o.z0.data as Float32Array;
      const F = o.z0.dims[2];
      const cond = new Float32Array(mo.length);
      for (let j = 0; j < mo.length; j++) cond[j] = lambd * md[j] + (1 - lambd) * mo[j];
      const noise = randn(z0.length);
      const psi = new Float32Array(z0.length);
      for (let j = 0; j < z0.length; j++) psi[j] = tau * noise[j] + (1 - tau) * z0[j];
      parts.push({ len: seg.length, peak, cond, psi, F });
      for (const t of Object.values(o)) t.dispose();
    }
  } finally {
    await s.release();
  }

  // 2. The flow-matching ODE (midpoint method), from the noisy start to the restored latent.
  s = await session(req, "resemble", "re_velocity.f16.onnx", "Resemble Enhance, part 2 of 4 (138 MB)", true);
  try {
    const steps = Math.max(1, Math.floor(nfe / 2));
    const ts = timeGrid(steps);
    for (let i = 0; i < n; i++) {
      const p = parts[i];
      const cond = tensor(p.cond!, [1, 128, p.F]);
      let psi = p.psi!;
      const v = async (t: number, at: Float32Array) => {
        const o = await s.run({ psi: tensor(at, [1, 64, p.F]), cond, t: tensor(new Float32Array([t]), [1]) });
        const out = (o.v.data as Float32Array).slice();
        o.v.dispose();
        return out;
      };
      for (let k = 0; k < steps; k++) {
        say(`Generating ${i + 1} of ${n}: step ${k + 1} of ${steps}…`, 0.05 + ((i + k / steps) / n) * 0.6);
        const t = ts[k], dt = ts[k + 1] - ts[k];
        const v1 = await v(t, psi);
        const mid = new Float32Array(psi.length);
        for (let j = 0; j < psi.length; j++) mid[j] = psi[j] + (dt * v1[j]) / 2;
        const v2 = await v(t + dt / 2, mid);
        const next = new Float32Array(psi.length);
        for (let j = 0; j < psi.length; j++) next[j] = psi[j] + dt * v2[j];
        psi = next;
      }
      p.psi = psi;
      p.cond = undefined;
    }
  } finally {
    await s.release();
  }

  // 3. Latent to vocoder features.
  s = await session(req, "resemble", "re_decode.f16.onnx", "Resemble Enhance, part 3 of 4 (102 MB)", true);
  try {
    for (const p of parts) {
      const o = await s.run({ psi1: tensor(p.psi!, [1, 64, p.F]) });
      p.h = (o.h.data as Float32Array).slice();
      o.h.dispose();
      p.psi = undefined;
    }
  } finally {
    await s.release();
  }

  // 4. Vocoder, on windows of W frames with C frames of context each side;
  // neighbouring windows overlap by X frames and are crossfaded.
  s = await session(req, "resemble", "re_vocoder.f16.onnx", "Resemble Enhance, part 4 of 4 (347 MB)", true);
  const W = 300, C = 32, X = 8;
  try {
    const totalWin = parts.reduce((t, p) => t + Math.ceil(p.F / W), 0);
    let done = 0;
    for (const p of parts) {
      const F = p.F;
      const h = p.h!;
      const noise = randn(128 * (F + 10));
      const out = new Float32Array(F * FR);
      for (let a = 0; a < F; a += W) {
        say(`Synthesising audio: ${done + 1} of ${totalWin}…`, 0.65 + (done / totalWin) * 0.35);
        const b = Math.min(F, a + W);
        const a0 = Math.max(0, a - X);
        const lo = Math.max(0, a0 - C), hi = Math.min(F, b + C);
        const hw = new Float32Array(160 * (hi - lo));
        for (let c = 0; c < 160; c++) hw.set(h.subarray(c * F + lo, c * F + hi), c * (hi - lo));
        const nw = new Float32Array(128 * (hi - lo + 10));
        for (let c = 0; c < 128; c++) nw.set(noise.subarray(c * (F + 10) + lo, c * (F + 10) + hi + 10), c * (hi - lo + 10));
        const o = await s.run({ h: tensor(hw, [1, 160, hi - lo]), noise: tensor(nw, [1, 128, hi - lo + 10]) });
        const y = o.wav.data as Float32Array;
        const seg = y.subarray((a0 - lo) * FR, (b - lo) * FR);
        const k = (a - a0) * FR;
        for (let j = 0; j < k; j++) {
          const w = (j + 0.5) / k;
          out[a0 * FR + j] = out[a0 * FR + j] * (1 - w) + seg[j] * w;
        }
        out.set(seg.subarray(k), a * FR);
        o.wav.dispose();
        done++;
      }
      p.y = out.subarray(0, p.len);
      for (let j = 0; j < p.y.length; j++) p.y[j] *= p.peak;
      p.h = undefined;
    }
  } finally {
    await s.release();
  }
  const pieces = parts.map((p) => {
    const full = new Float32Array(CH);
    full.set(p.y!);
    return full;
  });
  return crossfadeJoin(pieces, HOP, OV, x.length);
}

// ------------------------------------------------------------------ driver

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
  let cleaned: Float32Array | null = null; // the app's noise reduction, the tone reference
  for (const m of req.methods) {
    const t0 = performance.now();
    try {
      const y = m === "rnnoise" ? await runRnnoise(x) : m === "voicefixer" ? await runVoiceFixer(req, x) : await runResemble(req, x);
      if (m === "rnnoise") cleaned = y;
      post({ type: "status", method: m, text: `Done in ${((performance.now() - t0) / 1000).toFixed(0)} s`, fraction: 1 });
      send(m, y);
      if (m === "voicefixer") {
        // VoiceFixer comes out thin; match its spectral balance to the
        // noise-reduced recording (see tone.ts).
        try {
          cleaned ??= await runRnnoise(x, "vftone");
          post({ type: "status", method: "vftone", text: "Matching tone…", fraction: 0.5 });
          const z = toneMatch(y, cleaned, SR);
          post({ type: "status", method: "vftone", text: "VoiceFixer, EQ’d to the recording’s own tone", fraction: 1 });
          send("vftone", z);
        } catch (err) {
          post({ type: "error", method: "vftone", message: err instanceof Error ? err.message : String(err) });
        }
      }
    } catch (err) {
      console.error(err);
      post({ type: "error", method: m, message: err instanceof Error ? err.message : String(err) });
    }
  }
  post({ type: "done" });
};
