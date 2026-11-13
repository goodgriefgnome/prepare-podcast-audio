import "../style.css";
import "./restore.css";
import { openAudio } from "../io/mp3.ts";
import type { Method, RestoreRequest, RestoreResponse, VersionKey } from "./restore-worker.ts";
// Only the constant: importing it from the worker module would bundle the worker into the page.
const MODEL_FILES: Record<Exclude<Method, "rnnoise">, string[]> = {
  sidon: ["sidon_predictor.f16.onnx", "sidon_vocoder.f16.onnx"],
  pase: ["pase_encoder.f16.onnx", "pase_vocoder.f16.onnx"],
  flowse: ["flowse_mel.onnx", "flowse_velocity.f16.onnx", "flowse_vocoder.f16.onnx"],
  flowse_text: ["flowse_mel.onnx", "flowse_velocity.f16.onnx", "flowse_vocoder.f16.onnx", "flowse_vocab.txt"],
};

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const SR = 44100;
const TARGET = -16;

const LABELS: Record<VersionKey, string> = {
  original: "Original",
  rnnoise: "App’s noise reduction",
  sidon: "Sidon",
  pase: "PASE",
  flowse: "FlowSE",
  flowse_text: "FlowSE with transcript",
};
const METHODS: Method[] = ["rnnoise", "sidon", "pase", "flowse", "flowse_text"];
const ORDER: VersionKey[] = ["original", ...METHODS];
/** Models that run from files the person picks. */
const LOCAL = METHODS.filter((m): m is Exclude<Method, "rnnoise"> => m !== "rnnoise");

function clock(s: number): string {
  s = Math.max(0, Math.floor(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
function parseClock(v: string): number {
  const p = v.trim().split(":").map(Number);
  if (p.some((n) => !Number.isFinite(n) || n < 0)) return NaN;
  return p.reduce((a, n) => a * 60 + n, 0);
}

// ------------------------------------------------------------------ versions

interface Version {
  key: VersionKey;
  row: HTMLLIElement;
  status: HTMLElement;
  bar: HTMLElement;
  audio: Float32Array | null;
  lufs: number;
  urls: { raw?: string; matched?: string };
}
const versions = new Map<VersionKey, Version>();
let current: VersionKey | null = null;

function wavBlob(x: Float32Array, gain: number): Blob {
  const n = x.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const w = (o: number, s: string) => [...s].forEach((c, i) => dv.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); w(8, "WAVE"); w(12, "fmt ");
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, SR, true); dv.setUint32(28, SR * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  w(36, "data"); dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) dv.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(x[i] * gain * 32768))), true);
  return new Blob([buf], { type: "audio/wav" });
}

/** Gain to the target loudness, kept low enough that peaks don't clip. */
function matchGain(v: Version): number {
  if (!v.audio || !Number.isFinite(v.lufs)) return 1;
  let peak = 1e-9;
  for (const s of v.audio) peak = Math.max(peak, Math.abs(s));
  return Math.min(Math.pow(10, (TARGET - v.lufs) / 20), 0.99 / peak);
}

function urlFor(v: Version): string | null {
  if (!v.audio) return null;
  const matched = $<HTMLInputElement>("match").checked;
  if (matched) return (v.urls.matched ??= URL.createObjectURL(wavBlob(v.audio, matchGain(v))));
  return (v.urls.raw ??= URL.createObjectURL(wavBlob(v.audio, 1)));
}

function resetVersions(methods: Method[]) {
  for (const v of versions.values()) {
    if (v.urls.raw) URL.revokeObjectURL(v.urls.raw);
    if (v.urls.matched) URL.revokeObjectURL(v.urls.matched);
  }
  versions.clear();
  const list = $("versions");
  list.replaceChildren();
  current = null;
  const audio = $<HTMLAudioElement>("audio");
  audio.pause();
  audio.removeAttribute("src");
  for (const key of ORDER.filter((k) => k === "original" || methods.includes(k as Method))) {
    const row = document.createElement("li");
    row.className = "version";
    row.innerHTML = `<button class="pick-version" type="button" disabled><span class="key num"></span><span class="name"></span></button>
      <span class="status hint"></span><a class="btn" hidden download>WAV</a><div class="vbar"><div></div></div>`;
    row.querySelector(".name")!.textContent = LABELS[key];
    row.querySelector(".key")!.textContent = String(versions.size + 1);
    const v: Version = { key, row, status: row.querySelector(".status")!, bar: row.querySelector(".vbar > div")!, audio: null, lufs: NaN, urls: {} };
    v.status.textContent = key === "original" ? "Reading…" : "Waiting";
    row.querySelector("button")!.addEventListener("click", () => select(key));
    versions.set(key, v);
    list.append(row);
  }
  $("empty").hidden = true;
}

function refreshLinks(v: Version) {
  const a = v.row.querySelector("a")!;
  const url = urlFor(v);
  a.hidden = !url;
  if (url) {
    a.href = url;
    a.download = `${$<HTMLInputElement>("file").files?.[0]?.name.replace(/\.[^.]+$/, "") ?? "excerpt"}-${v.key}.wav`;
  }
}

function select(key: VersionKey) {
  const v = versions.get(key);
  if (!v?.audio) return;
  const audio = $<HTMLAudioElement>("audio");
  const t = audio.currentTime || 0;
  const playing = !audio.paused;
  current = key;
  audio.src = urlFor(v)!;
  audio.currentTime = t;
  if (playing) void audio.play();
  for (const o of versions.values()) o.row.classList.toggle("is-current", o.key === key);
  $("player").hidden = false;
}

// ------------------------------------------------------------------ player

const audio = $<HTMLAudioElement>("audio");
audio.addEventListener("play", () => ($("play").textContent = "❚❚"));
audio.addEventListener("pause", () => ($("play").textContent = "▶"));
audio.addEventListener("timeupdate", () => {
  const d = audio.duration || 0;
  $("time").textContent = `${clock(audio.currentTime)} / ${clock(d)}`;
  if (d) $<HTMLInputElement>("seek").value = String(Math.round((audio.currentTime / d) * 1000));
});
$("play").addEventListener("click", () => (audio.paused ? void audio.play() : audio.pause()));
$<HTMLInputElement>("seek").addEventListener("input", (e) => {
  if (audio.duration) audio.currentTime = (Number((e.target as HTMLInputElement).value) / 1000) * audio.duration;
});
document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  const n = Number(e.key);
  const shown = [...versions.keys()];
  if (n >= 1 && n <= shown.length) select(shown[n - 1]);
  else if (e.key === " " && current) {
    e.preventDefault();
    audio.paused ? void audio.play() : audio.pause();
  }
});
$<HTMLInputElement>("match").addEventListener("change", () => {
  for (const v of versions.values()) refreshLinks(v);
  if (current) select(current);
});

// ------------------------------------------------------------------ settings

let duration = 0;
const fileInput = $<HTMLInputElement>("file");
fileInput.addEventListener("change", async () => {
  const f = fileInput.files?.[0];
  $<HTMLButtonElement>("run").disabled = !f;
  if (!f) return;
  try {
    const src = await openAudio(f, f.name);
    duration = src.info.duration;
    $("file-info").textContent = `${f.name} · ${clock(duration)} · ${(src.info.sampleRate / 1000).toFixed(1)} kHz`;
  } catch (err) {
    $("file-info").textContent = err instanceof Error ? err.message : "This file couldn’t be read.";
    $<HTMLButtonElement>("run").disabled = true;
  }
});
const setting = (id: string, fallback: number) => Number(($(id) as HTMLInputElement | null)?.value ?? fallback);

// Model files chosen from disk, by name. Picking more adds to (or replaces) them.
const localFiles: Record<string, File> = {};
function refreshLocal() {
  for (const m of LOCAL) {
    const box = $<HTMLInputElement>(`m-${m}`);
    const missing = MODEL_FILES[m].filter((n) => !localFiles[n]);
    const wasDisabled = box.disabled;
    box.disabled = missing.length > 0;
    if (box.disabled) box.checked = false;
    else if (wasDisabled && m !== "flowse_text") box.checked = true;
    const need = document.querySelector<HTMLElement>(`.need[data-for="${m}"]`)!;
    need.textContent = missing.length ? `Needs ${missing.join(", ")}.` : "Files ready.";
  }
}
$<HTMLInputElement>("model-files").addEventListener("change", (e) => {
  for (const f of (e.target as HTMLInputElement).files ?? []) localFiles[f.name] = f;
  refreshLocal();
});
refreshLocal();

// ------------------------------------------------------------------ running

let worker: Worker | null = null;
$("run").addEventListener("click", () => {
  const f = fileInput.files?.[0];
  if (!f) return;
  let start = parseClock($<HTMLInputElement>("start").value);
  if (!Number.isFinite(start)) start = 0;
  const length = Number($<HTMLSelectElement>("length").value);
  if (duration && start >= duration - 1) start = Math.max(0, duration - length);
  const methods = METHODS.filter((m) => ($(`m-${m}`) as HTMLInputElement | null)?.checked);
  const transcript = $<HTMLTextAreaElement>("transcript").value.trim();
  if (methods.includes("flowse_text") && !transcript) {
    $("backend").textContent = "Type the excerpt’s transcript under FlowSE settings, or untick “FlowSE with transcript”.";
    ($<HTMLDetailsElement>("flowse-settings")).open = true;
    return;
  }
  resetVersions(methods);
  worker?.terminate();
  worker = new Worker(new URL("./restore-worker.ts", import.meta.url), { type: "module" });
  const runBtn = $<HTMLButtonElement>("run");
  runBtn.textContent = "Running…";
  runBtn.disabled = true;
  worker.onmessage = (e: MessageEvent<RestoreResponse>) => {
    const r = e.data;
    if (r.type === "backend") $("backend").textContent = `Running on: ${r.text}`;
    else if (r.type === "status") {
      const v = versions.get(r.method === "input" ? "original" : r.method);
      if (v) {
        v.status.textContent = r.text;
        v.bar.style.width = `${Math.round(r.fraction * 100)}%`;
      }
    } else if (r.type === "result") {
      const v = versions.get(r.method)!;
      v.audio = r.audio;
      v.lufs = r.lufs;
      v.bar.style.width = "0";
      v.row.querySelector("button")!.disabled = false;
      v.row.classList.add("is-ready");
      if (r.method === "original") v.status.textContent = `${r.seconds.toFixed(0)} s · ${r.lufs.toFixed(1).replace("-", "−")} LUFS`;
      else v.status.textContent += ` · ${r.lufs.toFixed(1).replace("-", "−")} LUFS before matching`;
      refreshLinks(v);
      if (!current) select(r.method);
    } else if (r.type === "error") {
      const v = versions.get(r.method === "input" ? "original" : r.method);
      if (v) {
        v.status.textContent = `Failed: ${r.message}`;
        v.row.classList.add("is-failed");
      }
    } else if (r.type === "done") {
      runBtn.textContent = "Run again";
      runBtn.disabled = false;
    }
  };
  const req: RestoreRequest = {
    type: "run",
    file: f,
    start,
    length,
    methods,
    flowse: { steps: setting("flowse-steps", 32), transcript },
    gpu: ($("gpu") as HTMLInputElement | null)?.checked ?? false,
    localFiles: { ...localFiles },
    seed: (Math.random() * 2 ** 32) >>> 0,
  };
  worker.postMessage(req);
});

// Multithreaded WebAssembly needs cross-origin isolation, which the service
// worker adds; reload once if this page loaded before it was in control.
if ("serviceWorker" in navigator && !import.meta.env.DEV) {
  navigator.serviceWorker.register("sw.js").then(() => {
    if (!self.crossOriginIsolated && navigator.serviceWorker.controller && !sessionStorage.getItem("coi-reload")) {
      sessionStorage.setItem("coi-reload", "1");
      location.reload();
    }
  }).catch(() => {});
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (!self.crossOriginIsolated && !sessionStorage.getItem("coi-reload")) {
      sessionStorage.setItem("coi-reload", "1");
      location.reload();
    }
  });
}
