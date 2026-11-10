import "./style.css";
import type { AudioInfo } from "./io/source.ts";
import { MAX_TRUE_PEAK, TARGET_LUFS, workingBytes, type Options, type Result, type Stage } from "./pipeline/process.ts";
import type { Request, Response } from "./worker.ts";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const views = {
  empty: $("view-empty"),
  ready: $("view-ready"),
  proc: $("view-proc"),
  done: $("view-done"),
};
type View = keyof typeof views;
function show(v: View) {
  for (const [k, el] of Object.entries(views)) el.hidden = k !== v;
}

let worker = newWorker();
function newWorker(): Worker {
  const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  w.onmessage = (e: MessageEvent<Response>) => onWorkerMessage(e.data);
  w.onerror = (e) => {
    console.error(e);
    failed("Something went wrong while processing. Please try again.");
  };
  return w;
}
const send = (r: Request) => worker.postMessage(r);

let file: File | null = null;
let info: AudioInfo | null = null;
let channels: 1 | 2 = 1;
let outputUrl: string | null = null;

// ---------------------------------------------------------------- formatting

function fmtDuration(s: number): string {
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return `${h ? h + ":" : ""}${mm}:${String(sec).padStart(2, "0")}`;
}
function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(b >= 1e8 ? 0 : 1)} MB`;
  return `${Math.max(1, Math.round(b / 1e3))} KB`;
}
const minus = (s: string) => s.replace("-", "−");
const fmtDb = (v: number, digits = 1) => (Number.isFinite(v) ? minus(v.toFixed(digits)) : "—");
function fmtRate(r: number): string {
  return `${(r / 1000).toFixed(r % 1000 ? 1 : 0)} kHz`;
}

// ---------------------------------------------------------------- choose

const fileInput = $<HTMLInputElement>("file");
const drop = $("drop");
fileInput.addEventListener("change", () => {
  if (fileInput.files?.[0]) choose(fileInput.files[0]);
  fileInput.value = "";
});
for (const t of [drop, document.body]) {
  t.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (!views.empty.hidden) drop.classList.add("over");
  });
  t.addEventListener("dragleave", () => drop.classList.remove("over"));
  t.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = (e as DragEvent).dataTransfer?.files?.[0];
    if (f && (views.empty.hidden === false || views.ready.hidden === false || views.done.hidden === false)) choose(f);
  });
}

function emptyError(msg: string | null) {
  const el = $("empty-error");
  el.textContent = msg ?? "";
  el.hidden = !msg;
}

function choose(f: File) {
  if (!views.proc.hidden) return;
  emptyError(null);
  file = f;
  info = null;
  send({ type: "probe", file: f });
}

function onInfo(i: AudioInfo) {
  info = i;
  $("ready-name").textContent = file!.name;
  const ch = i.channels === 1 ? "mono" : i.channels === 2 ? "stereo" : `${i.channels} channels`;
  $("ready-meta").textContent = [
    (i.durationIsEstimate ? "about " : "") + fmtDuration(i.duration),
    fmtRate(i.sampleRate),
    `${i.detail} ${ch}`,
    fmtBytes(file!.size),
  ].join(" · ");
  $<HTMLButtonElement>("ch-stereo").disabled = i.channels === 1;
  setChannels(1);
  $("ready-error").hidden = true;
  show("ready");
}

// ---------------------------------------------------------------- settings

function setChannels(c: 1 | 2) {
  channels = c;
  $("ch-mono").setAttribute("aria-pressed", String(c === 1));
  $("ch-stereo").setAttribute("aria-pressed", String(c === 2));
  const monoIn = info?.channels === 1;
  $("ch-hint").textContent = monoIn
    ? "Your recording is mono, so both speakers get the same sound."
    : c === 1
      ? "Best for voice: both speakers get the same sound, so nobody is stuck in one ear."
      : "Keeps the left–right mix. Use if you have music or a deliberate stereo mix. The file is 50% larger.";
  const kbps = c === 1 ? 128 : 192;
  $("spec-line").textContent = `MP3 · joint stereo${c === 1 ? " (mono sound)" : ""} · ${kbps}\u00a0kbps CBR · 44.1\u00a0kHz`;
  $("spec-size").textContent = info ? fmtBytes((info.duration * kbps * 1000) / 8).replace(" ", "\u00a0") : "";
}

// Output level: a plain gain after the limiter, in 3 dB steps.
const trimInput = $<HTMLInputElement>("trim");
const TRIM_KEY = "ppa.outputTrimDb";
try {
  const saved = localStorage.getItem(TRIM_KEY);
  if (saved !== null && Number.isFinite(Number(saved))) trimInput.value = saved;
} catch {
  /* storage unavailable */
}
function trimDb(): number {
  return Math.min(0, Math.round(Number(trimInput.value) / 3) * 3);
}
function updateTrim() {
  const t = trimDb();
  const nb = (s: string) => s.replace(/ /g, "\u00a0");
  $("trim-value").textContent = t === 0 ? "0 dB" : `${minus(String(t))} dB`;
  $("trim-hint").textContent =
    t === 0
      ? "Apple’s recommended loudness. Turn it down if your episodes sound louder than other shows."
      : `${minus(String(-t))} dB quieter than Apple’s recommendation, applied after everything else.`;
  $("spec-loud").textContent = `${nb(`${minus(String(-16 + t))} LUFS`)} integrated · peaks ≤ ${nb(`${minus(String(-1 + t))} dBTP`)}`;
  $("spec-note").textContent = t === 0 ? "Meets Apple Podcasts guidelines." : "Quieter than Apple’s −16 LUFS recommendation.";
  try {
    localStorage.setItem(TRIM_KEY, String(t));
  } catch {
    /* storage unavailable */
  }
}
trimInput.addEventListener("input", updateTrim);
updateTrim();
$("ch-mono").addEventListener("click", () => setChannels(1));
$("ch-stereo").addEventListener("click", () => {
  if (info?.channels !== 1) setChannels(2);
});
$("change").addEventListener("click", () => {
  show("empty");
  fileInput.click();
});

function options(): Options {
  const on = (id: string) => $<HTMLInputElement>(id).checked;
  return {
    channels,
    rumble: on("c-rumble"),
    noise: on("c-noise"),
    spikes: on("c-spikes"),
    sibilance: on("c-sibilance"),
    level: on("c-level"),
    outputTrimDb: trimDb(),
  };
}

// ---------------------------------------------------------------- process

const STEP_OF: Record<Stage, number> = { clean: 0, balance: 1, encode: 2, verify: 2 };
let startedAt = 0;
let wakeLock: { release(): Promise<void> } | null = null;

async function keepAwake(on: boolean) {
  try {
    if (on && !wakeLock && document.visibilityState === "visible" && "wakeLock" in navigator) {
      wakeLock = await (navigator as unknown as { wakeLock: { request(t: "screen"): Promise<{ release(): Promise<void> }> } }).wakeLock.request("screen");
    } else if (!on && wakeLock) {
      await wakeLock.release();
      wakeLock = null;
    }
  } catch {
    wakeLock = null;
  }
}
document.addEventListener("visibilitychange", () => {
  if (!views.proc.hidden) {
    wakeLock = null; // released automatically when hidden
    keepAwake(true);
  }
});

async function enoughSpace(): Promise<string | null> {
  if (!info || !navigator.storage?.estimate) return null;
  try {
    const { quota, usage } = await navigator.storage.estimate();
    if (!quota) return null;
    const need = workingBytes(info.duration, channels) * 1.05;
    const free = quota - (usage ?? 0);
    if (free < need) return `There isn’t enough free storage on this device to process this recording. It needs about ${fmtBytes(need)} of temporary space; about ${fmtBytes(free)} is available.`;
  } catch {
    /* unknown: try anyway */
  }
  return null;
}

$("go").addEventListener("click", async () => {
  if (!file || !info) return;
  const problem = await enoughSpace();
  if (problem) {
    $("ready-error").textContent = problem;
    $("ready-error").hidden = false;
    return;
  }
  $("proc-name").textContent = file.name;
  $("proc-duration").textContent = fmtDuration(info.duration);
  setProgress("clean", 0);
  $("eta").textContent = "Starting…";
  startedAt = performance.now();
  show("proc");
  keepAwake(true);
  send({ type: "process", file, options: options() });
});

$("cancel").addEventListener("click", () => {
  // Terminating is immediate and releases the temporary files' locks.
  worker.terminate();
  worker = newWorker();
  keepAwake(false);
  show("ready");
});

function setProgress(stage: Stage, fraction: number) {
  const step = STEP_OF[stage];
  for (const li of $("steps").querySelectorAll<HTMLLIElement>("li")) {
    const n = Number(li.dataset.step);
    li.className = n < step ? "done" : n === step ? "active" : "";
    li.lastElementChild!.textContent = n < step ? "done" : "";
  }
  const pct = Math.floor(fraction * 100);
  $("bar").querySelector("i")!.style.width = `${fraction * 100}%`;
  $("bar").setAttribute("aria-valuenow", String(pct));
  $("pct").textContent = `${pct}%`;
  const elapsed = (performance.now() - startedAt) / 1000;
  if (fraction > 0.03 && elapsed > 3) {
    const left = (elapsed / fraction) * (1 - fraction);
    $("eta").textContent = left < 60 ? "Less than a minute left" : `About ${Math.round(left / 60)} minute${Math.round(left / 60) === 1 ? "" : "s"} left`;
  }
}

// ---------------------------------------------------------------- done

const SCALE_MIN = -36, SCALE_MAX = -8;
const pos = (v: number) => `${((Math.min(SCALE_MAX, Math.max(SCALE_MIN, v)) - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * 100}%`;
{
  const target = views.done.querySelector<HTMLElement>(".scale .target")!;
  target.style.left = pos(-17);
  target.style.width = `${(2 / (SCALE_MAX - SCALE_MIN)) * 100}%`;
  $("ticks").innerHTML = [-34, -28, -22, -16, -10].map((t) => `<span style="left:${pos(t)}">${minus(String(t))}</span>`).join("");
}

function outputName(): string {
  const base = file!.name.replace(/\.[^.]+$/, "");
  return /\.mp3$/i.test(file!.name) ? `${base}-prepared.mp3` : `${base}.mp3`;
}

function onDone(r: Result, blob: Blob) {
  keepAwake(false);
  if (outputUrl) URL.revokeObjectURL(outputUrl);
  outputUrl = URL.createObjectURL(blob);
  const save = $<HTMLAnchorElement>("save");
  save.href = outputUrl;
  save.download = outputName();
  $("done-name").textContent = outputName();
  const ok = r.loudnessOk && r.peakOk;
  $("done-pill").textContent = ok ? "Ready to publish" : "Missed target";
  $("done-pill").classList.toggle("miss", !ok);
  const misses: string[] = [];
  if (!r.loudnessOk) misses.push(`its loudness is ${fmtDb(r.outputLufs)} LUFS instead of ${fmtDb(r.targetLufs, 0)}`);
  if (!r.peakOk) misses.push(`its true peak is ${fmtDb(r.outputTruePeak)} dBTP, above the ${fmtDb(MAX_TRUE_PEAK + r.targetLufs - TARGET_LUFS, 0)} dBTP limit`);
  $("done-miss").textContent = ok ? "" : `The finished MP3 missed the target: ${misses.join(", and ")}. This is a bug in the app; the file is still saved as processed.`;
  $("done-miss").hidden = ok;
  $("done-meta").textContent = `${fmtDuration(r.duration)} · ${r.channels === 1 ? "mono sound" : "stereo"} · ${r.bitrate} kbps · ${fmtBytes(r.bytes)}`;
  $("mark-before").style.left = pos(r.inputLufs);
  views.done.querySelector<HTMLElement>(".scale .target")!.style.left = pos(r.targetLufs - 1);
  $("mark-after").style.left = pos(r.outputLufs);
  $("lufs-before").textContent = `${fmtDb(r.inputLufs)} LUFS`;
  $("lufs-after").textContent = `${fmtDb(r.outputLufs)} LUFS`;
  $("stat-lufs").textContent = fmtDb(r.outputLufs);
  $("stat-tp").textContent = fmtDb(r.outputTruePeak);
  if (r.noiseReductionDb !== null) {
    $("stat-3").textContent = `${fmtDb(r.noiseReductionDb, 0)} dB`;
    $("stat-3-label").textContent = "background noise";
  } else {
    $("stat-3").textContent = String(r.spikesTamed);
    $("stat-3-label").textContent = r.spikesTamed === 1 ? "loud noise tamed" : "loud noises tamed";
  }
  show("done");
}

$("again").addEventListener("click", () => {
  file = null;
  info = null;
  show("empty");
});

function failed(msg: string) {
  keepAwake(false);
  if (info) {
    $("ready-error").textContent = msg;
    $("ready-error").hidden = false;
    show("ready");
  } else {
    emptyError(msg);
    show("empty");
  }
}

function onWorkerMessage(m: Response) {
  switch (m.type) {
    case "info":
      onInfo(m.info);
      break;
    case "progress":
      if (!views.proc.hidden) setProgress(m.progress.stage, m.progress.fraction);
      break;
    case "done":
      onDone(m.result, m.output);
      break;
    case "cancelled":
      show("ready");
      break;
    case "error":
      failed(m.userFacing ? m.message : `Something went wrong while processing: ${m.message}`);
      break;
  }
}

// ---------------------------------------------------------------- offline

if ("serviceWorker" in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register("sw.js").catch((e) => console.warn("Service worker not registered", e));
}
