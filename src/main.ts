import "./style.css";
import type { AudioInfo } from "./io/source.ts";
import { MAX_TRUE_PEAK, TARGET_LUFS, workingBytes, type Options, type Result, type Stage } from "./pipeline/process.ts";
import { Preview } from "./preview.ts";
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
  // The preview only runs while the settings screen is showing.
  if (v === "ready") showCurrent();
  else preview.stop();
}

/** One chosen file and what happened to it. */
interface Item {
  file: File;
  info: AudioInfo | null;
  /** Why it can't be processed (unreadable or unsupported). */
  error: string | null;
  result: Result | null;
  /** Object URL of the finished MP3. */
  url: string | null;
  /** Why processing failed. */
  failed: string | null;
}

let items: Item[] = [];
let current = 0;
let channels: 1 | 2 = 1;
let probeBatch = 0;

let worker = newWorker();
function newWorker(): Worker {
  const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  w.onmessage = (e: MessageEvent<Response>) => onWorkerMessage(e.data);
  w.onerror = (e) => {
    console.error(e);
    if (!views.proc.hidden) fileFailed("Something went wrong while processing this file.");
  };
  return w;
}
const send = (r: Request) => worker.postMessage(r);

const usable = () => items.filter((it) => it.info && !it.error);

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
const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

// ---------------------------------------------------------------- choose

const fileInput = $<HTMLInputElement>("file");
const drop = $("drop");
fileInput.addEventListener("change", () => {
  if (fileInput.files?.length) choose([...fileInput.files]);
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
    const files = [...((e as DragEvent).dataTransfer?.files ?? [])];
    if (files.length && views.proc.hidden) choose(files);
  });
}

function emptyError(msg: string | null) {
  const el = $("empty-error");
  el.textContent = msg ?? "";
  el.hidden = !msg;
}

function choose(files: File[]) {
  if (!views.proc.hidden) return;
  emptyError(null);
  releaseOutputs();
  probeBatch++;
  items = files.map((file) => ({ file, info: null, error: null, result: null, url: null, failed: null }));
  current = 0;
  items.forEach((it, i) => send({ type: "probe", id: probeBatch * 10000 + i, file: it.file }));
}

function onProbe(id: number, info: AudioInfo | null, error: string | null) {
  if (Math.floor(id / 10000) !== probeBatch) return; // an older selection
  const it = items[id % 10000];
  if (!it) return;
  it.info = info;
  it.error = error;
  if (items.some((x) => !x.info && !x.error)) return; // still probing
  const ok = usable();
  if (!ok.length) {
    emptyError(
      items.length === 1
        ? items[0].error
        : `None of these files can be processed. ${items.map((x) => `${x.file.name}: ${x.error}`).join(" ")}`,
    );
    show("empty");
    return;
  }
  current = items.findIndex((x) => x.info && !x.error);
  const anyStereo = ok.some((x) => x.info!.channels >= 2);
  $<HTMLButtonElement>("ch-stereo").disabled = !anyStereo;
  setChannels(1);
  $("ready-error").hidden = true;
  show("ready");
}

// ---------------------------------------------------------------- settings screen

const preview = new Preview({
  root: $("preview"),
  play: $<HTMLButtonElement>("pv-play"),
  original: $<HTMLButtonElement>("pv-original"),
  processed: $<HTMLButtonElement>("pv-processed"),
  seek: $<HTMLInputElement>("pv-seek"),
  time: $("pv-time"),
  status: $("pv-status"),
});

/** Shows the current file's details and points the preview at it. */
function showCurrent() {
  const it = items[current];
  if (!it) return;
  const multi = items.length > 1;
  $("pager").hidden = !multi;
  $("batch-note").hidden = usable().length < 2;
  $("pager-pos").textContent = `File ${current + 1} of ${items.length}`;
  $<HTMLButtonElement>("prev").disabled = current === 0;
  $<HTMLButtonElement>("next").disabled = current === items.length - 1;
  $("ready-name").textContent = it.file.name;
  const i = it.info;
  if (i && !it.error) {
    const ch = i.channels === 1 ? "mono" : i.channels === 2 ? "stereo" : `${i.channels} channels`;
    $("ready-meta").textContent = [
      (i.durationIsEstimate ? "about " : "") + fmtDuration(i.duration),
      fmtRate(i.sampleRate),
      `${i.detail} ${ch}`,
      fmtBytes(it.file.size),
    ].join(" · ");
  } else {
    $("ready-meta").textContent = fmtBytes(it.file.size);
  }
  $("file-error").textContent = it.error ? `${it.error} This file will be skipped.` : "";
  $("file-error").hidden = !it.error;
  $("preview").hidden = !!it.error;
  if (i && !it.error) preview.setFile(it.file, i.duration, optionsFor(it));
  else preview.stop();
}
$("prev").addEventListener("click", () => {
  if (current > 0) {
    current--;
    showCurrent();
  }
});
$("next").addEventListener("click", () => {
  if (current < items.length - 1) {
    current++;
    showCurrent();
  }
});

function settingsChanged() {
  const it = items[current];
  if (it?.info && !it.error && !views.ready.hidden) preview.setOptions(optionsFor(it));
}
for (const id of ["c-rumble", "c-noise", "c-spikes", "c-sibilance", "c-level"]) {
  $(id).addEventListener("change", settingsChanged);
}

function setChannels(c: 1 | 2) {
  channels = c;
  $("ch-mono").setAttribute("aria-pressed", String(c === 1));
  $("ch-stereo").setAttribute("aria-pressed", String(c === 2));
  const ok = usable();
  const monoCount = ok.filter((x) => x.info!.channels === 1).length;
  $("ch-hint").textContent =
    monoCount === ok.length
      ? ok.length === 1
        ? "Your recording is mono, so both speakers get the same sound."
        : "Your recordings are mono, so both speakers get the same sound."
      : c === 1
        ? "Best for voice: both speakers get the same sound, so nobody is stuck in one ear."
        : `Keeps the left–right mix. Use if you have music or a deliberate stereo mix. The file is 50% larger.${monoCount ? " Mono recordings stay mono sound." : ""}`;
  const kbps = c === 1 ? 128 : 192;
  $("spec-line").textContent = `MP3 · joint stereo${c === 1 ? " (mono sound)" : ""} · ${kbps} kbps CBR · 44.1 kHz`;
  const bytes = ok.reduce((s, x) => s + workingBytes(x.info!.duration, Math.min(c, x.info!.channels)), 0);
  $("spec-size").textContent = ok.length ? fmtBytes(bytes).replace(" ", "\u00a0") : "";
  $("spec-total").textContent = ok.length > 1 ? " in total" : "";
  settingsChanged();
}
$("ch-mono").addEventListener("click", () => setChannels(1));
$("ch-stereo").addEventListener("click", () => {
  if (!$<HTMLButtonElement>("ch-stereo").disabled) setChannels(2);
});

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
  const nb = (s: string) => s.replace(/ /g, " ");
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
  settingsChanged();
}
trimInput.addEventListener("input", updateTrim);
updateTrim();

$("change").addEventListener("click", () => {
  show("empty");
  fileInput.click();
});

/** The shared settings, with mono sound for mono recordings. */
function optionsFor(it: Item): Options {
  const on = (id: string) => $<HTMLInputElement>(id).checked;
  return {
    channels: it.info && it.info.channels === 1 ? 1 : channels,
    rumble: on("c-rumble"),
    noise: on("c-noise"),
    spikes: on("c-spikes"),
    sibilance: on("c-sibilance"),
    level: on("c-level"),
    outputTrimDb: trimDb(),
  };
}

// ---------------------------------------------------------------- process

const STEP_OF: Record<Stage, number> = { process: 0, verify: 1 };
let startedAt = 0;
let queue: number[] = [];
let queuePos = 0;
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
  if (!navigator.storage?.estimate) return null;
  try {
    const { quota, usage } = await navigator.storage.estimate();
    if (!quota) return null;
    const need = usable().reduce((s, x) => s + workingBytes(x.info!.duration, optionsFor(x).channels), 0) * 1.05;
    const free = quota - (usage ?? 0);
    if (free < need) return `There isn’t enough free storage on this device. The MP3s need about ${fmtBytes(need)}; about ${fmtBytes(free)} is available.`;
  } catch {
    /* unknown: try anyway */
  }
  return null;
}

function releaseOutputs() {
  for (const it of items) if (it.url) URL.revokeObjectURL(it.url);
}

$("go").addEventListener("click", async () => {
  if (!usable().length) return;
  const problem = await enoughSpace();
  if (problem) {
    $("ready-error").textContent = problem;
    $("ready-error").hidden = false;
    return;
  }
  releaseOutputs();
  for (const it of items) {
    it.result = null;
    it.url = null;
    it.failed = null;
  }
  queue = items.map((_, i) => i).filter((i) => items[i].info && !items[i].error);
  queuePos = 0;
  startedAt = performance.now();
  show("proc");
  keepAwake(true);
  startNext();
});

/** Total seconds of audio in the queue, and how much is already done. */
function queueSeconds(upTo = queue.length): number {
  return queue.slice(0, upTo).reduce((s, i) => s + items[i].info!.duration, 0);
}

function startNext() {
  if (queuePos >= queue.length) {
    keepAwake(false);
    renderResults();
    show("done");
    return;
  }
  const i = queue[queuePos];
  const it = items[i];
  $("proc-name").textContent = queue.length > 1 ? `File ${queuePos + 1} of ${queue.length} · ${it.file.name}` : it.file.name;
  $("proc-duration").textContent = fmtDuration(it.info!.duration);
  setProgress("process", 0);
  if (queuePos === 0) $("eta").textContent = "Starting…";
  send({ type: "process", file: it.file, options: optionsFor(it), outputName: `out-${i}.mp3`, fresh: queuePos === 0 });
}

function fileDone(r: Result, blob: Blob) {
  const it = items[queue[queuePos]];
  it.result = r;
  it.url = URL.createObjectURL(blob);
  queuePos++;
  startNext();
}

function fileFailed(msg: string) {
  const it = items[queue[queuePos]];
  if (!it) return;
  it.failed = msg;
  // Start a fresh worker in case the old one is in a bad state.
  worker.terminate();
  worker = newWorker();
  queuePos++;
  startNext();
}

$("cancel").addEventListener("click", () => {
  // Terminating is immediate and releases the output file’s lock.
  worker.terminate();
  worker = newWorker();
  keepAwake(false);
  releaseOutputs();
  for (const it of items) it.url = null;
  show("ready");
});

function setProgress(stage: Stage, fraction: number) {
  const step = STEP_OF[stage];
  for (const li of $("steps").querySelectorAll<HTMLLIElement>("li")) {
    const n = Number(li.dataset.step);
    li.className = n < step ? "done" : n === step ? "active" : "";
    li.lastElementChild!.textContent = n < step ? "done" : "";
  }
  // Overall progress across the batch, weighted by duration.
  const total = Math.max(1e-9, queueSeconds());
  const before = queueSeconds(queuePos);
  const here = queue[queuePos] !== undefined ? items[queue[queuePos]].info!.duration : 0;
  const overall = Math.min(1, (before + here * fraction) / total);
  const pct = Math.floor(overall * 100);
  $("bar").querySelector("i")!.style.width = `${overall * 100}%`;
  $("bar").setAttribute("aria-valuenow", String(pct));
  $("pct").textContent = `${pct}%`;
  const elapsed = (performance.now() - startedAt) / 1000;
  if (overall > 0.03 && elapsed > 3) {
    const left = (elapsed / overall) * (1 - overall);
    $("eta").textContent = left < 60 ? "Less than a minute left" : `About ${Math.round(left / 60)} minute${Math.round(left / 60) === 1 ? "" : "s"} left`;
  }
}

// ---------------------------------------------------------------- results

const SCALE_MIN = -36, SCALE_MAX = -8;
const pos = (v: number) => `${((Math.min(SCALE_MAX, Math.max(SCALE_MIN, v)) - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * 100}%`;

/** Download names, made unique within the batch. */
function outputNames(): Map<Item, string> {
  const names = new Map<Item, string>();
  const used = new Set<string>();
  for (const it of items) {
    const base = it.file.name.replace(/\.[^.]+$/, "");
    let name = /\.mp3$/i.test(it.file.name) ? `${base}-prepared.mp3` : `${base}.mp3`;
    for (let k = 2; used.has(name.toLowerCase()); k++) name = name.replace(/(-\d+)?\.mp3$/, `-${k}.mp3`);
    used.add(name.toLowerCase());
    names.set(it, name);
  }
  return names;
}

function renderResults() {
  const list = $("results");
  list.replaceChildren();
  const tpl = $<HTMLTemplateElement>("result-tpl");
  const names = outputNames();
  const shown = items.filter((it) => it.result || it.failed || it.error);
  let ready = 0, missed = 0, failed = 0;
  for (const it of shown) {
    const card = tpl.content.firstElementChild!.cloneNode(true) as HTMLElement;
    const f = (k: string) => card.querySelector<HTMLElement>(`[data-f="${k}"]`)!;
    const r = it.result;
    if (!r) {
      failed++;
      f("name").textContent = it.file.name;
      f("meta").textContent = "Not processed";
      f("pill").textContent = "Failed";
      f("pill").classList.add("miss");
      f("miss").textContent = it.error ?? it.failed ?? "";
      f("miss").hidden = false;
      for (const k of ["meter", "stats", "save"]) f(k).hidden = true;
      list.append(card);
      continue;
    }
    const ok = r.loudnessOk && r.peakOk;
    if (ok) ready++;
    else missed++;
    f("name").textContent = names.get(it)!;
    f("meta").textContent = `${fmtDuration(r.duration)} · ${r.channels === 1 ? "mono sound" : "stereo"} · ${r.bitrate} kbps · ${fmtBytes(r.bytes)}`;
    f("pill").textContent = ok ? "Ready to publish" : "Missed target";
    f("pill").classList.toggle("miss", !ok);
    const misses: string[] = [];
    if (!r.loudnessOk) misses.push(`its loudness is ${fmtDb(r.outputLufs)} LUFS instead of ${fmtDb(r.targetLufs, 0)}`);
    if (!r.peakOk) misses.push(`its true peak is ${fmtDb(r.outputTruePeak)} dBTP, above the ${fmtDb(MAX_TRUE_PEAK + r.targetLufs - TARGET_LUFS, 0)} dBTP limit`);
    f("miss").textContent = ok ? "" : `The finished MP3 missed the target: ${misses.join(", and ")}. This is a bug in the app; the file is still saved as processed.`;
    f("miss").hidden = ok;
    f("target").style.left = pos(r.targetLufs - 1);
    f("target").style.width = `${(2 / (SCALE_MAX - SCALE_MIN)) * 100}%`;
    f("before").style.left = pos(r.inputLufs);
    f("after").style.left = pos(r.outputLufs);
    f("ticks").innerHTML = [-34, -28, -22, -16, -10].map((t) => `<span style="left:${pos(t)}">${minus(String(t))}</span>`).join("");
    f("lufs-before").textContent = `${fmtDb(r.inputLufs)} LUFS`;
    f("lufs-after").textContent = `${fmtDb(r.outputLufs)} LUFS`;
    f("lufs").textContent = fmtDb(r.outputLufs);
    f("tp").textContent = fmtDb(r.outputTruePeak);
    if (r.noiseReductionDb !== null) {
      f("s3").textContent = `${fmtDb(r.noiseReductionDb, 0)} dB`;
      f("s3-label").textContent = "background noise";
    } else {
      f("s3").textContent = String(r.spikesTamed);
      f("s3-label").textContent = r.spikesTamed === 1 ? "loud noise tamed" : "loud noises tamed";
    }
    const save = f("save") as HTMLAnchorElement;
    save.href = it.url!;
    save.download = names.get(it)!;
    save.textContent = shown.length > 1 ? `Save ${names.get(it)}` : "Save MP3";
    list.append(card);
  }
  const done = ready + missed;
  $("done-title").textContent =
    shown.length === 1 ? (done ? "Your MP3 is ready" : "This file couldn’t be processed") : `${plural(done, "MP3", "MP3s")} ready`;
  const parts: string[] = [];
  if (missed) parts.push(`${missed} missed the loudness target`);
  if (failed) parts.push(`${plural(failed, "file", "files")} couldn’t be processed`);
  $("done-sub").textContent = parts.length ? `${parts.join("; ")}.` : shown.length > 1 ? "Save each one below." : "";
  $("done-sub").hidden = !$("done-sub").textContent;
}

$("again").addEventListener("click", () => {
  releaseOutputs();
  items = [];
  show("empty");
});

function onWorkerMessage(m: Response) {
  switch (m.type) {
    case "info":
      onProbe(m.id, m.info, null);
      break;
    case "probeError":
      onProbe(m.id, null, m.message);
      break;
    case "progress":
      if (!views.proc.hidden) setProgress(m.progress.stage, m.progress.fraction);
      break;
    case "done":
      fileDone(m.result, m.output);
      break;
    case "error":
      if (!views.proc.hidden) fileFailed(m.userFacing ? m.message : `Something went wrong while processing: ${m.message}`);
      break;
  }
}

// ---------------------------------------------------------------- offline

if ("serviceWorker" in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register("sw.js").catch((e) => console.warn("Service worker not registered", e));
}
