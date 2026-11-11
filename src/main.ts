import "./style.css";
import type { AudioInfo } from "./io/source.ts";
import { MAX_TRUE_PEAK, TARGET_LUFS, workingBytes, type Options, type Result } from "./pipeline/process.ts";
import type { Request, Response } from "./worker.ts";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------------------------------------------------------------- formatting

function clock(s: number): string {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
}
function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(b >= 1e8 ? 0 : 1)} MB`;
  return `${Math.max(1, Math.round(b / 1e3))} KB`;
}
const minus = (s: string | number) => String(s).replace("-", "−");
const fmtDb = (v: number, digits = 1) => (Number.isFinite(v) ? minus(v.toFixed(digits)) : "—");
const fmtRate = (r: number) => `${(r / 1000).toFixed(r % 1000 ? 1 : 0)} kHz`;
const nb = (s: string) => s.replace(/ /g, " ");
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

// ---------------------------------------------------------------- state

type Status = "probing" | "queued" | "working" | "ready" | "failed";

interface Item {
  id: number;
  file: File;
  info: AudioInfo | null;
  /** Set when the file can't be read at all. */
  unreadable: string | null;
  status: Status;
  progress: number;
  /** Seconds of MP3 written so far. */
  outputSeconds: number;
  result: Result | null;
  /** The finished MP3 and its object URL. */
  blob: Blob | null;
  url: string | null;
  /** Why processing failed. */
  failure: string | null;
  /** Object URL of the original, made when first played. */
  originalUrl: string | null;
  /** Counts runs, so each run writes a file of its own. */
  run: number;
  row: HTMLLIElement;
}

let items: Item[] = [];
let nextId = 1;
let paused = false;
let priority: number | null = null; // a file someone wants to hear next
let jobStarted = 0; // when the current file started (for the time estimate)
let firstJob = true;

// ---------------------------------------------------------------- settings

const TOGGLES = ["t-rumble", "t-noise", "t-spikes", "t-ess", "t-level"] as const;
interface Controls {
  channels: 1 | 2;
  level: number;
  toggles: Record<(typeof TOGGLES)[number], boolean>;
}
let uiChannels: 1 | 2 = 1;
const LEVEL_KEY = "ppa.outputTrimDb";

function readControls(): Controls {
  return {
    channels: uiChannels,
    level: Math.min(0, Math.round(Number($<HTMLInputElement>("level").value) / 3) * 3),
    toggles: Object.fromEntries(TOGGLES.map((id) => [id, $<HTMLInputElement>(id).checked])) as Controls["toggles"],
  };
}

function showControls(c: Controls) {
  uiChannels = c.channels;
  $("ch-mono").setAttribute("aria-pressed", String(c.channels === 1));
  $("ch-stereo").setAttribute("aria-pressed", String(c.channels === 2));
  $("ch-hint").textContent =
    c.channels === 1 ? "Both speakers get the same sound. Best for voice." : "Keeps the left–right mix. Mono recordings stay mono sound.";
  $<HTMLInputElement>("level").value = String(c.level);
  $("level-val").textContent = `${minus(c.level)} dB`;
  $("level-hint").textContent =
    c.level === 0 ? "Apple’s recommended loudness. Turn it down if your episodes sound louder than other shows." : `${minus(-c.level)} dB quieter than Apple’s recommendation, applied after everything else.`;
  for (const id of TOGGLES) $<HTMLInputElement>(id).checked = c.toggles[id];
  $("spec-1").textContent = `MP3 · joint stereo${c.channels === 1 ? " (mono sound)" : ""} · ${c.channels === 1 ? 128 : 192} kbps · 44.1 kHz`;
  $("spec-2").textContent = `${nb(`${minus(TARGET_LUFS + c.level)} LUFS`)} integrated · peaks ≤ ${nb(`${minus(MAX_TRUE_PEAK + c.level)} dBTP`)}`;
  $("spec-3").textContent = c.level === 0 ? "Meets Apple Podcasts guidelines." : `Quieter than Apple’s ${minus(TARGET_LUFS)} LUFS recommendation.`;
}

try {
  const saved = Number(localStorage.getItem(LEVEL_KEY));
  if (localStorage.getItem(LEVEL_KEY) !== null && Number.isFinite(saved)) $<HTMLInputElement>("level").value = String(saved);
} catch {
  /* storage unavailable */
}
let committed = readControls();
showControls(committed);

const same = (a: Controls, b: Controls) => JSON.stringify(a) === JSON.stringify(b);
const redoable = () => items.filter((it) => !it.unreadable && it.status !== "probing");

function controlsChanged() {
  showControls(readControls());
  // With nothing to redo, changes apply straight away.
  if (!redoable().length) commitSettings();
  renderPending();
}

function renderPending() {
  const pending = !same(readControls(), committed);
  const n = redoable().length;
  $("pending").hidden = !pending;
  $("settings-panel").classList.toggle("is-pending", pending);
  $("apply").textContent = `Apply and reprocess ${plural(n, "file")}`;
  $("set-hint").textContent = pending ? "Not applied yet" : "Apply to every file";
}

function commitSettings() {
  committed = readControls();
  try {
    localStorage.setItem(LEVEL_KEY, String(committed.level));
  } catch {
    /* storage unavailable */
  }
}

$("ch-mono").addEventListener("click", () => { uiChannels = 1; controlsChanged(); });
$("ch-stereo").addEventListener("click", () => { uiChannels = 2; controlsChanged(); });
$("level").addEventListener("input", controlsChanged);
for (const id of TOGGLES) $(id).addEventListener("change", controlsChanged);
$("undo").addEventListener("click", () => { showControls(committed); renderPending(); });
$("apply").addEventListener("click", () => {
  commitSettings();
  restartAll();
  renderPending();
});

/** The committed settings for one file (mono recordings always get mono sound). */
function optionsFor(it: Item): Options {
  const t = committed.toggles;
  return {
    channels: it.info && it.info.channels === 1 ? 1 : committed.channels,
    rumble: t["t-rumble"],
    noise: t["t-noise"],
    spikes: t["t-spikes"],
    sibilance: t["t-ess"],
    level: t["t-level"],
    outputTrimDb: committed.level,
  };
}

// ---------------------------------------------------------------- workers

const probeWorker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
probeWorker.onmessage = (e: MessageEvent<Response>) => onMessage(e.data);

let worker: Worker | null = null;
function procWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<Response>) => onMessage(e.data);
    worker.onerror = (e) => {
      console.error(e);
      const it = items.find((x) => x.status === "working");
      if (it) fail(it, "Something went wrong while processing this file.");
    };
    if (paused) worker.postMessage({ type: "pause" } satisfies Request);
  }
  return worker;
}
/** Stops whatever is being processed (terminating is immediate and releases its file). */
function stopWorker() {
  worker?.terminate();
  worker = null;
  for (const r of snapshotWaiters.values()) r(null);
  snapshotWaiters.clear();
}

const snapshotWaiters = new Map<number, (b: Blob | null) => void>();
function snapshot(it: Item): Promise<Blob | null> {
  if (it.status !== "working" || !worker) return Promise.resolve(null);
  return new Promise((resolve) => {
    snapshotWaiters.get(it.id)?.(null);
    snapshotWaiters.set(it.id, resolve);
    worker!.postMessage({ type: "snapshot", id: it.id } satisfies Request);
  });
}

function onMessage(m: Response) {
  if (m.type === "snapshot") {
    snapshotWaiters.get(m.id)?.(m.blob);
    snapshotWaiters.delete(m.id);
    return;
  }
  const it = items.find((x) => x.id === m.id);
  if (!it) return;
  switch (m.type) {
    case "info":
      it.info = m.info;
      it.status = "queued";
      break;
    case "probeError":
      it.unreadable = m.message;
      it.status = "failed";
      break;
    case "progress":
      if (it.status !== "working") return;
      it.progress = m.progress.fraction;
      it.outputSeconds = m.progress.outputSeconds;
      renderRow(it);
      if (player?.item === it) player.update();
      renderStatus();
      return;
    case "done":
      if (it.status !== "working") return;
      it.status = "ready";
      it.progress = 1;
      it.result = m.result;
      it.blob = m.output;
      it.url = URL.createObjectURL(m.output);
      it.outputSeconds = m.result.duration;
      break;
    case "error":
      if (it.status !== "working") return;
      fail(it, m.message);
      return;
  }
  renderAll();
  pump();
}

function fail(it: Item, message: string) {
  it.status = "failed";
  it.failure = message;
  // Start the next file in a fresh worker in case this one is in a bad state.
  stopWorker();
  renderAll();
  pump();
}

// ---------------------------------------------------------------- queue

async function enoughSpaceFor(it: Item): Promise<string | null> {
  try {
    const est = await navigator.storage?.estimate?.();
    if (!est?.quota) return null;
    const need = workingBytes(it.info!.duration, optionsFor(it).channels) * 1.05;
    const free = est.quota - (est.usage ?? 0);
    if (free < need) return `Not enough free storage on this device: the MP3 needs about ${fmtBytes(need)}, and ${fmtBytes(free)} is free.`;
  } catch {
    /* unknown: try anyway */
  }
  return null;
}

let pumping = false;
async function pump() {
  if (pumping || paused || items.some((x) => x.status === "working")) return;
  // List order, unless someone asked to hear a file next. Wait for files
  // still being read so a later file doesn't jump ahead of them.
  let it = items.find((x) => x.id === priority && x.status === "queued");
  if (!it) {
    for (const x of items) {
      if (x.status === "probing") return;
      if (x.status === "queued") {
        it = x;
        break;
      }
    }
  }
  if (!it) {
    renderStatus();
    keepAwake(false);
    return;
  }
  pumping = true;
  const problem = await enoughSpaceFor(it);
  pumping = false;
  if (it.status !== "queued") return pump(); // removed or changed meanwhile
  if (problem) {
    it.status = "failed";
    it.failure = problem;
    renderAll();
    return pump();
  }
  if (priority === it.id) priority = null;
  it.status = "working";
  it.progress = 0;
  it.outputSeconds = 0;
  jobStarted = performance.now();
  procWorker().postMessage({
    type: "process",
    id: it.id,
    file: it.file,
    options: optionsFor(it),
    // A fresh name per run: a stopped run's file can stay locked for a moment.
    outputName: `out-${it.id}-${++it.run}.mp3`,
    previous: it.run > 1 ? `out-${it.id}-${it.run - 1}.mp3` : null,
    fresh: firstJob,
  } satisfies Request);
  firstJob = false;
  keepAwake(true);
  renderAll();
}

function clearOutput(it: Item) {
  if (it.url) URL.revokeObjectURL(it.url);
  it.url = null;
  it.blob = null;
  it.result = null;
  it.progress = 0;
  it.outputSeconds = 0;
  it.failure = null;
}

/** New settings: every readable file starts again. */
function restartAll() {
  stopWorker();
  player?.outputChanged();
  for (const it of items) {
    if (it.unreadable || it.status === "probing") continue;
    clearOutput(it);
    it.status = "queued";
  }
  renderAll();
  pump();
}

function remove(it: Item) {
  if (player?.item === it) closePlayer();
  if (it.status === "working") stopWorker();
  clearOutput(it);
  if (it.originalUrl) URL.revokeObjectURL(it.originalUrl);
  it.row.remove();
  items = items.filter((x) => x !== it);
  renderAll();
  renderPending();
  pump();
}

$("pause").addEventListener("click", () => {
  paused = !paused;
  worker?.postMessage({ type: paused ? "pause" : "resume" } satisfies Request);
  if (!paused) {
    jobStarted = performance.now() - estimateElapsed();
    pump();
  }
  keepAwake(!paused && items.some((x) => x.status === "working"));
  renderAll();
});
function estimateElapsed(): number {
  // Keep the time estimate sensible after a pause.
  const it = items.find((x) => x.status === "working");
  return it ? (it.progress * it.info!.duration * 1000) / 4.4 : 0;
}

// ---------------------------------------------------------------- adding files

function addFiles(files: FileList | File[]) {
  for (const file of files) {
    const row = $<HTMLTemplateElement>("row-tpl").content.firstElementChild!.cloneNode(true) as HTMLLIElement;
    const it: Item = {
      id: nextId++, file, info: null, unreadable: null, status: "probing", progress: 0, outputSeconds: 0,
      result: null, blob: null, url: null, failure: null, originalUrl: null, run: 0, row,
    };
    const f = (k: string) => row.querySelector<HTMLElement>(`[data-f="${k}"]`)!;
    f("listen").addEventListener("click", () => (player?.item === it ? closePlayer() : openPlayer(it)));
    f("remove").addEventListener("click", () => remove(it));
    f("remove").setAttribute("aria-label", `Remove ${file.name}`);
    items.push(it);
    probeWorker.postMessage({ type: "probe", id: it.id, file } satisfies Request);
  }
  renderAll();
  renderPending();
}

const fileInput = $<HTMLInputElement>("file-input");
fileInput.addEventListener("change", () => {
  if (fileInput.files?.length) addFiles([...fileInput.files]);
  fileInput.value = "";
});
const addZone = $("add");
document.addEventListener("dragover", (e) => {
  e.preventDefault();
  addZone.classList.add("over");
});
document.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) addZone.classList.remove("over");
});
document.addEventListener("drop", (e) => {
  e.preventDefault();
  addZone.classList.remove("over");
  const files = [...(e.dataTransfer?.files ?? [])];
  if (files.length) addFiles(files);
});

// ---------------------------------------------------------------- rendering

/** Download names, made unique. */
function outputName(it: Item): string {
  const used = new Set<string>();
  for (const x of items) {
    const base = x.file.name.replace(/\.[^.]+$/, "");
    let name = /\.mp3$/i.test(x.file.name) ? `${base}-prepared.mp3` : `${base}.mp3`;
    for (let k = 2; used.has(name.toLowerCase()); k++) name = `${base}-${k}.mp3`;
    used.add(name.toLowerCase());
    if (x === it) return name;
  }
  return "output.mp3";
}

function chip(it: Item): [string, string] {
  switch (it.status) {
    case "probing":
      return ["queued", "Reading…"];
    case "queued":
      return ["queued", priority === it.id ? "Next" : "Waiting"];
    case "working":
      return ["working", paused ? "Paused" : `${Math.floor(it.progress * 100)}%`];
    case "ready":
      return it.result && !(it.result.loudnessOk && it.result.peakOk) ? ["miss", "Missed target"] : ["ready", "Ready"];
    case "failed":
      return ["failed", it.unreadable ? "Can’t read" : "Failed"];
  }
}

function renderRow(it: Item) {
  const f = (k: string) => it.row.querySelector<HTMLElement>(`[data-f="${k}"]`)!;
  const [cls, text] = chip(it);
  f("chip").className = `chip ${cls}`;
  f("chip").textContent = text;
  f("name").textContent = it.file.name;
  const i = it.info;
  f("meta").textContent = i
    ? [
        (i.durationIsEstimate ? "about " : "") + clock(i.duration),
        fmtRate(i.sampleRate),
        `${i.detail} ${i.channels === 1 ? "mono" : i.channels === 2 ? "stereo" : `${i.channels} channels`}`,
        fmtBytes(it.file.size),
      ].join(" · ")
    : fmtBytes(it.file.size);
  const res = f("result");
  res.className = "file-result";
  res.textContent = "";
  if (it.status === "failed") {
    res.classList.add("miss");
    res.textContent = it.unreadable ?? it.failure ?? "";
  } else if (it.status === "ready" && it.result) {
    const r = it.result;
    const third = r.noiseReductionDb !== null ? `${fmtDb(r.noiseReductionDb, 0)} dB noise` : plural(r.spikesTamed, "loud noise tamed", "loud noises tamed");
    const line = `${fmtDb(r.outputLufs)} LUFS · ${fmtDb(r.outputTruePeak)} dBTP · ${third} · ${fmtBytes(r.bytes)}`;
    if (r.loudnessOk && r.peakOk) {
      res.classList.add("num");
      res.textContent = line;
    } else {
      const misses: string[] = [];
      if (!r.loudnessOk) misses.push(`loudness is ${fmtDb(r.outputLufs)} LUFS instead of ${fmtDb(r.targetLufs, 0)}`);
      if (!r.peakOk) misses.push(`true peak is ${fmtDb(r.outputTruePeak)} dBTP, above ${fmtDb(MAX_TRUE_PEAK + r.targetLufs - TARGET_LUFS, 0)}`);
      res.classList.add("miss");
      res.textContent = `Missed the target: ${misses.join(", and ")}. This is a bug in the app; the file is still saved as processed.`;
    }
  }
  const listen = f("listen");
  listen.hidden = !!it.unreadable || it.status === "probing";
  listen.textContent = player?.item === it ? "Close player" : "Listen";
  listen.setAttribute("aria-expanded", String(player?.item === it));
  const save = f("save") as HTMLAnchorElement;
  save.hidden = it.status !== "ready" || !it.url;
  if (it.url) {
    save.href = it.url;
    save.download = outputName(it);
  }
  const bar = f("bar");
  bar.hidden = it.status !== "working";
  (bar.firstElementChild as HTMLElement).style.width = `${it.progress * 100}%`;
}

function renderAll() {
  const list = $("files");
  if (!items.length) {
    list.innerHTML = `<li class="empty"><span class="big">Add your recordings</span><span class="hint">Drop WAV or MP3 files below. Each one starts processing as soon as it’s added, using the settings on the left.</span></li>`;
  } else {
    list.querySelector(".empty")?.remove();
    for (const it of items) {
      if (it.row.parentElement !== list) list.append(it.row);
      renderRow(it);
    }
  }
  renderStatus();
}

function renderStatus() {
  const usable = items.filter((x) => x.info && !x.unreadable);
  const total = usable.reduce((s, x) => s + x.info!.duration, 0);
  const ready = items.filter((x) => x.status === "ready").length;
  $("files-summary").textContent = items.length ? `${plural(items.length, "file")} · ${clock(total)} of audio · ${ready} ready` : "";

  const todo = items.filter((x) => x.status === "queued" || x.status === "working" || x.status === "probing");
  const active = usable.filter((x) => x.status !== "failed");
  const totalDur = active.reduce((s, x) => s + x.info!.duration, 0);
  const doneDur = active.reduce((s, x) => s + x.info!.duration * (x.status === "ready" ? 1 : x.status === "working" ? x.progress : 0), 0);
  const frac = totalDur ? doneDur / totalDur : 0;
  $("overall-bar").style.width = `${frac * 100}%`;
  $("overall-bar-wrap").setAttribute("aria-valuenow", String(Math.floor(frac * 100)));
  $("overall-pct").textContent = todo.length ? `${Math.floor(frac * 100)}%` : "";
  // Speed from the current file; about 4× real time until there's data.
  const cur = items.find((x) => x.status === "working");
  const elapsed = (performance.now() - jobStarted) / 1000;
  const speed = cur && elapsed > 5 && cur.progress > 0.02 ? (cur.progress * cur.info!.duration) / elapsed : 4;
  const left = (totalDur - doneDur) / Math.max(0.5, speed);
  let text: string;
  if (!items.length) text = "Add files to start.";
  else if (!todo.length) text = ready ? (ready === items.length ? "All files are ready to save." : "Finished. Some files couldn’t be processed.") : "Nothing to process.";
  else if (paused) text = `Paused · ${plural(todo.length, "file")} to go`;
  else text = `Processing · ${plural(todo.length, "file")} to go · ${left < 60 ? "under a minute" : `about ${Math.round(left / 60)} min`} left`;
  $("overall-text").textContent = text;
  $("pause").hidden = !todo.length;
  $("pause").textContent = paused ? "Resume" : "Pause";
}

// ---------------------------------------------------------------- player

/**
 * Plays the original file or the MP3. While a file is processing, "Processed"
 * plays a copy of what has been written so far; each press of Play takes a
 * fresh copy, so the length grows.
 */
class Player {
  which: "original" | "processed" = "original";
  playing = false;
  pos = 0;
  private audio = new Audio();
  private snapUrl: string | null = null;
  private note = "";
  /** The note says to wait for output; it clears once there is some. */
  private waitingNote = false;
  private el: HTMLElement;

  constructor(readonly item: Item) {
    this.el = item.row.querySelector<HTMLElement>('[data-f="player"]')!;
    this.el.hidden = false;
    this.which = item.status === "ready" ? "processed" : "original";
    const q = (k: string) => this.el.querySelector<HTMLElement>(`[data-f="${k}"]`)!;
    q("play").onclick = () => (this.playing ? this.pause() : void this.play());
    q("original").onclick = () => this.select("original");
    q("processed").onclick = () => this.select("processed");
    (q("seek") as HTMLInputElement).oninput = (e) => this.seek(Number((e.target as HTMLInputElement).value));
    this.audio.preload = "auto";
    this.audio.ontimeupdate = () => {
      if (this.playing) {
        this.pos = this.audio.currentTime;
        this.update();
      }
    };
    this.audio.onended = () => {
      this.playing = false;
      this.pos = this.audio.currentTime;
      if (this.which === "processed" && this.item.status === "working") {
        this.note = "That’s everything written so far. Press Play to carry on from here.";
      } else {
        this.pos = 0;
      }
      this.update();
    };
    this.update();
  }

  private q(k: string) {
    return this.el.querySelector<HTMLElement>(`[data-f="${k}"]`)!;
  }

  private duration() {
    return this.item.info?.duration ?? 0;
  }

  /** Seconds of processed audio that can be played. */
  private available() {
    const it = this.item;
    return it.status === "ready" ? this.duration() : it.status === "working" ? it.outputSeconds : 0;
  }

  private async source(): Promise<string | null> {
    const it = this.item;
    if (this.which === "original") {
      it.originalUrl ??= URL.createObjectURL(it.file);
      return it.originalUrl;
    }
    if (it.status === "ready" && it.url) return it.url;
    if (it.status === "working") {
      const blob = await snapshot(it);
      if (!blob || blob.size < 8192) return null;
      if (this.snapUrl) URL.revokeObjectURL(this.snapUrl);
      this.snapUrl = URL.createObjectURL(blob);
      return this.snapUrl;
    }
    return null;
  }

  async play() {
    const it = this.item;
    this.note = "";
    this.waitingNote = false;
    if (this.which === "processed" && it.status === "queued") {
      // Hearing a file that hasn't started moves it to the front.
      priority = it.id;
      renderRow(it);
      this.note = "This file is processed next. Its output starts shortly after it begins.";
      this.update();
      return;
    }
    const url = await this.source();
    if (!url) {
      this.waitingNote = this.which === "processed" && it.status === "working";
      this.note = this.waitingNote
        ? "Nothing written yet: output starts once about a minute and a half of the recording has been read."
        : "This version can’t be played yet.";
      this.update();
      return;
    }
    if (this.audio.src !== url) {
      this.audio.src = url;
      await new Promise<void>((resolve) => {
        const done = () => resolve();
        this.audio.addEventListener("loadedmetadata", done, { once: true });
        this.audio.addEventListener("error", done, { once: true });
      });
    }
    const end = Number.isFinite(this.audio.duration) ? this.audio.duration : this.available();
    if (this.pos >= end - 0.5) {
      if (this.which === "processed" && it.status === "working") {
        this.note = "You’ve reached the end of what’s been written. Try again in a moment.";
        this.update();
        return;
      }
      this.pos = 0;
    }
    this.audio.currentTime = this.pos;
    try {
      await this.audio.play();
      this.playing = true;
    } catch {
      this.note = "Playback couldn’t start.";
    }
    this.update();
  }

  pause() {
    this.audio.pause();
    this.pos = this.audio.currentTime;
    this.playing = false;
    this.update();
  }

  select(which: "original" | "processed") {
    if (which === this.which) return;
    const resume = this.playing;
    if (this.playing) this.pause();
    this.which = which;
    this.audio.removeAttribute("src");
    this.audio.load();
    if (resume) void this.play();
    else this.update();
  }

  seek(to: number) {
    if (this.which === "processed") {
      // Can't go past what's written; stop a little short so there's something to hear.
      const limit = this.item.status === "working" ? Math.max(0, this.available() - 3) : this.available();
      this.pos = Math.min(to, limit);
    } else this.pos = to;
    const loaded = Number.isFinite(this.audio.duration) ? this.audio.duration : 0;
    if (this.playing && this.which === "processed" && this.item.status === "working" && this.pos > loaded - 0.5) {
      // Past the copy that's loaded: take a fresh one that reaches this far.
      this.audio.pause();
      this.playing = false;
      void this.play();
      return;
    }
    if (this.playing) {
      const max = Number.isFinite(this.audio.duration) ? this.audio.duration - 0.2 : this.pos;
      this.audio.currentTime = Math.min(this.pos, max);
    }
    this.note = "";
    this.update();
  }

  /** The processed output was thrown away (new settings). */
  outputChanged() {
    if (this.which === "processed") {
      this.pause();
      this.audio.removeAttribute("src");
      this.audio.load();
      this.pos = 0;
    }
  }

  update() {
    const it = this.item;
    const dur = this.duration();
    const play = this.q("play");
    play.textContent = this.playing ? "❚❚" : "▶";
    play.setAttribute("aria-label", this.playing ? "Pause" : "Play");
    this.q("original").setAttribute("aria-pressed", String(this.which === "original"));
    this.q("processed").setAttribute("aria-pressed", String(this.which === "processed"));
    const seek = this.q("seek") as HTMLInputElement;
    seek.max = String(dur);
    seek.value = String(this.pos);
    const avail = this.which === "processed" ? Math.min(1, this.available() / Math.max(1, dur)) : 1;
    seek.style.setProperty("--avail", `${avail * 100}%`);
    this.q("time").textContent = `${clock(this.pos)} / ${clock(dur)}`;
    if (this.waitingNote && (this.available() > 0 || it.status !== "working")) {
      this.note = "";
      this.waitingNote = false;
    }
    let status = this.note;
    if (!status) {
      if (this.which === "original") status = "The recording as it is.";
      else if (it.status === "ready") status = "The finished MP3.";
      else if (it.status === "working") status = this.available() > 0 ? `The MP3 as it’s being made: ${clock(this.available())} written so far.` : "Output starts once about a minute and a half has been read…";
      else if (it.status === "queued") status = priority === it.id ? "This file is processed next." : "Waiting to be processed. Press Play to process it next.";
      else status = "There’s no processed version.";
    }
    this.q("status").textContent = status;
  }

  close() {
    this.audio.pause();
    this.audio.removeAttribute("src");
    this.audio.load();
    if (this.snapUrl) URL.revokeObjectURL(this.snapUrl);
    this.el.hidden = true;
  }
}

let player: Player | null = null;
function openPlayer(it: Item) {
  closePlayer();
  player = new Player(it);
  renderRow(it);
}
function closePlayer() {
  if (!player) return;
  const it = player.item;
  player.close();
  player = null;
  renderRow(it);
}

// ---------------------------------------------------------------- staying awake / leaving

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
  if (items.some((x) => x.status === "working") && !paused) {
    wakeLock = null; // released automatically when hidden
    keepAwake(true);
  }
});
window.addEventListener("beforeunload", (e) => {
  // Finished files live only in this tab until saved.
  if (items.some((x) => x.status === "working" || x.status === "queued" || x.status === "ready")) e.preventDefault();
});

renderAll();
renderPending();

if ("serviceWorker" in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register("sw.js").catch((e) => console.warn("Service worker not registered", e));
}
