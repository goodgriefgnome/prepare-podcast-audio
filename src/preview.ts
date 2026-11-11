import type { Options } from "./pipeline/process.ts";
import type { Request, Response } from "./worker.ts";

export const PREVIEW_SECONDS = 180;

type Which = "original" | "processed";

/** Audio for one side of the A/B, arriving in chunks. */
class Track {
  chunks: { start: number; buf: AudioBuffer }[] = [];
  /** Seconds of audio received so far. */
  available = 0;
  complete = false;
  worker: Worker | null = null;
  /** 0–1 progress of the processed track while it's being made. */
  progress = 0;
  error: string | null = null;

  reset(): void {
    this.worker?.terminate();
    this.worker = null;
    this.chunks = [];
    this.available = 0;
    this.complete = false;
    this.progress = 0;
    this.error = null;
  }
}

/**
 * Streams the first three minutes of a file, as it is and as processed,
 * into one player with an A/B switch that keeps the playback position.
 */
export class Preview {
  private ctx: AudioContext | null = null;
  private tracks: Record<Which, Track> = { original: new Track(), processed: new Track() };
  private which: Which = "original";
  private playing = false;
  /** Waiting for the processed track to catch up with playback. */
  private stalled = false;
  private pos = 0; // seconds, while paused
  private startedAt = 0; // ctx.currentTime when pos was last set while playing
  private nodes: AudioBufferSourceNode[] = [];
  private gain: GainNode | null = null;
  private file: File | null = null;
  private options: Options | null = null;
  private optionsKey = "";
  private restartTimer = 0;
  private raf = 0;
  private duration = PREVIEW_SECONDS;

  constructor(
    private ui: {
      root: HTMLElement;
      play: HTMLButtonElement;
      original: HTMLButtonElement;
      processed: HTMLButtonElement;
      seek: HTMLInputElement;
      time: HTMLElement;
      status: HTMLElement;
    },
  ) {
    ui.play.addEventListener("click", () => (this.playing ? this.pause() : this.play()));
    ui.original.addEventListener("click", () => this.select("original"));
    ui.processed.addEventListener("click", () => this.select("processed"));
    ui.seek.addEventListener("input", () => this.seek(Number(ui.seek.value)));
    this.render();
  }

  /** Shows a new file: the original streams straight away. */
  setFile(file: File, durationSeconds: number, options: Options): void {
    this.stop();
    this.file = file;
    this.duration = Math.max(0.5, Math.min(PREVIEW_SECONDS, durationSeconds));
    this.pos = 0;
    this.tracks.original.reset();
    this.tracks.processed.reset();
    this.options = options;
    this.optionsKey = JSON.stringify(options);
    this.startTrack("original");
    this.render();
  }

  /** Settings changed: rebuild the processed side (debounced). */
  setOptions(options: Options): void {
    const key = JSON.stringify(options);
    if (key === this.optionsKey) return;
    this.options = options;
    this.optionsKey = key;
    if (!this.tracks.processed.worker && !this.tracks.processed.complete) return; // not started yet
    clearTimeout(this.restartTimer);
    this.restartTimer = window.setTimeout(() => {
      const wasPlaying = this.playing && this.which === "processed";
      if (wasPlaying) this.pause();
      this.tracks.processed.reset();
      this.startTrack("processed");
      if (wasPlaying) this.play();
      this.render();
    }, 400);
  }

  /** Stops playback and all preview work (e.g. when leaving the screen). */
  stop(): void {
    this.pause();
    clearTimeout(this.restartTimer);
    this.tracks.original.reset();
    this.tracks.processed.reset();
    this.file = null;
    this.render();
  }

  private startTrack(which: Which): void {
    if (!this.file || !this.options) return;
    const t = this.tracks[which];
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    t.worker = w;
    w.onmessage = (e: MessageEvent<Response>) => this.onMessage(which, w, e.data);
    w.onerror = () => {
      t.error = "The preview couldn’t be made.";
      this.render();
    };
    const req: Request =
      which === "original"
        ? { type: "previewOriginal", file: this.file, seconds: PREVIEW_SECONDS }
        : { type: "previewProcessed", file: this.file, options: this.options, seconds: PREVIEW_SECONDS };
    w.postMessage(req);
  }

  private onMessage(which: Which, from: Worker, m: Response): void {
    const t = this.tracks[which];
    if (t.worker !== from) return; // a cancelled run
    if (m.type === "pcm") {
      const ctx = this.audio();
      const n = m.channels[0].length;
      if (!n) return;
      const buf = ctx.createBuffer(m.channels.length, n, m.sampleRate);
      m.channels.forEach((c, i) => buf.copyToChannel(c as Float32Array<ArrayBuffer>, i));
      const chunk = { start: t.available, buf };
      t.chunks.push(chunk);
      t.available += buf.duration;
      if (this.playing && this.which === which && !this.stalled) this.schedule(chunk);
      if (this.stalled && this.which === which && t.available > this.pos + 3) this.resume();
    } else if (m.type === "pcmEnd") {
      t.complete = true;
      t.worker?.terminate();
      t.worker = null;
      if (this.stalled && this.which === which) this.resume();
    } else if (m.type === "progress") {
      t.progress = m.progress.fraction;
    } else if (m.type === "error") {
      t.error = m.message;
      t.worker?.terminate();
      t.worker = null;
    }
    this.render();
  }

  private audio(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.gain = this.ctx.createGain();
      this.gain.connect(this.ctx.destination);
    }
    return this.ctx;
  }

  private now(): number {
    if (!this.playing || this.stalled || !this.ctx) return this.pos;
    return this.pos + (this.ctx.currentTime - this.startedAt);
  }

  private schedule(chunk: { start: number; buf: AudioBuffer }): void {
    const ctx = this.ctx!;
    const end = chunk.start + chunk.buf.duration;
    const from = Math.max(this.now(), chunk.start);
    if (from >= end) return;
    const node = ctx.createBufferSource();
    node.buffer = chunk.buf;
    node.connect(this.gain!);
    const when = this.startedAt + (from - this.pos);
    node.start(Math.max(ctx.currentTime, when), from - chunk.start);
    node.onended = () => {
      this.nodes = this.nodes.filter((x) => x !== node);
    };
    this.nodes.push(node);
  }

  private startNodes(): void {
    const ctx = this.ctx!;
    this.startedAt = ctx.currentTime + 0.03;
    for (const c of this.tracks[this.which].chunks) this.schedule(c);
  }

  private stopNodes(): void {
    for (const n of this.nodes) {
      n.onended = null;
      try {
        n.stop();
      } catch {
        /* not started */
      }
    }
    this.nodes = [];
  }

  play(): void {
    if (!this.file) return;
    const ctx = this.audio();
    void ctx.resume();
    if (this.pos >= this.duration - 0.05) this.pos = 0;
    // Start making the processed side the first time anyone plays.
    const p = this.tracks.processed;
    if (!p.worker && !p.complete && !p.error) this.startTrack("processed");
    this.playing = true;
    this.stalled = false;
    this.ensureAvailable();
    if (!this.stalled) this.startNodes();
    this.tick();
    this.render();
  }

  pause(): void {
    if (!this.playing) return;
    this.pos = this.now();
    this.playing = false;
    this.stalled = false;
    this.stopNodes();
    cancelAnimationFrame(this.raf);
    this.render();
  }

  private ensureAvailable(): void {
    const t = this.tracks[this.which];
    if (!t.complete && this.pos > t.available - 0.2) {
      this.stalled = true;
      this.stopNodes();
    }
  }

  private resume(): void {
    this.stalled = false;
    if (this.playing) this.startNodes();
  }

  private select(which: Which): void {
    if (which === this.which) return;
    const at = this.now();
    this.stopNodes();
    this.which = which;
    this.pos = at;
    if (which === "processed") {
      const p = this.tracks.processed;
      if (!p.worker && !p.complete && !p.error) this.startTrack("processed");
    }
    if (this.playing) {
      this.stalled = false;
      this.ensureAvailable();
      if (!this.stalled) this.startNodes();
    }
    this.render();
  }

  private seek(to: number): void {
    const at = Math.max(0, Math.min(this.duration, to));
    this.stopNodes();
    this.pos = at;
    if (this.playing) {
      this.stalled = false;
      this.ensureAvailable();
      if (!this.stalled) this.startNodes();
    }
    this.render();
  }

  private tick = (): void => {
    if (!this.playing) return;
    const t = this.tracks[this.which];
    const at = this.now();
    if (!this.stalled) {
      if (at >= this.duration - 0.02 || (t.complete && at >= t.available - 0.02)) {
        this.pause();
        this.pos = 0;
        this.render();
        return;
      }
      if (!t.complete && at > t.available - 0.1) {
        // Caught up with processing: wait for more.
        this.pos = at;
        this.stalled = true;
        this.stopNodes();
      }
    }
    this.render();
    this.raf = requestAnimationFrame(this.tick);
  };

  private render(): void {
    const ui = this.ui;
    const at = this.now();
    ui.play.textContent = this.playing ? "Pause" : "Play";
    ui.play.setAttribute("aria-pressed", String(this.playing));
    ui.original.setAttribute("aria-pressed", String(this.which === "original"));
    ui.processed.setAttribute("aria-pressed", String(this.which === "processed"));
    ui.seek.max = String(this.duration);
    ui.seek.value = String(at);
    const avail = Math.min(1, this.tracks[this.which].available / this.duration);
    ui.seek.style.setProperty("--avail", `${avail * 100}%`);
    ui.time.textContent = `${clock(at)} / ${clock(this.duration)}`;
    const p = this.tracks.processed;
    let status = "";
    if (this.which === "processed") {
      if (p.error) status = p.error;
      else if (this.stalled) status = p.available === 0 ? "Preparing the processed preview… it starts once about a minute and a half has been read." : "Waiting for processing to catch up…";
      else if (!p.complete && (p.worker || p.available)) status = `Processing the preview… ${Math.floor(p.progress * 100)}%`;
    }
    ui.status.textContent = status;
    ui.status.hidden = !status;
  }
}

function clock(s: number): string {
  s = Math.max(0, Math.floor(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
