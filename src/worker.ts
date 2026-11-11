/// <reference lib="webworker" />
import { openAudio } from "./io/mp3.ts";
import { UnsupportedFileError, type AudioInfo } from "./io/source.ts";
import { OpfsStorage, type ByteStore, type Storage } from "./io/store.ts";
import { rnnoiseFactory } from "./pipeline/denoise.ts";
import { processAudio, type Options, type Progress, type Result } from "./pipeline/process.ts";

export type Request =
  | { type: "probe"; id: number; file: File }
  /**
   * Processes one file into `outputName` in the app's private storage.
   * `fresh` clears outputs left from earlier visits first.
   */
  | { type: "process"; id: number; file: File; options: Options; outputName: string; previous: string | null; fresh: boolean }
  | { type: "pause" }
  | { type: "resume" }
  /** Asks for a copy of the MP3 written so far by the file being processed. */
  | { type: "snapshot"; id: number };

export type Response =
  | { type: "info"; id: number; info: AudioInfo }
  | { type: "probeError"; id: number; message: string }
  | { type: "progress"; id: number; progress: Progress }
  | { type: "done"; id: number; result: Result; output: Blob }
  | { type: "error"; id: number; message: string }
  /** `blob` is null when that file isn't being processed any more. */
  | { type: "snapshot"; id: number; blob: Blob | null };

declare const self: DedicatedWorkerGlobalScope;
const post = (r: Response) => self.postMessage(r);

let current: { id: number; out: ByteStore | null } | null = null;
let paused: { promise: Promise<void>; resume: () => void } | null = null;

/** Storage that remembers the MP3 being written so it can be copied on request. */
function tracking(storage: Storage, onBytes: (b: ByteStore) => void): Storage {
  return {
    samples: (n, c) => storage.samples(n, c),
    bytes: async (n) => {
      const b = await storage.bytes(n);
      onBytes(b);
      return b;
    },
    close: (n) => storage.close(n),
    blob: (n) => storage.blob(n),
    remove: (n) => storage.remove(n),
  };
}

self.onmessage = async (e: MessageEvent<Request>) => {
  const msg = e.data;
  switch (msg.type) {
    case "probe":
      try {
        const src = await openAudio(msg.file, msg.file.name);
        post({ type: "info", id: msg.id, info: src.info });
      } catch (err) {
        const message = err instanceof UnsupportedFileError ? err.message : "This file couldn’t be read.";
        post({ type: "probeError", id: msg.id, message });
      }
      return;
    case "pause":
      if (!paused) {
        let resume = () => {};
        const promise = new Promise<void>((r) => (resume = r));
        paused = { promise, resume };
      }
      return;
    case "resume":
      paused?.resume();
      paused = null;
      return;
    case "snapshot": {
      const out = current && current.id === msg.id ? current.out : null;
      const bytes = out?.snapshot();
      post({ type: "snapshot", id: msg.id, blob: bytes ? new Blob([bytes as Uint8Array<ArrayBuffer>], { type: "audio/mpeg" }) : null });
      return;
    }
    case "process":
      await processOne(msg);
      return;
  }
};

async function processOne(msg: Extract<Request, { type: "process" }>) {
  const job = { id: msg.id, out: null as ByteStore | null };
  current = job;
  try {
    const src = await openAudio(msg.file, msg.file.name);
    const opfs = await OpfsStorage.open("work", msg.fresh);
    // Tidy up this file's previous run (it may still be locked; that's fine).
    if (msg.previous) await opfs.remove(msg.previous).catch(() => {});
    const storage = tracking(opfs, (b) => (job.out = b));
    const result = await processAudio(src, msg.options, {
      storage,
      outputName: msg.outputName,
      denoiser: rnnoiseFactory,
      onProgress: (progress) => post({ type: "progress", id: msg.id, progress }),
      checkCancel: () => {},
      gate: () => paused?.promise ?? Promise.resolve(),
    });
    current = null;
    post({ type: "done", id: msg.id, result, output: await storage.blob(result.outputName) });
  } catch (err) {
    current = null;
    console.error(err);
    const name = err instanceof DOMException ? err.name : "";
    const message =
      name === "QuotaExceededError"
        ? "Your device ran out of storage space. Free up some space, then remove and add this file again."
        : name === "NoModificationAllowedError"
          ? "This app is already processing files in another tab or window. Close it, then add this file again."
          : err instanceof UnsupportedFileError
            ? err.message
            : `Something went wrong while processing: ${err instanceof Error ? err.message : String(err)}`;
    post({ type: "error", id: msg.id, message });
  }
}
