/// <reference lib="webworker" />
import { openAudio } from "./io/mp3.ts";
import { UnsupportedFileError, type AudioInfo, type Planar } from "./io/source.ts";
import { OpfsStorage, type Storage } from "./io/store.ts";
import { rnnoiseFactory } from "./pipeline/denoise.ts";
import { OUT_RATE, processAudio, type Options, type Progress, type Result } from "./pipeline/process.ts";

export type Request =
  | { type: "probe"; id: number; file: File }
  /**
   * Processes one file into `outputName` in the app's private storage.
   * `fresh` clears outputs from earlier batches first.
   */
  | { type: "process"; file: File; options: Options; outputName: string; fresh: boolean }
  /** Streams the first `seconds` of the file as it is (original preview). */
  | { type: "previewOriginal"; file: File; seconds: number }
  /** Streams the first `seconds` of the file processed with `options`. */
  | { type: "previewProcessed"; file: File; options: Options; seconds: number };

export type Response =
  | { type: "info"; id: number; info: AudioInfo }
  | { type: "probeError"; id: number; message: string }
  | { type: "progress"; progress: Progress }
  | { type: "done"; result: Result; output: Blob }
  | { type: "error"; message: string; userFacing: boolean }
  /** A chunk of preview audio. */
  | { type: "pcm"; sampleRate: number; channels: Float32Array[] }
  /** The preview stream has ended. */
  | { type: "pcmEnd" };

declare const self: DedicatedWorkerGlobalScope;

const post = (r: Response, transfer: Transferable[] = []) => self.postMessage(r, transfer);

function postPcm(sampleRate: number, p: Planar) {
  const channels = p.map((c) => c.slice());
  post({ type: "pcm", sampleRate, channels }, channels.map((c) => c.buffer));
}

/** Previews never write files, but the pipeline wants a Storage. */
const noStorage: Storage = {
  samples: () => Promise.reject(new Error("unused")),
  bytes: () => Promise.reject(new Error("unused")),
  close: async () => {},
  blob: () => Promise.reject(new Error("unused")),
  remove: async () => {},
};

self.onmessage = async (e: MessageEvent<Request>) => {
  const msg = e.data;
  try {
    if (msg.type === "probe") {
      try {
        const src = await openAudio(msg.file, msg.file.name);
        post({ type: "info", id: msg.id, info: src.info });
      } catch (err) {
        const message = err instanceof UnsupportedFileError ? err.message : "This file couldn’t be read.";
        post({ type: "probeError", id: msg.id, message });
      }
      return;
    }
    if (msg.type === "previewOriginal") {
      const src = await openAudio(msg.file, msg.file.name);
      const max = Math.round(msg.seconds * src.info.sampleRate);
      let sent = 0;
      for await (const chunk of src.chunks()) {
        const n = Math.min(chunk[0].length, max - sent);
        postPcm(src.info.sampleRate, chunk.map((c) => c.subarray(0, n)));
        sent += n;
        if (sent >= max) break;
      }
      post({ type: "pcmEnd" });
      return;
    }
    if (msg.type === "previewProcessed") {
      const src = await openAudio(msg.file, msg.file.name);
      await processAudio(src, { ...msg.options, maxSeconds: msg.seconds }, {
        storage: noStorage,
        denoiser: rnnoiseFactory,
        onProgress: (progress) => post({ type: "progress", progress }),
        checkCancel: () => {},
        sink: (p) => postPcm(OUT_RATE, p),
      });
      post({ type: "pcmEnd" });
      return;
    }
    const src = await openAudio(msg.file, msg.file.name);
    const storage = await OpfsStorage.open("work", msg.fresh);
    const result = await processAudio(src, msg.options, {
      storage,
      outputName: msg.outputName,
      denoiser: rnnoiseFactory,
      onProgress: (progress) => post({ type: "progress", progress }),
      checkCancel: () => {},
    });
    post({ type: "done", result, output: await storage.blob(result.outputName) });
  } catch (err) {
    console.error(err);
    const name = err instanceof DOMException ? err.name : "";
    if (name === "QuotaExceededError") {
      post({ type: "error", userFacing: true, message: "Your device ran out of storage space while processing. Free up some space and try again." });
    } else if (name === "NoModificationAllowedError") {
      post({ type: "error", userFacing: true, message: "This app is already processing a file in another tab or window. Close it, or wait for it to finish, then try again." });
    } else {
      post({ type: "error", userFacing: err instanceof UnsupportedFileError, message: err instanceof Error ? err.message : String(err) });
    }
  }
};
