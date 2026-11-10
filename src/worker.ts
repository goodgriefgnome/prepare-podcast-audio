/// <reference lib="webworker" />
import { openAudio } from "./io/mp3.ts";
import { UnsupportedFileError, type AudioInfo } from "./io/source.ts";
import { OpfsStorage } from "./io/store.ts";
import { rnnoiseFactory } from "./pipeline/denoise.ts";
import { processAudio, type Options, type Progress, type Result } from "./pipeline/process.ts";

export type Request =
  | { type: "probe"; file: File }
  | { type: "process"; file: File; options: Options }
  | { type: "cancel" };

export type Response =
  | { type: "info"; info: AudioInfo }
  | { type: "progress"; progress: Progress }
  | { type: "done"; result: Result; output: Blob }
  | { type: "error"; message: string; userFacing: boolean }
  | { type: "cancelled" };

class Cancelled extends Error {}

declare const self: DedicatedWorkerGlobalScope;
let cancelled = false;

const post = (r: Response) => self.postMessage(r);

self.onmessage = async (e: MessageEvent<Request>) => {
  const msg = e.data;
  if (msg.type === "cancel") {
    cancelled = true;
    return;
  }
  try {
    if (msg.type === "probe") {
      const src = await openAudio(msg.file, msg.file.name);
      post({ type: "info", info: src.info });
      return;
    }
    cancelled = false;
    const src = await openAudio(msg.file, msg.file.name);
    const storage = await OpfsStorage.open("work");
    const result = await processAudio(src, msg.options, {
      storage,
      denoiser: rnnoiseFactory,
      onProgress: (progress) => post({ type: "progress", progress }),
      checkCancel: () => {
        if (cancelled) throw new Cancelled();
      },
    });
    post({ type: "done", result, output: await storage.blob(result.outputName) });
  } catch (err) {
    if (err instanceof Cancelled) {
      post({ type: "cancelled" });
      return;
    }
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
