/// <reference lib="webworker" />
import { rnnoiseFactory } from "./pipeline/denoise.ts";
import { type Block, type BlockResult, type FrontEndOptions, runBlock, transferables } from "./pipeline/frontend.ts";

/** Runs front-end blocks for the processing worker (see pipeline/frontend.ts). */
export type FrontEndRequest = { id: number; block: Block; opts: FrontEndOptions };
export type FrontEndResponse = { id: number; result: BlockResult } | { id: number; error: string };

declare const self: DedicatedWorkerGlobalScope;

self.onmessage = async (e: MessageEvent<FrontEndRequest>) => {
  const { id, block, opts } = e.data;
  try {
    const result = await runBlock(block, opts, rnnoiseFactory);
    self.postMessage({ id, result } satisfies FrontEndResponse, transferables(result));
  } catch (err) {
    self.postMessage({ id, error: err instanceof Error ? err.message : String(err) } satisfies FrontEndResponse);
  }
};
