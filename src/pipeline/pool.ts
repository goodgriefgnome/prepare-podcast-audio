import type { FrontEndRequest, FrontEndResponse } from "../frontend-worker.ts";
import type { Block, BlockResult, FrontEnd, FrontEndOptions } from "./frontend.ts";

interface Pending {
  resolve(r: BlockResult): void;
  reject(e: Error): void;
}

/**
 * Runs front-end blocks on `count` worker threads. Jobs queue until a worker
 * is free; results come back in whatever order they finish.
 */
export function workerFrontEnd(count: number): FrontEnd {
  const idle: Worker[] = [];
  const busy = new Map<Worker, number>();
  const pending = new Map<number, Pending>();
  const queue: FrontEndRequest[] = [];
  let nextId = 0;

  const dispatch = () => {
    while (idle.length && queue.length) {
      const w = idle.pop()!;
      const req = queue.shift()!;
      busy.set(w, req.id);
      w.postMessage(req, req.block.input.map((a) => a.buffer as ArrayBuffer));
    }
  };
  const fail = (w: Worker, message: string) => {
    const id = busy.get(w);
    busy.delete(w);
    if (id !== undefined) {
      pending.get(id)?.reject(new Error(message));
      pending.delete(id);
    }
  };

  for (let i = 0; i < count; i++) {
    const w = new Worker(new URL("../frontend-worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (e: MessageEvent<FrontEndResponse>) => {
      const r = e.data;
      busy.delete(w);
      idle.push(w);
      const p = pending.get(r.id);
      pending.delete(r.id);
      if ("error" in r) p?.reject(new Error(r.error));
      else p?.resolve(r.result);
      dispatch();
    };
    w.onerror = (e) => {
      e.preventDefault();
      fail(w, e.message || "A processing thread failed.");
      idle.push(w);
      dispatch();
    };
    idle.push(w);
  }

  return {
    concurrency: count,
    run(block: Block, opts: FrontEndOptions) {
      const id = nextId++;
      return new Promise<BlockResult>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        queue.push({ id, block, opts });
        dispatch();
      });
    },
  };
}

/** Threads to use for the front end: leave one core for the rest of the pipeline, at most 4. */
export function frontEndThreads(): number {
  const cores = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 2 : 2;
  return Math.max(1, Math.min(4, cores - 1));
}
