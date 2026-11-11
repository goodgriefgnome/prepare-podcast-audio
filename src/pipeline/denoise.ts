/**
 * Frame-based noise reducer running at 48 kHz on one channel.
 * `process` works in place on a frame of `frameSize` samples and returns a
 * voice-activity probability (0–1) for that frame.
 */
export interface Denoiser {
  readonly frameSize: number;
  /** Output lags input by this many samples. */
  readonly delay: number;
  process(frame: Float32Array): number;
  destroy(): void;
}

export interface DenoiserFactory {
  create(): Promise<Denoiser>;
}

interface RnnoiseExports {
  memory: WebAssembly.Memory;
  _initialize?: () => void;
  malloc(n: number): number;
  free(p: number): void;
  rnnoise_create(model: number): number;
  rnnoise_destroy(st: number): void;
  rnnoise_process_frame(st: number, out: number, input: number): number;
  rnnoise_get_frame_size(): number;
}

/** RNNoise built by rnnoise/build.sh (see the README). */
const WASM_URL = new URL("../vendor/rnnoise.wasm", import.meta.url);

async function wasmBytes(url: URL): Promise<ArrayBuffer> {
  if (url.protocol === "file:") {
    // Node (tests and the command-line runner).
    const fs = "node:fs/promises";
    const { readFile } = (await import(/* @vite-ignore */ fs)) as typeof import("node:fs/promises");
    const b = await readFile(url);
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Couldn't load the noise reducer (${res.status})`);
  return res.arrayBuffer();
}

async function rnnoiseModule(url: URL): Promise<WebAssembly.Module> {
  return WebAssembly.compile(await wasmBytes(url));
}

/**
 * The module is built without JavaScript glue; anything it imports (WASI
 * calls the C library might make on error paths) is not expected to run.
 */
function stubImports(mod: WebAssembly.Module): WebAssembly.Imports {
  const imports: Record<string, Record<string, WebAssembly.ImportValue>> = {};
  for (const imp of WebAssembly.Module.imports(mod)) {
    if (imp.kind !== "function") throw new Error(`RNNoise imports a ${imp.kind}: ${imp.module}.${imp.name}`);
    const name = `${imp.module}.${imp.name}`;
    (imports[imp.module] ??= {})[imp.name] =
      imp.name === "emscripten_notify_memory_growth"
        ? () => {}
        : () => {
            throw new Error(`RNNoise called ${name}`);
          };
  }
  return imports;
}

interface Instance {
  x: RnnoiseExports;
  frameSize: number;
}

async function instantiate(mod: WebAssembly.Module): Promise<Instance> {
  const inst = await WebAssembly.instantiate(mod, stubImports(mod));
  const x = inst.exports as unknown as RnnoiseExports;
  x._initialize?.();
  return { x, frameSize: x.rnnoise_get_frame_size() };
}

/** Creates a denoiser state in a module instance (several can share one). */
function createState({ x, frameSize }: Instance): Denoiser {
  const st = x.rnnoise_create(0);
  const buf = x.malloc(frameSize * 4);
  if (!st || !buf) throw new Error("RNNoise could not allocate its state");
  let heap = new Float32Array(x.memory.buffer, buf, frameSize);
  return {
    frameSize,
    // Measured: output lags input by two frames (frame + look-ahead), 20 ms.
    delay: 2 * frameSize,
    process(frame: Float32Array) {
      if (heap.buffer !== x.memory.buffer) heap = new Float32Array(x.memory.buffer, buf, frameSize);
      // RNNoise expects 16-bit-scaled samples.
      for (let i = 0; i < frameSize; i++) heap[i] = frame[i] * 32768;
      const vad = x.rnnoise_process_frame(st, buf, buf);
      for (let i = 0; i < frameSize; i++) frame[i] = heap[i] / 32768;
      return vad;
    },
    destroy() {
      x.rnnoise_destroy(st);
      x.free(buf);
    },
  };
}

/** A factory for a specific build (used to compare builds). */
export function rnnoiseFactoryFor(url: URL): DenoiserFactory {
  let inst: Promise<Instance> | null = null;
  return {
    async create() {
      inst ??= rnnoiseModule(url).then(instantiate);
      return createState(await inst);
    },
  };
}

/** The shipped build; one module instance per thread, shared by all states. */
export const rnnoiseFactory: DenoiserFactory = rnnoiseFactoryFor(WASM_URL);
