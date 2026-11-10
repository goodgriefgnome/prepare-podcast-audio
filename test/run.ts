/**
 * Runs the full pipeline in Node on a file and copies the result next to it.
 *
 *   node --experimental-transform-types test/run.ts input.wav output.mp3 [mono|stereo] [--off=noise,level,...] [--trim=-3]
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// The RNNoise build only runs in browser-like environments.
(globalThis as Record<string, unknown>).WorkerGlobalScope ??= class {};
(globalThis as Record<string, unknown>).DEBUG_PPA = true;

const { openAudio } = await import("../src/io/mp3.ts");
const { processAudio } = await import("../src/pipeline/process.ts");
const { rnnoiseFactory } = await import("../src/pipeline/denoise.ts");
const { NodeStorage } = await import("./node-storage.ts");

const [input, output, ch = "mono", ...rest] = process.argv.slice(2);
const off = new Set((rest.find((r) => r.startsWith("--off="))?.slice(6) ?? "").split(",").filter(Boolean));
const blob = await fs.openAsBlob(input);
const source = await openAudio(blob, input);
console.log("input", source.info);
const storage = new NodeStorage(fs.mkdtempSync(path.join(os.tmpdir(), "ppa-")));
const t0 = Date.now();
let lastStage = "";
const result = await processAudio(
  source,
  {
    channels: ch === "stereo" ? 2 : 1,
    rumble: !off.has("rumble"),
    noise: !off.has("noise"),
    spikes: !off.has("spikes"),
    sibilance: !off.has("sibilance"),
    level: !off.has("level"),
    outputTrimDb: Number(rest.find((r) => r.startsWith("--trim="))?.slice(7) ?? 0),
  },
  {
    storage,
    denoiser: rnnoiseFactory,
    onProgress(p) {
      if (p.stage !== lastStage) {
        console.log(`${((Date.now() - t0) / 1000).toFixed(1)}s ${p.stage} (${(p.fraction * 100).toFixed(0)}%)`);
        lastStage = p.stage;
      }
    },
    checkCancel() {},
  },
);
fs.copyFileSync(path.join(storage.dir, result.outputName), output);
fs.rmSync(storage.dir, { recursive: true, force: true });
console.log("result", result, `${((Date.now() - t0) / 1000).toFixed(1)}s`);
console.log("peak RSS MB", Math.round(process.resourceUsage().maxRSS / 1024));
