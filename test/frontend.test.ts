import assert from "node:assert/strict";
import { test } from "node:test";
import type { Denoiser, DenoiserFactory } from "../src/pipeline/denoise.ts";
import { BLOCK_SECONDS, type BlockResult, LEAD_SECONDS, runBlock, TAIL_SECONDS, WORK_RATE } from "../src/pipeline/frontend.ts";

/** A stand-in noise reducer with RNNoise's frame size and 20 ms delay but no other memory. */
const fakeDenoiser: DenoiserFactory = {
  async create(): Promise<Denoiser> {
    const F = 480;
    const hist = [new Float32Array(F), new Float32Array(F)];
    return {
      frameSize: F,
      delay: 2 * F,
      process(frame) {
        let e = 0;
        for (const v of frame) e += v * v;
        const out = hist.shift()!;
        hist.push(frame.slice());
        frame.set(out.map((v) => v * 0.5));
        return e / F > 1e-4 ? 1 : 0;
      },
      destroy() {},
    };
  },
};

function signal(seconds: number, rate: number, channels: number): Float32Array[] {
  let seed = 1;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32) * 2 - 1;
  return Array.from({ length: channels }, (_, c) =>
    Float32Array.from({ length: seconds * rate }, (_, i) => {
      const t = i / rate;
      const talk = Math.sin(2 * Math.PI * 0.3 * t) > 0 ? 0.2 : 0.01; // on/off "speech"
      return talk * Math.sin(2 * Math.PI * (220 + 30 * c) * t) + 0.003 * rnd();
    }),
  );
}

const cat = <T extends Float32Array | Float64Array>(parts: T[]): T => {
  const out = new (parts[0].constructor as { new (n: number): T })(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
};

const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

test("front-end blocks line up with one continuous run", async () => {
  const rate = 44100;
  const seconds = 2 * BLOCK_SECONDS + 17;
  const x = signal(seconds, rate, 2);
  const opts = { channels: 2 as const, rumble: true, noise: true, inRate: rate };
  const whole = await runBlock({ start: 0, end: null, from: 0, input: x.map((c) => c.slice()) }, opts, fakeDenoiser);

  const parts: BlockResult[] = [];
  let fade: Float32Array[] | null = null;
  for (let s = 0; ; s += BLOCK_SECONDS) {
    const last = s + BLOCK_SECONDS + TAIL_SECONDS >= seconds;
    const from = Math.max(0, s - LEAD_SECONDS);
    const end = last ? null : s + BLOCK_SECONDS;
    const input = x.map((c) => c.slice(from * rate, last ? c.length : (end! + TAIL_SECONDS) * rate));
    const r = await runBlock({ start: s, end, from, input }, opts, fakeDenoiser);
    if (!last) assert.equal(r.audio[0].length, BLOCK_SECONDS * WORK_RATE);
    if (fade) {
      // The fade-out must be the same audio as the next block's start.
      for (let c = 0; c < 2; c++) assert.ok(maxDiff(fade[c], r.audio[c].subarray(0, fade[c].length)) < 1e-5);
    }
    fade = r.fadeOut;
    parts.push(r);
    if (last) break;
  }

  for (let c = 0; c < 2; c++) {
    const a = cat(parts.map((p) => p.audio[c]));
    assert.equal(a.length, whole.audio[c].length);
    assert.ok(maxDiff(a, whole.audio[c]) < 1e-5, `audio channel ${c}`);
  }
  for (const key of ["vad", "k", "raw", "hf", "loudness"] as const) {
    const a = cat(parts.map((p) => p[key] as Float32Array));
    const w = whole[key] as Float32Array;
    assert.equal(a.length, w.length, `${key} length`);
    let scale = 0;
    for (const v of w) scale = Math.max(scale, Math.abs(v));
    assert.ok(maxDiff(a, w) <= 1e-4 * scale, `${key} values`);
  }
});
