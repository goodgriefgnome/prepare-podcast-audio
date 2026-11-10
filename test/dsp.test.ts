import { test } from "node:test";
import assert from "node:assert/strict";
import { LoudnessMeter } from "../src/dsp/loudness.ts";
import { Resampler } from "../src/dsp/resampler.ts";
import { TruePeakLimiter, TruePeakMeter } from "../src/dsp/truepeak.ts";

function sine(freq: number, fs: number, n: number, amp: number, phase = 0): Float32Array {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = amp * Math.sin((2 * Math.PI * freq * i) / fs + phase);
  return x;
}

function chunked(x: Float32Array[], size: number, fn: (c: Float32Array[]) => void) {
  for (let i = 0; i < x[0].length; i += size) fn(x.map((ch) => ch.subarray(i, Math.min(i + size, ch.length))));
}

test("BS.1770: stereo 1 kHz at −23 dBFS reads −23 LUFS", () => {
  for (const fs of [44100, 48000]) {
    const amp = Math.pow(10, -23 / 20);
    const s = sine(1000, fs, fs * 20, amp);
    const m = new LoudnessMeter(fs, 2);
    chunked([s, s], 4096, (c) => m.push(c));
    assert.ok(Math.abs(m.integrated() - -23) < 0.05, `${fs}: ${m.integrated()}`);
  }
});

test("BS.1770: mono is measured as dual mono", () => {
  const fs = 48000;
  const s = sine(1000, fs, fs * 10, Math.pow(10, -23 / 20));
  const dual = new LoudnessMeter(fs, 1, true);
  dual.push([s]);
  const single = new LoudnessMeter(fs, 1, false);
  single.push([s]);
  assert.ok(Math.abs(dual.integrated() - -23) < 0.05, `${dual.integrated()}`);
  assert.ok(Math.abs(single.integrated() - -26.01) < 0.05, `${single.integrated()}`);
});

test("BS.1770: gating ignores silence", () => {
  const fs = 48000;
  const s = sine(1000, fs, fs * 10, Math.pow(10, -23 / 20));
  const sil = new Float32Array(fs * 30);
  const m = new LoudnessMeter(fs, 2);
  m.push([s, s]);
  m.push([sil, sil]);
  assert.ok(Math.abs(m.integrated() - -23) < 0.1, `${m.integrated()}`);
});

for (const [a, b] of [
  [44100, 48000],
  [48000, 44100],
  [22050, 48000],
  [96000, 48000],
  [48000, 48000],
]) {
  test(`resampler ${a} → ${b}: length, gain and cleanliness`, () => {
    const n = a * 3 + 123;
    const f = 997;
    const x = sine(f, a, n, 0.5);
    const r = new Resampler(a, b, 1);
    const parts: Float32Array[] = [];
    chunked([x], 10007, (c) => parts.push(r.push(c)[0]));
    parts.push(r.flush()[0]);
    const y = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
    let o = 0;
    for (const p of parts) {
      y.set(p, o);
      o += p.length;
    }
    assert.equal(y.length, Math.round((n * b) / a));
    // Compare the middle against an ideal sine at the new rate.
    const ref = sine(f, b, y.length, 0.5);
    let err = 0, sig = 0;
    for (let i = b; i < y.length - b; i++) {
      err += (y[i] - ref[i]) ** 2;
      sig += ref[i] ** 2;
    }
    const snr = 10 * Math.log10(sig / err);
    assert.ok(snr > 80, `SNR ${snr.toFixed(1)} dB`);
  });
}

test("resampler removes content above the new Nyquist", () => {
  const x = sine(23000, 48000, 48000, 0.5);
  const r = new Resampler(48000, 44100, 1);
  const y = r.push([x])[0];
  let p = 0;
  for (let i = 2000; i < y.length - 2000; i++) p = Math.max(p, Math.abs(y[i]));
  assert.ok(20 * Math.log10(p / 0.5) < -80, `leak ${20 * Math.log10(p / 0.5)} dB`);
});

test("true-peak meter sees inter-sample peaks", () => {
  // fs/4 sine with 45° phase: samples at ±0.707·A, true peak A.
  const fs = 48000;
  const x = sine(fs / 4, fs, 4800, 0.5, Math.PI / 4);
  const m = new TruePeakMeter(1);
  m.push([x]);
  m.flush();
  assert.ok(Math.abs(m.dbtp - 20 * Math.log10(0.5)) < 0.3, `${m.dbtp}`);
});

test("limiter keeps true peak under the ceiling and preserves length", () => {
  const fs = 44100;
  const n = fs * 5;
  const x = new Float32Array(n);
  let seed = 1;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296) * 2 - 1;
  for (let i = 0; i < n; i++) x[i] = 0.3 * rnd() * (i % 20000 < 2000 ? 4 : 1) + 0.2 * Math.sin(i / 7);
  const lim = new TruePeakLimiter(fs, 2, -1);
  const outs: Float32Array[][] = [];
  chunked([x, x.map((v) => -v * 0.8)], 4096, (c) => outs.push(lim.push(c)));
  outs.push(lim.flush());
  const total = outs.reduce((s, o) => s + o[0].length, 0);
  assert.equal(total, n);
  const m = new TruePeakMeter(2);
  for (const o of outs) if (o[0].length) m.push(o);
  m.flush();
  assert.ok(m.dbtp <= -0.95, `TP ${m.dbtp}`);
  // Quiet parts are untouched (after the delay is accounted for).
  const y = new Float32Array(n);
  let off = 0;
  for (const o of outs) {
    y.set(o[0], off);
    off += o[0].length;
  }
  // 17000 samples (≈5 release time constants) after the last burst.
  const i = 219000;
  assert.ok(Math.abs(y[i] - x[i]) < 0.01 * Math.abs(x[i]) + 1e-4, `quiet sample changed ${y[i]} vs ${x[i]}`);
});
