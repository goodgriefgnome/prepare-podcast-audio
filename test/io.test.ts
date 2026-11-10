import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { openAudio } from "../src/io/mp3.ts";

const hasFfmpeg = (() => {
  try {
    execFileSync("ffmpeg", ["-version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
})();

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ppa-io-"));

/** 2 s of a 440 Hz sine at amplitude 0.5 in the given format. */
function make(name: string, args: string[]): string {
  const out = path.join(dir, name);
  execFileSync("ffmpeg", ["-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=2:sample_rate=48000", "-af", "volume=4", ...args, out]);
  return out;
}

async function decodeAll(file: string) {
  const src = await openAudio(await fs.openAsBlob(file), file);
  let n = 0;
  let peak = 0;
  for await (const chunk of src.chunks()) {
    n += chunk[0].length;
    for (const ch of chunk) for (const v of ch) peak = Math.max(peak, Math.abs(v));
  }
  return { info: src.info, n, peak };
}

const cases: [string, string[], Partial<{ channels: number; sampleRate: number; detail: string }>][] = [
  ["s16.wav", ["-c:a", "pcm_s16le"], { channels: 1, detail: "16-bit" }],
  ["s24-stereo.wav", ["-ac", "2", "-c:a", "pcm_s24le"], { channels: 2, detail: "24-bit" }],
  ["s32.wav", ["-c:a", "pcm_s32le"], { detail: "32-bit" }],
  ["f32.wav", ["-c:a", "pcm_f32le"], { detail: "32-bit float" }],
  ["f64.wav", ["-c:a", "pcm_f64le"], { detail: "64-bit float" }],
  ["u8.wav", ["-c:a", "pcm_u8"], { detail: "8-bit" }],
  ["rf64.wav", ["-c:a", "pcm_s24le", "-rf64", "always"], { detail: "24-bit" }],
  ["extensible-4ch.wav", ["-ac", "4", "-c:a", "pcm_s16le"], { channels: 4 }],
  ["44k.wav", ["-ar", "44100", "-c:a", "pcm_s16le"], { sampleRate: 44100 }],
];

for (const [name, args, expect] of cases) {
  test(`reads ${name}`, { skip: !hasFfmpeg }, async () => {
    const { info, n, peak } = await decodeAll(make(name, args));
    assert.equal(info.format, "wav");
    for (const [k, v] of Object.entries(expect)) assert.equal(info[k as keyof typeof info], v, k);
    assert.equal(n, Math.round(2 * info.sampleRate));
    assert.ok(Math.abs(info.duration - 2) < 1e-6);
    // ffmpeg's mono→stereo upmix lowers each channel by 3 dB; others don't.
    const ok = [0.5, 0.5 / Math.SQRT2].some((e) => Math.abs(peak - e) < 0.02);
    assert.ok(info.channels === 1 ? Math.abs(peak - 0.5) < 0.02 : ok, `peak ${peak}`);
  });
}

for (const [name, args, expectDetail] of [
  ["cbr.mp3", ["-c:a", "libmp3lame", "-b:a", "128k"], "128 kbps"],
  ["vbr.mp3", ["-c:a", "libmp3lame", "-q:a", "4"], "VBR"],
  ["tagged.mp3", ["-c:a", "libmp3lame", "-b:a", "96k", "-metadata", "title=Episode 1", "-id3v2_version", "4"], "96 kbps"],
  ["22k.mp3", ["-ar", "22050", "-c:a", "libmp3lame", "-b:a", "48k"], "48 kbps"],
] as const) {
  test(`reads ${name}`, { skip: !hasFfmpeg }, async () => {
    const { info, n, peak } = await decodeAll(make(name, [...args]));
    assert.equal(info.format, "mp3");
    assert.equal(info.detail, expectDetail);
    assert.ok(Math.abs(info.duration - 2) < 0.1, `duration ${info.duration}`);
    // Gapless decoding should give back (very nearly) the original length.
    assert.ok(Math.abs(n / info.sampleRate - 2) < 0.03, `decoded ${n / info.sampleRate}s`);
    assert.ok(Math.abs(peak - 0.5) < 0.06, `peak ${peak}`);
  });
}

test("rejects files that aren't WAV or MP3", { skip: !hasFfmpeg }, async () => {
  const f = make("x.flac", ["-c:a", "flac"]);
  await assert.rejects(openAudio(await fs.openAsBlob(f), f), /doesn’t look like an MP3/);
});
