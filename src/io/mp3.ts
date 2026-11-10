import { MPEGDecoder } from "mpg123-decoder";
import { type AudioSource, type AudioInfo, type Planar, UnsupportedFileError, readBytes } from "./source.ts";

const BITRATES: Record<string, number[]> = {
  "1-1": [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
  "1-2": [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
  "1-3": [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
  "2-1": [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
  "2-2": [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
  "2-3": [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
};
const RATES: Record<number, number[]> = { 1: [44100, 48000, 32000], 2: [22050, 24000, 16000], 25: [11025, 12000, 8000] };

export interface FrameHeader {
  version: 1 | 2 | 25;
  layer: 1 | 2 | 3;
  bitrate: number; // kbps
  sampleRate: number;
  channels: 1 | 2;
  length: number; // bytes
  samples: number;
}

export function parseFrameHeader(b: Uint8Array, o: number): FrameHeader | null {
  if (o + 4 > b.length) return null;
  if (b[o] !== 0xff || (b[o + 1] & 0xe0) !== 0xe0) return null;
  const vBits = (b[o + 1] >> 3) & 3;
  const lBits = (b[o + 1] >> 1) & 3;
  const brIdx = b[o + 2] >> 4;
  const srIdx = (b[o + 2] >> 2) & 3;
  const pad = (b[o + 2] >> 1) & 1;
  const mode = b[o + 3] >> 6;
  if (vBits === 1 || lBits === 0 || brIdx === 0 || brIdx === 15 || srIdx === 3) return null;
  const version = vBits === 3 ? 1 : vBits === 2 ? 2 : 25;
  const layer = (4 - lBits) as 1 | 2 | 3;
  const bitrate = BITRATES[`${version === 1 ? 1 : 2}-${layer}`][brIdx];
  const sampleRate = RATES[version][srIdx];
  let length: number;
  let samples: number;
  if (layer === 1) {
    length = (Math.floor((12 * bitrate * 1000) / sampleRate) + pad) * 4;
    samples = 384;
  } else if (layer === 2 || version === 1) {
    length = Math.floor((144 * bitrate * 1000) / sampleRate) + pad;
    samples = 1152;
  } else {
    length = Math.floor((72 * bitrate * 1000) / sampleRate) + pad;
    samples = 576;
  }
  return { version, layer, bitrate, sampleRate, channels: mode === 3 ? 1 : 2, length, samples };
}

function id3v2Size(b: Uint8Array): number {
  if (b.length >= 10 && b[0] === 0x49 && b[1] === 0x44 && b[2] === 0x33) {
    const size = ((b[6] & 0x7f) << 21) | ((b[7] & 0x7f) << 14) | ((b[8] & 0x7f) << 7) | (b[9] & 0x7f);
    return 10 + size + (b[5] & 0x10 ? 10 : 0);
  }
  return 0;
}

interface Probe {
  info: AudioInfo;
  start: number;
}

async function probe(blob: Blob): Promise<Probe> {
  let start = 0;
  // Skip any number of ID3v2 tags.
  for (;;) {
    const h = await readBytes(blob, start, 10);
    const s = id3v2Size(h);
    if (!s) break;
    start += s;
  }
  const buf = await readBytes(blob, start, 256 * 1024);
  for (let o = 0; o + 4 < buf.length; o++) {
    const h = parseFrameHeader(buf, o);
    if (!h) continue;
    // Confirm with the next two frames to avoid false syncs.
    const h2 = parseFrameHeader(buf, o + h.length);
    if (!h2 || h2.sampleRate !== h.sampleRate || h2.layer !== h.layer) continue;
    const h3 = parseFrameHeader(buf, o + h.length + h2.length);
    if (o + h.length + h2.length + 4 <= buf.length && (!h3 || h3.sampleRate !== h.sampleRate)) continue;
    const side = h.version === 1 ? (h.channels === 1 ? 17 : 32) : h.channels === 1 ? 9 : 17;
    let frames: number | null = null;
    const x = o + 4 + side;
    const tag = String.fromCharCode(...buf.subarray(x, x + 4));
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    if ((tag === "Xing" || tag === "Info") && x + 12 <= buf.length) {
      const flags = dv.getUint32(x + 4);
      if (flags & 1) frames = dv.getUint32(x + 8);
    } else if (String.fromCharCode(...buf.subarray(o + 36, o + 40)) === "VBRI" && o + 54 <= buf.length) {
      frames = dv.getUint32(o + 50);
    }
    let duration: number;
    let estimate = false;
    if (frames) duration = (frames * h.samples) / h.sampleRate;
    else {
      duration = ((blob.size - start - o) * 8) / (h.bitrate * 1000);
      estimate = true;
    }
    return {
      start: start + o,
      info: {
        format: "mp3",
        sampleRate: h.sampleRate,
        channels: h.channels,
        duration,
        durationIsEstimate: estimate,
        // A Xing/Info tag sits in a frame of its own whose bitrate can differ
        // from the audio frames, so report the next frame's.
        detail: tag === "Xing" || (frames && tag !== "Info") ? "VBR" : `${(tag === "Info" ? h2 : h).bitrate} kbps`,
      },
    };
  }
  throw new UnsupportedFileError("This doesn’t look like an MP3 file.");
}

export async function openMp3(blob: Blob): Promise<AudioSource> {
  const { info, start } = await probe(blob);
  async function* chunks(): AsyncGenerator<Planar> {
    const dec = new MPEGDecoder();
    await dec.ready;
    try {
      const step = 1 << 20;
      for (let o = start; o < blob.size; o += step) {
        const bytes = await readBytes(blob, o, step);
        const r = dec.decode(bytes);
        if (r.samplesDecoded > 0) yield mapChannels(r.channelData, r.samplesDecoded, info.channels);
      }
    } finally {
      dec.free();
    }
  }
  return { info, chunks };
}

function mapChannels(data: Float32Array[], n: number, channels: number): Planar {
  const out: Planar = [];
  for (let c = 0; c < channels; c++) out.push((data[c] ?? data[0]).slice(0, n));
  return out;
}

export async function openAudio(blob: Blob, name: string): Promise<AudioSource> {
  const head = await readBytes(blob, 0, 12);
  const s = String.fromCharCode(...head.subarray(0, 4));
  if (s === "RIFF" || s === "RF64" || s === "BW64") {
    const { openWav } = await import("./wav.ts");
    return openWav(blob);
  }
  if (/\.wav$/i.test(name)) throw new UnsupportedFileError("This WAV file is damaged or isn’t really a WAV file.");
  return openMp3(blob);
}
