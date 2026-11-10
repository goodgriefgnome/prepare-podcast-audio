import { type AudioSource, type AudioInfo, type Planar, UnsupportedFileError, readBytes } from "./source.ts";

const READ_SIZE = 1 << 20;

interface WavLayout {
  format: "int" | "float";
  bits: number;
  channels: number;
  sampleRate: number;
  dataOffset: number;
  dataSize: number;
}

function fourcc(v: DataView, o: number): string {
  return String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
}

/** Parses RIFF/WAVE, RF64 and BW64 headers, walking chunks through the file. */
async function parseLayout(blob: Blob): Promise<WavLayout> {
  const head = await readBytes(blob, 0, 12);
  if (head.length < 12) throw new UnsupportedFileError("This file is too short to be a WAV file.");
  const hv = new DataView(head.buffer, head.byteOffset, head.byteLength);
  const riff = fourcc(hv, 0);
  if ((riff !== "RIFF" && riff !== "RF64" && riff !== "BW64") || fourcc(hv, 8) !== "WAVE") {
    throw new UnsupportedFileError("This doesn’t look like a WAV file.");
  }
  const is64 = riff !== "RIFF";
  let ds64DataSize: number | null = null;
  let fmt: Omit<WavLayout, "dataOffset" | "dataSize"> | null = null;
  let pos = 12;
  while (pos + 8 <= blob.size) {
    const ch = await readBytes(blob, pos, 8);
    const cv = new DataView(ch.buffer, ch.byteOffset, 8);
    const id = fourcc(cv, 0);
    let size = cv.getUint32(4, true);
    const body = pos + 8;
    if (id === "ds64") {
      const b = await readBytes(blob, body, Math.min(size, 28));
      const bv = new DataView(b.buffer, b.byteOffset, b.byteLength);
      ds64DataSize = bv.getUint32(8, true) + bv.getUint32(12, true) * 2 ** 32;
    } else if (id === "fmt ") {
      const b = await readBytes(blob, body, Math.min(size, 40));
      const bv = new DataView(b.buffer, b.byteOffset, b.byteLength);
      let tag = bv.getUint16(0, true);
      const channels = bv.getUint16(2, true);
      const sampleRate = bv.getUint32(4, true);
      const bits = bv.getUint16(14, true);
      if (tag === 0xfffe && b.length >= 26) tag = bv.getUint16(24, true); // SubFormat GUID's first two bytes
      let format: "int" | "float";
      if (tag === 1 && [8, 16, 24, 32].includes(bits)) format = "int";
      else if (tag === 3 && (bits === 32 || bits === 64)) format = "float";
      else throw new UnsupportedFileError(`This WAV file uses an encoding that isn’t supported (format ${tag}, ${bits}-bit). Export it as 16- or 24-bit PCM.`);
      if (channels < 1) throw new UnsupportedFileError("This WAV file has no audio channels.");
      fmt = { format, bits, channels, sampleRate };
    } else if (id === "data") {
      if (!fmt) throw new UnsupportedFileError("This WAV file is damaged: the format header is missing.");
      if (is64 && size === 0xffffffff && ds64DataSize !== null) size = ds64DataSize;
      // Streamed or truncated files often have a wrong size; trust the file length.
      if (size === 0 || size === 0xffffffff || body + size > blob.size) size = blob.size - body;
      const frame = (fmt.bits / 8) * fmt.channels;
      return { ...fmt, dataOffset: body, dataSize: size - (size % frame) };
    }
    pos = body + size + (size & 1);
  }
  throw new UnsupportedFileError("This WAV file has no audio data.");
}

function convert(bytes: Uint8Array, l: WavLayout, out: Planar, frames: number): void {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const C = l.channels;
  const bps = l.bits / 8;
  let o = 0;
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < C; c++) {
      let s: number;
      if (l.format === "float") s = l.bits === 32 ? v.getFloat32(o, true) : v.getFloat64(o, true);
      else if (l.bits === 16) s = v.getInt16(o, true) / 32768;
      else if (l.bits === 24) {
        const b0 = bytes[o], b1 = bytes[o + 1], b2 = bytes[o + 2];
        let x = b0 | (b1 << 8) | (b2 << 16);
        if (x & 0x800000) x -= 0x1000000;
        s = x / 8388608;
      } else if (l.bits === 32) s = v.getInt32(o, true) / 2147483648;
      else s = (bytes[o] - 128) / 128;
      out[c][i] = s;
      o += bps;
    }
  }
}

export async function openWav(blob: Blob): Promise<AudioSource> {
  const l = await parseLayout(blob);
  const frameBytes = (l.bits / 8) * l.channels;
  const totalFrames = l.dataSize / frameBytes;
  const info: AudioInfo = {
    format: "wav",
    sampleRate: l.sampleRate,
    channels: l.channels,
    duration: totalFrames / l.sampleRate,
    durationIsEstimate: false,
    detail: `${l.bits}-bit${l.format === "float" ? " float" : ""}`,
  };
  async function* chunks(): AsyncGenerator<Planar> {
    const framesPerRead = Math.max(1, Math.floor(READ_SIZE / frameBytes));
    for (let f = 0; f < totalFrames; f += framesPerRead) {
      const n = Math.min(framesPerRead, totalFrames - f);
      const bytes = await readBytes(blob, l.dataOffset + f * frameBytes, n * frameBytes);
      const frames = Math.floor(bytes.length / frameBytes);
      if (frames === 0) return;
      const out = Array.from({ length: l.channels }, () => new Float32Array(frames));
      convert(bytes, l, out, frames);
      yield out;
    }
  }
  return { info, chunks };
}
