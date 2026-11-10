/** Planar audio chunk. */
export type Planar = Float32Array[];

export interface AudioInfo {
  format: "wav" | "mp3";
  sampleRate: number;
  channels: number;
  /** Duration in seconds. May be an estimate for MP3 without a VBR header. */
  duration: number;
  durationIsEstimate: boolean;
  /** Bits per sample (WAV) or bitrate in kbps (MP3), for display. */
  detail: string;
}

export interface AudioSource {
  info: AudioInfo;
  /** Yields planar chunks at the source's sample rate and channel count. */
  chunks(): AsyncGenerator<Planar>;
}

export class UnsupportedFileError extends Error {}

export async function readBytes(blob: Blob, offset: number, length: number): Promise<Uint8Array> {
  const end = Math.min(blob.size, offset + length);
  if (end <= offset) return new Uint8Array(0);
  return new Uint8Array(await blob.slice(offset, end).arrayBuffer());
}
