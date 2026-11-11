import { Rnnoise } from "@shiguredo/rnnoise-wasm";

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

let rnnoise: Promise<Rnnoise> | null = null;

export const rnnoiseFactory: DenoiserFactory = {
  async create() {
    rnnoise ??= Rnnoise.load();
    const r = await rnnoise;
    const st = r.createDenoiseState();
    const scaled = new Float32Array(r.frameSize);
    return {
      frameSize: r.frameSize,
      // Measured: output lags input by two frames (frame + look-ahead), 20 ms.
      delay: 2 * r.frameSize,
      process(frame: Float32Array) {
        // RNNoise expects 16-bit-scaled samples.
        for (let i = 0; i < frame.length; i++) scaled[i] = frame[i] * 32768;
        const vad = st.processFrame(scaled);
        for (let i = 0; i < frame.length; i++) frame[i] = scaled[i] / 32768;
        return vad;
      },
      destroy() {
        st.destroy();
      },
    };
  },
};
