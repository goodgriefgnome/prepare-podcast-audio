/**
 * The w2v-BERT 2.0 input features (Hugging Face SeamlessM4TFeatureExtractor,
 * as Sidon uses it): Kaldi-style 80-band log mel filterbank on 16 kHz audio,
 * each band normalised over the clip, then pairs of frames stacked into 160 values.
 */

const FRAME = 400, HOP = 160, NFFT = 512, MELS = 80, SR = 16000;
const PREEMPH = 0.97, FLOOR = 1.192092955078125e-7;

let tables: { window: Float64Array; fb: Float64Array[]; cos: Float64Array; sin: Float64Array; rev: Uint32Array } | null = null;

function build() {
  // Povey window: Hann (symmetric) to the power 0.85.
  const window = new Float64Array(FRAME);
  for (let i = 0; i < FRAME; i++) window[i] = Math.pow(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (FRAME - 1)), 0.85);
  // Kaldi mel scale, triangles drawn in mel space between 20 Hz and 8 kHz.
  const mel = (f: number) => 1127 * Math.log(1 + f / 700);
  const bins = NFFT / 2 + 1;
  const lo = mel(20), hi = mel(8000);
  const edges = Array.from({ length: MELS + 2 }, (_, i) => lo + ((hi - lo) * i) / (MELS + 1));
  const fb: Float64Array[] = [];
  for (let m = 0; m < MELS; m++) {
    const row = new Float64Array(bins);
    for (let k = 0; k < bins; k++) {
      const f = mel((k * (SR / 2)) / (bins - 1));
      const down = (f - edges[m]) / (edges[m + 1] - edges[m]);
      const up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
      row[k] = Math.max(0, Math.min(down, up));
    }
    fb.push(row);
  }
  const cos = new Float64Array(NFFT / 2), sin = new Float64Array(NFFT / 2);
  for (let i = 0; i < NFFT / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / NFFT);
    sin[i] = -Math.sin((2 * Math.PI * i) / NFFT);
  }
  const rev = new Uint32Array(NFFT);
  const bits = Math.log2(NFFT);
  for (let i = 0; i < NFFT; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  return { window, fb, cos, sin, rev };
}

/** In-place radix-2 FFT of length NFFT. */
function fft(re: Float64Array, im: Float64Array) {
  const { cos, sin, rev } = tables!;
  for (let i = 0; i < NFFT; i++) {
    const j = rev[i];
    if (j > i) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let size = 2; size <= NFFT; size *= 2) {
    const half = size / 2, step = NFFT / size;
    for (let s = 0; s < NFFT; s += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step], wi = sin[k * step];
        const a = s + k, b = a + half;
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

/** 16 kHz audio (−1…1) to input_features: (frames/2, 160), row-major. */
export function w2vBertFeatures(x: Float32Array): { data: Float32Array; frames: number } {
  tables ??= build();
  const { window, fb } = tables;
  const n = x.length < FRAME ? 0 : 1 + Math.floor((x.length - FRAME) / HOP);
  const feats = new Float64Array(n * MELS);
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT), frame = new Float64Array(FRAME);
  const power = new Float64Array(NFFT / 2 + 1);
  for (let t = 0; t < n; t++) {
    let mean = 0;
    for (let i = 0; i < FRAME; i++) {
      frame[i] = x[t * HOP + i] * 32768;
      mean += frame[i];
    }
    mean /= FRAME;
    for (let i = 0; i < FRAME; i++) frame[i] -= mean;
    for (let i = FRAME - 1; i > 0; i--) frame[i] -= PREEMPH * frame[i - 1];
    frame[0] *= 1 - PREEMPH;
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < FRAME; i++) re[i] = frame[i] * window[i];
    fft(re, im);
    for (let k = 0; k <= NFFT / 2; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < MELS; m++) {
      const row = fb[m];
      let s = 0;
      for (let k = 0; k <= NFFT / 2; k++) s += row[k] * power[k];
      feats[t * MELS + m] = Math.log(Math.max(FLOOR, s));
    }
  }
  // Normalise each band over the clip (sample variance).
  for (let m = 0; m < MELS; m++) {
    let mean = 0;
    for (let t = 0; t < n; t++) mean += feats[t * MELS + m];
    mean /= n;
    let v = 0;
    for (let t = 0; t < n; t++) v += (feats[t * MELS + m] - mean) ** 2;
    v /= Math.max(1, n - 1);
    const sd = Math.sqrt(v + 1e-7);
    for (let t = 0; t < n; t++) feats[t * MELS + m] = (feats[t * MELS + m] - mean) / sd;
  }
  // Pad to an even number of frames (padding value 1), then stack pairs.
  const frames = Math.ceil(n / 2);
  const data = new Float32Array(frames * 2 * MELS).fill(1);
  for (let i = 0; i < n * MELS; i++) data[i] = feats[i];
  return { data, frames };
}
