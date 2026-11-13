/**
 * Tone matching: gives a restored recording the long-term spectral balance of
 * a reference (the recording with the app's noise reduction), using a smooth,
 * linear-phase EQ. Generative vocoders tend to shift the balance (VoiceFixer
 * comes out thin: less low-mid body, more treble); this puts the speaker's own
 * timbre back without touching the restoration's detail.
 */

/** In-place iterative radix-2 complex FFT (n a power of two). `inverse` scales by 1/n. */
export function fft(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) (re[i] /= n), (im[i] /= n);
}

const N = 4096; // analysis frame
const HOP = 1024;

/** Mean power spectrum (N/2+1 bins) over the given frames. */
function averageSpectrum(x: Float32Array, frames: number[]): Float64Array {
  const out = new Float64Array(N / 2 + 1);
  const re = new Float64Array(N), im = new Float64Array(N);
  for (const f of frames) {
    for (let i = 0; i < N; i++) {
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
      re[i] = (x[f * HOP + i] ?? 0) * w;
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k <= N / 2; k++) out[k] += re[k] * re[k] + im[k] * im[k];
  }
  for (let k = 0; k <= N / 2; k++) out[k] /= Math.max(1, frames.length);
  return out;
}

export interface ToneMatchOptions {
  /** Largest boost or cut, dB. */
  maxDb?: number;
  /** Correction is held flat outside this range, Hz. */
  lowHz?: number;
  highHz?: number;
}

/** The EQ curve (dB per bin of an N-point FFT) that turns `y`'s balance into `ref`'s. */
export function toneCurve(y: Float32Array, ref: Float32Array, sr: number, o: ToneMatchOptions = {}): Float64Array {
  const maxDb = o.maxDb ?? 9, lowHz = o.lowHz ?? 150, highHz = o.highHz ?? 16000;
  // Speech frames: within 30 dB of the loudest frame of the reference.
  const count = Math.max(0, Math.floor((Math.min(y.length, ref.length) - N) / HOP) + 1);
  const energy = new Float64Array(count);
  for (let f = 0; f < count; f++) {
    let e = 0;
    for (let i = 0; i < N; i++) e += ref[f * HOP + i] ** 2;
    energy[f] = 10 * Math.log10(e / N + 1e-12);
  }
  const top = Math.max(...energy);
  const frames = [...energy.keys()].filter((f) => energy[f] > top - 30);
  const py = averageSpectrum(y, frames), pr = averageSpectrum(ref, frames);
  const bins = N / 2 + 1;
  const hz = (k: number) => (k * sr) / N;
  // Ratio smoothed over a third of an octave, in dB.
  const curve = new Float64Array(bins);
  for (let k = 1; k < bins; k++) {
    const f = Math.min(Math.max(hz(k), lowHz), highHz);
    const a = f * Math.pow(2, -1 / 6), b = f * Math.pow(2, 1 / 6);
    let sy = 0, sref = 0;
    for (let j = Math.max(1, Math.floor((a * N) / sr)); j <= Math.min(bins - 1, Math.ceil((b * N) / sr)); j++) {
      sy += py[j];
      sref += pr[j];
    }
    curve[k] = 10 * Math.log10((sref + 1e-20) / (sy + 1e-20));
  }
  curve[0] = curve[1];
  // Level is matched separately; keep 500 Hz–2 kHz at 0 dB so only the balance changes.
  let mid = 0, n = 0;
  for (let k = 0; k < bins; k++) if (hz(k) >= 500 && hz(k) < 2000) (mid += curve[k]), n++;
  mid /= n;
  for (let k = 0; k < bins; k++) curve[k] = Math.max(-maxDb, Math.min(maxDb, curve[k] - mid));
  return curve;
}

/** Applies an EQ curve (dB per bin of an N-point FFT) as a zero-delay linear-phase FIR. */
export function applyCurve(x: Float32Array, curve: Float64Array): Float32Array {
  // FIR of N taps from the zero-phase response, Hann-windowed.
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let k = 0; k <= N / 2; k++) {
    re[k] = Math.pow(10, curve[k] / 20);
    if (k > 0 && k < N / 2) re[N - k] = re[k];
  }
  fft(re, im, true);
  const taps = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const j = (i + N / 2) % N; // centre the impulse
    taps[i] = re[j] * (0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 0.5)) / N));
  }
  // Overlap-add convolution with FFTs of 2N; output delayed by N/2, removed below.
  const M = 2 * N, B = M - N + 1;
  const hr = new Float64Array(M), hi = new Float64Array(M);
  hr.set(taps);
  fft(hr, hi);
  const out = new Float64Array(x.length + M);
  const br = new Float64Array(M), bi = new Float64Array(M);
  for (let s = 0; s < x.length; s += B) {
    br.fill(0);
    bi.fill(0);
    for (let i = 0; i < B && s + i < x.length; i++) br[i] = x[s + i];
    fft(br, bi);
    for (let k = 0; k < M; k++) {
      const r = br[k] * hr[k] - bi[k] * hi[k];
      bi[k] = br[k] * hi[k] + bi[k] * hr[k];
      br[k] = r;
    }
    fft(br, bi, true);
    for (let i = 0; i < M; i++) out[s + i] += br[i];
  }
  return Float32Array.from(out.subarray(N / 2, N / 2 + x.length));
}

export function toneMatch(y: Float32Array, ref: Float32Array, sr: number, o?: ToneMatchOptions): Float32Array {
  return applyCurve(y, toneCurve(y, ref, sr, o));
}
