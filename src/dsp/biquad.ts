/** Second-order IIR section, transposed direct form II, double-precision state. */
export class Biquad {
  private z1 = 0;
  private z2 = 0;
  constructor(
    readonly b0: number,
    readonly b1: number,
    readonly b2: number,
    readonly a1: number,
    readonly a2: number,
  ) {}

  /** Filters `x` into `y` (may be the same array). */
  process(x: Float32Array, y: Float32Array = x): void {
    const { b0, b1, b2, a1, a2 } = this;
    let z1 = this.z1;
    let z2 = this.z2;
    for (let i = 0; i < x.length; i++) {
      const xi = x[i];
      const yi = b0 * xi + z1;
      z1 = b1 * xi - a1 * yi + z2;
      z2 = b2 * xi - a2 * yi;
      y[i] = yi;
    }
    this.z1 = z1;
    this.z2 = z2;
  }

  /** Like `process` but writes into a Float64Array (for measurement paths). */
  processTo64(x: Float32Array, y: Float64Array): void {
    const { b0, b1, b2, a1, a2 } = this;
    let z1 = this.z1;
    let z2 = this.z2;
    for (let i = 0; i < x.length; i++) {
      const xi = x[i];
      const yi = b0 * xi + z1;
      z1 = b1 * xi - a1 * yi + z2;
      z2 = b2 * xi - a2 * yi;
      y[i] = yi;
    }
    this.z1 = z1;
    this.z2 = z2;
  }

  process64(x: Float64Array): void {
    const { b0, b1, b2, a1, a2 } = this;
    let z1 = this.z1;
    let z2 = this.z2;
    for (let i = 0; i < x.length; i++) {
      const xi = x[i];
      const yi = b0 * xi + z1;
      z1 = b1 * xi - a1 * yi + z2;
      z2 = b2 * xi - a2 * yi;
      x[i] = yi;
    }
    this.z1 = z1;
    this.z2 = z2;
  }

  reset(): void {
    this.z1 = 0;
    this.z2 = 0;
  }

  clone(): Biquad {
    return new Biquad(this.b0, this.b1, this.b2, this.a1, this.a2);
  }

  /** Magnitude response at frequency `f` (Hz) for sample rate `fs`. */
  magnitudeAt(f: number, fs: number): number {
    const w = (2 * Math.PI * f) / fs;
    const cr = Math.cos(w), ci = -Math.sin(w);
    const c2r = Math.cos(2 * w), c2i = -Math.sin(2 * w);
    const nr = this.b0 + this.b1 * cr + this.b2 * c2r;
    const ni = this.b1 * ci + this.b2 * c2i;
    const dr = 1 + this.a1 * cr + this.a2 * c2r;
    const di = this.a1 * ci + this.a2 * c2i;
    return Math.hypot(nr, ni) / Math.hypot(dr, di);
  }
}

function fromAnalog(b0: number, b1: number, b2: number, a0: number, a1: number, a2: number): Biquad {
  return new Biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

/** RBJ cookbook high-pass. */
export function highPass(f0: number, q: number, fs: number): Biquad {
  const w0 = (2 * Math.PI * f0) / fs;
  const cos = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  return fromAnalog((1 + cos) / 2, -(1 + cos), (1 + cos) / 2, 1 + alpha, -2 * cos, 1 - alpha);
}

/** RBJ cookbook low-pass. */
export function lowPass(f0: number, q: number, fs: number): Biquad {
  const w0 = (2 * Math.PI * f0) / fs;
  const cos = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  return fromAnalog((1 - cos) / 2, 1 - cos, (1 - cos) / 2, 1 + alpha, -2 * cos, 1 - alpha);
}

/** Q values for the second-order sections of an even-order Butterworth filter. */
export function butterworthQs(order: number): number[] {
  const qs: number[] = [];
  for (let k = 0; k < order / 2; k++) {
    qs.push(1 / (2 * Math.cos((Math.PI * (2 * k + 1)) / (2 * order))));
  }
  return qs;
}

/** A cascade of biquads applied in series. */
export class Cascade {
  constructor(readonly stages: Biquad[]) {}
  process(x: Float32Array, y: Float32Array = x): void {
    this.stages[0].process(x, y);
    for (let i = 1; i < this.stages.length; i++) this.stages[i].process(y, y);
  }
  process64(x: Float64Array): void {
    for (const s of this.stages) s.process64(x);
  }
  clone(): Cascade {
    return new Cascade(this.stages.map((s) => s.clone()));
  }
  magnitudeAt(f: number, fs: number): number {
    return this.stages.reduce((m, s) => m * s.magnitudeAt(f, fs), 1);
  }
}

export function butterworthHighPass(order: number, f0: number, fs: number): Cascade {
  return new Cascade(butterworthQs(order).map((q) => highPass(f0, q, fs)));
}

/**
 * Linkwitz–Riley 4th-order crossover. low + high sums to an all-pass response,
 * so splitting, applying gain to one band and summing leaves the signal intact
 * apart from phase when the gain is 1.
 */
export class LR4Crossover {
  private lp: Cascade;
  private hp: Cascade;
  constructor(f0: number, fs: number) {
    const q = Math.SQRT1_2;
    this.lp = new Cascade([lowPass(f0, q, fs), lowPass(f0, q, fs)]);
    this.hp = new Cascade([highPass(f0, q, fs), highPass(f0, q, fs)]);
  }
  split(x: Float32Array, low: Float32Array, high: Float32Array): void {
    this.lp.process(x, low);
    this.hp.process(x, high);
  }
}
