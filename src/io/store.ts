import type { Planar } from "./source.ts";

/** Disk-backed float sample store (interleaved Float32). */
export interface SampleStore {
  readonly channels: number;
  readonly frames: number;
  append(p: Planar): void;
  read(start: number, n: number): Planar;
}

export interface ByteStore {
  readonly size: number;
  append(b: Uint8Array): void;
}

/** Where intermediate and output files live. */
export interface Storage {
  samples(name: string, channels: number): Promise<SampleStore>;
  bytes(name: string): Promise<ByteStore>;
  /** Flushes and closes everything opened under `name`. */
  close(name: string): Promise<void>;
  /** Returns a Blob for a closed byte file. */
  blob(name: string): Promise<Blob>;
  remove(name: string): Promise<void>;
}

/** The subset of FileSystemSyncAccessHandle we use. */
export interface SyncHandle {
  read(buf: ArrayBufferView, opts: { at: number }): number;
  write(buf: ArrayBufferView, opts: { at: number }): number;
  truncate(size: number): void | Promise<void>;
  flush(): void | Promise<void>;
  close(): void | Promise<void>;
}

export class HandleSampleStore implements SampleStore {
  frames = 0;
  private scratch = new Float32Array(0);
  constructor(
    private h: SyncHandle,
    readonly channels: number,
  ) {}

  append(p: Planar): void {
    const n = p[0].length;
    const C = this.channels;
    const need = n * C;
    if (this.scratch.length < need) this.scratch = new Float32Array(need);
    const s = this.scratch;
    for (let c = 0; c < C; c++) {
      const src = p[c];
      for (let i = 0, o = c; i < n; i++, o += C) s[o] = src[i];
    }
    const view = new Uint8Array(s.buffer, 0, need * 4);
    let at = this.frames * C * 4;
    let off = 0;
    while (off < view.length) {
      const w = this.h.write(view.subarray(off), { at });
      if (w <= 0) throw new Error("Couldn’t write to temporary storage. Your device may be out of space.");
      off += w;
      at += w;
    }
    this.frames += n;
  }

  read(start: number, n: number): Planar {
    n = Math.max(0, Math.min(n, this.frames - start));
    const C = this.channels;
    const buf = new Float32Array(n * C);
    const view = new Uint8Array(buf.buffer);
    let off = 0;
    while (off < view.length) {
      const r = this.h.read(view.subarray(off), { at: start * C * 4 + off });
      if (r <= 0) break;
      off += r;
    }
    const out: Planar = Array.from({ length: C }, () => new Float32Array(n));
    for (let c = 0; c < C; c++) {
      const d = out[c];
      for (let i = 0, o = c; i < n; i++, o += C) d[i] = buf[o];
    }
    return out;
  }
}

export class HandleByteStore implements ByteStore {
  size = 0;
  constructor(private h: SyncHandle) {}
  append(b: Uint8Array): void {
    let off = 0;
    while (off < b.length) {
      const w = this.h.write(b.subarray(off), { at: this.size + off });
      if (w <= 0) throw new Error("Couldn’t write the output file. Your device may be out of space.");
      off += w;
    }
    this.size += b.length;
  }
}

/** Origin Private File System storage, for use inside a dedicated worker. */
export class OpfsStorage implements Storage {
  private handles = new Map<string, SyncHandle>();
  private constructor(private dir: FileSystemDirectoryHandle) {}

  static async open(folder: string): Promise<OpfsStorage> {
    const root = await navigator.storage.getDirectory();
    // Clear anything left behind by an earlier run that was interrupted.
    try {
      await root.removeEntry(folder, { recursive: true });
    } catch {
      /* not there */
    }
    return new OpfsStorage(await root.getDirectoryHandle(folder, { create: true }));
  }

  private async handle(name: string): Promise<SyncHandle> {
    await this.close(name);
    const fh = await this.dir.getFileHandle(name, { create: true });
    const h = (await (fh as unknown as { createSyncAccessHandle(): Promise<SyncHandle> }).createSyncAccessHandle());
    await h.truncate(0);
    this.handles.set(name, h);
    return h;
  }

  async samples(name: string, channels: number): Promise<SampleStore> {
    return new HandleSampleStore(await this.handle(name), channels);
  }

  async bytes(name: string): Promise<ByteStore> {
    return new HandleByteStore(await this.handle(name));
  }

  async close(name: string): Promise<void> {
    const h = this.handles.get(name);
    if (!h) return;
    this.handles.delete(name);
    await h.flush();
    await h.close();
  }

  async blob(name: string): Promise<Blob> {
    await this.close(name);
    const fh = await this.dir.getFileHandle(name);
    return fh.getFile();
  }

  async remove(name: string): Promise<void> {
    await this.close(name);
    try {
      await this.dir.removeEntry(name);
    } catch {
      /* already gone */
    }
  }
}
