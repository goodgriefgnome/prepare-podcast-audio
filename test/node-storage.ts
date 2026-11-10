import fs from "node:fs";
import path from "node:path";
import {
  HandleByteStore,
  HandleSampleStore,
  type ByteStore,
  type SampleStore,
  type Storage,
  type SyncHandle,
} from "../src/io/store.ts";

class FsHandle implements SyncHandle {
  constructor(private fd: number) {}
  read(buf: ArrayBufferView, opts: { at: number }): number {
    return fs.readSync(this.fd, buf as Uint8Array, 0, buf.byteLength, opts.at);
  }
  write(buf: ArrayBufferView, opts: { at: number }): number {
    return fs.writeSync(this.fd, buf as Uint8Array, 0, buf.byteLength, opts.at);
  }
  truncate(size: number): void {
    fs.ftruncateSync(this.fd, size);
  }
  flush(): void {
    fs.fsyncSync(this.fd);
  }
  close(): void {
    fs.closeSync(this.fd);
  }
}

/** Storage backed by a directory on disk, for running the pipeline in Node. */
export class NodeStorage implements Storage {
  private open = new Map<string, FsHandle>();
  constructor(readonly dir: string) {
    fs.mkdirSync(dir, { recursive: true });
  }
  private handle(name: string): FsHandle {
    this.closeSync(name);
    const h = new FsHandle(fs.openSync(path.join(this.dir, name), "w+"));
    this.open.set(name, h);
    return h;
  }
  private closeSync(name: string): void {
    const h = this.open.get(name);
    if (h) {
      h.close();
      this.open.delete(name);
    }
  }
  async samples(name: string, channels: number): Promise<SampleStore> {
    return new HandleSampleStore(this.handle(name), channels);
  }
  async bytes(name: string): Promise<ByteStore> {
    return new HandleByteStore(this.handle(name));
  }
  async close(name: string): Promise<void> {
    this.closeSync(name);
  }
  async blob(name: string): Promise<Blob> {
    this.closeSync(name);
    return fs.openAsBlob(path.join(this.dir, name));
  }
  async remove(name: string): Promise<void> {
    this.closeSync(name);
    fs.rmSync(path.join(this.dir, name), { force: true });
  }
}
