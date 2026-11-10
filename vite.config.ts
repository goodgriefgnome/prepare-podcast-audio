import { createHash } from "node:crypto";
import { defineConfig, type Plugin } from "vite";

/**
 * Emits sw.js, a service worker that precaches every file in the build so
 * the app works offline once it has been opened.
 */
function serviceWorker(): Plugin {
  return {
    name: "precache-service-worker",
    apply: "build",
    generateBundle(_, bundle) {
      const files = Object.keys(bundle).filter((f) => !f.endsWith(".map"));
      const pub = ["manifest.webmanifest", "icon.svg", "icon-192.png", "icon-512.png", "apple-touch-icon.png"];
      const assets = ["./", ...files.map((f) => `./${f}`), ...pub.map((f) => `./${f}`)];
      const version = createHash("sha256").update(JSON.stringify(assets)).digest("hex").slice(0, 12);
      const source = `const CACHE = "ppa-${version}";
const ASSETS = ${JSON.stringify(assets)};
self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("ppa-") && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== self.location.origin) return;
  if (req.mode === "navigate") {
    e.respondWith(fetch(req).catch(() => caches.match("./", { ignoreSearch: true })));
    return;
  }
  e.respondWith(caches.match(req, { ignoreSearch: true }).then((hit) => hit || fetch(req)));
});
`;
      this.emitFile({ type: "asset", fileName: "sw.js", source });
    },
  };
}

export default defineConfig({
  base: "./",
  build: { target: "es2022", sourcemap: true, chunkSizeWarningLimit: 8000 },
  worker: { format: "es" },
  plugins: [serviceWorker()],
});
