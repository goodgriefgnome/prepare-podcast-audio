# Prepare Podcast Audio

A web app (PWA) that takes a spoken-word podcast recording, cleans it up, sets its loudness for podcast apps, and saves an MP3. Everything runs in the browser, so nothing is uploaded, and it works offline once loaded.

**Input:** WAV (8/16/24/32-bit PCM, 32/64-bit float, WAVE_FORMAT_EXTENSIBLE, RF64/BW64) or MP3 (CBR/VBR, any MPEG version).

**Output:** joint-stereo MP3, constant bitrate, 44.1 kHz. With mono sound (the default), both channels carry the same signal at 128 kbps. With stereo sound the left–right mix is kept, at 192 kbps. Stereo sound can only be chosen when the input is stereo.

**Loudness:** −16 LUFS integrated with true peak ≤ −1 dBTP, following Apple Podcasts' recommendation and measured per ITU-R BS.1770. The app checks these numbers by decoding the finished MP3. If the MP3 misses (more than 0.5 LU off, or peaks above the ceiling), the app says so; that indicates a bug, so it does not quietly re-encode.

**Output level:** a slider lowers the result in 3 dB steps (0 to −12 dB). It is a plain gain applied after the peak limiter, as the last step before encoding, so a −3 dB setting gives −19 LUFS with peaks ≤ −4 dBTP.

## What it does

Each step can be turned off in the UI except the loudness and peak stages.

| Step | How |
|---|---|
| Remove rumble | 4th-order Butterworth high-pass at 80 Hz. With this off, a 20 Hz subsonic/DC filter is still applied. |
| Reduce background noise | RNNoise (built from source with SIMD, see below), one instance per channel. Reduction is limited to about 18 dB by mixing a little of the original back in, which keeps voices natural. Its voice-activity output also drives the later stages. |
| Tame sudden loud noises | Finds bursts shorter than 1 s that are more than 6 dB above the loud end of the surrounding speech (95th percentile, ±3 s), then pulls them to 3 dB above it. Uses 30 ms look-ahead. |
| Even out speakers | Leveler that measures speech level over ±1.5 s windows and moves each stretch toward the average speech level of the surrounding minute (±12 dB max, boost capped in pauses, smoothed in both directions). This is followed by gentle 2.5:1 compression just above speech level. |
| Soften harsh "s" sounds | Split-band de-esser with a Linkwitz–Riley crossover at 5 kHz. It reduces the upper band only when that band dominates and is loud relative to speech: 3:1 ratio, at most 8 dB. |
| Loudness and peaks | Sliding-window gain to −16 LUFS, then a true-peak limiter (4× oversampled detection, 3 ms look-ahead), then the output level setting, then LAME. |

Because gains are planned with about a minute of look-ahead, the level stages can look ahead and behind instead of reacting the way a real-time plugin does.

## Using it

Everything is on one page. Settings are on the left (at the top on phones), and the files are on the right.

- **Adding files:** drop or browse for WAV and MP3 files. Each one starts processing as soon as it's added, one at a time in list order, using the current settings. Mono recordings always get mono sound. Files that can't be read are marked and skipped.
- **Listening:** each file has a Listen button that opens a player with Original and Processed. While a file is processing, Processed plays a copy of the MP3 written so far; every press of Play takes a fresh copy, so the length grows. The seek bar shades how much has been written. Asking to hear a file that hasn't started moves it to the front of the queue.
- **Changing settings:** with no files yet, changes apply straight away. Once files have been added, a change waits for **Apply and reprocess** (or **Undo changes**); until then every file keeps the previous settings.
- **Saving:** finished files show their loudness, peak and noise figures and a Save MP3 button. Processing can be paused and resumed.

## Long recordings

The whole job is a single streaming pass over the input, plus a check of the finished MP3. Audio is never held in memory as a whole, and the only file written is the MP3 (in the browser's Origin Private File System).

1. **Process:** decode → mix to output channels → resample to 48 kHz → high-pass → noise reduction → level features every 5 ms. This front end runs on 30 s blocks spread over several threads (see below). The cleaned audio then waits in a delay line of about 85 s. Gains for each 20 s stretch are worked out from the features of the ±65 s around it. The audio then has the gains applied and is resampled to 44.1 kHz, limited, set to the output level and encoded.
2. **Verify:** decode the MP3, measure loudness and true peak, and report a miss.

Because the MP3 is written as processing goes, it can be played before the file is finished. The worker hands the page a copy of the bytes written so far, and an ordinary `<audio>` element plays it.

Instead of whole-recording statistics, the level stages use a sliding window of about a minute (±30 s). This assumes a recording doesn't have minute-long stretches at very different levels:

- **Leveler:** aims each stretch of speech at the average speech level of the surrounding minute.
- **Loudness normalizer:** sets the gain so that the BS.1770 gated loudness of the surrounding minute is −16 LUFS. It allows for the de-esser's cut and the limiter. Evaluated once a second and interpolated.

### Threads

The front end runs on up to 4 extra threads (one fewer than the device's cores). The recording is split into 30 s blocks on whole-second boundaries. Each block starts 3 s early so the filters and the noise model have settled, and its first 3 s are thrown away. The processing worker takes the results back in order.

The noise model has a long memory, so a block never exactly matches one continuous run. At the joins the two differ by about −50 dB (relative to the average level), and now and then by more inside a block. Both are valid results. To keep joins seamless, each block also returns 1 s past its end, which is crossfaded into the start of the next one.

Everything after the front end stays on one thread: gains, the 44.1 kHz resampler, limiter and LAME all keep state from one sample to the next. On a 4-core desktop this sequential part sets the speed: about 11× real time for 48 kHz input.

Memory use is constant: the delay line (about 16 MB mono, 33 MB stereo), up to one 34 s block per thread in flight, plus about 30 MB per hour of per-5 ms arrays.

## Speech Restoration Lab

`restore.html` is a separate, experimental page for trying generative speech restoration on an excerpt of your own recording before anything like it goes into the app. It runs, side by side and on your device:

| Version | What it is |
|---|---|
| Original | The excerpt, mixed to mono at 44.1 kHz |
| App’s noise reduction | RNNoise with the app’s 18 dB limit (the baseline) |
| [VoiceFixer](https://github.com/haoheliu/voicefixer) | Restores a mel spectrogram (ResUNet), then a 44.1 kHz vocoder re-synthesises the voice. Mode 0 (the default). |


All versions can be loudness-matched to −16 LUFS for a fair comparison, and switched while playing.

[Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) (its enhancer: a denoiser, then latent conditional flow matching, then a UnivNet vocoder) is converted and supported by the code, but not offered on the page for now: its models are 730 MB, more than GitHub Pages should host. `restore-models/build.sh <dir> resemble` builds them; to try it, host them somewhere that allows cross-origin downloads, open the page with `?models=<url>`, and put its toggle back in `restore.html`.

[SelfRemaster](https://github.com/Takaaki-Saeki/ssl_speech_restoration_v2) isn't included: it has no general-purpose model. It learns from the degraded recordings themselves, so it would need training (on a GPU, with hours of audio) for each recording setup.

**How it runs.** The models are converted to ONNX by `restore-models/build.sh` and run with ONNX Runtime Web, on WebGPU when the browser has it and otherwise on the CPU (WebAssembly, multithreaded when the page is cross-origin isolated, which the service worker arranges). The conversion keeps the models' maths but changes how some parts are expressed so they run in a browser:

- STFTs and inverse STFTs are written as convolutions (matches the originals to 1e-8).
- Resemble Enhance's flow-matching ODE loop, noise and chunking run in JavaScript; its location-variable convolutions are rewritten as matrix multiplies (identical results), and its vocoder runs on 3 s windows with 0.3 s of context each side, because its intermediate tensors don't fit in WebAssembly's 4 GB for longer pieces.
- Weights are stored as 16-bit floats and computed in 32-bit (VoiceFixer's output changes by 0.04 dB on average; Resemble Enhance's flow model is unchanged).
- Long excerpts are processed in 30 s pieces with 1 s crossfades (both models' own segment length).

**Speed** (desktop, CPU, 4 threads, per second of audio): VoiceFixer about 1.8 s; Resemble Enhance about 13 s with default settings. WebGPU should be much faster.

**Downloads:** VoiceFixer is 216 MB, fetched on first use and kept in the browser's cache. Models are split into parts under 100 MB with an `index.json` (size and SHA-256 per model). The page looks for them in `models/` next to it, or wherever `?models=<url>` points. The deploy workflow builds VoiceFixer with `restore-models/build.sh` (cached until `restore-models/` changes) and publishes it with the site.

## Development

```bash
npm install
npm run dev        # dev server
npm test           # unit tests (Node 22+, uses ffmpeg if present for format tests)
npm run build      # production build in dist/
```

Run the full pipeline from the command line (Node), which is useful for listening tests:

```bash
node --experimental-transform-types test/run.ts input.wav output.mp3 [mono|stereo] [--off=noise,level,spikes,sibilance,rumble] [--trim=-3]
```

Drive the built app end to end in Chromium (after `npx vite preview --port 4173`):

```bash
node test/e2e.cjs input.wav[,more.mp3,...] outputDir [stereo] [screenshotDir]
```

### RNNoise

The app ships a WebAssembly build of [RNNoise](https://github.com/xiph/rnnoise) in `src/vendor/rnnoise.wasm` (BSD licence, `src/vendor/RNNOISE-COPYING`). `rnnoise/build.sh` builds it with Emscripten from a pinned RNNoise commit, using RNNoise's hand-written x86 vector code, which Emscripten translates to WebAssembly SIMD. That runs about 3× faster than a plain build, and its output differs from the plain build by about −64 dB.

- The **Build RNNoise** workflow builds it (plus slower variants for comparison) and publishes the files on the `rnnoise-build` branch. To update, change `rnnoise/build.sh`, let the workflow run, and copy `rnnoise-release.wasm` from that branch to `src/vendor/rnnoise.wasm`.
- The deploy workflow rebuilds it and refuses to deploy unless the committed file is identical.

`dist/` is a static site and can be served from any HTTPS host. OPFS and service workers need a secure context.

## Code map

- `src/main.ts`, `index.html`, `src/style.css` — the page: file list, queue, settings and player
- `src/worker.ts` — runs the pipeline in a dedicated worker
- `src/frontend-worker.ts`, `src/pipeline/pool.ts` — threads for the front end
- `src/pipeline/process.ts` — the passes described above
- `src/pipeline/dynamics.ts` — gain-curve computation (spikes, leveler, compressor, de-esser)
- `src/pipeline/frontend.ts` — the front end, run on blocks
- `src/pipeline/stream.ts` — streaming noise reduction and level features
- `src/pipeline/denoise.ts` — loads RNNoise
- `rnnoise/build.sh` — builds RNNoise
- `restore.html`, `src/restore/` — the Speech Restoration Lab
- `restore-models/` — converts VoiceFixer and Resemble Enhance to ONNX for the lab
- `src/dsp/` — filters, BS.1770 loudness, resampler, true-peak meter and limiter
- `src/io/` — WAV and MP3 readers, OPFS storage
