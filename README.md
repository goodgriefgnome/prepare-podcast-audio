# Prepare Podcast Audio

A web app (PWA) that takes a spoken-word podcast recording, cleans it up, sets its loudness for podcast apps, and saves an MP3. Everything runs in the browser, so nothing is uploaded, and it works offline once loaded.

**Input:** WAV (8/16/24/32-bit PCM, 32/64-bit float, WAVE_FORMAT_EXTENSIBLE, RF64/BW64) or MP3 (CBR/VBR, any MPEG version).

**Output:** joint-stereo MP3, constant bitrate, 44.1 kHz. With mono sound (the default), both channels carry the same signal at 128 kbps. With stereo sound the left–right mix is kept, at 192 kbps. Stereo sound can only be chosen when the input is stereo.

**Loudness:** −16 LUFS integrated with true peak ≤ −1 dBTP, following Apple Podcasts' recommendation and measured per ITU-R BS.1770. The app checks these numbers by decoding the finished MP3. If the MP3 misses the target, it corrects and re-encodes.

**Output level:** a slider lowers the result in 3 dB steps (0 to −12 dB). It is a plain gain applied after the peak limiter, as the last step before encoding, so a −3 dB setting gives −19 LUFS with peaks ≤ −4 dBTP.

## What it does

Each step can be turned off in the UI except the loudness and peak stages.

| Step | How |
|---|---|
| Remove rumble | 4th-order Butterworth high-pass at 80 Hz. With this off, a 20 Hz subsonic/DC filter is still applied. |
| Reduce background noise | RNNoise, one instance per channel. Reduction is limited to about 18 dB by mixing a little of the original back in, which keeps voices natural. Its voice-activity output also drives the later stages. |
| Tame sudden loud noises | Finds bursts shorter than 1 s that are more than 6 dB above the loud end of the surrounding speech (95th percentile, ±3 s), then pulls them to 3 dB above it. Uses 30 ms look-ahead. |
| Even out speakers | Leveler that measures speech level over ±1.5 s windows and moves each stretch toward the recording's median speech level (±12 dB max, boost capped in pauses, smoothed in both directions). This is followed by gentle 2.5:1 compression just above speech level. |
| Soften harsh "s" sounds | Split-band de-esser with a Linkwitz–Riley crossover at 5 kHz. It reduces the upper band only when that band dominates and is loud relative to speech: 3:1 ratio, at most 8 dB. |
| Loudness and peaks | Gain to −16 LUFS, then a true-peak limiter (4× oversampled detection, 3 ms look-ahead), then the output level setting, then LAME. |

Because the whole recording is analysed before any gain is applied, the level stages can look ahead and behind instead of reacting the way a real-time plugin does.

## Long recordings

Audio is never held in memory as a whole. The pipeline streams in 1-second chunks and keeps one intermediate file in the browser's Origin Private File System (OPFS):

1. **Clean** (one pass over the input file): decode → mix to output channels → resample to 48 kHz → high-pass → noise reduction → working file (32-bit float). Level features are collected every 5 ms on the way.
2. **Balance** (no I/O): compute the gain curves from those features, then run one read pass to measure the result's loudness.
3. **Encode:** read the working file → apply gains → resample to 44.1 kHz → loudness gain → limiter → MP3, written to OPFS.
4. **Verify:** decode the MP3 and measure loudness and true peak. If needed, adjust and re-encode.

Temporary disk use is about 690 MB per hour (mono) or 1.4 GB per hour (stereo), plus the MP3. The app checks the available storage quota before starting.

Memory use stays roughly constant regardless of length, apart from the per-5 ms analysis arrays (about 15 MB per hour).

Processing speed is around 4–5× faster than real time on a desktop browser. Noise reduction takes most of that time.

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
node test/e2e.cjs input.wav output.mp3 [stereo] [screenshotDir]
```

`dist/` is a static site and can be served from any HTTPS host. OPFS and service workers need a secure context.

## Code map

- `src/main.ts`, `index.html`, `src/style.css` — UI
- `src/worker.ts` — runs the pipeline in a dedicated worker
- `src/pipeline/process.ts` — the passes described above
- `src/pipeline/dynamics.ts` — gain-curve computation (spikes, leveler, compressor, de-esser)
- `src/pipeline/stream.ts` — streaming noise reduction and level features
- `src/dsp/` — filters, BS.1770 loudness, resampler, true-peak meter and limiter
- `src/io/` — WAV and MP3 readers, OPFS storage
