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
| Reduce background noise | RNNoise, one instance per channel. Reduction is limited to about 18 dB by mixing a little of the original back in, which keeps voices natural. Its voice-activity output also drives the later stages. |
| Tame sudden loud noises | Finds bursts shorter than 1 s that are more than 6 dB above the loud end of the surrounding speech (95th percentile, ±3 s), then pulls them to 3 dB above it. Uses 30 ms look-ahead. |
| Even out speakers | Leveler that measures speech level over ±1.5 s windows and moves each stretch toward the average speech level of the surrounding minute (±12 dB max, boost capped in pauses, smoothed in both directions). This is followed by gentle 2.5:1 compression just above speech level. |
| Soften harsh "s" sounds | Split-band de-esser with a Linkwitz–Riley crossover at 5 kHz. It reduces the upper band only when that band dominates and is loud relative to speech: 3:1 ratio, at most 8 dB. |
| Loudness and peaks | Sliding-window gain to −16 LUFS, then a true-peak limiter (4× oversampled detection, 3 ms look-ahead), then the output level setting, then LAME. |

Because gains are planned with about a minute of look-ahead, the level stages can look ahead and behind instead of reacting the way a real-time plugin does.

## Long recordings

The whole job is a single streaming pass over the input, plus a check of the finished MP3. Audio is never held in memory as a whole, and the only file written is the MP3 (in the browser's Origin Private File System).

1. **Process:** decode → mix to output channels → resample to 48 kHz → high-pass → noise reduction. The cleaned audio then waits in a delay line of about 85 s while level features are collected every 5 ms. Gains for each 20 s stretch are worked out from the features of the ±65 s around it. The audio then has the gains applied and is resampled to 44.1 kHz, limited, set to the output level and encoded.
2. **Verify:** decode the MP3, measure loudness and true peak, and report a miss.

Instead of whole-recording statistics, the level stages use a sliding window of about a minute (±30 s). This assumes a recording doesn't have minute-long stretches at very different levels:

- **Leveler:** aims each stretch of speech at the average speech level of the surrounding minute.
- **Loudness normalizer:** sets the gain so that the BS.1770 gated loudness of the surrounding minute is −16 LUFS. It allows for the de-esser's cut and the limiter. Evaluated once a second and interpolated.

Memory use is constant: the delay line (about 16 MB mono, 33 MB stereo) plus about 30 MB per hour of per-5 ms arrays. Processing runs about 4–5× faster than real time on a desktop; noise reduction is most of that.

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
