Model files for the [Speech Restoration Lab](https://goodgriefgnome.github.io/prepare-podcast-audio/restore.html). They're too large to host on the page, so download the ones you want here, then pick them in the lab under **Models from files on your disk**. They're read from your disk in the browser and never uploaded.

| Model | Files | Size |
|---|---|---|
| Sidon | `sidon_predictor.f16.onnx`, `sidon_vocoder.f16.onnx` | 503 MB |
| PASE | `pase_encoder.f16.onnx`, `pase_vocoder.f16.onnx` | 769 MB |
| FlowSE | `flowse_mel.onnx`, `flowse_velocity.f16.onnx`, `flowse_vocoder.f16.onnx`, plus `flowse_vocab.txt` to use a transcript | 720 MB |

The files are ONNX conversions of the published weights, built by `restore-models/build.sh --release` from this repository (`.github/workflows/restore-models.yml`). Each one was checked against the original model before being stored with 16-bit weights. `SHA256SUMS` lists their hashes. See `LICENSES.md` for the original projects and their licences.
