#!/bin/bash
# Converts speech restoration models to the ONNX files the restoration lab
# (restore.html) loads.
#
# usage: restore-models/build.sh <out-dir> [model ...]   (default: all three)
#
# Models:
#   sidon     Sidon v0.1. Downloads 1.0 GB, writes 503 MB.
#   pase      PASE. Downloads 1.5 GB, writes 769 MB.
#   flowse    FlowSE, with the Vocos vocoder. Downloads 4.1 GB, writes 720 MB.
#
# The files are written whole, with SHA256SUMS, for people to download from the
# GitHub release and open in the page (releases take files up to 2 GB).
#
# Needs Python with requirements.txt installed, git, curl and sha256sum.
# Each model's downloads are deleted once it's converted, to save disk space.
set -euo pipefail

SIDON_HF=https://huggingface.co/sarulab-speech/sidon-v0.1/resolve/b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52
PASE_REPO=https://github.com/cisco-open/pase
PASE_COMMIT=45614efae73dc49bf36018d1ca6a9319b3050038
PASE_HF=https://huggingface.co/cisco-ai/pase/resolve/31b1cf14f41d4d7a1a469eaa45ce235825d900f0
FLOWSE_REPO=https://github.com/Honee-W/FlowSE
FLOWSE_COMMIT=cfb81f171d804689bf7607afd0d12a6ee89de547
FLOWSE_HF=https://huggingface.co/flowse/wenetspeech4tts_Premium.pt.tar/resolve/df1bd3f8249a51f78d2b79bb037c9bdbcf660640
VOCOS_HF=https://huggingface.co/charactr/vocos-mel-24khz/resolve/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21

here=$(cd "$(dirname "$0")" && pwd)
out=$(realpath -m "$1")
shift
models=(sidon pase flowse)
[ $# -eq 0 ] || models=("$@")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$out" "$work/onnx"

fetch() { # url dest sha256
  mkdir -p "$(dirname "$2")"
  curl -sSL --fail --retry 3 -o "$2" "$1"
  echo "$3  $2" | sha256sum -c --quiet
}

clone() { # repo commit dest
  git clone --quiet "$1" "$3"
  git -C "$3" checkout --quiet "$2"
}

files=()
for model in "${models[@]}"; do
  case "$model" in
    sidon)
      echo "== Sidon"
      fetch "$SIDON_HF/feature_extractor_cpu.pt" "$work/sidon/feature_extractor_cpu.pt" fd9abc906a9048b3c047bd3d246a25f6485d09c2d6bb85e098f527f844efc019
      fetch "$SIDON_HF/decoder_cpu.pt" "$work/sidon/decoder_cpu.pt" 34dcc80fab75bd1336369ba5b2063f6da475887aa824db94b495c4059c5fdea4
      python "$here/export_sidon.py" "$work/sidon" "$work/onnx"
      rm -rf "$work/sidon"
      files+=(sidon_predictor sidon_vocoder)
      ;;
    pase)
      echo "== PASE"
      clone "$PASE_REPO" "$PASE_COMMIT" "$work/pase-src"
      fetch "$PASE_HF/DeWavLM.pt" "$work/pase/DeWavLM.pt" e347a95ad01d15f0c3ec86804f1177e2b197e16012718f03fde152bb4e78b74f
      fetch "$PASE_HF/Vocoder_Dual.pt" "$work/pase/Vocoder_Dual.pt" 400ea906dae8d1fccfc108e316925092b60836ad6320e87e5be6a7eeb7d65875
      PASE_SRC="$work/pase-src" python "$here/export_pase.py" "$work/pase" "$work/onnx" "$work/pase-src/test/audio_noisy/fileid_1.wav"
      rm -rf "$work/pase" "$work/pase-src"
      files+=(pase_encoder pase_vocoder)
      ;;
    flowse)
      echo "== FlowSE"
      clone "$FLOWSE_REPO" "$FLOWSE_COMMIT" "$work/flowse-src"
      fetch "$FLOWSE_HF/wenetspeech4tts_Premium.pt.tar" "$work/flowse/wenetspeech4tts_Premium.pt.tar" c342175051d1ca38338993228e8a632181b9b556f9f1e310ac3e07bef54f446f
      fetch "$VOCOS_HF/config.yaml" "$work/vocos/config.yaml" da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc
      fetch "$VOCOS_HF/pytorch_model.bin" "$work/vocos/pytorch_model.bin" 97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a
      FLOWSE_SRC="$work/flowse-src" python "$here/export_flowse.py" "$work/flowse/wenetspeech4tts_Premium.pt.tar" "$work/vocos" "$work/onnx"
      cp "$work/onnx/flowse_vocab.txt" "$out/"
      rm -rf "$work/flowse" "$work/vocos" "$work/flowse-src"
      # The mel graph is small; it stays float32.
      cp "$work/onnx/flowse_mel.onnx" "$out/"
      files+=(flowse_velocity flowse_vocoder)
      ;;
    *) echo "unknown model $model" >&2; exit 1 ;;
  esac
done

echo "== storing weights as float16"
for m in "${files[@]}"; do
  python "$here/fp16.py" "$work/onnx/$m.onnx" "$work/$m.f16.onnx"
  rm "$work/onnx/$m.onnx"
  mv "$work/$m.f16.onnx" "$out/"
done
cp "$here/LICENSES.md" "$out/"
(cd "$out" && shopt -s nullglob && sha256sum -- *.onnx *.txt LICENSES.md > SHA256SUMS)
cat "$out/SHA256SUMS"
