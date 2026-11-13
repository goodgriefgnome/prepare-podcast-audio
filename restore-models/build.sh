#!/bin/bash
# Converts speech restoration models (Resemble Enhance) to the ONNX files the restoration lab
# (restore.html) loads, split into parts under 100 MB, with an index.json the
# page reads.
#
# usage: restore-models/build.sh <out-dir> [resemble]
#
# Needs Python with requirements.txt installed, git, curl and sha256sum.
# Resemble Enhance: downloads 713 MB (Hugging Face), writes 730 MB.
set -euo pipefail

RESEMBLE_REPO=https://github.com/resemble-ai/resemble-enhance
RESEMBLE_COMMIT=8e978149bfe8abab3eb77d965d579a111afdb0ff
HF=https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2
PART_SIZE=95000000

here=$(cd "$(dirname "$0")" && pwd)
out=$(realpath -m "$1")
shift
models=("${@:-resemble}")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$out" "$work/onnx"

fetch() { # url dest sha256
  mkdir -p "$(dirname "$2")"
  curl -sSL --fail --retry 3 -o "$2" "$1"
  echo "$3  $2" | sha256sum -c --quiet
}

files=()
for model in "${models[@]}"; do
  case "$model" in
    resemble)
      echo "== Resemble Enhance"
      git clone --quiet "$RESEMBLE_REPO" "$work/resemble-enhance"
      git -C "$work/resemble-enhance" checkout --quiet "$RESEMBLE_COMMIT"
      fetch "$HF/hparams.yaml" "$work/re/hparams.yaml" 80c3f15bc5a5b2cacf2c698699a0f6599d62911c0d53e1d6dee895c0d7cbaeac
      fetch "$HF/ds/G/latest" "$work/re/ds/G/latest" 37a8eec1ce19687d132fe29051dca629d164e2c4958ba141d5f4133a33f0688f
      fetch "$HF/ds/G/default/mp_rank_00_model_states.pt" "$work/re/ds/G/default/mp_rank_00_model_states.pt" f9d035f318de3e6d919bc70cf7ad7d32b4fe92ec5cbe0b30029a27f5db07d9d6
      RE_SRC="$work/resemble-enhance" RE_RUN="$work/re" python "$here/export_re.py" "$work/onnx"
      files+=(re_front re_velocity re_decode re_vocoder)
      ;;
    *) echo "unknown model $model" >&2; exit 1 ;;
  esac
done

echo "== storing weights as float16 and splitting"
for m in "${files[@]}"; do
  python "$here/fp16.py" "$work/onnx/$m.onnx" "$work/$m.f16.onnx"
  (cd "$work" && split -b "$PART_SIZE" -d -a 2 "$m.f16.onnx" "$out/$m.f16.onnx.part")
done
cp "$here/LICENSES.md" "$out/"
python "$here/index.py" "$out"
cat "$out/index.json"
