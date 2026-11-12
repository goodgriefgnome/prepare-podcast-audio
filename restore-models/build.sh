#!/bin/bash
# Converts speech restoration models to the ONNX files the restoration lab
# (restore.html) loads, split into parts under 100 MB, with an index.json the
# page reads.
#
# usage: restore-models/build.sh <out-dir> [voicefixer] [resemble]
#   (default: voicefixer, which is all the page offers for now)
#
# Needs Python with requirements.txt installed, git, curl and sha256sum.
# VoiceFixer: downloads 625 MB (Zenodo), writes 216 MB.
# Resemble Enhance: downloads 713 MB (Hugging Face), writes 730 MB.
set -euo pipefail

RESEMBLE_REPO=https://github.com/resemble-ai/resemble-enhance
RESEMBLE_COMMIT=8e978149bfe8abab3eb77d965d579a111afdb0ff
HF=https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2
PART_SIZE=95000000

here=$(cd "$(dirname "$0")" && pwd)
out=$(realpath -m "$1")
shift
models=("${@:-voicefixer}")
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
    voicefixer)
      echo "== VoiceFixer"
      # VoiceFixer reads its checkpoints from ~/.cache/voicefixer, so give it a home of its own.
      vfhome="$work/home/.cache/voicefixer"
      fetch "https://zenodo.org/records/5600188/files/vf.ckpt?download=1" "$vfhome/analysis_module/checkpoints/vf.ckpt" 748411b70089cadf34a6c11054f95f3a454e614af562c23b13a82f6cb413109f
      fetch "https://zenodo.org/records/5469951/files/model.ckpt-1490000_trimed.pt?download=1" "$vfhome/synthesis_module/44100/model.ckpt-1490000_trimed.pt" 9410d0b528c10a251ae947bd299d1939b0b3247df680c81c4164e94f5d87dc45
      python -m pip download --quiet --no-deps -d "$work/pkg" voicefixer==0.1.3
      echo "7ee61b71c25fbdf1ef4e5407fe870ec0b351f8a82765806a8b09c37eb3a1162d  $work/pkg/voicefixer-0.1.3-py3-none-any.whl" | sha256sum -c --quiet
      python -m zipfile -e "$work/pkg/voicefixer-0.1.3-py3-none-any.whl" "$work/vfsrc"
      HOME="$work/home" VF_SRC="$work/vfsrc" python "$here/export_vf.py" "$work/onnx"
      files+=(vf)
      ;;
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
