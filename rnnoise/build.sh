#!/bin/bash
# Builds RNNoise as a standalone WebAssembly module (no JavaScript glue).
#
# usage: rnnoise/build.sh <out-dir> [variant ...]
#
# Needs Emscripten $EMSCRIPTEN_VERSION (emcc on PATH), git, wget and sha256sum.
# The variant "release" is what the app ships as src/vendor/rnnoise.wasm;
# the others exist to compare speed.
set -euo pipefail

EMSCRIPTEN_VERSION=4.0.8
RNNOISE_REPOSITORY=https://github.com/xiph/rnnoise
# Same commit as @shiguredo/rnnoise-wasm 2025.1.5, so the model is unchanged.
RNNOISE_COMMIT=70f1d256acd4b34a572f999a05c87bf00b67730d

out_dir=$(realpath -m "$1")
shift
variants=("${@:-release}")

found=$(emcc --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
if [ "$found" != "$EMSCRIPTEN_VERSION" ]; then
  echo "Need Emscripten $EMSCRIPTEN_VERSION, found $found" >&2
  exit 1
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git clone --quiet "$RNNOISE_REPOSITORY" "$work/rnnoise"
cd "$work/rnnoise"
git checkout --quiet "$RNNOISE_COMMIT"
# Fetches the model weights (src/rnnoise_data.c) and checks their hash.
./download_model.sh > /dev/null

sources=(
  src/denoise.c src/rnn.c src/pitch.c src/kiss_fft.c src/celt_lpc.c
  src/nnet.c src/nnet_default.c src/parse_lpcnet_weights.c
  src/rnnoise_data.c src/rnnoise_tables.c
)
exports=_rnnoise_create,_rnnoise_destroy,_rnnoise_process_frame,_rnnoise_get_frame_size,_malloc,_free

flags_for() {
  case "$1" in
    o2) echo "-O2" ;;                                    # like @shiguredo/rnnoise-wasm
    o3) echo "-O3" ;;
    simd) echo "-O3 -msimd128" ;;                        # compiler auto-vectorization only
    # RNNoise's hand-written x86 code, which Emscripten maps to Wasm SIMD.
    # (Its SSE-only path still needs the AVX types, so -mavx is the minimum.)
    avx) echo "-O3 -msimd128 -mavx" ;;
    avx2) echo "-O3 -msimd128 -mavx2" ;;
    release) echo "-O3 -msimd128 -mavx2" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}

mkdir -p "$out_dir"
for v in "${variants[@]}"; do
  flags=$(flags_for "$v")
  echo "== $v: $flags"
  # shellcheck disable=SC2086
  # DISABLE_DEBUG_FLOAT is what RNNoise's configure sets by default: use
  # the quantized weights only, and leave out the float copies.
  if ! emcc $flags -g0 -DDISABLE_DEBUG_FLOAT -Iinclude -Isrc "${sources[@]}" \
    -sSTANDALONE_WASM=1 --no-entry \
    -sEXPORTED_FUNCTIONS="$exports" \
    -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=256KB -sFILESYSTEM=0 \
    -o "$out_dir/rnnoise-$v.wasm"; then
    [ "$v" = release ] && exit 1
    echo "variant $v failed to build" >&2
    continue
  fi
  ls -l "$out_dir/rnnoise-$v.wasm"
done
cp COPYING "$out_dir/RNNOISE-COPYING"
