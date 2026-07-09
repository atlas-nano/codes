#!/bin/bash
# Build the 4D-LJ MD engine (CPU; CUDA optional).  See docs/README.md.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/engine"

echo "=== building md4d (CPU, OpenMP) ==="
gcc -O3 -march=native -fopenmp -o md4d md4d.c -lm
echo "  -> $ROOT/engine/md4d"

# Optional CUDA port (NVIDIA, compute 8.9 / RTX 40-series; adjust -arch + CUDA path).
CUDA="${CUDA:-/usr/local/cuda/bin}"
if [ -x "$CUDA/nvcc" ]; then
  echo "=== building md4d_gpu (CUDA, FP64) ==="
  PATH="$CUDA:$PATH" nvcc -O3 -arch=sm_89 -o md4d_gpu md4d_gpu.cu
  echo "  -> $ROOT/engine/md4d_gpu"
else
  echo "(skipping md4d_gpu: nvcc not found at $CUDA — CPU build is enough)"
fi
echo "=== done ==="
