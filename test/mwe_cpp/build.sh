#!/usr/bin/env bash
# Build the pure-C++ MWE against system MLIR/LLVM (no JLCS, no RepliBuild).
set -euo pipefail

cd "$(dirname "$0")"
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
echo
echo "built: $(pwd)/build/mwe"
