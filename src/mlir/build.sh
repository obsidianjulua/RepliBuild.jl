#!/bin/bash
# Production build script for JLCS MLIR Dialect
# Part of RepliBuild.jl toolchain

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"

echo "=============================================="
echo " Building JLCS MLIR Dialect (Production)"
echo "=============================================="

# Verify MLIR installation
echo -n "Checking for MLIR installation... "
if ! command -v mlir-tblgen &> /dev/null; then
    echo "✗"
    echo "ERROR: mlir-tblgen not found"
    echo "Install MLIR: yay -S mlir (Arch) or apt install mlir-21-dev (Ubuntu)"
    exit 1
fi
echo "✓"

echo -n "Checking for LLVM... "
if ! command -v llvm-config &> /dev/null; then
    echo "✗"
    echo "ERROR: llvm-config not found in PATH"
    exit 1
fi
LLVM_VERSION=$(llvm-config --version)
echo "✓ (version $LLVM_VERSION)"

# The dialect needs LLVM/MLIR 21+. Without this gate a too-old toolchain fails
# somewhere inside CMake or the TableGen run, and the error names a missing
# header or an unknown CMake target rather than the version — Debian/Ubuntu ship
# 14-18 as the default `llvm-config`, so this is the common case, not the rare one.
LLVM_MIN_MAJOR=21
LLVM_MAJOR=${LLVM_VERSION%%.*}
if ! [ "$LLVM_MAJOR" -ge "$LLVM_MIN_MAJOR" ] 2>/dev/null; then
    echo "ERROR: LLVM $LLVM_VERSION is too old — the JLCS dialect needs ${LLVM_MIN_MAJOR}+"
    echo "  Arch:   yay -S llvm mlir"
    echo "  Ubuntu: apt install llvm-${LLVM_MIN_MAJOR}-dev mlir-${LLVM_MIN_MAJOR}-dev"
    echo "  Then point this script at it, e.g.:"
    echo "    PATH=/usr/lib/llvm-${LLVM_MIN_MAJOR}/bin:\$PATH ./build.sh"
    exit 1
fi

# mlir-tblgen and llvm-config must come from the SAME install. With several LLVM
# versions side by side — the normal Debian/Ubuntu arrangement — PATH order can
# pick one of each, and the mismatch surfaces as unresolved symbols at dlopen
# time rather than as a build failure here.
MLIR_TBLGEN_VERSION=$(mlir-tblgen --version 2>/dev/null | command grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
if [ -n "$MLIR_TBLGEN_VERSION" ] && [ "${MLIR_TBLGEN_VERSION%%.*}" != "$LLVM_MAJOR" ]; then
    echo "ERROR: toolchain mismatch — llvm-config is $LLVM_VERSION but mlir-tblgen is $MLIR_TBLGEN_VERSION"
    echo "  These must come from one install. Check: which llvm-config mlir-tblgen"
    exit 1
fi

# Get LLVM/MLIR paths
LLVM_DIR=$(llvm-config --cmakedir)
MLIR_DIR=$(llvm-config --prefix)/lib/cmake/mlir

echo "LLVM CMake: $LLVM_DIR"
echo "MLIR CMake: $MLIR_DIR"

if [ ! -d "$MLIR_DIR" ]; then
    echo "ERROR: MLIR CMake package not found at $MLIR_DIR"
    echo "  llvm-config points at $(llvm-config --prefix), which has no lib/cmake/mlir."
    echo "  MLIR is packaged separately from LLVM on most distros — install it"
    echo "  (Arch: yay -S mlir, Ubuntu: apt install mlir-${LLVM_MIN_MAJOR}-dev)."
    exit 1
fi

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo ""
echo "Configuring CMake (${BUILD_TYPE} build)..."
cmake .. \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DLLVM_DIR="$LLVM_DIR" \
  -DMLIR_DIR="$MLIR_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
echo ""
echo "Building dialect library..."
cmake --build . -j$(nproc)

# Verify build
if [ -f "libJLCS.so" ]; then
    echo ""
    echo "=============================================="
    echo " Build Complete!"
    echo "=============================================="
    echo "Library: $(pwd)/libJLCS.so"
    # -L: libJLCS.so is a symlink to libJLCS.so.<soname>, and du on the link
    # itself reports 0.
    echo "Size: $(du -Lh libJLCS.so | cut -f1)"
    echo ""
    echo "To test from Julia:"
    echo "  cd $(dirname $(dirname $SCRIPT_DIR))"
    echo "  julia --project=. -e 'using RepliBuild; RepliBuild.MLIRNative.test_dialect()'"
    echo ""
    echo "Then confirm both tiers are live:"
    echo "  julia --project=. -e 'using RepliBuild; RepliBuild.check_environment()'"
    echo "=============================================="
else
    echo ""
    echo "ERROR: libJLCS.so not found after build"
    exit 1
fi
