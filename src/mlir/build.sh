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
    echo "Install MLIR: yay -S mlir (Arch) or apt install mlir-18-dev (Ubuntu)"
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

# Get LLVM/MLIR paths
LLVM_DIR=$(llvm-config --cmakedir)
MLIR_DIR=$(llvm-config --prefix)/lib/cmake/mlir

echo "LLVM CMake: $LLVM_DIR"
echo "MLIR CMake: $MLIR_DIR"

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
    echo "Size: $(du -h libJLCS.so | cut -f1)"
    echo ""
    echo "To test from Julia:"
    echo "  cd $(dirname $(dirname $SCRIPT_DIR))"
    echo "  julia --project=. -e 'include(\"src/MLIRNative.jl\"); using .MLIRNative; test_dialect()'"
    echo "=============================================="
else
    echo ""
    echo "ERROR: libJLCS.so not found after build"
    exit 1
fi
