#!/bin/bash
# Build script for JLCS MLIR Dialect

set -e  # Exit on error

echo "=============================================="
echo " Building JLCS MLIR Dialect"
echo "=============================================="

# Use system MLIR/LLVM (required for C API compatibility)
echo -n "Checking for system MLIR installation... "
if ! command -v mlir-tblgen &> /dev/null; then
    echo "✗"
    echo "ERROR: mlir-tblgen not found"
    echo "Install MLIR: yay -S mlir"
    exit 1
fi
echo "✓"

# Check if llvm-config is available
echo -n "Checking for LLVM... "
if ! command -v llvm-config &> /dev/null; then
    echo "✗"
    echo "ERROR: llvm-config not found in PATH"
    exit 1
fi
LLVM_VERSION=$(llvm-config --version)
echo "✓ (version $LLVM_VERSION)"

LLVM_DIR=$(llvm-config --cmakedir)
MLIR_DIR=$(llvm-config --prefix)/lib/cmake/mlir

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo ""
echo "Configuring CMake..."
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$LLVM_DIR" \
  -DMLIR_DIR="$MLIR_DIR"

# Build
echo ""
echo "Building dialect library..."
cmake --build . -j$(nproc)

echo ""
echo "=============================================="
echo " Build Complete!"
echo "=============================================="
echo "Library: $(pwd)/libJLCS.so"
echo ""
echo "Test from Julia:"
echo "  julia -e 'include(\"../MLIRNative.jl\"); MLIRNative.test_dialect()'"
echo "=============================================="
