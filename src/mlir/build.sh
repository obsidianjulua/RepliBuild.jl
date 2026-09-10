#!/bin/bash
# Production build script for JLCS MLIR Dialect
# Part of RepliBuild.jl toolchain

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"

# The LLVM floor is DERIVED, not restated. It used to be `LLVM_MIN_MAJOR=21`
# here and `const MIN_LLVM_VERSION = 21` in Julia — two copies of one fact, and
# a third implicit one in the prefix ladder, which is how that ladder ended up
# probing LLVM 20 down to 15 against a minimum of 21.
#
# No fallback literal on failure: a default here would silently become the
# second copy again the moment the grep stopped matching.
LLVM_ENV_JL="${SCRIPT_DIR}/../Builder/LLVMEnvironment.jl"
LLVM_MIN_MAJOR="$(sed -n 's/^const MIN_LLVM_VERSION = \([0-9][0-9]*\).*/\1/p' "$LLVM_ENV_JL" 2>/dev/null | head -1)"
if [ -z "$LLVM_MIN_MAJOR" ]; then
    echo "ERROR: could not read MIN_LLVM_VERSION from:"
    echo "  $LLVM_ENV_JL"
    echo
    echo "That constant is the single definition of the LLVM floor, and this"
    echo "script derives it rather than keeping a second copy. If the constant"
    echo "moved, fix the path above — do NOT hardcode the number back in."
    exit 1
fi

# One install hint, used by every failure path below. There were four before
# (two spellings of the Arch line, three of the Ubuntu line), and they
# disagreed with each other and with the Julia-side advice.
# Keep these three lines in step with `_LLVM_ADVICE` in
# src/Builder/EnvironmentDoctor.jl — test_toolchain_advice.jl asserts it.
install_hint() {
    echo "  Arch:          yay -S llvm mlir"
    echo "  Debian/Ubuntu: wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh ${LLVM_MIN_MAJOR}"
    echo "  Fedora:        dnf install llvm-devel mlir-devel clang-devel"
    echo
    echo "  Full diagnostics, including the C path and DWARF tools:"
    echo "    julia --project=\"${SCRIPT_DIR}/../..\" -e 'using RepliBuild; RepliBuild.check_environment()'"
}

echo "=============================================="
echo " Building JLCS MLIR Dialect (Production)"
echo "=============================================="

# Verify MLIR installation
echo -n "Checking for MLIR installation... "
if ! command -v mlir-tblgen &> /dev/null; then
    echo "✗"
    echo "ERROR: mlir-tblgen not found — the JLCS dialect cannot be built"
    install_hint
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

# The dialect needs LLVM/MLIR ${LLVM_MIN_MAJOR}+ (derived at the top of this
# file). Without this gate a too-old toolchain fails somewhere inside CMake or
# the TableGen run, and the error names a missing header or an unknown CMake
# target rather than the version — Debian/Ubuntu ship 14-18 as the default
# `llvm-config`, so this is the common case, not the rare one.
LLVM_MAJOR=${LLVM_VERSION%%.*}
if ! [ "$LLVM_MAJOR" -ge "$LLVM_MIN_MAJOR" ] 2>/dev/null; then
    echo "ERROR: LLVM $LLVM_VERSION is too old — the JLCS dialect needs ${LLVM_MIN_MAJOR}+"
    install_hint
    echo
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
    echo "  MLIR is packaged separately from LLVM on most distros — install it:"
    install_hint
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

# Verify build.
#
# CMake names the target by host convention: libJLCS.so on Linux, libJLCS.dylib
# on macOS, and libJLCS.dll under mingw (the `lib` prefix is kept, only the
# extension changes). Checking for ".so" unconditionally reported
# "ERROR: not found" after a build that had just linked the library correctly —
# a false failure on the one platform being brought up.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) JLCS_LIB="libJLCS.dll"   ;;
    Darwin)               JLCS_LIB="libJLCS.dylib" ;;
    *)                    JLCS_LIB="libJLCS.so"    ;;
esac

if [ -f "$JLCS_LIB" ]; then
    echo ""
    echo "=============================================="
    echo " Build Complete!"
    echo "=============================================="
    echo "Library: $(pwd)/$JLCS_LIB"
    # -L: on Linux libJLCS.so is a symlink to libJLCS.so.<soname>, and du on the
    # link itself reports 0.
    echo "Size: $(du -Lh "$JLCS_LIB" | cut -f1)"
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
    echo "ERROR: $JLCS_LIB not found after build"
    exit 1
fi
