# JLCS MLIR Dialect - Production Source

This directory contains the production MLIR dialect source code for RepliBuild.jl.

## Quick Build

```bash
./build.sh
```

This will:
1. Check for MLIR/LLVM installation
2. Run TableGen to generate C++ from `.td` files
3. Compile to `build/libJLCS.so`

## Directory Structure

```
src/mlir/
├── build.sh              # Production build script
├── CMakeLists.txt        # CMake configuration
│
├── JLCS.td              # Main dialect definition (TableGen)
├── JLInterfaces.td      # Interface definitions
├── OpBase.td            # Base operation definitions
├── Types.td             # Type definitions
│
├── IR/                  # C++ headers
│   ├── JLCSDialect.h   # Dialect class
│   ├── JLCSOps.h       # Operations
│   ├── JLCSTypes.h     # Types
│   └── JLCSLoweringUtils.h  # Helper functions
│
└── impl/                # C++ implementations
    ├── JLCSDialect.cpp # Dialect registration
    ├── JLCSOps.cpp     # Operation implementations
    ├── JLCSTypes.cpp   # Type storage
    ├── JLCSPasses.cpp  # LLVM lowering pass
    └── JLCSCHelpers.cpp # C API for Julia
```

## Testing

From the repository root:

```bash
julia --project=. -e 'include("src/MLIRNative.jl"); using .MLIRNative; test_dialect()'
```

## Documentation

Full documentation is available in `docs/mlir/`:

- **[README.md](../../docs/mlir/README.md)** - Complete guide for Julia developers
- **[TABLEGEN_GUIDE.md](../../docs/mlir/TABLEGEN_GUIDE.md)** - TableGen language reference
- **[EXAMPLES.md](../../docs/mlir/EXAMPLES.md)** - Practical usage examples

## Integration with RepliBuild

This dialect is integrated into the RepliBuild toolchain for:

- **Virtual method dispatch** (C++ vtables)
- **Complex struct layouts** (field access by offset)
- **Strided arrays** (cross-language arrays)
- **JIT compilation** (runtime code generation)

See [RepliBuild MLIR Integration](../../docs/mlir/README.md#integration-with-replibuild) for details.

## Requirements

- LLVM/MLIR 18+ (system installation)
- CMake 3.20+
- C++17 compiler

## Build Options

```bash
# Debug build
BUILD_TYPE=Debug ./build.sh

# Release build (default)
BUILD_TYPE=Release ./build.sh
```

## Troubleshooting

**Library not found after build:**
```bash
# Check if library exists
ls -lh build/libJLCS.so

# Check symbols
nm -D build/libJLCS.so | grep registerJLCSDialect
```

**MLIR not found:**
```bash
# Check installation
which mlir-tblgen
llvm-config --version

# Set paths manually if needed
export MLIR_DIR=/usr/lib/cmake/mlir
export LLVM_DIR=$(llvm-config --cmakedir)
```

## Development Workflow

1. Edit TableGen definitions (`.td` files)
2. Run `./build.sh` to rebuild
3. Test with Julia: `include("src/MLIRNative.jl")`
4. See changes in generated `.inc` files in `build/`

## Related Files

- [src/MLIRNative.jl](../MLIRNative.jl) - Julia bindings
- [src/JLCSIRGenerator.jl](../JLCSIRGenerator.jl) - IR generation utilities
- [test/mlir/](../../test/mlir/) - Test suite (if exists)
