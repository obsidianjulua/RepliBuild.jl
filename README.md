# RepliBuild.jl

**Automatic FFI generation using DWARF debug information**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/julia-%3E%3D1.9-blue)](https://julialang.org/)

RepliBuild generates Julia bindings for C/C++ libraries by extracting type information from DWARF debug data produced during compilation.

## Requirements

- Julia ≥ 1.9
- clang/LLVM toolchain
- Linux

---

## Installation

```julia
using Pkg
Pkg.add("RepliBuild")
```

Or from GitHub:
```julia
Pkg.add(url="https://github.com/obsidianjulua/RepliBuild.jl")
```

---

# API with params

```julia
# First run, use this command in your project root directory.
Replibuild.discover(build=true, wrap=true)

# use force to overide saftey for existing toml
Replibuild.discover(force=true, build=true, wrap=true)

# Load the module in julia and use the bindings
using .Bindings

# Clean build
RepliBuild.build(clean=true)

# Toolchain, and build stats
RepliBuild.info()

```


## Why Use DWARF?

Traditional FFI tools parse C++ headers (Clang.jl) or require manual annotations (CxxWrap.jl). RepliBuild extracts type information from compiled binaries using DWARF debug data.

**Advantages:**
- No header parsing
- build system integration(Pure Julia)
- Types as compiler sees them (post-instantiation)

**Limitations:**
- Only types present in DWARF
- Only standard-layout types supported
- Requires `-g` compilation flag
- ABI assumptions (Clang/GCC x86_64 Linux)

---

## Features

### Type Extraction from DWARF

**Supported (Working):**
- Base types: int, double, bool, char, sized integers
- Pointers: T*, const T*
- Standard-layout structs with member layout
- Function signatures (parameters and return types
- Template instantiations: Only those present in final DWARF (ODR-used)
- Classes: Detection works, but only standard-layout, no virtual methods

**In Development (MLIR Dialect):**
- Virtual methods, vtables, inheritance
- STL containers (implementation-defined layouts)
- Exception specifications
- Function pointers with unknown calling conventions

**[→ See MLIR Dialect Documentation](examples/Mlir/README.md)**

---

## MLIR Dialect for Advanced FFI

RepliBuild includes a custom MLIR dialect (JLCS) for handling complex C++ constructs that DWARF alone cannot fully represent. This enables:

- **Virtual method dispatch** through vtable analysis
- **C++ class hierarchies** with proper ABI handling
- **Strided arrays** for cross-language data sharing
- **JIT compilation** of FFI glue code

### Getting Started with MLIR

```bash
cd examples/Mlir
./build_dialect.sh

julia -e 'include("../../src/MLIRNative.jl"); using .MLIRNative; test_dialect()'
```

### Documentation

- **[Complete MLIR Guide](examples/Mlir/README.md)** - Architecture and setup for Julia developers
- **[TableGen Tutorial](examples/Mlir/TABLEGEN_GUIDE.md)** - Deep dive into MLIR's code generation DSL
- **[Practical Examples](examples/Mlir/EXAMPLES.md)** - Working code samples and patterns

---

## How It Works

```
C++ Source → clang++ -g → DWARF DIEs → Extract → Validate → Julia ccall
```

**Pipeline:**
1. Compile C++ with `-g` (generates DWARF debug info)
2. Extract type DIEs using `readelf --debug-dump=info`
3. Parse DW_TAG_structure_type, DW_TAG_subprogram, etc.
4. Cross-validate with LLVM IR for ABI layout
5. Generate Julia struct definitions and ccall wrappers

---

## License

MIT - See [LICENSE](LICENSE) for details.

---

## Citation

If you use RepliBuild in research, please cite:

```bibtex
@software{replibuild2025,
  title = {RepliBuild: Julia FFI Tools},
  author = {Jonathon Mohr},
  year = {2025},
  url = {https://github.com/obsidianjulua/RepliBuild.jl}
}
```

---

## Acknowledgments

- DragonFFI for pioneering DWARF + IR approach for C
- Julia for being the best

---
