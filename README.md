# RepliBuild.jl

**Automatic FFI generation using DWARF debug information**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/julia-%3E%3D1.9-blue)](https://julialang.org/)

---

## What is RepliBuild?

RepliBuild generates Julia bindings for C/C++ libraries by extracting type information from DWARF debug data produced during compilation. It targets **standard-layout, trivially-copyable types** that appear in DWARF DIEs.

**What it does:** Extracts types from compiled binaries, generates direct ccall wrappers.
**What it doesn't do:** Virtual methods, inheritance, STL containers, non-standard-layout types.

```julia
using RepliBuild

# 0. First time? Create config
RepliBuild.Discovery.discover()  # Auto-generates replibuild.toml

# 1. Compile C++ → library
RepliBuild.build()

# 2. Generate Julia wrapper
RepliBuild.wrap()

# 3. Use your C++ functions from Julia!
```

**Supported:**
- Standard-layout C structs
- Trivially-copyable C++ structs
- POD types
- Functions taking/returning these types

**Not supported:**
- C++ classes with virtual methods
- Inheritance
- STL containers
- Exception-throwing functions
- Types optimized out of DWARF

---

## Quick Example

### C++ Code
```cpp
struct Vector3d {
    double x, y, z;
};

Vector3d vec3_add(Vector3d a, Vector3d b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

double vec3_dot(Vector3d a, Vector3d b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
```

### Generated Julia Bindings (Automatic)
```julia
mutable struct Vector3d
    x::Cdouble
    y::Cdouble
    z::Cdouble
end

function vec3_add(a::Vector3d, b::Vector3d)::Vector3d
    ccall((:_Z8vec3_add8Vector3dS_, LIBRARY_PATH), Vector3d,
          (Vector3d, Vector3d), a, b)
end

function vec3_dot(a::Vector3d, b::Vector3d)::Cdouble
    ccall((:_Z8vec3_dot8Vector3dS_, LIBRARY_PATH), Cdouble,
          (Vector3d, Vector3d), a, b)
end
```

### Usage in Julia
```julia
using .MyLibrary

v1 = vec3_create(1.0, 2.0, 3.0)
v2 = vec3_create(4.0, 5.0, 6.0)
v_sum = vec3_add(v1, v2)        # Vector3d(5.0, 7.0, 9.0)
dot_product = vec3_dot(v1, v2)  # 32.0
```

---

## 📚 Examples & Documentation

**See real, working examples:**
- [docs/examples/](docs/examples/) - Complete examples with explanations
- [docs/examples/StructTest.jl](docs/examples/StructTest.jl) - **Real generated wrapper** you can inspect

**Quick links:**
- [01_simple_math.md](docs/examples/01_simple_math.md) - Simplest usage (5 min)
- [02_structs_and_classes.md](docs/examples/02_structs_and_classes.md) - DWARF extraction (10 min)
- [USAGE.md](USAGE.md) - Complete usage guide

---

## Why Use DWARF?

Traditional FFI tools parse C++ headers (Clang.jl) or require manual annotations (CxxWrap.jl). RepliBuild extracts type information from compiled binaries using DWARF debug data.

**Advantages:**
- No header parsing
- No build system integration
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
- Function signatures (parameters and return types)

**Partially Supported:**
- Template instantiations: Only those present in final DWARF (ODR-used)
- Classes: Detection works, but only standard-layout, no virtual methods

**Not Supported:**
- Virtual methods, vtables, inheritance
- STL containers (implementation-defined layouts)
- Types optimized out under -O2/-O3
- Exception specifications
- Function pointers with unknown calling conventions

###  Direct, Zero-Overhead Bindings

**Generated Julia code** :
```julia
# C++ struct automatically extracted
mutable struct Point
    x::Cdouble
    y::Cdouble
end

# Direct ccall - zero overhead
function create_point(arg1::Cdouble, arg2::Cdouble)::Point
    ccall((:_Z12create_pointdd, LIBRARY_PATH), Point,
          (Cdouble, Cdouble,), arg1, arg2)
end

# Usage:
p = create_point(3.0, 4.0)  #  Works, returns Point(3.0, 4.0)
```

**What you get:**
- Direct ccall to C/C++ functions
- Struct-by-value passing (for trivially-copyable types)
- Zero runtime overhead

**ABI Assumptions:**
- x86_64 Linux ABI (System V)
- Clang/GCC struct layout rules
- No padding removal under LTO
- DWARF matches actual compiled layout

### Validation

**Eigen Test:**
- 20,000+ type DIEs extracted from compiled Eigen code
- 14,769 class DIEs, 5,125 struct DIEs detected
- DWARF parsing successful

**Important:** Detection ≠ wrapping. Many Eigen types are not standard-layout and cannot be safely wrapped. The test validates DWARF parsing at scale, not FFI correctness for all types.

**Correctness boundary:** RepliBuild can extract types present in DWARF. It can only safely wrap standard-layout, trivially-copyable types.

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

## Quick Start

### 1. Create a C++ Project

```cpp
// mathlib.cpp
struct Point {
    double x;
    double y;
};

Point create_point(double x, double y) {
    return {x, y};
}

double distance(Point a, Point b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}
```

### 2. Create `replibuild.toml`

```toml
[project]
name = "mathlib"
root = "."
uuid = "your-uuid-here"

[compile]
source_files = ["mathlib.cpp"]
flags = ["-std=c++17", "-O2", "-g"]

[binary]
type = "shared"

[paths]
output = "julia"
```

### 3. Build and Generate Bindings

```julia
using RepliBuild

# Build and generate metadata
RepliBuild.build("/path/to/project")

# Generate type-safe bindings
RepliBuild.wrap("julia/libmathlib.so", tier=:introspective)
```

### 4. Use in Julia

```julia
include("julia/Mathlib.jl")
using .Mathlib

p1 = create_point(0.0, 0.0)
p2 = create_point(3.0, 4.0)
dist = distance(p1, p2)  # 5.0
```

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

**Key limitation:** Can only extract types present in final DWARF. Unused template instantiations, optimized-out members, and inlined boundaries are absent.

See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details and [LIMITATIONS.md](LIMITATIONS.md) for constraints.

---

## Comparison

| Feature | RepliBuild | Clang.jl | CxxWrap.jl |
|---------|------------|----------|------------|
| **Approach** | DWARF extraction | Header parsing | Manual wrapping |
| **Automatic** |  99% |  Partial | ❌ No |
| **Type Source** |  Compiler DWARF |  Source headers |  Manual annotations |
| **Struct Typing** |  Auto from DWARF |  Limited |  Manual |
| **Struct Members** |  Auto extracted |  Limited |  Manual |
| **Template Support** |  All instantiations | ❌ Limited |  Manual per instance |
| **User Code Required** |  minimal |  ~100s lines | ❌ ~1000s lines |
| **Handles Eigen** |  Yes (20K+ types) | ❌ Struggles |  Manual per function |
| **Zero Overhead** |  Direct ccall |  Direct ccall |  Depends on usage |

---

## Supported Types

- [x] Base types (int, double, bool, char, etc.)
- [x] Pointers (T*)
- [x] Const types (const T, const T*)
- [x] References (T&)
- [x] Struct names (detection and typing)
- [x] **Struct members** (automatic extraction from DWARF)
- [x] Return types
- [x] Function parameters
- [ ] Classes (including templates) - detection works, members TODO
- [ ] Enums (Phase 7)
- [ ] Arrays (Phase 7)
- [ ] Function pointers (Phase 7)
- [ ] STL containers (Phase 8)

---

## Requirements

- Julia ≥ 1.9
- clang/LLVM toolchain
- DWARF debug info (compile with `-g` flag)

Works on:
-  Linux (tested on Arch)
-  macOS (should work, not tested)
-  Windows (WSL recommended)

---

## Language Support

RepliBuild can extract DWARF from any LLVM-compiled language, but requires C-compatible ABI:

- **C:** Full support (C ABI is the target)
- **C++:** POD types, standard-layout structs, extern "C" functions
- **Rust:** Only `#[repr(C)]` types with `extern "C"` functions
- **Swift:** Only `@convention(c)` functions with C-compatible types
- **Fortran:** Only types with `BIND(C)` attribute

**Critical:** DWARF presence ≠ ABI compatibility. Even if the type appears in DWARF, it must follow C ABI rules to be safely wrapped.

See [LIMITATIONS.md](LIMITATIONS.md) for detailed requirements.

---

## Roadmap

### Phase 7: Enums and Arrays
- Enum support (DW_TAG_enumeration_type) - extractable
- Fixed-size arrays - can wrap as NTuple
- Function pointer support - limited (C function pointers only)

### Phase 8: Experimental STL (High Risk)
**Warning:** STL container layouts are implementation-defined and ABI-unstable.
- Vendor-specific layout extraction (libstdc++ vs libc++)
- Version-pinned wrappers
- Not recommended for production use
- Alternative: Use C-compatible interface layer

### Phase 9: Validation Tools
- DWARF-IR cross-validator (detect layout mismatches)
- ABI compatibility checker (compiler version drift)
- Runtime layout verification (sizeof assertions)

---

## Contributing

RepliBuild is pioneering DWARF-based FFI. We need help with:
- Testing on different platforms (macOS, Windows)
- Support for more complex C++ patterns
- Documentation and examples
- Performance optimization

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Performance

- **Small projects** (10 files): ~1s
- **Medium projects** (100 files): ~5s
- **Large projects** (Eigen, 20K+ types): ~20s

Generated bindings have **zero runtime overhead** - direct ccall, no wrappers.

---

## License

MIT - See [LICENSE](LICENSE) for details.

---

## Citation

If you use RepliBuild in research, please cite:

```bibtex
@software{replibuild2025,
  title = {RepliBuild: DWARF-Based Automatic FFI Generation for Julia},
  author = {[Jonathon Mohr/Obsidianjulua]},
  year = {2025},
  url = {https://github.com/[obsidianjulua]/RepliBuild.jl}
}
```

---

## Acknowledgments

- Inspired by the need for better C++/Julia interoperability
- DragonFFI for pioneering DWARF + IR approach for C
- Validates on real-world libraries (Eigen, OpenCV, etc.)
- Julia community for feedback on technical approach

---

## Technical Approach

**Novel for Julia, rare for any language:** RepliBuild is the first Julia system (and one of the first in any language) to combine three metadata sources for automatic FFI generation:

1. **DWARF debug information** (DW_TAG_* DIEs) - semantic type information from compilation
2. **LLVM IR** - canonical ABI struct layouts and calling conventions
3. **Symbol tables** (nm, readelf) - mangled/demangled function signatures

**Why this matters:** Traditional FFI tools rely on a single source:
- Header parsers (Clang.jl): Source AST only, limited by what's in headers
- Manual wrappers (CxxWrap.jl): Developer annotations, doesn't scale
- DragonFFI: DWARF + IR for C (pioneering work, but C-only)

RepliBuild extends the DWARF + IR approach to C++ with template support (instantiated templates), using three-way cross-validation for ABI correctness.

**Constraint:** Limited to types present in DWARF that are standard-layout and trivially-copyable. See [LIMITATIONS.md](LIMITATIONS.md).

---

## Contact

- GitHub Issues: Report bugs, request features
- Discussions: Ask questions, share projects
- Twitter: TODO - Announce releases, share updates

**Constraints:** See [LIMITATIONS.md](LIMITATIONS.md) for detailed correctness boundaries and rejection rules.
