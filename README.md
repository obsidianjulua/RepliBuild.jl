# RepliBuild.jl

**Automatic FFI generation using DWARF debug information**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/julia-%3E%3D1.9-blue)](https://julialang.org/)

**Automatic FFI generation using DWARF debug information**

RepliBuild generates Julia bindings for C/C++ libraries by extracting type information from DWARF debug data produced during compilation.
**What it does:** Extracts types from compiled binaries, generates direct ccall wrappers.
**What it doesn't do:** Virtual methods, inheritance, STL containers, non-standard-layout types.

```julia
using RepliBuild

# Use terminal from project root

cd("*")

# First time? Create config
discover()  # Auto-generates replibuild.toml

# If toml already exist
replibuild.toml already exists!
   Use discover(force=true) to regenerate

# 1. Compile C++ → library
build("replibuild.toml")

# 2. Generate Julia wrapper
wrap()

# 3. Use your C++ functions from Julia!
```

## Supported Types

- [x] Base types (int, double, bool, char, etc.)
- [x] Pointers (T*)
- [x] Const types (const T, const T*)
- [x] References (T&)
- [x] Struct names (detection and typing)
- [x] **Struct members** (automatic extraction from DWARF)
- [x] Return types
- [x] Function parameters
- [x] Classes (including templates) - detection works, members TODO
- [x] Enums
- [x] Arrays
- [x] Function pointers
- [ ] STL containers (Later version with MLIR)

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
build("replibuild.toml")

# Generate type-safe bindings
wrap()
```

### 4. Use in Julia

```julia
include("julia/Mathlib.jl")
using ..Mathlib

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

---

## Requirements

- Julia ≥ 1.9
- clang/LLVM toolchain

Works on:
-  Linux

---

## License

MIT - See [LICENSE](LICENSE) for details.

---

## Citation

If you use RepliBuild in research, please cite:

```bibtex
@software{replibuild2025,
  title = {RepliBuild: Julia FFI tools},
  author = {[Jonathon Mohr/obsidianjulua]},
  year = {2025},
  url = {https://github.com/obsidianjulua/RepliBuild.jl}
}
```

---

## Acknowledgments

- DragonFFI for pioneering DWARF + IR approach for C

---
