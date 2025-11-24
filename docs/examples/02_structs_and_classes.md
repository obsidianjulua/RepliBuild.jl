# Example 2: Structs and Classes

This example demonstrates RepliBuild's **DWARF extraction** - automatically extracting C++ struct definitions and generating matching Julia types.

## C++ Code

```cpp
// point.cpp
#include <cmath>

struct Point {
    double x;
    double y;
};

extern "C" {
    Point create_point(double x, double y) {
        Point p;
        p.x = x;
        p.y = y;
        return p;
    }

    Point add_points(Point a, Point b) {
        return Point{a.x + b.x, a.y + b.y};
    }

    double distance(Point a, Point b) {
        double dx = b.x - a.x;
        double dy = b.y - a.y;
        return std::sqrt(dx*dx + dy*dy);
    }
}
```

## Step 1: Config with Debug Info

**CRITICAL:** Add `-g` flag to extract DWARF!

```toml
[project]
name = "pointlib"
root = "."

[compile]
source_files = ["point.cpp"]
include_dirs = []
flags = ["-std=c++17", "-fPIC", "-O2", "-g"]  # ← -g for DWARF!

[binary]
type = "library"

[llvm]
toolchain = "auto"
```

## Step 2: Build + Wrap

```julia
using RepliBuild

RepliBuild.build()  # Extracts DWARF with -g flag
RepliBuild.wrap()   # Generates Julia struct + wrappers
```

Output shows DWARF extraction:
```
Parsing DWARF debug info...
Types collected: 2 base, 0 pointer, 1 struct, 0 class
   Struct/class members extracted: 2
    Extracted 3 return types from DWARF
    Extracted 1 struct/class definitions with members
✓ Metadata saved
```

## Step 3: Use It!

```julia
include("julia/Pointlib.jl")
using .Pointlib

# Julia struct automatically generated from C++!
p1 = Point(1.0, 2.0)
p2 = Point(4.0, 6.0)

# Call C++ functions
p3 = add_points(p1, p2)  # Point(5.0, 8.0)
d = distance(p1, p2)     # 5.0
```

## Generated Wrapper (julia/Pointlib.jl)

```julia
module Pointlib

using Libdl

const LIBRARY_PATH = "julia/libpointlib.so"

# =============================================================================
# Struct Definitions (from DWARF debug info)
# =============================================================================

# C++ struct: Point (2 members)
mutable struct Point
    x::Cdouble
    y::Cdouble
end

export create_point, add_points, distance

"""
    create_point(x::Cdouble, y::Cdouble) -> Point

Wrapper for C++ function: `create_point(double, double)`

# Metadata
- Mangled symbol: `_Z12create_pointdd`
- Type safety: ✓ From DWARF compilation
"""
function create_point(x::Cdouble, y::Cdouble)::Point
    ccall((:_Z12create_pointdd, LIBRARY_PATH), Point,
          (Cdouble, Cdouble), x, y)
end

"""
    add_points(a::Point, b::Point) -> Point

Wrapper for C++ function: `add_points(Point, Point)`

# Metadata
- Mangled symbol: `_Z10add_points5PointS_`
- Type safety: ✓ From DWARF compilation
"""
function add_points(a::Point, b::Point)::Point
    ccall((:_Z10add_points5PointS_, LIBRARY_PATH), Point,
          (Point, Point), a, b)
end

"""
    distance(a::Point, b::Point) -> Cdouble

Wrapper for C++ function: `distance(Point, Point)`

# Metadata
- Mangled symbol: `_Z8distance5PointS_`
- Type safety: ✓ From DWARF compilation
"""
function distance(a::Point, b::Point)::Cdouble
    ccall((:_Z8distance5PointS_, LIBRARY_PATH), Cdouble,
          (Point, Point), a, b)
end

end # module
```

## What RepliBuild Did

### 1. DWARF Extraction (The Magic!)
```
C++ struct Point {     →  DWARF DIE: DW_TAG_structure_type
    double x;          →  DW_TAG_member (name: x, type: DW_TAG_base_type:double)
    double y;          →  DW_TAG_member (name: y, type: DW_TAG_base_type:double)
};                     →  Size: 16 bytes (2 × double)

                       ↓

Julia mutable struct Point
    x::Cdouble        →  Field 1: offset 0, size 8
    y::Cdouble        →  Field 2: offset 8, size 8
end                   →  Total: 16 bytes (ABI-accurate!)
```

### 2. Three-Way Validation

1. **DWARF says:** Point is a struct with x::double, y::double
2. **LLVM IR says:** Point is 16 bytes, two doubles
3. **Symbols say:** Functions take/return Point by value

**All match** → Safe to generate wrapper ✅

### 3. Type Safety

**Without RepliBuild:**
```julia
# Manual wrapper - WRONG!
mutable struct Point
    x::Float32  # ❌ Should be Cdouble (Float64)!
    y::Float64
end
# Will crash with memory corruption!
```

**With RepliBuild:**
```julia
# Auto-generated - CORRECT!
mutable struct Point
    x::Cdouble  # ✅ Extracted from DWARF
    y::Cdouble  # ✅ ABI-accurate
end
```

## Real Example Output

This is based on `docs/StructTest.jl` - a **real generated wrapper**:

```julia
julia> include("docs/StructTest.jl")

julia> using .StructTest

julia> p1 = create_point(1.0, 2.0)
Point(1.0, 2.0)

julia> p2 = create_point(4.0, 6.0)
Point(4.0, 6.0)

julia> p3 = add_points(p1, p2)
Point(5.0, 8.0)

julia> d = distance(p1, p2)
5.0
```

## Key Points

✅ **Automatic struct extraction** - No manual type definitions
✅ **ABI-accurate** - DWARF + LLVM IR ensure correctness
✅ **Type-safe** - Field types match C++ exactly
✅ **Zero maintenance** - Change C++, rebuild, wrapper updates

## Requirements

**Supported:**
- ✅ Standard-layout structs (POD)
- ✅ Trivially-copyable types
- ✅ Structs passed by value

**Not supported:**
- ❌ Virtual methods
- ❌ Inheritance
- ❌ Non-trivial constructors
- ❌ STL containers (std::vector, std::string)

See `LIMITATIONS.md` for complete rejection rules.

## Try It Yourself

The example in `docs/StructTest.jl` is a real, working wrapper you can inspect!
