# Auto-generated Julia wrapper for vtable_test
# Generated: 2026-02-01 17:00:23
# Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
# Library: libvtable_test.so
# Metadata: compilation_metadata.json
#
# Type Safety: Excellent (~95%) - Types extracted from DWARF debug info
# Ground truth: Types come from compiled binary, not headers
# Manual edits: Minimal to none required

module VtableTest

using Libdl
import RepliBuild

const LIBRARY_PATH = "/home/grim/Desktop/Projects/RepliBuild.jl/test/vtable_test/julia/libvtable_test.so"

# Verify library exists
if !isfile(LIBRARY_PATH)
    error("Library not found: $LIBRARY_PATH")
end

function __init__()
    # Initialize the global JIT context with this library's vtables
    RepliBuild.JITManager.initialize_global_jit(LIBRARY_PATH)
end

# =============================================================================
# Compilation Metadata
# =============================================================================

const METADATA = Dict(
    "llvm_version" => "21.1.6",
    "clang_version" => "clang version 21.1.6",
    "optimization" => "0",
    "target_triple" => "x86_64-unknown-linux-gnu",
    "function_count" => 13,
    "generated_at" => "2026-02-01T17:00:17.265"
)

# =============================================================================
# Struct Definitions (from DWARF debug info)
# =============================================================================

# C++ struct: Circle (2 members)
struct Circle
    _vptr_Shape::Ptr{Cvoid}
    radius::Cdouble
end

# C++ struct: Rectangle (3 members)
struct Rectangle
    _vptr_Shape::Ptr{Cvoid}
    width::Cdouble
    height::Cdouble
end

# C++ struct: Shape (1 members)
struct Shape
    _vptr_Shape::Ptr{Cvoid}
end


export create_circle, create_rectangle, delete_shape, get_area, get_perimeter, Circle_area, Circle_perimeter, Rectangle_area, Rectangle_perimeter, Circle, Shape, Rectangle

"""
    create_circle(r::Cdouble) -> Ptr{Shape}

Wrapper for C++ function: `create_circle`

# Arguments
- `r::Cdouble`

# Returns
- `Ptr{Shape}`

# Metadata
- Mangled symbol: `create_circle`
- Type safety:  From compilation
"""

function create_circle(r::Cdouble)::Ptr{Shape}
    ccall((:create_circle, LIBRARY_PATH), Ptr{Shape}, (Cdouble,), r)
end

"""
    create_rectangle(w::Cdouble, h::Cdouble) -> Ptr{Shape}

Wrapper for C++ function: `create_rectangle`

# Arguments
- `w::Cdouble`
- `h::Cdouble`

# Returns
- `Ptr{Shape}`

# Metadata
- Mangled symbol: `create_rectangle`
- Type safety:  From compilation
"""

function create_rectangle(w::Cdouble, h::Cdouble)::Ptr{Shape}
    ccall((:create_rectangle, LIBRARY_PATH), Ptr{Shape}, (Cdouble, Cdouble,), w, h)
end

"""
    delete_shape(s::Ptr{Shape}) -> Cvoid

Wrapper for C++ function: `delete_shape`

# Arguments
- `s::Ptr{Shape}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `delete_shape`
- Type safety:  From compilation
"""

function delete_shape(s::Ptr{Shape})
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    return RepliBuild.JITManager.invoke("delete_shape", s)
end
"""
    get_area(s::Ptr{Shape}) -> Cdouble

Wrapper for C++ function: `get_area`

# Arguments
- `s::Ptr{Shape}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `get_area`
- Type safety:  From compilation
"""

function get_area(s::Ptr{Shape})
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    return RepliBuild.JITManager.invoke("get_area", s)
end
"""
    get_perimeter(s::Ptr{Shape}) -> Cdouble

Wrapper for C++ function: `get_perimeter`

# Arguments
- `s::Ptr{Shape}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `get_perimeter`
- Type safety:  From compilation
"""

function get_perimeter(s::Ptr{Shape})
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    return RepliBuild.JITManager.invoke("get_perimeter", s)
end
"""
    Circle_area(this::Ptr{Circle}) -> Cdouble

Wrapper for C++ function: `Circle::area() const`

# Arguments
- `this::Ptr{Circle}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK6Circle4areaEv`
- Type safety:  From compilation
"""

function Circle_area(this::Ptr{Circle})::Cdouble
    ccall((:_ZNK6Circle4areaEv, LIBRARY_PATH), Cdouble, (Ptr{Circle},), this)
end

"""
    Circle_perimeter(this::Ptr{Circle}) -> Cdouble

Wrapper for C++ function: `Circle::perimeter() const`

# Arguments
- `this::Ptr{Circle}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK6Circle9perimeterEv`
- Type safety:  From compilation
"""

function Circle_perimeter(this::Ptr{Circle})::Cdouble
    ccall((:_ZNK6Circle9perimeterEv, LIBRARY_PATH), Cdouble, (Ptr{Circle},), this)
end

"""
    Rectangle_area(this::Ptr{Rectangle}) -> Cdouble

Wrapper for C++ function: `Rectangle::area() const`

# Arguments
- `this::Ptr{Rectangle}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK9Rectangle4areaEv`
- Type safety:  From compilation
"""

function Rectangle_area(this::Ptr{Rectangle})::Cdouble
    ccall((:_ZNK9Rectangle4areaEv, LIBRARY_PATH), Cdouble, (Ptr{Rectangle},), this)
end

"""
    Rectangle_perimeter(this::Ptr{Rectangle}) -> Cdouble

Wrapper for C++ function: `Rectangle::perimeter() const`

# Arguments
- `this::Ptr{Rectangle}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK9Rectangle9perimeterEv`
- Type safety:  From compilation
"""

function Rectangle_perimeter(this::Ptr{Rectangle})::Cdouble
    ccall((:_ZNK9Rectangle9perimeterEv, LIBRARY_PATH), Cdouble, (Ptr{Rectangle},), this)
end


end # module VtableTest
