# Auto-generated Julia wrapper for vtable_test
# Generated: 2026-01-07 04:37:57
# Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
# Library: libvtable_test.so
# Metadata: compilation_metadata.json
#
# Type Safety: Excellent (~95%) - Types extracted from DWARF debug info
# Ground truth: Types come from compiled binary, not headers
# Manual edits: Minimal to none required

module VtableTest

const LIBRARY_PATH = "/home/grim/Desktop/Projects/RepliBuild.jl/test/vtable_test/julia/libvtable_test.so"

# Verify library exists
if !isfile(LIBRARY_PATH)
    error("Library not found: $LIBRARY_PATH")
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
    "generated_at" => "2026-01-07T04:37:54.445"
)

# =============================================================================
# Struct Definitions (from DWARF debug info)
# =============================================================================

# C++ struct: Circle (1 members)
mutable struct Circle
    radius::Cdouble
end

# C++ struct: Rectangle (2 members)
mutable struct Rectangle
    width::Cdouble
    height::Cdouble
end

# C++ struct: Shape (1 members)
mutable struct Shape
    _vptr_Shape::Ptr{Cvoid}
end


export create_circle, create_rectangle, delete_shape, get_area, get_perimeter, Circle_area, Circle_perimeter, Rectangle_area, Rectangle_perimeter, Circle, Shape, Rectangle

"""
    create_circle(r::Cdouble) -> Ptr{Cvoid}

Wrapper for C++ function: `create_circle`

# Arguments
- `r::Cdouble`

# Returns
- `Ptr{Cvoid}`

# Metadata
- Mangled symbol: `create_circle`
- Type safety:  From compilation
"""

function create_circle(r::Cdouble)::Ptr{Cvoid}
    ccall((:create_circle, LIBRARY_PATH), Ptr{Cvoid}, (Cdouble,), r)
end

"""
    create_rectangle(w::Cdouble, h::Cdouble) -> Ptr{Cvoid}

Wrapper for C++ function: `create_rectangle`

# Arguments
- `w::Cdouble`
- `h::Cdouble`

# Returns
- `Ptr{Cvoid}`

# Metadata
- Mangled symbol: `create_rectangle`
- Type safety:  From compilation
"""

function create_rectangle(w::Cdouble, h::Cdouble)::Ptr{Cvoid}
    ccall((:create_rectangle, LIBRARY_PATH), Ptr{Cvoid}, (Cdouble, Cdouble,), w, h)
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

function delete_shape(s::Ptr{Shape})::Cvoid
    ccall((:delete_shape, LIBRARY_PATH), Cvoid, (Ptr{Shape},), s)
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

function get_area(s::Ptr{Shape})::Cdouble
    ccall((:get_area, LIBRARY_PATH), Cdouble, (Ptr{Shape},), s)
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

function get_perimeter(s::Ptr{Shape})::Cdouble
    ccall((:get_perimeter, LIBRARY_PATH), Cdouble, (Ptr{Shape},), s)
end

"""
    Circle_area() -> Cdouble

Wrapper for C++ function: `Circle::area() const`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK6Circle4areaEv`
- Type safety:  From compilation
"""

function Circle_area()::Cdouble
    ccall((:_ZNK6Circle4areaEv, LIBRARY_PATH), Cdouble, (), )
end

"""
    Circle_perimeter() -> Cdouble

Wrapper for C++ function: `Circle::perimeter() const`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK6Circle9perimeterEv`
- Type safety:  From compilation
"""

function Circle_perimeter()::Cdouble
    ccall((:_ZNK6Circle9perimeterEv, LIBRARY_PATH), Cdouble, (), )
end

"""
    Rectangle_area() -> Cdouble

Wrapper for C++ function: `Rectangle::area() const`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK9Rectangle4areaEv`
- Type safety:  From compilation
"""

function Rectangle_area()::Cdouble
    ccall((:_ZNK9Rectangle4areaEv, LIBRARY_PATH), Cdouble, (), )
end

"""
    Rectangle_perimeter() -> Cdouble

Wrapper for C++ function: `Rectangle::perimeter() const`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK9Rectangle9perimeterEv`
- Type safety:  From compilation
"""

function Rectangle_perimeter()::Cdouble
    ccall((:_ZNK9Rectangle9perimeterEv, LIBRARY_PATH), Cdouble, (), )
end


end # module VtableTest
