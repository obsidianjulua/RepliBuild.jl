# Auto-generated Julia wrapper for vtable_test
# Generated: 2026-02-02 22:02:03
# Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
# Library: libvtable_test.so
# Metadata: compilation_metadata.json

module VtableTest

using Libdl
import RepliBuild

const LIBRARY_PATH = "/home/grim/Desktop/Projects/RepliBuild.jl/test/vtable_test/julia/libvtable_test.so"

# Verify library exists
if !isfile(LIBRARY_PATH)
    error("Library not found: $LIBRARY_PATH")
end

# Library handle for manual management if needed
const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    # Load library explicitly to ensure symbols are available
    LIB_HANDLE[] = Libdl.dlopen(LIBRARY_PATH)
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
    "generated_at" => "2026-02-02T22:02:03.672"
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


# =============================================================================
# Managed Types (Auto-Finalizers)
# =============================================================================

mutable struct ManagedShape
    handle::Ptr{Shape}
    
    function ManagedShape(ptr::Ptr{Shape})
        if ptr == C_NULL
            error("Cannot wrap NULL pointer in ManagedShape")
        end
        obj = new(ptr)
        finalizer(obj) do x
            # Call deleter: delete_shape(x.handle)
            ccall((:delete_shape, LIBRARY_PATH), Cvoid, (Ptr{Shape},), x.handle)
        end
        return obj
    end
end

# Allow passing Managed object to ccall expecting Ptr
Base.unsafe_convert(::Type{Ptr{Shape}}, obj::ManagedShape) = obj.handle

export ManagedShape

export create_circle, create_circle_safe, create_rectangle, create_rectangle_safe, delete_shape, get_area, get_perimeter, Circle_area, Circle_perimeter, Rectangle_area, Rectangle_perimeter, Circle, Shape, Rectangle

"""
    create_circle(r::Cdouble) -> Ptr{Shape}

Wrapper for C++ function: `create_circle`

# Arguments
- `r::Cdouble`

# Returns
- `Ptr{Shape}`

# Metadata
- Mangled symbol: `create_circle`
"""

function create_circle(r::Cdouble)::Ptr{Shape}
    ccall((:create_circle, LIBRARY_PATH), Ptr{Shape}, (Cdouble,), r)
end

"""
    create_circle_safe(r::Cdouble) -> ManagedShape

Safe wrapper for `create_circle` that returns a managed object with automatic finalization.
"""
function create_circle_safe(r::Cdouble)::ManagedShape
    ptr = create_circle(r)
    return ManagedShape(ptr)
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
"""

function create_rectangle(w::Cdouble, h::Cdouble)::Ptr{Shape}
    ccall((:create_rectangle, LIBRARY_PATH), Ptr{Shape}, (Cdouble, Cdouble,), w, h)
end

"""
    create_rectangle_safe(w::Cdouble, h::Cdouble) -> ManagedShape

Safe wrapper for `create_rectangle` that returns a managed object with automatic finalization.
"""
function create_rectangle_safe(w::Cdouble, h::Cdouble)::ManagedShape
    ptr = create_rectangle(w, h)
    return ManagedShape(ptr)
end
"""
    delete_shape(s::Any) -> Cvoid

Wrapper for C++ function: `delete_shape`

# Arguments
- `s::Ptr{Shape}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `delete_shape`
"""

function delete_shape(s::Any)::Cvoid
    ccall((:delete_shape, LIBRARY_PATH), Cvoid, (Ptr{Shape},), s)
end

"""
    get_area(s::Any) -> Cdouble

Wrapper for C++ function: `get_area`

# Arguments
- `s::Ptr{Shape}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `get_area`
"""

function get_area(s::Any)::Cdouble
    ccall((:get_area, LIBRARY_PATH), Cdouble, (Ptr{Shape},), s)
end

"""
    get_perimeter(s::Any) -> Cdouble

Wrapper for C++ function: `get_perimeter`

# Arguments
- `s::Ptr{Shape}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `get_perimeter`
"""

function get_perimeter(s::Any)::Cdouble
    ccall((:get_perimeter, LIBRARY_PATH), Cdouble, (Ptr{Shape},), s)
end

"""
    Circle_area(this::Any) -> Cdouble

Wrapper for C++ function: `Circle::area() const`

# Arguments
- `this::Ptr{Circle}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK6Circle4areaEv`
"""

function Circle_area(this::Any)::Cdouble
    ccall((:_ZNK6Circle4areaEv, LIBRARY_PATH), Cdouble, (Ptr{Circle},), this)
end

"""
    Circle_perimeter(this::Any) -> Cdouble

Wrapper for C++ function: `Circle::perimeter() const`

# Arguments
- `this::Ptr{Circle}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK6Circle9perimeterEv`
"""

function Circle_perimeter(this::Any)::Cdouble
    ccall((:_ZNK6Circle9perimeterEv, LIBRARY_PATH), Cdouble, (Ptr{Circle},), this)
end

"""
    Rectangle_area(this::Any) -> Cdouble

Wrapper for C++ function: `Rectangle::area() const`

# Arguments
- `this::Ptr{Rectangle}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK9Rectangle4areaEv`
"""

function Rectangle_area(this::Any)::Cdouble
    ccall((:_ZNK9Rectangle4areaEv, LIBRARY_PATH), Cdouble, (Ptr{Rectangle},), this)
end

"""
    Rectangle_perimeter(this::Any) -> Cdouble

Wrapper for C++ function: `Rectangle::perimeter() const`

# Arguments
- `this::Ptr{Rectangle}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK9Rectangle9perimeterEv`
"""

function Rectangle_perimeter(this::Any)::Cdouble
    ccall((:_ZNK9Rectangle9perimeterEv, LIBRARY_PATH), Cdouble, (Ptr{Rectangle},), this)
end


end # module VtableTest
