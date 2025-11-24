# Auto-generated Julia wrapper for struct_test
# Generated: 2025-11-24 16:21:38
# Generator: RepliBuild Wrapper (Tier 3: Introspective)
# Library: libstruct_test.so
# Metadata: compilation_metadata.json
#
# Type Safety: ✅ Perfect - Types extracted from compilation
# Language: Language-agnostic (via LLVM IR)
# Manual edits: None required

module StructTest

const LIBRARY_PATH = "/home/grim/Desktop/Projects/RepliBuild.jl/examples/struct_test/julia/libstruct_test.so"

# Verify library exists
if !isfile(LIBRARY_PATH)
    error("Library not found: $LIBRARY_PATH")
end

# =============================================================================
# Compilation Metadata
# =============================================================================

const METADATA = Dict(
    "llvm_version" => "21.1.5",
    "clang_version" => "clang version 21.1.5",
    "optimization" => "2",
    "target_triple" => "x86_64-unknown-linux-gnu",
    "function_count" => 3,
    "generated_at" => "2025-11-24T16:21:23.218"
)

# =============================================================================
# Struct Definitions (from C++)
# =============================================================================

# Opaque struct: Point
mutable struct Point
    data::NTuple{32, UInt8}  # Placeholder - actual size from DWARF
end


export add_points, create_point, distance

"""
    add_points(arg1::Point, arg2::Point) -> Any

Wrapper for C++ function: `add_points(Point, Point)`

# Arguments
- `arg1::Point`
- `arg2::Point`

# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z10add_points5PointS_`
- Type safety: ✅ From compilation
"""

function add_points(arg1::Point, arg2::Point)::Point
    return ccall((:_Z10add_points5PointS_, LIBRARY_PATH), Point, (Point, Point,), arg1, arg2)
end

"""
    create_point(arg1::Cdouble, arg2::Cdouble) -> Any

Wrapper for C++ function: `create_point(double, double)`

# Arguments
- `arg1::Cdouble`
- `arg2::Cdouble`

# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z12create_pointdd`
- Type safety: ✅ From compilation
"""

function create_point(arg1::Cdouble, arg2::Cdouble)::Point
    return ccall((:_Z12create_pointdd, LIBRARY_PATH), Point, (Cdouble, Cdouble,), arg1, arg2)
end

"""
    distance(arg1::Point, arg2::Point) -> Cdouble

Wrapper for C++ function: `distance(Point, Point)`

# Arguments
- `arg1::Point`
- `arg2::Point`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_Z8distance5PointS_`
- Type safety: ✅ From compilation
"""

function distance(arg1::Point, arg2::Point)::Cdouble
    ccall((:_Z8distance5PointS_, LIBRARY_PATH), Cdouble, (Point, Point,), arg1, arg2)
end


end # module StructTest
