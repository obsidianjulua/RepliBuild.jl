# Auto-generated Julia wrapper for basics_test
# Generated: 2026-01-26 02:09:21
# Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
# Library: libbasics_test.so
# Metadata: compilation_metadata.json
#
# Type Safety: Excellent (~95%) - Types extracted from DWARF debug info
# Ground truth: Types come from compiled binary, not headers
# Manual edits: Minimal to none required

module BasicsTest

using Libdl
import RepliBuild

const LIBRARY_PATH = "/home/grim/Desktop/Projects/RepliBuild.jl/test/basics_test/julia/libbasics_test.so"

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
    "function_count" => 5,
    "generated_at" => "2026-01-26T02:09:15.285"
)

# =============================================================================
# Struct Definitions (from DWARF debug info)
# =============================================================================

# C++ struct: PackedStruct (2 members)
mutable struct PackedStruct
    a::UInt8
    b::Cint
end

# C++ struct: PaddedStruct (2 members)
mutable struct PaddedStruct
    a::UInt8
    b::Cint
end

# C++ struct: __va_list_tag (4 members)
mutable struct __va_list_tag
    gp_offset::Cuint
    fp_offset::Cuint
    overflow_arg_area::Ptr{Cvoid}
    reg_save_area::Ptr{Cvoid}
end


export make_packed, make_padded, process_packed, process_padded, process_union, PaddedStruct, PackedStruct, __va_list_tag

"""
    make_packed(a::UInt8, b::Integer) -> PackedStruct

Wrapper for C++ function: `make_packed`

# Arguments
- `a::UInt8`
- `b::Cint`

# Returns
- `PackedStruct`

# Metadata
- Mangled symbol: `make_packed`
- Type safety:  From compilation
"""

function make_packed(a::UInt8, b::Integer)::PackedStruct
    b_c = Cint(b)  # Auto-converts with overflow check
    return ccall((:make_packed, LIBRARY_PATH), PackedStruct, (UInt8, Cint,), a, b_c)
end

"""
    make_padded(a::UInt8, b::Integer) -> PaddedStruct

Wrapper for C++ function: `make_padded`

# Arguments
- `a::UInt8`
- `b::Cint`

# Returns
- `PaddedStruct`

# Metadata
- Mangled symbol: `make_padded`
- Type safety:  From compilation
"""

function make_padded(a::UInt8, b::Integer)::PaddedStruct
    b_c = Cint(b)  # Auto-converts with overflow check
    return ccall((:make_padded, LIBRARY_PATH), PaddedStruct, (UInt8, Cint,), a, b_c)
end

"""
    process_packed(s::PackedStruct) -> Cvoid

Wrapper for C++ function: `process_packed`

# Arguments
- `s::PackedStruct`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `process_packed`
- Type safety:  From compilation
"""

function process_packed(s::PackedStruct)::Cvoid
    ccall((:process_packed, LIBRARY_PATH), Cvoid, (PackedStruct,), s)
end

"""
    process_padded(s::PaddedStruct) -> Cvoid

Wrapper for C++ function: `process_padded`

# Arguments
- `s::PaddedStruct`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `process_padded`
- Type safety:  From compilation
"""

function process_padded(s::PaddedStruct)::Cvoid
    ccall((:process_padded, LIBRARY_PATH), Cvoid, (PaddedStruct,), s)
end

"""
    process_union(u::Any) -> Cvoid

Wrapper for C++ function: `process_union`

# Arguments
- `u::Any`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `process_union`
- Type safety:  From compilation
"""

function process_union(u::Any)::Cvoid
    ccall((:process_union, LIBRARY_PATH), Cvoid, (Any,), u)
end


end # module BasicsTest
