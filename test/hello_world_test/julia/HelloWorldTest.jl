# Auto-generated Julia wrapper for hello_world_test
# Generated: 2026-02-02 21:08:31
# Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
# Library: libhello_world_test.so
# Metadata: compilation_metadata.json

module HelloWorldTest

using Libdl
import RepliBuild

const LIBRARY_PATH = "/home/grim/Desktop/Projects/RepliBuild.jl/test/hello_world_test/julia/libhello_world_test.so"

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
    "function_count" => 2,
    "generated_at" => "2026-02-02T21:08:29.163"
)

# =============================================================================
# Struct Definitions (from DWARF debug info)
# =============================================================================

# C++ struct: __va_list_tag (4 members)
struct __va_list_tag
    gp_offset::Cuint
    fp_offset::Cuint
    overflow_arg_area::Ptr{Cvoid}
    reg_save_area::Ptr{Cvoid}
end


export add, hello_world, __va_list_tag

"""
    add(a::Integer, b::Integer) -> Cint

Wrapper for C++ function: `add`

# Arguments
- `a::Cint`
- `b::Cint`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `add`
"""

function add(a::Integer, b::Integer)::Cint
    a_c = Cint(a)  # Auto-converts with overflow check
    b_c = Cint(b)  # Auto-converts with overflow check
    return ccall((:add, LIBRARY_PATH), Cint, (Cint, Cint,), a_c, b_c)
end

"""
    hello_world() -> Cvoid

Wrapper for C++ function: `hello_world`

# Arguments


# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `hello_world`
"""

function hello_world()::Cvoid
    ccall((:hello_world, LIBRARY_PATH), Cvoid, (), )
end


end # module HelloWorldTest
