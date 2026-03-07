# Auto-generated Julia wrapper for hello_world_test
# Generated: 2026-03-06 19:57:57
# Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
# Library: libhello_world_test.so
# Metadata: compilation_metadata.json

module HelloWorldTest

const Cintptr_t = Int
const Cuintptr_t = UInt

using Libdl
import RepliBuild
import Base: unsafe_convert

const LIBRARY_PATH = "/home/john/Desktop/Projects/RepliBuild.jl/test/hello_world_test/julia/libhello_world_test.so"
const THUNKS_LIBRARY_PATH = ""

# Verify library exists
if !isfile(LIBRARY_PATH)
    error("Library not found: $LIBRARY_PATH")
end

# Library handle for manual management if needed
const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    # Load library explicitly to ensure symbols are available
    LIB_HANDLE[] = Libdl.dlopen(LIBRARY_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
end
# =============================================================================
# Compilation Metadata
# =============================================================================

const METADATA = Dict(
    "llvm_version" => "21.1.8",
    "clang_version" => "clang version 21.1.8",
    "optimization" => "0",
    "target_triple" => "x86_64-unknown-linux-gnu",
    "function_count" => 1,
    "generated_at" => "2026-03-06T19:57:54.247"
)

const LTO_IR = ""  # LTO disabled for this build
const THUNKS_LTO_IR = ""

# =============================================================================
# Struct Definitions (from DWARF debug info)
# =============================================================================


export hello_world

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
