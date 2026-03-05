# Auto-generated Julia wrapper for callback_test
# Generated: 2026-03-04 19:05:00
# Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
# Library: libcallback_test.so
# Metadata: compilation_metadata.json

module CallbackTest

const Cintptr_t = Int
const Cuintptr_t = UInt

using Libdl
import RepliBuild
import Base: unsafe_convert

const LIBRARY_PATH = "/home/john/Desktop/Projects/RepliBuild.jl/test/callback_test/julia/libcallback_test.so"
const THUNKS_LIBRARY_PATH = "/home/john/Desktop/Projects/RepliBuild.jl/test/callback_test/julia/libcallback_test_thunks.so"

# Verify library exists
if !isfile(LIBRARY_PATH)
    error("Library not found: $LIBRARY_PATH")
end

# Library handles for manual management if needed
const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const THUNKS_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    # Load main library explicitly to ensure symbols are available
    LIB_HANDLE[] = Libdl.dlopen(LIBRARY_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    
    # Load AOT thunks library if it was successfully generated
    if !isempty(THUNKS_LIBRARY_PATH) && isfile(THUNKS_LIBRARY_PATH)
        THUNKS_HANDLE[] = Libdl.dlopen(THUNKS_LIBRARY_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    elseif false
        @warn "AOT Thunks library not found, but advanced FFI features are required. These features will fail at runtime."
    end
end
# =============================================================================
# Compilation Metadata
# =============================================================================

const METADATA = Dict(
    "llvm_version" => "21.1.8",
    "clang_version" => "clang version 21.1.8",
    "optimization" => "0",
    "target_triple" => "x86_64-unknown-linux-gnu",
    "function_count" => 2,
    "generated_at" => "2026-03-04T19:05:00.368"
)

export execute_binary_op, simulate_work

"""
    execute_binary_op(op::Any, a::Integer, b::Integer) -> Cint

Wrapper for C++ function: `execute_binary_op`

# Arguments
- `op::Ptr{Cvoid}` - Callback function
- `a::Cint`
- `b::Cint`

# Returns
- `Cint`

            # Callback Signatures
**Callback `op`**: Create using `@cfunction`
```julia
callback = @cfunction(my_callback, Cint, (Cint, Cint,)) Ptr{Cvoid}
```

# Metadata
- Mangled symbol: `execute_binary_op`
"""

function execute_binary_op(op::Any, a::Integer, b::Integer)::Cint
    a_c = Cint(a)  # Auto-converts with overflow check
    b_c = Cint(b)  # Auto-converts with overflow check
    return ccall((:execute_binary_op, LIBRARY_PATH), Cint, (Ptr{Cvoid}, Cint, Cint,), op, a_c, b_c)
end

"""
    simulate_work(iterations::Integer, cb::Any) -> Cvoid

Wrapper for C++ function: `simulate_work`

# Arguments
- `iterations::Cint`
- `cb::Ptr{Cvoid}` - Callback function

# Returns
- `Cvoid`

            # Callback Signatures
**Callback `cb`**: Create using `@cfunction`
```julia
callback = @cfunction(my_callback, Cvoid, (Cfloat,)) Ptr{Cvoid}
```

# Metadata
- Mangled symbol: `simulate_work`
"""

function simulate_work(iterations::Integer, cb::Any)::Cvoid
    iterations_c = Cint(iterations)  # Auto-converts with overflow check
    return ccall((:simulate_work, LIBRARY_PATH), Cvoid, (Cint, Ptr{Cvoid},), iterations_c, cb)
end


end # module CallbackTest
