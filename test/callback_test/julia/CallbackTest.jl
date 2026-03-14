# Auto-generated Julia wrapper for callback_test
# Generated: 2026-03-13 23:06:19
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
const THUNKS_LIBRARY_PATH = ""

# Verify library exists
if !isfile(LIBRARY_PATH)
    error("Library not found: $LIBRARY_PATH")
end

# Flush C stdout so printf output appears immediately in the Julia REPL
@inline _flush_cstdout() = ccall(:fflush, Cint, (Ptr{Cvoid},), C_NULL)

# Unbuffer C stdout on module load so printf output is visible in the REPL
let c_stdout = unsafe_load(cglobal(:stdout, Ptr{Cvoid}))
    ccall(:setvbuf, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Csize_t), c_stdout, C_NULL, 2, 0)
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
    "function_count" => 2,
    "generated_at" => "2026-03-13T23:06:19.306"
)

const LTO_IR = ""  # LTO disabled for this build
const THUNKS_LTO_IR = ""

export execute_binary_op, simulate_work

"""
    execute_binary_op(op::Any, a::Integer, b::Integer) -> Cint

Wrapper for `execute_binary_op`

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
    a_c = Cint(a)
    b_c = Cint(b)
    return ccall((:execute_binary_op, LIBRARY_PATH), Cint, (Ptr{Cvoid}, Cint, Cint,), op, a_c, b_c)
end

"""
    simulate_work(iterations::Integer, cb::Any) -> Cvoid

Wrapper for `simulate_work`

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
    iterations_c = Cint(iterations)
    return ccall((:simulate_work, LIBRARY_PATH), Cvoid, (Cint, Ptr{Cvoid},), iterations_c, cb)
end


end # module CallbackTest
