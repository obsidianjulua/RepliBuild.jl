# Auto-generated Julia wrapper for callback_test
# Generated: 2026-03-17 13:20:47
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

function __init__()
    # Initialize the global JIT context with this library's vtables
    RepliBuild.JITManager.initialize_global_jit(LIBRARY_PATH)
end
# =============================================================================
# Compilation Metadata
# =============================================================================

const METADATA = Dict(
    "llvm_version" => "21.1.8",
    "clang_version" => "clang version 21.1.8",
    "optimization" => "0",
    "target_triple" => "x86_64-pc-linux-gnu",
    "function_count" => 8,
    "generated_at" => "2026-03-17T13:20:40.927"
)

const LTO_IR = ""  # LTO disabled for this build
const THUNKS_LTO_IR = ""

# =============================================================================
# Forward Declarations (Opaque + Ptr-referenced types)
# =============================================================================


# =============================================================================
# Struct Definitions (from DWARF debug info)
# =============================================================================

# C++ struct: _Terminator (2 members)
struct _Terminator
    _M_this::Ptr{Cvoid}
    _M_r::Csize_t
end

# Zero-initializer for _Terminator
function _Terminator()
    ref = Ref{_Terminator}()
    GC.@preserve ref begin
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), Base.unsafe_convert(Ptr{Cvoid}, ref), 0, sizeof(_Terminator))
    end
    return ref[]
end


export execute_binary_op, simulate_work, throws_int, void_thrower, always_throws, safe_multiply, throws_midway, throws_if_negative, _Terminator

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

"""
    throws_int(x::Integer) -> Cint

Wrapper for `throws_int(int)`

# Arguments
- `x::Cint`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z10throws_inti`
"""

function throws_int(x::Integer)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    return RepliBuild.JITManager.invoke("_mlir_ciface__Z10throws_inti_thunk", Cint, x)
end
"""
    void_thrower() -> Cvoid

Wrapper for `void_thrower()`

# Arguments


# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_Z12void_throwerv`
"""

function void_thrower()
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    return RepliBuild.JITManager.invoke("_mlir_ciface__Z12void_throwerv_thunk")
end
"""
    always_throws(x::Integer) -> Cint

Wrapper for `always_throws(int)`

# Arguments
- `x::Cint`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z13always_throwsi`
"""

function always_throws(x::Integer)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    return RepliBuild.JITManager.invoke("_mlir_ciface__Z13always_throwsi_thunk", Cint, x)
end
"""
    safe_multiply(a::Integer, b::Integer) -> Cint

Wrapper for `safe_multiply(int, int)`

# Arguments
- `a::Cint`
- `b::Cint`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z13safe_multiplyii`
"""

function safe_multiply(a::Integer, b::Integer)::Cint
    a_c = Cint(a)
    b_c = Cint(b)
    return ccall((:_Z13safe_multiplyii, LIBRARY_PATH), Cint, (Cint, Cint,), a_c, b_c)
end

"""
    throws_midway(iterations::Integer) -> Cint

Wrapper for `throws_midway(int)`

# Arguments
- `iterations::Cint`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z13throws_midwayi`
"""

function throws_midway(iterations::Integer)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    return RepliBuild.JITManager.invoke("_mlir_ciface__Z13throws_midwayi_thunk", Cint, iterations)
end
"""
    throws_if_negative(x::Integer) -> Cint

Wrapper for `throws_if_negative(int)`

# Arguments
- `x::Cint`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z18throws_if_negativei`
"""

function throws_if_negative(x::Integer)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    return RepliBuild.JITManager.invoke("_mlir_ciface__Z18throws_if_negativei_thunk", Cint, x)
end

end # module CallbackTest
