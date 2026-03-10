# Auto-generated Julia wrapper for basics_test
# Generated: 2026-03-09 23:42:18
# Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
# Library: libbasics_test.so
# Metadata: compilation_metadata.json

module BasicsTest

const Cintptr_t = Int
const Cuintptr_t = UInt

using Libdl
import RepliBuild
import Base: unsafe_convert

const LIBRARY_PATH = "/home/john/Desktop/Projects/RepliBuild.jl/test/basics_test/julia/libbasics_test.so"
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
    "target_triple" => "x86_64-unknown-linux-gnu",
    "function_count" => 6,
    "generated_at" => "2026-03-09T23:42:18.886"
)

const LTO_IR = ""  # LTO disabled for this build
const THUNKS_LTO_IR = ""

# =============================================================================
# Forward Declarations (Opaque + Ptr-referenced types)
# =============================================================================


# =============================================================================
# Struct Definitions (from DWARF debug info)
# =============================================================================

# C union: NumberUnion (size 4 bytes)
mutable struct NumberUnion
    data::NTuple{4, UInt8}
end
NumberUnion() = NumberUnion(ntuple(i -> 0x00, 4))

# C++ struct: PackedStruct (2 members)
struct PackedStruct
    a::UInt8
    b::Cint
end

# Zero-initializer for PackedStruct
function PackedStruct()
    ref = Ref{PackedStruct}()
    GC.@preserve ref begin
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), Base.unsafe_convert(Ptr{Cvoid}, ref), 0, sizeof(PackedStruct))
    end
    return ref[]
end

# C++ struct: PaddedStruct (2 members)
struct PaddedStruct
    a::UInt8
    _pad_0::NTuple{3, UInt8}
    b::Cint
end

# Zero-initializer for PaddedStruct
function PaddedStruct()
    ref = Ref{PaddedStruct}()
    GC.@preserve ref begin
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), Base.unsafe_convert(Ptr{Cvoid}, ref), 0, sizeof(PaddedStruct))
    end
    return ref[]
end


"""Get union member `i` as `Cint` from `NumberUnion`."""
function get_i(u::NumberUnion)::Cint
    return unsafe_load(Ptr{Cint}(pointer_from_objref(u)))
end

"""Set union member `i` as `Cint` in `NumberUnion`."""
function set_i!(u::NumberUnion, v::Cint)
    unsafe_store!(Ptr{Cint}(pointer_from_objref(u)), v)
end

"""Get union member `f` as `Cfloat` from `NumberUnion`."""
function get_f(u::NumberUnion)::Cfloat
    return unsafe_load(Ptr{Cfloat}(pointer_from_objref(u)))
end

"""Set union member `f` as `Cfloat` in `NumberUnion`."""
function set_f!(u::NumberUnion, v::Cfloat)
    unsafe_store!(Ptr{Cfloat}(pointer_from_objref(u)), v)
end

export get_i, set_i!, get_f, set_f!, global_string, global_string_ptr, global_int, global_int_ptr, make_packed, make_padded, process_packed, process_padded, process_union, sum_ints, PaddedStruct, PackedStruct, NumberUnion

# =============================================================================
# Global Variables
# =============================================================================

"""
    global_string()

Get value of global variable `global_string`.
"""
function global_string()::Ptr{UInt8}
    ptr = cglobal((:global_string, LIBRARY_PATH), Ptr{UInt8})
    return unsafe_load(ptr)
end

"""
    global_string_ptr()

Get pointer to global variable `global_string`.
"""
function global_string_ptr()::Ptr{Ptr{UInt8}}
    return cglobal((:global_string, LIBRARY_PATH), Ptr{UInt8})
end

"""
    global_int()

Get value of global variable `global_int`.
"""
function global_int()::Cint
    ptr = cglobal((:global_int, LIBRARY_PATH), Cint)
    return unsafe_load(ptr)
end

"""
    global_int_ptr()

Get pointer to global variable `global_int`.
"""
function global_int_ptr()::Ptr{Cint}
    return cglobal((:global_int, LIBRARY_PATH), Cint)
end


"""
    make_packed(a::UInt8, b::Integer) -> PackedStruct

Wrapper for `make_packed`

# Arguments
- `a::UInt8`
- `b::Cint`

# Returns
- `PackedStruct`

# Metadata
- Mangled symbol: `make_packed`
"""

function make_packed(a::UInt8, b::Integer)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    return RepliBuild.JITManager.invoke("_mlir_ciface_make_packed_thunk", PackedStruct, a, b)
end
"""
    make_padded(a::UInt8, b::Integer) -> PaddedStruct

Wrapper for `make_padded`

# Arguments
- `a::UInt8`
- `b::Cint`

# Returns
- `PaddedStruct`

# Metadata
- Mangled symbol: `make_padded`
"""

function make_padded(a::UInt8, b::Integer)::PaddedStruct
    b_c = Cint(b)  # Auto-converts with overflow check
    return ccall((:make_padded, LIBRARY_PATH), PaddedStruct, (UInt8, Cint,), a, b_c)
end

"""
    process_packed(s::PackedStruct) -> Cvoid

Wrapper for `process_packed`

# Arguments
- `s::PackedStruct`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `process_packed`
"""

function process_packed(s::PackedStruct)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    return RepliBuild.JITManager.invoke("_mlir_ciface_process_packed_thunk", s)
end
"""
    process_padded(s::PaddedStruct) -> Cvoid

Wrapper for `process_padded`

# Arguments
- `s::PaddedStruct`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `process_padded`
"""

function process_padded(s::PaddedStruct)::Cvoid
    ccall((:process_padded, LIBRARY_PATH), Cvoid, (PaddedStruct,), s)
end

"""
    process_union(u::Any) -> Cvoid

Wrapper for `process_union`

# Arguments
- `u::Any`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `process_union`
"""

function process_union(u::Any)::Cvoid
    ccall((:process_union, LIBRARY_PATH), Cvoid, (Any,), u)
end

"""
    sum_ints(count::Integer) -> Cint

Wrapper for variadic C function: `sum_ints` (base call with fixed args only)
"""

function sum_ints(count::Integer)::Cint
    count_c = Cint(count)
    return ccall((:sum_ints, LIBRARY_PATH), Cint, (Cint,), count_c)
end


end # module BasicsTest
