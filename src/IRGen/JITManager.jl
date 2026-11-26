#!/usr/bin/env julia
# JITManager.jl - Manages the lifecycle of MLIR JIT compilation for C++ vtables
# Acts as the bridge between Julia wrappers and the MLIR execution engine.

module JITManager

using ..MLIRNative
import ..MLIRNative: CXX_PERSONALITY
using ..JLCSIRGenerator
using ..DWARFParser
import ..Compiler
using Libdl
import .._read_json

export get_jit_thunk, ensure_jit_initialized, JITContext, invoke, CxxException

"""
    LibraryEngine

One MLIR execution engine per wrapped binary. Multiple generated wrappers can
coexist in a session — each `initialize_global_jit(binary_path)` call creates
(or reuses) the engine for ITS binary instead of no-opping after the first
library wins. Thunk symbol names are derived from mangled C++ names, so they
are unique across libraries and can share one global symbol cache.
"""
mutable struct LibraryEngine
    binary_path::String
    mlir_ctx::Ptr{Cvoid}
    jit_engine::Union{Ptr{Cvoid}, Nothing}
    vtable_info::Union{VtableInfo, Nothing}
    init_error::Union{Exception, Nothing}
end

# Global singleton to manage JIT state
mutable struct JITContext
    # Legacy single-engine fields — mirror the FIRST successfully initialized
    # engine for backward compatibility. New code reads `engines`.
    mlir_ctx::Ptr{Cvoid}
    jit_engine::Union{Ptr{Cvoid}, Nothing}
    @atomic compiled_symbols::Dict{String, Ptr{Cvoid}}
    vtable_info::Union{VtableInfo, Nothing}
    @atomic initialized::Bool
    init_error::Union{Exception, Nothing}
    lock::ReentrantLock
    engines::Vector{LibraryEngine}

    function JITContext()
        new(C_NULL, nothing, Dict{String, Ptr{Cvoid}}(), nothing, false, nothing,
            ReentrantLock(), LibraryEngine[])
    end
end

const GLOBAL_JIT = JITContext()

# =============================================================================
# C++ Exception Propagation
# =============================================================================

"""
    CxxException <: Exception

Exception type for C++ exceptions caught by JLCS try_call thunks.
The message contains the C++ exception's what() string.
"""
struct CxxException <: Exception
    message::String
end

Base.showerror(io::IO, e::CxxException) = print(io, "C++ exception: ", e.message)

"""
    _check_pending_exception()

Check if a C++ exception was caught by the last JIT call and throw it as CxxException.
Called after every JIT invoke to propagate C++ exceptions to Julia.
"""
@inline function _check_pending_exception()
    if MLIRNative.has_pending_exception()
        msg = MLIRNative.get_pending_exception()
        MLIRNative.clear_pending_exception()
        throw(CxxException(msg))
    end
end

# =============================================================================
# Fast function pointer lookup with lock-free read path
# =============================================================================

"""
    _lookup_cached(func_name::String) -> Ptr{Cvoid}

Look up a JIT function pointer with caching.
Fast path: atomic snapshot read of an immutable Dict copy — no lock needed.
Slow path: JIT engine lookup + copy-on-write Dict swap under lock.
"""
@inline function _lookup_cached(func_name::String)::Ptr{Cvoid}
    # Fast path: read an atomic snapshot of the Dict reference.
    # Thread safety relies on copy-on-write: the slow path creates a NEW Dict
    # via copy(), mutates the copy, then atomically publishes it. Published
    # Dicts are never mutated, so readers always see a fully-constructed,
    # stable hash table. Julia's @atomic provides seq_cst ordering, ensuring
    # all mutations to the new Dict are visible before the reference is published.
    snapshot = @atomic GLOBAL_JIT.compiled_symbols
    ptr = get(snapshot, func_name, C_NULL)
    if ptr != C_NULL
        return ptr
    end

    # Slow path: look up in JIT engine and publish a new Dict copy
    lock(GLOBAL_JIT.lock) do
        # Double-check after acquiring lock (re-read atomic)
        current = @atomic GLOBAL_JIT.compiled_symbols
        ptr = get(current, func_name, C_NULL)
        if ptr != C_NULL
            return ptr
        end

        # Search every library engine — thunk names are mangled-derived and
        # unique across libraries, so the first hit is the right one.
        for eng in GLOBAL_JIT.engines
            eng.jit_engine === nothing && continue
            ptr = MLIRNative.lookup(eng.jit_engine, func_name)
            if ptr == C_NULL
                ptr = MLIRNative.lookup(eng.jit_engine, "_" * func_name)
            end
            ptr != C_NULL && break
        end
        if ptr == C_NULL && endswith(func_name, "_thunk")
            # The thunk exists under its in-place name: the wrapper predates
            # by-reference returns built in the caller's buffer.
            for eng in GLOBAL_JIT.engines
                eng.jit_engine === nothing && continue
                if MLIRNative.lookup(eng.jit_engine, func_name * "_inplace") != C_NULL ||
                   MLIRNative.lookup(eng.jit_engine, "_" * func_name * "_inplace") != C_NULL
                    throw(ErrorException(_stale_inplace_message(func_name)))
                end
            end
        end
        if ptr == C_NULL
            searched = isempty(GLOBAL_JIT.engines) ? "none" :
                join((basename(e.binary_path) *
                      (e.init_error === nothing ? "" : " [init failed: $(sprint(showerror, e.init_error))]")
                      for e in GLOBAL_JIT.engines), ", ")
            throw(ErrorException("JIT Error: Symbol not found: $func_name. " *
                "Engines searched: $searched. This may indicate the owning library's " *
                "JIT failed to initialize, a missing library, or a complex C++ type " *
                "that failed to compile through the MLIR backend."))
        end

        # Copy-on-write: create a new Dict so readers on the fast path
        # never observe a half-mutated hash table.
        updated = copy(current)
        updated[func_name] = ptr
        @atomic GLOBAL_JIT.compiled_symbols = updated
        return ptr
    end
end

"""
    return_slot(::Type{T}) -> slot

A home for a `T` the callee builds in place: a thunk `<m>_thunk_inplace`
(`FunctionGen.returns_in_caller_buffer`) takes `slot_address(slot)` as its first
argument and constructs the C++ object there, and `slot_value(slot)` is the Julia
value the wrapper returns. A `Ref{T}` for an isbits `T`; for a `mutable` `T` (a
named union or bitfield record, whose one field is its C bytes) the object
itself, zeroed, as `_invoke_call` allocates one. The generated call keeps the
slot alive with `GC.@preserve` across the call, so the address does not move
while C++ builds the object.
"""
@generated function return_slot(::Type{T}) where T
    if ismutabletype(T)
        if !(isconcretetype(T) && fieldcount(T) > 0 && sizeof(T) > 0 &&
             all(i -> isbitstype(fieldtype(T, i)), 1:fieldcount(T)))
            msg = "JIT invoke: cannot return `$T` in place: it is a mutable type " *
                  "without an inline byte layout to fill."
            return :(error($msg))
        end
        n = sizeof(T)
        return quote
            obj = ccall(:jl_new_struct_uninit, Any, (Any,), $T)::$T
            GC.@preserve obj ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t),
                                   pointer_from_objref(obj), 0, $n)
            obj
        end
    end
    if !isbitstype(T)
        msg = "JIT invoke: cannot return `$T` in place: it has no fixed C layout."
        return :(error($msg))
    end
    return :(Ref{$T}())
end
slot_address(slot::Base.RefValue) = Base.unsafe_convert(Ptr{Cvoid}, slot)
slot_address(slot) = pointer_from_objref(slot)
slot_value(slot::Base.RefValue) = slot[]
slot_value(slot) = slot

"""
    _stale_inplace_message(name) -> String

What a call to the thunk `name` means when the library has `<name>_inplace`
instead: the wrapper was generated before a pass-by-reference return was built
in the caller's buffer (`FunctionGen.returns_in_caller_buffer`, 2026-11-13),
and calls the thunk with the old shape. The new name exists so that call fails
here instead of passing its arguments where the buffer goes.
"""
_stale_inplace_message(name::AbstractString) =
    "thunk `$(name)` not found, but `$(name)_inplace` is: this wrapper was generated " *
    "before RepliBuild built by-reference returns in the caller's buffer, and calls " *
    "the thunk with the old shape. Re-wrap it with the RepliBuild now loaded."

@noinline function _jit_not_initialized_error()
    msg = "JIT not initialized."
    failed = [e for e in GLOBAL_JIT.engines if e.init_error !== nothing]
    if !isempty(failed)
        msg *= " Root cause: " * join(
            ("$(basename(e.binary_path)): $(sprint(showerror, e.init_error))" for e in failed), "; ")
    elseif GLOBAL_JIT.init_error !== nothing
        msg *= " Root cause: $(GLOBAL_JIT.init_error)"
    end
    error(msg)
end

# =============================================================================
# Arity-specialized invoke methods (zero heap allocation)
# =============================================================================

# MLIR ciface calling convention:
#   Scalar return (i32, f64, etc.):  T    ciface(args_ptr)     — direct return
#   Struct return:                   void ciface(T* sret, args_ptr) — sret convention
#   Void return:                     void ciface(args_ptr)

"""
    _invoke_call(fptr, ::Type{T}, inner_ptrs)

Call JIT function with correct ABI. Uses @generated to resolve ccall return type
at compile time (ccall requires a concrete type, not a TypeVar).
"""
@generated function _invoke_call(fptr::Ptr{Cvoid}, ::Type{T}, inner_ptrs::Vector{Ptr{Cvoid}}) where T
    if T === Any
        # An unresolved wrapper ret type would take the sret path below with
        # Ref{Any}() — an undefined reference the JIT then scribbles raw bytes
        # into. Refuse loudly with the actual cause instead.
        return :(error("JIT invoke: the generated wrapper's return type is the " *
                       "unresolved `Any` sentinel — its C type was not mapped " *
                       "(enum/struct missing from DWARF metadata?). Re-wrap with " *
                       "a current generator or add the type mapping."))
    end
    if isprimitivetype(T)
        # Scalar return: T ciface(void** args_ptr) — direct return
        if Sys.iswindows()
            # Catch in libJLCS; in-JIT landing pads AV under SEH (see JlcsJIT.cpp).
            # Every width the guard set can actually carry, and NOTHING ELSE.
            # `isprimitivetype` admits more than i32/i64/f32/f64/ptr — Compiler's
            # C map turns `__int128`/`__uint128_t` into Int128/UInt128 at 16
            # bytes — and the old `else => jlcs_guard_i64` called such a return
            # through an `int64_t (*)(void **)` and TRUNCATED it, silently, on
            # Windows only: Linux calls the thunk directly and returns all 128
            # bits. A width this set cannot express has to say so.
            g = if T <: AbstractFloat && sizeof(T) == 4
                :jlcs_guard_f32
            elseif T <: AbstractFloat && sizeof(T) == 8
                :jlcs_guard_f64
            elseif T <: Ptr
                :jlcs_guard_ptr
            elseif sizeof(T) <= 4
                :jlcs_guard_i32
            elseif sizeof(T) == 8
                :jlcs_guard_i64
            else
                nothing
            end
            if g === nothing
                msg = "JIT invoke: the Windows guard set has no form for a " *
                      "$(sizeof(T))-byte $T return. Windows routes every thunk " *
                      "call through libJLCS's jlcs_guard_* so the frame that " *
                      "catches is C++ — an in-JIT landing pad AVs under SEH, see " *
                      "JlcsJIT.cpp — and that set is i32/i64/f32/f64/ptr. Falling " *
                      "back to jlcs_guard_i64 would truncate this silently and " *
                      "disagree with Linux, which calls the thunk directly. Add a " *
                      "guard of this width to JLCSCAPIWrappers.cpp."
                return :(error($msg))
            end
            lib = GlobalRef(MLIRNative, :libJLCS)
            return :(ccall(($(QuoteNode(g)), $lib), $T,
                           (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), fptr, inner_ptrs))
        end
        return :(ccall(fptr, $T, (Ptr{Ptr{Cvoid}},), inner_ptrs))
    else
        # Struct return: void ciface(T* sret, void** args_ptr) — sret convention
        if ismutabletype(T)
            # A named union or a bitfield record is a `mutable struct` holding
            # its C bytes inline. `Ref{T}()` would hold an undefined reference,
            # and converting it for the sret slot threw UndefRefError on every
            # such return (abi_fuzz X1). Allocate the object, zero it, and give
            # the thunk the address of its fields.
            if !(isconcretetype(T) && fieldcount(T) > 0 && sizeof(T) > 0 &&
                 all(i -> isbitstype(fieldtype(T, i)), 1:fieldcount(T)))
                msg = "JIT invoke: cannot return `$T` by value: it is a mutable " *
                      "type without an inline byte layout to fill."
                return :(error($msg))
            end
            n = sizeof(T)
            call = if Sys.iswindows()
                lib = GlobalRef(MLIRNative, :libJLCS)
                :(ccall((:jlcs_guard_sret, $lib), Cvoid,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), fptr, p, inner_ptrs))
            else
                :(ccall(fptr, Cvoid, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), p, inner_ptrs))
            end
            return quote
                ret_obj = ccall(:jl_new_struct_uninit, Any, (Any,), $T)::$T
                GC.@preserve ret_obj begin
                    p = pointer_from_objref(ret_obj)
                    ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), p, 0, $n)
                    $call
                end
                ret_obj
            end
        end
        if Sys.iswindows()
            lib = GlobalRef(MLIRNative, :libJLCS)
            return quote
                ret_buf = Ref{$T}()
                GC.@preserve ret_buf begin
                    ccall((:jlcs_guard_sret, $lib), Cvoid,
                          (Ptr{Cvoid}, Ptr{$T}, Ptr{Ptr{Cvoid}}),
                          fptr, ret_buf, inner_ptrs)
                end
                ret_buf[]
            end
        end
        return quote
            ret_buf = Ref{$T}()
            GC.@preserve ret_buf begin
                ccall(fptr, Cvoid, (Ptr{$T}, Ptr{Ptr{Cvoid}}), ret_buf, inner_ptrs)
            end
            ret_buf[]
        end
    end
end

"""
    _invoke_void(fptr, inner_ptrs)

The void-return call, `void ciface(void** args)`: [`_invoke_call`](@ref)'s
counterpart, shared the same way by `invoke` and both AOT forms.

On Windows it goes through libJLCS's `jlcs_guard_void`, for the reason every
`_invoke_call` return does. The dialect lowers `try_call` to a plain call there
(an in-JIT landing pad AVs under SEH, see JlcsJIT.cpp), so the thunk does not
catch, and the guard is the C++ frame that does. That lowering is the same for a
thunk compiled into an AOT `_thunks` library. The AOT void forms used to call the
thunk directly, so a void C++ function that threw unwound into Julia and the
process died with no message (exit status 0x20474343, the SEH code of an
uncaught C++ exception), where the JIT `invoke` raised CxxException for the same
throw (`test_aot_thunks.jl`, probe `throw_void`).
"""
@generated function _invoke_void(fptr::Ptr{Cvoid}, inner_ptrs::Vector{Ptr{Cvoid}})
    if Sys.iswindows()
        lib = GlobalRef(MLIRNative, :libJLCS)
        return :(ccall((:jlcs_guard_void, $lib), Cvoid,
                       (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), fptr, inner_ptrs))
    end
    return :(ccall(fptr, Cvoid, (Ptr{Ptr{Cvoid}},), inner_ptrs))
end

# =============================================================================
# Pointer arguments: what a pointer position accepts
#
# A thunk loads a pointer parameter from its slot and hands it to C++. Nothing
# below this point can check it, and the wrapper's signature for a pointer is
# `::Any` (an owning object, a `Ref`, a `Ptr` all go there), so the argument is
# checked once, here: a generated Tier 2 body converts every argument bound to
# a pointer, a reference, a `this` or a `char*` with
# `pointer_arg(<declared type>, x)`, as `ccall` converts to the type its tuple
# declares. What is not a pointer to the declared thing is a `MethodError`
# before any C++ runs. Until 2026-11-17 the argument went to `invoke` as it
# came and was boxed: a struct value, a Julia function or a `Vector` passed for
# an object was read as an address.
# =============================================================================

"""
The pointee `pointer_arg` is told for a C++ object the wrapper has no Julia
type for: an STL container or `std::string` behind a pointer or reference, a
class only declared, the `this` of a class nothing emits (`Ptr{Opaque}`). No
Julia array is one, so none passes there.
"""
abstract type Opaque end

"The pointee `pointer_arg` is told for a function pointer (`Ptr{Code}`): a `Ptr` or a `@cfunction` closure."
abstract type Code end

"""
What `pointer_arg` is told for a pointer to plain `char` that is not `const`:
a buffer the function may write into (`char_buffer_arg`). It takes what a
`Cstring` position takes except a Julia string, a `Symbol` and `codeunits` of a
string, which are an `ArgumentError`.
"""
abstract type CharBuffer end

# A pointer to plain `char`, read-only (`Cstring`) or written (`CharBuffer`):
# both take bytes of either sign, and no object.
_char_pointee(::Type{P}) where {P} = P === Cstring || P === CharBuffer

"""
    ByValue{V}

What `pointer_arg` is told for a parameter C++ takes BY VALUE and the thunk
takes by address: an STL container, which C++ passes as a pointer to the
caller's temporary. `V` is the Julia value the wrapper returns such a container
as (its bytes). The value itself passes, by the address of a copy of those
bytes, and so does a pointer or `Ref` to an existing object.
"""
abstract type ByValue{V} end

"""
    PointerArg

One pointer argument, converted: the address the thunk's slot holds, and
`keep`, the Julia object that owns the storage behind it. `invoke` preserves
the `PointerArg` across the call.
"""
struct PointerArg{K}
    ptr::Ptr{Cvoid}
    keep::K
end

# Plain C data a `Ref` may point at: bits stored inline, or a record the
# wrapper emits as a `mutable struct` of its C bytes (a named union, a
# bitfield record), whose address is the address of those bytes.
_is_byte_record(::Type{T}) where {T} =
    isconcretetype(T) && ismutabletype(T) && fieldcount(T) == 1 &&
    fieldtype(T, 1) <: Tuple{Vararg{UInt8}} && sizeof(T) > 0
_plain_data(::Type{T}) where {T} = (isbitstype(T) && sizeof(T) > 0) || _is_byte_record(T)

# Arrays whose memory Base hands out a pointer to (`cconvert`/`unsafe_convert`).
# An `AbstractArray` outside this set (an STL wrapper's `CppVector`) is an
# object: it passes as what it converts to, never as data.
const _MemoryArray = Union{DenseArray, SubArray, Base.ReshapedArray, Base.ReinterpretArray}

# Whether an array of `E` is data for the declared type `P`: a `char*` takes
# bytes of either sign, `T*` only `T`s, and `T**` pointers of any pointee, as a
# pointer parameter takes them. (`void*` takes any array and never asks.)
function _array_matches(::Type{P}, ::Type{E}) where {P, E}
    bytes = E === UInt8 || E === Int8
    _char_pointee(P) && return bytes
    P <: Ptr || return false
    T = eltype(P)
    return T === E || (bytes && (T === UInt8 || T === Int8)) || (T <: Ptr && E <: Ptr)
end

@noinline _refuse_pointer_arg(::Type{P}, x) where {P} = throw(MethodError(pointer_arg, (P, x)))

# What a refusal means, printed under the `MethodError` it is.
function _pointer_arg_hint(io, exc, argtypes, kwargs)
    exc.f === pointer_arg || return nothing
    print(io, "\nA wrapper refused this argument where C++ takes a pointer. A pointer parameter takes a `Ptr`, ",
          "a `Ref` to plain data (`Ref(x)`, never the bare value), an object that owns one (a `Managed` ",
          "object, a factory struct), or an array of the pointee's type where C++ reads data; a `const char*` ",
          "also takes a string.")
    return nothing
end

function __init__()
    Base.Experimental.register_error_hint(_pointer_arg_hint, MethodError)
    return nothing
end

"""
    pointer_arg(P, x) -> PointerArg

The one answer to "may `x` be passed where C++ declares `P`", for every pointer
position of a Tier 2 call, and the conversion when it may. `P` is what the
wrapper says the position is: `Ptr{T}` for a pointer or reference to a `T` the
wrapper emits (`this` included), `Ptr{Cvoid}` for `void*`, `Cstring` for a
`const char*`, `CharBuffer` for a `char*` that is not `const`, `Ptr{Opaque}`,
`Ptr{Code}`, or `ByValue{V}`.

Accepted:

  * a `Ptr`, of any pointee (`C_NULL`; a derived object's pointer for its
    base's);
  * a `Ref` or `Ref(array, i)` to plain C data: an isbits value or a record
    held as its bytes (`_plain_data`). A `Ref{String}` or `Ref{Any}` holds a
    Julia object, not C data. `void*` takes any `Ref`, as `ccall` does;
  * an array of the pointee's type where the pointee is data (`T*` takes
    `T`s, a `char*` bytes, `T**` pointers, `void*` anything), through Base's
    own `cconvert`/`unsafe_convert`;
  * a record held as its bytes (`mutable struct`), by the address of those
    bytes;
  * for `Cstring`, an `AbstractString` with Julia's `Cstring` semantics: a
    `SubString` is copied and NUL-terminated, an embedded NUL is an
    `ArgumentError`. A byte pointer that is not a C string (`unsigned char*`)
    takes a string as `ccall` passes one to `Ptr{UInt8}`. For `CharBuffer`
    (a `char*` that is not `const`), a string, a `Symbol` or `codeunits` of a
    string is an `ArgumentError` (`char_buffer_arg`); bytes pass as for
    `Cstring`;
  * for `ByValue{V}`, a `V`;
  * anything else that converts with `Base.unsafe_convert(Ptr{Cvoid}, x)`: the
    wrapper's owning types (`Managed*`, a factory struct), the STL wrappers,
    a `@cfunction` closure. Never for a `char*`: an object is not a C string.

Everything else is a `MethodError` of this function, with a hint under it
that says what a pointer parameter takes (`_pointer_arg_hint`).
"""
pointer_arg(::Type{P}, x) where {P} = _pointer_arg(P, x)
function pointer_arg(::Type{ByValue{V}}, x) where {V}
    x isa V || return _pointer_arg(Ptr{Opaque}, x)
    r = Ref{V}(x)
    return PointerArg(Base.unsafe_convert(Ptr{Cvoid}, r), r)
end

_pointer_arg(::Type{P}, x::Ptr) where {P} = PointerArg(Ptr{Cvoid}(x), nothing)

function _pointer_arg(::Type{P}, x::Union{Base.RefValue{T}, Base.RefArray{T}}) where {P, T}
    (P === Ptr{Cvoid} || (P !== Ptr{Code} && _plain_data(T))) || _refuse_pointer_arg(P, x)
    return PointerArg(Ptr{Cvoid}(Base.unsafe_convert(Ptr{Cvoid}, x)), x)
end

function _pointer_arg(::Type{P}, x::_MemoryArray) where {P}
    P === CharBuffer && x isa Base.CodeUnits && _refuse_string_at_char_buffer(x)
    E = eltype(x)
    (P === Ptr{Cvoid} || (isbitstype(E) && sizeof(E) > 0 && _array_matches(P, E))) || _refuse_pointer_arg(P, x)
    cx = Base.cconvert(Ptr{E}, x)
    return PointerArg(Ptr{Cvoid}(Base.unsafe_convert(Ptr{E}, cx)), cx)
end

"""
    cstring_arg(x)

What a `ccall` site passes for a parameter bound to a pointer to plain `char`
(`Wrapper._points_to_plain_char`), whose ccall type is a byte pointer. An
`AbstractString` gets Julia's `Cstring` semantics, as `pointer_arg(Cstring, x)`
gives it on Tier 2: a `SubString` (or any string that is not a `String`) is
copied, which NUL-terminates it, and an embedded NUL is an `ArgumentError`. The
`String` returned is what `ccall` converts to the byte pointer, and roots.
A `Cstring` passes as its pointer. Anything else (a `Ptr`, `C_NULL`, a `Ref`,
a byte vector, a `Symbol`) is returned as it came, for `ccall` to convert as
it always did.

**A `String` is returned as the same object, and the call site must hold
nothing new for it.** Some C functions never return to their caller: Lua's
`luaL_error`, libpng's `png_error` and their like leave by `longjmp`, and are
called from a Julia callback to raise the library's error. The Julia frames
between the jump and its landing are abandoned without their GC frames being
popped, so a wrapper that rooted anything across the call leaves the task's
GC stack pointing into dead stack memory, and the next collection reads it
(the Hub's lua, 2026-11-21: a segfault in the collector). So for a `String`
this method inlines to a scan and the argument itself: the check's `throw`
is out of line, in a function that takes the caller's own argument, and the
wrapper compiles to what a bare `ccall` on the string did. A string of any
other type is copied and its copy has to be rooted; pass a `String` to a
function that leaves by `longjmp`.
"""
cstring_arg(x) = x
@inline function cstring_arg(x::String)
    _has_nul(x) && _refuse_embedded_nul(x)
    return x
end
function cstring_arg(x::AbstractString)
    cx = String(x)
    _has_nul(cx) && _refuse_embedded_nul(cx)
    return cx
end
cstring_arg(x::Cstring) = pointer(x)

"""
    char_buffer_arg(x)

What a `ccall` site passes for a parameter bound to a pointer to plain `char`
that is not `const` (`Wrapper._points_to_writable_char`): a buffer the function
may write into. A Julia string is refused there with an `ArgumentError` that
says what to pass instead: a `String` is immutable and may be shared (a
literal in compiled code is one object for every call), and the function would
write into it in place. So are a `SubString`, a `Symbol` (an interned name) and
`codeunits` of a string, which point at the same bytes. Everything else is
`cstring_arg`'s: a byte vector, a `Ptr`, `C_NULL`, a `Ref`, a `Cstring` pass as
they did.

The refusal is out of line, as `cstring_arg`'s is, so a `Ptr`, a buffer or an
integer argument gives the wrapper nothing new to hold across the call (a
function that leaves by `longjmp`; `cstring_arg`'s docstring).
"""
char_buffer_arg(x) = cstring_arg(x)
char_buffer_arg(x::Union{AbstractString, Symbol, Base.CodeUnits}) = _refuse_string_at_char_buffer(x)

@noinline _refuse_string_at_char_buffer(x) = throw(ArgumentError(
    "a $(x isa Base.CodeUnits ? "`codeunits` of a string" : "`$(nameof(typeof(x)))`") cannot be passed where " *
    "C takes a `char *` it may write into (not `const char *`): a Julia string is immutable and may be shared " *
    "(a string literal is one object for every call), and the function would change it in place. Pass a byte " *
    "buffer it can write into, NUL-terminated and as large as the function needs, for example " *
    "`buf = Vector{UInt8}(\"text\\0\")` or `buf = zeros(UInt8, 256)`, and read the text back with " *
    "`GC.@preserve buf unsafe_string(pointer(buf))`."))

# One `memchr` over the string's bytes, as Base's own `Cstring` conversion does.
@inline _has_nul(s::String) =
    ccall(:memchr, Ptr{Cvoid}, (Ptr{UInt8}, Cint, Csize_t), s, 0, sizeof(s)) != C_NULL

# Out of line on purpose: building the message allocates, and inlined into a
# wrapper that would give the wrapper a GC frame on every path (see above).
@noinline _refuse_embedded_nul(s::String) = throw(ArgumentError(
    "embedded NULs are not allowed in C strings: $(repr(s)). " *
    "For bytes passed with their length, pass `codeunits(s)` or a byte vector."))

function _pointer_arg(::Type{P}, x::AbstractString) where {P}
    P === CharBuffer && _refuse_string_at_char_buffer(x)
    if P === Cstring
        cx = cstring_arg(x)
        return PointerArg(Ptr{Cvoid}(pointer(cx)), cx)
    elseif P === Ptr{UInt8} || P === Ptr{Int8}
        cx = Base.cconvert(P, x)
        return PointerArg(Ptr{Cvoid}(Base.unsafe_convert(P, cx)), cx)
    end
    _refuse_pointer_arg(P, x)
end

_pointer_arg(::Type{P}, x::Cstring) where {P} =
    (_char_pointee(P) || P === Ptr{UInt8} || P === Ptr{Int8}) ? PointerArg(Ptr{Cvoid}(pointer(x)), nothing) :
                                                              _refuse_pointer_arg(P, x)

# A `Symbol` is a C string to Julia (`ccall` passes one for `Cstring`).
_pointer_arg(::Type{P}, x::Symbol) where {P} =
    P === Cstring ? PointerArg(Ptr{Cvoid}(pointer(Base.unsafe_convert(Cstring, x))), x) :
    P === CharBuffer ? _refuse_string_at_char_buffer(x) : _refuse_pointer_arg(P, x)

# Everything else: an object, by its bytes' address or by what it converts to.
# `applicable` is decided when this is compiled for `typeof(x)`, so the test
# costs a call nothing, and a value with no conversion is refused here, like
# every other refusal, instead of as a `MethodError` of `unsafe_convert`.
function _pointer_arg(::Type{P}, x) where {P}
    (_char_pointee(P) || (P === Ptr{Code} && !(x isa Base.CFunction))) && _refuse_pointer_arg(P, x)
    _is_byte_record(typeof(x)) && return PointerArg(pointer_from_objref(x), x)
    applicable(Base.unsafe_convert, Ptr{Cvoid}, x) || _refuse_pointer_arg(P, x)
    return PointerArg(Ptr{Cvoid}(Base.unsafe_convert(Ptr{Cvoid}, x)), x)
end

"""
    _arg_marshal_plan(argtypes) -> (setup, ptr_exprs, preserve_syms)

Build the argument-packing prologue shared by both `invoke` methods.

A thunk slot must hold a pointer to a location *containing* the argument value:
the emitted thunk double-loads, slot → storage → value. `Ref(x)` gives exactly
that for an isbits `x`. Two argument kinds are **already** an indirection and
must be flattened to a raw pointer first, or the double-load spends one level
too many and the callee receives the payload's own bytes as an address:

  * `AbstractString` — `Ref(::String)` points at the String object, not at its
    bytes (found live on tinyxml2 `XMLDocument::Parse`).
  * `Base.Ref` that is not already a `Ptr` — the spelling a Julia caller reaches
    for first when a parameter is annotated `::Ref{T}`, which is what a C++
    `T const&` generates. `Ptr{T} <: Ref{T}`, so the annotation accepts a
    `RefValue` silently and nothing type-checks it away; the thunk then reads
    the struct's first 8 bytes as an address. Presents as a SIGSEGV with a C++
    frame in the backtrace, so it reads as a thunk ABI bug rather than a caller
    mistake (found on `ImGui::ButtonEx` via `ImVec2 const&`, 2026-08-09).

The `!(<: Ptr)` term is load-bearing: a raw pointer is already the flattened
form and re-flattening it would strip the level the thunk needs.

In both cases the original is GC-preserved across the call, since only a raw
pointer into it reaches the slot.

Those two flattenings guess from the argument alone, because the plan does not
know what the position is. A wrapper generated since 2026-11-17 does not rely
on them: it converts every pointer argument itself with [`pointer_arg`](@ref),
which knows the declared type and refuses what is not a pointer to it, and
hands the plan a `PointerArg`: the slot holds its address and the `PointerArg`
(with the object it keeps) is preserved. The guesses stay for wrappers
generated before that, which pass their arguments as they came.

Kept as ONE plan consumed by both `invoke` methods — they carried byte-identical
copies of this, which is how a fix to one silently misses the other.
"""
function _arg_marshal_plan(argtypes)
    N = length(argtypes)
    ref_syms = [Symbol("r$i") for i in 1:N]
    src_syms = [Symbol("s$i") for i in 1:N]

    is_ptr_arg(T)  = T <: PointerArg
    is_str(T)      = T <: AbstractString
    is_boxed_ref(T) = T <: Base.Ref && !(T <: Ptr)
    indirect(T)    = is_ptr_arg(T) || is_str(T) || is_boxed_ref(T)

    setup = map(1:N) do i
        T = argtypes[i]
        if is_ptr_arg(T)
            :($(src_syms[i]) = args[$i]; $(ref_syms[i]) = Ref($(src_syms[i]).ptr))
        elseif is_str(T)
            :($(src_syms[i]) = args[$i]; $(ref_syms[i]) = Ref(pointer($(src_syms[i]))))
        elseif is_boxed_ref(T)
            :($(src_syms[i]) = args[$i];
              $(ref_syms[i]) = Ref(Base.unsafe_convert(Ptr{Cvoid}, $(src_syms[i]))))
        else
            :($(ref_syms[i]) = Ref(args[$i]))
        end
    end
    ptrs = [:(Base.unsafe_convert(Ptr{Cvoid}, $(ref_syms[i]))) for i in 1:N]
    preserve = vcat(ref_syms, [src_syms[i] for i in 1:N if indirect(argtypes[i])])
    return (setup, ptrs, preserve)
end

"""
    marshal_args(args...) -> (keepalive, inner_ptrs)

Marshal `args` into the `void**` a `_mlir_ciface_*` thunk expects, and return
the packed pointers together with everything that must outlive the call.

    keep, inner_ptrs = marshal_args(a, b, c)
    GC.@preserve keep inner_ptrs begin
        ccall(fptr, T, (Ptr{Ptr{Cvoid}},), inner_ptrs)
    end

This exists because there are two tiers that call the same thunks and they must
agree on how an argument becomes a pointer. `invoke` marshals through
[`_arg_marshal_plan`](@ref); the AOT dispatch path open-coded its own
`cconvert`/`unsafe_convert` sequence, and the copies drifted — the plan learned
`AbstractString` and boxed `Ref`, the open-coded version did not, so an AOT
wrapper died on `unsafe_convert(Ptr{UInt8}, ::Cstring)` for any string argument
while the JIT wrapper handled it. One plan, two callers, no second copy to
forget.

`@generated` for the same reason `invoke` is: the wrappers declare their
parameters `::Any`, so the argument's real type is knowable only per call site.
A generator emitting text cannot see it, which is why this could not be fixed
where the bug appeared to live.
"""
@generated function marshal_args(args::Vararg{Any, N}) where N
    (setup, ptrs, preserve_args) = _arg_marshal_plan(args)
    quote
        $(setup...)
        inner_ptrs = Ptr{Cvoid}[$(ptrs...)]
        # The tuple roots every source and Ref; preserving it at the call site
        # keeps them alive for exactly as long as inner_ptrs is used.
        return (($(preserve_args...),), inner_ptrs)
    end
end

"""
    invoke(func_name::String, ::Type{T}, args...) where T

Invoke a JIT-compiled function with return type T.
@generated: emits arity-specialized code for any N at compile time.
For N=1..4 this produces identical code to the old hand-written methods.
For N≥5 this eliminates the Vector{Any}/Vector{Ptr{Cvoid}} allocation
that the old generic fallback incurred on every call.
"""
@generated function invoke(func_name::String, ::Type{T}, args::Vararg{Any, N}) where {T, N}
    (setup, ptrs, preserve_args) = _arg_marshal_plan(args)

    quote
        (@atomic GLOBAL_JIT.initialized) || _jit_not_initialized_error()
        fptr = _lookup_cached(func_name)
        $(setup...)
        inner_ptrs = Ptr{Cvoid}[$(ptrs...)]
        result = GC.@preserve $(preserve_args...) begin
            _invoke_call(fptr, T, inner_ptrs)
        end
        _check_pending_exception()
        return result
    end
end

# =============================================================================
# Void-return invoke (no Type parameter = void return)
# =============================================================================

@generated function invoke(func_name::String, args::Vararg{Any, N}) where N
    (setup, ptrs, preserve_args) = _arg_marshal_plan(args)

    quote
        (@atomic GLOBAL_JIT.initialized) || _jit_not_initialized_error()
        fptr = _lookup_cached(func_name)
        $(setup...)
        inner_ptrs = Ptr{Cvoid}[$(ptrs...)]
        GC.@preserve $(preserve_args...) inner_ptrs begin
            _invoke_void(fptr, inner_ptrs)
        end
        _check_pending_exception()
        return nothing
    end
end

# =============================================================================
# AOT thunk invocation
#
# The same thunks, reached without a JIT. `build_aot_thunks` compiles the MLIR
# at wrap time into `lib<name>_thunks.so`, so the wrapper dlopens that instead
# of constructing an MLIR module in every process — the difference is ~34s of
# startup for a large library, and nothing else. The thunk is byte-for-byte the
# IR the JIT would have built; only the way its address is found differs.
#
# Which is exactly why these go through `_arg_marshal_plan` and `_invoke_call`
# rather than reimplementing them. The AOT path used to have private copies of
# both, and both drifted:
#
#   * marshalling forgot `AbstractString` and boxed `Ref`, so a string argument
#     raised `unsafe_convert(Ptr{UInt8}, ::Cstring)`;
#   * the return convention was picked by matching the type's NAME against a
#     hardcoded list of scalar spellings. `isprimitivetype(XMLError)` is true —
#     RepliBuild emits enums as real primitive types — but the string
#     "XMLError" is in no list and never would be, so an `i32 (void**)` thunk
#     got called as `void (i32*, void**)`. That reads args_ptr out of the
#     register holding the return buffer: a segfault inside the thunk, for
#     every enum-returning function in every C++ package.
#
# Two bugs, one cause: a decision the type system answers exactly, approximated
# with string matching. Sharing the implementation is the fix; there is no
# version of the duplicate that stays correct.
# =============================================================================

const _AOT_SYMBOLS = Dict{Tuple{Ptr{Cvoid},String},Ptr{Cvoid}}()
const _AOT_SYMBOL_LOCK = ReentrantLock()

"""
Address of `name` in the thunks library behind `handle`, cached.

`dlsym` is cheap but not free, and a thunk call site hits this on every
invocation; the JIT path caches its lookups for the same reason.
"""
function _lookup_aot(handle::Ptr{Cvoid}, name::String)
    handle == C_NULL && error(
        "AOT thunks library is not loaded (THUNKS_HANDLE is null). The wrapper " *
        "was generated with `aot_thunks = true` but lib*_thunks.so did not " *
        "open — run wrap() again, which compiles it, or set aot_thunks = false " *
        "to use the JIT.")
    lock(_AOT_SYMBOL_LOCK) do
        get!(_AOT_SYMBOLS, (handle, name)) do
            p = Libdl.dlsym(handle, name; throw_error = false)
            p === nothing && error(
                "AOT thunk `$name` is missing from the thunks library. It is " *
                "generated from the same MLIR the JIT would build, so a gap " *
                "here means the thunks .so is stale — rebuild the package.")
            p
        end
    end
end

"""
    invoke_aot(handle, func_name::String, ::Type{T}, args...) where T

Call an AOT-compiled thunk with return type `T`. Counterpart to [`invoke`](@ref)
and deliberately identical to it apart from symbol resolution.
"""
@generated function invoke_aot(handle::Ptr{Cvoid}, func_name::String,
                               ::Type{T}, args::Vararg{Any, N}) where {T, N}
    (setup, ptrs, preserve_args) = _arg_marshal_plan(args)
    quote
        fptr = _lookup_aot(handle, func_name)
        $(setup...)
        inner_ptrs = Ptr{Cvoid}[$(ptrs...)]
        result = GC.@preserve $(preserve_args...) begin
            _invoke_call(fptr, T, inner_ptrs)
        end
        _check_pending_exception()
        return result
    end
end

"""
    invoke_aot(handle, func_name::String, args...)

Void-return form — no `Type` parameter, same as [`invoke`](@ref)'s.
"""
@generated function invoke_aot(handle::Ptr{Cvoid}, func_name::String,
                               args::Vararg{Any, N}) where N
    (setup, ptrs, preserve_args) = _arg_marshal_plan(args)
    quote
        fptr = _lookup_aot(handle, func_name)
        $(setup...)
        inner_ptrs = Ptr{Cvoid}[$(ptrs...)]
        GC.@preserve $(preserve_args...) inner_ptrs begin
            _invoke_void(fptr, inner_ptrs)
        end
        _check_pending_exception()
        return nothing
    end
end

# =============================================================================
# Eagerly-resolved AOT dispatch
#
# `_lookup_aot` exists because the JIT resolves symbols lazily — a thunk's
# address does not exist until the JIT has compiled it, so the first call is the
# earliest moment a lookup can succeed, and caching it is the best available.
# An AOT thunks library is fully linked the moment `__init__` dlopens it, and the
# wrapper knows every thunk name at generation time, so neither the laziness nor
# the cache buys anything on that path: it pays a lock acquire plus a Dict lookup
# that re-hashes a ~40-char symbol name on EVERY call, forever, to answer a
# question settled at load.
#
# Measured on hello_world (1M calls, best of 7): `_lookup_aot` is 37.3 ns of a
# 53.6 ns call — the MLIR thunk itself is 3.5 ns against a 1.1 ns bare ccall, and
# the returned String is 11.3 ns. So ~70% of a Tier-2 call was symbol resolution.
# Resolving in `__init__` and passing the pointer takes hello_message to ~15 ns
# and its `_ptr` sibling to ~3.5 ns.
#
# What deliberately does NOT change: `_arg_marshal_plan`, `_invoke_call` and
# `_check_pending_exception` are shared verbatim. The comment above `invoke_aot`
# records what happened the last time this path kept private copies of the first
# two — a string argument that raised, and an enum return that segfaulted through
# a mismatched call convention. Symbol resolution is the one part that path
# already names as legitimately its own, which is why it is the part that moves.
# =============================================================================

@noinline function _aot_fptr_null()
    error("AOT thunk pointer is null: this function's thunk was not resolved " *
          "when the wrapper loaded. The wrapper's `__init__` warns at load " *
          "naming every thunk it could not find — the thunks library is stale " *
          "or was built from different sources. Run wrap() again; it compiles " *
          "the thunks library.")
end

"""
    open_thunks_library(thunks_path) -> Ptr{Cvoid}

Open an AOT thunks library, with libJLCS loaded first.

**The thunks library and `MLIRNative` have to reach the SAME libJLCS.** A thunk
that catches a C++ exception calls `jlcs_catch_current_exception`, which writes
the message into libJLCS's own `jlcs_exception_buffer`; `_check_pending_exception`
reads it back by ccall'ing `jlcs_has_pending_exception` in `MLIRNative.libJLCS`.
Two copies of libJLCS in one process means the thunk writes one buffer and Julia
reads the other, so every C++ exception crossing a thunk is silently swallowed —
which is why this pre-loads the existing library rather than letting a sibling
copy be vendored next to the thunks. The JIT path holds the same invariant by a
different route: it registers those four symbols out of `MLIRNative.libJLCS` by
hand (`initialize_global_jit`, "Register exception handling helper symbols").

On ELF the AOT path held it for free. `ld` records a RUNPATH naming libJLCS's
directory, so the loader binds to that exact file. **PE has no RUNPATH.** The
Windows loader searches the loading module's directory and `PATH`; libJLCS lives
in RepliBuild's own source tree, which is neither — so the thunks DLL did not
fail to *bind*, it failed to *open*, with `The specified module could not be
found`. That message names the module that was found and not the dependency that
was not, so it reads as a missing thunks library while the thunks library is
sitting right there.

Opening libJLCS by absolute path first satisfies both halves: the loader resolves
the import against an already-loaded module by name, and that module is the one
`MLIRNative` ccalls. Idempotent, and on ELF a no-op that names the file the
RUNPATH already resolved to.

**The loader matches by soname, so a thunks library linked against a libJLCS
of another soname is refused here.** On ELF the import is the soname of the
libJLCS the thunks were linked against (`libJLCS.so.<dialect ABI>`; LLVM's
version, `libJLCS.so.23.1`, before the dialect had its own). When the libJLCS
in use has another, the already-loaded one does not satisfy the import and the
loader goes looking by RUNPATH: it finds the old file where it still is (an
older artifact, a build directory that was built again) and loads it as a
second copy, with no error. That is the two-buffer case above, so it is an
error here that says to wrap again ([`_foreign_jlcs`](@ref)).
"""
function open_thunks_library(thunks_path::AbstractString)
    jlcs = MLIRNative.libJLCS_path()
    jlcs_state = if !isfile(jlcs)
        "MISSING — build the dialect with src/mlir/build.sh"
    else
        try
            Libdl.dlopen(jlcs, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
            "loaded"
        catch e
            "present but would not open: " * sprint(showerror, e)
        end
    end

    # What was loaded before is not this library's doing.
    before = _foreign_jlcs(jlcs)
    handle = try
        Libdl.dlopen(thunks_path, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    catch e
        said = sprint(showerror, e)
        relink = occursin(r"libJLCS\.(?:so|dll|dylib)", said) ? """


        The loader's message names a libJLCS file: the thunks were linked
        against a libJLCS that is no longer there. Wrap the package again
        (`RepliBuild.wrap`) to link them against the one above.""" : ""
        error("""
        AOT thunks library failed to open:
          $thunks_path
          $said

        Its thunks import `jlcs_catch_current_exception` from libJLCS, which must
        therefore load first:
          $jlcs
          $jlcs_state$relink""")
    end
    brought = setdiff(_foreign_jlcs(jlcs), before)
    if !isempty(brought)
        try
            Libdl.dlclose(handle)
        catch
        end
        error(_second_jlcs_message(thunks_path, jlcs, brought))
    end
    return handle
end

"""
    _foreign_jlcs(jlcs, loaded = Libdl.dllist()) -> Vector{String}

The libJLCS files among `loaded` that are not `jlcs`, the one RepliBuild uses:
a second copy of the library in this process. Compared as files (`realpath`),
since the library is reached through a symlink (`libJLCS.so` to
`libJLCS.so.<ABI>`) and the loader reports the name it was asked for.
"""
function _foreign_jlcs(jlcs::AbstractString, loaded = Libdl.dllist())
    own = _real_file(jlcs)
    return String[p for p in loaded if _is_jlcs_file(p) && _real_file(p) != own]
end

# `libJLCS.so`, `libJLCS.so.3`, `libJLCS.so.3.0.1`, `libJLCS.so.23.1jl`,
# `libJLCS.dll`. Not `libJLCS_thunks.so`, which is a wrapped project's.
_is_jlcs_file(path::AbstractString) =
    occursin(r"^libJLCS\.(?:so(?:\.[0-9A-Za-z.]+)?|dll|dylib)$", basename(path))

_real_file(path::AbstractString) = ispath(path) ? realpath(path) : abspath(path)

function _second_jlcs_message(thunks_path::AbstractString, jlcs::AbstractString, brought)
    return """
    AOT thunks library was linked against another libJLCS:
      $thunks_path
    It loaded
      $(join(brought, "\n  "))
    beside the libJLCS RepliBuild uses:
      $jlcs

    Each copy has its own exception buffer, so a C++ exception a thunk catches
    would never reach Julia. Wrap the package again (`RepliBuild.wrap`) to link
    its thunks against the libJLCS above."""
end

"""
    resolve_thunk!(slot, handle, name, missing) -> Ptr{Cvoid}

Resolve one AOT thunk into `slot` at module init, recording a miss in `missing`
rather than raising.

Collecting instead of throwing is deliberate: a stale thunks library is usually
missing a *set* of symbols, and failing on the first one turns a diagnosable
"these five thunks are gone, rebuild" into a guessing game. The module still
loads and every function whose thunk resolved still works; a call into one that
did not raises through [`_aot_fptr_null`](@ref).

`slot` is a `Ref`, not a `const Ptr`, because a raw pointer cannot be serialized
into a precompile image — it must be filled at `__init__` in the loading process.
"""
function resolve_thunk!(slot::Base.RefValue{Ptr{Cvoid}}, handle::Ptr{Cvoid},
                        name::AbstractString, missing::Vector{String})
    slot[] = C_NULL
    if handle == C_NULL
        push!(missing, String(name))
        return C_NULL
    end
    p = Libdl.dlsym(handle, name; throw_error = false)
    if p === nothing
        stale = endswith(name, "_thunk") &&
                Libdl.dlsym(handle, name * "_inplace"; throw_error = false) !== nothing
        push!(missing, stale ? _stale_inplace_message(name) : String(name))
        return C_NULL
    end
    slot[] = p
    return p
end

"""
    warn_missing_thunks(missing, library_path)

Report thunks that `__init__` could not resolve, once, naming all of them.
"""
function warn_missing_thunks(missing::Vector{String}, library_path::AbstractString)
    isempty(missing) && return nothing
    shown = length(missing) > 20 ? vcat(missing[1:20], ["… and $(length(missing) - 20) more"]) : missing
    @warn "AOT thunks missing from the thunks library — calling any of these " *
          "will raise. The thunks .so is generated from the same MLIR the JIT " *
          "would build, so a gap here means it is stale: run wrap() again, " *
          "which compiles it." *
          "\n  " * join(shown, "\n  ") library = library_path count = length(missing)
    return nothing
end

"""
    invoke_aot_ptr(fptr, ::Type{T}, args...) where T

Call an AOT thunk whose address was already resolved (see [`resolve_thunk!`](@ref)).

Identical to [`invoke_aot`](@ref) except that it is handed the function pointer
instead of finding it. Takes no symbol name **by design** — with the same
argument types as `invoke_aot` a name parameter would make the two calls
visually interchangeable while the first argument meant different things, and
the diagnostic it would carry is one `__init__` already emitted, naming every
missing thunk at once rather than one per call site.
"""
@generated function invoke_aot_ptr(fptr::Ptr{Cvoid}, ::Type{T},
                                   args::Vararg{Any, N}) where {T, N}
    (setup, ptrs, preserve_args) = _arg_marshal_plan(args)
    quote
        fptr == C_NULL && _aot_fptr_null()
        $(setup...)
        inner_ptrs = Ptr{Cvoid}[$(ptrs...)]
        result = GC.@preserve $(preserve_args...) begin
            _invoke_call(fptr, T, inner_ptrs)
        end
        _check_pending_exception()
        return result
    end
end

"""
    invoke_aot_ptr(fptr, args...)

Void-return form — no `Type` parameter, same as [`invoke_aot`](@ref)'s.
"""
@generated function invoke_aot_ptr(fptr::Ptr{Cvoid}, args::Vararg{Any, N}) where N
    (setup, ptrs, preserve_args) = _arg_marshal_plan(args)
    quote
        fptr == C_NULL && _aot_fptr_null()
        $(setup...)
        inner_ptrs = Ptr{Cvoid}[$(ptrs...)]
        GC.@preserve $(preserve_args...) inner_ptrs begin
            _invoke_void(fptr, inner_ptrs)
        end
        _check_pending_exception()
        return nothing
    end
end

# =============================================================================
# JIT Initialization
# =============================================================================

const _JIT_DUMP_CONFIGURED = Ref(false)

"""
    configure_jit_dump_session!()

Point LLVM's perf jitdump at a **session** directory, once per process.

The jitdump is one file per PROCESS holding every JIT'd symbol from every
engine — verified by loading two wrapped libraries in one session and getting a
single dump containing both libraries' thunks. So it is a session artifact, not
a library one, and filing it under a package directory would become a lie the
moment two wrappers coexist. The generated MLIR sources *are* per-library and do
live in `<pkg>/.debug/mlir` (see `MLIRNative.debug_dir_for`).

Off unless `REPLIBUILD_JIT_PROFILE` is set — the same variable the dialect reads
to decide whether to register the perf listener at all. Set it to a path to
choose the session root; any other value uses `~/.replibuild/jit-sessions`.
LLVM appends `.debug/jit/` to whatever it is given.
"""
function configure_jit_dump_session!()
    _JIT_DUMP_CONFIGURED[] && return nothing
    _JIT_DUMP_CONFIGURED[] = true

    prof = get(ENV, "REPLIBUILD_JIT_PROFILE", "")
    isempty(prof) && return nothing

    root = occursin('/', prof) ? prof : joinpath(homedir(), ".replibuild", "jit-sessions")
    dir = joinpath(root, "session-$(getpid())")
    try
        mkpath(dir)
        ENV["JITDUMPDIR"] = dir
        @info "RepliBuild JIT profiling on; jitdump → $(joinpath(dir, ".debug", "jit"))"
    catch e
        # Leaving JITDUMPDIR unset is not neutral: LLVM then writes to
        # $HOME/.debug/jit, which is exactly the unmanaged pile this replaces.
        @warn "REPLIBUILD_JIT_PROFILE set but the session dir is unusable; " *
              "jitdump will fall back to \$HOME/.debug/jit" dir exception=e
    end
    return nothing
end

"""
    _module_callees(metadata, needed_symbols) -> Set{String}

The mangled symbols the thunk module for this library will call: the
functions the manifest lists (every function in the metadata when there is no
manifest) less the pure virtual ones, whose thunks call through the vtable,
the STL method table's members, and the constructors and
destructors of the temporaries those functions' by-value class parameters
need (`JLCSIRGenerator.lifetime_callees`), which the module names whether or
not the manifest lists them.
"""
function _module_callees(metadata, needed_symbols)::Set{String}
    out = Set{String}()
    if needed_symbols !== nothing
        union!(out, needed_symbols)
    else
        for f in get(metadata, "functions", Any[])
            m = String(get(f, "mangled", ""))
            isempty(m) || push!(out, m)
        end
    end
    # A pure virtual member function's thunk calls through the vtable and
    # names no symbol (`Compiler._is_vtable_only`): there is nothing to
    # resolve, and looked up, its name would be reported unresolved and its
    # thunk left out.
    for f in get(metadata, "functions", Any[])
        Compiler._is_vtable_only(f) && delete!(out, String(get(f, "mangled", "")))
    end
    for (_, methods) in get(metadata, "stl_methods", Dict{String,Any}())
        for m in methods
            s = String(get(m, "mangled", ""))
            isempty(s) || push!(out, s)
        end
    end
    union!(out, JLCSIRGenerator.lifetime_callees(metadata, needed_symbols))
    return out
end

"""
    _preflight_callees(lookup, callees, own) -> (bound::Dict{String,Ptr{Cvoid}}, unresolved::Vector{String})

Which of `callees` the library does not define (`own`, its runtime symbol
names) but `lookup(name)` resolves (`dlsym` through the library's own handle;
`nothing` or `C_NULL` when it does not), and which it resolves nowhere. A
callee in `own` is ORC's to bind through the library, as always, and is
neither. Pure, so a test drives it with any `lookup`.
"""
function _preflight_callees(lookup, callees, own)
    bound = Dict{String,Ptr{Cvoid}}()
    unresolved = String[]
    for name in sort!(collect(callees))
        name in own && continue
        ptr = lookup(name)
        if ptr === nothing || ptr == C_NULL
            push!(unresolved, name)
        else
            bound[name] = ptr
        end
    end
    return (bound, unresolved)
end

"Remove the STL table entries whose symbol is in `unresolved`, in place."
function _drop_stl_methods!(metadata, unresolved)
    haskey(metadata, "stl_methods") || return metadata
    gone = Set{String}(unresolved)
    for (c, methods) in metadata["stl_methods"]
        filter!(m -> !(String(get(m, "mangled", "")) in gone), methods)
    end
    return metadata
end

"""
Remove the function rows whose symbol is in `unresolved`, in place: a
function the library's handle cannot resolve gets no thunk, and no thunk
calls it as the constructor or destructor of a by-value temporary
(`JLCSIRGenerator._collect_class_raii` reads these rows).
"""
function _drop_functions!(metadata, unresolved)
    haskey(metadata, "functions") || return metadata
    gone = Set{String}(unresolved)
    filter!(f -> !(String(get(f, "mangled", "")) in gone), metadata["functions"])
    return metadata
end

"""
    _assert_callees_declared(ir_source, lib)

Refuse a thunk module that calls a symbol it never declares
(`JLCSIRGenerator.undeclared_callees`), naming each. MLIR's translation of
such a module to LLVM IR dereferences null, so `create_jit` would end the
process; as an error here it is one library's failed initialization.
"""
function _assert_callees_declared(ir_source::AbstractString, lib::AbstractString)
    undeclared = JLCSIRGenerator.undeclared_callees(ir_source)
    isempty(undeclared) && return nothing
    error("the thunk module for $(lib) calls $(length(undeclared)) symbol(s) it never declares: " *
          join(undeclared, ", ") * ". This is a bug in RepliBuild's thunk generator, not in the library; " *
          "handing the module to the JIT would end the process.")
end

# Declarations the thunk module carries that no pre-flight owns: helpers
# registered process-wide before the engine exists (`register_symbol_global`
# in `initialize_global_jit`, the dialect's own runtime, the C++ EH entry
# points `JlcsJIT.cpp` defines) and the `__def_` type anchors the IR generator
# declares and never calls.
const _GLOBAL_DECLARATION_PREFIXES = ("dispatch_", "jlcs_", "__cxa_", "__gxx_personality", "_mlir_", "llvm.",
                                      "malloc", "free", "memcpy", "memset", "__def_")

"""
    _check_module_declarations(ir_source, own, bound, lib)

Every `func.func private @name(` the module declares is the library's
(`own`), bound through its handle (`bound`), or a helper registered before the
engine exists; anything else is named in a warning, because the first thunk
that reaches it would block the process rather than fail.
"""
function _check_module_declarations(ir_source::AbstractString, own, bound, lib::AbstractString)
    stray = String[]
    for m in eachmatch(r"func\.func private @([^\s(]+)\(", ir_source)
        n = String(m.captures[1])
        (n in own || haskey(bound, n)) && continue
        any(p -> startswith(n, p), _GLOBAL_DECLARATION_PREFIXES) && continue
        push!(stray, n)
    end
    isempty(stray) || @warn "JIT: $(basename(lib))'s thunk module declares $(length(stray)) symbol(s) nothing " *
                            "pre-flighted; a thunk that reaches one will block, not fail" symbols = join(sort(stray), ", ")
    return stray
end

"""
    initialize_global_jit(binary_path::String)

Initialize a JIT engine for `binary_path`. Called once per wrapped library,
from the generated module's `__init__`. Each library gets its own engine
(`LibraryEngine`); repeated calls for the same binary are no-ops, and one
library's initialization failure never disables another's Tier 2.
"""
function initialize_global_jit(binary_path::String)
    rp = try realpath(binary_path) catch; abspath(binary_path) end
    lock(GLOBAL_JIT.lock) do
        # Engine (or recorded failure) already exists for this binary → no-op.
        for eng in GLOBAL_JIT.engines
            eng.binary_path == rp && return
        end

        eng = LibraryEngine(rp, C_NULL, nothing, nothing, nothing)
        push!(GLOBAL_JIT.engines, eng)

        try
            # 1. Create MLIR Context
            eng.mlir_ctx = create_context()

            # 2. Parse VTable Info
            eng.vtable_info = DWARFParser.parse_vtables(rp)

            # Load metadata
            metadata_path = joinpath(dirname(rp), "compilation_metadata.json")
            metadata = if isfile(metadata_path)
                _read_json(metadata_path)
            else
                Dict()
            end
            # The owners' definitions of the types this library imported from
            # earlier libraries of its graph, if any (`type_imports.json`).
            JLCSIRGenerator.apply_type_imports!(metadata, dirname(rp))

            # Register dispatch_ symbols for virtual methods
            lib_handle = Libdl.dlopen(rp)
            for (class_name, class_info) in eng.vtable_info.classes
                for method in class_info.virtual_methods
                    dispatch_name = "dispatch_$(replace(method.mangled_name, "::" => "_", "(" => "_", ")" => "_"))"
                    ptr = something(Libdl.dlsym(lib_handle, method.mangled_name, throw_error=false), C_NULL)
                    if ptr != C_NULL
                        MLIRNative.register_symbol_global(dispatch_name, ptr)
                    end
                end
            end

            # 2b. Register exception handling helper symbols for JIT'd code
            for sym in (:jlcs_set_pending_exception, :jlcs_catch_current_exception,
                        :jlcs_has_pending_exception, :jlcs_clear_pending_exception)
                ptr = something(Libdl.dlsym(Libdl.dlopen(MLIRNative.libJLCS_path()), sym, throw_error=false), C_NULL)
                if ptr != C_NULL
                    MLIRNative.register_symbol_global(string(sym), ptr)
                end
            end

            # Register C++ runtime EH symbols (the personality routine plus the
            # __cxa_begin/end_catch pair). Use C_NULL handle to search the
            # default global symbol space.
            #
            # Neither the runtime's FILENAME nor the personality's NAME is
            # universal. GNU/Linux ships libstdc++; MSYS2 CLANG64 — the
            # x86_64-w64-windows-gnu environment this targets — ships libc++ as
            # a DLL; macOS ships libc++.1.dylib. And mingw unwinds with SEH, so
            # the personality is `__gxx_personality_seh0` there, which is why
            # the symbol comes from CXX_PERSONALITY rather than a literal.
            cxxrt_handle = C_NULL
            # Hoisted out of the `try` so the diagnostic below can name what was
            # actually attempted.
            cxxrt_candidates = if Sys.iswindows()
                ("libc++.dll", "libstdc++-6.dll")
            elseif Sys.isapple()
                ("libc++.1.dylib", "libc++.dylib")
            else
                ("libstdc++.so.6", "libstdc++.so")
            end
            try
                for cand in cxxrt_candidates
                    cxxrt_handle = something(
                        Libdl.dlopen(cand, Libdl.RTLD_LAZY, throw_error=false), C_NULL)
                    cxxrt_handle == C_NULL || break
                end
            catch; end
            unresolved_eh = String[]
            for sym in (Symbol(CXX_PERSONALITY), :__cxa_begin_catch, :__cxa_end_catch)
                ptr = C_NULL
                if cxxrt_handle != C_NULL
                    ptr = something(Libdl.dlsym(cxxrt_handle, sym, throw_error=false), C_NULL)
                end
                if ptr == C_NULL
                    # Fallback: search in already-loaded libraries (the C++ .so we built loads libstdc++)
                    ptr = something(Libdl.dlsym(lib_handle, sym, throw_error=false), C_NULL)
                end
                if ptr != C_NULL
                    MLIRNative.register_symbol_global(string(sym), ptr)
                else
                    push!(unresolved_eh, string(sym))
                end
            end
            # A skipped registration used to be silent, and silence is the worst
            # possible response here: ORC DEADLOCKS on an unresolved symbol
            # rather than raising — the same class the Tier-1 slice pre-flight
            # and `_lto_unresolved_symbols` exist for — so the next thunk that
            # references one of these hangs the process with nothing said. Name
            # it while there is still something to read. The cause is almost
            # always a C++ runtime this list does not know, or a personality
            # spelling the installed runtime does not export.
            isempty(unresolved_eh) || @warn(
                "JIT: C++ EH symbols unresolved. A thunk that needs one will HANG " *
                "rather than fail — ORC does not raise on an unresolved symbol.",
                missing_symbols = join(unresolved_eh, ", "),
                runtimes_tried  = join(cxxrt_candidates, ", "),
                runtime_opened  = cxxrt_handle == C_NULL ? "none" : "yes",
                personality     = CXX_PERSONALITY)

            # 3. Load thunk manifest (dead-thunk elimination)
            # If the wrapper wrote a manifest of which function thunks it actually
            # needs, only generate those. Otherwise generate everything (backward compat).
            manifest_path = joinpath(dirname(rp), "thunk_manifest.json")
            needed_symbols = if isfile(manifest_path)
                try
                    manifest = _read_json(manifest_path)
                    Set{String}(get(manifest, "function_thunks", String[]))
                catch
                    nothing
                end
            else
                nothing
            end

            # 3b. Pre-flight every callee the thunk module will name through
            # the library's OWN handle. A callee the library defines is bound by
            # ORC through the library as always. One it does not define but its
            # handle resolves (a C++ runtime member it calls, `runtime_library`
            # in the STL table; glibc's dlsym searches a library's dependency
            # tree) is bound after the engine exists at exactly that address,
            # on this engine (`register_symbol`), never through RTLD_DEFAULT,
            # whose answer could be another copy of the runtime. One the handle
            # cannot resolve gets no thunk: ORC does not raise on an unresolved
            # symbol, it blocks, so the module must never name it.
            own = something(Compiler._runtime_symbol_names(rp), Set{String}())
            callees = _module_callees(metadata, needed_symbols)
            (runtime_bound, unresolved) = _preflight_callees(
                n -> Libdl.dlsym(lib_handle, n; throw_error=false), callees, own)
            if !isempty(unresolved)
                @warn "JIT: $(length(unresolved)) callee(s) of $(basename(rp))'s thunks resolve neither " *
                      "in the library nor through its handle; their thunks are left out, and the " *
                      "wrapper's methods on them raise instead of hanging" unresolved = join(sort(unresolved), ", ")
                needed_symbols === nothing || setdiff!(needed_symbols, unresolved)
                _drop_stl_methods!(metadata, unresolved)
                _drop_functions!(metadata, unresolved)
            end

            # 4. Generate MLIR Module for vtables + needed function thunks
            ir_source = JLCSIRGenerator.generate_jlcs_ir(eng.vtable_info, metadata;
                                                          needed_symbols=needed_symbols)
            # A callee the module never declares would end the process in
            # `create_jit`, not raise: refused here, it fails this library alone.
            _assert_callees_declared(ir_source, basename(rp))
            # The declarations the module actually carries: every one must be
            # the library's, pre-flighted above, or a helper registered
            # globally, or the first call into it would block the process.
            _check_module_declarations(ir_source, own, runtime_bound, basename(rp))

            # 4. Parse and Lower Module
            #
            # debug_base puts the generated MLIR beside the wrapper it describes,
            # so gdb can open it when you break in a thunk — including for a
            # vendored wrapper, where a tempdir copy would not have travelled.
            # Unwritable (read-only depot) falls back to tempdir, keeping the
            # source view at the cost of co-location.
            debug_base = MLIRNative.debug_dir_for(rp)
            @debug "JIT debug artifacts for $(basename(rp))" mlir_sources=joinpath(debug_base, "mlir")
            mod = parse_module(eng.mlir_ctx, ir_source; debug_base=debug_base)

            # Lower JLCS -> LLVM
            if !lower_to_llvm(mod)
                error(MLIRNative._with_diagnostics(
                    "Failed to lower JLCS dialect to LLVM."))
            end

            # 5. Create JIT Engine with the C++ library and libJLCS for EH symbol resolution
            #
            # Must precede the FIRST engine: LLVM's perf listener is a process
            # singleton, so JITDUMPDIR is read once and the earliest setting wins.
            configure_jit_dump_session!()
            jlcs_lib_path = MLIRNative.libJLCS_path()

            # The object cache has to be requested HERE — it cannot be enabled
            # on a live engine — so the decision is read from the environment
            # before the engine exists rather than offered as an argument
            # nothing on this path could supply.
            want_obj = haskey(ENV, "REPLIBUILD_JIT_OBJDUMP")
            eng.jit_engine = create_jit(mod, opt_level=1, dump_object=want_obj,
                                        shared_libs=[rp, jlcs_lib_path])

            # The runtime-library callees, at the addresses the library's own
            # handle gave, on this engine alone. ORC materialises a thunk at its
            # first lookup, so a symbol defined here is seen by every thunk.
            for name in sort!(collect(keys(runtime_bound)))
                MLIRNative.register_symbol(eng.jit_engine, name, runtime_bound[name])
            end
            isempty(runtime_bound) ||
                @debug "JIT: $(length(runtime_bound)) runtime-library callee(s) bound through $(basename(rp))'s handle"

            if want_obj
                obj = joinpath(debug_base, "obj", first(splitext(basename(rp))) * ".o")
                try
                    MLIRNative.dump_object_file(eng.jit_engine, obj)
                    @info "JIT object written for $(basename(rp))" object=obj
                catch e
                    # Never fatal: the object is an inspection artifact, and a
                    # read-only install must still get a working engine.
                    @warn "REPLIBUILD_JIT_OBJDUMP set but the object could not be written" object=obj exception=e
                end
            end

            # Mirror the first successful engine into the legacy single-engine
            # fields for backward compatibility.
            if GLOBAL_JIT.jit_engine === nothing
                GLOBAL_JIT.mlir_ctx = eng.mlir_ctx
                GLOBAL_JIT.jit_engine = eng.jit_engine
                GLOBAL_JIT.vtable_info = eng.vtable_info
            end
            @atomic GLOBAL_JIT.initialized = true
        catch e
            eng.init_error = e isa Exception ? e : ErrorException(string(e))
            if GLOBAL_JIT.init_error === nothing
                GLOBAL_JIT.init_error = eng.init_error
            end
            @error "Failed to initialize JIT for $(basename(rp))" exception=e
            @warn "JIT initialization failed for $(basename(rp)). Tier 2 dispatch for this library will not work, but its ccall-based wrappers still function. Other libraries' engines are unaffected."
        end
    end
end

"""
    get_jit_thunk(class_name::String, method_name::String) -> Ptr{Cvoid}

Get a function pointer to a JIT-compiled thunk that performs virtual dispatch.
The thunk signature matches the C++ method (with 'this' as first arg).
"""
function get_jit_thunk(class_name::String, method_name::String)
    if !(@atomic GLOBAL_JIT.initialized)
        _jit_not_initialized_error()
    end

    safe_class = replace(class_name, "::" => "_")
    safe_method = replace(method_name, "::" => "_", "(" => "_", ")" => "_")
    thunk_name = "$(safe_class)_$(safe_method)"

    return _lookup_cached(thunk_name)
end

"""
    cleanup()

Destroy the JIT context and resources.
"""
function cleanup()
    lock(GLOBAL_JIT.lock) do
        for eng in GLOBAL_JIT.engines
            if eng.jit_engine !== nothing
                destroy_jit(eng.jit_engine)
                eng.jit_engine = nothing
            end
            if eng.mlir_ctx != C_NULL
                destroy_context(eng.mlir_ctx)
                eng.mlir_ctx = C_NULL
            end
        end
        empty!(GLOBAL_JIT.engines)

        # Legacy mirrors reference the first engine's handles, destroyed above.
        GLOBAL_JIT.jit_engine = nothing
        GLOBAL_JIT.mlir_ctx = C_NULL
        GLOBAL_JIT.vtable_info = nothing
        GLOBAL_JIT.init_error = nothing

        @atomic GLOBAL_JIT.initialized = false
        @atomic GLOBAL_JIT.compiled_symbols = Dict{String, Ptr{Cvoid}}()
    end
end

end # module JITManager
