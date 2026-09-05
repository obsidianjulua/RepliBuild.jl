#!/usr/bin/env julia
# JITManager.jl - Manages the lifecycle of MLIR JIT compilation for C++ vtables
# Acts as the bridge between Julia wrappers and the MLIR execution engine.

module JITManager

using ..MLIRNative
import ..MLIRNative: CXX_PERSONALITY
using ..JLCSIRGenerator
using ..DWARFParser
using Libdl
import JSON

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
        return :(ccall(fptr, $T, (Ptr{Ptr{Cvoid}},), inner_ptrs))
    else
        # Struct return: void ciface(T* sret, void** args_ptr) — sret convention
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

Kept as ONE plan consumed by both `invoke` methods — they carried byte-identical
copies of this, which is how a fix to one silently misses the other.
"""
function _arg_marshal_plan(argtypes)
    N = length(argtypes)
    ref_syms = [Symbol("r$i") for i in 1:N]
    src_syms = [Symbol("s$i") for i in 1:N]

    is_str(T)      = T <: AbstractString
    is_boxed_ref(T) = T <: Base.Ref && !(T <: Ptr)
    indirect(T)    = is_str(T) || is_boxed_ref(T)

    setup = map(1:N) do i
        T = argtypes[i]
        if is_str(T)
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
            ccall(fptr, Cvoid, (Ptr{Ptr{Cvoid}},), inner_ptrs)
        end
        _check_pending_exception()
        return nothing
    end
end

# =============================================================================
# AOT thunk invocation
#
# The same thunks, reached without a JIT. `build_aot_thunks` compiles the MLIR
# at build time into `lib<name>_thunks.so`, so the wrapper dlopens that instead
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
        "open — rebuild the package, or set aot_thunks = false to use the JIT.")
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
            ccall(fptr, Cvoid, (Ptr{Ptr{Cvoid}},), inner_ptrs)
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
          "or was built from different sources. Rebuild the package.")
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
        push!(missing, String(name))
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
          "would build, so a gap here means it is stale: rebuild the package." *
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
            ccall(fptr, Cvoid, (Ptr{Ptr{Cvoid}},), inner_ptrs)
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
                # use_mmap=false: a live mmap blocks deletion on Windows and is
                # released only at GC — see Builder/ThunkBuilder.jl.
                JSON.parsefile(metadata_path; use_mmap=false)
            else
                Dict()
            end

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
                ptr = something(Libdl.dlsym(Libdl.dlopen(MLIRNative.libJLCS), sym, throw_error=false), C_NULL)
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
            try
                candidates = if Sys.iswindows()
                    ("libc++.dll", "libstdc++-6.dll")
                elseif Sys.isapple()
                    ("libc++.1.dylib", "libc++.dylib")
                else
                    ("libstdc++.so.6", "libstdc++.so")
                end
                for cand in candidates
                    cxxrt_handle = something(
                        Libdl.dlopen(cand, Libdl.RTLD_LAZY, throw_error=false), C_NULL)
                    cxxrt_handle == C_NULL || break
                end
            catch; end
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
                end
            end

            # 3. Load thunk manifest (dead-thunk elimination)
            # If the wrapper wrote a manifest of which function thunks it actually
            # needs, only generate those. Otherwise generate everything (backward compat).
            manifest_path = joinpath(dirname(rp), "thunk_manifest.json")
            needed_symbols = if isfile(manifest_path)
                try
                    manifest = JSON.parsefile(manifest_path; use_mmap=false)
                    Set{String}(get(manifest, "function_thunks", String[]))
                catch
                    nothing
                end
            else
                nothing
            end

            # 4. Generate MLIR Module for vtables + needed function thunks
            ir_source = JLCSIRGenerator.generate_jlcs_ir(eng.vtable_info, metadata;
                                                          needed_symbols=needed_symbols)

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
            jlcs_lib_path = MLIRNative.libJLCS

            # The object cache has to be requested HERE — it cannot be enabled
            # on a live engine — so the decision is read from the environment
            # before the engine exists rather than offered as an argument
            # nothing on this path could supply.
            want_obj = haskey(ENV, "REPLIBUILD_JIT_OBJDUMP")
            eng.jit_engine = create_jit(mod, opt_level=1, dump_object=want_obj,
                                        shared_libs=[rp, jlcs_lib_path])

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
