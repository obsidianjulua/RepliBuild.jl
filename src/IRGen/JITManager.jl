#!/usr/bin/env julia
# JITManager.jl - Manages the lifecycle of MLIR JIT compilation for C++ vtables
# Acts as the bridge between Julia wrappers and the MLIR execution engine.

module JITManager

using ..MLIRNative
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
    invoke(func_name::String, ::Type{T}, args...) where T

Invoke a JIT-compiled function with return type T.
@generated: emits arity-specialized code for any N at compile time.
For N=1..4 this produces identical code to the old hand-written methods.
For N≥5 this eliminates the Vector{Any}/Vector{Ptr{Cvoid}} allocation
that the old generic fallback incurred on every call.
"""
@generated function invoke(func_name::String, ::Type{T}, args::Vararg{Any, N}) where {T, N}
    ref_syms = [Symbol("r$i") for i in 1:N]

    # Strings marshal as pointers to their bytes (NUL-terminated in Julia).
    # Packing Ref(::String) directly hands the callee a pointer into the
    # String OBJECT, not its data — segfault on first dereference (found live
    # on tinyxml2 XMLDocument::Parse). The String itself is GC-preserved via
    # the args tuple held across the call.
    str_syms = [Symbol("s$i") for i in 1:N]
    setup = [args[i] <: AbstractString ?
                 :($(str_syms[i]) = args[$i]; $(ref_syms[i]) = Ref(pointer($(str_syms[i])))) :
                 :($(ref_syms[i]) = Ref(args[$i])) for i in 1:N]
    ptrs = [:(Base.unsafe_convert(Ptr{Cvoid}, $(ref_syms[i]))) for i in 1:N]
    # Preserve the Refs AND the strings whose byte pointers were taken
    preserve_args = vcat(ref_syms, [str_syms[i] for i in 1:N if args[i] <: AbstractString])

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
    ref_syms = [Symbol("r$i") for i in 1:N]

    # Same String marshalling as the value-returning variant above
    str_syms = [Symbol("s$i") for i in 1:N]
    setup = [args[i] <: AbstractString ?
                 :($(str_syms[i]) = args[$i]; $(ref_syms[i]) = Ref(pointer($(str_syms[i])))) :
                 :($(ref_syms[i]) = Ref(args[$i])) for i in 1:N]
    ptrs = [:(Base.unsafe_convert(Ptr{Cvoid}, $(ref_syms[i]))) for i in 1:N]
    # Preserve the Refs AND the strings whose byte pointers were taken
    preserve_args = vcat(ref_syms, [str_syms[i] for i in 1:N if args[i] <: AbstractString])

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
# JIT Initialization
# =============================================================================

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
                JSON.parsefile(metadata_path)
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

            # Register C++ runtime EH symbols (__gxx_personality_v0, __cxa_begin/end_catch)
            # Use C_NULL handle to search the default global symbol space
            cxxrt_handle = C_NULL
            try
                # Try libstdc++ first, then libc++abi
                cxxrt_handle = something(Libdl.dlopen("libstdc++.so.6", Libdl.RTLD_LAZY | Libdl.RTLD_NOLOAD, throw_error=false), C_NULL)
                if cxxrt_handle == C_NULL
                    cxxrt_handle = something(Libdl.dlopen("libstdc++.so", Libdl.RTLD_LAZY, throw_error=false), C_NULL)
                end
            catch; end
            for sym in (:__gxx_personality_v0, :__cxa_begin_catch, :__cxa_end_catch)
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
                    manifest = JSON.parsefile(manifest_path)
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
            mod = parse_module(eng.mlir_ctx, ir_source)

            # Lower JLCS -> LLVM
            if !lower_to_llvm(mod)
                error("Failed to lower JLCS dialect to LLVM")
            end

            # 5. Create JIT Engine with the C++ library and libJLCS for EH symbol resolution
            jlcs_lib_path = MLIRNative.libJLCS
            eng.jit_engine = create_jit(mod, opt_level=1, shared_libs=[rp, jlcs_lib_path])

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
