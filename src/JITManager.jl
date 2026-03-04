#!/usr/bin/env julia
# JITManager.jl - Manages the lifecycle of MLIR JIT compilation for C++ vtables
# Acts as the bridge between Julia wrappers and the MLIR execution engine.

module JITManager

using ..MLIRNative
using ..JLCSIRGenerator
using ..DWARFParser
using Libdl
import JSON

export get_jit_thunk, ensure_jit_initialized, JITContext, invoke

# Global singleton to manage JIT state
mutable struct JITContext
    mlir_ctx::Ptr{Cvoid}
    jit_engine::Union{Ptr{Cvoid}, Nothing}
    compiled_symbols::Dict{String, Ptr{Cvoid}}
    vtable_info::Union{VtableInfo, Nothing}
    initialized::Bool
    lock::ReentrantLock

    function JITContext()
        new(C_NULL, nothing, Dict{String, Ptr{Cvoid}}(), nothing, false, ReentrantLock())
    end
end

const GLOBAL_JIT = JITContext()

# =============================================================================
# Fast function pointer lookup with lock-free read path
# =============================================================================

"""
    _lookup_cached(func_name::String) -> Ptr{Cvoid}

Look up a JIT function pointer with caching.
Fast path: lock-free Dict read for already-cached symbols.
Slow path: JIT engine lookup + cache with lock.
"""
@inline function _lookup_cached(func_name::String)::Ptr{Cvoid}
    # Fast path: Dict reads are safe without lock (single-writer pattern)
    ptr = get(GLOBAL_JIT.compiled_symbols, func_name, C_NULL)
    if ptr != C_NULL
        return ptr
    end

    # Slow path: look up in JIT engine and cache
    lock(GLOBAL_JIT.lock) do
        # Double-check after acquiring lock
        ptr = get(GLOBAL_JIT.compiled_symbols, func_name, C_NULL)
        if ptr != C_NULL
            return ptr
        end

        ptr = MLIRNative.lookup(GLOBAL_JIT.jit_engine, func_name)
        if ptr == C_NULL
            ptr = MLIRNative.lookup(GLOBAL_JIT.jit_engine, "_" * func_name)
        end
        if ptr == C_NULL
            error("JIT symbol not found: $func_name")
        end

        GLOBAL_JIT.compiled_symbols[func_name] = ptr
        return ptr
    end
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
Arity-specialized methods for 1-4 args eliminate the Any[] boxing overhead.
Uses _lookup_cached for lock-free symbol resolution on hot paths.
Handles both scalar returns (direct) and struct returns (sret) correctly.
"""

# 1-arg specialization
function invoke(func_name::String, ::Type{T}, a1) where T
    GLOBAL_JIT.initialized || error("JIT not initialized.")
    fptr = _lookup_cached(func_name)
    r1 = Ref(a1)
    inner_ptrs = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, r1)]
    GC.@preserve r1 begin
        return _invoke_call(fptr, T, inner_ptrs)
    end
end

# 2-arg specialization
function invoke(func_name::String, ::Type{T}, a1, a2) where T
    GLOBAL_JIT.initialized || error("JIT not initialized.")
    fptr = _lookup_cached(func_name)
    r1 = Ref(a1); r2 = Ref(a2)
    inner_ptrs = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, r1), Base.unsafe_convert(Ptr{Cvoid}, r2)]
    GC.@preserve r1 r2 begin
        return _invoke_call(fptr, T, inner_ptrs)
    end
end

# 3-arg specialization
function invoke(func_name::String, ::Type{T}, a1, a2, a3) where T
    GLOBAL_JIT.initialized || error("JIT not initialized.")
    fptr = _lookup_cached(func_name)
    r1 = Ref(a1); r2 = Ref(a2); r3 = Ref(a3)
    inner_ptrs = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, r1), Base.unsafe_convert(Ptr{Cvoid}, r2), Base.unsafe_convert(Ptr{Cvoid}, r3)]
    GC.@preserve r1 r2 r3 begin
        return _invoke_call(fptr, T, inner_ptrs)
    end
end

# 4-arg specialization
function invoke(func_name::String, ::Type{T}, a1, a2, a3, a4) where T
    GLOBAL_JIT.initialized || error("JIT not initialized.")
    fptr = _lookup_cached(func_name)
    r1 = Ref(a1); r2 = Ref(a2); r3 = Ref(a3); r4 = Ref(a4)
    inner_ptrs = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, r1), Base.unsafe_convert(Ptr{Cvoid}, r2), Base.unsafe_convert(Ptr{Cvoid}, r3), Base.unsafe_convert(Ptr{Cvoid}, r4)]
    GC.@preserve r1 r2 r3 r4 begin
        return _invoke_call(fptr, T, inner_ptrs)
    end
end

# Generic fallback for 5+ args
function invoke(func_name::String, ::Type{T}, args::Vararg{Any, N}) where {T, N}
    GLOBAL_JIT.initialized || error("JIT not initialized.")
    fptr = _lookup_cached(func_name)

    ref_args = Vector{Any}(undef, N)
    inner_ptrs = Vector{Ptr{Cvoid}}(undef, N)
    for (i, arg) in enumerate(args)
        r = Ref(arg)
        ref_args[i] = r
        inner_ptrs[i] = Base.unsafe_convert(Ptr{Cvoid}, r)
    end

    GC.@preserve ref_args begin
        return _invoke_call(fptr, T, inner_ptrs)
    end
end

# =============================================================================
# Void-return invoke (no Type parameter = void return)
# =============================================================================

function invoke(func_name::String, args::Vararg{Any, N}) where N
    GLOBAL_JIT.initialized || error("JIT not initialized.")
    fptr = _lookup_cached(func_name)

    ref_args = Vector{Any}(undef, N)
    inner_ptrs = Vector{Ptr{Cvoid}}(undef, N)
    for (i, arg) in enumerate(args)
        r = Ref(arg)
        ref_args[i] = r
        inner_ptrs[i] = Base.unsafe_convert(Ptr{Cvoid}, r)
    end

    GC.@preserve ref_args inner_ptrs begin
        ccall(fptr, Cvoid, (Ptr{Ptr{Cvoid}},), inner_ptrs)
    end
    return nothing
end

# =============================================================================
# JIT Initialization
# =============================================================================

"""
    initialize_global_jit(binary_path::String)

Initialize the global JIT context with vtable info from the binary.
This is called once when the wrapper module is loaded.
"""
function initialize_global_jit(binary_path::String)
    lock(GLOBAL_JIT.lock) do
        if GLOBAL_JIT.initialized
            return
        end

        try
            # 1. Create MLIR Context
            GLOBAL_JIT.mlir_ctx = create_context()

            # 2. Parse VTable Info
            GLOBAL_JIT.vtable_info = DWARFParser.parse_vtables(binary_path)

            # Load metadata
            metadata_path = joinpath(dirname(binary_path), "compilation_metadata.json")
            metadata = if isfile(metadata_path)
                JSON.parsefile(metadata_path)
            else
                Dict()
            end

            # 3. Generate MLIR Module for all vtables
            ir_source = JLCSIRGenerator.generate_jlcs_ir(GLOBAL_JIT.vtable_info, metadata)

            # 4. Parse and Lower Module
            mod = parse_module(GLOBAL_JIT.mlir_ctx, ir_source)

            # Lower JLCS -> LLVM
            if !lower_to_llvm(mod)
                error("Failed to lower JLCS dialect to LLVM")
            end

            # 5. Create JIT Engine with the C++ library registered for symbol resolution
            GLOBAL_JIT.jit_engine = create_jit(mod, opt_level=3, shared_libs=[binary_path])

            GLOBAL_JIT.initialized = true
            println("JIT Initialized for $binary_path")
        catch e
            @error "Failed to initialize JIT" exception=e
            @warn "JIT initialization failed. Functions using JIT dispatch (Tier 2) will not work, but ccall-based wrappers (Tier 1) will still function."
        end
    end
end

"""
    get_jit_thunk(class_name::String, method_name::String) -> Ptr{Cvoid}

Get a function pointer to a JIT-compiled thunk that performs virtual dispatch.
The thunk signature matches the C++ method (with 'this' as first arg).
"""
function get_jit_thunk(class_name::String, method_name::String)
    if !GLOBAL_JIT.initialized
        error("JIT not initialized. Call initialize_global_jit() first.")
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
        if GLOBAL_JIT.jit_engine !== nothing
            destroy_jit(GLOBAL_JIT.jit_engine)
            GLOBAL_JIT.jit_engine = nothing
        end

        if GLOBAL_JIT.mlir_ctx != C_NULL
            destroy_context(GLOBAL_JIT.mlir_ctx)
            GLOBAL_JIT.mlir_ctx = C_NULL
        end

        GLOBAL_JIT.initialized = false
        empty!(GLOBAL_JIT.compiled_symbols)
    end
end

end # module JITManager
