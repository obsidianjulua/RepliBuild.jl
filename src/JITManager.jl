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

"""
    invoke(func_name::String, ::Type{T}, args...) where T
    invoke(func_name::String, args...)

Invoke a JIT-compiled function managed by the global JIT context.

The thunk functions take a single `!llvm.ptr` argument (a void** array of pointers
to argument values) and optionally return a value. With `llvm.emit_c_interface`,
MLIR generates a `_mlir_ciface_*` wrapper:
  - Void return:   `ciface(void** args_ptr)`
  - Struct return:  `ciface(struct* sret_result, void** args_ptr)`

We use `lookup` + `ccall` to call the ciface wrapper directly, avoiding the
pitfalls of `mlirExecutionEngineInvokePacked`.
"""
function invoke(func_name::String, ::Type{T}, args...) where T
    if !GLOBAL_JIT.initialized
        error("JIT not initialized. Call initialize_global_jit() first.")
    end

    # Look up the ciface function pointer
    fptr = MLIRNative.lookup(GLOBAL_JIT.jit_engine, func_name)
    if fptr == C_NULL
        fptr = MLIRNative.lookup(GLOBAL_JIT.jit_engine, "_" * func_name)
    end
    if fptr == C_NULL
        error("JIT symbol not found: $func_name")
    end

    # Build the inner args array: void*[] where each entry points to an argument value
    ref_args = Any[]
    inner_ptrs = Ptr{Cvoid}[]
    for arg in args
        r = Ref(arg)
        push!(ref_args, r)
        push!(inner_ptrs, Base.unsafe_convert(Ptr{Cvoid}, r))
    end

    # Call ciface with sret convention: ciface(T* result, void** args_ptr) -> void
    ret_buf = Ref{T}()
    GC.@preserve ref_args inner_ptrs ret_buf begin
        ccall(fptr, Cvoid, (Ptr{T}, Ptr{Ptr{Cvoid}}), ret_buf, inner_ptrs)
    end
    return ret_buf[]
end

# Void-return overload
function invoke(func_name::String, args...)
    if !GLOBAL_JIT.initialized
        error("JIT not initialized. Call initialize_global_jit() first.")
    end

    fptr = MLIRNative.lookup(GLOBAL_JIT.jit_engine, func_name)
    if fptr == C_NULL
        fptr = MLIRNative.lookup(GLOBAL_JIT.jit_engine, "_" * func_name)
    end
    if fptr == C_NULL
        error("JIT symbol not found: $func_name")
    end

    ref_args = Any[]
    inner_ptrs = Ptr{Cvoid}[]
    for arg in args
        r = Ref(arg)
        push!(ref_args, r)
        push!(inner_ptrs, Base.unsafe_convert(Ptr{Cvoid}, r))
    end

    GC.@preserve ref_args inner_ptrs begin
        ccall(fptr, Cvoid, (Ptr{Ptr{Cvoid}},), inner_ptrs)
    end
    return nothing
end

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
            # This is fast enough to do at runtime, or we could serialize it
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
            # Don't rethrow — allow the module to load so ccall-based functions work
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

    # Construct the unique name used in the MLIR generator
    # Corresponds to: func.func @Base_foo(...)
    # We need to match the sanitization logic in JLCSIRGenerator.jl
    safe_class = replace(class_name, "::" => "_")
    safe_method = replace(method_name, "::" => "_", "(" => "_", ")" => "_")
    thunk_name = "$(safe_class)_$(safe_method)"

    # Check cache
    lock(GLOBAL_JIT.lock) do
        if haskey(GLOBAL_JIT.compiled_symbols, thunk_name)
            return GLOBAL_JIT.compiled_symbols[thunk_name]
        end

        # Lookup in JIT
        ptr = lookup(GLOBAL_JIT.jit_engine, thunk_name)
        
        if ptr == C_NULL
            # Try with leading underscore (common platform variation)
            ptr = lookup(GLOBAL_JIT.jit_engine, "_" * thunk_name)
        end

        if ptr == C_NULL
            error("JIT symbol not found: $thunk_name")
        end

        GLOBAL_JIT.compiled_symbols[thunk_name] = ptr
        return ptr
    end
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
