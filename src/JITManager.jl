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
    invoke(func_name::String, args...)

Invoke a JIT-compiled function managed by the global JIT context.
Arguments are passed by reference (pointers to values) to the JIT.
"""
function invoke(func_name::String, args...)
    if !GLOBAL_JIT.initialized
        error("JIT not initialized. Call initialize_global_jit() first.")
    end

    # Prepare arguments for mlirExecutionEngineInvokePacked
    # We need a vector of pointers to the arguments
    ptr_args = Vector{Ptr{Cvoid}}()
    ref_args = Any[] # Keep references alive
    
    for arg in args
        # Create a reference to the argument
        r = Ref(arg)
        push!(ref_args, r)
        # Get the pointer to the data
        push!(ptr_args, Base.unsafe_convert(Ptr{Cvoid}, r))
    end
    
    # Invoke via MLIRNative
    success = MLIRNative.jit_invoke(GLOBAL_JIT.jit_engine, func_name, ptr_args)
    
    if !success
        # Try with underscore prefix
        success = MLIRNative.jit_invoke(GLOBAL_JIT.jit_engine, "_" * func_name, ptr_args)
    end
    
    if !success
        error("Failed to invoke JIT function: $func_name")
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

            # 5. Create JIT Engine
            GLOBAL_JIT.jit_engine = create_jit(mod, opt_level=3)

            # 6. Register Symbols
            # We need to register the library's symbols so the JIT can find them
            # (e.g. the actual C++ methods being dispatched to)
            lib_handle = Libdl.dlopen(binary_path)
            
            # Register known method addresses from DWARF if needed
            # For now, we rely on dynamic symbol resolution in the JIT
            
            GLOBAL_JIT.initialized = true
            println("JIT Initialized for $binary_path")
        catch e
            @error "Failed to initialize JIT" exception=e
            rethrow(e)
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
