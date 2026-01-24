#!/usr/bin/env julia
# MLIRNative.jl - Julia bindings to JLCS MLIR Dialect
#
# Low-level ccall interface to MLIR C API and custom JLCS dialect
# Part of RepliBuild.jl toolchain for advanced FFI code generation

module MLIRNative

export create_context, create_module, destroy_context, parse_module, clone_module
export create_jit, destroy_jit, register_symbol, lookup, jit_invoke, invoke_safe, lower_to_llvm
export test_dialect, print_module

# =============================================================================
# MLIR C API Bindings
# =============================================================================

# Library paths
# Note: We use libJLCS for C API functions since it includes wrappers
# that link against the static MLIRCAPIIR library
const libJLCS_path = joinpath(@__DIR__, "mlir", "build", "libJLCS.so")
const libJLCS = libJLCS_path  # Alias for convenience

# Check if JLCS library exists
function check_library()
    if !isfile(libJLCS_path)
        error("""
        JLCS dialect library not found at: $libJLCS_path

        Build it first with:
            cd src/mlir
            ./build.sh

        For more information, see:
            docs/mlir/README.md
        """)
    end
end

# MLIR C API opaque types
const MlirContext = Ptr{Cvoid}
const MlirModule = Ptr{Cvoid}
const MlirOperation = Ptr{Cvoid}
const MlirLocation = Ptr{Cvoid}
const MlirStringRef = Ptr{Cvoid}
const MlirExecutionEngine = Ptr{Cvoid}
const MlirType = Ptr{Cvoid}

# =============================================================================
# Context Management
# =============================================================================

"""
    create_context() -> MlirContext

Create a new MLIR context and register the JLCS dialect.

The context must be destroyed with `destroy_context()` when done.
"""
function create_context()
    check_library()

    # Create MLIR context (using C API wrapper in libJLCS)
    ctx = ccall((:mlirContextCreate, libJLCS), MlirContext, ())

    if ctx == C_NULL
        error("Failed to create MLIR context")
    end

    # Register JLCS dialect with context
    ccall((:registerJLCSDialect, libJLCS), Cvoid, (MlirContext,), ctx)

    return ctx
end

"""
    destroy_context(ctx::MlirContext)

Destroy an MLIR context and free its resources.
"""
function destroy_context(ctx::MlirContext)
    ccall((:mlirContextDestroy, libJLCS), Cvoid, (MlirContext,), ctx)
end

# =============================================================================
# Module Management
# =============================================================================

"""
    create_module(ctx::MlirContext, location::MlirLocation) -> MlirModule

Create an empty MLIR module in the given context.
"""
function create_module(ctx::MlirContext, location::MlirLocation)
    return ccall((:mlirModuleCreateEmpty, libJLCS), MlirModule, (MlirLocation,), location)
end

"""
    create_module(ctx::MlirContext) -> MlirModule

Create an empty MLIR module with unknown location.
"""
function create_module(ctx::MlirContext)
    # Get unknown location
    loc = ccall((:mlirLocationUnknownGet, libJLCS), MlirLocation, (MlirContext,), ctx)
    return create_module(ctx, loc)
end

"""
    parse_module(ctx::MlirContext, source::String) -> MlirModule

Parse an MLIR module from a string.
"""
function parse_module(ctx::MlirContext, source::String)
    mod = ccall((:jlcsModuleCreateParse, libJLCS), MlirModule, (MlirContext, Cstring), ctx, source)
    if mod == C_NULL
        error("Failed to parse MLIR module")
    end
    return mod
end

"""
    clone_module(mod::MlirModule) -> MlirModule

Clone an MLIR module.
"""
function clone_module(mod::MlirModule)
    return ccall((:jlcs_module_clone, libJLCS), MlirModule, (MlirModule,), mod)
end

"""
    get_module_operation(mlir_module::MlirModule) -> MlirOperation

Get the operation backing a module.
"""
function get_module_operation(mlir_module::MlirModule)
    return ccall((:mlirModuleGetOperation, libJLCS), MlirOperation, (MlirModule,), mlir_module)
end

"""
    print_module(mlir_module::MlirModule)

Print an MLIR module to stdout.
"""
function print_module(mlir_module::MlirModule)
    op = get_module_operation(mlir_module)
    ccall((:mlirOperationDump, libJLCS), Cvoid, (MlirOperation,), op)
end

# =============================================================================
# Introspection
# =============================================================================

function get_function_op(mod::MlirModule, name::String)
    return ccall((:jlcs_module_get_function, libJLCS), MlirOperation, (MlirModule, Cstring), mod, name)
end

function get_function_type(op::MlirOperation)
    return ccall((:jlcs_function_get_type, libJLCS), MlirType, (MlirOperation,), op)
end

function get_num_inputs(type::MlirType)
    return ccall((:jlcs_function_type_get_num_inputs, libJLCS), Int, (MlirType,), type)
end

function get_input_type(type::MlirType, index::Int)
    return ccall((:jlcs_function_type_get_input, libJLCS), MlirType, (MlirType, Int), type, index)
end

function is_integer(type::MlirType)
    return ccall((:jlcs_type_is_integer, libJLCS), Bool, (MlirType,), type)
end

function get_integer_width(type::MlirType)
    return ccall((:jlcs_integer_type_get_width, libJLCS), Int, (MlirType,), type)
end

function is_f32(type::MlirType)
    return ccall((:jlcs_type_is_f32, libJLCS), Bool, (MlirType,), type)
end

function is_f64(type::MlirType)
    return ccall((:jlcs_type_is_f64, libJLCS), Bool, (MlirType,), type)
end

# =============================================================================
# Transformations
# =============================================================================

"""
    lower_to_llvm(module::MlirModule) -> Bool

Run standard lowering passes (Func -> LLVM, Arith -> LLVM) on the module.
Returns true on success.
"""
function lower_to_llvm(mod::MlirModule)
    return ccall((:jlcs_lower_to_llvm, libJLCS), Bool, (MlirModule,), mod)
end

# =============================================================================
# JIT Execution Engine
# =============================================================================

"""
    create_jit(module::MlirModule; opt_level=2, dump_object=false) -> MlirExecutionEngine

Create a JIT execution engine for the module.
Automatically attaches host data layout.
"""
function create_jit(mod::MlirModule; opt_level::Int=2, dump_object::Bool=false)
    jit = ccall((:jlcs_create_jit, libJLCS), MlirExecutionEngine, 
                (MlirModule, Cint, Bool), mod, opt_level, dump_object)
    if jit == C_NULL
        error("Failed to create JIT execution engine")
    end
    return jit
end

"""
    destroy_jit(jit::MlirExecutionEngine)

Destroy the JIT execution engine.
"""
function destroy_jit(jit::MlirExecutionEngine)
    ccall((:jlcs_destroy_jit, libJLCS), Cvoid, (MlirExecutionEngine,), jit)
end

"""
    register_symbol(jit::MlirExecutionEngine, name::String, addr::Ptr{Cvoid})

Register a runtime address (symbol) with the JIT.
Call this BEFORE invoking JIT functions that rely on external symbols.
"""
function register_symbol(jit::MlirExecutionEngine, name::String, addr::Ptr{Cvoid})
    ccall((:jlcs_jit_register_symbol, libJLCS), Cvoid, 
          (MlirExecutionEngine, Cstring, Ptr{Cvoid}), jit, name, addr
    )
end

"""
    lookup(jit::MlirExecutionEngine, name::String) -> Ptr{Cvoid}

Lookup a function address in the JIT.
"""
function lookup(jit::MlirExecutionEngine, name::String)
    return ccall((:jlcs_jit_lookup, libJLCS), Ptr{Cvoid}, 
                 (MlirExecutionEngine, Cstring), jit, name
    )
end

"""
    jit_invoke(jit::MlirExecutionEngine, name::String, args::Vector{Any})

Invoke a JIT function with arguments.
Note: Arguments must be pointers to the actual values (double indirection).
"""
function jit_invoke(jit::MlirExecutionEngine, name::String, args::Vector{Ptr{Cvoid}})
    # Pack arguments into an array of pointers
    # implementation detail: jlcs_jit_invoke expects void**
    return ccall((:jlcs_jit_invoke, libJLCS), Bool,
                 (MlirExecutionEngine, Cstring, Ptr{Ptr{Cvoid}}), jit, name, pointer(args))
end

"""
    invoke_safe(jit::MlirExecutionEngine, mod::MlirModule, name::String, args...)

Safely invoke a JIT function by verifying argument types against the MLIR module signature.
"""
function invoke_safe(jit::MlirExecutionEngine, mod::MlirModule, name::String, args...)
    # 1. Lookup function operation
    op = get_function_op(mod, name)
    if op == C_NULL
        error("Function '$name' not found in module")
    end

    # 2. Get function type
    ftype = get_function_type(op)
    if ftype == C_NULL
        error("Could not determine type for function '$name'")
    end

    num_inputs = get_num_inputs(ftype)
    
    # Check argument count (excluding return value buffer for now)
    if length(args) != num_inputs + 1
         # We expect N inputs + 1 return buffer for typical JIT invoke usage
         # If the function is void, currently we still might need to match signature carefully.
         # For simplicity, we assume strict packed invoke convention: args... + ret
    end

    # 3. Verify arguments
    # Note: args includes the return buffer at the end!
    
    ptr_args = Vector{Ptr{Cvoid}}()
    
    for i in 1:num_inputs
        arg_val = args[i]
        expected_type = get_input_type(ftype, i - 1)
        
        # Type Check
        valid = false
        if is_integer(expected_type)
            width = get_integer_width(expected_type)
            if width == 32 && arg_val isa Int32
                valid = true
            elseif width == 64 && arg_val isa Int64
                valid = true
            end
        elseif is_f32(expected_type) && arg_val isa Float32
            valid = true
        elseif is_f64(expected_type) && arg_val isa Float64
            valid = true
        end
        
        if !valid
            error("Argument mismatch at index $i. Expected MLIR type compatible with provided value: $arg_val")
        end
        
        # Prepare pointer
        # We need to keep the Ref alive. In this simple function scope, it should be fine.
        # But `Ref(arg_val)` creates a new Ref each time.
        # For a safer implementation, we should put Refs in a list to preserve them.
    end
    
    # We need to create Refs for all arguments to pass their pointers
    # and keep them alive during the call.
    # The last argument is the return buffer (if non-void return).
    
    ref_args = Any[]
    ptr_args = Ptr{Cvoid}[]
    
    # Handle Inputs
    for i in 1:num_inputs
        r = Ref(args[i])
        push!(ref_args, r)
        push!(ptr_args, unsafe_convert(Ptr{Cvoid}, r))
    end
    
    # Handle Return Buffer (Last argument)
    # We assume the user passed a Ref for the return value
    ret_buffer = args[end]
    if !(ret_buffer isa Ref)
         error("Last argument must be a Ref for the return value")
    end
    
    push!(ptr_args, unsafe_convert(Ptr{Cvoid}, ret_buffer))
    
    # Invoke
    return jit_invoke(jit, name, ptr_args)
end

# =============================================================================
# Testing and Diagnostics
# =============================================================================

"""
    test_dialect()

Test that the JLCS dialect loads and works correctly.

This creates a context, loads the dialect, and verifies basic functionality.
"""
function test_dialect()
    println("="^70)
    println(" JLCS MLIR Dialect Test")
    println("="^70)

    # Step 1: Check library exists
    print("Checking library... ")
    try
        check_library()
        println("✓")
    catch e
        println("✗")
        rethrow(e)
    end

    # Step 2: Create context
    print("Creating MLIR context... ")
    ctx = create_context()
    println("✓")

    # Step 3: Create module
    print("Creating MLIR module... ")
    mod = create_module(ctx)
    println("✓")

    # Step 4: Print module (should be empty)
    println("\nEmpty module:")
    print_module(mod)

    # Step 5: Create jlcs.type_info operation via parsing
    println("\nTesting parsing of jlcs.type_info...")
    ir = """
    module {
      jlcs.type_info @TestClass {
        size = 8 : i64,
        vtable_offset = 0 : i64,
        vtable_addr = 0x1234 : i64
      }
    }"""
    
    parsed_mod = parse_module(ctx, ir)
    print_module(parsed_mod)
    println("✓ Parsed successfully")

    # Step 6: Cleanup
    print("\nCleaning up... ")
    destroy_context(ctx)
    println("✓")

    println("\n" * "="^70)
    println(" All tests passed!")
    println("="^70)

    return true
end

# =============================================================================
# REPL Helpers
# =============================================================================

"""
    @with_context(body)

Execute body with an MLIR context, automatically cleaning up afterwards.

Example:
```julia
@with_context begin
    mod = create_module(ctx)
    print_module(mod)
end
```
"""
macro with_context(body)
    quote
        ctx = create_context()
        try
            $(esc(body))
        finally
            destroy_context(ctx)
        end
    end
end

end # module MLIRNative