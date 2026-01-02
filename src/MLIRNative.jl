#!/usr/bin/env julia
# MLIRNative.jl - Julia bindings to JLCS MLIR Dialect
#
# Low-level ccall interface to MLIR C API and custom JLCS dialect
# Part of RepliBuild.jl toolchain for advanced FFI code generation

module MLIRNative

export create_context, create_module, destroy_context, parse_module
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
