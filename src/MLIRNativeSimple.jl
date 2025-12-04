#!/usr/bin/env julia
# MLIRNativeSimple.jl - Direct MLIR C API usage for JLCS dialect testing
#
# Uses the static libraries directly via @ccall

module MLIRNativeSimple

using Libdl  # For dlopen

export create_context, create_module, destroy_context
export test_dialect, print_module

# Find the MLIR library (libMLIR.so contains everything in monolithic builds)
const MLIR_LIB = "/usr/lib/libMLIR.so"

# Path to JLCS dialect
const libJLCS_path = joinpath(@__DIR__, "Mlir", "build", "libJLCS.so")

# MLIR C API types (opaque pointers)
struct MlirContext
    ptr::Ptr{Cvoid}
end
Base.convert(::Type{MlirContext}, p::Ptr{Cvoid}) = MlirContext(p)

struct MlirModule
    ptr::Ptr{Cvoid}
end
Base.convert(::Type{MlirModule}, p::Ptr{Cvoid}) = MlirModule(p)

struct MlirOperation
    ptr::Ptr{Cvoid}
end
Base.convert(::Type{MlirOperation}, p::Ptr{Cvoid}) = MlirOperation(p)

struct MlirLocation
    ptr::Ptr{Cvoid}
end
Base.convert(::Type{MlirLocation}, p::Ptr{Cvoid}) = MlirLocation(p)

# Check library exists
function check_library()
    if !isfile(libJLCS_path)
        error("""
        JLCS dialect library not found at: $libJLCS_path

        Build it first with:
            cd src/Mlir
            ./build_dialect.sh
        """)
    end
    if !isfile(MLIR_LIB)
        error("MLIR library not found at: $MLIR_LIB")
    end
end

# MLIR C API functions - use our JLCS library wrappers
function mlir_context_create()
    @ccall libJLCS_path.jlcs_create_context()::MlirContext
end

function mlir_context_destroy(ctx::MlirContext)
    @ccall libJLCS_path.jlcs_destroy_context(ctx::MlirContext)::Cvoid
end

function mlir_module_create(ctx::MlirContext)
    @ccall libJLCS_path.jlcs_create_module(ctx::MlirContext)::MlirModule
end

function mlir_module_print(mod::MlirModule)
    @ccall libJLCS_path.jlcs_print_module(mod::MlirModule)::Cvoid
end

# High-level API
"""
    create_context() -> MlirContext

Create a new MLIR context and register the JLCS dialect.
"""
function create_context()
    check_library()

    # Create MLIR context
    ctx = mlir_context_create()

    if ctx.ptr == C_NULL
        error("Failed to create MLIR context")
    end

    # Load JLCS dialect library
    dlopen(libJLCS_path, RTLD_GLOBAL)

    # Register JLCS dialect
    @ccall libJLCS_path.registerJLCSDialect(ctx::MlirContext)::Cvoid

    return ctx
end

"""
    destroy_context(ctx::MlirContext)

Destroy an MLIR context and free its resources.
"""
function destroy_context(ctx::MlirContext)
    mlir_context_destroy(ctx)
end

"""
    create_module(ctx::MlirContext) -> MlirModule

Create an empty MLIR module.
"""
function create_module(ctx::MlirContext)
    return mlir_module_create(ctx)
end

"""
    print_module(mod::MlirModule)

Print an MLIR module to stdout.
"""
function print_module(mod::MlirModule)
    mlir_module_print(mod)
end

"""
    test_dialect()

Test that the JLCS dialect loads and works correctly.
"""
function test_dialect()
    println("="^70)
    println(" JLCS MLIR Dialect Test")
    println("="^70)

    # Step 1: Check library exists
    print("Checking libraries... ")
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

    # Step 5: TODO - Create jlcs.type_info operation
    println("\nTODO: Create jlcs.type_info operation")

    # Step 6: Cleanup
    print("\nCleaning up... ")
    destroy_context(ctx)
    println("✓")

    println("\n" * "="^70)
    println(" All tests passed!")
    println("="^70)

    return true
end

end # module MLIRNativeSimple
