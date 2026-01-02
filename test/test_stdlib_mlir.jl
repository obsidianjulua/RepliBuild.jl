using Test
using RepliBuild
using RepliBuild.JLCSIRGenerator
using RepliBuild.MLIRNative
using RepliBuild.DWARFParser

@testset "StdLib MLIR Generation" begin
    # 1. Setup paths
    test_dir = joinpath(@__DIR__, "stdlib_test")
    toml_path = joinpath(test_dir, "replibuild.toml")
    
    # 2. Build the project to generate metadata
    println("Building stdlib_test...")
    RepliBuild.discover(test_dir, force=true)
    RepliBuild.build(toml_path, clean=true)
    
    julia_dir = joinpath(test_dir, "julia")
    lib_path = joinpath(julia_dir, "libstdlib_test.so")
    
    # 3. Parse DWARF
    println("Parsing DWARF...")
    vtinfo = DWARFParser.parse_vtables(lib_path)
    
    # 4. Generate MLIR
    println("Generating MLIR...")
    ir = JLCSIRGenerator.generate_jlcs_ir(vtinfo)
    
    # Debug output
    println("Generated IR sample (first 20 lines):")
    println(join(first(split(ir, '\n'), 20), '\n'))
    
    # 5. Parse with MLIRNative
    println("Verifying with MLIR Parser...")
    ctx = create_context()
    
    # Register all dialects is handled by create_context now
    
    try
        # Attempt to parse the generated IR
        mod = parse_module(ctx, ir)
        @test mod != C_NULL
        println("✓ Successfully parsed generated MLIR module")
        
        # Print for visual inspection
        # print_module(mod)
    catch e
        println("Error parsing module: $e")
        rethrow(e)
    finally
        destroy_context(ctx)
    end
end
