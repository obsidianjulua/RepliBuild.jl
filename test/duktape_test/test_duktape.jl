using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using RepliBuild
using Test

@testset "Duktape Stress Test" begin
    println("\n" * "="^70)
    println("Building and Wrapping Duktape...")
    println("="^70)

    # Configuration path
    toml_path = joinpath(@__DIR__, "replibuild.toml")
    @test isfile(toml_path)
    
    # Run build
    library_path = RepliBuild.build(toml_path)
    @test isfile(library_path)
    
    # Run wrapper generation
    wrapper_path = RepliBuild.wrap(toml_path)
    @test isfile(wrapper_path)
    
    # Try to load the generated wrapper
    include(wrapper_path)
    
    # Use the compiled C API via wrapper to evaluate Javascript
    ctx = DuktapeTest.duk_create_heap(C_NULL, C_NULL, C_NULL, C_NULL, C_NULL)
    @test ctx != C_NULL
    
    # Evaluate a string: "21 * 2"
    # DUK_COMPILE_EVAL (8) | DUK_COMPILE_NOSOURCE (512) | DUK_COMPILE_STRLEN (1024) | DUK_COMPILE_NOFILENAME (2048) = 3592
    DuktapeTest.duk_eval_raw(ctx, "21 * 2", UInt64(0), UInt32(3592))
    
    # Get the integer result from the top of the stack (-1)
    res = DuktapeTest.duk_get_int(ctx, Int32(-1))
    
    @test res == 42
    println("JavaScript executed successfully: 21 * 2 = 42")
    
    # Clean up
    DuktapeTest.duk_destroy_heap(ctx)
end
