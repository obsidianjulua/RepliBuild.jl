# Verification script for cross-boundary callbacks

using Test
using Pkg

# Ensure RepliBuild is available
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using RepliBuild

# Define a Julia function to be called from C++
function my_add(a::Cint, b::Cint)::Cint
    return a + b
end

# We will mutate a captured variable to prove the callback fired multiple times
global progress_updates = 0
global last_progress = 0.0f0

function my_progress(p::Cfloat)::Cvoid
    global progress_updates
    global last_progress
    progress_updates += 1
    last_progress = p
    return nothing
end

@testset "Callback Test (Julia -> C++ -> Julia)" begin
    println("\n" * "="^70)
    println("Building and Wrapping Callback Test...")
    println("="^70)

    toml_path = joinpath(@__DIR__, "replibuild.toml")
    @test isfile(toml_path)

    # Build and Wrap
    library_path = RepliBuild.build(toml_path)
    @test isfile(library_path)
    
    wrapper_path = RepliBuild.wrap(toml_path)
    @test isfile(wrapper_path)
    
    # Load wrapper
    include(wrapper_path)
    
    println("\nTesting BinaryOp Callback...")
    
    # Create a C function pointer for the Julia function
    c_add = @cfunction(my_add, Cint, (Cint, Cint))
    
    # Call the C++ function, passing the Julia function pointer
    res = CallbackTest.execute_binary_op(Base.unsafe_convert(Ptr{Cvoid}, c_add), Int32(10), Int32(20))
    @test res == 30
    println("✓ BinaryOp callback executed successfully: 10 + 20 = $res")

    println("\nTesting Progress Callback...")
    
    global progress_updates = 0
    global last_progress = 0.0f0
    
    c_progress = @cfunction(my_progress, Cvoid, (Cfloat,))
    
    CallbackTest.simulate_work(Int32(5), Base.unsafe_convert(Ptr{Cvoid}, c_progress))
    
    @test progress_updates == 5
    @test last_progress ≈ 1.0f0
    
    println("✓ Progress callback fired $progress_updates times, final progress = $last_progress")
end
