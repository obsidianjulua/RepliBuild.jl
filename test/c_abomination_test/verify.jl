using Test

# Load the generated wrapper
include("julia/CAbominationTest.jl")
using .CAbominationTest

# Define global callback functions for @cfunction
function my_inner(b::Cfloat)::Cdouble
    return Cdouble(b * 2.0)
end

function my_outer(a::Cint)::Ptr{Cvoid}
    # Ignore a, just return the inner function pointer
    inner_cfunc = @cfunction(my_inner, Cdouble, (Cfloat,))
    return Base.unsafe_convert(Ptr{Cvoid}, inner_cfunc)
end

@testset "C Abomination Tests" begin
    @testset "NightmareStruct instantiation and mutation" begin
        # Create it (pass by value returned from C)
        n = CAbominationTest.create_nightmare(Int32(42), 1.0f0, 2.0f0, 3.0f0)
        
        # Verify the structure was passed correctly
        @test n.id == 42
        
        # We can't access inner members easily if they are within a massive byte blob/unions,
        # but let's check that passing the reference back to C works
        
        # Use Ref to simulate a pointer modification
        n_ref = Ref(n)
        CAbominationTest.mutate_nightmare(n_ref)
        
        # The id should be mutated
        @test n_ref[].id == 43
    end
    
    @testset "Opaque State" begin
        # Init opaque pointer
        state = CAbominationTest.init_opaque()
        @test state != C_NULL
        
        # Process it (takes state and NULL self_ref)
        CAbominationTest.process_opaque(state, C_NULL)
        
        # Process again
        CAbominationTest.process_opaque(state, C_NULL)
        
        # `state` is a Ptr{OpaqueState}. To use getproperty we need the struct.
        state_obj = unsafe_load(state)
        @test state_obj.counter == 2
        
        # Free it
        CAbominationTest.free_opaque(state, C_NULL)
    end
    
    @testset "Function Pointers" begin
        outer_cfunc = @cfunction(my_outer, Ptr{Cvoid}, (Cint,))
        
        # Execute it
        res = CAbominationTest.execute_outer(outer_cfunc, Int32(10), 5.0f0)
        
        # It should call my_outer(10), which returns my_inner, which is called with 5.0 -> 10.0
        @test res == 10.0
    end
end

println("✓ C Abomination Test Passed")
