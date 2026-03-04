using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using RepliBuild
using Test

function my_julia_add(a::Cint, b::Cint)::Cint
    return a + b
end

@testset "Complex FFI Features Pipeline" begin
    println("Running discovery, build, and wrap for complex FFI features...")
    test_dir = @__DIR__
    
    # Discover, build, and wrap
    RepliBuild.discover(test_dir, build=true, wrap=true)
    
    wrapper_path = joinpath(test_dir, "julia", "CustomTest.jl")
    
    @test isfile(wrapper_path)
    println("Including generated wrapper...")
    include(wrapper_path)
    
    @testset "Bitfields" begin
        # RepliBuild generated a struct for HardwareRegister and handles the bits internally 
        # or via JIT dispatch if we mutate it. 
        # Actually, let's just initialize an empty one and pass it to C to modify.
        # Since it's passed by pointer, we use Ref.
        reg = Ref(CustomTest.HardwareRegister()) # Initialize with 0s using new default constructor
        CustomTest.init_register(reg)
        
        # Read it back using C
        payload = CustomTest.read_payload(reg)
        @test payload == UInt32(42000)
        println("Bitfield payload read via C: ", payload)

        # Or read via RepliBuild accessors if they exist:
        if isdefined(CustomTest, :get_enable)
            # DWARF doesn't always provide bitfield info cleanly, but RepliBuild tries.
            println("Bitfield extraction accessors found! Enable: ", CustomTest.get_enable(reg[]))
        end
    end
    
    @testset "Unions" begin
        # Unions map to NTuple{N, UInt8} with accessors
        # Create an empty union using the new default constructor
        v = Ref(CustomTest.VariantValue())
        
        # Set a float value from C
        CustomTest.set_float_variant(v, 3.14f0)
        
        # Read the float using RepliBuild generated typed accessors!
        julia_f = CustomTest.get_as_float(v[])
        @test isapprox(julia_f, 3.14f0)
        println("Union read via generated accessor: ", julia_f)

        # Try to read it as a double via C
        c_double = CustomTest.get_double_variant(v)
        # It's an overlapping cast, it will be garbage or exactly what float maps to
        println("Union double interpretation: ", c_double)
    end

    @testset "Callbacks" begin
        # Callback Test
        # Wrap the function so it can be passed to C as a function pointer
        cb_ptr = @cfunction(my_julia_add, Cint, (Cint, Cint))
        
        res = CustomTest.apply_callback(Int32(20), Int32(22), cb_ptr)
        @test res == Int32(42)
        println("apply_callback(20, 22) = ", res)
    end
end
