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
        CAbominationTest.free_opaque(state)
    end

    @testset "Wrapper arity matches the C declaration" begin
        # free_opaque takes exactly ONE parameter (c_abomination.h:69). It came
        # out with a phantom second parameter `next::Ptr{SelfReferential}` —
        # the first MEMBER of a struct declared after it — because the DWARF
        # parser's parameter context outlived its DIE and the recorded
        # parameter array was still live. The wrapper then emitted a
        # two-argument ccall against a one-argument function.
        #
        # Assert on the method table, not on a call: a call that happens to
        # survive a wrong ccall signature proves nothing.
        for m in methods(CAbominationTest.free_opaque)
            @test m.nargs - 1 == 1   # nargs counts the function object itself
        end

        # Neighbours in the same header, to catch a mis-attribution that shifts
        # rather than duplicates.
        for m in methods(CAbominationTest.process_opaque)
            @test m.nargs - 1 == 2
        end
        for m in methods(CAbominationTest.init_opaque)
            @test m.nargs - 1 == 0
        end
    end
    
    @testset "Function Pointers" begin
        outer_cfunc = @cfunction(my_outer, Ptr{Cvoid}, (Cint,))

        # Execute it
        res = CAbominationTest.execute_outer(outer_cfunc, Int32(10), 5.0f0)

        # It should call my_outer(10), which returns my_inner, which is called with 5.0 -> 10.0
        @test res == 10.0
    end

    @testset "libc types in the API surface (§8)" begin
        # The load-blocking class: FILE resolves to `struct _IO_FILE`, which is
        # on _INTERNAL_TYPE_BLOCKLIST and never declared. Before the fix the
        # blocklist suppressed the declaration but not the USES, so
        # Ptr{Ptr{_IO_FILE}} reached the ccall tuple and the module raised
        # UndefVarError at include — killing every function in it, which is why
        # simply reaching this line is most of the assertion.
        @test isdefined(CAbominationTest, :stream_open)
        @test isdefined(CAbominationTest, :stream_write_tag)

        # FILE** degrades to Ptr{Ptr{Cvoid}}: undeclared leaf swapped, pointer
        # depth preserved, ABI unchanged.
        src = read(joinpath(@__DIR__, "julia", "CAbominationTest.jl"), String)
        code = replace(src, r"\"\"\".*?\"\"\""s => "")     # docstrings may name the C type
        @test !occursin("_IO_FILE", code)
        @test occursin("(Ptr{Ptr{Cvoid}}, Ptr{UInt8},)", code)

        # Still callable, and the pointer round-trips.
        out = Ref(Ptr{Cvoid}(C_NULL))
        @test CAbominationTest.stream_open(out, "/nonexistent") == 0
        @test CAbominationTest.stream_open(C_NULL, "/nonexistent") == -1
        @test CAbominationTest.stream_null() == Ptr{Cvoid}(C_NULL)
    end
end

println("✓ C Abomination Test Passed")
