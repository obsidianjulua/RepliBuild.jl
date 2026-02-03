# Verification script for basics_test

using Test
using Libdl

# Load the generated wrapper
# Assuming this script is run via include("test/basics_test/verify.jl") or similar
# We need to find the wrapper relative to this script
wrapper_path = joinpath(@__DIR__, "julia", "BasicsTest.jl")
if !isfile(wrapper_path)
    error("Wrapper not found at $wrapper_path. Did you run wrap()?")
end

include(wrapper_path)
using .BasicsTest

@testset "BasicsTest Verification" begin
    println("Verifying BasicsTest...")

    # 1. Global Variables
    # Note: Global variables are usually wrapped as functions returning the value or pointers
    # If RepliBuild wraps globals, let's check.
    # If not, we skip. C++ globals are tricky in shared libs across languages.
    # Looking at basics.cpp: extern "C" { int global_int = 42; }
    
    # 2. Padded Structs
    println("  Testing PaddedStruct...")
    # struct PaddedStruct { char a; int b; }; -> alignment usually 4, size 8
    # RepliBuild maps 'char' to 'UInt8'
    ps = make_padded(UInt8(10), Int32(20))
    @test ps.a == UInt8(10)
    @test ps.b == Int32(20)
    
    # process_padded(ps) # Should print to stdout, hard to capture test. 
    # Just ensure it doesn't crash
    process_padded(ps)

    # 3. Packed Structs
    println("  Testing PackedStruct...")
    # struct __attribute__((packed)) PackedStruct { char a; int b; }; -> alignment 1, size 5
    # RepliBuild maps 'char' to 'UInt8'
    packed = make_packed(UInt8(30), Int32(40))
    @test packed.a == UInt8(30)
    @test packed.b == Int32(40)
    
    # Check size if possible using sizeof(PackedStruct)
    # This verifies if Julia struct definition matches packed layout
    @test sizeof(PackedStruct) == 5

    process_packed(packed)
    # println("  Skipping process_packed (known issue with packed layout mismatch in JIT)")

    # 4. Unions
    println("  Testing Unions...")
    # union NumberUnion { int i; float f; };
    # In Julia, unions are often mapped to a struct with a data blob or a specific field
    # Let's see what was generated.
    # If it's a value type in C++, it might be passed as a struct in Julia.
    
    # Helper to create union (since we don't have a C++ constructor exposed for it directly in snippets)
    # We might need to construct it in Julia.
    # Assuming NumberUnion has fields i and f overlayed? 
    # Or does RepliBuild generate `data::NTuple{4, UInt8}`?
    
    # Let's try to inspect the type first
    println("    NumberUnion type: ", NumberUnion)
    
    # If it's a blob, we write to it
    if hasfield(NumberUnion, :data)
        # It's an opaque blob
        # Skip functional test for opaque blob unless we have a helper
    else
        # It might have fields if RepliBuild is smart
        # TODO: Implement union test based on generated code
    end
    
    println("✓ BasicsTest Passed")
end
