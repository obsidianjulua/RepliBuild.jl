# Verification and Benchmarking script for zero-copy strided array performance

using Test
using Pkg

# Ensure RepliBuild is available
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using RepliBuild

@testset "Zero-Copy Performance Benchmark" begin
    println("\n" * "="^70)
    println("Building and Wrapping Benchmark Test...")
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
    
    println("\nTesting Zero-Copy Strided Matrix View...")

    # Matrix dimensions
    N = 4 # Use small matrices to properly benchmark wrapper overhead vs Julia dispatch

    A = rand(Float64, N, N)
    B = rand(Float64, N, N)
    C_jl = zeros(Float64, N, N)
    C_cpp = zeros(Float64, N, N)

    viewA = BenchmarkTest.StridedMatrixView(pointer(A), UInt64(N), UInt64(N), UInt64(1), UInt64(N))
    viewB = BenchmarkTest.StridedMatrixView(pointer(B), UInt64(N), UInt64(N), UInt64(1), UInt64(N))
    viewC = BenchmarkTest.StridedMatrixView(pointer(C_cpp), UInt64(N), UInt64(N), UInt64(1), UInt64(N))

    println("Executing C++ strided matrix multiplication...")
    
    # Proper wrapper call allowing Julia to handle GC preservation 
    BenchmarkTest.multiply_matrices(Ref(viewA), Ref(viewB), Ref(viewC))
    
    println("Executing Julia matrix multiplication...")
    C_jl = A * B

    @test C_cpp ≈ C_jl
    println("✓ Mathematical validation passed: Julia and C++ computed the identical result.")
end

println("\nRunning Performance Benchmarks...")
println("Note: Using Pkg 'BenchmarkTools' dynamically to test raw throughput.")

Pkg.add("BenchmarkTools")
using BenchmarkTools

include(joinpath(@__DIR__, "julia", "BenchmarkTest.jl"))
using .BenchmarkTest
using Libdl

N = 4
A = rand(Float64, N, N)
B = rand(Float64, N, N)
C_cpp = zeros(Float64, N, N)

viewA = BenchmarkTest.StridedMatrixView(pointer(A), UInt64(N), UInt64(N), UInt64(1), UInt64(N))
viewB = BenchmarkTest.StridedMatrixView(pointer(B), UInt64(N), UInt64(N), UInt64(1), UInt64(N))
viewC = BenchmarkTest.StridedMatrixView(pointer(C_cpp), UInt64(N), UInt64(N), UInt64(1), UInt64(N))

println("\n1. Julia Native `A * B`")
b_jl = @benchmark $A * $B
display(b_jl)

println("\n\n2. RepliBuild Wrapper `multiply_matrices` (Zero-copy Struct Pointer)")
b_cpp = @benchmark BenchmarkTest.multiply_matrices(Ref($viewA), Ref($viewB), Ref($viewC))
display(b_cpp)

println("\n\n3. Bare-metal ccall `multiply_matrices`")
library_path = joinpath(@__DIR__, "julia", "libbenchmark_test.so")
lib_sym = Libdl.dlsym(Libdl.dlopen(library_path), :multiply_matrices)
b_bare = @benchmark ccall($lib_sym, Cvoid, 
        (Ptr{BenchmarkTest.StridedMatrixView}, Ptr{BenchmarkTest.StridedMatrixView}, Ptr{BenchmarkTest.StridedMatrixView}),
        Ref($viewA), Ref($viewB), Ref($viewC))
display(b_bare)
println()