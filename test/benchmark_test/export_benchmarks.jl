# Comprehensive Benchmarking Suite for Zero-Copy Matrices

using Test
using Pkg

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using RepliBuild
using RepliBuild.Introspect

# Define dimensions to test
const SIZES = [2, 4, 16, 64, 128]

println("
" * "="^70)
println("Building Benchmark Test...")
println("="^70)

toml_path = joinpath(@__DIR__, "replibuild.toml")
RepliBuild.build(toml_path)
RepliBuild.wrap(toml_path)

include(joinpath(@__DIR__, "julia", "BenchmarkTest.jl"))
using .BenchmarkTest

# Directory to save exports for the documentation
export_dir = joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "benchmarks")
mkpath(export_dir)

function pure_julia_multiply(A, B)
    return A * B
end

function cpp_wrapper_multiply(vA, vB, vC)
    BenchmarkTest.multiply_matrices(Ref(vA), Ref(vB), Ref(vC))
end

@testset "Export Benchmarks" begin
    for N in SIZES
        println("
Benchmarking Matrix Size: $(N)x$(N)")

        A = rand(Float64, N, N)
        B = rand(Float64, N, N)
        C_cpp = zeros(Float64, N, N)

        viewA = BenchmarkTest.StridedMatrixView(pointer(A), UInt64(N), UInt64(N), UInt64(1), UInt64(N))
        viewB = BenchmarkTest.StridedMatrixView(pointer(B), UInt64(N), UInt64(N), UInt64(1), UInt64(N))
        viewC = BenchmarkTest.StridedMatrixView(pointer(C_cpp), UInt64(N), UInt64(N), UInt64(1), UInt64(N))

        # GC.@preserve keeps backing arrays alive while raw pointers are in use
        GC.@preserve A B C_cpp begin
            # Introspect Benchmarks (Single Call Overhead tracking)
            res_jl = benchmark(pure_julia_multiply, A, B; samples=5000, warmup=10)
            res_cpp = benchmark(cpp_wrapper_multiply, viewA, viewB, viewC; samples=5000, warmup=10)
        end

        # Export JSONs specifically tagged with their size
        export_json(res_jl, joinpath(export_dir, "julia_mul_$(N)x$(N).json"))
        export_json(res_cpp, joinpath(export_dir, "cpp_mul_$(N)x$(N).json"))

        println("  Julia Median: $(res_jl.median_time) ns")
        println("  C++ Median:   $(res_cpp.median_time) ns")
    end
end

println("
All benchmarks exported to: $export_dir")