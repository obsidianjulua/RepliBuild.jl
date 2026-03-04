# JIT Edge Case Verification
# Tests small functions through Tier 1 (ccall) wrappers for correctness

using Test
using Pkg

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using RepliBuild

@testset "JIT Edge Cases" begin
    println("\n" * "="^70)
    println("Building JIT Edge Test...")
    println("="^70)

    test_dir = @__DIR__
    toml_path = joinpath(test_dir, "replibuild.toml")

    # Clean previous artifacts
    for d in ["build", "julia", ".replibuild_cache", "replibuild.toml"]
        p = joinpath(test_dir, d)
        ispath(p) && rm(p, recursive=true, force=true)
    end

    # Discover → Build → Wrap
    toml_path = RepliBuild.discover(test_dir, force=true, build=true, wrap=true)
    @test isfile(toml_path)

    wrapper_path = joinpath(test_dir, "julia", "JitEdgeTest.jl")
    @test isfile(wrapper_path)

    include(wrapper_path)
    using .JitEdgeTest

    @testset "scalar_add" begin
        result = JitEdgeTest.scalar_add(Cint(3), Cint(7))
        @test result == 10
        @test result isa Cint
    end

    @testset "scalar_mul" begin
        result = JitEdgeTest.scalar_mul(Cdouble(2.5), Cdouble(4.0))
        @test result == 10.0
        @test result isa Cdouble
    end

    @testset "identity" begin
        result = JitEdgeTest.identity(Cint(42))
        @test result == 42
    end

    @testset "write_sum" begin
        a = Ref(Cint(10))
        b = Ref(Cint(20))
        out = Ref(Cint(0))
        JitEdgeTest.write_sum(a, b, out)
        @test out[] == 30
    end

    @testset "make_pair (struct return)" begin
        pair = JitEdgeTest.make_pair(Cint(11), Cint(22))
        @test pair.first == 11
        @test pair.second == 22
    end

    @testset "pack_three (packed struct return)" begin
        triplet = JitEdgeTest.pack_three(UInt8('A'), Cint(999), UInt8('Z'))
        @test triplet.tag == UInt8('A')
        @test triplet.value == 999
        @test triplet.flag == UInt8('Z')
    end

    println("\n All JIT edge case tests passed.")
end
