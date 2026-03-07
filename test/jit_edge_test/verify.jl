# JIT Edge Case Verification
# Expects: wrapper already generated at julia/JitEdgeTest.jl

using Test

wrapper_path = joinpath(@__DIR__, "julia", "JitEdgeTest.jl")
if !isfile(wrapper_path)
    error("Wrapper not found at $wrapper_path. Did you run discover + build + wrap?")
end

include(wrapper_path)
using .JitEdgeTest

@testset "JIT Edge Case Verification" begin
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
end
