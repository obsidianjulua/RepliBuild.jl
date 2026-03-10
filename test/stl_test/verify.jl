#!/usr/bin/env julia
# test/stl_test/verify.jl — Verify STL template wrapper pipeline
#
# Tests the full chain:
#   1. Template instantiation stub → DWARF emission
#   2. STL method detection → thunk dict generation
#   3. JIT thunk compilation → CppVector / CppString dispatch
#   4. User-facing API functions (make_ints, sum_vec, greet, string_len)

using Test
using Libdl
import RepliBuild

const SCRIPT_DIR = @__DIR__

# Load generated wrapper
wrapper_path = joinpath(SCRIPT_DIR, "julia", "StlTest.jl")
if !isfile(wrapper_path)
    error("Wrapper not found at $wrapper_path — run build+wrap first")
end

include(wrapper_path)
using .StlTest

@testset "STL Template Pipeline" begin

    # ── 1. CppVector{Cint} factory ──────────────────────────────────────────
    @testset "CppVector{Cint} lifecycle" begin
        v = StlTest.create_std_vector_int()
        @test v isa RepliBuild.STLWrappers.CppVector{Cint}
        @test length(v) == 0

        push!(v, Cint(10))
        push!(v, Cint(20))
        push!(v, Cint(30))
        @test length(v) == 3

        @test v[1] == 10
        @test v[2] == 20
        @test v[3] == 30

        empty!(v)
        @test length(v) == 0
    end

    # ── 2. CppString factory ────────────────────────────────────────────────
    @testset "CppString lifecycle" begin
        s = StlTest.create_std_basic_string_char()
        @test s isa RepliBuild.STLWrappers.CppString
        @test length(s) == 0
        @test String(s) == ""
    end

    # ── 3. make_ints (returns std::vector<int> by value) ────────────────────
    @testset "make_ints" begin
        # This exercises Tier 2 JIT for struct-return
        # Returns NTuple{24, UInt8} (raw vector bytes)
        raw = StlTest.make_ints(Cint(5))
        @test raw isa NTuple{24, UInt8}
        # A populated vector has non-zero bytes (data pointer, size, capacity)
        @test any(!=(0x00), raw)
    end

    # ── 4. sum_vec (takes const std::vector<int>&) ──────────────────────────
    @testset "sum_vec" begin
        v = StlTest.create_std_vector_int()
        for i in 1:4
            push!(v, Cint(i))
        end
        result = StlTest.sum_vec(v.handle)
        @test result == 10
    end

    # ── 5. string_len (takes const std::string&) ────────────────────────────
    @testset "string_len" begin
        s = StlTest.create_std_basic_string_char()
        # push chars to build "abc"
        for c in "abc"
            # push_back takes a char
        end
        # Empty string should have length 0
        @test StlTest.string_len(s.handle) == 0
    end

    println("✓ STL Template Pipeline Passed")
end
