#!/usr/bin/env julia
# test/rust_test/verify.jl — Manual ccall verification of Rust cdylib
#
# This is the "gold standard" — hand-written ccall wrappers proving
# the Rust ABI works with Julia. A future GeneratorRust.jl should
# produce equivalent code automatically from DWARF + symbol introspection.

using Test
using Libdl

const SCRIPT_DIR = @__DIR__
const LIBPATH = joinpath(SCRIPT_DIR, "librust_test.so")
if !isfile(LIBPATH)
    error("Library not found at $LIBPATH — run: rustc --crate-type cdylib --edition 2021 -g -o librust_test.so src/lib.rs")
end

# repr(C) struct definitions (must be at top level)
struct Point
    x::Cdouble
    y::Cdouble
end

struct Rect
    origin_x::Cdouble
    origin_y::Cdouble
    size_x::Cdouble
    size_y::Cdouble
end

@testset "Rust FFI Pipeline" begin

    # ── 1. Scalar functions ──────────────────────────────────────────────
    @testset "Scalars" begin
        result = ccall((:add, LIBPATH), Cint, (Cint, Cint), 3, 7)
        @test result == 10

        product = ccall((:multiply_f64, LIBPATH), Cdouble, (Cdouble, Cdouble), 2.5, 4.0)
        @test product ≈ 10.0

        @test ccall((:is_positive, LIBPATH), Bool, (Cint,), 5) == true
        @test ccall((:is_positive, LIBPATH), Bool, (Cint,), -1) == false
        @test ccall((:is_positive, LIBPATH), Bool, (Cint,), 0) == false
    end

    # ── 2. repr(C) Structs ───────────────────────────────────────────────
    @testset "Structs" begin
        # point_new returns { double, double } in registers (16 bytes)
        p = ccall((:point_new, LIBPATH), Point, (Cdouble, Cdouble), 3.0, 4.0)
        @test p.x ≈ 3.0
        @test p.y ≈ 4.0

        # point_distance takes two &Point (ptr align 8)
        a = Ref(Point(0.0, 0.0))
        b = Ref(Point(3.0, 4.0))
        dist = GC.@preserve a b begin
            ccall((:point_distance, LIBPATH), Cdouble,
                  (Ptr{Point}, Ptr{Point}),
                  Base.unsafe_convert(Ptr{Point}, a),
                  Base.unsafe_convert(Ptr{Point}, b))
        end
        @test dist ≈ 5.0

        # Rect: nested struct (flattened to 4 doubles)
        r = Ref(Rect(0.0, 0.0, 10.0, 5.0))
        area = GC.@preserve r begin
            ccall((:rect_area, LIBPATH), Cdouble, (Ptr{Rect},),
                  Base.unsafe_convert(Ptr{Rect}, r))
        end
        @test area ≈ 50.0
    end

    # ── 3. String interop ────────────────────────────────────────────────
    @testset "Strings" begin
        len = ccall((:string_length, LIBPATH), Cint, (Cstring,), "hello")
        @test len == 5

        @test ccall((:string_length, LIBPATH), Cint, (Ptr{UInt8},), C_NULL) == -1

        # greet returns a heap-allocated C string
        raw_ptr = ccall((:greet, LIBPATH), Ptr{UInt8}, (Cstring,), "julia")
        @test raw_ptr != C_NULL
        greeting = unsafe_string(raw_ptr)
        @test greeting == "hello julia"
        # Free through Rust's allocator
        ccall((:free_string, LIBPATH), Cvoid, (Ptr{UInt8},), raw_ptr)
    end

    # ── 4. C-compatible enum ─────────────────────────────────────────────
    @testset "Enums" begin
        # Color: Red=0, Green=1, Blue=2
        red_name = ccall((:color_name, LIBPATH), Ptr{UInt8}, (Cint,), 0)
        @test unsafe_string(red_name) == "red"
        green_name = ccall((:color_name, LIBPATH), Ptr{UInt8}, (Cint,), 1)
        @test unsafe_string(green_name) == "green"
        blue_name = ccall((:color_name, LIBPATH), Ptr{UInt8}, (Cint,), 2)
        @test unsafe_string(blue_name) == "blue"
    end

    # ── 5. Array via raw pointer ─────────────────────────────────────────
    @testset "Arrays" begin
        data = Cint[1, 2, 3, 4, 5]
        total = GC.@preserve data begin
            ccall((:sum_array, LIBPATH), Clonglong,
                  (Ptr{Cint}, Csize_t),
                  pointer(data), length(data))
        end
        @test total == 15

        # Null/empty safety
        @test ccall((:sum_array, LIBPATH), Clonglong, (Ptr{Cint}, Csize_t), C_NULL, 0) == 0
    end

    # ── 6. Opaque type behind pointer ────────────────────────────────────
    @testset "Opaque types" begin
        c = ccall((:counter_new, LIBPATH), Ptr{Cvoid}, (Clonglong,), 10)
        @test c != C_NULL

        @test ccall((:counter_get, LIBPATH), Clonglong, (Ptr{Cvoid},), c) == 10

        ccall((:counter_increment, LIBPATH), Cvoid, (Ptr{Cvoid},), c)
        ccall((:counter_increment, LIBPATH), Cvoid, (Ptr{Cvoid},), c)
        ccall((:counter_increment, LIBPATH), Cvoid, (Ptr{Cvoid},), c)
        @test ccall((:counter_get, LIBPATH), Clonglong, (Ptr{Cvoid},), c) == 13

        ccall((:counter_free, LIBPATH), Cvoid, (Ptr{Cvoid},), c)
    end

    println("✓ Rust FFI Pipeline Passed")
end
