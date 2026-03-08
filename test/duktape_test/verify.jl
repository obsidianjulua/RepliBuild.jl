#!/usr/bin/env julia
# test/duktape_test/verify.jl — Verify Duktape wrapper via ccall
#
# Tests:
#   1. Create a Duktape heap (context)
#   2. Evaluate a simple JS expression
#   3. Push/pop values on the Duktape stack
#   4. Destroy the context cleanly
#
# Duktape's public API macros (duk_create_heap_default, duk_peval_string, etc.)
# aren't visible to the wrapper because they're preprocessor macros.
# We call the underlying C functions directly.

using Test
using Libdl

const SCRIPT_DIR = @__DIR__

# Load generated wrapper
wrapper_path = joinpath(SCRIPT_DIR, "julia", "DuktapeTest.jl")
if !isfile(wrapper_path)
    error("Wrapper not found at $wrapper_path — did you run build+wrap?")
end

include(wrapper_path)
using .DuktapeTest

"""Create a default Duktape heap (equivalent to duk_create_heap_default macro)."""
create_heap() = DuktapeTest.duk_create_heap(C_NULL, C_NULL, C_NULL, C_NULL, C_NULL)

"""Protected eval of a JS string (equivalent to duk_peval_string macro)."""
peval_string(ctx, src::String) = DuktapeTest.duk_peval_string(ctx, src)

@testset "Duktape Verification" begin

    # ── 1. Context creation ──────────────────────────────────────────────────
    @testset "Context lifecycle" begin
        ctx = create_heap()
        @test ctx != C_NULL
        DuktapeTest.duk_destroy_heap(ctx)
    end

    # ── 2. Simple JS evaluation ──────────────────────────────────────────────
    @testset "Eval JS expression" begin
        ctx = create_heap()
        @test ctx != C_NULL

        ret = peval_string(ctx, "1 + 2 + 3")
        @test ret == 0

        result = DuktapeTest.duk_get_number(ctx, Int32(-1))
        @test result ≈ 6.0

        DuktapeTest.duk_pop(ctx)
        DuktapeTest.duk_destroy_heap(ctx)
    end

    # ── 3. String evaluation ─────────────────────────────────────────────────
    @testset "Eval JS string concatenation" begin
        ctx = create_heap()
        @test ctx != C_NULL

        ret = peval_string(ctx, "'hello' + ' ' + 'world'")
        @test ret == 0

        result = DuktapeTest.duk_get_string(ctx, Int32(-1))
        @test unsafe_string(result) == "hello world"

        DuktapeTest.duk_pop(ctx)
        DuktapeTest.duk_destroy_heap(ctx)
    end

    # ── 4. Stack push/pop operations ─────────────────────────────────────────
    @testset "Stack push/pop" begin
        ctx = create_heap()
        @test ctx != C_NULL

        DuktapeTest.duk_push_int(ctx, Int32(42))
        top = DuktapeTest.duk_get_top(ctx)
        @test top == 1

        val = DuktapeTest.duk_get_int(ctx, Int32(-1))
        @test val == 42

        DuktapeTest.duk_pop(ctx)
        top = DuktapeTest.duk_get_top(ctx)
        @test top == 0

        DuktapeTest.duk_destroy_heap(ctx)
    end

    # ── 5. Boolean operations ────────────────────────────────────────────────
    @testset "Boolean push/get" begin
        ctx = create_heap()
        @test ctx != C_NULL

        DuktapeTest.duk_push_boolean(ctx, UInt32(1))
        val = DuktapeTest.duk_get_boolean(ctx, Int32(-1))
        @test val != 0

        DuktapeTest.duk_pop(ctx)
        DuktapeTest.duk_destroy_heap(ctx)
    end

    # ── 6. Error handling ────────────────────────────────────────────────────
    @testset "Eval syntax error returns nonzero" begin
        ctx = create_heap()
        @test ctx != C_NULL

        ret = peval_string(ctx, "function {{{ invalid")
        @test ret != 0

        DuktapeTest.duk_pop(ctx)
        DuktapeTest.duk_destroy_heap(ctx)
    end

    # ── 7. Multiple evals on same context ────────────────────────────────────
    @testset "Multiple evals" begin
        ctx = create_heap()
        @test ctx != C_NULL

        # First eval
        ret1 = peval_string(ctx, "2 * 21")
        @test ret1 == 0
        @test DuktapeTest.duk_get_number(ctx, Int32(-1)) ≈ 42.0
        DuktapeTest.duk_pop(ctx)

        # Second eval
        ret2 = peval_string(ctx, "Math.sqrt(144)")
        @test ret2 == 0
        @test DuktapeTest.duk_get_number(ctx, Int32(-1)) ≈ 12.0
        DuktapeTest.duk_pop(ctx)

        # Stack should be clean
        @test DuktapeTest.duk_get_top(ctx) == 0

        DuktapeTest.duk_destroy_heap(ctx)
    end

    println("✓ Duktape Verification Passed")
end
