#!/usr/bin/env julia
# test/test_multilib_jit.jl — two wrapped libraries in ONE session.
#
# GLOBAL_JIT was a single-engine singleton: the first wrapper's __init__ won,
# every later initialize_global_jit(other_binary) silently no-opped, and the
# second library's ENTIRE Tier 2 died with a misleading "Symbol not found /
# complex C++ type" error (found live composing box2d + pugixml, 2026-07-19).
# Any application composing two wrapped libraries hits this immediately.
#
# JITManager now keeps one LibraryEngine per binary behind the shared
# lock-free symbol cache (thunk names are mangled-derived, unique across
# libraries). This suite pins:
#   1. both wrappers load and BOTH dispatch Tier-2 thunks correctly,
#   2. the engine registry holds one engine per binary (no duplicates on
#      repeated init),
#   3. a missing symbol reports which engines were searched instead of the
#      old misleading single-engine message.
#
# Uses the mi_test and vi_test fixture wrappers (built by their verify
# scripts / devtests); skips cleanly if either is absent.

using Test

const MLIR_AVAILABLE = try
    using RepliBuild
    isfile(RepliBuild.MLIRNative.libJLCS)
catch
    false
end

if !MLIR_AVAILABLE
    @info "libJLCS not found — skipping multi-library JIT tests"
    exit(0)
end

const MI_WRAPPER = joinpath(@__DIR__, "mi_test", "julia", "MiTest.jl")
const VI_WRAPPER = joinpath(@__DIR__, "vi_test", "julia", "ViTest.jl")

if !(isfile(MI_WRAPPER) && isfile(VI_WRAPPER))
    @info "mi_test/vi_test wrappers not built — skipping multi-library JIT tests (run their verify scripts first)"
    exit(0)
end

include(MI_WRAPPER)
using .MiTest
include(VI_WRAPPER)
using .ViTest

@testset "multi-library JIT" begin

    @testset "one engine per binary" begin
        engines = RepliBuild.JITManager.GLOBAL_JIT.engines
        @test length(engines) == 2
        @test length(unique(e.binary_path for e in engines)) == 2
        @test all(e -> e.init_error === nothing, engines)
        @test all(e -> e.jit_engine !== nothing, engines)

        # Re-running init for an already-initialized binary is a no-op
        RepliBuild.JITManager.initialize_global_jit(engines[1].binary_path)
        @test length(RepliBuild.JITManager.GLOBAL_JIT.engines) == 2
    end

    @testset "first library dispatches Tier 2" begin
        p = MiTest.make_derived()
        @test p != C_NULL
        @test MiTest.Base1_get_a(p) == 111
        b2 = MiTest.Derived_as_Base2(p)
        @test MiTest.Base2_get_b(b2) == 1222   # override through the vtable
        MiTest.free_derived(p)
    end

    @testset "second library dispatches Tier 2" begin
        pd = ViTest.make_diamond()
        @test pd != C_NULL
        vb = ViTest.Diamond_as_VBase(pd)
        @test ViTest.VBase_tag(vb) == 1007     # override through the vbase vtable
        ViTest.free_diamond(pd)
    end

    @testset "missing symbol names the searched engines" begin
        err = try
            RepliBuild.JITManager._lookup_cached("_mlir_ciface_definitely_not_a_thunk")
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("Engines searched:", err.msg)
        @test occursin("libmi_test", err.msg)
        @test occursin("libvi_test", err.msg)
    end
end

println("✅ multi-library JIT tests passed")
