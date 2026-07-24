#!/usr/bin/env julia
# Tier-1 sliced-llvmcall dispatch (llvmcall slicing M3) — fixture-gated tests.
#
# Wraps test/slice_test/ with [wrap.tier1] enable = true and asserts:
#   1. Emission: eligible functions carry a `_SLICE_*` const + Base.llvmcall
#      on it; varargs and setjmp-closure functions stay ccall; slices land in
#      julia/slices/; TIER1_FUNCTIONS records exactly the Tier-1 surface.
#   2. Behavior through the GENERATED WRAPPER in mixed-tier mode, including
#      both coherence directions of the cJSON divergence class.
#   3. The [wrap.tier1] knobs parse.
#
# Usage: julia --project=. test/test_tier1_dispatch.jl

using Test
using TOML
using RepliBuild

const FIXTURE = joinpath(@__DIR__, "slice_test")
const TOML_PATH = joinpath(FIXTURE, "replibuild.toml")

RepliBuild.clean(TOML_PATH)
const LIB = RepliBuild.build(TOML_PATH)
const WRAPPER = RepliBuild.wrap(TOML_PATH)
const WRAPPER_TEXT = read(WRAPPER, String)

include(WRAPPER)

@testset "Tier-1 dispatch (slicing M3)" begin

@testset "Emission" begin
    # Tier-1 functions: slice const + llvmcall, no ccall in their bodies
    for fn in ("st_bump", "st_get_count", "st_apply", "st_call_op")
        @test occursin("const _SLICE_$fn", WRAPPER_TEXT)
        @test occursin("Base.llvmcall((_SLICE_$fn, \"$fn\")", WRAPPER_TEXT)
        @test isfile(joinpath(FIXTURE, "julia", "slices", "$fn.ll"))
    end

    # Varargs target: refused by the Slicer → no slice, @ccall machinery
    @test !occursin("_SLICE_st_sum", WRAPPER_TEXT)
    # setjmp closure: hazard-gated by default → plain ccall
    @test !occursin("_SLICE_st_guarded_div", WRAPPER_TEXT)
    @test occursin("ccall((:st_guarded_div, LIBRARY_PATH)", WRAPPER_TEXT)

    # The registry records exactly the Tier-1 surface
    @test isdefined(Slicetest, :TIER1_FUNCTIONS)
    for fn in ("st_bump", "st_get_count", "st_apply", "st_call_op")
        @test fn in Slicetest.TIER1_FUNCTIONS
    end
    @test !("st_guarded_div" in Slicetest.TIER1_FUNCTIONS)
    @test !("st_sum" in Slicetest.TIER1_FUNCTIONS)
end

@testset "Mixed-tier behavior through the wrapper" begin
    M = Slicetest

    # Fresh state; Tier-1 read
    @test M.st_get_count() == 0

    # Tier-1 write / Tier-3 read (via direct ccall into the .so)
    @test M.st_bump(5) == 5
    @test ccall((:st_get_count, LIB), Clong, ()) == 5

    # Tier-3 write / Tier-1 read
    @test ccall((:st_bump, LIB), Clong, (Clong,), 3) == 8
    @test M.st_get_count() == 8

    # Tier-1 dispatch through the const table
    @test M.st_apply(0, 21) == 42
    @test M.st_apply(1, 21) == -21
    @test M.st_apply(2, 9) == 81

    # Mixed: Tier-3 (wrapper ccall) writes the fn-ptr slot, Tier-1 dispatches
    @test M.st_call_op(21) == 42
    M.st_set_op(2)
    @test M.st_call_op(9) == 81
    M.st_set_op(0)
    @test M.st_call_op(9) == 18

    # Demoted functions still work through their ccall route
    @test M.st_guarded_div(84, 2) == 42
    @test M.st_guarded_div(1, 0) == -1
    @test isdefined(M, :st_sum)  # varargs base wrapper emitted (Tier 3)
end

@testset "Knob parsing" begin
    cfg = RepliBuild.ConfigurationManager.load_config(TOML_PATH)
    @test cfg.wrap.tier1.enable
    @test cfg.wrap.tier1.max_slice_kb == 64
    @test !cfg.wrap.tier1.allow_setjmp

    # Full knob surface parses from a scratch TOML
    tmp = joinpath(mktempdir(), "replibuild.toml")
    data = TOML.parsefile(TOML_PATH)
    data["wrap"]["tier1"] = Dict("enable" => true, "exclude" => ["st_bump"],
                                 "max_slice_kb" => 8, "allow_setjmp" => true)
    open(tmp, "w") do io; TOML.print(io, data); end
    cfg2 = RepliBuild.ConfigurationManager.load_config(tmp)
    @test cfg2.wrap.tier1.exclude == ["st_bump"]
    @test cfg2.wrap.tier1.max_slice_kb == 8
    @test cfg2.wrap.tier1.allow_setjmp
end

end  # testset
