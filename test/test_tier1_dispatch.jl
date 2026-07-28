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

@testset "Slices are precompile dependencies" begin
    # `read` feeds the slice IR to llvmcall (which needs a statically-evaluable
    # argument) but does NOT register the file with Julia's staleness check, so
    # without an include_dependency an edited or restored slice keeps being
    # served out of a stale .ji.
    for fn in ("st_bump", "st_get_count", "st_apply", "st_call_op")
        @test occursin("include_dependency(joinpath(@__DIR__, \"slices\", \"$fn.ll\"))",
                       WRAPPER_TEXT)
    end

    # Pairing invariant: every slice the wrapper reads is also declared as a
    # dependency. Catches a future emission branch that adds one without the
    # other — the failure mode is silent, so count them rather than spot-check.
    @test count("const _SLICE_", WRAPPER_TEXT) ==
          count("include_dependency(joinpath(@__DIR__, \"slices\"", WRAPPER_TEXT)
    # One const per Tier-1 CALL SITE, one TIER1_FUNCTIONS entry per Tier-1
    # NAME — overloads (st_collide_ / st__collide below) make the former
    # strictly larger, so this direction is `>=`, not `==`.
    @test count("const _SLICE_", WRAPPER_TEXT) >= length(Slicetest.TIER1_FUNCTIONS)
end

@testset "Slice constants are keyed on the mangled symbol" begin
    # `julia_name` is not injective over `mangled`: the emitter collapses `_+`
    # and rstrips a trailing `_`, so `st_collide_` and `st__collide` both land
    # on `st_collide`. A Julia-keyed slice const makes them share one binding
    # and silently rebinds the loser to the other's slice module — and because
    # Base.llvmcall resolves the const at CODEGEN, the break surfaces on the
    # first call, not at wrap time.
    @test occursin("const _SLICE_st_collide_ = read(", WRAPPER_TEXT)
    @test occursin("const _SLICE_st__collide = read(", WRAPPER_TEXT)
    @test occursin("Base.llvmcall((_SLICE_st_collide_, \"st_collide_\")", WRAPPER_TEXT)
    @test occursin("Base.llvmcall((_SLICE_st__collide, \"st__collide\")", WRAPPER_TEXT)

    # Both are the same generic function, one Tier-1 name.
    @test length(methods(Slicetest.st_collide)) == 2
    @test "st_collide" in Slicetest.TIER1_FUNCTIONS

    # Every emitted slice constant is distinct, and each is bound to exactly
    # one symbol. Stated over the whole wrapper so any future name-mangling
    # change that re-merges two symbols fails here rather than at a call site.
    consts = [m.captures[1] for m in eachmatch(r"^const (_SLICE_\w+) = read\("m, WRAPPER_TEXT)]
    @test length(consts) == length(unique(consts))
    bindings = Dict{String,Set{String}}()
    for m in eachmatch(r"Base\.llvmcall\(\((_SLICE_\w+), \"(\w+)\"\)", WRAPPER_TEXT)
        push!(get!(bindings, m.captures[1], Set{String}()), m.captures[2])
    end
    @test all(syms -> length(syms) == 1, values(bindings))

    # The payoff: both methods actually dispatch, to their OWN symbol. A
    # mis-bind is either "Module IR does not contain specified entry function"
    # or — when the wrong module happens to declare the name — a wrong value.
    @test Slicetest.st_collide(7) == 1007
    @test Slicetest.st_collide(7, 3) == 703
end

@testset "Only slices a call site reads are written" begin
    # Acceptance is weaker than emission: the pre-pass gates on `is_c_lto_safe`,
    # a call site additionally needs `lto_shape_ok` and must survive the
    # signature dedup. Writing on acceptance shipped slices nothing could reach
    # (19 in the Hub lua wrapper, every one a Cstring return), so the files are
    # written from the final chunks instead.
    slices_dir = joinpath(FIXTURE, "julia", "slices")
    on_disk = Set(replace.(readdir(slices_dir), ".ll" => ""))
    referenced = Set(m.captures[1] for m in
                     eachmatch(r"Base\.llvmcall\(\(_SLICE_\w+, \"(\w+)\"\)", WRAPPER_TEXT))
    @test on_disk == referenced

    # st_name is the gap made concrete: a pointer return passes the pre-pass's
    # `is_c_lto_safe`, so it IS sliced and pre-flighted, but `const char *`
    # becomes Cstring and `lto_shape_ok` sends the call site to ccall.
    @test occursin("ccall((:st_name, LIBRARY_PATH)", WRAPPER_TEXT)
    @test !occursin("_SLICE_st_name", WRAPPER_TEXT)
    @test !isfile(joinpath(slices_dir, "st_name.ll"))

    # Neither the varargs nor the setjmp function may leave a slice behind —
    # those two are refused/hazard-gated before acceptance, a different route
    # to the same requirement.
    @test !isfile(joinpath(slices_dir, "st_sum.ll"))
    @test !isfile(joinpath(slices_dir, "st_guarded_div.ll"))
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
