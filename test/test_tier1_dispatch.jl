#!/usr/bin/env julia
# Tier-1 sliced-llvmcall dispatch (llvmcall slicing M3) — fixture-gated tests.
#
# Wraps test/slice_test/ with [wrap.tier1] enable = true and asserts:
#   1. Emission: eligible functions carry a `_SLICE_*` path const + a
#      `@generated _TIER1_*` kernel (ccall-vs-llvmcall decided at generation
#      time); varargs and setjmp-closure functions stay ccall; slices land in
#      julia/slices/; TIER1_FUNCTIONS records exactly the Tier-1 surface.
#   2. Behavior through the GENERATED WRAPPER in mixed-tier mode, including
#      both coherence directions of the cJSON divergence class.
#   3. Output-mode demotion: every kernel demotes to ccall under
#      jl_generating_output, and a real consumer package precompiles a Tier-1
#      workload without deadlocking (the 2026-07-31 engine-lock class).
#   4. The [wrap.tier1] knobs parse.
#
# Usage: julia --project=. test/test_tier1_dispatch.jl

using Test
using TOML
using Pkg
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
    # Tier-1 functions: slice-path const + @generated kernel whose llvmcall
    # branch names the mangled symbol
    for fn in ("st_bump", "st_get_count", "st_apply", "st_call_op")
        @test occursin("const _SLICE_$fn = joinpath(@__DIR__, \"slices\", \"$fn.ll\")", WRAPPER_TEXT)
        @test occursin("@generated function _TIER1_$fn(", WRAPPER_TEXT)
        @test occursin("Base.llvmcall((\$ir, \"$fn\")", WRAPPER_TEXT)
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
    # The kernel's generator reads the slice at first call, which registers
    # nothing with Julia's staleness check — the top-level include_dependency
    # (content-tracked on 1.11+) is what invalidates a stale .ji when a slice
    # changes. It is isfile-guarded so a wrapper shipped without slices/ can
    # still precompile (its kernels demote to ccall).
    for fn in ("st_bump", "st_get_count", "st_apply", "st_call_op")
        @test occursin("isfile(_SLICE_$fn) && include_dependency(_SLICE_$fn)",
                       WRAPPER_TEXT)
    end

    # Pairing invariant: every slice-path const is also declared as a
    # dependency. Catches a future emission branch that adds one without the
    # other — the failure mode is silent, so count them rather than spot-check.
    @test count("const _SLICE_", WRAPPER_TEXT) ==
          count("include_dependency(_SLICE_", WRAPPER_TEXT)
    # ... and carries exactly one kernel per const.
    @test count("const _SLICE_", WRAPPER_TEXT) ==
          count("@generated function _TIER1_", WRAPPER_TEXT)
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
    @test occursin("const _SLICE_st_collide_ = joinpath(@__DIR__, \"slices\", \"st_collide_.ll\")", WRAPPER_TEXT)
    @test occursin("const _SLICE_st__collide = joinpath(@__DIR__, \"slices\", \"st__collide.ll\")", WRAPPER_TEXT)
    @test occursin("@generated function _TIER1_st_collide_(", WRAPPER_TEXT)
    @test occursin("@generated function _TIER1_st__collide(", WRAPPER_TEXT)

    # Both are the same generic function, one Tier-1 name.
    @test length(methods(Slicetest.st_collide)) == 2
    @test "st_collide" in Slicetest.TIER1_FUNCTIONS

    # Every emitted slice constant is distinct, and each points at exactly one
    # slice file (the kernel reads the const, so const↔symbol is const↔path).
    # Stated over the whole wrapper so any future name-mangling change that
    # re-merges two symbols fails here rather than at a call site.
    const_paths = [(m.captures[1], m.captures[2]) for m in
                   eachmatch(r"^const (_SLICE_\w+) = joinpath\(@__DIR__, \"slices\", \"([^\"]+)\.ll\"\)"m,
                             WRAPPER_TEXT)]
    @test length(const_paths) == length(unique(first.(const_paths)))
    @test length(const_paths) == length(unique(last.(const_paths)))
    # ... and each const's kernel exists under the derived name, so a const
    # cannot silently serve a different symbol's call site.
    for (c, mangled) in const_paths
        kernel = replace(c, r"^_SLICE_" => "_TIER1_")
        @test occursin("@generated function $kernel(", WRAPPER_TEXT)
        @test occursin("Base.llvmcall((\$ir, \"$mangled\")", WRAPPER_TEXT)
    end

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
    referenced = Set(m.captures[2] for m in
                     eachmatch(r"^const (_SLICE_\w+) = joinpath\(@__DIR__, \"slices\", \"([^\"]+)\.ll\"\)"m,
                               WRAPPER_TEXT))
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

@testset "Output-mode demotion (precompile safety)" begin
    # Every kernel must carry the generation-time demotion check: emitting a
    # sliced llvmcall inside a precompile worker deadlocks the JIT engine lock
    # when a declare binds a dlopened library's symbol (2026-07-31), and an
    # UNTAKEN top-level branch reaches emission through inference alone — so
    # this cannot be a runtime branch, and no kernel may skip it.
    @test count("@generated function _TIER1_", WRAPPER_TEXT) ==
          count("ccall(:jl_generating_output, Cint, ()) == 1 || !isfile(_SLICE_",
                WRAPPER_TEXT)
    # The demoted body targets the kernel's own symbol
    @test occursin("return :(ccall((:st_bump, LIBRARY_PATH)", WRAPPER_TEXT)

    # Live regression gate: a consumer package that EXECUTES a Tier-1 call in
    # its precompile workload must precompile to completion. Run it under a
    # deadline — the failure mode being guarded is a silent permanent hang, so
    # a hang must become a test failure, not a stuck suite.
    pkgdir = mktempdir()
    depot = mktempdir()
    replibuild_dir = dirname(@__DIR__)   # the wrapper does `import RepliBuild`
    mkpath(joinpath(pkgdir, "Probe", "src"))
    write(joinpath(pkgdir, "Probe", "Project.toml"), """
        name = "Probe"
        uuid = "d1f00d5e-7e5b-4f5e-9c3a-0b8f4a9c21aa"
        version = "0.1.0"

        [deps]
        Libdl = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
        RepliBuild = "4450f29b-7b71-45c6-8742-e7520a479938"
        """)
    write(joinpath(pkgdir, "Probe", "src", "Probe.jl"), """
        module Probe
        include($(repr(WRAPPER)))
        using .Slicetest
        if ccall(:jl_generating_output, Cint, ()) == 1
            Slicetest.st_bump(1) == 1 || error("Tier-1 workload returned wrong value")
        end
        end
        """)
    # The wrapper's `import RepliBuild` cannot resolve through a manifest-less
    # project on the load path (a project-as-package env doesn't serve its
    # project to OTHER envs' deps), so give Probe a real Manifest the way a
    # consumer would: Pkg.develop against this checkout.
    let old = Base.active_project()
        withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
            Pkg.activate(joinpath(pkgdir, "Probe"); io=devnull)
            Pkg.develop(path=replibuild_dir; io=devnull)
            Pkg.activate(old; io=devnull)
        end
    end
    # Temp depot first (all writes land there), real depot second so
    # RepliBuild's existing pkgimage is reused instead of rebuilt.
    cmd = setenv(`$(Base.julia_cmd()) --startup-file=no -e "using Probe"`,
                 "JULIA_LOAD_PATH" => "$(joinpath(pkgdir, "Probe")):@stdlib",
                 "JULIA_DEPOT_PATH" => "$depot:$(first(DEPOT_PATH))",
                 "HOME" => homedir(), "PATH" => ENV["PATH"])
    proc = run(pipeline(cmd, stdout=devnull, stderr=devnull), wait=false)
    finished = timedwait(() -> process_exited(proc), 120.0) == :ok
    finished || kill(proc, Base.SIGKILL)
    @test finished           # a hang here is the engine-lock deadlock class
    @test finished && success(proc)
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
