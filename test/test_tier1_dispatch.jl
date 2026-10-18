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
#   5. A Slicer throw never fails the wrap: on one target it demotes only that
#      function to ccall and caches nothing for it; on the whole module it turns
#      Tier 1 off for the wrap. Both warn.
#
# Usage: julia --project=. test/test_tier1_dispatch.jl

using Test
using TOML
using Pkg
using SHA
using Libdl
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
    pathsep = Sys.iswindows() ? ";" : ":"
    cmd = setenv(`$(Base.julia_cmd()) --startup-file=no -e "using Probe"`,
                 "JULIA_LOAD_PATH" => "$(joinpath(pkgdir, "Probe"))$(pathsep)@stdlib",
                 "JULIA_DEPOT_PATH" => "$depot$(pathsep)$(first(DEPOT_PATH))",
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

@testset "Portability guards: wrapper on a foreign build" begin
    # A generated wrapper is portable SOURCE, so the library sitting beside it
    # at runtime need not be the one it was generated from — someone pointing it
    # at a JLL, a distro package, or any independent rebuild. Two things then go
    # wrong, and each has its own guard:
    #
    #   1. Sliced Tier 1 binds `__rb_*` promoted statics that ONLY exist in a
    #      library RepliBuild built. Against a foreign build those declares are
    #      unresolvable, and an unresolved declare does not raise — ORC prints
    #      "Symbols not found" and BLOCKS FOREVER on the first call. The
    #      wrap-time pre-flight cannot help: it ran against a different file.
    #   2. Struct offsets, blob sizes and enum values are a snapshot of one
    #      compilation, and reading them wrong is silent.
    #
    # The stand-in for a foreign build is a plain clang build of the same
    # sources with no promotion pass — which is exactly what every non-RepliBuild
    # build of this library looks like.
    foreign_dir = mktempdir()
    foreign_so = joinpath(foreign_dir, basename(LIB))
    srcs = readdir(joinpath(FIXTURE, "src"), join=true)
    filter!(f -> endswith(f, ".c"), srcs)
    run(pipeline(`clang -O2 -fPIC -g -shared -I$(joinpath(FIXTURE, "include")) $srcs -o $foreign_so`,
                 stdout=devnull, stderr=devnull))

    @test isfile(foreign_so)
    # The premise: no promoted symbols here, unlike RepliBuild's own build.
    # `nm -D` is ELF dynsym; PE has none, so `-g` on Windows.
    nm_cmd = Sys.iswindows() ? `nm -g --defined-only $foreign_so` :
                               `nm -D --defined-only $foreign_so`
    @test !occursin("__rb_", read(nm_cmd, String))

    # Some slice really does bind a promoted static — otherwise this proves nothing.
    promoted_users = [f for (f, syms) in Slicetest.TIER1_DECLARES
                      if any(s -> startswith(s, "__rb_"), syms)]
    @test !isempty(promoted_users)

    # Guard 1: those declares are correctly seen as unresolvable against a
    # process where the foreign library is the only slicetest loaded... but this
    # process already has the REAL library open, so check the mechanism directly
    # on a symbol that cannot exist anywhere.
    @test !Slicetest._slice_symbols_resolve(["__rb_slicetest_definitely_not_a_symbol"])
    @test Slicetest._slice_symbols_resolve(String[])          # vacuous case is true
    @test Slicetest._slice_symbols_resolve(["st_bump"])        # a real exported symbol

    # Guard 2: build identity is recorded and actually distinguishes builds.
    # Identity is the library's SHA-256, not its GNU build ID — a build ID is
    # only present if the linker was asked for one, and the C bucket's
    # Clang_unified_jll does not emit one, which left this guard inert.
    @test !isempty(Slicetest.LIBRARY_SHA)
    @test Slicetest.BUILD_TARGET == string(Sys.MACHINE)
    @test bytes2hex(open(SHA.sha256, LIB)) == Slicetest.LIBRARY_SHA
    foreign_sha = bytes2hex(open(SHA.sha256, foreign_so))
    @test foreign_sha != Slicetest.LIBRARY_SHA     # independent builds differ

    # And the generator stamp is recorded for the other drift direction (the
    # wrapper going stale against RepliBuild's codegen while the .so matches).
    @test Slicetest.BUILD_GENERATOR == string(pkgversion(RepliBuild))

    # The end-to-end behaviour (wrapper staged next to the foreign .so warns,
    # demotes every promoted-static slice, and still computes correctly) is
    # exercised out-of-process, since it needs a fresh module load.
    driver = joinpath(foreign_dir, "drive.jl")
    stage = joinpath(foreign_dir, "stage")
    mkpath(stage)
    cp(joinpath(FIXTURE, "julia", "slices"), joinpath(stage, "slices"))
    cp(foreign_so, joinpath(stage, basename(LIB)), force=true)
    cp(WRAPPER, joinpath(stage, "Slicetest.jl"))
    write(driver, """
        include(joinpath(raw"$stage", "Slicetest.jl"))
        using .Slicetest
        # Would hang here before the runtime symbol check.
        @assert Slicetest.st_bump(5) == 5
        @assert Slicetest.st_get_count() == 5
        @assert Slicetest.st_apply(1, 21) == -21
        print("FOREIGN_OK")
    """)
    out = IOBuffer()
    ok = success(pipeline(`$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) $driver`,
                          stdout=out, stderr=devnull))
    @test ok
    @test occursin("FOREIGN_OK", String(take!(out)))
end

@testset "A Slicer throw costs Tier 1, never the wrap" begin
    # Last, because each wrap here rewrites julia/ and the slice cache; the end
    # of the testset puts both back. Wraps and `which` go through
    # `invokelatest`: a Slicer method is added and deleted below, and code
    # already running here keeps dispatching in the world it started in.
    Slicer = RepliBuild.Slicer
    cfg = RepliBuild.ConfigurationManager.load_config(TOML_PATH)
    slice_cache = joinpath(RepliBuild.ConfigurationManager.get_cache_path(cfg), "slices")
    julia_dir = RepliBuild.ConfigurationManager.get_output_path(cfg)
    victim = "st_get_count"                       # Tier 1 in every testset above
    kept = ("st_bump", "st_apply", "st_call_op")

    # Wrap and return the wrapper's path with what the wrap printed: the
    # `tier1:` summary goes to stdout, not to the logger. Echoed afterwards, so
    # a failure report inside the redirect is not lost.
    function wrap_logged(pattern)
        out_path, out_io = mktemp()
        w = try
            redirect_stdout(out_io) do
                @test_logs (:warn, pattern) match_mode=:any Base.invokelatest(RepliBuild.wrap, TOML_PATH)
            end
        finally
            close(out_io)
        end
        printed = read(out_path, String)
        print(printed)
        return w, printed
    end

    @testset "one target: demoted to ccall, the rest keep Tier 1" begin
        # A cache hit never reaches `_extract!`, and the first wrap cached every
        # verdict.
        rm(slice_cache; recursive=true, force=true)

        # Fault injection. A method more specific than `_extract!(mod, ::String)`
        # wins dispatch for every clone (an `LLVM.Module`): it throws for the
        # victim and hands every other target to the real method.
        @eval Slicer function _extract!(mod::LLVM.Module, target_name::String)
            target_name == $victim && throw(ArgumentError("injected Slicer fault"))
            return invoke(_extract!, Tuple{Any,String}, mod, target_name)
        end
        injected = Base.invokelatest(which, Slicer._extract!, Tuple{Slicer.LLVM.Module,String})
        @test injected.sig == Tuple{typeof(Slicer._extract!),Slicer.LLVM.Module,String}
        # The throw site is the `_extract!` call in an anonymous closure, so the
        # reason gives only its line; the type is named once.
        warned = Regex("(?s)Tier 1: 1 function\\(s\\) demoted to ccall because the " *
                       "Slicer\\s+threw.*$victim → slicing threw ArgumentError at " *
                       "Slicer\\.jl:\\d+: injected Slicer fault")
        w, printed = try
            wrap_logged(warned)
        finally
            Base.delete_method(injected)
        end
        @test occursin(r"tier1: \d+ slices accepted \(\d+ refused, 1 errored, ", printed)

        text = read(w, String)
        # The victim emits the ccall it would with Tier 1 off ...
        @test !occursin("_SLICE_$victim", text)
        @test occursin("ccall((:$victim, LIBRARY_PATH)", text)
        @test occursin(":$victim => :tier3,", text)
        @test !isfile(joinpath(julia_dir, "slices", "$victim.ll"))
        # ... and nothing else lost its slice.
        for fn in kept
            @test occursin("@generated function _TIER1_$fn(", text)
            @test occursin(":$fn => :tier1,", text)
            @test isfile(joinpath(julia_dir, "slices", "$fn.ll"))
        end

        # Every verdict was cached; the error was not.
        key_dir = only(readdir(slice_cache; join=true))
        @test !isfile(joinpath(key_dir, "$victim.json"))
        for fn in kept
            @test isfile(joinpath(key_dir, "$fn.json"))
        end

        # The wrapper loads and runs, in a fresh process because this one
        # already holds a `Slicetest`. The victim computes through ccall, a kept
        # function really splices llvmcall, and both address one counter.
        driver = joinpath(mktempdir(), "errored_slice_probe.jl")
        write(driver, """
            include($(repr(w)))
            using .Slicetest
            println("TIER ", Slicetest.dispatch_tier(:$victim), " ", Slicetest.dispatch_tier(:st_bump))
            println("TIER1 ", join(sort!(collect(Slicetest.TIER1_FUNCTIONS)), ","))
            println("VALUES ", Slicetest.st_bump(5), " ", Slicetest.st_get_count())
            """)
        out, err = IOBuffer(), IOBuffer()
        ok = success(pipeline(`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) $driver`;
                              stdout=out, stderr=err))
        probe = String(take!(out))
        ok || @info "errored-slice probe failed" probe err_text = String(take!(err))
        @test ok
        @test occursin("TIER tier3 tier1", probe)
        tier1_line = match(r"^TIER1 (.*)$"m, probe)
        tier1_names = tier1_line === nothing ? String[] : split(tier1_line.captures[1], ",")
        @test !(victim in tier1_names)
        @test all(fn -> fn in tier1_names, kept)
        @test occursin("VALUES 5 5", probe)
    end

    @testset "whole module: Tier 1 off for the wrap, every function on ccall" begin
        abi_ll = joinpath(RepliBuild.ConfigurationManager.get_build_path(cfg), "slicetest_abi.ll")
        saved = read(abi_ll)
        before = Set(readdir(slice_cache))
        new_keys = Dict{String,Vector{String}}()   # cache key => entries under it
        w, printed = try
            write(abi_ll, "this is not LLVM IR\n")
            wrap_logged(r"Tier 1: the Slicer failed on the promoted module")
        finally
            # Restored even when the wrap throws, so the end of the testset
            # still finds the fixture's module and a single cache key.
            write(abi_ll, saved)
            for k in setdiff(Set(readdir(slice_cache)), before)
                new_keys[k] = readdir(joinpath(slice_cache, k))
                rm(joinpath(slice_cache, k); recursive=true)
            end
        end

        text = read(w, String)
        @test !occursin("_SLICE_", text)
        @test !occursin("@generated function _TIER1_", text)
        @test !occursin("TIER1_FUNCTIONS", text)
        for fn in (victim, kept...)
            @test occursin("ccall((:$fn, LIBRARY_PATH)", text)
        end
        @test isempty(readdir(joinpath(julia_dir, "slices")))
        @test !occursin("tier1:", printed)     # the pre-pass stopped before its summary

        # The unparseable module got a key, and nothing was cached under it.
        @test length(new_keys) == 1
        @test all(isempty, values(new_keys))
    end

    # Back to the real method, and to the fixture the testsets above left. The
    # victim slices again because its error was never cached, and now its
    # verdict is.
    @test Base.invokelatest(which, Slicer._extract!, Tuple{Slicer.LLVM.Module,String}).sig ==
          Tuple{typeof(Slicer._extract!),Any,String}
    restored = read(Base.invokelatest(RepliBuild.wrap, TOML_PATH), String)
    @test occursin("@generated function _TIER1_$victim(", restored)
    @test isfile(joinpath(only(readdir(slice_cache; join=true)), "$victim.json"))
end

end  # testset
