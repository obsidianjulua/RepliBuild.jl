#!/usr/bin/env julia
# test/test_jit_runtime_symbols_live.jl — the JIT's callee pre-flight on a real
# library (2026-11-13). The unit half is test_jit_runtime_symbols.jl.
#
# `initialize_global_jit` pre-flights every callee its thunk module names
# through the wrapped library's OWN handle: the library's own symbols are
# ORC's to bind, a foreign one the handle resolves is bound on the engine at
# that address, one it does not resolve is dropped before the module exists.
# On Linux today no member of a listed STL container is foreign (the templates
# TU instantiates every one), so this proves two things about the mechanism and
# no new capability: the handle resolves every undefined std:: symbol the
# fixture's library calls in libstdc++ (version suffix cut; with it on, none
# resolve), and the STL fixture's Tier 2 CppString path still works through the
# engine the pre-flight configured. Runs after the stl_test fixture job built
# and wrapped the library.

using Test
using Libdl
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const _RSL_DIR = joinpath(@__DIR__, "stl_test")
const _RSL_SO = joinpath(_RSL_DIR, "julia", "libstl_test." * Libdl.dlext)
const _RSL_WRAPPER = joinpath(_RSL_DIR, "julia", "StlTest.jl")

if requires("JIT runtime-library callees, live", :libJLCS, :clangxx)
@testset "runtime-library callees resolve through the library's own handle, live" begin
    @test isfile(_RSL_SO) && isfile(_RSL_WRAPPER)      # the stl_test fixture job ran first
    C = RepliBuild.Compiler
    J = RepliBuild.JITManager

    # ELF: what the library leaves undefined is in its symbol table, and its
    # handle reaches its dependencies. A DLL has neither (the arm below).
    Sys.islinux() && @testset "every undefined std:: symbol resolves through the handle, suffix cut" begin
        m = read(`nm -gp $_RSL_SO`, String)
        d = read(`nm -gpC $_RSL_SO`, String)
        ml = C._nm_symbol_lines(m); dl = C._nm_symbol_lines(d)
        @test length(ml) == length(dl) && all(i -> ml[i][1] == dl[i][1], eachindex(ml))
        undefined_std = [ml[i][3] for i in eachindex(ml) if ml[i][1] == "U" && startswith(dl[i][3], "std::")]
        @test length(undefined_std) >= 1                   # the fixture calls into libstdc++
        resolved = C._resolved_through_library(_RSL_SO, undefined_std)
        @test resolved == Set(undefined_std)
        # The version suffix is why this looked impossible: with it on, nothing resolves.
        versioned = [first(split(l))[1] == 'U' ? String(split(l)[2]) : "" for l in split(m, '\n') if occursin("@GLIBCXX", l)]
        filter!(!isempty, versioned)
        if !isempty(versioned)
            @test isempty(C._resolved_through_library(_RSL_SO, versioned))
        end

        # The pre-flight, through the real handle: the library's own symbols are
        # never looked up, every foreign callee is bound at a non-null address.
        own = C._runtime_symbol_names(_RSL_SO)
        @test own !== nothing && !isempty(own)
        h = Libdl.dlopen(_RSL_SO, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
        try
            asked = String[]
            (bound, unresolved) = J._preflight_callees(
                n -> (push!(asked, n); Libdl.dlsym(h, n; throw_error = false)),
                union(Set(undefined_std), Set(first(collect(own), 3))), own)
            @test isempty(unresolved)
            @test Set(keys(bound)) == Set(undefined_std)
            @test all(p -> p != C_NULL, values(bound))
            @test isempty(intersect(Set(asked), own))
        finally
            Libdl.dlclose(h)
        end
    end

    # PE: a handle resolves that DLL's own exports and nothing it imports
    # (GetProcAddress), so the pre-flight binds nothing foreign there: a callee
    # in the C runtime or in libc++.dll is unresolved and is dropped.
    Sys.iswindows() && @testset "a DLL's handle resolves its own exports only" begin
        own = C._runtime_symbol_names(_RSL_SO)
        @test own !== nothing && !isempty(own)
        mine = first(sort!(collect(own)), 3)
        @test C._resolved_through_library(_RSL_SO, mine) == Set(mine)
        @test !("malloc" in own)                      # imported, never exported
        @test isempty(C._resolved_through_library(_RSL_SO, ["malloc"]))
        h = Libdl.dlopen(_RSL_SO, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
        try
            asked = String[]
            (bound, unresolved) = J._preflight_callees(
                n -> (push!(asked, n); Libdl.dlsym(h, n; throw_error = false)),
                union(Set(["malloc"]), Set(mine)), own)
            @test isempty(bound)
            @test unresolved == ["malloc"]
            @test asked == ["malloc"]                  # its own are ORC's, never asked
        finally
            Libdl.dlclose(h)
        end
    end

    @testset "the STL fixture's Tier 2 path through the pre-flighted engine" begin
        probe = """
        import RepliBuild
        include($(repr(_RSL_WRAPPER)))
        using .StlTest
        check(name, ok, detail) = println("PROBE ", name, ": ", ok ? "PASS" : "FAIL " * string(detail))
        s = StlTest.create_std_basic_string_char()
        check("string_empty", length(s) == 0, length(s))
        v = StlTest.create_std_vector_int()
        push!(v, Cint(4)); push!(v, Cint(6))
        check("vector_sum", StlTest.sum_vec(v) == 10, StlTest.sum_vec(v))
        eng = only(e for e in RepliBuild.JITManager.GLOBAL_JIT.engines if basename(e.binary_path) == $(repr(basename(_RSL_SO))))
        check("engine_up", eng.init_error === nothing && eng.jit_engine !== nothing, eng.init_error)
        println("PROBE_DONE")
        """
        r = run_isolated(`-e $probe`; timeout = 600)
        v = probe_verdicts(r.output)
        r.ok && haskey(v, "PROBE_DONE") || println(stderr, r.output)
        @test r.ok && haskey(v, "PROBE_DONE")
        for n in ("string_empty", "vector_sum", "engine_up")
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
    end
end
end # requires
