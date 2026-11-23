# =============================================================================
# A function that takes the calls of its struct's field constructor, live
# (2026-11-23; found 2026-11-18)
#
#     struct pair { int a; int b; };
#     int pair(int a, int b);
#
# are two names in C (tags have a namespace of their own) and in C++ (a
# function hides a class name), and one in Julia. Julia gives the struct
# `pair(::Cint, ::Cint)` and the converting `pair(a, b)`; the function's
# `pair(::Integer, ::Integer)` is more specific than the second, so
# `pair(1, 2)` called the function and `pair(Cint(1), Cint(2))` built the
# struct. No text defined a method twice, so neither the duplicate-method
# guard nor the name screen's constructor check saw it. The screen now moves
# the function (`pair_`) and says so; the struct keeps `pair(…)`. A function
# beside a bitfield record, a union or a byte blob (built from their bytes)
# and one of another arity keep their names.
#
# This builds test/field_ctor_shadow_test as C and as C++ (one source) in a
# temporary copy, wraps it and calls it from a child. The library-free half
# is in test_type_name_collisions.jl. The C++ functions are `noexcept`, so
# both wrappers are `ccall`s.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const FIELD_CTOR_DIR = joinpath(@__DIR__, "field_ctor_shadow_test")

function field_ctor_build(lang)
    root = mktempdir()
    for f in ("s.h", "s.c", "s.cpp")
        cp(joinpath(FIELD_CTOR_DIR, f), joinpath(root, f))
    end
    toml = joinpath(root, "replibuild.toml")
    write(toml, """
    [project]
    name = "fieldctor$lang"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(root))"

    [compile]
    flags = ["-fPIC", "-O0"]
    source_files = ["$(escape_string(joinpath(root, lang == "c" ? "s.c" : "s.cpp")))"]
    include_dirs = ["$(escape_string(root))"]

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "$lang"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    redirect_stdout(devnull) do
        RepliBuild.build(toml)
    end
    said = mktemp() do path, io
        redirect_stdout(() -> RepliBuild.wrap(toml), io)
        close(io)
        read(path, String)
    end
    wrapper = joinpath(root, "julia", lang == "c" ? "Fieldctorc.jl" : "Fieldctorcpp.jl")
    return (wrapper, said)
end

function field_ctor_check(lang)
    (wrapper, said) = field_ctor_build(lang)
    @test isfile(wrapper)
    @testset "the wrap moves the function and says why" begin
        reported = [strip(l) for l in split(said, '\n') if occursin(" is bound as ", l)]
        @test reported == ["function `pair` is bound as `pair_`: `pair(…)` with one argument per field builds " *
                           "the type `pair`, a function of that name with as many parameters would take " *
                           "those calls, and the type keeps them"]
    end
    @testset "the text" begin
        text = isfile(wrapper) ? read(wrapper, String) : ""
        @test RepliBuild.Wrapper._assert_wrapper_loadable(text, lang == "c" ? "Fieldctorc" : "Fieldctorcpp") === nothing
        @test occursin(r"^function pair_\(a::"m, text)
        @test !occursin(r"^function pair\(a::"m, text)
        @test occursin(r"^function flags\(lo::"m, text) && occursin(r"^function num\(i::"m, text)
    end
    @testset "called from a child" begin
        r = run_isolated(`$(joinpath(FIELD_CTOR_DIR, "probe.jl")) $wrapper $lang`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        cases = ["field_constructor_builds_the_struct", "function_under_moved_name", "nothing_else_moved"]
        for n in cases
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), cases) == length(cases)
    end
end

# One top-level testset, so a red C half does not hide the C++ half.
@testset "A function that takes its struct's field-constructor calls, live" begin
    @testset "C" begin
        field_ctor_check("c")
    end
    if requires("A function that takes its struct's field-constructor calls, live: C++", :clangxx)
        @testset "C++" begin
            field_ctor_check("cpp")
        end
    end
end
