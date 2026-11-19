# =============================================================================
# A function or global named like a struct, live (2026-11-18)
#
# test/type_name_collision_test is curl's shape (`const struct Curl_ssl
# *Curl_ssl;`) and its function twins:
#
#     extern const struct conf *conf;      /* a global named like its tag */
#     struct pool *pool(void);             /* a function taking nothing */
#     struct token token(void);            /* ... returning its struct by value */
#
# Each record's zero-initializer `T()` and the accessor or function of the
# same name were one method. The text defined it twice, so the later replaced
# the earlier on include and a precompiled package refused to load, and
# `_assert_no_duplicate_methods` refused the wrap. The name screen now moves
# the function (`conf_`, `pool_`, `token_`) and says so; `T()` stays the
# zeroed record and the global's pointer accessor stays `conf_ptr`.
#
# This builds the fixture with the C bucket, wraps it, and loads and calls the
# wrapper in a child. test_type_name_collisions.jl is the library-free half.
# Built in a temporary copy, so nothing is written under test/.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const TYPE_NAME_DIR = joinpath(@__DIR__, "type_name_collision_test")

@testset "A function or global named like a struct, live" begin
    root = mktempdir()
    for f in ("g.h", "g.c")
        cp(joinpath(TYPE_NAME_DIR, f), joinpath(root, f))
    end
    toml = joinpath(root, "replibuild.toml")
    write(toml, """
    [project]
    name = "typenamecollision"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(root))"

    [compile]
    flags = ["-fPIC"]
    source_files = ["$(escape_string(joinpath(root, "g.c")))"]
    include_dirs = ["$(escape_string(root))"]

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "c"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    redirect_stdout(devnull) do
        RepliBuild.build(toml)
    end
    path = tempname()
    open(path, "w") do io
        redirect_stdout(() -> RepliBuild.wrap(toml), io)
    end
    said = read(path, String)
    rm(path; force = true)
    wrapper = joinpath(root, "julia", "Typenamecollision.jl")
    @test isfile(wrapper)

    @testset "the wrap moves each function and says why" begin
        reported = [strip(l) for l in split(said, '\n') if occursin(" is bound as ", l)]
        for (from, to) in (("conf", "conf_"), ("pool", "pool_"), ("token", "token_"))
            @test "function `$from` is bound as `$to`: the type `$from` defines a method of the same " *
                  "signature itself (a constructor such as its zero-initializer `$from()`), and keeps it" in reported
        end
        @test length(reported) == 3
    end

    @testset "the text defines each method once" begin
        text = read(wrapper, String)
        @test RepliBuild.Wrapper._assert_wrapper_loadable(text, "Typenamecollision") === nothing
        @test occursin(r"^function conf_\(\)::Ptr\{conf\}$"m, text)
        @test occursin(r"^function conf_ptr\(\)::Ptr\{Ptr\{conf\}\}$"m, text)
        @test occursin(r"^function conf\(\)$"m, text)               # the zero-initializer
    end

    @testset "loaded and called from a child" begin
        r = run_isolated(`$(joinpath(TYPE_NAME_DIR, "probe.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["type_constructors_are_zeroed", "global_value_under_moved_name",
                 "global_pointer_keeps_its_name", "functions_under_moved_names", "nothing_else_moved"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
