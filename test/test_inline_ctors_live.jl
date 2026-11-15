# =============================================================================
# Forced inline constructors, live (2026-11-15)
#
# test_inline_constructors.jl (unit) checks the parser's facts, the plan, the
# unit's text and the export rewrite. This builds test/inline_ctor_test (a
# header whose constructors nothing in the .cpp uses, -fstandalone-debug as
# Bullet is built) and checks the build's two passes: the metadata names what
# the forced unit defined and what it skipped and why, the wrapper binds the
# forced constructors like any other, and a child constructs through them and
# reads a field back. Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const INLINE_CTOR_DIR = joinpath(@__DIR__, "inline_ctor_test")

if requires("forced inline constructors, live", :libJLCS, :clangxx)
@testset "Forced inline constructors, live" begin
    toml_path = joinpath(INLINE_CTOR_DIR, "replibuild.toml")
    src = joinpath(INLINE_CTOR_DIR, "src", "inline_ctor.cpp")
    inc = joinpath(INLINE_CTOR_DIR, "include")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "inlinector"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(INLINE_CTOR_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC", "-O2", "-fstandalone-debug"]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(inc))"]

    [link]
    enable_lto = false
    optimization_level = "2"

    [binary]
    type = "shared"

    [wrap]
    language = "cpp"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)

    build_log = mktemp() do path, io
        redirect_stdout(io) do
            RepliBuild.build(toml_path)
        end
        close(io)
        read(path, String)
    end
    RepliBuild.wrap(toml_path)
    wrapper = joinpath(INLINE_CTOR_DIR, "julia", "Inlinector.jl")
    @test isfile(wrapper)
    code = read(wrapper, String)

    @testset "the build ran a second pass for the unit" begin
        @test occursin(r"ctors: 5 inline constructor\(s\) of 4 class\(es\) declared in headers and defined nowhere", build_log)
        @test count("link: libinlinector.so", build_log) == 2
        unit = joinpath(INLINE_CTOR_DIR, ".replibuild_cache", "replibuild_ctors.cpp")
        @test isfile(unit)
        text = read(unit, String)
        @test occursin("#include \"$(joinpath(inc, "inline_ctor.h"))\"", text)
        @test occursin("template void replibuild_force::ctor<ic::Plain, int, int>(void*, int, int);", text)
        @test occursin("template void replibuild_force::ctor<ic::Abs, int>(void*, int);", text)
        @test !occursin("Defined", text) && !occursin("Priv", text) && !occursin("Del", text)
        # The library now defines them; the unit's helpers are not API.
        md = RepliBuild._read_json(joinpath(INLINE_CTOR_DIR, "julia", "compilation_metadata.json"))
        mangled = Set(String(f["mangled"]) for f in md["functions"])
        for s in ("_ZN2ic5PlainC2Ei", "_ZN2ic5PlainC2Eii", "_ZN2ic4DfltC2Eii", "_ZN2ic4ExplC2Ef")
            @test (s, s in mangled) == (s, true)
        end
        @test !any(f -> occursin("replibuild_force", String(get(f, "demangled", ""))), md["functions"])
        @test !any(s -> occursin("AbsC", s) || occursin("PrivC", s), mangled)
    end

    @testset "the metadata names what was forced and what was skipped, with reasons" begin
        md = RepliBuild._read_json(joinpath(INLINE_CTOR_DIR, "julia", "compilation_metadata.json"))
        plan = md["inline_constructors"]
        emitted = sort([(f["class"], f["symbol"]) for f in plan["forced"] if f["emitted"] == true])
        @test emitted == [("Dflt", "_ZN2ic4DfltC4Eii"), ("Expl", "_ZN2ic4ExplC4Ef"),
                          ("Plain", "_ZN2ic5PlainC4Ei"), ("Plain", "_ZN2ic5PlainC4Eii")]
        declined = [(f["class"], f["symbol"]) for f in plan["forced"] if f["emitted"] != true]
        @test declined == [("Abs", "_ZN2ic3AbsC4Ei")]          # abstract: the unit's guard
        @test sort([(s["class"], s["reason"]) for s in plan["skipped"]]) == [("Del", "deleted"), ("Priv", "private")]
        @test !any(f -> f["class"] == "Defined", plan["forced"]) && !any(s -> s["class"] == "Defined", plan["skipped"])
        # The parser's rows behind the plan.
        rows = md["record_abi"]["Priv"]["constructors"]
        @test [(r["symbol"], r["accessibility"]) for r in rows] == [("_ZN2ic4PrivC4Ei", "private")]
    end

    @testset "the wrapper binds a forced constructor like any other" begin
        @test occursin(r"\nfunction ic_Plain_Plain\(this::Any, v::Integer\)\n", code)
        @test occursin(r"\nfunction ic_Plain_Plain\(this::Any, a::Integer, b::Integer\)\n", code)
        @test occursin(r"\nfunction ic_Dflt_Dflt\(this::Any, a::Integer, b::Integer\)\n", code)
        @test occursin(r"\nfunction ic_Expl_Expl\(this::Any, f::", code)
        @test occursin(r"\nfunction ic_Defined_Defined\(this::Any, v::Integer\)\n", code)
        @test !occursin("function ic_Abs_Abs(", code)
        @test !occursin("function ic_Priv_Priv(", code)
        @test occursin("\nstruct Plain\n    v::Cint\nend\n", code)
    end

    @testset "constructed through the wrapper, from a child" begin
        r = run_isolated(`$(joinpath(INLINE_CTOR_DIR, "probe_inline_ctor.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["plain_forced_constructor", "default_argument_spelled", "explicit_constructor",
                 "defined_constructor_untouched", "declined_and_skipped_unbound"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
end # requires
