# =============================================================================
# A fragment header in the forced unit, live (2026-11-16)
#
# Jolt's Compute/CPU/WrapShaderBegin.h opens namespaces and a class that only
# its partner WrapShaderEnd.h closes. A class declared through the pair has
# the opening header as its declaration file, so the forced unit included it
# on its own; every header after it then sat inside its scope, the errors
# landed on the lines of rows that were fine, and those rows were skipped as
# "does not compile" (1,944 of Jolt's constructors) before the per-header
# check ever ran. The include block is now compiled by itself first.
#
# This builds test/fragment_header_test: two ordinary classes with inline
# constructors and a member, and one class declared through a begin/end pair.
# Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const FRAGMENT_HEADER_DIR = joinpath(@__DIR__, "fragment_header_test")

if requires("a fragment header in the forced unit, live", :libJLCS, :clangxx)
@testset "A fragment header in the forced unit, live" begin
    toml_path = joinpath(FRAGMENT_HEADER_DIR, "replibuild.toml")
    src = joinpath(FRAGMENT_HEADER_DIR, "src", "frag.cpp")
    inc = joinpath(FRAGMENT_HEADER_DIR, "include")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "fraghdr"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(FRAGMENT_HEADER_DIR))"

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
    redirect_stdout(devnull) do
        RepliBuild.build(toml_path)
    end
    md = RepliBuild._read_json(joinpath(FRAGMENT_HEADER_DIR, "julia", "compilation_metadata.json"))
    ctors = md["inline_constructors"]
    meths = md["inline_methods"]

    @testset "the fixture has the shape: a class declared in the opening half" begin
        files = [basename(String(f)) for f in md["struct_definitions"]["Wrapper"]["decl_files"]]
        @test files == ["wrap_begin.h"]
    end

    @testset "the rows that are fine are forced" begin
        @test sort([String(f["class"]) for f in ctors["forced"] if f["emitted"] == true]) == ["Other", "Plain"]
        # On the old order `Plain::twice` was skipped as "does not compile:
        # expected '}'": the fragment's unclosed scope, blamed on its row.
        @test [(String(f["class"]), String(f["name"])) for f in meths["forced"] if f["emitted"] == true] == [("Plain", "twice")]
        mangled = Set(String(f["mangled"]) for f in md["functions"])
        @test "_ZN2fh5PlainC2Ei" in mangled && "_ZNK2fh5Plain5twiceEv" in mangled
    end

    @testset "the fragment's own rows are skipped for their header, by name" begin
        for plan in (ctors, meths)
            sk = [(String(s["class"]), String(s["reason"])) for s in plan["skipped"]]
            @test length(sk) == 1 && sk[1][1] == "Wrapper"
            @test startswith(sk[1][2], "its header does not compile on its own (wrap_begin.h: ")
            # Nothing is blamed on a row.
            @test !any(r -> startswith(r[2], "does not compile"), sk)
        end
        unit = read(joinpath(FRAGMENT_HEADER_DIR, ".replibuild_cache", "replibuild_ctors.cpp"), String)
        @test !occursin("wrap_begin.h", unit) && occursin("frag.h", unit)
    end

    @testset "the wrapper loads and the forced definitions run" begin
        RepliBuild.wrap(toml_path)
        wrapper = joinpath(FRAGMENT_HEADER_DIR, "julia", "Fraghdr.jl")
        @test isfile(wrapper)
        script = joinpath(mktempdir(), "probe_fragment.jl")
        write(script, """
        const M = include(ARGS[1])
        mem = Libc.malloc(M.struct_size(:Plain))
        M.fh_Plain_Plain(mem, 21)
        ok = M.fh_plain_v(mem) == 21 && M.fh_Plain_twice(mem) == 42
        Libc.free(mem)
        println("PROBE forced_rows_run: ", ok ? "PASS" : "FAIL")
        println("PROBE_DONE")
        """)
        r = run_isolated(`$script $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        @test get(v, "forced_rows_run", "no PROBE line") == "PASS"
    end
end
end # requires
