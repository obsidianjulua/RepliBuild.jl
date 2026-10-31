#!/usr/bin/env julia
# test/test_graph_mode_live.jl — graph mode end to end on a CMake project,
# driven only through `RepliBuild.build`/`wrap`/`register`/`use`/`clean` on its
# replibuild.toml, which the test writes with a [graph] section.
#
# test/graph_mode_test/ (copied to a temp dir first; see its CMakeLists.txt):
#   cbase   C library with a counter; Core (C) links it and compiles against a
#   header a build step writes; cxxlib (C++) links Core and is the one target
#   the TOML selects; extra is not selected; gm_tool is an executable.
#
# Checks: the selected library and the shared libraries it links are built,
# nothing else; `Core` is wrapped as `RCore`; the graph module includes the
# wrappers in load order; in a child process, every library sees one copy of
# cbase's counter and cxx_total calls across both boundaries; an unchanged
# rebuild relinks and re-extracts nothing, and an edit to cxxlib relinks
# cxxlib alone; the registry vendors the graph as one unit and `use` returns
# the graph module, from the vendored copy once the project is deleted.
#
# Needs cmake, ninja, clang, clang++ and llvm-link. The planning half is
# test_graph_mode.jl (runtests).

using Test
using TOML
using RepliBuild

isdefined(@__MODULE__, :TestSupport) || include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const GML_FIXTURE = joinpath(@__DIR__, "graph_mode_test")

_gml_posix(p) = replace(String(p), '\\' => '/')

"""
The probe a child runs against the graph module the expression `load` returns
(after `prelude`, a top-level line): one PROBE line per check, and `total` as
the value cxx_total(5) returned.
"""
gml_probe_code(load::String, prelude::String) = """
    $prelude
    M = $load
    check(name, ok, detail) = println("PROBE ", name, ": ", ok ? "PASS" : "FAIL " * string(detail))
    check("module", nameof(M) === :GmFixture, nameof(M))
    check("rcore", isdefined(M, :RCore) && nameof(M.RCore) === :RCore && M.Core === Core, names(M))
    c = (M.Cbase.cbase_bump(), M.RCore.core_bump(), M.Cbase.cbase_count())
    check("one_cbase", c == (1, 2, 2), c)
    check("generated_header", M.RCore.core_version() == 7, M.RCore.core_version())
    check("core_scale", M.RCore.core_scale(5) == 36, M.RCore.core_scale(5))
    println("PROBE total: ", M.Cxxlib.cxx_total(5))
    println("PROBE_DONE")
    """

function gml_run(load::String; prelude::String = "")
    r = run_isolated(`-e $(gml_probe_code(load, prelude))`; timeout = 600)
    v = probe_verdicts(r.output)
    r.ok && haskey(v, "PROBE_DONE") || println(stderr, r.output)
    return v
end

gml_pass(v, name) = startswith(get(v, name, "no PROBE line"), "PASS")

if requires("Graph mode (live)", :cmake, :clang, :clangxx, :llvm_link)
if Sys.which("ninja") === nothing
    @testset "Graph mode (live) (skipped — not found: ninja)" begin
        @test_skip false
    end
else
@testset "Graph mode (live): a CMake project built, wrapped and registered" begin
    work = _gml_posix(mktempdir())
    proj = work * "/gm"
    cp(GML_FIXTURE, proj)
    # Written here, as test/*/*.toml is untracked: only cxxlib is selected, and
    # cbase and Core come with it because it links them.
    toml = proj * "/replibuild.toml"
    write(toml, """
        [project]
        name = "gm_fixture"
        root = "."

        [graph]
        source = "."
        targets = ["cxxlib"]
        build_type = "Release"
        """)
    home = work * "/home"
    withenv("REPLIBUILD_HOME" => home, "REPLIBUILD_JULIA_ENV" => "none") do

    # ── build ────────────────────────────────────────────────────────────────
    out = RepliBuild.build(toml)
    @test out == proj * "/julia"
    manifest = TOML.parsefile(out * "/graph.toml")
    libs = manifest["libraries"]
    @test manifest["module"] == "GmFixture"
    @test [l["name"] for l in libs] == ["cbase", "Core", "cxxlib"]   # cxxlib's closure, deps first
    @test [l["module"] for l in libs] == ["Cbase", "RCore", "Cxxlib"]
    @test [l["bucket"] for l in libs] == ["c", "c", "cpp"]
    @test libs[3]["link_dependencies"] == ["cbase", "Core"]
    @test !isdir(out * "/extra") && !isdir(out * "/gm_tool")         # not selected, not a library
    @test isfile(proj * "/build/cmake/gen/gm_version.h")             # the build step ran
    for l in libs
        @test isfile(out * "/" * l["library"]) && isfile(out * "/" * l["metadata"])
    end
    meta(n) = RepliBuild._read_json(out * "/" * only(filter(l -> l["name"] == n, libs))["metadata"])
    @test Set(f["name"] for f in meta("Core")["functions"]) ⊇ Set(["core_scale", "core_bump", "core_version"])

    # ── wrap ─────────────────────────────────────────────────────────────────
    graph_file = RepliBuild.wrap(toml)
    @test graph_file == out * "/GmFixture.jl"
    text = read(graph_file, String)
    @test [m.captures[1] for m in eachmatch(r"^include\(\"(.*)\"\)$"m, text)] ==
          ["cbase/lib/Cbase.jl", "Core/lib/RCore.jl", "cxxlib/lib/Cxxlib.jl"]
    for l in libs   # one wrapper per library, of a single library's shape
        w = read(out * "/" * l["wrapper"], String)
        @test occursin("\nmodule $(l["module"])\n", w)
        @test occursin("const LIBRARY_SHA", w) && occursin("function __init__()", w)
    end
    if Sys.iswindows()
        # Core opens cbase where it sits in the graph, relative to its wrapper.
        @test occursin("(\"../../cbase/lib/libcbase.dll\", ", read(out * "/Core/lib/RCore.jl", String))
    end
    io = IOBuffer()
    RepliBuild.GraphMode.describe(io, RepliBuild.ConfigurationManager.load_config(toml))
    @test occursin("library: Core as GmFixture.RCore [c]", String(take!(io)))

    v = gml_run("include($(repr(graph_file)))")
    for name in ("module", "rcore", "one_cbase", "generated_header", "core_scale")
        @test gml_pass(v, name)
    end
    @test get(v, "total", "") == "282"                               # 2 * ((5*7+1) + (5+100))

    # ── caches ───────────────────────────────────────────────────────────────
    files = [out * "/" * l[k] for l in libs for k in ("library", "metadata")]
    stamps = Dict(f => mtime(f) for f in files)
    RepliBuild.build(toml)
    @test all(f -> mtime(f) == stamps[f], files)                     # nothing relinked or re-extracted

    src = proj * "/cxxlib/cxxlib.cpp"
    write(src, replace(read(src, String), "#define GM_CXX_OFFSET 100" => "#define GM_CXX_OFFSET 200"))
    RepliBuild.build(toml)
    @test Set(f for f in files if mtime(f) != stamps[f]) ==
          Set([out * "/" * libs[3]["library"], out * "/" * libs[3]["metadata"]])   # cxxlib alone
    RepliBuild.wrap(toml)
    @test get(gml_run("include($(repr(graph_file)))"), "total", "") == "482"

    # ── registry ─────────────────────────────────────────────────────────────
    config = RepliBuild.ConfigurationManager.load_config(toml)
    h = RepliBuild.PackageRegistry.hash_config(config)
    @test h == RepliBuild.PackageRegistry.hash_config(config)
    entry = RepliBuild.register(toml)
    @test entry.module_name == "GmFixture" && entry.build_hash == h
    vend = home * "/builds/" * h
    @test isfile(vend * "/GmFixture.jl") && isfile(vend * "/graph.toml")
    for l in libs, k in ("library", "metadata", "wrapper")
        @test isfile(vend * "/" * l[k])
    end
    @test !any(isdir(vend * "/" * l["dir"] * "/" * d) for l in libs for d in ("ir", "build"))
    # A file no target lists still moves the hash: the generator's template.
    tmpl = proj * "/gm_version.h.in"
    orig = read(tmpl, String)
    write(tmpl, orig * "\n")
    @test RepliBuild.PackageRegistry.hash_config(config) != h
    write(tmpl, orig)
    @test RepliBuild.PackageRegistry.hash_config(config) == h

    use_code = "RepliBuild.use(\"gm_fixture\")"
    v = gml_run(use_code; prelude = "using RepliBuild")
    @test gml_pass(v, "one_cbase") && gml_pass(v, "rcore")
    @test get(v, "total", "") == "482"

    # ── clean, then the vendored graph alone ─────────────────────────────────
    RepliBuild.clean(toml)
    @test !isdir(proj * "/julia") && !isdir(proj * "/build")
    rm(proj; recursive = true)
    v = gml_run(use_code; prelude = "using RepliBuild")
    @test gml_pass(v, "one_cbase") && gml_pass(v, "generated_header")
    @test get(v, "total", "") == "482"

    end # withenv
end
end # ninja
end # requires
