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

# The same project, served from a bare git repository and named by
# `[graph] dependency`: the resolver clones and pins it under the project's
# deps cache, the graph is configured from the clone, and everything the
# local-source case checks holds. Then the dependency's clean rules: `deps =
# false` keeps the clone and the next build reuses it; a moved pin is another
# hash and another build; `clean` removes the clone.
if Sys.which("git") === nothing
    @testset "Graph mode (live): the CMake project from a pinned git dependency (skipped — not found: git)" begin
        @test_skip false
    end
else
@testset "Graph mode (live): the CMake project from a pinned git dependency" begin
    work = _gml_posix(mktempdir())
    gitrun(repo, args...) = run(`git -C $repo -c user.email=t@example.com -c user.name=tester -c commit.gpgsign=false -c tag.gpgsign=false -c core.autocrlf=false $(collect(args))`)
    gitout(repo, args...) = String(strip(read(`git -C $repo $(collect(args))`, String)))
    file_url(p) = "file://" * (Sys.iswindows() ? "/" * replace(abspath(p), '\\' => '/') : abspath(p))

    # upstream: the fixture committed; upstream.git: the bare repository the
    # project clones, like a forge would serve it.
    up = work * "/upstream"
    cp(GML_FIXTURE, up)
    run(`git -C $up init -q`)
    gitrun(up, "add", "-A"); gitrun(up, "commit", "-q", "-m", "fixture")
    bare = work * "/upstream.git"
    run(`git init -q --bare $bare`)
    gitrun(bare, "symbolic-ref", "HEAD", "refs/heads/main")      # a clone checks out HEAD: the branch pushed below
    gitrun(up, "remote", "add", "origin", bare)
    gitrun(up, "tag", "v1")
    gitrun(up, "push", "-q", "origin", "HEAD:refs/heads/main", "v1")
    sha1 = gitout(up, "rev-parse", "HEAD")
    @test occursin(r"^[0-9a-f]{40}$", sha1)

    proj = work * "/gmdep"
    mkpath(proj)
    toml = proj * "/replibuild.toml"
    # A tag and its commit, the Hub's convention: the tag is what is checked
    # out, the commit what the checkout must be.
    write_toml(tag, sha) = write(toml, """
        [project]
        name = "gm_fixture"
        root = "."

        [graph]
        dependency = "upstream"
        targets = ["cxxlib"]
        build_type = "Release"

        [dependencies.upstream]
        type = "git"
        url = "$(file_url(bare))"
        tag = "$tag"
        commit = "$sha"
        """)
    write_toml("v1", sha1)
    clone = proj * "/.replibuild_cache/deps/upstream"
    home = work * "/home"
    withenv("REPLIBUILD_HOME" => home, "REPLIBUILD_JULIA_ENV" => "none") do

    # ── build: resolve, then configure the clone ─────────────────────────────
    out = RepliBuild.build(toml)
    @test out == proj * "/julia"
    @test isdir(clone * "/.git") && isfile(clone * "/CMakeLists.txt")
    @test lowercase(gitout(clone, "rev-parse", "HEAD")) == sha1
    @test isfile(proj * "/.replibuild_cache/deps/upstream.resolved")
    manifest = TOML.parsefile(out * "/graph.toml")
    libs = manifest["libraries"]
    @test manifest["dependency"] == "upstream" && manifest["source"] == clone
    @test length(libs) == 3
    @test [l["name"] for l in libs] == ["cbase", "Core", "cxxlib"]
    @test [l["module"] for l in libs] == ["Cbase", "RCore", "Cxxlib"]
    @test !isdir(out * "/extra") && !isdir(out * "/gm_tool")
    @test isfile(proj * "/build/cmake/gen/gm_version.h")             # the build step ran, from the clone
    for l in libs
        @test isfile(out * "/" * l["library"]) && isfile(out * "/" * l["metadata"])
    end
    io = IOBuffer()
    RepliBuild.GraphMode.describe(io, RepliBuild.ConfigurationManager.load_config(toml))
    s = String(take!(io))
    @test occursin("CMake project $clone", s)
    @test occursin("from:    [dependencies.upstream] (git) $(file_url(bare)) @ $(sha1[1:12])", s)

    # ── wrap: the same graph the local source gives ──────────────────────────
    graph_file = RepliBuild.wrap(toml)
    @test graph_file == out * "/GmFixture.jl"
    @test [m.captures[1] for m in eachmatch(r"^include\(\"(.*)\"\)$"m, read(graph_file, String))] ==
          ["cbase/lib/Cbase.jl", "Core/lib/RCore.jl", "cxxlib/lib/Cxxlib.jl"]
    v = gml_run("include($(repr(graph_file)))")
    checked = 0
    for name in ("module", "rcore", "one_cbase", "generated_header", "core_scale")
        @test gml_pass(v, name)
        checked += 1
    end
    @test checked == 5
    @test get(v, "total", "") == "282"

    # ── registry: the hash covers the pin and the clone ──────────────────────
    # `use` hashes the loaded config with its dependencies resolved.
    hash_now() = RepliBuild.PackageRegistry.hash_config(
        RepliBuild.DependencyResolver.resolve_dependencies(RepliBuild.ConfigurationManager.load_config(toml)))
    h1 = hash_now()
    @test h1 == hash_now()
    entry = RepliBuild.register(toml)
    @test entry.build_hash == h1
    vend = home * "/builds/" * h1
    @test isfile(vend * "/GmFixture.jl") && isfile(vend * "/graph.toml")
    @test all(isfile(vend * "/" * l[k]) for l in libs for k in ("library", "metadata", "wrapper"))
    @test !occursin(clone, read(vend * "/cxxlib/lib/Cxxlib.jl", String)) ||
          occursin("LIBRARY_PATH", read(vend * "/cxxlib/lib/Cxxlib.jl", String))

    # ── clean(deps = false) keeps the clone, and the next build reuses it ────
    # The sentinel sits in the clone's .git, which the hash never reads: a
    # re-clone would lose it, a re-checkout keeps it.
    sentinel = clone * "/.git/gml_sentinel"
    write(sentinel, "")
    RepliBuild.clean(toml; deps = false)
    @test !isdir(proj * "/julia") && !isdir(proj * "/build")
    @test isfile(sentinel) && isfile(proj * "/.replibuild_cache/deps/upstream.resolved")
    @test !isdir(proj * "/.replibuild_cache/ir")
    @test hash_now() == h1                                            # the clone is the same source
    RepliBuild.build(toml)
    @test isfile(sentinel)                                            # verified, not cloned again
    @test isfile(out * "/" * libs[3]["library"])

    # ── a moved pin: another commit upstream, another hash, another build ────
    src = up * "/cxxlib/cxxlib.cpp"
    write(src, replace(read(src, String), "#define GM_CXX_OFFSET 100" => "#define GM_CXX_OFFSET 200"))
    gitrun(up, "commit", "-q", "-am", "offset 200")
    gitrun(up, "tag", "v2")
    gitrun(up, "push", "-q", "origin", "HEAD:refs/heads/main", "v2")
    sha2 = gitout(up, "rev-parse", "HEAD")
    @test sha2 != sha1
    write_toml("v2", sha2)
    @test lowercase(gitout(clone, "rev-parse", "HEAD")) == sha1       # the clone has not moved yet
    h2 = hash_now()                                                   # resolving re-checks out to the pin, as `use` does
    @test h2 != h1
    @test lowercase(gitout(clone, "rev-parse", "HEAD")) == sha2
    @test isfile(sentinel)                                            # the clone moved, it was not replaced
    RepliBuild.build(toml)
    @test hash_now() == h2                                            # the build changes no input
    RepliBuild.wrap(toml)
    @test get(gml_run("include($(repr(graph_file)))"), "total", "") == "482"
    # `use` sees a hash no vendored build has, and builds and vendors it.
    v = gml_run("RepliBuild.use(\"gm_fixture\")"; prelude = "using RepliBuild")
    @test gml_pass(v, "one_cbase") && gml_pass(v, "rcore")
    @test get(v, "total", "") == "482"
    @test isdir(home * "/builds/" * h2) && isdir(vend)
    @test RepliBuild.PackageRegistry._load_index().entries["gm_fixture"].build_hash == h2

    # ── a pin the tag does not meet is a refusal, before any configure ───────
    write_toml("v2", "f" ^ 40)
    e = try RepliBuild.build(toml); "" catch err; sprint(showerror, err) end
    @test occursin("upstream", e) && occursin("f" ^ 40, e)
    @test lowercase(gitout(clone, "rev-parse", "HEAD")) == sha2       # the clone is as it was
    write_toml("v2", sha2)

    # ── clean removes the clone with the cache ───────────────────────────────
    RepliBuild.clean(toml)
    @test !isdir(proj * "/.replibuild_cache") && !isdir(proj * "/julia") && !isdir(proj * "/build")
    io = IOBuffer()
    RepliBuild.GraphMode.describe(io, RepliBuild.ConfigurationManager.load_config(toml))
    @test occursin("(not fetched)", String(take!(io)))

    end # withenv
end
end # git
end # ninja
end # requires
