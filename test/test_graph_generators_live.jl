#!/usr/bin/env julia
# test/test_graph_generators_live.jl — `GraphGenerators.run_generators` against
# a real cmake configure of test/graph_generators_test/ with Ninja.
#
# The fixture's generated sources come from steps that run cmake only, in a
# library that links another library of the project, so cmake orders the steps
# after that library's artifact: the run must write them and compile nothing.
# A second library's header comes from a tool the project compiles, which must
# be refused before anything runs. The graph is built by hand to match what
# cmake's File API reports for the fixture.
#
# Needs cmake, ninja and a C compiler (configure identifies one), hence
# devtests. test_graph_generators.jl covers the planning without a toolchain.

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const GGL = RepliBuild.GraphGenerators
const BGL = RepliBuild.BuildGraph

_gg_posix(p) = replace(String(p), '\\' => '/')

# Files a compile or a link wrote, outside cmake's configure-time probes
# (CMakeFiles/<version>/ and CMakeFiles/CMakeScratch/).
function _gg_built(bld::String)
    out = String[]
    for (root, _, files) in walkdir(bld), f in files
        rel = _gg_posix(relpath(joinpath(root, f), bld))
        occursin(r"^CMakeFiles/(\d|CMakeScratch/)", rel) && continue
        occursin(r"\.(o|obj|a|lib|so|dll|exe)$", f) && push!(out, rel)
    end
    return sort!(out)
end

if requires("Graph generators (live)", :cmake, :clang)
if Sys.which("ninja") === nothing
    @testset "Graph generators (live) (skipped — not found: ninja)" begin
        @test_skip false
    end
else
@testset "Graph generators (live)" begin
    src = _gg_posix(joinpath(@__DIR__, "graph_generators_test"))
    bld = _gg_posix(mktempdir()) * "/build"
    out, code = RepliBuild.BuildBridge.execute("cmake",
        ["-S", src, "-B", bld, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"]; use_llvm_env = false)
    code == 0 || println(stderr, out)
    @test code == 0

    b(p) = bld * "/" * p
    s(p) = src * "/" * p
    lib(name) = Sys.iswindows() ? [b("lib$name.dll"), b("lib$name.dll.a")] : [b("lib$name.so")]
    exe(name) = Sys.iswindows() ? [b("$name.exe")] : [b(name)]
    grp = BGL.CompileGroup("C", String[], String[], [BGL.IncludeDir(b("gen"), false)])
    target(name, kind, sources, deps, artifacts) =
        BGL.GraphTarget(name * "::@6890427a1f51a3e7e1df", name, kind, sources, [grp],
                        [d * "::@6890427a1f51a3e7e1df" for d in deps], String[], artifacts)
    base = target("gg_base", :shared, [BGL.GraphSource(s("base.c"), 1, false)], String[], lib("gg_base"))
    gg = target("gg", :shared,
                [BGL.GraphSource(s("main.c"), 1, false), BGL.GraphSource(b("gen/gg_gen.c"), 1, true),
                 BGL.GraphSource(b("gen/gg_gen.h"), 0, true), BGL.GraphSource(b("gen/gg_table.h"), 0, true)],
                ["gg_base"], lib("gg"))
    tool = target("gg_tool", :executable, [BGL.GraphSource(s("tool.c"), 1, false)], String[], exe("gg_tool"))
    tooluser = target("gg_tooluser", :shared,
                      [BGL.GraphSource(s("tooluser.c"), 1, false), BGL.GraphSource(b("tool_out.h"), 0, true)],
                      ["gg_tool"], lib("gg_tooluser"))

    @testset "a tool the project compiles is refused before anything runs" begin
        g = BGL.TargetGraph(src, bld, [tool, tooluser])
        e = try
            GGL.run_generators(g; use_llvm_env = false)
            nothing
        catch err
            err
        end
        @test e isa GGL.GeneratorNeedsCompile
        if e isa GGL.GeneratorNeedsCompile
            n = only(e.needs)
            @test (n.target, n.why) == ("gg_tool", :tool)
            @test n.artifact == only(exe("gg_tool"))
            @test n.needed_by == [b("tool_out.h")]
        end
        @test !isfile(b("tool_out.h"))
        @test isempty(_gg_built(bld))
    end

    @testset "steps that run cmake only write the sources and compile nothing" begin
        g = BGL.TargetGraph(src, bld, [base, gg])
        r = GGL.run_generators(g; use_llvm_env = false)
        @test r.plan.requested == [b("gen/gg_gen.c"), b("gen/gg_gen.h"), b("gen/gg_table.h")]
        @test all(isfile, r.plan.requested)
        @test strip(read(b("gen/gg_table.h"), String)) == "#define GG_TABLE_VALUE (21 * 2)"
        @test length(r.ran) == 2
        @test sort(basename.(r.ran[1].outputs)) == ["gg_gen.c", "gg_gen.h"]
        @test basename.(r.ran[2].outputs) == ["gg_table.h"]
        # cmake's order-only input on libgg_base was left out, and nothing built it
        @test r.plan.dropped == ["gg_base"]
        @test isempty(_gg_built(bld))
        @test !any(isfile, vcat(base.artifacts, gg.artifacts))
        # the project's own build was never run: ninja keeps no log for it
        @test !isfile(b(".ninja_log"))
        @test isfile(r.manifest) && startswith(r.manifest, b("CMakeFiles/replibuild_generators/"))

        # every call runs every step again, and writes the same files
        before = read(b("gen/gg_table.h"), String)
        r2 = GGL.run_generators(g; use_llvm_env = false)
        @test length(r2.ran) == 2
        @test read(b("gen/gg_table.h"), String) == before
        @test isempty(_gg_built(bld))
    end
end
end # ninja
end # requires
