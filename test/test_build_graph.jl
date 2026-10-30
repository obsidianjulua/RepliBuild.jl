#!/usr/bin/env julia
# test/test_build_graph.jl — the build-graph contract (`Builder/BuildGraph.jl`)
# that the CMake reader, the graph executor and the generator step share.
# Needs no toolchain: it builds graphs by hand.

using Test
using RepliBuild

const BG = RepliBuild.BuildGraph

tgt(id, deps = String[]; kind = :static, sources = BG.GraphSource[], groups = BG.CompileGroup[]) =
    BG.GraphTarget(id, "name_" * id, kind, sources, groups, deps, String[], String[])
graph(ts...) = BG.TargetGraph("/src", "/build", collect(ts))
ids(order) = [t.id for t in order]

@testset "Build graph contract" begin

    @testset "dependencies come first, ties in sorted id order" begin
        # d needs b and c, b and c need a; e stands alone. Listed out of order,
        # so a pass in input order would put d first.
        g = graph(tgt("d", ["c", "b"]), tgt("c", ["a"]), tgt("e"), tgt("b", ["a"]), tgt("a"))
        @test ids(BG.topological_order(g)) == ["a", "b", "c", "d", "e"]
        # The same graph listed in another order gives the same answer.
        g2 = graph(tgt("a"), tgt("b", ["a"]), tgt("e"), tgt("c", ["a"]), tgt("d", ["b", "c"]))
        @test ids(BG.topological_order(g2)) == ids(BG.topological_order(g))
    end

    @testset "a dependency listed twice is one edge" begin
        g = graph(tgt("b", ["a", "a"]), tgt("a"))
        @test ids(BG.topological_order(g)) == ["a", "b"]
    end

    @testset "malformed graphs are errors that name the targets" begin
        err(g) = try BG.topological_order(g); "" catch e; sprint(showerror, e) end
        @test occursin("'ghost'", err(graph(tgt("a", ["ghost"]))))
        @test occursin("two targets have id 'a'", err(graph(tgt("a"), tgt("a"))))
        e = err(graph(tgt("a", ["b"]), tgt("b", ["a"]), tgt("c")))
        @test occursin("cycle", e)
        @test occursin("name_a", e) && occursin("name_b", e)
        @test !occursin("name_c", e)
    end

    @testset "target_by_id" begin
        g = graph(tgt("a"), tgt("b", ["a"]))
        @test BG.target_by_id(g, "b").name == "name_b"
        @test_throws ErrorException BG.target_by_id(g, "zz")
    end

    @testset "construction checks kind and compile-group indices" begin
        grp = BG.CompileGroup("C", ["-O2"], ["X=1"], [BG.IncludeDir("/inc", false)])
        ok = tgt("a"; sources = [BG.GraphSource("/a.c", 1, false), BG.GraphSource("/a.h", 0, false)],
                 groups = [grp])
        @test ok.compile_groups[1].includes[1].path == "/inc"
        @test_throws ArgumentError tgt("a"; kind = :library)
        @test_throws ArgumentError tgt("a"; sources = [BG.GraphSource("/a.c", 2, false)], groups = [grp])
        @test Set(BG.TARGET_KINDS) == Set([:shared, :static, :object, :module, :executable, :utility, :interface])
    end
end
