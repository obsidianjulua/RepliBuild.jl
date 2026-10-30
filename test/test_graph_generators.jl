#!/usr/bin/env julia
# test/test_graph_generators.jl — which build steps `GraphGenerators` asks for
# to make a graph's generated sources exist, in what order, and when it refuses.
#
# Needs no toolchain: graphs are built by hand, and ninja's graph comes from a
# fake that answers `-t targets`, `-t query` and `-t commands -s` with ninja's
# semantics. The edges mirror what cmake 4.4's Ninja generator writes; the
# parsers are fed text ninja 1.13 printed. test_graph_generators_live.jl runs
# the real thing.

using Test
using RepliBuild

const BG = RepliBuild.BuildGraph
const GG = RepliBuild.GraphGenerators

const ROOT = Sys.iswindows() ? "C:/rbgg" : "/rbgg"
const SRC = ROOT * "/src"
const BLD = ROOT * "/build"
bp(p) = BLD * "/" * p
sp(p) = SRC * "/" * p

# One build edge: the nodes it writes (ninja's spellings), the edge, its command.
struct FakeNinja <: GG.NinjaSource
    edges::Vector{Tuple{Vector{String},GG.NinjaEdge,String}}
end

_edge_index(f::FakeNinja, n) = findfirst(e -> n in e[1], f.edges)
GG.ninja_outputs(f::FakeNinja) = [o => e.rule for (outs, e, _) in f.edges for o in outs]
function GG.ninja_query(f::FakeNinja, nodes::Vector{String})
    out = Dict{String,Union{Nothing,GG.NinjaEdge}}()
    for n in nodes
        i = _edge_index(f, n)
        out[n] = i === nothing ? nothing : f.edges[i][2]
    end
    return out
end
# `-t commands -s`: each distinct non-phony edge once, in first-named order.
function GG.ninja_commands(f::FakeNinja, nodes::Vector{String})
    seen = Set{Int}()
    out = String[]
    for n in nodes
        i = _edge_index(f, n)
        (i === nothing || i in seen) && continue
        push!(seen, i)
        f.edges[i][2].rule == "phony" || push!(out, f.edges[i][3])
    end
    return out
end

E(rule; ex = String[], im = String[], oo = String[]) = GG.NinjaEdge(rule, ex, im, oo)
custom(outs, cmd; kw...) = (outs, E("CUSTOM_COMMAND"; kw...), cmd)
compile(out, target) = ([out], E("C_COMPILER__$(target)_unscanned_"; ex = [sp("x.c")]), "cc -c")
link(outs, kind, target) = (outs, E("C_$(kind)_LINKER__$(target)_"; ex = ["CMakeFiles/$target.dir/x.c.obj"]), "ld")

grp() = BG.CompileGroup("C", String[], String[], BG.IncludeDir[])
src(p; gen = false, group = 1) = BG.GraphSource(p, group, gen)
tgt(name, kind; sources = BG.GraphSource[], deps = String[], artifacts = String[]) =
    BG.GraphTarget(name, name, kind, sources, [grp()], deps, String[], artifacts)
graph(ts...) = BG.TargetGraph(SRC, BLD, collect(ts))

outs_of(plan) = [[basename(o) for o in s.outputs] for s in plan.steps]

# libbase, and a library `gg` that links it and lists two files one custom
# command writes. cmake gives the command an order-only input on libbase's
# artifact, and writes each output under a second, absolute spelling.
const BASE = tgt("base", :shared; sources = [src(sp("base.c"))],
                 artifacts = [bp("libbase.dll"), bp("libbase.dll.a")])
gg_target(extra = BG.GraphSource[]) =
    tgt("gg", :shared; deps = ["base"], artifacts = [bp("libgg.dll")],
        sources = vcat([src(sp("main.c")), src(bp("gen.c"); gen = true),
                        src(bp("gen.h"); gen = true, group = 0)], extra))
const COPY_CMD = "cmake -E copy $(sp("gen.h.in")) $(bp("gen.h")) && cmake -E copy $(sp("gen.c.in")) $(bp("gen.c"))"
base_edges() = [compile("CMakeFiles/base.dir/base.c.obj", "base"),
                link(["libbase.dll", "libbase.dll.a"], "SHARED_LIBRARY", "base"),
                (["cmake_object_order_depends_target_base"], E("phony"; oo = ["."]), "")]
gen_edge() = custom(["gen.h", "gen.c", bp("gen.h"), bp("gen.c")], COPY_CMD;
                    ex = [sp("gen.h.in"), sp("gen.c.in")], oo = ["libbase.dll"])

@testset "GraphGenerators" begin

    @testset "requested: generated sources, targets in dependency order, then listing order" begin
        a = tgt("a", :static; sources = [src(bp("a2.h"); gen = true, group = 0), src(sp("a.c")),
                                         src(bp("a1.c"); gen = true)])
        b = tgt("b", :shared; deps = ["a"],
                sources = [src(bp("b.c"); gen = true), src(bp("a1.c"); gen = true)])
        # listed b first: the order follows the graph, not the list
        @test GG.requested_sources(graph(b, a)) == [bp("a2.h"), bp("a1.c"), bp("b.c")]
        @test GG.requested_sources(graph(BASE)) == String[]
    end

    @testset "a library's artifact the command does not name is ordering, and dropped" begin
        ninja = FakeNinja(vcat(base_edges(), [gen_edge()]))
        plan = GG.plan_generators(graph(BASE, gg_target()), ninja)
        @test plan.requested == [bp("gen.c"), bp("gen.h")]
        @test isempty(plan.needs) && isempty(plan.unwritten) && isempty(plan.existing)
        @test plan.dropped == ["base"]
        # one step writes both
        @test length(plan.steps) == 1
        s = only(plan.steps)
        @test sort(s.outputs) == [bp("gen.c"), bp("gen.h")]
        @test s.command == COPY_CMD
        @test s.inputs == [sp("gen.h.in"), sp("gen.c.in")]
        @test isempty(s.after)

        m = GG.generator_manifest(plan)
        @test !occursin("libbase", m)
        @test occursin("builddir = CMakeFiles/replibuild_generators", m)
        line = only(filter(l -> startswith(l, "build "), split(m, '\n')))
        # one spelling per file: relative inside the build tree, escaped outside it
        esc(p) = replace(p, ':' => "\$:")
        @test line == "build gen.c gen.h: step $(esc(sp("gen.h.in"))) $(esc(sp("gen.c.in")))" ||
              line == "build gen.h gen.c: step $(esc(sp("gen.h.in"))) $(esc(sp("gen.c.in")))"
        @test occursin("  cmd = " * COPY_CMD, m)
    end

    @testset "chained steps run in dependency order, whatever order they are listed in" begin
        # table.h's step reads gen.h, which it names by cmake's absolute spelling
        table = custom(["table.h", bp("table.h")], "awk -f t.awk gen.h > table.h";
                       ex = [bp("gen.h"), sp("t.awk")], oo = ["libbase.dll"])
        ninja = FakeNinja(vcat(base_edges(), [table, gen_edge()]))
        # table.h listed before the files whose step it needs
        gg = tgt("gg", :shared; deps = ["base"],
                 sources = [src(bp("table.h"); gen = true, group = 0), src(sp("main.c")),
                            src(bp("gen.c"); gen = true), src(bp("gen.h"); gen = true, group = 0)])
        plan = GG.plan_generators(graph(BASE, gg), ninja)
        @test plan.requested == [bp("table.h"), bp("gen.c"), bp("gen.h")]
        @test length(plan.steps) == 2
        @test sort(outs_of(plan)[1]) == ["gen.c", "gen.h"]
        @test outs_of(plan)[2] == ["table.h"]
        @test bp("gen.h") in plan.steps[2].inputs
        # both spellings of gen.h are one node of the narrowed manifest
        m = GG.generator_manifest(plan)
        @test occursin("build table.h: step gen.h ", m)
        blines = filter(startswith("build "), split(m, '\n'))
        @test length(blines) == 2
        @test !any(l -> occursin(replace(bp("gen.h"), ':' => "\$:"), l), blines)
    end

    @testset "an executable the command runs is refused, naming the tool target" begin
        tool = tgt("tool", :executable; sources = [src(sp("tool.c"))], artifacts = [bp("tool.exe")])
        gt = tgt("gt", :shared; deps = ["tool"], artifacts = [bp("libgt.dll")],
                 sources = [src(sp("gt.c")), src(bp("tool_out.h"); gen = true, group = 0)])
        ninja = FakeNinja([compile("CMakeFiles/tool.dir/tool.c.obj", "tool"),
                           link(["tool.exe"], "EXECUTABLE", "tool"),
                           custom(["tool_out.h"], "$(bp("tool.exe")) $(bp("tool_out.h"))";
                                  oo = ["tool.exe"])])
        plan = GG.plan_generators(graph(tool, gt), ninja)
        n = only(plan.needs)
        @test (n.target, n.artifact, n.why, n.needed_by) == ("tool", bp("tool.exe"), :tool, [bp("tool_out.h")])
        @test isempty(plan.dropped)
        e = GG.GeneratorNeedsCompile(BLD, plan.needs)
        msg = sprint(showerror, e)
        @test occursin("target 'tool'", msg) && occursin("Nothing ran", msg)

        # With no artifacts to go by, cmake's rule name says executable, and the
        # target is named from it.
        bare = tgt("tool", :executable; sources = [src(sp("tool.c"))])
        n2 = only(GG.plan_generators(graph(bare, gt), ninja).needs)
        @test (n2.target, n2.why) == ("tool", :tool)
    end

    @testset "a library is a need when declared as an input or named by the command" begin
        # DEPENDS on the library file: an explicit input written by a link
        declared = custom(["gen.h", "gen.c"], COPY_CMD; ex = [sp("gen.h.in"), "libbase.dll.a"])
        plan = GG.plan_generators(graph(BASE, gg_target()), FakeNinja(vcat(base_edges(), [declared])))
        n = only(plan.needs)
        @test (n.target, n.artifact, n.why) == ("base", bp("libbase.dll.a"), :input)
        @test n.needed_by == [bp("gen.c"), bp("gen.h")]

        # the command loads the library by path (say, a stub generator)
        cmd = Sys.iswindows() ? "python stubgen.py " * replace(bp("libbase.dll"), '/' => '\\') :
                                "python stubgen.py " * bp("libbase.dll")
        named = custom(["gen.h", "gen.c"], cmd; ex = [sp("stubgen.py")], oo = ["libbase.dll"])
        plan = GG.plan_generators(graph(BASE, gg_target()), FakeNinja(vcat(base_edges(), [named])))
        n = only(plan.needs)
        @test (n.target, n.why) == ("base", :named)
        @test isempty(plan.dropped)
    end

    @testset "a requested file a compile writes is refused for itself" begin
        g = graph(BASE, tgt("odd", :static; deps = ["base"],
                            sources = [src(bp("CMakeFiles/base.dir/base.c.obj"); gen = true, group = 0)]))
        n = only(GG.plan_generators(g, FakeNinja(base_edges())).needs)
        @test (n.target, n.why, n.needed_by) == ("base", :input, [bp("CMakeFiles/base.dir/base.c.obj")])
    end

    @testset "a generated file no step writes: present is configure-time, absent is an error" begin
        dir = mktempdir()
        present = replace(joinpath(dir, "config.h"), '\\' => '/')
        write(present, "#define X 1\n")
        absent = replace(joinpath(dir, "never.h"), '\\' => '/')
        t = tgt("t", :static; sources = [src(present; gen = true, group = 0),
                                         src(absent; gen = true, group = 0)])
        # cmake's "may be missing" marker is a phony edge with no inputs
        plan = GG.plan_generators(graph(t), FakeNinja([([absent], E("phony"), "")]))
        @test plan.existing == [present]
        @test plan.unwritten == [absent]
        @test isempty(plan.steps)
        msg = sprint(showerror, GG.GeneratedSourceUnwritten(BLD, plan.unwritten))
        @test occursin("never.h", msg)
    end

    @testset "steps are told apart by their commands, not their inputs" begin
        # Two commands of one target with no DEPENDS read the same inputs (only
        # the target's order-only ones). One of them writes two files.
        a = custom(["a.h", "a2.h"], "gen a"; oo = ["libbase.dll"])
        b = custom(["b.h"], "gen b"; oo = ["libbase.dll"])
        t = tgt("t", :shared; deps = ["base"],
                sources = [src(bp(f); gen = true, group = 0) for f in ("a.h", "b.h", "a2.h")])
        plan = GG.plan_generators(graph(BASE, t), FakeNinja(vcat(base_edges(), [a, b])))
        @test [s.command for s in plan.steps] == ["gen a", "gen b"]
        @test outs_of(plan) == [["a.h", "a2.h"], ["b.h"]]
    end

    @testset "phony edges pass needs and ordering through" begin
        # gen.h's step DEPENDS on a utility target (a phony over its custom
        # command) and is ordered after another target's generated header
        # through cmake's object-order phony.
        util = custom(["CMakeFiles/util.util"], "gen util"; oo = ["libbase.dll"])
        other = custom(["other.h"], "gen other")
        edges = vcat(base_edges(),
                     [util, (["util"], E("phony"; ex = ["CMakeFiles/util.util"]), ""),
                      (["cmake_object_order_depends_target_o"], E("phony"; oo = ["other.h"]), ""),
                      other,
                      custom(["gen.h"], "gen h"; ex = ["util"],
                             oo = ["cmake_object_order_depends_target_o", "libbase.dll"])])
        o = tgt("o", :static; sources = [src(bp("other.h"); gen = true, group = 0)])
        t = tgt("t", :shared; deps = ["base", "o"], sources = [src(bp("gen.h"); gen = true, group = 0)])
        plan = GG.plan_generators(graph(BASE, o, t), FakeNinja(edges))
        @test [s.command for s in plan.steps] == ["gen other", "gen util", "gen h"]
        h = plan.steps[end]
        @test h.inputs == [bp("CMakeFiles/util.util")]
        @test h.after == [bp("other.h")]
        @test plan.dropped == ["base"]
        m = GG.generator_manifest(plan)
        @test occursin("build gen.h: step CMakeFiles/util.util || other.h", m)
    end

    @testset "a step that only one walk reaches as ordering is not run for it" begin
        # o's generated header is not requested, so the step that writes it is
        # ordering for gen.h's, not a step.
        edges = [custom(["other.h"], "gen other"),
                 (["cmake_object_order_depends_target_o"], E("phony"; oo = ["other.h"]), ""),
                 custom(["gen.h"], "gen h"; oo = ["cmake_object_order_depends_target_o"])]
        t = tgt("t", :shared; sources = [src(bp("gen.h"); gen = true, group = 0)])
        plan = GG.plan_generators(graph(t), FakeNinja(edges))
        @test [s.command for s in plan.steps] == ["gen h"]
        @test isempty(only(plan.steps).after)
    end

    @testset "manifest escaping" begin
        plan = GG.GeneratorPlan(BLD, String[], String[], String[],
                                [GG.GeneratorStep([bp("out dir/x\$.h")], "echo \$HOME > x",
                                                  [ROOT * "/in put/a.txt"], String[])],
                                GG.NeededTarget[], String[])
        m = GG.generator_manifest(plan)
        @test occursin("build out\$ dir/x\$\$.h: step " *
                       replace(ROOT * "/in\$ put/a.txt", ':' => "\$:"), m)
        @test occursin("  cmd = echo \$\$HOME > x", m)
        bad = GG.GeneratorPlan(BLD, String[], String[], String[],
                               [GG.GeneratorStep([bp("a|b.h")], "x", String[], String[])],
                               GG.NeededTarget[], String[])
        @test_throws ErrorException GG.generator_manifest(bad)
    end

    @testset "parsers read what ninja 1.13 prints" begin
        # `ninja -t query gen.h tool_out.h source.c` on cmake 4.4's build.ninja,
        # with Windows' "\r\n" line ends
        q = join(["gen.h:", "  input: CUSTOM_COMMAND", "    C:/s/gen.h.in", "    | gen.dep",
                  "    || libbase.dll", "  validations:", "    check.stamp", "  outputs:",
                  "    cmake_object_order_depends_target_gg",
                  "tool_out.h:", "  input: CUSTOM_COMMAND", "    || tool.exe", "  outputs:",
                  "    cmake_object_order_depends_target_gt",
                  "C:/s/source.c:", "  outputs:", "    CMakeFiles/gg.dir/source.c.obj", ""], "\r\n")
        d = GG.parse_query(q)
        @test d["gen.h"] == E("CUSTOM_COMMAND"; ex = ["C:/s/gen.h.in"], im = ["gen.dep"],
                              oo = ["libbase.dll"])
        @test d["tool_out.h"] == E("CUSTOM_COMMAND"; oo = ["tool.exe"])
        @test haskey(d, "C:/s/source.c") && d["C:/s/source.c"] === nothing
        @test length(d) == 3

        t = GG.parse_targets("cmake_object_order_depends_target_base: phony\r\n" *
                             "libbase.dll: C_SHARED_LIBRARY_LINKER__base_\r\n" *
                             "C:/b/gen.h: CUSTOM_COMMAND\r\n")
        @test t == ["cmake_object_order_depends_target_base" => "phony",
                    "libbase.dll" => "C_SHARED_LIBRARY_LINKER__base_",
                    "C:/b/gen.h" => "CUSTOM_COMMAND"]

        dir = mktempdir()
        log = joinpath(dir, ".ninja_log")
        write(log, "# ninja log v7\n0\t85\t8150914242241405\tgen.h\t83873a4395fe8d92\n" *
                   "0\t85\t8150914242241405\tsub/gen.c\t83873a4395fe8d92\n")
        @test GG._logged(log, BLD) == Set(GG._key.([bp("gen.h"), bp("sub/gen.c")]))
        @test isempty(GG._logged(joinpath(dir, "absent"), BLD))
    end

    @testset "build trees it does not read" begin
        # nothing generated: nothing to run, and no build tree needed
        r = GG.run_generators(graph(BASE))
        @test isempty(r.plan.steps) && isempty(r.ran) && r.manifest == ""

        dir = mktempdir()
        @test_throws ErrorException GG.ninja_program(dir)
        write(joinpath(dir, "CMakeCache.txt"),
              "# comment\n//help\nCMAKE_GENERATOR:INTERNAL=Unix Makefiles\n" *
              "CMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/make\n")
        msg = try GG.ninja_program(dir); "" catch e; sprint(showerror, e) end
        @test occursin("'Unix Makefiles'", msg) && occursin("-G Ninja", msg)
        write(joinpath(dir, "CMakeCache.txt"),
              "CMAKE_GENERATOR:INTERNAL=Ninja\nCMAKE_MAKE_PROGRAM:FILEPATH=C:/tools/ninja.exe\n")
        @test_throws ErrorException GG.ninja_program(dir)      # no build.ninja yet
        write(joinpath(dir, "build.ninja"), "")
        @test GG.ninja_program(dir) == "C:/tools/ninja.exe"
    end
end
