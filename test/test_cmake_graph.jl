#!/usr/bin/env julia
# test/test_cmake_graph.jl — the CMake File API reader (`Builder/CMakeGraph.jl`)
# that fills the build-graph contract. Needs no toolchain: it reads vendored
# codemodel-v2 replies (test/fixtures/cmake_graph/, written by
# test/tools/gen_cmake_graph_fixtures.jl), laid out under a fresh build
# directory with the generator's `@SOURCE@`/`@BUILD@` placeholders replaced.
#
# `graphfix` is the fixture project beside the replies; `curl` is libcurl at
# the Hub's pinned commit, trimmed to `libcurl_shared` and what it depends on.

using Test
using RepliBuild
import TOML

const CG = RepliBuild.CMakeGraph
const BG = RepliBuild.BuildGraph
const CMAKE_FIXTURES = joinpath(@__DIR__, "fixtures", "cmake_graph")

# Copy a fixture's reply under `<tmp>/build/.cmake/api/v1/reply`, substituting
# the placeholders, and return the directory and the roots. `edit` may rewrite
# a file's text on the way (file name => text => text).
function lay_out_fixture(name; edit = (f, text) -> text)
    dir = joinpath(CMAKE_FIXTURES, name)
    meta = TOML.parsefile(joinpath(dir, "FIXTURE.toml"))
    root = mktempdir()
    S = CG._posix(joinpath(root, "src"))
    B = CG._posix(joinpath(root, "build"))
    reply = joinpath(B, ".cmake", "api", "v1", "reply")
    mkpath(reply)
    files = readdir(joinpath(dir, "reply"))
    for f in files
        text = replace(read(joinpath(dir, "reply", f), String), "@SOURCE@" => S, "@BUILD@" => B)
        write(joinpath(reply, f), edit(f, text))
    end
    return (; B, S, reply, meta, files)
end

function read_fixture(name; kw...)
    fx = lay_out_fixture(name)
    g = CG.read_cmake_graph(fx.B; windows_shell = fx.meta["windows_shell"], kw...)
    return g, fx
end

named(g, n) = only(t for t in g.targets if t.name == n)
raw_target(fx, n) = RepliBuild._read_json(joinpath(fx.reply, "target-$n.json"))
errtext(f) = try f(); "" catch e; sprint(showerror, e) end

@testset "CMake File API reader" begin

    @testset "command fragments split by the shell that runs them" begin
        # Both encodings on every host: the reader picks one per reply.
        for windows in (true, false)
            @test CG._split_fragment("-O3 -DNDEBUG", windows) == ["-O3", "-DNDEBUG"]
            @test CG._split_fragment("  -lkernel32\t-luser32 ", windows) == ["-lkernel32", "-luser32"]
            @test CG._split_fragment("\"-DX=\\\"a b\\\"\"", windows) == ["-DX=\"a b\""]
            @test CG._split_fragment("\"C:/Program Files/x/libz.a\"", windows) == ["C:/Program Files/x/libz.a"]
            @test CG._split_fragment("", windows) == String[]
            @test CG._split_fragment("\"\"", windows) == [""]
            @test_throws ErrorException CG._split_fragment("\"open", windows)
        end
        # Windows: a backslash is literal unless a run of them ends at a quote.
        @test CG._split_windows("C:\\x\\libcurl.def -l\\\\y") == ["C:\\x\\libcurl.def", "-l\\\\y"]
        @test CG._split_windows("\"a\\\\\" b") == ["a\\", "b"]
        @test CG._split_windows("a\\\\\\\"b") == ["a\\\"b"]
        # POSIX: backslash escapes outside quotes, single quotes are literal,
        # and `$` is never expanded.
        @test CG._split_posix("a\\ b 'c \"d\" \\e' f") == ["a b", "c \"d\" \\e", "f"]
        @test CG._split_posix("\"\\\$HOME\\\\x\" \$y") == ["\$HOME\\x", "\$y"]
        @test_throws ErrorException CG._split_posix("'open")
    end

    @testset "paths are absolute, '/'-separated and normalised on any host" begin
        @test CG._normalize("C:/a/./b/../c") == "C:/a/c"
        @test CG._normalize("/a//b/./") == "/a/b"
        @test CG._normalize("/../x") == "/x"
        @test CG._normalize("a/../../b") == "../b"
        @test CG._abs("x/./y.obj", "C:/root") == "C:/root/x/y.obj"
        @test CG._abs("D:/elsewhere/z", "/root") == "D:/elsewhere/z"
        @test CG._abs("/abs/z", "C:/root") == "/abs/z"
        @test CG._is_abs("C:/x") && CG._is_abs("/x") && !CG._is_abs("x/C:/y")
    end

    @testset "graphfix: the fixture is what the generator wrote" begin
        fx = lay_out_fixture("graphfix")
        # The loop ran over every file the generator writes for four targets.
        @test sort(fx.files) == sort(["index-0.json", "codemodel-v2.json", "target-core.json",
                                      "target-ext.json", "target-ext_version.json",
                                      "target-helpers.json"])
        # No machine path survived into the vendored text.
        for f in fx.files
            text = read(joinpath(CMAKE_FIXTURES, "graphfix", "reply", f), String)
            @test !occursin(r"[A-Za-z]:/|/home/|/Users/", text)
        end
    end

    g, fx = read_fixture("graphfix")
    S, B = fx.S, fx.B
    core, ext, helpers, ver = named(g, "core"), named(g, "ext"), named(g, "helpers"), named(g, "ext_version")

    @testset "graphfix: roots, kinds and compile groups" begin
        @test g.source_dir == S
        @test g.build_dir == B
        @test [t.name for t in g.targets] == ["core", "ext", "ext_version", "helpers"]
        @test (core.kind, ext.kind, helpers.kind, ver.kind) == (:shared, :shared, :object, :utility)

        cg = only(core.compile_groups)
        @test cg.language == "CXX"
        # `-O3 -DNDEBUG` is one fragment in the reply and two arguments here.
        @test cg.flags[1:2] == ["-O3", "-DNDEBUG"]
        @test cg.defines == ["CORE_BUILDING=1", "core_EXPORTS"]
        @test [(i.path, i.is_system) for i in cg.includes] == [(S * "/core/include", false)]

        hg = only(helpers.compile_groups)
        @test hg.language == "C"
        @test isempty(hg.defines) && isempty(hg.includes)
        @test [(s.path, s.group, s.generated) for s in helpers.sources] == [(S * "/ext/helpers.c", 1, false)]

        eg = only(ext.compile_groups)
        @test eg.language == "CXX"
        @test eg.defines == ["EXT_LEVEL=2", "ext_EXPORTS"]
        @test !("EXT_LEVEL=2" in cg.defines)            # a PRIVATE define stays home
        @test (S * "/ext/ext.cpp", 1, false) in [(s.path, s.group, s.generated) for s in ext.sources]
        @test isempty(ver.compile_groups)
    end

    @testset "graphfix: a PUBLIC include reaches the dependent's compile group" begin
        incs = [i.path for i in only(ext.compile_groups).includes]
        @test S * "/core/include" in incs                # inherited through the link edge
        @test B * "/gen" in incs                         # ext's own PRIVATE one
    end

    @testset "graphfix: an OBJECT library is a dependency and object sources" begin
        @test helpers.id in ext.dependencies
        obj = only(helpers.artifacts)
        # The reply spells the artifact `CMakeFiles/helpers.dir/./ext/…` and the
        # consumer's source `…/helpers.dir/ext/…`; normalised, they are one file.
        @test obj == B * "/CMakeFiles/helpers.dir/ext/helpers.c.obj"
        @test occursin("/./", only(raw_target(fx, "helpers")["artifacts"])["path"])
        objsrc = [s for s in ext.sources if s.path == obj]
        @test length(objsrc) == 1
        @test only(objsrc).group == 0 && only(objsrc).generated
        @test !any(l -> occursin("helpers", l), ext.link_libraries)
    end

    @testset "graphfix: generated sources and the targets that hold their rules" begin
        @test [(s.path, s.group, s.generated) for s in core.sources] ==
              [(S * "/core/core.cpp", 1, false),
               (B * "/gen/table.cpp", 1, true),               # compiled, written by the build
               (B * "/gen/table.cpp.rule", 0, true)]          # the custom command's rule
        # A custom target holds the rule for a header it DEPENDS on; the header
        # itself is not one of its sources, and `ext` reaches it by an edge.
        paths = [s.path for s in ver.sources]
        @test B * "/gen/ext_version.h.rule" in paths
        @test !(B * "/gen/ext_version.h" in paths)
        @test all(s -> s.group == 0 && s.generated, ver.sources)
        @test ver.id in ext.dependencies
        @test core.id in ext.dependencies
        @test !any(s -> s.generated, helpers.sources)
    end

    @testset "graphfix: link libraries" begin
        @test "-lgraphfix_syslib" in ext.link_libraries
        # The reply's link step names core's import library; it is a target of
        # the graph, so it is not a link library.
        frags = [f["fragment"] for f in raw_target(fx, "ext")["link"]["commandFragments"]]
        @test any(f -> CG._abs(f, B) in core.artifacts, frags)
        @test !any(l -> CG._abs(l, B) in core.artifacts, ext.link_libraries)
        # Every argument of a multi-library fragment is its own entry.
        @test all(l -> !occursin(' ', l), ext.link_libraries)
        @test isempty(helpers.link_libraries) && isempty(ver.link_libraries)
        @test core.artifacts[1] == B * "/" * basename(core.artifacts[1])
    end

    @testset "graphfix: topological_order" begin
        order = [t.name for t in BG.topological_order(g)]
        # Ties break by sorted id ("core::…" < "ext_version::…" < "helpers::…").
        @test order == ["core", "ext_version", "helpers", "ext"]
        pos = Dict(n => i for (i, n) in enumerate(order))
        for t in g.targets, d in t.dependencies
            @test pos[BG.target_by_id(g, d).name] < pos[t.name]
        end
    end

    @testset "dependencies are sorted, whatever order the reply gives" begin
        # Two configures of the fixture project listed ext's dependencies in
        # two orders; the graph must not follow either.
        @test issorted(ext.dependencies)
        rev = lay_out_fixture("graphfix")
        t_path = joinpath(rev.reply, "target-ext.json")
        t = RepliBuild._read_json(t_path)
        reverse!(t["dependencies"])
        @test !issorted([d["id"] for d in t["dependencies"]])
        write(t_path, RepliBuild.JSON.json(t))
        g2 = CG.read_cmake_graph(rev.B; windows_shell = rev.meta["windows_shell"])
        @test named(g2, "ext").dependencies == ext.dependencies   # ids carry no paths
    end

    @testset "configuration choice" begin
        # Two configurations, as a multi-config generator writes them.
        two = lay_out_fixture("graphfix")
        cm_path = joinpath(two.reply, "codemodel-v2.json")
        cm = RepliBuild._read_json(cm_path)
        push!(cm["configurations"], merge(only(cm["configurations"]), Dict("name" => "Debug")))
        write(cm_path, RepliBuild.JSON.json(cm))
        e = errtext(() -> CG.read_cmake_graph(two.B; windows_shell = true))
        @test occursin("2 configurations", e) && occursin("\"Release\"", e) && occursin("\"Debug\"", e)
        @test length(CG.read_cmake_graph(two.B; config = "Debug", windows_shell = true).targets) == 4
        e = errtext(() -> CG.read_cmake_graph(two.B; config = "MinSizeRel", windows_shell = true))
        @test occursin("MinSizeRel", e) && occursin("\"Debug\"", e)
    end

    @testset "a missing or failed reply is an error that says so" begin
        e = errtext(() -> CG.read_cmake_graph(mktempdir()))
        @test occursin("no File API reply", e)
        failed = lay_out_fixture("graphfix")
        write(joinpath(failed.reply, "error-1.json"), "{}")   # sorts after index-0
        @test occursin("last configure", errtext(() -> CG.read_cmake_graph(failed.B)))
        bad = lay_out_fixture("graphfix"; edit = (f, text) -> f == "target-helpers.json" ?
            replace(text, "OBJECT_LIBRARY" => "UNKNOWN_LIBRARY") : text)
        e = errtext(() -> CG.read_cmake_graph(bad.B))
        @test occursin("'helpers'", e) && occursin("UNKNOWN_LIBRARY", e)
    end

    @testset "curl: libcurl_shared links the objects of libcurl_object" begin
        cg, cfx = read_fixture("curl")
        @test cfx.meta["upstream"]["commit"] == "68720b4837284335b2d63cb358f8f6ce65f5bc55"
        @test sort([t.name for t in cg.targets]) == ["libcurl_object", "libcurl_shared"]
        shared, obj = named(cg, "libcurl_shared"), named(cg, "libcurl_object")
        @test (shared.kind, obj.kind) == (:shared, :object)
        @test shared.dependencies == [obj.id]
        @test [t.name for t in BG.topological_order(cg)] == ["libcurl_object", "libcurl_shared"]

        # The edge compile_commands.json cannot express: the shared library
        # compiles next to nothing and takes every object of the object library.
        compiled = count(s -> s.group > 0, obj.sources)
        @test compiled >= 100
        @test length(obj.artifacts) == compiled
        objsrc = [s.path for s in shared.sources if s.group == 0 && s.generated]
        @test Set(objsrc) == Set(obj.artifacts)
        @test count(s -> s.group > 0, shared.sources) < 5

        og = only(obj.compile_groups)
        @test og.language == "C"
        @test "-fvisibility=hidden" in og.flags
        @test "BUILDING_LIBCURL" in og.defines
        @test cfx.S * "/include" in [i.path for i in og.includes]
        @test "libcurl_shared_EXPORTS" in shared.compile_groups[1].defines
        @test all(l -> startswith(l, "-l"), shared.link_libraries)
        @test !isempty(shared.link_libraries)
    end
end
