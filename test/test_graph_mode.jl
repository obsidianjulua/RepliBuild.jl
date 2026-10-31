#!/usr/bin/env julia
# test/test_graph_mode.jl — graph mode's planning half (`Builder/GraphMode.jl`):
# the [graph] section, the library module names, the graph module's text, the
# link and metadata cache keys, what the build asks the generators for, the
# registry's view of a graph (inputs, vendored files), and the Windows preload
# of a graph library's dependencies. Needs no toolchain: graphs are built by
# hand and nothing is compiled. test_graph_mode_live.jl (devtests §22) builds,
# wraps, registers and calls a CMake project.

using Test
using TOML
using RepliBuild

const GM = RepliBuild.GraphMode
const GMB = RepliBuild.GraphBuild
const GMG = RepliBuild.BuildGraph
const GMC = RepliBuild.ConfigurationManager

_gm_posix(p) = replace(String(p), '\\' => '/')
_gm_error(f) = try f(); "" catch e; sprint(showerror, e) end

function gm_project(dir::String, body::String)
    mkpath(dir)
    write(joinpath(dir, "replibuild.toml"), "[project]\nname = \"gm_unit\"\nroot = \".\"\n\n" * body)
    return joinpath(dir, "replibuild.toml")
end

gm_group(lang) = GMG.CompileGroup(lang, String[], String[], GMG.IncludeDir[])
gm_target(root, id, kind; deps = String[], sources = String[], generated = String[],
          lang = "C", artifacts = String[]) =
    GMG.GraphTarget(id, id, kind,
                    vcat([GMG.GraphSource(root * "/src/" * s, 1, false) for s in sources],
                         [GMG.GraphSource(root * "/build/" * s, 0, true) for s in generated]),
                    isempty(sources) ? GMG.CompileGroup[] : [gm_group(lang)], deps, String[], artifacts)

@testset "Graph mode" begin

    @testset "[graph] section" begin
        mktempdir() do dir
            toml = gm_project(joinpath(dir, "p"), """
                [paths]
                build = "out/b"
                output = "wrapped"

                [graph]
                source = "upstream"
                targets = ["Core", "Hist", "Core"]
                options = ["-DFOO=ON"]
                build_type = "RelWithDebInfo"
                """)
            gc = GMC.load_config(toml).graph
            root = _gm_posix(abspath(joinpath(dir, "p")))
            @test gc isa GMC.GraphConfig
            @test gc.source == root * "/upstream"
            @test gc.targets == ["Core", "Hist"]            # once each, in order
            @test gc.options == ["-DFOO=ON"]
            @test gc.build_type == "RelWithDebInfo"
            @test gc.cmake_dir == root * "/out/b/cmake"
            @test gc.out_dir == root * "/wrapped"
            @test !occursin('\\', gc.source * gc.cmake_dir * gc.out_dir)

            # Defaults: the project root is the CMake project; every library.
            d = GMC.load_config(gm_project(joinpath(dir, "q"), "[graph]\n")).graph
            qroot = _gm_posix(abspath(joinpath(dir, "q")))
            @test d.source == qroot && isempty(d.targets) && isempty(d.options)
            @test d.build_type == "Release"
            @test d.cmake_dir == qroot * "/build/cmake" && d.out_dir == qroot * "/julia"
            @test GMC.load_config(gm_project(joinpath(dir, "r"), "")).graph === nothing

            # A new section is strict: an unknown key or a wrong type names itself.
            e = _gm_error(() -> GMC.load_config(gm_project(joinpath(dir, "s"), "[graph]\ntarget = [\"x\"]\n")))
            @test occursin("[graph] target", e) && occursin("unknown key", e)
            e = _gm_error(() -> GMC.load_config(gm_project(joinpath(dir, "t"), "[graph]\ntargets = \"x\"\n")))
            @test occursin("[graph] targets", e)
            e = _gm_error(() -> GMC.load_config(gm_project(joinpath(dir, "u"),
                                                           "[graph]\n[ingest]\nlibrary = \"x.so\"\n")))
            @test occursin("[ingest] and [graph]", e)

            # save_config writes the section back, the source relative to the root,
            # and nothing it derived.
            cfg = GMC.load_config(toml)
            out = joinpath(dir, "saved.toml")
            GMC.save_config(RepliBuild.ConfigurationManager.RepliBuildConfig(
                (getfield(cfg, f) for f in fieldnames(typeof(cfg)) if f ∉ (:config_file, :loaded_at))...,
                out, cfg.loaded_at))
            saved = TOML.parsefile(out)["graph"]
            @test saved["targets"] == ["Core", "Hist"] && saved["options"] == ["-DFOO=ON"]
            @test saved["build_type"] == "RelWithDebInfo"
            @test sort!(collect(keys(saved))) == ["build_type", "options", "source", "targets"]
            @test GMC.parse_graph_config(Dict{String,Any}("graph" => saved), root,
                                         cfg.paths).source == root * "/upstream"

            # A forced re-discovery keeps the whole section.
            @test ("graph", "*") in RepliBuild.Discovery.PRESERVED_TOML_KEYS
            kept = RepliBuild.Discovery._collect_preserved_sections(toml)
            @test kept[("graph", "targets")] == ["Core", "Hist", "Core"]
            @test kept[("graph", "source")] == "upstream"
        end
    end

    @testset "library module names" begin
        names = GM.library_module_names
        # module_name_for first, then the rule.
        @test names(["cbase", "cxx_lib"], "G") == ["Cbase", "CxxLib"]
        # Bound in Base or Core: an R in front.
        @test names(["Core", "Base", "Main", "matrix", "string"], "G") ==
              ["RCore", "RBase", "RMain", "RMatrix", "RString"]
        # Named by every wrapper (`Libdl.dlopen`, `RepliBuild.SHA`).
        @test names(["libdl", "RepliBuild"], "G") == ["RLibdl", "RRepliBuild"]
        # A digit first: module_name_for's `_`.
        @test names(["3d"], "G") == ["_3d"]
        # The graph module's own name, and earlier libraries', ignoring case.
        @test names(["zlib"], "Zlib") == ["Zlib_2"]
        @test names(["foo_bar", "foo-bar", "FooBar"], "G") == ["FooBar", "FooBar_2", "FooBar_3"]
        @test names(["a_bc", "abc"], "G") == ["ABc", "Abc_2"]
        @test names(["core", "rcore"], "G") == ["RCore", "Rcore_2"]
        # A function of the input: same in, same out; order decides who renames.
        @test names(["rcore", "core"], "G") == ["Rcore", "RCore_2"]
        @test names(["b", "a", "Core"], "G") == names(["b", "a", "Core"], "G")

        u = GM.unusable_module_name
        @test u("Core") && u("Base") && u("Libdl") && u("end") && u("3d") && u("a-b")
        @test !u("RCore") && !u("Cbase") && !u("_3d")
        for n in names(["Core", "end", "main", "x", "Matrix", "3", "Libdl"], "Graph")
            @test !u(n)
            @test Meta.parse(n) === Symbol(n)
        end

        # The graph module is never renamed (the registry finds it by name); an
        # unusable one is an error that says what to set.
        mktempdir() do dir
            cfg = GMC.load_config(gm_project(joinpath(dir, "core"), "[graph]\n"))
            cfg = GMC.RepliBuildConfig(GMC.ProjectConfig("core", cfg.project.root, cfg.project.uuid),
                                       (getfield(cfg, f) for f in fieldnames(typeof(cfg))[2:end])...)
            @test occursin("[wrap] module_name", _gm_error(() -> GM.graph_module_name(cfg)))
        end
    end

    root = _gm_posix(mktempdir())
    out = root * "/julia"
    # zeta sorts last but is linked first; extra is not selected; util writes a
    # header beta reads.
    zeta = gm_target(root, "zeta", :shared; sources = ["zeta.c"])
    beta = gm_target(root, "beta", :shared; sources = ["beta.c"], deps = ["zeta", "util"])
    alpha = gm_target(root, "alpha", :shared; sources = ["alpha.cpp"], lang = "CXX", deps = ["beta", "zeta"])
    util = gm_target(root, "util", :utility; generated = ["gen/u.h", "gen/u.h.rule"])
    extra = gm_target(root, "extra", :shared; sources = ["extra.c"], generated = ["gen/x.h"],
                      artifacts = [root * "/build/libextra.dll"])
    g = GMG.TargetGraph(root * "/src", root * "/build", [alpha, extra, util, beta, zeta])
    mkpath(root * "/build/gen")
    write(root * "/build/gen/u.h", "")

    @testset "[graph] targets name targets of the graph" begin
        mktempdir() do dir
            cfg = GMC.load_config(gm_project(dir, "[graph]\ntargets = [\"alpha\", \"zeta\"]\n"))
            @test GM.check_targets(cfg, g) === nothing
            bad = GMC.load_config(gm_project(joinpath(dir, "b"), "[graph]\ntargets = [\"alpah\"]\n"))
            e = _gm_error(() -> GM.check_targets(bad, g))
            @test occursin("[graph] targets", e) && occursin("\"alpah\"", e)
            @test occursin("library targets: alpha, beta, extra, zeta", e)
        end
    end

    @testset "what the generators are asked for" begin
        @test GMB.scope_ids(g; only = ["alpha"]) == ["util", "zeta", "beta", "alpha"]
        @test GMB.scope_ids(g; only = ["zeta"]) == ["zeta"]
        @test Set(GMB.scope_ids(g)) == Set(["alpha", "beta", "extra", "util", "zeta"])

        sub = GM.generator_graph(g, ["alpha"])
        @test [t.id for t in sub.targets] == [t.id for t in g.targets]
        x = only(filter(t -> t.id == "extra", sub.targets))
        @test !any(s -> s.generated, x.sources)              # outside the scope: asks for nothing
        @test x.artifacts == extra.artifacts                   # still names its artifact
        @test length(x.sources) == 1
        # A CMake `.rule` source stands for the output it records, once.
        @test RepliBuild.GraphGenerators.requested_sources(sub) == [root * "/build/gen/u.h"]
        @test RepliBuild.GraphGenerators.requested_sources(g) ==
              [root * "/build/gen/x.h", root * "/build/gen/u.h"]
    end

    plan = GMB.plan_graph(g, out; only = ["alpha"], windows = false)
    names = GM.module_names(plan, "Graph")

    @testset "graph module text" begin
        @test [l.name for l in plan.libraries] == ["zeta", "beta", "alpha"]
        @test [names[l.id] for l in plan.libraries] == ["Zeta", "Beta", "Alpha"]
        entries = [names[l.id] => "$(l.name)/lib/$(names[l.id]).jl" for l in plan.libraries]
        text = GM.graph_module_text("Graph", entries)
        @test text == GM.graph_module_text("Graph", entries)
        # Includes in load order, never sorted, never Dict order.
        incs = [m.captures[1] for m in eachmatch(r"^include\(\"(.*)\"\)$"m, text)]
        @test incs == ["zeta/lib/Zeta.jl", "beta/lib/Beta.jl", "alpha/lib/Alpha.jl"]
        @test occursin("\nmodule Graph\n", text) && endswith(text, "end # module Graph\n")
        @test occursin("\nexport Zeta, Beta, Alpha\n", text)
        ex = Meta.parseall(text)
        @test !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)
        mod = only(filter(a -> a isa Expr && a.head == :module, ex.args))
        @test mod.args[2] === :Graph
        @test !occursin(root, text)                            # nothing machine-specific
        @test GM.graph_module_text("Empty", Pair{String,String}[]) ==
              GM.graph_module_text("Empty", Pair{String,String}[])
    end

    @testset "one config per library" begin
        mktempdir() do dir
            cfg = GMC.load_config(gm_project(dir, "[graph]\n[link]\noptimization_level = \"3\"\n"))
            lz = only(filter(l -> l.name == "zeta", plan.libraries))
            lc = GM.library_config(cfg, lz, names)
            @test lc.graph === cfg.graph
            @test lc.wrap.module_name == "Zeta" && GMC.get_module_name(lc) == "Zeta"
            @test lc.link.optimization_level == "3"
            @test lc.wrap.language == :c
            @test GMC.get_output_path(lc) == joinpath(lz.dir, "lib")
            # What GraphBuild links with, the same function and arguments.
            ll = GMB._library_config(lz; optimization_level = "3", module_name = "Zeta", graph = cfg.graph)
            @test ll.compile.source_files == lc.compile.source_files
            @test ll.compile.include_dirs == lc.compile.include_dirs
            @test ll.link.link_libraries == lc.link.link_libraries && ll.link.link_dirs == lc.link.link_dirs
            @test ll.binary.output_name == lc.binary.output_name
            @test GMC.get_module_name(ll) == GMC.get_module_name(lc) && ll.graph === lc.graph
        end
    end

    @testset "link and metadata keys" begin
        cplan = GMB.plan_graph(GMG.TargetGraph(root * "/src", root * "/build",
                                               [gm_target(root, "ca", :shared; sources = ["ca.c"]),
                                                gm_target(root, "cb", :shared; sources = ["cb.c"], deps = ["ca"])]),
                               root * "/keys"; windows = false)
        (ca, cb) = cplan.libraries
        for lib in cplan.libraries, tu in lib.tus
            mkpath(dirname(tu.ir_file))
            write(tu.ir_file, "; IR of " * tu.source)
            RepliBuild.Compiler.write_compile_key(tu.ir_file, "fp-" * lib.name)
        end
        key(lib, deps; level = "2") = GMB._link_key(lib, deps; optimization_level = level, fallback = false)
        ka = key(ca, Dict{String,String}())
        kb = key(cb, Dict("ca" => ka))
        @test occursin(r"^[0-9a-f]{64}$", ka) && ka != kb
        @test key(ca, Dict{String,String}()) == ka                     # a function of the inputs
        @test key(ca, Dict{String,String}(); level = "3") != ka
        @test key(cb, Dict("ca" => "x" ^ 64)) != kb                    # a dependency relinked
        write(only(ca.tus).ir_file, "; IR of something else")
        @test key(ca, Dict{String,String}()) != ka                     # by content, not mtime
        write(only(ca.tus).ir_file, "; IR of " * only(ca.tus).source)
        RepliBuild.Compiler.write_compile_key(only(ca.tus).ir_file, "other flags")
        @test key(ca, Dict{String,String}()) != ka

        # A recorded link is reused only under its own key, with its files present.
        RepliBuild.Compiler.write_compile_key(only(ca.tus).ir_file, "fp-ca")
        @test GMB._cached_link(ca, ka) === nothing                    # never linked
        mkpath(dirname(ca.library))
        write(ca.library, "lib")
        linked = ca.dir * "/build/ca_linked.ll"
        mkpath(dirname(linked))
        write(linked, "")
        GMB._write_manifest(ca, ca.library, [linked])
        write(GMB.link_key_path(ca), ka)
        @test GMB._cached_link(ca, ka) == (ca.library, [linked])
        @test GMB._cached_link(ca, kb) === nothing
        rm(linked)
        @test GMB._cached_link(ca, ka) === nothing
    end

    @testset "registry: inputs and vendored files" begin
        mktempdir() do dir
            p = joinpath(dir, "proj")
            toml = gm_project(p, "[graph]\n")
            for f in ("CMakeLists.txt", "a/a.c", "a/gen.h.in", "b/z.cpp", ".git/HEAD",
                      "build/cmake/CMakeCache.txt", "julia/x.dll", ".replibuild_cache/k",
                      ".debug/mlir/x.mlir", "z.txt")
                mkpath(dirname(joinpath(p, f)))
                write(joinpath(p, f), f)
            end
            cfg = GMC.load_config(toml)
            files = GM.input_files(cfg)
            @test first.(files) == ["CMakeLists.txt", "a/a.c", "a/gen.h.in", "b/z.cpp",
                                    "replibuild.toml", "z.txt"]
            @test all(f -> isfile(f.second) && endswith(_gm_posix(f.second), f.first), files)

            # A graph is vendored by its manifest: the graph module, the
            # manifest, each library's directory. Never its IR.
            o = cfg.graph.out_dir
            libs = [Dict{String,Any}("name" => n, "module" => m, "bucket" => "c", "library" => "$n/lib/lib$n.dll",
                                     "metadata" => "$n/lib/compilation_metadata.json",
                                     "wrapper" => "$n/lib/$m.jl") for (n, m) in (("ca", "Ca"), ("Core", "RCore"))]
            mkpath(o)
            open(io -> TOML.print(io, Dict("module" => "GmUnit", "libraries" => libs)),
                 joinpath(o, GM.MANIFEST), "w")
            @test GM.vendored_paths(cfg) == ["GmUnit.jl", "graph.toml", "ca/lib", "Core/lib"]
            @test !GM.vendored_complete(o)
            for l in libs, k in ("library", "metadata", "wrapper")
                mkpath(dirname(joinpath(o, l[k])))
                write(joinpath(o, l[k]), "")
            end
            @test !GM.vendored_complete(o)                     # no graph module yet
            write(joinpath(o, "GmUnit.jl"), "")
            @test GM.vendored_complete(o)
            rm(joinpath(o, "Core/lib/RCore.jl"))
            @test !GM.vendored_complete(o)
            @test !GM.vendored_complete(dir)                    # no manifest: not a graph

            io = IOBuffer()
            GM.describe(io, cfg)
            s = String(take!(io))
            @test occursin("library: Core as GmUnit.RCore", s) && occursin("(not wrapped)", s)
            @test occursin("targets: every library", s)
        end
    end

    @testset "Windows preload of a graph library's dependencies" begin
        # zeta's DLL where beta's config says it is, so `_link_dir_dlls` finds it.
        lz = only(filter(l -> l.name == "zeta", plan.libraries))
        lb = only(filter(l -> l.name == "beta", plan.libraries))
        mkpath(dirname(lz.library))
        write(replace(lz.library, r"\.so$" => ".dll"), "")
        gc = GMC.GraphConfig(root * "/src", String[], String[], "Release", root * "/build/cmake", out)
        graph_snippet = RepliBuild.Wrapper._link_dir_preload_snippet(GMB._library_config(lb; graph = gc))
        plain_snippet = RepliBuild.Wrapper._link_dir_preload_snippet(GMB._library_config(lb))
        if Sys.iswindows()
            # Where zeta sits in the graph, relative to beta's wrapper; then the
            # build-time path. A single library keeps the sibling lookup.
            @test occursin("(\"../../zeta/lib/libzeta.dll\", ", graph_snippet)
            @test occursin("normpath(joinpath(@__DIR__, _rel))", graph_snippet)
            @test occursin("joinpath(@__DIR__, basename(_dll))", plain_snippet)
            @test !occursin("_rel", plain_snippet)
            for s in (graph_snippet, plain_snippet)
                ex = Meta.parseall(s)
                @test !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)
            end
        else
            @test isempty(graph_snippet) && isempty(plain_snippet)   # ELF has the rpath
        end
    end
end
