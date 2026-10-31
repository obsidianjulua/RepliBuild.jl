# Builder/GraphMode.jl — graph mode end to end: a replibuild.toml with a
# [graph] section names a CMake project of several libraries, and `build`,
# `wrap`, the registry, `clean` and `info` act on all of them.
#
# The pieces already exist as functions: `CMakeGraph` reads the build graph
# CMake resolved at configure time, `GraphGenerators` runs the build-time steps
# that write its generated sources, and `GraphBuild` compiles and links every
# library with RepliBuild's toolchain. This module chains them and adds, for
# every library, the steps `Compiler.compile_project` runs for one: the
# `nounwind` and `byval` facts off the linked IR, `compilation_metadata.json`
# beside the library, and `verify_wrap_surface`. Each library goes through all
# of it under one config, `GraphBuild._library_config`, the config it was linked
# with.
#
# Layout (`[paths]` relative to the project root):
#
#   <build>/cmake/                  the CMake build tree (configure only)
#   <output>/<target>/lib/          one library, its metadata and its wrapper
#   <output>/<target>/{ir,build}/   its IR and link intermediates
#   <output>/<Graph>.jl             the graph module, including every wrapper
#   <output>/graph.toml             the graph manifest: libraries, load order
#
# `wrap` writes one wrapper per library, of the shape a single library gets
# (its own LIBRARY_PATH, LIBRARY_SHA, __init__, guards), in topological order,
# then the graph module `module <Graph>` that includes them in that order. Each
# __init__ opens its own library, so a library's dependencies are open before
# it is, which is the load order PE needs (Windows binds an import to a DLL
# already loaded under that name). Types are still emitted per library: a type
# two libraries share is two Julia types until cross-library type identity
# (docs/updates/CROSS_LIBRARY_TYPES.md) is built.
#
# Caching: GraphBuild skips the compile of an unchanged translation unit and
# the link of a library whose link key is unchanged (`GraphBuild._link_key`),
# and this module skips metadata extraction for a library whose link key is the
# one its metadata was extracted from. Every build still configures, runs the
# generator steps and checks every unit and link key.

module GraphMode

using TOML

import ..ConfigurationManager
import ..ConfigurationManager: RepliBuildConfig, GraphConfig, module_name_for, get_module_name,
                               get_build_path, get_output_path, get_cache_path
import ..BuildGraph
import ..CMakeGraph
import ..GraphGenerators
import ..GraphBuild
import ..GraphBuild: GraphPlan, LibraryPlan
import ..Compiler

export build_graph_project, wrap_graph_project, library_module_names, graph_module_text,
       input_files, vendored_paths, vendored_complete, describe

"The graph manifest's file name, in the graph's output directory."
const MANIFEST = "graph.toml"

# ============================================================================
# MODULE NAMES
# ============================================================================

"""
Module names every generated wrapper refers to by name (`Libdl.dlopen`,
`RepliBuild.SHA`). Inside a module of that name, the name means the module
itself, so these are unusable on top of everything bound in `Base` and `Core`.
"""
const WRAPPER_QUALIFIERS = ("Libdl", "RepliBuild")

"""
    unusable_module_name(name) -> Bool

Whether a module named `name` cannot be a library's module in a graph module:
it does not parse as that one identifier (a keyword, a digit first), or it is
bound in `Base` or `Core`, or it is one of `WRAPPER_QUALIFIERS`.

A bound name is unusable twice over. Inside the library's own module the name
is the module, so a wrapper's `Base.error(…)` (every generated diagnostic is
written that way) or `Core.…` would look in the library. And in the graph
module a submodule named `Core` is unreachable: `Graph.Core` is Julia's `Core`.

`Base` and `Core` are asked on the running Julia, so a name only a newer
release binds is renamed on that release alone.
"""
function unusable_module_name(name::AbstractString)::Bool
    Meta.parse(String(name); raise = false) === Symbol(name) || return true
    s = Symbol(name)
    return isdefined(Base, s) || isdefined(Core, s) || name in WRAPPER_QUALIFIERS
end

"""
    library_module_names(target_names, graph_module) -> Vector{String}

The module name of each library of a graph, in the order given (topological
order, so the answer is a function of the graph). One rule, applied in order:

1. `ConfigurationManager.module_name_for(target_name)`, the derivation a single
   library's module name has (CamelCase; a leading digit gets `_`).
2. An unusable name (`unusable_module_name`: bound in `Base` or `Core`, among
   `WRAPPER_QUALIFIERS`) gets an `R` in front: `Core` → `RCore`.
3. A name equal, ignoring case, to the graph module's or to an earlier
   library's gets `_2`, `_3`, … until it is neither. Case is ignored on every
   host, so the names do not depend on which file system wrapped the graph. A
   submodule named like its graph module would be unreachable: in `module X`,
   `X` is `X` itself.
"""
function library_module_names(target_names::AbstractVector{<:AbstractString},
                              graph_module::AbstractString)::Vector{String}
    taken = Set{String}([lowercase(graph_module)])
    out = String[]
    for t in target_names
        base = module_name_for(t)
        unusable_module_name(base) && (base = "R" * base)
        name = base
        k = 1
        while lowercase(name) in taken || unusable_module_name(name)
            k += 1
            name = string(base, "_", k)
        end
        push!(taken, lowercase(name))
        push!(out, name)
    end
    return out
end

"""
    graph_module_name(config) -> String

The graph module's name: `ConfigurationManager.get_module_name`, the one
derivation every reader of the wrapper file uses (`[wrap] module_name`, else
CamelCase of `[project] name`). It is not renamed, since the registry and a
scaffolded package find the file by that name: an unusable one is an error
that says to set `[wrap] module_name`.
"""
function graph_module_name(config::RepliBuildConfig)::String
    name = get_module_name(config)
    unusable_module_name(name) &&
        error("graph mode: the graph module would be named $name, which Julia binds already " *
              "(or which is not an identifier). Set [wrap] module_name in $(config.config_file).")
    return name
end

"The `GraphBuild` id => module name table for `plan`'s libraries."
function module_names(plan::GraphPlan, graph_module::AbstractString)::Dict{String,String}
    names = library_module_names([lib.name for lib in plan.libraries], graph_module)
    return Dict(lib.id => n for (lib, n) in zip(plan.libraries, names))
end

# ============================================================================
# PLAN AND CONFIG
# ============================================================================

"`[graph] targets` as `GraphBuild`'s `only`: `nothing` (every library) when empty."
_only(gc::GraphConfig) = isempty(gc.targets) ? nothing : gc.targets

"""
    check_targets(config, g)

Every `[graph] targets` entry names a target of `g` (by name or File API id);
an error that names the key and lists the library targets otherwise.
"""
function check_targets(config::RepliBuildConfig, g::BuildGraph.TargetGraph)
    known = Set{String}()
    for t in g.targets
        push!(known, t.id, t.name)
    end
    unknown = [x for x in config.graph.targets if !(x in known)]
    isempty(unknown) && return nothing
    libs = sort!([t.name for t in g.targets if t.kind in (:shared, :module, :static)])
    error("[graph] targets in $(config.config_file) names $(join(repr.(unknown), ", ")), " *
          "not a target of $(config.graph.source). Its library targets: $(join(libs, ", ")).")
end

"The plan for `g` under the project's `[graph]` section."
graph_plan(config::RepliBuildConfig, g::BuildGraph.TargetGraph)::GraphPlan =
    GraphBuild.plan_graph(g, config.graph.out_dir; only = _only(config.graph))

"""
    library_config(config, lib, names) -> RepliBuildConfig

The one config `lib` is linked, extracted and wrapped under:
`GraphBuild._library_config` with the project's `[link] optimization_level` and
`fallback`, the library's module name, and the project's `[graph]` section.
"""
function library_config(config::RepliBuildConfig, lib::LibraryPlan,
                        names::AbstractDict)::RepliBuildConfig
    return GraphBuild._library_config(lib; optimization_level = config.link.optimization_level,
                                      fallback = config.link.fallback,
                                      module_name = names[lib.id], graph = config.graph)
end

"""
    configure(config) -> BuildGraph.TargetGraph

Configure `[graph] source` into `[graph]`'s cmake directory with a File API
query (`CMakeGraph.cmake_graph`, the Ninja generator), and read the graph.
"""
function configure(config::RepliBuildConfig)::BuildGraph.TargetGraph
    gc = config.graph
    isfile(gc.source * "/CMakeLists.txt") ||
        error("[graph] source = $(gc.source) has no CMakeLists.txt: graph mode builds a CMake project.")
    mkpath(gc.cmake_dir)
    return CMakeGraph.cmake_graph(gc.source; build_dir = gc.cmake_dir, options = gc.options,
                                  build_type = gc.build_type)
end

"""
    generator_graph(g, only) -> BuildGraph.TargetGraph

`g` as the selected build asks the generators for it: a target outside the
build's scope (`GraphBuild.scope_ids`) keeps its artifacts, by which the
generator step planner names whose link a step would need, and lists no
generated source. Generating only what the selected libraries read keeps an
unselected target's steps (a dictionary whose tool RepliBuild refuses, say)
out of the build.
"""
function generator_graph(g::BuildGraph.TargetGraph, only)::BuildGraph.TargetGraph
    keep = Set(GraphBuild.scope_ids(g; only))
    targets = map(g.targets) do t
        t.id in keep && return t
        BuildGraph.GraphTarget(t.id, t.name, t.kind, [s for s in t.sources if !s.generated],
                               t.compile_groups, t.dependencies, t.link_libraries, t.artifacts)
    end
    return BuildGraph.TargetGraph(g.source_dir, g.build_dir, targets)
end

"""
    configured_plan(config) -> (GraphPlan, Dict{String,String})

The plan and module names of the graph the last `build` configured, read from
the CMake reply without configuring again. An error that says to build first
when there is no reply.
"""
function configured_plan(config::RepliBuildConfig)
    gc = config.graph
    g = try
        CMakeGraph.read_cmake_graph(gc.cmake_dir)
    catch e
        error("graph mode: no configured graph in $(gc.cmake_dir) " *
              "($(sprint(showerror, e))). Run RepliBuild.build($(repr(config.config_file))) first.")
    end
    plan = graph_plan(config, g)
    return (plan, module_names(plan, graph_module_name(config)))
end

# ============================================================================
# BUILD
# ============================================================================

"Where a library's metadata key is kept: the link key its metadata was extracted from."
_metadata_key_path(lib::LibraryPlan) = lib.dir * "/metadata.key"

"The metadata file `Compiler.save_compilation_metadata` writes beside `lib`'s library."
_metadata_path(lib::LibraryPlan) = dirname(lib.library) * "/compilation_metadata.json"

"""
    build_graph_project(config; force = false) -> String

`RepliBuild.build` for a `[graph]` project. Configure (`configure`), run the
build-time generators (`GraphGenerators.run_generators`), compile and link the
selected libraries and the shared libraries they link (`GraphBuild.build_graph`),
then, for each library, what `Compiler.compile_project` does after its link:
`nounwind` and `byval` off the linked IR, `compilation_metadata.json` beside the
library (`save_compilation_metadata`) and `verify_wrap_surface`. Writes the
graph manifest (`graph.toml`) and returns the output directory.

A library whose link key is the one its metadata was extracted from keeps that
metadata. `force` recompiles, relinks and re-extracts everything.
"""
function build_graph_project(config::RepliBuildConfig; force::Bool = false)::String
    gc = config.graph
    graph_module = graph_module_name(config)   # refused before anything runs
    println("RepliBuild | $(config.project.name) [graph]")
    start = time()

    g = configure(config)
    check_targets(config, g)
    run = GraphGenerators.run_generators(generator_graph(g, _only(gc)))
    isempty(run.ran) || println("  generators: $(length(run.ran)) step(s) ran")
    plan = graph_plan(config, g)
    names = module_names(plan, graph_module)
    results = GraphBuild.build_graph(plan; optimization_level = config.link.optimization_level,
                                     fallback = config.link.fallback, force,
                                     module_names = names, graph = gc)
    byid = Dict(r.id => r for r in results)

    extracted = 0
    for lib in plan.libraries
        r = byid[lib.id]
        key = strip(read(GraphBuild.link_key_path(lib), String))
        mkey = _metadata_key_path(lib)
        if !force && isfile(_metadata_path(lib)) && isfile(mkey) && strip(read(mkey, String)) == key
            continue
        end
        # Gone before extraction, so a failed one leaves nothing claiming success.
        rm(mkey; force = true)
        cfg = library_config(config, lib, names)
        println("  metadata: $(lib.name)")
        nounwind = Compiler._nounwind_definitions(r.linked_ir)
        byval = Compiler._byval_definitions(r.linked_ir)
        metadata = Compiler.save_compilation_metadata(cfg, r.sources, r.library;
                                                      nounwind = nounwind, byval = byval)
        Compiler.verify_wrap_surface(cfg, r.library, metadata)
        write(mkey, key)
        extracted += 1
    end
    n = length(plan.libraries)
    println(extracted == 0 ? "  metadata: $n cached" :
            "  metadata: $extracted/$n extracted ($(n - extracted) cached)")

    write_manifest(config, plan, names, graph_module)
    println("  done: $(round(time() - start, digits = 2))s")
    return gc.out_dir
end

"`p` relative to the graph's output directory, `/`-separated."
_rel(gc::GraphConfig, p::AbstractString) = replace(relpath(p, gc.out_dir), '\\' => '/')

"The wrapper file `wrap` writes for `lib`: `Wrapper.wrap_library` writes `<output>/<module>.jl`."
_wrapper_path(lib::LibraryPlan, names::AbstractDict) = dirname(lib.library) * "/" * names[lib.id] * ".jl"

"""
    write_manifest(config, plan, names, graph_module)

`<output>/graph.toml`: the graph module, and every library in load order with
its module, kind, bucket, directory, library, metadata and wrapper (paths
relative to the output directory) and the libraries it links. The registry
vendors the graph by it and `info` reads it. Keys sorted, libraries in
topological order, so the file is a function of the build.
"""
function write_manifest(config::RepliBuildConfig, plan::GraphPlan, names::AbstractDict,
                        graph_module::AbstractString)
    gc = config.graph
    byid = Dict(lib.id => lib for lib in plan.libraries)
    libs = Dict{String,Any}[]
    for lib in plan.libraries
        push!(libs, Dict{String,Any}(
            "id" => lib.id, "name" => lib.name, "module" => names[lib.id],
            "kind" => string(lib.kind), "bucket" => string(lib.bucket),
            "dir" => _rel(gc, lib.dir), "library" => _rel(gc, lib.library),
            "metadata" => _rel(gc, _metadata_path(lib)),
            "wrapper" => _rel(gc, _wrapper_path(lib, names)),
            "link_dependencies" => [byid[d].name for d in lib.link_dependencies]))
    end
    data = Dict{String,Any}("module" => String(graph_module), "source" => gc.source,
                            "targets" => gc.targets, "libraries" => libs)
    mkpath(gc.out_dir)
    open(gc.out_dir * "/" * MANIFEST, "w") do io
        TOML.print(io, data; sorted = true)
    end
    return nothing
end

# ============================================================================
# WRAP
# ============================================================================

"""
    graph_module_text(graph_module, entries) -> String

The graph module: `module <graph_module>`, one `include` per library wrapper in
`entries` (`module name => path relative to the graph module`, in load order),
and an `export` of the library modules. Nothing in it depends on when or where
it was written.
"""
function graph_module_text(graph_module::AbstractString,
                           entries::AbstractVector{<:Pair{<:AbstractString,<:AbstractString}})::String
    io = IOBuffer()
    print(io, """
        # Written by RepliBuild.wrap (graph mode). Do not edit: wrap writes it again.
        #
        # One module per library of the build graph, included in load order, each
        # library after the libraries it links. Each library's __init__ opens its
        # library, so a library's dependencies are open before it is: Windows binds
        # a DLL's imports to the modules already loaded under those names.
        module $graph_module

        """)
    for (_, path) in entries
        println(io, "include(", repr(String(path)), ")")
    end
    isempty(entries) || print(io, "\nexport ", join(first.(entries), ", "), "\n")
    print(io, "\nend # module ", graph_module, "\n")
    return String(take!(io))
end

"""
    wrap_graph_project(config) -> String

`RepliBuild.wrap` for a `[graph]` project: wrap every library the last build
built, in topological order, each under its own config (`library_config`) into
`<output>/<target>/lib/<Module>.jl`, then write the graph module
`<output>/<Graph>.jl` that includes them in that order. Returns its path.
"""
function wrap_graph_project(config::RepliBuildConfig)::String
    gc = config.graph
    (plan, names) = configured_plan(config)
    graph_module = graph_module_name(config)
    missing_builds = [lib.name for lib in plan.libraries
                      if !(isfile(lib.library) && isfile(_metadata_path(lib)))]
    isempty(missing_builds) ||
        error("graph mode: not built yet: $(join(missing_builds, ", ")). " *
              "Run RepliBuild.build($(repr(config.config_file))) first.")
    # Wrapper loads after Builder; reached when a wrap runs, as Discovery reaches `build`.
    W = getfield(parentmodule(@__MODULE__), :Wrapper)
    graph_file = gc.out_dir * "/" * graph_module * ".jl"
    entries = Pair{String,String}[]
    for lib in plan.libraries
        cfg = library_config(config, lib, names)
        path = W.wrap_library(cfg, lib.library; generate_docs = true)
        push!(entries, names[lib.id] => replace(relpath(path, dirname(graph_file)), '\\' => '/'))
    end
    text = graph_module_text(graph_module, entries)
    parsed = Meta.parseall(text)
    any(a -> a isa Expr && a.head in (:incomplete, :error), parsed.args) &&
        error("graph mode: the graph module for $graph_module does not parse:\n$text")
    write(graph_file, text)
    println("  wrap: $(graph_module).jl ($(length(entries)) librar$(length(entries) == 1 ? "y" : "ies"))")
    return graph_file
end

# ============================================================================
# REGISTRY
# ============================================================================

_canon(p::AbstractString) = Compiler._canon_path(ConfigurationManager._graph_path(p))

function _walk!(out::Vector{Pair{String,String}}, dir::String, rel::String, skip::Set{String})
    for name in readdir(dir)   # sorted
        p = dir * "/" * name
        if isdir(p)
            (islink(p) || name == ".git" || _canon(p) in skip) && continue
            _walk!(out, p, rel * name * "/", skip)
        elseif isfile(p)
            push!(out, (rel * name) => p)
        end
    end
    return out
end

"""
    input_files(config) -> Vector{Pair{String,String}}

Every file a build of the graph can read from its source tree, as
`path relative to [graph] source => absolute path`, in sorted order: the whole
tree under `[graph] source` except `.git`, symlinked directories, and the
project's build, output, cache and `.debug` directories when they lie inside it.

The registry's `hash_config` feeds these for a graph. The whole tree, because
a graph's inputs are not only its compiled sources and include directories: the
CMake files decide every flag, and a generator step reads templates and scripts
no target lists. Paths are relative, so a moved project keeps its hash.
"""
function input_files(config::RepliBuildConfig)::Vector{Pair{String,String}}
    gc = config.graph
    out = Pair{String,String}[]
    isdir(gc.source) || return out
    skip = Set{String}(_canon(p) for p in (get_build_path(config), get_output_path(config),
                                            get_cache_path(config),
                                            joinpath(config.project.root, ".debug")))
    return _walk!(out, gc.source, "", skip)
end

"The graph manifest under `dir`, or `nothing`."
function _manifest(dir::AbstractString)
    p = joinpath(dir, MANIFEST)
    return isfile(p) ? TOML.parsefile(p) : nothing
end

"""
    vendored_paths(config) -> Vector{String}

What the registry copies to vendor a built and wrapped graph, relative to the
output directory: the graph module, the manifest, and each library's directory
holding its library, metadata and wrapper. Not the IR or link intermediates.
An error when the graph is not built.
"""
function vendored_paths(config::RepliBuildConfig)::Vector{String}
    gc = config.graph
    m = _manifest(gc.out_dir)
    m === nothing && error("graph mode: $(gc.out_dir) has no $MANIFEST: the graph is not built")
    out = String[m["module"] * ".jl", MANIFEST]
    for lib in m["libraries"]
        push!(out, dirname(lib["library"]))
    end
    return unique(out)
end

"""
    vendored_complete(dir) -> Bool

Whether `dir` holds a whole vendored graph: its manifest, its graph module, and
each library with its metadata and wrapper. `false` for anything else, a
single library's vendored build included.
"""
function vendored_complete(dir::AbstractString)::Bool
    m = _manifest(dir)
    m === nothing && return false
    isfile(joinpath(dir, m["module"] * ".jl")) || return false
    return all(lib -> all(k -> isfile(joinpath(dir, lib[k])), ("library", "metadata", "wrapper")),
               m["libraries"])
end

# ============================================================================
# INFO
# ============================================================================

"""
    describe(io, config)

`RepliBuild.info`'s lines for a graph project: the CMake project, the selected
targets, and, once built, each library with its module and whether its wrapper
exists, then the graph module.
"""
function describe(io::IO, config::RepliBuildConfig)
    gc = config.graph
    println(io, "  graph:   CMake project $(gc.source)")
    println(io, "  targets: ", isempty(gc.targets) ? "every library" : join(gc.targets, ", "))
    m = _manifest(gc.out_dir)
    if m === nothing
        println(io, "  not built yet")
        return
    end
    for lib in m["libraries"]
        wrapped = isfile(joinpath(gc.out_dir, lib["wrapper"])) ? "" : " (not wrapped)"
        println(io, "  library: $(lib["name"]) as $(m["module"]).$(lib["module"]) " *
                    "[$(lib["bucket"])] $(lib["library"])$wrapped")
    end
    module_file = m["module"] * ".jl"
    println(io, "  module:  ", isfile(joinpath(gc.out_dir, module_file)) ? module_file : "not wrapped yet")
    return nothing
end

end # module GraphMode
