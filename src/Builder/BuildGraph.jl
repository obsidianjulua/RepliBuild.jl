# Builder/BuildGraph.jl — the build graph RepliBuild compiles from.
#
# A project's own build system (CMake, through its File API) has already
# resolved which targets exist, which sources each compiles, with which flags,
# defines and include path (including what each target inherits from the
# targets it depends on), what is generated during the build, and the order.
# RepliBuild compiles and links every target itself, so its DWARF and IR stay
# consistent, but it never re-derives that graph from the build files: it reads
# what the build system resolved, as it reads types from DWARF and not from a
# source AST.
#
# This module is the contract between the pieces of the multi-library build:
# the reader that fills it (`CMakeGraph`), the executor that compiles and links
# each target with RepliBuild's toolchain (`GraphBuild`), and the step that runs
# a project's build-time generators. It holds data and the one ordering rule,
# and runs no tool.
#
# Every path is absolute and `/`-separated: cmake writes `/`, Windows does not,
# and a mixed spelling of one file is two files to a Dict.

module BuildGraph

export IncludeDir, CompileGroup, GraphSource, GraphTarget, TargetGraph,
       TARGET_KINDS, topological_order, target_by_id

"""
    IncludeDir(path, is_system)

An include directory as the build system resolved it for one compile group,
including the directories inherited from the targets it depends on. `is_system`
marks a directory passed as `-isystem`.
"""
struct IncludeDir
    path::String
    is_system::Bool
end

"""
    CompileGroup(language, flags, defines, includes)

The translation units of one target that compile under the same language,
flags, defines and include path. `language` is the build system's spelling
(`"C"`, `"CXX"`). `flags` holds every other argument the build system passes
to the compiler, in order, and nothing file-specific: no input, no `-o`, no
`-c`, and no defines or include directories, which have their own fields.
`defines` are `NAME` or `NAME=VALUE`, without `-D`.
"""
struct CompileGroup
    language::String
    flags::Vector{String}
    defines::Vector{String}
    includes::Vector{IncludeDir}
end

"""
    GraphSource(path, group, generated)

A source file a target lists. `group` is an index into the target's
`compile_groups`, or 0 for a file the build system does not compile (a header
listed in the target). `generated` is true when a step of the build writes the
file, so it does not exist before that step runs.
"""
struct GraphSource
    path::String
    group::Int
    generated::Bool
end

"The `kind`s a `GraphTarget` may have."
const TARGET_KINDS = (:shared, :static, :object, :module, :executable, :utility, :interface)

"""
    GraphTarget(id, name, kind, sources, compile_groups, dependencies,
                link_libraries, artifacts)

One target. `id` is the build system's unique id for it and `name` its
user-facing name. `kind` is one of `TARGET_KINDS`; `:utility` is a target with
no compile of its own, such as a custom command that generates files.
`dependencies` are the `id`s of targets that must be built before this one.
`link_libraries` are the link inputs that are not targets of this graph
(system libraries by name, or absolute paths). `artifacts` are the files the
build system's own build would write, for reference only.
"""
struct GraphTarget
    id::String
    name::String
    kind::Symbol
    sources::Vector{GraphSource}
    compile_groups::Vector{CompileGroup}
    dependencies::Vector{String}
    link_libraries::Vector{String}
    artifacts::Vector{String}
    function GraphTarget(id, name, kind, sources, compile_groups, dependencies,
                         link_libraries, artifacts)
        kind in TARGET_KINDS ||
            throw(ArgumentError("BuildGraph: target '$name' has kind :$kind, not one of $TARGET_KINDS"))
        for s in sources
            0 <= s.group <= length(compile_groups) ||
                throw(ArgumentError("BuildGraph: target '$name' source $(s.path) names compile group " *
                                    "$(s.group) of $(length(compile_groups))"))
        end
        return new(id, name, kind, sources, compile_groups, dependencies, link_libraries, artifacts)
    end
end

"""
    TargetGraph(source_dir, build_dir, targets)

Every target of one configured project. `build_dir` is where the build system
configured it, which is where its generated files land.
"""
struct TargetGraph
    source_dir::String
    build_dir::String
    targets::Vector{GraphTarget}
end

"""
    target_by_id(g, id) -> GraphTarget

The target with that `id`; an error naming the id when there is none.
"""
function target_by_id(g::TargetGraph, id::AbstractString)::GraphTarget
    i = findfirst(t -> t.id == id, g.targets)
    i === nothing && error("BuildGraph: no target with id '$id'")
    return g.targets[i]
end

"""
    topological_order(g) -> Vector{GraphTarget}

Every target, each after the targets it depends on. Targets that become ready
together are taken in sorted `id` order, so the order is a function of the
graph and never of a Dict's iteration order. A dependency on an id the graph
does not have, two targets with one id, and a cycle are errors that name the
targets.
"""
function topological_order(g::TargetGraph)::Vector{GraphTarget}
    byid = Dict{String,GraphTarget}()
    for t in g.targets
        haskey(byid, t.id) && error("BuildGraph: two targets have id '$(t.id)' " *
                                    "('$(byid[t.id].name)' and '$(t.name)')")
        byid[t.id] = t
    end
    waiting = Dict{String,Int}(t.id => 0 for t in g.targets)
    users = Dict{String,Vector{String}}(t.id => String[] for t in g.targets)
    for t in g.targets, d in unique(t.dependencies)
        haskey(byid, d) ||
            error("BuildGraph: target '$(t.name)' depends on '$d', which is not in the graph")
        waiting[t.id] += 1
        push!(users[d], t.id)
    end
    ready = sort!([id for (id, n) in waiting if n == 0])
    order = GraphTarget[]
    while !isempty(ready)
        id = popfirst!(ready)
        push!(order, byid[id])
        for u in users[id]
            waiting[u] -= 1
            waiting[u] == 0 && push!(ready, u)
        end
        sort!(ready)
    end
    if length(order) != length(g.targets)
        stuck = sort!([byid[id].name for (id, n) in waiting if n > 0])
        error("BuildGraph: dependency cycle among targets: $(join(stuck, ", "))")
    end
    return order
end

end # module BuildGraph
