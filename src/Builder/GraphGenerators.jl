# Builder/GraphGenerators.jl — make a graph's generated sources exist.
#
# A target may list sources that a step of the build writes: a dictionary
# from rootcling, a moc file, protoc output, a header an awk script prints.
# Those steps are tools, not compiles RepliBuild has to own, so they are left
# to the project's own build system. This module asks it to run only the steps
# that write the sources the graph marks `generated`, and never a compile or a
# link: every library the graph names is compiled by RepliBuild, from the same
# toolchain as its DWARF.
#
# It reads the build graph through ninja's own tools (`-t targets`, `-t query`,
# `-t commands`) against the build.ninja cmake wrote, never by parsing it.
#
# Why not `ninja <output>`, or `cmake --build --target <utility>`: cmake's
# Ninja generator gives every custom command of a target an ORDER-ONLY input on
# the full artifact of every target that target depends on (it takes a
# target's dependencies as a superset of what each of its commands needs, and
# orders the commands after all of them). A library `gg` that links `base` and
# lists `gen.h` gets
#
#     build gen.h gen.c: CUSTOM_COMMAND gen.h.in gen.c.in || libbase.dll
#
# and `ninja gen.h` compiles and links `base` first (cmake 4.4, ninja 1.13). A
# utility target orders after its dependencies' artifacts in the same way. So
# the steps run from a narrowed manifest instead: the same commands, run by the
# same ninja from the same directory, with what each step declares it reads,
# and without cmake's order-only inputs on library artifacts. What a step does
# need from a compile or a link is refused before anything runs
# (`GeneratorNeedsCompile`, naming the target):
#
#   - a file a step declares (explicit or implicit input) that a compile or a
#     link writes (`:input`);
#   - an executable, which is how cmake records the tool a COMMAND runs: as an
#     order-only input, like the library ordering (`:tool`; rootcling);
#   - a library the step's command names by path (`:named`).
#
# Every other order-only input on an artifact is ordering, and is dropped; the
# plan lists the targets it came from (`dropped`).
#
# Every step runs on every call: the narrowed manifest keeps its own build log
# under `CMakeFiles/replibuild_generators/`, which is deleted first, because it
# does not carry the original steps' depfiles and its up-to-date answer could be
# stale. The log is ninja's record of what ran, and `GeneratorRun.ran` is read
# from it.
#
# Only a build tree configured with `-G Ninja` is supported.

module GraphGenerators

using ..BuildGraph
import ..BuildBridge

export NinjaEdge, GeneratorStep, NeededTarget, GeneratorPlan, GeneratorRun,
       GeneratorNeedsCompile, GeneratedSourceUnwritten, GeneratorStepFailed,
       requested_sources, plan_generators, generator_manifest, run_generators

# ============================================================================
# TYPES
# ============================================================================

"""
    NinjaEdge(rule, explicit, implicit, order_only)

The build edge that writes a node, as `ninja -t query` reports it: its rule
and its inputs in ninja's three kinds, spelled as ninja spells them.
"""
struct NinjaEdge
    rule::String
    explicit::Vector{String}
    implicit::Vector{String}
    order_only::Vector{String}
end

Base.:(==)(a::NinjaEdge, b::NinjaEdge) =
    a.rule == b.rule && a.explicit == b.explicit && a.implicit == b.implicit &&
    a.order_only == b.order_only
Base.hash(e::NinjaEdge, h::UInt) = hash((e.rule, e.explicit, e.implicit, e.order_only), h)

"""
    GeneratorStep(outputs, command, inputs, after)

One step of the project's build that writes generated sources. `outputs` are
the files the graph's walk reached through it (the command may write more),
`inputs` the files and other steps' outputs it declares it reads, and `after`
the other steps' outputs cmake orders it after. Paths are absolute and
`/`-separated. `command` is the build's own command line, verbatim.
"""
struct GeneratorStep
    outputs::Vector{String}
    command::String
    inputs::Vector{String}
    after::Vector{String}
end

"""
    NeededTarget(target, artifact, why, needed_by)

A compile or link that a generated source needs before its step can run.
`target` is the graph target that owns `artifact` (or ninja's rule name when no
target does). `why` is `:input` (a step declares the file), `:tool` (an
executable, which is how cmake records the tool a command runs) or `:named` (the
step's command names the library). `needed_by` are the generated sources whose
steps need it.
"""
struct NeededTarget
    target::String
    artifact::String
    why::Symbol
    needed_by::Vector{String}
end

"""
    GeneratorPlan

What `run_generators` would run, read off the build tree without running it.

- `build_dir` — the configured build tree.
- `requested` — every source the graph marks `generated`, in target order (each
  target after its dependencies) and then in the order the target lists them.
- `existing` — requested files no step writes that already exist (written at
  configure time).
- `unwritten` — requested files no step writes and that do not exist.
- `steps` — the steps that write the requested files, each after what it
  needs or is ordered after.
- `needs` — compiles and links the steps would need: the refusal.
- `dropped` — targets whose artifacts were order-only inputs of a step and were
  left out as ordering, sorted.
"""
struct GeneratorPlan
    build_dir::String
    requested::Vector{String}
    existing::Vector{String}
    unwritten::Vector{String}
    steps::Vector{GeneratorStep}
    needs::Vector{NeededTarget}
    dropped::Vector{String}
end

"""
    GeneratorRun(plan, ran, manifest, output)

The result of `run_generators`: the plan, the steps ninja's log records as run,
the narrowed manifest it ran (empty when there was nothing to run), and ninja's
output.
"""
struct GeneratorRun
    plan::GeneratorPlan
    ran::Vector{GeneratorStep}
    manifest::String
    output::String
end

"""
    GeneratorNeedsCompile(build_dir, needs)

Producing a generated source needs one of the project's own compiles or links
(`needs`, each naming its target). Nothing ran.
"""
struct GeneratorNeedsCompile <: Exception
    build_dir::String
    needs::Vector{NeededTarget}
end

function Base.showerror(io::IO, e::GeneratorNeedsCompile)
    print(io, "GeneratorNeedsCompile: producing the generated sources of $(e.build_dir) ",
          "would compile or link targets of the project, which RepliBuild compiles itself. ",
          "Nothing ran.")
    for n in e.needs
        why = n.why === :input ? "a step declares it as an input" :
              n.why === :tool  ? "an executable a step's target depends on (the tool its command runs)" :
                                 "a step's command names it"
        print(io, "\n  target '", n.target, "' (", n.artifact, "): ", why, "; needed by ",
              join(n.needed_by, ", "))
    end
end

"""
    GeneratedSourceUnwritten(build_dir, paths)

The graph marks these sources generated, but no step of the build writes them
and they do not exist.
"""
struct GeneratedSourceUnwritten <: Exception
    build_dir::String
    paths::Vector{String}
end

Base.showerror(io::IO, e::GeneratedSourceUnwritten) =
    print(io, "GeneratedSourceUnwritten: no step of the build in $(e.build_dir) writes ",
          join(e.paths, ", "), ", and ", length(e.paths) == 1 ? "it does" : "they do",
          " not exist. Was the tree configured from the same project, with the same options?")

"""
    GeneratorStepFailed(manifest, exitcode, output)

ninja ran the narrowed manifest and a step failed.
"""
struct GeneratorStepFailed <: Exception
    manifest::String
    exitcode::Int
    output::String
end

function Base.showerror(io::IO, e::GeneratorStepFailed)
    tail = join(last(split(e.output, '\n'), 40), '\n')
    how = e.exitcode == BuildBridge.SPAWN_FAILED ? "ninja did not start" : "exit $(e.exitcode)"
    print(io, "GeneratorStepFailed: a generator step failed ($how), running ", e.manifest,
          ":\n", tail)
end

# ============================================================================
# PATHS
# ============================================================================

# Every path inside the module is '/'-separated: cmake and ninja write '/', and
# Windows does not.
_posix(p::AbstractString) = Sys.iswindows() ? replace(String(p), '\\' => '/') : String(p)

_dir(p::AbstractString) = String(rstrip(_posix(p), '/'))

# A node ninja spells relative to the build tree, or absolute, as one absolute
# '/' path.
function _abs(p::AbstractString, build_dir::String)::String
    q = _posix(p)
    isabspath(q) || (q = build_dir * "/" * q)
    return _posix(normpath(q))
end

# The key two spellings of one file share. Windows folds case.
_key(p::AbstractString) = Sys.iswindows() ? lowercase(String(p)) : String(p)

# How the narrowed manifest spells a file: relative to the build tree when it
# is inside it, else absolute. One spelling per file, because ninja keys nodes
# by text: `gen.h` and `<build>/gen.h` would be two files to it.
function _spell(p::String, build_dir::String)::String
    startswith(_key(p), _key(build_dir) * "/") || return p
    return String(chop(p; head = length(build_dir) + 1, tail = 0))
end

# ============================================================================
# WHAT THE GRAPH ASKS FOR
# ============================================================================

"""
    requested_sources(g) -> Vector{String}

Every source `g` marks `generated`, once each: targets in `topological_order`,
then each target's sources in its own order.

A `.rule` source a target does not compile is CMake's record of a custom
command (the File API's "CMake Rules" source group, marked generated), which no
step of a Ninja build writes. It stands for the command's output, the same path
without `.rule`, and that is requested instead: for a header a custom target
DEPENDS on, the `.rule` path is the output's only spelling in the graph.
"""
function requested_sources(g::TargetGraph)::Vector{String}
    seen = Set{String}()
    out = String[]
    for t in topological_order(g), s in t.sources
        s.generated || continue
        p = _posix(normpath(_posix(s.path)))
        (s.group == 0 && endswith(p, ".rule")) && (p = p[1:end-ncodeunits(".rule")])
        _key(p) in seen && continue
        push!(seen, _key(p))
        push!(out, p)
    end
    return out
end

# ============================================================================
# READING NINJA'S GRAPH
# ============================================================================

"""
Where the planner reads a build's ninja graph from. A source answers three
questions, with the semantics of ninja's tools:

- `ninja_outputs(src)` — every node some edge writes, with its rule
  (`ninja -t targets all`);
- `ninja_query(src, nodes)` — the edge that writes each node, `nothing` for a
  node no edge writes (`ninja -t query`);
- `ninja_commands(src, nodes)` — the command of each distinct edge that writes
  one of `nodes`, in the order the nodes first name it (`ninja -t commands -s`).
"""
abstract type NinjaSource end

"The ninja of a configured build tree."
struct LiveNinja <: NinjaSource
    ninja::String
    build_dir::String
end

function ninja_outputs end
function ninja_query end
function ninja_commands end

# A Windows command line holds 32,767 characters; node lists go in chunks.
function _chunks(nodes::Vector{String}; limit::Int = 20_000)
    out = Vector{String}[]
    cur = String[]
    n = 0
    for x in nodes
        if !isempty(cur) && n + ncodeunits(x) + 1 > limit
            push!(out, cur)
            cur = String[]
            n = 0
        end
        push!(cur, x)
        n += ncodeunits(x) + 1
    end
    isempty(cur) || push!(out, cur)
    return out
end

# A ninja tool's stdout. Tools print nothing else there (no "Entering
# directory"), and warnings go to stderr.
function _ninja_tool(src::LiveNinja, args::Vector{String})::String
    cmd = Cmd(`$(src.ninja) -t $args`; dir = src.build_dir)
    out = IOBuffer()
    err = IOBuffer()
    p = try
        run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
    catch e
        e isa Base.IOError || rethrow()
        error("GraphGenerators: could not start ninja ($(src.ninja)): $(e.msg)")
    end
    p.exitcode == 0 ||
        error("GraphGenerators: `ninja -t $(first(args))` failed in $(src.build_dir) ",
              "(exit $(p.exitcode)):\n", String(take!(err)))
    return String(take!(out))
end

# Windows' C runtime writes "\r\n" to a text-mode stdout.
_lines(text::AbstractString) = [String(rstrip(l, '\r')) for l in split(text, '\n')]

"""
    parse_targets(text) -> Vector{Pair{String,String}}

`ninja -t targets all`: one `node: rule` per line.
"""
function parse_targets(text::AbstractString)::Vector{Pair{String,String}}
    out = Pair{String,String}[]
    for l in _lines(text)
        isempty(l) && continue
        r = findlast(": ", l)
        r === nothing && continue
        push!(out, l[1:prevind(l, first(r))] => l[nextind(l, last(r)):end])
    end
    return out
end

"""
    parse_query(text) -> Dict{String,Union{Nothing,NinjaEdge}}

`ninja -t query`: per node a `node:` line, then `  input: <rule>` and the
edge's inputs (`| ` implicit, `|| ` order-only) indented four, then other
sections (`  validations:`, `  outputs:`, …) this reads past. A node with no
`input:` line has no edge.
"""
function parse_query(text::AbstractString)::Dict{String,Union{Nothing,NinjaEdge}}
    out = Dict{String,Union{Nothing,NinjaEdge}}()
    node = nothing
    edge = nothing
    in_inputs = false
    flush() = node === nothing || (out[node] = edge)
    for l in _lines(text)
        isempty(l) && continue
        if !startswith(l, ' ') && endswith(l, ':')
            flush()
            node = String(chop(l))
            edge = nothing
            in_inputs = false
        elseif startswith(l, "  input: ")
            edge = NinjaEdge(String(chop(l; head = 9, tail = 0)), String[], String[], String[])
            in_inputs = true
        elseif startswith(l, "    ")
            in_inputs || continue
            s = String(chop(l; head = 4, tail = 0))
            if startswith(s, "|| ")
                push!(edge.order_only, String(chop(s; head = 3, tail = 0)))
            elseif startswith(s, "| ")
                push!(edge.implicit, String(chop(s; head = 2, tail = 0)))
            else
                push!(edge.explicit, s)
            end
        elseif startswith(l, "  ")
            in_inputs = false
        end
    end
    flush()
    return out
end

ninja_outputs(src::LiveNinja) = parse_targets(_ninja_tool(src, ["targets", "all"]))

function ninja_query(src::LiveNinja, nodes::Vector{String})
    out = Dict{String,Union{Nothing,NinjaEdge}}()
    for c in _chunks(nodes)
        merge!(out, parse_query(_ninja_tool(src, vcat("query", c))))
    end
    return out
end

ninja_commands(src::LiveNinja, nodes::Vector{String}) =
    filter!(!isempty, _lines(_ninja_tool(src, vcat("commands", "-s", nodes))))

# ============================================================================
# PLANNING
# ============================================================================

const _CUSTOM = "CUSTOM_COMMAND"

_is_phony(e) = e !== nothing && e.rule == "phony"
_is_custom(e) = e !== nothing && e.rule == _CUSTOM
# Written by a compile, a link, or any rule that is not a custom command.
_is_built(e) = e !== nothing && !_is_phony(e) && !_is_custom(e)

# The command of each custom node, from `ninja -t commands -s`, which prints
# each distinct edge once. Nodes whose edges differ in inputs are different
# edges, so a batch of them maps 1:1 to its lines. Nodes with the same inputs
# may still be different edges (two commands of one target with no DEPENDS
# read the same order-only inputs): a batch of them answers with one line (one
# edge) or one per node (all distinct); anything between is asked node by node.
function _commands(src::NinjaSource, nodes::Vector{String}, edges)::Dict{String,String}
    out = Dict{String,String}()
    groups = Dict{NinjaEdge,Vector{String}}()
    order = NinjaEdge[]
    for n in nodes
        e = edges[n]
        haskey(groups, e) || (groups[e] = String[]; push!(order, e))
        push!(groups[e], n)
    end
    function ask!(batch::Vector{String})
        lines = ninja_commands(src, batch)
        if length(lines) == length(batch)
            for (n, c) in zip(batch, lines)
                out[n] = c
            end
        elseif length(lines) == 1
            for n in batch
                out[n] = lines[1]
            end
        else
            for n in batch
                one = ninja_commands(src, [n])
                length(one) == 1 ||
                    error("GraphGenerators: ninja printed $(length(one)) command lines for '$n'")
                out[n] = one[1]
            end
        end
    end
    singles = String[only(groups[e]) for e in order if length(groups[e]) == 1]
    for c in _chunks(singles)
        ask!(c)
    end
    for e in order
        length(groups[e]) > 1 || continue
        for c in _chunks(groups[e])
            ask!(c)
        end
    end
    return out
end

# Which graph target writes a node that a compile or link edge writes, and its
# kind: by the target's artifacts, then by cmake's object directory
# (`CMakeFiles/<target>.dir/`), then by cmake's rule name
# (`<LANG>_COMPILER__<target>_…`, `<LANG>_<KIND>_LINKER__<target>_…`).
function _owner(g::TargetGraph, artifacts::Dict{String,GraphTarget}, abs::String, rule::String)
    t = get(artifacts, _key(abs), nothing)
    t === nothing || return (t.name, t.kind)
    byname = Dict{String,GraphTarget}(t.name => t for t in g.targets)
    m = match(r"/CMakeFiles/([^/]+)\.dir/", abs)
    if m !== nothing && haskey(byname, m.captures[1])
        t = byname[m.captures[1]]
        return (t.name, t.kind)
    end
    hits = sort!([n for n in keys(byname) if occursin("__" * n * "_", rule)];
                 by = n -> (-length(n), n))
    isempty(hits) || return (hits[1], byname[hits[1]].kind)
    kind = occursin("_EXECUTABLE_LINKER__", rule) ? :executable : :unknown
    return (rule, kind)
end

const _LIBRARY_KINDS = (:shared, :static, :object, :module)
const _LIBRARY_RULE = r"_(COMPILER|SCAN|DYNDEP|STATIC_LIBRARY_LINKER|SHARED_LIBRARY_LINKER|MODULE_LIBRARY_LINKER)__|^CMAKE_SYMLINK_LIBRARY$"

# An order-only artifact is ordering, and droppable, only when it is a
# library's: an executable is how cmake records the tool a command runs, and an
# artifact nothing identifies is treated as one.
function _droppable(kind::Symbol, rule::String)
    kind in _LIBRARY_KINDS && return true
    kind === :unknown && return occursin(_LIBRARY_RULE, rule)
    return false
end

# Whether a command names a file by its absolute path. cmake writes a target
# file into a command as an absolute path; on Windows with '\'.
function _names(command::String, abs::String)
    return occursin(_key(abs), _key(_posix(command)))
end

"""
    plan_generators(g, src) -> GeneratorPlan

Read off ninja's graph (`src`) which steps write the sources `g` marks
generated, and what each needs. Runs nothing.
"""
function plan_generators(g::TargetGraph, src::NinjaSource)::GeneratorPlan
    build_dir = _dir(g.build_dir)
    requested = requested_sources(g)

    # ninja's spelling of every node an edge writes
    spelled = Dict{String,String}()
    for (node, _) in ninja_outputs(src)
        k = _key(_abs(node, build_dir))
        haskey(spelled, k) || (spelled[k] = node)
    end

    # ninja spellings of the requested nodes some edge writes
    roots = String[spelled[_key(p)] for p in requested if haskey(spelled, _key(p))]

    # Walk ninja's graph from the requested nodes, querying in batches. An input
    # a step declares is a need; an order-only input is ordering. Phony edges
    # pass both through (their own order-only inputs as ordering). A custom
    # edge reached as a need is a step and is walked; one reached only as
    # ordering is not. A compile or link edge is never walked.
    edges = Dict{String,Union{Nothing,NinjaEdge}}()
    mode = Dict{String,Symbol}()
    discovered = String[]
    queue = Tuple{String,Symbol}[]
    function reach!(n::String, m::Symbol)
        old = get(mode, n, nothing)
        old === m && return
        old === :need && return
        old === nothing && push!(discovered, n)
        mode[n] = m
        push!(queue, (n, m))
    end
    foreach(n -> reach!(n, :need), roots)
    while !isempty(queue)
        todo = copy(queue)
        empty!(queue)
        unknown = unique!([n for (n, _) in todo if !haskey(edges, n)])
        isempty(unknown) || merge!(edges, ninja_query(src, unknown))
        for (n, m) in todo
            haskey(edges, n) ||
                error("GraphGenerators: ninja did not answer for node '$n'")
            e = edges[n]
            (_is_phony(e) || (_is_custom(e) && m === :need)) || continue
            foreach(i -> reach!(i, m), e.explicit)
            foreach(i -> reach!(i, m), e.implicit)
            foreach(i -> reach!(i, :order), e.order_only)
        end
    end

    # Steps: the custom nodes reached as a need, one per distinct command.
    custom = [n for n in discovered if mode[n] === :need && _is_custom(edges[n])]
    command = _commands(src, custom, edges)
    step_key = Dict{String,Int}()          # node → step index
    groups = Dict{Tuple{NinjaEdge,String},Int}()
    nodes_of = Vector{String}[]
    for n in custom
        k = (edges[n]::NinjaEdge, command[n])
        i = get!(groups, k) do
            push!(nodes_of, String[])
            length(nodes_of)
        end
        push!(nodes_of[i], n)
        step_key[n] = i
    end

    artifacts = Dict{String,GraphTarget}()
    for t in g.targets, a in t.artifacts
        artifacts[_key(_abs(a, build_dir))] = t
    end

    nsteps = length(nodes_of)
    need_steps = [Set{Int}() for _ in 1:nsteps]
    after_steps = [Set{Int}() for _ in 1:nsteps]
    inputs = [String[] for _ in 1:nsteps]
    needs = Dict{Tuple{String,String,Symbol},Set{Int}}()   # (target, artifact, why) → steps
    root_needs = Dict{Tuple{String,String,Symbol},Set{String}}()
    dropped = Set{String}()

    # What a step (or a requested node, `i == 0`) reaches through phony edges.
    function classify!(i::Int, n::String, m::Symbol, seen::Set{Tuple{String,Symbol}}, root::String)
        (n, m) in seen && return
        push!(seen, (n, m))
        e = edges[n]
        if e === nothing
            m === :need && i > 0 && push!(inputs[i], _abs(n, build_dir))
        elseif _is_phony(e)
            isempty(e.explicit) && isempty(e.implicit) && isempty(e.order_only) && return
            foreach(x -> classify!(i, x, m, seen, root), e.explicit)
            foreach(x -> classify!(i, x, m, seen, root), e.implicit)
            foreach(x -> classify!(i, x, :order, seen, root), e.order_only)
        elseif _is_custom(e)
            j = get(step_key, n, 0)
            j == 0 && return              # a custom output reached only as ordering
            i == 0 && return
            if m === :need
                push!(need_steps[i], j)
                push!(inputs[i], _abs(n, build_dir))
            else
                push!(after_steps[i], j)
            end
        else
            abs = _abs(n, build_dir)
            name, kind = _owner(g, artifacts, abs, e.rule)
            why = if m === :need
                :input
            elseif !_droppable(kind, e.rule)
                :tool
            elseif i > 0 && _names(command[first(nodes_of[i])], abs)
                :named
            else
                nothing
            end
            if why === nothing
                push!(dropped, name)
            elseif i == 0
                push!(get!(root_needs, (name, abs, why), Set{String}()), root)
            else
                push!(get!(needs, (name, abs, why), Set{Int}()), i)
            end
        end
    end

    for i in 1:nsteps
        e = edges[first(nodes_of[i])]::NinjaEdge
        seen = Set{Tuple{String,Symbol}}()
        foreach(x -> classify!(i, x, :need, seen, ""), e.explicit)
        foreach(x -> classify!(i, x, :need, seen, ""), e.implicit)
        foreach(x -> classify!(i, x, :order, seen, ""), e.order_only)
    end
    # The steps each requested file needs. A requested node is a step's output,
    # or a phony edge over steps' outputs, or written by a compile or a link (a
    # refusal); a phony edge that reaches nothing is ninja's "may be missing",
    # which is no step at all.
    root_steps = Dict{String,Set{Int}}()
    existing = String[]
    unwritten = String[]
    for p in requested
        s = get!(root_steps, p, Set{Int}())
        node = get(spelled, _key(p), nothing)
        built = false
        stack = node === nothing ? String[] : [node]
        seen = Set{String}()
        while !isempty(stack)
            x = pop!(stack)
            x in seen && continue
            push!(seen, x)
            e = edges[x]
            if haskey(step_key, x)
                push!(s, step_key[x])
            elseif _is_phony(e)
                append!(stack, e.explicit, e.implicit)
            elseif _is_built(e)
                built = true
                classify!(0, x, :need, Set{Tuple{String,Symbol}}(), p)
            end
        end
        isempty(s) && !built && push!(isfile(p) ? existing : unwritten, p)
    end

    # The requested files each step serves: through the steps it needs.
    served = [Set{String}() for _ in 1:nsteps]
    for (r, s) in root_steps
        stack = collect(s)
        seen = Set{Int}()
        while !isempty(stack)
            j = pop!(stack)
            j in seen && continue
            push!(seen, j)
            push!(served[j], r)
            append!(stack, need_steps[j])
        end
    end

    need_list = NeededTarget[]
    for (k, is) in needs
        by = Set{String}()
        foreach(i -> union!(by, served[i]), is)
        push!(need_list, NeededTarget(k[1], k[2], k[3], sort!(collect(by))))
    end
    for (k, rs) in root_needs
        push!(need_list, NeededTarget(k[1], k[2], k[3], sort!(collect(rs))))
    end
    sort!(need_list; by = n -> (n.target, n.artifact, n.why))
    # merge entries that differ only by where they were found
    merged = NeededTarget[]
    for n in need_list
        if !isempty(merged) && (merged[end].target, merged[end].artifact, merged[end].why) ==
                               (n.target, n.artifact, n.why)
            merged[end] = NeededTarget(n.target, n.artifact, n.why,
                                       sort!(unique!(vcat(merged[end].needed_by, n.needed_by))))
        else
            push!(merged, n)
        end
    end

    # Steps in dependency order; ties in the order the walk found them.
    waiting = [length(need_steps[i] ∪ after_steps[i]) for i in 1:nsteps]
    users = [Int[] for _ in 1:nsteps]
    for i in 1:nsteps, j in need_steps[i] ∪ after_steps[i]
        push!(users[j], i)
    end
    ready = sort!([i for i in 1:nsteps if waiting[i] == 0])
    order = Int[]
    while !isempty(ready)
        i = popfirst!(ready)
        push!(order, i)
        for u in users[i]
            waiting[u] -= 1
            waiting[u] == 0 && push!(ready, u)
        end
        sort!(ready)
    end
    length(order) == nsteps ||
        error("GraphGenerators: the steps of $build_dir depend on each other in a cycle")

    out_of(i) = unique!([_abs(n, build_dir) for n in nodes_of[i]])
    steps = GeneratorStep[]
    for i in order
        after = String[]
        for j in sort!(collect(after_steps[i]))
            j in need_steps[i] && continue
            push!(after, first(out_of(j)))
        end
        push!(steps, GeneratorStep(out_of(i), command[first(nodes_of[i])],
                                   unique!(copy(inputs[i])), after))
    end

    return GeneratorPlan(build_dir, requested, existing, unwritten, steps, merged,
                         sort!(collect(dropped)))
end

# ============================================================================
# THE NARROWED MANIFEST
# ============================================================================

const MANIFEST_DIR = "CMakeFiles/replibuild_generators"
const MANIFEST_FILE = "generators.ninja"

function _escape_path(p::String)::String
    (occursin('\n', p) || occursin('|', p)) &&
        error("GraphGenerators: a ninja manifest cannot spell the path '$p'")
    return replace(p, '$' => "\$\$", ' ' => "\$ ", ':' => "\$:")
end

function _escape_value(s::String)::String
    occursin('\n', s) && error("GraphGenerators: a step's command spans lines: $s")
    return replace(s, '$' => "\$\$")
end

"""
    generator_manifest(plan) -> String

The ninja manifest that runs `plan.steps`, from the build tree, with the
build's own commands. Every node has one spelling (relative to the build tree
inside it), and a step lists what it declares it reads, plus the outputs of the
steps cmake orders it after as order-only inputs.
"""
function generator_manifest(plan::GeneratorPlan)::String
    bd = plan.build_dir
    sp(p) = _escape_path(_spell(p, bd))
    io = IOBuffer()
    println(io, "# Written and run by RepliBuild.GraphGenerators, from the build tree")
    println(io, "#   ", bd)
    println(io, "# The build's own steps that write the sources RepliBuild compiles, with")
    println(io, "# their own commands, and without the order-only inputs on other targets'")
    println(io, "# artifacts that cmake gives every custom command.")
    println(io, "ninja_required_version = 1.5")
    println(io, "builddir = ", MANIFEST_DIR)
    println(io)
    println(io, "rule step")
    println(io, "  command = \$cmd")
    println(io, "  description = \$desc")
    for s in plan.steps
        println(io)
        print(io, "build ", join(map(sp, s.outputs), ' '), ": step")
        isempty(s.inputs) || print(io, ' ', join(map(sp, s.inputs), ' '))
        isempty(s.after) || print(io, " || ", join(map(sp, s.after), ' '))
        println(io)
        println(io, "  cmd = ", _escape_value(s.command))
        println(io, "  desc = ", _escape_value("Generating " *
                                               join([_spell(o, bd) for o in s.outputs], ", ")))
    end
    return String(take!(io))
end

# ============================================================================
# RUNNING
# ============================================================================

function _cmake_cache(path::String)::Dict{String,String}
    out = Dict{String,String}()
    for l in _lines(read(path, String))
        (isempty(l) || startswith(l, '#') || startswith(l, "//")) && continue
        m = match(r"^([^:=]+):[^=]*=(.*)$", l)
        m === nothing || (out[m.captures[1]] = m.captures[2])
    end
    return out
end

"""
    ninja_program(build_dir) -> String

The ninja cmake configured `build_dir` with (`CMAKE_MAKE_PROGRAM`). An error
when the tree is not configured, or not with the `Ninja` generator.
"""
function ninja_program(build_dir::String)::String
    cache = joinpath(build_dir, "CMakeCache.txt")
    isfile(cache) ||
        error("GraphGenerators: $build_dir is not a configured CMake build tree (no CMakeCache.txt)")
    c = _cmake_cache(cache)
    gen = get(c, "CMAKE_GENERATOR", "")
    gen == "Ninja" ||
        error("GraphGenerators: $build_dir was configured with the '$gen' generator. Generated " *
              "sources are produced through Ninja's build graph: configure with -G Ninja.")
    isfile(joinpath(build_dir, "build.ninja")) ||
        error("GraphGenerators: $build_dir has no build.ninja; cmake's generate step did not finish")
    prog = get(c, "CMAKE_MAKE_PROGRAM", "")
    return isempty(prog) ? "ninja" : prog
end

# The outputs ninja's build log records, as keys.
function _logged(path::String, build_dir::String)::Set{String}
    out = Set{String}()
    isfile(path) || return out
    for l in _lines(read(path, String))
        (isempty(l) || startswith(l, '#')) && continue
        f = split(l, '\t')
        length(f) >= 4 && push!(out, _key(_abs(f[4], build_dir)))
    end
    return out
end

"""
    run_generators(g; jobs = Sys.CPU_THREADS, use_llvm_env = true) -> GeneratorRun

Make every source `g` marks `generated` exist, by running the steps of the
project's own build that write them (see the module header), and never a
compile or a link of its targets. `g.build_dir` must be configured with
`-G Ninja`. A graph with no generated sources needs no build tree and runs
nothing.

Throws `GeneratorNeedsCompile` (naming the targets) before running anything
when a step needs one of the project's own compiles or links,
`GeneratedSourceUnwritten` when no step writes a missing generated source, and
`GeneratorStepFailed` when a step fails. `use_llvm_env` runs the steps in the
toolchain environment, as `SysConfigGen.cmake_probe` configures in it.
"""
function run_generators(g::TargetGraph; jobs::Int = Sys.CPU_THREADS,
                        use_llvm_env::Bool = true)::GeneratorRun
    build_dir = _dir(g.build_dir)
    if isempty(requested_sources(g))
        empty = GeneratorPlan(build_dir, String[], String[], String[], GeneratorStep[],
                              NeededTarget[], String[])
        return GeneratorRun(empty, GeneratorStep[], "", "")
    end
    ninja = ninja_program(build_dir)
    plan = plan_generators(g, LiveNinja(ninja, build_dir))
    isempty(plan.needs) || throw(GeneratorNeedsCompile(build_dir, plan.needs))
    isempty(plan.unwritten) || throw(GeneratedSourceUnwritten(build_dir, plan.unwritten))
    isempty(plan.steps) && return GeneratorRun(plan, GeneratorStep[], "", "")

    dir = build_dir * "/" * MANIFEST_DIR
    mkpath(dir)
    manifest = dir * "/" * MANIFEST_FILE
    write(manifest, generator_manifest(plan))
    # A fresh log makes every step dirty, so every step runs (module header).
    log = dir * "/.ninja_log"
    rm(log; force = true)
    rm(dir * "/.ninja_deps"; force = true)

    cmd = Cmd(`$ninja -f $(MANIFEST_DIR * "/" * MANIFEST_FILE) -j $jobs`; dir = build_dir)
    output, code = BuildBridge.run_command(cmd; use_llvm_env = use_llvm_env)
    code == 0 || throw(GeneratorStepFailed(manifest, code, output))

    logged = _logged(log, build_dir)
    ran = [s for s in plan.steps if any(o -> _key(o) in logged, s.outputs)]
    absent = filter(!isfile, plan.requested)
    isempty(absent) ||
        error("GraphGenerators: ninja ran $(length(ran)) step(s) from $manifest, and these ",
              "generated sources still do not exist: ", join(absent, ", "))
    return GeneratorRun(plan, ran, manifest, output)
end

end # module GraphGenerators
