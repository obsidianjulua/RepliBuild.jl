# DAGDiff.jl — Structural diff between C++ (DWARF) and Julia (ccall) IR DAGs
#
# Extension to per-function heuristics in DispatchLogic.jl.  Heuristics catch
# the obvious cases (packed returns, unions, STL).  DAGDiff catches structural
# mismatches that point-wise checks miss: transitive layout drift through
# by-value containment, member offset divergence even when sizes match, and
# cross-type propagation.
#
# Pipeline position:  DWARFParser → DAGDiff → JLCSIRGenerator / Wrapper
# Input:              compilation_metadata.json (same dict ThunkBuilder uses)
# Output:             DAGDiffResult with mismatch map + topo-sorted lowering order

module DAGDiff

export DAGMismatch, MismatchKind, DAGDiffResult, IRGraph, TypeNode, FunctionNode, MemberLayout
export build_cpp_graph, build_julia_graph, diff_graphs, topo_sort_mismatches, dag_diff
export needs_dag_thunk
export export_dot, export_graph_dot, render_dot

# ============================================================================
# Mismatch Classification
# ============================================================================

@enum MismatchKind begin
    LAYOUT_MISMATCH       # struct byte_size or member offsets differ
    RETURN_CONV_MISMATCH  # return type is a mismatched struct by value
    PARAM_CONV_MISMATCH   # parameter type is a mismatched struct by value
    MISSING_TYPE          # type referenced but absent from DWARF
    TRANSITIVE_MISMATCH   # function uses a type whose dependency chain has a mismatch
end

# ============================================================================
# Graph Node Types
# ============================================================================

struct MemberLayout
    name::String
    type_name::String
    offset::Int
    size::Int
end

struct TypeNode
    name::String
    byte_size::Int
    members::Vector{MemberLayout}
    kind::Symbol            # :struct, :union, :enum, :class
    has_vtable::Bool
    base_classes::Vector{String}
end

struct FunctionNode
    symbol::String          # mangled name (anchor between both DAGs)
    name::String            # demangled / human-readable
    return_type::String
    param_types::Vector{String}
    byte_offset::UInt64     # DW_AT_low_pc when available, else 0
end

# ============================================================================
# IR Graph — nodes + edges for one side of the diff
# ============================================================================

struct IRGraph
    types::Dict{String, TypeNode}
    functions::Dict{String, FunctionNode}
    type_edges::Dict{String, Set{String}}       # type → by-value contained types
    func_type_edges::Dict{String, Set{String}}   # function → types it uses by value
end

# ============================================================================
# Mismatch Record
# ============================================================================

struct DAGMismatch
    symbol::String          # mangled name or type name
    kind::MismatchKind
    dwarf_offset::UInt64    # byte offset from DWARF (member offset for layout, 0 otherwise)
    delta::Int              # size discrepancy (positive = DWARF larger, negative = Julia larger)
    detail::String
end

# ============================================================================
# Diff Result
# ============================================================================

struct DAGDiffResult
    mismatches::Vector{DAGMismatch}
    lowering_order::Vector{String}    # topo-sorted symbols needing thunks
    cpp_graph::IRGraph
    julia_graph::IRGraph
    mismatched_types::Set{String}     # types with layout mismatches
    mismatched_functions::Set{String}  # functions needing thunks (DAG-detected only)
end

# ============================================================================
# Query API
# ============================================================================

"""
    needs_dag_thunk(symbol::String, result::DAGDiffResult) -> Bool

Check if a function (by mangled name) was flagged by the DAG diff as needing
a thunk.  Used alongside the existing heuristic dispatch — the wrapper
generator should route to a thunk if EITHER the heuristic OR this returns true.
"""
function needs_dag_thunk(symbol::String, result::DAGDiffResult)::Bool
    return symbol in result.mismatched_functions
end

"""
    needs_dag_thunk(symbol::String, result::Nothing) -> false

No-op when DAG diff was not computed (backward compat).
"""
needs_dag_thunk(::String, ::Nothing) = false

# ============================================================================
# Build C++ Graph (DWARF ground truth)
# ============================================================================

"""
    build_cpp_graph(metadata::Dict) -> IRGraph

Build the C++ side DAG from compilation metadata (DWARF struct definitions +
function signatures).  This represents the *actual* ABI layout.
"""
function build_cpp_graph(metadata::Dict)::IRGraph
    types = Dict{String, TypeNode}()
    functions = Dict{String, FunctionNode}()
    type_edges = Dict{String, Set{String}}()
    func_type_edges = Dict{String, Set{String}}()

    dwarf_structs = get(metadata, "struct_definitions", Dict())

    # ── Type nodes from DWARF ──────────────────────────────────────────
    for (name, info) in dwarf_structs
        startswith(name, "__enum__") && continue

        byte_size = _parse_size(get(info, "byte_size", "0"))
        kind = Symbol(get(info, "kind", "struct"))
        members_raw = get(info, "members", [])

        members = MemberLayout[]
        contained = Set{String}()

        for m in members_raw
            m_name = get(m, "name", "")
            m_type = get(m, "c_type", get(m, "type", ""))
            m_offset = _parse_size(get(m, "offset", "0"))
            m_size = _parse_size(get(m, "size", "0"))

            # Resolve nested struct sizes from DWARF when parser left size=0
            if m_size == 0
                cleaned = _strip_qualifiers(m_type)
                if haskey(dwarf_structs, cleaned)
                    m_size = _parse_size(get(dwarf_structs[cleaned], "byte_size", "0"))
                end
            end

            push!(members, MemberLayout(m_name, m_type, m_offset, m_size))

            # Track by-value type edges (not pointers/refs)
            if !_is_indirection(m_type)
                cleaned = _strip_qualifiers(m_type)
                if haskey(dwarf_structs, cleaned) && !startswith(cleaned, "__enum__")
                    push!(contained, cleaned)
                end
            end
        end

        has_vtable = get(info, "is_polymorphic", false) == true ||
                     get(info, "has_vtable", false) == true
        base_raw = get(info, "base_classes", [])
        base_classes = base_raw isa Vector ? String[string(b) for b in base_raw] : String[]

        types[name] = TypeNode(name, byte_size, members, kind, has_vtable, base_classes)
        type_edges[name] = contained
    end

    # ── Function nodes from metadata ───────────────────────────────────
    for func in get(metadata, "functions", [])
        mangled = get(func, "mangled", "")
        isempty(mangled) && continue

        fname = get(func, "name", mangled)
        ret_type = get(get(func, "return_type", Dict()), "c_type", "void")
        params = get(func, "parameters", [])
        param_types = String[get(p, "c_type", "") for p in params]

        functions[mangled] = FunctionNode(mangled, fname, ret_type, param_types, UInt64(0))

        # Function → type edges (by-value params + return)
        deps = Set{String}()
        for pt in vcat([ret_type], param_types)
            if !_is_indirection(pt)
                cleaned = _strip_qualifiers(pt)
                if haskey(dwarf_structs, cleaned) && !startswith(cleaned, "__enum__")
                    push!(deps, cleaned)
                end
            end
        end
        func_type_edges[mangled] = deps
    end

    return IRGraph(types, functions, type_edges, func_type_edges)
end

# ============================================================================
# Build Julia Graph (what ccall / Julia alignment rules would assume)
# ============================================================================

"""
    build_julia_graph(metadata::Dict) -> IRGraph

Build the Julia-side DAG by computing Julia-aligned struct layouts from DWARF
member info.  Mirrors the alignment logic in Wrapper/Utils.jl:get_julia_aligned_size
so that the diff detects exactly the same mismatches the runtime would hit.
"""
function build_julia_graph(metadata::Dict)::IRGraph
    types = Dict{String, TypeNode}()
    functions = Dict{String, FunctionNode}()
    type_edges = Dict{String, Set{String}}()
    func_type_edges = Dict{String, Set{String}}()

    dwarf_structs = get(metadata, "struct_definitions", Dict())

    # ── Julia-aligned type nodes ───────────────────────────────────────
    for (name, info) in dwarf_structs
        startswith(name, "__enum__") && continue

        kind = Symbol(get(info, "kind", "struct"))
        members_raw = get(info, "members", [])

        julia_members = MemberLayout[]
        current_offset = 0
        max_align = 1
        contained = Set{String}()

        for m in members_raw
            m_name = get(m, "name", "")
            m_type = get(m, "c_type", get(m, "type", ""))
            m_size = _parse_size(get(m, "size", "0"))

            # Resolve nested struct sizes
            if m_size == 0
                cleaned = _strip_qualifiers(m_type)
                if haskey(dwarf_structs, cleaned)
                    m_size = _parse_size(get(dwarf_structs[cleaned], "byte_size", "0"))
                end
            end

            # Julia alignment: min(sizeof(field), 8), same as get_julia_aligned_size
            if kind == :union
                # Union: all members at offset 0
                push!(julia_members, MemberLayout(m_name, m_type, 0, m_size))
                max_align = max(max_align, min(max(m_size, 1), 8))
            else
                alignment = m_size > 8 ? 8 : m_size
                alignment = alignment == 0 ? 1 : alignment
                max_align = max(max_align, alignment)

                padding = (alignment - (current_offset % alignment)) % alignment
                current_offset += padding

                push!(julia_members, MemberLayout(m_name, m_type, current_offset, m_size))
                current_offset += m_size
            end

            if !_is_indirection(m_type)
                cleaned = _strip_qualifiers(m_type)
                if haskey(dwarf_structs, cleaned) && !startswith(cleaned, "__enum__")
                    push!(contained, cleaned)
                end
            end
        end

        # Final struct size
        byte_size = if kind == :union
            isempty(julia_members) ? 0 : maximum(m.size for m in julia_members)
        else
            final_pad = (max_align - (current_offset % max_align)) % max_align
            current_offset + final_pad
        end

        types[name] = TypeNode(name, byte_size, julia_members, kind, false, String[])
        type_edges[name] = contained
    end

    # ── Function nodes (same signatures — Julia uses the declared types) ─
    for func in get(metadata, "functions", [])
        mangled = get(func, "mangled", "")
        isempty(mangled) && continue

        fname = get(func, "name", mangled)
        ret_type = get(get(func, "return_type", Dict()), "c_type", "void")
        params = get(func, "parameters", [])
        param_types = String[get(p, "c_type", "") for p in params]

        functions[mangled] = FunctionNode(mangled, fname, ret_type, param_types, UInt64(0))

        deps = Set{String}()
        for pt in vcat([ret_type], param_types)
            if !_is_indirection(pt)
                cleaned = _strip_qualifiers(pt)
                if haskey(dwarf_structs, cleaned) && !startswith(cleaned, "__enum__")
                    push!(deps, cleaned)
                end
            end
        end
        func_type_edges[mangled] = deps
    end

    return IRGraph(types, functions, type_edges, func_type_edges)
end

# ============================================================================
# Structural Diff — Parallel Walk
# ============================================================================

"""
    diff_graphs(cpp::IRGraph, julia::IRGraph) -> Vector{DAGMismatch}

Walk both graphs in parallel (anchored on shared type/function names) and
record every structural mismatch.  Produces the raw mismatch list that
`topo_sort_mismatches` orders for safe lowering.
"""
function diff_graphs(cpp::IRGraph, julia::IRGraph)::Vector{DAGMismatch}
    mismatches = DAGMismatch[]
    mismatched_type_names = Set{String}()

    # ── Pass 1: Type layout comparison ─────────────────────────────────
    for (name, cpp_t) in cpp.types
        if !haskey(julia.types, name)
            push!(mismatches, DAGMismatch(name, MISSING_TYPE, UInt64(0), 0,
                "Type '$name' present in DWARF but no Julia layout computed"))
            push!(mismatched_type_names, name)
            continue
        end

        jl_t = julia.types[name]

        # Size mismatch
        if cpp_t.byte_size > 0 && jl_t.byte_size > 0 && cpp_t.byte_size != jl_t.byte_size
            delta = cpp_t.byte_size - jl_t.byte_size
            push!(mismatches, DAGMismatch(name, LAYOUT_MISMATCH, UInt64(0), delta,
                "Type '$name': DWARF size=$(cpp_t.byte_size)B vs Julia aligned size=$(jl_t.byte_size)B (delta=$(delta)B)"))
            push!(mismatched_type_names, name)
        end

        # Per-member offset comparison
        n = min(length(cpp_t.members), length(jl_t.members))
        for i in 1:n
            cm = cpp_t.members[i]
            jm = jl_t.members[i]
            if cm.offset != jm.offset
                delta = cm.offset - jm.offset
                push!(mismatches, DAGMismatch(name, LAYOUT_MISMATCH, UInt64(cm.offset), delta,
                    "Type '$name'.$(cm.name): DWARF offset=$(cm.offset) vs Julia offset=$(jm.offset) (delta=$(delta)B)"))
                push!(mismatched_type_names, name)
            end
        end

        # Member count mismatch (DWARF has members Julia doesn't see, or vice versa)
        if length(cpp_t.members) != length(jl_t.members) && !isempty(cpp_t.members)
            push!(mismatches, DAGMismatch(name, LAYOUT_MISMATCH, UInt64(0), 0,
                "Type '$name': DWARF has $(length(cpp_t.members)) members vs Julia $(length(jl_t.members))"))
            push!(mismatched_type_names, name)
        end
    end

    # ── Pass 2: Propagate type mismatches through containment edges ────
    # If type A contains type B by value and B is mismatched, A is also mismatched.
    changed = true
    while changed
        changed = false
        for (name, deps) in cpp.type_edges
            name in mismatched_type_names && continue
            for dep in deps
                if dep in mismatched_type_names
                    push!(mismatches, DAGMismatch(name, LAYOUT_MISMATCH, UInt64(0), 0,
                        "Type '$name' contains mismatched type '$dep' by value"))
                    push!(mismatched_type_names, name)
                    changed = true
                    break
                end
            end
        end
    end

    # ── Pass 3: Function convention mismatches ─────────────────────────
    for (mangled, cpp_func) in cpp.functions
        func_flagged = false

        # Return type by value through a mismatched struct
        ret_cleaned = _strip_qualifiers(cpp_func.return_type)
        if !_is_indirection(cpp_func.return_type) && ret_cleaned in mismatched_type_names
            push!(mismatches, DAGMismatch(mangled, RETURN_CONV_MISMATCH,
                cpp_func.byte_offset, 0,
                "Function '$(cpp_func.name)' returns mismatched type '$ret_cleaned' by value"))
            func_flagged = true
        end

        # Parameter types by value through mismatched structs
        for (i, pt) in enumerate(cpp_func.param_types)
            pt_cleaned = _strip_qualifiers(pt)
            if !_is_indirection(pt) && pt_cleaned in mismatched_type_names
                push!(mismatches, DAGMismatch(mangled, PARAM_CONV_MISMATCH,
                    cpp_func.byte_offset, 0,
                    "Function '$(cpp_func.name)' param $i ('$pt_cleaned') is a mismatched type by value"))
                func_flagged = true
            end
        end

        # Transitive: function depends on types whose containment chains reach a mismatch
        if !func_flagged
            deps = get(cpp.func_type_edges, mangled, Set{String}())
            all_deps = _transitive_closure(deps, cpp.type_edges)
            for dep in all_deps
                if dep in mismatched_type_names
                    push!(mismatches, DAGMismatch(mangled, TRANSITIVE_MISMATCH,
                        cpp_func.byte_offset, 0,
                        "Function '$(cpp_func.name)' transitively depends on mismatched type '$dep'"))
                    break  # one is enough to flag the function
                end
            end
        end
    end

    return mismatches
end

# ============================================================================
# Topological Sort — Safe Lowering Order
# ============================================================================

"""
    topo_sort_mismatches(mismatches::Vector{DAGMismatch}, cpp::IRGraph) -> Vector{String}

Produce a topologically sorted emission order for all symbols that need thunks.
Type thunks are emitted before function thunks that depend on them.  Cycles
(mutual recursion through pointers) are broken — pointer indirection is always
a fixed-size machine word, so it never needs a layout thunk.
"""
function topo_sort_mismatches(mismatches::Vector{DAGMismatch}, cpp::IRGraph)::Vector{String}
    thunk_symbols = Set{String}(m.symbol for m in mismatches)
    isempty(thunk_symbols) && return String[]

    # Adjacency list: edge from A→B means A must be emitted before B
    adj = Dict{String, Vector{String}}()
    in_degree = Dict{String, Int}()

    for s in thunk_symbols
        adj[s] = String[]
        in_degree[s] = 0
    end

    # Type→type: if B contains A by value, A must come before B
    for s in thunk_symbols
        for dep in get(cpp.type_edges, s, Set{String}())
            if dep in thunk_symbols
                push!(adj[dep], s)
                in_degree[s] += 1
            end
        end
    end

    # Function→type: type thunks before function thunks that use them
    for s in thunk_symbols
        for dep in get(cpp.func_type_edges, s, Set{String}())
            if dep in thunk_symbols
                push!(adj[dep], s)
                in_degree[s] += 1
            end
        end
    end

    # Kahn's algorithm
    queue = String[s for s in thunk_symbols if in_degree[s] == 0]
    sorted = String[]
    sizehint!(sorted, length(thunk_symbols))

    while !isempty(queue)
        node = popfirst!(queue)
        push!(sorted, node)
        for neighbor in adj[node]
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0
                push!(queue, neighbor)
            end
        end
    end

    # Remaining nodes form cycles — append them (cycle broken at pointer indirection)
    for s in thunk_symbols
        if !(s in Set(sorted))
            push!(sorted, s)
        end
    end

    return sorted
end

# ============================================================================
# Top-Level Entry Point
# ============================================================================

"""
    dag_diff(metadata::Dict) -> DAGDiffResult

Run the full DAG diff pipeline:
1. Build C++ graph from DWARF metadata
2. Build Julia graph from inferred alignment rules
3. Parallel walk → mismatch detection
4. Topo-sort mismatches for safe lowering order

Returns a DAGDiffResult that the wrapper generators query via `needs_dag_thunk()`.
"""
function dag_diff(metadata::Dict)::DAGDiffResult
    cpp_graph = build_cpp_graph(metadata)
    julia_graph = build_julia_graph(metadata)

    mismatches = diff_graphs(cpp_graph, julia_graph)
    lowering_order = topo_sort_mismatches(mismatches, cpp_graph)

    mismatched_types = Set{String}(
        m.symbol for m in mismatches
        if m.kind in (LAYOUT_MISMATCH, MISSING_TYPE)
    )
    mismatched_functions = Set{String}(
        m.symbol for m in mismatches
        if m.kind in (RETURN_CONV_MISMATCH, PARAM_CONV_MISMATCH, TRANSITIVE_MISMATCH)
    )

    n_types = length(mismatched_types)
    n_funcs = length(mismatched_functions)
    n_total = length(mismatches)
    if n_total > 0
        println("  dag-diff: $n_total mismatches ($n_types types, $n_funcs functions), $(length(lowering_order)) thunk sites")
    end

    return DAGDiffResult(mismatches, lowering_order, cpp_graph, julia_graph,
                         mismatched_types, mismatched_functions)
end

# ============================================================================
# DOT Visualization
# ============================================================================

"""
    export_dot(result::DAGDiffResult, path::String; side=:diff, show_members=true)

Export a DAG diff result to Graphviz DOT format.

`side` selects what to render:
- `:diff`  — both graphs overlaid, mismatches highlighted in red (default)
- `:cpp`   — C++ (DWARF) graph only
- `:julia` — Julia (inferred alignment) graph only

Mismatched types render with red fill, mismatched functions with orange fill.
Edges that propagate a mismatch are drawn in red.
"""
function export_dot(result::DAGDiffResult, path::String;
                    side::Symbol=:diff, show_members::Bool=true)
    graph = side == :julia ? result.julia_graph :
            side == :cpp   ? result.cpp_graph   : result.cpp_graph

    io = IOBuffer()
    println(io, "digraph DAGDiff {")
    println(io, "  rankdir=BT;")
    println(io, "  fontname=\"Helvetica\";")
    println(io, "  node [fontname=\"Helvetica\", fontsize=10];")
    println(io, "  edge [fontname=\"Helvetica\", fontsize=8];")
    println(io, "")

    # Legend
    if side == :diff
        println(io, "  subgraph cluster_legend {")
        println(io, "    label=\"Legend\"; style=dashed; fontsize=9; color=gray60;")
        println(io, "    node [shape=plaintext, fontsize=8];")
        println(io, "    l1 [label=\"Red = layout mismatch\", fontcolor=red3];")
        println(io, "    l2 [label=\"Orange = function needs thunk\", fontcolor=darkorange];")
        println(io, "    l3 [label=\"Gray = safe (ccall ok)\", fontcolor=gray50];")
        println(io, "    l1 -> l2 -> l3 [style=invis];")
        println(io, "  }")
        println(io, "")
    end

    # ── Type nodes ─────────────────────────────────────────────────────
    println(io, "  // Type nodes")
    for (name, tnode) in graph.types
        is_mismatched = name in result.mismatched_types

        # Node label
        label_lines = ["<B>$(escape_dot(name))</B>"]
        push!(label_lines, "$(tnode.kind) | $(tnode.byte_size)B")

        if side == :diff && is_mismatched
            # Show the delta
            if haskey(result.julia_graph.types, name)
                jl_size = result.julia_graph.types[name].byte_size
                push!(label_lines, "<FONT COLOR=\"red3\">DWARF=$(tnode.byte_size)B  Julia=$(jl_size)B</FONT>")
            end
        end

        if show_members && !isempty(tnode.members)
            push!(label_lines, " ")
            for m in tnode.members
                mline = "+$(m.offset) $(escape_dot(m.name)): $(escape_dot(m.type_name)) ($(m.size)B)"
                # Highlight offset mismatch per-member
                if side == :diff && is_mismatched && haskey(result.julia_graph.types, name)
                    jl_members = result.julia_graph.types[name].members
                    idx = findfirst(jm -> jm.name == m.name, jl_members)
                    if idx !== nothing && jl_members[idx].offset != m.offset
                        mline = "<FONT COLOR=\"red3\">+$(m.offset) $(escape_dot(m.name)): $(escape_dot(m.type_name)) ($(m.size)B)  [Julia: +$(jl_members[idx].offset)]</FONT>"
                    end
                end
                push!(label_lines, mline)
            end
        end

        label = "<" * join(label_lines, "<BR/>") * ">"
        fill = is_mismatched ? "fillcolor=\"#ffcccc\", style=filled," : ""
        border_color = is_mismatched ? "red3" : "gray60"
        println(io, "  \"t:$name\" [shape=record, $fill color=$border_color, label=$label];")
    end
    println(io, "")

    # ── Function nodes ─────────────────────────────────────────────────
    println(io, "  // Function nodes")
    for (mangled, fnode) in graph.functions
        is_mismatched = mangled in result.mismatched_functions

        # Short label: function name + return type
        short_name = length(fnode.name) > 40 ? fnode.name[1:37] * "..." : fnode.name
        label_lines = ["<B>$(escape_dot(short_name))</B>"]
        push!(label_lines, "$(escape_dot(fnode.return_type))($(join(map(escape_dot, fnode.param_types), ", ")))")

        if side == :diff && is_mismatched
            # Show mismatch reason
            for m in result.mismatches
                if m.symbol == mangled
                    reason = length(m.detail) > 60 ? m.detail[1:57] * "..." : m.detail
                    push!(label_lines, "<FONT COLOR=\"darkorange\">$(escape_dot(reason))</FONT>")
                    break
                end
            end
        end

        label = "<" * join(label_lines, "<BR/>") * ">"
        fill = is_mismatched ? "fillcolor=\"#ffe0b2\", style=filled," : ""
        border_color = is_mismatched ? "darkorange" : "gray60"
        println(io, "  \"f:$mangled\" [shape=ellipse, $fill color=$border_color, label=$label];")
    end
    println(io, "")

    # ── Type → type edges (by-value containment) ──────────────────────
    println(io, "  // Type containment edges")
    for (name, deps) in graph.type_edges
        for dep in deps
            haskey(graph.types, dep) || continue
            # Red if the edge propagates a mismatch
            propagates = dep in result.mismatched_types && name in result.mismatched_types
            color = propagates ? "red3" : "gray60"
            style = propagates ? "bold" : "solid"
            println(io, "  \"t:$dep\" -> \"t:$name\" [color=$color, style=$style, label=\"contains\", fontcolor=gray50];")
        end
    end
    println(io, "")

    # ── Function → type edges ─────────────────────────────────────────
    println(io, "  // Function → type edges")
    for (mangled, deps) in graph.func_type_edges
        for dep in deps
            haskey(graph.types, dep) || continue
            propagates = dep in result.mismatched_types && mangled in result.mismatched_functions
            color = propagates ? "darkorange" : "gray60"
            style = propagates ? "bold" : "dashed"
            println(io, "  \"t:$dep\" -> \"f:$mangled\" [color=$color, style=$style, label=\"uses\", fontcolor=gray50];")
        end
    end

    println(io, "}")

    dot_str = String(take!(io))
    write(path, dot_str)
    return path
end

"""
    export_graph_dot(graph::IRGraph, path::String; show_members=true)

Export a single IRGraph (no mismatch highlighting) to Graphviz DOT.
"""
function export_graph_dot(graph::IRGraph, path::String; show_members::Bool=true)
    # Build a dummy result with no mismatches so we can reuse the renderer
    empty_result = DAGDiffResult(
        DAGMismatch[], String[], graph, graph,
        Set{String}(), Set{String}()
    )
    return export_dot(empty_result, path; side=:cpp, show_members=show_members)
end

"""
    render_dot(result::DAGDiffResult, path::String; format="svg", side=:diff, show_members=true)

Export to DOT and render to an image via the `dot` command.
Requires Graphviz installed (`dot` on PATH).  Returns the output image path,
or the DOT path if `dot` is not available.

Supported formats: svg, png, pdf
"""
function render_dot(result::DAGDiffResult, path::String;
                    format::String="svg", side::Symbol=:diff, show_members::Bool=true)
    dot_path = replace(path, r"\.(svg|png|pdf)$" => "") * ".dot"
    export_dot(result, dot_path; side=side, show_members=show_members)

    out_path = replace(dot_path, ".dot" => ".$format")
    try
        run(pipeline(`dot -T$format -o $out_path $dot_path`, stderr=devnull))
        println("  dag-viz: $out_path")
        return out_path
    catch
        println("  dag-viz: dot not found, wrote $dot_path (install graphviz to render)")
        return dot_path
    end
end

"""Escape special characters for DOT HTML labels."""
function escape_dot(s::AbstractString)::String
    s = replace(s, "&" => "&amp;")
    s = replace(s, "<" => "&lt;")
    s = replace(s, ">" => "&gt;")
    s = replace(s, "\"" => "&quot;")
    return s
end

# ============================================================================
# Internal Helpers
# ============================================================================

"""Parse an integer that may be decimal, hex (0x…), or already numeric."""
function _parse_size(val)::Int
    val isa Integer && return Int(val)
    val isa AbstractString || return 0
    s = strip(string(val))
    isempty(s) && return 0
    try
        return (startswith(s, "0x") || startswith(s, "0X")) ?
            parse(Int, s[3:end], base=16) : parse(Int, s)
    catch
        return 0
    end
end

"""Strip const/volatile/restrict qualifiers and extra whitespace."""
function _strip_qualifiers(type_str::AbstractString)::String
    s = replace(type_str, r"\b(const|volatile|restrict)\b" => "")
    return strip(replace(s, r"\s+" => " "))
end

"""Check if a type string denotes pointer or reference indirection."""
function _is_indirection(type_str::AbstractString)::Bool
    return contains(type_str, "*") || contains(type_str, "&")
end

"""Compute transitive closure of a set through a dependency graph."""
function _transitive_closure(seeds::Set{String}, edges::Dict{String, Set{String}})::Set{String}
    visited = Set{String}()
    stack = collect(seeds)
    while !isempty(stack)
        t = pop!(stack)
        t in visited && continue
        push!(visited, t)
        for dep in get(edges, t, Set{String}())
            dep in visited || push!(stack, dep)
        end
    end
    return visited
end

end # module DAGDiff
