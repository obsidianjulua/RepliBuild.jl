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
export export_dot, export_graph_dot, render_dot, render_html

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
    bit_size::Int           # 0 = not a bitfield; >0 = number of bits
    bit_offset::Int         # absolute bit offset from struct start (0 if not a bitfield)
end

struct TypeNode
    name::String
    byte_size::Int
    members::Vector{MemberLayout}
    kind::Symbol            # :struct, :union, :enum, :class
    has_vtable::Bool
    base_classes::Vector{String}
    is_packed::Bool         # true if any member violates natural alignment
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
    type_edges::Dict{String, Set{String}}             # type → by-value contained types (includes inheritance)
    func_type_edges::Dict{String, Set{String}}         # function → types it uses by value
    inheritance_edges::Dict{String, Set{String}}       # derived → base classes (for visualization)
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
# Shared Topology Builder — edges + functions are identical on both sides
# ============================================================================

"""
    _build_shared_topology(metadata::Dict) -> (type_edges, func_type_edges, functions)

Extract the graph topology that is identical between the C++ and Julia DAGs:
by-value containment edges, function→type edges, and function nodes.  Each
graph builder then only needs to compute its own TypeNode layouts.
"""
function _build_shared_topology(metadata::Dict)
    dwarf_structs = get(metadata, "struct_definitions", Dict())

    type_edges = Dict{String, Set{String}}()
    func_type_edges = Dict{String, Set{String}}()
    functions = Dict{String, FunctionNode}()

    # ── Type containment edges (by-value, not through pointers/refs) ──
    for (name, info) in dwarf_structs
        startswith(name, "__enum__") && continue
        contained = Set{String}()
        for m in get(info, "members", [])
            m_type = get(m, "c_type", get(m, "type", ""))
            if !_is_indirection(m_type)
                cleaned = _strip_qualifiers(m_type)
                if haskey(dwarf_structs, cleaned) && !startswith(cleaned, "__enum__")
                    push!(contained, cleaned)
                end
            end
        end
        type_edges[name] = contained
    end

    # ── Function nodes + function→type edges ──────────────────────────
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

    return (type_edges, func_type_edges, functions)
end

# ============================================================================
# Build C++ Graph (DWARF ground truth)
# ============================================================================

"""
    build_cpp_graph(metadata::Dict) -> IRGraph

Build the C++ side DAG from compilation metadata (DWARF struct definitions +
function signatures).  This represents the *actual* ABI layout.
"""
function build_cpp_graph(metadata::Dict)::IRGraph
    type_edges, func_type_edges, functions = _build_shared_topology(metadata)
    types = Dict{String, TypeNode}()
    inheritance_edges = Dict{String, Set{String}}()
    dwarf_structs = get(metadata, "struct_definitions", Dict())

    for (name, info) in dwarf_structs
        startswith(name, "__enum__") && continue

        byte_size = _parse_size(get(info, "byte_size", "0"))
        kind = Symbol(get(info, "kind", "struct"))

        members = MemberLayout[]
        for m in get(info, "members", [])
            m_name = get(m, "name", "")
            m_type = get(m, "c_type", get(m, "type", ""))
            m_offset = _parse_size(get(m, "offset", "0"))
            m_size = _parse_size(get(m, "size", "0"))

            if m_size == 0
                cleaned = _strip_qualifiers(m_type)
                if haskey(dwarf_structs, cleaned)
                    m_size = _parse_size(get(dwarf_structs[cleaned], "byte_size", "0"))
                end
            end

            # Bitfield info from DWARF (bit_size, data_bit_offset, bit_offset_legacy)
            m_bit_size = _parse_size(get(m, "bit_size", 0))
            m_bit_offset = _parse_size(get(m, "data_bit_offset", 0))
            if m_bit_offset == 0
                m_bit_offset = _parse_size(get(m, "bit_offset_legacy", 0))
            end

            push!(members, MemberLayout(m_name, m_type, m_offset, m_size, m_bit_size, m_bit_offset))
        end

        has_vtable = get(info, "is_polymorphic", false) == true ||
                     get(info, "has_vtable", false) == true

        # Synthesize vptr if vtable present but DWARF didn't emit an explicit member
        if has_vtable && !any(m -> m.name == "__vtable_ptr" || m.name == "_vptr", members)
            pushfirst!(members, MemberLayout("__vtable_ptr", "void *", 0, 8, 0, 0))
        end

        base_raw = get(info, "base_classes", [])
        base_classes = if base_raw isa Vector
            String[b isa Dict ? string(get(b, "type", "")) : string(b) for b in base_raw]
        else
            String[]
        end

        # Detect packed layout: any member offset violates natural alignment
        is_packed = any(members) do m
            m.size > 0 && m.offset % min(max(m.size, 1), 8) != 0
        end

        # Add inheritance edges: base classes treated as by-value containment for
        # mismatch propagation (mismatched base → derived is also mismatched)
        inh = Set{String}()
        for bc in base_classes
            cleaned_bc = _strip_qualifiers(bc)
            if haskey(dwarf_structs, cleaned_bc)
                haskey(type_edges, name) && push!(type_edges[name], cleaned_bc)
                push!(inh, cleaned_bc)
            end
        end
        if !isempty(inh)
            inheritance_edges[name] = inh
        end

        types[name] = TypeNode(name, byte_size, members, kind, has_vtable, base_classes, is_packed)
    end

    return IRGraph(types, functions, type_edges, func_type_edges, inheritance_edges)
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
    type_edges, func_type_edges, functions = _build_shared_topology(metadata)
    types = Dict{String, TypeNode}()
    dwarf_structs = get(metadata, "struct_definitions", Dict())

    for (name, info) in dwarf_structs
        startswith(name, "__enum__") && continue

        kind = Symbol(get(info, "kind", "struct"))
        julia_members = MemberLayout[]
        current_offset = 0
        max_align = 1

        for m in get(info, "members", [])
            m_name = get(m, "name", "")
            m_type = get(m, "c_type", get(m, "type", ""))
            m_size = _parse_size(get(m, "size", "0"))

            if m_size == 0
                cleaned = _strip_qualifiers(m_type)
                if haskey(dwarf_structs, cleaned)
                    m_size = _parse_size(get(dwarf_structs[cleaned], "byte_size", "0"))
                end
            end

            # Julia alignment: min(sizeof(field), 8), same as get_julia_aligned_size
            if kind == :union
                push!(julia_members, MemberLayout(m_name, m_type, 0, m_size, 0, 0))
                max_align = max(max_align, min(max(m_size, 1), 8))
            else
                alignment = m_size > 8 ? 8 : m_size
                alignment = alignment == 0 ? 1 : alignment
                max_align = max(max_align, alignment)

                padding = (alignment - (current_offset % alignment)) % alignment
                current_offset += padding

                push!(julia_members, MemberLayout(m_name, m_type, current_offset, m_size, 0, 0))
                current_offset += m_size
            end
        end

        byte_size = if kind == :union
            isempty(julia_members) ? 0 : maximum(m.size for m in julia_members)
        else
            final_pad = (max_align - (current_offset % max_align)) % max_align
            current_offset + final_pad
        end

        types[name] = TypeNode(name, byte_size, julia_members, kind, false, String[], false)
    end

    return IRGraph(types, functions, type_edges, func_type_edges, Dict{String, Set{String}}())
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

        # Bitfield members — Julia cannot represent sub-byte fields
        for cm in cpp_t.members
            if cm.bit_size > 0
                push!(mismatches, DAGMismatch(name, LAYOUT_MISMATCH, UInt64(cm.offset), 0,
                    "Type '$name'.$(cm.name): bitfield ($(cm.bit_size) bits at bit offset $(cm.bit_offset)) — Julia cannot match sub-byte layout"))
                push!(mismatched_type_names, name)
            end
        end

        # Packed struct detection — C++ layout violates natural alignment but Julia assumes it
        if cpp_t.is_packed && !(name in mismatched_type_names)
            push!(mismatches, DAGMismatch(name, LAYOUT_MISMATCH, UInt64(0), 0,
                "Type '$name': packed layout detected (unnatural member alignment) — Julia uses natural alignment"))
            push!(mismatched_type_names, name)
        end
    end

    # ── Pass 2: Propagate type mismatches through containment edges ────
    # If type A contains type B by value and B is mismatched, A is also mismatched.
    # Single-pass reverse DFS from initially-mismatched types: O(V+E) vs the old
    # fixed-point loop's O(depth × E).
    reverse_edges = Dict{String, Vector{String}}()
    for (name, deps) in cpp.type_edges
        for dep in deps
            push!(get!(Vector{String}, reverse_edges, dep), name)
        end
    end
    stack = collect(copy(mismatched_type_names))
    while !isempty(stack)
        t = pop!(stack)
        for parent in get(reverse_edges, t, String[])
            parent in mismatched_type_names && continue
            push!(mismatches, DAGMismatch(parent, LAYOUT_MISMATCH, UInt64(0), 0,
                "Type '$parent' contains mismatched type '$t' by value"))
            push!(mismatched_type_names, parent)
            push!(stack, parent)
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

        # Transitive: function depends on types whose containment chains reach a mismatch.
        # Pass 2 already closed over containment edges, so mismatched_type_names is
        # transitively complete — a direct intersection suffices.
        if !func_flagged
            for dep in get(cpp.func_type_edges, mangled, Set{String}())
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
    sorted_set = Set(sorted)
    for s in thunk_symbols
        if !(s in sorted_set)
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
function _generate_dot(result::DAGDiffResult; side::Symbol=:diff, show_members::Bool=true,
                       show_only_mismatches::Bool=false)::String
    graph = side == :julia ? result.julia_graph :
            side == :cpp   ? result.cpp_graph   : result.cpp_graph

    # When filtering, compute visible nodes: mismatched + direct neighbors (1 hop)
    visible_types = Set{String}()
    visible_funcs = Set{String}()
    if show_only_mismatches
        union!(visible_types, result.mismatched_types)
        union!(visible_funcs, result.mismatched_functions)
        # Add direct neighbors for context
        for name in result.mismatched_types
            for dep in get(graph.type_edges, name, Set{String}())
                push!(visible_types, dep)
            end
        end
        for mangled in result.mismatched_functions
            for dep in get(graph.func_type_edges, mangled, Set{String}())
                push!(visible_types, dep)
            end
        end
    end

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

    # ── Type nodes (grouped by namespace for clustering) ────────────────
    # Group types by C++ namespace prefix (split on :: outside angle brackets)
    ns_groups = Dict{String, Vector{String}}()
    for (name, _) in graph.types
        show_only_mismatches && !(name in visible_types) && continue
        ns = _extract_namespace(name)
        push!(get!(Vector{String}, ns_groups, ns), name)
    end

    println(io, "  // Type nodes")
    for (ns, names) in sort!(collect(ns_groups); by=first)
        if !isempty(ns)
            safe_ns = replace(replace(ns, "::" => "_"), r"[^A-Za-z0-9_]" => "_")
            println(io, "  subgraph cluster_$safe_ns {")
            println(io, "    label=\"$(escape_dot(ns))\"; style=dashed; fontsize=9; color=gray60;")
        end

        indent = isempty(ns) ? "  " : "    "
        for name in names
            tnode = graph.types[name]
            is_mismatched = name in result.mismatched_types

            label_lines = ["<B>$(escape_dot(name))</B>"]
            push!(label_lines, "$(tnode.kind) | $(tnode.byte_size)B")

            if side == :diff && is_mismatched
                if haskey(result.julia_graph.types, name)
                    jl_size = result.julia_graph.types[name].byte_size
                    push!(label_lines, "<FONT COLOR=\"red3\">DWARF=$(tnode.byte_size)B  Julia=$(jl_size)B</FONT>")
                end
            end

            if show_members && !isempty(tnode.members)
                push!(label_lines, " ")
                for m in tnode.members
                    mline = "+$(m.offset) $(escape_dot(m.name)): $(escape_dot(m.type_name)) ($(m.size)B)"
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
            println(io, "$indent\"t:$name\" [shape=record, $fill color=$border_color, label=$label];")
        end

        if !isempty(ns)
            println(io, "  }")
        end
    end
    println(io, "")

    # ── Function nodes ─────────────────────────────────────────────────
    println(io, "  // Function nodes")
    for (mangled, fnode) in graph.functions
        show_only_mismatches && !(mangled in visible_funcs) && continue
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

    # ── Type → type edges (by-value containment + inheritance) ─────────
    println(io, "  // Type containment and inheritance edges")
    for (name, deps) in graph.type_edges
        inh_set = get(graph.inheritance_edges, name, Set{String}())
        for dep in deps
            haskey(graph.types, dep) || continue
            propagates = dep in result.mismatched_types && name in result.mismatched_types
            is_inheritance = dep in inh_set
            if is_inheritance
                color = propagates ? "blue" : "steelblue"
                style = propagates ? "bold,dashed" : "dashed"
                label = "inherits"
            else
                color = propagates ? "red3" : "gray60"
                style = propagates ? "bold" : "solid"
                label = "contains"
            end
            println(io, "  \"t:$dep\" -> \"t:$name\" [color=$color, style=$style, label=\"$label\", fontcolor=gray50];")
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
    return String(take!(io))
end

function export_dot(result::DAGDiffResult, path::String;
                    side::Symbol=:diff, show_members::Bool=true,
                    show_only_mismatches::Bool=false)
    dot_str = _generate_dot(result; side=side, show_members=show_members,
                            show_only_mismatches=show_only_mismatches)
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
                    format::String="svg", side::Symbol=:diff, show_members::Bool=true,
                    show_only_mismatches::Bool=false)
    dot_path = replace(path, r"\.(svg|png|pdf)$" => "") * ".dot"
    export_dot(result, dot_path; side=side, show_members=show_members,
               show_only_mismatches=show_only_mismatches)

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

"""
    render_html(result::DAGDiffResult, path::String; side=:diff, show_members=true)

Render an interactive HTML viewer with pan/zoom and click-to-highlight dependency chains.
SVG is rendered server-side via Graphviz and embedded inline — the output is fully
self-contained with zero external dependencies.  Works from `file://` or any HTTP server.

The JS interactivity is edge-type agnostic: when new edge kinds (scope, RAII, upstream
calls) are added to the DOT generator, highlight/trace works on them automatically.
"""
function render_html(result::DAGDiffResult, path::String;
                     side::Symbol=:diff, show_members::Bool=true,
                     show_only_mismatches::Bool=false)
    dot_str = _generate_dot(result; side=side, show_members=show_members,
                            show_only_mismatches=show_only_mismatches)

    n_types = length(result.cpp_graph.types)
    n_funcs = length(result.cpp_graph.functions)
    n_mismatches = length(result.mismatches)
    n_thunks = length(result.lowering_order)

    # Render DOT → SVG server-side via Graphviz
    svg_str = try
        mktempdir() do dir
            dp = joinpath(dir, "g.dot")
            sp = joinpath(dir, "g.svg")
            write(dp, dot_str)
            run(pipeline(`dot -Tsvg -o $sp $dp`, stderr=devnull))
            read(sp, String)
        end
    catch
        nothing
    end

    if svg_str === nothing
        dot_path = replace(path, ".html" => ".dot")
        write(dot_path, dot_str)
        println("  dag-viz: dot not found, wrote $dot_path (install graphviz to render)")
        return dot_path
    end

    # Extract <svg>…</svg> and strip fixed width/height so it fills the container
    svg_match = match(r"(<svg)([\s\S]*?)(</svg>)"s, svg_str)
    if svg_match !== nothing
        svg_tag = svg_match[1]
        svg_body = svg_match[2]
        svg_close = svg_match[3]
        # Remove width="..." height="..." so CSS controls sizing
        svg_body = replace(svg_body, r"\s+width=\"[^\"]*\"" => "")
        svg_body = replace(svg_body, r"\s+height=\"[^\"]*\"" => "")
        svg_str = svg_tag * svg_body * svg_close
    end

    io = IOBuffer()

    # ── HTML head + CSS ───────────────────────────────────────────────
    print(io, """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DAG Diff — RepliBuild</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;font-family:system-ui,-apple-system,sans-serif;background:#1a1a2e}
#bar{position:fixed;top:0;left:0;right:0;height:36px;background:#16213e;color:#e0e0e0;display:flex;align-items:center;padding:0 16px;font-size:13px;z-index:10;gap:16px;border-bottom:1px solid #0f3460}
#bar .stat{color:#94a3b8}
#bar .stat b{color:#e2e8f0}
#bar .hint{margin-left:auto;color:#64748b;font-size:11px}
#container{position:fixed;top:36px;left:0;right:0;bottom:0}
#container svg{width:100%;height:100%;cursor:grab}
#container svg.panning{cursor:grabbing}
.node.dimmed,.edge.dimmed{opacity:0.1;transition:opacity 0.2s}
.node.highlighted polygon,.node.highlighted ellipse{stroke-width:2.5px}
.edge.highlighted path,.edge.highlighted polygon{stroke-width:2px}
.node:not(.dimmed){cursor:pointer}
.node:not(.dimmed):hover polygon,.node:not(.dimmed):hover ellipse{filter:brightness(1.15)}
#tooltip{display:none;position:fixed;background:#16213e;color:#e2e8f0;padding:8px 12px;border-radius:6px;font-size:12px;max-width:360px;pointer-events:none;z-index:20;border:1px solid #0f3460;line-height:1.5}
</style>
</head>
<body>
<div id="bar">
  <span style="font-weight:600;color:#4fc3f7">DAG Diff</span>
  <span class="stat"><b>$n_types</b> types</span>
  <span class="stat"><b>$n_funcs</b> functions</span>
  <span class="stat"><b>$n_mismatches</b> mismatches</span>
  <span class="stat"><b>$n_thunks</b> thunks</span>
  <span class="hint">scroll = zoom &nbsp; drag = pan &nbsp; click node = trace chain &nbsp; Esc = reset</span>
</div>
<div id="container">
""")

    # ── Inline SVG ────────────────────────────────────────────────────
    print(io, svg_str)

    # ── JavaScript (no external deps) ─────────────────────────────────
    print(io, raw"""
</div>
<div id="tooltip"></div>
<script>
(function() {
const svg     = document.querySelector('#container svg');
const tooltip = document.getElementById('tooltip');
if (!svg) return;

// ── ViewBox state ────────────────────────────────────────────────
const vb = svg.viewBox.baseVal;
let view = { x: vb.x, y: vb.y, w: vb.width, h: vb.height };
const orig = { x: vb.x, y: vb.y, w: vb.width, h: vb.height };

function setView() {
    svg.setAttribute('viewBox', view.x + ' ' + view.y + ' ' + view.w + ' ' + view.h);
}

function svgPt(e) {
    var pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
}

// ── Zoom (scroll wheel) ─────────────────────────────────────────
svg.addEventListener('wheel', function(e) {
    e.preventDefault();
    var f = e.deltaY > 0 ? 1.12 : 1/1.12;
    var p = svgPt(e);
    view.x = p.x - (p.x - view.x) * f;
    view.y = p.y - (p.y - view.y) * f;
    view.w *= f;
    view.h *= f;
    setView();
}, { passive: false });

// ── Pan (click-drag) ────────────────────────────────────────────
var panning = false, panPt;
svg.addEventListener('mousedown', function(e) {
    if (e.button !== 0 || e.target.closest('.node')) return;
    panning = true; panPt = svgPt(e);
    svg.classList.add('panning');
});
window.addEventListener('mousemove', function(e) {
    if (!panning) return;
    var p = svgPt(e);
    view.x -= p.x - panPt.x;
    view.y -= p.y - panPt.y;
    setView();
    panPt = svgPt(e);
});
window.addEventListener('mouseup', function() {
    panning = false;
    svg.classList.remove('panning');
});

// ── Build adjacency from edge titles ─────────────────────────────
// Edge-type agnostic: works for containment, usage, scope, RAII, call edges
var fwd = {}, rev = {};
var edges = svg.querySelectorAll('.edge > title');
for (var i = 0; i < edges.length; i++) {
    var raw = edges[i].textContent.replace(/&#45;/g, '-').replace(/&gt;/g, '>');
    var m = raw.match(/^(.+?)\s*->\s*(.+)$/);
    if (!m) continue;
    var from = m[1], to = m[2];
    if (!fwd[from]) fwd[from] = [];
    if (!rev[to])   rev[to]   = [];
    fwd[from].push(to);
    rev[to].push(from);
}

// Bidirectional BFS — traces full dependency chain from a seed
function traceChain(seed) {
    var chain = {};
    chain[seed] = true;
    // Upstream
    var q = [seed];
    while (q.length) {
        var cur = q.shift();
        var deps = rev[cur] || [];
        for (var i = 0; i < deps.length; i++) {
            if (!chain[deps[i]]) { chain[deps[i]] = true; q.push(deps[i]); }
        }
    }
    // Downstream
    q = [seed];
    while (q.length) {
        var cur = q.shift();
        var deps = fwd[cur] || [];
        for (var i = 0; i < deps.length; i++) {
            if (!chain[deps[i]]) { chain[deps[i]] = true; q.push(deps[i]); }
        }
    }
    return chain;
}

// ── Click-to-highlight ──────────────────────────────────────────
function highlightChain(nodeId) {
    var chain = traceChain(nodeId);

    var nodes = svg.querySelectorAll('.node');
    for (var i = 0; i < nodes.length; i++) {
        var t = nodes[i].querySelector('title');
        var id = t ? t.textContent : '';
        if (chain[id]) {
            nodes[i].classList.remove('dimmed');
            nodes[i].classList.add('highlighted');
        } else {
            nodes[i].classList.add('dimmed');
            nodes[i].classList.remove('highlighted');
        }
    }

    var edgeEls = svg.querySelectorAll('.edge');
    for (var i = 0; i < edgeEls.length; i++) {
        var t = edgeEls[i].querySelector('title');
        var raw = t ? t.textContent.replace(/&#45;/g, '-').replace(/&gt;/g, '>') : '';
        var m = raw.match(/^(.+?)\s*->\s*(.+)$/);
        if (m && chain[m[1]] && chain[m[2]]) {
            edgeEls[i].classList.remove('dimmed');
            edgeEls[i].classList.add('highlighted');
        } else {
            edgeEls[i].classList.add('dimmed');
            edgeEls[i].classList.remove('highlighted');
        }
    }
}

function resetHighlight() {
    var all = svg.querySelectorAll('.node,.edge');
    for (var i = 0; i < all.length; i++) {
        all[i].classList.remove('dimmed', 'highlighted');
    }
    tooltip.style.display = 'none';
}

// Node click handlers
var nodeEls = svg.querySelectorAll('.node');
for (var i = 0; i < nodeEls.length; i++) {
    (function(g) {
        g.addEventListener('click', function(e) {
            e.stopPropagation();
            var t = g.querySelector('title');
            if (t) highlightChain(t.textContent);
        });
        // Hover tooltip
        g.addEventListener('mouseenter', function() {
            var t = g.querySelector('title');
            if (!t) return;
            var id = t.textContent;
            var up = (rev[id] || []).length;
            var dn = (fwd[id] || []).length;
            tooltip.innerHTML = up + ' upstream &middot; ' + dn + ' downstream';
            tooltip.style.display = 'block';
        });
        g.addEventListener('mousemove', function(e) {
            tooltip.style.left = (e.clientX + 12) + 'px';
            tooltip.style.top  = (e.clientY + 12) + 'px';
        });
        g.addEventListener('mouseleave', function() {
            tooltip.style.display = 'none';
        });
    })(nodeEls[i]);
}

// Background click or Escape to reset
svg.addEventListener('click', function(e) {
    if (!e.target.closest('.node')) resetHighlight();
});
window.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') resetHighlight();
    if (e.key === 'Home') { view.x=orig.x; view.y=orig.y; view.w=orig.w; view.h=orig.h; setView(); }
});
})();
</script>
</body>
</html>
""")

    html_str = String(take!(io))
    write(path, html_str)
    println("  dag-viz: $path (interactive)")
    return path
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

"""Extract the top-level C++ namespace from a type name, ignoring :: inside angle brackets."""
function _extract_namespace(name::AbstractString)::String
    depth = 0
    last_sep = 0
    i = 1
    while i <= ncodeunits(name)
        c = name[i]
        if c == '<'
            depth += 1
        elseif c == '>'
            depth = max(0, depth - 1)
        elseif depth == 0 && c == ':' && i < ncodeunits(name) && name[i+1] == ':'
            last_sep = i
        end
        i = nextind(name, i)
    end
    return last_sep > 0 ? name[1:prevind(name, last_sep)] : ""
end

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
