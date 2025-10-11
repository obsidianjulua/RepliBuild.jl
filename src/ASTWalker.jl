#!/usr/bin/env julia
# ASTWalker.jl - AST-based dependency analysis using Clang
# Walks C/C++ source files to extract include dependencies, symbols, and structure
# Uses LLVMEnvironment's isolated clang for parsing

module ASTWalker

using JSON
using ProgressMeter

"""
Dependency information for a source file
"""
mutable struct FileDependencies
    filepath::String
    includes::Vector{String}           # Direct #include statements
    resolved_includes::Vector{String}  # Resolved absolute paths
    symbols_defined::Vector{String}    # Functions, classes, variables defined
    symbols_used::Vector{String}       # External symbols referenced
    namespaces::Vector{String}
    classes::Vector{String}
    functions::Vector{String}
    is_header::Bool
    parse_errors::Vector{String}
end

"""
Complete dependency graph for a project
"""
struct DependencyGraph
    files::Dict{String,FileDependencies}
    include_graph::Dict{String,Vector{String}}  # file -> files it includes
    reverse_graph::Dict{String,Vector{String}}  # file -> files that include it
    compilation_order::Vector{String}           # Topologically sorted
end

"""
    extract_includes_simple(filepath::String) -> Vector{String}

Fast regex-based include extraction (doesn't require clang).
Used as fallback if AST parsing fails.
"""
function extract_includes_simple(filepath::String)
    if !isfile(filepath)
        return String[]
    end

    includes = String[]

    try
        content = read(filepath, String)

        # Match #include "..." and #include <...>
        regex = r"^\s*#\s*include\s+[<\"]([^>\"]+)[>\"]"m
        for m in eachmatch(regex, content)
            push!(includes, m.captures[1])
        end
    catch e
        @warn "Failed to read file: $filepath" exception=e
    end

    return unique(includes)
end

"""
    extract_includes_clang(filepath::String, clang_path::String, include_dirs::Vector{String}) -> FileDependencies

Use clang -E preprocessor to get detailed include information.
"""
function extract_includes_clang(filepath::String, clang_path::String, include_dirs::Vector{String})
    deps = FileDependencies(
        filepath,
        String[],
        String[],
        String[],
        String[],
        String[],
        String[],
        String[],
        endswith(filepath, ".h") || endswith(filepath, ".hpp"),
        String[]
    )

    # Build include flags
    include_flags = ["-I$dir" for dir in include_dirs]

    # Determine language
    ext = lowercase(splitext(filepath)[2])
    lang_flag = ext in [".cpp", ".cc", ".cxx", ".hpp"] ? "-xc++" : "-xc"

    try
        # Run clang -E to get preprocessed output with line markers
        cmd = `$clang_path -E $lang_flag $include_flags -H $filepath`

        # Capture stderr for -H output (include tree)
        output = IOBuffer()
        err_output = IOBuffer()

        process = run(pipeline(cmd, stdout=output, stderr=err_output), wait=false)
        wait(process)

        if process.exitcode == 0
            # Parse stderr for include hierarchy (-H output)
            err_str = String(take!(err_output))

            for line in split(err_str, '\n')
                # -H output format: ". /path/to/header" or ".. /path/to/nested"
                m = match(r"^\.+\s+(.+)$", line)
                if m !== nothing
                    included_file = strip(m.captures[1])
                    push!(deps.resolved_includes, included_file)
                end
            end
        else
            push!(deps.parse_errors, "Clang preprocessing failed with exit code $(process.exitcode)")
        end

    catch e
        push!(deps.parse_errors, "Failed to run clang: $e")
        @warn "Clang preprocessing failed for $filepath" exception=e
    end

    # Also get direct includes via simple regex
    deps.includes = extract_includes_simple(filepath)

    return deps
end

"""
    extract_symbols_nm(filepath::String, nm_path::String) -> (Vector{String}, Vector{String})

Extract defined and used symbols from compiled object file using nm.
Returns (defined_symbols, undefined_symbols).
"""
function extract_symbols_nm(filepath::String, nm_path::String)
    defined = String[]
    undefined = String[]

    try
        result = read(`$nm_path -C $filepath`, String)

        for line in split(result, '\n')
            if isempty(strip(line))
                continue
            end

            parts = split(strip(line))

            if length(parts) >= 2
                symbol_type = parts[1]
                symbol_name = join(parts[2:end], " ")

                # Defined symbols: T (text), D (data), B (bss)
                if symbol_type in ["T", "t", "D", "d", "B", "b"]
                    push!(defined, symbol_name)
                # Undefined symbols: U
                elseif symbol_type == "U"
                    push!(undefined, symbol_name)
                end
            end
        end
    catch e
        @debug "nm extraction failed for $filepath" exception=e
    end

    return (unique(defined), unique(undefined))
end

"""
    parse_source_structure(filepath::String) -> FileDependencies

Parse C++ source structure using regex (lightweight alternative to full AST).
Extracts namespaces, classes, and functions.
"""
function parse_source_structure(filepath::String)
    deps = FileDependencies(
        filepath,
        String[],
        String[],
        String[],
        String[],
        String[],
        String[],
        String[],
        endswith(filepath, ".h") || endswith(filepath, ".hpp"),
        String[]
    )

    if !isfile(filepath)
        push!(deps.parse_errors, "File not found")
        return deps
    end

    try
        content = read(filepath, String)

        # Extract namespaces
        for m in eachmatch(r"namespace\s+(\w+)", content)
            push!(deps.namespaces, m.captures[1])
        end

        # Extract classes/structs
        for m in eachmatch(r"\b(?:class|struct)\s+(\w+)", content)
            class_name = m.captures[1]
            # Skip forward declarations and common keywords
            if !(class_name in ["final", "alignas", "public", "private", "protected"])
                push!(deps.classes, class_name)
            end
        end

        # Extract function definitions (simplified)
        # Look for patterns like: type name(...) {
        for m in eachmatch(r"(\w+)\s+(\w+)\s*\([^)]*\)\s*\{", content)
            return_type = m.captures[1]
            func_name = m.captures[2]

            # Skip common keywords that might match
            if !(return_type in ["if", "while", "for", "switch"]) &&
               !(func_name in ["if", "while", "for", "switch"])
                push!(deps.functions, func_name)
            end
        end

    catch e
        push!(deps.parse_errors, "Parse error: $e")
        @warn "Failed to parse source structure: $filepath" exception=e
    end

    # Get includes
    deps.includes = extract_includes_simple(filepath)

    return deps
end

"""
    resolve_include_path(include_str::String, source_file::String, include_dirs::Vector{String}) -> String

Resolve an #include statement to an absolute path.
"""
function resolve_include_path(include_str::String, source_file::String, include_dirs::Vector{String})
    # Try relative to source file first (for #include "...")
    source_dir = dirname(abspath(source_file))
    relative_path = joinpath(source_dir, include_str)

    if isfile(relative_path)
        return abspath(relative_path)
    end

    # Try each include directory
    for inc_dir in include_dirs
        candidate = joinpath(inc_dir, include_str)
        if isfile(candidate)
            return abspath(candidate)
        end
    end

    # Not found
    return ""
end

"""
    build_dependency_graph(files::Vector{String}, include_dirs::Vector{String};
                          use_clang::Bool=true, clang_path::String="") -> DependencyGraph

Build complete dependency graph for source files.

# Arguments
- `files`: List of source files to analyze
- `include_dirs`: Include search paths
- `use_clang`: Use clang for detailed parsing (requires clang_path)
- `clang_path`: Path to clang++ executable

# Returns
- `DependencyGraph`: Complete dependency information
"""
function build_dependency_graph(files::Vector{String}, include_dirs::Vector{String};
                                use_clang::Bool=true, clang_path::String="")
    println("🔍 Building dependency graph for $(length(files)) files...")

    file_deps = Dict{String,FileDependencies}()

    # Analyze each file with progress bar
    p = Progress(length(files), desc="   Analyzing dependencies: ")
    for (i, filepath) in enumerate(files)
        abs_path = abspath(filepath)

        # Choose parsing method
        if use_clang && !isempty(clang_path) && isfile(clang_path)
            deps = extract_includes_clang(filepath, clang_path, include_dirs)
            # Enhance with structure parsing
            struct_info = parse_source_structure(filepath)
            deps.namespaces = struct_info.namespaces
            deps.classes = struct_info.classes
            deps.functions = struct_info.functions
        else
            deps = parse_source_structure(filepath)
        end

        # Resolve include paths
        for inc in deps.includes
            resolved = resolve_include_path(inc, filepath, include_dirs)
            if !isempty(resolved)
                push!(deps.resolved_includes, resolved)
            end
        end

        file_deps[abs_path] = deps
        next!(p)
    end

    # Build include graph
    include_graph = Dict{String,Vector{String}}()
    reverse_graph = Dict{String,Vector{String}}()

    for (file, deps) in file_deps
        include_graph[file] = unique(deps.resolved_includes)

        # Build reverse graph
        for included in deps.resolved_includes
            if !haskey(reverse_graph, included)
                reverse_graph[included] = String[]
            end
            push!(reverse_graph[included], file)
        end
    end

    # Topological sort for compilation order
    compilation_order = topological_sort(include_graph)

    println("   ✅ Dependency graph built:")
    println("      Files analyzed: $(length(file_deps))")
    println("      Include relationships: $(sum(length(v) for v in values(include_graph)))")

    return DependencyGraph(
        file_deps,
        include_graph,
        reverse_graph,
        compilation_order
    )
end

"""
    topological_sort(graph::Dict{String,Vector{String}}) -> Vector{String}

Topological sort of dependency graph to determine compilation order.
"""
function topological_sort(graph::Dict{String,Vector{String}})
    # Kahn's algorithm
    in_degree = Dict{String,Int}()
    nodes = Set{String}()

    # Collect all nodes
    for (node, deps) in graph
        push!(nodes, node)
        for dep in deps
            push!(nodes, dep)
        end
    end

    # Calculate in-degrees
    for node in nodes
        in_degree[node] = 0
    end

    for (node, deps) in graph
        for dep in deps
            in_degree[dep] = get(in_degree, dep, 0) + 1
        end
    end

    # Queue of nodes with no dependencies
    queue = [node for node in nodes if in_degree[node] == 0]
    result = String[]

    while !isempty(queue)
        node = popfirst!(queue)
        push!(result, node)

        # Reduce in-degree for dependents
        if haskey(graph, node)
            for dep in graph[node]
                in_degree[dep] -= 1
                if in_degree[dep] == 0
                    push!(queue, dep)
                end
            end
        end
    end

    return result
end

"""
    export_dependency_graph_json(graph::DependencyGraph, output_path::String)

Export dependency graph to JSON for visualization or further processing.
"""
function export_dependency_graph_json(graph::DependencyGraph, output_path::String)
    data = Dict(
        "files" => Dict(
            file => Dict(
                "includes" => deps.includes,
                "resolved_includes" => deps.resolved_includes,
                "namespaces" => deps.namespaces,
                "classes" => deps.classes,
                "functions" => deps.functions,
                "is_header" => deps.is_header,
                "errors" => deps.parse_errors
            )
            for (file, deps) in graph.files
        ),
        "include_graph" => graph.include_graph,
        "reverse_graph" => graph.reverse_graph,
        "compilation_order" => graph.compilation_order
    )

    open(output_path, "w") do f
        JSON.print(f, data, 2)
    end

    println("📝 Exported dependency graph: $output_path")
end

"""
    print_dependency_summary(graph::DependencyGraph)

Print summary statistics of the dependency graph.
"""
function print_dependency_summary(graph::DependencyGraph)
    println("="^70)
    println("Dependency Graph Summary")
    println("="^70)

    total_files = length(graph.files)
    header_count = count(d -> d.is_header, values(graph.files))
    source_count = total_files - header_count

    total_includes = sum(length(d.includes) for d in values(graph.files))
    total_namespaces = length(unique(vcat([d.namespaces for d in values(graph.files)]...)))
    total_classes = length(unique(vcat([d.classes for d in values(graph.files)]...)))
    total_functions = length(unique(vcat([d.functions for d in values(graph.files)]...)))

    files_with_errors = count(d -> !isempty(d.parse_errors), values(graph.files))

    println("\n📊 File Statistics:")
    println("   Total files:     $total_files")
    println("   Headers:         $header_count")
    println("   Sources:         $source_count")
    println("   With errors:     $files_with_errors")

    println("\n📚 Dependency Statistics:")
    println("   Total includes:  $total_includes")
    println("   Avg per file:    $(round(total_includes / max(total_files, 1), digits=1))")

    println("\n🏗️  Structure Statistics:")
    println("   Namespaces:      $total_namespaces")
    println("   Classes:         $total_classes")
    println("   Functions:       $total_functions")

    println("\n📋 Compilation Order:")
    println("   $(length(graph.compilation_order)) files in dependency order")

    if files_with_errors > 0
        println("\n⚠️  Errors:")
        for (file, deps) in graph.files
            if !isempty(deps.parse_errors)
                println("   ❌ $(basename(file)):")
                for err in deps.parse_errors
                    println("      • $err")
                end
            end
        end
    end

    println("="^70)
end

# Exports
export FileDependencies, DependencyGraph,
       build_dependency_graph,
       extract_includes_simple, extract_includes_clang,
       parse_source_structure,
       resolve_include_path,
       export_dependency_graph_json,
       print_dependency_summary

end # module ASTWalker
