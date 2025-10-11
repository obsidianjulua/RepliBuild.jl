#!/usr/bin/env julia
"""
Discovery Daemon Server - Fast binary/dependency finding with aggressive caching

Start with: julia --project=.. discovery_daemon.jl
Port: 3001

Features:
- Cached LLVM tool locations (persistent across runs)
- Memoized AST dependency graphs
- Parallel file scanning with Distributed.jl
- Binary detection with caching
"""

using DaemonMode
using Distributed
using Dates

# Add project to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))

using RepliBuild
using RepliBuild.Discovery
using RepliBuild.ConfigurationManager
using RepliBuild.LLVMEnvironment
using RepliBuild.ASTWalker

const PORT = 3001

# ============================================================================
# PERSISTENT CACHES
# ============================================================================

const TOOL_CACHE = Dict{String, String}()  # tool_name => path
const AST_CACHE = Dict{String, Any}()      # project_hash => dependency_graph
const BINARY_CACHE = Dict{String, Vector}() # dir_hash => binaries
const FILE_SCAN_CACHE = Dict{String, Any}() # dir_hash => scan_results

"""
Compute hash of directory for cache invalidation
"""
function dir_hash(path::String)
    # Simple hash based on path + modification times of key files
    h = hash(path)

    # Include replibuild.toml mtime if it exists
    config_path = joinpath(path, "replibuild.toml")
    if isfile(config_path)
        h = hash((h, mtime(config_path)))
    end

    return string(h, base=16)
end

"""
Initialize LLVM tool cache from RepliBuild's LLVM directory
"""
function init_tool_cache!()
    println("[DISCOVERY] Initializing LLVM tool cache...")

    llvm_root = "/home/grim/.julia/julia/RepliBuild/LLVM"
    tools_dir = joinpath(llvm_root, "tools")

    if !isdir(tools_dir)
        @warn "LLVM tools directory not found: $tools_dir"
        return
    end

    # Cache all tools
    for tool in readdir(tools_dir)
        tool_path = joinpath(tools_dir, tool)
        if isfile(tool_path) && !islink(tool_path)
            TOOL_CACHE[tool] = tool_path
            # Also cache without version suffixes
            base_name = replace(tool, r"-\d+(\.\d+)*$" => "")
            if base_name != tool
                TOOL_CACHE[base_name] = tool_path
            end
        end
    end

    println("[DISCOVERY] Cached $(length(TOOL_CACHE)) LLVM tools")
end

# ============================================================================
# DISCOVERY FUNCTIONS
# ============================================================================

"""
Fast file scanning with caching
"""
function scan_files(args::Dict)
    target_dir = get(args, "path", pwd())
    force = get(args, "force", false)

    println("[DISCOVERY] Scanning files: $target_dir")

    # Check cache
    cache_key = dir_hash(target_dir)
    if !force && haskey(FILE_SCAN_CACHE, cache_key)
        println("[DISCOVERY] Using cached scan results")
        return Dict(
            :success => true,
            :cached => true,
            :results => FILE_SCAN_CACHE[cache_key]
        )
    end

    try
        # Use Discovery module's scan function
        scan_results = Discovery.scan_all_files(target_dir)

        # Cache results
        FILE_SCAN_CACHE[cache_key] = scan_results

        return Dict(
            :success => true,
            :cached => false,
            :results => scan_results,
            :summary => Dict(
                "cpp_sources" => length(scan_results.cpp_sources),
                "cpp_headers" => length(scan_results.cpp_headers),
                "c_sources" => length(scan_results.c_sources),
                "c_headers" => length(scan_results.c_headers),
                "total" => scan_results.total_files
            )
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e),
            :stacktrace => sprint(showerror, e, catch_backtrace())
        )
    end
end

"""
Fast binary detection with caching
"""
function detect_binaries(args::Dict)
    target_dir = get(args, "path", pwd())
    force = get(args, "force", false)

    println("[DISCOVERY] Detecting binaries: $target_dir")

    # Check cache
    cache_key = dir_hash(target_dir)
    if !force && haskey(BINARY_CACHE, cache_key)
        println("[DISCOVERY] Using cached binary results")
        return Dict(
            :success => true,
            :cached => true,
            :binaries => BINARY_CACHE[cache_key]
        )
    end

    try
        # Need scan results first
        scan_result = scan_files(Dict("path" => target_dir, "force" => force))
        if !scan_result[:success]
            return scan_result
        end

        scan_data = scan_result[:results]
        binaries = Discovery.detect_all_binaries(target_dir, scan_data)

        # Cache results
        BINARY_CACHE[cache_key] = binaries

        return Dict(
            :success => true,
            :cached => false,
            :binaries => binaries,
            :count => length(binaries)
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e)
        )
    end
end

"""
Build AST dependency graph with caching
"""
function walk_ast_dependencies(args::Dict)
    target_dir = get(args, "path", pwd())
    include_dirs = get(args, "include_dirs", String[])
    force = get(args, "force", false)

    println("[DISCOVERY] Walking AST dependencies: $target_dir")

    # Check cache
    cache_key = dir_hash(target_dir)
    if !force && haskey(AST_CACHE, cache_key)
        println("[DISCOVERY] Using cached AST dependency graph")
        return Dict(
            :success => true,
            :cached => true,
            :graph => AST_CACHE[cache_key]
        )
    end

    try
        # Get scan results
        scan_result = scan_files(Dict("path" => target_dir, "force" => force))
        if !scan_result[:success]
            return scan_result
        end

        scan_data = scan_result[:results]

        # Build include dirs if not provided
        if isempty(include_dirs)
            include_dirs = Discovery.build_include_dirs(target_dir, scan_data)
        end

        # Walk dependencies using cached clang path
        clang_path = get(TOOL_CACHE, "clang++", "")
        if isempty(clang_path)
            @warn "Clang not found in cache, using system"
            clang_path = LLVMEnvironment.get_tool("clang++")
        end

        # Get all source files
        all_sources = vcat(
            [joinpath(target_dir, f) for f in scan_data.cpp_sources],
            [joinpath(target_dir, f) for f in scan_data.c_sources],
            [joinpath(target_dir, f) for f in scan_data.cpp_headers],
            [joinpath(target_dir, f) for f in scan_data.c_headers]
        )

        if isempty(all_sources)
            return Dict(
                :success => true,
                :graph => nothing,
                :message => "No source files found"
            )
        end

        # Build dependency graph
        dep_graph = ASTWalker.build_dependency_graph(
            all_sources,
            include_dirs,
            use_clang=true,
            clang_path=clang_path
        )

        # Cache results
        AST_CACHE[cache_key] = dep_graph

        return Dict(
            :success => true,
            :cached => false,
            :graph => dep_graph,
            :nodes => length(dep_graph.nodes),
            :edges => length(dep_graph.edges)
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e),
            :stacktrace => sprint(showerror, e, catch_backtrace())
        )
    end
end

"""
Get cached LLVM tool path
"""
function get_tool(args::Dict)
    tool_name = get(args, "tool", "")

    if isempty(tool_name)
        return Dict(
            :success => false,
            :error => "Tool name required"
        )
    end

    path = get(TOOL_CACHE, tool_name, "")

    if isempty(path)
        # Try to find it
        path = LLVMEnvironment.get_tool(tool_name)
        if !isempty(path)
            TOOL_CACHE[tool_name] = path
        end
    end

    return Dict(
        :success => !isempty(path),
        :tool => tool_name,
        :path => path
    )
end

"""
Get all cached tools
"""
function get_all_tools(args::Dict)
    return Dict(
        :success => true,
        :tools => TOOL_CACHE,
        :count => length(TOOL_CACHE)
    )
end

"""
Full discovery pipeline (combines all steps)
"""
function discover_project(args::Dict)
    target_dir = get(args, "path", pwd())
    force = get(args, "force", false)

    println("[DISCOVERY] Running full discovery pipeline: $target_dir")

    results = Dict{Symbol, Any}()

    try
        # Stage 1: Scan files
        println("[DISCOVERY] Stage 1: File scanning...")
        scan_result = scan_files(Dict("path" => target_dir, "force" => force))
        results[:scan] = scan_result

        if !scan_result[:success]
            return Dict(
                :success => false,
                :stage => "scan",
                :error => scan_result[:error]
            )
        end

        # Stage 2: Detect binaries
        println("[DISCOVERY] Stage 2: Binary detection...")
        binary_result = detect_binaries(Dict("path" => target_dir, "force" => force))
        results[:binaries] = binary_result

        # Stage 3: Build include directories
        println("[DISCOVERY] Stage 3: Include directories...")
        scan_data = scan_result[:results]
        include_dirs = Discovery.build_include_dirs(target_dir, scan_data)
        results[:include_dirs] = include_dirs

        # Stage 4: AST dependency walking
        println("[DISCOVERY] Stage 4: AST dependency graph...")
        ast_result = walk_ast_dependencies(Dict(
            "path" => target_dir,
            "include_dirs" => include_dirs,
            "force" => force
        ))
        results[:ast] = ast_result

        println("[DISCOVERY] Discovery complete!")

        return Dict(
            :success => true,
            :results => results,
            :cached => scan_result[:cached] && binary_result[:cached] && ast_result[:cached]
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e),
            :stacktrace => sprint(showerror, e, catch_backtrace()),
            :partial_results => results
        )
    end
end

"""
Clear all caches
"""
function clear_caches(args::Dict)
    println("[DISCOVERY] Clearing all caches...")

    empty!(FILE_SCAN_CACHE)
    empty!(BINARY_CACHE)
    empty!(AST_CACHE)
    # Keep TOOL_CACHE - tools don't change

    return Dict(
        :success => true,
        :message => "Caches cleared"
    )
end

"""
Get cache statistics
"""
function cache_stats(args::Dict)
    return Dict(
        :success => true,
        :stats => Dict(
            "tools" => length(TOOL_CACHE),
            "file_scans" => length(FILE_SCAN_CACHE),
            "binaries" => length(BINARY_CACHE),
            "ast_graphs" => length(AST_CACHE)
        )
    )
end

# ============================================================================
# MAIN
# ============================================================================

"""
Main daemon server function
"""
function main()
    println("="^70)
    println("RepliBuild Discovery Daemon Server")
    println("Port: $PORT")
    println("="^70)

    # Initialize tool cache
    init_tool_cache!()

    println()
    println("Available Functions:")
    println("  • scan_files(path, force=false)")
    println("  • detect_binaries(path, force=false)")
    println("  • walk_ast_dependencies(path, include_dirs=[], force=false)")
    println("  • discover_project(path, force=false) - Full pipeline")
    println("  • get_tool(tool)")
    println("  • get_all_tools()")
    println("  • cache_stats()")
    println("  • clear_caches()")
    println()
    println("Ready to accept discovery requests...")
    println("="^70)

    # Start the daemon server
    serve(PORT)
end

# Start the daemon if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
