#!/usr/bin/env julia
# RepliBuild.jl - C++ → Julia Build Orchestration System
# Focus: Dependency-aware parallel compilation for single and multi-library C++ projects

module RepliBuild

using TOML

# Version
const VERSION = v"1.1.0"

# ============================================================================
# LOAD CORE MODULES
# ============================================================================

# Internal utilities (not exported)
include("RepliBuildPaths.jl")
include("LLVMEnvironment.jl")
include("ConfigurationManager.jl")
include("BuildBridge.jl")

# Core build system modules
include("ASTWalker.jl")
include("Discovery.jl")
include("CMakeParser.jl")
include("ClangJLBridge.jl")
include("Compiler.jl")      # New: replaces Bridge_LLVM
include("Wrapper.jl")       # New: wrapper generation
include("WorkspaceBuilder.jl")

# REPL-friendly API
include("REPL_API.jl")

# Import submodules for internal use
using .RepliBuildPaths
using .LLVMEnvironment
using .ConfigurationManager
using .BuildBridge
using .ASTWalker
using .Discovery
using .CMakeParser
using .ClangJLBridge
using .Compiler           # New module
using .Wrapper            # New module
using .WorkspaceBuilder
using .REPL_API

# ============================================================================
# EXPORTS - Clean Build Orchestration API
# ============================================================================

# Core 5-function API
export discover, build, import_cmake, clean, info

# Advanced modules (for power users)
export ASTWalker, Discovery, CMakeParser, WorkspaceBuilder, LLVMEnvironment

# REPL API - re-export all convenience commands
export REPL_API, rbuild, rdiscover, rclean, rinfo, rwrap,
       rbuild_fast, rcompile, rparallel, rthreads, rcache_status

# ============================================================================
# PUBLIC API - Build Orchestration
# ============================================================================

"""
    discover(path::String="."; force::Bool=false) -> Dict

Analyze C++ project structure and generate dependency graph.

# Process
1. Scan for C++ source files (.cpp, .cc, .cxx)
2. Parse #include directives and build dependency graph
3. Detect include directories
4. Find link libraries
5. Cache results for incremental builds

# Arguments
- `path`: Project root directory (default: current directory)
- `force`: Force re-discovery even if cache exists (default: false)

# Returns
Dictionary with:
- `:source_files` - List of C++ source files
- `:include_dirs` - Include directories
- `:dependency_graph` - File dependency graph
- `:build_order` - Topologically sorted compilation order

# Examples
```julia
# Analyze current project
RepliBuild.discover()

# Force re-analysis
RepliBuild.discover(".", force=true)
```
"""
function discover(path::String="."; force::Bool=false)
    println(" RepliBuild Discovery")
    println("="^70)
    println("   Project: $path")
    println()

    result = Discovery.discover(path, force=force)

    println()
    println("="^70)
    println(" Discovery complete!")

    return result
end

"""
    build(path::String="."; parallel::Bool=true, clean_first::Bool=false) -> Dict

Build C++ project with intelligent orchestration.

# Auto-Detection
- Single library: Compile → Link → Generate .so
- Multi-library workspace: Parallel compilation with dependency ordering
- Uses cached dependency graph for optimal build order

# Process
1. Load cached discovery results
2. Detect project structure (single/multi-library)
3. Build with dependency-aware parallel compilation
4. Link libraries and executables
5. Return paths to built artifacts

# Arguments
- `path`: Project root directory (default: current directory)
- `parallel`: Enable parallel compilation (default: true)
- `clean_first`: Clean before building (default: false)

# Returns
Dictionary with:
- `:libraries` - Dict mapping library names to .so paths
- `:executables` - List of executable paths
- `:build_time` - Total build time in seconds

# Examples
```julia
# Build current project
RepliBuild.build()

# Build with full rebuild
RepliBuild.build(clean_first=true)

# Sequential build (no parallelism)
RepliBuild.build(parallel=false)
```
"""
function build(path::String="."; parallel::Bool=true, clean_first::Bool=false)
    println("RepliBuild - Build Orchestration")
    println("="^70)
    println("   Project: $path")
    println()

    start_time = time()

    if clean_first
        clean(path)
    end

    # Check if workspace (multi-library) or single library
    workspace_libs = detect_workspace_structure(path)

    result = if !isempty(workspace_libs)
        # Multi-library workspace build
        println(" Detected workspace with $(length(workspace_libs)) libraries")
        println("   Libraries: $(join(workspace_libs, ", "))")
        println()
        WorkspaceBuilder.build_workspace(path, parallel=parallel)
    else
        # Single library/executable build
        println(" Detected single library project")
        println()
        build_single_project(path)
    end

    build_time = time() - start_time

    println()
    println("="^70)
    println(" Build complete! ($(round(build_time, digits=1))s)")

    result[:build_time] = build_time
    return result
end

"""
    import_cmake(path::String="."; dry_run::Bool=false) -> Dict

Import CMake project and generate replibuild.toml files.

# Process
1. Parse CMakeLists.txt recursively
2. Extract targets, dependencies, and build settings
3. Generate replibuild.toml for each library
4. Create workspace structure

# Arguments
- `path`: Project root containing CMakeLists.txt (default: current directory)
- `dry_run`: Preview without creating files (default: false)

# Returns
Dictionary mapping directories to generated configs

# Examples
```julia
# Import CMake project
RepliBuild.import_cmake()

# Preview without writing files
RepliBuild.import_cmake(dry_run=true)
```
"""
function import_cmake(path::String="."; dry_run::Bool=false)
    println("RepliBuild - CMake Import")
    println("="^70)
    println("   Project: $path")
    println()

    result = CMakeParser.cmake_replicate(path, dry_run=dry_run)

    println()
    println("="^70)
    println(" Import complete!")

    return result
end

"""
    clean(path::String=".")

Remove all build artifacts and caches.

# Removes
- build/ directory
- julia/ directory (generated bindings)
- .replibuild_cache/ directory
- *.ll (LLVM IR files)
- *.o (object files)

# Examples
```julia
# Clean current project
RepliBuild.clean()

# Clean specific project
RepliBuild.clean("/path/to/project")
```
"""
function clean(path::String=".")
    println("Cleaning build artifacts...")

    dirs_to_remove = ["build", "julia", ".replibuild_cache"]

    for dir in dirs_to_remove
        dir_path = joinpath(path, dir)
        if isdir(dir_path)
            rm(dir_path, recursive=true, force=true)
            println("   ✓ Removed $dir/")
        end
    end

    # Also clean in subdirectories (workspace)
    for entry in readdir(path)
        entry_path = joinpath(path, entry)
        if isdir(entry_path) && !startswith(entry, ".")
            for dir in dirs_to_remove
                subdir_path = joinpath(entry_path, dir)
                if isdir(subdir_path)
                    rm(subdir_path, recursive=true, force=true)
                    println("   ✓ Removed $entry/$dir/")
                end
            end
        end
    end

    println("    Clean complete!")
    println()
end

"""
    info(path::String=".")

Display project build status and configuration.

# Shows
- Project structure (single/multi-library)
- Cached discovery results
- Build artifacts
- Dependency graph statistics
- Last build time

# Examples
```julia
# Show current project info
RepliBuild.info()
```
"""
function info(path::String=".")
    println("ℹ️  RepliBuild - Project Information")
    println("="^70)
    println("   Project: $path")
    println()

    # Load config if exists
    config_file = joinpath(path, "replibuild.toml")
    if isfile(config_file)
        data = TOML.parsefile(config_file)
        project = get(data, "project", Dict())
        println("Configuration:")
        println("   Name: $(get(project, "name", "unnamed"))")
        println("   Root: $(get(project, "root", path))")
        println()
    end

    # Check for cache
    cache_dir = joinpath(path, ".replibuild_cache")
    if isdir(cache_dir)
        cache_file = joinpath(cache_dir, "build_cache.toml")
        if isfile(cache_file)
            cache = TOML.parsefile(cache_file)
            discovery = get(cache, "discovery_results", Dict())

            println(" Discovery Results:")
            files = get(discovery, "files", Dict())
            cpp_count = length(get(files, "cpp_sources", []))
            hdr_count = length(get(files, "cpp_headers", []))
            println("   C++ Sources: $cpp_count")
            println("   Headers: $hdr_count")

            # Show dependency graph stats
            graph_file = get(discovery, "dependency_graph_file", "")
            if !isempty(graph_file)
                graph_path = joinpath(path, graph_file)
                if isfile(graph_path)
                    println("   Dependency graph: ✓ $(basename(graph_file))")
                end
            end
            println()
        end
    end

    # Check for workspace
    workspace_libs = detect_workspace_structure(path)
    if !isempty(workspace_libs)
        println("   Workspace Structure:")
        println("   Type: Multi-library")
        println("   Libraries: $(length(workspace_libs))")
        for lib in workspace_libs
            println("      • $lib")
        end
        println()
    else
        println("   Project Type: Single library")
        println()
    end

    # Check for build artifacts
    build_dir = joinpath(path, "build")
    if isdir(build_dir)
        ir_dir = joinpath(build_dir, "ir")
        if isdir(ir_dir)
            ll_files = filter(f -> endswith(f, ".ll"), readdir(ir_dir))
            if !isempty(ll_files)
                println("Build Artifacts:")
                println("LLVM IR files: $(length(ll_files))")
            end
        end
    end

    println("="^70)
end

# ============================================================================
# INTERNAL HELPERS
# ============================================================================

"""
Detect if path contains a multi-library workspace
"""
function detect_workspace_structure(path::String)
    libraries = String[]

    if !isdir(path)
        return libraries
    end

    for entry in readdir(path)
        # Skip hidden directories
        if startswith(entry, ".")
            continue
        end

        entry_path = joinpath(path, entry)
        if isdir(entry_path)
            toml_path = joinpath(entry_path, "replibuild.toml")
            if isfile(toml_path)
                push!(libraries, entry)
            end
        end
    end

    return libraries
end

"""
Build a single library/executable project
"""
function build_single_project(path::String)
    original_dir = pwd()

    try
        cd(path)

        # Load configuration
        config = ConfigurationManager.load_config("replibuild.toml")

        # Compile
        result = Compiler.compile_project(config)

        # Find output files
        libraries = Dict{String,String}()
        executables = String[]

        # Scan output directories
        for dir in ["julia", "build", "."]
            full_dir = joinpath(path, dir)
            if isdir(full_dir)
                for file in readdir(full_dir)
                    full_path = joinpath(full_dir, file)
                    if isfile(full_path)
                        if endswith(file, ".so") || endswith(file, ".dylib") || endswith(file, ".dll")
                            libraries[basename(file)] = full_path
                        elseif !endswith(file, ".ll") && !endswith(file, ".o") && !endswith(file, ".toml")
                            # Check if executable
                            try
                                stat_result = stat(full_path)
                                if stat_result.mode & 0o111 != 0
                                    push!(executables, full_path)
                                end
                            catch
                            end
                        end
                    end
                end
            end
        end

        return Dict(
            :libraries => libraries,
            :executables => executables
        )

    finally
        cd(original_dir)
    end
end

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

function __init__()
    # Initialize RepliBuild paths and directories
    RepliBuildPaths.ensure_initialized()
end

end # module RepliBuild
