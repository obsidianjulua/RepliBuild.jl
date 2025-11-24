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

# REPL_API.jl removed in v2.0 - deprecated

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

# ============================================================================
# EXPORTS - Clean Build Orchestration API
# ============================================================================

# Core 3-function user API (THIS IS ALL YOU NEED)
export build, wrap, info

# Utility functions
export clean

# Advanced modules (for power users who know what they're doing)
export Compiler, Wrapper, Discovery, ConfigurationManager

# INTERNAL: Not for end users - these are deprecated/confusing
# export discover, import_cmake, ASTWalker, CMakeParser, WorkspaceBuilder, LLVMEnvironment
# export REPL_API, rbuild, rdiscover, rclean, rinfo, rwrap, rbuild_fast, rcompile, rparallel, rthreads, rcache_status

# ============================================================================
# PUBLIC API - Build Orchestration
# ============================================================================

# Note: discover() is now INTERNAL ONLY - build() calls it automatically
function discover(path::String="."; force::Bool=false)
    result = Discovery.discover(path, force=force)
    return result
end

"""
    build(path="."; clean=false)

Compile C++ project → library (.so/.dylib/.dll)

**What it does:**
1. Compiles your C++ code to LLVM IR
2. Links and optimizes IR
3. Generates library file
4. Extracts metadata (DWARF + symbols) for wrapping

**What it does NOT do:**
- Does NOT generate Julia wrappers (use `wrap()` for that)

# Arguments
- `path`: Project directory (default: ".")
- `clean`: Clean before building (default: false)

# Returns
Library path (String) or Dict with `:library` key

# Examples
```julia
# Build C++ library
RepliBuild.build()

# Clean build
RepliBuild.build(clean=true)

# Then generate Julia wrappers:
RepliBuild.wrap()
```
"""
function build(path::String="."; clean::Bool=false)
    println("═"^70)
    println(" RepliBuild - Compile C++")
    println("═"^70)

    original_dir = pwd()

    try
        cd(path)

        if clean
            clean_internal(path)
        end

        # Load config
        config = ConfigurationManager.load_config("replibuild.toml")

        # Compile the project (C++ → IR → library + metadata)
        library_path = Compiler.compile_project(config)

        println()
        println("✓ Library: $library_path")
        println("✓ Metadata saved")
        println()
        println("Next: RepliBuild.wrap() to generate Julia bindings")
        println("═"^70)

        return library_path

    finally
        cd(original_dir)
    end
end

"""
    wrap(path="."; headers=String[])

Generate Julia wrapper from compiled library

**What it does:**
1. Loads metadata from build (DWARF + symbols)
2. Generates Julia module with ccall wrappers
3. Creates type definitions from C++ structs
4. Saves to julia/ directory

**Requirements:**
- Must run `build()` first
- Metadata must exist in julia/compilation_metadata.json

# Arguments
- `path`: Project directory (default: ".")
- `headers`: C++ headers for advanced wrapping (optional)

# Returns
Path to generated Julia wrapper file

# Examples
```julia
# Generate wrapper from metadata
RepliBuild.wrap()

# With headers for better type info
RepliBuild.wrap(headers=["mylib.h"])
```
"""
function wrap(path::String="."; headers::Vector{String}=String[])
    println("═"^70)
    println(" RepliBuild - Generate Julia Wrappers")
    println("═"^70)

    original_dir = pwd()

    try
        cd(path)

        # Load config
        config = ConfigurationManager.load_config("replibuild.toml")

        # Find library
        output_dir = ConfigurationManager.get_output_path(config)
        lib_name = ConfigurationManager.get_library_name(config)
        library_path = joinpath(output_dir, lib_name)

        if !isfile(library_path)
            error("Library not found: $library_path\nRun RepliBuild.build() first!")
        end

        # Check for metadata
        metadata_path = joinpath(output_dir, "compilation_metadata.json")
        if !isfile(metadata_path)
            @warn "No metadata found. Wrapper quality may be limited."
        end

        println(" Library: $(basename(library_path))")
        println()

        # Generate wrapper
        wrapper_path = Wrapper.wrap_library(
            config,
            library_path,
            headers=headers,
            tier=nothing,  # Auto-detect
            generate_tests=false,
            generate_docs=true
        )

        println()
        println("✓ Wrapper: $wrapper_path")
        println()
        println("Usage:")
        module_name = ConfigurationManager.get_module_name(config)
        println("  include(\"$wrapper_path\")")
        println("  using .$module_name")
        println("═"^70)

        return wrapper_path

    finally
        cd(original_dir)
    end
end

"""
    clean(path=".")

Remove build artifacts (build/, julia/, caches)

# Examples
```julia
RepliBuild.clean()
```
"""
function clean(path::String=".")
    clean_internal(path)
end

# Internal clean function
function clean_internal(path::String)
    dirs_to_remove = ["build", "julia", ".replibuild_cache"]

    for dir in dirs_to_remove
        dir_path = joinpath(path, dir)
        if isdir(dir_path)
            rm(dir_path, recursive=true, force=true)
            println("   ✓ Removed $dir/")
        end
    end
end

"""
    info(path=".")

Show project status (config, library, wrapper)

# Examples
```julia
RepliBuild.info()
```
"""
function info(path::String=".")
    println("═"^70)
    println(" RepliBuild - Project Info")
    println("═"^70)

    # Load config
    config_file = joinpath(path, "replibuild.toml")
    if !isfile(config_file)
        println("❌ No replibuild.toml found")
        println("   Create one with RepliBuild.Discovery.discover()")
        println("═"^70)
        return
    end

    data = TOML.parsefile(config_file)
    project = get(data, "project", Dict())

    println("Project: $(get(project, "name", "unnamed"))")
    println()

    # Check library
    julia_dir = joinpath(path, "julia")
    if isdir(julia_dir)
        lib_files = filter(f -> endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll"),
                          readdir(julia_dir))
        if !isempty(lib_files)
            println("✓ Library: $(lib_files[1])")
        else
            println("❌ No library built yet - run RepliBuild.build()")
        end

        # Check wrapper
        jl_files = filter(f -> endswith(f, ".jl"), readdir(julia_dir))
        if !isempty(jl_files)
            println("✓ Wrapper: $(jl_files[1])")
        else
            println("❌ No wrapper yet - run RepliBuild.wrap()")
        end
    else
        println("❌ No build output - run RepliBuild.build()")
    end

    println("═"^70)
end

# ============================================================================
# INTERNAL HELPERS - NOT FOR PUBLIC USE
# ============================================================================
# Old build_single_project() and detect_workspace_structure() removed
# These are replaced by simple build() + wrap() API

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

function __init__()
    # Initialize RepliBuild paths and directories
    RepliBuildPaths.ensure_initialized()
end

end # module RepliBuild
