#!/usr/bin/env julia
# RepliBuild.jl - C++ → Julia Build Orchestration System
# Focus: Dependency-aware parallel compilation for single and multi-library C++ projects

module RepliBuild

using TOML

# Version
const VERSION = v"2.0.3"

# ============================================================================
# LOAD CORE MODULES
# ============================================================================

# Internal utilities (not exported)
include("LLVMEnvironment.jl")
include("ConfigurationManager.jl")
include("BuildBridge.jl")

# Core build system modules
include("ASTWalker.jl")
include("Discovery.jl")
include("ClangJLBridge.jl")
include("Compiler.jl")
include("Wrapper.jl")
include("DWARFParser.jl")
include("JLCSIRGenerator.jl")
include("MLIRNative.jl")

# Introspection module
include("Introspect.jl")

# Import submodules for internal use
using .LLVMEnvironment
using .ConfigurationManager
using .BuildBridge
using .ASTWalker
using .Discovery
using .ClangJLBridge
using .Compiler
using .Wrapper
using .DWARFParser
using .JLCSIRGenerator
using .MLIRNative
using .Introspect

# ============================================================================
# EXPORTS - Clean Build Orchestration API
# ============================================================================

# Core 3-function user API (THIS IS ALL YOU NEED)
export build, wrap, info

# Discovery function for setup
export discover

# Utility functions
export clean

# Advanced modules (for power users who know what they're doing)
export Compiler, Wrapper, Discovery, ConfigurationManager, DWARFParser, JLCSIRGenerator, MLIRNative

# Introspection API
export Introspect

# ============================================================================
# PUBLIC API - Build Orchestration
# ============================================================================

"""
    discover(target_dir="."; force=false, build=false, wrap=false) -> String

Scan C++ project and generate replibuild.toml configuration file.

**This is the entry point for new projects.** Run this first to set up RepliBuild.

# Arguments
- `target_dir`: Project directory to scan (default: current directory)
- `force`: Force rediscovery even if replibuild.toml exists (default: false)
- `build`: Automatically run build() after discovery (default: false)
- `wrap`: Automatically run wrap() after build (requires build=true, default: false)

# Returns
Path to generated `replibuild.toml` file

# Workflow

## Basic workflow (step-by-step):
```julia
# 1. Discover and create config
toml_path = RepliBuild.discover()

# 2. Build the library
RepliBuild.build(toml_path)

# 3. Generate Julia wrappers
RepliBuild.wrap(toml_path)
```

## Chained workflow (automated):
```julia
# Discover → Build → Wrap (all at once)
toml_path = RepliBuild.discover(build=true, wrap=true)

# Or just discover and build
toml_path = RepliBuild.discover(build=true)
```

# Examples
```julia
# Discover current directory
RepliBuild.discover()

# Discover another directory
RepliBuild.discover("path/to/cpp/project")

# Force regenerate config
RepliBuild.discover(force=true)

# Full automated pipeline
RepliBuild.discover(build=true, wrap=true)
```
"""
function discover(path::String="."; force::Bool=false, build::Bool=false, wrap::Bool=false)
    result = Discovery.discover(path, force=force, build=build, wrap=wrap)
    return result
end

"""
    build(toml_path="replibuild.toml"; clean=false)

Compile C++ project → library (.so/.dylib/.dll)

**What it does:**
1. Compiles your C++ code to LLVM IR
2. Links and optimizes IR
3. Generates library file
4. Extracts metadata (DWARF + symbols) for wrapping

**What it does NOT do:**
- Does NOT generate Julia wrappers (use `wrap()` for that)

# Arguments
- `toml_path`: Path to replibuild.toml configuration file (default: "replibuild.toml")
- `clean`: Clean before building (default: false)

# Returns
Library path (String)

# Examples
```julia
# Build using replibuild.toml in current directory
RepliBuild.build()

# Build with specific config file
RepliBuild.build("path/to/replibuild.toml")

# Clean build
RepliBuild.build(clean=true)

# Then generate Julia wrappers:
RepliBuild.wrap("replibuild.toml")
```
"""
function build(toml_path::String="replibuild.toml"; clean::Bool=false)
    println("═"^70)
    println(" RepliBuild - Compile C++")
    println("═"^70)

    # Resolve absolute path to TOML file
    toml_path = abspath(toml_path)

    if !isfile(toml_path)
        error("Configuration file not found: $toml_path\nRun RepliBuild.Discovery.discover() first!")
    end

    project_dir = dirname(toml_path)
    original_dir = pwd()

    try
        cd(project_dir)

        if clean
            clean_internal(project_dir)
        end

        # Load config
        config = ConfigurationManager.load_config(toml_path)

        # Compile the project (C++ → IR → library + metadata)
        library_path = Compiler.compile_project(config)

        println()
        println("✓ Library: $library_path")
        println("✓ Metadata saved")
        println()
        println("Next: RepliBuild.wrap(\"$toml_path\") to generate Julia bindings")
        println("═"^70)

        return library_path

    finally
        cd(original_dir)
    end
end

"""
    wrap(toml_path="replibuild.toml"; headers=String[])

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
- `toml_path`: Path to replibuild.toml configuration file (default: "replibuild.toml")
- `headers`: C++ headers for advanced wrapping (optional)

# Returns
Path to generated Julia wrapper file

# Examples
```julia
# Generate wrapper using replibuild.toml in current directory
RepliBuild.wrap()

# Generate wrapper with specific config file
RepliBuild.wrap("path/to/replibuild.toml")

# With headers for better type info
RepliBuild.wrap("replibuild.toml", headers=["mylib.h"])
```
"""
function wrap(toml_path::String="replibuild.toml"; headers::Vector{String}=String[])
    println("═"^70)
    println(" RepliBuild - Generate Julia Wrappers")
    println("═"^70)

    # Resolve absolute path to TOML file
    toml_path = abspath(toml_path)

    if !isfile(toml_path)
        error("Configuration file not found: $toml_path\nRun RepliBuild.Discovery.discover() first!")
    end

    project_dir = dirname(toml_path)
    original_dir = pwd()

    try
        cd(project_dir)

        # Load config
        config = ConfigurationManager.load_config(toml_path)

        # Find library
        output_dir = ConfigurationManager.get_output_path(config)
        lib_name = ConfigurationManager.get_library_name(config)
        library_path = joinpath(output_dir, lib_name)

        if !isfile(library_path)
            error("Library not found: $library_path\nRun RepliBuild.build(\"$toml_path\") first!")
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
    clean(toml_path="replibuild.toml")

Remove build artifacts (build/, julia/, caches)

# Arguments
- `toml_path`: Path to replibuild.toml configuration file (default: "replibuild.toml")

# Examples
```julia
# Clean using replibuild.toml in current directory
RepliBuild.clean()

# Clean specific project
RepliBuild.clean("path/to/replibuild.toml")
```
"""
function clean(toml_path::String="replibuild.toml")
    # Resolve absolute path to TOML file
    toml_path = abspath(toml_path)

    if !isfile(toml_path)
        error("Configuration file not found: $toml_path")
    end

    project_dir = dirname(toml_path)
    clean_internal(project_dir)
end

# Internal clean function
function clean_internal(path::String)
    dirs_to_remove = ["build", "julia", ".replibuild_cache"]

    println("Cleaning build artifacts...")
    for dir in dirs_to_remove
        dir_path = joinpath(path, dir)
        if isdir(dir_path)
            rm(dir_path, recursive=true, force=true)
            println("   ✓ Removed $dir/")
        end
    end
end

"""
    info(toml_path="replibuild.toml")

Show project status (config, library, wrapper)

# Arguments
- `toml_path`: Path to replibuild.toml configuration file (default: "replibuild.toml")

# Examples
```julia
# Show info for current directory
RepliBuild.info()

# Show info for specific project
RepliBuild.info("path/to/replibuild.toml")
```
"""
function info(toml_path::String="replibuild.toml")
    println("═"^70)
    println(" RepliBuild - Project Info")
    println("═"^70)

    # Resolve absolute path to TOML file
    toml_path = abspath(toml_path)

    if !isfile(toml_path)
        println("❌ No replibuild.toml found at: $toml_path")
        println("   Create one with RepliBuild.Discovery.discover()")
        println("═"^70)
        return
    end

    project_dir = dirname(toml_path)

    data = TOML.parsefile(toml_path)
    project = get(data, "project", Dict())

    println("Config: $toml_path")
    println("Project: $(get(project, "name", "unnamed"))")
    println()

    # Check library
    julia_dir = joinpath(project_dir, "julia")
    if isdir(julia_dir)
        lib_files = filter(f -> endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll"),
                          readdir(julia_dir))
        if !isempty(lib_files)
            println("✓ Library: $(lib_files[1])")
        else
            println("❌ No library built yet - run RepliBuild.build(\"$toml_path\")")
        end

        # Check wrapper
        jl_files = filter(f -> endswith(f, ".jl"), readdir(julia_dir))
        if !isempty(jl_files)
            println("✓ Wrapper: $(jl_files[1])")
        else
            println("❌ No wrapper yet - run RepliBuild.wrap(\"$toml_path\")")
        end
    else
        println("❌ No build output - run RepliBuild.build(\"$toml_path\")")
    end

    println("═"^70)
end

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

end # module RepliBuild
