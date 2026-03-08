#!/usr/bin/env julia
# RepliBuild.jl - C++ → Julia Build Orchestration System
# Focus: Dependency-aware parallel compilation for single and multi-library C++ projects

module RepliBuild

using TOML
using JSON

# Version
const VERSION = v"2.4.0"

# ============================================================================
# LOAD CORE MODULES
# ============================================================================

# Internal utilities (not exported)
include("LLVMEnvironment.jl")
include("ConfigurationManager.jl")
include("BuildBridge.jl")
include("DependencyResolver.jl")

# Core build system modules
include("ASTWalker.jl")
include("Discovery.jl")
include("ClangJLBridge.jl")
include("Compiler.jl")
include("DWARFParser.jl")
include("MLIRNative.jl")
include("JLCSIRGenerator.jl")
include("JITManager.jl")
include("Wrapper.jl")
include("STLWrappers.jl")

# Introspection module
include("Introspect.jl")

# Polish modules
include("EnvironmentDoctor.jl")
include("PackageRegistry.jl")

# Import submodules for internal use
using .LLVMEnvironment
using .ConfigurationManager
using .BuildBridge
using .DependencyResolver
using .ASTWalker
using .Discovery
using .ClangJLBridge
using .Compiler
using .Wrapper
using .DWARFParser
using .JLCSIRGenerator
using .MLIRNative
using .JITManager
using .STLWrappers
using .Introspect
using .EnvironmentDoctor
using .PackageRegistry

# ============================================================================
# EXPORTS - Clean Build Orchestration API
# ============================================================================

# Core 3-function user API (THIS IS ALL YOU NEED)
export build, wrap, info

# Discovery function for setup
export discover

# Utility functions
export clean

# Environment diagnostics
export check_environment

# Package registry & scaffolding
export use, register, unregister, list_registry, scaffold_package

# Advanced modules (for power users who know what they're doing)
export Compiler, Wrapper, Discovery, ConfigurationManager, DWARFParser, JLCSIRGenerator, MLIRNative, STLWrappers

# Introspection API
export Introspect

"""
    check_environment(; verbose=true, throw_on_error=false) -> ToolchainStatus

Run environment diagnostics to verify LLVM 21+, MLIR, CMake, and other toolchain requirements.

Prints a colorful report showing which tools are found, their versions, and installation
instructions for anything missing. Use `throw_on_error=true` to abort on missing requirements.

# Example
```julia
status = RepliBuild.check_environment()
status.ready          # true if Tier 1 (ccall) builds will work
status.tier2_ready    # true if MLIR JIT tier is also available
```
"""
function check_environment(; verbose::Bool=true, throw_on_error::Bool=false)
    return EnvironmentDoctor.check_environment(verbose=verbose, throw_on_error=throw_on_error)
end

"""
    scaffold_package(name::String; path::String=".") -> String

Generate a standardized Julia package for distributing RepliBuild wrappers.

Creates a complete package with Project.toml, replibuild.toml, source stub,
deps/build.jl hook, and test skeleton. Edit the replibuild.toml to point at
your C/C++ source, then `Pkg.build()` compiles and wraps automatically.

# Example
```julia
RepliBuild.scaffold_package("MyEigenWrapper")
```
"""
function scaffold_package(name::String; path::String=".", from_registry::Bool=true)
    return PackageRegistry.scaffold_package(name; path=path, from_registry=from_registry)
end

"""
    use(name::String; force_rebuild=false, verbose=true) -> Module

Load a wrapper by registry name. Resolves dependencies, checks environment,
builds if needed, and returns the loaded Julia module.

# Example
```julia
Lua = RepliBuild.use("lua")
Lua.luaL_newstate()
```
"""
function use(name::String; force_rebuild::Bool=false, verbose::Bool=true)
    return PackageRegistry.use(name; force_rebuild=force_rebuild, verbose=verbose)
end

"""
    register(toml_path::String; name="", verified=false) -> RegistryEntry

Hash and store a replibuild.toml in the global registry (~/.replibuild/registry/).
Name is inferred from [project].name if not provided. Called automatically by `discover()`.
"""
function register(toml_path::String; name::String="", verified::Bool=false)
    return PackageRegistry.register(toml_path; name=name, verified=verified)
end

"""
    unregister(name::String)

Remove a package from the global registry.
"""
function unregister(name::String)
    PackageRegistry.unregister(name)
end

"""
    list_registry()

Print all registered packages in the global RepliBuild registry.
"""
function list_registry()
    PackageRegistry.list_registry()
end

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

function _build_aot_thunks(config, library_path)
    output_dir = ConfigurationManager.get_output_path(config)
    metadata_path = joinpath(output_dir, "compilation_metadata.json")
    
    if !isfile(metadata_path)
        @warn "Cannot AOT compile thunks: metadata not found."
        return
    end
    
    println("  aot: Generating MLIR thunks...")
    start_time = time()
    
    vtable_info = DWARFParser.parse_vtables(library_path)
    metadata = JSON.parsefile(metadata_path)
    ir_source = JLCSIRGenerator.generate_jlcs_ir(vtable_info, metadata)
    
    ctx = MLIRNative.create_context()
    try
        mod = MLIRNative.parse_module(ctx, ir_source)
        if mod == C_NULL
            error("Failed to parse generated MLIR for AOT.")
        end
        
        if !MLIRNative.lower_to_llvm(mod)
            error("Failed to lower MLIR to LLVM for AOT.")
        end
        
        thunks_obj = joinpath(output_dir, "thunks.o")
        if !MLIRNative.emit_object(mod, thunks_obj)
            error("Failed to emit object file for AOT thunks.")
        end
        
        # Link into a companion shared library
        lib_name = basename(library_path)
        thunks_name = replace(lib_name, ".so" => "_thunks.so", ".dylib" => "_thunks.dylib", ".dll" => "_thunks.dll")
        thunks_so = joinpath(output_dir, thunks_name)

        # Link thunks against the main library so C function symbols resolve
        lib_dir = dirname(abspath(library_path))
        linker = config.wrap.language == :c ? "clang" : "clang++"
        link_args = ["-shared", "-fPIC", "-o", thunks_so, thunks_obj,
                     "-L", lib_dir, "-l:$lib_name", "-Wl,-rpath,$lib_dir"]
        (output, exitcode) = BuildBridge.execute(linker, link_args)
        if exitcode != 0
            error("Failed to link thunks.o: $output")
        end
        
        # Emit LTO text IR for AOT thunks if LTO is enabled
        if config.link.enable_lto
            thunks_lto_name = replace(lib_name, ".so" => "_thunks_lto.ll", ".dylib" => "_thunks_lto.ll", ".dll" => "_thunks_lto.ll")
            thunks_lto_path = joinpath(output_dir, thunks_lto_name)
            if MLIRNative.emit_llvmir(mod, thunks_lto_path)
                lto_ir_text = read(thunks_lto_path, String)
                lto_ir_text = Compiler.sanitize_ir_for_julia(lto_ir_text)
                write(thunks_lto_path, lto_ir_text)

                # Assemble to bitcode via Julia's libLLVM for version-matched bc
                thunks_bc_path = replace(thunks_lto_path, ".ll" => ".bc")
                Compiler.assemble_bitcode(thunks_lto_path, thunks_bc_path)
            else
                @warn "Failed to emit LLVM IR for AOT thunks LTO."
            end
        end
        
        elapsed = round(time() - start_time, digits=2)
        size_kb = round(filesize(thunks_so) / 1024, digits=1)
        println("  aot: $thunks_name ($size_kb KB) in $(elapsed)s")
        
        # Cleanup
        rm(thunks_obj, force=true)
    catch e
        @warn "AOT MLIR compilation failed." exception=e
    finally
        MLIRNative.destroy_context(ctx)
    end
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

    # Validate environment before attempting build
    env_status = EnvironmentDoctor.check_environment(verbose=false)
    if !env_status.ready
        EnvironmentDoctor.check_environment(verbose=true, throw_on_error=true)
    end

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
        config = DependencyResolver.resolve_dependencies(config)

        # Compile the project (C++ → IR → library + metadata)
        library_path = Compiler.compile_project(config)

        # Build AOT thunks if enabled
        if config.compile.aot_thunks && config.binary.type != :executable
            _build_aot_thunks(config, library_path)
        end

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
        config = DependencyResolver.resolve_dependencies(config)

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


        # Generate wrapper
        wrapper_path = Wrapper.wrap_library(
            config,
            library_path,
            headers=headers,
            generate_tests=false,
            generate_docs=true
        )


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

    removed = String[]
    for dir in dirs_to_remove
        dir_path = joinpath(path, dir)
        if isdir(dir_path)
            rm(dir_path, recursive=true, force=true)
            push!(removed, dir)
        end
    end
    if !isempty(removed)
        println("  clean: $(join(removed, ", "))")
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
    toml_path = abspath(toml_path)

    if !isfile(toml_path)
        println("No replibuild.toml at: $toml_path")
        return
    end

    project_dir = dirname(toml_path)
    data = TOML.parsefile(toml_path)
    project = get(data, "project", Dict())

    println("RepliBuild | $(get(project, "name", "unnamed"))")

    julia_dir = joinpath(project_dir, "julia")
    if isdir(julia_dir)
        lib_files = filter(f -> endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll"),
                          readdir(julia_dir))
        if !isempty(lib_files)
            println("  library: $(lib_files[1])")
        else
            println("  library: not built")
        end

        jl_files = filter(f -> endswith(f, ".jl"), readdir(julia_dir))
        if !isempty(jl_files)
            println("  wrapper: $(jl_files[1])")
        else
            println("  wrapper: not generated")
        end

        lto_bc_files = filter(f -> endswith(f, "_lto.bc") && !contains(f, "thunks"), readdir(julia_dir))
        if !isempty(lto_bc_files)
            println("  lto_ir:  $(lto_bc_files[1])")
        end

        aot_bc_files = filter(f -> endswith(f, "_thunks_lto.bc"), readdir(julia_dir))
        if !isempty(aot_bc_files)
            println("  aot_ir:  $(aot_bc_files[1])")
        end

        aot_lib_files = filter(f -> contains(f, "_thunks") && (endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll")), readdir(julia_dir))
        if !isempty(aot_lib_files)
            println("  aot_lib: $(aot_lib_files[1])")
        end
    else
        println("  not built yet")
    end

end

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

end # module RepliBuild
