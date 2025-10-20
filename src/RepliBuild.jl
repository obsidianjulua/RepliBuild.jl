#!/usr/bin/env julia
# RepliBuild.jl - Main module for the RepliBuild build system
# A TOML-based build system leveraging LLVM/Clang for Julia bindings generation

module RepliBuild

# Version
const VERSION = v"0.1.1"

# Load all submodules in the correct order
include("UXHelpers.jl")  # Load UX helpers FIRST - needed by error handling
include("RepliBuildPaths.jl")  # Path management - user-local directories
include("LLVMEnvironment.jl")  # Load LLVM environment for toolchain isolation
include("ConfigurationManager.jl")  # Configuration management
include("Templates.jl")  # Project template initialization
include("ASTWalker.jl")  # AST dependency analysis
include("BuildHelpers.jl")  # Smart build utilities (config.h, pkg-config, etc.)
include("ModuleRegistry.jl")  # External library module system (NEW)
include("ModuleTemplateGenerator.jl")  # Module template generator (NEW)
include("BuildSystemDelegate.jl")  # Smart delegation to existing build systems (qmake, CMake, Meson)
include("Discovery.jl")  # Discovery pipeline
include("ProjectWizard.jl")  # Template-based project creation
include("ErrorLearning.jl")  # Error learning system (production-ready)
include("BuildBridge.jl")  # Command execution with error learning
include("CMakeParser.jl")  # CMake project import
include("LLVMake.jl")  # Core C++ → Julia compiler
include("JuliaWrapItUp.jl")  # Binary wrapper generation
include("ClangJLBridge.jl")  # Clang.jl binding integration
include("DaemonManager.jl")  # Daemon lifecycle management

# Re-export submodules
using .UXHelpers
using .RepliBuildPaths
using .LLVMEnvironment
using .ConfigurationManager
using .Templates
using .ASTWalker
using .BuildHelpers
using .ModuleRegistry
using .ModuleTemplateGenerator
using .BuildSystemDelegate
using .Discovery
using .ProjectWizard
using .ErrorLearning
using .BuildBridge
using .CMakeParser
using .LLVMake
using .JuliaWrapItUp
using .ClangJLBridge
using .DaemonManager

# Load Bridge_LLVM helper functions after modules are available
# (Bridge_LLVM uses the already-loaded modules above)
include("Bridge_LLVM.jl")

# ============================================================================
# IMPORT FUNCTIONS FOR RE-EXPORT
# ============================================================================

# Import path management functions
import .RepliBuildPaths: get_replibuild_dir, initialize_directories, ensure_initialized
import .RepliBuildPaths: get_module_search_paths, get_cache_dir, print_paths_info
import .RepliBuildPaths: get_config_value, set_config_value, migrate_old_structure

# Import module registry functions
import .ModuleRegistry: resolve_module, list_modules, register_module, get_module_info

# Import module template functions
import .ModuleTemplateGenerator: create_module_template, generate_from_pkg_config, generate_from_cmake

# Import build system delegate functions
import .BuildSystemDelegate: detect_build_system, delegate_build

# ============================================================================
# EXPORTS
# ============================================================================

# Export submodules themselves (production modules only)
export RepliBuildPaths, LLVMEnvironment, ConfigurationManager, Templates, ASTWalker, Discovery
export BuildHelpers, BuildBridge, CMakeParser, LLVMake, JuliaWrapItUp, ClangJLBridge, DaemonManager, ProjectWizard
export ModuleRegistry  # NEW: External library module system
export ModuleTemplateGenerator  # NEW: Module template generator
export BuildSystemDelegate  # NEW: Smart build system delegation
# ErrorLearning and UXHelpers are used internally, not exported to keep API surface small

# Export key types from LLVMake
export LLVMJuliaCompiler, CompilerConfig, TargetConfig

# Export key types from JuliaWrapItUp
export BinaryWrapper, WrapperConfig, BinaryInfo

# Export functions from submodules (already imported above)
export resolve_module, list_modules, register_module, get_module_info
export detect_build_system, delegate_build
export create_module_template, generate_from_pkg_config, generate_from_cmake
export get_replibuild_dir, initialize_directories, ensure_initialized
export get_module_search_paths, get_cache_dir, print_paths_info
export get_config_value, set_config_value, migrate_old_structure

# ============================================================================
# PRODUCTION API - Core user-facing functions for Julia/C++ workflows
# ============================================================================

# Project initialization
export init

# Template-based project creation (easier for beginners)
export create_project_interactive, available_templates, use_template

# Discovery & compilation pipeline (main workflows)
export discover, compile, compile_project, build

# CMake import (keep for convenience)
export import_cmake

# Binary wrapping (secondary workflow)
export wrap, wrap_binary, generate_wrappers, scan_binaries

# Binding generation (production)
export generate_bindings_clangjl, generate_from_config

# Info & help
export info, help, scan, analyze

# Daemon management (optional performance feature)
export start_daemons, stop_daemons, daemon_status, ensure_daemons

# LLVM toolchain management (advanced users)
export get_toolchain, verify_toolchain, print_toolchain_info, with_llvm_env

# Advanced API (for extension developers)
export discover_tools
# Internal: execute, capture, find_executable, command_exists
# Internal: parse_cmake_file, CMakeProject, CMakeTarget
# Internal: get_error_db, export_error_log, get_error_stats

"""
    init(project_dir::String="."; type::Symbol=:cpp)

Initialize a new RepliBuild project with the appropriate directory structure.

# Arguments
- `project_dir::String`: Directory to initialize (default: current directory)
- `type::Symbol`: Project type - `:cpp` for C++ source, `:binary` for binary wrapping

# Examples
```julia
RepliBuild.init("myproject")  # C++ project
RepliBuild.init("mybindings", type=:binary)  # Binary wrapping project
```
"""
function init(project_dir::String="."; type::Symbol=:cpp)
    mkpath(project_dir)

    if type == :cpp
        # C++ source project
        println("🚀 Initializing RepliBuild C++ project in: $project_dir")

        # Create directory structure
        for dir in ["src", "include", "julia", "build", "test"]
            mkpath(joinpath(project_dir, dir))
        end

        # Create replibuild.toml config
        config_file = joinpath(project_dir, "replibuild.toml")
        LLVMake.create_default_config(config_file)

        println("✅ C++ project initialized")
        println("📝 Edit $config_file to configure your project")
        println("📁 Put C++ sources in: $(joinpath(project_dir, "src"))")
        println("📁 Put headers in: $(joinpath(project_dir, "include"))")

    elseif type == :binary
        # Binary wrapping project
        println("🚀 Initializing RepliBuild binary wrapping project in: $project_dir")

        # Create directory structure
        for dir in ["lib", "bin", "julia_wrappers"]
            mkpath(joinpath(project_dir, dir))
        end

        # Create wrapper config
        config_file = joinpath(project_dir, "wrapper_config.toml")
        config = JuliaWrapItUp.create_default_wrapper_config()
        JuliaWrapItUp.save_wrapper_config(config, config_file)

        println("✅ Binary wrapping project initialized")
        println("📝 Edit $config_file to configure wrapper generation")
        println("📁 Put binary files in: $(joinpath(project_dir, "lib"))")

    else
        error("Unknown project type: $type. Use :cpp or :binary")
    end
end

"""
    compile(config_file::String="replibuild.toml")

Compile a C++ project to Julia bindings using the RepliBuild system.

# Arguments
- `config_file::String`: Path to replibuild.toml configuration file

# Examples
```julia
RepliBuild.compile()  # Use default replibuild.toml
RepliBuild.compile("custom_config.toml")
```
"""
function compile(config_file::String="replibuild.toml")
    println("🚀 RepliBuild - Compiling project")
    config = BridgeCompilerConfig(config_file)
    compile_project(config)
end

"""
    build(project_dir::String="."; config_file::String="replibuild.toml")

Universal build function that intelligently delegates to the appropriate build system.

This function:
1. Reads `replibuild.toml` to determine the build system (qmake, cmake, meson, etc.)
2. Uses Julia artifacts (JLL packages) when in Julia environment for reproducibility
3. Falls back to system tools when running standalone
4. Returns build artifacts (libraries, executables)

# Arguments
- `project_dir::String`: Project directory to build (default: current directory)
- `config_file::String`: Configuration file name (default: "replibuild.toml")

# TOML Configuration
Add a `[build]` section to your replibuild.toml:
```toml
[build]
system = "qmake"  # or "cmake", "meson", "autotools", "make"
qt_version = "Qt5"  # for Qt/qmake projects
build_dir = "build"  # optional, default: "build"
```

# Examples
```julia
# Build current project (auto-detect or use TOML)
RepliBuild.build()

# Build specific project
RepliBuild.build("/path/to/qt/project")

# Specify custom config
RepliBuild.build(".", config_file="custom.toml")
```

# Returns
Dict with keys:
- `:libraries` - Array of built library paths
- `:executables` - Array of built executable paths
- `:build_dir` - Build directory path
"""
function build(project_dir::String="."; config_file::String="replibuild.toml")
    println("🔨 RepliBuild - Universal Build System")
    println("   Project: $project_dir")

    # Delegate to BuildSystemDelegate
    return BuildSystemDelegate.delegate_build(project_dir, toml_path=config_file)
end

"""
    wrap(config_file::String="wrapper_config.toml")

Generate Julia wrappers for existing binary files.

# Arguments
- `config_file::String`: Path to wrapper configuration file

# Examples
```julia
RepliBuild.wrap()  # Use default wrapper_config.toml
RepliBuild.wrap("custom_wrapper.toml")
```
"""
function wrap(config_file::String="wrapper_config.toml")
    println("🚀 RepliBuild - Generating binary wrappers")
    wrapper = JuliaWrapItUp.BinaryWrapper(config_file)
    JuliaWrapItUp.generate_wrappers(wrapper)
end

"""
    wrap_binary(binary_path::String; config_file::String="wrapper_config.toml")

Wrap a specific binary file to Julia bindings.

# Arguments
- `binary_path::String`: Path to the binary file (.so, .dll, .dylib, etc.)
- `config_file::String`: Path to wrapper configuration file

# Examples
```julia
RepliBuild.wrap_binary("/usr/lib/libmath.so")
RepliBuild.wrap_binary("./build/libmylib.so")
```
"""
function wrap_binary(binary_path::String; config_file::String="wrapper_config.toml")
    println("🚀 RepliBuild - Wrapping binary: $binary_path")
    wrapper = JuliaWrapItUp.BinaryWrapper(config_file)
    JuliaWrapItUp.generate_wrappers(wrapper, specific_binary=binary_path)
end

"""
    discover_tools(config_file::String="replibuild.toml")

Discover LLVM/Clang tools available on the system using BuildBridge.

# Arguments
- `config_file::String`: Path to replibuild.toml configuration file

# Examples
```julia
RepliBuild.discover_tools()
```
"""
function discover_tools(config_file::String="replibuild.toml")
    println("🔍 RepliBuild - Discovering LLVM tools")
    config = BridgeCompilerConfig(config_file)
    discover_tools!(config)
end

"""
    import_cmake(cmake_file::String="CMakeLists.txt"; target::String="", output::String="replibuild.toml")

Import a CMake project and generate replibuild.toml configuration.

# Arguments
- `cmake_file::String`: Path to CMakeLists.txt file
- `target::String`: Specific target to import (empty = first target)
- `output::String`: Output path for replibuild.toml

# Examples
```julia
# Import first target from CMake project
RepliBuild.import_cmake("path/to/CMakeLists.txt")

# Import specific target
RepliBuild.import_cmake("opencv/CMakeLists.txt", target="opencv_core")
```
"""
function import_cmake(cmake_file::String="CMakeLists.txt"; target::String="", output::String="replibuild.toml")
    println("📦 Importing CMake project: $cmake_file")

    # Parse CMakeLists.txt
    cmake_project = CMakeParser.parse_cmake_file(cmake_file)

    println("✅ Found CMake project: $(cmake_project.project_name)")
    println("   Targets: $(join(keys(cmake_project.targets), ", "))")

    # Determine target
    target_name = target
    if isempty(target_name)
        if isempty(cmake_project.targets)
            error("No targets found in CMake project")
        end
        target_name = first(keys(cmake_project.targets))
        println("   Using target: $target_name")
    end

    # Generate replibuild.toml
    CMakeParser.write_replibuild_config(cmake_project, target_name, output)

    println("🎉 CMake import complete!")
    println("   Generated: $output")
    println("   Run: RepliBuild.compile(\"$output\")")

    return cmake_project
end

"""
    export_errors(output_path::String="error_log.md")

Export error learning database to Obsidian-friendly markdown.

# Examples
```julia
RepliBuild.export_errors("docs/errors.md")
```
"""
function export_errors(output_path::String="error_log.md")
    BuildBridge.export_error_log("replibuild_errors.db", output_path)
end

"""
    scan(path="."; generate_config=true, output="replibuild.toml")

Scan a directory and analyze its structure for RepliBuild compilation.
Auto-generates replibuild.toml if generate_config=true.

# Examples
```julia
RepliBuild.scan()  # Scan current directory
RepliBuild.scan("path/to/project")  # Scan specific directory
RepliBuild.scan(".", generate_config=false)  # Just analyze, don't generate config
RepliBuild.scan(".", output="my_config.toml")  # Custom output name
```
"""
function scan(path="."; generate_config=true, output="replibuild.toml")
    println("🔍 Scanning project: $path")

    # Use Discovery module to scan project
    result = Discovery.discover(path, force=true)

    if generate_config && haskey(result, :scan_results)
        println("📝 Generating configuration: $output")
        # The discover function already generates config, just make sure it exists
        config_path = joinpath(path, "replibuild.toml")
        if isfile(config_path) && output != "replibuild.toml"
            # Copy to requested output name
            cp(config_path, joinpath(path, output), force=true)
        end
    end

    return result
end

"""
    analyze(path=".")

Analyze project structure and return detailed analysis.

# Examples
```julia
result = RepliBuild.analyze("path/to/project")
println("Found \$(length(result[:scan_results].cpp_sources)) C++ files")
```
"""
function analyze(path=".")
    # Return discovery results for analysis (always force scan for analysis)
    return Discovery.discover(path, force=true)
end

"""
    info()

Display information about the RepliBuild build system.
"""
function info()
    println("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  RepliBuild Build System v$VERSION                 ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  A TOML-based build system leveraging LLVM/Clang           ║
    ║  for automatic Julia bindings generation                    ║
    ╚══════════════════════════════════════════════════════════════╝

    Components:
    • BuildBridge     - Command execution with error learning
    • CMakeParser     - Import CMake projects without running CMake
    • LLVMake         - C++ source → Julia compiler
    • JuliaWrapItUp   - Binary → Julia wrapper generator
    • Bridge_LLVM     - Orchestrator integrating all components

    Quick Start:
    1. Initialize project:   RepliBuild.init("myproject")
    2. Add C++ sources:      Put files in src/
    3. Configure:            Edit replibuild.toml
    4. Compile:              RepliBuild.compile()

    For binary wrapping:
    1. Initialize:           RepliBuild.init("mybindings", type=:binary)
    2. Add binaries:         Put .so files in lib/
    3. Configure:            Edit wrapper_config.toml
    4. Wrap:                 RepliBuild.wrap()

    Documentation: See README.md
    """)
end

"""
    help()

Display help information about RepliBuild commands.
"""
function help()
    println("""
    RepliBuild Build System - Command Reference

    Initialization:
      RepliBuild.init([dir])              Initialize C++ project
      RepliBuild.init(dir, type=:binary)  Initialize binary wrapping project

    Compilation:
      RepliBuild.compile([config])        Compile C++ → Julia
      RepliBuild.discover_tools([config]) Discover LLVM tools

    Binary Wrapping:
      RepliBuild.wrap([config])           Wrap all binaries
      RepliBuild.wrap_binary(path)        Wrap specific binary

    Information:
      RepliBuild.info()                   Show RepliBuild information
      RepliBuild.help()                   Show this help

    Configuration Files:
      replibuild.toml                     Main project configuration
      wrapper_config.toml            Binary wrapping configuration

    Examples:
      # Create and build C++ project
      RepliBuild.init("mymath")
      cd("mymath")
      # ... add C++ files to src/ ...
      RepliBuild.compile()

      # Wrap existing library
      RepliBuild.init("wrappers", type=:binary)
      cd("wrappers")
      RepliBuild.wrap_binary("/usr/lib/libcrypto.so")

    For detailed documentation, see the README.md file.
    """)
end

# ============================================================================
# DAEMON MANAGEMENT FUNCTIONS
# ============================================================================

# Global daemon system instance
const DAEMON_SYSTEM = Ref{Union{DaemonManager.DaemonSystem, Nothing}}(nothing)

"""
    start_daemons(;project_root=pwd())

Start all RepliBuild daemon servers (discovery, setup, compilation, orchestrator).
Replaces manual shell script execution.

# Examples
```julia
RepliBuild.start_daemons()  # Start in current directory
RepliBuild.start_daemons(project_root="/path/to/project")
```
"""
function start_daemons(;project_root=pwd())
    if !isnothing(DAEMON_SYSTEM[])
        println("⚠ Daemons already running. Use stop_daemons() first to restart.")
        return DAEMON_SYSTEM[]
    end

    # Clean up stale PID files
    DaemonManager.cleanup_stale_pids(project_root)

    # Start daemon system
    DAEMON_SYSTEM[] = DaemonManager.start_all(project_root=project_root)

    return DAEMON_SYSTEM[]
end

"""
    stop_daemons()

Stop all running RepliBuild daemons gracefully.

# Examples
```julia
RepliBuild.stop_daemons()
```
"""
function stop_daemons()
    if isnothing(DAEMON_SYSTEM[])
        println("No daemons are running")
        return
    end

    DaemonManager.stop_all(DAEMON_SYSTEM[])
    DAEMON_SYSTEM[] = nothing
end

"""
    daemon_status()

Display status of all RepliBuild daemons.

# Examples
```julia
RepliBuild.daemon_status()
```
"""
function daemon_status()
    if isnothing(DAEMON_SYSTEM[])
        println("No daemons are running")
        println("\nStart daemons with: RepliBuild.start_daemons()")
        return
    end

    DaemonManager.status(DAEMON_SYSTEM[])
end

"""
    ensure_daemons()

Check if all daemons are running and restart any that have crashed.
Returns true if all daemons are healthy.

# Examples
```julia
if !RepliBuild.ensure_daemons()
    println("Some daemons failed to restart")
end
```
"""
function ensure_daemons()
    if isnothing(DAEMON_SYSTEM[])
        println("Starting daemons...")
        start_daemons()
        return true
    end

    return DaemonManager.ensure_running(DAEMON_SYSTEM[])
end

# Show info on module load
function __init__()
    # Optional: Show a brief message when loaded
    # println("RepliBuild v$VERSION loaded. Type RepliBuild.help() for usage information.")
end

end # module RepliBuild
