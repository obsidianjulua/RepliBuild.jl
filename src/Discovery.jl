#!/usr/bin/env julia
# Discovery.jl - RepliBuild Discovery Pipeline

module Discovery

using Dates
using ProgressMeter
using UUIDs

# These modules are already loaded by RepliBuild.jl before Discovery.jl is included
# We just need to import them from the parent module scope
import ..LLVMEnvironment
import ..ConfigurationManager
import ..ASTWalker

# Use LLVMEnvironment to get correct LLVM path
function get_replibuild_llvm_root()
    replibuild_dir = dirname(@__DIR__)  # Go from src/ to RepliBuild/
    return joinpath(replibuild_dir, "LLVM")
end

# Define constant for use in configuration generation
const REPLIBUILD_LLVM_ROOT = get_replibuild_llvm_root()

"""
File scan results
"""
struct ScanResults
    cpp_sources::Vector{String}
    cpp_headers::Vector{String}
    c_sources::Vector{String}
    c_headers::Vector{String}
    binaries::Vector{String}
    static_libs::Vector{String}
    shared_libs::Vector{String}
    julia_files::Vector{String}
    config_files::Vector{String}
    docs::Vector{String}
    other::Vector{String}
    total_files::Int
end

"""
Binary detection results
"""
struct BinaryInfo
    path::String
    name::String
    type::Symbol  # :executable, :shared_lib, :static_lib
    size::Int
end

"""
    discover(target_dir::String=pwd(); force::Bool=false, unsafe::Bool=false, build::Bool=false, wrap::Bool=false) -> String

Main discovery pipeline - scans project and generates configuration.

# Process:
1. Check for existing replibuild.toml (project identified by presence of replibuild.toml)
2. Scan all files and categorize
3. Detect and analyze binaries
4. Walk AST dependencies using clang
5. Generate or update replibuild.toml with discovered data
6. Optionally run build and wrap pipeline

# Arguments
- `target_dir`: Project directory (default: current directory)
- `force`: Force rediscovery even if replibuild.toml exists
- `unsafe`: Bypass safety checks (use with extreme caution)
- `build`: Automatically run build() after discovery (default: false)
- `wrap`: Automatically run wrap() after build (requires build=true, default: false)

# Safety Features
- Discovery is scoped ONLY to target_dir and subdirectories
- Will not scan outside the project root
- Skips .git, build, node_modules, .cache directories

# Returns
- Path to generated `replibuild.toml` file

# Examples
```julia
# Discover only
toml_path = RepliBuild.Discovery.discover()

# Discover and build
toml_path = RepliBuild.Discovery.discover(build=true)

# Full pipeline: discover → build → wrap
toml_path = RepliBuild.Discovery.discover(build=true, wrap=true)

# Then use the TOML path:
RepliBuild.build(toml_path)
RepliBuild.wrap(toml_path)
```
"""
function discover(target_dir::String=pwd(); force::Bool=false, unsafe::Bool=false, build::Bool=false, wrap::Bool=false)
    println(" RepliBuild Discovery Pipeline")
    println("="^70)
    println(" Target: $target_dir")

    # SAFETY CHECK
    if !unsafe
        abs_target = abspath(target_dir)
        println("🔒 Safety: Scoped to $(abs_target) and subdirectories only")
    else
        @warn "  UNSAFE MODE: Discovery safety checks bypassed!"
    end
    println()

    # Check if already configured
    config_path = joinpath(target_dir, "replibuild.toml")
    if isfile(config_path) && !force
        println("  replibuild.toml already exists!")
        println("   Use discover(force=true) to regenerate")
        println("="^70)
        return config_path
    end

    # Stage 1: Scan files
    println(" Stage 1: Scanning files...")
    scan_results = scan_all_files(target_dir)
    print_scan_summary(scan_results)

    # Stage 2: Detect binaries
    println("\n Stage 2: Detecting binaries...")
    binaries = detect_all_binaries(target_dir, scan_results)
    print_binary_summary(binaries)

    # Stage 3: Build include directories
    println("\n Stage 3: Building include paths...")
    include_dirs = build_include_dirs(target_dir, scan_results)
    println("   Found $(length(include_dirs)) include directories")

    # Stage 4: Walk AST dependencies
    println("\n Stage 4: Walking AST dependencies...")
    dep_graph = walk_dependencies(target_dir, scan_results, include_dirs)

    # Stage 5: Generate configuration
    println("\n Stage 5: Generating replibuild.toml...")
    config = generate_config(target_dir, scan_results, binaries, include_dirs, dep_graph)

    # Stage 6: Initialize toolchain
    println("\n Stage 6: Initializing LLVM toolchain...")
    LLVMEnvironment.init_toolchain(config=config)

    # Diagnostic output (config is immutable, values already set during generation)
    include_dirs = ConfigurationManager.get_include_dirs(config)
    source_files = ConfigurationManager.get_source_files(config)

    if !isempty(include_dirs)
        println("   ✓ Configured $(length(include_dirs)) include directories")
    end

    if !isempty(source_files)
        println("   ✓ Configured $(length(source_files)) source files")
    end

    # Save config (writes TOML file)
    ConfigurationManager.save_config(config)

    println("\n Discovery complete!")
    println(" Configuration: $config_path")

    # Chain build and wrap if requested
    if build
        println()
        println(" Running build pipeline...")
        println("="^70)

        # Need to access parent module's build function
        # Import from parent RepliBuild module
        build_func = getfield(parentmodule(@__MODULE__), :build)
        library_path = build_func(config_path)

        if wrap
            println()
            println(" Running wrap pipeline...")
            println("="^70)

            wrap_func = getfield(parentmodule(@__MODULE__), :wrap)
            wrapper_path = wrap_func(config_path)

            println()
            println("="^70)
            println(" Full pipeline complete!")
            println(" - Config:  $config_path")
            println(" - Library: $library_path")
            println(" - Wrapper: $wrapper_path")
            println("="^70)
        else
            println()
            println("="^70)
            println(" Discovery + Build complete!")
            println(" - Config:  $config_path")
            println(" - Library: $library_path")
            println()
            println(" Next: RepliBuild.wrap(\"$config_path\") to generate Julia bindings")
            println("="^70)
        end
    else
        println()
        println(" Next: RepliBuild.build(\"$config_path\") to compile C++ library")
        println("="^70)
    end

    return config_path
end

"""
    scan_all_files(root_dir::String) -> ScanResults

Scan directory and categorize all files by type.

# Safety
- ONLY scans within root_dir and subdirectories
- Skips dangerous/irrelevant directories (.git, build, system dirs)
- Never follows symlinks outside project root
"""
function scan_all_files(root_dir::String)
    cpp_sources = String[]
    cpp_headers = String[]
    c_sources = String[]
    c_headers = String[]
    binaries = String[]
    static_libs = String[]
    shared_libs = String[]
    julia_files = String[]
    config_files = String[]
    docs = String[]
    other = String[]

    total = 0

    abs_root = abspath(root_dir)

    println("   Scanning scope: $abs_root")

    for (root, dirs, files_list) in walkdir(root_dir, follow_symlinks=false)
        # SAFETY: Ensure we're still within project root
        if !startswith(abspath(root), abs_root)
            @warn "Skipping directory outside project root: $root"
            empty!(dirs)  # Don't recurse
            continue
        end

        # Skip build/cache/system directories
        filter!(d -> !in(d, ["build", ".git", ".cache", "node_modules", ".replibuild_cache",
                            ".svn", ".hg", "__pycache__", "venv", ".venv"]), dirs)

        for file in files_list
            total += 1
            filepath = joinpath(root, file)
            relpath_file = relpath(filepath, root_dir)
            ext = lowercase(splitext(file)[2])

            # Categorize by extension and type
            if ext in [".cpp", ".cc", ".cxx", ".c++"]
                push!(cpp_sources, relpath_file)
            elseif ext in [".hpp", ".hxx", ".h++", ".hh"]
                push!(cpp_headers, relpath_file)
            elseif ext == ".h"
                # Detect C vs C++ header by content
                if is_cpp_header(filepath)
                    push!(cpp_headers, relpath_file)
                else
                    push!(c_headers, relpath_file)
                end
            elseif ext == ".c"
                push!(c_sources, relpath_file)
            elseif ext == ".so" || contains(file, ".so.")
                push!(shared_libs, relpath_file)
            elseif ext == ".a"
                push!(static_libs, relpath_file)
            elseif ext == ".jl"
                push!(julia_files, relpath_file)
            elseif ext in [".toml", ".json", ".yaml", ".yml", ".xml"]
                push!(config_files, relpath_file)
            elseif ext in [".md", ".txt", ".rst", ".org", ".pdf"]
                push!(docs, relpath_file)
            elseif ext == "" && is_binary(filepath)
                push!(binaries, relpath_file)
            else
                push!(other, relpath_file)
            end
        end
    end

    return ScanResults(
        cpp_sources, cpp_headers, c_sources, c_headers,
        binaries, static_libs, shared_libs,
        julia_files, config_files, docs, other,
        total
    )
end

"""
    is_cpp_header(filepath::String) -> Bool

Detect if .h file is C++ by scanning content.
"""
function is_cpp_header(filepath::String)
    if !isfile(filepath)
        return false
    end

    try
        content = read(filepath, String)
        # C++ indicators
        cpp_indicators = [
            r"\bclass\s+\w+",
            r"\bnamespace\s+\w+",
            r"\btemplate\s*<",
            r"::\w+",
            r"\bstd::",
            r"\bvirtual\s+",
            r"\boverride\b",
            r"\bconstexpr\b"
        ]

        return any(pattern -> occursin(pattern, content), cpp_indicators)
    catch
        return false
    end
end

"""
    is_binary(filepath::String) -> Bool

Check if file is binary executable (ELF magic).
"""
function is_binary(filepath::String)
    if !isfile(filepath)
        return false
    end

    try
        # Check executable bit
        if Sys.isunix()
            try
                run(`test -x $filepath`)
                return true
            catch
            end
        end

        # Check ELF magic bytes
        magic = read(filepath, 4)
        if length(magic) >= 4 && magic[1:4] == UInt8[0x7f, 0x45, 0x4c, 0x46]
            return true
        end
    catch
        return false
    end

    return false
end

"""
    detect_all_binaries(root_dir::String, scan::ScanResults) -> Vector{BinaryInfo}

Detect and analyze all binaries.
"""
function detect_all_binaries(root_dir::String, scan::ScanResults)
    binaries = BinaryInfo[]

    # Executables
    for file in scan.binaries
        filepath = joinpath(root_dir, file)
        if isfile(filepath)
            push!(binaries, BinaryInfo(
                file,
                basename(file),
                :executable,
                filesize(filepath)
            ))
        end
    end

    # Static libraries
    for file in scan.static_libs
        filepath = joinpath(root_dir, file)
        if isfile(filepath)
            push!(binaries, BinaryInfo(
                file,
                basename(file),
                :static_lib,
                filesize(filepath)
            ))
        end
    end

    # Shared libraries
    for file in scan.shared_libs
        filepath = joinpath(root_dir, file)
        if isfile(filepath)
            push!(binaries, BinaryInfo(
                file,
                basename(file),
                :shared_lib,
                filesize(filepath)
            ))
        end
    end

    return binaries
end

"""
    build_include_dirs(root_dir::String, scan::ScanResults) -> Vector{String}

Build list of include directories from discovered headers.
"""
function build_include_dirs(root_dir::String, scan::ScanResults)
    include_dirs = Set{String}()

    # Add directories containing headers
    for header in vcat(scan.cpp_headers, scan.c_headers)
        header_dir = dirname(header)
        if !isempty(header_dir) && header_dir != "."
            push!(include_dirs, abspath(joinpath(root_dir, header_dir)))
        end
    end

    # Add common standard locations
    push!(include_dirs, abspath(root_dir))

    include_path = joinpath(root_dir, "include")
    if isdir(include_path)
        push!(include_dirs, abspath(include_path))
    end

    src_path = joinpath(root_dir, "src")
    if isdir(src_path)
        push!(include_dirs, abspath(src_path))
    end

    return sort(collect(include_dirs))
end

"""
    walk_dependencies(root_dir::String, scan::ScanResults, include_dirs::Vector{String}) -> Union{DependencyGraph,Nothing}

Walk AST dependencies using clang from LLVMEnvironment.
"""
function walk_dependencies(root_dir::String, scan::ScanResults, include_dirs::Vector{String})
    # Get all source files
    all_sources = vcat(
        [joinpath(root_dir, f) for f in scan.cpp_sources],
        [joinpath(root_dir, f) for f in scan.c_sources],
        [joinpath(root_dir, f) for f in scan.cpp_headers],
        [joinpath(root_dir, f) for f in scan.c_headers]
    )

    if isempty(all_sources)
        println("     No source files found")
        return nothing
    end

    # Get clang from LLVM environment
    clang_path = LLVMEnvironment.get_tool("clang++")

    if isempty(clang_path)
        @warn "Clang not found in LLVM environment, skipping AST walk"
        return nothing
    end

    # Build dependency graph
    dep_graph = LLVMEnvironment.with_llvm_env() do
        ASTWalker.build_dependency_graph(
            all_sources,
            include_dirs,
            use_clang=true,
            clang_path=clang_path
        )
    end

    # Print summary
    ASTWalker.print_dependency_summary(dep_graph)

    # Export to JSON for inspection
    json_path = joinpath(root_dir, ".replibuild_cache", "dependency_graph.json")
    mkpath(dirname(json_path))
    ASTWalker.export_dependency_graph_json(dep_graph, json_path)

    return dep_graph
end

"""
    generate_config(root_dir, scan, binaries, include_dirs, dep_graph) -> RepliBuildConfig

Generate RepliBuild configuration from discovery results using new nested struct approach.
"""
function generate_config(root_dir::String, scan::ScanResults, binaries::Vector{BinaryInfo},
                        include_dirs::Vector{String}, dep_graph)
    # Get project name from absolute path (handles "." correctly)
    abs_root = abspath(root_dir)
    project_name = basename(abs_root)
    # Fallback if basename is empty
    if isempty(project_name)
        project_name = "project"
    end
    project_uuid = uuid4()

    # Get source files from dependency graph if available
    source_files = if !isnothing(dep_graph)
        # Filter to .cpp files in compilation order
        filter(f -> endswith(f, ".cpp") || endswith(f, ".cc") || endswith(f, ".cxx"),
               dep_graph.compilation_order)
    else
        scan.cpp_sources
    end

    # Create nested config structs following ConfigurationManager structure
    project_config = ConfigurationManager.ProjectConfig(
        project_name,
        abspath(root_dir),
        project_uuid
    )

    paths_config = ConfigurationManager.PathsConfig(
        "src",              # source
        "include",          # include
        "julia",            # output
        "build",            # build
        ".replibuild_cache" # cache
    )

    discovery_config = ConfigurationManager.DiscoveryConfig(
        true,                                    # enabled
        true,                                    # walk_dependencies
        10,                                      # max_depth
        ["build", ".git", ".cache"],            # ignore_patterns
        true                                     # parse_ast
    )

    compile_config = ConfigurationManager.CompileConfig(
        source_files,                            # source_files
        include_dirs,                            # include_dirs
        ["-std=c++17", "-fPIC"],         # flags
        Dict{String,String}(),                   # defines
        true                                     # parallel
    )

    link_config = ConfigurationManager.LinkConfig(
        "0",                                     # optimization_level (O0 for safety and DWARF accuracy)
        false,                                   # enable_lto
        String[]                                 # link_libraries
    )

    binary_config = ConfigurationManager.BinaryConfig(
        :shared,                                 # type
        "",                                      # output_name (auto-generate)
        false                                    # strip_symbols
    )

    wrap_config = ConfigurationManager.WrapConfig(
        true,                                    # enabled
        :clang,                                  # style
        "",                                      # module_name (auto-generate)
        true                                     # use_clang_jl
    )

    llvm_config = ConfigurationManager.LLVMConfig(
        :auto,                                   # toolchain
        ""                                       # version (auto-detect)
    )

    workflow_config = ConfigurationManager.WorkflowConfig(
        [:discover, :compile, :link, :binary, :wrap]  # stages
    )

    cache_config = ConfigurationManager.CacheConfig(
        true,                                    # enabled
        ".replibuild_cache"                      # directory
    )

    types_config = ConfigurationManager.TypesConfig(
        :warn,                                   # strictness
        true,                                    # allow_unknown_structs
        false,                                   # allow_unknown_enums
        true,                                    # allow_function_pointers
        Dict{String,String}()                    # custom_mappings
    )

    # Construct RepliBuildConfig from nested structs
    config = ConfigurationManager.RepliBuildConfig(
        project_config,
        paths_config,
        discovery_config,
        compile_config,
        link_config,
        binary_config,
        wrap_config,
        llvm_config,
        workflow_config,
        cache_config,
        types_config,
        joinpath(root_dir, "replibuild.toml"),  # config_file
        now()                                    # loaded_at
    )

    return config
end

"""
    print_scan_summary(scan::ScanResults)

Print file scan summary.
"""
function print_scan_summary(scan::ScanResults)
    println("    Scan Results:")
    println("      C++ Sources:    $(length(scan.cpp_sources))")
    println("      C++ Headers:    $(length(scan.cpp_headers))")
    println("      C Sources:      $(length(scan.c_sources))")
    println("      C Headers:      $(length(scan.c_headers))")
    println("      Binaries:       $(length(scan.binaries))")
    println("      Static Libs:    $(length(scan.static_libs))")
    println("      Shared Libs:    $(length(scan.shared_libs))")
    println("      Julia Files:    $(length(scan.julia_files))")
    println("      Total Files:    $(scan.total_files)")
end

"""
    print_binary_summary(binaries::Vector{BinaryInfo})

Print binary detection summary.
"""
function print_binary_summary(binaries::Vector{BinaryInfo})
    if isempty(binaries)
        println("    No binaries detected")
        return
    end

    println("    Detected $(length(binaries)) binaries:")

    for binary in binaries
        type_icon = binary.type == :executable ? "⚙️" :
                   binary.type == :shared_lib ? "" : "📕"
        size_kb = round(binary.size / 1024, digits=1)
        println("      $type_icon $(binary.name) ($(binary.type), $(size_kb) KB)")
    end
end

# Exports
export discover

end # module Discovery
