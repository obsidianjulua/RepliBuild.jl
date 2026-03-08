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
    if !unsafe
        abs_target = abspath(target_dir)
    else
        @warn "UNSAFE MODE: Discovery safety checks bypassed"
    end

    config_path = joinpath(target_dir, "replibuild.toml")
    if isfile(config_path) && !force
        return config_path
    end

    println("RepliBuild | discover $(basename(abspath(target_dir)))")

    scan_results = scan_all_files(target_dir)
    binaries = detect_all_binaries(target_dir, scan_results)
    include_dirs = build_include_dirs(target_dir, scan_results)
    dep_graph = walk_dependencies(target_dir, scan_results, include_dirs)
    config = generate_config(target_dir, scan_results, binaries, include_dirs, dep_graph)
    LLVMEnvironment.init_toolchain(config=config)

    source_files = ConfigurationManager.get_source_files(config)
    include_dirs = ConfigurationManager.get_include_dirs(config)
    println("  found: $(length(source_files)) sources, $(length(include_dirs)) include dirs")

    ConfigurationManager.save_config(config)

    # Auto-register in global package registry
    try
        register_func = getfield(parentmodule(@__MODULE__), :register)
        register_func(config_path)
    catch e
        @debug "Auto-registration skipped" exception=e
    end

    if build
        build_func = getfield(parentmodule(@__MODULE__), :build)
        library_path = build_func(config_path)

        if wrap
            wrap_func = getfield(parentmodule(@__MODULE__), :wrap)
            wrapper_path = wrap_func(config_path)
        end
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
        filter(f -> endswith(f, ".cpp") || endswith(f, ".cc") || endswith(f, ".cxx") || endswith(f, ".c"),
               dep_graph.compilation_order)
    else
        vcat(scan.cpp_sources, scan.c_sources)
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

    has_cpp = !isempty(scan.cpp_sources)
    default_flags = has_cpp ? ["-std=c++17", "-fPIC"] : ["-fPIC"]

    compile_config = ConfigurationManager.CompileConfig(
        source_files,                            # source_files
        include_dirs,                            # include_dirs
        default_flags,                           # flags
        Dict{String,String}(),                   # defines
        true,                                    # parallel
        false                                    # aot_thunks
    )

    link_config = ConfigurationManager.LinkConfig(
        "0",                                     # optimization_level (O0 for safety and DWARF accuracy)
        false,                                   # enable_lto
        String[],                                # link_libraries
        String[]                                 # link_dirs
    )

    binary_config = ConfigurationManager.BinaryConfig(
        :shared,                                 # type
        "",                                      # output_name (auto-generate)
        false                                    # strip_symbols
    )

    wrap_config = ConfigurationManager.WrapConfig(
        true,                                    # enabled
        :clang,                                  # style
        has_cpp ? :cpp : :c,                     # language
        "",                                      # module_name (auto-generate)
        true,                                    # use_clang_jl
        Dict{String,Vector{Vector{String}}}()    # varargs_overloads
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
        Dict{String,String}(),                   # custom_mappings
        String[],                                # templates
        String[]                                 # template_headers
    )

    dependencies_config = ConfigurationManager.DependenciesConfig(
        Dict{String, ConfigurationManager.DependencyItem}()
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
        dependencies_config,
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
end

"""
    print_binary_summary(binaries::Vector{BinaryInfo})

Print binary detection summary.
"""
function print_binary_summary(binaries::Vector{BinaryInfo})
end

# Exports
export discover

end # module Discovery
