#!/usr/bin/env julia
# Discovery.jl - RepliBuild Discovery Pipeline
# Orchestrates: file scanning, binary detection, AST dependency walking, config generation
# Entry point: RepliBuild.discover() - called after Templates.jl plants the structure

module Discovery

using Dates
using ProgressMeter
using UUIDs  # Add UUID import at module level

# Import sibling modules
include("LLVMEnvironment.jl")
include("ConfigurationManager.jl")
include("ASTWalker.jl")
include("UXHelpers.jl")

using .LLVMEnvironment
using .ConfigurationManager
using .ASTWalker
using .UXHelpers

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
    discover(target_dir::String=pwd(); force::Bool=false, unsafe::Bool=false) -> RepliBuildConfig

Main discovery pipeline - scans project and generates configuration.

# Process:
1. Check for existing replibuild.toml (project identified by presence of replibuild.toml)
2. Scan all files and categorize
3. Detect and analyze binaries
4. Walk AST dependencies using clang
5. Generate or update replibuild.toml with discovered data
6. Return configuration

# Arguments
- `target_dir`: Project directory (default: current directory)
- `force`: Force rediscovery even if replibuild.toml exists
- `unsafe`: Bypass safety checks (use with extreme caution)

# Safety Features
- Discovery is scoped ONLY to target_dir and subdirectories
- Will not scan outside the project root
- Skips .git, build, node_modules, .cache directories

# Returns
- `RepliBuildConfig`: Complete project configuration
"""
function discover(target_dir::String=pwd(); force::Bool=false, unsafe::Bool=false)
    println("🔍 RepliBuild Discovery Pipeline")
    println("="^70)
    println("📁 Target: $target_dir")

    # SAFETY CHECK
    if !unsafe
        abs_target = abspath(target_dir)
        println("🔒 Safety: Scoped to $(abs_target) and subdirectories only")
    else
        @warn "⚠️  UNSAFE MODE: Discovery safety checks bypassed!"
    end
    println()

    # Check if already configured
    config_path = joinpath(target_dir, "replibuild.toml")
    if isfile(config_path) && !force
        println("⚠️  replibuild.toml already exists!")
        println("   Use discover(force=true) to regenerate")
        return ConfigurationManager.load_config(config_path)
    end

    # Stage 1: Scan files
    println("📂 Stage 1: Scanning files...")
    scan_results = scan_all_files(target_dir)
    print_scan_summary(scan_results)

    # Stage 2: Detect binaries
    println("\n🔍 Stage 2: Detecting binaries...")
    binaries = detect_all_binaries(target_dir, scan_results)
    print_binary_summary(binaries)

    # Stage 3: Build include directories
    println("\n📚 Stage 3: Building include paths...")
    include_dirs = build_include_dirs(target_dir, scan_results)
    println("   Found $(length(include_dirs)) include directories")

    # Stage 4: Walk AST dependencies
    println("\n🌳 Stage 4: Walking AST dependencies...")
    dep_graph = walk_dependencies(target_dir, scan_results, include_dirs)

    # Stage 5: Generate configuration
    println("\n📝 Stage 5: Generating replibuild.toml...")
    config = generate_config(target_dir, scan_results, binaries, include_dirs, dep_graph)

    # Stage 6: Initialize toolchain and update config with discovered tools
    println("\n🔧 Stage 6: Initializing LLVM toolchain...")
    LLVMEnvironment.init_toolchain(config=config)

    # Move discovered data to compile section for actual use
    if haskey(config.discovery, "include_dirs") && !isempty(config.discovery["include_dirs"])
        config.compile["include_dirs"] = config.discovery["include_dirs"]
        println("   ✓ Copied $(length(config.discovery["include_dirs"])) include directories to compile section")
    end

    # Add discovered source files to compile section
    if haskey(config.discovery, "files")
        all_sources = vcat(
            get(config.discovery["files"], "cpp_sources", String[]),
            get(config.discovery["files"], "c_sources", String[])
        )
        if !isempty(all_sources)
            config.compile["source_files"] = all_sources
            println("   ✓ Added $(length(all_sources)) source files to compile section")
        end
    end

    # Save config with updated tool paths and include dirs
    ConfigurationManager.save_config(config)

    println("\n✅ Discovery complete!")
    println("📄 Configuration: $config_path")
    println("🚀 Next: julia -e 'using RepliBuild; RepliBuild.compile()'")
    println("="^70)

    return config
end

"""
    check_marker(target_dir::String) -> Bool

DEPRECATED: Check if directory has .replibuild_project marker.
Use ConfigurationManager.is_replibuild_project() instead.
"""
function check_marker(target_dir::String)
    @warn "check_marker is deprecated. Use ConfigurationManager.is_replibuild_project() instead."
    return ConfigurationManager.is_replibuild_project(target_dir)
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

    println("   🔒 Scanning scope: $abs_root")

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
        println("   ⚠️  No source files found")
        return nothing
    end

    # Get clang from LLVM environment
    clang_path = LLVMEnvironment.get_tool("clang++")

    if isempty(clang_path)
        @warn "Clang not found in LLVM environment, skipping AST walk"
        return nothing
    end

    # Build dependency graph
    dep_graph = ASTWalker.build_dependency_graph(
        all_sources,
        include_dirs,
        use_clang=true,
        clang_path=clang_path
    )

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

Generate RepliBuild configuration from discovery results.
"""
function generate_config(root_dir::String, scan::ScanResults, binaries::Vector{BinaryInfo},
                        include_dirs::Vector{String}, dep_graph)
    project_name = basename(abspath(root_dir))

    # Create configuration with UUID
    project_uuid = uuid4()  # Generate new UUID for discovered project

    config = ConfigurationManager.RepliBuildConfig(
        joinpath(root_dir, "replibuild.toml"),
        now(),
        "0.1.0",
        project_name,
        abspath(root_dir),
        project_uuid,  # Add the missing UUID parameter
        # Discovery stage
        Dict{String,Any}(
            "enabled" => true,
            "completed" => true,
            "timestamp" => string(now()),
            "files" => Dict(
                "cpp_sources" => scan.cpp_sources,
                "cpp_headers" => scan.cpp_headers,
                "c_sources" => scan.c_sources,
                "c_headers" => scan.c_headers,
                "total_scanned" => scan.total_files
            ),
            "binaries" => Dict(
                "executables" => [b.path for b in binaries if b.type == :executable],
                "static_libs" => [b.path for b in binaries if b.type == :static_lib],
                "shared_libs" => [b.path for b in binaries if b.type == :shared_lib],
                "total" => length(binaries)
            ),
            "include_dirs" => include_dirs,
            "dependency_graph_file" => isnothing(dep_graph) ? "" : ".replibuild_cache/dependency_graph.json"
        ),
        # Reorganize stage (disabled by default)
        Dict{String,Any}(
            "enabled" => false
        ),
        # Compile stage
        Dict{String,Any}(
            "enabled" => true,
            "output_dir" => "build/ir",
            "flags" => ["-std=c++17", "-fPIC", "-O2"],
            "parallel" => true,
            "emit_ir" => true
        ),
        # Link stage
        Dict{String,Any}(
            "enabled" => true,
            "output_dir" => "build/linked",
            "optimize" => true,
            "opt_level" => "O2",
            "lto" => false
        ),
        # Binary stage
        Dict{String,Any}(
            "enabled" => true,
            "output_dir" => "julia",
            "library_name" => "lib$(lowercase(project_name)).so",
            "library_type" => "shared",
            "rpath" => true
        ),
        # Symbols stage
        Dict{String,Any}(
            "enabled" => true,
            "method" => "nm",
            "demangle" => true,
            "filter_internal" => true
        ),
        # Wrap stage
        Dict{String,Any}(
            "enabled" => true,
            "output_dir" => "julia",
            "style" => "auto",
            "module_name" => uppercasefirst(project_name),
            "add_tests" => true,
            "add_docs" => true
        ),
        # Test stage
        Dict{String,Any}(
            "enabled" => false
        ),
        # LLVM configuration (stub - will be populated by init_toolchain)
        Dict{String,Any}(
            "root" => "",  # Will be auto-detected
            "source" => "",  # Will be set to "jll" or "intree"
            "use_replibuild_llvm" => true,
            "isolated" => true,
            "tools" => Dict{String,String}()  # Will be discovered
        ),
        # Target
        Dict{String,Any}(
            "triple" => "",
            "cpu" => "generic",
            "features" => String[]
        ),
        # Workflow
        Dict{String,Any}(
            "stages" => ["discovery", "compile", "link", "binary", "symbols", "wrap"],
            "stop_on_error" => true,
            "parallel_stages" => ["compile"]
        ),
        # Cache
        Dict{String,Any}(
            "enabled" => true,
            "directory" => ".replibuild_cache",
            "invalidate_on_change" => true
        ),
        # Raw data
        Dict{String,Any}()
    )

    return config
end

"""
    print_scan_summary(scan::ScanResults)

Print file scan summary.
"""
function print_scan_summary(scan::ScanResults)
    println("   📊 Scan Results:")
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
        println("   📦 No binaries detected")
        return
    end

    println("   📦 Detected $(length(binaries)) binaries:")

    for binary in binaries
        type_icon = binary.type == :executable ? "⚙️" :
                   binary.type == :shared_lib ? "📚" : "📕"
        size_kb = round(binary.size / 1024, digits=1)
        println("      $type_icon $(binary.name) ($(binary.type), $(size_kb) KB)")
    end
end

# Exports
export discover

end # module Discovery
