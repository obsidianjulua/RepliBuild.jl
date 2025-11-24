#!/usr/bin/env julia
# ConfigurationManager.jl - Centralized immutable configuration for RepliBuild
# Single source of truth for all build configuration
# All TOML parsing happens here - other modules receive pre-parsed config

module ConfigurationManager

using TOML
using Dates
using UUIDs

# =============================================================================
# IMMUTABLE CONFIGURATION STRUCTS
# =============================================================================

"""Nested struct for [project] section"""
struct ProjectConfig
    name::String
    root::String
    uuid::UUID
end

"""Nested struct for [paths] section"""
struct PathsConfig
    source::String
    include::String
    output::String
    build::String
    cache::String
end

"""Nested struct for [discovery] section"""
struct DiscoveryConfig
    enabled::Bool
    walk_dependencies::Bool
    max_depth::Int
    ignore_patterns::Vector{String}
    parse_ast::Bool
end

"""Nested struct for [compile] section"""
struct CompileConfig
    source_files::Vector{String}  # Can be empty (auto-discovered)
    include_dirs::Vector{String}  # Can be empty (auto-discovered)
    flags::Vector{String}
    defines::Dict{String,String}
    parallel::Bool
end

"""Nested struct for [link] section"""
struct LinkConfig
    optimization_level::String
    enable_lto::Bool
    link_libraries::Vector{String}
end

"""Nested struct for [binary] section"""
struct BinaryConfig
    type::Symbol  # :shared, :static, :executable
    output_name::String  # Empty = auto from project.name
    strip_symbols::Bool
end

"""Nested struct for [wrap] section"""
struct WrapConfig
    enabled::Bool
    style::Symbol  # :clang, :basic, :none
    module_name::String  # Empty = auto from project.name
    use_clang_jl::Bool
end

"""Nested struct for [llvm] section"""
struct LLVMConfig
    toolchain::Symbol  # :auto, :system, :jll
    version::String  # Empty = auto-detect
end

"""Nested struct for [workflow] section"""
struct WorkflowConfig
    stages::Vector{Symbol}
end

"""Nested struct for [cache] section"""
struct CacheConfig
    enabled::Bool
    directory::String
end

"""
Main immutable configuration structure.
All modules receive this struct - it's the single source of truth.
"""
struct RepliBuildConfig
    # Nested configs
    project::ProjectConfig
    paths::PathsConfig
    discovery::DiscoveryConfig
    compile::CompileConfig
    link::LinkConfig
    binary::BinaryConfig
    wrap::WrapConfig
    llvm::LLVMConfig
    workflow::WorkflowConfig
    cache::CacheConfig

    # Metadata
    config_file::String
    loaded_at::DateTime
end

# =============================================================================
# TOML PARSING (Single place where TOML is read)
# =============================================================================

"""
Load RepliBuildConfig from TOML file.
This is the ONLY function that parses TOML - all other modules use this.
"""
function load_config(toml_path::String="replibuild.toml")::RepliBuildConfig
    if !isfile(toml_path)
        @warn "Config file not found: $toml_path - creating default"
        return create_default_config(toml_path)
    end

    data = TOML.parsefile(toml_path)

    # Validate required sections
    validate_toml_data(data, toml_path)

    # Parse each section with defaults
    config = RepliBuildConfig(
        parse_project_config(data, toml_path),
        parse_paths_config(data),
        parse_discovery_config(data),
        parse_compile_config(data),
        parse_link_config(data),
        parse_binary_config(data),
        parse_wrap_config(data),
        parse_llvm_config(data),
        parse_workflow_config(data),
        parse_cache_config(data),
        toml_path,
        now()
    )

    return config
end

"""Parse [project] section"""
function parse_project_config(data::Dict, toml_path::String)::ProjectConfig
    project = get(data, "project", Dict())

    # Defaults
    default_root = dirname(abspath(toml_path))
    default_name = basename(default_root)

    name = get(project, "name", default_name)
    root = get(project, "root", default_root)

    # Parse or generate UUID
    uuid = if haskey(project, "uuid")
        try
            UUID(project["uuid"])
        catch
            @warn "Invalid UUID in config, generating new one"
            uuid4()
        end
    else
        uuid4()
    end

    return ProjectConfig(name, root, uuid)
end

"""Parse [paths] section"""
function parse_paths_config(data::Dict)::PathsConfig
    paths = get(data, "paths", Dict())

    return PathsConfig(
        get(paths, "source", "src"),
        get(paths, "include", "include"),
        get(paths, "output", "julia"),
        get(paths, "build", "build"),
        get(paths, "cache", ".replibuild_cache")
    )
end

"""Parse [discovery] section"""
function parse_discovery_config(data::Dict)::DiscoveryConfig
    discovery = get(data, "discovery", Dict())

    return DiscoveryConfig(
        get(discovery, "enabled", true),
        get(discovery, "walk_dependencies", true),
        get(discovery, "max_depth", 10),
        get(discovery, "ignore_patterns", ["build", ".git", ".cache"]),
        get(discovery, "parse_ast", true)
    )
end

"""Parse [compile] section"""
function parse_compile_config(data::Dict)::CompileConfig
    compile = get(data, "compile", Dict())

    return CompileConfig(
        get(compile, "source_files", String[]),
        get(compile, "include_dirs", String[]),
        get(compile, "flags", ["-std=c++17", "-fPIC"]),
        get(compile, "defines", Dict{String,String}()),
        get(compile, "parallel", true)
    )
end

"""Parse [link] section"""
function parse_link_config(data::Dict)::LinkConfig
    link = get(data, "link", Dict())

    return LinkConfig(
        get(link, "optimization_level", "2"),
        get(link, "enable_lto", false),
        get(link, "link_libraries", String[])
    )
end

"""Parse [binary] section"""
function parse_binary_config(data::Dict)::BinaryConfig
    binary = get(data, "binary", Dict())

    # Parse type as symbol
    type_str = get(binary, "type", "shared")
    binary_type = Symbol(type_str)

    # Validate type
    if !(binary_type in [:shared, :static, :executable])
        @warn "Invalid binary.type: $type_str, using :shared"
        binary_type = :shared
    end

    return BinaryConfig(
        binary_type,
        get(binary, "output_name", ""),  # Empty = auto-generate
        get(binary, "strip_symbols", false)
    )
end

"""Parse [wrap] section"""
function parse_wrap_config(data::Dict)::WrapConfig
    wrap = get(data, "wrap", Dict())

    # Parse style as symbol
    style_str = get(wrap, "style", "clang")
    wrap_style = Symbol(style_str)

    # Validate style
    if !(wrap_style in [:clang, :basic, :none])
        @warn "Invalid wrap.style: $style_str, using :clang"
        wrap_style = :clang
    end

    return WrapConfig(
        get(wrap, "enabled", true),
        wrap_style,
        get(wrap, "module_name", ""),  # Empty = auto-generate
        get(wrap, "use_clang_jl", true)
    )
end

"""Parse [llvm] section"""
function parse_llvm_config(data::Dict)::LLVMConfig
    llvm = get(data, "llvm", Dict())

    # Parse toolchain as symbol
    toolchain_str = get(llvm, "toolchain", "auto")
    toolchain = Symbol(toolchain_str)

    # Validate toolchain
    if !(toolchain in [:auto, :system, :jll])
        @warn "Invalid llvm.toolchain: $toolchain_str, using :auto"
        toolchain = :auto
    end

    return LLVMConfig(
        toolchain,
        get(llvm, "version", "")  # Empty = auto-detect
    )
end

"""Parse [workflow] section"""
function parse_workflow_config(data::Dict)::WorkflowConfig
    workflow = get(data, "workflow", Dict())

    # Parse stages as symbols
    stages_raw = get(workflow, "stages", ["discover", "compile", "link", "binary", "wrap"])
    stages = Symbol[Symbol(s) for s in stages_raw]

    return WorkflowConfig(stages)
end

"""Parse [cache] section"""
function parse_cache_config(data::Dict)::CacheConfig
    cache = get(data, "cache", Dict())

    return CacheConfig(
        get(cache, "enabled", true),
        get(cache, "directory", ".replibuild_cache")
    )
end

"""Validate TOML data has minimum required fields"""
function validate_toml_data(data::Dict, toml_path::String)
    # Project name is the only truly required field
    if !haskey(data, "project")
        @warn "Missing [project] section in $toml_path"
    elseif !haskey(data["project"], "name")
        @warn "Missing project.name in $toml_path"
    end
end

# =============================================================================
# CONFIGURATION CREATION
# =============================================================================

"""
Create a default configuration and save to file.
"""
function create_default_config(toml_path::String="replibuild.toml")::RepliBuildConfig
    project_root = dirname(abspath(toml_path))
    project_name = basename(project_root)

    config = RepliBuildConfig(
        ProjectConfig(project_name, project_root, uuid4()),
        PathsConfig("src", "include", "julia", "build", ".replibuild_cache"),
        DiscoveryConfig(true, true, 10, ["build", ".git", ".cache"], true),
        CompileConfig(String[], String[], ["-std=c++17", "-fPIC"], Dict{String,String}(), true),
        LinkConfig("2", false, String[]),
        BinaryConfig(:shared, "", false),
        WrapConfig(true, :clang, "", true),
        LLVMConfig(:auto, ""),
        WorkflowConfig([:discover, :compile, :link, :binary, :wrap]),
        CacheConfig(true, ".replibuild_cache"),
        toml_path,
        now()
    )

    save_config(config)
    println("✅ Created default configuration: $toml_path")

    return config
end

"""
Save configuration to TOML file.
Only saves user-configurable settings (not runtime data).
"""
function save_config(config::RepliBuildConfig)
    data = Dict{String,Any}()

    # [project]
    data["project"] = Dict(
        "name" => config.project.name,
        "root" => config.project.root,
        "uuid" => string(config.project.uuid)
    )

    # [paths]
    data["paths"] = Dict(
        "source" => config.paths.source,
        "include" => config.paths.include,
        "output" => config.paths.output,
        "build" => config.paths.build,
        "cache" => config.paths.cache
    )

    # [discovery]
    data["discovery"] = Dict(
        "enabled" => config.discovery.enabled,
        "walk_dependencies" => config.discovery.walk_dependencies,
        "max_depth" => config.discovery.max_depth,
        "ignore_patterns" => config.discovery.ignore_patterns,
        "parse_ast" => config.discovery.parse_ast
    )

    # [compile]
    compile_dict = Dict(
        "flags" => config.compile.flags,
        "parallel" => config.compile.parallel
    )
    # Only save if non-empty
    if !isempty(config.compile.source_files)
        compile_dict["source_files"] = config.compile.source_files
    end
    if !isempty(config.compile.include_dirs)
        compile_dict["include_dirs"] = config.compile.include_dirs
    end
    if !isempty(config.compile.defines)
        compile_dict["defines"] = config.compile.defines
    end
    data["compile"] = compile_dict

    # [link]
    link_dict = Dict(
        "optimization_level" => config.link.optimization_level,
        "enable_lto" => config.link.enable_lto
    )
    if !isempty(config.link.link_libraries)
        link_dict["link_libraries"] = config.link.link_libraries
    end
    data["link"] = link_dict

    # [binary]
    binary_dict = Dict(
        "type" => string(config.binary.type),
        "strip_symbols" => config.binary.strip_symbols
    )
    if !isempty(config.binary.output_name)
        binary_dict["output_name"] = config.binary.output_name
    end
    data["binary"] = binary_dict

    # [wrap]
    wrap_dict = Dict(
        "enabled" => config.wrap.enabled,
        "style" => string(config.wrap.style),
        "use_clang_jl" => config.wrap.use_clang_jl
    )
    if !isempty(config.wrap.module_name)
        wrap_dict["module_name"] = config.wrap.module_name
    end
    data["wrap"] = wrap_dict

    # [llvm]
    llvm_dict = Dict(
        "toolchain" => string(config.llvm.toolchain)
    )
    if !isempty(config.llvm.version)
        llvm_dict["version"] = config.llvm.version
    end
    data["llvm"] = llvm_dict

    # [workflow]
    data["workflow"] = Dict(
        "stages" => [string(s) for s in config.workflow.stages]
    )

    # [cache]
    data["cache"] = Dict(
        "enabled" => config.cache.enabled,
        "directory" => config.cache.directory
    )

    # Write to file
    open(config.config_file, "w") do io
        TOML.print(io, data)
    end
end

# =============================================================================
# CONFIGURATION HELPERS (Overrides and Merging)
# =============================================================================

"""
Merge compile flags into config (creates new config).
Used for runtime overrides like: compile(sources, flags=["-O3"])
"""
function merge_compile_flags(config::RepliBuildConfig, additional_flags::Vector{String})::RepliBuildConfig
    new_flags = vcat(config.compile.flags, additional_flags)

    new_compile = CompileConfig(
        config.compile.source_files,
        config.compile.include_dirs,
        new_flags,  # Updated
        config.compile.defines,
        config.compile.parallel
    )

    return RepliBuildConfig(
        config.project, config.paths, config.discovery,
        new_compile,  # Updated
        config.link, config.binary, config.wrap,
        config.llvm, config.workflow, config.cache,
        config.config_file, config.loaded_at
    )
end

"""
Update source files in config (creates new config).
Used after discovery finds C++ sources.
"""
function with_source_files(config::RepliBuildConfig, source_files::Vector{String})::RepliBuildConfig
    new_compile = CompileConfig(
        source_files,  # Updated
        config.compile.include_dirs,
        config.compile.flags,
        config.compile.defines,
        config.compile.parallel
    )

    return RepliBuildConfig(
        config.project, config.paths, config.discovery,
        new_compile,  # Updated
        config.link, config.binary, config.wrap,
        config.llvm, config.workflow, config.cache,
        config.config_file, config.loaded_at
    )
end

"""
Update include directories in config (creates new config).
Used after discovery finds include paths.
"""
function with_include_dirs(config::RepliBuildConfig, include_dirs::Vector{String})::RepliBuildConfig
    new_compile = CompileConfig(
        config.compile.source_files,
        include_dirs,  # Updated
        config.compile.flags,
        config.compile.defines,
        config.compile.parallel
    )

    return RepliBuildConfig(
        config.project, config.paths, config.discovery,
        new_compile,  # Updated
        config.link, config.binary, config.wrap,
        config.llvm, config.workflow, config.cache,
        config.config_file, config.loaded_at
    )
end

"""
Update both source files and include dirs (creates new config).
Used after discovery completes.
"""
function with_discovery_results(config::RepliBuildConfig;
                                 source_files::Vector{String}=String[],
                                 include_dirs::Vector{String}=String[])::RepliBuildConfig
    new_compile = CompileConfig(
        isempty(source_files) ? config.compile.source_files : source_files,
        isempty(include_dirs) ? config.compile.include_dirs : include_dirs,
        config.compile.flags,
        config.compile.defines,
        config.compile.parallel
    )

    return RepliBuildConfig(
        config.project, config.paths, config.discovery,
        new_compile,  # Updated
        config.link, config.binary, config.wrap,
        config.llvm, config.workflow, config.cache,
        config.config_file, config.loaded_at
    )
end

# =============================================================================
# ACCESSOR HELPERS
# =============================================================================

"""Get full output path (project_root + paths.output)"""
get_output_path(c::RepliBuildConfig) = joinpath(c.project.root, c.paths.output)

"""Get full build path (project_root + paths.build)"""
get_build_path(c::RepliBuildConfig) = joinpath(c.project.root, c.paths.build)

"""Get full cache path (project_root + cache.directory)"""
get_cache_path(c::RepliBuildConfig) = joinpath(c.project.root, c.cache.directory)

"""Get library output name (uses config or auto-generates)"""
function get_library_name(c::RepliBuildConfig)::String
    if !isempty(c.binary.output_name)
        return c.binary.output_name
    end

    # Auto-generate: lib<project_name>.so
    prefix = c.binary.type == :static ? "lib" : "lib"
    suffix = if c.binary.type == :static
        ".a"
    elseif Sys.isapple()
        ".dylib"
    elseif Sys.iswindows()
        ".dll"
    else
        ".so"
    end

    return prefix * c.project.name * suffix
end

"""Get wrapper module name (uses config or auto-generates)"""
function get_module_name(c::RepliBuildConfig)::String
    if !isempty(c.wrap.module_name)
        return c.wrap.module_name
    end

    # Auto-generate: CamelCase from project name
    name = replace(c.project.name, r"[^a-zA-Z0-9]" => "_")
    parts = split(name, "_")
    return join([uppercasefirst(p) for p in parts if !isempty(p)], "")
end

"""Check if a stage is in the workflow"""
is_stage_enabled(c::RepliBuildConfig, stage::Symbol) = stage in c.workflow.stages

"""Get all C++ source files (from config)"""
get_source_files(c::RepliBuildConfig) = c.compile.source_files

"""Get all include directories (from config)"""
get_include_dirs(c::RepliBuildConfig) = c.compile.include_dirs

"""Get compiler flags"""
get_compile_flags(c::RepliBuildConfig) = c.compile.flags

"""Should run parallel compilation?"""
is_parallel_enabled(c::RepliBuildConfig) = c.compile.parallel

"""Should use cache?"""
is_cache_enabled(c::RepliBuildConfig) = c.cache.enabled

# =============================================================================
# VALIDATION
# =============================================================================

"""
Validate configuration and return list of errors (empty if valid).
"""
function validate_config(config::RepliBuildConfig)::Vector{String}
    errors = String[]

    # Required fields
    if isempty(config.project.name)
        push!(errors, "project.name cannot be empty")
    end

    if !isdir(config.project.root)
        push!(errors, "project.root must exist: $(config.project.root)")
    end

    # Validate binary type
    if !(config.binary.type in [:shared, :static, :executable])
        push!(errors, "binary.type must be :shared, :static, or :executable")
    end

    # Validate wrap style
    if !(config.wrap.style in [:clang, :basic, :none])
        push!(errors, "wrap.style must be :clang, :basic, or :none")
    end

    # Validate llvm toolchain
    if !(config.llvm.toolchain in [:auto, :system, :jll])
        push!(errors, "llvm.toolchain must be :auto, :system, or :jll")
    end

    # Validate workflow stages
    valid_stages = [:discover, :compile, :link, :binary, :wrap]
    for stage in config.workflow.stages
        if !(stage in valid_stages)
            push!(errors, "Unknown workflow stage: $stage")
        end
    end

    return errors
end

"""
Validate config and throw error if invalid.
"""
function validate_config!(config::RepliBuildConfig)
    errors = validate_config(config)
    if !isempty(errors)
        error("Configuration validation failed:\n" * join(errors, "\n"))
    end
end

# =============================================================================
# UTILITIES
# =============================================================================

"""Print configuration summary"""
function print_config(config::RepliBuildConfig)
    println("="^70)
    println("RepliBuild Configuration")
    println("="^70)
    println()
    println("📦 Project:    $(config.project.name)")
    println("🔑 UUID:       $(config.project.uuid)")
    println("📁 Root:       $(config.project.root)")
    println("📄 Config:     $(config.config_file)")
    println("🕐 Loaded:     $(config.loaded_at)")
    println()
    println("🔧 Workflow:   $(join([string(s) for s in config.workflow.stages], " → "))")
    println("🏗️  Binary:     $(config.binary.type)")
    println("📝 Wrapper:    $(config.wrap.style)")
    println("⚙️  LLVM:       $(config.llvm.toolchain)")
    println("🔄 Parallel:   $(config.compile.parallel)")
    println("💾 Cache:      $(config.cache.enabled)")
    println()
    println("📂 Paths:")
    println("   Source:     $(config.paths.source)")
    println("   Include:    $(config.paths.include)")
    println("   Output:     $(config.paths.output)")
    println("   Build:      $(config.paths.build)")
    println("   Cache:      $(config.paths.cache)")
    println("="^70)
end

# =============================================================================
# EXPORTS
# =============================================================================

export RepliBuildConfig,
       ProjectConfig, PathsConfig, DiscoveryConfig, CompileConfig,
       LinkConfig, BinaryConfig, WrapConfig, LLVMConfig,
       WorkflowConfig, CacheConfig,
       load_config, save_config, create_default_config,
       merge_compile_flags, with_source_files, with_include_dirs, with_discovery_results,
       get_output_path, get_build_path, get_cache_path,
       get_library_name, get_module_name,
       is_stage_enabled, get_source_files, get_include_dirs, get_compile_flags,
       is_parallel_enabled, is_cache_enabled,
       validate_config, validate_config!,
       print_config

end # module ConfigurationManager
