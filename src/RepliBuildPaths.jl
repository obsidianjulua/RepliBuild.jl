#!/usr/bin/env julia
# RepliBuildPaths.jl - Centralized path management following Julia conventions
# All paths are user-local (~/.julia/replibuild/), never system-wide

module RepliBuildPaths

using TOML
using Dates

export get_replibuild_dir, get_module_search_paths, get_cache_dir
export get_config_path, initialize_directories, ensure_initialized
export get_registry_dir, get_logs_dir, print_paths_info
export get_config_value, set_config_value, migrate_old_structure
export get_project_replibuild_dir, get_project_cache_dir, get_project_modules_dir
export initialize_project_directories

"""
Get the main RepliBuild directory (~/.julia/replibuild/)
Can be overridden with JULIA_REPLIBUILD_DIR environment variable.
"""
function get_replibuild_dir()::String
    # Check environment variable first
    if haskey(ENV, "JULIA_REPLIBUILD_DIR")
        return abspath(ENV["JULIA_REPLIBUILD_DIR"])
    end

    # Default: ~/.julia/replibuild
    julia_dir = first(DEPOT_PATH)  # Usually ~/.julia
    return joinpath(julia_dir, "replibuild")
end

"""
Get the cache directory.
Can be overridden with REPLIBUILD_CACHE_DIR environment variable.
"""
function get_cache_dir()::String
    if haskey(ENV, "REPLIBUILD_CACHE_DIR")
        return abspath(ENV["REPLIBUILD_CACHE_DIR"])
    end

    return joinpath(get_replibuild_dir(), "cache")
end

"""
Get the logs directory.
"""
function get_logs_dir()::String
    return joinpath(get_replibuild_dir(), "logs")
end

"""
Get the registries directory (for module registries).
"""
function get_registry_dir()::String
    return joinpath(get_replibuild_dir(), "registries")
end

"""
Get the global config file path.
"""
function get_config_path()::String
    return joinpath(get_replibuild_dir(), "config.toml")
end

"""
Get module search paths in priority order:
1. Environment variable (REPLIBUILD_MODULE_PATH)
2. User modules directory (~/.julia/replibuild/modules/)
3. Config file specified paths
4. Built-in modules (from package installation)
"""
function get_module_search_paths()::Vector{String}
    paths = String[]

    # 1. Environment variable (highest priority)
    if haskey(ENV, "REPLIBUILD_MODULE_PATH")
        env_paths = split(ENV["REPLIBUILD_MODULE_PATH"], Sys.iswindows() ? ';' : ':')
        for p in env_paths
            expanded = abspath(expanduser(String(strip(p))))
            if isdir(expanded)
                push!(paths, expanded)
            end
        end
    end

    # 2. User modules directory
    user_modules = joinpath(get_replibuild_dir(), "modules")
    if isdir(user_modules)
        push!(paths, user_modules)
    end

    # 3. Config file paths
    config_file = get_config_path()
    if isfile(config_file)
        try
            config = TOML.parsefile(config_file)
            if haskey(config, "modules") && haskey(config["modules"], "search_paths")
                for p in config["modules"]["search_paths"]
                    expanded = abspath(expanduser(p))
                    if isdir(expanded)
                        push!(paths, expanded)
                    end
                end
            end
        catch e
            @warn "Failed to read config file for module paths" exception=e
        end
    end

    # 4. Built-in modules (from package installation)
    # __DIR__ is RepliBuild.jl/src, go up one level to find modules/
    builtin_modules = joinpath(dirname(@__DIR__), "modules")
    if isdir(builtin_modules)
        push!(paths, builtin_modules)
    end

    # Remove duplicates while preserving order
    return unique(paths)
end

"""
Create default configuration file if it doesn't exist.
"""
function create_default_config()
    config_path = get_config_path()

    if isfile(config_path)
        return  # Already exists
    end

    config = Dict{String,Any}(
        "version" => "0.1.0",
        "last_updated" => string(now()),

        "cache" => Dict(
            "enabled" => true,
            "max_size_gb" => 50,
            "cleanup_after_days" => 90
        ),

        "modules" => Dict(
            "search_paths" => String[],
            "registries" => String[]
        ),

        "build" => Dict(
            "parallel_jobs" => Sys.CPU_THREADS,
            "default_optimization" => "O2",
            "cache_ir" => true,
            "cache_objects" => true
        ),

        "llvm" => Dict(
            "prefer_source" => "jll",
            "isolated" => true
        ),

        "logging" => Dict(
            "level" => "info",
            "keep_logs" => 100
        ),

        "error_learning" => Dict(
            "enabled" => true,
            "share_anonymous_errors" => false
        )
    )

    open(config_path, "w") do io
        println(io, "# RepliBuild Global Configuration")
        println(io, "# Created: $(now())")
        println(io, "# Location: $config_path")
        println(io, "")
        TOML.print(io, config)
    end

    println("Created default config: $config_path")
end

"""
Initialize all RepliBuild directories.
Called automatically on first use or can be called manually.
"""
function initialize_directories(; verbose::Bool=true)
    base_dir = get_replibuild_dir()

    dirs_to_create = [
        base_dir,
        joinpath(base_dir, "modules"),
        joinpath(base_dir, "cache"),
        joinpath(base_dir, "cache", "toolchains"),
        joinpath(base_dir, "cache", "modules"),
        joinpath(base_dir, "cache", "builds"),
        joinpath(base_dir, "registries"),
        joinpath(base_dir, "logs")
    ]

    newly_created = String[]

    for dir in dirs_to_create
        if !isdir(dir)
            mkpath(dir)
            push!(newly_created, dir)
        end
    end

    # Create config if needed
    create_default_config()

    if verbose && !isempty(newly_created)
        println("Initialized RepliBuild directories:")
        println("   Base: $base_dir")
        for dir in newly_created
            rel_path = relpath(dir, base_dir)
            println("    $rel_path")
        end
        println(" RepliBuild ready")
    end

    return base_dir
end

"""
Check if RepliBuild is initialized, initialize if not.
This is called automatically by most RepliBuild operations.
"""
function ensure_initialized(; verbose::Bool=false)
    base_dir = get_replibuild_dir()

    if !isdir(base_dir)
        if verbose
            println("First-time setup: Initializing RepliBuild...")
        end
        initialize_directories(verbose=verbose)
    end

    return true
end

"""
Get project-local RepliBuild directory (.replibuild/ in project root).
"""
function get_project_replibuild_dir(project_root::String)::String
    return joinpath(project_root, ".replibuild")
end

"""
Get project-local cache directory.
"""
function get_project_cache_dir(project_root::String)::String
    return joinpath(get_project_replibuild_dir(project_root), "cache")
end

"""
Get project-local modules directory.
"""
function get_project_modules_dir(project_root::String)::String
    return joinpath(get_project_replibuild_dir(project_root), "modules")
end

"""
Initialize project-local directories.
"""
function initialize_project_directories(project_root::String; verbose::Bool=true)
    rb_dir = get_project_replibuild_dir(project_root)

    dirs_to_create = [
        rb_dir,
        joinpath(rb_dir, "cache"),
        joinpath(rb_dir, "modules"),
        joinpath(rb_dir, "logs")
    ]

    newly_created = String[]

    for dir in dirs_to_create
        if !isdir(dir)
            mkpath(dir)
            push!(newly_created, dir)
        end
    end

    if verbose && !isempty(newly_created)
        println(" Initialized project RepliBuild directories:")
        for dir in newly_created
            rel_path = relpath(dir, project_root)
            println("   $rel_path")
        end
    end

    return rb_dir
end

"""
Get configuration value from global config.
"""
function get_config_value(key::String, default=nothing)
    config_file = get_config_path()

    if !isfile(config_file)
        return default
    end

    try
        config = TOML.parsefile(config_file)
        # Support nested keys like "cache.max_size_gb"
        keys = split(key, '.')
        value = config
        for k in keys
            if haskey(value, k)
                value = value[k]
            else
                return default
            end
        end
        return value
    catch
        return default
    end
end

"""
Set configuration value in global config.
"""
function set_config_value(key::String, value)
    config_file = get_config_path()

    # Load existing config or create new
    config = if isfile(config_file)
        TOML.parsefile(config_file)
    else
        Dict{String,Any}()
    end

    # Support nested keys
    keys = split(key, '.')
    current = config

    for (i, k) in enumerate(keys)
        if i == length(keys)
            # Last key - set value
            current[k] = value
        else
            # Intermediate key - ensure dict exists
            if !haskey(current, k)
                current[k] = Dict{String,Any}()
            end
            current = current[k]
        end
    end

    # Update timestamp
    config["last_updated"] = string(now())

    # Write back
    open(config_file, "w") do io
        TOML.print(io, config)
    end

    println(" Config updated: $key = $value")
end

"""
Print information about RepliBuild directories and cache.
"""
function print_paths_info()
    println("="^70)
    println("RepliBuild Directory Structure")
    println("="^70)
    println()

    base = get_replibuild_dir()
    println(" Base directory: $base")
    println("   Exists: $(isdir(base))")

    if isdir(base)
        # Calculate directory sizes
        cache_dir = get_cache_dir()
        modules_dir = joinpath(base, "modules")

        println()
        println(" Subdirectories:")
        println("   Modules: $modules_dir")
        if isdir(modules_dir)
            n_modules = length(filter(f -> endswith(f, ".toml"), readdir(modules_dir)))
            println("      $(n_modules) module files")
        end

        println("   Cache: $cache_dir")
        if isdir(cache_dir)
            cache_size = try
                sum(map(f -> filesize(joinpath(cache_dir, f)), readdir(cache_dir))) / 1024^3
            catch
                0.0
            end
            println("      ~$(round(cache_size, digits=2)) GB")
        end

        println("   Registries: $(get_registry_dir())")
        println("   Logs: $(get_logs_dir())")
    end

    println()
    println(" Module search paths:")
    for (i, path) in enumerate(get_module_search_paths())
        exists = isdir(path)
        status = exists ? "✓" : "✗"
        println("   $i. $status $path")
    end

    println()
    println("  Configuration: $(get_config_path())")
    println("  Exists: $(isfile(get_config_path()))")

    println("="^70)
end

"""
Migrate old ~/.replibuild structure to new ~/.julia/replibuild location.
"""
function migrate_old_structure()
    old_dir = joinpath(homedir(), ".replibuild")
    new_dir = get_replibuild_dir()

    if !isdir(old_dir)
        println("ℹ️  No old structure found at $old_dir")
        return false
    end

    if isdir(new_dir)
        println("  New directory already exists: $new_dir")
        println("    Manual migration required")
        return false
    end

    println("🔄 Migrating from $old_dir to $new_dir")

    # Move entire directory
    try
        mv(old_dir, new_dir)
        println(" Migration complete")
        println("   Old: $old_dir → New: $new_dir")
        return true
    catch e
        println("❌ Migration failed: $e")
        println("   Please manually move: mv $old_dir $new_dir")
        return false
    end
end

end # module RepliBuildPaths
