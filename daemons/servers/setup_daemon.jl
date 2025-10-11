#!/usr/bin/env julia
"""
Setup Daemon Server - Fast configuration and directory structure management

Start with: julia --project=.. setup_daemon.jl
Port: 3002

Features:
- Cached directory structure templates
- TOML generation/validation without repeated I/O
- Include path resolution with caching
- ConfigurationManager integration
"""

using DaemonMode
using Dates

# Add project to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))

using RepliBuild
using RepliBuild.ConfigurationManager
using RepliBuild.Templates

const PORT = 3002

# ============================================================================
# PERSISTENT CACHES
# ============================================================================

const CONFIG_CACHE = Dict{String, ConfigurationManager.RepliBuildConfig}()  # path => config
const TEMPLATE_CACHE = Dict{Symbol, Dict}()  # template_type => template_data

"""
Initialize directory structure templates
"""
function init_template_cache!()
    println("[SETUP] Initializing template cache...")

    # C++ project template
    TEMPLATE_CACHE[:cpp_project] = Dict(
        "dirs" => ["src", "include", "julia", "build", "test", ".replibuild_cache"],
        "files" => Dict(
            ".replibuild_project" => "",
            ".gitignore" => "build/\n.replibuild_cache/\n*.so\n*.o\n*.ir\n*.bc\n"
        )
    )

    # Binary wrapping template
    TEMPLATE_CACHE[:binary_project] = Dict(
        "dirs" => ["lib", "bin", "julia_wrappers", ".replibuild_cache"],
        "files" => Dict(
            ".replibuild_project" => "",
            ".gitignore" => ".replibuild_cache/\njulia_wrappers/\n"
        )
    )

    println("[SETUP] Cached $(length(TEMPLATE_CACHE)) templates")
end

# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

"""
Create directory structure from template
"""
function create_structure(args::Dict)
    target_dir = get(args, "path", pwd())
    project_type = Symbol(get(args, "type", "cpp_project"))

    println("[SETUP] Creating project structure: $target_dir ($project_type)")

    try
        # Get template from cache
        if !haskey(TEMPLATE_CACHE, project_type)
            return Dict(
                :success => false,
                :error => "Unknown project type: $project_type"
            )
        end

        template = TEMPLATE_CACHE[project_type]

        # Create directories
        created_dirs = String[]
        for dir in template["dirs"]
            dir_path = joinpath(target_dir, dir)
            if !isdir(dir_path)
                mkpath(dir_path)
                push!(created_dirs, dir)
            end
        end

        # Create files
        created_files = String[]
        for (filename, content) in template["files"]
            file_path = joinpath(target_dir, filename)
            if !isfile(file_path)
                write(file_path, content)
                push!(created_files, filename)
            end
        end

        return Dict(
            :success => true,
            :created_dirs => created_dirs,
            :created_files => created_files,
            :project_type => project_type
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e),
            :stacktrace => sprint(showerror, e, catch_backtrace())
        )
    end
end

"""
Generate or load configuration
"""
function generate_config(args::Dict)
    target_dir = get(args, "path", pwd())
    force = get(args, "force", false)
    discovery_results = get(args, "discovery_results", nothing)

    println("[SETUP] Generating configuration: $target_dir")

    config_path = joinpath(target_dir, "replibuild.toml")

    try
        # Check cache first
        if !force && haskey(CONFIG_CACHE, config_path)
            cached_config = CONFIG_CACHE[config_path]
            # Check if file has been modified since cache
            if isfile(config_path) && mtime(config_path) <= cached_config.last_modified.value / 1000
                println("[SETUP] Using cached configuration")
                return Dict(
                    :success => true,
                    :cached => true,
                    :config_path => config_path
                )
            end
        end

        # Load or create config
        if isfile(config_path) && !force
            config = ConfigurationManager.load_config(config_path)
        else
            config = ConfigurationManager.create_default_config(config_path)
        end

        # Update with discovery results if provided
        if !isnothing(discovery_results)
            println("[SETUP] Updating config with discovery results...")

            # Update discovery section
            if haskey(discovery_results, :scan) && discovery_results[:scan][:success]
                scan_data = discovery_results[:scan][:results]

                ConfigurationManager.set_source_files(config, Dict(
                    "cpp_sources" => scan_data.cpp_sources,
                    "cpp_headers" => scan_data.cpp_headers,
                    "c_sources" => scan_data.c_sources,
                    "c_headers" => scan_data.c_headers
                ))
            end

            # Update include directories
            if haskey(discovery_results, :include_dirs)
                ConfigurationManager.set_include_dirs(config, discovery_results[:include_dirs])
            end

            # Update dependency graph reference
            if haskey(discovery_results, :ast) && discovery_results[:ast][:success]
                ast_data = discovery_results[:ast]
                if haskey(ast_data, :graph) && !isnothing(ast_data[:graph])
                    config.discovery["dependency_graph_file"] = ".replibuild_cache/dependency_graph.json"
                end
            end

            # Save updated config
            ConfigurationManager.save_config(config)
        end

        # Cache the config
        CONFIG_CACHE[config_path] = config

        return Dict(
            :success => true,
            :cached => false,
            :config_path => config_path,
            :config => config
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e),
            :stacktrace => sprint(showerror, e, catch_backtrace())
        )
    end
end

"""
Validate configuration
"""
function validate_config(args::Dict)
    config_path = get(args, "config", "replibuild.toml")

    println("[SETUP] Validating configuration: $config_path")

    try
        if !isfile(config_path)
            return Dict(
                :success => false,
                :valid => false,
                :error => "Configuration file not found: $config_path"
            )
        end

        config = ConfigurationManager.load_config(config_path)

        # Validation checks
        issues = String[]

        # Check required stages are enabled
        if !ConfigurationManager.is_stage_enabled(config, :compile)
            push!(issues, "Compile stage is disabled")
        end

        # Check source files exist
        source_files = ConfigurationManager.get_source_files(config)
        if isempty(source_files)
            push!(issues, "No source files configured")
        else
            # Verify files exist
            for (file_type, files) in source_files
                for file in files
                    file_path = joinpath(config.project_root, file)
                    if !isfile(file_path)
                        push!(issues, "Source file not found: $file")
                    end
                end
            end
        end

        # Check include directories exist
        include_dirs = ConfigurationManager.get_include_dirs(config)
        for dir in include_dirs
            if !isdir(dir)
                push!(issues, "Include directory not found: $dir")
            end
        end

        # Check LLVM tools
        if haskey(config.llvm, "tools")
            for (tool_name, tool_path) in config.llvm["tools"]
                if !isfile(tool_path)
                    push!(issues, "LLVM tool not found: $tool_name at $tool_path")
                end
            end
        end

        valid = isempty(issues)

        return Dict(
            :success => true,
            :valid => valid,
            :issues => issues,
            :config => config
        )

    catch e
        return Dict(
            :success => false,
            :valid => false,
            :error => string(e)
        )
    end
end

"""
Update configuration section
"""
function update_config(args::Dict)
    config_path = get(args, "config", "replibuild.toml")
    section = Symbol(get(args, "section", ""))
    data = get(args, "data", Dict())

    println("[SETUP] Updating config section: $section")

    try
        config = ConfigurationManager.load_config(config_path)

        # Update the appropriate section
        if section == :discovery
            merge!(config.discovery, data)
        elseif section == :compile
            merge!(config.compile, data)
        elseif section == :link
            merge!(config.link, data)
        elseif section == :binary
            merge!(config.binary, data)
        elseif section == :symbols
            merge!(config.symbols, data)
        elseif section == :wrap
            merge!(config.wrap, data)
        elseif section == :llvm
            merge!(config.llvm, data)
        elseif section == :target
            merge!(config.target, data)
        elseif section == :workflow
            merge!(config.workflow, data)
        else
            return Dict(
                :success => false,
                :error => "Unknown section: $section"
            )
        end

        # Save and cache
        ConfigurationManager.save_config(config)
        CONFIG_CACHE[config_path] = config

        return Dict(
            :success => true,
            :section => section,
            :updated => true
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e)
        )
    end
end

"""
Get configuration section
"""
function get_config_section(args::Dict)
    config_path = get(args, "config", "replibuild.toml")
    section = Symbol(get(args, "section", ""))

    try
        config = ConfigurationManager.load_config(config_path)

        section_data = if section == :discovery
            config.discovery
        elseif section == :compile
            config.compile
        elseif section == :link
            config.link
        elseif section == :binary
            config.binary
        elseif section == :symbols
            config.symbols
        elseif section == :wrap
            config.wrap
        elseif section == :llvm
            config.llvm
        elseif section == :target
            config.target
        elseif section == :workflow
            config.workflow
        else
            return Dict(:success => false, :error => "Unknown section: $section")
        end

        return Dict(
            :success => true,
            :section => section,
            :data => section_data
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e)
        )
    end
end

"""
Clear configuration cache
"""
function clear_cache(args::Dict)
    println("[SETUP] Clearing configuration cache...")
    empty!(CONFIG_CACHE)

    return Dict(
        :success => true,
        :message => "Configuration cache cleared"
    )
end

"""
Get cache statistics
"""
function cache_stats(args::Dict)
    return Dict(
        :success => true,
        :stats => Dict(
            "cached_configs" => length(CONFIG_CACHE),
            "templates" => length(TEMPLATE_CACHE)
        )
    )
end

# ============================================================================
# MAIN
# ============================================================================

"""
Main daemon server function
"""
function main()
    println("="^70)
    println("RepliBuild Setup Daemon Server")
    println("Port: $PORT")
    println("="^70)

    # Initialize template cache
    init_template_cache!()

    println()
    println("Available Functions:")
    println("  • create_structure(path, type='cpp_project')")
    println("  • generate_config(path, force=false, discovery_results=nothing)")
    println("  • validate_config(config='replibuild.toml')")
    println("  • update_config(config, section, data)")
    println("  • get_config_section(config, section)")
    println("  • cache_stats()")
    println("  • clear_cache()")
    println()
    println("Ready to accept setup requests...")
    println("="^70)

    # Start the daemon server
    serve(PORT)
end

# Start the daemon if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
