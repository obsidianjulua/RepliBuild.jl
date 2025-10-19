#!/usr/bin/env julia
# ConfigurationManager.jl - Unified TOML configuration management for RepliBuild
# Single source of truth for all build stages and component data flow
# All modules read/write through this manager to ensure consistency

module ConfigurationManager

using TOML
using Dates
using UUIDs

# Import UXHelpers for better error messages
import ..UXHelpers

"""
Build stage definitions
Each stage has its own section in the TOML for isolated data
"""
const BUILD_STAGES = [
    :discovery,      # File scanning, dependency walking, AST parsing
    :reorganize,     # File sorting and directory structure creation
    :compile,        # C++ to LLVM IR compilation
    :link,           # IR linking and optimization
    :binary,         # Shared library creation
    :symbols,        # Symbol extraction and analysis
    :wrap,           # Julia wrapper generation
    :test,           # Testing and verification
]

"""
Configuration structure with stage-specific data
"""
mutable struct RepliBuildConfig
    # Metadata
    config_file::String
    last_modified::DateTime
    version::String

    # Project information
    project_name::String
    project_root::String
    project_uuid::UUID

    # Stage: Discovery
    discovery::Dict{String,Any}

    # Stage: Reorganize
    reorganize::Dict{String,Any}

    # Stage: Compile
    compile::Dict{String,Any}

    # Stage: Link
    link::Dict{String,Any}

    # Stage: Binary
    binary::Dict{String,Any}

    # Stage: Symbols
    symbols::Dict{String,Any}

    # Stage: Wrap
    wrap::Dict{String,Any}

    # Stage: Test
    test::Dict{String,Any}

    # LLVM toolchain settings
    llvm::Dict{String,Any}

    # Target configuration
    target::Dict{String,Any}

    # Workflow settings
    workflow::Dict{String,Any}

    # Cache settings
    cache::Dict{String,Any}

    # Raw TOML data (for custom sections)
    raw_data::Dict{String,Any}
end

"""
    load_config(config_file::String="replibuild.toml") -> RepliBuildConfig

Load RepliBuild configuration from TOML file.
Creates default if not exists.
"""
function load_config(config_file::String="replibuild.toml")
    if !isfile(config_file)
        println("📝 Creating default configuration: $config_file")
        return create_default_config(config_file)
    end

    data = TOML.parsefile(config_file)

    # Extract sections with defaults
    project = get(data, "project", Dict())
    discovery = get(data, "discovery", Dict())
    reorganize = get(data, "reorganize", Dict())
    compile = get(data, "compile", Dict())
    link = get(data, "link", Dict())
    binary = get(data, "binary", Dict())
    symbols = get(data, "symbols", Dict())
    wrap = get(data, "wrap", Dict())
    test_section = get(data, "test", Dict())
    llvm = get(data, "llvm", Dict())
    target = get(data, "target", Dict())
    workflow = get(data, "workflow", Dict())
    cache = get(data, "cache", Dict())

    # Extract project root from config file location
    config_dir = dirname(abspath(config_file))

    # Get or generate UUID
    project_uuid = if haskey(project, "uuid")
        UUID(project["uuid"])
    else
        uuid4()  # Generate new UUID if missing
    end

    config = RepliBuildConfig(
        config_file,
        now(),
        get(data, "version", "0.1.0"),
        get(project, "name", basename(config_dir)),
        get(project, "root", config_dir),
        project_uuid,
        discovery,
        reorganize,
        compile,
        link,
        binary,
        symbols,
        wrap,
        test_section,
        llvm,
        target,
        workflow,
        cache,
        data
    )

    return config
end

"""
    save_config(config::RepliBuildConfig)

Save configuration back to TOML file.
All component data flows back through this function.
"""
function save_config(config::RepliBuildConfig)
    # Build TOML structure
    data = Dict{String,Any}()

    # Metadata
    data["version"] = config.version
    data["last_updated"] = string(now())

    # Project
    data["project"] = Dict(
        "name" => config.project_name,
        "root" => config.project_root,
        "uuid" => string(config.project_uuid)
    )

    # Stage-specific sections (only save non-empty)
    # IMPORTANT: Don't serialize large dependency graphs to TOML - they're already in JSON
    if !isempty(config.discovery)
        discovery_toml = copy(config.discovery)
        # Remove heavy data structures that don't belong in TOML
        # Dependency graph is saved separately as JSON
        delete!(discovery_toml, "dependency_graph")
        data["discovery"] = discovery_toml
    end
    if !isempty(config.reorganize)
        data["reorganize"] = config.reorganize
    end
    if !isempty(config.compile)
        data["compile"] = config.compile
    end
    if !isempty(config.link)
        data["link"] = config.link
    end
    if !isempty(config.binary)
        data["binary"] = config.binary
    end
    if !isempty(config.symbols)
        data["symbols"] = config.symbols
    end
    if !isempty(config.wrap)
        data["wrap"] = config.wrap
    end
    if !isempty(config.test)
        data["test"] = config.test
    end

    # System sections
    if !isempty(config.llvm)
        data["llvm"] = config.llvm
    end
    if !isempty(config.target)
        data["target"] = config.target
    end
    if !isempty(config.workflow)
        data["workflow"] = config.workflow
    end
    if !isempty(config.cache)
        data["cache"] = config.cache
    end

    # Write to file
    open(config.config_file, "w") do io
        TOML.print(io, data)
    end

    config.last_modified = now()
end

"""
    merge_config_section!(target::Dict, source::Dict)

Smart merge that preserves existing data:
- Arrays: append unique values
- Dicts: recursively merge
- Values: update only if different
"""
function merge_config_section!(target::Dict, source::Dict)
    for (key, new_value) in source
        if !haskey(target, key)
            # New key - just add it
            target[key] = new_value
        else
            existing_value = target[key]

            # Smart merge based on type
            if isa(existing_value, Vector) && isa(new_value, Vector)
                # Append unique array elements
                for item in new_value
                    if !(item in existing_value)
                        push!(existing_value, item)
                    end
                end
            elseif isa(existing_value, Dict) && isa(new_value, Dict)
                # Recursively merge dicts
                merge_config_section!(existing_value, new_value)
            else
                # Scalar value - update
                target[key] = new_value
            end
        end
    end
    return target
end

"""
    create_default_config(config_file::String) -> RepliBuildConfig

Create default RepliBuild configuration with all stages defined.
"""
function create_default_config(config_file::String)
    project_name = basename(dirname(abspath(config_file)))
    project_root = dirname(abspath(config_file))
    project_uuid = uuid4()  # Generate new UUID for new project

    config = RepliBuildConfig(
        config_file,
        now(),
        "0.1.0",
        project_name,
        project_root,
        project_uuid,
        # Discovery stage
        Dict{String,Any}(
            "enabled" => true,
            "scan_recursive" => true,
            "max_depth" => 10,
            "exclude_dirs" => ["build", ".git", ".cache", "node_modules"],
            "follow_symlinks" => false,
            "parse_ast" => true,
            "walk_dependencies" => true,
            "log_all_files" => true
        ),
        # Reorganize stage
        Dict{String,Any}(
            "enabled" => false,  # Optional stage
            "create_structure" => true,
            "sort_by_type" => true,
            "preserve_hierarchy" => false,
            "target_structure" => Dict(
                "cpp_sources" => "src",
                "cpp_headers" => "include",
                "c_sources" => "src",
                "c_headers" => "include",
                "julia_files" => "julia",
                "config_files" => "config",
                "docs" => "docs"
            )
        ),
        # Compile stage
        Dict{String,Any}(
            "enabled" => true,
            "output_dir" => "build/ir",
            "flags" => ["-std=c++17", "-fPIC"],
            "include_dirs" => String[],  # Populated by discovery
            "defines" => Dict{String,String}(),
            "emit_ir" => true,
            "emit_bc" => false,
            "parallel" => true
        ),
        # Link stage
        Dict{String,Any}(
            "enabled" => true,
            "output_dir" => "build/linked",
            "optimize" => true,
            "opt_level" => "O2",
            "opt_passes" => String[],  # Custom optimization passes
            "lto" => false
        ),
        # Binary stage
        Dict{String,Any}(
            "enabled" => true,
            "output_dir" => "julia",
            "library_name" => "",  # Auto-generated from project name
            "library_type" => "shared",  # shared, static
            "link_libraries" => String[],
            "rpath" => true
        ),
        # Symbols stage
        Dict{String,Any}(
            "enabled" => true,
            "method" => "nm",  # nm, objdump, llvm-nm
            "demangle" => true,
            "filter_internal" => true,
            "export_list" => true
        ),
        # Wrap stage
        Dict{String,Any}(
            "enabled" => true,
            "output_dir" => "julia",
            "style" => "auto",  # auto, basic, advanced, clangjl
            "module_name" => "",  # Auto-generated
            "add_tests" => true,
            "add_docs" => true,
            "type_mappings" => Dict{String,String}()
        ),
        # Test stage
        Dict{String,Any}(
            "enabled" => false,
            "test_dir" => "test",
            "run_tests" => false
        ),
        # LLVM settings
        Dict{String,Any}(
            "use_replibuild_llvm" => true,
            "isolated" => true
        ),
        # Target settings
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

    save_config(config)
    println("✅ Created default configuration: $config_file")

    return config
end

"""
    update_discovery_data(config::RepliBuildConfig, discovery_results::Dict)

Update discovery stage with scan results.
All discovered files, dependencies, and AST data flow here.
Merges intelligently - appends arrays, updates values, preserves existing data.
"""
function update_discovery_data(config::RepliBuildConfig, discovery_results::Dict)
    merge_config_section!(config.discovery, discovery_results)
    config.last_modified = now()
end

"""
    update_compile_data(config::RepliBuildConfig, compile_results::Dict)

Update compile stage with IR generation results.
Merges intelligently - appends arrays, updates values, preserves existing data.
"""
function update_compile_data(config::RepliBuildConfig, compile_results::Dict)
    merge_config_section!(config.compile, compile_results)
    config.last_modified = now()
end

"""
    update_link_data(config::RepliBuildConfig, link_results::Dict)

Update link stage with optimization and linking results.
Merges intelligently - appends arrays, updates values, preserves existing data.
"""
function update_link_data(config::RepliBuildConfig, link_results::Dict)
    merge_config_section!(config.link, link_results)
    config.last_modified = now()
end

"""
    update_binary_data(config::RepliBuildConfig, binary_results::Dict)

Update binary stage with shared library creation results.
Merges intelligently - appends arrays, updates values, preserves existing data.
"""
function update_binary_data(config::RepliBuildConfig, binary_results::Dict)
    merge_config_section!(config.binary, binary_results)
    config.last_modified = now()
end

"""
    update_symbols_data(config::RepliBuildConfig, symbol_results::Dict)

Update symbols stage with extracted symbols and metadata.
Merges intelligently - appends arrays, updates values, preserves existing data.
"""
function update_symbols_data(config::RepliBuildConfig, symbol_results::Dict)
    merge_config_section!(config.symbols, symbol_results)
    config.last_modified = now()
end

"""
    update_wrap_data(config::RepliBuildConfig, wrap_results::Dict)

Update wrap stage with Julia binding generation results.
Merges intelligently - appends arrays, updates values, preserves existing data.
"""
function update_wrap_data(config::RepliBuildConfig, wrap_results::Dict)
    merge_config_section!(config.wrap, wrap_results)
    config.last_modified = now()
end

"""
    get_stage_config(config::RepliBuildConfig, stage::Symbol) -> Dict

Get configuration for a specific build stage.
"""
function get_stage_config(config::RepliBuildConfig, stage::Symbol)
    if stage == :discovery
        return config.discovery
    elseif stage == :reorganize
        return config.reorganize
    elseif stage == :compile
        return config.compile
    elseif stage == :link
        return config.link
    elseif stage == :binary
        return config.binary
    elseif stage == :symbols
        return config.symbols
    elseif stage == :wrap
        return config.wrap
    elseif stage == :test
        return config.test
    else
        error("Unknown stage: $stage")
    end
end

"""
    is_stage_enabled(config::RepliBuildConfig, stage::Symbol) -> Bool

Check if a build stage is enabled.
"""
function is_stage_enabled(config::RepliBuildConfig, stage::Symbol)
    stage_config = get_stage_config(config, stage)
    return get(stage_config, "enabled", false)
end

"""
    get_include_dirs(config::RepliBuildConfig) -> Vector{String}

Get include directories from discovery/compile stage.
Centralized accessor to fix the include_dirs confusion.
"""
function get_include_dirs(config::RepliBuildConfig)
    # Priority: discovery results > compile config
    discovery_includes = get(config.discovery, "include_dirs", String[])
    if !isempty(discovery_includes)
        return discovery_includes
    end

    compile_includes = get(config.compile, "include_dirs", String[])
    return compile_includes
end

"""
    set_include_dirs(config::RepliBuildConfig, include_dirs::Vector{String})

Set include directories (stored in discovery stage).
"""
function set_include_dirs(config::RepliBuildConfig, include_dirs::Vector{String})
    config.discovery["include_dirs"] = include_dirs
    config.last_modified = now()
end

"""
    get_source_files(config::RepliBuildConfig) -> Dict{String,Vector{String}}

Get categorized source files from discovery stage.
"""
function get_source_files(config::RepliBuildConfig)
    return get(config.discovery, "files", Dict{String,Vector{String}}())
end

"""
    set_source_files(config::RepliBuildConfig, files::Dict{String,Vector{String}})

Set categorized source files in discovery stage.
"""
function set_source_files(config::RepliBuildConfig, files::Dict{String,Vector{String}})
    config.discovery["files"] = files
    config.last_modified = now()
end

"""
    get_dependency_graph(config::RepliBuildConfig) -> Dict

Get dependency graph from discovery stage.
"""
function get_dependency_graph(config::RepliBuildConfig)
    return get(config.discovery, "dependency_graph", Dict())
end

"""
    set_dependency_graph(config::RepliBuildConfig, graph::Dict)

Set dependency graph in discovery stage.
"""
function set_dependency_graph(config::RepliBuildConfig, graph::Dict)
    config.discovery["dependency_graph"] = graph
    config.last_modified = now()
end

"""
    is_replibuild_project(path::String) -> Bool

Check if directory is a RepliBuild project (has replibuild.toml).
"""
function is_replibuild_project(path::String)
    return isfile(joinpath(path, "replibuild.toml"))
end

"""
    get_project_uuid(path::String) -> Union{UUID,Nothing}

Get UUID from replibuild.toml if it exists.
"""
function get_project_uuid(path::String)
    config_path = joinpath(path, "replibuild.toml")
    if !isfile(config_path)
        return nothing
    end

    try
        data = TOML.parsefile(config_path)
        project = get(data, "project", Dict())
        if haskey(project, "uuid")
            return UUID(project["uuid"])
        end
    catch
    end

    return nothing
end

"""
    find_replibuild_projects(search_path::String=pwd(); max_depth::Int=3) -> Vector{Tuple{String,UUID,String}}

Recursively find all RepliBuild projects under search_path.
Returns vector of (path, uuid, project_name) tuples.
"""
function find_replibuild_projects(search_path::String=pwd(); max_depth::Int=3)
    projects = Tuple{String,UUID,String}[]

    function scan_dir(path::String, depth::Int)
        if depth > max_depth
            return
        end

        # Check if current directory is a RepliBuild project
        if is_replibuild_project(path)
            uuid = get_project_uuid(path)
            config = load_config(joinpath(path, "replibuild.toml"))
            if !isnothing(uuid)
                push!(projects, (path, uuid, config.project_name))
            end
        end

        # Scan subdirectories
        try
            for entry in readdir(path)
                subpath = joinpath(path, entry)
                if isdir(subpath) && !startswith(entry, ".") && entry != "build"
                    scan_dir(subpath, depth + 1)
                end
            end
        catch
            # Permission denied or other error - skip
        end
    end

    scan_dir(search_path, 0)
    return projects
end

"""
    print_config_summary(config::RepliBuildConfig)

Print summary of configuration.
"""
function print_config_summary(config::RepliBuildConfig)
    println("="^70)
    println("RepliBuild Configuration Summary")
    println("="^70)
    println()
    println("📦 Project: $(config.project_name)")
    println("🔑 UUID:    $(config.project_uuid)")
    println("📁 Root:    $(config.project_root)")
    println("📄 Config:  $(config.config_file)")
    println("🕐 Updated: $(config.last_modified)")
    println()
    println("🔧 Build Stages:")

    for stage in BUILD_STAGES
        enabled = is_stage_enabled(config, stage)
        status = enabled ? "✅" : "⬜"
        println("   $status $(stage)")
    end

    println()
    println("📊 Discovery Results:")
    files = get_source_files(config)
    if !isempty(files)
        for (file_type, file_list) in files
            println("   $(file_type): $(length(file_list)) files")
        end
    else
        println("   No files discovered yet")
    end

    println()
    println("📂 Include Directories:")
    includes = get_include_dirs(config)
    if !isempty(includes)
        for dir in includes
            println("   • $dir")
        end
    else
        println("   None specified")
    end

    println("="^70)
end

"""
    validate_config(config::RepliBuildConfig) -> Vector{String}

Validate configuration and return a list of errors (empty if valid).
"""
function validate_config(config::RepliBuildConfig)
    errors = String[]

    # Check required fields
    if isempty(config.project_name)
        push!(errors, "project.name cannot be empty")
    end

    if !isdir(config.project_root)
        push!(errors, "project.root must be a valid directory: $(config.project_root)")
    end

    # Validate paths exist and are within project
    for stage in [:compile, :link, :binary, :wrap]
        stage_config = getproperty(config, stage)
        if haskey(stage_config, "output_dir")
            outdir = stage_config["output_dir"]
            if !isempty(outdir) && !startswith(abspath(outdir), abspath(config.project_root))
                # Relative paths are ok
                if !startswith(outdir, ".")
                    push!(errors, "$stage.output_dir outside project root: $outdir")
                end
            end
        end
    end

    # Validate compiler flags are reasonable
    if haskey(config.compile, "flags")
        flags = config.compile["flags"]
        if !isa(flags, Vector)
            push!(errors, "compile.flags must be a vector of strings")
        else
            for flag in flags
                if !isa(flag, String)
                    push!(errors, "compile.flags must contain only strings, got: $(typeof(flag))")
                elseif startswith(flag, "-f") && contains(flag, "/")
                    # Suspicious flag with path
                    @warn "Suspicious compiler flag (contains path): $flag"
                end
            end
        end
    end

    # Validate include directories exist
    if haskey(config.compile, "include_dirs")
        for inc_dir in config.compile["include_dirs"]
            if !isempty(inc_dir) && !startswith(inc_dir, ".")
                abs_inc = isabspath(inc_dir) ? inc_dir : joinpath(config.project_root, inc_dir)
                if !isdir(abs_inc)
                    push!(errors, "Include directory does not exist: $inc_dir")
                end
            end
        end
    end

    # Validate source files exist
    if haskey(config.compile, "sources")
        for src in config.compile["sources"]
            if !isempty(src)
                abs_src = isabspath(src) ? src : joinpath(config.project_root, src)
                if !isfile(abs_src)
                    push!(errors, "Source file does not exist: $src")
                end
            end
        end
    end

    return errors
end

"""
    validate_and_fix!(config::RepliBuildConfig)

Validate configuration and attempt to fix common issues.
Throws an error if validation fails and cannot be fixed.
"""
function validate_and_fix!(config::RepliBuildConfig)
    errors = validate_config(config)

    if isempty(errors)
        return true
    end

    # Try to fix some common issues
    fixed = String[]

    # Ensure output directories are relative
    for stage in [:compile, :link, :binary, :wrap]
        stage_config = getproperty(config, stage)
        if haskey(stage_config, "output_dir")
            outdir = stage_config["output_dir"]
            if !isempty(outdir) && !startswith(outdir, ".") && !startswith(abspath(outdir), abspath(config.project_root))
                # Make it relative to build/
                stage_config["output_dir"] = joinpath("build", String(stage))
                push!(fixed, "Fixed $stage.output_dir to be within project")
            end
        end
    end

    # Re-validate after fixes
    errors = validate_config(config)

    if !isempty(errors)
        # Build comprehensive error message
        error_details = "Found $(length(errors)) validation error(s):\n" * join(errors, "\n")
        if !isempty(fixed)
            error_details *= "\n\nFixed issues:\n" * join(fixed, "\n")
        end

        # Throw helpful error with solutions
        throw(UXHelpers.HelpfulError(
            "Configuration Validation Failed",
            error_details,
            [
                "Review and fix the errors listed above in your replibuild.toml",
                "Run ConfigurationManager.validate_config(config) to see specific issues",
                "Use ConfigurationManager.create_default_config() to start fresh",
                "Check the documentation for correct configuration format"
            ],
            docs_link="https://github.com/user/RepliBuild.jl#configuration"
        ))
    end

    if !isempty(fixed)
        @info "Configuration fixes applied:\n" * join(fixed, "\n")
    end

    return true
end

# Exports
export RepliBuildConfig, BUILD_STAGES,
       load_config, save_config, create_default_config,
       update_discovery_data, update_compile_data, update_link_data,
       update_binary_data, update_symbols_data, update_wrap_data,
       get_stage_config, is_stage_enabled,
       get_include_dirs, set_include_dirs,
       get_source_files, set_source_files,
       get_dependency_graph, set_dependency_graph,
       print_config_summary,
       is_replibuild_project, get_project_uuid, find_replibuild_projects,
       merge_config_section!,
       validate_config, validate_and_fix!

end # module ConfigurationManager
