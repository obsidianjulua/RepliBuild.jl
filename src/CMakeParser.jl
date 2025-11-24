#!/usr/bin/env julia
# CMakeParser.jl - Parse CMakeLists.txt without running CMake
# Extract build configuration data and normalize it for RepliBuild

module CMakeParser

using TOML

# Note: ModuleRegistry was removed - external library resolution not yet implemented

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
Represents a CMake target (library or executable)
"""
mutable struct CMakeTarget
    name::String
    type::Symbol  # :executable, :static_library, :shared_library, :interface_library
    sources::Vector{String}
    include_dirs::Vector{String}
    compile_options::Vector{String}
    compile_definitions::Dict{String,String}
    link_libraries::Vector{String}
    properties::Dict{String,Any}
end

"""
Represents a parsed CMake project
"""
mutable struct CMakeProject
    project_name::String
    cmake_minimum_required::String
    root_dir::String
    targets::Dict{String,CMakeTarget}
    variables::Dict{String,String}
    subdirectories::Vector{String}
    find_packages::Vector{String}
end

# ============================================================================
# LEXER - Tokenize CMake commands
# ============================================================================

"""
Tokenize a CMake command line
Example: add_library(mylib SHARED foo.cpp bar.cpp)
  → ["add_library", "(", "mylib", "SHARED", "foo.cpp", "bar.cpp", ")"]
"""
function tokenize_line(line::AbstractString)
    tokens = String[]
    current = ""
    in_quotes = false
    in_parens = false

    for char in line
        if char == '"'
            in_quotes = !in_quotes
            if !in_quotes && !isempty(current)
                push!(tokens, current)
                current = ""
            end
        elseif char == '(' && !in_quotes
            if !isempty(current)
                push!(tokens, current)
                current = ""
            end
            push!(tokens, "(")
            in_parens = true
        elseif char == ')' && !in_quotes
            if !isempty(current)
                push!(tokens, current)
                current = ""
            end
            push!(tokens, ")")
            in_parens = false
        elseif isspace(char) && !in_quotes
            if !isempty(current)
                push!(tokens, current)
                current = ""
            end
        else
            current *= char
        end
    end

    if !isempty(current)
        push!(tokens, current)
    end

    return tokens
end

"""
Parse CMake command from tokens
Returns (command_name, args)
"""
function parse_command(tokens::Vector{String})
    if isempty(tokens)
        return ("", String[])
    end

    command = lowercase(tokens[1])
    args = String[]

    # Find opening paren
    paren_idx = findfirst(==("("), tokens)
    if paren_idx === nothing
        return (command, args)
    end

    # Extract arguments between parens
    close_paren = findlast(==(")"), tokens)
    if close_paren === nothing
        close_paren = length(tokens)
    end

    args = tokens[paren_idx+1:close_paren-1]

    return (command, args)
end

# ============================================================================
# PARSER - Extract CMake configuration
# ============================================================================

"""
Parse a CMakeLists.txt file with multi-line command support
"""
function parse_cmake_file(filepath::String)
    if !isfile(filepath)
        error("CMakeLists.txt not found: $filepath")
    end

    # Get absolute path of the CMakeLists.txt directory
    root_dir = dirname(abspath(filepath))
    project = CMakeProject(
        "",  # project_name
        "",  # cmake_minimum_required
        root_dir,
        Dict{String,CMakeTarget}(),
        Dict{String,String}(),
        String[],
        String[]
    )

    lines = readlines(filepath)

    # Preprocess: merge multi-line commands
    merged_lines = merge_multiline_commands(lines)

    # Process each command
    for line in merged_lines
        # Remove comments (but preserve # inside strings)
        line = remove_comments(line)
        line = strip(String(line))

        # Skip empty lines
        if isempty(line)
            continue
        end

        # Tokenize and parse
        tokens = tokenize_line(line)
        if isempty(tokens)
            continue
        end

        (command, args) = parse_command(tokens)

        # Process command
        process_cmake_command!(project, command, args, root_dir)
    end

    return project
end

"""
Merge multi-line CMake commands into single lines
Handles commands that span multiple lines with opening ( and closing )
"""
function merge_multiline_commands(lines::Vector{String})
    merged = String[]
    current_command = ""
    paren_depth = 0
    in_command = false

    for line in lines
        # Check for line continuation with backslash
        has_continuation = endswith(strip(line), "\\")
        if has_continuation
            line = line[1:end-1]  # Remove trailing backslash
        end

        # Count parentheses (but ignore those in comments and strings)
        line_cleaned = remove_comments(line)

        for char in line_cleaned
            if char == '('
                paren_depth += 1
                in_command = true
            elseif char == ')'
                paren_depth -= 1
            end
        end

        # Accumulate the line
        if in_command || has_continuation
            current_command *= " " * strip(line)
        else
            if !isempty(strip(line))
                push!(merged, strip(line))
            end
        end

        # If parentheses are balanced and no continuation, complete the command
        if paren_depth == 0 && !has_continuation && in_command
            push!(merged, strip(current_command))
            current_command = ""
            in_command = false
        end
    end

    # Add any remaining command
    if !isempty(strip(current_command))
        push!(merged, strip(current_command))
    end

    return merged
end

"""
Remove comments from a line, but preserve # inside strings
"""
function remove_comments(line::String)
    result = ""
    in_string = false
    escape_next = false

    for (i, char) in enumerate(line)
        if escape_next
            result *= char
            escape_next = false
            continue
        end

        if char == '\\'
            escape_next = true
            result *= char
            continue
        end

        if char == '"'
            in_string = !in_string
            result *= char
            continue
        end

        if char == '#' && !in_string
            # Rest of line is comment
            break
        end

        result *= char
    end

    return result
end

"""
Process a single CMake command
"""
function process_cmake_command!(project::CMakeProject, command::String, args::Vector{String}, root_dir::String)
    if command == "project"
        # project(MyProject)
        if !isempty(args)
            project.project_name = args[1]
        end

    elseif command == "cmake_minimum_required"
        # cmake_minimum_required(VERSION 3.10)
        version_idx = findfirst(==("VERSION"), args)
        if version_idx !== nothing && version_idx < length(args)
            project.cmake_minimum_required = args[version_idx + 1]
        end

    elseif command == "set"
        # set(VAR_NAME value)
        if length(args) >= 2
            var_name = args[1]
            var_value = join(args[2:end], " ")
            project.variables[var_name] = var_value
        end

    elseif command == "add_library"
        # add_library(mylib SHARED foo.cpp bar.cpp)
        parse_add_library!(project, args, root_dir)

    elseif command == "add_executable"
        # add_executable(myapp main.cpp)
        parse_add_executable!(project, args, root_dir)

    elseif command == "target_sources"
        # target_sources(mylib PRIVATE foo.cpp bar.cpp)
        parse_target_sources!(project, args, root_dir)

    elseif command == "target_include_directories"
        # target_include_directories(mylib PUBLIC include)
        parse_target_include_directories!(project, args, root_dir)

    elseif command == "target_compile_options"
        # target_compile_options(mylib PRIVATE -O2 -Wall)
        parse_target_compile_options!(project, args)

    elseif command == "target_compile_definitions"
        # target_compile_definitions(mylib PRIVATE DEBUG=1)
        parse_target_compile_definitions!(project, args)

    elseif command == "target_link_libraries"
        # target_link_libraries(mylib pthread m)
        parse_target_link_libraries!(project, args)

    elseif command == "add_subdirectory"
        # add_subdirectory(subdir)
        if !isempty(args)
            push!(project.subdirectories, args[1])
        end

    elseif command == "find_package"
        # find_package(OpenCV REQUIRED)
        if !isempty(args)
            push!(project.find_packages, args[1])
        end
    end
end

"""
Parse add_library command
"""
function parse_add_library!(project::CMakeProject, args::Vector{String}, root_dir::String)
    if isempty(args)
        return
    end

    target_name = args[1]
    type = :static_library  # default
    sources = String[]

    i = 2
    while i <= length(args)
        arg = args[i]

        if arg == "SHARED"
            type = :shared_library
        elseif arg == "STATIC"
            type = :static_library
        elseif arg == "INTERFACE"
            type = :interface_library
        elseif arg in ["PUBLIC", "PRIVATE", "INTERFACE"]
            # Skip visibility keywords
        else
            # It's a source file
            source_path = resolve_path(arg, root_dir)
            push!(sources, source_path)
        end

        i += 1
    end

    target = CMakeTarget(
        target_name,
        type,
        sources,
        String[],
        String[],
        Dict{String,String}(),
        String[],
        Dict{String,Any}()
    )

    project.targets[target_name] = target
end

"""
Parse add_executable command
"""
function parse_add_executable!(project::CMakeProject, args::Vector{String}, root_dir::String)
    if isempty(args)
        return
    end

    target_name = args[1]
    sources = String[]

    for i in 2:length(args)
        if !(args[i] in ["PUBLIC", "PRIVATE", "INTERFACE"])
            source_path = resolve_path(args[i], root_dir)
            push!(sources, source_path)
        end
    end

    target = CMakeTarget(
        target_name,
        :executable,
        sources,
        String[],
        String[],
        Dict{String,String}(),
        String[],
        Dict{String,Any}()
    )

    project.targets[target_name] = target
end

"""
Parse target_sources command
"""
function parse_target_sources!(project::CMakeProject, args::Vector{String}, root_dir::String)
    if length(args) < 2
        return
    end

    target_name = args[1]
    if !haskey(project.targets, target_name)
        return
    end

    target = project.targets[target_name]

    for i in 2:length(args)
        if !(args[i] in ["PUBLIC", "PRIVATE", "INTERFACE"])
            source_path = resolve_path(args[i], root_dir)
            push!(target.sources, source_path)
        end
    end
end

"""
Parse target_include_directories command
"""
function parse_target_include_directories!(project::CMakeProject, args::Vector{String}, root_dir::String)
    if length(args) < 2
        return
    end

    target_name = args[1]
    if !haskey(project.targets, target_name)
        return
    end

    target = project.targets[target_name]

    for i in 2:length(args)
        if !(args[i] in ["PUBLIC", "PRIVATE", "INTERFACE"])
            include_path = resolve_path(args[i], root_dir)
            push!(target.include_dirs, include_path)
        end
    end
end

"""
Parse target_compile_options command
"""
function parse_target_compile_options!(project::CMakeProject, args::Vector{String})
    if length(args) < 2
        return
    end

    target_name = args[1]
    if !haskey(project.targets, target_name)
        return
    end

    target = project.targets[target_name]

    for i in 2:length(args)
        if !(args[i] in ["PUBLIC", "PRIVATE", "INTERFACE"])
            push!(target.compile_options, args[i])
        end
    end
end

"""
Parse target_compile_definitions command
"""
function parse_target_compile_definitions!(project::CMakeProject, args::Vector{String})
    if length(args) < 2
        return
    end

    target_name = args[1]
    if !haskey(project.targets, target_name)
        return
    end

    target = project.targets[target_name]

    for i in 2:length(args)
        arg = args[i]
        if !(arg in ["PUBLIC", "PRIVATE", "INTERFACE"])
            # Parse definition (e.g., "DEBUG=1" or "FEATURE")
            parts = split(arg, '=')
            if length(parts) == 2
                target.compile_definitions[parts[1]] = parts[2]
            else
                target.compile_definitions[arg] = "1"
            end
        end
    end
end

"""
Parse target_link_libraries command
"""
function parse_target_link_libraries!(project::CMakeProject, args::Vector{String})
    if length(args) < 2
        return
    end

    target_name = args[1]
    if !haskey(project.targets, target_name)
        return
    end

    target = project.targets[target_name]

    for i in 2:length(args)
        if !(args[i] in ["PUBLIC", "PRIVATE", "INTERFACE"])
            push!(target.link_libraries, args[i])
        end
    end
end

# ============================================================================
# UTILITIES
# ============================================================================

"""
Resolve relative paths relative to CMakeLists.txt directory
"""
function resolve_path(path::String, root_dir::String)
    # Handle CMake variables (basic support)
    path = replace(path, "\${CMAKE_CURRENT_SOURCE_DIR}" => root_dir)
    path = replace(path, "\${PROJECT_SOURCE_DIR}" => root_dir)
    path = replace(path, "\$ENV{HOME}" => get(ENV, "HOME", ""))

    # If path is already absolute, return as-is
    if isabspath(path)
        return path
    end

    # If root_dir is empty or path starts with /, join carefully
    if isempty(root_dir)
        # Try to get absolute path from current directory
        if startswith(path, "/")
            return path
        else
            return abspath(path)
        end
    end

    # Make relative path absolute by joining with root_dir
    path = joinpath(root_dir, path)

    # Normalize the path (remove . and ..)
    return abspath(path)
end

"""
Substitute CMake variables in a string
"""
function substitute_variables(str::String, variables::Dict{String,String})
    result = str
    for (var, value) in variables
        result = replace(result, "\${$var}" => value)
        result = replace(result, "\$($var)" => value)
    end
    return result
end

# ============================================================================
# CONVERSION TO REPLIBUILD
# ============================================================================

"""
Convert CMakeProject to RepliBuild replibuild.toml configuration
"""
function to_replibuild_config(cmake_project::CMakeProject, target_name::String="")
    if isempty(target_name)
        # Use first target if not specified
        if isempty(cmake_project.targets)
            error("No targets found in CMake project")
        end
        target_name = first(keys(cmake_project.targets))
    end

    if !haskey(cmake_project.targets, target_name)
        error("Target '$target_name' not found in CMake project")
    end

    target = cmake_project.targets[target_name]

    # Build replibuild.toml structure
    config = Dict{String,Any}()

    # [project]
    config["project"] = Dict{String,Any}(
        "name" => target_name,
        "root" => cmake_project.root_dir
    )

    # [paths]
    # Determine source directory from source files
    source_dirs = unique([dirname(src) for src in target.sources])
    source_dir = isempty(source_dirs) ? "src" : source_dirs[1]

    config["paths"] = Dict{String,Any}(
        "source" => source_dir,
        "output" => "julia",
        "build" => "build"
    )

    # [compile]
    compile_config = Dict{String,Any}()

    # Always include the project root for subdirectory includes
    # (replicates CMake's include_directories(${PROJECT_SOURCE_DIR}))
    include_dirs = copy(target.include_dirs)

    # Find the actual project root (parent of all subdirs)
    # If we're in a subdirectory, add parent as include
    if occursin("/", cmake_project.root_dir)
        parent_dir = dirname(cmake_project.root_dir)
        # Check if parent has CMakeLists.txt (indicates it's the project root)
        if isfile(joinpath(parent_dir, "CMakeLists.txt"))
            if !in(parent_dir, include_dirs)
                pushfirst!(include_dirs, parent_dir)
            end
        end
    end

    if !isempty(include_dirs)
        compile_config["include_dirs"] = include_dirs
    end

    # Build compile flags
    flags = copy(target.compile_options)

    # Always add -fPIC for shared libraries (required for position-independent code)
    if target.type == :shared_library && !in("-fPIC", flags)
        push!(flags, "-fPIC")
    end

    # Filter out CMake-specific warnings
    flags = filter(opt -> !startswith(opt, "-W"), flags)

    if !isempty(flags)
        compile_config["flags"] = flags
    end

    if !isempty(target.compile_definitions)
        compile_config["defines"] = target.compile_definitions
    end

    # Add link libraries if present (important for executables)
    if !isempty(target.link_libraries)
        compile_config["link_libraries"] = target.link_libraries
    end

    config["compile"] = compile_config

    # [dependencies] - Resolve find_package() calls to Julia modules
    if !isempty(cmake_project.find_packages)
        dependencies = Dict{String,Any}()
        external_includes = String[]
        external_libs = String[]
        external_lib_dirs = String[]

        for pkg_name in cmake_project.find_packages
            println("🔍 External dependency detected: $pkg_name")

            # TODO: ModuleRegistry was removed - implement external library resolution
            # For now, just add as system library
            mod_info = nothing  # ModuleRegistry.resolve_module(pkg_name) - removed

            if !isnothing(mod_info)
                # Add to dependencies section
                dependencies[pkg_name] = Dict(
                    "source" => string(mod_info.source),
                    "version" => mod_info.version,
                    "julia_package" => mod_info.julia_package
                )

                # Merge paths into compile config
                append!(external_includes, mod_info.include_dirs)
                append!(external_lib_dirs, mod_info.library_dirs)
                append!(external_libs, mod_info.libraries)

                println("  ✓ Resolved: $(mod_info.source) ($(mod_info.julia_package))")
            else
                @warn "Could not resolve dependency: $pkg_name (will need manual configuration)"
                dependencies[pkg_name] = Dict(
                    "source" => "unresolved",
                    "note" => "Manual configuration required in replibuild.toml"
                )
            end
        end

        if !isempty(dependencies)
            config["dependencies"] = dependencies
        end

        # Merge external library paths into compile config
        if !isempty(external_includes)
            if haskey(compile_config, "include_dirs")
                append!(compile_config["include_dirs"], external_includes)
            else
                compile_config["include_dirs"] = external_includes
            end
        end

        if !isempty(external_lib_dirs)
            compile_config["library_dirs"] = external_lib_dirs
        end

        if !isempty(external_libs)
            if haskey(compile_config, "link_libraries")
                append!(compile_config["link_libraries"], external_libs)
            else
                compile_config["link_libraries"] = external_libs
            end
        end
    end

    # [target]
    config["target"] = Dict{String,Any}(
        "triple" => "",
        "cpu" => "generic",
        "opt_level" => "O2",
        "lto" => false
    )

    # [bridge]
    config["bridge"] = Dict{String,Any}(
        "auto_discover" => true,
        "enable_learning" => true,
        "cache_tools" => true
    )

    # [workflow]
    # Choose workflow stages based on target type
    if target.type == :executable
        stages = [
            "discover_tools",
            "compile_to_ir",
            "link_ir",
            "optimize_ir",
            "create_executable"
        ]
    else
        stages = [
            "discover_tools",
            "compile_to_ir",
            "link_ir",
            "optimize_ir",
            "create_library",
            "extract_symbols"
        ]
    end

    config["workflow"] = Dict{String,Any}(
        "stages" => stages,
        "parallel" => true
    )

    # [cache]
    config["cache"] = Dict{String,Any}(
        "enabled" => true,
        "directory" => ".bridge_cache"
    )

    return config
end

"""
Write replibuild.toml from CMakeProject
"""
function write_replibuild_config(cmake_project::CMakeProject, target_name::String, output_path::String="replibuild.toml")
    config = to_replibuild_config(cmake_project, target_name)

    open(output_path, "w") do io
        TOML.print(io, config)
    end

    println("✅ Generated replibuild.toml for target: $target_name")
    return output_path
end

# ============================================================================
# EXPORTS
# ============================================================================

"""
    cmake_replicate(root_dir::String=pwd(); dry_run::Bool=false) -> Dict{String,String}

Recursively replicate entire CMake build structure to RepliBuild.

# Process:
1. Parse root CMakeLists.txt
2. Find all add_subdirectory() calls
3. Recursively parse each subdirectory's CMakeLists.txt
4. Generate replibuild.toml in each directory with targets
5. Create dependency chain between subdirectories

# Arguments
- `root_dir`: Root directory containing top-level CMakeLists.txt
- `dry_run`: If true, don't write files, just return what would be created

# Returns
- Dictionary mapping directory paths to generated replibuild.toml content
"""
function cmake_replicate(root_dir::String=pwd(); dry_run::Bool=false)
    println("🔄 CMake → RepliBuild Replication")
    println("="^70)
    println("📁 Root: $root_dir")
    println()

    results = Dict{String,String}()

    # Parse recursively starting from root
    _cmake_replicate_recursive!(results, root_dir, root_dir, dry_run)

    println()
    println("="^70)
    println("✅ Replication complete!")
    println("📊 Generated $(length(results)) replibuild.toml files")

    return results
end

"""
Internal recursive helper for cmake_replicate
"""
function _cmake_replicate_recursive!(results::Dict{String,String}, root_dir::String, current_dir::String, dry_run::Bool)
    cmake_file = joinpath(current_dir, "CMakeLists.txt")

    if !isfile(cmake_file)
        return
    end

    println("📄 Parsing: $(relpath(cmake_file, root_dir))")

    # Parse this directory's CMakeLists.txt
    project = parse_cmake_file(cmake_file)

    # Track subdirectories to recurse into
    subdirs = copy(project.subdirectories)

    # Generate replibuild.toml for targets in this directory
    if !isempty(project.targets)
        for (target_name, target) in project.targets
            println("   └─ Target: $target_name ($(target.type))")

            # Generate replibuild.toml for this target
            toml_content = to_replibuild_config(project, target_name)
            toml_path = joinpath(current_dir, "replibuild.toml")

            if !dry_run
                open(toml_path, "w") do io
                    TOML.print(io, toml_content)
                end
                println("      ✓ Created: $(relpath(toml_path, root_dir))")
            end

            results[current_dir] = sprint(io -> TOML.print(io, toml_content))

            # Only process first target per directory (or combine them)
            break
        end
    end

    # Recurse into subdirectories
    for subdir in subdirs
        subdir_path = joinpath(current_dir, subdir)
        if isdir(subdir_path)
            _cmake_replicate_recursive!(results, root_dir, subdir_path, dry_run)
        end
    end
end

export
    # Data structures
    CMakeProject,
    CMakeTarget,

    # Parsing
    parse_cmake_file,

    # Conversion
    to_replibuild_config,

    # Recursive replication
    cmake_replicate,
    write_replibuild_config

end # module CMakeParser
