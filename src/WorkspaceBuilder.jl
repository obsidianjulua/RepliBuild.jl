#!/usr/bin/env julia
# WorkspaceBuilder.jl - Multi-library parallel build orchestration
# Handles CMake-style multi-library projects with dependency-aware parallel compilation

module WorkspaceBuilder

using TOML
using Distributed

# Import necessary items from parent module namespace
# Bridge_LLVM.jl functions are defined at RepliBuild module level
import ..BridgeCompilerConfig
import ..compile_project

export build_workspace, discover_workspace

"""
    LibraryTarget

Represents a single library target in the workspace
"""
mutable struct LibraryTarget
    name::String
    path::String
    config_file::String
    link_dependencies::Vector{String}  # Other libraries this depends on
    type::Symbol  # :library or :executable
    built::Bool
    output_file::String
end

"""
    WorkspaceGraph

Represents the entire multi-library project structure
"""
struct WorkspaceGraph
    root_dir::String
    targets::Dict{String,LibraryTarget}
    build_order::Vector{Vector{String}}  # Levels of parallel builds
end

"""
    discover_workspace(root_dir::String=".") -> WorkspaceGraph

Discover all library targets in a multi-library workspace.

# Process:
1. Find all replibuild.toml files in subdirectories
2. Parse each one to extract target name and dependencies
3. Build dependency graph
4. Compute parallel build order (topological sort by levels)

# Returns
WorkspaceGraph containing all targets and their build order
"""
function discover_workspace(root_dir::String=".")
    println("🔍 Discovering workspace structure...")
    println("   Root: $root_dir")

    root_dir = abspath(root_dir)
    targets = Dict{String,LibraryTarget}()

    # Find all replibuild.toml files
    toml_files = String[]

    # Check subdirectories for replibuild.toml
    for entry in readdir(root_dir)
        entry_path = joinpath(root_dir, entry)
        if isdir(entry_path)
            toml_path = joinpath(entry_path, "replibuild.toml")
            if isfile(toml_path)
                push!(toml_files, toml_path)
            end
        end
    end

    println("   Found $(length(toml_files)) library projects")

    # Parse each TOML to extract target info
    for toml_file in toml_files
        target = parse_library_target(toml_file)
        if !isnothing(target)
            targets[target.name] = target
            println("      ✓ $(target.name) ($(target.type))")
        end
    end

    # Compute build order using topological sort
    build_order = compute_build_order(targets)

    println("   Build order: $(length(build_order)) levels")
    for (level, names) in enumerate(build_order)
        if isempty(names)
            continue
        end
        println("      Level $level: $(join(names, ", ")) (parallel)")
    end

    return WorkspaceGraph(root_dir, targets, build_order)
end

"""
Parse a single library target from its replibuild.toml
"""
function parse_library_target(toml_file::String)
    data = TOML.parsefile(toml_file)

    project = get(data, "project", Dict())
    compile = get(data, "compile", Dict())
    workflow = get(data, "workflow", Dict())

    name = get(project, "name", "")
    if isempty(name)
        @warn "No project name in $toml_file"
        return nothing
    end

    path = dirname(toml_file)
    link_deps = get(compile, "link_libraries", String[])

    # Filter out system libraries (pthread, dl, rt, m, etc.)
    system_libs = ["pthread", "dl", "rt", "m", "c", "\${CMAKE_THREAD_LIBS_INIT}", "\${CMAKE_DL_LIBS}"]
    link_deps = filter(dep -> !(dep in system_libs), link_deps)

    # Determine if library or executable
    stages = get(workflow, "stages", String[])
    target_type = "create_executable" in stages ? :executable : :library

    return LibraryTarget(
        name,
        path,
        toml_file,
        link_deps,
        target_type,
        false,  # not built yet
        ""      # output file TBD
    )
end

"""
Compute parallel build order using topological sort by levels

Returns vector of vectors, where each inner vector contains library names
that can be built in parallel.
"""
function compute_build_order(targets::Dict{String,LibraryTarget})
    # Build dependency graph
    deps = Dict{String,Set{String}}()
    for (name, target) in targets
        deps[name] = Set(target.link_dependencies)
    end

    # Topological sort by levels (Kahn's algorithm variant)
    levels = Vector{Vector{String}}()
    remaining = Set(keys(targets))

    while !isempty(remaining)
        # Find all targets with no unbuilt dependencies
        level = String[]
        for name in remaining
            target_deps = get(deps, name, Set{String}())
            # Check if all dependencies are either built or not in our workspace
            unmet_deps = intersect(target_deps, remaining)
            if isempty(unmet_deps)
                push!(level, name)
            end
        end

        if isempty(level)
            # Circular dependency detected
            @warn "Circular dependency detected in workspace"
            @warn "Remaining targets: $(collect(remaining))"
            # Add all remaining to final level to avoid infinite loop
            push!(levels, collect(remaining))
            break
        end

        push!(levels, level)
        setdiff!(remaining, level)
    end

    return levels
end

"""
    build_workspace(root_dir::String="."; parallel::Bool=true) -> Dict

Build entire multi-library workspace with dependency-aware parallel compilation.

# Arguments
- `root_dir`: Root directory containing the workspace
- `parallel`: Enable parallel compilation of independent libraries (default: true)

# Process
1. Discover all library targets
2. Build libraries in dependency order
3. Use @spawn to build independent libraries in parallel
4. Collect all built libraries for final executable linking

# Returns
Dict with:
- `:libraries` - Dict mapping library names to .so file paths
- `:executables` - Vector of built executable paths
- `:build_order` - The computed build order levels
"""
function build_workspace(root_dir::String="."; parallel::Bool=true)
    println("🏗️  RepliBuild Workspace Build")
    println("="^70)

    # Discover workspace structure
    workspace = discover_workspace(root_dir)

    if isempty(workspace.targets)
        @warn "No library targets found in workspace"
        return Dict(:libraries => Dict(), :executables => String[], :build_order => [])
    end

    println()
    println("🔨 Building $(length(workspace.targets)) targets...")
    println()

    built_libraries = Dict{String,String}()
    built_executables = String[]

    # Build each level
    for (level_num, level_targets) in enumerate(workspace.build_order)
        if isempty(level_targets)
            continue
        end

        println("📦 Level $level_num: Building $(length(level_targets)) targets")
        println("   Targets: $(join(level_targets, ", "))")

        if parallel && length(level_targets) > 1
            # Build in parallel using tasks
            println("   ⚡ Parallel compilation enabled")

            tasks = Task[]
            for target_name in level_targets
                target = workspace.targets[target_name]
                task = @task build_single_target(target, built_libraries)
                push!(tasks, task)
                schedule(task)
            end

            # Wait for all tasks to complete and collect results
            for (i, task) in enumerate(tasks)
                result = fetch(task)
                target_name = level_targets[i]
                target = workspace.targets[target_name]

                if !isnothing(result)
                    if target.type == :library
                        built_libraries[target_name] = result
                        println("      ✅ $target_name → $result")
                    else
                        push!(built_executables, result)
                        println("      ✅ $target_name (executable) → $result")
                    end
                    target.built = true
                    target.output_file = result
                else
                    @warn "Failed to build $target_name"
                end
            end
        else
            # Build sequentially
            for target_name in level_targets
                target = workspace.targets[target_name]
                result = build_single_target(target, built_libraries)

                if !isnothing(result)
                    if target.type == :library
                        built_libraries[target_name] = result
                        println("      ✅ $target_name → $result")
                    else
                        push!(built_executables, result)
                        println("      ✅ $target_name (executable) → $result")
                    end
                    target.built = true
                    target.output_file = result
                else
                    @warn "Failed to build $target_name"
                end
            end
        end

        println()
    end

    println("="^70)
    println("✅ Workspace build complete!")
    println("   Libraries: $(length(built_libraries))")
    println("   Executables: $(length(built_executables))")

    return Dict(
        :libraries => built_libraries,
        :executables => built_executables,
        :build_order => workspace.build_order
    )
end

"""
Build a single library target

Returns the path to the built library (.so file) or nothing on failure
"""
function build_single_target(target::LibraryTarget, built_libraries::Dict{String,String})
    println("         🔨 Building $(target.name)...")

    # Change to target directory
    original_dir = pwd()

    try
        cd(target.path)

        # Build using Bridge_LLVM compile
        config = BridgeCompilerConfig("replibuild.toml")

        # Add library paths for dependencies
        if !isempty(target.link_dependencies)
            lib_dirs = String[]
            for dep_name in target.link_dependencies
                if haskey(built_libraries, dep_name)
                    lib_path = dirname(built_libraries[dep_name])
                    if !in(lib_path, lib_dirs)
                        push!(lib_dirs, lib_path)
                    end
                end
            end
        end

        # Compile the target
        result = compile_project(config)

        # Find the output file
        if target.type == :library
            # Look for .so file in build or julia directory
            output_dirs = ["julia", "build", "."]
            for dir in output_dirs
                full_dir = joinpath(target.path, dir)
                if isdir(full_dir)
                    for file in readdir(full_dir)
                        if endswith(file, ".so") || endswith(file, ".dylib") || endswith(file, ".dll")
                            return joinpath(full_dir, file)
                        end
                    end
                end
            end
        else
            # Look for executable
            output_dirs = ["julia", "build", "."]
            for dir in output_dirs
                full_dir = joinpath(target.path, dir)
                if isdir(full_dir)
                    for file in readdir(full_dir)
                        # Executables usually don't have extensions on Unix
                        if !endswith(file, ".so") && !endswith(file, ".o") && !endswith(file, ".ll")
                            exe_path = joinpath(full_dir, file)
                            if isfile(exe_path)
                                # Check if executable
                                try
                                    stat_result = stat(exe_path)
                                    if stat_result.mode & 0o111 != 0  # Has execute permission
                                        return exe_path
                                    end
                                catch
                                end
                            end
                        end
                    end
                end
            end
        end

        @warn "Could not find output file for $(target.name)"
        return nothing

    catch e
        @warn "Error building $(target.name): $e"
        return nothing
    finally
        cd(original_dir)
    end
end

end # module WorkspaceBuilder
