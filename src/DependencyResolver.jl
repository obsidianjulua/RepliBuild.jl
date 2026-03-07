#!/usr/bin/env julia
# DependencyResolver.jl - Fetches and injects dependencies into the build config

module DependencyResolver

using ..ConfigurationManager
using ..BuildBridge

export resolve_dependencies

"""
Resolve dependencies declared in the configuration.
Fetches git repos, queries pkg-config, and merges the resulting 
source files, include directories, and linker flags into a new configuration.
"""
function resolve_dependencies(config::RepliBuildConfig)::RepliBuildConfig
    if isempty(config.dependencies.items)
        return config
    end

    extra_includes = String[]
    extra_sources = String[]
    extra_link_dirs = String[]
    extra_link_libs = String[]

    cache_dir = ConfigurationManager.get_cache_path(config)
    deps_cache = joinpath(cache_dir, "deps")
    mkpath(deps_cache)

    for (name, dep) in config.dependencies.items
        println("  dependency: resolving $name ($(dep.type))")
        if dep.type == "git"
            dep_path = joinpath(deps_cache, name)
            if !isdir(dep_path)
                println("    cloning $(dep.url)...")
                try
                    run(`git clone --quiet $(dep.url) $dep_path`)
                    if !isempty(dep.tag)
                        cd(dep_path) do
                            run(`git checkout --quiet $(dep.tag)`)
                        end
                    end
                catch e
                    @warn "Failed to clone dependency $name" exception=e
                    continue
                end
            end
            
            # Simple heuristic: add dep_path and dep_path/include to includes
            push!(extra_includes, dep_path)
            if isdir(joinpath(dep_path, "include"))
                push!(extra_includes, joinpath(dep_path, "include"))
            end
            
            # Find all .cpp/.c files in the dep_path (ignoring tests/examples typically)
            for (root, dirs, files) in walkdir(dep_path)
                filter!(d -> !in(d, ["test", "tests", "testes", "example", "examples", "fuzzing", "build", ".git", "doc", "docs"]), dirs)
                for file in files
                    rel_path = replace(relpath(joinpath(root, file), dep_path), "\\" => "/")
                    should_exclude = false
                    for ex in dep.exclude
                        if file == ex || startswith(rel_path, ex) || endswith(rel_path, ex) || occursin(ex, rel_path)
                            should_exclude = true
                            break
                        end
                    end
                    if should_exclude
                        continue
                    end
                    if endswith(file, ".cpp") || endswith(file, ".cc") || endswith(file, ".cxx") || endswith(file, ".c")
                        push!(extra_sources, joinpath(root, file))
                    end
                end
            end
            
        elseif dep.type == "system"
            if isempty(dep.pkg_config)
                @warn "System dependency $name missing pkg_config name"
                continue
            end
            
            # Get cflags
            (cflags_out, exit1) = BuildBridge.execute("pkg-config", ["--cflags", dep.pkg_config])
            if exit1 == 0
                for flag in split(cflags_out)
                    if startswith(flag, "-I")
                        push!(extra_includes, flag[3:end])
                    end
                end
            end
            
            # Get libs
            (libs_out, exit2) = BuildBridge.execute("pkg-config", ["--libs", dep.pkg_config])
            if exit2 == 0
                for flag in split(libs_out)
                    if startswith(flag, "-L")
                        push!(extra_link_dirs, flag[3:end])
                    elseif startswith(flag, "-l")
                        push!(extra_link_libs, flag[3:end])
                    end
                end
            end
            
        elseif dep.type == "local"
            if isempty(dep.path)
                @warn "Local dependency $name missing path"
                continue
            end
            dep_path = abspath(joinpath(config.project.root, dep.path))
            
            if !isdir(dep_path)
                @warn "Local dependency $name path does not exist: $dep_path"
                continue
            end
            
            push!(extra_includes, dep_path)
            if isdir(joinpath(dep_path, "include"))
                push!(extra_includes, joinpath(dep_path, "include"))
            end
            
            # Find all .cpp files in the local dep
            for (root, dirs, files) in walkdir(dep_path)
                filter!(d -> !in(d, ["build", ".git", ".cache"]), dirs)
                for file in files
                    if endswith(file, ".cpp") || endswith(file, ".cc") || endswith(file, ".cxx")
                        push!(extra_sources, joinpath(root, file))
                    elseif endswith(file, ".a") || endswith(file, ".so") || endswith(file, ".dylib")
                        push!(extra_link_dirs, root)
                        libname = splitext(basename(file))[1]
                        if startswith(libname, "lib")
                            push!(extra_link_libs, libname[4:end])
                        else
                            push!(extra_link_libs, libname)
                        end
                    end
                end
            end
        else
            @warn "Unknown dependency type: $(dep.type) for $name"
        end
    end

    # Merge into a new CompileConfig and LinkConfig
    # Use unique to prevent duplicates
    new_compile = ConfigurationManager.CompileConfig(
        unique(vcat(config.compile.source_files, extra_sources)),
        unique(vcat(config.compile.include_dirs, extra_includes)),
        config.compile.flags,
        config.compile.defines,
        config.compile.parallel,
        config.compile.aot_thunks
    )
    
    new_link = ConfigurationManager.LinkConfig(
        config.link.optimization_level,
        config.link.enable_lto,
        unique(vcat(config.link.link_libraries, extra_link_libs)),
        unique(vcat(config.link.link_dirs, extra_link_dirs))
    )

    return ConfigurationManager.RepliBuildConfig(
        config.project, config.paths, config.discovery,
        new_compile, new_link, config.binary, config.wrap,
        config.llvm, config.workflow, config.cache, config.dependencies, config.types,
        config.config_file, config.loaded_at
    )
end

end # module