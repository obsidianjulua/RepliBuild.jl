#!/usr/bin/env julia
# DependencyResolver.jl - Fetches and injects dependencies into the build config

module DependencyResolver

using ..ConfigurationManager
using ..BuildBridge

export resolve_dependencies

# ---------------------------------------------------------------------------
# Git dependency cache markers (version-aware resolution)
#
# The clone at .replibuild_cache/deps/<name> is keyed on <name> alone, so a
# bare `isdir` check silently serves a stale checkout once the toml's tag or
# url changes. We record the resolved url+tag in a sidecar marker next to the
# clone and re-resolve when either drifts.
# ---------------------------------------------------------------------------

_dep_marker_path(deps_cache::AbstractString, name::AbstractString) =
    joinpath(deps_cache, name * ".resolved")

_tagdisp(tag::AbstractString) = isempty(tag) ? "default branch" : tag

function _write_dep_marker(path::AbstractString, url::AbstractString, tag::AbstractString)
    try
        open(path, "w") do io
            println(io, "url=", url)
            println(io, "tag=", tag)
        end
    catch e
        @warn "Could not write dependency cache marker" path exception=e
    end
    return nothing
end

function _read_dep_marker(path::AbstractString)
    url = ""; tag = ""
    for line in eachline(path)
        if startswith(line, "url=")
            url = String(line[5:end])
        elseif startswith(line, "tag=")
            tag = String(line[5:end])
        end
    end
    return (url, tag)
end

# Best-effort origin URL for a legacy clone that predates the marker.
function _git_origin_url(dep_path::AbstractString)
    try
        return String(strip(read(`git -C $dep_path remote get-url origin`, String)))
    catch
        return ""
    end
end

# What is currently cached at dep_path? Returns (url, tag, have_marker).
#   marker present            → authoritative url+tag
#   marker absent, dir present → infer url from git, tag unknown ("") ⇒ re-checkout
#   nothing cached            → ("", "", false)
function _read_dep_state(marker_path::AbstractString, dep_path::AbstractString)
    if isfile(marker_path)
        url, tag = _read_dep_marker(marker_path)
        return (url, tag, true)
    elseif isdir(dep_path)
        return (_git_origin_url(dep_path), "", false)
    else
        return ("", "", false)
    end
end

function _clone_dep(name::AbstractString, dep, dep_path::AbstractString, marker_path::AbstractString)::Bool
    println("    cloning $(dep.url)...")
    try
        # Use '--' to separate options from positional arguments
        run(`git clone --quiet -- $(dep.url) $dep_path`)
        if !isempty(dep.tag)
            cd(dep_path) do
                # Tag is validated by the caller (no '-' prefix), so safe to pass directly.
                # Do NOT use '--' here — it tells git to treat the arg as a file path.
                run(`git checkout --quiet $(dep.tag)`)
            end
        end
        _write_dep_marker(marker_path, dep.url, dep.tag)
        return true
    catch e
        @warn "Failed to clone dependency $name" exception=e
        # Never leave a half-clone behind — a partial dir would read as a valid cache.
        rm(dep_path; recursive=true, force=true)
        return false
    end
end

function _recheckout_dep(name::AbstractString, dep, dep_path::AbstractString, marker_path::AbstractString)::Bool
    try
        cd(dep_path) do
            # Fetch is best-effort: an offline switch to an already-local tag must still work.
            try
                run(`git fetch --quiet --tags origin`)
            catch e
                @warn "git fetch failed for $name; checking out against local refs" exception=e
            end
            if !isempty(dep.tag)
                run(`git checkout --quiet $(dep.tag)`)
            end
        end
        _write_dep_marker(marker_path, dep.url, dep.tag)
        return true
    catch e
        @warn "Failed to re-checkout dependency $name to '$(dep.tag)'" exception=e
        return false
    end
end

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

        # Validate dependency name — used as a directory name; reject traversal
        if contains(name, "..") || contains(name, "/") || contains(name, "\\") || startswith(name, "-")
            @warn "Rejecting dependency with unsafe name: $name"
            continue
        end

        if dep.type == "git"
            # Validate git URL and tag to prevent git option injection.
            # Git interprets arguments starting with '-' as options.
            if startswith(dep.url, "-")
                @warn "Rejecting git dependency $name: URL starts with '-' (potential option injection)"
                continue
            end
            if !isempty(dep.tag) && startswith(dep.tag, "-")
                @warn "Rejecting git dependency $name: tag starts with '-' (potential option injection)"
                continue
            end

            dep_path = joinpath(deps_cache, name)
            marker_path = _dep_marker_path(deps_cache, name)
            cached_url, cached_tag, have_marker = _read_dep_state(marker_path, dep_path)

            if !isdir(dep_path) || cached_url != dep.url
                # Absent, or the upstream URL changed → (re)clone from scratch.
                if isdir(dep_path)
                    println("    $name: url changed → re-cloning")
                    rm(dep_path; recursive=true, force=true)
                end
                _clone_dep(name, dep, dep_path, marker_path) || continue
            elseif cached_tag != dep.tag
                # Same repo, requested version drifted (or a legacy cache with no
                # recorded tag) → fetch + checkout the requested ref, else re-clone.
                if have_marker
                    println("    $name: $(_tagdisp(cached_tag)) → $(_tagdisp(dep.tag)), re-resolving")
                else
                    println("    $name: verifying cached checkout (no version marker)")
                end
                if !_recheckout_dep(name, dep, dep_path, marker_path)
                    rm(dep_path; recursive=true, force=true)
                    _clone_dep(name, dep, dep_path, marker_path) || continue
                end
            elseif !have_marker
                # Up to date but unmarked (legacy cache) → record the marker.
                _write_dep_marker(marker_path, dep.url, dep.tag)
            end
            
            # Heuristic: add common include directories from dep
            push!(extra_includes, dep_path)
            for subdir in ("include", "src")
                d = joinpath(dep_path, subdir)
                isdir(d) && push!(extra_includes, d)
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
        unique(vcat(config.link.link_dirs, extra_link_dirs)),
        config.link.fallback,
        config.link.promote_statics
    )

    return ConfigurationManager.RepliBuildConfig(
        config.project, config.paths, config.discovery,
        new_compile, new_link, config.binary, config.wrap,
        config.llvm, config.workflow, config.cache, config.dependencies, config.types,
        config.ingest, config.config_file, config.loaded_at
    )
end

end # module