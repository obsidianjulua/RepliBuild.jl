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
#
# CONTENT IDENTITY (2026-08-08). url+tag alone cannot answer "is this the same
# source as last time": a git tag is a mutable ref, so upstream — or whoever
# takes over the account — can repoint `v1.15` at different content and every
# check here would still match. Two layers close that:
#
#   1. The marker also records the RESOLVED commit (`git rev-parse HEAD`), so a
#      checkout that drifted out from under a cache hit is caught and re-resolved
#      loudly instead of silently compiling.
#   2. A recipe may declare `commit = "<40-hex>"` beside its tag. That is checked
#      after every clone/checkout and is a HARD ERROR on mismatch. It is the only
#      layer that survives `clean()` — which deletes the clone and the marker with
#      it — and therefore the only one that helps on the rebuild-from-cold path
#      that Hub `test.jl` takes.
#
# Neither layer is a substitute for reading the diff; they turn a silent content
# swap into a loud one.
#
# SUBMODULES (2026-10-30). `submodules = ["libs/math", …]` fetches exactly those
# paths of a superproject (boostorg/boost has ~170; it is never "all"). The pin
# covers them without a second key: the superproject's tree records each
# submodule's commit as a gitlink. The marker records the fetched list, so
# changing it re-resolves like a tag change, and a listed submodule whose
# checkout is not the commit its gitlink records is a hard error.
# ---------------------------------------------------------------------------

_dep_marker_path(deps_cache::AbstractString, name::AbstractString) =
    joinpath(deps_cache, name * ".resolved")

_tagdisp(tag::AbstractString) = isempty(tag) ? "default branch" : tag
_shortsha(sha::AbstractString) = length(sha) >= 12 ? sha[1:12] : (isempty(sha) ? "unknown" : sha)
_subsdisp(subs) = isempty(subs) ? "none" : join(subs, ", ")

# Submodule entries are validated at load to contain no ',' (ConfigurationManager).
function _write_dep_marker(path::AbstractString, url::AbstractString, tag::AbstractString,
                           commit::AbstractString="", submodules::AbstractVector=String[])
    try
        open(path, "w") do io
            println(io, "url=", url)
            println(io, "tag=", tag)
            println(io, "commit=", commit)
            println(io, "submodules=", join(submodules, ","))
        end
    catch e
        @warn "Could not write dependency cache marker" path exception=e
    end
    return nothing
end

# A marker written before submodules existed has no `submodules=` line, and its
# clone fetched none: an empty list is the truth for it.
function _read_dep_marker(path::AbstractString)
    url = ""; tag = ""; commit = ""; submodules = String[]
    for line in eachline(path)
        if startswith(line, "url=")
            url = String(line[5:end])
        elseif startswith(line, "tag=")
            tag = String(line[5:end])
        elseif startswith(line, "commit=")
            commit = String(line[8:end])
        elseif startswith(line, "submodules=")
            submodules = String[s for s in split(line[12:end], ',') if !isempty(s)]
        end
    end
    return (url, tag, commit, submodules)
end

# Resolved object name of the current checkout. "" when it cannot be determined
# (not a repo, detached in a broken state, git missing) — callers must treat an
# empty result as "unknown", never as "matches".
function _git_head_commit(dep_path::AbstractString)
    try
        return lowercase(String(strip(read(`git -C $dep_path rev-parse HEAD`, String))))
    catch
        return ""
    end
end

"""
Verify the checkout at `dep_path` against a `commit` pin declared in the recipe.

Hard error on mismatch: the recipe asserted exact content and upstream handed us
something else, which is indistinguishable from a moved tag or a hijacked repo.
Refusing to build is the whole point of the pin — a warning here would let the
compile proceed on unverified source, which is the failure this exists to stop.

No pin declared ⇒ nothing to verify (returns the resolved sha for the marker).

With `submodules=true` every listed submodule is verified as well
(`_verify_dep_submodules`), pinned or not. Clone and re-checkout pass `false` and
verify them after fetching: the superproject is verified first, because its
`.gitmodules` says where the submodules are fetched from.
"""
function _verify_dep_pin(name::AbstractString, dep, dep_path::AbstractString;
                         submodules::Bool=true)::String
    head = _git_head_commit(dep_path)
    pin = hasproperty(dep, :commit) ? dep.commit : ""

    if !isempty(pin)
        if isempty(head)
            error("""
                Dependency '$name' declares commit = "$pin" but its checkout could not be resolved.
                  path: $dep_path
                Refusing to build against unverifiable source.""")
        elseif head != lowercase(pin)
            error("""
                Dependency '$name' FAILED its commit pin — refusing to build.
                  url:      $(dep.url)
                  tag:      $(_tagdisp(dep.tag))
                  expected: $pin
                  actual:   $head
                A tag is a mutable ref. If upstream legitimately re-tagged, review the diff
                between those two commits before updating `commit` in the recipe.""")
        end
    end
    submodules && _verify_dep_submodules(name, dep, dep_path)
    return head
end

_dep_submodules(dep) = hasproperty(dep, :submodules) ? dep.submodules : String[]

# Entries are `/`-separated at load (ConfigurationManager._parse_dep_submodules).
_dep_subpath(dep_path::AbstractString, sub::AbstractString) = joinpath(dep_path, split(sub, '/')...)

# The commit the superproject's checked-out tree records for each of `paths`.
# Only gitlinks (mode 160000) count: a path that is a directory, a file, or
# absent is missing from the result. `-z` keeps paths unquoted.
function _recorded_submodule_commits(dep_path::AbstractString, paths::AbstractVector)
    recorded = Dict{String,String}()
    out = read(`git -C $dep_path ls-tree -z HEAD -- $paths`, String)
    for entry in split(out, '\0'; keepempty=false)
        meta, path = split(entry, '\t'; limit=2)
        mode, type, sha = split(meta)
        mode == "160000" && type == "commit" && (recorded[String(path)] = lowercase(sha))
    end
    return recorded
end

# A listed path the superproject records no submodule at is the recipe's error.
# Refuse it before `git submodule update`, whose failure would read as a failed
# fetch and resolve the dependency to nothing.
function _require_recorded_submodules(name::AbstractString, dep, dep_path::AbstractString)
    subs = _dep_submodules(dep)
    recorded = _recorded_submodule_commits(dep_path, subs)
    absent = [s for s in subs if !haskey(recorded, s)]
    isempty(absent) || error("""
        Dependency '$name' lists submodules its superproject does not record: $(join(absent, ", "))
          url: $(dep.url)
          tag: $(_tagdisp(dep.tag))
          at:  $(_git_head_commit(dep_path))
        Each entry is a submodule's path in the superproject, as `git submodule status` prints it.""")
    return recorded
end

# Fetch exactly the listed submodules, each at the commit the superproject
# records, never recursively. `--checkout` overrides a `.gitmodules`
# `update = none|merge|rebase`, which upstream controls. `--force`: the clone is
# a cache we own, the same reason `_recheckout_dep` forces its checkout.
function _fetch_dep_submodules(name::AbstractString, dep, dep_path::AbstractString)
    subs = _dep_submodules(dep)
    isempty(subs) && return nothing
    _require_recorded_submodules(name, dep, dep_path)
    println("    $name: fetching submodules $(_subsdisp(subs))")
    run(`git -C $dep_path submodule --quiet update --init --checkout --force --depth 1 -- $subs`)
    return nothing
end

"""
Verify every listed submodule's checkout against the commit the superproject's
tree records for it. Hard error on a mismatch or a missing checkout, like a
`commit` pin mismatch: the superproject's pin covers its submodules only while
they are what its tree says.
"""
function _verify_dep_submodules(name::AbstractString, dep, dep_path::AbstractString)
    subs = _dep_submodules(dep)
    isempty(subs) && return nothing
    recorded = _require_recorded_submodules(name, dep, dep_path)
    for s in subs
        sub_path = _dep_subpath(dep_path, s)
        # An unfetched submodule is an empty directory, and `git -C` there answers
        # with the superproject's HEAD: only a checkout with its own `.git` counts.
        actual = ispath(joinpath(sub_path, ".git")) ? _git_head_commit(sub_path) : ""
        actual == recorded[s] && continue
        error("""
            Dependency '$name' submodule '$s' is not at the commit its superproject records — refusing to build.
              url:      $(dep.url)
              tag:      $(_tagdisp(dep.tag))
              recorded: $(recorded[s])
              actual:   $(isempty(actual) ? "not fetched" : actual)
              path:     $sub_path
            Delete $dep_path (or `RepliBuild.clean` the project) to fetch it again.""")
    end
    return nothing
end

# Best-effort origin URL for a legacy clone that predates the marker.
function _git_origin_url(dep_path::AbstractString)
    try
        return String(strip(read(`git -C $dep_path remote get-url origin`, String)))
    catch
        return ""
    end
end

# What is currently cached at dep_path? Returns (url, tag, commit, submodules, have_marker).
#   marker present            → authoritative url+tag+commit+submodules
#   marker absent, dir present → infer url from git, tag/commit unknown ("") ⇒ re-checkout;
#                                no submodules (nothing before the marker fetched any)
#   nothing cached            → ("", "", "", [], false)
function _read_dep_state(marker_path::AbstractString, dep_path::AbstractString)
    if isfile(marker_path)
        url, tag, commit, submodules = _read_dep_marker(marker_path)
        return (url, tag, commit, submodules, true)
    elseif isdir(dep_path)
        return (_git_origin_url(dep_path), "", "", String[], false)
    else
        return ("", "", "", String[], false)
    end
end

function _clone_dep(name::AbstractString, dep, dep_path::AbstractString, marker_path::AbstractString)::Bool
    println("    cloning $(dep.url)...")
    cloned = false
    try
        # Use '--' to separate options from positional arguments
        run(`git clone --quiet -- $(dep.url) $dep_path`)
        cloned = true
        if !isempty(dep.tag)
            cd(dep_path) do
                # Tag is validated by the caller (no '-' prefix), so safe to pass directly.
                # Do NOT use '--' here — it tells git to treat the arg as a file path.
                run(`git checkout --quiet $(dep.tag)`)
            end
        end
        # Verify BEFORE the marker is written: a failed pin must not leave a record
        # claiming this content was resolved successfully.
        head = _verify_dep_pin(name, dep, dep_path; submodules=false)
        _fetch_dep_submodules(name, dep, dep_path)
        _verify_dep_submodules(name, dep, dep_path)
        println("    $name: resolved $(_tagdisp(dep.tag)) → $(_shortsha(head))")
        _write_dep_marker(marker_path, dep.url, dep.tag, head, _dep_submodules(dep))
        return true
    catch e
        # A failed PIN is a refusal to build, not a fetch failure — never swallow it
        # into a `continue`, or the dependency silently resolves to nothing and the
        # compile fails later with a missing-header error that names no cause.
        if !isa(e, ProcessFailedException) && cloned
            rm(dep_path; recursive=true, force=true)
            rethrow()
        end
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
                # --force: a re-checkout onto a tag that moved upstream leaves the old
                # ref locally; without it git reports "already on" and silently keeps
                # the stale tree. The clone is a cache we own, so discarding local
                # state here is correct.
                run(`git checkout --quiet --force $(dep.tag)`)
            end
        end
        # A checkout moves the gitlinks, not the submodules' working trees: they
        # are fetched again at what the new tree records.
        head = _verify_dep_pin(name, dep, dep_path; submodules=false)
        _fetch_dep_submodules(name, dep, dep_path)
        _verify_dep_submodules(name, dep, dep_path)
        _write_dep_marker(marker_path, dep.url, dep.tag, head, _dep_submodules(dep))
        return true
    catch e
        # Same rule as _clone_dep: a pin refusal propagates, a fetch/checkout failure
        # falls back to a fresh clone.
        isa(e, ProcessFailedException) || rethrow()
        @warn "Failed to re-checkout dependency $name to '$(dep.tag)'" exception=e
        return false
    end
end

"""
Add one source root of a git dependency: the include heuristic (the root,
`include/`, `src/`), then every `.c`/`.cpp`/`.cc`/`.cxx` under it outside the
skipped directories and `exclude`. `exclude` matches a file's path relative to
`dep_path`, the dependency's root, so a submodule's files are named as the
dependency's tree shows them (`libs/math/…`), whichever root found them.

A directory holding a `.git` is another checkout, which in a clone can only be
a fetched submodule, and never belongs to this root: a listed submodule is a
root of its own, and one fetched before its entry was removed adds nothing.
"""
function _collect_git_dep_root!(includes::Vector{String}, sources::Vector{String},
                                root::AbstractString, dep_path::AbstractString,
                                exclude::AbstractVector)
    push!(includes, root)
    for subdir in ("include", "src")
        d = joinpath(root, subdir)
        isdir(d) && push!(includes, d)
    end

    # Find all .cpp/.c files under the root (ignoring tests/examples typically)
    for (dir, dirs, files) in walkdir(root)
        filter!(d -> !in(d, ["test", "tests", "testes", "example", "examples", "fuzzing", "build", ".git", "doc", "docs"]) &&
                     !ispath(joinpath(dir, d, ".git")), dirs)
        for file in files
            rel_path = replace(relpath(joinpath(dir, file), dep_path), "\\" => "/")
            should_exclude = false
            for ex in exclude
                if file == ex || startswith(rel_path, ex) || endswith(rel_path, ex) || occursin(ex, rel_path)
                    should_exclude = true
                    break
                end
            end
            if should_exclude
                continue
            end
            if endswith(file, ".cpp") || endswith(file, ".cc") || endswith(file, ".cxx") || endswith(file, ".c")
                push!(sources, joinpath(dir, file))
            end
        end
    end
    return nothing
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

            declared_pin = hasproperty(dep, :commit) ? dep.commit : ""
            if !isempty(declared_pin) && startswith(declared_pin, "-")
                @warn "Rejecting git dependency $name: commit starts with '-' (potential option injection)"
                continue
            end

            dep_path = joinpath(deps_cache, name)
            marker_path = _dep_marker_path(deps_cache, name)
            cached_url, cached_tag, cached_commit, cached_subs, have_marker = _read_dep_state(marker_path, dep_path)

            # A cache hit on url+tag proves nothing about CONTENT — the tag may have
            # been repointed, or the checkout moved locally. Compare the recorded
            # object name against what is actually checked out.
            head_now = (have_marker && isdir(dep_path)) ? _git_head_commit(dep_path) : ""
            content_drifted = have_marker && !isempty(cached_commit) &&
                              !isempty(head_now) && head_now != cached_commit
            # The recipe's pin changing is a deliberate version move, not drift.
            pin_changed = !isempty(declared_pin) && !isempty(cached_commit) &&
                          lowercase(declared_pin) != cached_commit
            # Both lists are sorted and deduplicated (at load, and as written).
            subs_changed = cached_subs != _dep_submodules(dep)

            if !isdir(dep_path) || cached_url != dep.url
                # Absent, or the upstream URL changed → (re)clone from scratch.
                if isdir(dep_path)
                    println("    $name: url changed → re-cloning")
                    rm(dep_path; recursive=true, force=true)
                end
                _clone_dep(name, dep, dep_path, marker_path) || continue
            elseif cached_tag != dep.tag || content_drifted || pin_changed || subs_changed
                # Same repo, but something about the requested or actual content moved:
                # a tag bump, a changed pin, a changed submodule list, a legacy cache
                # with no recorded tag, or a checkout that drifted out from under the
                # marker. All re-resolve.
                if content_drifted
                    @warn """
                        Dependency '$name' checkout drifted from its recorded commit — re-resolving.
                          recorded: $cached_commit
                          actual:   $head_now
                        The cached clone is not the source that was resolved for the last build."""
                elseif pin_changed
                    println("    $name: pin $(_shortsha(cached_commit)) → $(_shortsha(declared_pin)), re-resolving")
                elseif have_marker && cached_tag == dep.tag
                    println("    $name: submodules $(_subsdisp(cached_subs)) → $(_subsdisp(_dep_submodules(dep))), re-resolving")
                elseif have_marker
                    println("    $name: $(_tagdisp(cached_tag)) → $(_tagdisp(dep.tag)), re-resolving")
                else
                    println("    $name: verifying cached checkout (no version marker)")
                end
                if !_recheckout_dep(name, dep, dep_path, marker_path)
                    rm(dep_path; recursive=true, force=true)
                    _clone_dep(name, dep, dep_path, marker_path) || continue
                end
            elseif !have_marker || isempty(cached_commit)
                # Up to date but unmarked, or marked by a pre-commit-tracking build →
                # verify any declared pin and record the resolved object name.
                head = _verify_dep_pin(name, dep, dep_path)
                _write_dep_marker(marker_path, dep.url, dep.tag, head, _dep_submodules(dep))
            else
                # Full cache hit with verified content identity. Still check the pin:
                # the recipe may have GAINED a `commit` since this clone was made.
                # A listed submodule moved off its recorded commit fails here.
                _verify_dep_pin(name, dep, dep_path)
            end

            # The dependency root and each fetched submodule are source roots alike.
            for root in vcat([dep_path], [_dep_subpath(dep_path, s) for s in _dep_submodules(dep)])
                _collect_git_dep_root!(extra_includes, extra_sources, root, dep_path, dep.exclude)
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
                    # `.c` too. A local C dependency used to contribute headers
                    # and nothing else: the sources were never compiled, and
                    # the symbols were simply absent from the library.
                    if endswith(file, ".cpp") || endswith(file, ".cc") || endswith(file, ".cxx") || endswith(file, ".c")
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
        config.compile.aot_thunks,
        config.compile.visibility
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