#!/usr/bin/env julia
# PackageRegistry.jl - Global registry for RepliBuild wrapper packages
#
# A project builds where it lives. `register` records it in ~/.replibuild/,
# vendors its build there (builds/<hash>/), and writes a Julia package around
# that build (packages/<Module>/) which it adds to a Julia environment. A
# wrapper then loads by `use(name)` from any session, or by `using <Module>`.

module PackageRegistry

using ..ConfigurationManager
using ..DependencyResolver
using ..EnvironmentDoctor
import ..Compiler
using TOML
using SHA
using Dates
using UUIDs
using Downloads
import Libdl
import Pkg

export register, unregister, use, list_registry, search, scaffold_package
export RegistryEntry, RegistryIndex

# =============================================================================
# CONSTANTS & PATHS
# =============================================================================

const REGISTRY_DIR = "registry"
const BUILDS_DIR = "builds"
const PACKAGES_DIR = "packages"
const INDEX_FILE = "index.toml"
const TOOLCHAIN_CACHE = "toolchain.toml"
const DEFAULT_HUB_URL = "https://raw.githubusercontent.com/obsidianjulua/RepliBuild-Hub/main"

"""Return the global RepliBuild home directory, respecting REPLIBUILD_HOME env var."""
function _replibuild_home()::String
    home = get(ENV, "REPLIBUILD_HOME", joinpath(homedir(), ".replibuild"))
    mkpath(home)
    return home
end

function _registry_dir()::String
    d = joinpath(_replibuild_home(), REGISTRY_DIR)
    mkpath(d)
    return d
end

function _builds_dir()::String
    d = joinpath(_replibuild_home(), BUILDS_DIR)
    mkpath(d)
    return d
end

function _packages_dir()::String
    d = joinpath(_replibuild_home(), PACKAGES_DIR)
    mkpath(d)
    return d
end

function _index_path()::String
    joinpath(_registry_dir(), INDEX_FILE)
end

# =============================================================================
# REGISTRY DATA STRUCTURES
# =============================================================================

struct RegistryEntry
    name::String
    hash::String
    toml_path::String           # the project's own replibuild.toml; `use` builds there
    source_url::String          # git URL if dependency-based
    source_tag::String          # git tag/branch
    registered_at::DateTime
    verified::Bool              # true once a build has been vendored
    project_root::String        # the project's root, absolute
    build_hash::String          # hash_config of the vendored build (builds/<hash>/); "" before one
    module_name::String         # the wrapper module, and the Julia package written around it
    julia_env::String           # environment directory the package is added to; "" for none
end

"""A copy of `entry` with the named fields replaced."""
_with(entry::RegistryEntry; kw...) =
    RegistryEntry((haskey(kw, f) ? kw[f] : getfield(entry, f) for f in fieldnames(RegistryEntry))...)

struct RegistryIndex
    entries::Dict{String, RegistryEntry}
end

# =============================================================================
# SHA256 CONTENT HASHING
# =============================================================================

"""
Compute SHA256 hash of a TOML config file's content.
Normalizes by parsing and re-serializing to ignore formatting differences.
"""
function hash_toml(toml_path::String)::String
    if !isfile(toml_path)
        error("TOML file not found: $toml_path")
    end
    data = TOML.parsefile(toml_path)
    # Normalize: serialize to canonical form
    buf = IOBuffer()
    TOML.print(buf, data)
    content = take!(buf)
    return bytes2hex(sha256(content))
end

"""
Compute SHA256 hash of a RepliBuildConfig (for build artifact caching).
Includes TOML + source content + dependency state.
"""
function hash_config(config::RepliBuildConfig)::String
    ctx = SHA.SHA256_CTX()

    # Hash the TOML config file
    if isfile(config.config_file)
        SHA.update!(ctx, read(config.config_file))
    end

    # Hash source file contents
    for src in config.compile.source_files
        if isfile(src)
            SHA.update!(ctx, read(src))
        end
    end

    # Hash include directory headers, recursively. A flat `readdir` missed
    # `include/<lib>/<lib>.h`, so editing a nested header left `use()` on the
    # cached build. Same walk `Compiler.compute_project_hash` already does;
    # the path is mixed in so a rename is a change even when the bytes are not.
    for inc_dir in config.compile.include_dirs
        isdir(inc_dir) || continue
        for (root, dirs, files) in walkdir(inc_dir)
            filter!(d -> !(d in (".git", "build", ".replibuild_cache")), dirs)
            sort!(dirs)
            for f in sort(files)
                any(endswith(f, ext) for ext in (".h", ".hpp", ".hxx", ".hh")) || continue
                path = joinpath(root, f)
                isfile(path) || continue
                SHA.update!(ctx, Vector{UInt8}(relpath(path, inc_dir)))
                SHA.update!(ctx, read(path))
            end
        end
    end

    # Hash git HEAD if in a repo
    git_head = _get_git_head(config.project.root)
    if !isempty(git_head)
        SHA.update!(ctx, Vector{UInt8}(git_head))
    end

    # Hash the generator itself: a cached build is only valid for the codegen
    # that produced it, otherwise use() serves wrappers from old RepliBuild
    # versions forever.
    SHA.update!(ctx, Vector{UInt8}(_generator_fingerprint()))

    # ...and only for the toolchain that built it. A C build made on Julia 1.13
    # (libLLVM 20) carries Tier 1 slices that LLVM 18 cannot parse, so use()
    # under 1.12 must miss rather than serve it. See `_toolchain_identity`.
    SHA.update!(ctx, Vector{UInt8}(Compiler._toolchain_identity(config)))

    return bytes2hex(SHA.digest!(ctx))
end

"""
Fingerprint of the RepliBuild generator itself: package version plus git HEAD
when running from a dev checkout (the version alone doesn't move on every
codegen change), plus a digest of any UNCOMMITTED changes — HEAD alone means a
dev editing the generator gets served wrappers from the pre-edit code until
they commit, which is exactly the local-dev workflow this cache serves.
Mixed into `hash_config` so stale wrappers miss the cache.
"""
function _generator_fingerprint()::String
    version = something(pkgversion(@__MODULE__), "unknown")
    root = pkgdir(@__MODULE__)
    head = root === nothing ? "" : _get_git_head(root)
    dirty = root === nothing ? "" : _get_git_dirty_digest(root)
    return "replibuild-$version@$head$dirty"
end

"""
The paths whose uncommitted edits can change what the generator emits: the
source tree and the package's own Project.toml (its version is in
`BUILD_GENERATOR`). CLAUDE.md, CHANGELOG.md, docs/ and test/ are not among
them: an edit there made every vendored build miss its cache and re-vendor,
on a checkout that is dirty in those files most of the time.
"""
const _GENERATOR_DIRTY_PATHS = ("src", "Project.toml")

"""
Content digest of the checkout's uncommitted state under
`_GENERATOR_DIRTY_PATHS`: tracked-file diffs plus the status listing (which
names untracked files). Empty string for a tree that is clean there or when git
is unavailable — degrading to the HEAD-only fingerprint rather than disabling
caching outright.
"""
function _get_git_dirty_digest(root::String)::String
    try
        paths = collect(_GENERATOR_DIRTY_PATHS)
        status = read(`git -C $root status --porcelain -- $paths`, String)
        isempty(strip(status)) && return ""
        diff = read(`git -C $root diff HEAD --no-color -- $paths`, String)
        return "+dirty-" * bytes2hex(SHA.sha256(status * diff))[1:12]
    catch
        return ""
    end
end

function _get_git_head(dir::String)::String
    try
        head_file = joinpath(dir, ".git", "HEAD")
        if isfile(head_file)
            head_content = strip(read(head_file, String))
            if startswith(head_content, "ref: ")
                ref_path = joinpath(dir, ".git", head_content[6:end])
                isfile(ref_path) && return strip(read(ref_path, String))
            end
            return head_content
        end
    catch
    end
    return ""
end

# =============================================================================
# INDEX I/O
# =============================================================================

function _load_index()::RegistryIndex
    path = _index_path()
    if !isfile(path)
        return RegistryIndex(Dict{String, RegistryEntry}())
    end

    data = TOML.parsefile(path)
    entries = Dict{String, RegistryEntry}()

    for (name, info) in data
        name isa String || continue
        info isa Dict || continue
        entries[name] = RegistryEntry(
            name,
            get(info, "hash", ""),
            get(info, "toml_path", ""),
            get(info, "source_url", ""),
            get(info, "source_tag", ""),
            DateTime(get(info, "registered_at", string(now()))),
            get(info, "verified", false),
            get(info, "project_root", ""),
            # Absent from an index written before entries pointed at their project.
            get(info, "build_hash", ""),
            get(info, "module_name", ""),
            get(info, "julia_env", ""),
        )
    end

    return RegistryIndex(entries)
end

"""Store one entry, replacing any of the same name."""
function _save_entry(entry::RegistryEntry)
    index = _load_index()
    index.entries[entry.name] = entry
    _save_index(index)
    return entry
end

function _save_index(index::RegistryIndex)
    path = _index_path()
    data = Dict{String, Any}()

    for (name, entry) in index.entries
        data[name] = Dict{String, Any}(
            "hash" => entry.hash,
            "toml_path" => entry.toml_path,
            "source_url" => entry.source_url,
            "source_tag" => entry.source_tag,
            "registered_at" => string(entry.registered_at),
            "verified" => entry.verified,
            "project_root" => entry.project_root,
            "build_hash" => entry.build_hash,
            "module_name" => entry.module_name,
            "julia_env" => entry.julia_env,
        )
    end

    open(path, "w") do io
        TOML.print(io, data)
    end
end

function _within_root(path::AbstractString, root::AbstractString)::Bool
    path = normpath(abspath(String(path)))
    root = normpath(abspath(String(root)))
    root == "/" || (root = rstrip(root, ['/', '\\']))
    return path == root || startswith(path, root * Base.Filesystem.path_separator)
end

"""
The project root a replibuild.toml names, absolute: `[project] root` through the
host's `[platform.*]` overlay, resolved against the TOML's folder, as
`load_config` resolves it.

The registry used to copy a project's sources into itself and build the copy,
because that is where the stored TOML's relative root pointed. Two projects with
the same relative paths then overwrote each other, and edits to the project
never reached `use`. Pointing at the project itself removes both.
"""
function _project_root(data::Dict, toml_path::String)::String
    data = try
        ConfigurationManager.apply_platform_overlay(data, toml_path)
    catch
        data
    end
    project = get(data, "project", Dict{String,Any}())
    raw = project isa Dict ? string(get(project, "root", ".")) : "."
    root = normpath(isabspath(raw) ? raw : joinpath(dirname(toml_path), raw))
    return root == "/" ? root : String(rstrip(root, ['/', '\\']))
end

# Namespace for package UUIDs derived from a registry name, when the project's
# TOML has no `[project] uuid`. Fixed: changing it changes every such UUID.
const _PACKAGE_UUID_NAMESPACE = UUID("8b3f5f1e-6c1a-5a52-9a63-3f0d2c5e7a41")

"""
The Julia package UUID for a registered project: its `[project] uuid`, which
`discover` writes once, or else one derived from the registry name — never a
fresh `uuid4()`, or the package would change identity on every vendoring.
"""
function _package_uuid(toml_path::String, name::AbstractString)::UUID
    project = get(TOML.parsefile(toml_path), "project", Dict{String,Any}())
    u = project isa Dict ? tryparse(UUID, string(get(project, "uuid", ""))) : nothing
    return u === nothing ? uuid5(_PACKAGE_UUID_NAMESPACE, "replibuild:" * String(name)) : u
end

"""The package version: `[project] version` when it parses, else 0.1.0."""
function _package_version(toml_path::String)::VersionNumber
    project = get(TOML.parsefile(toml_path), "project", Dict{String,Any}())
    v = project isa Dict ? tryparse(VersionNumber, string(get(project, "version", ""))) : nothing
    return something(v, v"0.1.0")
end

"""RepliBuild's own version: the generator a vendored wrapper needs at least."""
_replibuild_version()::VersionNumber = getfield(parentmodule(@__MODULE__), :VERSION)

"""
The `[compat]` entry for RepliBuild in a package RepliBuild writes: caret, so at
least the RepliBuild that wrote it and below the next major. Both the registry's
packages and `scaffold_package` use it; the scaffold's literal `"2"` resolved
RepliBuild 2.x.
"""
function _replibuild_compat()::String
    rb = _replibuild_version()
    return "$(rb.major).$(rb.minor).$(rb.patch)"
end

"""
Write `packages/<Module>/`: a Julia package whose `src/<Module>.jl` includes
the vendored wrapper in `builds/<hash>/`.

The wrapper finds its library and metadata beside itself, so the package holds
no binary. Vendoring a new build rewrites one small file rather than replacing a
DLL that a running session may hold open, which Windows refuses.
"""
function _write_package(module_name::String, uuid::UUID, version::VersionNumber,
                        wrapper_path::String)::String
    pkg_dir = joinpath(_packages_dir(), module_name)
    mkpath(joinpath(pkg_dir, "src"))
    project = Dict{String,Any}(
        "name" => module_name,
        "uuid" => string(uuid),
        "version" => string(version),
        "deps" => Dict{String,Any}(
            "Libdl" => string(Base.PkgId(Libdl).uuid),
            "RepliBuild" => _replibuild_uuid(),
        ),
        # Caret compat: at least the RepliBuild that generated the wrapper.
        "compat" => Dict{String,Any}(
            "RepliBuild" => _replibuild_compat(),
            "julia" => "1.10",
        ),
    )
    open(io -> TOML.print(io, project; sorted=true), joinpath(pkg_dir, "Project.toml"), "w")
    # A package's entry file is evaluated in `Base.__toplevel__`, which has no
    # `include` of its own, and the wrapper file defines `module <Module>`
    # itself, so it is evaluated there directly rather than inside a module.
    write(joinpath(pkg_dir, "src", module_name * ".jl"), """
        # Written by RepliBuild (register / use). It loads the build vendored in
        # $(dirname(wrapper_path)),
        # and is rewritten whenever that build changes. Do not edit.
        Base.include(Base.__toplevel__, $(repr(wrapper_path)))
        """)
    return pkg_dir
end

"""
The Julia environment `register` adds packages to unless told otherwise:
`ENV["REPLIBUILD_JULIA_ENV"]` (`default`, `active`, `none`, or a path), else
`:default`, the `@v#.#` environment every session can load from.
"""
function _default_julia_env()
    s = String(strip(get(ENV, "REPLIBUILD_JULIA_ENV", "")))
    isempty(s) && return :default
    s in ("default", "active", "none") && return Symbol(s)
    return s
end

"""The environment directory `env` names, or `nothing` for none."""
function _julia_env_dir(env)::Union{String,Nothing}
    (env === nothing || env === :none) && return nothing
    env === :default && return joinpath(first(DEPOT_PATH), "environments",
                                        "v$(Base.VERSION.major).$(Base.VERSION.minor)")
    env === :active && return dirname(Base.active_project())
    env isa AbstractString ||
        error("env must be :default, :active, :none, nothing, or a path; got $(repr(env))")
    p = abspath(String(env))
    return basename(p) in ("Project.toml", "JuliaProject.toml") ? dirname(p) : p
end

"""
The path the environment in `env_dir` loads package `name` from, if it lists the
package as developed by path; `""` when it lists it otherwise, `nothing` when it
does not list it at all.
"""
function _env_package_path(env_dir::AbstractString, name::AbstractString)::Union{String,Nothing}
    proj = joinpath(env_dir, "Project.toml")
    isfile(proj) || return nothing
    haskey(get(TOML.parsefile(proj), "deps", Dict{String,Any}()), name) || return nothing
    manifest = joinpath(env_dir, "Manifest.toml")
    isfile(manifest) || return ""
    entries = get(get(TOML.parsefile(manifest), "deps", Dict{String,Any}()), name, Any[])
    for e in entries
        e isa Dict && haskey(e, "path") || continue
        p = String(e["path"])
        return normpath(isabspath(p) ? p : joinpath(env_dir, p))
    end
    return ""
end

"""Whether `dir` is a registry install under some depot's `packages/`, not a dev checkout."""
_is_registry_install(dir::AbstractString)::Bool =
    any(d -> _within_root(dir, joinpath(d, "packages")), DEPOT_PATH)

"""
Add the vendored package at `pkg_dir` to the Julia environment in `env_dir`, by
path, so `using <Module>` loads it there. Returns whether it is there now.

Only environments change, never a package's own Project.toml (one with a
`name`): adding a dependency to RepliBuild's own project, say, would be a
change to RepliBuild. An environment without RepliBuild gets the running copy,
which generated the wrapper; one that has RepliBuild keeps the copy it has.
Pkg's automatic precompilation is off here; the first `using` precompiles.
"""
function _add_to_julia_env(pkg_dir::String, module_name::String, env_dir::String;
                           verbose::Bool=true)::Bool
    proj = joinpath(env_dir, "Project.toml")
    if isfile(proj) && haskey(TOML.parsefile(proj), "name")
        @warn "Not adding $module_name to $env_dir: that Project.toml belongs to a package, not an environment. Pass `env` an environment, or `nothing`."
        return false
    end
    current = _env_package_path(env_dir, module_name)
    current !== nothing && _within_root(current, pkg_dir) && _within_root(pkg_dir, current) && return true
    if current !== nothing
        @warn "Not adding $module_name to $env_dir: that environment already has a different package of that name."
        return false
    end
    rb_dir = pkgdir(parentmodule(@__MODULE__))
    try
        mkpath(env_dir)
        withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
            Pkg.activate(env_dir) do
                if _env_package_path(env_dir, "RepliBuild") === nothing
                    if rb_dir !== nothing && !_is_registry_install(rb_dir)
                        Pkg.develop(Pkg.PackageSpec(path=rb_dir); io=devnull)
                    else
                        Pkg.add(Pkg.PackageSpec(name="RepliBuild", uuid=UUID(_replibuild_uuid()),
                                                version=_replibuild_version()); io=devnull)
                    end
                end
                Pkg.develop(Pkg.PackageSpec(path=pkg_dir); io=devnull)
            end
        end
    catch e
        @warn "Could not add $module_name to the Julia environment $env_dir. RepliBuild.use still loads it." exception=e
        return false
    end
    verbose && println("  julia env: $env_dir — `using $module_name` loads it there")
    return true
end

"""Remove `module_name` from the environment in `env_dir`, if it is the vendored package."""
function _remove_from_julia_env(module_name::String, env_dir::String)
    pkg_dir = joinpath(_packages_dir(), module_name)
    current = _env_package_path(env_dir, module_name)
    (current !== nothing && _within_root(current, pkg_dir)) || return
    try
        Pkg.activate(env_dir) do
            Pkg.rm(Pkg.PackageSpec(name=module_name); io=devnull)
        end
    catch e
        @warn "Could not remove $module_name from the Julia environment $env_dir." exception=e
    end
end

"""Delete a directory the registry made, or say why it could not."""
function _rm_vendored(path::String)
    isdir(path) || return
    try
        rm(path; recursive=true, force=true)
    catch e
        @warn "Could not delete $path. A running Julia session may still have its library loaded; close it and delete the folder." exception=e
    end
end

# =============================================================================
# REGISTER / UNREGISTER
# =============================================================================

"""
    register(toml_path; name="", build=true, env=:default) -> RegistryEntry

Record a project in the global registry (`~/.replibuild/`) and vendor its build.

The project builds where it lives. `register` runs `build` and `wrap` on the
project itself (a cache hit when its build is current), copies the result to
`~/.replibuild/builds/<hash>/`, and writes a Julia package around it in
`~/.replibuild/packages/<Module>/`. From then on `use(name)` loads that build in
any session, and `using <Module>` loads it in the Julia environment `env`.

- `name`: the registry name; default `[project].name`.
- `build=false`: record the project only. `use(name)` builds and vendors it.
- `env`: where the package goes — `:default` (the `@v#.#` environment, which
  every session can load from), `:active`, the path of an environment, or
  `nothing`. `ENV["REPLIBUILD_JULIA_ENV"]` (`default`, `active`, `none`, or a
  path) changes the default. A package's own Project.toml is never changed.
"""
function register(toml_path::String; name::String="", verified::Bool=false,
                  build::Bool=true, env=_default_julia_env())::RegistryEntry
    toml_path = abspath(toml_path)
    if !isfile(toml_path)
        error("TOML file not found: $toml_path")
    end

    # Parse to get project name and dependency info
    data = TOML.parsefile(toml_path)
    if isempty(name)
        project = get(data, "project", Dict())
        name = get(project, "name", "")
        if isempty(name)
            error("Cannot infer package name: no [project].name in TOML and no name= provided")
        end
    end

    # Compute content hash
    content_hash = hash_toml(toml_path)

    # Extract git dependency info if present
    deps = get(data, "dependencies", Dict())
    source_url = ""
    source_tag = ""
    for (dep_name, dep_info) in deps
        if dep_info isa Dict && get(dep_info, "type", "") == "git"
            source_url = get(dep_info, "url", "")
            source_tag = get(dep_info, "tag", "")
            break
        end
    end

    # The entry points at the project's own TOML: `use` builds the project
    # where it lives. A build vendored earlier stays recorded until a new one
    # replaces it; `use` tells a stale one from the hash.
    env_dir = _julia_env_dir(env)
    previous = get(_load_index().entries, name, nothing)
    entry = RegistryEntry(
        name, content_hash, toml_path,
        source_url, source_tag,
        now(), verified, _project_root(data, toml_path),
        previous === nothing ? "" : previous.build_hash,
        previous === nothing ? "" : previous.module_name,
        env_dir === nothing ? "" : env_dir,
    )
    _save_entry(entry)
    println("  registered: $name ($(content_hash[1:12])…)  project: $(entry.project_root)")

    build || return entry
    return _build_and_vendor!(entry)
end

"""
Build and wrap the project where it lives, then vendor the result: copy its
output directory to `builds/<hash>/` and publish it (`_publish!`). Returns the
updated entry, saved.
"""
function _build_and_vendor!(entry::RegistryEntry; clean::Bool=false, verbose::Bool=true)::RegistryEntry
    _ensure_toolchain()
    parent_mod = parentmodule(@__MODULE__)
    parent_mod.build(entry.toml_path; clean=clean)
    parent_mod.wrap(entry.toml_path)
    # The same inputs `use` hashes: the loaded config with its dependencies resolved.
    config = DependencyResolver.resolve_dependencies(ConfigurationManager.load_config(entry.toml_path))
    config_hash = hash_config(config)
    _store_build(config_hash, ConfigurationManager.get_output_path(config))
    verbose && println("  vendored: $(_cached_build_dir(config_hash))")
    return _publish!(entry, config_hash, ConfigurationManager.get_module_name(config); verbose=verbose)
end

"""
Point the entry and its Julia package at the vendored build `config_hash`, and
add the package to the entry's Julia environment if it is not there yet.
"""
function _publish!(entry::RegistryEntry, config_hash::String, module_name::String;
                   verbose::Bool=true)::RegistryEntry
    wrapper = joinpath(_cached_build_dir(config_hash), module_name * ".jl")
    isfile(wrapper) || error("The vendored build $(_cached_build_dir(config_hash)) has no $module_name.jl.")
    pkg_dir = _write_package(module_name, _package_uuid(entry.toml_path, entry.name),
                             _package_version(entry.toml_path), wrapper)
    entry = _save_entry(_with(entry; build_hash=config_hash, module_name=module_name, verified=true))
    isempty(entry.julia_env) || _add_to_julia_env(pkg_dir, module_name, entry.julia_env; verbose=verbose)
    return entry
end

"""
The toolchain check `use` and `register` run before building, from the cached
result when it is fresh.
"""
function _ensure_toolchain()
    cached_status = _cached_env_check()
    if isnothing(cached_status)
        status = EnvironmentDoctor.check_environment(verbose=false, throw_on_error=false)
        _cache_env_check(status)
        status.tier1_ready || EnvironmentDoctor.check_environment(verbose=true, throw_on_error=true)
    elseif !cached_status.ready
        EnvironmentDoctor.check_environment(verbose=true, throw_on_error=true)
    end
    return nothing
end

"""
    unregister(name::String)

Remove a package from the global registry, with what the registry made from it:
its vendored build, its Julia package, and that package's place in a Julia
environment. The project itself is never touched.
"""
function unregister(name::String)
    index = _load_index()
    if !haskey(index.entries, name)
        @warn "Package '$name' not found in registry"
        return
    end

    entry = index.entries[name]
    others = [e for e in values(index.entries) if e.name != name]

    if !isempty(entry.module_name)
        isempty(entry.julia_env) || _remove_from_julia_env(entry.module_name, entry.julia_env)
        any(e -> e.module_name == entry.module_name, others) ||
            _rm_vendored(joinpath(_packages_dir(), entry.module_name))
    end
    # Another entry can vendor the identical build (the same project under two names).
    if !isempty(entry.build_hash) && !any(e -> e.build_hash == entry.build_hash, others)
        _rm_vendored(_cached_build_dir(entry.build_hash))
    end
    # An entry from before entries pointed at their project holds a TOML copy
    # in the registry. Never the project's own file.
    if isfile(entry.toml_path) && _within_root(entry.toml_path, _registry_dir())
        rm(entry.toml_path; force=true)
    end

    delete!(index.entries, name)
    _save_index(index)
    println("  unregistered: $name")
end

# =============================================================================
# LIST REGISTRY
# =============================================================================

"""
    list_registry()

Print all registered packages with their hash, verification status, and source info.
"""
function list_registry()
    index = _load_index()

    if isempty(index.entries)
        println("  (no packages registered)")
        println("  Use RepliBuild.register(\"path/to/replibuild.toml\") to add one.")
        return
    end

    println("  ┌─────────────────────────────────────────────────────────────┐")
    println("  │ RepliBuild Package Registry                                │")
    println("  └─────────────────────────────────────────────────────────────┘")
    println()

    for (name, entry) in sort(collect(index.entries); by=x->x[1])
        status = entry.verified ? "✓" : "○"
        hash_short = entry.hash[1:min(12, length(entry.hash))]
        source = if !isempty(entry.source_url)
            basename(entry.source_url) * (isempty(entry.source_tag) ? "" : "@$(entry.source_tag)")
        else
            "local"
        end
        # The build the entry records as vendored. This used to test
        # builds/<TOML hash>, but builds are stored under hash_config, so it
        # always said "not built".
        built = !isempty(entry.build_hash) && _has_cached_build(entry.build_hash) ?
            "vendored $(entry.build_hash[1:12])…" : "not built yet"
        project = isfile(entry.toml_path) ? entry.project_root : "(missing) $(entry.toml_path)"
        julia = isempty(entry.julia_env) ? "not in a Julia environment" :
                isempty(entry.module_name) ? "goes into $(entry.julia_env) once built" :
                "using $(entry.module_name), in $(entry.julia_env)"

        println("  $status $name")
        println("    project: $project")
        println("    hash:    $hash_short…")
        println("    source:  $source")
        println("    status:  $built")
        println("    julia:   $julia")
        println("    registered: $(Dates.format(entry.registered_at, "yyyy-mm-dd HH:MM"))")
        println()
    end
end

# =============================================================================
# HUB SEARCH
# =============================================================================

"""Return the hub base URL, respecting REPLIBUILD_HUB_URL env var."""
function _hub_url()::String
    get(ENV, "REPLIBUILD_HUB_URL", DEFAULT_HUB_URL)
end

"""Fetch and parse the hub index.toml from the remote hub URL."""
function _fetch_hub_index()::Dict{String, Any}
    url = _hub_url() * "/index.toml"
    local body::String
    try
        body = sprint() do io
            Downloads.download(url, io)
        end
    catch e
        error("Failed to fetch hub index from $url: $e")
    end
    return TOML.parse(body)
end

"""
    search(query::String="")

Search the RepliBuild Hub for available packages. Matches against package names,
descriptions, and tags. Call with no arguments to list everything.

```julia
RepliBuild.search()           # list all hub packages
RepliBuild.search("json")     # filter by keyword
```
"""
function search(query::String="")
    hub_index = _fetch_hub_index()

    if isempty(hub_index)
        println("  (hub index is empty)")
        return
    end

    query_lower = lowercase(strip(query))

    # Filter packages matching query (or show all if empty)
    matches = Pair{String, Dict{String,Any}}[]
    for (name, info) in hub_index
        info isa Dict || continue
        if isempty(query_lower) || _matches_query(name, info, query_lower)
            push!(matches, name => info)
        end
    end
    sort!(matches; by=first)

    if isempty(matches)
        println("  no packages matching \"$query\"")
        return
    end

    # Check local registry to show install status
    local_index = _load_index()

    println("  ┌─────────────────────────────────────────────────────────────┐")
    println("  │ RepliBuild Hub", isempty(query_lower) ? "" : "  ▸ \"$query_lower\"",
            repeat(" ", max(0, 45 - (isempty(query_lower) ? 0 : length(query_lower) + 5))), "│")
    println("  └─────────────────────────────────────────────────────────────┘")
    println()

    for (name, info) in matches
        desc = get(info, "description", "")
        ver = get(info, "version", "")
        lang = get(info, "language", "")
        tags = get(info, "tags", String[])
        local_status = haskey(local_index.entries, name) ? " (registered locally)" : ""

        println("  $name $ver [$lang]$local_status")
        !isempty(desc) && println("    $desc")
        !isempty(tags) && println("    tags: $(join(tags, ", "))")
        println()
    end

    println("  $(length(matches)) package$(length(matches) == 1 ? "" : "s") found")
    # `use(name)` alone fails for a name nobody registered, which is every Hub
    # package on a fresh install; the footer used to say "Install: use(name)".
    for line in _HUB_USE_STEPS
        println("  ", line)
    end
end

const _HUB_USE_STEPS = [
    "search lists Hub configs; it does not install them. To use one:",
    "  clone https://github.com/obsidianjulua/RepliBuild-Hub, then",
    "  RepliBuild.register(\"<clone>/packages/<name>/replibuild.toml\")",
    "  RepliBuild.use(\"<name>\")",
]

function _matches_query(name::String, info::Dict, query::String)::Bool
    # Match against name
    occursin(query, lowercase(name)) && return true
    # Match against description
    desc = get(info, "description", "")
    occursin(query, lowercase(desc)) && return true
    # Match against tags
    tags = get(info, "tags", String[])
    for tag in tags
        occursin(query, lowercase(tag)) && return true
    end
    # Match against language
    lang = get(info, "language", "")
    occursin(query, lowercase(lang)) && return true
    return false
end

# =============================================================================
# BUILD ARTIFACT CACHING
# =============================================================================

"""Check if build artifacts exist for a given config hash."""
function _has_cached_build(config_hash::String)::Bool
    build_dir = joinpath(_builds_dir(), config_hash)
    if !isdir(build_dir)
        return false
    end
    # Check for required artifacts
    has_lib = any(f -> endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll"),
                  readdir(build_dir))
    has_meta = isfile(joinpath(build_dir, "compilation_metadata.json"))
    has_wrapper = any(f -> endswith(f, ".jl"), readdir(build_dir))
    return has_lib && has_meta && has_wrapper
end

"""Store build artifacts in the global cache."""
function _store_build(config_hash::String, output_dir::String)
    build_dir = joinpath(_builds_dir(), config_hash)
    mkpath(build_dir)

    for f in readdir(output_dir; join=true)
        dest = joinpath(build_dir, basename(f))
        if isfile(f)
            cp(f, dest; force=true)
        elseif isdir(f)
            # Directories travel too — a Tier-1 wrapper resolves its slices
            # via joinpath(@__DIR__, "slices", …), so a cache dir without
            # slices/ silently demotes every kernel to ccall.
            cp(f, dest; force=true)
        end
    end
end

"""Get the path to cached build artifacts."""
function _cached_build_dir(config_hash::String)::String
    joinpath(_builds_dir(), config_hash)
end

# =============================================================================
# ENVIRONMENT CACHE
# =============================================================================

"""
Cache environment check results to avoid re-probing LLVM/Clang on every call.
Invalidated when LLVM version changes or cache is older than 24 hours.
"""
function _cached_env_check()::Union{ToolchainStatus, Nothing}
    cache_path = joinpath(_replibuild_home(), TOOLCHAIN_CACHE)
    if !isfile(cache_path)
        return nothing
    end

    try
        data = TOML.parsefile(cache_path)
        cached_at = DateTime(get(data, "cached_at", ""))
        # Invalidate after 24 hours
        if now() - cached_at > Hour(24)
            return nothing
        end

        # Quick check: is llvm-config still at the same path with same version?
        cached_llvm_path = get(data, "llvm_path", "")
        cached_llvm_version = get(data, "llvm_version", "")
        if !isempty(cached_llvm_path) && isfile(cached_llvm_path)
            current_version = try
                strip(read(`$cached_llvm_path --version`, String))
            catch
                ""
            end
            if current_version != cached_llvm_version
                return nothing  # version changed, re-probe
            end
        end

        # Return cached status
        ready = get(data, "ready", false)
        tier1 = get(data, "tier1_ready", false)
        tier2 = get(data, "tier2_ready", false)
        return ToolchainStatus(ToolStatus[], ready, tier1, tier2)
    catch
        return nothing
    end
end

function _cache_env_check(status::ToolchainStatus)
    cache_path = joinpath(_replibuild_home(), TOOLCHAIN_CACHE)

    # Find LLVM tool info for invalidation key
    llvm_path = ""
    llvm_version = ""
    for tool in status.tools
        if tool.name == "llvm-config" && tool.found
            llvm_path = tool.path
            llvm_version = tool.version
            break
        end
    end

    data = Dict{String, Any}(
        "cached_at" => string(now()),
        "ready" => status.ready,
        "tier1_ready" => status.tier1_ready,
        "tier2_ready" => status.tier2_ready,
        "llvm_path" => llvm_path,
        "llvm_version" => llvm_version,
    )

    open(cache_path, "w") do io
        TOML.print(io, data)
    end
end

# =============================================================================
# USE — the main orchestrator
# =============================================================================

"""
    use(name::String; force_rebuild::Bool=false, verbose::Bool=true)::Module

Load a registered wrapper by name, and return its module.

The project builds where it lives. When its inputs (TOML, sources, headers,
dependencies, toolchain, RepliBuild itself) match a vendored build, `use` loads
that build. Otherwise it builds and wraps the project in place, vendors the
result, and loads it — so edits to the project reach `use` without registering
again. A project that has moved or been deleted still loads from its last
vendored build. `force_rebuild=true` builds the project from nothing first.

```julia
Lua = RepliBuild.use("lua")
Lua.luaL_newstate()
```
"""
function use(name::String; force_rebuild::Bool=false, verbose::Bool=true)::Module
    index = _load_index()
    if !haskey(index.entries, name)
        error("Package '$name' not in registry. Use RepliBuild.list_registry() to see available packages,\n" *
              "or RepliBuild.register(\"path/to/replibuild.toml\") to add one. A RepliBuild-Hub config is\n" *
              "registered the same way, from a clone: <clone>/packages/<name>/replibuild.toml.")
    end
    entry = index.entries[name]

    verbose && println("RepliBuild | use $name")

    if !isfile(entry.toml_path)
        # The project moved or was deleted. What was vendored still loads.
        if !isempty(entry.build_hash) && !isempty(entry.module_name) && _has_cached_build(entry.build_hash)
            @warn "$name: its project is gone ($(entry.toml_path)); loading the build vendored for it. Register it again from where it is now to rebuild."
            return _load_wrapper(name, entry.build_hash, entry.module_name, verbose)
        end
        error("$name: its replibuild.toml is gone ($(entry.toml_path)) and nothing was vendored. " *
              "Register the project again from where it is now.")
    end

    config = DependencyResolver.resolve_dependencies(ConfigurationManager.load_config(entry.toml_path))
    config_hash = hash_config(config)
    module_name = ConfigurationManager.get_module_name(config)

    if !force_rebuild && _has_cached_build(config_hash)
        verbose && println("  cached build: $(config_hash[1:12])…")
        # The same inputs were vendored under another entry, or before an edit
        # that was then undone: point this entry and its package at that build.
        entry.build_hash == config_hash && entry.module_name == module_name ||
            (entry = _publish!(entry, config_hash, module_name; verbose=verbose))
        return _load_wrapper(name, config_hash, module_name, verbose)
    end

    verbose && println("  building in $(entry.project_root)")
    # `force_rebuild` builds from nothing. Without `clean`, an unchanged project
    # hash skipped the build, so it only re-wrapped: metadata extracted by an
    # older RepliBuild survived the one call meant to replace it.
    entry = _build_and_vendor!(entry; clean=force_rebuild, verbose=verbose)
    verbose && println("  done: $name ready")
    return _load_wrapper(name, entry.build_hash, entry.module_name, verbose)
end

"""Load a wrapper module from a vendored build."""
function _load_wrapper(name::String, config_hash::String, module_name::String, verbose::Bool)::Module
    build_dir = _cached_build_dir(config_hash)

    # Pick the wrapper by name, not first(): the generator names the file with
    # `get_module_name` (`vec2d` → `Vec2d.jl`). `titlecase` capitalizes after a
    # digit (`vec2d` → `Vec2D.jl`) and the loader then looked for a file the
    # generator never wrote.
    expected = module_name * ".jl"
    wrapper_path = joinpath(build_dir, expected)
    if !isfile(wrapper_path)
        avail = isdir(build_dir) ? join(filter(f -> endswith(f, ".jl"), readdir(build_dir)), ", ") : "(no such folder)"
        error("Wrapper '$expected' not found for '$name' in $build_dir. Available: $avail")
    end

    # The wrapper resolves LIBRARY_PATH relative to its own file first (the
    # .so copy _store_build placed beside it in this build dir), falling back
    # to the absolute path baked at generation time, so the vendored build
    # stays bound to its own library whatever the project's julia/ holds now.

    verbose && println("  loading: $wrapper_path")

    # Include the wrapper into an anonymous parent module to avoid name collisions.
    # The wrapper file defines `module X ... end` — Base.include returns the module.
    parent = Module()
    result = Base.include(parent, wrapper_path)
    if result isa Module
        return result
    end
    return parent
end

# =============================================================================
# SCAFFOLD PACKAGE (merged from Scaffold.jl)
# =============================================================================

"""
    scaffold_package(name::String; path::String=".", from_registry::Bool=true)::String

Generate a distributable Julia package structure for a RepliBuild wrapper.
If `from_registry` is true and the name matches a registered package, uses
the registered TOML instead of generating a blank template.

Returns the absolute path to the created package directory.
"""
function scaffold_package(name::String; path::String=".", from_registry::Bool=true)::String
    # Validate name (CamelCase)
    if !occursin(r"^[A-Z][a-zA-Z0-9]+$", name)
        error("Package name must be CamelCase starting with uppercase (e.g., 'MyEigenWrapper')")
    end

    pkg_dir = abspath(joinpath(path, name))
    if isdir(pkg_dir)
        error("Directory already exists: $pkg_dir")
    end

    # Check registry for existing TOML
    registry_toml = nothing
    if from_registry
        index = _load_index()
        # Try lowercase match
        for (reg_name, entry) in index.entries
            if lowercase(reg_name) == lowercase(name) || lowercase(reg_name) == lowercase(replace(name, "Wrapper" => ""))
                if isfile(entry.toml_path)
                    registry_toml = read(entry.toml_path, String)
                    println("  using registered TOML for: $(entry.name)")
                end
                break
            end
        end
    end

    # Create directory structure
    mkpath(joinpath(pkg_dir, "src"))
    mkpath(joinpath(pkg_dir, "deps"))
    mkpath(joinpath(pkg_dir, "test"))

    uuid = uuid4()

    # Project.toml
    open(joinpath(pkg_dir, "Project.toml"), "w") do io
        write(io, _project_toml(name, uuid))
    end

    # replibuild.toml — from registry or blank template
    open(joinpath(pkg_dir, "replibuild.toml"), "w") do io
        if !isnothing(registry_toml)
            write(io, registry_toml)
        else
            write(io, _replibuild_toml(name, uuid, pkg_dir))
        end
    end

    # src/<Name>.jl
    open(joinpath(pkg_dir, "src", "$name.jl"), "w") do io
        write(io, _main_module(name))
    end

    # deps/build.jl — uses RepliBuild.use() if registered, else raw build+wrap
    open(joinpath(pkg_dir, "deps", "build.jl"), "w") do io
        write(io, _build_jl(name))
    end

    # test/runtests.jl
    open(joinpath(pkg_dir, "test", "runtests.jl"), "w") do io
        write(io, _test_runtests(name))
    end

    # .gitignore
    open(joinpath(pkg_dir, ".gitignore"), "w") do io
        write(io, _gitignore())
    end

    println("  scaffolded: $pkg_dir")
    return pkg_dir
end

# =============================================================================
# SCAFFOLD TEMPLATES
# =============================================================================

"""Look up RepliBuild's own UUID from its loaded module identity, with a fallback
to parsing the package's Project.toml. Returns the canonical UUID string."""
function _replibuild_uuid()::String
    parent = parentmodule(@__MODULE__)
    pkgid = Base.PkgId(parent)
    if pkgid.uuid !== nothing
        return string(pkgid.uuid)
    end
    # Fallback: read Project.toml at the package root
    try
        proj = abspath(joinpath(@__DIR__, "..", "..", "Project.toml"))
        if isfile(proj)
            data = TOML.parsefile(proj)
            return get(data, "uuid", "")
        end
    catch
    end
    return ""
end

function _project_toml(name::String, uuid::UUID)::String
    rb_uuid = _replibuild_uuid()
    """
    name = "$name"
    uuid = "$uuid"
    version = "0.1.0"

    [deps]
    RepliBuild = "$rb_uuid"

    [compat]
    julia = "1.10"
    RepliBuild = "$(_replibuild_compat())"
    """
end

function _replibuild_toml(name::String, uuid::UUID, root::String)::String
    lower = lowercase(name)
    """
    [project]
    name = "$lower"
    uuid = "$uuid"
    root = "$root"

    [compile]
    flags = ["-std=c++17", "-fPIC"]
    source_files = []
    include_dirs = []

    [link]
    enable_lto = true
    optimization_level = "2"

    [binary]
    type = "shared"

    [wrap]
    style = "clang"

    [types]
    strictness = "warn"

    [cache]
    enabled = true
    """
end

function _main_module(name::String)::String
    """
    module $name

    using RepliBuild

    # Generated wrapper is loaded at build time via deps/build.jl
    const _wrapper_path = joinpath(@__DIR__, "..", "julia", "$name.jl")

    if isfile(_wrapper_path)
        include(_wrapper_path)
    else
        @warn "$name wrapper not built yet. Run: using Pkg; Pkg.build(\\\"$name\\\")"
    end

    end # module $name
    """
end

function _build_jl(name::String)::String
    lower = lowercase(replace(name, "Wrapper" => ""))
    """
    #!/usr/bin/env julia
    # deps/build.jl — Auto-build hook for $name
    # Called by Julia's Pkg manager on install/build

    using RepliBuild

    # Check toolchain first
    status = RepliBuild.check_environment(verbose=true, throw_on_error=false)
    if !status.tier1_ready
        @error "RepliBuild toolchain not available. See above for install instructions."
        exit(1)
    end

    # Build and wrap from the TOML
    toml = joinpath(@__DIR__, "..", "replibuild.toml")
    RepliBuild.build(toml)
    RepliBuild.wrap(toml)

    @info "$name built successfully"
    """
end

function _test_runtests(name::String)::String
    """
    using Test
    using $name

    @testset "$name" begin
        # Add your tests here
        @test true
    end
    """
end

function _gitignore()::String
    """
    build/
    julia/
    .replibuild_cache/
    *.so
    *.dylib
    *.dll
    """
end

end # module PackageRegistry
