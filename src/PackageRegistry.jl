#!/usr/bin/env julia
# PackageRegistry.jl - Global registry for RepliBuild wrapper packages
#
# Stores validated TOML configs as content-addressed entries in ~/.replibuild/
# Provides use(name) to resolve, build, wrap, and load wrappers by name.

module PackageRegistry

using ..ConfigurationManager
using ..DependencyResolver
using ..EnvironmentDoctor
using TOML
using SHA
using Dates
using UUIDs

export register, unregister, use, list_registry, scaffold_package
export RegistryEntry, RegistryIndex

# =============================================================================
# CONSTANTS & PATHS
# =============================================================================

const REGISTRY_DIR = "registry"
const BUILDS_DIR = "builds"
const INDEX_FILE = "index.toml"
const TOOLCHAIN_CACHE = "toolchain.toml"

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

function _index_path()::String
    joinpath(_registry_dir(), INDEX_FILE)
end

# =============================================================================
# REGISTRY DATA STRUCTURES
# =============================================================================

struct RegistryEntry
    name::String
    hash::String
    toml_path::String           # path to stored TOML in registry
    source_url::String          # git URL if dependency-based
    source_tag::String          # git tag/branch
    registered_at::DateTime
    verified::Bool              # true if build+wrap succeeded at least once
    project_root::String        # original project root at registration time
end

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

    # Hash include directory headers
    for inc_dir in config.compile.include_dirs
        if isdir(inc_dir)
            for f in sort(readdir(inc_dir; join=true))
                if isfile(f) && any(endswith(f, ext) for ext in [".h", ".hpp", ".hxx", ".hh"])
                    SHA.update!(ctx, read(f))
                end
            end
        end
    end

    # Hash git HEAD if in a repo
    git_head = _get_git_head(config.project.root)
    if !isempty(git_head)
        SHA.update!(ctx, Vector{UInt8}(git_head))
    end

    return bytes2hex(SHA.digest!(ctx))
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
        )
    end

    return RegistryIndex(entries)
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
        )
    end

    open(path, "w") do io
        TOML.print(io, data)
    end
end

# =============================================================================
# REGISTER / UNREGISTER
# =============================================================================

"""
    register(toml_path::String; name::String="", verified::Bool=false)::RegistryEntry

Hash and store a replibuild.toml in the global registry (~/.replibuild/registry/).
Name is inferred from [project].name if not provided.
Returns the created RegistryEntry.
"""
function register(toml_path::String; name::String="", verified::Bool=false)::RegistryEntry
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

    # Copy TOML into registry (content-addressed)
    stored_path = joinpath(_registry_dir(), "$(content_hash).toml")
    if !isfile(stored_path)
        cp(toml_path, stored_path)
    end

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

    # Get project root
    project = get(data, "project", Dict())
    project_root = get(project, "root", dirname(toml_path))

    entry = RegistryEntry(
        name, content_hash, stored_path,
        source_url, source_tag,
        now(), verified, project_root,
    )

    # Update index
    index = _load_index()
    index.entries[name] = entry
    _save_index(index)

    println("  registered: $name ($(content_hash[1:12])…)")
    return entry
end

"""
    unregister(name::String)

Remove a package from the global registry. Deletes stored TOML and cached builds.
"""
function unregister(name::String)
    index = _load_index()
    if !haskey(index.entries, name)
        @warn "Package '$name' not found in registry"
        return
    end

    entry = index.entries[name]

    # Remove stored TOML
    if isfile(entry.toml_path)
        rm(entry.toml_path; force=true)
    end

    # Remove cached builds
    build_dir = joinpath(_builds_dir(), entry.hash)
    if isdir(build_dir)
        rm(build_dir; recursive=true, force=true)
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
        cached = isdir(joinpath(_builds_dir(), entry.hash)) ? "cached" : "not built"

        println("  $status $name")
        println("    hash:   $hash_short…")
        println("    source: $source")
        println("    status: $cached")
        println("    registered: $(Dates.format(entry.registered_at, "yyyy-mm-dd HH:MM"))")
        println()
    end
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
        if isfile(f)
            dest = joinpath(build_dir, basename(f))
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

Load a wrapper by registry name. Resolves dependencies, checks environment,
builds if needed, wraps, and returns the loaded Julia module.

```julia
Lua = RepliBuild.use("lua")
Lua.luaL_newstate()
```
"""
function use(name::String; force_rebuild::Bool=false, verbose::Bool=true)::Module
    # 1. Lookup in registry
    index = _load_index()
    if !haskey(index.entries, name)
        error("Package '$name' not in registry. Use RepliBuild.list_registry() to see available packages,\nor RepliBuild.register(\"path/to/replibuild.toml\") to add one.")
    end
    entry = index.entries[name]

    verbose && println("RepliBuild | use $name")

    # 2. Load the stored TOML config
    if !isfile(entry.toml_path)
        error("Registry TOML missing: $(entry.toml_path)\nRe-register with RepliBuild.register()")
    end
    config = ConfigurationManager.load_config(entry.toml_path)

    # 3. Resolve dependencies (git clone, pkg-config, etc.)
    config = DependencyResolver.resolve_dependencies(config)

    # 4. Compute build hash (includes resolved sources)
    config_hash = hash_config(config)

    # 5. Check for cached build artifacts
    if !force_rebuild && _has_cached_build(config_hash)
        verbose && println("  cached build: $(config_hash[1:12])…")
        return _load_wrapper(name, config_hash, config, verbose)
    end

    # 6. Check environment (use cache when possible)
    cached_status = _cached_env_check()
    if isnothing(cached_status)
        status = EnvironmentDoctor.check_environment(verbose=false, throw_on_error=false)
        _cache_env_check(status)
        if !status.tier1_ready
            EnvironmentDoctor.check_environment(verbose=true, throw_on_error=true)
        end
    else
        if !cached_status.ready
            EnvironmentDoctor.check_environment(verbose=true, throw_on_error=true)
        end
    end

    # 7. Build (compile + link)
    verbose && println("  building...")

    # We need to call the parent module's build/wrap — access via Main.RepliBuild
    parent_mod = parentmodule(@__MODULE__)
    library_path = parent_mod.build(entry.toml_path; clean=false)

    # 8. Wrap (generate Julia bindings)
    verbose && println("  wrapping...")
    wrapper_path = parent_mod.wrap(entry.toml_path)

    # 9. Cache the build artifacts
    output_dir = ConfigurationManager.get_output_path(config)
    _store_build(config_hash, output_dir)

    # 10. Mark as verified in registry
    if !entry.verified
        verified_entry = RegistryEntry(
            entry.name, entry.hash, entry.toml_path,
            entry.source_url, entry.source_tag,
            entry.registered_at, true, entry.project_root,
        )
        index.entries[name] = verified_entry
        _save_index(index)
    end

    verbose && println("  done: $name ready")
    return _load_wrapper(name, config_hash, config, verbose)
end

"""Load a wrapper module from cached build artifacts."""
function _load_wrapper(name::String, config_hash::String, config::RepliBuildConfig, verbose::Bool)::Module
    build_dir = _cached_build_dir(config_hash)

    # Find the .jl wrapper file
    wrapper_files = filter(f -> endswith(f, ".jl"), readdir(build_dir))
    if isempty(wrapper_files)
        # Fall back to project output dir
        output_dir = ConfigurationManager.get_output_path(config)
        wrapper_files = filter(f -> endswith(f, ".jl"), readdir(output_dir))
        if isempty(wrapper_files)
            error("No wrapper .jl file found for '$name'")
        end
        wrapper_path = joinpath(output_dir, first(wrapper_files))
    else
        wrapper_path = joinpath(build_dir, first(wrapper_files))
    end

    # Also ensure the shared library is loadable — symlink from cache to expected location
    output_dir = ConfigurationManager.get_output_path(config)
    mkpath(output_dir)
    for f in readdir(build_dir)
        if any(endswith(f, ext) for ext in [".so", ".dylib", ".dll", ".json"])
            src = joinpath(build_dir, f)
            dst = joinpath(output_dir, f)
            if !isfile(dst)
                cp(src, dst; force=true)
            end
        end
    end

    # Load the wrapper as a module
    mod_name = Symbol(replace(titlecase(replace(name, "_" => " ")), " " => ""))
    verbose && println("  loading: $wrapper_path")

    # Include into a fresh module
    m = Module(mod_name)
    Base.include(m, wrapper_path)
    return m
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

function _project_toml(name::String, uuid::UUID)::String
    """
    name = "$name"
    uuid = "$uuid"
    version = "0.1.0"

    [deps]
    RepliBuild = "$(Base.PkgId(Base.UUID("00000000-0000-0000-0000-000000000000"), "RepliBuild").uuid)"

    [compat]
    julia = "1.10"
    RepliBuild = "2"
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
