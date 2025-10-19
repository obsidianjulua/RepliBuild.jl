#!/usr/bin/env julia
# ModuleRegistry.jl - External library resolution and integration
# Maps CMake find_package() to Julia JLL packages and system libraries

module ModuleRegistry

using TOML
using Pkg
using Artifacts
using Libdl

export resolve_module, register_module, list_modules, ModuleInfo

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
Represents an external library module with all necessary build information
"""
struct ModuleInfo
    name::String
    version::String
    source::Symbol  # :jll, :system, :custom, :artifact

    # Paths
    include_dirs::Vector{String}
    library_dirs::Vector{String}
    libraries::Vector{String}

    # Compiler flags
    compile_flags::Vector{String}
    link_flags::Vector{String}
    defines::Dict{String,String}

    # Package info (for JLL packages)
    julia_package::String  # e.g., "Boost_jll"
    jll_loaded::Bool

    # Metadata
    description::String
    components::Vector{String}  # e.g., ["system", "filesystem"] for Boost
    cmake_name::String  # Original CMake package name
end

# ============================================================================
# MODULE REGISTRY - In-memory cache
# ============================================================================

const REGISTRY = Dict{String,ModuleInfo}()
const MODULE_SEARCH_PATHS = String[]

function __init__()
    # Add default module search paths
    builtin_modules = joinpath(@__DIR__, "..", "modules")
    user_modules = joinpath(homedir(), ".replibuild", "modules")

    push!(MODULE_SEARCH_PATHS, builtin_modules)
    push!(MODULE_SEARCH_PATHS, user_modules)

    # Load builtin modules
    load_builtin_modules()
end

# ============================================================================
# JLL PACKAGE RESOLUTION
# ============================================================================

"""
    search_general_registry(package_name::String) -> Union{String, Nothing}

Search Julia General registry for a JLL package matching the given name.
Returns the exact package name if found, nothing otherwise.
"""
function search_general_registry(package_name::String)
    # Try multiple name variations
    candidates = [
        "$(package_name)_jll",           # Standard: Boost → Boost_jll
        "lib$(package_name)_jll",        # Prefixed: PNG → libpng_jll
        "Lib$(package_name)_jll",        # Capitalized: CURL → LibCURL_jll
        package_name,                     # Already correct: Boost_jll
    ]

    # Also try case variations
    titlecase_name = titlecase(package_name)

    append!(candidates, [
        "$(titlecase_name)_jll",
        "$(lowercase(package_name))_jll",
        "lib$(lowercase(package_name))_jll",
    ])

    # Remove duplicates
    unique!(candidates)

    # Check each candidate in the General registry
    for candidate in candidates
        try
            # Fast path: Check if package exists in any registry
            # Use Pkg.Registry.RegistryInstance to search
            registries = Pkg.Registry.reachable_registries()

            for reg in registries
                # Search packages in this registry
                for (uuid, pkg_entry) in reg.pkgs
                    if pkg_entry.name == candidate
                        println("  ✓ Found JLL package in registry: $candidate")
                        return candidate
                    end
                end
            end

        catch
            # Fallback: Manual registry search
            try
                registry_path = joinpath(first(DEPOT_PATH), "registries", "General")
                if isdir(registry_path)
                    # Search in registry TOML files
                    for (root, _, files) in walkdir(registry_path)
                        for file in files
                            if file == "Package.toml"
                                toml_path = joinpath(root, file)
                                pkg_data = TOML.parsefile(toml_path)
                                if get(pkg_data, "name", "") == candidate
                                    println("  ✓ Found JLL package in registry: $candidate")
                                    return candidate
                                end
                            end
                        end
                    end
                end
            catch
                continue
            end
        end
    end

    return nothing
end

"""
    resolve_jll_package(package_name::String) -> Union{ModuleInfo, Nothing}

Attempt to resolve a CMake package to a Julia JLL package.
Uses universal search - no hardcoded mappings required!
"""
function resolve_jll_package(package_name::String)
    # Quick common mappings for performance (optional fast path)
    quick_mappings = Dict(
        "ZLIB" => "Zlib_jll",
        "PNG" => "libpng_jll",
        "JPEG" => "JpegTurbo_jll",
        "CURL" => "LibCURL_jll",
        "SQLite3" => "SQLite_jll",
        "FFTW3" => "FFTW_jll",
        "Eigen3" => "Eigen_jll",
        "BLAS" => "OpenBLAS_jll",
        "LAPACK" => "OpenBLAS_jll",
    )

    # Try quick mapping first
    jll_name = get(quick_mappings, package_name, "$(package_name)_jll")

    # Try the standard name first (fastest path)
    if check_jll_installed(jll_name)
        return resolve_jll_module(jll_name, package_name)
    end

    # Search General registry for any matching JLL package
    println("  🔎 Searching Julia General registry for $package_name...")
    found_name = search_general_registry(package_name)

    if !isnothing(found_name)
        jll_name = found_name

        # Try to install if not already installed
        if !check_jll_installed(jll_name)
            try
                println("  📦 Adding JLL package: $jll_name to project...")
                Pkg.add(jll_name)
                println("  ✓ JLL package installed: $jll_name")
            catch e
                @warn "Could not install JLL package: $jll_name" exception=e
                return nothing
            end
        else
            println("  ✓ JLL package already installed: $jll_name")
        end

        return resolve_jll_module(jll_name, package_name)
    end

    @warn "No JLL package found for: $package_name (searched: $(package_name)_jll, lib$(package_name)_jll, etc.)"
    return nothing
end

"""
    check_jll_installed(jll_name::String) -> Bool

Check if a JLL package is already installed.
"""
function check_jll_installed(jll_name::String)
    try
        pkg_info = Pkg.dependencies()
        for (_, info) in pkg_info
            if info.name == jll_name
                return true
            end
        end
    catch
    end
    return false
end

"""
Extract build information from a loaded JLL package
"""
function resolve_jll_module(jll_name::String, cmake_name::String)
    # Load the JLL package
    try
        jll_mod = Base.require(Main, Symbol(jll_name))

        # Try multiple methods to get artifact directory
        artifact_dir = ""

        # Method 1: Use Artifacts.jl to query the package's Artifacts.toml
        # This is the most reliable method - directly read where the artifact is
        try
            pkg_info = Pkg.dependencies()
            jll_uuid = nothing

            for (uuid, info) in pkg_info
                if info.name == jll_name
                    jll_uuid = uuid
                    break
                end
            end

            if !isnothing(jll_uuid)
                # Find the package directory
                pkg_dir = Base.locate_package(Base.PkgId(jll_uuid, jll_name))
                if !isnothing(pkg_dir)
                    # Package source is in src/PackageName.jl, go up to package root
                    pkg_root = dirname(dirname(pkg_dir))
                    artifacts_toml = joinpath(pkg_root, "Artifacts.toml")

                    if isfile(artifacts_toml)
                        # Load and parse Artifacts.toml
                        artifacts_dict = Artifacts.load_artifacts_toml(artifacts_toml)

                        # JLL packages typically have one main artifact with the package name
                        # Try common artifact names
                        artifact_candidates = [
                            replace(jll_name, "_jll" => ""),  # Qt5Base_jll → Qt5Base
                            lowercase(replace(jll_name, "_jll" => "")),
                            jll_name,
                        ]

                        for candidate in artifact_candidates
                            if haskey(artifacts_dict, candidate)
                                # Get the artifact hash and ensure it's downloaded
                                artifact_hash = Artifacts.artifact_hash(candidate, artifacts_toml)
                                if !isnothing(artifact_hash)
                                    # Ensure artifact is installed
                                    artifact_path = Artifacts.artifact_path(artifact_hash)
                                    if isdir(artifact_path)
                                        artifact_dir = artifact_path
                                        println("  ✓ Found artifact via Artifacts.toml: $artifact_path")
                                        break
                                    end
                                end
                            end
                        end
                    end
                end
            end
        catch e
            @debug "Artifacts.toml method failed" exception=e
        end

        # Method 2: Check for artifact_dir function/constant (new JLL standard)
        if isempty(artifact_dir) && isdefined(jll_mod, :artifact_dir)
            try
                artifact_dir = string(jll_mod.artifact_dir)
            catch
                # May be a function
                try
                    artifact_dir = string(jll_mod.artifact_dir())
                catch
                end
            end
        end

        # Method 3: Check for libdir or PATH exported by the JLL
        if isempty(artifact_dir)
            # Many JLLs export a libdir_path or similar
            for name in [:libdir, :artifact_dir, :PATH_list]
                if isdefined(jll_mod, name)
                    try
                        val = getfield(jll_mod, name)
                        if isa(val, String) && isdir(val)
                            # libdir is typically artifact_root/lib
                            artifact_dir = dirname(val)
                            break
                        elseif isa(val, Ref)
                            path_val = string(val[])
                            if isdir(path_val)
                                artifact_dir = dirname(path_val)
                                break
                            end
                        elseif isa(val, Vector)
                            # PATH_list might be a vector of paths
                            for p in val
                                if isa(p, String) && isdir(p)
                                    artifact_dir = dirname(p)
                                    break
                                end
                            end
                            if !isempty(artifact_dir)
                                break
                            end
                        end
                    catch
                        continue
                    end
                end
            end
        end

        # Method 4: Look for library file paths in the module
        if isempty(artifact_dir)
            # JLL packages often export library paths like libQt5Core_path
            for name in names(jll_mod; all=true)
                name_str = string(name)
                if occursin("path", lowercase(name_str)) || endswith(name_str, "_PATH")
                    try
                        val = getfield(jll_mod, name)
                        if isa(val, String) && (endswith(val, ".so") || endswith(val, ".dylib") || endswith(val, ".dll") || endswith(val, ".a"))
                            if isfile(val)
                                # Library file found - artifact root is typically ../.. from lib/file.so
                                lib_dir = dirname(val)
                                artifact_dir = dirname(lib_dir)
                                println("  ✓ Found artifact via library path: $artifact_dir")
                                break
                            end
                        end
                    catch
                        continue
                    end
                end
            end
        end

        # Extract paths from artifact
        include_dirs = String[]
        lib_dirs = String[]
        libraries = String[]

        if !isempty(artifact_dir) && isdir(artifact_dir)
            # Standard include directory
            include_dir = joinpath(artifact_dir, "include")
            if isdir(include_dir)
                push!(include_dirs, include_dir)
            end

            # Standard lib directory
            lib_dir = joinpath(artifact_dir, "lib")
            if isdir(lib_dir)
                push!(lib_dirs, lib_dir)

                # Find libraries
                for file in readdir(lib_dir)
                    if endswith(file, ".so") || endswith(file, ".dylib") || endswith(file, ".dll") || endswith(file, ".a")
                        # Remove lib prefix and extension
                        libname = replace(file, r"^lib" => "")
                        libname = replace(libname, r"\.(so|dylib|dll|a).*" => "")
                        if !in(libname, libraries)
                            push!(libraries, libname)
                        end
                    end
                end
            end

            # Also check lib64
            lib64_dir = joinpath(artifact_dir, "lib64")
            if isdir(lib64_dir) && !in(lib64_dir, lib_dirs)
                push!(lib_dirs, lib64_dir)
            end
        end

        return ModuleInfo(
            cmake_name,
            "auto",  # Version from JLL
            :jll,
            include_dirs,
            lib_dirs,
            libraries,
            String[],  # No special compile flags
            String[],  # No special link flags
            Dict{String,String}(),
            jll_name,
            true,
            "Automatically resolved from $jll_name" * (isempty(artifact_dir) ? " (artifact path not extracted)" : ""),
            String[],
            cmake_name
        )

    catch e
        @warn "Failed to load JLL package: $jll_name" exception=e
        return nothing
    end
end

# ============================================================================
# SYSTEM LIBRARY RESOLUTION (pkg-config, FindLibrary)
# ============================================================================

"""
    resolve_system_library(name::String) -> Union{ModuleInfo, Nothing}

Try to find library using system tools (pkg-config, library search)
"""
function resolve_system_library(name::String)
    # Try pkg-config first
    try
        if success(`which pkg-config`)
            # Get compile flags
            cflags_output = read(`pkg-config --cflags $(lowercase(name))`, String)
            cflags = split(strip(cflags_output))

            # Get library flags
            libs_output = read(`pkg-config --libs $(lowercase(name))`, String)
            libs = split(strip(libs_output))

            # Parse flags
            include_dirs = String[]
            library_dirs = String[]
            libraries = String[]

            for flag in cflags
                if startswith(flag, "-I")
                    push!(include_dirs, flag[3:end])
                end
            end

            for flag in libs
                if startswith(flag, "-L")
                    push!(library_dirs, flag[3:end])
                elseif startswith(flag, "-l")
                    push!(libraries, flag[3:end])
                end
            end

            return ModuleInfo(
                name,
                "system",
                :system,
                include_dirs,
                library_dirs,
                libraries,
                String[flag for flag in cflags if startswith(flag, "-D") || startswith(flag, "-f")],
                String[flag for flag in libs if !startswith(flag, "-l") && !startswith(flag, "-L")],
                Dict{String,String}(),
                "",
                false,
                "Resolved via pkg-config",
                String[],
                name
            )
        end
    catch
        # pkg-config not available or failed, continue to library search
    end

    # Fallback: Search for library files directly
    search_paths = [
        "/usr/lib",
        "/usr/local/lib",
        "/opt/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]

    for path in search_paths
        for ext in [".so", ".dylib", ".a"]
            libfile = joinpath(path, "lib$(lowercase(name))$ext")
            if isfile(libfile)
                # Found library - try to infer include directory
                possible_includes = [
                    "/usr/include",
                    "/usr/local/include",
                    joinpath(dirname(dirname(path)), "include")
                ]

                return ModuleInfo(
                    name,
                    "system",
                    :system,
                    filter(isdir, possible_includes),
                    [path],
                    [lowercase(name)],
                    String[],
                    String[],
                    Dict{String,String}(),
                    "",
                    false,
                    "Found system library at $libfile",
                    String[],
                    name
                )
            end
        end
    end

    return nothing
end

# ============================================================================
# CUSTOM MODULE LOADING (from TOML files)
# ============================================================================

"""
    load_module_from_toml(toml_file::String) -> ModuleInfo

Load module definition from TOML file
"""
function load_module_from_toml(toml_file::String)
    data = TOML.parsefile(toml_file)

    module_data = get(data, "module", Dict())

    ModuleInfo(
        get(module_data, "name", "unknown"),
        get(module_data, "version", "unknown"),
        Symbol(get(module_data, "source", "custom")),
        get(module_data, "include_dirs", String[]),
        get(module_data, "library_dirs", String[]),
        get(module_data, "libraries", String[]),
        get(module_data, "compile_flags", String[]),
        get(module_data, "link_flags", String[]),
        get(module_data, "defines", Dict{String,String}()),
        get(module_data, "julia_package", ""),
        false,
        get(module_data, "description", ""),
        get(module_data, "components", String[]),
        get(module_data, "cmake_name", get(module_data, "name", "unknown"))
    )
end

"""
Load all builtin module definitions
"""
function load_builtin_modules()
    for search_path in MODULE_SEARCH_PATHS
        if !isdir(search_path)
            continue
        end

        for file in readdir(search_path)
            if endswith(file, ".toml")
                try
                    toml_path = joinpath(search_path, file)
                    mod_info = load_module_from_toml(toml_path)
                    REGISTRY[mod_info.name] = mod_info
                catch e
                    @warn "Failed to load module from $file" exception=e
                end
            end
        end
    end
end

# ============================================================================
# PUBLIC API
# ============================================================================

"""
    resolve_module(name::String; components::Vector{String}=String[]) -> Union{ModuleInfo, Nothing}

Resolve an external library module by name.
Tries in order: cached registry → JLL packages → system libraries → custom modules
"""
function resolve_module(name::String; components::Vector{String}=String[])
    # Check cache first
    if haskey(REGISTRY, name)
        cached = REGISTRY[name]
        # Validate components if specified
        if !isempty(components) && !isempty(cached.components)
            for comp in components
                if !(comp in cached.components)
                    @warn "Component $comp not found in $name module"
                end
            end
        end
        return cached
    end

    println("🔍 Resolving module: $name")

    # Try JLL packages
    mod_info = resolve_jll_package(name)
    if !isnothing(mod_info)
        println("  ✓ Resolved via JLL: $(mod_info.julia_package)")
        REGISTRY[name] = mod_info
        return mod_info
    end

    # Try system libraries
    mod_info = resolve_system_library(name)
    if !isnothing(mod_info)
        println("  ✓ Resolved via system")
        REGISTRY[name] = mod_info
        return mod_info
    end

    @warn "Could not resolve module: $name"
    return nothing
end

"""
    register_module(module_info::ModuleInfo)

Manually register a module in the registry
"""
function register_module(module_info::ModuleInfo)
    REGISTRY[module_info.name] = module_info
    println("✅ Registered module: $(module_info.name)")
end

"""
    list_modules() -> Vector{String}

List all available modules
"""
function list_modules()
    sort(collect(keys(REGISTRY)))
end

"""
    get_module_info(name::String) -> Union{ModuleInfo, Nothing}

Get cached module info without attempting resolution
"""
function get_module_info(name::String)
    get(REGISTRY, name, nothing)
end

"""
    export_module_config(module_info::ModuleInfo, output_file::String)

Export module information to TOML file for sharing/caching
"""
function export_module_config(module_info::ModuleInfo, output_file::String)
    data = Dict{String,Any}(
        "module" => Dict(
            "name" => module_info.name,
            "version" => module_info.version,
            "source" => string(module_info.source),
            "include_dirs" => module_info.include_dirs,
            "library_dirs" => module_info.library_dirs,
            "libraries" => module_info.libraries,
            "compile_flags" => module_info.compile_flags,
            "link_flags" => module_info.link_flags,
            "defines" => module_info.defines,
            "julia_package" => module_info.julia_package,
            "description" => module_info.description,
            "components" => module_info.components,
            "cmake_name" => module_info.cmake_name
        )
    )

    open(output_file, "w") do io
        TOML.print(io, data)
    end

    println("✅ Exported module config: $output_file")
end

end # module ModuleRegistry
