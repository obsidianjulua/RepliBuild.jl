#!/usr/bin/env julia
# Project.jl - High-level introspection tools for RepliBuild projects
# Provides functions to easily find and analyze LTO bitcode and AOT thunks

using TOML

"""
    project_artifacts(toml_path::String="replibuild.toml")

Discover all generated artifacts (library, LTO IR, AOT thunks) for a RepliBuild project.

# Returns
Dict with paths to the various artifacts.
"""
function project_artifacts(toml_path::String="replibuild.toml")
    toml_path = abspath(toml_path)
    if !isfile(toml_path)
        error("No replibuild.toml found at: $toml_path")
    end

    project_dir = dirname(toml_path)
    data = TOML.parsefile(toml_path)
    project_name = get(get(data, "project", Dict()), "name", "unnamed")

    julia_dir = joinpath(project_dir, "julia")
    if !isdir(julia_dir)
        return Dict{Symbol,String}()
    end

    artifacts = Dict{Symbol,String}()

    # Main library
    lib_files = filter(f -> endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll"), readdir(julia_dir))
    main_libs = filter(f -> !contains(f, "_thunks"), lib_files)
    if !isempty(main_libs)
        artifacts[:library] = joinpath(julia_dir, main_libs[1])
    end

    # Wrapper
    jl_files = filter(f -> endswith(f, ".jl"), readdir(julia_dir))
    if !isempty(jl_files)
        artifacts[:wrapper] = joinpath(julia_dir, jl_files[1])
    end

    # LTO IR
    lto_bc = filter(f -> endswith(f, "_lto.bc") && !contains(f, "thunks"), readdir(julia_dir))
    if !isempty(lto_bc)
        artifacts[:lto_ir] = joinpath(julia_dir, lto_bc[1])
    end

    # AOT IR
    aot_bc = filter(f -> endswith(f, "_thunks_lto.bc"), readdir(julia_dir))
    if !isempty(aot_bc)
        artifacts[:aot_ir] = joinpath(julia_dir, aot_bc[1])
    end

    # AOT Library
    aot_libs = filter(f -> contains(f, "_thunks") && (endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll")), lib_files)
    if !isempty(aot_libs)
        artifacts[:aot_lib] = joinpath(julia_dir, aot_libs[1])
    end

    return artifacts
end

"""
    lto_ir(toml_path::String="replibuild.toml")

Get the LLVM IR for the project's monolithic LTO bitcode.
"""
function lto_ir(toml_path::String="replibuild.toml")
    artifacts = project_artifacts(toml_path)
    if !haskey(artifacts, :lto_ir)
        error("LTO bitcode not found. Ensure the project was built with LTO enabled.")
    end
    return llvm_ir(artifacts[:lto_ir])
end

"""
    aot_ir(toml_path::String="replibuild.toml")

Get the LLVM IR for the project's AOT MLIR thunks.
"""
function aot_ir(toml_path::String="replibuild.toml")
    artifacts = project_artifacts(toml_path)
    if !haskey(artifacts, :aot_ir)
        error("AOT bitcode not found. Ensure the project was built with AOT thunks enabled.")
    end
    return llvm_ir(artifacts[:aot_ir])
end

"""
    aot_symbols(toml_path::String="replibuild.toml")

List the exported symbols from the generated AOT thunks library.
"""
function aot_symbols(toml_path::String="replibuild.toml")
    artifacts = project_artifacts(toml_path)
    if !haskey(artifacts, :aot_lib)
        error("AOT library not found. Ensure the project was built with AOT thunks enabled.")
    end
    return symbols(artifacts[:aot_lib], filter=:functions)
end
