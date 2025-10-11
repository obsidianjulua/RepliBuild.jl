#!/usr/bin/env julia
# ClangJLBridge.jl - Integration with Clang.jl for automatic Julia binding generation
# Uses Clang.jl's Generators to create type-aware Julia wrappers from C++ headers

module ClangJLBridge

using Clang.Generators
using TOML
using Dates

export generate_bindings_clangjl

"""
    generate_bindings_clangjl(config::Dict, lib_path::String, headers::Vector{String})

Generate Julia bindings using Clang.jl's generator.

# Arguments
- `config::Dict`: RepliBuild configuration dictionary (from TOML)
- `lib_path::String`: Path to compiled shared library
- `headers::Vector{String}`: List of header files to wrap

# Returns
- `String`: Path to generated Julia bindings file

# Examples
```julia
config = TOML.parsefile("replibuild.toml")
lib_path = "julia/libr_code.so"
headers = ["atom.h", "utils.h"]
bindings = generate_bindings_clangjl(config, lib_path, headers)
```
"""
function generate_bindings_clangjl(config::Dict, lib_path::String, headers::Vector{String})
    println("📝 Generating Julia bindings with Clang.jl...")

    # Extract configuration
    project = get(config, "project", Dict())
    paths = get(config, "paths", Dict())
    compile = get(config, "compile", Dict())
    binding = get(config, "binding", Dict())

    project_name = get(project, "name", "MyProject")
    project_root = get(project, "root", ".")
    output_dir = get(paths, "output", "julia")
    include_dirs = get(compile, "include_dirs", String[])
    compile_flags = get(compile, "flags", String[])

    # Clang.jl specific settings
    use_ccall_macro = get(binding, "use_ccall_macro", false)
    add_doc_strings = get(binding, "add_doc_strings", true)
    use_julia_native_enum = get(binding, "use_julia_native_enum", true)
    print_using = get(binding, "print_using", false)

    # Sanitize module name (must be valid Julia identifier)
    module_name = sanitize_module_name(project_name)

    # Output file path
    output_file = joinpath(output_dir, "$(module_name).jl")

    # Create Clang.jl options
    options = Dict(
        "general" => Dict(
            "library_name" => basename(lib_path),
            "library_names" => Dict(module_name => lib_path),
            "output_file_path" => abspath(output_file),
            "module_name" => module_name,
            "print_using" => print_using,
            "print_time" => false,
        ),
        "codegen" => Dict(
            "use_ccall_macro" => use_ccall_macro,
            "use_julia_native_enum" => use_julia_native_enum,
            "add_doc_string" => add_doc_strings,
            "always_NUL_terminated_string" => false,
        )
    )

    # Build compiler arguments
    args = get_default_args()

    # Add include directories
    for dir in include_dirs
        push!(args, "-I$dir")
    end

    # Add compile flags (but filter out incompatible ones)
    for flag in compile_flags
        # Skip flags that are C++-specific compilation options
        if !startswith(flag, "-O") && !startswith(flag, "-f") && !startswith(flag, "-W")
            push!(args, flag)
        end
    end

    # Ensure we parse as C++ if needed
    if !any(arg -> arg == "-xc++" || arg == "-std=c++", args)
        # Check if any header suggests C++
        is_cpp = any(h -> endswith(h, ".hpp") || endswith(h, ".hxx") || endswith(h, ".h++"), headers)
        if is_cpp || any(contains.(compile_flags, "c++"))
            push!(args, "-xc++")
        end
    end

    println("  📂 Processing $(length(headers)) header files...")
    println("  📁 Include dirs: $(length(include_dirs))")
    println("  🎯 Module name: $module_name")
    println("  📄 Output: $output_file")

    try
        # Create Clang.jl context
        ctx = create_context(headers, args, options)

        # Build bindings
        build!(ctx)

        # Add custom header/footer if needed
        add_replibuild_metadata(output_file, module_name, lib_path, headers)

        println("  ✅ Generated: $output_file")

        return output_file

    catch e
        @error "Failed to generate bindings with Clang.jl" exception=(e, catch_backtrace())
        println("  ❌ Error: $e")
        return nothing
    end
end

"""
    sanitize_module_name(name::String) -> String

Convert a project name into a valid Julia module identifier.

# Examples
```julia
sanitize_module_name("r_code")      # "R_code"
sanitize_module_name("my-project")  # "My_project"
sanitize_module_name("123test")     # "Mod_123test"
```
"""
function sanitize_module_name(name::String)
    if isempty(name)
        return "GeneratedModule"
    end

    # Replace invalid characters with underscore
    clean = replace(name, r"[^a-zA-Z0-9_]" => "_")

    # Ensure it starts with a letter (uppercase)
    if !isempty(clean) && isletter(clean[1])
        clean = uppercase(clean[1]) * clean[2:end]
    else
        clean = "Mod_" * clean
    end

    return clean
end

"""
    add_replibuild_metadata(file_path::String, module_name::String, lib_path::String, headers::Vector{String})

Add RepliBuild metadata comments to the generated bindings file.
"""
function add_replibuild_metadata(file_path::String, module_name::String, lib_path::String, headers::Vector{String})
    if !isfile(file_path)
        return
    end

    # Read existing content
    content = read(file_path, String)

    # Create header comment
    header = """
    # Auto-generated Julia bindings by RepliBuild + Clang.jl
    # Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    # Module: $module_name
    # Library: $(basename(lib_path))
    # Headers: $(join([basename(h) for h in headers], ", "))
    #
    # This file was generated automatically by RepliBuild using Clang.jl.
    # For more information: https://github.com/JuliaInterop/Clang.jl
    #
    # Original library: $(abspath(lib_path))
    #

    """

    # Prepend header
    write(file_path, header * content)
end

"""
    discover_headers(source_dir::String; recursive=true) -> Vector{String}

Discover header files in a directory.

# Arguments
- `source_dir::String`: Directory to search
- `recursive::Bool`: Whether to search recursively (default: true)

# Returns
- `Vector{String}`: List of absolute paths to header files
"""
function discover_headers(source_dir::String; recursive=true)
    headers = String[]

    if !isdir(source_dir)
        @warn "Source directory not found: $source_dir"
        return headers
    end

    walkfunc = recursive ? walkdir(source_dir) : [(source_dir, String[], readdir(source_dir))]

    for (root, dirs, files) in walkfunc
        # Skip common build/cache directories
        filter!(d -> !in(d, ["build", ".git", ".cache", "node_modules", "__pycache__", ".bridge_cache"]), dirs)

        for file in files
            ext = lowercase(splitext(file)[2])
            if ext in [".h", ".hpp", ".hxx", ".h++", ".hh"]
                push!(headers, abspath(joinpath(root, file)))
            end
        end
    end

    return headers
end

"""
    generate_from_config(config_file::String; lib_path::String="", headers::Vector{String}=String[])

Convenience function to generate bindings directly from RepliBuild config file.

# Arguments
- `config_file::String`: Path to replibuild.toml
- `lib_path::String`: Optional override for library path
- `headers::Vector{String}`: Optional override for header list

# Examples
```julia
generate_from_config("replibuild.toml")
generate_from_config("replibuild.toml", lib_path="custom/libfoo.so")
```
"""
function generate_from_config(config_file::String; lib_path::String="", headers::Vector{String}=String[])
    if !isfile(config_file)
        error("Config file not found: $config_file")
    end

    # Load config
    config = TOML.parsefile(config_file)

    # Determine library path
    if isempty(lib_path)
        project = get(config, "project", Dict())
        paths = get(config, "paths", Dict())
        project_name = get(project, "name", "MyProject")
        output_dir = get(paths, "output", "julia")
        lib_path = joinpath(output_dir, "lib$(lowercase(project_name)).so")
    end

    if !isfile(lib_path)
        error("Library not found: $lib_path\nCompile the project first with RepliBuild.compile()")
    end

    # Discover headers if not provided
    if isempty(headers)
        paths = get(config, "paths", Dict())
        source_dir = get(paths, "source", "src")
        include_paths = get(paths, "include", String[])

        # Search source directory
        headers = discover_headers(source_dir, recursive=false)

        # Search include directories
        for inc_dir in include_paths
            append!(headers, discover_headers(inc_dir, recursive=false))
        end

        if isempty(headers)
            @warn "No headers found. Searching recursively..."
            headers = discover_headers(source_dir, recursive=true)
        end

        println("  📂 Found $(length(headers)) headers")
    end

    if isempty(headers)
        error("No header files found. Please specify headers explicitly.")
    end

    return generate_bindings_clangjl(config, lib_path, headers)
end

end # module ClangJLBridge
