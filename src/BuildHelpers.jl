#!/usr/bin/env julia
# BuildHelpers.jl - Smart build utilities for common C/C++ patterns
# Handles config.h generation, pkg-config detection, and more

module BuildHelpers

using TOML

export detect_external_libraries, generate_config_h, detect_config_h_template
export pkg_config_exists, get_pkg_config_flags

"""
    pkg_config_exists(library::String) -> Bool

Check if a library is available via pkg-config.
"""
function pkg_config_exists(library::String)
    try
        run(pipeline(`pkg-config --exists $library`, devnull))
        return true
    catch
        return false
    end
end

"""
    get_pkg_config_flags(library::String) -> (cflags::Vector{String}, libs::Vector{String})

Get compiler and linker flags from pkg-config.
"""
function get_pkg_config_flags(library::String)
    cflags = String[]
    libs = String[]

    try
        # Get compile flags
        cflags_str = strip(read(`pkg-config --cflags $library`, String))
        if !isempty(cflags_str)
            cflags = split(cflags_str)
        end

        # Get linker flags
        libs_str = strip(read(`pkg-config --libs $library`, String))
        if !isempty(libs_str)
            libs = split(libs_str)
        end
    catch e
        @warn "Failed to get pkg-config flags for $library: $e"
    end

    return (cflags, libs)
end

"""
    detect_external_libraries(source_dir::String) -> Dict

Scan source files for #include statements and detect external libraries.
Returns a dict with detected libraries and their pkg-config names.
"""
function detect_external_libraries(source_dir::String)
    detected = Dict{String, Any}()

    # Common library headers → pkg-config names
    library_map = Dict(
        "sqlite3.h" => "sqlite3",
        "zlib.h" => "zlib",
        "png.h" => "libpng",
        "jpeglib.h" => "libjpeg",
        "curl/curl.h" => "libcurl",
        "openssl/ssl.h" => "openssl",
        "gtk/gtk.h" => "gtk+-3.0",
        "SDL2/SDL.h" => "sdl2",
        "SFML/Graphics.hpp" => "sfml-graphics",
        "boost/" => "boost",  # Partial match for boost headers
        "Eigen/" => "eigen3"
    )

    println("🔍 Scanning for external library dependencies...")

    includes = Set{String}()

    # Scan all C++ and header files
    for (root, dirs, files) in walkdir(source_dir)
        filter!(d -> !startswith(d, "."), dirs)  # Skip hidden dirs

        for file in files
            if endswith(file, ".cpp") || endswith(file, ".cc") ||
               endswith(file, ".h") || endswith(file, ".hpp")

                filepath = joinpath(root, file)
                content = try
                    read(filepath, String)
                catch
                    continue
                end

                # Extract #include statements
                for m in eachmatch(r"#include\s*[<\"]([^>\"]+)[>\"]", content)
                    push!(includes, m.captures[1])
                end
            end
        end
    end

    # Match includes to known libraries
    for include in includes
        for (header_pattern, pkg_name) in library_map
            if occursin(header_pattern, include) || startswith(include, header_pattern)
                if pkg_config_exists(pkg_name)
                    cflags, libs = get_pkg_config_flags(pkg_name)
                    detected[pkg_name] = Dict(
                        "header" => include,
                        "pkg_config" => pkg_name,
                        "cflags" => cflags,
                        "libs" => libs,
                        "available" => true
                    )
                    println("  ✓ Found: $pkg_name (via $include)")
                else
                    detected[pkg_name] = Dict(
                        "header" => include,
                        "pkg_config" => pkg_name,
                        "available" => false
                    )
                    println("  ⚠️  Detected $include but pkg-config $pkg_name not found")
                end
                break
            end
        end
    end

    if isempty(detected)
        println("  No external libraries detected")
    end

    return detected
end

"""
    detect_config_h_template(source_dir::String) -> Union{String, Nothing}

Look for config.h.in, config.h.cmake, or similar template files.
"""
function detect_config_h_template(source_dir::String)
    template_patterns = [
        "config.h.in",
        "config.h.cmake",
        "*_config.h.in",
        "config.hpp.in"
    ]

    for (root, dirs, files) in walkdir(source_dir)
        for file in files
            for pattern in template_patterns
                if contains(file, replace(pattern, "*" => ""))
                    return joinpath(root, file)
                end
            end
        end
    end

    return nothing
end

"""
    generate_config_h(template_path::String, output_path::String, defines::Dict{String,String})

Generate config.h from a template by substituting variables.
Handles both #cmakedefine and @VAR@ style templates.
"""
function generate_config_h(template_path::String, output_path::String, defines::Dict{String,String}=Dict())
    if !isfile(template_path)
        @warn "Template not found: $template_path"
        return false
    end

    println("📝 Generating config.h from template...")
    println("   Template: $template_path")
    println("   Output:   $output_path")

    template = read(template_path, String)
    output = template

    # Auto-detect common defines if not provided
    auto_defines = Dict{String,String}(
        "PROJECT_VERSION" => get(defines, "PROJECT_VERSION", "0.1.0"),
        "PROJECT_NAME" => get(defines, "PROJECT_NAME", basename(dirname(template_path))),
        "VERSION_MAJOR" => get(defines, "VERSION_MAJOR", "0"),
        "VERSION_MINOR" => get(defines, "VERSION_MINOR", "1"),
        "VERSION_PATCH" => get(defines, "VERSION_PATCH", "0"),
        "BUILD_TYPE" => get(defines, "BUILD_TYPE", "Release"),
        "CMAKE_SYSTEM_NAME" => get(defines, "CMAKE_SYSTEM_NAME", string(Sys.KERNEL)),
        "CMAKE_SIZEOF_VOID_P" => get(defines, "CMAKE_SIZEOF_VOID_P", string(sizeof(Ptr{Cvoid})))
    )

    # Merge user defines with auto-detected ones
    all_defines = merge(auto_defines, defines)

    # Process #cmakedefine statements (match entire line including trailing content)
    output = replace(output, r"#cmakedefine\s+(\w+).*$"m => s -> begin
        var = match(r"#cmakedefine\s+(\w+)", s).captures[1]
        if haskey(all_defines, var) && all_defines[var] != "0" && all_defines[var] != "OFF"
            "#define $var"
        else
            "/* #undef $var */"
        end
    end)

    # Process @VAR@ substitutions
    for (key, value) in all_defines
        output = replace(output, "@$key@" => value)
    end

    # Process ${VAR} substitutions
    for (key, value) in all_defines
        output = replace(output, "\${$key}" => value)
    end

    # Write output
    mkpath(dirname(output_path))
    write(output_path, output)

    println("  ✓ Generated config.h with $(length(all_defines)) defines")
    return true
end

"""
    auto_detect_build_requirements(project_root::String) -> Dict

Automatically detect build requirements and suggest configuration.
"""
function auto_detect_build_requirements(project_root::String)
    requirements = Dict{String,Any}()

    source_dir = joinpath(project_root, "src")
    if !isdir(source_dir)
        source_dir = project_root
    end

    # Detect external libraries
    requirements["external_libs"] = detect_external_libraries(source_dir)

    # Detect config.h template
    config_template = detect_config_h_template(project_root)
    if !isnothing(config_template)
        requirements["config_template"] = config_template
        println("  ✓ Found config template: $(basename(config_template))")
    end

    # Detect if project needs executable or library
    has_main = false
    for (root, dirs, files) in walkdir(source_dir)
        for file in files
            if endswith(file, ".cpp") || endswith(file, ".cc")
                content = try
                    read(joinpath(root, file), String)
                catch
                    continue
                end

                if contains(content, r"\bmain\s*\(")
                    has_main = true
                    requirements["entry_point"] = joinpath(root, file)
                    break
                end
            end
        end
        has_main && break
    end

    requirements["build_type"] = has_main ? "executable" : "library"
    println("  ✓ Detected build type: $(requirements["build_type"])")

    return requirements
end

end # module BuildHelpers
