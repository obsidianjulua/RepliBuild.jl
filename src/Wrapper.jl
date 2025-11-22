#!/usr/bin/env julia
# Wrapper.jl - Julia binding generation for compiled libraries
# Uses centralized RepliBuildConfig

module Wrapper

using Dates

# Import from parent RepliBuild module
import ..ConfigurationManager: RepliBuildConfig, get_output_path, get_module_name
import ..ClangJLBridge
import ..BuildBridge

export wrap_with_clang, wrap_basic, extract_symbols

# =============================================================================
# SYMBOL EXTRACTION
# =============================================================================

"""
Extract symbols from compiled binary using nm.
Returns vector of (symbol_name, symbol_type) tuples.
"""
function extract_symbols(binary_path::String; demangle::Bool=true)
    if !isfile(binary_path)
        error("Binary not found: $binary_path")
    end

    # Use nm to extract symbols
    cmd = demangle ? `nm -D --defined-only $binary_path` : `nm -D $binary_path`

    try
        output = read(cmd, String)
        symbols = Tuple{String,String}[]

        for line in split(output, '\n')
            if isempty(strip(line))
                continue
            end

            parts = split(strip(line))
            if length(parts) >= 3
                symbol_type = parts[2]
                symbol_name = join(parts[3:end], " ")  # Handle demangled names with spaces

                # Filter for functions (T) and data (D, B)
                if symbol_type in ["T", "t", "D", "d", "B", "b"]
                    push!(symbols, (symbol_name, symbol_type))
                end
            end
        end

        return symbols
    catch e
        @warn "Symbol extraction failed: $e"
        return Tuple{String,String}[]
    end
end

# =============================================================================
# CLANG.JL WRAPPING (Type-aware)
# =============================================================================

"""
Generate Julia bindings using Clang.jl (type-aware from headers).
This is the recommended approach when headers are available.
"""
function wrap_with_clang(config::RepliBuildConfig, library_path::String, headers::Vector{String})
    println("📝 Generating bindings with Clang.jl...")

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    if isempty(headers)
        error("Headers required for Clang.jl wrapping")
    end

    # Verify headers exist
    for header in headers
        if !isfile(header)
            @warn "Header not found: $header"
        end
    end

    # Build config for ClangJLBridge
    clang_config = Dict(
        "project" => Dict("name" => config.project.name),
        "compile" => Dict("include_dirs" => config.compile.include_dirs),
        "binding" => Dict(
            "use_ccall_macro" => false,
            "add_doc_strings" => true,
            "use_julia_native_enum" => true
        )
    )

    # Generate bindings
    output_file = ClangJLBridge.generate_bindings_clangjl(clang_config, library_path, headers)

    if isnothing(output_file)
        error("Binding generation failed")
    end

    println("  ✅ Generated: $output_file")

    return output_file
end

# =============================================================================
# BASIC WRAPPING (Symbol-only)
# =============================================================================

"""
Generate basic Julia bindings from symbols only (no type information).
Use this when headers are not available.
"""
function wrap_basic(config::RepliBuildConfig, library_path::String)
    println("📝 Generating basic bindings from symbols...")

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # Extract symbols
    symbols = extract_symbols(library_path, demangle=true)

    if isempty(symbols)
        @warn "No symbols found in library"
        return nothing
    end

    println("  📊 Found $(length(symbols)) symbols")

    # Generate basic module
    module_name = get_module_name(config)
    wrapper_content = generate_basic_wrapper(config, library_path, symbols, module_name)

    # Write to file
    output_dir = get_output_path(config)
    mkpath(output_dir)
    output_file = joinpath(output_dir, "$(module_name).jl")

    write(output_file, wrapper_content)

    println("  ✅ Generated: $output_file")

    return output_file
end

"""
Generate basic Julia wrapper module content.
"""
function generate_basic_wrapper(config::RepliBuildConfig, lib_path::String,
                                symbols::Vector{Tuple{String,String}}, module_name::String)
    # Filter to functions only
    functions = [(name, type) for (name, type) in symbols if type in ["T", "t"]]

    content = """
    # Auto-generated Julia wrapper for $(config.project.name)
    # Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    # Library: $(basename(lib_path))
    #
    # Note: This is a basic symbol-based wrapper.
    # For better type safety, use Clang.jl with headers.

    module $module_name

    using Libdl

    # Library path
    const LIB_PATH = raw"$(abspath(lib_path))"
    const LIB_HANDLE = Ref{Ptr{Nothing}}(C_NULL)

    function __init__()
        LIB_HANDLE[] = Libdl.dlopen(LIB_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    end

    # Check if library loaded
    is_loaded() = LIB_HANDLE[] != C_NULL

    """

    # Generate function wrappers
    function_count = 0
    for (symbol_name, _) in functions
        # Create valid Julia identifier
        julia_name = make_valid_identifier(symbol_name)

        if isempty(julia_name) || startswith(julia_name, "_")
            continue  # Skip internal symbols
        end

        # Basic wrapper (assumes void return, no args)
        # Users will need to customize these
        content *= """

        \"\"\"
            $julia_name()

        Wrapper for C/C++ function `$symbol_name`.
        Note: Signature is placeholder - customize as needed.
        \"\"\"
        function $julia_name(args...)
            ccall((:$symbol_name, LIB_HANDLE[]), Cvoid, (), args...)
        end
        """

        function_count += 1

        if function_count >= 50  # Limit to avoid huge files
            break
        end
    end

    if function_count >= 50
        content *= """

        # ... and $(length(functions) - 50) more functions
        # Edit this file to add wrappers as needed
        """
    end

    content *= """

    end # module $module_name
    """

    return content
end

"""
Convert C/C++ symbol to valid Julia identifier.
"""
function make_valid_identifier(name::String)::String
    # Remove C++ namespace
    name = replace(name, r"^.*::" => "")

    # Remove special characters
    name = replace(name, r"[^a-zA-Z0-9_]" => "_")

    # Ensure starts with letter
    if !isempty(name) && isdigit(name[1])
        name = "_" * name
    end

    # Avoid Julia keywords
    reserved = ["begin", "end", "if", "else", "while", "for", "function",
                "return", "break", "continue", "module", "struct", "type"]

    if name in reserved
        name = name * "_"
    end

    return name
end

end # module Wrapper
