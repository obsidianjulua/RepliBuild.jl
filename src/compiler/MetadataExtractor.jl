# MetadataExtractor.jl - Symbol Extraction and Metadata Management
# Extracts function signatures, types, and metadata from compiled binaries

module MetadataExtractor

using Dates
using JSON

import ...BuildBridge
import ...ConfigurationManager: RepliBuildConfig, get_build_path, get_module_name
import ..DWARFExtractor: extract_dwarf_return_types, dwarf_type_to_julia

export extract_symbols_from_binary, extract_mangled_name, extract_compilation_metadata,
       save_compilation_metadata, cpp_to_julia_type

# =============================================================================
# SYMBOL EXTRACTION FROM BINARIES
# =============================================================================

"""
Extract symbol information from compiled binary using nm.
Returns vector of symbol dictionaries with mangled/demangled names.
"""
function extract_symbols_from_binary(binary_path::String)
    # Run nm WITHOUT demangling to get mangled names
    (mangled_output, exitcode1) = BuildBridge.execute("nm", ["-g", "--defined-only", binary_path])
    if exitcode1 != 0
        @warn "Failed to extract symbols (mangled): $mangled_output"
        return Dict{String,Any}[]
    end

    # Run nm WITH demangling (--demangle)
    (demangled_output, exitcode2) = BuildBridge.execute("nm", ["-g", "--defined-only", "--demangle", binary_path])
    if exitcode2 != 0
        @warn "Failed to extract symbols (demangled): $demangled_output"
        return Dict{String,Any}[]
    end

    symbols = Dict{String,Any}[]

    # Parse nm output (both versions)
    # Format: <address> <type> <symbol_name>
    # Example: 0000000000001234 T functionName
    mangled_lines = split(mangled_output, '\n')
    demangled_lines = split(demangled_output, '\n')

    for (mangled_line, demangled_line) in zip(mangled_lines, demangled_lines)
        mangled_line = strip(mangled_line)
        demangled_line = strip(demangled_line)

        if isempty(mangled_line) || isempty(demangled_line)
            continue
        end

        # Parse lines: <address> <type> <name>
        mangled_parts = split(mangled_line)
        demangled_parts = split(demangled_line)

        if length(mangled_parts) >= 3 && length(demangled_parts) >= 3
            symbol_type = mangled_parts[2]

            # Only include text symbols (functions): T or W
            if symbol_type == "T" || symbol_type == "W"
                mangled_name = join(mangled_parts[3:end], " ")
                demangled_name = join(demangled_parts[3:end], " ")

                push!(symbols, Dict{String,Any}(
                    "mangled" => mangled_name,
                    "demangled" => demangled_name,
                    "type" => symbol_type
                ))
            end
        end
    end

    return symbols
end

"""
Extract mangled name from nm output line given the demangled name.
This helps correlate mangled and demangled symbols.
"""
function extract_mangled_name(nm_output::String, demangled::String)::String
    # Find line containing demangled name
    for line in split(nm_output, '\n')
        if contains(line, demangled)
            # Extract mangled name (third field)
            parts = split(strip(line))
            if length(parts) >= 3
                return parts[3]
            end
        end
    end
    return ""
end

# =============================================================================
# TYPE INFERENCE FROM C++ SIGNATURES
# =============================================================================

"""
Convert C++ type to Julia type for function signatures.
Handles primitives, pointers, references, const qualifiers.
"""
function cpp_to_julia_type(cpp_type::AbstractString)::String
    # Use DWARF type mapping as base
    return dwarf_type_to_julia(cpp_type)
end

# =============================================================================
# FUNCTION SIGNATURE PARSING
# =============================================================================

"""
Parse function signatures from symbol information.
Extracts function name, parameters, return type from demangled names.
"""
function parse_function_signatures(symbols::Vector{Dict{String,Any}})::Vector{Dict{String,Any}}
    functions = Dict{String,Any}[]

    for sym in symbols
        demangled = get(sym, "demangled", "")
        mangled = get(sym, "mangled", "")

        if isempty(demangled)
            continue
        end

        func_info = Dict{String,Any}(
            "mangled_name" => mangled,
            "demangled_name" => demangled,
            "name" => extract_function_name(demangled),
            "class" => extract_class_name(demangled),
            "return_type" => infer_return_type(demangled),
            "parameters" => parse_parameters(demangled)
        )

        push!(functions, func_info)
    end

    return functions
end

"""
Extract function name from demangled C++ signature.
Example: "MyClass::doSomething(int, double)" → "doSomething"
"""
function extract_function_name(demangled::String)::String
    # Remove parameter list
    func_part = split(demangled, "(")[1]

    # Remove class prefix if exists
    if contains(func_part, "::")
        parts = split(func_part, "::")
        return String(strip(parts[end]))
    end

    return String(strip(func_part))
end

"""
Extract class name from demangled C++ signature.
Example: "MyClass::doSomething(int, double)" → "MyClass"
Returns empty string for free functions.
"""
function extract_class_name(demangled::String)::String
    # Remove parameter list
    func_part = split(demangled, "(")[1]

    # Check for class prefix
    if contains(func_part, "::")
        parts = split(func_part, "::")
        if length(parts) >= 2
            return String(strip(parts[end-1]))
        end
    end

    return ""
end

"""
Infer return type from demangled signature (heuristic).
This is imperfect - DWARF info is more accurate.
"""
function infer_return_type(demangled::String)::Dict{String,Any}
    # Placeholder: Return type inference from name is unreliable
    # Should use DWARF info instead
    return Dict{String,Any}(
        "c_type" => "unknown",
        "julia_type" => "Any",
        "size" => 0
    )
end

"""
Parse parameter list from demangled C++ signature.
Example: "func(int, double, char*)" → [{type: "int"}, {type: "double"}, {type: "char*"}]
"""
function parse_parameters(demangled::String)::Vector{Dict{String,Any}}
    params = Dict{String,Any}[]

    # Extract parameter list between parentheses
    param_match = match(r"\(([^)]*)\)", demangled)
    if isnothing(param_match)
        return params
    end

    param_str = param_match.captures[1]
    if isempty(strip(param_str)) || param_str == "void"
        return params
    end

    # Split by comma (simple approach, doesn't handle nested templates)
    param_types = split(param_str, ",")

    for (i, param_type) in enumerate(param_types)
        param_type = strip(String(param_type))
        julia_type = cpp_to_julia_type(param_type)

        push!(params, Dict{String,Any}(
            "index" => i - 1,  # 0-indexed
            "c_type" => param_type,
            "julia_type" => julia_type,
            "name" => "arg$i"  # Generic name
        ))
    end

    return params
end

# =============================================================================
# TYPE REGISTRY BUILDING
# =============================================================================

"""
Build type registry from extracted functions.
Creates mappings for all types encountered.
"""
function build_type_registry(functions::Vector{Dict{String,Any}})::Dict{String,Any}
    registry = Dict{String,Any}(
        "base_types" => Dict{String,String}(),
        "custom_types" => Dict{String,String}(),
        "function_count" => length(functions)
    )

    # Collect all types from functions
    for func in functions
        # Return type
        if haskey(func, "return_type")
            ret_type = func["return_type"]
            if haskey(ret_type, "c_type") && haskey(ret_type, "julia_type")
                c_type = ret_type["c_type"]
                julia_type = ret_type["julia_type"]
                if c_type != "unknown"
                    registry["base_types"][c_type] = julia_type
                end
            end
        end

        # Parameter types
        if haskey(func, "parameters")
            for param in func["parameters"]
                if haskey(param, "c_type") && haskey(param, "julia_type")
                    c_type = param["c_type"]
                    julia_type = param["julia_type"]
                    registry["base_types"][c_type] = julia_type
                end
            end
        end
    end

    return registry
end

# =============================================================================
# METADATA COMPILATION AND PERSISTENCE
# =============================================================================

"""
Extract all compilation metadata: symbols, types, DWARF info.
This is the comprehensive metadata used for wrapper generation.
"""
function extract_compilation_metadata(config::RepliBuildConfig, source_files::Vector{String},
                                     binary_path::String)::Dict{String,Any}
    println("Extracting compilation metadata...")

    # Extract symbols using nm
    symbols = extract_symbols_from_binary(binary_path)
    println("  Found $(length(symbols)) symbols")

    # Parse function signatures
    functions = parse_function_signatures(symbols)
    println("  Parsed $(length(functions)) functions")

    # Extract DWARF debug info (return types + enums)
    (dwarf_types, enums_dict) = extract_dwarf_return_types(binary_path)

    # Merge DWARF return types into functions
    for func in functions
        mangled = func["mangled_name"]
        if haskey(dwarf_types, mangled)
            func["return_type"] = dwarf_types[mangled]
        end
    end

    # Build type registry
    type_registry = build_type_registry(functions)

    # Compile metadata
    metadata = Dict{String,Any}(
        "project" => config.project.name,
        "module_name" => get_module_name(config),
        "generated_at" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
        "source_files" => source_files,
        "binary_path" => binary_path,
        "symbols" => symbols,
        "functions" => functions,
        "enums" => enums_dict,
        "type_registry" => type_registry,
        "function_count" => length(functions),
        "symbol_count" => length(symbols)
    )

    return metadata
end

"""
Save compilation metadata to JSON file.
Returns path to saved metadata file.
"""
function save_compilation_metadata(config::RepliBuildConfig, source_files::Vector{String},
                                   binary_path::String)::String
    metadata = extract_compilation_metadata(config, source_files, binary_path)

    # Save to build directory
    build_dir = get_build_path(config)
    metadata_path = joinpath(build_dir, "compilation_metadata.json")

    open(metadata_path, "w") do io
        JSON.print(io, metadata, 2)  # Pretty print with 2-space indent
    end

    println("Metadata saved: $metadata_path")
    return metadata_path
end

end # module MetadataExtractor
