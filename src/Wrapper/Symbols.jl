# =============================================================================
# SYMBOL INFORMATION STRUCTURES
# =============================================================================

"""
    ParamInfo

Information about a function parameter.
"""
struct ParamInfo
    name::String                    # Parameter name (or generated: arg1, arg2, ...)
    cpp_type::String               # C++ type string
    julia_type::String             # Inferred Julia type
    is_const::Bool
    is_pointer::Bool
    is_reference::Bool
    default_value::Union{String,Nothing}
end

"""
    SymbolInfo

Comprehensive information about a symbol extracted from binary/headers.
"""
struct SymbolInfo
    # Basic info
    name::String                   # Symbol name (possibly mangled)
    demangled_name::String         # Demangled name (for C++)
    julia_name::String             # Valid Julia identifier

    # Type
    symbol_type::Symbol            # :function, :data, :weak, :undefined

    # Function-specific
    signature::String              # Raw signature (if available)
    return_type::String            # C++ return type
    julia_return_type::String      # Inferred Julia return type
    parameters::Vector{ParamInfo}  # Function parameters

    # Visibility
    visibility::Symbol             # :public, :private, :protected, :unknown

    # Source location (from debug symbols if available)
    source_file::String
    line_number::Int

    # Metadata
    metadata::Dict{String,Any}
end

"""
    create_symbol_info(name::String, type::Symbol, registry::TypeRegistry)

Create a SymbolInfo with basic information and inferred types.
"""
function create_symbol_info(name::String, type::Symbol, registry::TypeRegistry;
                           demangled::String="",
                           return_type::String="void",
                           params::Vector{ParamInfo}=ParamInfo[])

    # Generate Julia identifier
    julia_name = registry.language == :c ? make_c_identifier(isempty(demangled) ? name : demangled) : make_cpp_identifier(isempty(demangled) ? name : demangled)

    # Infer Julia return type
    julia_ret = infer_julia_type(registry, return_type)

    return SymbolInfo(
        name,
        isempty(demangled) ? name : demangled,
        julia_name,
        type,
        "",                    # signature
        return_type,
        julia_ret,
        params,
        :public,              # visibility
        "",                   # source_file
        0,                    # line_number
        Dict{String,Any}()    # metadata
    )
end

# =============================================================================
# WRAPPER STRATEGY: ALWAYS EXTRACT BEST, FALLBACK GRACEFULLY
# =============================================================================
# RepliBuild uses DWARF metadata (ground truth from compilation) when available,
# otherwise falls back to basic symbol extraction with conservative typing.

# =============================================================================
# SYMBOL EXTRACTION - Multi-Method
# =============================================================================

"""
    extract_symbols(binary_path::String, registry::TypeRegistry; demangle::Bool=true, method::Symbol=:nm)

Extract symbols from compiled binary using specified method.

# Methods
- `:nm` - Fast, basic symbol extraction
- `:objdump` - Detailed with debug info (TODO)
- `:all` - Try all methods and merge (TODO)

Returns vector of SymbolInfo objects.
"""
function extract_symbols(binary_path::String, registry::TypeRegistry;
                        demangle::Bool=true, method::Symbol=:nm)
    if !isfile(binary_path)
        error("Binary not found: $binary_path")
    end

    if method == :nm
        return extract_symbols_nm(binary_path, registry, demangle=demangle)
    elseif method == :objdump
        @warn "objdump extraction not yet implemented, falling back to nm"
        return extract_symbols_nm(binary_path, registry, demangle=demangle)
    elseif method == :all
        @warn "multi-method extraction not yet implemented, using nm"
        return extract_symbols_nm(binary_path, registry, demangle=demangle)
    else
        error("Unknown symbol extraction method: $method")
    end
end

"""
Extract symbols using nm command.
"""
function extract_symbols_nm(binary_path::String, registry::TypeRegistry; demangle::Bool=true)
    symbols = SymbolInfo[]

    # Use nm to extract symbols
    nm_cmd = demangle ? `nm -D --defined-only --demangle $binary_path` : `nm -D --defined-only $binary_path`

    try
        output = read(nm_cmd, String)

        for line in split(output, '\n')
            if isempty(strip(line))
                continue
            end

            parts = split(strip(line))
            if length(parts) < 3
                continue
            end

            # Parse nm output: address type name
            symbol_type_char = parts[2]
            symbol_name = join(parts[3:end], " ")  # Handle demangled names with spaces

            # Map nm symbol type to our enum
            symbol_type = if symbol_type_char in ["T", "t"]
                :function
            elseif symbol_type_char in ["D", "d", "B", "b"]
                :data
            elseif symbol_type_char in ["W", "w"]
                :weak
            elseif symbol_type_char in ["U", "u"]
                :undefined
            else
                continue  # Skip other symbol types
            end

            # Skip undefined symbols
            if symbol_type == :undefined
                continue
            end

            # Create symbol info
            # For nm, we don't have type information, so we use conservative defaults
            info = create_symbol_info(
                symbol_name,
                symbol_type,
                registry,
                demangled=symbol_name,
                return_type=(symbol_type == :function ? "void" : "char"),
                params=ParamInfo[]  # nm doesn't provide parameter info
            )

            # Skip internal/private symbols (starting with _)
            if startswith(info.julia_name, "_") || isempty(info.julia_name)
                continue
            end

            push!(symbols, info)
        end

        return symbols
    catch e
        @warn "Symbol extraction failed: $e"
        return SymbolInfo[]
    end
end

