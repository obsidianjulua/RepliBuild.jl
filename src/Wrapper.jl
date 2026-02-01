#!/usr/bin/env julia
# Wrapper.jl - Enterprise-grade Julia binding generation for compiled libraries
# Three-tier wrapping: Basic (symbol-only) → Advanced (header-aware) → Introspective (metadata-rich)

module Wrapper

using Dates
using JSON

# Import from parent RepliBuild module
import ..ConfigurationManager: RepliBuildConfig, get_output_path, get_module_name
import ..ClangJLBridge
import ..BuildBridge

export wrap_library, wrap_basic, extract_symbols
export TypeRegistry, SymbolInfo, ParamInfo
export TypeStrictness, STRICT, WARN, PERMISSIVE
export is_struct_like, is_enum_like, is_function_pointer_like

# =============================================================================
# TYPE SYSTEM - Comprehensive C/C++ to Julia Type Mapping
# =============================================================================

"""
    TypeStrictness

Controls how unknown/unmapped types are handled during FFI generation.

- `STRICT`: Error on any unmapped type (recommended for production)
- `WARN`: Warn and fallback to Any (useful for debugging)
- `PERMISSIVE`: Silent Any fallback (legacy compatibility)
"""
@enum TypeStrictness begin
    STRICT = 1
    WARN = 2
    PERMISSIVE = 3
end

"""
    TypeRegistry

Comprehensive registry for C/C++ → Julia type mappings.
Supports primitives, STL types, pointers, references, templates, and custom types.
"""
struct TypeRegistry
    # Core type mappings
    base_types::Dict{String,String}      # C/C++ primitive → Julia type
    stl_types::Dict{String,String}       # C++ STL → Julia type
    custom_types::Dict{String,String}    # User-defined types

    # Advanced rules
    pointer_suffix::String               # Default: "Ptr"
    reference_suffix::String             # Default: "Ref"
    const_handling::Symbol               # :strip, :preserve

    # Metadata
    compilation_metadata::Union{Nothing,Dict}

    # NEW: Type validation settings
    strictness::TypeStrictness           # How to handle unknown types
    allow_unknown_structs::Bool          # Treat unknown types as opaque structs
    allow_unknown_enums::Bool            # Treat unknown enums as Cint
    allow_function_pointers::Bool        # Map function pointers to Ptr{Cvoid}
end

"""
    create_type_registry(config::RepliBuildConfig; custom_types::Dict{String,String}=Dict{String,String}(),
                        strictness::Union{TypeStrictness,Nothing}=nothing,
                        allow_unknown_structs::Union{Bool,Nothing}=nothing,
                        allow_unknown_enums::Union{Bool,Nothing}=nothing,
                        allow_function_pointers::Union{Bool,Nothing}=nothing)

Create a TypeRegistry with comprehensive default mappings plus custom overrides.

# Arguments
- `config`: RepliBuild configuration (reads from config.types if parameters not provided)
- `custom_types`: Additional user-defined type mappings (merged with config.types.custom_mappings)
- `strictness`: Override - how to handle unknown types (defaults to config.types.strictness)
- `allow_unknown_structs`: Override - treat unknown types as opaque structs (defaults to config.types)
- `allow_unknown_enums`: Override - treat unknown enums as Cint (defaults to config.types)
- `allow_function_pointers`: Override - map function pointers to Ptr{Cvoid} (defaults to config.types)
"""
function create_type_registry(config::RepliBuildConfig;
                              custom_types::Dict{String,String}=Dict{String,String}(),
                              strictness::Union{TypeStrictness,Nothing}=nothing,
                              allow_unknown_structs::Union{Bool,Nothing}=nothing,
                              allow_unknown_enums::Union{Bool,Nothing}=nothing,
                              allow_function_pointers::Union{Bool,Nothing}=nothing)
    # Read from config.types, allow function parameters to override
    types_cfg = config.types

    # Convert Symbol to TypeStrictness
    final_strictness = if !isnothing(strictness)
        strictness
    elseif types_cfg.strictness == :strict
        STRICT
    elseif types_cfg.strictness == :permissive
        PERMISSIVE
    else
        WARN
    end

    # Use config values or provided overrides
    final_allow_structs = isnothing(allow_unknown_structs) ? types_cfg.allow_unknown_structs : allow_unknown_structs
    final_allow_enums = isnothing(allow_unknown_enums) ? types_cfg.allow_unknown_enums : allow_unknown_enums
    final_allow_fptrs = isnothing(allow_function_pointers) ? types_cfg.allow_function_pointers : allow_function_pointers

    # Merge custom types from config and parameters
    merged_custom = merge(types_cfg.custom_mappings, custom_types)

    # Base C types
    base_types = Dict{String,String}(
        # Void
        "void" => "Cvoid",

        # Boolean
        "bool" => "Bool",
        "_Bool" => "Bool",

        # Character types
        "char" => "Cchar",
        "signed char" => "Cchar",
        "unsigned char" => "Cuchar",
        "wchar_t" => "Cwchar_t",
        "char16_t" => "UInt16",
        "char32_t" => "UInt32",

        # Integer types
        "short" => "Cshort",
        "short int" => "Cshort",
        "signed short" => "Cshort",
        "unsigned short" => "Cushort",
        "int" => "Cint",
        "signed int" => "Cint",
        "unsigned int" => "Cuint",
        "unsigned" => "Cuint",
        "long" => "Clong",
        "long int" => "Clong",
        "signed long" => "Clong",
        "unsigned long" => "Culong",
        "long long" => "Clonglong",
        "long long int" => "Clonglong",
        "signed long long" => "Clonglong",
        "unsigned long long" => "Culonglong",

        # Fixed-width integer types
        "int8_t" => "Int8",
        "uint8_t" => "UInt8",
        "int16_t" => "Int16",
        "uint16_t" => "UInt16",
        "int32_t" => "Int32",
        "uint32_t" => "UInt32",
        "int64_t" => "Int64",
        "uint64_t" => "UInt64",

        # Floating point
        "float" => "Cfloat",
        "double" => "Cdouble",
        "long double" => "Float64",  # Julia doesn't have long double

        # Size/pointer types
        "size_t" => "Csize_t",
        "ssize_t" => "Cssize_t",
        "ptrdiff_t" => "Cptrdiff_t",
        "intptr_t" => "Cintptr_t",
        "uintptr_t" => "Cuintptr_t",
        "off_t" => "Int64",
        "time_t" => "Int64",
        "clock_t" => "Int64",
    )

    # C++ STL types
    stl_types = Dict{String,String}(
        # String types
        "std::string" => "String",
        "std::basic_string<char>" => "String",
        "std::string_view" => "String",
        "std::basic_string_view<char>" => "String",

        # Containers (conservative mappings - actual element types need parsing)
        "std::vector" => "Vector",
        "std::array" => "Vector",
        "std::deque" => "Vector",
        "std::list" => "Vector",
        "std::forward_list" => "Vector",

        # Associative containers
        "std::map" => "Dict",
        "std::unordered_map" => "Dict",
        "std::multimap" => "Dict",
        "std::set" => "Set",
        "std::unordered_set" => "Set",
        "std::multiset" => "Set",

        # Utility types
        "std::pair" => "Tuple",
        "std::tuple" => "Tuple",
        "std::optional" => "Union{Nothing,T} where T",
        "std::unique_ptr" => "Ptr",
        "std::shared_ptr" => "Ptr",
        "std::weak_ptr" => "Ptr",
    )

    return TypeRegistry(
        base_types,
        stl_types,
        merged_custom,              # custom_types (merged from config and params)
        "Ptr",                      # pointer_suffix
        "Ref",                      # reference_suffix
        :strip,                     # const_handling
        nothing,                    # compilation_metadata (TODO: load from config)
        final_strictness,           # strictness
        final_allow_structs,        # allow_unknown_structs
        final_allow_enums,          # allow_unknown_enums
        final_allow_fptrs           # allow_function_pointers
    )
end

# =============================================================================
# TYPE HEURISTICS - Smart detection of type categories
# =============================================================================

"""
    is_struct_like(cpp_type::String)::Bool

Check if a C++ type looks like a struct/class based on naming conventions.
Structs typically: start with uppercase, contain only alphanumeric + underscore.
"""
function is_struct_like(cpp_type::String)::Bool
    # Remove pointers, references, and whitespace
    base = strip(replace(replace(cpp_type, "*" => ""), "&" => ""))

    # Empty or primitive types are not structs
    if isempty(base) || base == "void"
        return false
    end

    # Struct names typically match: UpperCamelCase or snake_case with uppercase first letter
    # Examples: Matrix3x3, Grid, ComplexType, My_Struct
    return occursin(r"^[A-Z][A-Za-z0-9_]*$", base)
end

"""
    is_enum_like(cpp_type::String)::Bool

Check if a C++ type looks like an enum (similar heuristics to struct for now).
Future: enhance with DWARF metadata lookup.
"""
function is_enum_like(cpp_type::String)::Bool
    # For now, use same heuristics as struct
    # In future: check against extracted enum names from DWARF
    return is_struct_like(cpp_type)
end

"""
    is_function_pointer_like(cpp_type::String)::Bool

Check if a C++ type contains function pointer syntax.
Looks for patterns: (*name), (*)(args), (^name) (blocks)
"""
function is_function_pointer_like(cpp_type::String)::Bool
    # Function pointer patterns:
    # - int (*callback)(double, double)
    # - void (*cleanup)()
    # - typedef int (*IntCallback)(double, double)
    return occursin(r"\(\s*\*", cpp_type) || occursin(r"\(\s*\^", cpp_type)
end

# =============================================================================
# UNKNOWN TYPE HANDLING - Smart fallbacks with validation
# =============================================================================

"""
    handle_unknown_type(registry::TypeRegistry, cpp_type::String, context::String)::String

Handle unknown/unmapped C++ types based on registry strictness settings.

# Behavior
- `STRICT` mode: Error with helpful suggestions
- `WARN` mode: Warn and fallback to Any
- `PERMISSIVE` mode: Silent Any fallback

# Smart fallbacks (when enabled)
- Struct-like types → Use C++ name as opaque struct
- Enum-like types → Map to Cint
- Function pointers → Map to Ptr{Cvoid}
"""
function handle_unknown_type(registry::TypeRegistry, cpp_type::String, context::String)::String
    # Check if it looks like a struct
    if is_struct_like(cpp_type)
        if registry.allow_unknown_structs
            if registry.strictness == WARN
                @warn "Treating unknown type '$cpp_type' as opaque struct in $context"
            end
            # Remove pointer/reference markers for struct name
            base_type = strip(replace(replace(cpp_type, "*" => ""), "&" => ""))
            
            # Sanitize for Julia (e.g. Box<int> -> Box_int)
            sanitized = replace(replace(replace(base_type, "<" => "_"), ">" => ""), "," => "_")
            sanitized = replace(sanitized, " " => "")
            return sanitized
        end
    end

    # Check if it looks like an enum
    if is_enum_like(cpp_type)
        if registry.allow_unknown_enums
            if registry.strictness == WARN
                @warn "Treating enum '$cpp_type' as Cint in $context"
            end
            return "Cint"
        end
    end

    # Check if it's a function pointer
    if is_function_pointer_like(cpp_type)
        if registry.allow_function_pointers
            if registry.strictness == WARN
                @warn "Treating function pointer '$cpp_type' as Ptr{Cvoid} in $context"
            end
            return "Ptr{Cvoid}"
        end
    end

    # Apply strictness policy
    if registry.strictness == STRICT
        error("""
        ═══════════════════════════════════════════════════════════════
        FFI Type Mapping Error: Unknown C/C++ Type
        ═══════════════════════════════════════════════════════════════

        Unknown C/C++ type: '$cpp_type'
        Context: $context

        This type could not be mapped to a Julia type.

        ───────────────────────────────────────────────────────────────
        Suggestions:
        ───────────────────────────────────────────────────────────────

        1. Add custom type mapping in your code:
           registry = create_type_registry(config,
               custom_types=Dict("$cpp_type" => "YourJuliaType"))

        2. If this is a struct/class:
           - Ensure it's defined in your headers with debug info (-g flag)
           - Check that DWARF metadata extraction is working
           - Consider allowing unknown structs: allow_unknown_structs=true

        3. If this is an enum:
           - Verify enum extraction from DWARF is working
           - Enable enum fallback: allow_unknown_enums=true

        4. If this is a function pointer:
           - Enable function pointer support: allow_function_pointers=true
           - This will map to Ptr{Cvoid}

        5. If this is a template type:
           - Add template mapping for this specific instantiation
           - Example: "std::vector<$cpp_type>" => "Vector{JuliaType}"

        ───────────────────────────────────────────────────────────────
        Temporary Workaround:
        ───────────────────────────────────────────────────────────────

        To allow unmapped types temporarily (not recommended):

           registry = create_type_registry(config, strictness=WARN)

        Or for complete permissiveness (legacy mode):

           registry = create_type_registry(config, strictness=PERMISSIVE)

        ═══════════════════════════════════════════════════════════════
        """)
    elseif registry.strictness == WARN
        @warn """Unknown C/C++ type '$cpp_type' in $context, falling back to Any.
        This may cause runtime type errors. Consider adding a custom type mapping."""
        return "Any"
    else  # PERMISSIVE
        return "Any"
    end
end

"""
    infer_julia_type(registry::TypeRegistry, cpp_type::String; context::String="")::String

Infer Julia type from C/C++ type string using comprehensive rules.

# Type Inference Algorithm
1. Strip whitespace and qualifiers (const, volatile)
2. Check exact match in base_types
3. Check exact match in STL types
4. Parse pointer types (T* → Ptr{T})
5. Parse reference types (T& → Ref{T})
6. Parse const pointer types (const T* → Ptr{T})
7. Parse array types (T[N] → NTuple{N,T})
8. Parse template types (std::vector<T> → Vector{T})
9. Smart fallback via handle_unknown_type() based on strictness

# Arguments
- `registry`: TypeRegistry with type mappings and validation settings
- `cpp_type`: C/C++ type string to map
- `context`: Optional context string for error messages (e.g., "parameter 1 of function foo")

# Examples
```julia
infer_julia_type(reg, "int") # => "Cint"
infer_julia_type(reg, "const char*") # => "Cstring"
infer_julia_type(reg, "double*") # => "Ptr{Cdouble}"
infer_julia_type(reg, "std::string") # => "String"
infer_julia_type(reg, "std::vector<int>") # => "Vector{Cint}"
infer_julia_type(reg, "Matrix3x3", context="parameter 1") # => "Matrix3x3" or error in STRICT mode
```
"""
function infer_julia_type(registry::TypeRegistry, cpp_type::String; context::String="")::String
    if isempty(cpp_type)
        return handle_unknown_type(registry, "", context == "" ? "empty type string" : context)
    end

    # Clean up the type string
    clean_type = strip(cpp_type)

    # Strip const/volatile qualifiers (but remember them)
    is_const = contains(clean_type, "const")
    clean_type = replace(clean_type, r"\bconst\b" => "")
    clean_type = replace(clean_type, r"\bvolatile\b" => "")
    clean_type = strip(clean_type)

    # Handle special case: const char* and char* are Cstring
    if clean_type == "char*" || clean_type == "char *"
        return "Cstring"
    end

    # Check exact match in base types
    if haskey(registry.base_types, clean_type)
        return registry.base_types[clean_type]
    end

    # Check exact match in STL types
    if haskey(registry.stl_types, clean_type)
        return registry.stl_types[clean_type]
    end

    # Check custom types
    if haskey(registry.custom_types, clean_type)
        return registry.custom_types[clean_type]
    end

    # Parse pointer types: T* or T *
    if endswith(clean_type, "*")
        base_type = strip(replace(clean_type, r"\*$" => ""))

        # Special cases
        if base_type == "void"
            return "Ptr{Cvoid}"
        elseif base_type == "char"
            return "Cstring"
        end

        # Recursive inference for base type
        ptr_context = context == "" ? "pointer base type" : "$context (pointer to $base_type)"
        julia_base = infer_julia_type(registry, String(base_type); context=ptr_context)
        return "Ptr{$julia_base}"
    end

    # Parse reference types: T& or T &
    if endswith(clean_type, "&")
        base_type = strip(replace(clean_type, r"&$" => ""))
        ref_context = context == "" ? "reference base type" : "$context (reference to $base_type)"
        julia_base = infer_julia_type(registry, String(base_type); context=ref_context)
        return "Ref{$julia_base}"
    end

    # Parse array types: T[N]
    array_match = match(r"^(.+)\[(\d+)\]$", clean_type)
    if !isnothing(array_match)
        elem_type = strip(array_match.captures[1])
        size = parse(Int, array_match.captures[2])
        arr_context = context == "" ? "array element type" : "$context (array of $elem_type)"
        julia_elem = infer_julia_type(registry, String(elem_type); context=arr_context)
        return "NTuple{$size,$julia_elem}"
    end

    # Parse template types: std::vector<T>, std::map<K,V>, etc
    template_match = match(r"^([^<]+)<(.+)>$", clean_type)
    if !isnothing(template_match)
        template_name = strip(template_match.captures[1])
        template_args = strip(template_match.captures[2])

        # Handle std::vector<T> → Vector{T}
        if template_name == "std::vector"
            vec_context = context == "" ? "std::vector element type" : "$context (std::vector element)"
            elem_type = infer_julia_type(registry, String(template_args); context=vec_context)
            return "Vector{$elem_type}"
        end

        # Handle std::array<T, N> → Vector{T} (size lost in translation)
        if template_name == "std::array"
            # Parse "T, N"
            parts = split(template_args, ",")
            if !isempty(parts)
                arr_context = context == "" ? "std::array element type" : "$context (std::array element)"
                elem_type = infer_julia_type(registry, String(strip(parts[1])); context=arr_context)
                return "Vector{$elem_type}"
            end
        end

        # Handle std::pair<T1, T2> → Tuple{T1, T2}
        if template_name == "std::pair"
            parts = split(template_args, ",", limit=2)
            if length(parts) == 2
                pair_ctx1 = context == "" ? "std::pair first type" : "$context (std::pair first)"
                pair_ctx2 = context == "" ? "std::pair second type" : "$context (std::pair second)"
                t1 = infer_julia_type(registry, String(strip(parts[1])); context=pair_ctx1)
                t2 = infer_julia_type(registry, String(strip(parts[2])); context=pair_ctx2)
                return "Tuple{$t1,$t2}"
            end
        end

        # Handle std::map<K,V> → Dict{K,V}
        if template_name == "std::map" || template_name == "std::unordered_map"
            parts = split(template_args, ",", limit=2)
            if length(parts) == 2
                map_ctx_k = context == "" ? "std::map key type" : "$context (std::map key)"
                map_ctx_v = context == "" ? "std::map value type" : "$context (std::map value)"
                k = infer_julia_type(registry, String(strip(parts[1])); context=map_ctx_k)
                v = infer_julia_type(registry, String(strip(parts[2])); context=map_ctx_v)
                return "Dict{$k,$v}"
            end
        end

        # Generic template fallback - unknown template type
        ctx = context == "" ? "unknown template type '$clean_type'" : "$context (unknown template '$clean_type')"
        return handle_unknown_type(registry, String(clean_type), ctx)
    end

    # Unknown type - no matching rules
    ctx = context == "" ? "type inference" : context
    return handle_unknown_type(registry, String(clean_type), ctx)
end

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
    julia_name = make_julia_identifier(isempty(demangled) ? name : demangled)

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

# =============================================================================
# JULIA IDENTIFIER GENERATION
# =============================================================================

"""
    make_julia_identifier(name::String)::String

Convert C/C++ symbol to valid Julia identifier.

# Rules
- Remove C++ namespaces (Foo::bar → bar)
- Replace invalid characters with underscore
- Ensure starts with letter or underscore
- Avoid Julia keywords
- Handle operator overloads gracefully
"""
function make_julia_identifier(name::String)::String
    if isempty(name)
        return ""
    end

    # Remove C++ namespace
    clean = replace(name, r"^.*::" => "")

    # Handle C++ operators
    clean = replace(clean, "operator+" => "op_add")
    clean = replace(clean, "operator-" => "op_sub")
    clean = replace(clean, "operator*" => "op_mul")
    clean = replace(clean, "operator/" => "op_div")
    clean = replace(clean, "operator==" => "op_eq")
    clean = replace(clean, "operator!=" => "op_neq")
    clean = replace(clean, "operator<" => "op_lt")
    clean = replace(clean, "operator>" => "op_gt")
    clean = replace(clean, "operator[]" => "op_getindex")
    clean = replace(clean, "operator()" => "op_call")

    # Replace invalid characters with underscore
    clean = replace(clean, r"[^a-zA-Z0-9_!]" => "_")

    # Remove consecutive underscores
    clean = replace(clean, r"_{2,}" => "_")

    # Ensure starts with letter or underscore
    if !isempty(clean) && isdigit(clean[1])
        clean = "_" * clean
    end

    # Avoid Julia keywords
    julia_keywords = [
        "begin", "end", "if", "else", "elseif", "while", "for", "function",
        "return", "break", "continue", "module", "using", "import", "export",
        "struct", "mutable", "abstract", "type", "const", "global", "local",
        "let", "do", "try", "catch", "finally", "macro", "quote", "true",
        "false", "nothing", "missing", "NaN", "Inf"
    ]

    if lowercase(clean) in julia_keywords
        clean = clean * "_"
    end

    return clean
end

# =============================================================================
# HIGH-LEVEL WRAPPER API
# =============================================================================

"""
    wrap_library(config::RepliBuildConfig, library_path::String;
                 headers::Vector{String}=String[],
                 generate_tests::Bool=false,
                 generate_docs::Bool=true)

Generate Julia wrapper for compiled library.

Always uses introspective (DWARF metadata) wrapping when metadata is available,
otherwise falls back to basic symbol-only extraction with conservative types.

# Arguments
- `config`: RepliBuildConfig with wrapper settings
- `library_path`: Path to compiled library (.so, .dylib, .dll)
- `headers`: Optional header files (currently unused, reserved for future)
- `generate_tests`: Generate test file (default: false, TODO)
- `generate_docs`: Include comprehensive documentation (default: true)

# Returns
Path to generated Julia wrapper file
"""
function wrap_library(config::RepliBuildConfig, library_path::String;
                     headers::Vector{String}=String[],
                     generate_tests::Bool=false,
                     generate_docs::Bool=true)

    println(" RepliBuild Wrapper Generator")
    println("="^70)
    println("   Library: $(basename(library_path))")

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # Check for metadata (DWARF + symbol info from compilation)
    metadata_file = joinpath(dirname(library_path), "compilation_metadata.json")
    has_metadata = isfile(metadata_file)

    if !has_metadata
        @warn "No compilation metadata found. Did you compile with -g flag?"
        @warn "Falling back to basic symbol-only wrapper (conservative types, limited safety)"
        println("   Method: Basic symbol extraction")
        println()
        return wrap_basic(config, library_path, generate_docs=generate_docs)
    end

    # Use introspective wrapper (DWARF metadata - ground truth from compilation)
    println("   Method: Introspective (DWARF metadata)")
    println()
    return wrap_introspective(config, library_path, headers, generate_docs=generate_docs)
end

# =============================================================================
# TIER 1: BASIC WRAPPER (Symbol-Only)
# =============================================================================

"""
    wrap_basic(config::RepliBuildConfig, library_path::String; generate_docs::Bool=true)

Generate basic Julia wrapper from binary symbols only (no headers required).

Quality: ~40% - Conservative types, placeholder signatures, requires manual refinement.
Use when: Headers not available, quick prototyping, binary-only distribution.
"""
function wrap_basic(config::RepliBuildConfig, library_path::String; generate_docs::Bool=true)
    println("Generating basic wrapper (symbol-only extraction)...")
    println("Method: Symbol extraction (nm)")
    println("Type safety:   Conservative placeholders")
    println()

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # Create type registry
    registry = create_type_registry(config)

    # Extract symbols
    println("   Extracting symbols...")
    symbols = extract_symbols(library_path, registry, demangle=true, method=:nm)

    if isempty(symbols)
        @warn "No symbols found in library"
        return nothing
    end

    # Filter functions and data
    functions = filter(s -> s.symbol_type == :function, symbols)
    data_symbols = filter(s -> s.symbol_type == :data, symbols)

    println("  ✓ Found $(length(functions)) functions, $(length(data_symbols)) data symbols")
    println()

    # Generate wrapper module
    module_name = get_module_name(config)
    wrapper_content = generate_basic_module(config, library_path, functions, data_symbols,
                                           module_name, registry, generate_docs)

    # Write to file
    output_dir = get_output_path(config)
    mkpath(output_dir)
    output_file = joinpath(output_dir, "$(module_name).jl")

    write(output_file, wrapper_content)

    println("   Generated: $output_file")
    println("   Functions wrapped: $(min(length(functions), 50))")

    if length(functions) > 50
        println("    Limited to first 50 functions ($(length(functions) - 50) omitted)")
        println("     Consider using --tier=advanced with headers for complete wrapping")
    end

    println()
    return output_file
end

"""
Generate basic wrapper module content.
"""
function generate_basic_module(config::RepliBuildConfig, lib_path::String,
                               functions::Vector{SymbolInfo}, data_symbols::Vector{SymbolInfo},
                               module_name::String, registry::TypeRegistry, generate_docs::Bool)

    # Header with metadata
    header = """
    # Auto-generated Julia wrapper for $(config.project.name)
    # Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    # Generator: RepliBuild Wrapper (Basic: Symbol extraction)
    # Library: $(basename(lib_path))
    #
    #   TYPE SAFETY: BASIC (~40%)
    # This wrapper uses conservative type placeholders extracted from binary symbols.
    # For better type safety, recompile with -g flag to enable DWARF metadata extraction.

    """

    content = header

    # Module declaration
    content *= "module $module_name\n\n"
    content *= "using Libdl\n\n"

    # Library management
    content *= """
    # =============================================================================
    # LIBRARY MANAGEMENT
    # =============================================================================

    const _LIB_PATH = raw"$(abspath(lib_path))"
    const _LIB = Ref{Ptr{Nothing}}(C_NULL)
    const _LOAD_ERRORS = String[]

    function __init__()
        try
            _LIB[] = Libdl.dlopen(_LIB_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
        catch e
            push!(_LOAD_ERRORS, string(e))
            @error "Failed to load library $(basename(lib_path))" exception=e
        end
    end

    \"""
        is_loaded()

    Check if the library is successfully loaded.
    \"""
    is_loaded() = _LIB[] != C_NULL

    \"""
        get_load_errors()

    Get any errors that occurred during library loading.
    \"""
    get_load_errors() = copy(_LOAD_ERRORS)

    \"""
        get_lib_path()

    Get the path to the underlying library.
    \"""
    get_lib_path() = _LIB_PATH

    # Safety check macro
    macro check_loaded()
        quote
            if !is_loaded()
                error("Library not loaded. Errors: ", join(get_load_errors(), "; "))
            end
        end
    end

    """

    # Function wrappers
    content *= """
    # =============================================================================
    # FUNCTION WRAPPERS
    # =============================================================================

    """

    function_count = 0
    exports = String["is_loaded", "get_load_errors", "get_lib_path"]

    for func in functions
        if function_count >= 50
            break  # Limit to avoid huge files
        end

        func_wrapper = generate_basic_function_wrapper(func, registry, generate_docs)
        if !isnothing(func_wrapper)
            content *= func_wrapper * "\n"
            push!(exports, func.julia_name)
            function_count += 1
        end
    end

    if length(functions) > 50
        content *= """
        # ... and $(length(functions) - 50) more functions omitted
        # Regenerate with headers for complete wrapping:
        #   RepliBuild.wrap("$lib_path", headers=["your_header.h"])

        """
    end

    # Library info function
    content *= """
    # =============================================================================
    # METADATA
    # =============================================================================

    \"""
        library_info()

    Get information about the wrapped library.
    \"""
    function library_info()
        return Dict{Symbol,Any}(
            :name => "$(config.project.name)",
            :path => _LIB_PATH,
            :loaded => is_loaded(),
            :tier => :basic,
            :type_safety => "40% (conservative placeholders)",
            :functions_wrapped => $function_count,
            :functions_total => $(length(functions)),
            :data_symbols => $(length(data_symbols))
        )
    end

    """

    push!(exports, "library_info")

    # Exports
    content *= "# Exports\n"
    content *= "export " * join(unique(exports), ", ") * "\n\n"

    content *= "end # module $module_name\n"

    return content
end

"""
Generate wrapper for a single function (basic tier).
"""
function generate_basic_function_wrapper(func::SymbolInfo, registry::TypeRegistry, generate_docs::Bool)
    if isempty(func.julia_name)
        return nothing
    end

    # For basic tier, we use conservative Any types since we don't have parameter info
    wrapper = ""

    if generate_docs
        wrapper *= """
        \"""
            $(func.julia_name)(args...)

        Wrapper for C/C++ function `$(func.demangled_name)`.

        # Type Safety:   BASIC
        Signature uses placeholder types. Actual types unknown without headers.
        Return type and parameters may need manual adjustment.

        # C/C++ Symbol
        `$(func.name)`
        \"""
        """
    end

    wrapper *= """
    function $(func.julia_name)(args...)
        @check_loaded()
        ccall((:$(func.name), _LIB[]), Any, (), args...)
    end

    """

    return wrapper
end

# =============================================================================
# TIER 2: ADVANCED WRAPPER (Header-Aware via Clang.jl)
# =============================================================================

"""
    wrap_with_clang(config::RepliBuildConfig, library_path::String, headers::Vector{String}; generate_docs::Bool=true)

Generate advanced Julia wrapper using Clang.jl for type-aware binding generation.

Quality: ~85% - Accurate types from headers, production-ready with minor tweaks.
Use when: Headers available, need type safety, production deployment.
"""
function wrap_with_clang(config::RepliBuildConfig, library_path::String, headers::Vector{String};
                        generate_docs::Bool=true)
    println("  Generating header-aware wrapper (Clang.jl)...")
    println("   Method: Clang.jl header parsing")
    println("   Type safety:  Full (from headers)")
    println()

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    if isempty(headers)
        error("Headers required for advanced wrapping")
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
            "add_doc_strings" => generate_docs,
            "use_julia_native_enum" => true
        )
    )

    # Generate bindings via ClangJLBridge
    println("  Parsing headers with Clang.jl...")
    output_file = ClangJLBridge.generate_bindings_clangjl(clang_config, library_path, headers)

    if isnothing(output_file)
        error("Binding generation failed")
    end

    # TODO: Enhance generated file with our safety checks and metadata
    # For now, ClangJLBridge handles the generation

    println("   Generated: $output_file")
    println()

    return output_file
end

# =============================================================================
# FUNCTION POINTER SIGNATURE PARSING
# =============================================================================

"""
    parse_function_pointer_signature(fp_sig::String)::Union{String, Nothing}

Parse DWARF function pointer signature into Julia cfunction format.

# Format
DWARF format: `"function_ptr(return_type; param1, param2, ...)"`
Julia format: `"my_callback, return_type, (param1, param2, ...)"`

# Example
```julia
parse_function_pointer_signature("function_ptr(void; Ray*, AABB*, RayHit*)")
# => "my_callback, Cvoid, (Ptr{Ray}, Ptr{AABB}, Ptr{RayHit})"
```
"""
function parse_function_pointer_signature(fp_sig::String)::Union{String, Nothing}
    # Match: function_ptr(return_type) or function_ptr(return_type; params...)
    m = match(r"^function_ptr\(([^;)]+)(?:;\s*(.+))?\)$", fp_sig)

    if isnothing(m)
        return nothing
    end

    return_type = String(strip(m.captures[1]))
    params_str = isnothing(m.captures[2]) ? nothing : String(strip(m.captures[2]))

    # Convert C++ types to Julia types using type registry
    # Create a minimal type registry with standard type mappings
    base_types = Dict{String,String}(
        "void" => "Cvoid", "int" => "Cint", "unsigned int" => "Cuint",
        "long" => "Clong", "unsigned long" => "Culong",
        "short" => "Cshort", "unsigned short" => "Cushort",
        "char" => "Cchar", "unsigned char" => "Cuchar",
        "float" => "Cfloat", "double" => "Cdouble",
        "bool" => "Bool", "_Bool" => "Bool",
        "size_t" => "Csize_t", "int8_t" => "Int8", "uint8_t" => "UInt8",
        "int16_t" => "Int16", "uint16_t" => "UInt16",
        "int32_t" => "Int32", "uint32_t" => "UInt32",
        "int64_t" => "Int64", "uint64_t" => "UInt64"
    )

    registry = TypeRegistry(
        base_types, Dict{String,String}(), Dict{String,String}(),
        "Ptr", "Ref", :strip, nothing, WARN, true, true, true
    )

    # Map return type
    julia_return = infer_julia_type(registry, return_type, context="callback return type")

    # Map parameter types
    if isnothing(params_str) || isempty(strip(params_str))
        # No parameters - () in Julia
        return "my_callback, $julia_return, ()"
    else
        # Parse and map each parameter
        param_types = split(params_str, ",")
        julia_params = String[]

        for p in param_types
            p_stripped = String(strip(p))
            if !isempty(p_stripped)
                # Extract just the type by removing the parameter name
                # For "const double* x" or "size_t n", we need just the type part
                # Split by whitespace and take all but the last token (which is the param name)
                tokens = split(p_stripped)
                if length(tokens) > 1
                    # Check if last token looks like a parameter name (alphanumeric, not a type modifier)
                    last_token = tokens[end]
                    # If it doesn't end with *, &, or [], it's likely a parameter name
                    if !endswith(last_token, '*') && !endswith(last_token, '&') && !occursin('[', last_token)
                        # Remove the parameter name, keep the type
                        p_type = join(tokens[1:end-1], " ")
                    else
                        p_type = p_stripped
                    end
                else
                    p_type = p_stripped
                end

                julia_p = infer_julia_type(registry, p_type, context="callback parameter")
                push!(julia_params, julia_p)
            end
        end

        if isempty(julia_params)
            return "my_callback, $julia_return, ()"
        else
            param_tuple = "(" * join(julia_params, ", ") * ",)"  # Trailing comma for proper tuple
            return "my_callback, $julia_return, $param_tuple"
        end
    end
end

# =============================================================================
# TIER 3: INTROSPECTIVE WRAPPER (Metadata-Rich)
# =============================================================================

"""
    wrap_introspective(config::RepliBuildConfig, library_path::String, headers::Vector{String}; generate_docs::Bool=true)

Generate introspective Julia wrapper using compilation metadata for perfect type accuracy.

Quality: ~95% - Exact types from compilation, language-agnostic, zero manual configuration.
Use when: Metadata available from RepliBuild compilation, need perfect bindings.

This is the culmination of RepliBuild's vision: automatic, accurate, language-agnostic wrapping.
"""
function wrap_introspective(config::RepliBuildConfig, library_path::String, headers::Vector{String};
                           generate_docs::Bool=true)
    println("  Generating introspective wrapper (DWARF metadata)...")
    println("   Method: DWARF debug info + symbol metadata")
    println("   Type safety:  Excellent (~95% - ground truth from compilation)")
    println()

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # Load compilation metadata
    metadata_file = joinpath(dirname(library_path), "compilation_metadata.json")
    if !isfile(metadata_file)
        error("Compilation metadata not found: $metadata_file\nRun RepliBuild.build() first to generate metadata")
    end

    println("   Loading compilation metadata...")
    metadata = JSON.parsefile(metadata_file)

    if !haskey(metadata, "functions")
        error("Invalid metadata: missing 'functions' key")
    end

    functions = metadata["functions"]
    println("  ✓ Found $(length(functions)) functions with type information")

    # Extract supplementary types from headers (enums, unused types, etc.)
    header_types = if !isempty(headers)
        println("   Parsing headers for supplementary types...")
        ClangJLBridge.extract_header_types(headers)
    else
        # Auto-discover headers from include directories
        include_dirs = get(metadata, "include_dirs", String[])
        discovered_headers = String[]
        for inc_dir in include_dirs
            if isdir(inc_dir)
                append!(discovered_headers, ClangJLBridge.discover_headers(inc_dir, recursive=false))
            end
        end
        if !isempty(discovered_headers)
            println("   Parsing $(length(discovered_headers)) discovered headers...")
            ClangJLBridge.extract_header_types(discovered_headers)
        else
            Dict("enums" => Dict(), "constants" => Dict(), "typedefs" => Dict(), "structs" => String[])
        end
    end

    # Merge header types into metadata
    if !isempty(header_types["enums"])
        if !haskey(metadata, "header_enums")
            metadata["header_enums"] = header_types["enums"]
        else
            merge!(metadata["header_enums"], header_types["enums"])
        end
        println("  ✓ Merged $(length(header_types["enums"])) enums from headers")
    end

    # Store function pointer typedefs for callback documentation
    if haskey(header_types, "function_pointers") && !isempty(header_types["function_pointers"])
        metadata["function_pointer_typedefs"] = header_types["function_pointers"]
        println("  ✓ Merged $(length(header_types["function_pointers"])) function pointer typedefs from headers")
    end

    println()

    # Create type registry with metadata
    registry = create_type_registry(config)

    # Generate wrapper module
    module_name = get_module_name(config)
    wrapper_content = generate_introspective_module(config, library_path, metadata,
                                                    module_name, registry, generate_docs)

    # Write to file
    output_dir = get_output_path(config)
    mkpath(output_dir)
    output_file = joinpath(output_dir, "$(module_name).jl")

    write(output_file, wrapper_content)

    println("   Generated: $output_file")
    println("   Functions wrapped: $(length(functions))")
    println("   Type accuracy: ~95% (from compilation metadata)")
    println()

    return output_file
end

# =============================================================================
# DISPATCH LOGIC HELPERS
# =============================================================================

"""
    get_julia_aligned_size(members::Vector)

Calculate the size of a struct in Julia including standard padding alignment.
Used to detect if a C++ struct is 'packed' (Julia size > DWARF size).
"""
function get_julia_aligned_size(members::Vector)
    current_offset = 0
    max_align = 1

    for m in members
        # specific size of this member
        m_size = get(m, "size", 0)

        # simple alignment heuristic (size usually equals alignment for primitives)
        # generic pointer/int alignment cap at 8 bytes on 64-bit
        align = m_size > 8 ? 8 : m_size
        align = align == 0 ? 1 : align # handle empty/void

        # update generic alignment requirement
        max_align = max(max_align, align)

        # Add padding to current offset
        padding = (align - (current_offset % align)) % align
        current_offset += padding + m_size
    end

    # Final structure alignment padding
    padding = (max_align - (current_offset % max_align)) % max_align
    return current_offset + padding
end

"""
    is_ccall_safe(func_info, dwarf_structs)::Bool

Determine if a function is safe for standard `ccall`.
Returns false if:
1. Arguments are Packed Structs (alignment mismatch)
2. Arguments are Unions
3. Return type is a complex struct by value
"""
function is_ccall_safe(func_info, dwarf_structs)
    # 1. Check Return Type
    ret_type = String(get(func_info["return_type"], "c_type", ""))

    # If returning a struct by value (not pointer/void/primitive)
    if !contains(ret_type, "*") && !contains(ret_type, "void") &&
       !contains(ret_type, "int") && !contains(ret_type, "float") &&
       !contains(ret_type, "double") && !contains(ret_type, "bool")

        # Check if it's a known struct
        if haskey(dwarf_structs, ret_type)
            # Struct return by value is notoriously fragile in ABIs (large structs split registers)
            # Conservative: Send to MLIR if > 16 bytes
            s_info = dwarf_structs[ret_type]
            if parse(Int, get(s_info, "byte_size", "0")) > 16
                return false
            end
            
            # Check if it is a class (likely non-POD)
            if get(s_info, "kind", "struct") == "class"
                return false
            end
        end
    end

    # 2. Check Arguments
    for param in func_info["parameters"]
        c_type = get(param, "c_type", "")

        # Clean pointer syntax to check base type
        # Ensure we allocate a new String to satisfy strict JSON.Object requirements
        base_type = replace(c_type, "*" => "")
        base_type = String(strip(replace(base_type, "const" => "")))

        if haskey(dwarf_structs, base_type)
            s_info = dwarf_structs[base_type]

            # CHECK A: Is it a Union?
            if get(s_info, "kind", "struct") == "union"
                return false
            end
            
            # CHECK B: Is it a Class? (likely non-POD)
            if get(s_info, "kind", "struct") == "class"
                return false
            end

            # CHECK C: Is it Packed?
            dwarf_size = parse(Int, get(s_info, "byte_size", "0"))
            members = get(s_info, "members", [])
            julia_size = get_julia_aligned_size(members)

            # If DWARF says 5 bytes but Julia calculates 8, it's packed!
            if dwarf_size > 0 && dwarf_size != julia_size
                return false
            end
        end
    end

    return true
end

"""
Generate introspective wrapper module content using compilation metadata.
"""
function generate_introspective_module(config::RepliBuildConfig, lib_path::String,
                                      metadata, module_name::String,
                                      registry::TypeRegistry, generate_docs::Bool)

    # Header with metadata
    header = """
    # Auto-generated Julia wrapper for $(config.project.name)
    # Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    # Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
    # Library: $(basename(lib_path))
    # Metadata: compilation_metadata.json
    #
    # Type Safety: Excellent (~95%) - Types extracted from DWARF debug info
    # Ground truth: Types come from compiled binary, not headers
    # Manual edits: Minimal to none required

    module $module_name
    
    using Libdl
    import RepliBuild

    const LIBRARY_PATH = \"$(abspath(lib_path))\"

    # Verify library exists
    if !isfile(LIBRARY_PATH)
        error("Library not found: \$LIBRARY_PATH")
    end

    function __init__()
        # Initialize the global JIT context with this library's vtables
        RepliBuild.JITManager.initialize_global_jit(LIBRARY_PATH)
    end

    """

    # Metadata section
    compiler_info = get(metadata, "compiler_info", Dict())
    metadata_section = """
    # =============================================================================
    # Compilation Metadata
    # =============================================================================

    const METADATA = Dict(
        "llvm_version" => "$(get(compiler_info, "llvm_version", "unknown"))",
        "clang_version" => "$(get(compiler_info, "clang_version", "unknown"))",
        "optimization" => "$(get(compiler_info, "optimization_level", "unknown"))",
        "target_triple" => "$(get(compiler_info, "target_triple", "unknown"))",
        "function_count" => $(get(metadata, "function_count", 0)),
        "generated_at" => "$(get(metadata, "timestamp", "unknown"))"
    )

    """

    # Extract metadata
    functions = metadata["functions"]
    dwarf_structs = get(metadata, "struct_definitions", Dict())

    # Struct definitions
    # Collect all struct names from DWARF (excluding enums which have __enum__ prefix)
    struct_types = Set{String}()
    for (name, info) in dwarf_structs
        if !startswith(name, "__enum__") && haskey(info, "members")
            push!(struct_types, name)
        end
    end

    # =============================================================================
    # ENUM GENERATION (from DWARF)
    # =============================================================================

    enum_definitions = ""

    # Extract enums (stored with __enum__ prefix)
    enum_types = filter(k -> startswith(k, "__enum__"), keys(dwarf_structs))

    # Build set of enum names (without prefix) to exclude from struct generation
    enum_names = Set{String}()
    for enum_key in enum_types
        enum_name = replace(enum_key, "__enum__" => "")
        push!(enum_names, enum_name)
    end

    if !isempty(enum_types)
        enum_definitions *= """
        # =============================================================================
        # Enum Definitions (from DWARF debug info)
        # =============================================================================

        """

        for enum_key in sort(collect(enum_types))
            enum_name = replace(enum_key, "__enum__" => "")
            enum_info = dwarf_structs[enum_key]

            # Get enum metadata
            underlying_type = get(enum_info, "underlying_type", "int")
            julia_underlying = get(enum_info, "julia_type", "Int32")

            # Fallback to Int32 if type is unknown or Any
            if julia_underlying == "Any" || julia_underlying == "unknown"
                julia_underlying = "Int32"
            end

            enumerators = get(enum_info, "enumerators", [])

            if !isempty(enumerators)
                enum_definitions *= """
                # C++ enum: $enum_name (underlying type: $underlying_type)
                @enum $enum_name::$julia_underlying begin
                """

                for (i, enumerator) in enumerate(enumerators)
                    name = get(enumerator, "name", "Unknown")
                    value = get(enumerator, "value", 0)
                    enum_definitions *= "    $name = $value\n"
                end

                enum_definitions *= """
                end

                """
            end
        end

        enum_definitions *= "\n"
    end

    # =============================================================================
    # ENUM GENERATION (from Headers - supplementary)
    # =============================================================================
    # Add enums found in headers that weren't in DWARF (unused/optimized away)
    header_enums = get(metadata, "header_enums", Dict())
    if !isempty(header_enums)
        added_header_enums = 0
        for (enum_name, members) in header_enums
            # Skip if already defined from DWARF
            if enum_name in enum_names
                continue
            end

            if !isempty(members)
                if added_header_enums == 0
                    enum_definitions *= """
                    # =============================================================================
                    # Enum Definitions (from Headers - supplementary)
                    # =============================================================================
                    # These enums were not in DWARF (unused code eliminated by compiler)

                    """
                end

                enum_definitions *= """
                # C++ enum: $enum_name (from header - not in DWARF)
                @enum $enum_name::Cint begin
                """

                for (member_name, value) in members
                    enum_definitions *= "    $member_name = $value\n"
                end

                enum_definitions *= """
                end

                """
                added_header_enums += 1
            end
        end

        if added_header_enums > 0
            enum_definitions *= "\n"
        end
    end

    # =============================================================================
    # STRUCT GENERATION (from DWARF)
    # =============================================================================

    # Scan for referenced structs that are missing definitions (opaque types)
    opaque_structs = Set{String}()
    for (name, info) in dwarf_structs
        if !startswith(name, "__enum__") && haskey(info, "members")
            members = get(info, "members", [])
            for member in members
                julia_type = get(member, "julia_type", "Any")
                # Extract inner type from Ptr{T} (recursively for Ptr{Ptr{T}})
                inner = julia_type
                found_ptr = false
                while startswith(inner, "Ptr{") && endswith(inner, "}")
                    inner = inner[5:end-1]
                    found_ptr = true
                end

                if found_ptr
                    # If it looks like a struct (not a primitive) and not in defined structs
                    builtin_types = ["Cvoid", "Cint", "Cuint", "Clong", "Culong", "Cshort", "Cushort", 
                                     "Cchar", "Cuchar", "Cfloat", "Cdouble", "Bool", "UInt8", "Int8", 
                                     "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64", "Csize_t",
                                     "Any", "Nothing"]
                    
                    # Also exclude existing structs and NTuple
                    if !(inner in builtin_types) && !(inner in struct_types) && !startswith(inner, "NTuple")
                        push!(opaque_structs, inner)
                    end
                end
            end
        end
    end

    struct_definitions = ""

    if !isempty(opaque_structs)
        struct_definitions *= """
        # =============================================================================
        # Opaque Struct Declarations
        # =============================================================================

        """
        for name in sort(collect(opaque_structs))
            # Sanitize
            s_name = replace(replace(replace(name, "<" => "_"), ">" => ""), "," => "_")
            s_name = replace(s_name, " " => "")
            struct_definitions *= "mutable struct $s_name end\n"
        end
        struct_definitions *= "\n"
    end

    if !isempty(struct_types)
        struct_definitions *= """
        # =============================================================================
        # Struct Definitions (from DWARF debug info)
        # =============================================================================

        """

        # Topologically sort structs by dependencies
        # Build dependency graph: struct_name => [dependencies]
        struct_deps = Dict{String, Set{String}}()

        for struct_name in struct_types
            deps = Set{String}()

            if haskey(dwarf_structs, struct_name)
                struct_info = dwarf_structs[struct_name]
                members = get(struct_info, "members", [])

                for member in members
                    julia_type = get(member, "julia_type", "Any")

                    # Extract type name (handle Ptr{Foo} -> Foo)
                    type_match = match(r"Ptr\{([^}]+)\}", julia_type)
                    if !isnothing(type_match)
                        dep_type = type_match.captures[1]
                        # Check both original and sanitized forms
                        if dep_type in struct_types && dep_type != struct_name
                            push!(deps, dep_type)
                        end
                    elseif julia_type in struct_types && julia_type != struct_name
                        push!(deps, julia_type)
                    end
                end
            end

            struct_deps[struct_name] = deps
        end

        # Topological sort using Kahn's algorithm
        sorted_structs = String[]
        remaining = Dict(k => copy(v) for (k, v) in struct_deps)

        while !isempty(remaining)
            # Find structs with no dependencies
            ready = [name for (name, deps) in remaining if isempty(deps)]

            if isempty(ready)
                # Circular dependency - just take alphabetically first
                ready = [sort(collect(keys(remaining)))[1]]
            end

            for name in sort(ready)
                push!(sorted_structs, name)
                delete!(remaining, name)

                # Remove this struct from all dependency lists
                for deps in values(remaining)
                    delete!(deps, name)
                end
            end
        end

        # Helper to recursively flatten struct members from base classes
        function flatten_struct_members(s_name, visited=Set{String}())
            if s_name in visited
                return []
            end
            push!(visited, s_name)
            
            all_members = []
            if haskey(dwarf_structs, s_name)
                info = dwarf_structs[s_name]
                
                # First, add members from base classes
                bases = get(info, "base_classes", [])
                for base in bases
                    base_type = get(base, "type", "")
                    # Strip "class " or "struct " prefix if present
                    base_type = replace(base_type, r"^(class|struct)\s+" => "")
                    if !isempty(base_type)
                        append!(all_members, flatten_struct_members(base_type, visited))
                    end
                end
                
                # Then add own members
                own_members = get(info, "members", [])
                append!(all_members, own_members)
            end
            return all_members
        end

        for struct_name in sorted_structs
            # Skip if this is actually an enum (enums are generated separately)
            if struct_name in enum_names
                continue
            end

            # Sanitize struct name for Julia (replace < > , with _)
            julia_struct_name = replace(replace(replace(struct_name, "<" => "_"), ">" => ""), "," => "_")
            julia_struct_name = replace(julia_struct_name, " " => "")  # Remove spaces

            # Check if we have DWARF member information for this struct
            if haskey(dwarf_structs, struct_name)
                struct_info = dwarf_structs[struct_name]
                kind = get(struct_info, "kind", "struct")
                
                # SPECIAL HANDLING FOR UNIONS
                if kind == "union"
                    byte_size_str = get(struct_info, "byte_size", "0x0")
                    byte_size = parse(Int, byte_size_str)
                    
                    if byte_size == 0
                        # Fallback if size missing
                        members = get(struct_info, "members", [])
                        for m in members
                            m_size = get(m, "size", 0)
                            byte_size = max(byte_size, m_size)
                        end
                        if byte_size == 0; byte_size = 8; end # Panic fallback
                    end

                    struct_definitions *= """
                    # C++ union: $struct_name (size $byte_size bytes)
                    mutable struct $julia_struct_name
                        data::NTuple{$byte_size, UInt8}
                    end
                    
                    """
                    continue
                end

                # BUG FIX: Recursively get all members including base classes
                members = flatten_struct_members(struct_name)

                if !isempty(members)
                    # Sort members by offset to ensure correct order
                    sort!(members, by = m -> begin
                        off = get(m, "offset", "0x0")
                        isnothing(off) ? 0 : parse(Int, off)
                    end)
                    
                    member_count = length(members)
                    struct_definitions *= """
                    # C++ struct: $struct_name ($member_count members)
                    struct $julia_struct_name
                    """
                    
                    current_offset = 0
                    pad_idx = 0

                    for member in members
                        member_name = get(member, "name", "unknown")
                        julia_type = get(member, "julia_type", "Any")
                        
                        offset_val = get(member, "offset", "0x0")
                        offset = isnothing(offset_val) ? 0 : parse(Int, offset_val)
                        
                        # Insert padding if needed
                        if offset > current_offset
                            pad_size = offset - current_offset
                            struct_definitions *= "    _pad_$(pad_idx)::NTuple{$(pad_size), UInt8}\n"
                            pad_idx += 1
                            current_offset = offset
                        end

                        # Sanitize member name for Julia (replace invalid characters)
                        sanitized_name = replace(member_name, '$' => '_')

                        # Sanitize member types that reference other structs with template syntax
                        # Only sanitize custom struct names, not built-in Julia types like NTuple
                        sanitized_type = julia_type

                        # Don't sanitize built-in Julia types (NTuple, Ptr{Cint}, etc.)
                        builtin_types = ["NTuple", "Ptr", "Cint", "Cuint", "Cdouble", "Cfloat", "Clong", "Culonglong", "Cvoid", "UInt8", "Int8", "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64"]
                        is_builtin = any(startswith(julia_type, bt) for bt in builtin_types)

                        if !is_builtin
                            if occursin(r"Ptr\{[^}]+\}", julia_type)
                                # Extract type from Ptr{Type} for custom struct references
                                type_match = match(r"Ptr\{([^}]+)\}", julia_type)
                                if !isnothing(type_match)
                                    inner_type = type_match.captures[1]
                                    # Only sanitize if inner type is a custom struct (contains template chars)
                                    if occursin(r"[<>]", inner_type)
                                        sanitized_inner = replace(replace(replace(inner_type, "<" => "_"), ">" => ""), "," => "_")
                                        sanitized_inner = replace(sanitized_inner, " " => "")
                                        sanitized_type = "Ptr{$sanitized_inner}"
                                    end
                                end
                            elseif occursin(r"[<>]", julia_type)
                                # Direct custom struct reference with template syntax
                                sanitized_type = replace(replace(replace(julia_type, "<" => "_"), ">" => ""), "," => "_")
                                sanitized_type = replace(sanitized_type, " " => "")
                            end
                        end

                        struct_definitions *= "    $sanitized_name::$sanitized_type\n"
                        
                        # Update current offset
                        member_size = get(member, "size", 0)
                        # If size is 0 (e.g. unknown type), we can't reliably track offset
                        # But typically we rely on the next member's offset to insert padding
                        current_offset += member_size
                    end

                    struct_definitions *= """
                    end

                    """
                else
                    # Struct found but no members (empty struct or incomplete info)
                    struct_definitions *= """
                    # Opaque struct: $struct_name (no member info available)
                    mutable struct $julia_struct_name
                        data::NTuple{8, UInt8}  # Placeholder
                    end

                    """
                end
            else
                # No DWARF info available - generate opaque struct
                struct_definitions *= """
                # Opaque struct: $struct_name (no DWARF info)
                mutable struct $julia_struct_name
                    data::NTuple{32, UInt8}  # Placeholder - compile with -g for member info
                end

                """
            end
        end

        struct_definitions *= "\n"
    end

    # Function wrappers
    function_wrappers = ""

    # Track exported function names
    exports = String[]

    for func in functions
        func_name = func["name"]
        mangled = func["mangled"]
        demangled = func["demangled"]

        # =========================================================
        # TIERED DISPATCH DECISION
        # =========================================================

        # Determine if we should use MLIR or ccall
        use_mlir_dispatch = !is_ccall_safe(func, dwarf_structs)
        
        # BUG FIX: Make copies to allow modification (injecting 'this', refining types) without affecting metadata
        params = copy(func["parameters"])
        return_type = copy(func["return_type"])
        
        is_method = get(func, "is_method", false)
        class_name = get(func, "class", "")

        # Skip constructors for now (need special handling/factory functions)
        if is_method && func_name == class_name
            continue 
        end

        # BUG FIX: Inject missing 'this' pointer for methods
        # DWARF metadata sometimes omits the implicit 'this' parameter for methods
        if is_method && !isempty(class_name)
            # Check if 'this' is already present (it should be first)
            has_this = !isempty(params) && (params[1]["name"] == "this")
            
            if !has_this
                # Synthesize 'this' parameter
                # Sanitize class name for Julia type
                safe_class = replace(replace(replace(class_name, "<" => "_"), ">" => ""), "," => "_")
                safe_class = replace(safe_class, " " => "")
                
                this_param = Dict{String,Any}(
                    "name" => "this",
                    "c_type" => class_name * "*",
                    "julia_type" => "Ptr{" * safe_class * "}",
                    "position" => 0,
                    "is_synthesized" => true
                )
                pushfirst!(params, this_param)
            end
        end

        # BUG FIX: Refine return types (Ptr{Cvoid} -> Ptr{Struct})
        # If c_type indicates a pointer to a known struct, but julia_type is generic Ptr{Cvoid}, fix it.
        c_ret = get(return_type, "c_type", "")
        j_ret = get(return_type, "julia_type", "")
        
        if (j_ret == "Ptr{Cvoid}" || j_ret == "Any") && endswith(c_ret, "*")
             base_c = strip(c_ret[1:end-1])
             # Remove const/struct/class prefixes
             base_c = replace(base_c, r"^(const\s+|struct\s+|class\s+)+" => "")
             base_c = strip(base_c)
             
             if base_c in struct_types
                 # Sanitize
                 safe_base = replace(replace(replace(base_c, "<" => "_"), ">" => ""), "," => "_")
                 safe_base = replace(safe_base, " " => "")
                 return_type["julia_type"] = "Ptr{$safe_base}"
             end
        end

        # BUG FIX: Sanitize return types that are template instantiations (e.g. Box<double> -> Box_double)
        # This handles by-value returns of template types
        ret_type_str = get(return_type, "julia_type", "Cvoid")
        if occursin(r"[<>]", ret_type_str) && !startswith(ret_type_str, "Ptr{") && !startswith(ret_type_str, "Ref{")
             safe_ret = replace(replace(replace(ret_type_str, "<" => "_"), ">" => ""), "," => "_")
             safe_ret = replace(safe_ret, " " => "")
             return_type["julia_type"] = safe_ret
        end

        # Build parameter list with ergonomic types
        param_names = String[]
        param_types = String[]  # C types for ccall
        julia_param_types = String[]  # Julia types for function signature (may differ)
        needs_conversion = Bool[]

        for (i, param) in enumerate(params)
            # Sanitize parameter name (e.g., avoid 'end', 'function' keywords)
            safe_name = make_julia_identifier(param["name"])
            push!(param_names, safe_name)
            julia_type = param["julia_type"]
            c_type_name = get(param, "c_type", "")

            # Determine the actual C type for ccall
            # If julia_type is "Any" but c_type has a name, it's likely a struct
            actual_c_type = if julia_type == "Any" && !isempty(c_type_name) && c_type_name in struct_types
                c_type_name  # Use the struct name directly
            else
                julia_type
            end

            push!(param_types, actual_c_type)

            # Map C integer types to natural Julia types with range checking
            if actual_c_type == "Cint"  # Int32 in Julia
                push!(julia_param_types, "Integer")  # Accept any Integer, will validate
                push!(needs_conversion, true)
            elseif actual_c_type == "Clong"  # Platform-dependent
                push!(julia_param_types, "Integer")
                push!(needs_conversion, true)
            elseif actual_c_type == "Cshort"  # Int16
                push!(julia_param_types, "Integer")
                push!(needs_conversion, true)
            else
                push!(julia_param_types, actual_c_type)
                push!(needs_conversion, false)
            end
        end

        # Julia function name (avoid conflicts and sanitize)
        julia_name = func_name
        if is_method && !isempty(class_name)
            julia_name = "$(class_name)_$(func_name)"
        end

        # Sanitize function name - remove invalid characters
        julia_name = replace(julia_name, "~" => "destroy_")  # Destructor
        julia_name = replace(julia_name, "::" => "_")
        julia_name = replace(julia_name, "<" => "_")
        julia_name = replace(julia_name, ">" => "_")
        julia_name = replace(julia_name, "," => "_")
        julia_name = replace(julia_name, " " => "_")
        julia_name = replace(julia_name, "+" => "plus")
        julia_name = replace(julia_name, "=" => "assign")
        julia_name = replace(julia_name, "-" => "minus")
        julia_name = replace(julia_name, "*" => "mul")
        julia_name = replace(julia_name, "/" => "div")

        # Build function signature using ergonomic Julia types
        param_sig = join(["$(name)::$(typ)" for (name, typ) in zip(param_names, julia_param_types)], ", ")

        # Documentation
        doc_comment = ""
        if generate_docs
            # Build detailed argument documentation, detecting function pointers
            arg_docs = String[]
            callback_docs = String[]

            for (i, param) in enumerate(params)
                name = param["name"]
                c_type_name = get(param, "c_type", "")
                julia_type = param["julia_type"]

                # Check if this is a callback parameter (Ptr{Cvoid} from function pointer)
                if julia_type == "Ptr{Cvoid}" && (startswith(c_type_name, "function_ptr") || contains(c_type_name, "(*"))
                    # Try to resolve the actual signature from typedef or DWARF
                    fp_sig = nothing

                    # First: Check if we have a direct function pointer signature from DWARF
                    if haskey(param, "function_pointer_signature")
                        fp_sig = param["function_pointer_signature"]
                    end

                    # Second: Try to match against header typedef signatures (better than DWARF incomplete sigs)
                    # Even if we have a DWARF signature, prefer typedef if available
                    if haskey(metadata, "function_pointer_typedefs")
                        # Try to find the best matching typedef
                        # Strategy: Match by parameter name similarity or function name
                        best_typedef = nothing
                        best_score = 0

                        for (typedef_name, typedef_info) in metadata["function_pointer_typedefs"]
                            score = 0

                            # Check if typedef name is contained in parameter name or vice versa
                            # e.g., "TransformCallback" in "callback" or "transform" in "TransformCallback"
                            typedef_lower = lowercase(typedef_name)
                            param_lower = lowercase(name)
                            func_lower = lowercase(func_name)

                            if contains(typedef_lower, param_lower) || contains(param_lower, typedef_lower)
                                score += 10
                            end

                            # Check if typedef name is related to function name
                            # e.g., "Transform" in "register_transform_callback"
                            typedef_base = replace(typedef_lower, "callback" => "")
                            if contains(func_lower, typedef_base)
                                score += 20
                            end

                            if score > best_score
                                best_score = score
                                best_typedef = (typedef_name, typedef_info)
                            end
                        end

                        # Use best match, or first typedef if no good match
                        if !isnothing(best_typedef)
                            typedef_name, typedef_info = best_typedef
                            ret_type = typedef_info["return_type"]
                            params_list = typedef_info["parameters"]

                            # Construct DWARF-style signature for parsing
                            if isempty(params_list)
                                fp_sig = "function_ptr($ret_type)"
                            else
                                fp_sig = "function_ptr($ret_type; $(join(params_list, ", ")))"
                            end
                        elseif !isempty(metadata["function_pointer_typedefs"])
                            # Fallback: use first typedef
                            typedef_name, typedef_info = first(metadata["function_pointer_typedefs"])
                            ret_type = typedef_info["return_type"]
                            params_list = typedef_info["parameters"]

                            if isempty(params_list)
                                fp_sig = "function_ptr($ret_type)"
                            else
                                fp_sig = "function_ptr($ret_type; $(join(params_list, ", ")))"
                            end
                        end
                    end

                    # Generate documentation if we have a signature
                    if !isnothing(fp_sig)
                        julia_sig = parse_function_pointer_signature(fp_sig)

                        if !isnothing(julia_sig)
                            push!(arg_docs, "- `$(name)::Ptr{Cvoid}` - Callback function")
                            push!(callback_docs, """
                                **Callback `$name`**: Create using `@cfunction`
                                ```julia
                                callback = @cfunction($julia_sig) Ptr{Cvoid}
                                ```""")
                        else
                            # Fallback if parsing fails
                            push!(arg_docs, "- `$(name)::$(julia_type)` - Callback function (signature: $fp_sig)")
                        end
                    else
                        # No signature info available
                        push!(arg_docs, "- `$(name)::$(julia_type)` - Callback function (signature unknown)")
                    end
                else
                    # Regular parameter
                    push!(arg_docs, "- `$(name)::$(julia_type)`")
                end
            end

            # Build the docstring
            doc_parts = """
            \"\"\"
                $julia_name($param_sig) -> $(return_type["julia_type"])

            Wrapper for C++ function: `$demangled`

            # Arguments
            $(join(arg_docs, "\n"))

            # Returns
            - `$(return_type["julia_type"])`"""

            # Add callback documentation if any
            if !isempty(callback_docs)
                doc_parts *= "\n\n            # Callback Signatures\n"
                doc_parts *= join(callback_docs, "\n\n")
            end

            doc_parts *= """


            # Metadata
            - Mangled symbol: `$mangled`
            - Type safety:  From compilation
            \"\"\"
            """

            doc_comment = doc_parts
        end

        # =========================================================
        # BRANCH 1: MLIR DISPATCH (Robust Path)
        # =========================================================
        if use_mlir_dispatch

            # Build argument list for invoke
            invoke_args = join(param_names, ", ")

            func_def = """
            $doc_comment
            function $julia_name($param_sig)
                # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
                return RepliBuild.JITManager.invoke("$mangled", $invoke_args)
            end
            """

            function_wrappers *= func_def
            push!(exports, julia_name)
            continue # Skip the rest of the loop (ccall generation)
        end

        # =========================================================
        # BRANCH 2: CCALL (Fast Path) - Existing Logic
        # =========================================================

        # Build conversion logic for parameters
        conversion_code = ""
        ccall_param_names = String[]

        for (i, (name, c_type, needs_conv)) in enumerate(zip(param_names, param_types, needs_conversion))
            if needs_conv
                converted_name = "$(name)_c"
                push!(ccall_param_names, converted_name)

                # Generate range-checked conversion
                if c_type == "Cint"
                    conversion_code *= "    $converted_name = Cint($name)  # Auto-converts with overflow check\n"
                elseif c_type == "Clong"
                    conversion_code *= "    $converted_name = Clong($name)\n"
                elseif c_type == "Cshort"
                    conversion_code *= "    $converted_name = Cshort($name)\n"
                end
            else
                push!(ccall_param_names, name)
            end
        end

        ccall_args = join(ccall_param_names, ", ")

        # ccall needs tuple expression (Type1, Type2) not Tuple{Type1, Type2}
        ccall_types = if isempty(param_types)
            "()"
        else
            "($(join(param_types, ", ")),)"  # Note: trailing comma for single-element tuples
        end

        # Generate function body based on return type and conversions
        julia_return_type = return_type["julia_type"]
        c_return_type = return_type["c_type"]

        # Check if return type is a struct (not primitive, not pointer)
        is_struct_return = julia_return_type == "Any" && !contains(c_return_type, "*") && !contains(c_return_type, "void") && c_return_type != "unknown"
        
        # Check if this is a virtual method that requires dynamic dispatch
        is_virtual = get(func, "is_virtual", false)

        if is_virtual
            # JIT-based virtual dispatch wrapper
            # 1. Get the class and method name
            # 2. Call get_jit_thunk(class, method) to get the function pointer
            # 3. ccall that pointer
            
            cls_name = get(func, "class", "")
            
            # Sanitize return type if needed
            safe_c_ret = c_return_type
            if occursin(r"[<>]", safe_c_ret)
                 safe_c_ret = replace(replace(replace(safe_c_ret, "<" => "_"), ">" => ""), "," => "_")
                 safe_c_ret = replace(safe_c_ret, " " => "")
            end
            
            # Use julia_return_type if it's not Any/Cstring, otherwise use sanitized C type
            ret_type_sig = (julia_return_type == "Any" || julia_return_type == "Cstring") ? safe_c_ret : julia_return_type
            
            func_def = """
            $doc_comment
            function $julia_name($param_sig)::$ret_type_sig
                # Get thunk for virtual dispatch (JIT compiled on-demand)
                thunk_ptr = RepliBuild.JITManager.get_jit_thunk("$cls_name", "$func_name")
                
                # Call thunk
            $conversion_code    return ccall(thunk_ptr, $ret_type_sig, $ccall_types, $ccall_args)
            end

            """
        elseif is_struct_return
            # Struct-valued return - Julia uses the struct type directly
            # ccall will handle struct returns automatically if the Julia type matches
            
            # Sanitize C return type if it contains template chars (Box<int> -> Box_int)
            safe_c_ret = c_return_type
            if occursin(r"[<>]", safe_c_ret)
                 safe_c_ret = replace(replace(replace(safe_c_ret, "<" => "_"), ">" => ""), "," => "_")
                 safe_c_ret = replace(safe_c_ret, " " => "")
            end

            func_def = """
            $doc_comment
            function $julia_name($param_sig)::$safe_c_ret
            $conversion_code    return ccall((:$mangled, LIBRARY_PATH), $safe_c_ret, $ccall_types, $ccall_args)
            end

            """
        elseif julia_return_type == "Cstring"
            # Cstring with NULL check and conversion to String
            func_def = """
            $doc_comment
            function $julia_name($param_sig)::String
            $conversion_code    ptr = ccall((:$mangled, LIBRARY_PATH), Cstring, $ccall_types, $ccall_args)
                if ptr == C_NULL
                    error("$julia_name returned NULL pointer")
                end
                return unsafe_string(ptr)
            end

            """
        elseif !isempty(conversion_code)
            # Has parameter conversions
            func_def = """
            $doc_comment
            function $julia_name($param_sig)::$julia_return_type
            $conversion_code    return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
            end

            """
        else
            # Standard wrapper - no conversions needed
            func_def = """
            $doc_comment
            function $julia_name($param_sig)::$julia_return_type
                ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
            end

            """
        end

        function_wrappers *= func_def
        push!(exports, julia_name)

        # Generate convenience wrappers for ergonomic APIs
        # Two types:
        # 1. Struct pointers -> accept structs directly
        # 2. Const primitive array pointers -> accept Vector directly with GC.@preserve

        has_struct_ptr_params = false
        struct_ptr_indices = Int[]
        has_array_ptr_params = false
        array_ptr_indices = Int[]
        convenience_param_types = String[]
        convenience_param_names = String[]

        for (i, (ptype, pname)) in enumerate(zip(param_types, param_names))
            # Check if this is a Ptr{StructName} where StructName is a known struct
            if startswith(ptype, "Ptr{") && !startswith(ptype, "Ptr{C") && ptype != "Ptr{Cvoid}"
                # Extract the struct name
                struct_name = replace(ptype, r"^Ptr\{(.+)\}$" => s"\1")

                # Check if this struct exists in our generated types
                struct_key = "__struct__" * struct_name
                if haskey(dwarf_structs, struct_key) || struct_name in ["DenseMatrix", "SparseMatrix", "LUDecomposition", "QRDecomposition", "EigenResult", "ODESolution", "FFTResult", "Histogram", "OptimizationState", "OptimizationOptions", "PolynomialFit", "CubicSpline", "Vector3", "Matrix3x3"]
                    has_struct_ptr_params = true
                    push!(struct_ptr_indices, i)
                    push!(convenience_param_types, struct_name)
                    push!(convenience_param_names, pname)
                else
                    push!(convenience_param_types, ptype)
                    push!(convenience_param_names, pname)
                end
            # Check if this is Ptr{Cdouble}, Ptr{Cfloat}, Ptr{Cint}, etc. (array pointers)
            # These benefit from Vector{T} wrapper with automatic GC.@preserve
            elseif ptype in ["Ptr{Cdouble}", "Ptr{Cfloat}", "Ptr{Cint}", "Ptr{Cuint}",
                             "Ptr{Int32}", "Ptr{Int64}", "Ptr{UInt32}", "Ptr{UInt64}",
                             "Ptr{Float32}", "Ptr{Float64}"]
                # Check if parameter name suggests it's an input array (not output)
                # Common patterns: x, y, data, array, vector, signal, values, input, src
                pname_lower = lowercase(pname)
                is_likely_input = any(pattern -> contains(pname_lower, pattern),
                    ["x", "y", "data", "array", "vec", "signal", "value", "input", "src", "coef"])

                # Also check if it's NOT likely an output
                is_likely_output = any(pattern -> contains(pname_lower, pattern),
                    ["out", "result", "dest", "dst", "buffer"])

                if is_likely_input && !is_likely_output
                    has_array_ptr_params = true
                    push!(array_ptr_indices, i)
                    # Convert Ptr{Cdouble} -> Vector{Float64}, etc.
                    elem_type = replace(ptype, "Ptr{" => "", "}" => "")
                    julia_vec_type = if elem_type == "Cdouble"
                        "Vector{Float64}"
                    elseif elem_type == "Cfloat"
                        "Vector{Float32}"
                    elseif elem_type in ["Cint", "Int32"]
                        "Vector{Int32}"
                    elseif elem_type in ["Cuint", "UInt32"]
                        "Vector{UInt32}"
                    elseif elem_type == "Int64"
                        "Vector{Int64}"
                    elseif elem_type == "UInt64"
                        "Vector{UInt64}"
                    elseif elem_type == "Float64"
                        "Vector{Float64}"
                    elseif elem_type == "Float32"
                        "Vector{Float32}"
                    else
                        ptype  # Fallback to pointer type
                    end
                    push!(convenience_param_types, julia_vec_type)
                    push!(convenience_param_names, pname)
                else
                    push!(convenience_param_types, ptype)
                    push!(convenience_param_names, pname)
                end
            else
                push!(convenience_param_types, ptype)
                push!(convenience_param_names, pname)
            end
        end

        # Generate convenience wrapper if we have struct pointer or array pointer parameters
        if has_struct_ptr_params || has_array_ptr_params
            convenience_sig = join(["$pname::$ptype" for (pname, ptype) in zip(convenience_param_names, convenience_param_types)], ", ")

            # Build the ccall arguments
            convenience_ccall_args = String[]
            for (i, pname) in enumerate(param_names)
                if i in struct_ptr_indices
                    # Struct parameter: use Ref()
                    push!(convenience_ccall_args, "Ref($pname)")
                elseif i in array_ptr_indices
                    # Array parameter: use pointer()
                    push!(convenience_ccall_args, "pointer($pname)")
                else
                    # Regular parameter: pass as-is
                    push!(convenience_ccall_args, pname)
                end
            end
            convenience_ccall = join(convenience_ccall_args, ", ")

            # If we have array parameters, wrap the ccall in GC.@preserve
            # Collect all array parameter names for preservation
            array_pnames = [param_names[i] for i in array_ptr_indices]

            if !isempty(array_pnames)
                preserve_vars = join(array_pnames, " ")
                convenience_func = """
                # Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
                function $julia_name($convenience_sig)::$julia_return_type
                    return GC.@preserve $preserve_vars begin
                        ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $convenience_ccall)
                    end
                end

                """
            else
                # No array parameters, just struct parameters
                convenience_func = """
                # Convenience wrapper - accepts structs directly instead of pointers
                function $julia_name($convenience_sig)::$julia_return_type
                    return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $convenience_ccall)
                end

                """
            end

            function_wrappers *= convenience_func
        end
    end

    # Export statement (unique to handle overloaded functions)
    # Include enum types, enum values, struct types, and functions
    all_exports = copy(exports)

    # Add enum types
    for enum_key in enum_types
        enum_name = replace(enum_key, "__enum__" => "")
        push!(all_exports, enum_name)

        # Add enum constants
        if haskey(dwarf_structs, enum_key)
            enum_info = dwarf_structs[enum_key]
            enumerators = get(enum_info, "enumerators", [])
            for enumerator in enumerators
                enum_const_name = get(enumerator, "name", "")
                if !isempty(enum_const_name)
                    push!(all_exports, enum_const_name)
                end
            end
        end
    end

    # Add struct types
    for struct_name in struct_types
        if !(struct_name in enum_names)
            # Sanitize struct name for export
            julia_struct_name = replace(replace(replace(struct_name, "<" => "_"), ">" => ""), "," => "_")
            julia_struct_name = replace(julia_struct_name, " " => "")
            push!(all_exports, julia_struct_name)
        end
    end

    export_statement = if !isempty(all_exports)
        "export " * join(unique(all_exports), ", ") * "\n\n"
    else
        ""
    end

    # Footer
    footer = """

    end # module $module_name
    """

    return header * metadata_section * enum_definitions * struct_definitions * export_statement * function_wrappers * footer
end

end # module Wrapper
