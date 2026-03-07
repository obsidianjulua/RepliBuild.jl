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
import ..MLIRNative
import ..DWARFParser
import ..JLCSIRGenerator

export wrap_library, wrap_basic, extract_symbols
export TypeRegistry, SymbolInfo, ParamInfo
export TypeStrictness, STRICT, WARN, PERMISSIVE
export is_struct_like, is_enum_like, is_function_pointer_like

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Complete set of Julia reserved keywords and soft keywords for enum member escaping
const _JULIA_KEYWORDS = Set([
    "baremodule", "begin", "break", "catch", "const", "continue", "do",
    "else", "elseif", "end", "export", "false", "finally", "for",
    "function", "global", "if", "import", "in", "let", "local",
    "macro", "module", "mutable", "nothing", "quote", "return",
    "struct", "true", "try", "type", "using", "while", "abstract",
    "primitive", "where", "isa",
    # Not technically reserved but conflict as identifiers in enum context
    "and", "or", "not",
])

# Internal/compiler types that leak through DWARF but shouldn't be exported
const _INTERNAL_TYPE_BLOCKLIST = Set([
    "__va_list_tag", "__mbstate_t", "__loadu_pd", "__storeu_pd",
    "__loadu_ps", "__storeu_ps", "__loadu_si128", "__storeu_si128",
    "_va_list_tag", "_mbstate_t", "_loadu_pd", "_storeu_pd",
    "_loadu_ps", "_storeu_ps",
    "ldiv_t", "lldiv_t", "div_t", "max_align_t", "imaxdiv_t",
])

"""Escape a name if it's a Julia keyword, using var\"...\" syntax."""
function _escape_keyword(name::String)::String
    if name in _JULIA_KEYWORDS
        return "var\"$name\""
    end
    return name
end

# Built-in Julia types that never need forward declaration
const _JULIA_BUILTIN_TYPES = Set([
    "Cvoid", "Cint", "Cuint", "Clong", "Culong", "Cshort", "Cushort",
    "Cchar", "Cuchar", "Cfloat", "Cdouble", "Bool", "UInt8", "Int8",
    "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64", "Csize_t",
    "Clonglong", "Culonglong", "Cptrdiff_t", "Cssize_t", "Cwchar_t",
    "Cstring", "Float32", "Float64", "Any", "Nothing", "Cintptr_t", "Cuintptr_t",
])

"""
    _resolve_forward_ptr(julia_type, defined_names) -> String

Replace `Ptr{X}` (including nested `Ptr{Ptr{X}}`) with `Ptr{Cvoid}` when `X`
is a custom struct type that has not been defined yet. This avoids
`UndefVarError` from forward references while preserving ABI layout
(all pointers are the same size).
"""
function _resolve_forward_ptr(julia_type::AbstractString, defined_names::Set{String})::String
    m = match(r"^Ptr\{(.+)\}$", julia_type)
    isnothing(m) && return julia_type
    inner = m.captures[1]
    # Recurse for nested Ptr
    resolved_inner = _resolve_forward_ptr(inner, defined_names)
    # Check if the innermost type is defined or builtin
    base = inner
    while (bm = match(r"^Ptr\{(.+)\}$", base)) !== nothing
        base = bm.captures[1]
    end
    if base in _JULIA_BUILTIN_TYPES || base in defined_names
        return "Ptr{$resolved_inner}"
    end
    return "Ptr{Cvoid}"
end

# Helper: sanitize a C++ type name to a valid Julia struct/type identifier
function _sanitize_julia_type_name(name::AbstractString)::String
    s = replace(string(name), "::" => "_")
    s = replace(s, "<"  => "_")
    s = replace(s, ">"  => "")
    s = replace(s, ","  => "_")
    s = replace(s, " "  => "")
    s = replace(s, "-"  => "minus_")
    s = replace(s, "+"  => "plus_")
    s = replace(s, "*"  => "star_")
    # Collapse consecutive underscores and trim trailing ones so that every
    # call-site (struct definitions, field types, function parameters, …)
    # produces identical identifiers for the same C++ type.
    s = replace(s, r"_+" => "_")
    s = String(rstrip(s, '_'))
    if !isempty(s) && isdigit(s[1])
        s = "_" * s
    end
    if s in ("for", "if", "else", "while", "function", "struct", "end", "module", "using", "import", "export", "return", "continue", "break", "try", "catch", "finally", "macro", "quote", "let", "local", "global", "const", "do", "baremodule", "true", "false", "abstract", "type", "mutable", "primitive")
        s = "c_" * s
    end
    return s
end

"""
    _fuzzy_dwarf_lookup(c_type, dwarf_structs) -> Union{String, Nothing}

Fuzzy-match a C type name against DWARF struct definition keys.
DWARF keys for template types often have trailing " >" nesting artifacts.
"""
function _fuzzy_dwarf_lookup(c_type::AbstractString, dwarf_structs)
    haskey(dwarf_structs, c_type) && return String(c_type)
    key = c_type * " >"
    haskey(dwarf_structs, key) && return key
    c_norm = rstrip(c_type, [' ', '>'])
    for k in keys(dwarf_structs)
        rstrip(String(k), [' ', '>']) == c_norm && return String(k)
    end
    return nothing
end

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
        "intptr_t" => "Int64",
        "uintptr_t" => "UInt64",
        "off_t" => "Int64",
        "time_t" => "Int64",
        "clock_t" => "Int64",
    )

    # C++ STL types
    # Containers are mapped to Ptr{Cvoid} (opaque handles) since their ABI
    # is incompatible with Julia types. Use CppVector{T}/CppString for access.
    stl_types = Dict{String,String}(
        # String types - opaque handles (use CppString for access)
        "std::string" => "Ptr{Cvoid}",
        "std::basic_string<char>" => "Ptr{Cvoid}",
        "std::string_view" => "Ptr{Cvoid}",
        "std::basic_string_view<char>" => "Ptr{Cvoid}",

        # Containers - opaque handles (use CppVector{T} etc. for access)
        "std::vector" => "Ptr{Cvoid}",
        "std::array" => "Ptr{Cvoid}",
        "std::deque" => "Ptr{Cvoid}",
        "std::list" => "Ptr{Cvoid}",
        "std::forward_list" => "Ptr{Cvoid}",

        # Associative containers - opaque handles
        "std::map" => "Ptr{Cvoid}",
        "std::unordered_map" => "Ptr{Cvoid}",
        "std::multimap" => "Ptr{Cvoid}",
        "std::set" => "Ptr{Cvoid}",
        "std::unordered_set" => "Ptr{Cvoid}",
        "std::multiset" => "Ptr{Cvoid}",

        # Utility types (these can be POD-like)
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

"""
    is_stl_container_type(c_type::String)::Bool

Check if a C++ type is an STL container (non-POD, requires opaque handle).
"""
function is_stl_container_type(c_type::String)::Bool
    clean = strip(replace(c_type, r"\bconst\b" => ""))
    clean = strip(replace(clean, r"[*&]+$" => ""))
    clean = strip(clean)
    return any(p -> startswith(clean, p),
        ("std::vector", "std::basic_string", "std::string",
         "std::map", "std::unordered_map", "std::set", "std::unordered_set",
         "std::deque", "std::list", "std::forward_list",
         "std::multimap", "std::multiset"))
end

"""
    _split_template_args(args_str::String) -> Vector{String}

Split template arguments with bracket-aware parsing.
Handles nested templates like "std::map<std::string, std::vector<int>>".
"""
function _split_template_args(args_str::String)::Vector{String}
    result = String[]
    depth = 0
    current = IOBuffer()

    for c in args_str
        if c == '<'
            depth += 1
            write(current, c)
        elseif c == '>'
            depth -= 1
            write(current, c)
        elseif c == ',' && depth == 0
            push!(result, strip(String(take!(current))))
        else
            write(current, c)
        end
    end

    remaining = strip(String(take!(current)))
    if !isempty(remaining)
        push!(result, remaining)
    end

    return result
end

"""
    _is_stl_internal_type(name::String) -> Bool

Check if a DWARF struct name is an STL implementation-internal type that
should not be exposed to Julia users.
"""
function _is_stl_internal_type(name::String)::Bool
    # STL internal types from libstdc++ / libc++
    stl_internal_prefixes = (
        "_Alloc_hider", "_Guard", "_Guard_alloc", "_Storage",
        "_Temporary_value", "_UninitDestroyGuard", "_Vector_impl",
        "_Vector_base", "_Bvector", "_Deque_impl", "_List_impl",
        "_Rb_tree", "_Hashtable", "_Node_base", "_Node_alloc",
        "__gnu_cxx::", "std::_", "std::__", "__cxx",
        "allocator<", "char_traits<", "less<", "hash<", "equal_to<",
        "iterator<", "reverse_iterator<", "__normal_iterator<",
        "__wrap_iter<", "initializer_list<",
    )
    for prefix in stl_internal_prefixes
        if startswith(name, prefix)
            return true
        end
    end

    # Also filter types that are clearly STL containers themselves
    # (we handle them as opaque handles, not as Julia structs)
    if is_stl_container_type(name)
        return true
    end

    return false
end

"""
    _normalize_stl_for_dwarf(dwarf_name::String) -> String

Normalize a DWARF struct name for STL container matching.
DWARF names include allocator args: "std::vector<int, std::allocator<int>>"
We strip those to match the user-facing name: "std::vector<int>"
"""
function _normalize_stl_for_dwarf(dwarf_name::String)::String
    # std::vector<T, std::allocator<T>>
    m = match(r"^std::vector<(.+),\s*std::allocator<.+>>$", dwarf_name)
    if !isnothing(m)
        return "std::vector<$(strip(m.captures[1]))>"
    end
    # std::basic_string<char, ...>
    if startswith(dwarf_name, "std::basic_string<char")
        return "std::basic_string<char>"
    end
    return dwarf_name
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
            sanitized = _sanitize_julia_type_name(base_type)
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
        @warn """Unknown C/C++ type '$cpp_type' in $context, falling back to safe placeholder.
        This function will throw an error if called to prevent GC memory corruption.
        Consider adding a custom type mapping."""
        return "_UnsafeUnknown"
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
    
    # Strip C/C++ keywords that might come from DWARF
    clean_type = replace(clean_type, r"\bstruct\b" => "")
    clean_type = replace(clean_type, r"\bunion\b" => "")
    clean_type = replace(clean_type, r"\bclass\b" => "")
    
    clean_type = strip(clean_type)

    if clean_type in _INTERNAL_TYPE_BLOCKLIST
        return "Cvoid"
    end

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

        # Handle STL containers → Ptr{Cvoid} (opaque handles for JIT dispatch)
        if template_name in ("std::vector", "std::deque", "std::list", "std::forward_list",
                             "std::set", "std::unordered_set", "std::multiset",
                             "std::map", "std::unordered_map", "std::multimap",
                             "std::basic_string", "std::string")
            return "Ptr{Cvoid}"
        end

        # Handle std::array<T, N> → NTuple{N,T} or Ptr{Cvoid}
        if template_name == "std::array"
            parts = _split_template_args(template_args)
            if length(parts) >= 2
                arr_context = context == "" ? "std::array element type" : "$context (std::array element)"
                elem_type = infer_julia_type(registry, String(strip(parts[1])); context=arr_context)
                return "Ptr{Cvoid}"  # std::array has non-trivial ABI too
            end
        end

        # Handle std::pair<T1, T2> → Tuple{T1, T2} (POD-like, can be ccall'd)
        if template_name == "std::pair"
            parts = _split_template_args(template_args)
            if length(parts) == 2
                pair_ctx1 = context == "" ? "std::pair first type" : "$context (std::pair first)"
                pair_ctx2 = context == "" ? "std::pair second type" : "$context (std::pair second)"
                t1 = infer_julia_type(registry, String(strip(parts[1])); context=pair_ctx1)
                t2 = infer_julia_type(registry, String(strip(parts[2])); context=pair_ctx2)
                return "Tuple{$t1,$t2}"
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

    # Strip parameter list from demangled C++ names
    # e.g., "sum_vector(std::vector<int, ...> const&)" → "sum_vector"
    # Use bracket-aware parsing to find the first '(' at depth 0
    depth = 0
    paren_pos = 0
    for i in eachindex(name)
        c = name[i]
        if c == '<'
            depth += 1
        elseif c == '>'
            depth -= 1
        elseif c == '(' && depth == 0
            paren_pos = i
            break
        end
    end
    clean = paren_pos > 0 ? name[1:prevind(name, paren_pos)] : name

    # Remove [abi:cxx11] tags
    clean = replace(clean, r"\[abi:[^\]]*\]" => "")

    # Remove C++ namespace
    clean = replace(clean, r"^.*::" => "")

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


    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # Check for metadata (DWARF + symbol info from compilation)
    metadata_file = joinpath(dirname(library_path), "compilation_metadata.json")
    has_metadata = isfile(metadata_file)

    if !has_metadata
        @warn "No compilation metadata found. Did you compile with -g flag?"
        @warn "Falling back to basic symbol-only wrapper (conservative types, limited safety)"
        return wrap_basic(config, library_path, generate_docs=generate_docs)
    end

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

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # Create type registry
    registry = create_type_registry(config)

    # Extract symbols
    symbols = extract_symbols(library_path, registry, demangle=true, method=:nm)

    if isempty(symbols)
        @warn "No symbols found in library"
        return nothing
    end

    # Filter functions and data
    functions = filter(s -> s.symbol_type == :function, symbols)
    data_symbols = filter(s -> s.symbol_type == :data, symbols)

    # Generate wrapper module
    module_name = get_module_name(config)
    wrapper_content = generate_basic_module(config, library_path, functions, data_symbols,
                                           module_name, registry, generate_docs)

    # Write to file
    output_dir = get_output_path(config)
    mkpath(output_dir)
    output_file = joinpath(output_dir, "$(module_name).jl")

    write(output_file, wrapper_content)

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

    """

    content = header

    # Module declaration
    content *= "module $module_name\n\n"
    content *= "const Cintptr_t = Int\n"
    content *= "const Cuintptr_t = UInt\n\n"
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

    output_file = ClangJLBridge.generate_bindings_clangjl(clang_config, library_path, headers)

    if isnothing(output_file)
        error("Binding generation failed")
    end

    # TODO: Enhance generated file with our safety checks and metadata
    # For now, ClangJLBridge handles the generation

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

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # Load compilation metadata
    metadata_file = joinpath(dirname(library_path), "compilation_metadata.json")
    if !isfile(metadata_file)
        error("Compilation metadata not found: $metadata_file\nRun RepliBuild.build() first to generate metadata")
    end

    metadata = JSON.parsefile(metadata_file)

    if !haskey(metadata, "functions")
        error("Invalid metadata: missing 'functions' key")
    end

    functions = metadata["functions"]

    # Extract supplementary types from headers (enums, unused types, etc.)
    include_dirs = get(metadata, "include_dirs", String[])
    header_types = if !isempty(headers)
        ClangJLBridge.extract_header_types(headers, include_dirs)
    else
        # Auto-discover headers from include directories
        discovered_headers = String[]
        for inc_dir in include_dirs
            if isdir(inc_dir)
                append!(discovered_headers, ClangJLBridge.discover_headers(inc_dir, recursive=false))
            end
        end
        if !isempty(discovered_headers)
            ClangJLBridge.extract_header_types(discovered_headers, include_dirs)
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
    end

    # Store function pointer typedefs for callback documentation
    if haskey(header_types, "function_pointers") && !isempty(header_types["function_pointers"])
        metadata["function_pointer_typedefs"] = header_types["function_pointers"]
    end

    # Create type registry with metadata + typedef resolution table
    typedef_table = get(metadata, "typedef_table", Dict{String,Any}())
    # Convert to String,String for custom_types merge
    typedef_custom = Dict{String,String}(String(k) => String(v) for (k, v) in typedef_table if v != "Any")
    registry = create_type_registry(config, custom_types=typedef_custom)

    # AOT Compilation Pass
    thunks_lib_path = ""
    if config.compile.aot_thunks
        output_dir = get_output_path(config)
        lib_name = basename(library_path)
        thunks_name = replace(lib_name, ".so" => "_thunks.so", ".dylib" => "_thunks.dylib", ".dll" => "_thunks.dll")
        thunks_so = joinpath(output_dir, thunks_name)
        if isfile(thunks_so)
            thunks_lib_path = abspath(thunks_so)
        else
            @warn "AOT thunks enabled but companion library not found at $thunks_so"
        end
    end

    # Generate wrapper module
    module_name = get_module_name(config)
    wrapper_content = generate_introspective_module(config, library_path, metadata,
                                                    module_name, registry, generate_docs, thunks_lib_path)

    # Write to file
    output_dir = get_output_path(config)
    mkpath(output_dir)
    output_file = joinpath(output_dir, "$(module_name).jl")

    write(output_file, wrapper_content)

    println("  wrap: $(basename(output_file))")

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
    # 0. Check for STL container types (never ccall-safe)
    ret_type_str = String(get(func_info["return_type"], "c_type", ""))
    if is_stl_container_type(ret_type_str)
        return false
    end
    for param in func_info["parameters"]
        if is_stl_container_type(get(param, "c_type", ""))
            return false
        end
    end

    # 1. Check Return Type
    ret_type = ret_type_str

    # If returning a struct by value (not pointer/void/primitive)
    # For template types (containing '<'), skip primitive substring matching —
    # "Matrix<double, -1, -1>" contains "double" but is NOT a double return.
    is_template_ret = occursin('<', ret_type)
    is_primitive_ret = !is_template_ret &&
        (contains(ret_type, "int") || contains(ret_type, "float") ||
         contains(ret_type, "double") || contains(ret_type, "bool"))
    if !contains(ret_type, "*") && !contains(ret_type, "void") && !is_primitive_ret

        # Template return types (e.g. Matrix<double,-1,-1>) are always complex
        # struct returns — route to MLIR unconditionally.  DWARF may not have
        # an entry for the exact template instantiation.
        if is_template_ret
            return false
        end

        # Check if it's a known struct
        if haskey(dwarf_structs, ret_type)
            s_info = dwarf_structs[ret_type]

            # Struct return by value is notoriously fragile in ABIs (large structs split registers)
            # Conservative: Send to MLIR if > 16 bytes
            if parse(Int, get(s_info, "byte_size", "0")) > 16
                return false
            end

            # Check if it is a class (likely non-POD)
            if get(s_info, "kind", "struct") == "class"
                return false
            end

            # CHECK: Is return type a packed struct?
            # Packed struct returns use sret ABI which Julia ccall doesn't handle correctly
            dwarf_size = parse(Int, get(s_info, "byte_size", "0"))
            members = get(s_info, "members", [])
            julia_size = get_julia_aligned_size(members)
            if dwarf_size > 0 && dwarf_size != julia_size
                return false
            end
        end
    end

    # 2. Check Arguments
    for param in func_info["parameters"]
        c_type = get(param, "c_type", "")

        # If the parameter is passed by pointer (contains *), it's always ccall-safe
        # Only by-value struct parameters need special handling
        if contains(c_type, "*")
            continue
        end

        # Clean const prefix for base type lookup
        base_type = String(strip(replace(c_type, "const" => "")))

        if haskey(dwarf_structs, base_type)
            s_info = dwarf_structs[base_type]

            # CHECK A: Is it a Union?
            if get(s_info, "kind", "struct") == "union"
                return false
            end

            # CHECK B: Is it Packed?
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
    generate_vararg_wrappers(func_name, mangled, julia_name, params, return_type, overloads, generate_docs, demangled) -> (code, export_names)

Generate typed overload wrappers for a variadic C function.
Julia's `ccall` requires fixed signatures, so we generate:
- A base wrapper with only the fixed (non-variadic) parameters
- Typed overloads for each signature listed in `overloads`
All varargs wrappers use direct ccall (Tier 1), never JIT.
"""
function generate_vararg_wrappers(func_name::String, mangled::String, julia_name::String,
                                  params::Vector, return_type,
                                  overloads::Vector{Vector{String}},
                                  generate_docs::Bool, demangled::String)
    code = ""
    export_names = String[]

    # Build fixed parameter info
    fixed_param_names = String[]
    fixed_param_types = String[]  # Julia/ccall types
    fixed_julia_sig_types = String[]  # Ergonomic types for signature
    fixed_needs_conversion = Bool[]

    for param in params
        safe_name = make_julia_identifier(param["name"])
        if safe_name == "varargs..."
            continue  # Skip the varargs placeholder
        end
        push!(fixed_param_names, safe_name)

        julia_type = param["julia_type"]
        push!(fixed_param_types, julia_type)

        # Ergonomic type mapping (same logic as main wrapper gen)
        if julia_type in ["Cint", "Clong", "Cshort"]
            push!(fixed_julia_sig_types, "Integer")
            push!(fixed_needs_conversion, true)
        elseif startswith(julia_type, "Ptr{")
            push!(fixed_julia_sig_types, "Any")
            push!(fixed_needs_conversion, false)
        else
            push!(fixed_julia_sig_types, julia_type)
            push!(fixed_needs_conversion, false)
        end
    end

    julia_return_type = get(return_type, "julia_type", "Cvoid")

    # Build fixed parameter signature
    fixed_sig_parts = ["$(n)::$(t)" for (n, t) in zip(fixed_param_names, fixed_julia_sig_types)]
    fixed_sig = join(fixed_sig_parts, ", ")

    # Build conversion code for fixed params
    fixed_conversion = ""
    fixed_ccall_names = String[]
    for (name, ctype, needs_conv) in zip(fixed_param_names, fixed_param_types, fixed_needs_conversion)
        if needs_conv
            converted = "$(name)_c"
            push!(fixed_ccall_names, converted)
            fixed_conversion *= "    $converted = $ctype($name)\n"
        else
            push!(fixed_ccall_names, name)
        end
    end

    # --- Base wrapper (fixed args only) ---
    fixed_ccall_types = if isempty(fixed_param_types)
        "()"
    else
        "($(join(fixed_param_types, ", ")),)"
    end
    fixed_ccall_args = join(fixed_ccall_names, ", ")

    doc = ""
    if generate_docs
        doc = """
        \"\"\"
            $julia_name($fixed_sig) -> $julia_return_type

        Wrapper for variadic C function: `$demangled` (base call with fixed args only)
        \"\"\"
        """
    end

    code *= """
    $doc
    function $julia_name($fixed_sig)::$julia_return_type
    $fixed_conversion    return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $fixed_ccall_types, $fixed_ccall_args)
    end

    """
    push!(export_names, julia_name)

    # --- Typed overloads ---
    for va_types in overloads
        # Build overload function name: fname_Type1_Type2
        type_suffix = join(va_types, "_")
        overload_name = "$(julia_name)_$(type_suffix)"

        # Build variadic parameter names and types
        va_param_names = ["va_$(i)" for i in 1:length(va_types)]
        va_sig_parts = ["$(n)::$(t)" for (n, t) in zip(va_param_names, va_types)]

        # Full signature = fixed + variadic
        all_sig_parts = vcat(fixed_sig_parts, va_sig_parts)
        all_sig = join(all_sig_parts, ", ")

        # ccall types = fixed + variadic (with Vararg marker for proper ABI)
        all_ccall_types_list = vcat(fixed_param_types, va_types)
        all_ccall_types = "($(join(all_ccall_types_list, ", ")),)"

        # ccall args = fixed converted + variadic
        all_ccall_names = vcat(fixed_ccall_names, va_param_names)
        all_ccall_args = join(all_ccall_names, ", ")

        overload_doc = ""
        if generate_docs
            overload_doc = """
            \"\"\"
                $overload_name($all_sig) -> $julia_return_type

            Typed variadic overload for: `$demangled`
            Variadic types: $(join(va_types, ", "))
            \"\"\"
            """
        end

        code *= """
        $overload_doc
        function $overload_name($all_sig)::$julia_return_type
        $fixed_conversion    return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $all_ccall_types, $all_ccall_args)
        end

        """
        push!(export_names, overload_name)
    end

    return (code, export_names)
end

"""
Generate introspective wrapper module content using compilation metadata.
"""
function generate_introspective_module(config::RepliBuildConfig, lib_path::String,
                                      metadata, module_name::String,
                                      registry::TypeRegistry, generate_docs::Bool,
                                      thunks_lib_path::String="")

    # Track exported symbols
    exports = String[]

    # Header with metadata
    header = """
    # Auto-generated Julia wrapper for $(config.project.name)
    # Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    # Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
    # Library: $(basename(lib_path))
    # Metadata: compilation_metadata.json

    module $module_name
    
    const Cintptr_t = Int
    const Cuintptr_t = UInt

    using Libdl
    import RepliBuild
    import Base: unsafe_convert

    const LIBRARY_PATH = \"$(abspath(lib_path))\"
    const THUNKS_LIBRARY_PATH = \"$(thunks_lib_path)\"

    # Verify library exists
    if !isfile(LIBRARY_PATH)
        error("Library not found: \$LIBRARY_PATH")
    end

    """

    # Track if JIT is required (for virtual methods or complex ABI)
    requires_jit = false

    # Metadata section
    compiler_info = get(metadata, "compiler_info", Dict())
    lto_name = config.project.name
    lto_ir_block = config.link.enable_lto ? """
    # LTO: load LLVM bitcode at module parse time for Base.llvmcall zero-cost dispatch
    # Using bitcode (.bc) is significantly faster for Julia to parse than text IR (.ll)
    const LTO_IR_PATH = joinpath(@__DIR__, "$(lto_name)_lto.bc")
    const LTO_IR = isfile(LTO_IR_PATH) ? read(LTO_IR_PATH) : UInt8[]
    const THUNKS_LTO_IR_PATH = joinpath(@__DIR__, "$(lto_name)_thunks_lto.bc")
    const THUNKS_LTO_IR = isfile(THUNKS_LTO_IR_PATH) ? read(THUNKS_LTO_IR_PATH) : UInt8[]

    """ : """
    const LTO_IR = ""  # LTO disabled for this build
    const THUNKS_LTO_IR = ""

    """
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

    """ * lto_ir_block

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
    # AUTOMATIC FINALIZER GENERATION (Memory Safety)
    # =============================================================================

    # 1. Scan for Deleters
    # Map: StructName -> DeleterFunctionName
    deleters = Dict{String, String}()
    deleters_mangled = Dict{String, String}()  # struct_name -> mangled symbol

    for func in functions
        f_name = func["name"]
        demangled = func["demangled"]
        params = func["parameters"]
        
        # Criteria: 1 arg, arg is Ptr{Struct}, name implies deletion
        if length(params) == 1
            arg_type = params[1]["julia_type"]
            if startswith(arg_type, "Ptr{") && endswith(arg_type, "}")
                struct_name = arg_type[5:end-1]
                # Check if it's a known struct
                if struct_name in struct_types
                    # Check name patterns
                    lower_name = lowercase(f_name)
                    lower_demangled = lowercase(demangled)
                    
                    is_destructor = contains(demangled, "~")
                    is_deleter = contains(lower_name, "delete") || contains(lower_name, "destroy") || contains(lower_name, "free")
                    
                    if is_destructor || is_deleter
                        # Found a deleter for struct_name
                        # Prefer destructors over generic deleters if duplicates?
                        # For now, just take the first one
                        if !haskey(deleters, struct_name)
                            deleters[struct_name] = f_name
                            deleters_mangled[struct_name] = func["mangled"]
                        end
                    end
                end
            end
        end
    end

    # =============================================================================
    # IDIOMATIC CLASS WRAPPER PREPARATION
    # =============================================================================

    # Collect factory functions: free functions named create_/new_/make_/alloc_/init_/build_
    # that return a pointer to a known C++ class.  The class name is inferred via
    # (priority order): typed Ptr return type, c_type pointer base, name suffix.
    factory_funcs = Dict{String, Vector{Any}}()  # inferred_class_name -> [func_info, ...]
    for func in functions
        is_m = get(func, "is_method", false)
        is_m && continue
        f_name = func["name"]
        f_lower = lowercase(f_name)
        is_factory_name = (startswith(f_lower, "create_") || startswith(f_lower, "new_") ||
                           startswith(f_lower, "make_")   || startswith(f_lower, "alloc_") ||
                           startswith(f_lower, "init_")   || startswith(f_lower, "build_"))
        is_factory_name || continue

        ret       = func["return_type"]
        ret_julia = get(ret, "julia_type", "")
        ret_c     = get(ret, "c_type",     "")
        class_nm  = nothing

        # Strategy 1: infer from function name prefix (create_circle -> Circle)
        # This is more accurate for polymorphism where create_circle() returns Shape*
        if isnothing(class_nm)
            for prefix in ["create_", "new_", "make_", "alloc_", "init_", "build_"]
                if startswith(f_lower, prefix)
                    raw_suffix = f_name[length(prefix)+1:end]
                    if !isempty(raw_suffix)
                        # Split by underscore to handle snake_case to CamelCase (e.g. create_dense_matrix -> DenseMatrix)
                        parts = split(raw_suffix, "_")
                        class_nm = join([uppercasefirst(p) for p in parts])
                    end
                    break
                end
            end
        end

        # Strategy 2: typed pointer return (Ptr{X} where X != Cvoid)
        if isnothing(class_nm) && startswith(ret_julia, "Ptr{") && endswith(ret_julia, "}") && ret_julia != "Ptr{Cvoid}"
            class_nm = ret_julia[5:end-1]
        end

        # Strategy 3: c_type ends in * (pointer to a named type)
        if isnothing(class_nm) && endswith(ret_c, "*")
            base_c = strip(ret_c[1:end-1])
            base_c = replace(base_c, r"^(const\s+|struct\s+|class\s+)+" => "")
            base_c = replace(base_c, r"\s+" => "")
            if !isempty(base_c) && base_c != "void"
                class_nm = base_c
            end
        end

        if !isnothing(class_nm) && !isempty(string(class_nm))
            safe_nm = _sanitize_julia_type_name(string(class_nm))
            push!(get!(factory_funcs, safe_nm, Any[]), func)
        end
    end

    # Collect C++ constructors: is_method=true AND func_name == class_name
    class_constructors = Dict{String, Vector{Any}}()  # class_name -> [ctor_func_info, ...]
    for func in functions
        is_m   = get(func, "is_method", false)
        cls_nm = get(func, "class", "")
        is_m && !isempty(cls_nm) && func["name"] == cls_nm || continue
        safe_cls = _sanitize_julia_type_name(cls_nm)
        push!(get!(class_constructors, safe_cls, Any[]), func)
    end

    # Collect C++ destructors: is_method=true AND demangled contains "~"
    class_destructors = Dict{String, Any}()  # class_name -> dtor_func_info (first only)
    for func in functions
        is_m   = get(func, "is_method", false)
        cls_nm = get(func, "class", "")
        is_m && !isempty(cls_nm) && contains(func["demangled"], "~") || continue
        safe_cls = _sanitize_julia_type_name(cls_nm)
        haskey(class_destructors, safe_cls) || (class_destructors[safe_cls] = func)
    end

    # Collect non-constructor, non-destructor instance methods
    class_methods = Dict{String, Vector{Any}}()  # class_name -> [method_func_info, ...]
    for func in functions
        is_m   = get(func, "is_method", false)
        cls_nm = get(func, "class", "")
        is_m && !isempty(cls_nm) || continue
        func["name"] == cls_nm   && continue  # skip constructors
        contains(func["demangled"], "~") && continue  # skip destructors
        safe_cls = _sanitize_julia_type_name(cls_nm)
        push!(get!(class_methods, safe_cls, Any[]), func)
    end

    # Helper: sanitize a raw function name to the Julia identifier used in wrappers
    function _sanitize_julia_fn(name::String)::String
        n = replace(name, "~"  => "destroy_")
        n = replace(n, "::" => "_")
        n = replace(n, "<"  => "_")
        n = replace(n, ">"  => "_")
        n = replace(n, ","  => "_")
        n = replace(n, " "  => "_")
        n = replace(n, "+"  => "plus")
        n = replace(n, "="  => "assign")
        n = replace(n, "-"  => "minus")
        n = replace(n, "*"  => "mul")
        n = replace(n, "/"  => "div")
        return n
    end

    # Determine which classes get full idiomatic wrappers.
    # Eligibility: has ≥1 factory function (pointer-returning) AND a resolvable deleter.
    # Pure C++ constructor-only classes (no factory) are skipped — they require
    # operator new + placement-new which needs sizeof(Class); left as raw bindings.
    idiomatic_classes = Dict{String, Dict{String,Any}}()  # class_name -> config dict

    for cls_nm in keys(factory_funcs)
        factories = factory_funcs[cls_nm]
        deleter_jl = ""

        # 1. Direct match in deleters dict
        if haskey(deleters, cls_nm)
            raw_del = deleters[cls_nm]
            is_del_method = any(f -> f["name"] == raw_del && get(f, "is_method", false), functions)
            deleter_jl = is_del_method ? "$(cls_nm)_$(_sanitize_julia_fn(raw_del))" : _sanitize_julia_fn(raw_del)
        end

        # 2. DWARF-detected C++ destructor for the same class
        if isempty(deleter_jl) && haskey(class_destructors, cls_nm)
            dtor = class_destructors[cls_nm]
            deleter_jl = "$(cls_nm)_$(_sanitize_julia_fn(dtor["name"]))"
        end

        # 3. Factory return c_type base class points to a known deleter
        #    (e.g. create_circle returns Shape* -> deleters["Shape"] = "delete_shape")
        if isempty(deleter_jl)
            for fac_func in factories
                ret_c = get(fac_func["return_type"], "c_type", "")
                endswith(ret_c, "*") || continue
                base_c = strip(ret_c[1:end-1])
                base_c = replace(base_c, r"^(const\s+|struct\s+|class\s+)+" => "")
                base_c = replace(base_c, r"\s+" => "")
                if !isempty(base_c) && base_c != "void" && haskey(deleters, base_c)
                    raw_del = deleters[base_c]
                    is_del_method = any(f -> f["name"] == raw_del && get(f, "is_method", false), functions)
                    deleter_jl = is_del_method ? "$(base_c)_$(_sanitize_julia_fn(raw_del))" : _sanitize_julia_fn(raw_del)
                    break
                end
            end
        end

        isempty(deleter_jl) && continue  # no deleter found — skip, keep raw bindings only

        idiomatic_classes[cls_nm] = Dict{String,Any}(
            "deleter"   => deleter_jl,
            "factories" => factories,
            "ctors"     => get(class_constructors, cls_nm, Any[]),
        )
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

                seen_values = Set{Any}()
                seen_names = Set{String}()
                duplicate_defs = String[]
                for (i, enumerator) in enumerate(enumerators)
                    name = get(enumerator, "name", "Unknown")
                    name = _sanitize_julia_type_name(name)
                    value = get(enumerator, "value", 0)
                    name = _escape_keyword(name)
                    
                    if name in seen_names
                        continue
                    end
                    push!(seen_names, name)

                    if value in seen_values
                        push!(duplicate_defs, "const $name = $enum_name($value)")
                    else
                        push!(seen_values, value)
                        enum_definitions *= "    $name = $value\n"
                    end
                end

                enum_definitions *= """
                end

                """
                if !isempty(duplicate_defs)
                    enum_definitions *= join(duplicate_defs, "\n") * "\n\n"
                end
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

                # Choose underlying type based on value range
                min_val = minimum(v for (_, v) in members)
                max_val = maximum(v for (_, v) in members)
                underlying = if min_val >= 0 && max_val > typemax(Int32)
                    "UInt32"
                elseif min_val < typemin(Int32) || max_val > typemax(Int32)
                    "Int64"
                else
                    "Cint"
                end

                enum_definitions *= """
                # C++ enum: $enum_name (from header - not in DWARF)
                @enum $enum_name::$underlying begin
                """

                seen_values = Set{Any}()
                seen_names = Set{String}()
                duplicate_defs = String[]
                for (member_name, value) in members
                    member_name = _sanitize_julia_type_name(string(member_name))
                    member_name = _escape_keyword(member_name)
                    
                    if member_name in seen_names
                        continue
                    end
                    push!(seen_names, member_name)

                    if value in seen_values
                        push!(duplicate_defs, "const $member_name = $enum_name($value)")
                    else
                        push!(seen_values, value)
                        enum_definitions *= "    $member_name = $value\n"
                    end
                end

                enum_definitions *= """
                end

                """
                if !isempty(duplicate_defs)
                    enum_definitions *= join(duplicate_defs, "\n") * "\n\n"
                end
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

    # Create a mapping from sanitized Julia name back to raw C++ name for resolving dependencies
    julia_to_cpp_struct = Dict{String, String}()
    for struct_name in struct_types
        julia_to_cpp_struct[_sanitize_julia_type_name(struct_name)] = struct_name
    end

    # Scan for ALL referenced struct types that need forward declarations.
    # This includes:
    # 1. Types referenced via Ptr{T} that have no DWARF definition (truly opaque)
    # 2. Types referenced via Ptr{T} that DO have definitions but may appear later
    #    in topological order (forward reference problem)
    # We use `mutable struct Foo end` for truly opaque types, and for defined types
    # we don't forward-declare (they'll be defined in topo order).
    opaque_structs = Set{String}()
    ptr_referenced_structs = Set{String}()  # Struct types referenced via Ptr{X}

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

                # Scan for ALL struct type references in this julia_type
                # Handles: Ptr{Foo}, NTuple{N, Ptr{Foo}}, Ptr{Ptr{Foo}}, Ref{Foo}, etc.
                builtin_types = Set(["Cvoid", "Cint", "Cuint", "Clong", "Culong", "Cshort", "Cushort",
                                     "Cchar", "Cuchar", "Cfloat", "Cdouble", "Bool", "UInt8", "Int8",
                                     "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64", "Csize_t",
                                     "Clonglong", "Culonglong", "Cptrdiff_t", "Cssize_t", "Cwchar_t",
                                     "Cstring", "Float32", "Float64", "Any", "Nothing"])

                # Extract the base type by stripping known container prefixes
                base_ref = julia_type
                while true
                    if startswith(base_ref, "Ptr{") && endswith(base_ref, "}")
                        base_ref = base_ref[5:end-1]
                    elseif startswith(base_ref, "Ref{") && endswith(base_ref, "}")
                        base_ref = base_ref[5:end-1]
                    elseif startswith(base_ref, "NTuple{")
                        ntuple_match = match(r"NTuple\{\d+,\s*([^}]+)\}", base_ref)
                        if !isnothing(ntuple_match)
                            base_ref = strip(ntuple_match.captures[1])
                        else
                            break
                        end
                    else
                        break
                    end
                end
                
                base_ref = strip(base_ref)
                
                # If the inner type is not a builtin, register it
                if !(base_ref in builtin_types) && !isempty(base_ref)
                    # We compare the raw string with struct_types
                    if base_ref in struct_types
                        push!(ptr_referenced_structs, base_ref)
                    else
                        push!(opaque_structs, base_ref)
                    end
                end
            end
        end
    end

    struct_definitions = ""

    # Emit forward declarations for:
    # 1. Truly opaque types (no DWARF definition) — as mutable struct
    # 2. Struct types referenced via Ptr{X} — as empty struct (will be redefined with fields later)
    # This ensures Ptr{X} fields can reference types defined later in the file.
    # BUT: skip forward declarations for types that will get a real DWARF definition,
    # since Julia doesn't allow redefining a struct with the same name.
    all_forward_decls = union(opaque_structs, ptr_referenced_structs)

    # Build set of sanitized names that will get real DWARF definitions
    dwarf_defined_names = Set{String}()
    for (name, info) in dwarf_structs
        _is_stl_internal_type(name) && continue
        s = _sanitize_julia_type_name(name)
        s = replace(s, "*" => "Ptr")
        s = replace(s, "&" => "Ref")
        s = replace(s, r"[^a-zA-Z0-9_]" => "_")
        push!(dwarf_defined_names, s)
    end

    if !isempty(all_forward_decls)
        struct_definitions *= """
        # =============================================================================
        # Forward Declarations (Opaque + Ptr-referenced types)
        # =============================================================================

        """
        seen_forward_decls = Set{String}()
        for name in sort(collect(all_forward_decls))
            # Skip STL internal types
            _is_stl_internal_type(name) && continue

            # Sanitize
            s_name = _sanitize_julia_type_name(name)
            s_name = replace(s_name, "*" => "Ptr")
            s_name = replace(s_name, "&" => "Ref")
            s_name = replace(s_name, r"[^a-zA-Z0-9_]" => "_")

            # Skip names that are Julia keywords or builtins
            if s_name in ("char", "int", "void", "bool", "float", "double", "long", "short")
                continue
            end

            # Skip duplicates (different C++ names can sanitize to the same Julia identifier)
            s_name in seen_forward_decls && continue
            push!(seen_forward_decls, s_name)

            # Skip forward declaration if this type will get a real DWARF definition later
            s_name in dwarf_defined_names && continue

            # Both opaque types and pointer-referenced types should be immutable structs
            # to preserve inline C++ ABI layout (0-byte empty structs) if they are used by value.
            struct_definitions *= "struct $s_name end\n"
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
        struct_soft_deps = Dict{String, Set{String}}()

        for struct_name in struct_types
            deps = Set{String}()
            soft_deps = Set{String}()

            if haskey(dwarf_structs, struct_name)
                struct_info = dwarf_structs[struct_name]
                members = get(struct_info, "members", [])

                for member in members
                    julia_type = get(member, "julia_type", "Any")

                    # Extract type name from various wrapper types:
                    #   Ptr{Foo} -> Foo (soft dep)
                    #   NTuple{N, Foo} -> Foo (hard dep, needs full definition first)
                    #   Ref{Foo} -> Foo (hard dep)
                    #   Foo -> Foo (hard dep, direct by-value embedding)
                    dep_type = nothing
                    is_soft = false

                    if startswith(julia_type, "Ptr{")
                        ptr_match = match(r"Ptr\{([^}]+)\}", julia_type)
                        if !isnothing(ptr_match)
                            dep_type = strip(ptr_match.captures[1])
                            is_soft = true
                        end
                    elseif startswith(julia_type, "NTuple{")
                        ntuple_match = match(r"NTuple\{\d+,\s*([^}]+)\}", julia_type)
                        if !isnothing(ntuple_match)
                            dep_type = strip(ntuple_match.captures[1])
                        end
                    elseif startswith(julia_type, "Ref{")
                        ref_match = match(r"Ref\{([^}]+)\}", julia_type)
                        if !isnothing(ref_match)
                            dep_type = strip(ref_match.captures[1])
                        end
                    else
                        dep_type = julia_type
                    end

                    if !isnothing(dep_type) && haskey(julia_to_cpp_struct, dep_type)
                        cpp_dep = julia_to_cpp_struct[dep_type]
                        if cpp_dep != struct_name
                            if is_soft
                                push!(soft_deps, cpp_dep)
                            else
                                push!(deps, cpp_dep)
                            end
                        end
                    end
                end
            end

            struct_deps[struct_name] = deps
            struct_soft_deps[struct_name] = soft_deps
        end

        # Topological sort using Kahn's algorithm
        sorted_structs = String[]
        remaining_hard = Dict(k => copy(v) for (k, v) in struct_deps)
        remaining_soft = Dict(k => copy(v) for (k, v) in struct_soft_deps)

        while !isempty(remaining_hard)
            # Find structs with no hard AND no soft dependencies
            ready = [name for (name, deps) in remaining_hard if isempty(deps) && isempty(remaining_soft[name])]

            if isempty(ready)
                # Break soft dependency cycle: find structs with no hard dependencies
                ready = [name for (name, deps) in remaining_hard if isempty(deps)]
            end

            if isempty(ready)
                # Circular hard dependency - just take alphabetically first
                ready = [sort(collect(keys(remaining_hard)))[1]]
            end

            for name in sort(ready)
                push!(sorted_structs, name)
                delete!(remaining_hard, name)
                delete!(remaining_soft, name)

                # Remove this struct from all dependency lists
                for deps in values(remaining_hard)
                    delete!(deps, name)
                end
                for deps in values(remaining_soft)
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

        # Pre-merge DWARF keys that sanitize to the same Julia name.
        # Multiple C++ template nesting depths (e.g., "Matrix<double,-1,-1> >" vs
        # "Matrix<double,-1,-1> > >") sanitize to the same identifier.
        # Keep the one with the largest byte_size and most members.
        best_dwarf_key = Dict{String, String}()  # sanitized_name -> best struct_name
        for struct_name in sorted_structs
            struct_name in enum_names && continue
            _is_stl_internal_type(struct_name) && continue
            
            jname = _sanitize_julia_type_name(struct_name)
            jname = replace(jname, "*" => "Ptr")
            jname = replace(jname, "&" => "Ref")
            jname = replace(jname, r"[^a-zA-Z0-9_]" => "_")
            
            if !haskey(best_dwarf_key, jname)
                best_dwarf_key[jname] = struct_name
            else
                old_key = best_dwarf_key[jname]
                if haskey(dwarf_structs, struct_name) && haskey(dwarf_structs, old_key)
                    old_info = dwarf_structs[old_key]
                    new_info = dwarf_structs[struct_name]
                    old_bs = try; s = get(old_info, "byte_size", "0"); startswith(string(s), "0x") ? parse(Int, string(s)[3:end], base=16) : parse(Int, string(s)); catch; 0; end
                    new_bs = try; s = get(new_info, "byte_size", "0"); startswith(string(s), "0x") ? parse(Int, string(s)[3:end], base=16) : parse(Int, string(s)); catch; 0; end
                    old_mc = length(get(old_info, "members", []))
                    new_mc = length(get(new_info, "members", []))
                    if new_bs > old_bs || (new_bs == old_bs && new_mc > old_mc)
                        best_dwarf_key[jname] = struct_name
                    end
                elseif haskey(dwarf_structs, struct_name) && !haskey(dwarf_structs, old_key)
                    best_dwarf_key[jname] = struct_name
                end
            end
        end

        seen_struct_defs = Set{String}()
        defined_struct_names = Set{String}()
        if @isdefined(seen_forward_decls)
            # Only seed with forward decls that were actually emitted (not DWARF-defined skips)
            for s in seen_forward_decls
                if !(s in dwarf_defined_names)
                    push!(defined_struct_names, s)
                end
            end
        end
        blob_struct_names = Set{String}()  # Track which structs became byte-blobs
        for struct_name in sorted_structs
            # Skip if this is actually an enum (enums are generated separately)
            if struct_name in enum_names
                continue
            end

            # Skip STL internal implementation types (they leak from DWARF when
            # template instantiation is forced, but are not useful to Julia users)
            if _is_stl_internal_type(struct_name)
                continue
            end

            # Skip compiler/libc internal types
            if struct_name in _INTERNAL_TYPE_BLOCKLIST
                continue
            end

            # Sanitize struct name for Julia (replace < > , with _)
            julia_struct_name = _sanitize_julia_type_name(struct_name)
            julia_struct_name = replace(julia_struct_name, "*" => "Ptr")
            julia_struct_name = replace(julia_struct_name, "&" => "Ref")
            julia_struct_name = replace(julia_struct_name, r"[^a-zA-Z0-9_]" => "_")

            # Skip duplicate sanitized names — only process the "best" DWARF key
            # (the one with largest byte_size / most members)
            if julia_struct_name in seen_struct_defs
                continue
            end
            if haskey(best_dwarf_key, julia_struct_name) && best_dwarf_key[julia_struct_name] != struct_name
                continue
            end
            push!(seen_struct_defs, julia_struct_name)
            push!(defined_struct_names, julia_struct_name)

            # Skip DWARF struct generation if we are generating a high-level idiomatic wrapper
            if haskey(idiomatic_classes, julia_struct_name)
                continue
            end

            # Check if we have DWARF member information for this struct
            if haskey(dwarf_structs, struct_name)
                struct_info = dwarf_structs[struct_name]
                kind = get(struct_info, "kind", "struct")
                
                # SPECIAL HANDLING FOR UNIONS
                if kind == "union"
                    byte_size_str = get(struct_info, "byte_size", "0x0")
                    byte_size = parse(Int, byte_size_str)
                    members = get(struct_info, "members", [])

                    if byte_size == 0
                        # Fallback if size missing
                        for m in members
                            m_size = get(m, "size", 0)
                            byte_size = max(byte_size, m_size)
                        end
                        if byte_size == 0; byte_size = 8; end # Panic fallback
                    end

                    struct_definitions *= """
                    # C union: $struct_name (size $byte_size bytes)
                    mutable struct $julia_struct_name
                        data::NTuple{$byte_size, UInt8}
                    end
                    $julia_struct_name() = $julia_struct_name(ntuple(i -> 0x00, $byte_size))

                    """

                    # Generate typed accessor functions for each union member
                    for m in members
                        m_name = get(m, "name", "")
                        m_julia_type = get(m, "julia_type", "Any")
                        if isempty(m_name) || m_julia_type == "Any"
                            continue
                        end

                        # Sanitize member name for Julia identifier
                        safe_m_name = replace(m_name, r"[^A-Za-z0-9_]" => "_")

                        # Skip complex types (nested structs, arrays) — only primitive/pointer accessors
                        if startswith(m_julia_type, "NTuple{") || startswith(m_julia_type, "Vector{")
                            continue
                        end

                        struct_definitions *= """
                        \"\"\"Get union member `$m_name` as `$m_julia_type` from `$julia_struct_name`.\"\"\"
                        function get_$(safe_m_name)(u::$julia_struct_name)::$m_julia_type
                            return unsafe_load(Ptr{$m_julia_type}(pointer_from_objref(u)))
                        end

                        \"\"\"Set union member `$m_name` as `$m_julia_type` in `$julia_struct_name`.\"\"\"
                        function set_$(safe_m_name)!(u::$julia_struct_name, v::$m_julia_type)
                            unsafe_store!(Ptr{$m_julia_type}(pointer_from_objref(u)), v)
                        end

                        """
                        push!(exports, "get_$(safe_m_name)")
                        push!(exports, "set_$(safe_m_name)!")
                    end

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

                    # Check if this struct has bitfield members
                    has_bitfields = any(m -> haskey(m, "bit_size"), members)

                    if has_bitfields
                        byte_size_str = get(struct_info, "byte_size", "0x0")
                        byte_size = parse(Int, byte_size_str)
                        if byte_size == 0; byte_size = 8; end

                        struct_definitions *= """
                        # C struct with bitfields: $struct_name (size $byte_size bytes)
                        mutable struct $julia_struct_name
                            _data::NTuple{$byte_size, UInt8}
                        end
                        $julia_struct_name() = $julia_struct_name(ntuple(i -> 0x00, $byte_size))

                        """

                        # Generate accessor functions for each member
                        for member in members
                            member_name = get(member, "name", "unknown")
                            julia_type = get(member, "julia_type", "Any")
                            safe_member = replace(member_name, r"[^A-Za-z0-9_]" => "_")

                            # Sanitize julia_type to avoid < > in function signatures
                            sanitized_type = julia_type
                            if occursin(r"[<>]", julia_type)
                                if occursin(r"Ptr\{[^}]+\}", julia_type)
                                    type_match = match(r"Ptr\{([^}]+)\}", julia_type)
                                    if !isnothing(type_match)
                                        inner_type = String(type_match.captures[1])
                                        if _is_stl_internal_type(inner_type)
                                            sanitized_type = "Ptr{Cvoid}"
                                        else
                                            sanitized_type = "Ptr{$(_sanitize_julia_type_name(inner_type))}"
                                        end
                                    end
                                else
                                    if _is_stl_internal_type(julia_type)
                                        m_size = get(member, "size", 0)
                                        sanitized_type = m_size > 0 ? "NTuple{$m_size, UInt8}" : "Ptr{Cvoid}"
                                    else
                                        sanitized_type = _sanitize_julia_type_name(julia_type)
                                    end
                                end
                            end
                            julia_type = sanitized_type

                            if haskey(member, "bit_size")
                                bit_size = member["bit_size"]

                                # Determine absolute bit offset
                                bit_offset = if haskey(member, "data_bit_offset")
                                    member["data_bit_offset"]
                                elseif haskey(member, "bit_offset_legacy")
                                    byte_off = parse(Int, get(member, "offset", "0x0"))
                                    byte_off * 8 + member["bit_offset_legacy"]
                                else
                                    0
                                end

                                byte_pos = bit_offset >> 3
                                bit_within_byte = bit_offset & 7
                                mask = (1 << bit_size) - 1

                                # Getter with bit extraction
                                if bit_size <= 8 && bit_within_byte + bit_size <= 8
                                    # Single byte access
                                    struct_definitions *= """
                                    \"\"\"Get bitfield `$member_name` ($bit_size bits) from `$julia_struct_name`.\"\"\"
                                    function get_$(safe_member)(s::$julia_struct_name)::UInt32
                                        return UInt32((s._data[$(byte_pos + 1)] >> $bit_within_byte) & $(mask))
                                    end

                                    \"\"\"Set bitfield `$member_name` ($bit_size bits) in `$julia_struct_name`.\"\"\"
                                    function set_$(safe_member)!(s::$julia_struct_name, v::Integer)
                                        data = collect(s._data)
                                        cleared = data[$(byte_pos + 1)] & ~UInt8($(mask) << $bit_within_byte)
                                        data[$(byte_pos + 1)] = cleared | UInt8((UInt32(v) & $(mask)) << $bit_within_byte)
                                        s._data = NTuple{$byte_size, UInt8}(data)
                                    end

                                    """
                                else
                                    # Multi-byte bitfield — use unsafe_load for the containing integer
                                    struct_definitions *= """
                                    \"\"\"Get bitfield `$member_name` ($bit_size bits) from `$julia_struct_name`.\"\"\"
                                    function get_$(safe_member)(s::$julia_struct_name)::UInt32
                                        p = pointer(collect(s._data)) + $byte_pos
                                        raw = unsafe_load(Ptr{UInt32}(p))
                                        return (raw >> $bit_within_byte) & UInt32($(mask))
                                    end

                                    """
                                end
                                push!(exports, "get_$(safe_member)")
                                push!(exports, "set_$(safe_member)!")
                            else
                                # Non-bitfield member in a bitfield struct — byte-offset accessor
                                byte_off = parse(Int, get(member, "offset", "0x0"))
                                if julia_type != "Any" && !startswith(julia_type, "NTuple{")
                                    struct_definitions *= """
                                    \"\"\"Get non-bitfield member `$member_name` from `$julia_struct_name`.\"\"\"
                                    function get_$(safe_member)(s::$julia_struct_name)::$julia_type
                                        p = pointer(collect(s._data)) + $byte_off
                                        return unsafe_load(Ptr{$julia_type}(p))
                                    end

                                    """
                                    push!(exports, "get_$(safe_member)")
                                end
                            end
                        end

                        continue
                    end

                    # Check if any member has unresolvable size (template types with no DWARF size info).
                    # When DWARF byte_size is known but member sizes aren't, use a byte blob to ensure
                    # the Julia struct has the correct total size for ABI safety.
                    _known_c_primitives = Set([
                        "int", "unsigned int", "int32_t", "uint32_t",
                        "long", "unsigned long", "long long", "unsigned long long", "int64_t", "uint64_t",
                        "short", "unsigned short", "int16_t", "uint16_t",
                        "char", "unsigned char", "signed char", "int8_t", "uint8_t",
                        "float", "double", "long double", "bool", "_Bool",
                        "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
                        "wchar_t",
                    ])
                    has_unresolvable = false
                    for m in members
                        if get(m, "size", 0) == 0
                            c_type_raw = strip(replace(get(m, "c_type", ""), r"\bconst\b" => ""))
                            c_type_raw = strip(c_type_raw)
                            if endswith(c_type_raw, "*") || endswith(c_type_raw, "&")
                                continue
                            end
                            if c_type_raw in _known_c_primitives
                                continue
                            end
                            has_unresolvable = true
                            break
                        end
                    end

                    byte_size_str = get(struct_info, "byte_size", "0x0")
                    byte_size = isnothing(byte_size_str) ? 0 :
                        (startswith(string(byte_size_str), "0x") ?
                         parse(Int, string(byte_size_str)[3:end], base=16) :
                         parse(Int, string(byte_size_str)))

                    if has_unresolvable && byte_size > 0
                        member_count = length(members)
                        push!(blob_struct_names, julia_struct_name)
                        struct_definitions *= """
                        # C++ struct: $struct_name ($member_count members, byte blob for ABI safety)
                        struct $julia_struct_name
                            _data::NTuple{$(byte_size), UInt8}
                        end

                        # Zero-initializer for $julia_struct_name
                        function $julia_struct_name()
                            return $julia_struct_name(ntuple(i -> 0x00, $byte_size))
                        end

                        """

                        # Generate Base.getproperty accessor for named member access on byte-blob structs
                        _accessor_branches = String[]
                        _loadable_primitives = Dict(
                            "Cdouble" => ("Cdouble", 8), "Cfloat" => ("Cfloat", 4),
                            "Cint" => ("Cint", 4), "Cuint" => ("Cuint", 4),
                            "Clong" => ("Clong", 8), "Culong" => ("Culong", 8),
                            "Clonglong" => ("Clonglong", 8), "Culonglong" => ("Culonglong", 8),
                            "Cshort" => ("Cshort", 2), "Cushort" => ("Cushort", 2),
                            "Cchar" => ("Cchar", 1), "Cuchar" => ("Cuchar", 1),
                            "Csize_t" => ("Csize_t", 8), "Cptrdiff_t" => ("Cptrdiff_t", 8),
                            "Cssize_t" => ("Cssize_t", 8), "Bool" => ("Bool", 1),
                            "UInt8" => ("UInt8", 1), "Int8" => ("Int8", 1),
                            "UInt16" => ("UInt16", 2), "Int16" => ("Int16", 2),
                            "UInt32" => ("UInt32", 4), "Int32" => ("Int32", 4),
                            "UInt64" => ("UInt64", 8), "Int64" => ("Int64", 8),
                            "Float32" => ("Float32", 4), "Float64" => ("Float64", 8),
                        )
                        # Pre-compute in-context sizes for each member from offset gaps
                        _member_offsets = Int[]
                        for m in members
                            m_offset_str = get(m, "offset", nothing)
                            if isnothing(m_offset_str)
                                push!(_member_offsets, -1)
                            else
                                push!(_member_offsets, startswith(string(m_offset_str), "0x") ?
                                    parse(Int, string(m_offset_str)[3:end], base=16) :
                                    parse(Int, string(m_offset_str)))
                            end
                        end
                        for (mi, m) in enumerate(members)
                            m_name = get(m, "name", nothing)
                            isnothing(m_name) && continue
                            m_offset = _member_offsets[mi]
                            m_offset < 0 && continue
                            m_julia_type = get(m, "julia_type", "")
                            m_c_type = strip(get(m, "c_type", ""))

                            # Compute available space for this member from offset gaps
                            next_offset = byte_size
                            for j in (mi+1):length(_member_offsets)
                                if _member_offsets[j] >= 0
                                    next_offset = _member_offsets[j]
                                    break
                                end
                            end
                            available_size = next_offset - m_offset

                            # Pointer types
                            if endswith(m_c_type, "*")
                                push!(_accessor_branches, """
                                    if s === :$m_name
                                        return GC.@preserve x unsafe_load(Ptr{Ptr{Cvoid}}(pointer_from_objref(Ref(x._data)) + $m_offset))
                                    end""")
                            # Primitive types we can unsafe_load directly
                            elseif haskey(_loadable_primitives, m_julia_type)
                                jt, _ = _loadable_primitives[m_julia_type]
                                push!(_accessor_branches, """
                                    if s === :$m_name
                                        return GC.@preserve x unsafe_load(Ptr{$jt}(pointer_from_objref(Ref(x._data)) + $m_offset))
                                    end""")
                            # Nested struct types — extract sub-blob
                            else
                                m_sanitized = _sanitize_julia_type_name(m_c_type)
                                if !isempty(m_sanitized) && m_sanitized != m_c_type
                                    # Find the nested struct's byte_size using best_dwarf_key map
                                    nested_info = nothing
                                    best_key = get(best_dwarf_key, m_sanitized, nothing)
                                    if !isnothing(best_key) && haskey(dwarf_structs, best_key)
                                        nested_info = dwarf_structs[best_key]
                                    else
                                        # Fallback: search all structs for sanitized name match
                                        for (sk, sv) in dwarf_structs
                                            if _sanitize_julia_type_name(sk) == m_sanitized
                                                nested_info = sv
                                                break
                                            end
                                        end
                                    end
                                    if !isnothing(nested_info)
                                        nested_bs_str = get(nested_info, "byte_size", "0x0")
                                        nested_bs = isnothing(nested_bs_str) ? 0 :
                                            (startswith(string(nested_bs_str), "0x") ?
                                             parse(Int, string(nested_bs_str)[3:end], base=16) :
                                             parse(Int, string(nested_bs_str)))
                                        if nested_bs > 0
                                            if m_sanitized in blob_struct_names
                                                # Target is a byte-blob struct — extract bytes
                                                actual_size = min(nested_bs, available_size)
                                                if actual_size == nested_bs
                                                    push!(_accessor_branches, """
                                    if s === :$m_name
                                        bytes = GC.@preserve x ntuple(i -> unsafe_load(Ptr{UInt8}(pointer_from_objref(Ref(x._data)) + $m_offset + i - 1)), $nested_bs)
                                        return $(m_sanitized)(bytes)
                                    end""")
                                                else
                                                    push!(_accessor_branches, """
                                    if s === :$m_name
                                        raw = GC.@preserve x ntuple(i -> unsafe_load(Ptr{UInt8}(pointer_from_objref(Ref(x._data)) + $m_offset + i - 1)), $actual_size)
                                        padded = ntuple(i -> i <= $actual_size ? raw[i] : 0x00, $nested_bs)
                                        return $(m_sanitized)(padded)
                                    end""")
                                                end
                                            else
                                                # Target is a normal typed struct — unsafe_load directly
                                                push!(_accessor_branches, """
                                    if s === :$m_name
                                        return GC.@preserve x unsafe_load(Ptr{$m_sanitized}(pointer_from_objref(Ref(x._data)) + $m_offset))
                                    end""")
                                            end
                                        end
                                    end
                                end
                            end
                        end
                        if !isempty(_accessor_branches)
                            accessor_code = join(_accessor_branches, "\n")
                            struct_definitions *= """
                            function Base.getproperty(x::$julia_struct_name, s::Symbol)
                                s === :_data && return getfield(x, :_data)
                            $accessor_code
                                error("type $julia_struct_name has no field \$s")
                            end

                            """
                        end

                        continue
                    end

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

                        # Sanitize member name for Julia (replace invalid characters and keywords)
                        sanitized_name = replace(member_name, '$' => '_')
                        sanitized_name = make_julia_identifier(sanitized_name)

                        # Sanitize member types that reference other structs with template syntax
                        # Only sanitize custom struct names, not built-in Julia types like NTuple
                        sanitized_type = julia_type

                        # Don't sanitize built-in Julia types (NTuple, Ptr{Cint}, etc.)
                        builtin_types = ["NTuple", "Ptr", "Cint", "Cuint", "Cdouble", "Cfloat", "Clong", "Culong", "Cshort", "Cushort", "Cchar", "Cuchar", "Culonglong", "Clonglong", "Cvoid", "Csize_t", "Cptrdiff_t", "Cssize_t", "Cwchar_t", "Cstring", "Bool", "UInt8", "Int8", "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64", "Float32", "Float64"]
                        is_builtin = any(startswith(julia_type, bt) for bt in builtin_types)

                        if !is_builtin || occursin(r"[<>]", julia_type)
                            if occursin(r"Ptr\{[^}]+\}", julia_type)
                                # Extract type from Ptr{Type} for custom struct references
                                type_match = match(r"Ptr\{([^}]+)\}", julia_type)
                                if !isnothing(type_match)
                                    inner_type = String(type_match.captures[1])
                                    # Only sanitize if inner type is a custom struct (contains template chars)
                                    if occursin(r"[<>]", inner_type)
                                        if _is_stl_internal_type(inner_type)
                                            # STL-internal inner type: use Ptr{Cvoid} — never defined
                                            sanitized_type = "Ptr{Cvoid}"
                                        else
                                            sanitized_inner = _sanitize_julia_type_name(inner_type)
                                            sanitized_type = "Ptr{$sanitized_inner}"
                                        end
                                    end
                                end
                            elseif occursin(r"[<>]", julia_type)
                                # Direct custom struct reference with template syntax
                                if _is_stl_internal_type(julia_type)
                                    m_size = get(member, "size", 0)
                                    sanitized_type = m_size > 0 ? "NTuple{$m_size, UInt8}" : "Ptr{Cvoid}"
                                else
                                    sanitized_type = _sanitize_julia_type_name(julia_type)
                                end
                            end

                            # If the (now-sanitized) type refers to an STL-internal type that
                            # will be filtered out during struct generation, fall back to a byte
                            # buffer so the embedding struct remains valid Julia syntax.
                            if sanitized_type == julia_type && _is_stl_internal_type(julia_type)
                                m_size = get(member, "size", 0)
                                if m_size > 0
                                    sanitized_type = "NTuple{$m_size, UInt8}"
                                else
                                    # Size unknown; use Ptr{Cvoid} as a same-width placeholder
                                    sanitized_type = "Ptr{Cvoid}"
                                end
                            end
                        end

                        # If the field is Ptr{X} and X hasn't been defined yet,
                        # substitute Ptr{Cvoid} to avoid UndefVarError (same ABI size).
                        sanitized_type = _resolve_forward_ptr(sanitized_type, defined_struct_names)

                        struct_definitions *= "    $sanitized_name::$sanitized_type\n"
                        
                        # Update current offset
                        member_size = get(member, "size", 0)
                        # If size is 0 (e.g. unknown type), we can't reliably track offset
                        # But typically we rely on the next member's offset to insert padding
                        current_offset += member_size
                    end

                    struct_definitions *= """
                    end

                    # Zero-initializer for $julia_struct_name
                    function $julia_struct_name()
                        ref = Ref{$julia_struct_name}()
                        GC.@preserve ref begin
                            ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), Base.unsafe_convert(Ptr{Cvoid}, ref), 0, sizeof($julia_struct_name))
                        end
                        return ref[]
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

    # =============================================================================
    # GLOBAL VARIABLES GENERATION
    # =============================================================================
    
    global_vars = get(metadata, "globals", Dict())
    global_var_defs = ""
    
    if !isempty(global_vars)
        global_var_defs *= """
        # =============================================================================
        # Global Variables
        # =============================================================================

        """
        
        for (var_name, var_info) in global_vars
            julia_type = get(var_info, "julia_type", "Any")
            # Sanitize name
            safe_name = make_julia_identifier(var_name)
            
            # 1. Accessor for the value (read-only)
            # Only for simple types or pointers, struct values might be tricky
            # We use unsafe_load on the pointer
            
            global_var_defs *= """
            \"""
                $safe_name()

            Get value of global variable `$var_name`.
            \"""
            function $safe_name()::$julia_type
                ptr = cglobal((:$var_name, LIBRARY_PATH), $julia_type)
                return unsafe_load(ptr)
            end

            \"""
                $(safe_name)_ptr()

            Get pointer to global variable `$var_name`.
            \"""
            function $(safe_name)_ptr()::Ptr{$julia_type}
                return cglobal((:$var_name, LIBRARY_PATH), $julia_type)
            end

            """
            
            push!(exports, safe_name)
            push!(exports, "$(safe_name)_ptr")
        end
        
        global_var_defs *= "\n"
    end

    # Function wrappers
    function_wrappers = global_var_defs

    # 2. Generate Managed Types
    managed_types_def = ""
    if !isempty(deleters)
        managed_types_def *= """
        # =============================================================================
        # Managed Types (Auto-Finalizers)
        # =============================================================================

        """
        
        for (s_name, deleter) in deleters
            # Skip: a full idiomatic wrapper will be generated for this class
            haskey(idiomatic_classes, s_name) && continue
            # Sanitize names
            safe_s_name = _sanitize_julia_type_name(s_name)
            
            managed_name = "Managed$safe_s_name"
            
            # Use the mangled symbol for ccall (demangled names like ~Foo are invalid Julia symbols)
            deleter_sym = get(deleters_mangled, s_name, deleter)
            
            # Generate mutable struct with finalizer
            managed_types_def *= """
            mutable struct $managed_name
                handle::Ptr{$safe_s_name}
                
                function $managed_name(ptr::Ptr{$safe_s_name})
                    if ptr == C_NULL
                        error("Cannot wrap NULL pointer in $managed_name")
                    end
                    obj = new(ptr)
                    finalizer(obj) do x
                        # Call deleter: $deleter_sym(x.handle)
                        ccall((:$deleter_sym, LIBRARY_PATH), Cvoid, (Ptr{$safe_s_name},), x.handle)
                    end
                    return obj
                end
            end

            # Allow passing Managed object to ccall expecting Ptr
            unsafe_convert(::Type{Ptr{$safe_s_name}}, obj::$managed_name) = obj.handle
            
            export $managed_name

            """
        end
        
        struct_definitions *= managed_types_def
    end

    # =============================================================================
    # IDIOMATIC CLASS WRAPPERS  (High-level Julian API — sits on top of raw FFI)
    # =============================================================================
    # For every C++ class that has:
    #   - ≥1 factory function (pointer-returning) identified in the scan above
    #   - A resolvable GC-finalizer deleter (C delete_*/destroy_* or C++ destructor)
    # we emit:
    #   1. A `mutable struct ClassName { handle::Ptr{Cvoid} }` with inner constructors
    #      that call the raw factory function and attach a GC finalizer.
    #   2. An `unsafe_convert` for transparent use in raw FFI calls.
    #   3. Proxy methods `method(obj::ClassName, ...) = ClassName_method(obj.handle, ...)`.
    if !isempty(idiomatic_classes)
        struct_definitions *= """
        # =============================================================================
        # Idiomatic Class Wrappers
        # =============================================================================
        # High-level Julian structs wrapping C++ objects via opaque pointer handles.
        # Memory is automatically managed: Julia's GC calls the C++ destructor when
        # the last reference to the object is dropped.

        """

        for cls_nm in sort(collect(keys(idiomatic_classes)))
            cls_info   = idiomatic_classes[cls_nm]
            deleter_jl = cls_info["deleter"]
            factories  = cls_info["factories"]

            # Build one inner constructor block per factory function.
            ctor_blocks = String[]
            seen_ctor_sigs = Set{String}()

            for fac_func in factories
                fac_jl     = _sanitize_julia_fn(fac_func["name"])
                fac_params = get(fac_func, "parameters", Any[])

                sig_parts  = String[]
                call_args  = String[]
                used_names = Set{String}()

                for (i, p) in enumerate(fac_params)
                    p_nm = make_julia_identifier(get(p, "name", "arg$(i)"))
                    while p_nm in used_names; p_nm = "$(p_nm)_$(i)"; end
                    push!(used_names, p_nm)
                    p_jt = get(p, "julia_type", "Any")
                    push!(sig_parts, "$p_nm::$p_jt")
                    push!(call_args, p_nm)
                end

                param_sig = join(sig_parts, ", ")
                call_sig  = join(call_args, ", ")

                # Deduplicate identical constructor signatures (duplicate DWARF entries)
                param_sig in seen_ctor_sigs && continue
                push!(seen_ctor_sigs, param_sig)

                push!(ctor_blocks, """
                    function $cls_nm($param_sig)
                        handle = $fac_jl($call_sig)
                        Ptr{Cvoid}(handle) == C_NULL && error("$cls_nm: constructor returned NULL (allocation failed)")
                        obj = new(Ptr{Cvoid}(handle))
                        finalizer(obj) do o
                            $deleter_jl(o.handle)
                        end
                        return obj
                    end""")
            end

            isempty(ctor_blocks) && continue  # no usable constructors found

            ctors_str = join(ctor_blocks, "\n")

            struct_definitions *= """
            \"\"\"
                $cls_nm

            Idiomatic Julia wrapper for C++ class `$cls_nm`.
            Memory is automatically managed: Julia's GC calls the C++ destructor
            when this object is collected.
            \"\"\"
            mutable struct $cls_nm
                handle::Ptr{Cvoid}
            $ctors_str
            end

            # Allow passing a $cls_nm directly to raw FFI functions expecting a pointer.
            Base.unsafe_convert(::Type{Ptr{Cvoid}}, obj::$cls_nm) = obj.handle
            Base.unsafe_convert(::Type{Ptr{$cls_nm}}, obj::$cls_nm) = Ptr{$cls_nm}(obj.handle)

            """

            push!(exports, cls_nm)
        end
    end

    # =============================================================================
    # METHOD PROXIES  (Julian dispatch on idiomatic structs)
    # =============================================================================
    # For each instance method of an idiomatic class, emit a Julian wrapper:
    #   method_name(obj::ClassName, other_args...) = ClassName_method_name(obj.handle, ...)
    # This gives multi-dispatch semantics: the same method name works for multiple
    # classes (e.g. area(c::Circle) and area(r::Rectangle) dispatch independently).
    if !isempty(idiomatic_classes)
        for cls_nm in sort(collect(keys(idiomatic_classes)))
            methods = get(class_methods, cls_nm, Any[])
            isempty(methods) && continue

            struct_definitions *= """
            # Method proxies for $cls_nm
            """

            seen_proxy_sigs = Set{String}()

            for meth in methods
                meth_name = meth["name"]

                # The raw Julia wrapper for ClassName::method is named ClassName_method
                raw_jl = _sanitize_julia_fn("$(cls_nm)_$(meth_name)")

                # Proxy method name: just the method name part (multi-dispatch via Julia types)
                proxy_nm = _sanitize_julia_fn(meth_name)

                # Build parameter list, skipping the implicit 'this' pointer
                meth_params = get(meth, "parameters", Any[])
                sig_parts   = String[]
                call_args   = String["obj.handle"]  # pass the raw C pointer, not the Julia wrapper
                used_names  = Set{String}()

                for (i, p) in enumerate(meth_params)
                    p_nm_raw = get(p, "name", "arg$(i)")
                    lowercase(p_nm_raw) == "this" && continue
                    p_nm = make_julia_identifier(p_nm_raw)
                    while p_nm in used_names; p_nm = "$(p_nm)_$(i)"; end
                    push!(used_names, p_nm)
                    p_jt = get(p, "julia_type", "Any")
                    # Accept Any for pointer params so idiomatic wrappers pass transparently
                    dispatch_t = startswith(p_jt, "Ptr{") ? "Any" : p_jt
                    push!(sig_parts, "$p_nm::$dispatch_t")
                    push!(call_args, p_nm)
                end

                other_sig    = join(sig_parts, ", ")
                call_sig_str = join(call_args, ", ")

                # Deduplicate (same method may appear twice due to DWARF duplicate entries)
                proxy_key = "$(proxy_nm)($(cls_nm), $(other_sig))"
                proxy_key in seen_proxy_sigs && continue
                push!(seen_proxy_sigs, proxy_key)

                full_sig = isempty(other_sig) ? "obj::$cls_nm" : "obj::$cls_nm, $other_sig"

                struct_definitions *= """
                $proxy_nm($full_sig) = $raw_jl($call_sig_str)
                """

                push!(exports, proxy_nm)
            end

            struct_definitions *= "\n"
        end
    end



    # Track exported function names
    # exports already initialized at top


    for func in functions
        func_name = func["name"]
        mangled = func["mangled"]
        demangled = func["demangled"]

        # =========================================================
        # TIERED DISPATCH DECISION
        # =========================================================

        # Determine if we should use MLIR or ccall
        use_mlir_dispatch = !is_ccall_safe(func, dwarf_structs)
        if func_name == "pack_record"
            println("DEBUG: pack_record use_mlir_dispatch = $use_mlir_dispatch")
            println("DEBUG: is_ccall_safe returned ", is_ccall_safe(func, dwarf_structs))
            println("DEBUG: dwarf_structs keys = ", collect(keys(dwarf_structs)))
        end
        
        # BUG FIX: Make copies to allow modification (injecting 'this', refining types) without affecting metadata
        params = copy(func["parameters"])
        return_type = copy(func["return_type"])
        
        is_method = get(func, "is_method", false)
        is_vararg = get(func, "is_vararg", false)
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
                # Synthesize 'this' parameter.
                # The class_name may be namespace-qualified (e.g. "pugi::xml_document").
                # The Julia struct uses only the bare class name without namespace prefix,
                # so strip everything up to and including the last "::".
                # Also sanitize template brackets and other non-identifier characters.
                raw_class = class_name
                # Remove template parameters before splitting on ::
                # (so "std::vector<int>" → bare name is "vector" not "vector<int>")
                bare_class = let
                    angle_depth = 0
                    prefix_end = length(raw_class)
                    for (i, c) in enumerate(raw_class)
                        if c == '<'; angle_depth += 1
                        elseif c == '>'; angle_depth = max(0, angle_depth - 1)
                        end
                    end
                    # Find last "::" at angle-bracket depth 0
                    last_sep = 0
                    d = 0
                    i = 1
                    while i <= length(raw_class) - 1
                        c = raw_class[i]
                        if c == '<'; d += 1
                        elseif c == '>'; d = max(0, d - 1)
                        elseif c == ':' && raw_class[i+1] == ':' && d == 0
                            last_sep = i + 1
                            i += 1
                        end
                        i += 1
                    end
                    last_sep > 0 ? raw_class[last_sep+1:end] : raw_class
                end
                safe_class = _sanitize_julia_type_name(bare_class)
                safe_class = replace(safe_class, r"[^A-Za-z0-9_]" => "")
                safe_class = replace(safe_class, r"_+"            => "_")
                safe_class = String(strip(safe_class, '_'))
                # If garbled (empty or looks like an operator), fall back to Cvoid
                if isempty(safe_class) || startswith(safe_class, "operator")
                    safe_class = "Cvoid"
                end
                
                # Only synthesize 'this' when bare_class is a known struct type.
                # If it's a namespace name (e.g. "pugi") rather than a class, skip.
                if !isempty(safe_class) && safe_class != "Cvoid" && safe_class in struct_types
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
                 safe_base = _sanitize_julia_type_name(base_c)
                 return_type["julia_type"] = "Ptr{$safe_base}"
             end
        end

        # BUG FIX: Sanitize return types that are template instantiations (e.g. Box<double> -> Box_double)
        # This handles by-value returns of template types
        ret_type_str = get(return_type, "julia_type", "Cvoid")
        if occursin(r"[<>]", ret_type_str) && !startswith(ret_type_str, "Ptr{") && !startswith(ret_type_str, "Ref{")
            # If the raw return type is an STL-internal opaque type, use Ptr{Cvoid}
            if _is_stl_internal_type(ret_type_str)
                return_type["julia_type"] = "Ptr{Cvoid}"
            else
                safe_ret = _sanitize_julia_type_name(ret_type_str)
                return_type["julia_type"] = safe_ret
            end
        end

        # =========================================================
        # VARARGS INTERCEPTION
        # =========================================================
        if is_vararg
            # Sanitize function name for Julia (same logic as below)
            va_julia_name = func_name
            if is_method && !isempty(class_name)
                va_julia_name = "$(class_name)_$(func_name)"
            end
            va_julia_name = replace(va_julia_name, "~" => "destroy_")
            va_julia_name = replace(va_julia_name, "::" => "_")
            va_julia_name = replace(va_julia_name, "<" => "_")
            va_julia_name = replace(va_julia_name, ">" => "_")
            va_julia_name = replace(va_julia_name, "," => "_")
            va_julia_name = replace(va_julia_name, " " => "_")
            va_julia_name = replace(va_julia_name, "(" => "_")
            va_julia_name = replace(va_julia_name, ")" => "")
            va_julia_name = replace(va_julia_name, "&" => "ref")
            va_julia_name = replace(va_julia_name, "[" => "_")
            va_julia_name = replace(va_julia_name, "]" => "")
            va_julia_name = replace(va_julia_name, ":" => "_")
            va_julia_name = replace(va_julia_name, r"_+" => "_")
            va_julia_name = String(rstrip(va_julia_name, '_'))

            overloads = get(config.wrap.varargs_overloads, func_name, Vector{Vector{String}}())
            if isempty(overloads)
                @warn "Varargs function '$func_name' has no overloads configured in [wrap.varargs]. Generating base wrapper only."
            end

            va_code, va_exports = generate_vararg_wrappers(
                func_name, mangled, va_julia_name,
                params, return_type, overloads,
                generate_docs, demangled
            )
            function_wrappers *= va_code
            append!(exports, va_exports)
            continue  # Skip normal wrapper generation
        end

        # Build parameter list with ergonomic types
        param_names = String[]
        param_types = String[]  # C types for ccall
        julia_param_types = String[]  # Julia types for function signature (may differ)
        needs_conversion = Bool[]

        for (i, param) in enumerate(params)
            # Sanitize parameter name (e.g., avoid 'end', 'function' keywords)
            safe_name = make_julia_identifier(param["name"])
            # Ensure unique parameter names (duplicate names cause Julia syntax errors)
            if safe_name in param_names
                safe_name = "$(safe_name)_$(i)"
            end
            push!(param_names, safe_name)
            julia_type = param["julia_type"]

            # Sanitize C++ template types in parameter julia_type (e.g. "Ref{allocator<int> >}")
            if occursin(r"[<>]", julia_type)
                m = match(r"^(Ref|Ptr)\{(.*)\}$", julia_type)
                if m !== nothing
                    wrapper_kw, inner = m.captures
                    # If inner type is an STL-internal (allocator<>, char_traits<>, etc.),
                    # fall back to Ptr{Cvoid} — the type is opaque and not user-accessible.
                    if _is_stl_internal_type(String(inner))
                        julia_type = "Ptr{Cvoid}"
                    else
                        inner_safe = _sanitize_julia_type_name(inner)
                        julia_type = "$wrapper_kw{$inner_safe}"
                    end
                else
                    julia_type = _sanitize_julia_type_name(julia_type)
                end
            end
            # Also sanitize namespace separators (::) and other non-identifier chars
            # that can appear in Ptr{namespace::type} style references from DWARF
            if occursin(r"::|[^A-Za-z0-9_{},\[\]]", julia_type) && julia_type != "Any"
                m = match(r"^(Ref|Ptr)\{(.*)\}$", julia_type)
                if m !== nothing
                    wrapper_kw, inner = m.captures
                    inner_safe = _sanitize_julia_type_name(inner)
                    julia_type = isempty(inner_safe) ? "Ptr{Cvoid}" : "$wrapper_kw{$inner_safe}"
                else
                    julia_type = _sanitize_julia_type_name(julia_type)
                end
            end
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
            elseif startswith(actual_c_type, "Ptr{")
                # Relax pointer types to Any to allow Managed wrappers via Base.unsafe_convert
                push!(julia_param_types, "Any")
                push!(needs_conversion, false)
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
        julia_name = replace(julia_name, "(" => "_")
        julia_name = replace(julia_name, ")" => "")
        julia_name = replace(julia_name, "&" => "ref")
        julia_name = replace(julia_name, "[" => "_")
        julia_name = replace(julia_name, "]" => "")
        julia_name = replace(julia_name, ":" => "_")
        julia_name = replace(julia_name, r"_+" => "_")  # collapse consecutive underscores
        julia_name = String(rstrip(julia_name, '_'))

        # Build function signature using ergonomic Julia types
        param_sig_parts = String[]
        for (name, typ) in zip(param_names, julia_param_types)
            if name == "varargs..."
                push!(param_sig_parts, name)
            else
                push!(param_sig_parts, "$name::$typ")
            end
        end
        param_sig = join(param_sig_parts, ", ")

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
            \"\"\"
            """

            doc_comment = doc_parts
        end

        # =========================================================
        # BRANCH 1: MLIR DISPATCH (Robust Path)
        # =========================================================
        if use_mlir_dispatch
            requires_jit = true

            # Build argument list for invoke
            invoke_args = join(param_names, ", ")
            jit_ret_type = return_type["julia_type"]
            jit_c_ret = get(return_type, "c_type", "void")

            # Resolve "Any" return type to the Julia struct if we have a DWARF definition
            # This is critical: invoke(name, Any, ...) creates Ref{Any}() which corrupts
            # memory when the JIT writes raw struct bytes into it.
            if jit_ret_type == "Any" && jit_c_ret != "void"
                if haskey(dwarf_structs, jit_c_ret)
                    jit_ret_type = _sanitize_julia_type_name(jit_c_ret)
                else
                    # Fuzzy match: DWARF keys may have trailing " >" nesting artifacts
                    matched_key = _fuzzy_dwarf_lookup(jit_c_ret, dwarf_structs)
                    if matched_key !== nothing
                        jit_ret_type = _sanitize_julia_type_name(matched_key)
                    end
                end
            end

            # Determine if we need the return type overload of invoke
            # Void returns: invoke(name, args...)
            # Struct returns: invoke(name, RetType, args...)
            is_void_ret = jit_ret_type == "Cvoid" || jit_c_ret == "void"

            if config.compile.aot_thunks
                # The AOT compiled thunks expect a single argument: void** args
                thunk_sym = ":_mlir_ciface_$(mangled)_thunk"
                
                ptr_setup = ""
                if !isempty(invoke_args)
                    ptr_setup *= "    refs = ($(join(["Ref($a)" for a in param_names], ", ")),)\n"
                    ptr_setup *= "    inner_ptrs = Ptr{Cvoid}[$(join(["Base.unsafe_convert(Ptr{Cvoid}, r)" for r in ["refs[$i]" for i in 1:length(param_names)]], ", "))]\n"
                    ptr_setup *= "    GC.@preserve refs inner_ptrs begin\n"
                else
                    ptr_setup *= "    inner_ptrs = Ptr{Cvoid}[]\n"
                    ptr_setup *= "    GC.@preserve inner_ptrs begin\n"
                end
                
                if is_void_ret
                    invoke_call = "        if !isempty(THUNKS_LTO_IR)\n"
                    invoke_call *= "            Base.llvmcall((THUNKS_LTO_IR, \"_mlir_ciface_$(mangled)_thunk\"), Cvoid, Tuple{Ptr{Ptr{Cvoid}}}, inner_ptrs)\n"
                    invoke_call *= "        else\n"
                    invoke_call *= "            ccall(($thunk_sym, THUNKS_LIBRARY_PATH), Cvoid, (Ptr{Ptr{Cvoid}},), inner_ptrs)\n"
                    invoke_call *= "        end\n"
                    invoke_call *= "        return nothing\n"
                    invoke_call *= "    end"
                elseif jit_ret_type in ("Int8","UInt8","Int16","UInt16","Int32","UInt32","Int64","UInt64",
                                        "Float32","Float64","Cint","Clong","Culong","Csize_t","Cchar",
                                        "Cdouble","Cfloat","Bool","Ptr{Cvoid}","Any","Cstring")
                    # Scalar return directly
                    invoke_call = "        if !isempty(THUNKS_LTO_IR)\n"
                    invoke_call *= "            ret = Base.llvmcall((THUNKS_LTO_IR, \"_mlir_ciface_$(mangled)_thunk\"), $jit_ret_type, Tuple{Ptr{Ptr{Cvoid}}}, inner_ptrs)\n"
                    invoke_call *= "        else\n"
                    invoke_call *= "            ret = ccall(($thunk_sym, THUNKS_LIBRARY_PATH), $jit_ret_type, (Ptr{Ptr{Cvoid}},), inner_ptrs)\n"
                    invoke_call *= "        end\n"
                    invoke_call *= "    end\n"
                    invoke_call *= "    return ret"
                else
                    # Struct return via sret pointer
                    invoke_call = "        ret_buf = Ref{$jit_ret_type}()\n"
                    invoke_call *= "        GC.@preserve ret_buf begin\n"
                    invoke_call *= "            if !isempty(THUNKS_LTO_IR)\n"
                    invoke_call *= "                Base.llvmcall((THUNKS_LTO_IR, \"_mlir_ciface_$(mangled)_thunk\"), Cvoid, Tuple{Ptr{$jit_ret_type}, Ptr{Ptr{Cvoid}}}, ret_buf, inner_ptrs)\n"
                    invoke_call *= "            else\n"
                    invoke_call *= "                ccall(($thunk_sym, THUNKS_LIBRARY_PATH), Cvoid, (Ptr{$jit_ret_type}, Ptr{Ptr{Cvoid}}), ret_buf, inner_ptrs)\n"
                    invoke_call *= "            end\n"
                    invoke_call *= "        end\n"
                    invoke_call *= "    end\n"
                    invoke_call *= "    return ret_buf[]"
                end

                func_def = """
                $doc_comment
                function $julia_name($param_sig)
                    # [Tier 2] Dispatch to MLIR AOT Thunk (Complex ABI / Packed / Union)
                $ptr_setup
                $invoke_call
                end
                """
            else
                if is_void_ret
                    invoke_call = if isempty(invoke_args)
                        "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(mangled)_thunk\")"
                    else
                        "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(mangled)_thunk\", $invoke_args)"
                    end
                else
                    invoke_call = if isempty(invoke_args)
                        "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(mangled)_thunk\", $jit_ret_type)"
                    else
                        "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(mangled)_thunk\", $jit_ret_type, $invoke_args)"
                    end
                end

                func_def = """
                $doc_comment
                function $julia_name($param_sig)
                    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
                    return $invoke_call
                end
                """
            end

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

        # llvmcall needs Tuple{Type1, Type2} — built from raw param_types (no trailing comma)
        # CRITICAL: Ref{T} in Julia maps to ptr addrspace(10) in LLVM, but C++ IR uses
        # plain ptr (addrspace 0). For llvmcall we must use Ptr{T} and convert explicitly.
        llvmcall_param_types = String[]
        llvmcall_arg_names = String[]
        llvmcall_ref_args = String[]  # args that need GC.@preserve
        llvmcall_conversion_lines = String[]
        for (i, pt) in enumerate(param_types)
            arg_name = ccall_param_names[i]
            m = match(r"^Ref\{(.+)\}$", pt)
            if m !== nothing
                inner_type = m.captures[1]
                push!(llvmcall_param_types, "Ptr{$inner_type}")
                ptr_name = "__ptr_$(arg_name)"
                push!(llvmcall_arg_names, ptr_name)
                push!(llvmcall_ref_args, arg_name)
                push!(llvmcall_conversion_lines, "        $ptr_name = Base.unsafe_convert(Ptr{$inner_type}, $arg_name)")
            else
                push!(llvmcall_param_types, pt)
                push!(llvmcall_arg_names, arg_name)
            end
        end
        llvmcall_types = isempty(llvmcall_param_types) ? "" : join(llvmcall_param_types, ", ")
        llvmcall_args = join(llvmcall_arg_names, ", ")
        has_ref_params = !isempty(llvmcall_ref_args)

        # Generate function body based on return type and conversions
        julia_return_type = return_type["julia_type"]
        c_return_type = return_type["c_type"]

        # Check if return type is a struct (not primitive, not pointer)
        is_struct_return = julia_return_type == "Any" && !contains(c_return_type, "*") && !contains(c_return_type, "void") && c_return_type != "unknown"
        
        # Also detect resolved struct returns: julia_return_type was mapped to a
        # known struct name (e.g. Matrix_double_minus_1...) but is_struct_return
        # missed it because julia_return_type != "Any".
        returns_known_struct = is_struct_return ||
            haskey(julia_to_cpp_struct, julia_return_type) ||
            c_return_type in struct_types

        # Check if this is a virtual method that requires dynamic dispatch
        is_virtual = get(func, "is_virtual", false)

        # Build the llvmcall expression with proper Ref{T} → Ptr{T} handling.
        # ccall handles Ref{T} → C pointer automatically, but llvmcall sees Ref{T}
        # as ptr addrspace(10) while C++ IR uses plain ptr (addrspace 0).
        function _build_llvmcall_expr(ret_type_str, indent="        ")
            call_expr = "Base.llvmcall((LTO_IR, \"$mangled\"), $ret_type_str, Tuple{$llvmcall_types}, $llvmcall_args)"
            if !has_ref_params
                return "$(indent)return $call_expr"
            end
            lines = String[]
            for l in llvmcall_conversion_lines
                push!(lines, "$indent$l")
            end
            preserve_list = join(llvmcall_ref_args, " ")
            push!(lines, "$(indent)GC.@preserve $preserve_list begin")
            push!(lines, "$(indent)    return $call_expr")
            push!(lines, "$(indent)end")
            return join(lines, "\n")
        end

        # LTO eligibility: safe for primitive/pointer/small-POD structs (filtered earlier by is_ccall_safe),
        # non-virtual, no Cstring anywhere (llvmcall won't auto-convert String→Cstring like ccall does),
        # and NOT returning a struct by value (Base.llvmcall doesn't handle the sret ABI).
        lto_eligible = config.link.enable_lto &&
            !is_virtual &&
            !returns_known_struct &&
            julia_return_type != "Cstring" &&
            !any(t -> t == "Cstring", param_types)

        # Check for _UnsafeUnknown trap to prevent segfaults
        has_unknown_param = any(t -> t == "_UnsafeUnknown", param_types)
        is_unknown_return = julia_return_type == "_UnsafeUnknown"

        if has_unknown_param || is_unknown_return
            func_def = """
            $doc_comment
            function $julia_name($param_sig)
                error(\"\"\"
                FFI Safety Trap: Cannot call function '$julia_name'.
                One or more types could not be mapped to Julia safely:
                Return type: $julia_return_type (C++: $c_return_type)
                Parameter types: $(join(param_types, ", "))
                
                To fix this, add the missing type mapping to your replibuild.toml.
                Calling this function would have caused a segmentation fault.
                \"\"\")
            end

            """
        elseif is_virtual
            requires_jit = true
            
            # Sanitize return type if needed
            safe_c_ret = c_return_type
            if occursin(r"[<>]", safe_c_ret)
                 safe_c_ret = _sanitize_julia_type_name(safe_c_ret)
            end
            ret_type_sig = (julia_return_type == "Any" || julia_return_type == "Cstring") ? safe_c_ret : julia_return_type
            
            if config.compile.aot_thunks
                # AOT-based virtual dispatch wrapper
                # We call the statically generated MLIR thunk which natively handles the vtable math
                thunk_sym = ":thunk_$(mangled)"
                
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$ret_type_sig
                $conversion_code    return ccall(($thunk_sym, THUNKS_LIBRARY_PATH), $ret_type_sig, $ccall_types, $ccall_args)
                end

                """
            else
                # JIT-based virtual dispatch wrapper
                cls_name = get(func, "class", "")
                
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$ret_type_sig
                    # Get thunk for virtual dispatch (JIT compiled on-demand)
                    thunk_ptr = RepliBuild.JITManager.get_jit_thunk("$cls_name", "$func_name")
                    
                    # Call thunk
                $conversion_code    return ccall(thunk_ptr, $ret_type_sig, $ccall_types, $ccall_args)
                end

                """
            end
        elseif is_struct_return
            # Struct-valued return - Julia uses the struct type directly
            # ccall will handle struct returns automatically if the Julia type matches
            
            # Sanitize C return type if it contains template chars (Box<int> -> Box_int)
            safe_c_ret = c_return_type
            if occursin(r"[<>]", safe_c_ret)
                 safe_c_ret = _sanitize_julia_type_name(safe_c_ret)
            end

            if lto_eligible
                llvmcall_body = _build_llvmcall_expr(safe_c_ret)
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$safe_c_ret
                $conversion_code    if !isempty(LTO_IR)
                $llvmcall_body
                    else
                        return ccall((:$mangled, LIBRARY_PATH), $safe_c_ret, $ccall_types, $ccall_args)
                    end
                end

                """
            else
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$safe_c_ret
                $conversion_code    return ccall((:$mangled, LIBRARY_PATH), $safe_c_ret, $ccall_types, $ccall_args)
                end

                """
            end
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
            if lto_eligible
                llvmcall_body = _build_llvmcall_expr(julia_return_type)
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                $conversion_code    if !isempty(LTO_IR)
                $llvmcall_body
                    else
                        return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
                    end
                end

                """
            else
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                $conversion_code    return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
                end

                """
            end
        else
            # Standard wrapper - no conversions needed
            if lto_eligible
                llvmcall_body = _build_llvmcall_expr(julia_return_type)
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                    if !isempty(LTO_IR)
                $llvmcall_body
                    else
                        return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
                    end
                end

                """
            else
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                    ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
                end

                """
            end
        end

        function_wrappers *= func_def
        push!(exports, julia_name)

        # Generate safe wrapper if return type is a managed struct
        if startswith(julia_return_type, "Ptr{") && endswith(julia_return_type, "}")
            target_struct = julia_return_type[5:end-1]
            # Sanitize target struct name to match ManagedX keys
            safe_target = _sanitize_julia_type_name(target_struct)
            
            if (haskey(deleters, safe_target) || haskey(deleters, target_struct)) && !haskey(idiomatic_classes, safe_target) && !haskey(idiomatic_classes, target_struct)
                managed_name = "Managed$safe_target"
                safe_func_name = "$(julia_name)_safe"
                
                safe_wrapper = """
                \"""
                    $safe_func_name($param_sig) -> $managed_name

                Safe wrapper for `$julia_name` that returns a managed object with automatic finalization.
                \"""
                function $safe_func_name($param_sig)::$managed_name
                    ptr = $julia_name($ccall_args)
                    return $managed_name(ptr)
                end
                """
                
                function_wrappers *= safe_wrapper
                push!(exports, safe_func_name)
            end
        end

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

    # =================================================================
    # STL CONTAINER FACTORY FUNCTIONS
    # =================================================================
    stl_methods = get(metadata, "stl_methods", Dict())
    if !isempty(stl_methods)
        requires_jit = true
        function_wrappers *= """

        # =============================================================================
        # STL Container Factories
        # =============================================================================

        """

        for (container_type, methods) in stl_methods
            # Build thunks dict for this container
            thunks_entries = String[]
            for m in methods
                method_name = get(m, "method", "")
                mangled_name = get(m, "mangled", "")
                if !isempty(method_name) && !isempty(mangled_name)
                    push!(thunks_entries, "\"$method_name\" => \"$mangled_name\"")
                end
            end
            thunks_dict_str = "Dict{String,String}(" * join(thunks_entries, ", ") * ")"

            # Get container byte_size from DWARF struct_definitions
            byte_size = 24  # default for vector on 64-bit
            for (sname, sinfo) in dwarf_structs
                if contains(sname, container_type) || _normalize_stl_for_dwarf(sname) == container_type
                    bs_str = get(sinfo, "byte_size", "24")
                    byte_size = bs_str isa Integer ? bs_str : (startswith(bs_str, "0x") ? parse(Int, bs_str[3:end], base=16) : parse(Int, bs_str))
                    break
                end
            end

            # Generate sanitized factory name
            safe_name = replace(replace(replace(container_type, "::" => "_"), "<" => "_"), ">" => "")
            safe_name = replace(replace(safe_name, " " => ""), "," => "_")

            # Determine element type for CppVector
            if startswith(container_type, "std::vector<")
                elem_match = match(r"std::vector<(.+)>$", container_type)
                if !isnothing(elem_match)
                    elem_cpp = String(strip(elem_match.captures[1]))
                    elem_julia = infer_julia_type(registry, elem_cpp; context="STL factory element type")
                    # Map to ccall-compatible type
                    if elem_julia in ("Cint", "Int32")
                        elem_julia = "Cint"
                    elseif elem_julia in ("Cdouble", "Float64")
                        elem_julia = "Cdouble"
                    elseif elem_julia in ("Cfloat", "Float32")
                        elem_julia = "Cfloat"
                    elseif elem_julia in ("Int64",)
                        elem_julia = "Int64"
                    elseif elem_julia in ("UInt64",)
                        elem_julia = "UInt64"
                    end

                    factory_name = "create_$(safe_name)"
                    push!(exports, factory_name)

                    function_wrappers *= """
                    const _THUNKS_$(uppercase(safe_name)) = $thunks_dict_str

                    \"\"\"
                        $factory_name() -> CppVector{$elem_julia}

                    Create a new `$container_type` and return a Julia `CppVector{$elem_julia}` wrapper.
                    \"\"\"
                    function $factory_name()
                        buf = Libc.malloc($byte_size)
                        # Zero-initialize
                        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), buf, 0, $byte_size)
                        # Call C++ constructor
                        ctor = get(_THUNKS_$(uppercase(safe_name)), "constructor", "")
                        if !isempty(ctor)
                            RepliBuild.JITManager.invoke("_mlir_ciface_\$(ctor)_thunk", buf)
                        end
                        return RepliBuild.STLWrappers.CppVector{$elem_julia}(buf, true, _THUNKS_$(uppercase(safe_name)); byte_size=$byte_size)
                    end

                    """
                end
            elseif startswith(container_type, "std::basic_string") || container_type == "std::string"
                factory_name = "create_$(safe_name)"
                push!(exports, factory_name)

                function_wrappers *= """
                    const _THUNKS_$(uppercase(safe_name)) = $thunks_dict_str

                    \"\"\"
                        $factory_name() -> CppString

                    Create a new `$container_type` and return a Julia `CppString` wrapper.
                    \"\"\"
                    function $factory_name()
                        buf = Libc.malloc($byte_size)
                        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), buf, 0, $byte_size)
                        ctor = get(_THUNKS_$(uppercase(safe_name)), "constructor", "")
                        if !isempty(ctor)
                            RepliBuild.JITManager.invoke("_mlir_ciface_\$(ctor)_thunk", buf)
                        end
                        return RepliBuild.STLWrappers.CppString(buf, true, _THUNKS_$(uppercase(safe_name)); byte_size=$byte_size)
                    end

                """
            end
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

    # Add struct types (filter internal/compiler types)
    for struct_name in struct_types
        if !(struct_name in enum_names)
            # Sanitize struct name for export
            julia_struct_name = _sanitize_julia_type_name(struct_name)
            julia_struct_name = replace(julia_struct_name, "*" => "Ptr")
            julia_struct_name = replace(julia_struct_name, "&" => "Ref")
            julia_struct_name = replace(julia_struct_name, r"[^a-zA-Z0-9_]" => "_")
            # Skip internal/compiler types
            if julia_struct_name in _INTERNAL_TYPE_BLOCKLIST || struct_name in _INTERNAL_TYPE_BLOCKLIST
                continue
            end
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

    # Generate initialization block
    init_block = if config.compile.aot_thunks
        """
        # Library handles for manual management if needed
        const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
        const THUNKS_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

        function __init__()
            # Load main library explicitly to ensure symbols are available
            LIB_HANDLE[] = Libdl.dlopen(LIBRARY_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
            
            # Load AOT thunks library if it was successfully generated
            if !isempty(THUNKS_LIBRARY_PATH) && isfile(THUNKS_LIBRARY_PATH)
                THUNKS_HANDLE[] = Libdl.dlopen(THUNKS_LIBRARY_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
            elseif $requires_jit
                @warn "AOT Thunks library not found, but advanced FFI features are required. These features will fail at runtime."
            end
        end
        """
    else
        if requires_jit
            """
            function __init__()
                # Initialize the global JIT context with this library's vtables
                RepliBuild.JITManager.initialize_global_jit(LIBRARY_PATH)
            end
            """
        else
            """
            # Library handle for manual management if needed
            const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

            function __init__()
                # Load library explicitly to ensure symbols are available
                LIB_HANDLE[] = Libdl.dlopen(LIBRARY_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
            end
            """
        end
    end

    return header * init_block * metadata_section * enum_definitions * struct_definitions * export_statement * function_wrappers * footer
end

end # module Wrapper
