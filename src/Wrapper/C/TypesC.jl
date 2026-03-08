# =============================================================================
# TYPE SYSTEM - Comprehensive C/C++ to Julia Type Mapping
# =============================================================================

# =============================================================================
# TYPE HEURISTICS - Smart detection of type categories
# =============================================================================

"""
    is_c_struct_like(c_type::String)::Bool

Check if a C++ type looks like a struct/class based on naming conventions.
Structs typically: start with uppercase, contain only alphanumeric + underscore.
"""
function is_c_struct_like(c_type::String)::Bool
    # Remove pointers, references, and whitespace
    base = strip(replace(replace(c_type, "*" => ""), "&" => ""))

    # Empty or primitive types are not structs
    if isempty(base) || base == "void"
        return false
    end

    # Struct names typically match: UpperCamelCase or snake_case with uppercase first letter
    # Examples: Matrix3x3, Grid, ComplexType, My_Struct
    return occursin(r"^[A-Z][A-Za-z0-9_]*$", base)
end

"""
    is_c_enum_like(c_type::String)::Bool

Check if a C++ type looks like an enum (similar heuristics to struct for now).
Future: enhance with DWARF metadata lookup.
"""
function is_c_enum_like(c_type::String)::Bool
    # For now, use same heuristics as struct
    # In future: check against extracted enum names from DWARF
    return is_c_struct_like(c_type)
end

"""
    is_c_function_pointer_like(c_type::String)::Bool

Check if a C++ type contains function pointer syntax.
Looks for patterns: (*name), (*)(args), (^name) (blocks)
"""
function is_c_function_pointer_like(c_type::String)::Bool
    # Function pointer patterns:
    # - int (*callback)(double, double)
    # - void (*cleanup)()
    # - typedef int (*IntCallback)(double, double)
    return occursin(r"\(\s*\*", c_type) || occursin(r"\(\s*\^", c_type)
end





"""
    handle_unknown_c_type(registry::TypeRegistry, c_type::String, context::String)::String

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
function handle_unknown_c_type(registry::TypeRegistry, c_type::String, context::String)::String
    # Check if it looks like a struct
    if is_c_struct_like(c_type)
        if registry.allow_unknown_structs
            if registry.strictness == WARN
                @warn "Treating unknown type '$c_type' as opaque struct in $context"
            end
            # Remove pointer/reference markers for struct name
            base_type = strip(replace(replace(c_type, "*" => ""), "&" => ""))
            
            # Sanitize for Julia (e.g. Box<int> -> Box_int)
            sanitized = _sanitize_c_type_name(base_type)
            return sanitized
        end
    end

    # Check if it looks like an enum
    if is_c_enum_like(c_type)
        if registry.allow_unknown_enums
            if registry.strictness == WARN
                @warn "Treating enum '$c_type' as Cint in $context"
            end
            return "Cint"
        end
    end

    # Check if it's a function pointer
    if is_c_function_pointer_like(c_type)
        if registry.allow_function_pointers
            if registry.strictness == WARN
                @warn "Treating function pointer '$c_type' as Ptr{Cvoid} in $context"
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

        Unknown C/C++ type: '$c_type'
        Context: $context

        This type could not be mapped to a Julia type.

        ───────────────────────────────────────────────────────────────
        Suggestions:
        ───────────────────────────────────────────────────────────────

        1. Add custom type mapping in your code:
           registry = create_c_type_registry(config,
               custom_types=Dict("$c_type" => "YourJuliaType"))

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
           - Example: "std::vector<$c_type>" => "Vector{JuliaType}"

        ───────────────────────────────────────────────────────────────
        Temporary Workaround:
        ───────────────────────────────────────────────────────────────

        To allow unmapped types temporarily (not recommended):

           registry = create_c_type_registry(config, strictness=WARN)

        Or for complete permissiveness (legacy mode):

           registry = create_c_type_registry(config, strictness=PERMISSIVE)

        ═══════════════════════════════════════════════════════════════
        """)
    elseif registry.strictness == WARN
        @warn """Unknown C/C++ type '$c_type' in $context, falling back to safe placeholder.
        This function will throw an error if called to prevent GC memory corruption.
        Consider adding a custom type mapping."""
        return "_UnsafeUnknown"
    else  # PERMISSIVE
        return "Any"
    end
end

"""
    infer_c_type(registry::TypeRegistry, c_type::String; context::String="")::String

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
9. Smart fallback via handle_unknown_c_type() based on strictness

# Arguments
- `registry`: TypeRegistry with type mappings and validation settings
- `c_type`: C/C++ type string to map
- `context`: Optional context string for error messages (e.g., "parameter 1 of function foo")

# Examples
```julia
infer_c_type(reg, "int") # => "Cint"
infer_c_type(reg, "const char*") # => "Cstring"
infer_c_type(reg, "double*") # => "Ptr{Cdouble}"
infer_c_type(reg, "std::string") # => "String"
infer_c_type(reg, "std::vector<int>") # => "Vector{Cint}"
infer_c_type(reg, "Matrix3x3", context="parameter 1") # => "Matrix3x3" or error in STRICT mode
```
"""
function infer_c_type(registry::TypeRegistry, c_type::String; context::String="")::String
    if isempty(c_type)
        return handle_unknown_c_type(registry, "", context == "" ? "empty type string" : context)
    end

    # Clean up the type string
    clean_type = strip(c_type)

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
        julia_base = infer_c_type(registry, String(base_type); context=ptr_context)
        return "Ptr{$julia_base}"
    end

    # Parse reference types: T& or T &
    if endswith(clean_type, "&")
        base_type = strip(replace(clean_type, r"&$" => ""))
        ref_context = context == "" ? "reference base type" : "$context (reference to $base_type)"
        julia_base = infer_c_type(registry, String(base_type); context=ref_context)
        return "Ref{$julia_base}"
    end

    # Parse array types: T[N]
    array_match = match(r"^(.+)\[(\d+)\]$", clean_type)
    if !isnothing(array_match)
        elem_type = strip(array_match.captures[1])
        size = parse(Int, array_match.captures[2])
        arr_context = context == "" ? "array element type" : "$context (array of $elem_type)"
        julia_elem = infer_c_type(registry, String(elem_type); context=arr_context)
        return "NTuple{$size,$julia_elem}"
    end


    # Unknown type - no matching rules
    ctx = context == "" ? "type inference" : context
    return handle_unknown_c_type(registry, String(clean_type), ctx)
end

