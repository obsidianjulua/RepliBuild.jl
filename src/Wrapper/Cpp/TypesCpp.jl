# =============================================================================
# TYPE SYSTEM - Comprehensive C/C++ to Julia Type Mapping
# =============================================================================

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

Check if a C++ type looks like an enum based on naming conventions.
Returns false — enum detection is unreliable by name alone (uppercase names
are more likely structs/classes). Real enum identification uses DWARF metadata
via `_is_enum_type()` in DispatchLogic.jl which checks `__enum__` prefixed keys.
"""
function is_enum_like(cpp_type::String)::Bool
    # Enum detection requires DWARF metadata (__enum__ prefix).
    # Name-based heuristics can't distinguish enums from structs,
    # and treating an unknown struct as Cint would corrupt memory.
    return false
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
    _stl_name_match(clean, prefix) -> Bool

Match an STL type name with word-boundary awareness.
Returns true if `clean` equals `prefix` exactly, or starts with `prefix`
followed by `<` or ` ` (template args). Prevents false positives like
`std::string_view` matching `std::string`.
"""
function _stl_name_match(clean::AbstractString, prefix::AbstractString)::Bool
    startswith(clean, prefix) || return false
    ncodeunits(clean) == ncodeunits(prefix) && return true
    c = clean[ncodeunits(prefix) + 1]
    return c == '<' || c == ' '
end

"""
    is_stl_container_type(c_type::String)::Bool

Check if a C++ type is an STL container (non-POD, requires opaque handle).
"""
function is_stl_container_type(c_type::String)::Bool
    clean = strip(replace(c_type, r"\bconst\b" => ""))
    clean = strip(replace(clean, r"[*&]+$" => ""))
    clean = strip(clean)
    # Exact name or name followed by '<' (template args)
    stl_names = (
        "std::vector", "std::basic_string", "std::string",
        "std::map", "std::unordered_map", "std::set", "std::unordered_set",
        "std::deque", "std::list", "std::forward_list",
        "std::multimap", "std::multiset",
    )
    any(p -> _stl_name_match(clean, p), stl_names) && return true
    # DWARF often strips the std:: namespace prefix — these already require '<'
    return any(p -> startswith(clean, p),
        ("vector<", "basic_string<",
         "map<", "unordered_map<", "set<", "unordered_set<",
         "deque<", "list<", "forward_list<",
         "multimap<", "multiset<"))
end

"""
    get_stl_container_size(c_type::String) -> Int

Returns the exact ABI byte size of common STL containers on x86_64 SysV.
Returns 0 if unknown.
"""
function get_stl_container_size(c_type::String)::Int
    clean = strip(replace(c_type, r"^(const|struct|class|union)\b" => ""))
    clean = strip(replace(clean, r"[*&]+$" => ""))
    
    # Check for both "std::..." and raw prefixes (since DWARF sometimes strips std::)
    # Use _stl_name_match for std:: prefixes to avoid false positives
    # (e.g. "std::string_view" matching "std::string")
    if _stl_name_match(clean, "std::vector") || startswith(clean, "vector<")
        return 24
    elseif _stl_name_match(clean, "std::basic_string") || _stl_name_match(clean, "std::string") || startswith(clean, "basic_string<")
        return 32 # libstdc++ SSO string size is 32 bytes on 64-bit
    elseif _stl_name_match(clean, "std::shared_ptr") || startswith(clean, "shared_ptr<")
        return 16
    elseif _stl_name_match(clean, "std::unique_ptr") || startswith(clean, "unique_ptr<")
        return 8
    elseif _stl_name_match(clean, "std::unordered_map") || startswith(clean, "unordered_map<") || _stl_name_match(clean, "std::unordered_set") || startswith(clean, "unordered_set<")
        return 56 # Typical hashtable size — check before std::map/std::set to avoid prefix collision
    elseif _stl_name_match(clean, "std::map") || startswith(clean, "map<") || _stl_name_match(clean, "std::set") || startswith(clean, "set<")
        return 48 # Typical rb_tree size
    elseif _stl_name_match(clean, "std::list") || startswith(clean, "list<")
        return 24
    elseif _stl_name_match(clean, "std::deque") || startswith(clean, "deque<")
        return 80
    end
    return 0
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
        # libstdc++ / libc++ internal types
        "_Alloc_hider", "_Alloc_node", "_Auto_node",
        "_Guard", "_Guard_alloc", "_Storage",
        "_Temporary_value", "_UninitDestroyGuard", "_Vector_impl",
        "_Vector_base", "_Bvector", "_Deque_impl", "_List_impl",
        "_Rb_tree", "_Hashtable", "_Node_base", "_Node_alloc",
        "_Node_handle", "_Node_insert_return",
        "_Map_base", "_Insert", "_Rehash",
        "_Reuse_or_alloc_node", "_Head_base",
        "__gnu_cxx::", "std::_", "std::__", "__cxx",
        "__aligned_membuf", "__sv_wrapper", "__uses_alloc",
        "allocator<", "char_traits<", "less<", "hash<", "equal_to<",
        "iterator<", "reverse_iterator<", "__normal_iterator<",
        "__wrap_iter<", "initializer_list<",
        "pair<", "move_iterator<", "basic_string_view<",
        "value_compare", "Select1st<",
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
    handle_unknown_cpp_type(registry::TypeRegistry, cpp_type::String, context::String)::String

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
function handle_unknown_cpp_type(registry::TypeRegistry, cpp_type::String, context::String)::String
    # Check if it looks like a struct
    if is_struct_like(cpp_type)
        if registry.allow_unknown_structs
            if registry.strictness == WARN
                @warn "Treating unknown type '$cpp_type' as opaque struct in $context"
            end
            # Remove pointer/reference markers for struct name
            base_type = strip(replace(replace(cpp_type, "*" => ""), "&" => ""))
            
            # Sanitize for Julia (e.g. Box<int> -> Box_int)
            sanitized = _sanitize_cpp_type_name(base_type)
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
           registry = create_cpp_type_registry(config,
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

           registry = create_cpp_type_registry(config, strictness=WARN)

        Or for complete permissiveness (legacy mode):

           registry = create_cpp_type_registry(config, strictness=PERMISSIVE)

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
    infer_cpp_type(registry::TypeRegistry, cpp_type::String; context::String="")::String

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
9. Smart fallback via handle_unknown_cpp_type() based on strictness

# Arguments
- `registry`: TypeRegistry with type mappings and validation settings
- `cpp_type`: C/C++ type string to map
- `context`: Optional context string for error messages (e.g., "parameter 1 of function foo")

# Examples
```julia
infer_cpp_type(reg, "int") # => "Cint"
infer_cpp_type(reg, "const char*") # => "Cstring"
infer_cpp_type(reg, "double*") # => "Ptr{Cdouble}"
infer_cpp_type(reg, "std::string") # => "String"
infer_cpp_type(reg, "std::vector<int>") # => "Vector{Cint}"
infer_cpp_type(reg, "Matrix3x3", context="parameter 1") # => "Matrix3x3" or error in STRICT mode
```
"""
function infer_cpp_type(registry::TypeRegistry, cpp_type::String; context::String="")::String
    if isempty(cpp_type)
        return handle_unknown_cpp_type(registry, "", context == "" ? "empty type string" : context)
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
        julia_base = infer_cpp_type(registry, String(base_type); context=ptr_context)
        return "Ptr{$julia_base}"
    end

    # Parse reference types: T& or T &
    if endswith(clean_type, "&")
        base_type = strip(replace(clean_type, r"&$" => ""))
        ref_context = context == "" ? "reference base type" : "$context (reference to $base_type)"
        julia_base = infer_cpp_type(registry, String(base_type); context=ref_context)
        return "Ref{$julia_base}"
    end

    # Parse array types: T[N]
    array_match = match(r"^(.+)\[(\d+)\]$", clean_type)
    if !isnothing(array_match)
        elem_type = strip(array_match.captures[1])
        size = parse(Int, array_match.captures[2])
        arr_context = context == "" ? "array element type" : "$context (array of $elem_type)"
        julia_elem = infer_cpp_type(registry, String(elem_type); context=arr_context)
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
                elem_type = infer_cpp_type(registry, String(strip(parts[1])); context=arr_context)
                return "Ptr{Cvoid}"  # std::array has non-trivial ABI too
            end
        end

        # Handle std::pair<T1, T2> → Tuple{T1, T2} (POD-like, can be ccall'd)
        if template_name == "std::pair"
            parts = _split_template_args(template_args)
            if length(parts) == 2
                pair_ctx1 = context == "" ? "std::pair first type" : "$context (std::pair first)"
                pair_ctx2 = context == "" ? "std::pair second type" : "$context (std::pair second)"
                t1 = infer_cpp_type(registry, String(strip(parts[1])); context=pair_ctx1)
                t2 = infer_cpp_type(registry, String(strip(parts[2])); context=pair_ctx2)
                return "Tuple{$t1,$t2}"
            end
        end

        # Generic template fallback - unknown template type
        ctx = context == "" ? "unknown template type '$clean_type'" : "$context (unknown template '$clean_type')"
        return handle_unknown_cpp_type(registry, String(clean_type), ctx)
    end

    # Unknown type - no matching rules
    ctx = context == "" ? "type inference" : context
    return handle_unknown_cpp_type(registry, String(clean_type), ctx)
end

