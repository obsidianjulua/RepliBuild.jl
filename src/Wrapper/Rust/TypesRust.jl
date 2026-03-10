# =============================================================================
# TYPE SYSTEM - Comprehensive Rust to Julia Type Mapping
# =============================================================================

"""
    infer_rust_type(registry::TypeRegistry, rust_type::String; context::String="")::String

Infer Julia type from Rust type string using comprehensive rules according to DWARF output.

# Type Mapping
Rust types in DWARF are generally much cleaner than C++.
- i8, u8, i16, u16, i32, u32, i64, u64 -> Int8, UInt8, ...
- f32, f64 -> Float32, Float64
- bool -> Bool
- usize, isize -> Csize_t, Cssize_t
- *const T, *mut T, &T -> Ptr{T}
- () -> Cvoid
"""
function infer_rust_type(registry::TypeRegistry, rust_type::AbstractString; context::String="")::String
    # Clean whitespace
    clean_type = strip(rust_type)

    # Primitive types
    base_types = Dict{String,String}(
        "i8" => "Int8",
        "u8" => "UInt8",
        "i16" => "Int16",
        "u16" => "UInt16",
        "i32" => "Int32",  # Or Cint
        "u32" => "UInt32", # Or Cuint
        "i64" => "Int64",
        "u64" => "UInt64",
        "f32" => "Float32",
        "f64" => "Float64",
        "bool" => "Bool",
        "usize" => "Csize_t",
        "isize" => "Cssize_t",
        "()" => "Cvoid",
        "void" => "Cvoid"
    )

    if haskey(base_types, clean_type)
        return base_types[clean_type]
    end

    # Check for pointers and references
    # Example: "*const i8", "*mut i32", "i8*", "Point*"
    if occursin("*const ", clean_type) || occursin("*mut ", clean_type) || endswith(clean_type, "*")
        base = replace(clean_type, r"\*const |\*mut " => "")
        base = replace(base, "*" => "")
        base = strip(base)
        if base == "c_char" || base == "i8"
            return "Cstring"
        elseif base == "void" || base == "()"
            return "Ptr{Cvoid}"
        else
            base_julia = infer_rust_type(registry, base, context=context)
            if base_julia == "Any" || base_julia == "_UnsafeUnknown"
                return "Ptr{Cvoid}"
            end
            return "Ptr{$base_julia}"
        end
    end

    if startswith(clean_type, "&")
        # References map to Ptr in the ABI level
        base = strip(clean_type[2:end])
        base_julia = infer_rust_type(registry, base, context=context)
        if base_julia == "Any" || base_julia == "_UnsafeUnknown"
            return "Ptr{Cvoid}"
        end
        return "Ptr{$base_julia}"
    end

    # Strip Rust module paths if present (e.g., lib::Point -> Point)
    if occursin("::", clean_type)
        clean_type = split(clean_type, "::")[end]
    end

    # If it's a known struct/enum
    if haskey(registry.custom_types, clean_type)
        return registry.custom_types[clean_type]
    end

    # Fallback for structs/opaque types
    if registry.allow_unknown_structs
        # Temporary heuristic: if it looks like an enum or struct
        if clean_type == "Color" || clean_type == "Result" || clean_type == "Option"
            return "Cint"
        end
        if clean_type == "CapacityOverflow" || clean_type == "AllocError" || clean_type == "FormattingOptions" || clean_type == "Count"
            return "Cvoid"
        end
        return make_rust_identifier(clean_type)
    end

    if registry.strictness == 0 # STRICT
        error("Unknown Rust type: '$clean_type' in $context")
    elseif registry.strictness == 1 # WARN
        @warn "Unknown Rust type '$clean_type' in $context"
        return "_UnsafeUnknown"
    else
        return "Any"
    end
end
