module TypeUtils

export map_cpp_type, get_llvm_signature

"""
    map_cpp_type(type_str::String) -> String

Map C++ type string to MLIR type.
"""
function map_cpp_type(type_str::String)
    # Basic types
    if type_str == "void"
        return "" # Void return usually means no value
    elseif type_str == "int" || type_str == "int32_t" || type_str == "Cint" || type_str == "unsigned int" || type_str == "uint32_t" || type_str == "Cuint"
        return "i32"
    elseif type_str == "long" || type_str == "long long" || type_str == "int64_t" || type_str == "size_t" || type_str == "Csize_t" || type_str == "Clong" || type_str == "unsigned long" || type_str == "uint64_t" || type_str == "Culong" || type_str == "unsigned long long"
        return "i64"
    elseif type_str == "float" || type_str == "Cfloat"
        return "f32"
    elseif type_str == "double" || type_str == "Cdouble"
        return "f64"
    elseif type_str == "bool" || type_str == "Bool"
        return "i1"
    elseif type_str == "char" || type_str == "int8_t" || type_str == "UInt8" || type_str == "Cchar"
        return "i8"
    elseif endswith(type_str, "*") || contains(type_str, "*") || type_str == "unknown" # simplified pointer check
        return "!llvm.ptr"
    # Struct types?
    # If it matches a known struct name, we should return !llvm.struct<name> equivalent?
    # For now, let's treat unknown types as !llvm.ptr, but we might need to change this for pass-by-value.
    else
        # Check if it looks like a struct name (Alphanumeric)
        if occursin(r"^[A-Za-z0-9_]+$", type_str) && type_str != "Any"
             # Return as struct type alias (we will define these)
             return "!llvm.struct<\"$(type_str)\">"
        end
    end

    # Fallback
    # @warn "Unknown C++ type encountered: $type_str. Defaulting to !llvm.ptr."
    return "!llvm.ptr" 
end

end
