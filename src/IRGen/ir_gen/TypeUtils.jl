module TypeUtils

export map_cpp_type, get_llvm_signature, get_stl_container_size

"""
    map_cpp_type(type_str::String) -> String

Map C++ type string to MLIR type.
"""
function map_cpp_type(type_str::String)
    # Basic types
    if type_str == "void"
        return "" # Void return usually means no value
    elseif type_str == "int" || type_str == "int32_t" || type_str == "int32" || type_str == "Cint" || type_str == "unsigned int" || type_str == "uint32_t" || type_str == "uint32" || type_str == "Cuint"
        return "i32"
    elseif type_str == "long" || type_str == "long long" || type_str == "int64_t" || type_str == "int64" || type_str == "size_t" || type_str == "Csize_t" || type_str == "Clong" || type_str == "unsigned long" || type_str == "uint64_t" || type_str == "uint64" || type_str == "Culong" || type_str == "unsigned long long"
        return "i64"
    elseif type_str == "float" || type_str == "float32" || type_str == "Cfloat"
        return "f32"
    elseif type_str == "double" || type_str == "float64" || type_str == "Cdouble"
        return "f64"
    elseif type_str == "bool" || type_str == "_Bool" || type_str == "Bool"
        return "i1"
    # `uint8_t` and the `signed/unsigned char` spellings were missing here. The
    # gap was invisible while arrays mapped to `!llvm.ptr` without ever looking
    # at the element type; once they recurse, `uint8_t[16]` resolves its element
    # through this function and `uint8_t` fell to the struct-name branch below as
    # `!llvm.struct<"uint8_t">` (undefined), while `unsigned char` missed even
    # that (the space fails the identifier regex) and became `!llvm.ptr`.
    # ggml's quant blocks are declared entirely in these spellings.
    elseif type_str == "char" || type_str == "int8_t" || type_str == "int8" ||
           type_str == "uint8" || type_str == "uint8_t" || type_str == "UInt8" ||
           type_str == "unsigned char" || type_str == "signed char" || type_str == "Cchar" ||
           type_str == "Cuchar"
        return "i8"
    elseif type_str == "short" || type_str == "int16_t" || type_str == "int16" || type_str == "unsigned short" || type_str == "uint16_t" || type_str == "uint16"
        return "i16"
    # C built-in types that DWARF may strip qualifiers from
    elseif type_str == "ptrdiff_t" || type_str == "ssize_t" || type_str == "intptr_t" || type_str == "uintptr_t"
        return "i64"
    elseif type_str == "complex"
        # _Complex T — DWARF strips the element type; use opaque pointer since size is unknown
        return "!llvm.ptr"
    # Fixed-size arrays — MUST be tested before the pointer branch below, or
    # `char *[4]` matches on the '*' and collapses to a single pointer.
    #
    # There was no array case at all: `int8_t[32]` matched nothing, fell through
    # to the `!llvm.ptr` fallback, and claimed 8 bytes at align 8 instead of 32
    # bytes at align 1. That is why every ggml quant block failed layout —
    # `block_q8_0` reported `qs` "needs align 8 but sits at offset 2", and the
    # neighbouring blocks reported overlaps and past-the-end members for the
    # same reason. It also matters more than usual here because DWARF gives
    # these members `size = 0`, so the member record cannot supply the size
    # either; the mapped type is the only source of truth for the layout.
    #
    # The element must be a plain type name (optionally pointer-qualified). The
    # paren exclusion keeps `int (*)[4]` — a POINTER to an array, 8 bytes — out
    # of this branch; it is not an array member.
    elseif (am = match(r"^([A-Za-z_][A-Za-z0-9_ ]*?\s*\**)\s*((?:\[\d*\])+)$", type_str)) !== nothing &&
           !contains(type_str, "(")
        elem = map_cpp_type(String(strip(String(am.captures[1]))))
        isempty(elem) && return "!llvm.ptr"   # T = void: no sensible element
        # Innermost dimension binds tightest: `T[4][8]` is 4 arrays of 8 T.
        dims = [isempty(d.captures[1]) ? 0 : parse(Int, d.captures[1])
                for d in eachmatch(r"\[(\d*)\]", String(am.captures[2]))]
        t = elem
        for n in Iterators.reverse(dims)
            t = "!llvm.array<$n x $t>"
        end
        return t
    elseif endswith(type_str, "*") || contains(type_str, "*") ||
           endswith(type_str, "&") || contains(type_str, "&") || # references are pointers in the ABI
           type_str == "unknown" # simplified pointer check
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

"""
    get_llvm_signature(method) -> (String, String)

Get LLVM return type and argument types string from a method object.
Expects object to have `return_type` and `parameters` fields.
"""
function get_llvm_signature(method)
    # Map return type
    ret_type = map_cpp_type(getfield(method, :return_type))
    
    # Map parameters
    # Implicit 'this' pointer is always first arg for virtual methods
    arg_types = ["!llvm.ptr"]
    
    for param_type in getfield(method, :parameters)
        push!(arg_types, map_cpp_type(param_type))
    end
    
    return (ret_type, join(arg_types, ", "))
end

"""
    get_stl_container_size(c_type::String) -> Int

Returns the exact ABI byte size of common STL containers on x86_64 SysV.
Returns 0 if unknown.
"""
function get_stl_container_size(c_type::String)::Int
    clean = strip(replace(c_type, r"^(const|struct|class|union)\b" => ""))
    clean = strip(replace(clean, r"[*&]+$" => ""))

    if startswith(clean, "std::vector") || startswith(clean, "vector<")
        return 24
    elseif startswith(clean, "std::basic_string") || startswith(clean, "std::string") || startswith(clean, "basic_string<") || startswith(clean, "string")
        return 32
    elseif startswith(clean, "std::shared_ptr") || startswith(clean, "shared_ptr<")
        return 16
    elseif startswith(clean, "std::unique_ptr") || startswith(clean, "unique_ptr<")
        return 8
    elseif startswith(clean, "std::map") || startswith(clean, "map<") || startswith(clean, "std::set") || startswith(clean, "set<")
        return 48
    elseif startswith(clean, "std::unordered_map") || startswith(clean, "unordered_map<") || startswith(clean, "std::unordered_set") || startswith(clean, "unordered_set<")
        return 56
    elseif startswith(clean, "std::list") || startswith(clean, "list<")
        return 24
    elseif startswith(clean, "std::deque") || startswith(clean, "deque<")
        return 80
    end
    return 0
end

end
