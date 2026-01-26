module StructGen

using ..TypeUtils

export generate_struct_definitions, get_struct_type_string

"""
    get_struct_type_string(name::String, info::Any) -> String

Get the LLVM IR type string for a struct (e.g. `!llvm.struct<"Name", ...>`).
"""
function get_struct_type_string(name::String, info::Any)
    dwarf_size_str = get(info, "byte_size", "0")
    dwarf_size = try
        startswith(dwarf_size_str, "0x") ? parse(Int, dwarf_size_str) : parse(Int, dwarf_size_str)
    catch
        0
    end

    if get(info, "kind", "") == "union"
        return "!llvm.array<$(dwarf_size) x i8>"
    end
    
    members = get(info, "members", [])
    member_types = String[]
    sum_size = 0
    
    for m in members
        t = get(m, "c_type", "void*")
        # Recursion risk? If nested structs, we rely on map_cpp_type returning !llvm.struct<"Name"> (opaque)
        # But we want definition.
        # If we use full string recursively, we might loop.
        # So nested structs should be opaque alias, but top level usage should be full?
        # No, LLVM struct definition is flat.
        
        mlir_t = map_cpp_type(t)
        push!(member_types, mlir_t)
        
        m_size = try
            s = get(m, "size", 0)
            s isa String ? parse(Int, s) : s
        catch
            0
        end
        sum_size += m_size
    end
    
    is_packed = (sum_size == dwarf_size) && (dwarf_size > 0)
    packed_attr = is_packed ? "packed " : ""
    
    if isempty(member_types)
         return "!llvm.struct<\"$(name)\", opaque>"
    else
         return "!llvm.struct<\"$(name)\", $(packed_attr)($(join(member_types, ", ")))>"
    end
end

"""
    generate_struct_definitions(structs::Any) -> String

Generate LLVM struct type definitions via a registration function.
"""
function generate_struct_definitions(structs::Any)
    io = IOBuffer()
    
    println(io, "// Struct Definitions")
    
    for (name, info) in structs
        # Skip standard types
        if name in ["int", "float", "double", "bool", "char", "void"]
            continue
        end

        type_str = get_struct_type_string(name, info)
        
        # Use llvm.mlir.global to define the type body in the context
        println(io, "llvm.mlir.global private @__def_$(name)() : $(type_str)")
    end
    
    println(io, "")
    return String(take!(io))
end

end
