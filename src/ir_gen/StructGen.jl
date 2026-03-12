module StructGen

using ..TypeUtils

export generate_struct_definitions, get_struct_type_string, get_struct_definition_string, is_struct_packed, get_julia_offsets, get_llvm_equivalent_type_string, get_llvm_aligned_type_string

"""
    is_struct_packed(info::Any) -> Bool

Determine if a struct is packed (dwarf_size == sum(member_sizes)).
"""
function is_struct_packed(info::Any)
    dwarf_size_str = get(info, "byte_size", "0")
    dwarf_size = try
        startswith(dwarf_size_str, "0x") ? parse(Int, dwarf_size_str[3:end], base=16) : parse(Int, dwarf_size_str)
    catch
        0
    end
    
    if dwarf_size == 0
        return false
    end

    kind = get(info, "kind", "")
    if kind == "union" || kind == "enum"
        return false
    end
    
    members = get(info, "members", [])
    sum_size = 0
    
    for m in members
        m_size = try
            s = get(m, "size", 0)
            s isa String ? parse(Int, s) : s
        catch
            0
        end
        sum_size += m_size
    end
    
    return sum_size == dwarf_size
end

"""
    get_julia_offsets(info::Any, is_packed::Bool=false) -> Vector{Int}

Calculate the byte offsets of struct members according to Julia/C alignment rules.
Returns a vector of start offsets for each member.
"""
function get_julia_offsets(info::Any, is_packed::Bool=false)
    members = get(info, "members", [])
    offsets = Int[]
    current_offset = 0
    
    for m in members
        m_size = try
            s = get(m, "size", 0)
            s isa String ? parse(Int, s) : s
        catch
            0
        end
        
        if !is_packed
            # Alignment heuristic: alignment = min(size, 8)
            # Cap at 8 bytes (64-bit) usually
            align = m_size > 8 ? 8 : m_size
            align = align == 0 ? 1 : align
            
            # Add padding
            padding = (align - (current_offset % align)) % align
            current_offset += padding
        end
        
        push!(offsets, current_offset)
        
        current_offset += m_size
    end
    
    return offsets
end

"""
    get_struct_definition_string(name::String, info::Any) -> String

Get the MLIR type definition string for a struct.
"""
function get_struct_definition_string(name::String, info::Any)
    dwarf_size_str = get(info, "byte_size", "0")
    dwarf_size = try
        startswith(dwarf_size_str, "0x") ? parse(Int, dwarf_size_str[3:end], base=16) : parse(Int, dwarf_size_str)
    catch
        0
    end

    kind = get(info, "kind", "")
    if kind == "union"
        return "!llvm.array<$(dwarf_size) x i8>"
    elseif kind == "enum"
        underlying = get(info, "underlying_type", "i32")
        mlir_t = map_cpp_type(underlying)
        if isempty(mlir_t) || startswith(mlir_t, "!llvm.struct")
            mlir_t = "i32"
        end
        return "!llvm.struct<\"$(name)\", ($(mlir_t))>"
    end
    
    is_packed = is_struct_packed(info)
    
    members = get(info, "members", [])
    member_types = String[]
    sum_size = 0

    for m in members
        t = get(m, "c_type", "void*")
        m_size = try
            s = get(m, "size", 0)
            s isa String ? parse(Int, s) : s
        catch
            0
        end
        sum_size += m_size
        
        mlir_t = map_cpp_type(t)
        # Convert bare named struct references (!llvm.struct<"Name">) to aliases (!Struct_Name)
        # because bare named struct refs are only valid in recursive struct contexts in MLIR
        if startswith(mlir_t, "!llvm.struct<\"") && endswith(mlir_t, "\">")
            s_name = mlir_t[15:end-2]
            safe_name = replace(s_name, r"[^a-zA-Z0-9_]" => "_")
            mlir_t = "!Struct_$(safe_name)"
        end
        push!(member_types, mlir_t)
    end
    
    if !is_packed && dwarf_size > 0 && sum_size < dwarf_size
        # Struct size is larger than sum of parsed members (missing fields/unions or trailing padding).
        # We append a byte array to make up the difference so ABI sizing matches.
        push!(member_types, "!llvm.array<$(dwarf_size - sum_size) x i8>")
    end

    if isempty(member_types)
         return "!llvm.struct<\"$(name)\", opaque>"
    end

    # Emit !jlcs.c_struct for all defined structs (packed or unpacked)
    # to avoid dialect nesting errors (!llvm.struct cannot contain !jlcs.c_struct)
    offsets = get_julia_offsets(info, is_packed) # get accurate offsets
    
    # Format: !jlcs.c_struct<"Name", [types], [[offsets]], packed=true/false>
    types_str = join(member_types, ", ")
    
    offsets_typed = ["$(o) : i64" for o in offsets]
    offsets_str = "[$(join(offsets_typed, ", "))]"
    
    packed_val = is_packed ? "true" : "false"
    return "!jlcs.c_struct<\"$(name)\", [$(types_str)], [$(offsets_str)], packed = $(packed_val)>"
end

"""
    get_struct_type_string(name::String, info::Any) -> String

Get the MLIR type reference string (alias).
"""
function get_struct_type_string(name::String, info::Any)
    def_str = get_struct_definition_string(name, info)
    if endswith(def_str, "opaque>")
        return def_str
    end
    # Sanitize name for alias
    safe_name = replace(name, r"[^a-zA-Z0-9_]" => "_")
    return "!Struct_$(safe_name)"
end

"""
    get_llvm_equivalent_type_string(name::String, info::Any) -> String

Get the LLVM literal struct type string corresponding to the struct.
Used for constructing values of packed structs.
"""
function get_llvm_equivalent_type_string(name::String, info::Any)
    members = get(info, "members", [])
    member_types = String[]
    sum_size = 0
    
    for m in members
        t = get(m, "c_type", "void*")
        m_size = try
            s = get(m, "size", 0)
            s isa String ? parse(Int, s) : s
        catch
            0
        end
        sum_size += m_size
        mlir_t = map_cpp_type(t)
        push!(member_types, mlir_t)
    end
    
    dwarf_size_str = get(info, "byte_size", "0")
    dwarf_size = try
        startswith(dwarf_size_str, "0x") ? parse(Int, dwarf_size_str[3:end], base=16) : parse(Int, dwarf_size_str)
    catch
        0
    end

    is_packed = is_struct_packed(info)
    packed_attr = is_packed ? "packed " : ""
    
    if !is_packed && dwarf_size > 0 && sum_size < dwarf_size
        push!(member_types, "!llvm.array<$(dwarf_size - sum_size) x i8>")
    end
    
    if isempty(member_types)
         return "!llvm.struct<\"$(name)\", opaque>" # Fallback
    else
         # Return a literal struct (no name)
         return "!llvm.struct<$(packed_attr)($(join(member_types, ", "))) >"
    end
end

"""
    get_llvm_aligned_type_string(name::String, info::Any) -> String

Get an LLVM literal struct type string WITHOUT packed attribute.
Used for thunk return types so Julia can read the struct with natural alignment.
"""
function get_llvm_aligned_type_string(name::String, info::Any)
    members = get(info, "members", [])
    member_types = String[]
    sum_size = 0

    for m in members
        t = get(m, "c_type", "void*")
        m_size = try
            s = get(m, "size", 0)
            s isa String ? parse(Int, s) : s
        catch
            0
        end
        sum_size += m_size
        mlir_t = map_cpp_type(t)
        push!(member_types, mlir_t)
    end
    
    dwarf_size_str = get(info, "byte_size", "0")
    dwarf_size = try
        startswith(dwarf_size_str, "0x") ? parse(Int, dwarf_size_str[3:end], base=16) : parse(Int, dwarf_size_str)
    catch
        0
    end
    
    is_packed = is_struct_packed(info)

    if !is_packed && dwarf_size > 0 && sum_size < dwarf_size
        push!(member_types, "!llvm.array<$(dwarf_size - sum_size) x i8>")
    end

    if isempty(member_types)
         return "!llvm.struct<\"$(name)\", opaque>"
    else
         return "!llvm.struct<($(join(member_types, ", "))) >"
    end
end

"""
    generate_struct_definitions(structs::Any) -> (String, String)

Generate LLVM/JLCS struct type aliases and registration functions.
Returns (aliases_ir, registrations_ir).
"""
function generate_struct_definitions(structs::Any)
    io_aliases = IOBuffer()
    io_regs = IOBuffer()
    
    println(io_aliases, "// Struct Aliases")
    println(io_regs, "// Struct Definitions (Registration)")
    
    nodes = String[]
    node_map = Dict{String, Any}() # Name -> Info
    
    for (name, info) in structs
        if name in ["int", "float", "double", "bool", "char", "void"]
            continue
        end

        effective_name = name
        if startswith(name, "__enum__")
            effective_name = replace(name, "__enum__" => "")
        end
        
        push!(nodes, effective_name)
        node_map[effective_name] = info
    end
    
    deps = Dict{String, Set{String}}()
    
    for name in nodes
        info = node_map[name]
        d = Set{String}()
        deps[name] = d
        
        kind = get(info, "kind", "")
        if kind == "enum" || kind == "union"
            continue
        end
        
        members = get(info, "members", [])
        for m in members
            t = get(m, "c_type", "void*")
            if endswith(t, "*") || contains(t, "*")
                continue
            end
            
            mlir_t = map_cpp_type(t)
            # Use triple-quoted regex to avoid escape issues
            m_match = match(r"""!llvm.struct<\"([^\"]+)\">""", mlir_t)
            if m_match !== nothing
                dep_name = m_match.captures[1]
                if haskey(node_map, dep_name) && dep_name != name
                    push!(d, dep_name)
                end
            end
        end
    end
    
    sorted_nodes = String[]
    visited = Set{String}()
    stack = Set{String}()
    
    function visit(n)
        if n in visited
            return
        end
        if n in stack
            return
        end
        
        push!(stack, n)
        if haskey(deps, n)
            for d in deps[n]
                visit(d)
            end
        end
        delete!(stack, n)
        
        push!(visited, n)
        push!(sorted_nodes, n)
    end

    for n in nodes
        visit(n)
    end
    
    # Collect all alias names that will be defined so we can validate references
    defined_aliases = Set{String}()
    for name in sorted_nodes
        info = node_map[name]
        def_str = get_struct_definition_string(name, info)
        if !endswith(def_str, "opaque>")
            safe_name = replace(name, r"[^a-zA-Z0-9_]" => "_")
            push!(defined_aliases, "!Struct_$(safe_name)")
        end
    end

    for name in sorted_nodes
        info = node_map[name]
        def_str = get_struct_definition_string(name, info)
        
        if endswith(def_str, "opaque>")
            continue
        end
        
        # Replace any undefined alias references with !llvm.ptr
        def_str = replace(def_str, r"!Struct_[A-Za-z0-9_]+" => m -> m in defined_aliases ? m : "!llvm.ptr")

        # Emit alias
        safe_name = replace(name, r"[^a-zA-Z0-9_]" => "_")
        alias_name = "!Struct_$(safe_name)"
        println(io_aliases, "$(alias_name) = $(def_str)")

        # Use a dummy function to register usage (safe_name avoids <> in func name)
        println(io_regs, "func.func private @__def_$(safe_name)($(alias_name)) -> ()")
    end
    
    return (String(take!(io_aliases)), String(take!(io_regs)))
end
end