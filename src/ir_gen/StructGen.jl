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

    kind = get(info, "kind", "")
    if kind == "union"
        return "!llvm.array<$(dwarf_size) x i8>"
    elseif kind == "enum"
        underlying = get(info, "underlying_type", "i32")
        mlir_t = map_cpp_type(underlying)
        # Fallback to i32 if mapping fails or returns struct alias
        if isempty(mlir_t) || startswith(mlir_t, "!llvm.struct")
            mlir_t = "i32"
        end
        return "!llvm.struct<\"$(name)\", ($(mlir_t))>"
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
    
    # 1. Prepare nodes and strip __enum__
    nodes = String[]
    node_map = Dict{String, Any}() # Name -> Info
    
    for (name, info) in structs
        # Skip standard types
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
    
    # 2. Build dependency graph
    # Adjacency: A -> B means A depends on B (A uses B by value)
    deps = Dict{String, Set{String}}()
    
    for name in nodes
        info = node_map[name]
        d = Set{String}()
        deps[name] = d
        
        # If enum/union, no deps usually (underlying type is primitive)
        kind = get(info, "kind", "")
        if kind == "enum" || kind == "union"
            continue
        end
        
        members = get(info, "members", [])
        for m in members
            t = get(m, "c_type", "void*")
            # If pointer, skip
            if endswith(t, "*") || contains(t, "*")
                continue
            end
            
            # Map type to check if it's a struct
            mlir_t = map_cpp_type(t)
            
            # Check if it is !llvm.struct<"Name">
            m_match = match(r"!llvm.struct<\"([^\"]+)\">", mlir_t)
            if m_match !== nothing
                dep_name = m_match.captures[1]
                # If dep_name is in our nodes, add dependency
                if haskey(node_map, dep_name) && dep_name != name
                    push!(d, dep_name)
                end
            end
        end
    end
    
    # 3. Topological Sort
    sorted_nodes = String[]
    visited = Set{String}()
    stack = Set{String}()
    
    function visit(n)
        if n in visited
            return
        end
        if n in stack
            # Cycle detected, break it
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
    
    # 4. Emit
    for name in sorted_nodes
        info = node_map[name]
        type_str = get_struct_type_string(name, info)
        
        # Opaque types cannot be instantiated as globals with body
        if endswith(type_str, "opaque>")
            continue
        end
        
        # Use a dummy function declaration to register the struct type definition
        println(io, "llvm.func @__def_$(name)($(type_str)) -> !llvm.void")
    end
    
    println(io, "")
    return String(take!(io))
end
end
