#!/usr/bin/env julia
# DWARFParser.jl - Extract vtable and type information from DWARF debug data
# The final piece for universal FFI - parse what the compiler already knows

module DWARFParser

using JSON

export parse_vtables, VirtualMethod, MemberInfo, ClassInfo, VtableInfo

"""
Information about a virtual method
"""
struct VirtualMethod
    name::String              # Method name (e.g., "foo")
    mangled_name::String      # Mangled name (e.g., "_ZN4Base3fooEv")
    slot::Int                 # Vtable slot index
    return_type::String       # Return type
    parameters::Vector{String} # Parameter types
end

"""
Information about a data member (field)
"""
struct MemberInfo
    name::String
    type_name::String
    offset::Int
end

"""
Information about a C++ class with virtual methods
"""
struct ClassInfo
    name::String                      # Class name
    vtable_ptr_offset::Int           # Offset of vptr in object (usually 0)
    base_classes::Vector{String}     # Immediate base classes
    virtual_methods::Vector{VirtualMethod}
    members::Vector{MemberInfo}      # Data members
    size::Int                        # Class size in bytes
end

"""
Complete vtable information from binary
"""
struct VtableInfo
    classes::Dict{String, ClassInfo}           # class_name => ClassInfo
    vtable_addresses::Dict{String, UInt64}    # class_name => vtable address
    method_addresses::Dict{String, UInt64}    # mangled_name => function address
end

"""
    parse_dwarf_output(dwarf_text::String) -> Dict{String, ClassInfo}

Parse llvm-dwarfdump output to extract class and vtable information.
"""
function parse_dwarf_output(dwarf_text::String)
    classes = Dict{String, ClassInfo}()

    # State machine for parsing
    current_class = nothing
    current_class_name = ""
    current_vptr_offset = 0
    current_size = 0
    base_classes = String[]
    virtual_methods = VirtualMethod[]
    members = MemberInfo[]
    
    # Method parsing state
    current_method_name = ""
    current_method_mangled = ""
    current_method_slot = -1
    is_virtual_method = false
    in_subprogram = false

    for line in split(dwarf_text, '\n')
        # Detect class type
        if contains(line, "DW_TAG_class_type") || contains(line, "DW_TAG_structure_type")
            # Save pending method from previous class context if valid
            if is_virtual_method && !isempty(current_method_name)
                push!(virtual_methods, VirtualMethod(
                    current_method_name,
                    current_method_mangled,
                    current_method_slot,
                    "void", 
                    String[]
                ))
            end

            # Save previous class if exists
            if !isnothing(current_class_name) && !isempty(current_class_name)
                classes[current_class_name] = ClassInfo(
                    current_class_name,
                    current_vptr_offset,
                    copy(base_classes),
                    copy(virtual_methods),
                    copy(members),
                    current_size
                )
            end

            # Reset state
            current_class_name = ""
            current_vptr_offset = 0
            current_size = 0
            empty!(base_classes)
            empty!(virtual_methods)
            empty!(members)
            
            # Reset method state
            current_method_name = ""
            current_method_mangled = ""
            current_method_slot = -1
            is_virtual_method = false
            in_subprogram = false
            continue
        end
        
        # Detect start of a subprogram (method) - reset method state
        if contains(line, "DW_TAG_subprogram")
            # If previous method was virtual, save it
            if is_virtual_method && !isempty(current_method_name)
                push!(virtual_methods, VirtualMethod(
                    current_method_name,
                    current_method_mangled,
                    current_method_slot,
                    "void", # TODO: Parse return type
                    String[] # TODO: Parse parameters
                ))
            end
            
            current_method_name = ""
            current_method_mangled = ""
            current_method_slot = -1
            is_virtual_method = false
            in_subprogram = true
        end
        
        # Parse Members
        # DW_TAG_member
        if contains(line, "DW_TAG_member")
             # We just hit a member tag. We'll extract its info in subsequent lines or if they are on the same line (rare).
             # Simple state tracking for member is needed? 
             # Actually, dwarfdump output is nested. We can parse lines that follow until next TAG.
             # But our loop is line-by-line. 
             # Let's assume we can capture "DW_AT_name" etc. while "in_class" context.
        end

        # If we hit another tag that isn't subprogram or formal parameter, we are likely out of subprogram
        if contains(line, "DW_TAG_") && !contains(line, "DW_TAG_subprogram") && !contains(line, "DW_TAG_formal_parameter") && !contains(line, "DW_TAG_unspecified_parameters")
             in_subprogram = false
        end

        # Extract class/member/method name
        if contains(line, "DW_AT_name") 
            m = match(r"DW_AT_name\s+\(\"([^\"]+)\"\)", line)
            if !isnothing(m)
                name = m.captures[1]
                if isempty(current_class_name)
                    current_class_name = name
                elseif in_subprogram
                    # It's a method name
                    current_method_name = name
                elseif !in_subprogram && !isempty(current_class_name)
                    # Likely a member name, but we need to match it with DW_TAG_member context.
                    # Limitations of this simple parser: it assumes state based on recent TAG.
                    # We need a robust way to know we are in a member.
                    # IMPROVEMENT: Use the indentation level or look at the preceding TAG line.
                end
            end
        end
        
        # IMPROVED MEMBER PARSING:
        # We need to capture member details when we see DW_TAG_member.
        # But we are streaming lines.
        # Let's use a "last_tag" variable.
    end
    
    # Re-implementing the loop with better state tracking
    return parse_dwarf_output_robust(dwarf_text)
end

function parse_dwarf_output_robust(dwarf_text::String)
    classes = Dict{String, ClassInfo}()

    current_class_name = ""
    current_vptr_offset = 0
    current_size = 0
    base_classes = String[]
    virtual_methods = VirtualMethod[]
    members = MemberInfo[]
    
    # Member parsing
    pending_member_name = ""
    pending_member_type = ""
    pending_member_offset = -1
    
    # Method parsing
    pending_method_name = ""
    pending_method_mangled = ""
    pending_method_slot = -1
    is_virtual_method = false
    
    # Context
    in_class = false
    context = :none # :class, :method, :member
    
    lines = split(dwarf_text, '\n')
    
    function commit_class()
        if !isempty(current_class_name)
            classes[current_class_name] = ClassInfo(
                current_class_name,
                current_vptr_offset,
                copy(base_classes),
                copy(virtual_methods),
                copy(members),
                current_size
            )
        end
        current_class_name = ""
        current_vptr_offset = 0
        current_size = 0
        empty!(base_classes)
        empty!(virtual_methods)
        empty!(members)
        in_class = false
    end
    
    function commit_method()
        if is_virtual_method && !isempty(pending_method_name)
            push!(virtual_methods, VirtualMethod(
                pending_method_name,
                pending_method_mangled,
                pending_method_slot,
                "void", 
                String[]
            ))
        end
        pending_method_name = ""
        pending_method_mangled = ""
        pending_method_slot = -1
        is_virtual_method = false
    end

    function commit_member()
        if !isempty(pending_member_name) && !isempty(pending_member_type) && pending_member_offset != -1
            push!(members, MemberInfo(
                pending_member_name,
                pending_member_type,
                pending_member_offset
            ))
        end
        pending_member_name = ""
        pending_member_type = ""
        pending_member_offset = -1
    end

    for line in lines
        # Determine Tag
        if contains(line, "DW_TAG_class_type") || contains(line, "DW_TAG_structure_type")
            commit_method()
            commit_member()
            commit_class() # Close previous class
            context = :class
            in_class = true
            continue
        end
        
        if contains(line, "DW_TAG_subprogram")
            commit_method()
            commit_member()
            context = :method
            continue
        end
        
        if contains(line, "DW_TAG_member")
            commit_method()
            commit_member()
            context = :member
            continue
        end
        
        if contains(line, "DW_TAG_inheritance")
            commit_method()
            commit_member()
            context = :inheritance
            continue
        end
        
        if contains(line, "DW_TAG_") && !contains(line, "DW_TAG_formal_parameter")
             # Some other tag, reset context if specific
             # But keep class context
        end
        
        # Parse Attributes based on context
        
        # 1. Name
        if contains(line, "DW_AT_name")
            m = match(r"DW_AT_name\s+\(\"([^\"]+)\"\)", line)
            if !isnothing(m)
                name = m.captures[1]
                if context == :class
                    current_class_name = name
                elseif context == :method
                    pending_method_name = name
                elseif context == :member
                    pending_member_name = name
                end
            end
        end
        
        # 2. Type (for members and inheritance)
        if (context == :member || context == :inheritance) && contains(line, "DW_AT_type")
             # Try to capture type name from comment: DW_AT_type (0x123 "double")
             m = match(r"DW_AT_type\s+\(0x[0-9a-fA-F]+\s+\"([^\"]+)\"\)", line)
             if !isnothing(m)
                 type_name = m.captures[1]
                 if context == :member
                     pending_member_type = type_name
                 elseif context == :inheritance
                     push!(base_classes, type_name)
                 end
             else
                 if context == :member
                    pending_member_type = "void*" 
                 end
             end
        end

        # 3. Data Member Location (offset)
        if context == :member && contains(line, "DW_AT_data_member_location")
            # Can be constant (0x08) or loclist. We handle simple constant.
            m = match(r"DW_AT_data_member_location\s+\((0x[0-9a-fA-F]+|\d+)\)", line)
            if !isnothing(m)
                val_str = m.captures[1]
                pending_member_offset = startswith(val_str, "0x") ?
                    parse(Int, val_str[3:end], base=16) :
                    parse(Int, val_str)
            end
        end
        
        # 4. Class Size
        if context == :class && contains(line, "DW_AT_byte_size")
            m = match(r"DW_AT_byte_size\s+\((0x[0-9a-fA-F]+|\d+)\)", line)
            if !isnothing(m)
                val_str = m.captures[1]
                current_size = startswith(val_str, "0x") ?
                    parse(Int, val_str[3:end], base=16) :
                    parse(Int, val_str)
            end
        end
        
        # 5. Method Info
        if context == :method
            if contains(line, "DW_AT_virtuality") && (contains(line, "virtual") || contains(line, "(0x01)"))
                is_virtual_method = true
            end
            
            if contains(line, "DW_AT_vtable_elem_location")
                m = match(r"DW_OP_constu\s+(0x[0-9a-fA-F]+|\d+)", line)
                if !isnothing(m)
                     val_str = m.captures[1]
                     pending_method_slot = startswith(val_str, "0x") ?
                        parse(Int, val_str[3:end], base=16) :
                        parse(Int, val_str)
                end
            end
            
            if contains(line, "linkage_name")
                m = match(r"linkage_name\s+\(\"([^\"]+)\"\)", line)
                if !isnothing(m)
                    pending_method_mangled = m.captures[1]
                end
            end
        end
    end

    # Final commits
    commit_member()
    commit_method()
    commit_class()

    return classes
end

"""
    parse_symbol_table(nm_output::String) -> Tuple{Dict{String, UInt64}, Dict{String, UInt64}}

Parse nm output to extract vtable and method addresses.
Returns (vtable_addresses, method_addresses).
"""
function parse_symbol_table(nm_output::String)
    vtable_addrs = Dict{String, UInt64}()
    method_addrs = Dict{String, UInt64}()

    for line in split(nm_output, '\n')
        # Parse vtable addresses
        # Format: "0000000000003d20 V vtable for Base"
        m = match(r"^([0-9a-fA-F]+)\s+[VW]\s+vtable for (.+)$", line)
        if !isnothing(m)
            addr = parse(UInt64, m.captures[1], base=16)
            class_name = m.captures[2]
            vtable_addrs[class_name] = addr
            continue
        end

        # Parse method addresses
        # Format: "00000000000012a0 W Base::foo()"
        m = match(r"^([0-9a-fA-F]+)\s+[TW]\s+(.+)$", line)
        if !isnothing(m)
            addr = parse(UInt64, m.captures[1], base=16)
            mangled = m.captures[2]
            method_addrs[mangled] = addr
        end
    end

    return (vtable_addrs, method_addrs)
end

"""
    read_vtable_data(binary_path::String, vtable_addr::UInt64, num_entries::Int) -> Vector{UInt64}

Read actual vtable function pointers from binary at given address.
"""
function read_vtable_data(binary_path::String, vtable_addr::UInt64, num_entries::Int)
    # Use objdump to read the .data.rel.ro section
    cmd = `objdump -s --section=.data.rel.ro $binary_path`
    output = read(cmd, String)

    # Parse the hex dump to find our vtable
    # Format: " 3d20 00000000 00000000 503d0000 00000000"
    #         addr  <------- 16 bytes of data --------->

    ptrs = UInt64[]

    # This is simplified - real implementation would:
    # 1. Find the section containing vtable_addr
    # 2. Calculate offset within section
    # 3. Read num_entries * 8 bytes
    # 4. Parse as little-endian UInt64s

    return ptrs
end

"""
    parse_vtables(binary_path::String) -> VtableInfo

Extract complete vtable information from a binary using DWARF and symbol table.

# Arguments
- `binary_path`: Path to compiled binary with debug info

# Returns
- `VtableInfo` containing classes, vtable addresses, and method addresses
"""
function parse_vtables(binary_path::String)
    if !isfile(binary_path)
        error("Binary not found: $binary_path")
    end

    println("Parsing DWARF debug info from: $binary_path")

    # Extract DWARF info
    dwarf_cmd = `llvm-dwarfdump --debug-info $binary_path`
    dwarf_output = read(dwarf_cmd, String)

    # Extract symbol table
    nm_cmd = `nm $binary_path`
    nm_output = read(nm_cmd, String)

    # Parse both
    classes = parse_dwarf_output(dwarf_output)
    (vtable_addrs, method_addrs) = parse_symbol_table(nm_output)

    println("Found $(length(classes)) classes with virtual methods")
    println("Found $(length(vtable_addrs)) vtables")
    println("Found $(length(method_addrs)) methods")

    return VtableInfo(classes, vtable_addrs, method_addrs)
end

"""
    export_vtable_json(vtinfo::VtableInfo, output_path::String)

Export vtable information to JSON for inspection or use by other tools.
"""
function export_vtable_json(vtinfo::VtableInfo, output_path::String)
    data = Dict(
        "classes" => Dict(
            name => Dict(
                "size" => info.size,
                "vtable_ptr_offset" => info.vtable_ptr_offset,
                "base_classes" => info.base_classes,
                "members" => [
                    Dict(
                        "name" => m.name,
                        "type" => m.type_name,
                        "offset" => m.offset
                    ) for m in info.members
                ],
                "virtual_methods" => [
                    Dict(
                        "name" => m.name,
                        "mangled_name" => m.mangled_name,
                        "slot" => m.slot
                    ) for m in info.virtual_methods
                ]
            ) for (name, info) in vtinfo.classes
        ),
        "vtable_addresses" => Dict(
            name => string("0x", string(addr, base=16))
            for (name, addr) in vtinfo.vtable_addresses
        ),
        "method_addresses" => Dict(
            name => string("0x", string(addr, base=16))
            for (name, addr) in vtinfo.method_addresses
        )
    )

    open(output_path, "w") do f
        JSON.print(f, data, 2)
    end

    println("Exported vtable info to: $output_path")
end

end # module DWARFParser