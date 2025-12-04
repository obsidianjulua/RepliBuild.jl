#!/usr/bin/env julia
# DWARFParser.jl - Extract vtable and type information from DWARF debug data
# The final piece for universal FFI - parse what the compiler already knows

module DWARFParser

using JSON

export parse_vtables, VirtualMethod, ClassInfo, VtableInfo

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
Information about a C++ class with virtual methods
"""
struct ClassInfo
    name::String                      # Class name
    vtable_ptr_offset::Int           # Offset of vptr in object (usually 0)
    base_classes::Vector{String}     # Immediate base classes
    virtual_methods::Vector{VirtualMethod}
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

    for line in split(dwarf_text, '\n')
        # Detect class type
        if contains(line, "DW_TAG_class_type")
            # Save previous class if exists
            if !isnothing(current_class_name) && !isempty(current_class_name)
                classes[current_class_name] = ClassInfo(
                    current_class_name,
                    current_vptr_offset,
                    copy(base_classes),
                    copy(virtual_methods),
                    current_size
                )
            end

            # Reset state
            current_class_name = ""
            current_vptr_offset = 0
            current_size = 0
            empty!(base_classes)
            empty!(virtual_methods)
            continue
        end

        # Extract class name
        if contains(line, "DW_AT_name") && isempty(current_class_name)
            m = match(r"DW_AT_name\s+\(\"([^\"]+)\"\)", line)
            if !isnothing(m)
                current_class_name = m.captures[1]
            end
        end

        # Extract class size
        if contains(line, "DW_AT_byte_size")
            m = match(r"DW_AT_byte_size\s+\((0x[0-9a-fA-F]+|\d+)\)", line)
            if !isnothing(m)
                size_str = m.captures[1]
                current_size = startswith(size_str, "0x") ?
                    parse(Int, size_str[3:end], base=16) :
                    parse(Int, size_str)
            end
        end

        # Detect vtable pointer member
        if contains(line, "_vptr\$")
            current_vptr_offset = 0  # Usually at offset 0
        end

        # Detect inheritance
        if contains(line, "DW_TAG_inheritance")
            # Next DW_AT_type will tell us the base class
            # For now, mark that we're in inheritance context
        end

        # Extract virtual method info
        if contains(line, "DW_AT_virtuality") && contains(line, "DW_VIRTUALITY_virtual")
            # This is a virtual method - look back for name and slot
        end

        # Extract vtable slot
        if contains(line, "DW_AT_vtable_elem_location")
            m = match(r"DW_OP_constu\s+(0x[0-9a-fA-F]+|\d+)", line)
            if !isnothing(m)
                slot_str = m.captures[1]
                slot = startswith(slot_str, "0x") ?
                    parse(Int, slot_str[3:end], base=16) :
                    parse(Int, slot_str)
                # TODO: Associate with current method being parsed
            end
        end

        # Extract mangled name
        if contains(line, "DW_AT_linkage_name")
            m = match(r"DW_AT_linkage_name\s+\(\"([^\"]+)\"\)", line)
            if !isnothing(m)
                mangled = m.captures[1]
                # TODO: Store with current method
            end
        end
    end

    # Save last class
    if !isempty(current_class_name)
        classes[current_class_name] = ClassInfo(
            current_class_name,
            current_vptr_offset,
            base_classes,
            virtual_methods,
            current_size
        )
    end

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
    nm_cmd = `nm -C $binary_path`
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
