#!/usr/bin/env julia
# Binary.jl - Binary introspection tools (nm, objdump, readelf, dwarfdump)
# Wraps existing Compiler.jl functionality and adds new binary analysis tools

# ============================================================================
# SYMBOL ANALYSIS
# ============================================================================

"""
    symbols(binary_path::String; filter=:all, demangled=true)

Extract symbols from a binary using nm.

Wraps the existing `Compiler.extract_symbols_from_binary()` and returns
structured `SymbolInfo` objects.

# Arguments
- `binary_path::String` - Path to binary file
- `filter::Symbol` - Filter symbols (:all, :functions, :data, :weak)
- `demangled::Bool` - Return demangled names (default: true)

# Returns
Vector{SymbolInfo}

# Examples
```julia
# Get all function symbols
syms = symbols("lib.so", filter=:functions)

# Get all symbols with mangling
syms = symbols("lib.so", demangled=false)

# Export to CSV
export_csv(syms, "symbols.csv")
```
"""
function symbols(binary_path::String; filter::Symbol=:all, demangled::Bool=true)
    # Validate binary exists
    if !isfile(binary_path)
        error("Binary not found: $binary_path")
    end

    # Check if nm tool is available
    nm_tool = LLVMEnvironment.get_tool("nm")
    if isempty(nm_tool)
        nm_tool = "nm"  # Fallback to system nm
    end

    # Use existing Compiler function to extract symbols
    raw_symbols = Compiler.extract_symbols_from_binary(binary_path)

    # Convert to SymbolInfo structs
    symbol_infos = SymbolInfo[]

    for sym_dict in raw_symbols
        name = get(sym_dict, "mangled", "")
        demangled_name = get(sym_dict, "demangled", name)
        address = get(sym_dict, "address", "0x0")
        sym_type = get(sym_dict, "type", "T")

        # Map symbol type
        type_symbol = if sym_type == "T"
            :function
        elseif sym_type == "D" || sym_type == "B"
            :data
        elseif sym_type == "W" || sym_type == "w"
            :weak
        else
            :other
        end

        # Apply filter
        if filter != :all
            if filter == :functions && type_symbol != :function
                continue
            elseif filter == :data && type_symbol != :data
                continue
            elseif filter == :weak && type_symbol != :weak
                continue
            end
        end

        push!(symbol_infos, SymbolInfo(
            name,
            demangled_name,
            address,
            type_symbol,
            0  # Size not available from nm
        ))
    end

    return symbol_infos
end

# ============================================================================
# DWARF INFORMATION
# ============================================================================

"""
    dwarf_info(binary_path::String)

Extract complete DWARF debug information from a binary.

Wraps the existing `Compiler.extract_dwarf_return_types()` and returns
a structured `DWARFInfo` object.

# Arguments
- `binary_path::String` - Path to binary file

# Returns
DWARFInfo

# Examples
```julia
# Extract DWARF info
dwarf = dwarf_info("lib.so")

# Access structs
matrix = dwarf.structs["Matrix3x3"]
println("Size: \$(matrix.size) bytes")

# Access functions
func = dwarf.functions["compute"]
println(func.return_type)

# Export as dataset
export_dataset(dwarf, "training_data/")
```
"""
function dwarf_info(binary_path::String)
    # Validate binary exists
    if !isfile(binary_path)
        error("Binary not found: $binary_path")
    end

    # Use existing Compiler function to extract DWARF info
    try
        # This returns (functions_dict, types_dict)
        (func_metadata, type_metadata) = Compiler.extract_dwarf_return_types(binary_path)

        # Convert to structured types
        functions_map = Dict{String,FunctionInfo}()
        structs_map = Dict{String,StructInfo}()
        enums_map = Dict{String,Vector{Tuple{String,Int}}}()

        # Process functions
        if haskey(func_metadata, "functions") && func_metadata["functions"] isa Vector
            for func_dict in func_metadata["functions"]
                name = get(func_dict, "name", "")
                if isempty(name)
                    continue
                end

                mangled = get(func_dict, "mangled", name)
                demangled = get(func_dict, "demangled", name)
                return_type_dict = get(func_dict, "return_type", Dict())
                return_type = get(return_type_dict, "c_type", "void")

                # Process parameters
                params = Tuple{String,String}[]
                if haskey(func_dict, "parameters") && func_dict["parameters"] isa Vector
                    for (idx, param) in enumerate(func_dict["parameters"])
                        param_name = get(param, "name", "arg$idx")
                        param_type = get(param, "c_type", "unknown")
                        push!(params, (param_name, param_type))
                    end
                end

                is_method = get(func_dict, "is_method", false)
                class = get(func_dict, "class", nothing)

                functions_map[name] = FunctionInfo(
                    name, mangled, demangled,
                    return_type, params,
                    is_method, class
                )
            end
        end

        # Process structs from type registry
        if haskey(type_metadata, "struct_definitions")
            for (struct_name, struct_dict) in type_metadata["struct_definitions"]
                members = MemberInfo[]

                if haskey(struct_dict, "members") && struct_dict["members"] isa Vector
                    for member_dict in struct_dict["members"]
                        member_name = get(member_dict, "name", "")
                        c_type = get(member_dict, "c_type", "unknown")
                        julia_type = get(member_dict, "julia_type", "Any")
                        offset = get(member_dict, "offset", 0)

                        # Parse offset if it's a string like "0x00"
                        if offset isa String
                            offset = tryparse(Int, offset, base=16)
                            if offset === nothing
                                offset = 0
                            end
                        end

                        size = get(member_dict, "size", 0)

                        push!(members, MemberInfo(
                            member_name, c_type, julia_type, offset, size
                        ))
                    end
                end

                # Calculate size from members if not provided
                struct_size = get(struct_dict, "size", 0)
                if struct_size == 0 && !isempty(members)
                    # Size = last member offset + last member size
                    last_member = members[end]
                    struct_size = last_member.offset + last_member.size
                end

                structs_map[struct_name] = StructInfo(
                    struct_name,
                    struct_size,
                    get(struct_dict, "alignment", 1),
                    members,
                    String[],  # base_classes not in current metadata
                    false,     # is_polymorphic not in current metadata
                    nothing    # vtable_offset not in current metadata
                )
            end
        end

        # Process enums if available
        # (Current Compiler.jl doesn't extract enums, so this is empty for now)

        return DWARFInfo(
            binary_path,
            functions_map,
            structs_map,
            enums_map
        )

    catch e
        @warn "DWARF extraction failed: $e"
        # Return empty DWARFInfo on failure
        return DWARFInfo(
            binary_path,
            Dict{String,FunctionInfo}(),
            Dict{String,StructInfo}(),
            Dict{String,Vector{Tuple{String,Int}}}()
        )
    end
end

# ============================================================================
# DISASSEMBLY
# ============================================================================

"""
    disassemble(binary_path::String, symbol=nothing; syntax=:att)

Disassemble binary or specific symbol using llvm-objdump.

# Arguments
- `binary_path::String` - Path to binary file
- `symbol` - Optional symbol name to disassemble (default: entire binary)
- `syntax::Symbol` - Assembly syntax (:att or :intel, default: :att)

# Returns
String - Disassembled code

# Examples
```julia
# Disassemble entire binary
asm = disassemble("lib.so")

# Disassemble specific function
asm = disassemble("lib.so", "compute_fft", syntax=:intel)

# Save to file
open("disasm.s", "w") do io
    write(io, asm)
end
```
"""
function disassemble(binary_path::String, symbol=nothing; syntax::Symbol=:att)
    # Validate binary exists
    if !isfile(binary_path)
        error("Binary not found: $binary_path")
    end

    # Get objdump tool
    objdump = LLVMEnvironment.get_tool("llvm-objdump")
    if isempty(objdump)
        objdump = "objdump"  # Fallback to system objdump
    end

    # Build arguments
    args = ["-d", "-C"]  # Disassemble + demangle

    if syntax == :intel
        push!(args, "-M", "intel")
    end

    push!(args, binary_path)

    # Execute objdump
    (output, exitcode) = LLVMEnvironment.with_llvm_env() do
        BuildBridge.execute(objdump, args)
    end

    if exitcode != 0
        @warn "objdump failed: $output"
        return ""
    end

    # Filter by symbol if provided
    if symbol !== nothing
        lines = split(output, '\n')
        filtered_lines = String[]
        in_symbol = false

        for line in lines
            if occursin("<$symbol>:", line)
                in_symbol = true
            elseif in_symbol && occursin(r"^[0-9a-f]+:", line)
                # Continue collecting instructions
            elseif in_symbol && (isempty(strip(line)) || occursin('<', line))
                # End of symbol
                break
            end

            if in_symbol
                push!(filtered_lines, line)
            end
        end

        return join(filtered_lines, '\n')
    end

    return output
end

# ============================================================================
# HEADER INFORMATION
# ============================================================================

"""
    headers(binary_path::String)

Extract binary headers and sections using readelf/llvm-readelf.

# Arguments
- `binary_path::String` - Path to binary file

# Returns
HeaderInfo

# Examples
```julia
# Get header info
header = headers("lib.so")
println("Architecture: \$(header.architecture)")
println("Sections: \$(length(header.sections))")
```
"""
function headers(binary_path::String)
    # Validate binary exists
    if !isfile(binary_path)
        error("Binary not found: $binary_path")
    end

    # Get readelf tool
    readelf = LLVMEnvironment.get_tool("llvm-readelf")
    if isempty(readelf)
        readelf = "readelf"  # Fallback to system readelf
    end

    # Get file header
    (header_output, exitcode1) = LLVMEnvironment.with_llvm_env() do
        BuildBridge.execute(readelf, ["-h", binary_path])
    end

    # Get section headers
    (section_output, exitcode2) = LLVMEnvironment.with_llvm_env() do
        BuildBridge.execute(readelf, ["-S", binary_path])
    end

    # Parse file type and architecture
    file_type = "unknown"
    architecture = "unknown"
    entry_point = "0x0"

    for line in split(header_output, '\n')
        if occursin("Type:", line)
            file_type = strip(split(line, ':')[2])
        elseif occursin("Machine:", line)
            architecture = strip(split(line, ':')[2])
        elseif occursin("Entry point", line)
            entry_point = strip(split(line, ':')[2])
        end
    end

    # Parse sections
    sections = Tuple{String,Int,Int}[]
    for line in split(section_output, '\n')
        # Section format: [ Nr] Name Type Addr Off Size
        if occursin(r"^\s*\[\s*\d+\]", line)
            parts = split(line)
            if length(parts) >= 6
                name = parts[2]
                # Try to parse offset and size
                offset = tryparse(Int, parts[5], base=16)
                size = tryparse(Int, parts[6], base=16)

                if offset !== nothing && size !== nothing
                    push!(sections, (name, offset, size))
                end
            end
        end
    end

    return HeaderInfo(
        binary_path,
        file_type,
        architecture,
        sections,
        entry_point
    )
end

# ============================================================================
# DWARF DUMP
# ============================================================================

"""
    dwarf_dump(binary_path::String; section=:info)

Dump DWARF debug information using llvm-dwarfdump.

# Arguments
- `binary_path::String` - Path to binary file
- `section::Symbol` - DWARF section (:info, :types, :line, :frame, default: :info)

# Returns
String - Raw DWARF dump output

# Examples
```julia
# Dump DWARF info section
info = dwarf_dump("lib.so", section=:info)

# Dump type information
types = dwarf_dump("lib.so", section=:types)

# Save to file
open("dwarf.txt", "w") do io
    write(io, info)
end
```
"""
function dwarf_dump(binary_path::String; section::Symbol=:info)
    # Validate binary exists
    if !isfile(binary_path)
        error("Binary not found: $binary_path")
    end

    # Get dwarfdump tool
    dwarfdump = LLVMEnvironment.get_tool("llvm-dwarfdump")
    if isempty(dwarfdump)
        @warn "llvm-dwarfdump not found"
        return ""
    end

    # Build section argument
    section_arg = if section == :info
        "--debug-info"
    elseif section == :types
        "--debug-types"
    elseif section == :line
        "--debug-line"
    elseif section == :frame
        "--debug-frame"
    else
        "--debug-info"
    end

    # Execute dwarfdump
    (output, exitcode) = LLVMEnvironment.with_llvm_env() do
        BuildBridge.execute(dwarfdump, [section_arg, binary_path])
    end

    if exitcode != 0
        @warn "llvm-dwarfdump failed: $output"
        return ""
    end

    return output
end
