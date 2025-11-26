# DWARFExtractor.jl - DWARF Debug Info Parsing
# Extracts type information, return types, struct layouts, enums, and arrays from DWARF metadata

module DWARFExtractor

import ...BuildBridge

export dwarf_type_to_julia, get_type_size, extract_dwarf_return_types

# =============================================================================
# TYPE MAPPING: C/C++ → Julia
# =============================================================================

"""
Convert C/C++ type from DWARF to Julia type.
Handles base types, pointers, references, const/volatile qualifiers.
"""
function dwarf_type_to_julia(c_type::AbstractString)::String
    type_map = Dict(
        # Void
        "void" => "Cvoid",

        # Boolean
        "bool" => "Bool",
        "_Bool" => "Bool",

        # Characters
        "char" => "Cchar",
        "signed char" => "Cschar",
        "unsigned char" => "Cuchar",
        "wchar_t" => "Cwchar_t",
        "char16_t" => "Cwchar_t",
        "char32_t" => "Cwchar_t",

        # Integers - 8 bit
        "int8_t" => "Int8",
        "uint8_t" => "UInt8",

        # Integers - 16 bit
        "short" => "Cshort",
        "short int" => "Cshort",
        "signed short" => "Cshort",
        "signed short int" => "Cshort",
        "unsigned short" => "Cushort",
        "unsigned short int" => "Cushort",
        "int16_t" => "Int16",
        "uint16_t" => "UInt16",

        # Integers - 32 bit
        "int" => "Cint",
        "signed int" => "Cint",
        "signed" => "Cint",
        "unsigned" => "Cuint",
        "unsigned int" => "Cuint",
        "int32_t" => "Int32",
        "uint32_t" => "UInt32",

        # Integers - 64 bit
        "long" => "Clong",
        "long int" => "Clong",
        "signed long" => "Clong",
        "signed long int" => "Clong",
        "unsigned long" => "Culong",
        "unsigned long int" => "Culong",
        "long long" => "Clonglong",
        "long long int" => "Clonglong",
        "signed long long" => "Clonglong",
        "signed long long int" => "Clonglong",
        "unsigned long long" => "Culonglong",
        "unsigned long long int" => "Culonglong",
        "int64_t" => "Int64",
        "uint64_t" => "UInt64",

        # Floating point
        "float" => "Cfloat",
        "double" => "Cdouble",
        "long double" => "Float64",  # Julia doesn't have long double, use Float64

        # Standard library types
        "size_t" => "Csize_t",
        "ssize_t" => "Cssize_t",
        "ptrdiff_t" => "Cptrdiff_t",
        "intptr_t" => "Cintptr_t",
        "uintptr_t" => "Cuintptr_t",

        # Time types
        "time_t" => "Ctime_t",

        # Special types
        "__int128" => "Int128",
        "__uint128_t" => "UInt128",
    )

    # Direct match
    if haskey(type_map, c_type)
        return type_map[c_type]
    end

    # Handle pointer types (T*)
    if endswith(c_type, "*")
        # Strip pointer and get base type
        base_type = strip(replace(c_type, r"\*+$" => ""))
        base_type = replace(base_type, "const" => "")
        base_type = replace(base_type, "volatile" => "")
        base_type = strip(base_type)

        # Special case: char* is Cstring
        if base_type == "char"
            return "Cstring"
        end

        # General pointer
        return "Ptr{Cvoid}"  # Generic pointer, could be refined based on base_type
    end

    # Handle reference types (T&)
    if endswith(c_type, "&")
        return "Ref{Cvoid}"  # Could be refined
    end

    # Handle const/volatile qualifiers
    if startswith(c_type, "const ") || startswith(c_type, "volatile ")
        clean_type = replace(c_type, "const " => "")
        clean_type = replace(clean_type, "volatile " => "")
        return dwarf_type_to_julia(strip(clean_type))
    end

    # Unknown type - return Any for safety
    return "Any"
end

"""
Get type size in bytes from C/C++ type name.
"""
function get_type_size(c_type::AbstractString)::Int
    size_map = Dict(
        "void" => 0,
        "bool" => 1, "_Bool" => 1,
        "char" => 1, "signed char" => 1, "unsigned char" => 1,
        "int8_t" => 1, "uint8_t" => 1,
        "short" => 2, "unsigned short" => 2,
        "int16_t" => 2, "uint16_t" => 2,
        "int" => 4, "unsigned int" => 4,
        "int32_t" => 4, "uint32_t" => 4,
        "long" => 8, "unsigned long" => 8,  # x86_64
        "long long" => 8, "unsigned long long" => 8,
        "int64_t" => 8, "uint64_t" => 8,
        "float" => 4,
        "double" => 8,
        "long double" => 16,  # x86_64 extended precision
        "size_t" => 8, "ssize_t" => 8,  # x86_64
        "intptr_t" => 8, "uintptr_t" => 8,
        "__int128" => 16, "__uint128_t" => 16,
    )

    # Check for pointers/references (always 8 bytes on x86_64)
    if endswith(c_type, "*") || endswith(c_type, "&")
        return 8
    end

    return get(size_map, strip(c_type), 0)
end

# =============================================================================
# DWARF EXTRACTION: Return Types, Structs, Enums, Arrays
# =============================================================================

"""
Extract return types and struct definitions from DWARF debug info.
Returns: (return_types_dict, enums_dict)
  - return_types: Dict{mangled_name => {c_type, julia_type, size}}
  - enums: Dict{enum_name => {underlying_type, values: {name => value}}}
"""
function extract_dwarf_return_types(binary_path::String)::Tuple{Dict{String,Dict{String,Any}}, Dict{String,Dict{String,Any}}}
    println("Parsing DWARF debug info...")

    # Run readelf to get DWARF debug info
    (output, exitcode) = BuildBridge.execute("readelf", ["--debug-dump=info", binary_path])

    if exitcode != 0
        @warn "Failed to read DWARF info: $output"
        return (Dict{String,Dict{String,Any}}(), Dict{String,Dict{String,Any}}())
    end

    return_types = Dict{String,Dict{String,Any}}()
    enums_dict = Dict{String,Dict{String,Any}}()

    # Parse DWARF output to extract type information
    # Strategy: Find DW_TAG_subprogram entries, extract linkage_name and return type

    current_function = nothing
    current_linkage_name = nothing
    type_refs = Dict{String,Any}()  # offset => type_name (String) or type_info (Dict)

    # First pass: Build type reference table
    # We need to handle: base types, pointer types, const types, reference types
    offset_to_kind = Dict{String,Symbol}()  # Track what kind each offset is
    current_type_offset = nothing  # Track current type/struct being processed
    current_struct_context = nothing  # Track parent struct for members

    for line in split(output, '\n')
        line = strip(line)

        # Track ANY tag to avoid attribute pollution
        # Tags have format: <level><offset>: Abbrev Number: N (DW_TAG_*)
        if contains(line, "DW_TAG_")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                tag_offset = "0x" * offset_match.captures[1]
                type_refs["last_tag_offset"] = tag_offset  # Always update for ANY tag
            end
        end

        # Extract base type definitions (DW_TAG_base_type)
        # Example: <1><27>: Abbrev Number: 2 (DW_TAG_base_type)
        if contains(line, "DW_TAG_base_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = "unknown"
                offset_to_kind[current_type_offset] = :base
            end
        end

        # Extract pointer type definitions (DW_TAG_pointer_type)
        # Example: <1><9f6>: Abbrev Number: 40 (DW_TAG_pointer_type)
        #          <9f7>   DW_AT_type        : <0x41>
        if contains(line, "DW_TAG_pointer_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = Dict{String,Any}("kind" => "pointer", "target" => nothing)
                offset_to_kind[current_type_offset] = :pointer
            end
        end

        # Extract const type definitions (DW_TAG_const_type)
        # Example: <1><41>: Abbrev Number: 5 (DW_TAG_const_type)
        #          <42>   DW_AT_type        : <0x46>
        if contains(line, "DW_TAG_const_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = Dict{String,Any}("kind" => "const", "target" => nothing)
                offset_to_kind[current_type_offset] = :const
            end
        end

        # Extract reference type definitions (DW_TAG_reference_type)
        if contains(line, "DW_TAG_reference_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = Dict{String,Any}("kind" => "reference", "target" => nothing)
                offset_to_kind[current_type_offset] = :reference
            end
        end

        # Extract struct type definitions (DW_TAG_structure_type)
        # Example: <1><1f5f3e>: DW_TAG_structure_type
        #          DW_AT_name: "Vector3d"
        #          DW_AT_byte_size: 0x18
        if contains(line, "DW_TAG_structure_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                current_struct_context = current_type_offset  # Set context for members
                # Store struct info as a dict with members array
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "struct",
                    "name" => "unknown_struct",
                    "members" => []
                )
                offset_to_kind[current_type_offset] = :struct
            end
        end

        # Extract class type definitions (DW_TAG_class_type) - treat as struct
        if contains(line, "DW_TAG_class_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                current_struct_context = current_type_offset
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "struct",
                    "name" => "unknown_class",
                    "members" => []
                )
                offset_to_kind[current_type_offset] = :struct
            end
        end

        # Extract enum type definitions (DW_TAG_enumeration_type)
        # Example: <1><2e>: Abbrev Number: 3 (DW_TAG_enumeration_type)
        #          <2f>   DW_AT_name        : Color
        #          <35>   DW_AT_encoding    : 7  (unsigned)
        #          <36>   DW_AT_byte_size   : 4
        if contains(line, "DW_TAG_enumeration_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "enum",
                    "name" => "unknown_enum",
                    "underlying_type" => "unsigned int",  # Default
                    "values" => Dict{String,Int}()
                )
                offset_to_kind[current_type_offset] = :enum
            end
        end

        # Extract enumerator values (DW_TAG_enumerator)
        # Example: <2><3d>: Abbrev Number: 4 (DW_TAG_enumerator)
        #          <3e>   DW_AT_name        : Red
        #          <42>   DW_AT_const_value : 0
        if contains(line, "DW_TAG_enumerator")
            # Enumerators belong to the last enum tag
            # We'll capture name and value in subsequent lines
        end

        # Extract array type definitions (DW_TAG_array_type)
        # Example: <1><a5>: Abbrev Number: 11 (DW_TAG_array_type)
        #          <a6>   DW_AT_type        : <0x27>  (element type)
        if contains(line, "DW_TAG_array_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "array",
                    "element_type" => nothing,
                    "dimensions" => []  # Will be filled by DW_TAG_subrange_type
                )
                offset_to_kind[current_type_offset] = :array
            end
        end

        # Extract array dimensions (DW_TAG_subrange_type)
        # Example: <2><b0>: Abbrev Number: 12 (DW_TAG_subrange_type)
        #          <b1>   DW_AT_type        : <0xad>
        #          <b5>   DW_AT_upper_bound : 3  (array size - 1)
        if contains(line, "DW_TAG_subrange_type")
            # Subranges belong to the last array tag
            # Will capture upper_bound in next line
        end

        # Extract struct/class members (DW_TAG_member)
        # Example: <2><1f5f54>: Abbrev Number: 311 (DW_TAG_member)
        #          <1f5f55>   DW_AT_name        : x
        #          <1f5f57>   DW_AT_type        : <0x46>
        #          <1f5f5b>   DW_AT_data_member_location: 0
        if contains(line, "DW_TAG_member")
            # Members belong to current_struct_context
            # Will capture name, type, offset in subsequent lines
        end

        # Extract typedef definitions (DW_TAG_typedef)
        if contains(line, "DW_TAG_typedef")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = Dict{String,Any}("kind" => "typedef", "target" => nothing, "name" => "unknown_typedef")
                offset_to_kind[current_type_offset] = :typedef
            end
        end

        # Extract DW_AT_name for current type/struct
        if !isnothing(current_type_offset) && contains(line, "DW_AT_name")
            # Extract name value
            # Format: DW_AT_name        : double
            # or:     DW_AT_name        : (indirect string, offset: 0x...): Vector3d
            name_match = match(r"DW_AT_name\s+:\s+(?:\(indirect string[^)]+\):\s*)?(.+)", line)
            if !isnothing(name_match)
                type_name = strip(name_match.captures[1])

                # Update type_refs based on kind
                if haskey(offset_to_kind, current_type_offset)
                    kind = offset_to_kind[current_type_offset]
                    if kind == :base
                        type_refs[current_type_offset] = type_name
                    elseif kind == :struct
                        if isa(type_refs[current_type_offset], Dict)
                            type_refs[current_type_offset]["name"] = type_name
                        end
                    elseif kind == :enum
                        if isa(type_refs[current_type_offset], Dict)
                            type_refs[current_type_offset]["name"] = type_name
                        end
                    elseif kind == :typedef
                        if isa(type_refs[current_type_offset], Dict)
                            type_refs[current_type_offset]["name"] = type_name
                        end
                    end
                end
            end
        end

        # Extract DW_AT_type references (what a pointer/const/reference points to)
        if contains(line, "DW_AT_type")
            type_offset_match = match(r"DW_AT_type\s+:\s+<(0x[0-9a-fA-F]+)>", line)
            if !isnothing(type_offset_match)
                target_offset = type_offset_match.captures[1]

                # Determine which context this belongs to
                last_tag = get(type_refs, "last_tag_offset", nothing)
                if !isnothing(last_tag) && haskey(type_refs, last_tag) && isa(type_refs[last_tag], Dict)
                    # Update target for pointer/const/reference/array
                    if haskey(type_refs[last_tag], "target")
                        type_refs[last_tag]["target"] = target_offset
                    elseif haskey(type_refs[last_tag], "element_type")
                        # Array element type
                        type_refs[last_tag]["element_type"] = target_offset
                    end
                end
            end
        end

        # Extract array dimensions from DW_AT_upper_bound
        if contains(line, "DW_AT_upper_bound")
            bound_match = match(r"DW_AT_upper_bound\s+:\s+(\d+)", line)
            if !isnothing(bound_match)
                upper_bound = parse(Int, bound_match.captures[1])
                array_size = upper_bound + 1  # DWARF uses 0-indexed

                # Add dimension to last array tag
                last_tag = get(type_refs, "last_tag_offset", nothing)
                if !isnothing(last_tag) && haskey(type_refs, last_tag) && isa(type_refs[last_tag], Dict)
                    if haskey(type_refs[last_tag], "dimensions")
                        push!(type_refs[last_tag]["dimensions"], array_size)
                    end
                end
            end
        end

        # Extract DW_AT_count for array dimensions (alternative to upper_bound)
        if contains(line, "DW_AT_count")
            count_match = match(r"DW_AT_count\s+:\s+(\d+)", line)
            if !isnothing(count_match)
                count = parse(Int, count_match.captures[1])

                last_tag = get(type_refs, "last_tag_offset", nothing)
                if !isnothing(last_tag) && haskey(type_refs, last_tag) && isa(type_refs[last_tag], Dict)
                    if haskey(type_refs[last_tag], "dimensions")
                        push!(type_refs[last_tag]["dimensions"], count)
                    end
                end
            end
        end

        # Extract enumerator values
        if contains(line, "DW_AT_const_value") && haskey(offset_to_kind, get(type_refs, "last_tag_offset", ""))
            if offset_to_kind[type_refs["last_tag_offset"]] == :enum
                # This const_value belongs to an enumerator
                # Need to find the parent enum (last enum tag before this enumerator)
                value_match = match(r"DW_AT_const_value\s+:\s+(-?\d+)", line)
                if !isnothing(value_match)
                    enum_value = parse(Int, value_match.captures[1])
                    # Store temporarily, will associate with name when we see DW_AT_name
                    type_refs["last_enum_value"] = enum_value
                end
            end
        end

        # Capture enumerator name and associate with value
        if contains(line, "DW_TAG_enumerator")
            # Next DW_AT_name will be the enumerator name
            type_refs["awaiting_enumerator_name"] = true
        end

        if get(type_refs, "awaiting_enumerator_name", false) && contains(line, "DW_AT_name")
            name_match = match(r"DW_AT_name\s+:\s+(?:\(indirect string[^)]+\):\s*)?(.+)", line)
            if !isnothing(name_match)
                enum_name_str = strip(name_match.captures[1])
                enum_value = get(type_refs, "last_enum_value", 0)

                # Find parent enum (last enum type before this enumerator)
                for (offset, type_info) in type_refs
                    if isa(type_info, Dict) && get(type_info, "kind", "") == "enum"
                        type_info["values"][enum_name_str] = enum_value
                        break  # Only add to first (most recent) enum
                    end
                end

                delete!(type_refs, "awaiting_enumerator_name")
            end
        end

        # Extract underlying type for enums (DW_AT_type reference)
        # Note: Already handled by DW_AT_type processing above

        # Extract function definitions (DW_TAG_subprogram)
        # Example: <1><d99f>: DW_TAG_subprogram
        #          <d9a0>   DW_AT_name        : create_matrix
        #          <d9ae>   DW_AT_linkage_name: _Z13create_matrixv
        #          <d9bb>   DW_AT_type        : <0x46>  (return type offset)
        #  <2><...>: DW_TAG_formal_parameter (nested parameters)
        if contains(line, "DW_TAG_subprogram")
            current_function = nothing
            current_linkage_name = nothing
        end

        # Extract function linkage name (mangled name)
        if contains(line, "DW_AT_linkage_name") || contains(line, "DW_AT_MIPS_linkage_name")
            linkage_match = match(r"DW_AT_(?:MIPS_)?linkage_name\s*:\s*(?:\(indirect string[^)]+\):\s*)?(.+)", line)
            if !isnothing(linkage_match)
                current_linkage_name = strip(linkage_match.captures[1])
            end
        end

        # Extract function return type
        if !isnothing(current_linkage_name) && contains(line, "DW_AT_type")
            type_offset_match = match(r"DW_AT_type\s+:\s+<(0x[0-9a-fA-F]+)>", line)
            if !isnothing(type_offset_match)
                return_type_offset = type_offset_match.captures[1]

                # Store function with return type offset and empty parameters list
                return_types[current_linkage_name] = Dict{String,Any}(
                    "type_offset" => return_type_offset,
                    "c_type" => "unknown",
                    "julia_type" => "Any",
                    "size" => 0,
                    "parameters" => []  # Will be filled by DW_TAG_formal_parameter
                )

                # Don't reset current_linkage_name yet - we need it for parameters
            end
        end

        # Extract function parameters (DW_TAG_formal_parameter)
        # Example: <2><117>: DW_TAG_formal_parameter
        #          <11b>   DW_AT_name        : g
        #          <11e>   DW_AT_type        : <0x13bb>
        if contains(line, "DW_TAG_formal_parameter")
            # Mark that we're expecting parameter name and type
            type_refs["awaiting_parameter"] = true
            type_refs["current_param"] = Dict{String,Any}("name" => "", "type_offset" => "")
        end

        # Capture parameter name
        if get(type_refs, "awaiting_parameter", false) && contains(line, "DW_AT_name")
            name_match = match(r"DW_AT_name\s+:\s+(?:\(indexed string[^)]+\):\s*)?(.+)", line)
            if !isnothing(name_match)
                param_name = strip(name_match.captures[1])
                if haskey(type_refs, "current_param")
                    type_refs["current_param"]["name"] = param_name
                end
            end
        end

        # Capture parameter type and add to current function
        if get(type_refs, "awaiting_parameter", false) && contains(line, "DW_AT_type")
            type_offset_match = match(r"DW_AT_type\s+:\s+<(0x[0-9a-fA-F]+)>", line)
            if !isnothing(type_offset_match) && haskey(type_refs, "current_param")
                param_type_offset = type_offset_match.captures[1]
                type_refs["current_param"]["type_offset"] = param_type_offset

                # Add parameter to current function
                if !isnothing(current_linkage_name) && haskey(return_types, current_linkage_name)
                    push!(return_types[current_linkage_name]["parameters"],
                          Dict{String,Any}(
                              "name" => type_refs["current_param"]["name"],
                              "type_offset" => param_type_offset,
                              "c_type" => "unknown",
                              "julia_type" => "Any"
                          ))
                end

                # Reset parameter tracking
                delete!(type_refs, "awaiting_parameter")
                delete!(type_refs, "current_param")
            end
        end
    end

    # Second pass: Resolve type offsets to actual types
    function resolve_type(offset::AbstractString, depth::Int=0)::String
        offset_str = String(offset)  # Convert SubString to String

        if depth > 100
            return "Any"  # Prevent infinite recursion
        end

        if !haskey(type_refs, offset_str)
            return "Any"
        end

        type_info = type_refs[offset_str]

        # If it's a string, it's a resolved base type
        if isa(type_info, String)
            return type_info
        end

        # If it's a dict, resolve based on kind
        if isa(type_info, Dict)
            kind = get(type_info, "kind", "unknown")

            if kind == "pointer"
                target = get(type_info, "target", nothing)
                if isnothing(target)
                    return "Ptr{Cvoid}"
                end
                target_type = resolve_type(target, depth + 1)
                # Special case: char* → Cstring
                if target_type == "char"
                    return "Cstring"
                end
                return "Ptr{Cvoid}"  # Generic pointer for now
            elseif kind == "const"
                target = get(type_info, "target", nothing)
                if isnothing(target)
                    return "Any"
                end
                return resolve_type(target, depth + 1)  # Strip const qualifier
            elseif kind == "reference"
                target = get(type_info, "target", nothing)
                if isnothing(target)
                    return "Ref{Cvoid}"
                end
                return "Ref{Cvoid}"  # Generic reference for now
            elseif kind == "struct"
                return get(type_info, "name", "Any")
            elseif kind == "enum"
                return get(type_info, "name", "Any")
            elseif kind == "typedef"
                target = get(type_info, "target", nothing)
                if isnothing(target)
                    return get(type_info, "name", "Any")
                end
                return resolve_type(target, depth + 1)
            elseif kind == "array"
                # Multi-dimensional arrays are flattened in Julia FFI
                element_type_offset = get(type_info, "element_type", nothing)
                if isnothing(element_type_offset)
                    return "Any"
                end

                element_type = resolve_type(element_type_offset, depth + 1)
                dimensions = get(type_info, "dimensions", [])

                if isempty(dimensions)
                    return "Ptr{$element_type}"  # Unknown size array → pointer
                end

                # Calculate total size (product of all dimensions)
                total_size = prod(dimensions)

                # Convert to Julia type
                julia_element = dwarf_type_to_julia(element_type)
                return "NTuple{$total_size, $julia_element}"
            end
        end

        return "Any"
    end

    # Resolve all return types and parameters
    for (func_name, func_info) in return_types
        # Resolve return type
        type_offset = func_info["type_offset"]
        resolved_type = resolve_type(type_offset)

        # Convert C type to Julia type
        julia_type = dwarf_type_to_julia(resolved_type)

        func_info["c_type"] = resolved_type
        func_info["julia_type"] = julia_type
        func_info["size"] = get_type_size(resolved_type)

        # Resolve parameter types
        if haskey(func_info, "parameters")
            for param in func_info["parameters"]
                param_type_offset = param["type_offset"]
                resolved_param_type = resolve_type(param_type_offset)
                param_julia_type = dwarf_type_to_julia(resolved_param_type)

                param["c_type"] = resolved_param_type
                param["julia_type"] = param_julia_type
            end
        end
    end

    # Extract enums into separate dict
    for (offset, type_info) in type_refs
        if isa(type_info, Dict) && get(type_info, "kind", "") == "enum"
            enum_name = type_info["name"]
            if enum_name != "unknown_enum"
                # Resolve underlying type
                underlying_offset = get(type_info, "target", nothing)
                underlying_type = if !isnothing(underlying_offset)
                    resolve_type(underlying_offset)
                else
                    "unsigned int"  # Default
                end

                enums_dict[enum_name] = Dict{String,Any}(
                    "underlying_type" => underlying_type,
                    "values" => type_info["values"]
                )
            end
        end
    end

    println("  Found $(length(return_types)) return types")
    println("  Found $(length(enums_dict)) enums")

    return (return_types, enums_dict)
end

end # module DWARFExtractor
