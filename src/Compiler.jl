#!/usr/bin/env julia
# Compiler.jl - C++ to LLVM IR compilation and linking
# Full control over LLVM/Clang for Julia interoperability
# Extracts metadata during compilation for automatic wrapper generation

module Compiler

using Dates
using JSON

# Import from parent RepliBuild module
import ..ConfigurationManager: RepliBuildConfig, get_source_files, get_include_dirs,
                                get_compile_flags, get_build_path, get_output_path,
                                get_library_name, get_module_name, is_parallel_enabled,
                                is_cache_enabled
import ..BuildBridge
import ..LLVMEnvironment

export compile_to_ir, link_optimize_ir, create_library, create_executable, compile_project,
       extract_compilation_metadata, save_compilation_metadata

# =============================================================================
# INCREMENTAL BUILD SUPPORT
# =============================================================================

"""
Check if source file needs recompilation (based on mtime).
"""
function needs_recompile(source_file::String, ir_file::String, cache_enabled::Bool)::Bool
    if !cache_enabled
        return true  # Always recompile if cache disabled
    end

    if !isfile(ir_file)
        return true  # IR doesn't exist
    end

    # Compare modification times
    source_mtime = mtime(source_file)
    ir_mtime = mtime(ir_file)

    return source_mtime > ir_mtime
end

# =============================================================================
# COMPILATION: C++ → LLVM IR
# =============================================================================

"""
Compile a single C++ file to LLVM IR.
Returns (ir_file_path, success, exitcode).
"""
function compile_single_to_ir(config::RepliBuildConfig, cpp_file::String)
    # Generate IR file path
    build_dir = get_build_path(config)
    mkpath(build_dir)

    base_name = splitext(basename(cpp_file))[1]
    ir_file = joinpath(build_dir, base_name * ".ll")

    # Check if recompilation needed
    if !needs_recompile(cpp_file, ir_file, is_cache_enabled(config))
        return (ir_file, true, 0)  # Cache hit
    end

    # Build compiler command
    # Ensure -g is always present for DWARF metadata extraction
    base_flags = get_compile_flags(config)
    if !("-g" in base_flags)
        base_flags = vcat(base_flags, ["-g"])
    end

    cmd_args = vcat(
        ["-S", "-emit-llvm"],  # Emit LLVM IR
        base_flags,
        ["-I$dir" for dir in get_include_dirs(config)],
        ["-D$k=$v" for (k, v) in config.compile.defines],
        ["-o", ir_file, cpp_file]
    )

    # Compile using BuildBridge
    (output, exitcode) = BuildBridge.execute("clang++", cmd_args)

    if !isempty(output) && exitcode != 0
        println("  $(basename(cpp_file)): $output")
    end

    success = isfile(ir_file)
    return (ir_file, success, exitcode)
end

"""
Compile multiple C++ files to LLVM IR (with parallel support).
Returns vector of IR file paths.
"""
function compile_to_ir(config::RepliBuildConfig, cpp_files::Vector{String})
    println("Compiling to LLVM IR...")

    if isempty(cpp_files)
        @warn "No source files to compile"
        return String[]
    end

    # Check which files need compilation
    build_dir = get_build_path(config)
    mkpath(build_dir)

    files_to_compile = String[]
    ir_files = String[]
    cached_files = 0

    for cpp_file in cpp_files
        base_name = splitext(basename(cpp_file))[1]
        ir_file = joinpath(build_dir, base_name * ".ll")

        if needs_recompile(cpp_file, ir_file, is_cache_enabled(config))
            push!(files_to_compile, cpp_file)
        else
            cached_files += 1
        end
        push!(ir_files, ir_file)
    end

    if is_cache_enabled(config) && cached_files > 0
        cache_pct = round(100 * cached_files / length(cpp_files), digits=1)
        println("  ⚡ $cache_pct% cache hit ($cached_files/$(length(cpp_files)) files)")
    end

    if isempty(files_to_compile)
        println("All files cached, nothing to compile")
        return ir_files
    end

    println("Compiling $(length(files_to_compile)) files...")

    # Compile in parallel if enabled
    if is_parallel_enabled(config) && length(files_to_compile) > 1
        println("Using $(Threads.nthreads()) threads")

        results = Vector{Tuple{String,Bool,Int}}(undef, length(files_to_compile))
        Threads.@threads for i in 1:length(files_to_compile)
            results[i] = compile_single_to_ir(config, files_to_compile[i])
        end

        # Check for failures
        failures = [(file, res) for (file, res) in zip(files_to_compile, results) if !res[2]]
        if !isempty(failures)
            println("  $(length(failures)) compilation failures")
            for (file, (_, _, exitcode)) in failures
                println("     $(basename(file)) (exit: $exitcode)")
            end
            error("Compilation failed for $(length(failures)) files")
        end
    else
        # Sequential compilation
        for cpp_file in files_to_compile
            (ir_file, success, exitcode) = compile_single_to_ir(config, cpp_file)
            if !success
                error("Compilation failed for $cpp_file (exit: $exitcode)")
            end
        end
    end 
    return ir_files
end

# =============================================================================
# LINKING: LLVM IR → Optimized IR
# =============================================================================

"""
Link multiple LLVM IR files and optimize.
Returns path to linked IR file.
"""
function link_optimize_ir(config::RepliBuildConfig, ir_files::Vector{String}, output_name::String)
    println("Linking and optimizing IR...")

    if isempty(ir_files)
        error("No IR files to link")
    end

    # Output path
    build_dir = get_build_path(config)
    linked_ir = joinpath(build_dir, output_name * "_linked.ll")

    # Link using llvm-link
    (output, exitcode) = BuildBridge.execute("llvm-link", vcat(["-S"], ir_files, ["-o", linked_ir]))

    if exitcode != 0
        error("Linking failed: $output")
    end

    if !isfile(linked_ir)
        error("Linked IR file not created: $linked_ir")
    end

    println("Linked $(length(ir_files)) IR files → $(basename(linked_ir))")

    # Optimize if requested
    opt_level = config.link.optimization_level
    if opt_level != "0"
        println("Optimizing (O$opt_level)...")

        optimized_ir = joinpath(build_dir, output_name * "_opt.ll")
        opt_args = ["-S", "-O$opt_level"]

        # Add LTO if enabled
        if config.link.enable_lto
            push!(opt_args, "-lto")
        end

        push!(opt_args, linked_ir, "-o", optimized_ir)

        (output, exitcode) = BuildBridge.execute("opt", opt_args)

        if exitcode != 0
            @warn "Optimization failed, using unoptimized IR: $output"
        elseif isfile(optimized_ir)
            linked_ir = optimized_ir
            println("Optimized")
        end
    end

    return linked_ir
end

# =============================================================================
# BINARY CREATION: IR → Shared Library / Executable
# =============================================================================

"""
Create shared library from LLVM IR.
Returns path to library file.
"""
function create_library(config::RepliBuildConfig, ir_file::String, lib_name::String="")
    println("Creating shared library...")

    if !isfile(ir_file)
        error("IR file not found: $ir_file")
    end

    # Determine library name
    if isempty(lib_name)
        lib_name = get_library_name(config)
    end

    # Output path
    output_dir = get_output_path(config)
    mkpath(output_dir)
    lib_path = joinpath(output_dir, lib_name)

    # Compile IR to shared library
    cmd_args = [
        "-shared",  # Create shared library
        "-fPIC",    # Position independent code
        ir_file,
        "-o", lib_path
    ]

    # Add link libraries
    for lib in config.link.link_libraries
        push!(cmd_args, "-l$lib")
    end

    (output, exitcode) = BuildBridge.execute("clang++", cmd_args)

    if exitcode != 0
        error("Library creation failed: $output")
    end

    if !isfile(lib_path)
        error("Library file not created: $lib_path")
    end

    # Get file size
    size_mb = round(filesize(lib_path) / 1024 / 1024, digits=2)
    println("Created: $lib_name ($size_mb MB)")

    return lib_path
end

"""
Create executable from LLVM IR.
Returns path to executable file.
"""
function create_executable(config::RepliBuildConfig, ir_file::String, exe_name::String,
                          link_libraries::Vector{String}=String[],
                          lib_dirs::Vector{String}=String[])
    println("Creating executable...")

    if !isfile(ir_file)
        error("IR file not found: $ir_file")
    end

    # Output path
    output_dir = get_output_path(config)
    mkpath(output_dir)
    exe_path = joinpath(output_dir, exe_name)

    # Compile IR to executable
    cmd_args = [ir_file, "-o", exe_path]

    # Add library directories
    for dir in lib_dirs
        push!(cmd_args, "-L$dir")
    end

    # Add link libraries (config + additional)
    all_libs = vcat(config.link.link_libraries, link_libraries)
    for lib in all_libs
        push!(cmd_args, "-l$lib")
    end

    (output, exitcode) = BuildBridge.execute("clang++", cmd_args)

    if exitcode != 0
        error("Executable creation failed: $output")
    end

    if !isfile(exe_path)
        error("Executable file not created: $exe_path")
    end

    # Make executable
    chmod(exe_path, 0o755)

    size_kb = round(filesize(exe_path) / 1024, digits=1)
    println("Created: $exe_name ($size_kb KB)")

    return exe_path
end

# =============================================================================
# HIGH-LEVEL COMPILATION
# =============================================================================

"""
Complete compilation workflow: discover sources, compile, link, create binary.
This is the main entry point for building a project.
"""
function compile_project(config::RepliBuildConfig)
    println("="^70)
    println("RepliBuild Compiler")
    println("="^70)
    println("Project: $(config.project.name)")
    println("Root:    $(config.project.root)")
    println("="^70)
    println()

    start_time = time()

    # Get source files (from config or discovery)
    cpp_files = get_source_files(config)

    if isempty(cpp_files)
        @warn "No source files found in config"
        println("Run discover() first to find C++ sources")
        return nothing
    end

    println("Source files: $(length(cpp_files))")
    println("Compiler flags: $(join(get_compile_flags(config), " "))")
    println("Include dirs: $(length(get_include_dirs(config)))")
    println()

    # Step 1: Compile C++ → IR
    ir_files = compile_to_ir(config, cpp_files)

    # Step 2: Link & optimize IR
    output_name = config.project.name
    linked_ir = link_optimize_ir(config, ir_files, output_name)

    # Step 3: Create binary (library or executable)
    binary_path = if config.binary.type == :executable
        create_executable(config, linked_ir, config.project.name)
    else
        create_library(config, linked_ir)
    end

    elapsed = round(time() - start_time, digits=2)

    # Step 4: Extract and save compilation metadata
    metadata_path = save_compilation_metadata(config, cpp_files, binary_path)

    println()
    println("="^70)
    println("Build successful ($elapsed seconds)")
    println("Binary: $binary_path")
    println("Metadata: $metadata_path")
    println("="^70)

    return binary_path
end

# =============================================================================
# METADATA EXTRACTION - The Key to Automatic Wrapping
# =============================================================================

"""
Extract symbol information from compiled binary using nm.
Returns vector of symbol dictionaries with mangled/demangled names.
"""
function extract_symbols_from_binary(binary_path::String)
    # Run nm WITHOUT demangling to get mangled names
    (mangled_output, exitcode1) = BuildBridge.execute("nm", ["-g", "--defined-only", binary_path])
    if exitcode1 != 0
        @warn "nm command failed: $mangled_output"
        return Dict{String,Any}[]
    end

    # Run nm WITH demangling to get human-readable names
    (demangled_output, exitcode2) = BuildBridge.execute("nm", ["-gC", "--defined-only", binary_path])
    if exitcode2 != 0
        @warn "nm demangled command failed: $demangled_output"
        return Dict{String,Any}[]
    end

    # Build address → mangled mapping
    address_to_mangled = Dict{String,String}()
    for line in split(mangled_output, '\n')
        line = strip(line)
        if isempty(line)
            continue
        end
        parts = split(line)
        if length(parts) >= 3 && parts[2] == "T"
            address = parts[1]
            mangled = parts[3]
            address_to_mangled[address] = mangled
        end
    end

    # Build symbols array with both mangled and demangled
    symbols = Dict{String,Any}[]
    for line in split(demangled_output, '\n')
        line = strip(line)
        if isempty(line)
            continue
        end

        parts = split(line)
        if length(parts) >= 3
            address = parts[1]
            symbol_type = parts[2]
            demangled_name = join(parts[3:end], " ")

            # Only export T (text/code) symbols
            if symbol_type == "T"
                mangled_name = get(address_to_mangled, address, demangled_name)

                push!(symbols, Dict(
                    "mangled" => mangled_name,
                    "demangled" => demangled_name,
                    "type" => symbol_type,
                    "address" => address
                ))
            end
        end
    end

    return symbols
end

"""
Extract mangled symbol name from nm output.
"""
function extract_mangled_name(nm_output::String, demangled::String)::String
    # Try to find the mangled version
    for line in split(nm_output, '\n')
        if contains(line, " T ")
            parts = split(strip(line))
            if length(parts) >= 3
                # parts[3] is the mangled name
                return parts[3]
            end
        end
    end
    return demangled  # Fallback
end

# =============================================================================
# DWARF DEBUG INFO PARSING - Type Extraction
# =============================================================================

"""
Comprehensive C/C++ type to Julia type mapping.
Handles all standard C/C++ types including sized integers, pointers, and qualifiers.
"""
function dwarf_type_to_julia(c_type::AbstractString)::String
    # Clean up type (remove extra spaces, trailing qualifiers)
    c_type = strip(c_type)

    # Map based on comprehensive C/C++ type system
    type_map = Dict(
        # Void
        "void" => "Cvoid",

        # Boolean
        "bool" => "Bool",
        "_Bool" => "Bool",

        # Character types
        "char" => "Cchar",
        "signed char" => "Cchar",
        "unsigned char" => "Cuchar",
        "wchar_t" => "Cwchar_t",
        "char16_t" => "UInt16",
        "char32_t" => "UInt32",

        # Standard integers (platform-dependent sizes)
        "short" => "Cshort",
        "short int" => "Cshort",
        "signed short" => "Cshort",
        "signed short int" => "Cshort",
        "unsigned short" => "Cushort",
        "unsigned short int" => "Cushort",

        "int" => "Cint",
        "signed int" => "Cint",
        "signed" => "Cint",
        "unsigned int" => "Cuint",
        "unsigned" => "Cuint",

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

        # Fixed-width integers (stdint.h)
        "int8_t" => "Int8",
        "int16_t" => "Int16",
        "int32_t" => "Int32",
        "int64_t" => "Int64",
        "uint8_t" => "UInt8",
        "uint16_t" => "UInt16",
        "uint32_t" => "UInt32",
        "uint64_t" => "UInt64",

        # Floating point
        "float" => "Cfloat",
        "double" => "Cdouble",
        "long double" => "Float64",  # Julia doesn't have 80-bit float, use Float64

        # Size types
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

"""
Extract return types and struct definitions from DWARF debug info.
Returns: (return_types_dict, struct_defs_dict)
  - return_types: Dict{mangled_name => {c_type, julia_type, size}}
  - struct_defs: Dict{struct_name => {members: [{name, type, offset}]}}
"""
function extract_dwarf_return_types(binary_path::String)::Tuple{Dict{String,Dict{String,Any}}, Dict{String,Dict{String,Any}}}
    println("Parsing DWARF debug info...")

    # Run readelf to get DWARF debug info
    (output, exitcode) = BuildBridge.execute("readelf", ["--debug-dump=info", binary_path])

    if exitcode != 0
        @warn "Failed to read DWARF info: $output"
        return Dict{String,Dict{String,Any}}()
    end

    return_types = Dict{String,Dict{String,Any}}()

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
    current_subroutine_offset = nothing  # Track current subroutine type for parameters

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

        # Extract union type definitions (DW_TAG_union_type)
        if contains(line, "DW_TAG_union_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                current_struct_context = current_type_offset  # Set context for members
                # Store union info as a dict with members array
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "union",
                    "name" => "unknown_union",
                    "members" => []
                )
                offset_to_kind[current_type_offset] = :struct  # Treat as struct for generic handling
            end
        end

        # Extract class type definitions (DW_TAG_class_type)
        # Similar to struct, C++ distinguishes but for our purposes treat the same
        if contains(line, "DW_TAG_class_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                current_struct_context = current_type_offset  # Set context for members
                # Store class info as a dict with members array
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "class",
                    "name" => "unknown_class",
                    "members" => []
                )
                offset_to_kind[current_type_offset] = :class
            end
        end

        # Extract enum type definitions (DW_TAG_enumeration_type)
        # Example: <1><abc>: DW_TAG_enumeration_type
        #          DW_AT_name: "Color"
        #          DW_AT_type: <0x27>  (underlying type, e.g., int)
        #          DW_AT_byte_size: 0x04
        if contains(line, "DW_TAG_enumeration_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                current_struct_context = current_type_offset  # Use for enumerators
                # Store enum info as a dict with enumerators array
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "enum",
                    "name" => "unknown_enum",
                    "underlying_type" => nothing,
                    "byte_size" => nothing,
                    "enumerators" => []
                )
                offset_to_kind[current_type_offset] = :enum
            end
        end

        # Extract array type definitions (DW_TAG_array_type)
        # Example: <1><def>: DW_TAG_array_type
        #          DW_AT_type: <0x27>  (element type)
        if contains(line, "DW_TAG_array_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                # Store array info with element type and dimensions
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "array",
                    "element_type" => nothing,
                    "dimensions" => []
                )
                offset_to_kind[current_type_offset] = :array
            end
        end

        # Extract array subrange (dimensions)
        # Example: <2><xyz>: DW_TAG_subrange_type
        #          DW_AT_type: <0x123>  (index type)
        #          DW_AT_upper_bound: 8  (for array[9])
        if contains(line, "DW_TAG_subrange_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                subrange_offset = "0x" * offset_match.captures[1]
                type_refs[subrange_offset] = Dict{String,Any}(
                    "kind" => "subrange",
                    "upper_bound" => nothing,
                    "parent" => current_type_offset  # Link to parent array
                )
                offset_to_kind[subrange_offset] = :subrange
            end
        end

        # Extract enumerator (enum member)
        # Example: <2><ghi>: DW_TAG_enumerator
        #          DW_AT_name: "Red"
        #          DW_AT_const_value: 0
        if contains(line, "DW_TAG_enumerator")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                enumerator_offset = "0x" * offset_match.captures[1]
                type_refs[enumerator_offset] = Dict{String,Any}(
                    "kind" => "enumerator",
                    "name" => nothing,
                    "value" => nothing,
                    "parent" => current_struct_context  # Track which enum this belongs to
                )
                offset_to_kind[enumerator_offset] = :enumerator
            end
        end

        # Extract subroutine type (function pointer signature)
        # Example: <1><jkl>: DW_TAG_subroutine_type
        #          DW_AT_type: <0x27>  (return type)
        if contains(line, "DW_TAG_subroutine_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                current_subroutine_offset = current_type_offset  # Track for param collection
                # Store function signature info
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "subroutine",
                    "return_type" => nothing,
                    "parameters" => []
                )
                offset_to_kind[current_type_offset] = :subroutine
            end
        end

        # Extract typedef (type aliases)
        # Example: <1><390>: Abbrev Number: 34 (DW_TAG_typedef)
        #          <391>   DW_AT_type        : <0x398>
        #          <395>   DW_AT_name        : int32_t
        if contains(line, "DW_TAG_typedef")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                # Store typedef info - will resolve to underlying type
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "typedef",
                    "name" => nothing,
                    "target" => nothing
                )
                offset_to_kind[current_type_offset] = :typedef
            end
        end

        # Extract inheritance (base class)
        # Example: <2><218>: DW_TAG_inheritance
        #          <219>   DW_AT_type        : <0x135>  (base class ref)
        #          <21d>   DW_AT_data_member_location: 0
        #          <21e>   DW_AT_accessibility: DW_ACCESS_public
        if contains(line, "DW_TAG_inheritance")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                inheritance_offset = "0x" * offset_match.captures[1]
                type_refs[inheritance_offset] = Dict{String,Any}(
                    "kind" => "inheritance",
                    "base_type" => nothing,
                    "offset" => 0,
                    "accessibility" => "public",  # default
                    "parent" => current_struct_context
                )
                offset_to_kind[inheritance_offset] = :inheritance
            end
        end

        # Extract template type parameter
        # Example: <2><2448>: DW_TAG_template_type_parameter
        #          <2449>   DW_AT_type        : <0x1d9>  (actual type in instantiation)
        #          <244d>   DW_AT_name        : T
        if contains(line, "DW_TAG_template_type_parameter")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                template_offset = "0x" * offset_match.captures[1]
                type_refs[template_offset] = Dict{String,Any}(
                    "kind" => "template_type",
                    "name" => nothing,
                    "type" => nothing,
                    "parent" => current_struct_context
                )
                offset_to_kind[template_offset] = :template_type
            end
        end

        # Extract template value parameter
        # Example: <2><247b>: DW_TAG_template_value_parameter
        #          <247c>   DW_AT_type        : <0x1d9>
        #          <2480>   DW_AT_name        : N
        #          <2481>   DW_AT_const_value : 10
        if contains(line, "DW_TAG_template_value_parameter")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                template_offset = "0x" * offset_match.captures[1]
                type_refs[template_offset] = Dict{String,Any}(
                    "kind" => "template_value",
                    "name" => nothing,
                    "type" => nothing,
                    "value" => nothing,
                    "parent" => current_struct_context
                )
                offset_to_kind[template_offset] = :template_value
            end
        end

        # Extract namespace
        # Example: <1><8bf>: DW_TAG_namespace
        #          <8c0>   DW_AT_name        : math
        if contains(line, "DW_TAG_namespace")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                ns_offset = "0x" * offset_match.captures[1]
                type_refs[ns_offset] = Dict{String,Any}(
                    "kind" => "namespace",
                    "name" => nothing
                )
                offset_to_kind[ns_offset] = :namespace
            end
        end

        # Extract type name for base types, structs, classes, enums, and new types
        # Example: <28>   DW_AT_name        : (indexed string: 0x3): int
        if contains(line, "DW_AT_name") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] in [:base, :struct, :class, :enum, :enumerator, :typedef, :template_type, :template_value, :namespace]
                # Extract just the type name after the last colon
                name_match = match(r":\s*([^:]+)\s*$", line)
                if !isnothing(name_match)
                    type_name = String(strip(name_match.captures[1]))  # Convert SubString to String
                    if offset_to_kind[tag_offset] == :base
                        # Base types are stored as simple strings
                        type_refs[tag_offset] = type_name
                    elseif offset_to_kind[tag_offset] in [:struct, :class, :enum, :typedef, :template_type, :template_value, :namespace]
                        # These types are dicts - update the name field
                        if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                            type_refs[tag_offset]["name"] = type_name
                        end
                    elseif offset_to_kind[tag_offset] == :enumerator
                        # Enumerator: store name
                        if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                            type_refs[tag_offset]["name"] = type_name
                        end
                    end
                end
            end
        end

        # Extract DW_AT_type for pointer/const/reference/enum/array/subroutine/typedef/template/inheritance types
        # Example: <9f7>   DW_AT_type        : <0x41>
        if contains(line, "DW_AT_type") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] in [:pointer, :const, :reference, :enum, :array, :subroutine, :typedef, :template_type, :template_value, :inheritance]
                type_match = match(r"<(0x[^>]+)>", line)
                if !isnothing(type_match)
                    target_offset = String(type_match.captures[1])  # Convert SubString to String
                    if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                        # Pointer/const/reference: target
                        if offset_to_kind[tag_offset] in [:pointer, :const, :reference]
                            type_refs[tag_offset]["target"] = target_offset
                        # Enum: underlying type
                        elseif offset_to_kind[tag_offset] == :enum
                            type_refs[tag_offset]["underlying_type"] = target_offset
                        # Array: element type
                        elseif offset_to_kind[tag_offset] == :array
                            type_refs[tag_offset]["element_type"] = target_offset
                        # Subroutine: return type
                        elseif offset_to_kind[tag_offset] == :subroutine
                            type_refs[tag_offset]["return_type"] = target_offset
                        # Typedef: target type
                        elseif offset_to_kind[tag_offset] == :typedef
                            type_refs[tag_offset]["target"] = target_offset
                        # Template type/value parameter: actual type
                        elseif offset_to_kind[tag_offset] in [:template_type, :template_value]
                            type_refs[tag_offset]["type"] = target_offset
                        # Inheritance: base class type
                        elseif offset_to_kind[tag_offset] == :inheritance
                            type_refs[tag_offset]["base_type"] = target_offset
                        end
                    end
                end
            end
        end

        # Extract DW_AT_byte_size for enums, structs, classes, and unions
        if contains(line, "DW_AT_byte_size") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] in [:enum, :struct, :class, :union]
                # Match value after colon
                size_match = match(r":\s*(0x[0-9a-fA-F]+|\d+)", line)
                if !isnothing(size_match)
                    val_str = size_match.captures[1]
                    if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                        if !startswith(val_str, "0x")
                            val = parse(Int, val_str)
                            type_refs[tag_offset]["byte_size"] = "0x" * string(val, base=16)
                        else
                            type_refs[tag_offset]["byte_size"] = val_str
                        end
                    end
                end
            end
        end

        # Extract DW_AT_const_value for enumerators and template value parameters
        if contains(line, "DW_AT_const_value") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] in [:enumerator, :template_value]
                # Value can be decimal or hex
                value_match = match(r":\s*(-?\d+)", line)
                if !isnothing(value_match)
                    if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                        type_refs[tag_offset]["value"] = parse(Int, value_match.captures[1])
                    end
                end
            end
        end

        # Extract DW_AT_upper_bound for array subranges (DWARF 4 and earlier)
        if contains(line, "DW_AT_upper_bound") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] == :subrange
                # Upper bound can be decimal or hex
                bound_match = match(r":\s*(\d+)", line)
                if !isnothing(bound_match)
                    if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                        # Upper bound is size-1, so actual size is upper_bound + 1
                        type_refs[tag_offset]["upper_bound"] = parse(Int, bound_match.captures[1])

                        # Add dimension to parent array
                        parent_offset = get(type_refs[tag_offset], "parent", nothing)
                        if !isnothing(parent_offset) && haskey(type_refs, parent_offset) &&
                           isa(type_refs[parent_offset], Dict) && haskey(type_refs[parent_offset], "dimensions")
                            size = parse(Int, bound_match.captures[1]) + 1  # Array[9] has upper_bound=8
                            push!(type_refs[parent_offset]["dimensions"], size)
                        end
                    end
                end
            end
        end

        # Extract DW_AT_count for array subranges (DWARF 5+, more common)
        if contains(line, "DW_AT_count") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] == :subrange
                # Count is the actual array size
                count_match = match(r":\s*(\d+)", line)
                if !isnothing(count_match)
                    if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                        # Count is the actual size
                        size = parse(Int, count_match.captures[1])
                        type_refs[tag_offset]["count"] = size

                        # Add dimension to parent array
                        parent_offset = get(type_refs[tag_offset], "parent", nothing)
                        if !isnothing(parent_offset) && haskey(type_refs, parent_offset) &&
                           isa(type_refs[parent_offset], Dict) && haskey(type_refs[parent_offset], "dimensions")
                            push!(type_refs[parent_offset]["dimensions"], size)
                        end
                    end
                end
            end
        end

        # Extract struct/class members (DW_TAG_member)
        # Example: <2><aa>: DW_TAG_member
        #          DW_AT_name: "x"
        #          DW_AT_type: <0xbd>
        #          DW_AT_data_member_location: 0x00
        if contains(line, "DW_TAG_member")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                member_offset = "0x" * offset_match.captures[1]
                type_refs[member_offset] = Dict{String,Any}(
                    "kind" => "member",
                    "name" => nothing,
                    "type" => nothing,
                    "offset" => nothing,
                    "parent" => current_struct_context  # Track which struct/class this belongs to
                )
                offset_to_kind[member_offset] = :member
            end
        end

        # Extract member attributes
        if haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] == :member
                member_info = type_refs[tag_offset]

                # Member name
                if contains(line, "DW_AT_name")
                    name_match = match(r":\s*([^:]+)\s*$", line)
                    if !isnothing(name_match)
                        member_info["name"] = String(strip(name_match.captures[1]))
                    end
                end

                # Member type reference
                if contains(line, "DW_AT_type")
                    type_match = match(r"<(0x[^>]+)>", line)
                    if !isnothing(type_match)
                        member_info["type"] = String(type_match.captures[1])
                    end
                end

                # Member byte offset within struct
                if contains(line, "DW_AT_data_member_location")
                    # Match explicit attribute name and value
                    offset_match = match(r"DW_AT_data_member_location\s*:\s*(0x[0-9a-fA-F]+|\d+)", line)
                    if !isnothing(offset_match)
                        val_str = offset_match.captures[1]
                        # Normalize to hex string if decimal
                        if !startswith(val_str, "0x")
                            val = parse(Int, val_str)
                            member_info["offset"] = "0x" * string(val, base=16)
                        else
                            member_info["offset"] = val_str
                        end
                    end
                end

                # When we have all info, add to parent struct/class
                parent_offset = get(member_info, "parent", nothing)
                if !isnothing(parent_offset) && haskey(type_refs, parent_offset) &&
                   isa(type_refs[parent_offset], Dict) && haskey(type_refs[parent_offset], "members")
                    if !isnothing(member_info["name"]) && !isnothing(member_info["type"])
                        # Only add if we haven't already added this member
                        existing_names = [m["name"] for m in type_refs[parent_offset]["members"]]
                        if !(member_info["name"] in existing_names)
                            push!(type_refs[parent_offset]["members"], Dict(
                                "name" => member_info["name"],
                                "type" => member_info["type"],
                                "offset" => get(member_info, "offset", "0x00")
                            ))
                        end
                    end
                end
            end

            # Extract enumerator attributes and add to parent enum
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] == :enumerator
                enumerator_info = type_refs[tag_offset]

                # When we have all info, add to parent enum
                parent_offset = get(enumerator_info, "parent", nothing)
                if !isnothing(parent_offset) && haskey(type_refs, parent_offset) &&
                   isa(type_refs[parent_offset], Dict) && haskey(type_refs[parent_offset], "enumerators")
                    if !isnothing(enumerator_info["name"]) && !isnothing(enumerator_info["value"])
                        # Only add if we haven't already added this enumerator
                        existing_names = [e["name"] for e in type_refs[parent_offset]["enumerators"]]
                        if !(enumerator_info["name"] in existing_names)
                            push!(type_refs[parent_offset]["enumerators"], Dict(
                                "name" => enumerator_info["name"],
                                "value" => enumerator_info["value"]
                            ))
                        end
                    end
                end
            end
        end
    end

    # Debug: Show type refs collected
    base_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :base && isa(type_refs[k], String), keys(type_refs))
    pointer_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :pointer, keys(type_refs))
    struct_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :struct && isa(type_refs[k], Dict), keys(type_refs))
    class_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :class && isa(type_refs[k], Dict), keys(type_refs))
    enum_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :enum && isa(type_refs[k], Dict), keys(type_refs))
    array_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :array && isa(type_refs[k], Dict), keys(type_refs))
    subroutine_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :subroutine && isa(type_refs[k], Dict), keys(type_refs))

    # Count total members/enumerators extracted
    total_members = 0
    total_enumerators = 0
    for (k, v) in type_refs
        if isa(v, Dict)
            if haskey(v, "members")
                total_members += length(v["members"])
            end
            if haskey(v, "enumerators")
                total_enumerators += length(v["enumerators"])
            end
        end
    end

    println("Types collected: $base_count base, $pointer_count pointer, $struct_count struct, $class_count class")
    println("   Advanced types: $enum_count enum, $array_count array, $subroutine_count function_pointer")
    println("   Struct/class members: $total_members, Enum enumerators: $total_enumerators")

    # Collect all struct/class/enum names for type resolution
    struct_names = Set{String}()
    enum_names = Set{String}()

    for (offset, type_info) in type_refs
        if isa(type_info, Dict)
            kind = get(type_info, "kind", "")
            name = get(type_info, "name", "")

            if kind in ["struct", "class"] && !isempty(name) && name != "unknown" && name != "unknown_struct" && name != "unknown_class"
                push!(struct_names, name)
            elseif kind == "enum" && !isempty(name) && name != "unknown" && name != "unknown_enum"
                push!(enum_names, name)
            end
        end
    end

    # Helper function to resolve type references (follows pointer/const/reference chains)
    function resolve_type(type_ref::String, type_refs::Dict, visited::Set{String}=Set{String}())::String
        # Avoid infinite loops
        if type_ref in visited
            return "unknown"
        end
        push!(visited, type_ref)

        if !haskey(type_refs, type_ref)
            # Type not found in our table - likely a forward reference or external type
            return "unknown"
        end

        type_info = type_refs[type_ref]

        # Base type - just return the name
        if isa(type_info, String)
            return type_info
        end

        # Pointer/const/reference/enum/array/subroutine type - follow the chain
        if isa(type_info, Dict)
            kind = get(type_info, "kind", nothing)

            # Struct or class - return the name
            if kind in ["struct", "class"]
                return get(type_info, "name", "unknown")
            end

            # Enum - return the name
            if kind == "enum"
                return get(type_info, "name", "unknown_enum")
            end

            # Array - return element_type[size] or element_type[size1][size2]...
            if kind == "array"
                element_type_ref = get(type_info, "element_type", nothing)
                dimensions = get(type_info, "dimensions", [])

                if !isnothing(element_type_ref)
                    element_type = resolve_type(element_type_ref, type_refs, visited)
                    if !isempty(dimensions)
                        # Build array notation: type[size1][size2]...
                        dim_str = join(["[$d]" for d in dimensions], "")
                        return element_type * dim_str
                    else
                        # No dimensions specified (incomplete DWARF)
                        return element_type * "[]"
                    end
                end
                return "unknown[]"
            end

            # Subroutine (function pointer) - return detailed signature with parameters
            if kind == "subroutine"
                return_type_ref = get(type_info, "return_type", nothing)
                param_refs = get(type_info, "parameters", [])

                # Resolve return type
                ret_type_str = if !isnothing(return_type_ref)
                    resolve_type(return_type_ref, type_refs, visited)
                else
                    "void"
                end

                # Resolve parameter types
                param_type_strs = String[]
                for param_ref in param_refs
                    if isa(param_ref, String)
                        param_type = resolve_type(param_ref, type_refs, visited)
                        push!(param_type_strs, param_type)
                    elseif isa(param_ref, Dict)
                        # Parameter might have additional info (name, etc.)
                        param_type_ref = get(param_ref, "type", nothing)
                        if !isnothing(param_type_ref)
                            param_type = resolve_type(param_type_ref, type_refs, visited)
                            push!(param_type_strs, param_type)
                        end
                    end
                end

                # Build signature: function_ptr(return_type; param1, param2, ...)
                # Using semicolon to separate return from params for easier parsing
                if isempty(param_type_strs)
                    return "function_ptr($ret_type_str)"
                else
                    param_list = join(param_type_strs, ", ")
                    return "function_ptr($ret_type_str; $param_list)"
                end
            end

            # Typedef - special handling for well-known types
            if kind == "typedef"
                typedef_name = get(type_info, "name", "")

                # Preserve well-known typedefs that have portable Julia mappings
                # These should NOT be resolved to their underlying type
                well_known_typedefs = [
                    "size_t", "ssize_t", "ptrdiff_t",
                    "intptr_t", "uintptr_t",
                    "int8_t", "int16_t", "int32_t", "int64_t",
                    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
                    "wchar_t", "char16_t", "char32_t"
                ]

                if typedef_name in well_known_typedefs
                    # Preserve the typedef name for portable mapping
                    return typedef_name
                end

                # For other typedefs, resolve to underlying type
                target_ref = get(type_info, "target", nothing)
                if !isnothing(target_ref)
                    return resolve_type(target_ref, type_refs, visited)
                else
                    # If no target, return the typedef name itself
                    return typedef_name == "" ? "unknown" : typedef_name
                end
            end

            target = get(type_info, "target", nothing)

            if isnothing(target)
                # Pointer/const/reference without target (shouldn't happen but handle it)
                return kind == "pointer" ? "void*" : "unknown"
            end

            # Recursively resolve the target type
            target_type = resolve_type(target, type_refs, visited)

            # Build the full type string
            if kind == "pointer"
                return target_type * "*"
            elseif kind == "const"
                return "const " * target_type
            elseif kind == "reference"
                return target_type * "&"
            end
        end

        return "unknown"
    end

    # Second pass: Extract function return types and parameters
    current_function_offset = nothing
    current_function_name = nothing
    current_function_linkage = nothing
    current_function_level = nothing
    function_processed = false  # State flag: true once we leave function's own level
    params_for_this_function = []  # Collect parameters for current function, NEW name and usage

    params_for_this_function = [] # NEW: Temporary storage for parameters of the current function being parsed

    # Track formal parameters (similar to struct members)
    current_param_offset = nothing

    last_seen_level = nothing  # Track last level for attribute lines

    for line in split(output, '\n')
        line = strip(line)

        # Extract nesting level from DWARF format: <level><offset>
        # Tags have: <level><offset>: ...
        # Attributes have: <offset>   DW_AT_...
        level_match = match(r"^\s*<(\d+)><", line)
        if !isnothing(level_match)
            last_seen_level = parse(Int, level_match.captures[1])
        end
        # Use last seen level for attributes (which don't have level prefix)
        current_level = last_seen_level

        # Detect when we've left the function's own level (entered child tags)
        if !isnothing(current_function_offset) && !isnothing(current_level) && !isnothing(current_function_level)
            if current_level > current_function_level && !function_processed
                # If a function was just processed and its parameters haven't been associated
                # (e.g., if it was a void function without DW_AT_type), associate them now.
                function_key_prev = !isnothing(current_function_linkage) ? current_function_linkage : current_function_name
                if !isnothing(function_key_prev) && haskey(return_types, function_key_prev) && !haskey(return_types[function_key_prev], "parameters")
                     return_types[function_key_prev]["parameters"] = params_for_this_function
                end

                # We've entered a child tag (like DW_TAG_formal_parameter)
                # This means we've finished processing the function's own attributes
                # If no return type was found, it's void
                function_key = !isnothing(current_function_linkage) ? current_function_linkage : current_function_name
                if !isnothing(function_key) && !haskey(return_types, function_key)
                    return_types[function_key] = Dict(
                        "c_type" => "void",
                        "julia_type" => "Cvoid",
                        "size" => 0,
                        "parameters" => params_for_this_function # Ensure parameters are passed here too
                    )
                end
                function_processed = true
            end
        end

        # Detect function start (DW_TAG_subprogram)
        if contains(line, "DW_TAG_subprogram")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                # Before resetting for new function, if a function was just parsed, ensure its parameters are stored
                # This handles cases where a function might not have DW_AT_type (void return)
                if !isnothing(current_function_offset)
                    function_key_prev = !isnothing(current_function_linkage) ? current_function_linkage : current_function_name
                    if !isnothing(function_key_prev) && haskey(return_types, function_key_prev) && !haskey(return_types[function_key_prev], "parameters")
                        return_types[function_key_prev]["parameters"] = params_for_this_function
                    end
                end

                current_function_offset = "0x" * offset_match.captures[1]
                current_function_level = current_level
                current_function_name = nothing
                current_function_linkage = nothing
                function_processed = false  # Reset flag for new function
                params_for_this_function = []  # NEW: Reset for the new function
                current_subroutine_offset = nothing  # Reset subroutine context when entering function
            end
        end

        # Detect formal parameter (child of subprogram or subroutine_type)
        # Example: <2><7a>: DW_TAG_formal_parameter
        if contains(line, "DW_TAG_formal_parameter")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_param_offset = "0x" * offset_match.captures[1]

                # Determine position based on context (function or subroutine type)
                position = if !isnothing(current_subroutine_offset) && haskey(type_refs, current_subroutine_offset)
                    # Parameter belongs to subroutine type
                    length(type_refs[current_subroutine_offset]["parameters"])
                else
                    # Parameter belongs to function
                    length(params_for_this_function)
                end

                type_refs[current_param_offset] = Dict{String,Any}(
                    "kind" => "parameter",
                    "name" => nothing,
                    "type" => nothing,
                    "position" => position,
                    "parent_subroutine" => current_subroutine_offset  # Track which subroutine this belongs to
                )
                offset_to_kind[current_param_offset] = :parameter
            end
        end

        # Only process attributes at the function level (not in child tags like parameters)
        in_function_context = !isnothing(current_function_offset) &&
                             !isnothing(current_level) &&
                             !isnothing(current_function_level) &&
                             current_level == current_function_level

        # Extract function name (for C functions without mangling)
        # Example: <2f>   DW_AT_name        : (indexed string: 0xc): test_sin
        if contains(line, "DW_AT_name") && in_function_context && isnothing(current_function_name)
            name_match = match(r":\s*([^:]+)\s*$", line)
            if !isnothing(name_match)
                current_function_name = String(strip(name_match.captures[1]))  # Convert SubString to String
            end
        end

        # Extract linkage name (mangled name for C++ functions)
        # Example: <5c>   DW_AT_linkage_name: (indexed string: 0x8): _ZN10Calculator5powerEdd
        if contains(line, "DW_AT_linkage_name") && in_function_context
            # Extract just the mangled name after the last colon
            linkage_match = match(r":\s*([^:\s]+)\s*$", line)
            if !isnothing(linkage_match)
                current_function_linkage = String(strip(linkage_match.captures[1]))  # Convert SubString to String
            end
        end

        # Extract virtuality
        # Example: <60>   DW_AT_virtuality  : 1 (virtual)
        if contains(line, "DW_AT_virtuality") && in_function_context
            # Virtuality is usually 1 (virtual) or 2 (pure virtual)
            virt_match = match(r":\s*(\d+)", line)
            if !isnothing(virt_match)
                is_virtual = parse(Int, virt_match.captures[1]) > 0
                
                # Store this temporarily in type_refs for the current function offset
                if !isnothing(current_function_offset)
                    if !haskey(type_refs, current_function_offset)
                        type_refs[current_function_offset] = Dict{String,Any}()
                    end
                    if isa(type_refs[current_function_offset], Dict)
                        type_refs[current_function_offset]["is_virtual"] = is_virtual
                    end
                end
            end
        end

        # Extract function return type reference
        # CRITICAL: Only match DW_AT_type at the function's own level, not from parameters!
        if contains(line, "DW_AT_type") && in_function_context && !function_processed
            type_match = match(r"<(0x[^>]+)>", line)
            if !isnothing(type_match)
                type_ref = String(type_match.captures[1])  # Convert SubString to String
                # Resolve the type reference (follows pointer/const/reference chains)
                c_type = resolve_type(type_ref, type_refs)

                # Map to Julia type using comprehensive mapping
                # Check if it's a struct/class type first
                julia_type = if haskey(type_refs, type_ref) && isa(type_refs[type_ref], Dict) &&
                               get(type_refs[type_ref], "kind", nothing) in ["struct", "class"]
                    # It's a struct/class - use the struct name as the Julia type
                    c_type
                else
                    dwarf_type_to_julia(c_type)
                end
                type_size = get_type_size(c_type)

                # Use linkage name (C++) or fall back to function name (C)
                function_key = !isnothing(current_function_linkage) ? current_function_linkage : current_function_name

                # Check for virtuality stored earlier
                is_virtual = false
                if !isnothing(current_function_offset) && haskey(type_refs, current_function_offset) &&
                   isa(type_refs[current_function_offset], Dict)
                    is_virtual = get(type_refs[current_function_offset], "is_virtual", false)
                end

                # Store if we have a function identifier
                if !isnothing(function_key)
                    return_types[function_key] = Dict(
                        "c_type" => c_type,
                        "julia_type" => julia_type,
                        "size" => type_size,
                        "is_virtual" => is_virtual, # NEW: Store virtuality
                        "parameters" => params_for_this_function  # NEW: Use params_for_this_function
                    )
                end
                function_processed = true  # Mark as processed
            end
        end

        # Extract parameter attributes (name and type)
        # Parameters are child tags (level > function_level)
        if !isnothing(current_param_offset) && haskey(type_refs, current_param_offset) &&
           haskey(offset_to_kind, current_param_offset) && offset_to_kind[current_param_offset] == :parameter

            param_info = type_refs[current_param_offset]

            # Extract parameter name
            if contains(line, "DW_AT_name") && !isnothing(current_level) &&
               !isnothing(current_function_level) && current_level > current_function_level
                name_match = match(r":\s*([^:]+)\s*$", line)
                if !isnothing(name_match)
                    param_info["name"] = String(strip(name_match.captures[1]))
                end
            end

            # Extract parameter type
            if contains(line, "DW_AT_type") && !isnothing(current_level) &&
               !isnothing(current_function_level) && current_level > current_function_level
                type_match = match(r"<(0x[^>]+)>", line)
                if !isnothing(type_match)
                    param_info["type"] = String(type_match.captures[1])

                    # Check if this parameter belongs to a subroutine type or a function
                    parent_subroutine = get(param_info, "parent_subroutine", nothing)

                    if !isnothing(parent_subroutine) && haskey(type_refs, parent_subroutine)
                        # Add to subroutine type's parameter list
                        type_ref = param_info["type"]
                        # Store just the type reference for now, will resolve later
                        push!(type_refs[parent_subroutine]["parameters"], type_ref)
                        current_param_offset = nothing
                    elseif !isnothing(param_info["name"]) && !isnothing(param_info["type"])
                        # Add to function's parameter list (needs both name and type)
                        # Resolve type
                        type_ref = param_info["type"]
                        c_type = resolve_type(type_ref, type_refs)
                        julia_type = cpp_to_julia_type(c_type, struct_names, enum_names)

                        param_dict = Dict(
                            "name" => param_info["name"],
                            "c_type" => c_type,
                            "julia_type" => julia_type,
                            "position" => param_info["position"]
                        )

                        # If this is a function pointer, preserve the full signature
                        if startswith(c_type, "function_ptr(")
                            param_dict["function_pointer_signature"] = c_type
                        end

                        push!(params_for_this_function, param_dict)

                        # Reset param offset to avoid re-processing
                        current_param_offset = nothing
                    end
                end
            end
        end
    end

    # Handle last function in file (might be void)
    if !isnothing(current_function_offset)
        function_key = !isnothing(current_function_linkage) ? current_function_linkage : current_function_name
        if !isnothing(function_key)
            if !haskey(return_types, function_key)
                # No DW_AT_type found = void return
                return_types[function_key] = Dict(
                    "c_type" => "void",
                    "julia_type" => "Cvoid",
                    "size" => 0,
                    "parameters" => params_for_this_function # NEW: Use params_for_this_function
                )
            elseif !haskey(return_types[function_key], "parameters")
                # Function already has return type, but may need parameters added
                return_types[function_key]["parameters"] = params_for_this_function # NEW: Use params_for_this_function
            end
        end
    end

    if !isempty(return_types)
        println("    Extracted $(length(return_types)) return types from DWARF")
    else
        println("     No DWARF return type info found (compile with -g flag)")
    end

    # Extract struct definitions with member information
    struct_defs = Dict{String,Dict{String,Any}}()
    for (offset, type_info) in type_refs
        if isa(type_info, Dict) && get(type_info, "kind", nothing) in ["struct", "class"]
            struct_name = get(type_info, "name", "unknown")
            if struct_name != "unknown" && struct_name != "unknown_struct" && struct_name != "unknown_class"
                # Resolve member types to Julia types
                resolved_members = []
                for member in get(type_info, "members", [])
                    member_type_ref = get(member, "type", nothing)
                    if !isnothing(member_type_ref)
                        c_type = resolve_type(member_type_ref, type_refs)
                        julia_type = cpp_to_julia_type(c_type, struct_names, enum_names)
                        push!(resolved_members, Dict(
                            "name" => get(member, "name", "unknown"),
                            "c_type" => c_type,
                            "julia_type" => julia_type,
                            "size" => get_type_size(c_type),
                            "offset" => get(member, "offset", "0x00")
                        ))
                    end
                end

                if !isempty(resolved_members)
                    struct_def = Dict{String,Any}(
                        "kind" => get(type_info, "kind", "struct"),
                        "byte_size" => get(type_info, "byte_size", "0x0"),
                        "members" => resolved_members
                    )

                    # Add inheritance information if available
                    base_classes = []
                    for (inh_offset, inh_info) in type_refs
                        if isa(inh_info, Dict) && get(inh_info, "kind", nothing) == "inheritance" &&
                           get(inh_info, "parent", nothing) == offset
                            base_type_ref = get(inh_info, "base_type", nothing)
                            if !isnothing(base_type_ref)
                                base_type = resolve_type(base_type_ref, type_refs)
                                push!(base_classes, Dict(
                                    "type" => base_type,
                                    "accessibility" => get(inh_info, "accessibility", "public")
                                ))
                            end
                        end
                    end
                    if !isempty(base_classes)
                        struct_def["base_classes"] = base_classes
                    end

                    # Add template parameters if available
                    template_params = []
                    for (tmpl_offset, tmpl_info) in type_refs
                        if isa(tmpl_info, Dict) && get(tmpl_info, "parent", nothing) == offset
                            if get(tmpl_info, "kind", nothing) == "template_type"
                                type_ref = get(tmpl_info, "type", nothing)
                                param_type = !isnothing(type_ref) ? resolve_type(type_ref, type_refs) : "Any"
                                push!(template_params, Dict(
                                    "kind" => "type",
                                    "name" => get(tmpl_info, "name", "T"),
                                    "type" => param_type
                                ))
                            elseif get(tmpl_info, "kind", nothing) == "template_value"
                                type_ref = get(tmpl_info, "type", nothing)
                                param_type = !isnothing(type_ref) ? resolve_type(type_ref, type_refs) : "Int"
                                push!(template_params, Dict(
                                    "kind" => "value",
                                    "name" => get(tmpl_info, "name", "N"),
                                    "type" => param_type,
                                    "value" => get(tmpl_info, "value", nothing)
                                ))
                            end
                        end
                    end
                    if !isempty(template_params)
                        struct_def["template_params"] = template_params
                    end

                    struct_defs[struct_name] = struct_def
                end
            end
        end
    end

    if !isempty(struct_defs)
        println("    Extracted $(length(struct_defs)) struct/class definitions with members")
    end

    # Extract enum definitions with enumerator information
    enum_defs = Dict{String,Dict{String,Any}}()
    for (offset, type_info) in type_refs
        if isa(type_info, Dict) && get(type_info, "kind", nothing) == "enum"
            enum_name = get(type_info, "name", "unknown")
            if enum_name != "unknown" && enum_name != "unknown_enum"
                # Get underlying type
                underlying_type_ref = get(type_info, "underlying_type", nothing)
                underlying_c_type = "int"  # Default
                if !isnothing(underlying_type_ref)
                    underlying_c_type = resolve_type(underlying_type_ref, type_refs)
                end
                underlying_julia_type = cpp_to_julia_type(underlying_c_type, struct_names, enum_names)

                # Get enumerators
                enumerators = get(type_info, "enumerators", [])

                if !isempty(enumerators)
                    enum_defs[enum_name] = Dict(
                        "kind" => "enum",
                        "underlying_type" => underlying_c_type,
                        "julia_type" => underlying_julia_type,
                        "byte_size" => get(type_info, "byte_size", "0x04"),
                        "enumerators" => enumerators
                    )
                end
            end
        end
    end

    if !isempty(enum_defs)
        println("    Extracted $(length(enum_defs)) enum definitions with enumerators")
    end

    # Store enum_defs in struct_defs with special marker (for now, to maintain API compatibility)
    # Later we can extend the return type
    for (enum_name, enum_info) in enum_defs
        struct_defs["__enum__" * enum_name] = enum_info
    end

    return (return_types, struct_defs)
end

"""
Extract compilation metadata from source files and binary.
This is the core of automatic wrapper generation!
"""
function extract_compilation_metadata(config::RepliBuildConfig, source_files::Vector{String},
                                      binary_path::String)::Dict{String,Any}
    println(" Extracting compilation metadata...")

    # Extract symbols from compiled binary
    symbols = extract_symbols_from_binary(binary_path)
    println("   Found $(length(symbols)) exported symbols")

    # Extract return types and struct definitions from DWARF debug info (if available)
    (dwarf_return_types, struct_defs) = extract_dwarf_return_types(binary_path)

    # Collect struct/enum names for type resolution in function signatures
    sig_struct_names = Set{String}()
    sig_enum_names = Set{String}()
    for (name, info) in struct_defs
        if startswith(name, "__enum__")
            enum_name = replace(name, "__enum__" => "")
            push!(sig_enum_names, enum_name)
        else
            push!(sig_struct_names, name)
        end
    end

    # Parse function signatures (basic type inference from symbol names)
    functions = parse_function_signatures(symbols, sig_struct_names, sig_enum_names)

    # Merge DWARF return types and parameters into function metadata (overrides inference)
    for func in functions
        mangled = func["mangled"]
        if haskey(dwarf_return_types, mangled)
            dwarf_info = dwarf_return_types[mangled]

            # Merge return type (only the return type fields, not parameters)
            func["return_type"] = Dict(
                "c_type" => get(dwarf_info, "c_type", "void"),
                "julia_type" => get(dwarf_info, "julia_type", "Cvoid"),
                "size" => get(dwarf_info, "size", 0)
            )
            func["return_type_source"] = "dwarf"

            # Merge parameters if available from DWARF (at function level, not in return_type)
            if haskey(dwarf_info, "parameters") && !isempty(dwarf_info["parameters"])
                func["parameters"] = dwarf_info["parameters"]
                func["parameters_source"] = "dwarf"
            else
                func["parameters_source"] = "inferred"
            end
        else
            func["return_type_source"] = "inferred"
            func["parameters_source"] = "inferred"
        end
    end

    # Build type registry (basic types + inferred types)
    type_registry = build_type_registry(functions)

    # Gather compiler info
    llvm_version = try
        (out, _) = BuildBridge.execute("llvm-config", ["--version"])
        strip(out)
    catch
        "unknown"
    end

    clang_version = try
        (out, _) = BuildBridge.execute("clang++", ["--version"])
        first(split(out, '\n'))
    catch
        "unknown"
    end

    metadata = Dict{String,Any}(
        "timestamp" => string(now()),
        "project" => config.project.name,
        "binary_path" => binary_path,
        "binary_type" => string(config.binary.type),

        # Source information
        "source_files" => source_files,
        "include_dirs" => get_include_dirs(config),
        "compile_flags" => get_compile_flags(config),

        # Symbols and functions
        "symbols" => symbols,
        "functions" => functions,
        "function_count" => length(functions),

        # Type mappings
        "type_registry" => type_registry,
        "struct_definitions" => struct_defs,  # NEW: Struct member layout from DWARF

        # Compiler information
        "compiler_info" => Dict(
            "llvm_version" => llvm_version,
            "clang_version" => clang_version,
            "target_triple" => "x86_64-unknown-linux-gnu",  # TODO: detect
            "optimization_level" => config.link.optimization_level,
            "lto_enabled" => config.link.enable_lto
        ),

        # Metadata version (for future compatibility)
        "metadata_version" => "1.0"
    )

    return metadata
end

"""
Parse function signatures from symbol information.
Infers parameter types and return types from demangled names.
"""
function parse_function_signatures(symbols::Vector{Dict{String,Any}},
                                   struct_names::Set{String}=Set{String}(),
                                   enum_names::Set{String}=Set{String}())::Vector{Dict{String,Any}}
    functions = Dict{String,Any}[]

    for sym in symbols
        demangled = sym["demangled"]

        # Skip vtables, typeinfo, etc.
        if startswith(demangled, "vtable") || startswith(demangled, "typeinfo")
            continue
        end

        func_info = Dict{String,Any}(
            "name" => extract_function_name(demangled),
            "mangled" => sym["mangled"],
            "demangled" => demangled,
            "return_type" => infer_return_type(demangled),
            "parameters" => parse_parameters(demangled, struct_names, enum_names),
            "is_method" => contains(demangled, "::"),
            "class" => extract_class_name(demangled),
            "exported" => true
        )

        push!(functions, func_info)
    end

    return functions
end

"""
Extract function name from demangled signature.
Example: "Calculator::compute(int, int, char)" -> "compute"
"""
function extract_function_name(demangled::String)::String
    # Handle class methods
    if contains(demangled, "::")
        parts = split(demangled, "::")
        if length(parts) >= 2
            # Get last part before (
            name_part = parts[end]
            return split(name_part, '(')[1]
        end
    end

    # Handle free functions
    return split(demangled, '(')[1]
end

"""
Extract class name from method signature.
Example: "Calculator::compute(int, int)" -> "Calculator"
"""
function extract_class_name(demangled::String)::String
    if contains(demangled, "::")
        parts = split(demangled, "::")
        return parts[1]
    end
    return ""
end

"""
Infer return type from function signature (basic heuristic).
In future: extract from debug info or DWARF.
"""
function infer_return_type(demangled::String)::Dict{String,Any}
    # Default: assume int return for now
    # TODO: Parse DWARF debug info for exact types
    return Dict(
        "c_type" => "int",
        "julia_type" => "Cint",
        "size" => 4
    )
end

"""
Parse parameter types from demangled signature.
Example: "add(int, int)" -> [{"type": "int", "julia_type": "Cint"}, ...]
"""
function parse_parameters(demangled::String,
                         struct_names::Set{String}=Set{String}(),
                         enum_names::Set{String}=Set{String}())::Vector{Dict{String,Any}}
    params = Dict{String,Any}[]

    # Extract parameter list from signature
    if !contains(demangled, '(')
        return params
    end

    param_str = match(r"\((.*?)\)", demangled)
    if isnothing(param_str)
        return params
    end

    param_types = split(param_str.captures[1], ',')

    for (i, ptype) in enumerate(param_types)
        ptype = strip(ptype)
        if isempty(ptype)
            continue
        end

        julia_type = cpp_to_julia_type(ptype, struct_names, enum_names)

        push!(params, Dict(
            "name" => "arg$i",
            "c_type" => ptype,
            "julia_type" => julia_type,
            "position" => i-1
        ))
    end

    return params
end

"""
Convert C++ type to Julia type.
Basic type mapping - can be enhanced with type registry.

# Arguments
- `cpp_type`: C++ type string
- `struct_names`: Set of known struct/class names from DWARF
- `enum_names`: Set of known enum names from DWARF
"""
function cpp_to_julia_type(cpp_type::AbstractString,
                           struct_names::Set{String}=Set{String}(),
                           enum_names::Set{String}=Set{String}())::String
    # Strip qualifiers
    cpp_type = strip(cpp_type)
    cpp_type = replace(cpp_type, r"^const\s+" => "")
    cpp_type = replace(cpp_type, r"^volatile\s+" => "")

    type_map = Dict(
        "int" => "Cint",
        "unsigned int" => "Cuint",
        "long" => "Clong",
        "unsigned long" => "Culong",
        "short" => "Cshort",
        "unsigned short" => "Cushort",
        "char" => "UInt8",
        "unsigned char" => "UInt8",
        "float" => "Cfloat",
        "double" => "Cdouble",
        "bool" => "Bool",
        "void" => "Cvoid",
        "char*" => "Cstring",
        "const char*" => "Cstring",
        "size_t" => "Csize_t",
        "ssize_t" => "Cssize_t",
        "ptrdiff_t" => "Cptrdiff_t",
        "intptr_t" => "Cintptr_t",
        "uintptr_t" => "Cuintptr_t",
        "int8_t" => "Int8",
        "uint8_t" => "UInt8",
        "int16_t" => "Int16",
        "uint16_t" => "UInt16",
        "int32_t" => "Int32",
        "uint32_t" => "UInt32",
        "int64_t" => "Int64",
        "uint64_t" => "UInt64",
        "long long" => "Clonglong",
        "unsigned long long" => "Culonglong"
    )

    # Handle arrays: type[size] or type[size1][size2]
    # Examples: "double[3]" -> "NTuple{3, Cdouble}"
    #           "int[3][4]" -> "NTuple{12, Cint}"  (flattened)
    if contains(cpp_type, "[") && contains(cpp_type, "]")
        # Extract element type and dimensions
        array_match = match(r"^(.+?)(\[.+\])$", cpp_type)
        if !isnothing(array_match)
            element_type_str = strip(array_match.captures[1])
            dims_str = array_match.captures[2]

            # Parse dimensions: [3][4] -> [3, 4]
            dims = Int[]
            for dim_match in eachmatch(r"\[(\d+)\]", dims_str)
                push!(dims, parse(Int, dim_match.captures[1]))
            end

            if !isempty(dims)
                # Calculate total size (product of dimensions)
                total_size = prod(dims)

                # Map element type to Julia (with struct/enum awareness)
                element_julia_type = cpp_to_julia_type(element_type_str, struct_names, enum_names)

                # Return as NTuple (flattened)
                return "NTuple{$total_size, $element_julia_type}"
            end
        end
    end

    # Handle function pointers (and pointers to function pointers)
    # First, strip trailing '*' if present, and then check for "function_ptr("
    temp_cpp_type = replace(cpp_type, r"\*+$" => "") # Temporarily strip pointers for the check
    if startswith(temp_cpp_type, "function_ptr(")
        return "Ptr{Cvoid}"
    end

    # Check if it's a user-defined struct/class (before pointer handling)
    base_type = replace(cpp_type, r"[*&\s]+$" => "")  # Strip pointer/ref/spaces
    base_type = strip(base_type)

    if base_type in struct_names
        # Handle pointers: Vector3* → Ptr{Vector3}
        if contains(cpp_type, "*")
            return "Ptr{$base_type}"
        elseif contains(cpp_type, "&")
            return "Ref{$base_type}"
        else
            return base_type  # Use struct name directly
        end
    end

    # Check for enums
    if base_type in enum_names
        # Enums are used directly by name (will be @enum in Julia)
        return base_type
    end

    # Handle pointers (for non-struct types)
    if endswith(cpp_type, "*")
        # Map the base type first, then wrap in Ptr{}
        base = strip(cpp_type[1:end-1])
        base_julia = cpp_to_julia_type(base, struct_names, enum_names)
        # If base mapped to a specific type, use it; otherwise fall back to Ptr{Cvoid}
        if base_julia != "Any" && base_julia != base
            return "Ptr{$base_julia}"
        else
            return "Ptr{Cvoid}"
        end
    end

    # Handle references (for non-struct types)
    if endswith(cpp_type, "&")
        base = strip(cpp_type[1:end-1])
        return "Ref{$(cpp_to_julia_type(base, struct_names, enum_names))}"
    end

    # Fallback to type map or Any
    return get(type_map, cpp_type, "Any")
end

"""
Build type registry from functions.
Collects all unique types used in the codebase.
"""
function build_type_registry(functions::Vector{Dict{String,Any}})::Dict{String,Any}
    registry = Dict{String,Any}()

    # Add standard types
    standard_types = Dict(
        "int" => Dict("size" => 4, "alignment" => 4, "julia_type" => "Cint"),
        "float" => Dict("size" => 4, "alignment" => 4, "julia_type" => "Cfloat"),
        "double" => Dict("size" => 8, "alignment" => 8, "julia_type" => "Cdouble"),
        "char" => Dict("size" => 1, "alignment" => 1, "julia_type" => "UInt8"),
        "bool" => Dict("size" => 1, "alignment" => 1, "julia_type" => "Bool"),
        "void" => Dict("size" => 0, "alignment" => 0, "julia_type" => "Cvoid")
    )

    merge!(registry, standard_types)

    # Collect types from functions
    for func in functions
        # Add return type
        if haskey(func, "return_type")
            rt = func["return_type"]
            if haskey(rt, "c_type")
                registry[rt["c_type"]] = rt
            end
        end

        # Add parameter types
        if haskey(func, "parameters")
            for param in func["parameters"]
                if haskey(param, "c_type")
                    registry[param["c_type"]] = Dict(
                        "julia_type" => get(param, "julia_type", "Any"),
                        "inferred" => true
                    )
                end
            end
        end
    end

    return registry
end

"""
Save compilation metadata to JSON file next to binary.
This enables automatic wrapper generation!
"""
function save_compilation_metadata(config::RepliBuildConfig, source_files::Vector{String},
                                   binary_path::String)::String
    # Extract metadata
    metadata = extract_compilation_metadata(config, source_files, binary_path)

    # Save to JSON file next to binary
    output_dir = get_output_path(config)
    metadata_path = joinpath(output_dir, "compilation_metadata.json")

    open(metadata_path, "w") do io
        JSON.print(io, metadata, 2)  # Pretty print with indent=2
    end

    println("    Saved metadata: $metadata_path")
    return metadata_path
end

end # module Compiler
