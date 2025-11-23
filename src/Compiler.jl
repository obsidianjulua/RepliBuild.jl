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
    cmd_args = vcat(
        ["-S", "-emit-llvm"],  # Emit LLVM IR
        get_compile_flags(config),
        ["-I$dir" for dir in get_include_dirs(config)],
        ["-D$k=$v" for (k, v) in config.compile.defines],
        ["-o", ir_file, cpp_file]
    )

    # Compile using BuildBridge
    (output, exitcode) = BuildBridge.execute("clang++", cmd_args)

    if !isempty(output) && exitcode != 0
        println("  ❌ $(basename(cpp_file)): $output")
    end

    success = isfile(ir_file)
    return (ir_file, success, exitcode)
end

"""
Compile multiple C++ files to LLVM IR (with parallel support).
Returns vector of IR file paths.
"""
function compile_to_ir(config::RepliBuildConfig, cpp_files::Vector{String})
    println("🔧 Compiling to LLVM IR...")

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
        println("  ✅ All files cached, nothing to compile")
        return ir_files
    end

    println("  📝 Compiling $(length(files_to_compile)) files...")

    # Compile in parallel if enabled
    if is_parallel_enabled(config) && length(files_to_compile) > 1
        println("  🔄 Using $(Threads.nthreads()) threads")

        results = Vector{Tuple{String,Bool,Int}}(undef, length(files_to_compile))
        Threads.@threads for i in 1:length(files_to_compile)
            results[i] = compile_single_to_ir(config, files_to_compile[i])
        end

        # Check for failures
        failures = [(file, res) for (file, res) in zip(files_to_compile, results) if !res[2]]
        if !isempty(failures)
            println("  ❌ $(length(failures)) compilation failures")
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
    println("🔗 Linking and optimizing IR...")

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

    println("  ✅ Linked $(length(ir_files)) IR files → $(basename(linked_ir))")

    # Optimize if requested
    opt_level = config.link.optimization_level
    if opt_level != "0"
        println("  🚀 Optimizing (O$opt_level)...")

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
            println("  ✅ Optimized")
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
    println("📦 Creating shared library...")

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
    println("  ✅ Created: $lib_name ($size_mb MB)")

    return lib_path
end

"""
Create executable from LLVM IR.
Returns path to executable file.
"""
function create_executable(config::RepliBuildConfig, ir_file::String, exe_name::String,
                          link_libraries::Vector{String}=String[],
                          lib_dirs::Vector{String}=String[])
    println("🔨 Creating executable...")

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
    println("  ✅ Created: $exe_name ($size_kb KB)")

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
    println("🏗️  RepliBuild Compiler")
    println("="^70)
    println("📦 Project: $(config.project.name)")
    println("📁 Root:    $(config.project.root)")
    println("="^70)
    println()

    start_time = time()

    # Get source files (from config or discovery)
    cpp_files = get_source_files(config)

    if isempty(cpp_files)
        @warn "No source files found in config"
        println("💡 Run discover() first to find C++ sources")
        return nothing
    end

    println("📝 Source files: $(length(cpp_files))")
    println("🔧 Compiler flags: $(join(get_compile_flags(config), " "))")
    println("📂 Include dirs: $(length(get_include_dirs(config)))")
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
    println("✅ Build successful ($elapsed seconds)")
    println("📦 Binary: $binary_path")
    println("📝 Metadata: $metadata_path")
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
# DWARF DEBUG INFO PARSING - Perfect Type Extraction
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
Extract return types from DWARF debug info.
Returns: Dict{mangled_name => {c_type, julia_type, size}}
"""
function extract_dwarf_return_types(binary_path::String)::Dict{String,Dict{String,Any}}
    println("   🔍 Parsing DWARF debug info...")

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
    type_refs = Dict{String,String}()  # offset => type_name

    # First pass: Build type reference table
    for line in split(output, '\n')
        line = strip(line)

        # Extract base type definitions (DW_TAG_base_type)
        # Example: <1><27>: Abbrev Number: 2 (DW_TAG_base_type)
        if contains(line, "DW_TAG_base_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = "unknown"
            end
        end

        # Extract type name
        # Example: <28>   DW_AT_name        : (indexed string: 0x3): int
        if contains(line, "DW_AT_name") && haskey(type_refs, get(type_refs, "last_offset", ""))
            # Extract just the type name after the last colon
            name_match = match(r":\s*([^:]+)\s*$", line)
            if !isnothing(name_match)
                type_name = strip(name_match.captures[1])
                type_refs[type_refs["last_offset"]] = type_name
            end
        end

        # Track last seen offset
        offset_match = match(r"<\d+><([^>]+)>", line)
        if !isnothing(offset_match)
            type_refs["last_offset"] = "0x" * offset_match.captures[1]
        end
    end

    # Second pass: Extract function return types
    current_function_offset = nothing
    current_function_name = nothing
    current_function_linkage = nothing

    for line in split(output, '\n')
        line = strip(line)

        # Detect function start (DW_TAG_subprogram)
        if contains(line, "DW_TAG_subprogram")
            # Before starting new function, check if previous function had no return type (void)
            if !isnothing(current_function_offset)
                function_key = !isnothing(current_function_linkage) ? current_function_linkage : current_function_name
                if !isnothing(function_key) && !haskey(return_types, function_key)
                    # No DW_AT_type found = void return
                    return_types[function_key] = Dict(
                        "c_type" => "void",
                        "julia_type" => "Cvoid",
                        "size" => 0
                    )
                end
            end

            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_function_offset = "0x" * offset_match.captures[1]
                current_function_name = nothing
                current_function_linkage = nothing
            end
        end

        # Extract function name (for C functions without mangling)
        # Example: <2f>   DW_AT_name        : (indexed string: 0xc): test_sin
        if contains(line, "DW_AT_name") && !isnothing(current_function_offset) && isnothing(current_function_name)
            name_match = match(r":\s*([^:]+)\s*$", line)
            if !isnothing(name_match)
                current_function_name = strip(name_match.captures[1])
            end
        end

        # Extract linkage name (mangled name for C++ functions)
        # Example: <5c>   DW_AT_linkage_name: (indexed string: 0x8): _ZN10Calculator5powerEdd
        if contains(line, "DW_AT_linkage_name") && !isnothing(current_function_offset)
            # Extract just the mangled name after the last colon
            linkage_match = match(r":\s*([^:\s]+)\s*$", line)
            if !isnothing(linkage_match)
                current_function_linkage = strip(linkage_match.captures[1])
            end
        end

        # Extract function return type reference
        if contains(line, "DW_AT_type") && !isnothing(current_function_offset)
            type_match = match(r"<(0x[^>]+)>", line)
            if !isnothing(type_match)
                type_ref = type_match.captures[1]
                c_type = get(type_refs, type_ref, "unknown")

                # Map to Julia type using comprehensive mapping
                julia_type = dwarf_type_to_julia(c_type)
                type_size = get_type_size(c_type)

                # Use linkage name (C++) or fall back to function name (C)
                function_key = !isnothing(current_function_linkage) ? current_function_linkage : current_function_name

                # Store if we have a function identifier
                if !isnothing(function_key)
                    return_types[function_key] = Dict(
                        "c_type" => c_type,
                        "julia_type" => julia_type,
                        "size" => type_size
                    )
                end

                current_function_offset = nothing  # Done with this function
            end
        end
    end

    # Handle last function in file (might be void)
    if !isnothing(current_function_offset)
        function_key = !isnothing(current_function_linkage) ? current_function_linkage : current_function_name
        if !isnothing(function_key) && !haskey(return_types, function_key)
            # No DW_AT_type found = void return
            return_types[function_key] = Dict(
                "c_type" => "void",
                "julia_type" => "Cvoid",
                "size" => 0
            )
        end
    end

    if !isempty(return_types)
        println("   ✅ Extracted $(length(return_types)) return types from DWARF")
    else
        println("   ⚠️  No DWARF return type info found (compile with -g flag)")
    end

    return return_types
end

"""
Extract compilation metadata from source files and binary.
This is the core of automatic wrapper generation!
"""
function extract_compilation_metadata(config::RepliBuildConfig, source_files::Vector{String},
                                      binary_path::String)::Dict{String,Any}
    println("📊 Extracting compilation metadata...")

    # Extract symbols from compiled binary
    symbols = extract_symbols_from_binary(binary_path)
    println("   Found $(length(symbols)) exported symbols")

    # Extract return types from DWARF debug info (if available)
    dwarf_return_types = extract_dwarf_return_types(binary_path)

    # Parse function signatures (basic type inference from symbol names)
    functions = parse_function_signatures(symbols)

    # Merge DWARF return types into function metadata (overrides inference)
    for func in functions
        mangled = func["mangled"]
        if haskey(dwarf_return_types, mangled)
            func["return_type"] = dwarf_return_types[mangled]
            func["return_type_source"] = "dwarf"
        else
            func["return_type_source"] = "inferred"
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
function parse_function_signatures(symbols::Vector{Dict{String,Any}})::Vector{Dict{String,Any}}
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
            "parameters" => parse_parameters(demangled),
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
function parse_parameters(demangled::String)::Vector{Dict{String,Any}}
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

        julia_type = cpp_to_julia_type(ptype)

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
"""
function cpp_to_julia_type(cpp_type::AbstractString)::String
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
        "const char*" => "Cstring"
    )

    # Handle pointers
    if endswith(cpp_type, "*")
        return "Ptr{Cvoid}"
    end

    # Handle references
    if endswith(cpp_type, "&")
        base_type = strip(cpp_type[1:end-1])
        return "Ref{$(cpp_to_julia_type(base_type))}"
    end

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

    println("   ✅ Saved metadata: $metadata_path")
    return metadata_path
end

end # module Compiler
