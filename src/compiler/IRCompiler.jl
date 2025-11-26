# IRCompiler.jl - C++ to LLVM IR Compilation and Linking
# Handles compilation, optimization, and binary creation

module IRCompiler

import ...BuildBridge
import ...ConfigurationManager: RepliBuildConfig, get_source_files, get_include_dirs,
                                get_compile_flags, get_build_path, get_output_path,
                                get_library_name, is_parallel_enabled, is_cache_enabled

export compile_to_ir, link_optimize_ir, create_library, create_executable, needs_recompile

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
    cached_ir_files = String[]

    for cpp_file in cpp_files
        base_name = splitext(basename(cpp_file))[1]
        ir_file = joinpath(build_dir, base_name * ".ll")

        if needs_recompile(cpp_file, ir_file, is_cache_enabled(config))
            push!(files_to_compile, cpp_file)
        else
            push!(cached_ir_files, ir_file)
            println("  Cached: $(basename(cpp_file))")
        end
    end

    if isempty(files_to_compile)
        println("All files cached ($(length(cached_ir_files)) files)")
        return cached_ir_files
    end

    println("Compiling $(length(files_to_compile)) files...")

    # Compile files (parallel if enabled)
    results = if is_parallel_enabled(config) && length(files_to_compile) > 1
        println("  Using parallel compilation ($(Threads.nthreads()) threads)")
        # Use @threads for parallel execution
        result_vec = Vector{Tuple{String,Bool,Int}}(undef, length(files_to_compile))
        Threads.@threads for i in 1:length(files_to_compile)
            result_vec[i] = compile_single_to_ir(config, files_to_compile[i])
        end
        result_vec
    else
        [compile_single_to_ir(config, f) for f in files_to_compile]
    end

    # Collect successful IR files
    ir_files = vcat(cached_ir_files, [r[1] for r in results if r[2]])

    # Check for failures
    failures = [files_to_compile[i] for (i, r) in enumerate(results) if !r[2]]
    if !isempty(failures)
        error("Compilation failed for: $(join(failures, ", "))")
    end

    println("Compilation complete: $(length(ir_files)) IR files")
    return ir_files
end

# =============================================================================
# LINKING: Multiple IR → Single IR
# =============================================================================

"""
Link multiple LLVM IR files and optionally optimize.
Returns path to linked (and possibly optimized) IR file.
"""
function link_optimize_ir(config::RepliBuildConfig, ir_files::Vector{String}, output_name::String)
    println("Linking IR files...")

    if isempty(ir_files)
        error("No IR files to link")
    end

    # Build directory
    build_dir = get_build_path(config)

    # Output file path
    linked_ir = joinpath(build_dir, output_name * ".ll")

    # Link IR files using llvm-link
    link_args = vcat(ir_files, ["-S", "-o", linked_ir])
    (output, exitcode) = BuildBridge.execute("llvm-link", link_args)

    if exitcode != 0
        error("Linking failed: $output")
    end

    if !isfile(linked_ir)
        error("Linked IR file not created")
    end

    # Get linked file size
    size_kb = round(filesize(linked_ir) / 1024, digits=1)
    println("Linked: $output_name.ll ($size_kb KB)")

    # Optimize if optimization level is specified
    if !isempty(config.link.optimization_level)
        println("Optimizing IR...")

        optimized_ir = joinpath(build_dir, output_name * "_opt.ll")

        # Build optimization arguments
        opt_args = ["-S", config.link.optimization_level, linked_ir, "-o", optimized_ir]

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
                          link_flags::Vector{String}=String[])
    println("Creating executable...")

    if !isfile(ir_file)
        error("IR file not found: $ir_file")
    end

    # Output path
    output_dir = get_output_path(config)
    mkpath(output_dir)
    exe_path = joinpath(output_dir, exe_name)

    # Compile IR to executable
    cmd_args = vcat(
        [ir_file, "-o", exe_path],
        link_flags,
        ["-l$lib" for lib in link_libraries],
        ["-l$lib" for lib in config.link.link_libraries]
    )

    (output, exitcode) = BuildBridge.execute("clang++", cmd_args)

    if exitcode != 0
        error("Executable creation failed: $output")
    end

    if !isfile(exe_path)
        error("Executable file not created: $exe_path")
    end

    # Get file size
    size_kb = round(filesize(exe_path) / 1024, digits=1)
    println("Created: $exe_name ($size_kb KB)")

    return exe_path
end

end # module IRCompiler
