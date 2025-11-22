#!/usr/bin/env julia
# Compiler.jl - C++ to LLVM IR compilation and linking
# Uses centralized RepliBuildConfig (no TOML parsing here)

module Compiler

using Dates

# Import from parent RepliBuild module
import ..ConfigurationManager: RepliBuildConfig, get_source_files, get_include_dirs,
                                get_compile_flags, get_build_path, get_output_path,
                                get_library_name, is_parallel_enabled, is_cache_enabled
import ..BuildBridge
import ..LLVMEnvironment

export compile_to_ir, link_optimize_ir, create_library, create_executable, compile_project

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

    println("  ✅ Compiled $(length(files_to_compile)) files")

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

    println()
    println("="^70)
    println("✅ Build successful ($elapsed seconds)")
    println("📦 Output: $binary_path")
    println("="^70)

    return binary_path
end

end # module Compiler
