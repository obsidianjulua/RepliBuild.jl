#!/usr/bin/env julia
"""
Compilation Daemon Server - Parallel C++ → IR → Binary with aggressive caching

Start with: julia --project=.. -p 4 compilation_daemon.jl
Port: 3003

Features:
- Parallel compilation with Distributed.@spawn
- Cached IR files (only recompile changed sources)
- Persistent LLVM environment (no reload overhead)
- Build queue with dependency ordering
- Error handler integration for intelligent retry
"""

using DaemonMode
using Distributed
using Dates

# Add workers if not already added
if nprocs() == 1
    addprocs(4)  # Add 4 worker processes
end

# Load project on all workers
@everywhere push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))

@everywhere using RepliBuild
@everywhere using RepliBuild.ConfigurationManager
@everywhere using RepliBuild.LLVMEnvironment
@everywhere using RepliBuild.BuildBridge
@everywhere using RepliBuild.LLVMake

const PORT = 3003

# ============================================================================
# PERSISTENT CACHES
# ============================================================================

const IR_CACHE = Dict{String, Tuple{String, Float64}}()  # source_path => (ir_path, mtime)
const BINARY_CACHE = Dict{String, Tuple{String, Float64}}()  # project_hash => (binary_path, mtime)
const BUILD_STATS = Dict{String, Any}()  # Statistics tracking

"""
Compute hash of source file for cache invalidation
"""
function source_hash(path::String)
    if !isfile(path)
        return ""
    end
    return string(hash((path, mtime(path))), base=16)
end

"""
Check if IR cache is valid for source file
"""
function is_ir_cached(source_path::String, ir_output_dir::String)
    if !haskey(IR_CACHE, source_path)
        return false, ""
    end

    ir_path, cached_mtime = IR_CACHE[source_path]

    # Check if source file has been modified
    if !isfile(source_path)
        return false, ""
    end

    current_mtime = mtime(source_path)
    if current_mtime > cached_mtime
        return false, ""
    end

    # Check if IR file exists
    if !isfile(ir_path)
        return false, ""
    end

    return true, ir_path
end

# ============================================================================
# COMPILATION FUNCTIONS
# ============================================================================

@everywhere function compile_source_to_ir(source_path::String, output_dir::String,
                                          include_dirs::Vector{String}, flags::Vector{String},
                                          clang_path::String)
    try
        # Ensure output directory exists
        mkpath(output_dir)

        # Generate output path
        source_name = splitext(basename(source_path))[1]
        ir_path = joinpath(output_dir, "$source_name.ll")

        # Build clang command
        args = vcat(
            ["-S", "-emit-llvm"],
            flags,
            ["-I$dir" for dir in include_dirs],
            ["-o", ir_path],
            [source_path]
        )

        # Execute compilation
        output, exitcode = BuildBridge.execute(clang_path, args, use_llvm_env=true)

        if exitcode != 0
            return Dict(
                :success => false,
                :source => source_path,
                :error => output,
                :exitcode => exitcode
            )
        end

        return Dict(
            :success => true,
            :source => source_path,
            :ir_path => ir_path,
            :cached => false
        )

    catch e
        return Dict(
            :success => false,
            :source => source_path,
            :error => string(e),
            :exception => true
        )
    end
end

"""
Compile all sources to IR in parallel
"""
function compile_parallel(args::Dict)
    config_path = get(args, "config", "replibuild.toml")
    force = get(args, "force", false)

    println("[COMPILE] Starting parallel compilation")

    try
        # Load configuration
        config = ConfigurationManager.load_config(config_path)

        # Get source files
        source_files = ConfigurationManager.get_source_files(config)
        cpp_sources = get(source_files, "cpp_sources", String[])
        c_sources = get(source_files, "c_sources", String[])

        all_sources = vcat(
            [joinpath(config.project_root, f) for f in cpp_sources],
            [joinpath(config.project_root, f) for f in c_sources]
        )

        if isempty(all_sources)
            return Dict(
                :success => false,
                :error => "No source files to compile"
            )
        end

        # Get compilation settings
        output_dir = get(config.compile, "output_dir", "build/ir")
        output_dir = joinpath(config.project_root, output_dir)
        include_dirs = ConfigurationManager.get_include_dirs(config)
        flags = get(config.compile, "flags", ["-std=c++17", "-fPIC"])

        # Get clang path
        clang_path = get(config.llvm, "tools", Dict())["clang++"]

        println("[COMPILE] Compiling $(length(all_sources)) source files...")
        println("[COMPILE] Output: $output_dir")
        println("[COMPILE] Workers: $(nprocs())")

        # Check cache and filter sources that need compilation
        sources_to_compile = String[]
        cached_results = Dict[]

        for source in all_sources
            cached, ir_path = is_ir_cached(source, output_dir)

            if cached && !force
                println("[COMPILE] ✓ Cached: $(basename(source))")
                push!(cached_results, Dict(
                    :success => true,
                    :source => source,
                    :ir_path => ir_path,
                    :cached => true
                ))
            else
                push!(sources_to_compile, source)
            end
        end

        # Compile remaining sources in parallel
        compile_results = Dict[]

        if !isempty(sources_to_compile)
            println("[COMPILE] Compiling $(length(sources_to_compile)) files in parallel...")

            # Distribute compilation across workers
            futures = [@spawnat :any compile_source_to_ir(
                source, output_dir, include_dirs, flags, clang_path
            ) for source in sources_to_compile]

            # Collect results
            for (i, future) in enumerate(futures)
                result = fetch(future)
                push!(compile_results, result)

                if result[:success]
                    # Update cache
                    IR_CACHE[result[:source]] = (result[:ir_path], mtime(result[:source]))
                    println("[COMPILE] ✓ Compiled: $(basename(result[:source]))")
                else
                    println("[COMPILE] ✗ Failed: $(basename(result[:source]))")
                    println("    Error: $(result[:error])")
                end
            end
        end

        # Combine results
        all_results = vcat(cached_results, compile_results)

        # Count successes
        success_count = count(r -> r[:success], all_results)
        failed_count = length(all_results) - success_count
        cached_count = count(r -> get(r, :cached, false), all_results)

        overall_success = failed_count == 0

        println("[COMPILE] Compilation complete:")
        println("  ✓ Success: $success_count")
        println("  ✗ Failed:  $failed_count")
        println("  ⚡ Cached:  $cached_count")

        return Dict(
            :success => overall_success,
            :total => length(all_results),
            :success_count => success_count,
            :failed_count => failed_count,
            :cached_count => cached_count,
            :results => all_results,
            :ir_files => [r[:ir_path] for r in all_results if r[:success]]
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e),
            :stacktrace => sprint(showerror, e, catch_backtrace())
        )
    end
end

"""
Link IR files into single module
"""
function link_ir(args::Dict)
    ir_files = get(args, "ir_files", String[])
    output_path = get(args, "output", "")
    llvm_link_path = get(args, "llvm_link", "")

    println("[COMPILE] Linking $(length(ir_files)) IR files...")

    try
        if isempty(ir_files)
            return Dict(:success => false, :error => "No IR files to link")
        end

        if isempty(output_path)
            return Dict(:success => false, :error => "Output path required")
        end

        # Ensure output directory exists
        mkpath(dirname(output_path))

        # Build llvm-link command
        args_list = vcat(["-S", "-o", output_path], ir_files)

        # Execute linking
        output, exitcode = BuildBridge.execute(llvm_link_path, args_list, use_llvm_env=true)

        if exitcode != 0
            return Dict(
                :success => false,
                :error => output,
                :exitcode => exitcode
            )
        end

        println("[COMPILE] ✓ Linked IR: $output_path")

        return Dict(
            :success => true,
            :output => output_path
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e)
        )
    end
end

"""
Optimize IR with opt
"""
function optimize_ir(args::Dict)
    ir_path = get(args, "ir_path", "")
    output_path = get(args, "output", "")
    opt_level = get(args, "opt_level", "O2")
    opt_path = get(args, "opt", "")

    println("[COMPILE] Optimizing IR: $ir_path")

    try
        # Build opt command
        args_list = ["-$opt_level", "-S", "-o", output_path, ir_path]

        # Execute optimization
        output, exitcode = BuildBridge.execute(opt_path, args_list, use_llvm_env=true)

        if exitcode != 0
            return Dict(
                :success => false,
                :error => output,
                :exitcode => exitcode
            )
        end

        println("[COMPILE] ✓ Optimized: $output_path")

        return Dict(
            :success => true,
            :output => output_path
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e)
        )
    end
end

"""
Compile IR to native object file
"""
function compile_to_object(args::Dict)
    ir_path = get(args, "ir_path", "")
    output_path = get(args, "output", "")
    llc_path = get(args, "llc", "")

    println("[COMPILE] Compiling IR to object: $ir_path")

    try
        # Build llc command
        args_list = ["-filetype=obj", "-o", output_path, ir_path]

        # Execute compilation
        output, exitcode = BuildBridge.execute(llc_path, args_list, use_llvm_env=true)

        if exitcode != 0
            return Dict(
                :success => false,
                :error => output,
                :exitcode => exitcode
            )
        end

        println("[COMPILE] ✓ Object file: $output_path")

        return Dict(
            :success => true,
            :output => output_path
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e)
        )
    end
end

"""
Link object files into shared library
"""
function link_shared_library(args::Dict)
    object_files = get(args, "object_files", String[])
    output_path = get(args, "output", "")
    link_libraries = get(args, "libraries", String[])
    clang_path = get(args, "clang", "")

    println("[COMPILE] Linking shared library: $output_path")

    try
        # Build clang++ command for linking
        args_list = vcat(
            ["-shared", "-fPIC"],
            object_files,
            ["-o", output_path],
            ["-l$lib" for lib in link_libraries]
        )

        # Execute linking
        output, exitcode = BuildBridge.execute(clang_path, args_list, use_llvm_env=true)

        if exitcode != 0
            return Dict(
                :success => false,
                :error => output,
                :exitcode => exitcode
            )
        end

        println("[COMPILE] ✓ Shared library: $output_path")

        return Dict(
            :success => true,
            :output => output_path
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e)
        )
    end
end

"""
Full compilation pipeline: sources → IR → optimize → object → library
"""
function compile_full_pipeline(args::Dict)
    config_path = get(args, "config", "replibuild.toml")
    force = get(args, "force", false)

    println("[COMPILE] Running full compilation pipeline")

    try
        config = ConfigurationManager.load_config(config_path)

        # Stage 1: Compile sources to IR in parallel
        compile_result = compile_parallel(Dict("config" => config_path, "force" => force))
        if !compile_result[:success]
            return compile_result
        end

        ir_files = compile_result[:ir_files]

        # Stage 2: Link IR files
        linked_ir_path = joinpath(config.project_root, get(config.link, "output_dir", "build/linked"), "linked.ll")
        link_result = link_ir(Dict(
            "ir_files" => ir_files,
            "output" => linked_ir_path,
            "llvm_link" => config.llvm["tools"]["llvm-link"]
        ))

        if !link_result[:success]
            return link_result
        end

        # Stage 3: Optimize
        opt_level = get(config.link, "opt_level", "O2")
        optimized_ir_path = joinpath(dirname(linked_ir_path), "optimized.ll")

        opt_result = optimize_ir(Dict(
            "ir_path" => linked_ir_path,
            "output" => optimized_ir_path,
            "opt_level" => opt_level,
            "opt" => config.llvm["tools"]["opt"]
        ))

        if !opt_result[:success]
            return opt_result
        end

        # Stage 4: Compile to object
        object_path = joinpath(dirname(optimized_ir_path), "output.o")
        obj_result = compile_to_object(Dict(
            "ir_path" => optimized_ir_path,
            "output" => object_path,
            "llc" => config.llvm["tools"]["llc"]
        ))

        if !obj_result[:success]
            return obj_result
        end

        # Stage 5: Link shared library
        library_name = get(config.binary, "library_name", "lib$(lowercase(config.project_name)).so")
        library_path = joinpath(config.project_root, get(config.binary, "output_dir", "julia"), library_name)

        lib_result = link_shared_library(Dict(
            "object_files" => [object_path],
            "output" => library_path,
            "libraries" => get(config.binary, "link_libraries", String[]),
            "clang" => config.llvm["tools"]["clang++"]
        ))

        if !lib_result[:success]
            return lib_result
        end

        println("[COMPILE] ✅ Full pipeline complete!")
        println("[COMPILE] Library: $library_path")

        return Dict(
            :success => true,
            :library_path => library_path,
            :compile_stats => compile_result,
            :stages => Dict(
                "compile" => compile_result[:success],
                "link" => link_result[:success],
                "optimize" => opt_result[:success],
                "object" => obj_result[:success],
                "library" => lib_result[:success]
            )
        )

    catch e
        return Dict(
            :success => false,
            :error => string(e),
            :stacktrace => sprint(showerror, e, catch_backtrace())
        )
    end
end

"""
Clear compilation caches
"""
function clear_caches(args::Dict)
    println("[COMPILE] Clearing compilation caches...")

    empty!(IR_CACHE)
    empty!(BINARY_CACHE)

    return Dict(
        :success => true,
        :message => "Caches cleared"
    )
end

"""
Get cache statistics
"""
function cache_stats(args::Dict)
    return Dict(
        :success => true,
        :stats => Dict(
            "ir_files" => length(IR_CACHE),
            "binaries" => length(BINARY_CACHE),
            "workers" => nprocs()
        )
    )
end

# ============================================================================
# MAIN
# ============================================================================

"""
Main daemon server function
"""
function main()
    println("="^70)
    println("RepliBuild Compilation Daemon Server")
    println("Port: $PORT")
    println("Workers: $(nprocs())")
    println("="^70)

    println()
    println("Available Functions:")
    println("  • compile_parallel(config, force=false) - Parallel C++ → IR")
    println("  • link_ir(ir_files, output, llvm_link)")
    println("  • optimize_ir(ir_path, output, opt_level='O2', opt)")
    println("  • compile_to_object(ir_path, output, llc)")
    println("  • link_shared_library(object_files, output, libraries, clang)")
    println("  • compile_full_pipeline(config, force=false) - Complete build")
    println("  • cache_stats()")
    println("  • clear_caches()")
    println()
    println("Ready to accept compilation requests...")
    println("="^70)

    # Start the daemon server
    serve(PORT)
end

# Start the daemon if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
