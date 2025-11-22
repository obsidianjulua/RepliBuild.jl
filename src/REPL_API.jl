#!/usr/bin/env julia
# REPL_API.jl - User-friendly REPL commands for RepliBuild
# Provides convenient shortcuts for interactive use

"""
REPL-friendly API for RepliBuild

Quick commands:
- `rbuild()` - Build current project
- `rdiscover()` - Discover project structure
- `rclean()` - Clean build artifacts
- `rinfo()` - Show project information
- `rwrap()` - Generate Julia wrappers
"""
module REPL_API

# Import parent RepliBuild functionality
import ..RepliBuild
import ..Discovery
import ..ClangJLBridge
import ..WorkspaceBuilder

# Bridge_LLVM functions are defined at parent module level, not as separate module
import ..BridgeCompilerConfig
import ..compile_to_ir
import ..link_optimize_ir
import ..create_library

export rbuild, rdiscover, rclean, rinfo, rwrap
export rbuild_fast, rcompile, rparallel
export rthreads, rcache_status

# ============================================================================
# CORE COMMANDS
# ============================================================================

"""
    rbuild(path="."; parallel=true, clean=false)

Quick build command - discovers and builds C++ project.

# Arguments
- `path::String`: Project directory (default: current directory)
- `parallel::Bool`: Enable parallel compilation (default: true)
- `clean::Bool`: Clean before building (default: false)

# Examples
```julia
rbuild()                    # Build current project
rbuild("myproject")         # Build specific project
rbuild(clean=true)          # Clean build
rbuild(parallel=false)      # Sequential build
```
"""
function rbuild(path::String="."; parallel::Bool=true, clean::Bool=false)
    if clean
        rclean(path)
    end

    RepliBuild.build(path; parallel=parallel)
end

"""
    rdiscover(path="."; force=false)

Discover C++ project structure and generate replibuild.toml.

# Arguments
- `path::String`: Project directory (default: current directory)
- `force::Bool`: Force re-discovery (default: false)

# Examples
```julia
rdiscover()                 # Discover current project
rdiscover("myproject")      # Discover specific project
rdiscover(force=true)       # Force re-scan
```
"""
function rdiscover(path::String="."; force::Bool=false)
    RepliBuild.discover(path; force=force)
end

"""
    rclean(path=".")

Clean build artifacts and caches.

# Examples
```julia
rclean()                    # Clean current project
rclean("myproject")         # Clean specific project
```
"""
function rclean(path::String=".")
    RepliBuild.clean(path)
end

"""
    rinfo(path=".")

Display project information.

# Examples
```julia
rinfo()                     # Show current project info
rinfo("myproject")          # Show specific project info
```
"""
function rinfo(path::String=".")
    RepliBuild.info(path)
end

# ============================================================================
# ADVANCED COMMANDS
# ============================================================================

"""
    rbuild_fast(sources; output="libproject.so", flags=["-std=c++17", "-O2"], parallel=true)

Fast compilation without discovery - for quick iterations.

# Arguments
- `sources`: Vector of source files or glob pattern
- `output::String`: Output library name
- `flags::Vector{String}`: Compiler flags
- `parallel::Bool`: Enable parallel compilation

# Examples
```julia
rbuild_fast(["src/main.cpp", "src/util.cpp"])
rbuild_fast("src/*.cpp", output="mylib.so")
rbuild_fast(["src/app.cpp"], flags=["-std=c++20", "-O3"])
```
"""
function rbuild_fast(sources; output::String="libproject.so",
                     flags::Vector{String}=["-std=c++17", "-O2"],
                     parallel::Bool=true)

    # Convert glob pattern to files if string
    source_files = if isa(sources, String)
        # Simple glob expansion
        glob_to_files(sources)
    else
        sources
    end

    println("⚡ Fast compilation: $(length(source_files)) files")

    # Create minimal config
    config = BridgeCompilerConfig("replibuild.toml")
    config.parallel = parallel
    config.compile_flags = flags

    # Compile
    ir_files = compile_to_ir(config, source_files)

    # Link
    output_name = replace(output, ".so" => "")
    linked_ir = link_optimize_ir(config, ir_files, output_name)

    # Create library
    create_library(config, linked_ir, output)

    println("✅ Built: $output")
    return output
end

"""
    rcompile(files...; flags=["-std=c++17"])

Compile specific files without linking.

# Examples
```julia
rcompile("main.cpp", "util.cpp")
rcompile("src/app.cpp", flags=["-std=c++20", "-O3"])
```
"""
function rcompile(files...; flags::Vector{String}=["-std=c++17"])
    source_files = collect(files)

    config = BridgeCompilerConfig("replibuild.toml")
    config.compile_flags = flags

    ir_files = compile_to_ir(config, source_files)

    println("✅ Compiled $(length(ir_files)) files to IR")
    return ir_files
end

"""
    rwrap(lib_path; style=:auto, headers=String[])

Generate Julia wrappers for a compiled library.

# Arguments
- `lib_path::String`: Path to compiled library (.so, .dylib, .dll)
- `style::Symbol`: Wrapper style - `:auto`, `:clang`, or `:binary`
- `headers::Vector{String}`: Header files (for :clang style)

# Examples
```julia
rwrap("libmyproject.so")                    # Auto-detect
rwrap("libexternal.so", style=:binary)      # Binary-only wrapping
rwrap("libapp.so", style=:clang, headers=["app.h"])
```
"""
function rwrap(lib_path::String; style::Symbol=:auto, headers::Vector{String}=String[])
    if !isfile(lib_path)
        error("Library not found: $lib_path")
    end

    # Auto-detect style
    if style == :auto
        # Check if headers available
        style = isempty(headers) ? :binary : :clang
    end

    if style == :clang
        if isempty(headers)
            error("Headers required for :clang style wrapping")
        end

        println("📝 Generating wrappers with Clang.jl...")

        config = Dict(
            "project" => Dict("name" => basename(dirname(lib_path))),
            "compile" => Dict("include_dirs" => String[]),
        )

        return ClangJLBridge.generate_bindings_clangjl(config, lib_path, headers)
    else  # :binary
        println("📝 Generating wrappers from binary symbols...")

        @warn "Binary-only wrapping not yet implemented. Use style=:clang with headers instead."
        println("  💡 Tip: Provide headers for better type-aware bindings:")
        println("     rwrap(\"$lib_path\", style=:clang, headers=[\"myheader.h\"])")
        return nothing
    end
end

"""
    rparallel(enabled=true)

Enable or disable parallel compilation globally.

# Examples
```julia
rparallel(true)    # Enable parallel builds
rparallel(false)   # Disable parallel builds
```
"""
function rparallel(enabled::Bool=true)
    ENV["REPLIBUILD_PARALLEL"] = string(enabled)
    println(enabled ? "✅ Parallel compilation enabled" : "⚠️  Parallel compilation disabled")
end

"""
    rthreads()

Show number of available threads for parallel compilation.

# Examples
```julia
rthreads()  # Display thread count
```
"""
function rthreads()
    println("💻 Available threads: $(Threads.nthreads())")
    println("   CPU cores: $(Sys.CPU_THREADS)")

    if Threads.nthreads() == 1
        println("⚠️  Running single-threaded. Start Julia with: julia --threads=auto")
    end

    return Threads.nthreads()
end

"""
    rcache_status(path=".")

Show build cache status.

# Examples
```julia
rcache_status()             # Current project cache
rcache_status("myproject")  # Specific project
```
"""
function rcache_status(path::String=".")
    cache_dir = joinpath(path, ".bridge_cache")

    if !isdir(cache_dir)
        println("❌ No cache directory found")
        return
    end

    # Count cached files
    ir_files = filter(f -> endswith(f, ".ll"), readdir(cache_dir, join=true))

    println("📊 Build Cache Status")
    println("   Location: $cache_dir")
    println("   Cached IR files: $(length(ir_files))")

    if !isempty(ir_files)
        total_size = sum(filesize.(ir_files))
        println("   Total size: $(round(total_size / 1024^2, digits=2)) MB")

        # Show age of oldest/newest
        mtimes = mtime.(ir_files)
        oldest = minimum(mtimes)
        newest = maximum(mtimes)

        println("   Oldest: $(Dates.unix2datetime(oldest))")
        println("   Newest: $(Dates.unix2datetime(newest))")
    end
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
Simple glob pattern expansion for file patterns.
"""
function glob_to_files(pattern::String)
    # Simple implementation - just handle *.cpp style patterns
    dir = dirname(pattern)
    if isempty(dir)
        dir = "."
    end

    base_pattern = basename(pattern)
    regex_pattern = replace(base_pattern, "*" => ".*", "?" => ".")
    regex = Regex("^" * regex_pattern * "\$")

    files = String[]
    for file in readdir(dir)
        if occursin(regex, file)
            push!(files, joinpath(dir, file))
        end
    end

    return files
end

# ============================================================================
# HELP MESSAGE
# ============================================================================

function __init__()
    println("""
    📦 RepliBuild REPL API loaded!

    Quick commands:
      rbuild()      - Build project
      rdiscover()   - Scan project
      rclean()      - Clean artifacts
      rinfo()       - Show info
      rwrap()       - Generate wrappers
      rthreads()    - Check threads

    Advanced:
      rbuild_fast(["src/main.cpp"])
      rcompile("file.cpp")
      rcache_status()

    Tip: Start Julia with --threads=auto for parallel builds!
    """)
end

end # module REPL_API
