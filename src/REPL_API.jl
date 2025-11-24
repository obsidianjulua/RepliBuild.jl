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
import ..Wrapper

# Import from Compiler and ConfigurationManager modules
import ..Compiler: compile_to_ir, link_optimize_ir, create_library
import ..ConfigurationManager: RepliBuildConfig, load_config, get_output_path,
                                get_library_name, get_module_name

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

    # Load config
    config = load_config("replibuild.toml")

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

    config = load_config("replibuild.toml")
    config = ConfigurationManager.merge_compile_flags(config, flags)

    ir_files = compile_to_ir(config, source_files)

    println("✅ Compiled $(length(ir_files)) files to IR")
    return ir_files
end

"""
    rwrap(lib_path; tier=:auto, headers=String[], generate_tests=false, generate_docs=true)

Generate Julia wrappers for a compiled library using the new 3-tier wrapper system.

# Arguments
- `lib_path::String`: Path to compiled library (.so, .dylib, .dll)
- `tier::Symbol`: Wrapper tier - `:auto`, `:basic`, `:advanced`, or `:introspective`
- `headers::Vector{String}`: Header files (for advanced/introspective tiers)
- `generate_tests::Bool`: Generate test file (default: false)
- `generate_docs::Bool`: Generate documentation (default: true)

# Tier Descriptions
- `:auto` - Automatically detect best tier based on available information
- `:basic` - Symbol-only extraction (~40% quality, no headers needed)
- `:advanced` - Header-aware with Clang.jl (~85% quality, requires headers)
- `:introspective` - Metadata-rich from compilation (~95% quality, future)

# Examples
```julia
rwrap("libmyproject.so")                                    # Auto-detect tier
rwrap("libexternal.so", tier=:basic)                       # Basic symbol wrapping
rwrap("libapp.so", tier=:advanced, headers=["app.h"])      # Advanced with headers
rwrap("lib.so", headers=["lib.h"], generate_tests=true)    # With tests
```
"""
function rwrap(lib_path::String=""; tier::Symbol=:auto,
               headers::Vector{String}=String[],
               generate_tests::Bool=false,
               generate_docs::Bool=true)

    # If no lib_path provided, try to find one in output directory
    if isempty(lib_path)
        config = load_config("replibuild.toml")
        output_dir = get_output_path(config)
        lib_name = get_library_name(config)
        lib_path = joinpath(output_dir, lib_name)

        if !isfile(lib_path)
            error("No library found at $lib_path. Build project first with rbuild()")
        end

        println("📦 Using: $lib_path")
    end

    if !isfile(lib_path)
        error("Library not found: $lib_path")
    end

    # Load or create config
    config = if isfile("replibuild.toml")
        load_config("replibuild.toml")
    else
        # Create minimal config for external library
        ConfigurationManager.create_default_config(
            "replibuild.toml",
            project_name=basename(dirname(lib_path))
        )
    end

    # Convert tier symbol to WrapperTier enum if not :auto
    wrapper_tier = if tier == :auto
        nothing  # Let Wrapper module auto-detect
    elseif tier == :basic
        Wrapper.TIER_BASIC
    elseif tier == :advanced
        Wrapper.TIER_ADVANCED
    elseif tier == :introspective
        Wrapper.TIER_INTROSPECTIVE
    else
        error("Unknown tier: $tier. Use :auto, :basic, :advanced, or :introspective")
    end

    # Generate wrapper using new Wrapper module
    output_file = Wrapper.wrap_library(
        config,
        lib_path,
        headers=headers,
        tier=wrapper_tier,
        generate_tests=generate_tests,
        generate_docs=generate_docs
    )

    return output_file
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
    """)
end

end # module REPL_API
