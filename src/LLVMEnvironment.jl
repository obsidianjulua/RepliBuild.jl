#!/usr/bin/env julia
# LLVMEnvironment.jl - Complete LLVM Toolchain Environment Manager
# Isolates and manages the RepliBuild-specific LLVM 20.1.2 installation
# Ensures all RepliBuild operations use the local toolchain without polluting system environment

module LLVMEnvironment

using Pkg

# Conditional import - will try to use LLVM_full_assert_jll if available
const LLVM_JLL_AVAILABLE = Ref{Bool}(false)

function __init__()
    try
        @eval using LLVM_full_assert_jll
        LLVM_JLL_AVAILABLE[] = true
    catch
        LLVM_JLL_AVAILABLE[] = false
    end
end

"""
LLVM toolchain configuration and paths
Represents the complete isolated LLVM ecosystem
Can source from either in-tree LLVM or LLVM_full_assert_jll
"""
struct LLVMToolchain
    # Root paths
    root::String
    bin_dir::String
    lib_dir::String
    include_dir::String
    libexec_dir::String
    share_dir::String

    # Version info
    version::String
    version_major::Int
    version_minor::Int
    version_patch::Int

    # Core tools
    tools::Dict{String,String}

    # Libraries
    libraries::Dict{String,String}

    # Compiler flags
    cxxflags::Vector{String}
    ldflags::Vector{String}
    libs::String

    # Environment variables
    env_vars::Dict{String,String}

    # Isolated environment flag
    isolated::Bool

    # Toolchain source: "intree" or "jll"
    source::String
end

"""
Global LLVM toolchain instance (lazy initialization)
"""
const GLOBAL_LLVM_TOOLCHAIN = Ref{Union{LLVMToolchain,Nothing}}(nothing)

"""
    get_replibuild_llvm_root() -> String

Get the absolute path to RepliBuild's in-tree LLVM installation.
"""
function get_replibuild_llvm_root()
    # __DIR__ points to /home/grim/.julia/julia/RepliBuild/src
    # We need to go up one level to /home/grim/.julia/julia/RepliBuild
    # Then access LLVM subdirectory
    replibuild_dir = dirname(@__DIR__)  # Go from src/ to RepliBuild/
    llvm_root = joinpath(replibuild_dir, "LLVM")

    if !isdir(llvm_root)
        error("RepliBuild LLVM installation not found at: $llvm_root\n" *
              "Expected location: /home/grim/.julia/julia/RepliBuild/LLVM")
    end

    return abspath(llvm_root)
end

"""
    get_jll_llvm_root() -> Union{String,Nothing}

Get the absolute path to LLVM_full_assert_jll installation if available.
Returns nothing if LLVM_full_assert_jll is not installed.
"""
function get_jll_llvm_root()
    if !LLVM_JLL_AVAILABLE[]
        return nothing
    end

    try
        jll = @eval LLVM_full_assert_jll

        # Prefer artifact system over older fields
        if hasproperty(jll, :artifacts)
            return first(values(jll.artifacts))
        elseif hasproperty(jll, :artifact_dir)
            return jll.artifact_dir
        elseif hasproperty(jll, :artifact_path)
            return jll.artifact_path
        else
            @warn "LLVM_full_assert_jll has no exposed artifact path"
            return nothing
        end
    catch e
        @warn "Failed to read LLVM_full_assert_jll root: $e"
        return nothing
    end
end

"""
get_llvm_root(source::Symbol=:auto; config=nothing) -> Tuple{String,String}

Determine which LLVM root to use based on priority:
1. Project TOML config (if present and valid)
2. LLVM_full_assert_jll (if installed)
3. In-tree LLVM (if exists)
4. System LLVM (search /usr, /usr/local, etc.)
"""

function get_llvm_root(source::Symbol=:auto; config=nothing)
    # 1. User TOML override
    if config !== nothing && haskey(config, :llvm) && haskey(config.llvm, "root")
        user_root = config.llvm["root"]
        if !isempty(user_root) && isdir(user_root)
            src = get(config.llvm, "source", "custom")
            @info "Using LLVM root from TOML: $user_root"
            return (abspath(user_root), src)
        else
            @warn "Invalid LLVM root in TOML: $user_root — falling back"
        end
    end

    # 2. Force or prefer JLL
    if source == :jll || source == :auto
        jll_root = get_jll_llvm_root()
        if jll_root !== nothing && isdir(jll_root)
            return (abspath(jll_root), "jll")
        elseif source == :jll
            error("LLVM_full_assert_jll requested but unavailable")
        end
    end

    # 3. In-tree fallback
    try
        intree_root = get_replibuild_llvm_root()
        @info "Using in-tree LLVM toolchain"
        return (intree_root, "intree")
    catch e
        @warn "In-tree LLVM not found: $e"
    end

    # 4. System LLVM as last resort
    @info "Searching for system LLVM installation..."
    system_root = find_system_llvm()
    if system_root !== nothing
        @info "Using system LLVM toolchain at: $system_root"
        return (system_root, "system")
    end

    # No LLVM found - throw error
    error("LLVM Toolchain not found. RepliBuild requires an LLVM installation with clang++. Please install LLVM or set LLVM_CONFIG environment variable.")
end

"""
    find_system_llvm() -> Union{String,Nothing}

Search common system paths for LLVM installation.
Returns the root directory if found, nothing otherwise.
"""
function find_system_llvm()
    # Common LLVM installation prefixes
    search_paths = [
        "/usr",
        "/usr/local",
        "/opt/llvm",
        "/opt/homebrew",  # macOS
        "/usr/lib/llvm-20",
        "/usr/lib/llvm-19",
        "/usr/lib/llvm-18",
        "/usr/lib/llvm-17",
        "/usr/lib/llvm-16",
        "/usr/lib/llvm-15",
    ]

    for prefix in search_paths
        # Check for clang++ and llvm-config in bin directory
        bin_dir = joinpath(prefix, "bin")
        if isdir(bin_dir)
            clang_path = joinpath(bin_dir, "clang++")
            llvm_config_path = joinpath(bin_dir, "llvm-config")

            if isfile(clang_path) && isfile(llvm_config_path)
                # Verify it's a working LLVM by checking lib and include dirs
                lib_dir = joinpath(prefix, "lib")
                include_dir = joinpath(prefix, "include")

                if isdir(lib_dir) && isdir(include_dir)
                    return prefix
                end
            end
        end
    end

    return nothing
end

"""
    discover_llvm_tools(llvm_root::String, source::String="intree") -> Dict{String,String}

Discover all LLVM/Clang tools in the toolchain.
Returns a dictionary mapping tool names to absolute paths.

# Arguments
- `llvm_root`: Root directory of LLVM installation
- `source`: "intree" uses tools/, "jll"/"system" use bin/
"""
function discover_llvm_tools(llvm_root::String, source::String="intree")
    tools = Dict{String,String}()
    # JLL and system packages use standard bin/ directory, in-tree uses tools/
    tools_dir = (source == "jll" || source == "system") ? joinpath(llvm_root, "bin") : joinpath(llvm_root, "tools")

    if !isdir(tools_dir)
        @warn "LLVM tools directory not found: $tools_dir"
        return tools
    end

    # Essential LLVM tools
    essential_tools = [
        # Clang/LLVM compilers
        "clang", "clang++", "clang-20",

        # LLVM core tools
        "llvm-config", "llvm-link", "llvm-as", "llvm-dis",
        "opt", "llc", "lli",

        # Analysis and inspection
        "llvm-nm", "llvm-objdump", "llvm-ar", "llvm-ranlib",
        "llvm-readobj", "llvm-readelf", "llvm-dwarfdump",
        "llvm-symbolizer", "llvm-size", "llvm-strings",

        # Optimization and transformation
        "llvm-extract", "llvm-split", "llvm-reduce",
        "llvm-stress", "llvm-opt-report",

        # Linking and libraries
        "lld", "ld.lld", "ld64.lld", "lld-link",
        "llvm-lto", "llvm-lto2",

        # Code coverage and profiling
        "llvm-cov", "llvm-profdata", "llvm-profgen",

        # Clang tools
        "clang-format", "clang-tidy", "clang-check",
        "clang-query", "clangd", "clang-scan-deps",
        "clang-repl", "clang-refactor",

        # Debugging and sanitizers
        "llvm-debuginfod", "llvm-debuginfod-find",

        # MLIR tools (if available)
        "mlir-opt", "mlir-translate", "mlir-tblgen",

        # Miscellaneous
        "FileCheck", "count", "not",
        "dsymutil", "bugpoint"
    ]

    for tool_name in essential_tools
        tool_path = joinpath(tools_dir, tool_name)

        # Check if it exists (might be a symlink)
        if isfile(tool_path) || islink(tool_path)
            tools[tool_name] = tool_path
        end
    end

    return tools
end

"""
    discover_llvm_libraries(llvm_root::String) -> Dict{String,String}

Discover LLVM shared libraries.
"""
function discover_llvm_libraries(llvm_root::String)
    libraries = Dict{String,String}()
    lib_dir = joinpath(llvm_root, "lib")

    if !isdir(lib_dir)
        @warn "LLVM lib directory not found: $lib_dir"
        return libraries
    end

    # Key LLVM libraries
    key_libs = [
        "libLLVM.so.20.1jl",
        "libLLVM-20jl.so",
        "libclang.so.20.1.2jl",
        "libclang-cpp.so.20.1jl",
        "libmlir_async_runtime.so.20.1jl",
        "libmlir_float16_utils.so.20.1jl"
    ]

    # Find all shared libraries
    for (root, dirs, files) in walkdir(lib_dir)
        for file in files
            ext = splitext(file)[2]
            if ext == ".so" || contains(file, ".so.")
                lib_path = joinpath(root, file)
                lib_name = replace(file, r"\.so.*$" => "")
                lib_name = replace(lib_name, r"^lib" => "")
                libraries[lib_name] = lib_path
            end
        end
    end

    return libraries
end

"""
    query_llvm_config(llvm_config::String) -> Tuple

Query llvm-config for build configuration.
Returns (cxxflags, ldflags, libs)
"""
function query_llvm_config(llvm_config::String)
    cxxflags = String[]
    ldflags = String[]
    libs = ""

    try
        # Get C++ flags
        cxx_output = strip(read(`$llvm_config --cxxflags`, String))
        cxxflags = split(cxx_output)

        # Get linker flags
        ld_output = strip(read(`$llvm_config --ldflags`, String))
        ldflags = split(ld_output)

        # Get libraries
        libs = strip(read(`$llvm_config --libs core support`, String))

    catch e
        @warn "Failed to query llvm-config: $e"
    end

    return (cxxflags, ldflags, libs)
end

"""
    parse_llvm_version(version_str::String) -> Tuple{Int,Int,Int}

Parse LLVM version string into (major, minor, patch).
"""
function parse_llvm_version(version_str::String)
    # Remove trailing 'jl' suffix if present
    clean_version = replace(version_str, r"jl$" => "")

    parts = split(clean_version, '.')
    major = length(parts) >= 1 ? parse(Int, parts[1]) : 0
    minor = length(parts) >= 2 ? parse(Int, parts[2]) : 0
    patch = length(parts) >= 3 ? parse(Int, parts[3]) : 0

    return (major, minor, patch)
end

"""
    build_environment_vars(llvm_root::String, bin_dir::String, lib_dir::String, source::String) -> Dict{String,String}

Build environment variables for isolated LLVM toolchain.
"""
function build_environment_vars(llvm_root::String, bin_dir::String, lib_dir::String, source::String)
    env_vars = Dict{String,String}()

    # PATH - prepend LLVM bin directory
    current_path = get(ENV, "PATH", "")
    env_vars["PATH"] = "$bin_dir:$current_path"

    # LLVM-specific variables
    env_vars["LLVM_ROOT"] = llvm_root
    env_vars["LLVM_DIR"] = lib_dir  # For CMake find_package
    env_vars["Clang_DIR"] = lib_dir

    # For system toolchains, DO NOT set library/include paths to avoid breaking default search paths
    if source == "system"
        # Force clear LD_LIBRARY_PATH and CPATH to avoid Julia's libs from interfering with system tools
        env_vars["LD_LIBRARY_PATH"] = ""
        env_vars["CPATH"] = ""
        return env_vars
    end

    # LD_LIBRARY_PATH - add LLVM lib directory
    current_ld_path = get(ENV, "LD_LIBRARY_PATH", "")
    env_vars["LD_LIBRARY_PATH"] = isempty(current_ld_path) ? lib_dir : "$lib_dir:$current_ld_path"

    # LIBRARY_PATH - for linking
    current_lib_path = get(ENV, "LIBRARY_PATH", "")
    env_vars["LIBRARY_PATH"] = isempty(current_lib_path) ? lib_dir : "$lib_dir:$current_lib_path"

    # CPATH - for C/C++ includes
    include_dir = joinpath(llvm_root, "include")
    current_cpath = get(ENV, "CPATH", "")
    env_vars["CPATH"] = isempty(current_cpath) ? include_dir : "$include_dir:$current_cpath"

    return env_vars
end

"""
    init_toolchain(;isolated::Bool=true, config=nothing, source::Symbol=:auto) -> LLVMToolchain

Initialize the LLVM toolchain.

# Arguments
- `isolated::Bool`: If true, uses RepliBuild's local LLVM. If false, allows system LLVM fallback.
- `config`: Optional RepliBuildConfig - reads paths from TOML if provided
- `source::Symbol`: Toolchain source - :auto (prefer JLL), :intree (force in-tree), :jll (force JLL)

# Returns
- `LLVMToolchain`: Configured toolchain instance

# Priority:
1. If config provided and has llvm.root → use TOML paths
2. If config empty → auto-discover based on source preference
3. Validate all tools exist, discover if missing
"""
function init_toolchain(; isolated::Bool=true, config=nothing, source::Symbol=:auto)
    # Read from config struct first (source of truth), or auto-discover
    (llvm_root, toolchain_source) = if config !== nothing && !isempty(config.llvm.version)
        println("Initializing LLVM Toolchain from Config")
        # Config has LLVM info - check if we can get root from somewhere
        # For now, just auto-discover since config.llvm doesn't store full root path
        (root, src) = get_llvm_root(config.llvm.toolchain)
        (root, String(config.llvm.toolchain))
    else
        println("Initializing LLVM Toolchain (auto-discover)")
        (root, src) = get_llvm_root(source)
        (root, src)
    end

    println("   Root: $llvm_root")
    println("   Source: $toolchain_source")

    # Setup paths (JLL and system use bin/, in-tree uses tools/)
    bin_dir = (toolchain_source == "jll" || toolchain_source == "system") ? joinpath(llvm_root, "bin") : joinpath(llvm_root, "tools")
    lib_dir = joinpath(llvm_root, "lib")
    include_dir = joinpath(llvm_root, "include")
    libexec_dir = joinpath(llvm_root, "libexec")
    share_dir = joinpath(llvm_root, "share")

    # Verify critical directories
    for (name, dir) in [("bin", bin_dir), ("lib", lib_dir), ("include", include_dir)]
        if !isdir(dir)
            error("Critical LLVM directory missing: $name at $dir")
        end
    end

    # Get version
    llvm_config = joinpath(bin_dir, "llvm-config")
    version_str = "20.1.2jl"  # Default

    if isfile(llvm_config)
        try
            version_str = String(strip(read(`$llvm_config --version`, String)))
        catch
            @warn "Could not query LLVM version, using default: $version_str"
        end
    end

    (major, minor, patch) = parse_llvm_version(String(version_str))

    println("   Version: $version_str (LLVM $major.$minor.$patch)")

    # Auto-discover tools (config doesn't cache tools in new immutable struct)
    println("   Tools: Auto-discovering...")
    tools = discover_llvm_tools(llvm_root, toolchain_source)
    println("   Tools: Discovered $(length(tools)) tools (cached for next run)")

    # Validate tools exist, warn if missing
    for (name, path) in tools
        if !isfile(path) && !islink(path)
            @warn "Tool specified in TOML not found: $name at $path"
        end
    end

    # Discover libraries
    libraries = discover_llvm_libraries(llvm_root)
    println("   Libraries: $(length(libraries)) discovered")

    # Query llvm-config
    (cxxflags, ldflags, libs) = query_llvm_config(llvm_config)

    # Build environment variables
    env_vars = build_environment_vars(llvm_root, bin_dir, lib_dir, toolchain_source)

    toolchain = LLVMToolchain(
        llvm_root,
        bin_dir,
        lib_dir,
        include_dir,
        libexec_dir,
        share_dir,
        version_str,
        major,
        minor,
        patch,
        tools,
        libraries,
        cxxflags,
        ldflags,
        libs,
        env_vars,
        isolated,
        toolchain_source
    )

    println(" LLVM Toolchain initialized")

    return toolchain
end

"""
    get_toolchain() -> LLVMToolchain

Get or initialize the global LLVM toolchain.
"""
function get_toolchain()
    if GLOBAL_LLVM_TOOLCHAIN[] === nothing
        GLOBAL_LLVM_TOOLCHAIN[] = init_toolchain()
    end
    return GLOBAL_LLVM_TOOLCHAIN[]
end

"""
    with_llvm_env(f::Function)

Execute function with LLVM environment variables set.
Temporarily modifies ENV for the duration of the function call.

# Example
```julia
with_llvm_env() do
    run(`clang++ --version`)
end
```
"""
function with_llvm_env(f::Function)
    toolchain = get_toolchain()

    # Save original environment
    original_env = Dict{String,String}()
    for (key, value) in toolchain.env_vars
        original_env[key] = get(ENV, key, "")
    end

    try
        # Set LLVM environment
        for (key, value) in toolchain.env_vars
            ENV[key] = value
        end

        # Execute function
        return f()
    finally
        # Restore original environment
        for (key, original_value) in original_env
            if isempty(original_value)
                delete!(ENV, key)
            else
                ENV[key] = original_value
            end
        end
    end
end

"""
    get_tool(tool_name::String) -> String

Get absolute path to an LLVM tool.
Returns empty string if tool not found.

# Example
```julia
clang_path = get_tool("clang++")
```
"""
function get_tool(tool_name::String)
    toolchain = get_toolchain()
    return get(toolchain.tools, tool_name, "")
end

"""
    has_tool(tool_name::String) -> Bool

Check if an LLVM tool is available.
"""
function has_tool(tool_name::String)
    !isempty(get_tool(tool_name))
end

"""
    get_library(lib_name::String) -> String

Get absolute path to an LLVM library.
"""
function get_library(lib_name::String)
    toolchain = get_toolchain()
    return get(toolchain.libraries, lib_name, "")
end

"""
    get_include_flags() -> Vector{String}

Get C++ include flags for LLVM headers.
"""
function get_include_flags()
    toolchain = get_toolchain()
    return ["-I$(toolchain.include_dir)"]
end

"""
    get_link_flags() -> Vector{String}

Get linker flags for LLVM libraries.
"""
function get_link_flags()
    toolchain = get_toolchain()
    return ["-L$(toolchain.lib_dir)", "-Wl,-rpath,$(toolchain.lib_dir)"]
end

"""
    run_tool(tool_name::String, args::Vector{String}; capture_output::Bool=true)

Run an LLVM tool with isolated environment.

# Example
```julia
(output, exitcode) = run_tool("clang++", ["--version"])
```
"""
function run_tool(tool_name::String, args::Vector{String}; capture_output::Bool=true)
    tool_path = get_tool(tool_name)

    if isempty(tool_path)
        return ("Tool not found: $tool_name", 1)
    end

    # Run with LLVM environment
    return with_llvm_env() do
        cmd = `$tool_path $args`

        try
            if capture_output
                output = read(cmd, String)
                return (output, 0)
            else
                run(cmd)
                return ("", 0)
            end
        catch e
            if isa(e, ProcessFailedException)
                try
                    output = read(cmd, String)
                    return (output, 1)
                catch
                    return ("Process failed: $e", 1)
                end
            else
                return ("Error: $e", 1)
            end
        end
    end
end

"""
    print_toolchain_info()

Print detailed information about the LLVM toolchain.
"""
function print_toolchain_info()
    toolchain = get_toolchain()

    println("="^70)
    println("LLVM Toolchain Information")
    println("="^70)
    println()
    println("🎯 Source: $(toolchain.source)")
    println()
    println(" Paths:")
    println("   Root:       $(toolchain.root)")
    println("   Bin:        $(toolchain.bin_dir)")
    println("   Lib:        $(toolchain.lib_dir)")
    println("   Include:    $(toolchain.include_dir)")
    println()
    println(" Version:")
    println("   $(toolchain.version)")
    println("   Major: $(toolchain.version_major)")
    println("   Minor: $(toolchain.version_minor)")
    println("   Patch: $(toolchain.version_patch)")
    println()
    println("Tools: ($(length(toolchain.tools)) available)")
    for (name, path) in sort(collect(toolchain.tools))
        exists = isfile(path) || islink(path)
        status = exists ? "✓" : "✗"
        println("   $status $name")
    end
    println()
    println(" Libraries: ($(length(toolchain.libraries)) available)")
    count = 0
    for (name, path) in sort(collect(toolchain.libraries))
        if count < 10  # Limit output
            exists = isfile(path)
            status = exists ? "✓" : "✗"
            println("   $status $name")
            count += 1
        end
    end
    if length(toolchain.libraries) > 10
        println("   ... and $(length(toolchain.libraries) - 10) more")
    end
    println()
    println("Environment Variables:")
    for (key, value) in sort(collect(toolchain.env_vars))
        # Truncate long paths for display
        display_value = length(value) > 60 ? value[1:57] * "..." : value
        println("   $key = $display_value")
    end
    println()
    println("Isolation: $(toolchain.isolated ? "Enabled" : "Disabled")")
    println("="^70)
end

"""
    verify_toolchain() -> Bool

Verify that the LLVM toolchain is properly installed and functional.
"""
function verify_toolchain()
    println(" Verifying LLVM Toolchain...")

    toolchain = get_toolchain()
    all_ok = true

    # Check essential tools
    essential = ["clang++", "llvm-config", "llvm-link", "opt", "llc"]

    for tool in essential
        if has_tool(tool)
            println("   $tool")
        else
            println("  ❌ $tool (missing)")
            all_ok = false
        end
    end

    # Test clang++
    println("\n🧪 Testing clang++...")
    (output, exitcode) = run_tool("clang++", ["--version"])

    if exitcode == 0
        println("   clang++ is functional")
        println("     $(split(output, '\n')[1])")
    else
        println("  ❌ clang++ test failed")
        all_ok = false
    end

    # Test llvm-config
    println("\n Testing llvm-config...")
    (output, exitcode) = run_tool("llvm-config", ["--version"])

    if exitcode == 0
        println("   llvm-config is functional")
        println("     Version: $(strip(output))")
    else
        println("  ❌ llvm-config test failed")
        all_ok = false
    end

    if all_ok
        println("\n LLVM Toolchain verification PASSED")
    else
        println("\n❌ LLVM Toolchain verification FAILED")
    end

    return all_ok
end

# Module exports
export LLVMToolchain,
    get_toolchain,
    init_toolchain,
    with_llvm_env,
    get_tool,
    has_tool,
    get_library,
    get_include_flags,
    get_link_flags,
    run_tool,
    print_toolchain_info,
    verify_toolchain,
    get_jll_llvm_root,
    get_llvm_root,
    LLVM_JLL_AVAILABLE

end # module LLVMEnvironment
