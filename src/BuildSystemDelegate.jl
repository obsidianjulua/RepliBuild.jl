#!/usr/bin/env julia
# BuildSystemDelegate.jl - Smart delegation to existing build systems via JLL packages
# Philosophy: Don't reimplement build systems, orchestrate them!

module BuildSystemDelegate

using Pkg
using TOML

export detect_build_system, delegate_build, BuildSystemType
export QtBuildDelegate, CMakeBuildDelegate, MesonBuildDelegate

# ============================================================================
# BUILD SYSTEM DETECTION
# ============================================================================

@enum BuildSystemType begin
    CMAKE
    QMAKE
    MESON
    AUTOTOOLS
    MAKE
    CARGO
    UNKNOWN
end

"""
    detect_build_system(project_dir::String) -> BuildSystemType

Detect which build system a project uses by looking for signature files.
"""
function detect_build_system(project_dir::String)
    files = readdir(project_dir)

    # Priority order (most specific to least specific)
    if any(f -> endswith(f, ".pro"), files)
        return QMAKE
    elseif "CMakeLists.txt" in files
        return CMAKE
    elseif "meson.build" in files
        return MESON
    elseif "configure.ac" in files || "configure.in" in files
        return AUTOTOOLS
    elseif "Cargo.toml" in files
        return CARGO
    elseif "Makefile" in files || "makefile" in files
        return MAKE
    else
        return UNKNOWN
    end
end

# ============================================================================
# BUILD DELEGATES - Use JLL packages to run actual build systems
# ============================================================================

"""
Abstract interface for build system delegates.
Each delegate knows how to:
1. Install required JLL packages
2. Run the build system
3. Extract build artifacts
"""
abstract type BuildDelegate end

"""
    delegate_build(project_dir::String; config::Dict=Dict()) -> Dict

Smart build delegation:
1. Detect build system
2. Create appropriate delegate
3. Run build via JLL packages
4. Return artifact locations
"""
function delegate_build(project_dir::String; config::Dict=Dict())
    build_type = detect_build_system(project_dir)

    println("🔍 Detected build system: $build_type")

    delegate = create_delegate(build_type, project_dir, config)

    if isnothing(delegate)
        error("Unsupported build system: $build_type")
    end

    # Delegate does all the work!
    return execute_build(delegate)
end

function create_delegate(build_type::BuildSystemType, project_dir::String, config::Dict)
    if build_type == QMAKE
        return QtBuildDelegate(project_dir, config)
    elseif build_type == CMAKE
        return CMakeBuildDelegate(project_dir, config)
    elseif build_type == MESON
        return MesonBuildDelegate(project_dir, config)
    else
        return nothing
    end
end

# ============================================================================
# Qt/qmake DELEGATE - Uses Qt JLL packages
# ============================================================================

"""
QtBuildDelegate - Delegates to Qt's qmake/make via Qt5Base_jll

How it works:
1. Ensures Qt5Base_jll is installed (has qmake)
2. Runs qmake to generate Makefiles
3. Runs make to build
4. Extracts resulting libraries
"""
mutable struct QtBuildDelegate <: BuildDelegate
    project_dir::String
    config::Dict
    qt_version::String  # "Qt5" or "Qt6"
    build_dir::String

    function QtBuildDelegate(project_dir::String, config::Dict)
        qt_version = get(config, "qt_version", "Qt5")
        build_dir = get(config, "build_dir", joinpath(project_dir, "build"))
        new(project_dir, config, qt_version, build_dir)
    end
end

"""
Execute Qt build by calling qmake and make from Qt JLL
"""
function execute_build(delegate::QtBuildDelegate)
    println("🚀 Qt Build Delegation")
    println("   Project: $(delegate.project_dir)")
    println("   Qt Version: $(delegate.qt_version)")

    # Step 1: Ensure Qt JLL packages are available
    qt_jll = ensure_qt_tools(delegate.qt_version)

    # Step 2: Find qmake from JLL
    qmake_path = get_qmake_path(qt_jll)

    # Step 3: Create build directory
    mkpath(delegate.build_dir)

    # Step 4: Run qmake (delegate to Qt's build system!)
    println("   Running qmake...")
    result = run_qt_command(qmake_path, delegate.project_dir, delegate.build_dir)

    if result[:success]
        # Step 5: Run make
        println("   Running make...")
        make_result = run_make(delegate.build_dir)

        if make_result[:success]
            # Step 6: Find and return built artifacts
            return extract_qt_artifacts(delegate.build_dir)
        else
            error("Make failed: $(make_result[:error])")
        end
    else
        error("qmake failed: $(result[:error])")
    end
end

"""
Ensure Qt JLL packages are installed and return the main package
"""
function ensure_qt_tools(qt_version::String)
    jll_name = qt_version == "Qt6" ? "Qt6Base_jll" : "Qt5Base_jll"

    # Check if installed
    pkg_info = Pkg.dependencies()
    is_installed = any(p -> p.second.name == jll_name, pkg_info)

    if !is_installed
        println("   📦 Installing $jll_name...")
        Pkg.add(jll_name)
    else
        println("   ✓ $jll_name already installed")
    end

    # Load the JLL package
    return Base.require(Main, Symbol(jll_name))
end

"""
Get qmake path from Qt JLL package
"""
function get_qmake_path(qt_jll)
    # Qt JLL packages expose qmake_path or similar
    # Try multiple methods to find it

    # Method 1: Direct qmake_path export
    if isdefined(qt_jll, :qmake_path)
        return string(qt_jll.qmake_path)
    end

    # Method 2: Find in bin directory
    for name in names(qt_jll; all=true)
        if occursin("path", lowercase(string(name)))
            try
                val = getfield(qt_jll, name)
                if isa(val, String) && isdir(val)
                    # Check if this is bin dir
                    qmake = joinpath(val, "qmake")
                    if isfile(qmake)
                        return qmake
                    end
                end
            catch
            end
        end
    end

    # Method 3: Search artifact directory
    # Get any path from the JLL and work backwards
    for name in names(qt_jll; all=true)
        try
            val = getfield(qt_jll, name)
            if isa(val, String) && isfile(val)
                # This is some file in the artifact
                artifact_root = dirname(dirname(val))
                bin_dir = joinpath(artifact_root, "bin")
                qmake = joinpath(bin_dir, "qmake")
                if isfile(qmake)
                    return qmake
                end
            end
        catch
        end
    end

    error("Could not find qmake in Qt JLL package")
end

"""
Run qmake command
"""
function run_qt_command(qmake_path::String, project_dir::String, build_dir::String)
    try
        # Find .pro file
        pro_files = filter(f -> endswith(f, ".pro"), readdir(project_dir))
        if isempty(pro_files)
            return Dict(:success => false, :error => "No .pro file found")
        end

        pro_file = joinpath(project_dir, first(pro_files))

        # Run qmake in build directory
        cd(build_dir) do
            run(`$qmake_path $pro_file`)
        end

        return Dict(:success => true)
    catch e
        return Dict(:success => false, :error => string(e))
    end
end

"""
Run make in build directory
"""
function run_make(build_dir::String; jobs::Int=Sys.CPU_THREADS)
    try
        cd(build_dir) do
            run(`make -j$jobs`)
        end
        return Dict(:success => true)
    catch e
        return Dict(:success => false, :error => string(e))
    end
end

"""
Extract built artifacts (libraries, executables) from build directory
"""
function extract_qt_artifacts(build_dir::String)
    artifacts = Dict(
        :libraries => String[],
        :executables => String[],
        :build_dir => build_dir
    )

    # Find all .so, .dylib, .dll files
    for (root, dirs, files) in walkdir(build_dir)
        for file in files
            path = joinpath(root, file)
            if endswith(file, ".so") || endswith(file, ".dylib") || endswith(file, ".dll")
                push!(artifacts[:libraries], path)
            elseif isfile(path) && uperm(stat(path)) & 0x1 != 0  # Executable bit
                # Might be an executable
                if !endswith(file, ".so") && !endswith(file, ".a")
                    push!(artifacts[:executables], path)
                end
            end
        end
    end

    return artifacts
end

# ============================================================================
# CMAKE DELEGATE - Uses CMake JLL
# ============================================================================

mutable struct CMakeBuildDelegate <: BuildDelegate
    project_dir::String
    config::Dict
    build_dir::String

    function CMakeBuildDelegate(project_dir::String, config::Dict)
        build_dir = get(config, "build_dir", joinpath(project_dir, "build"))
        new(project_dir, config, build_dir)
    end
end

function execute_build(delegate::CMakeBuildDelegate)
    println("🚀 CMake Build Delegation")

    # Ensure CMake_jll is available
    ensure_cmake()

    # TODO: Similar pattern - call cmake, then make
    # For now, placeholder
    println("   CMake delegation not yet implemented")
    return Dict(:success => false)
end

function ensure_cmake()
    # Check for CMake_jll
    # Similar to ensure_qt_tools
end

# ============================================================================
# MESON DELEGATE
# ============================================================================

mutable struct MesonBuildDelegate <: BuildDelegate
    project_dir::String
    config::Dict
    build_dir::String

    function MesonBuildDelegate(project_dir::String, config::Dict)
        build_dir = get(config, "build_dir", joinpath(project_dir, "build"))
        new(project_dir, config, build_dir)
    end
end

function execute_build(delegate::MesonBuildDelegate)
    println("🚀 Meson Build Delegation")
    println("   Meson delegation not yet implemented")
    return Dict(:success => false)
end

end # module BuildSystemDelegate
