#!/usr/bin/env julia
# Scaffold.jl - Standardized package scaffolding for RepliBuild wrapper packages
# Generates the exact directory structure needed to distribute a replibuild.toml as a Julia package

module Scaffold

using UUIDs
using Dates

export scaffold_package

# =============================================================================
# TEMPLATES
# =============================================================================

function _project_toml(name::String, uuid::UUID)::String
    return """
    name = "$name"
    uuid = "$(string(uuid))"
    version = "0.1.0"
    authors = ["$(get(ENV, "USER", get(ENV, "USERNAME", "author")))"]

    [deps]
    RepliBuild = "4450f29b-7b71-45c6-8742-e7520a479938"
    Libdl = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

    [compat]
    RepliBuild = "2"
    julia = "1.10"

    [extras]
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

    [targets]
    test = ["Test"]
    """
end

function _replibuild_toml(name::String, uuid::UUID, root::String)::String
    return """
    # RepliBuild configuration for $name
    # Edit this file to point at your C/C++ source code.
    # See: https://obsidianjulua.github.io/RepliBuild.jl/stable/config/

    [project]
    name = "$(lowercase(name))"
    root = "$root"
    uuid = "$(string(uuid))"

    [compile]
    # source_files = ["src/mylib.cpp"]
    # include_dirs = ["include"]
    flags = ["-std=c++17", "-fPIC"]
    parallel = true
    aot_thunks = false

    [link]
    enable_lto = false
    optimization_level = "2"

    [binary]
    type = "shared"
    strip_symbols = false

    [cache]
    enabled = true
    directory = ".replibuild_cache"

    [wrap]
    style = "clang"
    use_clang_jl = true
    enabled = true

    [types]
    strictness = "warn"
    allow_unknown_structs = true
    allow_function_pointers = true
    """
end

function _main_module(name::String)::String
    return """
    module $name

    using RepliBuild

    # Path to the generated wrapper (built by deps/build.jl)
    const _wrapper_path = joinpath(@__DIR__, "..", "julia", "$(name).jl")

    function __init__()
        if !isfile(_wrapper_path)
            @warn \"\"\"
            $name wrapper not found at \$(_wrapper_path).
            Run `using Pkg; Pkg.build("$name")` to compile and generate bindings.
            \"\"\"
        end
    end

    # Include the generated wrapper if it exists
    if isfile(joinpath(@__DIR__, "..", "julia", "$(name).jl"))
        include(joinpath(@__DIR__, "..", "julia", "$(name).jl"))
    end

    end # module $name
    """
end

function _build_jl(name::String)::String
    return """
    #!/usr/bin/env julia
    # Build script for $name
    # Automatically invoked by Pkg.build("$name")

    using RepliBuild

    const PROJECT_ROOT = dirname(@__DIR__)
    const TOML_PATH = joinpath(PROJECT_ROOT, "replibuild.toml")

    if !isfile(TOML_PATH)
        @warn "replibuild.toml not found at \$TOML_PATH — skipping build."
    else
        # Check environment before building
        status = RepliBuild.check_environment(verbose=true, throw_on_error=false)
        if !status.ready
            @error \"\"\"
            RepliBuild toolchain not ready. Cannot compile $name.
            Run `RepliBuild.check_environment()` for details.
            \"\"\"
        else
            println("[$(name)] Building C/C++ library...")
            RepliBuild.build(TOML_PATH)

            println("[$(name)] Generating Julia wrappers...")
            RepliBuild.wrap(TOML_PATH)

            println("[$(name)] Build complete.")
        end
    end
    """
end

function _test_runtests(name::String)::String
    return """
    using Test
    using $name

    @testset "$name" begin
        # Add your tests here
        @test true
    end
    """
end

function _gitignore()::String
    return """
    build/
    julia/
    .replibuild_cache/
    *.so
    *.dylib
    *.dll
    *.o
    Manifest.toml
    """
end

# =============================================================================
# PUBLIC API
# =============================================================================

"""
    scaffold_package(name::String; path::String=".") -> String

Generate a standardized Julia package structure for distributing RepliBuild wrappers.

Creates:
```
<name>/
├── Project.toml        (depends on RepliBuild)
├── replibuild.toml     (build configuration — edit this)
├── .gitignore
├── src/
│   └── <name>.jl       (stub that includes generated wrapper)
├── deps/
│   └── build.jl        (hook that calls RepliBuild.build + wrap)
└── test/
    └── runtests.jl     (test skeleton)
```

# Arguments
- `name`: Package name (e.g., "MyEigenWrapper"). Must be a valid Julia identifier.
- `path`: Parent directory to create the package in (default: current directory)

# Returns
Absolute path to the created package directory.

# Example
```julia
RepliBuild.scaffold_package("MyEigenWrapper")
# Creates MyEigenWrapper/ with full package structure
# Edit replibuild.toml to point at your C/C++ source, then:
#   cd MyEigenWrapper && julia -e 'using Pkg; Pkg.build()'
```
"""
function scaffold_package(name::String; path::String=".")::String
    # Validate package name
    if !occursin(r"^[A-Z][A-Za-z0-9_]*$", name)
        error("Package name must be a valid Julia identifier starting with uppercase: got \"$name\"")
    end

    pkg_dir = abspath(joinpath(path, name))

    if isdir(pkg_dir)
        error("Directory already exists: $pkg_dir")
    end

    uuid = uuid4()

    # Create directory structure
    mkpath(joinpath(pkg_dir, "src"))
    mkpath(joinpath(pkg_dir, "deps"))
    mkpath(joinpath(pkg_dir, "test"))

    # Write files
    write(joinpath(pkg_dir, "Project.toml"), _project_toml(name, uuid))
    write(joinpath(pkg_dir, "replibuild.toml"), _replibuild_toml(name, uuid, pkg_dir))
    write(joinpath(pkg_dir, "src", "$name.jl"), _main_module(name))
    write(joinpath(pkg_dir, "deps", "build.jl"), _build_jl(name))
    write(joinpath(pkg_dir, "test", "runtests.jl"), _test_runtests(name))
    write(joinpath(pkg_dir, ".gitignore"), _gitignore())

    println("[RepliBuild] 📦 Scaffolded package: $name")
    println()
    println("  $pkg_dir/")
    println("  ├── Project.toml")
    println("  ├── replibuild.toml  ← edit this to point at your C/C++ source")
    println("  ├── src/")
    println("  │   └── $name.jl")
    println("  ├── deps/")
    println("  │   └── build.jl")
    println("  └── test/")
    println("      └── runtests.jl")
    println()
    println("  Next steps:")
    println("    1. Edit replibuild.toml with your source files and include dirs")
    println("    2. cd $name && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.build()'")
    println()

    return pkg_dir
end

end # module Scaffold
