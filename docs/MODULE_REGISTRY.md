# RepliBuild Module Registry

## Vision: A Centralized Module Registry for C/C++ Libraries

Similar to Julia's General registry for packages, we need a **General registry for build modules**.

## Current State (v1.2)

**Built-in Modules:**
- Qt5, Boost, Eigen, Zlib (shipped with RepliBuild.jl)

**User Modules:**
- `~/.julia/replibuild/modules/` (personal modules)

**Project Modules:**
- `.replibuild/modules/` (project-specific overrides)

## Future: Centralized Registry

### Structure

```
RepliBuildModules/                    # Registry repository
├── Registry.toml                     # Registry metadata
├── Q/
│   └── Qt5/
│       ├── Package.toml              # Module metadata
│       ├── Versions.toml             # Version history
│       └── Deps.toml                 # Component dependencies
├── B/
│   └── Boost/
│       ├── Package.toml
│       ├── Versions.toml
│       └── Deps.toml
└── ...
```

### Registry.toml

```toml
name = "RepliBuild General"
uuid = "..."
repo = "https://github.com/RepliBuild/Modules.git"
description = "Official RepliBuild module registry"

[packages]
# Auto-generated list of all packages
```

### Package.toml (per module)

```toml
name = "Qt5"
uuid = "..."
repo = "https://github.com/RepliBuild/Modules.git"
subdir = "Q/Qt5"
description = "Qt5 cross-platform framework"
license = "LGPL-3.0"

[jll]
package = "Qt5Base_jll"
components = ["Core", "Widgets", "Network", "Gui", "Sql", "Xml"]

[maintainers]
github = ["maintainer1", "maintainer2"]
```

### Versions.toml

```toml
["5.15.2"]
git-tree-sha1 = "..."
module-hash = "..."

["5.15.3"]
git-tree-sha1 = "..."
module-hash = "..."
```

## Module Creation Workflow

### 1. Manual Creation

```julia
using RepliBuild

# Create template
RepliBuild.create_module_template("SDL2")

# Edit ~/.julia/replibuild/modules/SDL2.toml
# Test locally
mod = RepliBuild.resolve_module("SDL2")

# Submit to registry
# (will be automated)
```

### 2. From pkg-config

```julia
# Auto-generate from system package
RepliBuild.generate_from_pkg_config("gtk+-3.0")

# Output: ~/.julia/replibuild/modules/Gtk3.toml
# Review and submit
```

### 3. From CMake Discovery

```julia
# Parse CMakeLists.txt to generate module
RepliBuild.generate_from_cmake("OpenCV")

# Analyzes find_package() calls
# Extracts components and flags
# Output: ~/.julia/replibuild/modules/OpenCV.toml
```

### 4. From Julia JLL Package

```julia
# Introspect existing JLL
RepliBuild.generate_from_jll("OpenBLAS_jll")

# Extracts artifact paths
# Discovers exported libraries
# Output: ~/.julia/replibuild/modules/OpenBLAS.toml
```

## Parser System for Auto-Generation

### CMakeLists.txt Parser

```julia
module CMakeListsParser

"""
Parse CMakeLists.txt to extract module information.

Extracts:
- project(NAME VERSION)
- find_package() calls
- target_link_libraries()
- target_include_directories()
- Compiler flags
"""
function parse_cmakelists(file::String) -> ModuleInfo
    content = read(file, String)

    # Extract project info
    project_match = match(r"project\s*\(\s*(\w+).*?VERSION\s+([\d.]+)", content)

    # Extract dependencies
    dependencies = extract_find_packages(content)

    # Extract flags
    flags = extract_compile_flags(content)

    return ModuleInfo(...)
end

function extract_find_packages(content::String)
    # find_package(Qt5 REQUIRED COMPONENTS Core Widgets)
    # find_package(Boost COMPONENTS system filesystem)
    packages = []

    for m in eachmatch(r"find_package\s*\(\s*(\w+)(?:.*?COMPONENTS\s+(.*?))?\)", content)
        pkg_name = m.captures[1]
        components = m.captures[2] !== nothing ? split(m.captures[2]) : []
        push!(packages, (name=pkg_name, components=components))
    end

    return packages
end

end # module
```

### pkg-config Parser

```julia
module PkgConfigParser

"""
Query pkg-config for module information.
"""
function parse_pkg_config(pkg_name::String) -> ModuleInfo
    # Get package info
    version = read(`pkg-config --modversion $pkg_name`, String)
    cflags = split(read(`pkg-config --cflags $pkg_name`, String))
    libs = split(read(`pkg-config --libs $pkg_name`, String))

    # Parse flags
    include_dirs = [flag[3:end] for flag in cflags if startswith(flag, "-I")]
    defines = [flag for flag in cflags if startswith(flag, "-D")]
    lib_dirs = [flag[3:end] for flag in libs if startswith(flag, "-L")]
    libraries = [flag[3:end] for flag in libs if startswith(flag, "-l")]

    return ModuleInfo(
        name = pkg_name,
        version = strip(version),
        include_dirs = include_dirs,
        library_dirs = lib_dirs,
        libraries = libraries,
        compile_flags = defines
    )
end

end # module
```

### JLL Introspection

```julia
module JLLIntrospector

"""
Introspect JLL package to extract module information.
"""
function introspect_jll(jll_name::String) -> ModuleInfo
    # Load JLL
    jll_mod = Base.require(Main, Symbol(jll_name))

    # Extract artifact directory
    artifact_dir = discover_artifact_dir(jll_mod)

    # Find libraries
    lib_dir = joinpath(artifact_dir, "lib")
    libraries = []

    for file in readdir(lib_dir)
        if endswith(file, ".so") || endswith(file, ".dylib")
            # Parse library name
            libname = parse_library_name(file)
            push!(libraries, libname)
        end
    end

    # Find headers
    include_dir = joinpath(artifact_dir, "include")
    headers = []

    if isdir(include_dir)
        for (root, dirs, files) in walkdir(include_dir)
            for file in files
                if endswith(file, ".h") || endswith(file, ".hpp")
                    push!(headers, relpath(joinpath(root, file), include_dir))
                end
            end
        end
    end

    return ModuleInfo(
        name = replace(jll_name, "_jll" => ""),
        jll_package = jll_name,
        artifact_dir = artifact_dir,
        include_dirs = [include_dir],
        library_dirs = [lib_dir],
        libraries = libraries,
        headers = headers
    )
end

end # module
```

## Registry Management Commands (Future)

```julia
# Add registry
RepliBuild.add_registry("https://github.com/RepliBuild/Modules")

# Update registries
RepliBuild.update_registries()

# Search for modules
RepliBuild.search_modules("opencv")
# Output:
# Found 2 modules:
#   OpenCV - Computer vision library (v4.5.0)
#   OpenCV_Contrib - OpenCV contributed modules (v4.5.0)

# Install module
RepliBuild.install_module("OpenCV")
# Downloads to ~/.julia/replibuild/modules/

# List installed modules
RepliBuild.list_modules()
# Output:
# Installed modules:
#   Qt5 (builtin)
#   Boost (builtin)
#   OpenCV (user)
#   SDL2 (user)

# Update module
RepliBuild.update_module("OpenCV")

# Remove module
RepliBuild.remove_module("OpenCV")
```

## Contributing Modules to Registry

### Step 1: Create Module Locally

```bash
# Create module
julia> using RepliBuild
julia> RepliBuild.create_module_template("MyLib")

# Edit and test
vim ~/.julia/replibuild/modules/MyLib.toml
julia> mod = RepliBuild.resolve_module("MyLib")
```

### Step 2: Validate Module

```julia
# Validate module format
RepliBuild.validate_module("MyLib")
# ✓ Module name is valid
# ✓ JLL package exists
# ✓ All components are valid
# ✓ Version constraints are valid
# ✓ No syntax errors
```

### Step 3: Submit to Registry

```bash
# Fork registry repo
git clone https://github.com/RepliBuild/Modules
cd Modules

# Add module
mkdir -p M/MyLib
cp ~/.julia/replibuild/modules/MyLib.toml M/MyLib/Module.toml

# Create Package.toml
cat > M/MyLib/Package.toml << EOF
name = "MyLib"
uuid = "$(uuidgen)"
repo = "https://github.com/RepliBuild/Modules.git"
EOF

# Create Versions.toml
cat > M/MyLib/Versions.toml << EOF
["1.0.0"]
git-tree-sha1 = "$(git hash-object M/MyLib/Module.toml)"
EOF

# Commit and PR
git add M/MyLib/
git commit -m "Add MyLib module"
git push origin add-mylib
# Create PR on GitHub
```

## Module Quality Standards

For registry acceptance, modules must:

1. **Be tested** - Work with at least one real project
2. **Have documentation** - Usage examples in Module.toml comments
3. **Follow naming conventions** - Match JLL package naming
4. **Specify versions** - Include version constraints
5. **List components** - If library has multiple components
6. **Include metadata** - License, homepage, description

## Roadmap

### - Registry Infrastructure
- [ ] Registry.toml format specification
- [ ] Registry management CLI
- [ ] Module validation system
- [ ] CI/CD for registry PRs

### - Auto-Generation
- [ ] CMake modules to RepliBuid and Julia bindings to jll automaticaly
- [ ] pkg-config introspection
- [ ] JLL package introspection
- [ ] Bulk generation from Julia packages

### - Community Registry
- [ ] Launch official RepliBuild Modules registry
- [ ] Auto-sync with Julia General (for JLLs)
- [ ] Module rating/verification system
- [ ] Community contribution guidelines

### - Advanced Features
- [ ] Dependency resolution between modules
- [ ] Module versioning with semantic constraints
- [ ] Module inheritance (e.g., Qt6 extends Qt5)
- [ ] Private registries for organizations

## Example: Generating Modules from Julia Ecosystem

```julia
# Scan all JLL packages in Julia General
using Pkg

jll_packages = []
for (uuid, pkg) in Pkg.dependencies()
    if endswith(pkg.name, "_jll")
        push!(jll_packages, pkg.name)
    end
end

println("Found $(length(jll_packages)) JLL packages")

# Generate modules for each
for jll in jll_packages
    try
        println("Generating module for $jll...")
        RepliBuild.generate_from_jll(jll)
    catch e
        @warn "Failed to generate $jll: $e"
    end
end

# Result: 500+ auto-generated modules!
```

## Get Involved

Want to help build the registry?

1. **Create modules** for libraries you use
2. **Test existing modules** with real projects
3. **Improve parsers** for better auto-generation
4. **Write documentation** and examples
5. **Join discussions** on GitHub
