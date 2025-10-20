# RepliBuild Module System

## Philosophy: The Julia Way of Building C/C++ Projects

The RepliBuild module system bridges **years of C/C++ build complexity** with **Julia's reproducible package ecosystem**.

### The Problem We Solve

Traditional C/C++ build systems require:
- Manual installation of dependencies
- System-specific paths (`/usr/lib`, `/opt/homebrew`, etc.)
- Flag discovery via CMake's `find_package()` or `pkg-config`
- Build scripts that break across platforms

Julia has JLL packages (binary artifacts) but they're just **wrappers**. They provide binaries but don't help with:
- Flag-specific builds (custom compiler flags, optimization levels)
- Component selection (Boost::system vs Boost::filesystem)
- Version-locked builds
- Integration with existing build systems

### The Solution: RepliBuild Modules

**RepliBuild Modules** = JLL packages + Build Logic + Flag Templates

A module is a `.toml` file that describes:
1. How to get the library (JLL, system, or custom)
2. What compiler/linker flags are needed
3. What components are available
4. How to detect and validate the library

## Module Structure

### Minimal Module (`Qt5.toml`)

```toml
[module]
name = "Qt5"
cmake_name = "Qt5"  # What CMake calls it
description = "Qt5 GUI framework"

# Resolution strategy (priority order)
[resolution]
prefer = "jll"  # Try JLL first
fallback = ["system", "custom"]  # Fall back to system if JLL unavailable

# JLL Package information
[jll]
package = "Qt5Base_jll"
auto_install = true  # Automatically install if not present

# Components (for libraries with multiple pieces)
[components]
Core = { jll_export = "libQt5Core", required = true }
Widgets = { jll_export = "libQt5Widgets", required = false }
Network = { jll_export = "libQt5Network", required = false }

# System fallback (pkg-config or manual)
[system]
pkg_config = "Qt5Core"  # For pkg-config detection
search_paths = ["/usr/lib/qt5", "/opt/qt5", "/usr/local/opt/qt5"]

# Compiler flags (appended to JLL-provided flags)
[flags]
compile = ["-fPIC", "-DQT_NO_DEBUG"]  # Extra compile flags
link = ["-Wl,-rpath,\$ORIGIN"]  # Extra linker flags
defines = { QT_CORE_LIB = "1", QT_GUI_LIB = "1" }

# Version requirements
[version]
minimum = "5.12.0"
maximum = "5.99.99"
```

### Advanced Module (`Boost.toml`)

```toml
[module]
name = "Boost"
cmake_name = "Boost"
description = "Boost C++ Libraries"

[resolution]
prefer = "jll"
fallback = ["system"]

[jll]
package = "boost_jll"
auto_install = true

# Boost has MANY components
[components]
system = { required = true, jll_export = "libboost_system" }
filesystem = { required = false, jll_export = "libboost_filesystem" }
regex = { required = false, jll_export = "libboost_regex" }
thread = { required = false, jll_export = "libboost_thread" }
iostreams = { required = false, jll_export = "libboost_iostreams" }
program_options = { required = false, jll_export = "libboost_program_options" }

# Component dependencies (filesystem needs system)
[component_deps]
filesystem = ["system"]
thread = ["system"]

[system]
pkg_config = "boost"
header_check = "boost/version.hpp"

[flags]
compile = ["-DBOOST_ALL_DYN_LINK"]
# Boost often needs specific flags per component
[flags.per_component]
thread = { compile = ["-pthread"], link = ["-lpthread"] }
regex = { link = ["-licuuc", "-licudata"] }  # ICU dependency

[version]
minimum = "1.70.0"
```

### Custom Build Module (`CustomLLVM.toml`)

```toml
[module]
name = "LLVM"
cmake_name = "LLVM"
description = "LLVM Compiler Infrastructure with custom build flags"

[resolution]
prefer = "custom"  # Force custom build
fallback = ["jll", "system"]

# Custom build instructions
[custom]
git_url = "https://github.com/llvm/llvm-project.git"
git_tag = "llvmorg-17.0.6"
build_script = "llvm_build.jl"  # Julia script that builds it

# Build configuration
[custom.cmake_args]
CMAKE_BUILD_TYPE = "Release"
LLVM_ENABLE_PROJECTS = "clang;lld;compiler-rt"
LLVM_TARGETS_TO_BUILD = "X86;ARM;AArch64"
LLVM_ENABLE_ASSERTIONS = "OFF"
LLVM_BUILD_LLVM_DYLIB = "ON"

[flags]
compile = ["-fno-rtti"]
link = ["-lLLVM-17"]

[version]
exact = "17.0.6"
```

## Usage in Projects

### In `replibuild.toml`

```toml
[project]
name = "MyQtApp"

# List required modules
[dependencies]
Qt5 = { components = ["Core", "Widgets", "Network"], version = ">=5.15" }
Boost = { components = ["system", "filesystem"], version = ">=1.75" }
Zlib = {}  # Use defaults

# Override module settings per-project
[dependencies.Qt5.flags]
compile = ["-DQT_DISABLE_DEPRECATED_BEFORE=0x060000"]

# Use custom build for specific dependency
[dependencies.LLVM]
resolution = "custom"  # Force custom build
custom_flags = ["-O3", "-march=native"]
```

### In Julia Code

```julia
using RepliBuild

# Initialize project (uses replibuild.toml)
config = RepliBuild.init("myproject")

# Modules are automatically resolved
qt5 = RepliBuild.get_module("Qt5")
println("Qt5 include dirs: ", qt5.include_dirs)
println("Qt5 libraries: ", qt5.libraries)

# Build with resolved dependencies
RepliBuild.build()
```

## Module Resolution Algorithm

```
1. Read replibuild.toml dependencies
2. For each dependency:
   a. Load module descriptor (Qt5.toml)
   b. Check resolution strategy:
      - If prefer="jll":
        1. Check if JLL package installed
        2. If not and auto_install=true: Pkg.add(package)
        3. Extract paths from JLL artifact
        4. Apply custom flags from module + project
      - If prefer="system":
        1. Try pkg-config
        2. Try manual library search
        3. Validate version
      - If prefer="custom":
        1. Check if already built (cached)
        2. If not: run custom build script
        3. Cache build artifacts
   c. Merge flags (module defaults + project overrides)
   d. Validate components exist
3. Generate final compiler/linker flags
4. Pass to build system
```

## Creating a New Module

### Quick Start Template

```bash
# Create new module
mkdir -p ~/.replibuild/modules
cd ~/.replibuild/modules

# Use template generator
julia -e 'using RepliBuild; RepliBuild.create_module_template("MyLibrary")'
```

This generates:

```toml
[module]
name = "MyLibrary"
cmake_name = "MyLibrary"
description = "Description of MyLibrary"

[resolution]
prefer = "jll"
fallback = ["system"]

[jll]
package = "MyLibrary_jll"
auto_install = true

[system]
pkg_config = "mylibrary"

[flags]
compile = []
link = []

[version]
minimum = "1.0.0"
```

### Module Development Workflow

1. **Create module descriptor**: `MyLib.toml`
2. **Test resolution**:
   ```julia
   using RepliBuild.ModuleRegistry
   mod = resolve_module("MyLib")
   println(mod)
   ```
3. **Validate flags**:
   ```julia
   # Create test project
   RepliBuild.init("test_mylib")
   # Add module to replibuild.toml
   # Build and verify
   RepliBuild.build()
   ```
4. **Publish module**:
   - Submit to RepliBuild module registry
   - Or keep in `~/.replibuild/modules/` for personal use

## Built-in Modules

RepliBuild ships with modules for common libraries:

- `Qt5.toml`, `Qt6.toml`
- `Boost.toml`
- `Eigen.toml`
- `OpenCV.toml`
- `LLVM.toml`
- `FFTW.toml`
- `HDF5.toml`
- `NetCDF.toml`
- `BLAS.toml`, `LAPACK.toml`

## Why This Is Revolutionary

### Before RepliBuild Modules

```cmake
# CMakeLists.txt
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)
# Hope it works across platforms...
# Hope the right version is installed...
# Hope system paths are configured...
```

### With RepliBuild Modules

```toml
# replibuild.toml
[dependencies]
Qt5 = { components = ["Core", "Widgets"] }
```

Done. Works everywhere. Reproducible. Cached. Version-locked.

### The Julia Advantage

1. **JLL packages** = Reproducible binaries (like conda, but better)
2. **Artifacts** = Cached, immutable, shared across projects
3. **Pkg** = Dependency resolution built-in
4. **Module descriptors** = Build logic + flags + components

Combined = **First-class C/C++ build system in Julia**

## Future: Module Registry

Create a centralized registry like Julia's General registry:

```
RepliBuildModules/
├── Registry.toml
├── Q/
│   └── Qt5/
│       ├── Package.toml
│       └── Versions.toml
├── B/
│   └── Boost/
│       ├── Package.toml
│       └── Versions.toml
└── ...
```

Install modules:
```julia
using RepliBuild
RepliBuild.add_module_registry("https://github.com/RepliBuild/Modules")
RepliBuild.install_module("Qt5")
```

## Summary

RepliBuild modules solve the "build logic" gap:

- JLL packages = **binaries** ✅
- RepliBuild modules = **binaries + build flags + logic** ✅✅✅

This is what Julia's C/C++ ecosystem has been missing.