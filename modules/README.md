# RepliBuild Module Registry

This directory contains **module descriptors** for external C/C++ libraries.

## What Are Modules?

Modules bridge the gap between:
- **JLL packages** (prebuilt binaries from Julia ecosystem)
- **Build logic** (compiler flags, components, version constraints)

A module tells RepliBuild:
1. How to find the library (JLL → system → custom build)
2. What components are available
3. What compiler/linker flags are needed
4. How to validate versions

## Available Modules

Built-in modules provided by RepliBuild:

| Module | Type | Description |
|--------|------|-------------|
| `Qt5.toml` | Framework | Qt5 cross-platform GUI framework |
| `Boost.toml` | Library Collection | Boost C++ libraries (30+ components) |
| `Eigen.toml` | Header-only | Linear algebra template library |
| `Zlib.toml` | Simple Library | Compression library |

## Using Modules

In your `replibuild.toml`:

```toml
[dependencies]
Qt5 = { components = ["Core", "Widgets"], version = ">=5.15" }
Boost = { components = ["system", "filesystem"] }
Eigen = {}  # Header-only, no components needed
```

Then:

```julia
using RepliBuild

# Modules are automatically resolved
config = RepliBuild.init("myproject")
RepliBuild.build()
```

## Creating Your Own Modules

### Quick Start

```julia
using RepliBuild

# Generate template
create_module_template("MyLibrary")

# Or from pkg-config
generate_from_pkg_config("sdl2")

# Or from CMake
generate_from_cmake("OpenCV")
```

### Module Structure

See [MODULE_SYSTEM.md](../docs/MODULE_SYSTEM.md) for full documentation.

Minimal example:

```toml
[module]
name = "MyLib"
cmake_name = "MyLib"
description = "My awesome library"

[resolution]
prefer = "jll"

[jll]
package = "MyLib_jll"
auto_install = true

[flags]
compile = []
link = []
```

## Custom Module Locations

RepliBuild searches for modules in:

1. **Built-in**: `RepliBuild.jl/modules/` (this directory)
2. **User modules**: `~/.replibuild/modules/`
3. **Project modules**: `<project>/.replibuild/modules/`

Place your custom modules in `~/.replibuild/modules/` for reuse across projects.

## Contributing Modules

Want to add a module to the built-in registry?

1. Create module descriptor: `MyLib.toml`
2. Test with a real project
3. Submit PR to RepliBuild.jl with:
   - Module file
   - Example usage
   - Testing notes

## Examples

### Simple Library (Zlib)

```toml
[module]
name = "Zlib"
cmake_name = "ZLIB"

[jll]
package = "Zlib_jll"

[components]
zlib = { jll_export = "libz", required = true }
```

### Multi-Component Library (Qt5)

```toml
[module]
name = "Qt5"

[components]
Core = { jll_export = "libQt5Core", required = true }
Widgets = { jll_export = "libQt5Widgets", required = false }

[component_deps]
Widgets = ["Core"]  # Widgets needs Core
```

### Header-Only Library (Eigen)

```toml
[module]
name = "Eigen"

[components]
headers = { required = true, header_only = true }

[metadata]
header_only = true
```

### Custom Build

```toml
[module]
name = "MyCustomLib"

[resolution]
prefer = "custom"

[custom]
git_url = "https://github.com/user/lib.git"
git_tag = "v1.0.0"
build_script = "build_mycustomlib.jl"

[custom.cmake_args]
CMAKE_BUILD_TYPE = "Release"
BUILD_SHARED_LIBS = "ON"
```

## Troubleshooting

**Module not found:**
```julia
julia> ModuleRegistry.resolve_module("MyLib")
# Check search paths
julia> ModuleRegistry.MODULE_SEARCH_PATHS
```

**JLL package not installing:**
```julia
# Install manually
julia> using Pkg; Pkg.add("MyLib_jll")
```

**Wrong version resolved:**
```toml
# In replibuild.toml
[dependencies.MyLib]
version = ">=2.0.0"
resolution = "system"  # Force system instead of JLL
```

## Future: Shared Registry

Coming soon: A centralized registry of community-contributed modules, similar to Julia's General registry.

```julia
# Future API
using RepliBuild
RepliBuild.add_registry("https://github.com/RepliBuild/Modules")
RepliBuild.install_module("OpenCV")
```

## Learn More

- [Full Documentation](../docs/MODULE_SYSTEM.md)
- [Examples](../examples/)
- [Contributing Guide](../CONTRIBUTING.md)
