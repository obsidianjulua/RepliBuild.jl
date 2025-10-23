# Module Registry API

Functions for managing external library modules.

## Module Functions

```@docs
RepliBuild.resolve_module
RepliBuild.list_modules
RepliBuild.register_module
RepliBuild.get_module_info
RepliBuild.create_module_template
RepliBuild.generate_from_pkg_config
RepliBuild.generate_from_cmake
```

## Function Reference

### resolve_module

```julia
resolve_module(module_name::String)
```

Resolve a module by name and return its configuration.

**Resolution order:**
1. JLL packages (if configured)
2. pkg-config (if configured)
3. CMake find_package (if configured)
4. System paths (fallback)

**Arguments:**
- `module_name::String`: Name of module to resolve

**Returns:** Module configuration dict, or `nothing` if not found

**Examples:**
```julia
info = RepliBuild.resolve_module("OpenCV")
if info !== nothing
    println("Include dirs: ", info.include_dirs)
    println("Libraries: ", info.lib_names)
    println("Link flags: ", info.link_flags)
end
```

---

### list_modules

```julia
list_modules()
```

List all available modules in search paths.

**Returns:** Array of module information dicts

**Examples:**
```julia
modules = RepliBuild.list_modules()

for mod in modules
    println("$(mod.name) v$(mod.version)")
    println("  $(mod.description)")
end
```

Output:
```
OpenCV v4.5.0
  OpenCV computer vision library
Qt5 v5.15.2
  Qt5 GUI framework
Boost v1.70.0
  Boost C++ libraries
```

---

### register_module

```julia
register_module(name::String, config_path::String)
```

Register a module template from a file.

**Arguments:**
- `name::String`: Module name
- `config_path::String`: Path to module TOML file

**Returns:** Nothing

**Examples:**
```julia
# Register custom module
RepliBuild.register_module("MyLib", "/path/to/MyLib.toml")

# Verify registration
modules = RepliBuild.list_modules()
@assert "MyLib" in [m.name for m in modules]
```

---

### get_module_info

```julia
get_module_info(module_name::String)
```

Get detailed information about a specific module.

**Arguments:**
- `module_name::String`: Module name

**Returns:** Dict with module metadata and configuration

**Examples:**
```julia
info = RepliBuild.get_module_info("OpenCV")

println("Name: ", info.name)
println("Version: ", info.version)
println("Description: ", info.description)
println("Libraries: ", info.lib_names)
println("Include dirs: ", info.include_dirs)
println("Detection method: ", info.detection_method)
```

---

### create_module_template

```julia
create_module_template(name::String; output::String="")
```

Create a new module template interactively or with defaults.

**Arguments:**
- `name::String`: Module name
- `output::String`: Output file path (default: `~/.replibuild/modules/<name>.toml`)

**Returns:** Path to created template

**Examples:**
```julia
# Create template interactively
path = RepliBuild.create_module_template("MyLib")

# Edit the generated template
# Then register it
RepliBuild.register_module("MyLib", path)
```

Generated template:
```toml
[module]
name = "MyLib"
version = "1.0.0"
description = "My external library"

[library]
lib_names = []
include_dirs = []
link_flags = []

[package]
pkg_config = ""
cmake_package = ""
jll_package = ""

[detection]
header = ""
min_version = ""
```

---

### generate_from_pkg_config

```julia
generate_from_pkg_config(pkg_name::String, module_name::String="")
```

Auto-generate module template from pkg-config.

**Arguments:**
- `pkg_name::String`: pkg-config package name
- `module_name::String`: Module name (default: same as pkg_name)

**Returns:** Path to generated module file

**Examples:**
```julia
# Generate OpenCV module from pkg-config
RepliBuild.generate_from_pkg_config("opencv4", "OpenCV")

# Verify
info = RepliBuild.get_module_info("OpenCV")
println("Include dirs: ", info.include_dirs)
```

**Requires:** `pkg-config` command available

Runs:
```bash
pkg-config --modversion opencv4
pkg-config --cflags opencv4
pkg-config --libs opencv4
```

---

### generate_from_cmake

```julia
generate_from_cmake(package_name::String;
                    components::Vector{String}=[],
                    module_name::String="")
```

Auto-generate module template from CMake find_package.

**Arguments:**
- `package_name::String`: CMake package name
- `components::Vector{String}`: CMake components (optional)
- `module_name::String`: Module name (default: same as package_name)

**Returns:** Path to generated module file

**Examples:**
```julia
# Generate Boost module
RepliBuild.generate_from_cmake("Boost",
                               components=["system", "filesystem"],
                               module_name="Boost")

# Generate Qt5 module
RepliBuild.generate_from_cmake("Qt5",
                               components=["Core", "Gui", "Widgets"],
                               module_name="Qt5")
```

**Requires:** `cmake` command available

---

## Path Management

### get_module_search_paths

```julia
get_module_search_paths()
```

Get all directories searched for modules.

**Returns:** Array of directory paths

**Examples:**
```julia
paths = RepliBuild.get_module_search_paths()

for path in paths
    println("Searching: $path")
end
```

Default paths:
```
~/.replibuild/modules
/usr/share/replibuild/modules
/usr/local/share/replibuild/modules
```

---

## Module Configuration Format

Module TOML structure:

```toml
[module]
name = "LibraryName"
version = "1.0.0"
description = "Library description"

[library]
# Library names (without lib prefix/extension)
lib_names = ["mylib", "mylib_utils"]

# Include directories
include_dirs = [
    "/usr/include/mylib",
    "/usr/local/include/mylib"
]

# Link flags
link_flags = ["-lmylib", "-lmylib_utils"]

# Library search directories (optional)
lib_dirs = ["/usr/local/lib"]

# Preprocessor defines (optional)
defines = ["USE_MYLIB"]

# Compiler flags (optional)
cxx_flags = ["-std=c++17"]

# Platform-specific (optional)
[library.linux]
lib_names = ["mylib"]
include_dirs = ["/usr/include/mylib"]

[library.macos]
lib_names = ["mylib"]
include_dirs = ["/usr/local/include/mylib"]

[library.windows]
lib_names = ["mylib"]
include_dirs = ["C:/Program Files/MyLib/include"]

[package]
# Detection methods
pkg_config = "mylib"                    # pkg-config package
cmake_package = "MyLib"                 # CMake package
cmake_components = ["core", "utils"]    # CMake components
jll_package = "MyLib_jll"              # Julia JLL package

[detection]
# Availability checks
header = "mylib/api.h"                 # Header to check
symbol = "mylib_init"                  # Symbol to check
min_version = "1.0.0"                  # Minimum version
version_command = "mylib-config --version"  # Version command

[dependencies]
# Module dependencies
modules = ["OtherModule"]
```

---

## Examples

### Using Modules in Projects

```toml
# replibuild.toml
[project]
name = "MyApp"

[dependencies]
modules = ["OpenCV", "Qt5"]

[compilation]
sources = ["src/main.cpp"]
include_dirs = ["include"]
# OpenCV and Qt5 include_dirs automatically added
# Link flags automatically added
```

### Creating Custom Module

```julia
# Create module template
RepliBuild.create_module_template("CustomLib")

# Edit ~/.replibuild/modules/CustomLib.toml
# Add library configuration

# Use in project
```

```toml
[dependencies]
modules = ["CustomLib"]
```

### Module with Variants

```toml
[module]
name = "OpenCV"

[variants.minimal]
lib_names = ["opencv_core"]

[variants.standard]
lib_names = ["opencv_core", "opencv_imgproc"]

[variants.full]
lib_names = [
    "opencv_core",
    "opencv_imgproc",
    "opencv_video",
    "opencv_ml"
]
```

Use in project:

```toml
[dependencies]
modules = [
    { name = "OpenCV", variant = "full" }
]
```

---

## Debugging

### Verbose Module Resolution

```julia
ENV["REPLIBUILD_DEBUG"] = "1"
info = RepliBuild.resolve_module("OpenCV")
```

Output:
```
🔍 Resolving module: OpenCV
  Checking JLL: OpenCV_jll... not found
  Checking pkg-config: opencv4... found ✅
  Include dirs: /usr/include/opencv4
  Libraries: opencv_core, opencv_imgproc
  Link flags: -lopencv_core -lopencv_imgproc
✅ Module resolved via pkg-config
```

---

## See Also

- **[Module System Guide](../guide/modules.md)**: Complete module guide
- **[Configuration](../guide/configuration.md)**: Configuration reference
- **[Core API](core.md)**: Core functions
