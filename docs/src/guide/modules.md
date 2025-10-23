# Module System

RepliBuild's module system manages external library dependencies with automatic resolution and configuration.

## Overview

Modules are reusable configuration templates for common libraries (OpenCV, Qt, Boost, etc.) that:

- Define library locations and compile flags
- Support pkg-config, CMake, and JLL package detection
- Provide platform-specific configurations
- Auto-detect library availability

## Quick Start

```julia
using RepliBuild

# List available modules
RepliBuild.list_modules()

# Use a module in your project
```

In `replibuild.toml`:

```toml
[dependencies]
modules = ["OpenCV", "Qt5"]
```

## Built-in Modules

RepliBuild includes templates for common libraries:

- **OpenCV** - Computer vision
- **Qt5/Qt6** - GUI framework
- **Boost** - C++ utilities
- **Eigen** - Linear algebra
- **SFML** - Multimedia
- **SDL2** - Game development
- **GLFW** - OpenGL windowing
- **SQLite** - Database

## Using Modules

### In replibuild.toml

```toml
[dependencies]
modules = ["OpenCV"]

[compilation]
sources = ["src/vision.cpp"]
headers = ["include/vision.h"]
include_dirs = ["include"]
```

RepliBuild automatically adds:
- OpenCV include paths
- Link flags for OpenCV libraries
- Required defines

### Resolving Modules

```julia
# Get module information
info = RepliBuild.get_module_info("OpenCV")

println("Include dirs: ", info.include_dirs)
println("Libraries: ", info.lib_names)
println("Link flags: ", info.link_flags)
```

### Module Resolution Order

1. **JLL packages** (if available and preferred)
2. **pkg-config** (if configured)
3. **CMake** find_package (if configured)
4. **System paths** (fallback)

## Creating Module Templates

### Module File Structure

Create `~/.replibuild/modules/MyLib.toml`:

```toml
[module]
name = "MyLib"
version = "1.0.0"
description = "My external library"

[library]
# Library names (without lib prefix or extension)
lib_names = ["mylib", "mylib_utils"]

# Include directories
include_dirs = [
    "/usr/include/mylib",
    "/usr/local/include/mylib"
]

# Link flags
link_flags = [
    "-lmylib",
    "-lmylib_utils"
]

# Preprocessor defines
defines = ["USE_MYLIB", "MYLIB_VERSION=1"]

[package]
# pkg-config package name
pkg_config = "mylib"

# Or CMake package
cmake_package = "MyLib"
cmake_components = ["core", "utils"]

# Or Julia JLL
jll_package = "MyLib_jll"

[detection]
# Header to check
header = "mylib/api.h"

# Symbol to check
symbol = "mylib_init"

# Minimum version
min_version = "1.0.0"

# Version check command
version_command = "mylib-config --version"
```

### Register Module

```julia
using RepliBuild

# Register module template
RepliBuild.register_module("MyLib", "/path/to/MyLib.toml")

# Or create from scratch
RepliBuild.create_module_template("MyLib")
```

## Module from pkg-config

Auto-generate module from pkg-config:

```julia
# Generate module for OpenCV
RepliBuild.generate_from_pkg_config("opencv4", "OpenCV")
```

This creates `~/.replibuild/modules/OpenCV.toml`:

```toml
[module]
name = "OpenCV"
version = "4.5.4"  # Auto-detected

[library]
lib_names = ["opencv_core", "opencv_imgproc"]  # From pkg-config --libs
include_dirs = ["/usr/include/opencv4"]  # From pkg-config --cflags

[package]
pkg_config = "opencv4"
```

## Module from CMake

Generate from CMake find_package:

```julia
# Generate module for Boost
RepliBuild.generate_from_cmake("Boost",
    components=["system", "filesystem"])
```

Creates module with CMake detection:

```toml
[module]
name = "Boost"

[package]
cmake_package = "Boost"
cmake_components = ["system", "filesystem"]
cmake_version = "1.70.0"

[library]
# Auto-filled when CMake finds package
lib_names = ["boost_system", "boost_filesystem"]
```

## Platform-Specific Modules

```toml
[module]
name = "MyLib"

# Linux configuration
[library.linux]
lib_names = ["mylib"]
include_dirs = ["/usr/include/mylib"]
lib_dirs = ["/usr/lib/x86_64-linux-gnu"]

# macOS configuration
[library.macos]
lib_names = ["mylib"]
include_dirs = ["/usr/local/include/mylib"]
lib_dirs = ["/usr/local/lib"]
link_flags = ["-framework", "CoreFoundation"]

# Windows configuration
[library.windows]
lib_names = ["mylib"]
include_dirs = ["C:/Program Files/MyLib/include"]
lib_dirs = ["C:/Program Files/MyLib/lib"]

# Detection (platform-independent)
[package]
pkg_config = "mylib"
```

## Module Composition

Modules can depend on other modules:

```toml
[module]
name = "MyApp"

[dependencies]
# Require these modules
modules = ["OpenCV", "Qt5"]

[library]
lib_names = ["myapp"]
include_dirs = ["include"]

# Inherits include_dirs and link_flags from OpenCV and Qt5
```

## Advanced Module Features

### Version Requirements

```toml
[dependencies]
modules = [
    { name = "OpenCV", version = ">=4.0" },
    { name = "Qt5", version = "~5.15" },  # 5.15.x
    { name = "Boost", version = "^1.70" }  # >=1.70, <2.0
]
```

### Optional Modules

```toml
[dependencies]
# Required modules
modules = ["Qt5"]

# Optional modules
[dependencies.optional]
OpenCV = { feature = "vision", default = false }
CUDA = { feature = "gpu", default = false }
```

Build with features:

```julia
RepliBuild.compile(features=["vision", "gpu"])
```

### Module Variants

```toml
[module]
name = "OpenCV"

# Standard variant
[variants.standard]
lib_names = ["opencv_core", "opencv_imgproc"]

# Full variant
[variants.full]
lib_names = [
    "opencv_core",
    "opencv_imgproc",
    "opencv_video",
    "opencv_ml",
    "opencv_objdetect"
]

# Minimal variant
[variants.minimal]
lib_names = ["opencv_core"]
```

Use variant:

```toml
[dependencies]
modules = [
    { name = "OpenCV", variant = "full" }
]
```

## Module Discovery

### Search Paths

```julia
# Get module search paths
paths = RepliBuild.get_module_search_paths()

# Default paths:
# - ~/.replibuild/modules
# - /usr/share/replibuild/modules
# - /usr/local/share/replibuild/modules
```

### Add Search Path

```toml
# ~/.replibuild/config.toml
[paths]
module_paths = [
    "~/.replibuild/modules",
    "/opt/mycompany/replibuild/modules",
    "~/projects/modules"
]
```

### List Available Modules

```julia
# List all modules
modules = RepliBuild.list_modules()

for mod in modules
    println("$(mod.name) v$(mod.version) - $(mod.description)")
end
```

## Example Modules

### OpenCV Module

```toml
[module]
name = "OpenCV"
version = "4.5.0"
description = "OpenCV computer vision library"

[library]
lib_names = [
    "opencv_core",
    "opencv_imgproc",
    "opencv_highgui",
    "opencv_video"
]

include_dirs = [
    "/usr/include/opencv4",
    "/usr/local/include/opencv4"
]

link_flags = [
    "-lopencv_core",
    "-lopencv_imgproc",
    "-lopencv_highgui",
    "-lopencv_video"
]

defines = ["USE_OPENCV"]

[package]
pkg_config = "opencv4"
cmake_package = "OpenCV"
cmake_components = ["core", "imgproc", "highgui", "video"]
jll_package = "OpenCV_jll"

[detection]
header = "opencv2/opencv.hpp"
min_version = "4.0.0"
```

### Qt5 Module

```toml
[module]
name = "Qt5"
version = "5.15.0"
description = "Qt5 GUI framework"

[library]
lib_names = [
    "Qt5Core",
    "Qt5Gui",
    "Qt5Widgets"
]

[library.linux]
include_dirs = [
    "/usr/include/qt5",
    "/usr/include/qt5/QtCore",
    "/usr/include/qt5/QtGui",
    "/usr/include/qt5/QtWidgets"
]

[library.macos]
include_dirs = [
    "/usr/local/opt/qt5/include",
    "/usr/local/opt/qt5/include/QtCore",
    "/usr/local/opt/qt5/include/QtGui",
    "/usr/local/opt/qt5/include/QtWidgets"
]

[package]
pkg_config = "Qt5Core Qt5Gui Qt5Widgets"
cmake_package = "Qt5"
cmake_components = ["Core", "Gui", "Widgets"]
jll_package = "Qt5Base_jll"

[detection]
header = "QtCore/QObject"
version_command = "qmake -query QT_VERSION"
```

### Boost Module

```toml
[module]
name = "Boost"
version = "1.70.0"
description = "Boost C++ libraries"

[library]
lib_names = [
    "boost_system",
    "boost_filesystem",
    "boost_thread"
]

[library.linux]
include_dirs = ["/usr/include"]
lib_dirs = ["/usr/lib/x86_64-linux-gnu"]

[library.macos]
include_dirs = ["/usr/local/include"]
lib_dirs = ["/usr/local/lib"]

[package]
cmake_package = "Boost"
cmake_components = ["system", "filesystem", "thread"]
jll_package = "Boost_jll"

[detection]
header = "boost/version.hpp"
min_version = "1.60.0"
```

## Troubleshooting

### Module Not Found

**Error:**
```
ERROR: Module 'OpenCV' not found
```

**Solution:**

```julia
# Check module search paths
RepliBuild.print_paths_info()

# List available modules
RepliBuild.list_modules()

# Create module if needed
RepliBuild.create_module_template("OpenCV")
```

### Library Not Detected

**Error:**
```
WARNING: OpenCV not found on system
```

**Solution:**

1. Install library:
```bash
# Ubuntu
sudo apt-get install libopencv-dev

# macOS
brew install opencv
```

2. Or use JLL:
```toml
[dependencies]
jll_packages = ["OpenCV_jll"]
```

3. Or specify manually:
```toml
[library.OpenCV]
include_dirs = ["/custom/path/include"]
lib_dirs = ["/custom/path/lib"]
```

### Version Conflict

**Error:**
```
ERROR: OpenCV version 3.4.0 < required 4.0.0
```

**Solution:**

Update library or adjust requirement:

```toml
[dependencies]
modules = [
    { name = "OpenCV", version = ">=3.0" }
]
```

## Module Development

### Testing Module

```julia
# Test module resolution
module_info = RepliBuild.resolve_module("OpenCV")

if module_info !== nothing
    println("✅ Module found")
    println("Include dirs: ", module_info.include_dirs)
    println("Libraries: ", module_info.lib_names)
else
    println("❌ Module not found")
end
```

### Debugging Module

```julia
# Enable verbose module resolution
ENV["REPLIBUILD_DEBUG"] = "1"

info = RepliBuild.resolve_module("OpenCV")
```

Shows:
```
🔍 Searching for module: OpenCV
  Checking JLL: OpenCV_jll... not found
  Checking pkg-config: opencv4... found
  ✅ Resolved via pkg-config
```

## Best Practices

### 1. Prefer JLL Packages

```toml
[package]
jll_package = "MyLib_jll"  # Reproducible, cross-platform
```

### 2. Provide Fallbacks

```toml
[package]
jll_package = "OpenCV_jll"      # First choice
pkg_config = "opencv4"          # Second choice
cmake_package = "OpenCV"        # Third choice
```

### 3. Platform-Specific Only When Needed

```toml
# Good: Common configuration
[library]
lib_names = ["mylib"]

# Only override when different
[library.windows]
lib_names = ["mylib", "ws2_32"]  # Extra libs on Windows
```

### 4. Document Requirements

```toml
[module]
name = "MyLib"
description = "My library (requires version >= 2.0)"

[detection]
min_version = "2.0.0"
```

## Next Steps

- **[Configuration](configuration.md)**: Module configuration reference
- **[Build Systems](build-systems.md)**: Integration with build systems
- **[Examples](../examples/multi-module.md)**: Multi-module projects
