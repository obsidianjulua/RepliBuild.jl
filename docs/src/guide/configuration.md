# Configuration Reference

Complete reference for `replibuild.toml` and `wrapper_config.toml` configuration files.

## replibuild.toml

Main configuration file for C++ compilation projects.

### [project] Section

Basic project metadata.

```toml
[project]
name = "MyProject"              # Project name (required)
version = "1.0.0"               # Semantic version
description = "My C++ library"  # Short description
authors = ["Your Name"]         # List of authors
license = "MIT"                 # License identifier
```

### [compilation] Section

Compilation settings for C++ source code.

```toml
[compilation]
# Source files
sources = [
    "src/main.cpp",
    "src/utils.cpp"
]

# Or scan directories
source_dirs = ["src", "lib"]
exclude_patterns = ["*_test.cpp", "*~"]

# Header files for binding generation
headers = [
    "include/api.h",
    "include/types.h"
]

# Include directories
include_dirs = [
    "include",
    "/usr/local/include"
]

# Libraries to link
link_libs = [
    "m",           # Math library
    "pthread",     # POSIX threads
    "stdc++"       # C++ standard library
]

# Library search paths
lib_dirs = [
    "/usr/local/lib",
    "/opt/mylib/lib"
]

# Linker flags
link_flags = [
    "-Wl,-rpath=/usr/local/lib",
    "-Wl,--no-undefined"
]

# C++ compiler flags
cxx_flags = [
    "-std=c++17",
    "-Wall",
    "-Wextra",
    "-O2"
]

# C++ standard (alternative to cxx_flags)
cxx_standard = "c++17"  # c++11, c++14, c++17, c++20, c++23

# Optimization level
optimization = "2"  # 0, 1, 2, 3, s (size), fast

# Debug symbols
debug = true  # -g

# Position independent code
pic = true    # -fPIC

# Preprocessor definitions
defines = [
    "NDEBUG",
    "MY_FEATURE=1",
    "VERSION=\"1.0\""
]

# Parallel compilation
parallel = true
num_jobs = 4  # 0 = all cores

# Precompiled header
precompiled_header = "include/pch.h"
```

### [output] Section

Output configuration.

```toml
[output]
# Library name
library_name = "libmyproject"

# Output directory
output_dir = "build"

# Julia module name
julia_module_name = "MyProject"

# Specific output libraries
libraries = [
    "build/lib/libcore.so",
    "build/lib/libutils.so"
]

# Or use patterns
library_patterns = ["build/**/*.so"]

# Headers for binding generation
headers = [
    "include/api.h",
    "build/generated/config.h"
]
```

### [bindings] Section

Binding generation options.

```toml
[bindings]
# Namespaces to wrap (empty = all)
namespaces = [
    "MyLib",
    "MyLib::Core"
]

# Functions to export (empty = all public)
export_functions = [
    "add",
    "multiply",
    "compute"
]

# Exclude specific functions
exclude_functions = [
    "internal_helper",
    "debug_*"  # Wildcards supported
]

# Classes to wrap
export_classes = [
    "Calculator",
    "Matrix",
    "Solver"
]

# Structs to wrap
export_structs = [
    "Vector3",
    "Color",
    "Point"
]

# Enums to wrap
export_enums = [
    "ErrorCode",
    "State"
]

# Export templates?
export_templates = false

# Generate high-level Julia API?
generate_high_level = true

# Binding style
style = "cxxwrap"  # or "ccall", "clangjl"
```

### [build] Section

Build system integration.

```toml
[build]
# Build system type
system = "cmake"  # cmake, qmake, meson, autotools, make

# Source directory
source_dir = "."

# Build directory
build_dir = "build"

# CMake options
cmake_options = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON"
]

# qmake options
qt_version = "Qt5"  # or "Qt6"
pro_file = "myapp.pro"

# Meson options
meson_options = [
    "-Dbuildtype=release"
]

# Autotools options
configure_options = [
    "--prefix=/usr/local",
    "--enable-shared"
]
bootstrap = false  # Run autoreconf first?

# Make options
make_targets = ["all"]
make_options = ["-j4"]
make_command = "make"  # or "gmake"

# Use Julia JLL packages?
use_jll = true

# Prefer system tools over JLL?
prefer_system = false

# JLL packages
[build.jll_packages]
cmake = "CMAKE_jll"
ninja = "Ninja_jll"
qt5 = "Qt5Base_jll"

# Environment variables
[build.environment]
CC = "clang"
CXX = "clang++"
CFLAGS = "-O3"
CXXFLAGS = "-O3"
```

### [dependencies] Section

External dependencies.

```toml
[dependencies]
# RepliBuild modules
modules = [
    "OpenCV",
    "Qt5",
    "Boost"
]

# Julia JLL packages
jll_packages = [
    "Zlib_jll",
    "OpenSSL_jll",
    "SQLite_jll"
]

# pkg-config packages
pkg_config = [
    "opencv4",
    "libpng",
    "gtk+-3.0"
]

# System packages (informational)
system_packages = [
    "libboost-dev",
    "libpng-dev"
]
```

### Platform-Specific Configuration

Override settings per platform.

```toml
# Linux-specific
[compilation.linux]
link_libs = ["rt", "dl"]
cxx_flags = ["-pthread"]
lib_dirs = ["/usr/lib/x86_64-linux-gnu"]

[build.linux]
cmake_options = ["-DLINUX=ON"]

# macOS-specific
[compilation.macos]
link_libs = []
cxx_flags = ["-framework", "CoreFoundation"]
lib_dirs = ["/usr/local/lib"]

[build.macos]
cmake_options = [
    "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15",
    "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64"
]

# Windows-specific
[compilation.windows]
link_libs = ["ws2_32", "bcrypt"]
cxx_flags = ["/EHsc", "/MD"]

[build.windows]
cmake_options = [
    "-G", "Visual Studio 16 2019",
    "-A", "x64"
]
```

### [test] Section

Testing configuration.

```toml
[test]
# Test source files
test_sources = [
    "test/test_main.cpp",
    "test/test_utils.cpp"
]

# Test executable name
test_executable = "test_runner"

# Test frameworks
frameworks = ["catch2", "gtest"]

# Run tests after build?
auto_run = true
```

## wrapper_config.toml

Configuration for binary wrapping projects.

### [wrapper] Section

Main wrapper settings.

```toml
[wrapper]
# Directories to scan for binaries
scan_dirs = [
    "lib",
    "bin",
    "/usr/local/lib"
]

# Output directory for wrappers
output_dir = "julia_wrappers"

# Generate high-level API?
generate_high_level = true

# Symbol filtering (regex)
include_patterns = [
    "^mylib_",
    "^MYLIB_"
]

exclude_patterns = [
    "^_internal",
    "^test_",
    "debug"
]

# Platform detection
auto_detect_platform = true
```

### [library.*] Sections

Per-library configuration.

```toml
[library.libmath]
# Library path
path = "lib/libmath.so"

# Or platform-specific
[library.libmath.linux]
path = "lib/linux/libmath.so"

[library.libmath.macos]
path = "lib/macos/libmath.dylib"

[library.libmath.windows]
path = "lib/windows/math.dll"

# Module name
module_name = "LibMath"

# Functions to export (empty = all)
exports = [
    "sin",
    "cos",
    "sqrt"
]

# Exclude functions
excludes = [
    "internal_*"
]

# Include patterns
include_patterns = ["^math_"]
exclude_patterns = ["^_"]
```

### Function Signatures

Annotate function signatures for better wrappers.

```toml
[library.libmath.functions.add]
return_type = "Cdouble"
arg_types = ["Cdouble", "Cdouble"]

[library.libmath.functions.multiply]
return_type = "Cdouble"
arg_types = ["Cdouble", "Cdouble"]

[library.libmath.functions.create_vector]
return_type = "Ptr{Cvoid}"
arg_types = ["Cint"]

[library.libmath.functions.process_array]
return_type = "Cvoid"
arg_types = ["Ptr{Cdouble}", "Csize_t"]
```

### Struct Definitions

Define C struct layouts.

```toml
[library.libmath.structs.Vector3]
fields = [
    { name = "x", type = "Cdouble" },
    { name = "y", type = "Cdouble" },
    { name = "z", type = "Cdouble" }
]

[library.libmath.structs.Matrix4x4]
fields = [
    { name = "data", type = "NTuple{16, Cfloat}" }
]

[library.libmath.structs.ComplexNumber]
fields = [
    { name = "real", type = "Cdouble" },
    { name = "imag", type = "Cdouble" }
]
```

### Callback Definitions

Define callback function signatures.

```toml
[library.libmath.callbacks.CompareFunc]
signature = "Cint (Ptr{Cvoid}, Ptr{Cvoid})"

[library.libmath.callbacks.ProcessFunc]
signature = "Cvoid (Ptr{Cdouble}, Csize_t, Ptr{Cvoid})"
```

### Enum Definitions

```toml
[library.libmath.enums.ErrorCode]
type = "Cint"
values = [
    { name = "SUCCESS", value = 0 },
    { name = "ERROR_INVALID", value = 1 },
    { name = "ERROR_OVERFLOW", value = 2 }
]

[library.libmath.enums.State]
type = "Cuint"
values = [
    { name = "IDLE", value = 0 },
    { name = "RUNNING", value = 1 },
    { name = "STOPPED", value = 2 }
]
```

## Module Templates

External library module configuration (`~/.replibuild/modules/OpenCV.toml`).

```toml
[module]
name = "OpenCV"
version = "4.5.0"
description = "OpenCV computer vision library"

[library]
# Library names
lib_names = [
    "opencv_core",
    "opencv_imgproc",
    "opencv_highgui"
]

# Include directories
include_dirs = [
    "/usr/include/opencv4",
    "/usr/local/include/opencv4"
]

# Link flags
link_flags = [
    "-lopencv_core",
    "-lopencv_imgproc",
    "-lopencv_highgui"
]

# Preprocessor defines
defines = ["USE_OPENCV"]

[package]
# Use pkg-config for detection
pkg_config = "opencv4"

# Or CMake
cmake_package = "OpenCV"
cmake_components = ["core", "imgproc"]

# Or JLL
jll_package = "OpenCV_jll"

[detection]
# Header to check for availability
header = "opencv2/opencv.hpp"

# Function to check for linking
symbol = "cv::Mat"

# Minimum version
min_version = "4.0.0"
```

## Global Configuration

User-wide settings (`~/.replibuild/config.toml`).

```toml
[global]
# Default compiler
default_compiler = "clang++"

# Default C++ standard
default_cxx_standard = "c++17"

# Default optimization level
default_optimization = "2"

# Number of parallel jobs
default_jobs = 0  # 0 = all cores

[paths]
# Module search paths
module_paths = [
    "~/.replibuild/modules",
    "/usr/share/replibuild/modules"
]

# Cache directory
cache_dir = "~/.replibuild/cache"

[llvm]
# LLVM installation
llvm_dir = "/usr/lib/llvm-14"

# Prefer system LLVM over JLL?
prefer_system = false

[defaults]
# Default to JLL packages?
use_jll = true

# Generate high-level API by default?
generate_high_level = true

# Default binding style
binding_style = "cxxwrap"

[error_learning]
# Error database path
db_path = "~/.replibuild/error_db.sqlite"

# Enable error learning?
enabled = true

# Export errors for analysis?
auto_export = false
```

## Type Reference

### C to Julia Type Mapping

| C Type | Julia Type | Size |
|--------|-----------|------|
| `char` | `Cchar` | 1 byte |
| `unsigned char` | `Cuchar` | 1 byte |
| `short` | `Cshort` | 2 bytes |
| `unsigned short` | `Cushort` | 2 bytes |
| `int` | `Cint` | 4 bytes |
| `unsigned int` | `Cuint` | 4 bytes |
| `long` | `Clong` | 4/8 bytes |
| `unsigned long` | `Culong` | 4/8 bytes |
| `long long` | `Clonglong` | 8 bytes |
| `float` | `Cfloat` | 4 bytes |
| `double` | `Cdouble` | 8 bytes |
| `void*` | `Ptr{Cvoid}` | ptr size |
| `char*` | `Ptr{Cchar}` | ptr size |
| `size_t` | `Csize_t` | ptr size |
| `ptrdiff_t` | `Cptrdiff_t` | ptr size |

### Common Patterns

**C string:**
```toml
arg_types = ["Ptr{Cchar}"]  # const char*
```

**Output parameter:**
```toml
arg_types = ["Ptr{Cint}"]  # int* output
```

**Array:**
```toml
arg_types = ["Ptr{Cdouble}", "Csize_t"]  # double* arr, size_t len
```

**Callback:**
```toml
arg_types = ["Ptr{Cvoid}"]  # function pointer
```

## Configuration Validation

Validate configuration:

```julia
using RepliBuild

# Validate replibuild.toml
config = RepliBuild.load_config("replibuild.toml")
RepliBuild.validate_config(config)

# Check for common issues
RepliBuild.check_config("replibuild.toml")
```

## Examples

### Minimal C++ Project

```toml
[project]
name = "Simple"

[compilation]
sources = ["src/main.cpp"]
headers = ["include/api.h"]
include_dirs = ["include"]

[output]
library_name = "libsimple"
```

### Full-Featured Project

```toml
[project]
name = "AdvancedLib"
version = "2.1.0"
description = "Advanced C++ library with many features"

[compilation]
source_dirs = ["src"]
headers = ["include/api.h", "include/types.h"]
include_dirs = ["include", "/usr/local/include"]
link_libs = ["boost_system", "pthread"]
cxx_standard = "c++20"
optimization = "3"
defines = ["NDEBUG", "PRODUCTION"]
parallel = true

[bindings]
namespaces = ["AdvancedLib"]
export_classes = ["Engine", "Processor"]
generate_high_level = true

[dependencies]
modules = ["Boost"]
jll_packages = ["Boost_jll"]

[build]
system = "cmake"
cmake_options = ["-DCMAKE_BUILD_TYPE=Release"]

[output]
library_name = "libadvanced"
julia_module_name = "AdvancedLib"
```

## Next Steps

- **[C++ Workflow](cpp-workflow.md)**: Apply configuration to C++ projects
- **[Build Systems](build-systems.md)**: Build system integration
- **[Module System](modules.md)**: External library modules
- **[Examples](../examples/simple-cpp.md)**: Real-world configurations
