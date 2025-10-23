# Build System Integration

RepliBuild integrates with existing build systems (CMake, qmake, Meson, Autotools) to build C/C++ projects and generate Julia bindings.

## Overview

Instead of replacing your build system, RepliBuild **delegates** to it:

1. RepliBuild detects or uses configured build system
2. Runs build using Julia artifacts (JLL packages) or system tools
3. Locates built libraries
4. Generates Julia bindings

Supported build systems:
- **CMake** (3.10+)
- **qmake** (Qt5/Qt6)
- **Meson** + Ninja
- **Autotools** (./configure && make)
- **Make** (standalone Makefiles)

## Universal Build Function

```julia
using RepliBuild

# Auto-detect and build
RepliBuild.build()

# Build specific directory
RepliBuild.build("/path/to/project")

# With custom config
RepliBuild.build(".", config_file="custom.toml")
```

## CMake Projects

### Auto-Detection

RepliBuild detects CMake if `CMakeLists.txt` exists:

```julia
cd("cmake_project")
RepliBuild.build()  # Automatically uses CMake
```

### Configuration

```toml
[project]
name = "MyCMakeProject"

[build]
system = "cmake"
build_dir = "build"
cmake_options = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DCMAKE_INSTALL_PREFIX=/usr/local"
]

[output]
julia_module_name = "MyCMakeLib"
```

### Example: Building OpenCV

```julia
RepliBuild.init("opencv_project")
cd("opencv_project")
```

```toml
[build]
system = "cmake"
source_dir = "/path/to/opencv"
build_dir = "build"

cmake_options = [
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_TESTS=OFF",
    "-DWITH_CUDA=ON"
]

[output]
julia_module_name = "OpenCV"
libraries = ["opencv_core", "opencv_imgproc", "opencv_highgui"]
```

```julia
RepliBuild.build()
```

### CMake Import

Convert existing CMake project:

```julia
# Parse CMakeLists.txt and generate replibuild.toml
RepliBuild.import_cmake("path/to/CMakeLists.txt")

# Import specific target
RepliBuild.import_cmake("opencv/CMakeLists.txt", target="opencv_core")
```

Generated `replibuild.toml`:

```toml
[project]
name = "opencv_core"

[build]
system = "cmake"
source_dir = "opencv"

[compilation]
sources = [
    "src/core.cpp",
    "src/matrix.cpp",
    # ... (extracted from CMake)
]
include_dirs = [
    "include",
    "modules/core/include"
]

[output]
library_name = "libopencv_core"
```

### CMake with JLL Packages

Use Julia artifacts for reproducibility:

```toml
[build]
system = "cmake"
use_jll = true  # Use CMAKE_jll instead of system cmake

[build.jll_packages]
cmake = "CMAKE_jll"
ninja = "Ninja_jll"
```

RepliBuild automatically uses JLL if in Julia environment.

## qmake / Qt Projects

### Configuration

```toml
[project]
name = "QtApp"

[build]
system = "qmake"
qt_version = "Qt5"  # or "Qt6"
pro_file = "myapp.pro"
build_dir = "build"

[output]
julia_module_name = "QtApp"
```

### Example: Qt Application

```julia
RepliBuild.init("qt_calculator")
cd("qt_calculator")
```

Create `calculator.pro`:

```qmake
QT += core gui widgets

TARGET = calculator
TEMPLATE = lib
CONFIG += shared

SOURCES += src/calculator.cpp
HEADERS += include/calculator.h
INCLUDEPATH += include
```

```toml
[build]
system = "qmake"
qt_version = "Qt5"
pro_file = "calculator.pro"

[output]
julia_module_name = "Calculator"
```

```julia
RepliBuild.build()
```

### Qt with JLL

```toml
[build]
system = "qmake"
qt_version = "Qt5"
use_jll = true

[build.jll_packages]
qt5 = "Qt5Base_jll"
```

### Platform-Specific Qt Configuration

```toml
[build.linux]
qt_path = "/usr/lib/qt5"

[build.macos]
qt_path = "/usr/local/opt/qt5"

[build.windows]
qt_path = "C:/Qt/5.15.2/msvc2019_64"
```

## Meson Projects

### Configuration

```toml
[build]
system = "meson"
build_dir = "builddir"
meson_options = [
    "-Dbuildtype=release",
    "-Doptimization=3"
]

[build.jll_packages]
meson = "Meson_jll"
ninja = "Ninja_jll"
```

### Example: Meson Project

```julia
RepliBuild.build("meson_project")
```

RepliBuild runs:
```bash
meson setup builddir
ninja -C builddir
```

### Cross-Compilation with Meson

```toml
[build]
system = "meson"
cross_file = "cross/aarch64.ini"
```

## Autotools Projects

### Configuration

```toml
[build]
system = "autotools"
build_dir = "build"

configure_options = [
    "--prefix=/usr/local",
    "--enable-shared",
    "--disable-static"
]
```

### Example: Build with Autotools

```julia
RepliBuild.build("autotools_project")
```

Runs:
```bash
./configure --prefix=/usr/local --enable-shared
make -j$(nproc)
```

### Bootstrap Autotools

```toml
[build]
system = "autotools"
bootstrap = true  # Run autoreconf -i first
```

## Makefile Projects

### Simple Makefile

```toml
[build]
system = "make"
build_dir = "."
make_targets = ["all"]
make_options = ["-j4"]
```

### Custom Make Commands

```toml
[build]
system = "make"
make_command = "gmake"  # Use gmake on BSD
make_targets = ["library", "install"]
```

### Example

```makefile
# Makefile
CC = gcc
CFLAGS = -O2 -fPIC -Iinclude

SRCS = src/lib.c src/utils.c
OBJS = $(SRCS:.c=.o)

libmylib.so: $(OBJS)
\t$(CC) -shared -o $@ $^

%.o: %.c
\t$(CC) $(CFLAGS) -c $< -o $@
```

```toml
[build]
system = "make"
make_targets = ["libmylib.so"]

[output]
library_name = "libmylib"
```

## Advanced Build Configuration

### Multi-Step Builds

```toml
[build]
system = "custom"

[build.steps]
configure = "python configure.py --prefix=$PREFIX"
build = "ninja -C build"
install = "ninja -C build install"
```

### Environment Variables

```toml
[build]
system = "cmake"

[build.environment]
CC = "clang"
CXX = "clang++"
CFLAGS = "-O3 -march=native"
CXXFLAGS = "-O3 -march=native"
```

### Parallel Builds

```toml
[build]
parallel = true
num_jobs = 0  # Use all cores

# Or specify:
# num_jobs = 4
```

### Build Dependencies

```toml
[dependencies]
# Use Julia JLL packages
jll_packages = [
    "Zlib_jll",
    "OpenSSL_jll",
    "SQLite_jll"
]

# System packages (informational)
system_packages = [
    "libboost-dev",
    "libpng-dev"
]
```

### Custom Build Scripts

```toml
[build]
system = "custom"
script = "scripts/build.sh"
```

`scripts/build.sh`:
```bash
#!/bin/bash
set -e

echo "Custom build script"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Locating Build Artifacts

### Automatic Detection

RepliBuild automatically finds:
- `*.so` files in `build/`, `lib/`, `.`
- `*.dylib` on macOS
- `*.dll` on Windows

### Manual Specification

```toml
[output]
# Specific library paths
libraries = [
    "build/lib/libcore.so",
    "build/lib/libutils.so"
]

# Or use patterns
library_patterns = ["build/**/*.so"]
```

### Header Extraction

```toml
[output]
# Generate bindings from these headers
headers = [
    "include/api.h",
    "build/generated/config.h"
]
```

## Platform-Specific Builds

### Linux

```toml
[build.linux]
system = "cmake"
cmake_options = ["-DLINUX_SPECIFIC=ON"]
environment = { CC = "gcc", CXX = "g++" }
```

### macOS

```toml
[build.macos]
system = "cmake"
cmake_options = [
    "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15",
    "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64"
]
```

### Windows

```toml
[build.windows]
system = "cmake"
cmake_options = [
    "-G", "Visual Studio 16 2019",
    "-A", "x64"
]
```

## Troubleshooting

### Build System Not Found

**Error:**
```
ERROR: CMake not found
```

**Solution:**
Use JLL packages:

```toml
[build]
use_jll = true
```

Or install system package:
```bash
# Ubuntu
sudo apt-get install cmake

# macOS
brew install cmake
```

### Wrong Build Directory

**Error:**
```
ERROR: No CMakeLists.txt found
```

**Solution:**
Specify source directory:

```toml
[build]
source_dir = "path/to/source"
build_dir = "path/to/build"
```

### Build Fails with JLL

**Solution:**
Fall back to system tools:

```toml
[build]
use_jll = false
prefer_system = true
```

### Missing Dependencies

**Error:**
```
CMake Error: Could not find Boost
```

**Solution:**
Add Julia JLL or system paths:

```toml
[dependencies]
jll_packages = ["Boost_jll"]

[build.environment]
Boost_ROOT = "/usr/local"
```

### Qt Not Found

**Error:**
```
Project ERROR: Unknown module(s) in QT: widgets
```

**Solution:**
Set Qt path:

```toml
[build]
qt_version = "Qt5"

[build.linux]
qt_path = "/usr/lib/qt5"
environment = { QTDIR = "/usr/lib/qt5" }
```

## Integration Examples

### CMake + pkg-config

```toml
[build]
system = "cmake"

[dependencies]
pkg_config = ["opencv4", "libpng"]
```

### qmake + Custom Libraries

```toml
[build]
system = "qmake"
qt_version = "Qt5"

[build.environment]
LIBS = "-L/usr/local/lib -lboost_system"
INCLUDEPATH = "/usr/local/include"
```

### Meson + Multiple Targets

```toml
[build]
system = "meson"
build_targets = ["core_lib", "utils_lib"]
```

## Performance Tips

### 1. Enable Parallel Builds

```toml
[build]
parallel = true
num_jobs = 0  # Use all cores
```

### 2. Use Ninja with CMake

```toml
[build]
system = "cmake"
cmake_options = ["-G", "Ninja"]

[build.jll_packages]
ninja = "Ninja_jll"
```

### 3. Ccache for Incremental Builds

```toml
[build.environment]
CC = "ccache gcc"
CXX = "ccache g++"
```

### 4. LTO for Release Builds

```toml
[build]
cmake_options = [
    "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"
]
```

## Next Steps

- **[Configuration](configuration.md)**: Complete TOML reference
- **[Module System](modules.md)**: External library management
- **[Examples](../examples/qt-app.md)**: Qt application example
- **[CMake Import](cmake-import.md)**: CMake integration details
