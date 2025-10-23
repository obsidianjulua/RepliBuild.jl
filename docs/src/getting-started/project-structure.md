# Project Structure

RepliBuild uses a standardized directory structure for both C++ compilation and binary wrapping workflows.

## C++ Source Projects

Created with `RepliBuild.init("project")` or `RepliBuild.init("project", type=:cpp)`:

```
myproject/
├── replibuild.toml       # Main configuration file
├── src/                  # C++ source files (.cpp, .cc, .cxx)
├── include/              # C++ header files (.h, .hpp)
├── julia/                # Generated Julia modules
├── build/                # Compiled libraries and build artifacts
└── test/                 # Test files
```

### Directory Purposes

#### `src/`
C++ implementation files. All `.cpp`, `.cc`, and `.cxx` files here are candidates for compilation.

**Example:**
```
src/
├── calculator.cpp
├── vector_ops.cpp
└── matrix.cpp
```

#### `include/`
C++ header files. Used for:
- Public API declarations
- Type definitions
- Template implementations

**Example:**
```
include/
├── calculator.h
├── vector_ops.h
└── types.h
```

#### `julia/`
Auto-generated Julia modules. **Do not edit manually** - these files are regenerated on each compilation.

**Example:**
```
julia/
├── MyProject.jl          # Main module
├── calculator_wrapper.jl # Function wrappers
└── types_wrapper.jl      # Type definitions
```

#### `build/`
Build artifacts and intermediate files:

```
build/
├── libmyproject.so       # Compiled shared library
├── *.o                   # Object files
└── compile_commands.json # Compilation database (for IDEs)
```

#### `test/`
Test files for your project:

```
test/
├── runtests.jl
├── test_calculator.jl
└── test_vectors.jl
```

### Configuration File

`replibuild.toml` defines compilation settings:

```toml
[project]
name = "MyProject"
version = "0.1.0"
description = "My C++ library for Julia"

[compilation]
sources = ["src/calculator.cpp", "src/vector_ops.cpp"]
headers = ["include/calculator.h", "include/vector_ops.h"]
include_dirs = ["include"]
link_libs = ["m", "pthread"]

[output]
library_name = "libmyproject"
output_dir = "build"
julia_module_name = "MyProject"
```

## Binary Wrapping Projects

Created with `RepliBuild.init("project", type=:binary)`:

```
mybindings/
├── wrapper_config.toml   # Wrapper configuration
├── lib/                  # Binary library files
├── bin/                  # Binary executables (optional)
└── julia_wrappers/       # Generated Julia wrappers
```

### Directory Purposes

#### `lib/`
Binary library files to wrap:

```
lib/
├── libmath.so
├── libcrypto.so.1.1
└── libssl.so.1.1
```

Supported formats:
- **Linux**: `.so`, `.so.1`, `.so.1.2.3`
- **macOS**: `.dylib`
- **Windows**: `.dll`

#### `bin/`
Optional directory for executable binaries:

```
bin/
├── mytool
└── myapp
```

#### `julia_wrappers/`
Generated Julia wrapper modules:

```
julia_wrappers/
├── libmath_wrapper.jl
├── libcrypto_wrapper.jl
└── LibMath.jl           # High-level module
```

### Wrapper Configuration

`wrapper_config.toml`:

```toml
[wrapper]
scan_dirs = ["lib", "bin"]
output_dir = "julia_wrappers"

[library.libmath]
path = "lib/libmath.so"
module_name = "LibMath"
exports = ["sin", "cos", "sqrt"]  # Functions to export

[library.libcrypto]
path = "lib/libcrypto.so.1.1"
module_name = "LibCrypto"
```

## Multi-Module Projects

Large projects can use the module system:

```
myproject/
├── replibuild.toml
├── modules/              # External library modules
│   ├── OpenCV.toml
│   ├── Qt5.toml
│   └── Boost.toml
├── src/
├── include/
└── julia/
```

### Module Files

`modules/OpenCV.toml`:

```toml
[module]
name = "OpenCV"
version = "4.5.0"

[library]
lib_names = ["opencv_core", "opencv_imgproc"]
include_dirs = ["/usr/include/opencv4"]
link_flags = ["-lopencv_core", "-lopencv_imgproc"]

[package]
pkg_config = "opencv4"  # Use pkg-config for detection
```

Reference in `replibuild.toml`:

```toml
[dependencies]
modules = ["OpenCV", "Qt5"]
```

## Build System Integration Projects

Projects using CMake, qmake, Meson, etc.:

```
qtproject/
├── replibuild.toml       # RepliBuild configuration
├── CMakeLists.txt        # Or .pro file, meson.build, etc.
├── src/
├── include/
├── build/                # Build directory (managed by build system)
└── julia/                # Generated wrappers
```

### Configuration

`replibuild.toml` for build integration:

```toml
[project]
name = "QtApp"

[build]
system = "cmake"          # or "qmake", "meson", "autotools"
build_dir = "build"
cmake_options = ["-DCMAKE_BUILD_TYPE=Release"]

# For Qt/qmake
qt_version = "Qt5"
pro_file = "qtapp.pro"

[output]
julia_module_name = "QtApp"
```

## RepliBuild System Directories

RepliBuild maintains its own directory structure:

```
~/.replibuild/             # Linux/macOS
AppData/Local/RepliBuild/  # Windows

├── modules/               # Installed module templates
│   ├── OpenCV/
│   ├── Qt5/
│   └── Boost/
├── cache/                 # Build cache
├── config.toml            # Global configuration
└── error_db.sqlite        # Error learning database
```

### Path Functions

```julia
# Get RepliBuild directory
RepliBuild.get_replibuild_dir()

# Get cache directory
RepliBuild.get_cache_dir()

# Get module search paths
RepliBuild.get_module_search_paths()

# Print all paths
RepliBuild.print_paths_info()
```

### Initialize Directories

```julia
# Ensure all directories exist
RepliBuild.initialize_directories()
```

## Project Templates

Create projects from templates:

```julia
# List available templates
RepliBuild.available_templates()

# Create from template
RepliBuild.use_template("simple_cpp", "myproject")
```

Available templates:
- `simple_cpp` - Basic C++ library
- `qt_app` - Qt application
- `binary_wrapper` - Binary wrapping project
- `cmake_project` - CMake-based project

## Best Practices

### Source Organization

**Good:**
```
src/
├── core/
│   ├── calculator.cpp
│   └── vector.cpp
├── utils/
│   └── helpers.cpp
└── api.cpp
```

**Avoid:**
```
src/
├── everything.cpp        # Too monolithic
└── temp_backup.cpp       # No temporary files
```

### Header Organization

**Good:**
```
include/
├── mylib/
│   ├── core/
│   │   ├── calculator.h
│   │   └── vector.h
│   └── utils/
│       └── helpers.h
└── mylib.h               # Main public header
```

### Build Output

Keep build artifacts separate:
```toml
[output]
output_dir = "build"      # Not "." or "src"
```

### Version Control

`.gitignore` for RepliBuild projects:

```gitignore
# Build artifacts
build/
*.o
*.so
*.dylib
*.dll

# Generated Julia code
julia/*_wrapper.jl

# Cache
.replibuild_cache/

# Keep
!replibuild.toml
!wrapper_config.toml
```

## Migrating Existing Projects

### From CMake

```julia
# Import CMake project
RepliBuild.import_cmake("CMakeLists.txt")

# Review generated replibuild.toml
# Then compile
RepliBuild.compile()
```

### From Existing C++ Code

```julia
# Scan directory
RepliBuild.scan("path/to/cpp/code")

# Review generated replibuild.toml
# Adjust as needed
# Compile
RepliBuild.compile()
```

## Next Steps

- **[C++ Workflow](../guide/cpp-workflow.md)**: Complete compilation guide
- **[Configuration](../guide/configuration.md)**: Detailed TOML reference
- **[Build Systems](../guide/build-systems.md)**: Integration with CMake, qmake, etc.
