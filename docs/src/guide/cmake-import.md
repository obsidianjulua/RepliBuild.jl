# CMake Import

Import existing CMake projects into RepliBuild without running CMake.

## Overview

RepliBuild can parse `CMakeLists.txt` files and extract:
- Source files
- Header files
- Include directories
- Link libraries
- Compiler flags
- Targets

This generates a `replibuild.toml` configuration for native Julia builds.

## Quick Start

```julia
using RepliBuild

# Import CMake project
RepliBuild.import_cmake("path/to/CMakeLists.txt")

# Import specific target
RepliBuild.import_cmake("CMakeLists.txt", target="mylib")

# Compile with RepliBuild
RepliBuild.compile()
```

## Step-by-Step Guide

### 1. Analyze CMake Project

```julia
using RepliBuild

# Parse CMakeLists.txt
cmake_project = RepliBuild.import_cmake("opencv/CMakeLists.txt")

# Inspect project
println("Project: ", cmake_project.project_name)
println("Targets: ", keys(cmake_project.targets))
```

Output:
```
Project: OpenCV
Targets: opencv_core, opencv_imgproc, opencv_video, ...
```

### 2. Select Target

```julia
# Import specific target
RepliBuild.import_cmake("opencv/CMakeLists.txt",
                        target="opencv_core",
                        output="opencv_core.toml")
```

### 3. Generated Configuration

`opencv_core.toml`:

```toml
[project]
name = "opencv_core"
version = "4.5.0"

[compilation]
sources = [
    "modules/core/src/matrix.cpp",
    "modules/core/src/array.cpp",
    # ... extracted from CMake
]

headers = [
    "modules/core/include/opencv2/core.hpp",
    # ...
]

include_dirs = [
    "modules/core/include",
    "include",
    "build/include"
]

link_libs = ["z", "pthread"]

cxx_flags = ["-std=c++11"]

[output]
library_name = "libopencv_core"
output_dir = "build/lib"
```

### 4. Compile

```julia
RepliBuild.compile("opencv_core.toml")
```

## Supported CMake Features

### add_library()

```cmake
add_library(mylib SHARED
    src/core.cpp
    src/utils.cpp
)
```

Extracts:
- Library name: `mylib`
- Sources: `src/core.cpp`, `src/utils.cpp`
- Type: shared library

### target_include_directories()

```cmake
target_include_directories(mylib
    PUBLIC include
    PRIVATE src/internal
)
```

Extracts:
- Public includes: `include`
- Private includes: `src/internal`

### target_link_libraries()

```cmake
target_link_libraries(mylib
    PUBLIC pthread
    PRIVATE z
)
```

Extracts:
- Link libraries: `pthread`, `z`

### target_compile_options()

```cmake
target_compile_options(mylib PRIVATE
    -Wall
    -Wextra
    -O2
)
```

Extracts:
- Compiler flags: `-Wall`, `-Wextra`, `-O2`

### target_compile_definitions()

```cmake
target_compile_definitions(mylib PRIVATE
    USE_FEATURE=1
    NDEBUG
)
```

Extracts:
- Defines: `USE_FEATURE=1`, `NDEBUG`

### find_package()

```cmake
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem)
```

Generates module dependencies:

```toml
[dependencies]
modules = ["OpenCV"]
cmake_packages = [
    { name = "Boost", components = ["system", "filesystem"] }
]
```

## Advanced Usage

### Import Multiple Targets

```julia
# Import all targets
project = RepliBuild.import_cmake("CMakeLists.txt")

for (name, target) in project.targets
    println("Importing: $name")
    RepliBuild.import_cmake("CMakeLists.txt",
                            target=name,
                            output="$(name).toml")
end
```

### Selective Import

```julia
# Import only specific file patterns
RepliBuild.import_cmake("CMakeLists.txt",
                        target="mylib",
                        include_patterns=["src/**/*.cpp"],
                        exclude_patterns=["**/test_*.cpp"])
```

### Override Configuration

```julia
# Import and customize
project = RepliBuild.import_cmake("CMakeLists.txt", target="mylib")

# Edit configuration
config = TOML.parsefile("mylib.toml")
config["compilation"]["optimization"] = "3"
config["compilation"]["cxx_flags"] = ["-march=native"]

# Save modified config
open("mylib.toml", "w") do io
    TOML.print(io, config)
end
```

### Merge with Existing Config

```julia
# Import and merge
cmake_config = RepliBuild.import_cmake("CMakeLists.txt", target="mylib")

# Merge with manual configuration
manual_config = """
[bindings]
export_classes = ["MyClass"]
generate_high_level = true

[test]
auto_run = true
"""

# Combine
final_config = merge_toml("mylib.toml", manual_config)
```

## Handling Complex CMake

### Conditional Compilation

CMake:
```cmake
if(USE_CUDA)
    target_sources(mylib PRIVATE cuda/kernels.cu)
    target_link_libraries(mylib PRIVATE cuda)
endif()
```

RepliBuild import includes conditional sources:

```toml
[compilation]
sources = ["src/core.cpp"]

# Document conditional sources
# Manual: Add cuda/kernels.cu if USE_CUDA=ON
```

Add manually if needed:

```toml
[compilation.cuda]
sources = ["cuda/kernels.cu"]
link_libs = ["cuda"]
```

### Generator Expressions

CMake:
```cmake
target_compile_options(mylib PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)
```

Import extracts both configurations:

```toml
# Debug flags
[compilation.debug]
optimization = "0"
debug = true

# Release flags
[compilation.release]
optimization = "3"
defines = ["NDEBUG"]
```

### Subdirectories

CMake:
```cmake
add_subdirectory(core)
add_subdirectory(utils)
add_subdirectory(plugins)
```

RepliBuild resolves paths:

```toml
[compilation]
sources = [
    "core/src/main.cpp",
    "utils/src/helper.cpp",
    "plugins/src/plugin.cpp"
]
```

## Working with Popular Projects

### OpenCV

```julia
# Clone OpenCV
run(`git clone https://github.com/opencv/opencv.git`)

# Import specific module
RepliBuild.import_cmake("opencv/CMakeLists.txt", target="opencv_core")

# Build with RepliBuild
RepliBuild.compile("opencv_core.toml")
```

### Eigen

```julia
# Eigen is header-only, but can still import
RepliBuild.import_cmake("eigen/CMakeLists.txt")
```

Generates:

```toml
[compilation]
headers = ["Eigen/Core", "Eigen/Dense"]
include_dirs = ["eigen"]

# Header-only library
header_only = true
```

### Boost

```julia
# Boost uses its own build system (b2), but has CMake support
RepliBuild.import_cmake("boost/CMakeLists.txt",
                        target="boost_system")
```

## Limitations

### Not Supported

1. **Generator expressions** - Partially supported
2. **Complex scripting** - Only basic CMake functions
3. **External projects** - Must import separately
4. **Custom commands** - Not extracted

### Workarounds

**For complex projects**, use build system delegation instead:

```toml
[build]
system = "cmake"
cmake_options = ["-DCMAKE_BUILD_TYPE=Release"]
```

Then:

```julia
RepliBuild.build()  # Uses CMake, then generates bindings
```

## Debugging Import

### Verbose Output

```julia
ENV["REPLIBUILD_DEBUG"] = "1"
RepliBuild.import_cmake("CMakeLists.txt")
```

Shows:
```
🔍 Parsing CMakeLists.txt
  Found project: MyProject
  Found target: mylib
    Sources: 5 files
    Headers: 3 files
    Include dirs: 2 paths
  ✅ Import complete
```

### Manual Inspection

```julia
# Parse without generating config
project = RepliBuild.parse_cmake_file("CMakeLists.txt")

# Inspect targets
for (name, target) in project.targets
    println("\nTarget: $name")
    println("  Type: ", target.type)
    println("  Sources: ", target.sources)
    println("  Include dirs: ", target.include_dirs)
    println("  Link libs: ", target.link_libraries)
end
```

### Dry Run

```julia
# Preview generated TOML without writing
toml_content = RepliBuild.generate_replibuild_toml(cmake_project, "mylib")
println(toml_content)
```

## Best Practices

### 1. Import Minimal Target

```julia
# Good: Import only what you need
RepliBuild.import_cmake("CMakeLists.txt", target="core_lib")

# Avoid: Importing entire project
# RepliBuild.import_cmake("CMakeLists.txt")  # Too much
```

### 2. Review Generated Config

Always review and adjust `replibuild.toml`:

```julia
# Import
RepliBuild.import_cmake("CMakeLists.txt", target="mylib")

# Review
# Edit replibuild.toml manually

# Test
RepliBuild.compile()
```

### 3. Use Delegation for Complex Projects

For projects with complex CMake logic:

```toml
[build]
system = "cmake"  # Let CMake handle complexity

[output]
julia_module_name = "MyLib"
```

### 4. Document Manual Changes

```toml
# Generated from CMakeLists.txt
# Manual additions:
# - Added CUDA sources
# - Modified optimization flags
# - Added custom module dependency

[compilation]
sources = [
    # Auto-generated sources...
    "cuda/kernels.cu"  # Manual addition
]
```

## Integration with Build Systems

### CMake + RepliBuild Hybrid

Keep CMake, use RepliBuild for bindings:

```julia
# 1. Build with CMake
RepliBuild.build()  # Uses CMake

# 2. Generate bindings from CMake artifacts
RepliBuild.wrap_binary("build/lib/libmylib.so")

# Or import configuration
RepliBuild.import_cmake("CMakeLists.txt", target="mylib")
RepliBuild.generate_bindings_clangjl("build/lib/libmylib.so",
                                     headers=["include/api.h"])
```

## Next Steps

- **[Build Systems](build-systems.md)**: Full build system integration
- **[Configuration](configuration.md)**: TOML configuration details
- **[Examples](../examples/simple-cpp.md)**: Complete examples
