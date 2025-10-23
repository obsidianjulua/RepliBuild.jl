# Quick Start

Get started with RepliBuild in 5 minutes.

## Workflow 1: C++ Source to Julia

### Step 1: Create Project

```julia
using RepliBuild

# Initialize C++ project
RepliBuild.init("mymath")
```

This creates:
```
mymath/
├── replibuild.toml    # Configuration
├── src/               # C++ source files
├── include/           # C++ headers
├── julia/             # Generated Julia modules
├── build/             # Build artifacts
└── test/              # Tests
```

### Step 2: Add C++ Code

Create `mymath/src/calculator.cpp`:

```cpp
#include "calculator.h"

namespace Math {
    int add(int a, int b) {
        return a + b;
    }

    int multiply(int a, int b) {
        return a * b;
    }
}
```

Create `mymath/include/calculator.h`:

```cpp
#ifndef CALCULATOR_H
#define CALCULATOR_H

namespace Math {
    int add(int a, int b);
    int multiply(int a, int b);
}

#endif
```

### Step 3: Configure

Edit `mymath/replibuild.toml`:

```toml
[project]
name = "MyMath"
version = "0.1.0"

[compilation]
sources = ["src/calculator.cpp"]
headers = ["include/calculator.h"]
include_dirs = ["include"]

[output]
library_name = "libmymath"
output_dir = "build"
```

### Step 4: Compile

```julia
cd("mymath")
RepliBuild.compile()
```

### Step 5: Use in Julia

```julia
include("julia/MyMath.jl")
using .MyMath

result = Math.add(5, 3)  # Returns 8
product = Math.multiply(4, 7)  # Returns 28
```

## Workflow 2: Wrap Existing Binary

### Step 1: Initialize Wrapper Project

```julia
using RepliBuild

RepliBuild.init("bindings", type=:binary)
```

### Step 2: Wrap Binary

```julia
cd("bindings")

# Wrap a system library
RepliBuild.wrap_binary("/usr/lib/libz.so.1")
```

### Step 3: Use Wrapper

```julia
include("julia_wrappers/libz_wrapper.jl")
using .LibZWrapper

# Call functions from the wrapped library
```

## Workflow 3: Build with CMake/qmake

### For CMake Projects

```julia
using RepliBuild

# Let RepliBuild handle CMake
RepliBuild.build("path/to/cmake/project")
```

### For Qt/qmake Projects

Add to `replibuild.toml`:

```toml
[build]
system = "qmake"
qt_version = "Qt5"
```

Then:

```julia
RepliBuild.build()
```

## Common Operations

### Scan Existing Project

```julia
# Analyze directory structure
RepliBuild.scan("path/to/project")
```

### Import CMake Project

```julia
# Convert CMakeLists.txt to replibuild.toml
RepliBuild.import_cmake("path/to/CMakeLists.txt")
```

### Start Performance Daemons

```julia
# Optional: Start background compilation daemons
RepliBuild.start_daemons()

# Now compilations will be faster
RepliBuild.compile()

# Stop when done
RepliBuild.stop_daemons()
```

### List Available Modules

```julia
# Show installed library modules
RepliBuild.list_modules()
```

### Get Help

```julia
# Show command reference
RepliBuild.help()

# Show system info
RepliBuild.info()
```

## Example Project: Complete Workflow

Here's a complete example from start to finish:

```julia
using RepliBuild

# 1. Create project
RepliBuild.init("vector_lib")
cd("vector_lib")

# 2. Create C++ code (do this manually or via editor)
# Add vector.cpp to src/
# Add vector.h to include/

# 3. Configure (edit replibuild.toml)
# Set sources, headers, etc.

# 4. Compile
RepliBuild.compile()

# 5. Test
include("julia/VectorLib.jl")
using .VectorLib

v = Vector.create(10)
Vector.dot_product(v, v)

# 6. Build release version
RepliBuild.build()
```

## Next Steps

- **[Project Structure](project-structure.md)**: Understand directory layout
- **[C++ Workflow](../guide/cpp-workflow.md)**: Detailed C++ compilation guide
- **[Configuration](../guide/configuration.md)**: TOML configuration reference
- **[Examples](../examples/simple-cpp.md)**: More complete examples

## Troubleshooting Quick Start

### "Command not found" errors
Make sure required build tools are installed:
```julia
RepliBuild.discover_tools()
```

### Include path issues
Add to `replibuild.toml`:
```toml
[compilation]
include_dirs = ["include", "/usr/local/include"]
```

### Linking errors
Specify libraries:
```toml
[compilation]
link_libs = ["m", "pthread"]
```

### Need help?
```julia
RepliBuild.help()
```
