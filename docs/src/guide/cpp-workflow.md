# C++ to Julia Workflow

Complete guide to compiling C++ source code into Julia-callable modules.

## Overview

The C++ to Julia workflow:

1. **Initialize** project structure
2. **Add** C++ source and headers
3. **Configure** compilation settings
4. **Compile** to shared library and Julia bindings
5. **Use** from Julia code

## Step-by-Step Guide

### 1. Initialize Project

```julia
using RepliBuild

RepliBuild.init("mylib")
cd("mylib")
```

Creates:
```
mylib/
├── replibuild.toml
├── src/
├── include/
├── julia/
├── build/
└── test/
```

### 2. Add C++ Code

Create `include/math_ops.h`:

```cpp
#ifndef MATH_OPS_H
#define MATH_OPS_H

namespace MathOps {
    // Basic arithmetic
    double add(double a, double b);
    double multiply(double a, double b);

    // Vector operations
    struct Vector3 {
        double x, y, z;
    };

    double dot_product(const Vector3& a, const Vector3& b);
    Vector3 cross_product(const Vector3& a, const Vector3& b);

    // Class example
    class Calculator {
    public:
        Calculator();
        ~Calculator();

        void set_precision(int p);
        double compute(const char* expression);

    private:
        int precision;
    };
}

#endif
```

Create `src/math_ops.cpp`:

```cpp
#include "math_ops.h"
#include <cmath>
#include <stdexcept>

namespace MathOps {

    double add(double a, double b) {
        return a + b;
    }

    double multiply(double a, double b) {
        return a * b;
    }

    double dot_product(const Vector3& a, const Vector3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    Vector3 cross_product(const Vector3& a, const Vector3& b) {
        Vector3 result;
        result.x = a.y * b.z - a.z * b.y;
        result.y = a.z * b.x - a.x * b.z;
        result.z = a.x * b.y - a.y * b.x;
        return result;
    }

    // Calculator implementation
    Calculator::Calculator() : precision(6) {}

    Calculator::~Calculator() {}

    void Calculator::set_precision(int p) {
        if (p < 0) throw std::invalid_argument("Precision must be non-negative");
        precision = p;
    }

    double Calculator::compute(const char* expression) {
        // Simplified example - parse and evaluate expression
        // In real code, use a proper parser
        return 42.0;
    }
}
```

### 3. Configure Compilation

Edit `replibuild.toml`:

```toml
[project]
name = "MathOps"
version = "1.0.0"
description = "Mathematical operations library"

[compilation]
# Source files to compile
sources = [
    "src/math_ops.cpp"
]

# Header files for binding generation
headers = [
    "include/math_ops.h"
]

# Include directories
include_dirs = [
    "include"
]

# Libraries to link against
link_libs = [
    "m",        # Math library (Linux)
    "stdc++"    # C++ standard library
]

# Compiler flags
cxx_flags = [
    "-std=c++17",
    "-O2",
    "-Wall",
    "-Wextra"
]

[output]
library_name = "libmathops"
output_dir = "build"
julia_module_name = "MathOps"

[bindings]
# Namespace to wrap
namespaces = ["MathOps"]

# Functions to export (empty = all public functions)
export_functions = []

# Classes to wrap
export_classes = ["Calculator"]

# Structs to wrap
export_structs = ["Vector3"]
```

### 4. Compile

```julia
using RepliBuild

# Compile the project
RepliBuild.compile()
```

Output:
```
🚀 RepliBuild - Compiling project
📦 Loading configuration: replibuild.toml
🔍 Discovering LLVM tools...
✅ Found clang: /path/to/clang
✅ Found clang++: /path/to/clang++
🔨 Compiling C++ sources...
   ✅ src/math_ops.cpp
🔗 Linking library: libmathops.so
📝 Generating Julia bindings...
   ✅ MathOps module
   ✅ Function wrappers
   ✅ Type definitions
✅ Compilation complete!
```

### 5. Use from Julia

```julia
include("julia/MathOps.jl")
using .MathOps

# Call functions
result = MathOps.add(5.0, 3.0)
println("5 + 3 = $result")  # 8.0

product = MathOps.multiply(4.0, 7.0)
println("4 * 7 = $product")  # 28.0

# Use structs
v1 = MathOps.Vector3(1.0, 2.0, 3.0)
v2 = MathOps.Vector3(4.0, 5.0, 6.0)

dot = MathOps.dot_product(v1, v2)
println("Dot product: $dot")  # 32.0

cross = MathOps.cross_product(v1, v2)
println("Cross product: $(cross.x), $(cross.y), $(cross.z)")

# Use classes
calc = MathOps.Calculator()
MathOps.set_precision(calc, 4)
result = MathOps.compute(calc, "2 + 2")
println("Result: $result")
```

## Advanced Configuration

### Multiple Source Files

```toml
[compilation]
sources = [
    "src/core/calculator.cpp",
    "src/core/vector.cpp",
    "src/utils/parser.cpp",
    "src/utils/formatter.cpp"
]
```

Or use wildcards:

```toml
[compilation]
source_dirs = ["src"]  # Recursively find all .cpp files
exclude_patterns = ["*_test.cpp", "*_backup.cpp"]
```

### Include Paths

```toml
[compilation]
include_dirs = [
    "include",
    "/usr/local/include",
    "/opt/mylib/include",
    "../common/include"  # Relative paths OK
]
```

### External Libraries

```toml
[compilation]
link_libs = [
    "boost_system",
    "boost_filesystem",
    "pthread",
    "ssl",
    "crypto"
]

# Library search paths
lib_dirs = [
    "/usr/local/lib",
    "/opt/mylib/lib"
]

# Link flags
link_flags = [
    "-Wl,-rpath=/usr/local/lib",
    "-Wl,--no-undefined"
]
```

### Compiler Options

```toml
[compilation]
# C++ standard
cxx_standard = "c++17"  # or "c++11", "c++14", "c++20"

# Optimization
optimization = "2"      # -O2
# optimization = "3"    # -O3
# optimization = "s"    # -Os (size)

# Debug symbols
debug = true            # -g

# Custom flags
cxx_flags = [
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-fPIC",
    "-march=native"
]

# Preprocessor defines
defines = [
    "NDEBUG",
    "MY_FEATURE_ENABLED",
    "VERSION=1.0"
]
```

### Platform-Specific Settings

```toml
[compilation.linux]
link_libs = ["rt", "dl"]
cxx_flags = ["-pthread"]

[compilation.macos]
link_libs = []
cxx_flags = ["-framework", "CoreFoundation"]

[compilation.windows]
link_libs = ["ws2_32", "bcrypt"]
cxx_flags = ["/EHsc", "/MD"]
```

### Binding Generation Options

```toml
[bindings]
# Namespaces to wrap (empty = all)
namespaces = ["MyLib", "MyLib::Core"]

# Functions to export (empty = all public)
export_functions = [
    "add",
    "multiply",
    "compute"
]

# Exclude specific functions
exclude_functions = [
    "internal_helper",
    "debug_print"
]

# Classes to wrap
export_classes = [
    "Calculator",
    "Matrix",
    "Solver"
]

# Export templates?
export_templates = false

# Generate high-level Julia API?
generate_high_level = true
```

## Compilation Strategies

### Incremental Builds

```julia
# Only recompile changed files
RepliBuild.compile()  # Automatic incremental build
```

RepliBuild tracks:
- Source file modification times
- Header dependencies
- Configuration changes

### Clean Rebuild

```julia
# Force full rebuild
rm("build", recursive=true)
RepliBuild.compile()
```

### Parallel Compilation

```toml
[compilation]
parallel = true
num_jobs = 4  # Or use all cores: num_jobs = 0
```

### Debug vs Release

**Debug build:**
```toml
[compilation]
optimization = "0"
debug = true
defines = ["DEBUG", "VERBOSE_LOGGING"]
```

**Release build:**
```toml
[compilation]
optimization = "3"
debug = false
defines = ["NDEBUG"]
cxx_flags = ["-march=native", "-flto"]
```

## Troubleshooting

### Include Not Found

**Error:**
```
fatal error: 'myheader.h' file not found
```

**Solution:**
```toml
[compilation]
include_dirs = ["include", "/path/to/headers"]
```

### Undefined Reference

**Error:**
```
undefined reference to `boost::filesystem::path::path()`
```

**Solution:**
```toml
[compilation]
link_libs = ["boost_filesystem", "boost_system"]
```

### Multiple Definition

**Error:**
```
multiple definition of `myfunction'
```

**Solution:**
Use `inline` or move to `.cpp`:

```cpp
// header.h
inline int myfunction() { return 42; }  // Add inline
// OR move to .cpp file
```

### ABI Compatibility

**Error:**
```
undefined symbol: _ZN7MyClass6methodEv
```

**Solution:**
Ensure matching C++ standard:

```toml
[compilation]
cxx_standard = "c++17"  # Match your library
```

### Template Issues

Templates must be in headers:

```cpp
// vector.h
template<typename T>
class Vector {
public:
    T dot(const Vector<T>& other) const {
        // Implementation must be in header
        return /* ... */;
    }
};
```

### Platform-Specific Code

```cpp
#ifdef __linux__
    #include <unistd.h>
#elif defined(_WIN32)
    #include <windows.h>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
#endif
```

Configure per-platform:

```toml
[compilation.linux]
sources = ["src/linux_impl.cpp"]

[compilation.windows]
sources = ["src/windows_impl.cpp"]
```

## Performance Tips

### 1. Use Link-Time Optimization

```toml
[compilation]
cxx_flags = ["-flto"]
link_flags = ["-flto"]
```

### 2. Profile-Guided Optimization

```bash
# Step 1: Build with profiling
# Add to replibuild.toml:
# cxx_flags = ["-fprofile-generate"]

# Step 2: Run representative workload
julia> include("julia/MyLib.jl")
julia> # ... run typical operations ...

# Step 3: Rebuild with profile data
# cxx_flags = ["-fprofile-use"]
```

### 3. Precompiled Headers

Create `include/pch.h`:

```cpp
// pch.h - Precompiled header
#include <vector>
#include <string>
#include <iostream>
#include <memory>
```

```toml
[compilation]
precompiled_header = "include/pch.h"
```

### 4. Use Daemons for Faster Iteration

```julia
# Start compilation daemons
RepliBuild.start_daemons()

# Now compilations are faster
RepliBuild.compile()  # Uses daemon cache

# Stop when done
RepliBuild.stop_daemons()
```

## Testing

Create `test/runtests.jl`:

```julia
using Test
include("../julia/MathOps.jl")
using .MathOps

@testset "MathOps Tests" begin
    @testset "Basic arithmetic" begin
        @test MathOps.add(2.0, 3.0) ≈ 5.0
        @test MathOps.multiply(4.0, 5.0) ≈ 20.0
    end

    @testset "Vector operations" begin
        v1 = MathOps.Vector3(1.0, 0.0, 0.0)
        v2 = MathOps.Vector3(0.0, 1.0, 0.0)

        dot = MathOps.dot_product(v1, v2)
        @test dot ≈ 0.0

        cross = MathOps.cross_product(v1, v2)
        @test cross.z ≈ 1.0
    end

    @testset "Calculator" begin
        calc = MathOps.Calculator()
        MathOps.set_precision(calc, 2)
        result = MathOps.compute(calc, "10 / 3")
        @test result ≈ 3.33 atol=0.01
    end
end
```

Run tests:

```julia
include("test/runtests.jl")
```

## Next Steps

- **[Binary Wrapping](binary-wrapping.md)**: Wrap existing libraries
- **[Build Systems](build-systems.md)**: Integrate with CMake, qmake
- **[Configuration](configuration.md)**: Complete TOML reference
- **[Examples](../examples/simple-cpp.md)**: More complete examples
