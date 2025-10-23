# Example: Simple C++ Library

Complete walkthrough of creating a simple C++ math library and using it from Julia.

## Project Overview

We'll create a simple math library with:
- Basic arithmetic functions
- Vector operations
- A Calculator class

## Step 1: Initialize Project

```julia
using RepliBuild

# Create project
RepliBuild.init("simple_math")
cd("simple_math")
```

Directory structure:
```
simple_math/
├── replibuild.toml
├── src/
├── include/
├── julia/
├── build/
└── test/
```

## Step 2: Create C++ Code

### Header File

Create `include/math_ops.h`:

```cpp
#ifndef MATH_OPS_H
#define MATH_OPS_H

namespace SimpleMath {
    // Basic operations
    double add(double a, double b);
    double subtract(double a, double b);
    double multiply(double a, double b);
    double divide(double a, double b);

    // Vector operations
    struct Vector2D {
        double x, y;
    };

    double vector_length(const Vector2D& v);
    Vector2D vector_add(const Vector2D& a, const Vector2D& b);
    double vector_dot(const Vector2D& a, const Vector2D& b);

    // Calculator class
    class Calculator {
    public:
        Calculator();
        ~Calculator();

        void clear();
        void set_accumulator(double value);
        double get_accumulator() const;

        void add(double value);
        void subtract(double value);
        void multiply(double value);
        void divide(double value);

    private:
        double accumulator;
    };
}

#endif // MATH_OPS_H
```

### Implementation File

Create `src/math_ops.cpp`:

```cpp
#include "math_ops.h"
#include <cmath>
#include <stdexcept>

namespace SimpleMath {

    // Basic operations
    double add(double a, double b) {
        return a + b;
    }

    double subtract(double a, double b) {
        return a - b;
    }

    double multiply(double a, double b) {
        return a * b;
    }

    double divide(double a, double b) {
        if (b == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        return a / b;
    }

    // Vector operations
    double vector_length(const Vector2D& v) {
        return std::sqrt(v.x * v.x + v.y * v.y);
    }

    Vector2D vector_add(const Vector2D& a, const Vector2D& b) {
        return Vector2D{a.x + b.x, a.y + b.y};
    }

    double vector_dot(const Vector2D& a, const Vector2D& b) {
        return a.x * b.x + a.y * b.y;
    }

    // Calculator implementation
    Calculator::Calculator() : accumulator(0.0) {}

    Calculator::~Calculator() {}

    void Calculator::clear() {
        accumulator = 0.0;
    }

    void Calculator::set_accumulator(double value) {
        accumulator = value;
    }

    double Calculator::get_accumulator() const {
        return accumulator;
    }

    void Calculator::add(double value) {
        accumulator += value;
    }

    void Calculator::subtract(double value) {
        accumulator -= value;
    }

    void Calculator::multiply(double value) {
        accumulator *= value;
    }

    void Calculator::divide(double value) {
        if (value == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        accumulator /= value;
    }
}
```

## Step 3: Configure Project

Edit `replibuild.toml`:

```toml
[project]
name = "SimpleMath"
version = "1.0.0"
description = "Simple math library example"

[compilation]
# Source files
sources = ["src/math_ops.cpp"]

# Headers for binding generation
headers = ["include/math_ops.h"]

# Include directories
include_dirs = ["include"]

# Link with math library
link_libs = ["m"]

# Compiler flags
cxx_flags = [
    "-std=c++11",
    "-Wall",
    "-O2",
    "-fPIC"
]

[output]
library_name = "libsimplemath"
output_dir = "build"
julia_module_name = "SimpleMath"

[bindings]
# Wrap the SimpleMath namespace
namespaces = ["SimpleMath"]

# Export all functions and classes
export_functions = []  # Empty = all
export_classes = ["Calculator"]
export_structs = ["Vector2D"]

# Generate high-level Julia API
generate_high_level = true
```

## Step 4: Compile

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
✅ Found clang++: /usr/bin/clang++
🔨 Compiling C++ sources...
   ✅ src/math_ops.cpp
🔗 Linking library: build/libsimplemath.so
📝 Generating Julia bindings...
   ✅ SimpleMath module
   ✅ Function wrappers
   ✅ Type definitions
✅ Compilation complete!
```

## Step 5: Use from Julia

Create `test_math.jl`:

```julia
# Load the generated module
include("julia/SimpleMath.jl")
using .SimpleMath

# Test basic operations
println("=== Basic Operations ===")
result = SimpleMath.add(5.0, 3.0)
println("5 + 3 = $result")

result = SimpleMath.multiply(4.0, 7.0)
println("4 * 7 = $result")

result = SimpleMath.divide(10.0, 2.0)
println("10 / 2 = $result")

# Test vector operations
println("\n=== Vector Operations ===")
v1 = SimpleMath.Vector2D(3.0, 4.0)
println("v1 = ($(v1.x), $(v1.y))")

length = SimpleMath.vector_length(v1)
println("Length of v1 = $length")

v2 = SimpleMath.Vector2D(1.0, 2.0)
v3 = SimpleMath.vector_add(v1, v2)
println("v1 + v2 = ($(v3.x), $(v3.y))")

dot = SimpleMath.vector_dot(v1, v2)
println("v1 · v2 = $dot")

# Test Calculator class
println("\n=== Calculator ===")
calc = SimpleMath.Calculator()

SimpleMath.set_accumulator(calc, 10.0)
println("Initial value: $(SimpleMath.get_accumulator(calc))")

SimpleMath.add(calc, 5.0)
println("After +5: $(SimpleMath.get_accumulator(calc))")

SimpleMath.multiply(calc, 2.0)
println("After *2: $(SimpleMath.get_accumulator(calc))")

SimpleMath.divide(calc, 3.0)
println("After /3: $(SimpleMath.get_accumulator(calc))")

SimpleMath.clear(calc)
println("After clear: $(SimpleMath.get_accumulator(calc))")
```

Run the test:

```julia
include("test_math.jl")
```

Output:
```
=== Basic Operations ===
5 + 3 = 8.0
4 * 7 = 28.0
10 / 2 = 5.0

=== Vector Operations ===
v1 = (3.0, 4.0)
Length of v1 = 5.0
v1 + v2 = (4.0, 6.0)
v1 · v2 = 11.0

=== Calculator ===
Initial value: 10.0
After +5: 15.0
After *2: 30.0
After /3: 10.0
After clear: 0.0
```

## Step 6: Add Tests

Create `test/runtests.jl`:

```julia
using Test

# Load the module
include("../julia/SimpleMath.jl")
using .SimpleMath

@testset "SimpleMath Tests" begin
    @testset "Basic Operations" begin
        @test SimpleMath.add(2.0, 3.0) ≈ 5.0
        @test SimpleMath.subtract(10.0, 3.0) ≈ 7.0
        @test SimpleMath.multiply(4.0, 5.0) ≈ 20.0
        @test SimpleMath.divide(10.0, 2.0) ≈ 5.0

        # Test error handling
        @test_throws ErrorException SimpleMath.divide(1.0, 0.0)
    end

    @testset "Vector Operations" begin
        v1 = SimpleMath.Vector2D(3.0, 4.0)
        @test SimpleMath.vector_length(v1) ≈ 5.0

        v2 = SimpleMath.Vector2D(1.0, 2.0)
        v3 = SimpleMath.vector_add(v1, v2)
        @test v3.x ≈ 4.0
        @test v3.y ≈ 6.0

        dot = SimpleMath.vector_dot(v1, v2)
        @test dot ≈ 11.0
    end

    @testset "Calculator" begin
        calc = SimpleMath.Calculator()

        SimpleMath.set_accumulator(calc, 10.0)
        @test SimpleMath.get_accumulator(calc) ≈ 10.0

        SimpleMath.add(calc, 5.0)
        @test SimpleMath.get_accumulator(calc) ≈ 15.0

        SimpleMath.multiply(calc, 2.0)
        @test SimpleMath.get_accumulator(calc) ≈ 30.0

        SimpleMath.divide(calc, 3.0)
        @test SimpleMath.get_accumulator(calc) ≈ 10.0

        SimpleMath.clear(calc)
        @test SimpleMath.get_accumulator(calc) ≈ 0.0

        # Test error handling
        @test_throws ErrorException SimpleMath.divide(calc, 0.0)
    end
end
```

Run tests:

```julia
include("test/runtests.jl")
```

Output:
```
Test Summary:      | Pass  Total
SimpleMath Tests   |   14     14
  Basic Operations |    5      5
  Vector Operations|    4      4
  Calculator       |    5      5
```

## Step 7: Package the Library

Create `Project.toml` for the Julia package:

```toml
name = "SimpleMath"
uuid = "12345678-1234-1234-1234-123456789abc"
version = "1.0.0"

[deps]

[compat]
julia = "1.6"
```

## Complete Project Structure

```
simple_math/
├── Project.toml
├── replibuild.toml
├── src/
│   └── math_ops.cpp
├── include/
│   └── math_ops.h
├── julia/
│   ├── SimpleMath.jl
│   └── (generated bindings)
├── build/
│   └── libsimplemath.so
├── test/
│   └── runtests.jl
└── test_math.jl
```

## Tips and Variations

### Add More Features

```cpp
// In math_ops.h
namespace SimpleMath {
    // Power function
    double power(double base, double exponent);

    // Trigonometry
    double sind(double degrees);
    double cosd(double degrees);
}
```

Recompile:
```julia
RepliBuild.compile()
```

### Optimize for Release

```toml
[compilation]
optimization = "3"
cxx_flags = ["-std=c++11", "-O3", "-march=native", "-flto"]
debug = false
defines = ["NDEBUG"]
```

### Add Documentation

```julia
"""
    SimpleMath

A simple math library demonstrating RepliBuild.

Provides basic arithmetic, vector operations, and a calculator class.
"""
module SimpleMath
    # Generated code...
end
```

## Next Steps

- **[Qt Application Example](qt-app.md)**: Build a Qt GUI application
- **[Binary Wrapping Example](binary-wrap.md)**: Wrap existing libraries
- **[Multi-Module Example](multi-module.md)**: Use multiple dependencies
