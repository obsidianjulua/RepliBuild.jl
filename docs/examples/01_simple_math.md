# Example 1: Simple Math Functions

This example shows the **simplest possible** RepliBuild usage - wrapping basic C++ functions.

## C++ Code

```cpp
// mathlib.cpp
extern "C" {
    int add(int a, int b) {
        return a + b;
    }

    double multiply(double a, double b) {
        return a * b;
    }

    long factorial(int n) {
        if (n <= 1) return 1;
        long result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
}
```

## Step 1: Create Config

Create `replibuild.toml`:

```toml
[project]
name = "mathlib"
root = "."

[compile]
source_files = ["mathlib.cpp"]
include_dirs = []
flags = ["-std=c++17", "-fPIC", "-O2"]

[binary]
type = "library"
output_name = "libmathlib.so"

[llvm]
toolchain = "auto"

[cache]
enabled = true
```

## Step 2: Build + Wrap

```julia
using RepliBuild

# Compile C++ → library
RepliBuild.build()

# Generate Julia wrapper
RepliBuild.wrap()
```

Output:
```
══════════════════════════════════════════════════════════════════════
 RepliBuild - Compile C++
══════════════════════════════════════════════════════════════════════
✓ Library: julia/libmathlib.so
✓ Metadata saved

══════════════════════════════════════════════════════════════════════
 RepliBuild - Generate Julia Wrappers
══════════════════════════════════════════════════════════════════════
✓ Wrapper: julia/Mathlib.jl
   Functions wrapped: 3
```

## Step 3: Use It!

```julia
include("julia/Mathlib.jl")
using .Mathlib

# Call C++ functions from Julia
result = add(5, 3)           # 8
product = multiply(2.5, 4.0) # 10.0
fact = factorial(5)          # 120
```

## Generated Wrapper (julia/Mathlib.jl)

RepliBuild automatically generates:

```julia
module Mathlib

using Libdl

const _LIB_PATH = "julia/libmathlib.so"
const _LIB = Ref{Ptr{Nothing}}(C_NULL)

function __init__()
    _LIB[] = Libdl.dlopen(_LIB_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
end

export add, multiply, factorial

function add(arg1::Cint, arg2::Cint)::Cint
    ccall((:add, _LIB[]), Cint, (Cint, Cint), arg1, arg2)
end

function multiply(arg1::Cdouble, arg2::Cdouble)::Cdouble
    ccall((:multiply, _LIB[]), Cdouble, (Cdouble, Cdouble), arg1, arg2)
end

function factorial(arg1::Cint)::Clong
    ccall((:factorial, _LIB[]), Clong, (Cint,), arg1)
end

end # module
```

## Key Points

✅ **Zero boilerplate** - Just write C++, RepliBuild handles the rest
✅ **Type-safe** - Correct Julia types (Cint, Cdouble, Clong)
✅ **Automatic** - No manual wrapper writing
✅ **Fast** - Direct ccall, no overhead

## What RepliBuild Did

1. **Compiled** mathlib.cpp → LLVM IR → libmathlib.so
2. **Extracted** symbol table (nm -g)
3. **Generated** Julia module with ccall wrappers
4. **Detected** types from function signatures

## Try It Yourself

See `test_cpp_project/` for a complete working example with this pattern.
