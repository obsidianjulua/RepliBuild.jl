# Binary Wrapping

Generate Julia wrappers for existing binary libraries without source code.

## Overview

Binary wrapping allows you to call functions from compiled libraries (`.so`, `.dll`, `.dylib`) in Julia without recompiling or having source code.

RepliBuild analyzes binary files using system tools (`nm`, `objdump`, `dumpbin`) and generates Julia `ccall` wrappers.

## Quick Start

```julia
using RepliBuild

# Initialize wrapper project
RepliBuild.init("mybindings", type=:binary)
cd("mybindings")

# Wrap a library
RepliBuild.wrap_binary("/usr/lib/libm.so.6")

# Use the wrapper
include("julia_wrappers/libm_wrapper.jl")
```

## Step-by-Step Guide

### 1. Initialize Project

```julia
RepliBuild.init("wrappers", type=:binary)
```

Creates:
```
wrappers/
├── wrapper_config.toml
├── lib/
├── bin/
└── julia_wrappers/
```

### 2. Add Binary Files

Copy libraries to `lib/`:

```bash
cp /usr/lib/libz.so.1 wrappers/lib/
cp /usr/lib/libssl.so.1.1 wrappers/lib/
```

### 3. Configure Wrapper

Edit `wrapper_config.toml`:

```toml
[wrapper]
scan_dirs = ["lib", "bin"]
output_dir = "julia_wrappers"

# Generate high-level API?
generate_high_level = true

# Symbol filtering
include_patterns = ["^z", "^SSL_"]  # Regex patterns
exclude_patterns = ["^_internal", "test_"]

[library.libz]
path = "lib/libz.so.1"
module_name = "LibZ"

# Specific functions to wrap
exports = [
    "compress",
    "uncompress",
    "deflate",
    "inflate"
]

[library.libssl]
path = "lib/libssl.so.1.1"
module_name = "LibSSL"
```

### 4. Generate Wrappers

```julia
using RepliBuild
cd("wrappers")

# Wrap all configured libraries
RepliBuild.wrap()

# Or wrap specific binary
RepliBuild.wrap_binary("lib/libz.so.1")
```

### 5. Use Wrappers

```julia
include("julia_wrappers/LibZ.jl")
using .LibZ

# High-level API (if generated)
data = Vector{UInt8}("Hello, World!")
compressed = LibZ.compress(data)
decompressed = LibZ.uncompress(compressed)

# Low-level ccall wrappers also available
```

## Advanced Features

### Automatic Symbol Discovery

```julia
# Scan binary and list symbols
RepliBuild.scan_binaries("lib/libmath.so")
```

Output:
```
📦 Scanning: lib/libmath.so
Found 42 exported symbols:

Functions:
  - sin
  - cos
  - tan
  - sqrt
  - pow
  ...

Variables:
  - errno
  - stdin
  - stdout
```

### Symbol Filtering

```toml
[library.mylib]
path = "lib/libmylib.so"

# Include only symbols matching these patterns
include_patterns = [
    "^mylib_",      # Functions starting with mylib_
    "^MYLIB_",      # Constants
    "Vector.*"      # Vector operations
]

# Exclude symbols matching these patterns
exclude_patterns = [
    "_internal",    # Internal functions
    "^test_",       # Test functions
    "debug"         # Debug helpers
]
```

### Type Annotations

Help RepliBuild generate better wrappers:

```toml
[library.libmath]
path = "lib/libmath.so"

# Annotate function signatures
[library.libmath.functions.add]
return_type = "Cdouble"
arg_types = ["Cdouble", "Cdouble"]

[library.libmath.functions.multiply]
return_type = "Cdouble"
arg_types = ["Cdouble", "Cdouble"]

[library.libmath.functions.create_vector]
return_type = "Ptr{Cvoid}"
arg_types = ["Cint"]  # size
```

Generated Julia code:

```julia
function add(a::Cdouble, b::Cdouble)::Cdouble
    ccall((:add, libmath), Cdouble, (Cdouble, Cdouble), a, b)
end
```

### Struct Definitions

Define C structs for proper type handling:

```toml
[library.libvector]
path = "lib/libvector.so"

# Define struct layout
[library.libvector.structs.Vector3]
fields = [
    { name = "x", type = "Cdouble" },
    { name = "y", type = "Cdouble" },
    { name = "z", type = "Cdouble" }
]

[library.libvector.structs.Matrix4x4]
fields = [
    { name = "data", type = "NTuple{16, Cfloat}" }
]
```

Generated Julia code:

```julia
struct Vector3
    x::Cdouble
    y::Cdouble
    z::Cdouble
end

struct Matrix4x4
    data::NTuple{16, Cfloat}
end
```

### Callback Functions

Wrap libraries that use callbacks:

```toml
[library.libcallback]
path = "lib/libcallback.so"

# Define callback type
[library.libcallback.callbacks.CompareFunc]
signature = "Cint (Ptr{Cvoid}, Ptr{Cvoid})"
```

Generated:

```julia
const CompareFunc = Ptr{Cvoid}

function sort_array(arr::Vector, compare_fn)
    # Convert Julia function to C callback
    c_compare = @cfunction($compare_fn, Cint, (Ptr{Cvoid}, Ptr{Cvoid}))
    ccall((:sort_array, libcallback), Cvoid, (Ptr{Cvoid}, Cint, CompareFunc),
          arr, length(arr), c_compare)
end
```

### Platform-Specific Binaries

```toml
[wrapper]
scan_dirs = ["lib"]

# Linux
[library.mylib.linux]
path = "lib/linux/libmylib.so"

# macOS
[library.mylib.macos]
path = "lib/macos/libmylib.dylib"

# Windows
[library.mylib.windows]
path = "lib/windows/mylib.dll"
```

RepliBuild automatically selects the correct path for the current platform.

## Wrapping Common Libraries

### OpenSSL

```julia
RepliBuild.init("openssl_bindings", type=:binary)
cd("openssl_bindings")
```

`wrapper_config.toml`:

```toml
[library.libssl]
path = "/usr/lib/libssl.so.1.1"
module_name = "LibSSL"
include_patterns = ["^SSL_", "^TLS_"]

[library.libcrypto]
path = "/usr/lib/libcrypto.so.1.1"
module_name = "LibCrypto"
include_patterns = ["^EVP_", "^RSA_", "^AES_"]
```

```julia
RepliBuild.wrap()
```

### zlib

```toml
[library.libz]
path = "/usr/lib/libz.so.1"
module_name = "LibZ"
exports = [
    "compress", "compress2",
    "uncompress",
    "deflate", "deflateEnd",
    "inflate", "inflateEnd"
]

# Function signatures
[library.libz.functions.compress]
return_type = "Cint"
arg_types = ["Ptr{UInt8}", "Ptr{Culong}", "Ptr{UInt8}", "Culong"]
```

### SQLite

```toml
[library.libsqlite3]
path = "/usr/lib/libsqlite3.so.0"
module_name = "LibSQLite3"
include_patterns = ["^sqlite3_"]

# Common functions
[library.libsqlite3.functions.sqlite3_open]
return_type = "Cint"
arg_types = ["Ptr{UInt8}", "Ptr{Ptr{Cvoid}}"]

[library.libsqlite3.functions.sqlite3_exec]
return_type = "Cint"
arg_types = ["Ptr{Cvoid}", "Ptr{UInt8}", "Ptr{Cvoid}", "Ptr{Cvoid}", "Ptr{Ptr{UInt8}}"]
```

## Working with C++

C++ libraries use name mangling, making wrapping more complex.

### Demangle C++ Symbols

```julia
# View mangled symbols
RepliBuild.scan_binaries("lib/libcpp.so")
```

Output shows mangled names:
```
_ZN10Calculator3addEii
_ZN10Calculator8multiplyEii
```

### Wrap C++ with Extern "C"

Best practice: Provide C API:

```cpp
// wrapper.h
#ifdef __cplusplus
extern "C" {
#endif

typedef void* CalculatorHandle;

CalculatorHandle calculator_create();
void calculator_destroy(CalculatorHandle calc);
int calculator_add(CalculatorHandle calc, int a, int b);

#ifdef __cplusplus
}
#endif
```

Then wrap the C API:

```toml
[library.libcalculator]
path = "lib/libcalculator.so"
exports = [
    "calculator_create",
    "calculator_destroy",
    "calculator_add"
]
```

### Direct C++ Wrapping

For libraries without C API, use CxxWrap.jl integration:

```julia
# RepliBuild can generate CxxWrap stubs
RepliBuild.generate_bindings_clangjl("lib/libcpp.so",
    headers=["include/calculator.h"])
```

## Troubleshooting

### Symbol Not Found

**Error:**
```
ERROR: Symbol 'myfunction' not found in library
```

**Solution:**
Check symbol exists:

```bash
nm -D lib/mylib.so | grep myfunction
```

If using C++, ensure `extern "C"`.

### Wrong Number of Arguments

**Error:**
```
ERROR: ccall: wrong number of arguments
```

**Solution:**
Specify correct signature:

```toml
[library.mylib.functions.myfunction]
arg_types = ["Cint", "Ptr{Cdouble}", "Csize_t"]
```

### Type Mismatch

**Error:**
```
ERROR: type mismatch in ccall
```

**Solution:**
Match C types exactly:

| C Type | Julia Type |
|--------|-----------|
| `int` | `Cint` |
| `long` | `Clong` |
| `float` | `Cfloat` |
| `double` | `Cdouble` |
| `char*` | `Ptr{Cchar}` |
| `void*` | `Ptr{Cvoid}` |
| `size_t` | `Csize_t` |

### Library Not Loaded

**Error:**
```
ERROR: could not load library "libmylib.so"
```

**Solution:**

1. Use absolute path:
```toml
path = "/usr/lib/libmylib.so"
```

2. Or set `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH
```

3. Or use `@rpath`:
```julia
const libmylib = joinpath(@__DIR__, "../lib/libmylib.so")
```

### Version Conflicts

Different library versions:

```toml
# Specify exact version
[library.libssl]
path = "/usr/lib/libssl.so.1.1.1"  # Exact version
# Or symlink
path = "/usr/lib/libssl.so"        # Latest
```

## Best Practices

### 1. Test Before Wrapping

```bash
# Verify library loads
ldd lib/mylib.so

# Check dependencies are available
nm -D lib/mylib.so
```

### 2. Wrap Only What You Need

```toml
# Don't wrap entire library
exports = [
    "essential_function_1",
    "essential_function_2"
]
```

### 3. Add Type Safety

```julia
# wrapper.jl
function safe_add(a::Int, b::Int)::Int
    # Validate inputs
    result = ccall((:add, libmath), Cint, (Cint, Cint), a, b)
    return Int(result)
end
```

### 4. Handle Errors

```julia
function safe_open(filename::String)
    handle = ccall((:open_file, libio), Ptr{Cvoid}, (Ptr{UInt8},), filename)
    if handle == C_NULL
        error("Failed to open file: $filename")
    end
    return handle
end
```

### 5. Document Wrappers

```julia
"""
    compress(data::Vector{UInt8}) -> Vector{UInt8}

Compress data using zlib deflate algorithm.

# Arguments
- `data::Vector{UInt8}`: Input data to compress

# Returns
- Compressed data as `Vector{UInt8}`

# Example
```julia
compressed = LibZ.compress(b"Hello, World!")
```
"""
function compress(data::Vector{UInt8})
    # Implementation
end
```

## Performance Considerations

### Minimize Allocations

```julia
# Bad: allocates on each call
function add(a, b)
    result = Ref{Cint}()
    ccall((:add, lib), Cvoid, (Cint, Cint, Ptr{Cint}), a, b, result)
    return result[]
end

# Good: direct return
function add(a, b)
    ccall((:add, lib), Cint, (Cint, Cint), a, b)
end
```

### Batch Operations

```julia
# Process arrays in bulk
function batch_process(data::Vector{Float64})
    ccall((:process_array, lib), Cvoid,
          (Ptr{Cdouble}, Csize_t),
          data, length(data))
end
```

### Preload Libraries

```julia
# Load once at module init
const libmath = "/usr/lib/libmath.so"

function __init__()
    # Verify library loads
    Libdl.dlopen(libmath)
end
```

## Next Steps

- **[Build Systems](build-systems.md)**: Integrate with existing build systems
- **[Configuration](configuration.md)**: Complete TOML reference
- **[Examples](../examples/binary-wrap.md)**: Binary wrapping examples
- **[Advanced](../advanced/error-learning.md)**: Error learning system
