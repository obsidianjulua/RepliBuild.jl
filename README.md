# RepliBuild.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://obsidianjulua.github.io/RepliBuild.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://obsidianjulua.github.io/RepliBuild.jl/dev/)

**ABI-aware C/C++ compiler bridge for Julia, powered by MLIR.**

RepliBuild ingests C/C++ source code, compiles it through an LLVM/MLIR pipeline, introspects DWARF debug metadata, and emits type-safe Julia bindings with correct struct layout, enum definitions, and calling conventions. Functions that require non-trivial ABI handling (packed structs, unions, virtual dispatch) are automatically routed through a JIT tier built on a custom MLIR dialect.

## Philosophy

RepliBuild is a **source-based wrapper generator**. It intentionally bypasses JLLs and BinaryBuilder to utilize modern LLVM 21 and MLIR for perfect, zero-edit C++ bindings. This requires the user to have a local toolchain installed, but in exchange provides:

- **Perfectly optimized binaries** tailored to the host machine (native `-march`, LTO across the FFI boundary)
- **Automatic DWARF-based wrappers** that standard tools cannot generate — struct layout, enum mappings, virtual dispatch, and idiomatic Julia types are all derived from debug metadata, not manual annotations
- **Zero-cost abstractions** — eligible functions are inlined across the C++/Julia boundary via `Base.llvmcall`

By owning the heavy toolchain requirement, RepliBuild occupies a different point in the design space than BinaryBuilder: instead of distributing pre-compiled binaries that work everywhere, it generates *exact* bindings on your machine with *zero* manual FFI boilerplate. If you are wrapping a C++ library with templates, virtual methods, or complex ABI, this is the tool for it.

Run `RepliBuild.check_environment()` to verify your toolchain is ready.

## Install

```julia
using Pkg
Pkg.add("RepliBuild")
```

Requires Julia 1.10+ and a system LLVM/Clang toolchain (auto-detected, falls back to JLL).

## Usage

### Step by step

```julia
using RepliBuild

# 1. Scan source files, resolve #include dependencies, generate replibuild.toml
RepliBuild.discover("path/to/project")

# 2. Compile to LLVM IR, optimize, link, emit shared library
RepliBuild.build("path/to/project/replibuild.toml")

# 3. Parse DWARF metadata, generate Julia module with ccall wrappers
RepliBuild.wrap("path/to/project/replibuild.toml")
```

Or chain everything in one call:

```julia
RepliBuild.discover("path/to/project", build=true, wrap=true)
```

### One-liner with the package registry

```julia
using RepliBuild

# Verify your toolchain is ready.
RepliBuild.check_environment()

# Register a project once
RepliBuild.register("path/to/project/replibuild.toml")

# Then load it anywhere — builds on first call, cached thereafter
Lua = RepliBuild.use("lua_wrapper")
```

The generated module lives in `julia/ProjectName.jl` and can be loaded directly:

```julia
include("path/to/project/julia/ProjectName.jl")
using .ProjectName
```

### Utilities

```julia
# Show project info (source files, config, build status)
RepliBuild.info("path/to/project/replibuild.toml")

# Clean build artifacts
RepliBuild.clean("path/to/project/replibuild.toml")

# Check LLVM/Clang/MLIR toolchain status
RepliBuild.check_environment()
```

### Package registry

```julia
# Register a project in the global registry (~/.replibuild/registry/)
RepliBuild.register("replibuild.toml")

# List all registered packages
RepliBuild.list_registry()

# Load a registered package (builds + wraps + caches automatically)
MyLib = RepliBuild.use("my_lib")

# Remove a package from the registry
RepliBuild.unregister("my_lib")

# Scaffold a distributable Julia package from a registered project
RepliBuild.scaffold_package("MyLibWrapper")
```

## What it handles

- **Structs** with correct field order, alignment padding, and forward declarations for circular references
- **Enums** mapped to Julia `@enum` with correct underlying types
- **Unions** as `NTuple{N,UInt8}` with typed getter/setter accessors
- **Bitfields** with bit-level extraction
- **Function pointers** and variadic functions (typed overloads via config)
- **Multi-level pointers** (`T**` -> `Ptr{Ptr{T}}`) and reference types (`T&` -> `Ref{T}`)
- **C++ virtual methods** via MLIR JIT thunks (or statically via AOT thunks)
- **Idiomatic Julia structs** — factory/destructor pairs are detected and wrapped into `mutable struct` types with GC-managed finalizers and multiple-dispatch method proxies
- **Automatic finalizers** for types with destructors (generates `ManagedX` wrappers with GC integration)
- **Global variables** via `cglobal` accessors
- **Git/local/system dependencies** — external C/C++ libraries fetched, filtered, and compiled automatically via `[dependencies]` in `replibuild.toml`
- **Template instantiation** — declare `templates = ["std::vector<int>"]` and RepliBuild forces Clang to emit the DWARF for those types
- **Zero-cost LTO dispatch** — when `enable_lto = true`, eligible functions are emitted as `Base.llvmcall` paths that let Julia's JIT inline C++ code directly into hot loops

## Architecture

RepliBuild operates as a two-tier dispatch system:

**Tier 1 (ccall / llvmcall)** — Standard functions with POD arguments and scalar/small struct returns go through direct `ccall`. Zero overhead beyond the foreign call itself. When `enable_lto = true`, eligible functions are upgraded to `Base.llvmcall`, embedding the C++ LLVM bitcode directly into Julia's JIT pipeline so the compiler can inline across the language boundary. This makes the C++ IR visible to Julia's optimizer, enabling cross-language AD with tools like Enzyme.jl.

**Tier 2 (MLIR JIT / AOT)** — Functions involving packed structs, unions, large struct returns, or virtual dispatch are compiled through a custom MLIR dialect (`jlcs`) that handles ABI marshalling correctly. The JIT engine caches compiled symbols with a lock-free read path for hot calls. When `aot_thunks = true`, these thunks are pre-compiled to a static `.so` at build time, eliminating JIT startup cost entirely.

The tier selection is automatic. The wrapper generator analyzes each function's signature against DWARF metadata and routes accordingly.

**Idiomatic wrappers** — On top of the raw bindings, `Wrapper.jl` clusters factory functions, destructors, and methods by C++ class name and emits `mutable struct` types with GC-managed finalizers and multiple-dispatch method proxies, so user code reads like natural Julia rather than raw FFI.

### Pipeline

```
C/C++ Source + [dependencies] (git/local/system)
    |
    v
Dependency Resolver (clone/update, filter excludes, inject into compile graph)
    |
    v
Discovery (scan files, parse #include graph)
    |
    v
Compilation (Clang -> LLVM IR, per-file caching, parallel)
    |
    v
Linking (IR merge, LTO, optimization → .so + optional _lto.ll + optional _thunks.so)
    |
    v
Binary (shared library + DWARF metadata extraction)
    |
    v
Wrapping (DWARF introspection -> raw ccall/llvmcall wrappers + idiomatic mutable structs)
    |
    v
JIT Init (MLIR IR generation -> execution engine, on demand — or AOT .so, if pre-compiled)
```

## Configuration

`replibuild.toml` is generated by `discover()` and can be customized:

```toml
[project]
name = "MyProject"

[compile]
flags = ["-std=c++17", "-fPIC", "-O3"]
parallel = true
aot_thunks = false        # Pre-compile MLIR C++ vtable thunks into a static .so

[link]
optimization_level = "3"
enable_lto = false        # Emit LLVM IR for Base.llvmcall zero-cost dispatch

[wrap]
style = "clang"
use_clang_jl = true

[wrap.varargs]
# Typed overloads for variadic functions
printf = [["Cstring", "Cint"], ["Cstring", "Cdouble"]]

[types]
strictness = "warn"           # "strict", "warn", or "permissive"
allow_unknown_structs = true
allow_function_pointers = true
templates = ["std::vector<int>", "std::vector<double>"]
template_headers = ["<vector>"]

# External git dependency: fetched, filtered, and compiled automatically
[dependencies.cjson]
type = "git"
url = "https://github.com/DaveGamble/cJSON"
tag = "v1.7.18"
exclude = ["test", "fuzzing", "CMakeLists.txt"]
```

See [docs/src/config.md](docs/src/config.md) for the full reference.

## Tested against

The test suite compiles and wraps real-world C/C++ projects end-to-end:

| Project | Source | What it validates |
|---------|--------|-------------------|
| **Lua 5.4.6** | 30 files, full VM + stdlib | State management, stack ops, code eval, Julia<->Lua callbacks, coroutines |
| **Duktape 2.7.0** | 101K-line JS engine amalgamation | Compiles monolithic C, evaluates JavaScript from Julia |
| **SQLite 3.49.1** | 261K-line database engine | Full C API wrap with varargs support |
| **Stress test** | Vectors, matrices, complex numerics | DWARF extraction, struct layout, introspection toolkit |
| **VTable test** | C++ inheritance hierarchy | Virtual dispatch via MLIR JIT (Circle/Rectangle polymorphism) |
| **Callback test** | Bidirectional FFI | Julia `@cfunction` passed to C++ event loops |
| **Benchmark test** | Strided matrix views | Zero-copy struct pointer passing, matches bare `ccall` at ~94ns for 4x4 matmul |
| **JIT edge cases** | Scalars, structs, packed structs | 3-tier benchmark: bare ccall vs wrapper vs MLIR JIT |

Test sources for Lua, Duktape, and SQLite are downloaded on demand via `setup.jl` scripts or git config's in replibuild.toml (not vendored).

## Introspection

RepliBuild includes a built-in analysis toolkit covering binary analysis, Julia IR inspection, LLVM tooling, benchmarking, and data export:

```julia
using RepliBuild.Introspect

# Binary analysis
Introspect.symbols("lib.so", filter=:functions)
Introspect.dwarf_info("lib.so")
Introspect.disassemble("lib.so", "my_function")

# Julia IR inspection
Introspect.code_llvm(my_func, (Cint, Cint))
Introspect.analyze_type_stability(my_func, (Cint, Cint))
Introspect.analyze_simd(my_func, (Cint, Cint))

# LLVM tooling
Introspect.optimize_ir("build/module.ll", "3")
Introspect.compare_optimization("build/module.ll", ["0", "2", "3"])

# Benchmarking
result = Introspect.benchmark(f, args...; samples=1000)
Introspect.export_json(result, "bench.json")
```

See [docs/src/introspect.md](docs/src/introspect.md) for the full 25+ function reference.

## Documentation

- [User Guide](docs/src/guide.md) — Workflow and configuration
- [Configuration Reference](docs/src/config.md) — All `replibuild.toml` options
- [Introspection Tools](docs/src/introspect.md) — Binary analysis, benchmarking, data export
- [MLIR / JLCS Dialect](docs/src/mlir.md) — JIT internals and the custom dialect
- [Changelog](CHANGELOG.md)

## License

MIT
