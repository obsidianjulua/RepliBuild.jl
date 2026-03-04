# RepliBuild.jl

**A No-BS, Zero-Overhead C++ ↔ Julia JIT Bridge.**

RepliBuild.jl isn't just another `ccall` wrapper generator. It is a full ABI-aware compiler bridge powered by a custom MLIR dialect (`jlcs`). It ingests raw C++ source code, parses the DWARF debug info, maps out deep C++ inheritance and vtables, and JIT-compiles it directly into type-safe, native Julia bindings.

## What it actually does

1. **JIT-to-JIT Execution**: Passes Julia closures into C++ event loops and executes C++ virtual methods from Julia. It crosses the boundary seamlessly.
2. **Native VTable Dispatch**: Automatically resolves and calls C++ virtual methods directly in IR (`jlcs.vcall`). No "fragile base class" hacks. 
3. **Zero-Copy Arrays**: Implements N-dimensional strided array views natively in MLIR (`jlcs.load_array_element`). Your `Vector{Float64}` maps exactly to C++ multi-dimensional memory layouts without copies.
4. **Dependency-Aware Builds**: Discovers your C++ files, parses `#include` trees, caches aggressively, and builds parallelized shared libraries via an embedded LLVM/MLIR toolchain.
5. **No Boilerplate**: You don't write bindings. It extracts `struct` layouts, enums, function pointers, and methods straight from the compiled DWARF metadata.

## Quickstart (The No-BS Workflow)

Drop this in the root of your C++ project:

```julia
using Pkg; Pkg.add("RepliBuild")
using RepliBuild

# 1. Scans your C++ files, builds the `#include` graph, and generates `replibuild.toml`
RepliBuild.discover()

# 2. Compiles your C++ code to LLVM IR, optimizes it, and emits a shared library
RepliBuild.build()

# 3. Parses the DWARF data and generates the native Julia wrapper module
RepliBuild.wrap()
```

Or just do it all at once:
```julia
RepliBuild.discover(build=true, wrap=true)
```

## Documentation

- **[No-BS User Guide](guide.md)**: Detailed instructions on workflows and configuration.
- **[API Reference](api.md)**: Documentation for the public API.
- **[Introspection](introspect.md)**: Deep dive into binary analysis and performance tools.
- **[MLIR / JLCS Dialect Internals](mlir.md)**: Advanced guide for MLIR integration.

## Realistic Configuration

The `replibuild.toml` file gives you full control over the build process:

```toml
[project]
name = "MyEngine"
root = "."

[compile]
flags = ["-O3", "-std=c++17", "-fPIC"]
parallel = true

[wrap]
style = "clang"
use_clang_jl = true

[types]
allow_unknown_structs = true
allow_function_pointers = true
strictness = "warn"
```