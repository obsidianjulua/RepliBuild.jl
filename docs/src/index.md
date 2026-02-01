# RepliBuild.jl

**Production-Ready C++ to Julia Build Orchestration**

RepliBuild.jl is a powerful build system designed to seamlessly bridge the gap between C++ projects and Julia. It handles the entire lifecycle: discovering source files, compiling them with dependency-aware caching, linking optimized libraries, and automatically generating high-performance Julia wrappers.

## Key Features

- **Dependency-Aware Compilation**: Smart caching ensures only modified files are recompiled.
- **Parallel Builds**: Leverages multi-threading to speed up compilation of large projects.
- **Automatic Wrapping**: Generates `ccall` bindings, struct definitions, and enums directly from binary metadata (DWARF).
- **Introspection Toolkit**: Built-in tools to analyze binary symbols, debug info, optimization passes, and performance.
- **MLIR Integration**: Low-level bindings to MLIR for advanced IR manipulation and JIT compilation.

## Installation

```julia
using Pkg
Pkg.add("RepliBuild")
```

## Quick Start

### 1. Discovery
Scan your project to generate a `replibuild.toml` configuration file.

```julia
using RepliBuild
RepliBuild.discover()
```

### 2. Build
Compile your C++ sources into a shared library.

```julia
RepliBuild.build()
```

### 3. Wrap
Generate the Julia interface.

```julia
RepliBuild.wrap()
```

### One-Liner
For a fresh project, you can do it all at once:

```julia
RepliBuild.discover(build=true, wrap=true)
```

## Documentation

- **[User Guide](guide.md)**: Detailed instructions on workflows and configuration.
- **[API Reference](api.md)**: Documentation for the public API.
- **[Introspection](introspect.md)**: Deep dive into binary analysis and performance tools.
- **[MLIR / JLCS](mlir.md)**: Advanced guide for MLIR integration.

## Configuration

The `replibuild.toml` file gives you full control over the build process:

```toml
[project]
name = "MyProject"

[compile]
flags = ["-O3", "-std=c++17"]
parallel = true

[wrap]
style = "clang"
```

## Advanced Usage

RepliBuild isn't just a build tool; it's a platform for analyzing C++/Julia interoperability. Use `RepliBuild.Introspect` to benchmark calls, inspect generated LLVM IR, or verify SIMD optimizations.

```julia
using RepliBuild.Introspect

# Check if your wrapper is type stable
analyze_type_stability(MyModule.my_function, (1.0, 2))

# Inspect the generated assembly
code_native(MyModule.my_function, (Float64, Int))
```