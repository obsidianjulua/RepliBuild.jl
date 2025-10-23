# RepliBuild.jl

**A TOML-based build system for automatic Julia bindings generation from C++ code and binary libraries.**

RepliBuild is a Julia build system that bridges C++ and Julia through automatic binding generation. It supports compiling C++ source code into Julia-callable libraries and wrapping existing binary files with minimal configuration.

## Core Features

- **C++ to Julia Compilation**: Compile C++ source code directly into Julia modules
- **Binary Wrapping**: Generate Julia wrappers for existing `.so`, `.dll`, and `.dylib` files
- **Build System Integration**: Works with CMake, qmake, Meson, and Autotools
- **Module Registry**: Manage external library dependencies with automatic resolution
- **Error Learning**: Intelligent error database that learns from build failures
- **Daemon System**: Optional background daemons for faster compilation
- **LLVM/Clang Integration**: Leverages LLVM toolchain for C++ analysis and compilation

## Workflows

### 1. C++ Source → Julia Module

```julia
# Initialize project
RepliBuild.init("myproject")

# Edit replibuild.toml configuration
# Add your C++ sources to src/

# Compile
RepliBuild.compile()
```

### 2. Existing Binary → Julia Wrapper

```julia
# Initialize binary wrapping project
RepliBuild.init("mybindings", type=:binary)

# Wrap binary
RepliBuild.wrap_binary("/usr/lib/libmath.so")
```

### 3. Build System Integration

```julia
# Build with detected build system
RepliBuild.build()  # Auto-detects CMake, qmake, Meson, etc.
```

## Quick Navigation

- **[Quick Start](getting-started/quickstart.md)**: Get started in 5 minutes
- **[C++ Workflow](guide/cpp-workflow.md)**: Complete C++ to Julia compilation guide
- **[Binary Wrapping](guide/binary-wrapping.md)**: Wrap existing libraries
- **[API Reference](api/core.md)**: Complete function reference
- **[Examples](examples/simple-cpp.md)**: Real-world usage examples

## System Requirements

- Julia 1.12 or later
- LLVM/Clang toolchain (managed automatically via JLL packages)
- For binary wrapping: `nm`, `objdump`, or `dumpbin` (platform-dependent)
- For build integration: CMake, qmake, Meson, etc. (as needed)

## Installation

```julia
using Pkg
Pkg.add("RepliBuild")
```

Or from source:

```julia
using Pkg
Pkg.add(url="https://github.com/obsidianjulua/RepliBuild.jl")
```

## Philosophy

RepliBuild focuses on **functionality over marketing**. The documentation prioritizes:

1. **Working code examples** that you can copy and run
2. **Clear configuration patterns** for common use cases
3. **Troubleshooting guides** based on real build errors
4. **API reference** with practical usage

No fluff, just the tools you need to integrate C++ with Julia.
