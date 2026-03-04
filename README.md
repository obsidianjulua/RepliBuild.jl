# RepliBuild.jl

**A No-BS, Zero-Overhead C++ ↔ Julia JIT Bridge.**

RepliBuild.jl isn't just another `ccall` wrapper generator. It is a full ABI-aware compiler bridge powered by a custom MLIR dialect (`jlcs`). It ingests raw C++ source code, parses the DWARF debug info, maps out deep C++ inheritance and vtables, and JIT-compiles it directly into type-safe, native Julia bindings.

If you have a massive C/C++ project (like Duktape or SQLite) and you want it in Julia right now without writing manual bindings, or you need to pass strided matrices back and forth with *literally zero* wrapper overhead (matching bare-metal `ccall` speeds to the nanosecond), RepliBuild is what you use.

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

## Realistic Configuration

The `replibuild.toml` controls everything. Here is a real-world setup for a high-performance C++ engine:

```toml
[project]
name = "MyEngine"
root = "."

[compile]
flags = ["-std=c++17", "-fPIC", "-O3"]
parallel = true
source_files = ["src/engine.cpp", "src/math.cpp"]
include_dirs = ["include", "src"]

[link]
optimization_level = "3"

[wrap]
style = "clang"
use_clang_jl = true

[types]
allow_unknown_structs = true
allow_function_pointers = true
strictness = "warn"
```

## Proof it works

Check out the `test/` directory. We battle-test this engine against:
- **Duktape (`duktape_test`)**: Compiles the entire 3.25 MB `duktape.c` monolithic Javascript engine and evaluates JS natively from Julia.
- **SQLite (`sqlite_test`)**: Wraps the full SQLite3 C API automatically.
- **Virtual Dispatch (`vtable_test`)**: Instantiates derived C++ classes (like `Circle` and `Rectangle`) and natively dispatches virtual methods (`get_area`) correctly from Julia.
- **Bi-directional Callbacks (`callback_test`)**: Passes native Julia JIT closures (`@cfunction`) down to C++ which calls them back in a loop without losing stack context.
- **Zero-Copy Benchmarks (`benchmark_test`)**: Passes zero-allocation strided struct views for multi-dimensional memory without copying. For small matrices (e.g. 4x4), the MLIR thunk completely strips generic dispatch overhead, executing natively in ~94ns, perfectly mirroring bare-metal `ccall` performance with zero wrapper penalty.

## Documentation

- **[docs/src/guide.md](docs/src/guide.md)**: The user guide on workflows and configuration.
- **[docs/src/mlir.md](docs/src/mlir.md)**: Deep dive into the MLIR / JLCS Dialect Internals.
