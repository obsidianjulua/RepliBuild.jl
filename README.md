# RepliBuild.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://obsidianjulua.github.io/RepliBuild.jl/dev/)
[![Julia 1.10+](https://img.shields.io/badge/julia-1.10+-9558B2?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ABI-aware C/C++ compiler bridge for Julia. Compiles C/C++ source through an LLVM/MLIR pipeline, introspects DWARF debug metadata, and emits type-safe Julia bindings with correct struct layout, enum definitions, and calling conventions. Functions are automatically routed to one of three calling tiers — `llvmcall` with LTO bitcode, MLIR AOT thunks, or `ccall` — based on ABI complexity.

## Requirements

- Julia 1.10+
- LLVM 21+ and Clang (system install; auto-detected, JLL fallback available for Tier 1)
- CMake 3.20+ and `mlir-tblgen` (required for the `jlcs` MLIR dialect / Tier 2 only)
- `libJLCS.so` built from `src/mlir/` via `./build.sh` (Tier 2 only; built automatically by `deps/build.jl`)

Run `RepliBuild.check_environment()` to verify which tiers are available.

## Installation

```julia
julia> using Pkg; Pkg.add("RepliBuild")
```

## Quick start

```julia
using RepliBuild

# Scan a C/C++ project, generate replibuild.toml, compile, and wrap in one call
RepliBuild.discover("path/to/project", build=true, wrap=true)

# Load the generated module
include("path/to/project/julia/MyProject.jl")
using .MyProject
```

Or step by step:

```julia
toml = RepliBuild.discover("path/to/project")  # generates replibuild.toml
RepliBuild.build(toml)                          # Clang → LLVM IR → .so + DWARF metadata
RepliBuild.wrap(toml)                           # DWARF → Julia module in julia/
```

### Package registry

`use` is the one-call path: resolves dependencies, builds if needed, caches artifacts in `~/.replibuild/builds/<hash>/`, and returns a loaded `Module`.

```julia
RepliBuild.register("path/to/project/replibuild.toml")  # one-time registration
Lua = RepliBuild.use("lua")                              # build + wrap + load, cached
Lua.luaL_newstate()
```

```julia
RepliBuild.list_registry()                    # print all registered packages
RepliBuild.unregister("lua")                  # remove from registry
RepliBuild.scaffold_package("LuaWrapper")     # generate a distributable Julia package
```

## Three-tier dispatch

Tier selection is automatic. The wrapper generator analyses each function signature against DWARF metadata and emits the appropriate calling convention.

| Tier | Mechanism | When selected |
|------|-----------|---------------|
| **1** | `Base.llvmcall((bitcode, sym), ...)` | POD args, scalar/pointer return, LTO bitcode available |
| **2** | MLIR AOT thunks via `libJLCS.so` | Packed structs, unions, large struct return, C++ virtual dispatch |
| **3** | `ccall` | Fallback when bitcode is unavailable |

**Tier 1 (llvmcall / LTO)** — LTO artifacts are LLVM bitcode (`.bc`) assembled by `Clang_unified_jll` to guarantee version-matched IR. The C++ IR merges into Julia's JIT pipeline, enabling cross-language inlining, vectorization, and AD (e.g. Enzyme.jl). Enabled by `enable_lto = true` in `[link]`; defaults to `true` for pure-C projects.

**Tier 2 (MLIR JIT / AOT)** — A custom MLIR dialect (`jlcs`) handles ABI marshalling for non-trivial types. JIT symbols are cached with a lock-free read path. Set `aot_thunks = true` to pre-compile thunks to a `_thunks.so` at build time, eliminating JIT startup cost.

**Tier 3 (ccall)** — Direct `ccall` with zero setup. The unconditional fallback.

## What gets wrapped

- **Structs** — correct field order, alignment padding, topological sort for circular references, `Ptr{X}` soft-dependency handling
- **Enums** — `@enum` with correct underlying types; Clang.jl AST walker (not regex) handles `enum class`, hex values, namespaces
- **Unions** — `NTuple{N,UInt8}` with typed getter/setter accessors
- **Bitfields** — bit-level extraction
- **Function pointers** — DWARF signature parsing to `@cfunction`-compatible type strings
- **Variadic functions** — typed overloads declared in `[wrap.varargs]`
- **Multi-level pointers / references** — `T**` → `Ptr{Ptr{T}}`, `T&` → `Ref{T}`
- **C++ virtual methods** — MLIR JIT thunks or static AOT thunks
- **Idiomatic wrappers** — factory/destructor pairs clustered by class name into `mutable struct` with GC-managed finalizers and multiple-dispatch method proxies (`ManagedX`)
- **Global variables** — `cglobal` accessors
- **Templates** — declare `templates = ["std::vector<int>"]`; RepliBuild forces Clang to emit DWARF for those instantiations

## Rust Integration (Experimental)

RepliBuild supports generating introspective wrappers for Rust via DWARF debug metadata. To use it, set `language = "rust"` in the `[wrap]` section of your `replibuild.toml`.

**Current Requirements:**
Because Rust does not have a stable ABI, you must expose a C-compatible interface for RepliBuild to successfully bind to it:
- Functions must be marked with `extern "C"` and `#[no_mangle]`.
- Structs and enums must use `#[repr(C)]` or `#[repr(IntType)]`.

The wrapper generator will automatically strip internal Rust standard library leakage (like `core::fmt`, `alloc::string`, etc.) from the DWARF output, resolving only the public C-compatible types. Native Rust ABI integration is planned for a future release.

## Pipeline

```
C/C++ Source + [dependencies]
    │
    ▼
DependencyResolver   — clone/update git deps, filter excludes, inject into compile graph
    │
    ▼
Discovery            — scan files, resolve #include graph, emit replibuild.toml
    │
    ▼
Compiler             — Clang/clang++ → per-file LLVM IR; incremental mtime + project-hash cache; parallel
    │
    ▼
Linker               — llvm-link + llvm-opt → .so/.dylib + _lto.bc (Tier 1) + _thunks.so (Tier 2, optional)
    │
    ▼
DWARFParser          — llvm-dwarfdump → ClassInfo / VtableInfo structs
    │
    ▼
Wrapper              — DWARF + nm symbols → Julia module (C and C++ generators are independent)
    │
    ▼
JITManager           — on-demand: JLCSIRGenerator emits jlcs IR → MLIRNative JIT → thunk cache
```

### Caching

Two independent layers:
- **Per-file IR cache** (`.replibuild_cache/`) — mtime-based, skips individual source files whose IR is current.
- **Project content hash** — hashes `replibuild.toml`, all source and header contents, and git HEAD. If the hash matches, `build()` exits in sub-second time without invoking any compiler.

## Configuration

`replibuild.toml` is generated by `discover()` and is hand-editable:

```toml
[project]
name = "MyProject"

[compile]
flags    = ["-std=c++17", "-fPIC", "-O3"]
parallel = true

[link]
optimization_level = "3"
enable_lto         = false   # true → emit _lto.bc for Base.llvmcall (Tier 1)

[binary]
type           = "shared"    # "shared" | "static" | "executable"
strip_symbols  = false

[wrap]
language     = "cpp"         # "c" | "cpp"  (auto-detected by discover())
use_clang_jl = true
aot_thunks   = false         # true → pre-compile MLIR thunks to _thunks.so (Tier 2)

[wrap.varargs]
printf = [["Cstring", "Cint"], ["Cstring", "Cdouble"]]

[types]
strictness             = "warn"   # "strict" | "warn" | "permissive"
allow_unknown_structs  = true
allow_function_pointers = true
templates              = ["std::vector<int>"]
template_headers       = ["<vector>"]

[cache]
enabled   = true
directory = ".replibuild_cache"

[dependencies.cjson]
type    = "git"
url     = "https://github.com/DaveGamble/cJSON"
tag     = "v1.7.18"
exclude = ["test", "fuzzing", "CMakeLists.txt"]
```

`language = "c"` defaults `enable_lto = true`; `language = "cpp"` defaults `enable_lto = false`.

Full reference: [Configuration Reference](https://obsidianjulua.github.io/RepliBuild.jl/dev/config/)

## Introspection

`RepliBuild.Introspect` provides binary analysis, Julia IR inspection, LLVM pass tooling, and benchmarking:

```julia
using RepliBuild.Introspect

Introspect.symbols("lib.so", filter=:functions)
Introspect.dwarf_info("lib.so")
Introspect.disassemble("lib.so", "my_function")

Introspect.code_llvm(my_func, (Cint, Cint))
Introspect.analyze_type_stability(my_func, (Cint, Cint))
Introspect.analyze_simd(my_func, (Cint, Cint))

Introspect.optimize_ir("build/module.ll", "3")
Introspect.compare_optimization("build/module.ll", ["0", "2", "3"])

result = Introspect.benchmark(f, args...; samples=1000)
Introspect.export_json(result, "bench.json")
```

Full reference (25+ functions): [Introspection Tools](https://obsidianjulua.github.io/RepliBuild.jl/dev/introspect/)

## Test coverage

The CI suite (`test/runtests.jl`) runs the stress test, MLIR dialect tests, and integration tests. The full developer suite (`test/devtests.jl`) additionally wraps real-world C/C++ projects end-to-end:

| Project | Source | Coverage |
|---------|--------|----------|
| Lua 5.4.6 | 30 files, full VM + stdlib | State management, stack ops, `lua_pcall`, Julia↔Lua callbacks, coroutines |
| SQLite 3.49.1 | 261 K-line database engine | Full C API, varargs, opaque pointer lifecycle |
| Duktape 2.7.0 | 101 K-line JS engine (amalgamation) | Monolithic C compile, `duk_eval_string`, stack round-trips |
| cJSON | Multi-file C library | Git dependency resolution, `[dependencies]` workflow |
| Stress test | Vectors, matrices, numerics | DWARF extraction, struct layout, all wrapper generator paths |
| VTable test | C++ inheritance hierarchy | Virtual dispatch via MLIR JIT (Circle/Rectangle polymorphism) |
| Callback test | Bidirectional FFI | Julia `@cfunction` passed to C++ event loops |
| Benchmark test | Strided 4×4 matrix views | Zero-copy struct pointer passing; wrapper overhead ~94 ns vs bare `ccall` |
| JIT edge cases | Scalars, packed structs, unions | Three-tier benchmark: `ccall` vs wrapper vs MLIR JIT |

External sources (Lua, SQLite, Duktape) are downloaded on demand by `setup.jl` scripts and are not vendored.

## Public API

```
discover(path; force, build, wrap)   → toml_path
build(toml_path; clean)              → library_path
wrap(toml_path; headers)             → wrapper_path
use(name; force_rebuild, verbose)    → Module
register(toml_path; name, verified)  → RegistryEntry
unregister(name)
list_registry()
clean(toml_path)
info(toml_path)
check_environment(; verbose, throw_on_error) → ToolchainStatus
scaffold_package(name; path)         → package_path
```

## Documentation

- [User Guide](https://obsidianjulua.github.io/RepliBuild.jl/dev/guide/)
- [Configuration Reference](https://obsidianjulua.github.io/RepliBuild.jl/dev/config/)
- [Introspection Tools](https://obsidianjulua.github.io/RepliBuild.jl/dev/introspect/)
- [MLIR / JLCS Dialect](https://obsidianjulua.github.io/RepliBuild.jl/dev/mlir/)
- [Architecture](https://obsidianjulua.github.io/RepliBuild.jl/dev/architecture/)
- [Changelog](CHANGELOG.md)

## License

MIT
