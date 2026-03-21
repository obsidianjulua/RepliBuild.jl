# RepliBuild.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://obsidianjulua.github.io/RepliBuild.jl/dev/)
[![Julia 1.10+](https://img.shields.io/badge/julia-1.10+-9558B2?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RepliBuild is an ABI-aware C/C++ compiler bridge for Julia. It compiles C/C++ source through an LLVM/MLIR pipeline, introspects DWARF debug metadata and symbol tables, and emits type-safe Julia bindings with correct struct layout, enum definitions, and calling conventions — no hand-written bindings, no header annotations, no build system integration required.

The core idea: point RepliBuild at a C/C++ project, and it produces a Julia module where every function, struct, enum, union, template instantiation, and virtual method is callable with correct ABI semantics. Functions are automatically routed to one of three calling tiers based on ABI complexity.

## Three-Tier Dispatch

Every function signature is analyzed against DWARF metadata at wrap time. The wrapper generator selects the most efficient calling mechanism that can handle the function's ABI requirements:

| Tier | Mechanism | When Selected | Overhead |
|------|-----------|---------------|----------|
| **1** | `Base.llvmcall` with LTO bitcode | POD args, scalar/pointer return, LTO bitcode available | Zero — C/C++ IR merges into Julia's JIT, enabling cross-language inlining and vectorization |
| **2** | MLIR AOT/JIT thunks via `libJLCS.so` | Packed structs, unions, large struct return, virtual dispatch, exception-throwing functions | One thunk indirection — compiled once, cached with lock-free read path |
| **3** | `ccall` | Fallback when bitcode is unavailable | Standard FFI overhead |

**Tier 1** — LTO artifacts are LLVM bitcode assembled by `Clang_unified_jll` to guarantee version-matched IR with Julia's internal LLVM. The C/C++ IR merges directly into Julia's JIT pipeline. This enables cross-language optimizations: inlining, SIMD vectorization, and compatibility with AD tools like Enzyme.jl. Enabled by default for pure-C projects.

**Tier 2** — A custom MLIR dialect (`jlcs`) handles ABI marshalling that `llvmcall` and `ccall` cannot express. This includes packed struct returns, C++ virtual dispatch through vtables, scoped RAII lifetime management, and C++ exception catching. Thunks can be JIT-compiled on first call or pre-compiled to a `_thunks.so` at build time (`aot_thunks = true`).

**Tier 3** — Direct `ccall` with zero setup. The unconditional fallback.

## The JLCS MLIR Dialect

RepliBuild includes a custom MLIR dialect — **JLCS** (Julia C-Struct) — that models C/C++ ABI semantics as first-class IR operations. The dialect is defined in TableGen (`src/mlir/`), compiled to `libJLCS.so`, and accessed from Julia via C API wrappers.

### Types

| Type | Mnemonic | Purpose |
|------|----------|---------|
| `CStructType` | `!jlcs.c_struct<"Name", [types], [offsets], packed=bool>` | C-ABI struct with explicit field types, byte offsets, and packing. The core type for representing any C/C++ struct with layout fidelity. |
| `ArrayViewType` | `!jlcs.array_view<element, rank>` | Universal strided array descriptor — maps to Julia `Array{T,N}`, NumPy ndarray, and C++ strided arrays for zero-copy interop. |

### Operations

**Metadata and field access:**

| Op | Mnemonic | Description |
|----|----------|-------------|
| `TypeInfoOp` | `jlcs.type_info` | Declares a C struct type with its Julia name, MLIR struct type, C++ base class (inheritance), and destructor symbol. Module-level metadata. |
| `GetFieldOp` | `jlcs.get_field` | Reads a struct field at a byte offset. Lowers to GEP + load. |
| `SetFieldOp` | `jlcs.set_field` | Writes a struct field at a byte offset. Lowers to GEP + store. |

**Function calls:**

| Op | Mnemonic | Description |
|----|----------|-------------|
| `VirtualCallOp` | `jlcs.vcall` | Calls a C++ virtual method: loads the vtable pointer from the object, indexes the function pointer by slot, and calls with the object as `this`. |
| `FFECallOp` | `jlcs.ffe_call` | Foreign function execution — calls an external C/C++ function with ABI-correct argument marshalling. |
| `TryCallOp` | `jlcs.try_call` | Like `ffe_call` but emits `invoke` + landing pad to catch C++ exceptions. On catch, stores the `what()` message in a thread-local buffer and returns a sentinel. Julia checks and throws `CxxException`. |

**Array access:**

| Op | Mnemonic | Description |
|----|----------|-------------|
| `LoadArrayElementOp` | `jlcs.load_array_element` | Loads from a strided multi-dimensional array view using linearized offset computation. |
| `StoreArrayElementOp` | `jlcs.store_array_element` | Stores to a strided multi-dimensional array view. |

**RAII (C++ object lifetime):**

| Op | Mnemonic | Description |
|----|----------|-------------|
| `ConstructorCallOp` | `jlcs.ctor_call` | Calls a C++ constructor with `this` pointer + parameters. |
| `DestructorCallOp` | `jlcs.dtor_call` | Calls a C++ destructor on an object pointer. |
| `ScopeOp` | `jlcs.scope` | Region-based RAII scope — takes managed object pointers and destructor symbols. Body ops are inlined during lowering; destructors are emitted in reverse order (C++ semantics). |
| `YieldOp` | `jlcs.yield` | Terminator for `jlcs.scope` regions. |

### Example IR

```mlir
// Declare a packed struct with field offsets
jlcs.type_info "Vec3", !jlcs.c_struct<"Vec3", [f32, f32, f32], [0, 4, 8], packed = false>, "", ""

// RAII scope: construct, use, auto-destruct
%alloca = llvm.alloca 1 x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr
jlcs.scope(%alloca : !llvm.ptr) dtors([@_ZN4BaseD1Ev]) {
  jlcs.ctor_call @_ZN4BaseC1Ei(%alloca, %val) : (!llvm.ptr, i32) -> ()
  %result = jlcs.vcall @Base::area(%alloca) { vtable_offset = 0, slot = 2 } : (!llvm.ptr) -> f64
  jlcs.yield
}

// Exception-safe call
%ret = jlcs.try_call %arg0 { callee = @_Z12might_throwi } : (i32) -> i32
```

All JLCS ops lower to LLVM IR via `JLCSPasses.cpp`. The lowered IR is executed either through the MLIR JIT engine (on-demand) or compiled ahead-of-time into `_thunks.so`.

## What Gets Wrapped

RepliBuild introspects DWARF debug metadata and symbol tables to generate bindings for:

- **Structs** — Correct field order, alignment padding, topological dependency sort, `Ptr{X}` soft-dependency handling for circular references
- **Enums** — `@enum` with correct underlying types; Clang.jl AST walker handles `enum class`, hex values, namespaces
- **Unions** — `NTuple{N,UInt8}` backing with typed getter/setter accessors
- **Bitfields** — Bit-level extraction from packed representations
- **Function pointers** — DWARF signature parsing to `@cfunction`-compatible type strings
- **Variadic functions** — Typed overloads declared in `[wrap.varargs]`
- **Multi-level pointers / references** — `T**` to `Ptr{Ptr{T}}`, `T&` to `Ref{T}`
- **C++ virtual methods** — MLIR JIT thunks or static AOT thunks via vtable slot dispatch
- **C++ exceptions** — Functions that throw are routed through `jlcs.try_call`; exceptions surface as `CxxException` in Julia
- **Templates** — Declare `templates = ["std::vector<int>"]`; RepliBuild forces Clang to emit DWARF for those instantiations
- **STL containers** — `CppVector{T}`, `CppString`, `CppMap{K,V}` wrappers with Julia `AbstractArray`/`AbstractDict` interfaces
- **Idiomatic wrappers** — Factory/destructor pairs clustered by class name into `mutable struct` with GC finalizers and multiple-dispatch method proxies
- **Global variables** — `cglobal` accessors

## Pipeline

```
C/C++ Source + [dependencies]
    |
    v
DependencyResolver   -- clone/update git deps, filter excludes, inject into compile graph
    |
    v
Discovery            -- scan files, resolve #include graph, emit replibuild.toml
    |
    v
Compiler             -- Clang/clang++ -> per-file LLVM IR; incremental mtime + project-hash cache
    |
    v
Linker               -- llvm-link + llvm-opt -> .so/.dylib + _lto.bc (Tier 1) + _thunks.so (Tier 2)
    |
    v
DWARFParser          -- llvm-dwarfdump + nm -> ClassInfo / VtableInfo / MemberInfo structs
    |
    v
DispatchLogic        -- per-function tier routing: is_ccall_safe(), is_c_lto_safe()
    |
    v
Wrapper              -- DWARF + symbols -> Julia module (C and C++ generators are independent)
    |
    v
JITManager           -- on-demand: JLCSIRGenerator -> MLIR JLCS IR -> MLIRNative JIT -> thunk cache
```

### Caching

Two independent layers ensure sub-second rebuilds:

- **Per-file IR cache** (`.replibuild_cache/`) — mtime-based, skips individual source files whose IR is current.
- **Project content hash** — SHA256 of `replibuild.toml` + all source/header contents + git HEAD. If the hash matches, `build()` exits without invoking any compiler.

### Key Modules

| Module | Role |
|--------|------|
| `Compiler.jl` | Core build engine: per-file LLVM IR, incremental cache, parallel compilation, LTO linking, bitcode assembly |
| `DWARFParser.jl` | Parses `llvm-dwarfdump` + `nm` output into structured `ClassInfo`, `VtableInfo`, `MemberInfo` |
| `Wrapper/Generator.jl` | Top-level `wrap_library()` — routes to C or C++ generator based on language config |
| `Wrapper/C/GeneratorC.jl` | C wrapper generator: struct packing, enums, bitfields, unions, varargs |
| `Wrapper/Cpp/GeneratorCpp.jl` | C++ wrapper generator: templates, STL, vtable thunks, managed types |
| `Wrapper/DispatchLogic.jl` | Per-function tier routing decisions (`is_ccall_safe`, `is_c_lto_safe`) |
| `JLCSIRGenerator.jl` | DWARF types to MLIR JLCS IR generation |
| `MLIRNative.jl` | Julia `ccall` bindings to `libJLCS.so` C API |
| `JITManager.jl` | MLIR JIT lifecycle, atomic copy-on-write symbol cache, `CxxException` handling |
| `Discovery.jl` | Project scanning, `#include` graph resolution, `replibuild.toml` generation |
| `PackageRegistry.jl` | Content-addressed build cache at `~/.replibuild/`, `use()`/`register()` API |

## Quick Start

```julia
using RepliBuild

# One call: scan, compile, and wrap a C/C++ project
RepliBuild.discover("path/to/project", build=true, wrap=true)

# Load the generated module
include("path/to/project/julia/MyProject.jl")
using .MyProject
```

Or step by step:

```julia
toml = RepliBuild.discover("path/to/project")  # generates replibuild.toml
RepliBuild.build(toml)                          # Clang -> LLVM IR -> .so + DWARF
RepliBuild.wrap(toml)                           # DWARF -> Julia module
```

### Package Registry

```julia
RepliBuild.register("path/to/project/replibuild.toml")  # one-time registration
Lua = RepliBuild.use("lua")                              # build + wrap + load, cached
Lua.luaL_newstate()

RepliBuild.search("xml")                         # search the RepliBuild Hub
RepliBuild.list_registry()                       # print all registered packages
```

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
enable_lto         = false   # true -> emit _lto.bc for Base.llvmcall (Tier 1)

[binary]
type           = "shared"    # "shared" | "static" | "executable"
strip_symbols  = false

[wrap]
language     = "cpp"         # "c" | "cpp" (auto-detected by discover())
use_clang_jl = true
aot_thunks   = false         # true -> pre-compile MLIR thunks to _thunks.so (Tier 2)

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

## Requirements

- Julia 1.10+
- LLVM 21+ and Clang (system install; auto-detected, JLL fallback for Tier 1)
- CMake 3.20+ and `mlir-tblgen` (Tier 2 only)

Run `RepliBuild.check_environment()` to verify which tiers are available.

## Test Coverage

| Project | Description |
|---------|-------------|
| Lua 5.4.6 | Full VM + stdlib: state management, stack ops, callbacks, coroutines |
| SQLite 3.49.1 | 261 K-line C API: varargs, opaque pointer lifecycle |
| Duktape 2.7.0 | 101 K-line JS engine amalgamation: monolithic compile, stack round-trips |
| cJSON | Multi-file C library: git dependency resolution |
| Stress test | Vectors, matrices, numerics, vtable dispatch, RAII, MLIR/AOT |
| Callback test | Bidirectional FFI with C++ exception propagation |
| STL test | `CppVector`, `CppString`, `CppMap` lifecycle and interop |

## Documentation

- [Why RepliBuild](https://obsidianjulua.github.io/RepliBuild.jl/dev/why-replibuild/) — What it solves, comparison to alternatives
- [User Guide](https://obsidianjulua.github.io/RepliBuild.jl/dev/guide/)
- [Configuration Reference](https://obsidianjulua.github.io/RepliBuild.jl/dev/config/)
- [Introspection Tools](https://obsidianjulua.github.io/RepliBuild.jl/dev/introspect/)
- [MLIR / JLCS Dialect](https://obsidianjulua.github.io/RepliBuild.jl/dev/mlir/)
- [Architecture](https://obsidianjulua.github.io/RepliBuild.jl/dev/architecture/)
- [Changelog](CHANGELOG.md)

## License

MIT
