# RepliBuild.jl

```@meta
CurrentModule = RepliBuild
```

ABI-aware C/C++ compiler bridge for Julia. Point it at source code, get type-safe Julia bindings — correct struct layouts, enum definitions, calling conventions, inheritance, and virtual dispatch — without writing a single `ccall` by hand.

**New to RepliBuild?** Read [Why RepliBuild](why-replibuild.md) for the design rationale, the DWARF-as-source-of-truth approach, and how offsets, vtable slots, and packing flags flow from the compiler into Julia bindings.

## Overview

RepliBuild compiles your C/C++ source with Clang, then combines multiple information sources to generate bindings that are correct by construction:

- **DWARF debug metadata** — struct member offsets, sizes, function signatures, vtable layout, base-class subobject offsets, bitfield positions. This is the compiler's own record of what it produced — always accurate for the target platform.
- **Symbol tables** (`nm`) — mangled C++ names and function addresses. The authoritative linking identity.
- **Clang.jl AST** — enums the compiler optimized away, function pointer typedefs, macro definitions. Fills gaps where DWARF is incomplete.
- **Cross-verification** — DWARF struct size is checked against Julia's alignment calculation. If they disagree, the struct is packed and gets routed to an MLIR thunk instead of `ccall` (which would silently misalign fields).

Functions are automatically routed to one of three calling tiers — `Base.llvmcall` with LTO bitcode, MLIR thunks, or `ccall` — based on ABI complexity.

### Three-tier dispatch

| Tier | Mechanism | When selected |
|------|-----------|---------------|
| 1 | `Base.llvmcall` | POD args, scalar/pointer return, LTO bitcode available |
| 2 | MLIR thunks (`libJLCS.so`) | Packed structs, unions, large struct returns, C++ classes, virtual dispatch, exceptions |
| 3 | `ccall` | Direct call into the `.so`; the unconditional fallback |

Tier selection is automatic — the wrapper generator analyses each function signature against DWARF metadata and emits the appropriate calling convention.

!!! note "Current Tier 1 status"
    Tier 1 embeds the library's linked LLVM bitcode into the wrapper for `Base.llvmcall`. The mechanism is real and verified, but whole-module embedding is **scale-limited**: at whole-library scale (hundreds of functions) it can crash Julia's JIT, and it duplicates file-local `static` state between the embedded IR and the `.so` (calls through Tier 1 and Tier 3 can then observe different internal state). Production configurations therefore set `[link] enable_lto = false` and dispatch through Tier 3; C++ projects default to LTO off. The planned fix is per-function bitcode slicing rather than whole-module embedding. See [Zero-cost LTO dispatch](guide.md#Zero-Cost-LTO-Dispatch-(current-status)) for details.

## Quick start

```julia
using RepliBuild

# Scan a C/C++ project, generate replibuild.toml, compile, and wrap
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

### Package registry and the Hub

```julia
RepliBuild.register("path/to/project/replibuild.toml")  # one-time local registration
Lua = RepliBuild.use("lua")                              # build + wrap + load, cached
Lua.luaL_newstate()

RepliBuild.search("xml")   # search the community Hub (RepliBuild-Hub) for ready-made configs
```

`use()` checks the local registry first, then fetches the package configuration from the [RepliBuild-Hub](https://github.com/obsidianjulua/RepliBuild-Hub) community registry on a miss. Builds are cached at `~/.replibuild/builds/<hash>/`; the cache key includes RepliBuild's own version, so upgrading the generator automatically rebuilds stale wrappers.

## What gets wrapped

- **Structs** with correct field order, alignment padding, and topological sort for circular references; struct-typed members resolve to named fields when the layout can be proven exact
- **Enums** via `@enum` with correct underlying types (Clang.jl AST walker)
- **Unions** as `NTuple{N,UInt8}` with typed getter/setter accessors
- **Bitfields** with exact byte-span accessors (reads and writes never touch bytes outside the field's span)
- **Function pointers** with DWARF signature parsing to `@cfunction`-compatible types
- **Variadic functions** as true variadic calls (`@ccall` semicolon form — formally correct on x86-64 SysV, including float varargs), with typed overloads via `[wrap.varargs]`
- **Macros** with auto-generated typed shims via `[wrap.macros]`
- **`char*` returns** with an ownership-aware policy: `Union{String,Nothing}` (NULL → `nothing`), declared deallocators via `[wrap.cstring_owned]`, and raw `<name>_ptr` variants for lifetime-sensitive callers
- **Multi-level pointers / references** — `T**` → `Ptr{Ptr{T}}`, `T&` → `Ref{T}`
- **C++ classes** — methods, in-place constructor/destructor thunks, `Managed` handle types with GC finalizers backed by DWARF-resolved destructors
- **C++ virtual methods** — vtable dispatch that honors overrides (a base-class wrapper invoked on a derived object reaches the override)
- **C++ inheritance** — non-virtual multiple inheritance with `<Derived>_as_<Base>` upcast helpers, and virtual inheritance with dynamic (vtable-resident) upcasts
- **C++ exceptions** — may-throw functions route through landing-pad thunks; escaped exceptions surface in Julia as `CxxException` with the original `what()` message
- **Idiomatic wrappers** — factory/destructor pairs → `mutable struct` with GC finalizers
- **Global variables** via `cglobal` accessors (unresolvable types degrade to a `_ptr` accessor rather than an unsafe getter)
- **Templates** — declare in `[types].templates`; RepliBuild forces DWARF emission

## Example: wrapping a git dependency

```toml
# replibuild.toml
[dependencies.cjson]
type    = "git"
url     = "https://github.com/DaveGamble/cJSON"
tag     = "v1.7.18"
exclude = ["test", "fuzzing"]
```

```julia
RepliBuild.build("replibuild.toml")
RepliBuild.wrap("replibuild.toml")

include("julia/MyCjsonWrapper.jl")
using .MyCjsonWrapper

obj = cJSON_CreateObject()
cJSON_AddStringToObject(obj, "key", "value")
```

## Configuration

The [`replibuild.toml`](config.md) file controls the entire build. Generated by `discover()`, hand-editable:

```toml
[project]
name = "MyEngine"

[compile]
flags      = ["-O3", "-std=c++17", "-fPIC"]
parallel   = true
aot_thunks = false           # true → pre-compile MLIR thunks into <name>_thunks.so

[link]
enable_lto         = false   # true → emit _lto.bc for Base.llvmcall (Tier 1; see status note above)
optimization_level = "3"

[wrap]
language     = "cpp"         # "c" | "cpp" (auto-detected by discover())
use_clang_jl = true

[types]
strictness = "warn"
templates  = ["std::vector<int>"]
template_headers = ["<vector>"]

[dependencies.mylib]
type = "git"
url  = "https://github.com/example/mylib"
tag  = "v1.0.0"
```

Hand-curated sections survive re-discovery: `discover(force=true)` preserves user-intent keys (`[types].templates`/`template_headers`, `[wrap].varargs`/`macros`/`shim_headers`/`cstring_owned`) instead of regenerating them empty.

See the [Configuration Reference](config.md) for all available options.

## System requirements

- **Julia 1.10+** (developed on 1.12).
- **C projects need no external LLVM.** Compilation uses the Clang JLL shipped with the Julia ecosystem, and the link/optimize/assemble steps run in-process on Julia's own libLLVM — version-matched by construction.
- **C++ projects and Tier 2 need a system LLVM/MLIR toolchain (21+)** for the JLCS dialect (`libJLCS.so`) and the external thunk pipeline, plus CMake 3.20+ and `mlir-tblgen` to build the dialect (`cd src/mlir && ./build.sh`).
- Run `RepliBuild.check_environment()` to see exactly which tiers are available on your system, with install instructions for anything missing.

## Documentation

- **[Why RepliBuild](why-replibuild.md)** — Design rationale: DWARF as source of truth, offset stability, how DWARF + symbols + AST combine, and what the JLCS dialect adds
- **[How It Works](how-it-works.md)** — Two JITs, one IR: how Julia and C++ converge at the LLVM level
- **[Architecture](architecture.md)** — Full system architecture, pipeline stages, tier dispatch, design decisions
- **[User Guide](guide.md)** — Workflows, dependencies, LTO, AOT thunks, templates, registry, ingest
- **[Configuration Reference](config.md)** — Complete `replibuild.toml` option reference
- **[Release Notes](release-notes.md)** — What changed in v2.5.8 through v3.0.1
- **[API Reference](api.md)** — Public API documentation
- **[MLIR / JLCS Dialect](mlir.md)** — Custom MLIR dialect, type system, operations, JIT manager
- **[Benchmarks](benchmarks.md)** — Zero-copy benchmark data
- **[RepliBuildTooling.jl](https://github.com/obsidianjulua/RepliBuildTooling.jl)** — Companion package: binary analysis, IR inspection, benchmarking, dataset export
- **[Internals](internals.md)** — Module architecture for contributors
