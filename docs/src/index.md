# RepliBuild.jl

```@meta
CurrentModule = RepliBuild
```

ABI-aware C/C++ compiler bridge for Julia. Point it at source code, get type-safe Julia bindings — correct struct layouts, enum definitions, calling conventions, inheritance, and virtual dispatch — without writing a single `ccall` by hand.

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
| 1 | `Base.llvmcall` on a per-function bitcode slice | POD args, scalar/pointer return, `[wrap.tier1] enable = true` |
| 2 | MLIR thunks (`libJLCS.so`) | Packed structs, unions, large struct returns, C++ classes, virtual dispatch, exceptions |
| 3 | `ccall` | Direct call into the `.so`; the unconditional fallback |

Tier selection is automatic — the wrapper generator analyses each function signature against DWARF metadata and emits the appropriate calling convention.

Tier 2 is the architectural centre of the project: RepliBuild expresses ABI marshalling as **first-class compiler IR** in a purpose-built MLIR dialect (JLCS), derived from the same DWARF that describes the types, lowered by a pass that models the x86-64 SysV ABI explicitly, and debuggable at source level in gdb. See [ABI Marshalling as Compiler IR](mlir.md) for the full treatment.

!!! note "Current Tier 1 status"
    Tier 1 runs on **per-function bitcode slices** — declarations-only modules holding one function body, with everything it reaches left as a `declare` bound to the `.so` at JIT time. Opt in per project with `[wrap.tier1] enable = true` (C only, default off); it is live at library scale (Lua: 209 slices accepted, 190 emitted across 189 functions, the rest cleanly on `ccall`). The older `[link] enable_lto` payload embeds the **whole linked module** per call site and remains scale-limited — it can crash Julia's JIT on large libraries and duplicates file-local `static` state — so production configs keep `enable_lto = false`; C++ defaults to LTO off. The two knobs are independent. See [Zero-cost LTO dispatch](guide.md#Zero-Cost-LTO-Dispatch) for details.

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
enable_lto         = false   # true → emit _lto.bc and llvmcall the whole module (scale-limited; see note above)
optimization_level = "3"

[wrap]
language     = "cpp"         # "c" | "cpp" (auto-detected by discover())
use_clang_jl = true

[wrap.tier1]                 # C only — per-function bitcode slices for Base.llvmcall
enable = false

[types]
strictness = "warn"
templates  = ["std::vector<int>"]
template_headers = ["<vector>"]

[dependencies.mylib]
type = "git"
url  = "https://github.com/example/mylib"
tag  = "v1.0.0"
```

Hand-curated sections survive re-discovery: `discover(force=true)` preserves user-intent keys (`[types].templates`/`template_headers`, `[wrap].varargs`/`macros`/`shim_headers`/`cstring_owned`/`tier1`, `[link].promote_statics`) instead of regenerating them empty.

Discovery writes what it can see. What it *cannot* infer — template instantiations, macros, vararg signatures, `char*` ownership, the flags upstream's build system supplied — you declare, and the [Configuration Reference](config.md) documents every key, every default, and [exactly what each class of library requires you to state](config.md#2.-What-discovery-cannot-know).

## System requirements

- **Julia 1.10+** (developed on 1.12).
- **C projects need no external LLVM.** Compilation uses the Clang JLL shipped with the Julia ecosystem, and the link/optimize/assemble steps run in-process on Julia's own libLLVM — version-matched by construction.
- **C++ projects and Tier 2 need a system LLVM/MLIR toolchain (21+)** for the JLCS dialect (`libJLCS.so`) and the external thunk pipeline, plus CMake 3.20+ and `mlir-tblgen` to build the dialect (`cd src/mlir && ./build.sh`).
- Run `RepliBuild.check_environment()` to see exactly which tiers are available on your system, with install instructions for anything missing.

## Documentation

- **[Workflow](guide.md)** — `discover → build → wrap → use`, dependencies, LTO, AOT thunks, templates, the registry and Hub, ingest
- **[Configuration](config.md)** — The complete `replibuild.toml` reference: what discovery cannot know and you must declare, every section and key with its true default, what fails loudly vs. what is silently ignored, worked configs, and a symptom → key troubleshooting index
- **[ABI Marshalling as Compiler IR](mlir.md)** — The JLCS MLIR dialect: why marshalling is compiled IR, the thunk contract, the op reference, the SysV lowering, source-level debugging, and the failure classes the design is built around
- **[Using a Wrapper](using-wrappers.md)** — Building a precompiled Julia package on a generated wrapper: vendoring, the JIT lifecycle, C++ object lifetimes, and the C++-isms your layer encapsulates
- **[API Reference](api.md)** — The public entry points
- **[The Inheritance ABI](inheritance-abi.md)** — How MI and virtual inheritance become callable: upcast helpers, class-local vcall dispatch
- **[Internals & Dispatch](internals.md)** — Pipeline, the three dispatch tiers, DWARF extraction, module-by-module reference, caching
- **[Release Notes](release-notes.md)** — Version history
