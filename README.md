# RepliBuild.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://obsidianjulua.github.io/RepliBuild.jl/dev/)
[![Julia 1.10+](https://img.shields.io/badge/julia-1.10+-9558B2?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RepliBuild is an ABI-aware C/C++ bridge for Julia. It compiles (or ingests) a library, reads the **DWARF debug metadata out of the compiled binary**, and generates a Julia module in which every function, struct, enum, union, and bitfield is callable with the calling convention and memory layout the compiler *actually emitted* — no hand-written bindings, no header annotations, no generated files to maintain.

The learning curve is one sentence: **do the opposite of every other wrapper tool.** You never write or edit bindings. You drive RepliBuild like a CLI tool with its own toolkit — point it at source or a `.so`, describe the library's quirks in a TOML, and load the result.

## The inversion

Every mainstream binding generator (Clang.jl, rust-bindgen, SWIG, jextract, …) works the same way: parse the headers, trust them, emit declarations, and hand the hard ABI cases to you. RepliBuild inverts each step:

| Header-based generators | RepliBuild |
|---|---|
| Parse headers and **trust** the source | Reads **DWARF from the compiled binary** — layout truth, not source truth |
| You maintain binding declarations / a generated file in your repo | You maintain a `replibuild.toml`; the module is disposable output, regenerated on demand |
| By-value structs, sret, packed layouts, bitfields: hope the host FFI copes, or write shims | Every function is **classified against the real ABI**; struct layouts are *proven* exact or kept opaque — never approximated |
| Wrong bindings fail at runtime, silently if you're unlucky | Unprovable ABI crossings **refuse loudly** at the call site with an explanation |
| Library quirks get patched into generator forks | Quirks are **declared in the TOML**: macros, varargs signatures, `char*` ownership |

The result is a different failure model: RepliBuild would rather tell you "this call cannot be made safely, here's why" than produce a binding that corrupts memory.

## 60 seconds to a working library

```julia
using RepliBuild

C = RepliBuild.use("cjson")        # fetch config from the Hub, build, wrap, load, cache

doc = C.cJSON_Parse("""{"answer": 42}""")
C.cJSON_GetNumberValue(C.cJSON_GetObjectItem(doc, "answer"))   # 42.0
C.cJSON_Print(doc)                 # String — the malloc'd C buffer is freed for you
C.cJSON_Delete(doc)
```

The [RepliBuild Hub](https://github.com/obsidianjulua/RepliBuild-Hub) carries configs for lua, sqlite, zlib, box2d, duktape, lz4, xxhash, cglm, and more — `RepliBuild.search("json")` to browse. Builds are content-addressed and cached at `~/.replibuild/`, keyed on sources, config, **and the generator itself**, so upgrading RepliBuild transparently regenerates stale wrappers.

**Your own project** — three verbs, or one:

```julia
toml = RepliBuild.discover("path/to/project")   # scan sources, emit replibuild.toml
RepliBuild.build(toml)                          # clang → LLVM IR → .so + DWARF
RepliBuild.wrap(toml)                           # DWARF → Julia module

RepliBuild.discover("path/to/project", build=true, wrap=true)   # or all at once
```

**Prebuilt library** (autotools, CMake soup, vendor `.so`) — skip the build, ingest the artifact:

```julia
RepliBuild.ingest("/path/to/libfoo.so", headers=["/path/to/include"],
                  build=true, wrap=true)        # requires the .so built with -g
```

The toolkit verbs: `discover`, `build`, `wrap`, `use`, `ingest`, `register`, `search`, `scaffold_package`, `check_environment`, `clean`, `info`.

## How it works

```
C/C++ source + [dependencies]                    or: existing .so (ingest mode)
       │
DependencyResolver ── clone pinned git deps, filter excludes
       │
Discovery          ── #include graph → replibuild.toml
       │
Compiler           ── clang → per-file LLVM IR (incremental, fingerprinted cache)
       │
Linker             ── .so + optional LTO bitcode / AOT thunks
       │
DWARFParser        ── llvm-dwarfdump + nm → types, layouts, vtables, symbols
       │
DispatchLogic      ── per-function ABI classification → tier routing
       │
Wrapper            ── Julia module: structs, enums, functions, docs, safety traps
```

Every function is routed to one of three call mechanisms:

| Tier | Mechanism | Notes |
|------|-----------|-------|
| **3** | `ccall` into the `.so` | The production default — what Hub configs use |
| **2** | MLIR thunks via the JLCS dialect | C++ ABI cases `ccall` can't express: virtual dispatch, packed/large struct returns, exception-safe calls |
| **1** | `Base.llvmcall` with LTO bitcode | C IR merged into Julia's JIT (cross-language inlining). Real, but currently scale-limited and opt-in — see caveats below |

Struct emission follows one rule — **exact or opaque, never approximate**. The generator types every member and *proves* that Julia's layout reproduces each DWARF offset and the total size; on success you get named fields, on any doubt an opaque byte blob with accessors. Blobs that could silently misclassify under the x86-64 SysV ABI (float-bearing or packed, ≤16 bytes, crossing by value) generate a loud `error()` stub instead of a corrupting call.

## What gets wrapped

- **Structs** — proven layouts with named fields and explicit padding; topological ordering; forward/circular references
- **Enums** — `@enum` with correct underlying types, duplicate-value handling, header-only enum recovery
- **Unions & bitfields** — byte-blob backing with typed accessors; bitfields read/write their exact byte span
- **Macros** — `[wrap.macros]` compiles typed C shims for value- and function-like macros (`SQLITE_OK()`, `deflateInit(strm, level)`), so macro-only APIs become callable
- **Variadic functions** — `[wrap.varargs]` declares typed overloads, emitted as true variadic calls (`@ccall` semicolon form — correct AL/XMM protocol on SysV)
- **`char*` returns** — `Union{String,Nothing}` (NULL is a value, not an exception) plus a raw `_ptr` variant; `[wrap.cstring_owned]` declares malloc'd returns and the wrapper frees them through the library's own deallocator
- **Function pointers** — DWARF-derived `@cfunction` signatures in the docstrings
- **Globals** — value + pointer accessors (pointer-only when the type can't be proven)
- **C++** — virtual methods through vtable thunks, exception capture (`CxxException`), template instantiation on request, STL wrappers (`CppVector`, `CppString`, `CppMap`), factory/destructor pairs clustered into GC-finalized managed types

## The TOML is the interface

`discover()` generates it; you edit it. All library-specific knowledge lives here — never in generator code:

```toml
[project]
name = "cjson"

[dependencies.cjson]
type = "git"
url  = "https://github.com/DaveGamble/cJSON.git"
tag  = "v1.7.18"                    # pin tags, not branches
exclude = ["test", "fuzzing"]

[compile]
flags      = ["-O2", "-fPIC"]
aot_thunks = false                  # true → pre-compile Tier-2 thunks to _thunks.so

[link]
enable_lto = false                  # false → Tier 3 ccall (production default)

[wrap]
language     = "c"                  # "c" | "cpp" (auto-detected)
shim_headers = ["cJSON.h"]          # headers the macro shims #include

[wrap.macros.CJSON_VERSION_MAJOR]   # value macro → CJSON_VERSION_MAJOR()
ret = "int"

[wrap.varargs]                      # typed overloads for variadic functions
# fmt_fn = [["Cint"], ["Cstring", "Cint"]]

[wrap.cstring_owned]                # malloc'd char* returns: copy, then free via
cJSON_Print = "cJSON_free"          # the library's own deallocator

[types]
strictness = "warn"                 # "strict" | "warn" | "permissive"
templates  = ["std::vector<int>"]   # C++: force DWARF for these instantiations

[cache]
enabled = true
```

## The JLCS MLIR dialect (Tier 2)

MLIR's dialect ecosystem is almost entirely about compute lowering; RepliBuild uses it for something nobody else does — **ABI marshalling as first-class IR**. The JLCS dialect (TableGen-defined, `src/mlir/`) models C/C++ interop semantics directly: `!jlcs.c_struct` types with explicit offsets and packing, `jlcs.ffe_call` / `jlcs.try_call` (exception-safe invoke + landing pad), `jlcs.vcall` (vtable dispatch), constructor/destructor ops with region-based RAII scopes.

```mlir
jlcs.type_info "Vec3", !jlcs.c_struct<"Vec3", [f32, f32, f32], [0, 4, 8], packed = false>, "", ""
%ret = jlcs.try_call %arg0 { callee = @_Z12might_throwi } : (i32) -> i32   // C++ exception → CxxException
```

Ops lower to LLVM IR and execute via MLIR JIT or ahead-of-time `_thunks.so`. Full op reference: [MLIR / JLCS docs](https://obsidianjulua.github.io/RepliBuild.jl/dev/mlir/).

## Requirements

- **Julia 1.10+** (developed on 1.12)
- **C libraries: nothing else.** The C pipeline is self-contained — clang ships via `Clang_unified_jll`, and link/optimize/assemble run in-process on Julia's own libLLVM, version-matched by construction.
- **C++ / Tier 2: system LLVM + MLIR 21+**, plus CMake 3.20+ and `mlir-tblgen` to build the dialect (`cd src/mlir && ./build.sh`).

`RepliBuild.check_environment()` reports which tiers your machine supports.

## Scope and honest limits

- **Single-target today:** ABI classification is x86-64 SysV (Linux). Win64/AAPCS are not modeled yet.
- **Tier 1 is parked:** `Base.llvmcall` embeds the whole linked module per call site — fine at toy scale, unusable at whole-library scale, and mixed Tier-1/Tier-3 dispatch can diverge on file-local `static` state. Hub configs pin Tier 3 (`enable_lto = false`); the eventual fix is per-function bitcode slicing.
- **C++ multiple inheritance** is not modeled (single inheritance is); virtual dispatch through secondary bases would need `this`-pointer adjustment.
- The full ledger of known-unbuilt pieces lives in the repo and stays honest — see the changelog and docs.

## Battle testing

| Project | Exercises |
|---------|-----------|
| Lua 5.4 | Full VM: state, stack ops, callbacks — live-verified through `use()` |
| SQLite (amalgamation) | 300+ functions: varargs (`sqlite3_mprintf`), macros, opaque handle lifecycle |
| cJSON | Owned `char*` returns, NULL policy, leak-checked (flat RSS over 200k prints) |
| box2d3 | The ABI gauntlet: 664 exported symbols, 99 structs — all resolve to proven named-field layouts |
| Duktape, zlib, lz4, mpack, xxhash, cglm, tomlc17 | Hub packages |
| Library-free fixtures | ABI round-trip traces, convenience-overload guards, varargs emission, generator policy suite — 380+ CI tests, no toolchain required |

## Documentation

- [Why RepliBuild](https://obsidianjulua.github.io/RepliBuild.jl/dev/why-replibuild/) — DWARF as source of truth, the exact-or-opaque rule, design rationale
- [User Guide](https://obsidianjulua.github.io/RepliBuild.jl/dev/guide/) · [Configuration Reference](https://obsidianjulua.github.io/RepliBuild.jl/dev/config/) · [Introspection Tools](https://obsidianjulua.github.io/RepliBuild.jl/dev/introspect/) · [MLIR / JLCS](https://obsidianjulua.github.io/RepliBuild.jl/dev/mlir/) · [Architecture](https://obsidianjulua.github.io/RepliBuild.jl/dev/architecture/)
- [CHANGELOG](CHANGELOG.md) — v3.0.0 is the first registered release since 2.5.7; the **"Breaking changes since v2.5.7"** section covers everything between

## License

MIT
