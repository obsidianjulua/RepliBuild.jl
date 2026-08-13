# RepliBuild.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://obsidianjulua.github.io/RepliBuild.jl/dev/)
[![Julia 1.10+](https://img.shields.io/badge/julia-1.10+-9558B2?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RepliBuild is an ABI-aware C/C++ bridge for Julia. It **compiles a library from source**, reads the **DWARF debug metadata out of the compiled binary**, and generates a Julia module in which every function, struct, enum, union, and bitfield is callable with the calling convention and memory layout the compiler *actually emitted* — no hand-written bindings, no header annotations, no generated files to maintain.

The learning curve is one sentence: **do the opposite of every other wrapper tool.** You drive RepliBuild like a CLI tool with its own toolkit — point it at source, describe the library's quirks in a TOML, and load the result. (There is also an experimental `ingest` mode for prebuilt C binaries — see below — but the source build is the tool: RepliBuild's own compilation is what guarantees the DWARF it consumes.)

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

**Using the generated wrapper** — `wrap()` writes a self-contained module next to the `.so` (they travel as a unit); load it like any Julia file:

```julia
include("path/to/project/julia/MyProject.jl")   # module name = CamelCased project name
using .MyProject

MyProject.some_function(...)                     # everything exported; docs on each symbol
```

Hub packages skip this — `RepliBuild.use("name")` returns the loaded module directly.

Building a real package on top of a wrapper (vendoring layout, precompilation, JIT lifecycle, C++ object lifetimes, multiple wrapped libraries) is covered in the manual page **"Using a Wrapper in Your Package"**, with a working reference app at `RepliBuild-Hub/examples/BoxWorld`.

**Prebuilt library — `ingest` mode (EXPERIMENTAL, C only).** For **C** libraries whose build systems the source pipeline can't reproduce (autotools, CMake with code generators), you can skip the build and wrap an existing binary:

```julia
RepliBuild.ingest("/path/to/libfoo.so", headers=["/path/to/include"],
                  language=:c, build=true, wrap=true)   # .so must be built with -g
```

Know what you're getting: ingest is a fallback, not the flagship. It dispatches through plain `ccall` only (no bitcode, no MLIR thunks), and extraction is best-effort — the binary was built by a compiler and debug-info configuration RepliBuild doesn't control. **C++ API surfaces are not supported in ingest mode**: classes, methods, templates, and virtual dispatch need the MLIR dialect marshalling that only the source-build pipeline generates. For a C++ library, build it from source with RepliBuild, or ingest its C API variant if it ships one.

The toolkit verbs: `discover`, `build`, `wrap`, `use`, `ingest`, `register`, `search`, `scaffold_package`, `check_environment`, `clean`, `info`.

## How it works

```
C/C++ source + [dependencies]        or: existing C .so (ingest mode, experimental)
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
| **1** | `Base.llvmcall` on a per-function bitcode slice | C IR merged into Julia's JIT (cross-language inlining). C only, opt-in via `[wrap.tier1] enable` — see below |

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

`discover()` generates it; you edit it. All library-specific knowledge lives here — never in generator code. Discovery sees the *shape* of a source tree and nothing else, so the things it cannot infer are things you state: template instantiations (an uninstantiated `std::vector<int>` generated no code, so it is not in the DWARF — **STL support means STL in the TOML**), macros, vararg signatures, `char*` ownership, and the flags upstream's build system would have supplied. The reference page enumerates all of them, per class of library, with the symptom you get for each omission.

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
enable_lto = false                  # whole-module llvmcall payload — leave off (see below)

[wrap]
language     = "c"                  # "c" | "cpp" (auto-detected)
shim_headers = ["cJSON.h"]          # headers the macro shims #include

[wrap.tier1]
enable = true                       # C only → Base.llvmcall on per-function bitcode slices

[wrap.macros.CJSON_VERSION_MAJOR]   # value macro → CJSON_VERSION_MAJOR()
ret = "int"

[wrap.varargs]                      # typed overloads: JULIA types, variadic args only
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

A foreign call is a compilation problem, so RepliBuild compiles it. Where a binding generator would paste a C shim or interpret a signature at runtime, RepliBuild emits a small program in a purpose-built MLIR dialect, lowers it, and runs the result — **ABI marshalling as first-class IR**, which as far as this project knows nobody else does with MLIR.

The JLCS dialect (TableGen-defined, `src/mlir/`) models C/C++ interop semantics directly: `!jlcs.c_struct` types carrying explicit field offsets and packing, `jlcs.ffe_call` / `jlcs.try_call` (exception-safe invoke + landing pad), `jlcs.vcall` (vtable dispatch that honors overrides), `jlcs.marshal_arg` / `marshal_ret` (Julia-aligned ↔ C-packed), and constructor/destructor ops inside region-based RAII scopes for Itanium's non-trivial by-value parameters.

```mlir
func.func @_ZNK5Base25get_bEv_thunk(%args_ptr: !llvm.ptr) -> i32
    attributes { llvm.emit_c_interface } {
  %arg_ptr_1 = llvm.getelementptr %args_ptr[%idx_1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
  %val_ptr_1 = llvm.load %arg_ptr_1 : !llvm.ptr -> !llvm.ptr     // slot → storage
  %val_1     = llvm.load %val_ptr_1 : !llvm.ptr -> !llvm.ptr     // storage → `this`
  %ret_val   = "jlcs.vcall"(%val_1) { class_name = @Base2, slot = 2 : i64, … } : (!llvm.ptr) -> i32
  return %ret_val : i32
}
```

Four things fall out of choosing an IR over a shim: the struct offsets in the marshalling code and in the Julia wrapper are the *same DWARF numbers*, read once; the ops carry verifiers, so a malformed thunk fails at parse instead of at runtime; the x86-64 SysV rules live in one readable pass (`classifySysVStruct`) rather than being implied by a code generator; and because the emitted dialect is written to disk and the JIT registers DWARF pointing at it, **gdb stops inside the generated MLIR by file and line** — `disassemble /s` interleaves dialect ops with the machine code they became. Ops execute through a per-library MLIR JIT engine, or ahead-of-time in a companion `_thunks.so`.

Full treatment — the thesis, the op reference, the lowering, source-level debugging, and the failure classes the design exists to make loud: [ABI Marshalling as Compiler IR](https://obsidianjulua.github.io/RepliBuild.jl/dev/mlir/).

## Requirements

- **Julia 1.10+** (developed on 1.12)
- **C libraries: nothing else.** The C pipeline is self-contained — clang ships via `Clang_unified_jll`, and link/optimize/assemble run in-process on Julia's own libLLVM, version-matched by construction.
- **C++ / Tier 2: system LLVM + MLIR 21+**, plus CMake 3.20+ and `mlir-tblgen` to build the dialect (`cd src/mlir && ./build.sh`).

`RepliBuild.check_environment()` reports which tiers your machine supports.

## Scope and honest limits

- **Single-target today:** ABI classification is x86-64 SysV (Linux). Win64/AAPCS are not modeled yet.
- **Tier 1 runs on slices, and only for C.** `[wrap.tier1] enable` cuts a declarations-only module per function — one body, everything it reaches left as a `declare` bound to the `.so` at JIT time — so slice size tracks the function, not the library (`lua_gettop`: 15.8 MB → 2.8 KB). Live at scale on Lua: 209 slices accepted, 190 emitted across 189 functions, the rest cleanly on `ccall`. Default off, and C++ has no slicing path. The **older whole-module payload (`[link] enable_lto`) stays parked**: it embeds the entire linked module per call site, which is unusable at library scale and diverges on file-local `static` state between the embedded copy and the `.so`. Hub configs keep `enable_lto = false`. Perf at library scale is not yet characterized — the 3.3×/call figure is one spiked function, not a suite.
- **C++ inheritance is modeled, with two gates.** Non-virtual multiple inheritance and virtual inheritance both work — static `<Derived>_as_<Base>` upcasts, dynamic vtable-resident `<Derived>_as_<VBase>` upcasts, and class-local `jlcs.vcall` dispatch that honors overrides (`test/mi_test/` 38/38, `test/vi_test/` 33/33). Still direct-called by design: destructors (exact-class semantics for finalizers and RAII), and virtual methods with struct-shaped signatures, since the vcall lowering does no sret/packed coercion yet.
- **Tier 2 needs a system LLVM/MLIR install.** It is the largest dependency in the project; C-only libraries never touch it.
- The full ledger of known-unbuilt pieces lives in the repo and stays honest — see the changelog and [the boundaries section](https://obsidianjulua.github.io/RepliBuild.jl/dev/mlir/) of the MLIR page.

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

- [User Guide](https://obsidianjulua.github.io/RepliBuild.jl/dev/guide/) — `discover → build → wrap → use`, dependencies, templates, macros, varargs, AOT thunks, ingest
- [Configuration Reference](https://obsidianjulua.github.io/RepliBuild.jl/dev/config/) — every `replibuild.toml` key with its true default, **what discovery cannot know and you must declare**, what fails loudly vs. what is silently ignored, and a symptom → key troubleshooting index
- [ABI Marshalling as Compiler IR](https://obsidianjulua.github.io/RepliBuild.jl/dev/mlir/) — the JLCS dialect: why marshalling is compiled IR, the thunk contract, the op reference, the SysV lowering, debugging a thunk in gdb
- [Using a Wrapper in Your Package](https://obsidianjulua.github.io/RepliBuild.jl/dev/using-wrappers/) · [The Inheritance ABI](https://obsidianjulua.github.io/RepliBuild.jl/dev/inheritance-abi/) · [Internals & Dispatch](https://obsidianjulua.github.io/RepliBuild.jl/dev/internals/) · [API Reference](https://obsidianjulua.github.io/RepliBuild.jl/dev/api/)
- [CHANGELOG](CHANGELOG.md) — v3.0.0 is the first registered release since 2.5.7; the **"Breaking changes since v2.5.7"** section covers everything between

## License

MIT
