# Architecture

RepliBuild compiles C/C++ source through an LLVM/MLIR pipeline, introspects DWARF debug metadata emitted by the compiler itself, and generates type-safe Julia bindings with correct struct layout, enum definitions, and calling conventions. Functions requiring non-trivial ABI handling are automatically routed through a custom MLIR dialect and JIT tier.

This page documents the full system architecture. For the public API surface see [API Reference](@ref), and for per-module internals see [Internals](@ref "RepliBuild Internals").

## System overview

```
   discover("path/")          build("replibuild.toml")        wrap("replibuild.toml")
           │                            │                              │
           ▼                            ▼                              ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐
│ Builder/ config      │   │ Builder/ compile     │   │ Wrapper/ binding gen     │
├──────────────────────┤   ├──────────────────────┤   ├──────────────────────────┤
│ Discovery            │   │ Compiler             │   │ DWARFParser              │
│ ConfigurationManager │   │ BuildBridge          │   │ DispatchLogic            │
│ LLVMEnvironment      │   │ DependencyResolver   │   │ Generator (C / C++)      │
│ EnvironmentDoctor    │   │ ThunkBuilder (AOT)   │   │ Symbols, FunctionPtrs    │
└──────────────────────┘   └──────────────────────┘   └────────────┬─────────────┘
        emits                    emits                              │
   replibuild.toml          .so + DWARF + .ll                       ▼
                                                       Generated Julia module
                                                       with per-function tier:

                                                       Tier 1 → llvmcall (LTO)
                                                       Tier 3 → ccall
                                                       Tier 2 ─────────┐
                                                                       │
                                                                       ▼
                                              ┌──────────────────────────────────┐
                                              │ IRGen/                           │
                                              ├──────────────────────────────────┤
                                              │ JLCSIRGenerator                  │
                                              │   ↳ ir_gen/ submodules           │
                                              │ MLIRNative → libJLCS.so          │
                                              │ JITManager (lock-free cache)     │
                                              │ DAGDiff (struct DAG → tier       │
                                              │   decisions + lowering order)    │
                                              └──────────────────────────────────┘
```

## The dual-LLVM design

RepliBuild deliberately uses **two separate LLVM installations**, one per language bucket:

- **The C bucket needs no external LLVM.** Source is compiled to IR by the Clang JLL (`Clang_unified_jll`), and the link → optimize → assemble steps run **in-process on Julia's resident libLLVM** — the same LLVM version that will consume the IR. This version-lock eliminates the classic failure mode where an external `llvm-link` at a different version silently drops or mangles DWARF. An in-process failure is a hard error rather than a silent downgrade; `[link] fallback = true` is the escape hatch to the external `llvm-link`/`opt` pipeline.
- **The C++ bucket uses a system LLVM/MLIR toolchain (21+).** C++ requires the JLCS MLIR dialect (`libJLCS.so`, built against system MLIR) for Tier 2 thunks, and final `.ll → .so` codegen shells out to system `clang++`. The dialect is a build artifact pinned to the installed MLIR's minor version — rebuild it (`cd src/mlir && ./build.sh`) after a system MLIR upgrade.

`RepliBuild.check_environment()` reports which buckets and tiers are available.

## Pipeline stages

The lifecycle of a C/C++ project through RepliBuild proceeds in six stages. Each stage is implemented by one or more Julia modules that can be invoked independently.

### Stage 1 — Discovery

**Module:** `src/Builder/Discovery.jl`

Scans source files, parses the `#include` graph, resolves external dependencies, and emits a `replibuild.toml` configuration file. The language (`:c` or `:cpp`) is auto-detected from source file extensions and written to `wrap.language` in the generated config. New projects are automatically registered in the global registry (`~/.replibuild/registry/`).

Forced re-discovery (`discover(force=true)`) **preserves user-intent configuration**: hand-curated keys that cannot be derived from source (`[types].templates`/`template_headers`, `[wrap].varargs`/`macros`/`shim_headers`/`cstring_owned`) carry over from the existing TOML instead of being regenerated empty. A regenerated non-empty value wins; an empty slot inherits the preserved value.

### Stage 2 — Dependency resolution

**Module:** `src/Builder/DependencyResolver.jl`

Processes the `[dependencies]` table in `replibuild.toml`:

| Type | Mechanism |
|------|-----------|
| `git` | Shallow clone into `.replibuild_cache/deps/<name>/`; re-fetches when `tag` changes |
| `local` | Scanned in-place; no copy |
| `system` | `pkg-config --cflags` to inject include paths |

The `exclude` list is applied after scanning, allowing you to trim large repositories down to the files you need. Resolved source files merge into the compilation graph before stage 3.

### Stage 3 — Compilation

**Module:** `src/Builder/Compiler.jl`, `src/Builder/BuildBridge.jl`

Each source file is compiled to LLVM IR (`.ll` text format) — `.c` files via the JLL `clang`, `.cpp` files via system `clang++`. Key details:

- **Incremental cache:** Per-file caching in `.replibuild_cache/`, keyed on source `mtime` **plus a compile fingerprint** (compile flags, defines, include dirs, LLVM version, target triple, stored in a `.ll.key` sidecar). Editing one source recompiles only that file; changing flags or defines correctly busts the whole set.
- **Project content hash:** SHA-256 of `replibuild.toml` + all source + all headers + git HEAD. If the hash matches cached artifacts, `build()` returns in sub-second time.
- **Parallel dispatch:** Enabled by default (`compile.parallel = true`).
- **Template instantiation:** If `[types].templates` is set, a stub `.cpp` is auto-generated to force Clang to emit DWARF for the requested template instantiations.
- **Macro shims:** If `[wrap.macros]` is set, typed C/C++ wrapper functions are generated so preprocessor macros appear in the compiled binary's debug metadata. Shims are pinned to default symbol visibility (they survive `-fvisibility=hidden` and LTO internalization), and a **header-collision guard** verifies each shim `#include` resolves inside the project/dependency tree — a system-installed copy of the same header at a different version would otherwise silently bake wrong macro values.
- **IR sanitization:** The compiler strips attributes incompatible with Julia's internal LLVM, removes `va_start`/`va_end` intrinsics from varargs function bodies (varargs are routed through true-variadic `@ccall` wrapper generation), and cleans mismatched debug metadata.

### Stage 4 — Linking

**Module:** `src/Builder/Compiler.jl` (link phase), `src/Builder/BuildBridge.jl`, `src/Builder/ThunkBuilder.jl` (AOT thunks)

For **C**, sanitized per-file IR is linked (`LLVM.link!`), optimized (new pass manager, `default<O…>`), and assembled to bitcode **in-process on Julia's libLLVM**; final codegen to the shared library shells to clang. For **C++**, the external `llvm-link`/`opt` pipeline is used. Output is the target shared library (`.so`/`.dylib`/`.dll`).

Optional artifacts:

| Artifact | Condition | Purpose |
|----------|-----------|---------|
| `<name>_lto.bc` | `enable_lto = true` | LLVM bitcode for `Base.llvmcall` embedding (Tier 1). Assembled version-matched to Julia's LLVM. |
| `<name>_thunks.so` | `aot_thunks = true` | Pre-compiled MLIR thunks for Tier 2 dispatch without JIT startup cost. |

### Stage 5 — Wrapping

**Module:** `src/Wrapper.jl`, `src/Wrapper/` subpackages, `src/Builder/DWARFParser.jl`, `src/Builder/ASTWalker.jl`

DWARF metadata is extracted from the compiled binary. The parsers build structured types (`ClassInfo`, `VtableInfo`, `MemberInfo`, `VirtualMethod`) that feed into the wrapper generator. Parsing is **depth-aware**: DIE tree depth attributes member/enumerator/inheritance entries to the correct enclosing type even when nested type definitions are interleaved between members (clang emits a nested enum's DIE at its first point of use, mid-member-list).

The wrapper generator emits a complete, loadable Julia module containing:

- Struct definitions with correct field order, alignment padding (`_pad_N`), and topological sort for circular references. Struct-typed members resolve to **named fields** when an exact-layout proof succeeds (Julia's natural layout reproduces every DWARF offset and the total size); anything unprovable stays an opaque byte blob — exact or opaque, never approximate.
- `@enum` definitions with correct underlying types (extracted by the Clang.jl AST walker)
- Union representations as `NTuple{N,UInt8}` with typed getter/setter accessors
- Bitfield accessors with exact byte-span assembly (no out-of-bounds reads or writes at struct tails)
- Function wrappers dispatched to the appropriate tier (see [Three-tier dispatch](#Three-tier-dispatch))
- `char*` returns under the ownership-aware Cstring policy (`Union{String,Nothing}`, `[wrap.cstring_owned]` deallocators, raw `_ptr` variants)
- C++ class support: base-class member flattening with subobject-offset rebasing, `<Derived>_as_<Base>` static upcast helpers for multiple inheritance, dynamic `<Derived>_as_<VBase>` upcasts for virtual bases (the helper reads the vbase offset through the object's vtable at runtime), and `Managed` handle types whose GC finalizers call the DWARF-resolved destructor
- Idiomatic `mutable struct` wrappers with GC-managed finalizers for factory/destructor pairs
- Global variable accessors via `cglobal` (value getters only for cleanly resolved types; `_ptr` accessors otherwise)

Three generator tracks exist (`src/Wrapper/C/GeneratorC.jl`, `src/Wrapper/Cpp/GeneratorCpp.jl`, and an experimental `src/Wrapper/Rust/GeneratorRust.jl` for `extern "C"` + `#[repr(C)]` Rust), selected by `wrap.language`.

### Stage 6 — JIT initialization (on demand)

**Modules:** `src/IRGen/JLCSIRGenerator.jl`, `src/IRGen/MLIRNative.jl`, `src/IRGen/JITManager.jl`

When the generated Julia module loads, its `__init__` initializes the Tier 2 JIT:

1. Reads `thunk_manifest.json` — the list of thunks the wrapper actually dispatches to — so dead thunks are never generated (dead-thunk elimination)
2. Generates MLIR IR in the JLCS dialect from cached DWARF metadata
3. Parses the IR via `MLIRNative.parse_module()` and lowers `jlcs` → `func` → `llvm` dialect
4. Creates the execution engine. A **pre-flight type check** in `libJLCS` refuses any module containing a type that cannot be translated to LLVM IR — the failure is a catchable Julia error that degrades the module to "Tier 2 disabled" instead of a native crash
5. Caches compiled symbol pointers in a lock-free dictionary on first use

Subsequent calls to the same symbol are a single unsynchronized `Dict` read — no locks, no allocation, no JIT overhead.

If `aot_thunks = true`, thunks are pre-compiled at build time into `_thunks.so` and reached via `ccall`. No JIT initialization occurs at all.

## Three-tier dispatch

Every exported function is analyzed by the wrapper generator and routed to one of three calling tiers based on ABI complexity. Tier selection is fully automatic — no user annotation is required.

### Tier 1 — `Base.llvmcall` with LTO bitcode

For functions with simple, POD-safe signatures, when `enable_lto = true`. The library's LLVM bitcode is embedded as a module-level constant and passed to Julia's JIT, enabling **cross-language inlining** — Julia can inline C code into hot loops and vectorize across the FFI boundary.

**Eligibility (all must hold):** no STL container types; primitive/pointer/void return; primitive/pointer/small-struct parameters; not a virtual method; no struct-by-value return; no `Cstring` parameters or return; not a packed-struct or union return.

!!! warning "Tier 1 is currently scale-limited"
    Whole-module bitcode embedding has two verified failure modes at library scale:

    1. **JIT scale.** `Base.llvmcall` embeds the whole linked module per call site; at hundreds of functions this can crash Julia's JIT. Small modules work; whole libraries do not reliably.
    2. **Duplicated static state.** File-local (`static`) definitions stay internal to the embedded bitcode, so a Tier-1 call and a Tier-3 call can operate on *different copies* of internal library state (observed live: a parser's error message written to the embedded copy of a `static` buffer while the error-reporting getter read the `.so`'s copy).

    Production configurations therefore set `[link] enable_lto = false`, routing these functions through Tier 3. The eventual fix is per-function bitcode slicing rather than whole-module embedding. C projects default `enable_lto = true` for experimentation; C++ projects default it off.

### Tier 2 — MLIR thunks (JIT or AOT)

For functions requiring complex ABI marshalling: packed structs, unions, large struct returns, by-value C++ class parameters, virtual dispatch, exception propagation.

| Mode | Config | Mechanism | Startup cost | Per-call cost |
|------|--------|-----------|--------------|---------------|
| **JIT** | `aot_thunks = false` | `JITManager.invoke()` at runtime | Module-load IR generation + first-call compile | Lock-free lookup after first call |
| **AOT** | `aot_thunks = true` | Pre-compiled `_thunks.so` + `ccall` | Zero (pre-compiled) | Same as `ccall` |

The JLCS MLIR dialect models the C/C++ ABI contract (struct layout, field offsets, vtable dispatch, RAII scopes, exception landing pads) as first-class IR operations, which are lowered through MLIR's standard pipeline to native machine code. The lowering performs full **x86-64 SysV struct classification**: small aligned structs (≤16 bytes) travel in registers with one scalar per eightbyte, exactly as clang would emit; larger or misaligned structs use sret/pointer conventions. See [MLIR / JLCS Dialect](@ref "MLIR & JLCS Dialect") for the full dialect specification.

### Tier 3 — `ccall`

Direct `ccall` into the shared library with zero setup. This is the unconditional fallback that always works, and — with Tier 1 parked — the production path for POD-safe functions.

### Tier selection flow

```
Function signature
        │
        ▼
is_ccall_safe()?
        │
        ├── yes ─→  enable_lto && LTO bitcode available && is_c_lto_safe()?
        │             ├── yes ─→  Tier 1: Base.llvmcall (cross-language inlining)
        │             └── no  ─→  Tier 3: ccall
        │
        └── no  ─→  aot_thunks = true?
                      ├── yes ─→  Tier 2 (AOT): ccall into <libname>_thunks.so
                      └── no  ─→  Tier 2 (JIT): JITManager.invoke()
```

The decision function `is_ccall_safe()` in `src/Wrapper/DispatchLogic.jl` inspects each function's DWARF metadata to determine ABI safety. It checks for STL container types, struct return sizes, packed struct layout mismatches (DWARF size vs Julia aligned size), union parameters, non-POD class types, and per-function `noexcept` to route may-throw functions through `jlcs.try_call`. For struct-graph cases where pairwise heuristics miss transitive layout mismatches, `src/IRGen/DAGDiff.jl` performs a structural type-graph diff to surface the bad cases and produce a topo-sorted lowering order for multi-type thunks.

## DWARF extraction

**Modules:** `src/Builder/DWARFParser.jl` (vtable/class extraction), `src/Builder/Compiler.jl` (metadata extraction at build time)

DWARF debug metadata is the single source of truth for all type information. RepliBuild compiles with `-g` and reads the debug metadata that the compiler itself emitted — the binary and its DWARF are produced by the same code path that decided the layout, so they cannot disagree.

### Why DWARF is the load-bearing input

**Offsets are observed values.** Every `DW_TAG_member` carries a `DW_AT_data_member_location` attribute holding the byte offset of that field within its enclosing struct, exactly as the compiled code uses it. The wrapper generator reads this offset directly. There is no re-derivation of alignment rules, no platform-specific padding computation, no `__attribute__` interpretation — the value in the metadata is the value the binary accesses. The same applies to `DW_AT_byte_size` for total struct size, bitfield position attributes, and subrange sizes for arrays.

**Stability across platforms and compiler flags.** The DWARF is produced by the same toolchain invocation that produced the `.so`. When the target changes — different platform, optimization level, `__attribute__` set, `#pragma pack` boundary, C++ ABI version — both the binary and its DWARF change together. The wrapper generated against a given binary is correct for that binary specifically.

**Compiler-version resilience.** Wrappers survive compiler upgrades. If a Clang revision changes how it lays out a tail-padding edge case or vtable thunk arrangement, the new DWARF reflects the new layout, and a regenerated wrapper picks it up. RepliBuild does not encode layout rules; it encodes a procedure for reading them.

**Coverage of constructs not in source.** DWARF carries information that headers cannot express directly: vtable slot indices via `DW_AT_vtable_elem_location`, base-class subobject offsets via `DW_TAG_inheritance` (a constant for non-virtual bases; a DWARF *expression* encoding a vtable-relative read for virtual bases), the compiler-chosen integral underlying type of an enum, and the bit positions of bitfields packed inside their storage units.

### Extraction flow

```
DWARF dump of binary.so
   │
   ├── DW_TAG_class_type / DW_TAG_structure_type
   │      ├── DW_AT_name, DW_AT_byte_size
   │      ├── DW_TAG_member            → MemberInfo (name, type,
   │      │                              DW_AT_data_member_location,
   │      │                              bitfield position attributes)
   │      ├── DW_TAG_subprogram        → VirtualMethod (name, mangled,
   │      │   [DW_AT_virtuality]         slot from DW_AT_vtable_elem_location)
   │      └── DW_TAG_inheritance       → base_classes with subobject offsets
   │                                     (DW_AT_data_member_location) and the
   │                                     virtual-base flag (DW_AT_virtuality);
   │                                     virtual bases carry a vtable-relative
   │                                     offset expression, parsed into
   │                                     vbase_vtable_offset
   │
   ├── DW_TAG_enumeration_type         → Enum definitions (chosen underlying type)
   ├── DW_TAG_union_type               → Union layout (DW_AT_byte_size)
   ├── DW_TAG_variable                 → Global variables (with DW_AT_location)
   ├── DW_TAG_typedef                  → Type aliases (resolved through DW_AT_type chains)
   └── DW_TAG_subprogram (free fn)     → Function signatures
              ├── DW_TAG_formal_parameter (in order) → Parameter types
              └── DW_AT_type                         → Return type
```

The parsers walk the DIE (Debug Information Entry) tree, resolve type references across compilation units, and fold typedef chains. **Parent attribution is depth-indexed**: a member/enumerator DIE at tree depth *d* attributes to the type last seen at depth *d−1*, so nested type definitions emitted between members (a routine clang behavior) cannot steal subsequent members from their enclosing class. Anonymous structs and unions are tracked through their parent context.

### Cross-verification against the symbol table

`nm` provides the authoritative linking identity. For each DWARF subprogram entry with a `DW_AT_linkage_name`, the parser confirms the symbol exists in the binary's symbol table:

- **Both present** — DWARF wins for type information; `nm` provides the canonical address and any visibility/weak attributes.
- **DWARF only** — Function was declared in debug metadata but elided by the linker (typically inlined or dead-code-eliminated). The wrapper is skipped.
- **`nm` only** — Symbol exists in the binary but has no DWARF (often library-internal helpers compiled without `-g`). Surfaced for manual wrapping if needed but not auto-bound.

### Data structures

| Type | Key fields | Role |
|------|-----------|------|
| `ClassInfo` | `name`, `vtable_ptr_offset`, `base_classes`, `base_offsets`, `virtual_bases`, `virtual_methods`, `members`, `size` | Complete class/struct description, including the inheritance chain with subobject offsets and virtual-base flags |
| `VtableInfo` | `classes`, `vtable_addresses`, `method_addresses` | All class metadata for a binary, indexed for fast lookup during IR generation |
| `VirtualMethod` | `name`, `mangled_name`, `slot`, `return_type`, `parameters` | Single virtual method descriptor; `slot` is the index into the declaring class's primary vtable |
| `MemberInfo` | `name`, `type_name`, `offset`, bitfield position | Struct field with byte offset (bitfield fields carry sub-byte position) |

## Caching strategy

RepliBuild uses four independent caching layers to minimize rebuild time:

### 1. Per-file IR cache (mtime + compile fingerprint)

Each source file's compiled LLVM IR is cached in `.replibuild_cache/`. Recompilation triggers on a changed `mtime` **or** a changed compile fingerprint (flags, defines, include dirs, LLVM version, target triple — stored in a `.ll.key` sidecar). Config changes can never silently reuse stale IR.

### 2. Project-level content hash (SHA-256)

A hash of `replibuild.toml` + all source contents + all header contents + git HEAD. If the hash matches the cached artifacts, `build()` returns in sub-second time without invoking any compiler.

### 3. Global registry cache (generator-fingerprinted)

`~/.replibuild/builds/<hash>/` stores full build artifacts, gating `use()`. The hash covers the TOML, sources, headers, and the project's git HEAD — **plus RepliBuild's own version and git HEAD** (the generator fingerprint), so upgrading RepliBuild automatically invalidates wrappers produced by older codegen instead of serving them forever. Cached wrappers resolve their `.so` sibling-first (`@__DIR__`), with the baked absolute path as fallback.

### 4. Toolchain cache

`~/.replibuild/toolchain.toml` caches the result of LLVM/Clang environment probing with a 24-hour TTL, avoiding repeated filesystem searches for the toolchain.

## Generated output layout

```
<project>/
├── replibuild.toml                    # Configuration (generated by discover(), hand-editable)
├── build/                             # LLVM IR files (.ll), intermediate objects
├── julia/
│   ├── <LibName>.so                   # Compiled shared library
│   ├── <LibName>_lto.bc               # LTO bitcode (if enable_lto = true)
│   ├── <LibName>_thunks.so            # AOT thunks (if aot_thunks = true)
│   ├── compilation_metadata.json      # Symbol + DWARF metadata
│   ├── thunk_manifest.json            # Tier-2 thunks the wrapper dispatches to
│   └── <ModuleName>.jl                # Generated Julia wrapper module
└── .replibuild_cache/                 # Incremental compile cache
```

The generated Julia module contains:

| Section | Contents |
|---------|----------|
| Module constants | `LIBRARY_PATH` (sibling-first resolution), `LTO_IR` (embedded bitcode), `THUNKS_LIBRARY_PATH` |
| Struct definitions | Correct field order, alignment padding, forward declarations for circular references, base-class member flattening with subobject-offset rebasing |
| Enum definitions | `@enum` with correct underlying types |
| Union representations | `NTuple{N,UInt8}` backing with typed getter/setter accessors |
| Function wrappers | Tier 1 (`llvmcall`), Tier 2 (`JITManager.invoke()` or AOT thunk `ccall`), Tier 3 (`ccall`), true-variadic overloads, global variable accessors |
| Cstring policy | `Union{String,Nothing}` returns, declared deallocators, raw `_ptr` variants |
| C++ class support | `Managed` handles with destructor finalizers, `as_<Base>` / `as_<VBase>` upcast helpers, method proxies |
| Bitfield accessors | Exact byte-span extraction and assignment |

## Performance characteristics

### Per-call overhead vs bare `ccall`

| Scenario | Tier | Median | vs bare `ccall` |
|----------|------|--------|-----------------|
| `scalar_add` | Pure Julia | 30 ns | 1.0x |
| `scalar_add` | Bare `ccall` | 30 ns | 1.0x (baseline) |
| `scalar_add` | Wrapper `ccall` | 40 ns | 1.33x |
| `scalar_add` | **LTO `llvmcall`** | **30 ns** | **1.0x** |
| `pack_record` (packed struct) | Bare `ccall` (unsafe) | crashes | --- |
| `pack_record` (packed struct) | Wrapper thunk (DWARF) | 80 ns | safe |

### Hot loop (1M iterations)

| Tier | ns/iter | Note |
|------|---------|------|
| Pure Julia | 0.677 | `@inbounds` native loop |
| Bare `ccall` | 1.800 | Hand-written FFI |
| Wrapper `ccall` | 2.026 | Generated (LTO disabled) |
| **LTO `llvmcall`** | **0.677** | **Julia JIT inlines C++ IR** |
| Whole loop in C++ | 0.997 | Single `ccall` to C++ loop |

The LTO path matches pure Julia performance because Julia's LLVM JIT sees the C++ bitcode and optimizes across the FFI boundary — the language boundary ceases to exist at the IR level. These measurements are from small benchmark modules; see the Tier 1 status note above for why whole-library builds currently keep LTO off.

## Key design decisions

**Source-based compilation.** RepliBuild compiles C/C++ source locally as part of the wrap pipeline. The build is part of the toolchain rather than an external artifact, which gives the wrapper generator access to DWARF metadata from the same compiler that produced the `.so`, enables LTO bitcode for Tier 1 inlining, and allows binaries to be tailored to the host machine. For **C** libraries with elaborate build systems that RepliBuild's source pipeline cannot reproduce, the experimental `ingest()` accepts an externally built debug binary and skips compilation — DWARF extraction and wrapper generation still run, with Tier 3 (`ccall`) dispatch only and best-effort extraction quality. C++ API surfaces are not supported in ingest mode (dialect thunks require the source build).

**DWARF as the source of truth.** Struct layout, inheritance hierarchies, bitfield positions, and virtual dispatch tables are always ABI-correct for the binary at hand because the metadata is produced by the same toolchain pass that decided the layout. Field offsets, sizes, and vtable slots are observed values, not derived from layout rules. See [DWARF extraction](#DWARF-extraction) for the specific properties this enables.

**Custom MLIR dialect over ad-hoc codegen.** The JLCS dialect provides a principled intermediate representation for C++ interop. Operations like `vcall`, `try_call`, and `scope` encode ABI semantics — vtable dispatch, exception landing pads, RAII destruction order, SysV struct classification — that would be error-prone to emit as raw LLVM IR. The MLIR framework handles verification, lowering, and JIT compilation through its standard pass infrastructure.

**Fail loud, degrade gracefully.** ABI cases that cannot be handled correctly refuse loudly (generation-time errors, call-time traps) rather than corrupting silently. A Tier 2 initialization failure — including the pre-flight rejection of untranslatable IR types — degrades the module to "Tier 2 disabled" with Tier 3 still functional, never a process crash. If `libJLCS.so` is not built, Tier 3 still works for all POD-safe functions. If the toolchain is incomplete, `check_environment()` reports exactly what is missing with OS-specific install instructions.

**Lock-free hot path.** The JIT manager uses a double-check caching pattern: after the first call to any JIT function, subsequent calls are a single unsynchronized `Dict` lookup — no locks, no allocation, no JIT overhead.

## Module map

For detailed documentation of each internal module, see [Internals](@ref "RepliBuild Internals").

### Core API

| Module | Source | Responsibility |
|--------|--------|----------------|
| `RepliBuild` | `src/RepliBuild.jl` | Top-level module. Exports `discover`, `build`, `wrap`, `use`, `search`, `ingest`, `check_environment`. |
| `Builder` | `src/Builder.jl` | Umbrella module for the build-pipeline subpackages (config, discovery, compile, link, deps, DWARF). |
| `ConfigurationManager` | `src/Builder/ConfigurationManager.jl` | Load, validate, merge `replibuild.toml` into a typed `RepliBuildConfig`. |
| `LLVMEnvironment` | `src/Builder/LLVMEnvironment.jl` | Detect system LLVM/Clang toolchain; fall back to JLL toolchains. |
| `EnvironmentDoctor` | `src/Builder/EnvironmentDoctor.jl` | `check_environment()` — validates the C bucket (JLL clang + Julia libLLVM) and the C++/Tier 2 bucket (system LLVM/MLIR 21+, CMake, mlir-tblgen, libJLCS.so). |

### Discovery and dependencies

| Module | Source | Responsibility |
|--------|--------|----------------|
| `Discovery` | `src/Builder/Discovery.jl` | Walk a project directory, resolve `#include` graph, emit `replibuild.toml`; preserve user-intent keys across forced re-discovery. |
| `DependencyResolver` | `src/Builder/DependencyResolver.jl` | Fetch git/local/system deps from `[dependencies]`, filter excludes, inject into compile graph. |
| `PackageRegistry` | `src/Builder/PackageRegistry.jl` | Local registry (`~/.replibuild/`) — `use()`, `register()`, `list_registry()`, `unregister()`; Hub fetch on registry miss; generator-fingerprinted build cache. |

### Compilation and linking

| Module | Source | Responsibility |
|--------|--------|----------------|
| `Compiler` | `src/Builder/Compiler.jl` | Per-file C/C++ to LLVM IR compilation. Fingerprinted incremental cache, parallel dispatch, template instantiation, macro shims + collision guard, IR sanitization, in-process C link/opt/assemble on Julia's libLLVM, DWARF metadata extraction. |
| `BuildBridge` | `src/Builder/BuildBridge.jl` | Shell out to `clang`, `llvm-link`, `opt`, `nm`. Low-level compiler driver (C++ pipeline and C fallback path). |
| `ThunkBuilder` | `src/Builder/ThunkBuilder.jl` | AOT thunk path. Drives JLCSIRGenerator → LLVM IR → `llc` → linked `<libname>_thunks.so` companion library used when `aot_thunks = true`. |

### DWARF extraction

| Module | Source | Responsibility |
|--------|--------|----------------|
| `DWARFParser` | `src/Builder/DWARFParser.jl` | Parse DWARF dumps. Extract `ClassInfo`, `VtableInfo`, `MemberInfo`, `VirtualMethod`, base-class subobject offsets, virtual-base vtable offsets. Handles unions, bitfields, globals, typedefs. |
| `ASTWalker` | `src/Builder/ASTWalker.jl` | Clang.jl-based AST walker for enum extraction. |
| `ClangJLBridge` | `src/Builder/ClangJLBridge.jl` | Clang.jl integration for header parsing. |

### Wrapper generation

| Module | Source | Responsibility |
|--------|--------|----------------|
| `Wrapper` | `src/Wrapper.jl` | Orchestrates Julia wrapper module generation. |
| `Wrapper.Generator` | `src/Wrapper/Generator.jl` | Top-level `wrap_library()` entry point; dispatches to the language generator. |
| `Wrapper.DispatchLogic` | `src/Wrapper/DispatchLogic.jl` | Per-function tier routing (`is_ccall_safe`, `is_c_lto_safe`). |
| `Wrapper.C.GeneratorC` | `src/Wrapper/C/GeneratorC.jl` | Full C wrapper generator (structs with exact-layout proof, enums, functions, varargs, Cstring policy). |
| `Wrapper.Cpp.GeneratorCpp` | `src/Wrapper/Cpp/GeneratorCpp.jl` | Full C++ wrapper generator (classes, inheritance, upcasts, Managed handles, virtual dispatch). |
| `Wrapper.Rust.GeneratorRust` | `src/Wrapper/Rust/GeneratorRust.jl` | Experimental Rust generator (`extern "C"` + `#[repr(C)]` surfaces). |
| `Wrapper.Cpp.STLWrappers` | `src/Wrapper/Cpp/STLWrappers.jl` | STL container type detection and accessor generation. |

### MLIR and JIT

| Module | Source | Responsibility |
|--------|--------|----------------|
| `IRGen` | `src/IRGen.jl` | Umbrella for IR generation and MLIR bindings. |
| `JLCSIRGenerator` | `src/IRGen/JLCSIRGenerator.jl` | Emit MLIR JLCS dialect IR from `VtableInfo` + metadata. Orchestrates the `ir_gen/` submodules; shared by both JIT (JITManager) and AOT (ThunkBuilder) paths. |
| `ir_gen/TypeUtils` | `src/IRGen/ir_gen/TypeUtils.jl` | C++ to MLIR type mapping (`double` to `f64`, `int*` to `!llvm.ptr`, etc.). |
| `ir_gen/StructGen` | `src/IRGen/ir_gen/StructGen.jl` | Struct type aliases and registration IR; packed-vs-aligned LLVM struct type strings; LLVM-literal inlining for packed structs nested inside other struct bodies. |
| `ir_gen/FunctionGen` | `src/IRGen/ir_gen/FunctionGen.jl` | Function thunks: `jlcs.try_call`/`ffe_call` calls, `jlcs.vcall` for virtual methods, `jlcs.marshal_arg`/`marshal_ret` for packed structs, `jlcs.scope` RAII temporaries for non-trivial by-value class parameters. |
| `ir_gen/ArrayViewGen` | `src/IRGen/ir_gen/ArrayViewGen.jl` | Zero-copy get/set thunks over fixed-size primitive array members via `jlcs.load/store_array_element`. |
| `ir_gen/STLContainerGen` | `src/IRGen/ir_gen/STLContainerGen.jl` | MLIR thunks for STL container accessors. |
| `DAGDiff` | `src/IRGen/DAGDiff.jl` | Structural type-graph diff. Detects transitive layout mismatches beyond pairwise heuristics and produces a topo-sorted lowering order for multi-type thunks. |
| `MLIRNative` | `src/IRGen/MLIRNative.jl` | Low-level `ccall` bindings to `libJLCS.so`: context management, module parsing, JIT engine creation, `lower_to_llvm`, symbol lookup, pending-exception buffer access. |
| `JITManager` | `src/IRGen/JITManager.jl` | Singleton `GLOBAL_JIT`. Thunk-manifest-driven initialization, lock-free symbol cache, arity-specialized `@generated` `invoke` (zero-allocation at any arity), `CxxException` propagation. |

### Introspection toolkit

Binary/DWARF analysis, Julia IR inspection, LLVM pass tooling, benchmarking, and
dataset export live in the companion package
[RepliBuildTooling.jl](https://github.com/obsidianjulua/RepliBuildTooling.jl) — an
opt-in extra that depends on the core (never the reverse), keeping heavy analysis
dependencies (DataFrames, CSV, …) out of the backend's precompile path.

### JLCS MLIR dialect (C++)

| File | Role |
|------|------|
| `src/mlir/JLCSDialect.td` | Dialect registration and namespace definition |
| `src/mlir/JLCSOps.td` | Operation definitions: `type_info`, `get_field`, `set_field`, `vcall`, `load_array_element`, `store_array_element`, `ffe_call`, `try_call`, `ctor_call`, `dtor_call`, `scope`, `yield`, `marshal_arg`, `marshal_ret` |
| `src/mlir/Types.td` | Type definitions: `!jlcs.c_struct<>`, `!jlcs.array_view<>` |
| `src/mlir/JLInterfaces.td` | Interface definitions |
| `src/mlir/CMakeLists.txt` | Build config: TableGen processing, whole-archive JIT linking |
| `src/mlir/build.sh` | Build script. Produces `src/mlir/build/libJLCS.so` |
| `src/mlir/impl/` | C++ implementation files for operation verifiers, lowering passes (including the SysV struct classifier), and the JIT entry points with the pre-flight type guard |
