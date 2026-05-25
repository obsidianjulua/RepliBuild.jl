# Architecture

RepliBuild compiles C/C++ source through an LLVM 21+ / MLIR pipeline, introspects DWARF debug metadata emitted by the compiler itself, and generates type-safe Julia bindings with correct struct layout, enum definitions, and calling conventions. Functions requiring non-trivial ABI handling are automatically routed through a custom MLIR dialect and JIT tier.

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

                                                       Tier 1 → ccall / llvmcall
                                                       Tier 3 → ccall fallback
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

## Pipeline stages

The lifecycle of a C/C++ project through RepliBuild proceeds in six stages. Each stage is implemented by one or more Julia modules that can be invoked independently.

### Stage 1 — Discovery

**Module:** `src/Builder/Discovery.jl`

Scans source files, parses the `#include` graph, resolves external dependencies, and emits a `replibuild.toml` configuration file. The language (`:c` or `:cpp`) is auto-detected from source file extensions and written to `wrap.language` in the generated config. New projects are automatically registered in the global registry (`~/.replibuild/registry/`).

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

Each source file is compiled to LLVM IR (`.ll` text format) via `clang` (`.c`) or `clang++` (`.cpp`). Key details:

- **Incremental cache:** Per-file mtime tracking in `.replibuild_cache/`. Only changed files are recompiled.
- **Project content hash:** SHA-256 of `replibuild.toml` + all source + all headers + git HEAD. If the hash matches cached artifacts, `build()` returns in sub-second time.
- **Parallel dispatch:** Enabled by default (`compile.parallel = true`).
- **Template instantiation:** If `[types].templates` is set, a stub `.cpp` is auto-generated to force Clang to emit DWARF for the requested template instantiations.
- **Macro shims:** If `[wrap.macros]` is set, typed C/C++ wrapper functions are generated so preprocessor macros appear in the compiled binary's debug metadata.
- **IR sanitization:** The compiler strips LLVM 19+ attributes incompatible with Julia's internal LLVM, removes `va_start`/`va_end` intrinsics from varargs function bodies (varargs are routed entirely through `ccall` wrapper generation), and cleans mismatched debug metadata.

### Stage 4 — Linking

**Module:** `src/Builder/Compiler.jl` (link phase), `src/Builder/BuildBridge.jl`, `src/Builder/ThunkBuilder.jl` (AOT thunks)

Sanitized per-file IR is merged via `llvm-link`, optimized by `llvm-opt`, and linked into the target shared library (`.so`/`.dylib`/`.dll`).

Optional artifacts:

| Artifact | Condition | Purpose |
|----------|-----------|---------|
| `<name>_lto.bc` | `enable_lto = true` | LLVM bitcode for `Base.llvmcall` embedding (Tier 1). Assembled via `Clang_unified_jll` to guarantee LLVM version match with Julia. |
| `<name>_thunks.so` | `aot_thunks = true` | Pre-compiled MLIR thunks for Tier 2 dispatch without JIT startup cost. |

### Stage 5 — Wrapping

**Module:** `src/Wrapper.jl`, `src/Wrapper/` subpackages, `src/Builder/DWARFParser.jl`, `src/Builder/ASTWalker.jl`

DWARF metadata is extracted from the compiled binary via `llvm-dwarfdump`. The parser builds structured types (`ClassInfo`, `VtableInfo`, `MemberInfo`, `VirtualMethod`) that feed into the wrapper generator.

The wrapper generator emits a complete, loadable Julia module containing:

- Struct definitions with correct field order, alignment padding (`_pad_N`), and topological sort for circular references
- `@enum` definitions with correct underlying types (extracted by the Clang.jl AST walker)
- Union representations as `NTuple{N,UInt8}` with typed getter/setter accessors
- Bitfield accessors with bit-shift extraction
- Function wrappers dispatched to the appropriate tier (see [Three-tier dispatch](@ref))
- Idiomatic `mutable struct` wrappers with GC-managed finalizers for factory/destructor pairs
- Global variable accessors via `cglobal`

Two independent generator tracks exist (`src/Wrapper/C/GeneratorC.jl` and `src/Wrapper/Cpp/GeneratorCpp.jl`), selected automatically by `wrap.language`.

### Stage 6 — JIT initialization (on demand)

**Modules:** `src/IRGen/JLCSIRGenerator.jl`, `src/IRGen/MLIRNative.jl`, `src/IRGen/JITManager.jl`

When the generated Julia module calls a Tier 2 function for the first time, the JIT subsystem:

1. Generates MLIR IR in the JLCS dialect from cached DWARF metadata
2. Parses the IR via `MLIRNative.parse_module()`
3. Lowers `jlcs` → `func` → `llvm` dialect → native LLVM IR → machine code
4. Caches the compiled symbol pointer in a lock-free dictionary

Subsequent calls to the same symbol are a single unsynchronized `Dict` read — no locks, no allocation, no JIT overhead.

If `aot_thunks = true`, thunks are pre-compiled at build time into `_thunks.so` and loaded via `ccall` at module parse time. No JIT initialization occurs at all.

## Three-tier dispatch

Every exported function is analyzed by the wrapper generator and routed to one of three calling tiers based on ABI complexity. Tier selection is fully automatic — no user annotation is required.

### Tier 1 — `ccall` and `llvmcall` (LTO)

For functions with simple, POD-safe signatures. Zero overhead.

**Conditions (all must hold):**
- No STL container types in parameters or return
- Return type is: primitive, pointer, void, or small aligned struct (16 bytes or fewer)
- All parameters are: primitive, pointer, or small struct — NOT unions, NOT packed structs, NOT non-POD classes

When `enable_lto = true`, eligible Tier 1 functions are upgraded to `Base.llvmcall`. The C/C++ LLVM bitcode is embedded as a module-level constant and passed directly to Julia's JIT compiler, enabling **cross-language inlining** — Julia can inline C++ code into hot loops, apply vectorization, and propagate through AD frameworks like Enzyme.jl.

```julia
# Generated wrapper with LTO path
function add(a::Cint, b::Cint)::Cint
    if !isempty(LTO_IR)
        return Base.llvmcall((LTO_IR, "_Z3addii"), Cint, Tuple{Cint, Cint}, a, b)
    else
        return ccall((:_Z3addii, LIBRARY_PATH), Cint, (Cint, Cint), a, b)
    end
end
```

**Additional LTO eligibility constraints** (beyond Tier 1):
- NOT a virtual method
- Does NOT return a struct by value
- No `Cstring` parameters or return (`llvmcall` does not auto-convert like `ccall`)

### Tier 2 — MLIR JIT or AOT thunks

For functions requiring complex ABI marshalling: packed structs, unions, large struct returns, C++ virtual dispatch.

| Mode | Config | Mechanism | Startup cost | Per-call cost |
|------|--------|-----------|--------------|---------------|
| **JIT** | `aot_thunks = false` | `JITManager.invoke()` at runtime | First-call JIT compilation | Lock-free after first call |
| **AOT** | `aot_thunks = true` | Pre-compiled `_thunks.so` + `ccall` | Zero (pre-compiled) | Same as `ccall` |

The JLCS MLIR dialect models the C ABI contract (struct layout, field offsets, vtable dispatch) as first-class IR operations, which are lowered through MLIR's standard pipeline to native machine code. See [MLIR / JLCS Dialect](@ref "MLIR & JLCS Dialect") for the full dialect specification.

### Tier 3 — `ccall` fallback

Direct `ccall` with zero setup. Used when LTO bitcode is unavailable (e.g., `enable_lto = false` builds or stripped binaries). This is the unconditional fallback that always works.

### Tier selection flow

```
Function signature
        │
        ▼
is_ccall_safe()?
        │
        ├── yes ─→  LTO bitcode available?
        │             ├── yes ─→  Tier 1: Base.llvmcall (cross-language inlining)
        │             └── no  ─→  Tier 3: ccall (POD fallback)
        │
        └── no  ─→  aot_thunks = true?
                      ├── yes ─→  Tier 2 (AOT): ccall into <libname>_thunks.so
                      └── no  ─→  Tier 2 (JIT): JITManager.invoke()
```

The decision function `is_ccall_safe()` in `src/Wrapper/DispatchLogic.jl` inspects each function's DWARF metadata to determine ABI safety. It checks for STL container types, struct return sizes, packed struct layout mismatches (DWARF size vs Julia aligned size), union parameters, non-POD class types, and per-function `noexcept` to route may-throw functions through `jlcs.try_call`. For struct-graph cases where pairwise heuristics miss transitive layout mismatches, `src/IRGen/DAGDiff.jl` performs a structural type-graph diff to surface the bad cases and produce a topo-sorted lowering order for multi-type thunks.

## DWARF extraction

**Module:** `src/Builder/DWARFParser.jl`

DWARF debug metadata is the single source of truth for all type information. RepliBuild compiles with `-g` and reads the debug metadata that the compiler itself emitted — the binary and its DWARF are produced by the same code path that decided the layout, so they cannot disagree.

### Why DWARF is the load-bearing input

Several properties of DWARF make it the right anchor for ABI-correct binding generation, rather than one input among many:

**Offsets are observed values.** Every `DW_TAG_member` carries a `DW_AT_data_member_location` attribute holding the byte offset of that field within its enclosing struct, exactly as the compiled code uses it. The wrapper generator reads this offset directly. There is no re-derivation of alignment rules, no platform-specific padding computation, no `__attribute__` interpretation — the value in the metadata is the value the binary accesses. The same applies to `DW_AT_byte_size` for total struct size, `DW_AT_bit_offset` and `DW_AT_bit_size` for bitfield positions, and `DW_AT_byte_size` on subrange types for arrays.

**Stability across platforms and compiler flags.** The DWARF is produced by the same toolchain invocation that produced the `.so`. When the target changes — different platform, optimization level, `__attribute__` set, `#pragma pack` boundary, C++ ABI version — both the binary and its DWARF change together. The wrapper generated against a given binary is correct for that binary specifically. There is no "the wrapper assumed natural alignment, but this build used `-fpack-struct`" failure mode, because the wrapper never assumed anything.

**Compiler-version resilience.** Wrappers survive compiler upgrades. If a Clang revision changes how it lays out a tail-padding edge case or vtable thunk arrangement, the new DWARF reflects the new layout, and a regenerated wrapper picks it up. RepliBuild does not encode layout rules; it encodes a procedure for reading them.

**Coverage of constructs not in source.** DWARF carries information that headers cannot express directly: vtable slot indices via `DW_AT_virtuality` on subprograms, base-class member offsets via `DW_TAG_inheritance`, the compiler-chosen integral underlying type of an enum, sret parameter positions for return-by-value structs above the platform's small-return threshold, and the bit positions of bitfields packed inside their storage units.

### Extraction flow

```
llvm-dwarfdump binary.so
   │
   ├── DW_TAG_class_type / DW_TAG_structure_type
   │      ├── DW_AT_name, DW_AT_byte_size
   │      ├── DW_TAG_member            → MemberInfo (name, type,
   │      │                              DW_AT_data_member_location,
   │      │                              DW_AT_bit_offset, DW_AT_bit_size)
   │      ├── DW_TAG_subprogram        → VirtualMethod (name, mangled,
   │      │   [DW_AT_virtuality]         slot from DW_AT_vtable_elem_location)
   │      └── DW_TAG_inheritance       → base_classes (with DW_AT_data_member_location)
   │
   ├── DW_TAG_enumeration_type         → Enum definitions (chosen underlying type)
   ├── DW_TAG_union_type               → Union layout (DW_AT_byte_size)
   ├── DW_TAG_variable                 → Global variables (with DW_AT_location)
   ├── DW_TAG_typedef                  → Type aliases (resolved through DW_AT_type chains)
   └── DW_TAG_subprogram (free fn)     → Function signatures
              ├── DW_TAG_formal_parameter (in order) → Parameter types
              └── DW_AT_type                         → Return type
```

The parser walks the DIE (Debug Information Entry) tree from `llvm-dwarfdump --debug-info`, resolves type references across compilation units, and folds typedef chains. Where DWARF references a type by offset, the parser maintains an offset → entry map so the reference is resolved to the concrete type. Anonymous structs and unions are tracked through their parent context.

### Cross-verification against the symbol table

`nm` provides the authoritative linking identity. For each DWARF subprogram entry with a `DW_AT_linkage_name`, the parser confirms the symbol exists in the binary's symbol table. Three outcomes are possible:

- **Both present** — DWARF wins for type information; `nm` provides the canonical address and any visibility/weak attributes.
- **DWARF only** — Function was declared in debug metadata but elided by the linker (typically because it was inlined or dead-code-eliminated). The wrapper is skipped.
- **`nm` only** — Symbol exists in the binary but has no DWARF (often library-internal helpers compiled without `-g`). These are surfaced for manual wrapping if needed but not auto-bound.

### Data structures

The parser produces structured Julia types that feed into both the wrapper generator and the MLIR IR generator:

| Type | Fields | Role |
|------|--------|------|
| `ClassInfo` | `name`, `vtable_ptr_offset`, `base_classes`, `virtual_methods`, `members`, `size` | Complete class/struct description, including inheritance chain and computed vtable layout |
| `VtableInfo` | `classes`, `vtable_addresses`, `method_addresses` | All class metadata for a binary, indexed for fast lookup during IR generation |
| `VirtualMethod` | `name`, `mangled_name`, `slot`, `return_type`, `parameters` | Single virtual method descriptor; `slot` is the index into the vtable for `jlcs.vcall` |
| `MemberInfo` | `name`, `type_name`, `offset`, `bit_offset`, `bit_size` | Struct field with byte offset (bitfield fields carry sub-byte position) |

## Caching strategy

RepliBuild uses four independent caching layers to minimize rebuild time:

### 1. Per-file IR cache (mtime-based)

Each source file's compiled LLVM IR is cached in `.replibuild_cache/` keyed by filepath. On recompilation, only files with changed `mtime` are recompiled.

### 2. Project-level content hash (SHA-256)

A hash of `replibuild.toml` + all source contents + all header contents + git HEAD. If the hash matches the cached artifacts, `build()` returns in sub-second time without invoking any compiler.

### 3. Global registry cache

`~/.replibuild/builds/<hash>/` stores full build artifacts. The `use()` function checks this cache first, enabling instant loads of previously built packages across projects.

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
│   └── <ModuleName>.jl                # Generated Julia wrapper module
└── .replibuild_cache/                 # Incremental compile cache
```

The generated Julia module contains:

| Section | Contents |
|---------|----------|
| Module constants | `LIBRARY_PATH`, `LTO_IR` (embedded bitcode), `THUNKS_LTO_IR` |
| Struct definitions | Correct field order, alignment padding, forward declarations for circular references, base class member flattening |
| Enum definitions | `@enum` with correct underlying types |
| Union representations | `NTuple{N,UInt8}` backing with typed getter/setter accessors |
| Function wrappers | Tier 1 (`ccall`/`llvmcall`), Tier 2 (`JITManager.invoke()` or AOT thunk `ccall`), variadic overloads, global variable accessors |
| Idiomatic wrappers | `mutable struct` types with GC finalizers, multiple-dispatch method proxies, `Base.unsafe_convert` for pointer passing |
| Bitfield accessors | Bit-shift extraction functions |

## Performance characteristics

### Per-call overhead vs bare `ccall`

| Scenario | Tier | Median | vs bare `ccall` |
|----------|------|--------|-----------------|
| `scalar_add` | Pure Julia | 30 ns | 1.0x |
| `scalar_add` | Bare `ccall` | 30 ns | 1.0x (baseline) |
| `scalar_add` | Wrapper `ccall` | 40 ns | 1.33x |
| `scalar_add` | **LTO `llvmcall`** | **30 ns** | **1.0x** |
| `pack_record` (packed struct) | Bare `ccall` (unsafe) | crashes | --- |
| `pack_record` (packed struct) | Wrapper `ccall` (DWARF) | 80 ns | safe |

### Hot loop (1M iterations)

| Tier | ns/iter | Note |
|------|---------|------|
| Pure Julia | 0.677 | `@inbounds` native loop |
| Bare `ccall` | 1.800 | Hand-written FFI |
| Wrapper `ccall` | 2.026 | Generated (LTO disabled) |
| **LTO `llvmcall`** | **0.677** | **Julia JIT inlines C++ IR** |
| Whole loop in C++ | 0.997 | Single `ccall` to C++ loop |

The LTO path matches pure Julia performance because Julia's LLVM JIT sees the C++ bitcode and optimizes across the FFI boundary — the language boundary ceases to exist at the IR level.

## Key design decisions

**Source-based compilation.** RepliBuild compiles C/C++ source locally as part of the wrap pipeline. The build is part of the toolchain rather than an external artifact, which gives the wrapper generator access to DWARF metadata from the same compiler that produced the `.so`, enables LTO bitcode for Tier 1 inlining, and allows binaries to be tailored to the host machine's compiler flags and target triple. The tradeoff is requiring a local LLVM 21+ toolchain. For libraries with elaborate build systems that RepliBuild's source pipeline cannot reproduce, `ingest()` accepts an externally built debug binary and skips compilation — DWARF extraction and wrapper generation still run, with Tier 3 (`ccall`) dispatch only.

**DWARF as the source of truth.** RepliBuild compiles with `-g` and reads the debug metadata that the compiler itself emitted. Struct layout, inheritance hierarchies, bitfield positions, and virtual dispatch tables are always ABI-correct for the binary at hand because the metadata is produced by the same toolchain pass that decided the layout. Field offsets, sizes, and vtable slots are observed values, not derived from layout rules — so platform-specific alignment, packing pragmas, attribute effects, and compiler-version edge cases all flow through automatically. See [DWARF extraction](@ref) for the specific properties this enables.

**Custom MLIR dialect over ad-hoc codegen.** The JLCS dialect provides a principled intermediate representation for C++ interop. Operations like `vcall` and `get_field` encode ABI semantics that would be error-prone to emit as raw LLVM IR directly. The MLIR framework handles lowering, optimization, and JIT compilation through its standard pass infrastructure.

**Lock-free hot path.** The JIT manager uses a double-check caching pattern: after the first call to any JIT function, subsequent calls are a single unsynchronized `Dict` lookup — no locks, no allocation, no JIT overhead. Julia's `Dict` is safe for concurrent reads under a single-writer pattern.

**Graceful degradation.** If `libJLCS.so` is not built, Tier 1 (`ccall`) still works for all POD-safe functions. If LTO is disabled, wrappers fall back to standard `ccall`. If the toolchain is incomplete, `check_environment()` reports exactly what is missing with OS-specific install instructions.

## Module map

For detailed documentation of each internal module, see [Internals](@ref "RepliBuild Internals").

### Core API

| Module | Source | Responsibility |
|--------|--------|----------------|
| `RepliBuild` | `src/RepliBuild.jl` | Top-level module. Exports `discover`, `build`, `wrap`, `use`, `ingest`, `check_environment`. |
| `Builder` | `src/Builder.jl` | Umbrella module for the build-pipeline subpackages (config, discovery, compile, link, deps, DWARF). |
| `ConfigurationManager` | `src/Builder/ConfigurationManager.jl` | Load, validate, merge `replibuild.toml` into a typed `RepliBuildConfig`. |
| `LLVMEnvironment` | `src/Builder/LLVMEnvironment.jl` | Detect system LLVM/Clang toolchain; fall back to `LLVM_full_jll`. |
| `EnvironmentDoctor` | `src/Builder/EnvironmentDoctor.jl` | `check_environment()` — validates LLVM 21+, Clang, mlir-tblgen, CMake, libJLCS.so. |

### Discovery and dependencies

| Module | Source | Responsibility |
|--------|--------|----------------|
| `Discovery` | `src/Builder/Discovery.jl` | Walk a project directory, resolve `#include` graph, emit `replibuild.toml`. |
| `DependencyResolver` | `src/Builder/DependencyResolver.jl` | Fetch git/local/system deps from `[dependencies]`, filter excludes, inject into compile graph. |
| `PackageRegistry` | `src/Builder/PackageRegistry.jl` | Global `~/.replibuild/registry/` — `use()`, `register()`, `list_registry()`, `unregister()`. |

### Compilation and linking

| Module | Source | Responsibility |
|--------|--------|----------------|
| `Compiler` | `src/Builder/Compiler.jl` | Per-file C/C++ to LLVM IR compilation. Incremental mtime cache, parallel dispatch, template instantiation, IR sanitization. |
| `BuildBridge` | `src/Builder/BuildBridge.jl` | Shell out to `clang`, `llvm-link`, `llvm-opt`, `nm`. Low-level compiler driver. |
| `ThunkBuilder` | `src/Builder/ThunkBuilder.jl` | AOT thunk path. Drives JLCSIRGenerator → LLVM IR → `llc` → linked `<libname>_thunks.so` companion library used when `aot_thunks = true`. |

### DWARF extraction

| Module | Source | Responsibility |
|--------|--------|----------------|
| `DWARFParser` | `src/Builder/DWARFParser.jl` | Parse `llvm-dwarfdump` output. Extract `ClassInfo`, `VtableInfo`, `MemberInfo`, `VirtualMethod`. Handles unions, bitfields, globals, typedefs. |
| `ASTWalker` | `src/Builder/ASTWalker.jl` | Clang.jl-based AST walker for enum extraction. |
| `ClangJLBridge` | `src/Builder/ClangJLBridge.jl` | Clang.jl integration for header parsing. |

### Wrapper generation

| Module | Source | Responsibility |
|--------|--------|----------------|
| `Wrapper` | `src/Wrapper.jl` | Orchestrates Julia wrapper module generation. |
| `Wrapper.Generator` | `src/Wrapper/Generator.jl` | Top-level `wrap_library()` entry point; dispatches to C or C++ generator. |
| `Wrapper.DispatchLogic` | `src/Wrapper/DispatchLogic.jl` | Per-function tier routing (`is_ccall_safe`, `is_c_lto_safe`). |
| `Wrapper.C.GeneratorC` | `src/Wrapper/C/GeneratorC.jl` | Full C wrapper generator (structs, enums, functions, LTO, thunks). |
| `Wrapper.Cpp.GeneratorCpp` | `src/Wrapper/Cpp/GeneratorCpp.jl` | Full C++ wrapper generator (same features + virtual dispatch). |
| `Wrapper.Cpp.STLWrappers` | `src/Wrapper/Cpp/STLWrappers.jl` | STL container type detection and accessor generation. |

### MLIR and JIT

| Module | Source | Responsibility |
|--------|--------|----------------|
| `IRGen` | `src/IRGen.jl` | Umbrella module for IR generation and MLIR bindings. |
| `JLCSIRGenerator` | `src/IRGen/JLCSIRGenerator.jl` | Emit MLIR JLCS dialect IR from `VtableInfo`. Orchestrates the `ir_gen/` submodules and is shared by both JIT (JITManager) and AOT (ThunkBuilder) paths. |
| `ir_gen/TypeUtils` | `src/IRGen/ir_gen/TypeUtils.jl` | C++ to MLIR type mapping (`double` to `f64`, `int*` to `!llvm.ptr`, etc.). |
| `ir_gen/StructGen` | `src/IRGen/ir_gen/StructGen.jl` | Generate `jlcs.type_info` operations from `ClassInfo`. Topological sort for inheritance. Packed-vs-aligned LLVM struct type strings for call signatures. |
| `ir_gen/FunctionGen` | `src/IRGen/ir_gen/FunctionGen.jl` | Generate `func.func` thunks for virtual methods and regular functions. Emits `jlcs.marshal_arg`/`marshal_ret` ops for packed structs, `jlcs.try_call` for may-throw functions, `jlcs.vcall` for virtual methods. |
| `ir_gen/STLContainerGen` | `src/IRGen/ir_gen/STLContainerGen.jl` | Generate MLIR thunks for STL container accessors. |
| `DAGDiff` | `src/IRGen/DAGDiff.jl` | Structural type-graph diff. Detects transitive layout mismatches beyond pairwise heuristics and produces a topo-sorted lowering order for multi-type thunks. |
| `MLIRNative` | `src/IRGen/MLIRNative.jl` | Low-level `ccall` bindings to `libJLCS.so`: context management, module parsing, JIT engine creation, `lower_to_llvm`, symbol lookup, pending-exception buffer access. |
| `JITManager` | `src/IRGen/JITManager.jl` | Singleton `GLOBAL_JIT`. Lock-free symbol cache (atomic snapshot + copy-on-write publishing), arity-specialized `invoke` (0–4 args) with `@generated` return-type dispatch, `CxxException` propagation. |

### Introspection toolkit

| Module | Source | Responsibility |
|--------|--------|----------------|
| `Introspect` | `src/Introspect.jl` | Umbrella module for binary analysis and diagnostics. |
| `Introspect.Binary` | `src/Introspect/Binary.jl` | `symbols()`, `dwarf_info()`, `disassemble()` — binary artifact analysis. |
| `Introspect.Julia` | `src/Introspect/Julia.jl` | Julia IR introspection (`code_lowered`, `code_typed`, `code_llvm`, `code_native`). |
| `Introspect.LLVM` | `src/Introspect/LLVM.jl` | LLVM pass tooling, IR optimization, pass comparison. |
| `Introspect.Benchmarking` | `src/Introspect/Benchmarking.jl` | `benchmark()` with configurable samples, suite execution. |
| `Introspect.DataExport` | `src/Introspect/DataExport.jl` | Export results to JSON and CSV. |

### JLCS MLIR dialect (C++)

| File | Role |
|------|------|
| `src/mlir/JLCSDialect.td` | Dialect registration and namespace definition |
| `src/mlir/JLCSOps.td` | Operation definitions: `type_info`, `get_field`, `set_field`, `vcall`, `load_array_element`, `store_array_element`, `ffe_call`, `try_call`, `ctor_call`, `dtor_call`, `scope`, `yield`, `marshal_arg`, `marshal_ret` |
| `src/mlir/Types.td` | Type definitions: `!jlcs.c_struct<>`, `!jlcs.array_view<>` |
| `src/mlir/JLInterfaces.td` | Interface definitions |
| `src/mlir/CMakeLists.txt` | Build config: TableGen processing, whole-archive JIT linking |
| `src/mlir/build.sh` | Build script. Produces `src/mlir/build/libJLCS.so` |
| `src/mlir/impl/` | C++ implementation files for dialect operations and lowering passes |
