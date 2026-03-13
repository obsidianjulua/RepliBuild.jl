# Architecture

RepliBuild compiles C/C++ source through an LLVM 21+ / MLIR pipeline, introspects DWARF debug metadata emitted by the compiler itself, and generates type-safe Julia bindings with correct struct layout, enum definitions, and calling conventions. Functions requiring non-trivial ABI handling are automatically routed through a custom MLIR dialect and JIT tier.

This page documents the full system architecture. For the public API surface see [API Reference](@ref), and for per-module internals see [Internals](@ref "RepliBuild Internals").

## System overview

```
+------------------------------------------------------------------------+
|                      User API (3 core functions)                       |
|                                                                        |
|  discover("path/")         build("replibuild.toml")                    |
|  --- scan & configure ---  --- compile & link ---                      |
|                                                                        |
|                            wrap("replibuild.toml")                     |
|                            --- introspect & emit Julia module ---      |
+------------+--------------------------+--------------------------------+
             |                          |
             v                          v
+------------------------+  +--------------------------------------------+
|  Configuration Layer   |  |          Compiler Pipeline                  |
|                        |  |                                            |
|  Discovery.jl          |  |  Compiler.jl -> BuildBridge.jl -> Linker   |
|  ConfigurationManager  |  |  DependencyResolver.jl                     |
|  LLVMEnvironment.jl    |  |  LLVMEnvironment.jl                        |
|  EnvironmentDoctor.jl  |  |                                            |
+------------+-----------+  +---------------------+----------------------+
             |                                    |
             |    replibuild.toml                 |  .so + DWARF + .ll
             v                                    v
+------------------------------------------------------------------------+
|                     Binding Generation                                  |
|                                                                        |
|  DWARFParser.jl --> Wrapper.jl --> Generated Julia Module               |
|       |                |              |                                |
|       |           +----+----+    Tier 1: ccall / llvmcall              |
|       |           | Tier    |    Tier 2: JITManager.invoke()           |
|       |           | Select  |         or AOT thunk ccall               |
|       |           +----+----+                                          |
|       |                |                                               |
|       v                v                                               |
|  JLCSIRGenerator --> MLIRNative --> JITManager                         |
|  (ir_gen/ modules)   (libJLCS.so)  (lock-free cache)                  |
+------------------------------------------------------------------------+
```

## Pipeline stages

The lifecycle of a C/C++ project through RepliBuild proceeds in six stages. Each stage is implemented by one or more Julia modules that can be invoked independently.

### Stage 1 — Discovery

**Module:** `src/Discovery.jl`

Scans source files, parses the `#include` graph, resolves external dependencies, and emits a `replibuild.toml` configuration file. The language (`:c` or `:cpp`) is auto-detected from source file extensions and written to `wrap.language` in the generated config. New projects are automatically registered in the global registry (`~/.replibuild/registry/`).

### Stage 2 — Dependency resolution

**Module:** `src/DependencyResolver.jl`

Processes the `[dependencies]` table in `replibuild.toml`:

| Type | Mechanism |
|------|-----------|
| `git` | Shallow clone into `.replibuild_cache/deps/<name>/`; re-fetches when `tag` changes |
| `local` | Scanned in-place; no copy |
| `system` | `pkg-config --cflags` to inject include paths |

The `exclude` list is applied after scanning, allowing you to trim large repositories down to the files you need. Resolved source files merge into the compilation graph before stage 3.

### Stage 3 — Compilation

**Module:** `src/Compiler.jl`, `src/BuildBridge.jl`

Each source file is compiled to LLVM IR (`.ll` text format) via `clang` (`.c`) or `clang++` (`.cpp`). Key details:

- **Incremental cache:** Per-file mtime tracking in `.replibuild_cache/`. Only changed files are recompiled.
- **Project content hash:** SHA-256 of `replibuild.toml` + all source + all headers + git HEAD. If the hash matches cached artifacts, `build()` returns in sub-second time.
- **Parallel dispatch:** Enabled by default (`compile.parallel = true`).
- **Template instantiation:** If `[types].templates` is set, a stub `.cpp` is auto-generated to force Clang to emit DWARF for the requested template instantiations.
- **Macro shims:** If `[wrap.macros]` is set, typed C/C++ wrapper functions are generated so preprocessor macros appear in the compiled binary's debug metadata.
- **IR sanitization:** The compiler strips LLVM 19+ attributes incompatible with Julia's internal LLVM, removes `va_start`/`va_end` intrinsics from varargs function bodies (varargs are routed entirely through `ccall` wrapper generation), and cleans mismatched debug metadata.

### Stage 4 — Linking

**Module:** `src/Compiler.jl` (link phase), `src/BuildBridge.jl`

Sanitized per-file IR is merged via `llvm-link`, optimized by `llvm-opt`, and linked into the target shared library (`.so`/`.dylib`/`.dll`).

Optional artifacts:

| Artifact | Condition | Purpose |
|----------|-----------|---------|
| `<name>_lto.bc` | `enable_lto = true` | LLVM bitcode for `Base.llvmcall` embedding (Tier 1). Assembled via `Clang_unified_jll` to guarantee LLVM version match with Julia. |
| `<name>_thunks.so` | `aot_thunks = true` | Pre-compiled MLIR thunks for Tier 2 dispatch without JIT startup cost. |

### Stage 5 — Wrapping

**Module:** `src/Wrapper.jl`, `src/Wrapper/` subpackages, `src/DWARFParser.jl`, `src/ASTWalker.jl`

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

**Modules:** `src/JLCSIRGenerator.jl`, `src/MLIRNative.jl`, `src/JITManager.jl`

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
                  +--------------+
                  |  Function    |
                  |  Signature   |
                  +------+-------+
                         |
                  +------v-------+
                  | is_ccall_    |
                  | safe()?      |
                  +--+-------+---+
                  yes|       |no
                     |       |
            +--------v--+  +-v----------------+
            |  Tier 1   |  | aot_thunks?      |
            |  ccall    |  +--+------------+---+
            +-----+-----+  yes|            |no
                  |           |            |
            +-----v--+  +----v-----+ +----v-----------+
            | LTO?   |  | Tier 2   | | Tier 2         |
            +--+--+--+  | ccall    | | JITManager     |
            yes|  |no   | thunks   | | .invoke()      |
               |  |     | .so      | | (runtime JIT)  |
        +------v+ |     +----------+ +----------------+
        |llvm   | |
        |call   | |
        +-------+ |
            +------v--+
            | ccall   |
            | (std)   |
            +---------+
```

The decision function `is_ccall_safe()` in `src/Wrapper.jl` inspects each function's DWARF metadata to determine ABI safety. It checks for STL container types, struct return sizes, packed struct layout mismatches (DWARF size vs Julia aligned size), union parameters, and non-POD class types.

## DWARF extraction

**Module:** `src/DWARFParser.jl`

DWARF debug metadata is the single source of truth for all type information. Rather than parsing C/C++ headers (which miss ABI details like padding, vtable layout, and actual sizes), RepliBuild compiles with `-g` and reads the debug metadata that the compiler itself emitted.

### Extraction flow

```
llvm-dwarfdump binary.so
    |
    +-- DW_TAG_class_type / DW_TAG_structure_type
    |      +-- DW_AT_name, DW_AT_byte_size
    |      +-- DW_TAG_member -> MemberInfo (name, type, DW_AT_data_member_location)
    |      +-- DW_TAG_subprogram [virtual] -> VirtualMethod (name, mangled, slot)
    |      +-- DW_TAG_inheritance -> base_classes
    |
    +-- DW_TAG_enumeration_type -> Enum definitions
    +-- DW_TAG_union_type -> Union layout
    +-- DW_TAG_variable -> Global variables
    +-- DW_TAG_typedef -> Type aliases
```

### Data structures

The parser produces structured Julia types that feed into both the wrapper generator and the MLIR IR generator:

| Type | Fields | Role |
|------|--------|------|
| `ClassInfo` | `name`, `vtable_ptr_offset`, `base_classes`, `virtual_methods`, `members`, `size` | Complete class/struct description |
| `VtableInfo` | `classes`, `vtable_addresses`, `method_addresses` | All class metadata for a binary |
| `VirtualMethod` | `name`, `mangled_name`, `slot`, `return_type`, `parameters` | Single virtual method descriptor |
| `MemberInfo` | `name`, `type_name`, `offset` | Struct field with byte offset |

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
+-- replibuild.toml                    # Configuration (generated by discover(), hand-editable)
+-- build/                             # LLVM IR files (.ll), intermediate objects
+-- julia/
|   +-- <LibName>.so                   # Compiled shared library
|   +-- <LibName>_lto.bc               # LTO bitcode (if enable_lto = true)
|   +-- <LibName>_thunks.so            # AOT thunks (if aot_thunks = true)
|   +-- compilation_metadata.json      # Symbol + DWARF metadata
|   +-- <ModuleName>.jl                # Generated Julia wrapper module
+-- .replibuild_cache/                 # Incremental compile cache
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

**Source-based, not binary-based.** RepliBuild compiles C/C++ source locally rather than wrapping pre-compiled binaries (JLLs / BinaryBuilder). This gives it perfect DWARF metadata, enables LTO across the FFI boundary, and allows binaries tailored to the host machine. The tradeoff is requiring a local LLVM 21+ toolchain.

**DWARF as the source of truth.** Rather than parsing headers (which miss ABI details like padding, vtable layout, and actual sizes), RepliBuild compiles with `-g` and reads the debug metadata that the compiler itself emitted. Struct layout, inheritance hierarchies, and virtual dispatch tables are always ABI-correct.

**Custom MLIR dialect over ad-hoc codegen.** The JLCS dialect provides a principled intermediate representation for C++ interop. Operations like `vcall` and `get_field` encode ABI semantics that would be error-prone to emit as raw LLVM IR directly. The MLIR framework handles lowering, optimization, and JIT compilation through its standard pass infrastructure.

**Lock-free hot path.** The JIT manager uses a double-check caching pattern: after the first call to any JIT function, subsequent calls are a single unsynchronized `Dict` lookup — no locks, no allocation, no JIT overhead. Julia's `Dict` is safe for concurrent reads under a single-writer pattern.

**Graceful degradation.** If `libJLCS.so` is not built, Tier 1 (`ccall`) still works for all POD-safe functions. If LTO is disabled, wrappers fall back to standard `ccall`. If the toolchain is incomplete, `check_environment()` reports exactly what is missing with OS-specific install instructions.

## Module map

For detailed documentation of each internal module, see [Internals](@ref "RepliBuild Internals").

### Core API

| Module | Source | Responsibility |
|--------|--------|----------------|
| `RepliBuild` | `src/RepliBuild.jl` | Top-level module. Exports `discover`, `build`, `wrap`, `use`, `check_environment`. |
| `ConfigurationManager` | `src/ConfigurationManager.jl` | Load, validate, merge `replibuild.toml` into a typed `RepliBuildConfig`. |
| `LLVMEnvironment` | `src/LLVMEnvironment.jl` | Detect system LLVM/Clang toolchain; fall back to `LLVM_full_jll`. |
| `EnvironmentDoctor` | `src/EnvironmentDoctor.jl` | `check_environment()` — validates LLVM 21+, Clang, mlir-tblgen, CMake, libJLCS.so. |

### Discovery and dependencies

| Module | Source | Responsibility |
|--------|--------|----------------|
| `Discovery` | `src/Discovery.jl` | Walk a project directory, resolve `#include` graph, emit `replibuild.toml`. |
| `DependencyResolver` | `src/DependencyResolver.jl` | Fetch git/local/system deps from `[dependencies]`, filter excludes, inject into compile graph. |
| `PackageRegistry` | `src/PackageRegistry.jl` | Global `~/.replibuild/registry/` — `use()`, `register()`, `list_registry()`, `unregister()`. |

### Compilation and linking

| Module | Source | Responsibility |
|--------|--------|----------------|
| `Compiler` | `src/Compiler.jl` | Per-file C/C++ to LLVM IR compilation. Incremental mtime cache, parallel dispatch, template instantiation, IR sanitization. |
| `BuildBridge` | `src/BuildBridge.jl` | Shell out to `clang`, `llvm-link`, `llvm-opt`, `nm`. Low-level compiler driver. |

### DWARF extraction

| Module | Source | Responsibility |
|--------|--------|----------------|
| `DWARFParser` | `src/DWARFParser.jl` | Parse `llvm-dwarfdump` output. Extract `ClassInfo`, `VtableInfo`, `MemberInfo`, `VirtualMethod`. Handles unions, bitfields, globals, typedefs. |
| `ASTWalker` | `src/ASTWalker.jl` | Clang.jl-based AST walker for enum extraction. |
| `ClangJLBridge` | `src/ClangJLBridge.jl` | Clang.jl integration for header parsing. |

### Wrapper generation

| Module | Source | Responsibility |
|--------|--------|----------------|
| `Wrapper` | `src/Wrapper.jl` | Orchestrates Julia wrapper module generation. Contains `is_ccall_safe()` tier selection logic. |
| `Wrapper.Generator` | `src/Wrapper/Generator.jl` | Top-level `wrap_library()` entry point; dispatches to C or C++ generator. |
| `Wrapper.C.GeneratorC` | `src/Wrapper/C/GeneratorC.jl` | Full C wrapper generator (structs, enums, functions, LTO, thunks). |
| `Wrapper.Cpp.GeneratorCpp` | `src/Wrapper/Cpp/GeneratorCpp.jl` | Full C++ wrapper generator (same features + virtual dispatch). |
| `STLWrappers` | `src/STLWrappers.jl` | STL container type detection and accessor generation. |

### MLIR and JIT

| Module | Source | Responsibility |
|--------|--------|----------------|
| `JLCSIRGenerator` | `src/JLCSIRGenerator.jl` | Emit MLIR JLCS dialect IR from `VtableInfo`. Orchestrates `src/ir_gen/` submodules. |
| `ir_gen/TypeUtils` | `src/ir_gen/TypeUtils.jl` | C++ to MLIR type mapping (`double` to `f64`, `int*` to `!llvm.ptr`, etc.). |
| `ir_gen/StructGen` | `src/ir_gen/StructGen.jl` | Generate `jlcs.type_info` operations from `ClassInfo`. Topological sort for inheritance. |
| `ir_gen/FunctionGen` | `src/ir_gen/FunctionGen.jl` | Generate `func.func` thunks for virtual methods and regular functions. |
| `ir_gen/STLContainerGen` | `src/ir_gen/STLContainerGen.jl` | Generate MLIR thunks for STL container accessors. |
| `MLIRNative` | `src/MLIRNative.jl` | Low-level `ccall` bindings to `libJLCS.so`: context management, module parsing, JIT engine, `lower_to_llvm`, `lookup`. |
| `JITManager` | `src/JITManager.jl` | Singleton `GLOBAL_JIT`. Lock-free symbol cache, arity-specialized `invoke` (0-4 args), `@generated` ABI dispatch. |

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
| `src/mlir/JLCSOps.td` | Operation definitions: `type_info`, `get_field`, `set_field`, `vcall`, `load_array_element`, `store_array_element`, `ffe_call` |
| `src/mlir/Types.td` | Type definitions: `!jlcs.c_struct<>`, `!jlcs.array_view<>` |
| `src/mlir/JLInterfaces.td` | Interface definitions |
| `src/mlir/CMakeLists.txt` | Build config: TableGen processing, whole-archive JIT linking |
| `src/mlir/build.sh` | Build script. Produces `src/mlir/build/libJLCS.so` |
| `src/mlir/impl/` | C++ implementation files for dialect operations and lowering passes |
