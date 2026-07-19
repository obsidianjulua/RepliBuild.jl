# RepliBuild Internals

This section documents the internal modules that power RepliBuild. These are generally not needed for standard usage but are valuable for contributors, advanced integration, or understanding the system's behavior. For the high-level architecture see [Architecture](architecture.md).

## Wrapper

**Source:** `src/Wrapper.jl`, `src/Wrapper/`

The `Wrapper` package generates Julia FFI modules from DWARF metadata and binary symbol tables. It is structured as a two-track system: a C generator and a C++ generator, selected automatically via `config.wrap.language`.

### Module layout

| Module | Source | Role |
|--------|--------|------|
| `Wrapper.Generator` | `src/Wrapper/Generator.jl` | Top-level `wrap_library()` entry point; dispatches to C or C++ generator |
| `Wrapper.DispatchLogic` | `src/Wrapper/DispatchLogic.jl` | Per-function tier routing decisions (`is_ccall_safe`, `is_c_lto_safe`) |
| `Wrapper.TypeRegistry` | `src/Wrapper/TypeRegistry.jl` | `TypeRegistry` and `TypeStrictness` — shared type-resolution context |
| `Wrapper.Symbols` | `src/Wrapper/Symbols.jl` | `ParamInfo` / `SymbolInfo` structs for structured symbol data |
| `Wrapper.FunctionPointers` | `src/Wrapper/FunctionPointers.jl` | DWARF `function_ptr(...)` signature to Julia `@cfunction` type string |
| `Wrapper.Utils` | `src/Wrapper/Utils.jl` | Keyword escaping, identifier sanitization shared between generators |
| `Wrapper.C.GeneratorC` | `src/Wrapper/C/GeneratorC.jl` | Full C wrapper generator (structs, enums, functions, LTO, thunks) |
| `Wrapper.C.TypesC` | `src/Wrapper/C/TypesC.jl` | C type heuristics and base type map |
| `Wrapper.C.UtilsC` | `src/Wrapper/C/UtilsC.jl` | C-specific identifier/format helpers |
| `Wrapper.C.IdentifiersC` | `src/Wrapper/C/IdentifiersC.jl` | C name sanitization |
| `Wrapper.Cpp.GeneratorCpp` | `src/Wrapper/Cpp/GeneratorCpp.jl` | Full C++ wrapper generator (classes, inheritance flattening with subobject-offset rebasing, `as_<Base>`/`as_<VBase>` upcasts, `Managed` handles, virtual dispatch) |
| `Wrapper.Cpp.TypesCpp` | `src/Wrapper/Cpp/TypesCpp.jl` | C++ type map including STL, templates, references |
| `Wrapper.Cpp.IdentifiersCpp` | `src/Wrapper/Cpp/IdentifiersCpp.jl` | Namespace stripping, operator sanitization |
| `Wrapper.Cpp.UtilsCpp` | `src/Wrapper/Cpp/UtilsCpp.jl` | C++ formatting helpers |
| `Wrapper.Cpp.STLWrappers` | `src/Wrapper/Cpp/STLWrappers.jl` | STL container type detection and accessor generation |
| `Wrapper.Rust.GeneratorRust` | `src/Wrapper/Rust/GeneratorRust.jl` | Experimental Rust generator — requires `extern "C"` + `#[repr(C)]` surfaces |

### Language selection

`wrap.language` is an extensible dispatch key — `"c"` and `"cpp"` are the first two targets, with additional language generators planned:

```toml
[wrap]
language = "c"   # selects C generator + clang toolchain
language = "cpp" # selects C++ generator + clang++ toolchain (default)
```

`discover()` sets this automatically based on the scanned source files. Adding a new language means adding a generator under `src/Wrapper/<Lang>/` and registering it in `Wrapper/Generator.jl`.

### Tier selection logic

The function `is_ccall_safe()` in `src/Wrapper/DispatchLogic.jl` is the core dispatch decision. It inspects each function's DWARF metadata and returns `true` (Tier 1 / `ccall`) or `false` (Tier 2 / MLIR).

**Checks performed:**

1. **STL container types** — Any STL type in parameters or return forces Tier 2
2. **Return type safety:**
   - Template returns (contains `<`) → Tier 2 (unpredictable ABI)
   - Struct return by value > 16 bytes → Tier 2 (too large for `ccall` sret)
   - Non-POD class return → Tier 2
   - Packed struct return (DWARF size != Julia aligned size) → Tier 2
3. **Parameter type safety:**
   - Union parameters → Tier 2
   - Packed struct parameters → Tier 2
4. **Exception safety** — Per-function `is_noexcept` flag from DWARF. If absent (function may throw) and the module's `may_throw` setting is on, the function routes through `jlcs.try_call` rather than `ccall`.

For struct-graph cases where pairwise heuristics miss transitive layout mismatches (a non-packed struct that *contains* a packed struct, for example), `src/IRGen/DAGDiff.jl` performs a structural type-graph diff to surface bad cases and produces a topo-sorted lowering order for multi-type thunks.

Functions routed to Tier 2 are further divided between JIT dispatch (`JITManager.invoke()`) and AOT thunks (`ccall` to `_thunks.so`), controlled by the `aot_thunks` config flag.

### Idiomatic wrapper generation

Beyond raw `ccall` bindings, the wrapper generator clusters related C++ functions by class name to produce idiomatic Julia types:

1. **Factory detection:** Functions matching `create_X`, `new_X`, `make_X`, `alloc_X`, `init_X`, or returning `X*` are identified as constructors.
2. **Destructor detection:** Functions matching `delete_X`, `destroy_X`, `free_X`, `dealloc_X`, or `X_destroy` are identified as destructors.
3. **Method clustering:** Functions taking `X*` as their first parameter and associated with the same DWARF class are grouped as instance methods.

The result is a `mutable struct ManagedX` with a raw `Ptr{Cvoid}` handle, a registered `finalizer` calling the C++ destructor, and multiple-dispatch method proxies that pass the pointer via `Base.unsafe_convert`.

```@autodocs
Modules = [RepliBuild.Wrapper]
Order = [:function, :type]
Private = false
```

## Compiler

**Source:** `src/Builder/Compiler.jl`

The `Compiler` module handles the translation of C/C++ source code into LLVM IR and shared libraries. It oversees the entire build pipeline from dependency management down to IR optimization.

### Build pipeline

1. **Auto-discovery and dependency resolution:** Scans the project directory, resolving file paths and external git/local dependencies to merge into the build graph.
2. **Pre-processing (shims and templates):** Dynamically generates C/C++ shim files for configured macros and explicitly instantiates templates based on `replibuild.toml` settings. This allows normally invisible constructs to manifest in the final binary and DWARF metadata. Macro shims are pinned to default symbol visibility, and a **header-collision guard** verifies each direct shim `#include` resolves inside the project/dependency tree — the shim TU lives under the build cache, so a bare include could otherwise fall through the `-I` path to a system-installed header at a different version and silently bake wrong macro values.
3. **Compilation to LLVM IR:** Translates source code into `.ll` text format — `.c` via the JLL `clang`, `.cpp` via system `clang++`. The per-file cache is keyed on source `mtime` **plus a compile fingerprint** (flags, defines, include dirs, LLVM version, target triple); config changes can never silently reuse stale IR.
4. **IR transformation and sanitization:** Strips attributes incompatible with Julia's internal LLVM JIT, removes `va_start`/`va_end` intrinsics from varargs function bodies (varargs are routed through true-variadic `@ccall` wrapper generation), and cleans mismatched debug metadata.
5. **Link / optimize / assemble:** For **C**, these steps run **in-process on Julia's resident libLLVM** — `LLVM.link!` for linking, the new pass manager (`default<O…>`) for optimization, and in-process bitcode assembly — version-matched to the JLL clang that emitted the IR. A failure is a hard error; `[link] fallback = true` selects the external `llvm-link`/`opt` pipeline instead. **C++ always uses the external pipeline.**
6. **Codegen:** The final `.ll → .so` step shells to clang/clang++.

### Metadata extraction

At build time the compiler also extracts DWARF metadata into `compilation_metadata.json` (functions, struct definitions, enums, globals). DIE parsing is **depth-aware**: readelf DIE headers carry the tree depth, and member/enumerator/inheritance/template DIEs at depth *d* attribute to the type last seen at depth *d−1* — so nested type definitions interleaved between members (routine clang output) cannot steal subsequent members from the enclosing class. The recorded emitter version is the compiler that actually produced the IR, not whichever `llvm-config` happens to be on the PATH.

```@autodocs
Modules = [RepliBuild.Compiler]
Order = [:function, :type]
Private = false
```

## Configuration Manager

**Source:** `src/Builder/ConfigurationManager.jl`

The single source of truth for all build settings. Handles TOML parsing, validation, and merging into a typed `RepliBuildConfig` struct.

```@autodocs
Modules = [RepliBuild.ConfigurationManager]
Order = [:function, :type]
Private = false
```

## Discovery

**Source:** `src/Builder/Discovery.jl`

Scans the filesystem to identify C/C++ source files, headers, and dependencies. Auto-detects project language (`:c` vs `:cpp`) from the scanned source extensions and sets `wrap.language` accordingly in the generated `replibuild.toml`.

```@autodocs
Modules = [RepliBuild.Discovery]
Order = [:function, :type]
Private = false
```

## DWARFParser

**Source:** `src/Builder/DWARFParser.jl`

Parses `llvm-dwarfdump` output to extract structured type information from compiled binaries. This is the bridge between C++ debug metadata and Julia wrapper generation.

### Data structures

| Type | Fields | Role |
|------|--------|------|
| `ClassInfo` | `name`, `vtable_ptr_offset`, `base_classes`, `base_offsets`, `virtual_bases`, `virtual_methods`, `members`, `size` | Complete class/struct description with byte-level layout, inheritance chain (subobject offsets), and virtual-base flags |
| `VtableInfo` | `classes`, `vtable_addresses`, `method_addresses` | Aggregate metadata for all classes in a binary |
| `VirtualMethod` | `name`, `mangled_name`, `slot`, `return_type`, `parameters` | Single virtual method with the slot index in its declaring class's primary vtable |
| `MemberInfo` | `name`, `type_name`, `offset` | Struct field with byte offset from struct base |

### Extraction targets

| DWARF Tag | Extracted Data |
|-----------|----------------|
| `DW_TAG_class_type` / `DW_TAG_structure_type` | Class/struct name, byte size, members, virtual methods, inheritance |
| `DW_TAG_member` | Field name, type, `DW_AT_data_member_location` (byte offset) |
| `DW_TAG_subprogram` (with virtual flag) | Virtual method name, mangled name, vtable slot (`DW_AT_vtable_elem_location`) |
| `DW_TAG_inheritance` | Base class with subobject offset; for virtual bases, the vtable-relative offset expression parsed into `vbase_vtable_offset` |
| `DW_TAG_enumeration_type` | Enum definitions |
| `DW_TAG_union_type` | Union layout |
| `DW_TAG_variable` | Global variables |
| `DW_TAG_typedef` | Type aliases |

```@autodocs
Modules = [RepliBuild.DWARFParser]
Order = [:function, :type]
Private = false
```

## JLCSIRGenerator

**Source:** `src/IRGen/JLCSIRGenerator.jl`, `src/IRGen/ir_gen/`

Transforms parsed DWARF metadata (`VtableInfo`) into MLIR source text in the JLCS dialect. The generated IR is then parsed and either JIT-compiled by `MLIRNative` (Tier 2 JIT) or written to disk and AOT-compiled by `ThunkBuilder` (Tier 2 AOT). Both paths share this module — there is no separate AOT IR generator.

### Submodules

| Module | Source | Input | Output |
|--------|--------|-------|--------|
| `TypeUtils` | `src/IRGen/ir_gen/TypeUtils.jl` | C++ type string | MLIR type string (`f64`, `i32`, `!llvm.ptr`, etc.) |
| `StructGen` | `src/IRGen/ir_gen/StructGen.jl` | struct metadata | Struct type aliases + registration IR; aligned-vs-packed LLVM struct type strings; packed structs nested by value in other struct bodies are inlined as byte-identical LLVM literals |
| `FunctionGen` | `src/IRGen/ir_gen/FunctionGen.jl` | function or virtual method metadata | external `func.func private @mangled` decl + public `func.func @mangled_thunk` wrapper with `llvm.emit_c_interface`; scope-RAII temporaries for non-trivial by-value class params |
| `ArrayViewGen` | `src/IRGen/ir_gen/ArrayViewGen.jl` | fixed-size primitive array members | Zero-copy get/set thunks through `jlcs.load/store_array_element` |
| `STLContainerGen` | `src/IRGen/ir_gen/STLContainerGen.jl` | STL method metadata | Accessor thunks for `size()`, `data()`, etc. |

### Generation flow

`generate_jlcs_ir(vtinfo, metadata; needed_symbols)` produces a complete MLIR module:

1. **Struct aliases + registration:** type aliases for all extracted structs (packed structs as `!jlcs.c_struct`, padded structs as `!llvm.struct` with packed members inlined as LLVM literals)
2. **Type info operations:** `jlcs.type_info` for each class with non-empty members, carrying the DWARF-resolved destructor and the base/virtual-base tables
3. **Function thunks:** `func.func @mangled_thunk` wrappers carrying `llvm.emit_c_interface` — filtered by `needed_symbols` (the wrapper's thunk manifest, i.e. dead-thunk elimination). Each body unpacks `%args_ptr` (ciface convention), emits `jlcs.marshal_arg` for packed-struct parameters, `jlcs.scope` copy-construct/destruct brackets for non-trivial by-value class parameters, `jlcs.ffe_call` / `jlcs.try_call` (per-function noexcept routing) or `jlcs.vcall` (virtual instance methods with scalar/pointer signatures), and `jlcs.marshal_ret` for packed-struct returns
4. **STL container thunks:** Accessor thunks for detected STL containers (size, data, push_back, etc.)
5. **Array-view thunks:** rank-1 strided accessors for fixed-size primitive array members

## DAGDiff

**Source:** `src/IRGen/DAGDiff.jl`

Structural type-graph diff used by tier selection and IR generation when a struct may contain other structs whose layouts disagree between Julia and C++. The pairwise check in `is_ccall_safe()` catches direct packed-vs-aligned mismatches; `DAGDiff` catches the transitive cases — a non-packed struct that contains a packed struct as a field, a struct chain through a typedef alias, etc. It outputs a topo-sorted lowering order so that the MLIR thunks for dependent types are emitted in the right sequence.

## ThunkBuilder

**Source:** `src/Builder/ThunkBuilder.jl`

AOT compilation path for Tier 2 thunks. When `aot_thunks = true` in `replibuild.toml`, this module drives the same `JLCSIRGenerator.generate_jlcs_ir()` used by the JIT path, lowers the result to LLVM IR via `MLIRNative.lower_to_llvm()`, writes the LLVM IR to disk, runs `llc` to produce an object file, and links the object file with the user's compiled library into a companion shared library named `<libname>_thunks.so`.

The Julia wrapper then `ccall`s into the AOT thunks rather than calling `JITManager.invoke`. There is no MLIR JIT at runtime — `libJLCS.so` is only needed at build time for the lowering step. After AOT compilation, the user can ship the wrapped library + thunks `.so` without bundling LLVM/MLIR runtime libraries.

## MLIRNative

**Source:** `src/IRGen/MLIRNative.jl`

Low-level `ccall` bindings to `libJLCS.so`, the compiled JLCS MLIR dialect shared library. Provides context management, module parsing, JIT engine creation, LLVM lowering, and symbol lookup.

See the [MLIR / JLCS Dialect](@ref "MLIR & JLCS Dialect") page for the full API reference.

## JITManager

**Source:** `src/IRGen/JITManager.jl`

Singleton runtime (`GLOBAL_JIT`) for Tier 2 function dispatch. Manages the MLIR context, JIT execution engine, and compiled symbol cache.

### Key design points

- **Manifest-driven initialization:** `initialize_global_jit()` reads `thunk_manifest.json` — the thunks the wrapper actually dispatches to — so dead thunks are never generated. Any initialization failure (including the pre-flight rejection of untranslatable IR types in `libJLCS`) degrades the module to "Tier 2 disabled" with `ccall` wrappers intact, never a process crash.
- **Lock-free hot path:** `_lookup_cached()` reads from an `@atomic` snapshot of the symbol dictionary with no locking. The cache is published copy-on-write — a fresh dict is built with the new entry and atomically swapped in. Readers always see a stable, immutable snapshot.
- **Arity specialization:** `invoke` is `@generated`, emitting arity-specialized code for any argument count — stack-allocated `Ref`s and a fixed-size `Ptr{Cvoid}[]`, allocation-free at every arity. `String` arguments marshal as pointers to their bytes with the `String` GC-preserved across the call.
- **`@generated` return dispatch:** `_invoke_call` resolves at compile time whether the return type is a primitive (direct `ccall` return) or a struct (`sret` buffer allocation). An unresolved `Any` return fails loudly with the actual cause instead of corrupting memory.
- **Exception propagation:** After every Tier 2 call, `_check_pending_exception()` polls the thread-local exception buffer set by `jlcs.try_call` lowering. If a C++ exception was caught during the call, a `CxxException` is thrown with the original `what()` message.

### Calling convention

All Tier 2 functions use a unified `ciface` calling convention:

| Return | Signature |
|--------|-----------|
| Scalar | `T ciface(void** args_ptr)` |
| Struct | `void ciface(T* sret, void** args_ptr)` |
| Void | `void ciface(void** args_ptr)` |

## BuildBridge

**Source:** `src/Builder/BuildBridge.jl`

Low-level compiler driver that shells out to `clang`, `clang++`, `llvm-link`, `opt`, `llvm-as`, and `nm`. All subprocess invocations go through this module, providing a single point of control for toolchain interaction. It serves the C++ pipeline and the C bucket's `[link] fallback = true` escape hatch; the default C path links and optimizes in-process on Julia's libLLVM (see [Compiler](#Compiler)).

## LLVMEnvironment

**Source:** `src/Builder/LLVMEnvironment.jl`

Detects the system LLVM/Clang toolchain by searching standard paths and version-suffixed binaries. Falls back to `LLVM_full_jll` when no system toolchain is found. Caches results in `~/.replibuild/toolchain.toml` with a 24-hour TTL.

## EnvironmentDoctor

**Source:** `src/Builder/EnvironmentDoctor.jl`

`check_environment()` validates both toolchain buckets: the C bucket (JLL clang + Julia's resident libLLVM — no external install required) and the C++/Tier 2 bucket (system LLVM/MLIR 21+, Clang, `mlir-tblgen`, CMake 3.20+, and `libJLCS.so`). Returns a `ToolchainStatus` struct indicating which tiers are available, with OS-specific install instructions for missing components.

## DependencyResolver

**Source:** `src/Builder/DependencyResolver.jl`

Processes the `[dependencies]` table from `replibuild.toml`. Supports three dependency types:

| Type | Mechanism |
|------|-----------|
| `git` | Shallow clone (`--depth 1`) into `.replibuild_cache/deps/<name>/`; re-fetches on tag change |
| `local` | Scanned in-place; no copying |
| `system` | `pkg-config --cflags` to inject include paths |

The `exclude` list is applied after scanning. Resolved source files merge into the compilation graph before the compile step.

## PackageRegistry

**Source:** `src/Builder/PackageRegistry.jl`

Local package registry at `~/.replibuild/registry/`. Provides:

- `register()` — Store a project's build configuration
- `use()` — Build + wrap + load, with artifact caching in `~/.replibuild/builds/<hash>/`; on a local miss, fetches the package config from the RepliBuild-Hub community registry
- `search()` — Query the Hub index by name, description, tags, or language
- `list_registry()` — Print all registered packages with hash, source, and build status
- `unregister()` — Remove a package and clean cached builds

The build-cache key (`hash_config`) covers the TOML, sources, headers, and project git HEAD **plus the generator fingerprint** — RepliBuild's own version and git revision — so upgrading RepliBuild invalidates wrappers produced by older codegen. Cached wrappers resolve their `.so` sibling-first via `@__DIR__`, with the baked absolute path as fallback.

The `REPLIBUILD_HOME` environment variable overrides the default registry location; `REPLIBUILD_HUB_URL` points Hub operations at a private mirror.

## STLWrappers

**Source:** `src/Wrapper/Cpp/STLWrappers.jl`

Detects STL container types (`std::vector`, `std::string`, `std::map`, etc.) in DWARF metadata and generates accessor functions. These are used by the MLIR IR generator (`src/IRGen/ir_gen/STLContainerGen.jl`) to produce JIT thunks for STL container methods.

## ASTWalker

**Source:** `src/Builder/ASTWalker.jl`

Clang.jl-based AST walker for enum extraction. Handles `enum class`, hex values, namespaces, and other constructs that are difficult to extract reliably from DWARF alone. Replaces the earlier regex-based approach.

## ClangJLBridge

**Source:** `src/Builder/ClangJLBridge.jl`

Integration module for Clang.jl header parsing. Used by the wrapper generator when `use_clang_jl = true` to supplement DWARF metadata with AST-level information.

## Scaffold

**Source:** `src/Builder/PackageRegistry.jl` (`scaffold_package` function)

Generates a distributable Julia package from a registered RepliBuild project. The scaffolded package includes the compiled shared library, generated wrapper module, and a standard Julia `Project.toml` — ready for `Pkg.add()`.

## Introspect

**Source:** `src/Introspect.jl`, `src/Introspect/`

Umbrella module for binary analysis, Julia IR inspection, LLVM pass tooling, benchmarking, and data export. See [Introspection Tools](@ref) for the full API reference.

| Submodule | Source | Role |
|-----------|--------|------|
| `Binary` | `src/Introspect/Binary.jl` | `symbols()`, `dwarf_info()`, `dwarf_dump()`, `disassemble()`, `headers()` |
| `Julia` | `src/Introspect/Julia.jl` | `code_lowered()`, `code_typed()`, `code_llvm()`, `code_native()`, `code_warntype()`, analysis functions |
| `LLVM` | `src/Introspect/LLVM.jl` | `llvm_ir()`, `optimize_ir()`, `compare_optimization()`, `run_passes()`, `compile_to_asm()` |
| `Benchmarking` | `src/Introspect/Benchmarking.jl` | `benchmark()`, `benchmark_suite()`, `track_allocations()` |
| `DataExport` | `src/Introspect/DataExport.jl` | `export_json()`, `export_csv()`, `export_dataset()` |
| `Types` | `src/Introspect/Types.jl` | Shared type definitions for the introspection subsystem |
