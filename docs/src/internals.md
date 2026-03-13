# RepliBuild Internals

This section documents the internal modules that power RepliBuild. These are generally not needed for standard usage but are valuable for contributors, advanced integration, or understanding the system's behavior. For the high-level architecture see [Architecture](@ref).

## Wrapper

**Source:** `src/Wrapper.jl`, `src/Wrapper/`

The `Wrapper` package generates Julia FFI modules from DWARF metadata and binary symbol tables. It is structured as a two-track system: a C generator and a C++ generator, selected automatically via `config.wrap.language`.

### Module layout

| Module | Source | Role |
|--------|--------|------|
| `Wrapper.Generator` | `src/Wrapper/Generator.jl` | Top-level `wrap_library()` entry point; dispatches to C or C++ generator |
| `Wrapper.TypeRegistry` | `src/Wrapper/TypeRegistry.jl` | `TypeRegistry` and `TypeStrictness` — shared type-resolution context |
| `Wrapper.Symbols` | `src/Wrapper/Symbols.jl` | `ParamInfo` / `SymbolInfo` structs for structured symbol data |
| `Wrapper.FunctionPointers` | `src/Wrapper/FunctionPointers.jl` | DWARF `function_ptr(...)` signature to Julia `@cfunction` type string |
| `Wrapper.Utils` | `src/Wrapper/Utils.jl` | Keyword escaping, identifier sanitization shared between generators |
| `Wrapper.C.GeneratorC` | `src/Wrapper/C/GeneratorC.jl` | Full C wrapper generator (structs, enums, functions, LTO, thunks) |
| `Wrapper.C.TypesC` | `src/Wrapper/C/TypesC.jl` | C type heuristics and base type map |
| `Wrapper.Cpp.GeneratorCpp` | `src/Wrapper/Cpp/GeneratorCpp.jl` | Full C++ wrapper generator (same feature set + virtual dispatch) |
| `Wrapper.Cpp.TypesCpp` | `src/Wrapper/Cpp/TypesCpp.jl` | C++ type map including STL, templates, references |
| `Wrapper.Cpp.IdentifiersCpp` | `src/Wrapper/Cpp/IdentifiersCpp.jl` | Namespace stripping, operator sanitization |

### Language selection

`wrap.language` is an extensible dispatch key — `"c"` and `"cpp"` are the first two targets, with additional language generators planned:

```toml
[wrap]
language = "c"   # selects C generator + clang toolchain
language = "cpp" # selects C++ generator + clang++ toolchain (default)
```

`discover()` sets this automatically based on the scanned source files. Adding a new language means adding a generator under `src/Wrapper/<Lang>/` and registering it in `Wrapper/Generator.jl`.

### Tier selection logic

The function `is_ccall_safe()` in `src/Wrapper.jl` is the core dispatch decision. It inspects each function's DWARF metadata and returns `true` (Tier 1 / `ccall`) or `false` (Tier 2 / MLIR).

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

**Source:** `src/Compiler.jl`

The `Compiler` module handles the translation of C/C++ source code into LLVM IR and shared libraries. It oversees the entire build pipeline from dependency management down to IR optimization.

### Build pipeline

1. **Auto-discovery and dependency resolution:** Scans the project directory, resolving file paths and external git/local dependencies to merge into the build graph.
2. **Pre-processing (shims and templates):** Dynamically generates C/C++ shim files for configured macros and explicitly instantiates templates based on `replibuild.toml` settings. This allows normally invisible constructs to manifest in the final binary and DWARF metadata.
3. **Compilation to LLVM IR:** Translates source code into `.ll` text format via `clang`/`clang++`.
4. **IR transformation and sanitization:** Strips LLVM 19+ attributes incompatible with Julia's internal LLVM JIT, removes `va_start`/`va_end` intrinsics from varargs function bodies (varargs are routed entirely through `ccall` wrapper generation), and cleans mismatched debug metadata.
5. **Bitcode assembly:** The sanitized IR is converted into `.bc` binary format for zero-cost LTO loading. Uses `Clang_unified_jll` to guarantee LLVM version compatibility with Julia.
6. **Linking:** Object files are linked into the target shared library.

### Language-aware compilation

`.c` files are compiled with `clang`; `.cpp` files with `clang++`. For C projects, `create_library()` and `create_executable()` also use `clang` as the linker driver.

### Bitcode assembly

`Compiler.assemble_bitcode(ll_path, bc_path)` converts sanitized LLVM IR text (`.ll`) to binary bitcode (`.bc`). It prefers `Clang_unified_jll.clang -emit-llvm` so the resulting bitcode exactly matches the LLVM version bundled with Julia, maximizing `Base.llvmcall` compatibility. Falls back to system `llvm-as` if the JLL is unavailable.

This function is called by both the main LTO pipeline (`link_optimize_ir`) and the AOT thunks path (`_build_aot_thunks`).

```@autodocs
Modules = [RepliBuild.Compiler]
Order = [:function, :type]
Private = false
```

## Configuration Manager

**Source:** `src/ConfigurationManager.jl`

The single source of truth for all build settings. Handles TOML parsing, validation, and merging into a typed `RepliBuildConfig` struct.

```@autodocs
Modules = [RepliBuild.ConfigurationManager]
Order = [:function, :type]
Private = false
```

## Discovery

**Source:** `src/Discovery.jl`

Scans the filesystem to identify C/C++ source files, headers, and dependencies. Auto-detects project language (`:c` vs `:cpp`) from the scanned source extensions and sets `wrap.language` accordingly in the generated `replibuild.toml`.

```@autodocs
Modules = [RepliBuild.Discovery]
Order = [:function, :type]
Private = false
```

## DWARFParser

**Source:** `src/DWARFParser.jl`

Parses `llvm-dwarfdump` output to extract structured type information from compiled binaries. This is the bridge between C++ debug metadata and Julia wrapper generation.

### Data structures

| Type | Fields | Role |
|------|--------|------|
| `ClassInfo` | `name`, `vtable_ptr_offset`, `base_classes`, `virtual_methods`, `members`, `size` | Complete class/struct description with byte-level layout |
| `VtableInfo` | `classes`, `vtable_addresses`, `method_addresses` | Aggregate metadata for all classes in a binary |
| `VirtualMethod` | `name`, `mangled_name`, `slot`, `return_type`, `parameters` | Single virtual method with vtable slot index |
| `MemberInfo` | `name`, `type_name`, `offset` | Struct field with byte offset from struct base |

### Extraction targets

| DWARF Tag | Extracted Data |
|-----------|----------------|
| `DW_TAG_class_type` / `DW_TAG_structure_type` | Class/struct name, byte size, members, virtual methods, inheritance |
| `DW_TAG_member` | Field name, type, `DW_AT_data_member_location` (byte offset) |
| `DW_TAG_subprogram` (with virtual flag) | Virtual method name, mangled name, vtable slot |
| `DW_TAG_inheritance` | Base class references |
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

**Source:** `src/JLCSIRGenerator.jl`, `src/ir_gen/`

Transforms parsed DWARF metadata (`VtableInfo`) into MLIR source text in the JLCS dialect. The generated IR is then parsed and JIT-compiled by `MLIRNative`.

### Submodules

| Module | Source | Input | Output |
|--------|--------|-------|--------|
| `TypeUtils` | `src/ir_gen/TypeUtils.jl` | C++ type string | MLIR type string (`f64`, `i32`, `!llvm.ptr`, etc.) |
| `StructGen` | `src/ir_gen/StructGen.jl` | `ClassInfo` + members | `jlcs.type_info` operation with field types and offsets |
| `FunctionGen` | `src/ir_gen/FunctionGen.jl` | `VirtualMethod` | `func.func @thunk_...` wrapper function |
| `STLContainerGen` | `src/ir_gen/STLContainerGen.jl` | STL method metadata | Accessor thunks for `size()`, `data()`, etc. |

### Generation flow

`generate_jlcs_ir(vtinfo::VtableInfo)` produces a complete MLIR module:

1. **External dispatch declarations:** `llvm.func @mangled_name(...)` for each method with a resolved address
2. **Type info operations:** `jlcs.type_info` for each class with non-empty members (topological sort for inheritance)
3. **Virtual method thunks:** `func.func @thunk_...` for each virtual method
4. **Function thunks:** Thunks for regular functions from compilation metadata
5. **STL container thunks:** Accessor thunks for detected STL containers

## MLIRNative

**Source:** `src/MLIRNative.jl`

Low-level `ccall` bindings to `libJLCS.so`, the compiled JLCS MLIR dialect shared library. Provides context management, module parsing, JIT engine creation, LLVM lowering, and symbol lookup.

See the [MLIR / JLCS Dialect](@ref "MLIR & JLCS Dialect") page for the full API reference.

## JITManager

**Source:** `src/JITManager.jl`

Singleton runtime (`GLOBAL_JIT`) for Tier 2 function dispatch. Manages the MLIR context, JIT execution engine, and compiled symbol cache.

### Key design points

- **Lock-free hot path:** The `_lookup_cached()` function uses a double-check pattern — cached symbols are read from a `Dict` without locking. Only first-call misses acquire the lock.
- **Arity specialization:** Hand-specialized `invoke` methods for 0-4 arguments avoid heap allocation of `Any[]`. Stack-allocated `Ref`s and fixed-size `Ptr{Cvoid}[]` keep the hot path allocation-free.
- **`@generated` return dispatch:** `_invoke_call` uses `@generated` to resolve at compile time whether the return type is a primitive (direct `ccall` return) or a struct (`sret` buffer allocation).
- **Variadic fallback:** 5+ argument calls use dynamic allocation as a fallback.

### Calling convention

All Tier 2 functions use a unified `ciface` calling convention:

| Return | Signature |
|--------|-----------|
| Scalar | `T ciface(void** args_ptr)` |
| Struct | `void ciface(T* sret, void** args_ptr)` |
| Void | `void ciface(void** args_ptr)` |

## BuildBridge

**Source:** `src/BuildBridge.jl`

Low-level compiler driver that shells out to `clang`, `clang++`, `llvm-link`, `llvm-opt`, `llvm-as`, and `nm`. All subprocess invocations go through this module, providing a single point of control for toolchain interaction.

## LLVMEnvironment

**Source:** `src/LLVMEnvironment.jl`

Detects the system LLVM/Clang toolchain by searching standard paths and version-suffixed binaries. Falls back to `LLVM_full_jll` when no system toolchain is found. Caches results in `~/.replibuild/toolchain.toml` with a 24-hour TTL.

## EnvironmentDoctor

**Source:** `src/EnvironmentDoctor.jl`

`check_environment()` validates the complete toolchain: LLVM 21+, Clang, `mlir-tblgen`, CMake 3.20+, and `libJLCS.so`. Returns a `ToolchainStatus` struct indicating which tiers are available. Provides OS-specific install instructions for missing components.

## DependencyResolver

**Source:** `src/DependencyResolver.jl`

Processes the `[dependencies]` table from `replibuild.toml`. Supports three dependency types:

| Type | Mechanism |
|------|-----------|
| `git` | Shallow clone (`--depth 1`) into `.replibuild_cache/deps/<name>/`; re-fetches on tag change |
| `local` | Scanned in-place; no copying |
| `system` | `pkg-config --cflags` to inject include paths |

The `exclude` list is applied after scanning. Resolved source files merge into the compilation graph before the compile step.

## PackageRegistry

**Source:** `src/PackageRegistry.jl`

Global package registry at `~/.replibuild/registry/`. Provides:

- `register()` — Store a project's build configuration
- `use()` — Build + wrap + load, with artifact caching in `~/.replibuild/builds/<hash>/`
- `list_registry()` — Print all registered packages with hash, source, and build status
- `unregister()` — Remove a package and clean cached builds

The `REPLIBUILD_HOME` environment variable can override the default registry location.

## STLWrappers

**Source:** `src/STLWrappers.jl`

Detects STL container types (`std::vector`, `std::string`, `std::map`, etc.) in DWARF metadata and generates accessor functions. These are used by the MLIR IR generator (`ir_gen/STLContainerGen.jl`) to produce JIT thunks for STL container methods.

## ASTWalker

**Source:** `src/ASTWalker.jl`

Clang.jl-based AST walker for enum extraction. Handles `enum class`, hex values, namespaces, and other constructs that are difficult to extract reliably from DWARF alone. Replaces the earlier regex-based approach.

## ClangJLBridge

**Source:** `src/ClangJLBridge.jl`

Integration module for Clang.jl header parsing. Used by the wrapper generator when `use_clang_jl = true` to supplement DWARF metadata with AST-level information.

## Scaffold

**Source:** `src/Scaffold.jl`

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
