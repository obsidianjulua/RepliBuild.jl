# Changelog

All notable changes to RepliBuild.jl are documented in this file.

## v2.5.6

### New: DAG Diff — Structural Mismatch Detection Between C++ and Julia IR

Added a DAG-based structural diff algorithm that compares C++ layouts (DWARF ground truth) against Julia's inferred alignment rules. This extends the existing per-function heuristics in `DispatchLogic.jl` — heuristics catch the obvious cases (packed returns, unions, STL), while DAGDiff catches what point-wise checks miss: transitive layout drift through by-value containment chains.

**Algorithm:**
1. Build C++ graph from DWARF metadata (struct sizes, member offsets, containment edges)
2. Build Julia graph by computing `min(sizeof(field), 8)` aligned layouts from the same members
3. Parallel walk — match nodes structurally, record size and per-member offset mismatches
4. Propagate mismatches transitively through by-value containment (if Inner is packed and Outer contains Inner by value, Outer is also mismatched)
5. Flag functions that pass or return mismatched types by value
6. Topo-sort (Kahn's algorithm) all thunk sites for safe lowering order — types before the functions that depend on them

**Integration:**
- `DAGDiff.needs_dag_thunk(symbol, result)` queries the mismatch map — wrapper generators check this alongside existing heuristics, routing to MLIR thunks if either fires
- Backward compatible: `needs_dag_thunk(_, nothing)` returns `false` when DAG diff is not computed
- Wired into both C and C++ generator dispatch sites in `GeneratorC.jl` and `GeneratorCpp.jl`

**Visualization:**
- `export_dot(result, path)` — Graphviz DOT export with mismatch color-coding (red = layout mismatch, orange = function needs thunk, gray = safe)
- `render_dot(result, path)` — renders DOT to SVG/PNG/PDF via the `dot` command
- Per-member offset annotations, containment edges, propagation edge coloring
- Three view modes: `:diff` (both graphs overlaid), `:cpp` (DWARF only), `:julia` (inferred alignment only)

**TOML configuration:**
```toml
[wrap]
dag = true   # exports DAG graphs to <project_root>/dag/
```

When enabled, the wrap stage automatically exports `diff.svg`, `cpp.svg`, `julia.svg`, and `diff.dot` to a `dag/` folder in the project root.

**Files:**
- `src/IRGen/DAGDiff.jl` — New module (~780 lines): graph types, builders, diff algorithm, topo-sort, query API, DOT visualization
- `src/Builder/ConfigurationManager.jl` — Added `dag::Bool` to `WrapConfig`
- `src/Wrapper/Generator.jl` — DAG diff computed before wrapper generation; graphs exported when `dag=true`
- `src/Wrapper/C/GeneratorC.jl`, `src/Wrapper/Cpp/GeneratorCpp.jl` — Dispatch sites augmented with `needs_dag_thunk` check
- `test/dag_test/` — 178 tests covering graph building, structural diff, transitive propagation, topo-sort, query API, DOT export, and a rendered gallery of 7 scenarios

**Stress test results (73 functions, `test/stress_test/`):**
- 25 mismatches detected: 14 types (vtable offsets on polymorphic classes, compound struct padding, bool alignment, STL internals), 5 functions routed to thunks (`compute_lu`, `compute_qr`, `compute_eigen`, `solve_ode_rk4`, `solve_ode_adaptive`)
- Transitive propagation working: `uniform_real_distribution<double>` flagged solely because it contains `param_type` by value

## v2.5.5

### Refactor: Module Hierarchy — Flat Source → Organized Subsystems

Replaced the flat `src/*.jl` layout (14 top-level files, ~12k lines) with a three-subsystem hierarchy. Each subsystem has a thin orchestration shim that controls include order — all implementation lives in subdirectories.

**Top-level shims:**
- **`Builder.jl`** — Build mechanics: config, environment, compile, link, DWARF, package registry
- **`IRGen.jl`** — MLIR/JIT: native bindings, IR generation, JIT execution
- **`Wrapper.jl`** — Julia binding generation: type mapping, dispatch routing, codegen
- **`Introspect.jl`** — Analysis tooling: binary, Julia code, LLVM IR, benchmarking

**Subsystem layout:**
```
src/
  RepliBuild.jl            ← module root, exports, public API delegation
  Builder.jl               ← shim: includes Builder/*.jl
  Builder/
    LLVMEnvironment.jl, ConfigurationManager.jl, BuildBridge.jl,
    DependencyResolver.jl, ASTWalker.jl, Discovery.jl, ClangJLBridge.jl,
    Compiler.jl, DWARFParser.jl, EnvironmentDoctor.jl, PackageRegistry.jl,
    ThunkBuilder.jl
  IRGen.jl                 ← shim: includes IRGen/*.jl
  IRGen/
    MLIRNative.jl, JLCSIRGenerator.jl, JITManager.jl
    ir_gen/  (FunctionGen.jl, StructGen.jl, STLContainerGen.jl, TypeUtils.jl)
  Wrapper.jl               ← shim: includes Wrapper/**/*.jl
  Wrapper/
    Utils.jl, TypeRegistry.jl, Symbols.jl, FunctionPointers.jl,
    DispatchLogic.jl, Generator.jl
    C/    (GeneratorC.jl, TypesC.jl, IdentifiersC.jl, UtilsC.jl)
    Cpp/  (GeneratorCpp.jl, TypesCpp.jl, IdentifiersCpp.jl, UtilsCpp.jl, STLWrappers.jl)
  Introspect.jl            ← shim: includes Introspect/*.jl
  Introspect/
    Types.jl, Binary.jl, Julia.jl, LLVM.jl, Benchmarking.jl,
    DataExport.jl, Project.jl
```

- **`RepliBuild.jl`** is now a pure delegation layer — loads the four subsystem shims, `using`s their modules, and re-exports the public API. No implementation logic remains at the top level.
- **`ThunkBuilder.jl`** extracted from `Compiler.jl` — bridges Builder and IRGen (needs `Wrapper.is_c_lto_safe`), loaded after Wrapper to satisfy the cross-subsystem dependency.
- **`PackageRegistry.jl`** moved from `Hub/` into `Builder/`.
- Stable path constants (`PROJECT_ROOT`, `SRC_DIR`) in `RepliBuild.jl` replace `@__DIR__` in submodules so file moves don't break paths.
- Net deletion: ~11,900 lines of duplicated top-level files removed.

### Fixed: Wrapper Generator Bug Audit (23 bugs)

Comprehensive audit and fix pass across C, C++, and shared wrapper subsystems. Full report in `BUG_AUDIT.md`.

**HIGH (8) — Crashes, memory corruption, silent wrong codegen:**
- **C-1/C-2/CPP-9**: sret llvmcall passed `Ref{T}` (GC addrspace 10) where `Ptr{T}` (raw addrspace 0) was needed — address space mismatch crashes. Fixed with `Base.unsafe_convert` to raw pointer before llvmcall. sret path now also applies integer widening and `Ref→Ptr` conversion matching the main llvmcall path.
- **C-3**: Use-after-free in bitfield/packed struct accessor — `pointer(collect(s._data))` created a GC-eligible temporary. Wrapped in `GC.@preserve`.
- **CPP-8**: C++ llvmcall missing `cconvert` for pointer params — ported the C generator's `Ptr` conversion logic.
- **CPP-10**: Debug `println` statements left in template codegen — removed.
- **U-1**: `_resolve_forward_ptr` flattened `Ptr{Ptr{T}}` to `Ptr{Cvoid}` — now only collapses bare unknown struct names, preserves nested pointer indirection.
- **D-1**: `is_ccall_safe` used uncleaned return type (with `const`) for DWARF lookup — changed to `cleaned_ret`.

**MEDIUM (11) — Incorrect output in edge cases, misrouting:**
- **C-4**: Convenience wrapper DWARF lookup used `"__struct__" * name` prefix that doesn't exist — switched to bare name lookup.
- **C-5/L-4**: `_sanitize_c_type_name` stripped spaces (`" " => ""`), destroying multi-word types like `"unsigned int"` → `"unsignedint"`. Changed to `" " => "_"`. Same fix applied to C++ side in `_sanitize_cpp_type_name`.
- **C-6/L-3**: `is_c_enum_like`/`is_enum_like` were identical to their `is_struct_like` counterparts — any uppercase type got dual-classified. Now return `false`; real enum detection uses DWARF `__enum__` keys via `_is_enum_type()`.
- **C-7**: `long double` mapped to `Float64` (8 bytes) but x86-64 ABI uses 16-byte slots — changed to `NTuple{2, UInt64}` and removed from `_CCALL_SAFE_PRIMITIVES` to force struct safety checks.
- **CPP-1/L-1**: `make_cpp_identifier`/`make_c_identifier` lowercased before keyword check — `"Begin"` incorrectly matched `"begin"`. Removed `lowercase` call (Julia keywords are case-sensitive).
- **CPP-2**: Operator replacement ordering hit single-char operators before compounds — `operator<<` became `op_lt<` instead of `op_lshift`. Reordered longest-match-first.
- **CPP-3/4/5**: STL type detection used `startswith` prefix matching — `std::string_view` false-matched `std::string`, `std::set_difference` matched `std::set`. Added `_stl_name_match()` with word-boundary awareness (requires `<` or ` ` after prefix). Also reordered `unordered_map`/`unordered_set` before `map`/`set` in size lookup.
- **CPP-6**: Safe wrapper used `ccall_args` (containing converted names like `a_c`) instead of original `param_names` — generated code referenced variables that didn't exist in scope.
- **CPP-7**: Template this-pointer checked sanitized name against `struct_types` but DWARF stores raw names (e.g. `"Box<double>"` not `"Box_double"`) — now checks both `bare_class` and `safe_class`.

**LOW (4) — Cosmetic, minor edge cases:**
- **L-2**: `_sanitize_c_type_name` could return empty string from all-special-character input — now returns `"_UnknownType"` fallback.
- **U-2**: `_parse_int_or_hex` missed uppercase `0X` prefix — added `|| startswith(s, "0X")` check.

### New: C++ Exception Catching via `jlcs.try_call`

Added `TryCallOp` (`jlcs.try_call`) to the JLCS MLIR dialect — a variant of `ffe_call` that emits LLVM `invoke` + landing pad to catch C++ exceptions at the ABI boundary.

- **`jlcs.try_call`** — On exception: catches via `__gxx_personality_v0`, extracts the `std::exception::what()` message, stores it in a thread-local buffer via `jlcs_set_pending_exception()`, and returns a zero/null sentinel. The Julia caller checks `jlcs_has_pending_exception()` after return and throws a `CxxException` if set.
- **`CxxException <: Exception`** — New Julia exception type in `JITManager.jl`, wrapping the C++ error message string. Thrown automatically by Tier 2 thunks when the callee raises.
- **Dispatch routing** — `DispatchLogic.jl` updated to route functions marked `noexcept=false` through `try_call` instead of `ffe_call`.
- **C API wrappers** — `jlcs_create_try_call_op`, `jlcs_set_pending_exception`, `jlcs_has_pending_exception`, `jlcs_get_pending_exception`, `jlcs_clear_pending_exception` added to `JLCSCAPIWrappers.cpp` and `MLIRNative.jl`.
- **Lowering** — `TryCallOpLowering` in `JLCSPasses.cpp`: emits `llvm.invoke` to the callee with a landing pad that calls `__cxa_begin_catch`, extracts the `what()` string, calls `jlcs_set_pending_exception`, then `__cxa_end_catch`. Non-exception path falls through normally.
- **Callback test suite** — Extended with C++ functions that throw (`throw std::runtime_error`), verifying that exceptions propagate as `CxxException` on the Julia side.

### Refactor: JIT Symbol Cache — Atomic Copy-on-Write

Replaced the lock-free Dict read pattern in `JITManager.jl` with a proper atomic copy-on-write scheme:

- `compiled_symbols` field is now `@atomic` on `JITContext`.
- **Fast path**: reads an atomic snapshot of the Dict reference — no lock, no race.
- **Slow path**: creates a new Dict copy with the added entry, then atomically swaps the reference.
- Added `init_error` field to `JITContext`; initialization failures are stored and surfaced via `_jit_not_initialized_error()` at all 7 call sites.

### Refactor: Code Cleanup

- **Eliminated `map_cpp_type_to_mlir`** — Deleted the duplicate in `JLCSIRGenerator.jl`, replaced the one call site with `map_cpp_type` from `TypeUtils`.
- **Fixed cross-module reach-through** — Moved `get_stl_container_size` into `TypeUtils.jl`, replaced `Main.RepliBuild.Wrapper.get_stl_container_size` with a direct call.
- **Extracted `getPackedSizeInBits`** — Moved to a static free function in `JLCSPasses.cpp`, removed from both `FFECallOpLowering` and `TryCallOpLowering`.
- **Removed `JL_SubtypeInterface` dead code** — Deleted `JLInterfaces.td`, removed its include from `JLCS.td`.
- **Moved `ASTWalker.jl`** → `Wrapper/ASTWalker.jl`, **`STLWrappers.jl`** → `Wrapper/Cpp/STLWrappers.jl`.
- **MLIR API migration** — All `rewriter.create<Op>(...)` calls in `JLCSPasses.cpp` updated to LLVM 21's `Op::create(rewriter, ...)` builder pattern.

### Removed: Rust Wrapper Generator

Deleted `src/Wrapper/Rust/` (GeneratorRust.jl, TypesRust.jl, IdentifiersRust.jl) and `compile_rust_project()` from Compiler.jl. The experimental Rust generator required `extern "C"` + `#[repr(C)]` on everything, making it effectively a C-ABI wrapper with extra steps. Would need a julia/rust contributer to help with rust because I dont understand the borrow checker enough to deal with it hands on.

### Changed: LTO Global Variable Deduplication

`sanitize_ir_for_julia` now converts externally-visible global variable definitions to `external` declarations in the LTO bitcode. Prevents "Duplicate definition" JIT errors when the shared library is also loaded via `dlopen`.

### Changed: Exports Reorganization

Reorganized `RepliBuild.jl` exports into categorized sections (Core Build Orchestration, Configuration, Compiler Tooling, DWARF Analysis, etc.) and exported additional compiler utility functions for advanced use.

## v2.5.3

### New: STL Map Support (`std::map`, `std::unordered_map`)

Full wrapper generation for `std::map<K,V>` and `std::unordered_map<K,V>` containers, matching the existing `CppVector{T}` and `CppString` pattern.

- **`CppMap{K,V} <: AbstractDict{K,V}`** — New mutable wrapper type in `STLWrappers.jl` that holds an opaque pointer to the C++ map. Lifetime managed by GC finalizer. Supports `getindex`, `setindex!`, `haskey`, `delete!`, `length`, `isempty`, and `empty!` via JIT-compiled MLIR thunks.
- **`CppUnorderedMap{K,V}`** — Type alias for `CppMap{K,V}` (same thunk interface).
- **Map-specific thunk signatures** — `map_at` (key by const ref → value ref) and `map_subscript` (key by const ref → value ref) added to `STLContainerGen.jl`, distinguishing map key-lookup semantics from vector index-lookup.
- **`_classify_stl_method`** — Now accepts an optional `container_type` parameter. `operator[]` and `at()` are classified as `map_subscript`/`map_at` for map containers vs `subscript`/`at` for vectors.
- **Wrapper codegen** — `GeneratorCpp.jl` emits `create_std_map_*()` factory functions for map templates, mirroring the existing vector factory pattern. Template args are parsed via `_split_template_args` to extract K and V types.
- **`_normalize_stl_elem_type`** — Extracted from inline type mapping into a shared helper in `UtilsCpp.jl`. Used by both vector and map factory codegen.
- **`_is_stl_internal_type`** — Expanded blocklist with 13 additional libstdc++/libc++ internal types (`_Alloc_node`, `_Node_handle`, `_Map_base`, `_Insert`, `_Rehash`, `pair<`, `Select1st<`, etc.) that leak through DWARF when wrapping map containers.
- **DWARF byte_size lookup** — Improved container size resolution: uses `get_stl_container_size` first, then fuzzy-matches DWARF keys (now also matches stripped `std::` prefix).

### New: Hub Search (`RepliBuild.search`)

- **`RepliBuild.search(query="")`** — Search the RepliBuild Hub (community package registry) for available packages. Matches against name, description, tags, and language. Shows install status for locally registered packages.
- **`_fetch_hub_index()`** — Fetches and parses `index.toml` from the hub URL via `Downloads.jl`.
- **`REPLIBUILD_HUB_URL`** — Environment variable override for private registries/mirrors.
- Added `Downloads` to `Project.toml` dependencies.

### New: STL Map Test Suite

- `test/stl_test/` — Extended with `std::map<int,int>` coverage: `make_int_map`, `map_lookup`, `map_size` C++ API functions, `CppMap` lifecycle tests (create, insert, read, haskey, delete, empty), and map-passing tests through `const std::map<int,int>&` parameters.

### Changed: Test Directory Consolidation

Reduced the test directory from 14 subdirectories + 8 top-level files to 6 subdirectories + 3 top-level files. All test content preserved through merges:

- **`c_test/`** — Absorbed `basics_test` (PaddedStruct, PackedStruct, NumberUnion, globals, variadic `sum_ints`) and `jit_edge_test` (identity, write_sum, make_pair, PackedTriplet). Pure C with LTO.
- **`stress_test/`** — Absorbed `vtable_test` (Shape/Rectangle/Circle virtual dispatch), `raii_test` (Tracker ctor/dtor), and all standalone MLIR test files (`test_mlir.jl`, `test_mlir_safety.jl`, `test_aot.jl`, `test_raii.jl`). New `verify.jl` covers numerics, vtable dispatch, and conditional MLIR/AOT/RAII sections.
- **`devtests.jl`** — Rewritten to reference the consolidated 6-test suite. Removed duktape setup and standalone MLIR includes.
- **`runtests.jl`** — Added `search` to API surface check.
- **`test_registry.jl`** — Registry integration test updated from `basics_test` to `c_test`.
- **Deleted:** `lua_test/`, `duktape_test/`, `mydir/`, `rust_demo/`, `basics_test/`, `jit_edge_test/`, `vtable_test/`, `raii_test/`, `pugixml_test.jl`, `test_mlir.jl`, `test_mlir_safety.jl`, `test_aot.jl`, `test_raii.jl`.

### Refactor: Dispatch Logic

- **`DispatchLogic.jl`** — Extracted routing logic (`is_ccall_safe`, `is_c_lto_safe`) into a dedicated module, decoupling it from `Utils.jl` and `Generator.jl`.
- **C-Specific LTO Safety** — Introduced `is_c_lto_safe()` for fine-grained C dispatch gates, routing functions with packed struct or union returns to sret thunks while preserving direct `ccall` for safe returns.

### Improved: Cross-Platform DWARF & Target Detection

- **Target Triple Detection** — Added `_detect_target_triple()` in `Compiler.jl` to gracefully determine the host target via `clang -dumpmachine` with a fallback to `Sys.MACHINE`.
- **Robust DWARF Parsing** — `extract_dwarf_return_types` now searches for `llvm-readelf` and `llvm-dwarfdump` as fallbacks when the system `readelf` is missing, improving cross-platform reliability.
- **Return Type Inference** — Added `infer_return_type()` to fallback to demangled C++ function name patterns (e.g. `is_*` -> `bool`, `create_*` -> `void*`) when DWARF debug info is unavailable.

### Refactor: Core API Exports

- **Organized Exports** — Completely reorganized `RepliBuild.jl` exports into categorized sections (Core Build Orchestration, Configuration, Compiler Tooling, DWARF Analysis, LLVM Environment, etc.) for better module discoverability.

## v2.5.2

### New: RAII Dialect Operations

Added C++ constructor, destructor, and scoped lifetime operations to the JLCS MLIR dialect — encoding RAII semantics directly in the IR rather than relying on ad-hoc `llvm.call` emission.

**New operations:**

| Operation | Mnemonic | Purpose |
|-----------|----------|---------|
| `ConstructorCallOp` | `jlcs.ctor_call` | Call a C++ constructor with `this` pointer + parameters |
| `DestructorCallOp` | `jlcs.dtor_call` | Call a C++ destructor with `this` pointer |
| `ScopeOp` | `jlcs.scope` | Region-based RAII scope that guarantees destructor calls at exit |
| `YieldOp` | `jlcs.yield` | Terminator for `jlcs.scope` regions |

- **`jlcs.ctor_call`** — Takes a `FlatSymbolRefAttr` callee and variadic arguments. First argument is always the object pointer (`this`). Lowers to a direct `llvm.call`.
- **`jlcs.dtor_call`** — Takes a `FlatSymbolRefAttr` callee and a single object pointer. Lowers to a direct `llvm.call`.
- **`jlcs.scope`** — Takes managed object pointers as operands and an `ArrayAttr` of matching destructor symbols. Contains a single-block body region. During lowering, body ops are inlined and destructor calls are emitted in **reverse order** (C++ destruction semantics). Not `IsolatedFromAbove` — body can reference values from the enclosing scope.

```mlir
jlcs.scope(%ptr : !llvm.ptr) dtors([@_ZN4BaseD1Ev]) {
  jlcs.ctor_call @_ZN4BaseC1Ev(%ptr) : (!llvm.ptr) -> ()
  // ... use object ...
  jlcs.yield
}
// destructor called automatically here
```

### Changed: `TypeInfoOp` — Destructor Metadata

- `jlcs.type_info` now accepts a fourth argument `destructorName` (default `""`), storing the mangled C++ destructor symbol for the class. IR generators updated to emit the new format.

### New: RAII Test Suite

- `test/test_raii.jl` — 26 tests covering parsing, lowering, and JIT execution of all RAII ops against a compiled C++ test library (`test/raii_test/tracker.cpp`). Validates constructor side effects, destructor side effects, parameterized constructors, scoped lifetime with automatic cleanup, and multi-object scopes with reverse destruction order.

## v2.5.0

### New: Rust Introspective Wrapper Generator

Introduced full support for Rust C-compatible libraries via a dedicated DWARF-based introspective wrapper generator (`src/Wrapper/Rust/`).

- **New `language = "rust"` configuration:** Automatically selects the `rustc` compiler and the Rust generator backend.
- **Topological Struct Ordering:** Autonomously sorts custom structures by dependency, handling pointers (`Ptr{X}`) as soft dependencies to seamlessly emit idiomatic `mutable struct` forward-declarations.
- **DWARF Standard Library Filtering:** Actively identifies and strips out deep internal compiler/stdlib types (like `core::fmt`, `alloc::string`, `std::io::error`, and closure environments) that "leak" through the DWARF metadata, ensuring the Julia wrapper only exposes your public API.
- **Native Enum Resolution:** Correctly infers the underlying primitive types (`Int32`, `UInt32`, `UInt64`, etc.) from DWARF representations, successfully converting signed negative DWARF enum values into their corresponding unsigned native values.
- **ABI Safety Requirements:** Currently, only C-compatible Rust endpoints are supported. Functions must be marked with `extern "C"` and `#[no_mangle]`, and structures/enums must use `#[repr(C)]` or `#[repr(u32)]` to lock their layout for FFI. True native Rust ABI integration (via compiler AST injection) is planned for a future release.

## v2.4.3

### Bug Fix: `WrapConfig` constructor mismatch in Discovery

Fixed a `MethodError` when calling `discover()` caused by the `WrapConfig` constructor in `Discovery.jl` missing the `macros` and `shim_headers` fields added to the struct definition. Empty defaults are now passed for both fields.

## v2.4.2

### Refactor: Wrapper Modularization

The monolithic `src/Wrapper.jl` (~4600 lines) has been split into a structured `src/Wrapper/` package with separate C and C++ sub-packages. `src/Wrapper.jl` is now a thin re-export shim.

**New module layout:**

| File | Lines | Role |
|------|-------|------|
| `src/Wrapper/Generator.jl` | 727 | Top-level `wrap_library()` API; routes to C or C++ generator based on `config.wrap.language` |
| `src/Wrapper/TypeRegistry.jl` | 99 | `TypeRegistry` struct and `TypeStrictness` enum (`:strict`/`:warn`/`:permissive`) |
| `src/Wrapper/Symbols.jl` | 193 | `ParamInfo` and `SymbolInfo` structs for structured symbol representation |
| `src/Wrapper/FunctionPointers.jl` | 77 | DWARF function-pointer signature parser → Julia `@cfunction`-compatible type strings |
| `src/Wrapper/Utils.jl` | 69 | Shared identifier escaping and keyword utilities |
| `src/Wrapper/C/GeneratorC.jl` | 2060 | Full C introspective wrapper generator |
| `src/Wrapper/C/TypesC.jl` | 281 | C type heuristics (`is_c_struct_like`, `is_c_enum_like`) and base type mapping |
| `src/Wrapper/C/IdentifiersC.jl` | 35 | C identifier sanitization |
| `src/Wrapper/C/UtilsC.jl` | 21 | C-specific utilities |
| `src/Wrapper/Cpp/GeneratorCpp.jl` | 2806 | C++ introspective wrapper generator |
| `src/Wrapper/Cpp/TypesCpp.jl` | 428 | C++ type mapping including STL, template, and reference types |
| `src/Wrapper/Cpp/IdentifiersCpp.jl` | 81 | C++ identifier sanitization (namespace stripping, operator handling) |
| `src/Wrapper/Cpp/UtilsCpp.jl` | 44 | C++ utilities |

The C and C++ generators are now fully independent — no shared mutable state, no conditional branching on language inside generation loops. Each generator emits correct stdout-unbuffering preamble, LTO/thunks blocks, struct definitions, and function wrappers for its language.

### Improved: Compiler — JLL-First C Compilation

- C source files (`.c`) are now compiled via `Clang_unified_jll.clang` when available. This produces LLVM IR that exactly matches Julia's internal LLVM version, guaranteeing `Base.llvmcall` compatibility for LTO-enabled C projects. Falls back to system `clang` if the JLL is unavailable.
- `create_library()` and `create_executable()` now select `clang` vs `clang++` based on `config.wrap.language` (previously always used `clang++`).
- `clang --version` probe in metadata extraction also respects `config.wrap.language`.

### New: `wrap.language` Configuration Field

A new `language` field in the `[wrap]` section of `replibuild.toml` selects the generator and compiler toolchain for the project. This field is designed as an extensible language dispatch key — `"c"` and `"cpp"` are the first two targets, with more languages planned.

```toml
[wrap]
language = "c"   # or "cpp" (default)
```

- **`"c"`** — Selects the C generator, compiles with `clang`, and defaults `enable_lto = true` so pure-C libraries get zero-cost `llvmcall` dispatch automatically.
- **`"cpp"`** — Selects the C++ generator (existing behavior), defaults `enable_lto = false`.
- `discover()` auto-detects language from the scanned source files and sets this field accordingly.

### New: C Abomination Stress Test

`test/c_abomination_test/` — a C stress test deliberately constructed to exercise the hardest edge cases the C wrapper generator must handle:

- Deeply nested anonymous structs and unions (3 levels)
- Bitfield members (`uint8_t f1 : 1`, `f2 : 3`, `f3 : 4`)
- Multi-dimensional arrays of structs
- Nested function pointer typedefs (`OuterCallback` returning `InnerCallback`)
- Flexible array members
- Opaque pointer lifecycle (`init_opaque` / `destroy_opaque`)
- Multi-file C project (header + source, pure C, LTO enabled)

### Changed: `.gitignore`

- Added `*.bak` to suppress editor backup files.
- Added `__pycache__/` and `*.pyc` to suppress Python bytecache from helper scripts.

## v2.4.1

### Improved: LTO Pipeline — Bitcode-First Loading

- LTO artifacts now ship as LLVM bitcode (`.bc`) instead of text IR (`.ll`). Julia parses `.bc` substantially faster, reducing wrapper module load time for large libraries.
- The generated wrapper reads bitcode as `UInt8[]` (`read(LTO_IR_PATH)`) — `Base.llvmcall` accepts both text and binary IR.
- `LTO_IR_PATH` and `THUNKS_LTO_IR_PATH` now point to `.bc` files; the `.ll` text files are retained as build-time intermediates only.
- AOT thunks pipeline (`_build_aot_thunks`) also emits `.bc` alongside the `.ll` sanitized IR.

### Improved: LLVM 21+ IR Compatibility

Seven additional LLVM 21 attribute and instruction forms stripped from the sanitized LTO IR to prevent Julia's (potentially older) internal LLVM from rejecting the bitcode:

- `allocptr` pointer-attribute keyword
- `samesign` qualifier on `icmp` comparisons
- `range(...)` return-value attribute
- `nuw`/`nsw` qualifiers on `trunc` instructions
- `nneg` qualifier on `zext` and `uitofp` instructions
- Multi-range `initializes((...), (...))` attribute (previous regex only handled single-range form)
- Complete attribute block replacement: all `attributes #N = { ... }` blocks are now reduced to `{ alwaysinline }`, eliminating future breakage from `allockind`, `allocsize`, `memory(errnomem:...)`, and similar LLVM-version-specific keywords

Both the main LTO path (`link_optimize_ir`) and the AOT thunks path (`_build_aot_thunks`) apply the full set of transforms.

### New: `assemble_bitcode` — JLL-First Bitcode Assembly

- New exported `Compiler.assemble_bitcode(ll_path, bc_path)` function replaces inline `llvm-as` calls throughout the pipeline.
- **Strategy**: first attempts `Clang_unified_jll.clang -emit-llvm` to produce bitcode using the exact same LLVM version Julia uses internally, guaranteeing `llvmcall` compatibility. Falls back to system `llvm-as` if the JLL path is unavailable.

### Improved: C Source File Compilation

- `.c` files are now compiled with `clang` instead of `clang++`. This prevents C code from being parsed with C++ semantics (implicit `extern "C"`, C99 restriction differences, etc.) and silences spurious `clang++` warnings on pure-C projects like SQLite and Duktape.

### Fixed: Wrapper — Forward Declaration Robustness

Three independent bugs corrected in `Wrapper.jl`, validated against SQLite (269 functions), cJSON, http-parser, Duktape, and the full 81-test CI suite:

- **Parameter/return type scanning for opaque structs** — Forward declarations previously only scanned struct members. Types like `sqlite3_blob` that appear exclusively in function signatures (never as struct members) were missing their `mutable struct Foo end` forward declarations, causing `UndefVarError` at module load time.
- **Enum names excluded from forward declarations** — Enum types defined via `@enum` were receiving duplicate empty-struct forward declarations that shadowed the enum. The forward-declaration pass now skips any name already registered as an enum.
- **Union accessor type sanitization and deferred emission** — Union member type names now go through `_sanitize_julia_type_name()` to match the actual emitted struct names (e.g. `__pthread_mutex_s` → `_pthread_mutex_s`). Unknown `Ptr{X}` inner types fall back to `Ptr{Cvoid}`. Accessor functions are now emitted after all struct definitions, eliminating forward-reference errors.

### Fixed: Wrapper — Struct Dependency Ordering

- Introduced `_JULIA_BUILTIN_TYPES` constant — a comprehensive set of all Julia/C interop scalar types that should never trigger a forward declaration or a hard dependency.
- New `_resolve_forward_ptr(julia_type, defined_names)` helper: for any `Ptr{X}` (including nested `Ptr{Ptr{X}}`), replaces `X` with `Cvoid` when `X` is an as-yet-undefined custom struct. This avoids forward-reference errors while preserving correct ABI (all pointers are pointer-sized).
- Struct topological sort now treats `Ptr{X}` as a **soft** dependency (ordering hint only) and `NTuple{N,X}` / `Ref{X}` as **hard** dependencies (inline embedding requires the full definition). Pointer-heavy C++ headers no longer trigger topological sort failures.
- `infer_julia_type` internal-type blocklist check is now applied before any other type dispatch, ensuring compiler-internal types (`__va_list_tag`, `ldiv_t`, etc.) never reach struct or function generation.

### Fixed: Wrapper — Template Struct Member Sanitization

- Union and struct member types containing `<>` (C++ template syntax) are sanitized before emission: `Ptr{stl_internal<char>}` → `Ptr{Cvoid}`, bare template types → size-based `NTuple{N,UInt8}` or `Ptr{Cvoid}`.
- Prevents syntax errors in generated wrappers for libraries that expose STL types in their public interface (tested against Duktape and ImGui configs).

### Improved: Metadata — Absolute Include Paths

- `include_dirs` in `compilation_metadata.json` are now stored as absolute paths. This prevents `wrap()` from failing when called from a working directory different from the project root.

### New: Test Suite

- **Registry test suite** (`test/test_registry.jl`) — 494-line isolated test covering the full `register`/`unregister` lifecycle, content-addressed deduplication, TOML hash normalization, build artifact caching, environment-check TTL, index persistence, and error cases. Uses isolated `REPLIBUILD_HOME` via temp dirs to avoid polluting the user's real registry.
- **Duktape integration test** (`test/duktape_test/`) — Wraps the Duktape JS engine (pure C amalgamation, ccall tier, LTO off). Tests heap lifecycle, `duk_eval_string`, stack push/pop, and string/number/boolean round-trips.
- **Developer test runner** (`test/devtests.jl`) — New script for developer machines that runs the full integration suite (Lua, SQLite, cJSON, Duktape, vtable, JIT edge cases, registry). Separated from CI to keep `runtests.jl` fast.
- **CI cleanup** — Removed ~15 outdated standalone test directories (`benchmark_test`, `custom_test`, `hello_world_test`, `lto_benchmark_test`, `stdlib_test`, `stl_test`, etc.) that were superseded by the unified stress-test suite.

### Changed: Documentation Layout

- `docs/ARCHITECTURE.md` → `docs/architecture.md`
- `docs/DEEP_TECHNICAL_ANALYSIS.md` → `docs/technical-reference.md`
- `benchmark_results.md` (repo root) → `docs/benchmark_results.md`
- Removed `docs/TECHNICAL_INDEX.md` and `docs/TECHNICAL_SUMMARY.txt` (content superseded by architecture and technical-reference docs)
- `*.code-workspace` added to `.gitignore`

## v2.4.0

### New: Global Package Registry

- **`RepliBuild.use("lua")`** — One-call wrapper loading: looks up the registry, resolves git/system/local dependencies, checks the environment, builds if needed, wraps, caches artifacts, and returns a loaded Julia module.
- **`RepliBuild.register(toml_path)`** — Hash (SHA256) and store a replibuild.toml in the global registry at `~/.replibuild/registry/`. Auto-called by `discover()`.
- **`RepliBuild.list_registry()`** — Print all registered packages with hash, source, build status, and registration date.
- **`RepliBuild.unregister(name)`** — Remove a package from the registry and clean cached builds.
- **Global build artifact caching** in `~/.replibuild/builds/<hash>/` — repeated `use()` calls load cached builds instantly.
- **Environment check caching** in `~/.replibuild/toolchain.toml` — avoids re-probing LLVM/Clang on every call (24h TTL).
- `discover()` now auto-registers the generated TOML in the global registry.
- `scaffold_package()` pulls TOML from registry when the name matches a registered package.
- Scaffold.jl merged into PackageRegistry.jl — single unified module for package management.
- Respects `REPLIBUILD_HOME` env var for custom registry location (default: `~/.replibuild/`).

### Fixed: Enum Extraction

- Replaced regex-based enum extraction with Clang.jl AST walker — correctly ignores Doxygen comments, handles `enum class`, hex values, and namespaces.
- Complete Julia keyword escaping (`in`, `and`, `or`, `not`, `isa`, `where` etc.) via shared `_JULIA_KEYWORDS` set.
- Internal type blocklist (`__va_list_tag`, `ldiv_t`, etc.) filters compiler internals from exports.
- Auto-detects enum underlying type (`UInt32`/`Int64`) for values exceeding `Int32` range.
- Eigen wrapper: 1507 → 1106 lines, all 14 verify.jl tests pass.

## v2.3.0

### New: Environment Diagnostics ("Doctor")

- **`RepliBuild.check_environment()`** — Comprehensive toolchain validation that checks for LLVM 21+, Clang, mlir-tblgen, CMake, and the compiled JLCS dialect. Prints a colorful, readable diagnostic report with per-OS installation instructions when tools are missing.
- Automatically runs before `build()` — if the toolchain is incomplete, users get actionable fix instructions instead of cryptic cmake/ccall failures.
- Returns a `ToolchainStatus` struct for programmatic use (`status.ready`, `status.tier1_ready`, `status.tier2_ready`).

### New: Standardized Package Scaffolding

- **`RepliBuild.scaffold_package("MyEigenWrapper")`** — Generates a complete, distributable Julia package structure for RepliBuild wrappers: `Project.toml`, `replibuild.toml`, `src/` stub, `deps/build.jl` hook, and `test/` skeleton.
- Standardizes how wrapper packages are structured and distributed. Users edit `replibuild.toml` and run `Pkg.build()`.

### New: Automatic JLCS MLIR Dialect Compilation

- **`deps/build.jl`** — Automatically compiles the JLCS MLIR dialect (`libJLCS.so`) when RepliBuild is installed via `Pkg.add`. Detects CMake, LLVM, and MLIR, runs the build, and caches the result with a source-content hash.
- Graceful degradation: if the MLIR toolchain is missing, Tier 1 (ccall) builds still work; only Tier 2 (MLIR JIT) is unavailable.

### Improved: Aggressive Hash-Based Caching

- **Project-level content hashing** — The build cache now hashes `replibuild.toml` content, all source file contents, all header file contents, and the git HEAD of the project root. If the hash matches the cached artifacts, `build()` returns in sub-second time without invoking any compiler.
- Replaces the previous mtime-only file cache (which is still used for per-file IR caching) with a project-wide fast-exit path.

### Improved: README Philosophy Section

- Added a "Philosophy" section explaining the source-based approach vs JLLs/BinaryBuilder, framing the heavy toolchain requirement as a deliberate design choice for zero-overhead, zero-edit bindings.

## v2.2.1

### Fix: Wrapper Generator — C++ Namespace & Operator Correctness

Seven bugs fixed in `Wrapper.jl` that caused the generated wrapper to fail parsing or crash at runtime when wrapping real-world C++ libraries (validated against pugixml 1.15):

- **Template type sanitization on `Ptr{}`-wrapped builtins** — `Ptr{xml_stream_chunk<char>}` was skipping angle-bracket sanitization because the outer `Ptr` triggered `is_builtin`. Now also sanitizes when `<>` are present, regardless of `is_builtin`.
- **STL-internal type check on wrapped inner types** — `_is_stl_internal_type` was called on `Ptr{char_traits<char>}` (starts with `Ptr{`), always returning false. Now extracts the inner type before checking.
- **Destructor finalizers use mangled symbol** — Finalizers generated `ccall((:~ClassName, lib), ...)` which is a syntax error at Julia parse time. Now uses the mangled C++ symbol (`_ZN...D2Ev`) from `deleters_mangled`.
- **`this` parameter namespace prefix stripping** — When a class is `pugi::xpath_query`, the Julia struct is `xpath_query` (no namespace). Now correctly strips the namespace prefix by scanning for the last `::` at angle-bracket depth 0.
- **Namespace-only "class" guard for free functions** — Free functions in a C++ namespace (e.g. `pugi::get_memory_allocation_function`) were parsed with `class="pugi"` and received a spurious synthesized `this` parameter. Now only synthesizes `this` if the bare class name is a known struct type.
- **Operator function name `>` depth confusion** — `operator>=` / `operator>` contain `>` which corrupted angle-bracket depth tracking, producing garbled type names. Now heavily sanitizes `safe_class` and falls back to `Cvoid` for any `operator…` class.
- **Parameter `::` sanitization** — Namespace-qualified types in DWARF parameter lists (e.g. `pugi::xml_attribute`) were emitted verbatim. Added a second sanitization pass to convert `::` and remaining non-identifier characters.

### New: Build Orchestration & Dependency Resolution
- **Zero-Boilerplate Git Dependencies** — `DependencyResolver.jl` introduces native `[dependencies]` blocks in `replibuild.toml` to automatically fetch, filter (via `exclude`), and inject raw external C/C++ git repositories into the Clang compilation pipeline.
- Bypasses the need for BinaryBuilder / JLL packages for local development, guaranteeing full DWARF extraction on arbitrary upstream code.

### New: Cross-Language LTO (Link-Time Optimization)
- **Zero-Cost Abstractions via `Base.llvmcall`** — When `enable_lto = true`, the compiler now emits an LLVM Bitcode payload (`_lto.bc` and `_lto.ll`). The generated Julia wrapper intercepts safe primitive/pointer FFI boundaries and dynamically loads the LLVM IR at parse-time, routing the execution through `Base.llvmcall` instead of `ccall` to allow Julia's JIT to inline C++ code directly into Julia hot loops.

### New: MLIR Ahead-Of-Time (AOT) Thunks
- **Static C++ Vtable Dispatch** — Introduced `aot_thunks` flag in the configuration to statically compile MLIR JLCS thunks directly into `.o` artifacts, linking them into a native `_thunks.so` companion library during the `build()` phase.
- Generated `Wrapper.jl` now conditionally emits purely static `ccall` bindings that bypass the `JITManager` runtime entirely for zero-overhead, statically-verifiable polymorphic execution.

### New: Automated Template Instantiation
- **Declarative Template Resolution** — Added `templates` and `template_headers` to the `[types]` config. The compiler automatically generates dummy C++ source files to force Clang to instantiate the requested types (e.g. `std::vector<int>`), guaranteeing they appear in the DWARF debug metadata for MLIR processing and FFI wrapping.

### Improved: Wrapper Ergonomics
- **Idiomatic Julian Classes** — The wrapper generator now semantically clusters factory functions (`create_circle`), destructors (`delete_shape`), and instance methods from the DWARF metadata to emit high-level, idiomatic `mutable struct` wrappers.
- **Julian Multiple Dispatch** — C++ instance methods are automatically proxied via multiple dispatch (e.g., `area(c::Circle)`) passing the raw C pointers via `Base.unsafe_convert`.
- **Automatic Garbage Collection** — C++ object lifecycles are now safely and natively managed by Julia's GC via implicitly registered finalizers on the generated structs.

## v2.1.0

### New: MLIR JIT Compilation Pipeline

- **JITManager.jl** — New module managing MLIR JIT lifecycle with lock-free symbol cache and arity-specialized `invoke` methods (1-4 args, zero heap allocation)
- **Tiered dispatch** — Functions auto-classified as ccall-safe (Tier 1) or JIT-required (Tier 2). Packed structs, unions, virtual dispatch, and large struct returns route through MLIR JIT transparently
- **ir_gen/ submodule** — `TypeUtils.jl`, `StructGen.jl`, `FunctionGen.jl` for modular MLIR IR generation with topological struct sorting and packed struct marshalling

### New: Wrapper Generator Capabilities

- **Union support** — `mutable struct` with `NTuple{N,UInt8}` backing + typed getter/setter accessors
- **Bitfield support** — Bit-shift extraction for single-byte, `unsafe_load`-based for multi-byte fields
- **Variadic function support** — Typed overloads from `[wrap.varargs]` config
- **Global variable accessors** — `cglobal` + `unsafe_load` wrappers
- **Automatic finalizer generation** — Detects destructors/deleters, generates `ManagedX` types with GC-traced finalizers and `Base.unsafe_convert`
- **Virtual method dispatch** — Generates JIT thunk wrappers for virtual functions
- **Forward declarations** — Opaque/circular struct references handled via forward-declared empty structs
- **Base class member flattening** — Inherited fields prepended in struct definitions
- **Struct padding** — Explicit `_pad_N::NTuple{K,UInt8}` fields for correct memory layout

### Improved: DWARF Parser

- Union, bitfield, global variable, typedef extraction from debug info
- Varargs and virtual method detection
- Robust state-machine rewrite (`parse_dwarf_output_robust`) replacing fragile implicit tracking
- Struct member data — `MemberInfo` with offsets now propagated through the pipeline

### Improved: Compiler

- Multi-level pointer resolution (`T**` -> `Ptr{Ptr{T}}`)
- Reference type resolution (`T&` -> `Ref{JuliaType}`)
- Expanded type map — `ssize_t`, `ptrdiff_t`, `intptr_t`, `int8_t`..`uint64_t`, etc.
- Library search path (`-L`) support
- Const/volatile stripping uses word-boundary regex (no more mangling `"constructor"` -> `"ruor"`)

### Improved: MLIRNative

- JIT execution engine — `create_jit`, `destroy_jit`, `lookup`, `jit_invoke`, `invoke_safe`
- Module cloning, function introspection, type predicates
- `lower_to_llvm` pass pipeline

### Changed: Dependencies

- **Added**: `BenchmarkTools`, `Libdl`
- **Removed**: `Distributed`, `RepliBuildPaths.jl` (451-line directory management system)
- **Julia minimum**: 1.9 -> 1.10
- **Clang compat**: now accepts 0.18 + 0.19

## v2.0.3

- Initial public release with DWARF-based wrapper generation
- Clang.jl integration for header parsing
- Introspection toolkit (binary analysis, benchmarking, data export)
- MLIR/JLCS dialect foundation
