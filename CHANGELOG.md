# Changelog

All notable changes to RepliBuild.jl are documented in this file.

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
