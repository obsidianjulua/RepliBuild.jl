# Internals & Dispatch

!!! note "Developer documentation"
    This page is for contributors. The user manual starts at [Home](index.md).
    Sister pages: [JLCS / MLIR](mlir.md), [Inheritance ABI](inheritance-abi.md),
    [Developer index](developer.md).

RepliBuild compiles C/C++ with Clang, reads the DWARF debug metadata the compiler emits about its own output, and generates Julia bindings that are correct by construction — struct offsets, enum underlying types, vtable slots, base-subobject offsets, and calling conventions all come from the compiler's own record rather than being guessed. Where DWARF is incomplete (enums the optimizer folded away, macro definitions, function-pointer typedefs) the Clang.jl AST fills the gap; where Julia's computed struct alignment disagrees with the DWARF size, the struct is treated as packed and routed away from `ccall`.

This page is the technical reference for how that pipeline is assembled — the stages, the three dispatch tiers, and the modules behind each.

Tier 2 — the MLIR/JLCS marshalling layer — has its own page: [ABI Marshalling as Compiler IR](mlir.md) covers the dialect's types and ops, the SysV lowering, the thunk calling contract, source-level debugging, and the failure classes the design exists to make loud. This page describes where it sits; that one describes what it is.

## The pipeline

Source becomes a loadable Julia module in seven stages, each owned by one module:

1. **DependencyResolver** — resolve the `[dependencies]` table (git / local / system) and merge the external sources into the build graph.
2. **Discovery** — scan files and the `#include` graph; write or refresh `replibuild.toml`.
3. **Compiler** — Clang compiles each translation unit to LLVM IR, cached on source `mtime` plus a compile fingerprint (flags, defines, include dirs, LLVM version, target triple).
4. **Linker** — link, optimize, and assemble the IR into the `.so` (plus LTO bitcode when `enable_lto` is on).
5. **DWARFParser** — `llvm-dwarfdump` + `nm` produce `ClassInfo` / `VtableInfo` and the `compilation_metadata.json` layout facts.
6. **Wrapper generator** — DWARF + symbols drive the C or C++ generator to emit the Julia module.
7. **JITManager** — at module load, stand up a per-library MLIR thunk engine for any Tier-2 calls.

For **C** projects the link/optimize/assemble steps run **in-process on Julia's resident libLLVM**, version-matched to the JLL clang that emitted the IR (no external LLVM, no DWARF-dropping version skew); a failure is a hard error unless `[link] fallback = true` selects the external `llvm-link`/`opt` pipeline. **C++** always uses the external pipeline plus the system MLIR dialect. Only the final `.ll → .so` codegen shells to clang in both buckets.

## Three dispatch tiers

Each wrapped function is routed to exactly one calling tier, chosen from its DWARF signature:

| Tier | Mechanism | Selected when |
|------|-----------|---------------|
| 1 | `Base.llvmcall` against a per-function bitcode slice | POD args, scalar/pointer return, `[wrap.tier1] enable = true` (C only) |
| 2 | MLIR thunk via `libJLCS.so` (JIT at load, or AOT `_thunks.so`) | packed structs, unions, large by-value struct returns, C++ classes, virtual dispatch, exceptions |
| 3 | `ccall` into the `.so` | the unconditional fallback |

The routing decision lives in `Wrapper/DispatchLogic.jl` (`is_ccall_safe`, `is_c_lto_safe`); the [Tier selection logic](#Tier-selection-logic) section below lists the exact checks.

!!! note "Two Tier-1 payloads, one supported"
    Tier 1 can carry its IR two ways, and the knobs are independent.

    **Per-function slices — `[wrap.tier1] enable`** (C only, default off) is the supported path. `_tier1_slice_prepass` (`Wrapper/C/GeneratorC.jl`) runs `IRGen/Slicer.jl` over every `is_c_lto_safe` non-varargs candidate, applies the hazard/size policy, and `dlsym`-pre-flights each slice's declarations against the real `.so`. Acceptance only makes a function *eligible*: a call site additionally needs `lto_shape_ok` (no Cstring or struct crossing) and must survive the signature dedup, so `_tier1_emit_slices!` writes `julia/slices/<mangled>.ll` from the final wrapper text — only slices a call site actually reads reach disk, and `TIER1_FUNCTIONS` is derived in the same pass. Rests on static promotion (`_promote_statics_libllvm`) to guarantee every declared symbol reaches the dynamic symbol table. Decided at generation time — non-accepted functions emit plain `ccall`, and the wrapper exports `TIER1_FUNCTIONS`.

    **Whole-module bitcode — `[link] enable_lto`** embeds the entire linked module at each call site: scale-limited (can crash Julia's JIT on large libraries) and it duplicates file-local `static` state between the embedded bitcode and the `.so`. Production configurations set `enable_lto = false`; C++ defaults to LTO off. Treat it as an experimentation path for small stateless kernels.

    See [Tier 1](tier1.md).

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

The function `is_ccall_safe()` in `src/Wrapper/DispatchLogic.jl` is the core dispatch decision. It inspects each function's DWARF metadata and returns `false` when the signature needs an MLIR thunk (Tier 2), `true` when it can be called directly (Tier 1 or Tier 3).

Choosing *between* the two direct tiers is a second, C-only step. `is_c_lto_safe()` marks a function Tier-1-eligible unless its **return type** is a packed struct or a by-value union — C has no templates, vtables, STL, or inheritance, so those are the only ABI hazards, and by-value parameters are fine either way. Eligibility is necessary but not sufficient: with `[wrap.tier1] enable = true` the function must also survive slicing, the hazard/size policy, and the `dlsym` pre-flight. Anything that does not, and everything when Tier 1 is off, emits `ccall`.

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
6. **Static promotion** (`_promote_statics_libllvm`, C in-process bucket, `[link] promote_statics`): see below.
7. **Codegen:** The final `.ll → .so` step shells to clang/clang++.

### Static promotion

Post-optimization and pre-codegen, every function or global that a Tier-1 bitcode slice may bind by `declare` **but that cannot reach the `.so`'s dynamic symbol table** is renamed to an exported `__rb_<lib>_<name>` with external linkage and default visibility. The test is *"not exportable"*, not *"internal linkage"*: `default<O…>` runs no internalize pass, so it covers both internal/private linkage (file-local statics) **and** external-linkage-but-`hidden`/`protected` symbols — Lua's `LUAI_FUNC` functions and `LUAI_DDEF` tables are exactly the latter shape. Only *internal* constants stay internal: no symbol exists for a slice to bind, and read-only data has no divergence class.

The rename happens on the one module (`<name>_abi.ll`) that becomes both the `.so` and the slice source, so the two are bit-identical by construction — which is what removes the whole-module path's duplicated-`static` failure class rather than papering over it. The old→new map lands in `compilation_metadata.json` under `promoted_symbols`, and `extract_symbols_from_binary` filters `__rb_*` so promoted statics never surface as wrappable API.

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
| `StructGen` | `src/IRGen/ir_gen/StructGen.jl` | struct metadata | Struct type aliases + registration IR; aligned-vs-packed LLVM struct type strings; members laid out at their **DWARF offsets** with explicit padding and verified against a Julia mirror of LLVM's `abiSize`/`abiAlign`, degrading to a correctly-sized opaque region when they cannot be; packed structs nested by value in other struct bodies are inlined as byte-identical LLVM literals |
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

## Slicer

**Source:** `src/IRGen/Slicer.jl`

Per-function bitcode slicing for Tier 1, on Julia's resident libLLVM. `slice_library(abi_ll; targets, cache_dir)` parses the promoted module once and clones it per target (`LLVMCloneModule`), then strips the clone to declarations: `LLVMFunctionDeleteBody` for every reached function, `LLVMSetInitializer2(gv, NULL)` for reached mutable and external constant globals, internalize + `globaldce` for everything unreached. Internal constants are embedded rather than declared. Every slice is verified before it is returned, and results are cached content-addressed under `<cache>/slices/`.

The closure is **one level deep by construction** — a declared function contributes no edges of its own — so slice size tracks the target function, not the library. `lua_gettop` cuts 15.8 MB down to 2.8 KB; `luaL_openlibs` lands at 6 KB. This is why `max_slice_kb` is a tripwire rather than a tuning knob.

Anything the Slicer cannot slice *correctly* comes back as a **refusal** with a reason, never as silently-wrong IR: a variadic target, a `blockaddress` into a body being deleted, alias/ifunc, or an unpromoted module (the fail-loud guard against slicing `_opt.ll` by mistake). Softer shapes come back as hazard flags for the generator's gate — `:setjmp_family`, `:varargs_callee`, `:noinline`, `:weak`, `:inline_asm`, `:module_asm`.

Each `SliceResult` also records the symbols the slice `declare`s, post-DCE and excluding intrinsics. `_tier1_preflight!` in the C generator `dlopen`s the `.so` `RTLD_GLOBAL` and `dlsym`s each one — the exact lookup ORC will perform at first call — because an unresolved `declare` does not raise: ORC prints `Symbols not found: [...]` and then blocks forever. A miss demotes that function to `ccall`; a `.so` that will not `dlopen` disables Tier 1 for the whole wrap.

## ThunkBuilder

**Source:** `src/Builder/ThunkBuilder.jl`

AOT compilation path for Tier 2 thunks. When `aot_thunks = true` in `replibuild.toml`, this module drives the same `JLCSIRGenerator.generate_jlcs_ir()` used by the JIT path, lowers the result with `MLIRNative.lower_to_llvm()`, emits an object file through `MLIRNative.emit_object()`, and links it against the user's compiled library (`clang`/`clang++ -shared`, rpath'd to the library directory) into a companion shared library named `<libname>_thunks.so`. With `[link] enable_lto` on it additionally emits and assembles the thunks' own LTO bitcode. An AOT failure is a warning, not a build failure — the JIT path remains available.

The Julia wrapper then `ccall`s into the AOT thunks rather than calling `JITManager.invoke`. There is no MLIR JIT at runtime — `libJLCS.so` is only needed at build time for the lowering step. After AOT compilation, the user can ship the wrapped library + thunks `.so` without bundling LLVM/MLIR runtime libraries.

## MLIRNative

**Source:** `src/IRGen/MLIRNative.jl`

Low-level `ccall` bindings to `libJLCS.so`, the compiled JLCS MLIR dialect shared library. Provides context management, module parsing, JIT engine creation, LLVM lowering, symbol lookup, and the object/IR emission used by the AOT path. Building the dialect (`cd src/mlir && ./build.sh`) is required only for the C++/Tier-2 bucket.

Two behaviours here are load-bearing beyond plain FFI plumbing. `parse_module` names the parse buffer after a **content-hashed file it writes** under the library's `.debug/mlir/` — MLIR's parser stamps that name onto every op as a `FileLineColLoc`, and the lowering turns it into the emitted DWARF's `DIFile`, which is what makes a JIT'd thunk steppable in gdb. And `jit_source_path` falls back to a temp directory when `.debug` is unwritable (a read-only install), because losing co-location costs nothing while losing the source view costs the whole capability.

The dialect itself — its two types, fourteen ops, lowering pass, and calling contract — is documented in [ABI Marshalling as Compiler IR](mlir.md).

## JITManager

**Source:** `src/IRGen/JITManager.jl`

Runtime for Tier 2 dispatch: **one MLIR execution engine per wrapped binary** (`LibraryEngine`), held in a process-wide `GLOBAL_JIT` behind a shared, lock-free thunk cache.

### Key design points

- **Per-library engines.** `initialize_global_jit(binary_path)` is called from each generated module's `__init__` and creates (or reuses) the engine for *its* binary. Multiple wrappers coexist in one session — previously the first wrapper won and the second library's entire Tier 2 silently died, found while composing box2d with pugixml. A per-library initialization failure degrades only that library, and a missing-symbol error names every engine that was searched.
- **Manifest-driven initialization:** `initialize_global_jit()` reads `thunk_manifest.json` — the thunks the wrapper actually dispatches to — so dead thunks are never generated. Any initialization failure (including the pre-flight rejection of untranslatable IR types in `libJLCS`) degrades the module to "Tier 2 disabled" with `ccall` wrappers intact, never a process crash.
- **Symbol registration before lowering:** the engine is given the library and `libJLCS.so` as shared libraries, and the C++ runtime EH symbols (`__gxx_personality_v0`, `__cxa_begin_catch`, `__cxa_end_catch`) plus the `jlcs_*` exception helpers are registered explicitly, since JIT'd landing pads reference them by name.
- **Lock-free hot path:** `_lookup_cached()` reads from an `@atomic` snapshot of the symbol dictionary with no locking. The cache is published copy-on-write — a fresh dict is built with the new entry and atomically swapped in. Readers always see a stable, immutable snapshot.
- **Arity specialization:** `invoke` is `@generated`, emitting arity-specialized code for any argument count — stack-allocated `Ref`s and a fixed-size `Ptr{Cvoid}[]`, allocation-free at every arity. A thunk slot holds a pointer to the argument's *storage*, so `Ref(x)` is the right shape for an `isbits` `x`; the two kinds that are **already** an indirection — an `AbstractString`, and a `Base.Ref` that is not a `Ptr` (what a caller passes for a C++ `T const&` parameter) — are flattened to a raw pointer first and GC-preserved across the call. Both `invoke` methods share one `_arg_marshal_plan`, because two copies is how a fix to one silently misses the other.
- **`@generated` return dispatch:** `_invoke_call` resolves at compile time whether the return type is a primitive (direct `ccall` return) or a struct (`sret` buffer allocation). An unresolved `Any` return fails loudly with the actual cause instead of corrupting memory.
- **Exception propagation:** After every Tier 2 call, `_check_pending_exception()` polls the thread-local exception buffer set by `jlcs.try_call` lowering. If a C++ exception was caught during the call, a `CxxException` is thrown with the original `what()` message.

### Calling convention

All Tier 2 functions use a unified `ciface` calling convention:

| Return | Signature |
|--------|-----------|
| Scalar | `T ciface(void** args_ptr)` |
| Struct | `void ciface(T* sret, void** args_ptr)` |
| Void | `void ciface(void** args_ptr)` |

## Debug

**Source:** `src/Debug/Debug.jl`

Static inspection of what the Tier-2 pipeline actually emitted, for a package
this process never built. `thunks(pkg)` lists the thunk symbols;
`mlir_body(pkg, symbol)` prints the generated dialect; `disassemble(pkg;
symbol=…)` shells to `objdump -dS` so dialect ops and machine code interleave;
`dwarf(pkg; section=…)` shows the address → MLIR-line table; `walk(pkg, symbol)`
does the common combination in one call.

The object file it disassembles only exists when the JIT's object cache was
enabled, and MLIR requires that at engine-creation time — so it is read from
`REPLIBUILD_JIT_OBJDUMP` **before the wrapper loads**, not passed as an argument.
Nothing here links gdb or LLVM: it shells to `objdump` and `llvm-dwarfdump`.
See [Debugging a thunk](mlir.md#9.-Debugging-a-thunk) for the live-process
counterpart.

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
