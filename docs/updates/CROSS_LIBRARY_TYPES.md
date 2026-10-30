# Cross-library types — design

**Date:** 2026-10-30
**Status:** design only. No code is changed by this document.
**Base:** `main` @ `e0dfddb`, plus the build-graph contract
(`build-graph/contract` @ `9c9c507`, `src/Builder/BuildGraph.jl`). Every
function named here was read on that date. Line numbers are left out on purpose.

**Problem.** Several libraries are built from one project graph and each gets
its own wrapper, while library B's API uses types that library A defines. In
ROOT, Tree's API takes Hist's `TH1*`, and Hist's API uses Core's `TObject`.

**Recommendation, in short.** Option 1: B's wrapper imports the types A owns
and does not emit them. Ownership is decided by the build graph plus what each
wrapper actually emitted, not by header paths or symbol names alone. Each
library's wrapper stays its own module, binding only its own library. The
modules are delivered as submodules of one generated **graph module**, in
topological order, so an import is a relative `const X = ..A.X`. Section 6 has
the details, section 7 the tests.

---

## 1. What "the same type" has to mean

A C++ type crossing from A to B has to be one Julia type on both sides. All of
these have to work:

- a `TObject*` that a Core function returns, passed to a Hist function;
- a `Managed<TObject>` passed the same way;
- a `TObject` passed by value;
- a Core enum value passed as a Hist argument;
- a Hist struct that embeds Core's record;
- a Hist class that derives from Core's class;
- user code that dispatches on `Core.TObject`.

Today none of these works except the raw pointer (section 2.6).

---

## 2. Today: every wrapper is self-contained

### 2.1 Types come from one binary's DWARF

- **Parsing.** `Compiler.extract_dwarf_return_types(binary_path)` dumps one
  library and calls `parse_dwarf_dump(output; file_tables)`.
  `_dwarf_file_tables` resolves each compile unit's `DW_AT_decl_file` indices,
  and the parser records a `decl_file_path` on each type DIE.
- **The only provenance filter is `drops_type`.** It calls
  `_is_system_decl_file`, a blocklist of toolchain and system header
  directories. It also drops an anonymous type whose named parent is dropped.
  It asks "is this a toolchain header?", never "which project library declared
  this?".
- **One definition per bare name.** Every named definition is collected per
  bare name, and `_canonical_definition` keeps the one whose `_canonical_text`
  sorts first. Definitions that disagree on `_layout_signature` mark the kept
  one `conflict`, with `conflict_sizes`.
- **ABI facts.** `record_abi` carries size, pass convention and emptiness for
  every record a function passes by value. It is collected from all
  definitions, before the provenance gate.
- **The metadata file.** `extract_compilation_metadata` writes
  `compilation_metadata.json` beside the library, holding:
  - `struct_definitions`, with enums under `__enum__<name>`;
  - `record_abi`;
  - `functions` (this binary's `nm -g --defined-only`, merged with DWARF);
  - `globals`;
  - `typedef_table`.

So a type is in B's metadata exactly when B's DWARF has a full definition of
it. Which types those are is a compiler decision.

### 2.2 What clang puts in B's DWARF (measured)

RepliBuild adds only `-g` (`compile_single_to_ir`), so clang's default
debug-info homing applies.

**The probe.** A two-TU probe on the reference host: clang 23.1.4,
`x86_64-w64-windows-gnu`, `-g -O0`.
- `a.h` declares three types:
  - a POD `Plain`;
  - an enum `Color`;
  - a class `Base` with an out-of-line constructor and out-of-line virtuals.
- `a.cpp` defines `Base`'s members and uses `Plain`.
- `b.cpp` defines `Derived : Base` and `int use(Base*, Plain, Color)`.

| Type | in `a.o` | in `b.o` |
|---|---|---|
| `Base` (dynamic, key function in `a.cpp`) | full definition | **declaration only** (`DW_AT_declaration`) |
| `Plain` (header-only POD) | full definition | full definition |
| `Color` (header-only enum, unused in `a.cpp`) | **absent** | full definition |

Vtable homing and constructor homing (the open F4 entry in CLAUDE.md) keep a
dynamic or constructor-homed class's full description in the one library that
emits its vtable or constructor.

**What this means for B.**
- B's metadata holds A's header-only types in full.
- It holds A's dynamic classes not at all: they reach B's signatures and
  members as names with no definition.
- A header type that A never uses is not in A's metadata either.

Any ownership rule has to work from those facts.

### 2.3 What the generators emit

`generate_introspective_module_cpp` (GeneratorCpp) emits, from that one
metadata file:

- **Forward declarations.** Every name a member or signature reaches without a
  definition is emitted as `struct X end`. B therefore emits its own empty
  `TObject`.
- **Records.** Records are emitted in Kahn order, with ties broken by
  `sort(ready)`. Their layouts are proved by `_prove_julia_layout`, or emitted
  as a blob.
- **Base members.** `flatten_struct_members` flattens base members into the
  derived record, but only for bases present in **this** library's
  `struct_definitions`. A base that B's DWARF only declares contributes no
  fields.
- **Upcasts.** `<Derived>_as_<Base>` helpers come from `_collect_upcasts!`,
  whose walk stops at a base absent from this metadata.
- **Deleters and `Managed<X>`.**
  - The deleters scan reads this library's `functions`: the library's deleter
    function for X, or X's D0.
  - `Managed<X>` defines `unsafe_convert(::Type{Ptr{X}}, ::ManagedX)`, and its
    finalizer calls `ccall((:sym, LIBRARY_PATH), …)`.
- **Factory structs.** The factory struct (`idiomatic_classes`) uses D0 only
  when `has_subclass`, the set of classes something **in this library** derives
  from, lacks the class.
- **`<f>_safe` wrappers.** Every function returning `Ptr{X}`, where X has a
  deleter, gets an `<f>_safe` that wraps the pointer in `Managed<X>`.
- **Enums.**
  - Enums come from DWARF, plus supplementary header enums
    (`ClangJLBridge._extract_enums_via_ast!`).
  - A header enum is kept when its file lies under a named header's directory
    or an include dir. In a graph build B's include path contains A's
    directories, because `BuildGraph.IncludeDir` carries the inherited ones. So
    B can pick up A's enums through this path even when B's DWARF lacks them.
- **Introspection.** `_layout_chunk` emits `STRUCT_SIZES`/`STRUCT_OFFSETS` for
  every emitted record, and `_dispatch_tier_chunk` emits `DISPATCH_TIER`.
- **Parameters.** A pointer parameter is `::Any` in the Julia signature and
  `Ptr{X}` in the ccall. A by-value parameter is typed.

`generate_introspective_module_c` (GeneratorC) has the same model for C:
- records through `_resolve_exact_layout`;
- `@enum`, with `const Alias = Canonical` for an identical member set;
- `_resolve_forward_ptr`, which degrades an unemitted name to `Ptr{Cvoid}`;
- `_as_declared` for signature spellings.

**The precedent for the change.** `wrap_introspective` already mutates the
metadata in one place, before both generators run: `_apply_symbol_filter!`,
`_drop_unresolvable_globals!` and `_apply_surface_type_filter!`. That is the
pattern section 6 follows.

### 2.4 Tier 2

**Engines and the thunk cache.**
- `JITManager.initialize_global_jit(LIBRARY_PATH)` creates one `LibraryEngine`
  per `realpath` in `GLOBAL_JIT.engines`.
- It reads `compilation_metadata.json` and `thunk_manifest.json` **from the
  library's own directory**, runs `DWARFParser.parse_vtables` on that binary,
  and calls `JLCSIRGenerator.generate_jlcs_ir(vtinfo, metadata;
  needed_symbols)` at load time.
- The thunk cache, `GLOBAL_JIT.compiled_symbols`, is keyed by the thunk name
  `_mlir_ciface_<mangled>_thunk` **alone**. `_lookup_cached` walks the engines
  in init order, and the first engine that has the name wins for good.

**Symbol resolution.** `jlcsEngineCreate` (`JlcsJIT.cpp`) puts the library
and libJLCS each in a JITDylib. The thunk module's own JITDylib gets a
process-wide search generator.

**Crossings.** `FunctionGen.thunk_crossings` calls `by_value_crossing` on
every operand.
- A pointer or reference is `:direct` and needs no layout.
- A record found in `struct_definitions` (`_fuzzy_struct_lookup`) is
  `:layout`.
- Anything else falls back to `record_abi`, giving `:indirect`, `:ignore` or
  `:opaque`.
- `:opaque` and `conflicted_within`'s `:conflict` both trap, and no thunk is
  emitted.
- `sysv_record_abi` classifies a record's members. A member it cannot classify
  is an `:unknown` leaf, which traps a record of 16 bytes or less on SysV.

**Bodies.** `StructGen._member_mlir_type` spells members. A record member with
no table entry becomes `!llvm.array<N x i8>` of the member's DWARF size.

**RAII.** `JLCSIRGenerator._collect_class_raii` takes destructor and
copy-constructor symbols from **this** library's `functions`.

**vcall.** The slot comes from this binary's `parse_vtables`
(`DW_AT_vtable_elem_location`), and the vptr is loaded from the object at run
time. `VtableInfo.classes` is keyed by bare name.

**AOT.** `ThunkBuilder.build_aot_thunks` links `lib<name>_thunks` against this
library (`-l:`) and against libJLCS by absolute path. On PE every thunk symbol
must resolve against those two at link time.

### 2.5 Registry and packaging

- **`register` / vendoring.** `register` calls `_build_and_vendor!`.
  - `_store_build` copies the output directory into `builds/<hash_config>/`.
  - `_publish!` and `_write_package` write `packages/<Module>/`.
  - Its `Project.toml` `[deps]` lists only `Libdl` and `RepliBuild`.
  - Its UUID is `_package_uuid`, a uuid5 of the name.
  - Its entry file is `Base.include(Base.__toplevel__, "<builds/hash>/<Module>.jl")`.
- **`use(name)`.** `use` calls `_load_wrapper`, which runs
  `Base.include(Module(), wrapper_path)`: a **fresh anonymous parent on every
  call**. Even one library `use`d twice gives two unrelated sets of types.
- **No dependencies between entries.** Nothing lets one registered package
  depend on another.
- **`hash_config` inputs.** It covers the TOML, the sources, the include-dir
  headers, git HEAD, `_generator_fingerprint()` and `_toolchain_identity`. It
  does not cover link inputs, so a rebuilt A does not move B's hash. (It also
  feeds `SHA.update!` directly, not through `_key_feed!` as CLAUDE.md's caching
  section says. Not part of this design, but any input this design adds should
  go through whichever feed that section settles on.)
- **How B links A today.** Through `[link] link_libraries` + `link_dirs`.
  - ELF: `create_library` emits `-L`, `-Wl,-rpath` and `$ORIGIN`.
  - PE: `__init__` dlopens each DLL that `_link_dir_dlls` finds, through
    `_link_dir_preload_snippet`.

### 2.6 What breaks when B's DWARF holds A's types

Checked with Julia 1.13.3 on two hand-written modules shaped like two wrappers.
Each declares its own `X` and `@enum Color`, and A has a `ManagedX` with
`unsafe_convert(::Type{Ptr{A.X}}, …)`.

| From A | Into a B function whose ccall type is | Today |
|---|---|---|
| `Ptr{A.X}` | `Ptr{B.X}` (Julia param `::Any`) | **works**: Base converts between pointer types |
| `A.ManagedX` | `Ptr{B.X}` | `MethodError: no method matching unsafe_convert(::Type{Ptr{B.X}}, ::A.ManagedX)` |
| `A.X` by value | `B.X` | `MethodError: Cannot convert A.X to B.X` |
| `A.Color` | `B.Color` | `MethodError: Cannot convert A.Color to B.Color` |
| idiomatic class wrapper `A.X` | `Ptr{B.X}` | same as `Managed`: its `unsafe_convert` methods are keyed on `Ptr{Cvoid}` and `Ptr{A.X}` |

The layout side is worse than the identity side, because of 2.2:

- **Empty placeholders.** B emits `struct TObject end` (no fields) for every A
  class it only declares.
- **Missing base fields.** B's classes deriving from A's lose the inherited
  fields (`flatten_struct_members`), and B's upcast helpers stop at A's
  boundary.
- **Traps.** B's Tier-2 crossings of an A class by value trap: there is no
  definition, so the crossing is `:opaque`.
- **Duplicated PODs and enums.** B emits A's header-only PODs and enums again,
  with its own `STRUCT_SIZES` entries and its own `@enum`.
- **Duplicated inline functions.** An inline function from A's headers, with
  default visibility, is emitted weak in every library that uses it, and on ELF
  it is exported from each. It is then wrapped twice. Its thunks, from two
  metadata files, share one name, so `_lookup_cached` serves whichever engine
  initialised first to both wrappers.
- **Dispatch.** User methods on `A.X` never see `B.X`.

---

## 3. Requirements any design must keep

These come from CLAUDE.md. Each is cited by the invariant it protects.

- **DWARF of the real compile is the source of truth.** No design may invent a
  layout. A type's layout comes from the metadata of a library that has its
  full definition.
- **Emitted, not intended.** Generated tables and guards are derived from the
  final chunks. A cross-library reference must be checked against what the
  other wrapper *emitted*, not against what its metadata *defines*.
  - `surface_types_only`, `exclude_symbols`, the STL filters, the
    `best_dwarf_key` merge and enum aliasing all make those two differ.
- **Only ccall type positions resolve eagerly.**
  `_assert_wrapper_loadable` must keep passing on B's text. `_collect_defined!`
  descends `:const`, so `const X = ..A.X` counts as a definition.
- **Two generators, one answer.** Every derivation has to work for C and for
  C++, with the shared part in `Wrapper/Utils.jl`. A graph can mix a C library
  with a C++ one: the two sanitizers (`_sanitize_c_type_name`,
  `_sanitize_cpp_type_name`) spell some names differently.
- **Determinism.**
  - No emitted text follows Dict or Set order.
  - Picking a winner iterates sorted keys, or `topological_order`, which breaks
    ties by sorted id.
  - Persisted JSON is written sorted.
  - Every persisted key goes through `_key_feed!`.
- **Precompile.** Wrappers precompile inside consumer packages: process side
  effects stay in `__init__`.
- **PE.**
  - Each DLL binds only what its own export directory lists.
  - No RUNPATH, so load order matters.
  - One libJLCS per process, never vendored twice.
  - `libc_free` is bound per DLL through `_pe_crt_dll`.
- **Bare-name keys stay.** Qualified keys are a cross-cutting change (CLAUDE.md
  "Distinct types sharing a bare name"). A design may add facts beside a key,
  not change the key.

---

## 4. Deciding who owns a type

Options 1 and 3 both need an answer to "which library's wrapper defines K?".
There are three candidate signals.

**1. `decl_file` provenance** (the library whose sources or headers declare
K). `decl_file_path` is already parsed. As the rule it fails four ways:

- `BuildGraph.IncludeDir` holds the include path **with inherited
  directories** and no owner. Only `GraphSource` entries with `group == 0`
  (headers a target lists) attribute a header to a target, and many build
  systems do not list headers.
- ROOT copies every public header into one `include/` directory of the build
  tree before compiling. To confirm against root-recon. If so, every ROOT
  type's `decl_file` names the same directory.
- Probe row `Color`: the declaring library need not contain the type at all,
  so the "owner" by header would have nothing to emit.
- A path rule is host-shaped: `/` vs `\`, case folding, symlinks.

Useful later as a tie-breaker and for the report (section 6.9), not as the
rule.

**2. Exported symbols** (the library exporting K's methods, vtable `_ZTV`,
typeinfo `_ZTI`).
- **Strong for:** a dynamic class with a key function (exactly one library
  defines its vtable strongly), and a class with out-of-line methods. ROOT
  classes declare `Streamer` out of line through `ClassDef` and define it in
  the library's dictionary, so in practice their vtable lives in one library.
- **No signal for PODs and enums:** they have no symbols.
- **Misleading for inline members:** inline members and the vtables of classes
  with no key function are emitted weak in every user, and on ELF each user
  exports them.
- **No weak/strong distinction on PE:** the export directory does not
  distinguish weak from strong.

Useful as a **guard**: one member of K defined strongly in two libraries means
both claim K. The metadata already has what that needs:
`extract_symbols_from_binary` keeps each symbol's `nm` letter in
`symbols[].type`, and `T` is a strong definition on ELF. Not usable as the
rule.

**3. The graph plus what each wrapper emitted.** Wrap the libraries in
`topological_order`. A library owns K when no earlier library's wrapper emitted
K as its own, and the identity check below passes.
- Homed classes are covered, because only the owner has the definition.
- Header-only types are covered, because the dependency is wrapped first and
  wins.
- It fails only for **siblings**: two libraries, neither depending on the other,
  that both use a header-only type no common dependency uses. Each then keeps
  its own copy, as today, and the wrap reports them.

**Identity check** (when is B's K the same type as A's K?). The bare key alone
is not enough: `a::Config` and `b::Config` share it.
- The two entries must share a **scope**. This is a new metadata fact, the
  enclosing namespace or class chain of the DIE.
- When both sides have a layout, they must have equal `_layout_signature`s.
- **Same scope, different layout** means the two libraries were compiled with
  different views of one type: different defines, packing or flags, an ODR
  violation between them. That is a wrap error naming both libraries, not
  something to degrade around.
- **Different scope** means two types. B keeps its own and the wrap reports
  the name.

Signal 3 is the rule, with the identity check, and with signal 2 as a
guard.

---

## 5. Options

### Option 1 — B imports A's wrapper module and does not emit A's types

#### Module and package dependencies

B's text names A's module. There are two ways to make that name resolve:

- **(a) Per-library packages.** B does `import A`, so A must be a package in
  the environment and B's `Project.toml` must list A's UUID.
  - Then `use()` must load through `Base.require`, not
    `Base.include(Module(), …)`. Otherwise `use("A")` creates a second copy of
    A's types beside the one `import A` loaded.
  - A reference to a type whose owner is **not** in B's link closure (Tree →
    Hist's `TH1`, if libTree does not link libHist) cannot be an import: it can
    form a cycle. Core's headers forward-declare classes from libraries that
    depend on Core.
- **(b) One graph module.** One generated module per graph with one submodule
  per library, included in `topological_order`, and an import is
  `const X = ..A.X`.
  - A later submodule can name any earlier one, so a type owned by any earlier
    library imports without a link dependency and without a cycle.
  - One package, one UUID, one precompile unit.

#### Registry

- **With (b):** one entry vendors the graph's outputs (every library, every
  wrapper file, the graph module) to `builds/<hash>/`, and `use(name)` returns
  the graph module. "A project builds where it lives" still holds.
- **With (a):** the registry needs dependencies between entries, which it does
  not have.

#### Virtual dispatch and `Managed`

- **vcall already crosses libraries.** It loads the vptr from the object at run
  time, and Itanium keeps a base's slots at the start of every derived vtable.
  So A's thunk for `TObject::Draw`, given a Hist object upcast to `TObject*`,
  dispatches to Hist's override. What is needed is a correct `this`.
  - Single inheritance needs no adjustment.
  - Under multiple inheritance, B's `<Derived>_as_<ABase>` helpers need A's
    base layouts. B then reads them from A's metadata.
- **`Managed<K>` is emitted once, by K's owner.** Its finalizer calls the
  owner's library, which on PE is the only DLL that exports the deleter.
- **One rule widens to the graph.** `has_subclass` ("nothing derives from K")
  only sees K's own library, and a dependent library deriving from K breaks the
  D0 factory struct the same way.

#### Layout conflicts

- **Per library:** unchanged (`conflict`, `conflict_sizes`).
- **Across libraries:** the identity check decides, and a cross-library
  mismatch never silently picks one layout.
- **Import and conflict:** an imported entry carries the owner's `conflict`
  marks, so `conflicted_within` traps a by-value crossing in B exactly as it
  would in A.

#### Tier 2 thunks naming another library's record

B's thunks need A's definitions:
- for A's records crossing by value;
- for B's records that embed A's records (`StructGen` bodies);
- for `sysv_record_abi` and `record_abi`;
- for A's RAII symbols (`_collect_class_raii`).

JITManager and ThunkBuilder read only B's directory, so B's wrap has to
persist the definitions it took from owners for them (6.4). Not re-wrapping A's
inline functions in B also removes the shared thunk names.

#### Windows PE

- **B's DLL binds only B's functions.** B.dll's export directory is unchanged.
- **Load order is load-bearing.** Submodule `__init__`s run in include order,
  so A.dll is loaded by absolute path before B's `__init__` loads B.dll. B.dll's
  import of A.dll then resolves to the already-loaded module. PE has no
  RUNPATH.
- **libJLCS stays one copy.** Every wrapper opens it through
  `MLIRNative.libJLCS`.
- **AOT link.** If one of B's thunks names an A symbol (the copy constructor or
  destructor of an imported class), B's AOT thunks DLL must also link A's
  import library. ELF binds it at dlopen, and the JIT engine resolves it
  process-wide.
- **One CRT.** A graph is compiled with one toolchain, so every DLL uses one
  CRT and each wrapper keeps its own `libc_free`.

#### Determinism

- The owner is chosen by topological order, never by a Dict.
- Import lines are sorted, and the manifests are written sorted.
- B's text now depends on A's emitted types, so B's identity (project hash,
  `hash_config`, the build-identity check) must include them.

**Cost:** a pre-pass, an emitted-type manifest, a skip predicate in both
generators, and a sidecar the thunk side reads.

### Option 2 — one Julia module that binds symbols from several libraries

- **Module and package dependencies:** none, which is the attraction. One type
  per name for free.
- **Registry:** one entry, and one wrapper file for the whole graph.
  - Any library's change regenerates and re-precompiles all of it.
  - `_assert_wrapper_loadable` refusing one name refuses the graph.
  - ROOT is on the order of 10^5 functions in one file.
  - Nothing can be loaded partially.
- **Virtual dispatch and `Managed`:** natural. Each `Managed` deleter calls the
  library that defines it, so `LIBRARY_PATH` becomes a per-library constant on
  every ccall.
- **Layout conflicts:** every library's types share one bare-name namespace.
  - `_canonical_definition` would run across libraries.
  - A name that two libraries use for different types (pimpls, nested
    `Config`/`impl`) becomes a `conflict` whose by-value crossings trap across
    the whole graph.
  - Function names collide too: `_julia_function_name` is per symbol, and
    `_dedup_method_chunks` would silently keep one of two same-signature
    methods from different libraries.
- **Tier 2 thunks:** JITManager is per binary. The thunks of one module span
  engines, and each engine's `generate_jlcs_ir` still needs records from the
  other libraries' metadata: the same merge Option 1 needs.
- **Windows PE:** `_runtime_defined_names` and `_gate_to_exports` must run per
  library and be kept per function. This is workable.
- **Determinism:** the merge order must be fixed (topological). This is
  workable.

**Verdict:** right for two or three tightly coupled libraries, wrong for a
graph.

### Option 3 — a shared types module for the whole graph

- **Module and package dependencies:** one `<Graph>Types` module holds every
  record and enum of every library, and each library wrapper imports it and
  emits functions only.
- **Registry:**
  - The types module depends on every library's metadata, so any rebuild
    changes it and invalidates every wrapper's precompile cache.
  - It also has to be loadable on its own.
- **Virtual dispatch and `Managed`:**
  - `Managed`, factory structs, `<f>_safe` and upcasts need a library (the
    deleter's symbol), so either the types module binds every library, which
    is Option 2 for those, or they stay per library.
  - Since the types module holds every type, ownership is still needed for
    exactly those.
- **Layout conflicts:** same as Option 2, at graph scale.
- **Tier 2 thunks:** natural, because the merged table is the types module's
  metadata.
- **Windows PE:** fine if the module binds no library. Not so if `Managed`
  lives there.
- **Determinism:**
  - `_canonical_definition` across libraries changes which definition a name
    keeps whenever a library is added.
  - A mixed C/C++ graph must emit every type with **one** generator's rules.
    GeneratorC's `_resolve_exact_layout` and GeneratorCpp's
    `_prove_julia_layout` differ deliberately ("Two generators, one answer"
    keeps them isolated).

**Verdict:** no. It concentrates every cross-library conflict in one place, and
still needs ownership for everything that binds a symbol.

### Comparison

| | Option 1 (import from owner) | Option 2 (one module) | Option 3 (types module) |
|---|---|---|---|
| One Julia type per C++ type | yes, where an owner exists; siblings reported | yes | yes |
| Bare-name collisions across libraries | only between a library and its owners, checked | graph-wide | graph-wide |
| Unit of regeneration | one library and its dependents | whole graph | whole graph |
| Mixed C/C++ graph | each library keeps its generator | one generator for the whole module | one generator for all types |
| `Managed`/finalizers | owner's library | per-ccall library | needs ownership anyway |
| Tier 2 | sidecar with owners' definitions | same merge | merged table |
| New pieces | manifest, pre-pass, skip predicate, sidecar | per-function library, cross-library dedup | types module, ownership for bindings |

---

## 6. Recommendation: Option 1, delivered as one graph module

### 6.1 The ownership rule

Graph mode builds every library first, then wraps them one by one in
`BuildGraph.topological_order`, over the targets RepliBuild wraps (`:shared`
and `:module`). Static and object targets are folded into the libraries that
link them and own nothing. For library B, before either generator runs:

1. **Candidates.** The candidates are every key in B's `struct_definitions`
   (records and `__enum__` enums), plus every record name B's signatures and
   members reach without a definition. The second set is the one GeneratorCpp
   turns into forward declarations today. It is computed by one shared function
   both generators use.
2. **Look up an owner.** For each candidate K, in sorted order, look K up in
   the type manifests (6.4) of the libraries wrapped before B, walked in
   topological order. Wrapped libraries own disjoint sets, so at most one
   matches, except for siblings. Among siblings the first in topological order
   is taken, and the rest are reported.
3. **Check identity.** The match must pass the identity check (section 4):
   - the scopes intersect;
   - the layout signatures are equal when B also has a layout;
   - the guard holds: no member of K is defined strongly (`symbols[].type ==
     "T"`) in both B and the owner. On PE the guard does not run, because a
     COMDAT copy and a definition look alike there. The live fixture (7.2)
     measures whether that holds.

   The outcome is one of four:
   - identity holds: K is **imported**;
   - scope mismatch: K is **own**, reported;
   - same scope with a layout mismatch: a **wrap error** naming both libraries
     and the differing signature;
   - guard failure: a **wrap error**, because two libraries claim to define K.
4. **No match:** K is **own** if B has its definition. Otherwise it stays a
   **placeholder**, which is today's forward declaration (6.6).
5. **Record the result.** B's own types go into B's type manifest for the
   libraries after it.

Every step iterates sorted keys or the topological order. A single-library
build has no earlier manifests, so every K is own and the wrapper is
byte-identical to today's.

### 6.2 The graph module

`wrap` in graph mode writes `julia/<Graph>.jl`:

```julia
module ROOT                     # [project] name, through module_name_for
include("RCore.jl")             # one file per library, unchanged in shape
include("Hist.jl")              # module Hist … const TObject = ..RCore.TObject …
include("Tree.jl")
end
```

- **Per-library files.** Each file keeps today's shape: its own `LIBRARY_PATH`,
  `LIBRARY_SHA`, `__init__`, guards and introspection tables. A per-library
  file whose imports name a sibling loads only through the graph module.
- **Load order.** `__init__`s run in include order, which is the PE load order
  that Option 1 needs (section 5, "Windows PE").
- **Module names.** Each library's module name is derived once, through
  `module_name_for`, and a name bound in `Base` or `Core` is renamed
  deterministically (`Core` → `RCore`). ROOT's first library is `Core`.
  - Julia accepts a submodule named `Core`: checked on 1.13.3, and `..Core`
    imports resolve to it.
  - But such a submodule shadows Julia's `Core` in the graph module's namespace.
- **Registry.** The registry vendors the graph as one unit, and `use(name)`
  returns the graph module.
  - `hash_config` gains every library's inputs, since one entry now covers all
    of them.
  - That belongs to the graph-mode task, not to the type work.
- **Load cost.** Loading the graph module loads every wrapped library. A graph
  wraps the targets the TOML selects plus their dependency closure, which keeps
  ROOT's load set to what the user asked for.

### 6.3 Metadata changes (minimal)

Both are additions beside the existing keys. Neither changes a key, and
neither moves which definition `_canonical_definition` keeps.

1. **`scopes` on every record and enum entry.** It holds the sorted, unique
   enclosing namespace/class chains (`"TMath"`, `"ROOT::Experimental"`, `""`
   for global) of all the definitions merged under the key. It is collected in
   `parse_dwarf_dump` from the DIE's parent chain, and added after
   `_canonical_definition` picks the kept entry, so `_canonical_text`
   comparisons are unchanged.
   - It is needed for the identity check.
   - It takes a rebuild to appear: delete `.replibuild_cache/project_hash`.
   - Metadata without it fails safe. A missing `scopes` on either side means
     "cannot prove identity": the type stays own and is reported.
2. **`decl_files` on the same entries.** It holds the sorted, unique
   `decl_file_path`s, normalised with `_posix`.
   - It is not used by the rule in 6.1.
   - It is recorded now so the sibling report can name the header, and so
     hoisting (6.9) needs no second rebuild.

Nothing else in `compilation_metadata.json` changes. Imports are a wrap-time
decision and live in wrap artifacts.

### 6.4 New wrap artifacts

Both are written sorted, with a `version` key, like `_write_thunk_manifest`.

**`type_manifest.json`**, written by every wrap beside the library: what this
wrapper **emitted** as its own. It is derived from the final chunks, like
`DISPATCH_TIER`. Per type key it holds:

- `julia_name`, as this wrapper's sanitizer spelled it;
- `kind`: `struct`, `mutable`, `blob`, `enum` or `placeholder`;
- `byte_size` and `scopes`;
- the layout facts a dependent's layout proof needs to embed it: GeneratorCpp's
  `jl_layouts`/`jl_mutable` entry, and GeneratorC's resolved layout;
- the owner's `Managed`/idiomatic names, if emitted;
- the sha256 of the metadata entry the type was emitted from.

The manifest also lists the mangled symbols the wrapper binds, which is what
6.5 item 6 needs. The importer takes `julia_name` from here and never re-derives
it, so a C owner and a C++ importer agree.

**`type_imports.json`**, written by B's wrap beside B's library: for each
imported key, the owner's module, its `julia_name`, and the owner's full
metadata entry. It carries:
- the `struct_definitions` entry with its `conflict` marks;
- the `record_abi` facts;
- the owner's RAII symbols for the class.

`JITManager.initialize_global_jit` and `ThunkBuilder.build_aot_thunks` merge it
into the metadata they hand to `generate_jlcs_ir`, through **one** helper in
`IRGen`. The owner's entry replaces B's entry for the key, and is added where B
had none. Without the file, nothing changes.

### 6.5 Generator changes

1. **One decision function and one pre-pass.**
   - `Wrapper/Utils.jl` gains `_resolve_type_imports(metadata, earlier)`,
     implementing 6.1, and `_apply_type_imports!(metadata, imports)`.
   - The pre-pass marks each imported entry
     `"imported" => Dict("module" => …, "name" => …)`, replaces its body with
     the owner's, and adds owner entries for referenced-only names.
   - It runs in `wrap_introspective` after `_apply_surface_type_filter!`, so a
     surface-filtered library imports only what it reaches.
2. **One emission predicate, `_emits_type(key, info)`,** shared by both
   generators. Every loop that defines something per type asks it:
   - forward declarations;
   - struct definitions;
   - enums, including header enums whose name an owner emits;
   - the `Managed`/deleter scan and `<f>_safe`;
   - `idiomatic_classes` and union accessors;
   - setters and getters;
   - `_layout_chunk`, so `STRUCT_SIZES` lists own types only;
   - upcasts whose derived class is imported.
   
   An imported key emits one line instead,
   `const <B's spelling> = ..<Owner>.<julia_name>`. Imports are emitted first,
   before enums, forward declarations and struct definitions (any of which may
   name an imported type), sorted by (owner, name).
3. **Imported entries stay visible to every reader.** Imported entries remain
   in `struct_definitions`, so everything that *reads* layouts sees the owner's
   definition without a second lookup path:
   - `flatten_struct_members`, which now gets A's base fields;
   - `_collect_upcasts!`, `_prove_julia_layout` and `_resolve_exact_layout`;
   - `is_ccall_safe` and `_c_byvalue_verdict`;
   - `_resolve_forward_ptr`'s emitted set.

   An embedded imported record takes the owner's `jl_layouts` entry from the
   manifest. A `mutable` owner type makes the embedding record a blob, exactly
   as a local one would.
4. **A write-time guard, `_assert_no_imported_redefinition`.** It runs in
   `_assert_wrapper_loadable`'s sequence and checks B's **text**: no `struct`,
   `mutable struct`, `@enum` or `Managed…` definition for any imported name.
   A loop that forgets the predicate is a refused wrap, not a load-time
   "invalid redefinition of constant" in a consumer.
5. **Graph-wide `has_subclass`.** Before A's wrap, it adds the base edges of
   every graph library that depends on A (their metadata exists, because graph
   mode builds first). So a D0 factory struct is emitted only for a class
   nothing **in the graph** derives from.
6. **Functions an earlier wrapper binds.**
   - B drops a function whose mangled symbol an earlier library's manifest
     lists as bound: an inline member or free function of A's headers, emitted
     weak in B.
   - The wrap prints a count line, like the filters'.
   - This keeps one Julia function per C++ function, and removes the
     same-name thunks behind `_lookup_cached`'s first-engine-wins.
7. **`<f>_safe` stays local.** B emits `<f>_safe` only for types whose deleter
   B itself owns.
   - A pointer that B returns to an imported class gets no automatic
     finalizer: whether B transfers ownership is not in DWARF, and ROOT getters
     (`TTree::GetHistogram`) return borrowed pointers.
   - The user wraps it in `A.ManagedK` deliberately.
8. **Identity across rebuilds.**
   - B's project hash and `hash_config` include the sha256 of every manifest B
     imported from, so a rebuilt A re-wraps B.
   - B's wrapper gains `IMPORTED_FROM = Dict(:RCore => "<manifest sha>")`, and
     `_check_build_identity` warns when an owner's loaded manifest differs.
9. **PE AOT link.** `ThunkBuilder.build_aot_thunks` adds an owner's import
   library to the link only when B's thunk module names one of the owner's
   symbols. `_assert_aot_thunks_present` is extended to refuse a thunks library
   with an undefined reference outside B, libJLCS and the CRT.

### 6.6 A reference whose owner comes later

When B references K only by declaration, and K's owner is wrapped **after**
B, B keeps today's placeholder `struct K end`. This happens when there is no
link dependency and the sorted-id tie-break puts the owner later. The owner,
which can see B, then adds:

```julia
Base.unsafe_convert(::Type{Ptr{..Tree.TH1}}, o::ManagedTH1) = Ptr{..Tree.TH1}(o.handle)
```

It does the same for its idiomatic wrapper, for every earlier placeholder of
an owned type.
- The owner learns of the placeholders from the earlier manifests
  (`kind = "placeholder"`).
- Raw pointers already convert (2.6).
- No by-value use is possible through a placeholder, today or after.
- User code that dispatches on `Ptr{Hist.TH1}` does not match `Ptr{Tree.TH1}`.
  That is documented, not fixed.

### 6.7 What does not change

- **Single-library builds:** they write a `type_manifest.json` and nothing reads
  it, so their wrappers stay byte-identical. That is the regression diff.
- **Tier 3 call bodies:** a ccall still calls B's own library, and pointer
  parameters stay `::Any`.
- **Tier 1:** it is quarantined, C only, and declares symbols per library;
  imports do not touch slices.
- **Per-library conflicts:** `conflict`/`conflict_sizes` and
  `conflicted_within` are unchanged.

### 6.8 How the recommendation meets each concern

- **Julia modules:** submodules of one graph module, with relative `const`
  imports. No new package dependencies.
- **Registry:** one entry per graph. `use()` returns the graph module, so the
  fresh `Module()` parent no longer splits types within a graph.
- **Precompile:** the graph module precompiles as one unit inside a consumer,
  and side effects stay in each submodule's `__init__`.
- **Virtual dispatch:** through the owner's thunks with an upcast `this`. B's
  upcasts reach A's bases through the owner's layout.
- **`Managed`:** emitted only by the owner, the D0 rule is graph-wide, and
  `<f>_safe` stays local.
- **Conflicts:** an imported entry carries the owner's marks. A cross-library
  mismatch is reported or refused, never silently merged.
- **Tier 2:** the `type_imports.json` sidecar feeds `generate_jlcs_ir`,
  `sysv_record_abi`, `record_abi` and RAII with the owner's facts, and the
  duplicate thunk names go away.
- **Windows PE:** load order comes from the include order, there is one
  libJLCS, the AOT import-library link is added only when needed, and every
  wrapper keeps its own `libc_free`.
- **Determinism:** ownership follows the topological order, imports and
  manifests are sorted, and manifest digests are part of B's identity.

### 6.9 Later, not in the first implementation

- **Hoisting sibling-only types.** A header-only type used by siblings but by
  no common dependency stays duplicated in v1, and is reported with its
  `decl_files`. Phase 2 would emit it in the lowest common dependency whose
  target lists that header (`GraphSource` with `group == 0`). That library's
  wrapper then emits a type taken from another library's metadata.
- **Cross-project imports** (Hub packages linking each other through
  `link_dirs`). The same manifests work with per-library packages, but need
  registry dependencies and `use()` loading through `Base.require`.
- **Virtual delete.** A `Managed<K>` finalizer that calls D0 through the vtable
  would be right for any dynamic type. Today it calls K's D0 by symbol, which
  is right only when the object's dynamic type is K. ROOT's
  `TClass::New`-made objects are where that matters.

---

## 7. Test plan

### 7.1 `runtests` — synthetic metadata, no toolchain

These tests go in a new `test/test_cross_library_types.jl`, wired into
`runtests.jl`. The fixtures are hand-built metadata dicts and manifests for
three libraries: `XCore` → `XHist` → `XTree`, with `XTree` also referencing an
`XHist` type by declaration only.

1. **Ownership.**
   - `TObject` (full in `XCore`, a declaration in `XHist`) gives
     `XHist: imported from XCore`.
   - `Plain` (full in both, same scope and layout) gives imported.
   - `a::Config` vs `b::Config` gives own plus a report entry.
   - Same scope with different `_layout_signature` gives an error naming both
     libraries.
   - Missing `scopes` gives own plus a report.
   - Sibling-only `Color` stays own in each and is reported.
   - An owner that comes later gives a placeholder, and the owner's text
     contains the `unsafe_convert` for it.
   - Assert the number of decisions checked equals the number expected.
2. **Emission.** Run both `generate_introspective_module_cpp` and
   `generate_introspective_module_c` on `XHist`'s metadata plus `XCore`'s
   manifest, and assert:
   - `const TObject = ..XCore.TObject` is present;
   - there is no `struct TObject`, `ManagedTObject` or `STRUCT_SIZES` entry for
     `TObject`;
   - a derived record's fields include the flattened base fields;
   - `_assert_wrapper_loadable` passes.

   `_assert_no_imported_redefinition` refuses a hand-written text that
   redefines an imported name. A C owner with a C++ importer takes `julia_name`
   from the manifest.
3. **Determinism.** Build the inputs with shuffled Dict insertion orders and
   shuffled manifest key orders, in the style of
   `test_emission_determinism.jl`. The decisions, the import block and both
   manifests must be byte-identical.
4. **Thunk side.** `generate_jlcs_ir` on `XHist`'s metadata merged with
   `type_imports.json` emits a thunk where unmerged metadata trapped `:opaque`.
   This is a by-value `Plain`, plus a record embedding `TObject`. With no
   sidecar the MLIR text is byte-identical to today's.
5. **No-op.** For a single library, metadata in gives a byte-identical wrapper
   out against the current generator, for one C and one C++ fixture metadata
   file. This is the guard that the change is inert outside graph mode.

### 7.2 Live two-library fixture — devtests, needs clang (and libJLCS for Tier 2)

The fixture is `test/xlib_test/`, a library `xa` and a library `xb` that
links it. Its test file, `test/test_cross_library_types_live.jl`, is wired into
`devtests.jl`. It is driven through graph mode once that lands, and through two
tomls with `[link] link_dirs` until then. The fixture shape:

- **`xa`:**
  - `class Base` with an out-of-line constructor, a virtual `get()`, and a
    virtual destructor (key function in `xa.cpp`);
  - a POD `Plain`;
  - an enum `Color` that `xa.cpp` uses;
  - `Base* create_base()`, `void destroy_base(Base*)`;
  - `Plain make_plain()`.
- **`xb`:**
  - `class Derived : public Base` overriding `get()`;
  - `int use(Base*, Plain, Color)`;
  - `Plain twice(Plain)`, a C++ non-noexcept method taking `Plain` by value,
    which goes to Tier 2;
  - `Base* create_derived()`;
  - a struct `Holder { Plain p; Base* b; }` returned by value.

Assertions:

- **Identity:** `XB.Base === XA.Base`, `XB.Plain === XA.Plain`,
  `XB.Color === XA.Color`.
- **Crossings:** `XA.ManagedBase(XA.create_base())` passes to `XB.use`; a
  `Plain` from `XA.make_plain()` passes by value to `XB.use` and to Tier-2
  `XB.twice`; `XA.Color` passes too.
- **Virtual dispatch:** `XA`'s `Base_get` on `XB.create_derived()` returns
  `Derived`'s value.
- **Layout:** `fieldnames(XB.Derived)` includes `Base`'s fields, and `XB` has
  no `STRUCT_SIZES[:Base]`.
- **Determinism:** wrapping twice is byte-identical.
- **Precompile:** a temp consumer package (Pkg offline, `REPLIBUILD_HOME` a
  temp dir, `REPLIBUILD_JULIA_ENV=none`) loads the graph module.
- **PE:** the same file runs on the Windows host. It asserts that `xa.dll` is
  loaded before `xb.dll`, and that `xb`'s AOT thunks library has no undefined
  reference outside `xb`, libJLCS and the CRT.

**Not planned:** a Hub rebuild. This change touches no Hub package's calling
path; the single-library no-op test (7.1 item 5) stands in for it.

---

## 8. Out of scope: ROOT-specific pieces

- **Dictionaries and `TClass`.** `rootcling` output (`G__*.cxx`) is compiled
  into each library and wrapped as ordinary C++ like any other source. Its
  reflection (`TClass::GetClass`, `IsA`, `Streamer`, schema evolution, the
  `.pcm` modules) is not used for types: DWARF stays the only type source.
- **cling.** No interpreter, no `gInterpreter`, no on-the-fly instantiation. A
  template instantiation exists for Julia only if some library compiled it.
- **Autoloading and the plugin manager.** `rootmap` files and
  `TROOT::LoadClass` are not consulted. The graph module loads what the graph
  wraps.
- **I/O of user classes.** Reading a `TFile` holding a class needs that class's
  dictionary at run time, which is ROOT's business.
- **ROOT object ownership.** By default a histogram belongs to the current
  directory (`TH1::AddDirectory`), and `TObject` participates in ROOT's
  cleanup lists. A `Managed<TH1>` finalizer deleting a directory-owned
  histogram is a double free. DWARF does not describe ROOT's ownership, so any
  automatic finalizer for `TObject`-derived classes is a ROOT-layer decision
  above this design.
- **ROOT on Windows.** RepliBuild's Windows target is MSYS2 CLANG64, and ROOT
  supports MSVC there. The two-library fixture covers PE for the design itself.

---

## 9. Decisions for John

1. **Delivery.** The graph module with submodules (recommended), or
   per-library packages with registry dependencies. The second loads
   libraries one at a time and composes across projects, but needs `use()`
   rewritten and cannot import from a non-dependency.
2. **Reference imports.** Recommended: let B import a type whose owner is an
   earlier library that is **not** in B's link closure. This is the Tree →
   Hist `TH1` case. It costs B's text depending on unrelated libraries' order.
3. **Same scope, different layout.** Recommended: a wrap error. The
   alternative is own-plus-report, letting B keep its own type.
4. **Sibling-only types.** Recommended: report them in v1 and hoist in
   phase 2. The alternative is to hoist now.
