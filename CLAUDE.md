# CLAUDE.md

Guidance for Claude Code working in this repository. This file is loaded into every
session, so it holds **only what changes how you work here**: commands, architecture,
invariants that must not be broken, and open problems. History — measurements, blast
radii, dated sweeps, how a bug was found — belongs in `CHANGELOG.md` and git, not here.
The pre-2026-09-22 long-form version of this file is in git history if a rationale is
ever needed (`git log -p -- CLAUDE.md`).

## Project Overview

RepliBuild.jl is an ABI-aware C/C++ compiler bridge for Julia. It compiles C/C++
through LLVM/MLIR, introspects DWARF metadata, and emits type-safe Julia bindings.
Each function is routed to one of three tiers based on ABI complexity:

| Tier | Mechanism | When |
|------|-----------|------|
| 1 | `Base.llvmcall` on bitcode | POD args, scalar/pointer return — **experimental, off by default** |
| 2 | MLIR thunks via the JLCS dialect (`libJLCS.so`) | C++ methods, by-value aggregates, packed structs, unions, virtual dispatch, EH |
| 3 | `ccall` | Everything else; the default for C |

Positioning: a hands-on dev tool for wrapping libraries **from source**, not a
JLL/BinaryBuilder replacement. DWARF from the real compile is the source of truth —
RepliBuild operates on what Clang already resolved, not on a source AST.

## Build & Test Commands

```bash
julia --project=. test/runtests.jl      # CI tests, no C++ toolchain needed
julia --project=. test/devtests.jl      # full integration (C++ needs system LLVM 21+/MLIR, clang, cmake)
julia --project=. test/test_<name>.jl   # single file
cd src/mlir && ./build.sh               # build the JLCS dialect (Tier 2 only)
julia --project=. -e 'using RepliBuild; RepliBuild.check_environment()'
julia --project=docs docs/make.jl       # docs build
```

## API

```julia
toml = RepliBuild.discover("path/to/project")   # scan, write replibuild.toml
RepliBuild.build(toml)                          # clang → IR → .so + DWARF metadata
RepliBuild.wrap(toml)                           # generate julia/<Name>.jl
RepliBuild.discover("path", build=true, wrap=true)   # all three

RepliBuild.clean(toml); RepliBuild.info(toml)
RepliBuild.scaffold_package("LuaWrapper"); RepliBuild.check_environment()

RepliBuild.register("pkg/replibuild.toml"); M = RepliBuild.use("name")   # local registry only, no Hub fallback
RepliBuild.list_registry(); RepliBuild.unregister("name")
RepliBuild.search("xml")                        # lists Hub catalog; does NOT fetch — register then use

RepliBuild.VisibilityProbe.visibility_probe(toml)    # does this library annotate its exports?
RepliBuild.SysConfigGen.cmake_probe(...)             # capture cmake-generated headers (see below)
RepliBuild.Debug.walk(pkg, "<thunk>_thunk")          # static thunk inspection (see Debugging)
```

**Ingest mode (`RepliBuild.ingest("/path/libfoo.so", headers=[…])`) is experimental,
best-effort, and C only** (issue #4). It skips compilation and wraps a prebuilt `-g`
library via Tier 3 only. Do not re-inflate its claims in docs. `[ingest]
extra_link_libs` is dlopen'd `RTLD_GLOBAL` before the main library; prefer a full
soname over an `-l` name (glibc ≥ 2.34 merged `m`/`pthread`/`dl`/`rt` into libc).

The full TOML reference is `docs/src/config.md` — keep it authoritative when adding
keys. A new user-intent key must also be added to `Discovery.PRESERVED_TOML_KEYS`, or
`discover(force=true)` silently deletes it. Unknown TOML keys produce no warning.

## RepliBuild-Hub

Community registry of `replibuild.toml` configs:
[obsidianjulua/RepliBuild-Hub](https://github.com/obsidianjulua/RepliBuild-Hub),
cloned beside this repo. `packages/<name>/replibuild.toml` (git dependency on upstream,
pinned tag + `commit`), listed in `index.toml`. `REPLIBUILD_HUB_URL` overrides the
index location. `examples/BoxWorld/` is the reference consumer package. The `hub-wrap`
skill drives adding a package.

- **Dev loop: build + wrap in place, never `discover`.** Hub tomls are hand-rolled
  (varargs, macros, excludes) and discover would strip them.
- **`test.jl` REBUILDS, `test_deep.jl` verifies.** `test.jl` calls `RepliBuild.clean`,
  which deletes `build/`, `julia/` and `.replibuild_cache/` including the vendored
  clone, then rebuilds from cold (minutes, needs network). To check a wrapper works,
  use `test_deep.jl`. Read a test script before running it. (box2d/tinyxml2/pugixml
  `test.jl` are exceptions — verifiers only.)
- Julia block-buffers redirected stdout: an empty log from a running or killed job is
  normal. Judge progress from artifacts on disk and `pgrep`.
- **A fix computed at build time needs a rebuild, not a re-wrap.** `class`, signatures
  and DWARF types live in `compilation_metadata.json`. The project hash will skip the
  rebuild (`cache: project unchanged`), so delete `.replibuild_cache/project_hash` to
  force re-extraction. That keeps the clone and the per-file IR cache.
- **To date a fix into an artifact, grep the artifact for the fix's signature**, never
  the mtime.
- **Consumer compensation is a feature request.** A workaround hand-rolled in several
  Hub consumers is evidence of a missing generator feature. When a policy moves into
  the generator, grep the consumers for the workaround.

### SysConfigGen — generated headers (`src/Builder/SysConfigGen.jl`)

Unblocks libraries whose headers come from cmake's configure step (`config.h`,
`configure_file`'d public headers, generated `.c`). `cmake_probe` (configure only,
under `with_llvm_env`, `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` by default) →
`capture_config` (copies into `packages/<name>/config/`, writes `SYSCONFIG.md`) →
`toml_fragment`. Tests: `test/test_sysconfiggen.jl` (devtests §18). pcre2 is the
reference package and the byte-identical regression case.

- `compile_commands.json` is the evidence for flags, **per-target** uniformity (the
  same sources configured as shared + static is not divergence), the exclude list, and
  whether a generated `.c` is library source or `try_compile` scratch. With no
  `compile_commands.json` nothing is classified.
- Captured files land under the deepest build-tree `-I` root (`layout = :auto`), so
  `<sundials/sundials_config.h>` resolves. Two files mapping to one destination is a
  hard error.
- **Boundary:** configure-time generation works. Build-time custom commands (libpng's
  `pnglibconf.h`) do not; an empty `generated_headers` is the signal. Check for a
  shipped `.prebuilt` fallback first. Autotools is untested.
- **A wrong harvested config does not fail the build.** It produces a library that
  is not the library. Package tests must assert behaviour that depends on the probes
  (pcre2 checks `\p{L}` on UTF-8 and reads the version from the `.so`). Each package
  keeps a `harvest.jl` that pin-checks HEAD and errors if the target stopped being
  uniform.
- Every path inside the module is `/`-separated: normalise with `_posix` where a host
  path enters, and join with `"/"`. cmake always writes `/`, and Windows does not.

## Architecture

```
C/C++ source → DependencyResolver → Discovery → Compiler (clang → per-file IR)
  → link + opt (+ static promotion) → .so + bitcode → DWARF + symbol extraction
  → verify_wrap_surface → Wrapper generator (GeneratorC / GeneratorCpp)
  → JITManager / ThunkBuilder (Tier 2 thunks, JIT or AOT)
```

### Toolchains — two LLVMs, deliberately

- **C bucket needs no external LLVM.** `Clang_unified_jll` emits IR. Link, opt,
  static promotion and bitcode assembly run **in-process on Julia's own libLLVM**,
  which is version-matched to that clang, so there is no DWARF-dropping skew. A
  failure is a hard error. `[link] fallback = true` selects the external
  `llvm-link`/`opt`, accepted only at the same major as `Base.libllvm_version`.
  Final `.ll → .so` still shells to clang. Check `DW_AT_producer` (`…jl`) to prove a
  C artifact came from the JLL.
- **C++ / Tier 2 uses system LLVM+MLIR 21+** (22.1.x on the reference host) for the
  dialect and the external pipeline. `libJLCS.so` is a gitignored build artifact
  pinned to the installed MLIR minor SONAME: rebuild after a minor upgrade. A patch
  bump inside a minor needs nothing. `deps/build.jl` builds it at install time when
  possible and never throws. Tier 3 alone is the whole product for C.
- Julia's bundled libLLVM and system libLLVM coexist in one process. The GDB JIT
  descriptor resolves to Julia's copy; this is correct, do not "fix" it.
- For LLVM/JIT crashes, check `libJLCS.so`'s mtime/SONAME against the installed MLIR,
  and dual-LLVM symbol scope, before theorising.

### Tier 1 — quarantined side project

`Base.llvmcall` dispatch is **John's side project**: it ships and works, but it is not
a supported tier. `[wrap.tier1] enable` defaults false and every Hub config pins
`[link] enable_lto = false`, so no shipped package takes it. Its suites
(`test_static_promotion`, `test_slicer`, `test_tier1_dispatch`) are **deliberately
unwired** from devtests; `runtests.jl`'s wiring guard names them in an `experimental`
list. They also need `test/slice_test/replibuild.toml`, which is gitignored and
machine-local (John's call: document it, don't track it). **Do not un-quarantine,
re-wire or invest in it (M4/perf) unless John asks.**

If you are asked to work on it:
- Two independent payloads. `[link] enable_lto` (default true for C) embeds the whole
  linked module per call. It segfaults at library scale and duplicates file-local
  statics, so it is parked. `[wrap.tier1] enable` gives per-function
  declarations-only slices (`src/IRGen/Slicer.jl`) bound to the `.so` by symbol.
- All llvmcall emission lives in `src/Wrapper/C/Tier1C.jl`. Registry emission is keyed
  on **what was emitted**, not on the config flag. Acceptance is weaker than emission,
  and slice files and `TIER1_FUNCTIONS` are derived from the final chunks. A
  tier1-off wrapper contains no llvmcall machinery at all.
- **Static promotion** (M1, `_promote_statics_libllvm`) renames every function/global
  a slice may `declare` but that cannot reach dynsym (internal, or hidden/protected)
  to exported `__rb_<lib>_<name>`. `extract_symbols_from_binary` filters `__rb_*` out
  of the API. Only internal constants stay local. Embedding one in a slice requires
  `unnamed_addr`; lua's 18 refusals come back by widening M1 to promote
  address-significant internal constants (not done).
- **An unresolved `declare` does not raise. ORC blocks forever.** Every slice is
  pre-flighted with dlsym through the library **handle** (`RTLD_LOCAL`, then
  `dlclose`), never `RTLD_DEFAULT`. Wrappers re-check at load (`TIER1_DECLARES`) and
  demote to ccall on a miss.
- **llvmcall in a precompile worker deadlocks on the JIT lock** once it declares a
  dlopened symbol. Call sites are therefore `@generated` kernels that splice ccall
  when `jl_generating_output()` or when the slice file is missing. `dispatch_tier`
  returns `:deferred` in output mode. The doctrine: llvmcall is a passenger, any
  doubt resolves to ccall.
- Slice constants are keyed on the **mangled** symbol (`julia_name` is not
  injective). Slices are `include_dependency`s; Julia 1.11+ tracks those by content,
  so `touch` does not invalidate.

### Key modules

- `Builder/Compiler.jl` — build engine: per-file IR, caches, link/opt, promotion,
  DWARF parse (`parse_dwarf_dump`, text-only seam), symbol extraction, metadata,
  `verify_wrap_surface`.
- `Builder/ConfigurationManager.jl` — TOML → immutable `RepliBuildConfig`.
  `test_config_surface.jl` requires every field to be consumed or listed in
  `RESERVED_UNUSED` with a reason.
- `Builder/Discovery.jl` — scanning, include graph, TOML generation,
  `PRESERVED_TOML_KEYS`.
- `Builder/DependencyResolver.jl` — git/local deps (drift markers, option-injection
  guards). `[compile] source_files` **merges** with discovered sources.
- `Builder/DWARFParser.jl` — llvm-dwarfdump + nm → ClassInfo/VtableInfo.
- `Builder/SysConfigGen.jl`, `Builder/VisibilityProbe.jl` — probes, run once per
  package or version bump, not on the build path.
- `Builder/PackageRegistry.jl` — `~/.replibuild/` registry and build cache.
- `Wrapper/Generator.jl` — orchestration and the write-time refusals
  (`_assert_wrapper_loadable`, `_assert_wrapper_parses`, `_assert_base_calls_qualified`,
  `_assert_cstring_policy`, …). `exclude_symbols` and `surface_types_only` filtering
  happen here.
- `Wrapper/C/GeneratorC.jl`, `Wrapper/Cpp/GeneratorCpp.jl` — the two generators;
  `Wrapper/Utils.jl` holds the shared derivations. `Wrapper/Rust/` is experimental
  (`extern "C"` + `#[repr(C)]` only).
- `Wrapper/DispatchLogic.jl` — tier selection. Any C++ function not `noexcept` goes to
  Tier 2. `is_ccall_safe` is the only gate; DAGDiff was removed because it diffed
  DWARF against natural alignment, not the emitted struct with its `_pad_N`. A layout
  check must model what the generator emits.
- `IRGen/JLCSIRGenerator.jl` + `ir_gen/` (`FunctionGen`, `StructGen`, `ArrayViewGen`) —
  DWARF → JLCS MLIR.
- `IRGen/MLIRNative.jl` — ccall bindings to libJLCS. `IRGen/JITManager.jl` — **one
  engine per wrapped binary** behind a shared thunk cache; a per-library init failure
  degrades only that library. `Builder/ThunkBuilder.jl` — AOT thunks.
- `Debug/Debug.jl` — static inspection of emitted thunks.
- `src/mlir/` — the JLCS dialect (TableGen `JLCSOps.td`/`Types.td`, lowering in
  `impl/JLCSPasses.cpp`, C API and JIT in `impl/JLCSCAPIWrappers.cpp`, `JlcsJIT.cpp`).

## Invariants — do not break these

Each of these guards against a bug that has actually shipped. Most have a test. Run
the test named when you touch the area.

### Symbol tables and the wrap surface

- **Metadata comes from `nm -g`; the runtime resolves through dynsym (`nm -D`, or the
  PE export directory).** `verify_wrap_surface` runs before the project hash is saved.
  **R1**: an empty surface is a hard error, with or without promotion. **R2**: every
  wrappable function must be in the dynamic table under its mangled name.
  (`test_wrap_surface_guard.jl`, devtests §6c.)
- **`-fvisibility=hidden` erases the API of a library that does not annotate its
  exports**, or whose export macro is conditional and inactive. cJSON's `CJSON_PUBLIC`
  needs `CJSON_API_VISIBILITY`; ggml/llama need their `*_SHARED`/`*_BUILD` defines.
  Check that the macro is **active** (`clang -E` with the defines), not just present.
  Measure with `VisibilityProbe` before choosing hidden; the verdict is
  per-configuration. RepliBuild's own macro shims must not count toward the verdict.
- `[compile] visibility` is consumed in `get_compile_flags` because that list feeds
  `compute_compile_fingerprint`. Fold a flag in downstream and the IR cache serves the
  wrong visibility.
- `[wrap] exclude_symbols` uses anchored globs (`*`, `?` only, everything else is
  literal), applied at wrap time on both wrap paths. A pattern matching nothing is a
  hard error, and hit counting must not short-circuit.
- `[wrap] surface_types_only` is **opt-in and must stay default-off**. The reachability
  walk cannot see types used only through function-pointer members (md4c) or
  constant-only enums (argon2). `surface_types_extra` is the hatch, and an unmatched
  pattern is a hard error. Seed the walk from method `class` and from globals that are
  **in dynsym**. `_runtime_defined_names` returns `nothing` when it could not look, and
  callers must fail safe on it. **Check public enums by hand** when a package enables it.
- `_drop_unresolvable_globals!` removes globals absent from dynsym before either
  generator runs, unconditionally.
- `extract_symbols_from_binary` filters `__rb_*` and Itanium adjustor thunks
  (`_is_itanium_thunk`: `_ZTh`/`_ZTv`/`_ZTc`, while keeping `_ZTV`/`_ZTI`/`_ZTS`/`_ZTT`).
  A thunk wrapped as an API function is a zero-arg call that segfaults.
  (`test_symbol_hygiene.jl`.)
- `[link] link_dirs` emits `-Wl,-rpath` (`$ORIGIN` first) alongside `-L`. `-L` alone
  loads the distro copy at runtime, giving two copies of process-global state. This is
  what makes Hub packages composable.

### DWARF extraction

- **`class` is the demangler's prefix, not a scope.** Both `class` and name come from
  `_qualified_name_parts`: cut at the function's own parameter list
  (`_param_list_paren`, which skips a `decltype (…)` return and a local entity's
  enclosing `f(…)::`), strip the return type (template functions mangle it in;
  bail on `operator`), then split on `::` at depth 0 (template args contain `::`).
  A lambda's class therefore carries its enclosing signature, and its name is
  `operator`, like every `operator()`. (`test_local_entity_names.jl`.)
- Readelf DIE context must close on depth (`Abbrev Number: 0`). Signatures are snapshot
  with `copy`. Parameters must be **direct children** of the subprogram.
  `check_param_arity!` hard-errors on an over-count, since that would emit a wrong
  ccall. (`test_dwarf_attribution.jl`.)
- By-value aggregate returns through a typedef are repaired in a post-pass (type and
  size). Do **not** swap the return mapper for the parameter mapper: it maps `char*`
  differently and would dismantle the Cstring policy.
- Anonymous structs/unions get synthesized names (`Parent_u`, `Parent_anonN`), which
  must be pre-sanitized. They are deduplicated across CUs and emitted as aligned
  opaque regions; C11 anonymous members are injected as fields.
- **Two parsers, two dialects.** `Compiler.jl` parses **GNU binutils** output only
  (readelf on Linux, GNU objdump on Windows; byte-identical). `DWARFParser.jl` parses
  llvm-dwarfdump. Never copy a regex between them, and never route the readelf
  parser to llvm-dwarfdump.

### Generators

- **Two generators, one answer.** C and C++ duplicate type sets by design, for
  isolation. Shared decisions must still agree, so put the derivation in
  `Wrapper/Utils.jl` rather than copying it. Agreement is necessary, not sufficient.
- **Receiver (`this`) gates**: `FunctionGen._has_receiver` and GeneratorCpp's
  `_cpp_this_param` must agree (corpus test in `test_symbol_hygiene.jl` over the
  vendored `test/fixtures/receiver_gate_corpus.json`; regenerate with
  `test/gen_receiver_corpus.jl`). A ctor/dtor always has a receiver.
  `_cpp_innermost_scope` keeps `<…>` and `_bare_type_name_cpp` strips it; they are
  not interchangeable.
- Generated wrappers precompile inside consumer packages: dedup methods by dispatch
  signature, and put all process side effects in `__init__`.
- **The library populates the wrapper's namespace.** Generated diagnostics must call
  `Base.error(...)` (enforced by `_assert_base_calls_qualified`), and helpers must
  qualify `Base.get`/`Base.string`/…. `ccall` is syntax: `Base.ccall` is an
  UndefVarError.
- Only **ccall type positions** resolve eagerly, so an undeclared type there kills
  the module; `_assert_wrapper_loadable` checks exactly those. Unknown leaves degrade
  to `Ptr{Cvoid}` via `_resolve_forward_ptr`, gated on types **actually emitted**.
- A type constructor (`Ptr`) is not a type name. Names bound in Base/Core must never
  be rejected by a screen (`_assert_no_bound_name_rejected`).
- **Cstring policy has one derivation**, `_cstring_wrapper_pair`. Each tier supplies
  only the call body. `char*` returns become `Union{String,Nothing}` plus a `_ptr`
  sibling on every tier. (`test_cstring_policy.jl`.)
- Blob structs get `setproperty`/`setproperties` built in the same loop as the getters.
  `with()` forwards to them.
- **Function names come from `_julia_function_name` only.** That covers both
  generators, both varargs paths, and GeneratorCpp's proxy and deleter
  references. It is total: a catch-all after the curated list, an empty name
  falls back to the mangled symbol, and a reserved word gets a `_` suffix.
  Never inline a replace-list again. The curated part is frozen, because
  consumers call those names. (`test_function_name_derivation.jl`.)
- Enums with identical member sets emit as `const Alias = Canonical`, not a second
  `@enum`. An all-underscore name sanitizes to `c__` in **both** sanitizers. C++
  `""` stays an empty-name sentinel; do not harden it.
- Emitted introspection (`DISPATCH_TIER`/`dispatch_tier`, `STRUCT_SIZES`/
  `STRUCT_OFFSETS`/`struct_size`/`member_offset`) is derived from the final chunks, is
  unexported, and uses each generator's own sanitizer. `_vptr$X` member names must be
  sanitized, because `$` in a const table breaks the load.
- Build identity: wrappers carry `LIBRARY_SHA` (sha256 of the file — not a build ID,
  which the JLL linker never emits) and `BUILD_GENERATOR`, and warn on drift.
- Validate generated or bulk-edited Julia with `Meta.parseall` **and** check for
  `head in (:incomplete, :error)`. It does not throw on an unterminated file.

### JLCS dialect and ABI

- **Thunk arg slots hold a pointer to the argument's storage.** One load gives the
  storage and a second gives the value; pointer args double-load. Callers put `&struct`
  in the slot, not `&Ref`.
- `classifySysVStruct` decides MEMORY vs register class. Register-class structs coerce
  one scalar per eightbyte. MEMORY-class by-value args use alloca + `llvm.byval(T)`
  **on the call site and the declaration**. `ffe_call`/`try_call` share one
  `buildSysVCallShape`. Win64 is a separate `classifyWin64Struct` behind `AbiTarget`.
  (`test_struct_abi.jl`: only a system-clang callee catches a mismatch, because
  self-JIT'd callees share the bug. `gap_probe` is the discriminating case.)
- **Emitted struct size must equal DWARF `byte_size`.** StructGen lays members out at
  DWARF offsets with explicit padding (`_apply_dwarf_layout`). A struct that cannot be
  laid out degrades to an opaque `byte_size` region with a warning — never to the wrong
  size, which smashes the sret buffer. (`test_struct_layout.jl`.)
- `!jlcs.c_struct` must never appear inside an `!llvm.struct` body; inline the literal.
- `dtor_call` has no VTT operand. Its arity gate (exactly one final arg) is
  load-bearing for D2 destructors of classes with virtual bases.
- vcall uses class-local coordinates (`vtable_offset = this_offset = 0` plus own slot).
  MI callers upcast via `as_Base`; virtual-base upcasts are dynamic and wrapper-side.
  Destructors stay direct calls.
- **Every declared builder/accessor must be defined.** `libJLCS.so` must `dlopen` with
  `RTLD_NOW` (`test_jlcs_invariants.jl` §D, run in a subprocess). Match `4mlir4jlcs`,
  not `_ZN`, since const members mangle `_ZNK`. Op producer liveness is computed from
  `JLCSOps.td` (§E). Verifier coverage is not yet computed.
- The C++ personality (`__gxx_personality_seh0` on mingw, `_v0` elsewhere) is one fact
  in three spellings (`JLCSPasses.cpp`, `JLCSCAPIWrappers.cpp`,
  `MLIRNative.CXX_PERSONALITY`); `test_cxx_personality.jl` holds them equal.
  `uwtable` is stamped on every function with a body, independent of landing pads.
  The landing pads are still required on Linux.

### Caching

1. Per-file IR cache (`.replibuild_cache/`), keyed on a compile-fingerprint `.key`
   sidecar. The hash in the IR filename identifies the source, not the flags.
2. Project content hash (`.replibuild_cache/project_hash`): TOML, sources, include
   dirs (recursive), git HEAD.
3. Registry build cache (`~/.replibuild/builds/<hash>/`, gates `use()`). It includes
   `_generator_fingerprint`: RepliBuild version, HEAD, **and a dirty-tree digest**, so
   uncommitted generator edits miss. `_store_build` copies directories.

`RepliBuild.VERSION` is derived from `Project.toml`; never reintroduce a literal. It
feeds the fingerprint and `BUILD_GENERATOR`. To check what is registered in General,
read `R/RepliBuild/Versions.toml` from the tarball that `path =` in
`~/.julia/registries/General.toml` names. That is `General.tar.zst` now, read with
`tar --zstd`. A leftover `General.tar.gz` beside it is stale and lacks recent
versions. 3.3.2 is a permanent hole there. Registration is manual and
outward-facing, so ask first.

## Debugging Tier 2 thunks

Start here for any ABI question: register state is observable.

- **Static (preferred):** set `ENV["REPLIBUILD_JIT_OBJDUMP"]="1"` **before** the wrapper
  loads (engines are created once per library per process), then
  `D = RepliBuild.Debug; D.thunks(pkg); D.walk(pkg, sym); D.disassemble(pkg; symbol=s);
  D.dwarf(pkg; section="line")`. Debug info is line-tables-only, so there are no locals.
- **gdb** stops in the emitted `.debug/mlir/jlcs_<hash>.mlir` at source level (ORC's
  GDB listener plus MLIR's FileLineColLoc; nothing extra to build):
  ```bash
  timeout -s KILL 175 gdb -batch -nx -ex 'set pagination off' -ex 'set confirm off' \
    -ex 'handle SIGSEGV nostop noprint pass' -ex 'set breakpoint pending on' \
    -ex 'break <mangled>_thunk' -ex run -ex 'bt 1' -ex 'info source' -ex 'disassemble /s' \
    --args julia --project=. test/mi_test/verify.jl
  ```
  Pending breakpoints and SIGSEGV pass-through are both mandatory; Julia's GC uses
  SIGSEGV.

## Testing rules

- **Every `test_*.jl` is included by exactly one suite.** `runtests.jl` enforces
  wiring (parsing real `include` lines) and disjointness (the `SHARED` dict holds
  deliberate exceptions, currently empty). CI files need no toolchain.
- **No test may read the Hub or a home-directory path.** Grep for the shape
  (`homedir()`, `expanduser`, `"/home/`), not the string. Vendor fixtures instead,
  with a generator script that refuses to write a vacuous fixture
  (`gen_receiver_corpus.jl`, `gen_thunk_symbols.jl` are the only files allowed to
  touch the Hub). Hub rebuilds are the integration test.
- **Guard against vacuous green.** Assert that the loop actually ran
  (`checked == expected`). Never wrap corpus loading in a bare `try/catch`.
  Negative-check a new guard by neutering the fix, using the exact method signature
  (a less specific override loses dispatch and proves nothing).
- **Never `exit()` in a file a suite includes.** It ends devtests with a success status
  and silently skips everything after it; use a skipped testset. Still open:
  `test_mlir_templates.jl:26` (runs first in the libJLCS group), `test_jlcs_invariants.jl`
  `:32`/`:294`, `test_jlcs_producers.jl:42`, `test_struct_abi.jl:41`/`:47`,
  `test_multilib_jit.jl:33`.
- devtests aborts a file at the first failing top-level testset, so one red hides an
  unknown number of unrun tests. Put cheap probes early.
- Assert **deltas** on process-global state (JIT engines, fixture counters), not
  absolutes, because files share one process. Assert **names**, not counts, downstream
  of static promotion.
- A background command's exit code is the wrapper's. Write `echo "EXIT=$?" >> log` and
  read that.
- A test that states one host's answer instead of the invariant goes red on the other
  host (Linux vs Windows). Drive host-conditional code with both spellings.
- A regression check for a generator change: re-wrap pugixml (full debug info) or box2d
  and diff against a wrapper from unpatched HEAD. It should be byte-identical apart
  from timestamps. For dialect changes, compare machine code, not IR text.
- Fixtures: `c_test/`, `stress_test/`, `stl_test/`, `mi_test/`, `vi_test/`,
  `callback_test/`, `c_abomination_test/`, `abi_nested_test/`,
  `convenience_overload_test/`, `struct_abi/`, `slice_test/` (Tier 1). Fixture tomls
  are regenerated by devtests' `discover(force=true)`, with `CURATED_FIXTURE_CONFIG`
  re-seeded after.

## Windows (`x86_64-w64-windows-gnu`, MSYS2 CLANG64) — since v4.0.0

Linux and Windows are supported; macOS is refused at load (no AAPCS64 classifier).
Setup and full findings: `docs/updates/WINDOWS_PORT.md`. The rules that bite code
changes:

- DWARF dumper: GNU `objdump` on Windows. `_is_gnu_binutils` must reject llvm-objdump
  (on PATH in CLANG64), which otherwise parses **zero functions** silently. Use
  `Compiler._gnu_objdump()` wherever GNU objdump specifically is needed.
- The API is the **PE export directory** (`objdump -p`), not `nm`. mingw links the CRT
  statically into every DLL. Shims need `__declspec(dllexport)` (`RB_SHIM_EXPORT`,
  keyed on `_WIN32`). One dllexport disables auto-export for the whole image, so
  `create_library` adds `--export-all-symbols` only when shims are the only exports.
- C bucket needs a sysroot (`_c_bucket_sysroot`: `REPLIBUILD_C_SYSROOT` → clang prefix
  → `MSYS2_ROOT`) and `-rtlib=compiler-rt`.
- libc++ ABI tags (`size[abi:…]()`) and `exclude_from_explicit_instantiation` shape the
  STL path. Anchor `basic_string` matches.
- LLP64: `long` is 4 bytes. Use `C_LONG_MLIR`/`C_WCHAR_MLIR` (`test_llp64_widths.jl`).
- ORC deadlocks on unresolved symbols, so the LTO path has
  `_lto_unresolved_symbols`. There is no `dlsym` in a Windows process.
- JIT `.pdata` is registered with `RtlAddFunctionTable` in the dialect-owned LLJIT
  (`JlcsJIT.cpp`). Never call `__register_frame` on COFF; it poisons unwind
  process-wide.
- AOT thunks import from libJLCS. `JITManager.open_thunks_library` opens libJLCS by
  absolute path first. **Never vendor a second copy** — two exception buffers swallow
  every C++ exception.
- `FILE` is `struct _iobuf`. Bake paths into generated Julia with `repr()` and into
  TOML with `TOML.print`. `JSON.parsefile` mmaps and blocks deletion. `_canon_path` is
  identity off Windows. `/dev/null` passed to `run` is not a path.
- Before believing any "open on Windows" note, check `git log -S` on the symptom.

## Fresh clone

Verified 2026-08-26 by cloning to a tempdir: `Pkg.instantiate`, `runtests.jl` green,
`build.sh` builds, `check_environment` reports tiers. `build.sh` gates on LLVM ≥ 21,
`mlir-tblgen`/`llvm-config` major match, and `lib/cmake/mlir` presence. Redo the clone
check before claiming portability again; reading `.gitignore` does not settle it.

## Open problems

- **`static` member functions get a phantom `this`** in both receiver gates (e.g.
  tinyxml2 `XMLDocument::ErrorIDToName`). About 17 sites in the Hub. Fix: read
  `DW_AT_object_pointer`, which is present on the **definition** DIE only (declaration
  DIEs of instance methods lack it, so a declaration-fed gate would strip every
  `this`). Carry it through the `DW_AT_specification` merge. Readelf spells it
  `DW_AT_object_pointer: <0x70>`, llvm-dwarfdump `DW_AT_object_pointer(0x…)`. Template
  constructors (`gguf_kv<…>`) miss the ctor-name test and survive only via
  `struct_types`.
- **C ccall coverage**: 98.6% (5368/5444, 2026-08-29), target 99%. The 76 misses:
  43 varargs (declaration work via `[wrap.varargs]`), 27 mpack by-value aggregate
  returns (typed correctly but emitted as ABI-trap stubs), 5 mpack by-value aggregate
  args, and 1 counting artefact (zlib `gzgetc`). Measure `ccall` targets, not emitted
  names; a trap stub is not coverage.
- The `exit(0)` sites listed under Testing rules.
- `.replibuild_cache/slices/<modkey>/` accumulates unboundedly (harmless).

## Not Yet Built (roadmap, not bugs)

Absence is the default state on this project. A missing feature is an unbuilt piece;
only something contradicting what **is** built is a bug. Log new gaps here, dated, and
don't file them as defects. Never extrapolate a narrow entry into "X is unbuilt" —
read the generator output first. (User-facing C++ RAII — `Managed<T>` finalizers with
DWARF-resolved destructors — **is** built.)

- Array-view Julia-side accessors for the rank-1 thunks, plus rank ≥ 2 members.
- `is_struct_packed` over-classifies padding-free structs (wasted work, not wrong).
- Op verifiers for `ffe_call`, `try_call`, `load/store_array_element`, `ctor_call`,
  `yield`, `marshal_ret`. A computed verifier-coverage guard would also help.
- In-IR virtual-base upcast op (no producer needs one yet).
- vcall for struct-shaped virtual signatures (these keep static dispatch).
- AAPCS64 classifier (macOS/ARM).
- Codegen `.ll → .so` still shells to clang; not internalised.

## Maintaining this file

- Keep it lean. When you fix something, put the story in `CHANGELOG.md` under
  `## Unreleased`, and add a line here only if it creates a rule a future session must
  follow.
- Derived claims (counts, line numbers, versions, test totals) are what rot. Prefer
  naming the test that computes them over writing the number. Avoid `file:line`
  citations; name the function.
- The file is tracked: no machine-local paths.
