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

- **Rebuild the Hub only in three cases:**
  - before a release, for stability;
  - for the one package that exercises the calling path being changed, or that
    caused the error;
  - when John asks.

  A rebuild or re-wrap sweep is never routine verification of a change. Use the
  change's own regression test plus runtests and devtests. A sweep costs hours of
  wall-clock time, and outside those cases its wrappers come back byte-identical.
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
in the toolchain env, `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` by default) →
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
  C artifact came from the JLL. `_clang_for_c_bucket` uses the system clang only
  when the JLL cannot be loaded (`_jll_clang()` is `nothing`); a JLL clang that
  does not start is reported as `BuildBridge.SPAWN_FAILED`, never replaced.
- **C++ / Tier 2 uses system LLVM+MLIR 21+** (23.1.3 under MSYS2 CLANG64 on the
  reference host, which is Windows; on Linux, CI's `tier2-linux.yml` runs devtests
  against apt.llvm.org's LLVM/MLIR 22 and 23) for the dialect and the external
  pipeline. `libJLCS.so` (`libJLCS.dll` on Windows: it
  imports `libLLVM-<major>.dll` and links MLIR statically) is a gitignored build
  artifact pinned to the installed MLIR minor SONAME: rebuild after a minor upgrade. A patch
  bump inside a minor needs nothing. `deps/build.jl` builds it at install time when
  possible and never throws. On Windows it runs build.sh through MSYS2's
  `bash.exe -l` with `MSYSTEM=CLANG64`, which is a CLANG64 shell from any
  process; `src/mlir/windows_build.jl` is that command's one copy (the doctor's
  advice uses it too). Tier 3 alone is the whole
  product for C.
- Julia's bundled libLLVM and system libLLVM coexist in one process. The GDB JIT
  descriptor resolves to Julia's copy; this is correct, do not "fix" it.
- For LLVM/JIT crashes, check `libJLCS.so`'s mtime/SONAME against the installed MLIR,
  and dual-LLVM symbol scope, before theorising.
- **One LLVM per module.** The C bucket is chosen by `[wrap] language`, never by
  file extension: a C++ package compiles its `.c` files with the system clang. IR
  from an older LLVM read by a newer one can lose its debug info with only a
  warning, and then every function in it wraps from an inferred signature.
  `_assert_debug_info_kept` turns that warning into a build error; keep it on every
  path where one LLVM reads another's IR.
- The final IR -> library step runs codegen at `[link] optimization_level` with
  `-disable-llvm-passes` (`_codegen_opt_flags`). Never drop the `-O`: the `-O0`
  backend is what clang uses without it.
- **Two Julia releases, two LLVM.jl majors.** RepliBuild supports the two Julia
  releases in the CI matrix (`.github/workflows/test.yml`), each with its own
  bundled LLVM, and LLVM.jl 9 and 10. Verify on both Julias, not only the juliaup
  default. LLVM.jl 10 changed what some names mean without renaming them:
  `isconstant(gv)` asks whether a value is an LLVM `Constant`, which every global
  is (the old meaning is `constant(gv)`), an attribute's `kind` is a Symbol, and
  `unnamed_addr` returns an enum. Read such names through `LLVM.API`, which is the
  same in both majors. A scan for names that no longer exist cannot find this
  class, so diff the method definitions; the synthetic-IR testsets at the top of
  `test_slicer.jl` and `test_static_promotion.jl` catch the known cases.

### Tier 1 — quarantined side project

`Base.llvmcall` dispatch is **John's side project**: it ships and works, but it is not
a supported tier. `[wrap.tier1] enable` defaults false and every Hub config pins
`[link] enable_lto = false` — but Hub recipes DO turn slicing on (lua, sqlite, zlib,
cglm, … — `grep -l 'enable = true' packages/*/replibuild.toml` under `[wrap.tier1]`),
so `use()` of those packages takes Tier 1. (This line said "no shipped package takes
it" until 2026-09-26; it was wrong for as long as those recipes existed.) Its suites
(`test_static_promotion`, `test_slicer`, `test_tier1_dispatch`) are **deliberately
unwired** from devtests; `runtests.jl`'s wiring guard names them in an `experimental`
list. They also need `test/slice_test/replibuild.toml`, which is gitignored and
machine-local (John's call: document it, don't track it). The one Tier 1 test
devtests runs is `test_tier1_alias_binding.jl` (§7e, John's call, 2026-10-24): it
guards how a slice binds, which every Hub recipe with slicing on depends on. **Do
not un-quarantine, re-wire or invest in it (M4/perf) unless John asks.**

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
- **A slicing throw demotes; it never fails the wrap.** On one target it is an
  errored refusal (`Slicer.errored`, tag `:slicer_error`) that is never cached;
  on the whole module it turns Tier 1 off for that wrap. Keep new slicing work
  inside those two catches. A nonzero `errored` in the `tier1:` line is a Slicer
  bug.
- Slice constants are keyed on the **mangled** symbol (`julia_name` is not
  injective). Slices are `include_dependency`s; Julia 1.11+ tracks those by content,
  so `touch` does not invalidate.
- **A slice declares its library's own symbols by their `__rb_<lib>_` alias,
  never by the public name** (2026-10-24, Linux). ORC resolves a `declare`
  process-wide and the pre-flight only asks whether the library *has* the
  symbol, so a slice that called another public function of its library bound
  whichever library defined that name first. Julia loads its own `libz.so.1` at
  startup, and the Hub's zlib ran `gzclose` as a slice whose `gzclose_w` was
  Julia's, which freed a handle the other zlib allocated (`double free or
  corruption`); Julia bundles zlib, zstd, pcre2, curl, ssh2, git2,
  mbedtls/openssl, gmp, mpfr, openlibm, nghttp2. So with `[wrap.tier1] enable`,
  promotion also gives every exported definition the exported LLVM alias
  `__rb_<lib>_<name>` (`_alias_exports!`, `exported_aliases` in the metadata, a
  `dllexport` aliasee giving a `dllexport` alias), and the Slicer declares the
  alias (`_claim_rb_aliases!`). A reached exported definition with no alias is a
  **refusal** that names the rebuild, never a public-name declare and never an
  error; declarations the library does not define (libc) keep their names; a
  tier1-off build is unchanged. An alias shares its aliasee's address, so
  anything that keys `nm` output by address must let the real name win
  (`extract_symbols_from_binary`). The project-hash tag is v3 so a stale Tier 1
  build relinks once. Windows has no process-wide namespace and never showed
  it. Julia's crash reporter then deadlocks in `malloc`, so the symptom is a
  process in a futex wait at zero CPU: run it under gdb (`handle SIGSEGV nostop
  noprint pass`). The alias is only half of it: the library's **native** calls
  to its own exports go through the PLT and are interposed the same way (the
  rule below under symbol tables), and `LD_DEBUG=bindings` is the evidence for
  which library a reference bound to. (`test_tier1_alias_binding.jl`.)
- **A slice is IR for the libLLVM that printed it**, and newer IR does not parse
  in an older LLVM (`icmp samesign` is LLVM 20; Julia 1.12 is 18). The slice cache
  is per libLLVM major (`_slice_cache_key`), and a kernel keeps its ccall body
  when the running major is not the wrapper's `TIER1_LIBLLVM_MAJOR`
  (`test_toolchain_switch.jl`).
- **On Windows a longjmp cannot come back into a JIT'd slice frame.** Julia's JIT
  gives each function unwind info for exactly `[start, end)`, a slice ending in a
  noreturn call returns past it, and the SEH unwind mingw's `longjmp` performs dies
  with STATUS_BAD_STACK. `_setjmp_slices_safe` holds that fact: `allow_setjmp` is
  ignored on Windows, and test_slicer skips its `(1, 0)` call there.

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
- `Builder/PackageRegistry.jl` — `~/.replibuild/` registry. **A project builds
  where it lives.** An entry points at the project's own TOML; `register` builds
  and wraps in place, vendors the output to `builds/<hash_config>/` (immutable:
  a session may hold a vendored DLL open), and writes `packages/<Module>/`, a
  Julia package whose entry file `Base.include`s the vendored wrapper into
  `Base.__toplevel__` (a bare `include` does not exist there). `use` re-vendors
  when the hash moves and loads the last vendored build if the project is gone.
  Never copy a project's sources into the registry: copies made two projects
  with the same relative paths overwrite each other, and edits never reached
  `use`. `register` adds the package only to environments (no `name` in their
  Project.toml), never to a package's own; test suites set `REPLIBUILD_HOME` to a
  temp dir and `REPLIBUILD_JULIA_ENV=none`, and a test that wants an
  environment passes a temp one with Pkg offline.
- `Wrapper/Generator.jl` — orchestration and the write-time refusals
  (`_assert_wrapper_loadable`, `_assert_wrapper_parses`, `_assert_base_calls_qualified`,
  `_assert_cstring_policy`, …). `exclude_symbols` and `surface_types_only` filtering
  happen here.
- `Wrapper/C/GeneratorC.jl`, `Wrapper/Cpp/GeneratorCpp.jl` — the two generators;
  `Wrapper/Utils.jl` holds the shared derivations. There is no Rust generator (the
  experimental `Wrapper/Rust/` was deleted in `7dfe3ee`).
- `Wrapper/DispatchLogic.jl` — tier selection. Any C++ function not `noexcept` goes to
  Tier 2, and **noexcept means a declared-noexcept bare name (candidate) AND a
  `nounwind` mangled definition in the linked IR** (`_mark_noexcept!`). Never route on
  a name alone: a throw through a Tier-3 ccall aborts the process
  (`test_noexcept_routing.jl`, `callback_test` "bare-name noexcept collision"). A
  `nounwind`-only widening would expose Tier-3 emitter gaps (see CHANGELOG
  2026-09-26). `is_ccall_safe` is the only gate; DAGDiff was removed because it diffed
  DWARF against natural alignment, not the emitted struct with its `_pad_N`. A layout
  check must model what the generator emits.
- `IRGen/JLCSIRGenerator.jl` + `ir_gen/` (`FunctionGen`, `StructGen`, `ArrayViewGen`) —
  DWARF → JLCS MLIR.
- `IRGen/MLIRNative.jl` — ccall bindings to libJLCS. `IRGen/JITManager.jl` — **one
  engine per wrapped binary** behind a shared thunk cache; a per-library init failure
  degrades only that library. `Builder/ThunkBuilder.jl` — AOT thunks, compiled
  as wrap's last stage.
- `Debug/Debug.jl` — static inspection of emitted thunks.
- `src/mlir/` — the JLCS dialect (TableGen `JLCSOps.td`/`Types.td`, lowering in
  `impl/JLCSPasses.cpp`, C API and JIT in `impl/JLCSCAPIWrappers.cpp`, `JlcsJIT.cpp`).
  `impl/JLCSAbiOracle.cpp` is test only, compiled when CMake finds LLVM 23+'s ABI
  library.

## Invariants — do not break these

Each of these guards against a bug that has actually shipped. Most have a test. Run
the test named when you touch the area.

### Symbol tables and the wrap surface

- **Metadata comes from `nm -g`; the runtime resolves through dynsym (`nm -D`, or the
  PE export directory).** `verify_wrap_surface` runs before the project hash is saved.
  **R1**: an empty surface is a hard error, with or without promotion. **R2**: every
  wrappable function must be in the dynamic table under its mangled name.
  (`test_wrap_surface_guard.jl`, devtests §6c.)
- **A thunk calls its target by name, so the target must be in that runtime set**,
  not merely in `nm`. On PE a method the DLL does not export is still a global `T`
  in the COFF symbol table. `parse_vtables` gates method addresses with
  `_gate_to_exports`, and the STL method table goes through
  `_runtime_symbol_names`, the one source for "what the loader binds"
  (`verify_wrap_surface` reads it too). One unbindable thunk fails the library's
  whole JIT module. (`test_windows_port.jl`, `test_libcxx_exports.jl`.)
- **libc++ 23 hides its own functions** with a `#pragma clang attribute` in
  `__configuration/namespace.h`. It applies on COFF too, and no macro opts out. On
  MinGW clang turns hidden into `-exclude-symbols:`, so those members never reach the
  export directory. `create_library` un-hides hidden `std::__N` definitions in the IR
  for `[types] templates` packages (`_unhide_libcxx_ir`), on both codegen paths.
  Members `libc++.dll` exports itself are never auto-exported by lld (their `__imp_`
  twin wins) and stay unbound.
- `nm -D` output is parsed only by `Compiler._nm_dynamic_names`, which cuts ELF
  `@VER`/`@@VER` suffixes. Mangled output only: demangled names can contain `@`.
- **AOT thunks are compiled by wrap's last stage, never by build.** Only the C++
  generator knows which thunks a wrapper binds. It writes that set to
  `thunk_manifest.json` on **every** wrap, JIT included (JITManager compiles only
  what it lists), then hands the same set to `ThunkBuilder.build_aot_thunks`
  (`wrap_library`'s `aot_compile` stage), and `_assert_aot_thunks_present` checks
  the library's exports. Build ran before the set existed and had to guess it, so
  a new AOT package took two build-and-wrap passes. A stage that writes no library
  (`AOTStageFailed`) gives a JIT wrapper. (`test_thunk_manifest.jl`,
  `test_aot_thunks.jl`.)
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
- **A library binds its own exported symbols to itself.** On ELF a library's own
  PLT calls and GOT loads resolve through the global scope first, so a library
  loaded earlier with the same names takes them: the Hub's zlib called Julia's
  `libz.so.1` from its own `gzclose` (2026-10-24). Every library `create_library`
  links passes `_self_binding_link_flags` (`-Wl,-Bsymbolic` on ELF, nothing on PE,
  where a DLL never imports its own symbols). A new link site must pass it too, and
  a Tier 1 slice, being code outside the library, needs the `__rb_` aliases on top
  (Tier 1 section). `LD_DEBUG=bindings` says which file a reference bound to.
  (`test_codegen_flags.jl`, `test_tier1_alias_binding.jl`.)

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
- **A type's parts are collected in pass 1, before pass 2 resolves any signature.**
  That includes a subroutine type's parameters (its direct-child formal parameters,
  then `...` for `DW_TAG_unspecified_parameters`): clang emits a type after the
  first function that uses it. `resolve_type` gives each branch of a signature its
  own cycle guard (`copy(visited)`); a shared one made the second `int` of
  `int (*)(int, int)` a cycle. (`test_dwarf_attribution.jl`.)
- By-value aggregate returns through a typedef are repaired in a post-pass (type and
  size). Do **not** swap the return mapper for the parameter mapper: it maps `char*`
  differently and would dismantle the Cstring policy.
- **A DIE with no `DW_AT_type` is `void`**, pointed at or qualified: a
  target-less pointer is `void*`, a target-less `const` is `const void` (clang's
  `const void *`), and the transparent qualifiers (`volatile`, `restrict`,
  `_Atomic`) give `void`. `resolve_type` never answers `unknown` for one; that
  spelling once reached 1257 parameters in llama.cpp's metadata.
  (`test_dwarf_attribution.jl`.)
- Anonymous structs/unions get synthesized names (`Parent_u`, `Parent_anonN`), which
  must be pre-sanitized. They are deduplicated across CUs and emitted as aligned
  opaque regions; C11 anonymous members are injected as fields. One named after a
  parent the provenance gate drops is dropped with it (`drops_type`): that is a
  lambda's closure, declared in the project but named after libc++'s
  `__func<(lambda at <path>:<line>:<col>)>`.
- **Types are keyed by bare name, and one name can have several definitions**: the
  same type spelled differently by C and C++ units (`_Bool`/`bool`, `_anon1`/`data`),
  or distinct types sharing a name (a pimpl `impl` per class). `_canonical_definition`
  keeps one by content and marks `conflict` when layouts differ, with every
  definition's size in `conflict_sizes`. A pass that lets one
  DIE win iterates `_die_order`, never `type_refs`, whose hash order moves with the
  Julia release and every offset shift. Everything downstream keys on the bare name
  (`resolve_type`, member and signature `c_type`s, `_cpp_innermost_scope`), so
  qualified keys would be a cross-cutting change. (`test_emission_determinism.jl`,
  devtests `test_determinism_fixture.jl`.)
- **No call passes a conflicted name by value**, nor a record holding one by value
  at any depth. An enum crosses as an integer of its width, so it traps only when
  its `conflict_sizes` differ. `FunctionGen.conflicted_within` is the one walk:
  `by_value_crossing` returns `:conflict`, `is_ccall_safe` sends a C++ function to
  BRANCH 1's trap, and GeneratorC and the shared varargs path trap with
  `_conflict_trap_body`. (`test_conflicted_names.jl`, devtests
  `test_conflict_trap_fixture.jl`.)
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
  `test/tools/gen_receiver_corpus.jl`). A ctor/dtor always has a receiver. A definition
  DIE's `DW_AT_object_pointer` is recorded as `has_object_pointer`, and `false` (a
  static member) overrides the class heuristic in both gates. Declaration DIEs never
  carry the attribute, so a missing key must keep the heuristic.
  `_cpp_innermost_scope` keeps `<…>` and `_bare_type_name_cpp` strips it; they are
  not interchangeable.
- Generated wrappers precompile inside consumer packages: dedup methods by dispatch
  signature, and put all process side effects in `__init__`.
- **The library populates the wrapper's namespace.** Generated diagnostics must call
  `Base.error(...)` (enforced by `_assert_base_calls_qualified`), and helpers must
  qualify `Base.get`/`Base.string`/…. `ccall` is syntax: `Base.ccall` is an
  UndefVarError.
- **`Any` is never a foreign-call type**, in either position: as a return it
  dereferences an integer as a Julia object, and as an argument it passes the box
  (`jl_value_t*`) instead of the value. `_assert_no_any_ccall_return` and
  `_assert_no_any_ccall_argument` refuse both. The one exemption is a variadic
  `@ccall` argument, which is the user's explicit `[wrap.varargs]` declaration.
  `::Any` in a Julia signature is normal. (`test_wrapper_type_bindings.jl`.)
- **A Tier-3 by-value record is classified per target, from its fields' type
  alignment, never a member's size** (C1: `{ char; short[3]; }` got an sret
  the callee never reads).
  - `_c_return_in_memory`, `_c_explicit_sret`, `_c_blob_param_misclassified` and
    `_c_register_class_misstated` are the answers, and `_c_has_unaligned_field`
    is the walk. `_c_record_abi()` is the only place GeneratorC reads the target.
  - Register-class traps (a float-bearing blob, a mixed float/integer region) are
    SysV's only: Win64 crosses a record by size alone on both sides.
  - Whether a record is packed at all is `_record_natural_layout`, the one answer
    for `is_c_lto_safe` and `_is_struct_unsafe`.
  - GeneratorCpp sends to Tier 2 a function that crosses by value a ≤16-byte
    byte blob, or a record whose fields put bytes over a `float` or `double`,
    because bytes classify INTEGER (`_crosses_small_blob`).
  - BRANCH 0 runs only when no blob trap fired.

  Tests: `test_c_sret_classification.jl`, `test_c_sret_abi.jl`,
  `test_record_natural_layout.jl`.
- **A hidden return buffer has the record's alignment.** Julia's own sret
  buffer is aligned to the stand-in (a byte blob: 1), and the callee may store
  with `movaps`. `_c_record_align` is the record's (the parser records
  `DW_AT_alignment` on records and members as `alignment`, a vector member's as
  its size), and BRANCH 0 passes a buffer it aligned itself wherever that
  exceeds the stand-in's and 8 (`_c_sret_buffer_align`). The metadata needs a
  rebuild to carry it. Record-level `alignment` also routes C++ to Tier 2
  (`_is_struct_unsafe`).
- **A signature spells a record or enum the way it is declared**, which is
  `_sanitize_c_type_name` of its C name. GeneratorC re-spells every return and
  parameter type through `_as_declared`, a table of the names the sanitizer
  changes. A raw keyword (`baremodule`) in a ccall type position fails the parse
  (abi_fuzz I1). (`test_c_sret_classification.jl`.)
- Only **ccall type positions** resolve eagerly, so an undeclared type there kills
  the module; `_assert_wrapper_loadable` checks exactly those. Unknown leaves degrade
  to `Ptr{Cvoid}` via `_resolve_forward_ptr`, gated on types **actually emitted**.
- A type constructor (`Ptr`) is not a type name. Names bound in Base/Core must never
  be rejected by a screen (`_assert_no_bound_name_rejected`).
- **Cstring policy has one derivation**, `_cstring_wrapper_pair`. Each tier supplies
  only the call body. `char*` returns become `Union{String,Nothing}` plus a `_ptr`
  sibling on every tier. (`test_cstring_policy.jl`.)
- Blob structs get `setproperty`/`setproperties` built in the same loop as the getters.
  `with()` forwards to them. C++ field structs get `setproperty` too
  (`_field_setter_chunk`), so a record keeps its setters when it stops being a blob.
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
  which an ELF link carries only when asked) and `BUILD_GENERATOR`, and warn on drift.
  **Links are reproducible**, so the same IR gives the same `LIBRARY_SHA`: every link
  of a shipped binary passes `_reproducible_link_flags` (on PE, lld and GNU ld write
  the clock into the header otherwise). A new link flag must keep the output a
  function of the inputs: no clock, no temporary path, no randomness
  (`test_reproducible_link.jl`).
- **Emitted text must not depend on Dict or Set order.** A Dict's iteration order
  is the hash order of its keys, which differs between Julia releases, and for
  metadata read back from JSON it also follows the file's key order. That holds for
  a Dict the generator builds itself as much as for metadata. Iterate sorted keys
  wherever order reaches the wrapper or picks a winner (export lists, STL
  factories, `Managed` blocks, header enums, globals; the struct topo sort breaks
  ties with `sort(ready)`), and a lookup with several matches takes the smallest key
  (`_fuzzy_struct_lookup`, `_fuzzy_dwarf_lookup`). Otherwise the byte-identical
  regression diff below breaks, and a wrapper changes with the Julia release.
- **Supplementary header enums are the project's own.** The Clang.jl walk in
  `ClangJLBridge._extract_enums_via_ast!` keeps an `EnumDecl` only when its file
  lies under a named header's directory or an include dir. Clang.jl's libclang is
  the running Julia's LLVM generation, so anything it pulls in from system
  headers (libstdc++'s `errc`) would make the wrapper follow the Julia release.
  (`test_header_enum_provenance.jl`.)
- **The wrapper file is `julia/<module name>.jl`, and the module name has one
  derivation**, `ConfigurationManager.module_name_for` (`[wrap] module_name`, else
  CamelCase of `[project] name`), behind `get_module_name` and
  `wrapper_module_name(toml_path)`. Anything that must find the file wrap writes
  asks that, never the package name: a scaffold's `[project] name` is lowercased
  and a registry TOML names the library. Windows folds case and hides a guess.
- **A callback's `@cfunction` example comes from the header typedef its parameter
  is declared with** (`function_pointer_typedef`, `_callback_typedef_signature`),
  else from its DWARF signature (`_dwarf_callback_signature`), never from a name
  match: that documented other callbacks' signatures. The DWARF path strips exactly
  one pointer, splits at depth 0, and spells each type through the generator's own
  parameter step (`_ccall_param_type`), and it needs metadata marked
  `function_pointer_parameters` (older metadata has empty lists). A variadic
  callback, a pointer to one and an unspellable type get the signature as text,
  escaped. Both generators document parameters through `_param_docs!`.
- Validate generated or bulk-edited Julia with `Meta.parseall` **and** check for
  `head in (:incomplete, :error)`. It does not throw on an unterminated file.
  `module 3x` parses as a module named `3`, so `_assert_wrapper_parses` also
  requires the module name to be a Symbol, and `get_module_name` never starts
  one with a digit (abi_fuzz M1).

### JLCS dialect and ABI

- **Thunk arg slots hold a pointer to the argument's storage.** One load gives the
  storage and a second gives the value; pointer args double-load. Callers put `&struct`
  in the slot, not `&Ref`.
- **A record's SysV class comes from its DWARF facts, never from the body it
  crosses as.** `FunctionGen.sysv_record_abi` (`ir_gen/SysVRecordABI.jl`)
  classifies members at their offsets, bitfields by bit range, nested records
  and unions recursively, `long double` as X87, `_Complex` as two halves,
  `__int128` as two INTEGER eightbytes, with the record's alignment from
  `DW_AT_alignment` or its members'; the generator states the verdict on every
  `ffe_call`/`try_call` (`sysv_args`, one per operand, and `sysv_ret`:
  `mem:<align>`, `regs:int,sse`, `regs:none,sse`, `ignore`) and
  `buildCallShape` acts on it under the SysV target. A body cannot carry these
  facts: StructGen's pad arrays are bytes, a tiled body spells a nested record
  or a union as bytes, a bitfield sits in its unit's whole type, and no body
  is aligned above 8. `classifySysVStruct` (the body reading) stays only for
  IR with no statement, which is hand-written test IR. Register-class records
  coerce one scalar per eightbyte; MEMORY-class by-value args use alloca +
  `llvm.byval(T)` **on the call site and the declaration**, at the stated
  alignment. `ffe_call`/`try_call` share one `buildCallShape`. Win64 is a
  separate `classifyWin64Struct` behind `AbiTarget` and never reads the
  statement. A `long double` record returned by value is refused on SysV
  (`sysv_trap_reason`: st0), so is a record of 16 bytes or less whose
  member kinds DWARF does not settle; `thunk_crossings` reports both as
  `abi_trap`, which GeneratorCpp explains apart from a missing layout.
  (`test_sysv_record_abi.jl`, `test_sysv_live.jl`.)
- **`test_sysv_abi_oracle.jl` diffs the SysV classification against LLVM's ABI
  library** (`llvm/ABI`, LLVM 23+; skips elsewhere). `jlcs_probe_sysv_call_shapes`
  and `jlcs_abi_oracle_sysv` are test seams: nothing on the JIT or wrap path
  calls them, and the oracle must stay optional so libJLCS builds against 21 and
  22. CMake takes the component from `libLLVMABI.a` where one exists, else
  link-probes the LLVM dylib for it (Arch ships only the dylib, and it carries
  `llvm::abi`); a dylib without it means the test skips, never a build error.
  Every disagreement fails; `SAO_CURATED` says `:trap` only where the
  refusal is the right answer (the `long double` return). A change to the
  classification that moves a verdict must move that table and
  `test_sysv_live.jl`'s expectations in the same change. `thunk_crossings` asks
  `sysv_trap_reason` for the host's ABI, so a module generated on Windows keeps
  records a SysV host refuses; the test applies the SysV answer.
  (`test_struct_abi.jl`: only a system-clang callee catches a mismatch, because
  self-JIT'd callees share the bug. `gap_probe` is the discriminating case.)
- **Emitted struct size must equal DWARF `byte_size`.** StructGen lays members out at
  DWARF offsets with explicit padding (`_apply_dwarf_layout`). When natural alignment
  cannot place them (packed, `#pragma pack`), the body is PACKED with explicit pads
  (`_packed_dwarf_layout`), so the classifier sees the misaligned field. That check
  measures each member as the body emits it: an alias `!Struct_X` has X's alignment,
  not the 1 of its bytes, or a `#pragma pack` body comes out unpacked (abi_fuzz X8).
  Only what no body can model (overlaps, unmeasurable members) degrades to an opaque
  `byte_size` region — never to the wrong size, which smashes the sret buffer.
  (`test_struct_layout.jl`, `test_packed_layout.jl`, `test_thunk_byte_regions.jl`.)
- **The Julia struct IS the C layout.** GeneratorCpp proves each struct's fields
  against DWARF (`_prove_julia_layout`) and emits a C-layout byte blob when they
  cannot reproduce it; GeneratorC does the same (`_resolve_exact_layout`). So the thunk
  reads a Julia value at DWARF offsets (`_dwarf_member_offsets`), never at offsets
  derived from alignment rules. A bitfield has no DWARF byte offset, so it is read
  where the tiled body puts it. `is_struct_packed` (member sizes summing to
  `byte_size`) admits a tiled body only when every member and every bitfield's bits
  sit where the tiling says: Itanium can sum without tiling (abi_fuzz X9,
  `test_thunk_byte_regions.jl`). A named union is `mutable`, so it embeds as a pointer;
  any struct holding one is a blob, checked by type: an 8-byte union at an aligned
  offset fits the proof's pointer measure. In GeneratorCpp a field takes a type only when
  the proof can measure it and it has the member's DWARF size (`_member_field`);
  otherwise the field is the member's bytes. A member with no offset (a static,
  `__clang_vtable`) is not a field. A forward declaration never reuses an emitted
  enum's name: Julia 1.12 lets `struct X end` replace `@enum X`.
- **A named union or bitfield record moves by value only through its bytes.** Both
  are a `mutable struct` whose one field is the C bytes.
  - `_invoke_call` allocates one for a Tier 2 sret return. `Ref{T}()` is an
    undefined reference for a mutable `T` (abi_fuzz X1).
  - An accessor builds one from bytes (`jl_byte_records`), never with
    `unsafe_load(Ptr{T})`, which reads C's bytes as a reference (X5).
  - A bitfield accessor gathers exactly the bytes the field spans, into an integer
    wide enough for them (X6).

  (`test_cpp_mutable_records.jl`.)
- `!jlcs.c_struct` must never appear inside an `!llvm.struct` body; inline the literal.
- `jlcs.marshal_arg`'s `memberTypes` are the packed body's own fields, read off
  that type. Spelled from the C types, a record, enum or array-of-records member
  does not type-check, and the module fails to translate, taking every thunk in
  the library with it (`test_jlcs_producers.jl` §H).
- **A by-value record crosses a thunk with its layout, as nothing, by address, or not
  at all.** Nothing means an empty class, which SysV gives no class. By address means
  DWARF `DW_CC_pass_by_reference`, and such a return goes through an sret slot, first
  argument, before `this`. Not at all means an ABI trap on the Julia side.
  `FunctionGen.by_value_crossing` is the one verdict; the thunk generator and
  GeneratorCpp both act on it. `record_abi` in metadata carries the DWARF facts. A
  record must never fall to `!llvm.ptr`: that is how empty classes, declared-only
  types, `string_view` and every template spelling were mis-passed.
  (`test_by_value_crossing.jl`.)
- **A byte region never crosses a thunk as an array.** A record with no
  field-by-field body (a union, a bitfield record, overlapping members) is
  `!llvm.array<N x i8>` to StructGen, and LLVM splits an array argument into one
  register per element (abi_fuzz X2). By value it crosses as `_byte_blob_type(N)`,
  a packed struct of the same bytes, and its SysV class comes from the record's
  facts, not the bytes (`sysv_record_abi`): a float union of 16 bytes or less,
  once refused as unclassifiable, is INTEGER like C's. (`test_thunk_byte_regions.jl`.)
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

### Subprocesses

- **The toolchain environment rides on the `Cmd`, never on `ENV`.**
  `LLVMEnvironment.llvm_cmd` attaches it for `BuildBridge.run_command`, `run_tool`
  and Discovery's include scan. `run_command` runs concurrently (`Threads.@threads`
  for the IR compile, `asyncmap` for objects). Overlapping `with_llvm_env`
  save/restore cycles deleted the user's `LD_LIBRARY_PATH`/`CPATH` after a parallel
  C++ build. `with_llvm_env` is lock-serialized but still global, so it must not be
  called anywhere in `src/`. A key the toolchain maps to `""` is a force-clear: the
  child must get it empty, not inherit the parent's value. On Windows the merge is
  case-insensitive (`_overlay_env`). (`test_llvm_env_isolation.jl`.)

### Caching

1. Per-file IR cache (`.replibuild_cache/`), keyed on a compile-fingerprint `.key`
   sidecar. The hash in the IR filename identifies the source, not the flags.
2. Project content hash (`.replibuild_cache/project_hash`): TOML, sources, include
   dirs (recursive), git HEAD, and the toolchain (`_toolchain_identity`: Julia's
   libLLVM for C, the system clang for C++, plus the triple). A Julia switch
   rebuilds a C project and leaves a C++ one built.

Every persisted key (these two, the registry's `hash_config`, the IR file name's
directory digest `_ir_dir_key` and the templates TU's `.key` fingerprint) is
SHA-256 fed through `_key_feed!`, never `Base.hash`, which differs between Julia
releases.
3. Registry build cache (`~/.replibuild/builds/<hash>/`, gates `use()`). It includes
   `_generator_fingerprint`: RepliBuild version, HEAD, **and a dirty-tree digest**, so
   uncommitted generator edits miss. `_store_build` copies directories.

`RepliBuild.VERSION` is derived from `Project.toml`; never reintroduce a literal. It
feeds the fingerprint and `BUILD_GENERATOR`. To check what is registered in General,
read `R/RepliBuild/Versions.toml` from the tarball that `path =` in
`~/.julia/registries/General.toml` names: `General.tar.gz` (`tar -xzOf`) or
`General.tar.zst` (`tar --zstd -xOf`), depending on the Pkg version. A tarball
beside it that `path =` does not name is stale and lacks recent versions. 3.3.2
is a permanent hole there. Registration is manual and
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
  with a generator script that refuses to write a vacuous fixture. Hand-run tools live
  in `test/tools/` (`gen_receiver_corpus.jl`, `gen_thunk_symbols.jl`,
  `ab_virtual_dispatch.jl`) and are the only files allowed to touch the Hub. Hub
  rebuilds are the release integration test; RepliBuild-Hub says when to run one.
  `test/tools/abi_fuzz.jl` (unwired, touches no Hub) generates a C unit per seed,
  wraps it as C and as C++, and checks both tiers against the compiler's own
  `sizeof`/`offsetof` and the values each callee records. Its header says how to
  run it and how to minimise a failure.
- **Guard against vacuous green.** Assert that the loop actually ran
  (`checked == expected`). Never wrap corpus loading in a bare `try/catch`.
  Negative-check a new guard by neutering the fix, using the exact method signature
  (a less specific override loses dispatch and proves nothing).
- **Never `exit()` in a file a suite includes.** It ends devtests with a success status
  and silently skips everything after it. A file that needs libJLCS or a tool wraps
  its body in `if requires("what", :libJLCS, :clangxx) … end # requires`
  (`test/support/TestSupport.jl`), which records a skipped testset. `runtests.jl`
  refuses an include-time `exit` on the parsed AST. A `struct` a `ccall` names must be
  declared **above** that block: the block is one top-level expression, and `ccall`
  resolves its types before it runs.
- **A crash or a hang runs in a child**: `run_isolated` (SIGKILL timeout, output
  through a file, echoed on failure). A probe script prints one `PROBE <name>: PASS|FAIL
  …` line per case and `PROBE_DONE`; assert on `probe_verdicts`.
- **Output is `@test`, not prints.** A failed `@test` inside a testset does not throw,
  so a `println("✓ …")` after it prints on failure too. `@info` only for what the
  summary cannot show (which platform arm ran).
- devtests aborts a file at the first failing top-level testset, so one red hides an
  unknown number of unrun tests. Put cheap probes early.
- Assert **deltas** on process-global state (JIT engines, fixture counters), not
  absolutes, because files share one process. Assert **names**, not counts, downstream
  of static promotion.
- A background command's exit code is the wrapper's. Write `echo "EXIT=$?" >> log` and
  read that.
- A test that states one host's answer instead of the invariant goes red on the other
  host (Linux vs Windows). Drive host-conditional code with both spellings.
- A regression check for a generator change is the change's own regression test plus
  runtests and devtests. For ABI work, also run `test/tools/abi_fuzz.jl` on the same
  seeds before and after.
  - Re-wrap a Hub package and diff it against a wrapper from unpatched HEAD only when
    the change exercises that package, or before a release. pugixml has full debug
    info; box2d also works.
  - The re-wrap should be byte-identical apart from timestamps.
  - For dialect changes, compare machine code, not IR text.
- Fixtures: `c_test/`, `stress_test/`, `stl_test/`, `mi_test/`, `vi_test/`,
  `callback_test/`, `c_abomination_test/`,
  `convenience_overload_test/`, `aot_test/`, `slice_test/` (Tier 1) are
  discover-driven projects. Fixture tomls are regenerated by devtests'
  `discover(force=true)`, with `CURATED_FIXTURE_CONFIG` re-seeded after
  (`test_aot_thunks.jl` sets its own `aot_thunks = true` the same way).
  `abi_nested_test/` and `c_sret_test/` are projects whose test writes the toml
  and runs build + wrap.
  `struct_abi/`, `jlcs_producers/`, `mlir_templates/` and `jit_unwind/` are
  single-file clang++ fixtures their test compiles directly; `fixtures/` holds
  the vendored JSON corpora.

## Windows (`x86_64-w64-windows-gnu`, MSYS2 CLANG64) — since v4.0.0

Linux and Windows are supported; macOS is refused at load (no AAPCS64 classifier).
Setup and full findings: `docs/updates/WINDOWS_PORT.md`. The rules that bite code
changes:

- `check_environment` judges the MSYS2 environment by where `clang++` lives, never
  by `MSYSTEM`: Julia usually runs outside any MSYS2 shell, and Git Bash sets
  `MSYSTEM=MINGW64`. A `libJLCS.dll` that will not load is a missing
  `clang64\bin` PATH entry as often as a gone LLVM (`_clang64_bin_off_path`).
  JIT C++ wrappers run `llvm-dwarfdump` by its bare name at load
  (`DWARFParser.parse_vtables`), so the doctor probes that name.
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
- AOT thunks reach libJLCS: on ELF the thunks library links it (the exception
  buffer); on PE it imports only the wrapped DLL and the CRT, and libJLCS is the
  caller's guard (below). `JITManager.open_thunks_library` opens libJLCS by
  absolute path first. **Never vendor a second copy** — two exception buffers swallow
  every C++ exception.
- **Thunks do not catch on Windows**, JIT or AOT: the dialect lowers `try_call` to a
  plain call there, and libJLCS's `jlcs_guard_*` is the frame that catches. Call a
  thunk only through `JITManager._invoke_call` / `_invoke_void`; a bare
  `ccall(fptr, …)` lets a throw unwind into Julia and ends the process
  (`test_aot_thunks.jl`, probe `throw_void`).
- **Two C runtimes in one process.** Julia links `msvcrt.dll`; RepliBuild-built DLLs
  allocate from the UCRT. Never `Libc.free` library memory, in generated code or
  in a Hub test: wrappers define `libc_free` (bound via `Compiler._pe_crt_dll`).
  Generated code that `Libc.malloc`s storage must also be the code that frees it.
- Per-host config lives in `[platform.<host>]`, merged by `apply_platform_overlay`
  inside `load_config`. Anything else that reads a replibuild.toml for build inputs
  must go through it too (registry staging does). Keep Hub bases correct for Linux
  and override on Windows: older RepliBuild ignores the table.
- Relative `[compile]` paths resolve against `root` at load; `save_config` writes
  paths under the project relative with `/`. Never write a machine path into a
  tracked TOML.
- Windows filesystems fold case. Any file named after a C identifier goes through a
  case-safe name (`Slicer.slice_file_stem`); `LUA_YIELD` and `lua_yield` collided.
- PE has no RUNPATH. `[link] link_dirs` DLLs are opened before the library, at the
  build's load check and in the wrapper's `__init__` (`_link_dir_dlls`,
  `_link_dir_preload_snippet`).
- A command line holds 32,767 characters. A command that takes one argument per
  translation unit goes through `_args_with_response_file`, which passes a long
  input list as `@file` (llamacpp + Vulkan, 384 TUs, needed ~39,000). llvm-link
  reads the file by Windows rules and clang by GNU rules, where `\` escapes, so
  inputs are written quoted with `/`. A program that never started returns
  `BuildBridge.SPAWN_FAILED`, not 1: check it before blaming the tool
  (`test_response_files.jl`, devtests §19).
- `FILE` is `struct _iobuf`. Bake paths into generated Julia with `repr()` and into
  TOML with `TOML.print`. Read JSON only through `_read_json` (`JSON.parsefile` mmaps
  on 0.21, which blocks deletion, and rejects `use_mmap` on 1.x;
  `test_json_mmap_hygiene.jl` enforces it). `_canon_path` is identity off Windows.
  `/dev/null` passed to `run` is not a path.
- Before believing any "open on Windows" note, check `git log -S` on the symptom.

## Fresh clone

Verified 2026-08-26 by cloning to a tempdir: `Pkg.instantiate`, `runtests.jl` green,
`build.sh` builds, `check_environment` reports tiers. `build.sh` gates on LLVM ≥ 21,
`mlir-tblgen`/`llvm-config` major match, and `lib/cmake/mlir` presence. Redo the clone
check before claiming portability again; reading `.gitignore` does not settle it.

## Open problems

- **C ccall coverage**: 98.6% (5368/5444, 2026-08-29), target 99%. The 76 misses:
  43 varargs (declaration work via `[wrap.varargs]`), 27 mpack by-value aggregate
  returns (typed correctly but emitted as ABI-trap stubs), 5 mpack by-value aggregate
  args, and 1 counting artefact (zlib `gzgetc`). Measure `ccall` targets, not emitted
  names; a trap stub is not coverage. The mpack stubs are SysV-only since
  2026-10-18: Win64 calls them.
- **Over-aligned records, what the C sret fix does not cover** (2026-10-18).
  GeneratorCpp's own ccall path has no aligned buffer; only a record-level
  `alignment` keeps a function off it, so a member-level `alignas` record can
  still get a misaligned sret there. Nor does GeneratorC's `[wrap.varargs]`
  path, which skips BRANCH 0 altogether. An alignment only a typedef carries
  (`typedef struct S T __attribute__((aligned(16)))`) is not in clang's DWARF.
  **SysV byval parameters of 16-aligned MEMORY records are wrong when they
  follow another stack argument** (run 2026-10-24, Linux, Tier 3 C:
  `take2(Big24, D4A)` with `D4A` `aligned(16)` of four doubles reads the record
  from the next 16-aligned slot after the 24-byte argument, Julia's ccall placed
  it 8 bytes earlier; `take1(D4A)` and `take3(Big24, Big24, D4A)` pass because
  their slots happen to be 16-aligned; fixture in the gitignored
  `.claude/albig-2026-10-24/`). GeneratorC's parameter traps stop at 16 bytes
  and never read alignment; `_c_record_align` reaches returns only. Tier 2
  passes and returns 16-aligned records at 16 (`test_sysv_live.jl`); Win64
  parameters are safe (LLVM's `byval` lowering puts the copy at 16).
- `.replibuild_cache/slices/<modkey>/` accumulates unboundedly (harmless).
- **The `[wrap.varargs]` path skips every SysV by-value guard** (2026-10-24).
  GeneratorC's varargs interception `continue`s before `blob_abi_offenders`,
  BRANCH 0's sret handling and the x87 trap run, so a variadic function whose
  fixed parameters include a packed or float-bearing record of 16 bytes or less
  gets a plain `@ccall` with the record spelled by name. Hub varargs are
  printf-style with scalar fixed arguments, so nothing shipped hits it.
- **`abi_fuzz.jl` registered every unit into the default Julia environment**
  (2026-10-24). It set `REPLIBUILD_HOME` but not `REPLIBUILD_JULIA_ENV`, and
  4.3.0's `discover(build=true, wrap=true)` registers, which `Pkg.develop`s the
  package into `@v1.13`: 180 path deps into a tmpfs scratch directory after one
  `--seeds=1:60` run. Removed with `Pkg.rm`; the tool now sets the variable to
  `none`. Anything that calls `discover(…, wrap=true)` from a test or tool must
  set both variables; a `REPLIBUILD_HOME` that is not `~/.replibuild` with a
  default env is the shape to refuse.
- **Distinct types sharing a bare name keep one definition between them**
  (2026-10-23). It is marked `conflict` in the metadata, and the build names them
  (llama.cpp: pimpl `impl`s, `input_decision`, `llm_graph_input_kpool`). A by-value
  crossing of one is an ABI trap, but every other reference by that name gets the
  kept layout: a `Ptr{T}`/`Ref{T}` parameter, a method's receiver, a global, a
  field of another record. No wrapped Hub function reached one by value or those
  ways when measured. Disambiguating needs scoped names through `resolve_type`, the
  generators and the thunks, and C's case (two `.c` files, two `struct state`s) has
  no scope to qualify by.
- **Registered 3.0–4.1.x still allow LLVM.jl 10 in General** (`["3 - 4.1"] LLVM =
  "9 - 10"` in `R/RepliBuild/Compat.toml`), although nothing before 4.2.1 handles
  it: their Tier 1 code makes the reads 4.2.1 fixed. 4.2.0 is capped at `"9"`.
  4.2.1 was never registered; its fixes ship in 4.3.0. Once 4.3.0 is registered
  the resolver prefers it, so the old versions are reached only when another
  package holds RepliBuild back. Capping them is a
  General PR (outward-facing, ask first).

### Confirmed defects — 2026-09-26 audit

Re-reproduced from clean builds of library-free fixtures (runner and fixtures in the
gitignored `.claude/audit-2026-09-26/`, `run_all.sh`; each line prints CONFIRMED,
GUARDED or NOT REPRODUCED). Fixed, with the story in CHANGELOG: F1–F3 and F6–F28.
A fix that lives in metadata reaches a Hub package only on a **rebuild** (delete
`.replibuild_cache/project_hash`), not a re-wrap. Every fixed finding has an
in-repo regression test, negative-checked by neutering its fix; the harness is no
longer the only proof. Still open:

- **A by-value type only declared in this binary** (F4, constructor homing) is now an
  ABI trap instead of a SIGSEGV, but the function is still uncallable until the package
  builds with `-Xclang -fno-use-ctor-homing` or `-fstandalone-debug`. Neither is a
  default yet (see Not Yet Built).

## Not Yet Built (roadmap, not bugs)

Absence is the default state on this project. A missing feature is an unbuilt piece;
only something contradicting what **is** built is a bug. Log new gaps here, dated, and
don't file them as defects. Never extrapolate a narrow entry into "X is unbuilt" —
read the generator output first. (User-facing C++ RAII — `Managed<T>` finalizers with
DWARF-resolved destructors — **is** built.)

- By-value toolchain types (`std::string_view`, `back_insert_iterator`, …) have ABI
  facts in `record_abi` but no layout, so their functions are ABI traps. Keeping the
  layout of records reachable by value would make them callable (2026-09-27).
- Default `-Xclang -fno-use-ctor-homing` for C++ builds. It restores the definition of
  header-only value types at a fraction of `-fstandalone-debug`'s size, but changes
  every C++ package's metadata, so it needs a Hub-wide measurement first.
- Array-view Julia-side accessors for the rank-1 thunks, plus rank ≥ 2 members.
- `is_struct_packed` over-classifies padding-free structs (wasted work, not wrong).
- Op verifiers for `ffe_call`, `try_call`, `load/store_array_element`, `ctor_call`,
  `yield`, `marshal_ret`. A computed verifier-coverage guard would also help.
- In-IR virtual-base upcast op (no producer needs one yet).
- vcall for struct-shaped virtual signatures (these keep static dispatch).
- AAPCS64 classifier (macOS/ARM).
- Codegen `.ll → .so` still shells to clang; not internalised.
- STL factories size containers from `get_stl_container_size`'s table. GeneratorCpp's
  DWARF `byte_size` lookup never matches: STL records are not in
  `struct_definitions` (0 hits across 57 metadata files, 2026-10-02).
- Template parameters from GNU binutils dumps. binutils spells the tags
  `DW_TAG_template_type_param` and `DW_TAG_template_value_param`, which the
  parser's `…_parameter` match never sees, so `template_params` is always empty
  (2026-10-23).
- STL members that live in the C++ runtime library are unbound: libc++'s
  `basic_string<char>` extern-template members on Windows (`libc++.dll` exports
  them), libstdc++'s on Linux (undefined in the wrapped library). The JIT searches
  only the wrapped library. Binding them needs the engine to also search the
  library's runtime dependencies (2026-10-16).

## Maintaining this file

- Keep it lean. When you fix something, put the story in `CHANGELOG.md` under
  `## Unreleased`, and add a line here only if it creates a rule a future session must
  follow.
- Derived claims (counts, line numbers, versions, test totals) are what rot. Prefer
  naming the test that computes them over writing the number. Avoid `file:line`
  citations; name the function.
- The file is tracked: no machine-local paths.
