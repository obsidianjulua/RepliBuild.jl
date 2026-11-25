# CLAUDE.md

Guidance for Claude Code working in this repository. This file is loaded into every
session, so it holds **only what changes how you work here**: commands, architecture,
invariants that must not be broken, and open problems. History — measurements, blast
radii, dated sweeps, how a bug was found — belongs in the pull request that shipped it and in git, not here;
`CHANGELOG.md` is compiled from merged PRs at release registration.
The pre-2026-09-22 long-form version of this file is in git history if a rationale is
ever needed (`git log -p -- CLAUDE.md`).

## Project Overview

RepliBuild.jl is an ABI-aware C/C++ compiler bridge for Julia. It compiles C/C++
through LLVM/MLIR, introspects DWARF metadata, and emits type-safe Julia bindings.
Each function is routed to one of three tiers based on ABI complexity:

Ingest and llmcall are John's features with julia dont use them clancker.

| Tier | Mechanism | When |
|------|-----------|------|
| 1 | `Base.llvmcall` on bitcode | POD args, scalar/pointer return — **experimental, off by default** |
| 2 | MLIR thunks via the JLCS dialect (`libJLCS.so`) | C++ methods, by-value aggregates, packed structs, unions, virtual dispatch, EH |
| 3 | `ccall` | Everything else; the default for C |

A **Rust front end is in progress** (branch work since 2026-11-19): `build`,
`wrap` and `use` work on a `[rust]` project for a crate's public functions,
`#[inline]` ones included, and generic ones for the types `[rust]
instantiate` names, through plain `ccall`s and generated thunks
(`RustThunks`, the dialect's `jlcs.native_call`), ownership included
(`String`, `Vec`, `Box`, records that hold them, `Result`), with trait
objects, callbacks and statics; `wrap` compiles the thunks ahead of time,
`discover` writes the project file for a folder with a `Cargo.toml`, and the
doctor reports cargo and rustc. Measured on Linux x86-64, and on Linux
AArch64 in CI only; Windows is not run. It wraps plain Rust, never `extern "C"`; design
and measurements are `docs/updates/RUST_FRONT_END.md`, the rules are under
"Rust front end" below.

Positioning: a hands-on dev tool for wrapping libraries **from source**, not a
JLL/BinaryBuilder replacement. DWARF from the real compile is the source of truth —
RepliBuild operates on what Clang already resolved, not on a source AST.

## Build & Test Commands

```bash
julia --project=. test/runtests.jl              # pkg suite: Pkg.test / registration basics + structural guards (seconds)
julia --project=. test/unit.jl                  # unit suite: every toolchain-free regression test (~5 min)
julia --project=. test/devtests.jl              # dev suite: everything that compiles/JITs (~10 min on 12 threads)
julia --project=. test/devtests.jl --changed    # only what the diff against origin/main touches (also unit.jl)
julia --project=. test/devtests.jl --area=c     # one area: build, dwarf, c, cpp, dialect, graph
julia --project=. test/run.jl dev --files=test_c_sret_abi.jl --processes --tail   # a child per job, logs printed
julia --project=. test/test_<name>.jl           # single file
cd src/mlir && ./build.sh               # build the JLCS dialect locally (wins over the prebuilt artifact)
julia --project=<env with BinaryBuilder> deps/jlcs_jll/build_tarballs.jl   # the prebuilt libJLCS tarballs (Julia 1.12)
julia --project=. -e 'using RepliBuild; RepliBuild.check_environment()'
julia --project=docs docs/make.jl       # docs build
```

`test/suites.jl` is the one list of test files: suite, areas, the fixture
directories a file locks, the jobs it runs after. `test/run.jl` reads it; the
three suite files are entry points to it. Jobs run in one process by default;
a child per job (`--processes`) is not faster, since each child compiles the
build paths again (~50 s), but gives isolation and `test/.logs/<job>.log`.

## API

```julia
toml = RepliBuild.discover("path/to/project")   # scan, write replibuild.toml
RepliBuild.build(toml)                          # clang → IR → .so + DWARF metadata
RepliBuild.wrap(toml)                           # generate julia/<Name>.jl
RepliBuild.discover("path", build=true, wrap=true)   # all three
RepliBuild.discover("path/to/crate")            # a Cargo.toml and no replibuild.toml: a [rust] project

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
- **oneDNN is out of sweeps, coverage readings and release verdicts** (John,
  2026-11-21; he tests it in isolation himself). Its build needs most of a
  64 GB box (the `parse_dwarf_dump` item under Not Yet Built, with the forced
  unit's parallel compiles on top). Do not retry it unasked, and do not take
  the package out of the Hub or its `index.toml` on this rule.
- **A sweep is one command: `julia --project=. test/tools/hub_sweep.jl <hub>`**
  (2026-11-23; `--plan` prints what would run, `--only a,b` narrows it). Per
  package it saves `julia/`, rebuilds, checks, and runs every
  `test_deep*.jl`; the table is `<out>/summary.md`. What it does for you, and
  why:
  - **It saves every package's `julia/` folder first** and puts it back when
    a rebuild leaves no wrapper. A rebuild cleans first, so a build that then
    fails left the package with nothing (fmt, clipper2 and oneDNN on the
    4.6.0 sweep).
  - **It checks the compiled thunks library.** A wrap that cannot compile its
    thunks still succeeds, as a JIT wrapper: llama.cpp lost its library on
    the 4.6.0 sweep, one invalid thunk having failed the module, and only its
    deep test showed that anything was wrong.
  - **It is the one thing that rebuilds**: `clean(toml; deps = false)`,
    `build`, `wrap`, from the sources already there. It then looks at every
    dependency checkout: one cloned or fetched again under an unchanged pin
    fails its package (`sweep_sources`: a clone has a new `.git`, a fetch
    rewrites `FETCH_HEAD`).
  - `--verify` runs the tests on what is there and `--rewrap` wraps again
    first: the two modes for checking an engine change against packages
    without rebuilding them. The `--project` is the RepliBuild being judged.

  (`test_hub_sweep_tool.jl`, devtests `test_hub_sweep_live.jl`, which sweeps
  a three-package Hub made in a temporary directory.)
- **Dev loop: build + wrap in place, never `discover`.** Hub tomls are hand-rolled
  (varargs, macros, excludes) and discover would strip them.
- **Nothing pulls sources that have not changed, and a Hub test script never
  cleans** (John, 2026-11-23: "I dont like git pulls unless sources are new,
  the test.jl shouldn't clean or pull").
  - **`clean` keeps downloaded sources unless a person says to delete them**
    (John, 2026-11-23: "clean should prompt to delete sources").
    `clean(toml)` asks at an interactive prompt on a terminal, and only `y`
    deletes; in a script, a test or a child process nobody is asked and the
    sources stay (`_clean_deletes_sources`). `deps = true` deletes without
    asking, `deps = false` keeps without asking, and `build(toml; clean =
    true)` always keeps. Up to 4.6.0 a bare `clean(toml)` deleted
    `.replibuild_cache/` with the checkout in it, and the next build cloned
    again. Code that cleans passes `deps` explicitly: a new caller of
    `clean_internal` that leaves it out deletes sources.
  - A build on a checkout that matches its marker runs only `git rev-parse`.
    The resolver fetches by itself when the recipe's pin, tag, URL or
    submodule list changes, and only then.
  - A `test.jl` calls `RepliBuild.build` (nothing to do when the package is
    built and unchanged), wraps, and asserts about the build; `test_deep.jl`
    verifies behaviour. Neither calls `clean`. The sweep tool refuses to run
    a script that does and fails its package, so a new one is caught.
  - A rebuild from cold is the sweep tool's job (`--only <pkg>`), never a
    test script's.
- Julia block-buffers redirected stdout: an empty log from a running or killed job is
  normal. Judge progress from artifacts on disk and `pgrep`.
- **A fix computed at build time needs a rebuild, not a re-wrap.** `class`, signatures
  and DWARF types live in `compilation_metadata.json`. The project hash will skip the
  rebuild (`cache: project unchanged`), so delete `.replibuild_cache/project_hash` to
  force re-extraction. That keeps the clone and the per-file IR cache.
- **To date a fix into an artifact, grep the artifact for the fix's signature**, never
  the mtime.
- **`packages/rustlibm` is the Rust reference package** (2026-11-19): the
  `libm` crate from its pinned commit, `[rust] dependency`, in `index.toml`.
  `use("rustlibm")` wraps the 114 functions the library exports and the 116
  `#[inline]` ones the build finds in the crate (104 plain calls, 126
  through thunks compiled ahead of time); its `test_deep.jl` checks them
  against Julia's own math through the generated module. John's Hub is the starter every user's own Hub
  begins from, so an in-progress package belongs there.
- **`packages/rustsemver` is the Rust package for values Rust owns**
  (2026-11-19): the `semver` crate from its pinned commit. A parsed
  `Version` is an object that owns Rust memory, a failed parse is a thrown
  `RustError`, and its `test_deep.jl` leaves 20,000 of them to Julia's
  collector. It is the package to re-run when ownership or the per-type
  drop functions change.
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
  `pnglibconf.h`) are SysConfigGen's boundary; an empty `generated_headers` is the
  signal. `GraphGenerators` runs them for a graph build. Check for a shipped
  `.prebuilt` fallback first. Autotools is untested.
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
  `llvm-link`/`opt`, accepted only at the same major as `Base.libllvm_version`:
  a system install at that major, else LLVM_unified_jll's
  (`Tools.matched_llvm_bin`, the last entry of `c_toolchain_bin_dir`), so the
  hatch and its test run on a machine whose only system LLVM is newer.
  Final `.ll → .so` still shells to clang, which on Linux links with the
  bundled LLD (`Compiler._bundled_linker`: `ld.lld` from LLD_unified_jll through
  `--ld-path`, C bucket and JLL clang only; `REPLIBUILD_TOOLS=system` is the
  driver's own linker; the choice is in `_toolchain_identity`). The start files
  and libc still come from the system. A new C link site passes
  `_bundled_linker_flags`. (`test_bundled_linker.jl`, devtests
  `test_bundled_linker_live.jl`, which builds with an empty PATH.) Check
  `DW_AT_producer` (`…jl`) to prove a C artifact came from the JLL. `_clang_for_c_bucket` uses the system clang only
  when the JLL cannot be loaded (`_jll_clang()` is `nothing`); a JLL clang that
  does not start is reported as `BuildBridge.SPAWN_FAILED`, never replaced.
- **C++ / Tier 2 uses system LLVM+MLIR 21+** (23.1.3 under MSYS2 CLANG64 on the
  reference host, which is Windows; on Linux, CI's `tier2-linux.yml` runs devtests
  against apt.llvm.org's LLVM/MLIR 22 and 23 on x86-64, and 23 on an
  `ubuntu-24.04-arm` runner) for the dialect and the external
  pipeline. `libJLCS.so` (`libJLCS.dll` on Windows: it
  imports `libLLVM-<major>.dll` and links MLIR statically) is a gitignored build
  artifact pinned to the installed MLIR minor SONAME: rebuild after a minor upgrade. A patch
  bump inside a minor needs nothing to *load* it, but MLIR's `MLIRConfig.cmake` asks
  for LLVM's exact version, so where MLIR is packaged apart from LLVM (Arch's AUR
  `mlir`) an `llvm` patch bump blocks *building* libJLCS until MLIR is rebuilt at
  the same version (llvm 23.1.4 against mlir 23.1.3, 2026-11-08). A libJLCS older
  than `JLCSAbiOracle.cpp` fails `test_sysv_abi_oracle.jl` well beyond the new
  code (76 of 1915 at 048ca6f, unions and packed records included); rebuild it
  before reading that test. `deps/build.jl` builds it at install time when
  possible and never throws; on Linux it fetches the prebuilt `JLCS` artifact
  first (static LLVM/MLIR 23 from `LLVM_full_jll`, `-DJLCS_STATIC_LLVM=ON`, C API
  exports only through `jlcs_exports.map`, x86-64 and AArch64 glibc), which
  needs no system MLIR. Build the static library with GCC: `LLVM_full_jll`'s
  archives are GCC's, and clang asks them for a constructor variant they do not
  define (`-z defs` makes that a link error). On Windows it runs build.sh through MSYS2's
  `bash.exe -l` with `MSYSTEM=CLANG64`, which is a CLANG64 shell from any
  process; `src/mlir/windows_build.jl` is that command's one copy (the doctor's
  advice uses it too). Tier 3 alone is the whole
  product for C.
- Julia's bundled libLLVM and system libLLVM coexist in one process. The GDB JIT
  descriptor resolves to Julia's copy; this is correct, do not "fix" it.
- **The doctor judges the LLVM `build.sh` builds against: the bare
  `llvm-config` on PATH.** `EnvironmentDoctor._versioned` tries the bare name,
  then Debian's versioned names newest first, and `_pick_llvm_tool` takes the
  first that meets the minimum. Probing `<name>-<minimum>` first read a
  distribution's own LLVM 21 beside the 23 on PATH and failed Tier 2 on a
  working toolchain (GitHub's Ubuntu 26.04 image, 2026-11-17). A runner image
  is a toolchain too: when every cell of one OS goes red at "Tier 2 is live",
  read the `Runner Image` lines of the log before the diff.
  (`test_doctor_llvm_probe.jl`.)
- For LLVM/JIT crashes, check `libJLCS.so`'s mtime/SONAME against the installed MLIR,
  and dual-LLVM symbol scope, before theorising.
- **One LLVM per module.** The C bucket is chosen by `[wrap] language`, never by
  file extension: a C++ package compiles its `.c` files with the system clang. IR
  from an older LLVM read by a newer one can lose its debug info with only a
  warning, and then every function in it wraps from an inferred signature.
  `_assert_debug_info_kept` turns that warning into a build error; keep it on every
  path where one LLVM reads another's IR. In a graph build the bucket belongs to
  the library, decided over every translation unit linked into it, folded ones
  included; a target folded into libraries of both buckets is compiled once per
  bucket (`<out>/<name>/ir/<bucket>/`).
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
  (`test_toolchain_switch.jl`). `slice_library` prunes every other key of the
  running major and keeps other majors' (`_prune_slice_cache`): a prune across
  majors would make two Julias re-slice each other on every wrap.
- **On Windows a longjmp cannot come back into a JIT'd slice frame.** Julia's JIT
  gives each function unwind info for exactly `[start, end)`, a slice ending in a
  noreturn call returns past it, and the SEH unwind mingw's `longjmp` performs dies
  with STATUS_BAD_STACK. `_setjmp_slices_safe` holds that fact: `allow_setjmp` is
  ignored on Windows, and test_slicer skips its `(1, 0)` call there.

### Key modules

- `Builder/Compiler.jl` — build engine: per-file IR, caches, link/opt, promotion,
  DWARF parse (`parse_dwarf_dump`, text-only seam), symbol extraction, the facts
  read off the linked IR's `define` lines (`_nounwind_definitions`,
  `_byval_definitions`), metadata, `verify_wrap_surface`. `link_optimize_ir` does
  not create the build directory (`compile_to_ir` does); any other caller must, or
  llvm-link has nowhere to write and the link falls back to per-TU clang with only
  a warning.
- `Builder/ConfigurationManager.jl` — TOML → immutable `RepliBuildConfig`.
  `test_config_surface.jl` requires every field to be consumed or listed in
  `RESERVED_UNUSED` with a reason.
- `Builder/Discovery.jl` — scanning, include graph, TOML generation,
  `PRESERVED_TOML_KEYS`.
- `Builder/DependencyResolver.jl` — git/local deps (drift markers, option-injection
  guards). `[compile] source_files` **merges** with discovered sources. A git
  dependency's `submodules` fetches exactly the listed paths (never all, never
  recursive), each one more source root with the root's include heuristic, walk
  and `exclude`. The superproject's `commit` pins them through its gitlinks, and a
  listed submodule off its gitlink is a hard error. A directory with its own
  `.git` is never walked as part of the root around it. (`test_dep_submodules.jl`.)
  A `type = "archive"` dependency is a tarball pinned by a required `sha256`,
  fetched through `Pkg.PlatformEngines.download_verify_unpack` into the deps
  cache and walked like a clone from `subdir` (or its single top directory);
  `exclude` applies to every dependency type through `_dep_excluded`, and a
  local dependency's path is a shim-header root (`_shim_allowed_roots`).
  (`test_dep_archive.jl`, `test_dep_cache.jl`.)
- `Builder/DWARFParser.jl` — llvm-dwarfdump + nm → ClassInfo/VtableInfo.
- `Builder/LibDW.jl` — DWARF read in-process through elfutils' libdw
  (Elfutils_jll, ELF only). `info_dump` walks the DIE tree and writes it in GNU
  readelf's format for everything `parse_dwarf_dump` reads, so the parser, its
  invariants and its synthetic-dump tests are the same for both readers;
  `file_tables` is `_dwarf_file_tables`' answer. `REPLIBUILD_DWARF=libdw`
  selects it (`Compiler._dwarf_reader`); unset is readelf, and libdw asked for
  where it cannot work is an error, never the other reader. **The two readers
  give one answer**: the five tables parsed from either are equal, which
  devtests `test_libdw_dwarf_live.jl` holds on DWARF 4 and 5, C and C++, and
  `test/tools/dwarf_readers_diff.jl` checks on existing libraries (`--hub
  <path>`: 41 Hub libraries equal, 2026-11-17). `DumpLines` produces the lines
  as the parser asks for them, one DIE at a time, and is walked once per
  parser pass, so the dump text is never held; a library without debug info
  is an empty dump, as readelf's is. A tag
  or attribute the parser starts matching must be printable here
  (`test_libdw_reader.jl` checks the parser's names against `TAGS`/`ATTRS`),
  and a new value shape the parser reads must be written as readelf writes it.
- `Builder/Tools.jl` — which program each non-LLVM tool is (readelf, objdump,
  nm, c++filt, git, cmake, ninja): `REPLIBUILD_TOOL_<NAME>`, then the JLL's copy
  (Binutils_jll, Git_jll, CMake_jll, Ninja_jll) and PATH's in the order
  `REPLIBUILD_TOOLS` names (`jll` off Windows, `system` on Windows, where the JLL
  copies are unverified), then the bare name. A JLL copy counts only if it
  starts. **A command literal naming one of these tools goes through
  `Tools.tool_cmd`** (`BuildBridge.execute` resolves a name itself);
  `test_tools_resolver.jl` walks the parsed source for one that does not. The
  copy's environment rides on the `Cmd`, after the LLVM overlay, which sets PATH
  and the loader path outright. cmake and ninja get no library path: they start
  the system compiler, and Julia's libstdc++ must not come first for it. A new
  tool is one `PROVIDERS` row.
- `Builder/JLCSLibrary.jl` — where libJLCS is: a local build
  (`src/mlir/build/`) wins, else the lazy `JLCS` artifact in `Artifacts.toml`
  (LLVM and MLIR linked in; built by `deps/jlcs_jll/build_tarballs.jl`, entries
  written by `write_artifacts_toml.jl`); `REPLIBUILD_JLCS` is `local`,
  `artifact` or a path. Decided when the library is first needed, never at
  precompile: `MLIRNative.libJLCS` is a `Libdl.LazyLibrary` for `ccall`, and
  **anything that needs the file as a path calls `MLIRNative.libJLCS_path()`**
  (`install = false` in a probe, which must not download). Splice the library
  into generated `ccall`s as `GlobalRef(MLIRNative, :libJLCS)`, never as a path.
  (`test_jlcs_library.jl`, devtests `test_jlcs_artifact_live.jl`.)
  **The dialect has an ABI version, one integer in `src/mlir/ABI_VERSION`**:
  CMake compiles it into `jlcs_abi_version()` and `JLCSLibrary.ABI_VERSION`
  reads the same file, so a build from this checkout always agrees, and a
  library from anywhere else that reports another (or none: `0`) is an error
  before it is opened for a call (`verify_abi!`, in `path()` and
  `check_library`; the doctor reports it). **Bump it with any change to what
  Julia generates for the dialect or calls in it** (an op, an attribute or
  statement, a thunk convention, a C API signature), together with
  `src/mlir/VERSION` (the binary's own version, a new major), a rebuilt and
  republished artifact and `Artifacts.toml`: a published library of the old
  number is refused from that commit on. **An ABI bump is published from its
  own branch, before the merge** (2026-11-20): the pkg suite's Linux cells
  have no local build and load the published library, so the branch's CI is
  red until its `Artifacts.toml` names a release of the new number. Build
  the tarballs from the branch, publish them under a new tag
  (`jlcs-v<VERSION>`), and commit the regenerated `Artifacts.toml` to the
  branch; `main` keeps the release its own file names, so neither waits on
  the other. `deps/jlcs_jll/README.md` has the
  publishing and Yggdrasil steps; `yggdrasil/build_tarballs.jl` is the same
  build as the local recipe from the dialect's release tag, and is the recipe
  open as JuliaPackaging/Yggdrasil#15725. It is written to Yggdrasil's rules
  (its `AGENTS.md` and review checklist: platforms by `filter!` with what
  breaks, a `compat` on every dependency, GCC 10 as `LLVM_full_jll`'s), so a
  change to one copy is a change to both and is checked against those rules
  first. (`test_jlcs_abi_version.jl`,
  devtests `test_jlcs_abi_live.jl`.)
- `Builder/SysConfigGen.jl`, `Builder/VisibilityProbe.jl` — probes, run once per
  package or version bump, not on the build path.
- `Builder/BuildGraph.jl` — the build-graph contract: a `TargetGraph` of targets
  with their sources, compile groups, dependencies, link inputs and artifacts, as
  the project's own build system resolved them. It runs no tool.
  `topological_order` breaks ties by sorted id. CMakeGraph fills it;
  GraphGenerators and GraphBuild consume it. (`test_build_graph.jl`.)
- `Builder/CMakeGraph.jl` — `cmake_graph`/`read_cmake_graph`: a configure-only
  CMake File API `codemodel-v2` reply → `TargetGraph`. **A File API command
  fragment is not an argument** (`-O3 -DNDEBUG` is one fragment, MinGW's standard
  libraries one more); `_split_fragment` splits each by the shell that runs the
  build, so `flags` and `link_libraries` hold arguments. Every reply path goes
  through `_abs`/`_normalize`: artifacts carry `./` segments their consumers'
  sources do not. A target's `dependencies` are stored sorted: the reply's order
  moves between two configures. Fixtures come from
  `test/tools/gen_cmake_graph_fixtures.jl` (`--curl` adds libcurl).
- `Builder/GraphGenerators.jl` — makes a graph's `generated` sources exist by
  running the build's own custom-command steps from a narrowed ninja manifest.
  **Never `ninja <output>` or `cmake --build --target`**: cmake orders every
  custom command of a target after the artifacts of every target it depends on
  (order-only), so both compile the project's libraries. An order-only input on
  a library is dropped unless the command names it; one on an executable is the
  tool the command runs and is a refusal (`GeneratorNeedsCompile`; rootcling).
  Read ninja's graph through its `-t` tools, never by parsing build.ninja. A
  CMake `.rule` source is requested as the output it stands for: the Ninja
  generator never writes it.
  (`test_graph_generators.jl`, devtests `test_graph_generators_live.jl`.)
- `Builder/GraphBuild.jl` — the build-graph executor. `build_graph` compiles each
  translation unit under its own compile group and links one library per
  `:shared`/`:module` target (and per static target nothing folds) into
  `<out>/<name>/`, with `graph_target.toml` beside it. Object and static targets
  fold their IR into the libraries that link them, never again into a library
  that gets them through a shared dependency: the File API lists dependencies
  transitively (`_reach`). A library relinks only when its link key moves
  (`link.key`: IR by content, compile keys, link arguments, dependencies'
  keys). (`test_graph_build.jl`, devtests `test_graph_build_live.jl`.)
- `Builder/GraphMode.jl` — a `[graph]` project end to end. `build`: configure
  (`cmake_graph`), `run_generators` over the selection's scope
  (`GraphBuild.scope_ids`), `build_graph`, then per library what
  `compile_project` does after a link (nounwind/byval, metadata,
  `verify_wrap_surface`). `wrap`: one wrapper per library in topological order,
  then `<output>/<Graph>.jl` including them in that order (PE load order). **One
  config per library** for link, metadata and wrap: `GraphBuild._library_config`
  with `module_name` and `graph`. **Library module names come from
  `library_module_names` only** (`Core` → `RCore`; a submodule named `Core`, or
  like its parent, is unreachable). A vendored graph must bind its own copies:
  ELF graph libraries carry a soname, and a graph library's PE preload opens
  dependencies relative to the wrapper (`_link_dir_preload_snippet`, gated on
  `config.graph`). `wrap` hands each library the type manifests of the ones
  wrapped before it, so a type an earlier library owns is imported, never
  emitted again (`Wrapper/TypeImports.jl`; the invariant under Generators).
  (`test_graph_mode.jl`, devtests `test_graph_mode_live.jl`.)
- `Builder/RustBuild.jl`, `RustABI.jl`, `RustDWARF.jl` — the Rust front end's
  build side. `RustBuild.build_rust_project` writes a ROOT crate under
  `[paths] build` (`extern crate <name>;`, a `dylib`, the wrapped crate a path
  dependency), has cargo and rustc compile AND link it (the profile stated
  whole), then writes `compilation_metadata.json` with `"language": "rust"`:
  one row per function (`RustABI.join_function`), the types they reach
  (`RustDWARF`), and per type that crosses by value the compiler's answer to
  "must this be dropped" with the function that drops it (`RustGlue`, the
  metadata's `glue`). `RustABI`, `RustDWARF` and `RustGlue` are text in,
  tables out, and run no tool. `IRGen/RustThunks.jl` turns the rows into thunks (one
  `jlcs.native_call` each), JITs them (`load`), compiles them into a library
  of their own (`aot_library`, which `wrap` runs) and calls one (`call_raw`).
  `RustBuild.discover_project` is `discover` for a Cargo folder.
  `Wrapper/Rust/GeneratorRust.jl` is `wrap`: the Julia module, from the
  metadata alone. `RustRuntime.jl` is what that module calls: `Library`,
  `invoke`, `RustPanic`.
  (`test_rust_abi.jl`, `test_rust_dwarf.jl`, `test_rust_config.jl`,
  `test_rust_thunks.jl`, `test_rust_wrap.jl`, `test_rust_glue.jl`,
  `test_rust_discover.jl`, `test_doctor_rust.jl`, devtests
  `test_rust_build_live.jl`, `test_rust_thunks_live.jl`,
  `test_rust_wrap_live.jl`.)
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
  environment passes a temp one with Pkg offline. Any tool that calls
  `discover(…, wrap=true)` must set both too: `register` otherwise
  `Pkg.develop`s every project into the default environment (abi_fuzz put 180
  scratch path deps into `@v1.13` on 2026-10-24 before it set the variable).
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
  (`test_noexcept_routing.jl`, `callback_test` "bare-name noexcept collision").
  **An attribute group (`#N`) belongs to its own IR file**: LLVM numbers them per
  module, so a reader of `define` lines over several files (a per-TU link, a graph
  library) resolves each file's groups in that file (`_nounwind_definitions`). A
  `nounwind`-only widening would expose Tier-3 emitter gaps (see CHANGELOG
  2026-09-26). A by-value SIMD vector, bare or inside a record at any depth,
  also goes to Tier 2 (`FunctionGen.crosses_as_vector`, `holds_vector`): a
  C-linkage or `noexcept` function crossing one was a plain ccall, which faulted
  on Win64 and read the wrong registers on SysV; the gate gives the vector
  answer GeneratorC gives (2026-11-01). `is_ccall_safe` is the only gate; DAGDiff was removed because it diffed
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
  **R1**: an empty surface is a hard error, with or without promotion; a
  templates-only package has no `functions`, so its STL method table stands
  for its surface, and a table with no member is empty too. **R2**: every
  wrappable function must be in the dynamic table under its mangled name.
  (`test_wrap_surface_guard.jl`, devtests §6c.)
- **A thunk calls its target by name, so the target must be in that runtime set**,
  not merely in `nm`. On PE a method the DLL does not export is still a global `T`
  in the COFF symbol table. `parse_vtables` gates method addresses with
  `_gate_to_exports`, and the STL method table goes through
  `_runtime_symbol_names`, the one source for "what the loader binds"
  (`verify_wrap_surface` reads it too). One unbindable thunk fails the library's
  whole JIT module. (`test_windows_port.jl`, `test_libcxx_exports.jl`.)
  **A callee outside that set is bound only through the library's own
  handle** (2026-11-13): `extract_stl_method_symbols` admits an undefined
  member of a listed container when `dlsym` through the library (opened
  `RTLD_NOW | RTLD_LOCAL`, then closed) resolves it (`runtime_library`), and
  `initialize_global_jit` pre-flights every callee the thunk module names
  (`_preflight_callees`): the library's own are ORC's, a foreign one the
  handle resolves is registered on that engine at that address
  (`register_symbol`), never through `RTLD_DEFAULT`, whose answer could be
  Julia's own libstdc++ (the zlib lesson), and one the handle cannot resolve
  gets no thunk, since ORC blocks on an unresolved symbol instead of raising.
  (`test_jit_runtime_symbols.jl`, devtests `test_jit_runtime_symbols_live.jl`.)
- **libc++ 23 hides its own functions** with a `#pragma clang attribute` in
  `__configuration/namespace.h`. It applies on COFF too, and no macro opts out. On
  MinGW clang turns hidden into `-exclude-symbols:`, so those members never reach the
  export directory. `create_library` un-hides hidden `std::__N` definitions in the IR
  for `[types] templates` packages (`_unhide_libcxx_ir`), on both codegen paths.
  Members `libc++.dll` exports itself are never auto-exported by lld (their `__imp_`
  twin wins) and stay unbound.
- `nm -D` output is parsed only by `Compiler._nm_dynamic_names`, which cuts ELF
  `@VER`/`@@VER` suffixes. Mangled output only: demangled names can contain `@`.
- **A symbol's mangled and demangled names are paired by position, never by
  address** (`_pair_nm_outputs`, 2026-11-13): `extract_symbols_from_binary`
  takes both `nm` listings with `-p` (the symbol table's own order; nm's
  default sort collates demangled names under `-C`) and reads them line by
  line. Several symbols share an address: Tier 1's `__rb_` aliases, and
  clang's constructor and destructor aliases (`-mconstructor-aliases`; at
  `-O1` and above a derived class's trivial destructor is its base's D2), so
  an address map gave tinyxml2's `~XMLComment` the symbol of `~XMLNode`. The
  map stays only as the fallback for listings that do not line up. The
  ORDER of the metadata's functions, which every wrapper follows, is still
  nm's sorted demangled listing (`_order_like_listing`): it is a locale
  collation (`en_US.UTF-8` folds case and skips punctuation) that nothing in
  Julia reproduces, so a third `nm -gC` call supplies it, with D1 before D2
  among aliases so the dedup keeps the variant it always kept.
  (`test_symbol_hygiene.jl`, `test_cpp_finalizers.jl` and its live twin.)
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
  hard error, and hit counting must not short-circuit. The build reads it once too,
  so a dropped symbol seeds no toolchain record; the glob dialect lives at package
  level (`_glob_matcher`) so both sides compile a pattern the same way.
- `[wrap] surface_types_only` is **opt-in and must stay default-off**. The reachability
  walk cannot see types used only through function-pointer members (md4c) or
  constant-only enums (argon2). `surface_types_extra` is the hatch, and an unmatched
  pattern is a hard error. A pattern hits only on a key **outside** the reach (the
  `&&` in `_apply_surface_type_filter!` skips `matches` for reached keys), so a
  keep-pattern a walk fix made redundant fails as "matched nothing" while the type is
  in `struct_definitions`: boostmath's five, on HEAD after 48befe9 (2026-10-29). The
  error is right (a stale pin), its diagnosis is wrong. Seed the walk from method `class` and from globals that are
  **in dynsym**. `_runtime_defined_names` returns `nothing` when it could not look, and
  callers must fail safe on it. **Check public enums by hand** when a package enables it.
  **A spelling reaches a key in every dress** (`_surface_core_spellings`: `const
  T&`, `T*`, `Ref{T}`/`Ptr{T}`/`NTuple{N, T}`, and `ns::T` with the scope cut at
  depth 0 of the template arguments), because a key keeps its template arguments
  and no identifier token equals `normal_distribution<double, policy<…> >`; before
  that, boostmath's five distributions were emitted as empty structs (2026-10-25).
  A new way a signature spells a record goes into that function, never into a
  second resolver. **The keep patterns are matched against every key, reachable
  or not**: `matches` is the hit counter, and a short-circuit on reachability
  read a pattern for a reachable type as "matched nothing" (2026-10-31).
  (`test_surface_types.jl`.)
- `_drop_unresolvable_globals!` removes globals absent from dynsym before either
  generator runs, unconditionally.
- **A global is keyed by the symbol the loader binds** (2026-11-18,
  `Compiler._library_variables`): `DW_AT_linkage_name`, else the DWARF name.
  Keyed by the bare name, `cfg::depth` and `Stat::count` were looked up in
  dynsym as `depth` and `count` and dropped. A static member's definition has
  no name of its own: it takes name, type, scope and access from the
  declaration its `DW_AT_specification` names (`DW_TAG_variable` in DWARF 5,
  an external `DW_TAG_member` in DWARF 4, kept in `static_member_decls`
  because pass 1 files a member as a field). An entry carries `symbol`, `name`,
  `scope` and, when not public, `accessibility`; anything that asks the symbol
  table about a global reads `_global_symbol` (the key, for older metadata),
  never the bare name. GeneratorCpp names the accessor by the qualified name
  (`cfg_depth`, `geo_Vec3_live_count`; an unscoped global keeps its name byte
  for byte) and emits a value accessor only for a type the module can hold
  (`std::string` gets `_ptr` alone). A private or protected static member gets
  no accessor (`_drop_nonpublic_globals!`); a `constexpr` member has no symbol
  and is listed in the metadata's `global_constants`, bound nowhere. A
  variable whose scope chain is unknown is skipped, never bound by its bare
  name. C keys are already the symbol and C wrappers are unchanged. Needs a
  rebuild. (`test_cpp_globals.jl`, devtests `test_cpp_globals_live.jl`.)
- **A thread-local global is read through an accessor the build compiles**
  (2026-11-23, `compile_project`'s step 4c). `cglobal` resolves once, in the
  thread that calls first. The linked IR says which variables are
  `thread_local` (`_thread_local_definitions`: the same on every target; `nm`
  prints `D`/`B`, and clang writes no DWARF location for one on COFF), and a
  generated unit (`replibuild_tls.<ext>`, `_tls_unit_text`) declares each
  with an incomplete type of its own in its namespace and exports
  `replibuild_tls_<symbol>` returning its address: the compiler computes it,
  and a C++ use goes through the variable's wrapper, which runs a dynamic
  initializer. The unit's IR must name each symbol (`_tls_rows_checked`).
  The wrapper calls the accessor on every use (`_thread_local_global_chunk`,
  both generators); a static data member (`record_member`), a variable the
  library does not export, and graph mode get refusals naming the reason
  (`tls_refused`). Metadata without `thread_local` keeps `cglobal`. The
  accessors, `_ZTH` and `_ZTW` are not API (`extract_symbols_from_binary`);
  on PE the accessors are `dllexport` and count as shims in
  `_pe_export_intent`. Needs a rebuild. (`test_tls_globals.jl`, devtests
  `test_tls_globals_live.jl`.)
- `extract_symbols_from_binary` filters `__rb_*`, Itanium adjustor thunks
  (`_is_itanium_thunk`: `_ZTh`/`_ZTv`/`_ZTc`, while keeping `_ZTV`/`_ZTI`/`_ZTS`/`_ZTT`)
  and the templates TU's own `replibuild_force::` helpers (`ref<T>`, `tag<T, N>`),
  which exist to be named, not called. A thunk wrapped as an API function is a
  zero-arg call that segfaults. (`test_symbol_hygiene.jl`, `test_stl_table_matching.jl`.)
- `[link] link_dirs` emits `-Wl,-rpath` (`$ORIGIN` first) alongside `-L`. `-L` alone
  loads the distro copy at runtime, giving two copies of process-global state. This is
  what makes Hub packages composable.
- **The shim file is written when `[wrap.macros]` or `[wrap] shim_defines` has
  entries** (`generate_macro_shims`). A check for the file asks both
  (`verify_shim_headers`); a check for shim functions asks `macros` alone, since
  only macros emit `RB_SHIM_EXPORT` (`create_library`'s PE export gate). The
  defines go into that file and no other: in `[compile] flags` they reach every
  file, and a second file that includes an stb-style header defines its whole
  implementation again. (`test_shim_defines.jl`, devtests
  `test_shim_defines_live.jl`.)
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
- **A definition finds its declaration wherever it is in the dump.** An
  out-of-line definition has no name, only `DW_AT_specification: <offset>`,
  and the parser assumed the declaration came first. It need not: clang 22
  linked `test/default_arg_test` with the class and its declarations in the
  SECOND unit and the definitions in the first, each naming an offset ahead
  of it, so every definition was dropped whole (no parameter names, no
  `has_object_pointer`, a static member wrapped with a receiver). Clang 18
  and 23 put the class first, so only CI's LLVM 22 cells showed it
  (2026-11-15). `linkage_by_offset` is now filled by a pass over the dump
  before the walk, and a definition's parameters (`definition_params`) are
  kept when its declaration is read afterwards. A parser assumption about
  DIE ORDER is a toolchain fact: test it with the order reversed.
  (`test_dwarf_attribution.jl`.)
- **A later declaration does not erase a definition's fact.** A function's
  entry is rebuilt by every DIE of its key that has a return type, a later
  compile unit's in-class declaration included; `has_object_pointer` is
  carried across that rebuild. It was not: a static member with a non-void
  return, declared again by a unit after the one defining it, lost its
  `false` and got a receiver from the class heuristic. Unit order follows
  the IR files' names (a hash of the directory is in them), so the same
  source wrapped right on one host and wrong on another, and CI's LLVM 22
  cells were the ones that showed it (2026-11-15). Any other fact a
  definition DIE records outside that `Dict(...)` needs the same carry.
  (`test_dwarf_attribution.jl`, both link orders vendored.)
- **A type's parts are collected in pass 1, before pass 2 resolves any signature.**
  That includes a subroutine type's parameters (its direct-child formal parameters,
  then `...` for `DW_TAG_unspecified_parameters`): clang emits a type after the
  first function that uses it. `resolve_type` gives each branch of a signature its
  own cycle guard (`copy(visited)`); a shared one made the second `int` of
  `int (*)(int, int)` a cycle. (`test_dwarf_attribution.jl`.)
- **A SIMD vector is not an array.** DWARF spells one as the array of its lanes
  (`float[4]`), and the metadata marks it `vector_size` on the member, parameter
  or return (`_vector_size`). Anything that classifies a by-value crossing reads
  that, never the spelling: SysV classifies a vector by its size (`<1 x float>`
  is INTEGER, `<4 x float>` one XMM register), and the `NTuple` standing in for
  it is classified lane by lane. Metadata without the key must fail safe. A
  parameter or return spelled as an array is a vector (`crosses_as_vector`).
  A member aligned to its own size is refused on SysV until a rebuild, because
  an `aligned` array type has the same shape; the parser marks that array
  `vector_size: null`. (`test_simd_vectors.jl`, devtests
  `test_simd_vectors_live.jl`.)
- By-value aggregate returns through a typedef are repaired in a post-pass (type and
  size). Do **not** swap the return mapper for the parameter mapper: it maps `char*`
  differently and would dismantle the Cstring policy.
- **An array DIE keeps its own `DW_AT_byte_size`**, and `_die_byte_size` prefers
  it to lanes × element: clang writes it only when the lanes do not fill a
  vector (`ext_vector_type(3)` of float is 16 bytes, not 12), and SysV
  classifies the vector by that size. (`test_simd_vectors.jl`.)
- **A record over 16 bytes that is one SIMD vector at any depth is `:unknown`**
  on SysV (`_sysv_sole_vector`): with `-mavx` clang passes
  `struct { struct { v8f v; } in; }` in a YMM register like the one-level
  record, and the metadata does not say how the callee was built.
- **A DIE with no `DW_AT_type` is `void`**, pointed at or qualified: a
  target-less pointer is `void*`, a target-less `const` is `const void` (clang's
  `const void *`), and the transparent qualifiers (`volatile`, `restrict`,
  `_Atomic`) give `void`. `resolve_type` never answers `unknown` for one; that
  spelling once reached 1257 parameters in llama.cpp's metadata.
  (`test_dwarf_attribution.jl`.)
- **Every defined library record has a `struct_definitions` entry, members or
  not** (2026-11-14, `_memberless_record_entry`): `kind`, `byte_size`,
  `members = []` and its `base_classes`. A derived class adding only methods
  (Bullet's `btBoxShape`, `btQuaternion`), a stateless functor, an interface.
  Nothing lays it out in the metadata: GeneratorCpp flattens the non-virtual
  bases into fields (a non-virtual base is a hard dependency of the struct
  order, or a memberless record sorts first and its inherited fields come
  out as bytes), the SysV classifier reads the bases' leaves, StructGen
  pads the thunk body to `byte_size`; flattening the entry would delete the
  `base_classes` the upcasts, `has_subclass` and `graph_subclassed` read. An
  EMPTY class now has a one-byte entry too, so `by_value_crossing` asks
  `record_abi`'s `empty` before `:layout` (dropped on SysV, a register pad on
  Win64/AAPCS64) and `is_ccall_safe` sends that crossing to the thunk. Left
  out: a declaration (no `byte_size`), GNU C's size-0 struct, an anonymous
  record, a lambda's closure type. A toolchain record keeps the
  `_laid_out_from_bases` rule. (`test_memberless_records.jl`, devtests
  `test_memberless_live.jl`.)
- **An inline constructor the library declares and defines nowhere is
  defined by the build** (2026-11-15, `generate_inline_constructors`). After
  the first link `_inline_constructor_plan` takes declared minus defined:
  every library record's declared constructors
  (`record_abi[name]["constructors"]`: the `C4` keys with the declaration
  DIE's `accessibility`, `deleted` and `artificial`) less those the symbol
  table has a `C1`/`C2`/`C3` for. A generated unit (`replibuild_ctors.cpp`,
  `-fno-inline` like the templates TU) includes each record's header and
  explicitly instantiates `replibuild_force::ctor<T, A...>`, whose placement
  new odr-uses the constructor; `compile_project` compiles it, relinks and
  extracts again, and the metadata's `inline_constructors` carries each
  row's outcome for the wrap report. The class and the parameters are
  spelled from the demangled `C4` key (`_demangle`: `__cxa_demangle` from
  Julia's own C++ runtime, else `c++filt`), never from DWARF `c_type`s: a
  declaration's parameters carry no names, so the parser keeps none, and a
  bare name cannot qualify `btRigidBody::btRigidBodyConstructionInfo`.
  Skipped and named: a constructor template, a deleted, private or protected
  one, an implicit one, a conflicted name, a record with no header, an
  unspellable name, one `exclude_symbols` drops, a header that does not
  compile on its own; the unit's guard (`is_abstract`, `is_constructible`)
  declines an abstract class silently and the report says so. A hidden
  definition is made exportable at link (`_forced_exportable_ir`: `hidden`
  dropped on ELF, `dllexport` added on PE with `--export-all-symbols` asked
  for on the shims' terms). The pass sees only a definition some CU's DWARF
  carries: under constructor homing a class whose constructors are all
  inline and unused is declaration-only everywhere, and `-fstandalone-debug`
  (Bullet, box2d) is what makes it appear. Graph mode does not run it.
  (`test_inline_constructors.jl`, devtests `test_inline_ctors_live.jl`.)
- **An inline member function the library declares and defines nowhere is
  defined by the same unit** (2026-11-15, `_inline_method_plan`,
  `generate_inline_definitions`). The parser writes every member function a
  record declares to `record_abi[name]["methods"]` (the constructor rows'
  facts plus `virtual`); the plan forces the public, non-virtual ones no
  symbol defines, spelled from the demangled name (`_method_spelling`: the
  LAST paren group is the parameter list, the name is cut at the last depth-0
  `::` or at `::operator`, a depth-0 space before it is a return type and so
  a member template). The unit takes each address through
  `replibuild_force::args<A...>::m` / `mc` (the overload taking exactly those
  parameters, `const` or not, a static through the function-pointer overload,
  the return type deduced because a mangled name does not carry it) into a
  `used` constant; the symbols join the sidecar and the export rewrite.
  - **Every row is one line of the unit**, and a compile failure is traced to
    rows by the lines its diagnostics name (`_unit_bad_lines`, a note
    belonging to the error before it). A constructor's error sits in the
    placement-new helper and names no row, so constructors alone fall back
    to compiles of their own: in groups first, halved where one fails
    (`_alone_verdicts`), so only a row tried alone is ever given an error,
    under the file name it always had (the error line can name it). Keep a
    new row kind on one line.
  - **The include block is compiled by itself before any row is judged**
    (2026-11-16). A header that is half of a pair (Jolt's
    `WrapShaderBegin.h` opens namespaces its partner closes; a class
    declared through the pair has the opening half as its `decl_file`)
    leaves every header after it inside its scope, so the errors land on
    rows that are fine: 1,944 of Jolt's constructors were skipped as "does
    not compile". When the include block fails, each header is compiled
    alone, in parallel (`check_headers!`), dropped from every row's list,
    and a row left with none is skipped as "its header does not compile on
    its own", which the wrap report groups apart. Row tracing is only
    sound once the includes are.
  - **A member the unit's IR only `declare`s is skipped**: it has no body in
    its header (an out-of-line member kept hidden, or in a file the build
    does not compile), and its address in a constant is a data relocation
    the loader resolves eagerly, so the library would not load. The probe
    compiles to IR for this, not `-fsyntax-only`.
  - A virtual member comes with the vtable and is never forced. A member of
    a class template's specialization is `[types] templates`' to
    instantiate (`_TEMPLATE_MEMBER_SKIP`): forced, Bullet's
    `btAlignedObjectArray<T>` and allocator members were 2,990 of 4,999.
    Private, protected, implicit and deleted members are passed over
    silently; the rest are named in `inline_methods` and the wrap report.
  - Metadata without `methods` rows plans nothing: a package gets its inline
    methods on a rebuild, not a re-wrap.

  (`test_inline_methods.jl`, devtests `test_inline_methods_live.jl`.)
- **The forced unit includes headers, and a header can define things**
  (2026-11-21). An implementation header defines functions or variables with
  external linkage and is meant for ONE translation unit: fmt's
  `format-inl.h` without `FMT_HEADER_ONLY` (a forced record is declared
  there), clipper2's `clipper.export.h` (in the recipe's `[wrap]
  shim_headers`, so first in every generated unit). The library's own unit
  includes it; the forced unit including it again defined each symbol twice,
  the relink was "multiple definition of", and neither package BUILT on the
  4.6.0 release sweep. Every compile of the unit had succeeded: nothing read
  what it defined.
  - **The unit's IR is read for strong definitions that are not its own**
    (`_foreign_strong_definitions`: external linkage, not `linkonce_odr`,
    `weak_odr`, `internal` or a declaration; the unit's own are the
    `replibuild_default::` entry points). Only when there is one, each
    header is compiled alone to IR (an umbrella header by itself, a row's
    header beside the umbrella), the ones that define such a symbol leave
    the umbrella or every row's list (`drop_defining_headers!`), a row left
    with no header is skipped with that reason, and the rows are judged
    once more. A unit that still defines one is not linked at all.
  - **A relink that fails with the forced unit falls back to the library
    without it** (`_forced_unit_refused`): the unit is RepliBuild's own
    addition, and the project's library builds without it. Every row is
    reported with the linker's line. The FIRST link is the project's own
    and stays a hard error.
  - A new kind of row, or anything else the unit includes, must keep the
    unit free of strong definitions other than `replibuild_default::`, or
    `_is_forced_unit_own` has to learn it.

  (`test_forced_unit_definitions.jl`, devtests
  `test_defining_headers_live.jl`.)
- **A call that leaves default arguments out is a function the forced unit
  compiles** (2026-11-15, `_default_argument_plan`). A default exists only in
  the header (no `DW_AT_default_value` from clang, and the mangled name
  spells every parameter). For each public constructor and member function
  of a library record and each proper prefix of its parameters, one row in
  namespace `replibuild_default` makes that call on `Class* self`; a call
  the header has no defaults for is a compile error on its row and is
  dropped SILENTLY (most rows; Bullet tries 4,150).
  - The arguments have exactly the member's parameter types, so C++ takes
    that member or calls it ambiguous (dropped). A prefix a sibling declares
    in full is never generated: that call is the sibling's.
  - A row is kept only when the member it calls is defined, by the library
    (`defined` in the plan) or by the unit (`_ir_defined_declared`): a call
    to a member defined nowhere would not load.
  - **The function takes the member's identity in the metadata, not a
    clone's.** Its symbol's demangled name is rewritten to `Class::f(prefix)`
    before `parse_function_signatures`, and
    `_apply_default_argument_identity!` names its first parameter `this` and
    sets `has_object_pointer`, so both receiver gates, `_julia_function_name`,
    `[wrap.rename]` and `exclude_symbols` treat it as the method with fewer
    parameters; the parameters and return type are the function's own DWARF.
    An entry carries `default_arguments` (`of`, `omitted`). A function taking
    `Class*` first is called as a member is on Itanium, mingw included.
  - **A static member's row is refused in the unit, not from DWARF**: each
    member row names `&Class::f` through `replibuild_force::nonstatic<full
    parameters>`, which has no plain-function overload, so a static member's
    short call does not compile. `has_object_pointer == false` on the
    member's definition is only a second guard: clang 22 does not record it
    for every static member, and CI's LLVM 22 cells bound one (the LLVM 23
    cells and the dev box did not). A check that reads a DWARF fact needs a
    toolchain-independent twin, or a test on both LLVMs.
  - `replibuild_default::` must stay OUT of `extract_symbols_from_binary`'s
    filter (`replibuild_force::` is in it): these are API.

  (`test_default_args.jl`, devtests `test_default_args_live.jl`.)
- **`[types] records` makes a class no source mentions visible** (2026-11-15,
  `generate_visible_records`). Debug info describes the types a unit uses, so
  a header-only class nothing in the library mentions (Bullet's
  `btDefaultMotionState`) has no DIE, standalone or not, and the
  inline-constructor pass has nothing to force. Each entry is instantiated
  as `replibuild_force::record<T>(T*)` in `replibuild_records.cpp`
  (`[wrap] shim_headers`, then `[types] template_headers`), compiled in the
  FIRST pass beside the templates TU so the first extraction sees the class.
  Its body takes `sizeof(T)`: an entry that does not compile (a typo, a
  declared-only name, a missing header) is a build error naming it, found by
  compiling each entry alone; one that compiles and has no
  `struct_definitions` entry (constructor homing under plain `-g`) is a
  warning naming it (`_report_invisible_records`). Never `template class T;`,
  which is invalid for a non-template and is why `[types] templates` could
  not do this. (`test_visible_records.jl`, devtests
  `test_visible_records_live.jl`.)
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
- **A constant GNU readelf prints from four or eight bytes is hex**
  (`DW_AT_count : 0x10000`; one or two bytes print in decimal). Read it with
  `_dwarf_count`, never `(\d+)`, which takes the `0`: an array of 65536
  elements or more had no elements and no size (lz4, miniz, pcre2, box2d,
  libjpeg-turbo and three more Hub libraries), and a bitfield 8 KiB into its
  record had offset 0. A fix here reaches a package on a rebuild.
  (devtests `test_libdw_dwarf_live.jl`.)
- **A declaration file is an absolute path, whatever the DWARF version.**
  `_dwarf_file_tables` reads each table by its own column header (DWARF 4 and
  earlier print `Entry Dir Time Size Name`; read as DWARF 5's shape, every name
  of a `-gdwarf-4` build came out as `0\t0\t<name>`), and `parse_dwarf_dump`
  resolves a relative name against the unit's `DW_AT_comp_dir`, which DWARF 4's
  table does not carry. (devtests `test_libdw_dwarf_live.jl`.)
- **Two parsers, two dialects.** `Compiler.jl` parses **GNU binutils** output only
  (readelf on Linux, GNU objdump on Windows; byte-identical). `DWARFParser.jl` parses
  llvm-dwarfdump. Never copy a regex between them, and never route the readelf
  parser to llvm-dwarfdump.
  The two dumpers also spell tags differently: binutils writes
  `DW_TAG_template_type_param`, llvm-dwarfdump `…_parameter`. The readelf parser
  matches the short spelling, a prefix of both; a new tag match must be checked
  against a real readelf dump, not against llvm-dwarfdump's
  (`test_dwarf_attribution.jl`).

- **A pointer or reference to an enum is an address** (2026-11-18,
  `cpp_to_julia_type`: `E*` is `Ptr{E}`, `E&` `Ref{E}`; only by value is it the
  enum). Mapped to the enum, `void out(enum E *)` took the value and C wrote
  through it. A generator set built by scanning emitted text for type names
  adds `_pending_enum_type_names(enum_chunks)`: the enum section is pending
  records, not text. A parameter or return still `Any` after the mapper is one
  trap stub naming the C type (`_unmapped_crossings`), on every ccall-emitting
  path of both generators. An enum defined with no enumerators has an entry
  (`std::byte`), a toolchain one only where the exported surface names it, and
  such enums skip the member-set dedup. (`test_type_mapping.jl`, devtests
  `test_type_mapping_live.jl`.)
- **A base-object variant's VTT is that variant's own parameter** (2026-11-18):
  the definition with one parameter more than its declaration gets an entry
  under its own symbol, the VTT marked `artificial`, and the unified C4/D4 key
  keeps the declared list in any DIE order. An artificial parameter after
  `this` is not the test: an implicit copy constructor's source is artificial
  too. (`test_structor_variants.jl`.)
- **A symbol no DIE describes is never called with a guessed signature**
  (2026-11-18). It takes the signature of a described exported function at its
  address (`_alias_signature_key`, `alias_of`), or stays `return_type_source:
  inferred`, which both generators emit as `_no_debug_info_stub`. An alias of a
  `static`, hidden or `__rb_`-promoted function is refused: a local name is no
  key into the function table. `parameters_source: inferred` alone (K&R) is
  still called. (`test_alias_signatures.jl`, devtests
  `test_alias_signatures_live.jl`.)

### Generators

- **The STL screen is asked with the record's definition.** `_is_stl_internal_type`
  matches bare names, which is all DWARF keys records by, and screened
  `boost::core::basic_string_view<char>` like `std::string_view`. Every record
  definition carries `scope` (the parser's `_scope_of`, from namespace DIEs that
  set the depth context like records), and GeneratorCpp asks
  `_is_stl_internal_record(name, dwarf_structs)` through its `_stl_internal`
  closure: outside `std`/`__gnu_cxx`/`__cxx*` a record is never STL-internal; with
  no scope, or inside `std`, the name decides as before. A record marked
  `"toolchain"` (promoted, above) is never STL-internal either. Never call the
  name-only screen in the generator. The key needs a rebuild.
  (`test_record_scope.jl`, `test_toolchain_records.jl`.)
- **A `[types] templates` entry is matched to the compiler's spelling, never to
  its own text.** The templates TU instantiates `replibuild_force::tag<T, N>(T*)`
  per entry, and `_stl_methods_from_nm` reads the demangled `T` off the symbol
  table to key the STL method table by the entry as written
  (`std::vector<Clipper2Lib::Point64>` reaches `vector<Point<long>>`'s members;
  the entry's `container` field carries the canonical spelling). Only
  `_STL_TABLE_CONTAINERS` get a table; any other template is instantiated for
  its DWARF alone. An entry that yields no table is a `@warn` from
  `_stl_table_gaps`, never silence, and a vector factory whose element has no
  Julia layout in the wrapper is a `@warn`, never `CppVector{Any}`.
  (`test_stl_table_matching.jl`, `stl_test/verify.jl`'s `Pt64` testset.)
- **Integer parameters are `::Integer` unless an overload set disagrees on their
  width.** `_integer_width_overloads` predicts each wrapper's name and dispatch
  signature the way the function loop builds them and marks the positions a set
  of distinct symbols differs on; those keep the C type so dispatch separates
  `f(int)` from `f(long long)`. A width is the type the alias names on this host
  (`_concrete_ctype`): `int` and `long` are one width on LLP64, `long` and `long
  long` on LP64, and a set that differs only in spelling stays `::Integer`.
  `_method_sig_keys` keys by the same concrete types, so two spellings of one
  width dedup instead of defining one signature twice, which precompilation
  refuses. A change to how the loop types or names a function
  must move the prediction with it. (`test_integer_width_overloads.jl`.)
  **The same rule types a pointer and an enum** (2026-11-13): a pointer
  parameter is `::Any` (a `Managed` object, a `Ref`, a `String` pass through
  `unsafe_convert`) and an emitted enum relaxes to `::Any` too (an `Integer`
  still passes), unless the set disagrees on the position, which then takes
  `_strict_param_spelling`: the pointer's target as a `Union` of what callers
  pass (`_CHAR_POINTER_UNION` for a `char*`; `Ptr{T}`, `Ref{T}`,
  `AbstractVector{T}` and the owning types for a record; `Any` for `void*`
  and for a target nothing emits), the enum as itself, never `Union{E,
  Integer}`, which would take an integer overload's arguments. The
  prediction and the loop spell a strict position through that one function.
  A `T&&` overload beside its `const T&` twin is not emitted at all
  (`_rvalue_shadowed`): both are one Julia signature and the kept one used
  to be whichever came last, a move where the caller meant a copy. The thunk
  manifest is the set the FINAL text calls (`_thunks_named_in_text`), so a
  chunk the dedup dropped leaves no thunk; the 87 it left across the Hub
  were compiled for nothing. (`test_tier2_typed_dispatch.jl`.)
- **A C wrapper's integer parameter takes an enum too** (2026-11-15,
  `_C_INTEGER_PARAM`, `Union{Integer, Base.Enum}`): C converts an enumerator
  silently and raylib's `IsKeyDown(int key)` is called with `KEY_SPACE`, an
  open enum that is not an `Integer`. GeneratorC and the C arm of the varargs
  path's fixed parameters spell it through that constant; C++ keeps
  `::Integer`, where `f(int)` may stand beside `f(E)`'s `::Any` method. The
  relaxed types are `_C_INTEGER_PARAM_CTYPES`: the C aliases and the
  fixed-width types (`int32_t` is `Int32` in the metadata), one list for
  both paths. (`test_c_generator_policies.jl`, `c_test/verify.jl`.)
  **The converse holds too** (2026-11-18): a C wrapper's by-value open-enum
  parameter is `Union{E, Integer}`, converted with `E(x)`
  (`_open_enum_type_names`; the varargs path reads the parameter's
  `open_enum` mark). Bitmask aliases, pointers to enums and graph-imported
  enums are unchanged, and C++ keeps the enum strict inside an overload set.
- **A by-value `char` is the host's `char`, and a bitfield has its declared
  type's sign** (2026-11-18). The metadata spells `char` as `UInt8` (pointer
  positions rely on it), so a by-value `char` parameter was `zeroext` and a
  `Cchar` argument a `MethodError`; `_char_param_ctype` respells it `Cchar`
  (and `signed char` as `Int8`, since `Cchar` is `UInt8` on AArch64) inside
  both generators' `_ccall_param_type`, so a re-wrap fixes old metadata.
  Every integer argument is converted through `_integer_arg_conversion`
  (both generators' ccall bodies, the C varargs arm, `_tier2_invoke_args`),
  which takes -128..255 for a `char`. `FunctionGen._narrow_int_extension`
  reads the C spelling before `julia_type` for the same reason. Inside a C++
  overload set a `char` position is `::Cchar`, so a `UInt8` reaches it only on
  a host whose `char` is unsigned. A signed bitfield reads sign-extended as
  the type `_bitfield_signed_type` gives (the declared type, an enum's
  underlying type, `Cchar` for plain `char`) and its setter takes exactly the
  field's range (`_bitfield_sign_extend`, `_bitfield_signed_store`); unsigned
  and `bool` fields emit what they did. A test here states its expectation
  through `Cchar`, never `Int8`. (`test_char_bitfield_emission.jl`, devtests
  `test_char_bitfield_live.jl`.)
- **A thunk extends an integer narrower than `int` to 32 bits, and an enum is
  an integer** (2026-11-16, `_enum_mlir_int`, `_narrow_int_extension`). The
  x86-64 convention as compilers implement it has the CALLER extend such an
  argument, and clang's callees read it 32 bits wide (`i8 zeroext`). A
  by-value enum took the record path, `!llvm.struct<"E", (i8)>`, which is
  coerced to a whole register read from one byte of storage, so the bytes
  above were stack garbage: Jolt's `GetNumActiveBodies(EBodyType)` (`enum
  class : uint8`) indexed an array with it. An enum now crosses as its
  underlying integer by `byte_size`, never through `map_cpp_type(julia_type)`
  (a struct for `Int8`, `UInt16`), and any `i1`/`i8`/`i16` argument is loaded
  at its width, extended (`llvm.zext`/`llvm.sext` by the parameter's or the
  enum's signedness; plain `char` follows `Cchar`) and passed to a callee
  declared `i32`. A real one-member record is still a record: its callee
  reads only its bytes. (devtests `test_narrow_args_live.jl`, whose child
  segfaults on the old lowering.)
- **A Tier 2 argument is converted to its C width in the emitted body.** A
  thunk loads the C parameter's width from the argument's storage, so a
  Julia integer narrower than the C type was read past its box
  (`as_int(attr, Int8(-1))`: garbage, pugixml 2026-11-03). `_tier2_invoke_args`
  spells every `::Integer` parameter through `_integer_arg_conversion`
  (`Ctype(name)`, with `char`'s wider range) and an emitted enum as its base
  type; a new scalar kind in a Tier 2 signature must go
  through it. (`test_tier2_scalar_widths.jl`, devtests
  `test_cpp_record_routing.jl`'s `scalar_widths`.)
- **Two generators, one answer.** C and C++ duplicate type sets by design, for
  isolation. Shared decisions must still agree, so put the derivation in
  `Wrapper/Utils.jl` rather than copying it. Agreement is necessary, not sufficient.
- **One Julia type per C++ type across a graph** (2026-11-10,
  `Wrapper/TypeImports.jl`, design `docs/updates/CROSS_LIBRARY_TYPES.md`).
  Every wrap writes `type_manifest.json` beside its library, derived from the
  FINAL text: the types it defined as its own, with Julia name, kind, size,
  `scopes`, layout signature, field layout, `Managed` type, strongly defined
  members, and the symbols the wrapper binds. In graph mode `wrap` goes in
  topological order and `_resolve_type_imports` decides each type against the
  earlier manifests: the first earlier library that emitted it owns it when
  the scopes meet, the layout signatures agree where both have one (unequal in
  one scope is a wrap ERROR, never a silent pick) and no member is defined
  strongly in both (an error; skipped on PE). Scope unknown keeps the type
  own, reported; a name reached only by declaration imports on the bare name.
  `_apply_type_imports!` gives the entry the owner's body marked `"imported"`,
  so every layout reader sees the owner's definition, and **every per-type
  emission loop in both generators asks `_emits_type`** (forward declarations,
  structs, enums, `Managed`, factory structs, upcasts, setters, the layout
  table). An imported type is one `const` naming the owner's through
  `Base.parentmodule(@__MODULE__)` (`..Owner.X` is import syntax, not an
  expression), emitted before the enums, and an imported enum brings one more
  per enumerator, aliasing the name the owner's text BINDS (the manifest's
  `enumerators`: `[base, bound, value]`), with the owner's collision rule
  applied against the importer's own names (`World.Happy === Hello.Happy`; the
  Hub's hello_world report, 2026-11-11); `_assert_no_imported_redefinition`
  refuses a text that defines one. The owner's `Managed`, factory struct and
  `<f>_safe` are the owner's alone; `has_subclass` is widened with
  `graph_subclassed`; a function an earlier wrapper binds is dropped; a
  placeholder of a later owner gets the owner's `unsafe_convert`. The thunk
  side reads `type_imports.json` through `JLCSIRGenerator.apply_type_imports!`
  (JIT and AOT), whose RAII rows are `imported_raii` and get no thunk. The
  metadata keys `scopes` and `decl_files` are collected BESIDE the definitions
  (`_record_provenance!`), never inside them, or a header path would decide
  which definition `_canonical_definition` keeps. A single-library wrap has
  no earlier manifests and stays byte-identical. (`test_cross_library_types.jl`,
  devtests `test_cross_library_types_live.jl` on `test/xlib_test/`.)
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
- **A finalizer frees what it owns.** `Managed<X>` and the struct a factory
  function makes call the class's deleter by mangled symbol: the library's
  deleter function (no receiver by `FunctionGen._has_receiver`, one `X*`, named
  delete/destroy/free), else D0. D1 and D2 end the lifetime without freeing
  and are `<X>_destroy_<X>`; D0 is `<X>_delete_<X>` (2026-11-14,
  `_is_deleting_destructor`, decided in `_julia_function_name` by the symbol
  alone), with a docstring that says it frees through the class's `operator
  delete`. Named `destroy_` too, a class whose only destructor is D0 (an
  implicit virtual destructor at -O2: 93 of Bullet's 259) ran `delete` under
  the non-freeing name, and the Hub's Bullet run freed the box twice. The wrap
  report names such classes. A class with neither a deleter nor D0 gets no
  owning type. A factory struct's D0 also needs a class nothing in the library derives
  from, and a factory makes the class its pointer names, or a subclass its name
  names. **A method proxy keeps `obj` alive across the call** (`GC.@preserve
  obj X_method(obj.handle, …)`): the owner is dead once its handle is read,
  and a Tier 2 invoke allocates before C++ runs, so its finalizer could delete
  the object under the call (2026-11-10). Anything that passes a handle out of
  an owning type into Julia code, rather than straight into a `ccall`, needs
  the same. (`test_cpp_finalizers.jl`, devtests `test_cpp_finalizers_live.jl`.)
- **A factory class a record's field names is defined before every record**
  (2026-11-18, `_idiomatic_early_type`). The owning `mutable struct X` a
  factory makes was written after every record while the struct loop counted
  it defined at its DWARF position, so `struct Cage { Animal *pet; }` emitted
  `pet::Ptr{Animal}` above `Animal` and `using` was `UndefVarError`. For a
  class a field names (`Ptr{X}`, `Ptr{Ptr{X}}`, `NTuple{N, Ptr{X}}`), the type
  goes in a slot ahead of the struct definitions with one inner constructor
  that only stores the handle (`X(::Base.Val{:_rb_adopt}, ::Ptr{Cvoid})`), and
  its factory constructors are OUTER constructors in the idiomatic section,
  after every type: a constructor's argument types are evaluated where it is
  defined, and a factory may take the record holding the pointer, so moving
  the whole struct earlier only moves the failure. A class no field names
  keeps its inner constructors byte for byte. **Julia evaluates a field type,
  a type parameter's bound and a method's argument types at definition**:
  `_assert_types_defined_before_use` (inside `_assert_wrapper_loadable`)
  refuses a wrapper that names, in one of those positions, a type the module
  binds only further down. A new emitter that writes a type or a typed
  signature must place it after what it names. (`test_factory_field_order.jl`,
  devtests `test_factory_field_order_live.jl`.)
- **One method per dispatch signature, everywhere** (2026-11-18). An
  emitter's dedup keys on the dispatch signature (`_concrete_sig_type`, as
  `_method_sig_keys` does), never on the parameter text: factory
  constructors keyed on text with the names in it defined `Animal(::Cint)`
  twice for `make_animal(int legs)` and `make_animal_doubled(int half)` (the
  later replaced the earlier silently, and a precompiled consumer refused to
  load), and dropped one of two factories whose parameter text was equal
  (`finalizer_test`'s `Rect`, `vi_test`'s `Left`). A signature exactly one
  factory has keeps its plain constructor byte for byte; one several share
  is none of theirs: each is `X(Val(:<factory>), …)`, the tag being the
  factory's `_julia_function_name`, in the inner and the early (outer) form
  (`_factory_ctor_plan`), with the struct's docstring and a `wrap:` line
  naming them. **Never a silent pick between two different C++ functions.**
  Method proxies are keyed the same way and collapse only onto one raw
  target. `_assert_no_duplicate_methods` (in `_assert_wrapper_loadable`)
  refuses a wrapper whose parsed text defines one method twice.
  (`test_factory_ctor_collisions.jl`, devtests
  `test_factory_ctor_collisions_live.jl`, `test_cpp_finalizers.jl`.)
- **A type's constructor keeps its name over a function of that name**
  (2026-11-18, `NameScreen._constructor_clashes`). C gives struct tags a
  namespace of their own, so a global or a no-argument function named like a
  struct (curl's `const struct Curl_ssl *Curl_ssl;`, `struct pool
  *pool(void);`) defined the same Julia method as the struct's
  zero-initializer `T()`: the later replaced the earlier silently, and
  `_assert_no_duplicate_methods` then refused the whole wrap. The name
  screen lets a function stand beside a type of its name, since methods
  beside a type are its constructors; that is wrong only when the type's own
  emitted text already defines that signature. `_screen_verdict` now moves
  such a kept function to `name_` (reason `:type_constructor`, through the
  one existing mechanism, `_function_alternative`) and the wrap's `names:`
  report says so; `T()` stays the zeroed record for every struct, union,
  byte blob and bitfield record, in both generators. It reads the text with
  the guard's own `_method_definitions`, and only when a kept function shares
  a name with a type. A global's `<name>_ptr` accessor is named from the
  library's spelling and screened on its own, never derived from the moved
  value accessor: `conf_ptr()`, `Dict_ptr()`, not `Dict__ptr()`.
  `[wrap.rename]` still chooses a function's name first; it does not reach
  global accessors. **The same holds for the implicit field constructor**
  (2026-11-23, `_field_constructor_clashes`, reason `:field_constructor`): a
  kept function with a method of as many positional arguments as a struct of
  its name has fields (`_field_constructor_arities`: no inner constructor, not
  only `NTuple{N, UInt8}` fields) moves, since `pair(::Integer, ::Integer)`
  is more specific than Julia's converting `pair(a, b)` and took
  `pair(1, 2)`; no text defines that method twice. A byte blob, a bitfield
  record or a union is built from its bytes and moves nothing.
  (`test_type_name_collisions.jl`, devtests
  `test_type_name_collisions_live.jl`, `test_field_ctor_shadow_live.jl`,
  `test_name_screen.jl`.)
- **A pointer or reference return is typed from its declaration**
  (`_typed_pointer_return`, both generators): `T*`, `const T*`, `T&`, `T**` give
  `Ptr{T}`/`Ptr{Ptr{T}}` when `T` is a type the module emits
  (`_typed_return_names`, read off the emitted text), opaque `struct T end`
  handles included. A **mutable** record stays `Ptr{Cvoid}` (X5: a typed return
  invites `unsafe_load`), and so do `void*`, function pointers and unknown names;
  `char*` is the Cstring policy's. Typing is not ownership: a return typed only
  this way never gets the `_safe`/`Managed` wrapper (`newly_typed_return`), since
  `GetIO()` returns memory the context owns. A handle that appears only in returns
  has no `struct T end` (forward declarations come from parameters) and stays
  `Ptr{Cvoid}`. A function pointer is the type's own `(`, read outside the
  template arguments (2026-11-24): read over the whole spelling, a
  `const std::function<int (int)>*` return stayed `Ptr{Cvoid}` and did not
  pass where a `const std::function&` is taken.
  (`test_typed_pointer_returns.jl`, `test_placeholder_references.jl`.)
- **A reference to a smart pointer also takes a `Ref` of the bytes one is
  returned as** (2026-11-24, `_value_ref_bytes`, spelled by
  `_by_address_ctype` in the loop and in `_integer_width_overloads`'
  prediction). A `std::shared_ptr` or `std::unique_ptr` is a placeholder
  (`struct shared_ptr_Node end`, in `placeholder_struct_names`) with no value,
  and returned by value it is its bytes. A reference parameter typed
  `Ref{shared_ptr_Node}` took only a pointer already typed that way, so
  `sp_read(Ref(sp_make(5)))`, the manual's way for a returned container, was a
  `MethodError`. The position is now `Union{Ref{P}, Base.RefValue{NTuple{N,
  UInt8}}}`, `N` from `stl_value_bytes`, the size the by-value return has; on
  the plain-call path its `ccall` type is `Ptr{Cvoid}`, which only those two
  reach.
  - **Never `::Any` there.** That was tried first: any `Ptr` then passes
    (`pointer_arg`'s rule for a pointer position), and a `@cfunction` pointer
    given where a `const std::function&` is taken was read as the object and
    the process died. An untyped pointer is anything; a reference to a
    placeholder takes a typed pointer, or bytes of the one size.
  - A reference to any other placeholder (a `std::function`, a stream, a
    class only declared) is `Ref{P}` as before. A `std::ostringstream*` at a
    `std::ostream&` still needs the pointer converted by hand.

  (`test_placeholder_references.jl`, devtests
  `test_placeholder_references_live.jl`.)
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
  On the Tier 2 path GeneratorCpp resolves an `Any` return through the
  cv-stripped C spelling (`FunctionGen._strip_top_cv`; DWARF spells
  `key_value_pair::key()`'s return `const basic_string_view<char>`), and a
  return still `Any` after that is emitted as an ABI trap stub
  (`_unresolved_return_trap_msg`), never as `invoke(…, Any, …)`, which the JIT
  refuses only at the call. A parameter the metadata spells `Ref{Any}` or
  `Ptr{Any}` (a reference to `long double` or `__int128`) is `Ptr{Cvoid}`: the
  address of a caller's buffer, since `Ref{Any}(x)` points at a box slot.
  (`test_cpp_return_resolution.jl`.)
- **A Tier-3 by-value record is classified per target, from its fields' type
  alignment, never a member's size** (C1: `{ char; short[3]; }` got an sret
  the callee never reads).
  - `_c_return_in_memory`, `_c_explicit_sret`, `_c_blob_param_misclassified` and
    `_c_register_class_misstated` are the answers, and `_c_has_unaligned_field`
    is the walk. `_c_record_abi()` is the only place GeneratorC reads the target.
  - Register-class traps (a float-bearing blob, a mixed float/integer region) are
    SysV's only: Win64 crosses a record by size alone on both sides. And they
    fire only where the record's eightbyte classes, read off its DWARF facts
    (`sysv_record_abi`, the Tier 2 classifier, through `_rec_sysv_classes`),
    include an SSE eightbyte or padding C passes nowhere: a float member alone
    makes nothing SSE, since INTEGER wins the merge of an eightbyte's members,
    so a union of an integer and a double (mpack's `mpack_tag_t`) or a float
    beside a bitfield (`BF12`) is INTEGER, INTEGER and the blob travels as C
    does (2026-11-10, the last 31 Hub C misses). A record whose facts cannot
    be read keeps the refusal.
  - **Win64 passes a record over 8 bytes by reference to a copy aligned as the
    record is, and Julia's copy is aligned as the stand-in.** So
    `_c_byval_param_shape` under `:win64` carries a record aligned above 8
    whose size is a multiple of 16 in `_RB_ByVal16{N}`, refuses one whose size
    is not, and asks no `exact` (a copy aligned more than the callee reads is
    harmless). 16 is all Julia can give: an AVX callee's `vmovaps ymm` is not
    covered. A bare SIMD vector is passed the same way there: 8 bytes or less
    is refused (ccall would put the tuple in a register; it faulted), 16 and up
    is carried when the tuple has the vector's size. (`test_simd_vectors.jl`,
    `test_c_sret_classification.jl`, devtests `test_simd_vectors_live.jl`,
    `test_c_sret_abi.jl`.)
  - Whether a record is packed at all is `_record_natural_layout`, the one answer
    for `is_c_lto_safe` and `_is_struct_unsafe`.
  - GeneratorCpp sends to Tier 2 a function that crosses by value a ≤16-byte
    byte blob, or a record whose fields put bytes over a `float` or `double`,
    because bytes classify INTEGER (`_crosses_small_blob`).
  - BRANCH 0 runs only when no refusal fired.
  - **Both emission paths take one verdict.** `_c_byvalue_verdict` (the
    register-window refusals, the stack slots and their carriers, `long
    double`, BRANCH 0's return shape) is computed once per function and
    consumed by the plain path and by the `[wrap.varargs]` branch, which
    hands `generate_vararg_wrappers` its `trap`, `sret` and per-parameter
    `byval_carrier`. A guard added to one path and not the other is W5 again.

  Tests: `test_c_sret_classification.jl`, `test_c_sret_abi.jl`,
  `test_record_natural_layout.jl`, `test_varargs_emission.jl`.
- **A hidden return buffer has the record's alignment.** Julia's own sret
  buffer is aligned to the stand-in (a byte blob: 1), and the callee may store
  with `movaps`. `_c_record_align` is the record's (the parser records
  `DW_AT_alignment` on records and members as `alignment`, a vector member's as
  its size), and BRANCH 0 passes a buffer it aligned itself wherever that
  exceeds the stand-in's and 8 (`_c_sret_buffer_align`). The metadata needs a
  rebuild to carry it. The same `_c_record_align` above 8 routes C++ to Tier 2
  (`_is_struct_unsafe`), never the record's own `alignment` alone: a member's
  `alignas` puts DW_AT_alignment on the member only, the record holding it, and
  any record holding that, name none and have natural layouts, and a ccall read
  such a by-value record from the wrong stack slot (2026-11-10). A member with
  no byte offset and no bits (a C++ static) counts for nothing there.
  (`test_record_natural_layout.jl`, devtests `test_cpp_record_routing.jl`.)
- **A by-value MEMORY record's stack slot is placed by the compiler's `byval`
  alignment, never by DWARF.** Julia's `ccall` writes no alignment on its
  `byval`, so the backend uses the stand-in's (8), and a record C aligns to 16
  lands one slot early behind any stack-resident argument (W2: a plausible
  wrong value). The build reads each definition's `byval(...) align N` off
  the linked IR (`Compiler._byval_definitions`, beside `nounwind`) into
  `byval_params`; GeneratorC matches them to the parameters its classifier
  calls MEMORY (`_c_byval_facts`: counts and names must agree, else no fact)
  and `_c_byval_param_shape` decides. An exact 16 on a record over 16 bytes
  and a multiple of 16 crosses inside `_RB_ByVal16{N}` (chunks of `<2 x
  double>`, whose LLVM type is 16-aligned); anything else above 8 is an ABI
  Safety Trap, including an alignment DWARF alone asserts (metadata without
  the key: the trap says to rebuild) and 32 and above (Julia's allocas and
  heap are 16-aligned at most, and a `<4 x double>` carrier faults in the
  byval copy). Where the IR says 8 and DWARF says 16, the IR is right: a
  `packed` record with naturally placed members has the unpacked record's
  DIEs. `_c_record_align` over-estimating is harmless for a return buffer and
  wrong for a slot, so it never chooses a carrier. Win64 marks no `byval`.
- **A signature spells a record or enum the way it is declared**, which is
  `_sanitize_c_type_name` of its C name. GeneratorC re-spells every return and
  parameter type through `_as_declared`, a table of the names the sanitizer
  changes. A raw keyword (`baremodule`) in a ccall type position fails the parse
  (abi_fuzz I1). (`test_c_sret_classification.jl`.)
- Only **ccall type positions** resolve eagerly, so an undeclared type there kills
  the module; `_assert_wrapper_loadable` checks exactly those, and refuses a C
  array spelling (`float[4]`, a vector's DWARF name) there like an undefined
  type. Unknown leaves degrade
  to `Ptr{Cvoid}` via `_resolve_forward_ptr`, gated on types **actually emitted**.
- A type constructor (`Ptr`) is not a type name. Names bound in Base/Core must never
  be rejected by a screen (`_assert_no_bound_name_rejected`).
- **Cstring policy has one derivation**, `_cstring_wrapper_pair`. Each tier supplies
  only the call body. `char*` returns become `Union{String,Nothing}` plus a `_ptr`
  sibling on every tier. (`test_cstring_policy.jl`.)
- **Only a pointer to plain `char` is a C string** (2026-11-18), in a
  parameter and in a return; `_points_to_plain_char` decides, for every tier
  (Tier 2's `_pointer_arg_type` asks it too). Every `ccall` path (GeneratorC,
  GeneratorCpp's `ccall` path, the `[wrap.varargs]` fixed parameters) binds
  such an argument through `JITManager.cstring_arg` (`_cstring_param_arg`) and
  KEEPS the byte-pointer `ccall` type: a string then has `Cstring` semantics
  (a `SubString` is copied and NUL-terminated, where it was read to its
  parent's end; an embedded NUL is an `ArgumentError`, where it truncated),
  while a `Ptr`, a `Ref` and a byte buffer still pass, which a `Cstring`
  ccall type would refuse. The conversion is bound to a name (`<p>_c`), which
  the llvmcall path derives its own temporaries from. A `signed char*` return
  is `Ptr{Int8}` with no `_ptr` sibling: the build maps it so, and
  `_signed_char_returns!` corrects older metadata at wrap from the return's
  `c_type`, rewriting only a spelling that names `signed char`. A typedef of
  `char*` the metadata records under its own name keeps what it had.
  (`test_cstring_params.jl`, devtests `test_cstring_params_live.jl`.)
- **A wrapper for a `const char*` holds no GC root when it is given a
  `String`** (2026-11-21, `JITManager.cstring_arg(::String)`). Some C functions
  never return to their caller: `luaL_error`, `png_error` and libjpeg's
  `error_exit` leave by `longjmp`, and a Julia callback calls them to raise
  the library's error. The Julia frames between the jump and its landing are
  abandoned with their GC frames still linked, so a wrapper that rooted
  anything across the call leaves the task's GC stack pointing into dead
  stack memory, and the next collection reads it: the Hub's lua died in the
  collector on every run of its deep test (found by the 4.6.0 release sweep;
  #130's conversion had inlined Base's `Cstring` check, whose `throw` builds a
  message, and that gave the whole wrapper a frame). So the `String` method
  inlines to one `memchr` and the argument itself, with the refusal
  (`_refuse_embedded_nul`) out of line. **Anything a generator adds to a
  ccall wrapper's argument path must compile to no GC frame for the
  argument types a C caller's idiom uses** (a `String`, a `Ptr`, an integer):
  read `code_llvm` for `gcframe`, as the test does. A `SubString` is copied
  and its copy rooted, and Base roots a `Vector`'s memory across any `ccall`,
  so neither is safe at a function that longjmps. (`test_cstring_arg_roots.jl`,
  devtests `test_longjmp_wrappers_live.jl`.)
- **A `String` holding a NUL is refused at every `const char*`, one that
  travels with a length included** (John, 2026-11-21: "Keep refusing").
  `mpack_write_str(w, "a\0b", 3)` and `lua_pushlstring(L, "a\0b\0c", 5)`
  passed all their bytes before #130 and are an `ArgumentError` since: DWARF
  cannot tell `(const char*, length)` from a C string. `codeunits(s)` or a
  byte vector passes, the error says so, and a Hub test that sends bytes with
  a NUL spells them that way. Do not relax the refusal for a parameter that
  looks like it has a length beside it.
- **A Julia string is refused at a `char*` that is not `const`** (John,
  2026-11-23: "refuse a string but it needs to remain functional"). The
  function may write through it, and a `String` is immutable and shared (a
  literal is one object for every run of its line: `upcase("abc")` left the
  method's literal `"ABC"`). `_points_to_writable_char` decides from the
  spelling before the last `*` (`_strip_top_cv` drops a leading `const`);
  every `ccall` path binds such an argument through
  `JITManager.char_buffer_arg` (`_cstring_param_arg`) and Tier 2 tells
  `pointer_arg` `JITManager.CharBuffer`. A `String`, `SubString`, `Symbol`
  and `codeunits` of a string are an `ArgumentError` saying to pass a byte
  buffer; everything else passes exactly as `cstring_arg` passes it, so the
  function stays callable. The refusal is out of line, like
  `_refuse_embedded_nul`, so no `Ptr`, buffer or integer argument gets a GC
  frame (`test_cstring_arg_roots.jl`). `char* const` is a buffer;
  `char *const` is no C string to `_points_to_plain_char` at all
  (`_strip_top_cv` reads the one spelling and not the other) and stays as it
  was. (`test_cstring_params.jl`, `test_tier2_pointer_args.jl`, devtests
  `test_cstring_params_live.jl`.)
- Blob structs get `setproperty`/`setproperties` built in the same loop as the getters.
  `with()` forwards to them. C++ field structs get `setproperty` too
  (`_field_setter_chunk`), so a record keeps its setters when it stops being a blob.
- **Function names come from `_julia_function_name` only.** That covers both
  generators, both varargs paths, and GeneratorCpp's proxy and deleter
  references. It is total: a catch-all after the curated list, an empty name
  falls back to the mangled symbol, and a reserved word gets a `_` suffix.
  Never inline a replace-list again. The curated part is frozen, because
  consumers call those names. **`[wrap.rename]` is applied inside it**
  (`_RENAMES`, set by `with_renames` around the generator run in
  `wrap_library`), keyed by mangled symbol or derived name as anchored globs,
  so the overload prediction, both varargs paths and the proxy and deleter
  references see the chosen name; a reference to a function must pass its
  mangled symbol to the derivation (the proxy's short name is the one call
  with `rename = false`). `_check_renames` then refuses an unmatched key and
  a chosen name another function derives or the wrapper binds otherwise. The
  old name is bound nowhere. (`test_function_name_derivation.jl`.)
- **An operator's name is a function of the operator alone** (2026-11-18).
  Which operator a function is comes from its symbol (`_operator_spelling`,
  demangled in-process), never from `Class_name` text: `Cls_foo_operator` is
  `Cls::foo_operator` or `Cls_foo::operator()`. An operator whose curated
  spelling was shared or made from its parameters takes its word from
  `_OPERATOR_WORDS` (`operatorless`, `operatorlessless`, `operatorcall`,
  `operatorindex`, `operatorminusgreater`, `operator_new_array`, …); every
  other keeps its curated name, `<=>` (`operator_assign`) and `~`
  (`operatordestroy`) included. `()`, `[]` and `,` stay bound under the old
  shared `Class_operator` as well, as copies placed before the dedup
  (`_operator_legacy_aliases`), and a `[wrap.rename]` key on the old name
  still matches; the factory proxy's bare `operator` is not kept. **Any scan
  of demangled text that tracks `<>` depth steps over an operator's token**
  (`Compiler._operator_token_at`): the `<` of `operator<` read as a template
  bracket is what left the parameter list in the name, and an operator
  template's return type in its class. Names reach a package on a re-wrap,
  an operator template's class on a rebuild.
  (`test_function_name_derivation.jl` "operators", `test_local_entity_names.jl`.)
- **A pure virtual member function has a row and no symbol** (2026-11-18,
  `Compiler._is_vtable_only`). The build writes a `vtable_only` function row
  for every public pure virtual a library record declares and no symbol
  defines (`_vtable_only_plan`: linkage name, DWARF signature and
  `vtable_slot` from the declaration DIE's `DW_AT_vtable_elem_location`, in
  front of the symbols' rows so a defined function still wins the dedup;
  the metadata's `vtable_methods` names what was passed over: a destructor,
  a private or protected member, a variadic one, a class with internal
  linkage). Its `mangled` names no symbol, so **every reader that takes a
  row's name for a symbol asks the predicate first**: `verify_wrap_surface`
  (R2 skips the row, and R1 does not count it as a surface),
  `JITManager._module_callees`, `VisibilityProbe`. Its thunk is one
  `jlcs.vcall` with no callee declared, JIT or AOT; GeneratorCpp never sends
  it to `ccall` or the varargs interception. Where the op cannot carry the
  signature (a record by value), both sides take
  `JLCSIRGenerator.vtable_only_refusal`'s verdict, which runs the thunk
  generator on the one row, and the wrapper is a refusal. Only PURE virtuals
  get a row: a non-pure virtual with no exported symbol may be hidden on
  purpose. **The constructor of a class with a pure virtual left is a
  refusal** (`record_abi[class]["pure_virtuals"]`, `_mark_abstract_records!`,
  `_abstract_constructor_refusals!`): the base-object constructor on storage
  of its own leaves an object whose pure slots hold `__cxa_pure_virtual`, and
  a library method that calls one ends the process. Abstractness is claimed
  only where the DWARF settles it (not behind a virtual base, which the VTT
  rule refuses already, nor for a declared-only base or a bare name two
  scopes define). Needs a rebuild. (`test_pure_virtual_methods.jl`, devtests
  `test_pure_virtual_live.jl`, `test_virtual_classes_live.jl`.)
- **A C enum is open: any value of its underlying type is a value of the enum**
  (2026-11-08). Never emit `@enum`, which refuses undeclared values: libvterm's
  `VTERM_KEY_FUNCTION(n)` is `VTERM_KEY_FUNCTION_0 + n`, and `VTermKey(257)` was an
  ArgumentError. `_open_enum_chunk` emits `primitive type E <:
  RepliBuild.OpenEnums.OpenEnum{T} <width> end`, the `enumerators` table, and one
  `const` per enumerator; `OpenEnum <: Base.Enum`, so integer conversion, `ccall`
  passing and `Cuint(e)` in a Tier 2 body are Base's. A declared value prints as
  an `@enum` member did (`namemap`, defined only while Base has it), an
  undeclared one as `E(v)`. A bitmask enum stays an integer alias
  (`_is_bitflag_enum`). Wrappers name `RepliBuild.OpenEnums`, so the header's
  `import RepliBuild` must precede the enum section. (`test_open_enums.jl`.)
- **The bitflag test is total over any enumerator value** (2026-11-16):
  `_is_bitflag_enum` works in `Int128`, because a 64-bit unsigned enum holds
  values `Int64` cannot (Jolt's `EIterationStatus : uint64`,
  `0xffff000000000000`), and `Int64(v)` on one threw and ended the whole
  wrap. An enum screen answers `false` for what it cannot judge, never an
  exception: one enum is not worth the module.
  (`test_c_generator_policies.jl`, devtests `test_wide_enums_live.jl`.)
- **An enumerator's value is read through `_enumerator_value`** (2026-11-17,
  both generators). `Project.toml` allows two JSON majors and they read an
  integer above `typemax(Int64)` differently: 1.x gives a `BigInt`, 0.21 an
  `Int64` wrapped modulo 2^64, with no error. The metadata on disk was right
  and the wrapper bound `Status(-281474976710656)` on an `OpenEnum{UInt64}`,
  so the live test above was green or red by which JSON the gitignored
  Manifest had resolved. The repair is made where the width and signedness
  are known: a negative value of an 8-byte unsigned enum is the value plus
  2^64, and nothing else moves (a signed enum's negative enumerator is a
  value). `_read_json` itself stays lossy on 0.21 for such integers, so a new
  metadata integer that can exceed `Int64` needs the same kind of repair at
  its reader, or to be written as a string; and a test that reads metadata
  back must pass on both majors. (`test_enumerator_values.jl`.)
- Enums with identical member sets emit as `const Alias = Canonical`, not a second
  enum type. An all-underscore name sanitizes to `c__` in **both** sanitizers. C++
  `""` stays an empty-name sentinel; do not harden it.
- **An enumerator never rebinds another top-level name.** Julia 1.12+ lets
  `struct object` silently replace `kind::object` bound by an earlier `@enum`
  (Boost.JSON, 2026-11-01). Both generators hold the enum section back as
  `_PendingEnum`s and `_render_enum_chunks` renders it last, against
  `_toplevel_names` of the rest of the wrapper's text: an enumerator whose name
  is also a record, function, global, constant, enum type or another enum's
  enumerator is bound as `<enum>_<name>` (both enums move when two share a
  member), the rest keep their names byte for byte. `_enumerator_julia_name` is
  the one derivation, and the export list carries the names the enum's `const`s
  bind. Never push enum text into `enum_chunks` directly, and never bind a
  top-level name twice from any emitter: `_assert_no_duplicate_toplevel_names`
  refuses the wrapper (methods overload, one type takes constructors, nothing
  else coexists). (`test_enumerator_collisions.jl`.)
- **A class-nested type also answers to the name its functions carry**
  (2026-11-15, `_nested_type_aliases`, GeneratorCpp only). A record is keyed
  and emitted by its bare name (`ClosestRayResultCallback`) while its
  functions are `btCollisionWorld_ClosestRayResultCallback_…`; the wrapper
  adds `const <sanitize(scope::name)> = <name>` after the layout tables and
  keys `STRUCT_SIZES`/`STRUCT_OFFSETS` by both. Additive only: the bare
  name never changes, since consumers use it. An alias needs exactly one
  scope, no `conflict`, and a scope whose innermost component is a record of
  the metadata (a namespace alone gives none), and is dropped when the
  module binds the name already (a free function `World_Mark`). It does not
  disambiguate two same-named types; that is still the open problem below.
  (`test_nested_type_aliases.jl`, devtests `test_nested_type_aliases_live.jl`.)
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

- **Every top-level name a wrapper binds from a library identifier passes the
  name screen** (2026-11-18, `Wrapper/NameScreen.jl`). It asks Julia
  (`_function_name_refusal`: syntax, a binding every module owns such as `eval`
  and `include`, a Base type whose constructor a method would extend) and the
  wrapper's own text (`_name_uses`), and keeps no list. A function moves to
  `<name>_`, a record to `c_<name>` in the metadata the generators read
  (`_screen_records`; never in a type sanitizer, since the metadata spells a
  library `struct Bool` and C's `bool` alike), an enumerator to
  `<enum>_<name>`; a module name the text uses is a wrap error naming `[wrap]
  module_name`. The first generation is a probe and a second may follow, so a
  generator must not modify its metadata. `[wrap.rename]` wins. Text the
  generators write calls Base as `Base.f`: a bare `f(` takes that name from
  every library that has one. A new emitter goes through
  `_julia_function_name` / `_screened_function_name`, never around them.
  (`test_name_screen.jl`, devtests `test_name_screen_live.jl`.)
- **A Tier 2 body converts every pointer argument and `this` with
  `JITManager.pointer_arg(<declared type>, x)`** (2026-11-18). The positions
  stay `::Any`, and `_arg_marshal_plan` cannot tell a pointer position from a
  value's, so a struct value, a Julia function or a `Vector` was boxed and read
  as an address. `_pointer_arg_type` names the position (`Ptr{T}`,
  `Ptr{Cvoid}`, `Cstring` for a pointer to plain `char`, `Ptr{Opaque}`,
  `Ptr{Code}`, `ByValue{V}` for an STL container taken by value, which the
  thunk takes by address); `pointer_arg` alone says what is accepted, and the
  rest is its `MethodError`. A new owning type defines
  `Base.unsafe_convert(::Type{Ptr{Cvoid}}, x)`; a by-value parameter the thunk
  takes by address is spelled through `_by_address_ctype` in the function loop
  AND in `_integer_width_overloads`. The `ccall` path of a C++ wrapper does not
  use it. (`test_tier2_pointer_args.jl`, devtests
  `test_tier2_pointer_args_live.jl`.)
- **`<X>_<X>` is the complete-object constructor (C1) and `<X>_destroy_<X>` the
  complete-object destructor (D1)** (2026-11-18, `_structor_bindings`). C2/D2
  stand in only for a class with no virtual base; with one and no C1/D1 the
  wrapper is a refusal. The other variant gets no chunk and no thunk, so
  nothing may rely on the order of variants in the metadata, and a consumer
  that names a thunk reads the name from the wrapper. (`test_structor_variants.jl`,
  devtests `test_virtual_classes_live.jl`.)
- **A function with C linkage in a C++ library reaches `ccall` only when its
  definition is `nounwind` in the linked IR** (2026-11-18, metadata `nounwind`
  from `_mark_noexcept!`, `DispatchLogic._is_nounwind`). `extern "C"` changes
  the name, not what the body throws, and a throw through `ccall` is
  `std::terminate`. No key is "not proven" and takes the thunk, so a C++
  package's C API runs through thunks until it is rebuilt. Its `<f>_safe` is
  emitted on either path (`_managed_safe_wrapper`). Only a per-file `-O` moves
  the count, never `[link] optimization_level` alone. A synthetic C-linkage
  function in a test that means the ccall path states `"nounwind" => true`.
  (`test_c_linkage_unwind.jl`, devtests `test_c_linkage_unwind_live.jl`.)

### JLCS dialect and ABI

- **Thunk arg slots hold a pointer to the argument's storage.** One load gives the
  storage and a second gives the value; pointer args double-load. Callers put `&struct`
  in the slot, not `&Ref`.
- **A pass-by-reference return is built in the caller's buffer** (2026-11-13).
  When `FunctionGen.returns_in_caller_buffer` holds (a `DW_CC_pass_by_reference`
  return, either a `:layout` record or an STL container by value on its own
  `:direct` path, whose byte-blob return LLVM's backend demoted to a hidden
  pointer to its own temporary), the thunk is `thunk_symbol(m, true)`
  (`<m>_thunk_inplace`), returns void, and takes the caller's buffer in argument
  slot 0, passed like any pointer argument (so the other arguments start at slot
  1); it forwards that pointer as the callee's hidden return pointer and never
  allocas or reads the return. C++ builds such an object where it will live and
  never moves it bytewise; the old thunk built it in its own frame and MLIR's
  C-interface wrapper copied the bytes out, which left any object that points
  into itself (`std::string`'s short buffer) dangling. Every call site asks the
  same predicate and name (GeneratorCpp allocates the buffer with
  `JITManager.return_slot`, calls the void `invoke`/`invoke_aot_ptr` with
  `slot_address` first under `GC.@preserve`, and returns `slot_value`, so the
  Julia return type does not change); an AOT slot's table entry is the thunk it
  resolves (`_aot_fptr_const_name!`'s `thunk`), and the manifest scan maps both
  spellings back to the mangled symbol (`_thunks_named_in_text`). The new name
  makes a stale wrapper fail as a missing thunk, with a message saying to
  re-wrap, instead of passing arguments where the buffer goes.
  (`test_by_value_crossing.jl`, devtests `test_sret_in_caller_buffer.jl`.)
- **A `std::basic_string<char>` returned by value is a Julia `String`**
  (2026-11-13, `StdStrings`). It is read where C++ built it (the caller's buffer,
  above) and ended there, never from a copy: libstdc++'s short string points
  into itself. The build records, on the string's `record_abi` entry,
  `string_layout` (`Compiler._string_layout`: leaves with dotted paths and
  absolute offsets through nested records, unions and non-empty bases, the
  character type, and the destructor declared in the record, which pass 2 files
  by the DIE open a level up); `StdStrings.string_reader` recognises libstdc++'s
  `_M_p`/`_M_string_length` and libc++'s `__l`/`__s` with the `__is_long_` bit
  and refuses anything else; GeneratorCpp's `_string_return_plan` emits
  `StdStrings.take_string!` or keeps the bytes with a comment naming the type
  and the reason (old metadata says rebuild; `wchar_t` and the other character
  types are not decoded). The destructor (`D1`) is found per host at first use,
  never at wrap time: through the wrapped library's handle, which on ELF also
  searches its dependencies (libstdc++ defines it when the library does not),
  and on PE the C++ runtime by name (`cxx_runtime_libraries(abi)`). None found:
  the bytes, and a warning naming the function once. libstdc++ homes
  `_Alloc_hider` behind its constructors, so a unit that emits no
  `basic_string` constructor (Ubuntu 24.04's libstdc++ 13 at `-std=c++17`;
  16 emits one) has it as a declaration-only DIE and `_M_dataplus` stays a
  leaf: the reader takes that leaf, of that type and spanning exactly a
  pointer (a declaration has no byte_size, so the span is the distance to the
  next leaf), as `_M_p`, and refuses any other span. Never decode by hardcoded
  offsets or sizes. **A `std::string` PARAMETER (`const&` or by value) takes
  a Julia string** (2026-11-15): the build records on `string_layout` the
  declared constructor that builds from `(const char*, size_type[, const
  allocator&])` (`_string_constructor`, from the record's `C4` declarations,
  never a hardcoded mangling), GeneratorCpp's `_string_param_plan` types the
  position `Union{AbstractString, Ptr{Cvoid}, STLWrappers.CppString}`
  (`_STD_STRING_PARAM`, one spelling, read by the loop
  and by `_integer_width_overloads`' prediction through its `string_param`
  predicate, so a `const char*` sibling stays its own method) and the body
  builds the object in a slot (`StdStrings.string_arg`, the constructor found
  per host at first use like the destructor), passes its address (the thunk
  takes both spellings as a pointer; `DW_CC_pass_by_reference` makes by-value
  the caller's temporary too) and ends it in a `finally` (`release!`). A
  pointer or a `CppString` passes through unended (stl_test's
  `string_len(CppString)`). Such a function takes Tier 2 even when
  `noexcept`. No constructor on the host is an error naming the function,
  never bytes it did not build. (`test_std_strings.jl`, devtests
  `test_sret_in_caller_buffer.jl`.)
- **One struct-ABI rule set per convention, chosen by the host, in two places
  that must agree.** `RepliBuild.host_record_abi()` (`:sysv`, `:win64`,
  `:aapcs64`) is the one place Julia reads the host; every classifier takes it
  as `abi`, and `generate_jlcs_ir` / `generate_function_thunks` take it too, so
  a test generates another host's module here. `AbiTarget` / `kHostAbi` in
  `JLCSPasses.cpp` is the C++ twin (`#error` for any other host, by name). A
  module states ONE convention's facts: `sysv_args`/`sysv_ret`, or
  `aapcs64_args`/`aapcs64_ret` for an AArch64 host (`record_statement`,
  `record_statement_attrs`); the dialect reads its target's and falls back to
  the body. The probe seams (`jlcs_probe_sysv_call_shapes`,
  `jlcs_probe_aapcs64_call_shapes`) run the real `buildCallShape` under an
  explicit target on any host; `-DJLCS_FORCE_ABI_WIN64` / `_AAPCS64` (CMake
  options) build another host's LOWERING for inspection, never over the
  working libJLCS.
- **AAPCS64 (Linux AArch64)** (`ir_gen/AAPCS64RecordABI.jl`,
  `classifyAAPCS64Struct`; rules read off clang 23): an HFA (at most four
  leaves of one floating type, no padding, nested records and arrays
  flattened, a union's members once per position, empty bases skipped) goes
  in floating registers at ANY size; otherwise 16 bytes or less in general
  registers, `i128` (an even pair) when the alignment is 16, where an
  ARGUMENT reads the unadjusted alignment (members' and nested records' own,
  never the record's own `aligned`: `_sysv_record_align(…; own = false)`) and
  a RETURN the adjusted one, a return of 8 bytes or less `iN` of its size;
  over 16 bytes a pointer to a copy with no `byval`, or sret. Field alignment
  never matters. The coerced block is ONE `regs` entry (`i64`, `[2 x i64]`,
  `i128`, `iN`, `[N x fT]`), stored and reloaded through a 16-aligned slot:
  an array or `i128` keeps the backend's registers consecutive where a
  literal struct would not, and the backend spills a block whole, so the SysV
  register-count demotion stays SysV's. A C++ empty record is one `i64` in
  and nothing out (`by_value_crossing`'s `ret`). A record holding a SIMD
  vector is refused (HVA not modelled). `long double` is
  `C_LONG_DOUBLE_MLIR` (`f128` there). GeneratorC: `_c_class_window` (16 / 64
  / 0) replaces the literal 16 of the register-window refusals;
  `_c_return_in_memory` is never `true` on AAPCS64 (an HFA over 16 bytes
  returns in registers; Julia's lowering decides from the stand-in's fields);
  `_c_register_pair_misplaced` is its one trap. **RuntimeDyld on AArch64 ELF
  needs `setAutoClaimResponsibilityForObjectSymbols`** (`JlcsJIT.cpp`): the
  `.eh_frame` personality goes through a per-object `DW.ref.__gxx_personality_v0`
  slot the MC layer emits after the IR layer listed the symbols, and unclaimed
  it is "Symbols not found" for every thunk of the library (first arm64 run,
  2026-11-09). **A hidden return pointer needs `llvm.sret` there**: AArch64's
  indirect result register is x8, not the first argument register, so the
  generator states a pass-by-reference record's slot (`sret:<bytes>`) and
  `buildCallShape` stamps the attribute on it and on its own sret slots,
  under AAPCS64 alone (x86-64 IR unchanged; `Ticket` came back as garbage). (`test_aapcs64_record_abi.jl`, devtests
  `test_aapcs64_abi_spec.jl` against clang's aarch64 IR; live only on
  tier2-linux's arm64 cell.)
- **A 16-byte vector is (SSE, SSEUP) and merges like any class** (2026-11-16,
  the `:vec16` leaf). The classifier refused any record holding one as "one
  SSE register no call statement states", which is true of a vector alone.
  In a union the merge decides, and clang agrees: beside the array of its
  lanes (Jolt's `Vec3`: `union { __m128; float[4]; }`) SSEUP against SSE is
  SSE, so (SSE, SSE), two registers, `{ double, double }` in clang's IR;
  beside integers (`UVec4`) it is (INTEGER, INTEGER). Both are `regs:`
  statements the dialect already acts on, and about 1,000 Jolt functions
  were traps for nothing. Only where (SSE, SSEUP) survives the merge (a
  vector alone, or beside a member covering one eightbyte) is the record one
  XMM register and `:unstated`. Read what clang emits for the shape before
  believing a description of it: the Hub's report said "one SSE register".
  AAPCS64 refuses a `:vec16` leaf as it did `:unstated`.
  (`test_sysv_abi_oracle.jl`, devtests `test_simd_union_live.jl`.)
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
  A member's type is spelled by `StructGen._member_mlir_type`, never `map_cpp_type`
  directly: that answers `!llvm.ptr` for a template spelling (`Vec<int, 3>`,
  `pair<int*, int>`), and every record holding one shrank. A tiled body
  (`is_struct_packed`) is measured too: emitted members that do not add up to
  `byte_size` fall to the offset-driven layout (`_apply_dwarf_layout`), since a
  member spelled to the wrong type tiles by its DWARF size and was returned short.
  A synthetic fixture that sizes `long` at 8 is LP64-only: `long` is i32 on Win64,
  so the measurement re-lays it out there (Windows CI, PR #72); size 8 is `long long`.
  To check a StructGen change for that without Windows, rebind
  `RepliBuild.C_LONG_MLIR` to `"i32"` with `Core.eval` and run the file.
  (`test_struct_layout.jl`, `test_packed_layout.jl`, `test_thunk_byte_regions.jl`,
  `test_template_members.jl`, `test_toolchain_records.jl`.)
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
  DWARF `DW_CC_pass_by_reference`, and such a return goes through the caller's
  buffer (above), first argument, before `this`. Not at all means an ABI trap on the
  Julia side.
  `FunctionGen.by_value_crossing` is the one verdict; the thunk generator and
  GeneratorCpp both act on it. `record_abi` in metadata carries the DWARF facts. A
  record must never fall to `!llvm.ptr`: that is how empty classes, declared-only
  types, `string_view` and every template spelling were mis-passed. **A `*` or
  `&` inside template arguments is not the type's declarator**: every test for a
  pointer, reference, function or array spelling reads
  `FunctionGen._outside_template_args(t)` (`_table_entry`, `_sysv_record_of`,
  `is_ccall_safe`), because `pair<key_value_pair *, bool>` and `result<T &, E>`
  classed as pointers returned a 16-byte record in a register, which on Win64
  lacks its hidden pointer (Boost.JSON `object::emplace`, access violation,
  2026-11-01). The converse holds for the thunk's return type: a `*` or `&`
  on the type itself makes the return `!llvm.ptr`, whatever it points at, and
  the STL-size fallback in `generate_function_thunks` never measures the
  container behind a reference (`path::native()` returned a stack slot).
  (`test_by_value_crossing.jl`, `test_cpp_return_resolution.jl`.)
  **The build reads the name the same way** (2026-11-23,
  `Compiler._by_value_record_name`): it decides which records get
  `record_abi` facts, and it read the declarator characters over the whole
  spelling, so `unique_ptr<int[], default_delete<int[]>>`, `vector<Node *>`,
  `function<int (int)>` and an empty `Tag<int *>` had none. With no `pass`
  fact a `std::unique_ptr<int[]>` was returned as eight bytes in a register
  while the callee wrote through the first argument register, and a class
  with a destructor (`Holder<int *>`) was returned and passed in registers.
  Needs a rebuild.
  (`test_template_arg_records.jl`, devtests
  `test_template_arg_records_live.jl`.)
- **A record C++ takes by value through the caller's object's address, with
  no layout here, is a pointer position** (2026-11-23,
  `FunctionGen.takes_object_address`: `:indirect` and no table entry, a
  `std::function` by value). The thunk's slot holds the object's address, so
  GeneratorCpp spells the position as a pointer (`_by_address_ctype`, in the
  loop and in `_integer_width_overloads`' prediction through its
  `object_address` predicate) and converts the argument with
  `pointer_arg(Ptr{Opaque}, x)`. Spelled by its own name the position took
  only a value of the empty struct that name is bound to; passed as it came,
  a Julia function was boxed and the box read as the object. An STL
  container keeps `ByValue`, and a record with a layout crosses as its
  bytes.
- **A toolchain record the library reaches by value keeps its layout**
  (2026-10-29, merged 2026-11-08). The provenance gate keeps a record declared in a
  toolchain header out of `struct_definitions`, so `std::pair<double, double>`
  returned, `std::string_view` taken, or a library `Span { std::pair<int, int>; … }`
  had no layout and every crossing trapped. The gate now sets those definitions
  aside, and `Compiler._promote_toolchain_records!` puts back what an **exported**
  function or global crosses by value (the dynamic-symbol gate, read with
  `_runtime_defined_names`, which lives in the Builder for this; constructors
  through `_structor_dwarf_key`; less `_is_stdlib_internal`, since libstdc++'s
  `namespace std` has default visibility and its instantiations are in dynsym
  even under -fvisibility=hidden), and what each record so reached holds by value,
  library records included, transitively; marked `"toolchain" => true`, **only
  when the whole by-value closure has a layout**. Ungated, ggml-vulkan's hidden
  C++ promoted 123 Vulkan-Hpp records, `vk::Event` and `vk::Semaphore` among them
  (Base names), and with dynsym alone 55 `_Rb_tree_iterator`-class records; gated,
  llama.cpp promotes 0 and boostmath exactly `pair<double, double>`. A library record nothing exported crosses by value promotes
  nothing, and neither does a symbol `[wrap] exclude_symbols` drops: the exported
  set loses it first (`_without_excluded_symbols`, matched as wrap matches, and a
  global on its exported name because `symbols` is code only), or glfw's exported
  `_glfw` state global kept glibc's `pthread_mutex_t` and friends. It never promotes an STL
  container (`stl_container_size`; those keep their own path) or an
  `INTERNAL_TYPE_BLOCKLIST` name, and a library record of the same name keeps its
  entry. **A toolchain record is laid out from its bases** (2026-11-13,
  `_laid_out_from_bases`): libstdc++'s `tuple` and `optional` have no member of
  their own, so the parser gives a toolchain record whose bytes come from
  non-virtual bases an entry (a library record still gets none), and promotion
  puts every non-empty base's members, at depth, at the base's offset, in
  place of the base; a shared name is numbered (`_M_head_impl_1`), and a union
  drops a 1-byte empty-record member (`_Storage`'s `_Empty_byte`). Readers then
  see members at offsets, as for any record. `complex` is sized by the parser
  (`_complex_base_name`). `std::complex<long double>` is callable on both
  conventions: MEMORY (sret) on x86-64 SysV, an HFA of two `f128` on AArch64
  (`long double` is a quad there), so a test reading one back reads it in C++,
  never by decoding x87 bytes. A 1-byte empty base (`__pair_base`, a library `Tag`) is marked
  `"empty"` on the base entry and the SysV classifier skips it; unmarked, every
  record deriving from one trapped, the library's own included. Whether a member
  is held by value reads `_outside_template_args` (`pair<int*, bool>` is a
  record). `stl_container_size`, `_scope_suffixes` and `_outside_template_args`
  live at package level because the Builder asks them too.
  (`test_toolchain_records.jl`, devtests 8g `test_toolchain_byvalue_live.jl`.)
- **A qualifier is cut before a C spelling reaches `map_cpp_type`**
  (2026-11-21). `map_cpp_type` is a table of exact spellings, and anything it
  has no entry for is `!llvm.ptr`: `const int`, `const double`, `const Kind`
  and `volatile int` all were. Three places handed it a qualified spelling,
  and each was a different failure:
  - a record MEMBER (`StructGen._member_mlir_type`, which cut the qualifiers
    for its template checks and then passed the uncut spelling on): eight
    bytes for a four-byte member, so the body failed its own size check and
    the record fell to a byte region. 70 of llama.cpp's 951 records.
  - a by-value PARAMETER and a RETURN (`generate_function_thunks`): a
    definition's DWARF keeps the qualifier (`double scaled(const double x,
    const int n)`), the callee was declared `(ptr, ptr) -> f64`, the double
    went to an integer register, and `scaled(2.5, 4)` was 0.0 with no error.
  A pointer's or reference's spelling maps the same with or without them, so
  the cut is `_strip_top_cv` at the call, never inside the table: the
  `jlcs.type_info` bodies (`JLCSIRGenerator`, dumper names) still pass
  qualified names and get `!llvm.ptr`, and a named ref there with no
  definition would not parse. **A new caller of `map_cpp_type` with a C
  spelling from the metadata cuts the qualifiers first.**
  (`test_const_members.jl`, devtests `test_const_members_live.jl`.)
- **A field-by-field crossing asks what StructGen emitted, not what the
  metadata says** (2026-11-21). `is_struct_packed` reads the metadata (the
  members tile the record). The body can still be a byte region: StructGen
  measures what it emitted, and one member it cannot type (a bit-precise
  integer, anything the table lacks) sends the whole record there. Both
  crossing paths took the body as the "packed" type and emitted
  `jlcs.marshal_arg` / `jlcs.marshal_ret` on it; on `!llvm.array<N x i8>`
  neither verifies, and one such thunk loses the library's whole module:
  llama.cpp's Tier 2 did not initialise on the 4.6.0 release sweep
  (`LLM_TN::operator()` returning `LLM_TN_IMPL`, an inline method the forced
  unit defines). A byte-region body crosses as `_byte_blob_type(N)` on the
  packed branch too, as it always did on the other.
  (`test_const_members.jl` "B", devtests `test_const_members_live.jl`.)
- **A container's by-value size is the DWARF's, then the table's**
  (2026-11-23, `FunctionGen.stl_value_bytes`). `stl_container_size` keys on
  the template's name and one name has several sizes: `std::vector<bool>` is
  40 bytes in libstdc++ where every other `std::vector` is 24, and libc++'s
  containers are not libstdc++'s. The thunk side already read the size from
  `record_abi` (`_record_abi_bytes`) while GeneratorCpp sized its return
  buffer and its `ByValue{NTuple{N}}` from the table, so a returned
  `vector<bool>` was built 40 bytes long in a 24-byte buffer. Every place
  that sizes a container VALUE asks `stl_value_bytes(c_type, record_abi)`;
  `stl_container_size` alone answers only "is this a container". The table
  is the fallback for metadata with no fact and must never be smaller than
  the object (`vector<bool>` is 40 there). The `create_std_*` factories
  still size from the table: they have no `record_abi` entry to read (Not
  Yet Built). **`std::vector<bool>` gets no `CppVector` factory**
  (`_is_bool_vector_element`, a warning naming it): it is bit-packed, its
  `operator[]` returns a proxy and `push_back` takes a `bool` by value, so
  made as a `CppVector{Bool}` `push!` stored wrong bits and indexing read the
  proxy as an address. (`test_template_arg_records.jl`, devtests
  `test_template_arg_records_live.jl`.)
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

- **A by-value class parameter's temporary is made from the class's
  declarations, and the module declares what it calls** (2026-11-18).
  `_collect_class_raii` (keyed by the `struct_definitions` key) and
  `FunctionGen.value_temporary` decide: copy constructor; move constructor
  where C++ has no copy; bytes only where the copy is trivial; else
  `:uncopyable` and a trap. A symbol's absence never means "trivial", and a
  temporary is ended by D1, or D2 where it takes no VTT, never by D0. Every
  callee a `jlcs.scope` or `ctor_call` names is declared once by
  `generate_function_thunks`, only where nothing else declared it (a second
  `func.func private` kills the module), and pre-flighted
  (`lifetime_callees`). `undeclared_callees` refuses a module before the JIT
  and the AOT stage: MLIR's translation dereferences null for an undeclared
  callee, which killed the process at `using`. A new op that names a callee
  must be read by it. (`test_byvalue_class_temporaries.jl`, devtests
  `test_byvalue_class_live.jl`.)
- **Every virtual method with a slot is a `jlcs.vcall`** (2026-11-18). The
  class gate is `_has_type_info` (a size, and members or virtual methods; `[]`
  is not a field list the dialect parses), `DWARFParser._merge_class_dies`
  keeps a class's definition when a later unit only declares it, and
  GeneratorCpp sends every `virtual` row of `record_abi[...]["methods"]` to
  Tier 2, `noexcept` or not: a `ccall` is a static call. (`test_virtual_dispatch.jl`.)
- **`ccall` passes `Int128`/`UInt128` as two halves**, which is C's placement
  only while both get registers (2026-11-18). `_c_int128_misplaced` traps in
  the C generator, `is_ccall_safe` sends C++ to the thunk (`i128`), and
  `thunk_crossings` refuses it under Win64, on a reading of clang's MinGW
  output no Windows host has run. (`test_type_mapping_live.jl`.)
- **A name that is a record of the library is that record**
  (`map_cpp_type(t, records)`): the scalar map also answers Julia's spellings
  of C types (`Bool`, `Cint`, `UInt8`), and a `struct Bool` by value was a
  one-byte thunk return. (`test_name_screen.jl`.)

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

- **`llvm-dwarfdump` is the system's first, and a bundled copy only from
  LLVM 20** (2026-11-17, `Tools.PROVIDERS`: `system_first`, `min_llvm`).
  `DWARFParser` reads the type NAMES the dumper prints, and they reach the
  thunk IR (`jlcs.type_info` bodies and base names). LLVM 18 prints a typedef
  without its scope (`__to_type` for `std::ctype_base::__to_type`): on the
  Hub's 42 libraries Julia 1.13's bundled LLVM 20 dumper gave a system LLVM
  23's classes exactly, and Julia 1.12's LLVM 18 differed on 11, with
  different IR for 9 of 10 C++ packages. So the dumper of the LLVM that
  compiled the library wins wherever it is on PATH, the bundled one is what a
  machine with no LLVM has, and on a Julia whose LLVM is older there is none.
  A tool that reads DWARF is not version-free because DWARF is: check what it
  PRINTS at each bundled LLVM, on both supported Julias, before making a
  bundled copy the default. (`test_tools_resolver.jl`, devtests
  `test_bundled_dwarfdump_live.jl`.)

### Caching

1. Per-file IR cache (`.replibuild_cache/`), keyed on a compile-fingerprint `.key`
   sidecar. The hash in the IR filename identifies the source, not the flags.
2. Project content hash (`.replibuild_cache/project_hash`): TOML, sources, include
   dirs (recursive), git HEAD, and the toolchain (`_toolchain_identity`: Julia's
   libLLVM for C, the system clang for C++, plus the triple). A Julia switch
   rebuilds a C project and leaves a C++ one built.

**A marker names what was built.** `compile_project` hashes the project once,
before anything reads it, and saves that value. On a fast-path miss it removes
the marker before building, so a build that fails after its link leaves no
marker. A compile counts only on exit 0 with its output present, and the old
IR and `.key` go first (`compile_single_to_ir`, `GraphBuild._compile_one`): a
compiler that never started deletes nothing. A new build path keeps all three
(`test_project_hash.jl`, `test_compile_never_ran.jl`,
`test_build_markers_live.jl`).

Every persisted key (these two, the registry's `hash_config`, the IR file name's
directory digest `_ir_dir_key`, the templates TU's `.key` fingerprint, and graph
mode's `link.key` and `metadata.key`) is
SHA-256 fed through `_key_feed!`, never `Base.hash`, which differs between Julia
releases.
3. Registry build cache (`~/.replibuild/builds/<hash>/`, gates `use()`). It includes
   `_generator_fingerprint`: RepliBuild version, HEAD, **and a dirty digest of
   `src/` and `Project.toml`** (`_GENERATOR_DIRTY_PATHS`), so uncommitted generator
   edits miss while a dirty CLAUDE.md or CHANGELOG.md does not re-vendor every
   build. `_store_build` copies directories.

`RepliBuild.VERSION` is derived from `Project.toml`; never reintroduce a literal. It
feeds the fingerprint and `BUILD_GENERATOR`. To check what is registered in General,
read `R/RepliBuild/Versions.toml` from the tarball that `path =` in
`~/.julia/registries/General.toml` names: `General.tar.gz` (`tar -xzOf`) or
`General.tar.zst` (`tar --zstd -xOf`), depending on the Pkg version. A tarball
beside it that `path =` does not name is stale and lacks recent versions. 3.3.2
is a permanent hole there. Registration is manual and
outward-facing, so ask first.

### Rust front end (in progress)

- **A Rust project never enters the C/C++ pipeline.** `compile_project`, the
  generators and ~30 sites in `Compiler.jl` take any language that is not C
  for C++ (`language == :c ? a : b`). `[rust]` selects the front end
  (`config.rust`, like `config.graph`), `build` branches on it, and
  `compile_project` and `wrap_library` refuse a Rust config. Anything that
  copies a `RepliBuildConfig` carries `config.rust`: `resolve_dependencies`
  runs before the branch.
- **rustc's ABI is read, never predicted.** The call is the function's
  `define` line in the IR of THAT build (`RustABI.definitions`), read as text;
  rustc's IR is never loaded into an LLVM. DWARF says what each parameter is,
  and `join_function` matches the two: every option (`ignore`, `direct`,
  `pair`, `cast`, `indirect`) is checked against the IR types it would take,
  and a function whose IR parameters are not all accounted for in exactly ONE
  way is a refusal with its reason, never a guess. A new way rustc passes
  something goes into `_options` and the return arm together.
- **`ccall` is right for a Rust function only where every IR type is a scalar
  or a pointer.** A two-scalar return (`{ i32, i32 }`: `Option<i32>`) is wrong
  through `ccall` and right through a thunk that states the define line; on
  SysV some pairs come out right by coincidence, which is not evidence. A
  hidden return pointer is the first argument on x86-64 only (AArch64: x8).
  A Rust panic through `jlcs.try_call` segfaults: the catching frame must be
  Rust's (`catch_unwind`), outside a thunk that makes a plain call.
- **The surface is exported ∩ defined in the crate's IR ∩ the crate's own.**
  "The crate's own" is the root of the v0 mangled path (`RustABI.v0_root`,
  which parses the name: an `impl` block's path is not in the demangled
  text), or a name rustc did not mangle. Never the crate's name appearing in
  a symbol: an instantiation of `core`'s generic carries it as a suffix. A
  Rust library exports the standard library (1,743 symbols for 17 functions)
  and its DWARF holds std's too.
- **A function the optimizer merged into another stays a function of its
  own** (`RustBuild.MERGED_FUNCTION_FLAGS`, two LLVM options through
  rustc's stable `-C llvm-args`). Above opt-level 1 LLVM keeps one of two
  functions with the same code; rustc's default leaves the other as an IR
  ALIAS, with no `define` line and no DIE, so its Rust types are nowhere
  (`flip(Side)` came out typed as `flag(bool)`, and `string_ref_len(&String)`
  as taking a `Secret`). With the options the other is a function that jumps
  to the kept one, with its own `define` line and debug info. They are
  LLVM's options: an LLVM without them refuses to start
  (`_unknown_llvm_argument`), and the build then runs without them. So the
  alias path stays (`RustABI.aliases`, row key `alias_of`,
  `own_debug_info: false`: the call is right and the types are the
  TARGET's), and a generator must not present such a row's types as the
  function's own. A one-byte value that is only 0 or 1 (a `bool`, a
  two-variant enum, `Result<(), E>`) is `i1` in the call.
- **An optimized build keeps incremental compilation on.** Above opt-level 0
  rustc gives a small `pub fn` no symbol unless incremental is on (4 of 17
  kept one at opt-level 3; 17 of 17 with it). `cargo_args` states the whole
  profile (`codegen-units=1`, so exactly one `<crate>-<hash>.ll`); changing
  one setting there needs the surface re-measured. **The profile is not the
  last word on a setting, so what the build depends on is pinned where
  nothing overrides it**: `CARGO_INCREMENTAL` and `CARGO_BUILD_INCREMENTAL`
  override every profile and are removed from cargo's environment (`_cargo`;
  GitHub's runners set `CARGO_INCREMENTAL=0`, and every optimized build in CI
  lost a third of its symbols, 2026-11-20), and `build.incremental` in a
  configuration file does too, so it is stated on the command line. Before
  relying on a cargo setting, ask what overrides it.
- **Where rustc writes a crate's files is cargo's business, and it is read
  from the names cargo reports** (`_output_dirs`, `root_ir`, `_crate_ir`).
  Through cargo 1.99 every crate's files were in `target/debug/deps/`; from
  1.100 each unit has `target/debug/build/<package>/<hash>/out/`, and the
  build, which looked for the root crate's IR beside the wrapped crate's,
  failed on its first line with a new cargo. A hashed file name cargo reports
  (the `.rlib`, or the `.rmeta` beside the unhashed copy it makes for the
  user) gives the directory and the hash; the root crate's IR is beside the
  file the reported library is a copy of, since a profile directory keeps
  one unit per set of flags. Never join `deps` to a path.
  (`test_rust_config.jl`.)
- **The Rust toolchain is the user's and moves under the tests.** CI uses
  the runner image's Rust (`tier2-linux.yml` prints it), which is newer than
  a dev box's. Run the three live Rust tests on the newest stable as well as
  the installed one before calling a Rust change done:
  `RUSTUP_TOOLCHAIN=<version> CARGO_INCREMENTAL=0 julia --project=.
  test/test_rust_<build|thunks|wrap>_live.jl` (`rustup toolchain install
  <version> --profile minimal` puts one beside the default). An `#[inline]` function
  has no symbol at any level: the library has it only because the build
  names it in the root crate (below).
- **A TOML entry only for what no compile of the same code can say** (John,
  2026-11-19, "same with c and cpp"). So `[rust] instantiate` exists, for a
  generic function nobody instantiated: nothing in a compile of the crate
  says which type a user wants. And nothing else about a crate has an entry:
  its `#[inline]` functions are the compiler's to list, and are found
  (`RustBuild.unexported_functions`: a second compile of the wrapped crate
  alone with `-C link-dead-code`, into the same target directory, linked
  into nothing; the normal build stays up to date). Before proposing a
  `[rust]` key, ask whether a compile could answer.
- **A function with no symbol is NAMED in the root crate, and that is all**
  (`rb_glue.rs`: `pub static __rb_fn_<n>: __RbAddr = __RbAddr((<expr>) as
  *const ());`). rustc then compiles it there; the root crate's IR
  (`deps/<root>.ll`) says which function each static points at
  (`RustGlue.address_table`), and its `define` line and DIE are read like
  any function's. Never state such a function's signature in generated
  Rust. Its metadata row has `address` (the static) and `instance`
  (`"inline"` or `"instantiate"`), and **it is called by address, never by
  name**: the module reads the static with `cglobal` and hands the value to
  a plain call or to the function's thunk (`RustThunks.callee_addresses` is
  the same answer for `call_raw`). Anything that takes a row's `symbol`
  for a name the loader binds must ask for `address` first.
- **A found function's path is spelled like a type's** (`function_expression`,
  then the probe's chosen paths), and one with no public path is not the
  crate's API: it is left out without a word, and only counted
  (`instances.not_public`). An `instantiate` entry that does not compile is
  the project file's own line: a build error with the compiler's message.
- **A type no path spells is taken from a function that returns it**
  (`RustGlue.positions`, `__rb_dropfn_<n>`): a `const fn` given the function
  has the compiler pick `__rb_drop_owned::<T>` or `__rb_drop_plain::<T>` for
  its return type, or for the `Ok`, `Err` or `Some` inside it. Which of the
  two is "is this type dropped"; the picked function's DIE names `T`, which
  is checked. Such a `glue` entry has `by_address`. Spelling is tried first
  (it also gives a `Vec` its two functions).
- **A function pointer and the signature it points at are two types**
  (`RustDWARF`: the pointer is `fn(i32) -> i32`, its target `{fn} fn(i32) ->
  i32`). rustc names the pointer and leaves the signature unnamed; keyed
  alike they were one entry marked `conflict`, and every function taking a
  callback was refused from the first build side on (found 2026-11-19).
  A key built from a spelling must not be one rustc also uses as a name.
- **A Rust function pointer is never made outside Rust.** How one is called
  is the compiler's business. Per kind taken (`RustGlue.callbacks`), one
  line of `rb_glue.rs` is a Rust function of that signature (the library is
  given THAT, through `__rb_cb_at_<n>`), which calls on to an `extern "C"`
  function in a slot (`__rb_cb_set_<n>`). The C function is a plain function
  of the module, one per kind (`_rb_cb_<k>`, the "Callbacks" section), which
  calls the Julia function the wrapper holds in `_RB_CB_F_<k>` for the length
  of the call: the holder and the slot are set under `_RB_CALLBACKS` and put
  back in a `finally`. **Never `@cfunction(\$f, …)`**: a closure is made a C
  function only on some platforms, and on AArch64 every callback failed with
  "closures are not supported on this platform" (first AArch64 run,
  2026-11-20; `test_rust_wrap.jl` refuses one in a module's text). An error
  in the Julia function is kept and thrown
  after the call returns: it must not unwind through Rust. An `extern "C"
  fn` parameter takes Julia's C function directly. A callback carries only
  what C does (`emit_function`'s callback arm); anything else is a stub with
  the type named. The slot holds for the length of the call: a callback the
  library keeps is not built.
- **A reference to a trait object is written by the library**
  (`__rb_dyn_<n>(p: *const T, out: *mut &dyn Trait)`, one per implementing
  type): the second pointer is the type's table of methods, which only the
  compiler has. Which traits, types and methods comes from the paths of the
  functions the build describes (`RustGlue.trait_objects`: `<T as
  Trait>::method`), and the function's DIE says what its line named
  (metadata `dyn`). A trait's method through a trait object is the
  compiler's shim, named `<dyn Trait as Trait>::method` like any function
  with no symbol (`instance: "dyn"`, Julia name `Trait_method`). `&mut dyn`
  refuses a plain Julia value (`_rb_dyn_mut_<Trait>`): the change would be
  made to a copy.
- **A static is read where it is, each time** (`_emit_statics`: `NAME()`,
  `NAME_ptr()`): never copied into the module, since a `static mut` or an
  atomic changes. The statics are the crate IR's exported globals
  (`RustABI.statics`) that a variable DIE describes.
- **A destructor is never offered as a function** (`<T as
  core::ops::drop::Drop>::drop`): Rust refuses a direct call of one, and a
  value dropped that way would be dropped again by its owner.
- **What the project file asks for is part of the metadata key**
  (`_metadata_key`'s `asked`): `instantiate` changes the library only once
  a build acts on it, so a key of the library and IR alone served the old
  metadata for a changed list.
- **The library is a generated root crate; the wrapped crate is its
  dependency** (`root_files`, `<[paths] build>/rust/root`). A `#![no_std]`
  crate cannot be the end of a link (libm: "`#[panic_handler]` function
  required"), a pinned checkout must not be written into (the lock file is
  the root's, seeded from the crate's), and the Rust RepliBuild must supply
  itself (the catching frame, the drop functions) needs a crate to live in.
  Consequences
  to keep: the flags reach EVERY crate through `CARGO_ENCODED_RUSTFLAGS`
  (`rustflags`; what follows `--` reaches the root alone), with
  `--cap-lints=warn`, because a crate that denies its own warnings stops
  building when rustc grows one and cargo does not cap a path dependency;
  the crate's IR is `deps/<crate>-<hash>.ll`, named from the `.rlib` cargo
  reports for the crate's own manifest (`find_artifacts`); cargo runs from
  the crate's directory, where rustup reads `rust-toolchain.toml`.
  (devtests `test_rust_build_live.jl`, the `nostd` fixture.)
- **`[rust] dependency` is the Hub's shape**: the manifest is relative to a
  `[dependencies]` checkout (`RustBuild.manifest_path`,
  `ConfigurationManager._checkout_relative`, shared with `[graph]`), and
  everything in `RustBuild` takes the located config (`_located`), never
  `config.rust.manifest`.
- **`jlcs.native_call` is the Rust front end's op and nobody else's.** It
  lowers to exactly the call it states: operand and result types as given,
  `arg_attrs` (one dictionary per operand, the LLVM dialect's names) copied
  to the call site and the declaration, no `CallShape`. With no `callee`
  attribute the first operand is the function's address and the call goes
  through it (`arg_attrs` is then one dictionary per ARGUMENT, which the
  address is not): the form `RustThunks` emits. Anything added to its
  lowering is a second ABI rule. The C and C++ generators must never emit it
  (`test_jlcs_invariants.jl` §E), and a new op must be in the conversion
  target's illegal list, or a failed lowering pattern is not a failed pass.
- **A Rust thunk is `void thunk(void **args, void *ret)`** (`RustThunks`):
  one slot per IR parameter holding a pointer to that parameter's value (a
  `ptr` parameter is stored as a pointer, no exceptions), the hidden return
  pointer left out because it is `ret` itself, a returned pair stored at the
  offsets DWARF gives its two scalars, a `bool` a byte in storage. Only the
  attributes that change how an argument is passed go on the op (`_arg_attrs`:
  zeroext, signext, inreg, sret, byval, and `align` with those two).
- **A thunk names no function: its last slot is the function's ADDRESS**
  (itself, not a pointer to it), and the thunk calls through it. So a thunk
  module declares nothing and has nothing for a loader or a JIT to resolve
  (ORC waits forever on a symbol it cannot find; there is none to find), the
  same text is JIT'd or compiled ahead of time, and a function the library
  exports and one it has no symbol for are called the same way. Never put a
  callee's name back into a thunk: the ahead-of-time library is linked
  against nothing and would not load. What the slots hold is a named
  convention (`RustThunks.CONVENTION`, `rust-thunk-2`): the module passes it
  to `RustRuntime.Library`, and a module written for another (or for none: a
  wrapper from before this) is refused there, before any call, since its
  slots would be read wrong. Change what a slot holds and the name changes
  with it.
- **`wrap` compiles the thunks, and there is no key for it**
  (`GeneratorRust.compile_thunks`, `RustThunks.aot_library`,
  `lib<name>_thunks.<ext>` beside the library). Which functions call through
  a thunk is the generator's answer, so the stage runs on a first
  generation's report and the text naming the library is a second one, with
  the same calls. The library depends on NOTHING, by design: it is linked
  `-nostdlib` with the C bucket's clang and linker, so a thunk must stay
  loads, stores and one call (anything that makes LLVM ask the C runtime for
  a function would be an undefined symbol; `test_rust_thunks_live.jl` reads
  `NEEDED` and the undefined symbols). A stage that writes no library
  (no libJLCS, no `llc`: `MLIRNative.emit_object` starts it) is a JIT
  wrapper and never a failed wrap; an older thunks library is removed first.
  The module bakes the file's hash, and `RustRuntime` uses the file only
  when it is that file, else the JIT.
- **A Rust value crosses as its bytes** (`GeneratorRust`). A record, tuple
  or enum with data is a Julia struct of `_rb_bytes` with a property per
  field at its DWARF offset; a by-value argument is encoded into a byte
  buffer and the call's IR arguments are read from that buffer where the
  join says (`pass`, and the leaves' offsets for a pair). Never a Julia
  struct of fields: the compiler reorders fields and fills niches. The
  generator's own calls into Base are qualified (`Base.convert`): a crate's
  function names fill the module (`test_rust_wrap.jl` holds the list).
- **A Rust function is a plain `ccall` only when every type of its `define`
  line is a scalar or a pointer AND it is `nounwind`** (`emit_function`'s
  `plain`); else its thunk, and inside the catching frame when it is not
  `nounwind` (`RustRuntime.invoke`, `guarded`). A panic through a bare
  `ccall` or an unguarded thunk ends the process. One bit (`i1`) and a
  hidden return pointer are the thunk's, never a `ccall`'s.
- **Only Rust ends a Rust panic.** The root crate carries `__rb_catch` and
  `__rb_panic_message` (`RustBuild._CATCH_FRAME`): RepliBuild's own
  functions, so `extern "C"` and unmangled, and never part of the surface
  (the surface is the WRAPPED crate's IR). Its panic hook records a panic
  only inside `__rb_catch` and passes any other on to the previous hook.
- **Whether a type is dropped is the compiler's answer, never its shape's**
  (`RustGlue`, the metadata's `glue`). A record of one integer is dropped
  when its author wrote `impl Drop` (the fixture's `Guard`), and a record
  with a raw pointer is plain data when they did not (`View`). The root
  crate has one generated line per type that crosses by value
  (`rb_glue.rs`): `__rb_drop_<n>(p: *mut T, _owns: *const [u8;
  needs_drop::<T>() as usize])`. The build reads that function's DWARF for
  which type `p` points at (a spelling that named another type is refused)
  and for the array's length (1: dropped, 0: plain). `GeneratorRust.owns` is
  only the fallback for a type with no answer (`unmovable`): such a type is
  moved only when its shape shows nothing it could own, and a function that
  would move one that may is a stub with its reason in `UNSUPPORTED`.
  Nothing is left out silently.
- **One row of `rb_glue.rs` is one line, and no two rows have the same
  code.** A compile error names its line (cargo runs with
  `--message-format=json`; `RustGlue.error_lines`), the row is dropped and
  the rest built again (`_plan_glue!`). Each drop function returns its own
  number for a null pointer: functions with identical code are merged and
  the merged one loses the DIE this is read from (measured before the
  constant: 3 of 14 had none). Keep a new per-type function either unique
  or unread.
- **A type is spelled by a path that resolves, found by a probe compile.**
  DWARF names a type where it is DEFINED (`rb_fix::inner::Token`); source
  can name it only where it is public (`rb_fix::Token`). Every candidate
  (`RustGlue.candidates`: the path, then with less of its middle) is one
  `use` line; rustc reports all the lines that do not resolve in one pass.
  The trailing `alloc::alloc::Global` is cut (`source_name`): it cannot be
  named in stable Rust. A type with no path has no line and says why
  (`glue_refused`). Never write a type into Rust source from its DWARF name
  directly.
- **A `String`'s and a `Vec`'s layout is never spelled in Julia.** The root
  crate reads and builds them with the standard library's own functions
  (`__rb_string_view`, `__rb_string_new`, per `Vec` type `__rb_vec_view_<n>`
  and `__rb_vec_new_<n>`). A `Vec`'s buffer is a pointer to bytes in DWARF;
  its element type is the type's first template parameter (`params`,
  `RustGlue.vec_element`).
- **A value has one holder** (`GeneratorRust.decode`'s `take`). Taken (a
  return): a `String` or a readable `Vec` is copied and Rust's dropped there;
  anything else that is dropped becomes the Julia object that holds its
  bytes, with `RustRuntime.drop!` as its finalizer. Not taken (a field, an
  element): copied when it can be, and NOT READ when it owns something that
  cannot be, since a second holder would drop it again. Written by value
  (`encode`), an owning object is moved: `_rb_take!` empties it, and its
  address or a second move is then an error. A blob copy of a type the
  compiler did not call plain is never made for the same reason.
- **A fieldless enum is stored as the integer its DWARF names**
  (`_enum_base`, `underlying`): `core::cmp::Ordering` is a signed byte and
  `Less` is -1, and `@enum` checks its values when the module loads, so an
  unsigned enum of it made every module reaching an `Ord` impl fail to
  LOAD. A wrapper that parses has not been shown to load: `test_rust_wrap.jl`
  evaluates the types section, and a survey of real crates must load each
  module (`.claude/rust-probe-2026-11-19/survey_wrap.jl` in the working
  checkout, hand-run, network).
- **`Option` and `Result` are read by the discriminant DWARF gives**, niche
  values included (`_option`'s `:niche`, `_result`): a value that may not
  fit `Int64` arrives as a decimal string, and `_discr_literal` writes it as
  a literal of the discriminant's width. A returned `Result` is its `Ok`
  value or a thrown `RustError` of its `Err`; anywhere else it is a value
  like any enum.
- **`rb_glue.rs` belongs to one compiler, crate and feature set**
  (`rb_glue.json`, `_glue_identity`, the file's hash): a file that does not
  match its record is replaced by an empty one before any compile, and a
  build that fails with lines in the file (of either kind: a type's or a
  function's) is tried once more without them, so a crate that changed
  under its lines still builds.
- **A wrapper JITs nothing at load.** `RustRuntime.Library` opens the
  library and, when there is one, its thunks library; a thunk session is
  created only at the first call of a thunk that library does not have, and
  never while Julia is generating output. A wrapper of plain calls needs no
  libJLCS, and neither does one with its thunks library (devtests
  `test_rust_wrap_live.jl` asserts libJLCS is never loaded in that child).
- **The doctor's Rust probes are optional, and a Rust project is asked only
  about them** (`EnvironmentDoctor.rust_ready`: cargo, rustc, a `c++filt`
  that reads one of rustc's v0 names, the DWARF and symbol readers). `build`
  decides which question to ask from the project file
  (`ConfigurationManager.is_rust_toml`) before it loads the configuration;
  a status with no probes (the registry's cache) is not ready, so the
  registry skips its own check for a Rust project and `build`'s runs.
  (`test_doctor_rust.jl`.)
- **`discover` asks cargo and scans nothing** (`RustBuild.discover_project`):
  a folder with a `Cargo.toml` and NO replibuild.toml is a crate, and the
  file written is its name and `[rust]`. An existing `[rust]` project is
  kept byte for byte, with `force` too (the scan is for C and C++ sources).
  cargo answers for the whole workspace from any member's manifest, so
  `discovered_crate` takes the manifest's own package first, lets only the
  workspace ROOT stand for its members, and refuses several library crates
  with the file to write. (`test_rust_discover.jl`.)
- **The registry hashes a Rust project's own inputs** (`hash_config`'s rust
  branch: `RustBuild.input_files`, `source_identity`, `toolchain_identity`):
  `[compile] source_files` is C and C++ and would miss every `.rs`.
  `PackageRegistry` is included after the Rust modules for this, so
  `RustBuild` takes the generator fingerprint from its caller.
- **The dialect's ABI version is 3 (2 with `native_call`, 3 with its address
  form), and the published libJLCS is 3.0.0 at that number** (release
  `jlcs-v3.0.0`, the one `Artifacts.toml` names; its two tarballs' hashes
  are the file's). The next change to the dialect's ABI needs a new release
  before its branch's CI can be green (`JLCSLibrary` above).
- **`RustDWARF` builds a unit's DIE tree only when the unit names a wanted
  symbol**, and resolves after the tree is built, so DIE order cannot matter.
  Types are keyed by SCOPED name (`core::option::Option<i32>::Some`); two
  layouts under one key are marked `conflict` and have no shape. A constant
  that may not fit `Int64` (a niche discriminant) is carried as a decimal
  string; `_number` answers `nothing` for a location expression.
- `rust_metadata.key` gates re-extraction (library, IR, toolchain, generator
  fingerprint) and is removed before one. `_generator_fingerprint` does not
  see the contents of UNTRACKED files: while a new `src/` file is untracked,
  delete the key to re-extract.
- `src/` parses JSON in two places only: `_read_json` (a file) and
  `_parse_json` (text a tool printed). `test_json_mmap_hygiene.jl` holds both.

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

- **Every `test_*.jl` is listed in `test/suites.jl`, in exactly one suite**
  (`:unit`, toolchain-free; `:dev`, compiles or JITs), or in its `EXPERIMENTAL`
  list. `runtests.jl` enforces it, and that no suite file `include`s a test
  file itself. A new file is wired by adding one line there: its `areas`
  (which parts of `src/` it exercises, so `--changed` can pick it), `locks`
  (every fixture directory it builds in or reads from: under `--processes`
  two jobs sharing a lock never run at once) and `after` (the
  `fixture:<name>` jobs whose wrapper it reads; a file that runs after a
  fixture must lock it too). A dev file must stand alone in a fresh process
  (`using Test`, `using RepliBuild`, `REPLIBUILD_HOME` set if unset, the
  fixture it reads named in `after`) and assert deltas, since a selection
  runs only some of its neighbours.
  `areas_of_path` is the change map; a source file no rule knows selects
  everything, so a new `src/` directory needs a rule there.
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
- A file aborts at its first failing top-level testset, so one red hides an
  unknown number of unrun tests in that file. Put cheap probes early. The
  other files still run: run.jl nests each job in one testset, so a failing
  file no longer ends the suite.
- Assert **deltas** on process-global state (JIT engines, fixture counters), not
  absolutes, because files share one process. Assert **names**, not counts, downstream
  of static promotion.
- A background command's exit code is the wrapper's. Write `echo "EXIT=$?" >> log` and
  read that. Under `--processes` the runner's own logs are `test/.logs/<job>.log`
  and `test/.logs/summary.tsv`.
- A test that states one host's answer instead of the invariant goes red on the other
  hosts (Linux x86-64, Linux AArch64, Windows). Drive host-conditional code with
  every spelling, through the `abi` argument, never `Sys.iswindows()`.
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
  discover-driven projects. The first seven are the dev suite's
  `fixture:<name>` jobs (`test/support/fixture.jl`): `discover(force=true)`
  regenerates the toml, the fixture's `curated` config in `suites.jl` is
  re-seeded after, then build, wrap and its `verify.jl` in a child
  (`test_aot_thunks.jl` sets its own `aot_thunks = true` the same way).
  `abi_nested_test/` and `c_sret_test/` are projects whose test writes the toml
  and runs build + wrap.
  `struct_abi/`, `jlcs_producers/`, `mlir_templates/` and `jit_unwind/` are
  single-file clang++ fixtures their test compiles directly; `fixtures/` holds
  the vendored JSON corpora and `fixtures/cmake_graph/` the vendored File API
  replies. `graph_build_test/` (a graph described by hand, no cmake),
  `graph_generators_test/` (configured with cmake and Ninja) and
  `graph_mode_test/` (a CMake project; its test writes a `[graph]` TOML into a
  copy, since `test/*/*.toml` is untracked) are graph mode's live fixtures.

- Rust: `test/rust_test/crate` is the live fixture crate (one directory
  deeper than the others: `test/*/*.toml` is untracked and would take its
  `Cargo.toml`) and `test/rust_test/nostd` a second one (`#![no_std]`,
  `#![deny(warnings)]`); `verify.jl` (plain `ccall`) and `verify_thunks.jl`
  (thunks) are the child-process probes. A real crate's check clones from
  the network, so it is a Hub package's file, never a repository test.
  `test/fixtures/rust/lean_o{0,3}.*` are vendored from `lean.rs` by
  `test/tools/gen_rust_fixtures.jl` (IR lines as `.ir.txt`: `*.ll` is
  ignored). `owned_o3.metadata.json` is the ownership half of the fixture
  crate's own metadata (`test/tools/gen_rust_owned_fixture.jl`, which needs
  cargo): regenerate it when `METADATA_VERSION` or a row's meaning moves.
  The thunk probes run twice, through the JIT and through the thunks library
  `wrap` compiles; a probe script says which it used (`PROBE thunks_from`),
  so a pass through one cannot stand for the other.

- **A test that pins emitted text pins the part it is about** (the thunk and
  the return type, the type of one position), not a whole call. Six tests went
  red at one integration because another change respelled an argument
  (2026-11-18). A clean textual merge says nothing: run the merged suites
  before calling a set of branches done.
- `.debug/mlir/` files are named by content and never rewritten: a test that
  reads the module empties the directory first, never takes the newest by
  mtime.

## Windows (`x86_64-w64-windows-gnu`, MSYS2 CLANG64) — since v4.0.0

Linux (x86-64 and AArch64) and Windows (x86-64) are supported; macOS is refused
at load (Mach-O plumbing and Apple's arm64 PCS, see Not Yet Built).
Setup and full findings: `docs/updates/WINDOWS_PORT.md`. The rules that bite code
changes:

- `check_environment` judges the MSYS2 environment by where `clang++` lives, never
  by `MSYSTEM`: Julia usually runs outside any MSYS2 shell, and Git Bash sets
  `MSYSTEM=MINGW64`. A `libJLCS.dll` that will not load is a missing
  `clang64\bin` PATH entry as often as a gone LLVM (`_clang64_bin_off_path`).
  JIT C++ wrappers run `llvm-dwarfdump` at load (`DWARFParser.parse_vtables`),
  resolved by `Tools` (below, under Subprocesses); the doctor reports that copy.
- DWARF dumper: GNU `objdump` on Windows. `_is_gnu_binutils` must reject llvm-objdump
  (on PATH in CLANG64), which otherwise parses **zero functions** silently. Use
  `Compiler._gnu_objdump()` wherever GNU objdump specifically is needed. The
  identity is `GNU Binutils` or a line opening `GNU <program>` for any binutils
  program: Fedora prints `GNU nm version 2.46.1-1.fc44`, with no `Binutils`
  (issue #68). The candidate list is `_dumper_candidates`, and the doctor
  reports each candidate's first line when none is accepted.
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
- **A library handle resolves only that DLL's own exports.** On Linux a name looked
  up through libjulia's path reaches libc through its dependencies; on Windows
  `could not load symbol`. A test whose wrapper really calls a libc function names
  the C runtime there (`Libdl.dlpath("msvcrt")`, `test_open_enums.jl`).
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

## Performance

Where the time goes was measured on Jolt, the largest Hub package
(2026-11-17), and the method is the rule:

- **Measure on a scratch copy, old code and new in the same directory.** A
  `git archive HEAD` checkout beside the worktree runs the old code. Output
  paths are baked into the metadata, the wrapper and the forced unit's skip
  reasons, so two directories never compare equal. Delete
  `.replibuild_cache/project_hash` between runs; the IR cache stays warm.
  The proof of an optimization is the same `LIBRARY_SHA`, the same metadata
  apart from `timestamp`, and the same wrapper and manifests apart from
  `# Generated:`. An AOT thunks library also embeds the path of libJLCS's
  directory in its RUNPATH, so its hash follows the checkout.
- **Julia's profiler does not sample a task waiting on a process**, and its
  counts are thread snapshots, not seconds. Stage wall time comes from
  timestamps on the build's own log lines.
- **A helper asked once per type or per symbol must not scan every function
  or symbol.** Index once and ask the index (`_strong_member_index`,
  `children_by_parent`); `_forced_exportable_ir` asks the symbol set about
  each definition's name. Each of these was quadratic at Jolt's size
  (13,882 functions, 1,306 types, 12,000 forced symbols) and invisible on a
  fixture.
- A compile of the forced unit parses the library's whole include block:
  count compiles. Rows may be tried together (`_alone_verdicts`); headers
  may not, since one that fails alone can compile after another.
- `parse_dwarf_dump` is linear, about 3 µs a line (llama.cpp: 7M lines,
  21 s). Its cost at oneDNN's size is memory (Not Yet Built), not the scans.
- The first `build` in a process spends about 25 s compiling RepliBuild's
  own code and the first `wrap` about as long, whatever the library's size.
  Nothing is precompiled. `precompile(Compiler.compile_project,
  (RepliBuildConfig,))` and `precompile(build, (String,))` recover the
  build's share, at that cost per precompile of the package; a directive on
  `wrap` recovers nothing (the generators are reached by dynamic dispatch),
  so wrap would need a workload that runs one. Not done: John's call.
- **A function that only sometimes calls `build`, `wrap` or `register` must
  not make them part of its own compile** (2026-11-19). `discover` ends in
  `_build_and_wrap`, whose calls were known at compile time, so the first
  `discover` of a session compiled all three before it ran: 87 s to keep a
  file or to write one. They go through `Base.invokelatest` there (6 s, the
  scan's own code). Measure the first call of an entry point with a runtime
  argument (`f(dir, force) = discover(dir; force)`): a literal keyword lets
  the compiler prune the branch and hides the cost.
  (`test_rust_discover.jl`.)

## Open problems

- **C ccall coverage** is measured by `test/tools/c_ccall_coverage.jl` (reads the
  Hub's wrappers and `.so`s as they are; rebuilds nothing; pass the Hub's path
  from a worktree). 2026-11-22, 4.6.0 wrappers after the release sweep, 30 C
  packages: 99.97% net of recipe exclusions, 99.99% with the two
  sibling-reached names counted (zlib's `gzgetc` and mpack's
  `mpack_expect_cstr_match`, callable under their names through a sibling
  symbol), the one miss left being micropython's `mp_decimal_exp`, a trap;
  the same as the 4.5.0 wrappers before it, and 98.6% over 17 packages in
  August. The varargs misses went
  with `[wrap.varargs]` (`@ccall` is a call) and mpack's 31 `mpack_tag_t` stubs
  with the register-class rule (the record's eightbyte classes, not the
  presence of a float). Count `ccall` targets in parsed code, not emitted
  names: a trap stub is not coverage, and a name a recipe's `exclude_symbols`
  drops (glfw's `_glfw*`) is excluded, not missed. Not an open problem any
  more; kept for the method.
- **C++ coverage** is measured by `test/tools/cpp_thunk_coverage.jl`, the C
  tool's twin (reads the Hub's wrappers, `compilation_metadata.json`,
  `thunk_manifest.json` and the graph manifests as they are; rebuilds
  nothing; pass the Hub's path). The surface is one function per mangled
  symbol net of `exclude_symbols`; bound means a `ccall` target or an invoked
  thunk in the PARSED code (`_mlir_ciface_<m>_thunk` or `…_thunk_inplace`, an
  AOT slot through `_THUNK_SLOTS`, an AOT virtual `thunk_<m>`), never the
  manifest alone;
  every miss is a trap split by the generator's reason text or a drop split
  by what the generator did, or set aside as expected when several C++ entry
  points stand behind one Julia method by design (destructor variants,
  `const`/non-`const` pairs, an rvalue overload beside its twin, two C
  spellings of one integer width), measured over the callable surface.
  2026-11-22, 4.6.0 wrappers after the release sweep, the Hub's indexed C++
  libraries (oneDNN's row is still its 4.5.0 wrapper): 99.64% of 13,210
  callable, the forced unit's functions and the pure virtual rows included,
  40 traps, 7 drops, 725 expected. The traps: 19 pure virtuals with a record
  by value, 16 `long double` by value (13 of them in boostmath's detail
  classes), 4 records with no layout (`std::function`, Vulkan-Hpp's
  `ArrayProxy`), and fmt's `format_error` constructor. The drops: four
  functions in clipper2 and fmt whose overload set collapses to one `::Any`
  signature, imgui's two undeclared varargs, and pugixml's
  `xml_text::data()`, whose Julia name another function has. Jolt is not in
  `index.toml`; `--all` counts it (97.5%, 122 traps, 211 drops). **The tool
  must know every spelling of a thunk's name**: until 2026-11-22 it matched
  only `<m>_thunk`, so every function whose return is built in the caller's
  buffer (`<m>_thunk_inplace`) was counted as dropped, 59 on this
  measurement, under a cause ("dispatch collision") that sent the reader to
  the generator's dedup. Its "thunks in a manifest nothing invokes" line is
  the check: it is 0 when the tool and the generator agree.
- **C++ standard-library types** are measured by
  `test/tools/cpp_type_matrix.jl` (2026-11-23; hand-run, reads no Hub, about
  ten minutes). `test/tools/cpp_type_matrix/<family>/` is one small library
  per type family (`lib.cpp`), the `[types] templates` it needs
  (`family.toml`) and a probe (`probe.jl`) that calls the wrapper from Julia
  and answers one verdict per cell: make, pass by value, pass by reference,
  return by value, return by reference. The tool builds and wraps each with
  the sweep tool, runs the probes in children, and writes `<out>/matrix.md`;
  the last measurement is `docs/updates/CPP_TYPE_MATRIX.md`, and what it
  found is listed under "Confirmed defects — 2026-11-23" below.
  - **Bound is not usable.** The coverage tool reads 96% of these libraries'
    functions as bound. A function is bound when its thunk is invoked
    somewhere; whether a user has anything to pass it, or can read what it
    returns, is the matrix's question. Before saying a standard type works,
    read its row.
  - **A probe uses what the manual gives a user** (`docs/src/calling.md`):
    the module's functions and types, `create_std_*()`, `Ref(x)`. A `NO WAY`
    is its author's reading, so it shows what was looked for (`nomake`
    checks that the wrapper has no `create_…` function first).
  - **A check the library makes on a value Julia holds is made first on one
    that never left C++** (`intact(control, got, want)`): a check that fails
    on its own control proves nothing.
  - A probe that dies is run again from the case after it and the case is
    CRASH, so a case never relies on an earlier one. A new type is a
    function in a family's `lib.cpp` and a case in its `probe.jl`; a new
    family is a directory.

  (`test_cpp_type_matrix_tool.jl`.)
- **Over-aligned records, what the C sret and byval fixes do not cover**
  (2026-10-18, narrowed 2026-10-25 and 2026-11-10). GeneratorCpp's own ccall
  path has no aligned buffer, and every alignment DWARF records (the record's,
  a member's, a nested record's) keeps a function off it. An
  alignment only a typedef carries (`typedef struct S T
  __attribute__((aligned(16)))`) is not in clang's DWARF, so a return buffer
  for one is placed by the stand-in; a parameter's slot is read off the IR's
  `byval` and is right. GeneratorC's parameters, its `[wrap.varargs]` path and
  Tier 2 are covered (the invariants above; `test_c_sret_abi.jl`,
  `test_varargs_emission.jl`, `test_sysv_live.jl`).
- **Distinct types sharing a bare name keep one definition between them**
  (2026-10-23). It is marked `conflict` in the metadata, and the build names them
  (llama.cpp: pimpl `impl`s, `input_decision`, `llm_graph_input_kpool`). A by-value
  crossing of one is an ABI trap, but every other reference by that name gets the
  kept layout: a `Ptr{T}`/`Ref{T}` parameter, a method's receiver, a global, a
  field of another record. No wrapped Hub function reached one by value or those
  ways when measured. Disambiguating needs scoped names through `resolve_type`, the
  generators and the thunks, and C's case (two `.c` files, two `struct state`s) has
  no scope to qualify by.
- **Registered 3.2–4.1.x still allow LLVM.jl 10 in General** (`["3 - 4.1"] LLVM =
  "9 - 10"` in `R/RepliBuild/Compat.toml`), although none of them handles it:
  their static promotion and Tier 1 code makes the reads 4.2.1 fixed. 3.0–3.1
  predate that code and call only what main still calls on LLVM.jl 10, so they
  keep the range. 4.2.0 is capped at `"9"`.
  4.2.1 was never registered; its fixes ship in 4.3.0. 4.3.0 and later are
  registered, so the resolver prefers them, and the old versions are reached only when another
  package holds RepliBuild back. The fix is a General PR (outward-facing, John
  opens it): split the entry into `["3 - 3.1"]` at `"9 - 10"` and `["3.2 - 4.1"]`
  at `"9"`, which changes exactly the nine versions 3.2.0 to 4.1.1.

### Confirmed defects — 2026-09-26 audit

Re-reproduced from clean builds of library-free fixtures (runner and fixtures in the
gitignored `.claude/audit-2026-09-26/`, `run_all.sh`; each line prints CONFIRMED,
GUARDED or NOT REPRODUCED). Fixed, with the story in CHANGELOG: F1–F3 and F6–F28.
A fix that lives in metadata reaches a Hub package only on a **rebuild** (delete
`.replibuild_cache/project_hash`), not a re-wrap. Every fixed finding has an
in-repo regression test, negative-checked by neutering its fix; the harness is no
longer the only proof. Still open:

- **A by-value type only declared in this binary** (F4, constructor homing) is an
  ABI trap instead of a SIGSEGV, and stays uncallable until the package builds
  with `-fstandalone-debug` (or `-Xclang -fno-use-ctor-homing`). With the
  definition in the DWARF the build also defines the class's inline
  constructors since 2026-11-15 (the DWARF-extraction invariant), so the type
  can then be built from Julia as well as passed. A plain `-g` build cannot
  see such a class's constructors at all (its DIE is a declaration), so the
  pass closes F4 only where one of the flags makes the definition appear;
  neither flag is a default (see Not Yet Built).

Found later, and fixed (2026-11-13): **every pass-by-reference record return in
Tier 2 was moved bytewise out of a dead frame.** The thunk passed its own alloca
as the hidden return pointer and returned the bytes. It was masked because every
by-reference record wrapped until then was trivially relocatable in practice
(its bytes meant the same at any address), and no test returned an object that
points into itself. libstdc++'s short `std::string` is one: its data pointer
came back aimed at the thunk's frame, and its destructor then freed a stack
address. Fixed by the caller's-buffer convention (JLCS invariants);
`test_sret_in_caller_buffer.jl` fails without it.

### Confirmed defects — 2026-11-17 audit

Found by calling 22 library-free probe libraries through the whole pipeline
from Julia. Fixed 2026-11-18, each with a regression test named in the
invariants above: the load-time segfault for a by-value class with a
destructor, enum pointers and references passed by value, Tier 2 pointer
positions that dereferenced anything, base-object constructors bound for
virtual-base classes, static calls for virtual methods, identifiers that broke
the module or added methods to Base, `unsigned __int128`, the `char8/16/32_t`
traps, `extern "C"` functions that throw, alias symbols; then signed
bitfields, by-value `char`, `signed char*` returns, C strings on the `ccall`
paths, C++ globals outside the global namespace, pure virtual methods,
operator names, a record's pointer to a factory class, factories that share
a constructor signature, and a global or no-argument function named like its
struct. Still open:

- **Same-named enums in two scopes collapse** (`Mode`, `ns::Mode`): one's
  enumerators are bound nowhere and its functions return the other type. The
  bare-name problem above, with constants lost.
- **A value moved into a by-value parameter is stale afterwards.** The thunk
  moves as C++ does, but the caller's Julia value is an immutable bits copy,
  so using it again is a use-after-free. What owns a move-only object on the
  Julia side is John's call.
- **Two library names that collide after a move are refused, not resolved**:
  a library that defines both a struct-named global `conf` and its own
  `conf_`, or a global `conf` beside a function `conf_ptr(void)`. The
  duplicate-method guard refuses the wrap; `_function_alternative` checks
  Julia's names only.
- **Factory constructors take the strict C type**: `Animal(4)` is a
  `MethodError` (`Animal(legs::Cint)`) while `make_animal(4)` takes any
  `Integer`. Relaxing it changes every factory class's text and makes
  collisions more frequent; `_factory_ctor_plan` must then key on the relaxed
  spelling.
- **A C++ wrapper types a `const unsigned char` or `const signed char`
  parameter strictly** (found 2026-11-21): `a::UInt8`, `b::Int8`, where the
  unqualified spelling is `::Integer`. The list of relaxed C spellings is
  matched before the qualifier is cut. A `MethodError` for an `Int`, never a
  wrong call.
- **A class template instantiated on a pointer type has two spellings**
  (found 2026-11-23): its functions' `class` is the demangler's
  (`Holder<int*>`), its layout is keyed by DWARF's (`Holder<int *>`), and
  `_class_table_key` does not meet them. So `_collect_class_raii` files the
  class's destructor and constructors under a key the thunk generator never
  asks, and a by-value parameter of such a class crosses as a copy of its
  bytes: no copy constructor makes the temporary and no destructor ends it.
  The call itself is right (`test_template_arg_records_live.jl`'s
  `class_with_destructor_by_value` takes either answer). Its Julia names
  differ the same way (`Holder_intmul_Holder` beside `Holder_int_star`).
- **The forced unit exports constructors of records a recipe keeps out of its
  surface** (found 2026-11-21, the release sweep): boostmath hides everything
  but its shim, and its dynamic table now has 19 constructors of
  `boost::math::detail::` classes, so the package's own export-table check
  fails. The engine does what the inline-constructor pass says. Whether
  `surface_types_only` should keep such records out of the forced unit, or
  the recipe should exclude them, is John's call.

### Confirmed defects — 2026-11-23 type matrix

Found by `test/tools/cpp_type_matrix.jl` on 22 standard-library probe
libraries (libstdc++, Linux x86-64, thunks compiled at load); every case is in
`docs/updates/CPP_TYPE_MATRIX.md`. Fixed the same day, with the rules in the
invariants above (`test_template_arg_records.jl`): `std::vector<bool>`
returned by value was sized as a `std::vector<T>` and overran its buffer, and
`std::unique_ptr<T[]>` came back in a register where C++ uses a hidden
pointer. Fixed 2026-11-24 (`test_placeholder_references.jl`): a reference to
a smart pointer refused `Ref` of the bytes one is returned as, and a
`std::function*` return was untyped. The matrix was measured again that day
and no case kills the process. Still open:

- **A by-value return is built in the caller's buffer and then moved.**
  `slot_value` copies the bytes into a Julia value, and `Ref(r)` or a
  by-value pass (`pointer_arg(ByValue{V}, x)`) copies them again. A type that
  points into itself is not valid at the new address: `std::list`,
  `std::map`, `std::set` and `std::unordered_map` (a marker inside the
  object), and a short `std::string` wherever `take_string!` does not read it
  in place (inside a `pair`, an `optional`, a `filesystem::path`;
  `std::wstring`, `std::u16string`). Lookups happen to work, `rbegin()`
  answers wrong, and a walk to `end()` does not stop. The same values are
  never ended: nothing runs a destructor on the bytes, and a by-value pass
  lends where C++ moves. What owns such an object on the Julia side is John's
  call, the same one as for a move-only parameter above.
- **Nothing makes a `std::function`, a `std::variant` or a stream.** A
  reference to one is typed `Ref{<empty struct>}` and takes a typed pointer
  the library returned: a `std::function` it owns passes by value and by
  reference. For `std::variant` and `std::istream` no function returns one,
  which the coverage tool counts as bound, and a `std::ostringstream*` at a
  `std::ostream&` needs the pointer converted by hand. (A reference to a
  smart pointer takes `Ref` of its bytes, and a `std::function*` return is
  typed, since 2026-11-24: the Generators invariants.)
- **`create_std_basic_string_char()` is zeroed memory** (its method table has
  no constructor), so `String(s)` on a new one is an error; and
  **`create_std_map_std_string_int()` is a `CppMap{Ptr{Cvoid}, Cint}`**, so a
  `String` key is a `MethodError`.
- **The refusal for a toolchain record with no layout advises
  `-fstandalone-debug`** when the library is already built with it
  (`std::function` returned by value, `std::variant`, a fixed-extent
  `std::span`).

## Not Yet Built (roadmap, not bugs)

Absence is the default state on this project. A missing feature is an unbuilt piece;
only something contradicting what **is** built is a bug. Log new gaps here, dated, and
don't file them as defects. Never extrapolate a narrow entry into "X is unbuilt" —
read the generator output first. (User-facing C++ RAII — `Managed<T>` finalizers with
DWARF-resolved destructors — **is** built.)

- **Rust, what is left** (2026-11-19; `docs/updates/RUST_FRONT_END.md`
  §21 to §24). A generic function is wrapped only for the types
  `[rust] instantiate` names. Of trait objects and callbacks: a callback
  the library keeps and calls later, a callback that takes a string, a
  slice or a record by value, two Rust function pointers of one kind in one
  call (the library has one function of that signature to be given), a Julia type implementing a Rust trait, a
  function that returns `&dyn Trait`, a `dyn` of two traits. Of ownership:
  building an owning record or an enum with data from Julia (only `String`
  and `Vec` are built), `Box::new` from a Julia value, a type nothing with
  a public path returns, a slice of what is not plain data, a value aligned
  beyond 16 bytes (tiny-skia's SIMD records). Also unbuilt: an `extern "C"`
  function taking a record the C way, a crate with several codegen units,
  `show` for a Rust record (it prints as its bytes), and an object file
  written by libJLCS itself, so that compiling thunks ahead of time needs no
  system `llc` (a machine with only the prebuilt libJLCS gets the JIT
  wrapper; the same holds for C++ AOT thunks). Not checked, and not checkable from Julia: that a value
  borrowing from another does not outlive it, and that a value is dropped
  on the thread that made it (a finalizer runs where the collector does).
  Linux x86-64 is measured on the dev box. Linux AArch64 runs only in CI
  (tier2-linux's arm cell: the three live tests pass there, compiled and
  JIT'd, since 2026-11-20), so an AArch64 failure is first seen on a pushed
  branch. Windows is not run: CI's Windows cells run the pkg and unit suites
  only.
- **Handing the calling-convention rules to LLVM** (measured 2026-11-20
  against the LLVM 23.1.5 sources and `main`). LLVM's ABI library
  (`llvm/ABI`) is what `test_sysv_abi_oracle.jl` already diffs the SysV rules
  against. At 23.1.5 it classifies x86-64 SysV and BPF only; on `main` AArch64
  too (AAPCS, Darwin, Win64 kinds); nothing for x86-64 Windows on either.
  MLIR ships the dialect's half (`mlir/ABI`: `ABITypeMapper`, and
  `ABIRewriteContext`, "each MLIR dialect that wants ABI lowering provides a
  concrete subclass"; installed with MLIR 23). ClangIR is in the tree and is
  not a route: distribution clangs are built without it
  (`-DCLANG_ENABLE_CIR=ON`), its call lowering runs on a test target at
  23.1.5, and its `MissingFeatures.h` lists ABI argument info, exceptions and
  thunks among 252 gaps. It also compiles source, where this project reads
  the library that was built. A swap would make `JLCSAbiOracle.cpp` the
  classifier and retire `SysVRecordABI.jl` and the C++ SysV rules. Three
  things block it: the dialect still builds against LLVM 21 and 22, which
  have no such library; Tier 3 classifies in Julia with no libJLCS, so C
  would gain a dependency; Win64 has no upstream rules. Until then the oracle
  is the value. Add an AArch64 oracle when the dialect can build against an
  LLVM with `createAArch64TargetInfo`: AArch64 is verified only in CI. To
  re-measure: `llvm/lib/ABI/Targets/` and the header of
  `clang/lib/CIR/Dialect/Transforms/CallConvLoweringPass.cpp`.
- Toolchain records by value that still trap (narrowed 2026-11-13; `tuple`,
  `optional` and `complex` are laid out, above): one with a virtual base, a
  struct (not a union) holding an empty record as a member, and anything
  holding a type nothing lays out. libc++'s `tuple` (`__tuple_impl` over
  `__tuple_leaf` bases) has the same shape as libstdc++'s and is not yet
  measured on a libc++ host.
- libdw as the default DWARF reader (2026-11-17). It is opt-in
  (`REPLIBUILD_DWARF=libdw`); John's call. Not holding the dump text is not
  the memory fix by itself: measured peak RSS, readelf against libdw, was
  1.68 against 1.57 GiB on Jolt and 2.41 against 2.21 GiB on llama.cpp. The
  rest is the parser's own tables (next entry), which either reader fills.
- `parse_dwarf_dump` holds the whole `readelf` dump and every DIE as a `Dict`.
  oneDNN's (`libonednn.so`) parse passed a 24 GB cap and was OOM-killed at
  2519a16 (2026-10-29), so a oneDNN build needs most of a 64 GB box. A streaming
  parse, or dropping DIEs the tables never read, would bound it.
- StructGen has its own `_outside_template_args`, a different function from the
  package-level one (it keeps the `<>`, collapses only the outer arguments and
  answers `nothing` without a `<`). Same name, two meanings; renaming one would
  stop a caller picking the wrong one.
- Default `-Xclang -fno-use-ctor-homing` for C++ builds. It restores the definition of
  header-only value types at a fraction of `-fstandalone-debug`'s size, but changes
  every C++ package's metadata, so it needs a Hub-wide measurement first.
  Narrowed 2026-11-15: once the definition is in the DWARF the build defines
  the class's inline constructors itself, so the flag's whole payoff is the
  definition, and a package that wants such a class built from Julia sets
  one of the two flags today (Bullet and box2d already do).
- `[wrap] exclude_symbols` does not reach enums or records (2026-11-16): it
  filters functions and globals. An enum or type a recipe wants out has no
  key (Jolt tried it on `EIterationStatus`: "matched nothing").
- Default arguments outside the forced unit's reach (2026-11-15): a static
  member's short call (static-ness is not known when the unit is written), a
  free function's (no class DIE declares it), an operator's, and a class
  template specialization's members.
- The forced unit (inline constructors and member functions) does not run in
  graph mode, like the templates TU. Outside it too (2026-11-15): a free
  inline function or operator, which no class DIE declares; an inline
  destructor nothing emitted (a destructor has no address to take); a
  volatile or ref-qualified member; and a second pass for the records the
  forced definitions themselves make visible.
- Array-view Julia-side accessors for the rank-1 thunks, plus rank ≥ 2 members.
- `is_struct_packed` over-classifies padding-free structs (wasted work, not wrong).
- Op verifiers for `ffe_call`, `try_call`, `load/store_array_element`, `ctor_call`,
  `yield`, `marshal_ret`. A computed verifier-coverage guard would also help.
- In-IR virtual-base upcast op (no producer needs one yet).
- vcall for struct-shaped virtual signatures (these keep static dispatch; a
  PURE virtual with one has no symbol to call instead and is a refusal).
  Carrying it needs `VirtualCallOp`'s lowering to go through
  `buildCallShape` with the `sysv_args`/`sysv_ret` statements, as `try_call`
  does: a dialect and ABI change.
- A non-pure inline virtual no translation unit emitted has no function row
  (2026-11-18): only pure virtuals get a `vtable_only` row.
- A value for a `constexpr` or in-class `static const` member with no symbol
  (2026-11-18): the metadata lists it in `global_constants`; decoding
  `DW_AT_const_value` is not built.
- Thread-local globals without an accessor (2026-11-23): a static
  `thread_local` data member (the unit would need its class's definition,
  as the forced unit includes record headers), a graph-mode library (no
  step 4c), and an ingested library (no IR to read, nothing can be added to
  it: its thread-local globals keep `cglobal`). The accessors' PE path
  (`dllexport`, the export gate) has not run on a Windows host.
- `[wrap.rename]` does not reach a global's accessor, and a scoped accessor
  name that equals another top-level binding (`cfg_depth()` beside
  `cfg::depth`) refuses the wrapper (2026-11-18).
- Julia operator methods (`Base.:<`, `getindex`, call syntax) for C++
  operators (2026-11-18): operators are named functions only.
- **macOS** (2026-11-09; the classifier was never the whole of it). Mach-O
  keeps DWARF out of the dylib (a `.dSYM` from `dsymutil`: the extractor must
  run it and point the dumper there), GNU readelf does not read Mach-O (so
  `parse_dwarf_dump` needs an llvm-dwarfdump dialect, or a GNU objdump built
  with Mach-O BFD), there is no dynsym (`_runtime_symbol_names` needs the
  exports trie, `nm -gU` / `dyld_info -exports`, and C symbols carry a leading
  underscore), and the link flags differ (no `-Bsymbolic`: two-level namespace
  binds a dylib to itself like PE; `@loader_path` for `$ORIGIN`;
  `-install_name`; `-no_uuid` for a reproducible link). Apple's arm64 PCS also
  departs from AAPCS64 (empty records dropped in C++, variadics on the stack):
  `kHostAbi` `#error`s on `__APPLE__`. Prototype on an Intel Mac runner first;
  x86-64 macOS already has its classifier.
- A record that IS one 16-byte SIMD vector, by value on SysV (2026-11-16):
  `struct { __m128 m; }` goes in one XMM register (SSE, SSEUP), which needs a
  vector-typed `regs` statement the dialect does not have (`<4 x float>`,
  `<2 x double>`, `<2 x i64>` by the lane type). A vector in a union with
  its lanes is not this case and crosses today.
- AAPCS64 homogeneous vector aggregates (a record holding a SIMD vector), and
  Windows on ARM (2026-11-09). Both refused by name.
- Codegen `.ll → .so` still shells to clang; not internalised. (The linker it
  starts is the bundled LLD for C on Linux since 2026-11-17; C++, Windows and
  the AOT thunks library still link with the system's.)
- STL factories size containers from `get_stl_container_size`'s table. GeneratorCpp's
  DWARF `byte_size` lookup never matches: STL records are not in
  `struct_definitions` (0 hits across 57 metadata files, 2026-10-02).
- Graph mode (2026-10-30) runs end to end for a CMake project configured with
  Ninja, from a directory of the project or from a `[dependencies]` checkout
  (`[graph] dependency`, 2026-11-10). Not built: other
  `[wrap]`/`[compile]`/`[types]` keys per library, a project-level fast path
  (every build configures and runs every generator step), and parallel
  metadata extraction.
- Cross-library types in graph mode, what Option 1 left out (2026-11-10;
  `docs/updates/CROSS_LIBRARY_TYPES.md` 6.9): hoisting a header-only type two
  siblings use but no common dependency does (each keeps its own, reported);
  imports between Hub packages linking each other (`link_dirs`), which need
  registry dependencies and `use()` through `Base.require`; virtual delete for
  a `Managed<K>` whose object's dynamic type is a subclass. Also: a header
  enum B's DWARF lacks is emitted from the header as `Cint` where A's DWARF
  says `unsigned int` (2026-10-31). (A class with no members of its own has an
  entry since 2026-11-14, bases included, so `graph_subclassed` sees it.)
- Graph mode for CERN ROOT (2026-10-30), measured in `docs/updates/ROOT_RECON.md`.
  Linux only: ROOT on Windows is MSVC, and its CMake refuses MSYS2 CLANG64. Every
  ROOT library's dictionary needs ROOT's own patched LLVM/Clang 20 and cling built
  first. Hand `--target rootcling` to ROOT's build (31.9 min on 4 cores); a system
  LLVM can replace only the bundled LLVM half, and only 20.1.x. Generate each
  dictionary after its dependencies' `.pcm` files exist; the `.so` inputs Ninja
  lists are ordering only. First wrap set: Core, Thread, RIO, then MathCore →
  Matrix → Hist.
- A type's Julia name carries its full template arguments
  (`_sanitize_c_type_name`, `_sanitize_cpp_type_name`) and has no rename key
  (2026-11-13; functions got `[wrap.rename]` that day). A type's name reaches
  struct definitions, every signature spelling it (`_as_declared`), the
  layout tables, `type_manifest.json` and the thunk side, through more than
  one derivation, so a `[types.rename]` is not a one-place change.
- A `std::vector` whose element is itself a `std::vector` (or any type with no
  Julia layout in the wrapper) gets a method table but no `CppVector` factory,
  with a warning (2026-11-13; clipper2's `Paths64`/`PathsD`). A factory needs
  an element type `CppVector{T}` can hold.
- **Boost is wrapped a library at a time** (2026-10-31): Hub `boostfilesystem`,
  `boostcharconv`, `boostjson`, each `boostorg/boost` with `submodules` set to
  the closure `tools/boostdep/depinst/depinst.py` computes (the guide's Boost
  section has the recipe). Not built: a `std::string` table without the
  instantiation TU (below). (A `std::string` returned by value is a `String`
  since 2026-11-13, above.)
- Wide strings (`wchar_t`, `char16_t`, `char32_t`) are not decoded in either
  direction (2026-11-13): the caller chooses the transcoding. (`std::string`
  parameters take a Julia string since 2026-11-15, above.)
- **`long double` by value is a trap on every tier** (scoped 2026-11-03, not
  built). The scalar is x87 extended precision: SysV passes it in memory and
  returns it in `ST0`, which `ccall` cannot reach, so Tier 3 can never carry
  it and a C package stays shimmed. A thunk could: LLVM has `x86_fp80`,
  `map_cpp_type` already spells a `long double` member `f80`, and
  `_table_entry` refuses only the bare scalar (`:opaque`). The design, if it
  is ever wanted: cross it through Tier 2 as a 16-byte carrier
  (`NTuple{16, UInt8}`: the thunk loads and stores `x86_fp80` from the slot,
  no ABI change beyond declaring the type) with explicit Julia conversions
  to and from `Float64`/`BigFloat`, never an automatic `fpext`/`fptrunc`
  behind a flag, which would round 64-bit mantissas to 53 silently. Route
  C-linkage functions that cross one to Tier 2 like vectors; keep records
  holding one refused on SysV returns (`sysv_trap_reason`). Live tests:
  SysV with poisoned registers, Win64, and `abi_fuzz`, whose 62 `XFAIL`s per
  60 seeds are exactly this type. Payoff measured on the Hub: 5 stubs in
  boostcharconv (`to_chars(long double)` and kin) and 3 in fmt, no consumer
  behind them, and MSVC's `long double` is a `double`, so Windows-native
  libraries rarely expose it. About three days with both hosts verified.
- STL members that live in the C++ runtime library (2026-10-16, corrected
  2026-11-13). On Linux they bind: a `[types] templates` entry instantiates
  every member in the library itself (stl_test: 205 `basic_string` members
  defined, none undefined), and a member a library calls in libstdc++ without
  instantiating it resolves through the library's OWN handle, since glibc's
  `dlsym` searches a library's dependency tree (382 of 382 undefined `std::`
  symbols across the Hub's C++ libraries, once nm's `@GLIBCXX_…` version
  suffix is cut from the name; with it on, none resolve, which is what the
  earlier entry measured). `extract_stl_method_symbols` admits such a member
  to the table (`runtime_library`) and `initialize_global_jit` pre-flights
  every callee through the handle and binds a foreign one on the engine at
  that address (`_preflight_callees`, `register_symbol`), never through
  `RTLD_DEFAULT`. What stays unbound: Windows/libc++, where a DLL handle sees
  only its own exports (`libc++.dll` exports the extern-template members, so
  a named search of it would be the shape), and a container a library uses
  but does not list in `[types] templates`, which has no table at all. AOT
  thunks need nothing: the thunks library leaves a libstdc++ callee undefined
  and binds it at `dlopen` from the global scope, where Julia's bundled
  libstdc++ comes first; ABI-compatible for `basic_string`, and a symbol newer
  than Julia's copy would miss there.
- `std::string` without the instantiation TU (2026-11-13; what the
  `std::string`-by-value decoder would want): a table built from the
  libstdc++ the library's handle resolves (its `nm -D`, filtered to
  `basic_string<char>` members the handle binds) instead of from the
  library's own symbols. Not built: it reads another library's symbol table
  at build, a step away from the compile's own DWARF as the source of truth,
  so it is John's call. A `[types]` key that merely skipped the TU would give
  a half table (only the members the library itself calls) and was refused.

## Maintaining this file

- Keep it lean. When you fix something, put the story in the pull request's
  description under a `## Changelog` heading (dated title, what shipped, what was
  found, which tests), and add a line here only if it creates a rule a future
  session must follow. **A PR never edits `CHANGELOG.md`** (John, 2026-11-14): open
  PRs conflicted on its Unreleased section, so that section is compiled from the
  merged PRs' `## Changelog` headings when a release is registered.
- Derived claims (counts, line numbers, versions, test totals) are what rot. Prefer
  naming the test that computes them over writing the number. Avoid `file:line`
  citations; name the function.
- The file is tracked: no machine-local paths.
