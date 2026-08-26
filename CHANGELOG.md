# Changelog

All notable changes to RepliBuild.jl are documented in this file.

## v3.3.3 (2026-08-26)

Patch. A Tier-2 call spent most of its time re-answering a question settled at
load; a config key that never did anything now does; a fresh Linux clone is
verified by cloning rather than by reading `.gitignore`; and the Win64 struct ABI
classifier lands with its specification test wired in.

### Tier-2 calls resolved the same symbol on every invocation (2026-08-26)

`invoke_aot(handle, name, …)` went through a `ReentrantLock` and a `Dict` that
re-hashed a ~40-character symbol name **on every call**, to find an address that
cannot change after `__init__` dlopens the thunks `.so`.

Measured on hello_world, 1M calls, best of seven:

| stage | ns/call |
|---|---|
| raw `ccall` → C++ symbol (Tier 3) | 1.1 |
| `ccall` → thunk, fptr pre-resolved | 3.5 |
| `_lookup_aot` alone (lock + Dict) | 37.3 |
| `hello_message()` | 53.6 |

The MLIR thunk costs **2.4 ns over a bare `ccall`**. Symbol resolution was ~70%
of the call.

The laziness was inherited from the JIT path, where it is correct — a thunk's
address does not exist until the JIT compiles it. An AOT thunks library is fully
linked at `dlopen`, and the generator knows every thunk name at emission, so
neither the laziness nor the cache buys anything there. `__init__` now fills one
`Ref{Ptr{Cvoid}}` slot per thunk and each call site reads it through
`invoke_aot_ptr`. A `Ref` rather than a `const Ptr` because a raw pointer cannot
be serialized into a precompile image.

| | before | after |
|---|---|---|
| `hello_message()` | 53.6 ns | **17.8 ns** |
| `hello_message_ptr()` | 37.8 ns | **6.8 ns** |
| first call | 4 allocs / 528 B | **1 alloc / 32 B** |

**That first-call delta was never the String.** The 32-byte `unsafe_string` copy
is present in every call including the second; the extra three allocations were
the symbol cache's `Dict` allocating its backing arrays. The `_ptr` variant shows
it plainly — it allocated 3 on its own first call and 0 thereafter.

`_arg_marshal_plan`, `_invoke_call` and `_check_pending_exception` are shared
verbatim; only symbol resolution moves, which `JITManager`'s own comment already
names as the legitimate divergence between the two paths. Slots are derived from
the **final post-dedup wrapper text**, so a call site cannot read a slot no
`__init__` fills, and a `_FPTR_*` no emission site registered raises at wrap time
rather than becoming an `UndefVarError` on whichever call the user makes first.
Slot names are keyed on the **mangled symbol**, for the reason
`_slice_const_name!` records: `julia_name` is not injective over it, and two
functions sharing one slot would each call whichever thunk resolved last.

A missing thunk now warns **at load, naming every one at once**, and raises at
the call pointing back at that warning — rather than raising one at a time.

Verified: llamacpp re-wrapped (2280 call sites → 2271 slots; the nine-slot gap is
exactly its nine Tier-2 `char*` returns, whose `f`/`f_ptr` pairs share a slot),
`test_deep` 33/33 against a real 274 MB model and the native C oracle; box2d
15/15, tinyxml2 11/11, pugixml 13/13, imgui 202/202. A consumer-precompile probe
confirms `Ref` identity survives the pkgimage — the one way this design could
have failed silently.

### Generated MLIR did not travel with the AOT wrapper (2026-08-26)

`ThunkBuilder` called `parse_module` without `debug_base`, so every AOT package's
`.mlir` fell through to `tempdir()` instead of `<pkg>/.debug/mlir`. The JIT path
has passed it since the debugging work landed. AOT thunks are the ones that
actually ship beside the library, so gdb needs the co-location more there, and
`/tmp` clears on reboot. `DW_AT_comp_dir` now points into the package.

### `[ingest] extra_link_libs` did nothing, and the test could not tell (2026-08-26)

The key was parsed, documented as *"additional -l libs at load time"*, serialized
back out, and **read by nothing**. `test_ingest.jl` "covered" it by asserting the
value round-tripped through the parser — which is true whether or not the feature
exists. Same class as *a wrong harvested config does not fail the build*.

Now emitted as an `__init__` prologue in both generators, spliced **ahead of every
`dlopen`** — opened afterwards they resolve nothing.

New `test_config_surface.jl` (in CI, no toolchain) refuses the whole class: every
`RepliBuildConfig` field must be consumed or named in `RESERVED_UNUSED` with a
reason, and `extra_link_libs` is asserted by **executing** what the generator
emits. Nine of 81 fields were inert: `[binary] strip_symbols`, the whole
`[discovery]` section, `[wrap] enabled`/`style`, `[workflow] stages` (reachable
only via `is_stage_enabled`, which has no callers), `[project] uuid`, and
`[paths] source`/`include`/`cache`.

Getting the guard right took four wrong versions, each instructive: bare
field-name matching is unsafe (`enabled` exists on three structs, and the only
`.style` outside `ConfigurationManager` is `tooltip.style` in a JavaScript string
literal in DAGDiff); `save_config` touches every field there is, so counting
CM-internal reads makes the whole surface look live; and export lists wrap across
lines with only the first carrying the keyword, which is how an exported
zero-caller accessor read as consumed. The guard caught its own author on day
one — `getfield(config, :ingest)` is invisible to it.

**Prefer a full soname or path over an `-l` name.** `-l` is a link-time spelling:
on glibc ≥ 2.34 `m`/`pthread`/`dl`/`rt` are merged into libc and
`/usr/lib/libm.so` is a linker script, so `-lm` links while `dlopen("libm.so")`
fails. The wrapper warns and continues, correctly — there is nothing to preload.
The documented example was updated accordingly.

### A fresh Linux clone works, and that is now checked by cloning (2026-08-26)

Audited by `git clone --local` to a tempdir — tracked files only, no build
artifacts — rather than by reading `.gitignore`. `Pkg.instantiate()` resolves,
`runtests.jl` is green with **0 failures across 25 testsets**, `src/mlir/build.sh`
produces a 21.7 MB `libJLCS.so`, `MLIRNative.test_dialect()` passes, and
`check_environment()` reports both tiers live.

Five things that clone found:

- **`build.sh` had no LLVM version gate.** It printed the version and proceeded,
  so a too-old toolchain died inside CMake or TableGen naming a missing header
  rather than the version. Debian/Ubuntu's default `llvm-config` is 14–18, making
  this the common case. Now gates on major ≥ 21 with per-distro install lines,
  plus a **`mlir-tblgen` ↔ `llvm-config` major-version match** — several LLVMs
  side by side is the normal Ubuntu arrangement and `PATH` order can pick one of
  each, where the mismatch otherwise surfaces as unresolved symbols at `dlopen`
  rather than as a build failure — plus a check that `lib/cmake/mlir` exists under
  the LLVM prefix, since MLIR is a separate package on most distros.
- **`build.sh`'s success message named a file that does not exist**
  (`src/MLIRNative.jl`; it lives at `src/IRGen/MLIRNative.jl`) — the first thing a
  contributor runs after a successful build. Its size report also read `Size: 0`,
  `du` on the symlink rather than its target.
- **`check_environment()` called `ccall` "Tier 1".** It is Tier 3. The first
  command in the README taught the tier model backwards.
- **`gen_receiver_corpus.jl` hardcoded `~/Desktop/Projects/RepliBuild-Hub`**, so
  the corpus-regeneration tool was unusable by anyone else. Now the repo's sibling
  by default, `REPLIBUILD_HUB_PATH` to override.
- **`test_symbol_hygiene` went vacuously weaker on a fresh clone — 89 asserts
  there against 97 here, both green.** Its strongest testset read the built
  `mi_test`/`vi_test` `.so` and `continue`d when absent, so the check driving the
  real predicate over real `nm` output — the one guarding the Itanium-thunk
  SIGSEGV class — only ever ran on one machine. **That is the exact failure this
  project already recorded for the Hub sweep.** Symbols are vendored now
  (`test/fixtures/thunk_symbols.json`, regenerated by `test/gen_thunk_symbols.jl`);
  the behavioural assertions run everywhere (fresh clone 100/100, dev box
  102/102 — the two extra are a live-vs-vendored staleness cross-check that
  legitimately needs the artifact), `@test swept == 2` refuses to assert into an
  empty loop, and the generator refuses to write a fixture containing no
  `_ZTh`/`_ZTv` symbols.

Still failing from a fresh clone, deliberately: `test_slicer`,
`test_static_promotion` and `test_tier1_dispatch`, because
`test/slice_test/replibuild.toml` is gitignored and nothing regenerates it. They
are the quarantined Tier-1 three, unwired from `devtests.jl`, so no suite reaches
them.

### Win64 (Microsoft x64) struct ABI classifier (2026-08-24, landed 2026-08-26)

The struct classifier was x86-64 SysV only. It now selects between two
conventions behind `enum class AbiTarget { SysV, Win64 }`, with
`classifyWin64Struct` implementing Microsoft x64 as a separate and much narrower
algorithm rather than a variation on SysV. On Linux `kHostAbi` stays `SysV`, so
this changes no behaviour on the current host, and a non-x86-64 build is an
`#error` rather than a silent misapply of x86-64 rules to AAPCS.

`-DJLCS_FORCE_ABI_WIN64` compiles the Win64 rules into a Linux `libJLCS.so` that
cannot run anything — the JIT would emit Win64 calls into a SysV process — but
can be inspected. It is the only way to exercise the Win64 *lowering*, not just
the decision table, without a Windows host. Build it to a scratch path, never
over the working `libJLCS.so`.

Four pinned divergences from SysV, three of them silent: size is the only
criterion (1/2/4/8 bytes in a register, everything else indirect, **including**
the 9–16 byte band SysV splits across two registers); aggregates never reach XMM
(`{float,float}` is `i64` here, XMM0 under SysV); coercion is `iN` of the
struct's own size, not always `i64`; and indirect arguments take **no** `byval`
attribute.

`test_win64_abi.jl` (devtests §6b, 95/95) is a **specification** test and the
distinction is load-bearing. `test_struct_abi.jl` proves the SysV path against a
real `clang++` callee because a self-JIT'd callee shares the JIT's own convention
and cannot catch a mismatch — and that trick is unavailable here, since a Win64
callee cannot be loaded on Linux. The oracle is clang lowering the same
signatures for `x86_64-w64-windows-gnu`. **It catches an encoded rule that
disagrees with clang; it does not prove the lowering runs correctly on Windows.**
Only a Windows host does that.

### `exit(0)` inside an include ends the suite with a success status (2026-08-26)

Two test files skipped unavailable prerequisites by calling `exit(0)`. Run
standalone that is correct and indistinguishable from a skip. Run as an
`include` from `devtests.jl` it terminates the **entire suite with exit code 0** —
every section after it silently never runs, and the run still looks green. Both
(`test_win64_abi.jl`, `test_multilib_jit.jl`) are skipped testsets now, verified
by hiding the prerequisite and confirming execution continues past the include.

### `devtests.jl` runs to the end again (2026-08-26)

`test_multilib_jit.jl` §"one engine per binary" had been a standing red since
2026-08-20: `length(engines) == 2` evaluating `3 == 2`. Not a product bug — the
file passed 14/14 standalone and an A/B against the pre-session commit was
byte-identical. It asserted an **absolute count of a process global**
(`GLOBAL_JIT.engines`) inside a suite that shares one process, so it was only
valid while nothing before it touched the JIT, and some earlier file started
initializing a third engine.

The subject of that testset is "one engine *per binary*", which is a statement
about the **delta** from loading two libraries. It snapshots the engine set on
entry and asserts the delta, so it is now independent of every prior file — the
same fix, for the same reason, as `test_slicer`'s coherence testset. It also
`@info`s any pre-existing engines, so the next time suite order changes the
identity of the extra binary is reported rather than mysterious.

Because `devtests.jl` aborts the file at the first failing top-level testset,
everything after §13 had been unrun in-suite — and fixing it immediately surfaced
what had been hiding.

`test_debug_inspection.jl` §"MLIR sources and thunk enumeration" asserted that
`_ZThn16_N7DiamondD0Ev_thunk` and `_ZTv0_n32_NK7Diamond3tagEv_thunk` appear in
the generated thunk list. They have been **deliberately absent since 2026-08-13**:
Itanium adjustor thunks are vtable-slot entry points, never API functions, and
`_is_itanium_thunk` removes them at the source. The test was written before that
filter and never ran again to notice — it fails standalone too, so this was a
stale expectation rather than process-state leakage.

**Inverted rather than deleted**, the same treatment cjson's Tier-1 testset got:
it now asserts no `_ZTh`/`_ZTv`/`_ZTc` thunk is emitted, so a regression that
re-admits them fails loudly. Paired with an `nm` check that `vi_test.so` still
*defines* those symbols — otherwise the guard would pass just as happily against
a fixture that had quietly stopped exercising the class.

## v3.3.2 (2026-08-23)

Patch. Scale defects found by putting v3.3.1's AOT path on llamacpp
(3686 functions, 39 MB), none reachable at tinyxml2's 283 — plus one found by
putting it on hello_world's three.

### An AOT wrapper described itself as something it was not (2026-08-23)

`_dispatch_facts` classifies each emitted function by the tier it dispatches
through, reading the FINAL chunks rather than the generator's intent. It knew
two spellings. There are three:

| | shape | emitted for |
|---|---|---|
| 1 | `JITManager.invoke(…)` | JIT, any function |
| 2 | `JITManager.invoke_aot(THUNKS_HANDLE[], …)` | **AOT, ordinary function** |
| 3 | `ccall((:thunk_<mangled>, THUNKS_LIBRARY_PATH), …)` | AOT, virtual method |

Shape 3 is `GeneratorCpp`'s AOT virtual-dispatch branch and was always
recognised. **Shape 2 arrived with v3.3.1's `invoke_aot` work and the classifier
was never taught it**, so an AOT-dispatched ordinary function matched no branch,
fell past the Tier-3 test as well, and was dropped from `DISPATCH_TIER`
**entirely** rather than filed wrongly.

Omission is the worse failure. On a library with no virtual methods the only
rows left are whatever genuinely `ccall`s — which reads as a **uniform Tier-3
wrapper**, and `_dispatch_tier_chunk` renders a uniform table as a *sentence*:

```julia
# Every function in this module dispatches through Tier 3 (`ccall` straight into the library).
```

hello_world emitted that over four Tier-2 functions. **llamacpp emitted it over
2280 `invoke_aot` call sites.** `DISPATCH_TIER` and `dispatch_tier` vanished from
both, so a consumer asking got `UndefVarError` — loud, at least — while the
wrapper's own header stated a tier it did not use.

**Order in the tier test is load-bearing** and now says so: shape 3 contains
`ccall((:`, so the Tier-2 branch must precede the Tier-3 one. The first draft of
this fix *replaced* the `THUNKS_LIBRARY_PATH` term instead of adding to it,
which would have re-filed every AOT virtual method as a plain ccall — caught by
`test_introspection`'s existing AOT fixture, which is exactly what it was for.

**Guard: `_dispatch_facts` now refuses rather than returning a short table.** A
chunk naming `JITManager.` or `llvmcall` that matches no branch is a classifier
that has fallen behind emission, and it raises naming the functions. Keyed on
the dispatch machinery, not on `ccall`, so an ordinary helper that calls libc
stays unclassified in peace. Negative-checked by removing the `invoke_aot` term:
the guard fires on `aot_plain` before any assertion runs — the class is caught
at generation time, not merely in CI.

Covered by `test_introspection.jl` (46 → 49 asserts, no toolchain): shape 2 as
its own fixture — untested until now, and untested is how it broke — plus the
refusal and a must-not-flag helper.

### Generated wrappers now explain what a thunk is (2026-08-23)

`is_ccall_safe` routes anything not marked `noexcept` to Tier 2, so a C++
wrapper is mostly thunk calls: hello_world 4 of 5, llamacpp 2280 sites. The
first thing a C++ user meets is therefore `invoke_aot` at every call site, with
nothing on hand to say what it is — `docs/src/mlir.md` is the deep treatment and
is reached only after someone has already decided to trust the tool.

The dispatch section now opens with a short note — a thunk is the `extern "C"`
shim you would hand-write, generated and compiled; why `ccall` can't express the
call; AOT vs JIT; that the marshalling is compiled in rather than decided per
call; that gdb steps into it by file and line. Emitted only when the wrapper has
Tier-2 call sites, so a pure-`ccall` wrapper never mentions thunks.

### hello_world runs on AOT thunks (2026-08-23)

`[compile] aot_thunks = true`, as the smallest possible exerciser of the AOT
path — and the one that surfaced the classifier bug, since with no virtual
methods it hits shape 2 exclusively.

| | JIT | AOT |
|---|---|---|
| wrapper load | 3.712s | **0.128s** |
| JIT engines at load | 1 | **0** |

`libproject_thunks.so` is 16.5 KB, built in 2.3s. The 29× is entirely engine
init: the cost was never the thunk count.

### CLAUDE.md is tracked (2026-08-23)

The working-state log ships with the repo now. `runtests.jl`, `test_struct_abi.jl`,
`FunctionGen.jl` and the docs already cited it, so those references had been
dangling for everyone who cloned. Machine-local paths, the home directory in a
gdb transcript, and links into a private note store were scrubbed first.

### One symbol, two vtables, two definitions (2026-08-22)

`generate_jlcs_ir` emits virtual-method declarations and `thunk_<mangled>`
definitions by walking `vtinfo.classes` and, inside that, each class's
`virtual_methods`. **One symbol can be listed in more than one class's vtable**,
so both loops could emit it twice, and MLIR rejects the whole module:
`redefinition of symbol named '_ZN20llm_graph_input_dsv49set_inputEPK12llama_ubatch'`.
No thunks library, and the AOT path unusable on any library big enough to
contain such a symbol.

Both loops had a set sitting next to them that did not guard them. The
declaration loop consulted `fthunk_decls`, which only excludes the *other* pass's
symbols and says nothing about this loop repeating itself. The definition loop
**filled** `generated_symbols` and never read it — that set is consumed by the
function-thunk filter further down, so it stopped that pass duplicating this
one's work while doing nothing about this one duplicating its own.

`needed_symbols` was hiding it. With a thunk manifest most of these symbols land
in `fthunk_decls` and get skipped, so a repeat needs two vtable listings **and**
absence from the manifest. `build_aot_thunks` passes no manifest — it runs at
build time, before `wrap` has written one — so the AOT path met the whole class
at once.

Deduping by name was checked rather than assumed: llamacpp's two colliding pairs
are byte-identical declarations, same arity and same types. A genuine signature
conflict is a different bug and is now reported instead of silently collapsed.

After: 245 declarations and 3659 definitions, zero duplicates, module parses and
lowers, `libllamacpp_thunks.so` builds at 4.3 MB in 52s.

### A wrapper guard could not survive its own scale (2026-08-22)

`_assert_no_shadowed_ccall_types` matched `function … ccall(…)` with a single
regex spanning the whole wrapper, which needs
`(?:(?!^function )[\s\S])*?` to stop at the next definition — a negative
lookahead evaluated at every character of every body. A function containing no
`ccall` makes that scan run to the next `function` and backtrack, and a Tier-2
wrapper is mostly such functions. llamacpp exhausted PCRE outright:

```
ERROR: PCRE.exec error: JIT stack limit reached
```

That refused to write the wrapper at all — with `aot_thunks` on **or** off, so it
blocked llamacpp entirely, not just the AOT path. It now splits on the `^function`
anchor first and matches within each body: linear, and bounded per function.
Refusal and pass-through behaviour verified unchanged on both the shape the guard
exists for and the shapes it must not flag.

### An unmappable by-value return is refused at the call site, not the wrapper (2026-08-22)

`format[abi:cxx11]` returns `std::basic_string` by value. `cf09702` correctly
drops STL types from `struct_definitions`, so nothing can name it and the mapper
answered `Any` — which the generator emitted straight into a foreign call:

```julia
function format_abi_cxx11(fmt::Any)::Any
    return @ccall LIBRARY_PATH.var"_Z6formatB5cxx11PKcz"(fmt::Ptr{UInt8};)::Any
end
```

**This is not a regression.** That line shipped for as long as the function has
existed; `_assert_no_any_ccall_return` is newer and refused to write the wrapper
containing it. The guard was right — `Any` in a foreign return position tells
Julia the callee returned a `jl_value_t*`, so the result is dereferenced as a
Julia object. But refusing the wrapper takes 5,614 working functions down with
the one broken declaration.

Neither escape works. `Any` corrupts on dereference; `Cvoid` leaves a MEMORY-class
aggregate's hidden sret pointer unaccounted for, so the ABI is wrong rather than
merely lossy. The value cannot be discarded any more safely than it can be
returned.

So the function is now emitted as an **FFI Safety Trap** — the pattern the C++
generator already used for unmappable *parameters*, extended to returns. The
module loads, everything else works, and this one raises with an explanation if
called.

The fix lives in `generate_vararg_wrappers`, which is shared by both generators.
That placement is the actual finding: varargs functions `continue` past normal
wrapper generation entirely, so a fix applied at the ordinary return-type
decision never runs for them — and `_Z6format…PKc**z**` is varargs.

### A vendored thunks library loaded the build tree's copy of its own library (2026-08-22)

`build_aot_thunks` linked the companion library with only
`-Wl,-rpath,<build dir>`, an absolute path. The rest of RepliBuild resolves
**sibling-first** — `LIBRARY_PATH` and `THUNKS_LIBRARY_PATH` both prefer a copy
beside the wrapper — precisely so a generated set can be vendored into a
consumer. The thunks library's own `NEEDED libllamacpp.so` did not.

So a vendored copy loaded its sibling `.so` through the wrapper and the **build
tree's** `.so` through the dynamic loader: two copies of a 41 MB library in one
process, each with its own static state, both torn down at exit.

```
double free or corruption (!prev)
signal (6): Aborted
```

Then it **hung**, because Julia's crash handler tries to symbolize the abort —
`jl_print_native_codeloc` → `DWARFContext::create` → `operator new` — and that
deadlocks on the malloc lock the abort came from. The visible symptom was a load
that never returned; the abort scrolled past above it.

Every test had already passed at that point. `test_deep.jl` reported 33/33 and
exited 0 for the *package*, where both paths name the same file, so nothing
caught it until the artifacts were vendored into LlamaChat.

Now `-Wl,-rpath,$ORIGIN` ahead of the build directory. RUNPATH entries are
searched in order, so a sibling wins when one exists — matching the wrapper —
and the absolute path remains for a thunks library used where it was built.

**Read exit codes, not output.** Every earlier run of this piped through `tail`,
which reports the pipeline's status, not the program's. `33/33` and `275/275`
both printed cleanly from processes that then aborted on the way out.

### llamacpp on AOT thunks

With all three fixes, the whole point of the exercise is reachable:

| | JIT | AOT |
|---|---|---|
| wrapper load | 24.83s | **5.38s** |
| `test_deep.jl` | 33/33 in 35.4s | **33/33 in 1.0s** |
| JIT engines at load | 1 | **0** |

The testset collapse is the same saving counted twice — the engine was
initialising lazily on the first Tier-2 call, so the tests were paying for it
too. 2280 dispatch sites, `libllamacpp_thunks.so` at 4.3 MB.

## v3.3.1 (2026-08-22)

Patch. One release-hygiene fix that v3.3.0 needs, and one correctness fix to the
AOT thunk dispatch path.

### `RepliBuild.VERSION` did not move with `Project.toml` (2026-08-22)

**v3.3.0 as registered fails its own test suite.** `runtests.jl` has asserted
`RepliBuild.VERSION == pkgversion(RepliBuild)` since long before this release, and
the v3.3.0 bump edited `Project.toml` and left the constant at `v"3.2.0"` — so the
package reports one version to Pkg and another to anything reading
`RepliBuild.VERSION`, and the first testset goes red. Caught by that assertion,
which is what it is for. Both are `3.3.1` now.

### AOT thunk dispatch shares the JIT's marshalling and return convention (2026-08-22)

`config.compile.aot_thunks` compiles Tier-2 thunks at build time into
`lib<name>_thunks.so`, so loading a wrapper is a `dlopen` rather than building and
JITing an MLIR module in every process. It is off in every Hub package, and this
is why: the emitted dispatch was broken in two independent ways, the first
crashing early enough to hide the second.

The thunk is the same object either way — the generated MLIR for
`XMLDocument::Parse` is byte-for-byte identical between the two paths, which is
what localised this to the Julia side. Only the way its address is found should
have differed. Instead the AOT branch carried private copies of two decisions the
JIT path already made through `JITManager`, and both copies had fallen behind:

- **Marshalling.** It open-coded a `cconvert`/`unsafe_convert` pair per parameter.
  `_arg_marshal_plan` had since learned `AbstractString` and boxed `Ref`; the copy
  had not, so any string argument raised
  `unsafe_convert(Ptr{UInt8}, ::Cstring)` on a call the JIT wrapper accepted.
- **Return convention.** Scalar-vs-sret was chosen by testing the return type's
  **name** against a hardcoded list of scalar spellings. RepliBuild emits enums as
  real primitive types, so `isprimitivetype(XMLError)` is `true` while the string
  `"XMLError"` matches no entry and never could. An `i32 (void**)` thunk was then
  called as `void (i32*, void**)` — which reads `args_ptr` out of the register
  holding the return buffer. **Segfault inside the thunk, for every
  enum-returning function in every C++ package.**

Both are the same mistake: approximating, in generated text, a question the type
system answers exactly at the call site. Neither was fixable where it appeared —
these wrappers declare their parameters `::Any`, so the argument's real type is
knowable only per call site, which a `@generated` function can see and an emitter
cannot.

`JITManager.invoke_aot(handle, sym, T, args…)` is now `invoke` with exactly one
difference — it resolves the symbol with `dlsym` on the thunks handle instead of
asking the JIT engine — and the generator's AOT branch is a mirror of its JIT
branch. 89 lines of emission became a delegation, and the type-name list is gone.
`register_symbol`-style lookups are cached per (handle, symbol).

Measured on tinyxml2, same package and same verifier on both tiers: **11/11 AOT,
11/11 JIT**, load **5.0s → 1.0s**, and `GLOBAL_JIT.engines` is **0** on the AOT
path — no MLIR module is constructed at load at all.

**Removed:** the LTO/`llvmcall` variant of the AOT packed-args call site went with
the open-coded branch. It required `enable_lto` **and** `aot_thunks`, which no
package sets — and it carried both defects above.

## v3.3.0 (2026-08-22)

Minor, not patch. The theme is **C++ methods that were being called without their
`this`** — three independent defects in the two receiver gates, plus the DWARF
extraction feeding them — and the fixes change emitted signatures across every C++
package. Alongside: `char*` returns stopped escaping raw from Tier 2, all three
producer-less JLCS ops gained producers, Tier 2 accepts a `Base.Ref`, variadic
overloads accept the values callers actually write, and the wrapper writer grew
three more refusals. No exported API was removed since v3.2.0.

**Upgrade note — C++ packages need a REBUILD, not a re-wrap.** The scope a method
belongs to (`class` in `compilation_metadata.json`) is computed by
`extract_compilation_metadata` at **build** time, so re-running `wrap()` replays the
old scopes and reproduces the old signatures. Worse, the project content hash will
skip the rebuild and print `cache: project unchanged` — delete
`.replibuild_cache/project_hash` to force re-extraction while keeping the vendored
clone and the per-file IR cache. C packages are structurally unaffected: a C symbol
has no `::`, so `class` is always empty and none of this reaches them.

### The dialect builds at install time now, or says why not (2026-08-22)

`libJLCS.so` is not shipped and cannot be — it links the system MLIR and LLVM,
so a prebuilt copy is wrong on any machine whose LLVM differs. The consequence
used to land at the worst moment: `Pkg.add` succeeded, `using RepliBuild`
succeeded, and the first call that reached Tier 2 raised "JLCS dialect library
not found — build it first" naming a path inside a read-only depot.

A new `deps/build.jl` runs `src/mlir/build.sh` at install time when the machine
can, and otherwise explains what is missing. **It never throws.** A failing
`deps/build.jl` fails `Pkg.add` outright, and that would be the wrong trade:
RepliBuild without the dialect is still the whole product for C libraries, since
Tier 3 is plain `ccall`. Only Tier 2 — C++ member functions and by-value
aggregates — needs MLIR. A machine with no system MLIR should get a working
install and a clear account of what it does not have.

It also checks the existing library by **dlopening it**, not by `isfile`. That
is the case a rolling distro produces: `libJLCS.so` links
`libMLIR.so.<version>`, so an LLVM upgrade leaves a file that is present,
plausible, and unloadable. Asking the loader is what distinguishes "built" from
"built against an LLVM that is gone", and it triggers a rebuild instead of a
mystery at the first Tier-2 call.

`MLIRNative.check_library`'s error is still there and unchanged. It is a
backstop now rather than the only line of defence.

### The JIT engine explained its failures to nobody (2026-08-22)

Three defects in `JLCSCAPIWrappers.cpp` and its Julia bindings, found by reading
the JIT path rather than by anything failing. Two of them had the same shape: a
parameter accepted and then dropped, so an API promised something it had never
done. No behaviour changes on a healthy load — `initialize_global_jit` still
takes the same time and produces the same engine.

**MLIR's diagnostics were going to stderr and being discarded.** No
`ScopedDiagnosticHandler` was registered anywhere, so every failure — a parse
error, a failed lowering, a refused `ExecutionEngine::create` — reported itself
to the default handler, which prints and returns. Julia then raised `Failed to
parse MLIR module` with the explanation on a stream that, during a wrapper's
`__init__`, frequently is not attached to anything. The worst case was
`moduleTypesAreLLVMCompatible`: it calls `emitError` naming the offending op and
type, which is the entire reason it emits one, and that went to stderr too — a
refusal explained by nothing.

Diagnostics are now captured into a thread-local buffer and appended to the
Julia error. `MLIRNative.take_diagnostics()` reads and clears it. The handler
returns `success()` so MLIR does not also print: one report, delivered to the
caller that can act on it, rather than two with one of them into the void. It is
scoped per fallible call, not registered once at context creation, so a failure
cannot inherit diagnostics from unrelated work.

The locations are the point, not a bonus. A parse is named after the
content-keyed `.mlir` that `jit_source_path` already writes for gdb, so the
`loc("…":LINE:COL)` in the error text names the same file and line the debugger
opens when you break in the resulting thunk:

```
error: loc("/…/jlcs_7697cb4881e7.mlir":418:7): 'func.return' op has 0 operands,
       but enclosing function (@f) returns 1
```

**`attachHostDataLayout` could null-deref, and warn-and-continue was worse.**
`createTargetMachine` may return null and was dereferenced one line later — an
uncatchable SIGSEGV taking the host process with it, which is the exact failure
mode `moduleTypesAreLLVMCompatible` was written to prevent thirty lines below.
The `target` lookup above it was already guarded; this was not.

The bigger defect was the existing failure path. That attribute tells the LLVM
conversion passes how wide a pointer is and where struct fields land; without it
they fall back to a default layout and carry on. A wrapper whose entire job is
matching the host ABI would therefore lower, JIT, run, and return quietly wrong
answers, having printed a warning nobody reads. It now returns `bool` and
`jlcs_create_jit_with_libs` refuses to JIT without a host data layout. A guess
about the ABI is not a degraded mode of a thing that exists to get the ABI right.

**`register_symbol(jit, …)` took an engine and ignored it.** Both it and
`register_symbol_global` called one C entry point that unconditionally did
`DynamicLibrary::AddSymbol`, so the per-engine isolation the signature implies
had never existed. Two wrappers defining the same mangled name — two libraries
vendoring the same C++ class, or one library reached by two paths — shared a
single process-global slot, last registration winning, and the loser's thunks
dispatched into the other library's code.

A non-null engine now registers through `mlirExecutionEngineRegisterSymbol`
(already exported, never called). Null still means the global table, and that is
deliberate rather than residue: ORC resolves a symbol when it materialises the
code referencing it, so anything the engine must see during `create_jit` has to
be registered before an engine exists to hold it — which is precisely the window
`initialize_global_jit` uses for its `dispatch_` and EH helpers. `register_symbol`
now **throws** on a null engine instead of silently falling back, which is what
keeps the old claim from being makeable again.

`initialize_global_jit` still uses the global path, correctly. Moving it means
registering after `create_jit`, and whether ORC materialises anything during
engine creation — with the `dispatch_` symbols not yet present — is not
established. That is a load-order change, not a rename.

### Tier 1 is quarantined: experimental, unwired from the suite (2026-08-19)

Tier 1 — `Base.llvmcall` over per-function bitcode slices — is now explicitly a
**side project inside RepliBuild rather than a supported tier**. It still ships
and still works; nothing about the feature changed. What changed is that the
project stops pretending it is load-bearing.

The honest position, which the code already reflected: `[wrap.tier1] enable`
defaults to `false`, every Hub config pins `[link] enable_lto = false`, and so
**no shipped package takes a Tier-1 path by default**. Meanwhile nearly every
Tier-1 mechanism — the symbol pre-flight, the `@generated` output-mode demotion,
`_slice_const_name!`'s collision registry, the address-significance refusal, the
two portability guards emitted into every wrapper, `DISPATCH_TIER` vs
`dispatch_tier` — exists to make Tier 1 fall back to `ccall` safely. The doctrine
is that llvmcall is a passenger tier and never the driver.

**`devtests.jl` no longer runs its three suites** (`test_static_promotion.jl`,
`test_slicer.jl`, `test_tier1_dispatch.jl`). Each rebuilds `test/slice_test/`,
which is real compute for a path nothing takes.

They are **not orphaned**, and the distinction matters: `runtests.jl`'s "Every
test file is wired into a suite" testset — which exists because
`test_tier1_dispatch.jl` once shipped wired into neither suite and went unnoticed
until an audit — now carries an explicit `experimental` list naming all three,
plus two new assertions so the list cannot rot. One fails if a quarantined file
is deleted or renamed (a stale entry hides nothing while claiming to); the other
fails if a quarantined file is *also* wired (the quarantine leaked, and devtests
is paying for it while the list says otherwise). A **new** Tier-1 test file still
fails the guard until someone consciously adds it. Negative-checked: a bogus
exemption plus a re-added `include` fails exactly those two of the four asserts.

Run them by hand when working on Tier 1:

```bash
julia --project=. test/test_slicer.jl
```

Docs updated to match — `guide.md` had been calling `[wrap.tier1]` "the supported
path", which is exactly backwards; it now carries an experimental warning, as do
`config.md` and `index.md`. Improvements by PR are welcome, which is the point of
keeping it in the tree rather than deleting it.

Side effect worth noting: those three files were the only consumers of
`test/slice_test/replibuild.toml`, which is gitignored and regenerated by nothing
(`slice_test` is absent from `INTEGRATION_TESTS`). Unwiring them means the main
suite no longer depends on an untracked, machine-local config.

### `test_cache_invalidation.jl` was watching a file that never existed (2026-08-19)

The second of the two standing `devtests.jl` reds, and like the first it was the
test's fault. It failed **standalone** too, at 3/6.

It watched `build/m.ll` and compared mtimes across builds to prove that a
`[compile] flags` change invalidates the per-file IR cache. That path has not
existed since the cache was keyed: per-file IR lands at `build/m.c-<8hex>.ll`.
`mtime` on a missing file returns `0.0`, so `m3 > m2` was comparing `0.0 > 0.0`.

**Two hashes, answering different questions** — the thing to know before touching
this. The hash in the *filename* identifies the SOURCE and is stable across
compile-config changes; the compile fingerprint lives in the `.key` sidecar beside
it, and that is what gates the cache. Measured across the four builds, it moves
exactly as designed: `aebbc4e5` → `c101f5b1` when `-fvisibility=hidden` is added →
`aebbc4e5` when it is reverted. The testset asserts the sidecar now, including
that reversion restores the *original* fingerprint — a stronger statement than "it
recompiled", since an invalidation scheme that merely churned (a timestamp, a
counter) would satisfy the change case and fail the revert.

Its other assertion, `nsyms(so) == 1`, had become **unsatisfiable**. It counted
exported symbols to show `-fvisibility=hidden` took effect, but `internal_helper`
is external-linkage-and-hidden under that flag — the LUAI_FUNC shape — so static
promotion (2026-07-22, added after this test was written) correctly re-exports it
as `__rb_cachetest_internal_helper` with default visibility. Two symbols go in,
two come out, and only their *names* record that anything happened. Asserted by
name now, with the promoted name pinned explicitly so the next reader does not
have to re-derive why the count cannot move. 11/11.

**Also: `_assert_no_any_ccall_return` shipped guarded but ungated.** Its sibling
got a full testset in the same commit; this one had none, which is precisely the
shape that let `test_tier1_dispatch.jl` sit unwired and unrun. Now covered in
`test_wrapper_type_bindings.jl` — both call forms refused, and four must-not-flag
shapes that keep it narrow enough to be reachable: `::Any` in a Julia signature
(which the C generator emits for every unmodellable parameter), `Any` in a ccall
*argument* tuple, a `::Any`-annotated function with no foreign call in it, and a
well-typed `@ccall`. Plus the reachability check against the write path. CI 907.

### `test_slicer.jl` was red for two reasons, neither of them slicing (2026-08-19)

Both were the test's fault, and both made `devtests.jl` misreport — it aborts at
the first failing top-level testset, so this one file was hiding the ten after it.

**State leakage between suite files.** The coherence testset opened with
`t1_get_count() == 0`, commented "fresh process state", which is false in-suite:
`test_static_promotion.jl` (§7b) runs first over the same fixture in the same
process, and its single-copy proof *deliberately* writes absolute values —
`unsafe_store!(counter_ptr, 100)` then `st_bump(3)` leaves the counter at 103, and
`st_set_op(2)` leaves the op slot at `op_square`. Exactly 7 failures in-suite
against 154/154 standalone, reading as a Tier-1 regression when nothing was wrong
with the slices.

The subject of that testset is **coherence** — that a Tier-1 slice and a Tier-3
ccall address one datum — which is a statement about deltas, never about the
starting value. It asserts deltas now, which makes it independent of *every* prior
file rather than of one particular prior file. §7b's writes are not cleanup it
forgot; absolute stores are its proof mechanism, so the invariant belonged here.
The new opening assertion is strictly stronger than the `== 0` it replaces: that
one checked Tier 1 against a constant, this one checks both tiers against each
other before a single delta is taken.

**A core-engine test depended on another repo's build.** "Slicer: lua at scale"
read the RepliBuild-Hub lua build by absolute path, so it tracked whichever library
version happened to be built there. lua 5.5.1 turned `luaL_openlibs` into
`#define luaL_openlibs(L) luaL_openselectedlibs(L, ~0, 0)`; RepliBuild correctly
emitted `replibuild_shim_luaL_openlibs`, the plain symbol stopped existing, and the
slice was refused "function not found in module". Hub rebuilds are the integration
test — this file tests the mechanic.

Replaced by a self-contained fixture, `test/slice_test/src/slice_scale.c`: 64
leaves, 8 mids, one `st_sc_hub` whose transitive closure is the entire TU. It keeps
what lua covered (breadth, declarations-only under heavy fan-out, live `llvmcall`
against the same `.so` Tier 3 calls) and states it more sharply, because the
numbers are now knowable:

- The declarations-only property is **quantified** rather than gestured at — the
  hub's slice must be under 1/20th of the module it reaches all of, and under 60 KB.
- **The fan-out is asserted to be real.** `[link] optimization_level` is `"2"`, so
  without `noinline` the leaves fold into the mids and the mids into the hub, and a
  slice of an inlined-flat hub would satisfy "exactly one `define`" while testing
  nothing. Negative-checked: dropping `noinline` fails exactly the 9 fan-out
  assertions and nothing else.
- An **oracle independent of the other tier**. Tier-1/Tier-3 agreement proves
  coherence, not correctness — both tiers can be wrong together, which is how the
  eightbyte-coercion bugs survived. The fixture's closed form
  (`hub(v) = 192v + 8416`) is asserted alongside the cross-tier comparison.

Slicer 154/154 and scale 169/169, standalone and in §7b→§7c order.

### Two more ways a wrapper could kill its own module at include (2026-08-16)

`ccall` resolves its argument tuple and return type **eagerly, at method
definition**, so one bad type annotation takes the whole module down at `include` —
the class `_assert_wrapper_loadable` was built for in v3.2.0, when the name resolved
to nothing. libcurl found two more routes into it, plus a third shape that resolves
fine and crashes later.

- **A parameter can shadow a type its own ccall names.** `struct bufq *bufq` is
  ordinary C — the parameter and its struct share a name — and it emitted
  `function f(bufq::Any, …)` over `ccall(…, (Ptr{bufq}, …), bufq, …)`, where
  `Ptr{bufq}` resolves to the **local**. Julia refuses the method with "could not
  evaluate ccall argument type (it might depend on a local variable)". 26 functions
  over 5 type names (`dynhds`, `cshutdn`, `bufq`, `cpool`, `ssl_peer`), and **all
  1004 of libcurl's functions dead on it**. `_undefined_ccall_types` could not see
  it: `bufq` *is* declared by the module — it is merely unreachable from inside that
  one function. Fixed at the emission site by renaming the parameter, never the type
  (the tuple still has to reach it), and guarded by
  `_assert_no_shadowed_ccall_types`.
- **The `@ccall` form was never scanned.** `_undefined_ccall_types` matched only the
  classic `ccall((:sym, LIB), R, (T, …), …)` shape, and `@ccall` is what
  `generate_vararg_wrappers` emits — so **every variadic function was invisible to
  the guard**. libcurl's `curl_mfprintf(FILE *fd, …)` wrote `fd::Ptr{_IO_FILE}` out,
  Julia refused it at include, and the guard reported nothing wrong. Both forms are
  scanned now. The emission bug behind it is the same shape as the Cstring-policy
  one below: the variadic branch `continue`s past the emission loop, and
  `_resolve_forward_ptr` — the 2026-08-02 fix for exactly this class — lives inside
  the loop. Fixed parameters go through it before the varargs path sees them.
- **`Any` is not an emittable foreign return type; it is a crash.** In a foreign
  call `Any` declares that the callee returns a `jl_value_t*`, so whatever integer
  came back is dereferenced as a Julia object — SIGSEGV inside dispatch, on a later
  call, with a stack naming neither the wrapper nor the library. `Any` reaches a
  return position when the type mapper could not name the type; the C emission loop
  branches on that (it is the struct-return signal) and the variadic path did not,
  so it spliced the word straight into the `@ccall`. **libcurl shipped 18 — every
  `curl_easy_setopt`, `curl_multi_setopt`, `curl_share_setopt` and
  `curl_easy_getinfo` overload, i.e. the entire configuration API.** The variadic
  path now prefers the name sitting in the metadata's `c_type` when the module
  emitted it (`curl_easy_setopt` → `CURLcode`, which its non-variadic siblings
  already return), and falls back to `Cvoid` — discarding a value is recoverable,
  corrupting one is not. Guarded by `_assert_no_any_ccall_return`.

Both new guards run on the emitted **text**, from `_assert_wrapper_loadable`,
beside their sibling and for the same reason: the bug is the generator's
bookkeeping disagreeing with what it actually wrote. The shadowing guard is gated in
`test_wrapper_type_bindings.jl` — including the depth-aware parameter split, so a
`Union{AbstractString,Cstring}` annotation is not read as two parameters — and was
reproduced with no library in play, which is what makes it the generator's bug and
not curl's.

### Variadic overloads accept the values callers actually write (2026-08-16)

`f_Cint(fmt, 42)` did not work. Each variadic slot was annotated with its
**declared** type, so `va_1::Cint` rejected the `Int64` literal every caller writes
before the `ccall` was ever reached. The signature takes a **widened** type now —
`Integer`, `Real`, `Union{AbstractString,Cstring}`, `Any` for pointers — while the
`@ccall` keeps the declared one, where `cconvert`/`unsafe_convert` run and their
results stay rooted for the call. Widening the signature changes nothing about the
ABI, and it cannot introduce an ambiguity by construction: every overload is its own
named function (`f_Cint`, `f_Cdouble`), so no two of these signatures compete.

**The types that cannot legally name a variadic slot are now rejected at wrap
time.** C default argument promotion (C11 6.5.2.2p6–7) converts `float` to `double`
and every integer type of rank below `int` to `int` before the callee sees it, so
there is no ABI in which a caller writes 4 bytes into a slot read with
`va_arg(ap, double)`. Julia does not promote, and neither did this generator:
`va_1::Cfloat` wrote exactly `Cfloat`, the callee read 8 bytes where 4 were written,
and formatted whatever followed. **Wrong output, no crash** — the worst failure mode
a `printf` wrapper has, and one nothing downstream can catch, since the wrapper
loads and the `ccall` is well-formed. `Cfloat`/`Float32`/`Float16`,
`Cchar`/`Cuchar`, `Cshort`/`Cushort`, `Bool` and the sub-`int` sized integers now
error at wrap time, naming the type to use instead.

**Duplicate-method drops name what they dropped.** `_dedup_method_chunks` reported a
count, which cannot distinguish a D1/D2 destructor pair — where dropping one is
exactly right — from an `::Any`-collapsed overload pair, where a **distinct C++
entry point silently became unreachable** (imgui's `TreeNode(const char*, const
char*, …)` losing to the `void const*` form: same `(Any, Any)` signature, different
function). It now names the signature, the symbol dropped and the symbol that
shadowed it, reading the mangled name back off each chunk's own docstring rather
than threading it through. The same symbol on both sides means one C++ entry point
emitted two chunks, and prints without a shadower.

Fixed while here: `_method_sig_keys` split its parameter list on every comma, so a
type containing one — `Union{AbstractString,Cstring}`, `NTuple{8,UInt8}` — became
two arguments and silently changed the dedup key. It is depth-aware now.

### `class` is the demangler's prefix, not a scope — three ways to lose `this` (2026-08-13)

`class` in `compilation_metadata.json` is everything before the last `::` in the
demangled text preceding the first top-level `(`. That is a scope only when the
string is `Scope::name(args)`. Three cases where it is not, all found in one
session, all ending in a method wrapper called without its receiver — and **both**
receiver gates trusted it.

- **Itanium adjustor thunks** (`_ZTh`/`_ZTv`/`_ZTc`) demangle to `non-virtual thunk
  to Derived::~Derived()`, so the scope came out as the *phrase* `non-virtual thunk
  to Derived`. No aggregate is named that, so neither gate synthesized `this` and
  the wrapper exported a documented **zero-argument** function calling a method that
  needs `this` in rdi. Not latent: `ViTest.virtual_thunk_to_Diamond_tag()` cored the
  process — the virtual form dereferences the garbage `this` twice
  (`mov (%rdi),%rax; mov -0x20(%rax),%rax`) to read the vcall offset out of the vptr
  before it ever reaches the callee. Filtered at the source in
  `extract_symbols_from_binary`, beside the `__rb_*` exclusion, so it clears
  metadata, wrapper, manifest and `.mlir` in one place — **a thunk is a vtable-slot
  entry point, never an API function**; an override is reached through the vtable or
  after an explicit upcast, both already exercised by the fixtures. 12 symbols,
  fixtures only, 0 across the Hub.
- **Member function templates.** Itanium mangles a function template's return type
  into the symbol, so `bool gguf_reader::read<int>(…)` gave the scope
  `bool gguf_reader`. `gguf_reader` *is* a real class, so both gates declined and
  **every argument slid one register** — the silent direction. And template
  arguments contain `::`, so the flat split also cut inside them
  (`bool llama_model_loader::get_arr<std::vector<int, std`, with the function name
  coming back as `allocator<int> > >`); the comment on that split claimed "safe – no
  templates here". Class and name now derive from one depth-aware walker (angle
  **and** paren depth, clamped so `operator>`/`operator->` cannot drive it negative).
  The return-type strip bails on any prefix containing `operator` — a conversion
  operator is the only C++ name with a legitimate top-level space — degrading to the
  historical answer rather than to a new wrong one.
- **A constructor or destructor always has a receiver.** `struct_types` is not a
  complete list of a library's classes — a `.cpp`-local polymorphic class gets no
  type DIE — so when DWARF also stopped describing `this`, box2d's seven internal
  contact subclasses and imgui's eight `ImVector_*_destroy_ImVector` emitted
  zero-argument destructors. That is a fact about the debug info; "a dtor has a
  `this`" needs no table. Both gates test it first, and Hub-wide there are now
  **zero** zero-argument method wrappers.

**Validated by corpus diff before landing, not by eye**: old versus new over all
10,618 demangled names in the Hub and fixtures, every distinct change read. The
number that matters is the receiver decision — **62 methods gained `this`, 0 lost
it** (llamacpp 48, pugixml 8, box2d 4, tinyxml2 2). imgui's 29 class changes are
namespace→namespace and behaviourally inert, so counting class diffs alone would
have overstated the blast radius 2×.

**The same missing receiver had quietly gutted RAII.** The deleter scan inferred
"this function deletes a `Ptr{X}`" from `params[1]`, so when `this` disappeared the
`Managed` tables collapsed — box2d 30 → 1, tinyxml2 11 → 2, llamacpp 127 → 18. A
destructor names its own class, so the scan reads `class` now and never the argument
shape: tinyxml2 11/11, llamacpp 127/127, and **pugixml 2 → 10, imgui 8 → 79**, which
gained finalizers they should always have had. box2d ends at 29 rather than 30
because the missing one was a bug being removed: `Managedb2Fixture`'s finalizer
called `b2Body::DestroyFixture` passing the **fixture** as `this`, and qualified only
because the old metadata gave that method a single parameter.

The C++ gate is one function now, `_cpp_this_param`. Inlined in the emission loop,
the only way to test it was to re-implement it — a third copy of a decision that
already existed twice — and the first draft of the agreement test did exactly that
and reported 26 phantom disagreements, because it stripped template arguments where
the real gate keeps them. Gated by `test/test_symbol_hygiene.jl` (94 asserts, no
toolchain, in `runtests`), which drives **both real predicates** over every C++ Hub
package's metadata and fails on any disagreement — the assertion that would have
caught the ImGui incident, the thunks and these destructors, all three being gate
disagreements or shared blind spots. Negative-checked from four directions;
desyncing one gate reports 310 disagreements. Its own `checked > 1000` counter then
caught a bug in itself: a `try/catch` around `JSON.parsefile` swallowed
`UndefVarError: JSON`, so every package was skipped and the sweep proved nothing
while passing standalone.

All six C++ Hub packages rebuilt and green at baseline — box2d 15/15, tinyxml2
11/11, pugixml 13/13, imgui 202/202, llamacpp 33/33, hello_world loads — each with 0
undefined exports. mi_test 43/43, vi_test 40/40, stl_test 28/28. **pugixml is the
regression canary**: byte-identical apart from its timestamp, since full debug info
already supplied every `this`.

### All 14 JLCS ops have a producer, and liveness is computed now (2026-08-13)

`jlcs.dtor_call`, `jlcs.get_field` and `jlcs.set_field` were declared, lowered and
registered, and never emitted. This changelog and `docs/src/mlir.md` said otherwise
about `dtor_call` for months: `FunctionGen` builds `jlcs.scope(…) dtors([@sym, …])`
and `ScopeOpLowering::emitDestructors` emits `LLVM::CallOp` directly, so
`DestructorCallOpLowering` was registered and unreachable — and nothing could catch
it, because the destructor *tally* fires through the scope lowering and the suite
passed without the op existing.

- **`dtor_call`** — a destructor thunk is exactly this op's shape (one object
  pointer in, void out), so it stops going out as a generic `ffe_call`/`try_call`
  carrying SysV coercion a `(ptr) -> void` signature has no use for. It gained
  `may_throw` (UnitAttr plus invoke/landing-pad lowering, mirroring `vcall`'s) so
  the switch preserves EH exactly. **The arity gate is load-bearing, not
  defensive**: under Itanium a base-object destructor (D2) of a class with virtual
  bases takes a second **VTT** argument and `dtor_call` has no operand for it, so
  those keep the ffe/try path — converting them would drop the VTT silently.
  Negative-checked by relaxing `== 1` to `>= 1`, and the paired
  `_ZN2VbD1Ev`/`_ZN2VbD2Ev` fixtures prove the exclusion is arity, not name or class.
- **`get_field`/`set_field`** — the ciface hands a thunk a `void**`, so argument
  slot `i` is a field read at byte offset `8i`: one op instead of a constant plus a
  pointer-scaled GEP plus a load. `ArrayViewGen` also builds the `!jlcs.array_view`
  descriptor with one `set_field` per field — **the same terms the array-op lowering
  already reads it in**, where before it was a GEP chain on one side and byte
  offsets on the other with nothing tying them together.

**This changes what the IR says, not what it does, and that is proven rather than
argued**: all 188 symbols across `mi_test`/`vi_test`/`stl_test` compile to
byte-identical machine code across the switch, compared per function with `objdump`.
A text or even LLVM-IR diff proves nothing here — the IR differs in GEP element type
and `!dbg` numbering — so compare instructions, and `test_jlcs_producers.jl` §G pins
the per-shape equality permanently. The payoff is that the `.mlir` is the debugger's
source file, so `jlcs.get_field … {fieldOffset = 8}` is what gdb shows.

All three gained verifiers: the lowerings took a pointer operand on faith, which
ODS's `AnyType` accepts and LLVM translation then rejects far from the mistake.
`get_field`/`set_field` also reject a negative `fieldOffset`.

**Liveness is derived now, not asserted.** `test_jlcs_invariants.jl` §E reads the
mnemonics out of `JLCSOps.td`, greps `src/**/*.jl` for emission, and fails naming
any op whose tier moved — negative-checked against the previous state, where it
reports exactly `dtor_call, get_field, set_field`. That list had gone stale in the
docs three times; writing the guard is what stops a fourth.

Invariants 17/17, producers 57/57, templates 87/87, struct-ABI 30/30, multilib
14/14, mi 38/38, vi 33/33, stl 28/28.

### A dispatch tier decided a presentation policy: `char*` returns from Tier 2 (2026-08-12)

Calling `hello_message()` returned `Cstring(0x7fb608c37000)`. `use_mlir_dispatch`
emits its `invoke` and then `continue`s past the ccall branch, and the whole
`Cstring` policy lived below that line — so a Tier-2 `char*` return got **no
`Union{String,Nothing}` copy, no NULL check, no `_ptr` sibling, and its
`[wrap.cstring_owned]` deallocator silently discarded**. The reach is everything
C++: `is_ccall_safe` routes any function not marked `noexcept` to Tier 2, so only
`extern "C"` symbols kept the good path. **75 offenders, measured** — imgui 30,
tinyxml2 22, pugixml 13, llamacpp 9, hello_world 1 — **not one with a `_ptr`
sibling**. Not a memory-safety bug (the pointer was valid), and the leak risk was
latent, since both `cstring_owned` declarations in the Hub are C.

The fix is one derivation rather than a patched branch: `_cstring_wrapper_pair` owns
the policy wrapper, the `_ptr` sibling and the free, and a tier supplies only two
call-body strings and decides nothing else. Four emission sites consume it — C
ccall, C++ ccall, C++ Tier-2 JIT, C++ Tier-2 AOT. **Verified behaviour-preserving by
construction, not by eye**: re-wrapping cjson gives a byte-identical wrapper, and
box2d — 439 Tier-2 functions, no `char*` return — diffs **zero lines** against a
wrapper generated from unpatched HEAD. That diff is the real regression test for a
heredoc restructure.

`_assert_cstring_policy` refuses a wrapper where a `char*` return escapes as a bare
`Cstring` outside a `::Union{String,Nothing}` function and outside a `_ptr` variant.
**Its own first draft had the bug it exists to catch**: an unanchored
`@ccall .*::Cstring` matched an *argument* annotation and flagged all four of
box2d's void-returning `b2Log`/`b2Dump` overloads. Every pattern is anchored on the
return position now. Gated by `test/test_cstring_policy.jl` (53 asserts, no
toolchain, in `runtests`), including a check that the guard is reachable from
`_assert_wrapper_loadable`.

Separately and in the opposite direction: C++ docstrings advertised `-> Cstring`
while the ccall path already returned `Union{String,Nothing}`, so all 33 of
llamacpp's Union-returning functions were mis-documented. Both generators rewrite
the documented return now.

**The consumer fallout is the tell that the fix was right.** Three Hub drivers
hand-rolled a `cstr(x)` helper to undo the missing policy, and imgui's deep test
went 202/202 → 183 passing + 14 errored the moment the wrapper started returning
Strings. The ergonomic layer had been doing the wrapper's job. When a policy moves
into the generator, grep the consumers for the compensation before declaring
victory. Re-wrapped Hub-wide: policy-function count now equals `_ptr` count in every
package, **+77 `_ptr` variants**, and imgui deep is back to 202/202.

### Consumer compensation is a feature request: `dispatch_tier`, `struct_size` (2026-08-12)

A consumer written against a Hub wrapper hand-rolls a workaround for whatever the
generator failed to emit, and that workaround is executable evidence of a missing
feature. **The discriminator is recurrence** — one package is library ergonomics,
twelve is a gap. Two were mined out of `packages/*/test*.jl` and shipped:

- **`dispatch_tier(f)` and `DISPATCH_TIER`** — 12 files carried a helper reaching
  into the private `_TIER1_*` kernel to `code_typed` it and string-match
  `"llvmcall"`, 11 of them byte-identical. Nobody trusted `TIER1_FUNCTIONS`, because
  it records *intent* while the `@generated` kernel demotes at generation time. Both
  are emitted now: `DISPATCH_TIER` for the intent, `dispatch_tier` for the reality.
  Proven equivalent to the hand-rolled helper on all 84 of cjson's Tier-1 functions,
  0 disagreements — and the gap is real: move `julia/slices` away and `DISPATCH_TIER`
  still says `:tier1` while `dispatch_tier` says `:tier3` and the function still
  returns the right answer.
- **`struct_size` / `member_offset`, backed by `STRUCT_SIZES`/`STRUCT_OFFSETS`** —
  four packages re-parsed `compilation_metadata.json` for facts the generator held
  while emitting. Scoped to structs the module actually declares. It lands exactly
  on the recorded pain: `llama_context_params` is 160 bytes with `embeddings` at
  128, which callers had been patching through hand-rolled offset tables.

Both are derived from the final chunks post-dedup, same discipline as
`_tier1_emit_slices!`, and neither is exported — qualified access only, so they
cannot shadow a consumer's names.

**The self-audit of this feature found two silent-wrong-answer defects in it**,
which is the part worth remembering: a feature built to remove silent wrongness is
not automatically free of it.

- The layout table re-derived the type-name spelling instead of using each
  generator's own sanitizer, so a trailing underscore made
  `ImChunkStream<ImGuiTableSettings>` never match the emitted name and the type was
  **dropped silently**. Cost: 100 of imgui's 262 declared structs, 29 of tinyxml2's
  42, 17 of pugixml's 69 — precisely the templated types whose size a caller cannot
  compute any other way. After: imgui 162 → 248, tinyxml2 13 → 24, pugixml 52 → 62,
  llamacpp 434 → 1222.
- `DISPATCH_TIER` was last-write-wins over a non-injective key. One Julia name can
  carry methods on different tiers — a Tier-1 primary plus a Tier-3 convenience
  overload is the common shape — so the table answered confidently and wrongly for
  **48 names** (cglm 35, miniaudio 5, llamacpp 3, lz4 3, imgui 2). They are recorded
  `:mixed` now, which fails an ill-posed `=== :tier1` assertion loudly instead of
  satisfying it by luck.

**A third came from checking the JIT hot path**: `dispatch_tier` calls `code_typed`
on a `@generated` kernel, which **forces that kernel to generate** — it looks
read-only and is not. Runtime order is safe, measured both ways; output mode was
not. A consumer doing `const T = M.dispatch_tier(:f)` at module scope froze
`:tier3` into its pkgimage while the same function re-generates to `llvmcall` in the
next session — reporting the precompile worker, not the session that runs. It
returns **`:deferred`** in output mode now and does not probe at all, which also
keeps generation out of the one place the sliced-llvmcall deadlock class lives. And
`Base.ccall(...)` is `UndefVarError` — `ccall` is syntax, not a binding — the exact
opposite hazard to the Base-qualification rule the emitted helpers otherwise follow,
so the guard carries a comment saying so.

Hub sweep: all 20 packages re-wrapped, 21 call sites migrated, 12 helper definitions
deleted (3 of them already dead), 4 metadata re-parsers replaced by delegation. All
16 deep verifiers green at their recorded counts.

### Docs: the dialect page came back, and the TOML reference was wrong (2026-08-12)

`docs/src/mlir.md` had been dropped from git alongside pages that were genuinely
retired, and its `make.jl` entry went with it. The file survived untracked in the
working tree, so the loss showed only as a 404 at `/dev/mlir/` — the sidebar simply
had no Tier-2 dialect page. It is tracked and relisted, and rewritten as the
architecture thesis for the dialect: what `ccall` cannot express, why not a C shim
and why not libffi, the DWARF→JLCS→LLVM→ORC pipeline, the thunk contract with a
worked example carried from the Julia call site to the three instructions gdb shows,
a 14-op table with **producer and verifier status per op**, the SysV lowering, and
the failure classes the design exists to make loud.

`docs/src/config.md` was rewritten as the authoritative TOML reference. It needed a
rewrite rather than a patch because of what it *claimed*:

- The `[wrap.varargs]` example was wrong twice over — it showed C type names where
  the parser requires Julia ones, and included the fixed format argument when only
  variadic args are listed. Copying it produces a hard error at wrap.
- The preserved-keys list said 6; `Discovery.PRESERVED_TOML_KEYS` has 8. Same error
  in three other pages.
- `optimization_level` documented a default of `"2"`; the parser's absent-key
  default is `"0"`.
- `[dependencies].commit` was entirely undocumented, despite being in every Hub
  config and a hard error on mismatch.
- A dependency's default `type` is `"local"`, not `"git"` — so a `[dependencies.x]`
  with a `url` and no `type` silently never clones.
- `exclude` was described as glob-matched; it is `==` / prefix / suffix / substring.
- **Keys that are parsed and never consulted** are marked as such now: the whole
  `[discovery]` section, `[workflow] stages`, `[wrap] style`, `[binary]
  strip_symbols`, three `[paths]` keys, and `[project] version`. `extra_link_libs`
  is `[ingest]`-only and silently inert under `[link]`. Unknown keys get **no
  warning at all**, which the page now says outright.

Also corrected across the manual: the README claimed C++ multiple inheritance is not
modelled (MI and VI have both been built since 2026-07-17), `internals.md` said
`ThunkBuilder` shells to `llc` and described `JITManager` as a singleton engine, the
Tier-1 lua figures were stale everywhere, and the README linked four pages that have
never existed. The docs build exits 0 with **zero** link or xref warnings.

**The 2026-07-25 audit pattern held again**: everything wrong was a *derived* claim
— counts, defaults, key lists, links — never the prose explaining how something
works. Check derived claims first.

### Tier 2 accepts a `Base.Ref` argument without segfaulting (2026-08-09)

A thunk slot holds a pointer to a location containing the argument value — the
emitted thunk double-loads, slot → storage → value. `Ref(x)` gives that for an
isbits `x`, but a `Base.RefValue` **is already an indirection**, so wrapping it
again spends one level too many and the callee receives the struct's own first eight
bytes as an address.

That is the spelling a caller reaches for first: a C++ `T const&` parameter
generates `::Ref{T}`, and `Ptr{T} <: Ref{T}` means the annotation accepts a
`RefValue` silently with nothing to type-check it away. It presents as a raw SIGSEGV
with a C++ frame in the backtrace, so it reads as a thunk ABI bug rather than a
caller mistake. Found on `ImGui::ButtonEx` dereferencing `size_arg`; the blast
radius is every C++ package with by-reference struct params, and imgui is full of
them — `Button`, `Dummy`, `SetNextWindowSize`, every `ImDrawList` primitive.

It is flattened to a raw pointer first, exactly as `AbstractString` already was — a
case that exists for the same reason. The `!(<: Ptr)` term is load-bearing: a raw
pointer is already the flattened form, and re-flattening would strip the level the
thunk needs. Both `invoke` methods carried byte-identical copies of this prologue,
which is how a fix to one silently misses the other, so it is one
`_arg_marshal_plan` now.

Verified on a five-line by-reference fixture with no library in play — `Ptr`,
`RefValue`, two `RefValue`s and mixed — then at the original crash site:
`Button(label, Ref(ImVec2))` returns and the item rect measures 80×20, the exact
vector passed, so the bytes arrive rather than merely not crashing. runtests
626/626, imgui deep 202/202.

### DWARF extraction: real signatures, and no system-header types (2026-08-09)

Four defects, one root cause — information DWARF carries that nothing read.
Measured coverage before this was 23 of 36 tags and 14 of 60 attributes.

1. **`DW_AT_specification` was never followed** — 22.8k occurrences. An out-of-line
   C++ method definition carries neither `DW_AT_linkage_name` nor `DW_AT_name`, only
   a back-reference to its in-class declaration. Functions are keyed by linkage
   name, so the definition DIE was discarded whole, taking its named parameters with
   it; the declaration survived, and its parameters are unnamed.
2. **Typed-but-unnamed parameters were dropped.** A parameter's ABI identity is its
   type and position — the name is documentation. Requiring both dropped every
   parameter of every declaration-only DIE, **including the implicit `this`**.
   `DW_AT_artificial` marks that receiver, so it is named `this` rather than `argN`,
   which is what stops both generators from injecting a second one; either attribute
   order is handled, since the parameter context must not stay open past its own DIE
   (the 2026-07-26 phantom-parameter leak).
3. **C++ reference returns are pointers, not `Ref`s.** Both mappers render `T&` as
   `Ref{T}` — correct and ergonomic for a parameter, where Julia converts at the
   boundary. In return position `JITManager` allocates the buffer as `Ref{T}()`, so
   `Ref{Cvoid}` asks for `Ref{Ref{Cvoid}}()`: an `UndefRefError` raised before the
   call is even made, reported as `Ref{Nothing}` because `Cvoid === Nothing`, which
   reads as a void-return bug. `ImGui::GetIO()` died here.
4. **Types had no `DW_AT_decl_file` gate.** Functions are gated by
   `nm --defined-only`; types were not, so `hello_world.cpp` — which declares zero
   types — emitted six (`max_align_t`, `__mbstate_t`, `ldiv_t`, …) plus accessors
   and exports for them. A system-directory blocklist, deliberately not a
   project-root allowlist: dropping a type the project declares produces an
   undeclared name in a signature, while keeping a stray system type costs only
   noise.

`GeneratorCpp` additionally degrades a parameter annotation naming a type the module
never emitted to `Any`, since argument types resolve eagerly at method definition
and an unbound name kills the whole module at include. That was latent until (1)
recovered `xml_attribute::set_name(std::basic_string_view<…>)` — 44 uses, wrapper
dead at load.

Measured: ImGui arity warnings **3891 → 2**; methods disagreeing with their own
mangled signature **1103 → 0**; parameter names real (`Begin(name, p_open, flags)`,
was `arg1..arg3`). pugixml 0 warnings, 303 functions with a DWARF-supplied receiver,
407 signatures gaining real names. hello_world types 6 → 1 — only the compiler
builtin `__va_list_tag`, which has no `decl_file` — exports 7 → 4, and its dag-diff
line disappears.

**Expect `struct_definitions` counts to drop on rebuild**, which is this fix and not
a regression: llamacpp goes 2864 → 465, and every dropped name was classified —
**0 project types lost, all 292 `llama_*`/`ggml_*`/`gguf_*` retained**; the 2399
dropped are libstdc++ and libc (`basic_string`, `basic_ostream`, `cpu_set_t`,
`div_t`, `error_category`).

### MLIR thunks: no receiver for namespace-scoped functions (2026-08-09)

The Julia generator already gated `this` synthesis on the scope being a real
aggregate. The thunk generator did not — it keyed on `is_method` alone, which
upstream is `contains(demangled_prefix, "::")` and is therefore true for both
`Obj::get()` and `ImGui::GetVersion()`. Itanium mangles the two identically, so the
name cannot tell them apart.

The two sides then disagreed about the argument array: Julia correctly emitted a
zero-argument call passing an empty `Ptr{Cvoid}[]`, while the thunk loaded
`args_ptr[0]` and dereferenced it. SIGSEGV on the first Tier-2 call. **Dear ImGui:
788 thunked `ImGui::` functions, 174 of them zero-arg — immediate crash — and the
remaining 614 with every argument shifted one slot.**

Two generators deriving one decision independently, the same shape as `ffe_call` and
`try_call` each carrying their own SysV coercion. Note the aggregate table keys on
the bare name (`xml_document`) while `class` carries full scope
(`pugi::xml_document`), so every `::` suffix is tried, angle-bracket-depth aware —
matching the full name alone silently strips `this` from every namespaced class
method, which took pugixml's `load_string` from 3 args to 2 before it was caught.

`ImGui::GetVersion()` returns `"1.92.9b"` where it previously segfaulted. mi_test
38/38, vi_test 33/33, stl_test 28/28, producers 36/36.

**Two generators agreeing is necessary, not sufficient** — see the receiver-gate
entry above, where this fix brought both sides to a shared *wrong* answer on the
Itanium thunks they had previously merely disagreed about.

### DAG diff was reporting its own layout model as wrapper drift (2026-08-08)

`DAGDiff` compares the DWARF layout against the layout Julia would compute, and
routes any function returning or taking a divergent struct by value to Tier 2.
Auditing whether it still earns its keep turned up 209 functions forced to Tier 2
Hub-wide, of which **3 were flagged by nothing else** — all three
`RETURN_CONV_MISMATCH`, the class that produced the 2026-08-05 heap smasher.

All three were bugs in the model, not in the wrappers. Julia's own `sizeof` and
`fieldoffset` on the emitted structs match DWARF byte for byte:

- **Enum members sized 0.** Enums are keyed `"__enum__<name>"` in
  `struct_definitions` while a member's `c_type` names them bare, so the lookup
  missed every one. In `build_julia_graph` the running offset then never
  advanced, and every member *after* an enum reported as drifted — with
  `byte_size` still correct, the tail pad absorbing the difference. `b2BodyDef`
  presented as 80 bytes, 18 members, 8 wrong offsets.
- **Aggregates aligned to their size.** `min(size, 8)` is right for scalars and
  wrong for any struct wider than its alignment. `ma_vec3f` — three floats, 12
  bytes, 4-aligned — was treated as 8-aligned, pushing the next member from 36
  to 40 and `ma_spatializer_listener_config` from 48 bytes to 56. Julia reports
  `sizeof` 48 and `fieldoffset` 36.
- **Packed detection used the same rule**, so the same 4-aligned member at
  offset 36 read as "unnatural alignment" and the type was called packed, which
  propagates to every function returning it by value.

All three now go through `_type_alignment` and `_member_size`, one derivation
each. Hub-wide: mismatched types 902 → 710, functions forced to Tier 2 209 → 70,
and **functions where DAGDiff was the only reason: 3 → 0**.

That last number is the honest summary: on the current Hub, DAGDiff's entire
apparent unique contribution was its own false positives, and every genuine
mismatch it reports is also caught by `is_ccall_safe`. It is kept because the
class it uniquely covers — member drift with a matching `byte_size`, which the
struct-size guard cannot see — is the one that corrupts silently. A model that
cries wolf on 139 functions cannot be trusted to catch the one that matters.
Pinned by 11 new asserts including a genuinely-packed control, so narrowing the
screen cannot become disabling it.

**No wrapper bug was found.**

### Array-view thunks now honour the manifest and the type blocklist (2026-08-08)

`generate_function_thunks` has consulted the wrapper's `thunk_manifest.json`
since dead-thunk elimination existed. The array-view producer, added later,
never received it — so it emitted a get/set pair for every fixed-size array
member in the DWARF whether or not anything could call it.

Found by diffing what the JIT emitted against what the manifest asked for, which
is a comparison `RepliBuild.Debug` made possible earlier the same day:

```
emitted   2759          # llamacpp
manifest  2335
          ────
           424  all prefixed jlcs_av_, 212 pairs across 208 owning types
```

`grep -c jlcs_av_ julia/Llamacpp.jl` is **0**. Every one of them was lowered and
JIT-compiled on each load, reachable by nothing — 15% of the package's thunks.
Not a leak so much as the consumer side being unbuilt: `ArrayViewGen` produces
the thunks, and `GeneratorCpp` does not yet emit the Julia accessors that call
them (see the array-view roadmap entry). Gating on the manifest makes the cost
follow the feature — they return by themselves once accessors are emitted and
land in the manifest. `nothing` still means "no manifest, emit everything".

The type blocklist applies too, and separately: 7 of those pairs were over
`_IO_FILE` members. A wrapper never declares a type for a blocklisted internal,
so an accessor over one could not have been called even in principle.
`INTERNAL_TYPE_BLOCKLIST` therefore moves from `Wrapper` to package level —
`IRGen` loads first and could not see it, which is exactly why the second
producer went unscreened. `Wrapper._INTERNAL_TYPE_BLOCKLIST` is now an alias to
the same object, asserted identical in the tests rather than kept in step by
hand.

llamacpp: 2759 → 2335 thunks, MLIR 4.88 → 4.15 MB, object 3.21 → 2.86 MB.
Hub-wide the reachability gate currently reaches 4 of 16 packages — **only those
four ship a `thunk_manifest.json`**, and the rest take the documented
emit-everything fallback. Re-wrapping them writes a manifest and closes the
remaining 474.

### `RepliBuild.Debug` — read the emitted code without running anything (2026-08-08)

The gdb path from the entry below needs a live process stopped at exactly the
right moment. This is the same information from a file.

`enableObjectDump` was the one `ExecutionEngineOptions` debug field still at its
default — and `dumpObject` had been a parameter of `jlcs_create_jit`, and of
Julia's `create_jit`, since the JIT was written: threaded all the way down and
then dropped on the floor. The knob read as supported and did nothing. It is now
wired, plus a `jlcs_dump_object` export, so the engine's emitted object can be
written to disk.

`REPLIBUILD_JIT_OBJDUMP` turns it on. The object lands at `<pkg>/.debug/obj/<lib>.o`
beside the generated MLIR, and carries the same DWARF the JIT registers with gdb
— so `objdump -dS` interleaves dialect ops with the machine code they became,
with no debugger involved:

```
  %ret_val = "jlcs.vcall"(%val_1) { slot = 2 : i64, may_throw } : (!llvm.ptr) -> i32
     5db:  48 8b 07           mov    (%rdi),%rax
     5de:  48 8b 40 10        mov    0x10(%rax),%rax
     5ea:  ff d0              call   *%rax
```

The new `RepliBuild.Debug` module reads those artifacts: `thunks`, `mlir_body`,
`disassemble`, `dwarf`, and `walk` (one thunk, both views). It touches no live
state, so it answers about a package this process never built. Source resolution
needs no working directory — `DW_AT_comp_dir` is absolute.

Off unless asked for: the object cache retains every emitted object for the
engine's lifetime. It also cannot be enabled after the fact — MLIR needs the
cache to exist when the engine is created, and engines are cached per library
per process — so the environment is read before the engine exists rather than
offered as an argument nothing on that path could supply, and asking for a dump
without it is a hard error naming the recipe rather than an empty file.

Two things this pins that nothing else would have noticed, because thunks keep
working without them: that the DWARF still points at the generated MLIR, and
that `.debug_info` contains **only** a compile unit — no `DW_TAG_subprogram`, no
variable DIEs. That is what `LineTablesOnly` means here, and it is why `info
locals` and `p %val_1` come back empty in gdb while file and line work perfectly.
Asserted negatively in `test_debug_inspection.jl` (devtests §16, 43 asserts), so
raising `emissionKind` to `Full` announces itself as a failing test.

### JIT'd thunks are debuggable in gdb, at source level (2026-08-08)

Break on a mangled thunk name and gdb stops *inside the emitted MLIR*, by file
and line, with `list` and `disassemble /s` working:

```
Thread 1 "julia" hit Breakpoint 1, _ZNK5Base15get_aEv_thunk () at jlcs_83a242c27fb37885.mlir:126
126	  %val_ptr_1 = llvm.load %arg_ptr_1 : !llvm.ptr -> !llvm.ptr
Producer is MLIR.   Compiled with DWARF 4 debugging format.
```

Three pieces, each small:

- `jlcsModuleCreateParse` takes a `sourceName`. MLIR's parser stamps a
  `FileLineColLoc` naming its buffer onto every op; left unnamed, in-memory text
  gets `-` and the debugger asks for a file that cannot exist. Passing the extra
  argument to an older two-parameter build is harmless under x86-64 SysV, so a
  stale symlinked dialect degrades to unnamed parsing rather than misbehaving.
- `createDIScopeForLLVMFuncOpPass` runs last in the lowering pipeline — it
  operates on `LLVM::LLVMFuncOp`, which only exist after `ConvertFuncToLLVM`. It
  materializes the `DISubprogram` that every `DILocation` needs as a scope.
  Without it the object carries no DWARF at all, and the JIT event listeners
  have nothing to report.
- The module text is written to `<pkg>/.debug/mlir/jlcs_<sha256[1:16]>.mlir` —
  the file those locations point at. It is not diagnostic output, it is the
  debugger's source file. Content-keyed, so re-running a build reuses one path
  instead of accumulating a file per run. It sits beside the wrapper rather than
  in a tempdir so it travels with a vendored one; an unwritable install falls
  back to tempdir, losing co-location rather than the source view.

`clean()` now removes `.debug` alongside `build/` and `julia/`, and it is
gitignored — regenerated at the next JIT init from the module text, so the
capability is self-healing and a tracked copy could only ever be stale.

Two gdb flags are mandatory, not stylistic: `set breakpoint pending on` (the
thunk does not exist until the JIT emits it, so the breakpoint cannot resolve at
load) and `handle SIGSEGV nostop noprint pass` (Julia's GC uses SIGSEGV for
write barriers). `disassemble /s` interleaves dialect ops with the machine code
they became.

What it changes: eightbyte coercion, `byval` vs aggregate splitting, sret buffer
sizing and vtable slot arithmetic move from reasoned to observed. The 2026-08-05
heap smasher (`llama_context_params` emitting 200 bytes against a native 160)
would have been `break` → `finish` → `x/25gx` on the sret buffer.

### perf jitdump is opt-in; it had been writing to `$HOME` unasked (2026-08-08)

MLIR's `ExecutionEngineOptions` defaults **both** JIT listeners on, and
`MLIRExecutionEngine` is linked `--whole-archive`, so nothing in this repo ever
had to ask for it. LLVM hardcodes a `.debug/jit` suffix under `$JITDUMPDIR`,
which was never set — so every JIT'ing process dropped a jitdump in
`$HOME/.debug/jit` with nothing rotating or expiring it. 718 directories /
164 MB had accumulated unnoticed.

The perf listener is now behind `REPLIBUILD_JIT_PROFILE`, read in both places
that matter: the dialect decides whether to register it, and Julia points
`JITDUMPDIR` at `~/.replibuild/jit-sessions/session-<pid>/` (or at the variable's
value, if it looks like a path) before the **first** engine is created — LLVM's
perf listener is a process singleton, so the earliest setting wins. Filing the
dump per package would have been wrong: it is one file per *process* holding
every symbol from every engine, verified by loading two wrapped libraries in one
session and getting a single dump containing both.

The GDB listener stays on unconditionally. It has no filesystem side effects,
and it is what makes the entry above work.

## v3.2.0 (2026-08-08)

Minor, not patch: Tier 1 is un-parked, wrappers gained portability guards and
struct setters, anonymous unions are modelled, and `[dependencies]` grew a new
key. It also carries the Tier-2 struct-ABI fix below, which **requires re-wrapping
vendored wrappers** — a patch number would tell `~3.1` users this was safe to take
blindly. No exported API was removed since v3.1.0.

### Git dependencies can be pinned to an immutable commit (2026-08-08)

A git tag is a mutable ref. Upstream — or whoever takes over the account — can
force-push `v1.15` to different content, and `git checkout v1.15` fetches it with
no signal. Every dependency RepliBuild builds from was pinned that way, which is
the vector of the 2026 AUR supply-chain waves applied to C sources.

`[dependencies.<name>]` now accepts `commit = "<40-hex>"` beside `tag`. It is
verified after every clone and checkout, and a mismatch is a **hard error** that
refuses to build — a warning would let the compile proceed on unverified source,
which is the failure this exists to stop. The value is validated at config-parse
time (full 40-hex only; abbreviated shas are rejected as ambiguous against future
history) so a malformed pin cannot reach the resolver and read as "no pin", and it
round-trips through serialization so `discover(force=true)` cannot silently strip
it.

Separately, the `<name>.resolved` cache marker now records the resolved object
name from `git rev-parse HEAD`, so a cached checkout that drifted out from under
the marker is caught and re-resolved loudly rather than compiled. That layer dies
with `clean()`, which deletes the clone and the marker together — the declared pin
is the only one that survives the cold-rebuild path, which is exactly the path Hub
`test.jl` takes on every run.

Pins are optional; a recipe without one keeps the previous trust-the-tag
behaviour. All 19 RepliBuild-Hub recipes are now pinned. Note `git fetch --tags`
does not move an existing local tag without `--force`, so a warm cache is sticky
while a cold rebuild follows the move — deliberately asymmetric in the safe
direction, with the pin as the detector.

### libJLCS.so shipped ten undefined symbols for its entire life (2026-08-07)

`dlopen(libJLCS.so, RTLD_NOW)` refuses the library. Eight op `build()` bodies
(`TypeInfoOp`, `GetFieldOp`, `SetFieldOp`, `VirtualCallOp`, `LoadArrayElementOp`,
`StoreArrayElementOp`, `MarshalArgOp`, `MarshalRetOp`) and `ArrayViewType`'s
`getElementType()`/`getRank()` were declared and never defined. Two causes, one
shape — TableGen emits a declaration and expects the body in C++: for the ops it
is `skipDefaultBuilders = 1` with a bodyless `OpBuilder<(ins …)>`; for the type it
is `genStorageClass = 0`, where `CStructType`'s manual accessors were written and
`ArrayViewType`'s, ten lines away, were not.

It survived because `MLIRNative.jl` reaches the library through
`ccall((:sym, path), …)`, which binds **lazily** — an undefined symbol nothing
calls is never looked up — and because every producer builds IR as text and parses
it, so no call site ever existed. The library loaded, the dialect worked, and the
whole Tier-2 suite was green on a binary the loader would reject if asked to
resolve it eagerly. Surfaced only by an unrelated `RTLD_NOW` probe.

Guarded by `test_jlcs_invariants.jl` §D: a fresh **subprocess** `dlopen` with
`RTLD_NOW` (an already-open lazy handle would not be upgraded), plus an
`nm -D --undefined-only` sweep so a failure names every offender rather than the
first one the loader trips over.

### Export lists and type screens derive from emitted source, not intent (2026-08-07)

Two defects with one root: a generator deciding something by consulting its own
bookkeeping rather than the source it actually produced.

**Exports — 102 undefined names across the Hub.** The export list was built
independently of the definitions, and both generators computed it *before*
`_dedup_method_chunks` (the C one also before `_tier1_emit_slices!` and the Tier-1
registry). Names dropped after that point stayed on the export line, so `using`
offered bindings that raise `UndefVarError`. `_export_statement` now filters
against `_defined_names(module_body)` read off the parsed emitted source, and is
built below every step that can remove a definition. Measured: sqlite 64,
llamacpp 22, lua 12, miniaudio 2, zlib 2 → **0 everywhere**.

**Union accessors — 76 dropped, all valid.** The undeclared-type filter captured
`Ptr` out of `::Ptr{Cvoid}` — the *constructor*, not the type. No wrapper defines
`Ptr`, so every pointer-typed accessor was dropped whatever its pointee: sqlite's
`p4union` kept 1 member of 16, lua's `Value` 3 of 6. A `sqlite3_value` you could
not read a pointer out of. `_JULIA_TYPE_CTORS` joins the accepted set; the second
regex still extracts the pointee, so a genuinely undeclared one is caught as
before.

### Unmodellable structs report at `@debug`, not `@warn` (2026-08-06)

`StructGen` is reached from per-library JIT engine init, which runs inside a
generated wrapper's `__init__` — the **consumer's** load path, on their stderr. A
whole-library wrap reaches every system and STL type the headers dragged in
(llamacpp: 57 of 2864, almost entirely libstdc++ internals), so `using LlamaChat`
printed eleven paragraphs about `_BracketMatcher<…regex_traits<wchar_t>>` before
doing anything.

Nothing an application user can act on, and the degrade is safe by construction:
the region is emitted at its exact DWARF size, so the ABI is unaffected and only
field addressability from Tier-2 IR is given up. `JULIA_DEBUG=RepliBuild` opts
back in, and the full set stays inspectable as `StructGen._LAYOUT_WARNED` with no
logging configured. Also uncapped — at `@debug` you are only seeing these because
you asked, and a truncated list is the wrong answer to "show me every struct you
could not model".

### The dead llvm-dwarfdump fallback is gone — it was silently wrong (2026-08-06)

`extract_dwarf_return_types` tried readelf, then fell back to `llvm-dwarfdump`.
Both halves were dead. macOS is unreachable by construction (`__init__`
hard-errors on non-Linux), and the readelf-unavailable half was **worse than
dead**: `parse_dwarf_dump` is a readelf-format parser keyed on
`<level><offset>: Abbrev Number: N (DW_TAG_*)`, where the level is load-bearing —
closing DIE contexts on depth is what fixed the phantom-parameter leak.
`llvm-dwarfdump` prints neither a level nor an abbrev number, so it matches
nothing.

Measured on `libzlib.so`: readelf yields 198 functions / 17 structs / 5 globals /
51 typedefs; the same parser on a valid 1.4 MB dwarfdump of the same binary yields
**0 / 0 / 0 / 0** — at exit code 0, so the "Failed to read DWARF info" warning
never fired either. The build would continue, every function would fall back to
`parameters_source: "inferred"`, and the wrapper would ship with signatures
guessed from symbol names. A fallback that returns empty converts a missing tool
into a plausible-looking wrapper. Missing readelf is now a hard error naming
binutils, and a non-empty dump that parses to zero functions is a hard error
naming the dialect mismatch.

### ⚠ Tier-2 MEMORY-class struct ABI — re-wrap your vendored wrappers (2026-08-05)

**Action required.** If you vendored a generated wrapper for a library that
passes or returns structs by value, re-run `wrap()` and replace it. A wrapper
generated before this change can corrupt the caller's heap on a Tier-2 call,
silently, with every test passing.

A non-packed struct was closed with one trailing filler of
`byte_size - sum(member sizes)`. That double-counts: LLVM inserts the *interior*
alignment padding itself, and DWARF reports enum members with `size = 0`, so the
filler paid a second time for gaps the natural layout had already paid for.
Every non-packed struct with interior padding came out **larger than the C type
it models** — and because `llvm.emit_c_interface` stores a MEMORY-class result
straight into the caller's buffer, while `JITManager`'s `_invoke_call` sizes that
`Ref{T}` from the Julia struct (the true `byte_size`), every such call wrote past
a live Julia object.

Measured on llama.cpp: `llama_context_params` emitted 200 bytes against a native
160 and overran a 160-byte `Ref` by 34; `llama_model_params` 80 against 72, by 8.
**Every member offset held the right value**, which is why it presented as
intermittent corruption — a garbage `n_ctx` in one session, a SIGSEGV in the next
— rather than as marshalling, and why it was first misdiagnosed as "the 160-byte
return is broken, the 72-byte one is fine". Both marshalled perfectly and then
overran; only the amount differed. pugixml's `xml_parse_result` (24 B, three
members summing to 8) emitted 40 bytes and had been overrunning its `Ref` by 16
on every `load_string` **since it shipped**, with all 13 of its tests green.

Members are now laid out at their DWARF offsets with explicit padding, verified
against a Julia mirror of the dialect's own `abiSize`/`abiAlign` — one shared
derivation for all three body builders, since the buggy filler line had been
copy-pasted into each. A struct that cannot be laid out consistently degrades to
a `byte_size`-sized opaque region and warns: opaque, but never the wrong size.

Three further fixes on the same path:

- **MEMORY-class by-value ARGS now use `llvm.byval`** (closes a standing ledger
  entry). SysV passes a >16-byte by-value struct as a caller-owned copy in the
  outgoing stack argument area. The lowering did neither shape right: packed
  structs went as a bare pointer (by *reference* — the callee reads an address
  where it expects bytes), non-packed ones as an LLVM first-class aggregate (the
  backend splits it per element across registers, shifting every later argument).
  `llama_model_load_from_file(path, llama_model_params)` segfaulted on the
  second. Both now take one path — alloca + store + pointer with `llvm.byval(T)`
  and `llvm.align`, stamped on the call site *and* the declaration — emitting
  what clang emits. `ffe_call` and `try_call` had carried independent copies of
  the whole coercion; they share one `buildSysVCallShape` now.
- **Fixed-size array members** map to `!llvm.array` instead of falling through to
  the `!llvm.ptr` fallback, which claimed 8 bytes at align 8 for `int8_t[32]`.
  Every ggml quant block failed layout on this. Tested before the pointer branch
  so `char *[4]` does not match on the `*`; `uint8_t` and the signed/unsigned
  `char` spellings were missing from the scalar table too, invisible while arrays
  never looked at their element type.
- **Byte-blob structs got setters.** They had accessors in one direction only,
  and the only constructors are the zero-initializer and the raw-bytes inner one,
  so a param struct built *by* the library was read-only. On llama.cpp that is
  the only path in — an embedding model returns NULL unless `ctx.embeddings` is
  set, and callers were patching bytes through hand-rolled offset tables copied
  out of `compilation_metadata.json`. Now `setproperty(x, :f, v)` and
  `setproperties(x; f = v, …)`; immutability just means they return a new value.
  `Base.setproperty!` is defined only to replace Julia's "immutable struct cannot
  be modified" with a message naming the alternative.

### Generated wrappers no longer damage their consumer (2026-08-05 → 08-06)

A generated module is a namespace the **library** populates, and its export list
is harvested from every symbol that reached the debug info — libstdc++ included.
Two ways that reached out and broke code outside the wrapper:

- **`error` is rebindable by the library.** llama.cpp pulls in libstdc++'s
  `std::codecvt_base::result`, whose members include one named `error`, so the
  emitted `@enum` rebound it for the whole module: every failure path in the
  wrapper — including the long-standing `getproperty` "no field" branch — raised
  `MethodError: objects of type result are not callable` instead of its message.
  cJSON is a second, independent instance (a `struct error`). Nine emission sites
  are `Base.`-qualified now, and `_assert_base_calls_qualified` refuses to write
  a wrapper containing an unqualified one — keyed on a string-literal first
  argument, which is what separates the generator's own diagnostics from the
  library's, since a library owning the name legitimately emits `struct error`,
  `function error()` and `return error(Ptr{UInt8}())`.
- **Base-shadowing names are withheld from `export`.** The guard above protects
  the wrapper from itself; this protects whoever `using`s it. llamacpp exports
  `all`, `error`, `stat`, `symlink`; sqlite exports `Expr`, `Module`, `stat`;
  cjson exports `error` — so `using` such a module shadowed the caller's binding,
  and a bare `error("…")` in *their* code became an `UndefVarError`, invisible
  until their first failure path ran. Those names stay **defined and reachable**
  as `Mod.name`; only the `export` is withheld, and the wrapper carries a banner
  naming them. All three generators shared one emission line that had been
  duplicated three ways; it is one `_export_statement` now.
- **The load path is quiet.** `parse_vtables` ran four `println`s during a
  wrapper's `__init__`, i.e. into the *consumer's* stdout — enough to corrupt any
  program whose stdout is data. Now `@debug`; opt back in with
  `JULIA_DEBUG=RepliBuild`.

### Build pipeline + diagnostics hardening (2026-08-05)

Found by putting llama.cpp (195 translation units, a 252 MB DWARF dump) through
the pipeline; none of it is llama.cpp-specific.

- **IR outputs collided on basename** — 195 sources produced 190 `.ll` files.
  `splitext(basename(f))` keyed the output, so `src/llama.cpp` and
  `src/models/llama.cpp` shared one, as did `ggml.c` and `ggml.cpp` in the *same*
  directory. Damage at three layers, all silent: the second compile overwrote the
  first's IR (that TU vanished from the library), `ir_files` still had one entry
  per source so the survivor was handed to the linker twice, and
  `needs_recompile` compared one source's mtime against another's IR — with
  `Threads.@threads` able to tear the same file. One `_ir_output_path` keyed on
  the full source path, shared by both call sites that had their own copy.
- **Linking reported success with unresolvable symbols.** `ld -shared` permits
  them — that is what makes `DT_NEEDED` work — and wrappers dlopen `RTLD_LAZY`,
  so nothing failed at load either; the first sign was a call into the void.
  `_assert_library_resolves` dlopens `RTLD_NOW | RTLD_LOCAL` after the link.
- **Diagnostics.** A 73,570-line arity warning is capped at 20 plus a count, and
  `done: Ns` no longer omits the DWARF phase (`elapsed` was computed before step
  4 and printed after it — on llama.cpp it reported 73 s for a 19-minute build).
- **`parse_module` keeps the module text** when MLIR refuses it. The diagnostic
  is `loc("-":LINE:COL)` against a string that was then discarded.

### Wrapper emission: syntax, undeclared types, portability (2026-08-01 → 08-05)

- **Wrappers are parsed before they are written.** `_assert_wrapper_loadable`
  checks that ccall signatures name declared types, but a malformed *identifier*
  never gets that far — the file dies in the parser and one bad character kills
  the module. DWARF spells a lambda's type as
  `(lambda at ./src/llama-model-loader.cpp:1538:79)`; those parens, slashes and
  colons reached a struct field, and all 98,094 lines of the llama.cpp wrapper
  were a syntax error, discovered only on `include`. `_assert_wrapper_parses`
  now refuses to write it, naming the line and dumping the rejected source. It
  caught three further emitter bugs on its first run: an unsanitized global's
  type, an undeclared bare struct-field type, and union accessors screened
  against the generator's *intent* rather than what it emitted.
- **Undeclared types in signatures degrade to `Ptr{Cvoid}`.**
  `_INTERNAL_TYPE_BLOCKLIST` suppressed the *declaration* of libc internals that
  leak through DWARF, but nothing suppressed their *uses*, so a library with a
  `FILE*` in its surface emitted `ccall(…, (Ptr{Ptr{_IO_FILE}}, …), …)` against a
  type it never bound. `ccall` resolves its type tuple eagerly, so that is an
  `UndefVarError` at *include* — all 1178 of miniaudio's functions dead at once.
- **Anonymous struct/union support** in the C generator. An aggregate DIE with no
  `DW_AT_name` was dropped on export, so the member referencing it typed `Any`,
  failed `_resolve_exact_layout`, and degraded the whole enclosing struct to an
  opaque blob despite DWARF carrying the complete tree. tomlc17's `toml_datum_t`
  and `toml_result_t` came out with no named fields — you could parse a document
  but not read a value back.
- **DWARF DIE attribution fix + arity guard.** `free_opaque` (one parameter) was
  extracted with a phantom second one, emitting a two-argument ccall against a
  one-argument function. `check_param_arity!` compares every extracted signature
  against an independent count off the DIE tree; an over-count is now fatal.
- **Portability guards.** A generated wrapper is portable *source*, but its
  contents are a snapshot of one compilation — struct offsets, blob sizes, enum
  values and sliced IR all come from a specific build. Wrappers now dlsym-check
  their slice declares at load (an unresolved declare does not raise; ORC blocks
  forever on the first call) and warn on a build-ID mismatch.

### Tier-1 slice correctness (2026-07-28)

- **Slice constants are keyed on the mangled symbol, never `julia_name`**, which
  is not injective over it. Two functions shared one constant and the second
  silently won — no redefinition warning under Julia 1.12 binding partitions —
  and since `Base.llvmcall` resolves the constant at *codegen*, the loser broke
  on its first call rather than at wrap time.
- **The slice pre-flight is scoped to the library**, not the process. It resolved
  through `dlsym(RTLD_DEFAULT, …)` after an `RTLD_GLOBAL` dlopen it never closed,
  so wrapping B after A verified B's slices against **A's** exports — symbols
  absent from a consumer's process, so the slice shipped and deadlocked there.
- **Embedding an internal constant requires `unnamed_addr`.** Duplication is
  harmless only if the address is dead; otherwise it is the cJSON divergence
  class rotated from value identity onto address identity. Costs 18 of 208 lua
  functions, which demote to Tier 3.
- **Only slices a call site reads are written** — acceptance is strictly weaker
  than emission, and the gap shipped 19 orphan `.ll` files in the lua wrapper.

### Tier 1 un-parked: per-function `llvmcall` bitcode slicing (2026-07-22 → 07-25)

Tier 1 (`Base.llvmcall`) has been real but effectively parked: LTO embeds the
whole linked module *per call site*, which works for toy 1–2 function modules
and segfaults at library scale on Julia 1.12.6 (box2d3's 730 fns), so production
Hub configs set `[link] enable_lto = false` and everything fell back to `ccall`.
The fix is not a smaller module — it's a *different* module. A slice is
declarations-only: one function body, and every callee and global it reaches
left as a bare `declare`, resolved at JIT time against the `.so` the wrapper
already dlopen'd `RTLD_GLOBAL`. Size stops tracking library size and starts
tracking function size — `lua_gettop` goes 15.8 MB → 2.8 KB, and even
`lua_pcallk`/`luaL_openlibs` come out kilobyte-sized. Three pieces:

- **M1 — static promotion** (`_promote_statics_libllvm`, Builder/Compiler.jl).
  A slice can only bind a symbol that reaches the `.so`'s dynamic symbol table.
  Post-optimization, pre-codegen, every function or global that a slice may bind
  by `declare` but that cannot be dlsym'd is renamed to an exported
  `__rb_<lib>_<name>` with external linkage and default visibility, on the exact
  module that becomes both the `.so` and the slice source — one truth,
  bit-identical. Old→new map lands in `compilation_metadata.json` under
  `promoted_symbols`; `extract_symbols_from_binary` filters `__rb_*` so promoted
  statics never surface as wrappable API. This also kills the cJSON
  static-state divergence class by construction: there is exactly one copy of
  file-local mutable state, and Tier 1 and Tier 3 provably see it (the fixture
  writes through `dlsym` and reads back through the API, and inverse). Default
  on for the in-process C bucket; `[link] promote_statics = false` opts out.
- **M2 — the Slicer** (`src/IRGen/Slicer.jl`). `slice_library(abi_ll; targets,
  cache_dir)` parses the promoted module once and clones per target
  (`LLVMCloneModule`), then strips to declarations (`LLVMFunctionDeleteBody`,
  `LLVMSetInitializer2(gv, NULL)`), embedding internal constants — they have no
  symbol to bind and read-only data has no divergence class. Every slice is
  verified before it's returned. Anything it cannot slice *correctly* comes back
  as a refusal with a reason, never as silently-wrong IR: variadic target,
  `blockaddress` into a body being deleted, alias/ifunc, or an unpromoted
  module. Softer shapes surface as hazard flags for M3's gate
  (`:setjmp_family`, `:varargs_callee`, `:noinline`, `:weak`, `:inline_asm`,
  `:module_asm`). Content-hash cache under `<cache>/slices/`.
- **M3 — emission + dispatch gate** (`_tier1_slice_prepass`, Wrapper/C/
  GeneratorC.jl). New `[wrap.tier1] enable` (default off) runs the Slicer over
  every `is_c_lto_safe` non-varargs candidate, applies the policy
  (`max_slice_kb` = 64 as a tripwire rather than a tuning knob, `allow_setjmp`,
  `exclude`), writes `julia/slices/<mangled>.ll`, and routes accepted functions
  through `Base.llvmcall` on their slice. Non-accepted functions emit `ccall`
  exactly as before, and the wrapper carries a `TIER1_FUNCTIONS` registry of
  what actually got Tier 1.

**Slice symbol pre-flight** (`_tier1_preflight!`). An unresolved slice `declare`
does not raise: ORC prints `JIT session error: Symbols not found: [...]` and
then **blocks forever** on the first call — verified by killing a 180 s run at
the wall. So before any slice reaches disk, the pre-pass dlopens the `.so`
`RTLD_GLOBAL` and `dlsym(RTLD_DEFAULT, …)`-checks every name in
`SliceResult.declares` — the exact lookup ORC will perform. A miss demotes that
one function to `ccall` with a warning naming the symbol; a `.so` that won't
dlopen disables Tier 1 for the whole wrap rather than shipping unverified
slices. `declares` is recorded post-DCE (intrinsics excluded) and round-trips
through the slice cache. Converts "unresolved symbol → JIT deadlock" into a
clean fallback, same discipline as the macro-shim collision guard.

**Promotion covers hidden visibility, not just internal linkage.** The rule that
matters is "cannot reach the dynsym", and `default<O2>` runs no internalize
pass, so an external-linkage symbol marked hidden survives as `define hidden` /
`hidden constant`: invisible to `dlsym`, but read by the Slicer's boundary
policy as "a symbol exists" and bound by `declare`. Both halves of Lua's macro
vocabulary are this shape — `LUAI_FUNC` functions and `LUAI_DDEF` tables — and
the constant case additionally needed the const exemption narrowed to
*internal* constants only (`luaT_typenames_` cost four lua functions their
slices; `luaL_checktype` and `luaL_typeerror` regain Tier 1, while
`lua_typename`/`luaL_tolstring` stay out under the unrelated `Cstring` gate).
The hidden-const class was found by the new pre-flight, at lua scale, doing
exactly its job.

Gated by `test/test_static_promotion.jl` (69) and `test/test_slicer.jl` (127 +
22 at lua scale) over `test/slice_test/`, which now carries both hidden classes
(`st_hidden_scale` fn, `ST_HIDDEN_TABLE` const) as regression locks — reverting
either fix fails them, and reverting the promotion fix without the pre-flight
reproduces the deadlock. Live at scale on Hub lua: 227 slices accepted, 208
functions emitted Tier 1, zero demotions, wrapper exercised clean.

Remaining: M4 (perf characterization at scale — the M0 spike measured 0.4 ns vs
1.13 ns/call clobbered, and 0 ns in a pure loop via LICM through the FFI
boundary, but that is one function, not a library).

## v3.1.0 (2026-07-19)

### Introspection toolkit split into RepliBuildTooling.jl — breaking export change (2026-07-19)

The introspection & analysis subsystem (`Introspect`) moved out of the core into a
new companion package, [RepliBuildTooling.jl](https://github.com/obsidianjulua/RepliBuildTooling.jl),
the opt-in "extra" to the backend. It depends on RepliBuild (never the reverse) and
imports the five backend primitives it needs by name — `extract_symbols_from_binary`,
`extract_dwarf_return_types`, `execute`, `get_tool`, `with_llvm_env` (all already
public). The DAG engine and its renderer stay in core.

- **Core sheds five dependencies**, all Introspect-only: `BenchmarkTools` (a *dead*
  dep — declared, referenced nowhere), `CSV`, `DataFrames`, `Statistics`,
  `InteractiveUtils`. The backend's precompile path no longer pulls DataFrames/CSV.
- **Breaking — exports removed.** RepliBuild no longer exports the `Introspect`
  submodule or its API: `symbols`, `dwarf_info`, `disassemble`, `headers`, the Julia
  `code_*`/`analyze_*` functions, `llvm_ir`/`optimize_ir`/`run_passes`/…,
  `benchmark`/`benchmark_suite`/`track_allocations`, `export_json`/`export_csv`/
  `export_dataset`/`to_dataframe`, and the introspection result types. Migrate
  `RepliBuild.Introspect.foo(...)` → `using RepliBuildTooling; foo(...)`.
- **Docs:** the *Introspection Tools* page moved to the Tooling package; the core
  `architecture`/`internals`/`index` pages now point to it. `benchmarks.md` stays in
  core (it documents the core's zero-copy dialect) with its tool reference updated.
- `VERSION` const realigned to Project.toml (was drifting at `3.0.1` vs `3.0.2`);
  full CI suite green (409 tests) after the split.

### Wrappers are now application-grade: per-library JIT engines, precompilable wrappers (2026-07-19)

Dogfood pass: built a real Julia package (`RepliBuild-Hub/examples/BoxWorld`, a
physics sandbox on the box2d wrapper — Project.toml, ergonomic layer, 15-test
suite) instead of another verify script. Two production blockers surfaced
immediately; both fixed engine-side:

- **Per-library JIT engines.** `GLOBAL_JIT` was a single-engine singleton: the
  first wrapper's `__init__` won, every later `initialize_global_jit` silently
  no-opped, and the second library's ENTIRE Tier 2 died with a misleading
  "Symbol not found / complex C++ type" error (found live composing box2d +
  pugixml). JITManager now keeps one `LibraryEngine` per binary behind the
  existing lock-free symbol cache (thunk names are mangled-derived, unique
  across libraries; lookups search all engines). One library's init failure
  disables Tier 2 for that library only, and the missing-symbol error now
  names every engine searched plus any per-library init failures. Legacy
  single-engine fields mirror the first engine for compatibility. Pinned by
  `test/test_multilib_jit.jl` (devtests §13, 14/14) on the mi_test + vi_test
  wrappers.
- **Generated wrappers precompile inside packages.** Distinct C++ symbols can
  collapse to the same Julia name+signature (destructor D1/D2 pairs; overloads
  whose params all map to `::Any`). At script include() that's a last-wins
  warning — under package precompilation method overwriting is a hard ERROR,
  so no C++ wrapper could be vendored into a precompiled package. Both
  generators now deduplicate emitted definitions by dispatch signature,
  keeping the LAST (include()-identical semantics): box2d dropped 16
  duplicates, tinyxml2 59, pugixml 108 — all suites still green (13/13,
  11/11, 15/15). Also moved the C++ generator's stdout-unbuffering from
  module top level into `__init__` (top-level side effects run at PRECOMPILE
  time and never re-run at load in a precompiled package; the C generator
  already did this correctly).

New documentation page **"Using a Wrapper in Your Package"** (docs/src/
using-wrappers.md, wired into the manual) covering the vendoring layout, the
two-layer discipline (ABI layer vs ergonomic layer), precompilation rules,
the JIT lifecycle from a consumer's perspective, multi-library composition,
by-value handle conventions, finalizer-warming discipline, and the C++-isms
an app layer must encapsulate (ctor-only classes, header-inline defaults,
abstract-shape vtables) — all demonstrated live by BoxWorld.

Recorded findings for the ledger (not fixed here): ctor thunks are not
emitted for arg-taking constructors of factory-less classes (`b2World` needs
the raw-ccall pattern); header-inline ctors/accessors are unreachable by any
binding (defaults must be replicated at DWARF offsets — inherent, now
documented); planting compiler vtables for abstract-shape instances is
expert-level and could grow a generated helper.

## v3.0.1 (2026-07-18)

The inheritance-ABI + Tier-2-correctness release. The full C++ inheritance arc, shipped whole: non-virtual multiple inheritance, the vcall producer (overrides dispatch through the vtable), and virtual inheritance (dynamic vbase upcasts) — closing ledger entries open since 2026-05-29. On top of it, the Tier-2 ABI-correctness pass driven by pugixml: SysV small-struct register classification for `try_call`/`ffe_call` (≤16-byte struct returns/args no longer force sret), nested-packed-struct type inlining (the pugixml load segfault), and a JIT pre-flight type guard so an incompatible type degrades to "Tier 2 disabled" instead of killing the host process. Around those: the Tier-2 virtual-method thunk-routing fix (virtual instance methods were never callable through `invoke()`), the discover(force) user-intent TOML preservation fix (root cause of six weeks of silent stl_test red), the nested-type member-attribution parser fix (found wrapping box2d), the ingest honesty pass from the first outside-user report (issue #4), and the tinyxml2-era Tier-2 dispatch fixes. First C++ Hub package exercising all of it: box2d 2.4.1.

Verification state at release: CI 404, producers 26/26, invariants 10/10, templates 87/87, struct-ABI traces 15/15, mi_test 38/38, vi_test 33/33, stl_test 28/28, stress+c_test green, Hub tinyxml2 11/11, pugixml 13/13, box2d 15/15.

Sections below are ordered latest-first within the release.


### pugixml load segfault: nested packed structs, JIT pre-flight guard, SysV small-struct ABI (2026-07-18)

The pugixml wrapper SIGSEGV'd at module load inside `translateModuleToLLVMIR`
(`PtrLikeTypeInterface::getMemorySpace`) — uncatchable, killing the whole
process (docs/pugixml-jit-init-segfault.md). Root-caused via library-free
fixture (two hand-written structs + one thunk) and fixed three ways:

- **Nested packed structs (the crash).** A padding-free struct nested by value
  in a padded struct put its `!jlcs.c_struct` alias inside an `!llvm.struct`
  body; the type converter never rewrites inside already-legal LLVM struct
  bodies, so the foreign type survived lowering into LLVM-IR translation.
  `StructGen` now inlines the byte-identical `!llvm.struct<packed (…)>` literal
  instead of the alias (recursively) whenever a member's target is
  c_struct-classified. Not any of the suspected categories (function pointers,
  std:: members, system structs) — pure struct nesting, and tinyxml2 only
  dodged it by never nesting a padding-free struct by value.
- **JIT pre-flight guard.** `jlcs_create_jit_with_libs` walks all op, block-arg,
  and `llvm.func` signature types and refuses (null → catchable Julia error →
  "Tier 2 disabled" degradation) any module with a type failing
  `mlir::LLVM::isCompatibleType` — naming the type and op. A bad type can
  never again take down the host process at `ExecutionEngine::create`.
- **SysV small-struct return/arg ABI (found by the now-runnable pugixml
  verifier).** `try_call`/`ffe_call` lowering forced sret for EVERY packed
  struct return, but ≤16-byte aligned structs are register-class on x86-64 —
  native `pugi::xml_node::first_child()` returns `{void*}` in RAX, so the sret
  call shifted `this` into the sret slot and the thunk returned stack garbage
  (JIT code addresses as node handles). New `classifySysVStruct` classifies
  MEMORY (>16B or genuinely misaligned fields → sret/pointer, unchanged) vs
  register class (coerce one scalar per eightbyte: INTEGER→i64, SSE→f64,
  clang-style), applied to returns AND by-value args in both lowerings. This
  also fixes latent per-element splitting mismatches (`{int,int}` shares RAX;
  `{float,float}` shares XMM0) that LLVM's naive struct lowering gets wrong.

Pinned by `test/test_struct_abi.jl` (devtests §12): nesting fixture lowers+JITs,
guard throws catchably on the exact pre-fix crash IR, and the ABI matrix runs
against a REAL clang++-compiled callee (self-JIT'd callees share the JIT's own
convention and can't catch a mismatch). Verification: pugixml loads, its Hub
test passes 13/13 (first wrapper exercising 8-byte by-value handle returns);
regression sweep green — tinyxml2 11/11, templates 87/87, invariants 10/10,
producers 26/26, mi 38/38, vi 33/33, CI runtests full pass.

Still unbuilt (ledger): MEMORY-class by-value args keep their pre-existing
conventions (packed → raw pointer, non-packed → direct struct), neither of
which matches native stack-copy passing for trivially-copyable >16B structs —
no wrapper exercises that path yet.

### Ingest honesty pass (issue #4)

First outside-user report ([#4](https://github.com/obsidianjulua/RepliBuild.jl/issues/4)): ingesting a CMake-built C++ `.so` produced wrappers that don't parse, with no signal that C++ ingest was never supported. Ingest is a fallback, not a flagship feature — the docs oversold it and nothing guarded it. Now:

- **Guards at both entry paths**: `ingest(language=:cpp)` and `ingest_library` on a `language="cpp"` config warn loudly that the C++ API surface of an ingested binary is unsupported (dialect thunks require the source build; at best the `extern "C"` surface works) and point at the C-variant / source-build alternatives. `ingest` also validates `language ∈ (:c, :cpp)` and notes experimental status for `:c`.
- **README/docs de-oversell**: the pitch no longer claims "point it at a `.so`" as co-equal with the source build; the ingest section is labeled **EXPERIMENTAL, C only** with the real constraints (Tier-3 only, best-effort extraction, C++ unsupported); `why-replibuild.md`/`architecture.md` aligned; CLAUDE.md carries the status doctrine so docs edits don't re-inflate the claims.
- **New README section: "Using the generated wrapper"** — the `include` + `using .Module` pattern the issue explicitly asked for (docs described building wrappers but not using them).
- Fixed a latent v3.0.1 miss: `runtests.jl` hardcoded `VERSION == v"3.0.0"`; it now compares against `pkgversion(RepliBuild)` so version bumps can't desync it.

Remaining from #4 (not this pass): the C++ generator can still emit invalid identifiers for reference-carrying template type names (`&` survives `_sanitize_cpp_type_name`), and generation has no parse-gate — a generated wrapper should never fail `include` without a generation-time warning. Tracked for a generator-hardening pass.


### Nested-type member attribution fix (found wrapping box2d for the Hub)

Members declared AFTER a nested type definition inside a class silently vanished from extracted metadata: clang emits a nested enum/struct DIE between the member DIEs (at first reference — `Type m_type; float m_radius;` puts the `Type` enum's DIE, enumerators, and null terminator between the two members), and `Compiler.jl`'s flat `current_struct_context` flipped to the nested type for its children and never restored the enclosing class. Every subsequent member attributed to the enum and was dropped — box2d's `b2Shape::m_radius` was the live casualty.

Fix: **depth-aware parent attribution**. readelf DIE headers carry the tree depth (`<2><2331>:`), previously discarded; the parser now maintains a depth-indexed context map (`context_by_depth`), and member/enumerator/inheritance/template DIEs at depth d attribute to the type last seen at depth d−1, with `current_struct_context` as fallback. Reproduced library-free per the hub-wrap guard (`NestedEnumHolder` in the mi_test fixture — the enum-typed member must come FIRST or clang hoists the nested DIE past all members and the bug doesn't trigger), fixed, and pinned by mi_test verify (38/38). Generalization confirmed: c_test 70/70, vi_test 33/33, box2d re-wrap recovers `m_radius`.


### Virtual inheritance: dynamic vbase upcasts, diamond-proven

The last unbuilt piece of the inheritance ABI. A virtual base's offset is **not static** — a standalone `Left` and a `Left`-inside-`Diamond` place the shared `VBase` at different offsets, and only the object's vtable knows which (the vbase-offset entry below the vtable address point). The MI-era policy of detect-and-reject-loudly is replaced with actual support:

- **Extraction (both parsers):** the `DW_AT_data_member_location` on a virtual inheritance edge is a DWARF *expression* (`DW_OP_dup, deref, constu N, minus, deref, plus` = "this + \*(vptr − N)"), not a constant. Both parsers now parse it into `vbase_vtable_offset = −N` (readelf and llvm-dwarfdump renderings pinned empirically). Fixed in passing: the readelf constant-regex previously matched the "7" of "`7 byte block:`" on virtual edges — a bogus static offset 7, latent because consumers gated on `virtual=true`.
- **Wrapper: dynamic `<Derived>_as_<VBase>` upcasts.** The helper reads the object's vptr and the vbase-offset entry at runtime: `p + *(vptr + vboff)`. The *same* helper is correct for every dynamic type — the vi_test canary shows `Left_as_VBase` resolving +16 on a standalone `Left` and +32 on a Diamond-backed one. Transitive virtual bases compose (Diamond has no direct `VBase` edge in DWARF; `Diamond_as_VBase` static-adjusts to `Left` then goes dynamic). Non-virtual MI upcasts unchanged.
- **`jlcs.type_info` gained a virtual-base table**: `vbaseNames`/`vbaseVtableOffsets` paired ArrayAttrs carrying the vtable-relative coordinate (virtual bases never appear in the static `baseNames`/`baseOffsets` table); verifier extended for the new pair. The old omit-everything-and-warn policy is gone.
- **No dialect lowering changes needed** — the ledger predicted "vtable-resident offset reads in the dialect lowering", but the class-local-coordinates + caller-side-upcast architecture means the dynamic read lives in the Julia wrapper and everything else composes: the vcall producer needed **zero changes** for vbase-declared methods, overrides of vbase methods re-home into the derived primary vtable exactly like regular MI (empirically: `Diamond::tag` slot 3), and complete-object ctors/dtors (C1/D1, already preferred by the RAII resolver) handle vbase construction.
- Layout flattening still (correctly, by design) skips vbase members — their offsets are dynamic; access goes through the upcast + the base's own accessors.

Proven live in `test/vi_test/` (diamond fixture, wired into devtests, **33/33**): the 16-vs-32 same-helper canary; all three views of a Diamond (`Diamond_as_VBase`, `Left_as_VBase`, `Right_as_VBase∘Diamond_as_Right`) resolve the ONE shared `VBase` (single-copy proof, including tail-padding reuse `d@28` extracted faithfully); `VBase_tag` through the vbase vtable dispatches `Diamond`'s override (1007) from a pointer whose caller has zero derived-type knowledge, while the standalone `Left` gets `VBase::tag` (7) from the identical call; polymorphic delete through `Left*` destroys the Diamond. Dialect-level vbase table parse/verify in templates §8d (**87/87**).

Regression state: CI 404, producers 26/26, invariants 10/10, templates 87/87, mi_test 31/31, vi_test 33/33, stl_test 28/28.

### stl_test regression fixed: discover(force) no longer destroys user-intent TOML config

The KNOWN RED flagged below is closed, and the wrapper generator was never broken. Root cause: `discover(force=true)` regenerates `replibuild.toml` from scratch and `generate_config` emits the user-intent keys empty — so forced re-discovery silently destroyed `[types].templates`, killing the instantiation stub → DWARF → STL wrapper chain. Broken since `4117a8e` (2026-06-02) made devtests always force-rediscover fixtures. Full narrative: `docs/updates/2026-07-17-stl-test-regression.md`.

- **`discover` now preserves user-intent TOML keys across forced re-discovery** (`Discovery.PRESERVED_TOML_KEYS`: `[types].templates`/`template_headers`, `[wrap].varargs`/`macros`/`shim_headers`/`cstring_owned`). Regenerated non-empty values win; empty/absent slots get the preserved value; a `preserved: …` line reports what carried over. This was a systemic footgun for any user project with hand-curated wrap config, not just fixtures.
- **devtests seeds curated fixture config** (`CURATED_FIXTURE_CONFIG`, applied between discover and build) so fresh clones — where the gitignored TOML doesn't exist yet — are deterministic.
- New CI guard `test/test_toml_preservation.jl` (21 tests). Live proof both ways: fresh-clone seeding path and preservation-only re-discovery path each yield all 7 `create_std_*` factories and stl_test verify **28/28**. CI total now **404**.

### vcall producer: virtual methods dispatch through the vtable (overrides honored)

`jlcs.vcall` moves from "exercised only by hand-written test IR" to **emitted by the codegen pipeline** — the first production consumer of the op, built directly on the MI groundwork below. Wrapper Tier-2 calls to virtual instance methods previously direct-called the mangled symbol (`p->Base2::get_b()` static semantics — overrides ignored); their thunks now read the vptr, index the slot, and call indirectly, so **a base-class wrapper invoked on a derived object reaches the override**.

- **Dispatch coordinates are class-local, universally** — an empirically pinned Itanium/DWARF fact (dwarfdump, clang, 2026-07-17): `DW_AT_vtable_elem_location` is the slot in the *declaring* class's own primary vtable, and Itanium re-homes overrides of non-primary-base methods into the derived class's primary vtable (fixture: `Base2::get_b` slot 2 in Base2; the `Derived::get_b` override slot 3 in *Derived's* vtable, after the dtor pair and `get_a`). Since every `ClassName_method` wrapper takes a ClassName-relative `this`, the producer always emits `vtable_offset = vptr offset (0), this_offset = 0, slot = method's own slot` — MI correctness comes from the caller-side upcast (`as_Base2`) plus the this-adjusting thunks the compiler already planted in secondary vtables. No introducer-walk needed.
- **`jlcs.vcall` gained `may_throw`** (UnitAttr, absent = plain indirect call, pre-existing lowering unchanged). When present, the lowering emits an **indirect invoke + landing pad** with the same sentinel-continue EH model as `try_call` (`__cxa_begin_catch` → `jlcs_catch_current_exception` → `__cxa_end_catch`, zero-sentinel result), personality installed on the parent function. Built with the ODS indirect-invoke builder (null callee + `var_callee_type`), steering clear of the historical hand-rolled-operandSegmentSizes SIGSEGV. The producer sets it under the same `may_throw && !noexcept` rule as `try_call`.
- **Producer gating**: `generate_jlcs_ir` builds a mangled-symbol → (class, slot, vptr-offset) table from DWARF vtable info; FunctionGen swaps the direct call for `jlcs.vcall` only for instance methods with scalar/pointer signatures (the vcall lowering does no sret/packed coercion — struct-shaped signatures keep the direct-call path). **Destructors are excluded by design**: `Managed` finalizers and the scope-RAII producer require exact-class destructor calls, not dynamic dispatch.
- Proven live in `test/mi_test/` (31/31): `Base2_get_b` on an upcast `Derived` returns the override's value (1222, not the base's 222) through the secondary vtable's adjusting thunk; a C++-side `Base2*`-that-is-really-`Derived` (caller has zero derived-type knowledge) dispatches the override; mutation through a non-overridden virtual composes with override reads; non-virtual methods keep static semantics (the wrong-`this` canary pins unchanged); polymorphic deletion through the base pointer works. Dialect-level EH path JIT-proven in templates §8c (84/84): emitted LLVM carries `invoke`/`landingpad`/personality and dispatch+`this`-adjustment behave identically under it.

**Known issue (pre-existing, unrelated — bisected to before this work):** `test/stl_test` wrapper generation silently omits the STL factory section (`create_std_vector_int` etc. missing; verify errors 6/8). Reproduces on a pristine tree at the previous commit. **Resolved the same day — see the stl_test section above** (config destruction in discover, not codegen).

Regression state: CI 383, producers 26/26, invariants 10/10, templates **84/84**, mi_test **31/31**, stress_test + c_test green (stl_test red for the pre-existing reason above).

### Multiple-inheritance ABI: dialect, extraction, layout, upcasts

Non-virtual multiple inheritance is now modeled end-to-end ("Not Yet Built" ledger entry since 2026-05-29; closed 2026-07-17). Virtual inheritance remains unbuilt and is now *detected and rejected loudly* at every consumer instead of silently mis-handled.

**Dialect** (rebuilt against MLIR 22.1.6):

- **`jlcs.vcall` gained `this_offset`** (I64, default 0 — all pre-MI IR keeps its exact semantics). The lowering GEPs `args[0]` by `this_offset` bytes before the indirect call, so a method dispatched through a non-primary base receives a pointer to *its base subobject*, matching how secondary-vtable entries are compiled under Itanium. `vtable_offset` stays relative to the original pointer (reads the secondary vptr as before); both offsets equal the base subobject offset in the standard secondary-base case. Previously the vtable was read from the right offset but `this` was passed unadjusted — every secondary-base virtual call read the primary base's data.
- **`jlcs.type_info` gained a base table**: paired `baseNames`/`baseOffsets` ArrayAttrs (attr-dict, default empty — old IR parses unchanged) recording each base class and its static subobject offset. `superType` stays as the primary base for single-inheritance consumers.
- **Verifiers on both** (`VirtualCallOp::verify`, `TypeInfoOp::verify`): vcall requires the object-pointer operand; type_info rejects base-table arity mismatches and wrong element kinds — same idiom as the scope/marshal_arg verifiers.
- JIT-executed proof in `test_mlir_templates.jl` §8b against a hand-rolled two-vtable MI object: secondary-base dispatch with `this_offset = 16` reads the secondary base's member (222), the pinned pre-fix semantics (`this_offset` omitted) observably read the primary base's member (111) through the same vtable slot, and mutation through the secondary base lands at the right byte. Templates suite 56 → **71/71**.

**Extraction** (both DWARF parsers): `DW_AT_data_member_location` on `DW_TAG_inheritance` — the base subobject offset, previously dropped by both parsers (every base silently recorded at offset 0) — and `DW_AT_virtuality` (virtual-base flag) are now captured. `Compiler.jl`'s metadata gains `"offset"`/`"virtual"` per base and sorts `base_classes` by subobject offset (Dict iteration order was nondeterministic); `DWARFParser.ClassInfo` gains parallel `base_offsets`/`virtual_bases` vectors (positional 6-arg construction still works via a compat constructor).

**Consumers**:

- `JLCSIRGenerator.generate_type_info_ir` emits the base table (omitted with a loud warning for virtual-inheritance classes — recording a static offset for a vtable-resident quantity would be a lie).
- `GeneratorCpp.flatten_struct_members` rebases base-class member offsets by the base subobject offset when flattening derived layouts (previously base-relative offsets were used raw — correct only while every base sat at offset 0), including DWARF4 `data_bit_offset` bitfield rebasing; same-named members from different bases get deterministic collision renames. Virtual bases are skipped loudly.
- **`<Derived>_as_<Base>` upcast helpers** are emitted for every class with a non-zero-offset base: `Derived_as_Base2(obj)` applies the static Itanium adjustment (accepts raw `Ptr` or any `.handle` wrapper). This is what makes calling a secondary base's methods on a derived object *correct* — the method wrappers take base-relative `this`.

**Tier-2 virtual-method thunk routing fix** (pre-existing, exposed by the MI fixture — the first test to ever call a *virtual* instance method through the wrapper's Tier-2 path): generated wrappers dispatch via `JITManager.invoke("_mlir_ciface_<mangled>_thunk", …)`, but `generate_jlcs_ir` routed virtual methods to the legacy vmethod-IR pass, which emits `thunk_<mangled>` direct-call wrappers **nothing ever looks up** — every virtual instance method was uncallable through `invoke()` (`Symbol not found`). Virtual methods the wrapper's thunk manifest declares are now routed through the FunctionGen ciface pass like any other method. Note the resulting call has statically-named-class semantics (`p->Base2::get_b()`); override-honoring dispatch through the vtable is the future vcall producer, now unblocked by `this_offset` + the base table.

Live-verified end-to-end on a compiler-laid-out two-base fixture (`test/mi_test/`, wired into devtests): metadata carries `Base1@0`/`Base2@16` (with `extra` at `0x1c` — Itanium tail-padding reuse, extracted correctly), type_info base table parses verifier-clean, `Derived_as_Base2` == +16, and live calls prove the layout — primary-base call unadjusted, secondary-base non-virtual `double_b` returns 444 via the upcast (and the pinned wrong-`this` call observably returns 2×`a`), virtual `get_b` through the newly-routed Tier-2 thunk returns 222, mutation through `Base2` observed via `Derived::get_sum`. `verify.jl` 27/27.

Regression state: CI 383, producers 26/26, invariants 10/10, templates 71/71, mi_test 27/27, stress_test + c_test full pipeline green.

### Tier-2 C++ dispatch fixes (found live rebuilding tinyxml2)

Clean-rebuilding the tinyxml2 Hub package end-to-end surfaced three real defects in the C++ JIT dispatch path — none had a covering test because no Tier-2 wrapper had been driven through a full construct→call→destruct cycle with these argument/return shapes:

- **Enum-by-value returns crashed** (`XMLDocument::Parse → XMLError`). The C++ generator resolved an enum return (DWARF key `__enum__X`) to a single-member struct `!llvm.struct<"X",(i32)>`; MLIR's `emit_c_interface` then used the **sret** convention (`void ciface(T* sret, void** args)`), but the Julia side calls the `@enum` back as a scalar (`T ciface(void** args)`) — the args pointer landed in the sret slot and the call dereferenced garbage. `ir_gen/FunctionGen.jl` now lowers enum returns to their bare underlying integer (returns by value, ABI-identical to the `@enum`); `GeneratorCpp.jl`'s `Any`-return resolution checks `__enum__` before the struct branches so the concrete `@enum` type reaches `invoke`.
- **`String` arguments to JIT thunks crashed.** `JITManager.invoke` packed `Ref(::String)`, handing the callee a pointer to the String *object* rather than its bytes — segfault on first dereference inside the C++ function. Now marshals `Ref(pointer(str))` with the String GC-preserved across the call, for both the value- and void-returning `invoke` variants.
- **Unresolved `Any` return now fails loudly.** `_invoke_call(::Type{Any}, …)` used to take the struct-sret path with `Ref{Any}()` (an undefined reference the JIT scribbled into → `UndefRefError`/corruption). It now `error()`s with the actual cause (return type unmapped — stale wrapper or missing DWARF mapping).

Live-verified: tinyxml2 rebuilt from cleared caches, then construct (placement `XMLDocument` ctor thunk) → `Parse` (`XML_SUCCESS`) → `FirstChildElement`/`GetText` (reads "42", "hello hub") → `SetText` → non-deleting dtor thunk, all through Tier-2 MLIR dispatch. Regression state unchanged: dialect templates 56/56, invariants 10/10, producers 26/26, CI 383.


### DWARF-driven producers for the JLCS RAII and strided-array ops

The two producer-less op families in the dialect ("Not Yet Built" ledger since 2026-05-29) now have production emitters, verified executing through the real MLIR JIT (`test/test_jlcs_producers.jl`, 26/26, devtests §11):

**Scope-RAII producer** (`ir_gen/FunctionGen.jl` + `JLCSIRGenerator._collect_class_raii`). Per-class destructor (D1-preferred) and copy-constructor (C1-preferred, `(const T&)` signature) symbols are resolved from DWARF metadata once per wrap. Three effects:

- `jlcs.type_info` now carries the resolved `destructorName` (was always `""`).
- **By-value params of classes with an emitted destructor are non-trivial for the purposes of calls under the Itanium C++ ABI — the callee expects a POINTER to a caller-owned temporary.** The thunk generator previously passed such classes as raw bits per SysV classification, a silent miscompile. Thunks now alloca a temporary, copy-construct it inside a `jlcs.scope` (`jlcs.ctor_call` with the copy ctor when resolvable, byte-copy when the copy is trivial), pass its address, and destruct it at scope exit — reverse order for multiple params, arity co-generated with the managed-pointer list. `try_call`'s sentinel-continue EH model means the normal path is the only path, so scope-exit destructor coverage is total.
- Symbol presence is the gate (a trivial destructor is never emitted as a symbol), so trivially-copyable classes keep their existing by-value path unchanged.

**Array-view producer** (`ir_gen/ArrayViewGen.jl`, new). Every fixed-size primitive array struct member (`double vals[4]`, `int32_t ids[16]`, …) gets a zero-copy get/set thunk pair that materializes an `ArrayView` descriptor (data/dims/strides/rank) over the member in place and accesses elements through `jlcs.load_array_element`/`store_array_element`. Rank 1 today; the descriptor already carries what bounds-checking and rank ≥ 2 need. This is also the first time these ops have ever been *executed* (the invariants suite only proved parse+lower) — they run correctly.

Remaining wiring (ledger): `GeneratorCpp.jl` does not yet emit user-facing Julia accessors calling the array thunks; multi-dimensional members are skipped.

Also recorded: `StructGen.is_struct_packed` classifies any padding-free struct as "packed" (`sum(sizes) == byte_size`), sending aligned no-padding structs down the `marshal_arg` path unnecessarily — benign but wasteful; the scope-RAII gate deliberately takes precedence over it.

### First JLCS op verifiers: `jlcs.scope` + `jlcs.marshal_arg`

The dialect's two known lowering segfaults are now parse-time diagnostics. `ScopeOp::verify()` rejects managed_ptrs/destructors arity mismatches (the old crash: `emitDestructors` indexed `managedPtrs` by destructor position and walked off the end) and non-symbol destructor entries; `MarshalArgOp::verify()` rejects memberTypes/juliaOffsets arity mismatches (the field loop indexed offsets by member position) and wrong element kinds. Malformed IR that used to SIGSEGV inside `translateModuleToLLVMIR`-adjacent lowering now fails `parse_module` with a real diagnostic.

`test/test_jlcs_invariants.jl` is fully green for the first time — **10/10, zero `@test_broken`** (both A2/B2 probes flipped from "expected crash" to asserting `:parse` rejection). The producer suite doubles as the positive control: production-emitted scope/marshal IR passes verification (26/26). Dialect rebuilt against system MLIR 22.1.6, templates 56/56.

The remaining ops (`vcall`, `type_info`, `ffe_call`/`try_call`, field/array ops) stay verifier-less — no known crash paths — tracked in the ledger to grow verifiers as their producers mature.

Regression state: dialect templates 56/56, invariants **10/10 (no test_broken)**, CI suite 383, producers 26/26.

## v3.0.0

C-generator audit release (2026-07-10). Ownership and ABI edges of the ergonomic layer are closed: the struct-by-value convenience footgun is removed, variadic calls are formally correct on x86-64 SysV, `char*` returns get one ownership-aware policy (`[wrap.cstring_owned]`) plus raw `_ptr` variants, macro shims survive `-fvisibility=hidden`, two silent-corruption edges are trapped or fixed (misaligned ≤16B blob params, bitfield tail overrun), and the registry build cache stops serving wrappers generated by outdated codegen.

**Registry note:** the last version registered in General is **v2.5.7**. Internal versions v2.5.8 and v2.5.9 were never registered — their changes ship as part of this release (their sections below stand as historical detail). The version jumps to 3.0.0 because the generated-wrapper API changes shape (see below), and semver puts breaking changes in the major number.

### Breaking changes since v2.5.7 (last registered version)

Generated-wrapper API — wrappers regenerate automatically on the next `use()`/`wrap()` (the cache is fingerprinted, below), but *calling code* may need updates:

- **Struct-by-value convenience overloads are gone** (C and C++). `f(unsafe_load(p))` patterns must pass the pointer or a `Ref` instead — every such call was UB-adjacent (the callee saw a pointer to a temporary copy; frees and stores corrupted memory, crash-proven on `cJSON_Delete`).
- **`char*` returns are `Union{String,Nothing}` instead of `String`-or-throw.** NULL now returns `nothing`; code relying on the old "returned NULL pointer" error must check `=== nothing`. Every such function gains an exported raw `<name>_ptr` variant (additive).
- **Nested-member structs resolve to named fields** instead of one `_data::NTuple` blob (v2.5.8). Code reaching into `x._data` on affected types must switch to the named fields; documented accessors are unaffected.
- **Multi-byte bitfield accessors changed shape:** getters return the smallest `UInt` covering `bit_size` (previously sized by offset+width span); setters accept negative integers with wrapping semantics (previously `InexactError`).
- **Globals with unresolvable DWARF types no longer get a value getter** — only `<name>_ptr()::Ptr{Cvoid}`. The old getter read memory as a boxed Julia pointer (garbage or crash).
- **Misaligned ≤16B opaque-blob params by value now throw an ABI trap** instead of silently corrupting arguments (float-bearing blobs already trapped since v2.5.8; this closes the all-integer packed case).

Programmatic API:

- **`WrapConfig` gained a positional field** `cstring_owned::Dict{String,String}` (before `dag`) — code constructing `WrapConfig` positionally must add it. TOML users are unaffected.

Behavioral (same signatures, corrected semantics):

- Vararg wrappers lower as true variadic calls (`@ccall` semicolon form) — float varargs no longer depend on leftover AL.
- `use()` cache keys include the generator fingerprint: every registered package rebuilds once after upgrading RepliBuild.
- Wrappers resolve their library sibling-first via `@__DIR__`; macro shims are pinned to default visibility (so `[wrap.macros]` works under `-fvisibility=hidden`).

New TOML surface (additive): `[wrap.cstring_owned]` — `func = "free_symbol"` declares a malloc'd `char*` return; the wrapper frees it through that symbol after copying.

### Struct-by-value convenience overloads removed (double-free footgun)

Both wrapper generators (`GeneratorC.jl`, `GeneratorCpp.jl` — the block was duplicated verbatim) emitted a "convenience" overload for every function with a `Ptr{Struct}` parameter: `f(x::MyStruct)` taking the struct **by value** and passing `Ref(local copy)` to the ccall. For any C function that frees, mutates-and-retains, or stores that pointer, this is undefined behavior — the callee receives a pointer into a temporary Julia-owned copy. Crash-proven 2026-07-10: the generated `cJSON_Delete(item::cJSON)` overload aborts with glibc `double free or corruption`, because `cJSON_Delete` calls `free()` on Julia GC memory. Retaining functions (e.g. `cJSON_AddItemToArray`, which stores the pointer into the array) corrupt silently instead of aborting — worse.

The overload class is **removed entirely** rather than gated: ownership is not recoverable from DWARF, so any `delete/free/destroy`-style name blocklist is guaranteed incomplete (a store-the-pointer function like `AddItemToArray` matches no such pattern). Ergonomics loss is negligible — the base wrapper's pointer params are `::Any` and already accept `Ref(x)`, pointers, and (for arrays) `Vector`s directly.

Survivors, pinned by test: the `Vector{T}` convenience overload for input-array params (`Ptr{Cdouble}` etc. under `GC.@preserve`) stays, and its `Cstring` returns follow the base wrapper's policy (see *Cstring return policy* below) instead of leaking a raw `Cstring`. The surviving path also gains the base wrapper's struct-return sentinel guard on the C++ side (previously it emitted the boxed-`Any` ccall return, a latent segfault).

Guarded by `test/test_convenience_overloads.jl` + `test/convenience_overload_test/` (devtests §10): a library-free fixture with a `free()`-taking `grip_free(Grip*)` traces compile → DWARF → wrap in a subprocess and asserts no by-value method exists, a by-value call refuses loudly (MethodError, never reaching `free()`), the pointer lifecycle round-trips, and the Vector path survives with `String`-aligned `Cstring` returns.

**Upgrade note:** code that called the by-value overloads (`f(unsafe_load(p))` patterns) must pass the pointer or a `Ref` instead — every such call site was UB-adjacent even when it appeared to work (the callee saw a copy, so mutations were dropped and stores/frees corrupted).

### Variadic call ABI (`generate_vararg_wrappers`)

Typed vararg overloads — and the fixed-args base wrapper — called variadic C functions through a **flat non-variadic ccall type tuple**; the code comment claimed a "Vararg marker for proper ABI" but none was ever emitted. On x86-64 SysV the callee's `va_start` prologue gates its XMM0–7 spill on AL, and only a variadic call site sets AL: int/pointer varargs worked de facto, but **float varargs (`sqlite3_mprintf_Cdouble`, gzprintf's Cdouble overload, …) only read correctly when leftover AL happened to be nonzero**. A live probe passing proved nothing — the failure is nondeterministic by construction.

All vararg wrappers now emit the `@ccall` semicolon form, `@ccall LIBRARY_PATH.var"sym"(fixed::T…; va_1::Cdouble)::Ret`, which lowers to a true variadic foreigncall: the callee is declared variadic in LLVM IR (`call i32 (ptr, ...)`) and the backend emits the AL setup (`movb $N, %al` observed in `code_native`). The base wrapper keeps a trailing `;` with zero varargs — the callee is still variadic, so AL must be set (to the count of vector registers used by the *fixed* args) even then. `var"…"` hardens the symbol position against keyword-shaped C identifiers. Per-arg vararg types are preserved (the old tuple form couldn't have expressed heterogeneous varargs correctly anyway).

Fixed in passing: the `"varargs..."` placeholder skip compared the *sanitized* param name (which mangles to `varargs_`), so it never matched — metadata paths that include the placeholder leaked `varargs_::` into generated signatures. The raw name is checked now.

Verified: a variadic C fixture built through the full pipeline — generated overload IR shows the variadic callee, float/int/zero-vararg calls return correct values. Regression: `test/test_varargs_emission.jl` pins the emission strings and the macro-expansion property (`Expr(:cconv, _, nreq)` with `nreq > 0` = variadic; the bug was `nreq == 0`).

### Registry build cache: generator fingerprint in `hash_config`

`hash_config` hashed TOML + sources + headers + *project* git HEAD but **not RepliBuild itself**, so `use()` served cached wrappers from old generators forever — observed: May 3/May 31 pre-v2.5.8 wrappers still live in `~/.replibuild/builds`, one of which crash-loads because Tier-1 `llvmcall` was baked in at lua scale. The hash now mixes in `_generator_fingerprint()` — package version **plus RepliBuild's own git HEAD** (dev checkouts move per commit; the version alone demonstrably doesn't get bumped every release). Every pre-existing cache entry misses once and rebuilds with current codegen on next `use()`; stale entries are orphaned, not deleted.

### Wrapper library resolution: sibling-first, baked path as fallback

Generated wrappers baked an absolute `LIBRARY_PATH` (for registry builds: into the shared `~/.replibuild/registry/julia/`), which the next build of any package overwrites — stranding every cached wrapper in `~/.replibuild/builds/<hash>/` even though each per-hash dir holds its own `.so` copy that the wrapper ignored. C and C++ wrappers now resolve the library **next to their own file first** (`joinpath(@__DIR__, basename(baked))`) and fall back to the baked absolute path; same for `THUNKS_LIBRARY_PATH`. Verified: renaming the baked `.so` away and reloading the cached wrapper works — the sibling copy wins.

### Cstring return policy: NULL → `nothing`, `[wrap.cstring_owned]`, raw `_ptr` variants

`char*`-returning wrappers previously converted via `unsafe_string` and **threw on NULL** — but a NULL `char*` is a value in C APIs (`cJSON_GetErrorPtr` before any error, `sqlite3_column_text` on a NULL column), not an exception. Worse, for functions returning **malloc'd** buffers (`cJSON_Print`, `sqlite3_mprintf`) the pointer was dropped after copying — an unfixable leak per call, since the caller never saw the pointer to free it.

The policy is now defined once (`_cstring_policy_lines` in `Wrapper/Utils.jl`) and spliced into **every** emission site — base wrappers, the surviving array-convenience overloads, vararg base + typed overloads, C and C++ generators — so it cannot drift between sites again:

- Return type is `Union{String,Nothing}`: NULL → `nothing`, else a copied `String`.
- **Ownership is declared in the TOML**, not guessed: `[wrap.cstring_owned]` maps a function to its library's deallocator (`cJSON_Print = "cJSON_free"`), and the wrapper frees the C buffer through that symbol after copying. Ownership of a returned `char*` is not recoverable from DWARF — same law as the convenience-overload removal, resolved the same way the varargs gap is: per-library facts live in the TOML.
- Every `Cstring`-returning function also gets an exported **`<name>_ptr` raw variant** (same args, returns the `Cstring` unchanged, no copy, no NULL check, never freed) for lifetime-sensitive callers and owned returns without a declared deallocator.

**Upgrade note:** call sites relying on the NULL throw must check `=== nothing`; docstrings now show `Union{String,Nothing}`.

### Macro shims pinned to default visibility

`[wrap.macros]` shims (`replibuild_shim_<NAME>` in the generated `replibuild_shims.c`) carried default attributes, so a project built with `-fvisibility=hidden` turned every shim **local** in the `.so` — and since the wrapper's function list comes from `nm -g --defined-only`, all declared macros silently vanished from the module. Live config hitting this: box2d3 (4 `[wrap.macros]` entries + `-fvisibility=hidden`). Shims are now emitted `__attribute__((used, visibility("default")))` — `used` additionally survives LTO internalization.

### ABI trap for misaligned ≤16B blob params by value

A packed struct with a misaligned member is MEMORY class under SysV **even at ≤16 bytes** — passed on the stack — while its opaque `NTuple{N,UInt8}` blob image classifies INTEGER and travels in registers. A by-value crossing of such a param silently fed the callee garbage; the existing guard only trapped the float-bearing (SSE-class) case, and `is_c_lto_safe` only gates *returns* (returns were already correct via the explicit-sret branch). The `blob_abi_offenders` param scan now also traps ≤16B blob params with any misaligned member, regardless of float content. Aligned all-integer blobs (INTEGER on both views) and >16B blobs (MEMORY on both views) remain callable, as before.

### Bitfield multi-byte accessors: exact byte-span assembly

Multi-byte bitfield getters/setters loaded a power-of-2 container (`UInt16/32/64`) at the field's byte offset — which can overhang the struct tail when the container is wider than the spanned bytes (e.g. a 17-bit field starting in the struct's last 3 bytes). The getter read out of bounds; the **setter wrote out of bounds into a heap Vector**. Accessors now assemble exactly `ceil((bit_offset_in_byte + bit_size)/8)` bytes with plain tuple indexing (no pointers), clamped to the DWARF `byte_size` with a generation-time warning on inconsistent DWARF. Setters also accept negative values via wrapping conversion (`v % UIntN`) instead of throwing `InexactError`.

### C generator cleanups

- **Dead packed-struct branch removed** (~50 lines): every `!layout_verified` struct with positive `byte_size` already became an opaque blob in the preceding branch, so the packed detection could never be reached. A comment marks the invariant.
- **Unresolved-type globals fail safe:** a global whose DWARF type didn't resolve emitted `cglobal(..., Any)` + `unsafe_load` — reading memory as a boxed Julia pointer (garbage or crash). Such globals now get only a `<name>_ptr()::Ptr{Cvoid}` accessor; the value getter is emitted only for clean, resolved types.
- **Callback docs no longer guess:** when fuzzy name-matching found no `[function_pointer_typedefs]` candidate, the docstring fell back to **the first typedef in the table** — documenting an arbitrary `@cfunction` signature users would build crashing callbacks from. No positive match now means the DWARF signature or "signature unknown", never a guess.

### Version markers aligned

`Project.toml` (stale at 2.5.8), `RepliBuild.VERSION` (stale at 2.5.7), and the `runtests.jl` pin had drifted three ways; all now say 3.0.0. The fingerprint reads `Project.toml` via `pkgversion`, so keeping it bumped now has teeth.

## v2.5.9

Dialect fix for C++ virtual dispatch: `jlcs.vcall` now translates all the way to LLVM IR instead of segfaulting at emit.

### `jlcs.vcall` emit crash (operandSegmentSizes)

`jlcs.vcall` *lowered* cleanly (the conversion pass returned success), but emitting the result — `emit_llvmir`, and therefore `emit_object`, which calls it first — **SIGSEGV'd inside `translateModuleToLLVMIR`** (`OperandRange::split` → `DenseArrayAttr::getSize`). `VirtualCallOpLowering` hand-built the indirect `llvm.call` via a raw `OperationState` and set `operandSegmentSizes = {1, nArgs, 0}` — a **3-entry** array. But `llvm.call` carries `AttrSizedOperandSegments` with **two** operand groups (`callee_operands`, `op_bundle_operands`); for an indirect call the callee pointer is the *first element of `callee_operands`*, so the correct value is `{1 + nArgs, 0}`. It also omitted `var_callee_type`. During translation the 3-entry array was split against a 2-segment op and walked off the end.

The lowering now uses the dedicated indirect-call builder `CallOp(LLVMFunctionType, ValueRange)`, which sets `operandSegmentSizes` and `var_callee_type` correctly. Value- and void-returning calls both emit; the indirect call comes out as `call … %slot(ptr %this, …)`.

**Scope / why this was latent:** no production producer emits `jlcs.vcall` — `generate_virtual_method_ir` resolves each virtual method to a *direct* `llvm.call @<mangled>` from DWARF vtable data, so the C++ AOT thunk path never hit this code. The op is exercised only by hand-written IR, and the existing `vcall` tests stopped at parse+lower (the prior code comment even claimed the AOT path "works"). So this was a real but off-production-path defect in a test-only op. It was **not** version skew: verified against system LLVM/MLIR 22.1.6 with a library-free minimal fixture, and a control op (`scope`) emits through the identical translator in the same dual-LLVM process.

Guarded by a new emit regression in `test/test_mlir_templates.jl` §8 (value + void): lower → `emit_llvmir` → assert the indirect call is present. Unblocks — but does not implement — the multiple-inheritance `this`-adjustment (a secondary-base `vcall` still passes `this` unadjusted; tracked under "Not Yet Built").

## v2.5.8

ABI-correctness release for the C path. Headline: structs with struct-typed members now resolve to named Julia fields instead of opaque byte blobs, closing a silent by-value miscompile. Plus C-wrapper ergonomics, a library-free ABI trace test, and verified compatibility with system LLVM/MLIR 22.1.6.

### Nested-Struct Member Resolution (C path)

The C generator emitted an opaque `_data::NTuple{N,UInt8}` byte blob for **any** struct with a struct-typed member — even when every member was itself a resolved type. For a struct ≤16 bytes whose members are floats, that byte image misclassifies under the x86-64 SysV ABI: the real struct travels in SSE registers (XMM), the byte blob claims INTEGER registers. Consequence when such a struct crossed `ccall` **by value**: returns came back as garbage (register noise, e.g. `1e-13`), and arguments fed the callee garbage — which in practice aborted the process (e.g. Box2D's own `b2IsValidAABB` assert → SIGILL). On box2d3 this was 58 of 664 symbols (the geometry/query cluster: `b2Body_GetTransform`, `b2Body_ComputeAABB`, `b2Body_GetMassData`, `b2World_OverlapAABB`, the `b2Compute*`/`b2Collide*` families).

`GeneratorC.jl` now runs an **exact-layout proof** before falling back to a blob:

- Every member is typed with a Julia field of exactly known `(size, alignment)` — primitives, pointers, enums, `NTuple{N,·}`, and structs already emitted with verified named fields (topological emission order guarantees member-before-container).
- The emitter then **proves** Julia's natural layout (explicit align-1 `_pad_N` fillers across DWARF offset gaps + natural field alignment) reproduces every DWARF member offset *and* the DWARF total `byte_size`.
- Proof passes → named fields, and the struct is registered as eligible to be an inline member of later structs. Any doubt → keep the opaque blob.

**Why:** the root cause was a member-resolution bailout, not the ABI classifier — `b2Vec2`/`b2Rot` resolved fine, the generator just refused to compose them. The proof is the safety boundary: **exact or opaque, never approximate**. Packed structs (unaligned members) and bitfield structs still blob correctly. On box2d3 the blast radius drops from 58 to **0**; all 99 previously-opaque structs resolve to named fields, including the 96-byte `b2WorldDef` with correct padding.

### ABI Safety Trap for Residual Float Blobs

A ≤16-byte float-bearing struct that *stays* opaque (genuinely unreproducible layout — packed floats, bitfields, unresolvable member types) and would cross `ccall` by value now generates a loud `error()` stub instead of a silently-corrupting call. MEMORY-class returns (>16 bytes, or unaligned) are unaffected — they still route through the explicit-sret branch, which is byte-exact even for a blob. Register-class float blobs are the only unfixable case, and they now fail closed.

### C Wrapper Ergonomics

- **`Cfloat`/`Cdouble` parameters loosened to `::Real`** (mirrors the existing `::Integer` widening), with a checked convert at the call site. `step(w, 1/60, 4)` and integer-literal float args now work instead of throwing `MethodError` on the strict `Cfloat` slot.
- **`with(s; field=value, …)` helper** emitted in every generated C module — the idiomatic way to customize an immutable `*Def`-style struct (`with(DefaultWorldDef(); gravity = Vec2(0, -10))`). Not exported; call as `Mod.with(...)`.
- **Blob accessor GC-preserve fix** — `getproperty` on byte-blob structs now roots the `Ref(getfield(x, :_data))` temporary that actually holds the storage, instead of preserving the immutable value `x` (a no-op). Closes a latent use-after-free under GC pressure.
- **Docstrings** for struct-returning functions show the resolved struct name instead of the `Any` metadata sentinel.

### Library-Free ABI Trace Test

`test/test_abi_nested.jl` + `test/abi_nested_test/` — a self-contained C fixture (nested-float, nested-int, packed, and array-of-struct members) that traces compile → DWARF → wrap → live by-value crossings in a subprocess, so an ABI break can never take down the test session. Asserts named-field resolution, exact round-trips through registers, MEMORY-class controls, and that packed structs refuse by-value crossings loudly. Wired into `devtests.jl` as section 8. This is the structural-proof gate: the bug was reproduced library-free before any generator change.

### Per-File IR Cache Correctness

The per-file IR cache (`<file>.ll` under the build dir) was keyed on source mtime alone (`needs_recompile`). Changing `[compile].flags`, `defines`, or include dirs in `replibuild.toml` does **not** move any source mtime, so the cache reported a hit and silently reused IR built with the *old* configuration — the resulting `.so` looks fine but was compiled wrong. The only workaround was `rm -rf .replibuild_cache build`. The project-level content hash already saw the toml change and ran the pipeline, but the per-file gate then independently decided "source unchanged, skip" — the two layers disagreed.

`needs_recompile` now also checks a **compile fingerprint** — a hash of the compile flags, defines, include-dir paths, `Base.libllvm_version`, and target triple — stored in a `<file>.ll.key` sidecar. A mismatch (or a missing key, i.e. a cache from before this fix) forces recompilation; the fingerprint is computed once per build and threaded through the parallel compile path. Editing one source still recompiles only that file (the fingerprint excludes individual source content — that stays the mtime's job); a flag/define/include change correctly busts the whole set. Guarded by `test/test_cache_invalidation.jl` (devtests §9): a `-fvisibility=hidden` toggle must recompile with no manual cache clear *and* drop the now-hidden internal symbol from the export table.

### LLVM / MLIR 22.1.6 Compatibility

System LLVM/MLIR moved 22.1.5 → 22.1.6 (a patch release). The JLCS dialect (`src/mlir`) was clean-rebuilt against it with **zero source changes** — TableGen and all six translation units compile unchanged, and the binary links the same `libMLIR.so.22.1` SONAME. Verified functionally: `test_mlir_templates.jl` 50/50 (CStructs, sret, RAII ordering, virtual dispatch, TypeInfoOp), `test_jlcs_invariants.jl` 6 pass + 2 expected `@test_broken` (the known missing op verifiers). The C-bucket text-IR cleaning shims in `Compiler.jl` are keyed on the LLVM *major* version (e.g. `ptrtoaddr → ptrtoint`), so a patch bump introduces no new opcodes and needs no new shim.

### Upgrade Notes

No API breaks. Generated C wrappers change shape for libraries with nested-member structs: affected types now expose **named fields** instead of a single `_data::NTuple`, and their accessors move from `getproperty` byte-extraction to real struct fields. Code that reached into `x._data` directly (never the intended interface) must switch to the named fields; code using the documented field/accessor names is unaffected and gains correctness. If you updated system MLIR, rebuild the dialect: `cd src/mlir && ./build.sh`.

## v2.5.7

Stabilization release on top of the v2.5.6 DAG diff work. Focus: cross-LLVM compatibility, sret correctness on the C path, dialect op fixes, and wiring of orphaned test suites into CI.

### Per-Language LLVM Toolchain Routing

`LLVMEnvironment` now resolves toolchain binaries through a per-language bucket rather than a single global PATH lookup. The `:c` bucket targets the LLVM version that matches Julia's internal libLLVM (Tier 1 `llvmcall` + LTO bitcode must be ABI-compatible with what Julia loads), while the `:cpp` bucket targets the system LLVM/MLIR (currently 22+) needed for the JLCS dialect.

- `LLVMEnvironment.resolve_tool(name, language)` is the new entry point — replaces the unscoped form at every call site (Compiler.jl, ThunkBuilder.jl)
- IR sanitize pass on link to strip attributes/metadata that the older internal LLVM rejects when consuming bitcode produced by a newer system clang
- Documents the dual-bucket reality in README — system LLVM 21+ for the C/C++ pipeline, internal Julia LLVM (18–20) for `llvmcall` consumption, coexisting by design

**Why:** A single global LLVM version cannot serve both Tier 1 (must match Julia internal) and the C++ MLIR dialect (must match system LLVM 22). Buckets make the constraint explicit and let the C and C++ paths evolve at independent cadences. Ground for the upcoming C-path internalization (Julia 1.12.6 + LLVM 18, no system fallback).

### C sret Return Classification + Thunk Path Consolidation

Fixed the C generator returning structs by-value through `ccall` when the platform ABI requires sret (caller-allocated return slot passed as a hidden first pointer arg). Previously: silent layout corruption on structs >16 bytes returned from C. Now: routes through the consolidated thunk path that allocates and passes the return slot explicitly.

- `GeneratorC.jl` reduced ~170 lines, dispatch logic for sret unified with the existing C++ thunk emission
- `Compiler.jl` thunk plumbing collapsed into one entry point — was duplicated across C/C++

### dlsym/dlopen Returning `nothing` on Newer Julia

`Libdl.dlsym` / `dlopen` started returning `nothing` instead of throwing on newer Julia versions when symbols/libraries cannot be resolved. The JIT path's symbol resolver assumed a thrown error and crashed downstream with a less informative message. Fixed across `JITManager.jl` — explicit `nothing` checks at all 7 call sites with the same `init_error` surfacing pattern introduced in v2.5.5.

### MLIR Dialect Fixes

- **MarshalArg / RetOp missing assemblyFormat** — both ops parsed but failed to print, breaking round-trip and `mlir-opt` debugging. Added `assemblyFormat` to JLCSOps.td.
- **JIT selftest** added to verify the dialect loads and lowers cleanly on every build (catches missing op declarations before they hit a wrap call).
- **JLCSCAPIWrappers.cpp** — new file exposing C wrappers for dialect APIs needed by Julia bindings.
- **JLCSPasses.cpp** — internal cleanup, `getPackedSizeInBits` consolidation continued from v2.5.5.

### DAG Diff Tuning

~700 lines reworked in `DAGDiff.jl` based on stress-test feedback from v2.5.6:
- Tighter propagation rules — pointer-to-mismatched no longer propagates (only by-value containment does)
- DOT export polish, mismatch annotations more readable
- Query API stabilized

### Test Infrastructure

- Wired three orphan test suites into `runtests.jl` / `devtests.jl`: DAGDiff (1336 lines), MLIR templates (736 lines), exception handling (101 lines)
- `.gitignore` patches to exclude per-project `dag/` exports and build artifacts
- `test/c_test/verify.jl` updated to match consolidated thunk path
- `test/stress_test/verify.jl` removed (188 lines) — replaced by `test_introspect.jl` + `introspect_demo.jl` which cover the same ground via the public API

### Misc

- README dual-toolchain clarification
- MLIR documentation pass
- Stress-test introspect demos use `joinpath(@__DIR__, "..", "..")` so they activate the right project regardless of where they're invoked from
- `src/mlir/build.sh` apt hint corrected to `mlir-21-dev`

### Upgrade Notes

No API breaks. If you were calling `LLVMEnvironment.resolve_tool(name)` directly (not part of the public API but possible in downstream tooling), you must now pass a language: `resolve_tool(name, :c)` or `resolve_tool(name, :cpp)`.

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
