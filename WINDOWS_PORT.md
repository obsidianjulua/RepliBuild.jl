# Windows Port — working notes

Status as of 2026-09-05, against `main` @ `65a8b37`. Target is
**`x86_64-w64-windows-gnu`** (mingw), native host build — not a cross-compile from
Linux, and not MSVC.

Every file:line in this document was verified against `main` on the date above.
Line numbers drift; re-check before trusting one.

---

## ⚠️ Work from `main`. Do not check out `windows-win64-abi`.

There is a branch named `windows-win64-abi`. It sounds like the Windows branch and
it is not:

```
commits on branch not in main: 0
commits on main not in branch: 22
branch tip: 67b1910  2026-08-23
main tip:   65a8b37  2026-09-03
```

Its work was **already merged**. `main` carries the Win64 ABI classifier
(`AbiTarget` in `src/mlir/impl/JLCSPasses.cpp`) and `test/test_win64_abi.jl`.
Checking that branch out silently reverts a month of unrelated work — SysConfigGen,
the receiver-gate fixes, the tracked `CLAUDE.md`. The branch is a stale leftover.

---

## Environment setup

### Git — before cloning

```powershell
git config --global core.autocrlf false
git config --global core.longpaths true
```

`core.autocrlf false` is **required, not a preference**. Git for Windows defaults to
converting LF→CRLF, which corrupts `src/mlir/build.sh`; MSYS2 bash then fails with
`$'\r': command not found`, which presents as a broken script rather than a
line-ending problem.

### Clone

Clone over **HTTPS**. The canonical checkout's remotes use SSH host aliases bound to
per-identity keys that do not exist on a fresh machine. Both repos are public.

```powershell
mkdir "$env:USERPROFILE\Desktop\Projects" -Force; cd "$env:USERPROFILE\Desktop\Projects"
git clone https://github.com/obsidianjulua/RepliBuild.jl.git
```

Optional — the Hub is the integration corpus, useful once the engine loads, not
needed to start:

```powershell
git clone https://github.com/obsidianjulua/RepliBuild-Hub.git
```

Confirm the branch:

```powershell
cd RepliBuild.jl; git branch --show-current; git log -1 --oneline
```

Must report `main`.

### Julia — match the reference host exactly

```powershell
juliaup add 1.12.7; juliaup default 1.12.7; julia --version
```

Parity is not cosmetic. The **C bucket links, optimizes and assembles in-process on
Julia's resident libLLVM** (18.1.7 on Julia 1.12.7), version-matched to the
`Clang_unified_jll` clang that emits the IR. A different Julia means a different
libLLVM and a different failure surface — divergence there looks like a port bug and
is not one.

### MSYS2 toolchain

Launch **MSYS2 CLANG64** — that specific Start-menu entry. Not `MSYS2 MSYS`, not
`MINGW64`, not `UCRT64`.

Two reasons it must be CLANG64:

1. It is the `x86_64-w64-windows-gnu` environment. Julia's official Windows binaries
   are themselves a mingw-w64 build, so `julia.exe`/`libjulia.dll` are GNU-ABI.
   Targeting MSVC would give wrapper DLLs a different C++ ABI (mangling, EH model,
   STL) than the process loading them.
2. It is the only MSYS2 repo carrying the MLIR package.

```
pacman -Syu
```

This closes the terminal partway through core updates. Reopen MSYS2 CLANG64 and
repeat until it reports nothing to do, then:

```
pacman -S --needed mingw-w64-clang-x86_64-toolchain mingw-w64-clang-x86_64-mlir \
                   mingw-w64-clang-x86_64-cmake mingw-w64-clang-x86_64-ninja git
```

`mingw-w64-clang-x86_64-toolchain` is a package **group** — accept all of it. It
supplies clang, lld, compiler-rt and llvm. The llvm half matters as much as the
compiler: `llvm-dwarfdump`, `llvm-nm` and `llvm-readobj` are what the port has to
move onto, away from GNU `readelf` and `nm -D`.

Verify:

```
clang --version && cmake --version && ninja --version && llvm-config --version
```

**`llvm-config --version` will report 21.1.8.** The Linux reference host is on
22.1.8. Expect MLIR C++ API skew between them to be the first compile error in
`src/mlir/build.sh` — that is known and expected, not a misconfiguration. MSYS2
shipping MLIR prebuilt is what clears the "LLVM+MLIR 21+" requirement without
building LLVM from source.

`src/mlir/build.sh` is bash and runs as-is under MSYS2.

---

## Already portable — no work needed

Verified, do not spend time here:

- **Build identity.** Moved off the ELF `PT_NOTE` GNU build ID to a **sha256 of the
  library file**. Format-agnostic: reads the same on PE and Mach-O.
- **Library extension switching** — `.so`/`.dylib`/`.dll` is already handled in
  `src/Wrapper/Generator.jl`.
- **`Libdl.dlopen`** — portable as used.
- **CRT safety**, by construction: `Libc.malloc` → placement-construct → dtor thunk →
  `Libc.free` is symmetric, and `[wrap.cstring_owned]` deallocators are library
  symbols, so nothing frees across a CRT boundary.

The codebase is **scoped** to Linux, not fused to it — there is exactly one
`Sys.islinux` branch in all of `src/`.

---

## The punch list

Ordered by when it will bite.

| # | Site | Problem |
|---|---|---|
| 1 | `src/RepliBuild.jl:62` | Hard `error()` on `!Sys.islinux()` |
| 2 | MSYS2 MLIR 21.1.8 vs reference 22.1.8 | First compile error in `src/mlir/build.sh` |
| 3 | `src/Builder/Compiler.jl:1893` | GNU `readelf` — ELF only, cannot read PE |
| 4 | `src/Wrapper/Symbols.jl:131` **and** `src/Wrapper/Utils.jl:1082` | `nm -D` — ELF dynamic symbols; a DLL needs the export table |
| 5 | `src/Wrapper/TypeRegistry.jl:59` vs `src/IRGen/ir_gen/TypeUtils.jl:16` + `src/IRGen/ir_gen/ArrayViewGen.jl:38` | LLP64 split-brain |
| 6 | `test/test_abi_nested.jl:6` | Asserts SysV-only XMM behaviour |

### Status — 2026-09-05

Branch `windows-port-2026-09`. `runtests.jl` is green (30 testsets, 1062
assertions, exit 0) and every `devtests.jl` integration fixture passes:
Pipeline 12/12, Stress Test 4/4, MI 43/43, VI 40/40, STL 28/28, c_test 72/72,
C Abomination 15/15, Callbacks 3/3, Ingest 8/8, MLIR Templates 87/87. The
dialect builds and loads as `src/mlir/build/libJLCS.dll` (46 MB), so Tier 2 is
live.

Read that as "no fixture fails", not "one run prints all of it": the unwind
abort in **Open** below truncates some runs after the callback fixture, so the
Ingest and MLIR-Templates figures come from runs that got past it and from
running those two files directly. Every run so far reports **zero** test
failures — every assertion that executes, passes.

| # | State | Notes |
|---|---|---|
| 1 | done | Gate admits Windows. macOS still refused — the AAPCS64 classifier really is unbuilt. |
| 2 | **not real** | MSYS2 CLANG64 ships MLIR **22.1.8**, matching the reference. There is no version skew. |
| 3 | done, **not as prescribed** | See below — the readelf parser was kept, not replaced. |
| 4 | done | Plus a third site the doc did not list, which is the one that actually broke builds. |
| 5 | done | With the cross-tier width guard the doc asked for: `test/test_llp64_widths.jl`. |
| 6 | **open** | Still asserts SysV. Latent — it passes today. |

**#3 did not need a second parser, and did not need dwarfdump.** GNU `objdump`
is BFD-based, reads `pei-x86-64`, and shares binutils' `dwarf.c` printer with
`readelf` — `objdump --dwarf=X` and `readelf --debug-dump=X` are the same code
emitting the same text. Verified against a real PE DLL: 193/193 identical DIE
lines and an identical `rawline` table. So `Compiler.jl` keeps its readelf-
dialect parser untouched and only swaps the tool. Routing PE through
`llvm-dwarfdump` as the doc suggested would have been the harder path, not the
easier one: dwarfdump is a genuinely different dialect and the parser would
have needed the rewrite this avoids. `_dwarf_dumper()` therefore requires GNU
objdump specifically and rejects `llvm-objdump` by checking `--version` for
`GNU`. Note also that `DWARFParser.jl` lives in `src/Builder/`, not `src/IRGen/`.

**#4's real form.** The doc named two `nm -D` sites, and both were broken —
`nm -D` on a PE is not an empty answer but a hard error, *"File format has no
dynamic symbol table"*. But the site that actually broke builds was a third one
the doc did not list: `extract_symbols_from_binary` in `Compiler.jl`, using
`nm -g --defined-only`. **mingw links its C runtime, startup code and unwinder
statically into every DLL**, so nm reports `snprintf`, `memcpy`, `atexit`,
`fprintf` and `abort` exactly like the library's own functions. On ELF that code
lives in a shared libc and is only *referenced*, so nm's answer and the API have
always been the same list and nothing had to tell them apart. All three sites now
read the PE export directory (`objdump -p`), which is the strongest available
authority: `GetProcAddress` consults only that directory, so a symbol outside it
cannot be reached through the library even in principle. Measured — c_test 115
nm symbols / 28 exports, mi_test 133/43, vi_test 142/56, stl_test 691/513, and
zero CRT symbols in any export table.

---

## Found during the port — none of it in the punch list

Roughly in order of how much damage each did.

**libc++ is not libstdc++, and the STL path assumed libstdc++ three ways.**
CLANG64 ships libc++. (a) Nearly every inline member is `_LIBCPP_HIDE_FROM_ABI`,
which mangles an ABI tag in *after* the name — `size[abi:nqe220108]() const` —
and every test in `_classify_stl_method` is a `startswith(sig, "size(")`, so
every tagged member was dropped. `std::vector<int>` came out of a real build
with one method. (b) `_normalize_stl_type` matched `basic_string` with a bare
`startswith`, so libc++'s nested helpers normalised to the string itself and
`~__annotation_guard()` was recorded as the string's destructor — an
`EXCEPTION_ACCESS_VIOLATION` inside `__is_long`. (c) `_LIBCPP_HIDE_FROM_ABI`
also expands to `exclude_from_explicit_instantiation`, so `template class C;`
emits **nothing** for those members; they exist only where odr-used, and there
is no macro to switch it off. The generated instantiation TU now carries a
never-called function that names each member, guarded by `if constexpr` so one
body covers vector/string/map/set/deque/list, syntax-checked before it is
trusted so nothing that builds today can regress.

**The C bucket had no libc.** `.c` goes through `Clang_unified_jll`'s clang to
stay version-locked to Julia's libLLVM. That JLL already targets
`x86_64-w64-windows-gnu` — the triple was never the problem — but it is a bare
compiler artifact with no headers and no CRT, so every C file died on its first
system `#include`. It also defaults to `-rtlib=libgcc`, which CLANG64 (a
compiler-rt/libc++/libunwind environment) does not ship at all.

**ORC deadlocks on an unresolved symbol; it does not raise.** Monolithic LTO
embeds the whole post-LTO module and `Base.llvmcall`s it, so every symbol that
module declares must resolve in the consumer's process. `mathkit.c` calls
`snprintf` and `sqrt`; neither is exported by anything in the process, and the
run hung with `JIT session error: Symbols not found: [ snprintf ]`. That path
had no pre-flight. `_tier1_preflight!` has done exactly this for per-function
slices all along — and could not run on Windows either, because
`_symbol_resolves_via` called `ccall(:dlsym, ...)` and there is no `dlsym`
symbol in a Windows process.

**mingw unwinds with SEH.** The personality is `__gxx_personality_seh0`, not
`__gxx_personality_v0`, and the Windows C++ runtime exports only the former.
The name was written down twice in C++ (`JLCSPasses.cpp`,
`JLCSCAPIWrappers.cpp`) and needed a third in Julia; it is now
`MLIRNative.CXX_PERSONALITY`, with `test_cxx_personality.jl` pinning the two
languages together.

**PE binds every symbol at link time.** ELF lets a `.so` carry undefined symbols
and resolves them at `dlopen` through `RTLD_GLOBAL`. Thunk libraries therefore
need the full link — main library and `libJLCS` both named.

**`FILE` is `struct _iobuf` here.** `INTERNAL_TYPE_BLOCKLIST` named only glibc's
`_IO_FILE` family, so the screen caught nothing on Windows and `_iobuf` was
declared and exported into wrappers' API surface.

**A live mmap blocks deletion.** `JSON.parsefile` defaults to `use_mmap=true`
and Julia drops the mapping only at GC, so `clean()` failed on a tree RepliBuild
had just built — and the file left behind was `compilation_metadata.json`, not
the `.dll` anyone would have suspected. Free on POSIX, where a mapped file still
unlinks.

**Windows paths are not string literals.** `C:\Users` contains `\U`. Hand-rolled
TOML writing and hand-quoted paths in generated wrappers both failed to parse —
the generated module could not be loaded at all. Fixed with `TOML.print` and
`repr`.

**Smaller, but each cost a debugging session.** `PATH` separator is `;`; tools
need `.exe`; `LLVM_CONFIG` was advertised in an error message and read nowhere;
`llc`/`llvm-config` version mismatch reported from a silent hardcoded fallback;
`cglobal(:stdout)` does not exist (`__acrt_iob_func` does); a path-comparison
guard hardcoded `/` and so flagged every correctly-vendored header; OneDrive
marks build trees `ReadOnly + ReparsePoint` and `rm` fails with `EACCES`.

### Open

- **#6** above — `test_abi_nested.jl` still asserts SysV XMM classification.
- **`libunwind: pc not in table`** aborts the `devtests.jl` parent process after
  the callback fixture, in some runs, before the last two sections. Zero test
  failures in every run — every assertion that executes passes — and it does not
  reproduce standalone (callback verify is 3/3 over repeated runs) nor when the
  parent's build/wrap/spawn sequence is replayed on its own. It needs prior
  in-parent JIT activity, which points at SEH unwind-table registration for
  ORC-JIT'd frames on COFF rather than at anything in the fixture.
- `get_library_name` hardcodes `.so` on every platform. Harmless — `LoadLibrary`
  does not care about the extension — but it is why a Windows build produces
  `libstl_test.so`.
- `test_tier1_dispatch.jl:237-238` uses a Unix `:` separator for
  `JULIA_LOAD_PATH`/`JULIA_DEPOT_PATH`. Tier 1 is quarantined.

---

### 1 — the platform gate

`src/RepliBuild.jl:62` errors out on any non-Linux kernel. Nothing loads until this
is relaxed; `using RepliBuild` fails. It is a deliberate gate, not an accident — the
message names the four assumptions (ELF `.so`, DWARF via llvm-dwarfdump, GNU nm,
Linux LLVM layout). Relax it to admit Windows only once the items below are real,
or the failures move from one clear message to four confusing ones deep in the
pipeline.

### 3 — the two DWARF parsers

`src/Builder/Compiler.jl` parses **GNU readelf output and nothing else** (its own
comment at ~:1878 says so; the version probe is at :1893). `src/IRGen/DWARFParser.jl`
parses **llvm-dwarfdump**, which is the portable path.

`CLAUDE.md` flags this two-parser split as a hazard — *"two parsers, two dialects,
do not copy a regex between them"* — and on Windows it is the opposite: **the split
is what makes the port viable**, because one of the two already speaks a portable
dialect. Route PE through the dwarfdump path rather than teaching the readelf parser
a second format.

### 4 — symbol extraction, two call sites

```
src/Wrapper/Symbols.jl:131   nm -D --defined-only [--demangle] <binary>
src/Wrapper/Utils.jl:1082    nm -D --defined-only <thunks_lib>
```

Both are ELF dynamic-symbol reads. A DLL needs the PE export table
(`llvm-readobj --coff-exports`, or `llvm-nm`).

**Fixing only the first leaves AOT thunk verification broken in a way that looks
like a thunk bug.** The second site is easy to miss — earlier notes recorded only
`Symbols.jl:131`.

PE also inverts the export model: `dllexport` is opt-in, where ELF exports by
default. That is a design problem for `__rb_*` static promotion, which assumes a
symbol becomes visible by being renamed with default visibility. Static promotion is
Tier 1, which is **quarantined** (`[wrap.tier1] enable` defaults false, no shipped
package takes a Tier-1 path), so this can be deferred — do not let it block the
port.

### 5 — LLP64 split-brain

`long` is **4 bytes on Windows**, 8 on Linux. The codebase disagrees with itself:

```julia
# src/Wrapper/TypeRegistry.jl:59        — CORRECT, self-corrects on Windows
"long" => "Clong", "long int" => "Clong", "signed long" => "Clong", "unsigned long" => "Culong",

# src/IRGen/ir_gen/TypeUtils.jl:16      — WRONG on Windows, groups long with 64-bit types
elseif type_str == "long" || type_str == "long long" || type_str == "int64_t" || ... "size_t" ...

# src/IRGen/ir_gen/ArrayViewGen.jl:38   — WRONG on Windows
"long" => "i64", "unsigned long" => "i64", "int64_t" => "i64", "uint64_t" => "i64",
```

`Clong` resolves to `Int32` on Windows automatically, so **Tier 3 would say 4 bytes
while the Tier 2 MLIR thunk says 8 — with nothing comparing them.** Silent wrong
answers, not a crash.

This is the highest-value fix and it is **independently testable from Linux** using
the clang-as-oracle method below. Worth doing before any Windows-specific code, and
it needs a guard that compares the two tiers' width for the same type — the
disagreement is the bug, so the test must be the comparison.

### 6 — a SysV-only test expectation

`test/test_abi_nested.jl:6` states the contract: *a 16-byte all-float struct travels
in XMM registers*. That is SysV. On Win64 it **inverts to sret** — aggregates never
reach XMM. Needs a target-conditional expectation before it runs on Windows.

Note it has no `using RepliBuild`; it is meant to be `include`d from `devtests.jl`,
not run standalone.

---

## Already built: the Win64 ABI classifier

`src/mlir/impl/JLCSPasses.cpp` carries both conventions behind
`enum class AbiTarget { SysV, Win64 }`, selected at compile time by `kHostAbi`
(`#ifdef _WIN32`), with `#error` on non-x86-64 so AArch64 cannot silently inherit
x86 struct rules. Both rule sets always compile, so the inactive one cannot rot.
SysV behaviour is bit-identical on Linux.

Four divergences from SysV, three of them silent:

- **Size is the only criterion** — 1/2/4/8 bytes in a register, everything else
  indirect, *including* the 9–16 byte band SysV splits across two registers.
- **Aggregates never reach XMM** — `{float,float}` is `i64`; XMM0 under SysV.
- **Coercion is `iN` of the struct's own size**, not always `i64`. Matters on return:
  reading `i64` from RAX when the callee wrote EAX gives a garbage high half.
- **Indirect arguments take no `byval`** — the existing alloca+store already *is* the
  caller-allocated temporary.

### clang is a Win64 oracle from Linux

```
clang --target=x86_64-w64-windows-gnu -S -emit-llvm
```

lowers signatures for Windows with no mingw headers, linker, or Windows host. This
is pinned in `test/test_win64_abi.jl` (95/95, `devtests` §6b).

**It is a SPECIFICATION test, not a behavioural one.** A Win64 callee cannot be
loaded or run on Linux, so it catches an encoded rule that disagrees with clang — it
does **not** prove the lowering runs correctly on Windows. Until this VM existed,
that was the unprovable half. Proving it here is the point of the exercise.

`-DJLCS_FORCE_ABI_WIN64` compiles the Win64 rules into a Linux `libJLCS.so` that can
be inspected but cannot run anything. Build it to a scratch path, never over a
working `libJLCS.so`.

---

## Testing

- `julia --project=. test/runtests.jl` — no toolchain required. Start here; it should
  be the first thing that passes on Windows.
- `julia --project=. test/devtests.jl` — needs the full toolchain and `libJLCS.so`.

Two suite hazards worth knowing before interpreting a red run:

- **devtests aborts the whole file at the first failing top-level testset.** A single
  red hides an unknown number of unrun tests. It is not one broken test.
- **`exit(0)` inside an included file ends the suite with a success status.** Six
  such sites remain across five files, most gated on `!MLIR_AVAILABLE`. With
  `libJLCS.so` absent, `test_mlir_templates.jl` — included *first* among the libJLCS
  group — exits the process and everything after it never runs **while the suite
  reports exit 0**. On a fresh Windows checkout with no dialect built yet, this is
  the default state. Do not read that green as meaning anything.

---

## Beyond Windows

**macOS** needs the AAPCS64 classifier, the one unbuilt convention. An x86 macOS VM
tests the *wrong* ABI (modern macOS is arm64) and violates Apple's EULA on non-Apple
hardware. But AAPCS64 is provable from Linux today with the same clang-as-oracle
method used for Win64 — no Apple hardware required for the specification half.
