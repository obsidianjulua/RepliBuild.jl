# Changelog

All notable changes to RepliBuild.jl are documented in this file.

## Unreleased

### `[wrap] shim_defines` builds a header's implementation from the TOML (2026-10-26)

- **A single-header library needed a file of its own, or a workaround with two traps.** An stb-style header compiles its implementation only where a macro such as `STB_IMAGE_IMPLEMENTATION` is defined. From the TOML alone that took a `-D` in `[compile] flags` plus a `[wrap.macros]` entry, because `generate_macro_shims` wrote the shim file only for macros: with no entry the build had no source and stopped at "Compilation produced no library". The `-D` then reached every file the build compiles, so a library source that included the same header defined every function a second time and the link failed. And `discover(force=true)` rewrites `[compile] flags`, so the define vanished; the next build had no implementation, and its functions left the wrapper with no error. Found while documenting header-only libraries.
- **`[wrap] shim_defines`** (`["STB_IMAGE_IMPLEMENTATION"]`, or `NAME=VALUE`) is written as `#define` lines at the top of the shim file, before its `#include`s, and in no other file. The file is written when `macros` or `shim_defines` has entries, so `shim_headers` plus `shim_defines` puts the implementation in the library on the first build, and the first wrap binds it. A bare `NAME` is `1`, as `-DNAME` is. An entry that is not one macro definition on one line is a load error, since it is pasted into C and a dropped define fails silently; so is `shim_defines` without `shim_headers`. `discover(force=true)` keeps the key. The shim header guard now runs for a defines-only file too, since a system copy of the header would build that copy's implementation. With no macros the file has no `RB_SHIM_EXPORT` functions, so on Windows the auto-export default publishes the implementation.
- **Tests.** `test_shim_defines.jl` (runtests): the file written with defines alone (defines before includes, no shim functions) and with macros, no file with neither, the `#define` spelling, seven refused entries and the missing `shim_headers`, the `save_config` round trip, and the guard on a defines-only file. `test_toml_preservation.jl` keeps the key. `test_shim_defines_live.jl` (devtests §7f) builds `shim_defines_test/stbish.h` with no source and no macro entry and calls its implementation in a child process; builds it again beside a library source that includes the header, which links against the one implementation; and shows the same project with the `-D` in `[compile] flags` failing the link. Each part neutered alone fails them: no `#define` written (3 unit, 4 live: the TOML-only build succeeds and its functions are missing), the old macros-only gate (4 unit, 1 live), the old guard gate (2), the preserved key removed (2), validation off (6).

### `ingest` registers the way `discover` does (2026-10-26)

- **A bare `ingest` built, wrapped and changed the user's Julia environment.** `ingest` defaulted `register=true` and called `PackageRegistry.register` right after writing the TOML, before its own `build`/`wrap`, inside a `try` that logged failures at `@debug`. Since 4.3.0 `register` builds and wraps in place, vendors to `~/.replibuild/builds/<hash>/` and adds a package to `@v#.#`, so `ingest(lib; headers=…)` with the default `build=false` did all of that, and a failure (a library without DWARF, say) left a registry entry and said nothing. The comment claimed this mirrored `discover`, which registers only with `build=true, wrap=true`. Found in a walkthrough of the user docs.
- **Now `discover`'s rule.** A bare call writes the TOML and nothing else; `build=true, wrap=true` calls `register`, which builds, wraps, and then vendors; `register=false` builds and wraps only. A registration failure is thrown. The docstring and the Ingest sections of `guide.md` and `api.md` say so.
- **Tests.** `test_ingest.jl` (runtests), with `REPLIBUILD_HOME` in a temp dir and `REPLIBUILD_JULIA_ENV=none`: a default call leaves the registry empty and builds nothing, and with the registry home blocked by a file, `build=true, wrap=true` throws the registry's `IOError` instead of reaching the build. With the old tail restored both fail (an entry `fake` is registered; the build's `ErrorException` is thrown instead). devtests' ingest block: `build=true, wrap=true` on the c_test library leaves an entry with a vendored wrapper and no Julia environment.

### `surface_types_only` reaches a template-spelled record (2026-10-25)

- **A record keyed with its template arguments was unreachable from every signature that names it.** `_surface_type_keys` matched a spelling whole or by identifier token. A template instantiation's key keeps its arguments (`normal_distribution<double, boost::math::policies::policy<…> >`), so no token equals it, and a signature never spells it whole: boostmath's `pdf`/`cdf`/`quantile` take their distribution as `const T<…>&` (`Ref{T<…>}`), its constructors and accessors as `T<…>*`, and its methods record `boost::math::T<…>` as the class. All five distributions were dropped with the internals and emitted as empty structs, with no error; the Hub package kept them through `surface_types_extra`. Found wrapping Boost.Math, 2026-10-25.
- **Every dress is peeled before the lookup.** `_surface_core_spellings` takes a spelling to its fixed point under one removal at a time: `Ptr{…}`/`Ref{…}`/`NTuple{N, …}` unwrapped, a leading `const`/`volatile`/`struct`/`class`/`union`/`enum` cut, a trailing `*`, `&`, `&&`, `const`, `volatile` or `[N]` cut, and the leading scope cut at depth 0 of the template arguments (`_cpp_innermost_scope`, so an argument's own `ns::` stays, as the key carries it). Each form is tried as a key, bare and under `__enum__`; the token walk is unchanged, so nothing reachable before is lost. On boostmath's metadata the filter now keeps the five distributions with no keep-pattern (13 of 48 records, the same 13 the pattern list kept).
- **Tests.** `test_surface_types.jl` (runtests): the key is reached from each dress on its own (`const T<…>&` as c_type only, `Ref{T<…>}` as julia_type only, `T<…>*`, `Ptr{T<…>}`, `T<…> const* const` as a return, `NTuple{2, T<…>}`, `ns::T<…>` as the class, `const ns::T<…>&&`), a spelling whose argument lacks the key's `ns::` is not, and a name ending in `const` is not undressed. Reverting `resolve` to the whole-or-token match fails all eight cases.

### `clean(toml; deps=false)` keeps the vendored git clones (2026-10-25)

- `clean` removed `.replibuild_cache/` whole, and DependencyResolver's clones live in its `deps/`, so a Hub `test.jl`, which cleans before it rebuilds, re-cloned its dependency every run: minutes of each boostmath run. `deps=false` removes every other cache entry (the IR cache, the project hash, the slices) and keeps `deps/`, so the next build re-checks each clone's pin marker instead of fetching; with no `deps/` present the cache directory goes as before. The default is unchanged. `test_clean.jl` (runtests).

### Tier 2 classifies a by-value record from its DWARF facts (2026-10-25)

- **Every family the oracle listed was wrong live.** The dialect classified a record for x86-64 SysV from the LLVM body StructGen emitted for it, and a body cannot carry what the convention asks. An explicit pad array is bytes like a field's, so an eightbyte of a float and padding went in a general register: `{float; double}` crossed in RDI and XMM0 where C uses XMM0 and XMM1, and so did `{double; float}`, `{void*; float}`, `{int : 32; float}`, `aligned(16) float`, `_Complex float` and a `{double; float}` nested in a tiled body. A tiled body spells a nested record, an enum or a union's members as bytes, which hid a misaligned field inside: a packed `{char; {short; float}}` and a union holding a packed `{char; int}` went in registers where C uses memory, and abi_fuzz seeds 21 and 22 (`packed {char; {enum}}`) segfaulted on that shape, the callee reading a shifted pointer or writing its result through a shifted register. A bitfield's storage unit sat at its byte offset with its whole type, which read as misaligned where C checks only the bits: `#pragma pack(1) {char; int : 31}` went in memory where C uses one register. `long double` and `_Complex` members were `!llvm.ptr`, so a 16-byte `{long double}` or `{_Complex double}` emitted as 8 bytes. No body says `aligned(16)`, so `__int128` and `aligned(16)` records got 8-aligned byval and sret slots. Run live on Linux (clang 23.1.3, Julia 1.13.2) with the SSE registers poisoned before each call, 12 of 19 records returned garbage at -O0 and -O2; three that passed as arguments did so because Julia's own copy had left the second eightbyte in XMM1.
- **The facts are read on the Julia side and stated on the call.** `FunctionGen.sysv_record_abi` (`src/IRGen/ir_gen/SysVRecordABI.jl`) classifies a record from the metadata: members at their offsets, bitfields by bit range with no alignment check, an unnamed bitfield as padding, nested records and arrays at their real offsets, a union's members at its start, integers, pointers and enums INTEGER, `float`/`double`/`_Float16` SSE, `long double` X87 with X87UP above it, `_Complex` as its two halves, `__int128` as two INTEGER eightbytes; merge as clang's `X86_64ABIInfo` does (equal stays, NO_CLASS yields, MEMORY wins, then INTEGER, an x87 class against another is MEMORY, else SSE), a misaligned field anywhere makes MEMORY, so does more than 16 bytes or an X87UP not under X87. The record's alignment is `DW_AT_alignment` where clang wrote it, else the members' lowered until the layout fits it (a `packed {char; long double}` is 17 bytes, so 16 lowers to 1, as clang says). A typedef'd scalar is read through the member's `julia_type`. The thunk generator states the verdict on every `jlcs.ffe_call`/`jlcs.try_call`: `sysv_args`, one entry per operand, and `sysv_ret`, each `mem:<align>`, `regs:int,sse`, `regs:none,sse` (a low eightbyte of padding, passed nowhere) or `ignore` (a record of padding). A call with no record by value states nothing, so its IR is unchanged.
- **`buildCallShape` acts on the statement under the SysV target.** A stated record takes its class from it: byval and sret slots at the stated alignment (16 for `aligned(16)`, `__int128`, `long double`), the high eightbyte alone read from and stored to offset 8 when the low one is padding, no operand and a void call for a record of padding. An operand or result with no statement is classified from its body as before, which hand-written IR (`test_struct_abi.jl`) relies on. The Win64 rules never read the statement. A malformed statement is a lowering error. The probe seam prints `none` and `ignore` for the new shapes. Design choice: the alternative was carrying the leaf list into `!jlcs.c_struct` and classifying in C++. The Julia side already needed the verdict for its own refusals (an x87 return, an unreadable member), the facts are host-independent so the probe keeps diffing the SysV answer on a Windows box, and one derivation on one side leaves the C++ as the actor and the oracle as the check on the whole.
- **StructGen emits `long double`, `_Complex` and `__int128` members at their size.** `map_cpp_type` gives `f80` (10 bytes stored, 16 allocated and 16-aligned, in a packed body too), `!llvm.struct<(f32, f32)>` / `(f64, f64)`, and `i128`; the layout table measures them. A `long double` by itself stays `:opaque` to `by_value_crossing`, which both generators trap as before.
- **What SysV still refuses, and why.** A `long double` record returned by value: C leaves it in st0, where no thunk looks (as an argument it goes in memory, byval 16, and works). A record of 16 bytes or less holding a member whose kind DWARF does not settle (a spelling the classifier does not know and the metadata gives no Julia type for, a record with no definition, an STL type): its class is unknowable. `thunk_crossings` reports both as `abi_trap`, and GeneratorCpp's stub says so, apart from the "layout missing from DWARF" explanation it kept giving. On Win64 neither is refused, and the generator states nothing for the unreadable record (`sysv_statement`); asking for its statement threw and took every thunk of the library with it (Windows CI on this branch). `_byte_region_unclassifiable` and `_record_float_risk` are gone: a float union or a two-bitfield record of 16 bytes or less, which SysV refused as bytes that might hold a float, is INTEGER like C's and is called.
- **Tests.** `test_sysv_abi_oracle.jl`: 0 of 1165 crossings disagree (306 records, 919 functions); every curated record is `:agree` but the `long double` return, which is `:trap`; `SAO_FAMILIES`/`SAO_FUZZ_KNOWN` are gone and every disagreement fails, named by its shape. Neutered (`sysv_call_fact` returning no statement, so the dialect reads the body), 50 disagree and the suite goes red, and the live probe fails 22 of its 49 directions. `test_sysv_record_abi.jl` (runtests) pins each rule on synthetic metadata, the statements the generator emits, and the refusals. `test_sysv_live.jl` (devtests §12c) builds `test/sysv_live_test` as C++ at -O0 and -O2, one record per rule plus the ones that used to be refused and the controls, and a child calls each both ways with the SSE registers poisoned: 212 pass, the `long double` return traps as expected. `test_thunk_byte_regions.jl` asserts the classes of its unions and bitfield records and the statements on their calls. `test/tools/abi_fuzz.jl --seeds=1:60 --modes=cpp` on Linux: 1402 checks, 1368 pass, 0 fail, 34 expected refusals (`long double` scalars); the run of 2026-10-24 had 5 failures in 3 units. runtests.jl is green on Julia 1.13.2 (Arch, clang 23.1.3), and devtests.jl gives 3580 pass, 0 broken: the 47 broken of 2026-10-24 are gone.
### Tier 3: over-aligned records passed by value, and the `[wrap.varargs]` guards (2026-10-25)

- **A MEMORY-class record C passes in a 16-byte-aligned stack slot landed one slot early (review W2, 2026-10-24).** `take2(Big24, D4A)`, with `D4A` an `aligned(16)` record of four doubles, returned 494.5 for 4883.5 on Linux: C reads the record at the next 16-aligned slot after the 24-byte argument, and Julia's `ccall` of the stand-in passes it `byval` with no alignment, which the X86 backend takes as `max(type, 8)` = 8. `take1(D4A)` and `take3(Big24, Big24, D4A)` passed because their slots happened to be 16-aligned. No error, a plausible value. GeneratorC's parameter traps stopped at 16 bytes and never read alignment; the 2026-10-18 alignment work reached return buffers only.
- **The compiler's own slot alignment is recorded at build time.** `Compiler._byval_definitions` reads each definition's `byval(%struct.T) align N` parameters off the linked IR, beside `_nounwind_definitions`, and `_mark_byval_params!` writes them to the function as `byval_params` (record name and alignment, in parameter order; an empty list for a definition with none, no key for a function the IR only declares or when there is no IR). DWARF cannot always say what C used: a `packed` record whose members sit at their natural offsets has the same DIEs as the unpacked one, `#pragma pack(8)` capping an `__int128` member likewise, and an alignment only a typedef carries is not in DWARF at all. Win64 definitions carry no `byval` (a record goes by reference to a copy), so the key is an empty list there and nothing changes. A fix computed at build time: an existing package gets it on a rebuild (delete `.replibuild_cache/project_hash`), not a re-wrap.
- **GeneratorC passes a 16-aligned slot through a carrier and refuses the rest.** `_c_byval_facts` matches the recorded parameters to the ones its own classifier calls MEMORY (the counts and names must agree, else no fact is attributed), and `_c_byval_param_shape` decides per parameter: Julia's own byval for a slot aligned 8 or less; for an exact 16 on a record over 16 bytes and a multiple of 16, the value's bytes go into `_RB_ByVal16{N}`, a struct of `<2 x double>` chunks whose LLVM type is 16-aligned, so the backend places the slot where C reads it (`_rb_byval16`, emitted once per wrapper that needs it, ahead of the functions); an ABI Safety Trap for everything else above 8, with the reason: a 32-aligned slot (Julia's allocas and heap are 16-aligned at most, and a `<4 x double>` carrier segfaults in the byval copy, measured), a size that is not a multiple of 16 (an alignment a typedef added), a 16-byte MEMORY record (a carrier that small travels in XMM), and metadata without `byval_params`, whose trap says a rebuild records it. Where the IR says 8 and DWARF says 16 (the `packed` case), the IR wins and the plain ccall stays. Win64 is untouched: `_c_record_abi()` gates the whole decision. Options weighed: a compiled C shim per function (the mechanism `[wrap.macros]` uses) needs the function set, which only the DWARF of the link it would have to precede can name, so a second compile and link plus the PE export rules; routing to Tier 2 gives the record byval alignment 8 today as well (`test_sysv_abi_oracle.jl`'s `:underaligned` family).
- **The `[wrap.varargs]` path takes the same verdict (review W5).** GeneratorC's VARARGS INTERCEPTION branch `continue`d before `blob_abi_offenders`, BRANCH 0's sret and the `long double` trap, so a variadic function whose fixed parameters held a packed or float-bearing record of 16 bytes or less got a plain `@ccall` with the record spelled by name, and one returning such a record read it from registers C never wrote. `_c_byvalue_verdict` is now the one derivation for both paths (the register-window refusals, the stack slots and their carriers, `long double`, and BRANCH 0's return shape); `generate_vararg_wrappers` takes `trap` (the stub body, under every name the function would define) and `sret` (the explicit, optionally aligned, buffer ahead of the fixed arguments and the `;`), and a per-parameter `byval_carrier`. Its fixed parameters go through `_ccall_param_type` as the plain path's do. GeneratorCpp's varargs branch passes the `long double` trap too.
- **Tests.** `test_c_sret_classification.jl` (runtests): tables for `_c_byval_param_shape` and `_c_byval_facts`, `_byval_definitions` on a synthetic module (a quoted name, llvm-link's `.N`, a coerced register-class record, sret, a `byval` without `align`, a declaration), and the emission form of each over-aligned argument with and without facts, for both conventions. `test_varargs_emission.jl` (runtests): one variadic function per shape (packed, float blob, `long double`, 16- and 32-aligned, plain and aligned sret), both conventions, every call site still variadic. `test_c_sret_abi.jl` (devtests): `take_*` for all eight over-aligned records after a 24-byte stack argument, every argument member and the return checked live in a child, the wrapper's shape, and the build's `byval_params` against clang's lowering of the fixture for both targets. Negative checks, each with the exact method signature: `_c_byval_param_shape` answering `:own` for every input turns the live `take_d4a` into the wrong value and the form tables red; `_byval_definitions` returning an empty table turns every carrier into the no-record refusal; `_x87_crossings` returning nothing makes the varargs `long double` stubs plain calls.

### Template parameters parse from GNU binutils dumps (2026-10-25)

- **`template_params` was empty for every instantiation on Linux.** The readelf parser matched `DW_TAG_template_type_parameter` and `DW_TAG_template_value_parameter`, llvm-dwarfdump's spelling; GNU binutils writes `DW_TAG_template_type_param` and `DW_TAG_template_value_param`, so the match never hit. It now matches the short spelling, which is a prefix of both. `S<int, 3>` from a real readelf dump gives `T = int` and `N = int, 3`. Metadata only: nothing in the generators reads `template_params` yet. `test_dwarf_attribution.jl` parses the readelf dump and the same dump respelled the llvm-dwarfdump way.
### The registry's dirty digest follows the generator's inputs (2026-10-25)

- **Every note in CLAUDE.md re-vendored every build.** `hash_config` mixes in `_generator_fingerprint`, which hashes the checkout's uncommitted state so a generator edit cannot be served from a stale vendored build. It hashed the whole tree, and a dev checkout is dirty in CLAUDE.md or CHANGELOG.md most of the time, so `use` missed its cache after each saved sentence. The digest now covers `src/` and `Project.toml` (`_GENERATOR_DIRTY_PATHS`): tracked diffs and untracked files there, nothing elsewhere. `test_registry.jl`: a scratch git repo with edits to CLAUDE.md, docs/ and an untracked note keeps an empty digest; a source edit, an untracked source file and a Project.toml edit each change it. Widening the paths back to the whole tree fails the first assertion.
### `const void` is a type, not an unknown (2026-10-25)

- **Every `const void *` parameter and return had c_type `unknown*`.** clang writes `const void *` as a pointer to a `DW_TAG_const_type` with no `DW_AT_type`, since `void` has no DIE. `resolve_type` already answered `void*` for a target-less pointer and `void` for a target-less `volatile`/`restrict`/`_Atomic`, but `unknown` for a target-less `const`. llama.cpp's metadata carried 1257 of them. The julia_type was right (`Ptr{Cvoid}`, the mappers strip `const` and see `void*`), so calls were correct; the metadata and the callback signatures built from it were misspelled. A target-less const now resolves to `const void`. `test_dwarf_attribution.jl`: a synthetic dump with `const void *`, `const volatile void *` and `const char *` parameters and a `const void *` return; nothing about the function may be `unknown`.
### Header enums come from the project's headers; a scaffold includes the wrapper file wrap writes (2026-10-25)

- **The Clang.jl enum walk took every `EnumDecl` in the translation unit.** A project header that includes `<system_error>` or `<charconv>` brought libstdc++'s `errc` and `_Integer_kind` into the wrapper as "supplementary" enums, and which ones depended on the libclang Clang.jl ships for the running Julia (LLVM 18 on 1.12, 20 on 1.13): pugixml's wrapper differed by 101 lines between the two Julias, with the thunk manifest identical. `_extract_enums_via_ast!` now keeps only declarations whose file lies under the directory of a named header or under an include dir (normalised; case-folded on Windows). `test_header_enum_provenance.jl` (runtests): a header that includes a foreign header by absolute path and a second one through an include dir keeps `Color`, `Shared` and a namespaced `enum class`, and drops `Foreign`; no system header is involved, so the check does not depend on libstdc++ being found. Neutering the gate makes `Foreign` appear.
- **`scaffold_package` wrote a TOML whose module name did not match the file its module included.** `[project] name` was the package name lowercased, so wrap derived `Scaffoldprobe` and wrote `julia/Scaffoldprobe.jl`, while `src/ScaffoldProbe.jl` included `julia/ScaffoldProbe.jl` and warned "wrapper not built yet" on every case-sensitive filesystem; Windows folds case, which is how the release missed it. The same TOML wrote `root` as the absolute package path in a basic string, which on Windows holds `\a`-style escapes and does not parse; it is now `"."`, the TOML's own folder, as `discover` writes it, so the scaffold is also relocatable. The blank TOML now pins `[wrap] module_name` to the package name, and the scaffolded module asks the TOML for the file through the new `ConfigurationManager.wrapper_module_name(toml_path)`, so a registry TOML that names the library (`lua`, inside `LuaWrapper`) includes `julia/Lua.jl`. `get_module_name` and `wrapper_module_name` share one derivation, `module_name_for`. `test_registry.jl`'s scaffold testset checks the pinned key, both derivations and the module text; neutering the pin fails it.
### The SysV ABI oracle builds against a dylib-only LLVM, and runs on Linux (2026-10-24)

- **Arch ships no `libLLVMABI.a`; its `libLLVM.so.23.1` carries `llvm::abi`.** The dialect's CMake looked for the archive alone, so on Arch (llvm, mlir and clang 23.1.3) libJLCS built with the oracle off and `test_sysv_abi_oracle.jl` skipped, while Linux CI (apt.llvm.org, archive shipped) ran it. With no archive and `LLVM_LINK_LLVM_DYLIB`, CMake now compiles and links the same probe against the `LLVM` imported target (`check_cxx_source_compiles` with `CMAKE_REQUIRED_LIBRARIES LLVM`), keyed on the LLVM version like the archive check. Resolved, the oracle is compiled in and nothing new is linked: libJLCS's `NEEDED` set is unchanged. The archive path is as before, and a dylib without the component gives the same libJLCS as before with the test skipping. A plain `./build.sh` in the existing build directory picks the change up.
- **First run on Linux.** On an Arch box (Julia 1.13.2 and 1.12.7; the host layout has Itanium bitfields and an 8-byte `long`) the corpus is 300 records and 907 functions (48 refused on SysV); 1108 crossings compared, 42 disagree, every one in a known family, so the test passes with 42 `@test_broken`. Windows had 47 of 1097 with MS bitfields.
- **The disagreements are wrong values in live calls.** A scratch C++ fixture with one record per family and C++-linkage `take_R(R)`/`make_R()` (not noexcept, so each call takes a Tier 2 thunk), the callee at -O0 and again at -O2, the XMM registers poisoned before each call and padding masked out of the comparison: every `int_for_sse` record comes back with garbage in its float eightbyte, and four of the seven are wrong as arguments (`{float; double}`, `{int : 32; float}`, a nested `{double; float}`, `_Complex float`; the other three pass only because Julia's copy of the value leaves the second eightbyte in XMM1). `{char; {short; float}}` packed, a union holding a packed record, `#pragma pack(1) {char; int : 31}`, `long double` and `_Complex double` are wrong in both directions. The two `underaligned` records (`{__int128; long long}`, `aligned(16) {long long ×3}`) pass at both levels: that family needs a callee that loads its byval slot with an aligned instruction. The agreeing controls pass. The fixture is kept in the gitignored `.claude/sysv-live-2026-10-24/`.
- **The fuzzer sees the same defects on Linux.** `test/tools/abi_fuzz.jl --seeds=1:60` (the oracle's seed range) on the same box: 120 units built and loaded, every C (Tier 3) check passes, and 5 checks in 3 C++ (Tier 2) units fail. Seeds 21 and 22 segfault: a packed `{char; {enum}}` record, 5 bytes with the enum at offset 1, is MEMORY class to C and a register to JLCS (`reg_for_mem`), so in `f2(R3, R1, short, void*, R2, int, const R3*, …)` every later argument shifts and the callee dereferences a shifted pointer, and in `T2 f3(int)` the callee writes its result through what it takes for the hidden return pointer. Seed 24 is `int_for_sse` with the cascade: `{int[1]; {unsigned char : 1; float}}` crosses as two integer eightbytes, the callee reads the next argument as the float and a string pointer as the one after. The same seeds pass on Win64, whose classifier crosses these records by size. Output and the minimised repros are kept in the gitignored `.claude/abi-fuzz-linux-2026-10-24/`.
- **Tier 1 binds a slice's public declares through the process, and Julia ships zlib.** Rebuilding the Hub's zlib with 4.3.0 on Linux and running its `test_deep.jl` aborts in the gz file API with `double free or corruption (!prev)`: gdb shows `gzclose_w` executing from Julia's own `lib/julia/libz.so.1` (zlib 1.3.2), called from a Tier 1 slice of the Hub's zlib 1.3.1. The slice declares the public name, ORC resolves declares process-wide, and Julia's libz was loaded first. Static promotion covers internal symbols only; most sliced zlib functions declare another public zlib function (`deflate`, `deflateEnd`, `inflate`, `deflateInit2_`, `gzclose_w`, …). The same recipe passes on Windows, which has no process-wide symbol namespace. Not a 4.3.0 regression: the 4.1.0 wrapper routes the same functions to Tier 1. Julia's crash reporter then deadlocks allocating on the corrupted heap, so the process looks hung rather than crashed. Fixed the next day, together with the native half of the same binding (the library's own PLT calls were interposed too): see "A Tier 1 slice binds its library's own symbols by alias, never by public name" below.
- **Hub subset on Linux, 4.3.0.** cjson, md4c, expat, stl, zlib, mpack, pugixml, lua, clipper2 and box2d rebuilt in place from their vendored clones with `project_hash` removed (every build re-extracted, no cache hits, 1 to 2 minutes each on the N150) and verified: all pass except zlib, above. clipper2's wrap compiled `libclipper2_thunks.so` from its manifest in one pass. `register` of cjson into an isolated `REPLIBUILD_HOME` and environment with Pkg offline, `use` twice, and `using Cjson` from that environment in a fresh Julia all work. Relinking cjson gives the same `LIBRARY_SHA`. **One determinism miss:** re-wrapping pugixml on Julia 1.12.7 after 1.13.2 differs by 101 lines, all in the "supplementary enums from headers" block, where 1.13.2 emits libstdc++'s `errc` and `_Integer_kind` and 1.12.7 does not: `_extract_enums_via_ast!` has no header-location filter and Clang.jl's libclang is Julia's LLVM generation (18 vs 20). The thunk manifest is identical.
- **`abi_fuzz.jl` no longer registers its units into the default Julia environment.** It set `REPLIBUILD_HOME` to its work directory but not `REPLIBUILD_JULIA_ENV`, and since 4.3.0 `discover(build=true, wrap=true)` registers the project, which `Pkg.develop`s it into `@v1.13`. One `--seeds=1:60` run left 180 path deps pointing into a tmpfs scratch directory in the user's default environment, found by the commit review and removed with `Pkg.rm`. The tool now sets `REPLIBUILD_JULIA_ENV=none`, as the test suites do.
- **Over-aligned by-value parameters through Tier 3 are wrong on SysV after another stack argument** (review finding, run live). A C `take2(Big24, D4A)`, `D4A` an `aligned(16)` record of four doubles, returns 494.5 for 4883.5 on Linux: C reads the record at the next 16-aligned stack slot after the 24-byte `Big24`, Julia's ccall of the generated stand-in passes it `byval` with alignment 8, one slot earlier. `take1(D4A)` and `take3(Big24, Big24, D4A)` pass because their slots happen to be 16-aligned. The 2026-10-18 fix gave hidden return buffers the record's alignment; GeneratorC's parameter traps stop at 16 bytes and never read alignment. Fixture in the gitignored `.claude/albig-2026-10-24/`. Also from the same commit review, two Linux-only gaps recorded in CLAUDE.md: `scaffold_package` writes the project name lowercased so the scaffold includes a wrapper file name that only a case-folding filesystem matches, and the `[wrap.varargs]` path skips every SysV by-value guard.
### Parallel builds no longer rewrite the caller's environment (2026-10-24; found 2026-10-02)

`BuildBridge.run_command` ran every tool inside `LLVMEnvironment.with_llvm_env`.
That function saved the toolchain's keys from the process-global `ENV`, set them,
ran the tool and restored them. The per-TU IR compile calls `run_command` under
`Threads.@threads`, and the per-TU object compile under `asyncmap`. Overlapping
save/restore cycles do not nest. Say A saves the user's `LD_LIBRARY_PATH` and
sets `""` (a system toolchain force-clears it). B saves `""`, A restores the
user's value, and then B restores `""` by deleting the variable. After a parallel
C++ build the user's `LD_LIBRARY_PATH` and `CPATH` were gone, `PATH` kept the LLVM
bin prefix, and `LLVM_ROOT`/`LLVM_DIR`/`Clang_DIR` stayed set. Every later
subprocess inherited that. With `JULIA_NUM_THREADS=auto`, `stress_test`'s three
TUs were enough. In two of three parallel devtests runs, the `verify` child lost
the `LD_LIBRARY_PATH` that libJLCS needed and died on `libLLVM.so.22.1`.
Repeated discover+build of `stress_test` with sentinel values corrupted builds
2 and 3 of 3 on the old code, and 0 of 6 with the fix.

- The environment rides on the command. `LLVMEnvironment.llvm_cmd(cmd)` merges
  `toolchain.env_vars` over the command's own environment (or a copy of `ENV`)
  with `setenv`. This process's `ENV` is not written. `run_command`, `run_tool`
  and Discovery's include scan use it; `ASTWalker` takes a new `prepare_cmd`
  keyword for that. A key mapped to `""` still reaches the child set and empty.
  A bare command name still resolves against the toolchain `PATH`, because libuv
  searches the child's environment.
- On Windows the merge is case-insensitive. An inherited `Path` or `CLANG_DIR` is
  dropped before `PATH`/`Clang_DIR` is added. `addenv` would hand the child both
  spellings.
- `with_llvm_env` stays exported, with no caller in `src/`. It is serialized on a
  lock, so overlapping calls queue instead of corrupting each other. A variable
  that was set to `""` is now restored as `""`; it used to be deleted.
- `test_llvm_env_isolation.jl` (CI) builds a toolchain with the real
  `build_environment_vars`, runs julia children that print what they received,
  and samples `ENV` while overlapping `BuildBridge.execute` calls run in both
  concurrent shapes. Run against the old tree, the sampler fails in both shapes on
  every run. Each part of the fix was also neutered alone, and its assertions
  failed. An AST guard refuses a call to `with_llvm_env` anywhere in `src/`; it
  found three in the old tree.
### A Tier 1 slice binds its library's own symbols by alias, never by public name (2026-10-24)

- **The defect.** A slice is declarations-only, ORC resolves a `declare` process-wide, and the pre-flight (`_tier1_preflight!`) only asks whether the library *has* each symbol. Static promotion covered internal and hidden symbols, but a slice that called another **public** function of its own library declared that public name, and the process gave it the first library that defined it. Julia loads its own `lib/julia/libz.so.1` (zlib 1.3.2) at startup, before any wrapper's library. The Hub's zlib (1.3.1, `[wrap.tier1] enable = true`) ran `gzclose` as a slice whose `gzclose_w` was Julia's: it freed a `gz_state` the Hub's zlib had allocated with another layout, and `packages/zlib/test_deep.jl` aborted with `double free or corruption (!prev)`. 24 of the 45 sliced zlib functions declared a name Julia's libz also exports (`deflate`, `deflateEnd`, `inflate`, `deflateInit2_`, `gzclose_w`, `adler32_z`, `crc32_z`, …). Julia bundles zlib, zstd, pcre2, curl, ssh2, git2, mbedtls/openssl, gmp, mpfr, openlibm and nghttp2, so a Tier 1 recipe for any of those was wrong on Linux. Windows has no process-wide namespace and did not show it. Julia's crash reporter then deadlocks in `malloc` on the corrupted heap, so the failing process looked hung in a futex wait with no CPU; gdb with `handle SIGSEGV nostop noprint pass` shows the abort.
- **The fix, build half.** With `[wrap.tier1] enable`, `_promote_statics_libllvm` runs a third pass, `_alias_exports!`: every exported definition — external linkage, default visibility, a function or a global — gets the exported LLVM alias `__rb_<lib>_<name>` on the same address, recorded as `public name => alias` in `exported_aliases.json` and the metadata's `exported_aliases`. Not aliased, and therefore declared by name as before: declarations (the library does not define them), weak and `available_externally` definitions (a slice reaching a weak symbol is hazard-demoted), `llvm.*`, names already promoted to `__rb_*`, and thread-local globals (the C API cannot declare an alias for one). A `dllexport` aliasee gives a `dllexport` alias, so on PE the alias reaches the export directory whenever the aliasee does; `_pe_export_intent`'s decision is unchanged because the alias carries `dllexport` only where the module already had one. The rename map (`promoted_symbols`) is what it was, and a tier1-off build is byte-for-byte what it was: no alias, same dynamic symbol table, same `LIBRARY_SHA`.
- **The fix, slice half.** `Slicer._extract!` keeps its signature and first calls `_claim_rb_aliases!` on the clone: it reads the `__rb_*` aliases into `aliasee => alias`, moves each out of the way (a scratch name, internal linkage, so `globaldce` removes it, since the C API cannot erase an alias) and internalizes every other alias too, so no alias ever reaches a slice. The closure walk now demands an alias for every reached exported definition (`_is_exported_definition`: a function or global with external linkage and default visibility that is not `__rb_*`) and records the rename; after the bodies are deleted and the initializers dropped, each such declaration takes its alias name. A reached exported definition with no alias refuses the slice, with a reason that names the rebuild (`delete .replibuild_cache/project_hash`); it is a verdict, cached like any refusal, never a `:slicer_error`. `SLICER_VERSION` is 6: a v5 entry declares public names.
- **An alias shares its aliasee's address, and symbol extraction keyed `nm` by address.** `extract_symbols_from_binary` builds `address => mangled name` from `nm -g` and let the last line win, and `nm` sorts by the locale's collation, so on this box `__rb_zcollide_gzclose_w` displaced `gzclose_w` while `zc_verbosity` displaced its alias: five of the fixture's eight functions vanished from the metadata, and which five depended on the host's locale. A `__rb_` name no longer displaces another name at its address, and the aliases stay filtered out of the API as before.
- **Existing Tier 1 builds relink once.** The project-hash tag is `replibuild-project-hash-v3`: a v2 build's promoted module has no aliases, and a wrap over it would refuse every slice that reaches a public definition (correct, ccall, with warnings). The bump rebuilds every project once after the upgrade, which for an unchanged project is the link stage only (the per-file IR cache is keyed separately); `use` of a registered package reaches the same rebuild through `hash_config`.
- **Regression test: `test/test_tier1_alias_binding.jl` (devtests §7e, the one Tier 1 test devtests runs, by John's call).** Synthetic IR first: the alias set is exactly the exported definitions (the thread-local, weak, declared, hidden and internal cases all excluded, `dllexport` mirrored), a sliced `api` declares `__rb_probe_helper`, `__rb_probe_pub_state`, `__rb_probe_pub_tab` and keeps `free_like` by name, and a module promoted without aliases refuses rather than errors. Live: `test/tier1_alias_test/` names its public functions after zlib's exports (`gzclose_w`, `gzclose`, `gzclose_r`, `adler32_z`, `adler32`) plus public data (`z_verbose`, `z_errmsg`), each returning a sentinel; a decoy library with every one of those names and other values is `dlopen`ed `RTLD_GLOBAL` before the wrapper, in a child process. Every function takes Tier 1, every `TIER1_DECLARES` entry is a `__rb_zcollide_` name, every call returns the fixture's sentinel, and `code_llvm` of the `gzclose` kernel calls `__rb_zcollide_gzclose_w`. Two lessons the fixture had to learn: a leaf returning a literal is constant-propagated into its callers at `-O2` (IPSCCP, `noinline` notwithstanding) and the caller has nothing left to declare, so the leaves read a `zc_seed` global; and an exported mutable global is reached through the GOT, so an earlier `RTLD_GLOBAL` definition interposes it for the library's *native* code as well, which is ELF and not the subject, so `zc_seed` is hidden (zlib's `-DHAVE_HIDDEN` shape) and promotion renames it. `test_slicer.jl`'s fan-out hub now declares `__rb_slicetest_st_sc_mid_N`, and `test_static_promotion.jl` checks the alias map against the dynamic symbol table. Negative-checked by neutering each half with its exact method signature inside its module and running the file: `_is_exported_definition(gv) = false` in `Slicer` fails 12 assertions of the synthetic alias-declare testset (public names declared); `_alias_exports!(mod, lib_token::String, aliases::Dict{String,String}) = aliases` in `Compiler` fails 5 of the exported-aliases testset (no alias in the module); `_self_binding_link_flags(; windows::Bool = Sys.iswindows()) = String[]` fails the five native probes and the `SYMBOLIC` check (the decoy's and Julia's copies answer the library's own calls); the one-line `address_to_mangled[address] = mangled` of old fails 28, starting with `dwarf: 3 functions` instead of 9. A first run of the negative script appended each override *after* `end # module`, which defines it in the parent module and neuters nothing: every case passed, which is the vacuous green the rule warns about. The three experimental suites and this file are green on Julia 1.13.2 and 1.12; runtests and devtests (59 testsets, none skipped) are green on 1.13.2.
- **The same defect has a second route, and the alias fix alone did not make the Hub's zlib pass.** Rebuilt with the aliases, `test_deep.jl` still died of `double free or corruption`. `LD_DEBUG=bindings` on the rebuilt `libzlib.so` showed why: the library's **own native** references to `gzclose_w`, `deflate`, `inflate`, `deflateEnd` and `adler32_z` bound to Julia's `libz.so.1`. On ELF a library's call to its own exported function goes through the PLT and its load of an exported global through the GOT, and the dynamic linker resolves both through the global scope before the library itself (symbol interposition), so the Hub zlib's native `gzclose` handed Julia's `gzclose_w` a `gz_state` the other zlib laid out. The slice was the route gdb happened to catch; the native route was there all along and is why the gz layer, whose state is private, broke while the stream API, whose `z_stream` is public and stable across the two versions, worked by accident. `create_library` now links every library `-Wl,-Bsymbolic` on ELF (`_self_binding_link_flags`; nothing on PE, where a DLL never goes through its import table for its own symbols): the rebuilt zlib's dynamic section carries `SYMBOLIC` and the only dynamic relocations left are libc's. Both halves are needed: `-Bsymbolic` binds references *inside* the library, and a slice is code outside it whose `declare` still goes through ORC's process lookup, which the alias answers. Nothing RepliBuild builds is linked against by an executable, so the data half of `-Bsymbolic` has no copy relocation to split. `test_tier1_alias_binding.jl` checks the native half too: with the decoy loaded first, a plain `ccall` of the fixture's `gzclose` reaches its own `gzclose_w`, and its native reads of `z_verbose` see the value a Tier 1 write stored. `test_codegen_flags.jl` drives both spellings of the flag.
- **Live.** The Hub's zlib rebuilt in place on Linux (`project_hash` removed; `tier1: 118 slices accepted (5 refused, 0 errored, 0 hazard-gated, 1 oversize, 0 unresolved-symbol)`, 112 emitted for 107 functions, every `TIER1_DECLARES` entry for a zlib symbol a `__rb_zlib_` name) and `packages/zlib/test_deep.jl` passes, 271 of 271, including the gz file API and the Tier 1 liveness testset. The machine reset once mid-run (the N150's known fault); the committed half survived, the scratch scripts did not.

### The SysV struct classifier, diffed against LLVM's ABI library (2026-10-24)

- **A second opinion for `classifySysVStruct` and `buildCallShape`.** LLVM 23 ships an ABI lowering library (`llvm/ABI`) with an x86-64 SysV classifier written to match clang. `test/test_sysv_abi_oracle.jl` (devtests §12b) asks it and the dialect about the same records and lists every disagreement. The corpus is one C unit: the records and functions `test/tools/abi_fuzz` generates for 60 seeds, 38 hand-written records for what the generator never makes (`long double`, `__int128`, `_Complex`, `aligned(16)`, unnamed bitfields, an empty record), and `take_R(R)`/`make_R(void)` for each record. It is built once as C, and both sides read the records from the build's DWARF metadata: about 1100 by-value crossings in 900 functions.
- **Two test seams in libJLCS.** `jlcs_probe_sysv_call_shapes` runs the real `buildCallShape` with `AbiTarget::SysV` over every `jlcs.ffe_call`/`jlcs.try_call` in a module, on any host, so the Windows dev box checks the Linux convention. It is fed the module `generate_jlcs_ir` writes for the metadata, so StructGen's and FunctionGen's choices are part of what is checked. The lowering itself is unchanged and still takes the host's ABI. `jlcs_abi_oracle_sysv` (`src/mlir/impl/JLCSAbiOracle.cpp`) builds the records as `llvm::abi` types and runs `computeInfo`. The record's alignment is DW_AT_alignment where clang writes it (it writes the actual alignment whenever `aligned` or `#pragma pack` applies), else the members' alignment lowered to what the layout shows.
- **The oracle is checked against clang.** For each hand-written record, its answers equal clang's own `x86_64-unknown-linux-gnu` lowering of the C definition (byval and sret alignment included).
- **Optional, LLVM 23+.** CMake compiles the oracle only when it finds the ABI library, by compiling the calls it makes rather than reading a version (22 has the type system alone, 21 no library). It links the static `libLLVMABI.a` file, not the `LLVMABI` target, whose static Core and Support would sit beside the LLVM dylib. Elsewhere libJLCS builds exactly as before and the test skips (`:llvm_abi` in `TestSupport.have`). Checked with a build that forces the check off.
- **47 of 1097 crossings disagree on Windows. These are SysV defects in Tier 2, not fixed here.** Each family is `@test_broken`, one per crossing, with a hand-written repro, and the report prints each record's C definition and a minimal record:
  - *Padding is INTEGER.* `classifySysVStruct` counts StructGen's explicit pad arrays as INTEGER, so an eightbyte holding a float and padding goes in a general register. `{float; double}` crosses in RDI and XMM0 where C uses XMM0 and XMM1, and so do `{double; float}`, `{void*; float}`, `{int : 32; float}`, `aligned(16) float`. A tiled (`is_struct_packed`) body spells a nested record that is not tiled as bytes, which hides its floats the same way (`struct { struct { double; float; } in; }`).
  - *Misaligned fields hidden in bytes.* A nested record in a tiled body, an enum inside a packed literal, and a union's members are bytes to the classifier. A packed record holding one of them at an odd offset crosses in registers where C uses memory.
  - *A bitfield's storage unit tiled at its byte offset* reads as a misaligned field: `#pragma pack(1) { char c; int b : 31; }` goes in memory where C passes one register (`i40`).
  - *`long double` and `_Complex` members map to `!llvm.ptr`.* A tiled 16-byte `{ long double x; }` or `{ _Complex double z; }` emits as 8 bytes, against the rule that an emitted struct is its DWARF size, and crosses in RAX/RDI where C uses memory and st0, or XMM0 and XMM1. FunctionGen does not trap these, although `long double` by itself is an ABI trap. The size is wrong on Win64 too, where the 8-byte body goes in a register and C passes the 16-byte record by reference.
  - *Over-alignment.* A 16-aligned record (`aligned(16)`, `__int128`) gets byval and sret alignment 8; C gives 16, and byval alignment places the stack slot.
- **Tests.** A disagreement in no family fails, and so does one that appears only inside a call whose record agrees alone (register accounting). Negative-checked against two dialect builds: one with no SSE class (the hand-written SSE records and the harness go red) and one without the "does not fit, so memory" rule (the spill guards and 17 in-call disagreements go red). `test/tools/abi_fuzz/model.jl` exports `emit_enum` and `emit_record`, which the corpus uses. On Windows (LLVM/MLIR 23.1.3) the test passes on Julia 1.13.2 and 1.12.7, runtests.jl is green, and devtests.jl gives 3264 pass and 48 broken: these 47 and the one already broken on main.

### `Pkg.build` builds the JLCS dialect on Windows (2026-10-23)

- **Every RepliBuild install on Windows started without Tier 2, and said so quietly.** `deps/build.jl` never tried to build on Windows: build.sh needs an MSYS2 CLANG64 shell, which it took `Pkg.build` not to be. Each version installs into a new depot folder, so every upgrade also lost the dialect. The first C++ wrapper to load then logged "JLCS dialect library not found", and `check_environment()` showed a single yellow WARN among green OKs on a box where every tool the build needs had passed.
- **MSYS2's bash is a CLANG64 shell from any process.** Started as a login shell with `MSYSTEM=CLANG64`, its `/etc/profile` puts `clang64/bin` first on PATH. `deps/build.jl` now runs build.sh that way: `<MSYS2_ROOT>/usr/bin/bash.exe -l <src/mlir/build.sh>`, with `MSYS2_ROOT` defaulting to `C:/msys64`. On the reference box a cold build takes about 15 seconds, and a folder with a space in its path builds too. As on Linux, it never throws: no MSYS2, or a CLANG64 without cmake, `mlir-tblgen` or `llvm-config`, gives the warning and a working Tier 3 install. A library that built but will not load because `clang64\bin` is not on PATH is reported as that.
- **A minor LLVM upgrade now rebuilds the library.** `libLLVM-<major>.dll` is named by major alone, so a library built against another minor still loads, and a loadability check alone passed it. `deps/build.jl` compares build.sh's `jlcs_llvm_version` stamp with CLANG64's `llvm-config`, as the doctor does.
- **The advice is runnable from where the user is.** The doctor's libJLCS advice, which `MLIRNative.check_library`'s error shares, now gives `using Pkg; Pkg.build("RepliBuild")`, and a one-line PowerShell command that needs no MSYS2 shell: `C:/msys64/usr/bin/env.exe MSYSTEM=CLANG64 /usr/bin/bash -l '<…>/src/mlir/build.sh'`. It also says to close Julia sessions before a rebuild, because Windows will not overwrite a loaded DLL (the link fails with "Permission denied"). Both `deps/build.jl` and the doctor take the command from `src/mlir/windows_build.jl`, one copy for both. When the dialect is the only Tier 2 piece missing, the summary says "Tier 2 (MLIR thunks) not built yet" and prints the `Pkg.build` line.
- **Tests.** `test_user_messages.jl` (runtests): the command and the PowerShell line, built from a Windows path with a space under a non-default root; `MSYS2_ROOT` normalisation; the "not built yet" summary when only the library is missing, but not when it is stale or a tool is missing; and `deps/build.jl` runs the MSYS2 command. The Windows branch of `deps/build.jl` was run by hand on a scratch copy: a cold build, an already-built library, a stale stamp, a folder with a space, and a bogus `MSYS2_ROOT`, and the printed PowerShell line was run from Windows PowerShell 5.1.

## v4.3.0 (2026-10-23)

**The registry builds a project where it lives; AOT thunks come from wrap.**
Minor, because two public calls change what they do. `register(toml)` builds
and wraps the project in place, vendors the build to
`~/.replibuild/builds/<hash>/`, and adds a Julia package around it to a Julia
environment, so `using MyProject` works; `use` follows edits by rebuilding in
place. A bare `discover` no longer registers anything. With `[compile]
aot_thunks = true`, `wrap` compiles the thunks library for exactly the thunks
it binds, in one pass, and `build` no longer compiles thunks. Also in this
release: the ABI fixes the differential fuzzer found (Tier 3 records return
where the target says, through a buffer with the record's alignment; Tier 2
unions and bitfield records cross by value as their bytes, and `#pragma pack`
bodies stay packed), and switching Julia or the C++ toolchain rebuilds what
the switch affects. v4.2.1 was never registered, so its changes (LLVM.jl 10,
Julia 1.13 in the support window, Tier 1 on Windows) ship here too.

Merged after the release commit and registered with it (the first six entries
below: PRs #25 to #29, and the `scaffold_package` compat): the same source wraps
the same on Julia 1.12 and 1.13 and after an upstream edit, and lambda closure
types are no longer emitted; a call that passes by value a type whose name
several types share is an ABI trap; function-pointer types carry their
parameter lists, and a callback's `@cfunction` example comes from its own
signature; Windows links are reproducible, and a link longer than one Windows
command line goes through a response file; `scaffold_package` writes a
RepliBuild compat that admits the RepliBuild that wrote it.

Upgrading from 4.2.0: rebuild, then re-wrap (`build(toml; clean=true)`, or
delete `.replibuild_cache/project_hash`), because the metadata now records
member sizes, record alignment, conflicted names and function-pointer
parameters. The first build also compiles every file once more, because IR
file names changed. A project registered by an earlier RepliBuild keeps
building the copy that version made; register it again, from the project.

Verified on Windows (MSYS2 CLANG64, LLVM/MLIR 23.1.3) on the release commit:
runtests.jl 2702 and devtests.jl 1331 plus 1 broken (already broken on main),
each on Julia 1.12.7 and 1.13.2, and the docs build. PR #23's CI ran Linux
devtests against LLVM/MLIR 22 and 23. The Hub was swept on Windows the same day:
all 26 packages at their old pins, then 18 upstream version bumps. imgui and
llamacpp were rebuilt for PR #22 from their previous manifests and finished
in one pass. hello_world, fmt, tinyxml2, pugixml and box2d were re-wrapped on
the release commit and pass. Each of PRs #25 to #29 was verified on its own
branch (its entry gives the counts) and passed its PR's CI, Linux devtests
against LLVM/MLIR 22 and 23 included; CI passed again on the final commit.

### `scaffold_package` writes a RepliBuild compat its package can resolve (2026-10-23)

- **A scaffolded package asked for RepliBuild 2.** `_project_toml` wrote `RepliBuild = "2"` into the generated `Project.toml`, a literal left from the 2.x line. Pkg therefore resolved the package against RepliBuild 2.x, the newest version that entry admits, not the RepliBuild whose `replibuild.toml` template and `deps/build.jl` it was given.
- **The entry is now `_replibuild_compat()`**: the caret spelling of the running RepliBuild's version, which the registry's own packages (`_write_package`) already wrote. Both take it from that one function.
- **Not covered; the scaffold is stale in other ways too.** Its `replibuild.toml` sets `root` to the absolute path of the scaffolded folder, so the package breaks when moved, and turns `[link] enable_lto` on, which is parked. Its `src/<Name>.jl` includes `julia/<Name>.jl`, but `wrap` names the file after `get_module_name` (`Scaffoldprobe.jl` for `ScaffoldProbe`), which only a case-insensitive filesystem finds, and the wrapper's module then nests inside the package's.
- **Tests.** `test_registry.jl` (runtests): a scaffolded `Project.toml` names RepliBuild's UUID, and its compat, read by `Pkg.Versions.semver_spec`, admits `RepliBuild.VERSION`, excludes 2.5.3 and the next major, and equals `_replibuild_compat()`. With `"2"` put back, the testset fails.

### The same IR links to the same library on Windows (2026-10-23)

- **Every relink of an unchanged project gave a different DLL.** A PE image holds its link time in two fields: the COFF file header's `TimeDateStamp`, and the debug directory entry lld writes when the image has debug info (its `.buildid` CodeView record). lld fills both from the clock. Two links of the same IR a second apart differed in those two fields and nowhere else. A wrapper's `LIBRARY_SHA` is the SHA-256 of the library file, so it changed with them: llamacpp's cached IR, relinked on another day, changed that one line of a 60336-line wrapper. A CI job that judges a Hub bump by diffing regenerated wrappers would see it on every run.
- **Every link of a shipped binary passes `Compiler._reproducible_link_flags()`**, which is `-Wl,--no-insert-timestamp` on Windows and nothing elsewhere. It writes 0 to both fields. It covers `create_library` (the IR link and the per-TU-object fallback), `create_executable`, and the AOT thunks library.
  - Both PE linkers take that spelling: CLANG64's lld 23.1.3, which the system clang++ and the C bucket's JLL clang both link with, and GNU ld 2.47 from MSYS2's MINGW64. lld-link's `/Brepro` writes a hash of the image there instead, and GNU ld has no such option.
  - lld's build ID is a hash of the image without those two fields, so it already repeated across relinks; the flag does not change it.
- **ELF needed nothing.** It holds no link time, and a build ID, when a linker writes one, is a hash of the output. lld cross-linking `x86_64-linux-gnu` on the Windows box gave identical files a second apart, with and without `--build-id`. Linux devtests (`tier2-linux.yml`) will run the new test against the real toolchains; it has not run there yet.
- **`LIBRARY_SHA` means what it did**: the SHA-256 of the library file, not a build ID. What changed is that it is stable. A rebuild that relinks the same IR keeps it, so it moves only when the library's bytes do. The bytes still move, correctly, when the project's path does (DWARF holds absolute paths), as well as with any change upstream of the link. The directory Julia is started from does not reach them: the IR's compile directory is the project root, and two cold builds from different working directories gave the same IR and the same library. The wrapper's other build-dependent lines are timestamps by name: `# Generated:` (the wrap) and `METADATA["generated_at"]` (the build).
- **Tests.**
  - test_reproducible_link.jl (devtests §9b): a C fixture (JLL clang, in-process link) and a C++ fixture (system clang++, llvm-link, opt) are each built, have their project hash deleted, and are built again in a later second from the same per-file IR. An executable is linked twice. The bytes must match, and on PE both link-time fields must be 0 (`TestSupport.pe_link_times`, which reads the image itself). With the flags neutered by their exact method signature, 9 of 21 fail, in all three fixtures; the files differed at exactly the two fields.
  - test_aot_thunks.jl: the thunks library that pass 2 links again from the same thunk set has pass 1's bytes and a 0 header timestamp, although a child process ran between the two links. Neutered, both checks fail.
  - test_codegen_flags.jl (runtests): both host spellings, and that each of the three link functions passes the flags. Removing the call from any one of them fails it.
  - The eight devtests fixtures that build a library (stress, mi, vi, stl, c, c_abomination, callback, aot), relinked from the per-file IR devtests had just cached: all eight identical, stl_test's regenerated templates TU included. With the flags neutered, all eight differ in the header's four bytes and the debug entry's.
  - runtests 2700 on Julia 1.12.7 and 2708 on 1.13.2; devtests 1354 plus 1 broken (already broken on main) on both. The new test and its negative check give the same results on both Julias. With the emission-determinism change below merged in: runtests 2768 and devtests 1395 plus 1 broken, on both Julias.

### Function-pointer types carry their parameter lists (2026-10-23)

Upgrading: rebuild (delete `.replibuild_cache/project_hash`) to get the signatures and the examples built from them. A re-wrap of older metadata gets only the generator fixes, and no example from DWARF.

- **No function pointer read from DWARF had parameters.** Pass 2 of `parse_dwarf_dump` reads formal parameters only inside an open subprogram, and it never opened a `DW_TAG_subroutine_type`'s context. Every function-pointer `c_type` was therefore `function_ptr(R)*`, parameters, members and globals alike: llama.cpp's four-argument `ggml_backend_eval_callback` was `function_ptr(bool)*`. Pass 1 now collects a subroutine type's parameter types in dump order: its direct-child formal parameters, then `...` for `DW_TAG_unspecified_parameters`. So `typedef bool (*eval_cb)(void *, bool, void *)` is `function_ptr(_Bool; void*, _Bool, void*)*`, and `int (*)(const char *, ...)` is `function_ptr(int; const char*, ...)*`. It is pass 1 because that pass builds the type table: clang emits a subroutine type after the first function that takes it, and pass 2 resolves each signature as it meets it, so a list collected there would be complete for signatures met after the type and empty for those met before it.
- **A signature's branches shared one cycle guard.** `resolve_type` passed one `visited` set to the return type and to every parameter, so a type met twice looked like a cycle: `int (*)(int, int)` came out `function_ptr(int; unknown, unknown)` as soon as there were parameters. Each branch now gets its own copy.
- **The suspected mis-attribution did not happen.** Pass 1 left its subroutine context on the dump's last subroutine type, and pass 2 started with it. The formal parameters of a subroutine type before the dump's first subprogram therefore recorded that last type as their parent. Nothing was pushed onto it, because pass 2 pushed a parameter only inside an open subprogram, which is also why no list ever had parameters. That pass-2 branch and its state are gone.
- **Consumers of the new form.** Every reader of `c_type` strings was checked for a function-pointer type with `;`, commas, brackets and nested parentheses inside its own parentheses. Tier routing, ccall and thunk types, sizes and the by-value record facts do not change: each of them tests for `*`, `&`, `(` or `function_ptr(` before reading anything inside the parentheses, so a function pointer is `Ptr{Cvoid}` or `!llvm.ptr` there. Four consumers read the whole string for something only its outer type could mean:
  - **`cpp_to_julia_type` took an array's extents from the first `[`.** Two callbacks that each take a pointer to an array, `function_ptr(void; int[4]*)*[2]`, became `NTuple{8, Ptr{Cvoid}}`. A struct with that member is a 16-byte field read and written as 64, a record that is no longer natural, and a global loaded at 64 bytes. The extents are now the trailing run, which is how `parse_dwarf_dump`'s member-size fallback already read them. `int (*p[2])[4]` became eight ints the same way, before any of this.
  - **GeneratorC's `_struct_has_float_member` matched any `float` in a member's type.** So an array of callbacks taking a `float` counted as a float member, which on SysV makes a small blob passed by value an ABI-trap stub. An array is now judged by its element, which is a pointer here. `float*[4]` was the same false positive before.
  - **The callback docstring's `(signature: …)` text is escaped.** A callback without an example (see below) is documented with DWARF's signature as text, which now names every parameter type, and it is written into a `"""` docstring, where `\` escapes and `$` interpolates.
  - **`surface_types_only` now reaches the types a callback's signature names.** The reachability walk takes every identifier in a `c_type`. Those types are the ones a user needs to write the callback, which is the class `surface_types_extra` exists for. What a callback receives as `void *` stays unreachable: that is md4c's `MD_*_DETAIL`. No Hub package sets the key.
- **A callback without a header typedef gets its `@cfunction` example from DWARF** (John's decision). The example came only from the header typedef the parameter is declared with. So a callback spelled out in the declaration, `int (*cb)(int)`, had none, and nor had one whose typedef the header scan cannot read.
  - `_dwarf_callback_signature` builds the example from the DWARF signature. It strips exactly one pointer and splits the lists at depth 0. Each type is spelled the way the generator spells a parameter's ccall type, through `_ccall_param_type`: one function per generator, which the parameter loop now calls too.
  - There is no example for a variadic callback, a pointer to a callback (an out-parameter) or a type the wrapper cannot spell. Those keep the signature as text.
  - Both generators now document parameters through one function, `_param_docs!`, which replaces two identical copies of the loop.
  - Metadata built before this change spells every signature `function_ptr(R)*`, and an example from it would document a four-argument callback as taking none. So the build now marks its metadata `function_pointer_parameters`, and without the mark there is no example from DWARF.
- **Measured.** After devtests, each of the 13 built fixtures was rebuilt from its IR cache (`project_hash` deleted) and wrapped twice, with the parent commit's code and with this change.
  - Metadata: 4 fixtures change and 9 are identical. 15 function-pointer parameters (`c_type` and `function_pointer_signature`) and 2 struct members gain their parameter lists: stress_test 10 parameters, callback_test 2, determinism_test 2, and c_abomination_test 1 plus the 2 members. Its parameter is a function pointer that returns one, `function_ptr(function_ptr(double; float)*; int)*`. `type_registry` keys follow the strings; nothing in `src` reads them. No `julia_type`, size, offset, global or typedef changes.
  - Wrappers: 12 of 13 are identical, so the parameter-step refactor changed no line. determinism_test's one callback without a header typedef, `dt_run_raw`'s, now has the example `@cfunction(my_callback, Cint, (Cint,))` where it had the text `(signature: function_ptr(int)*)`. Every other fixture callback is declared with a typedef the headers spell, and keeps its example.
  - llama.cpp at its Hub pin, in a copy, DWARF only (nothing rebuilt or wrapped): the tables `parse_dwarf_dump` builds from its DLL have 190 function-pointer entries. Before, none had a parameter list; now 173 do, and the other 17 take none. With the signature text masked, those 173 entries are identical before and after: their `julia_type`s, sizes, offsets and recorded typedefs did not change. `ggml_backend_eval_callback` is `function_ptr(bool; int, ggml_tensor*, ggml_tensor*, void*)*`. Of the API's 21 callback parameters, 3 record no typedef. One is `llama_set_abort_callback`'s, whose definition spells the type out; its DWARF signature, `function_ptr(bool; void*)*`, gives `@cfunction(my_callback, Bool, (Ptr{Cvoid},))`. The other two are out-parameters that point to a callback (`ggml_log_callback *`), and get no example.
- **Tests.**
  - `test_dwarf_attribution.jl` (devtests §14) has two new testsets. One parses a GNU objdump of a C unit that clang emitted with every subroutine type after its users. It covers a typedef'd callback (`void*` twice, around `_Bool`, which is also the return type), a variadic one, a raw `int (*)(int, int)`, and struct members including a function pointer that takes a function pointer. The other puts a subroutine type before the first subprogram and one after it, with a struct between them.
  - `test_function_pointer_types.jl` (runtests, new) covers the consumers: array extents, the float-member check, reachability through a callback, and the docstring text in both generators. It covers the DWARF examples too:
    - The structural rules, with a mapper that names every type.
    - In both generators, one signature with and without a header typedef gives one example.
    - A declared record, one declared under a sanitized name (`baremodule`, so `Ptr{c_baremodule}`), an enum, an undeclared pointee and a callback parameter are spelled correctly.
    - No example for a variadic callback, a pointer to a callback, a reference, or `long double`, nor from older metadata.
  - `test_emission_determinism.jl` and `test_determinism_fixture.jl` now expect DWARF's example for their callbacks without a typedef, and still never the decoy's.
  - Each fix was checked by neutering it: no pass-1 collection, a shared `visited` (on the parameters, and on the return), no `...`, a subroutine context that never closes, the first-`[` regex, the whole-string float check, either generator's unescaped text, and a reachability walk that skips function pointers. For the examples: no DWARF example at all, variadic allowed, any pointer level, a split that ignores `<>`, either generator's raw `julia_type` in place of its parameter step, GeneratorC without `_as_declared`, and no metadata guard. Each fails its testset.
- **Verified** on Windows (MSYS2 CLANG64, LLVM/MLIR 23.1.3):
  - runtests: 2833 on Julia 1.12.7 and on 1.13.2. That is the parent commit's 2754, plus 69 in the new file and 2 in `test_emission_determinism.jl`, plus 8 `Symbol hygiene` checks that run only when the fixture libraries are already built.
  - devtests: 1389 plus 1 broken on 1.12.7 and on 1.13.2. That is the parent commit's counts plus §14's 18 new assertions and one in §8d. The broken one is the C in-process pipeline test, already broken on main. `Debug: static inspection` counts the `.mlir` dumps earlier runs left on disk, so it moves by one between runs.
  - `test/tools/abi_fuzz.jl` was not run: its type model has no function pointers and no pointers to arrays, so none of its units reaches the changed code.

### A type whose name several types share no longer crosses a call by value (2026-10-23)

Upgrading: rebuild (delete `.replibuild_cache/project_hash`) to record each conflicted name's sizes. A re-wrap alone traps conflicted records already, and treats every conflicted enum as differing in width.

- **What was wrong.** Types are keyed by their bare DWARF name. Since the entry below, `parse_dwarf_dump` keeps one definition per name and marks it `conflict` when the definitions are distinct types. Every reference by that name gets the kept layout, and a by-value crossing got it too. `FunctionGen.by_value_crossing` answered from the struct table before it consulted `record_abi`, whose own `conflict` would have made the crossing a trap. GeneratorCpp's ccall passed the kept Julia struct, and GeneratorC never asked. A function taking llama.cpp's `llama_vocab::impl` by value would have been called with another `impl`'s layout.
- **Measured first, read-only.** Each Hub package's shipped library was re-parsed with the new parser, and every function in its existing metadata was checked, members and bases included. Across 26 packages (9644 wrapped functions, 1549 of them taking or returning a record by value) and the 13 devtests fixtures, no function takes, returns or holds by value a record whose name is conflicted.
  - The Hub has 13 conflicted names in 6 packages: llamacpp 7, tinyxml2 2 (`MemPoolT<N>::Block`, `Item`), fmt 1, cmark 1, and two enums, `Type` in box2d and `ContentsType` in imgui.
  - The one by-value use is box2d's `b2Contact::AddType`, which takes `b2Shape::Type`. All five `Type`s are 4 bytes.
  - None of the 13 names is reached through a pointer or reference parameter, a method's receiver or a global either.
  - A stricter comparison found no other conflicts. Across the fixtures and the Hub, 758 record names have several definitions. Adding each placed member's float, integer or pointer class to `_layout_signature` marks none beyond the 12 records it already marks.
  - With this change, the new verdict over the same 9990 Hub functions, against libraries re-parsed by the new parser, traps none. box2d's `Type` records sizes `[4]`, so `AddType` stays callable. No Hub wrapper changes, so none was rebuilt.
- **Now a by-value crossing of a conflicted record is an ABI trap, and so is one of a record that holds such a record by value at any depth** (a member, an array member, a base):
  - `by_value_crossing` returns `:conflict`. `conflicted_within` walks bases and placed members, and stops at a record that names itself as its base: llama.cpp's kept `graph` and `extra_buffer_type` do, since a derived class has its base's bare name. `thunk_crossings` gives the reason, the thunk generator writes `// no thunk for …`, and GeneratorCpp's BRANCH 1 emits its trap stub. The stub's message says why for this cause; a layout missing from DWARF keeps its old text, word for word.
  - `is_ccall_safe` sends such a C++ function to the thunk path, so a `noexcept` or `extern "C"` one reaches the same trap.
  - GeneratorC traps it before BRANCH 0, so it gets neither an sret buffer nor a convenience overload, both of which would ccall. The shared varargs path (`generate_vararg_wrappers`) continues past both generators' traps, so it traps too, under every name it would define.
  - The stub (`_conflict_trap_body`, `_CONFLICT_TRAP_NOTE`) raises:
    ```
    ABI Safety Trap: cannot call 'f'. It takes `impl` by value, and distinct records in this library are named `impl` (8 and 16 bytes). RepliBuild keys types by their bare DWARF name and keeps one definition per name, which need not be the one this function means, so no call could pass it correctly. Keep the function out with [wrap] exclude_symbols.
    ```
- **An enum traps only when its definitions differ in width** (John's decision). An enum crosses as an integer of its width, so box2d's five `Type`s, which differ only in their enumerators, still cross as one `i32` and `AddType` stays callable. The parser now records `conflict_sizes`, the distinct byte sizes of all the definitions, on every conflicted entry. An enum entry without it (metadata from the entry below) is treated as differing.
- Not covered: a `Ptr{T}` or `Ref{T}` parameter typed with a conflicted name, a method's receiver, a global, and a field of another record still get the kept layout on the Julia side. The measurement found no such use in the Hub. See Open problems in CLAUDE.md.
- **Tests.** Each fix was checked by neutering it.
  - `test_conflicted_names.jl` (runtests) uses a synthetic two-unit dump: two `impl`s, two 4-byte `Mode`s, and a 1-byte and a 4-byte `Kind`. It checks the sizes the parser records and every verdict: direct, qualified, a member, an array member, a base, nested, a self-based record, by address, the enum widths, and metadata without sizes. It also checks the thunk generator's skips, `is_ccall_safe`, and both generators' stubs (varargs included). A layout missing from DWARF must keep its message. All 10 neuters (parser, verdict ×2, walk ×2, routing, GeneratorC, varargs, enum width, the C++ message) fail it.
  - `test_conflict_trap_fixture.jl` (devtests §8e) builds `test/conflict_trap_test/` (two C++ units: `impl`, `Mode` and `Kind` in two namespaces) and `test/conflict_trap_c_test/` (two C files, two `struct state`s) from source. It checks the sizes in the real metadata, that each trap has no tier row and no thunk, and that every other function keeps its tier. A child process then calls everything. The traps raise; `Duo`, a pointer to an `impl` and `Mode` return the right values, through Tier 3 and through thunks. Neutering the parser, the verdict, the routing or GeneratorC fails it.
- **Verified** on Windows (MSYS2 CLANG64, LLVM/MLIR 23.1.3):
  - runtests: 2893 on Julia 1.12.7 and on 1.13.2. That is the entry below's 2754 plus this entry's 131, plus 8 that `test_symbol_hygiene.jl` asserts only once the mi_test and vi_test fixtures are built: a first 1.12.7 run in a fresh tree, before any devtests, gave 2885.
  - devtests: 1431 plus 1 broken on 1.12.7, 1430 plus 1 broken on 1.13.2. That is the entry below's 1371 and 1370 plus §8e's 60. The broken test is the C in-process pipeline, as before. The one count apart is still `Debug: static inspection`, which follows the `.mlir` dumps on disk.

### The same source wraps the same on every Julia release and after every upstream edit (2026-10-23)

Upgrading: rebuild (delete `.replibuild_cache/project_hash`) to get the DWARF changes. A re-wrap alone gets the generator's. A C++ package's wrapper changes once: the kept definition of a name that has several moves to the content-chosen one, and lambda closure types leave. The first build after the change also compiles every TU once more, because IR file names changed.

A dry run of an unattended llama.cpp bump (b11819 → b11828, nine upstream commits, no public header change) built, wrapped and passed `test_deep.jl` with 0 API changes, yet the wrapper had about 1000 changed lines. Re-running the same source on Julia 1.12.7 and 1.13.2 separated the causes. A CI job that judges a bump by diffing the wrapper could not have read that diff.

- **DWARF extraction kept the last definition it met, in a `Dict`'s order.** `parse_dwarf_dump` keys types by their bare DWARF name and walked `type_refs`, a `Dict` keyed by DIE-offset strings, so the last definition of a name in that Dict's order was the one written. That order is the hash order of the offset strings, which changes with the Julia release (1.12 and 1.13 hash strings differently) and with every shift of the offsets, i.e. every upstream edit. Measured on llama.cpp: same source and path on 1.12.7 against 1.13.2 gave 2 struct sizes and 21 layouts that differed; nine upstream commits on one Julia gave 2 sizes and 16 layouts. Two kinds of names have several definitions:
  - **The same type, spelled differently.** C and C++ compile units both define ggml's records: `_Bool` against `bool`, a C11 anonymous union member `_anon1` against C++'s named `data` (`GGML_COMMON_AGGR_U`), and only a class's vtable home carries `__clang_vtable`. These now differ only in spelling, and one of them is kept by content.
  - **Distinct types sharing a bare name**: six pimpl `impl`s, two `input_decision`s, two `llm_graph_input_kpool`s. A bare-name key can describe only one. The one kept is now chosen by content and marked `"conflict"`, and the build prints which names they are. Keys stay bare: `resolve_type`, every member and signature `c_type`, and the generators' class lookup (`_cpp_innermost_scope`) all name types that way, so qualifying the keys is a change across all of them, not done here (see Open problems in CLAUDE.md).
  - `_canonical_definition` keeps the definition whose sorted-key text (`_canonical_text`) sorts first, and marks `conflict` when the definitions disagree on `_layout_signature`. `_layout_signature` covers kind, size, alignment, each placed member's offset and size, and each base's offset. Enums, globals (a C++17 inline variable is defined in every unit that uses it) and the typedef table are chosen the same way. Every pass where one DIE can win over another now walks `_die_order` (dump order), never `type_refs` itself. That includes the anonymous-aggregate naming pass, which sorted the offset strings: `"0xff00"` sorts after `"0x10000"`, so an offset gaining a digit reordered two parents.
  - The choice depends on the definitions only. Neither the Julia release, nor the DIE offsets, nor the order of the compile units moves it.
  - `children_by_parent` is built in dump order too, so a record's children are in declaration order. Template parameters would keep their written order, but none are parsed from a GNU dump today (see Not Yet Built).
- **Lambda closure types are no longer emitted** (John's decision). A lambda stored in a `std::function` or a `std::tuple` has a closure class declared at the lambda, in the project's source, so the provenance gate kept it. The anonymous-aggregate pass named it after the libc++ record that holds it, `__func<(lambda at <abs path>:<line>:<col>), …>`. llama.cpp had 53 of them. No function, global or other struct referenced any of them. They appeared only in `STRUCT_SIZES`/`STRUCT_OFFSETS` and in one another, and they changed name whenever the checkout moved or the source above the lambda gained a line: 602 of the dry run's changed lines. An anonymous type named after a parent the gate drops is now dropped with it (`drops_type`, transitively). A project record's own anonymous members are kept.
- **`Managed*` blocks are emitted in sorted order.** GeneratorCpp iterated its `deleters` Dict, so the same metadata gave 17 blocks (280 lines) in another order on 1.13.2.
- **A callback's `@cfunction` example is its own typedef's, or there is none.** The example came from whichever header typedef's name best matched the parameter's or the function's name. Ties and a first-entry fallback followed `Dict` order, and the pick was often another callback's: `ggml_backend_compare_graph_backend` documented a signature unrelated to its `ggml_backend_eval_callback` on both Julias, and `ggml_backend_cpu_set_abort_callback`'s example was wrong on 1.12.7. The DWARF parser now records the typedef a function-pointer parameter is declared with (`function_pointer_typedef`, the innermost of a typedef chain), and both generators take that typedef's header signature (`_callback_typedef_signature`). A parameter without one got no example, since DWARF's own signature had no parameter list. It has one now, and gives the example (see "Function-pointer types carry their parameter lists" above). The metadata key needs a rebuild; a re-wrap of older metadata has no examples.
- **More `Dict`-ordered emission and lookups, found by sweeping both generators and the thunk generator.** None of these was in the dry run's diff, because llama.cpp exercises none of them:
  - Global accessors and header-only enums were emitted in Dict order (both generators).
  - Of several metadata keys that sanitize, or fuzzy-match, to one name, the first met was used: `julia_to_cpp_struct`, the nested-struct size lookups, `_fuzzy_dwarf_lookup`, and FunctionGen's `_fuzzy_struct_lookup`, which picks a thunk's by-value layout. Each now takes the smallest key.
  - StructGen emitted aliases in Dict order. When an enum and a record shared a bare name, which one `!Struct_<name>` described depended on that order too. It is now the enum's, which is what `by_value_crossing` already decides.
  - The JLCS generator's vtable declarations, type_info ops and vcall table, the STL thunks and the array-view thunks now iterate sorted.
- **IR file names use SHA-256** (`_ir_dir_key`, through `_key_feed!`) of the source directory, after `_canon_path`. With `Base.hash`, switching Julia renamed every IR file: llama.cpp compiled all 226 TUs again, and `build/` kept both sets (2.4 GB). The templates TU's `.key` fingerprint had the same flaw. The first build after this change compiles every TU once more. The old `<src>-<8 hex>.ll` set is left in `build/` and is safe to delete (`clean` does).
- **Docs.** `calling.md` said a function pointer's signature is in its docstring. It was there only when the parameter is declared with a typedef the headers spell, until DWARF's signature gave one too (see "Function-pointer types carry their parameter lists" above).
- **Tests.** Each one was checked by neutering its fix.
  - `test_emission_determinism.jl` (runtests). Synthetic objdump DWARF laid out per compile unit. The same units in six orders and offsets give one set of tables. Spellings are kept by content and distinct types are marked. The closure goes with its libc++ parent, and a project record's anonymous union stays. The typedef is recorded through a chain. Both generators' text does not change when every metadata Dict is rebuilt at another capacity and in another order. Globals, header enums and `Managed` blocks are sorted. Callback examples come only from the parameter's typedef. The fuzzy lookups and StructGen give one answer for every Dict layout. Golden SHA-256 values cover the IR name and the templates fingerprint. Each neutered fix, including the old last-writer-wins loop, `Managed` in Dict order, the name-similarity scoring and `Base.hash` names, fails its testset.
  - `test_determinism_fixture.jl` (devtests §8d, `test/determinism_test/`). A C unit and two C++ units with ggml-style shared headers, two pimpl `impl`s, a lambda in a `std::function`, a class with a vtable home, three globals and a decoy callback typedef, built from source. The real objdump DWARF gives the same tables at three offset shifts and in three unit orders. In the metadata, the C spellings are kept, `impl` is marked, there is no closure, and `dt_hooks_u` is present. Each source has exactly one IR file, named by a SHA-256 the test computes itself. The wrapper has `Managed` blocks and globals in sorted order, the typedef's `@cfunction` example and no other. Re-wrapping from a metadata file with every object's keys reversed gives identical text. All eight neutered fixes fail it.
- **Verified** on Windows (MSYS2 CLANG64, LLVM/MLIR 23.1.3):
  - runtests: 2754 on Julia 1.12.7 and on 1.13.2 (v4.3.0 on this box: 2694, plus the new file's 60).
  - devtests on 1.12.7: 1371 plus 1 broken, which is v4.3.0's 1331 plus 1 broken (the C in-process pipeline test) and §8d's 40.
  - devtests on 1.13.2: 1370 plus 1 broken. The one count apart is `Debug: static inspection`, which asserts once per `.mlir` dump on disk. `.debug/` outlives `clean_test_dir`, so that count follows the files earlier runs left. vi_test's Tier 2 module text hashed to the same dump name on both Julias.
  - **Every built devtests fixture wraps the same on both Julias.** After devtests, each of the 13 built fixtures (C and C++) was re-wrapped from the same build on 1.12.7 and on 1.13.2: 13 identical, generation time aside. The v4.3.0 generator on the same metadata gives 7 that differ: `Managed` order (aot_test, determinism_test, mi_test, stress_test), the callback example (callback_test, c_abomination_test: a different guessed signature on each Julia) and global order (c_sret_test).
  - **llamacpp**, at its Hub pin b11819, copied to a short path and built cold. The Hub clone was not touched.
    - Built and wrapped on 1.12.7, then wrapped from the same build on 1.13.2: identical, 60336 lines, thunk manifest too.
    - Rebuilt on 1.13.2: all 226 IR files were reused (`compile: 226 cached`), and the metadata is equal key for key. The wrapper differs only in `LIBRARY_SHA`, because lld writes the link time into the PE header, so every relink gives a new DLL.
    - Against the Hub's wrapper (v4.3.0 code, same pin): 1053 functions with no signature change, 1088 API functions at the same tiers, 0 trap stubs before and after. The 53 lambda types are gone. The one-time move to content-chosen definitions changes 4 sizes and 13 layouts. The sizes are 4 of the 7 conflicted names (`extra_buffer_type`, `graph`, `graph_mtp`, `impl`, `input_decision`, `layer`, `llm_graph_input_kpool`). The layouts are spellings: the C unit's (`block_q4_K` keeps `_anon1`, `ggml_cplan` and `ggml_compute_params` keep `_Bool`), and a vtable home's `__clang_vtable`, which is no field, on 9 classes. `extra_buffer_type` is the one conflicted name among them.
    - `test_deep.jl` with the nomic model: 33/33.

### A link longer than one Windows command line goes through a response file (2026-10-22)

- **A C++ build with enough translation units could not link on Windows.** CreateProcess accepts at most 32,767 characters of command line, and the link put every per-TU `.ll` on one `llvm-link` command line. The Hub's llamacpp with ggml-vulkan (384 TUs) came to ~39,000 characters from its checkout depth. llvm-link never started, and the `clang++ -shared` fallback, handed the same list, failed with `ENAMETOOLONG`. The CPU-only build (226 TUs, ~25,000 characters) fits.
- **Long input lists go to a response file.** On Windows, when a link's command line would pass 30,000 characters, `Compiler._args_with_response_file` writes the inputs to `build/<name>.rsp` and passes `@<file>`. Flags stay on the command line. It covers llvm-link in `link_optimize_ir`, both `clang -shared` calls in `create_library` (the IR list and the per-TU objects), and `create_executable`.
  - llvm-link reads a response file by Windows rules, but the clang driver reads it by GNU rules, where `\` is an escape (measured: `C:\Users\…` reached clang as `C:Users…`). So each input is written double-quoted with `/` separators, which both read the same. A path holding a quote or a line break is refused.
  - Off Windows nothing changes.
- **A program that never started is reported as one.** `BuildBridge._run_command_impl` returned a spawn failure as `("Error: <the whole command>", 1)`. So `link_optimize_ir` warned "llvm-link failed (exit 1)" about a tool that had not run, and 45 KB of command reached the log. It now returns `BuildBridge.SPAWN_FAILED` (-1) and "could not start <program> (<n> arguments, <chars> characters of command line): <reason>". The llvm-link warning then says the tool was not run, and why.
- **The C bucket's JLL clang is reported when it does not start, not replaced.** `_clang_for_c_bucket` caught every error and ran the system clang instead. So a JLL clang that loaded but did not start would have silently become a system-clang build of the C bucket: a binary the OS refuses to run (Smart App Control refused Clang_unified_jll's pkgimage and libclang.dll on 2026-10-18), or a command line past the limit. That is the IR-version mismatch the bucket exists to prevent, with nothing to explain it. Such a failure now returns `SPAWN_FAILED`, with a message that names the JLL binary and says the system clang was not used in its place. The fallback stays for a Clang package or JLL that cannot be loaded (`_jll_clang()` returns `nothing`).
- **Tests.**
  - test_response_files.jl (runtests): the writer's threshold, both host spellings, the file's encoding, the refusal, and the four call sites; then spawn failures: a missing program on every host, and on Windows a 40,000-character command line. Then the C bucket: the real JLL clang given a 200 KB argument (no host starts that), a JLL clang whose binary is missing, and a JLL that cannot be loaded, which still falls back. Neutering the writer, the spawn check, or the C-bucket report fails its testset; with the old catch-all, the missing-binary case ran the system clang and returned 0.
  - test_link_cmdline_limit.jl (devtests §19): the real llvm-link and clang read a forced response file on every host, from a directory with a space. Then 200 one-line TUs at a deep path build end to end; on Windows that link passes the limit. With the writer neutered, that build fails with `ENAMETOOLONG`.
  - TestSupport gains `:llvm_link`, found on the `PATH` that `with_llvm_env` sets.
  - The Hub's llamacpp with ggml-vulkan, built cold from a 70-character package path (the Hub checkout's is 64): llvm-link read all 384 inputs from `build/llamacpp_link.rsp`, test_deep.jl passes 33/33, and all 13 layers run on the Vulkan GPU.
  - On main with #25 merged: runtests 2803 on Julia 1.13.2 and 1.12.7; devtests (1.13.2) 1384 plus 1 broken (already broken on main). Two existing testsets count what is on disk: "Symbol hygiene" adds 8 once the MI/VI fixtures are built, and "Debug: static inspection" adds 1 per `.mlir` file in vi_test's `.debug/`.

### AOT thunks are compiled by wrap, in one pass (2026-10-22)

- **A new `[compile] aot_thunks = true` package needed `build` and `wrap` twice.** `build` compiled the thunks library, but which thunks a wrapper binds is decided by `wrap`'s generator, which saved that set to `thunk_manifest.json` only afterwards. With no manifest yet, the first build guessed: a thunk for every function in the metadata.
  - One thunk the guess could not compile or link sank the whole library (on Windows every function a thunk calls must resolve at link time), so the first wrap fell back to JIT.
  - Or the wrap bound a thunk the guess lacked, and refused.
  - Either way only a second build and wrap, reading the manifest, converged. The 2026-09-10 fix made that second pass reachable; it did not remove it.
- **`wrap` compiles the thunks now, as its last stage.** The C++ generator decides the set, writes the manifest, then calls the AOT stage with exactly that set, and `_assert_aot_thunks_present` checks the library's exports. The stage is `ThunkBuilder.build_aot_thunks(config, lib; needed_symbols)`, passed down by `Wrapper.wrap_library` (`aot_compile`). One derivation, one pass. `build` no longer compiles thunks, and `wrap` takes the time it used to spend on them.
- **When the stage writes no library**, the generator throws `AOTStageFailed` and `wrap_library` generates the wrapper again with JIT dispatch, as a missing library did before. `build_aot_thunks` returns whether it wrote the library.
- **Messages.** The export check no longer tells you to re-run `build()` and `wrap()`: a missing symbol now means the MLIR generator was asked for it and emitted no thunk. The load-time messages say to run `wrap()` again rather than rebuild.
- **Docs.** config, Ship a package, internals and the MLIR page say `wrap` compiles AOT thunks. The internals and MLIR pages also said AOT thunks need no MLIR at run time. They need `libJLCS` there, as Ship a package already said.
- **Tests.**
  - test_aot_thunks.jl (devtests §5c): a cold `build` leaves no thunks library, and the first `wrap` compiles exactly the thunks it binds. Against main's code the same test fails there, because build's guess also compiled thunks for the Tier-3 `extern "C"` factories.
  - test_thunk_manifest.jl: the stage gets the generator's set once, after the manifest; `wrap_library` falls back to JIT when the stage fails; the export check still refuses a stage that reports success without a library. Removing the stage call, the fallback, or the manifest-first order each fails its testset.
  - imgui, the smaller of the two Hub packages that use AOT, built cold from a copy (the Hub clone was not touched). Main's first pass compiled a 731.5 KB thunks library from the guess, and its second pass the 696.0 KB exact set. This branch compiles the 696.0 KB set on the first wrap (1384 AOT call sites, no JIT), and imgui's test_deep.jl passes 202/202 against it. imgui's guess happened to compile on this box, so it does not show the two-pass failure itself.
  - runtests 2677 on Julia 1.12.7 and 1.13.2; test_aot_thunks.jl 175 on both; devtests (1.12.7) 1320 + 1 broken (already broken on main). The docs build has no new warnings.

### The registry builds a project where it lives, vendors that build, and adds it to the Julia environment (2026-10-21)

- **A project builds in its own root, and only there.** `discover`, `build` and `wrap` already worked in the project. `use` did not: it built a copy of the project's sources that `register` had staged inside the registry, because the stored TOML's relative `root` pointed there.
  - Two projects with the same relative paths overwrote each other's staged sources, so `use("twin_a")` loaded twin_b's library.
  - Edits to the project never reached `use` until it was registered again.
  - A registered project's first `use` compiled it a second time.
- **`register` vendors the project's own build.**
  - An entry points at the project's TOML.
  - `register` runs `build` and `wrap` on the project in place, which is a cache hit when they are current, and copies the output to `~/.replibuild/builds/<hash>/`, keyed on `hash_config`.
  - It writes `~/.replibuild/packages/<Module>/`, a Julia package whose entry file includes that vendored wrapper.
  - It adds that package to a Julia environment: `env`, by default the `@v#.#` one; `:active`, a path, or `nothing` are the alternatives, and `REPLIBUILD_JULIA_ENV` changes the default. `using <Module>` then loads it.
  - An environment without RepliBuild gets the running copy, which generated the wrapper. A package's own Project.toml, one with a `name`, is never changed.
  - `register(toml; build=false)` records the project only.
- **`use`** loads the vendored build while the project's inputs still hash to it. After an edit it rebuilds and wraps the project in place, vendors the result, and points the package at it. It loads the last vendored build, with a warning, when the project has moved or been deleted.
- **Vendored builds are immutable folders.** A session that has a build loaded keeps it. On Windows no open DLL is ever overwritten: re-vendoring rewrites the package's small entry file.
- **The package's entry file** uses `Base.include(Base.__toplevel__, …)`. A package file is evaluated in `Base.__toplevel__`, where a bare `include` does not exist, and the wrapper defines its own module.
- **`unregister`** removes the vendored build, the package and its place in the environment. It never removes the project's files; the old code deleted the stored TOML, which is now the project's own.
- **`discover`:** a bare `discover` no longer registers, since there is nothing to vendor yet. `discover(...; build=true, wrap=true)` ends with `register`.
- **Entries registered by an earlier RepliBuild** point at their registry copy and keep building it until registered again.
- **Tests no longer touch the user's registry or Julia environments.** devtests' chained `discover` had registered fixtures in the real `~/.replibuild` on every run. Both suites now set `REPLIBUILD_HOME` to a temp dir and `REPLIBUILD_JULIA_ENV=none`.

### User docs and messages say what RepliBuild does; registry entries get their own folders (2026-10-21)

Two checks found these. One followed the user docs alone, as a new user, in a fresh environment: a small C library and a small C++ library, the registry, and six deliberate mistakes. The other audited every user-facing message. Every call in the walkthrough returned the right value; what failed was the guidance around it.

- **Registry (behaviour).**
  - Every entry staged its project into one shared folder, and a relative `root` resolves to the stored TOML's folder. Two projects with the same relative paths overwrote each other's sources, so `use("twin_a")` loaded twin_b's library.
  - Each entry now has a folder of its own (`_entry_dir`), holding its build tree too. Registering again replaces the directories it staged, and `unregister` removes the folder. An entry registered before this keeps the shared folder until it is registered again.
  - `use(name; force_rebuild=true)` now builds from nothing (`build(clean=true)`). It used to re-wrap only: an unchanged project hash skipped the build, so metadata from an older RepliBuild survived the call meant to replace it.
- **`discover(path; build=true, wrap=true)`** on a project that already had its TOML returned before building or wrapping, silently. It now keeps the TOML and runs both.
- **`wrap` without `compilation_metadata.json`** asked "Did you compile with -g flag?", although RepliBuild always passes `-g`, then fell back to `wrap_basic`. `_assert_base_calls_qualified` refuses that function's output, so the fallback could never succeed. It is now an error that says to run `build`.
- **`check_environment()`**:
  - A `libJLCS.dll` that would not load because `clang64\bin` was missing from PATH was reported as built against an LLVM that is gone, with the advice to rebuild. It now says which of the two it is (`_clang64_bin_off_path`).
  - The MSYS2 probe read `MSYSTEM`. Julia started from PowerShell was told to switch to the CLANG64 shell, and Git Bash (which sets `MSYSTEM=MINGW64`) got a C++ ABI warning while every tool came from clang64. It now reads where `clang++` lives.
  - New probe: `llvm-dwarfdump`. Every JIT C++ wrapper runs it, by that bare name, when it loads; Tier 2 now requires it.
  - The summary said "C and C++ wrappers will build" on the C bucket's evidence alone. It now has a separate C++ line, decided by `clang++` and `llvm-config`.
  - "Tier 2 unavailable" said C++ "falls back to ccall where it can". It now says what is actually left: a C++ wrapper's noexcept functions whose arguments ccall can pass.
  - The install advice:
    - Debian/Ubuntu now lists the MLIR packages that `llvm.sh` does not install.
    - Windows says to put `clang64\bin` on PATH, and the cmake advice is per platform.
    - The dialect build line is written with `/`, because bash strips backslashes, and names `Pkg.build` on Linux.
- **The build's load check** reported a Smart App Control block, and a DLL missing from PATH, as "unresolved symbols". `_load_failure_message` now tells the three apart.
- **Registry messages.**
  - `search` ended with "Install: RepliBuild.use(...)", which fails for an unregistered name. It now prints the register-then-use steps.
  - `use`'s error for a name not in the registry now mentions the Hub.
  - `list_registry`'s "cached / not built" looked under the TOML's hash, but builds are stored under `hash_config`, so it always said "not built". It now says whether `use` has built the entry.
  - `unregister`'s removal of cached builds had the same mismatch, and is gone.
- **Messages that named the wrong fix.**
  - "Run discover() first to find C++ sources" appeared for a project discover had already scanned.
  - The generated stale-wrapper warning said only "Re-run wrap()". A fix in what `build` extracts also needs `build(toml; clean=true)`.
  - The FFI trap and the strict-mode type error recommended an internal function. They now name `[types.custom]` and the other `[types]` keys.
  - `clean` on Windows now says that a loaded library blocks the delete.
  - The "Run … first" messages print the path with `repr`, so it pastes into Julia.
- **Docs.**
  - install: where the dialect lives for a `Pkg.add` install and when to rebuild it; `llvm-dwarfdump` on PATH; binutils on Linux; the full Windows setup; Smart App Control.
  - calling: how to construct and destroy a C++ object (`T()` is a zeroed value, not a construction); `Managed*` and `_safe` as they are generated; by-value versus pointer struct parameters; argument types; `libc_free`; when `dispatch_tier` exists.
  - Ship a package: a C++ wrapper needs libJLCS where it runs, JIT or AOT, plus the Windows runtime DLLs.
  - Registry: `use` builds from the copy `register` made.
  - Troubleshooting: the Windows loader cases and calling errors.
  - API: `LIBRARY_SHA`, not `BUILD_ID`.
  - The changelog page covers 4.1 to `main` and says which fixes need a rebuild rather than a re-wrap.
  - README: the broken WINDOWS_PORT.md link, and the stale autocrlf step.
  - WINDOWS_PORT.md marks its setup section superseded, and its one open item closed by `f99e07f`.
- **Generated wrappers** change only in the text of those two messages: the stale-wrapper warning and the FFI trap.
- **Tests:**
  - `test_user_messages.jl` (new, runtests).
  - Four new testsets in `test_registry.jl`.
  - Each of the seven code fixes was undone, using its exact method signature. Each time, its own testset failed.
- **Verified** on Windows (MSYS2 CLANG64, LLVM/MLIR 23.1.3):
  - runtests: 2660 on Julia 1.12.7 and on 1.13.2.
  - devtests (1.12.7): 1318, plus the 1 broken test already broken on main.
  - A local docs build: no broken links or anchors.

### Tier 2: a tiled record's bitfield is read where its body puts it (abi_fuzz X9; 2026-10-21)

- **A record holding a bitfield could reach the callee with the bitfield, or members after it, wrong.**
  - `is_struct_packed` calls a record packed when its member sizes sum to its size, and its thunk body then tiles the members end to end (`get_julia_offsets(info, true)`).
  - The marshal that copies the Julia value into that body reads each member at its DWARF offset. A bitfield has none, and `_dwarf_member_offsets` then fell back to aligned guesses for the whole record.
  - On MinGW a `#pragma pack(1) {unsigned; unsigned long long : 7}` is 12 bytes, with the bitfield's own 8-byte unit at 4. The body put the unit at 4, but the marshal read it from 8: the callee got four bytes of the record and four from past its end.
  - A member without a DWARF offset is now read where the body puts it.
- **A matching sum is no longer taken for a tiling.** `is_struct_packed` also requires every member with a DWARF offset to sit at it, and every bitfield's bits to lie inside the unit the tiling gives it.
  - MinGW lays bitfields out MS-style, one declared-size unit per run, so its exact sums are real tilings.
  - Itanium packs a bitfield into the neighbouring bytes: `{short; int : 4; short}` sums to its 8 bytes with the second short at 4, not 6. Such a record now crosses as its bytes.
- **Tests:**
  - `test_thunk_byte_regions.jl` (runtests) checks the offsets of the three failing shapes (fuzz units 12, 31 and 39), the marshal they produce, and the Itanium counter-example. With the offset fix removed 4 checks fail; with the tiling check removed, 5.
  - `cpp_routing_test` gains `PB12`, passed and returned live. Without the fix, `pb12_sum` read `m2` as 0.
- **Fuzzer, C++ mode, seeds 1–40, Julia 1.13.2:** the 15 failures left after the merged fixes pass, and nothing regresses. That mode now has no failures: against the base, all 153 pass.

### Tier 2 thunks: unions and bitfield records cross by value as their bytes; `#pragma pack` bodies stay packed (abi_fuzz X2, X8; 2026-10-20)

- **X2: a union or bitfield record passed by value reached the callee as its first byte, or crashed it.**
  - A record with no field-by-field body (a union, a bitfield record, overlapping members) is `!llvm.array<N x i8>` to StructGen, and the thunk passed and returned it as that array.
  - LLVM splits an array argument into one register per element, so the callee read byte 0 and whatever the other registers held. On Win64, which passes a record of any size but 1, 2, 4 or 8 by reference, the callee read that byte as an address.
  - Such a record now crosses as `_byte_blob_type(N)`, a packed struct of the same bytes, in both directions. The dialect's SysV and Win64 classifiers place it as C places a record without float members.
  - On SysV, one of 16 bytes or less that may hold a `float` or `double` is refused with a trap stub (`_byte_region_unclassifiable`). C passes it in SSE registers, and its bytes cannot say which ones. Win64 crosses a record by size alone, so nothing is refused there.
- **X8: a `#pragma pack` record holding a pointer-aligned record got an unpacked thunk body.**
  - `_sized_member_types` measured an alias member (`!Struct_X`) as X's bytes, alignment 1, while the body emitted the alias, which has X's alignment.
  - The natural-layout check accepted the member at offset 4, LLVM moved it to 8, and the body came out 4 bytes larger than the record.
  - The member is now measured as what is emitted, so the body is packed.
- **Tests:**
  - `test_thunk_byte_regions.jl` (runtests) checks the X8 body, which records are byte regions, the SysV and Win64 verdicts, and the declarations a thunk module gets. With each fix removed in turn (the X2 crossing, the SSE refusal, the X8 measure), 5, 1 and 2 checks fail.
  - `cpp_routing_test` gains the unions `U4` and `U12`, the bitfield record `BF8` and a `#pragma pack(4)` record `PK28`. `test_cpp_record_routing.jl` (devtests) checks that each routes to Tier 2 and calls each live.
  - `make_u4`, `make_u12` and `make_bf8` return each record by value through a thunk, live, and pass it back in. That takes X1 (the record Julia allocates) and X2 (the thunk receiving it as C returns it) together.
  - The repros `X2-18-cpp-call_f2-union`, `X2-53-cpp-call_f3-bitfield`, `X2-99-cpp-call_f1-bitfield` and `X8-49-cpp-call_f4` pass.
- **Fuzzer, with this entry, X1/X5/X6 and C2/I1/M1 together, compared with the base on Julia 1.13.2:**
  - C++ mode, seeds 1–40: failures drop from 153 to 15. 138 checks go from FAIL to PASS, 95 more than X1/X5/X6 alone, including `9/cpp ret/f5`, and none the other way.
  - The 15 left are calls that pass a record holding a bitfield. Some members arrive wrong: the bitfield, or members laid out after it. Tier 3 passes the same calls. That is X9, fixed in the entry above.
  - C mode, seeds 1–106: no failures. Identifier mode, seeds 1–40: the same verdicts as C2/I1/M1 alone.

### Tier 2 and C++ accessors: unions and bitfield records move through their bytes; wide bitfields (abi_fuzz X1, X5, X6; 2026-10-20)

The C++ generator emits a named union and a bitfield record as a `mutable struct` whose one field is the record's C bytes. In Julia such a type is a reference, and three places moved one as if it were not.

- **X1: every Tier 2 function returning a union or bitfield record threw `UndefRefError`.**
  - `JITManager._invoke_call` allocated the sret buffer as `Ref{T}()`, which is an undefined reference for a mutable `T`, and the ccall conversion threw.
  - It now allocates the record, zeroes it, and hands the thunk the address of its bytes, both through the Windows guard and in the direct call.
  - A mutable type with no inline byte layout is refused before the call.
- **X5: accessors read a union or bitfield-record member with `unsafe_load(Ptr{T}(p))`**, which reads C's bytes as an object reference.
  - The bitfield-record accessor and the blob `getproperty` now build such a member from its bytes.
  - The blob setter stores the bytes back (new `_blob_store_expr` kind `:mutable_record`).
- **X6: bitfields wider than 32 bits.**
  - The getter loaded a `UInt32` and masked it with a mask that does not fit, so it always threw. It could also read past the record, and there was no setter.
  - The getter and a new setter gather exactly the bytes the field spans, into an integer wide enough for them. A field over 32 bits reads as `UInt64`.
- X4's repro already passed on this base (ef8f822 sizes the member, which then degrades to bytes).
- **Tests:** `test_cpp_mutable_records.jl` (runtests). It evaluates the generated accessors in a scratch module and returns a record through `_invoke_call` from a Julia callback standing in for the thunk.
  - With each fix removed in turn, 1, 2 and 2 checks fail.
  - The repros `X1-48-cpp-call_f1`, `X5-73-cpp-access_R3` and `X6-31-cpp-access_R4` pass.
- **Fuzzer, compared with the base** (C++ mode, seeds 1–40, Julia 1.13.2):
  - Failures drop from 153 to 110; 44 checks go from FAIL to PASS.
  - One went the other way: `9/cpp ret/f5`, a crash inside a callee whose call check fails on both runs. That callee takes a bitfield record by value (X2, open on that branch), so whether it crashes depends on what the misplaced registers hold.

### Tier 3: one refused function stays one; records are spelled as declared; module names are identifiers (abi_fuzz C2, I1, M1; 2026-10-20)

- **A `long double` parameter could refuse the whole wrapper (C2).**
  - BRANCH 0, the explicit-sret path, never applied the `long double` trap.
  - So a function that took a `long double` and returned a record through a hidden pointer got a ccall that declared the `long double` as `Any`, and `_assert_no_any_ccall_argument` refused the whole module.
  - BRANCH 0 now leaves such a function to BRANCH 2, which emits its trap stub, and the rest of the library wraps.
  - No `long double` support was added.
- **A record named with a Julia keyword broke the wrapper (I1).**
  - Records and enums are declared under `_sanitize_c_type_name` of their C name (`baremodule` becomes `c_baremodule`).
  - Signatures, however, took the metadata's raw spelling unless it held `<>`, `::` or other non-identifier characters. `ccall(..., baremodule, ...)` reached a type position, and the module failed to parse.
  - GeneratorC now re-spells return and parameter types through `_as_declared`, a table of the names the sanitizer changes. That covers `Ptr{…}`/`Ref{…}` and `Any` parameters named by their C type.
  - Names the sanitizer leaves alone are untouched, so ordinary wrappers are byte-identical.
  - `begin` and `elseif` were missing from both sanitizers' keyword lists, so `typedef struct {…} begin;` emitted `struct begin` and the wrapper did not parse. Both lists now escape every word the parser rejects as a type name; `test_enum_underscore_name.jl` checks each word against the parser for both generators.
- **A project directory starting with a digit gave a module that did not load (M1).**
  - `get_module_name` produced `3cCallF21`, and `module 3cCallF21` parses as a module named `3`.
  - The derived name now gets a `_` prefix, or becomes `Wrapper` when nothing is left.
  - `_assert_wrapper_parses` now refuses a module whose name is not an identifier.
- **SysV: a blob return of 16 bytes or less whose fields' alignment cannot be proven is refused**, as the parameter path already did. An unaligned field would make it MEMORY class while its byte image comes back in registers. Win64 is unaffected.
- **Tests:**
  - `test_c_sret_classification.jl` covers C2, I1 and the SysV case, for both ABIs. `test_function_name_derivation.jl` covers M1.
  - With each fix removed in turn, 6, 10, 2, 3 and 1 checks fail.
  - The fuzzer repros `C2-31-c-load` and `I1-11n-c-load` pass.
- **Fuzzer, compared with the base on Julia 1.13.2** (C mode, shape seeds 1–106 and identifier seeds 1–40):
  - C2's 4 failures pass. The 96 checks behind the 4 refused wrappers now run: 86 pass, and 10 are `long double` stubs.
  - Identifier passes go from 61 to 101, and no failure from a keyword-named type is left.
  - Nothing that passed before fails. Unit 33n now loads, and 9 of its checks fail on names that shadow Base bindings (I4/I5), which are still open.

### Enum and typedef'd members have a size, and the C++ struct emitter can take them (2026-10-18)

- **`parse_dwarf_dump` gave three kinds of member size 0**: an enum member, a
  member whose type is a typedef of a record or an enum, and an array of records.
  It sized a member from its C type's spelling and a lookup of record names.
  `_die_byte_size` follows the member's type DIE instead. A typedef or a qualifier
  leads to its target, and an enum or a record gives its `byte_size`. An array is
  its element times its dimensions, and a pointer or a reference is 8. On the
  Hub, 1,908 member sizes go from 0 to a size, in 23 of 26 packages. Each fits its
  slot (the next member's offset, or the record's end), and no size that was
  already set changes. Apart from the timestamp, nothing else in the metadata
  changes. A package gets this on a rebuild (delete
  `.replibuild_cache/project_hash`), not a re-wrap.
- **GeneratorCpp's struct field path now meets those records, and a size-0
  member had been masking its gaps.** Such a record was always a blob. Re-wrapped
  on the new metadata, box2d, fmt, imgui and llamacpp did not load, and llamacpp
  had 194 structs of the wrong size.
  - **Arrays of records.** An array named its element wrong:
    `NTuple{2, ImVector<unsigned int>}` was sanitized whole into
    `NTuple_2_ImVector_unsigned_int`, and `NTuple{16, b2ContactRegister}` named a
    struct emitted later. `_usable_field_type` now peels the array and checks the
    element like any other field type.
  - **Members with no offset.** A static data member, or clang's
    `__clang_vtable`, has no offset. It was sorted as 0 and became a field at the
    front of the struct. It is not a field any more (`layout_members`). Every
    polymorphic class carries `__clang_vtable`, which on its own made each of them
    a blob.
  - **Fields the layout proof cannot measure.** These were emitted anyway. A
    bitfield record or an opaque placeholder is a `mutable struct`, a reference
    where C has the value inline. A forward declaration is an empty
    `struct X end`. The proof answered `:unknown` and the fields went out
    unproven: `llama_hparams` came out 492 bytes for 29180. A type is now a
    field's type only when the proof can measure it and it has the member's DWARF
    size. Otherwise the member is its bytes (`_member_field`).
  - **Enum members are typed.** An enum member is now typed as its enum
    (`k::Kind`). The layout comes from the enum as emitted; an alias has its
    canonical's.
  - **A forward declaration could replace an enum.** Typed enum fields exposed
    this. `dwarf_defined_names` held raw enum names, so an enum emitted under
    another name (`Type` → `c_Type`) also got `struct c_Type end`, and Julia 1.12
    lets that replace the `@enum`. In box2d and fmt, every later `::c_Type` meant
    an empty struct; box2d's `b2Contact_AddType` took two of them.
  - **A record holding a named union is a blob, checked by type.** A named
    union is a `mutable struct`, so as a field it is a reference, and the proof
    measures it as a pointer. An 8-byte union at an aligned offset fits that
    measure. abi_fuzz unit 104's `R5`, which ends in `union T4 m6`, got fields,
    so `R5` was not plain bits and C's bytes in it crashed the GC. Before, its
    size-0 enum members had made it a blob.
  - **Setters for records that stop being blobs.** A blob struct gets
    `setproperty`/`setproperties`; a field struct got neither. llama.cpp's
    consumers set `llama_context_params` and `llama_model_params` fields with
    them. Every C++ field struct now gets `setproperty`, plus a `setproperty!`
    whose error says to use it (`_field_setter_chunk`).
  - **Routing.** A record of 16 bytes or less goes to Tier 2, as a blob does, when
    its fields put bytes where C has a `float` or `double`: a member written as
    bytes, or a field whose record is a blob (`_type_has_sse`,
    `_crosses_small_blob`). Bytes over integers or an enum stay on ccall.
- **The thunk generator met them too, and a whole JIT module failed (abi_fuzz
  X3).** StructGen calls a record packed when its member sizes add up to its
  `byte_size`, and such a record crosses a thunk through `jlcs.marshal_arg`,
  member by member into its packed body. The body holds a record, an enum or an
  array of records as that type's bytes. `memberTypes` spelled the member from
  its C type instead, so the `insertvalue` did not type-check and the module
  failed to translate to LLVM IR, taking every Tier 2 function in the library
  with it.
  - With sizes, many more records add up: abi_fuzz C++ mode went from 4 units
    failing this way to 33.
  - `memberTypes` are now read off the packed body itself.
  - The 4 units where X3 showed before (41, 52, 54, 78) build their engines
    too, so X3's bitfield-and-enum shape is fixed as well.
- **Hub.** All 26 packages were rebuilt with the new parser, and the 7 C++ ones
  re-wrapped. That re-wrap came before two of the fixes above, the named-union
  rule and the `marshal_arg` fix, and neither was measured on the Hub. The first
  can only turn a field struct into a blob. The verifiers were not rerun.
  - The 19 C wrappers are byte-identical apart from `LIBRARY_SHA`, since a rebuild
    relinks the library.
  - Blob counts in the C++ wrappers: tinyxml2 18 → 0, fmt 34 → 5, pugixml 30 → 0,
    box2d 53 → 3, imgui 31 → 0, llamacpp 317 → 0.
  - Loaded and measured against DWARF, every emitted struct has its `byte_size`
    and every field its offset. That includes 8 structs that were already wrong
    (imgui 2, llamacpp 6).
  - No `DISPATCH_TIER` entry changed.
- **Tests.**
  - `test_dwarf_member_sizes.jl` (runtests) parses a vendored GNU objdump dump of
    a C file with each member kind, and calls `_die_byte_size` directly.
  - `test_record_natural_layout.jl` runs the C++ generator on synthetic metadata
    covering every case above. It checks the emitted text and the tiers, then
    loads the module and checks the sizes, the enum field and the setters, and
    that every struct emitted immutable is plain bits.
  - `test_cpp_record_routing.jl` (devtests) checks `EnumRec`'s typed fields and
    its Tier 3 route on a real build.
  - `test_packed_layout.jl` (runtests) checks that each `marshal_arg`'s
    `memberTypes` are its packed body's fields. `test_jlcs_producers.jl` §H
    (devtests) builds the engine for such a module.
  - Negative checks: with the old member sizing, the parser test fails 15 of 24
    and the live test 3 of 24. With each of the 9 generator guards removed in
    turn, the unit test fails 1 to 4 checks. With `memberTypes` spelled from the
    C types again, the packed-layout test fails 2 and §H fails to build its
    engine.
- abi_fuzz, C mode, seeds 1–106, Julia 1.12.7 and 1.13.2: no check changes
  against the previous commit.
- abi_fuzz, C++ mode, seeds 1–106, against the previous commit on the same Julia:
  - FAILs go from 527 to 428 on 1.12.7 and from 529 to 432 on 1.13.2.
  - 97 checks go from FAIL to PASS on each, and none goes the other way. On
    1.12.7 six more checks changed, all of them ones that already differed
    between the baseline runs. Two went back to FAIL (59/cpp `ret/f1` and
    77/cpp `ret/f2`), the X2/X5 garbage reads the previous entry saw pass.
  - 41 of the 97 are X3's engine failures. X3 is gone: 54 checks to 0.
  - X7's two units (57 and 79) now pass: the misaligned enum member had size
    0, so the record was never classed packed.
  - X4's repro unit (53) passes its layout checks (8 FAILs to 2).
  - X5 is unchanged.

### Tier 3: a hidden return buffer has the record's alignment (2026-10-18)

- **A record returned through a hidden pointer could crash its callee.**
  Julia's ccall allocates that buffer at the alignment of the Julia type
  standing in for the record: 1 for an `NTuple{N,UInt8}` byte blob, 4 for a
  struct of `Float32`s. C's callee may assume the record's own alignment and
  store with `movaps`. `S16 get(void) { return g; }`, with a 16-aligned `S16`,
  compiles to `movaps %xmm0, (%rcx)`, and Julia put a blob's buffer at
  `-24(%rbp)`: an access violation on every call that was not inlined into a
  frame where it happened to land aligned. A record is more aligned than its
  stand-in with `aligned(N)` on it, an `_Alignas` member, a member of an
  over-aligned typedef (cglm's `vec4` style), a vector member, a `long double`,
  or any of these nested.
- **The parser records the alignment.** `DW_AT_alignment`, which clang emits
  in DWARF 4 as well, becomes `alignment` on the record and on the member in
  `struct_definitions`. Clang puts it on the member when it comes from an
  over-aligned typedef, whose own DIE carries none. A vector member
  (`DW_AT_GNU_vector`) gets its size. Packages need a rebuild to carry it;
  delete `.replibuild_cache/project_hash`.
- `_c_record_align` is the record's alignment: the largest of its own
  attribute, its members', and its member types' alignments (`long double`
  16, nested records, arrays by element). A member type nothing can measure
  makes it the largest power of two dividing the size, which is never too
  small. `_c_sret_buffer_align` asks for the wrapper's own buffer where that
  alignment is above both the stand-in's and 8: on x86-64 nothing faults on a
  misaligned access below 16.
- BRANCH 0 emits it: `Ref{NTuple{size + align - 1, UInt8}}()`, the record at
  the first aligned address inside, and `unsafe_load` of the stand-in from
  there. Such a return takes this branch whatever `c_safe` says. A function
  with a `long double` parameter keeps its x87 stub instead, because BRANCH 0
  would hand its `Any` argument to ccall and refuse the wrapper (C2).
- C++: the record-level key wakes `_is_struct_unsafe`'s check that sends a
  record aligned above 8 to Tier 2. In the Hub only libc++'s `aligned_storage`
  unions carry one, and unions go to Tier 2 already.
- Tests. `test_c_sret_classification.jl` (runtests) pins `_c_record_align`
  for every shape above and the plain ones, `_c_sret_buffer_align`, the parser
  on a vendored GNU objdump dump (record attribute, `_Alignas` member,
  over-aligned typedef member, vector, nested record), and what BRANCH 0 emits
  for both ABIs, including the `long double` guard. `c_sret_test` gains eight
  makers that copy an aligned global out, so clang stores the return with
  `movaps`. `test_c_sret_abi.jl` (devtests) calls them through `invokelatest`,
  so the buffer is the wrapper's own, and checks `_c_record_align` against the
  alignment clang gives each record's global for both targets.
  Negative-checked: with the old rule the unit test fails 25 of 231 and the
  live child dies of an access violation in `make_ab16`; without the vector
  rule `HVec` comes out 4-aligned in both tests; without the parser's capture
  they fail 8 of 231 and 24 of 82; without the `long double` guard, 3 of 231.
  runtests.jl passes on Julia 1.12.7 and 1.13.2, devtests.jl on 1.12.7, and
  the live test on both.
- **abi_fuzz** (C mode, shape seeds 1–106 and identifier seeds 1–40, Julia
  1.12.7 and 1.13.2), compared with this branch before the merge and this
  change: no check changes status, and no verdict text changes beyond paths,
  timings and timestamps. The fuzzer generates no over-aligned record. In
  identifier mode, units 16n, 26n and 27n quote a parse error from a function
  that is now in BRANCH 0's explicit-sret form. b934795 generates the same
  text from the same metadata, so that comes from the merged
  `_record_natural_layout` commit, not from this one.
- **Hub**: the 19 C packages were rebuilt with `project_hash` removed, so
  their metadata carries alignment. That covers box2d3's SIMD solver records,
  duktape's and lua's `jmp_buf` holders, miniaudio's and raylib's atomics, and
  xxhash's 64-aligned `XXH3_state_s`. On the same DWARF, b934795's parser and
  this one give identical records apart from the new keys. Re-wrapped from
  that metadata, every wrapper is byte-identical to b934795's under the SysV
  rules and the Win64 rules: no Hub function returns an over-aligned record
  through a hidden pointer. Every verifier passes on the rebuilt Hub (17
  `test_deep.jl`, curl's `test.jl`, pcre2 loads, and the mpack tag check).
  The Hub's earlier metadata came from an older parser (2026-10-11), whose
  member sizes differ. The rebuilt wrappers differ from the earlier ones only
  in a timestamp and, in box2d3, miniaudio, raylib and tomlc17, in the
  >16-byte returns the merged commit moves between a plain ccall and the
  explicit sret.
- Not covered, in CLAUDE.md's Open problems: GeneratorCpp's own ccall path,
  the `[wrap.varargs]` path, an alignment only a typedef carries, SysV byval
  parameters, and SysV records holding a `long double`.

### Tier 3: the register-class traps are SysV's only (2026-10-18)

- **Win64 refused by-value crossings it makes correctly.** Two GeneratorC
  traps assume SysV's register classes. The float-member trap refuses a
  record of 16 bytes or less kept as an `NTuple{N,UInt8}` blob when it has a
  float member, because SysV passes it in XMM registers while the blob claims
  INTEGER. `abi_class_risk` refuses the same way a named-field struct holding
  an opaque region of mixed float and integer members. Win64 gives a record no
  register class: C passes and returns it by size alone (1, 2, 4 or 8 bytes as
  an integer of that width, any other size by reference to a copy or through a
  hidden return pointer), and Julia lowers a non-primitive type of the same
  size the same way. There the traps only cost callable functions: 31 in the
  abi_fuzz corpus after C1, and all 31 of Hub mpack's stubs (`mpack_tag_t` is
  16 bytes and holds a union of integers and floats).
- `_c_register_class_misstated(byte_size, class_risk; abi)` decides both traps
  per target, from `_c_record_abi()`: on SysV at 16 bytes or less when the
  stand-in may misstate the class, as before, and never on Win64. SysV's
  MEMORY-class exemption for returns and its unaligned-field blob trap are
  unchanged.
- **C2 can now refuse a whole wrapper where Win64 used to refuse one
  function.** BRANCH 0 (explicit sret) runs only when no blob trap fired, and
  it never applies the `long double` trap (abi_fuzz C2). A function that takes
  a float-bearing blob or a mixed-region struct by value, returns a record
  BRANCH 0 takes (over 16 bytes and nominated by `c_safe`), and also takes a
  `long double` used to get the `long double` stub on Win64. It now reaches
  BRANCH 0, and its `Any` argument refuses the wrapper. No fuzz unit and no Hub
  package has one. Fixing C2 closes it.
- Tests: `test_c_sret_classification.jl` pins the predicate and what is
  emitted for both ABIs: float blobs as parameters and as a register-sized
  return, an mpack-shaped mixed region, the struct embedding it, and an
  all-float region that is callable on both. `test_c_generator_policies.jl`
  checks a float-bearing bitfield blob as parameter and return for both ABIs,
  and loads the Win64 module, whose blob crossings reach their ccall.
  `abi_nested_test`'s probe now expects `packedfv_sum` (a 9-byte packed float
  blob) refused on SysV and called on Win64, where it returns 11.0.
  Negative-checked through the exact method signature. With SysV's rule on
  every target the two unit tests fail 16 of 154 and 8 of 96 checks and the
  probe fails on Windows. With no trap on any target they fail 16 of 154 and
  6 of 96, all of them the SysV checks.
- **abi_fuzz** (C mode, shape seeds 1–106 and identifier seeds 1–40, Julia
  1.12.7 and 1.13.2, which agree check for check), compared with f40496f's
  results: 57 checks go from GAP to PASS, every one an ABI-trap refusal, and
  no other status changes. 31 functions in 18 units are now calls: 22 take a
  float-bearing blob of 7 to 16 bytes, 1 returns an 8-byte one, and 8 take or
  return a 4- or 16-byte struct holding a mixed region. Their call and return
  checks compare every member with what the callee read or wrote. Totals go
  from 2237 PASS, 4 FAIL and 57 GAP to 2294, 4 and 0, and the 4 FAILs are C2.
  The 31 trap gap lines are gone. 8 `param-type` gap lines appear, all on
  newly callable functions: the trap had hidden the existing gap where a C
  `char` parameter is declared `::UInt8`. In identifier mode nothing changes
  status. In units 31n and 34n three stubs became calls ahead of the syntax
  errors that still refuse those wrappers, which moves the line number in the
  quoted parse error.
- **Hub**: the 19 C packages, re-wrapped without a rebuild and compared with
  wrappers from a clean export of f40496f. Under the SysV rules (generated on
  Windows from the same metadata) all 19 are byte-identical apart from
  timestamps. Under the Win64 rules only mpack changes: its 31 trap stubs
  become Tier 3 ccalls and gain their 31 `DISPATCH_TIER` entries, and nothing
  else in it moves. Every package's verifier passes against the Win64
  wrappers (17 `test_deep.jl`, curl's `test.jl`, and pcre2 loads). mpack's
  `test_deep.jl` calls none of the 31, so a separate live check called them
  against `libmpack.dll` (143 checks, all passing). It reads every returned
  tag through its fields and through the library's own pointer getters,
  checks equality and ordering of tags passed in, compares what
  `mpack_write_tag` encodes with the MessagePack spec byte for byte, reads tags
  back with `mpack_read_tag`, `mpack_peek_tag`, `mpack_expect_tag` and
  `mpack_node_tag`, and makes 400 000 calls under GC. Against f40496f's
  wrapper the same check fails 62 checks and errors 5.
- **Found on the way: a Tier-3 sret buffer had the Julia type's alignment.** A
  callee returning a record aligned above its stand-in may store it with
  `movaps`, and on Win64 that crashed. The returns this change makes callable
  are register-sized blobs, which come back in RAX, and mixed-region structs,
  whose Julia alignment is the record's own DWARF-derived one. Fixed by
  "a hidden return buffer has the record's alignment" above.
- SysV is covered by the unit tests; no SysV callee ran.

### Whether a record is packed comes from its member types' alignment, in C and C++ (2026-10-18)

- **`get_julia_aligned_size` is gone.** It decided whether a record was packed by
  laying the members out again with each member's size as its alignment, capped
  at 8, and comparing the result with the DWARF size. C's `is_c_lto_safe` and
  C++'s `_is_struct_unsafe` (the `is_ccall_safe` choice between ccall and a
  thunk) both used it. It called a record packed whenever a member's size was not
  its alignment (`short[3]`, `float[3]`, a nested 8-byte `Vec2`) and whenever a
  member was an enum, which the metadata records with size 0. Its C++ nested check
  looked one level down with no type table, so a record packed two levels down
  read as natural and reached ccall; on SysV its blob travels in registers while
  C passes it in memory.
- **`_record_natural_layout`** (Utils.jl) is now the one answer for both. A
  record is natural when each member, in offset order, sits at the first offset
  after the previous one that is a multiple of its type's alignment, the size is
  the members' end rounded up to the largest alignment, and every member record
  is natural too, at any depth. Unions, bitfields and members it cannot measure
  are not natural. Static data members and the `__clang_vtable` placeholder (no
  offset) are skipped. It reuses C1's member primitives (`_c_scalar_layout`,
  `_c_member_record`), which now also find C++ names through their `::` scopes and
  the `" >"` key artifacts.
- **GeneratorCpp keeps a byte blob of 16 bytes or less off the ccall path**
  (`_crosses_small_blob`). A natural record the struct emitter cannot prove (an
  enum member with size 0 is enough) is emitted as `NTuple{N,UInt8}`, which
  claims INTEGER for every SysV eightbyte, so a function crossing one now goes to
  the thunk. Above 16 bytes a record crosses in memory on both ABIs and the
  blob's bytes are exact. With the old size test such records happened to read as
  packed; with a precise one they would have reached ccall.
- **Hub, re-wrapped without a rebuild against the previous commit** (C++ against
  main): box2d, pugixml, tinyxml2, imgui, fmt, hello_world and 15 C packages are
  byte-identical apart from timestamps.
  - llamacpp: 15 extern "C" functions that take or return its parameter structs
    by value (`llama_model_default_params`, `llama_model_load_from_file`,
    `llama_context_default_params`, `ggml_opt_init`, …) go from Tier 2 to ccall.
    The structs are natural and 28 to 160 bytes, so even as byte blobs they cross
    exactly. They gain the `*_safe` RAII variants ccall'd functions get.
  - box2d3 (39), miniaudio (30) and raylib (16) functions returning natural
    records of 20 to 304 bytes go from the explicit sret to a plain ccall;
    tomlc17 (6) and miniaudio's `ma_job_init` go the other way, since their
    records hold a union. Above 16 bytes both forms pass the hidden pointer on
    both ABIs.
  - Tier 1 emits the same slices. `is_c_lto_safe` also filters its slicing
    candidates, which gain or lose record-returning functions (box2d3 +47,
    miniaudio +33/−1, tomlc17 −6), so the `tier1:` stats line moves (box2d3
    accepts 406 slices instead of 363). But Tier 1's shape gate never sends a
    record return to llvmcall: box2d3 still emits 26 slices and miniaudio 743, and
    no `TIER1_*` line changed.
- Tests: `test_record_natural_layout.jl` (runtests) covers the layout answer, C++
  routing, `is_c_lto_safe` and the blob guard through the C++ generator.
  `test_cpp_record_routing.jl` with `cpp_routing_test` (devtests) builds a C++
  library, checks each function's tier, and calls every function in both
  directions. Negative-checked through the exact method signatures: with the old
  size test the unit test fails 23 of 48 and the live test 6 of 24 (`CS3` and
  `Mass` back on Tier 2, `Top6` back on ccall); without the blob guard they fail 1
  and 2 (`EnumRec` rides a ccall as a blob).
- abi_fuzz, C and C++ modes, seeds 1–106, Julia 1.12.7 and 1.13.2: C mode
  changes no check against the previous commit on either Julia. C++ mode changes
  none on 1.13.2; on 1.12.7 two went from FAIL to PASS (59/cpp `ret/f1`, 77/cpp
  `ret/f2`), both among the 12 X2/X5 checks that read garbage and already
  differed between the baseline runs. The fuzzer's C++ functions have C++
  linkage and are not `noexcept`, so they reach Tier 2 whatever the record; this
  routing is exercised by `test_cpp_record_routing.jl`, not by the fuzzer.

### Tier 3: where C returns a record is decided per target, from real alignment (C1, 2026-10-18)

- **A small record returned by value could get a hidden return pointer the
  callee does not read.** GeneratorC decided that a record returns through
  memory by testing each member's offset against the member's SIZE, capped at
  8, as if that were its alignment. `{ char m1; short m2[3]; }` is 8 bytes and
  comes back in RAX on both ABIs, but its 6-byte array at offset 2 looked
  unaligned, so the wrapper passed a hidden sret pointer. The callee then read
  every argument one register late and the return was garbage, with no error.
  abi_fuzz found it (C1, seeds 3 and 36): `f4` read `p1.m1` as `0x62ffbc10`
  instead of `0x79aa7901`, and seed 3's `f2` crashed. The same test sat in the
  blob-parameter trap and the blob-return guard, and none of the three knew
  Win64, which returns 1, 2, 4 and 8 bytes in RAX whatever the members are.
- **Alignment comes from the field's type** (`_c_has_unaligned_field`): an
  array's element type, each field of a nested record at its offset from the
  outermost record (a packed member record can hold an unaligned field),
  every element of an array of records, and an enum's underlying integer. The
  size test never looked at nested records or enums at all, because the
  metadata records them with size 0. Bitfields are skipped, as clang skips
  them.
- **The return class is per target** (`_c_return_in_memory`): Win64 by size
  alone, SysV above 16 bytes or with an unaligned field. `_c_record_abi()` is
  the one place the C generator reads the target, and
  `generate_introspective_module_c` takes it as `abi`, so a test asks for
  either convention on any host.
- **The wrapper passes the hidden pointer itself only where Julia's lowering
  would not** (`_c_explicit_sret`): SysV with an unaligned field at 16 bytes or
  less, a rule Julia's lowering never applies. Win64 decides by size on both
  sides, so there Julia passes the pointer. Above 16 bytes the explicit form
  stays where it was. A first version sent every Win64 hidden-pointer return
  through the explicit form, and abi_fuzz showed that also widens C2: that
  branch never applies the `long double` trap, so units 58 and 61 refused the
  whole wrapper instead of one function. The explicit form now fires on Win64
  only where it did before.
- **Blob parameters** (`_c_blob_param_misclassified`): the SysV trap for a
  ≤16-byte blob whose record is MEMORY class uses the real alignment, and also
  refuses a record whose alignment cannot be proven. It no longer fires on
  Win64, where the record and its blob are passed the same way (1/2/4/8 bytes
  in a register, anything else by reference). The float-member trap is
  unchanged on both targets.
- **Blob returns**: a ≤16-byte float-bearing blob is exempt from the trap when
  C returns it through a hidden pointer on this target. On Win64 that makes the
  3, 5, 6, 7 and 9 to 16-byte ones callable; a 1, 2, 4 or 8-byte one is still
  refused, because the guard's SSE premise is SysV's.
- Tests: `test_c_sret_classification.jl` (runtests) checks the classifier and
  the emitted ccall for both ABIs on any host. `test_c_sret_abi.jl`
  (devtests) builds `c_sret_test`, calls every function and checks every
  argument and return member, then checks `_c_return_in_memory` against
  clang's lowering of the same file for `x86_64-unknown-linux-gnu` and
  `x86_64-w64-windows-gnu`. `test_c_generator_policies.jl` now checks the
  blob-parameter trap for both ABIs. Negative-checked by restoring the old rule
  through the exact method signatures: the unit test fails 52 of 114 checks,
  and the live test fails 23 of 36 (`f4` reads `0xfffffffff41fd5f0` for
  `0x79aa7901` and returns zeros, then `f2` crashes the child).
- **abi_fuzz on the first run's seeds** (shape seeds 1–106 and identifier
  seeds 1–40, C mode, Julia 1.12.7 and 1.13.2, which agree check for check):
  the four C1 FAILs (3/c and 36/c, `call` and `ret`) pass, and both saved
  repros pass all 8 checks. 19 ABI-trap GAPs become PASS on Win64. 9 of them
  are ≤16-byte packed or bitfield blob parameters with an unaligned field,
  which Win64 passes like the record; 10 are 15- and 16-byte float-bearing blob
  returns, which Win64 returns through a hidden pointer. Totals go from 2214
  PASS, 8 FAIL and 76 GAP to 2237, 4 and 57; the 4 FAILs left are C2. In
  identifier mode one check changed: unit 3n's `include` was a trap stub only
  because the size test called its packed parameter misaligned. As a real
  ccall it names `__x_s`, which the sanitizer declares as `_x_s` (the I6
  class), so the wrapper is now refused when it is written instead of failing
  to load on I5.
- **Hub**: the 19 C packages, re-wrapped without a rebuild, give wrappers
  byte-identical to unpatched HEAD's apart from timestamps, under the Win64
  rules and under the SysV rules (generated on Windows from the same
  metadata). Every package's verifier passes.
- SysV is covered by the unit test and by clang's lowering; no SysV callee has
  run.

### abi_fuzz: generated C shapes as a differential ABI test (2026-10-18)

- **`test/tools/abi_fuzz.jl`**, a hand-run tool wired into no suite. Each seed
  generates one C unit with structs, unions, bitfields, packed and
  `#pragma pack` records, anonymous members, enums, arrays, by-value and
  pointer parameters and spills, plus an exported `sizeof`/`offsetof` table. It
  wraps the unit as C (Tier 3) and the same text as C++ (Tier 2), and checks in a
  child: layout against the table (read through `cglobal`), the accessors, every
  argument member as the callee recorded it, and every member of each returned
  record against a formula. `--ident` swaps in identifier stress (Julia keywords,
  Base/Core names, the wrapper's own names, all-underscore names). Failures are
  delta-debugged into a minimal `.c` under the work directory.
- Negative-checked: with `GeneratorC._resolve_exact_layout`'s offset proof
  removed in-process (`--neuter=resolve_exact_layout`), 11 of 30 C units fail
  59 checks, the first at seed 3. The unpatched run fails none of them.
- It finds bugs and does not fix them. The first runs' findings went to the
  maintainer, not into this file.

### Switching toolchains rebuilds what the switch affects (2026-10-18)

- **A Tier 1 wrapper built on Julia 1.13 failed on Julia 1.12** at the first
  call, with "Failed to parse LLVM assembly: expected icmp predicate". Its slices
  are IR printed by 1.13's libLLVM 20, and `icmp samesign` does not parse in
  LLVM 18. A copy of Hub zlib built on 1.13 and loaded on 1.12 failed 18 and
  errored 59 of its `test_deep.jl` assertions; built on 1.12 and loaded on 1.13
  it passed.
- **Kernels keep their ccall body under another libLLVM major.** The Tier 1
  registry records `TIER1_LIBLLVM_MAJOR`, and every `@generated` kernel adds it
  to its generation-time doubts. Rebuilding under the running Julia brings the
  slices back.
- **The slice cache is per libLLVM major** (`Slicer._slice_cache_key`). A `wrap`
  after a Julia switch, with no rebuild, sees the same module text, and the cache
  served the other Julia's slices.
- **Every persisted build key carries the toolchain.** `_toolchain_identity` is
  Julia's libLLVM version for a C project (the JLL clang is pinned to it) and
  the system clang for a C++ one, plus the target triple. The project hash had
  no toolchain in it: a Julia switch rebuilt a project only because `Base.hash`
  hashes the same bytes differently in 1.12 and 1.13. The project hash and the
  per-file IR fingerprint are now SHA-256 (`_key_feed!`), stable across Julia
  releases, so a Julia switch rebuilds a C project and no longer recompiles a
  C++ one. The registry's `hash_config` carries the identity too, so `use()`
  under one Julia does not serve a build made under another.
- The first build after upgrading recompiles every file once: the fingerprint
  format changed.
- `test_toolchain_switch.jl` (runtests) covers the identities, both keys, the
  slice cache key, and a kernel run against the C runtime's `abs` with a slice
  that makes llvmcall and ccall disagree. Negative-checked: with the guard
  disabled the kernel took llvmcall under a mismatched major (777, not 5);
  without the toolchain the project hash did not move; without the major, the
  cache key did not either.

## v4.2.1 (2026-10-17, never registered; released in v4.3.0)

**LLVM.jl 10.** Patch. LLVM.jl compat widens to `9, 10`. Tier 1's Slicer and
static promotion read the names whose meaning LLVM.jl 10 changed through
`LLVM.API`, so they behave the same on both majors; nothing else needed a
change. A Slicer exception now demotes that one function to ccall instead of
failing the wrap. On Windows the Tier 1 suites run, and `[wrap.tier1]
allow_setjmp` is ignored with a warning, because a longjmp back into a JIT'd
slice frame ends the process there. CI runs runtests.jl on Julia 1.12 and 1.13,
on Linux and Windows. No API change, and a 4.2.0 wrapper keeps working. Tier 1
slice caches rebuild once (`SLICER_VERSION` 5). Verified on Windows (MSYS2
CLANG64, LLVM/MLIR 23.1.3): runtests.jl, the full devtests.jl and the three
Tier 1 suites on LLVM.jl 10.2.1 under Julia 1.12.7 and 1.13.2, and the Tier 1
suites and runtests.jl on LLVM.jl 9.13.2. Hub lua builds and wraps with Tier 1
(342 functions on llvmcall) and passes test_deep. Tier 2 was not re-run on
Linux.

### LLVM.jl 10 and Julia 1.13 in the support window (2026-10-17)

- `Project.toml` compat is `LLVM = "9, 10"`. Every other LLVM.jl call in
  RepliBuild (the C bucket's in-process link and opt, and the rest of static
  promotion) means the same on both majors.
- CI runs runtests.jl on Julia 1.12.7 and 1.13.2 on both hosts. It resolves
  fresh, so it now tests LLVM.jl 10; LLVM.jl 9 is checked only where an
  environment pins it.

### Tier 1: a Slicer throw demotes that function instead of failing the wrap (2026-10-17)

- Nothing caught on the way from `_tier1_slice_prepass` through
  `Slicer.slice_library` to `_extract!`, so one exception while slicing one
  function failed `RepliBuild.wrap` for the whole library. Two of the LLVM.jl 10
  breaks below did exactly that: the `TypeError` on any internal constant, and
  the verifier rejecting `noinline` + `alwaysinline`. Tier 1's doctrine is that
  any doubt resolves to ccall, and a Slicer exception is a doubt.
- **Per target.** `slice_library` runs `_extract!` in `try`/`finally`, so the
  clone is always disposed, and turns an exception into that target's refusal,
  tagged `:slicer_error` (`Slicer.errored`). The reason is one line naming the
  exception and the innermost `Slicer.jl` frame. An errored refusal is never
  cached, so a transient failure does not stick until the module or
  `SLICER_VERSION` changes. `InterruptException` still propagates.
- **Whole module.** A throw that escapes `slice_library` (reading or parsing the
  promoted module, stripping debug info, the `llvm.*` sweep) gives one `@warn`
  with its backtrace, and Tier 1 is off for that wrap. Every function uses
  ccall, as when the promoted module is missing.
- **Visible.** The `tier1:` summary line counts `errored` apart from `refused`,
  and one `@warn` lists each errored function with its reason. Each one is a
  Slicer bug.
- `SliceResult`, the cache format and `SLICER_VERSION` are unchanged.
- Tests, in the two quarantined suites (still unwired):
  - `test_slicer.jl` slices synthetic IR, ahead of the fixture build, whose
    target with `optnone` and no `noinline` fails `LLVM.verify`. The other
    targets still slice, and only verdicts are cached. It also checks that an
    interrupt propagates.
  - `test_tier1_dispatch.jl` ends with two real wraps of the fixture. One
    injects a throw for one target with a more specific `_extract!` method.
    The wrap succeeds and warns, that function dispatches through ccall, the
    others keep Tier 1, and no cache entry is written for it. The other wrap
    has an unparseable promoted module: the wrap succeeds with Tier 1 off.
  - Negative-checked by neutering, one at a time, the per-target catch, the
    no-cache rule, the interrupt rethrow and the whole-module catch.
- Verified on Windows (MSYS2 CLANG64) with LLVM.jl 9.13.2 and 10.2.1 on Julia
  1.12.7 and 1.13.2. `test_tier1_dispatch.jl` passes 294/294 on all four (254
  before, plus 40). `test_slicer.jl` gains 19 passes and otherwise gives what it
  gave before on Windows: 11 errors from its `ccall(:dlsym)` line, and the
  `st_guarded_div(1, 0)` assertion skipped by the driver it was run through,
  because that longjmp ends the process. runtests.jl passes (1995) on all four.

### Tier 1 on LLVM.jl 10: the Slicer and static promotion read what they read on 9 (2026-10-17)

- **LLVM.jl 10 changed four names that Tier 1 reads, at five call sites.** Two
  failed loudly in the Slicer: `local_unnamed_addr` is gone and `unnamed_addr`
  returns the `UnnamedAddr` enum (a `TypeError` on any internal constant), and
  `inline_asm!` is gone. Two were silent. An attribute's `kind` is a Symbol, so
  `_has_enum_attr` never matched the integer kind ID: a `returns_twice` callee
  lost its `:setjmp_family` hazard, and a `noinline` target also got
  `alwaysinline`, which the verifier rejects, so the wrap failed. And
  `isconstant(gv)` now asks whether a value is an LLVM `Constant`, which every
  global is. In the Slicer a mutable static took the constant branch, and was
  embedded when it carried `unnamed_addr` or `local_unnamed_addr` (the cJSON
  divergence class). In `_promote_statics_libllvm` no internal mutable static
  was promoted, so every C library built with LLVM.jl 10 exported fewer
  `__rb_*` symbols; only Tier 1 reads them.
- **On LLVM.jl 9 the Slicer never cleared module asm.** `inline_asm!(mod, "")`
  appends unless given `overwrite=true`. A slice with module asm is
  hazard-gated to ccall, so no wrapper shipped one.
- All five call sites read through `LLVM.API`, the C binding, which is the same
  in LLVM.jl 9.13.2 and 10.2.1 for Julia's LLVM 18 and 20, so there is no
  version branch. `SLICER_VERSION` is 5: a v4 cache entry written with LLVM.jl
  10 can lack `:setjmp_family`.
- `test_slicer.jl` and `test_static_promotion.jl` each open with a testset on
  synthetic IR, ahead of the fixture build, so it runs where
  `test/slice_test/replibuild.toml` is missing. Negative-checked on Julia 1.12
  and 1.13: with a read put back, its testset fails on LLVM.jl 10, and the
  module-asm one fails on 9 too.
- Verified on Windows (MSYS2 CLANG64) with LLVM.jl 9.13.2 and 10.2.1 on Julia
  1.12.7 and 1.13.2. The three Tier 1 suites give unpatched main's results on
  LLVM.jl 9, plus the new testsets. On Windows those results include failures
  that predate this change: `ccall(:dlsym)` and `nm -D` in `test_slicer.jl` and
  `test_static_promotion.jl`, and `test_slicer.jl`'s direct llvmcall of
  `st_guarded_div`, whose longjmp ends the process. runtests.jl passes. Hub
  zlib builds, wraps and passes test_deep 271/271 on all four; all 102 of its
  Tier 1 kernels compile to llvmcall, and LLVM.jl 9 and 10 give the same Tier 1
  surface, promoted module and byte-identical slices.

### Tier 1 suites pass on Windows; setjmp slices stay on ccall there (2026-10-17)

- **`test_static_promotion.jl` read the dynamic symbol table with `nm -D`, and a
  PE file has none**, so "Dynamic symbol table matches" errored before asserting
  anything. On PE the listing is now the export directory (`_pe_exported_names`),
  which holds names only. Each export's kind (`T`, `D`, `R`) comes from the COFF
  symbol table, restricted to the exported names: mingw links its C runtime
  statically, so that table also defines `DllMainCRTStartup` and friends. ELF
  still reads `nm -D`, and every assertion reads the listing unchanged. A
  fixture-free testset drives the listing with canned ELF and PE output on every
  host.
- **Both Tier 1 suites called `ccall(:dlsym, …, C_NULL, …)`, and Windows has no
  `dlsym`.** They now call `Wrapper._symbol_resolves_via(C_NULL, …)`, which is
  that same ccall on ELF and walks the loaded modules on Windows.
- **A longjmp back into a JIT'd slice frame ends the process on Windows**
  (STATUS_BAD_STACK, 0xC0000028). `test_slicer.jl` llvmcalls `st_guarded_div`'s
  slice directly, and with (1, 0) it took the rest of the file down. Julia JITs
  ELF objects on Windows and registers each function's unwind info for exactly
  `[start, end)`. The slice ends in `call longjmp`, so the call's return address
  is the function's end and nothing covers it; the COFF build of the same
  function pads the call with an `int3`. mingw's `setjmp` hands the C runtime a
  frame, so `longjmp` is an SEH unwind that has to walk back into that frame,
  and it stops there. In a probe, nulling `_setjmp`'s frame argument makes (1, 0)
  return -1, and so does a trap after the call. Neither rewrite is applied.
- **`[wrap.tier1] allow_setjmp` is ignored on Windows**, with a warning that says
  why. `_setjmp_slices_safe` holds the fact; `_tier1_allowed_hazards` and
  `test_slicer.jl`'s (1, 0) call both read it, and `test_windows_port.jl` drives
  it with both spellings. Before, the knob bypassed the hazard gate on Windows,
  and the pre-flight then demoted the slice by accident: GetProcAddress on the
  DLL's handle cannot see C runtime symbols such as `_setjmp`, and the warning
  blamed static promotion.
- Verified on Windows (MSYS2 CLANG64) on Julia 1.12.7 and 1.13.2, with LLVM.jl
  9.13.2 and with LLVM.jl 10.2.1 plus the Slicer and promotion fixes for it:
  `test_static_promotion.jl` 16/16 and 74/74, `test_slicer.jl` 153/154 (the
  (1, 0) call skipped) and 169/169, `test_tier1_dispatch.jl` 254/254, plus the
  testsets those fixes add. runtests.jl is 2006/2006 in all four configurations.
  Negative-checked: with the PE filter neutered, the fixture-free testset fails
  4 of 16; pointed at a promotion-less build, the PE listing misses all nine
  promoted names; with every kind faked as `T`, the data and read-only checks
  fail; the unpatched `test_slicer.jl` errors 11 times on `dlsym`; with
  `_setjmp_slices_safe` neutered, `test_slicer.jl` dies with 0xC0000028 and the
  policy testset fails 4 of 11.

## v4.2.0 (2026-10-16)

**LLVM 23 and Windows, end to end.** Minor. New: `[platform.<host>]` tables in
replibuild.toml, JSON 1.x, the JLCS dialect on LLVM/MLIR 23, and end-to-end
coverage of AOT thunks; LLVM.jl compat is 9. Nothing removed from the exports.
Behaviour changes: relative `[compile]` paths resolve against the project root,
codegen runs at `[link] optimization_level`, and a C++ package compiles its `.c`
files with the system clang. A 4.1.1 wrapper keeps working. Build-time fixes, the
libc++ exports among them, need a rebuild (delete `.replibuild_cache/project_hash`);
generator fixes need a re-wrap. Verified on Windows (MSYS2 CLANG64, LLVM/MLIR
23.1.3) with runtests.jl and the full devtests.jl. CI runs runtests.jl on Linux and
Windows; Tier 2 was not re-run on Linux for this release.

### discover runs without a system LLVM (2026-10-16)

- A C project needs nothing but Julia, yet `discover` raised "LLVM Toolchain not
  found" on a machine with no LLVM installed. Its dependency walk is optional and
  meant to skip when clang is missing, but `get_tool` raises, rather than
  returning `""`, when there is no toolchain at all, and `discover` then also
  warmed the toolchain for a later build. Both now degrade: the walk is skipped
  with a warning that says why, the config takes the scanned sources, and a build
  that needs system LLVM still raises its own error. The first GitHub CI run found
  it: the Ubuntu runner has no LLVM that RepliBuild accepts, and
  `test_registry.jl`'s c_test integration runs `discover` on a fresh clone.
  `test_discover_without_llvm.jl`, negative-checked.

### AOT thunks: an end-to-end suite, and a void thunk that threw no longer ends the process on Windows (2026-10-16)

- **`test_aot_thunks.jl` (devtests §5c).** The AOT path had no end-to-end
  coverage: stress_test/verify.jl hand-rolls its own link, and
  test_introspection.jl only checks that a failed AOT build leaves no stale
  library. The new suite builds `test/aot_test/` (C++, no STL) with
  `[compile] aot_thunks = true` through discover → build → wrap twice: cold, and
  again once `thunk_manifest.json` exists, when AOT must emit exactly the
  manifest's thunks (the cold build also emits thunks for Tier-3 functions,
  which no wrapper binds). Each pass checks the companion library and its
  exports, the manifest, and the wrapper (AOT slots, no JIT dispatch), then a
  child process calls through it: plain members, records by value at 8, 16 and
  24 bytes (argument and return, including a member returning one), virtual
  override dispatch, and a C++ throw for every return shape (i32, f64, pointer,
  sret, void, a non-`std::exception`), with no JIT engine created.
- **`JITManager._invoke_void`.** On Windows the dialect lowers `try_call` to a
  plain call, so a thunk does not catch; libJLCS's `jlcs_guard_*` does.
  `_invoke_call` (every non-void return) and the JIT `invoke` void form went
  through the guard. The AOT void forms (`invoke_aot` and `invoke_aot_ptr`
  without a return type) called the thunk directly, so a void C++ function that
  threw through an AOT thunk killed the process with no message (exit status
  0x20474343, the SEH code of an uncaught C++ exception), where the JIT wrapper
  raised `CxxException` for the same throw. All three void forms now share
  `_invoke_void`. Found by the suite's `throw_void` probe.

### LLVM 23 on Windows: libc++ members reach the export table; STL thunks bind only exported symbols (2026-10-16)

Found by running devtests on Windows (MSYS2 CLANG64, LLVM/MLIR 23.1.3) for the
first time since the toolchain moved.

- **libc++ 23 hides its own functions.** `__configuration/namespace.h` opens every
  `std` block with `#pragma clang attribute … push(__attribute__((
  __exclude_from_explicit_instantiation__, __visibility__("hidden"),
  __abi_tag__(…))), apply_to = function)`. It applies on COFF, where
  `_LIBCPP_HIDDEN` is empty, and no macro reaches it
  (`_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS` does not). On MinGW clang emits
  `-exclude-symbols:` for hidden definitions, so `std::vector<int>::vector()` and
  the rest were `T` in nm and absent from the DLL's export directory. stl_test had
  513 exports and passed 28/28 on LLVM 22; on LLVM 23 it had 222, and all seven STL
  testsets errored with "Symbol not found": one unbindable thunk failed the
  library's whole JIT module, `make_ints` included. `create_library` now makes
  hidden `std::__N` definitions default visibility in the IR before codegen, for
  C++ packages with `[types] templates` (`_unhide_libcxx_ir`), on the merged module
  and on the per-TU fallback. A package's own `-fvisibility=hidden`, clang's
  helpers and function-local entities keep their visibility. stl_test: 531
  exports, 28/28.
- **The STL method table takes only bindable symbols.** `extract_stl_method_symbols`
  read `nm -g`, which on PE is the COFF symbol table (an unexported member is still
  `T`), and also accepted local `t`/`w`. It is now gated by `_runtime_symbol_names`
  (the PE export directory, else `nm -D`), the same set `verify_wrap_surface` now
  reads through. An unbindable member drops out of its container's wrapper with a
  warning instead of taking the engine down, and a C1 constructor first in nm no
  longer shadows an exported C2.
- **Known limitation, unchanged.** Seven `std::basic_string<char>` members (`at`,
  `find`, `erase`, `insert`, `push_back`, `reserve`, the destructor) are
  explicit-instantiation copies that `libc++.dll` also exports, and lld does not
  auto-export a symbol whose `__imp_` twin is in an import library. The gate leaves
  them out and the warning names them, which matches Linux, where they are
  undefined in the library. With no destructor thunk `CppString` registers no
  finalizer, so a string grown past the small-string buffer leaks; nothing crashes.

`test_libcxx_exports.jl` (33 tests), both guards negative-checked. Hub fmt, the only Hub
package with `[types] templates`, needs a rebuild on Windows, not a re-wrap.

### Pre-release fixes: devtests runs to the end (2026-10-16)

- **`test_c_inprocess.jl` names the fixture's files absolutely.** It copies the
  discovered c_test config into a temp dir and changes only `root`. Since the
  portable-paths change `source_files = ["src/mathkit.c"]` resolves against that
  root, so clang got `<tmp>/src/mathkit.c` and three testsets errored, which also
  stopped devtests before the ten suites after it. Test-only; devtests had not run
  since that change. 14 passes plus the known skip for the LLVM-18 fallback tools.
- **`_pe_export_intent` closes what it opens.** It read IR with `eachline(path)` and
  returned from inside the loop. That iterator closes the file only once exhausted,
  so the handle stayed open until GC, and on Windows an open handle blocks deletion:
  runtests logged EBUSY cleaning a temp dir, and the same handle could hold IR that
  `clean` or a rebuild had to remove. It was the only early-exit `eachline` over a
  path in `src/`. `test_windows_port.jl`.
- **`deps/build.jl` stops telling Windows users the dialect is Linux-only.** Every
  `Pkg.add`/`Pkg.build` on Windows printed "RepliBuild's dialect is Linux-only",
  which has been false since 4.0.0, and the script looked for `libJLCS.so`. It now
  names the library by host extension and reports whether `libJLCS.dll` is built
  and loads. Otherwise it gives the CLANG64-shell build step, because `Pkg.build`
  does not run in one, and points at `check_environment()` for the package list
  rather than repeating it. install.md: the verified toolchain is 23.1.3 under
  CLANG64, and the WINDOWS_PORT.md link points at `docs/updates/`.

### Windows compatibility: per-host config, portable paths, the library's own free (2026-10-11)

Found by sweeping the Hub on Windows (Julia 1.12.7, MSYS2 CLANG64, LLVM 23).

- **`[platform.<host>]` overlays.** A replibuild.toml could only be right for one
  OS: sqlite, glfw, miniaudio, curl and llamacpp link `dl`/`rt`, which do not
  exist under mingw; lua builds with `-DLUA_USE_LINUX`; glfw excludes the win32
  sources. `apply_platform_overlay` merges `[platform.linux|windows|macos]` over
  the base in `load_config` before any section parser runs (tables key by key,
  everything else replaced, unknown names a hard error). `discover(force=true)`
  preserves the tables and registry staging reads through them. Older RepliBuild
  ignores the table and sees the base, so Hub configs keep the Linux base and
  override on Windows. `test_platform_overlay.jl`; config.md §2.15.
- **Config paths are portable.** Relative `[compile] source_files` /
  `include_dirs` reached clang as written, so they meant "relative to the process
  cwd"; they now resolve against `root` at load, which is what registry staging
  already assumed. `save_config` (so `discover`) writes paths under the project
  relative with `/` separators. Hub hello_world, generated by an older discover,
  carried `/home/…` paths and built on no other machine. `test_portable_paths.jl`;
  config.md §2.16.
- **`libc_free` in every wrapper.** Julia links `msvcrt.dll`; a CLANG64-built DLL
  allocates from the UCRT. mpack's deep test freed the growable writer's buffer
  with `Libc.free` and died with STATUS_HEAP_CORRUPTION. Both generators now emit
  `libc_free(ptr)` bound to the C runtime the library imports
  (`Compiler._pe_crt_dll`, read from its PE import table; `Libc.free` off
  Windows), and `[wrap.cstring_owned] f = "free"` routes through it instead of
  looking `free` up in a DLL that only imports it. mpack test_deep 171/171 on
  Windows with the Hub test using `P.libc_free`. `test_windows_port.jl`.

`runtests.jl` on Windows: 1918 passes, no failures. Each new test negative-checked.

### Windows: slice files survive case folding; link_dirs DLLs load without rpath (2026-10-11)

- **Slice file names.** Tier-1 slices (and the Slicer cache) were named by the raw
  symbol. NTFS folds case, so lua's value macro `LUA_YIELD` and function-like
  macro `lua_yield` shared `replibuild_shim_LUA_YIELD.ll`, the later write won, and
  `LUA_YIELD()` threw "Module IR does not contain specified entry function".
  `Slicer.slice_file_stem` appends 8 hex digits of SHA-1 to any name with an
  uppercase letter; all-lowercase names keep their file. `SLICER_VERSION` 4
  drops caches that may already hold one symbol's IR under another's name. Only
  file naming changed; no Tier-1 policy did.
- **`[link] link_dirs` on Windows.** ELF gets `-Wl,-rpath` per link dir; PE has no
  RUNPATH, so imgui linked against `packages/glfw/julia/libglfw.dll` and then
  could not load ("The specified module could not be found"), at the build's own
  load check first. `Compiler._link_dir_dlls` names the `link_libraries` DLLs found
  in `link_dirs`; the load check opens them first, and both generators emit the
  same preload in `__init__` (next to the wrapper first, then the build-time path).
  Windows binds an import to an already-loaded module of that name, which also
  keeps one copy of glfw's process-global state.

Hub on Windows with these: lua 103/103 + 189/189, imgui 97/97 + 202/202, glfw
38/38, sqlite deep 145/145. `runtests.jl` 1927 passes.

### Machine code from the link level, not -O0; SYSCONFIG.md hides the user's paths (2026-10-11)

- **Codegen opt level.** `create_library` handed clang the optimized IR with only
  `-shared -fPIC`, so every library came out of the `-O0` backend (FastISel and
  the fast register allocator), on every host. On LLVM 23 that backend crashed
  outright on llama.cpp ("Cannot emit physreg copy instruction" in
  `llama_kv_cache::find_slot`; reproduced on the extracted function at `-O0`,
  clean at `-O2`). `_codegen_opt_flags` passes `[link] optimization_level` with
  `-Xclang -disable-llvm-passes`, so codegen runs at that level without
  re-running the IR pipeline; the per-TU fallback gets the same flags.
  `test_codegen_flags.jl`.
- **SysConfigGen on Windows.** `SYSCONFIG.md` recorded the probe's checkout path,
  which on Windows is under the user profile
  (`C:/Users/<name>/AppData/Local/Temp/…`). `_display_path` writes `<tmp>` / `~`
  prefixes instead; the regenerate snippet names the real output directory rather
  than always `config`. Found harvesting curl's Windows config.

### One toolchain per module: a C++ package's `.c` files lost their DWARF (2026-10-11)

The C bucket was chosen per FILE: every `.c` compiled with the JLL clang 18, even
inside a C++ package whose modules are linked by the system llvm-link. LLVM 23's
verifier rejects clang 18's debug info ("Function-local variables are not allowed
in a DICompileUnit's global variables list"); llvm-link printed `warning: ignoring
invalid debug info`, stripped every `!dbg` from that module and exited 0. llama.cpp
had six C files and the linked module kept one C compile unit: 439 `ggml_*` and
all `quantize_*`/`dequantize_*` signatures fell back to `parameters_source =
"inferred"`, so `ggml_version` wrapped as returning `Cint` and `ggml_init` as
returning an integer. Only test_deep's version check noticed.

- The C bucket is a property of the package (`[wrap] language = "c"`). A C++
  package compiles its `.c` files with the system clang as well.
- `_assert_debug_info_kept` fails the build when llvm-link or the IR->library
  clang reports dropped debug info, naming the modules.
- `compute_compile_fingerprint` hashes the system clang's identity for C++
  packages, so an LLVM upgrade by the package manager invalidates the IR cache
  instead of serving the old compiler's IR to the new one's tools.

`test_codegen_flags.jl` ("one toolchain per module").

- **libJLCS drift.** `check_environment` reported "dialect compiled" whenever the
  file existed, so after a package-manager LLVM upgrade it certified Tier 2 on a
  library whose libLLVM import was gone. `build.sh` writes the llvm-config version
  to `src/mlir/build/jlcs_llvm_version`; the doctor fails libJLCS when that
  disagrees with today's llvm-config on major.minor, and otherwise confirms the
  library loads.
- **SysConfigGen and OBJECT libraries.** curl compiles its code into
  `libcurl_object` (CMake gives only SHARED targets `<target>_EXPORTS`, so it
  classifies static) and `libcurl_shared` compiles a stub; `main_target` picked the
  stub and `toml_fragment` proposed excluding all of `lib/`. Selection is
  unchanged, but the probe display and the fragment now say which target holds
  the code and name the `target=` override.
- Docs: the Windows install command includes `mingw-w64-x86_64-binutils` (GNU
  objdump, required for DWARF extraction) and the CLANG64 PATH requirement;
  `docs/updates/WINDOWS_PORT.md` records this pass.

### libJLCS builds against LLVM/MLIR 23 (2026-10-11)

MSYS2 CLANG64 moved to LLVM/MLIR 23.1.3, and `src/mlir/build.sh` failed in
`JlcsJIT.cpp`: LLVM 23's `LLJITBuilder::ObjectLinkingLayerCreator` takes
`(ExecutionSession &, jitlink::JITLinkMemoryManager &)`. RuntimeDyld brings its own
memory manager, so an adapter guarded by `LLVM_VERSION_MAJOR >= 23` accepts the
new argument and ignores it; LLVM 22 compiles the original path unchanged.

Verified on Windows (Julia 1.12.7, LLVM/MLIR 23.1.3): `MLIRNative.test_dialect()`
passes, `check_environment` reports both tiers, and Hub tinyxml2 builds, wraps and
passes `test.jl` (11/11) through JIT-mode Tier-2 thunks.

### Fixes salvaged from the quarantined internalize-symbol-tables branch (2026-10-11)

That branch (0399c6e) came off a compromised machine and is dropped unmerged. Its
symbol-table internalization (`ObjectReader`) is not taken: `nm` and GNU `objdump`
stay the authorities. The fixes it carried that do not depend on `ObjectReader`
were re-derived against main, reviewed line by line, and re-typed. Nothing was
cherry-picked.

- **Thunk manifest in JIT mode.** 1c4255b put `_write_thunk_manifest` inside the
  `aot_thunks` branch. A failed AOT link falls back to a JIT wrap, which then
  recorded nothing, so the next build emitted the same unfiltered thunk set and
  failed again. The write is unconditional now (`test_thunk_manifest.jl`).
- **PE vmethod gate.** `parse_vtables` took method addresses from `nm`. On PE a
  method the DLL does not export is still a global `T` in the COFF symbol table
  (ELF links the same hidden method as a local `t`), so the legacy vmethod pass
  emitted `thunk_*` calls PE cannot bind. `_gate_to_exports` filters them through
  the export directory (`_pe_exported_names`). ELF is unchanged.
- **ELF symbol-version suffixes.** The three `nm -D` readers
  (`verify_wrap_surface`, `_runtime_defined_names`, `_assert_aot_thunks_present`)
  kept `name@@VER`, so a library built with a version script looked as if it
  exported nothing DWARF named. They share one parser now,
  `Compiler._nm_dynamic_names`, which cuts the suffix. It takes mangled output
  only, since demangled names can contain `@`; `extract_symbols_nm` reads
  demangled output and is untouched.
- **mingw-w64-crt 14.** `__mingw_dll_do_global_ctors/dtors` join the PE
  static-runtime set in `test_symbol_hygiene.jl` (`dllcrt2.o` references them,
  `libmingw32.a` defines them; checked against crt 14.0.0.r426).
- `check_environment`'s Windows binutils advice named the CLANG64 package, which
  is llvm-tools. It names `mingw-w64-x86_64-binutils` now, matching
  `_no_dwarf_dumper_error`.

Each new test was negative-checked by neutering its fix. Windows, Julia 1.12.7:
`runtests.jl` 1871 passes, no failures. The PE gate and the suffix cut are tested
on fabricated `nm` output only; neither has been exercised on a real versioned ELF
library or a real C++ DLL yet, and `devtests.jl` was not run.

### JSON 1.x: one read path for every metadata file (2026-10-02)

`[compat] JSON` was `"0.21"`, and JSON.jl is at 1.10. In an environment that already
held JSON 1 — 176 General packages have a release requiring it — Pkg answered
`]add RepliBuild` with **2.0.3**, the last release that admitted JSON 1, and no
error; asking for RepliBuild 4 there was unsatisfiable. Widening the bound alone
would have traded that for a hard break: every metadata read was
`JSON.parsefile(path; use_mmap=false)`, JSON 1 has no `use_mmap` (`MethodError` in
`JSON.LazyOptions`), and its bare `parsefile` returns `JSON.Object`, not `Dict`. A
one-file C library compiled and linked, then `build` died at its first metadata
read. Several of those reads sit inside a `try`, so other paths would have fallen
back quietly instead of failing.

All twelve reads now go through `RepliBuild._read_json(path) =
JSON.parse(read(path, String); dicttype=Dict{String,Any})`. It never maps on either
major, so the Windows `clean()` fix holds, and it returns the same types on both.
Compat is `"0.21, 1"`. `test_json_mmap_hygiene.jl` refuses any other read entry
point in `src/` and checks the helper's types under whichever major resolved. It
was negative-checked three ways: a reverted call site is named; a reverted helper
is caught by its text on 0.21 and by text and types on 1.x.

Verified on JSON 0.21.4 and 1.9.0 (1.10.0 is not installed on the reference host),
with Tier 2 on MLIR/LLVM 22.1: `runtests.jl` 1847 and `devtests.jl` 1018 passes on
both, no failures. With the export-order fix below, the seven integration fixtures
rebuilt in one directory are byte-identical across the two majors, wrappers and
`.so` alike.

Writes are unchanged: `JSON.print(io, x, 2)` exists on both majors. As text, JSON 1
writes object keys in a different order and drops the trailing newline; the data is
equal. Two differences no RepliBuild-written file exercises today: JSON 1 refuses
NaN/Inf, where 0.21 wrote `null`, and reads an integer outside Int64 as `BigInt`,
where 0.21 silently wrapped it (`18446744073709551615` read back as `-1`). None of
the 113 metadata files on the reference host holds a float or an out-of-range
integer.

### Wrapper text no longer follows metadata key order (2026-10-02)

Found by A/B-ing the JSON change: rebuilt in one directory, six of the seven
integration fixtures' wrappers were byte-identical across JSON majors, and
`StressTest.jl` swapped two names in its `export` line. Both generators collect
struct and enum names into `Set`s filled from `struct_definitions`, which is read
back from `compilation_metadata.json`, so their iteration order was the file's key
order — and the two JSON majors write keys in different orders. Re-wrapping with
every metadata `Dict` rebuilt in four different insertion orders and capacities
also moved `StlTest.jl`'s export line, through GeneratorCpp's per-container STL
factory loop.

The type-export loops in both generators, and GeneratorCpp's STL factory section
(the per-container loop and its DWARF `byte_size` lookup), now iterate sorted keys.
Under the same four orderings all seven wrappers are byte-identical; without the
fix the same harness moves two of them. Export order carries no meaning to Julia —
this keeps the byte-identical regression diff honest. Expect a one-time reordering
of `export` lines (and of STL factories) on the next re-wrap.

The DWARF `byte_size` lookup itself never matched anything: STL records are not in
`struct_definitions` (0 hits across 57 metadata files), so every STL factory is
sized by `get_stl_container_size`'s table. Recorded in CLAUDE.md, not changed.

### LLVM.jl compat: `"9"`, not `"9, 10"` (2026-10-02)

`[compat] LLVM = "9, 10"` (since 2026-05-08) admitted a major that has never been
released — General's newest LLVM.jl is 9.13.2 — so the resolver was free to pair
RepliBuild with LLVM.jl 10 the day it is registered, untested. Now `"9"`. Every
`LLVM.*` / `LLVM.API.*` name used under `src/` is present in 9.8.2 and 9.13.2;
9.0–9.7 were not checked. This binds only the next release: General still records
`LLVM = "9 - 10"` for every registered 3.x and 4.x version, and capping those takes
a General PR.

### Test suite: one skip rule, one output convention, the audit's proofs in-repo (2026-09-27)

Test-only; nothing under `src/` changed.

- **No file can end devtests early.** Six toolchain files skipped with `exit(0)`
  when libJLCS was missing, and inside an `include` that ends the whole run with a
  success status. `test_mlir_templates.jl` ran first in the libJLCS group, so a
  machine without the dialect skipped most of devtests and reported green. They
  wrap their body in `if requires(...)` from the new `test/support/TestSupport.jl`,
  which records a skipped testset. `runtests.jl` refuses an include-time `exit` in
  any suite-included file, checked on the parsed AST (a subprocess driver in a
  string literal is not a hit). The guard was run against the old tree first and
  named all seven sites.
- **Crash- and hang-mode checks share one runner.** `run_isolated` runs a child
  julia with a SIGKILL timeout and output through a file, and `probe_verdicts`
  parses the `PROBE <name>: PASS|FAIL` protocol. The two probe parents had no
  timeout, and `test_jlcs_invariants.jl` ran whatever `julia` was on `PATH`.
- **`@test`, not prints.** A failed `@test` inside a testset does not throw, so the
  `println("  ✓ …")` after it printed on failure too. Removed from
  `test_mlir_templates.jl`, the fixture `verify.jl` scripts and the pass banners.
  One of them, "✓ NumberUnion (getters available)", had no test behind it at all;
  `c_test/verify.jl` now passes a C union by value both ways.
  `test_jlcs_invariants.jl` §C accepted every outcome ("record, don't presume");
  its answer has been known since 2026-07-16 and is asserted.
- **Retired and moved.** `stress_test/run_jit_test.jl` (a March `@assert` script, a
  subset of `stress_test/verify.jl`, pre-v4 tier numbering) and `c_test/debug_aot.jl`
  (a March debug script that wiped `c_test` when run) are deleted. The hand-run
  Hub tools move to `test/tools/`.
- **Audit gaps ported.** Eight 2026-09-26 fixes were proven only by the gitignored
  harness in `.claude/audit-2026-09-26/`. Each now has a test, and each test was
  negative-checked by neutering its fix:
  - F7, C-bucket pipe drain: `test_c_bucket_pipe.jl` (CI; a SIGKILL-bounded child
    with 4000 warnings; the old body is killed at the timeout).
  - F9, `__int128`/`_Complex` bind Base's types: `test_wrapper_type_bindings.jl`,
    driving both generators. The existing check asserted membership in
    `Wrapper._FOREIGN_BUILTIN_TYPES`, which no generator reads.
  - F15/F17, a local dependency compiles `.c` and resolves against the TOML, not
    the cwd: `test_dep_cache.jl`.
  - F16, `register` stages a relative-root project's inputs: `test_registry.jl`.
  - F20, SysV register exhaustion: `test_struct_abi.jl` §E against a clang++
    callee (65 instead of 84 with the dialect fix reversed in a scratch libJLCS).
  - F21, array members record their size: `test_dwarf_attribution.jl` (verbatim
    clang 22 readelf), plus `Arr3` in the `stress_test` fixture.
  - F22, C++ anonymous unions are inlined: `Variant` in `stress_test`.
  - F23, a failed AOT pass leaves no stale `_thunks` library: `test_introspection.jl`.
  - F25, global-scope constructors are emitted: `Span` in `stress_test`.

### Empty classes passed by value follow the host ABI (2026-09-27)

SysV drops an empty class: no register, no stack slot, and an empty return is
`void`. Win64 does not. clang for `x86_64-w64-windows-gnu` passes
`struct Unit {}` as `i8` and returns it as `i8`; `alignas(8)` widens that to
`i64`. The thunk used the SysV rule on both hosts, so
`unit_scaled(Unit(), 4, Unit(), 2)` — `x * 10 + y` — returned a garbage
register on Windows. A Win64 empty record of register width is now passed as
a zero of that width (the byte has no state). The thunk stays `void` on an
empty return, and the callee is declared with the `iN` clang actually emits.

## v4.1.1 (2026-09-27)

**The 2026-09-26 audit.** Patch. No new configuration keys, and nothing removed
from the exports. A wrapper generated by 4.1.0 is unchanged until it is
regenerated; the fixes below are in the generators and the DWARF reader.

### Inline static methods, and arrays of pointers (2026-09-27)

- An inline static method's signature lives on the abstract DIE (`DW_AT_inline`,
  no `low_pc`, no `DW_AT_object_pointer`). The concrete copy only has
  `DW_AT_abstract_origin` and the code. The object-pointer fact was read from
  neither, so the wrapper invented a `this` (`EdgeSegment::create`). The abstract
  DIE is now the definition that records it. The concrete copy is not, so an
  inline instance method stays an instance method.
- `NTuple{N, Ptr{T}}` is ordered before `T`, because Julia resolves `T` when the
  struct is defined. If that order is impossible, the field becomes
  `NTuple{N, Ptr{Cvoid}}` instead of failing to load (`DynArray<XMLNode*, 10>`).

### `long double` is an ABI trap, both directions (2026-09-27)

`long double` on x86-64 is the 80-bit x87 type. Julia has no type for it, and
`ccall` has no shape for it. A return comes back in the x87 register `ST0`, where
`ccall` never looks; the return mapper's `NTuple{2, UInt64}` read RAX:RDX and returned
plausible garbage (audit F5b). A parameter is X87 class, passed in memory; the
parameter mapper's `Any` was only caught by the Any-argument guard, which refused to
write the whole wrapper until the symbol was excluded (F5a).

- Both generators now emit such a function as an **ABI Safety Trap** that names
  what it crosses and how to keep it out (`_x87_crossings`, one predicate for both).
  The wrapper writes and loads, and every other function works. Tier-2 thunks
  already refuse it (`by_value_crossing` → `:opaque`).
- A `long double*` is an ordinary pointer and is unaffected.
- Tests: `test_by_value_crossing.jl` "long double is a trap in both generators", and
  `c_test` (`lerp_ld`, `ld_half`; devtests).
- Not built: a thunk could take the value in and out through `f80`
  (`fpext`/`fptrunc` against `Float64`) for C++, which already has Tier 2. The C
  bucket has no thunks, so there it stays a trap.

### Packed C++ structs hold the C layout (2026-09-27)

`__attribute__((packed))` and `#pragma pack` structs (audit F24) were worse than the
audit recorded ("pointer access only"). On a fixture of 12 calls, 7 were wrong.

- **GeneratorCpp emitted fields at natural alignment.** A packed `{char; int}` came
  out 8 bytes with the int at 4 (C: 5 bytes, int at 1). The thunk converted between
  that layout and the C one, so by-value calls worked. Everything else read the wrong
  bytes: pointers into C memory (`unsafe_load` of a C array), arrays Julia wrote for
  C, and any struct embedding one (`Holder { Pk; double }` was 24 bytes, C has 16).
- **A `#pragma pack(2)` struct has padding**, so StructGen did not treat it as
  packed. Natural alignment could not place its members, and it degraded to
  `!llvm.array<14 x i8>`. The SysV classifier never saw a struct, so it was broken
  by value in both directions.
- **The thunk read Julia's value at offsets guessed from `min(size, 8)`**
  (`get_julia_offsets`). That guess is also wrong for an ordinary padding-free struct
  with a byte array: `{char; char[3]; int}` was read at 0, 3, 8 instead of 0, 1, 4.

What changed:

- **GeneratorCpp proves each struct's Julia layout against DWARF.**
  `_prove_julia_layout` applies GeneratorC's exact-or-opaque rule to the fields it
  would emit. When those fields cannot reproduce the C layout, it emits the C-layout
  byte blob with offset accessors instead.
- **The thunk reads Julia's value at DWARF offsets**, and a truly packed return goes
  back as the packed value.
- **StructGen gives a struct natural alignment cannot place a packed body with
  explicit pads.** The classifier then sees the misaligned field and picks MEMORY,
  as clang does.
- **Blob accessors now cover struct and union members with plain names.** They
  used to require a template or scoped spelling.

The proof also found a pre-existing bug. A struct embedding a **named** union stored
it as a pointer, because named unions are `mutable struct`, so the struct was the
wrong size. box2d `b2ClipVertex` (16 bytes against 12) and `b2ManifoldPoint`, and one
fmt specs struct, are now C-layout blobs with every member readable. box2d's verifier
passes 15/15 on the re-wrap.

Measured:

- **Thunks.** Unchanged for all Hub metadata: no Hub package passes a truly packed
  struct by value.
- **StructGen.** 14 structs (llamacpp 2, onednn 12) get a packed body instead of an
  opaque array. Both have the same size and alignment; onednn's verifier passes
  141/141.
- **Re-wrap A/B.** box2d changes 2 structs and fmt 1; clipper2, msdfgen, pugixml,
  stl, tinyxml2 and onednn are identical.
- **Tests.** `test_packed_layout.jl` (CI, verbatim clang 22 readelf; each part goes
  red with its function neutered) and stress_test "packed structs hold the C layout".

### By-value records the thunk could not type (2026-09-27)

A by-value parameter or return whose record had no entry in `struct_definitions`
became `!llvm.ptr` in the Tier-2 thunk, from `map_cpp_type`'s fallback for a
non-identifier spelling or from `generate_jlcs_ir`'s rewrite of a bodyless
`!llvm.struct<"X">`. Its bytes were loaded as a pointer and passed in one register.
On a fixture of 13 such functions, 9 returned a wrong value and 1 segfaulted. The
other 3 were right by accident: `box_get(Box<double>)` returned whatever was already
in XMM0.

- An **empty class** (`struct Tag {}` has no members, so no entry) shifted every
  later argument one register. `with_tag(Tag(), 41)` returned 257, and a call that
  should throw returned instead. Returning one read RAX from a `void` callee: SIGSEGV.
- A **type only declared here** (constructor homing, F4) gave NaN or SIGSEGV.
- A **template or qualified spelling** was never looked up. `Box<double>` went in as
  a pointer and came back from RAX while the callee wrote XMM0. msdfgen's
  `BitmapSection<float, N>` (24 bytes, which belong on the stack), fmt's
  `basic_string_view<char>` and clipper2's `Point<long>` went in as pointers, and
  qualified enums (clipper2, onednn, pugixml, tinyxml2) did too.
- A **toolchain type** the provenance gate leaves out, such as pugixml's 22
  `std::string_view` overloads, passed a 16-byte view as one pointer.
- A **derived class with no members of its own** has no entry either.

What changed:

- **`record_abi` in metadata.** For every record a function takes or returns by
  value, it holds `byte_size`, `pass` (DWARF `DW_AT_calling_convention`: `value` or
  `reference`) and `empty` (no data member, no base). Toolchain types are included.
- **`FunctionGen.by_value_crossing` decides each crossing**, looking names up by every
  scope suffix. The thunk generator and GeneratorCpp act on the same verdict:
  - a record with a layout gets its DWARF type, template and qualified spellings
    included, and a qualified enum gets its integer;
  - an empty record is dropped from the callee's arguments (clang does the same) and
    its slot is skipped; an empty return makes the callee `void`, and the Julia
    function returns the singleton;
  - a record with no layout gets no thunk. The Julia function becomes an **ABI Safety
    Trap** that names the type and the fix (`-Xclang -fno-use-ctor-homing` or
    `-fstandalone-debug`, or `exclude_symbols`), and the module still loads.
- **The method dedup prefers a callable definition over a trap** for one Julia
  signature. pugixml's `append_child`, `set_name` and 20 more now bind the `const
  char*` overload, not the `string_view` one. pugixml's `test.jl` gives that collision
  as the reason it navigates positionally.
- **A record named after a Base binding** (`Pair`, `Vector`, `Type`, …) is emitted as
  `c_<name>`, and parameter and return annotations now use that name too. Before,
  `p::Pair` meant `Base.Pair`, and box2d's `b2Contact::AddType(…, b2Shape::Type,
  b2Shape::Type)` was annotated `::Type`.

### Non-trivial records by value (2026-09-27)

A class with a user destructor or copy constructor is non-trivial for the purposes
of calls. Itanium passes it as a pointer to a caller-owned temporary and returns it
through a hidden sret pointer, at every size. Only parameters of a class with an
emitted destructor were handled. `Owner make_owner(int)`, a 4-byte class, was read
from EAX while the callee stored the object through the register holding the `int`:
SIGSEGV. Above 16 bytes the size-based classifier picks sret anyway, which is why
this went unseen.

- A `pass: reference` return is lowered through an sret slot the thunk owns. The
  slot is the first argument, before `this`.
- A `pass: reference` parameter that has a layout and no destructor to run is passed
  by address. `invoke` already holds a private copy of the bytes.
- `is_ccall_safe` keeps any function with such a record by value off Tier 3.
- A `pass: reference` record with no layout is still passed by address as a
  parameter (the caller supplies a pointer to a real object). As a return it is a trap.

Measured, for both entries:

- **Thunks.** Old and new generator on current Hub metadata give 2,875
  byte-identical thunks across box2d, imgui, llamacpp, stl and hello_world. The
  changed thunks (fmt 219, msdfgen 43, pugixml 8, clipper2 3, onednn 2, tinyxml2 1)
  are all a by-value record or qualified enum that was `!llvm.ptr`. Trapped: pugixml
  22, and fmt 55, a count taken from pre-C4/D4 metadata.
- **Hub verifiers** against the new thunks, without a rebuild: pugixml 13/13,
  tinyxml2 11/11, msdfgen 24 + 1 broken (unchanged), onednn 141/141.
- **Re-wrap A/B.** Wrappers are identical except box2d `AddType`, pugixml's 22 dedup
  winners and fmt's traps.
- **What reaches a Hub package when.** JIT-mode packages get the layout and enum fixes
  at load, because thunks are generated from metadata. Traps and the name fixes need
  a re-wrap. Empty and non-trivial records need `record_abi`, so a **rebuild**.
- **Tests.** `test_by_value_crossing.jl` (CI, verbatim clang 22 readelf; each part
  goes red with its function neutered) and stress_test "by-value records without a
  scalar spelling". That second test fails with the old verdict (`unit_scaled` →
  12804, `cell_doubled` → 6.9e-310).

### The 2026-09-26 audit, continued

Silent-wrong and crash fixes from the re-verification harness. Each one is
the line the harness names.

- **`_Complex` is two types.** DWARF names both widths `complex`. The base-type
  byte size now rewrites that to `complex float` (8) or `complex double` (16)
  before the Julia map, so a wrapper is no longer refused because every complex
  argument was `Any`.
- **`__int128` no longer shadows `Base.Int128`.** The type map already said
  `Int128`; the generator then emitted `struct Int128 end` because that name
  was not in its builtin set. `Int128`, `UInt128`, `ComplexF32` and `ComplexF64`
  are builtins. A call with a real `Int128` reaches the ccall.
- **`NTuple{4, Ptr{Later}}` is not a type named `Ptr_Later`.** The forward-decl
  scan stopped at the first `}`. Brace matching peels it to `Later`, and a
  pointer array lays out as `NTuple{N, Ptr{Cvoid}}` (the width does not depend
  on the pointee, which may not be declared yet).
- **The C-bucket clang pipe no longer deadlocks** on more than ~64 KiB of
  diagnostics. Output is drained into an `IOBuffer` while the process runs.
- **A local dependency compiles `.c` files.** It already compiled `.cpp`.
- **`register` / `use` keep the project.** A relative `root` is resolved against
  the TOML file, not the caller's cwd, and `register` copies that project's
  relative include dirs and sources next to the stored TOML. `use()` of a
  package with `root = "."` no longer clones dependencies into the caller's
  directory or compiles against `/usr/include`. The registry build cache hashes
  include directories recursively. The per-file IR cache's compile fingerprint
  hashes those headers too, so a nested header edit is not linked from the
  previous IR. The wrapper file is loaded under `get_module_name`
  (`vec2d` → `Vec2d.jl`), not `titlecase`.
- **SysConfigGen proposals.** cmake command quotes are stripped
  (`-DMSDFGEN_PUBLIC=""` → `-DMSDFGEN_PUBLIC=`), every emitted string goes
  through `TOML.print`, and non-`-D` flags (`-std=`, `-fvisibility=`) are kept.
- **A failed AOT build deletes the previous `_thunks.so`.** Wrap, finding it
  gone, emits JIT dispatch from this build's metadata instead of binding the
  stale thunks by name.
- **Static members have no `this`.** A definition DIE with a body and no
  `DW_AT_object_pointer` records `has_object_pointer = false`. Both receiver
  gates honour that. A declaration, which never carries the attribute, does
  not record it, so out-of-line instance methods keep their receiver.
- **Global constructors are emitted.** `T::T` was skipped because the bare
  name equals the class; `ns::T::T` was not. The name is `T_T`.
- **Anonymous C++ unions are immutable byte regions**, so a parent struct
  inlines them instead of storing a pointer.
- **A float-array member records its size.** `float v[3]` had member size 0,
  so the return path treated the struct as unsizable and emitted an integer
  byte blob. The callee returns that struct in XMM; the blob was read from
  general-purpose registers.
- **SysV register exhaustion.** A register-class struct that does not fit in
  the integer and SSE registers still free is passed in memory, as one
  operand. It used to be split across the leftover registers and the stack.
- `long double` is still the x87 return the ccall ABI does not model. A
  parameter is refused by the `Any` guard; a return is still the wrong bits.
  A 0-byte by-value stub (constructor homing) is unchanged: the package-side
  fix is `-fstandalone-debug`, which is not the default because a merged
  module of that size exhausts clang's source-location counter. An
  attribute-packed C++ struct is still emitted at natural alignment. A byte
  blob of the C size does not match the thunk, which returns the aligned
  layout, so field reads move to the wrong offset.

### Constructors and destructors read their signatures from DWARF; `T&&` is a reference (2026-09-26)

- **Every constructor and destructor missed its DWARF.** clang writes the unified
  `C4`/`D4` spelling into `DW_AT_linkage_name`, while the symbol table carries the
  Itanium variants `C1`/`C2`/`C3` and `D0`/`D1`/`D2`. The exact-name join in
  `extract_compilation_metadata` never matched. Every structor was
  `parameters_source: inferred` from the demangled string, where a by-value struct
  is only a name, so it was typed `Any`. msdfgen's
  `LinearSegment(Vector2, Vector2, EdgeColor)` stored `p0=(1,3)` and garbage.
  `_structor_dwarf_key` now rewrites the symbol's marker to C4/D4 and accepts it
  only when DWARF has exactly that key. Measured on 8 Hub C++ packages rebuilt from
  scratch copies: all 559 structor symbols now come from DWARF (they had 0), with
  real parameter names and `this`. Verifiers are unchanged (tinyxml2 11, pugixml
  13, box2d 15, clipper2 233, stl 35, imgui 202). msdfgen's two `@test_broken` pins
  for this and the next item turned into unexpected passes, and are now plain tests.
- The `int` return some template-class destructors were given, from
  `infer_return_type` splitting the class on a `::` inside `<>`, disappears with
  this: destructors take `void` from DWARF.
- **`T&&` was unmodelled.** `DW_TAG_rvalue_reference_type` had no `type_refs` entry,
  so every move constructor's parameter resolved to `"unknown"`. The string mappers
  (`dwarf_type_to_julia`, `cpp_to_julia_type`, `TypesCpp.infer_cpp_type`) stripped a
  single `&`, which turned `T&&` into `Ref{Ref{T}}`. It now resolves to `T&&` and
  maps to one `Ref{T}`; it is an address at the ABI level, exactly like `T&`.
- Tests: `test_dwarf_attribution.jl` "rvalue references are references" (verbatim
  readelf), `test_symbol_hygiene.jl` for the C4/D4 key rewrite (all six variants, a
  template class, and refusal of an ambiguous pair), and a `geom::Segment2D(Point2D,
  Point2D)` constructor in the `stress_test` fixture, checked by metadata source and
  by the values it stores. The fixture test goes red with the join neutered.
- Found while doing this, not changed: GeneratorCpp skips a constructor when its
  bare name equals its **qualified** class, which only happens at global scope. So
  `ns::T::T` is emitted and `T::T` is not (logged in CLAUDE.md open problems).

### A throwing C++ function can no longer take the ccall path by sharing a name (2026-09-26)

Tier 3 (`ccall`) has no landing pad. A C++ exception thrown through it runs
`std::terminate` and the process dies (`terminate called after throwing …`, exit 134).
So a C++ function may take Tier 3 only if no exception can leave it. That was decided
by `_scan_noexcept_functions`, a regex over the sources that returns **bare** names.
Every function with one of those names was marked noexcept, so:

- `B::get(int)`, which throws, rode along with `A::get() const noexcept`;
- every constructor of a class with a `noexcept` move constructor was "noexcept",
  including the ones that validate and throw;
- `noexcept(false)` matched the regex too.

**Hub fmt shipped eight of these**: `fmt::file::file(path, flags)`,
`fmt::buffered_file::buffered_file(path, mode)`, `fmt::file::dup2(int)` (its
`dup2(int, std::error_code&) noexcept` overload supplied the name),
`fmt::ostream::ostream`, and their ABI-tagged twins. Re-wrapped from the Hub's current
metadata, `dup2(-1)` aborted with `std::system_error`. After the fix it raises
`CxxException` and the process lives.

- A name is now only a **candidate**. The proof is LLVM's `nounwind` on the function's
  own mangled definition in the IR the binary is linked from (`_nounwind_definitions`;
  clang stamps it on every `noexcept` definition). A definition that can unwind is
  never `nounwind`, so every false positive above is gone.
- Deliberately **not** widened: `nounwind` alone would also move every function the
  optimizer proved non-throwing to Tier 3. Tried on the Hub, that surfaced Tier-3
  emitter gaps the thunks tolerate: box2d `b2Contact::AddType`'s nested enum `Type`,
  and fmt `buffer<char>`'s constructor with an inferred `Any` parameter. Both were
  refused loudly by the wrapper guards, but that is a separate change. With the
  intersection, the routing can only lose Tier-3 functions, never gain them.
- The DWARF pass no longer sets `is_noexcept` whenever a function-level line contains
  the substring "noexcept", and the demangled-name substring check is gone. With no IR
  (ingest), nothing is noexcept, so all C++ goes to Tier 2.
- Measured on 8 Hub C++ packages rebuilt from scratch copies: only fmt's routing
  changed, where exactly those 8 functions left Tier 3 and none entered it. Verifiers
  are unchanged: tinyxml2 11, pugixml 13, box2d 15, clipper2 233, stl 35, imgui 202,
  msdfgen 21 plus 3 pinned broken.
- Tests: `test_noexcept_routing.jl` (CI; the rule tests go red with the name-only rule
  swapped back in by exact signature). The `callback_test` fixture gains
  `collide::{Quiet,Loud}::value` and `collide::checked() noexcept(false)`, asserted by
  tier and by a real throw in a subprocess, because a regression is an abort.

### `restrict`, `volatile` and `_Atomic` resolve; `Any` can no longer reach a ccall argument (2026-09-26)

The DWARF parser had no entry for `DW_TAG_restrict_type`, `DW_TAG_volatile_type` or
`DW_TAG_atomic_type`, so `resolve_type` answered `"unknown"` for every type behind
one. `restrict` sits on the pointer itself, so `int *restrict a` came out as a
parameter of type `Any` — and `Any` in a ccall argument tuple passes the Julia
object (a `jl_value_t*`), not the pointer. The call completed and returned a
plausible, wrong answer. `sum(a, 4)` over `[1,2,3,4]` returned a large garbage
integer; a `restrict` NULL check saw a non-NULL box.

- **Hub cglm shipped 58 of these** (`glmc_mat4_make(const float *__restrict src, …)`
  and every other `__restrict` parameter). On its Tier-1 path they raised
  `Malformed llvmcall`. On the ccall splice a precompiled consumer gets, they wrote
  garbage into `dest`.
- The three qualifiers now resolve straight through to their target type. They
  carry no ABI information, and adding a prefix to the type string would give
  every downstream mapper a spelling only some of them strip. A qualifier with no
  target qualifies `void` (`volatile void *` → `Ptr{Cvoid}`).
- **Measured across the Hub** by running the old and new parser on the same readelf
  dumps (33 packages; llamacpp and onednn not run): 860 function signatures and
  11 structs changed, and every change is `unknown` becoming a real type.
  Everything else was identical. The 11 structs are those with a `volatile` or
  `_Atomic` member (lmdb, lua, sqlite). Six now get typed fields instead of a byte
  blob, and all 11 still equal their DWARF `byte_size`. A/B rebuilds of lmdb, lua,
  sqlite, tinyxml2, pugixml and cglm pass their own verifiers unchanged
  (cglm test_deep 753/753).
- **Guard: `_assert_no_any_ccall_argument`**, the argument-side twin of
  `_assert_no_any_ccall_return`. The wrapper refuses to write if any foreign call
  declares an argument `Any` (`ccall` tuple, `@ccall` fixed arguments, Tier-1
  `llvmcall` `Tuple{…}`). It walks the parsed AST, because the return type in
  front of the tuple can contain commas. Variadic `@ccall` arguments are exempt:
  they are the user's own `[wrap.varargs]` declaration, and Hub lua passes a Julia
  object to `%p` on purpose. The guard also makes `long double` parameters loud,
  since they still map to `Any` (open, see CLAUDE.md). A library with one fails to
  wrap until the symbol is excluded, instead of shipping a silently wrong call.
  Before this change, a comment in `test_wrapper_type_bindings.jl` called
  argument-position `Any` "legitimate". That belief is how the cglm functions
  shipped, and the comment is corrected.
- Tests: `test_dwarf_attribution.jl` "restrict / volatile / _Atomic resolve through"
  (verbatim clang 22 readelf; it goes red with the pre-fix parser
  `include_string`'d back in), and `test_wrapper_type_bindings.jl` "Foreign call
  must not take Any". With the pre-fix parser, the guard alone refuses the audit
  fixture and names both `restrict` functions.

### Wrap-surface fixture reads the PE export directory (2026-09-24)

`test/test_wrap_surface_guard.jl`'s oracle ran `nm -D --defined-only`. That is
ELF's dynamic symbol table, and on a PE image GNU `nm` exits 1 with "File
format has no dynamic symbol table" — the same fact the guard itself already
handles by reading the export directory. The fixture check threw before any
assertion, which aborted `devtests.jl` after a green prefix. `dynsym` now
asks `_pe_exported_names` on Windows and `nm -D` elsewhere, so the oracle and
the guard read one table.

### Local entities and `decltype` returns get their own name and class (2026-09-24)

The build's `_qualified_name_parts` cut a demangled name at its first `(` outside
`<>`. Two GNU demangler shapes put a paren group in front of the function's own
parameter list, and both lost the real name. This closes the "still open" item
from v4.1.0 below.

- **Lambdas took the enclosing function's name.** In
  `fmt::v12::detail::write_fixed<…>(…)::{lambda(…)#3}::operator()(…) const` the
  cut fell on `write_fixed`'s parameter list. The call operator got name
  `write_fixed<…>` and class `fmt::v12::detail`, so lambdas #1 and #3 shared one
  Julia name and dispatch signature. Wrap reported 12 distinct entry points
  collapsed, and 10 of them were these lambdas. The survivors became extra
  methods on the real `write_fixed`, `write_int` and `do_write_float` functions.
  Class is now the demangler's prefix, enclosing signature included
  (`…write_fixed<…>(…)::{lambda(…)#3}`). The name is `operator`, which is the
  spelling every `operator()` already has.
- **imgui's lambda invokers gained a phantom `this`.** imgui_demo converts two
  lambdas to function pointers. Their static `__invoke` got class
  `ExampleDualListBox` and `ExampleAssetsBrowser`, which are real structs, so
  both receiver gates added a `this`. The shipped wrapper passes three arguments
  to `__invoke(self, idx)`, each one slot off. These are demo classes, not the
  ImGui API. The class is now the closure type, which is not a struct, and both
  gates say no. The imgui wrapper changes on its next rebuild.
- **A `decltype (…)` return type emptied the name.** `decltype ({parm#1}(0))
  fmt::v12::loc_value::visit<…>(…)` was cut at the decltype's `(`, which left
  `"decltype "`, and the return-type strip removed the rest. 8 fmt `visit`
  methods had name `""`, class `""` and `is_method` false. The wrapper named
  them by mangled symbol (v4.1.0). They are now
  `fmt_v12_loc_value_visit_fmt_v12_detail_loc_writer_char` and its siblings.
- **The fix.** `_param_list_paren` starts from the old answer. It skips a group
  that belongs to a `decltype`, or that is followed by cv/ref qualifiers and
  `::` (a local entity's enclosing function, or `(anonymous namespace)`). It
  ignores the `(` inside `{lambda(…)#N}`. Every other shape gets the old answer
  back. The return type is stripped from the outermost name only, because
  `A::f() const::Local::g()` has a depth-0 space that is not one. `_name_prefix`
  uses it too, so `is_method` agrees with the class. `parse_parameters`, the
  fallback when DWARF has no parameters, still takes the first group. DWARF
  supplies the parameters for all 47 functions below.
- **Only these functions moved.** The old and new derivations were run over
  every function in all 33 Hub packages' metadata. 47 changed (fmt 45, imgui
  2), all of them one of these two shapes. Both receiver gates agree on all
  3,087 method rows the new derivation produces. fmt was rebuilt twice from
  identical scratch copies, once with HEAD's derivation and once with the fix.
  The metadata differs only in `name`, `class` and `is_method` of those 45
  functions. Unreachable entry points went from 12 to 2. The wrapper gains 45
  names and loses 12: the 8 mangled fallbacks, plus 4 names that only a
  misnamed lambda carried (`write_ptr<…>` and `for_each_codepoint<…>`, whose
  real functions are not exported). Both wrappers load. The 2 collisions left
  are `detail::buffer<char>::data()` and its `const` overload, which differ only
  in the constness of `this`.

`test/test_local_entity_names.jl`.

## v4.1.0 (2026-09-24)

**Bounding the wrap surface.** A wrapper used to be whatever `nm` and DWARF
reported. For libraries that compile vendored internals into the same `.so`, that
made wrappers that did not load, or that exported symbols the loader could not
resolve. This release adds guards that refuse those builds, and knobs that narrow
the surface on purpose.

API changes. `RepliBuild.DAGDiff` is removed; it was exported. `[wrap] dag` is
now ignored with a warning. New keys, all opt-in: `[compile] visibility`,
`[wrap] exclude_symbols`, `[wrap] surface_types_only` and
`surface_types_extra`. `[link] link_dirs` now emits an rpath as well as `-L`.

### Function names are one derivation, and total (2026-09-24)

fmt 12.1 instantiates `detail::write_padded` on a lambda's closure type. The
GNU demangler spells that type `…::{lambda(fmt::v12::basic_appender<char>)#1}&>`.
The function-name replace-list had no catch-all, so `{`, `}` and `#` reached 58
definitions, and `_assert_wrapper_parses` refused the module. The Hub's fmt had
no `Fmt.jl` at all, with or without DAGDiff. This is the function-name spelling
of a failure class already fixed for types: DWARF's `(lambda at file.cpp:L:C)`
type spelling broke the llama.cpp wrapper in the same way (2026-08).

- **One derivation.** The list existed as four inline copies (C and C++, main
  and varargs), plus a shorter fifth one in GeneratorCpp for the names that
  proxies and deleters reference. The copies had drifted: C had no `@` or `~`,
  and the varargs paths had no `+ = - * /`. All five now call
  `_julia_function_name` (`Wrapper/Utils.jl`). It keeps the old list, then turns
  any character `Base.is_id_char` rejects into `_`. The lambda number survives,
  so distinct lambdas keep distinct names. `%`, `^` and `|` become `mod`, `xor`
  and `or`, because `_` would collapse all three onto `operator()`'s name.
- **Empty names were silent.** 8 fmt functions return `decltype ({parm#1}(0))`
  and arrive from the build with the name `""`. They were emitted as
  `function (this::Any, vis::Any)`. That is an anonymous function: valid syntax,
  so nothing refused it, and it binds nothing. They now take the mangled
  symbol, which is the name demangle failures already get. All-underscore names
  become `c__`. The 29 reserved words (`end`, `local`, …) get a `_` suffix. The
  parser is the reference here, not `Base.isidentifier`, which returns `true`
  for "end" on 1.13. Contextual words (`type`, `in`, `where`) are legal function
  names and stay as they are.
- **No shipped name changed.** The old and new derivations were run over every
  function in the metadata of all 33 Hub packages (11,670 functions, 74 of them
  varargs). Every name that was a valid identifier comes out byte-identical. No
  new name is invalid, and no names merged. Re-wrapped pugixml, box2d and lua
  are byte-identical apart from the timestamp and the baked path. fmt now wraps
  (12,202 lines) and loads. Its 362 "unbound export" drops are gone: they were
  an artifact of the body not parsing.
- **Still open, and not a sanitizer issue.** The build's `_qualified_name_parts`
  cuts the `…::{lambda(…)#N}::operator()` tail off a lambda's call operator. As
  a result, 10 fmt `write_fixed` lambdas share one name, and
  `_dedup_method_chunks` reports them as unreachable. The same function also
  produces the empty names. Verifying a fix there needs a rebuild, not a
  re-wrap.

`test/test_function_name_derivation.jl`.

### DAGDiff removed (2026-09-23)

`src/IRGen/DAGDiff.jl` is gone. Wrap consulted it in one decision, OR'd with
`is_ccall_safe`, and Build never called it. A static audit on 2026-09-23 over
all 10 C++ Hub packages (4,733 functions) found it flagging 28 functions
(fmt 24, pugixml 3, llamacpp 1). All 28 are already sent to Tier 2 by
`is_ccall_safe`, so 0 are flagged only by DAGDiff. That matches the 2026-08-08
audit (3 → 0). `is_ccall_safe` already sends every non-noexcept mangled C++
function to Tier 2. The Julia-side layout models natural alignment without the
generator's explicit `_pad_N`/`_pad_tail` fields, so ~93% of its ~24.6k
reported mismatches are gaps the emitted struct already corrects: the diff
compares DWARF against a layout the generator never emits.
`DAGDiffResult.lowering_order` is computed and read by nothing. The
`<project>/dag/` HTML/DOT renderer went with the module. `[wrap] dag` now warns
and is ignored.

### Linking a module too large for one clang process (2026-09-21)

`create_library` hands clang the linked module as one translation unit. Clang
gives every debug location a `SourceLocation` in a 32-bit offset space, and
Hub onednn's merged `-g -fstandalone-debug` module (3.9 GB of linked IR, 21.4M
metadata records) runs out of them: `translation unit is too large for Clang
to process: ran out of source locations`. This is one process's budget, not a
parallelism limit. Passing clang the IR vector instead is no fix, because that
is still a single `SourceManager`.

On that diagnostic, each per-TU IR file is now compiled to its own object in
parallel, under `build/obj/`. Objects newer than their input are reused. The
objects are then linked with the same flags. The trigger is the diagnostic,
not a size threshold, because the budget is spent by debug locations, not
bytes. Debug info is kept: stripping it would produce a `.so` the wrapper
cannot read. `compile_project` passes the unmerged list as `per_tu_ir`. A
direct `create_library` call without that list errors, since a merged module
cannot be split after the fact. No CI test covers this path; onednn's build is
the integration test.

### Enumerator named `_` no longer empties the module (2026-09-20)

oneDNN spells enumerators literally `_`. `_sanitize_cpp_type_name` collapsed and
trimmed that to `""`, and `_assert_wrapper_parses` refused a 271k-line wrapper
over two enumerators. All-underscore names now sanitize to `c__` in **both**
sanitizers. Plain `_` would parse, but Julia treats it as write-only, so the
member could never be referenced. The C sanitizer previously renamed such a
member to `_UnknownType`. C++ keeps `""` as the empty-name sentinel its callers
rely on. `test/test_enum_underscore_name.jl`.

### Identical enums alias instead of rebinding members (2026-09-20)

DWARF records `std::map` member typedefs (`key_type`, `value_type`) as separate
enums carrying the underlying enum's enumerators. The second `@enum` silently
rebound every member, so every ccall taking the real enum raised
`MethodError: Cannot convert … key_type` — while onednn's deep test stayed
green. An enum whose member set is identical to one already emitted is now
emitted as `const Alias = Canonical`, in both generators.
`test/test_enum_alias_collision.jl`.

### `[wrap] surface_types_only`, and unresolvable globals dropped (2026-09-20)

`surface_types_only = true` emits only the types reachable from exported
functions, method receivers (`class`) and globals present in the dynamic symbol
table. On onednn that is 79 of 2974 types, and it is what makes the module load:
internal nGEN types shadowed `Base.Integer` and emitted duplicate fields. The
option is **opt-in**. A reachability walk cannot see types used only through
function-pointer members (md4c) or constant-only enums (argon2), so
`surface_types_extra` (anchored globs) retains them. A pattern that matches
nothing is a hard error.

Independently and unconditionally, `_drop_unresolvable_globals!` now removes
globals absent from the dynamic symbol table before either generator runs. An
accessor for such a global could only fail at the call site. That was 656 of
onednn's 1006 functions, and every one of llamacpp's 4099 globals.
`test/test_surface_types.jl`.

### `[link] link_dirs` emits an rpath (2026-09-20)

`-L` answers where a library is at link time, not at load time. Linking against
another Hub package's `.so` recorded a bare `DT_NEEDED` that resolved to the
distro copy at runtime, so the process held two copies of process-global state.
Each `link_dirs` entry now also emits `-Wl,-rpath`, with `$ORIGIN` first. This
is skipped on Windows, where PE has no RUNPATH.

### `[wrap] exclude_symbols` (2026-09-20)

Anchored globs (`*`, `?`; every other character is literal, so `operator+`
means itself) matched against a function's name, demangled name and mangled
name, and against globals. They apply at wrap time on both wrap paths, so
changing the filter re-wraps without recompiling, and `compilation_metadata.json`
keeps the full picture. A pattern that matches nothing is a hard error (a stale
filter), as is filtering away every function. `test/test_exclude_symbols.jl`.

### `[compile] visibility` and `VisibilityProbe` (2026-09-15)

`-fvisibility=hidden` narrows the API to what a library annotates. For a library
that does not annotate its exports, or whose export macro is conditional and
inactive, it erases the whole API. `RepliBuild.VisibilityProbe.visibility_probe(toml)`
compiles the package's sources with the flag and reads visibility off the IR, so
you can measure before choosing. RepliBuild's own macro shims are excluded from
the verdict; counting them read lua as annotated. The verdict is per
configuration: cJSON annotates only with `CJSON_API_VISIBILITY` defined.
`[compile] visibility = "hidden"` is the declared form. It goes through
`get_compile_flags`, so it invalidates the IR cache. It also honours the legacy
raw flag, refuses a contradiction between the two, and survives
`discover(force=true)`. `test/test_visibility_probe.jl`,
`test/test_compile_visibility.jl`.

### Wrappable functions must be reachable at runtime (2026-09-15)

Metadata is correlated against `nm -g`, while the wrapper resolves through the
dynamic table (`nm -D` / PE exports). `verify_wrap_surface` runs before the
project hash is saved. It refuses an empty surface (R1) and any wrappable
function missing from the dynamic table under its mangled name (R2). Across 31
Hub packages R2 finds no offenders, which is what justifies a hard error. R1
originally applied only when static promotion had renamed symbols. That missed
the more common path — `-fvisibility=hidden` with promotion off, which is how
llamacpp shipped a zero-function wrapper — and it now fires on any empty
surface. `test/test_wrap_surface_guard.jl`.

## v4.0.1 (2026-09-13)

**Julia 1.13 / libLLVM 20 stability.** 4.0.0 shipped against Julia 1.12 and
libLLVM 18; 1.13 moved the bundled LLVM to 20, which broke the Slicer's
LLVMExtra calls, exposed a toolchain-discovery ladder that version-checked
nothing, and turned the C external hatch into a cross-major parse failure.
`Pkg.test()` also did not run at all. No API changes.

### C-bucket external hatch must not use a newer PATH opt (2026-09-13)

`[link] fallback = true` for C used to take PATH's `llvm-link`/`opt` when
`/usr/lib/llvm20/bin` was absent. On Windows that is MSYS2 LLVM **22** against
Julia 1.13's JLL clang **20**. `opt` wrote IR that `create_library` then
handed to clang 20, which died on `nocreateundeforpoison` (`unterminated
attribute group`) — a parse error, not a version error. Default C
(`fallback = false`, in-process libLLVM 20) was already fine.

`c_toolchain_bin_dir` now accepts a PATH `llvm-link` only if its major equals
`Base.libllvm_version.major`, and looks for `.exe`. A C fallback with no
matched bin is a hard error naming the hatch, not a clang parse. The
devtests hatch case `@test_skip`s when the bin is absent, so the rest of
the suite still runs.

### Unwind tables stop being a side effect of the landing pad (2026-09-13)

`setUwtableAsync` was called only inside the three `may_throw` lowerings, so a
thunk got an unwind table only if it also got a landing pad. Those are separate
questions: the table is what lets a throw *leave* the frame, which is prior to
who catches it. Windows proves the split — it lowers every `may_throw` to a
plain call and catches in libJLCS's `jlcs_guard_*`, reaching that frame only
because `.pdata` exists — yet on Windows `hasInvoke` is permanently false, so
the call never fired there at all.

The post-conversion walk in `JLCSPasses.cpp` and the matching one in
`JLCSCAPIWrappers.cpp` (AOT) now stamp `uwtable` on every `llvm.func` with a
body (`!isExternal()`), independently of the personality attribute, which stays
gated on an actual invoke. Restores gdb `bt` through non-throwing thunks;
stepping was never affected (that is `.debug_line`).

**The landing pads are NOT removable on top of this, and that was measured.**
`jlcs_guard_*` is reached through a `Sys.iswindows()`-gated path in
`JITManager`; on Linux Julia calls the thunk directly, so no C++ frame exists
to catch in and the in-thunk landing pad is the only catcher. Deleting them
with uwtable in place aborts — `terminate called after throwing an instance of
'std::runtime_error'`, SIGABRT. Un-gating the guard call path is the
prerequisite for that cleanup, and it puts a C++ frame in every Tier-2 call.

### Windows: three new CI tests made runnable on PE (2026-09-12)

`Pkg.test()` died on Windows in files that landed with Unix fixtures while the
same suite was green on Linux — three independent instances of "the assertion
is true of the Unix spelling of the fact". The fake `llvm-config` was a shebang
script, so `_llvm_major_of` always scored 0 and newest-wins never saw a usable
prefix (Windows now emits `.cmd` shims; Unix keeps the shebang). Toolchain
advice shelled to `sed`/`head` and compared Arch/Fedora/Debian lines against
`_LLVM_ADVICE`, which on Windows is the MSYS2 pacman stanza (the doctor now
names Unix and Windows copies separately). Project-hash scratch TOMLs wrote
`C:\Users\...` into a quoted string, and TOML reads `\U` as a unicode escape,
so every testset errored before hashing anything. `Pkg.test()` exit 0 on
Julia 1.13.0, Windows.

### Slicer: use LLVM.jl setters that exist on libLLVM 20 (2026-09-12)

Julia 1.13 ships libLLVM 20, where LLVMExtra's `LLVMSetInitializer2` and
`LLVMSetPersonalityFn2` were folded into the upstream C API. LLVM.jl 9.13 puts
those Extra names on `LLVM.API` only for older libLLVM, so on 20 they are
simply missing — zlib wrap hit it on the first reached global with
`UndefVarError: LLVMSetInitializer2`. Now calls `LLVM.initializer!` /
`LLVM.personality!`, which already pick the right symbol per version. Verified
on Julia 1.13.0 with Hub `test_deep.jl` 271/271.

### `Pkg.test()` actually runs now (2026-09-10)

The suite was green under `julia --project=. test/runtests.jl` and **failed
under `Pkg.test()`** — which is how CI invokes it. `Pkg.test()` builds an
environment from `[deps]` plus the `[targets]` test extras, and
`test_struct_layout.jl` imports `Logging`, which was in neither. It ran 21
testsets and died with "Package Logging not found in current path". Nothing in
the tree said which invocation was authoritative.

`Logging` added to `[extras]` and `[targets].test`. Guarded in `runtests.jl`:
every `using`/`import` in `test/*.jl` must name something in `[deps]`, the test
target, or the package itself, and every `[targets].test` entry must exist in
`[extras]`. It checks the *declaration* rather than shelling to `Pkg.test`,
which keeps it cheap and names both the package and the importing file. Two
guards against asserting into an empty loop: the scan must find `Test` and at
least five distinct imports.

### Project hash walks include dirs recursively (2026-09-10)

The include-dir walk was `readdir`, top level only, so a header one directory
down could be edited freely while the build reported "project unchanged"
against modified sources. `include/<lib>/<lib>.h` is the ordinary C layout —
SUNDIALS generates exactly `include/sundials/sundials_config.h` — so the
headers most likely to matter were the ones least likely to be seen. Now
`walkdir`, with the path hashed alongside the bytes so a rename (or two headers
swapping contents) is not invisible, and sorted descent so the key cannot
depend on enumeration order.

Recorded because the originally *reported* bug was not the one that existed: a
source edit does invalidate the project cache, verified twice. The earlier
symptom was `build()` having saved its hash inside the same process whose
`wrap()` then threw on the AOT assert — so the next run's "project unchanged"
was correct, and the `rm .replibuild_cache/project_hash` in that session was
cargo-cult. Seeding the thunk manifest was what actually unblocked it.

### LLVM prefix search: gate on version, prefer newest (2026-09-10)

The C prefix ladder listed `/usr/lib/llvm-20` down to `-15` — every entry
*below* the 21 minimum, with 21+ absent entirely — so a Debian box with LLVM 22
at `/usr/lib/llvm-22` had its only usable install walked straight past.
Auditing that turned up a worse defect underneath: **the loop version-checked
nothing.** It returned the first prefix carrying `clang++` and `llvm-config`,
so where the distro default `/usr` LLVM is too old, that was the answer, and
the failure surfaced later inside CMake naming a missing header instead of here
naming a version.

The list is now generated from `MIN_LLVM_VERSION..MAX_PROBED_LLVM_VERSION`, and
selection picks the newest candidate that *meets* the minimum. A prefix whose
`llvm-config` answers something unparseable scores 0 and is skipped rather than
accepted. `MIN_LLVM_VERSION` moved to `LLVMEnvironment` (loads first) with
`EnvironmentDoctor` importing it instead of declaring its own `= 21` — three
copies of that fact is how the ladder drifted below a minimum it did not share.

### Toolchain advice: one floor, one install hint (2026-09-10)

Removes the last copy of the LLVM floor, `LLVM_MIN_MAJOR=21` in
`src/mlir/build.sh`, which now **derives** it by reading `const
MIN_LLVM_VERSION` out of `LLVMEnvironment.jl`. No fallback literal when that
read fails — a default is exactly how the second copy comes back — so it errors
naming the file instead. Also removes a reassignment further down that was
silently clobbering the derived value with 21 again.

`build.sh` carried four install hints that disagreed: two spellings of the Arch
line, three of the Ubuntu line, one with `mlir-21-dev` hardcoded next to a
`${LLVM_MIN_MAJOR}` interpolation in the same file. Collapsed to one
`install_hint()` used by all three failure paths, matching the doctor's
`_LLVM_ADVICE`. `test_toolchain_advice.jl` (20 asserts, wired) holds the
invariant: `build.sh` must derive rather than restate the floor, pin no package
version, define exactly one hint with no package-manager line outside it, and
its Arch/Fedora package sets must equal the doctor's.

### Quadratic DWARF parse, a `wrap()` network reach, and the drop report (2026-09-09)

`parse_dwarf_dump` answered two per-struct questions ("which DIEs name me as
parent?") by rescanning all of `type_refs` each time — aggregates × DIEs × 2,
quadratic in the binary. Hub curl: 196,841 DIEs, 6,594 aggregates, 415s, and
nearly all of it two scans that found **nothing**, since curl is C and has zero
`DW_TAG_inheritance` and zero template params. Every C library paid the full
bill to build two empty arrays. One O(N) parent index up front:

```
liblua      203K lines     7.11s ->  1.35s
libcurl     873K lines   414.8s  ->  3.43s
llamacpp   5.02M lines              17.6s   (264MB dump, 71697 fns)
```

Byte-identical output, verified by SHA over sorted `struct_defs` on lua, curl,
and on tinyxml2/pugixml which carry real inheritance (8 and 10 structs with
`base_classes`).

### AOT bootstrap deadlock, and a doctor that certified untested tiers (2026-09-10)

A fresh C++ package with `aot_thunks = true` could not converge. AOT reads
`thunk_manifest.json` at BUILD to decide what to emit; only `wrap` can compute
that set — and wrap's `_assert_aot_thunks_present` threw **before** the manifest
write. So AOT emitted nothing, wrap demanded N, wrap died, the manifest stayed
stale, and the next build read the same stale file. The error's own advice
("rebuild so AOT sees the same symbols") was unreachable, because rebuilding
re-read the file wrap had refused to update.

Recording what wrap needs is not the same act as certifying the library
provides it. `_write_thunk_manifest` now runs first and returns a note the
refusal quotes, so the cycle converges in two passes with no hand-editing:
pass 1 refuses and updates 0 → 5, pass 2 builds clean. Verified on Hub
`hello_world` from a hand-reset empty manifest. One writer now — Generator.jl's
duplicate write is gone, since a write on the far side of the refusal is
exactly the one that never happened.

### First Windows run of the 2026-09-06 Linux work — and the port leftovers it surfaced (2026-09-07)

`julia 1.12.7`, MSYS2 CLANG64, system LLVM/MLIR **22.1.8**, `libJLCS.dll` present.
The STL half of that work was already correct here; the SysConfigGen half was
not, and nothing about it raised.

**SysConfigGen compared '/' paths against '\' ones and said no to every one.**
`capture_config`'s new classification and layout are the first code in that
module to *depend* on the path spelling rather than merely tolerate it, and its
two sources disagree on Windows: `compile_commands.json`, which cmake always
writes '/'-separated, and `walkdir`/`relpath`/`abspath`, which answer in the
host separator. Every comparison silently returned false.

- **The real generated TU was classified as try_compile scratch and dropped.**
  `compiled_gen` is keyed on the build-relative path off the compile line
  ('/'), `gen_sources` on the walk ('\'), so the set difference put *every*
  generated source in `scratch_sources`. On the cmake fixture `gen_table.c` —
  which the target compiles — was skipped by the capture and left out of
  `[compile] source_files`. The library would have linked without it.
- **`_is_cmake_internal` matched nothing**, so cmake's own
  `CMakeFiles/<ver>/CompilerIdC/CMakeCCompilerId.c` reached the classifier.
- **`_build_include_roots` and `_translate_includes` returned empty.**
  `abspath` spells the separator natively on one side of a `startswith` whose
  other side is '/'-joined. `include_roots` came out `[""]` instead of
  `["include", ""]`, so `:auto` collapsed to the build-tree layout and a header
  included as `<fixt/fixt_config.h>` landed where `-Iconfig` cannot resolve it
  — precisely the defect the layout work exists to fix.
- **`_rel_source` returned absolute paths, and that HUNG the process.**
  `_collapse_excludes` walks `dirname` until it is `""`, but `dirname` is
  idempotent at a filesystem root — `"C:\\"`, and `"/"` on POSIX — so the loop
  only ever terminated for a relative path. The suite spun a core with no
  output and no error. It now stops at the fixed point whatever it is handed,
  which closes the latent POSIX case with it.

Fixed at ONE seam rather than a branch per site: `_posix` normalizes wherever a
host path enters the module, and is `Sys.iswindows()`-keyed for the same reason
`Compiler._canon_path` is — a backslash is a legal POSIX filename character.
`_rel_source` and `_translate_includes` join through `_join_rel` rather than
`joinpath`, since their results are matched against `config_rel * "/"`
downstream — and `_join_rel` keeps `joinpath`'s empty-prefix behaviour
deliberately: `cmake_probe` defaults `clone_rel` to `""`, where a bare
`a * "/" * b` yields an ABSOLUTE path. Windows
now returns the Linux answers exactly on the cmake fixture: `generated_sources`
`["gen_table.c"]`, `scratch_sources` `["PROBE_TEST/ltest.c"]`, `include_roots`
`["include", ""]`, target files `["a.c", "b.c", "config/gen_table.c"]`.
`test_sysconfiggen.jl` **61/61**, the same count as Linux.

**The STL force-ctor fix is right on Windows; its assertions were libstdc++'s
spelling of it.** libc++ emits standalone `vector[abi:nqe220108]()` /
`map[abi:nqe220108]()` at `-O2` and `extract_stl_method_symbols` binds exactly
one constructor per container, the default one — the fix works as built. What
was red is the `-O2` emission probe: three `nm` regexes written without libc++'s
inline namespace (`std::__1::`) or its `_LIBCPP_HIDE_FROM_ABI` tag, and two
`full_signature` equalities against the untagged spelling. Restated as
invariants — optional inline namespace, optional `[abi:...]`, tag stripped
before comparing which ctor bound. Same lesson as `_canon_path`: a test that
states one host's answer instead of the invariant goes red on the other for a
correct build.

`_classify_stl_method`'s iterator guard had that shape in the CODE, not the
test: libc++ spells vector's iterator `__wrap_iter`, which `r"iterator"i` does
not match, so Windows bound `erase`/`insert` overloads Linux drops. Benign
today — only `CppMap.delete!` reads an `erase` thunk, and map's libc++ iterator
is `__map_iterator`, which the old guard does catch (verified against a real
`-O2` build, not assumed) — but the hosts now bind the same set instead of
resting on that. `test_stl_extract.jl` **56/56**.

**`[wrap.macros]` produced nothing on Windows, because `visibility("default")`
is not how PE exports.** Found by taking SysConfigGen past its own tests: a
pcre2 harvested and built entirely on Windows came out with **14 shims defined
by `nm -g` and 0 in the export directory**, which is the list the wrapper reads
on PE — so every `PCRE2_*` constant was missing from the module and 23 of the
package's 24 failures were `UndefVarError: PCRE2_CASELESS` and friends.
Reduced: `visibility("default")` and `__declspec(dllexport)` both DEFINE a
symbol in a PE, and only the second exports it. The shim TU now carries
`RB_SHIM_EXPORT`, keyed on `_WIN32` — the compiler's target decides, not the
host running RepliBuild — with the ELF spelling unchanged (it is load-bearing
there for `-fvisibility=hidden` projects like box2d3).

That fix alone would have been a worse bug than the one it closed. **A
`dllexport` anywhere turns mingw's auto-export off for the WHOLE image**, so
adding ours to a library that exports nothing explicitly ERASES its API:
measured on a two-function probe DLL, the export table went from
`{lib_a, lib_b, shim}` to `{shim}` — the wrapper would then emit the constants
and nothing else, with no error anywhere. `create_library` now asks
`_pe_export_intent` whether anything OTHER than a shim is `dllexport`ed,
reading it off the IR (LLVM prints the linkage beside the symbol name, so one
pass answers it, and it works on the LTO path where a filename test would not).
When ours are the only explicit exports it hands back `-Wl,--export-all-symbols`
— which is exactly what the auto-export default expands to, CRT still excluded
by ld's own list, so the link produces the set it always would have.

Measured both shapes on Windows:
- pcre2 (exports explicitly, via `PCRE2_EXP_DECL`) — export table **78 API +
  14 shims, zero CRT**; no `--export-all-symbols`, so its chosen surface is
  untouched. `packages/pcre2/test.jl` went **39 pass / 8 fail / 16 error →
  76 / 1 / 0**, the one remaining failure being the suite's own
  `endswith(lib, "libpcre2.so")`, i.e. Linux's file extension.
- a library exporting nothing of its own — all three symbols exported and all
  three wrapped, including the shim.

Guarded twice, both toolchain-free: the emission split in
`test_c_generator_policies.jl` (including a count, so one un-marked shim is a
failure rather than one silently missing constant), and `_pe_export_intent`'s
three decisions in `test_windows_port.jl`.

**Two port leftovers, both `/dev/null`-class, both older than this work:**

- **`test_win64_abi.jl` skipped on the only host that can execute the rules it
  pins.** Its oracle probe ran `clang --target=... -o /dev/null`, and `run` does
  not go through a shell, so clang got the literal path and resolved it as
  `\dev\null`. The probe concluded a CLANG64 clang "cannot target
  x86_64-w64-windows-gnu" — its own host — and the file `@test_skip`ped.
  Writing to a real temp file instead: **95/95 on Windows**, up from one skip.
  The table encoded on Linux as a specification is now checked against clang on
  a machine that can run it.
- **`test_jlcs_invariants.jl` §D ran `nm -D` on a PE**, which is a hard error
  ("File format has no dynamic symbol table"), not an empty answer — the same
  fact the port already recorded for the wrapper's three export sites, at a
  fourth site nobody converted. Nor is there anything for it to find: PE binds
  every symbol at link time, so a DLL still carrying an unresolved
  `mlir::jlcs::` reference cannot link. The RTLD_NOW probe above it is the whole
  check on Windows and is the stronger half. **16/16**, was 16 + 1 error.

### Three more tests that stated Linux's answer, and the objdump behind one of them (2026-09-07)

Each was red on Windows for a reason unrelated to what it tests, and each was
already red before the pull.

**`test_struct_abi` 28/2 → 30/30. Both failures were `long`.** §A's synthetic
metadata declared a member `c_type = "long"` and asserted
`!llvm.struct<packed (i64)>`; the generator emitted `packed (i32)`, which is
CORRECT — `C_LONG_MLIR` is `i32` under LLP64. §C's C fixture was
`typedef struct { long a, b, c; } B3;`, so 12 bytes on Windows, while the
hand-written MLIR beside it said `!llvm.struct<(i64, i64, i64)>` and the Julia
side read `NTuple{3,Int64}` — `b3_make(7, 8, 9)` came back
`(0x8_00000007, 0xFFFFFFFF_00000009, 4)`, three 32-bit values read as 64-bit.
The shape under test is MEMORY-class-by-sret, not the width of a `long`, so the
fixture and its metadata now say `long long` (8 bytes on both hosts) and every
trace describes one struct again. `test_llp64_widths.jl` owns the width
question and is 68/68 precisely because it asserts the invariant instead.

**`test_debug_inspection` 47/1 → 66/66, and 24 assertions that were never
reached now run.** Three defects stacked:

- The testset shelled to `` `timeout -s KILL 170 julia …` ``. On Windows that
  name resolves to System32's `timeout.exe`, which PAUSES rather than running a
  command (`ERROR: Invalid value for timeout (/T) specified`), so the child
  never ran and everything behind `if ok` was skipped. The kill is done from
  Julia now — `kill(proc, SIGKILL)` is TerminateProcess on Windows, so the
  "cannot be ignored" property the wrapper was chosen for still holds.
- With the child running, `Debug.disassemble` failed: it hardcoded `objdump`
  and passes `--disassemble=<symbol>`, GNU spelling that llvm-objdump rejects
  outright — and **in a CLANG64 shell the bare name IS llvm-objdump**, the trap
  `_dwarf_dumper` already exists for, at a site nobody had converted. New
  `Compiler._gnu_objdump()` screens with `_is_gnu_binutils` and knows MSYS2's
  GCC-flavoured trees; `Debug` resolves through it (memoised) rather than
  keeping a second answer. Kept separate from `_dwarf_dumper` on purpose: that
  one may legitimately return `readelf`, which cannot disassemble.
- Then `DW_AT_comp_dir`: llvm-dwarfdump prints it as a C string LITERAL, so on
  Windows every separator arrives doubled and a native-spelling `occursin`
  found nothing. Unescaped before matching, identity off Windows.

Guarded in `test_windows_port.jl`: `_gnu_objdump` must return an objdump that
PASSES `_is_gnu_binutils`, and `Debug._objdump()` must be the same answer.

**Still red on Windows, and NOT from this work — each confirmed identical at
`ddd2d7f` (the commit before the pull), fixtures and all:**

- `test_c_inprocess` **9/1** — and, unlike the rest of this entry, **NOT a
  Windows property**. The `[link] fallback = true` escape hatch runs the
  EXTERNAL optimizer, and system LLVM 22.1.8 writes debug RECORDS into textual
  IR (`#dbg_declare(ptr %2, !15, …)` — reproduced directly with `opt -O2 -S`);
  the `Clang_unified_jll` 18.1.7 that links the C bucket cannot parse that
  syntax and answers `expected instruction opcode`. That is version skew across
  the documented two-LLVM split, not an object format, so it should reproduce
  on Linux with the same pair — unverified there. The default in-process path
  is green.
- `test_struct_abi` and `test_debug_inspection` were on this list and are
  **fixed above** — both turned out to be tests stating Linux's answer, not
  library defects.
- `callback_test/test_exceptions.jl` aborts standalone with `0xC00000FF`, and
  `devtests.jl` still dies at `libunwind: pc not in table` after §4 with zero
  test failures. Same known SEH/COFF unwind-registration class already recorded
  under **Open on Windows**; running each remaining section in its own process
  is the way around it, and all of them were run that way for this entry.

### `capture_config` tells try_compile scratch from library code, and keeps a generated header's include form (2026-09-06)

Two defects in `SysConfigGen.capture_config`, both surfaced harvesting
SUNDIALS, both structural — reproduced on a hand-written cmake fixture with
the library out of the picture (`test/test_sysconfiggen.jl`
§"scratch classification + generated header layout", suite 61/61):

- **cmake `try_compile` scratch was captured as generated library source.**
  `check_c_source_compiles` hides its probe under `CMakeFiles/CMakeTmp/`,
  which `_is_cmake_internal` already drops, but a hand-rolled `try_compile`
  with its own bindir lands in plain sight — SUNDIALS' monotonic-timer probe
  leaves `POSIX_TIMER_TEST/ltest.c`, six lines with its own `main()`. Added to
  `[compile] source_files`, as the skill tells authors to do with generated
  `.c`, it plants a `main` symbol in the shared object. `CMakeProbe` now
  splits `generated_sources` from `scratch_sources` on one mechanical test —
  does any target in `compile_commands.json` compile the file — and
  `capture_config` skips the scratch, recording the skip in `SYSCONFIG.md`.
  With no `compile_commands.json` there is no evidence, so nothing is
  classified rather than guessed.

- **Captured paths were flattened to basenames, breaking any library that
  includes its config header under a subdirectory.** SUNDIALS generates
  `include/sundials/sundials_config.h` and includes it as
  `<sundials/sundials_config.h>` at 25 sites, so a flat `config/` with
  `-Iconfig` resolved nothing. Files are now laid out under the deepest
  build-tree `-I` root containing them, so the package reproduces the include
  form upstream compiles with. `flatten::Bool` is replaced by
  `layout::Symbol` (`:auto` default, `:flat`, `:build_tree`), and two files
  landing on one destination is now a hard error instead of a `@warn` plus a
  silent overwrite.

`_rel_source` shares the layout derivation, so `[compile] source_files` names
generated TUs at the path the capture actually wrote them to. Regression: the
pcre2 harvest restages **byte-identical** (its build tree generates everything
at the root, where `:auto` reduces exactly to basenames) and `packages/pcre2`
`test.jl` is **77/77 from clean**.

### STL default ctors survive `-O2`; `Class::method` split is paren-aware (2026-09-06)

Wrapping a templates-only package at `-O2` (no user TU that named `T t;`)
left `create_std_map_*` memset-zeroing a live `std::map`. Two defects,
both in `Compiler.jl`, both gated by `test/test_stl_extract.jl`:

- The generated force TU took `T&` and never default-constructed.
  `template class T;` does not emit the header-inline default ctor as a
  standalone symbol, and `noinline` on the force function is not enough —
  clang still inlines `vector()` *into* that caller at `-O2`. The force
  body now does `T c;`, and that generated TU alone compiles `-fno-inline`.
- `extract_stl_method_symbols` split demangled `Class::method(args)` on
  the last `::` at angle-bracket depth 0, so `std::allocator` in constructor
  arguments stole the split. The scan is paren-aware now. Only empty-paren
  `T()` classifies as `constructor` (the factory calls `T()` with no extra
  args). Iterator `erase` overloads, which the old split happened to drop,
  are skipped so `CppMap.delete!` keeps `erase(K const&)`.

Hub `packages/stl` rebuilt at `-O2` with its odr-use / `-O0` pin removed:
`test_deep.jl` **35/35**.

## v4.0.0 (2026-09-05)

**RepliBuild runs on Windows.** Target is `x86_64-w64-windows-gnu` — mingw under
MSYS2 CLANG64, a native host build, not a cross-compile from Linux and not MSVC.
All three tiers are live: the C bucket compiles and links, the JLCS dialect
builds and loads as `libJLCS.dll`, and MLIR AOT thunks work with C++ exceptions
crossing them intact.

Major for two reasons. The supported-platform set changed — the load-time gate
now admits two kernels instead of one, and `check_environment` answers for both.
And one generated-wrapper API changed shape: a C bitflag enum is emitted as an
integer alias rather than a closed `@enum`, so regenerating a wrapper changes how
its mask-shaped enums construct and print — 59 of the 544 `@enum`s across the
shipped Hub wrappers, concentrated in imgui.

Linux behaviour is unchanged. Every platform fix is either conditional on the
host or a correctness fix that was latent here — the SysV path is bit-identical,
and `runtests.jl` is green on the Linux reference host.

### The port found more than the punch list predicted

`WINDOWS_PORT.md` listed six blockers before the work started. Two of the six
were wrong, and most of the actual work was in things it did not name at all.
That document now records the outcome rather than the plan; the sections below
are the summary.

Wrong prediction #1: **there was no MLIR version skew.** The punch list expected
MSYS2 to ship MLIR 21.1.8 against the reference host's 22.1.8 and called that
"the first compile error in `src/mlir/build.sh`". MSYS2 CLANG64 ships 22.1.8.
The dialect compiled with two benign warnings and nothing else.

Wrong prediction #2: **the DWARF parser did not need replacing**, and routing PE
through `llvm-dwarfdump` — the prescription — would have been the harder path.
See below.

### DWARF on PE: swap the tool, keep the parser (2026-09-05)

`Compiler.parse_dwarf_dump` reads GNU readelf output and nothing else, and
readelf is an ELF-only reader. The obvious fix was to route PE through
`DWARFParser.jl`, which speaks llvm-dwarfdump — the portable dialect.

That was backwards. GNU `objdump` is BFD-based, reads `pei-x86-64`, and shares
binutils' `dwarf.c` printer with readelf: `objdump --dwarf=X` and
`readelf --debug-dump=X` are the same code emitting the same text. Verified
against a real PE DLL — **193/193 identical DIE lines** and an identical
`rawline` table. So the container was the blocker, not the debug info, and the
readelf-dialect parser is untouched. llvm-dwarfdump is a genuinely different
dialect; taking it would have required the parser rewrite this avoids.

`_dwarf_dumper()` picks readelf on Linux and objdump on Windows, and **requires
the candidate to identify as GNU binutils**. That is load-bearing rather than
defensive: in an MSYS2 CLANG64 shell, `objdump` on `PATH` *is* `llvm-objdump`,
whose output has no DIE level and no abbrev number. Accepting it would not fail
loudly — it would parse zero functions and ship a wrapper of inferred
signatures, which is the exact silent-guess failure the llvm-dwarfdump fallback
was deleted for. Identity is the string `GNU Binutils`, not a `GNU` substring:
llvm-objdump's own `--version` says "compatible with GNU objdump".

`ingest()` no longer refuses a `.dll` for its container either. Its DWARF
presence check ran `readelf -S` and reported "No DWARF debug info found —
rebuild upstream with `-g`", which on a mingw DLL is actively wrong: the DWARF
is there and readable, and rebuilding fixes nothing. The three outcomes —
binutils missing, container unreadable, ELF without `.debug_info` — are now
three separate errors.

### PE exports are the API (2026-09-05)

One root fact behind four defects: **mingw links its C runtime, startup code and
unwinder statically into every DLL**. So `nm -g --defined-only` reports
`snprintf`, `memcpy`, `atexit`, `fprintf` and `abort` exactly like the library's
own functions. On ELF that code lives in a shared libc and is only *referenced*,
so nm's answer and the library's API have always been the same list and nothing
ever had to tell them apart.

RepliBuild generated a wrapper for `snprintf` as an exported function of the
fixture library. All three symbol sites now read the **PE export directory**
(`objdump -p`), which is the strongest available authority: `GetProcAddress`
consults only that directory, so a symbol outside it cannot be reached through
the library even in principle.

Measured: c_test **115 nm symbols / 28 exports**, mi_test 133/43, vi_test
142/56, stl_test 691/513 — and zero CRT symbols in any export table.

The punch list named two `nm -D` sites and both were broken (`nm -D` on a PE is
not an empty answer but a hard error, *"File format has no dynamic symbol
table"*). The site that actually broke builds was a third one it did not list,
`extract_symbols_from_binary`. `nm` is kept where its `T`/`W`/`D` column is
needed — the export directory does not say whether a name is a function or a
global — and screened against the export set.

### ORC deadlocks on an unresolved symbol; it does not raise (2026-09-05)

The monolithic LTO path embeds the whole post-LTO module and `Base.llvmcall`s
it, so every symbol that module declares must resolve in the consumer's process
at first call. c_test's `mathkit.c` calls `snprintf` and `sqrt`; neither is
exported by anything in a Windows process. ORC printed
`JIT session error: Symbols not found: [ snprintf ]` and then **hung**.

That path had no pre-flight at all, though `_tier1_preflight!` has done exactly
this for per-function slices all along. `_lto_unresolved_symbols` now asks
before the wrapper is written and demotes the whole path to `ccall`, which is
always correct — only the zero-cost path is lost.

`_symbol_resolves_via` is what made the pre-flight reachable: it called
`ccall(:dlsym, …)`, and there is no `dlsym` symbol in a Windows process, so the
raw ccall raised "could not load symbol" instead of answering. `Libdl` is the
portable spelling, and the `RTLD_DEFAULT` arm walks the loaded-module list since
Windows has no null-handle lookup.

### libc++ is not libstdc++, three separate ways (2026-09-05)

CLANG64 ships libc++, and the STL extraction path assumed libstdc++ in three
places.

**ABI tags.** Nearly every inline member is `_LIBCPP_HIDE_FROM_ABI`, which
mangles a tag in *after* the name — `size[abi:nqe220108]() const`. Every test in
`_classify_stl_method` is a `startswith(sig, "size(")`, so every tagged member
was dropped: `std::vector<int>` came out of a real build with **one** method,
the destructor, which survived only because its test is on the leading `~`.

**A bare `startswith` on `basic_string`.** libc++'s nested helpers normalised to
the string itself, and dedup keeps the first match per method name — so
`~__annotation_guard()` was recorded as the string's destructor and the real
`~basic_string()` never got in. The symptom was an `EXCEPTION_ACCESS_VIOLATION`
inside `__is_long`. Now `$`-anchored, like the vector and map branches always
were.

**Explicit instantiation emits nothing.** `_LIBCPP_HIDE_FROM_ABI` also expands
to `exclude_from_explicit_instantiation`, so `template class std::map<int,int>;`
puts none of those members in the binary — they exist only where some TU
odr-uses them. There is no macro to switch it off; `__config` gates the
attribute on `__has_attribute`. The generated instantiation TU now carries a
never-called function per container whose only job is to name each member.
Every touch is guarded by `if constexpr` on whether the expression is valid at
all, so **one generic body covers vector, string, map, set, deque and list** —
`std::list` has no `operator[]`, `std::vector` no `key_type`, `std::map` no
`push_back`, and each drops out. The TU is syntax-checked before it is trusted
and falls back to the bare instantiations, so nothing that builds today can
regress.

stl_test went from 3/4 with access violations to **28/28**; extraction from 12
methods to 51.

### The C bucket had no libc (2026-09-05)

`.c` goes through `Clang_unified_jll`'s clang so its IR is version-locked to
Julia's resident libLLVM. That JLL already targets `x86_64-w64-windows-gnu` —
the triple was never the problem — but it is a bare compiler artifact with no
headers and no CRT, and its `InstalledDir` is a Julia artifact directory with
nothing to fall back to. Every C file died on its first system `#include`.

`_c_bucket_sysroot()` finds the MSYS2 CLANG64 prefix — `REPLIBUILD_C_SYSROOT`,
then the on-`PATH` clang's own prefix, then `MSYS2_ROOT` — accepting a candidate
only if it really holds `include/math.h`, since an MSYS2 install carries empty
`ucrt64/` and `mingw64/` trees that would otherwise be picked. A caller's own
`--sysroot` wins.

Linking needed a second fact: clang's mingw driver defaults to `-rtlib=libgcc`,
and CLANG64 is a compiler-rt/libc++/libunwind environment carrying no libgcc at
all. MSYS2's own clang is configured with `CLANG_DEFAULT_RTLIB=compiler-rt`; the
JLL is not and cannot be, so it is told here — asked of the sysroot rather than
the platform, since UCRT64 and MINGW64 are GCC environments where libgcc is
present and correct.

### MLIR AOT thunks: a load-order problem that read as a missing file (2026-09-05)

**PE binds every symbol at link time.** ELF lets a `.so` carry undefined symbols
and resolve them at `dlopen` through `RTLD_GLOBAL`; a thunks library on PE needs
the full link, naming both the main library and `libJLCS`.

And that link is what stopped AOT. Naming `libJLCS` turns it into a PE *import*,
resolved by the loader — which searches the loading module's directory and
`PATH`. `libJLCS.dll` lives in RepliBuild's own source tree, which is neither.
So every AOT thunks library failed to *open*, with `The specified module could
not be found`: a message naming the module that **was** found and not the
dependency that was not, so it reads as a missing thunks library while the
thunks library is sitting right there next to the wrapper. ELF never showed this
because `-Wl,-rpath` is in the link line — inside the `!Sys.iswindows()` branch,
because PE has no RUNPATH.

**The fix is ordering, not placement.** Vendoring a copy of `libJLCS.dll` beside
the thunks would satisfy the loader and silently break exceptions:
`jlcs_catch_current_exception` writes the message into libJLCS's *own*
`jlcs_exception_buffer`, and `_check_pending_exception` reads it back by
ccall'ing into `MLIRNative.libJLCS`. Two copies in one process is two buffers —
the thunk writes one, Julia reads the other, and every C++ exception crossing a
thunk disappears. `JITManager.open_thunks_library` opens the libJLCS that
`MLIRNative` names, by absolute path, *before* the thunks library, and both
generators route through it. The JIT path holds the same invariant by another
route, registering those symbols out of `MLIRNative.libJLCS` by hand.

Measured on hello_world: `libhello_world_thunks.dll` (16.5 KB) imports exactly
**one** symbol from `libJLCS.dll` — `jlcs_catch_current_exception`. Verified end
to end rather than "the DLL opens": thunks build, `_assert_aot_thunks_present`
passes against the PE export table, `dispatch_tier` reports `tier2` for all
three functions, calls return correct values, and a C++ `std::runtime_error`
thrown across a thunk arrives in Julia as
`CxxException("C++ exception: negative input: -7")` — message intact, which is
only true while one libJLCS holds one buffer.

### mingw unwinds with SEH (2026-09-05)

The personality function is `__gxx_personality_seh0`, not `__gxx_personality_v0`,
and the Windows C++ runtime exports only the former. The name was written down
twice in C++ (`JLCSPasses.cpp`, `JLCSCAPIWrappers.cpp`) and needed a third in
Julia. It is now `MLIRNative.CXX_PERSONALITY`, with `test_cxx_personality.jl`
pinning the two languages together — `test_mlir_templates.jl` had hardcoded the
Linux spelling, so correct Windows output failed the test.

### LLP64: two tiers disagreed about `long`, with nothing comparing them (2026-09-05)

C's `long` is 32 bits on Win64 and 64 on every Unix64 target — the one integer
whose width the word size does not settle. `Wrapper` got this right for free by
emitting Julia's `Clong`; the IRGen producers hardcoded `i64` in two separate
hand-filled tables (`TypeUtils.map_cpp_type`, `ArrayViewGen._AV_ELEM_MLIR`).

A disagreement raises no error anywhere. Tier 3 would say 4 bytes while the
Tier 2 MLIR thunk says 8, the thunk reads a different number of bytes than the
caller wrote, and the value is silently wrong. The width is one fact now,
`C_LONG_MLIR`, at package level for the same reason `INTERNAL_TYPE_BLOCKLIST` is
there: both sides of the tier boundary need it. `C_WCHAR_MLIR` follows for
`wchar_t` (UTF-16 on Windows, UTF-32 elsewhere), and the DWARF size table
follows `sizeof(Clong)`/`sizeof(Cwchar_t)` rather than restating them.

`test_llp64_widths.jl` asserts the **invariant**, not the constant: for every
spelling both tiers know, the two tiers agree on the width. That also catches the
next type added to one table and not the other, on whichever platform CI runs.
Negative-tested by reintroducing the bug — it names both offending types and
their byte widths.

### Windows paths are not string literals (2026-09-05)

`C:\Users` reaches a parser as `\U`. One bug class, three homes: both wrapper
generators baked `LIBRARY_PATH` into generated Julia by hand-quoting, so **every
generated wrapper on Windows failed to parse**; `ingest()` hand-rolled its TOML,
a second and disagreeing derivation of what `save_config` already did correctly;
and 11 test sites interpolated tempdirs into TOML. Now `repr()` and `TOML.print`.

### Smaller platform assumptions, each of which cost a session (2026-09-05)

- `FILE` is `struct _iobuf` under the UCRT, not `struct _IO_FILE`.
  `INTERNAL_TYPE_BLOCKLIST` named only the glibc spellings, so the screen caught
  nothing: a `FILE**` parameter came out as `Ptr{Ptr{_iobuf}}` and `_iobuf` was
  declared and **exported into the wrapper's API surface**. The UCRT locale and
  multibyte internals were added beside the glibc entries they mirror.
- `LLVM_CONFIG` was named in the "toolchain not found" error and read **nowhere**
  — following the advice changed nothing. Now honoured, and probed with `.exe`.
- Toolchain discovery looked for extensionless binaries, so a complete Windows
  install reported zero tools; llvm-config probing missed `.exe` and fell back to
  a hardcoded `"20.1.2jl"` *silently*, so a coherent LLVM 22 toolchain reported
  20 and warned about every tool in it.
- `PATH` was assembled with a Unix `:`, which does not fail loudly on Windows —
  it yields one unparseable entry and the prepended bin dir is simply absent.
- The shim-header collision guard compared with a hardcoded `/`, so the
  containment test never matched and the guard flagged the correctly-vendored
  header — failing the build for the packages it exists to protect.
- Generated `__init__` did `cglobal(:stdout)`, which is a glibc data symbol and
  not a symbol at all under the UCRT (a macro over `__acrt_iob_func`). Being in
  `__init__`, it took the whole module down. Now `@static`-branched and
  best-effort.
- `JSON.parsefile` defaults to `use_mmap=true` and Julia drops the mapping only
  at GC, so `clean()` failed on a tree RepliBuild had just built — and the file
  left behind was `compilation_metadata.json`, not the `.dll` anyone would have
  suspected. Free on POSIX, where a mapped file still unlinks.
- CMake names the dialect by host convention, so the build produced
  `libJLCS.dll` while `build.sh` verified `[ -f libJLCS.so ]` and printed
  "ERROR: not found" **after successfully linking it**. Both sites now follow the
  host; `MLIRNative` uses `Libdl.dlext`.
- `M_PI` is an X/Open extension. glibc exposes it from `<cmath>` anyway; the UCRT
  gates it behind `_USE_MATH_DEFINES`, so two stress-test fixtures failed to
  compile and took devtests' first testset with them.

### C bitflag enums are integer aliases, not closed `@enum`s (2026-09-02)

**This is the breaking change for existing wrappers**, and it needs a regenerate
to appear.

C spells two different things with one keyword. `enum { RED, GREEN, BLUE }` is a
closed set of alternatives and is exactly a Julia `@enum`. `enum { NULLTERM =
1<<0, STABLE = 1<<1, COMPOSE = 1<<3 }` is an **algebra** — callers OR members
together and pass a value that is not itself a member. Every enum took the first
shape, so the second could not be called at all: utf8proc's documented way to
request NFC is `NULLTERM|STABLE|COMPOSE == 11`, and `utf8proc_option_t(11)` threw
`ArgumentError: invalid value for Enum`. Reported from the Hub.

A bitflag enum now emits an integer alias plus named constants:

```julia
const utf8proc_option_t = Cuint
const UTF8PROC_NULLTERM = utf8proc_option_t(1)
```

Chosen over widening the call sites because **nothing else has to change**:
`Ptr{<name>}` resolves through the alias, a `ccall` argument typed `<name>`
resolves to the integer, and the alias and every member still reach `export`.
`|`, `&` and `~` become Julia's own. The cost, accepted: members print as
integers, and two masks over one underlying type are one Julia type.

Detection is the whole risk — a false positive strips a real enumeration of its
type and its named printing, a false negative just leaves the old behaviour — so
four rules are all required: at least three distinct single-bit members, no
contiguous runs, every value spelled only in declared bits, no negatives. The
fourth is bit **density**, and it is the one that earns its place: without it
libstdc++'s `_Ios_Seekdir` — `{0, 1, 2, 65536}`, beg/cur/end plus a sentinel —
passes every other test. A mask allocates consecutive bits; an enumeration that
happens to land on powers of two does not. Flags starting high are fine
(`ImGuiTabBarFlagsPrivate` occupies bits 20–22); gaps disqualify, offsets do not.

Blast radius measured over all **544 `@enum`s in the shipped Hub wrappers**, not
estimated: **59 reclassify, 485 unchanged** — the density rule is what took that
from 64 to 59. Every one of the 59 is a real mask: `ImGuiWindowFlags`,
`ma_sound_flags`, `mz_zip_flags`, `ggml_tensor_flag`, `chartype_t`,
`HUF_flags_e`. imgui carries most of them; existing wrappers keep their closed
`@enum`s until regenerated.

### A Windows-authored test was red on Linux (2026-09-05)

`test_windows_port.jl` asserted that `_is_system_decl_file` recognises MSYS2 and
WinSDK headers — spelled the way a Windows host writes them, `C:\msys64\…` and
`C:/Program Files/…`. `_canon_path` is deliberately the identity off Windows,
because a backslash is a legal character in a POSIX filename and POSIX paths are
case-sensitive, so three of those assertions could only ever pass on Windows.
The file was written on the port host and never run here.

Split into the two facts it was conflating. The **pattern list** is a plain
substring set and is now driven on every host with pre-canonicalised spellings,
so a Linux run still guards the Windows entries. The **canonicalisation** is
asserted as `== Sys.iswindows()`, which pins the host-conditionality as
deliberate instead of skipping it. 54/54 either way.

Same class as the vacuous-green failures this project keeps finding, rotated:
a test that states one host's answer rather than the invariant.

### Verified

**Windows** (`x86_64-w64-windows-gnu`, MSYS2 CLANG64, MLIR 22.1.8, Julia
1.12.7), measured on the port host: `runtests.jl` 31 testsets, exit 0.
`devtests.jl` reports **zero test failures** — Pipeline 12/12, Stress Test 4/4,
MI 43/43, VI 40/40, STL 28/28, c_test 72/72, C Abomination 15/15, Callbacks 3/3,
Ingest 8/8, MLIR Templates 87/87. The dialect builds and loads as
`src/mlir/build/libJLCS.dll` (46 MB).

Read that as "no fixture fails", not "one run printed all of it": the unwind
abort in **Open** below truncates some runs, so the last two figures come from
runs that got past it and from running those files directly.

**Linux** (reference host, current tree): `runtests.jl` **31 testsets, 1127
assertions, exit 0**. The SysV path is unchanged — `kHostAbi` selects SysV on
Linux and both rule sets always compile, so the inactive one cannot rot.

### Open

- **`libunwind: pc not in table`** aborts the `devtests.jl` parent process after
  the callback fixture, in some runs, before the last two sections. Zero test
  failures in every run — every assertion that executes passes — and it does not
  reproduce standalone (callback verify 3/3 over repeated runs) nor when the
  parent's build/wrap/spawn sequence is replayed on its own. It needs prior
  in-parent JIT activity, which points at SEH unwind-table registration for
  ORC-JIT'd frames on COFF rather than at anything in the fixture.
- **`test_abi_nested.jl` still states the SysV expectation** — a 16-byte all-float
  struct in XMM. On Win64 that inverts to sret. Latent: it passes today, and
  `xform_not_blob` is pinned so a Win64 sret blob cannot pass by accident.
- **PE inverts the export model** — `dllexport` is opt-in where ELF exports by
  default — which is a design problem for `__rb_*` static promotion. That is
  Tier 1, which is quarantined and which no shipped package takes, so it is
  deferred rather than solved.
- **macOS stays refused, for a real reason rather than an untested one.** The
  AAPCS64 classifier is unbuilt: `JLCSPasses.cpp` carries SysV and Win64 only and
  `#error`s on non-x86-64, so an arm64 Mach-O host has no struct-passing rules to
  apply. The specification half is provable from Linux today with the same
  clang-as-oracle method used for Win64.

## v3.3.4 (2026-08-30)

Patch, and mostly guards that were not guarding. A build-identity check that
returned on its first line for every package that had one; a docstring on every
generated function that Julia never attached; a virtual method the AOT path
built under a name no wrapper looks up; a `.ll → .so` step reaching for a
different clang than the one that emitted the IR. Also: `using RepliBuild`
stops publishing 221 names, and a `[wrap.varargs]` table now records the
upstream commit that proved it.

### The build-identity check never ran (2026-08-30)

Every generated C wrapper carried `BUILD_ID`, the GNU build ID of the library
it was generated from, and compared it at load. A build ID only exists if the
linker was asked to emit one, and RepliBuild never asks: the C bucket links
through `Clang_unified_jll` (`_clang_for_c_bucket`), which unlike the system
driver emits no note by default. So `BUILD_ID` interpolated empty and
`_check_build_identity` returned on its first line — for every C package built
after that routing landed.

Identity is the **SHA-256 of the library file** now. No linker flag, identical
behaviour for both buckets, and it also catches post-link modification. What it
gives up: a build ID survives `strip` and a file hash does not. The `binary`
stage strips before `wrap` runs, so the hash is taken of the already-stripped
file and the normal flow is unaffected — stripping a library *after* generating
its wrapper will now warn.

**The C++ bucket had no identity check at all**, despite its wrappers carrying
the same snapshot the C ones do — struct offsets, blob byte sizes, enum values,
sliced IR. It runs the same check now, in all three `__init__` variants.

**A second drift direction, because the hash cannot see it.** A file hash
catches a library that moved under a wrapper that was not regenerated. It is
blind to the reverse — the wrapper going stale against RepliBuild's codegen
while the `.so` matches byte for byte — and that is live here: struct layout,
dispatch tiers and ABI handling have all moved under existing wrappers this
year. `BUILD_GENERATOR` holds the generating version and is compared against
`pkgversion(RepliBuild)` at load. Neither check subsumes the other, so both run
separately with separate warnings.

`SHA` is re-exposed as `RepliBuild.SHA`: a wrapper vendored into a consumer
package cannot `using SHA` unless that package declares the dep, and routing
through RepliBuild — which the wrapper already imports — leaves the consumer's
Project.toml untouched.

### Every generated docstring was detached from its function (2026-08-30)

The doc heredocs end in a newline and every splice site emitted
`$doc_comment\nfunction …`, putting a blank line between the docstring and its
definition. Julia's docstring attachment does not survive that: the string is
evaluated as a no-op literal and the function ships undocumented, with no error
anywhere.

Measured across the shipped Hub wrappers by counting docstring closes, **11118
are followed by a blank line then `function`, against 1558 closing flush**. The
1558 are the generator's inline boilerplate — struct `setproperty`, the
`_cstring_wrapper_pair` `_ptr` siblings — which has always closed tight. That
asymmetry inside a single generated file is what makes this findable at all.

Chomped at all four splice sites: C, C++, and both vararg forms. Every existing
wrapper stays undocumented until regenerated.

### `long int`, `signed char` and 20 more spellings on the parameter path (2026-08-29)

DWARF emits the expanded C forms — `long int`, `short int`, `signed char`,
`unsigned`, `long long int` — and `_CPP_TO_JULIA_TYPE_MAP` carried only the
short ones, so those parameters resolved to `Any` while the return path typed
them correctly. 23 spellings were in `_DWARF_TYPE_MAP` and absent here; the 22
that are ordinary integers were added.

**`long double` is the 23rd and stays out, deliberately.** It is x87 80-bit,
SysV class X87, passed on the x87 stack — not in the integer registers an
`NTuple{2,UInt64}` would claim. The return map's answer is a storage size, not
a calling convention. Typing it here would convert a loud `Any` into a silently
mis-passed argument, which is strictly worse. It should stay the one gap until
the classifier grows an X87 path.

Also removed: `"char*" => "Cstring"` and `"const char*" => "Cstring"`, which
were unreachable — `cpp_to_julia_type` peels a trailing `*` before the lookup,
so a pointer never reaches a pointer-spelled key. Inert, but they read as if
this map unified the return and parameter mappings, which it must not: that
split is what `_cstring_wrapper_pair`, the `_ptr` siblings and
`_assert_cstring_policy` all rest on. The place someone would try to "fix" it
now says so.

Blast radius measured: across every Hub package exactly 11 parameters carried a
missing spelling — `long double` 6 and `__int128` 4 (both fmt, which has no
build artifacts), `signed char` 1 (llamacpp).

### The copy-constructor matcher never matched a namespaced class (2026-08-29)

`_collect_class_raii` tested `occursin("$(cls)::$(cls)(", demangled)`. A
constructor demangles as `<fully-qualified class>::<bare name>(` —
`pugi::xml_node::xml_node(…)` — so interpolating the class twice built
`pugi::xml_node::pugi::xml_node(`, which matches no real demangle. Every
namespaced class silently recorded no copy constructor, and the scope-RAII
producer never emitted the caller-owned temporary the Itanium ABI wants for a
by-value parameter of such a class. Non-namespaced fixture classes were
unaffected, which is why mi/vi/stl never caught it. Two copy constructors
gained across the Hub, both pugixml.

**STL container size was hardcoded twice.** `TypesCpp` used the word-boundary
aware `_stl_name_match` (added so `std::string_view` does not match
`std::string`); `TypeUtils` used bare `startswith` and answered 32 for
`std::string_view`. FunctionGen reads the TypeUtils one, so a by-value
`string_view` would have had a 32-byte blob thunk on the thunk side and "not a
container, size 0" on the wrapper side — thunk and wrapper disagreeing about a
buffer, the ImGui receiver-miss shape. Latent; no by-value `string_view` in the
Hub today. Now one table, in TypeUtils, because Wrapper imports IRGen and not
the reverse.

Both found by an audit of the "one fact, two derivations" class
(`docs/updates/2026-08-29-one-fact-two-derivations.md`).

### `using RepliBuild` published 221 names (2026-08-29)

Roughly 195 of them were implementation detail: every DWARF parser entry point,
the whole MLIR context/JIT lifecycle, the IR-gen thunk emitters, BuildBridge's
command runners, the config accessors, the LLVM toolchain helpers.
**`names(RepliBuild)` reports 31 now** — 13 verbs and the 18 submodules. The
verbs are the six pipeline steps
(`build`/`wrap`/`info`/`discover`/`clean`/`ingest`), the registry
(`use`/`register`/`unregister`/`list_registry`/`search`/`scaffold_package`) and
`check_environment`; the submodule names are the deliberate escape hatch every
Hub consumer reaches through.

**Removing an `export` removes no binding.** `RepliBuild.Compiler.compile_to_ir`
and `RepliBuild.parse_vtables` still resolve; only the namespace dump stops.
That matters because `execute`, `capture`, `lookup`, `search`, `info` and
`build` are all plausible names in a consumer's own code — the same
namespace-competition argument the generated wrappers already apply when they
withhold `error` and `stat`.

Measured before cutting: of the 221, only 59 appear as a bare word anywhere
under `test/`, `docs/` or the Hub, and the Hub's share turned out to be
entirely comments and prose (`lookup` in five files is the English word).

### A wrapper-bound virtual took the legacy thunk path on AOT builds (2026-08-29)

A virtual method the wrapper binds must go through the function-thunk pass:
only FunctionGen emits the `_mlir_ciface_<mangled>_thunk` convention that
`invoke`/`invoke_aot_ptr` looks up, and only it emits `jlcs.vcall`, so an
override is honoured instead of getting `p->Class::method()` static semantics.
The legacy vmethod-IR pass emits `thunk_<mangled>`, which no wrapper has looked
up since 2026-07-17.

`needed_symbols` is the routing key for that decision — not merely dead-thunk
elimination. JITManager passes the thunk manifest and gets the right answer;
`build_aot_thunks` passed nothing, so every virtual fell to the legacy pass.
AOT built a symbol the wrapper never asks for and omitted the one it does.

Two changes, because the manifest alone is not sufficient. ThunkBuilder reads
`thunk_manifest.json` and threads it through as JITManager does. And
`needed_symbols === nothing` no longer means "no virtual is wrapper-needed":
AOT runs at **build** and the manifest is written at **wrap**, so a cold build
has none, and the old default reproduced the bug on any fresh package. Absent a
manifest, routing falls back to metadata's function list — the collection
GeneratorCpp iterates when deciding what to emit. Virtuals outside it
(inherited `std::exception::what`, D4 deleting destructors, `std::type_info`
members) still take the legacy path; they are reached through the vtable
internally and never become API.

Verified on clipper2: all seven now define `_mlir_ciface_*_thunk`, zero legacy
`thunk__Z*` remain, wrap succeeds with 157 `invoke_aot_ptr` sites and 0 legacy
ccalls, and `Clipper2Exception::what` comes back as `Union{String,Nothing}` —
the Cstring policy applies, which it did not on the legacy path.

The diagnosis is `docs/updates/2026-08-29-virtual-branch-policy-audit.md`. It
also caught that the obvious fix — pointing the wrapper at `thunk_<mangled>` —
reverts the 2026-07-17 routing decision, drops override-honouring dispatch,
carries the wrong arity and skips the Cstring pair. That approach was
implemented, measured with `test/ab_virtual_dispatch.jl`, and abandoned on the
audit's evidence.

### A wrapper could bind AOT thunks its library does not define (2026-08-29)

The AOT pass (build, from compilation metadata) and the wrapper generator
(wrap, from the dispatch decision) can disagree about which thunks exist, and
nothing checked. A slot the thunks library does not define resolves to `C_NULL`
and raises on whichever call the user makes first, announced only by a
load-time warning — the silent-until-you-touch-it class.

Wrap already knows the exact slot list and the thunks library exists by then,
since AOT runs at build, so it can simply be asked what it defines.
`_aot_thunk_slot_names` derives the referenced set once and `_aot_thunk_symbol`
owns the `_mlir_ciface_<mangled>_thunk` spelling once, so the check and the
emitted `_THUNK_SLOTS` table cannot diverge.

Found clipper2's gap immediately — 7 of 156 — which the entry above fixes.
`test_introspection` 60 → 69, including a fires-on-a-real-library check against
libc rather than trusting the missing-file branch to stand in for it.

### `[wrap.varargs]` is pinned to the commit it was verified against (2026-08-27)

A varargs table is the one part of a `replibuild.toml` a rebuild cannot
re-derive. Sources, flags, headers and excludes all fail loudly when they stop
matching upstream — a missing file, an unresolvable header, a link error.
Variadic signatures do not: they are a human's reading of upstream's API, and
after a version bump they regenerate wrappers describing calls that may no
longer exist, reporting success either way.

```toml
[wrap.varargs]
proven_at = "7579fc9d7ed90240487251dfb69168f8e64e9294"
lua_pushfstring = [["Cstring"], ["Cint"]]
```

Enforced in `wrap_library` against the resolved `[dependencies.*]` commit —
the entry matching `project.name`, else the sole entry carrying one. Prefixes
match, so a short hash works. A mismatch is an error naming both commits and
the affected functions. **A pin with nothing to compare against is also an
error**: it reads as provenance and provides none. A varargs table with no pin
warns. Claimed only when the value is a `String`, so a C function genuinely
named `proven_at` still parses as an overload table.

Enforcement sits in `wrap_library` rather than `load_config` because wrapping
is the only stage that consumes varargs — a bare `build()` has nothing to go
stale — and because the config-surface guard correctly refused to count a field
read only inside its own parser. Seven Hub packages carry varargs tables and
will warn until pinned: curl, lua, box2d, box2d3, zlib, imgui, sqlite.

### The C bucket's final link used a different clang than its IR (2026-08-27)

The C bucket keeps every tool that touches its IR version-matched: JLL clang
emits it, Julia's resident libLLVM links and optimizes it. The final
`.ll → .so` step was not on that path — `create_library` and
`create_executable` shelled straight to system `clang`.

That breaks on any C source containing a weak-symbol NULL check. JLL clang 18
legally emits `icmp eq (ptr @weak_fn, ptr null)` as a constant expression, LLVM
19 removed icmp constexprs, and system clang 22 then rejects IR this pipeline
produced itself:

```
error: icmp constexprs are no longer supported
```

zstd's tracing hooks are the first Hub source to do it. `_clang_for_c_bucket`
is extracted and `create_library`, `create_executable` and
`compile_single_to_ir` all route through it.

Deliberately **not** `LLVMEnvironment.resolve_tool(name, :c)`: that looks for a
parallel system install, and Arch's `llvm18` package ships LLVM tools with no
clang binary at all — so it finds `/usr/lib/llvm18/bin`, misses clang, and
silently returns the bare name, landing back on the system compiler. It remains
correct for `llvm-link` and `opt`, which that package does ship.

Verified by rebuilding Hub zstd with its `-DZSTD_TRACE=0` pin **removed**: 26
`icmp eq (ptr @ZSTD_trace…)` constexprs present in the per-file IR and in the
assembled `zstd_abi.ll`, build succeeds at 4.2 MB with 4 `ZSTD_trace` symbols
in the `.so`. That pin can come out.

### gcc's `DW_OP_lit` spelling lost every virtual-base offset (2026-08-27)

Both DWARF parsers matched only clang's `DW_OP_constu N` when reading the
vbase-offset expression off a `DW_TAG_inheritance` edge. gcc emits the
single-byte `DW_OP_lit<N>` form for N ≤ 31, so every gcc-built library —
libstdc++ among them — failed to parse and silently lost the offset.

The two producers failed *differently*, which is why nothing caught it:
`Compiler.jl` (readelf) never set the key, so `_collect_upcasts!` saw `nothing`
and took its `@debug` branch; `DWARFParser.jl` (llvm-dwarfdump) left the pushed
default 0, and JLCSIRGenerator skips on `!= 0`. Either way the virtual-base
upcast helper was simply not emitted, and the only trace was an `@debug` line
invisible at default log level. No wrong pointers — a missing capability
announcing itself to nobody.

Both regexes accept either spelling now, and the `@debug` fall-through is an
error. A vbase-offset of 0 is structurally impossible — the entry always sits
below the vtable address point, so a parsed value is always negative — so both
`nothing` and 0 mean "unparsed" and stop the build naming the class and base.
An unhandled `DW_OP` form lands there instead of silently degrading.

Found while probing libstdc++'s iostream diamond. Neither path is reachable
from a supported config today (RepliBuild consumes only DWARF it generated with
clang, and `ingest` refuses C++), so the widened regex is defensive; the guard
is the part that earns its place now.

### Configure-time header generation is a build capability: `SysConfigGen` (2026-08-26)

A library whose headers come from feature detection cannot be *built* without
this — the compile has nothing to include — so it belongs in the builder, not
in RepliBuildTooling, which is for looking at what was produced.
`RepliBuildTooling/src/Introspect/CMakeHarvest.jl` moves to
`src/Builder/SysConfigGen.jl` as a proper module.

The lift was clean; the one thing the file took from its old host's namespace
was `execute`, now an explicit `using ..BuildBridge: execute`. That dependency
was invisible until the move and surfaced as an `UndefVarError` the first time
the end-to-end test ran in the new home.

Renamed while moving, because "harvest" reads as generated filler rather than
as what the code does: `harvest_config` → `capture_config`, `propose_toml` →
`toml_fragment`, `HARVEST.md` → `SYSCONFIG.md` (the emitted provenance file).
`cmake_probe`, `main_target` and `uniform` keep their names — those describe
the operation.

Tests moved into `test/test_sysconfiggen.jl`, wired at devtests §18 because the
end-to-end case shells to cmake; 31/31, driving a self-contained generated
CMakeLists, so no network and no vendored library. Tooling stays green at
97/97 with zero remaining references. Hub-side, pcre2's `harvest.jl` uses
`RepliBuild.SysConfigGen` and its checked-in `config/HARVEST.md` became
`config/SYSCONFIG.md` — which matters beyond cosmetics, since
`packages/pcre2/test.jl` asserts that file by name.

### One suite owns each test file, and the version is one number (2026-08-26)

The wiring guard's only rule was "included by *a* suite", which a file sitting
in **both** satisfies. `test_registry.jl` did exactly that: no suite owned it,
it ran twice for anyone running both, and nothing in the tree answered "is this
a CI test or a toolchain test?" — the same ambiguity that lets a toolchain
dependency drift into CI unnoticed. It is runtests-owned now, verified
toolchain-free rather than assumed (run with clang hidden from `PATH` it passes
6/6). `runtests.jl` asserts the two suites are disjoint, with a `SHARED` dict —
empty — for a deliberate exception, so an overlap has to be someone's decision
rather than nobody's accident.

`Project.toml` and `const VERSION` were two independent literals with nothing
reconciling them. **`VERSION` is derived from `Project.toml` now**, with an
`include_dependency` so editing Project.toml invalidates this module's
precompile cache — without it the old number stays baked into the `.ji` and the
derivation would be a fiction.

Not cosmetic: `VERSION` feeds `_generator_fingerprint`, which gates the
registry build cache, so a wrong version serves a stale-codegen wrapper instead
of rebuilding it — and it is stamped into every emitted wrapper as
`BUILD_GENERATOR`. Guarded in `runtests.jl` by three checks: it equals the
TOML, it equals `pkgversion(RepliBuild)` (an independent read through Julia's
own resolution), and the source still contains the derivation rather than a
re-introduced literal.

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
