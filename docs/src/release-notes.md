# Changelog

The full, dated history is
[`CHANGELOG.md`](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/CHANGELOG.md)
in the repository. This page is what a user upgrading should know.

This manual is built from `main`. `Pkg.add("RepliBuild")` installs the newest
registered release, which can be older than what is described here;
`pkgversion(RepliBuild)` says which version you have.

## After any upgrade

- **Generator fixes** reach a wrapper when it is generated again. Run
  `wrap(toml)` for a wrapper you `include`; a wrapper written by an older
  RepliBuild warns when it loads. Packages loaded through `use()` are re-wrapped
  by themselves, because its cache key includes RepliBuild's version.
- **Fixes in what `build` extracts** (the metadata) need a rebuild as well, and
  a build whose inputs did not change is skipped — by `use()` too. Run
  `build(toml; clean=true)` before `wrap`, or delete
  `.replibuild_cache/project_hash` and run `build(toml)`, which keeps the
  compiled IR. For a registered package, call `use(name; force_rebuild=true)`
  once. The notes below say when a release has such fixes.
- **On Windows**, build the JLCS dialect again: the new version lives in a new
  folder ([Install](install.md#The-JLCS-dialect)).

## Not yet released (on `main`)

- **ABI fixes from the differential fuzzer**, which generates C records and
  functions and checks both tiers against the compiler.
  - Tier 3 (C): whether a record comes back in registers or through a hidden
    buffer is decided per target, from real alignment; that buffer has the
    record's alignment; one `long double` parameter refuses that function, not
    the whole wrapper; records and enums named after Julia keywords are spelled
    as declared; a module name never starts with a digit.
  - Tier 2 (C++): unions and bitfield records pass and return by value as their
    bytes; `#pragma pack` bodies stay packed; bitfields wider than 32 bits get
    working accessors.
- **Switching Julia or the C++ toolchain** rebuilds what the switch affects.
- **Rebuild, then re-wrap.** Member sizes and record alignment are now
  recorded in the metadata.

## v4.2.1 (2026-10-17)

**LLVM.jl 10.** LLVM.jl compat widens to `9, 10`. Tier 1 (experimental, off by
default) reads the names LLVM.jl 10 changed the meaning of, so it behaves the same
on both, and a Slicer exception demotes that one function instead of failing the
wrap. On Windows, `[wrap.tier1] allow_setjmp` is ignored with a warning. No API
change; a 4.2.0 wrapper keeps working.

## v4.2.0 (2026-10-16)

**LLVM 23 and Windows, end to end.** New: `[platform.<host>]` tables in
`replibuild.toml`, JSON 1.x, the JLCS dialect on LLVM/MLIR 23, and end-to-end
coverage of AOT thunks. Behaviour changes: relative `[compile]` paths resolve
against the project root, codegen runs at `[link] optimization_level`, and a C++
package compiles its `.c` files with the system clang. A 4.1.1 wrapper keeps
working. Build-time fixes, the libc++ exports among them, need a rebuild;
generator fixes need a re-wrap.

## v4.1.0 and v4.1.1 (2026-09-24, 2026-09-27)

**Bounding the wrap surface.** `RepliBuild.DAGDiff` is removed (it was
exported), and `[wrap] dag` is ignored with a warning. New keys, all opt-in:
`[compile] visibility`, `[wrap] exclude_symbols`, `[wrap] surface_types_only`
and `surface_types_extra`. `[link] link_dirs` now emits an rpath as well as
`-L`. 4.1.1 is a patch of generator and DWARF-reader fixes: no new keys, nothing
removed, and a 4.1.0 wrapper is unchanged until it is regenerated.

## v4.0.0 and v4.0.1 (2026-09-05, 2026-09-13)

- **Windows is supported.** `x86_64-w64-windows-gnu` (mingw, MSYS2 CLANG64) —
  not MSVC. See [Install](install.md). macOS is still refused at load, because
  the AAPCS64 ABI classifier is not built.
- **C bitflag enums are integer aliases, not `@enum`s.** This appears when you
  **regenerate** a wrapper; existing ones are unchanged until then. An enum whose
  members are a mask now emits `const Flags = Cuint` plus named constants, so
  `A | B` works and can be passed. Detection is conservative — 59 of 544 enums
  across the shipped Hub wrappers reclassify — but if you had been matching on an
  enum instance, or printing one and expecting its name, that is what changed.
  `Ptr{Flags}`, `ccall` arguments typed `Flags`, and `export` all behave as
  before.
- **4.0.1** made Julia 1.13 (libLLVM 20) work. No API changes.

## What still matters from the 3.x line

Calling code written against older generated APIs may need these updates:

- **A pointer parameter takes a pointer or a `Ref`.** Overloads that accepted a
  struct value for an `S*` parameter were removed: the callee got a pointer to
  a temporary copy. A by-value struct parameter still takes the value.
- **`char*` returns are `Union{String,Nothing}`.** NULL is `nothing`, not an
  exception. Declare malloc'd returns in `[wrap.cstring_owned]`. Every such
  function has a raw `<name>_ptr` sibling.
- **`[wrap.varargs]` lists Julia types, variadic arguments only**, as a list of
  lists: `f = [["Cint"]]`. C types belong in `[wrap.macros]`, not here.
- **`discover(force=true)` preserves hand-edited keys** (templates, macros,
  varargs, shim headers, cstring ownership). Keep the rest of a hand-written
  config in version control anyway.
- **Vendored wrappers resolve their `.so` sibling-first**, so a copy in `lib/`
  stays bound to the `.so` next to it.

## Where to read more

- [CHANGELOG.md](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/CHANGELOG.md)
  — every release, including the v3.0 generated-API break and the inheritance
  work in v3.0.1.
- Dated engineering notes (contributors): [`docs/updates/`](https://github.com/obsidianjulua/RepliBuild.jl/tree/main/docs/updates)
  in the repo.
