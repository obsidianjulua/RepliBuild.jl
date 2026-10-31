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

## v4.3.1 (2026-10-31)

**Multi-library projects build and wrap from CMake's own build graph.**

- **Graph mode.** A `[graph]` section ([TOML reference](config.md)) builds a
  CMake project of several libraries. RepliBuild runs only CMake's configure
  step and reads its File API: every target, its sources, flags, defines and
  include path, including what it inherits from the libraries it links. It runs
  the build steps that write generated sources, compiles and links each library
  itself (C libraries with Julia's clang, C++ with the system clang), and
  extracts each one's metadata. `wrap` writes one wrapper per library and a
  graph module that includes them in load order, so `G.Shapes.f(…)` calls a
  function of the `shapes` library. A library named like a Julia binding gets
  an `R` (`Core` → `RCore`). `register` and `use` handle the graph as one unit.
  Types are still per library: a struct two libraries share is a different
  Julia type in each wrapper.
- **Git submodules.** `[dependencies.<n>] submodules = ["libs/math"]` fetches
  exactly those submodules of a git dependency, at the commits its pinned
  `commit` records, so a library inside a superproject such as `boostorg/boost`
  no longer needs a dependency of its own.
- **`[wrap] shim_defines`** builds a single-header library's implementation
  (`STB_IMAGE_IMPLEMENTATION`) from the TOML, in the shim file only.
- **C++ finalizers free what they own.** A `Managed<X>` or factory struct calls
  the library's deleter function for the class, else its deleting destructor;
  a class with neither gets no owning type. Re-wrap to pick it up.
- **A Tier 2 record holding a template member** (`std::array`, `std::pair`, a
  project template) keeps its size when it crosses by value.
- **`clean(toml; deps=false)`** keeps the vendored git clones, so a rebuild from
  cold checks their pins instead of fetching them again.
- **`ingest` registers the way `discover` does:** only with `build=true,
  wrap=true`, after wrapping. A bare call writes the TOML and nothing else.
- **`Pkg.build` builds the JLCS dialect on Windows**, and rebuilds it after an
  LLVM minor upgrade.
- **Fixes in what `build` extracts — rebuild, then re-wrap:**
  - Tier 2 on Linux classifies a record passed or returned by value from its
    DWARF facts, not from the body it crosses as. Records with padding, nested
    records, unions, bitfields, `long double` or `__int128` crossed wrong.
  - Tier 3 on Linux places a by-value record C aligns to 16 in the stack slot C
    reads, or refuses the call.
  - Template parameters are read on Linux, and `const void *` is recorded as
    itself instead of an unknown type.
  - On Linux every library binds its own exported functions to itself
    (`-Bsymbolic`), and Tier 1 slices name them by alias: a library that Julia
    also loads (zlib) no longer calls Julia's copy.
- **`surface_types_only`** keeps the records a signature spells with template
  arguments.
- **Header enums** come only from the project's own headers, and
  **`scaffold_package`** includes the wrapper file `wrap` writes.
- Parallel builds no longer clear the caller's `LD_LIBRARY_PATH` and `CPATH`.

## v4.3.0 (2026-10-23)

**The registry builds a project where it lives; AOT thunks come from `wrap`.**
`register` and `use` change what they do (below), so this is a minor release.
v4.2.1 was never registered: its changes reach users with this release.

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
- **Rebuild, then re-wrap.** Member sizes, record alignment, names that
  several types share, and function-pointer parameters are now recorded in the
  metadata. The first build also compiles every file once more, because IR file
  names changed.
- **The registry builds a project where it lives.**
  - `register` builds and wraps the project in place, vendors the build to
    `~/.replibuild/builds/<hash>/`, and adds a Julia package around it to your
    Julia environment (`env`, default the `@v#.#` one). `using MyProject` loads
    it from there, and `use("myproject")` loads it too.
  - `use` follows edits by rebuilding the project in place, and loads the last
    vendored build if the project has moved.
  - A bare `discover` no longer registers anything; `discover(...; build=true,
    wrap=true)` ends by registering.
  - The registry no longer copies a project's sources into itself, so two
    projects with the same relative paths can no longer overwrite each other.
  - A project registered by an earlier RepliBuild keeps building the copy that
    version made. Register it again, from the project.
- **AOT thunks in one pass.** With `[compile] aot_thunks = true`, `wrap`
  compiles the thunks library as its last step, for exactly the thunks the
  wrapper binds; `build` no longer compiles thunks. A new package used to need
  `build` and `wrap` twice.
- **The same source wraps the same** on Julia 1.12 and 1.13 and after an
  upstream edit. A regenerated wrapper changes once: lambda closure types are no
  longer emitted, and where several types share a name, the definition kept is
  chosen by content.
- **A call that passes by value a type whose name several types share** is an
  ABI-trap stub that says why, instead of using whichever layout was kept.
- **Function-pointer types carry their parameter lists**, and a callback's
  `@cfunction` example comes from its own typedef or DWARF signature, never
  from a name match.
- **Windows links are reproducible**: the same IR gives the same DLL, so
  `LIBRARY_SHA` changes only when the library does. A link longer than one
  Windows command line goes through a response file.
- **`scaffold_package`** writes a RepliBuild compat entry that admits the
  RepliBuild that generated the package. It wrote `RepliBuild = "2"`, which
  resolved RepliBuild 2.x.

## v4.2.1 (2026-10-17, released in v4.3.0)

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
