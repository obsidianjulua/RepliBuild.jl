# Changelog

The full, dated history is
[`CHANGELOG.md`](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/CHANGELOG.md)
in the repository. This page is what a user of the current release should know.

Current version is **v3.3.4**.

## What still matters from the 3.x line

Wrappers regenerate automatically through the fingerprinted `use()` cache.
Calling code written against older generated APIs may need these updates:

- **Pass a pointer or `Ref` for by-value structs.** Overloads that took a
  struct by value were removed (they were UB-adjacent).
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

## v3.3.4 (2026-08-30)

Guards that were not guarding, now are: the wrapper's build-identity check
hashes the library file (and compares `BUILD_GENERATOR` to the installed
RepliBuild); generated docstrings actually attach to their functions; AOT
virtual methods use the symbol the wrapper looks up. `using RepliBuild` no
longer dumps ~200 internal names into your namespace.

## Where to read more

- [CHANGELOG.md](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/CHANGELOG.md)
  — every release, including the v3.0 generated-API break and the inheritance
  work in v3.0.1.
- Dated engineering notes (contributors): [`docs/updates/`](https://github.com/obsidianjulua/RepliBuild.jl/tree/main/docs/updates)
  in the repo.
