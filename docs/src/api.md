# API Reference

```@meta
CurrentModule = RepliBuild
```

The public API is the verbs below. Everything else — compiler, generators,
DWARF parser, MLIR bindings — is internal. Contributors: [Developer](developer.md).

Every entry point takes a path to a `replibuild.toml` (or a project directory
that contains one) and is idempotent: unchanged inputs hit the content-hash
cache and return.

## Build lifecycle

```julia
toml = RepliBuild.discover("path/to/project")   # scan → replibuild.toml
RepliBuild.build(toml)                           # clang → .so
RepliBuild.wrap(toml)                            # .so → julia/<Module>.jl
```

`discover(..., build=true, wrap=true)` runs the whole pipeline; on a project
that already has a `replibuild.toml` it keeps that file and runs `build` and
`wrap`. `discover(force=true)` regenerates the config but **preserves** the
hand-edited keys listed in
[Edit the TOML](config.md#1.-How-the-file-is-created-and-maintained).
`build(clean=true)` deletes `build/`, `julia/` and `.replibuild_cache/` first,
then builds from nothing.

```@docs
discover
build
wrap
clean
info
```

## Registry

A project builds where it lives. `register` builds and wraps it in place, vendors
the build to `~/.replibuild/builds/<hash>/`, and adds a Julia package around it to
a Julia environment (`env`, default the `@v#.#` environment). `use("name")` loads
the vendored build, or rebuilds the project in place after an edit. A fresh
install has an empty registry — there is no Hub fallback.

```julia
RepliBuild.register("replibuild.toml")   # also: env=:active, env="path", env=nothing, build=false
RepliBuild.use("myproject")              # name from [project].name
using MyProject                          # from the Julia environment register added it to
RepliBuild.list_registry()
RepliBuild.search("xml")                 # browse Hub TOML names; does not register
```

`REPLIBUILD_HOME` relocates `~/.replibuild/`. `REPLIBUILD_JULIA_ENV`
(`default`, `active`, `none`, or a path) changes where `register` adds packages.
`REPLIBUILD_HUB_URL` points `search` at a private index mirror. The vendored
build's hash includes RepliBuild's version, so after an upgrade `use` wraps each
registered package again, once; a fix in what `build` extracts also needs
`use(name; force_rebuild=true)`. See [Registry](use.md).

```@docs
register
use
search
list_registry
unregister
scaffold_package
```

## Ingest (experimental, C only)

For C libraries whose build system the source pipeline cannot reproduce. Build
the `.so` yourself with `-g`. C++ API surfaces are **not** supported.

```julia
toml = RepliBuild.ingest("/path/to/libfoo.so",
                         headers=["/path/to/include"],
                         name="foo", language=:c,
                         build=true, wrap=true)
```

As with `discover`, only `build=true, wrap=true` registers (after wrapping);
`register=false` skips it, and a bare call writes the TOML only.

```@docs
ingest
```

## Environment

```@docs
check_environment
```

`status.ready` — C builds will work. `status.tier2_ready` — C++ and the hard
ABI path will work. See [Install](install.md).

## What a generated wrapper exposes

Not exported. Qualified access only. Full usage: [Call a wrapper](calling.md#Introspection).

| Name | What it answers |
|------|-----------------|
| `struct_size(name)` | Byte size of a wrapped struct |
| `member_offset(name, member)` | Byte offset of a member |
| `LIBRARY_PATH` | The library the wrapper calls (the copy beside the wrapper first, else the build-time path) |
| `LIBRARY_SHA` / `BUILD_TARGET` / `BUILD_GENERATOR` | SHA-256 of the library it was generated against, the target triple, and the RepliBuild version that wrote it |
| `dispatch_tier(f)` | Which path `f` will actually take (`:tier1` / `:tier2` / `:tier3`, or `:unknown` / `:mixed` / `:deferred`) |
| `DISPATCH_TIER` | What the generator emitted |

`dispatch_tier` and `DISPATCH_TIER` exist only in a wrapper that mixes tiers or
has Tier 1 kernels; a single-tier wrapper names its tier in a comment instead.
