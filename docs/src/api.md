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

`discover(..., build=true, wrap=true)` runs the whole pipeline. `discover(force=true)`
regenerates the config but **preserves** the hand-edited keys listed in
[Edit the TOML](config.md#1.-How-the-file-is-created-and-maintained).
`build(clean=true)` ignores the cache.

```@docs
discover
build
wrap
clean
info
```

## Registry

`use("name")` loads a project already in the **local** registry. `discover`
registers for you; otherwise call `register` first. A fresh install has an
empty registry — there is no Hub fallback.

```julia
RepliBuild.register("replibuild.toml")
RepliBuild.use("myproject")              # name from [project].name; cached
RepliBuild.list_registry()
RepliBuild.search("xml")                 # browse Hub TOML names; does not register
```

`REPLIBUILD_HOME` relocates `~/.replibuild/`. `REPLIBUILD_HUB_URL` points
`search` at a private index mirror. The `use` cache key includes RepliBuild's
version, so an upgrade rebuilds each registered package once.

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
| `dispatch_tier(f)` | Which path `f` will actually take (`:tier1` / `:tier2` / `:tier3`, or `:unknown` / `:mixed` / `:deferred`) |
| `DISPATCH_TIER` | What the generator emitted |
| `struct_size(name)` | Byte size of a wrapped struct |
| `member_offset(name, member)` | Byte offset of a member |
| `BUILD_ID` / `BUILD_TARGET` / `BUILD_GENERATOR` | Identity of the library and the generator that wrote this file |
