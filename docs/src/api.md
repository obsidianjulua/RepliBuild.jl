# API

```@meta
CurrentModule = RepliBuild
```

The functions you call. Each takes the path of a `replibuild.toml`, or of the
folder that holds one, and does nothing when its inputs have not changed.

## Check the machine

```@docs
check_environment
```

`status.ready` is true when C libraries can be wrapped, and
`status.tier2_ready` when C++ libraries can. [Set up](install.md) explains the
report.

## Build and wrap

```julia
toml = RepliBuild.discover("path/to/project")   # write replibuild.toml
RepliBuild.build(toml)                           # compile the library
RepliBuild.wrap(toml)                            # write julia/<Name>.jl
```

- `discover(path; build = true, wrap = true)` does all three and registers the
  result.
- `discover` keeps an existing `replibuild.toml`. `discover(path; force = true)`
  rewrites it, keeping only [some keys](config.md#What-a-forced-discover-keeps).
- `build(toml; clean = true)` deletes everything built first. Downloaded
  sources are kept.
- `clean(toml)` keeps downloaded sources too. At the Julia prompt it asks
  whether to delete them; `clean(toml; deps = true)` deletes them without
  asking, and `deps = false` keeps them without asking.

```@docs
discover
build
wrap
clean
info
```

## Register and load

```julia
RepliBuild.register("replibuild.toml")
M = RepliBuild.use("myproject")
```

[Load it anywhere](use.md) explains what each does.

```@docs
register
use
list_registry
unregister
search
scaffold_package
```

| Environment variable | Effect |
|:---------------------|:-------|
| `REPLIBUILD_HOME` | Where registered builds are kept. Default `~/.replibuild`. |
| `REPLIBUILD_JULIA_ENV` | Which Julia environment `register` adds packages to: `default`, `active`, `none`, or a path. |
| `REPLIBUILD_HUB_URL` | Where `search` reads the list of recipes from. |

## Helpers for writing a recipe

| Function | Use |
|:---------|:----|
| `RepliBuild.VisibilityProbe.visibility_probe(toml)` | Says whether `[compile] visibility = "hidden"` is safe for this library, and which functions it would drop. [Leaving things out](toml.md#Leaving-things-out) |
| `RepliBuild.SysConfigGen.cmake_probe(source_dir; args)` | Runs a library's CMake configure step to find the headers it generates. [Headers the build system generates](toml.md#Headers-the-build-system-generates) |
| `RepliBuild.SysConfigGen.capture_config(probe, dir)` | Copies those headers into `dir`. |
| `RepliBuild.SysConfigGen.toml_fragment(probe)` | Prints suggested `[compile]` and `exclude` entries. |

## In every generated module

Not exported; call them with the module's name. [Call the wrapper](calling.md#Sizes-and-offsets)

| Name | Gives |
|:-----|:------|
| `struct_size(name)` | A struct's size in bytes |
| `member_offset(name, member)` | A member's offset in bytes |
| `STRUCT_SIZES`, `STRUCT_OFFSETS` | All of them, as `Dict`s |
| `libc_free(ptr)` | Frees memory the library got from `malloc` |
| `LIBRARY_PATH` | The library file the module calls |
| `LIBRARY_SHA`, `BUILD_GENERATOR` | The library's SHA-256, and the RepliBuild version that wrote the module |

## Ingest

Wraps a C library you compiled yourself with `-g`, without building it from
source. It is experimental and does not support C++. Prefer `discover`,
`build` and `wrap` whenever the source builds.

```@docs
ingest
```
