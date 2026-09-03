# RepliBuild

```@meta
CurrentModule = RepliBuild
```

Point RepliBuild at C or C++ source. It compiles the library, reads the ABI the
compiler actually emitted, and writes a Julia module you can call. You drive it
with a few verbs and a `replibuild.toml`. You do not write `ccall`s, and you do
not maintain generated bindings.

This site is the user manual: how to install, how to wrap a library, what to
edit in the TOML, and how to call the result.

## I want to…

| Goal | Page |
|:-----|:-----|
| Check the toolchain | [Install](install.md) |
| Wrap a C/C++ project | [Wrap a library](guide.md) |
| Fill in macros, varargs, ownership, flags, excludes | [Edit the TOML](config.md) |
| Call the generated functions, structs, and C++ classes | [Call a wrapper](calling.md) |
| Ship a Julia package on top of a wrapper | [Ship a package](using-wrappers.md) |
| Reload a project you already wrapped | [Registry](use.md) |
| Look up a function | [API](api.md) |
| Something failed or is missing | [Troubleshooting](troubleshooting.md) |

How RepliBuild is built — the dialect, the inheritance ABI, the pipeline — is
under [Developer](developer.md). You do not need it to wrap a library.

## First wrap

```julia
using RepliBuild

toml = RepliBuild.discover("path/to/project")   # writes replibuild.toml
# edit the TOML: flags, excludes, macros, varargs, ownership
RepliBuild.build(toml)                          # clang → .so
RepliBuild.wrap(toml)                           # .so → julia/MyProject.jl

include("path/to/project/julia/MyProject.jl")
using .MyProject
```

Or all at once: `RepliBuild.discover("path/to/project", build=true, wrap=true)`.
Then open the TOML and fill in what discovery cannot see — that edit is the
whole skill. [Wrap a library](guide.md) walks it; [Edit the TOML](config.md)
is the reference.

`discover` also registers the project locally. Later you can reload it with
`RepliBuild.use("myproject")` (`[project].name`) instead of `include`. A fresh
install has an empty registry — `use("cjson")` does not work until you have
registered that name yourself. See [Registry](use.md).

## The verbs

| Function | What it does |
|:---------|:---------------|
| `discover(path)` | Scan a source tree, write `replibuild.toml`, register locally. |
| `build(toml)` | Compile to a `.so` plus debug metadata. |
| `wrap(toml)` | Emit `julia/<Module>.jl`. |
| `register(toml)` | Put a project in the local registry so `use` finds it. |
| `use("name")` | Build, wrap, and load a **registered** project (cached). |
| `list_registry()` / `unregister("name")` | Inspect / drop local entries. |
| `search("xml")` | Browse Hub config *names* (does not install or register). |
| `clean(toml)` | Remove `build/`, `julia/`, caches. |
| `info(toml)` | Print whether the library and wrapper exist. |
| `check_environment()` | Report which toolchains this machine has. |
| `scaffold_package("Name")` | Skeleton of a distributable Julia package. |
| `ingest(so; headers=…)` | Experimental. Wrap a prebuilt **C** `.so`. |

Every path takes a `replibuild.toml` (or a directory that contains one) and is
idempotent: unchanged inputs hit the cache and return.

## What you edit

`replibuild.toml` is the interface. Discovery writes the shape of the tree.
You add the things it cannot see:

- preprocessor macros and the headers they live in
- typed overloads for variadic functions
- which `char*` returns you own, and which free function releases them
- compile flags, `-D` defines, include dirs, link libraries
- paths that must not compile (tests, examples, CLIs)
- C++ template instantiations you actually want in the wrapper

Omit one of those and the build usually *succeeds* — the wrapper is just missing
the thing, or leaks, or wraps an empty struct. The symptom → key table is in
[Troubleshooting](troubleshooting.md).

## Requirements

- **Linux only.** ELF `.so`, DWARF, GNU `nm`.
- **Julia 1.10+** (developed on 1.12).
- **C libraries: nothing else.** Clang ships as a JLL; link/optimize/assemble
  run on Julia's own libLLVM.
- **C++ libraries:** system LLVM/MLIR 21+, CMake 3.20+, and `cd src/mlir && ./build.sh`
  once. See [Install](install.md).
