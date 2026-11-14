# RepliBuild

```@meta
CurrentModule = RepliBuild
```

Point RepliBuild at C or C++ source. It compiles the library, reads the ABI the
compiler actually emitted, and writes a Julia module you can call. You drive it
with a few verbs and a `replibuild.toml`. You do not write `ccall`s, and you do
not maintain generated bindings.

It wraps a library from its source, on your machine. It is not a replacement for
BinaryBuilder and JLL packages, which ship prebuilt binaries for every platform.

This site is the user manual. It has two halves, because using RepliBuild
is two jobs:

- **The recipe.** `replibuild.toml` is the whole interface: every fact about a
  library that the compiler cannot tell RepliBuild — where the source comes
  from and at which pin, the flags its build system would pass, its macros and
  variadic functions, who frees what, which symbols are not API, what a
  function should be called — is one key in that file, and nothing else is
  ever written by hand. [The recipe](toml.md) explains the model and maps
  everything a recipe can do.
- **The wrapper.** What `wrap` writes is an ordinary Julia module. [Call a
  wrapper](calling.md) is how C and C++ types, objects, overloads, strings and
  standard-library values look from Julia, and [Ship a package](using-wrappers.md)
  is how to build a Julia package on one.

## I want to…

| Goal | Page |
|:-----|:-----|
| Check the toolchain | [Install](install.md) |
| Wrap a C/C++ project, start to finish | [Wrap a library](guide.md) |
| Understand what the TOML is and what it can do | [The recipe](toml.md) |
| Fill in macros, varargs, ownership, flags, excludes, renames | [Edit the TOML](config.md) |
| Build from a pinned git tag, a tarball, or a Boost submodule set | [Dependencies](dependencies.md) |
| Wrap a CMake project of several libraries as one module | [Graph mode](graph.md) |
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
RepliBuild.build(toml)                          # clang → .so
RepliBuild.wrap(toml)                           # .so → julia/MyProject.jl

include("path/to/project/julia/MyProject.jl")
using .MyProject
```

Or all at once: `RepliBuild.discover("path/to/project", build=true, wrap=true)`.
Then see what the wrapper is missing, fill in what discovery cannot see, and
build and wrap again. That edit is the whole skill.
[Wrap a library](guide.md) walks it; [Edit the TOML](config.md) is the
reference.

The project builds where it lives. `RepliBuild.register(toml)` then vendors the
build to `~/.replibuild/` and adds a Julia package around it to your Julia
environment, so `RepliBuild.use("myproject")` (`[project].name`) or
`using MyProject` loads it in any session. The one-shot `discover` above ends by
registering. A fresh install has an empty registry — `use("cjson")` does not work
until you have registered that name yourself. See [Registry](use.md).

## The verbs

| Function | What it does |
|:---------|:---------------|
| `discover(path)` | Scan a source tree, write `replibuild.toml`. An existing one is kept unless `force=true`. With `build=true, wrap=true`, also build, wrap and `register`. |
| `build(toml)` | Compile to a `.so` plus debug metadata, in the project. |
| `wrap(toml)` | Emit `julia/<Module>.jl`, in the project. |
| `register(toml)` | Build and wrap in place, vendor the build to `~/.replibuild/`, and add it to the Julia environment as a package. |
| `use("name")` | Load a **registered** project: its vendored build, or a fresh one after an edit. |
| `list_registry()` / `unregister("name")` | Inspect / drop registry entries. |
| `search("xml")` | Browse Hub config *names* (does not install or register). |
| `clean(toml)` | Remove `build/`, `julia/`, caches. `deps=false` keeps the vendored git clones. |
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
- the pinned upstream the library comes from, and the parts of it that are not
  the library
- a short name for a function whose derived one is not usable

Omit one of those and the build usually *succeeds* — the wrapper is just missing
the thing, or leaks, or wraps an empty struct. [The recipe](toml.md) maps
every key to what it does; the symptom → key table is in
[Troubleshooting](troubleshooting.md).

## Requirements

- **Linux (x86-64 or AArch64) or Windows (x86-64).** Linux is ELF `.so`;
  Windows is PE `.dll` under `x86_64-w64-windows-gnu` (mingw, MSYS2 CLANG64),
  not MSVC. Both read DWARF through GNU binutils. macOS is refused at load — its
  Mach-O plumbing and Apple's arm64 calling convention are not built.
- **Julia 1.12 or 1.13**, the versions CI tests, and what `Project.toml`
  admits (`julia = "1.12"`).
- **C libraries:** on Linux, GNU binutils (`readelf`, `nm`), which almost every
  distribution has. Clang ships as a JLL; link/optimize/assemble run on Julia's
  own libLLVM. On Windows that JLL borrows the MSYS2 CLANG64 sysroot for its
  headers and CRT, and DWARF extraction needs GNU `objdump`.
- **C++ libraries:** system LLVM/MLIR 21+, CMake 3.20+, and the JLCS dialect
  library built against that LLVM. See [Install](install.md). Every machine
  that *runs* a C++ wrapper needs the LLVM/MLIR and the dialect too, not only
  the one that generated it ([Ship a package](using-wrappers.md#Deployment)).
- **git**, for a `[dependencies]` entry of `type = "git"`.

This manual writes `.so`; on Windows read `.dll`.
