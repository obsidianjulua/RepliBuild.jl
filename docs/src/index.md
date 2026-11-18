# RepliBuild

```@meta
CurrentModule = RepliBuild
```

RepliBuild turns the source of a C or C++ library into a Julia module that
calls it. It fetches the source, compiles the library, works out every
function, struct, enum and class from the result, and writes the module.

**It does everything except what has to be stated, and what has to be stated
goes in one file, `replibuild.toml`.** That is the design: where the source
is, how the library is meant to be built, and the few things no compiler can
know. Using RepliBuild well is knowing what to put in that file.

```julia
using RepliBuild

toml = RepliBuild.discover("path/to/mylib")     # writes replibuild.toml
RepliBuild.build(toml)                          # compiles the library
RepliBuild.wrap(toml)                           # writes julia/Mylib.jl

include("path/to/mylib/julia/Mylib.jl")
using .Mylib
```

It builds from source, on your machine. It does not replace Julia's prebuilt
binary packages (JLLs); it is for a library you build yourself, at the version
and with the flags you choose.

## The manual

There are three things to do, and the manual is in that order.

**1. [Set up](install.md).** Install the package and run
`RepliBuild.check_environment()`. RepliBuild brings some of its tools and
takes the rest from your system; the page says which. C libraries need almost
nothing installed. C++ libraries need `clang++`.

**2. Write the recipe.** `replibuild.toml` is the only file you write.

| | |
|:--|:--|
| [Wrap your first library](guide.md) | The whole process once, on a small example. Start here. |
| [What goes in the TOML](toml.md) | **The main page.** What to look for in your library, and the entry each thing needs. |
| [Every key](config.md) | The lookup table: every key, its default, what it does. |
| [Where the source comes from](dependencies.md) | How to feed it source: files in your project, a folder on disk, a git repository, a tarball, or several at once. Each behaves differently. |
| [Several libraries](libraries.md) | Many sources into one library, a CMake project of several libraries with shared types, or two wrappers that link each other. |
| [CMake projects](graph.md) | The `[graph]` section in detail. |
| [Header-only libraries](header-only.md) | Libraries, or parts of an API, that live in headers. |

**3. Use the wrapper.**

| | |
|:--|:--|
| [Call the wrapper](calling.md) | How C and C++ functions, structs, strings, objects and containers look in Julia. |
| [Load it anywhere](use.md) | `register` a finished wrapper and load it by name. |
| [Build a package on it](using-wrappers.md) | Ship a Julia package that contains a wrapper. |
| [Make it fast](performance.md) | What a call costs, C++ stubs compiled ahead of time, precompiling, and compiling a wrapper into a system image. |
| [Debug it](debugging.md) | Step from Julia into the library's own source, and through the generated C++ call stubs. |

When something goes wrong: [Troubleshooting](troubleshooting.md), which is
indexed by error message.

## What you get

The wrapper is a Julia module generated for one build of one library, and
checked against that build. It is more than a list of foreign calls:

- **It is correct by construction.** Every struct's layout, every signature and
  every enum value comes from the compiled library, not from reading headers.
  A struct Julia cannot lay out exactly is not guessed at. A function that
  cannot be called safely becomes one that raises, never one that corrupts
  memory.
- **It speaks C++.** Methods, virtual dispatch, constructors and destructors,
  overloads, default arguments, classes passed by value, `std::string`,
  containers, inheritance. A C++ exception arrives as a Julia exception.
  Objects can be owned by Julia's garbage collector.
- **It is fast.** A C function is called directly, at the cost of a
  hand-written call. C++ call stubs can be compiled ahead of time, the module
  precompiles inside a package, and the package can be compiled into a system
  image, so a program that calls C++ starts in about the time Julia does
  ([Make it fast](performance.md)).
- **It can be debugged, all the way down.** The library is built with its
  debug information, so a crash names the C or C++ source line and a debugger
  stops in the library's own code. The generated C++ call stubs have readable
  source too, and a debugger steps through them between your Julia call and
  the C++ method ([Debug it](debugging.md)).
- **It documents itself.** Every function has a docstring with its Julia
  signature and C types, and the module answers questions about sizes and
  offsets.

## What you put in the TOML

RepliBuild finds the library's functions and types by itself. The recipe holds
what it cannot find:

- where the source is: files in the project, a folder, a git repository at a
  commit, a tarball with a hash;
- which of those files are tests, examples or tools, and not the library;
- whether it is one library or several;
- the flags and defines the library's own build would pass;
- macros that are part of the API;
- which extra arguments a `printf`-style function should accept;
- which returned strings the caller must free, and with what;
- C++ templates you want to use, such as `std::vector<double>`;
- what to leave out, and what to rename.

A missing entry usually does not fail the build. It gives a wrapper that is
missing that thing. [What goes in the TOML](toml.md) is the checklist to go
through for each library.

## The commands

| Command | Does |
|:--------|:-----|
| `check_environment()` | Checks the tools on this machine and says how to install what is missing. |
| `discover(path)` | Scans a source folder and writes `replibuild.toml`. Keeps an existing one. |
| `build(toml)` | Compiles the library. |
| `wrap(toml)` | Writes the Julia module, `julia/<Name>.jl`. |
| `info(toml)` | Says whether the library and the module exist. |
| `clean(toml)` | Deletes everything built. |
| `register(toml)` | Makes a finished wrapper loadable by name in any session. |
| `use("name")` | Loads a registered wrapper, rebuilding it if the project changed. |
| `list_registry()`, `unregister("name")` | Lists and removes registered wrappers. |
| `search("xml")` | Lists ready-made recipes in RepliBuild-Hub. |
| `scaffold_package("Name")` | Creates a Julia package that builds a library when installed. |

Each takes the path of a `replibuild.toml`, or of the folder holding one.
Running a command again when nothing changed does no work.
[API](api.md) has the full signatures.

## Requirements

- **Linux** (x86-64 or AArch64) or **Windows** (x86-64, with MSYS2). macOS is
  not supported.
- **Julia 1.13 or 1.14.** For Julia 1.12, use the RepliBuild 4.5 releases.
- For C++ libraries, `clang++` and LLVM 21 or newer.

[Set up](install.md) has the details. This manual writes `.so` for a library
file; on Windows it is `.dll`.
