# RepliBuild.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://obsidianjulua.github.io/RepliBuild.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://obsidianjulua.github.io/RepliBuild.jl/dev/)
[![Julia 1.13 | 1.14](https://img.shields.io/badge/julia-1.13%20%7C%201.14-9558B2?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Point RepliBuild at C or C++ source, or at a Rust crate. It compiles the
library, reads the ABI the compiler actually emitted, and writes a Julia module
you can call. You drive it with a few verbs and a `replibuild.toml`. You do not
write `ccall`s, and you do not maintain generated bindings.

It wraps a library from its source, on your machine. It is not a replacement for
BinaryBuilder and JLL packages, which ship prebuilt binaries for every platform.

**[User manual](https://obsidianjulua.github.io/RepliBuild.jl/stable/)**: set up,
write the recipe, use the wrapper.

## First wrap

```julia
using RepliBuild

toml = RepliBuild.discover("path/to/project")   # scan sources, write replibuild.toml
RepliBuild.build(toml)                          # clang → .so
RepliBuild.wrap(toml)                           # .so → julia/MyProject.jl

include("path/to/project/julia/MyProject.jl")
using .MyProject
```

Or `RepliBuild.discover("path/to/project", build=true, wrap=true)` in one call.
Then see what the wrapper is missing, add it to the TOML (macros, variadic
overloads, `char*` ownership, flags, excludes) and build and wrap again:
discovery writes the shape of the tree; you add the things it cannot see.

The project builds where it lives. When the wrapper is ready,
`RepliBuild.register(toml)` vendors that build to `~/.replibuild/` and adds a
Julia package around it to your Julia environment, so `RepliBuild.use("myproject")`
or `using MyProject` loads it in any session; the one-shot `discover` above ends
by registering. A fresh install has an empty registry: `use("cjson")` is not a
first-run command.

## What you get

The wrapper is generated for one build of one library and checked against it.
Layouts, signatures and enum values come from the compiled library. C
functions are called directly, at the cost of a hand-written call. C++ works
in full: methods, virtual dispatch, overloads, default arguments, classes by
value, `std::string`, containers, exceptions. C++ call stubs can be compiled
ahead of time, the module precompiles in a package, and the package can go
into a system image
([Make it fast](https://obsidianjulua.github.io/RepliBuild.jl/stable/performance/)).
And it can be debugged all the way down: a debugger stops in the library's own
source and steps through the generated C++ call stubs
([Debug it](https://obsidianjulua.github.io/RepliBuild.jl/stable/debugging/)).

A Rust crate is wrapped as it is, with no `extern "C"` and no `#[repr(C)]`:
cargo builds it, and the module has its public functions, its structs and
enums, `String` and `Vec` as Julia's own, `Option` as a value or `nothing`,
and a `Result`'s error or a panic as a Julia exception
([Rust crates](https://obsidianjulua.github.io/RepliBuild.jl/stable/rust/)).

## What you edit

`replibuild.toml` is the interface. Typical entries:

```toml
[project]
name = "cjson"

[dependencies.cjson]
type    = "git"
url     = "https://github.com/DaveGamble/cJSON.git"
tag     = "v1.7.18"
exclude = ["test", "fuzzing"]

[compile]
flags = ["-O2", "-fPIC"]

[link]
enable_lto = false

[wrap]
language     = "c"
shim_headers = ["cJSON.h"]

[wrap.cstring_owned]
cJSON_Print = "cJSON_free"

[wrap.macros.CJSON_VERSION_MAJOR]
ret = "int"

[wrap.varargs]
# Julia types, variadic args only, list-of-lists
# fmt_fn = [["Cint"], ["Cstring", "Cint"]]

[types]
templates = ["std::vector<int>"]    # C++: template types to compile in
```

[What goes in the TOML](https://obsidianjulua.github.io/RepliBuild.jl/stable/toml/)
is the checklist of what to look for in a library and the entry each thing
needs; [Every key](https://obsidianjulua.github.io/RepliBuild.jl/stable/config/)
is the reference.

A CMake project that builds several libraries is one recipe too: a `[graph]`
section takes the build from CMake, builds every library, and wraps them as one
Julia module with a submodule per library
([CMake projects](https://obsidianjulua.github.io/RepliBuild.jl/stable/graph/)).

## Requirements

- **Linux (x86-64 or AArch64) or Windows (x86-64, with MSYS2 CLANG64).** macOS
  is not supported.
- **Julia 1.13 or 1.14**, the versions CI tests. Julia 1.12 is served by the
  4.5 line, whose last release is 4.5.1.
- **C libraries:** on Linux, nothing but the C library's development files
  (`libc6-dev`, `glibc-devel`), which a machine with `gcc` already has. The
  C compiler, the linker, `readelf`, `nm`, `git`, `cmake` and `ninja` come with
  the package; the system's copies are used when those cannot start, or first
  with `REPLIBUILD_TOOLS=system`.
- **C++ libraries:** `clang++` and LLVM 21 or newer on `PATH`. RepliBuild's
  C++ support library, `libJLCS`, is downloaded ready-made on Linux and built
  by `Pkg.build("RepliBuild")` on Windows.
- **Rust crates:** `cargo` and `rustc`, from [rustup](https://rustup.rs) or
  your distribution. Linux only so far; Windows has not been run.

`RepliBuild.check_environment()` checks all of it and prints the command that
installs whatever is missing.
[Set up](https://obsidianjulua.github.io/RepliBuild.jl/stable/install/) has the
steps for each system, including
[Windows](https://obsidianjulua.github.io/RepliBuild.jl/stable/install/#Windows).

A machine that only runs a wrapper needs less than one that builds it:
[Deployment](https://obsidianjulua.github.io/RepliBuild.jl/stable/using-wrappers/#Deployment).

The 4.5 releases use the system's `readelf`, `nm` and `git`, and build
`libJLCS` against a system LLVM/MLIR; the Set up page says how.

## Documentation

The [user manual](https://obsidianjulua.github.io/RepliBuild.jl/stable/) follows
the three things you do:

1. [Set up](https://obsidianjulua.github.io/RepliBuild.jl/stable/install/)
2. Write the recipe:
   [Wrap your first library](https://obsidianjulua.github.io/RepliBuild.jl/stable/guide/) ·
   [What goes in the TOML](https://obsidianjulua.github.io/RepliBuild.jl/stable/toml/) ·
   [Every key](https://obsidianjulua.github.io/RepliBuild.jl/stable/config/) ·
   [Where the source comes from](https://obsidianjulua.github.io/RepliBuild.jl/stable/dependencies/) ·
   [Several libraries](https://obsidianjulua.github.io/RepliBuild.jl/stable/libraries/) ·
   [CMake projects](https://obsidianjulua.github.io/RepliBuild.jl/stable/graph/) ·
   [Header-only libraries](https://obsidianjulua.github.io/RepliBuild.jl/stable/header-only/)
3. Use the wrapper:
   [Call the wrapper](https://obsidianjulua.github.io/RepliBuild.jl/stable/calling/) ·
   [Load it anywhere](https://obsidianjulua.github.io/RepliBuild.jl/stable/use/) ·
   [Build a package on it](https://obsidianjulua.github.io/RepliBuild.jl/stable/using-wrappers/) ·
   [Make it fast](https://obsidianjulua.github.io/RepliBuild.jl/stable/performance/) ·
   [Debug it](https://obsidianjulua.github.io/RepliBuild.jl/stable/debugging/)

Also: [Troubleshooting](https://obsidianjulua.github.io/RepliBuild.jl/stable/troubleshooting/) ·
[API](https://obsidianjulua.github.io/RepliBuild.jl/stable/api/) ·
[Internals](https://obsidianjulua.github.io/RepliBuild.jl/stable/developer/), for contributors ·
[CHANGELOG](CHANGELOG.md)

## License

MIT
