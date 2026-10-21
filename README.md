# RepliBuild.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://obsidianjulua.github.io/RepliBuild.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://obsidianjulua.github.io/RepliBuild.jl/dev/)
[![Julia 1.12 | 1.13](https://img.shields.io/badge/julia-1.12%20%7C%201.13-9558B2?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Point RepliBuild at C or C++ source. It compiles the library, reads the ABI the
compiler actually emitted, and writes a Julia module you can call. You drive it
with a few verbs and a `replibuild.toml`. You do not write `ccall`s, and you do
not maintain generated bindings.

**[User manual](https://obsidianjulua.github.io/RepliBuild.jl/stable/)** — install,
wrap a library, edit the TOML, call the result.

## First wrap

```julia
using RepliBuild

toml = RepliBuild.discover("path/to/project")   # scan sources, write replibuild.toml
# edit the TOML: flags, excludes, macros, varargs, ownership
RepliBuild.build(toml)                          # clang → .so
RepliBuild.wrap(toml)                           # .so → julia/MyProject.jl

include("path/to/project/julia/MyProject.jl")
using .MyProject
```

Or `RepliBuild.discover("path/to/project", build=true, wrap=true)` in one call.
Then open the TOML — discovery writes the shape of the tree; you add the things
it cannot see.

`discover` registers the project locally, so later `RepliBuild.use("myproject")`
reloads it without `include`. A fresh install has an empty registry:
`use("cjson")` is not a first-run command.

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
templates = ["std::vector<int>"]    # C++: force DWARF for these instantiations
```

The [configuration page](https://obsidianjulua.github.io/RepliBuild.jl/stable/config/)
lists every key, what discovery cannot infer, and a symptom → key index.

## Requirements

- **Linux or Windows.** Linux is ELF `.so`; Windows is PE `.dll` under
  `x86_64-w64-windows-gnu` (mingw, MSYS2 CLANG64) — not MSVC. Both read DWARF
  through GNU binutils. macOS is refused at load: the AAPCS64 ABI classifier is
  not built, so arm64 Mach-O has no struct-passing rules to apply.
- **Julia 1.12 or 1.13**, the versions CI tests. `Project.toml` still admits
  1.10 and 1.11, which nothing tests.
- **C libraries:** on Linux, GNU binutils (`readelf`, `nm`), which almost every
  distribution has. Clang ships as a JLL, and link/optimize/assemble run on
  Julia's own libLLVM. On Windows, MSYS2 CLANG64 and GNU `objdump` (below).
- **C++ libraries:** system LLVM/MLIR 21+, and the JLCS dialect library built
  against it. On Linux `Pkg.build("RepliBuild")` builds it when the LLVM/MLIR
  tools are on `PATH`; on Windows you run `build.sh` from an MSYS2 CLANG64
  shell. [Install](https://obsidianjulua.github.io/RepliBuild.jl/stable/install/)
  has the steps.

`RepliBuild.check_environment()` reports which of those this machine has.

### Windows

Use the **MSYS2 CLANG64** environment specifically — it is the
`x86_64-w64-windows-gnu` target that matches Julia's own mingw build, and the
only MSYS2 repo shipping MLIR. In a CLANG64 shell:

```
pacman -S --needed mingw-w64-clang-x86_64-toolchain mingw-w64-clang-x86_64-mlir \
                   mingw-w64-clang-x86_64-cmake mingw-w64-clang-x86_64-ninja git \
                   mingw-w64-x86_64-binutils
```

(`mingw-w64-x86_64-binutils` supplies GNU `objdump` for DWARF extraction; CLANG64's
own "binutils" is llvm-objdump, which is rejected.) Then add
`C:\msys64\clang64\bin` to the Windows `PATH`.

The C bucket compiles through the Clang JLL, which carries no headers or CRT of
its own, so it borrows the CLANG64 sysroot. That is found automatically;
`REPLIBUILD_C_SYSROOT` overrides it.

[Install](https://obsidianjulua.github.io/RepliBuild.jl/stable/install/#Windows)
has the full setup. [WINDOWS_PORT.md](docs/updates/WINDOWS_PORT.md) records what
the port found.

## Documentation

- [User manual](https://obsidianjulua.github.io/RepliBuild.jl/stable/) — the complete index
- [Wrap a library](https://obsidianjulua.github.io/RepliBuild.jl/stable/guide/) ·
  [Edit the TOML](https://obsidianjulua.github.io/RepliBuild.jl/stable/config/) ·
  [Call a wrapper](https://obsidianjulua.github.io/RepliBuild.jl/stable/calling/) ·
  [API](https://obsidianjulua.github.io/RepliBuild.jl/stable/api/)
- [Developer](https://obsidianjulua.github.io/RepliBuild.jl/stable/developer/) — architecture (JLCS, inheritance ABI, internals)
- [CHANGELOG](CHANGELOG.md)

## License

MIT
