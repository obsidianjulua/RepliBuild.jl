# RepliBuild.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://obsidianjulua.github.io/RepliBuild.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://obsidianjulua.github.io/RepliBuild.jl/dev/)
[![Julia 1.10+](https://img.shields.io/badge/julia-1.10+-9558B2?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Point RepliBuild at C or C++ source. It compiles the library, reads the ABI the
compiler actually emitted, and writes a Julia module you can call. You drive it
with a few verbs and a `replibuild.toml`. You do not write `ccall`s, and you do
not maintain generated bindings.

**[User manual](https://obsidianjulua.github.io/RepliBuild.jl/stable/)** — install,
use a Hub package, wrap your own library, edit the TOML, call the result.

## 60 seconds

```julia
using RepliBuild

C = RepliBuild.use("cjson")        # Hub config → build → wrap → load (cached)

doc = C.cJSON_Parse("""{"answer": 42}""")
C.cJSON_GetNumberValue(C.cJSON_GetObjectItem(doc, "answer"))   # 42.0
C.cJSON_Print(doc)                 # String; the malloc'd C buffer is freed
C.cJSON_Delete(doc)
```

The [RepliBuild Hub](https://github.com/obsidianjulua/RepliBuild-Hub) carries
configs for lua, sqlite, zlib, box2d, pugixml, curl, and more —
`RepliBuild.search("json")` to browse. Builds are cached at `~/.replibuild/`.

## Wrap your own

```julia
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

Hub packages skip `include`: `use("name")` returns the loaded module.

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

- **Linux only.** ELF `.so`, DWARF, GNU `nm`.
- **Julia 1.10+** (developed on 1.12).
- **C libraries: nothing else.** Clang ships as a JLL.
- **C++ libraries:** system LLVM/MLIR 21+, then `cd src/mlir && ./build.sh`.

`RepliBuild.check_environment()` reports which of those this machine has.

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
