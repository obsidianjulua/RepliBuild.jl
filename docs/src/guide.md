# Wrap a library

Three verbs, then an edit. Discovery writes a config that compiles. You tell it
the things it cannot see. Then you build and wrap.

```julia
using RepliBuild

toml = RepliBuild.discover("path/to/project")   # scan → replibuild.toml
# edit replibuild.toml  ← this is the work
RepliBuild.build(toml)                          # clang → .so
RepliBuild.wrap(toml)                           # .so → julia/MyProject.jl

include("path/to/project/julia/MyProject.jl")
using .MyProject
```

One-shot (still edit the TOML afterwards — discovery cannot invent macros):

```julia
RepliBuild.discover("path/to/project", build=true, wrap=true)
```

It builds and wraps the project where it lives, then registers it: the build is
vendored to `~/.replibuild/` and added to your Julia environment as a package,
so `RepliBuild.use("myproject")` and `using MyProject` load it from any session
([Registry](use.md)). On a project that already has a `replibuild.toml`, the same
call keeps that file.

`RepliBuild.clean(toml)` removes `build/`, `julia/`, and `.replibuild_cache/`
(git dependencies are fetched again on the next build).
`RepliBuild.clean(toml; deps=false)` keeps the clones under
`.replibuild_cache/deps/` and removes the rest of the cache, so a rebuild from
cold checks each dependency's pin instead of fetching it again.
`RepliBuild.build(toml; clean=true)` does that first and then builds.
`RepliBuild.info(toml)` prints whether the `.so` and wrapper exist.

## 1. Discover

```julia
RepliBuild.discover()                    # cwd
RepliBuild.discover("path/to/project")
RepliBuild.discover("path/to/project", force=true)   # rescan
```

Discovery walks the tree, follows `#include`s, sets `wrap.language` from the
file extensions, and writes `replibuild.toml`. Without `force=true`, an existing
`replibuild.toml` is left as it is. Discovery alone registers nothing: register
the project once its wrapper is ready (step 4).

`force=true` regenerates the file but **keeps** the keys you are expected to
hand-edit: `[types].templates` / `template_headers`, `[wrap].varargs` /
`macros` / `shim_headers` / `cstring_owned` / `tier1`, `[link].promote_statics`.
A `preserved: …` line in the output reports what carried over.

Everything else is rewritten from the scan — including `[compile] flags`,
`include_dirs`, and `[dependencies]`. Keep a hand-written config in version
control. Do not rely on `force=true` to round-trip it.

## 2. Edit the TOML

Open `replibuild.toml`. This is the whole interface. The keys you almost always
touch:

| You need… | You set… |
|-----------|----------|
| Tests/examples/CLIs out of the build | `exclude` on the dependency |
| Upstream's `-D` / `-O` / `-std=` | `[compile] flags` or `[compile.defines]` |
| Headers not in `<root>`, `<root>/include`, `<root>/src` | `[compile] include_dirs` |
| `-lm`, `-lpthread`, `-ldl` | `[link] link_libraries` |
| Macro API (`lua_pop`, `SQLITE_OK`, `deflateInit`) | `[wrap.macros.*]` + `shim_headers` |
| `printf`-style extra arguments | `[wrap.varargs]` |
| malloc'd `char*` returns | `[wrap.cstring_owned]` |
| Header-only "def" structs with no fields in the wrapper | `-fstandalone-debug` in flags |
| C++ `std::vector<int>` (or any template) in the wrapper | `[types] templates` + `template_headers` |
| A git pin that survives a moved tag | `commit` (full 40-hex SHA) |

The complete list, every default, and worked configs:
[Edit the TOML](config.md). Symptom → key: [Troubleshooting](troubleshooting.md).

A C library from git looks like this:

```toml
[project]
name = "cjson"

[dependencies.cjson]
type    = "git"
url     = "https://github.com/DaveGamble/cJSON.git"
tag     = "v1.7.18"
commit  = "acc76239bee01d8e9c858ae2cab296704e52d916"
exclude = ["test", "fuzzing"]

[compile]
flags = ["-O2", "-fPIC"]

[link]
enable_lto         = false
optimization_level = "2"

[wrap]
language     = "c"
shim_headers = ["cJSON.h"]

[wrap.cstring_owned]
cJSON_Print = "cJSON_free"

[cache]
enabled = true
```

Leave `[link] enable_lto = false`. Leave `[wrap.tier1] enable` off unless you
are working on that experiment. Hub configs do the same.

## 3. Build and wrap

```julia
RepliBuild.build(toml)     # .so lands in julia/ next to compilation_metadata.json
RepliBuild.wrap(toml)      # julia/MyProject.jl  (module name = CamelCase of [project].name)
```

Load it:

```julia
include("path/to/project/julia/MyProject.jl")
using .MyProject
```

The `.so` and the `.jl` travel as a unit. Do not edit the generated module —
regenerate with `wrap()`, put Julia-idiomatic API in your own code. See
[Call a wrapper](calling.md) and [Ship a package](using-wrappers.md).

Everything so far stays in the project: `build/`, `julia/` and
`.replibuild_cache/` under its root.

## 4. Register

```julia
RepliBuild.register(toml)
```

Once the wrapper does what you want, `register` makes it loadable from any
session. It builds and wraps the project in place (a cache hit when nothing
changed), vendors the build to `~/.replibuild/builds/<hash>/`, and adds a Julia
package around it to your Julia environment. Then:

```julia
M = RepliBuild.use("myproject")   # any session
using MyProject                   # any session, from the Julia environment
```

`use` follows later edits to the project by rebuilding it in place. See
[Registry](use.md).

## C vs C++

```toml
[wrap]
language = "c"     # JLL clang, no extra toolchain
language = "cpp"   # system clang++, needs the C++ install
```

C++ needs the [Install](install.md) toolchain.

`discover()` sets this from the scanned extensions. Setting it wrong gives you
a build that works and a wrapper that is missing the reason you picked the
other language.

C++ extras you will edit:

```toml
[compile]
flags = ["-O2", "-fPIC", "-std=c++17", "-fstandalone-debug"]

[types]
templates        = ["std::vector<int>"]
template_headers = ["<vector>"]
```

A template that is never instantiated does not exist in the binary. Naming it
in `[types] templates` is how it gets into the wrapper.

## Ingest (experimental, C only)

When the source pipeline cannot reproduce the build (autotools, a cmake code
generator), build the `.so` yourself with `-g` and ingest it. No compilation;
DWARF extraction and wrap only. **C++ API surfaces are not supported** — wrap
C++, or ingest a C API variant.

```julia
toml = RepliBuild.ingest("/path/to/libfoo.so",
                         headers=["/path/to/include"],
                         name="foo", language=:c,
                         build=true, wrap=true)
```

That writes an `[ingest]` section. Its presence is the mode switch. Prefer
`discover`/`build`/`wrap` whenever the sources compile under one flag set.
