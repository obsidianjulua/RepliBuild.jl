# The recipe

`replibuild.toml` is the whole interface. Everything RepliBuild does to a
library — what it compiles, with which flags, from which pinned upstream, how
the result is named and trimmed and presented in Julia — is decided by that
file and by nothing else: there is no generator fork, no patched header, no
hand-written binding layer to maintain. This page explains what the file is
for and what can be done with it. [Edit the TOML](config.md) is the key-by-key
reference.

## What it is for

RepliBuild is for one situation: **you have the source of a C or C++ library,
and you want a Julia module that calls it.** It compiles that source itself,
with its own toolchain and one consistent flag set, then reads the ABI the
compiler actually emitted (the DWARF debug information and the symbol table of
the library it just linked) and generates the Julia module from that.

So most of what a binding needs is never written by anyone:

| Who knows it | What | Where it lives |
|:-------------|:-----|:---------------|
| The compiler | every type's layout, every signature, enum values, which calls throw, how each record crosses the calling convention | DWARF, read at `build` — never in the recipe |
| The source tree | which files to compile, include roots, C or C++ | written by `discover()` into the recipe |
| You | what the compiler erased or never saw | the recipe keys you add |

The third row is the recipe's real content. A compiler erases macros, does not
know what you mean to pass to a variadic function, cannot say who frees a
returned `char*`, and generates nothing for a template nobody instantiated. A
repository also holds tests, examples and backends that are not the library,
and a build system supplies defines no header declares. Each of those is one
key. A finished recipe for a large library is usually a few dozen lines, and
every line in it is a fact about that library that could not be derived.

This is a developer tool that works from source on your machine. It does not
replace BinaryBuilder and JLL packages, which ship prebuilt binaries for every
platform. Where a recipe fits is a library you build yourself, at a version you
pin, with the flags you choose — or one no JLL covers.

## Three shapes of project

| The project is… | The recipe has | Page |
|:----------------|:---------------|:-----|
| One library: your files, a pinned upstream, or both | `[project]`, `[compile]`, `[dependencies]`, `[wrap]`, … | [Edit the TOML](config.md), [Dependencies](dependencies.md) |
| A CMake project of several libraries that link each other | a `[graph]` section | [Graph mode](graph.md) |
| A prebuilt C `.so` with debug info | an `[ingest]` section (experimental, C only) | [Wrap a library](guide.md#Ingest-(experimental,-C-only)) |

The presence of `[graph]` or `[ingest]` selects the mode; a recipe without
either builds one library.

## The loop

```julia
toml = RepliBuild.discover("path/to/project")   # write the recipe from the tree
RepliBuild.build(toml)                           # compile, link, read DWARF
RepliBuild.wrap(toml)                            # write julia/<Module>.jl
# read the wrapper, add what is missing to the recipe, build and wrap again
```

`discover` writes what a scan of the tree can see. A scan cannot see intent, so
`discover(force = true)` rescans and **keeps** the keys only you can write
(templates, macros, varargs, ownership, excludes, renames, dependencies, the
`[graph]` section, platform overlays — the full list is in
[Edit the TOML](config.md#1.-How-the-file-is-created-and-maintained)). Once you
have edited a recipe, keep it in version control and stop running `discover`
on it.

Every key takes effect at one of two steps:

| Takes effect at `build` (compiled into the library or its metadata) | Takes effect at `wrap` (re-wrap only, no recompile) |
|:--------------------------|:--------------------------|
| `[compile]` flags, defines, include dirs, `visibility`, `source_files` | `[wrap] exclude_symbols` |
| `[link]` libraries, dirs, optimisation level | `[wrap.rename]` |
| `[dependencies]` (what is cloned, pinned, fetched, excluded) | `[wrap.varargs]` |
| `[types] templates`, `template_headers` | `[wrap.cstring_owned]` |
| `[wrap.macros]`, `shim_headers`, `shim_defines` (the shim is compiled into the library) | `[wrap] surface_types_only`, `surface_types_extra`, `headers` |
| `[graph]` targets, options, build type | |

`build` is cached on a hash of the recipe, the sources and the toolchain, so
running it after any edit is cheap when nothing it reads changed.

## What a recipe can do

| You want to… | Key | Reference |
|:-------------|:----|:----------|
| Build from a pinned upstream git tag, verified by commit | `[dependencies.<n>]` `type = "git"`, `tag`, `commit` | [Dependencies](dependencies.md) |
| Build from a release tarball, verified by hash | `type = "archive"`, `url`, `sha256`, `subdir` | [Dependencies](dependencies.md#Archives) |
| Take a few libraries out of a superproject (Boost) | `submodules = [...]` | [Dependencies](dependencies.md#Submodules) |
| Leave tests, examples, backends out of the compile | `exclude` on the dependency | [§2.6](config.md#2.6-Excluding-what-is-not-the-library) |
| Supply the defines and include dirs the upstream build would | `[compile] flags`, `defines`, `include_dirs` | [§2.7](config.md#2.7-Include-roots), [§2.9](config.md#2.9-Defines-the-build-system-would-have-supplied) |
| Link `-lm`, a system library, or a library you built | `[link] link_libraries`, `link_dirs` | [§2.8](config.md#2.8-Link-libraries) |
| Call a macro, or a `static inline` function | `[wrap.macros.<NAME>]`, `shim_headers` | [§2.2](config.md#2.2-Preprocessor-macros) |
| Build a single-header (stb-style) library with no file of your own | `shim_defines = ["X_IMPLEMENTATION"]` | [§2.2](config.md#2.2-Preprocessor-macros) |
| Call a variadic function with arguments | `[wrap.varargs]` | [§2.3](config.md#2.3-Variadic-functions) |
| Free a returned `char*` with the library's deallocator | `[wrap.cstring_owned]` | [§2.4](config.md#2.4-char*-return-ownership) |
| Get `std::vector`, `std::map`, `std::string` containers | `[types] templates` | [§2.1](config.md#2.1-C-templates-and-the-STL-—-"STL-needs-STL-in-the-TOML") |
| Give a header-only struct its fields | `-fstandalone-debug` or `-Xclang -fno-use-ctor-homing` in flags | [§2.5](config.md#2.5-Header-only-aggregates-need-fstandalone-debug) |
| Narrow the API to what upstream exports | `[compile] visibility = "hidden"` (probe first) | [§2.12](config.md#2.12-Export-visibility-—-what-the-wrapper's-API-*is*) |
| Drop symbols that are compiled in but not API | `[wrap] exclude_symbols` | [§2.13](config.md#2.13-exclude_symbols-—-trimming-a-surface-without-touching-visibility) |
| Emit only the types the API reaches | `[wrap] surface_types_only` | [§2.14](config.md#2.14-surface_types_only-—-the-same-question,-asked-about-types) |
| Give a function (a 600-character template name) a name of your own | `[wrap.rename]` | [§2.17](config.md#2.17-rename-—-a-name-of-your-own) |
| Keep one recipe right on Linux and Windows | `[platform.linux]`, `[platform.windows]` | [§2.15](config.md#2.15-One-config,-two-hosts-—-[platform.host]) |
| Ship Tier-2 thunks precompiled instead of JIT'd | `[compile] aot_thunks = true` | [§3](config.md#[compile]) |
| Wrap every library of a CMake project as one Julia module | `[graph]` | [Graph mode](graph.md) |

What a recipe does **not** hold is as deliberate. You never write a struct
layout, a signature, a calling convention, a dispatch tier or a type mapping:
those are read from what the compiler did, every time, so they cannot drift
from the binary. If a wrapper is wrong about one of them, that is a RepliBuild
bug to report, not a key to add.

## An annotated recipe

A C library from git, with each kind of residue:

```toml
[project]
name = "lua"                         # → liblua.so, module Lua, julia/Lua.jl
root = "."

[dependencies.lua]                   # the source: cloned, pinned, walked
type    = "git"
url     = "https://github.com/lua/lua.git"
tag     = "v5.4.7"
commit  = "1ab3208a1fceb12fca8f24ba57d6e13c5bff15e3"   # a moved tag is a hard error
exclude = ["onelua.c", "lua.c", "luac.c", "ltests.c"]   # the interpreter, not the library

[compile]
flags = ["-O1", "-fPIC", "-DLUA_USE_LINUX"]   # what lua's Makefile would pass

[link]
optimization_level = "2"
link_libraries     = ["m", "dl"]

[platform.windows.compile]           # the same recipe on Windows
flags = ["-O1"]
[platform.windows.link]
link_libraries = []

[wrap]
language     = "c"
shim_headers = ["lua.h", "lauxlib.h"]

[wrap.macros.lua_pop]                # a macro, compiled into a real function
ret  = "void"
args = ["lua_State*", "int"]

[wrap.varargs]                       # what you mean to pass to a variadic
proven_at       = "1ab3208a1fceb12fca8f24ba57d6e13c5bff15e3"
lua_pushfstring = [["Cstring"], ["Cint"], ["Cstring", "Cint"]]
```

Everything else about lua — 33 source files' worth of structs, enums,
signatures and calling conventions — comes from the compile.

## Recipes are shareable

A recipe names its upstream by URL and pin and holds no machine paths
(relative paths resolve against the project), so it builds the same library on
any machine with the toolchain. The
[RepliBuild-Hub](https://github.com/obsidianjulua/RepliBuild-Hub) is a
community collection of them — lua, sqlite, zlib, pcre2, imgui, box2d, fmt,
pugixml, llama.cpp, libwebp and more — and is the best place to see what real
recipes look like. `RepliBuild.search("xml")` lists its names; to use one,
copy its directory, then `register` it ([Registry](use.md)).
