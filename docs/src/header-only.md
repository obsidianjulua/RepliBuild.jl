# Header-only libraries

RepliBuild wraps what ends up in the compiled library. Code in a header ends
up there only when some source file uses it. A library that is *only* headers
therefore starts with nothing to wrap, and a normal library can still have
part of its API missing because that part is inline.

The fix is always the same: get the header's code compiled into the library.
How depends on how the header delivers its code.

| The header has | Add | Section |
|:---------------|:----|:--------|
| An implementation behind a macro (the stb style) | `[wrap] shim_defines` | [An implementation section](#An-implementation-section) |
| `static inline` C functions | `[wrap.macros]` entries | [static inline functions](#static-inline-functions) |
| C `inline` functions that are not `static` | One declaration in a `.c` file | [C inline functions](#C-inline-functions) |
| C++ class templates | `[types] templates` | [C++ templates](#C-templates) |
| C++ constructors and methods written inside a class | `-fstandalone-debug` | [C++ classes](#C-classes) |
| C++ free functions, operators, function templates | A keep file | [C++: a keep file](#C:-a-keep-file) |

## Starting a header-only project

On a folder that holds only headers, `discover` reports `found: 0 sources` and
`build` stops with `Compilation produced no library`. With no source files to
look at, `discover` also guesses `language = "c"`, even for C++ headers.

So set the language and flags yourself:

```toml
[project]
name = "hol"

[compile]
flags        = ["-O2", "-fPIC", "-std=c++17"]     # drop -std for C
include_dirs = ["."]

[link]
enable_lto         = false
optimization_level = "2"

[wrap]
language = "cpp"                                   # or "c"
```

Then add one of the recipes below. If the recipe is a file of your own, add
the file first and `discover` gets the language right by itself.

## An implementation section

A single-header library in the stb style compiles its implementation only in a
file that defines a macro before including it. Name the header and the macro:

```toml
[wrap]
shim_headers = ["stb_image.h"]
shim_defines = ["STB_IMAGE_IMPLEMENTATION"]
```

RepliBuild writes a small source file that defines the macro and includes the
header, and compiles it into the library. No file of your own is needed.

Each `shim_defines` entry is `NAME`, or `NAME=VALUE`.

Do not put the macro in `[compile] flags` instead. A flag reaches every file
in the build. As soon as two files include the header, the implementation is
compiled twice and the link fails; in a C build the error is `In-process
libLLVM link failed for C source`, which does not say why.

A file of your own does the same job, if you prefer one:

```c
/* hol_impl.c */
#define HOL_IMPLEMENTATION
#include "hol.h"
```

This brings in the functions of the implementation section. The header's
`static inline` functions and macros are still missing, and nothing reports
them: see the next section.

## static inline functions

A `static inline` function never becomes part of the library. Declare it under
`[wrap.macros]`, exactly as you would a function-like macro
([Macros](toml.md#Macros)):

```toml
[wrap]
shim_headers = ["hol.h"]

[wrap.macros.hol_dot]                 # static inline float hol_dot(HolVec a, HolVec b)
ret  = "float"
args = ["HolVec", "HolVec"]
```

`hol_dot(a, b)` then exists in Julia. Structs passed by value work. A library
that is nothing but `static inline` functions needs no source file at all:
list them and build.

This is for C. A C++ name with a namespace cannot be an entry.

## C inline functions

A C function declared `inline` without `static` is compiled only when one
source file asks for it, by declaring it again without `inline`:

```c
/* holi_emit.c */
#include "holi.h"
int holi_triple(int v);     /* this file now provides holi_triple */
```

Put that file in the project and build.

## C++ templates

A class template is compiled only for the type arguments someone uses. List
the ones you want:

```toml
[types]
templates        = ["hol::Box<double>", "std::vector<int>"]
template_headers = ["hol.hpp"]
```

Every member of each listed instantiation is compiled, constructors included.
For a header-only template library this can be the whole recipe.
[C++ templates and containers](toml.md#C-templates-and-containers) has the
rules. A function template cannot be listed here; use a keep file.

## C++ classes

Constructors and member functions written inside a class body are handled for
you. The build compiles in every public one that the library does not already
contain, and also the calls that leave default arguments out. It needs the
class to be fully described in the build, which one flag ensures:

```toml
[compile]
flags = ["-O2", "-fPIC", "-std=c++17", "-fstandalone-debug"]
```

`wrap` reports what it added and what it skipped: private, deleted and
template members, and abstract classes, are skipped.

Two things are not covered:

- A class that **no** source file in the build mentions. Name it in
  [`[types] records`](toml.md#Types-declared-only-in-headers).
- Free functions and operators outside any class. Use a keep file.

## C++: a keep file

A keep file is a source file of your own whose only job is to use things from
the headers so that they are compiled. Nothing calls it.

```cpp
// hol_keep.cpp
#include "hol.hpp"

// A free function or operator: taking its address compiles it.
[[gnu::used]] static auto keep_twice = &hol::twice;

// A function template: instantiate it explicitly.
template double hol::biggest<double>(double, double);
```

`[[gnu::used]]` stops the optimizer from removing the reference.

The functions appear in the wrapper under their own names, like any other C++
function: `hol_twice`, `hol_biggest_double`.

When the library comes from git, the keep file is yours and lives beside the
TOML. List it in `source_files`; it is compiled along with the dependency:

```toml
[dependencies.hol]
type    = "git"
url     = "https://github.com/example/hol.git"
tag     = "v1.0"
commit  = "…"                                  # the full commit
exclude = ["tests/", "examples/"]

[compile]
flags        = ["-O2", "-fPIC", "-std=c++17", "-fstandalone-debug"]
source_files = ["shim/hol_keep.cpp"]

[wrap]
language = "cpp"
```

A header-only repository's tests and examples are often its only `.cpp` files,
and they would all be compiled, so exclude them. The keep file includes the
library as a user would (`#include "hol/hol.hpp"`): the dependency's top
folder, `include/` and `src/` are already searched.

## Boost and other large template libraries

**Wrap one Boost library at a time.** `boostorg/boost` is a repository of
about 170 submodules, and a plain clone of it is nearly empty. List the
submodules your library needs
([Git submodules](dependencies.md#Git-submodules)): its own, plus the
header-only ones it includes.

Then two things from the library's own `CMakeLists.txt`:

- Its private defines (`BOOST_FILESYSTEM_SOURCE`, and options such as
  `BOOST_FILESYSTEM_NO_CXX20_ATOMIC_REF` for a `-std=c++17` build) go in
  `[compile] flags`.
- The submodules contain `.cpp` files that are not library code:
  configuration checks (`config/`, `checks/`), generators (`tools/`),
  benchmarks. `exclude` them by their `libs/<module>/…` path.

A Boost library that can stand alone is simpler as its own dependency:
`boostorg/math` with `-DBOOST_MATH_STANDALONE`.

Most of a Boost library's public API is inline, so it needs a keep file. Under
the library's own `*_SOURCE` define the headers hide that inline API; `#undef`
the define in the keep file before the first include.

**Template names get long.** A template instantiation's Julia name includes
its arguments, defaults too. Boost.Math's `pdf` comes out over 500 characters.
Give such functions names of your own, and let one pattern cover a family:

```toml
[wrap.rename]
"boost_math_pdf_*" = "pdf"
```

Find the candidates with
`filter(n -> occursin("pdf", string(n)), names(M; all = true))`.

A large template library also brings thousands of internal types with it. If
the module does not load because of them, use
[`surface_types_only`](toml.md#Leaving-things-out).

## Growing a wrapper from the REPL

The recipes that need no file of your own can be tried in one Julia session,
one entry at a time:

```julia
using RepliBuild
ENV["REPLIBUILD_JULIA_ENV"] = "none"        # keep it out of your Julia environment
toml = RepliBuild.discover("hol")           # a folder holding hol.h
# edit replibuild.toml: shim_headers, shim_defines …
RepliBuild.register(toml)
H = RepliBuild.use("hol")
# add a [wrap.macros] entry, then in the same session:
H = RepliBuild.use("hol")                   # rebuilt and reloaded, as a new module
```

Each `use` after an edit rebuilds in a few seconds and returns a fresh module
([Load it anywhere](use.md)).

## Checking what was compiled

The source files RepliBuild writes for these recipes are in
`.replibuild_cache/` inside the project: `replibuild_shims.c` (or `.cpp`) for
`[wrap.macros]` and `shim_defines`, and `replibuild_templates.cpp` for
`[types] templates`. Open them to see exactly what was compiled.
