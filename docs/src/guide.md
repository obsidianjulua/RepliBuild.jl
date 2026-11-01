# Wrap a library

Three verbs, a look, then an edit. Discovery writes a config that compiles.
The first wrapper shows you what it could not see. You add that to the TOML,
then build and wrap again.

```julia
using RepliBuild

toml = RepliBuild.discover("path/to/project")   # scan → replibuild.toml
RepliBuild.build(toml)                          # clang → .so
RepliBuild.wrap(toml)                           # .so → julia/MyProject.jl

include("path/to/project/julia/MyProject.jl")
using .MyProject
# look at what you got, edit replibuild.toml  ← this is the work
# then build and wrap again
```

One-shot (still edit the TOML afterwards — discovery cannot invent macros):

```julia
RepliBuild.discover("path/to/project", build=true, wrap=true)
```

It builds and wraps the project where it lives, then registers it: the build is
vendored to `~/.replibuild/` and added to your Julia environment as a package,
so `RepliBuild.use("myproject")` and `using MyProject` load it from any session
([Registry](use.md)). On a project that already has a `replibuild.toml`, the same
call keeps that file. `wrap=true` needs `build=true`; without it, the call only
discovers.

Registering adds a package to your default Julia environment (`@v1.x`). For a
throwaway experiment, set `ENV["REPLIBUILD_JULIA_ENV"] = "none"` first, or call
`discover`, `build` and `wrap` separately: they register nothing.

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
the project once its wrapper is ready (step 5).

The file it writes compiles, and no more:

- **It is a debug build.** `[compile] flags` has no `-O` (`["-fPIC"]` for C,
  `["-std=c++17", "-fPIC"]` for C++) and `[link] optimization_level` is `"0"`.
  For an optimized library add `-O2` to the flags and set
  `optimization_level = "2"`, as the Hub configs do.
- **Some of its sections do nothing.** `[discovery]`, `[workflow]`,
  `[paths] source`/`include`/`cache`, `[wrap] style` and `[binary]
  strip_symbols` are read and not acted on. Leave them or delete them
  ([Edit the TOML](config.md#5.-What-fails-loudly,-what-warns,-what-is-ignored)).

`force=true` regenerates the file but **keeps** the keys you are expected to
hand-edit: `[types] templates`/`template_headers`; `[wrap] varargs`/`macros`/
`shim_headers`/`shim_defines`/`cstring_owned`/`exclude_symbols`/`surface_types_only`/
`surface_types_extra`/`headers`/`tier1`; `[link] promote_statics`;
`[compile] visibility`; and the `[platform.<host>]` tables. A `preserved: …`
line in the output reports what carried over.

Everything else is rewritten from the scan — including `[compile] flags`,
`include_dirs`, and `[dependencies]`. Keep a hand-written config in version
control. Do not rely on `force=true` to round-trip it.

## 2. Build and wrap

```julia
RepliBuild.build(toml)     # .so lands in julia/ next to compilation_metadata.json
RepliBuild.wrap(toml)      # julia/MyProject.jl  (module name = CamelCase of [project].name)
```

Load it:

```julia
include("path/to/project/julia/MyProject.jl")
using .MyProject
```

The first build and wrap in a Julia session are the slow ones, because Julia
compiles RepliBuild's own code on first use: on a one-file library the first
`wrap` took about ten seconds, and the next ones about one.

The `.so` and the `.jl` travel as a unit. Do not edit the generated module —
regenerate with `wrap()`, put Julia-idiomatic API in your own code. See
[Call a wrapper](calling.md) and [Ship a package](using-wrappers.md).

Everything so far stays in the project: `build/`, `julia/` and
`.replibuild_cache/` under its root.

## 3. Look at what you got

A build that succeeds is not a complete wrapper. Before you edit, look:

- **The output.** `dwarf: N functions, M types` is what DWARF gave the
  generator. Warnings name what it could not do: a struct that became a byte
  blob, a `[wrap.varargs]` table without `proven_at`, names withheld from
  `export`.
- **The names.** `names(MyProject)` lists what is exported. A type whose name
  would shadow one of Base's (a struct called `Number`) is defined but not
  exported: reach it as `MyProject.Number`.
- **Structs with no fields:** a header-only struct needs `-fstandalone-debug`.
- **Missing functions:** a macro needs `[wrap.macros]`, a template needs
  `[types] templates`, and an inline function needs a file that compiles it
  ([Header-only libraries](#Header-only-libraries)).
- **Stubs that raise.** A function RepliBuild could not prove safe to call is a
  stub that raises `ABI Safety Trap` or `FFI Safety Trap`; the rest of the
  module works.
  `filter(l -> occursin("Safety Trap", l), readlines("julia/MyProject.jl"))`
  lists them.
- **`?MyProject.f`** shows a function's Julia signature and its C argument
  types.

[Troubleshooting](troubleshooting.md) maps each symptom to the key that fixes it.

## 4. Edit the TOML, then build and wrap again

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
| A header-only library, or a header's inline functions | A file that compiles them, or `[wrap.macros]`: [Header-only libraries](#Header-only-libraries) |
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

Then `build(toml)` and `wrap(toml)` again; inputs that did not change are cache
hits. Load the result in a fresh Julia session. A session keeps the library it
has already loaded, and on Windows a build cannot replace a DLL that a running
session has loaded: the link fails with `failed to write output … Permission
denied`, and `clean` fails the same way.

## 5. Register

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
other language. The value is not checked: anything but `"c"`, `"C"` included,
takes the C++ toolchain.

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

## Header-only libraries

RepliBuild wraps what the compiler put into the library. A header that no
compiled file uses puts nothing there, so a header-only library starts with
nothing to wrap, however much of its API the header holds. Reading the header
would not change that: the code, its symbols and its DWARF exist only once
something compiles it. (RepliBuild does read headers, in the Clang.jl pass that
`[wrap] headers` scopes, but only for enums and callback typedefs, never for a
function.) So you give the library a file that uses it, and from there the
build is the normal one.

On a folder of headers alone, `discover` reports `found: 0 sources`, and
`build` throws `Compilation produced no library` after a warning that says to
add sources. `discover` picks the language from source files only, so with
none it also writes `language = "c"` and C flags, even for C++ headers. Add
your file first and then run `discover` (`force=true` if the TOML exists),
and it gets both right. The recipes that add no file of your own
(`shim_defines`, `[wrap.macros]` or `[types] templates` alone) need `language`
and the flags set by hand.

What to add depends on how the header delivers its code:

| The header has | Add |
|----------------|-----|
| An implementation section behind a macro (the stb style) | `[wrap] shim_defines`, or a two-line `.c` that defines the macro and includes the header |
| `static inline` C functions, or macros | `[wrap.macros]` entries |
| C `inline` functions that are not `static` | A declaration of each, without `inline`, in a `.c` |
| C++ inline functions, methods, constructors, templates | A keep file |
| C++ class templates only | `[types] templates` |

### An implementation section

Name the header and its macro in the TOML:

```toml
[wrap]
shim_headers = ["hol.h"]
shim_defines = ["HOL_IMPLEMENTATION"]
```

RepliBuild writes the define at the top of its shim file, before the header's
`#include`, so the first `build` compiles the implementation into the library
and the first `wrap` binds it, with no file of your own
([Edit the TOML §2.2](config.md#2.2-Preprocessor-macros)). The define reaches
that file only: a library source that includes the same header still sees just
the declarations, and links against the one implementation.

A file of your own does the same:

```c
/* hol_impl.c */
#define HOL_IMPLEMENTATION
#include "hol.h"
```

Put it in the project, where `discover` finds it, or list it in
`[compile] source_files`. miniaudio's repository ships such a file
(`miniaudio.c`), so its Hub config adds nothing.

Either way, define the macro in one place, and not in `[compile] flags`. A flag
reaches every file the build compiles, the shim file included, so once two of
them include the header, every function has two definitions and the link
stops. In a C build that error is `In-process libLLVM link failed for C
source`, which does not say why.

The functions of the implementation section are now in the library. The
header's inline functions and macros still are not, and nothing reports them
missing.

### `static inline` functions and macros

A `static inline` function never becomes a symbol of the library: each file
that calls it gets a private copy. List it under `[wrap.macros]`
([Edit the TOML §2.2](config.md#2.2-Preprocessor-macros)) as you would a
function-like macro. The shim RepliBuild compiles calls it by name, the same
way:

```toml
[wrap]
shim_headers = ["hol.h"]

[wrap.macros.hol_dot]
ret  = "float"
args = ["HolVec", "HolVec"]
```

`hol_dot(a, b)` in Julia calls the shim, and the shim calls the library's
inline function. Structs by value work. A library that is nothing but inline
functions needs no file of its own: the shim file is then the whole build.

### C `inline` functions

A C function declared `inline` and not `static` has only an inline definition
in the header, and no file emits it until one asks. A declaration without
`inline`, in one `.c` file, asks:

```c
#include "holi.h"
int holi_triple(int v);   /* this file now emits holi_triple */
```

That is the library's own function, with its own DWARF, and no shim.

### C++: a keep file

C++ emits an inline function (one defined inside its class, or declared
`inline`) only in a file that uses it, and at `-O2` the optimizer drops it
again once every call to it is inlined. A keep file uses each one in a way that
survives:

```cpp
// hol_keep.cpp: compiles the header-only library into the build.
// Nothing calls these.
#include "hol.hpp"

// Functions and methods: taking the address emits the definition.
[[gnu::used]] static auto keep_twice = &hol::twice;
[[gnu::used]] static auto keep_inc   = &hol::Counter::inc;

// Templates: an explicit instantiation emits every member the template
// defines, constructors included.
template struct hol::Box<double>;
template double hol::biggest<double>(double, double);

// Constructors and destructors have no address. Calling them from a function
// the optimizer leaves alone keeps their definitions.
#pragma clang optimize off
[[gnu::used]] static void keep_special() { hol::Gauge a; hol::Gauge b(3); }
#pragma clang optimize on
```

Without the `#pragma`, an `-O2` build inlines the constructors and the
destructor into `keep_special` and drops them.

What reaches the wrapper is the library's own code under its own symbols,
named as every C++ wrapper names things: `hol_twice`, `hol_Counter_inc`,
`hol_Box_double_get`, `hol_biggest_double`, the constructor `hol_Gauge_Gauge`
and the destructor `hol_Gauge_destroy_Gauge`. A method takes a pointer to the
object first, and constructors work on storage you provide
([Call a wrapper](calling.md)). It is a C++ build like any other, so it needs
the C++ toolchain.

`[types] templates` ([Edit the TOML §2.1](config.md#2.-What-discovery-cannot-know))
writes the class-template part of this for you, and with no other source it is
the whole build. A function template cannot be named there: instantiate it in
the keep file.

`[wrap.macros]` is a poor fit for C++. A namespaced name cannot be an entry,
because the name becomes part of the shim's.

### A header-only library from git

The keep file is yours, not the library's, so it lives in your package folder
next to the TOML. List it in `[compile] source_files`, relative to the package
folder. It is compiled along with whatever the dependency contributes:

```toml
[dependencies.hol]
type    = "git"
url     = "https://github.com/example/hol.git"
tag     = "v1.0"
commit  = "…"                     # full 40-hex SHA
exclude = ["tests", "examples"]

[compile]
flags        = ["-O2", "-fPIC", "-std=c++17"]
source_files = ["shim/hol_keep.cpp"]

[wrap]
language = "cpp"
```

The resolver compiles every C and C++ source file in the clone, and a
header-only repository's tests and examples are often all it has, so exclude
them. The clone's root, `include/` and `src/` are include directories, so the
keep file includes the header the way a user would: `#include "hol/hol.hpp"`.

### Boost and other large template libraries

Two things shape a Boost build first, and apply to any library built like it.

**A superproject is wrapped one library at a time.** `boostorg/boost` is a
superproject of about 170 submodules, and a plain clone of it is an empty tree.
`[dependencies.<n>] submodules` fetches exactly the submodules a library's
headers reach ([Edit the TOML](config.md), the `[dependencies]` table), so one
compiled Boost library is one package: its own `src/`, the header-only
libraries it includes, and nothing else. Boost's own tool computes that list
from a checkout of the superproject with `tools/boostdep` and the library
fetched:

```bash
python tools/boostdep/depinst/depinst.py -X test -X example -X doc filesystem
git submodule status | grep -v '^-'      # the closure; drop libs/headers and tools/
```

Two more things come from the library's own `CMakeLists.txt`. Its `PRIVATE`
defines are what its sources need (`BOOST_FILESYSTEM_SOURCE`, and the opt-outs
its probes would set on this toolchain, such as
`BOOST_FILESYSTEM_NO_CXX20_ATOMIC_REF` on a `-std=c++17` build); put them in
`[compile] flags`, since every file has to see them. And the submodules carry
sources outside `test/`, `example/` and `doc/` that are not the library:
configure checks (`config/`, `checks/`), code generators (`tools/`,
`preprocess*/`) and benchmarks, which the resolver would compile, so `exclude`
them by their `libs/<module>/…` path. A library that stands alone can still be
its own dependency instead: `boostorg/math` with `-DBOOST_MATH_STANDALONE`.

Most of a Boost library's public API is inline wrappers over its compiled
`detail::` functions, so a keep file (above) gives it symbols. Under the
library's `*_SOURCE` define the headers leave that inline API undefined, so
the keep file `#undef`s it before its first include and compiles as a user of
the library. A `BOOST_FORCEINLINE` member gets no symbol either way; call
the `detail::` function it forwards to. Hub `boostfilesystem`, `boostcharconv`
and `boostjson` are the worked examples.

**A template instantiation's name spells out its arguments**, defaults
included, and there is no key to rename it yet. Boost.Math's `pdf` comes out
over 600 characters long, because its policy type spells out thirteen
`default_policy` arguments. List the candidates with
`filter(n -> occursin("pdf", string(n)), names(M; all = true))`, and bind the
one you want to a short name in your own module: `const pdf = getfield(M, name)`.

A large template library also puts its internal types in DWARF.
`[wrap] surface_types_only` ([Edit the TOML](config.md), §2.14) emits only the
types its functions reach, template spellings included.

### From the REPL

The TOML recipes need no file of your own, so a header-only library can be
brought in from one Julia session and grown entry by entry:

```julia
using RepliBuild
ENV["REPLIBUILD_JULIA_ENV"] = "none"        # a throwaway: no environment changes
toml = RepliBuild.discover("hol")           # a folder holding hol.h
# in replibuild.toml:  [wrap] shim_headers = ["hol.h"]
#                             shim_defines = ["HOL_IMPLEMENTATION"]
RepliBuild.register(toml)
H = RepliBuild.use("hol")                   # the implementation, bound
# add a [wrap.macros] entry for an inline function, then, same session:
H = RepliBuild.use("hol")                   # rebuilt and loaded again, a new module
```

Each `use` after an edit rebuilds, loads a fresh copy and returns a new module,
in a few seconds ([Registry](use.md)). From there you call the library from
Julia, or write the parts you want in Julia against its types.

### Also

- A struct that comes out with no fields needs
  [`-fstandalone-debug`](config.md#2.5-Header-only-aggregates-need-fstandalone-debug).
- Keep `[compile] visibility` at its default. The library's inline functions
  carry no export annotation, which is what the visibility probe
  ([Edit the TOML](config.md), §2.12) looks for.
- The files RepliBuild writes for you, `replibuild_shims.c` (`.cpp` in a C++
  build) and `replibuild_templates.cpp`, are in `.replibuild_cache/`. Open them
  to see what was compiled.

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

Registration follows `discover`: a bare `ingest(so; headers=…)` writes the TOML
and does nothing else, while `build=true, wrap=true` builds, wraps, and then
registers the project, which vendors it to `~/.replibuild/` and adds it to your
Julia environment (`REPLIBUILD_JULIA_ENV` changes which). Pass `register=false`
to build and wrap without registering.
