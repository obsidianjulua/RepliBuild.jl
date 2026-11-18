# What goes in the TOML

**RepliBuild does everything except what has to be stated, and everything
that has to be stated goes in `replibuild.toml`.** That is the design. There
is no second file, no script to maintain, no generated code to patch. If the
wrapper is wrong or incomplete, the fix is a line in this file.

This page goes through a library the way you should: one question at a time,
and for each, how to tell whether your library needs it, what to write, and
what happens if you leave it out. [Every key](config.md) is the lookup table
for when you know which key you want.

## What you state, and what you never state

You never describe the library's functions, structs, enums or classes, how its
files depend on each other, or how a call is made. RepliBuild compiles the
library and takes all of that from the result, so it cannot disagree with the
binary.

What you state is what no tool can work out, because it is not in the source
tree's shape or in a compiled library:

- where the source is, and which files in it are not the library;
- the flags and defines the library's own build system would pass;
- what the compiler discards or never sees: macros, code that exists only in
  headers, templates nobody instantiated;
- what a signature does not say: which extra arguments a variadic function
  takes, who frees a returned string;
- your own choices: what to leave out of the wrapper, what to call things.

Leaving one of these out rarely fails the build. The usual result is a wrapper
that loads and is missing something, so go through the list deliberately.

The first item matters most. RepliBuild can only build what it is fed, and how
you feed it differs for files in your project, a folder on disk, a git
repository and a tarball: [Where the source comes from](dependencies.md).

## The checklist

Read the library's public headers and its build files (`CMakeLists.txt`,
`Makefile`) with these questions.

| Question | If yes | Section |
|:---------|:-------|:--------|
| Where is the source: in the project, a folder on disk, git, a tarball? | `source_files`, `[dependencies]` | [Where the source is](#Where-the-source-is) |
| Does it hold tests, examples, tools, other platforms' code? | `exclude`, or a shorter `source_files` | [Files that are not the library](#Files-that-are-not-the-library) |
| Is it more than one library? | several sources, `[graph]`, or `link_dirs` | [Several libraries](libraries.md) |
| Does its build pass `-D` defines, a `-std=`, extra include folders? | `[compile]` | [Flags, defines and include folders](#Flags,-defines-and-include-folders) |
| Does its build write a `config.h` or other header? | capture it, or use graph mode | [Headers the build system generates](#Headers-the-build-system-generates) |
| Does it use `sin`, threads, `dlopen`, or another library? | `[link]` | [Libraries it links](#Libraries-it-links) |
| Is part of the API `#define`d? | `[wrap.macros]` | [Macros](#Macros) |
| Are there `static inline` functions, or is it a single-header library? | `[wrap.macros]`, `shim_defines` | [Code that lives in headers](#Code-that-lives-in-headers) |
| Are there functions ending in `...`? | `[wrap.varargs]` | [Variadic functions](#Variadic-functions) |
| Does a function return a `char*` the caller must free? | `[wrap.cstring_owned]` | [Strings the caller must free](#Strings-the-caller-must-free) |
| Do structs or classes come out empty? | `-fstandalone-debug` | [Types declared only in headers](#Types-declared-only-in-headers) |
| C++: do you want `std::vector`, `std::map`, or a library template? | `[types] templates` | [C++ templates and containers](#C-templates-and-containers) |
| Does the wrapper carry internal functions or types? | `visibility`, `exclude_symbols`, `surface_types_only` | [Leaving things out](#Leaving-things-out) |
| Is a name unusable? | `[wrap.rename]` | [Names](#Names) |
| Must one recipe work on Linux and Windows? | `[platform.*]` | [Linux and Windows in one file](#Linux-and-Windows-in-one-file) |
| Is it a CMake project of several libraries? | `[graph]` | [CMake projects](graph.md) |

## A complete recipe

Lua, from its git repository. Each block answers one of the questions above.

```toml
[project]
name = "lua"                       # → liblua.so, module Lua, julia/Lua.jl

[dependencies.lua]                 # where the source is
type    = "git"
url     = "https://github.com/lua/lua.git"
tag     = "v5.4.7"
commit  = "1ab3208a1fceb12fca8f24ba57d6e13c5bff15e3"
exclude = ["onelua.c", "lua.c", "luac.c", "ltests.c"]   # programs and tests, not the library

[compile]
flags = ["-O2", "-fPIC", "-DLUA_USE_LINUX"]             # what Lua's Makefile passes

[link]
enable_lto         = false
optimization_level = "2"
link_libraries     = ["m", "dl"]                        # -lm -ldl

[wrap]
language     = "c"
shim_headers = ["lua.h", "lauxlib.h"]                   # headers the macros come from

[wrap.macros.lua_pop]                                   # part of Lua's API is macros
ret  = "void"
args = ["lua_State*", "int"]

[wrap.varargs]                                          # lua_pushfstring(L, fmt, ...)
proven_at       = "1ab3208a1fceb12fca8f24ba57d6e13c5bff15e3"
lua_pushfstring = [["Cstring"], ["Cint"], ["Cstring", "Cint"]]

[platform.windows.compile]                              # what differs on Windows
flags = ["-O2"]
[platform.windows.link]
link_libraries = []
```

Lua's 33 source files, and every struct, enum and function in them, are not
mentioned. Those come from the build.

## The four lines every recipe has

```toml
[project]
name = "mylib"

[compile]
flags = ["-O2", "-fPIC"]

[link]
enable_lto         = false
optimization_level = "2"

[wrap]
language = "c"
```

- **`[project] name`** names everything: the library file (`libmylib.so`), the
  Julia module (`Mylib`, CamelCase of the name) and the name you
  [`use`](use.md) it by. `[wrap] module_name` overrides the module name.
- **`[wrap] language`** is `"c"` or `"cpp"`. It chooses the compiler, so it
  must match the library: a C++ library built as `"c"` loses its classes and
  methods. `discover` sets it from the file extensions. Spell it in lower case;
  anything other than `"c"` is treated as C++.
- **`[compile] flags`** are passed to the compiler for every file. Always
  include `-fPIC`. `discover` writes no `-O` flag, which is an unoptimized
  build: add `-O2`, and set **`[link] optimization_level = "2"`** to match.
  For C++, add the standard the library needs (`-std=c++17`).
- **`[link] enable_lto = false`**. Keep this line in every recipe. `discover`
  writes it; a C recipe written by hand without it switches on an experimental
  calling mode.

## Where the source is

This is the first thing to state, and each way of stating it behaves
differently. [Where the source comes from](dependencies.md) is the full
account; the short version:

```toml
# 1. Files in the project folder: listed, one by one. build never searches for them.
[compile]
source_files = ["src/mylib.c", "extra/glue.c"]

# 2. A folder on disk: searched on every build. Its tests ARE compiled unless excluded.
[dependencies.mylib]
type    = "local"
path    = "../mylib"
exclude = ["tests/", "examples/"]

# 3. A git repository: cloned, checked against the commit, searched.
[dependencies.cjson]
type   = "git"
url    = "https://github.com/DaveGamble/cJSON.git"
tag    = "v1.7.18"
commit = "acc76239bee01d8e9c858ae2cab296704e52d916"   # the full 40-character commit

# 4. A tarball: downloaded, checked against the hash, unpacked, searched.
[dependencies.quickjs]
type   = "archive"
url    = "https://bellard.org/quickjs/quickjs-2026-06-04.tar.xz"
sha256 = "b376e839b322978313d929fd20663b11ba58b75df5a46c126dd19ea2fa70ad2a"
```

A recipe may use any number of these together. All of it is compiled into the
one library.

- `discover` writes `source_files` for a project folder, listing every source
  file it finds, tests included. Read the list and delete what does not
  belong.
- Always give `commit` with a git dependency. A tag can be moved to different
  code; with `commit` set, a moved tag stops the build instead of silently
  compiling something else.

## Files that are not the library

**How to tell.** Look at the top level of the source. Folders such as `test`,
`examples`, `tools`, `bench`, `fuzz`, `cli`, a bundled copy of another library,
or one folder per platform or backend are not the library. Any `.c` or `.cpp`
file with a `main` function is a program, not the library.

RepliBuild compiles every source file it is given, so these have to be kept
out. How depends on where the source is:

| Source | Kept out by |
|:-------|:------------|
| Files in the project folder | Deleting their lines from `source_files` |
| A `local` folder | `exclude`. Only `build`, `.git` and `.cache` are skipped by themselves |
| `git` or `archive` | `exclude`. Folders named `test`, `tests`, `example`, `examples`, `fuzzing`, `build`, `doc` and `docs` are skipped by themselves |

```toml
[dependencies.box2d]
type    = "git"
url     = "https://github.com/erincatto/box2d.git"
tag     = "v2.4.1"
commit  = "9ebbbcd960ad424e03e5de6e66a40764c16f51bc"
exclude = ["testbed/", "unit-test/", "extern/"]   # a GUI demo, tests, bundled libraries
```

An entry matches when it is the file's name, or appears anywhere in the file's
path inside the dependency. So `"tools/"` removes a folder and `"luac.c"`
removes one file, and a short entry such as `"t"` removes far too much. Write
folder entries with a trailing `/`.

**If you leave it out.** The build fails on a duplicate `main` or a missing
header, or it succeeds and the wrapper is full of test helpers and another
library's functions. A `main` among `names(M)` is the sign.

## Flags, defines and include folders

**How to tell.** Open the library's `CMakeLists.txt` or `Makefile` and look for
`-D…`, `add_definitions`, `target_compile_definitions`, `-std=`, and
`include_directories`. A README section called "Building" or "Configuration"
often lists the defines that switch features on.

```toml
[compile]
flags = [
    "-O2", "-fPIC", "-std=c++17",
    "-DGGML_USE_CPU",                      # a feature switch
    '-DGGML_VERSION="0.18.1"',             # a define with a string value
]
include_dirs = [
    "include",                                         # in the project
    ".replibuild_cache/deps/llamacpp/ggml/include",    # inside a fetched dependency
]

[compile.defines]            # the same as -DSQLITE_THREADSAFE=1 in flags
SQLITE_THREADSAFE = "1"
```

- Relative paths are relative to the folder holding `replibuild.toml`. Do not
  write absolute paths into a recipe you intend to share.
- A git or archive dependency lives in `.replibuild_cache/deps/<name>/`. Its
  top folder, its `include/` and its `src/` are searched for headers already.
  A `local` folder is searched at its top and in its `include/` only. Name any
  other folder under `include_dirs`.
- `[compile.defines]` always writes `-Dname=value`. For a define with no value
  (`-DFOO`), use `flags`.

**If you leave it out.** A missing include folder is a compile error naming the
header. A missing define is either a compile error in one file, or a feature of
the library that is silently switched off.

## Headers the build system generates

**How to tell.** The build stops on a header that is not in the repository:
`config.h: No such file or directory`, or `<mylib>_config.h`. The repository
has a `config.h.in`, `config.h.cmake` or a `configure_file` line in its
`CMakeLists.txt` instead.

Do not write the header by hand. For a CMake project there are two ways:

1. **Let CMake drive the build.** Add a `[graph]` section, and RepliBuild runs
   the project's configure step itself: [CMake projects](graph.md).
2. **Capture the generated files once** and keep them beside the recipe:

   ```julia
   using RepliBuild
   const S = RepliBuild.SysConfigGen

   probe = S.cmake_probe("path/to/library/source";
                         args = ["-DBUILD_TESTING=OFF"])    # the library's own CMake options
   S.capture_config(probe, "config")        # copies the generated headers into ./config/
   print(S.toml_fragment(probe))            # proposed [compile] flags, include dirs and excludes
   ```

   `cmake_probe` runs only CMake's configure step, with the same compiler the
   build will use. `capture_config` also writes a `SYSCONFIG.md` recording how
   the files were produced. `toml_fragment` prints a suggestion: read it, and
   paste what applies. The captured files describe one platform; capture again
   for each platform and after upgrading the library.

A header written while the library *builds*, by a script or a helper program,
is beyond the second way. Use `[graph]`, or look in the repository for a
ready-made copy (libpng ships `pnglibconf.h.prebuilt`).

## Libraries it links

**How to tell.** The library includes `<math.h>` (`m`), `<pthread.h>`
(`pthread`), `<dlfcn.h>` (`dl`), or the headers of another library. Its build
files have `target_link_libraries` or `-l…`.

```toml
[link]
link_libraries = ["m", "dl"]              # -lm -ldl
link_dirs      = ["/opt/thing/lib"]       # where to find a library that is not a system one
```

A library found through `link_dirs` is also the one loaded at run time, so the
wrapper uses the copy you linked and not another that happens to be installed.

`m`, `dl`, `pthread` and `rt` are Linux names. A recipe that should also build
on Windows overrides the list there
([Linux and Windows in one file](#Linux-and-Windows-in-one-file)).

**If you leave it out.** The build still succeeds, because a shared library may
have unresolved names. The failure comes later, as `could not load symbol` at
the first call that needs the missing library, or not at all on your machine
and then on someone else's.

## Macros

**How to tell.** Search the public headers for `#define`. Two kinds matter:
constants the API expects you to pass (`SQLITE_OK`, `LUA_REGISTRYINDEX`), and
function-like macros that the documentation presents as functions (`lua_pop`,
`deflateInit`).

A macro is gone once the library is compiled, so each one you want is declared
with its types, and RepliBuild compiles a small real function for it:

```toml
[wrap]
shim_headers = ["lua.h", "lauxlib.h"]     # the headers that define the macros

[wrap.macros.lua_pop]                     # #define lua_pop(L, n) …
ret  = "void"
args = ["lua_State*", "int"]

[wrap.macros.LUA_REGISTRYINDEX]           # #define LUA_REGISTRYINDEX (…)
ret = "int"
```

- `ret` and `args` are **C types**, written as you would in C.
- With `args`, the entry is a function-like macro. Without `args`, it is a
  constant. `args = []` is a function-like macro taking no arguments.
- In Julia both are functions: `lua_pop(L, 1)` and `LUA_REGISTRYINDEX()`.
- `shim_headers` are written as you would `#include` them. Each must be found
  inside the project or a dependency. If one resolves to a copy installed on
  the system instead, the build stops, because that copy may be a different
  version: give a longer path (`"c/blake3.h"`) or add the folder to
  `include_dirs`.

**If you leave it out.** The macro is not in the wrapper, and nothing says so.

## Code that lives in headers

Three cases, in rising order of work. [Header-only libraries](header-only.md)
covers all of them in full.

**A single-header library** in the stb style, whose implementation is compiled
only where a macro is defined:

```toml
[wrap]
shim_headers = ["stb_image.h"]
shim_defines = ["STB_IMAGE_IMPLEMENTATION"]
```

That is the whole recipe for it: no source file of your own. Do not put the
define in `[compile] flags` instead. There it reaches every file, the
implementation is compiled more than once, and the link fails.

**`static inline` functions** in a C header are declared like macros, under
`[wrap.macros]`.

**C++ functions defined inside a class** need nothing. RepliBuild compiles them
in itself, provided the class is fully described
([Types declared only in headers](#Types-declared-only-in-headers)).

## Variadic functions

**How to tell.** A declaration ends in `...`: `int ml_log(const char *fmt, ...)`.

The declaration does not say what the extra arguments are, so list the
combinations you want to call. Each becomes its own Julia function:

```toml
[wrap.varargs]
proven_at       = "1ab3208a1fceb12fca8f24ba57d6e13c5bff15e3"
lua_pushfstring = [
    ["Cstring"],              # lua_pushfstring(L, fmt, const char*)
    ["Cint"],                 # lua_pushfstring(L, fmt, int)
    ["Cstring", "Cint"],      # lua_pushfstring(L, fmt, const char*, int)
]
```

Three rules:

1. **List only the extra arguments.** The fixed ones (`L` and `fmt` here) are
   already known.
2. **Use Julia's names for C types**, not C's: `Cint`, `Cuint`, `Clong`,
   `Culong`, `Clonglong`, `Csize_t`, `Cdouble`, `Cfloat`, `Cstring`, `Cchar`,
   `Bool`, the sized types (`Int32`, `UInt64`, `Float64`, …), and `Ptr{Name}`
   for a pointer. This is the opposite of `[wrap.macros]`, which takes C types.
3. **It is a list of lists.** One combination with one argument is
   `[["Cint"]]`, not `["Cint"]`.

In Julia each combination is named after its types:
`lua_pushfstring_Cstring(L, fmt, s)`, `lua_pushfstring_Cstring_Cint(L, fmt, s, i)`.
The plain `lua_pushfstring(L, fmt)` takes the fixed arguments only.

**`proven_at`** is the commit of the library you checked these combinations
against. Without it, `wrap` warns on every run. With it, upgrading the library
to another commit stops the build until you have looked at the function again
and updated the value. It is compared with the `commit` of the dependency, so a
project with no git dependency cannot set it and keeps the warning.

**If you leave it out.** Only the fixed-arguments form exists, and `wrap`
warns once per function.

## Strings the caller must free

**How to tell.** A function returns `char*` (not `const char*`), and the
documentation says the caller owns the result, or names a function to release
it: `cJSON_Print` and `cJSON_free`, `sqlite3_mprintf` and `sqlite3_free`.

```toml
[wrap.cstring_owned]
cJSON_Print     = "cJSON_free"
sqlite3_mprintf = "sqlite3_free"
strdup_like_fn  = "free"            # released with the C library's free
```

The Julia function then copies the text into a `String`, frees the C buffer
with the function you named, and returns the `String`.

Without an entry, a `char*` result is still returned as a `String` (or
`nothing` for a NULL pointer), and the C buffer is left alone. That is correct
for strings the library keeps, such as `sqlite3_errmsg`, which is why it is the
default.

**If you leave it out.** Every call leaks the buffer.

## Types declared only in headers

**How to tell.** A struct or class you can see in a header appears in the
wrapper with no fields, cannot be constructed, or functions that take it by
value raise an `ABI Safety Trap`. Typical victims are the "settings" structs a
caller fills in (`b2BodyDef`, `llama_context_params`) and C++ classes whose
constructors are written inside the class.

The compiler records a type's full definition only where the library's own code
needs it. One flag makes it record every type the headers define:

```toml
[compile]
flags = ["-O2", "-fPIC", "-std=c++17", "-fstandalone-debug"]
```

Add it to C++ recipes from the start. For a large C++ library where it makes
the build too big, `"-Xclang", "-fno-use-ctor-homing"` (two entries in `flags`)
covers the most common case, classes with header-defined constructors, at a
fraction of the size.

With the definitions recorded, RepliBuild also does three things for C++
classes without being asked:

- a **constructor** defined in the header becomes callable;
- a **member function** defined in the class body becomes callable;
- a call that **leaves out default arguments** becomes a shorter method of the
  same function.

`wrap` prints a line for each, saying how many it added and what it skipped.

**A class that still does not appear** is one that no source file of the
library mentions at all. Name it:

```toml
[types]
records = ["btDefaultMotionState"]        # spelled as in C++, with its namespace
```

## C++ templates and containers

**How to tell.** You want to create a `std::vector<double>` in Julia and hand
it to the library, or the API uses an instantiation of the library's own
template (`fmt::basic_memory_buffer<char>`), and the type is missing from the
wrapper.

A template is compiled only for the type arguments somebody uses. List the
ones you want:

```toml
[types]
templates        = ["std::vector<double>", "std::string", "std::map<int, int>"]
template_headers = ["<fmt/format.h>"]      # headers for templates that are not std::
```

- Entries are C++ type names, written as in C++.
- The standard headers for `std::vector`, `std::string`, `std::map`,
  `std::unordered_map`, `std::set`, `std::deque` and `std::list` are found
  for you. Name any other header in `template_headers`: `<…>` for a system
  header, a plain name for one of the library's.
- `std::vector`, `std::string`, `std::map` and `std::unordered_map` get a
  Julia constructor, `create_std_vector_double()`, and work like Julia
  collections ([Call the wrapper](calling.md#Containers)). Any other template
  is compiled so its type exists in the wrapper, with no constructor.
- A function template cannot be listed here
  ([Header-only libraries](header-only.md#C:-a-keep-file)).

A library whose API uses no templates needs none of this. Functions that take
or return `std::string` work without an entry.

**If you leave it out.** The type is missing, or there is no way to create one
from Julia. An entry that produces nothing is reported by `build` as a warning
naming it.

## Leaving things out

By default the wrapper has a function for everything the library exports, and
a type for everything the build recorded. For many libraries that includes
internals. There are three tools, in the order to reach for them.

### `exclude_symbols`: name what should go

```toml
[wrap]
exclude_symbols = ["_pcre2_*", "matmul_*", "vk::*"]
```

Each entry is a pattern for a function or global variable to leave out. `*`
matches any text and `?` one character; everything else is literal, including
`(`, `+`, `<` and `::`. A pattern must match a whole name, so use a trailing
`*` for a prefix.

A C function is matched by its name. A C++ function is matched by its full C++
signature, `shapes::Rect::scale(double)`, so end the pattern with `*`:
`shapes::Rect::scale*` for one function's overloads, `vk::*` for a namespace.
Its symbol name in the library (`_ZN6shapes4Rect5scaleEd`) matches too.

A pattern that matches nothing is an error. It means the library changed or
the pattern has a typo, and either way the recipe is no longer doing what you
wrote it to do.

This is applied by `wrap`: changing it needs no rebuild.

### `visibility`: use the library's own list of public functions

Many libraries mark their public functions with an export macro
(`PCRE2_EXPORT`, `B2_API`, `CJSON_PUBLIC`). For those, one key hides
everything unmarked:

```toml
[compile]
visibility = "hidden"
```

!!! danger "Check before you set it"
    On a library that does **not** mark its public functions, this hides the
    entire API and the build is refused. Ask first:

    ```julia
    RepliBuild.VisibilityProbe.visibility_probe("replibuild.toml")
    ```

    ```
    VisibilityReport: pcre2
      external defs      : 105
      library annotated  : 78  (74.3%)
      verdict            : annotated
      wrapped today      : 119
        retained         : 92
        DROPPED          : 27
    ```

    `annotated` means the key is safe. Read the dropped list anyway: those
    functions leave the wrapper, and the probe cannot tell an internal helper
    from a public function the authors forgot to mark.

Some libraries only mark their functions when a define is set (cJSON needs
`-DCJSON_API_VISIBILITY`, ggml needs `-DGGML_SHARED`). The probe uses your
recipe's flags, so run it again after changing them, and after upgrading the
library.

### `surface_types_only`: drop types the API never uses

A C++ library that compiles a large internal codebase can record thousands of
internal types. If the module fails to load because of them (a type that
shadows a Julia name, duplicate fields), keep only the types the wrapped
functions and globals reach:

```toml
[wrap]
surface_types_only  = true
surface_types_extra = ["MD_*_DETAIL", "Argon2_ErrorCodes"]   # types to keep anyway
```

`surface_types_extra` is for types a caller needs that no function signature
names: a struct passed to a callback as `void*`, or an enum whose values are
returned as plain `int`. Patterns work as in `exclude_symbols`, and one that
matches nothing is an error.

Leave this off unless the module does not load. It can remove types you need.

## Names

A C function keeps its name. A C++ function is named after its namespace, class
and name, joined with `_`: `tinyxml2::XMLDocument::Parse` is
`tinyxml2_XMLDocument_Parse`. Template arguments are part of the name, which
can make it hundreds of characters long.

Give a function a name of your own with `[wrap.rename]`:

```toml
[wrap.rename]
"boost_math_pdf_*"            = "pdf"          # every match becomes a method of `pdf`
shapes_Rect_area              = "rect_area"    # by the name the wrapper gives it
_ZN8stl_test6scaledIlEET_S1_i = "scaled_long"  # by its symbol name in the library
```

- The key is the name the wrapper gives the function without the rename, or
  the function's symbol name in the library. Patterns work as in
  `exclude_symbols`.
- The value must be a valid Julia name.
- Several functions may be given one name. They become methods of one Julia
  function, told apart by their argument types.
- The old name no longer exists.
- A key that matches nothing, or a new name that is already taken, is an error.

Types cannot be renamed. In your own code, `const Normal =
M.normal_distribution_double_…` gives one a short name.

## Linux and Windows in one file

Write the recipe for Linux. Put what differs on Windows in
`[platform.windows.*]` tables, which are laid over the rest on that system:

```toml
[compile]
flags = ["-O2", "-fPIC", "-DLUA_USE_LINUX"]

[link]
link_libraries = ["m", "dl"]

[platform.windows.compile]
flags = ["-O2"]                       # replaces the list above

[platform.windows.link]
link_libraries = []                   # an empty list means none

[platform.windows.dependencies.glfw]
exclude = ["x11_", "glx_", "posix_", "linux_"]   # the exclude list of [dependencies.glfw], on Windows
```

A value in a platform table replaces the value outside it. Lists are replaced
whole, not merged. The platform names are `linux`, `windows` and `macos`; any
other name is an error.

## When an edit takes effect

| Needs `build`, then `wrap` | Needs only `wrap` |
|:---------------------------|:------------------|
| `[compile]`: flags, defines, include folders, `visibility`, `source_files` | `[wrap] exclude_symbols` |
| `[link]`: libraries, folders, optimization level | `[wrap.rename]` |
| `[dependencies]` | `[wrap.varargs]` |
| `[types] templates`, `template_headers`, `records` | `[wrap.cstring_owned]` |
| `[wrap.macros]`, `shim_headers`, `shim_defines` | `[wrap] surface_types_only`, `surface_types_extra` |

Running `build` when nothing it reads has changed costs almost nothing, so when
unsure, run both.

## Mistakes the file does not catch

- **A misspelled key is ignored, silently.** If a setting seems to do nothing,
  check its spelling and its section against [Every key](config.md).
- **`[wrap.macros]` takes C types; `[wrap.varargs]` takes Julia types.**
- **A dependency with a `url` and no `type` is not fetched.** The default type
  is `"local"`.
- **A new file in the project folder is not compiled** until it is in
  `source_files`.
- **A `local` folder's `tests/` are compiled** unless `exclude` names them.
- **`extra_link_libs` is not a `[link]` key.** The key is `link_libraries`.

## Recipes to copy from

[RepliBuild-Hub](https://github.com/obsidianjulua/RepliBuild-Hub) holds working
recipes for lua, sqlite, zlib, pcre2, cJSON, imgui, box2d, fmt, pugixml,
llama.cpp, libwebp and others. Find one whose library is built like yours and
start from it. `RepliBuild.search("xml")` lists them by keyword.
