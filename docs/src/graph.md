# CMake projects

An ordinary recipe builds **one** library from a list of files and one set of
flags. Some projects cannot be described that way: their CMake build produces
several libraries that link each other, gives individual files their own
flags, or writes headers before compiling.

For those, add a `[graph]` section. RepliBuild then asks CMake how the project
is built, builds every library that way with its own compiler, and wraps them
as one Julia module with a submodule per library.

[Several libraries](libraries.md) compares this with the other ways of
handling more than one library.

## When to use it

- The project builds **several libraries** and you want each one, with one
  library's types usable in another's functions.
- Files need **their own flags** (`-mavx2` on some files only), or the project
  needs a **generated `config.h`**.
- Source files are **generated** during the build.

Only CMake projects work this way. Projects built with autotools or Meson do
not.

On Linux nothing extra needs installing: RepliBuild brings `cmake` and
`ninja`. On Windows, install the CMake and Ninja packages
([Set up](install.md#Windows)). A C++ library in the project needs the C++
setup as usual.

## The recipe

The smallest one, for a CMake project in the project folder:

```toml
[project]
name = "hello_world"        # the module: julia/HelloWorld.jl

[graph]
source  = "."               # the folder holding CMakeLists.txt
targets = ["world"]         # the CMake target to wrap, plus the libraries it links
```

The CMake project can also be fetched. This recipe builds four libraries of
libwebp from its git repository:

```toml
[project]
name = "libwebp"

[dependencies.libwebp]
type   = "git"
url    = "https://github.com/webmproject/libwebp.git"
tag    = "v1.6.0"
commit = "4fa21912338357f89e4fd51cf2368325b59e9bd9"

[graph]
dependency = "libwebp"                    # the CMake project is that dependency
targets    = ["webpdemux", "libwebpmux"]  # webp and sharpyuv come with them
options    = ["-DBUILD_SHARED_LIBS=ON",
              "-DWEBP_BUILD_CWEBP=OFF", "-DWEBP_BUILD_DWEBP=OFF"]
```

| Key | What to put there |
|:----|:------------------|
| `source` | The folder holding `CMakeLists.txt`. Default `"."`. With `dependency`, a folder inside it. |
| `dependency` | The `[dependencies]` entry that is the CMake project: `git`, `archive` or `local` ([Where the source comes from](dependencies.md)). Its `exclude` is not used; CMake decides which files are compiled. |
| `targets` | The library targets you want, by their CMake names (the first argument of `add_library`). Leave it out for every library in the project. |
| `options` | What you would pass to `cmake` on the command line. Use it to switch off tests, tools and examples, and to ask for shared libraries (`-DBUILD_SHARED_LIBS=ON`) when the project leaves that open. |
| `build_type` | `CMAKE_BUILD_TYPE`. Default `"Release"`. |

`discover` does not write a `[graph]` section. Add it by hand.

Then build and wrap as usual:

```julia
RepliBuild.build("replibuild.toml")
RepliBuild.wrap("replibuild.toml")
RepliBuild.info("replibuild.toml")     # lists the libraries and their module names
```

## Which libraries are built

You name the targets you want. Every library those targets link, directly or
through another, is built and wrapped too. In the libwebp recipe, `webpdemux`
and `libwebpmux` both link `webp`, which links `sharpyuv`:

```
sharpyuv ← webp ← libwebpmux
                ← webpdemux
```

All four are built. A library that none of the chosen targets links
(`webpdecoder` here) is not. A program cannot be chosen as a target.

Each library is built as C or as C++ according to its own source files. Static
and object libraries inside the project are folded into the shared libraries
that use them.

## What you get

```
julia/
├── Libwebp.jl            # the module you load
├── graph.toml            # a list of the libraries and their module names
├── sharpyuv/lib/         # one folder per library: the library file and its wrapper
├── webp/lib/
└── …
```

```julia
include("julia/Libwebp.jl")

W  = Libwebp.Webp
MX = Libwebp.Libwebpmux
W.WebPGetEncoderVersion()
```

**Load the top module, never one library's wrapper by itself.** The libraries
depend on each other and the top module loads them in the right order.

**Module names.** The top module is named after the project (or
`[wrap] module_name`). Each library's submodule is the CamelCase of its CMake
target name. A name Julia already uses gets an `R` in front (`Core` becomes
`RCore`). `info(toml)` prints all of them.

## Types shared between libraries

A C or C++ type that several libraries use is **one** Julia type:

```julia
HelloWorld.World.Vec2  === HelloWorld.Hello.Vec2     # true
HelloWorld.World.Happy === HelloWorld.Hello.Happy    # an enum's values too
```

So a value, a pointer or an object that one library returns can be passed
straight to another library's functions.

Two types count as the same only when they really are:

- Same name in two different namespaces: two types.
- A type declared in a source file, not a header: private to its library.
- The same type with different layouts in two libraries: `wrap` stops with an
  error naming both. It never picks one silently.

## Rebuilding and registering

- Unchanged files are not recompiled and unchanged libraries are not linked
  again. CMake's configure step runs on every `build`.
- `clean(toml)` removes everything built and keeps a fetched project.
  `clean(toml; deps = true)` deletes the fetched project too.
- `register(toml)` registers the whole project as one, and `use(name)` returns
  the top module ([Load it anywhere](use.md)).

## Limits

- **The other recipe keys do not apply.** `[compile]`, `[types]`, and the
  `[wrap]` keys for macros, variadic functions, renames and exclusions are not
  used in a graph project: flags come from CMake, and each library is wrapped
  as it is. A macro or `static inline` function therefore has no Julia
  function; call the exported function it forwards to. `[link]
  optimization_level` does apply.
- **Generated files that need one of the project's own programs** (a code
  generator the project builds first) are refused, with the program named.
- **Types are shared within one project**, not between two separately wrapped
  projects.
