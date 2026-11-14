# Graph mode

A recipe without `[graph]` builds **one** library from a set of source files
and one flag set. Many real projects are not that: a CMake project builds
several libraries that link each other, gives individual files their own flags
(`-mavx2` on the AVX2 kernels), and writes headers at configure or build time.
Graph mode takes all of that from the project's own CMake description and
wraps every library, as one Julia module with a submodule per library.

The key table is [`[graph]`](config.md#[graph]-*(preserved)*) in the TOML
reference.

## When to use it

- The project builds **several libraries** and you want each as its own
  `.so`, with one library's types usable in another's functions.
- The library needs **per-file flags** or a **configure step** that a single
  flag set cannot reproduce (libwebp: SIMD files with their own `-msse4.1` /
  `-mavx2`, and a `config.h` from CMake).
- The project has **generated sources**: files a build-time custom command
  writes before anything compiles.

It needs `cmake` and `ninja` on the machine that builds, beside the usual
toolchain (and system LLVM/MLIR for any C++ library in the graph). It reads
CMake projects only; an autotools or Meson build is not a graph.

## A recipe

The smallest graph recipe, for a CMake project in the project directory:

```toml
[project]
name = "hello_world"        # the graph module: julia/HelloWorld.jl

[graph]
source     = "."            # the directory holding CMakeLists.txt
targets    = ["world"]      # world, and the shared libraries it links
build_type = "Release"
```

The CMake project can instead be a pinned upstream. This is the Hub's libwebp
recipe, four libraries from upstream's own CMake project at a verified commit:

```toml
[project]
name = "libwebp"

[dependencies.libwebp]
type   = "git"
url    = "https://github.com/webmproject/libwebp.git"
tag    = "v1.6.0"
commit = "4fa21912338357f89e4fd51cf2368325b59e9bd9"

[graph]
dependency = "libwebp"                    # configure that checkout
source     = "."                          # CMakeLists.txt at its root
targets    = ["webpdemux", "libwebpmux"]  # webp and sharpyuv come with them
options    = ["-DBUILD_SHARED_LIBS=ON",
              "-DWEBP_BUILD_CWEBP=OFF", "-DWEBP_BUILD_DWEBP=OFF"]
```

`dependency` is any `git`, `archive` or `local` entry, resolved exactly as for
a single library — cloned, pinned, `submodules` fetched, an archive's hash
checked ([Dependencies](dependencies.md)) — before the checkout is
configured. `options` are passed to the configure step after RepliBuild's own;
use them to turn off what you do not want built (tools, tests, examples) and
to choose shared libraries where upstream's `add_library` leaves the kind to
`BUILD_SHARED_LIBS`.

Then the usual verbs:

```julia
RepliBuild.build("replibuild.toml")    # configure, generate, compile, link, extract
RepliBuild.wrap("replibuild.toml")     # one wrapper per library + the graph module
RepliBuild.info("replibuild.toml")     # what the graph holds and how it is named
```

## What `build` does

1. **Configure only.** CMake configures the project with the Ninja generator
   and writes its File API reply: every target, its sources, each source's
   compile flags, defines and include directories, its link line and its
   dependencies. Nothing is compiled by CMake or Ninja.
2. **Generated sources.** Files the reply marks as generated are written by
   running the build's own custom-command steps for exactly those files —
   never by building the project's libraries. A step that would need one of
   the project's own executables first (ROOT's `rootcling`) is refused with
   the target named.
3. **Compile.** Every translation unit is compiled to LLVM IR with
   RepliBuild's toolchain, under the flags CMake gave that file.
4. **Link.** One `.so` per shared or module library target. An OBJECT or
   STATIC library is folded into the libraries that link it (and not again
   into a library that only gets it through a shared dependency); a static
   library nothing folds is linked on its own.
5. **Extract.** Each library gets its own `compilation_metadata.json`, as a
   single-library build would.

### How a library's dependencies resolve

`targets` names the libraries you want. The selection is closed over their
link dependencies: every shared library a selected target links, directly or
through another, is built and wrapped too, and ordered before the libraries
that link it. In the libwebp recipe, `webpdemux` and `libwebpmux` both link
`webp`, which links `sharpyuv`, so the graph is

```
sharpyuv ← webp ← libwebpmux
                ← webpdemux
```

and the load order is `sharpyuv`, `webp`, then the two in a stable order. A
library nothing selected links is not built (`webpdecoder` here), and an
executable or utility target cannot be selected. Empty `targets` means every
library target.

A library links the others **by the files RepliBuild built**, never by
CMake's: each has its file name as its soname on Linux, and on Windows each
wrapper opens the DLLs it links where they sit in the graph before its own.
System libraries on a target's link line (`-lm`, `/usr/lib/libz.so.1`) are
linked as CMake says.

Each library is C or C++ by its own translation units, and compiled by that
toolchain: a C library in a mixed graph still takes the C path (and Tier 3
`ccall`), a C++ one the system LLVM and Tier 2 thunks where it needs them.

## What `wrap` writes

```
julia/
├── Libwebp.jl                  # the graph module
├── graph.toml                  # each library, its module, what it links
├── sharpyuv/lib/               # one directory per library:
│   ├── libsharpyuv.so          #   the library
│   ├── Sharpyuv.jl             #   its wrapper
│   ├── compilation_metadata.json
│   └── type_manifest.json      #   the types this wrapper defined
├── webp/lib/ …
└── …
```

The graph module includes every library's wrapper in load order, so each
library is open before anything that links it:

```julia
include("julia/Libwebp.jl")
W  = Libwebp.Webp
MX = Libwebp.Libwebpmux
W.WebPGetEncoderVersion()
```

**Load a library's wrapper only through the graph module.** A wrapper that
imports types from an earlier library refers to it through the graph module,
so including it alone fails.

**Module names.** The graph module is named like a single library's wrapper:
`[wrap] module_name`, else CamelCase of `[project] name`. Each library's
module is CamelCase of its CMake target name, with two renames: a name Julia
binds already (`Core`, `Base`, `Libdl`, `RepliBuild`) gets an `R` in front
(`Core` → `RCore`), and a name equal, ignoring case, to the graph module's or
an earlier library's gets `_2`, `_3`, …. `info(toml)` and `graph.toml` print
the names.

## One type across libraries

A C or C++ type several libraries use is **one Julia type**. The first
library in load order whose wrapper emits it owns it, and every later library
imports it:

```julia
HelloWorld.World.Vec2 === HelloWorld.Hello.Vec2     # true
HelloWorld.World.Happy === HelloWorld.Hello.Happy   # an enum's values come too
```

So a value, a pointer or a `Managed` object one library returns passes
straight into another library's functions, a class in one library derives
from a class in another and keeps its base's fields, and a record embedding
another library's record keeps its layout. Functions an earlier library's
wrapper already binds are not bound again.

Whether two libraries' types are the same type is checked, not assumed:

- Their enclosing scopes must meet: two `Config`s in two namespaces are two
  types, and each library keeps its own.
- A type declared only in a source file (`.c`, `.cpp`), never a header, is
  local to that file and stays its library's own: libwebp's decoder and
  demuxer each have a file-local `MemBuffer`, and they are two types.
- One type with two layouts in two libraries, or with a member both define
  strongly, is a **wrap error** naming both libraries — never a silent pick.
- A header-only type two libraries use that share no dependency that emits it
  (libwebp's `WebPData`, used by mux and demux but not by `webp`) stays one
  type per library, and `wrap` reports it.

An enum imported this way brings its enumerators, under the names the owner
bound, so a caller never needs to know which library owns an enum.

## Caching, cleaning, registering

- An unchanged translation unit is not recompiled, and a library whose IR,
  flags, link arguments and dependencies are unchanged is not relinked or
  re-extracted. Every `build` still configures and runs the generator steps.
- `clean(toml)` removes the CMake build tree, the output directory and a
  dependency's clone; `clean(toml; deps = false)` keeps the clone and checks
  its pin on the next build.
- `register(toml)` vendors the whole graph as one unit and `use(name)` returns
  the graph module ([Registry](use.md)). A change to any file of the CMake
  project is a new build.

## Not built yet

- Per-library recipe keys. `[compile]` and `[types]` and the other `[wrap]`
  keys (macros, varargs, renames, `exclude_symbols`) do not apply to graph
  libraries: flags come from CMake and each library is wrapped with defaults.
  A `static inline` helper in a graph library's header therefore has no shim;
  call the exported function it forwards to (libwebp's `WebPConfigInit` is
  `WebPConfigInitInternal(config, preset, quality, WEBP_ENCODER_ABI_VERSION)`).
  `[link] optimization_level` and `fallback` apply to every library.
- Generator steps that need one of the project's own executables.
- A type shared by two *separate* recipes (two Hub packages, one linking the
  other) is not imported across them; that needs registry dependencies.
