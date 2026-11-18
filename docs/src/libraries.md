# Several libraries

One recipe normally builds one library. Real projects are often more than one:
a library that needs another, a project whose build produces four libraries, a
wrapper that must share state with a second wrapper. RepliBuild handles each of
these, and which one you have decides what you write.

| You have | You write | You get |
|:---------|:----------|:--------|
| Several pieces of source that belong in one library | One recipe with several sources | One library, one module |
| A CMake project that builds several libraries | One recipe with a `[graph]` section | One module, with a submodule per library and shared types |
| Two libraries you wrap separately, one using the other | Two recipes, the second linking the first | Two modules that share one copy of the first library |
| Unrelated libraries | One recipe each | Independent modules that load side by side |

## Several sources, one library

Everything a recipe is given is compiled into the same library: its own
`source_files` and every `[dependencies]` entry. If library A cannot be built
without library B's source, and you do not need B as a library of its own, put
both in one recipe:

```toml
[project]
name = "app"

[dependencies.core]
type   = "git"
url    = "https://github.com/example/core.git"
tag    = "v2.1"
commit = "…"

[dependencies.plugins]
type    = "local"
path    = "../plugins"
exclude = ["tests/"]

[compile]
source_files = ["glue/bridge.c"]
```

The result is one `libapp.so` and one module, `App`, holding the functions of
all three. All of it is compiled with the same flags, which is the limit of
this approach: if the pieces need different flags, use one of the next two.

[Where the source comes from](dependencies.md) covers each kind of source.

## A CMake project of several libraries

When a project's own CMake build produces several libraries that link each
other, do not flatten it. Add a `[graph]` section and name the libraries you
want:

```toml
[project]
name = "libwebp"

[dependencies.libwebp]
type   = "git"
url    = "https://github.com/webmproject/libwebp.git"
tag    = "v1.6.0"
commit = "4fa21912338357f89e4fd51cf2368325b59e9bd9"

[graph]
dependency = "libwebp"
targets    = ["webpdemux", "libwebpmux"]
options    = ["-DBUILD_SHARED_LIBS=ON",
              "-DWEBP_BUILD_CWEBP=OFF", "-DWEBP_BUILD_DWEBP=OFF"]
```

That is all you state. From it RepliBuild works out the rest:

- **Which libraries are needed.** You name two; they link `webp`, which links
  `sharpyuv`. All four are built.
- **How each file is compiled.** Every file gets the flags, defines and include
  folders CMake gives it, including files that need their own (`-mavx2` on
  some files only).
- **What must be generated first.** Headers and sources the project writes
  during configuration or the build are produced before compiling.
- **What goes into which library.** Static and object libraries inside the
  project are folded into the shared libraries that use them, once each.
- **The order to build and load them in**, so that each library finds the ones
  it depends on, and finds *these* copies rather than others installed on the
  system.
- **Which types are the same type.** A struct, class or enum used by several
  of the libraries is one Julia type, so a value one library returns goes
  straight into another's functions:

```julia
include("julia/Libwebp.jl")

W, D = Libwebp.Webp, Libwebp.Webpdemux
# each library is a submodule; a type they share is one type in both
```

[CMake projects](graph.md) has the keys, the rules for shared types, and the
limits.

## Two recipes, one using the other

Sometimes two libraries should stay separate wrappers: each is useful alone,
each has its own recipe, and one links the other. The example is Dear ImGui's
GLFW backend, which calls GLFW.

Build the first library with its own recipe. In the second recipe, link it
from the first one's output folder:

```toml
# imgui/replibuild.toml
[link]
link_libraries = ["glfw", "GL"]
link_dirs      = ["../glfw/julia"]        # the folder where the glfw recipe wrote libglfw.so
```

Two things follow from `link_dirs`, and both matter:

- **The second library loads the copy you linked**, not another `libglfw`
  installed on the system. Without this, a session using both wrappers would
  hold *two* GLFWs: the GLFW wrapper would initialise one, the ImGui library
  would call the other, and a window created through one would be garbage to
  the other. With it, there is one copy and both modules use it.
- **Build order becomes yours to keep.** The first recipe must be built before
  the second, or the link fails with `cannot find -lglfw`.

In Julia, load both modules and pass pointers from one to the other:

```julia
include("glfw/julia/Glfw.jl")
include("imgui/julia/Imgui.jl")

win = Glfw.glfwCreateWindow(1280, 720, "demo", C_NULL, C_NULL)
Imgui.ImGui_ImplGlfw_InitForOpenGL(win, true)
```

What does **not** happen between two separate recipes is type sharing: each
module defines its own Julia types, so `Glfw.GLFWwindow` and a `GLFWwindow`
the ImGui module may define are different types. Pointers pass between them
because a pointer parameter accepts any pointer. If the two libraries exchange
structs by value, wrap them as one CMake project instead, where types are
shared.

**When you ship them**, put both wrappers' files in the same folder. Each
library looks beside itself first for the libraries it links, so a folder
holding `Glfw.jl`, `libglfw.so`, `Imgui.jl` and `libimgui.so` keeps working
wherever it is installed ([Build a package on it](using-wrappers.md)).

On Windows there is no such search path; the wrapper opens the linked
libraries itself before its own, from `link_dirs` and from beside itself. The
recipe is the same.

## Unrelated libraries in one session

Any number of wrappers can be loaded together. Each is its own module with its
own library, and they do not interfere:

```julia
L = RepliBuild.use("lua")
S = RepliBuild.use("sqlite")
Z = RepliBuild.use("zlib")
```

One guarantee is worth knowing, because it is what makes this safe. A library
RepliBuild builds always calls **its own** functions, even when another
library with the same function names is already loaded. Julia itself loads its
own copies of zlib, pcre2, curl, git2, gmp and others at startup. A zlib you
wrap does not get mixed up with Julia's: your module calls your build, at your
version, with your flags.

The same holds for failures. If one C++ library cannot start, its functions
raise when called, and every other loaded wrapper keeps working.

## Choosing

- The pieces need **one set of flags** and you want **one module**: several
  sources in one recipe.
- The project **has a CMake build** that produces the libraries: `[graph]`.
  This is the only shape that shares types between libraries.
- The libraries are **independent projects** and one happens to link the
  other: two recipes and `link_dirs`.
