# Wrap your first library

This page takes a small C library from source to a Julia module you can call,
start to finish. Every later page is detail on one of its steps.

The work has a fixed shape:

1. **`discover`** scans the source folder and writes `replibuild.toml`.
2. **`build`** compiles the library. **`wrap`** writes the Julia module.
3. You **look at what is missing** and **add it to `replibuild.toml`**.
4. `build` and `wrap` again.

Step 3 is the only part that needs you. RepliBuild works out every function,
struct and enum from the compiled library. What it cannot work out is what the
compiler threw away or never saw: macros, the arguments you mean to pass to a
`printf`-style function, who frees a returned string, which files in the
repository are not the library. Those go in the TOML.

## The library

A folder with a header and a source file:

```
mathlib/
├── include/mathlib.h
└── src/mathlib.c
```

```c
/* include/mathlib.h */
#define ML_VERSION 3
#define ML_SQUARE(x) ((x) * (x))

typedef struct { double x, y; } ml_vec2;
typedef enum { ML_OK = 0, ML_EDOM = 1 } ml_status;

double    ml_dot(ml_vec2 a, ml_vec2 b);
ml_status ml_normalize(ml_vec2 *v);
char     *ml_describe(ml_vec2 v);      /* allocated; free it with ml_free */
void      ml_free(void *p);
int       ml_log(const char *fmt, ...);
```

`src/mathlib.c` implements them, and calls `sqrt` from the math library.

## 1. Discover

```julia
using RepliBuild

toml = RepliBuild.discover("mathlib")
```

```
RepliBuild | discover mathlib
  found: 1 sources, 3 include dirs
```

That writes `mathlib/replibuild.toml` and returns its path. The parts of the
file that matter:

```toml
[project]
name = "mathlib"
root = "."

[compile]
flags        = ["-fPIC"]
source_files = ["src/mathlib.c"]
include_dirs = [".", "include", "src"]

[link]
optimization_level = "0"
enable_lto         = false

[wrap]
language = "c"
```

`discover` found the source file and the include folders, and saw that the
library is C. It writes several more sections with default values; you can
leave them or delete them.

`source_files` is the list of what will be compiled, and from now on it is
yours: `build` compiles exactly the files on it and never searches the folder
again. `discover` lists every source file it finds, tests and examples
included, so on a real project read the list and delete what is not the
library.

If a `replibuild.toml` already exists, `discover` keeps it. Once you have
edited the file, it is yours: keep it in version control and do not run
`discover` on it again.

## 2. Build and wrap

```julia
RepliBuild.build(toml)
RepliBuild.wrap(toml)
```

```
RepliBuild | mathlib
  sources: 1  includes: 3
  compile: 1 files
  link: libmathlib.so (0.01 MB)
  dwarf: 5 functions, 2 types
  done: 1.98s
┌ Warning: Varargs function 'ml_log' has no overloads configured in [wrap.varargs]. Generating base wrapper only.
  wrap: Mathlib.jl
```

`build` compiled the library into `mathlib/julia/libmathlib.so`. `wrap` wrote
the Julia module beside it, `mathlib/julia/Mathlib.jl`. The module is named
after the project, in CamelCase.

The first `build` and `wrap` in a Julia session take several seconds longer
than later ones.

## 3. Load it and look

```julia
include("mathlib/julia/Mathlib.jl")
using .Mathlib

v = ml_vec2(3.0, 4.0)
ml_dot(v, v)                 # 25.0

r = Ref(v)
ml_normalize(r)              # ML_OK
r[]                          # ml_vec2(0.6, 0.8)
```

It works, and it is not finished. `names(Mathlib)` lists what the module
exports, and comparing that with the header shows what is missing:

| In the header | In the first wrapper | Why |
|:--------------|:---------------------|:----|
| `ML_VERSION`, `ML_SQUARE` | absent, with no warning | Macros do not exist in a compiled library. |
| `ml_log(fmt, ...)` | `ml_log(fmt)` only | Nothing says which extra arguments you want to pass. The warning above named it. |
| `ml_describe` | returns a `String`, and leaks the C buffer on every call | Nothing says the caller must free it. |
| `sqrt` from the math library | works here, by luck | The library was linked without `-lm`. |

A build that succeeds is not a finished wrapper. Reading the library's header
against this kind of list is the job, and [What goes in the TOML](toml.md) is
the full list.

## 4. Edit the TOML

Open `mathlib/replibuild.toml` and add what the library needs. This is the
whole file after the edit, with the inert sections removed:

```toml
[project]
name = "mathlib"

[compile]
flags        = ["-O2", "-fPIC"]          # an optimized build
source_files = ["src/mathlib.c"]
include_dirs = ["include"]

[link]
enable_lto         = false
optimization_level = "2"
link_libraries     = ["m"]               # -lm

[wrap]
language     = "c"
shim_headers = ["mathlib.h"]             # the header the macros below come from

[wrap.macros.ML_VERSION]                 # a constant macro
ret = "int"

[wrap.macros.ML_SQUARE]                  # a function-like macro, used on doubles
ret  = "double"
args = ["double"]

[wrap.varargs]                           # the calls to ml_log you want to make
ml_log = [["Cint"], ["Cstring", "Cdouble"]]

[wrap.cstring_owned]                     # ml_describe's result is freed with ml_free
ml_describe = "ml_free"
```

## 5. Build and wrap again

Start a new Julia session, so the old library is not loaded, then:

```julia
using RepliBuild
toml = "mathlib/replibuild.toml"

RepliBuild.build(toml)
RepliBuild.wrap(toml)

include("mathlib/julia/Mathlib.jl")
using .Mathlib

ML_VERSION()                                    # 3
ML_SQUARE(1.5)                                  # 2.25
ml_describe(ml_vec2(3.0, 4.0))                  # "(3, 4)", and the C buffer is freed
ml_log_Cint("n=%d\n", 7)                        # prints n=7
ml_log_Cstring_Cdouble("%s=%g\n", "pi", 3.14)   # prints pi=3.14
```

Only what changed is recompiled. `wrap` keeps the previous module as
`Mathlib.jl.prev`, so you can compare the two to see what an edit did.

`wrap` now warns that `[wrap.varargs]` is "unversioned". That warning is for
libraries fetched from git, where the overloads should name the commit they
were checked against ([What goes in the TOML](toml.md#Variadic-functions)).
A library in your own folder has no commit and keeps the warning.

## 6. Use it from anywhere

So far the wrapper is loaded with `include` and a path. To load it by name in
any session:

```julia
RepliBuild.register(toml)
```

```julia
# any later session
using RepliBuild
M = RepliBuild.use("mathlib")
M.ml_dot(M.ml_vec2(1.0, 2.0), M.ml_vec2(3.0, 4.0))
```

`register` also adds the wrapper to your default Julia environment as a
package, so `using Mathlib` works too. [Load it anywhere](use.md) has the
details, and says how to keep it out of your environment.

## The same thing in one call

```julia
RepliBuild.discover("mathlib", build = true, wrap = true)
```

That discovers, builds, wraps and registers. It is a quick way to see whether a
library builds at all. The result still needs step 3.

## C++

The steps are the same. `discover` sets `language = "cpp"` when it finds C++
sources, and the build then uses the system's `clang++`
([Set up](install.md)). What differs is what you add to the TOML:

```toml
[compile]
flags = ["-O2", "-fPIC", "-std=c++17", "-fstandalone-debug"]

[link]
enable_lto         = false
optimization_level = "2"

[wrap]
language = "cpp"

[types]
templates = ["std::vector<double>"]      # containers you want to create from Julia
```

`-fstandalone-debug` is worth adding to every C++ recipe from the start: without
it, classes and structs that are only declared in headers can come out with no
fields and no constructors ([What goes in the TOML](toml.md#Types-declared-only-in-headers)).

A C++ wrapper names every function after its namespace and class, and methods
take the object first:

```julia
r = shapes_make_rect(2.0, 3.0)      # shapes::make_rect
shapes_Rect_area(r)                 # shapes::Rect::area
shapes_Shape_rename(r, "door")      # takes a std::string: pass a Julia string
```

[Call the wrapper](calling.md) covers objects, strings, containers and
exceptions.

## Day to day

| To | Call |
|:---|:-----|
| Rebuild after editing the TOML or the sources | `build(toml)`, then `wrap(toml)` |
| Regenerate the module after a TOML edit that does not change the library | `wrap(toml)` alone ([which keys](toml.md#When-an-edit-takes-effect)) |
| See whether the library and module exist | `info(toml)` |
| Start over | `clean(toml)`, or `build(toml; clean = true)` |
| Start over, keeping the downloaded sources | `clean(toml; deps = false)` |

Three habits:

- **Load a rebuilt wrapper in a new Julia session.** A session keeps the
  library it loaded. On Windows a build cannot even replace a library that a
  running session has open: it fails with `Permission denied`.
- **Never edit the generated module.** `wrap` overwrites it. Change the TOML,
  or put your own code in your own module
  ([Build a package on it](using-wrappers.md)).
- **Keep `replibuild.toml` in version control.** It is the only file you write.

## Where things go

Everything stays inside the project folder:

| Path | Holds |
|:-----|:------|
| `replibuild.toml` | The recipe. The only file you edit. |
| `julia/` | The result: `Mathlib.jl`, `libmathlib.so` (`.dll` on Windows) and the data files the module reads. These files belong together. |
| `build/` | Intermediate files. |
| `.replibuild_cache/` | Compiled files reused between builds, and sources fetched from git. |

`build/` and `.replibuild_cache/` can be deleted at any time and should not be
committed.

## Next

- Your library is somewhere else (another folder, a git repository, a
  tarball), or is fed from several places:
  [Where the source comes from](dependencies.md).
- It is more than one library: [Several libraries](libraries.md).
- Go through your library's needs one by one:
  [What goes in the TOML](toml.md).
- It is header-only, or keeps its API in inline functions:
  [Header-only libraries](header-only.md).
- It is a CMake project that builds several libraries:
  [CMake projects](graph.md).
