# Build a package on it

For a script, `include` the wrapper and call it. This page is the next step: a
Julia package of your own that contains a wrapper, loads fast, and gives your
users a Julia-style API.

A complete example is
[BoxWorld](https://github.com/obsidianjulua/RepliBuild-Hub/tree/main/examples/BoxWorld)
in RepliBuild-Hub, a physics sandbox built on the Box2D wrapper.

## Two layers

- **The generated wrapper**: everything in the project's `julia/` folder. Copy
  it into your package unchanged. Never edit it; regenerate it with `wrap`.
- **Your code**: the types, defaults, names and lifetimes you want your users
  to see, written against the wrapper.

Anything specific to how you want the library to feel in Julia goes in your
code. That is what lets you regenerate the wrapper when the library is
upgraded.

## Layout

Copy the whole `julia/` folder into your package, here as `lib/`:

```
MyPkg/
├── Project.toml            # RepliBuild and Libdl listed under [deps]
├── lib/                    # the project's julia/ folder, unchanged
│   ├── Box2d.jl
│   ├── libbox2d.so
│   └── …                   # the .json files beside them
├── src/MyPkg.jl
└── test/runtests.jl
```

```julia
module MyPkg

using RepliBuild

include(joinpath(@__DIR__, "..", "lib", "Box2d.jl"))
using .Box2d

# your API here

end
```

The generated module loads two packages, `RepliBuild` and `Libdl`, and your
package must declare both, or it fails to precompile with `package MyPkg does
not have Libdl in its dependencies`:

```julia
using Pkg
Pkg.activate("MyPkg")
Pkg.add(["RepliBuild", "Libdl"])
```

The wrapper finds its library beside itself, so the copy works wherever the
package is installed. Keep every file from `julia/` together; the module reads
the `.json` files when it loads.

A [CMake project](graph.md)'s wrapper is copied the same way, as a tree: the
whole `julia/` folder, with its per-library folders left where they are.
Include the top module.

## Deployment

The library file is built for one system. A package that ships it works on
that operating system and processor only. For several systems, either ship one
build per system, or have the package build the library when it is installed
([below](#A-package-that-builds-the-library)).

What a machine needs to **run** a wrapper it did not build:

| Wrapper | Linux | Windows |
|:--------|:------|:--------|
| C | RepliBuild, as a Julia dependency | The same |
| C++ | The same. Nothing else to install | MSYS2 CLANG64 with the MLIR package, `C:\msys64\clang64\bin` on `PATH`, and `Pkg.build("RepliBuild")` ([Set up](install.md#Windows)) |

On Linux, RepliBuild downloads its C++ support library by itself, and brings
a copy of `llvm-dwarfdump`, which a C++ wrapper runs when it loads, for
machines that have no LLVM. On Windows the support library is built on each
machine.

Do not strip the library. A C++ wrapper reads information from it when it
loads.

### Ship C++ wrappers with their stubs compiled

By default a C++ wrapper compiles its call stubs each time it loads. For a
package you distribute, have `wrap` compile them once:

```toml
[compile]
aot_thunks = true
```

`wrap` then writes a second library, `lib<name>_thunks.so`, into `julia/`.
Ship it with the rest. Loading is many times faster and calls are faster. [Make it fast](performance.md)
has the measurements, and how to go further with a system image.

## A package that builds the library

```julia
RepliBuild.scaffold_package("MyLibWrapper")     # writes ./MyLibWrapper/
```

That creates a package whose installation (`Pkg.build`) compiles and wraps the
library on the user's machine:

| File | What it is |
|:-----|:-----------|
| `Project.toml` | Depends on RepliBuild and `Libdl`, the two packages a wrapper loads. |
| `replibuild.toml` | The recipe. If a project named `MyLibWrapper` or `MyLib` is registered, its recipe; otherwise a template to fill in. |
| `deps/build.jl` | Runs `check_environment`, then `build` and `wrap`. |
| `src/MyLibWrapper.jl` | Includes the generated wrapper once it exists. |
| `test/runtests.jl`, `.gitignore` | A test stub. Built files are ignored by git. |

Before the first build, edit the template recipe: name the sources
(`[dependencies]` or `[compile] source_files`), set `[wrap] language`, and set
`[link] enable_lto = false`.

A package scaffolded by a RepliBuild 4.5 release or older has a `Project.toml` that
lists only RepliBuild, and fails to load once its wrapper exists. Add the
missing dependency: `Pkg.activate("MyLibWrapper"); Pkg.add("Libdl")`.

Every machine that installs such a package needs the full
[setup](install.md) for the library's language.

## Precompilation

The generated module precompiles as part of your package, and the package can
be compiled into a system image
([Make it fast](performance.md#Compile-it-into-a-system-image)). Two rules for
your own code:

- **Do not store addresses in constants.** A `const` computed from `dlsym`,
  `cglobal` or a pointer the library returned is evaluated when the package is
  precompiled, and is wrong in the next session. Use a `Ref` and fill it in
  your module's `__init__`.
- **Plain data is fine.** Sizes and offsets from `M.struct_size` and
  `M.member_offset` can be constants.

## If a C++ library fails to start

When a C++ wrapper cannot prepare its calls at load (the support library is
missing, or `llvm-dwarfdump` is not found), loading your package still
succeeds. The wrapper logs `Failed to initialize JIT for lib….so` with the
reason, and its C++ functions raise when called. Other wrapped libraries in the
same session are not affected.

`RepliBuild.check_environment()` on that machine says what is missing.

## Things to wrap once

The wrapper is a faithful copy of a C or C++ API. Your package hides the
awkward parts behind ordinary Julia functions.

**Objects.** Give each C++ object a Julia type that holds the pointer, creates
the object in its constructor, and releases it in a finalizer:

```julia
mutable struct World
    mem::Vector{UInt8}            # the object's memory; kept alive by this field

    function World(gravity)
        mem = zeros(UInt8, Box2d.struct_size(:b2World))
        Box2d.b2World_b2World(pointer(mem), Ref(gravity))     # construct in place
        w = new(mem)
        finalizer(destroy!, w)
        return w
    end
end

handle(w::World) = pointer(w.mem)

function destroy!(w::World)
    isempty(w.mem) && return                                  # already destroyed
    Box2d.b2World_destroy_b2World(handle(w))                  # run the destructor
    empty!(w.mem)
    return
end
```

- Keep the memory reachable for as long as the object lives. Here the field
  does that.
- Make destruction safe to call twice, and offer it as a function, so users
  can release resources at a known point. The finalizer is the safety net.
- Without `aot_thunks`, the first call of each C++ function does some one-time
  preparation, and a finalizer is a poor place for that. Either build with
  `aot_thunks = true`, or create and destroy one object in your module's
  `__init__` so the destructor has been called once before any finalizer runs.
- A class whose only destructor wrapper is `Class_delete_Class` frees its own
  memory. It must come from the library (`new`), not from memory you
  allocated ([Call the wrapper](calling.md#Construct-an-object-yourself)).
- Where the wrapper has a `Managed<Class>` type, it already does all of this:
  use it.

**Sizes and offsets.** Take every one from `M.struct_size` and
`M.member_offset`. They follow the library when it is upgraded; a number
written in your code does not.

**Missing inline functions.** A small function defined in a header outside any
class (`b2Dot(a, b)`) may not be in the wrapper. Either get it compiled in
([Header-only libraries](header-only.md#C:-a-keep-file)) or write it in Julia
against the struct's fields.

## Testing

RepliBuild tests that calls reach the library correctly. Test what your
package adds:

- create, use, destroy, and destroy again;
- the library's behaviour through your API;
- the finalizer: drop an object, then `GC.gc()`;
- for a C++ library, at least one call that reaches C++, so that a library
  that failed to start fails the test.
