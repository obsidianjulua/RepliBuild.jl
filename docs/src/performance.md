# Make it fast

A wrapper is generated code that RepliBuild checked against the library it
calls. It is built to add nothing to the cost of a call that the call does not
need. This page says what a call costs, and the three steps that take a
wrapper from "works" to "starts instantly and calls at full speed":

1. build the library optimized;
2. compile the C++ stubs ahead of time (`aot_thunks = true`);
3. put the wrapper in a package, and the package in a system image.

The numbers below were measured on one machine, on the small example libraries
this manual uses. They show the size of each effect. Measure your own library
before deciding anything.

## What a call costs

**A C function is called directly.** The generated function converts its
arguments and makes the call; there is no layer in between. It costs what the
same call would cost if you wrote it by hand:

| | Per call |
|:--|:--|
| `ml_dot(v, v)` through the wrapper | 4.4 ns |
| The same call written by hand with `ccall` | 4.3 ns |

**A C++ function is called through a stub.** A C++ call needs more than a
jump: the object pointer goes in the right place, a virtual method is looked
up in the object's table, a class passed or returned by value is laid out the
way the compiler lays it out, and a C++ exception is caught and turned into a
Julia one. RepliBuild generates a stub per function that does exactly that,
compiled to machine code. How the stub is compiled decides its cost, which is
the next section.

Functions that need none of it, such as a `noexcept` function taking plain
numbers and pointers, are called directly even in a C++ wrapper.

## Build the library optimized

`discover` writes an unoptimized build. For anything but debugging:

```toml
[compile]
flags = ["-O2", "-fPIC"]

[link]
optimization_level = "2"
```

Both lines matter: the first optimizes each file, the second the library as a
whole. Add the flags the library's own build would use for speed
(`-march=…`, `-DNDEBUG`); RepliBuild adds none by itself.

## C++ stubs compiled ahead of time

By default a C++ wrapper compiles its stubs each time it is loaded, in the
session that loads it. One line makes `wrap` compile them once, into a library
that ships with the wrapper:

```toml
[compile]
aot_thunks = true
```

`wrap` then writes `lib<name>_thunks.so` into `julia/`, beside the library.

It is faster in both ways that matter. On the example C++ library:

| | Stubs compiled at load (default) | `aot_thunks = true` |
|:--|:--|:--|
| Loading the wrapper, fresh session | 8.7 s | 0.5 s |
| A virtual method call | 35 ns | 7 ns |
| A function taking and returning structs by value | 40 ns | 13 ns |

Loading is faster because nothing is compiled. Calls are faster because the
wrapper calls each stub through a plain function pointer.

**Use it for anything you ship or run repeatedly.** The default is right while
you are still changing the recipe: `wrap` is quicker without it, and
[`RepliBuild.Debug`](debugging.md#Read-a-stub-without-running-anything) can
show a stub's machine code only for stubs compiled at load.

What it changes for the people who run the wrapper:
[Deployment](using-wrappers.md#Deployment).

## Put the wrapper in a package

Loaded with `include`, a wrapper is compiled by Julia in every session that
loads it. Inside a package it is precompiled once, like any Julia code.

[Build a package on it](using-wrappers.md) shows the layout. The generated
module is written to precompile: everything that depends on the running
process (opening the library, finding function addresses) happens when the
package is *loaded*, never when it is precompiled.

On the example library, a package holding the ahead-of-time wrapper goes from
`julia` starting to a C++ method's result in about one second:

```bash
julia --project=. -e 'using MyShapes; println(MyShapes.area(2.0, 3.0))'    # 1.1 s in all
```

`RepliBuild.register` gives you the same without writing a package: it wraps
the module in one and adds it to your Julia environment, so `using Mathlib`
loads a precompiled wrapper ([Load it anywhere](use.md)).

## Compile it into a system image

A system image is Julia with your packages already loaded and compiled. A
package that contains a wrapper can be built into one with
[PackageCompiler](https://github.com/JuliaLang/PackageCompiler.jl), and then
the wrapper, your code on top of it, and the compiled form of the calls you
make are all there when Julia starts.

```julia
# build_image.jl
using PackageCompiler

create_sysimage([:MyShapes];
                sysimage_path = "MyShapes.so",
                project       = @__DIR__,
                script        = joinpath(@__DIR__, "workload.jl"))
```

```julia
# workload.jl: runs while the image is being written. What it calls is compiled in.
using MyShapes

# While an image is being written, a module's start-up code has not run yet,
# so the wrapper has not opened its library. Run it before calling anything.
Base.invokelatest(MyShapes.Shapes.__init__)

Base.invokelatest() do
    MyShapes.area(2.0, 3.0)          # call what you want compiled into the image
end
```

```bash
julia build_image.jl
julia -J MyShapes.so --project=. -e 'using MyShapes; println(MyShapes.area(2.0, 3.0))'
```

The whole progression on the example library, as the time from starting
`julia` to printing a C++ method's result:

| How the wrapper is loaded | Start to result |
|:--------------------------|:----------------|
| `include`, stubs compiled at load | about 9 s |
| In a package, `aot_thunks = true`, precompiled | 1.1 s |
| The same package in a system image | 0.5 s |
| For comparison: `julia -e 'println(6.0)'` | 0.3 s |

With the image, a program that calls into C++ starts in nearly the time Julia
itself takes to start. The image is large (about 480 MB here, most of it Julia
and RepliBuild) and takes a few minutes to build.

Four rules, each learned from a real image:

- **Run the wrapper's `__init__` in the workload**, as above. Without it the
  wrapper's function pointers are still empty, the workload's first call fails
  with `AOT thunk pointer is null`, and the image is not built. It is safe to
  run twice; the image runs it again at every start.
- **Build the wrapper with `aot_thunks = true`.** Stubs compiled at load are
  compiled again in every session, image or not.
- **Do not leave live state in the workload.** Anything it stores in a global
  is frozen into the image: an "already initialised" flag, an open handle, a
  pointer. Keep the workload's variables inside a `let` block or a function,
  and do not load data you do not want baked in.
- **An image is a snapshot.** Julia runs the package inside the image and
  ignores later edits to its source. Rebuild the image after changing the
  package or re-wrapping the library.

A complete example, with a GUI, three wrappers and a workload that draws
frames, is
[LlamaChat](https://github.com/obsidianjulua/RepliBuild-Hub/tree/main/examples/LlamaChat)
in RepliBuild-Hub (`sysimage/build.jl` and `sysimage/workload.jl`).

## In your own code

- **Reuse `Ref`s in a loop.** `Ref(x)` for a pointer argument allocates. Make
  one outside the loop and assign into it: `r[] = x`.
- **Pass the struct, not a copy of its fields.** Structs go to C by value at no
  extra cost; there is nothing to gain from unpacking them.
- **Use the `_ptr` variant** of a string-returning function when you do not
  need a Julia `String`: it skips the copy
  ([Strings](calling.md#Strings)).
- **Keep type-stable code around the calls.** Every wrapper function has
  concrete argument and return types. Time is lost in the Julia code around a
  call far more often than in the call.
- **Batch across the boundary where the library lets you.** One call that
  processes an array beats a million calls that each process an element, in
  any language.

## Measuring

```julia
using BenchmarkTools
@btime ml_dot($v, $v)
```

To compare the two ways of compiling stubs, build the project twice, in two
folders, with and without `aot_thunks`, and time the same script against each.
