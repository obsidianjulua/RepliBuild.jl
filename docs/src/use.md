# Registry

A project builds where it lives. You develop it there with `discover`, `build`
and `wrap`, and load `julia/<Module>.jl` with `include`. When the wrapper is
ready, `register` makes it loadable from anywhere: it vendors the build to
`~/.replibuild/` and adds a Julia package around it to your Julia environment.
Then either of these loads it, in any session:

```julia
M = RepliBuild.use("myproject")   # by registry name ([project].name)
using MyProject                   # as a package, from the Julia environment
```

A fresh install has an empty registry. `use` does not fetch Hub packages, and
`search` does not register anything.

## Register a project

```julia
RepliBuild.register("path/to/project/replibuild.toml")
```

That:

1. runs `build` and `wrap` on the project itself. When its build is current, the
   build is a cache hit.
2. copies the result (the wrapper, the library, its metadata) to
   `~/.replibuild/builds/<hash>/`, where the hash covers the TOML, the sources,
   the headers, the toolchain and RepliBuild itself;
3. writes a Julia package around that build, `~/.replibuild/packages/<Module>/`,
   and adds it to a Julia environment — by default the `@v#.#` environment, which
   every session can load from.

`discover(path; build=true, wrap=true)` does all of that in one call: it builds
and wraps the project, then registers it. A bare `discover(path)` registers
nothing, because there is no build to vendor yet.

Choose the environment with `env`:

```julia
RepliBuild.register(toml; env=:default)            # the @v#.# environment (default)
RepliBuild.register(toml; env=:active)             # the active environment
RepliBuild.register(toml; env="path/to/env")       # a specific environment
RepliBuild.register(toml; env=nothing)             # none; `use` still loads it
```

`ENV["REPLIBUILD_JULIA_ENV"]` (`default`, `active`, `none`, or a path) changes
the default. `register` only adds to environments: it never changes a package's
own Project.toml. An environment without RepliBuild gets the copy that built the
wrapper; one that already has RepliBuild keeps it.

`register(toml; build=false)` records the project without building it. The
first `use` builds and vendors it.

## What `use` loads

`use("name")` compares the project's inputs with the build it vendored:

- **Unchanged:** it loads the vendored build. No compiler runs.
- **Edited** (sources, headers, TOML, dependencies, the toolchain, or a new
  RepliBuild): it builds and wraps the project where it lives, vendors the new
  build, points the package at it, and loads it. You do not register again.
- **The project moved or was deleted:** it loads the last vendored build, with
  a warning. Register the project again from its new place to rebuild it.

A project registered by an earlier RepliBuild points at the copy of its sources
that version made inside the registry, and `use` keeps building that copy.
Register the project again, from the project, to move it to the new scheme.

```julia
RepliBuild.use("myproject"; force_rebuild=true)
```

`force_rebuild=true` builds the project from nothing first: it deletes the
project's `build/`, `julia/` and `.replibuild_cache/`, so git dependencies are
fetched again. Use it once after an upgrade whose notes say a fix needs a
rebuild ([Changelog](release-notes.md)).

`using MyProject` loads whatever build was vendored last: after an edit, run
`use("myproject")` (or `register` again) to vendor the new one, then start a new
session. Each build is vendored to a folder of its own, so a session that has the
old one loaded keeps it, and on Windows no open library is ever overwritten.

Call `use` at top level. It defines the module while your code runs, so a
function that calls `use` and then calls into the module fails with
`method too new to be called from this world context`. Inside such a function,
write `Base.@invokelatest M.f(x)`.

**Iterating in one session.** `use` is also the REPL loop. Edit the project or
its TOML, call `use` again, and in the same session it rebuilds, vendors the
new build to a folder of its own and returns a new module: about 2.5 s after an
edit on a one-header C library, 3 s on a C++ one. Keep the module it returns
(`M = RepliBuild.use("myproject")`). One you got earlier keeps calling the
build it was loaded from, so nothing you hold breaks. Including the project's
own `julia/MyProject.jl` cannot iterate like this: the rebuild in place has to
replace a library the session has loaded, which Windows refuses. Each edit
leaves its vendored build under `~/.replibuild/builds/`, and nothing deletes
the older ones (`unregister` removes the last), so clear them out by hand now
and then.

## Inspect and remove

```julia
RepliBuild.list_registry()          # each project, its vendored build, its environment
RepliBuild.unregister("myproject")
```

`unregister` removes what the registry made: the vendored build, the package,
and the package's place in the Julia environment. It never touches the project.
On Windows, a session that has loaded the vendored build (through `use`, or
`using` the package) still has its DLL open, so `unregister` there warns and
leaves that build's folder behind; delete it after the session ends.

`ENV["REPLIBUILD_HOME"]` relocates the whole tree (`registry/`, `builds/`,
`packages/`).

Missing name:

```
Package 'cjson' not in registry. Use RepliBuild.list_registry() to see available packages,
or RepliBuild.register("path/to/replibuild.toml") to add one. A RepliBuild-Hub config is
registered the same way, from a clone: <clone>/packages/<name>/replibuild.toml.
```

That is the empty-registry case. Wrap the library, or `register` its TOML.

## Hub catalog (optional)

[RepliBuild-Hub](https://github.com/obsidianjulua/RepliBuild-Hub) is a collection
of ready-made `replibuild.toml` files — lua, sqlite, cjson, zlib, box2d, and
others. `search` lists them. It does **not** install them.

```julia
RepliBuild.search()          # names, versions, tags
RepliBuild.search("xml")
```

To load one, register its TOML from a clone of the Hub, then `use` it or
`using` it. `search` prints the same steps:

```julia
RepliBuild.register("path/to/RepliBuild-Hub/packages/cjson/replibuild.toml")
C = RepliBuild.use("cjson")
```

`register` builds the package where its TOML lives, in the Hub clone: it fetches
the library's source at the pinned commit and compiles it, which can take
minutes.

`ENV["REPLIBUILD_HUB_URL"]` points `search` at a private mirror of that index.
It does not change `use`.

What the generated functions look like: [Call a wrapper](calling.md).
The wrap path if you are starting from source: [Wrap a library](guide.md).
