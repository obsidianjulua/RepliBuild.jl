# Load it anywhere

While you are working on a wrapper you load it with `include` and a path.
`register` makes a finished one loadable by name, from any Julia session:

```julia
RepliBuild.register("path/to/mathlib/replibuild.toml")
```

```julia
# any later session
using RepliBuild
M = RepliBuild.use("mathlib")       # by the project's name

using Mathlib                       # or as a package
```

Nothing is registered until you register it. A new installation has no
wrappers, and `use` does not download any.

## What `register` does

1. It runs `build` and `wrap` on the project, where the project is. If the
   build is up to date this takes no time.
2. It copies the result to `~/.replibuild/builds/`, so the registered copy
   does not change while you keep working on the project.
3. It adds the wrapper to a Julia environment as a package, by default the
   environment every session loads from (`@v1.x`). That is what makes
   `using Mathlib` work.

`discover(path; build = true, wrap = true)` ends by doing the same.

### Choose the Julia environment

```julia
RepliBuild.register(toml)                          # the default environment
RepliBuild.register(toml; env = :active)           # the active environment
RepliBuild.register(toml; env = "path/to/env")     # a specific environment
RepliBuild.register(toml; env = nothing)           # none: only `use` loads it
```

`ENV["REPLIBUILD_JULIA_ENV"]` sets the default: `default`, `active`, `none`,
or a path. Set it to `none` while experimenting, so trial wrappers stay out of
your environment.

## What `use` does

`use("name")` checks whether the project changed since it was registered.

- **Not changed:** it loads the registered copy. Nothing is compiled.
- **Changed** (the TOML, the sources, the compiler, or RepliBuild itself): it
  builds and wraps the project again, registers the new result and loads it.
  You do not call `register` again.
- **The project folder is gone:** it loads the last registered copy and warns.

```julia
RepliBuild.use("mathlib"; force_rebuild = true)    # rebuild from nothing first
```

`force_rebuild = true` deletes the project's `build/`, `julia/` and
`.replibuild_cache/` first, so fetched sources are downloaded again. Use it
after a RepliBuild upgrade whose [changelog](release-notes.md) says a fix needs
a rebuild.

`using Mathlib` loads the copy registered last. After you edit the project,
call `use("mathlib")` once to register the new build, then start a new session.

Call `use` at the top level, not inside a function that then calls into the
module ([Call the wrapper](calling.md#Load-it)).

## Edit and reload in one session

`use` is also the fastest way to iterate. Edit the TOML or the sources, call
`use` again, and it rebuilds and returns a **new** module in a few seconds:

```julia
M = RepliBuild.use("mathlib")
# edit replibuild.toml …
M = RepliBuild.use("mathlib")       # rebuilt; M is the new module
```

Keep the module `use` returns. One you got earlier keeps working with the
build it was loaded from.

Each rebuild leaves a folder under `~/.replibuild/builds/`. Nothing deletes
the old ones, so clear that folder out now and then.

## List and remove

```julia
RepliBuild.list_registry()
RepliBuild.unregister("mathlib")
```

`unregister` removes the registered copy and the package. It never touches
your project. On Windows, a session that has the wrapper loaded keeps its
library open: `unregister` then warns and leaves that folder, which you can
delete once the session has ended.

`ENV["REPLIBUILD_HOME"]` moves `~/.replibuild` somewhere else.

A name that is not registered gives:

```
Package 'cjson' not in registry. Use RepliBuild.list_registry() to see available packages,
or RepliBuild.register("path/to/replibuild.toml") to add one.
```

## Ready-made recipes

[RepliBuild-Hub](https://github.com/obsidianjulua/RepliBuild-Hub) is a
collection of finished `replibuild.toml` files: lua, sqlite, cJSON, zlib,
box2d and others.

```julia
RepliBuild.search()              # list them
RepliBuild.search("xml")         # by keyword
```

`search` only lists. To use one, clone the Hub and register the recipe:

```bash
git clone https://github.com/obsidianjulua/RepliBuild-Hub
```

```julia
RepliBuild.register("RepliBuild-Hub/packages/cjson/replibuild.toml")
C = RepliBuild.use("cjson")
```

`register` downloads the library's source at the commit the recipe names and
compiles it, which can take minutes for a large library.

## CMake projects

A [CMake project](graph.md) registers as one unit, and `use` returns its top
module: `RepliBuild.use("libwebp").Webp`.
