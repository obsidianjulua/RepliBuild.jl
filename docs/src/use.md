# Registry

`use("name")` loads a project that is already in **your** local registry
(`~/.replibuild/registry/`). A fresh install has none. It does not fetch Hub
packages, and `search` does not register anything.

## After you wrap something

`discover()` registers the project (`[project].name`) when it writes the TOML.
From then on you can skip `include`:

```julia
toml = RepliBuild.discover("path/to/project", build=true, wrap=true)

MyProject = RepliBuild.use("myproject")   # name from [project].name
MyProject.some_function(...)
```

Call `use` at top level. It defines the module while your code runs, so a
function that calls `use` and then calls into the module fails with
`method too new to be called from this world context`. Inside a function, call
the module through `Base.invokelatest`.

## What `use` builds from

Registering copies the TOML, and the include directories and source files it
names relative to the project, into the registry: each project into a folder
of its own under `~/.replibuild/registry/`. `use` builds from that copy, never
from your tree. (A TOML whose `[project] root` is an absolute path is the
exception: nothing is copied, and `use` builds the original tree.) So:

- The first `use` builds the project again, in the registry folder. Later calls
  load the cached build from `~/.replibuild/builds/<hash>/` until the stored
  TOML, the copied sources or RepliBuild itself change. The key includes
  RepliBuild's version, so upgrading rebuilds once instead of serving a stale
  wrapper.
- After you edit the sources, run `register(toml)` again. Until then `use`, and
  `use(name; force_rebuild=true)`, build the copy made last time.
- Sources a `[dependencies]` entry fetches come from its pinned tag or commit,
  not from a copy.

```julia
RepliBuild.use("myproject"; force_rebuild=true)
```

`force_rebuild=true` ignores the cached build and builds the registry copy again
from nothing; git dependencies are fetched again. Use it once after an upgrade
whose notes say a fix needs a rebuild ([Changelog](release-notes.md)).

## Register by hand

If you wrote the TOML yourself (or copied one), register it before `use`:

```julia
RepliBuild.register("path/to/replibuild.toml")   # name from [project].name
RepliBuild.list_registry()
RepliBuild.unregister("myproject")
```

`ENV["REPLIBUILD_HOME"]` relocates the whole tree (`registry/` and `builds/`).

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

To actually load one, register its TOML, then `use` the name. `search` prints
the same steps:

```julia
# after cloning RepliBuild-Hub
RepliBuild.register("path/to/RepliBuild-Hub/packages/cjson/replibuild.toml")
C = RepliBuild.use("cjson")
```

The first `use` fetches the library's source at the pinned commit and builds it,
which can take minutes.

`ENV["REPLIBUILD_HUB_URL"]` points `search` at a private mirror of that index.
It does not change `use`.

What the generated functions look like: [Call a wrapper](calling.md).
The wrap path if you are starting from source: [Wrap a library](guide.md).
