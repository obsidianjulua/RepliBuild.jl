# Registry

`use("name")` loads a project that is already in **your** local registry
(`~/.replibuild/registry/`). A fresh install has none. It does not fetch Hub
packages, and `search` does not register anything.

## After you wrap something

`discover()` registers the project automatically (`[project].name`). From then
on you can skip `include`:

```julia
toml = RepliBuild.discover("path/to/project", build=true, wrap=true)

MyProject = RepliBuild.use("myproject")   # name from [project].name
MyProject.some_function(...)
```

First `use` after a wrap is usually a cache hit. Later calls stay cached at
`~/.replibuild/builds/<hash>/` until the TOML, the sources, or RepliBuild
itself change. The cache key includes the generator version, so upgrading
RepliBuild rebuilds once instead of serving a stale wrapper.

```julia
RepliBuild.use("myproject"; force_rebuild=true)
```

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
Package 'cjson' not in registry. Use RepliBuild.list_registry() to see
available packages, or RepliBuild.register("path/to/replibuild.toml") to add one.
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

To actually load one, register its TOML, then `use` the name:

```julia
# after cloning or downloading RepliBuild-Hub
RepliBuild.register("path/to/RepliBuild-Hub/packages/cjson/replibuild.toml")
C = RepliBuild.use("cjson")
```

`ENV["REPLIBUILD_HUB_URL"]` points `search` at a private mirror of that index.
It does not change `use`.

What the generated functions look like: [Call a wrapper](calling.md).
The wrap path if you are starting from source: [Wrap a library](guide.md).
