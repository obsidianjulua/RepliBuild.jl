# Use a library

The Hub is a set of ready-made `replibuild.toml` files for popular C/C++
libraries. `use` is the one call that fetches, builds, wraps, and loads.

```julia
using RepliBuild

Lua = RepliBuild.use("lua")
L = Lua.luaL_newstate()
Lua.luaL_openlibs(L)
Lua.luaL_dostring(L, "print(1 + 1)")
Lua.lua_close(L)
```

First call compiles from source (minutes). Later calls hit
`~/.replibuild/builds/<hash>/` and load instantly. The cache key includes
RepliBuild's own version, so upgrading the generator rebuilds each package once
instead of serving a stale wrapper.

## Browse the Hub

```julia
RepliBuild.search()          # everything
RepliBuild.search("xml")     # name, description, tags, or language
RepliBuild.search("cpp")
```

The index lives in [RepliBuild-Hub](https://github.com/obsidianjulua/RepliBuild-Hub).
Packages there today include lua, sqlite, cjson, zlib, zstd, lz4, pcre2, curl,
box2d, pugixml, tinyxml2, imgui, fmt, and more. Names are lowercase, no
underscores.

`use` checks your local registry first, then the Hub. A private Hub mirror is
`ENV["REPLIBUILD_HUB_URL"]`.

## Local registry

`discover()` registers the project automatically. You can also do it by hand:

```julia
RepliBuild.register("path/to/replibuild.toml")   # name from [project].name
RepliBuild.list_registry()
RepliBuild.unregister("myproject")
```

Local entries live in `~/.replibuild/registry/`. `ENV["REPLIBUILD_HOME"]`
relocates the whole tree (`registry/` and `builds/`).

```julia
RepliBuild.use("myproject")                 # cached build
RepliBuild.use("myproject"; force_rebuild=true)
```

## What `use` returns

A loaded Julia module. Call it like the C API, with the types the wrapper
generated. Hub packages skip the `include` step — that is only for [wrappers you
built yourself](guide.md).

```julia
C = RepliBuild.use("cjson")
C.cJSON_Parse("{\"a\": 1}")
```

What the generated functions, structs, and C++ handles look like:
[Call a wrapper](calling.md).

## When a Hub package is not enough

If the library is not on the Hub, or you need different flags, wrap it yourself:
[Wrap a library](guide.md). If you already have a working TOML, `register` it
and `use` it by name from then on.
