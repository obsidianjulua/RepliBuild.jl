# API Reference

```@meta
CurrentModule = RepliBuild
```

The public API is small: a build lifecycle (`discover → build → wrap`), a local
registry with a community Hub (`register` / `use` / `search`), an experimental
ingest path for pre-built binaries, and an environment check. Everything else —
the compiler, wrapper generators, DWARF parser, MLIR/JLCS dialect bindings — is
internal and documented under [Internals](internals.md).

Every entry point takes a path to a `replibuild.toml` (or a project directory
that contains one) and is idempotent: unchanged inputs hit the content-hash cache
and return immediately.

## Build lifecycle

The four-stage pipeline. `discover` writes a `replibuild.toml`, `build` compiles
source to a `.so` + DWARF metadata, `wrap` emits the Julia module, `clean` removes
generated artifacts. `info` reports status without touching anything.

```julia
toml = RepliBuild.discover("path/to/project")   # scan → replibuild.toml
RepliBuild.build(toml)                           # clang → LLVM IR → .so + compilation_metadata.json
RepliBuild.wrap(toml)                            # DWARF + symbols → julia/<Module>.jl
```

`discover` accepts `build=true, wrap=true` to run the whole pipeline in one call.
Re-running `discover(force=true)` regenerates the config but **preserves the
hand-curated keys** that cannot be derived from source — `[types].templates` /
`template_headers`, `[wrap].varargs` / `macros` / `shim_headers` /
`cstring_owned` / `tier1`, and `[link].promote_statics` — so a forced re-scan
never eats your intent. Everything else is regenerated from the scan. See
[what discovery cannot know](config.md#2.-What-discovery-cannot-know) for the
information you are expected to supply by hand. `build(clean=true)` forces a
full rebuild past the cache.

```@docs
discover
build
wrap
clean
info
```

## Package registry and the Hub

`register` records a project in the local registry (`~/.replibuild/registry/`).
`use` is the one-call "give me a loaded module": it resolves dependencies, builds
(or serves the cached build from `~/.replibuild/builds/<hash>/`), wraps, and
returns the loaded Julia module. On a local-registry miss, `use`/`search` fall
through to the [RepliBuild-Hub](https://github.com/obsidianjulua/RepliBuild-Hub)
community registry.

```julia
Lua = RepliBuild.use("lua")          # build + wrap + load (cached), Hub on miss
Lua.luaL_newstate()

RepliBuild.search("xml")             # query the Hub by name/description/tags/language
RepliBuild.list_registry()           # what's registered locally
```

The build-cache key includes RepliBuild's own version and git revision, so
upgrading the generator rebuilds each package once with current codegen instead of
serving a stale wrapper. `scaffold_package` turns a registered project into a
distributable, `Pkg.add`-able Julia package. Two environment overrides:
`REPLIBUILD_HOME` relocates the registry, `REPLIBUILD_HUB_URL` points Hub
operations at a private mirror.

```@docs
register
use
search
list_registry
unregister
scaffold_package
```

## Ingest (experimental, C only)

For **C** libraries whose build systems RepliBuild's source pipeline can't
reproduce (autotools, CMake code generators, configure scripts), build the `.so`
yourself with `-g` and ingest it — RepliBuild skips compilation and runs only
DWARF extraction + wrapper generation. Ingested libraries dispatch through Tier 3
(`ccall`) exclusively; the C++ API surface of an ingested binary is **not**
supported (classes/methods/templates/virtual dispatch need the thunks only the
source build produces — at best the `extern "C"` surface works). Prefer the source
build; reach for ingest only when you must.

```julia
toml = RepliBuild.ingest("/path/to/libfoo.so",
                         headers=["/path/to/include"],
                         name="foo", language=:c,
                         build=true, wrap=true)
```

```@docs
ingest
```

## Environment

`check_environment` validates both toolchain buckets — the C bucket (JLL clang +
Julia's resident libLLVM, no external install) and the C++/Tier-2 bucket (system
LLVM/MLIR 21+, Clang, `mlir-tblgen`, CMake, and `libJLCS.so`) — and reports which
dispatch tiers are available, with OS-specific install hints for anything missing.

```@docs
check_environment
```

## What a generated wrapper exposes about itself

Beyond the wrapped API, every generated module carries facts about its own
generation. None of these are exported — reach for them qualified:

| Name | What it answers |
|---|---|
| `dispatch_tier(f)` | Which tier `f` **actually** dispatches through now: `:tier1` / `:tier2` / `:tier3`, `:unknown` (not wrapped), `:mixed` (methods on several tiers), `:deferred` (asked during precompilation — ask at runtime) |
| `DISPATCH_TIER` | `Symbol => Symbol` — the tier the generator **emitted** for each function |
| `struct_size(name)` | Byte size of a wrapped struct, from DWARF |
| `member_offset(name, member)` | Byte offset of a member within it |
| `STRUCT_SIZES` / `STRUCT_OFFSETS` | The tables behind those two |
| `TIER1_FUNCTIONS` / `TIER1_DECLARES` | C only: functions on a bitcode slice, and the symbols each slice declares |
| `BUILD_ID` / `BUILD_TARGET` / `BUILD_GENERATOR` | Identity of the library and generator the wrapper came from |

`dispatch_tier` and `DISPATCH_TIER` disagree exactly when a Tier-1 kernel demotes
(missing `slices/`, unresolvable declare); the function reports what will
actually run, the table what was intended. `dispatch_tier` forces the kernel to
generate, so it is not read-only and refuses to answer during precompilation.

## Debugging generated thunks

`RepliBuild.Debug` inspects what the Tier-2 pipeline emitted for a package —
including one this process never built — without running anything:

```julia
D = RepliBuild.Debug
D.thunks(pkg)                       # thunk symbols available to ask about
D.mlir_body(pkg, symbol)            # the generated dialect for one thunk
D.disassemble(pkg; symbol = s)      # objdump -dS: ops interleaved with machine code
D.dwarf(pkg; section = "line")      # the address → MLIR-line table
D.walk(pkg, symbol)                 # the common combination, one call
```

The disassembly path needs `ENV["REPLIBUILD_JIT_OBJDUMP"] = "1"` set **before
the wrapper loads**. The live-process counterpart — breaking inside the emitted
MLIR under gdb — is documented in
[Debugging a thunk](mlir.md#9.-Debugging-a-thunk).
