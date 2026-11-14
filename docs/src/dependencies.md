# Dependencies

Where a library's source comes from, and how it gets into the build. The
key-by-key table is [`[dependencies]`](config.md#[dependencies]) in the TOML
reference; this page is how the pieces fit.

## A dependency is source, not a link

A `[dependencies.<name>]` entry is **source code that compiles into your
library**. RepliBuild fetches it, walks it for C and C++ files, adds its
include roots, and compiles every file with the project's own flags and
toolchain into the one `.so` the recipe builds. That is why the DWARF is
consistent across the whole library: one compiler, one flag set, one link.

A recipe's sources are the union of three things:

1. the files a scan of the project root finds (`discover` lists them),
2. `[compile] source_files`, which **adds** to that set, never replaces it,
3. every dependency's files.

To *link against* a library you do not want to compile — a system `libm`, a
library another package built — use [`[link] link_libraries` and
`link_dirs`](config.md#2.8-Link-libraries), or a `system` dependency (below).
To wrap several libraries that link each other, each as its own `.so`, use
[graph mode](graph.md).

## The four types

| `type` | The source is | Pinned by |
|:-------|:--------------|:----------|
| `"git"` | a clone of `url` at `tag`, under `.replibuild_cache/deps/<name>/` | `commit` (40 hex); a checkout that disagrees is a hard error |
| `"archive"` | a tarball at `url` (or a local path), unpacked under the same cache | `sha256`, required; checked before unpacking |
| `"local"` | a directory at `path`, used in place, never copied | nothing; it is your tree |
| `"system"` | no source: `pkg-config --cflags --libs <pkg_config>` adds `-I`, `-L` and `-l` | the installed package |

The default `type` is `"local"`, not `"git"`: an entry with a `url` and no
`type` does not clone.

```toml
[dependencies.cjson]
type    = "git"
url     = "https://github.com/DaveGamble/cJSON.git"
tag     = "v1.7.18"
commit  = "acc76239bee01d8e9c858ae2cab296704e52d916"
exclude = ["test", "fuzzing"]
```

## How a dependency is walked

For each dependency root (the clone, the unpacked `subdir`, the local `path`,
and each fetched submodule):

- `<root>`, `<root>/include` and `<root>/src` become include directories, so a
  file includes the library's headers the way a user of the library would.
  Headers anywhere else are named in [`[compile] include_dirs`](config.md#2.7-Include-roots).
- Every `.c`, `.cpp`, `.cc` and `.cxx` file under the root is compiled, except
  in directories named `test`, `tests`, `testes`, `example`, `examples`,
  `fuzzing`, `build`, `.git`, `doc` or `docs`, and except what `exclude`
  names.
- A directory holding its own `.git` is never walked as part of the root
  around it: a nested repository is a dependency of its own, or nothing.

`exclude` entries are matched against the path relative to the dependency's
root, generously: an entry hits when it equals the file name or is a prefix,
suffix or substring of the relative path ([§2.6](config.md#2.6-Excluding-what-is-not-the-library)).
Excluding aggressively is normal — a repository is usually more than its
library: command-line tools with their own `main`, benchmarks, code
generators, configure probes, other platforms' backends.

## Pinning

A tag is a name that can move; `commit` is the content. With `commit` set, the
clone is checked out at `tag` and its `HEAD` must be that commit, or the build
stops. A marker beside the clone records url, tag, the resolved commit and the
fetched submodules, so a changed pin re-fetches, an unchanged one is reused,
and a checkout someone moved by hand is caught.

```julia
RepliBuild.clean(toml)                 # removes the clone too; the next build fetches again
RepliBuild.clean(toml; deps = false)   # keeps the clone; the next build re-checks its pin
```

## Archives

A library that ships as a release tarball rather than a git repository (most
of what GNU and the national labs distribute) is an `archive` dependency. It is
downloaded and unpacked by Pkg's own download-verify-unpack engine; the hash is
checked before anything is unpacked, and there is no "latest": without
`sha256` the recipe does not load.

```toml
[dependencies.cdf]
type    = "archive"
url     = "https://spdf.gsfc.nasa.gov/pub/software/cdf/dist/cdf39_1/linux/cdf39_1-dist-cdf.tar.gz"
sha256  = "f9574315a028e8f71257fd55c375639c95fea607a5df70fb6629dc1e22adc68b"
subdir  = "cdf39_1-dist/src"          # the library's tree inside the archive
exclude = ["tools", "lib/libmain.c"]
```

`subdir` is the dependency's root inside the archive. Without it, an archive
whose top level is a single directory has that directory as its root.

## Submodules

A clone fetches no git submodules. For most libraries that is right; for a
**superproject** — a repository that is mostly submodules, such as
`boostorg/boost` with about 170 of them — the clone is an almost empty tree.
`submodules` lists the ones to fetch:

```toml
[dependencies.boost]
type       = "git"
url        = "https://github.com/boostorg/boost.git"
tag        = "boost-1.86.0"
commit     = "<40-hex commit of the superproject>"
submodules = ["libs/filesystem", "libs/config", "libs/core", "libs/assert"]
                                       # … and the rest of its closure (below)
exclude    = ["libs/config/checks", "libs/config/tools"]
```

How they resolve:

- **Exactly the listed paths are fetched, never all and never recursively.**
  Each is fetched shallow at the commit the superproject's tree records for it
  (its gitlink), so the superproject's `commit` pins every submodule too. A
  submodule's own submodules are not fetched, and cannot be listed: only paths
  the superproject itself records are valid, and any other is a hard error.
- A fetched submodule whose checkout is not its gitlink's commit is a hard
  error, like a pin mismatch. Delete the clone (or `clean`) to fetch it again.
- **Each submodule is one more source root of the same dependency.** Its root,
  `include/` and `src/` are include directories, its sources are walked with
  the same skipped directories, and everything compiles into the one library.
  `exclude` paths are relative to the superproject's root: `libs/config/checks`.
- Changing the list re-resolves the dependency, as a tag change does. A
  submodule not on the list contributes nothing, even when an earlier list
  fetched it.

### Which submodules a library needs

The list is the library's **dependency closure**: every submodule whose
headers the library's sources and public headers include, transitively. Too
short a list fails loudly (a missing header at compile); too long a list
compiles code that is not your library. Two ways to compute it:

- **With the superproject's own tool**, where there is one. Boost ships
  `tools/boostdep`; in a checkout of the superproject with the library fetched:

  ```bash
  python tools/boostdep/depinst/depinst.py -X test -X example -X doc filesystem
  git submodule status | grep -v '^-'      # the fetched set; drop libs/headers and tools/
  ```

- **By building.** Start with the library's own submodule, build, and add the
  submodule that owns each header the compiler reports missing. Every step is
  a fast failure, and the result is exactly what the compile reaches.

A fetched header-only submodule often carries sources that are not the
library either — configure checks, generators, benchmarks — which the walker
would compile; exclude them by their `libs/<module>/…` path. Where a library
stands alone outside its superproject (`boostorg/math` with
`-DBOOST_MATH_STANDALONE`), depending on it directly is simpler. Boost in
particular is covered in [Wrap a library](guide.md#Boost-and-other-large-template-libraries).

## Several dependencies

Every entry merges into the same library, so a project can combine an
upstream with a small vendored helper, or two upstreams that must be compiled
together. When something needs to know which dependency *is* the library —
`[wrap.varargs] proven_at`, which is checked against an upstream commit — it
takes the entry named like `[project] name`, or the only entry with a
`commit`.

## Local dependencies

`type = "local"` points at a directory you already have, relative to the
project or absolute (relative keeps the recipe portable). It is walked like a
clone, `exclude` included, and it is a valid root for the headers a macro shim
includes. Nothing is copied: edits to it reach the next build.

## The CMake project as a dependency

In [graph mode](graph.md), `[graph] dependency` names a `git`, `archive` or
`local` entry whose checkout is the CMake project to configure. It is resolved
the same way (cloned, pinned, submodules fetched, archive verified), but its
`exclude` does not apply: there, CMake lists the sources.
