# Where the source comes from

RepliBuild builds the library itself, so the first thing a recipe must state is
what to build it from. There are five ways to give it source, and a recipe can
use several at once. Everything they supply is added together and compiled,
with one set of flags, into **one** library.

| The source is | You write | RepliBuild does |
|:--------------|:----------|:----------------|
| Files in the project folder | `[compile] source_files`, a list of files | Compiles exactly those files |
| A folder of source on disk | `[dependencies.x]` with `type = "local"` | Finds and compiles the source files in it, on every build |
| A git repository | `type = "git"` | Clones it, checks the commit, finds and compiles its source files |
| A release tarball | `type = "archive"` | Downloads it, checks its hash, unpacks it, finds and compiles its source files |
| An installed system package | `type = "system"` | Compiles nothing: takes include and link flags from `pkg-config` |

The three that find files for you do **not** behave the same way. Read the
section for the one you use.

To link against a library without compiling it, see
[Libraries it links](toml.md#Libraries-it-links). To build several libraries
that stay separate, see [Several libraries](libraries.md).

## Files in the project folder

The project folder is the one holding `replibuild.toml`. Its files are
compiled only when they are listed:

```toml
[compile]
source_files = ["src/core.c", "src/io.c", "vendor/util/util.c"]
include_dirs = ["src", "include"]
```

**`build` never searches the project folder.** The list is the whole truth:
a file not on it is not compiled, and a file on it that does not exist is an
error.

`discover` writes the list for you, once. It walks the whole project folder and
lists **every** `.c`, `.cpp`, `.cc` and `.cxx` file it finds, skipping only
folders named `build`, `.git`, `.cache`, `node_modules`, `.replibuild_cache`,
`.svn`, `.hg`, `__pycache__`, `venv` and `.venv`. It also writes
`include_dirs`: the project folder, its `include/` and `src/`, and every folder
that contains a header.

So after `discover`, read the list:

- **Delete what is not the library.** `discover` does not skip `tests/` or
  `examples/` here. A project with a `tests/t_core.c` gets that file in the
  list, and its `main` in the library.
- **Add a file later by adding a line.** New files are not picked up by
  `build`. Running `discover(path; force = true)` rewrites the list, and also
  resets `[compile] flags`
  ([what it keeps](config.md#What-a-forced-discover-keeps)).

Paths are relative to the project folder. Keep them relative, so the recipe
works on another machine.

## A folder of source on disk

When the source is a folder you would rather point at than list file by file
(a library checked out beside your project, a folder of generated files, a
vendored copy), make it a `local` dependency:

```toml
[dependencies.mylib]
type    = "local"
path    = "../mylib"              # relative to the TOML, or absolute
exclude = ["tests/", "examples/", "tools/"]
```

On every build RepliBuild walks that folder and compiles every `.c`, `.cpp`,
`.cc` and `.cxx` file in it. Nothing is copied: the folder is used where it
is, a new file in it is compiled by the next build, and an edit reaches the
next build.

Three things are particular to a local folder:

- **Tests and examples are compiled unless you exclude them.** Only folders
  named `build`, `.git` and `.cache` are skipped. A git or archive dependency
  skips `tests/` and `examples/` by itself; a local folder does not.
- **Headers are searched in the folder and in its `include/`.** Not in its
  `src/`. If the library's sources include headers from `src/` or anywhere
  else, add that folder to `[compile] include_dirs`.
- **Compiled libraries found inside it are linked.** A `libfoo.a` or
  `libfoo.so` anywhere in the folder adds its folder to the link search path
  and `foo` to the linked libraries. That is how you feed in a part that is
  already built. If the folder holds stale build output you do not want
  linked, `exclude` it or delete it.

A local folder is not checked against anything. What is in it is what gets
built.

Do not point a local dependency at the project folder itself (`path = "."`):
the walk would enter RepliBuild's own cache and output folders. List the
project's own files in `source_files`.

## A git repository

```toml
[dependencies.cjson]
type    = "git"
url     = "https://github.com/DaveGamble/cJSON.git"
tag     = "v1.7.18"
commit  = "acc76239bee01d8e9c858ae2cab296704e52d916"
exclude = ["fuzzing/"]
```

RepliBuild clones the repository into `.replibuild_cache/deps/cjson/` inside
the project, checks out `tag` (a tag or a branch), and compiles its source
files.

- **Always set `commit`**: the full 40-character commit that `tag` must be at.
  A tag is a name its owner can move to different code. With `commit` set, a
  moved tag stops the build instead of compiling something you did not review.
  Get it with `git rev-parse v1.7.18^{commit}` in a clone, or from the tag's
  page on the hosting site. To upgrade the library, change `tag` and `commit`
  together.
- **Headers** are searched in the clone's top folder, its `include/` and its
  `src/`.
- **Folders named** `test`, `tests`, `testes`, `example`, `examples`,
  `fuzzing`, `build`, `doc` and `docs` are skipped. Name anything else that is
  not the library in `exclude`
  ([what to look for](toml.md#Files-that-are-not-the-library)).
- A folder inside the clone that is itself a git repository is not entered.
- Submodules are not fetched unless you list them
  ([below](#Git-submodules)).

The clone is reused by later builds. It is fetched again when `url`, `tag`,
`commit` or `submodules` change, or after `clean(toml)`.

## A release tarball

```toml
[dependencies.quickjs]
type    = "archive"
url     = "https://bellard.org/quickjs/quickjs-2026-06-04.tar.xz"
sha256  = "b376e839b322978313d929fd20663b11ba58b75df5a46c126dd19ea2fa70ad2a"
exclude = ["qjs.c", "qjsc.c", "run-test262.c", "unicode_gen.c"]   # programs, not the library
```

- `sha256` is required. The download is checked against it before anything is
  unpacked, and a mismatch stops the build. Get it with
  `sha256sum file.tar.gz`.
- `url` may also be the path of a tarball already on disk, relative to the
  project.
- `subdir` names the folder inside the archive where the library's source
  starts, when it is not at the top: `subdir = "cdf39_1-dist/src"`. An archive
  whose top level is a single folder needs no `subdir`.
- Once unpacked it is treated exactly like a git clone: same header folders,
  same skipped folders, same `exclude`.

## An installed system package

```toml
[dependencies.zlib]
type       = "system"
pkg_config = "zlib"
```

Nothing is compiled. `pkg-config --cflags --libs zlib` supplies the include
folders, the link folders and the libraries, so that your own sources can
include and link the installed package. It needs `pkg-config` and the
package's development files on the machine, and the recipe then only builds
where they are installed.

## Side by side

| | Project files | `local` | `git` and `archive` |
|:--|:--|:--|:--|
| Which files are compiled | Those in `source_files` | All found in the folder | All found in the clone or archive |
| Found again on each build | No | Yes | Yes |
| Folders skipped without being asked | (none; the list is explicit) | `build`, `.git`, `.cache` | `test`, `tests`, `testes`, `example`, `examples`, `fuzzing`, `build`, `doc`, `docs`, `.git` |
| `exclude` applies | No: delete the line from `source_files` | Yes | Yes |
| Headers searched in | `include_dirs` only | The folder, `include/` | The top folder, `include/`, `src/` |
| Compiled libraries inside are linked | No | Yes | No |
| Checked against | Nothing | Nothing | `commit` / `sha256` |

Source files are those ending in `.c`, `.cpp`, `.cc` or `.cxx`.

`exclude` entries match a file's name, or any part of its path inside the
dependency: `"tools/"` removes a folder, `"qjs.c"` one file.

## Using several at once

Everything adds up. A recipe's library is built from its `source_files`, plus
every dependency's files, with the include folders of all of them. Real
recipes combine sources routinely:

**A fetched library plus files of your own.** The library comes from git; a
file beside the recipe adds to it.

```toml
[dependencies.hol]
type   = "git"
url    = "https://github.com/example/hol.git"
tag    = "v1.0"
commit = "…"

[compile]
source_files = ["shim/hol_keep.cpp"]      # yours, in the project folder
```

**A fetched library plus a folder of generated files.** llama.cpp's Vulkan
shaders are produced by a generator the build cannot run. They are generated
once, kept in a folder beside the recipe, and fed in as a second dependency:

```toml
[dependencies.llamacpp]
type    = "git"
url     = "https://github.com/ggml-org/llama.cpp.git"
tag     = "b10286"
commit  = "cd0fa6051abd04fd7f82ccb42ad378ec43885455"
exclude = ["common/", "tools/"]           # and about thirty more

[dependencies.vkshaders]
type = "local"
path = "vulkan"                           # generated .cpp files, kept in the recipe's folder
```

**A fetched library plus headers its build system would generate.** The
captured `config.h` lives in a folder beside the recipe and is found through
`include_dirs`
([Headers the build system generates](toml.md#Headers-the-build-system-generates)):

```toml
[dependencies.pcre2]
type   = "git"
url    = "https://github.com/PCRE2Project/pcre2.git"
tag    = "pcre2-10.45"
commit = "2dce7761b1831fd3f82a9c2bd5476259d945da4d"

[compile]
include_dirs = ["config"]                 # the captured config.h and pcre2.h
source_files = ["config/pcre2_chartables.c"]
```

The order of dependencies does not matter. When two of them supply a file with
the same function, the link fails on the duplicate: exclude one.

## Git submodules

A clone does not fetch submodules. For most libraries that is right. For a
repository that is mostly submodules, such as `boostorg/boost`, list the ones
you need:

```toml
[dependencies.boost]
type       = "git"
url        = "https://github.com/boostorg/boost.git"
tag        = "boost-1.86.0"
commit     = "<the 40-character commit of that tag>"
submodules = ["libs/filesystem", "libs/config", "libs/core", "libs/assert"]
exclude    = ["libs/config/checks/", "libs/config/tools/"]
```

- **Only the listed submodules are fetched**, each at the commit the main
  repository records for it. The main repository's `commit` therefore fixes
  every submodule too.
- A submodule's own submodules are not fetched and cannot be listed.
- Each fetched submodule is treated like the clone itself: its top folder,
  `include/` and `src/` are searched for headers, and its sources are
  compiled.
- `exclude` paths start from the main repository's top:
  `libs/config/checks/`.

### Which submodules to list

List every submodule whose headers your library includes, directly or through
another header. A list that is too short fails clearly, with a missing header
at compile time. A list that is too long compiles code that is not your
library.

- **Build and add.** Start with the library's own submodule. Build. Add the
  submodule that owns each header the compiler reports missing. Repeat.
- **Use the repository's own tool**, when it has one. In a checkout of Boost:

  ```bash
  python tools/boostdep/depinst/depinst.py -X test -X example -X doc filesystem
  git submodule status | grep -v '^-'      # what was fetched; leave out libs/headers and tools/
  ```

A fetched header-only submodule often still contains `.cpp` files that are not
library code: configuration checks, generators, benchmarks. Exclude them.
[Header-only libraries](header-only.md#Boost-and-other-large-template-libraries)
has more on Boost.

## Seeing what was fed in

`build` prints the totals, after one line per dependency:

```
  dependency: resolving ext (local)
RepliBuild | feed
  sources: 3  includes: 3
  compile: 3 files
```

If `sources` is not the number you expect, something was found that you did
not intend, or something was not found. A `main` function in the wrapper
(`names(M)`) means a program or a test was compiled in.

## Fetching again

```julia
RepliBuild.clean(toml)                 # deletes fetched sources too; the next build downloads again
RepliBuild.clean(toml; deps = false)   # keeps them; the next build checks them against the recipe
```
