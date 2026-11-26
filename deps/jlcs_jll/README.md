# The JLCS dialect as a binary

The dialect's sources are `src/mlir`. Everything here is about shipping the
library built from them, so a machine without a system LLVM/MLIR can run
Tier 2. The sources do not move: the binary is built from this repository at
a commit.

| File | What it is |
|---|---|
| `build_tarballs.jl` | The BinaryBuilder recipe, building this working tree. |
| `yggdrasil/build_tarballs.jl` | The same build from a pinned commit: the file that goes to Yggdrasil. |
| `write_artifacts_toml.jl` | Writes the package's `Artifacts.toml` entries from `./products`. |

Two numbers, both in `src/mlir`:

- `ABI_VERSION`: one integer. libJLCS exports it and RepliBuild refuses a
  library that reports another. Bump it with any change to what Julia
  generates for the dialect or calls in it (an op, an attribute, a thunk
  convention, a C API signature).
- `VERSION`: the binary's own version (`3.0.1`), independent of RepliBuild's.
  Bump it for every published build; a change of `ABI_VERSION` is a new major,
  and the major is always `ABI_VERSION` (CMake refuses anything else).

The library's soname is the dialect's: `libJLCS.so.<ABI_VERSION>`, a link to
`libJLCS.so.<VERSION>`. CMake sets it after `add_llvm_library`, which would
give the library LLVM's version (`libJLCS.so.23.1jl` from `LLVM_full_jll`, as
3.0.0 was). So a build against another LLVM keeps its name, and the name
moves only when `ABI_VERSION` does. A thunks library records the soname of
the libJLCS it was linked against: across a soname change every AOT package
is wrapped again, and RepliBuild refuses one that was not
(`JITManager.open_thunks_library`).

## Today: a release on this repository

RepliBuild downloads the library through the lazy `JLCS` entry of
`Artifacts.toml`, from a release of this repository.

1. Bump `src/mlir/VERSION` (and `ABI_VERSION` if the dialect's interface
   changed), and `version` in `yggdrasil/build_tarballs.jl` to match.
2. Build, from an environment with BinaryBuilder on Julia 1.12:
   `julia +1.12 --project=<env> deps/jlcs_jll/build_tarballs.jl`
3. `julia deps/jlcs_jll/write_artifacts_toml.jl https://github.com/obsidianjulua/RepliBuild.jl/releases/download/jlcs-v<VERSION>`
4. Create the release `jlcs-v<VERSION>` with the two `JLCS.v<VERSION>.*.tar.gz`
   files from `./products` (`gh release create ... --latest=false`), then
   commit `Artifacts.toml`. Upload exactly the files step 3 hashed.

Run step 2 from this directory: the tarballs land in `./products`, which is
where step 3 reads them.

**A change of `ABI_VERSION` is published from its own branch, before it
merges.** CI's Linux `runtests` cells have no local build and load the
published library, so the branch is red until its `Artifacts.toml` names a
release of the new number. The release is a tag of its own and `main`'s
`Artifacts.toml` still names the old one, so neither waits on the other:
build from the branch, publish, commit `Artifacts.toml` to the branch.

## Registering `JLCS_jll` in General

A JLL is not a repository you write: Yggdrasil builds the recipe and
publishes `JuliaBinaryWrappers/JLCS_jll.jl` for you. The recipe is open there
as JuliaPackaging/Yggdrasil#15725.

`yggdrasil/build_tarballs.jl` is that recipe, kept here so the two builds
stay one (`test_jlcs_abi_version.jl`). It is written to Yggdrasil's own rules
(`AGENTS.md` and `.agents/skills/reviewing-prs/SKILL.md` in their
repository; read them before changing it):

- The source is the commit of the dialect's release tag, with
  `tag="jlcs-v$(version)"`, which their build checks the hash against. So a
  new dialect version is tagged (the release above) before its recipe moves.
- Platforms start from `supported_platforms()` and each exclusion is a
  `filter!` with a comment saying what breaks. What builds: x86-64 and
  AArch64 Linux, glibc and musl.
- Every `Dependency` has a `compat` bound that the standard-library JLL of
  the oldest Julia in `julia_compat` satisfies.
- GCC 10, the version `LLVM_full_jll` 23 is built with.
- Two licences are installed: this repository's, and `LLVM_full_jll`'s, since
  LLVM's and MLIR's code is linked into the library.
- The recipe sets no soname: the dialect's CMake does. Moving the recipe to
  another `LLVM_full_jll` with the same sources is a new build of the same
  JLL version; a new soname (a new `ABI_VERSION`) is a new JLL version.

Before a recipe change goes to Yggdrasil, in a checkout of it, with its own
environment:

    cd J/JLCS
    julia --project=../.. build_tarballs.jl --verbose --deploy=local

then load `~/.julia/dev/JLCS_jll` in a fresh environment on Julia 1.10 (the
`julia_compat` floor) and on a current one, and run RepliBuild's Tier 2 tests
with `REPLIBUILD_JLCS=<JLCS_jll.libJLCS_path>`.

When the JLL is registered: add `JLCS_jll` to RepliBuild's `Project.toml`
with a compat bound on the version that carries this `ABI_VERSION`, and have
`Builder/JLCSLibrary.jl` take `JLCS_jll.libJLCS` where it takes the artifact
now (a local build still first). `Artifacts.toml` and the release route can
then go.

For a later dialect change the loop is: bump, tag and release here, move the
recipe's `version` and commit, open the Yggdrasil PR, then the compat bound.
The ABI check is what catches a RepliBuild paired with the wrong `JLCS_jll`
if a bound is ever set wrong.
