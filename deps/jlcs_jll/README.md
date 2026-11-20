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
- `VERSION`: the binary's own version (`3.0.0`), independent of RepliBuild's.
  Bump it for every published build; a change of `ABI_VERSION` is a new major.

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
publishes `JuliaBinaryWrappers/JLCS_jll.jl` for you.

1. Merge the dialect change to `main` here, and note the commit.
2. In `yggdrasil/build_tarballs.jl`, set the `GitSource` commit to it (the
   file in this repository carries a placeholder of zeros) and check `version`.
3. Open a pull request to `JuliaPackaging/Yggdrasil` adding that file as
   `J/JLCS/build_tarballs.jl`. Its CI builds both platforms; on merge
   `JLCS_jll` is registered.
4. Then, in RepliBuild: add `JLCS_jll` to `Project.toml` with a compat bound
   on the version that carries this `ABI_VERSION`, and have
   `Builder/JLCSLibrary.jl` take `JLCS_jll.libJLCS` where it takes the
   artifact now (a local build still first). `Artifacts.toml` and the release
   route can then go.

For a later dialect change the loop is steps 1 to 3 again, then the compat
bound. The ABI check is what catches a RepliBuild paired with the wrong
`JLCS_jll` if a bound is ever set wrong.
