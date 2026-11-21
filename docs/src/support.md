# Support and versions

What a RepliBuild release promises, which releases get fixes, where it runs,
and what to do after you update.

## Which releases get fixes

| Your Julia | RepliBuild | Fixes |
|:-----------|:-----------|:------|
| 1.13, 1.14 | 4.6 and later | Every fix lands here first and ships in the next release. |
| 1.12 | 4.5 | A fix that matters on Julia 1.12 is released as a 4.5.x version. |
| 1.11 and older | none | Use a supported Julia. |

`Pkg.add("RepliBuild")` picks the newest release your Julia can run, so you
never choose between these by hand.

Older releases get no new code. When an older release turns out not to work
with a newer version of a package it depends on, its entry in Julia's package
registry is narrowed so that the package manager never installs the two
together.

## Where it runs

| Platform | Status |
|:---------|:-------|
| Linux, x86-64 | Supported. Every test runs on each pull request, with LLVM 22 and 23. |
| Linux, AArch64 | Supported. Every test runs on each pull request, with LLVM 23. |
| Windows, x86-64 with MSYS2 | Supported for C and C++. The tests that need no compiler run on each pull request. [Rust crates](rust.md) are not run on Windows yet. |
| macOS | Not supported. `using RepliBuild` refuses to load. |
| Windows on ARM, other architectures | Not supported. |

[Set up](install.md) lists what each platform needs installed.

## What a version number means

- **A major release** (5.0) is the only one that may change how you call
  RepliBuild (`build`, `wrap`, `use` and the rest) or what a `replibuild.toml`
  that worked before does. The [Changelog](release-notes.md) says what to change.
- **Minor and patch releases** (4.7, 4.6.1) add and fix. Code that calls
  RepliBuild and recipes that worked keep working. A minor release may need a
  newer Julia; the package manager then keeps you on the release your Julia can
  run.

## The names in your wrapper

Your code calls the functions and types the wrapper defines, so their names
are part of what a release promises:

- A release that changes the name a wrapper gives something keeps the old name
  working as well. Code written against the old wrapper keeps running after you
  wrap again, and the [Changelog](release-notes.md) lists each change.
- An old name is dropped only in a major release, or when it never worked: a
  C function named `eval` or `include`, for example, names that every Julia
  module already has.
- Wrapping again with a newer release can add functions, including ones an
  older release left out, and can reword the message of a function it still
  cannot call.

## After you update RepliBuild

Some fixes are in how RepliBuild reads the built library, so they reach your
wrapper only when the library is built again. Updating RepliBuild alone does
not make `build` do that: it compares your sources and your compiler, finds
them unchanged, prints `cache: project unchanged`, and only the wrapper is
written again.

For a project you build yourself:

```julia
RepliBuild.clean("replibuild.toml"; deps=false)   # keeps downloaded dependencies
RepliBuild.build("replibuild.toml")
RepliBuild.wrap("replibuild.toml")
```

For a registered package, `use` notices the new RepliBuild and wraps again by
itself. To build it again too:

```julia
M = RepliBuild.use("name"; force_rebuild=true)    # downloads git dependencies again
```

Read the [Changelog](release-notes.md) entries between the version you had and
the one you have now.

## RepliBuild-Hub

Each package in [RepliBuild-Hub](https://github.com/obsidianjulua/RepliBuild-Hub)
is a recipe pinned to one tagged release of its library and to that release's
commit. Before each RepliBuild release, the Hub's packages are built and
wrapped with it and their tests are run.

A recipe stays on its pinned release until someone moves the pin. It does not
follow its library's new releases by itself, so `RepliBuild.search` shows the
version each package wraps.

## Reporting a problem

Open an issue at
[github.com/obsidianjulua/RepliBuild.jl/issues](https://github.com/obsidianjulua/RepliBuild.jl/issues)
with:

- the RepliBuild version, from `pkgversion(RepliBuild)`;
- the output of `RepliBuild.check_environment()`;
- your `replibuild.toml`, and where the library's source comes from;
- the full error or the wrong result, and the call that gave it.

When one function of a library cannot be wrapped yet, leave it out with
[`[wrap] exclude_symbols`](toml.md#Leaving-things-out) so the rest is written,
and report that function.
