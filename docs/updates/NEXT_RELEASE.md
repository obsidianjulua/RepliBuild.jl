# Before the next registration

Written 2026-11-27. `main` is 4.6.0 plus the pull requests merged since (#140
to #151), and none of that is in a registered version. This is what has to be
true before the next one is registered.

A box is checked by the pull request, or the run, that makes it true, with
the evidence in that pull request. Nothing is registered while a box is open,
unless John strikes the line. It is a gate, not a roadmap: what waits until
after the release is at the end.

## What "stable" means in this release

- **C is stable through `ccall`** (John, 2026-11-27: "c stable is ccall").
  The promise covers a C library built from source and called through the
  plain calls its wrapper makes.
- Outside the promise: `[wrap.tier1]` (function slices run through
  `llvmcall`), `[link] enable_lto`, and `ingest`.
- C++ and Rust make no such claim in this release.

## 1. A full run on a Windows box

CI's Windows cells run the two suites that need no compiler. The suite that
compiles and calls libraries does not run there.

- [ ] `libJLCS` built again on the box (Set up, "Windows", step 6), and
      `check_environment()` reports both tiers live.
- [ ] `test/runtests.jl`, `test/unit.jl` and `test/devtests.jl` pass on the
      release commit. The release pull request records the Julia and LLVM
      versions and the three totals.
- [ ] Read by name, because none has run on a Windows host:
  - thread-local globals (`test_tls_globals_live.jl`): the accessors are
    exported another way there;
  - 128-bit integers (`test_type_mapping_live.jl`);
  - by-value container returns (`test_template_arg_records_live.jl`) and
    references to smart pointers (`test_placeholder_references_live.jl`):
    the sizes are libc++'s there, not libstdc++'s.
- [ ] The type matrix measured there once
      (`julia --project=. test/tools/cpp_type_matrix.jl`). No case kills the
      process. Its table goes into `CPP_TYPE_MATRIX.md` beside Linux's.
- [ ] Every failure is fixed, or named in the release notes as a limit on
      Windows. None is left unread.

## 2. C++ ownership

A C++ object returned by value is built in a buffer and its bytes are then
copied into a Julia value. Three things are wrong with that today:

- **An object that points into itself is invalid in Julia's hands**:
  `std::list`, `std::map`, `std::set`, `std::unordered_map`, a short
  `std::string` inside a `pair`, an `optional` or a `filesystem::path`,
  `std::wstring`, `std::u16string`. These are the matrix's ten WRONG cases.
- **Nothing ends it.** No destructor runs on the bytes.
- **A by-value pass lends where C++ moves**, and a value moved into a
  parameter can be used again afterwards.

- [ ] The model is decided (John's call) and written into `CLAUDE.md`: what
      holds such an object on the Julia side, what ends it, what a by-value
      pass does to it.
- [ ] No call hands Julia an object that is invalid where Julia holds it.
- [ ] Such an object is ended once, or the wrapper says in words that it is
      not.
- [ ] Using a value after it was moved is an error, never a read of a dead
      object.
- [ ] A unit test and a live test for each rule, each negative-checked.
- [ ] The Hub packages that return such types are wrapped again and their
      deep tests pass. The pull request counts them.

## 3. The type matrix

Last measured 2026-11-24: 191 cases; 98 work, 31 are opaque handles, 20 take
no plain Julia value, 5 are refused, 24 have nothing to pass, 10 are wrong, 3
are errors, none kills the process.

- [ ] Measured again on the release commit, on Linux. **No CRASH and no
      WRONG.**
- [ ] The three errors are fixed or explained in the document's reading.
- [ ] `docs/updates/CPP_TYPE_MATRIX.md` is that measurement, with the reading
      brought up to date.

Not a gate unless John says so: building a `std::function` from a Julia
function, which is most of the 24 cases with nothing to pass.

## 4. C through `ccall`

- [ ] `test/tools/c_ccall_coverage.jl` on the swept Hub loses no function
      against the 4.6.0 reading (10,315 of 10,316).
- [ ] `test/tools/abi_fuzz.jl` on the usual seeds fails nowhere but the
      expected `long double` cases.
- [ ] The thirteen Hub C recipes that turn slicing on pass their deep tests
      on plain calls too. The evidence for a `ccall` promise has to be a
      `ccall` run.
- [ ] The support page says what is stable and what is outside it, in the
      words of "What stable means" above.

## 5. The release itself

- [ ] **The Hub sweep, as a rebuild**
      (`julia --project=. test/tools/hub_sweep.jl <hub>`): two fixes since
      4.6.0 are made at build time. Every package is clean, or its failure
      is explained in the release pull request. boostmath's export-table
      check is red today and needs its decision first.
- [ ] **The release notes say what a user upgrading must do:**
  - a package with compiled thunks is wrapped again, because the library's
    name changed;
  - a Julia string is refused where a function may write into it, and a
    byte buffer is passed there;
  - thread-local globals and the two by-value return fixes reach a package
    on a rebuild;
  - `clean` keeps downloaded sources.
- [ ] `CHANGELOG.md`'s section is compiled from the merged pull requests'
      `## Changelog` headings, after the sweep's own fixes have landed.
- [ ] The version number is chosen (John) and `Project.toml` carries it.
- [ ] CI is green on the release commit: Tests on all three hosts, Tier 2 on
      LLVM 22, LLVM 23 and AArch64, and Docs.
- [ ] The installed copy works: `Pkg.add` from the repository at the release
      commit into a throwaway environment, then build, wrap and use a small
      C, C++ and Rust project.
- [ ] The docs site is being served before the tag is made.
- [ ] Registration, on John's go.

## Not gates

- `JLCS_jll` in General. The release ships with the download
  `Artifacts.toml` names, and moves to the JLL when it exists.
- A 4.5.x release carrying the same fixes: decided on its own.
- Building a `std::function` from a Julia function, two types with one name,
  Rust on Windows, macOS.

## Decisions that are John's

- The ownership model (section 2).
- The version number.
- boostmath's export-table check: the engine keeps hidden records out of the
  generated unit, or the recipe excludes them.
- Whether the Windows matrix measurement is a gate or only a reading.
- Whether a 4.5.3 follows.
