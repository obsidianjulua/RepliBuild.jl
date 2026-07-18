# Technical update — 2026-07-17 (evening): the stl_test regression was config destruction, not codegen

Follow-up to
[2026-07-17-multiple-inheritance-and-vcall.md](2026-07-17-multiple-inheritance-and-vcall.md),
which flagged `test/stl_test` as KNOWN RED. Diagnosed, root-caused, and fixed
the same day. The interesting part: **the wrapper generator was never broken.**

## Diagnosis trail

1. `nm -D libstl_test.so | grep create_` → **0 symbols**. Build-side, not
   wrap-side: the factories were never in the library.
2. Current `build/` had `stl_api.ll` but **no `replibuild_templates.ll`** —
   the template-instantiation stub wasn't being generated.
3. Stub generation (`Compiler.generate_template_instantiations`) is gated on
   `config.types.templates`. The regenerated `replibuild.toml` had **no
   `templates` entry at all**.
4. The TOML that *looked* correct that morning (with
   `templates = ["std::vector<int>", …]`) was a gitignored **survivor from
   2026-06-02** — deleted from git at `e59d2b2`, lingering on disk, and
   overwritten by every `discover(force=true)` since.

## Root cause

`discover(force=true)` regenerates `replibuild.toml` from scratch;
`generate_config` emits the user-intent keys (`[types].templates` etc.)
**empty** — they cannot be derived from source. Nothing merged the existing
file's values back. So forced re-discovery silently destroyed hand-curated
config.

Commit `4117a8e` (2026-06-02, message "`.`") made devtests always
`discover(force=true)` on fixtures. From that day, every devtests run
destroyed stl_test's `templates` section: no templates → no instantiation
stub → no DWARF for `std::vector<int>`/`std::string`/`std::map` → no
`stl_methods` metadata → the wrapper *correctly* emitted nothing. Six weeks
of silent red, masked because interim sweeps cited CI/producers/templates/
invariants rather than the full devtests integration list.

This was a **systemic footgun**, not a fixture quirk: any user project
carrying `[wrap.cstring_owned]`, `[wrap.varargs]`, `[wrap.macros]`,
`[wrap].shim_headers`, or `[types].templates` lost them on re-discovery.
(The c_test "varargs has no overloads configured" warning is the same
mechanism's fingerprint — though c_test never committed varargs config.)

## Fix (two layers)

1. **`Discovery.jl`: user-intent preservation.** `discover` snapshots
   `PRESERVED_TOML_KEYS` — `[types].templates`, `[types].template_headers`,
   `[wrap].varargs`, `[wrap].macros`, `[wrap].shim_headers`,
   `[wrap].cstring_owned` — from the existing TOML before regeneration and
   merges them back after `save_config`, printing a
   `preserved: …` line. Policy: a regenerated **non-empty** value wins
   (future discovery may learn to derive some of these); empty/absent gets
   the preserved value. Adding a new user-intent TOML key means adding it to
   the whitelist, or force-rediscovery will eat it.
2. **devtests: curated fixture seeding.** Fixture TOMLs are gitignored
   (machine-specific paths), so a fresh clone has nothing to preserve. The
   integration loop now runs discover → `apply_curated_config` →
   build → wrap, seeding machine-independent curated sections
   (`CURATED_FIXTURE_CONFIG`; today: stl_test's templates) after every
   regeneration. Preservation keeps them alive between runs; seeding makes
   fresh clones deterministic.

## Verification

- `test/test_toml_preservation.jl` (new, in CI runtests): collect/restore
  round-trip, non-whitelisted keys excluded, non-empty-regenerated-wins,
  degenerate inputs (missing/garbage/empty TOML) are no-ops. 21/21.
- Two-pass live proof: (1) fresh-clone simulation — TOML deleted, discover,
  seed, build → 7 `create_std_*` factories, verify **28/28**; (2) forced
  re-discovery with **no** seeding — `preserved: types.template_headers,
  types.templates` printed, templates intact in the regenerated TOML,
  rebuild → 7 factories, verify **28/28**.
- Full CI: **404/404** (383 + 21 new).

## Moral

The "regression" survived two code bisects pointing at *nothing* — because
the breakage wasn't in code that runs, it was in **state the pipeline
destroys**. When a gitignored file is both load-bearing and regenerated,
its content is part of the system and needs an owner: either derive it,
preserve it, or seed it. Now all three are true for fixture config.
