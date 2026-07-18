# Technical update — 2026-07-17 (late): virtual inheritance, diamond-proven

Third update of the day, completing the inheritance ABI arc
([MI + vcall](2026-07-17-multiple-inheritance-and-vcall.md) →
[stl_test root-cause](2026-07-17-stl-test-regression.md) → this). The MI-era
policy for virtual inheritance was detect-and-reject-loudly; it is now
supported.

## The problem in one sentence

A virtual base's subobject offset is **not a property of the class** — it is
a property of the *most-derived object*: a standalone `Left` places the
shared `VBase` at +16, a `Left`-inside-`Diamond` at +32 (tail-padding reuse
puts `d` at 28), and only the object's vtable knows which.

## The empirical pins (fixture first, code second)

`test/vi_test/`: `VBase` (virtual, data + virtual `tag()`), `Left`/`Right` :
virtual `VBase`, `Diamond` : `Left`, `Right`, overriding `tag()`.

1. **The DWARF location on a virtual edge is an expression, not a constant**:
   `DW_OP_dup, DW_OP_deref, DW_OP_constu 0x18, DW_OP_minus, DW_OP_deref,
   DW_OP_plus` = "this + \*(vptr − 24)" — the vbase-offset entry lives 24
   bytes below the vtable address point. Pinned in both parser input formats
   (readelf: `DW_OP_constu: 24`, decimal, semicolons; llvm-dwarfdump:
   `DW_OP_constu 0x18`, hex, commas).
2. **Diamond has NO direct inheritance DIE for VBase** — only `Left`@0 and
   `Right`@16. Transitive virtual bases must be resolved by walking
   non-virtual chains.
3. **Overrides of vbase methods re-home like regular MI**: `Diamond::tag`
   reports slot 3 in *Diamond's own primary vtable*. Class-local vcall
   coordinates keep working.
4. A latent extraction bug surfaced: the readelf constant-regex matched the
   `7` of "`7 byte block:`" on virtual edges — a bogus static offset,
   harmless only because every consumer gated on `virtual=true`. Now
   guarded and pinned dead by a test.

## What was built

- **Extraction (both parsers)**: the expression is parsed into
  `vbase_vtable_offset = −N` (`Compiler.jl` base dicts get
  `"vbase_vtable_offset"`; `DWARFParser.ClassInfo` gets a parallel
  `vbase_vtable_offsets` vector, compat constructors preserved). The static
  `offset` stays 0 for virtual edges — there is no static offset to record.
- **Dynamic upcast helpers** (`GeneratorCpp`): the upcast section now walks
  the base DAG. Non-virtual chains accumulate static offsets (MI unchanged);
  a virtual edge emits a **dynamic** helper:
  `p + unsafe_load(vptr + vboff)` after loading `vptr` from the (possibly
  statically pre-adjusted) subobject pointer. `Diamond_as_VBase` composes
  transitively: static +0 to `Left`, then dynamic. One helper text, correct
  for every dynamic type.
- **`jlcs.type_info` virtual-base table**: `vbaseNames`/`vbaseVtableOffsets`
  paired ArrayAttrs (attr-dict, default empty), verifier extended. Virtual
  bases never appear in the static `baseNames`/`baseOffsets` table. The
  omit-and-warn policy is gone.
- **Flattening**: vbase members are deliberately never flattened (dynamic
  offsets); access is upcast + the base's own accessors. The skip is now
  `@debug`-level, by-design.

## What did NOT need building

The MI ledger predicted virtual inheritance "means vtable-resident offset
reads in the dialect lowering." **False** — a pleasant consequence of the
architecture: since vcall dispatch coordinates are class-local and callers
upcast before invoking, the dynamic read lives entirely in the Julia-side
helper. The vcall producer, the dialect lowering, FunctionGen thunks, and
the RAII machinery (C1/D1 complete-object ctors/dtors handle vbase
construction by definition) all worked unchanged on the diamond.

## Proof (test/vi_test/verify.jl, 33/33; templates §8d, 87/87)

- **The canary**: `Left_as_VBase` — the same generated function — resolves
  +16 on a standalone `Left` and +32 on a `Left*` that is really a
  `Diamond`. Any static implementation gets one of them wrong.
- **Single copy**: `Diamond_as_VBase(pd) == Left_as_VBase(pd) ==
  Right_as_VBase(Diamond_as_Right(pd))` — three inheritance paths, one
  shared `VBase`.
- **Dispatch**: `VBase_tag` through the upcast returns 7 for the standalone
  `Left` and **1007 (Diamond's override)** for the Diamond-backed pointer —
  identical call sites, the vbase vtable does the rest. `Diamond_tag` via
  the re-homed primary slot agrees. Polymorphic delete through `Left*`
  destroys the whole Diamond.
- Extraction asserts pin the metadata exactly (−24 coordinates, no fake
  static offsets, `d@28` tail-padding reuse read faithfully).

## Verification matrix (end of day, third pass)

| Suite | Result |
|---|---|
| CI (`test/runtests.jl`) | 404/404 |
| JLCS producers | 26/26 |
| JLCS invariants | 10/10 |
| MLIR templates | **87/87** |
| mi_test | 31/31 |
| vi_test (new) | **33/33** |
| stl_test | 28/28 |
