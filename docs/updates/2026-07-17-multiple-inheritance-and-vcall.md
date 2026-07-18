# Technical update — 2026-07-17: Multiple inheritance + the vcall producer

One day, two connected pieces of work. The morning built **non-virtual
multiple inheritance end-to-end** (a "Not Yet Built" ledger entry since
2026-05-29); the afternoon built the **vcall producer** on top of it, so
virtual methods now dispatch through the vtable and **overrides are honored**.
Along the way, two pre-existing defects surfaced — one fixed, one bisected and
tracked. Everything below is live-verified; test counts at the end.

Roadmap counterpart: [TODO.md](../../TODO.md) (what deliberately remains
unbuilt).

---

## Part 1 — Multiple inheritance

### The gap, precisely

Single inheritance worked because every modeled base sat at offset 0. Three
independent layers each assumed that:

1. **Dialect:** `jlcs.vcall` read the vtable at `obj + vtable_offset` but
   passed `this` unadjusted — a secondary-base method received a pointer to
   the *derived* object's start and read the wrong subobject.
   `jlcs.type_info` carried a single `superType` string.
2. **Extraction:** BOTH DWARF parsers dropped `DW_AT_data_member_location` on
   `DW_TAG_inheritance` — every base was silently recorded at offset 0. The
   metadata for MI didn't exist anywhere.
3. **Wrapper:** `flatten_struct_members` appended base members with their
   base-relative offsets, raw.

### What was built

**Dialect** (`src/mlir/JLCSOps.td`, `JLCSPasses.cpp`, rebuilt against MLIR
22.1.6):

- `vcall` gained **`this_offset`** (I64, default 0 — all pre-MI IR keeps
  exact semantics). The lowering GEPs `args[0]` by it before the indirect
  call. `vtable_offset` stays relative to the *original* pointer; in the
  standard Itanium secondary-base case both equal the base subobject offset.
- `type_info` gained a **base table**: paired `baseNames`/`baseOffsets`
  ArrayAttrs riding in the attr-dict (default empty → old IR parses
  unchanged; `superType` remains the primary base).
- **Verifiers** on both (4 of the ops now have them): vcall requires the
  object-pointer operand; type_info rejects base-table arity mismatches and
  wrong element kinds.

**Extraction** (both parsers): the inheritance edge's
`DW_AT_data_member_location` and `DW_AT_virtuality` are captured.
`Compiler.jl` metadata gains `"offset"`/`"virtual"` per base and sorts
`base_classes` by subobject offset (Dict iteration order was
nondeterministic). `DWARFParser.ClassInfo` gained parallel
`base_offsets`/`virtual_bases` vectors with a 6-arg compat constructor.

**Consumers:**

- `generate_type_info_ir` emits the base table (omits it with a loud warning
  for virtual-inheritance classes — a static offset for a vtable-resident
  quantity would be a lie).
- `flatten_struct_members` rebases base member offsets by the base subobject
  offset (including DWARF4 `data_bit_offset` for bitfields), with
  deterministic collision renames for same-named members from different
  bases. Virtual bases are skipped loudly.
- **`<Derived>_as_<Base>` upcast helpers** are emitted for every class with a
  non-zero-offset base (accept raw `Ptr` or any `.handle` wrapper). This is
  what makes calling a secondary base's methods on a derived object correct —
  method wrappers take base-relative `this`.

**Policy:** virtual inheritance is *detected and rejected loudly at every
consumer* rather than silently mis-handled. Building it means vtable-resident
(VTT) offset reads in the lowering — tracked in TODO.md.

### Pre-existing defect #1, found and fixed: virtual methods were uncallable via Tier 2

The MI fixture was the first test ever to call a *virtual* instance method
through a wrapper's Tier-2 path — and it couldn't. Generated wrappers
dispatch via `JITManager.invoke("_mlir_ciface_<mangled>_thunk", …)`, but
`generate_jlcs_ir` routed virtual methods to the legacy vmethod-IR pass,
which emits `thunk_<mangled>` direct-call wrappers **nothing ever looks up**.
Every virtual instance method failed with `Symbol not found`. Fix:
manifest-needed virtual methods route through the FunctionGen ciface pass
like any other method.

---

## Part 2 — The vcall producer

Part 1's routing fix made virtual methods callable, but as **direct symbol
calls** — `p->Base2::get_b()` static semantics, overrides ignored. The
producer replaces those direct calls with real vtable dispatch.

### The empirical fact that collapsed the design

Before writing producer code, the fixture gained a `Derived::get_b` override
and its DWARF was dumped. Result (clang, llvm-dwarfdump):

- `DW_AT_vtable_elem_location` is **always the slot in the declaring class's
  own primary vtable** — Itanium re-homes an override of a non-primary-base
  method into the derived class's primary vtable. Concretely: `Base2::get_b`
  reports slot 2 (in Base2's vtable, after the virtual-dtor pair at 0/1);
  the `Derived::get_b` override reports slot 3 (in *Derived's* primary
  vtable, after its dtor pair and `get_a`) — **not** Base2's slot 2.
- Slot numbering counts the virtual-dtor pair and is address-point-relative,
  matching exactly how the vcall lowering indexes the loaded vptr.

Consequence: since every `ClassName_method` wrapper takes a
ClassName-relative `this`, dispatch coordinates are **class-local,
universally**: `vtable_offset = 0, this_offset = 0, slot = the method's own
slot`. The planned introducer-walk over the base table was unnecessary. MI
correctness comes from the caller-side `as_Base2` upcast plus the
this-adjusting thunks the compiler already planted in secondary vtables.

### What was built

- **`may_throw` UnitAttr on `vcall`.** Absent = the plain indirect call
  (pre-existing lowering, unchanged). Present = **indirect invoke + landing
  pad** with try_call's sentinel-continue EH model (`__cxa_begin_catch` →
  `jlcs_catch_current_exception` → `__cxa_end_catch`, zero-sentinel result),
  personality installed on the parent function. Built with the ODS
  indirect-invoke builder (null callee + `var_callee_type`) — deliberately
  avoiding the hand-rolled operandSegmentSizes path that historically
  SIGSEGV'd for indirect calls.
- **Producer** (`generate_jlcs_ir` → `generate_function_thunks`): a
  mangled-symbol → (class, slot, vptr-offset) table from DWARF vtable info.
  FunctionGen swaps the direct `try_call` for `jlcs.vcall` on instance
  methods with **scalar/pointer signatures only** (the vcall lowering does no
  sret/packed coercion; struct-shaped signatures keep the direct-call path).
  `may_throw` follows the same `may_throw && !noexcept` rule as try_call.
- **Destructors are excluded by design.** `Managed` finalizers and the
  scope-RAII producer require exact-class destructor calls, not dynamic
  dispatch. Polymorphic deletion through a base pointer uses a C++-side
  helper (`free_base2` pattern in the fixture).

### Proof

`test/mi_test/` (compiler-laid-out two-base fixture, wired into devtests):

- `Base2_get_b` on an upcast `Derived` returns the **override's** 1222 —
  through the secondary vtable's adjusting thunk — where static dispatch
  returned the base's 222.
- A C++-handed `Base2*`-that-is-really-`Derived` (caller has zero
  derived-type knowledge) dispatches the override.
- Mutation through a non-overridden virtual composes with override reads
  (`set_b(999)` → `get_b` = 1999); `Derived_get_sum` sees it.
- Non-virtual methods keep static semantics — the wrong-`this` canary
  (`Base2_double_b` on an unadjusted derived pointer reads `a` through
  Base2's code) stays pinned as documentation of what the upcast fixes.
- Polymorphic deletion through the base pointer works.

Dialect level: templates §8b JIT-executes MI `vcall` against a hand-rolled
two-vtable object (including the pinned pre-fix wrong-`this` semantics);
§8c JIT-executes the `may_throw` EH path and checks the emitted LLVM carries
`invoke`/`landingpad`/personality.

---

## Pre-existing defect #2, bisected and tracked: stl_test

The regression sweep caught `test/stl_test` red: the generated wrapper
silently omits the entire STL factory section (`create_std_vector_int` etc.;
verify errors 6/8). **Bisect-proven pre-existing**: reproduces on a pristine
tree at `df97231` with zero working-tree changes — not caused by any of the
above. Marked KNOWN RED in CLAUDE.md; repro + suspects in TODO.md.

---

## Gotchas worth remembering

- **ODS `DefaultValuedAttr<ArrayAttr, X>` wraps X in
  `$_builder.getArrayAttr(...)` itself.** Writing the full builder call as
  the default produces `[[]]` — an array containing one empty array — which
  trips the verifier on every parse. The default string is just `"{}"`.
- **Julia docstring placement:** inserting a docstring'd helper between an
  existing docstring and its function makes the first docstring document the
  second docstring → precompile error "cannot document the following
  expression".
- **Itanium tail-padding reuse is real and extracted correctly:** the
  fixture's `Derived::extra` lands at offset `0x1c` — inside Base2's tail
  padding — and the extraction reports it faithfully.

## Verification matrix (end of day)

| Suite | Result |
|---|---|
| CI (`test/runtests.jl`) | 383/383 |
| JLCS producers | 26/26 |
| JLCS invariants | 10/10 |
| MLIR templates (was 56) | **84/84** |
| mi_test (new) | **31/31** |
| stress_test / c_test pipelines | green |
| stl_test | red — pre-existing regression, tracked |
