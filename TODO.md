# TODO

Consolidated roadmap, updated **2026-07-17** after the multiple-inheritance +
vcall-producer work landed. Companion technical narrative:
[docs/updates/2026-07-17-multiple-inheritance-and-vcall.md](docs/updates/2026-07-17-multiple-inheritance-and-vcall.md).

The CLAUDE.md "Not Yet Built" ledger remains the source of truth for
*unbuilt-vs-bug* classification; this file adds priority and grouping on top.
On this project **absence is the default state, not a failure** — items below
are unbuilt pieces unless explicitly marked REGRESSION.

---

## 🔴 Known red (regression, not roadmap)

### stl_test wrapper generation silently omits the STL factory section
- **Status:** REGRESSION, pre-existing — bisect-proven NOT from the 2026-07-17
  MI/vcall work (reproduces on a pristine tree at `df97231` with zero
  working-tree changes).
- **Symptom:** generated `StlTest.jl` ends after `make_ints`; zero
  `create_std_vector_int` / `create_std_basic_string_char` definitions;
  `test/stl_test/verify.jl` errors 6 of 8 testsets with `UndefVarError`.
  Generation does not crash — the template-instantiation factory section is
  silently skipped.
- **Repro:**
  ```
  julia --project=. -e 'import RepliBuild; RepliBuild.discover("test/stl_test", force=true, build=true, wrap=true)'
  julia --project=. test/stl_test/verify.jl
  ```
- **Next:** bisect between the last known-green run and `df97231`; suspects are
  the Tier-2 GeneratorCpp changes in that commit or earlier. Likely a
  swallowed exception or a filter misclassifying the STL structs.

---

## Virtual dispatch / MI follow-ups (near-term, unblocked by 2026-07-17 work)

### vcall for struct-shaped virtual signatures
The vcall producer gates on scalar/pointer signatures because the vcall
lowering does no sret/packed ABI coercion. Virtual methods returning big
structs by value (or taking packed-struct by-value params) keep the
direct-call static-dispatch path. Fix = port `try_call`'s coercion block
(sret slot for >128-bit / packed returns, stack-slot for packed args) into
`VirtualCallOpLowering`. Rare shape for virtual methods; correctness today is
"static dispatch", not corruption.

### Virtual destructor dispatch / polymorphic deletion
Destructors are **deliberately** excluded from vcall routing — `Managed`
finalizers and the scope-RAII producer require exact-class destructor calls.
Consequence: polymorphic deletion through a base pointer needs a C++-side
helper (`void free_base2(Base2* b) { delete b; }` in the mi_test fixture is
the pattern). A wrapper-level story (vcall through the deleting-dtor slot for
handles whose static type is a base) is designable now but interacts with
finalizer semantics — think before building.

### Virtual inheritance
Still NOT modeled, now **rejected loudly** everywhere instead of silently
mis-handled (`DW_AT_virtuality` on the inheritance edge is captured by both
parsers; type_info emission warns and omits the base table; layout flattening
skips vbases with a warning). vbase offsets live in the vtable (VTT), not the
static layout — building this means vtable-resident offset reads in the
dialect lowering. Substantial; separate project.

---

## Dialect / IR-gen

### Op verifiers for the remaining ops
`scope`, `marshal_arg`, `vcall`, `type_info` have verifiers. Still none on
`ffe_call`/`try_call`, `get_field`/`set_field`, array ops — no known crash
paths; add opportunistically as producers mature (get/set_field are now the
only ops with no producer at all).

### Array-view: user-facing accessors + rank ≥ 2
The producer (`ir_gen/ArrayViewGen.jl`) emits rank-1 get/set thunks over
fixed-size primitive array members, executing through the JIT. Missing:
`GeneratorCpp.jl` does not emit the user-facing Julia accessor functions that
call these thunks, and `T [N][M]` members are skipped by the rank-1 regex.
The descriptor's dims/rank fields are populated but unused by the lowering —
bounds checking would build on them.

### is_struct_packed over-classification
`StructGen.is_struct_packed` returns true for ANY padding-free struct, so
aligned no-padding structs take the `marshal_arg` path unnecessarily.
Byte-identical results, wasted work. Real test = compare member offsets
against naturally-aligned offsets. (The scope-RAII gate deliberately runs
before it — keep that ordering.)

---

## ABI / platform

### Multi-target ABI classifier
sret/byval classification in `ffe_call`/`try_call` lowering is hardcoded
x86_64 SysV (`>128 bits → sret`, 64-bit ptrs). Win64/AAPCS not modeled. The
Hub is currently a single-target hub.

### Tier 1 per-function bitcode slicing
`Base.llvmcall` whole-module embedding segfaults at whole-library scale and
duplicates file-local statics (two independent reasons Hub configs pin
Tier 3). The real fix is per-function bitcode slicing, not whole-module
embedding. Tier 1 stays parked until then.

---

## Cross-repo

### RepliBuild-Hub pending TOML edits
The Hub repo (`~/Desktop/Projects/RepliBuild-Hub/`) still carries the three
TOML edits from the tinyxml2 session (uncommitted there). Push/PR when next
touching the Hub.
