# TODO

Consolidated roadmap, updated **2026-07-17** after the multiple-inheritance +
vcall-producer work landed. Companion technical narrative:
[docs/updates/2026-07-17-multiple-inheritance-and-vcall.md](docs/updates/2026-07-17-multiple-inheritance-and-vcall.md).

The CLAUDE.md "Not Yet Built" ledger remains the source of truth for
*unbuilt-vs-bug* classification; this file adds priority and grouping on top.
On this project **absence is the default state, not a failure** — items below
are unbuilt pieces unless explicitly marked REGRESSION.

---

## ✅ Recently closed

### stl_test STL factory regression — FIXED 2026-07-17
Root cause was **not** codegen: `discover(force=true)` regenerated
`replibuild.toml` from scratch and destroyed the user-intent
`[types].templates` section (broken since `4117a8e`, 2026-06-02, when
devtests adopted always-regenerate). No templates → no instantiation stub →
no DWARF → no STL wrapper section, silently. Fixed two ways: `discover`
now **preserves user-intent TOML keys** across forced re-discovery
(`Discovery.PRESERVED_TOML_KEYS`: templates, template_headers, varargs,
macros, shim_headers, cstring_owned; pinned by
`test/test_toml_preservation.jl`), and devtests **seeds curated fixture
config** after each regeneration so fresh clones are deterministic. Narrative:
[docs/updates/2026-07-17-stl-test-regression.md](docs/updates/2026-07-17-stl-test-regression.md).

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

### Virtual inheritance — BUILT 2026-07-17 (residuals only)
Dynamic `<Derived>_as_<VBase>` upcasts (vptr-indirect, correct for every
dynamic type), `vbase_vtable_offset` extraction in both parsers, `type_info`
vbase table; diamond-proven in `test/vi_test/` 33/33 with the vcall producer
needing zero changes. Residuals: a dialect-level vbase upcast op (no producer
needs one — build when IR must adjust inside a thunk) and vbase member
flattening (deliberately never — offsets are dynamic; access = upcast + the
base's own accessors). Narrative:
[docs/updates/2026-07-17-virtual-inheritance.md](docs/updates/2026-07-17-virtual-inheritance.md).

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
