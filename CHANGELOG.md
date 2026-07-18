# Changelog

All notable changes to RepliBuild.jl are documented in this file.

## Unreleased (post-3.0.0)

### Nested-type member attribution fix (found wrapping box2d for the Hub)

Members declared AFTER a nested type definition inside a class silently vanished from extracted metadata: clang emits a nested enum/struct DIE between the member DIEs (at first reference — `Type m_type; float m_radius;` puts the `Type` enum's DIE, enumerators, and null terminator between the two members), and `Compiler.jl`'s flat `current_struct_context` flipped to the nested type for its children and never restored the enclosing class. Every subsequent member attributed to the enum and was dropped — box2d's `b2Shape::m_radius` was the live casualty.

Fix: **depth-aware parent attribution**. readelf DIE headers carry the tree depth (`<2><2331>:`), previously discarded; the parser now maintains a depth-indexed context map (`context_by_depth`), and member/enumerator/inheritance/template DIEs at depth d attribute to the type last seen at depth d−1, with `current_struct_context` as fallback. Reproduced library-free per the hub-wrap guard (`NestedEnumHolder` in the mi_test fixture — the enum-typed member must come FIRST or clang hoists the nested DIE past all members and the bug doesn't trigger), fixed, and pinned by mi_test verify (38/38). Generalization confirmed: c_test 70/70, vi_test 33/33, box2d re-wrap recovers `m_radius`.


### Virtual inheritance: dynamic vbase upcasts, diamond-proven

The last unbuilt piece of the inheritance ABI. A virtual base's offset is **not static** — a standalone `Left` and a `Left`-inside-`Diamond` place the shared `VBase` at different offsets, and only the object's vtable knows which (the vbase-offset entry below the vtable address point). The MI-era policy of detect-and-reject-loudly is replaced with actual support:

- **Extraction (both parsers):** the `DW_AT_data_member_location` on a virtual inheritance edge is a DWARF *expression* (`DW_OP_dup, deref, constu N, minus, deref, plus` = "this + \*(vptr − N)"), not a constant. Both parsers now parse it into `vbase_vtable_offset = −N` (readelf and llvm-dwarfdump renderings pinned empirically). Fixed in passing: the readelf constant-regex previously matched the "7" of "`7 byte block:`" on virtual edges — a bogus static offset 7, latent because consumers gated on `virtual=true`.
- **Wrapper: dynamic `<Derived>_as_<VBase>` upcasts.** The helper reads the object's vptr and the vbase-offset entry at runtime: `p + *(vptr + vboff)`. The *same* helper is correct for every dynamic type — the vi_test canary shows `Left_as_VBase` resolving +16 on a standalone `Left` and +32 on a Diamond-backed one. Transitive virtual bases compose (Diamond has no direct `VBase` edge in DWARF; `Diamond_as_VBase` static-adjusts to `Left` then goes dynamic). Non-virtual MI upcasts unchanged.
- **`jlcs.type_info` gained a virtual-base table**: `vbaseNames`/`vbaseVtableOffsets` paired ArrayAttrs carrying the vtable-relative coordinate (virtual bases never appear in the static `baseNames`/`baseOffsets` table); verifier extended for the new pair. The old omit-everything-and-warn policy is gone.
- **No dialect lowering changes needed** — the ledger predicted "vtable-resident offset reads in the dialect lowering", but the class-local-coordinates + caller-side-upcast architecture means the dynamic read lives in the Julia wrapper and everything else composes: the vcall producer needed **zero changes** for vbase-declared methods, overrides of vbase methods re-home into the derived primary vtable exactly like regular MI (empirically: `Diamond::tag` slot 3), and complete-object ctors/dtors (C1/D1, already preferred by the RAII resolver) handle vbase construction.
- Layout flattening still (correctly, by design) skips vbase members — their offsets are dynamic; access goes through the upcast + the base's own accessors.

Proven live in `test/vi_test/` (diamond fixture, wired into devtests, **33/33**): the 16-vs-32 same-helper canary; all three views of a Diamond (`Diamond_as_VBase`, `Left_as_VBase`, `Right_as_VBase∘Diamond_as_Right`) resolve the ONE shared `VBase` (single-copy proof, including tail-padding reuse `d@28` extracted faithfully); `VBase_tag` through the vbase vtable dispatches `Diamond`'s override (1007) from a pointer whose caller has zero derived-type knowledge, while the standalone `Left` gets `VBase::tag` (7) from the identical call; polymorphic delete through `Left*` destroys the Diamond. Dialect-level vbase table parse/verify in templates §8d (**87/87**).

Regression state: CI 404, producers 26/26, invariants 10/10, templates 87/87, mi_test 31/31, vi_test 33/33, stl_test 28/28.

### stl_test regression fixed: discover(force) no longer destroys user-intent TOML config

The KNOWN RED flagged below is closed, and the wrapper generator was never broken. Root cause: `discover(force=true)` regenerates `replibuild.toml` from scratch and `generate_config` emits the user-intent keys empty — so forced re-discovery silently destroyed `[types].templates`, killing the instantiation stub → DWARF → STL wrapper chain. Broken since `4117a8e` (2026-06-02) made devtests always force-rediscover fixtures. Full narrative: `docs/updates/2026-07-17-stl-test-regression.md`.

- **`discover` now preserves user-intent TOML keys across forced re-discovery** (`Discovery.PRESERVED_TOML_KEYS`: `[types].templates`/`template_headers`, `[wrap].varargs`/`macros`/`shim_headers`/`cstring_owned`). Regenerated non-empty values win; empty/absent slots get the preserved value; a `preserved: …` line reports what carried over. This was a systemic footgun for any user project with hand-curated wrap config, not just fixtures.
- **devtests seeds curated fixture config** (`CURATED_FIXTURE_CONFIG`, applied between discover and build) so fresh clones — where the gitignored TOML doesn't exist yet — are deterministic.
- New CI guard `test/test_toml_preservation.jl` (21 tests). Live proof both ways: fresh-clone seeding path and preservation-only re-discovery path each yield all 7 `create_std_*` factories and stl_test verify **28/28**. CI total now **404**.

### vcall producer: virtual methods dispatch through the vtable (overrides honored)

`jlcs.vcall` moves from "exercised only by hand-written test IR" to **emitted by the codegen pipeline** — the first production consumer of the op, built directly on the MI groundwork below. Wrapper Tier-2 calls to virtual instance methods previously direct-called the mangled symbol (`p->Base2::get_b()` static semantics — overrides ignored); their thunks now read the vptr, index the slot, and call indirectly, so **a base-class wrapper invoked on a derived object reaches the override**.

- **Dispatch coordinates are class-local, universally** — an empirically pinned Itanium/DWARF fact (dwarfdump, clang, 2026-07-17): `DW_AT_vtable_elem_location` is the slot in the *declaring* class's own primary vtable, and Itanium re-homes overrides of non-primary-base methods into the derived class's primary vtable (fixture: `Base2::get_b` slot 2 in Base2; the `Derived::get_b` override slot 3 in *Derived's* vtable, after the dtor pair and `get_a`). Since every `ClassName_method` wrapper takes a ClassName-relative `this`, the producer always emits `vtable_offset = vptr offset (0), this_offset = 0, slot = method's own slot` — MI correctness comes from the caller-side upcast (`as_Base2`) plus the this-adjusting thunks the compiler already planted in secondary vtables. No introducer-walk needed.
- **`jlcs.vcall` gained `may_throw`** (UnitAttr, absent = plain indirect call, pre-existing lowering unchanged). When present, the lowering emits an **indirect invoke + landing pad** with the same sentinel-continue EH model as `try_call` (`__cxa_begin_catch` → `jlcs_catch_current_exception` → `__cxa_end_catch`, zero-sentinel result), personality installed on the parent function. Built with the ODS indirect-invoke builder (null callee + `var_callee_type`), steering clear of the historical hand-rolled-operandSegmentSizes SIGSEGV. The producer sets it under the same `may_throw && !noexcept` rule as `try_call`.
- **Producer gating**: `generate_jlcs_ir` builds a mangled-symbol → (class, slot, vptr-offset) table from DWARF vtable info; FunctionGen swaps the direct call for `jlcs.vcall` only for instance methods with scalar/pointer signatures (the vcall lowering does no sret/packed coercion — struct-shaped signatures keep the direct-call path). **Destructors are excluded by design**: `Managed` finalizers and the scope-RAII producer require exact-class destructor calls, not dynamic dispatch.
- Proven live in `test/mi_test/` (31/31): `Base2_get_b` on an upcast `Derived` returns the override's value (1222, not the base's 222) through the secondary vtable's adjusting thunk; a C++-side `Base2*`-that-is-really-`Derived` (caller has zero derived-type knowledge) dispatches the override; mutation through a non-overridden virtual composes with override reads; non-virtual methods keep static semantics (the wrong-`this` canary pins unchanged); polymorphic deletion through the base pointer works. Dialect-level EH path JIT-proven in templates §8c (84/84): emitted LLVM carries `invoke`/`landingpad`/personality and dispatch+`this`-adjustment behave identically under it.

**Known issue (pre-existing, unrelated — bisected to before this work):** `test/stl_test` wrapper generation silently omits the STL factory section (`create_std_vector_int` etc. missing; verify errors 6/8). Reproduces on a pristine tree at the previous commit. **Resolved the same day — see the stl_test section above** (config destruction in discover, not codegen).

Regression state: CI 383, producers 26/26, invariants 10/10, templates **84/84**, mi_test **31/31**, stress_test + c_test green (stl_test red for the pre-existing reason above).

### Multiple-inheritance ABI: dialect, extraction, layout, upcasts

Non-virtual multiple inheritance is now modeled end-to-end ("Not Yet Built" ledger entry since 2026-05-29; closed 2026-07-17). Virtual inheritance remains unbuilt and is now *detected and rejected loudly* at every consumer instead of silently mis-handled.

**Dialect** (rebuilt against MLIR 22.1.6):

- **`jlcs.vcall` gained `this_offset`** (I64, default 0 — all pre-MI IR keeps its exact semantics). The lowering GEPs `args[0]` by `this_offset` bytes before the indirect call, so a method dispatched through a non-primary base receives a pointer to *its base subobject*, matching how secondary-vtable entries are compiled under Itanium. `vtable_offset` stays relative to the original pointer (reads the secondary vptr as before); both offsets equal the base subobject offset in the standard secondary-base case. Previously the vtable was read from the right offset but `this` was passed unadjusted — every secondary-base virtual call read the primary base's data.
- **`jlcs.type_info` gained a base table**: paired `baseNames`/`baseOffsets` ArrayAttrs (attr-dict, default empty — old IR parses unchanged) recording each base class and its static subobject offset. `superType` stays as the primary base for single-inheritance consumers.
- **Verifiers on both** (`VirtualCallOp::verify`, `TypeInfoOp::verify`): vcall requires the object-pointer operand; type_info rejects base-table arity mismatches and wrong element kinds — same idiom as the scope/marshal_arg verifiers.
- JIT-executed proof in `test_mlir_templates.jl` §8b against a hand-rolled two-vtable MI object: secondary-base dispatch with `this_offset = 16` reads the secondary base's member (222), the pinned pre-fix semantics (`this_offset` omitted) observably read the primary base's member (111) through the same vtable slot, and mutation through the secondary base lands at the right byte. Templates suite 56 → **71/71**.

**Extraction** (both DWARF parsers): `DW_AT_data_member_location` on `DW_TAG_inheritance` — the base subobject offset, previously dropped by both parsers (every base silently recorded at offset 0) — and `DW_AT_virtuality` (virtual-base flag) are now captured. `Compiler.jl`'s metadata gains `"offset"`/`"virtual"` per base and sorts `base_classes` by subobject offset (Dict iteration order was nondeterministic); `DWARFParser.ClassInfo` gains parallel `base_offsets`/`virtual_bases` vectors (positional 6-arg construction still works via a compat constructor).

**Consumers**:

- `JLCSIRGenerator.generate_type_info_ir` emits the base table (omitted with a loud warning for virtual-inheritance classes — recording a static offset for a vtable-resident quantity would be a lie).
- `GeneratorCpp.flatten_struct_members` rebases base-class member offsets by the base subobject offset when flattening derived layouts (previously base-relative offsets were used raw — correct only while every base sat at offset 0), including DWARF4 `data_bit_offset` bitfield rebasing; same-named members from different bases get deterministic collision renames. Virtual bases are skipped loudly.
- **`<Derived>_as_<Base>` upcast helpers** are emitted for every class with a non-zero-offset base: `Derived_as_Base2(obj)` applies the static Itanium adjustment (accepts raw `Ptr` or any `.handle` wrapper). This is what makes calling a secondary base's methods on a derived object *correct* — the method wrappers take base-relative `this`.

**Tier-2 virtual-method thunk routing fix** (pre-existing, exposed by the MI fixture — the first test to ever call a *virtual* instance method through the wrapper's Tier-2 path): generated wrappers dispatch via `JITManager.invoke("_mlir_ciface_<mangled>_thunk", …)`, but `generate_jlcs_ir` routed virtual methods to the legacy vmethod-IR pass, which emits `thunk_<mangled>` direct-call wrappers **nothing ever looks up** — every virtual instance method was uncallable through `invoke()` (`Symbol not found`). Virtual methods the wrapper's thunk manifest declares are now routed through the FunctionGen ciface pass like any other method. Note the resulting call has statically-named-class semantics (`p->Base2::get_b()`); override-honoring dispatch through the vtable is the future vcall producer, now unblocked by `this_offset` + the base table.

Live-verified end-to-end on a compiler-laid-out two-base fixture (`test/mi_test/`, wired into devtests): metadata carries `Base1@0`/`Base2@16` (with `extra` at `0x1c` — Itanium tail-padding reuse, extracted correctly), type_info base table parses verifier-clean, `Derived_as_Base2` == +16, and live calls prove the layout — primary-base call unadjusted, secondary-base non-virtual `double_b` returns 444 via the upcast (and the pinned wrong-`this` call observably returns 2×`a`), virtual `get_b` through the newly-routed Tier-2 thunk returns 222, mutation through `Base2` observed via `Derived::get_sum`. `verify.jl` 27/27.

Regression state: CI 383, producers 26/26, invariants 10/10, templates 71/71, mi_test 27/27, stress_test + c_test full pipeline green.

### Tier-2 C++ dispatch fixes (found live rebuilding tinyxml2)

Clean-rebuilding the tinyxml2 Hub package end-to-end surfaced three real defects in the C++ JIT dispatch path — none had a covering test because no Tier-2 wrapper had been driven through a full construct→call→destruct cycle with these argument/return shapes:

- **Enum-by-value returns crashed** (`XMLDocument::Parse → XMLError`). The C++ generator resolved an enum return (DWARF key `__enum__X`) to a single-member struct `!llvm.struct<"X",(i32)>`; MLIR's `emit_c_interface` then used the **sret** convention (`void ciface(T* sret, void** args)`), but the Julia side calls the `@enum` back as a scalar (`T ciface(void** args)`) — the args pointer landed in the sret slot and the call dereferenced garbage. `ir_gen/FunctionGen.jl` now lowers enum returns to their bare underlying integer (returns by value, ABI-identical to the `@enum`); `GeneratorCpp.jl`'s `Any`-return resolution checks `__enum__` before the struct branches so the concrete `@enum` type reaches `invoke`.
- **`String` arguments to JIT thunks crashed.** `JITManager.invoke` packed `Ref(::String)`, handing the callee a pointer to the String *object* rather than its bytes — segfault on first dereference inside the C++ function. Now marshals `Ref(pointer(str))` with the String GC-preserved across the call, for both the value- and void-returning `invoke` variants.
- **Unresolved `Any` return now fails loudly.** `_invoke_call(::Type{Any}, …)` used to take the struct-sret path with `Ref{Any}()` (an undefined reference the JIT scribbled into → `UndefRefError`/corruption). It now `error()`s with the actual cause (return type unmapped — stale wrapper or missing DWARF mapping).

Live-verified: tinyxml2 rebuilt from cleared caches, then construct (placement `XMLDocument` ctor thunk) → `Parse` (`XML_SUCCESS`) → `FirstChildElement`/`GetText` (reads "42", "hello hub") → `SetText` → non-deleting dtor thunk, all through Tier-2 MLIR dispatch. Regression state unchanged: dialect templates 56/56, invariants 10/10, producers 26/26, CI 383.


### DWARF-driven producers for the JLCS RAII and strided-array ops

The two producer-less op families in the dialect ("Not Yet Built" ledger since 2026-05-29) now have production emitters, verified executing through the real MLIR JIT (`test/test_jlcs_producers.jl`, 26/26, devtests §11):

**Scope-RAII producer** (`ir_gen/FunctionGen.jl` + `JLCSIRGenerator._collect_class_raii`). Per-class destructor (D1-preferred) and copy-constructor (C1-preferred, `(const T&)` signature) symbols are resolved from DWARF metadata once per wrap. Three effects:

- `jlcs.type_info` now carries the resolved `destructorName` (was always `""`).
- **By-value params of classes with an emitted destructor are non-trivial for the purposes of calls under the Itanium C++ ABI — the callee expects a POINTER to a caller-owned temporary.** The thunk generator previously passed such classes as raw bits per SysV classification, a silent miscompile. Thunks now alloca a temporary, copy-construct it inside a `jlcs.scope` (`jlcs.ctor_call` with the copy ctor when resolvable, byte-copy when the copy is trivial), pass its address, and destruct it at scope exit — reverse order for multiple params, arity co-generated with the managed-pointer list. `try_call`'s sentinel-continue EH model means the normal path is the only path, so scope-exit destructor coverage is total.
- Symbol presence is the gate (a trivial destructor is never emitted as a symbol), so trivially-copyable classes keep their existing by-value path unchanged.

**Array-view producer** (`ir_gen/ArrayViewGen.jl`, new). Every fixed-size primitive array struct member (`double vals[4]`, `int32_t ids[16]`, …) gets a zero-copy get/set thunk pair that materializes an `ArrayView` descriptor (data/dims/strides/rank) over the member in place and accesses elements through `jlcs.load_array_element`/`store_array_element`. Rank 1 today; the descriptor already carries what bounds-checking and rank ≥ 2 need. This is also the first time these ops have ever been *executed* (the invariants suite only proved parse+lower) — they run correctly.

Remaining wiring (ledger): `GeneratorCpp.jl` does not yet emit user-facing Julia accessors calling the array thunks; multi-dimensional members are skipped.

Also recorded: `StructGen.is_struct_packed` classifies any padding-free struct as "packed" (`sum(sizes) == byte_size`), sending aligned no-padding structs down the `marshal_arg` path unnecessarily — benign but wasteful; the scope-RAII gate deliberately takes precedence over it.

### First JLCS op verifiers: `jlcs.scope` + `jlcs.marshal_arg`

The dialect's two known lowering segfaults are now parse-time diagnostics. `ScopeOp::verify()` rejects managed_ptrs/destructors arity mismatches (the old crash: `emitDestructors` indexed `managedPtrs` by destructor position and walked off the end) and non-symbol destructor entries; `MarshalArgOp::verify()` rejects memberTypes/juliaOffsets arity mismatches (the field loop indexed offsets by member position) and wrong element kinds. Malformed IR that used to SIGSEGV inside `translateModuleToLLVMIR`-adjacent lowering now fails `parse_module` with a real diagnostic.

`test/test_jlcs_invariants.jl` is fully green for the first time — **10/10, zero `@test_broken`** (both A2/B2 probes flipped from "expected crash" to asserting `:parse` rejection). The producer suite doubles as the positive control: production-emitted scope/marshal IR passes verification (26/26). Dialect rebuilt against system MLIR 22.1.6, templates 56/56.

The remaining ops (`vcall`, `type_info`, `ffe_call`/`try_call`, field/array ops) stay verifier-less — no known crash paths — tracked in the ledger to grow verifiers as their producers mature.

Regression state: dialect templates 56/56, invariants **10/10 (no test_broken)**, CI suite 383, producers 26/26.

## v3.0.0

C-generator audit release (2026-07-10). Ownership and ABI edges of the ergonomic layer are closed: the struct-by-value convenience footgun is removed, variadic calls are formally correct on x86-64 SysV, `char*` returns get one ownership-aware policy (`[wrap.cstring_owned]`) plus raw `_ptr` variants, macro shims survive `-fvisibility=hidden`, two silent-corruption edges are trapped or fixed (misaligned ≤16B blob params, bitfield tail overrun), and the registry build cache stops serving wrappers generated by outdated codegen.

**Registry note:** the last version registered in General is **v2.5.7**. Internal versions v2.5.8 and v2.5.9 were never registered — their changes ship as part of this release (their sections below stand as historical detail). The version jumps to 3.0.0 because the generated-wrapper API changes shape (see below), and semver puts breaking changes in the major number.

### Breaking changes since v2.5.7 (last registered version)

Generated-wrapper API — wrappers regenerate automatically on the next `use()`/`wrap()` (the cache is fingerprinted, below), but *calling code* may need updates:

- **Struct-by-value convenience overloads are gone** (C and C++). `f(unsafe_load(p))` patterns must pass the pointer or a `Ref` instead — every such call was UB-adjacent (the callee saw a pointer to a temporary copy; frees and stores corrupted memory, crash-proven on `cJSON_Delete`).
- **`char*` returns are `Union{String,Nothing}` instead of `String`-or-throw.** NULL now returns `nothing`; code relying on the old "returned NULL pointer" error must check `=== nothing`. Every such function gains an exported raw `<name>_ptr` variant (additive).
- **Nested-member structs resolve to named fields** instead of one `_data::NTuple` blob (v2.5.8). Code reaching into `x._data` on affected types must switch to the named fields; documented accessors are unaffected.
- **Multi-byte bitfield accessors changed shape:** getters return the smallest `UInt` covering `bit_size` (previously sized by offset+width span); setters accept negative integers with wrapping semantics (previously `InexactError`).
- **Globals with unresolvable DWARF types no longer get a value getter** — only `<name>_ptr()::Ptr{Cvoid}`. The old getter read memory as a boxed Julia pointer (garbage or crash).
- **Misaligned ≤16B opaque-blob params by value now throw an ABI trap** instead of silently corrupting arguments (float-bearing blobs already trapped since v2.5.8; this closes the all-integer packed case).

Programmatic API:

- **`WrapConfig` gained a positional field** `cstring_owned::Dict{String,String}` (before `dag`) — code constructing `WrapConfig` positionally must add it. TOML users are unaffected.

Behavioral (same signatures, corrected semantics):

- Vararg wrappers lower as true variadic calls (`@ccall` semicolon form) — float varargs no longer depend on leftover AL.
- `use()` cache keys include the generator fingerprint: every registered package rebuilds once after upgrading RepliBuild.
- Wrappers resolve their library sibling-first via `@__DIR__`; macro shims are pinned to default visibility (so `[wrap.macros]` works under `-fvisibility=hidden`).

New TOML surface (additive): `[wrap.cstring_owned]` — `func = "free_symbol"` declares a malloc'd `char*` return; the wrapper frees it through that symbol after copying.

### Struct-by-value convenience overloads removed (double-free footgun)

Both wrapper generators (`GeneratorC.jl`, `GeneratorCpp.jl` — the block was duplicated verbatim) emitted a "convenience" overload for every function with a `Ptr{Struct}` parameter: `f(x::MyStruct)` taking the struct **by value** and passing `Ref(local copy)` to the ccall. For any C function that frees, mutates-and-retains, or stores that pointer, this is undefined behavior — the callee receives a pointer into a temporary Julia-owned copy. Crash-proven 2026-07-10: the generated `cJSON_Delete(item::cJSON)` overload aborts with glibc `double free or corruption`, because `cJSON_Delete` calls `free()` on Julia GC memory. Retaining functions (e.g. `cJSON_AddItemToArray`, which stores the pointer into the array) corrupt silently instead of aborting — worse.

The overload class is **removed entirely** rather than gated: ownership is not recoverable from DWARF, so any `delete/free/destroy`-style name blocklist is guaranteed incomplete (a store-the-pointer function like `AddItemToArray` matches no such pattern). Ergonomics loss is negligible — the base wrapper's pointer params are `::Any` and already accept `Ref(x)`, pointers, and (for arrays) `Vector`s directly.

Survivors, pinned by test: the `Vector{T}` convenience overload for input-array params (`Ptr{Cdouble}` etc. under `GC.@preserve`) stays, and its `Cstring` returns follow the base wrapper's policy (see *Cstring return policy* below) instead of leaking a raw `Cstring`. The surviving path also gains the base wrapper's struct-return sentinel guard on the C++ side (previously it emitted the boxed-`Any` ccall return, a latent segfault).

Guarded by `test/test_convenience_overloads.jl` + `test/convenience_overload_test/` (devtests §10): a library-free fixture with a `free()`-taking `grip_free(Grip*)` traces compile → DWARF → wrap in a subprocess and asserts no by-value method exists, a by-value call refuses loudly (MethodError, never reaching `free()`), the pointer lifecycle round-trips, and the Vector path survives with `String`-aligned `Cstring` returns.

**Upgrade note:** code that called the by-value overloads (`f(unsafe_load(p))` patterns) must pass the pointer or a `Ref` instead — every such call site was UB-adjacent even when it appeared to work (the callee saw a copy, so mutations were dropped and stores/frees corrupted).

### Variadic call ABI (`generate_vararg_wrappers`)

Typed vararg overloads — and the fixed-args base wrapper — called variadic C functions through a **flat non-variadic ccall type tuple**; the code comment claimed a "Vararg marker for proper ABI" but none was ever emitted. On x86-64 SysV the callee's `va_start` prologue gates its XMM0–7 spill on AL, and only a variadic call site sets AL: int/pointer varargs worked de facto, but **float varargs (`sqlite3_mprintf_Cdouble`, gzprintf's Cdouble overload, …) only read correctly when leftover AL happened to be nonzero**. A live probe passing proved nothing — the failure is nondeterministic by construction.

All vararg wrappers now emit the `@ccall` semicolon form, `@ccall LIBRARY_PATH.var"sym"(fixed::T…; va_1::Cdouble)::Ret`, which lowers to a true variadic foreigncall: the callee is declared variadic in LLVM IR (`call i32 (ptr, ...)`) and the backend emits the AL setup (`movb $N, %al` observed in `code_native`). The base wrapper keeps a trailing `;` with zero varargs — the callee is still variadic, so AL must be set (to the count of vector registers used by the *fixed* args) even then. `var"…"` hardens the symbol position against keyword-shaped C identifiers. Per-arg vararg types are preserved (the old tuple form couldn't have expressed heterogeneous varargs correctly anyway).

Fixed in passing: the `"varargs..."` placeholder skip compared the *sanitized* param name (which mangles to `varargs_`), so it never matched — metadata paths that include the placeholder leaked `varargs_::` into generated signatures. The raw name is checked now.

Verified: a variadic C fixture built through the full pipeline — generated overload IR shows the variadic callee, float/int/zero-vararg calls return correct values. Regression: `test/test_varargs_emission.jl` pins the emission strings and the macro-expansion property (`Expr(:cconv, _, nreq)` with `nreq > 0` = variadic; the bug was `nreq == 0`).

### Registry build cache: generator fingerprint in `hash_config`

`hash_config` hashed TOML + sources + headers + *project* git HEAD but **not RepliBuild itself**, so `use()` served cached wrappers from old generators forever — observed: May 3/May 31 pre-v2.5.8 wrappers still live in `~/.replibuild/builds`, one of which crash-loads because Tier-1 `llvmcall` was baked in at lua scale. The hash now mixes in `_generator_fingerprint()` — package version **plus RepliBuild's own git HEAD** (dev checkouts move per commit; the version alone demonstrably doesn't get bumped every release). Every pre-existing cache entry misses once and rebuilds with current codegen on next `use()`; stale entries are orphaned, not deleted.

### Wrapper library resolution: sibling-first, baked path as fallback

Generated wrappers baked an absolute `LIBRARY_PATH` (for registry builds: into the shared `~/.replibuild/registry/julia/`), which the next build of any package overwrites — stranding every cached wrapper in `~/.replibuild/builds/<hash>/` even though each per-hash dir holds its own `.so` copy that the wrapper ignored. C and C++ wrappers now resolve the library **next to their own file first** (`joinpath(@__DIR__, basename(baked))`) and fall back to the baked absolute path; same for `THUNKS_LIBRARY_PATH`. Verified: renaming the baked `.so` away and reloading the cached wrapper works — the sibling copy wins.

### Cstring return policy: NULL → `nothing`, `[wrap.cstring_owned]`, raw `_ptr` variants

`char*`-returning wrappers previously converted via `unsafe_string` and **threw on NULL** — but a NULL `char*` is a value in C APIs (`cJSON_GetErrorPtr` before any error, `sqlite3_column_text` on a NULL column), not an exception. Worse, for functions returning **malloc'd** buffers (`cJSON_Print`, `sqlite3_mprintf`) the pointer was dropped after copying — an unfixable leak per call, since the caller never saw the pointer to free it.

The policy is now defined once (`_cstring_policy_lines` in `Wrapper/Utils.jl`) and spliced into **every** emission site — base wrappers, the surviving array-convenience overloads, vararg base + typed overloads, C and C++ generators — so it cannot drift between sites again:

- Return type is `Union{String,Nothing}`: NULL → `nothing`, else a copied `String`.
- **Ownership is declared in the TOML**, not guessed: `[wrap.cstring_owned]` maps a function to its library's deallocator (`cJSON_Print = "cJSON_free"`), and the wrapper frees the C buffer through that symbol after copying. Ownership of a returned `char*` is not recoverable from DWARF — same law as the convenience-overload removal, resolved the same way the varargs gap is: per-library facts live in the TOML.
- Every `Cstring`-returning function also gets an exported **`<name>_ptr` raw variant** (same args, returns the `Cstring` unchanged, no copy, no NULL check, never freed) for lifetime-sensitive callers and owned returns without a declared deallocator.

**Upgrade note:** call sites relying on the NULL throw must check `=== nothing`; docstrings now show `Union{String,Nothing}`.

### Macro shims pinned to default visibility

`[wrap.macros]` shims (`replibuild_shim_<NAME>` in the generated `replibuild_shims.c`) carried default attributes, so a project built with `-fvisibility=hidden` turned every shim **local** in the `.so` — and since the wrapper's function list comes from `nm -g --defined-only`, all declared macros silently vanished from the module. Live config hitting this: box2d3 (4 `[wrap.macros]` entries + `-fvisibility=hidden`). Shims are now emitted `__attribute__((used, visibility("default")))` — `used` additionally survives LTO internalization.

### ABI trap for misaligned ≤16B blob params by value

A packed struct with a misaligned member is MEMORY class under SysV **even at ≤16 bytes** — passed on the stack — while its opaque `NTuple{N,UInt8}` blob image classifies INTEGER and travels in registers. A by-value crossing of such a param silently fed the callee garbage; the existing guard only trapped the float-bearing (SSE-class) case, and `is_c_lto_safe` only gates *returns* (returns were already correct via the explicit-sret branch). The `blob_abi_offenders` param scan now also traps ≤16B blob params with any misaligned member, regardless of float content. Aligned all-integer blobs (INTEGER on both views) and >16B blobs (MEMORY on both views) remain callable, as before.

### Bitfield multi-byte accessors: exact byte-span assembly

Multi-byte bitfield getters/setters loaded a power-of-2 container (`UInt16/32/64`) at the field's byte offset — which can overhang the struct tail when the container is wider than the spanned bytes (e.g. a 17-bit field starting in the struct's last 3 bytes). The getter read out of bounds; the **setter wrote out of bounds into a heap Vector**. Accessors now assemble exactly `ceil((bit_offset_in_byte + bit_size)/8)` bytes with plain tuple indexing (no pointers), clamped to the DWARF `byte_size` with a generation-time warning on inconsistent DWARF. Setters also accept negative values via wrapping conversion (`v % UIntN`) instead of throwing `InexactError`.

### C generator cleanups

- **Dead packed-struct branch removed** (~50 lines): every `!layout_verified` struct with positive `byte_size` already became an opaque blob in the preceding branch, so the packed detection could never be reached. A comment marks the invariant.
- **Unresolved-type globals fail safe:** a global whose DWARF type didn't resolve emitted `cglobal(..., Any)` + `unsafe_load` — reading memory as a boxed Julia pointer (garbage or crash). Such globals now get only a `<name>_ptr()::Ptr{Cvoid}` accessor; the value getter is emitted only for clean, resolved types.
- **Callback docs no longer guess:** when fuzzy name-matching found no `[function_pointer_typedefs]` candidate, the docstring fell back to **the first typedef in the table** — documenting an arbitrary `@cfunction` signature users would build crashing callbacks from. No positive match now means the DWARF signature or "signature unknown", never a guess.

### Version markers aligned

`Project.toml` (stale at 2.5.8), `RepliBuild.VERSION` (stale at 2.5.7), and the `runtests.jl` pin had drifted three ways; all now say 3.0.0. The fingerprint reads `Project.toml` via `pkgversion`, so keeping it bumped now has teeth.

## v2.5.9

Dialect fix for C++ virtual dispatch: `jlcs.vcall` now translates all the way to LLVM IR instead of segfaulting at emit.

### `jlcs.vcall` emit crash (operandSegmentSizes)

`jlcs.vcall` *lowered* cleanly (the conversion pass returned success), but emitting the result — `emit_llvmir`, and therefore `emit_object`, which calls it first — **SIGSEGV'd inside `translateModuleToLLVMIR`** (`OperandRange::split` → `DenseArrayAttr::getSize`). `VirtualCallOpLowering` hand-built the indirect `llvm.call` via a raw `OperationState` and set `operandSegmentSizes = {1, nArgs, 0}` — a **3-entry** array. But `llvm.call` carries `AttrSizedOperandSegments` with **two** operand groups (`callee_operands`, `op_bundle_operands`); for an indirect call the callee pointer is the *first element of `callee_operands`*, so the correct value is `{1 + nArgs, 0}`. It also omitted `var_callee_type`. During translation the 3-entry array was split against a 2-segment op and walked off the end.

The lowering now uses the dedicated indirect-call builder `CallOp(LLVMFunctionType, ValueRange)`, which sets `operandSegmentSizes` and `var_callee_type` correctly. Value- and void-returning calls both emit; the indirect call comes out as `call … %slot(ptr %this, …)`.

**Scope / why this was latent:** no production producer emits `jlcs.vcall` — `generate_virtual_method_ir` resolves each virtual method to a *direct* `llvm.call @<mangled>` from DWARF vtable data, so the C++ AOT thunk path never hit this code. The op is exercised only by hand-written IR, and the existing `vcall` tests stopped at parse+lower (the prior code comment even claimed the AOT path "works"). So this was a real but off-production-path defect in a test-only op. It was **not** version skew: verified against system LLVM/MLIR 22.1.6 with a library-free minimal fixture, and a control op (`scope`) emits through the identical translator in the same dual-LLVM process.

Guarded by a new emit regression in `test/test_mlir_templates.jl` §8 (value + void): lower → `emit_llvmir` → assert the indirect call is present. Unblocks — but does not implement — the multiple-inheritance `this`-adjustment (a secondary-base `vcall` still passes `this` unadjusted; tracked under "Not Yet Built").

## v2.5.8

ABI-correctness release for the C path. Headline: structs with struct-typed members now resolve to named Julia fields instead of opaque byte blobs, closing a silent by-value miscompile. Plus C-wrapper ergonomics, a library-free ABI trace test, and verified compatibility with system LLVM/MLIR 22.1.6.

### Nested-Struct Member Resolution (C path)

The C generator emitted an opaque `_data::NTuple{N,UInt8}` byte blob for **any** struct with a struct-typed member — even when every member was itself a resolved type. For a struct ≤16 bytes whose members are floats, that byte image misclassifies under the x86-64 SysV ABI: the real struct travels in SSE registers (XMM), the byte blob claims INTEGER registers. Consequence when such a struct crossed `ccall` **by value**: returns came back as garbage (register noise, e.g. `1e-13`), and arguments fed the callee garbage — which in practice aborted the process (e.g. Box2D's own `b2IsValidAABB` assert → SIGILL). On box2d3 this was 58 of 664 symbols (the geometry/query cluster: `b2Body_GetTransform`, `b2Body_ComputeAABB`, `b2Body_GetMassData`, `b2World_OverlapAABB`, the `b2Compute*`/`b2Collide*` families).

`GeneratorC.jl` now runs an **exact-layout proof** before falling back to a blob:

- Every member is typed with a Julia field of exactly known `(size, alignment)` — primitives, pointers, enums, `NTuple{N,·}`, and structs already emitted with verified named fields (topological emission order guarantees member-before-container).
- The emitter then **proves** Julia's natural layout (explicit align-1 `_pad_N` fillers across DWARF offset gaps + natural field alignment) reproduces every DWARF member offset *and* the DWARF total `byte_size`.
- Proof passes → named fields, and the struct is registered as eligible to be an inline member of later structs. Any doubt → keep the opaque blob.

**Why:** the root cause was a member-resolution bailout, not the ABI classifier — `b2Vec2`/`b2Rot` resolved fine, the generator just refused to compose them. The proof is the safety boundary: **exact or opaque, never approximate**. Packed structs (unaligned members) and bitfield structs still blob correctly. On box2d3 the blast radius drops from 58 to **0**; all 99 previously-opaque structs resolve to named fields, including the 96-byte `b2WorldDef` with correct padding.

### ABI Safety Trap for Residual Float Blobs

A ≤16-byte float-bearing struct that *stays* opaque (genuinely unreproducible layout — packed floats, bitfields, unresolvable member types) and would cross `ccall` by value now generates a loud `error()` stub instead of a silently-corrupting call. MEMORY-class returns (>16 bytes, or unaligned) are unaffected — they still route through the explicit-sret branch, which is byte-exact even for a blob. Register-class float blobs are the only unfixable case, and they now fail closed.

### C Wrapper Ergonomics

- **`Cfloat`/`Cdouble` parameters loosened to `::Real`** (mirrors the existing `::Integer` widening), with a checked convert at the call site. `step(w, 1/60, 4)` and integer-literal float args now work instead of throwing `MethodError` on the strict `Cfloat` slot.
- **`with(s; field=value, …)` helper** emitted in every generated C module — the idiomatic way to customize an immutable `*Def`-style struct (`with(DefaultWorldDef(); gravity = Vec2(0, -10))`). Not exported; call as `Mod.with(...)`.
- **Blob accessor GC-preserve fix** — `getproperty` on byte-blob structs now roots the `Ref(getfield(x, :_data))` temporary that actually holds the storage, instead of preserving the immutable value `x` (a no-op). Closes a latent use-after-free under GC pressure.
- **Docstrings** for struct-returning functions show the resolved struct name instead of the `Any` metadata sentinel.

### Library-Free ABI Trace Test

`test/test_abi_nested.jl` + `test/abi_nested_test/` — a self-contained C fixture (nested-float, nested-int, packed, and array-of-struct members) that traces compile → DWARF → wrap → live by-value crossings in a subprocess, so an ABI break can never take down the test session. Asserts named-field resolution, exact round-trips through registers, MEMORY-class controls, and that packed structs refuse by-value crossings loudly. Wired into `devtests.jl` as section 8. This is the structural-proof gate: the bug was reproduced library-free before any generator change.

### Per-File IR Cache Correctness

The per-file IR cache (`<file>.ll` under the build dir) was keyed on source mtime alone (`needs_recompile`). Changing `[compile].flags`, `defines`, or include dirs in `replibuild.toml` does **not** move any source mtime, so the cache reported a hit and silently reused IR built with the *old* configuration — the resulting `.so` looks fine but was compiled wrong. The only workaround was `rm -rf .replibuild_cache build`. The project-level content hash already saw the toml change and ran the pipeline, but the per-file gate then independently decided "source unchanged, skip" — the two layers disagreed.

`needs_recompile` now also checks a **compile fingerprint** — a hash of the compile flags, defines, include-dir paths, `Base.libllvm_version`, and target triple — stored in a `<file>.ll.key` sidecar. A mismatch (or a missing key, i.e. a cache from before this fix) forces recompilation; the fingerprint is computed once per build and threaded through the parallel compile path. Editing one source still recompiles only that file (the fingerprint excludes individual source content — that stays the mtime's job); a flag/define/include change correctly busts the whole set. Guarded by `test/test_cache_invalidation.jl` (devtests §9): a `-fvisibility=hidden` toggle must recompile with no manual cache clear *and* drop the now-hidden internal symbol from the export table.

### LLVM / MLIR 22.1.6 Compatibility

System LLVM/MLIR moved 22.1.5 → 22.1.6 (a patch release). The JLCS dialect (`src/mlir`) was clean-rebuilt against it with **zero source changes** — TableGen and all six translation units compile unchanged, and the binary links the same `libMLIR.so.22.1` SONAME. Verified functionally: `test_mlir_templates.jl` 50/50 (CStructs, sret, RAII ordering, virtual dispatch, TypeInfoOp), `test_jlcs_invariants.jl` 6 pass + 2 expected `@test_broken` (the known missing op verifiers). The C-bucket text-IR cleaning shims in `Compiler.jl` are keyed on the LLVM *major* version (e.g. `ptrtoaddr → ptrtoint`), so a patch bump introduces no new opcodes and needs no new shim.

### Upgrade Notes

No API breaks. Generated C wrappers change shape for libraries with nested-member structs: affected types now expose **named fields** instead of a single `_data::NTuple`, and their accessors move from `getproperty` byte-extraction to real struct fields. Code that reached into `x._data` directly (never the intended interface) must switch to the named fields; code using the documented field/accessor names is unaffected and gains correctness. If you updated system MLIR, rebuild the dialect: `cd src/mlir && ./build.sh`.

## v2.5.7

Stabilization release on top of the v2.5.6 DAG diff work. Focus: cross-LLVM compatibility, sret correctness on the C path, dialect op fixes, and wiring of orphaned test suites into CI.

### Per-Language LLVM Toolchain Routing

`LLVMEnvironment` now resolves toolchain binaries through a per-language bucket rather than a single global PATH lookup. The `:c` bucket targets the LLVM version that matches Julia's internal libLLVM (Tier 1 `llvmcall` + LTO bitcode must be ABI-compatible with what Julia loads), while the `:cpp` bucket targets the system LLVM/MLIR (currently 22+) needed for the JLCS dialect.

- `LLVMEnvironment.resolve_tool(name, language)` is the new entry point — replaces the unscoped form at every call site (Compiler.jl, ThunkBuilder.jl)
- IR sanitize pass on link to strip attributes/metadata that the older internal LLVM rejects when consuming bitcode produced by a newer system clang
- Documents the dual-bucket reality in README — system LLVM 21+ for the C/C++ pipeline, internal Julia LLVM (18–20) for `llvmcall` consumption, coexisting by design

**Why:** A single global LLVM version cannot serve both Tier 1 (must match Julia internal) and the C++ MLIR dialect (must match system LLVM 22). Buckets make the constraint explicit and let the C and C++ paths evolve at independent cadences. Ground for the upcoming C-path internalization (Julia 1.12.6 + LLVM 18, no system fallback).

### C sret Return Classification + Thunk Path Consolidation

Fixed the C generator returning structs by-value through `ccall` when the platform ABI requires sret (caller-allocated return slot passed as a hidden first pointer arg). Previously: silent layout corruption on structs >16 bytes returned from C. Now: routes through the consolidated thunk path that allocates and passes the return slot explicitly.

- `GeneratorC.jl` reduced ~170 lines, dispatch logic for sret unified with the existing C++ thunk emission
- `Compiler.jl` thunk plumbing collapsed into one entry point — was duplicated across C/C++

### dlsym/dlopen Returning `nothing` on Newer Julia

`Libdl.dlsym` / `dlopen` started returning `nothing` instead of throwing on newer Julia versions when symbols/libraries cannot be resolved. The JIT path's symbol resolver assumed a thrown error and crashed downstream with a less informative message. Fixed across `JITManager.jl` — explicit `nothing` checks at all 7 call sites with the same `init_error` surfacing pattern introduced in v2.5.5.

### MLIR Dialect Fixes

- **MarshalArg / RetOp missing assemblyFormat** — both ops parsed but failed to print, breaking round-trip and `mlir-opt` debugging. Added `assemblyFormat` to JLCSOps.td.
- **JIT selftest** added to verify the dialect loads and lowers cleanly on every build (catches missing op declarations before they hit a wrap call).
- **JLCSCAPIWrappers.cpp** — new file exposing C wrappers for dialect APIs needed by Julia bindings.
- **JLCSPasses.cpp** — internal cleanup, `getPackedSizeInBits` consolidation continued from v2.5.5.

### DAG Diff Tuning

~700 lines reworked in `DAGDiff.jl` based on stress-test feedback from v2.5.6:
- Tighter propagation rules — pointer-to-mismatched no longer propagates (only by-value containment does)
- DOT export polish, mismatch annotations more readable
- Query API stabilized

### Test Infrastructure

- Wired three orphan test suites into `runtests.jl` / `devtests.jl`: DAGDiff (1336 lines), MLIR templates (736 lines), exception handling (101 lines)
- `.gitignore` patches to exclude per-project `dag/` exports and build artifacts
- `test/c_test/verify.jl` updated to match consolidated thunk path
- `test/stress_test/verify.jl` removed (188 lines) — replaced by `test_introspect.jl` + `introspect_demo.jl` which cover the same ground via the public API

### Misc

- README dual-toolchain clarification
- MLIR documentation pass
- Stress-test introspect demos use `joinpath(@__DIR__, "..", "..")` so they activate the right project regardless of where they're invoked from
- `src/mlir/build.sh` apt hint corrected to `mlir-21-dev`

### Upgrade Notes

No API breaks. If you were calling `LLVMEnvironment.resolve_tool(name)` directly (not part of the public API but possible in downstream tooling), you must now pass a language: `resolve_tool(name, :c)` or `resolve_tool(name, :cpp)`.

## v2.5.6

### New: DAG Diff — Structural Mismatch Detection Between C++ and Julia IR

Added a DAG-based structural diff algorithm that compares C++ layouts (DWARF ground truth) against Julia's inferred alignment rules. This extends the existing per-function heuristics in `DispatchLogic.jl` — heuristics catch the obvious cases (packed returns, unions, STL), while DAGDiff catches what point-wise checks miss: transitive layout drift through by-value containment chains.

**Algorithm:**
1. Build C++ graph from DWARF metadata (struct sizes, member offsets, containment edges)
2. Build Julia graph by computing `min(sizeof(field), 8)` aligned layouts from the same members
3. Parallel walk — match nodes structurally, record size and per-member offset mismatches
4. Propagate mismatches transitively through by-value containment (if Inner is packed and Outer contains Inner by value, Outer is also mismatched)
5. Flag functions that pass or return mismatched types by value
6. Topo-sort (Kahn's algorithm) all thunk sites for safe lowering order — types before the functions that depend on them

**Integration:**
- `DAGDiff.needs_dag_thunk(symbol, result)` queries the mismatch map — wrapper generators check this alongside existing heuristics, routing to MLIR thunks if either fires
- Backward compatible: `needs_dag_thunk(_, nothing)` returns `false` when DAG diff is not computed
- Wired into both C and C++ generator dispatch sites in `GeneratorC.jl` and `GeneratorCpp.jl`

**Visualization:**
- `export_dot(result, path)` — Graphviz DOT export with mismatch color-coding (red = layout mismatch, orange = function needs thunk, gray = safe)
- `render_dot(result, path)` — renders DOT to SVG/PNG/PDF via the `dot` command
- Per-member offset annotations, containment edges, propagation edge coloring
- Three view modes: `:diff` (both graphs overlaid), `:cpp` (DWARF only), `:julia` (inferred alignment only)

**TOML configuration:**
```toml
[wrap]
dag = true   # exports DAG graphs to <project_root>/dag/
```

When enabled, the wrap stage automatically exports `diff.svg`, `cpp.svg`, `julia.svg`, and `diff.dot` to a `dag/` folder in the project root.

**Files:**
- `src/IRGen/DAGDiff.jl` — New module (~780 lines): graph types, builders, diff algorithm, topo-sort, query API, DOT visualization
- `src/Builder/ConfigurationManager.jl` — Added `dag::Bool` to `WrapConfig`
- `src/Wrapper/Generator.jl` — DAG diff computed before wrapper generation; graphs exported when `dag=true`
- `src/Wrapper/C/GeneratorC.jl`, `src/Wrapper/Cpp/GeneratorCpp.jl` — Dispatch sites augmented with `needs_dag_thunk` check
- `test/dag_test/` — 178 tests covering graph building, structural diff, transitive propagation, topo-sort, query API, DOT export, and a rendered gallery of 7 scenarios

**Stress test results (73 functions, `test/stress_test/`):**
- 25 mismatches detected: 14 types (vtable offsets on polymorphic classes, compound struct padding, bool alignment, STL internals), 5 functions routed to thunks (`compute_lu`, `compute_qr`, `compute_eigen`, `solve_ode_rk4`, `solve_ode_adaptive`)
- Transitive propagation working: `uniform_real_distribution<double>` flagged solely because it contains `param_type` by value

## v2.5.5

### Refactor: Module Hierarchy — Flat Source → Organized Subsystems

Replaced the flat `src/*.jl` layout (14 top-level files, ~12k lines) with a three-subsystem hierarchy. Each subsystem has a thin orchestration shim that controls include order — all implementation lives in subdirectories.

**Top-level shims:**
- **`Builder.jl`** — Build mechanics: config, environment, compile, link, DWARF, package registry
- **`IRGen.jl`** — MLIR/JIT: native bindings, IR generation, JIT execution
- **`Wrapper.jl`** — Julia binding generation: type mapping, dispatch routing, codegen
- **`Introspect.jl`** — Analysis tooling: binary, Julia code, LLVM IR, benchmarking

**Subsystem layout:**
```
src/
  RepliBuild.jl            ← module root, exports, public API delegation
  Builder.jl               ← shim: includes Builder/*.jl
  Builder/
    LLVMEnvironment.jl, ConfigurationManager.jl, BuildBridge.jl,
    DependencyResolver.jl, ASTWalker.jl, Discovery.jl, ClangJLBridge.jl,
    Compiler.jl, DWARFParser.jl, EnvironmentDoctor.jl, PackageRegistry.jl,
    ThunkBuilder.jl
  IRGen.jl                 ← shim: includes IRGen/*.jl
  IRGen/
    MLIRNative.jl, JLCSIRGenerator.jl, JITManager.jl
    ir_gen/  (FunctionGen.jl, StructGen.jl, STLContainerGen.jl, TypeUtils.jl)
  Wrapper.jl               ← shim: includes Wrapper/**/*.jl
  Wrapper/
    Utils.jl, TypeRegistry.jl, Symbols.jl, FunctionPointers.jl,
    DispatchLogic.jl, Generator.jl
    C/    (GeneratorC.jl, TypesC.jl, IdentifiersC.jl, UtilsC.jl)
    Cpp/  (GeneratorCpp.jl, TypesCpp.jl, IdentifiersCpp.jl, UtilsCpp.jl, STLWrappers.jl)
  Introspect.jl            ← shim: includes Introspect/*.jl
  Introspect/
    Types.jl, Binary.jl, Julia.jl, LLVM.jl, Benchmarking.jl,
    DataExport.jl, Project.jl
```

- **`RepliBuild.jl`** is now a pure delegation layer — loads the four subsystem shims, `using`s their modules, and re-exports the public API. No implementation logic remains at the top level.
- **`ThunkBuilder.jl`** extracted from `Compiler.jl` — bridges Builder and IRGen (needs `Wrapper.is_c_lto_safe`), loaded after Wrapper to satisfy the cross-subsystem dependency.
- **`PackageRegistry.jl`** moved from `Hub/` into `Builder/`.
- Stable path constants (`PROJECT_ROOT`, `SRC_DIR`) in `RepliBuild.jl` replace `@__DIR__` in submodules so file moves don't break paths.
- Net deletion: ~11,900 lines of duplicated top-level files removed.

### Fixed: Wrapper Generator Bug Audit (23 bugs)

Comprehensive audit and fix pass across C, C++, and shared wrapper subsystems. Full report in `BUG_AUDIT.md`.

**HIGH (8) — Crashes, memory corruption, silent wrong codegen:**
- **C-1/C-2/CPP-9**: sret llvmcall passed `Ref{T}` (GC addrspace 10) where `Ptr{T}` (raw addrspace 0) was needed — address space mismatch crashes. Fixed with `Base.unsafe_convert` to raw pointer before llvmcall. sret path now also applies integer widening and `Ref→Ptr` conversion matching the main llvmcall path.
- **C-3**: Use-after-free in bitfield/packed struct accessor — `pointer(collect(s._data))` created a GC-eligible temporary. Wrapped in `GC.@preserve`.
- **CPP-8**: C++ llvmcall missing `cconvert` for pointer params — ported the C generator's `Ptr` conversion logic.
- **CPP-10**: Debug `println` statements left in template codegen — removed.
- **U-1**: `_resolve_forward_ptr` flattened `Ptr{Ptr{T}}` to `Ptr{Cvoid}` — now only collapses bare unknown struct names, preserves nested pointer indirection.
- **D-1**: `is_ccall_safe` used uncleaned return type (with `const`) for DWARF lookup — changed to `cleaned_ret`.

**MEDIUM (11) — Incorrect output in edge cases, misrouting:**
- **C-4**: Convenience wrapper DWARF lookup used `"__struct__" * name` prefix that doesn't exist — switched to bare name lookup.
- **C-5/L-4**: `_sanitize_c_type_name` stripped spaces (`" " => ""`), destroying multi-word types like `"unsigned int"` → `"unsignedint"`. Changed to `" " => "_"`. Same fix applied to C++ side in `_sanitize_cpp_type_name`.
- **C-6/L-3**: `is_c_enum_like`/`is_enum_like` were identical to their `is_struct_like` counterparts — any uppercase type got dual-classified. Now return `false`; real enum detection uses DWARF `__enum__` keys via `_is_enum_type()`.
- **C-7**: `long double` mapped to `Float64` (8 bytes) but x86-64 ABI uses 16-byte slots — changed to `NTuple{2, UInt64}` and removed from `_CCALL_SAFE_PRIMITIVES` to force struct safety checks.
- **CPP-1/L-1**: `make_cpp_identifier`/`make_c_identifier` lowercased before keyword check — `"Begin"` incorrectly matched `"begin"`. Removed `lowercase` call (Julia keywords are case-sensitive).
- **CPP-2**: Operator replacement ordering hit single-char operators before compounds — `operator<<` became `op_lt<` instead of `op_lshift`. Reordered longest-match-first.
- **CPP-3/4/5**: STL type detection used `startswith` prefix matching — `std::string_view` false-matched `std::string`, `std::set_difference` matched `std::set`. Added `_stl_name_match()` with word-boundary awareness (requires `<` or ` ` after prefix). Also reordered `unordered_map`/`unordered_set` before `map`/`set` in size lookup.
- **CPP-6**: Safe wrapper used `ccall_args` (containing converted names like `a_c`) instead of original `param_names` — generated code referenced variables that didn't exist in scope.
- **CPP-7**: Template this-pointer checked sanitized name against `struct_types` but DWARF stores raw names (e.g. `"Box<double>"` not `"Box_double"`) — now checks both `bare_class` and `safe_class`.

**LOW (4) — Cosmetic, minor edge cases:**
- **L-2**: `_sanitize_c_type_name` could return empty string from all-special-character input — now returns `"_UnknownType"` fallback.
- **U-2**: `_parse_int_or_hex` missed uppercase `0X` prefix — added `|| startswith(s, "0X")` check.

### New: C++ Exception Catching via `jlcs.try_call`

Added `TryCallOp` (`jlcs.try_call`) to the JLCS MLIR dialect — a variant of `ffe_call` that emits LLVM `invoke` + landing pad to catch C++ exceptions at the ABI boundary.

- **`jlcs.try_call`** — On exception: catches via `__gxx_personality_v0`, extracts the `std::exception::what()` message, stores it in a thread-local buffer via `jlcs_set_pending_exception()`, and returns a zero/null sentinel. The Julia caller checks `jlcs_has_pending_exception()` after return and throws a `CxxException` if set.
- **`CxxException <: Exception`** — New Julia exception type in `JITManager.jl`, wrapping the C++ error message string. Thrown automatically by Tier 2 thunks when the callee raises.
- **Dispatch routing** — `DispatchLogic.jl` updated to route functions marked `noexcept=false` through `try_call` instead of `ffe_call`.
- **C API wrappers** — `jlcs_create_try_call_op`, `jlcs_set_pending_exception`, `jlcs_has_pending_exception`, `jlcs_get_pending_exception`, `jlcs_clear_pending_exception` added to `JLCSCAPIWrappers.cpp` and `MLIRNative.jl`.
- **Lowering** — `TryCallOpLowering` in `JLCSPasses.cpp`: emits `llvm.invoke` to the callee with a landing pad that calls `__cxa_begin_catch`, extracts the `what()` string, calls `jlcs_set_pending_exception`, then `__cxa_end_catch`. Non-exception path falls through normally.
- **Callback test suite** — Extended with C++ functions that throw (`throw std::runtime_error`), verifying that exceptions propagate as `CxxException` on the Julia side.

### Refactor: JIT Symbol Cache — Atomic Copy-on-Write

Replaced the lock-free Dict read pattern in `JITManager.jl` with a proper atomic copy-on-write scheme:

- `compiled_symbols` field is now `@atomic` on `JITContext`.
- **Fast path**: reads an atomic snapshot of the Dict reference — no lock, no race.
- **Slow path**: creates a new Dict copy with the added entry, then atomically swaps the reference.
- Added `init_error` field to `JITContext`; initialization failures are stored and surfaced via `_jit_not_initialized_error()` at all 7 call sites.

### Refactor: Code Cleanup

- **Eliminated `map_cpp_type_to_mlir`** — Deleted the duplicate in `JLCSIRGenerator.jl`, replaced the one call site with `map_cpp_type` from `TypeUtils`.
- **Fixed cross-module reach-through** — Moved `get_stl_container_size` into `TypeUtils.jl`, replaced `Main.RepliBuild.Wrapper.get_stl_container_size` with a direct call.
- **Extracted `getPackedSizeInBits`** — Moved to a static free function in `JLCSPasses.cpp`, removed from both `FFECallOpLowering` and `TryCallOpLowering`.
- **Removed `JL_SubtypeInterface` dead code** — Deleted `JLInterfaces.td`, removed its include from `JLCS.td`.
- **Moved `ASTWalker.jl`** → `Wrapper/ASTWalker.jl`, **`STLWrappers.jl`** → `Wrapper/Cpp/STLWrappers.jl`.
- **MLIR API migration** — All `rewriter.create<Op>(...)` calls in `JLCSPasses.cpp` updated to LLVM 21's `Op::create(rewriter, ...)` builder pattern.

### Removed: Rust Wrapper Generator

Deleted `src/Wrapper/Rust/` (GeneratorRust.jl, TypesRust.jl, IdentifiersRust.jl) and `compile_rust_project()` from Compiler.jl. The experimental Rust generator required `extern "C"` + `#[repr(C)]` on everything, making it effectively a C-ABI wrapper with extra steps. Would need a julia/rust contributer to help with rust because I dont understand the borrow checker enough to deal with it hands on.

### Changed: LTO Global Variable Deduplication

`sanitize_ir_for_julia` now converts externally-visible global variable definitions to `external` declarations in the LTO bitcode. Prevents "Duplicate definition" JIT errors when the shared library is also loaded via `dlopen`.

### Changed: Exports Reorganization

Reorganized `RepliBuild.jl` exports into categorized sections (Core Build Orchestration, Configuration, Compiler Tooling, DWARF Analysis, etc.) and exported additional compiler utility functions for advanced use.

## v2.5.3

### New: STL Map Support (`std::map`, `std::unordered_map`)

Full wrapper generation for `std::map<K,V>` and `std::unordered_map<K,V>` containers, matching the existing `CppVector{T}` and `CppString` pattern.

- **`CppMap{K,V} <: AbstractDict{K,V}`** — New mutable wrapper type in `STLWrappers.jl` that holds an opaque pointer to the C++ map. Lifetime managed by GC finalizer. Supports `getindex`, `setindex!`, `haskey`, `delete!`, `length`, `isempty`, and `empty!` via JIT-compiled MLIR thunks.
- **`CppUnorderedMap{K,V}`** — Type alias for `CppMap{K,V}` (same thunk interface).
- **Map-specific thunk signatures** — `map_at` (key by const ref → value ref) and `map_subscript` (key by const ref → value ref) added to `STLContainerGen.jl`, distinguishing map key-lookup semantics from vector index-lookup.
- **`_classify_stl_method`** — Now accepts an optional `container_type` parameter. `operator[]` and `at()` are classified as `map_subscript`/`map_at` for map containers vs `subscript`/`at` for vectors.
- **Wrapper codegen** — `GeneratorCpp.jl` emits `create_std_map_*()` factory functions for map templates, mirroring the existing vector factory pattern. Template args are parsed via `_split_template_args` to extract K and V types.
- **`_normalize_stl_elem_type`** — Extracted from inline type mapping into a shared helper in `UtilsCpp.jl`. Used by both vector and map factory codegen.
- **`_is_stl_internal_type`** — Expanded blocklist with 13 additional libstdc++/libc++ internal types (`_Alloc_node`, `_Node_handle`, `_Map_base`, `_Insert`, `_Rehash`, `pair<`, `Select1st<`, etc.) that leak through DWARF when wrapping map containers.
- **DWARF byte_size lookup** — Improved container size resolution: uses `get_stl_container_size` first, then fuzzy-matches DWARF keys (now also matches stripped `std::` prefix).

### New: Hub Search (`RepliBuild.search`)

- **`RepliBuild.search(query="")`** — Search the RepliBuild Hub (community package registry) for available packages. Matches against name, description, tags, and language. Shows install status for locally registered packages.
- **`_fetch_hub_index()`** — Fetches and parses `index.toml` from the hub URL via `Downloads.jl`.
- **`REPLIBUILD_HUB_URL`** — Environment variable override for private registries/mirrors.
- Added `Downloads` to `Project.toml` dependencies.

### New: STL Map Test Suite

- `test/stl_test/` — Extended with `std::map<int,int>` coverage: `make_int_map`, `map_lookup`, `map_size` C++ API functions, `CppMap` lifecycle tests (create, insert, read, haskey, delete, empty), and map-passing tests through `const std::map<int,int>&` parameters.

### Changed: Test Directory Consolidation

Reduced the test directory from 14 subdirectories + 8 top-level files to 6 subdirectories + 3 top-level files. All test content preserved through merges:

- **`c_test/`** — Absorbed `basics_test` (PaddedStruct, PackedStruct, NumberUnion, globals, variadic `sum_ints`) and `jit_edge_test` (identity, write_sum, make_pair, PackedTriplet). Pure C with LTO.
- **`stress_test/`** — Absorbed `vtable_test` (Shape/Rectangle/Circle virtual dispatch), `raii_test` (Tracker ctor/dtor), and all standalone MLIR test files (`test_mlir.jl`, `test_mlir_safety.jl`, `test_aot.jl`, `test_raii.jl`). New `verify.jl` covers numerics, vtable dispatch, and conditional MLIR/AOT/RAII sections.
- **`devtests.jl`** — Rewritten to reference the consolidated 6-test suite. Removed duktape setup and standalone MLIR includes.
- **`runtests.jl`** — Added `search` to API surface check.
- **`test_registry.jl`** — Registry integration test updated from `basics_test` to `c_test`.
- **Deleted:** `lua_test/`, `duktape_test/`, `mydir/`, `rust_demo/`, `basics_test/`, `jit_edge_test/`, `vtable_test/`, `raii_test/`, `pugixml_test.jl`, `test_mlir.jl`, `test_mlir_safety.jl`, `test_aot.jl`, `test_raii.jl`.

### Refactor: Dispatch Logic

- **`DispatchLogic.jl`** — Extracted routing logic (`is_ccall_safe`, `is_c_lto_safe`) into a dedicated module, decoupling it from `Utils.jl` and `Generator.jl`.
- **C-Specific LTO Safety** — Introduced `is_c_lto_safe()` for fine-grained C dispatch gates, routing functions with packed struct or union returns to sret thunks while preserving direct `ccall` for safe returns.

### Improved: Cross-Platform DWARF & Target Detection

- **Target Triple Detection** — Added `_detect_target_triple()` in `Compiler.jl` to gracefully determine the host target via `clang -dumpmachine` with a fallback to `Sys.MACHINE`.
- **Robust DWARF Parsing** — `extract_dwarf_return_types` now searches for `llvm-readelf` and `llvm-dwarfdump` as fallbacks when the system `readelf` is missing, improving cross-platform reliability.
- **Return Type Inference** — Added `infer_return_type()` to fallback to demangled C++ function name patterns (e.g. `is_*` -> `bool`, `create_*` -> `void*`) when DWARF debug info is unavailable.

### Refactor: Core API Exports

- **Organized Exports** — Completely reorganized `RepliBuild.jl` exports into categorized sections (Core Build Orchestration, Configuration, Compiler Tooling, DWARF Analysis, LLVM Environment, etc.) for better module discoverability.

## v2.5.2

### New: RAII Dialect Operations

Added C++ constructor, destructor, and scoped lifetime operations to the JLCS MLIR dialect — encoding RAII semantics directly in the IR rather than relying on ad-hoc `llvm.call` emission.

**New operations:**

| Operation | Mnemonic | Purpose |
|-----------|----------|---------|
| `ConstructorCallOp` | `jlcs.ctor_call` | Call a C++ constructor with `this` pointer + parameters |
| `DestructorCallOp` | `jlcs.dtor_call` | Call a C++ destructor with `this` pointer |
| `ScopeOp` | `jlcs.scope` | Region-based RAII scope that guarantees destructor calls at exit |
| `YieldOp` | `jlcs.yield` | Terminator for `jlcs.scope` regions |

- **`jlcs.ctor_call`** — Takes a `FlatSymbolRefAttr` callee and variadic arguments. First argument is always the object pointer (`this`). Lowers to a direct `llvm.call`.
- **`jlcs.dtor_call`** — Takes a `FlatSymbolRefAttr` callee and a single object pointer. Lowers to a direct `llvm.call`.
- **`jlcs.scope`** — Takes managed object pointers as operands and an `ArrayAttr` of matching destructor symbols. Contains a single-block body region. During lowering, body ops are inlined and destructor calls are emitted in **reverse order** (C++ destruction semantics). Not `IsolatedFromAbove` — body can reference values from the enclosing scope.

```mlir
jlcs.scope(%ptr : !llvm.ptr) dtors([@_ZN4BaseD1Ev]) {
  jlcs.ctor_call @_ZN4BaseC1Ev(%ptr) : (!llvm.ptr) -> ()
  // ... use object ...
  jlcs.yield
}
// destructor called automatically here
```

### Changed: `TypeInfoOp` — Destructor Metadata

- `jlcs.type_info` now accepts a fourth argument `destructorName` (default `""`), storing the mangled C++ destructor symbol for the class. IR generators updated to emit the new format.

### New: RAII Test Suite

- `test/test_raii.jl` — 26 tests covering parsing, lowering, and JIT execution of all RAII ops against a compiled C++ test library (`test/raii_test/tracker.cpp`). Validates constructor side effects, destructor side effects, parameterized constructors, scoped lifetime with automatic cleanup, and multi-object scopes with reverse destruction order.

## v2.5.0

### New: Rust Introspective Wrapper Generator

Introduced full support for Rust C-compatible libraries via a dedicated DWARF-based introspective wrapper generator (`src/Wrapper/Rust/`).

- **New `language = "rust"` configuration:** Automatically selects the `rustc` compiler and the Rust generator backend.
- **Topological Struct Ordering:** Autonomously sorts custom structures by dependency, handling pointers (`Ptr{X}`) as soft dependencies to seamlessly emit idiomatic `mutable struct` forward-declarations.
- **DWARF Standard Library Filtering:** Actively identifies and strips out deep internal compiler/stdlib types (like `core::fmt`, `alloc::string`, `std::io::error`, and closure environments) that "leak" through the DWARF metadata, ensuring the Julia wrapper only exposes your public API.
- **Native Enum Resolution:** Correctly infers the underlying primitive types (`Int32`, `UInt32`, `UInt64`, etc.) from DWARF representations, successfully converting signed negative DWARF enum values into their corresponding unsigned native values.
- **ABI Safety Requirements:** Currently, only C-compatible Rust endpoints are supported. Functions must be marked with `extern "C"` and `#[no_mangle]`, and structures/enums must use `#[repr(C)]` or `#[repr(u32)]` to lock their layout for FFI. True native Rust ABI integration (via compiler AST injection) is planned for a future release.

## v2.4.3

### Bug Fix: `WrapConfig` constructor mismatch in Discovery

Fixed a `MethodError` when calling `discover()` caused by the `WrapConfig` constructor in `Discovery.jl` missing the `macros` and `shim_headers` fields added to the struct definition. Empty defaults are now passed for both fields.

## v2.4.2

### Refactor: Wrapper Modularization

The monolithic `src/Wrapper.jl` (~4600 lines) has been split into a structured `src/Wrapper/` package with separate C and C++ sub-packages. `src/Wrapper.jl` is now a thin re-export shim.

**New module layout:**

| File | Lines | Role |
|------|-------|------|
| `src/Wrapper/Generator.jl` | 727 | Top-level `wrap_library()` API; routes to C or C++ generator based on `config.wrap.language` |
| `src/Wrapper/TypeRegistry.jl` | 99 | `TypeRegistry` struct and `TypeStrictness` enum (`:strict`/`:warn`/`:permissive`) |
| `src/Wrapper/Symbols.jl` | 193 | `ParamInfo` and `SymbolInfo` structs for structured symbol representation |
| `src/Wrapper/FunctionPointers.jl` | 77 | DWARF function-pointer signature parser → Julia `@cfunction`-compatible type strings |
| `src/Wrapper/Utils.jl` | 69 | Shared identifier escaping and keyword utilities |
| `src/Wrapper/C/GeneratorC.jl` | 2060 | Full C introspective wrapper generator |
| `src/Wrapper/C/TypesC.jl` | 281 | C type heuristics (`is_c_struct_like`, `is_c_enum_like`) and base type mapping |
| `src/Wrapper/C/IdentifiersC.jl` | 35 | C identifier sanitization |
| `src/Wrapper/C/UtilsC.jl` | 21 | C-specific utilities |
| `src/Wrapper/Cpp/GeneratorCpp.jl` | 2806 | C++ introspective wrapper generator |
| `src/Wrapper/Cpp/TypesCpp.jl` | 428 | C++ type mapping including STL, template, and reference types |
| `src/Wrapper/Cpp/IdentifiersCpp.jl` | 81 | C++ identifier sanitization (namespace stripping, operator handling) |
| `src/Wrapper/Cpp/UtilsCpp.jl` | 44 | C++ utilities |

The C and C++ generators are now fully independent — no shared mutable state, no conditional branching on language inside generation loops. Each generator emits correct stdout-unbuffering preamble, LTO/thunks blocks, struct definitions, and function wrappers for its language.

### Improved: Compiler — JLL-First C Compilation

- C source files (`.c`) are now compiled via `Clang_unified_jll.clang` when available. This produces LLVM IR that exactly matches Julia's internal LLVM version, guaranteeing `Base.llvmcall` compatibility for LTO-enabled C projects. Falls back to system `clang` if the JLL is unavailable.
- `create_library()` and `create_executable()` now select `clang` vs `clang++` based on `config.wrap.language` (previously always used `clang++`).
- `clang --version` probe in metadata extraction also respects `config.wrap.language`.

### New: `wrap.language` Configuration Field

A new `language` field in the `[wrap]` section of `replibuild.toml` selects the generator and compiler toolchain for the project. This field is designed as an extensible language dispatch key — `"c"` and `"cpp"` are the first two targets, with more languages planned.

```toml
[wrap]
language = "c"   # or "cpp" (default)
```

- **`"c"`** — Selects the C generator, compiles with `clang`, and defaults `enable_lto = true` so pure-C libraries get zero-cost `llvmcall` dispatch automatically.
- **`"cpp"`** — Selects the C++ generator (existing behavior), defaults `enable_lto = false`.
- `discover()` auto-detects language from the scanned source files and sets this field accordingly.

### New: C Abomination Stress Test

`test/c_abomination_test/` — a C stress test deliberately constructed to exercise the hardest edge cases the C wrapper generator must handle:

- Deeply nested anonymous structs and unions (3 levels)
- Bitfield members (`uint8_t f1 : 1`, `f2 : 3`, `f3 : 4`)
- Multi-dimensional arrays of structs
- Nested function pointer typedefs (`OuterCallback` returning `InnerCallback`)
- Flexible array members
- Opaque pointer lifecycle (`init_opaque` / `destroy_opaque`)
- Multi-file C project (header + source, pure C, LTO enabled)

### Changed: `.gitignore`

- Added `*.bak` to suppress editor backup files.
- Added `__pycache__/` and `*.pyc` to suppress Python bytecache from helper scripts.

## v2.4.1

### Improved: LTO Pipeline — Bitcode-First Loading

- LTO artifacts now ship as LLVM bitcode (`.bc`) instead of text IR (`.ll`). Julia parses `.bc` substantially faster, reducing wrapper module load time for large libraries.
- The generated wrapper reads bitcode as `UInt8[]` (`read(LTO_IR_PATH)`) — `Base.llvmcall` accepts both text and binary IR.
- `LTO_IR_PATH` and `THUNKS_LTO_IR_PATH` now point to `.bc` files; the `.ll` text files are retained as build-time intermediates only.
- AOT thunks pipeline (`_build_aot_thunks`) also emits `.bc` alongside the `.ll` sanitized IR.

### Improved: LLVM 21+ IR Compatibility

Seven additional LLVM 21 attribute and instruction forms stripped from the sanitized LTO IR to prevent Julia's (potentially older) internal LLVM from rejecting the bitcode:

- `allocptr` pointer-attribute keyword
- `samesign` qualifier on `icmp` comparisons
- `range(...)` return-value attribute
- `nuw`/`nsw` qualifiers on `trunc` instructions
- `nneg` qualifier on `zext` and `uitofp` instructions
- Multi-range `initializes((...), (...))` attribute (previous regex only handled single-range form)
- Complete attribute block replacement: all `attributes #N = { ... }` blocks are now reduced to `{ alwaysinline }`, eliminating future breakage from `allockind`, `allocsize`, `memory(errnomem:...)`, and similar LLVM-version-specific keywords

Both the main LTO path (`link_optimize_ir`) and the AOT thunks path (`_build_aot_thunks`) apply the full set of transforms.

### New: `assemble_bitcode` — JLL-First Bitcode Assembly

- New exported `Compiler.assemble_bitcode(ll_path, bc_path)` function replaces inline `llvm-as` calls throughout the pipeline.
- **Strategy**: first attempts `Clang_unified_jll.clang -emit-llvm` to produce bitcode using the exact same LLVM version Julia uses internally, guaranteeing `llvmcall` compatibility. Falls back to system `llvm-as` if the JLL path is unavailable.

### Improved: C Source File Compilation

- `.c` files are now compiled with `clang` instead of `clang++`. This prevents C code from being parsed with C++ semantics (implicit `extern "C"`, C99 restriction differences, etc.) and silences spurious `clang++` warnings on pure-C projects like SQLite and Duktape.

### Fixed: Wrapper — Forward Declaration Robustness

Three independent bugs corrected in `Wrapper.jl`, validated against SQLite (269 functions), cJSON, http-parser, Duktape, and the full 81-test CI suite:

- **Parameter/return type scanning for opaque structs** — Forward declarations previously only scanned struct members. Types like `sqlite3_blob` that appear exclusively in function signatures (never as struct members) were missing their `mutable struct Foo end` forward declarations, causing `UndefVarError` at module load time.
- **Enum names excluded from forward declarations** — Enum types defined via `@enum` were receiving duplicate empty-struct forward declarations that shadowed the enum. The forward-declaration pass now skips any name already registered as an enum.
- **Union accessor type sanitization and deferred emission** — Union member type names now go through `_sanitize_julia_type_name()` to match the actual emitted struct names (e.g. `__pthread_mutex_s` → `_pthread_mutex_s`). Unknown `Ptr{X}` inner types fall back to `Ptr{Cvoid}`. Accessor functions are now emitted after all struct definitions, eliminating forward-reference errors.

### Fixed: Wrapper — Struct Dependency Ordering

- Introduced `_JULIA_BUILTIN_TYPES` constant — a comprehensive set of all Julia/C interop scalar types that should never trigger a forward declaration or a hard dependency.
- New `_resolve_forward_ptr(julia_type, defined_names)` helper: for any `Ptr{X}` (including nested `Ptr{Ptr{X}}`), replaces `X` with `Cvoid` when `X` is an as-yet-undefined custom struct. This avoids forward-reference errors while preserving correct ABI (all pointers are pointer-sized).
- Struct topological sort now treats `Ptr{X}` as a **soft** dependency (ordering hint only) and `NTuple{N,X}` / `Ref{X}` as **hard** dependencies (inline embedding requires the full definition). Pointer-heavy C++ headers no longer trigger topological sort failures.
- `infer_julia_type` internal-type blocklist check is now applied before any other type dispatch, ensuring compiler-internal types (`__va_list_tag`, `ldiv_t`, etc.) never reach struct or function generation.

### Fixed: Wrapper — Template Struct Member Sanitization

- Union and struct member types containing `<>` (C++ template syntax) are sanitized before emission: `Ptr{stl_internal<char>}` → `Ptr{Cvoid}`, bare template types → size-based `NTuple{N,UInt8}` or `Ptr{Cvoid}`.
- Prevents syntax errors in generated wrappers for libraries that expose STL types in their public interface (tested against Duktape and ImGui configs).

### Improved: Metadata — Absolute Include Paths

- `include_dirs` in `compilation_metadata.json` are now stored as absolute paths. This prevents `wrap()` from failing when called from a working directory different from the project root.

### New: Test Suite

- **Registry test suite** (`test/test_registry.jl`) — 494-line isolated test covering the full `register`/`unregister` lifecycle, content-addressed deduplication, TOML hash normalization, build artifact caching, environment-check TTL, index persistence, and error cases. Uses isolated `REPLIBUILD_HOME` via temp dirs to avoid polluting the user's real registry.
- **Duktape integration test** (`test/duktape_test/`) — Wraps the Duktape JS engine (pure C amalgamation, ccall tier, LTO off). Tests heap lifecycle, `duk_eval_string`, stack push/pop, and string/number/boolean round-trips.
- **Developer test runner** (`test/devtests.jl`) — New script for developer machines that runs the full integration suite (Lua, SQLite, cJSON, Duktape, vtable, JIT edge cases, registry). Separated from CI to keep `runtests.jl` fast.
- **CI cleanup** — Removed ~15 outdated standalone test directories (`benchmark_test`, `custom_test`, `hello_world_test`, `lto_benchmark_test`, `stdlib_test`, `stl_test`, etc.) that were superseded by the unified stress-test suite.

### Changed: Documentation Layout

- `docs/ARCHITECTURE.md` → `docs/architecture.md`
- `docs/DEEP_TECHNICAL_ANALYSIS.md` → `docs/technical-reference.md`
- `benchmark_results.md` (repo root) → `docs/benchmark_results.md`
- Removed `docs/TECHNICAL_INDEX.md` and `docs/TECHNICAL_SUMMARY.txt` (content superseded by architecture and technical-reference docs)
- `*.code-workspace` added to `.gitignore`

## v2.4.0

### New: Global Package Registry

- **`RepliBuild.use("lua")`** — One-call wrapper loading: looks up the registry, resolves git/system/local dependencies, checks the environment, builds if needed, wraps, caches artifacts, and returns a loaded Julia module.
- **`RepliBuild.register(toml_path)`** — Hash (SHA256) and store a replibuild.toml in the global registry at `~/.replibuild/registry/`. Auto-called by `discover()`.
- **`RepliBuild.list_registry()`** — Print all registered packages with hash, source, build status, and registration date.
- **`RepliBuild.unregister(name)`** — Remove a package from the registry and clean cached builds.
- **Global build artifact caching** in `~/.replibuild/builds/<hash>/` — repeated `use()` calls load cached builds instantly.
- **Environment check caching** in `~/.replibuild/toolchain.toml` — avoids re-probing LLVM/Clang on every call (24h TTL).
- `discover()` now auto-registers the generated TOML in the global registry.
- `scaffold_package()` pulls TOML from registry when the name matches a registered package.
- Scaffold.jl merged into PackageRegistry.jl — single unified module for package management.
- Respects `REPLIBUILD_HOME` env var for custom registry location (default: `~/.replibuild/`).

### Fixed: Enum Extraction

- Replaced regex-based enum extraction with Clang.jl AST walker — correctly ignores Doxygen comments, handles `enum class`, hex values, and namespaces.
- Complete Julia keyword escaping (`in`, `and`, `or`, `not`, `isa`, `where` etc.) via shared `_JULIA_KEYWORDS` set.
- Internal type blocklist (`__va_list_tag`, `ldiv_t`, etc.) filters compiler internals from exports.
- Auto-detects enum underlying type (`UInt32`/`Int64`) for values exceeding `Int32` range.
- Eigen wrapper: 1507 → 1106 lines, all 14 verify.jl tests pass.

## v2.3.0

### New: Environment Diagnostics ("Doctor")

- **`RepliBuild.check_environment()`** — Comprehensive toolchain validation that checks for LLVM 21+, Clang, mlir-tblgen, CMake, and the compiled JLCS dialect. Prints a colorful, readable diagnostic report with per-OS installation instructions when tools are missing.
- Automatically runs before `build()` — if the toolchain is incomplete, users get actionable fix instructions instead of cryptic cmake/ccall failures.
- Returns a `ToolchainStatus` struct for programmatic use (`status.ready`, `status.tier1_ready`, `status.tier2_ready`).

### New: Standardized Package Scaffolding

- **`RepliBuild.scaffold_package("MyEigenWrapper")`** — Generates a complete, distributable Julia package structure for RepliBuild wrappers: `Project.toml`, `replibuild.toml`, `src/` stub, `deps/build.jl` hook, and `test/` skeleton.
- Standardizes how wrapper packages are structured and distributed. Users edit `replibuild.toml` and run `Pkg.build()`.

### New: Automatic JLCS MLIR Dialect Compilation

- **`deps/build.jl`** — Automatically compiles the JLCS MLIR dialect (`libJLCS.so`) when RepliBuild is installed via `Pkg.add`. Detects CMake, LLVM, and MLIR, runs the build, and caches the result with a source-content hash.
- Graceful degradation: if the MLIR toolchain is missing, Tier 1 (ccall) builds still work; only Tier 2 (MLIR JIT) is unavailable.

### Improved: Aggressive Hash-Based Caching

- **Project-level content hashing** — The build cache now hashes `replibuild.toml` content, all source file contents, all header file contents, and the git HEAD of the project root. If the hash matches the cached artifacts, `build()` returns in sub-second time without invoking any compiler.
- Replaces the previous mtime-only file cache (which is still used for per-file IR caching) with a project-wide fast-exit path.

### Improved: README Philosophy Section

- Added a "Philosophy" section explaining the source-based approach vs JLLs/BinaryBuilder, framing the heavy toolchain requirement as a deliberate design choice for zero-overhead, zero-edit bindings.

## v2.2.1

### Fix: Wrapper Generator — C++ Namespace & Operator Correctness

Seven bugs fixed in `Wrapper.jl` that caused the generated wrapper to fail parsing or crash at runtime when wrapping real-world C++ libraries (validated against pugixml 1.15):

- **Template type sanitization on `Ptr{}`-wrapped builtins** — `Ptr{xml_stream_chunk<char>}` was skipping angle-bracket sanitization because the outer `Ptr` triggered `is_builtin`. Now also sanitizes when `<>` are present, regardless of `is_builtin`.
- **STL-internal type check on wrapped inner types** — `_is_stl_internal_type` was called on `Ptr{char_traits<char>}` (starts with `Ptr{`), always returning false. Now extracts the inner type before checking.
- **Destructor finalizers use mangled symbol** — Finalizers generated `ccall((:~ClassName, lib), ...)` which is a syntax error at Julia parse time. Now uses the mangled C++ symbol (`_ZN...D2Ev`) from `deleters_mangled`.
- **`this` parameter namespace prefix stripping** — When a class is `pugi::xpath_query`, the Julia struct is `xpath_query` (no namespace). Now correctly strips the namespace prefix by scanning for the last `::` at angle-bracket depth 0.
- **Namespace-only "class" guard for free functions** — Free functions in a C++ namespace (e.g. `pugi::get_memory_allocation_function`) were parsed with `class="pugi"` and received a spurious synthesized `this` parameter. Now only synthesizes `this` if the bare class name is a known struct type.
- **Operator function name `>` depth confusion** — `operator>=` / `operator>` contain `>` which corrupted angle-bracket depth tracking, producing garbled type names. Now heavily sanitizes `safe_class` and falls back to `Cvoid` for any `operator…` class.
- **Parameter `::` sanitization** — Namespace-qualified types in DWARF parameter lists (e.g. `pugi::xml_attribute`) were emitted verbatim. Added a second sanitization pass to convert `::` and remaining non-identifier characters.

### New: Build Orchestration & Dependency Resolution
- **Zero-Boilerplate Git Dependencies** — `DependencyResolver.jl` introduces native `[dependencies]` blocks in `replibuild.toml` to automatically fetch, filter (via `exclude`), and inject raw external C/C++ git repositories into the Clang compilation pipeline.
- Bypasses the need for BinaryBuilder / JLL packages for local development, guaranteeing full DWARF extraction on arbitrary upstream code.

### New: Cross-Language LTO (Link-Time Optimization)
- **Zero-Cost Abstractions via `Base.llvmcall`** — When `enable_lto = true`, the compiler now emits an LLVM Bitcode payload (`_lto.bc` and `_lto.ll`). The generated Julia wrapper intercepts safe primitive/pointer FFI boundaries and dynamically loads the LLVM IR at parse-time, routing the execution through `Base.llvmcall` instead of `ccall` to allow Julia's JIT to inline C++ code directly into Julia hot loops.

### New: MLIR Ahead-Of-Time (AOT) Thunks
- **Static C++ Vtable Dispatch** — Introduced `aot_thunks` flag in the configuration to statically compile MLIR JLCS thunks directly into `.o` artifacts, linking them into a native `_thunks.so` companion library during the `build()` phase.
- Generated `Wrapper.jl` now conditionally emits purely static `ccall` bindings that bypass the `JITManager` runtime entirely for zero-overhead, statically-verifiable polymorphic execution.

### New: Automated Template Instantiation
- **Declarative Template Resolution** — Added `templates` and `template_headers` to the `[types]` config. The compiler automatically generates dummy C++ source files to force Clang to instantiate the requested types (e.g. `std::vector<int>`), guaranteeing they appear in the DWARF debug metadata for MLIR processing and FFI wrapping.

### Improved: Wrapper Ergonomics
- **Idiomatic Julian Classes** — The wrapper generator now semantically clusters factory functions (`create_circle`), destructors (`delete_shape`), and instance methods from the DWARF metadata to emit high-level, idiomatic `mutable struct` wrappers.
- **Julian Multiple Dispatch** — C++ instance methods are automatically proxied via multiple dispatch (e.g., `area(c::Circle)`) passing the raw C pointers via `Base.unsafe_convert`.
- **Automatic Garbage Collection** — C++ object lifecycles are now safely and natively managed by Julia's GC via implicitly registered finalizers on the generated structs.

## v2.1.0

### New: MLIR JIT Compilation Pipeline

- **JITManager.jl** — New module managing MLIR JIT lifecycle with lock-free symbol cache and arity-specialized `invoke` methods (1-4 args, zero heap allocation)
- **Tiered dispatch** — Functions auto-classified as ccall-safe (Tier 1) or JIT-required (Tier 2). Packed structs, unions, virtual dispatch, and large struct returns route through MLIR JIT transparently
- **ir_gen/ submodule** — `TypeUtils.jl`, `StructGen.jl`, `FunctionGen.jl` for modular MLIR IR generation with topological struct sorting and packed struct marshalling

### New: Wrapper Generator Capabilities

- **Union support** — `mutable struct` with `NTuple{N,UInt8}` backing + typed getter/setter accessors
- **Bitfield support** — Bit-shift extraction for single-byte, `unsafe_load`-based for multi-byte fields
- **Variadic function support** — Typed overloads from `[wrap.varargs]` config
- **Global variable accessors** — `cglobal` + `unsafe_load` wrappers
- **Automatic finalizer generation** — Detects destructors/deleters, generates `ManagedX` types with GC-traced finalizers and `Base.unsafe_convert`
- **Virtual method dispatch** — Generates JIT thunk wrappers for virtual functions
- **Forward declarations** — Opaque/circular struct references handled via forward-declared empty structs
- **Base class member flattening** — Inherited fields prepended in struct definitions
- **Struct padding** — Explicit `_pad_N::NTuple{K,UInt8}` fields for correct memory layout

### Improved: DWARF Parser

- Union, bitfield, global variable, typedef extraction from debug info
- Varargs and virtual method detection
- Robust state-machine rewrite (`parse_dwarf_output_robust`) replacing fragile implicit tracking
- Struct member data — `MemberInfo` with offsets now propagated through the pipeline

### Improved: Compiler

- Multi-level pointer resolution (`T**` -> `Ptr{Ptr{T}}`)
- Reference type resolution (`T&` -> `Ref{JuliaType}`)
- Expanded type map — `ssize_t`, `ptrdiff_t`, `intptr_t`, `int8_t`..`uint64_t`, etc.
- Library search path (`-L`) support
- Const/volatile stripping uses word-boundary regex (no more mangling `"constructor"` -> `"ruor"`)

### Improved: MLIRNative

- JIT execution engine — `create_jit`, `destroy_jit`, `lookup`, `jit_invoke`, `invoke_safe`
- Module cloning, function introspection, type predicates
- `lower_to_llvm` pass pipeline

### Changed: Dependencies

- **Added**: `BenchmarkTools`, `Libdl`
- **Removed**: `Distributed`, `RepliBuildPaths.jl` (451-line directory management system)
- **Julia minimum**: 1.9 -> 1.10
- **Clang compat**: now accepts 0.18 + 0.19

## v2.0.3

- Initial public release with DWARF-based wrapper generation
- Clang.jl integration for header parsing
- Introspection toolkit (binary analysis, benchmarking, data export)
- MLIR/JLCS dialect foundation
