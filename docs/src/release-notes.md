# Release Notes

Condensed highlights of recent releases. The authoritative, fully detailed history lives in the repository [CHANGELOG](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/CHANGELOG.md).

## v3.0.1 (2026-07-18) — the inheritance-ABI and Tier-2-correctness release

**C++ inheritance, end to end.**

- **Multiple inheritance** — base-class subobject offsets are extracted from `DW_TAG_inheritance`, derived layouts flatten base members at their true offsets, and `<Derived>_as_<Base>` upcast helpers apply the static Itanium adjustment so secondary-base methods receive a correct base-relative `this`.
- **Virtual dispatch honors overrides** — virtual instance methods now dispatch through `jlcs.vcall`: the thunk reads the object's vptr, indexes the slot, and calls indirectly, so a base-class wrapper invoked on a derived object reaches the override. Dispatch coordinates are class-local (an empirically pinned Itanium/DWARF fact); destructors deliberately remain direct calls for exact-class RAII semantics.
- **Virtual inheritance** — a virtual base's offset is vtable-resident and dynamic, not static. Both DWARF parsers now decode the vbase location *expression*, `jlcs.type_info` carries a virtual-base table, and the wrapper emits dynamic `<Derived>_as_<VBase>` upcasts that read the offset through the object's vtable at runtime — the same helper is correct for every dynamic type. Diamond-proven: one shared base, override dispatch through the vbase vtable, polymorphic deletion.

**Tier-2 ABI correctness (driven by the pugixml and tinyxml2 Hub packages).**

- **SysV small-struct classification** — the thunk lowering previously forced sret for every packed struct return; native x86-64 returns ≤16-byte aligned structs in registers, so calls shifted every argument by one slot. `try_call`/`ffe_call` now classify MEMORY vs register class and coerce register-class structs one scalar per eightbyte, clang-style, for returns and by-value arguments. Verified against a real clang-compiled callee.
- **Nested packed structs** — a padding-free struct nested by value inside a padded struct leaked a JLCS dialect type into LLVM struct bodies, crashing LLVM-IR translation at module load. Such members now inline as byte-identical LLVM literals.
- **JIT pre-flight guard** — `create_jit` refuses modules containing untranslatable types with a catchable error naming the type and op; a bad type degrades the module to "Tier 2 disabled" instead of killing the process.
- **Scope-RAII by-value parameters** — under Itanium, a by-value parameter of a class with a non-trivial destructor is passed as a pointer to a caller-owned temporary. Thunks now copy-construct the temporary in a `jlcs.scope` and destruct it at scope exit; the old raw-bits pass was a silent miscompile.
- **Thunk routing** — virtual instance methods were previously routed to a legacy thunk pass whose symbols the wrapper never looked up; they now flow through the standard ciface thunk pass and are callable through `invoke()`.
- **Dialect verifiers** — `jlcs.scope`, `jlcs.marshal_arg`, `jlcs.vcall`, and `jlcs.type_info` verify their attribute arity and element kinds at parse time; the previously known lowering crashes are now parse diagnostics.

**Robustness and honesty.**

- **discover(force) preserves user intent** — forced re-discovery no longer destroys hand-curated TOML sections (`[types].templates`, `[wrap].varargs`/`macros`/`shim_headers`/`cstring_owned`); this had silently broken template-dependent fixtures for six weeks.
- **Depth-aware DWARF attribution** — members declared after a nested type definition no longer vanish from metadata (clang interleaves nested-type DIEs mid-member-list; the parser now attributes by DIE tree depth).
- **Ingest honesty pass** — ingest is documented and guarded as an experimental, C-only fallback; both entry points warn that C++ API surfaces of ingested binaries are unsupported.
- **Macro-shim header-collision guard** — shim includes are verified to resolve inside the project/dependency tree, not to a system-installed header at a different version.

## v3.0.0 (2026-07-10) — the C-generator audit release

Ownership and ABI edges of the ergonomic layer closed. **Breaking changes** to the generated-wrapper API (wrappers regenerate automatically via the fingerprinted cache; calling code may need updates):

- **Struct-by-value convenience overloads removed** — every such call was UB-adjacent (the callee saw a pointer to a temporary copy; frees and stores corrupted memory). Pass a pointer or `Ref` instead.
- **`char*` returns are `Union{String,Nothing}`** — NULL returns `nothing` instead of throwing. Ownership of malloc'd returns is declared in `[wrap.cstring_owned]` (the wrapper frees through the declared symbol after copying); every `Cstring`-returning function gains a raw `<name>_ptr` variant.
- **True variadic calls** — vararg wrappers emit the `@ccall` semicolon form, so the callee is declared variadic in LLVM IR and the SysV `AL` protocol is correct; float varargs no longer depend on leftover register state.
- **Nested-member structs resolve to named fields** (v2.5.8) instead of `_data::NTuple` blobs, under an exact-layout proof.
- **Bitfield accessors assemble exact byte spans** — no out-of-bounds reads/writes at struct tails; setters accept negative integers with wrapping semantics.
- **Registry cache fingerprints the generator** — `use()` can no longer serve wrappers generated by outdated codegen; each package rebuilds once after upgrading RepliBuild.
- **Wrappers resolve their library sibling-first** via `@__DIR__`, making cached builds self-contained.
- **Macro shims pinned to default visibility** — `[wrap.macros]` survives `-fvisibility=hidden` builds.
- Misaligned small blob parameters by value now trap loudly instead of corrupting silently; unresolvable-type globals get a `_ptr` accessor only.

## v2.5.8 / v2.5.9 (2026-06, internal)

Never separately registered; shipped as part of v3.0.0.

- **Nested-struct member resolution (C)** — structs whose members are themselves structs resolve to named Julia fields under an exact-layout proof (explicit padding fields, every DWARF offset and total size reproduced) instead of opaque byte blobs. This closed a silent by-value miscompile where a small float-bearing struct's blob image classified INTEGER while the real struct traveled in XMM registers.
- **ABI safety trap** for residual register-class float blobs: fail closed rather than corrupt.
- **Per-file IR cache correctness** — the cache is keyed on a compile fingerprint in addition to `mtime`, so flag/define/include changes rebuild without manual cache clearing.
- **`jlcs.vcall` emit fix** — the indirect-call lowering used a malformed operand-segment encoding that crashed LLVM-IR emission; rebuilt on the dedicated indirect-call builder.
- Verified compatibility with system LLVM/MLIR 22.x for the dialect build.
