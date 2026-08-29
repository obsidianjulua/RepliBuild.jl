# Engine audit — resurrected `elseif is_virtual` vs policies that landed while it was dead

**Date:** 2026-08-29
**Scope:** C++ wrapper emission in `GeneratorCpp.jl`, against the uncommitted
local change that re-sources virtuality from `DWARFParser.parse_vtables`.
**Not in scope:** Hub package work, generator patches. This is the audit the
clipper2 missing-seven reduced to.

The branch under audit is `elseif is_virtual` in
[`src/Wrapper/Cpp/GeneratorCpp.jl`](../../src/Wrapper/Cpp/GeneratorCpp.jl)
(~2923). It has been unreachable since it was written. Five presentation /
hygiene policies, plus the vcall producer, landed on the *other* dispatch
paths in the meantime. The local uncommitted change makes that branch live
for AOT wraps. This document is the checklist for whether it can ship.

---

## How it died

`metadata["functions"][i]["is_virtual"]` is never written. Measured 0 of 2777
functions across every C++ Hub package (clipper2 183, pugixml 411, tinyxml2
284, box2d 518, imgui 1377, hello_world 4). The same is true of `test/mi_test`
(0 of 31), including `Base2::get_b` / `Derived::get_b`. Function objects carry
`is_method`, `is_noexcept`, `is_vararg` — not `is_virtual`.

`Compiler.jl` *does* store `"is_virtual"` — on `return_types[function_key]`,
and only when `DW_AT_virtuality` is seen under `in_function_context`. Virtuality
sits on the in-class **declaration** DIE. The out-of-line definition the
extractor walks does not carry it, so the key never reaches the function
records the wrapper reads.

The generator then did `is_virtual = get(func, "is_virtual", false)`. Always
false. The `elseif is_virtual` body — AOT `ccall((:thunk_<mangled>,
THUNKS_LIBRARY_PATH), …)` / JIT `get_jit_thunk` — has been dead code.

Virtual methods therefore fell through to generic Tier 2 and asked for
`_mlir_ciface_<mangled>_thunk`. AOT (`ThunkBuilder.build_aot_thunks`) calls
`generate_jlcs_ir` **with no manifest**, so `JLCSIRGenerator` still routes
those symbols through `generate_virtual_method_ir` and emits `thunk_<mangled>`
— no ciface, different arity, different name. Nothing defined the symbol the
wrapper bound. Clipper2 surfaced it as 7 unresolved `_FPTR_*` slots of 156
(`packages/clipper2/GENERATOR-aot-thunk-gap.md` in the Hub).

That mismatch is real. Matching the two names is not, by itself, a correct
fix — see [What the AOT pass actually emits](#what-the-aot-pass-actually-emits).

## How the local change resurrects it

Uncommitted on `main` (`GeneratorCpp.jl` +65, `Utils.jl` +97,
`test/test_introspection.jl` +44; harness `test/ab_virtual_dispatch.jl`
untracked):

1. Wrap-time `parse_vtables(lib_path)` → `vtable_virtuals::Set` of mangled
   names. Same source AOT already reads.
2. `takes_virtual_path = config.compile.aot_thunks && (mangled in vtable_virtuals)`.
   Generic MLIR is skipped so the function can reach `elseif is_virtual`.
3. `is_virtual = mangled in vtable_virtuals` (no longer the never-written key).
4. Wrap-time `_assert_aot_thunks_present` against `_FPTR_*` / `_mlir_ciface_*`.

A vtable-parse failure degrades to the old answer (nothing is virtual) with a
`@warn`. Gated on `aot_thunks` so JIT wraps keep today's generic-MLIR path.

Clipper2 is the live intersection: 25 unique vtable virtuals, 7 of them in
`compilation_metadata.json` — exactly the missing seven.

| class | method | julia return / C return | metadata nargs |
|---|---|---|---|
| `Clipper2Lib::PolyPath64` | `Clear` | `Cvoid` / `void` | 1 |
| `Clipper2Lib::PolyPath64` | `Count` | `Csize_t` / `size_t` | 1 |
| `Clipper2Lib::PolyPath64` | `AddChild` | `Ptr{Cvoid}` / `PolyPath64*` | 2 |
| `Clipper2Lib::PolyPathD` | `Clear` / `Count` / `AddChild` | same shapes | 1 / 1 / 2 |
| `Clipper2Lib::Clipper2Exception` | `what` | **`Cstring` / `const char*`** | 1 |

`libclipper2_thunks.so` defines `thunk_<mangled>` for all seven and does
**not** define the corresponding `_mlir_ciface_*_thunk`. Today's shipped
wrapper asks for the ciface via `invoke_aot_ptr(_FPTR_…[], …)` and, for
`what`, through `_cstring_wrapper_pair`.

---

## Verdict

| Policy | Landed | Against the resurrected branch |
|---|---|---|
| `_cstring_wrapper_pair` | 2026-08-12 | **FAIL** — branch builds the call itself |
| Receiver gates (`_cpp_this_param` / `_has_receiver`) | 2026-08-13 | **PASS on the Julia side; FAIL at the thunk** |
| `_dedup_method_chunks` | 2026-08-08 | **PASS** — still runs on final chunks; one emission |
| `Base.`-qualification | 2026-08-05 | **PASS** as written (no generator diagnostic in the body) |
| Docstring `Cstring` → `Union{String,Nothing}` | 2026-08-12 | **FAIL** — rewrite is unconditional, body does not match |

Adjacent policies the resurrection also collides with, even though they were
not named in the ask: the **vcall producer** (2026-07-17), the **FunctionGen
ciface routing fix** (same day), `_assert_cstring_policy`,
`_assert_aot_thunks_present` / `_aot_thunk_symbol`, `_dispatch_facts` shape 3,
the Itanium thunk filter, `_assert_wrapper_parses`,
`_assert_no_any_ccall_return`. Those are below, after the five.

---

## 1. `_cstring_wrapper_pair` — FAIL

**Policy (CHANGELOG v3.3.0, 2026-08-12).** A `char*` return is a presentation
question, not a dispatch question. One helper owns the `Union{String,Nothing}`
wrapper, the `<name>_ptr` sibling, the NULL check, the copy, and
`[wrap.cstring_owned]`. A tier supplies two call-body strings and decides
nothing else. Four emission sites consume it: C ccall, C++ ccall, C++ Tier-2
JIT, C++ Tier-2 AOT. The C++ MLIR branch used to `continue` past the ccall
policy and leaked 75–77 bare `Cstring`s across five Hub packages.

`_assert_cstring_policy` refuses a wrapper where a `char*` return escapes as
a bare `Cstring` outside a `::Union{String,Nothing}` function and outside a
`_ptr` variant. Every pattern is anchored on the **return** position.

**What the resurrected branch emits** (`GeneratorCpp.jl` 2923–2960), for
`julia_return_type == "Cstring"`:

```julia
ret_type_sig = (julia_return_type == "Any" || julia_return_type == "Cstring") ?
               safe_c_ret : julia_return_type
# …
function $julia_name($param_sig)::$ret_type_sig
    return ccall((:thunk_$mangled, THUNKS_LIBRARY_PATH), $ret_type_sig, $ccall_types, $ccall_args)
end
```

`_cstring_wrapper_pair` is not called. No `_ptr` sibling is exported. No
NULL check, no `unsafe_string`, no `cstring_owned` free. The generic Cstring
arm (~2993) and both MLIR Cstring arms (~2707, ~2737) still go through the
helper; this arm is a fifth site the 2026-08-12 derivation never saw.

**Clipper2 `what` is the live case.** Metadata:

```
return_type = { c_type: "const char*", julia_type: "Cstring" }
```

So `ret_type_sig` becomes the C spelling `const char*`, and the wrapper would
contain:

```julia
function Clipper2Lib_Clipper2Exception_what(this::Any)::const char*
    return ccall((:thunk__ZNK11Clipper2Lib17Clipper2Exception4whatEv, THUNKS_LIBRARY_PATH),
                 const char*, (Ptr{Clipper2Exception},), this)
end
```

That is not valid Julia. `_assert_wrapper_parses` (first guard in
`_assert_wrapper_loadable`) refuses the **entire** module. Tinyxml2 (22
`Cstring` returns), pugixml (14), imgui (31) will hit the same shape on
whichever of those returns are virtual — wrap-stop, Hub-wide, not a degraded
function.

If `c_type` were already `Cstring`, parse would succeed and
`_assert_cstring_policy` would then refuse: the AOT ccall shape is one of the
five patterns the guard was taught (`test_cstring_policy.jl` "tier-2 AOT").
The dead branch launders `Cstring` to the C spelling, so the policy guard
never sees `Cstring` in the return position. The file dies at parse instead
of at the policy it skipped. Either way it cannot ship; the presentation
policy is not applied.

Today's *shipped* `what` (generic AOT path) is the policy:

```julia
function Clipper2Lib_Clipper2Exception_what(this::Any)::Union{String,Nothing}
    ptr = RepliBuild.JITManager.invoke_aot_ptr(_FPTR_…[], Cstring, this)
    ptr == C_NULL && return nothing
    s = unsafe_string(ptr)
    return s
end
function Clipper2Lib_Clipper2Exception_what_ptr(this::Any)::Cstring
    return RepliBuild.JITManager.invoke_aot_ptr(_FPTR_…[], Cstring, this)
end
```

The resurrection replaces that pair with illegal Julia.

The comment immediately above the docstring rewrite still claims:

> Every `char*` return is emitted through `_cstring_wrapper_pair` on BOTH
> dispatch paths.

That sentence becomes false the moment `elseif is_virtual` fires.

---

## 2. Receiver gates — PASS on the wrapper; FAIL at the thunk

**Policy (CHANGELOG v3.3.0, 2026-08-13).** Two gates, one decision: does this
method take `this`? `FunctionGen._has_receiver` (MLIR thunk) and
`GeneratorCpp._cpp_this_param` (Julia wrapper). They must agree on every
input. The C++ gate is one function, called from the emission loop — not
inlined, not paraphrased by tests. Itanium `_ZTh`/`_ZTv`/`_ZTc` adjustor
thunks are filtered in `extract_symbols_from_binary` so they never become
API. Constructors/destructors always have a receiver, even when
`struct_types` has no DIE for a `.cpp`-local class.

**Julia side: the gate still runs.** `this` injection is in the **shared
preamble** (~2209–2218), before MLIR, before `elseif is_virtual`. The
virtual body splices the same `param_sig` / `ccall_types` / `ccall_args`.
Clipper2 `what` already has `this` in metadata (`const Clipper2Exception*`),
so injection is a no-op. Constructors are skipped entirely
(`func_name == class_name`). The FFI-safety trap still precedes the virtual
arm, so `_UnsafeUnknown` / unnameable aggregate returns never reach it.

Itanium thunks do not sneak back in through `vtable_virtuals`: wrap only
emits functions that are already in `metadata["functions"]`, and those were
filtered at extract. Clipper2's vtable set contains D4 deleting-dtors and
`std::exception::what` / `std::type_info::*` that are **not** in metadata;
they do not grow the wrapper.

**Thunk side: a third copy of the receiver decision, and it disagrees.**
`generate_virtual_method_ir` → `get_llvm_signature` **always prepends**
`!llvm.ptr` for `this`, then maps every DWARF formal. In-class method DIEs
already list the artificial `this`. Measured on clipper2:

| method | metadata args (wrapper ccall) | DWARF formals | `thunk_*` LLVM args | C++ callee |
|---|---|---|---|---|
| `PolyPath64::Clear` | 1 (`this`) | `["PolyPath64 *"]` | **2** | 1 |
| `PolyPath64::Count` | 1 | `["const PolyPath64 *"]` | **2** | 1 |
| `PolyPath64::AddChild` | 2 (`this`, `path`) | `["PolyPath64 *", "const Path64 &"]` | **3** | 2 |
| `Clipper2Exception::what` | 1 | `["const Clipper2Exception *", "void *", "void *"]` | **4** | 1 |

IR for `Clear`:

```mlir
func.func @thunk__ZN11Clipper2Lib10PolyPath645ClearEv(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> () {
  llvm.call @_ZN11Clipper2Lib10PolyPath645ClearEv(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
  return
}
```

That is the ImGui / Itanium-thunk failure shape: two sides of the call
disagree about the argument array. On SysV a trailing extra register is
often ignored by the C++ callee, so `Clear`/`Count` can *appear* to work
with `this` in `%rdi`. `AddChild` is the same coincidence if the wrapper's
two arguments occupy the thunk's first two slots and the real path pointer
lands in `%rsi`. It is still the wrong arity at the Julia/`ccall` boundary,
and `what`'s extra `"void *", "void *"` is DWARF-parser leakage, not ABI.

FunctionGen does **not** have this bug: it synthesizes `this` from the same
metadata the wrapper uses, via `_has_receiver`. That agreement is what
`test_symbol_hygiene.jl` exists to keep. The resurrected path bypasses
FunctionGen, so the agreement test does not cover it.

---

## 3. `_dedup_method_chunks` — PASS

**Policy (v3.2.0, 2026-08-08; drop-detail 2026-08-19).** Distinct C++ symbols
can collapse to one Julia name+signature (D1/D2 dtors, `::Any`-collapsed
overloads). Method overwriting is a hard error under package precompilation.
Dedup runs on the final chunks, last definition kept, and names the mangled
symbol it dropped by reading the chunk's own docstring.

**Still on the write path.** `func_chunks = _dedup_method_chunks(func_chunks)`
is after the emission loop, before `_dispatch_facts`, `_aot_thunk_slot_chunk`,
`_assert_aot_thunks_present`, and `_export_statement`. Virtual chunks have the
same `Mangled symbol: \`…\`` docstring line, so drop messages still name a
symbol.

`takes_virtual_path` skips generic MLIR and `continue`s out of that arm, so a
function cannot be emitted twice (once generic, once virtual). The
partial-overlap hazard the Cstring pair introduced — chunk `{f, f_ptr}` vs
chunk `{f}` sharing one key — does not arise here, because the virtual arm
does not emit a pair (see §1). If a later fix adds `_cstring_wrapper_pair` to
this arm, `_method_sig_keys` is already plural and `_dedup_method_chunks`
already keeps a two-function chunk when any of its keys is new. That
behaviour is tested; it does not need to change for this branch.

---

## 4. `Base.`-qualification — PASS as written

**Policy (2026-08-05).** A generated module is a namespace the library
populates. `error` is rebindable (`std::codecvt_base::result::error` on
llama.cpp; `struct error` on cJSON). Generator diagnostics must be
`Base.error("…")`. `_assert_base_calls_qualified` refuses to write a wrapper
containing an unqualified call whose first argument is a string literal —
the emitted text, not the generator's opinion of what it emitted.

The resurrected body contains no `error(` / `all(` of its own. The
FFI-safety trap that *precedes* it already uses `Base.error("""…""")`. The
guard still runs on the whole file at write time, so a future line in this
arm that does `error("missing thunk")` will be refused rather than shipped.
Nothing in the current body needs qualifying. No change required for this
policy; no exemption either.

`get_jit_thunk` (the AOT-off arm of the same `elseif`) also emits no
diagnostic. That arm is separately unreachable for ordinary throwing C++
virtuals — see [AOT-off](#aot-off-the-jit-virtual-arm-stays-mostly-dead).

---

## 5. Docstring return-type rewrite — FAIL

**Policy (same 2026-08-12 commit as the Cstring pair).** The C++ generator
used to advertise `-> Cstring` while the ccall path already returned
`Union{String,Nothing}` (33 llamacpp functions). Both generators now rewrite
the documented return when `julia_type == "Cstring"`.

The rewrite is **unconditional and early** (~2556–2563), before any dispatch
arm:

```julia
doc_ret = String(return_type["julia_type"])
doc_ret == "Cstring" && (doc_ret = "Union{String,Nothing}")
```

The comment says this is safe *because* both dispatch paths go through
`_cstring_wrapper_pair`. The resurrected path does not. Result for `what`:

| surface | today (generic AOT) | resurrected branch |
|---|---|---|
| docstring | `-> Union{String,Nothing}` | `-> Union{String,Nothing}` |
| function annotation | `::Union{String,Nothing}` | `::const char*` (illegal) |
| `_ptr` sibling + its docstring | present | absent |

The docs tell the truth about a function the body no longer implements.
On clipper2 the parse guard fires first, so the lying docstring never
reaches a user. On a virtual method whose `c_type` sanitized to a legal
Julia type other than `Cstring` (`Ptr{UInt8}`, a known struct pointer),
parse would succeed, the policy guard would not fire (return position is
not `Cstring`), and the docs would be the only remaining witness — the
original 2026-08-12 defect, inverted.

---

## Adjacent collisions (load-bearing, even though they were not named)

### The vcall producer, and the 2026-07-17 routing fix

This is the architectural one. CHANGELOG v3.0.1, 2026-07-17; write-up
[`docs/updates/2026-07-17-multiple-inheritance-and-vcall.md`](2026-07-17-multiple-inheritance-and-vcall.md).

Two decisions landed the same day, in this order:

1. **Routing.** Wrapper-needed virtuals go through FunctionGen's ciface
   pass, *not* through `generate_virtual_method_ir`. The legacy pass emits
   `thunk_<mangled>` direct-call wrappers **nothing ever looks up**. The
   comment is still in `JLCSIRGenerator.jl` (281–289) and still names that
   as the reason the MI fixture could not call a virtual instance method.
2. **vcall producer.** FunctionGen then emits `jlcs.vcall` (vptr → slot →
   indirect call, `may_throw` matching try_call) so a base-class wrapper
   invoked on a derived object reaches the **override**. Direct
   `llvm.call @<mangled>` is `p->Class::method()` static semantics.
   Destructors are excluded by design (Managed / RAII need exact-class
   dtors). Proven on `test/mi_test`.

The resurrected AOT arm does the opposite of (1) on purpose: it looks up
the symbol the 2026-07-17 comment says nothing should look up. And that
symbol is still produced by (the opposite of) (2):

```mlir
%result = llvm.call @_ZNK11Clipper2Lib17Clipper2Exception4whatEv(%arg0, %arg1, %arg2, %arg3)
```

No `jlcs.vcall`. No slot. No vptr. The GeneratorCpp comment on the AOT arm
says "the statically generated MLIR thunk which natively handles the vtable
math". That sentence describes FunctionGen's vcall producer. It does not
describe `generate_virtual_method_ir`.

JIT with a thunk manifest already does the right thing: `needed_symbols`
keeps wrapper-needed virtuals *out* of `gen_pre`, they land in
`fthunk_decls`, FunctionGen emits `_mlir_ciface_<mangled>_thunk` with
`jlcs.vcall`, the wrapper `invoke`s that name. AOT never passes
`needed_symbols` (`ThunkBuilder.jl:30`). That is why AOT still emits
`thunk_*` for virtuals, and why the wrapper's generic path cannot find
them.

Matching the wrapper to `thunk_*` papers over the missing symbol by
reverting the routing fix and dropping override-honoring dispatch. Passing
the manifest into AOT, keeping virtuals on the generic Tier-2 arm
(`invoke_aot_ptr` + `_cstring_wrapper_pair` + `_FPTR_*`), and letting
FunctionGen emit vcall, is the same derivation the JIT path already uses.

### `_assert_aot_thunks_present` does not see this arm

The wrap-time presence check (same uncommitted diff) shares one derivation
with the slot table: scan `_FPTR_\w+` in the final text, format
`_mlir_ciface_$(taken[slot])_thunk`. The virtual AOT arm does not emit
`_FPTR_*` and does not call `invoke_aot_ptr`. After resurrection the
clipper2 seven **leave the checked set**. The check was written to catch
those seven; the resurrection makes it stop seeing them.

If a `thunk_<mangled>` is missing, wrap succeeds, `ccall` fails at the call
site. Shape 3 in `_dispatch_facts` already classifies
`THUNKS_LIBRARY_PATH` as Tier 2 — the classifier was taught a dead branch
so it would not mis-file it — but classification is not presence.

### AOT-off: the JIT virtual arm stays mostly dead

`takes_virtual_path` is gated on `aot_thunks`. C++ functions that may throw
are never ccall-safe (`DispatchLogic.is_ccall_safe`). So with AOT off,
virtuals take generic MLIR JIT (`invoke("_mlir_ciface_…")`), never
`elseif is_virtual`. That generic JIT path, *with a manifest*, is the
vcall path. Good.

The JIT half of `elseif is_virtual` (`get_jit_thunk("$cls_name",
"$func_name")`) looks up `$(safe_class)_$(safe_method)`, which is not
`thunk_<mangled>` and is not `_mlir_ciface_<mangled>_thunk`. Nothing in
production emits that name. Even if a `noexcept` virtual reached this arm,
the lookup would miss.

### FFI trap / `Any` ccall returns

The `_UnsafeUnknown` / unnameable-aggregate trap is *before* `elseif
is_virtual` and uses `Base.error`. `_assert_no_any_ccall_return` still
scans the file. A virtual whose `julia_return_type` is `Any` but that
failed the struct-return tests would set `ret_type_sig = safe_c_ret` and
could still put a non-`Any` C spelling in the ccall — or an illegal one.
Not the named-five, but the same "this arm sanitizes returns differently
from every other arm" pattern as Cstring.

---

## What a policy-faithful fix looks like

Not implemented here. The shape that does not reopen the 2026-07-17 and
2026-08-12 bugs:

1. **Do not resurrect `elseif is_virtual` as an AOT emission path.** Leave
   virtual methods on the generic Tier-2 arm, which already has
   `_cstring_wrapper_pair`, `_FPTR_*` / `invoke_aot_ptr`, `Base.`-qualified
   traps, and the docstring rewrite agreeing with the body.
2. **Give AOT the same manifest JIT already has.**
   `ThunkBuilder.build_aot_thunks` should pass `needed_symbols` into
   `generate_jlcs_ir`. Virtuals the wrapper needs then go through
   FunctionGen (`_mlir_ciface_<mangled>_thunk` + `jlcs.vcall`), not
   `generate_virtual_method_ir`.
3. **Keep `_aot_thunk_symbol` as `_mlir_ciface_$(mangled)_thunk`.** One
   spelling, wrap-time presence check included. The clipper2 seven become
   ordinary missing-ciface failures until AOT is rebuilt with the
   manifest — which is the disagreement `_assert_aot_thunks_present` was
   written to stop at wrap, not at first call.
4. **Delete or quarantine the dead `elseif is_virtual` body**, including
   the JIT `get_jit_thunk` arm that looks up a name nothing emits. A
   classifier (`_dispatch_facts` shape 3) that exists to recognise a
   branch should not outlive the branch.
5. If someone still wants a dedicated virtual arm: it must call
   `_cstring_wrapper_pair` with two thunk-call bodies, use the same
   `_FPTR_*` slot derivation as the generic AOT arm, keep `this` from
   `_cpp_this_param`, and bind a FunctionGen/vcall symbol — not
   `thunk_<mangled>`. A fifth copy of the Cstring presentation is how
   the 75-function leak happened.

Sourcing virtuality from `parse_vtables` is the right *question* (prefer
the artifact over the never-written bookkeeping). Using that answer to
steer functions onto a branch that missed every later policy is the wrong
*use* of the answer.

---

## Pins (reproducible from this tree)

```text
# metadata never carries is_virtual
# Hub C++ packages: 0 of 2777; mi_test: 0 of 31

# clipper2 intersection (the missing seven)
parse_vtables(libclipper2.so)        → 25 unique virtual mangled names
∩ compilation_metadata.json         → 7
nm -D libclipper2_thunks.so         → thunk_<those 7> defined,
                                       _mlir_ciface_<those 7>_thunk absent

# generate_virtual_method_ir arity (prepended this + DWARF formals)
Clear:    2 LLVM args vs 1 metadata / 1 C++
Count:    2 vs 1 / 1
AddChild: 3 vs 2 / 2
what:     4 vs 1 / 1   (DWARF formals include two extra void*)

# shipped what() is already the Cstring pair on generic AOT
Clipper2.jl:6112   function …_what(this::Any)::Union{String,Nothing}
                   ptr = invoke_aot_ptr(_FPTR_…[], Cstring, this)
```

Local uncommitted files at audit time: `src/Wrapper/Cpp/GeneratorCpp.jl`,
`src/Wrapper/Utils.jl`, `test/test_introspection.jl`,
`test/ab_virtual_dispatch.jl` (untracked A/B harness; restores Hub
`packages/*/julia/` in a `finally`).
