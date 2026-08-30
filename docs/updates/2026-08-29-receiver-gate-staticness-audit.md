# Audit — receiver gate staticness, test integrity, coverage ledger

**Date:** 2026-08-29
**Scope:** Verify claims about RepliBuild.jl's C++ receiver gate against the
Hub (`RepliBuild-Hub/packages`). Independent engine audits in Part B.
**Not in scope:** Fixes, commits, Hub rebuilds, remote git on RepliBuild.jl.

Method: each task was finished before the next was started. Disagreement is
cited to a file:line or a command output. The previous agent's proposed
`DW_AT_object_pointer` fix is **not** rejected on the static/instance split
(that part survives), but it has a DIE-kind trap that would sink a
declaration-only implementation.

---

# Part A — the receiver gate

## TASK 1: CONFIRMED

**Claim:** `is_method` is a string test for `::` in the demangled name prefix,
true for static members too; neither gate receives a staticness input; the
decision is `(class_name, func_name, struct_types)` alone.

**EVIDENCE:**

`src/Builder/Compiler.jl:4784`:

```julia
"is_method" => contains(_name_prefix(demangled), "::"),
```

Serialized function records have no staticness field. Keys on a live Hub
record (`tinyxml2` `ErrorIDToName`): `class`, `demangled`, `exported`,
`is_method`, `is_noexcept`, `is_vararg`, `mangled`, `name`, `parameters`,
`parameters_source`, `return_type`, `return_type_source`. No `is_static`,
no `object_pointer`, no `artificial`.

`_cpp_this_param` (`GeneratorCpp.jl:118-142`) takes
`(class_name, func_name, struct_types)` and synthesizes iff the innermost
scope (or its sanitized form) is in `struct_types`, or
`_is_ctor_or_dtor_cpp` fires.

`_has_receiver` (`FunctionGen.jl:50-66`) takes `(func, structs)` and returns
true iff `_is_ctor_or_dtor(func)` or any `_scope_suffixes(class)` fuzzy-matches
`structs`.

Call sites (`GeneratorCpp.jl:2162`, `FunctionGen.jl:204`) only test
`params[1]["name"] == "this"` before synthesizing.

**NOTES:** The FunctionGen docstring at `FunctionGen.jl:31-40` already states
that `is_method` is true for both namespace-scoped free functions and static
members, and that DWARF's `DW_TAG_namespace` vs `DW_TAG_class_type` /
`DW_AT_object_pointer` is not recorded. That is an accurate description of
the gap, not a fix.

---

## TASK 2: CONFIRMED

**Claim:** tinyxml2 upstream `ErrorIDToName` is a one-parameter static, the
emitted wrapper takes two (`this`, `errorID`).

**EVIDENCE:**

Upstream, `packages/tinyxml2/.replibuild_cache/deps/tinyxml2/tinyxml2.h:1893`:

```cpp
static const char* ErrorIDToName(XMLError errorID);
```

Emitted, `packages/tinyxml2/julia/Tinyxml2.jl:3212-3214`:

```julia
function tinyxml2_XMLDocument_ErrorIDToName(this::Any, errorID::Any)::Union{String,Nothing}
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    ptr = RepliBuild.JITManager.invoke("_mlir_ciface__ZN8tinyxml211XMLDocument13ErrorIDToNameENS_8XMLErrorE_thunk", Cstring, this, errorID)
```

Metadata (`parameters_source: "dwarf"`) has a single parameter `errorID:XMLError`
and `"is_method": true`, `"class": "tinyxml2::XMLDocument"`. The gate then
injects `this`.

The same pattern is live on the other two tinyxml2 synthesis sites:

- `tinyxml2.h:1696` `static void DeleteAttribute(XMLAttribute* attribute);`
  → `Tinyxml2.jl:1923` `function tinyxml2_XMLElement_DeleteAttribute(this::Any, attribute::Any)`
- `tinyxml2.h:973` `static void DeleteNode(XMLNode* node);`
  (instance overload `XMLDocument::DeleteNode` correctly has `this` in metadata
  and was not in the sweep)

---

## TASK 3: CONFIRMED (both carry the phantom)

**Claim (inferred, not previously measured):** the MLIR thunk reads 2 argument
slots, so both generators agree on the wrong arity.

**EVIDENCE:**

Current dump `packages/tinyxml2/.debug/mlir/jlcs_e31a014807af0705.mlir:1238-1248`:

```
func.func private @_ZN8tinyxml211XMLDocument13ErrorIDToNameENS_8XMLErrorE(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
func.func @_ZN8tinyxml211XMLDocument13ErrorIDToNameENS_8XMLErrorE_thunk(%args_ptr: !llvm.ptr) -> !llvm.ptr
  %idx_1 = arith.constant 0 : i64    # slot 0 — phantom this
  ...
  %idx_2 = arith.constant 1 : i64    # slot 1 — errorID, displaced
  %ret_val = jlcs.try_call %val_1, %val_2 { callee = @_ZN8tinyxml211XMLDocument13ErrorIDToNameENS_8XMLErrorE } : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
```

**Slot count: 2.** An older dump (`jlcs_2d8605cf1bf84bdc.mlir:920-926`) is the
same arity via `jlcs.get_field` at offsets 0 and 8.

The two generators agree. This is worse than a one-sided bug: Julia passes two
values, the thunk loads two slots, SysV places the phantom pointer in `%rdi`
and the real `XMLError` in `%rsi`. Nothing downstream drops the extra argument.
Tier-2 dispatch does not tolerate it.

---

## TASK 4: CONFIRMED — with a DIE-kind trap that would sink a naive fix

**Claim:** `DW_AT_object_pointer` is present for `inst` and absent for `stat`,
in both tools' output.

**Which tool each parser reads:**

| Parser | Tool | Evidence |
|---|---|---|
| `Compiler.jl` `extract_dwarf_return_types` | **GNU readelf only** | `Compiler.jl:2492-2525`. Explicitly rejects llvm-dwarfdump (no depth, no abbrev number). No fallback. |
| `DWARFParser.jl` | **llvm-dwarfdump** | `DWARFParser.jl:79`, `452`: ``dwarf_cmd = `llvm-dwarfdump --debug-info $binary_path` `` |

`Compiler.jl:2510-2512` already records the split: two dialects, two parsers,
neither reads the other's output. Neither parser currently **extracts**
`DW_AT_object_pointer` — the string appears in `Compiler.jl` only as a comment
on the definition DIE (`Compiler.jl:4075`).

**Probe** (`/tmp/st.cpp`, `clang++ -std=c++17 -g -c`):

`llvm-dwarfdump` on the **definition** DIE of `W::inst`:

```
0x0000005f: DW_TAG_subprogram
              DW_AT_object_pointer(0x00000070)
              DW_AT_specification(0x00000032 "_ZN1W4instEi")
```

`W::stat` definition (`0x00000085`): no `DW_AT_object_pointer`.

`readelf --debug-dump=info`:

```
 <1><5f>: DW_TAG_subprogram
    <67>   DW_AT_object_pointer: <0x70>
```

`stat` definition `<1><85>`: no `DW_AT_object_pointer`.

Spelling differs:

- llvm-dwarfdump: `DW_AT_object_pointer(0x00000070)`
- readelf: `DW_AT_object_pointer: <0x70>`

A parser written against the wrong dialect matches nothing. That is exactly
the class that caused a bug earlier this month.

**NOTES — say this loudly:** the attribute is on the **definition DIE only**.
The in-class **declaration** DIE of `inst` (`<2><32>`) has `DW_AT_declaration`,
an unnamed `DW_AT_artificial` first parameter, and **no** `DW_AT_object_pointer`.
The declaration DIE of `stat` has neither artificial first param nor
`object_pointer`. See TASK 7.3. A fix that keys the gate on `object_pointer`
read from declaration DIEs would classify every instance method as static.

Ctor/dtor definition DIEs also carry it (separate probe `/tmp/ctor.cpp`):
`DW_AT_object_pointer` present on `_ZN1WC2Ei` and `_ZN1WD2Ev`, absent on
`_ZN1W4statEi`. So keying the gate on definition-DIE `object_pointer` would
**not** break the 2026-08-13 ctor/dtor exception for `.cpp`-local classes;
it would make that exception redundant, provided the parser actually sees
the definition DIE.

---

## TASK 5: PARTIAL

**Claim:** 39 non-ctor/dtor synthesis sites across six C++ Hub packages, a mix
of genuine static defects and instance methods whose DWARF dropped `this`.
The other agent identified statics by recognising library names.

**EVIDENCE:** The sweep script reproduces **39** sites (clipper2 0). Classification
is from upstream declarations / Itanium mangles, not from library-name
recognition.

| # | Package | Site | Mangled / header | Category |
|---|---|---|---|---|
| 3 | tinyxml2 | `XMLElement::DeleteAttribute(attribute)` | `…DeleteAttributeEPNS_12XMLAttributeE`; `tinyxml2.h:1696` `static` | **static defect** |
| | | `XMLDocument::ErrorIDToName` | `tinyxml2.h:1893` `static` | **static defect** |
| | | `XMLNode::DeleteNode` | `tinyxml2.h:973` `static` | **static defect** |
| 2 | pugixml | `xpath_variable_set::_clone` | `pugixml.hpp:1262` `static bool _clone(...)` | **static defect** |
| | | `xpath_variable_set::_destroy` | `pugixml.hpp:1263` `static void _destroy(...)` | **static defect** |
| 6 | box2d | `b2Joint::Create` / `Destroy` | `b2_joint.h:165-166` `static` | **static defect** |
| | | `b2Contact::{InitializeRegisters,Create,AddType,Destroy}` | `b2_contact.h:200-205` `static` | **static defect** |
| 5 | imgui | `ExampleAsset::CompareWithSortSpecs` | `imgui_demo.cpp:10897` `static` | **static defect** |
| | | `ExampleAppConsole::TextEditCallbackStub` | `imgui_demo.cpp:9304` `static` | **static defect** |
| | | `ExampleDualListBox::CompareItemsByValue` | `imgui_demo.cpp:2727` `static` | **static defect** |
| | | `ExampleDualListBox::ApplySelectionRequests` | mangled `…ENUl…E_8__invokeES3_i`, params `['self','idx']` | **lambda `__invoke`**, not the instance method (that one has `this` in metadata) |
| | | `ExampleAssetsBrowser::Draw` | mangled `…ENUl…E_8__invokeES4_i`, params `['self_','idx']` | **lambda `__invoke`**; real `Draw` is `['this','title','p_open']` |
| 23 | llamacpp | `llm_tokenizer_wpm_session::preprocess` | `llama-vocab.cpp:818` `static` | **static defect** |
| | | 22 × `gguf_kv::gguf_kv<T>` | mangled `_ZN7gguf_kvC2I…` — complete-object **constructors** | **instance, synthesis correct** |

**Report: 17 static/lambda defects / 22 instance-with-dropped-this (all
template constructors).** Zero ordinary non-ctor instance methods in the 39.

**NOTES:** The sweep's ctor skip is `nm == bare` (`gguf_kv == gguf_kv`).
Template constructor names are `gguf_kv<signed char>`, so they leaked into
the 39. `_is_ctor_or_dtor_cpp` uses the same comparison against
`_bare_type_name_cpp(class)` (`GeneratorCpp.jl:50`) and would also miss
them — they still get a receiver because `gguf_kv` is in `struct_types`.
That is the number the `object_pointer` fix must not break: **22 constructor
instantiations plus every instance method that already has `this` in
metadata** (those are outside the 39; synthesis does not fire).

The other agent's implied "most of llamacpp's 23 are static" is **wrong**.
22 of 23 are constructors. Library-name recognition would have misclassified
them.

Live wrapper confirmation that static defects are not metadata-only:
`Llamacpp.jl:81887` `function llm_tokenizer_wpm_session_preprocess(this::Any, text::Any, normalizer_opts::…)`.

---

## TASK 6: 6a CONFIRMED / 6b CONFIRMED / 6c CONFIRMED / 6d CONFIRMED

Probe: `/tmp/probe.cpp`, `clang++ -std=c++26 -g -c`.

### 6a CONFIRMED — deducing `this` (C++23)

Definition DIE of `D::get` (`readelf`):

```
 <1><2d1>: DW_TAG_subprogram
    <2d9>   DW_AT_object_pointer: <0x2e2>
    <2de>   DW_AT_specification: <0x2b8>
 <2><2e2>: DW_TAG_formal_parameter
    <2e6>   DW_AT_name: self
    (no DW_AT_artificial)
```

The object parameter is named `self`, marked `DW_AT_object_pointer`, **not**
`DW_AT_artificial`. Both gates test the literal name `"this"`:

- `GeneratorCpp.jl:2162`: `has_this = !isempty(params) && (params[1]["name"] == "this")`
- `FunctionGen.jl:204`: `isempty(params) || get(params[1], "name", "") != "this"`

They will miss `self` and synthesize a second receiver. Not live in the Hub
(no C++23 explicit-object parameters). Structural.

### 6b CONFIRMED — `char8_t`

Probe DIE:

```
0x0000051c: DW_TAG_base_type
              DW_AT_name("char8_t")
              DW_AT_encoding(DW_ATE_UTF)
              DW_AT_byte_size(0x01)
```

Mapped in `src/`: `char16_t`/`char32_t`/`wchar_t` at
`TypeRegistry.jl:56`, `Compiler.jl:2191-2192` and `5166-5167`. **No `char8_t`
anywhere in `src/`.** Unmapped → `Any`.

### 6c CONFIRMED — coroutine clones are local, `-g` drops them

```
nm /tmp/probe.o | grep co_ramp
0000000000000030 T _Z7co_rampv
0000000000000320 t _Z7co_rampv.__await_suspend_wrapper__final
00000000000002d0 t _Z7co_rampv.__await_suspend_wrapper__init
0000000000000730 t _Z7co_rampv.cleanup
0000000000000540 t _Z7co_rampv.destroy
0000000000000370 t _Z7co_rampv.resume
```

`T` = global text (the ramp); `t` = local. `nm -g --defined-only` emits only
`_Z7co_rampv`. Extraction shells that at `Compiler.jl:1976`
(`extract_symbols_from_binary`) and `1637` (STL). Resume/destroy/cleanup never
reach any later filter.

**NOTES:** `Compiler.jl:1991-1996` includes lowercase `t`/`w` in
`code_symbol_types`, with a comment that this is so static C functions
surface. `nm -g` has already dropped them. The type filter cannot resurrect
what `-g` discarded. 6c's safety claim still holds; the comment is misleading.

### 6d CONFIRMED — concepts split the demanglers

Mangled: `_Z5twiceITk3NumiET_S0_` (`Tk` constraint).

| Tool | Result |
|---|---|
| `llvm-cxxfilt` | `int twice<int>(int)` |
| binutils `c++filt` | `_Z5twiceITk3NumiET_S0_` (unchanged) |
| `nm -gC` (what extraction uses, `Compiler.jl:1983`) | `_Z5twiceITk3NumiET_S0_` (unchanged) |

A constrained template whose only named spelling is the `Tk` mangle will enter
the pipeline undemangled.

---

## TASK 7 — adversarial

### 7.1 Is the current gate right for some reason the other agent missed?

**No, not for statics.** TASK 3 measured the inferred half: both generators
agree on two slots; `invoke(..., this, errorID)` passes both; the thunk loads
both; SysV puts the phantom in `%rdi`. Nothing downstream drops it. Tier-2
does not tolerate an extra slot.

The gate **is** right for the original purpose — instance methods whose DWARF
omitted `this`. That subset must keep working. The 22 `gguf_kv` template
constructors in TASK 5 are that subset.

`Compiler.jl:3878-3888` and `4248-4264` already rename a position-0
`DW_AT_artificial` unnamed parameter to `"this"`. Instance methods that
survive that path never hit synthesis. Synthesis is the remainder: DWARF
described a parameter list that does not start with `this`. Statics and
dropped-`this` instance methods are indistinguishable in that remainder
without `object_pointer` or the declaration-DIE artificial param.

### 7.2 Would keying on `DW_AT_object_pointer` break the ctor/dtor rule?

**Not if it is read from definition DIEs.** `/tmp/ctor.cpp` shows
`object_pointer` on `C2`/`D2` definition DIEs. `.cpp`-local box2d contact
classes (`b2CircleContact` and siblings — the 2026-08-13 reason
`struct_types` is not a complete list) would still have definition DIEs
with `object_pointer` even without a type DIE. The ctor/dtor exception
(`FunctionGen.jl:51-59`, `GeneratorCpp.jl:100-109`) would become redundant
rather than wrong.

**It would break them if the parser only looks at declaration DIEs**, which
do not carry the attribute (TASK 4). That is the trap. `Compiler.jl:4069-4090`
already follows `DW_AT_specification` from definition to declaration to
merge named parameters; `object_pointer` would have to be copied in that
same merge, from the definition side.

A "don't synthesize if `object_pointer` is absent" rule, fed only from
declarations, would deny `this` to every instance method. Backwards. This
audit's most valuable output.

C++23 `self` (TASK 6a): `object_pointer` **is** present and names `self`.
"Don't synthesize when `object_pointer` is present" (and treat the named
target as the receiver) would also fix the double-receiver, provided the
definition DIE is the one consulted.

### 7.3 Declaration, definition, or both?

**Definition only**, clang 22.1.8, DWARF 5, both dumpers. Declaration DIEs of
non-static members have an unnamed `DW_AT_artificial` first parameter instead.
Statics have neither.

### 7.4 Anything else in these files not covered above

- Template constructor names (`gguf_kv<T>`) are not recognised as constructors
  by `_is_ctor_or_dtor` / `_is_ctor_or_dtor_cpp` (name ≠ bare class). Harmless
  today only because the class is in `struct_types`.
- Nested-lambda `__invoke` is recorded as the enclosing method
  (`ExampleDualListBox::ApplySelectionRequests` with params `self, idx`).
  `extract_function_name` takes the last `::` component of a demangle that
  apparently does not keep `{lambda}::__invoke` as the name. The gate then
  synthesizes `ExampleDualListBox* this` onto a static closure invoker.
- Parameter dicts dropped the `artificial` flag before serialization, so even
  `_has_receiver(func, …)` cannot see it.
- `FunctionGen._scope_suffixes` is a deliberate superset of
  `_cpp_innermost_scope` (`FunctionGen.jl:47-48`). No live Hub disagreement
  (TASK 8), but the two gates can still over-admit relative to each other if
  a suffix other than the innermost is a struct key.

---

## TASK 8: PARTIAL — latent hazard, no live collision

**Claim:** `is_method` is also true for namespace-scoped free functions; the
gate declines only because the namespace is not in `struct_types`;
`_cpp_innermost_scope` takes the last `::` component, so `a::b::freefn`
synthesizes `this` if any struct is named `b`.

### 8.1 Innermost scope takes the last depth-0 `::` component

**CONFIRMED.** `GeneratorCpp.jl:67-84`, docstring at `54-58`. Template
arguments are kept (`tensor_traits<block_q4_0, 8l>`), unlike
`_bare_type_name_cpp`.

### 8.2 How many `is_method` functions are saved only by a struct miss

Angle-bracket-aware innermost, matching the Julia. `is_method` functions
whose class/innermost is **not** a `struct_definitions` key:

| Package | Saved by miss | Dominant class values |
|---|---|---|
| tinyxml2 | 54 | `tinyxml2::XMLUtil` 18, `XMLComment`/`XMLUnknown`/`XMLDeclaration` 11 each (real classes missing from the aggregate table, not namespaces) |
| pugixml | 7 | **`pugi` 7** — canonical namespace free functions (`get_memory_allocation_function`, `as_utf8`, …) |
| box2d | 42 | `b2CircleContact` and six sibling `.cpp`-local contact classes (the 2026-08-13 ctor/dtor case) |
| imgui | 853 | **`ImGui` 829** — the Dear ImGui namespace (the 2026-08-08 788-thunk bug, now saved) |
| clipper2 | 38 | **`Clipper2Lib` 28**, `Clipper2Lib::detail` 2, plus a few real classes |
| llamacpp | 753 | **`ggml::cpu::repack` 48**, plus many real `llama_model_*` subclasses missing from the table |

True namespace-scoped free functions currently saved by the miss: pugixml 7,
imgui 829, clipper2 ~30, llamacpp `ggml::cpu::repack` 48. Wrappers are
zero-arg as they should be: `Pugixml.jl:4865` `pugi_get_memory_allocation_function()`,
`Imgui.jl:24259` `ImGui_GetVersion()`.

### 8.3 Namespace/struct name collision

Exact-key test: innermost of a not-in-structs class equals a
`struct_definitions` key. Also `_has_receiver` suffix test (any scope suffix
in structs).

**No live collision in any of the six packages.** `pugi`, `ImGui`,
`Clipper2Lib`, `repack`, `detail`, `tinyxml2` are not struct keys.

llamacpp **does** have a struct key `impl`. Functions whose class is
`llama_file::impl` / `llama_vocab::impl` / … are nested **classes** named
`impl`, not a namespace `impl`. They already have `this` in metadata. That
is not the hazard.

**Verdict: latent hazard, not a live defect.** Useful. Do not conflate with
TASK 5.

A naive `split('<')` on `pugi::xml_object_range<pugi::xml_attribute_iterator>`
produces a false `xml_attribute_iterator>` "collision". The real innermost
scope is angle-bracket-aware and does not cut there.

---

# Part B — independent engine audits

## TASK 9: findings (vacuous green)

Calibration instances 1–4 were not re-reported. Tier-1's three unwired files
are out of scope per the request.

### Finding 1 (load-bearing) — `exit(0)` in files `devtests.jl` includes

`CLAUDE.md:491` records the class as **fixed**:

> Two files skipped unavailable prerequisites that way (`test_win64_abi.jl`,
> `test_multilib_jit.jl`) — […] Both are skipped testsets now […]
> **Never `exit` in a file a suite includes.**

`test_win64_abi.jl` was converted. These still `exit(0)`:

| File | Line | Guard |
|---|---|---|
| `test/test_mlir_templates.jl` | 26 | `!MLIR_AVAILABLE` |
| `test/test_jlcs_invariants.jl` | 32 | `!MLIR_AVAILABLE` |
| `test/test_jlcs_producers.jl` | 42 | `!MLIR_AVAILABLE` |
| `test/test_struct_abi.jl` | 41, 47 | `!MLIR_AVAILABLE` / no `clang++` |
| `test/test_multilib_jit.jl` | **33** | `!MLIR_AVAILABLE` |

`test_multilib_jit.jl:39-42` comments "Skip, never `exit`" — that comment
applies only to the later `WRAPPERS_BUILT` path. The `MLIR_AVAILABLE` path
still exits.

`devtests.jl:188` includes `test_mlir_templates.jl` **first** among the
libJLCS files, under a comment that it "Self-skips". If `libJLCS.so` is
missing, that include `exit(0)`s the process: callback exceptions, JLCS
invariants, Win64 ABI, in-process C, producers, struct ABI, multilib,
dwarf attribution, anonymous unions, debug inspection, sysconfiggen never
run, and the suite is **green**.

**If I deleted libJLCS, would these tests go red?** No. They would take the
rest of the suite with them and report success. Same genus as calibration
instance 2, one notch worse (it kills neighbours).

Assertion that would make it real: a skipped `@testset` (as `test_win64_abi.jl`
and the `WRAPPERS_BUILT` half of multilib already do), never `exit`.

### Finding 2 (low) — `test_registry.jl:469-472`

```julia
if !isdir(ctest_dir)
    @warn "c_test/ not found — skipping registry integration"
    return
end
```

The integration testset returns without asserting that it ran. `c_test/` is
in the tree, so this is defensive. If someone ran the file from a stripped
layout, registry integration would skip green. The rest of the file still
runs. Ranked low.

### Looked at, not findings

- `test_ingest.jl:41-47` extra_link_libs round-trip is now labeled as
  parse-only; behavioural coverage moved to `test_config_surface.jl`.
  Calibration instance 1, already fixed.
- `test_symbol_hygiene.jl:124` `if isfile(so)` live-nm check is extra; the
  vendored corpus plus `@test swept == 2` still fail if the feature is
  deleted. Calibration 2/4, already fixed.
- `test_debug_inspection.jl:68-70` `all(isfile, srcs)` is preceded by
  `@test !isempty(srcs)`.
- Empty-collection `@test isempty` / `@test all` hits that assert a
  *specific* empty result (e.g. producers given `needed_symbols=Set()`)
  would go red if the feature started emitting. Not vacuous.

---

## TASK 10: PARTIAL — percentage held; the "27 returns FIXED" claim did not

**Recorded (2026-08-20):** 5166 / 5242 = 98.6%, 16 C packages, 76 misses
(varargs 43, by-value aggregate return 27 all mpack *reported fixed*,
by-value aggregate arg 5 all mpack, zlib `gzgetc` 1).

**Re-measured 2026-08-29**, same method: exported `T`/`W` in each shipped
`.so` minus `__rb_*`, versus `ccall((:sym, …)` targets in the wrapper.

17 indexed C packages (zstd added; 202/202). Totals:

**5368 / 5444 = 98.6%**  (5166+202 / 5242+202 — zstd is a clean 100% add)

| Package | Hit/exp | % | Misses |
|---|---|---|---|
| cglm | 742/742 | 100 | 0 |
| box2d3 | 426/426 | 100 | 0 |
| cjson | 92/92 | 100 | 0 |
| lua | 242/245 | 98.8 | 3 |
| duktape | 326/329 | 99.1 | 3 |
| zlib | 123/125 | 98.4 | 2 |
| xxhash | 50/50 | 100 | 0 |
| tomlc17 | 10/10 | 100 | 0 |
| lz4 | 145/145 | 100 | 0 |
| miniz | 117/117 | 100 | 0 |
| mpack | 299/331 | 90.3 | **32** |
| sqlite | 298/306 | 97.4 | 8 |
| blake3 | 23/23 | 100 | 0 |
| miniaudio | 1177/1178 | 99.9 | 1 |
| pcre2 | 119/119 | 100 | 0 |
| zstd | 202/202 | 100 | 0 |
| curl | 977/1004 | 97.3 | 27 |
| **total** | **5368/5444** | **98.6** | **76** |

Ten of seventeen at 100% (recorded nine of sixteen). Miss count still **76**.

### Classification of the 76 — the recorded split still holds

**Varargs (43), unchanged:**
- curl 27: `Curl_failf`, `Curl_infof`, ten `Curl_trc_*`, `Curl_getinfo`,
  `Curl_mime_add_header`, `Curl_pp_sendf`, `curlx_dyn_addf`, plus public
  `curl_easy_setopt`/`getinfo`, `curl_multi_setopt`, `curl_share_setopt`,
  `curl_formadd`, `curl_m{,f,s,sn,a}printf`
- sqlite 8: `sqlite3_config`, `sqlite3_db_config`, `sqlite3_log`,
  `sqlite3_mprintf`, `sqlite3_snprintf`, `sqlite3_str_appendf`,
  `sqlite3_test_control`, `sqlite3_vtab_config`
- lua 3: `luaL_error`, `lua_gc`, `lua_pushfstring`
- duktape 3: `duk_error_raw`, `duk_push_error_object_raw`, `duk_push_sprintf`
- zlib 1: `gzprintf`
- miniaudio 1: `ma_log_postf`

**By-value aggregate return (27, all mpack) — NOT gone from the miss list.**

mpack's wrapper **was** re-wrapped after the typing work: `mpack_tag_make_nil`
is documented `-> mpack_tag_t` and then:

```julia
function mpack_tag_make_nil()
    Base.error("""
    ABI Safety Trap: cannot call 'mpack_tag_make_nil' through ccall.
    ...
    returns `mpack_tag_t` by value.
```

`Mpack.jl` has 31 `ABI Safety Trap` stubs and 313 `ccall((:mpack` sites.
Against the coverage **method** (ccall targets), the 27 are still misses.
The "FIXED" claim is about `julia_type` no longer being `Any`, not about
ccall reachability. John's 99% target is a ccall-reachability number.

The 27: `mpack_node_tag`, `mpack_peek_tag`, `mpack_read_tag`,
`mpack_tag_{array,bin,bool,double,false,float,int,map,nil,str,true,uint}`,
`mpack_tag_make_{array,bin,bool,double,false,float,int,map,nil,str,true,uint}`.

**By-value aggregate argument (5, all mpack), still open:**
`mpack_tag_cmp`, `mpack_tag_equal`, `mpack_write_tag`, `mpack_expect_tag`,
`mpack_expect_cstr_match`.

**One-off (1):** zlib `gzgetc`. Still not a ccall target. The wrapper
exposes Julia `gzgetc` which ccall's **`:gzgetc_`** (`Zlib.jl:2836`,
metadata symbol `gzgetc_`). The `.so` also exports `gzgetc` (the zlib
macro/function). Methodology-miss, possibly a shim/macro question as
recorded. Undiagnosed, not newly broken.

**Specifically: are mpack's 27 by-value-aggregate-return misses gone from
the shipped wrapper?** **No.** They are trap stubs, not ccall. mpack was
re-wrapped (trap text would not exist in a pre-fix wrapper), but the
coverage number does not move.

---

## TASK 11: CLAUDE.md derived-claim drift

Op-verifier list **held**. Almost everything else that is a number or a
line citation has moved. Pattern matches the 2026-07-25 audit.

### Op-verifier list — no drift

`src/mlir/JLCSOps.td` has 14 ops, 7 with `hasVerifier = 1`:
`type_info`, `get_field`, `set_field`, `vcall`, `dtor_call`, `scope`,
`marshal_arg`. Verifier-less: `load_array_element`, `store_array_element`,
`ffe_call`, `try_call`, `ctor_call`, `yield`, `marshal_ret`.
Matches `CLAUDE.md:460`.

### Line-number citations

| Claim | Recorded | Actual | Status |
|---|---|---|---|
| `default_lto = language==:c` | `ConfigurationManager.jl:335` | `:357` (`:335` is `parse_compile_config`'s `get(data, "compile")`) | **stale** — already re-anchored 311→335 in the 2026-07-25 audit, drifted again |
| `use_mlir_dispatch` continue past ccall | `GeneratorCpp.jl:2531` | `:2547` (`if use_mlir_dispatch`); `:2531` is callback-doc assembly | **stale** |
| `is_ccall_safe` noexcept route | `DispatchLogic.jl:151` | `:151-152` the `if` / `return false` | holds |
| `GeneratorC` rewrites `doc_ret` | `GeneratorC.jl:2659` | `:2365` (`doc_ret == "Cstring" && (doc_ret = "Union{String,Nothing}")`); `:2659` is llvmcall `join` | **stale** |
| dtor `~` test, same as `_collect_class_raii` | `GeneratorCpp.jl:125` | `:49` (`startswith(func_name, "~")`); `:125` is `_cpp_this_param` docstring | **stale** |
| `enableGDBNotificationListener = true` | `JLCSCAPIWrappers.cpp:343` | `:481`; `:343` is `mlirIntegerTypeGetWidth` | **stale** |
| FileLineColLoc note | `MLIRNative.jl:122` | docstring `:123-128` | close enough |
| `clean()` regenerates `.debug` from module text | `RepliBuild.jl:701` | `clean_internal` `:650-653`; `:701` is `lib_files = filter(...)` inside `info()` | **stale** |
| Managed finalizers / DWARF dtors | `GeneratorCpp.jl:1638` | Managed emission `:1767`/`:1909`; `:1638` is NTuple degrade for unknown members. Re-anchored 1476→1638 on 2026-07-25, drifted again | **stale** |
| parameter-path `cpp_to_julia_type` | `~Compiler.jl:4229` | resolve + map `:4254-4256`; `:4229` is `parent_subroutine = get(...)` | drifted, `~` saved it |
| `HOME` into subprocess env | `test_tier1_dispatch.jl:238` | `:238` `"HOME" => homedir()` | holds |
| `exit(0)` class fixed in `test_multilib_jit.jl` | `CLAUDE.md:491` | still `exit(0)` at `test_multilib_jit.jl:33` on `!MLIR_AVAILABLE` | **stale (the claim that it was fixed)** |

### Test counts (`N/N`) — internally inconsistent; toolchain tests not re-run

| Claim | Recorded values in CLAUDE.md | Source `@test` lines (not a run) | Notes |
|---|---|---|---|
| mi_test | 31/31, 38/38, 43/43 | `verify.jl` 49 `@test` | three recorded values; **not re-run** (needs rebuild/JIT) |
| vi_test | 33/33, 40/40 | `verify.jl` 42 `@test` | **not re-run** |
| stl_test | 28/28 | `verify.jl` 36 `@test` | **not re-run** |
| test_symbol_hygiene | 97/97 | 32 `@test` lines, several in loops | file claims 97; standalone needs `using RepliBuild` from `runtests.jl`. **not re-run as a suite** |
| test_jlcs_producers | 26/26 and 57/57 | 61 `@test` | **not re-run** |
| test_struct_abi | 15/15 and 30/30 | 35 `@test` | **not re-run** |
| test_multilib_jit | 14/14 | 14 live `@test` + `@test_skip` | source matches 14; **not re-run** |
| test_dwarf_attribution | 18 asserts, suite 31/31 | 41 `@test` | **not re-run** |
| test_anonymous_unions | 35/35 | 41 `@test` | **not re-run** |
| test_tier1_dispatch | 42/42, 64 asserts, 95/95, 108/108, 254/254 | (quarantined / unwired from some paths) | recorded values contradict each other |
| Hub `test_deep.jl` | imgui 202, llamacpp 33, lua 172, … | **not re-run** (would need Hub drivers; `test.jl` is a rebuild) | |

`test_symbol_hygiene` standalone (`julia --project=. test/test_symbol_hygiene.jl`)
errors `UndefVarError: RepliBuild` — it is runtests-owned. That is not a
vacuous-green finding; it is just not a valid standalone entry point.

### Package / symbol / export counts

| Claim | Recorded | Actual | File:line of claim |
|---|---|---|---|
| Hub sweep "all 20 packages" | 20 | `index.toml` **24** entries | `CLAUDE.md:373` (historical 2026-08-12; still reads as current inventory if you stop there) |
| indexed / on disk | "24 indexed, 25 on disk" was requested as a category; **that exact phrase is not in CLAUDE.md** | indexed **24**, on disk **26** (`addc`, `hello_world` unindexed) | — |
| C packages in the ccall ledger | 16 | **17** (zstd) | `CLAUDE.md:518` |
| hello_world / tinyxml2 / pugixml / box2d / imgui / llamacpp exports | 6 / 313 / 518 / 680 / 3040 / 5615 (`CLAUDE.md:346`) | unique `export` names 5 / 312 / 517 / 679 / **3106** / 5614 | imgui is the real move (+66); others ±1 may be counting method) |
| "zero zero-arg method wrappers Hub-wide"; survivors pugixml `pugi::get_memory_*` | `CLAUDE.md:336` | those two still exist (`Pugixml.jl:4865`, `:4915`). Wrapper `function foo()` hits are dominated by generated Julia struct default constructors, not C++ method wrappers. Did not exhaustively re-prove the "zero method wrappers" half | **partial** |
| ccall coverage | 5166/5242 = 98.6% | 5368/5444 = 98.6%, same 76 misses | `CLAUDE.md:518` — percentage held, inputs grew by zstd |

### Version claims

| Claim | Recorded | Actual |
|---|---|---|
| Julia | 1.12.6 (`CLAUDE.md:687-688`) | **1.12.7** |
| `Base.libllvm_version` | 18.1.7 | 18.1.7 |
| system LLVM/clang | 22.1.8 | clang 22.1.8, `llvm-config` 22.1.8 |
| `Project.toml` / `RepliBuild.VERSION` | "both `3.1.0`, no drift" (`CLAUDE.md:687`) | **3.3.3** |
| C++ / Tier 2 "22.1.6 here" | `CLAUDE.md:690` | 22.1.8 (the bump paragraph below it already moved to 22.1.8; this bullet did not) |

LLVM/MLIR 22.1.8 itself held. Julia patch and the package version did not.

---

# What this means for the waiting engine fix

The static-member diagnosis **survives**. The live tinyxml2 wrapper and its
thunk are a measured two-slot phantom, not an inference. The 17 Hub statics /
lambda invokers are real defects. The 22 `gguf_kv<T>` constructors are not.

`DW_AT_object_pointer` **does** distinguish `inst` from `stat` in both
readelf and llvm-dwarfdump. It is the right bit. It lives on the
**definition DIE only**, and the two parsers speak different dialects.
A declaration-only implementation of the proposed fix is backwards — it
would strip `this` from every instance method, including the box2d
`.cpp`-local dtors the 2026-08-13 exception exists to protect.

`Compiler.jl` already walks definition DIEs via `DW_AT_specification`
(`:4069-4090`). That is the seam. Readelf spelling is
`DW_AT_object_pointer: <0x70>`. Do not copy an llvm-dwarfdump regex into
the readelf parser.

Do not key the Julia generator on `"this"` as the only receiver name if
C++23 explicit-object parameters are in scope (TASK 6a); `object_pointer`'s
target is the name to keep.
