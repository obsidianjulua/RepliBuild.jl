# Engine audit — one fact, two derivations

**Date:** 2026-08-29
**Scope:** RepliBuild.jl working tree, `main` (`8deddfe`). One failure class
only: a single fact computed in two places from different inputs, with
nothing reconciling them. Both sites are individually reasonable. They
disagree on some input, and the disagreement is silent.

**Not findings:** receiver gates (`_has_receiver` / `_cpp_this_param`,
guarded by `test_symbol_hygiene`), DW_OP_constu vs DW_OP_lit (both parsers
accept both now), virtuality routing (AOT now takes the wrap manifest /
metadata function list as `wrappable`), `VERSION` (derived from
`Project.toml`), `_aot_thunk_symbol` / `_cstring_wrapper_pair` / post-dedup
slot scan. Tier 1 quarantine, parked LTO, `is_struct_packed` over-classify,
ingest C-only, `RESERVED_UNUSED` config — deliberate.

Four confirmed instances of this class already landed and were used only to
calibrate. Three remaining splits, ranked by confidence.

---

## FINDING 1 — “Is this a constructor?” uses the full class string in one place and the bare name in another

**FACT:** whether a C++ function is a constructor of its class.

**SITE A:** [`src/Wrapper/Cpp/GeneratorCpp.jl:47-50`](../../src/Wrapper/Cpp/GeneratorCpp.jl) —
`_is_ctor_or_dtor_cpp`: `func_name == _bare_type_name_cpp(class_name)`
(innermost scope, templates stripped). Same rule as
`FunctionGen._is_ctor_or_dtor`. Feeds `this` injection.

**SITE B:** same file, [2154-2156](../../src/Wrapper/Cpp/GeneratorCpp.jl)
(emission skip) and [382](../../src/Wrapper/Cpp/GeneratorCpp.jl) (factory
collect): `func["name"] == class_name` (full DWARF class, including `ns::`).
`_collect_class_raii`'s copy-ctor test
([`src/IRGen/JLCSIRGenerator.jl:55`](../../src/IRGen/JLCSIRGenerator.jl))
is the same full-string idea: `occursin("$(cls)::$(cls)(", demangled)`.

**DIVERGE:** `tinyxml2::XMLComment::XMLComment`. Name `"XMLComment"`, class
`"tinyxml2::XMLComment"`. Gate is true; skip/collect is false. Copy-ctor
matcher looks for `tinyxml2::XMLComment::tinyxml2::XMLComment(` in a
demangle that is `tinyxml2::XMLComment::XMLComment(…)`.

**CONSEQUENCE:** silent. The skip comment says constructors are withheld
for factory handling. Namespaced ctors still emit as ordinary methods
(`tinyxml2_XMLComment_XMLComment` is in the live Hub wrapper). Factory
collect misses them. Scope-RAII never records a copy ctor for a
namespaced class.

**REPRO:**

```julia
using RepliBuild
W = RepliBuild.Wrapper
W._is_ctor_or_dtor_cpp("tinyxml2::XMLComment", "XMLComment")  # true
"XMLComment" == "tinyxml2::XMLComment"                       # false
occursin("pugi::xml_node::pugi::xml_node(",
         "pugi::xml_node::xml_node(pugi::xml_node const&)")  # false
```

**CONFIDENCE:** high — wrapper export is on disk. Did not verify a live
by-value `xml_node` param taking the RAII path without a copy ctor.

---

## FINDING 2 — C `char` / `char*` maps to two different Julia types

**FACT:** Julia type of C `char` and `char*`.

**SITE A:** [`src/Builder/Compiler.jl:2264`](../../src/Builder/Compiler.jl)
`dwarf_type_to_julia` — `_DWARF_TYPE_MAP["char"] = "Cchar"` (2187);
`char*` special-cased to `Cstring` (2281–2283). Used for **returns**.

**SITE B:** same file, [5158](../../src/Builder/Compiler.jl)
`cpp_to_julia_type` — `_CPP_TO_JULIA_TYPE_MAP["char"] = "UInt8"` (5121);
pointer peel at 5241–5248 runs **before** the map, so the map entries
`"char*" => "Cstring"` / `"const char*" => "Cstring"` (5128–5129) are
never hit. Used for **parameters**.

**DIVERGE:** `"char"` → `Cchar` vs `UInt8`. `"char*"` / `"const char*"` →
`Cstring` vs `Ptr{UInt8}`. `"signed char"` → `Cchar` vs `Any`.

**CONSEQUENCE:** silent. Live Hub: tinyxml2 `isVoidElement` param
`const char*` is `Ptr{UInt8}` in metadata; `char*` returns go through
site A as `Cstring`. ccall still converts `String` to `Ptr{UInt8}`, so
nothing throws. The Cstring policy and the param type are two answers to
the same C spelling.

**REPRO:**

```julia
using RepliBuild
C = RepliBuild.Compiler
(C.dwarf_type_to_julia("char"),        C.cpp_to_julia_type("char"))
# ("Cchar", "UInt8")
(C.dwarf_type_to_julia("char*"),       C.cpp_to_julia_type("char*"))
# ("Cstring", "Ptr{UInt8}")
(C.dwarf_type_to_julia("const char*"), C.cpp_to_julia_type("const char*"))
```

**CONFIDENCE:** high — functions exist; tinyxml2 metadata matches. Did not
check whether any ccall of a `char*` param mis-converts a Julia `Cchar`
vs `UInt8` at the boundary.

---

## FINDING 3 — STL container byte size is hardcoded twice, and `string_view` hits only one

**FACT:** SysV x86_64 byte size of an STL container spelling.

**SITE A:** [`src/IRGen/ir_gen/TypeUtils.jl:118-125`](../../src/IRGen/ir_gen/TypeUtils.jl)
— FunctionGen thunk fallback when DWARF has no struct.
`startswith(clean, "std::string")` and `startswith(clean, "string")`.

**SITE B:** [`src/Wrapper/Cpp/TypesCpp.jl:104-114`](../../src/Wrapper/Cpp/TypesCpp.jl)
— wrapper allocation / `is_stl_container_type`. Uses `_stl_name_match`
(66–71) specifically so `"std::string_view"` does **not** match
`"std::string"`.

**DIVERGE:** `"std::string_view"`, `"string_view"`, `"string"` → TypeUtils
**32**, TypesCpp **0**. `std::string` is 32 on both.

**CONSEQUENCE:** silent until a by-value `string_view` (or a DWARF-stripped
`string`) crosses FFI. FunctionGen will emit a 32-byte blob thunk
(`_byte_blob_type`); the wrapper will not treat it as an STL container
(`is_stl_container_type` is false, size 0). Same shape as the ImGui
receiver miss: thunk and wrapper disagree about the buffer.

**REPRO:**

```julia
using RepliBuild
a = RepliBuild.JLCSIRGenerator.TypeUtils.get_stl_container_size
b = RepliBuild.Wrapper.get_stl_container_size
(a("std::string_view"), b("std::string_view"))  # (32, 0)
(a("std::string"),      b("std::string"))       # (32, 32)
```

**CONFIDENCE:** high on the functions; medium on blast radius — grepped
llamacpp `compilation_metadata.json` and found **no** by-value
`string_view` in function signatures. Did not wrap a fixture that
returns `std::string_view`.

---

## Looked at, not a finding

| Site | Why it is not this class (anymore) |
|---|---|
| `FunctionGen._has_receiver` / `_cpp_this_param` | Two copies, but `test_symbol_hygiene` drives both over the Hub corpus |
| DW_OP_constu vs DW_OP_lit | Both `DWARFParser.jl` and `Compiler.jl` accept both spellings |
| `is_virtual` metadata vs `parse_vtables` | AOT routing now uses wrap's `wrappable` set; the `elseif is_virtual` branch is still dead, but wrap and AOT no longer ask for different symbols |
| `const VERSION` | Derived from `Project.toml` |
| `_aot_thunk_symbol`, `_cstring_wrapper_pair`, `_aot_thunk_slot_names` | One derivation, shared by emitter and checker |
| `_sanitize_mlir_symbol` vs `_sanitize_cpp_type_name` | They disagree on `>` / `*` (trailing `_`, `Ptr` vs `star`), but no consumer joins the two namespaces |
| union/bitfield `parse(Int, byte_size_str)` vs `_parse_dwarf_size` | On Julia 1.12 `parse(Int, "0x70")` is 112; they agree here. Not shown to disagree on this toolchain |
