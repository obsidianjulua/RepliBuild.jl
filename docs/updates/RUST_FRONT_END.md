# A Rust front end: design and measurements

Measured 2026-10-31 on Windows (MSYS2 CLANG64). Nothing in `src/` or `test/`
changed. The question is how RepliBuild wraps a Rust crate with no source
edits: it owns the build, takes types from the DWARF of that build, takes call
shapes from the LLVM IR rustc emits, and calls plain Rust functions with the
compiler's own lowering. This document also sets the order in which to build
that.

Everything below was measured on one scratch crate, `rbprobe` (Appendix A),
and a prototype of the generated sibling crate, `rbprobe_rb` (Appendix B),
unless it is marked as read or inferred. Linux was not measured (§9).

---

## 0. Answers first

1. **DWARF gives an exact layout for every type, including `repr(Rust)` field
   reordering.** Enums with data arrive as `DW_TAG_variant_part`, with the
   discriminant's location and every variant's value. Fat pointers are named
   structures (`&str`, `&dyn rbprobe::Area`). Thin pointers carry their Rust
   spelling (`&mut …`, `alloc::boxed::Box<…>`). Vtables are named variables
   with an address. Run over this dump, the existing parser reads plain
   records, tuples, fat pointers and fieldless enums exactly. It flattens every
   data enum into overlapping members and drops the discriminant. It merges
   every `Some`/`None`/`Ok`/`Err` by bare name, and it reduces `&T`, `&mut T`
   and `Box<T>` to `T*`.
2. **A `define` line alone is enough to make a correct call. It is not enough
   to make a wrapper.** All 12 probes that called with exactly the define
   line's types returned the right value. The 3 that called the same functions
   through `ccall`'s C-ABI lowering were wrong: `{ double, double }` and
   `{ i16, i16 }` returns, and `u128`. What the define line cannot say is what a `ptr` means, a borrow
   or a move. DWARF says that, one formal parameter per Rust parameter, and
   each one maps to 0, 1 or 2 IR parameters. What each crate type exports:
   - a `cdylib`: only its 5 `extern "C"` functions;
   - a `dylib` at opt-level 0: all 49 crate items, plus std;
   - a `dylib` at opt-level 3: 18 of the 46 non-generic functions.

   Mangling is v0 only.
3. **MSYS2's rustc bundles no LLVM: it loads the system `libLLVM-23.dll`.**
   Julia's LLVM 20 cannot parse rustc 1.99's define lines (`dead_on_return`).
   The plan never loads rustc's IR into any LLVM. RepliBuild reads the define
   lines as text, as it already does for clang's, and cargo and rustc do the
   link.
4. **What goes in the TOML.**
   - **Generic instantiations** are declared, with path and signature, into
     a generated sibling crate that becomes the root of the artifact.
     rustc type-checks the declaration.
   - **Drop glue** needs no TOML. The generator emits a `drop_in_place`
     function for each type that crosses by value.
   - **Panics** must end in `catch_unwind` in a generated Rust frame.
     `C-unwind` hands the panic to Julia, which kills the process. Tier 2's
     C++ EH catches it, and Rust then aborts.
5. **Staging.**
   - **Stage 1** wraps the `extern "C"` surface through the C path. It is
     plumbing.
   - **Stage 2** makes Rust-ABI calls through Tier 2 from define lines. This
     is what makes the design more than the deleted C-ABI wrapper.
   - **Stage 3** adds the generated crate: generics, ownership and panics.

---

## 1. Environment

| | |
|---|---|
| Host | Windows 11, MSYS2 CLANG64 |
| rustc | 1.99.1 (58cf8f9c3 2026-10-13) (Rev1, Built by MSYS2 project); host `x86_64-pc-windows-gnullvm`; `LLVM version: 23.1.3` |
| cargo | 1.99.1 (5f94df478 2026-08-27) (MSYS2) |
| clang | 23.1.4 (MSYS2) |
| Julia | 1.13.3, `Base.libllvm_version` 20.1.8 |
| DWARF dumper | GNU objdump 2.47 (MSYS2's `mingw64/bin`), picked by `Compiler._dwarf_dumper` |
| Demangler | `llvm-cxxfilt` (CLANG64), which reads v0 |

Rust was not installed. `pacman -S mingw-w64-clang-x86_64-rust` added rust
1.99.1-1 and two libraries (libgit2, http-parser), and reused the installed
`llvm` 23.1.4. LLVM and MLIR did not move, so `libJLCS.dll` needs no rebuild.

Builds (dev profile with `debug = 2`, `codegen-units = 1`,
`incremental = false`; each took about 0.5 s):

- `cargo rustc --lib --crate-type cdylib -- -C debuginfo=2 --emit=llvm-ir,link`,
  and the same with `dylib`;
- `--release` with `-C debuginfo=2 -C codegen-units=1`, and opt-level 1;
- `rbprobe_rb` with `RUSTFLAGS="--emit=llvm-ir,link" cargo build`, which writes
  one `.ll` per crate from one build.

Calls were made from Julia 1.13 with `ccall`. For first-class aggregate
returns they went through `Base.llvmcall` by function pointer with exactly the
define line's types. That was a measuring device only: Tier 1 stays
quarantined, and the product path is a JLCS thunk (§4.2).

---

## 2. What the deleted generator did

`compile_rust_project` (deleted in `7dfe3ee`) ran `rustc --crate-type cdylib
-g --emit=link` on the first `.rs` file and read its DWARF the way the C path
does. `GeneratorRust.jl` emitted a `ccall` per function by `DW_AT_name`, and
`infer_rust_type` mapped Rust type names by string. A list of 40 names and
the patterns `::`, `<` and `{closure` hid std types. Its fixture,
`test/rust_test/src/lib.rs`, was all `#[no_mangle] extern "C"` and
`#[repr(C)]`, and its `verify.jl` was hand-written `ccall`s.

A cdylib exports only `#[no_mangle] extern "C"` items. In this crate that is 5
of 5, and each of the 46 Rust-ABI functions is either `internal` or not
compiled at all (§4.3). So the generator
could only see an API the author had already written as C. It read no IR, and
it handled no enums with data, generics, `Drop` or panics. CHANGELOG's entry
calls it "effectively a C-ABI wrapper with extra steps".

The new design differs in two places. §4 calls any `pub fn` with rustc's own
lowering. §6 handles generics, ownership and panics without touching the crate.

---

## 3. DWARF

### 3.1 What rustc emits

- DWARF 4, `DW_AT_language 28 (Rust)`, producer `clang LLVM (rustc version
  1.99.1 …)`. There is one compile unit per crate. MSYS2's std rlibs carry no
  DWARF: the dylib's only CU is rbprobe's, and std appears only as code
  instantiated or inlined inside it.
- The dylib's dump has 23,185 lines. Its tag counts: 1,340
  `formal_parameter`, 686 `template_type_param`, 658 `subprogram`, 411
  `inlined_subroutine`, 399 `variable`, 308 `member`, 262 `lexical_block`, 206
  `namespace`, 203 `structure_type`, 101 `pointer_type`, 81 `enumerator`, 59
  `variant`, 29 `variant_part`, 15 `base_type`, 11 `array_type`, 5
  `enumeration_type`, 3 `subroutine_type`.
- There is no `typedef`, `const_type`, `reference_type`, `union_type` or
  `class_type`.

### 3.2 The layouts are exact

`Particle` is declared `tag: u8, mass: f64, id: u32, flag: u8, pos: [f32; 3]`.
DWARF puts `mass@0, id@8, pos@12, tag@24, flag@25` with `byte_size 32` and
`DW_AT_alignment 8`: rustc reordered the fields, and DWARF records the
result. Every record and every member carries `DW_AT_alignment`. `String`
nests as `vec@0` → `Vec{buf@0, len@16}` → `RawVec{inner@0}` →
`RawVecInner{cap@0, ptr@8}`, so a `String` is cap, ptr, len, with ptr and
cap swapped against the declaration order.

Both layouts were checked by calling the library:

- `Particle::new(41, 2.0)` read back `id@8 = 41` and `mass@0 = 2.0`;
- `make_greeting("julia")`, decoded at those three offsets, read `"hello julia"`
  (§4.2).

### 3.3 New shapes

| Shape | Example | What the DWARF says |
|---|---|---|
| Tagged enum | `Shape`, 24 bytes | `structure_type` → `variant_part` with `DW_AT_discr` naming an artificial unnamed member (`u64` at 0) → three `variant`s with `discr_value` 0, 1, 2. Each holds a member named after the variant, typed by a nested `structure_type` of the enum's whole size whose fields sit after the tag (`Circle.r@8`, `Rect.w@8, h@16`). |
| Niche enum | `Slot`, 8 bytes (`Empty \| Full(Box<Particle>)`) | The discriminant is the Box pointer (`u64` at 0). `Empty` has `discr_value 0`; `Full` has none, which makes it the default variant (every other value). |
| Niche in a `bool` | `Mode`, 1 byte (`On(bool) \| Off \| Auto`) | The discriminant is the `bool` byte; `Off` is 2, `Auto` 3, and `On` is the default. |
| `Option<i32>` | 8 bytes | Tag `u32` at 0 (`None` 0, `Some` 1), payload at 4. |
| Fieldless enum | `Level` | `enumeration_type` with `DW_AT_enum_class`, underlying `u8`. |
| Fat pointer | `&str`, `&[f64]`, `&dyn Area` | `structure_type` named `&str`: `data_ptr@0, length@8`. `&dyn rbprobe::Area`: `pointer@0, vtable@8` (typed `&[usize; 5]`). `Box<dyn Area>` is a structure too. |
| Thin pointer | `&Particle`, `&mut String`, `Box<Particle>`, `*const ()` | `pointer_type` whose `DW_AT_name` is the Rust spelling, e.g. `alloc::boxed::Box<rbprobe::Particle, alloc::alloc::Global>`. |
| Vtable | `<Particle as Area>` | A `DW_TAG_variable` named `<rbprobe::Particle as rbprobe::Area>::{vtable}` with `DW_OP_addr`. Its type, `…::{vtable_type}` (40 bytes), is `drop_in_place@0, size@8, align@16, __method3@24, __method4@32`. Slots are positional; the trait has no DIE. |
| Names | | Bare names inside `DW_TAG_namespace` (206 of them, including `{impl#N}` for impl blocks). Instantiations are spelled in the name (`Pair<i32>`, `Vec<i64, alloc::alloc::Global>`) with `template_type_param` children. Tuples are `(i32, f64)` with members `__0`, `__1`; closures are `{closure_env#0}`. |
| Methods | `Particle::new`, `bump`, `kinetic` | Declaration subprograms nested in the struct DIE, with linkage names. `self` is an ordinary first formal parameter. |
| Zero-sized type | `Z` | `byte_size 0`. A ZST parameter is a formal parameter in DWARF and absent from the IR (§4.1). |
| Repeated names | | 13 `Some`, 13 `None`, 7 `Ok`, 7 `Err`, 2 `Empty` structure DIEs (one per enum or instantiation that has the variant), and 7 `{closure_env#0}`. |

### 3.4 What `parse_dwarf_dump` makes of it

The probe ran `Compiler.parse_dwarf_dump` over the GNU objdump dump, with
`_dwarf_file_tables`. It found 416 functions, 139 `struct_defs` and 60
`record_abi` entries. `check_param_arity!` warned of 23 under-counts, all on
std functions inlined into the crate.

| | Result |
|---|---|
| Plain records, tuples, fat pointers | Exact: `Particle` (all five reordered offsets), `Pair<i32>`, `Two16`, `Three32`, `F2`, `CPoint`, `Holder`, `(i32, f64)`, `&str`, `&[f64]`, `&dyn rbprobe::Area`, `String`, `Vec<i64, …>` |
| Fieldless enum | Exact: `__enum__Level` with `Low = 0, Mid = 1, High = 2` |
| Enums with data | Flattened. `Shape` → `[Circle@0, Rect@0, Empty@0]`, `Slot` → `[Empty@0, Full@0]`, `Mode` → `[On@0, Off@0, Auto@0]`. The discriminant member, its offset and every `discr_value` are gone. |
| Variant structs | Keyed by bare name at top level. `Some` keeps one definition marked `conflict` with sizes `[0, 4, 8, 16, 24]`; `None` and `Empty` are absent. |
| Pointers | `&Particle` and `Box<Particle>` both become `Particle*`, so `unbox_id(b: Box<Particle>)` and `particle_mass(p: &Particle)` get the same parameter type. |
| Functions | All of the crate's functions, with one DWARF parameter per Rust parameter (`str_len(s::&str)`, `dyn_area(a::&dyn rbprobe::Area)`). `class` is empty for every method, because `_qualified_name_parts` reads Itanium names. |
| `record_abi` | `pass` is `nothing` for every record (Rust emits no `DW_AT_calling_convention`); `empty` is true for ZSTs. |

### 3.5 What it would need

- **Variant parsing**, giving an enum layout fact: the discriminant's offset,
  size and type, each variant's `discr_value` or "default", and each variant's
  payload type. Without it a data enum can only cross as opaque bytes.
- **Scoped type keys.** An enum scopes its variants (`Option<i32>::Some`), and
  the namespace path scopes everything else. CLAUDE.md lists "distinct types
  sharing a bare name" as an open problem; in Rust it is the default case.
- **Pointer kinds**, read from the pointer DIE's name: `&`, `&mut`, `*const`,
  `*mut`, `Box<…>`, `NonNull<…>`. Ownership (§6.2) depends on this.
- **Rust base names**: `u8`…`u128`, `i8`…`i128`, `f32`, `f64`, `bool`, `char`,
  `usize`, `isize` and `()`, not C spellings. A `char` is a 4-byte Unicode
  scalar value, so it is `UInt32` in Julia, not `Char`.
- **Methods scoped by DWARF nesting**, not by demangling. Trait-impl methods
  sit in `{impl#N}` namespaces, and only their v0 linkage names say which
  trait they implement.
- **Vtable variables** (address and slot types), and only if Julia is to
  dispatch on `dyn` itself (§6.4).

The recommendation is a separate pass over the same GNU-binutils text,
selected by `DW_AT_language 28`, that fills the same metadata shapes plus the
new facts. It should not be more branches inside `parse_dwarf_dump`. That
function's C++ heuristics (the Itanium `class`, the receiver gates,
bare-name canonicalisation) are the parts that are wrong for Rust.

---

## 4. Calls

### 4.1 The lowered signatures

These are the IR `define` lines of the dylib at opt-level 0 (names
demangled, debug metadata dropped). The release build adds attributes
(`noundef`, `noalias`, `nonnull`, `dereferenceable(N)`, `captures(…)`,
`dead_on_return`, `range(…)`). For the three exported functions compared
(`string_len_owned`, `unbox_id`, `make_area`), it does not change a type.

| Rust | `define` |
|---|---|
| `str_len(s: &str) -> usize` | `i64 (ptr %s.0, i64 %s.1)` |
| `slice_sum(v: &[f64]) -> f64` | `double (ptr align 8 %v.0, i64 %v.1)` |
| `index_of(v: &[i32], i: usize) -> i32` | `i32 (ptr align 4 %v.0, i64 %v.1, i64 %i)` |
| `dyn_area(a: &dyn Area) -> f64` | `double (ptr %a.0, ptr align 8 %a.1)` |
| `make_area(k: u32) -> Box<dyn Area>` | `{ ptr, ptr } (i32 %k)` |
| `two16_swap(t: Two16) -> Two16` (two `u16`) | `{ i16, i16 } (i16 %t.0, i16 %t.1)` |
| `f2_make(x: f64, y: f64) -> F2` (two `f64`) | `{ double, double } (double %x, double %y)` |
| `f2_dot(a: F2, b: F2) -> f64` | `double (double %a.0, double %a.1, double %b.0, double %b.1)` |
| `tuple_ret(a: i32) -> (i32, f64)` | `{ i32, double } (i32 %a)` |
| `checked_div(a: i32, b: i32) -> Option<i32>` | `{ i32, i32 } (i32 %a, i32 %b)` |
| `three32_sum(t: Three32) -> u32` (12 bytes) | `i32 (ptr align 4 %t)` |
| `three32_make(a: u32) -> Three32` | `void (ptr sret([12 x i8]) align 4 %_0, i32 %a)` |
| `particle_mass(p: &Particle) -> f64` | `double (ptr align 8 %p)` |
| `particle_by_value(p: Particle) -> Particle` | `void (ptr sret([32 x i8]) align 8 %_0, ptr align 8 %p)` |
| `make_greeting(name: &str) -> String` | `void (ptr sret([24 x i8]) align 8 %_0, ptr %0, i64 %1)` |
| `string_len_owned(s: String) -> usize` | `i64 (ptr align 8 %s)` |
| `parse_num(s: &str) -> Result<i64, String>` | `void (ptr sret([24 x i8]) align 8 %_0, ptr %s.0, i64 %s.1)` |
| `boxed_particle(id: u32) -> Box<Particle>` | `align 8 ptr (i32 %id)` |
| `slot_make(id: u32) -> Slot` | `align 8 ptr (i32 %id)` |
| `opt_ref_id(p: Option<&Particle>) -> u32` | `i32 (ptr align 8 %p)` |
| `mode_code(m: Mode) -> i32` | `i32 (i8 %0)` |
| `level_next(l: Level) -> Level` | `i8 (i8 %l)` |
| `char_code(c: char) -> u32` | `i32 (i32 %c)` |
| `wide(x: u128) -> u128` | `i128 (i128 %x)` |
| `Particle::kinetic(&self, v: f64) -> f64` | `double (ptr align 8 %self, double %v)` |
| `<Shape as Area>::label(&self) -> String` | `void (ptr sret([24 x i8]) align 8 %_0, ptr align 8 %self)` |
| `largest::<f64>(xs: &[f64]) -> f64` | `double (ptr align 8 %xs.0, i64 %xs.1)` |
| *second crate, `small` (Appendix A.2)* | |
| `b3(x: B3) -> B3` (three `u8`) | `i24 (i24 %0)` |
| `i3(x: I3) -> I3` (three `u16`) | `i48 (i48 %0)` |
| `l2(x: L2) -> L2` (`u64`, `u32`) | `{ i64, i32 } (i64 %x.0, i32 %x.1)` |
| `e1(x: E1) -> E1` (three `f32`) | `void (ptr sret([12 x i8]) align 4 %_0, ptr align 4 %x)` |
| `z(_z: Z, a: i32) -> Z` (`Z` is zero-sized) | `void (i32 %a)` |
| `flag(b: bool) -> bool` | `zeroext i1 (i1 zeroext %b)` |
| `opt_box(b: Option<Box<u32>>) -> Option<Box<u32>>` | `align 4 ptr (ptr align 4 %b)` |

The rules that show (rustc's Rust-ABI adjustment, not the C ABI):

- A two-scalar value (a fat pointer, a pair struct, a tuple, `Option<i32>`) is
  two IR arguments, and returns as a first-class `{ a, b }`.
- Any other aggregate up to 8 bytes is one integer of its size (`i24`, `i48`).
- Anything larger goes by plain `ptr` to a copy the caller made, never `byval`,
  and returns through `sret([N x i8])`.
- A zero-sized value is nothing.
- `bool` is `zeroext i1`.
- A function that may unwind carries `personality ptr @rust_eh_personality`.

On the same target, Win64's C ABI passes a 16-byte record such as `F2` by
reference and returns it through a hidden pointer. The Rust ABI passes two
doubles and returns them in two registers.

### 4.2 Calling from the define line alone

| Probe | How | Result |
|---|---|---|
| `must_positive(5)` | `ccall`, `i32 (i32)` | 5 |
| `str_len("héllo")` | `ccall`, `(Ptr{UInt8}, Int64)` | 6 |
| `slice_sum([1, 2.5, 4])` | `ccall`, `(Ptr{Float64}, Int64)` | 7.5 |
| `f2_dot((1,2), (3,4))` | `ccall`, four `Float64` | 11.0 |
| `f2_make(1.5, 2.5)` | IR signature, `{ double, double }` | (1.5, 2.5) |
| the same, as a C struct return | `ccall` returning `F2` | **wrong**: (1.29e-311, 0.0) |
| `two16_swap(1, 2)` | IR signature, `{ i16, i16 }` | (2, 1) |
| the same, as a C struct return | `ccall` returning `Two16` | **wrong**: (2, 0) |
| `checked_div(7, 2)`, `checked_div(1, 0)` | IR signature, `{ i32, i32 }` | (1, 3); tag 0 |
| `make_area(0)` then `dyn_area` | IR signature `{ ptr, ptr }`, then `ccall (ptr, ptr)` | 6.0; the vtable's size/align read 24/8 |
| `Particle::new`, `particle_by_value`, `kinetic` | `ccall` with `sret` buffer and `ptr` | right values at DWARF offsets |
| `make_greeting("julia")` | `ccall` with `sret` buffer | `"hello julia"`, then freed by the binary's `drop_glue::<String>` |
| `wide(2^100 + 7)` | IR signature, `i128 (i128)` | right |
| the same | `ccall` with `UInt128` | **wrong**: Julia's `ccall` lowers `UInt128` differently from LLVM's `i128` on this target |

Conclusions:

- **The define line alone gives a correct call.** All 12 calls made with
  exactly the define line's types returned the right value, with nothing else
  known about the function.
- **The define line alone does not give a wrapper.**
  - At opt-level 0, `particle_mass(p: &Particle)` and
    `particle_by_value(p: Particle)` both take `ptr align 8`. One borrows; the
    other receives a copy that the callee owns and drops. `string_len_owned`
    frees the `String` it is given.
  - Value names do not settle it either. `%s.0`, `%s.1` mark a split
    parameter when the parameter is a plain binding, but `make_greeting`'s
    `name` is `%0, %1` and `must_positive`'s `x` is `%0`.

  DWARF says what each parameter is; the IR says how it is passed.
- **The join** walks DWARF's formal parameters in order against the IR's
  parameters after any `sret`. Each DWARF parameter takes:
  - 0 IR parameters, if its `byte_size` is 0;
  - 2, if its DWARF shape is two scalars and the next two IR types match them;
  - otherwise 1: a scalar, `iN` of its size, or `ptr` for an indirect
    aggregate.

  A function whose IR parameters are not all consumed exactly is refused. That
  is the rule `_c_byval_facts` already applies to `byval`: counts and names
  must agree, else no fact.
- **Tier 3 can carry only part of this.** It works where every IR type is a
  scalar or a pointer, which covered most probes. First-class aggregate returns
  and `i128` need a thunk.
- **JLCS cannot issue these calls today.** `buildCallShape` classifies every
  `!llvm.struct` operand and result as a C record (by the SysV statement, or
  by `classifyWin64Struct`). On Win64 that returns `F2` through a hidden
  pointer, which is the C ABI and the measured wrong answer. A Rust call needs
  a form that passes the define line's types through verbatim, with their
  ABI attributes (`sret(T)` and its `align`, `zeroext` and `signext`), and no
  classification. That is the one dialect change Stage 2 needs.

### 4.3 What each crate type exports

| Build | Exports | Crate items exported | Notes |
|---|---|---|---|
| `cdylib`, opt-level 0 | 5 | 0 | Exactly the five `#[no_mangle] extern "C"` names. Its IR has 12 defines: the 7 Rust ones are `internal`, and only functions reachable from the C surface were compiled at all. |
| `dylib`, opt-level 0 | 1,919 | all 49 | 1,912 v0 symbols, plus the 5 `extern "C"` names, `rust_eh_personality` and the metadata symbol. The 49 crate items are 46 non-generic functions (38 free functions, 3 inherent methods, 4 trait-impl methods, `Drop::drop`), the static `DROPS`, and the 2 instantiations the crate made (`largest::<f64>`, `Pair::<i32>::sum`). std is linked statically and exported too, as is the drop glue the crate instantiated. |
| `dylib`, opt-level 1 | — | 28 functions | |
| `dylib`, opt-level 3 | 1,806 | 18 functions | Small non-generic `pub fn`s are cross-crate-inlinable and get neither a symbol nor a define (`str_len`, `f2_dot`, `wide`, `Particle::new`, …). Without share-generics, `largest::<f64>` is gone. |
| opt-level 3, `-C link-dead-code=yes` | — | 18 functions | No effect. |
| opt-level 3, `-Z cross-crate-inline-threshold=never` | — | all 46 | An unstable flag, measured under `RUSTC_BOOTSTRAP=1` only. It confirms the cause; it is not usable. |

Mangling:

- **v0 only.** `-C symbol-mangling-version=legacy` fails on 1.99.1 ("requires
  `-Z unstable-options`"). Every Rust symbol is `_R…`. The Itanium machinery
  (`_qualified_name_parts`, `_is_itanium_thunk`) must never be pointed at
  these names. CLANG64's `llvm-cxxfilt` demangles v0.
- **The crate disambiguator depends on the build.** rbprobe is
  `CsUmzxmMh9ni_7rbprobe` when built as the root dylib and
  `Cs2NyCfjW8iWo_7rbprobe` when built as rbprobe_rb's dependency. Names must
  be read from the build RepliBuild ran, never written down.
- **An instantiation carries the crate that made it as a suffix.**
  `core::ptr::drop_glue::<alloc::string::String>`, instantiated in rbprobe, is
  `_RINvNtCs5EhuIZY1uw6_4core3ptr9drop_glue…ECsUmzxmMh9ni_7rbprobe`. The
  surface filter keys on the path root (the crate the item belongs to), not on
  the crate's name appearing anywhere in the symbol.
- Rust 1.99 names drop glue `core::ptr::drop_glue::<T>`. The vtable slot is
  still called `drop_in_place`.

---

## 5. LLVM

- **MSYS2's rustc uses the system LLVM.** `rustc.exe` loads `libLLVM-23.dll`
  from `clang64/bin` (per `ldd`), the same DLL that clang 23.1.4 and
  libJLCS use. `rustc -vV` reports 23.1.3, the version it was built against.
  On the reference host, rustc's IR and libJLCS's thunk modules therefore come
  from one libLLVM.
- **rustup's toolchains carry rustc's own LLVM** inside `rustc_driver`. This
  was not measured here. On Linux, where rustup is the usual source and CI's
  LLVM is apt.llvm.org 22 or 23, rustc's LLVM will usually differ from the
  system's.
- **Julia's LLVM cannot read this IR.** Julia 1.13.3's libLLVM is 20.1.8. It
  parsed `captures(none)` and `range(…)` from rustc's release define lines and
  rejected `dead_on_return` (`error: expected ')' at end of argument list`).
  rustc wrote that attribute on `string_len_owned`'s by-value `String`
  parameter at opt-level 3. So rustc's IR cannot be loaded into
  Julia's LLVM, and the same holds for any LLVM older than rustc's.
- **The target matches clang's.** The IR's triple is `x86_64-pc-windows-gnu`
  (host `x86_64-pc-windows-gnullvm`), and its datalayout is byte-identical to
  clang's for `x86_64-w64-windows-gnu`. The target features differ (rustc's
  baseline adds `cx16`, `sse3`, `sahf`), which no call depends on.

The plan:

- **cargo builds and rustc links.** RepliBuild's own link step (in-process
  libLLVM for C, clang for C++) never sees a Rust object or Rust IR.
- **RepliBuild reads only the define lines, as text**, the way
  `_nounwind_definitions` and `_byval_definitions` already read clang's
  (`_ir_split_params` splits the parameter list). With more than one codegen
  unit, rustc writes one `.ll` per unit (16 files at `codegen-units=16`). The
  reader takes them all, as those functions already take a list.
- **The thunk module is JLCS's own.** It is built from transcribed signatures
  and lowered by the system LLVM. "One LLVM per module" holds because no LLVM
  reads another's IR, so `_assert_debug_info_kept` has nothing to check here.
- **One risk remains: two LLVMs lowering the same IR type**, rustc's for the
  definition and the system's for the thunk's call. On MSYS2 they are one
  DLL. Where they differ (rustup), record rustc's LLVM major in the toolchain
  identity, and refuse types whose lowering changed between majors. `i128` is
  the measured example of an IR type with two lowerings on this target
  (`ccall`'s and LLVM's).
- **The toolchain identity for the project hash** is `rustc -vV` (commit hash
  and LLVM version), `cargo -V`, `Cargo.lock` and the target triple.

---

## 6. What must be declared

### 6.1 Generic instantiations: a generated sibling crate

A dylib holds only the instantiations its own code made. `largest::<f64>`
(used by `largest_f64`) and `Pair::<i32>::sum` were exported at opt-level 0
only because share-generics is on in the dev profile; at opt-level 3 neither
was. `largest::<i32>` exists nowhere, and DWARF has nothing for a generic
nobody instantiated.

The prototype `rbprobe_rb` (Appendix B) is a `dylib` crate that depends on
rbprobe as an rlib. It has one non-generic function per instantiation:
`__rb_largest_i32` calls `rbprobe::largest::<i32>`, and `__rb_pair_f64_sum`
calls `Pair::<f64>::sum`. One `cargo build` built both crates, and both
functions answered (9 and 3.75). **The sibling crate is the root of the
artifact RepliBuild wraps.** `rbprobe_rb.dll` exports 49 symbols rooted in
rbprobe beside the shim's 8, so the process has one copy of the crate and of
its statics.

As with `[types] templates`, the set is declared. The generator cannot spell
an instantiation's signature: RepliBuild reads no Rust source, and a generic
that was never instantiated has no DWARF. So the declaration carries the
signature, and rustc type-checks it when it builds the generated crate. A
wrong one is a compile error, never a silent wrong call. The forwarders get
`#[inline(never)]`, so they keep a symbol at any opt-level (§4.3).

### 6.2 Ownership and drop

- **Drop glue exists only for types the crate drops itself.** The dylib has
  `drop_glue::<String>`, `<Vec<i64>>`, `<Vec<u8>>` and `<Box<Particle>>`. It
  has none for `Holder`, which the crate builds and returns but never drops,
  and none for `Box<dyn Area>`.
- **Calling `Drop::drop` is not dropping.** `<Holder as Drop>::drop` is
  exported. Calling it ran the user's `drop` (`DROPS` went 0 → 1) and freed
  neither field.
- **`drop_glue::<String>` from the binary freed** the `String` that
  `make_greeting` returned.
- **The generated crate's drop function freed `Holder`.**
  `__rb_drop_holder(p: *mut Holder) { ptr::drop_in_place(p) }` ran the user's
  `drop` and freed both fields. `__rb_drop_box_dyn_area` compiled, and pulled
  `drop_glue::<Box<dyn Area>>` into the binary, but was not called.

So drop glue needs no TOML. The generator emits a `drop_in_place` function for
every type that crosses by value out of Rust, and it knows those types from
DWARF.

On the Julia side:

- An owned value lives in an owning wrapper whose finalizer calls that function
  (the `Managed<X>` pattern).
- A by-value Rust parameter is a move: the callee drops it, as
  `string_len_owned` does, so the wrapper is spent after the call.
- `&T` and `&mut T` are borrows.

The pointer DIE's name (§3.5) is what tells these apart. Rust memory is freed
only by Rust drop glue, never by `Libc.free`; the two-runtime rule for C on
Windows already says the same.

A TOML key is needed only where DWARF's name is not a path the generated
crate can write. Two cases:

- a type defined in a private module and re-exported with `pub use`;
- the default allocator parameter that DWARF spells out, as in
  `Box<dyn rbprobe::Area, alloc::alloc::Global>`. Naming `alloc::alloc::Global`
  is unstable.

rustc rejects a path it cannot name when it builds the generated crate, so
both fail loudly.

### 6.3 Panics: `catch_unwind` in one generated Rust frame

| Case | Result |
|---|---|
| A Rust-ABI function panics, called by `ccall` | The panic message prints, then the process exits `0xC00000FF` (STATUS_BAD_FUNCTION_TABLE): unwinding fails at the first Julia frame. |
| An `extern "C-unwind"` function panics | The same, `0xC00000FF`. |
| The same inside a Julia `try` | The same. |
| An `extern "C"` function panics | Rust aborts: "panic in a function that cannot unwind". |
| A C++ `catch (...)` around the panic (a clang++ DLL shaped like `jlcs_guard_*`) | The catch runs (`cpp_guard: caught`). At the end of the catch block: "fatal runtime error: Rust panics must be rethrown, aborting". The C++ handler sees the panic although `libc++.dll` does not import the `libunwind.dll` the Rust DLL uses; both unwinders presumably run on Windows SEH (inferred). Rust aborts when foreign code destroys its exception. |
| Generated `__rb_try_must_positive` (`catch_unwind` around the call) | Two panics caught in a row, with the message recovered (`"not positive: -1"`), then a normal call returned 9. The process survives. |
| Generated `__rb_catch(thunk, env)` around a clang-compiled C frame that calls the Rust function (a stand-in for an AOT thunk) | Panics at -1 and -2 caught, 6 returned. The process survives. |

So neither existing mechanism works. `C-unwind` delivers the panic to Julia,
which cannot take it. Tier 2's C++ EH may catch a Rust panic but may not end
it.

The last row is the design: Julia → `__rb_catch` (Rust, generated once,
`catch_unwind`) → JLCS thunk → Rust function. It is the same Windows
structure the C++ path already has, where `jlcs_guard_*` is the catching
frame outside the thunk. Here the catching frame is written in Rust, because
only Rust may end a Rust panic. One trampoline serves every thunk, and no
Rust source is generated per function.

**Routing.** rustc stamps `nounwind` where it can prove a function cannot
unwind. At opt-level 3 it did so for `slice_sum` and `drops`, but not for
`must_positive`, `dyn_area` or `make_area`. At opt-level 0 it did so for no
crate function at all. A function whose define has `nounwind` can be called
without the trampoline, as `_nounwind_definitions` already decides for C++.
`-C panic=abort` would turn every panic into a process abort, so RepliBuild
builds with the default `panic=unwind`.

**Not measured:** a JIT'd thunk frame in the middle (the C stand-in has an
AOT thunk's shape). JLCS registers JIT `.pdata` with `RtlAddFunctionTable`,
and C++ exceptions already unwind through JLCS thunks on Windows. Stage 3 must
measure the Rust case.

### 6.4 Trait objects

`Box<dyn Area>` and `&dyn Area` cross as `{ ptr, ptr }` and `(ptr, ptr)`
(data, vtable). A `dyn` value made by one function was passed to another
correctly. To call a trait method on a `dyn` from Julia, there are two ways:

- **A generated forwarder**, `fn __rb_area(a: &dyn Area) -> f64 { a.area() }`,
  where rustc does the dispatch.
- **The slot index.** DWARF names slots only by position (`__method3`), so the
  slot-to-method map would come from reading a vtable's relocated entries and
  matching them to the addresses of the exported impl methods.

Forwarders are simpler, and are declared the way instantiations are.

### 6.5 Proposed TOML

```toml
[project]
name = "rbprobe"

[rust]
manifest = "Cargo.toml"     # the crate to wrap; cargo resolves its dependencies
package = "rbprobe"
opt_level = 0               # 0: every non-generic pub fn keeps a symbol (§4.3)
features = []
catch_panics = true         # every call without `nounwind` goes through __rb_catch

[[rust.instantiate]]        # rustc type-checks `sig` when it builds the generated crate
path = "rbprobe::largest::<i32>"
sig = "fn(&[i32]) -> i32"

[[rust.instantiate]]
path = "rbprobe::Pair::<f64>::sum"
sig = "fn(&rbprobe::Pair<f64>) -> f64"

[rust.paths]                # a public path for a type DWARF names where it is defined
"rbprobe::detail::Holder" = "rbprobe::Holder"
```

Each new key goes into `docs/src/config.md` and `Discovery.PRESERVED_TOML_KEYS`,
and is consumed or accounted for in `test_config_surface.jl`.

---

## 7. Staging

### Stage 1: the `extern "C"` surface through the C path

- **Build:** cargo, `crate-type = cdylib`, `-C debuginfo=2`, in RepliBuild's
  own target directory.
- **Surface:** the PE export directory (5 of 5 here).
- **Types:** DWARF through the existing parser, which reads `#[repr(C)]`
  records exactly (`CPoint`), plus a table of Rust base names for GeneratorC,
  since DWARF spells `f64`, `i32`, `usize`.
- **Calls:** Tier 3, unchanged.
- **Panics:** a panic in an `extern "C"` function aborts inside Rust
  (measured). An `extern "C-unwind"` function must be refused, because its
  panic kills the process.

What it proves: the cargo driver, artifact discovery, the toolchain identity
and project hash, and that the DWARF path holds for Rust. It has the deleted
generator's scope, done through the existing C path. For a crate whose author
wrote no C API it adds nothing. That is why it comes first, and also why it is
not the point.

### Stage 2: Rust-ABI calls through Tier 2, from define lines

- **Build:** `crate-type = dylib`, with the wrapped crate at opt-level 0.
  Cargo's per-package profiles can keep its dependencies optimized (not
  measured).
- **Surface:** exports whose v0 path root is the crate.
- **Signatures:** a `_rust_abi_definitions` reader for the define lines,
  joined to the DWARF parameters by the 0/1/2 rule, with refusal on any
  mismatch (§4.2).
- **Types:** the Rust DWARF pass (§3.5). Enums with data become byte blobs
  with a discriminant reader; keys are scoped; pointer kinds are kept.
- **Calls:** the JLCS call form that keeps the define line's types (§4.2). A
  function whose IR types are all scalars and pointers may stay on `ccall`.
- **Scope:** borrowed and plain-data values only. A function that moves an
  owned type (`Box`, `String`, `Vec`, anything with drop glue or a `Drop`
  impl) in either direction waits for Stage 3, because Stage 2 could not free
  it.

What it proves: the claim the whole design rests on, that an unmodified
crate's `pub fn`s are callable with rustc's own lowering. §4.2 already shows
it by hand for 12 calls. **This stage is what makes the front end more than
the deleted C-ABI wrapper.**

### Stage 3: the generated crate (generics, ownership, panics)

The generated sibling crate becomes the root of the artifact (a dylib), with
the wrapped crate as an rlib dependency. It holds:

- the declared instantiations;
- one `drop_in_place` function per owned type;
- `__rb_catch`;
- forwarders for `dyn` methods;
- `#[inline(never)]` forwarders when `opt_level > 0`.

On the Julia side: owning wrappers with finalizers, a move spends its
wrapper, and a panic comes back as a Julia exception carrying the panic
message.

What it proves: that the crate's whole API is usable, not only its
monomorphic, borrow-only subset, and that the process survives a panic.

### Each stage's test

- **Stage 1:** the probe crate's `cpoint_*` functions agree with hand-written
  `ccall`s (the role the deleted `verify.jl` had).
- **Stage 2:** the §4.2 call probe as a devtest, plus a negative check of the
  join's refusal.
- **Stage 3:** the shim probe, the catch probe, and a drop counter (`DROPS`)
  for every owned type, run in a child process (`run_isolated`), since the
  failure mode is a process abort.

---

## 8. Rules this would add

These are candidates for CLAUDE.md once the code exists.

- rustc's IR is read as text and never loaded into any LLVM.
- A Rust symbol is read from the build, never written down: the crate
  disambiguator changes with the build configuration.
- The Rust surface is filtered on the v0 path root, not on the crate's name
  appearing in a symbol.
- A join between DWARF and a define line that does not account for every IR
  parameter is a refusal.
- A Rust panic ends in a Rust `catch_unwind` frame. It never reaches a Julia
  frame or a C++ `catch`.
- Rust memory is freed by Rust drop glue only.
- A data enum's layout comes from its `DW_TAG_variant_part`, never from its
  variants' members.

---

## 9. Not measured

- **Linux.** This host has no WSL. Still to measure there:
  - rustup's rustc and its own LLVM;
  - the Rust-ABI shapes on SysV (the small-aggregate rules in §4.1 come from
    rustc's Rust-ABI adjustment and should hold; the define lines will say);
  - ELF exports through `nm -D`;
  - readelf's dump of the same DWARF.
- **rustup's toolchain on Windows** (`x86_64-pc-windows-gnu`, with its own LLVM
  and MinGW runtime).
- **A JIT'd thunk frame** between `__rb_catch` and the panicking function.
- **Larger builds and processes:**
  - crates with dependencies, build scripts, features or workspaces;
  - `-C prefer-dynamic` (std as a DLL);
  - two wrapped Rust crates in one process. Each dylib carries its own std, so
    that would mean two allocators and two panic runtimes (inferred).
- **Other language shapes:** DWARF 5 (`-C dwarf-version`), unions,
  `#[repr(transparent)]`, `#[repr(u8)]` enums with data, `impl Trait`
  returns, closures, `async`.
- **The cost of a thunk** against a `ccall`.

---

## Appendix A. The probe crates

### A.1 `rbprobe`

`Cargo.toml`: package `rbprobe`, edition 2024, and `[profile.dev] debug = 2,
codegen-units = 1, incremental = false`.

```rust
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Particle { pub tag: u8, pub mass: f64, pub id: u32, pub flag: u8, pub pos: [f32; 3] }
impl Particle {
    pub fn new(id: u32, mass: f64) -> Particle { Particle { tag: 7, mass, id, flag: 1, pos: [1.0, 2.0, 3.0] } }
    pub fn kinetic(&self, v: f64) -> f64 { 0.5 * self.mass * v * v }
    pub fn bump(&mut self) { self.id += 1; }
}
pub fn particle_mass(p: &Particle) -> f64 { p.mass }
pub fn particle_by_value(p: Particle) -> Particle { Particle { id: p.id + 100, ..p } }

pub struct Two16 { pub a: u16, pub b: u16 }
pub struct Three32 { pub a: u32, pub b: u32, pub c: u32 }
pub struct F2 { pub x: f64, pub y: f64 }
pub fn two16_swap(t: Two16) -> Two16 { Two16 { a: t.b, b: t.a } }
pub fn three32_sum(t: Three32) -> u32 { t.a + t.b + t.c }
pub fn three32_make(a: u32) -> Three32 { Three32 { a, b: a + 1, c: a + 2 } }
pub fn f2_make(x: f64, y: f64) -> F2 { F2 { x, y } }
pub fn f2_dot(a: F2, b: F2) -> f64 { a.x * b.x + a.y * b.y }
pub fn tuple_ret(a: i32) -> (i32, f64) { (a * 2, a as f64 / 2.0) }
pub fn wide(x: u128) -> u128 { x.wrapping_mul(3) }
pub fn char_code(c: char) -> u32 { c as u32 }

pub enum Level { Low, Mid, High }
pub fn level_next(l: Level) -> Level {
    match l { Level::Low => Level::Mid, Level::Mid => Level::High, Level::High => Level::Low }
}
pub enum Shape { Circle { r: f64 }, Rect { w: f64, h: f64 }, Empty }
pub fn make_shape(k: u32) -> Shape {
    match k { 0 => Shape::Circle { r: 1.5 }, 1 => Shape::Rect { w: 2.0, h: 3.0 }, _ => Shape::Empty }
}
pub fn shape_area_ref(s: &Shape) -> f64 {
    match s {
        Shape::Circle { r } => std::f64::consts::PI * r * r,
        Shape::Rect { w, h } => w * h,
        Shape::Empty => 0.0,
    }
}
pub fn shape_area_val(s: Shape) -> f64 { shape_area_ref(&s) }
pub enum Slot { Empty, Full(Box<Particle>) }
pub fn slot_id(s: &Slot) -> u32 { match s { Slot::Empty => 0, Slot::Full(p) => p.id } }
pub fn slot_make(id: u32) -> Slot {
    if id == 0 { Slot::Empty } else { Slot::Full(Box::new(Particle::new(id, 1.0))) }
}
pub enum Mode { On(bool), Off, Auto }
pub fn mode_code(m: Mode) -> i32 {
    match m { Mode::On(true) => 11, Mode::On(false) => 10, Mode::Off => 0, Mode::Auto => -1 }
}
pub fn checked_div(a: i32, b: i32) -> Option<i32> { if b == 0 { None } else { Some(a / b) } }
pub fn opt_ref_id(p: Option<&Particle>) -> u32 { p.map(|p| p.id).unwrap_or(0) }
pub fn parse_num(s: &str) -> Result<i64, String> { s.trim().parse::<i64>().map_err(|e| e.to_string()) }

pub fn str_len(s: &str) -> usize { s.len() }
pub fn make_greeting(name: &str) -> String { format!("hello {name}") }
pub fn string_len_owned(s: String) -> usize { s.len() }
pub fn sum_vec(v: Vec<i64>) -> i64 { v.iter().sum() }
pub fn make_vec(n: usize) -> Vec<i64> { (0..n as i64).collect() }
pub fn slice_sum(v: &[f64]) -> f64 { v.iter().sum() }
pub fn boxed_particle(id: u32) -> Box<Particle> { Box::new(Particle::new(id, 2.0)) }
pub fn unbox_id(b: Box<Particle>) -> u32 { b.id }

pub struct Holder { pub name: String, pub data: Vec<u8> }
pub static DROPS: AtomicU64 = AtomicU64::new(0);
impl Drop for Holder { fn drop(&mut self) { DROPS.fetch_add(1, Ordering::SeqCst); } }
pub fn holder_new(name: &str, n: usize) -> Holder { Holder { name: name.to_string(), data: vec![0u8; n] } }
pub fn holder_len(h: &Holder) -> usize { h.name.len() + h.data.len() }
pub fn drops() -> u64 { DROPS.load(Ordering::SeqCst) }

pub fn largest<T: PartialOrd + Copy>(xs: &[T]) -> T {
    let mut m = xs[0];
    for &x in xs { if x > m { m = x; } }
    m
}
pub struct Pair<T> { pub a: T, pub b: T }
impl<T: std::ops::Add<Output = T> + Copy> Pair<T> { pub fn sum(&self) -> T { self.a + self.b } }
pub fn largest_f64(xs: &[f64]) -> f64 { largest(xs) }
pub fn pair_i32_sum(p: &Pair<i32>) -> i32 { p.sum() }

pub trait Area { fn area(&self) -> f64; fn label(&self) -> String; }
impl Area for Shape { fn area(&self) -> f64 { shape_area_ref(self) } fn label(&self) -> String { "shape".to_string() } }
impl Area for Particle { fn area(&self) -> f64 { 0.0 } fn label(&self) -> String { format!("particle {}", self.id) } }
pub fn dyn_area(a: &dyn Area) -> f64 { a.area() }
pub fn make_area(k: u32) -> Box<dyn Area> {
    if k == 0 { Box::new(make_shape(1)) } else { Box::new(Particle::new(k, 3.0)) }
}
pub fn total_area(items: &[Box<dyn Area>]) -> f64 { items.iter().map(|a| a.area()).sum() }

pub fn must_positive(x: i32) -> i32 { if x <= 0 { panic!("not positive: {x}"); } x }
pub fn index_of(v: &[i32], i: usize) -> i32 { v[i] }

#[repr(C)]
pub struct CPoint { pub x: f64, pub y: f64 }
#[unsafe(no_mangle)]
pub extern "C" fn cpoint_norm(p: CPoint) -> f64 { (p.x * p.x + p.y * p.y).sqrt() }
#[unsafe(no_mangle)]
pub extern "C" fn cpoint_scale(p: *const CPoint, k: f64) -> CPoint {
    let p = unsafe { &*p };
    CPoint { x: p.x * k, y: p.y * k }
}
#[unsafe(no_mangle)]
pub extern "C" fn c_panics(x: i32) -> i32 { must_positive(x) }
#[unsafe(no_mangle)]
pub extern "C-unwind" fn c_unwind_panics(x: i32) -> i32 { must_positive(x) }
#[unsafe(no_mangle)]
pub extern "C" fn c_drops() -> u64 { drops() }
```

### A.2 `small` (the small-aggregate rules)

Built with `rustc --edition 2024 --crate-type dylib -C debuginfo=2 -C
codegen-units=1 --emit=llvm-ir,link`.

```rust
pub struct B3 { pub a: u8, pub b: u8, pub c: u8 }
pub struct I3 { pub a: u16, pub b: u16, pub c: u16 }
pub struct L2 { pub a: u64, pub b: u32 }
pub struct E1 { pub a: f32, pub b: f32, pub c: f32 }
pub struct Z;
pub fn b3(x: B3) -> B3 { x }
pub fn i3(x: I3) -> I3 { x }
pub fn l2(x: L2) -> L2 { x }
pub fn e1(x: E1) -> E1 { x }
pub fn z(_z: Z, a: i32) -> Z { let _ = a; Z }
pub fn flag(b: bool) -> bool { !b }
pub fn opt_box(b: Option<Box<u32>>) -> Option<Box<u32>> { b }
```

---

## Appendix B. The generated-crate prototype, `rbprobe_rb`

`Cargo.toml`: `crate-type = ["dylib"]`, `rbprobe = { path = "../rbprobe" }`,
and the same dev profile. Built with `RUSTFLAGS="--emit=llvm-ir,link" cargo
build`. Every item is a plain Rust-ABI function; nothing is `extern "C"`.

```rust
pub fn __rb_largest_i32(xs: &[i32]) -> i32 { rbprobe::largest::<i32>(xs) }
pub fn __rb_pair_f64_sum(p: &rbprobe::Pair<f64>) -> f64 { p.sum() }

pub unsafe fn __rb_drop_holder(p: *mut rbprobe::Holder) { unsafe { core::ptr::drop_in_place(p) } }
pub unsafe fn __rb_drop_box_dyn_area(p: *mut Box<dyn rbprobe::Area>) { unsafe { core::ptr::drop_in_place(p) } }

use std::cell::RefCell;
thread_local! { static LAST_PANIC: RefCell<String> = RefCell::new(String::new()); }
fn record(e: Box<dyn std::any::Any + Send>) {
    let msg = if let Some(s) = e.downcast_ref::<String>() { s.clone() }
              else if let Some(s) = e.downcast_ref::<&str>() { s.to_string() }
              else { "panic".to_string() };
    LAST_PANIC.with(|m| *m.borrow_mut() = msg);
}
pub fn __rb_try_must_positive(x: i32, out: &mut i32) -> bool {
    match std::panic::catch_unwind(|| rbprobe::must_positive(x)) {
        Ok(v) => { *out = v; true }
        Err(e) => { record(e); false }
    }
}
pub fn __rb_last_panic_len() -> usize { LAST_PANIC.with(|m| m.borrow().len()) }
pub fn __rb_last_panic_copy(buf: &mut [u8]) -> usize {
    LAST_PANIC.with(|m| { let b = m.borrow(); let n = b.len().min(buf.len());
                          buf[..n].copy_from_slice(&b.as_bytes()[..n]); n })
}

// One generic catching frame: Julia -> __rb_catch -> thunk -> Rust target.
pub fn __rb_catch(thunk: extern "C-unwind" fn(*mut u8), env: *mut u8) -> bool {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| thunk(env))) {
        Ok(()) => true,
        Err(e) => { record(e); false }
    }
}
```

The thunk stand-in for the last probe in §6.3 is a C file compiled with
`clang -O1 -shared`:

```c
typedef int (*target_t)(int);
struct env { target_t f; int x; int out; };
__declspec(dllexport) void thunk_like(struct env *e) { e->out = e->f(e->x); }
```

The C++ guard for the `catch (...)` probe is `extern "C" int cpp_guard(int
(*f)(int), int x) { try { return f(x); } catch (...) { return -999; } }`,
compiled with `clang++ -shared`.
