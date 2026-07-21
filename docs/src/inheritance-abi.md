# The inheritance ABI

A reasoning guide to how RepliBuild makes C++ inheritance — multiple inheritance
(MI) and virtual inheritance (VMI) — callable from Julia. This page teaches the
*mental model* so you can read a generated wrapper and know what it's doing. For
the dated, blow-by-blow record of how it was built, see the update notes:
[MI + vcall](../updates/2026-07-17-multiple-inheritance-and-vcall.md) and
[virtual inheritance](../updates/2026-07-17-virtual-inheritance.md).

## The one invariant

Everything below follows from a single rule:

!!! note "The invariant"
    Every generated `Class_method(this, …)` wrapper expects `this` to already
    point at the **`Class` subobject** — not at the start of whatever derived
    object happens to contain it.

Because the wrapper knows the *static type* at each call site, it can choose the
right offset to turn a derived pointer into the base pointer the method wants.
That conversion is the **upcast helper**. Dispatch itself stays dumb and
*class-local*; all the difficulty of inheritance is pushed into "given a
`Derived`, produce the correct base-subobject pointer."

So there is really only one question to answer, in three increasingly hard forms:

| Case | Where is the base? | Upcast |
|------|--------------------|--------|
| Single / primary base | offset 0 | none — a `Derived*` already *is* a `Base*` |
| Multiple inheritance | a fixed offset | add a compile-time constant |
| Virtual inheritance | depends on the most-derived type | read the offset from the object's vtable at runtime |

## 1. Single inheritance — nothing to do

The base sits at offset 0, so the derived pointer is bit-for-bit a valid base
pointer. No helper is emitted; you pass the handle straight through.

## 2. Multiple inheritance — add a constant

```text
Derived : Base1, Base2

 offset:   0                16                32
          ┌────────────────┬────────────────┬─────────────────────┐
          │ Base1 subobject│ Base2 subobject│ Derived's own data  │
          │ (vptr1, data)  │ (vptr2, data)  │                     │
          └────────────────┴────────────────┴─────────────────────┘
          ▲                ▲
          p (== Base1*)    Derived_as_Base2(p) = p + 16
```

The `+16` is not guessed — it is `DW_AT_data_member_location` on the
`DW_TAG_inheritance` edge in DWARF, captured at extraction time. The generated
helper is exactly:

```julia
function Derived_as_Base2(obj)::Ptr{Cvoid}
    p = obj isa Ptr ? Ptr{Cvoid}(obj) : Ptr{Cvoid}(obj.handle)
    return p + 16                       # static offset, known at compile time
end
```

`Base1` gets no helper (offset 0). Base *members* are handled the same way:
`flatten_struct_members` rebases each inherited member by its base's subobject
offset, so `derived.some_base2_field` reads from the right place.

## 3. Virtual inheritance — read the offset from the vtable

A diamond shares one copy of the virtual base. The catch: **the shared base's
offset is a property of the most-derived object, not of the class**. A standalone
`Left` puts `VBase` at +16; a `Left` embedded in a `Diamond` puts it at +32. A
`Left*` cannot know statically which world it is in — only the object's vtable
does.

```text
Diamond : Left, Right          (Left, Right : virtual VBase)

 offset:   0             16          28    32
          ┌─────────────┬───────────┬─────┬──────────────┐
          │ Left        │ Right     │  d  │ shared VBase │  ← one copy, shared
          │ (vptr_L)    │ (vptr_R)  │     │ (vptr_V)     │
          └─────────────┴───────────┴─────┴──────────────┘
          ▲
          p

                    Left's vtable (what vptr_L points into)
                    ┌────────────────────────────┐
      vptr_L − 24   │  vbase-offset = 32         │  ← the entry we read
                    ├────────────────────────────┤
      vptr_L  ────▶ │  (address point)           │
                    │  RTTI, &tag(), …           │
                    └────────────────────────────┘

  Left_as_VBase(p) = p + *(vptr_L − 24) = p + 32
```

The DWARF location on a virtual edge is an *expression*, not a constant
(`this + *(vptr − 24)`); the parser distills it into `vbase_vtable_offset = −24`.
The generated helper does the two-hop read — load the vptr, read the vbase-offset
entry below the address point, add it:

```julia
function Left_as_VBase(obj)::Ptr{Cvoid}
    p    = obj isa Ptr ? Ptr{Cvoid}(obj) : Ptr{Cvoid}(obj.handle)
    vptr = unsafe_load(Ptr{Ptr{Int64}}(p))   # load the vtable pointer
    return p + unsafe_load(vptr + (-24))      # read the dynamic vbase offset
end
```

!!! tip "The canary that proves it's dynamic"
    `Left_as_VBase` is *one generated function*, yet it resolves +16 on a
    standalone `Left` and +32 on a `Left*` that is really a `Diamond` — because
    it reads the answer out of each object's own vtable. Any static
    implementation gets one of the two wrong. This is the pin in
    `test/vi_test/verify.jl`.

When a virtual base is reached through a static chain (e.g.
`Diamond_as_VBase` = static +0 to `Left`, then dynamic), the helper composes:
apply the static prefix to `p` first, then do the vtable read. One helper text is
correct for every dynamic type.

## Virtual dispatch and overrides — why you never do slot math

You might expect dispatching a virtual method on a secondary or virtual base to
require walking vtables and adjusting `this`. It doesn't, because the C++ compiler
already did the hard part:

1. It **re-homes** an override into the derived class's *own primary* vtable — an
   override of `Base2::get_b` in `Derived` reports its slot in `Derived`'s vtable,
   not `Base2`'s. `Diamond::tag` re-homes the same way.
2. It plants **`this`-adjusting thunks** in the secondary vtables that fix the
   pointer back before entering the override.

So RepliBuild dispatches **class-local**: index the method's own slot, with a
class-relative `this` — `vtable_offset = 0, this_offset = 0`. The caller-side
upcast plus the compiler's thunks do the pointer adjustment; the `jlcs.vcall`
lowering never has to. This is why virtual inheritance needed *zero* changes to
the dialect or the thunk generator — the only new machinery was the dynamic read
in the Julia upcast helper.

Two deliberate exceptions:

- **Destructors stay direct calls.** `Managed` finalizers and scope-RAII need
  *exact-class* destruction, not dynamic dispatch. Polymorphic deletion through a
  base pointer goes through a small C++-side helper.
- **`vcall` gates on scalar/pointer signatures.** Virtual methods that return a
  struct by value or take a packed struct by value keep the direct-call path (the
  vcall lowering does no sret/packed coercion yet).

## Where each piece lives

Three seams turn "what the C++ ABI says" into "something Julia can call":

1. **Extraction** — `DWARFParser.jl` + `Compiler.jl`. Captures the static base
   offset (`DW_AT_data_member_location`) and parses the virtual edge's location
   expression into `vbase_vtable_offset`. `ClassInfo` carries parallel
   `base_offsets` / `virtual_bases` / `vbase_vtable_offsets` vectors; bases are
   sorted by subobject offset for determinism.
2. **Metadata** — `jlcs.type_info` carries two tables: `baseNames`/`baseOffsets`
   (static) and `vbaseNames`/`vbaseVtableOffsets` (dynamic). Virtual bases never
   appear in the static table.
3. **Wrapper** — `GeneratorCpp.jl`. `_collect_upcasts!`
   ([src/Wrapper/Cpp/GeneratorCpp.jl:1539](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/src/Wrapper/Cpp/GeneratorCpp.jl#L1539))
   walks the base DAG accumulating static offsets, then emits either the static
   helper ([:1591](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/src/Wrapper/Cpp/GeneratorCpp.jl#L1591))
   or the dynamic helper ([:1613](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/src/Wrapper/Cpp/GeneratorCpp.jl#L1613)).

## Reading and using a generated wrapper

The whole integration contract, at a glance:

- `X_as_Y(p) = p + N` → `Y` is a normal (MI) base at fixed offset `N`.
- `X_as_Y(p) = p + unsafe_load(vptr + N)` → `Y` is a **virtual** base; the offset
  is read from the vtable.
- **To call a `Y` method on an `X` object, pass `X_as_Y(obj)` as the handle.**
  That is the only rule you have to remember — the compiler pre-arranged
  everything else.
- **Members:** MI base members are flattened onto the derived accessor (rebased by
  offset). Virtual base members are deliberately *not* flattened — their offset is
  dynamic, so you reach them by upcasting (`X_as_VBase`) and using the base's own
  accessors.

## What is deliberately not built

So you don't chase these as bugs — they're unbuilt pieces, tracked in
[TODO.md](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/TODO.md):

- MEMORY-class (>16-byte, or misaligned) by-value struct **arguments** don't yet
  match native stack-copy passing — nothing exercises the path (non-trivial
  classes take the RAII pointer path first).
- `vcall` dispatch for struct-shaped virtual signatures (see the gate above).
- The struct ABI classifier is **x86-64 SysV only**; Win64 / AAPCS are not
  modeled.

## Where the executable truth lives

- `test/mi_test/` — compiler-laid-out two-base fixture (31/31): override through a
  secondary vtable, C++-handed base pointer, a pinned wrong-`this` canary.
- `test/vi_test/` — the diamond (33/33): the same-helper 16-vs-32 canary, single
  shared copy across three paths, override dispatch through the vbase vtable.
- Dialect level: MLIR templates §8b (MI `vcall`) and §8d (the diamond).
