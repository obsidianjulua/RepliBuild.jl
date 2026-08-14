# ABI Marshalling as Compiler IR — the JLCS Dialect

This is the architecture page. Every other page in this manual describes what
RepliBuild *does*; this one describes the mechanism that makes the hard half of
it possible, and argues for why it is built the way it is.

The short version: **a foreign call is a compilation problem, so RepliBuild
compiles it.** Where a wrapper generator would emit a C shim or interpret a
signature at runtime, RepliBuild emits a small program in a purpose-built MLIR
dialect (JLCS), lowers it to LLVM IR, and executes the result. The marshalling
code for a function is *derived from the same DWARF that describes its types*,
type-checked by MLIR, lowered by a pass that models the platform ABI explicitly,
and — because it is real code with real debug info — steppable in gdb.

---

## 1. The problem

A foreign function call is not one operation. It is a small pile of decisions
the compiler normally makes for you, invisibly, at the call site:

- Which arguments go in registers, which on the stack, and in what order.
- Whether a struct is passed as bytes in registers, as a hidden caller-owned
  stack copy, or by reference.
- Whether the return value comes back in `RAX`/`XMM0`, or through a hidden
  pointer the caller supplies (`sret`).
- Whether the C++ object's `this` pointer needs adjusting before the callee sees
  it, and whether the callee is even known statically (virtual dispatch).
- What happens to the Julia stack frame if the callee throws.

`ccall` makes exactly one of those decisions available to you — the argument
type tuple — and derives the rest from Julia's own view of the types. That view
is correct whenever Julia's idea of a type matches the C compiler's. The whole
category of cases RepliBuild exists for is the category where it does not:

| Case | Why `ccall` cannot express it |
|---|---|
| Packed / attribute-aligned structs | Julia lays the struct out its own way; the offsets diverge silently |
| MEMORY-class (>16 B) by-value struct arguments | SysV wants a caller-owned stack copy; Julia has no `byval` spelling |
| Large by-value struct returns | Needs an `sret` pointer with ABI-correct alignment for a layout Julia may not model |
| Unions crossing by value | Register class depends on *all* overlapping fields, not on the arm you picked |
| C++ virtual methods | The callee address is in the object's vtable, not in the symbol table |
| C++ exceptions | An escaping exception unwinds through a Julia frame that has no landing pad |
| Non-trivial by-value class parameters | Itanium requires a copy-constructed temporary, destructed after the call |

Two conventional answers exist, and both were rejected on purpose.

**Write a C shim.** Generate `extern "C"` glue, compile it, `ccall` that. This
works, and it is what most binding generators fall back to. What it costs is the
thing RepliBuild is built to avoid: the shim is *source text in a second
language*, so its correctness is only checkable by compiling it, its knowledge
of layout comes from headers rather than from the binary, and every
library-specific quirk grows a new hand-written shim that someone has to
maintain. The failure mode is a shim that compiles cleanly and marshals wrongly.

**Interpret the signature at runtime.** libffi builds the call frame from a
description, per call. That is a genuinely general answer, but it is an
interpreter in the hot path, it models the ABI in a closed C library you cannot
extend for C++ specifics (vtables, RAII scopes, landing pads), and there is
nothing to inspect afterwards — a wrong call is a wrong call, with no artifact
between the description and the crash.

## 2. The claim

> Marshalling should be **compiled code generated from the compiler's own record
> of the library**, expressed in an IR whose types can be checked and whose
> lowering can be read.

MLIR's dialect ecosystem is almost entirely about compute lowering — tensors,
loops, hardware targets. JLCS uses the same machinery for the interop boundary,
which as far as this project is aware nobody else does. Four properties fall out
of that choice, and they are the argument for it:

1. **One source of truth.** The struct offsets in `!jlcs.c_struct` and the
   offsets the wrapper generator uses for Julia field access are the same DWARF
   numbers, read once. A shim-based design has two derivations that can drift.
2. **Type-checked marshalling.** Ops carry verifiers, the type converter refuses
   what it cannot translate, and a malformed thunk fails at *parse* — before
   anything executes. Several ops gained verifiers specifically because a
   malformed hand-written body used to reach the lowering and segfault it.
3. **The ABI is a pass, not a convention.** `classifySysVStruct` is ~40 lines
   that state the x86-64 SysV rules explicitly. When the rule was wrong (see
   §10) the fix was one function, applied to every call site at once, with a
   fixture that pins it against a real `clang++`-compiled callee.
4. **It is inspectable.** The generated dialect is written to disk, the JIT
   registers DWARF pointing at it, and gdb stops *inside the emitted MLIR* by
   file and line. Every ABI claim on this page is checkable with a breakpoint
   rather than by reasoning (§9).

The cost is honest and worth stating: Tier 2 needs a system LLVM/MLIR install
and a compiled dialect (`libJLCS.so`), which is the single largest dependency in
the project. C-only projects never touch it.

## 3. Where thunks sit

```
compilation_metadata.json  +  DWARF vtables          (Builder/)
                    │
                    ▼
        JLCSIRGenerator.generate_jlcs_ir            Julia emits MLIR *text*
        ├─ StructGen      → type aliases, DWARF-offset struct bodies
        ├─ FunctionGen    → one `func.func @<mangled>_thunk` per function
        ├─ ArrayViewGen   → strided accessors for fixed-size array members
        └─ STLContainerGen→ container accessor thunks
                    │  MLIR source text
                    ▼
        jlcsModuleCreateParse(ctx, text, "<pkg>/.debug/mlir/jlcs_<hash>.mlir")
                    │  every op stamped with a FileLineColLoc naming that buffer
                    ▼
        jlcs_lower_to_llvm  ── jlcs-lower-to-llvm
                            ── convert-func-to-llvm
                            ── convert-arith-to-llvm
                            ── reconcile-unrealized-casts
                            ── DIScopeForLLVMFuncOp   (line-table DWARF)
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
  ORC JIT engine           emit_object → clang -shared
  (one per library)        → lib<name>_thunks.so        [aot_thunks = true]
        │                        │
        ▼                        ▼
  JITManager.invoke(       direct ccall into the
   "_mlir_ciface_…")       companion library
```

The generator is Julia code that produces MLIR **as text**, which is then
parsed. That is a deliberate choice with two consequences worth knowing: it is
why no producer has ever needed the dialect's C++ `build()` methods (which is
how ten of them shipped undefined for years — §10), and it is why the parse
buffer can be a real file on disk, which is what makes the debugger work.

## 4. The thunk contract

Every Tier-2 function is reached through **one** C signature, regardless of what
it actually takes:

| Return | Emitted `_mlir_ciface_*` signature |
|---|---|
| Scalar / pointer | `T ciface(void** args)` |
| Any aggregate | `void ciface(T* sret, void** args)` |
| Void | `void ciface(void** args)` |

The two sides agree on which is which without negotiating: `emit_c_interface`
routes *every* struct-typed result through a result pointer, and
`JITManager._invoke_call` picks the `sret` shape for every non-`isprimitivetype`
Julia return type. (Note this is the ciface boundary, not the SysV classification
of §7 — the thunk's own call into the library is classified separately.)

That uniformity is forced by the Julia side. `ccall` needs its signature at
*macro expansion* time, so a runtime-general dispatcher cannot build one; a
single `void**` entry point can be called for any arity, and `JITManager.invoke`
is `@generated`, so each concrete argument tuple still specializes into
allocation-free code. `llvm.emit_c_interface` on the `func.func` is what makes
MLIR emit the `_mlir_ciface_` wrapper implementing exactly this.

### The arg-slot convention

**A slot holds a pointer to the argument's storage, not the argument.** The
thunk therefore double-loads:

```mlir
%arg_ptr_1 = llvm.getelementptr %args_ptr[%idx_1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
%val_ptr_1 = llvm.load %arg_ptr_1 : !llvm.ptr -> !llvm.ptr   // slot → storage address
%val_1     = llvm.load %val_ptr_1 : !llvm.ptr -> !llvm.ptr   // storage → value
```

`Ref(x)` on the Julia side produces exactly that shape for any `isbits` `x`.
Two argument kinds are *already* an indirection and must be flattened to a raw
pointer first, or the double-load spends one level too many and the callee
receives the payload's own bytes as an address:

- an `AbstractString` — `Ref(::String)` points at the String object, not at its
  bytes;
- a `Base.Ref` that is not already a `Ptr` — the spelling a caller reaches for
  when a parameter is annotated `::Ref{T}`, which is what a C++ `T const&`
  generates. `Ptr{T} <: Ref{T}`, so the annotation accepts a `RefValue`
  silently.

Both are handled by `_arg_marshal_plan` (one plan shared by both `invoke`
methods, because two copies is how a fix to one silently misses the other), and
the original is GC-preserved across the call since only a raw pointer into it
reaches the slot.

Callers packing a **by-value struct** argument put `&struct` in the slot
directly — not `&Ref` — because the struct's storage *is* what the callee wants.

### Worked example

A virtual method on a secondary base, from `test/mi_test/`. Julia side, as
emitted by the wrapper generator:

```julia
function Base2_get_b(this::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    return RepliBuild.JITManager.invoke("_mlir_ciface__ZNK5Base25get_bEv_thunk", Int32, this)
end
```

The thunk that name resolves to, as generated and as written to
`test/mi_test/.debug/mlir/jlcs_<hash>.mlir`:

```mlir
func.func private @_ZNK5Base25get_bEv(!llvm.ptr) -> i32
func.func @_ZNK5Base25get_bEv_thunk(%args_ptr: !llvm.ptr) -> i32
    attributes { llvm.emit_c_interface } {
  %idx_1     = arith.constant 0 : i64
  %arg_ptr_1 = llvm.getelementptr %args_ptr[%idx_1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
  %val_ptr_1 = llvm.load %arg_ptr_1 : !llvm.ptr -> !llvm.ptr
  %val_1     = llvm.load %val_ptr_1 : !llvm.ptr -> !llvm.ptr
  %ret_val   = "jlcs.vcall"(%val_1)
      { class_name = @Base2, vtable_offset = 0 : i64, slot = 2 : i64,
        this_offset = 0 : i64, may_throw } : (!llvm.ptr) -> i32
  return %ret_val : i32
}
```

and what that lowers to, read out of gdb's `disassemble /s` at the breakpoint:

```
%val_ptr_1 = llvm.load %arg_ptr_1     →  mov (%rdi),%rax     slot → storage address
%val_1     = llvm.load %val_ptr_1     →  mov (%rax),%rdi     storage → `this`
jlcs.vcall … { slot = 2 }             →  mov 0x10(%rax),%rax 2 × 8 = 0x10
```

The slot arithmetic is checkable by eye. That is the point of §9.

## 5. Types

The dialect defines two types. Both exist because LLVM's type system cannot say
the thing the ABI needs said.

### `!jlcs.c_struct`

```mlir
!jlcs.c_struct<"Vec3", [f32, f32, f32], [[0 : i64, 4 : i64, 8 : i64]], packed = false>
```

A C-ABI struct carrying its **Julia type name**, its flattened field types, the
**explicit byte offset of every field**, and a packed flag. An `!llvm.struct`
implies offsets from its element list; `!jlcs.c_struct` states them, which is
what lets the marshalling ops move a field from where Julia put it to where C
expects it without either side guessing. It converts to
`LLVM::LLVMStructType::getLiteral(fields, isPacked)` during lowering.

!!! warning "A `c_struct` must never appear inside an `!llvm.struct` body"
    The type converter treats `!llvm.struct` as legal and never rewrites its
    body — signature positions get rebuilt, body-op types do not. A nested
    `c_struct` alias therefore survives lowering and segfaults
    `translateModuleToLLVMIR` inside `PtrLikeTypeInterface::getMemorySpace`.
    `StructGen` inlines the byte-identical `!llvm.struct<packed (…)>` literal
    for `c_struct`-classified members instead, and `jlcs_create_jit` pre-flights
    every type in the module with `mlir::LLVM::isCompatibleType`, returning null
    (catchable in Julia → "Tier 2 disabled for this library") rather than
    crashing the process on anything foreign that still slips through.

### `!jlcs.array_view`

```mlir
!jlcs.array_view<f64, 3>
```

A strided array descriptor — element type and rank — whose runtime layout is

```c
struct ArrayView { T* data; int64_t* dims; int64_t* strides; int64_t rank; };
//                 @0       @8             @16               @24
```

chosen to be simultaneously expressible as a Julia `Array{T,N}` view, a NumPy
`ndarray`, a C++ strided span, and a Rust slice. The producer
(`ir_gen/ArrayViewGen.jl`) materializes one **in place** over a fixed-size array
member — the data pointer aims into the caller's struct, so element access is
zero-copy. Rank 1 today; the descriptor already carries dims and rank for the
generalization, and the lowering reads data and strides only.

## 6. The ops

Fourteen ops, **all fourteen with a producer** as of 2026-08-13. Liveness is
tracked deliberately, because "defined in TableGen" and "something emits it" are
different claims — and it is checked mechanically rather than asserted here, by
`test_jlcs_invariants.jl` §E, which reads the mnemonics out of `JLCSOps.td`,
greps `src/` for emission, and fails naming any op whose tier moved. That guard
exists because this table was wrong for months (see below).

| Op | Producer | Verifier | Role |
|---|---|---|---|
| `jlcs.type_info` | `JLCSIRGenerator` | ✅ | class metadata: layout, primary base, destructor, MI/VI base tables |
| `jlcs.ffe_call` | `FunctionGen` | — | direct call to a named symbol, SysV-coerced |
| `jlcs.try_call` | `FunctionGen` | — | as `ffe_call`, plus invoke + landing pad |
| `jlcs.vcall` | `FunctionGen` | ✅ | vtable dispatch: read vptr, index slot, adjust `this`, indirect call |
| `jlcs.marshal_arg` | `FunctionGen` | ✅ | Julia-aligned struct pointer → C-packed struct value |
| `jlcs.marshal_ret` | `FunctionGen` | — | C-packed struct value → Julia-aligned struct value |
| `jlcs.scope` | `FunctionGen` | ✅ | RAII region; destructors fire in reverse at exit |
| `jlcs.ctor_call` | `FunctionGen` | — | constructor on an object pointer |
| `jlcs.dtor_call` | `FunctionGen` | ✅ | destructor on an object pointer (exactly one operand) |
| `jlcs.yield` | `FunctionGen` | — | `jlcs.scope` terminator |
| `jlcs.load_array_element` | `ArrayViewGen` | — | strided read through an `!jlcs.array_view` |
| `jlcs.store_array_element` | `ArrayViewGen` | — | strided write |
| `jlcs.get_field` | `FunctionGen`, `ArrayViewGen` | ✅ | field read by byte offset |
| `jlcs.set_field` | `ArrayViewGen` | ✅ | field write by byte offset |

### How the last three got producers

Until 2026-08-13 the bottom three rows read `none`, and one of them read `none`
without saying so. `jlcs.dtor_call` was credited to `FunctionGen` here and in
the internal devlog for months while nothing emitted it: the scope-RAII producer
builds `jlcs.scope(...) dtors([@sym, ...])`, and it is the *scope's* lowering
that emits those calls (as `LLVM::CallOp`, directly), so `DestructorCallOp` had
a registered and unreachable conversion pattern. The pairing with `ctor_call` —
which does have a producer — is what made the gap read as symmetry.

- **`dtor_call`** — a destructor thunk is exactly this op's shape: one object
  pointer in, void out. It used to go out as a generic `ffe_call`/`try_call`
  with the full SysV coercion machinery attached to a signature that needs
  none. The op gained a `may_throw` unit attribute (mirroring `vcall`'s) with
  an invoke + landing-pad lowering, so a thunk moving onto it keeps the exact
  EH semantics it had — DWARF marks no destructor `noexcept`, so in practice
  C++ destructor thunks take the EH path.

  The **arity gate is load-bearing**: under Itanium a base-object destructor
  (`D2`) of a class with virtual bases takes a second VTT argument, and this op
  has no operand to carry it. `vi_test` has three (`_ZN4LeftD2Ev(this, vtt)`).
  Those keep the `ffe_call`/`try_call` path; converting them would drop the VTT
  silently. The gate reads the *final* argument list, after `this` synthesis,
  not the DWARF parameter list.

- **`get_field` / `set_field`** — the thunk argument array is a record with a
  fixed layout: the ciface convention hands the thunk a `void**`, so reading
  argument slot `i` is a field read at byte offset `8i`. That is what both
  producers now say, in one op, instead of a constant plus a pointer-scaled GEP
  plus a load. `ArrayViewGen` additionally builds the `!jlcs.array_view`
  descriptor with one `set_field` per field (data@0, dims@8, strides@16,
  rank@24) — the same terms the array-op lowering reads it back in, which
  previously was a GEP chain on the producer side and byte offsets on the
  consumer side with nothing tying them together.

This changes what the IR *says*, not what it *does*, and that is checked rather
than asserted: every function of `mi_test`, `vi_test` and `stl_test` (188
symbols) compiles to byte-identical machine code across the switch, and
`test_jlcs_producers.jl` §G pins the per-shape equivalence directly. The
motivation is that the `.mlir` is the debugger's source file (see *Debugging*
below) — `jlcs.get_field ... {fieldOffset = 8}` is what gdb shows at the
breakpoint, and the argument-slot convention it encodes is the one this
project's notes record as having cost a debugging session.

### Metadata: `type_info`

```mlir
jlcs.type_info "Derived",
  !jlcs.c_struct<"Derived", [i32], [[28 : i64]], packed = false>,
  "Base1", "_ZN7DerivedD0Ev"
  {baseNames = ["Base1", "Base2"], baseOffsets = [0 : i64, 16 : i64]}
```

Erased at lowering — it exists so the layout facts travel *with* the IR that
uses them. `superType` is the primary base (single-inheritance consumers).
`baseNames`/`baseOffsets` pair 1:1 and carry each non-virtual base subobject's
static offset. `vbaseNames`/`vbaseVtableOffsets` carry virtual bases, keyed on
the **negative** byte position of the vbase-offset entry relative to the vtable
address point — a virtual base has no static offset, so it can only be named by
where to *read* the offset. `destructorName` is the DWARF-resolved complete-object
destructor, which is what `Managed` finalizers and the RAII scope call. The
verifier enforces the 1:1 pairing, because consumers index one array by the
other's position.

### Calls: `ffe_call`, `try_call`, `vcall`

`ffe_call` is the direct call. `try_call` is the same call marshalled
identically and terminated differently — `llvm.invoke` with a landing pad, where
a caught C++ exception is recorded through `jlcs_catch_current_exception()` and
the call yields a **zero sentinel** rather than unwinding. The Julia side polls
`jlcs_has_pending_exception()` after every Tier-2 call and raises
`CxxException` with the original `what()` string. Sentinel-continue rather than
unwind-through-Julia is the whole reason a C++ exception is survivable here.

Which one a function gets is per-function: the module-level `may_throw` (set for
C++ builds) is overridden by the function's own DWARF `noexcept` flag.

`vcall` is dispatch through the object's vtable, so **overrides are honored** —
a direct symbol call is `p->Class::method()` static semantics, which is the
wrong answer for a base-class wrapper invoked on a derived object. The op takes
class-local coordinates (`vtable_offset`, `slot`, `this_offset`) and its
producer always emits `vtable_offset = this_offset = 0`; see
[The inheritance ABI](inheritance-abi.md) for why class-local coordinates are
universally correct under Itanium. Two deliberate exclusions: destructors stay
direct calls (exact-class semantics for finalizers and RAII), and struct-shaped
signatures stay direct because the `vcall` lowering does no sret/packed
coercion.

### RAII: `scope`, `ctor_call`, `dtor_call`, `yield`

```mlir
jlcs.scope(%tmp : !llvm.ptr) dtors([@_ZN4BaseD1Ev]) {
  jlcs.ctor_call @_ZN4BaseC1ERKS_(%tmp, %src) : (!llvm.ptr, !llvm.ptr) -> ()
  %r = jlcs.ffe_call %tmp { callee = @_Z5takesB } : (!llvm.ptr) -> i32
  llvm.store %r, %retslot : i32, !llvm.ptr
  jlcs.yield
}
```

Under Itanium, a by-value parameter of a class with a non-trivial destructor is
passed as a **pointer to a caller-owned temporary**, destructed by the caller
after the call — not as raw bits in registers. Passing the bits miscompiles
those calls. The producer detects the case by *symbol presence* (a truly trivial
destructor is never emitted as a symbol, so a class with one in the metadata is
non-trivial for the purposes of calls), copy-constructs a temporary when a copy
constructor is resolvable and bit-copies when the copy is trivial, and brackets
the call in a `jlcs.scope`. Destructors fire in reverse order at scope exit.

`jlcs.scope` has no results, so a return value leaves through a stack slot —
that is why the RAII path allocates `%raii_retslot`. Because `try_call` converts
an exception into sentinel-and-continue, the normal path is the *only* path, so
destructor coverage is total.

The verifier exists because an arity mismatch between `managed_ptrs` and
`destructors` used to SIGSEGV inside the lowering, which indexes one by the
other's position.

### Marshalling: `marshal_arg`, `marshal_ret`

```mlir
%packed = jlcs.marshal_arg %ptr
  { memberTypes = [i32, f64], juliaOffsets = [0 : i64, 8 : i64] }
  : (!llvm.ptr) -> !llvm.struct<packed (i32, f64)>
```

The layout-mismatch pair. `marshal_arg` reads each member from the **Julia**
offset and assembles the **C-packed** value; `marshal_ret` does the inverse for
returns. `memberTypes` and `juliaOffsets` pair 1:1 and the verifier says so —
same reason as `type_info`.

## 7. The lowering

One pass, `jlcs-lower-to-llvm`, run as a partial conversion with all fourteen
ops marked illegal and the LLVM + arith dialects legal. `func.func` is
*dynamically* legal — a signature containing `!jlcs.c_struct` cannot be handled
by the stock `ConvertFuncToLLVM` type converter, so those functions are claimed
by this pass with a converter that knows the type. The rest of the pipeline is
stock (`convert-func-to-llvm`, `convert-arith-to-llvm`,
`reconcile-unrealized-casts`), plus `DIScopeForLLVMFuncOp` last — see §9.

Inter-pass verification is deliberately off for the pipeline: the JLCS pass
emits `llvm.call` ops referencing `func.func` symbols that do not become
`llvm.func` until the next pass. The final module is fully verified.

### The SysV classifier

The heart of the whole thing is `classifySysVStruct`, and it is worth reading as
prose because every by-value struct crossing in RepliBuild runs through it:

1. Compute `abiSize`. `0` or `> 16` bytes → **MEMORY class**, full stop.
2. Otherwise flatten the struct to leaves with absolute offsets
   (`collectAbiLeaves`, recursing through nested structs and arrays).
3. If any leaf sits at an offset that is not a multiple of its natural size, the
   struct is genuinely misaligned (attribute-packed) → **MEMORY class**.
4. Otherwise mark, per eightbyte, whether any overlapping leaf is integer and
   whether any is floating point. An eightbyte is **SSE** only if every
   overlapping leaf is FP; otherwise **INTEGER**.
5. Emit one coercion scalar per eightbyte: `f64` for SSE, `i64` for INTEGER.

From that classification the shared `buildSysVCallShape` derives the entire call:

| Shape | Lowering |
|---|---|
| MEMORY-class **return** | alloca (`memorySlotAlign`), pass as first argument, call returns void, load the result back |
| Register-class **return** | call returns the coerced scalar(s); store to a slot, reload as the original struct type |
| Register-class **argument** | store to a slot, load one scalar per eightbyte, pass them as separate arguments — exactly clang's coercion |
| MEMORY-class **argument** | alloca + store + pass the pointer with `llvm.byval(T)` and `llvm.align` |
| Everything else | passed through unchanged |

Two details are load-bearing and were each a bug first:

- **`byval` must be stamped on the call site *and* the declaration.** The call
  site is what the backend lowers; the declaration is what the verifier checks
  it against. `retargetCallee` rewrites the external declaration to the coerced
  signature and copies the attributes onto it.
- **Slot alignment is explicit** (`memorySlotAlign` = `max(8, abiAlign)`).
  Clang never emits an sret/byval slot below 8 on x86-64, and a struct RepliBuild
  failed to model field-by-field degrades to `!llvm.array<N x i8>`, whose natural
  alignment is 1 — leaving alignment implicit would hand native code an
  underaligned buffer for exactly the types least understood.

`ffe_call` and `try_call` share one derivation of all of this. They used to
carry independent copies, which is precisely how a fix to one silently missed
the other; that shape recurs throughout §10.

### Exception lowering

`try_call` (and `vcall` with `may_throw`) emit `llvm.invoke` with a landing pad
that calls `__cxa_begin_catch` / `jlcs_catch_current_exception` /
`__cxa_end_catch`, records the message in a thread-local buffer, and continues
with a zero-valued result. A post-pipeline walk stamps `__gxx_personality_v0`
onto any `llvm.func` containing an `llvm.invoke` — it must run after
`ConvertFuncToLLVM`, since only `LLVM::LLVMFuncOp` carries a personality
attribute. The C++ runtime symbols themselves are registered into the JIT by
`JITManager` at engine setup.

## 8. Struct bodies come from DWARF offsets

A struct body emitted for a thunk must have the size and layout of the **C**
type, not of anything Julia or LLVM would pick. `StructGen` lays members out at
their DWARF offsets with explicit `!llvm.array<N x i8>` padding between them,
and verifies the result against a Julia mirror of LLVM's own `abiSize`/`abiAlign`
(`_mlir_layout`). One shared derivation (`_apply_dwarf_layout`) feeds all three
body builders.

A struct that cannot be laid out consistently — overlapping members, a member
whose emitted type will not sit at its offset, an unmeasurable member, or
members summing past `byte_size` — **degrades to a `byte_size`-byte opaque
region and warns**. Opaque, but never the wrong size.

That rule replaced the original "close the struct with one trailing filler of
`byte_size - Σ member sizes`", which double-counted: LLVM inserts interior
alignment padding itself, and DWARF reports enum members with `size = 0`. Every
non-packed struct with interior padding came out **larger than the C type**, and
since `llvm.emit_c_interface` stores a MEMORY-class result straight into the
caller's buffer while `JITManager` sizes that buffer from the *Julia* struct,
every such call wrote past a live Julia object. Measured on llama.cpp:
`llama_context_params` emitted 200 bytes against a native 160, overrunning a
160-byte `Ref` by 34. Every member offset was correct, which is why it presented
as intermittent corruption rather than as a marshalling bug. Pinned by
`test/test_struct_layout.jl`, which needs no toolchain and is negative-checked
against the old rule.

## 9. Debugging a thunk

**This is the default way to debug Tier 2. Reach for it before reading IR and
reasoning about it.**

Three pieces, each added for its own reason, compose into a source-level
debugger for generated marshalling code:

1. `options.enableGDBNotificationListener = true` — ORC registers every JIT'd
   object with gdb through `__jit_debug_register_code`.
2. MLIR's **parser** stamps every op with a `FileLineColLoc` naming its buffer,
   and `DIScopeForLLVMFuncOpPass` turns that into the emitted DWARF's `DIFile`.
3. The `.debug/mlir/jlcs_<hash>.mlir` dump is the file those locations point at
   — so it is not diagnostic output, it is **the debugger's source file**.

Nothing needs to be enabled and nothing links gdb: RepliBuild is the *publisher*
of LLVM's GDB JIT interface, and gdb reads the descriptor out of the inferior.
`clean()` removes `.debug`, and it regenerates on the next JIT init.

```bash
timeout -s KILL 175 gdb -batch -nx \
  -ex 'set pagination off' -ex 'set confirm off' -ex 'set listsize 4' \
  -ex 'handle SIGSEGV nostop noprint pass' \
  -ex 'set breakpoint pending on' \
  -ex 'break _ZNK5Base15get_aEv_thunk' -ex 'run' \
  -ex 'bt 1' -ex 'info source' -ex 'disassemble /s' \
  --args julia --project=. test/mi_test/verify.jl
```

```
Thread 1 "julia" hit Breakpoint 1, _ZNK5Base15get_aEv_thunk () at jlcs_83a242c27fb37885.mlir:126
126	  %val_ptr_1 = llvm.load %arg_ptr_1 : !llvm.ptr -> !llvm.ptr
Producer is MLIR.   Compiled with DWARF 4 debugging format.
```

Two flags are mandatory and fail silently without explanation: `set breakpoint
pending on` (the thunk does not exist until the JIT emits it, so the breakpoint
cannot resolve at load) and `handle SIGSEGV nostop noprint pass` (Julia's GC
uses SIGSEGV for write barriers). `disassemble /s` is the payoff — dialect ops
interleaved with the machine code they became.

### The static path — `RepliBuild.Debug`

gdb needs a live process stopped at the right moment. `RepliBuild.Debug` needs a
file, carries the same information, cannot wedge, and works on a package this
process never built. **Prefer it.**

```julia
ENV["REPLIBUILD_JIT_OBJDUMP"] = "1"   # BEFORE the wrapper loads — see below
D = RepliBuild.Debug

D.thunks("test/vi_test")                            # what you can ask about
D.walk("test/vi_test", "_ZNK5VBase3tagEv_thunk")    # MLIR + emitted asm, one call
D.disassemble(pkg; symbol = s)                      # objdump -dS, ops ↔ machine code
D.mlir_body(pkg, symbol)                            # just the dialect
D.dwarf(pkg; section = "line")                      # the address → MLIR-line table
```

The env var is **read, not passed**, and that is forced: MLIR must have the
object cache when the engine is *created*, and engines are cached per library
per process, so there is no later moment at which an argument could arrive.
Setting it after the first Tier-2 call is too late, and says so.

`.debug_info` holds only a compile unit — no `DW_TAG_subprogram`, no variable
DIEs. That is what `LineTablesOnly` means here, and it is why file and line are
perfect in gdb while `info locals` is empty. It is pinned *negatively* in
`test_debug_inspection.jl`, so raising the emission kind shows up as a failing
test rather than as a surprise.

Optional profiling: `REPLIBUILD_JIT_PROFILE` turns on LLVM's perf listener and
routes the per-process jitdump to a session directory. Off by default — the
listener writes to `$HOME/.debug/jit` with nothing rotating or expiring it.

## 10. Failure classes this architecture has hit

Recorded because the *shapes* recur, and because a reader deciding how much to
trust the marshalling layer deserves the real list rather than a claim of
correctness.

**Two generators, one guard.** The Julia wrapper generator and the MLIR thunk
generator derive the same decisions independently, so a gate added to one
silently misses the other. Dear ImGui exposed it: `is_method` is a string
heuristic (`::` in the demangled prefix) and Itanium mangles a namespace-scoped
free function identically to a member function, so the thunk generator
synthesized a `this` for `ImGui::GetVersion()`. The thunk loaded `args_ptr[0]`
and dereferenced it while Julia passed an empty array — 174 zero-arg functions
segfaulting on first call, 614 more with every argument shifted one slot. Both
generators now consult the aggregate table (a namespace is never a key in it),
matching on every `::` suffix, angle-bracket-depth aware. The same shape produced
the independent `ffe_call`/`try_call` SysV copies (§7).

**Emitted size ≠ DWARF `byte_size`.** §8. A live heap smasher on every
MEMORY-class struct return, invisible because every member offset was right.

**MEMORY-class by-value arguments had two wrong shapes.** Packed structs went as
a bare pointer (by reference — the callee reads an address where it expects
bytes); non-packed ones went as an LLVM first-class aggregate, which the backend
splits per *element*, shifting every later argument. One `llvm.byval` path now.
The discriminating fixture is `gap_probe` in `test/struct_abi/`, driven against
a real `clang++` callee — a self-JIT'd callee cannot catch a convention mismatch,
because both sides share it.

**Enum returns must be bare integers.** A struct result — even
`!llvm.struct<(i32)>` — makes `emit_c_interface` use the sret convention, while
the Julia side calls an `@enum` back as a scalar: the args pointer lands in the
sret slot and the call dereferences garbage. Found on tinyxml2's
`XMLDocument::Parse → XMLError`.

**A tier decided a presentation policy.** The C++ generator's Tier-2 branch
`continue`s past the ccall emission path, and the `char*` return policy
(`Union{String,Nothing}`, the `_ptr` sibling, the `[wrap.cstring_owned]` free)
lived only on that path — so 75 functions across five Hub packages handed back a
raw `Cstring` and silently discarded any declared deallocator. Not a marshalling
bug: the pointer was correct, it just arrived undressed. The rule the fix
encodes is the general one — **a dispatch tier decides how a function is called,
never how its result is presented** — and it is enforced by
`_assert_cstring_policy` on the wrapper write path rather than by convention.

**Ten undefined symbols shipped in `libJLCS.so` for its entire life.**
`dlopen(…, RTLD_NOW)` refused the library. Eight op `build()` bodies and two
`ArrayViewType` accessors were declared by TableGen (`skipDefaultBuilders`,
`genStorageClass = 0`) and never defined in C++. Nothing caught it because
`ccall((:sym, path), …)` binds **lazily** and every producer builds IR as text,
so no call site ever existed: the library loaded, the dialect worked, and the
whole Tier-2 suite was green on a binary the loader would reject if asked to
resolve it eagerly. The guard is a fresh-subprocess `RTLD_NOW` `dlopen` plus an
`nm -D --undefined-only` sweep (`test_jlcs_invariants.jl` §D).

**Nested `c_struct` in an `llvm.struct` body.** §5 — a whole-process load crash
on pugixml, now prevented at emission and caught by the `create_jit` pre-flight.

**Unbounded arity and malformed operand segments.** `llvm.call`'s
`AttrSizedOperandSegments` has two groups, not three; a hand-rolled
`OperationState` for the indirect vcall wrote `{1, nArgs, 0}` where
`{1 + nArgs, 0}` was required, and translation walked off the end. Use the
dedicated indirect-call builder.

The through-line: **every one of these was silent.** That is why the invariants
suite, the layout mirror, the `RTLD_NOW` probe, the op verifiers, and the
`create_jit` type pre-flight all exist — the architecture's real requirement is
not that marshalling be right, it is that wrong marshalling be *loud*.

## 11. Boundaries

Deliberately unbuilt, so they are not chased as bugs:

- **Array views are rank 1.** The producer's regex skips `T[N][M]`; the
  descriptor's dims/rank fields are populated but unread by the lowering, and
  the user-facing Julia accessors that would call these thunks are not emitted
  yet.
- **`vcall` gates on scalar/pointer signatures.** Virtual methods returning a
  struct by value or taking a packed struct by value keep the direct-call path;
  closing this means giving the vcall lowering `try_call`'s coercion.
- **Seven ops have no verifier** (`ffe_call`, `try_call`,
  `load/store_array_element`, `ctor_call`, `yield`, `marshal_ret`). No known
  crash paths, but hand-written IR gets no schema help. `get_field`,
  `set_field` and `dtor_call` gained theirs on 2026-08-13 with their producers
  — all three lowerings take a pointer operand on faith, which ODS's `AnyType`
  accepts and LLVM translation then rejects far from the mistake.
- **The classifier is x86-64 SysV only.** The 16-byte MEMORY threshold, 64-bit
  pointers, and i64/f64 eightbytes are hardcoded. Win64 and AAPCS are not
  modeled.
- **No in-IR virtual-base upcast op.** Virtual inheritance works through a
  dynamic upcast on the *Julia* side ([The inheritance ABI](inheritance-abi.md)),
  which needed zero dialect changes. An op would only be required for IR that
  must do the adjustment inside a thunk, and no producer needs that today.

## 12. AOT thunks

`[compile] aot_thunks = true` runs the identical generator and lowering at build
time, emits an object, and links it against the library into a companion
`lib<name>_thunks.so`. The wrapper then `ccall`s that library instead of calling
`JITManager.invoke`: no JIT at load, no MLIR runtime dependency at all after the
build. `libJLCS.so` is still required *to build*.

## 13. Building the dialect

```bash
cd src/mlir && ./build.sh
```

Needs system LLVM/MLIR 21+ (22.1.x here), CMake 3.20+, and `mlir-tblgen`. The
artifact is `src/mlir/build/libJLCS.so`, gitignored, and pinned to the installed
MLIR's **minor** SONAME (`libMLIR.so.22.1`).

- A **patch** bump inside a minor needs nothing — SONAMEs are unchanged and the
  existing `.so` keeps resolving. Rebuilding anyway is cheap and was verified
  clean.
- A **minor** bump requires `rm -rf build && ./build.sh` and a Tier-2 re-run.
  Check the release notes for MLIR entries first; that decides whether the bump
  is a formality or real work.
- A worktree or fresh clone has **no** `libJLCS.so` (it is a build artifact), so
  Tier 2 fails there in a way that reads exactly like a code regression. Build
  it, or symlink `src/mlir/build` from a tree that has one.

---

**Where the executable truth lives:** `test_mlir_templates.jl` (dialect
semantics: CStructs, RAII, vcall, sret, MI and the VI diamond),
`test_jlcs_invariants.jl` (op arity/liveness probes, the `RTLD_NOW` symbol
guard), `test_jlcs_producers.jl` (scope-RAII and array-view producers executing
through the real JIT), `test_struct_abi.jl` (SysV small-struct and `byval` ABI
against a real `clang++` callee), `test_struct_layout.jl` (DWARF-offset layout,
no toolchain), `test_multilib_jit.jl` (two wrappers, one session),
`test_debug_inspection.jl` (the debug surfaces above), and the fixtures
`test/mi_test/`, `test/vi_test/`, `test/stl_test/`.
