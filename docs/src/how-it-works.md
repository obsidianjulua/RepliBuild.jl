# How It Works: Two JITs, One IR

Most people think of FFI as a wall between languages — you call a C function, pay some overhead, and get a result back. RepliBuild works differently. Instead of bridging two separate worlds, it makes both languages converge at the same level: **LLVM intermediate representation**. The FFI boundary is not a wall — it is two compilers meeting in the middle.

```
Julia source                          C++ source
     |                                     |
     v                                     v
Julia compiler                         Clang
     |                                     |
     v                                     v
Julia LLVM IR  -------- merge ------>  C++ LLVM IR
                          |
                          v
                    Machine code
```

Both Julia and C++ compile to LLVM IR. When both IRs are visible to the same JIT compiler, the language boundary ceases to exist — the optimizer can inline, vectorize, and constant-fold across it as if everything were written in one language.

But LLVM IR alone is not enough. When the two languages disagree on how data is laid out in memory (packed structs, vtable dispatch, platform-specific alignment), a second compiler — the MLIR JIT — steps in with the JLCS dialect to translate between them.

This page walks through concrete examples from the RepliBuild test suite to show exactly what happens at each level.

## The easy case: scalar functions

Consider the simplest possible C function:

```cpp
// test/jit_edge_test/src/jit_edges.cpp
int scalar_add(int a, int b) {
    return a + b;
}
```

Clang compiles this to LLVM IR:

```llvm
; test/jit_edge_test/build/jit_edges.ll
define i32 @scalar_add(i32 noundef %0, i32 noundef %1) {
  %5 = load i32, ptr %3, align 4
  %6 = load i32, ptr %4, align 4
  %7 = add nsw i32 %5, %6
  ret i32 %7
}
```

The operation is a single `add nsw i32` instruction. RepliBuild's wrapper generator sees that both arguments and the return type are primitives, so `is_ccall_safe()` returns `true` and the function is routed to Tier 1:

```julia
# test/jit_edge_test/julia/JitEdgeTest.jl (generated)
function scalar_add(a::Integer, b::Integer)::Cint
    a_c = Cint(a)
    b_c = Cint(b)
    return ccall((:scalar_add, LIBRARY_PATH), Cint, (Cint, Cint), a_c, b_c)
end
```

When Julia's JIT compiles this wrapper, we can inspect the LLVM IR it actually produces with `@code_llvm`:

```llvm
; @code_llvm scalar_add(1, 2) — Julia's compiled IR for the wrapper
define i32 @julia_scalar_add(i64 signext %"a::Int64", i64 signext %"b::Int64") {
top:
  ; Julia truncates Int64 → Int32 (with bounds check)
  %7 = trunc i64 %"a::Int64" to i32
  %6 = trunc i64 %"b::Int64" to i32

  ; The ccall — a direct call to the C++ symbol within LLVM IR
  %8 = call i32 @jlplt_scalar_add_got.jit(i32 %7, i32 %6)
  ret i32 %8
}
```

Both the C++ function and the Julia wrapper exist as LLVM IR. The `ccall` compiles to a `call` instruction **within** LLVM IR — not a call across a language boundary. The PLT stub (`@jlplt_scalar_add_got.jit`) resolves to the compiled C++ code at load time.

For an even simpler case where the types already match, the Julia IR becomes almost trivial:

```llvm
; @code_llvm scalar_mul(1.0, 2.0) — Float64 types match directly
define double @julia_scalar_mul(double %"a::Float64", double %"b::Float64") {
top:
  %0 = call double @jlplt_scalar_mul_got.jit(double %"a::Float64", double %"b::Float64")
  ret double %0
}
```

Two lines: call the C++ function, return the result. No conversion, no boxing, no overhead beyond the indirect call itself.

This is already efficient (one indirect call), but the language boundary still exists as a call instruction. LTO removes it entirely.

## LTO: the boundary vanishes

When `enable_lto = true`, RepliBuild embeds the C++ LLVM bitcode directly into the Julia module as a constant:

```julia
# Generated wrapper with LTO enabled
const LTO_IR = read("mylib_lto.bc")  # C++ LLVM bitcode

function scalar_add(a::Cint, b::Cint)::Cint
    if !isempty(LTO_IR)
        return Base.llvmcall((LTO_IR, "scalar_add"), Cint, Tuple{Cint, Cint}, a, b)
    else
        return ccall((:scalar_add, LIBRARY_PATH), Cint, (Cint, Cint), a, b)
    end
end
```

`Base.llvmcall` passes the C++ bitcode to Julia's LLVM JIT at compile time. The JIT reads the C++ IR, sees `add nsw i32`, and **inlines it** into the calling Julia function. The result:

```llvm
; Julia's JIT output — C++ code inlined, no call instruction
define void @julia_run_cpp_math_loop(...) {
top:
  br label %L22

L22:
  %value_phi = phi double [ 0.0, %top ], [ %next_val, %L22 ]

  ; These were C++ operations — now inlined into the Julia loop:
  %cpp_mul_result = fmul double %value_phi, 3.14159
  %cpp_add_result = fadd double %cpp_mul_result, 1.0

  %next_val = add i64 %iv, 1
  %exitcond = icmp eq i64 %next_val, %max_iters
  br i1 %exitcond, label %L_exit, label %L22

L_exit:
  ret void
}
```

There are **zero `call` instructions**. The C++ `fmul` and `fadd` operations are fused directly into the Julia loop nest. The language boundary has ceased to exist at the IR level — Julia's optimizer treats the C++ code as its own.

The performance confirms it:

| Path | ns/iter |
|------|---------|
| Pure Julia | 0.677 |
| LTO `llvmcall` | 0.677 |
| Wrapper `ccall` | 2.026 |

The LTO path matches pure Julia exactly because the optimizer sees identical IR.

## When alignment breaks: packed structs

Not all functions can merge at the IR level. Consider this packed struct:

```cpp
// test/jit_edge_test/include/jit_edges.h
#pragma pack(push, 1)
typedef struct PackedTriplet {
    char tag;       // offset 0, 1 byte
    int value;      // offset 1, 4 bytes (NOT offset 4 — packed, no padding)
    char flag;      // offset 5, 1 byte
} PackedTriplet;    // total: 6 bytes
#pragma pack(pop)

PackedTriplet pack_three(char tag, int value, char flag);
```

Clang compiles this to LLVM IR with a packed struct type and 1-byte alignment:

```llvm
; The angle brackets <{ }> mean "packed" — no padding between fields
%struct.PackedTriplet = type <{ i8, i32, i8 }>

define void @pack_three(ptr sret(%struct.PackedTriplet) align 1 %0,
                        i8 signext %1, i32 %2, i8 signext %3) {
  ; Stores at offsets 0, 1, 5 — no alignment padding
  %9  = getelementptr inbounds %struct.PackedTriplet, ptr %0, i32 0, i32 0
  store i8  %1,  ptr %9,  align 1
  %11 = getelementptr inbounds %struct.PackedTriplet, ptr %0, i32 0, i32 1
  store i32 %2,  ptr %11, align 1    ; i32 at offset 1, aligned to 1 byte
  %13 = getelementptr inbounds %struct.PackedTriplet, ptr %0, i32 0, i32 2
  store i8  %3,  ptr %13, align 1
  ret void
}
```

The critical detail: `align 1` on the `sret` pointer and all stores. The `i32` field sits at byte offset 1, not offset 4.

Now look at what Julia sees:

```julia
# Generated struct definition (from DWARF metadata)
struct PackedTriplet
    tag::UInt8      # offset 0, 1 byte
    value::Cint     # offset 4, 4 bytes — Julia aligns i32 to 4 bytes!
    flag::UInt8     # offset 8, 1 byte
end
# sizeof(PackedTriplet) == 12 in Julia (with padding)
# sizeof(PackedTriplet) == 6 in C++ (packed)
```

Julia's type system enforces natural alignment: an `Int32` must start at a 4-byte boundary. Julia's `PackedTriplet` is 12 bytes with padding. C++ expects 6 bytes with no padding. If Julia passed this struct via `ccall`, the `value` field would be at the wrong offset — undefined behavior, likely a crash.

This is exactly what `is_ccall_safe()` detects. It compares the DWARF-reported struct size (6 bytes) against Julia's aligned size (12 bytes). When they differ, the function is routed to Tier 2:

```julia
# Generated wrapper — routed to MLIR JIT (Tier 2)
function pack_three(tag::UInt8, value::Integer, flag::UInt8)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    return RepliBuild.JITManager.invoke(
        "_mlir_ciface_pack_three_thunk", PackedTriplet, tag, value, flag)
end
```

## The MLIR thunk: translating between ABIs

The MLIR JIT generates a thunk that bridges Julia's natural alignment and C++'s packed layout. Here is the **actual generated MLIR IR** for `pack_three`, produced by the `JLCSIRGenerator` from the DWARF metadata:

```mlir
// Struct aliases — the dialect registers both packed and aligned layouts
!Struct_PackedTriplet = !jlcs.c_struct<"PackedTriplet", [i8, i32, i8],
    [[0 : i64, 1 : i64, 5 : i64]], packed = true>

module {
  // External C++ function — returns packed struct
  func.func private @pack_three(i8, i32, i8) -> !llvm.struct<packed (i8, i32, i8)>

  // Thunk: bridges Julia's ciface calling convention to C++ packed sret
  func.func @pack_three_thunk(%args_ptr: !llvm.ptr)
      -> !llvm.struct<(i8, i32, i8)>
      attributes { llvm.emit_c_interface } {

    // Step 1: Extract each argument from Julia's pointer array
    //         Julia passed: [ptr_to_tag, ptr_to_value, ptr_to_flag]
    %idx_1 = arith.constant 0 : i64
    %arg_ptr_1 = llvm.getelementptr %args_ptr[%idx_1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %val_ptr_1 = llvm.load %arg_ptr_1 : !llvm.ptr -> !llvm.ptr
    %val_1 = llvm.load %val_ptr_1 : !llvm.ptr -> i8        // tag

    %idx_2 = arith.constant 1 : i64
    %arg_ptr_2 = llvm.getelementptr %args_ptr[%idx_2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %val_ptr_2 = llvm.load %arg_ptr_2 : !llvm.ptr -> !llvm.ptr
    %val_2 = llvm.load %val_ptr_2 : !llvm.ptr -> i32       // value

    %idx_3 = arith.constant 2 : i64
    %arg_ptr_3 = llvm.getelementptr %args_ptr[%idx_3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %val_ptr_3 = llvm.load %arg_ptr_3 : !llvm.ptr -> !llvm.ptr
    %val_3 = llvm.load %val_ptr_3 : !llvm.ptr -> i8        // flag

    // Step 2: Call C++ function — returns packed struct (6 bytes)
    %ret_packed = jlcs.ffe_call %val_1, %val_2, %val_3
        { callee = @pack_three } : (i8, i32, i8) -> !llvm.struct<packed (i8, i32, i8)>

    // Step 3: Convert packed result to aligned layout for Julia
    //         Extract each field from packed, insert into aligned struct
    %ret_aligned_undef = llvm.mlir.undef : !llvm.struct<(i8, i32, i8)>
    %ret_field_1 = llvm.extractvalue %ret_packed[0] : !llvm.struct<packed (i8, i32, i8)>
    %ret_aligned_1 = llvm.insertvalue %ret_field_1, %ret_aligned_undef[0] : !llvm.struct<(i8, i32, i8)>
    %ret_field_2 = llvm.extractvalue %ret_packed[1] : !llvm.struct<packed (i8, i32, i8)>
    %ret_aligned_2 = llvm.insertvalue %ret_field_2, %ret_aligned_1[1] : !llvm.struct<(i8, i32, i8)>
    %ret_field_3 = llvm.extractvalue %ret_packed[2] : !llvm.struct<packed (i8, i32, i8)>
    %ret_aligned_3 = llvm.insertvalue %ret_field_3, %ret_aligned_2[2] : !llvm.struct<(i8, i32, i8)>

    return %ret_aligned_3 : !llvm.struct<(i8, i32, i8)>
  }
}
```

The key operation is in three steps:

1. **Load arguments** from Julia's pointer array (Julia passed `Ref`s, the thunk dereferences them)
2. **Call C++** via `jlcs.ffe_call` which handles the packed sret convention
3. **Convert the result** from `packed (i8, i32, i8)` (6 bytes, C++ layout) to `(i8, i32, i8)` (12 bytes, Julia-aligned layout) by extracting and reinserting each field

The `llvm.emit_c_interface` attribute tells MLIR to generate the `_mlir_ciface_pack_three_thunk` entry point that Julia's `JITManager` calls.

Meanwhile, Julia's JIT compiles the `pack_three` wrapper to this LLVM IR:

```llvm
; @code_llvm pack_three(UInt8(1), Int32(42), UInt8(1))
define void @julia_pack_three(ptr sret({ i8, i32, i8 }) align 8 %sret_return,
                              i8 zeroext %"tag::UInt8",
                              i32 signext %"value::Int32",
                              i8 zeroext %"flag::UInt8") {
top:
  %sret_box = alloca [3 x i32], align 4
  call void @j_invoke(%sret_box, @"jl_global_thunk_name",
                      i8 %"tag::UInt8", i32 %"value::Int32", i8 %"flag::UInt8")
  call void @llvm.memcpy(ptr %sret_return, ptr %sret_box, i64 12)
  ret void
}
```

Julia's IR calls `JITManager.invoke()` (compiled as `@j_invoke`), which resolves to the MLIR-compiled thunk via the lock-free symbol cache. The result comes back through an sret buffer. Both JITs produced their IR independently — Julia compiled the wrapper, MLIR compiled the thunk — and they meet at the `call` instruction.

### The call sequence

Here is what happens when Julia calls `pack_three(UInt8(1), Int32(42), UInt8(1))`:

```
Julia                          MLIR Thunk                        C++
─────                          ──────────                        ───
1. Create Refs:
   r1 = Ref(UInt8(1))
   r2 = Ref(Int32(42))
   r3 = Ref(UInt8(1))

2. Build pointer array:
   inner_ptrs = [&r1, &r2, &r3]
                    │
                    ▼
              3. ciface entry:
                 _mlir_ciface_pack_three_thunk(inner_ptrs)
                    │
              4. Load from Julia-aligned ptrs:
                 tag   = load i8  from inner_ptrs[0]
                 value = load i32 from inner_ptrs[1]
                 flag  = load i8  from inner_ptrs[2]
                    │
              5. Call with packed sret (align 1):
                 @pack_three(%sret, tag, value, flag)
                                                          │
                                                    6. Store packed:
                                                       offset 0: tag
                                                       offset 1: value
                                                       offset 5: flag
                                                          │
              7. Return packed result ◄───────────────────┘
                    │
8. Read result  ◄───┘
   from sret buffer
```

The MLIR thunk is a **translator between two valid but incompatible memory layouts**. Julia's layout is correct for Julia (natural alignment). C++'s layout is correct for C++ (packed). The thunk bridges them at the IR level by moving individual field values — not by copying memory blocks.

## Virtual dispatch: same problem, same solution

C++ and Julia both solve the same fundamental problem: given a value, select the right implementation of a function based on its runtime type. They just use different mechanisms.

### C++ approach: vtables

```cpp
// test/vtable_test/include/shapes.h
class Shape {
public:
    virtual double area() const { return 0.0; }
    virtual double perimeter() const { return 0.0; }
};

class Circle : public Shape {
    double radius;
public:
    double area() const override;       // slot 1 in vtable
    double perimeter() const override;  // slot 2 in vtable
};

class Rectangle : public Shape {
    double width, height;
public:
    double area() const override;
    double perimeter() const override;
};
```

At the machine level, calling `shape->area()` means:
1. Read the vtable pointer from `shape` (first 8 bytes of the object)
2. Index into the vtable to get the function pointer for `area` (slot 1)
3. Call the function pointer

The JLCS dialect makes this explicit as a verifiable IR operation:

```mlir
// jlcs.vcall encodes the vtable dispatch contract:
// "read vtable at offset 0, index slot 1, call with %obj"
%result = jlcs.vcall @Shape::area(%obj)
    { vtable_offset = 0 : i64, slot = 1 : i64 } : (!llvm.ptr) -> f64
```

This lowers through MLIR's pass pipeline to the same LLVM IR that a C++ compiler would generate: a GEP to the vtable pointer, a load of the function pointer at the slot offset, and an indirect call.

### Julia approach: multiple dispatch

RepliBuild generates idiomatic Julia wrappers that use Julia's own dispatch mechanism:

```julia
# test/vtable_test/julia/VtableTest.jl (generated)

# Idiomatic types wrapping C++ objects
mutable struct Circle
    handle::Ptr{Cvoid}
    function Circle(r::Cdouble)
        handle = create_circle(r)
        obj = new(Ptr{Cvoid}(handle))
        finalizer(obj) do o
            delete_shape(o.handle)
        end
        return obj
    end
end

mutable struct Rectangle
    handle::Ptr{Cvoid}
    function Rectangle(w::Cdouble, h::Cdouble)
        handle = create_rectangle(w, h)
        obj = new(Ptr{Cvoid}(handle))
        finalizer(obj) do o
            delete_shape(o.handle)
        end
        return obj
    end
end

# Multiple dispatch — Julia selects the method by type
area(obj::Circle) = Circle_area(obj.handle)
area(obj::Rectangle) = Rectangle_area(obj.handle)
```

User code reads naturally:

```julia
c = Circle(5.0)
r = Rectangle(3.0, 4.0)

area(c)  # → Julia dispatches to Circle_area → calls _ZNK6Circle4areaEv
area(r)  # → Julia dispatches to Rectangle_area → calls _ZNK9Rectangle4areaEv
```

Julia's JIT compiles each dispatch path independently. Here is the actual LLVM IR for `area(::Circle)`:

```llvm
; @code_llvm area(Circle(5.0))
define double @julia_area(ptr noundef nonnull align 8 dereferenceable(8) %"obj::Circle") {
top:
  ; Load the raw C++ pointer from the Circle handle field
  %"obj::Circle.handle" = load ptr, ptr %"obj::Circle", align 8

  ; Call the mangled C++ symbol directly — one instruction
  %0 = call double @jlplt__ZNK6Circle4areaEv_got.jit(ptr %"obj::Circle.handle")
  ret double %0
}
```

And the x86 assembly it compiles to:

```asm
; @code_native area(Circle(5.0))
julia_area:
  push    rbp
  mov     rdi, qword ptr [rdi]           ; load handle from Circle struct
  movabs  rax, offset _ZNK6Circle4areaEv  ; address of C++ Circle::area()
  mov     rbp, rsp
  call    rax                             ; call C++ directly
  pop     rbp
  ret
```

Three real instructions: load the pointer, call the C++ function, return. Julia's multiple dispatch resolved `area(::Circle)` to `Circle_area` at compile time, and the JIT compiled it down to a direct call to the mangled C++ symbol. No vtable indirection, no runtime type check — Julia already knows the concrete type.

### The parallel

Both languages solve the same problem — **select the right implementation based on the runtime type** — using different mechanisms that compile to the same kind of machine code:

| | C++ | Julia | MLIR (JLCS) |
|---|---|---|---|
| **Mechanism** | vtable pointer + slot index | method table + type tag | `jlcs.vcall` operation |
| **Dispatch** | Indirect call through function pointer | Compiled specialization per type | Lowers to indirect call |
| **Result** | Same machine code | Same machine code | Same machine code |

The JLCS dialect does not invent a new dispatch mechanism. It encodes the **existing** C++ vtable dispatch as a structured, verifiable IR operation that MLIR can reason about, optimize, and lower to correct native code.

## Both JITs produce IR for the same function

To see the two JITs operating side by side, consider that the MLIR JIT generates thunks for **every** function in the library — even scalar ones that Tier 1 handles via `ccall`. Here is the MLIR thunk for `scalar_add` alongside Julia's compiled IR for the same function:

**Julia's JIT** produces this LLVM IR for `scalar_add` (via `@code_llvm`):

```llvm
define i32 @julia_scalar_add(i64 signext %"a::Int64", i64 signext %"b::Int64") {
  %6 = trunc i64 %"b::Int64" to i32
  %7 = trunc i64 %"a::Int64" to i32
  %8 = call i32 @jlplt_scalar_add_got.jit(i32 %7, i32 %6)
  ret i32 %8
}
```

Julia reasons: truncate `Int64` to `Int32`, call the C++ symbol, return.

**MLIR's JIT** produces this IR for the same function (via `JLCSIRGenerator`):

```mlir
func.func private @scalar_add(i32, i32) -> i32

func.func @scalar_add_thunk(%args_ptr: !llvm.ptr) -> i32
    attributes { llvm.emit_c_interface } {
  %idx_1 = arith.constant 0 : i64
  %arg_ptr_1 = llvm.getelementptr %args_ptr[%idx_1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
  %val_ptr_1 = llvm.load %arg_ptr_1 : !llvm.ptr -> !llvm.ptr
  %val_1 = llvm.load %val_ptr_1 : !llvm.ptr -> i32

  %idx_2 = arith.constant 1 : i64
  %arg_ptr_2 = llvm.getelementptr %args_ptr[%idx_2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
  %val_ptr_2 = llvm.load %arg_ptr_2 : !llvm.ptr -> !llvm.ptr
  %val_2 = llvm.load %val_ptr_2 : !llvm.ptr -> i32

  %ret_val = jlcs.ffe_call %val_1, %val_2 { callee = @scalar_add } : (i32, i32) -> i32
  return %ret_val : i32
}
```

MLIR reasons: dereference the pointer array to get `i32` values, call the C++ symbol via `jlcs.ffe_call`, return.

Both JITs are reasoning about the **same operation** — "call `scalar_add` with two `i32` arguments" — using their own IR and their own type system. Julia uses `trunc` to convert from its native `Int64`. MLIR uses `llvm.getelementptr` + `llvm.load` to unpack the ciface pointer array. Both arrive at the same `call @scalar_add(i32, i32) -> i32` in the final machine code.

For `scalar_add`, Tier 1 (`ccall`) wins because it skips the pointer array overhead. But for `pack_three`, only the MLIR thunk can correctly marshal the packed struct layout. The tier selection in `is_ccall_safe()` picks the right path automatically.

**Clang** produced this LLVM IR for the C++ side — the target both JITs ultimately call:

```llvm
define i32 @scalar_add(i32 noundef %0, i32 noundef %1) {
  %5 = load i32, ptr %3, align 4
  %6 = load i32, ptr %4, align 4
  %7 = add nsw i32 %5, %6
  ret i32 %7
}
```

Three compilers. Three IRs. One `add` instruction. The language boundary exists only in source code — by the time any of these reach the CPU, they are all the same machine code.

## The full picture

Both compilation pipelines produce LLVM IR. For simple functions (POD types, scalar returns), the IRs merge directly — either through `ccall` (a function call within LLVM IR) or `llvmcall` (the C++ IR is inlined into the Julia IR by the JIT).

For complex cases (packed structs, unions, virtual dispatch), the MLIR JIT generates thunks in the JLCS dialect that translate between the two ABIs. These thunks are lowered through MLIR's standard pipeline to LLVM IR and compiled to native machine code:

```
Julia source                                C++ source
     |                                           |
     v                                           v
Julia compiler                               Clang
     |                                           |
     v                                           v
Julia LLVM IR                              C++ LLVM IR
     |                                           |
     +--- Tier 1 (simple): ccall ─────────> symbol resolution
     |                                           |
     +--- Tier 1 (LTO): llvmcall ──> merge ─────+──> Machine code
     |                                           |
     +--- Tier 2: JITManager.invoke() ──────────>|
                    |                             |
                    v                             |
              JLCS MLIR dialect                   |
                    |                             |
                    v                             |
              func dialect                        |
                    |                             |
                    v                             |
              LLVM dialect                        |
                    |                             |
                    v                             |
              MLIR LLVM IR ──> Machine code ──> calls C++ symbols
```

The JLCS dialect is the **missing piece** — it handles the cases where Julia's LLVM IR and C++'s LLVM IR cannot merge directly because they disagree on data layout. The dialect encodes the ABI contract (field offsets, packing flags, vtable slots) as first-class IR operations that MLIR can verify and lower to correct code.

After the first call to any Tier 2 function, the compiled machine code is cached in the `JITManager`'s lock-free symbol dictionary. Subsequent calls are a single `Dict` read (no lock, no JIT overhead) followed by a `ccall` to the cached function pointer — the same cost as a regular `ccall`.

The result: whether a function goes through Tier 1 (`ccall`/`llvmcall`) or Tier 2 (MLIR thunks), the generated machine code is equivalent. The dispatch tier is chosen automatically based on ABI complexity, and the user never needs to know which path a given function takes.
