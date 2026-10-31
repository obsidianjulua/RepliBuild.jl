# LLVM's ABI library on the live Tier 2 path — design and measurement

**Date:** 2026-10-31
**Status:** design and measurement only. No code is changed by this document.
**Base:** `main` @ `f6f90df`. Measured on the reference host: Windows 11, MSYS2
CLANG64 LLVM/MLIR 23.1.4, Julia 1.13.3, and a libJLCS built with the oracle on
(CMake found `libLLVMABI.a`). Every function named here was read on that date.
Line numbers are left out on purpose.

**Question.** Tier 2 decides how each by-value record crosses a thunk under
x86-64 SysV with RepliBuild's own classifier, `FunctionGen.sysv_record_abi`
(`ir_gen/SysVRecordABI.jl`). It states the verdict on each `jlcs.ffe_call` and
`jlcs.try_call` (`sysv_args`, `sysv_ret`), and `buildCallShape`
(`impl/JLCSPasses.cpp`) acts on it. LLVM 23 ships a classifier for the same
rules (`llvm/ABI`), and today it is only a test oracle
(`impl/JLCSAbiOracle.cpp`, `test/test_sysv_abi_oracle.jl`). What would it take
for that library to decide on the live JIT/AOT path, keeping RepliBuild's
classifier as the fallback?

**Answer, in short.**

- The installed library has one classifier that matters here: x86-64 SysV.
  It has none for Win64, and it **ignores the calling convention it is given**:
  `CallingConv::Win64` gets SysV answers. On the reference host the swap changes
  nothing. Its gate must be the host convention, never the library's presence.
- Its type description can carry every DWARF fact RepliBuild records, and some
  RepliBuild does not record (SIMD vectors, `__float128`, member pointers). The
  limit is the metadata, not the library.
- Put the swap on the generator side. `sysv_record_abi` asks the library
  through a libJLCS export (the oracle's JSON contract, promoted), and the
  answer goes through the existing `sysv_args`/`sysv_ret` statement.
  `buildCallShape` does not change.
- On disagreement, **refuse**: an ABI trap whose reason names both answers.
  Where the two agree, the emitted answer is LLVM's.
- Measured today: 1168 record crossings, **0 disagree**. The one `:trap` in
  `SAO_CURATED`, `make_c_ld`, is the st0 return, and LLVM answers it exactly:
  `direct x86_fp80` (X87).
- **New finding, outside the oracle's reach:** on SysV, a record holding a
  16-byte SIMD vector crosses in two XMM registers on both tiers, and C
  passes it in one (section 4.3). The metadata spells the vector as an array, and
  both sides of the oracle read that spelling. LLVM's library gets it right only
  once the parser records the vector.
- Tier 3 stays guard-and-refuse. The library can supply only the "what C wants"
  half of GeneratorC's guards, and only on SysV.

---

## 1. What the installed library covers

### 1.1 Targets

`libLLVMABI.a` holds six objects: `Types`, `FunctionInfo`, `TargetInfo`,
`IRTypeMapper`, `BPF`, `X86`. `TargetInfo.h` has two factories:
`createBPFTargetInfo` and `createX86_64TargetInfo(TB, X86AVXABILevel,
Has64BitPointers, ABICompatInfo)`. Every classifier symbol in the `X86` object
belongs to `X86_64TargetInfo` (`classify`, `merge`, `postMerge`,
`getIndirectResult`, …). There is no Win64, i386 or AArch64 classifier.

`X86_64TargetInfo::computeInfo` does not dispatch on the function's calling
convention. A probe (below) built the same records under `CallingConv::C` and
`CallingConv::Win64`:

| record | `CallingConv::C` | `CallingConv::Win64`, library | Win64, as clang lowers it |
|---|---|---|---|
| `{float, float}` | direct `<2 x float>` (XMM) | direct `<2 x float>` | `i64` in RCX |
| `{float, float, float}` | direct `{<2 x float>, float}` | direct `{<2 x float>, float}` | by reference |
| `{char[3]}` | direct `i24` | direct `i24` | by reference |

On Windows the library returns a wrong answer and no error. The swap's gate is
`kHostAbi` in the dialect and `Sys.iswindows()` in Julia, which is the gate
`sysv_trap_reason` and `classifyStruct` already use. The library's export
existing is not a gate.

By LLVM release (from the CMake probe's comment, not re-measured here): 21 has
no ABI library, 22 has the type system only, 23 has the x86-64 classifier. CI's
`tier2-linux.yml` on LLVM 22 always takes the fallback. On 23 it depends on how
the distribution packages the component (archive or dylib). The CMake probe
already handles both.

The probe is a 150-line C++ program linked against `libLLVMABI.a` and
`libLLVM-23.dll`. It and the two Julia scripts behind section 4 are kept in this
session group's shared notes, not in the repository.

### 1.2 What a type description needs

| kind | inputs |
|---|---|
| `IntegerType` | bit width, alignment, signed, `_BitInt` flag |
| `FloatType` | semantics (`half`, `bfloat`, `single`, `double`, `x87`, `quad`), alignment |
| `PointerType` | size, alignment, address space |
| `ArrayType` | element, count, size in bits, matrix flag |
| `VectorType` | element, count, alignment |
| `ComplexType` | element, alignment |
| `MemberPointerType` | function or data, size, alignment |
| `RecordType` | fields (`FieldInfo`: type, offset in bits, bitfield flag and width, unnamed-bitfield flag), base classes, virtual base classes, size, alignment, packing, flags |

The record flags are `CanPassInRegisters`, `IsUnion`, `IsTransparent`,
`IsCXXRecord`, `IsPolymorphic` and `HasFlexibleArrayMember`. The target takes
three more inputs: the AVX level (`None`, `AVX`, `AVX512`), 64-bit pointers,
and `ABICompatInfo`. That last one has five flags, and its
`fromVersion(clangMajor)` is a TODO in 23.1.4 that returns the current ABI's
defaults.

Two traps for whoever writes the adapter. Both were measured:

- **`RecordFlags` defaults to `None`, and a record without
  `CanPassInRegisters` is passed and returned by reference** (`indirect`,
  `byval` false), C or C++. The C adapter must set the flag on every record.
  The oracle does.
- **The AVX level changes the answer.** `{__m256}` is `indirect, align 32` at
  `None` and `direct <8 x float>` at `AVX`. The callee's compile flags decide it
  (`-mavx`, `-march=`). `None` is clang's x86-64 baseline. This matters only once
  vectors reach the description (section 4.3).

### 1.3 RepliBuild's DWARF facts, expressed

Every fact `sysv_record_abi` reads has a place in the description:

| DWARF fact (where RepliBuild keeps it) | `llvm::abi` | measured |
|---|---|---|
| members at offsets (`members[].offset`) | `FieldInfo(type, offset_bits)` | the whole corpus |
| bitfields (`bit_size`, `data_bit_offset`); unnamed ones | `IsBitField`, `BitFieldWidth`; `IsUnnamedBitfield` | `c_bf`, `c_bf_f`, `c_bf2_f`, `c_pp1_bf`, `c_ubf_f`, `c_ubf_d` agree |
| unions (`kind = "union"`) | `getUnionType` (every field at 0) | `c_u_fi`, `c_u_ff`, `c_u_ip`, `c_u_pk`, `c_u_ld_i` agree |
| nested records, arrays | a `RecordType` or `ArrayType` field | `c_nest_f2d`, `c_wrap_df`, `c_fa3` agree |
| `long double` | `FloatType(x87)`, align 16 | `{long double}`: argument `indirect, byval, align 16`; return `direct x86_fp80` |
| `_Complex` | `ComplexType` | `c_cf`, `c_cd`, `c_f_cf` agree |
| `__int128` | `IntegerType(128)`, align 16 | `c_i128` (direct `i128`, two INTEGER), `c_i128_l` (MEM 16) agree |
| `DW_AT_alignment` on a record (`alignment`) | the record's `Align` (a required input) | `c_al8_c`, `c_al16_big` agree |
| member alignment, `packed`, `#pragma pack` | member offsets plus the record's alignment | `c_al16_f`, `c_pk_*`, `c_pp2_cd`, `c_pk_alnest` agree. The oracle passes `StructPacking::Default` for every record, because the misaligned-field rule reads offsets |
| empty record (size 0 in C, 1 in C++) | a `RecordType` with no fields | `ignore` both ways, C and C++ (probe) |
| `DW_CC_pass_by_reference` (`record_abi[...]["pass"]`) | `IsCXXRecord` without `CanPassInRegisters` | `{int, int}`: `indirect, byval false, align 4` both ways (probe) |
| base classes (`base_classes`) | `RecordType` bases | `D : B{double} {int}` → direct `{double, i32}` (probe) |
| virtual bases, polymorphic classes | `VirtualBaseClasses`, `IsPolymorphic` | such a class is already `pass_by_reference` in DWARF |
| flexible array member (an unbounded array) | `HasFlexibleArrayMember` | MEMORY: argument align 8, return align 4 (probe). By reading the code, RepliBuild's verdict is also MEMORY, `mem:8` |

The library can also take facts that RepliBuild's metadata does not carry:

| fact | `llvm::abi` | library's answer | RepliBuild today |
|---|---|---|---|
| SIMD vector (`DW_AT_GNU_vector`) | `VectorType` | `{__m128}` → direct `<4 x float>` (one XMM) | spelled `float[4]`: section 4.3 |
| `__float128` | `FloatType(quad)` | direct `fp128` (one XMM: SSE, SSEUP) | `:unknown`, so a SysV trap |
| member pointer | `MemberPointerType` | data → `i64`; function → `{i64, i64}` | not measured |
| transparent union | `IsTransparent` | its first field (`ptr`) | a union of one INTEGER eightbyte, by reading: same register |

**So the question in the brief has a yes answer.** Everything RepliBuild records
can be expressed, and the limit is what the metadata records. A member whose type is
not in the metadata (`:unknown`: an STL container, a declared-only record) has
no description for anyone, and stays a trap.

---

## 2. Where the swap goes

There are three places the decision could be made:

1. **Generator side, carried by the existing statement.** `sysv_record_abi`
   asks the library, and FunctionGen states the answer exactly as it states its
   own today.
2. **A new statement carrying the type description**, with `buildCallShape`
   calling the library while lowering.
3. **`buildCallShape` alone, from the body.** Not possible. The body does not
   have the facts, which is why the statements exist (`SysVRecordABI.jl`'s
   header lists what a body cannot carry).

**Recommendation: 1.** Reasons:

- **One answer for both generators.** GeneratorCpp decides at wrap time which
  functions get a Julia call and which get a trap (`thunk_crossings` →
  `sysv_trap_reason`). FunctionGen decides what each call states. Both ask
  `sysv_record_abi`. Under option 2 the lowering would decide the shape while
  Julia still decides the trap: two derivations of one fact.
- **The decision is in the text.** The statement shows in
  `.debug/mlir/jlcs_<hash>.mlir` and in `Debug.walk`, and the AOT and JIT paths
  compile the same text.
- **The grammar already carries every answer the library gives** for the
  records RepliBuild classifies today. The library coerces more narrowly
  (`i32`, `i24`, `float`, `<2 x float>`) where the statement says `i64` or
  `f64` per eightbyte. Each pair uses the same register, and the slot the thunk
  stores to or loads from is never smaller than the record. That is why the
  oracle compares classes, not coerce types.
- **`buildCallShape` is untouched**, including its register accounting and the
  Win64 path. The 30 crossings that spill when registers run out agree with the
  library's own accounting (section 4.1), so moving the accounting into the
  library gains nothing that was measured.

What option 1 takes:

1. **libJLCS.** Promote the oracle's entry point to a live export with the same
   JSON contract, compiled under the CMake probe that exists today. That keeps
   it optional at build time in the same way. CMake looks for
   `llvm/ABI/TargetInfo.h`, then compiles a probe that makes the oracle's calls
   against `libLLVMABI.a`, or link-probes the LLVM dylib. Only when one succeeds
   does it add `impl/JLCSAbiOracle.cpp` and its library. Without it, libJLCS on
   LLVM 21 and 22 builds as it does today. The library comes from the LLVM
   install libJLCS is pinned to, which is normally the install whose clang
   compiles every Tier 2 callee. So classifier and compiler do not drift
   apart, and `ABICompatInfo`'s current-ABI default is right. The
   comments that say "test only, nothing on the wrapping or JIT path calls this"
   change with it: the oracle source's header, `CMakeLists.txt`, and the oracle
   bullet in CLAUDE.md. The header of `JLCSAbiOracle.cpp` is already out of
   date: it still names `classifySysVStruct` as the classifier under test.
2. **The adapter moves into `src/`.** The DWARF → `llvm::abi` description is
   `sao_type` and `sao_record` in the test. It moves into `SysVRecordABI.jl`
   and gains what C++ needs and the C-only corpus never exercised:
   `base_classes`, `record_abi`'s `pass` (`CanPassInRegisters`,
   `IsCXXRecord`), and `empty`. The test's version takes a record's alignment
   from `Wrapper._c_record_align` and lowers it in place. The live one must use
   the classifier's own `_sysv_record_align`, so there is one derivation.
3. **`sysv_record_abi` asks per record and per direction.** It describes the
   record as `void take(R)` and as `R make(void)`, because the two answers differ
   for `long double` (argument MEMORY 16, return X87), for flexible array
   members (align 8 against 4) and for pass-by-reference. The answer maps onto
   `SysVRecordABI` like this:
   - `direct` → `:regs` with its classes (`NONE` at offset 8);
   - `indirect` → `:memory` with its alignment;
   - `ignore` → `:ignore`;
   - a lone `x86_fp80` return → `:x87`.

   An answer outside the grammar (SSEUP, a YMM vector, X87 anywhere else)
   becomes `:unknown`, with the library's answer as the reason, and traps as
   today.
4. **Availability.** One `dlsym` on libJLCS per process, with
   `Sys.iswindows()` checked before it. Results are cached per record key and
   metadata.
5. **Provenance.** `thunk_manifest.json` records which classifier decided and
   its LLVM version. Wrap time and load time can run different libJLCS builds:
   a wrapper made on 22 and loaded on 23, or the reverse. Under the policy in
   section 3 the library can add traps but never removes one. So a wrapper made
   with the library traps a superset of what a load-time generator without it
   traps, which is harmless. In the reverse case, a wrapper made without the
   library calls a function that the load-time generator then refuses, and the
   call must fail with that reason. The stamp is what makes the failure
   explainable.
6. **Tests.**
   - `test_sysv_abi_oracle.jl` must keep diffing RepliBuild's classifier against
     the library. Once the live path takes the library's answer, the test's JLCS
     side has to force RepliBuild's classifier, through a keyword on
     `sysv_record_abi` threaded through `generate_jlcs_ir`. Otherwise the diff
     compares the library with itself and passes without checking anything.
   - Add a C++ corpus: bases, pass-by-reference, empty classes, polymorphic
     classes. Check it against `clang++ --target=x86_64-unknown-linux-gnu` the
     way the curated C records are checked. The adapter's C++ half is unmeasured
     until this exists.
   - Add a disagreement test with a stubbed library answer that asserts the trap
     and its reason, negative-checked by neutering the comparison.

---

## 3. Disagreement: refuse

| policy | what a disagreement becomes |
|---|---|
| refuse | an ABI trap (`abi_trap`) naming both verdicts; GeneratorCpp's stub explains it |
| prefer LLVM | LLVM's verdict, with RepliBuild's logged |
| log only | whichever side is primary, silently |

**Recommendation: refuse.** Where the two agree, the statement is the library's
answer, and its provenance says so.

- **A live disagreement has three possible causes**: RepliBuild's classifier is
  wrong, the adapter is wrong, or the library is wrong. The library is checked
  against clang on the 44 curated records (88 checks). The adapter is checked
  only through the oracle corpus, which is C only. The live path sees records
  no test has seen: Hub packages, and C++ records with bases and flags.
  Preferring LLVM trusts new adapter code on the live path. Preferring
  RepliBuild defeats the point. Logging only is one of those two with the
  evidence buried.
- **A wrong call shape usually gives a plausible wrong value, not a crash**
  (W2 and C1 in CLAUDE.md, and section 4.3 here). So the rule is Tier 1's: any doubt resolves to the
  safe path. For a by-value record on Tier 2 the safe path is the ABI trap,
  which already exists and which GeneratorCpp already explains.
- **A trap that names both answers and the record is a bug report.** The build
  prints the count the way it prints the `tier1:` line. Today's number is 0 of
  1168.

The path to "the library decides" alone: once a release's Hub rebuild shows
zero disagreement traps, flip the policy to "prefer LLVM, log RepliBuild's
answer". From then on RepliBuild's classifier is the fallback where the library
is absent, and nothing else.

**A third witness, for later.** For a function the library defines, clang's own
lowering is in the linked IR's `define` line. The build already reads two facts
from it (`_nounwind_definitions`, `_byval_definitions`). That is the compiler that
built the callee, not a reimplementation of it. It exists on Win64 too, where
`llvm::abi` has nothing, and it would have shown the vector case below directly:
the definition takes `<4 x float>` where the thunk passes `double, double`. A
check that each thunk's callee declaration lowers to the definition's parameter
list would catch any classifier's error, the library's included. It needs a
mapping from DWARF parameters to IR parameters: a record can split into two, an
empty record vanishes, and `sret` and `this` come first. That mapping is a
classification of its own, so this is a check, not a source. It is not designed
here.

---

## 4. Measured

### 4.1 The oracle test

`julia --project=. test/test_sysv_abi_oracle.jl` on the reference host, at
`f6f90df`:

```
SysV classification, JLCS vs LLVM's ABI library: 306 records, 912 functions
(19 refused on SysV), 1168 record crossings compared, 0 disagree.
Test Summary:                                    | Pass  Total  Time
SysV struct classification vs LLVM's ABI library | 1886   1886  2.8s
```

The 306 records are 44 curated ones and 262 from abi_fuzz seeds 1 to 60. Of
the 1168 crossings, 87 belong to curated records and 1081 to abi_fuzz records
and functions. The oracle-against-clang testset ran (clang is on `PATH`) and
passed for all 88 curated `take_`/`make_` functions.

Every verdict in the agreed set, as the library gave it:

| verdict | crossings |
|---|---|
| `REG[INTEGER]` | 362 |
| `MEM(align 8)` | 353 |
| `REG[INTEGER,INTEGER]` | 184 |
| `SRET(align 8)` | 95 |
| `SRET(align 1)` | 44 |
| `SRET(align 4)` | 42 |
| `REG[INTEGER,SSE]` | 20 |
| `REG[SSE,SSE]` | 20 |
| `SRET(align 2)` | 16 |
| `REG[SSE,INTEGER]` | 11 |
| `REG[SSE]` | 10 |
| `MEM(align 16)` | 4 |
| `SRET(align 16)` | 3 |
| `IGNORE` | 2 |
| `REG[NONE,SSE]` | 2 |

30 of the `MEM(align 8)` crossings, counted the way the test's harness counts
them, are records that go in registers alone and in memory inside a call
because the registers ran out. The library passes 17 of them directly on the
stack and 13 as a `byval`. Both sides agree on all 30.

The test's own caveat holds and matters for section 4.3: both sides read one
description of each record, so a wrong DWARF fact is not caught.

### 4.2 The `:trap` rows of `SAO_CURATED`, and the other refusals

`SAO_CURATED` says `:trap` for exactly one crossing:

| function | why FunctionGen refuses | LLVM's library |
|---|---|---|
| `make_c_ld` (`{ long double x; }` returned) | `sysv_trap_reason`: "SysV returns on the x87 stack (st0), where a thunk does not look" | `direct x86_fp80`, classes `[X87]`: returned in st0 |

The library does not refuse it. It gives the exact shape, and clang agrees: the
oracle-against-clang testset covers `make_c_ld`. The refusal is the thunk's,
because the statement grammar has no x87 return. Accepting it would take a
statement for it, a callee declared `-> f80` (the X86 backend reads st0 for an
`f80` result), and a store into the record's slot. That is a candidate follow-up
once the library decides, not part of the swap. `take_c_ld` already agrees:
`MEM(align 16)`.

The other 18 refused functions are abi_fuzz functions that take a bare
`long double` (10) or return one (8):

- take: `s10_f1`, `s12_f1`, `s12_f2`, `s31_f5`, `s35_f2`, `s35_f3`, `s46_f1`,
  `s52_f4`, `s57_f4`, `s58_f2`;
- return: `s5_f2`, `s16_f3`, `s16_f6`, `s18_f1`, `s26_f1`, `s28_f4`, `s29_f3`,
  `s36_f5`.

FunctionGen refuses them before any classification: `by_value_crossing` calls a
`long double` `:opaque`, because no Julia value holds 80 bits. The library
classifies them (an X87 return; an X87 argument goes on the stack). The oracle
reports such an argument as `REG[X87]` only because its register replay counts
INTEGER and SSE and nothing else. The swap gains nothing here.

### 4.3 What the oracle cannot see: SIMD vector members

A five-record C unit, built by RepliBuild on the reference host, then asked
three questions. What the metadata says. What `sysv_record_abi` answers, and
the shape the dialect's SysV probe (`jlcs_probe_sysv_call_shapes`, the oracle
test's JLCS side) gives it. What clang emits for `x86_64-unknown-linux-gnu`.

| record | metadata member | `sysv_record_abi` | JLCS shape, SysV | clang, SysV |
|---|---|---|---|---|
| `{ v4f v; }` (`float`, `vector_size(16)`) | `float[4]`, size 16, align 16 | `regs:sse,sse` | `f64, f64` (XMM0, XMM1) | one XMM: `<4 x float>` |
| `{ v2d v; }` (`double`, `vector_size(16)`) | `double[2]`, size 16, align 16 | `regs:sse,sse` | `f64, f64` | one XMM: `<2 x double>` |
| `{ v2f v; }` (`vector_size(8)`) | `float[2]`, size 8, align 8 | `regs:sse` | `f64` | one XMM: same register |
| `{ v2f v; int i; }` | `float[2]`, `int` | `regs:sse,int` | `f64, i64` | `SSE, INTEGER`: same |
| `{ __float128 q; }` | `__float128` | `:unknown` | refused on SysV | one XMM: `fp128` |

The two 16-byte vector records disagree with C. As an argument, the thunk puts
the low eightbyte in XMM0 and the high one in XMM1, and the callee reads all
four lanes from XMM0, whose upper half is zero or stale. As a return, the thunk
reads the upper half from XMM1. The result is a plausible wrong value, not a
crash. **This was not run on Linux.** It was measured with the dialect's SysV
probe on the reference host, the same path the oracle test uses.

Tier 3 makes the same mistake. Asked for SysV (`Wrapper._c_record_abi`
redefined to `:sysv` in a throwaway process, then a re-wrap), GeneratorC emits
`ccall((:take_c_v4f, …), Cfloat, (c_v4f,), x)` and a `c_v4f` return with no
trap. The Julia stand-in is `struct c_v4f; v::NTuple{4, Cfloat}; end`, and
Julia's `ccall` lowers a 16-byte struct of four `Float32` as two SSE
eightbytes. Windows is unaffected. Win64 passes a 16-byte record by reference:
the IR definitions take `ptr`, and both tiers pass one.

**Why the oracle misses it.** The metadata spells a vector member as an array
(`float[4]`). The parser keeps only a vector's size, as its alignment, and drops
`DW_AT_GNU_vector`. The oracle's adapter reads that spelling as an array too,
so both sides agree on a wrong description. The clang check covers the curated
records only, and none of them has a vector.

The library gets it right once it is told (`{__m128}` → direct `<4 x float>`
in the probe). That needs:

- the parser to record the vector flag;
- the adapter to pass a `VectorType`;
- for the thunk to call the function at all, a statement for one 128-bit SSE
  register (SSE, SSEUP), which the grammar does not have.

Until then the right answer for both tiers is a trap. A bare vector parameter
(`float take_v4f(v4f)`) has the same spelling, `float[4]`, and Tier 3 wraps it
as `NTuple{4, Cfloat}` by value. It was not measured on either tier.

This defect exists today, and fixing it does not depend on the swap. It is
listed as the first step in section 6.

---

## 5. Tier 3

Tier 3's call is lowered by Julia's own `ccall` code, which classifies the Julia
stand-in type GeneratorC writes, not the C type. No library changes what Julia
does with a `ccall`. Each GeneratorC guard compares two answers: what C wants
(`_c_return_in_memory`, `_c_explicit_sret`, `_c_blob_param_misclassified`,
`_c_register_class_misstated`, combined in `_c_byvalue_verdict`), and what Julia
will do with the stand-in. Where the two differ, it traps or reshapes (BRANCH 0,
`_RB_ByVal16`, the aligned sret buffer).

The library could supply only the first half, and only on SysV. The second half
is RepliBuild's model of Julia's lowering, and it stays that. The `byval` slot
alignment, the one fact where Tier 3 went wrong in a way DWARF could not show,
already comes from clang's own IR (`_byval_definitions`), which is a stronger
source than the library for that fact.

**So Tier 3 stays guard-and-refuse.** Once the Tier 2 swap lands, GeneratorC's
SysV "what C wants" can ask the same `sysv_record_abi`, so both generators take
one answer. That is where the vector case would be caught on Tier 3. It would
also replace GeneratorC's coarser float-risk test (`class_risk`) with an exact
class, which may let through records that are trapped today. This was not
measured.

---

## 6. Steps, in order

1. **The vector fact** (a live defect on SysV, independent of the swap). The
   parser records `DW_AT_GNU_vector`, from both dumpers. Check the spelling
   against a real readelf dump, as the two-dialect rule says. `sysv_record_abi`
   traps a record holding a vector, and GeneratorC traps it on SysV. Add a
   curated vector record to `SAO_CURATED` with a clang check. A 128-bit SSE
   statement that lets these calls through is a later step.
2. **A C++ corpus in the oracle test**, still test only: bases,
   pass-by-reference, empty classes, polymorphic classes. This measures the
   adapter's C++ half before anything depends on it.
3. **The swap**: the live export, the adapter in `src/`, `sysv_record_abi`
   with the refuse policy, the provenance stamp, a forced-RepliBuild mode for
   the oracle test, and the disagreement test. Update the CLAUDE.md oracle
   bullet ("nothing on the JIT or wrap path calls it") in the same change.
4. **Before a release**, rebuild the Hub's C++ packages on a Linux LLVM 23
   host, and count disagreement traps.
5. **Later, each on its own:**
   - an x87 return statement (`make_c_ld` and every `long double` record
     return);
   - an SSEUP statement (16-byte vectors, `__float128`);
   - the `define`-line check (section 3).

   Each lets through functions the library classifies exactly.

Never on Win64. The library's Win64 answers are SysV's (section 1.1).
