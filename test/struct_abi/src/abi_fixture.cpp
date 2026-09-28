// test/struct_abi/src/abi_fixture.cpp — native callee for the x86-64 SysV
// small-struct ABI trace test (test_struct_abi.jl). Compiled with the system
// clang++ so the register conventions are the REAL ones, not the JIT's own —
// self-JIT'd callees can't catch a convention mismatch (both sides share it).
//
// Shapes covered (all trivially copyable):
//   H1 {void*}      8B, one INTEGER eightbyte  → RAX / RDI
//   P2 {int,int}    8B, two ints share one eightbyte (RAX packs both)
//   F2 {float,float}8B, one SSE eightbyte      → XMM0 (both floats)
//   B3 {long long x3} 24B, MEMORY class        → sret, and byval as an ARG
//                  `long long`, NOT `long`: this file states sizes, and `long`
//                  is the one integer the word size does not settle (4 bytes
//                  under LLP64, 8 under Unix64). As `long` the struct was 12
//                  bytes on Windows while the traces below — hand-written MLIR
//                  at i64, and NTuple{3,Int64} on the Julia side — still said
//                  24, so b3_make returned (0x8_00000007, …) for (7, 8, 9).
//                  The shape under test is MEMORY-class-by-sret, not the width
//                  of a long; test_llp64_widths.jl owns that question.
//   Gap            24B, MEMORY class WITH interior padding — the shape the
//                  emitted-size bug mis-modelled (2026-08-05)
//   DI, F3 + spill register-class structs that arrive after the registers they
//                  need are used up — passed in memory, whole (2026-09-26)

extern "C" {

typedef struct { void* p; } H1;
H1 h1_make(void* v) { H1 h; h.p = v; return h; }

typedef struct { int a, b; } P2;
P2 p2_make(int a, int b) { P2 s; s.a = a; s.b = b; return s; }
int p2_sum(P2 x) { return x.a + x.b; }

typedef struct { float x, y; } F2;
F2 f2_make(float x, float y) { F2 s; s.x = x; s.y = y; return s; }
float f2_sum(F2 v) { return v.x + v.y; }

typedef struct { long long a, b, c; } B3;
B3 b3_make(long long a, long long b, long long c) { B3 s; s.a = a; s.b = b; s.c = c; return s; }

// MEMORY-class struct as an ARGUMENT. SysV wants a caller-owned copy in the
// outgoing stack argument area (`byval`); passing the aggregate as an LLVM
// first-class value lets the backend split it per element across registers,
// and passing a bare pointer hands the callee an address where it expects
// bytes. Only a system-clang callee can catch either — a self-JIT'd one shares
// whatever convention the JIT chose.
long long b3_sum(B3 v) { return v.a + v.b + v.c; }

// Interior padding (4 bytes after `a`) plus a trailing bool run, then tail
// padding: 24 bytes whose member sizes sum to only 19. That gap is what the
// old "one trailing filler of byte_size - sum(member sizes)" rule paid for a
// second time, emitting a 32-byte body for a 24-byte type.
typedef struct { int a; void* p; int b; bool f1, f2, f3; } Gap;

Gap gap_make(int a, void* p, int b) {
    Gap g; g.a = a; g.p = p; g.b = b; g.f1 = true; g.f2 = false; g.f3 = true;
    return g;
}

// Every field participates, so a mis-marshalled by-value copy is a wrong
// NUMBER rather than a crash we might read as something else.
long long gap_probe(Gap g) {
    return (long long)g.a * 1000000LL + (long long)g.b * 1000LL
         + (g.f1 ? 1 : 0) + (g.f2 ? 2 : 0) + (g.f3 ? 4 : 0)
         + (g.p == (void*)0x1234 ? 100 : 0);
}

// Register exhaustion. SysV has 6 integer and 8 SSE argument registers, and a
// register-class struct that does not fit in what is LEFT goes to memory whole:
// it is never split between the last free registers and the stack. `DI` needs
// one SSE and one INTEGER register and arrives after six ints; `F3` needs two
// SSE and arrives after eight doubles. Both are memory operands here. The thunk
// counted no registers and split them (audit F20, 2026-09-26). Every argument
// feeds the sum, so a misplaced one is a wrong number: 84 when right.
typedef struct { double d; int i; } DI;      // 16B: SSE, INTEGER
typedef struct { float x, y, z; } F3;        // 12B: SSE, SSE
double spill(int a, int b, int c, int d, int e, int f, DI s,
             double x0, double x1, double x2, double x3,
             double x4, double x5, double x6, double x7, F3 t) {
    return a + b + c + d + e + f + s.d + s.i
         + x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + t.x + t.y + t.z;
}

}
