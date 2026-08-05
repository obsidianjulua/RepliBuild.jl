// test/struct_abi/src/abi_fixture.cpp — native callee for the x86-64 SysV
// small-struct ABI trace test (test_struct_abi.jl). Compiled with the system
// clang++ so the register conventions are the REAL ones, not the JIT's own —
// self-JIT'd callees can't catch a convention mismatch (both sides share it).
//
// Shapes covered (all trivially copyable):
//   H1 {void*}      8B, one INTEGER eightbyte  → RAX / RDI
//   P2 {int,int}    8B, two ints share one eightbyte (RAX packs both)
//   F2 {float,float}8B, one SSE eightbyte      → XMM0 (both floats)
//   B3 {long x3}   24B, MEMORY class           → sret, and byval as an ARG
//   Gap            24B, MEMORY class WITH interior padding — the shape the
//                  emitted-size bug mis-modelled (2026-08-05)

extern "C" {

typedef struct { void* p; } H1;
H1 h1_make(void* v) { H1 h; h.p = v; return h; }

typedef struct { int a, b; } P2;
P2 p2_make(int a, int b) { P2 s; s.a = a; s.b = b; return s; }
int p2_sum(P2 x) { return x.a + x.b; }

typedef struct { float x, y; } F2;
F2 f2_make(float x, float y) { F2 s; s.x = x; s.y = y; return s; }
float f2_sum(F2 v) { return v.x + v.y; }

typedef struct { long a, b, c; } B3;
B3 b3_make(long a, long b, long c) { B3 s; s.a = a; s.b = b; s.c = c; return s; }

// MEMORY-class struct as an ARGUMENT. SysV wants a caller-owned copy in the
// outgoing stack argument area (`byval`); passing the aggregate as an LLVM
// first-class value lets the backend split it per element across registers,
// and passing a bare pointer hands the callee an address where it expects
// bytes. Only a system-clang callee can catch either — a self-JIT'd one shares
// whatever convention the JIT chose.
long b3_sum(B3 v) { return v.a + v.b + v.c; }

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
long gap_probe(Gap g) {
    return (long)g.a * 1000000L + (long)g.b * 1000L
         + (g.f1 ? 1 : 0) + (g.f2 ? 2 : 0) + (g.f3 ? 4 : 0)
         + (g.p == (void*)0x1234 ? 100 : 0);
}

}
