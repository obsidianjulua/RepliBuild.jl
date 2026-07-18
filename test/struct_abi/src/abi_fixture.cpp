// test/struct_abi/src/abi_fixture.cpp — native callee for the x86-64 SysV
// small-struct ABI trace test (test_struct_abi.jl). Compiled with the system
// clang++ so the register conventions are the REAL ones, not the JIT's own —
// self-JIT'd callees can't catch a convention mismatch (both sides share it).
//
// Shapes covered (all trivially copyable):
//   H1 {void*}      8B, one INTEGER eightbyte  → RAX / RDI
//   P2 {int,int}    8B, two ints share one eightbyte (RAX packs both)
//   F2 {float,float}8B, one SSE eightbyte      → XMM0 (both floats)
//   B3 {long x3}   24B, MEMORY class           → sret

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

}
