#ifndef C_SRET_H
#define C_SRET_H

/* Library-free fixture for where C returns a record by value (Tier 3, C1).

   A record comes back either in registers or through a hidden pointer the
   caller passes as an extra first argument (sret). The wrapper has to pick the
   same one: an sret the callee does not expect shifts every argument by one and
   the return is garbage, with no error. Win64 decides by size alone (1, 2, 4
   and 8 bytes in RAX); SysV by size (> 16) or an unaligned field.

   Every function records what it read in c1_rec[], so the probe compares each
   argument member with what it passed. No system headers: the test also asks
   clang to lower this file for both targets. */

typedef struct { char m1; short m2[3]; } T4;            /* 8 B: RAX on both (C1, abi_fuzz seed 36) */
typedef struct { int m1; } R5;                          /* 4 B */
typedef struct { int m1; long long m2; } R4;            /* 16 B */
typedef struct { int n; char s[12]; } IC12;             /* 16 B: SysV INTEGER,INTEGER; Win64 sret */
typedef struct __attribute__((packed)) { char c; short s; char d; } P4;     /* 4 B: Win64 RAX; SysV MEMORY */
typedef struct __attribute__((packed)) { char c; int i; char pad[3]; } P8; /* 8 B: Win64 RAX; SysV MEMORY */
typedef struct { long long a, b; int c; } Big24;        /* 24 B: memory on both */
typedef struct { int n; short s[8]; } IS20;            /* 20 B: memory on both; natural, so a plain ccall */
typedef struct __attribute__((packed)) { char c; long long x; long long y; } PK17; /* 17 B: the explicit sret */
typedef struct __attribute__((packed)) { char x; short y; } In3;           /* 3 B, align 1 */
typedef struct { In3 in; char c[3]; } Out6;             /* 6 B: y at 1, SysV MEMORY; Win64 sret */
typedef struct { float f; int b : 3; int c; } BF12;     /* 12 B float blob: Win64 sret; SysV refused */

/* Records more aligned than the Julia type standing in for them. Each maker
   copies an aligned global out (`return g_x;`), which clang stores through the
   hidden return pointer with movaps: a buffer aligned only to the stand-in
   faults. The wrapper must pass a buffer at the record's alignment. */
typedef float al_vec4[4] __attribute__((aligned(16)));
typedef float al_f4v __attribute__((vector_size(16)));
typedef struct { _Alignas(16) float a; float b; int c : 3; } AB16;          /* 16 B blob, member _Alignas: Win64 sret; SysV refused */
typedef struct __attribute__((aligned(16))) { float x, y, z, w; } V4A;     /* 16 B: Win64 sret; SysV SSE,SSE */
typedef struct __attribute__((aligned(16))) { double a, b, c, d; } D4A;    /* 32 B, natural fields: memory on both */
typedef struct { char c; _Alignas(16) int i; } MA32;                       /* 32 B: memory on both */
typedef struct __attribute__((aligned(32))) { double d[2]; } A32;          /* 32 B, 32-aligned */
typedef struct { al_f4v v; int k; } HVec;                                  /* vector member: 16-aligned, no attribute */
typedef struct { al_vec4 v; float s; } HV4;                                /* member of an over-aligned typedef */
typedef struct { V4A a; float t; } NestA;                                  /* nested aligned record */

extern unsigned long long c1_rec[16];
extern int c1_rec_n;
extern AB16 g_ab16;
extern V4A g_v4a;
extern D4A g_d4a;
extern MA32 g_ma32;
extern A32 g_a32;
extern HVec g_hvec;
extern HV4 g_hv4;
extern NestA g_nesta;

T4    f4(R5 p1);                        /* C1, abi_fuzz seed 36 */
T4    f2(T4 p1, R4 p3);                 /* C1, abi_fuzz seed 3: the same record in and out */
IC12  make_ic12(int n, char c);
int   ic12_sum(IC12 v);
P4    make_p4(short s);
P8    make_p8(int i);
int   take_p8(P8 v);                    /* Win64: i64 on both views; SysV: refused */
Big24 make_big24(long long a, long long b, int c);
IS20  make_is20(int n);
PK17  make_pk17(long long x);
Out6  make_out6(short y);
BF12  make_bf12(float f, int b, int c); /* Win64: sret; SysV: refused */
AB16  make_ab16(float a, int c);        /* Win64: aligned sret; SysV: refused */
V4A   make_v4a(float x);
D4A   make_d4a(double a);
MA32  make_ma32(char c, int i);
A32   make_a32(double d);
HVec  make_hvec(float v, int k);
HV4   make_hv4(float v, float s);
NestA make_nesta(float x, float t);

#endif
