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
typedef struct { int n; short s[8]; } IS20;            /* 20 B: memory on both; an explicit sret here */
typedef struct __attribute__((packed)) { char x; short y; } In3;           /* 3 B, align 1 */
typedef struct { In3 in; char c[3]; } Out6;             /* 6 B: y at 1, SysV MEMORY; Win64 sret */
typedef struct { float f; int b : 3; int c; } BF12;     /* 12 B float blob: Win64 sret; SysV refused */

extern unsigned long long c1_rec[16];
extern int c1_rec_n;

T4    f4(R5 p1);                        /* C1, abi_fuzz seed 36 */
T4    f2(T4 p1, R4 p3);                 /* C1, abi_fuzz seed 3: the same record in and out */
IC12  make_ic12(int n, char c);
int   ic12_sum(IC12 v);
P4    make_p4(short s);
P8    make_p8(int i);
int   take_p8(P8 v);                    /* Win64: i64 on both views; SysV: refused */
Big24 make_big24(long long a, long long b, int c);
IS20  make_is20(int n);
Out6  make_out6(short y);
BF12  make_bf12(float f, int b, int c); /* Win64: sret; SysV: refused */

#endif
