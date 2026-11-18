/* test/typemap_c_test: how a C type becomes a Julia type (test_type_mapping_live.jl).
 *
 *   - a pointer to an enum, in every position: parameter, array parameter,
 *     return, struct member, global;
 *   - `unsigned __int128` and `__int128`, in the registers and beyond them;
 *   - a scalar the type map has no Julia type for (`_Float16`).
 */
#ifndef TYPEMAP_H
#define TYPEMAP_H

enum Small { SM_A, SM_B, SM_C };                       /* an open enum */
enum Perm  { P_R = 1, P_W = 2, P_X = 4, P_ALL = 7 };    /* a bitmask: an integer alias */

typedef struct { enum Small *kinds; int n; } Holder;

void         out_small(enum Small *out);                /* *out = SM_C */
void         smalls_fill(enum Small xs[], int n);       /* xs[i] = i % 3 */
int          smalls_sum(const enum Small *xs, int n);
int          small_swap(enum Small *a, enum Small *b);  /* swaps, returns 1 */
enum Small  *smalls_static(void);                       /* {SM_C, SM_B} */
enum Small **smalls_handle(void);                       /* &pointer to the above */
Holder       holder_make(void);                         /* {smalls_static(), 2} */
void         out_perm(enum Perm *out);                  /* *out = P_R | P_X */
extern enum Small *g_smalls;                            /* smalls_static() */

typedef unsigned __int128 u128;
typedef __int128 i128;

unsigned __int128 u128_echo(unsigned __int128 v);
__uint128_t       u128_add(__uint128_t a, __uint128_t b);
unsigned long long u128_hi(u128 v);
unsigned long long u128_lo(u128 v);
u128              u128_make(unsigned long long hi, unsigned long long lo);
void              u128_out(u128 *out);                  /* 0xfedcba9876543210_0123456789abcdef */
/* Both halves in registers: four integer arguments leave r8 and r9. */
u128 u128_after4(long a, long b, long c, long d, u128 v);
/* A double takes no integer register. */
u128 u128_after_double(double x, long a, long b, long c, long d, u128 v);
/* One integer register left: C puts the whole value on the stack. */
u128 u128_after5(long a, long b, long c, long d, long e, u128 v);
/* None left, and an odd stack word before it: the 16-byte boundary shows. */
u128 u128_after7(long a, long b, long c, long d, long e, long f, long g, u128 v);
i128 i128_after5(long a, long b, long c, long d, long e, i128 v);
/* The value on the stack leaves r9 free for the argument after it. */
u128 u128_then_long(long a, long b, long c, long d, long e, u128 v, long tail);

_Float16 f16_echo(_Float16 x);
int      f16_bits(_Float16 x);
int      plain_add(int a, int b);

#endif
