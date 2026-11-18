#include "typemap.h"

static enum Small small_store[2] = { SM_C, SM_B };
enum Small *g_smalls = small_store;
static enum Small *small_handle = small_store;

void out_small(enum Small *out) { *out = SM_C; }
void smalls_fill(enum Small xs[], int n) { for (int i = 0; i < n; i++) xs[i] = (enum Small)(i % 3); }
int  smalls_sum(const enum Small *xs, int n) { int s = 0; for (int i = 0; i < n; i++) s += (int)xs[i]; return s; }
int  small_swap(enum Small *a, enum Small *b) { enum Small t = *a; *a = *b; *b = t; return 1; }
enum Small  *smalls_static(void) { return small_store; }
enum Small **smalls_handle(void) { return &small_handle; }
Holder holder_make(void) { Holder h = { small_store, 2 }; return h; }
void out_perm(enum Perm *out) { *out = (enum Perm)(P_R | P_X); }

unsigned __int128 u128_echo(unsigned __int128 v) { return v; }
__uint128_t u128_add(__uint128_t a, __uint128_t b) { return a + b; }
unsigned long long u128_hi(u128 v) { return (unsigned long long)(v >> 64); }
unsigned long long u128_lo(u128 v) { return (unsigned long long)v; }
u128 u128_make(unsigned long long hi, unsigned long long lo) { return ((u128)hi << 64) | lo; }
void u128_out(u128 *out) { *out = ((u128)0xfedcba9876543210ULL << 64) | 0x0123456789abcdefULL; }
u128 u128_after4(long a, long b, long c, long d, u128 v) { return v + (u128)(a + b + c + d); }
u128 u128_after_double(double x, long a, long b, long c, long d, u128 v) { return v + (u128)(a + b + c + d) + (u128)(long)x; }
u128 u128_after5(long a, long b, long c, long d, long e, u128 v) { return v + (u128)(a + b + c + d + e); }
u128 u128_after7(long a, long b, long c, long d, long e, long f, long g, u128 v) { return v + (u128)(a + b + c + d + e + f + g); }
i128 i128_after5(long a, long b, long c, long d, long e, i128 v) { return v - (i128)(a + b + c + d + e); }
u128 u128_then_long(long a, long b, long c, long d, long e, u128 v, long tail) { return v + (u128)(a + b + c + d + e) + (u128)tail * 1000; }

/* No arithmetic on the half float: that would call a runtime helper. */
_Float16 f16_echo(_Float16 x) { return x; }
int f16_bits(_Float16 x) { union { _Float16 f; unsigned short u; } v = { x }; return v.u; }
int plain_add(int a, int b) { return a + b; }
