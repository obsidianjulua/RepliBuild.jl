#include "sysv_live.h"
#include <string.h>

/* Fill every SSE argument register with a NaN pattern, so a value the thunk
   left in the wrong register cannot be read back from a register that still
   held it by chance. The integer registers are overwritten by any call. */
extern "C" void poison_regs() {
    unsigned long long junk = 0x7ff8dead7ff8deadULL;
    __asm__ volatile(
        "movq %0, %%xmm0\n\tmovq %0, %%xmm1\n\tmovq %0, %%xmm2\n\tmovq %0, %%xmm3\n\t"
        "movq %0, %%xmm4\n\tmovq %0, %%xmm5\n\tmovq %0, %%xmm6\n\tmovq %0, %%xmm7\n\t"
        : : "r"(junk)
        : "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7");
}

/* Each record has one canonical value; take_R weights every member differently,
   so a member read from the wrong register or slot changes the answer.
   image_R writes the canonical bytes straight from a memset(0) local (no
   by-value return, so padding is 0); image_mask_R writes 0xFF over every
   member's bytes (a bitfield: all its bits), so a comparison can ignore
   padding, which C leaves unspecified when a record moves in registers. */
#define FILL(m) memset(&(m), 0xFF, sizeof(m))
#define IMAGE(R, SETS, ONES) \
    R make_##R() { R r; memset(&r, 0, sizeof r); SETS; return r; } \
    extern "C" void image_##R(void *out) { R r; memset(&r, 0, sizeof r); SETS; memcpy(out, &r, sizeof r); } \
    extern "C" void image_mask_##R(void *out) { R r; memset(&r, 0, sizeof r); ONES; memcpy(out, &r, sizeof r); } \
    extern "C" unsigned size_##R() { return (unsigned)sizeof(R); }

IMAGE(c_f_d, (r.x = 1.5f, r.y = 2.25), (FILL(r.x), FILL(r.y)))
double take_c_f_d(c_f_d v) { return v.x + 10.0 * v.y; }                       /* 24.0 */
IMAGE(c_d_f, (r.d = 2.25, r.f = 1.5f), (FILL(r.d), FILL(r.f)))
double take_c_d_f(c_d_f v) { return v.d + 10.0 * v.f; }                       /* 17.25 */
IMAGE(c_p_f, (r.p = (void *)(uintptr_t)0x1000, r.f = 1.5f), (FILL(r.p), FILL(r.f)))
double take_c_p_f(c_p_f v) { return (double)(uintptr_t)v.p + 10.0 * v.f; }    /* 4111.0 */
IMAGE(c_ubf_f, (r.f = 1.5f), (FILL(r.f)))
double take_c_ubf_f(c_ubf_f v) { return 10.0 * v.f; }                         /* 15.0 */
IMAGE(c_f_pad, (r.x = 1.5f, r.y = 2.25), (FILL(r.x), FILL(r.y)))
double take_c_f_pad(c_f_pad v) { return v.x + 10.0 * v.y; }                   /* 24.0 */
IMAGE(c_wrap_df, (r.in.d = 2.25, r.in.f = 1.5f), (FILL(r.in.d), FILL(r.in.f)))
double take_c_wrap_df(c_wrap_df v) { return v.in.d + 10.0 * v.in.f; }         /* 17.25 */
IMAGE(c_al16_f, (r.f = 1.5f), (FILL(r.f)))
double take_c_al16_f(c_al16_f v) { return 10.0 * v.f; }                       /* 15.0 */
IMAGE(c_cf, (__real__ r.z = 1.5f, __imag__ r.z = 2.5f), (FILL(r.z)))
double take_c_cf(c_cf v) { return __real__ v.z + 10.0 * __imag__ v.z; }       /* 26.5 */
IMAGE(c_cd, (__real__ r.z = 1.5, __imag__ r.z = 2.5), (FILL(r.z)))
double take_c_cd(c_cd v) { return __real__ v.z + 10.0 * __imag__ v.z; }       /* 26.5 */
IMAGE(c_ubf_d, (r.d = 2.5), (FILL(r.d)))
double take_c_ubf_d(c_ubf_d v) { return 10.0 * v.d; }                         /* 25.0 */
IMAGE(c_ld, (r.x = 1.5L), (memset(&r.x, 0xFF, 10)))
double take_c_ld(c_ld v) { return (double)v.x; }                              /* 1.5 */
IMAGE(c_u_ld_i, (r.x = 1.5L), (memset(&r.x, 0xFF, 10)))
double take_c_u_ld_i(c_u_ld_i v) { return (double)v.x; }                      /* 1.5 */
IMAGE(c_s_f, (r.s = 3, r.f = 1.5f), (FILL(r.s), FILL(r.f)))
double take_c_s_f(c_s_f v) { return v.s + 10.0 * v.f; }                       /* 18.0 */
IMAGE(c_pk_sf, (r.c = 7, r.in.s = 3, r.in.f = 1.5f), (FILL(r.c), FILL(r.in.s), FILL(r.in.f)))
double take_c_pk_sf(c_pk_sf v) { return v.c + 10.0 * v.in.s + 100.0 * v.in.f; } /* 187.0 */
IMAGE(c_pk_ci, (r.c = 7, r.i = 100), (FILL(r.c), FILL(r.i)))
double take_c_pk_ci(c_pk_ci v) { return v.c + 10.0 * v.i; }                   /* 1007.0 */
IMAGE(c_u_pk, (r.p.c = 7, r.p.i = 100), (FILL(r.p.c), FILL(r.p.i)))
double take_c_u_pk(c_u_pk v) { return v.p.c + 10.0 * v.p.i; }                 /* 1007.0 */
IMAGE(c_pk_alnest, (r.c = 7, r.s.x = 3), (FILL(r.c), FILL(r.s.x)))
double take_c_pk_alnest(c_pk_alnest v) { return v.c + 10.0 * v.s.x; }         /* 37.0 */
IMAGE(c_pp1_bf, (r.c = 7, r.b = 1000), (FILL(r.c), r.b = -1))
double take_c_pp1_bf(c_pp1_bf v) { return v.c + 10.0 * v.b; }                 /* 10007.0 */
IMAGE(c_u_fi, (r.f = 1.5f), (FILL(r.f)))
double take_c_u_fi(c_u_fi v) { return 10.0 * v.f; }                           /* 15.0 */
IMAGE(c_bf2_f, (r.a = 3, r.b = 5, r.f = 1.5f), (r.a = -1, r.b = -1, FILL(r.f)))
double take_c_bf2_f(c_bf2_f v) { return v.a + 10.0 * v.b + 100.0 * v.f; }     /* 203.0 */
IMAGE(c_i128_l, (r.x = 5, r.y = 9), (FILL(r.x), FILL(r.y)))
double take_c_i128_l(c_i128_l v) { return (double)(long long)v.x + 10.0 * v.y; } /* 95.0 */
IMAGE(c_al16_big, (r.a = 1, r.b = 2, r.c = 3), (FILL(r.a), FILL(r.b), FILL(r.c)))
double take_c_al16_big(c_al16_big v) { return v.a + 10.0 * v.b + 100.0 * v.c; } /* 321.0 */
IMAGE(c_f_i, (r.x = 1.5f, r.i = 7), (FILL(r.x), FILL(r.i)))
double take_c_f_i(c_f_i v) { return v.x + 10.0 * v.i; }                       /* 71.5 */
IMAGE(c_d2, (r.a = 1.5, r.b = 2.5), (FILL(r.a), FILL(r.b)))
double take_c_d2(c_d2 v) { return v.a + 10.0 * v.b; }                         /* 26.5 */
IMAGE(c_c_d, (r.c = 7, r.d = 2.5), (FILL(r.c), FILL(r.d)))
double take_c_c_d(c_c_d v) { return v.c + 10.0 * v.d; }                       /* 32.0 */
