/* The definitions behind simd_live.h, included by simd_live.c and
   simd_live.cpp so the C and C++ builds are one library. */
#include "simd_live.h"

/* Fill every SSE argument register with a NaN pattern, so a value the call
   left in the wrong register cannot be read back from a register that still
   held it by chance. The integer registers are overwritten by any call. */
SL_C void poison_regs(void) {
    unsigned long long junk = 0x7ff8dead7ff8deadULL;
    __asm__ volatile(
        "movq %0, %%xmm0\n\tmovq %0, %%xmm1\n\tmovq %0, %%xmm2\n\tmovq %0, %%xmm3\n\t"
        "movq %0, %%xmm4\n\tmovq %0, %%xmm5\n\tmovq %0, %%xmm6\n\tmovq %0, %%xmm7\n\t"
        : : "r"(junk)
        : "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7");
}

/* Each record has one canonical value, and take_R weights every lane
   differently, so a lane read from the wrong register changes the answer.
   image_R writes the canonical bytes from a zeroed local; image_mask_R
   writes 0xFF over every member, so a comparison can ignore padding. */
#define SL_FILL(m) __builtin_memset(&(m), 0xFF, sizeof(m))
#define SL_IMAGE(R, SETS, ONES) \
    R make_##R(void) { R r; __builtin_memset(&r, 0, sizeof r); SETS; return r; } \
    SL_C void image_##R(void *out) { R r; __builtin_memset(&r, 0, sizeof r); SETS; __builtin_memcpy(out, &r, sizeof r); } \
    SL_C void image_mask_##R(void *out) { R r; __builtin_memset(&r, 0, sizeof r); ONES; __builtin_memcpy(out, &r, sizeof r); } \
    SL_C unsigned size_##R(void) { return (unsigned)sizeof(R); }

SL_IMAGE(r_v1f, (r.v[0] = 1.5f), (SL_FILL(r.v)))
double take_r_v1f(r_v1f x) { return x.v[0]; }                                          /* 1.5 */
SL_IMAGE(r_v2i, (r.v[0] = 3, r.v[1] = -4), (SL_FILL(r.v)))
double take_r_v2i(r_v2i x) { return x.v[0] + 10.0 * x.v[1]; }                          /* -37.0 */
SL_IMAGE(r_v2f, (r.v[0] = 1.25f, r.v[1] = 2.5f), (SL_FILL(r.v)))
double take_r_v2f(r_v2f x) { return x.v[0] + 10.0 * x.v[1]; }                          /* 26.25 */
SL_IMAGE(r_v1d, (r.v[0] = 6.75), (SL_FILL(r.v)))
double take_r_v1d(r_v1d x) { return x.v[0]; }                                          /* 6.75 */
SL_IMAGE(r_v2f_i, (r.v[0] = 0.5f, r.v[1] = 1.5f, r.i = 7), (SL_FILL(r.v), SL_FILL(r.i)))
double take_r_v2f_i(r_v2f_i x) { return x.v[0] + 10.0 * x.v[1] + 100.0 * x.i; }        /* 715.5 */
SL_IMAGE(r_v4f, (r.v[0] = 0.5f, r.v[1] = 1.0f, r.v[2] = 1.5f, r.v[3] = 2.0f), (SL_FILL(r.v)))
double take_r_v4f(r_v4f x) { return x.v[0] + 10.0 * x.v[1] + 100.0 * x.v[2] + 1000.0 * x.v[3]; }  /* 2160.5 */

double take_v2f(sl_v2f v) { return v[0] + 10.0 * v[1]; }                               /* (1.25, 2.5): 26.25 */
sl_v2f make_v2f(float a) { sl_v2f r = {a, 2 * a}; return r; }
double take_v4f(sl_v4f v) { return v[0] + 10.0 * v[1] + 100.0 * v[2] + 1000.0 * v[3]; } /* (0.5, 1, 1.5, 2): 2160.5 */
sl_v4f make_v4f(float a) { sl_v4f r = {a, 2 * a, 3 * a, 4 * a}; return r; }

/* C linkage (the C++ generator's ccall gate): the same values as above. */
SL_C double take_c_v4f(sl_v4f v) { return v[0] + 10.0 * v[1] + 100.0 * v[2] + 1000.0 * v[3]; }
SL_C double take_c_r_v4f(r_v4f x) { return take_r_v4f(x); }
SL_C r_v4f make_c_r_v4f(void) { return make_r_v4f(); }
SL_C double take_c_r_v2f(r_v2f x) { return take_r_v2f(x); }
SL_C r_v2f make_c_r_v2f(void) { return make_r_v2f(); }
