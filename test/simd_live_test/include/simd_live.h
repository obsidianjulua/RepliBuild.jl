/* Live SIMD vectors for test_simd_vectors_live.jl: records holding a vector of
   each size class and bare 8- and 16-byte vectors, by value both ways. Built
   once as C (simd_live.c: GeneratorC, Tier 3 ccalls) and once as C++
   (simd_live.cpp: take_/make_ have C++ linkage and are not noexcept, so they
   are Tier 2 thunks). image_R / image_mask_R / size_R / poison_regs take
   pointers or nothing and have C linkage: Tier 3, trusted.

   x86-64 SysV classifies a vector by its size, never lane by lane:
     r_v1f   { <1 x float> }        4 bytes   INTEGER
     r_v2i   { <2 x int> }          8 bytes   SSE
     r_v2f   { <2 x float> }        8 bytes   SSE (as its lanes)
     r_v1d   { <1 x double> }       8 bytes   MEMORY
     r_v2f_i { <2 x float>; int }  16 bytes   SSE, INTEGER (as its lanes)
     r_v4f   { <4 x float> }       16 bytes   one SSE register (SSE, SSEUP)
   Win64 passes and returns each record by size. */
#ifndef SIMD_LIVE_H
#define SIMD_LIVE_H

#ifdef __cplusplus
#define SL_C extern "C"
#else
#define SL_C
#endif

typedef float sl_v1f __attribute__((vector_size(4)));
typedef int sl_v2i __attribute__((vector_size(8)));
typedef float sl_v2f __attribute__((vector_size(8)));
typedef double sl_v1d __attribute__((vector_size(8)));
typedef float sl_v4f __attribute__((vector_size(16)));

typedef struct r_v1f { sl_v1f v; } r_v1f;
typedef struct r_v2i { sl_v2i v; } r_v2i;
typedef struct r_v2f { sl_v2f v; } r_v2f;
typedef struct r_v1d { sl_v1d v; } r_v1d;
typedef struct r_v2f_i { sl_v2f v; int i; } r_v2f_i;
typedef struct r_v4f { sl_v4f v; } r_v4f;

SL_C void poison_regs(void);

#define SL_DECL(R) \
    double take_##R(R v); \
    R make_##R(void); \
    SL_C void image_##R(void *out); \
    SL_C void image_mask_##R(void *out); \
    SL_C unsigned size_##R(void);

SL_DECL(r_v1f) SL_DECL(r_v2i) SL_DECL(r_v2f) SL_DECL(r_v1d) SL_DECL(r_v2f_i) SL_DECL(r_v4f)

/* Bare vectors. */
double take_v2f(sl_v2f v);
sl_v2f make_v2f(float a);
double take_v4f(sl_v4f v);
sl_v4f make_v4f(float a);

/* C linkage, for the C++ build: a function the C++ generator's ccall gate
   (`is_ccall_safe`) judges. It used to pass these to a plain ccall; a vector
   crossing now takes the thunk, which states a record's class and refuses a
   bare vector. In the C build they are ordinary C functions, like the above. */
SL_C double take_c_v4f(sl_v4f v);
SL_C double take_c_r_v4f(r_v4f x);
SL_C r_v4f make_c_r_v4f(void);
SL_C double take_c_r_v2f(r_v2f x);
SL_C r_v2f make_c_r_v2f(void);

#endif
