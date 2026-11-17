#ifndef SIMD_UNION_H
#define SIMD_UNION_H
#include <immintrin.h>

/* Library-free fixture for a 16-byte SIMD vector in a union (2026-11-16, from
   Jolt's Vec3/Vec4/Quat/UVec4: `union { __m128 mValue; float mF32[4]; }`).
   The classifier refused any record holding a 16-byte vector as "one SSE
   register no call statement states". That is true of a vector ALONE. In a
   union the eightbyte merge decides: SSEUP against SSE is SSE, so the float
   union is (SSE, SSE), two registers, which clang passes as
   { double, double }; against INTEGER it is INTEGER, so the integer union
   is two general registers. About 1,000 Jolt functions were ABI traps. */

struct alignas(16) Vec4f {                 /* Jolt's Vec3 / Vec4 */
    union { __m128 m; float f[4]; };
};
struct alignas(16) Vec4u {                 /* Jolt's UVec4 */
    union { __m128i m; unsigned u[4]; };
};
struct Quatf { Vec4f v; };                 /* Jolt's Quat: the union one level down */
struct Lone { __m128 m; };                 /* a vector alone: one SSE register */

Vec4f v_add(Vec4f a, Vec4f b);
float v_lane(Vec4f a, int i);
Vec4f v_make(float x, float y, float z, float w);
/* Mixed with other arguments, so a wrong register class shows. */
float v_mixed(int k, Vec4f a, double d, Vec4f b, float s);
Vec4u u_or(Vec4u a, Vec4u b);
unsigned u_lane(Vec4u a, int i);
Quatf q_conj(Quatf q);
float q_w(Quatf q);
float lone_x(Lone l);

struct Body {
    Vec4f pos;
    Vec4f getPos() const;
    void setPos(Vec4f p);
    Vec4f scaled(Vec4f by, float k) const;
};

#endif
