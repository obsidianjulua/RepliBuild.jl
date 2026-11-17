#include "simd_union.h"

Vec4f v_add(Vec4f a, Vec4f b) { Vec4f r; r.m = _mm_add_ps(a.m, b.m); return r; }
float v_lane(Vec4f a, int i) { return a.f[i]; }
Vec4f v_make(float x, float y, float z, float w) { Vec4f r; r.m = _mm_set_ps(w, z, y, x); return r; }
float v_mixed(int k, Vec4f a, double d, Vec4f b, float s) { return k + a.f[0] + (float)d + b.f[3] + s; }
Vec4u u_or(Vec4u a, Vec4u b) { Vec4u r; r.m = _mm_or_si128(a.m, b.m); return r; }
unsigned u_lane(Vec4u a, int i) { return a.u[i]; }
Quatf q_conj(Quatf q) { Quatf r; r.v.m = _mm_xor_ps(q.v.m, _mm_set_ps(0.0f, -0.0f, -0.0f, -0.0f)); return r; }
float q_w(Quatf q) { return q.v.f[3]; }
float lone_x(Lone l) { return _mm_cvtss_f32(l.m); }

Vec4f Body::getPos() const { return pos; }
void Body::setPos(Vec4f p) { pos = p; }
Vec4f Body::scaled(Vec4f by, float k) const { Vec4f r; r.m = _mm_mul_ps(_mm_mul_ps(pos.m, by.m), _mm_set1_ps(k)); return r; }
