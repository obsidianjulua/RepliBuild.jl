#include "memberless.h"

namespace mb {

bool P::operator()(int a, int b) const { return a < b; }
Poly::~Poly() {}
PolyKid::PolyKid(int v) { id = v * 2; }

float dsum(Derived d) { return d.m[0] + d.m[1] + d.m[2] + d.m[3]; }
Derived dmake(float x) { Derived d; for (int i = 0; i < 4; ++i) d.m[i] = x + i; return d; }
int pcall(P p, int a, int b) { return p(a, b) ? a : b; }
int vb_first(VB v) { return (int)v.m[0]; }
int vb_make(float x) { VB v; v.m[0] = x; v.id = 7; return vb_first(v); }
int kid_id(PolyKid *k) { return k->id; }

}
