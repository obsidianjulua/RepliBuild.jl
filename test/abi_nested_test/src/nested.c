#include "nested.h"

XForm make_xform(float a, float b, float c, float d) {
    XForm t = { { a, b }, { c, d } };
    return t;
}

Vec2 xform_p(XForm t) { return t.p; }

Mass make_mass(float m, float cx, float cy, float i) {
    Mass md = { m, { cx, cy }, i };
    return md;
}

float mass_total(Mass md) { return md.m + md.c.x + md.c.y + md.i; }

Disc make_disc(float cx, float cy, float r) {
    Disc d = { { cx, cy }, r };
    return d;
}

float disc_area(Disc d) { return 3.14159265f * d.r * d.r + d.center.x * 0.0f; }

Poly make_poly(void) {
    Poly p = { { { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f } }, 3 };
    return p;
}

float poly_sum(Poly p) {
    float s = 0.0f;
    for (int k = 0; k < p.n; k++) s += p.v[k].x + p.v[k].y;
    return s;
}

NestInt make_nestint(void) {
    NestInt v = { { 100, 200 }, 300, 400 };
    return v;
}

int nestint_sum(NestInt v) { return v.ip.a + v.ip.b + v.c + v.d; }

PackedFV make_packedfv(void) {
    PackedFV s = { 7, 1.5f, 2.5f };
    return s;
}

float packedfv_sum(PackedFV s) { return (float)s.tag + s.a + s.b; }

float scale_vec(Vec2 v, float s) { return (v.x + v.y) * s; }
