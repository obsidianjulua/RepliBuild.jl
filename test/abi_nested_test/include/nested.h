#ifndef NESTED_H
#define NESTED_H

// Library-free trace fixture for nested-struct-member ABI resolution.
// Every struct here crosses the FFI boundary BY VALUE somewhere below;
// distinctive constants make register-class garbage unmistakable.

typedef struct { float x, y; } Vec2;                  // flat float, 8B  (control: resolved)
typedef struct { Vec2 p; Vec2 q; } XForm;             // nested float, 16B (SSE,SSE — the bug class)
typedef struct { float m; Vec2 c; float i; } Mass;    // nested mid-struct, 16B
typedef struct { Vec2 center; float r; } Disc;        // nested float, 12B (SSE,SSE)
typedef struct { Vec2 v[3]; int n; } Poly;            // array-of-struct member, 28B (MEMORY — control)
typedef struct { short a, b; } Pair;                  // flat int, 4B (control)
typedef struct { Pair ip; short c, d; } NestInt;      // nested int, 8B (INTEGER — blob-coincidence-safe)

// Packed: float at offset 1 (unaligned member). Layout cannot be reproduced
// with Julia natural alignment -> must stay opaque; by-value crossings of it
// must fail loudly, never silently corrupt.
typedef struct __attribute__((packed)) { char tag; float a; float b; } PackedFV; // 9B

XForm make_xform(float a, float b, float c, float d);
Vec2  xform_p(XForm t);            // 16B nested-float ARG
Mass  make_mass(float m, float cx, float cy, float i);
float mass_total(Mass md);         // 16B nested-float ARG
Disc  make_disc(float cx, float cy, float r);
float disc_area(Disc d);           // 12B nested-float ARG
Poly  make_poly(void);
float poly_sum(Poly p);            // 28B MEMORY ARG (control)
NestInt make_nestint(void);
int   nestint_sum(NestInt v);      // 8B nested-int ARG (control)
PackedFV make_packedfv(void);
float packedfv_sum(PackedFV s);    // 9B packed-float ARG (safety-net target)
float scale_vec(Vec2 v, float s);  // Cfloat param ergonomics probe

#endif
