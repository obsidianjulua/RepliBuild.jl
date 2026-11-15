#ifndef MEMBERLESS_H
#define MEMBERLESS_H

/* Library-free fixture for records with no data member of their own
   (2026-11-14, from Bullet3's GENERATOR-memberless-class-size report). The
   DWARF parser gave such a record no `struct_definitions` entry, so the
   wrapper had no `STRUCT_SIZES` entry to size its constructor's storage by,
   its Julia type was an empty `struct X end`, and every class deriving from
   it lost the bases above it. Now every defined library record has an entry;
   the generator flattens the non-virtual bases into fields, and the thunk
   classifies the record from its bases' leaves. */

namespace mb {

struct Quad { float m[4]; };
struct Derived : Quad {};                            /* bytes all from the base: 16, SSE,SSE */
struct P { bool operator()(int a, int b) const; };   /* stateless functor: 1 byte nobody reads */
struct Poly { virtual ~Poly(); int id; };
struct PolyKid : Poly { PolyKid(int v); };           /* Bullet's shape: methods, no fields */
struct VB : Poly, virtual Quad {};                   /* no member of its own, a virtual base */

float dsum(Derived d);
Derived dmake(float x);
int pcall(P p, int a, int b);
int vb_first(VB v);
int vb_make(float x);
int kid_id(PolyKid *k);

}

#endif
