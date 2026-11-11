#include "xb.h"

Derived::Derived(int id, int extra) : Shape(id), extra_(extra) {}
int Derived::get() const { return id_ * 10 + extra_; }

SubLeaf::SubLeaf(int v) : Leaf(v), w_(v * 2) {}
int SubLeaf::value() const { return v_ + w_ + 1000; }

int use(Shape *s, Plain p, Color c) { return s->get() + p.x + static_cast<int>(p.y) + static_cast<int>(c); }
Plain twice(Plain p) { Plain q = { p.x * 2, p.y * 2 }; return q; }
Shape *spawn_derived(int id, int extra) { return new Derived(id, extra); }
Leaf *spawn_subleaf(int v) { return new SubLeaf(v); }
Holder make_holder(int x) { Holder h = { { x, x * 0.5 }, nullptr }; return h; }
Pt xb_pt(Pt p) { Pt q = { p.a * 3, p.b * 3 }; return q; }
int mode_twice(Mode m) { return static_cast<int>(m) * 2; }
