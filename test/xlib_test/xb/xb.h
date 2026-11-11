#ifndef XB_H
#define XB_H
#include "xa.h"

// Derives from xa's classes: its layout flattens their fields, and a call
// through xa's virtual `get` on one of these must reach the override.
class Derived : public Shape {
public:
    Derived(int id, int extra);
    int get() const override;
private:
    int extra_;
};

class SubLeaf : public Leaf {
public:
    explicit SubLeaf(int v);
    int value() const override;
private:
    int w_;
};

// Embeds xa's record and points at xa's class.
struct Holder { Plain p; Shape *s; };

// None of these is noexcept: a C++ function goes to Tier 2, so the by-value
// records cross through thunks that need the owners' layouts. `spawn_`, not a
// factory prefix: the classes keep their layout structs.
int use(Shape *s, Plain p, Color c);
Plain twice(Plain p);
Shape *spawn_derived(int id, int extra);
Leaf *spawn_subleaf(int v);
Holder make_holder(int x);
Pt xb_pt(Pt p);
int mode_twice(Mode m);

#endif
