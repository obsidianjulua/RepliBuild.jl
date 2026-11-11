#ifndef XA_H
#define XA_H
#include "xc.h"

// A polymorphic class with a key function (`get` is defined in xa.cpp), so its
// vtable and full DWARF description live in this library alone: a dependent
// library's DWARF only declares it.
class Shape {
public:
    explicit Shape(int id);
    virtual ~Shape();
    virtual int get() const;
    int id() const;
protected:
    int id_;
};

// A virtual destructor and a factory, and no deleter function: on its own,
// `Leaf` would get a factory struct whose finalizer calls the deleting
// destructor directly. xb derives from it, so across the graph it must not.
class Leaf {
public:
    explicit Leaf(int v);
    virtual ~Leaf();
    virtual int value() const;
protected:
    int v_;
};

// Header-only POD and enum: every library that uses them has their full DWARF.
struct Plain { int x; double y; };
enum Color { RED = 1, GREEN = 2, BLUE = 3 };

Shape *create_shape(int id);
void destroy_shape(Shape *s);
Leaf *make_leaf(int v);
Plain make_plain(int x, double y);
int color_value(Color c);
Pt xa_shift(Pt p, int d);
int shape_get(const Shape *s);

#endif
