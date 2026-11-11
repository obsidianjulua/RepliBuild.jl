#include "xa.h"

Shape::Shape(int id) : id_(id) {}
Shape::~Shape() {}
int Shape::get() const { return id_ * 10; }
int Shape::id() const { return id_; }

Leaf::Leaf(int v) : v_(v) {}
Leaf::~Leaf() {}
int Leaf::value() const { return v_; }

Shape *create_shape(int id) { return new Shape(id); }
void destroy_shape(Shape *s) { delete s; }
Leaf *make_leaf(int v) { return new Leaf(v); }
Plain make_plain(int x, double y) { Plain p = { x, y }; return p; }
int color_value(Color c) { return static_cast<int>(c) * 100; }
Pt xa_shift(Pt p, int d) { p.a += d; return p; }
int shape_get(const Shape *s) { return s->get(); }
