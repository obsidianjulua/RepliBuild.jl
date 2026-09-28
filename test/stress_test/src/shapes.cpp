// M_PI is not standard C or C++ — it is an X/Open extension. glibc exposes it
// from <cmath> regardless, which is why this fixture built on Linux with no
// define; the UCRT gates it behind _USE_MATH_DEFINES. The macro must precede
// EVERY include that can reach <math.h>, hence above the project header too.
// Inert on glibc.
#define _USE_MATH_DEFINES
#include "shapes.h"
#include <cmath>

Rectangle::Rectangle(double w, double h) : width(w), height(h) {}
double Rectangle::area() const { return width * height; }
double Rectangle::perimeter() const { return 2 * (width + height); }

Circle::Circle(double r) : radius(r) {}
double Circle::area() const { return M_PI * radius * radius; }
double Circle::perimeter() const { return 2 * M_PI * radius; }

Span::Span(int l, int h) : lo(l), hi(h) {}
int Span::width() const { return hi - lo; }

extern "C" {
    Shape* create_rectangle(double w, double h) { return new Rectangle(w, h); }
    Shape* create_circle(double r) { return new Circle(r); }
    double get_area(const Shape* s) { return s->area(); }
    double get_perimeter(const Shape* s) { return s->perimeter(); }
    void delete_shape(Shape* s) { delete s; }
}

geom::Segment2D::Segment2D(Point2D from, Point2D to) : a(from), b(to) {}
double geom::Segment2D::length2() const {
    double dx = b.x - a.x, dy = b.y - a.y;
    return dx * dx + dy * dy;
}

int geom::unit_scaled(Unit, int x, Unit, int y) { return x * 10 + y; }
geom::Unit geom::unit_make(int) { return Unit{}; }
geom::Cell<double> geom::cell_doubled(Cell<double> c) { return Cell<double>{c.v * 2}; }
std::size_t geom::name_length(std::string_view s) { return s.size(); }
geom::Ticket::~Ticket() {}
geom::Ticket geom::ticket_make(int n) { return Ticket{n}; }
geom::Packed geom::packed_make(int a) { Packed p; p.c = (char)a; p.i = a + 1; return p; }
double geom::packed_sum(Packed p) { return p.c + p.i; }
geom::Pack2 geom::pack2_make(int a) { Pack2 p; p.c = (char)a; p.i = a + 1; p.d = a + 0.5; return p; }
double geom::pack2_sum(Pack2 p) { return p.c + p.i + p.d; }
static geom::Packed packed_rows[3] = {{1, 10}, {2, 20}, {3, 30}};
const geom::Packed *geom::packed_table() { return packed_rows; }
geom::Variant geom::variant_make(int a) { Variant v; v.u.i = a; v.d = a + 0.25; return v; }
double geom::variant_sum(Variant v) { return v.u.i + v.d; }
geom::Arr3 geom::arr3_make(float a) { return Arr3{{a, a + 1, a + 2}}; }
double geom::arr3_sum(Arr3 s) { return s.v[0] + s.v[1] + s.v[2]; }
