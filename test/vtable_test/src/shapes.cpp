#include "shapes.h"
#include <cmath>

Rectangle::Rectangle(double w, double h) : width(w), height(h) {}
double Rectangle::area() const { return width * height; }
double Rectangle::perimeter() const { return 2 * (width + height); }

Circle::Circle(double r) : radius(r) {}
double Circle::area() const { return M_PI * radius * radius; }
double Circle::perimeter() const { return 2 * M_PI * radius; }

extern "C" {
    Shape* create_rectangle(double w, double h) { return new Rectangle(w, h); }
    Shape* create_circle(double r) { return new Circle(r); }
    double get_area(const Shape* s) { return s->area(); }
    double get_perimeter(const Shape* s) { return s->perimeter(); }
    void delete_shape(Shape* s) { delete s; }
}
