#pragma once

class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const { return 0.0; }
    virtual double perimeter() const { return 0.0; }
};

class Rectangle : public Shape {
    double width, height;
public:
    Rectangle(double w, double h);
    double area() const override;
    double perimeter() const override;
};

class Circle : public Shape {
    double radius;
public:
    Circle(double r);
    double area() const override;
    double perimeter() const override;
};

// A constructor taking value-type structs BY VALUE. Constructors are named C4 in
// DWARF and C1/C2 in the symbol table; until 2026-09-26 that join always missed,
// the signature was guessed from the demangled string (`Point2D` → `Any`), and the
// thunk stored garbage. See verify.jl, "constructor by-value struct arguments".
// In a namespace on purpose: GeneratorCpp skips a constructor whose bare name
// equals its QUALIFIED class, which only happens at global scope (open problem).
namespace geom {
struct Point2D { double x, y; };

class Segment2D {
public:
    Segment2D(Point2D from, Point2D to);
    double length2() const;
    Point2D a, b;
};
}

extern "C" {
    Shape* create_rectangle(double w, double h);
    Shape* create_circle(double r);
    double get_area(const Shape* s);
    double get_perimeter(const Shape* s);
    void delete_shape(Shape* s);
}
