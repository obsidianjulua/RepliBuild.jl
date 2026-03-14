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

extern "C" {
    Shape* create_rectangle(double w, double h);
    Shape* create_circle(double r);
    double get_area(const Shape* s);
    double get_perimeter(const Shape* s);
    void delete_shape(Shape* s);
}
