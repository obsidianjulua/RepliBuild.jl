#pragma once
struct Shape {
    double w;
    Shape(double x) : w(x) {}
    double area() const { return w * w; }
};
