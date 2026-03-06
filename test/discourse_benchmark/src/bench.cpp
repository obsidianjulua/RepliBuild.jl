#include "bench.h"
#include <math.h>

extern "C" {

int scalar_add(int a, int b) {
    return a + b;
}

double scalar_mul(double a, double b) {
    return a * b;
}

double add_to(double acc, double val) {
    return acc + val;
}

double accumulate_array(const double* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

Point2D make_point(double x, double y) {
    Point2D p;
    p.x = x;
    p.y = y;
    return p;
}

double point_distance_sq(Point2D a, Point2D b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

PackedRecord pack_record(char tag, int value, char flag) {
    PackedRecord r;
    r.tag   = tag;
    r.value = value;
    r.flag  = flag;
    return r;
}

} // extern "C"
