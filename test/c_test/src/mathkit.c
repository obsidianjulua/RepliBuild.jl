#include "mathkit.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <stdarg.h>

/* --- Scalar arithmetic --- */

int32_t add_i32(int32_t a, int32_t b) {
    return a + b;
}

double lerp(double a, double b, double t) {
    return a + (b - a) * t;
}

int32_t apply_op(MathOp op, int32_t a, int32_t b) {
    switch (op) {
        case OP_ADD: return a + b;
        case OP_SUB: return a - b;
        case OP_MUL: return a * b;
        case OP_DIV: return (b != 0) ? a / b : 0;
        default:     return 0;
    }
}

/* --- Point operations --- */

Point2D point_add(Point2D a, Point2D b) {
    Point2D r = { a.x + b.x, a.y + b.y };
    return r;
}

double point_dist(Point2D a, Point2D b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

Point2D point_scale(Point2D p, double s) {
    Point2D r = { p.x * s, p.y * s };
    return r;
}

/* --- AABB helpers --- */

AABB aabb_from_points(const Point2D *pts, size_t n) {
    AABB box;
    if (n == 0) {
        box.min.x = box.min.y = 0.0;
        box.max.x = box.max.y = 0.0;
        return box;
    }
    box.min = pts[0];
    box.max = pts[0];
    for (size_t i = 1; i < n; i++) {
        if (pts[i].x < box.min.x) box.min.x = pts[i].x;
        if (pts[i].y < box.min.y) box.min.y = pts[i].y;
        if (pts[i].x > box.max.x) box.max.x = pts[i].x;
        if (pts[i].y > box.max.y) box.max.y = pts[i].y;
    }
    return box;
}

int aabb_contains(AABB box, Point2D p) {
    return (p.x >= box.min.x && p.x <= box.max.x &&
            p.y >= box.min.y && p.y <= box.max.y) ? 1 : 0;
}

double aabb_area(AABB box) {
    double w = box.max.x - box.min.x;
    double h = box.max.y - box.min.y;
    return (w > 0 && h > 0) ? w * h : 0.0;
}

/* --- Array reduction --- */

Stats array_stats(const double *data, size_t n) {
    Stats s = { 0.0, DBL_MAX, -DBL_MAX, n };
    if (n == 0) { s.min_val = 0.0; s.max_val = 0.0; return s; }
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
        if (data[i] < s.min_val) s.min_val = data[i];
        if (data[i] > s.max_val) s.max_val = data[i];
    }
    s.mean = sum / (double)n;
    return s;
}

/* --- String utility --- */

size_t greet(const char *name, char *buf, size_t buf_len) {
    int written = snprintf(buf, buf_len, "Hello, %s!", name);
    return (written < 0) ? 0 : (size_t)written;
}

/* --- Struct layout edge cases (from basics_test) --- */

int global_int = 42;
const char *global_string = "Hello from Global";

PaddedStruct make_padded(char a, int b) {
    PaddedStruct s;
    s.a = a;
    s.b = b;
    return s;
}

PackedStruct make_packed(char a, int b) {
    PackedStruct s;
    s.a = a;
    s.b = b;
    return s;
}

int get_union_int(NumberUnion u) {
    return u.i;
}

float get_union_float(NumberUnion u) {
    return u.f;
}

int sum_ints(int count, ...) {
    va_list args;
    va_start(args, count);
    int sum = 0;
    for (int i = 0; i < count; i++) {
        sum += va_arg(args, int);
    }
    va_end(args);
    return sum;
}

/* --- JIT edge cases (from jit_edge_test) --- */

int identity(int x) {
    return x;
}

void write_sum(const int *a, const int *b, int *out) {
    *out = *a + *b;
}

PairResult make_pair(int a, int b) {
    PairResult r;
    r.first = a;
    r.second = b;
    return r;
}

PackedTriplet pack_three(char tag, int value, char flag) {
    PackedTriplet t;
    t.tag = tag;
    t.value = value;
    t.flag = flag;
    return t;
}

/* --- Bitfield helpers --- */

SingleByteBits make_single_bits(unsigned a, unsigned b, unsigned c) {
    SingleByteBits s;
    s.a = a;
    s.b = b;
    s.c = c;
    return s;
}

void read_single_bits(SingleByteBits s, unsigned *a, unsigned *b, unsigned *c) {
    *a = s.a;
    *b = s.b;
    *c = s.c;
}

MultiByteBits make_multi_bits(unsigned x, unsigned y, unsigned z) {
    MultiByteBits s;
    s.x = x;
    s.y = y;
    s.z = z;
    return s;
}

void read_multi_bits(MultiByteBits s, unsigned *x, unsigned *y, unsigned *z) {
    *x = s.x;
    *y = s.y;
    *z = s.z;
}

WideBits make_wide_bits(unsigned tag, unsigned data, unsigned flag) {
    WideBits s;
    s.tag = tag;
    s.data = data;
    s.flag = flag;
    return s;
}

void read_wide_bits(WideBits s, unsigned *tag, unsigned *data, unsigned *flag) {
    *tag = s.tag;
    *data = s.data;
    *flag = s.flag;
}
