#ifndef MATHKIT_H
#define MATHKIT_H

#include <stdint.h>
#include <stddef.h>

/* Simple enum for operations */
typedef enum {
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2,
    OP_DIV = 3
} MathOp;

/* A 2D point */
typedef struct {
    double x;
    double y;
} Point2D;

/* Axis-aligned bounding box */
typedef struct {
    Point2D min;
    Point2D max;
} AABB;

/* Stats result from an array reduction */
typedef struct {
    double mean;
    double min_val;
    double max_val;
    size_t count;
} Stats;

/* Scalar arithmetic */
int32_t add_i32(int32_t a, int32_t b);
double  lerp(double a, double b, double t);
int32_t apply_op(MathOp op, int32_t a, int32_t b);

/* Point operations */
Point2D point_add(Point2D a, Point2D b);
double  point_dist(Point2D a, Point2D b);
Point2D point_scale(Point2D p, double s);

/* AABB helpers */
AABB    aabb_from_points(const Point2D *pts, size_t n);
int     aabb_contains(AABB box, Point2D p);
double  aabb_area(AABB box);

/* Array reduction */
Stats   array_stats(const double *data, size_t n);

/* String utility — returns length of filled buffer */
size_t  greet(const char *name, char *buf, size_t buf_len);

#endif /* MATHKIT_H */
