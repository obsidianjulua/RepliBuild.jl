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

/* ── Struct layout edge cases (from basics_test) ────────────────────────── */

/* Struct with padding: 1 byte (a) + 3 bytes padding + 4 bytes (b) = size 8 */
typedef struct {
    char a;
    int  b;
} PaddedStruct;

/* Packed struct: 1 byte (a) + 4 bytes (b) = size 5 */
#pragma pack(push, 1)
typedef struct {
    char a;
    int  b;
} PackedStruct;
#pragma pack(pop)

/* Union: max(sizeof(int), sizeof(float)) = 4 bytes */
typedef union {
    int   i;
    float f;
} NumberUnion;

/* Global variables */
extern int         global_int;
extern const char *global_string;

PaddedStruct make_padded(char a, int b);
PackedStruct make_packed(char a, int b);
int          get_union_int(NumberUnion u);
float        get_union_float(NumberUnion u);

/* Variadic function */
int sum_ints(int count, ...);

/* ── JIT edge cases (from jit_edge_test) ────────────────────────────────── */

/* Simplest passthrough */
int identity(int x);

/* Pointer output param */
void write_sum(const int *a, const int *b, int *out);

/* Small struct return (2 fields, 8 bytes) */
typedef struct {
    int first;
    int second;
} PairResult;

PairResult make_pair(int a, int b);

/* Packed struct return */
#pragma pack(push, 1)
typedef struct {
    char tag;
    int  value;
    char flag;
} PackedTriplet;
#pragma pack(pop)

PackedTriplet pack_three(char tag, int value, char flag);

#endif /* MATHKIT_H */
