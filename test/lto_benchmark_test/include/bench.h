#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Scalar primitives — baseline call overhead
int scalar_add(int a, int b);
double scalar_mul(double a, double b);

// Hot loop candidate — single step function, called millions of times
double add_to(double acc, double val);

// Full loop in C++ — for comparison against per-call overhead
double accumulate_array(const double* data, int n);

// Struct return — 16-byte POD
typedef struct {
    double x;
    double y;
} Point2D;

Point2D make_point(double x, double y);
double point_distance_sq(Point2D a, Point2D b);

#ifdef __cplusplus
}
#endif

// Packed struct — requires ABI-aware dispatch (cannot be done with naive ccall)
#pragma pack(push, 1)
typedef struct {
    char  tag;
    int   value;
    char  flag;
} PackedRecord;
#pragma pack(pop)

#ifdef __cplusplus
extern "C" {
#endif
PackedRecord pack_record(char tag, int value, char flag);
#ifdef __cplusplus
}
#endif
