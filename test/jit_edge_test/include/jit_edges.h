#ifndef JIT_EDGES_H
#define JIT_EDGES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Scalar edge cases ---

// Simplest possible: 2 args, value return
int scalar_add(int a, int b);

// Float ABI path
double scalar_mul(double a, double b);

// Single arg passthrough (minimum call overhead)
int identity(int x);

// Pointer args, void return
void write_sum(const int* a, const int* b, int* out);

// --- Struct return edge cases ---

// Small struct return (2 fields, 8 bytes, naturally aligned)
typedef struct PairResult {
    int first;
    int second;
} PairResult;

PairResult make_pair(int a, int b);

// Packed struct return (char+int+char = 6 bytes packed, 12 bytes aligned)
// This tests the packed struct marshalling through JIT thunks
#pragma pack(push, 1)
typedef struct PackedTriplet {
    char tag;
    int value;
    char flag;
} PackedTriplet;
#pragma pack(pop)

PackedTriplet pack_three(char tag, int value, char flag);

#ifdef __cplusplus
}
#endif

#endif // JIT_EDGES_H
