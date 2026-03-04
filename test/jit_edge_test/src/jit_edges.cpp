#include "jit_edges.h"

int scalar_add(int a, int b) {
    return a + b;
}

double scalar_mul(double a, double b) {
    return a * b;
}

int identity(int x) {
    return x;
}

void write_sum(const int* a, const int* b, int* out) {
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
