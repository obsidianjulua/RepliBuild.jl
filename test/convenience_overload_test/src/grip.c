#include <stdio.h>
#include <stdlib.h>
#include "grip.h"

Grip* grip_new(int value, double weight) {
    Grip* g = (Grip*)malloc(sizeof(Grip));
    g->value = value;
    g->weight = weight;
    return g;
}

void grip_free(Grip* g) {
    free(g);
}

int grip_value(const Grip* g) {
    return g->value;
}

double sum_xs(const double* xs, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += xs[i];
    return s;
}

char* describe_values(const double* values, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += values[i];
    char* buf = (char*)malloc(64);
    snprintf(buf, 64, "n=%d sum=%.1f", n, s);
    return buf;
}
