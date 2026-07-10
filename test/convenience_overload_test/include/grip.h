#ifndef GRIP_H
#define GRIP_H

/* Minimal ownership-shaped API: a heap-allocated struct with a free()-taking
 * destructor (the double-free footgun shape), an input-array function (the
 * Vector convenience survivor), and a malloc'd-string return (Cstring policy). */

typedef struct Grip {
    int value;
    double weight;
} Grip;

Grip* grip_new(int value, double weight);
void grip_free(Grip* g);
int grip_value(const Grip* g);

double sum_xs(const double* xs, int n);
char* describe_values(const double* values, int n);

#endif
