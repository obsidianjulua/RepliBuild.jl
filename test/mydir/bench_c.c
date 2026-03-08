/* bench_c.c — pure-C baseline timer
 * Prints median ns/call for iadd, fmadd, hsum to stdout.
 * Build: gcc -O2 -o bench_c bench_c.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define K       200000
#define ROUNDS  9
#define VLEN    1024

/* the functions under test (same code as main.c) */
static inline int    iadd(int a, int b)                { return a + b; }
static inline double fmadd(double a, double b, double c) { return a*b + c; }
static double hsum(const double* v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i];
    return s;
}

static int cmp_double(const void* a, const void* b) {
    double x = *(double*)a, y = *(double*)b;
    return (x > y) - (x < y);
}

static double bench(void (*fn)(double*), double* sink) {
    double times[ROUNDS];
    for (int r = 0; r < ROUNDS; r++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < K; i++) fn(sink);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        long ns = (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec);
        times[r] = (double)ns / K;
    }
    qsort(times, ROUNDS, sizeof(double), cmp_double);
    return times[ROUNDS/2];
}

static double sink_val = 0.0;
static double vec[VLEN];

static void do_iadd(double* s)  { *s += iadd(3, 7); }
static void do_fmadd(double* s) { *s += fmadd(1.5, 2.5, 0.1); }
static void do_hsum(double* s)  { *s += hsum(vec, VLEN); }

int main(void) {
    for (int i = 0; i < VLEN; i++) vec[i] = (double)i / VLEN;
    printf("iadd  %.2f\n",  bench(do_iadd,  &sink_val));
    printf("fmadd %.2f\n",  bench(do_fmadd, &sink_val));
    printf("hsum  %.2f\n",  bench(do_hsum,  &sink_val));
    /* prevent DCE */
    if (sink_val < -1e300) printf("sink=%f\n", sink_val);
    return 0;
}
