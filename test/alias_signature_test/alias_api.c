#include "alias_api.h"

/* See alias_api.h. */

int al_real(int x) { return x + 1; }
int al_name(int x) __attribute__((alias("al_real")));

double al_scaled_real(double x, int n) { return x * n; }
double al_scaled(double x, int n) __attribute__((weak, alias("al_scaled_real")));

AlPair al_pair_real(double a, double b) { AlPair p = {a + b, a - b}; return p; }
AlPair al_pair(double a, double b) __attribute__((alias("al_pair_real")));

long al_wide_real(long a, long b, long c, long d, long e, long f, long g) {
    return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g;
}
long al_wide(long a, long b, long c, long d, long e, long f, long g) __attribute__((alias("al_wide_real")));

int al_old_style() { return 1978; }

/* A definition the compiler is told to write no debug information for, like
   one from a file compiled without -g: exported, and nothing describes it. */
__attribute__((nodebug)) int al_undescribed(int x) { return x + 2; }
