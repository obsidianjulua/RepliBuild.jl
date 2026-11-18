#include "alias_xapi.h"

/* See alias_xapi.h. An alias names its aliasee by symbol, so the C++ ones
   are spelled mangled. */

extern "C" int ax_real(int x) { return x + 1; }
extern "C" int ax_name(int x) __attribute__((alias("ax_real")));

__attribute__((used)) static int local_mul(int a, int b) { return a * b; }
extern "C" int ax_via_static(int a, int b) __attribute__((alias("_ZL9local_mulii")));

double ax_mangled_real(double x, int n) { return x * n; }
double ax_mangled(double x, int n) __attribute__((alias("_Z15ax_mangled_realdi")));

extern "C" __attribute__((nodebug)) int ax_undescribed(int x) { return x + 2; }
__attribute__((nodebug)) int ax_undescribed_cxx(int x) { return x + 3; }
