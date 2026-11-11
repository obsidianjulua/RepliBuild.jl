#include "xc.h"

Pt xc_make(int a, double b) { Pt p = { a, b }; return p; }
double xc_sum(Pt p) { return p.a + p.b; }
int xc_mode(enum Mode m) { return (int)m * 10; }
